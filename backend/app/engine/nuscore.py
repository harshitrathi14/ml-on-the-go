"""
Nu Score: calibrated probability -> 0–1000 score (higher = safer) -> bands.

* Calibration: Platt scaling (logistic regression on the model's log-odds)
  fitted on the test split, so the score reflects observed bad rates rather
  than a model's raw output. Platt is smooth and monotone, unlike isotonic
  regression which saturates to 0/1 on the extremes of a small test set.
* Scaling: log-odds with a fixed points-to-double-odds, anchored so that
  even odds (p = 0.5) sit at 500; clipped to [0, 1000].
* Bands: A+ … E cut-offs are learnt from the data (test + OOT scores):
  quantile seeds, then adjacent bands are merged until bad rate rises
  monotonically from A+ to E and no band is smaller than MIN_SHARE.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression

PDO = 60.0          # points to double the odds
ANCHOR = 500.0      # score at 1:1 odds
BAND_NAMES = ["A+", "A", "B", "C", "D", "E"]
SEED_SHARES = [0.10, 0.20, 0.25, 0.20, 0.15, 0.10]   # safest first
MIN_SHARE = 0.05


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class Calibrator:
    def __init__(self) -> None:
        self.lr = LogisticRegression(C=1e6, max_iter=1000)

    def fit(self, p: np.ndarray, y: np.ndarray) -> "Calibrator":
        self.lr.fit(_logit(p).reshape(-1, 1), np.asarray(y, dtype=int))
        return self

    def __call__(self, p: np.ndarray) -> np.ndarray:
        return np.clip(self.lr.predict_proba(_logit(p).reshape(-1, 1))[:, 1], 1e-4, 1 - 1e-4)


def to_score(p_bad: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p_bad, dtype=float), 1e-4, 1 - 1e-4)
    log_odds_good = np.log((1 - p) / p)
    return np.clip(np.round(ANCHOR + PDO / np.log(2) * log_odds_good), 0, 1000).astype(int)


def fit_bands(score: np.ndarray, y: np.ndarray) -> List[Dict[str, Any]]:
    """Learn cut-offs on scores with known outcomes."""
    score = np.asarray(score)
    y = np.asarray(y, dtype=int)
    order = np.argsort(-score, kind="stable")           # safest first
    s_sorted, y_sorted = score[order], y[order]
    n = len(score)
    edges = np.cumsum([0] + SEED_SHARES)
    idx = [int(round(e * n)) for e in edges]
    segments = [(idx[i], idx[i + 1]) for i in range(len(SEED_SHARES)) if idx[i + 1] > idx[i]]

    def rate(seg):
        a, b = seg
        return y_sorted[a:b].mean() if b > a else 0.0

    # Merge until bad rate is monotone (rising towards E) and shares are sane.
    changed = True
    while changed and len(segments) > 1:
        changed = False
        for i in range(len(segments) - 1):
            share = (segments[i][1] - segments[i][0]) / n
            if rate(segments[i]) > rate(segments[i + 1]) or share < MIN_SHARE:
                segments[i:i + 2] = [(segments[i][0], segments[i + 1][1])]
                changed = True
                break
        last_share = (segments[-1][1] - segments[-1][0]) / n
        if not changed and last_share < MIN_SHARE and len(segments) > 1:
            segments[-2:] = [(segments[-2][0], segments[-1][1])]
            changed = True

    names = BAND_NAMES[:len(segments)]
    if len(segments) < len(BAND_NAMES):
        # Keep the extremes meaningful when fewer bands survive.
        names = [BAND_NAMES[0]] + BAND_NAMES[len(BAND_NAMES) - len(segments) + 1:]
    bands = []
    for name, (a, b) in zip(names, segments):
        bands.append({"band": name, "min_score": int(s_sorted[b - 1]), "max_score": int(s_sorted[a])})
    # Make the ranges contiguous and open-ended at both ends.
    for i in range(len(bands)):
        bands[i]["max_score"] = 1000 if i == 0 else bands[i - 1]["min_score"] - 1
    bands[-1]["min_score"] = 0
    return bands


def assign_band(score: np.ndarray, bands: List[Dict[str, Any]]) -> np.ndarray:
    score = np.asarray(score)
    out = np.full(len(score), bands[-1]["band"], dtype=object)
    for band in bands:
        mask = (score >= band["min_score"]) & (score <= band["max_score"])
        out[mask] = band["band"]
    return out


def band_table(score: np.ndarray, y: np.ndarray, bands: List[Dict[str, Any]],
               reference: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
    """Per-band population share, bad rate, lift, capture; PSI vs a reference score set."""
    score = np.asarray(score)
    y = np.asarray(y, dtype=int)
    labels = assign_band(score, bands)
    base_rate = y.mean() if len(y) else 0.0
    total_bad = max(y.sum(), 1)
    rows, cum_bad = [], 0
    ref_labels = assign_band(reference, bands) if reference is not None else None
    for band in reversed(bands):        # riskiest first for cumulative capture
        mask = labels == band["band"]
        n = int(mask.sum())
        bad = int(y[mask].sum())
        cum_bad += bad
        row = {
            "band": band["band"], "min_score": band["min_score"], "max_score": band["max_score"],
            "count": n, "share": round(n / max(len(y), 1), 4),
            "bad_rate": round(bad / n, 4) if n else None,
            "lift": round((bad / n) / base_rate, 3) if n and base_rate else None,
            "cumulative_bad_capture": round(cum_bad / total_bad, 4),
        }
        if ref_labels is not None:
            row["reference_share"] = round(float((ref_labels == band["band"]).mean()), 4)
        rows.append(row)
    rows.reverse()
    if ref_labels is not None:
        for row in rows:
            e, a = max(row["reference_share"], 1e-4), max(row["share"], 1e-4)
            row["psi_contribution"] = round(float((a - e) * np.log(a / e)), 5)
    return rows


def cutoff_curve(score: np.ndarray, y: np.ndarray, step: int = 20) -> List[Dict[str, float]]:
    """For each approve-above cut-off: approval rate, bad rate among approved,
    share of bads rejected."""
    score = np.asarray(score)
    y = np.asarray(y, dtype=int)
    total_bad = max(y.sum(), 1)
    rows = []
    for cut in range(0, 1001, step):
        approved = score >= cut
        n = int(approved.sum())
        rows.append({
            "cutoff": cut,
            "approval_rate": round(n / max(len(y), 1), 4),
            "approved_bad_rate": round(float(y[approved].mean()), 4) if n else 0.0,
            "bads_rejected": round(float(y[~approved].sum() / total_bad), 4),
        })
    return rows


def histogram(score: np.ndarray, y: np.ndarray, bin_width: int = 25) -> Dict[str, List[float]]:
    edges = np.arange(0, 1000 + bin_width, bin_width)
    good = np.histogram(score[y == 0], bins=edges)[0]
    bad = np.histogram(score[y == 1], bins=edges)[0]
    return {"edges": edges[:-1].tolist(), "good": good.tolist(), "bad": bad.tolist()}
