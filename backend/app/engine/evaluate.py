"""
Model evaluation the way a credit-risk validation team reads it.

Every split gets discrimination (AUC, Gini, KS, PR-AUC), calibration
(Brier, log-loss, expected calibration error, reliability curve),
threshold metrics at the F1-optimal cut-off, lift/gains by decile and the
ROC curve. Stability compares splits with score PSI and AUC drops.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix, f1_score,
    log_loss, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve,
)

ROC_MAX_POINTS = 200
DECILES = 10


def _thin(values: np.ndarray, n: int = ROC_MAX_POINTS) -> np.ndarray:
    if len(values) <= n:
        return values
    keep = np.unique(np.linspace(0, len(values) - 1, n).astype(int))
    return values[keep]


def ks_statistic(y: np.ndarray, p: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Probability cut-off that maximises F1 (the 0.5 default is rarely right
    for imbalanced credit outcomes)."""
    precision, recall, thresholds = precision_recall_curve(y, p)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-9, None)
    if len(f1) == 0:
        return 0.5
    return float(thresholds[int(np.argmax(f1))])


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(ece)


def reliability_curve(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> Dict[str, List[float]]:
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    predicted, observed, counts = [], [], []
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            predicted.append(float(p[mask].mean()))
            observed.append(float(y[mask].mean()))
            counts.append(int(mask.sum()))
    return {"predicted": predicted, "observed": observed, "count": counts}


def lift_table(y: np.ndarray, p: np.ndarray, n: int = DECILES) -> List[Dict[str, float]]:
    """Deciles from the riskiest (highest p) down: capture, lift, bad rate."""
    order = np.argsort(-p, kind="stable")
    y_sorted = y[order]
    total_bad = max(y.sum(), 1)
    base_rate = y.mean() if len(y) else 0.0
    edges = np.linspace(0, len(y), n + 1).astype(int)
    rows, cum_bad = [], 0
    for i in range(n):
        seg = y_sorted[edges[i]:edges[i + 1]]
        if len(seg) == 0:
            continue
        cum_bad += seg.sum()
        rows.append({
            "decile": i + 1,
            "population_pct": round(100 * len(seg) / len(y), 2),
            "bad_rate": round(float(seg.mean()), 4),
            "lift": round(float(seg.mean() / base_rate), 3) if base_rate else 0.0,
            "cumulative_capture": round(float(cum_bad / total_bad), 4),
            "cumulative_population": round(float(edges[i + 1] / len(y)), 4),
        })
    return rows


def psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index between two score distributions."""
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return 0.0
    e_hist = np.histogram(expected, bins=edges)[0] / max(len(expected), 1)
    a_hist = np.histogram(actual, bins=edges)[0] / max(len(actual), 1)
    e_hist = np.clip(e_hist, 1e-4, None)
    a_hist = np.clip(a_hist, 1e-4, None)
    return float(np.sum((a_hist - e_hist) * np.log(a_hist / e_hist)))


def evaluate_split(y: np.ndarray, p: np.ndarray, threshold: float) -> Dict[str, Any]:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    pred = (p >= threshold).astype(int)
    fpr, tpr, _ = roc_curve(y, p)
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else 0.5
    return {
        "metrics": {
            "roc_auc": auc,
            "gini": 2 * auc - 1,
            "ks": ks_statistic(y, p) if len(np.unique(y)) > 1 else 0.0,
            "pr_auc": float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float(y.mean()),
            "brier": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "ece": expected_calibration_error(y, p),
            "accuracy": float(accuracy_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "threshold": float(threshold),
            "positive_rate": float(y.mean()),
            "n": int(len(y)),
        },
        "confusion": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "roc_curve": {"fpr": _thin(fpr).tolist(), "tpr": _thin(tpr).tolist()},
        "reliability": reliability_curve(y, p),
        "lift": lift_table(y, p),
    }


def stability(evals: Dict[str, Dict[str, Any]], probs: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Gaps between cohorts. Test/OOT figures are reported, never used for selection."""
    val_auc = evals["validation"]["metrics"]["roc_auc"]
    return {
        "overfit_gap": evals["train"]["metrics"]["roc_auc"] - val_auc,
        "test_drop": val_auc - evals["test"]["metrics"]["roc_auc"],
        "oot_drop": val_auc - evals["oot"]["metrics"]["roc_auc"],
        "calibration_drop": val_auc - evals["calibration"]["metrics"]["roc_auc"],
        "score_psi_calibration": psi(probs["train"], probs["calibration"]),
        "score_psi_oot": psi(probs["train"], probs["oot"]),
    }


# Selection uses ONLY train / validation / calibration cohorts. Test and OOT
# stay untouched so their metrics are honest estimates of future performance.
SELECTION_WEIGHTS = {"discrimination": 0.45, "calibration": 0.20, "consistency": 0.15, "overfit": 0.10, "simplicity": 0.10}


def composite_score(model: Dict[str, Any], all_models: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Champion selection score in [0, 1] from selection-safe cohorts:
    validation AUC, validation Brier, validation→calibration consistency,
    train→validation overfit gap, and simplicity. Components are scaled
    against the other candidates so no single metric dominates by unit."""
    aucs = np.array([m["evaluations"]["validation"]["metrics"]["roc_auc"] for m in all_models])
    briers = np.array([m["evaluations"]["validation"]["metrics"]["brier"] for m in all_models])
    fit_secs = np.array([m["fit_seconds"] for m in all_models])

    def rel(value, values, higher_better=True):
        lo, hi = float(values.min()), float(values.max())
        if hi - lo < 1e-9:
            return 1.0
        x = (value - lo) / (hi - lo)
        return x if higher_better else 1 - x

    s = model["stability"]
    discrimination = rel(model["evaluations"]["validation"]["metrics"]["roc_auc"], aucs)
    calibration = rel(model["evaluations"]["validation"]["metrics"]["brier"], briers, higher_better=False)
    consistency = float(np.clip(1 - abs(s["calibration_drop"]) / 0.10, 0, 1)) * 0.7 \
        + float(np.clip(1 - s["score_psi_calibration"] / 0.25, 0, 1)) * 0.3
    overfit = float(np.clip(1 - max(s["overfit_gap"], 0) / 0.10, 0, 1))
    simplicity = 0.5 * rel(np.log1p(model["fit_seconds"]), np.log1p(fit_secs), higher_better=False) \
        + 0.5 * (1.0 if model["family"] in ("logistic", "scorecard") else 0.6)
    components = {
        "discrimination": round(float(discrimination), 4),
        "calibration": round(float(calibration), 4),
        "consistency": round(float(consistency), 4),
        "overfit": round(float(overfit), 4),
        "simplicity": round(float(simplicity), 4),
    }
    total = sum(components[k] * w for k, w in SELECTION_WEIGHTS.items())
    return {"score": round(float(total), 4), "components": components, "weights": SELECTION_WEIGHTS,
            "selection_cohorts": ["train", "validation", "calibration"]}
