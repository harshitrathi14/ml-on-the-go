"""
Population Stability Index (PSI) drift monitoring.

Reference distributions are fitted on the training cohort only and stored
as aggregate bins (never rows): numeric quantile cuts with open tails, or
the top categories with an "other" bucket, plus a separate "missing" bin.
Comparison uses symmetric 0.5 pseudo-counts so empty bins are defined and
shares sum to one. PSI < 0.1 stable, 0.1–0.25 moderate, > 0.25 significant.

Ported from the codex implementation's stability.py and adapted to the
engine's cohorts and model bundle.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

MAX_BINS = 10
THRESHOLDS = {"stable": 0.1, "moderate": 0.25}


def fit_distribution(values, max_bins: int = MAX_BINS) -> Dict[str, Any]:
    series = pd.Series(values)
    if pd.api.types.is_numeric_dtype(series):
        finite = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
        finite = finite[np.isfinite(finite)]
        cuts = np.unique(np.quantile(finite, np.linspace(0, 1, max_bins + 1))) if len(finite) else np.array([])
        profile: Dict[str, Any] = {"type": "numeric", "cuts": cuts.tolist()}
    else:
        top = series.dropna().astype(str).value_counts().head(max_bins).index.tolist()
        profile = {"type": "categorical", "categories": top}
    profile["counts"] = _counts(series, profile).tolist()
    profile["rows"] = int(len(series))
    return profile


def _counts(series: pd.Series, profile: Dict[str, Any]) -> np.ndarray:
    missing = series.isna().to_numpy()
    if profile["type"] == "numeric":
        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        missing = missing | ~np.isfinite(values)
        bins = np.searchsorted(profile["cuts"], values[~missing], side="right")
        counts = np.bincount(bins, minlength=len(profile["cuts"]) + 1)
    else:
        categories = profile["categories"]
        lookup = {value: i for i, value in enumerate(categories)}
        bins = [lookup.get(value, len(categories)) for value in series[~missing].astype(str)]
        counts = np.bincount(bins, minlength=len(categories) + 1)
    return np.append(counts, missing.sum())


def compare(profile: Dict[str, Any], values) -> Dict[str, Any]:
    current = _counts(pd.Series(values), profile)
    expected = np.asarray(profile["counts"], dtype=float)
    p = (expected + 0.5) / (expected.sum() + 0.5 * len(expected))
    q = (current + 0.5) / (current.sum() + 0.5 * len(current))
    psi = float(np.sum((q - p) * np.log(q / p)))
    return {
        "psi": round(psi, 5),
        "status": "stable" if psi < THRESHOLDS["stable"] else "moderate" if psi < THRESHOLDS["moderate"] else "significant",
        "reference_count": int(expected.sum()),
        "current_count": int(current.sum()),
        "missing_pct": round(100 * int(current[-1]) / max(int(current.sum()), 1), 2),
    }


def fit_profiles(frame: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    return {str(col): fit_distribution(frame[col]) for col in frame.columns}


def feature_drift(profiles: Dict[str, Dict[str, Any]], frame: pd.DataFrame) -> List[Dict[str, Any]]:
    rows = []
    for col, profile in profiles.items():
        if col not in frame.columns:
            continue
        rows.append({"feature": col, **compare(profile, frame[col])})
    return sorted(rows, key=lambda r: (-r["psi"], r["feature"]))


def summarise(feature_rows: List[Dict[str, Any]], score: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "score": score,
        "features": feature_rows[:40],
        "n_significant": sum(r["status"] == "significant" for r in feature_rows),
        "n_moderate": sum(r["status"] == "moderate" for r in feature_rows),
        "method": ("PSI against training-defined bins (numeric quantiles with open tails, or top categories "
                   "plus 'other'), separate missing bin, symmetric 0.5 smoothing. Descriptive drift, not a "
                   "model-quality test; small cohorts are noisy."),
    }
