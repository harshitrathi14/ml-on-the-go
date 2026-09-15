"""
Deep diagnostics measured on the actual cohorts.

* bootstrap 95% interval for the test AUC (row bootstrap on the fixed
  predictor: conditional uncertainty, not a correction for selection)
* predictor rows in test that also appear in train (leakage / duplicate smell)
* constant or all-missing features, highly correlated numeric pairs
* automatic plain-English warnings a validation team would raise

Ported from the codex implementation's deep_diagnostics.py and warning engine.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

BOOTSTRAP_REPLICATES = 200
CORRELATION_THRESHOLD = 0.8


def bootstrap_auc(y: np.ndarray, p: np.ndarray, seed: int, replicates: int = BOOTSTRAP_REPLICATES) -> Optional[List[float]]:
    rng = np.random.default_rng(seed)
    aucs = []
    n = len(y)
    if n == 0:
        return None
    for _ in range(replicates):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) == 2:
            aucs.append(roc_auc_score(y[idx], p[idx]))
    if not aucs:
        return None
    lo, hi = np.quantile(aucs, [0.025, 0.975])
    return [round(float(lo), 4), round(float(hi), 4)]


def data_checks(train: pd.DataFrame, test: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Any]:
    numeric = train[feature_cols].select_dtypes(include=np.number)
    pairs = []
    if 1 < len(numeric.columns) <= 600:
        matrix = numeric.corr().abs()
        cols = list(matrix.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                v = matrix.iloc[i, j]
                if np.isfinite(v) and v >= CORRELATION_THRESHOLD:
                    pairs.append({"left": cols[i], "right": cols[j], "abs_correlation": round(float(v), 4)})
        pairs.sort(key=lambda r: -r["abs_correlation"])
    constants = [c for c in feature_cols if train[c].nunique(dropna=True) <= 1]
    train_hashes = set(pd.util.hash_pandas_object(train[feature_cols].astype(str), index=False).tolist())
    duplicates = int(pd.util.hash_pandas_object(test[feature_cols].astype(str), index=False).isin(train_hashes).sum())
    return {
        "constant_or_all_missing_features": constants,
        "correlated_pairs": pairs[:25],
        "test_rows_matching_training_predictors": duplicates,
        "test_rows": int(len(test)),
    }


def warnings_for(champion: Dict[str, Any], cohort_sizes: Dict[str, int], cohort_positives: Dict[str, int],
                 split_info: Dict[str, Any], checks: Dict[str, Any], calibration_effect: Dict[str, float]) -> List[str]:
    out = []
    ev = champion["evaluations"]
    if split_info.get("strategy") == "time":
        out.append("Out-of-time cohort holds the latest observation dates; repeated borrowers still need grouped validation.")
    else:
        out.append("No usable time column: the 'OOT' cohort is a random hold-out, so it measures stability, not future performance.")
    out.append("Test and OOT rows never touched fitting, selection or calibration; validation and calibration rows did.")
    small = [n for n in ("validation", "calibration", "test", "oot") if cohort_positives.get(n, 0) < 30]
    if small:
        out.append(f"Fewer than 30 positive outcomes in {', '.join(small)}: metrics and band bad rates there are uncertain.")
    if ev["test"]["metrics"]["roc_auc"] < 0.6:
        out.append("Test discrimination is weak (ROC-AUC below 0.60); these scores need further development before use.")
    gap = ev["train"]["metrics"]["roc_auc"] - ev["validation"]["metrics"]["roc_auc"]
    if gap > 0.10:
        out.append(f"Train-to-validation AUC gap is {gap:.2f} (> 0.10): investigate overfitting or unstable features.")
    drop = ev["validation"]["metrics"]["roc_auc"] - ev["oot"]["metrics"]["roc_auc"]
    if drop > 0.05:
        out.append(f"AUC drops {drop:.2f} from validation to OOT: the pattern may be shifting over time.")
    if calibration_effect.get("brier_after", 0) > calibration_effect.get("brier_before", 0):
        out.append("Calibration slightly worsened test Brier loss; reported as-is (test was not used to tune it).")
    if checks["test_rows_matching_training_predictors"] > 0:
        out.append(f"{checks['test_rows_matching_training_predictors']} test rows have identical predictor values to "
                   "training rows: possible duplicates or repeated borrowers inflating the test score.")
    if checks["constant_or_all_missing_features"]:
        out.append(f"{len(checks['constant_or_all_missing_features'])} feature(s) are constant or all missing and carry no signal.")
    if checks["correlated_pairs"]:
        top = checks["correlated_pairs"][0]
        out.append(f"{len(checks['correlated_pairs'])} highly correlated feature pair(s), e.g. {top['left']} ~ {top['right']} "
                   f"(|r| = {top['abs_correlation']:.2f}); importances are shared between them.")
    return out
