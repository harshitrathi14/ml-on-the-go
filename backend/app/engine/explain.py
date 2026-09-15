"""
SHAP explanations for the champion and reason codes for scored records.

Tree ensembles use the exact TreeExplainer, linear models the
LinearExplainer, everything else (MLP, transformer) a sampled
permutation explainer. All outputs are bounded in size so they can be
returned to the browser.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

GLOBAL_ROWS = 2000
BEESWARM_ROWS = 800
TOP_FEATURES = 15
REASON_CODES = 4


def _tree_model(model) -> bool:
    name = type(model).__name__
    return name in ("RandomForestClassifier", "ExtraTreesClassifier", "HistGradientBoostingClassifier",
                    "XGBClassifier", "LGBMClassifier", "CatBoostClassifier", "GradientBoostingClassifier")


class Explainer:
    """Wraps a fitted estimator working on the preprocessed matrix."""

    def __init__(self, model, background: np.ndarray, feature_names: List[str], seed: int = 42):
        import shap

        self.model = model
        self.feature_names = feature_names
        self.kind = "tree" if _tree_model(model) else "linear" if type(model).__name__ == "LogisticRegression" else "sampled"
        rng = np.random.default_rng(seed)
        bg = background[rng.choice(len(background), size=min(200, len(background)), replace=False)]
        if self.kind == "tree":
            self._explainer = shap.TreeExplainer(model)
        elif self.kind == "linear":
            self._explainer = shap.LinearExplainer(model, bg)
        else:
            self._explainer = shap.PermutationExplainer(lambda X: model.predict_proba(X)[:, 1], bg, seed=seed)

    def values(self, X: np.ndarray) -> np.ndarray:
        """SHAP values towards the positive (bad) class, shape (n, features)."""
        if self.kind == "sampled":
            X = X[:min(len(X), 300)]
            out = self._explainer(X, max_evals=2 * X.shape[1] + 1, silent=True)
            return np.asarray(out.values)
        raw = self._explainer.shap_values(X, check_additivity=False) if self.kind == "tree" else self._explainer.shap_values(X)
        if isinstance(raw, list):                       # sklearn forests: [class0, class1]
            raw = raw[1]
        raw = np.asarray(raw)
        if raw.ndim == 3:                               # (n, features, classes)
            raw = raw[:, :, -1]
        return raw

    def global_summary(self, X: np.ndarray, seed: int = 42, display: Optional[np.ndarray] = None) -> Dict[str, Any]:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), size=min(GLOBAL_ROWS, len(X)), replace=False)
        shap_values = self.values(X[idx])
        # Values shown on the axes: the raw matrix when given (same rows/columns).
        sample = (display if display is not None else X)[idx][:len(shap_values)]
        mean_abs = np.abs(shap_values).mean(axis=0)
        order = np.argsort(-mean_abs)[:TOP_FEATURES]
        importance = [{"feature": self.feature_names[i], "mean_abs_shap": round(float(mean_abs[i]), 5)} for i in order]

        # Beeswarm: per top feature, (shap, feature value percentile) pairs.
        rows = rng.choice(len(shap_values), size=min(BEESWARM_ROWS, len(shap_values)), replace=False)
        beeswarm = []
        for i in order:
            col = sample[:, i]
            ranks = pd.Series(col).rank(pct=True).to_numpy()
            beeswarm.append({
                "feature": self.feature_names[i],
                "shap": np.round(shap_values[rows, i], 5).tolist(),
                "value_pct": np.round(ranks[rows], 3).tolist(),
                "value": np.round(col[rows], 4).tolist(),
            })
        dependence = []
        for i in order[:3]:
            dependence.append({"feature": self.feature_names[i],
                               "x": np.round(sample[rows, i], 4).tolist(),
                               "shap": np.round(shap_values[rows, i], 5).tolist()})
        return {"kind": self.kind, "importance": importance, "beeswarm": beeswarm, "dependence": dependence,
                "rows_explained": int(len(shap_values))}

    def reason_codes(self, X: np.ndarray, raw: Optional[pd.DataFrame] = None) -> List[List[Dict[str, Any]]]:
        """Top risk-increasing contributions per row (adverse-action style)."""
        shap_values = self.values(X)
        out = []
        for r in range(len(shap_values)):
            order = np.argsort(-shap_values[r])[:REASON_CODES]
            reasons = []
            for i in order:
                if shap_values[r, i] <= 0:
                    continue
                name = self.feature_names[i]
                base = name.split("__")[0].replace("missingindicator_", "")
                value = None
                if raw is not None and base in raw.columns:
                    v = raw.iloc[r][base]
                    value = None if pd.isna(v) else (float(v) if isinstance(v, (int, float, np.number)) else str(v))
                reasons.append({"feature": name, "contribution": round(float(shap_values[r, i]), 5), "value": value})
            out.append(reasons)
        return out
