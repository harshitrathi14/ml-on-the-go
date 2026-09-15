"""
Model bundle: everything needed to score a new application after the
training data has been deleted.

    <job_dir>/model/bundle.joblib   preprocessor, champion estimator, calibrator,
                                    Nu Score bands, SHAP background sample
    <job_dir>/model/meta.json       input schema, feature names, summary metrics
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from . import nuscore
from .explain import Explainer
from .preprocess import InputSchema, prepare_frame

BUNDLE_VERSION = 1


class ModelBundle:
    def __init__(self, preprocessor, model, calibrator, schema: InputSchema, feature_names: List[str],
                 bands: List[Dict[str, Any]], background: np.ndarray, meta: Dict[str, Any],
                 blend: Optional[Dict[str, Any]] = None, alt_preprocessor=None):
        self.preprocessor = preprocessor          # the champion's (scaled or raw) matrix
        self.alt_preprocessor = alt_preprocessor  # the other matrix, for blend members that need it
        self.model = model
        self.calibrator = calibrator
        self.schema = schema
        self.feature_names = feature_names
        self.bands = bands
        self.background = background
        self.meta = meta
        # Optional AI-enhanced blend: {"members": [(estimator, uses_alt_matrix)], "weights", "intercept"}.
        self.blend = blend
        self._explainer: Optional[Explainer] = None

    # ------------------------------------------------------------- persist

    def save(self, model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "version": BUNDLE_VERSION, "preprocessor": self.preprocessor, "model": self.model,
            "calibrator": self.calibrator, "bands": self.bands, "background": self.background,
            "blend": self.blend, "alt_preprocessor": self.alt_preprocessor,
        }, model_dir / "bundle.joblib", compress=3)
        (model_dir / "meta.json").write_text(json.dumps({
            "version": BUNDLE_VERSION, "schema": self.schema.to_dict(), "feature_names": self.feature_names,
            **self.meta,
        }))

    @classmethod
    def load(cls, model_dir: Path) -> "ModelBundle":
        blob = joblib.load(model_dir / "bundle.joblib")
        meta = json.loads((model_dir / "meta.json").read_text())
        schema = InputSchema.from_dict(meta.pop("schema"))
        feature_names = meta.pop("feature_names")
        return cls(blob["preprocessor"], blob["model"], blob["calibrator"], schema, feature_names,
                   blob["bands"], blob["background"], meta, blend=blob.get("blend"),
                   alt_preprocessor=blob.get("alt_preprocessor"))

    # --------------------------------------------------------------- score

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def blended_probability(self, X_pre: np.ndarray, X_alt: Optional[np.ndarray], champion_p: np.ndarray) -> np.ndarray:
        """Stacked probability: logistic meta-learner over member log-odds
        (same construction as engine.train._stack)."""
        if not self.blend:
            return champion_p
        cols = []
        for est, uses_alt in self.blend["members"]:
            if est is None:                       # the champion itself
                cols.append(champion_p)
            else:
                matrix = X_alt if uses_alt else X_pre
                if matrix is None:
                    return champion_p
                cols.append(est.predict_proba(matrix)[:, 1])
        stack = np.column_stack([self._logit(c) for c in cols])
        logit = self.blend["intercept"] + stack @ np.asarray(self.blend["weights"])
        return 1 / (1 + np.exp(-logit))

    def score(self, records: pd.DataFrame, with_reasons: bool = True) -> List[Dict[str, Any]]:
        frame = prepare_frame(self.schema.coerce(records), self.schema)
        X = np.asarray(self.preprocessor.transform(frame), dtype=float)
        X_alt = np.asarray(self.alt_preprocessor.transform(frame), dtype=float) if self.alt_preprocessor is not None else None
        raw = self.model.predict_proba(X)[:, 1]
        p = self.calibrator(self.blended_probability(X, X_alt, raw))
        scores = nuscore.to_score(p)
        bands = nuscore.assign_band(scores, self.bands)
        reasons = self.explainer().reason_codes(X, raw=frame) if with_reasons and len(X) <= 2000 else [[] for _ in scores]
        base_scores = nuscore.to_score(self.calibrator(raw))
        out = []
        for i in range(len(scores)):
            out.append({
                "probability": round(float(p[i]), 5),
                "nu_score": int(scores[i]),
                "band": str(bands[i]),
                "base_score": int(base_scores[i]),
                "ai_adjustment": int(scores[i] - base_scores[i]),
                "reasons": reasons[i] if i < len(reasons) else [],
            })
        return out

    def drift(self, records: pd.DataFrame, scores) -> Optional[Dict[str, Any]]:
        """PSI of a scored batch against the training cohort (features + score)."""
        from . import drift as drift_mod

        profiles = self.meta.get("drift_profiles")
        baseline = self.meta.get("score_baseline")
        if not profiles or not baseline:
            return None
        frame = prepare_frame(self.schema.coerce(records), self.schema)
        return drift_mod.summarise(drift_mod.feature_drift(profiles, frame),
                                   drift_mod.compare(baseline, np.asarray(scores)))

    def explainer(self) -> Explainer:
        if self._explainer is None:
            self._explainer = Explainer(self.model, self.background, self.feature_names)
        return self._explainer

    def form_schema(self) -> Dict[str, Any]:
        """What the scoring form needs: fields with types, defaults and options."""
        fields = []
        for col in self.schema.numeric:
            fields.append({"name": col, "type": "number", "default": self.schema.medians.get(col),
                           "range": self.schema.ranges.get(col)})
        for col in self.schema.categorical:
            fields.append({"name": col, "type": "category", "default": self.schema.modes.get(col),
                           "options": self.schema.categories.get(col, [])})
        return {"fields": fields, "bands": self.bands, "model": self.meta.get("champion"),
                "target": self.meta.get("target_col")}
