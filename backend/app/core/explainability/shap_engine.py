"""
SHAP-based Model Explainability Engine.

Provides global and local model explanations using SHAP values.
Essential for regulatory compliance and model governance in financial services.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


@dataclass
class ShapExplanation:
    """SHAP explanation for predictions."""

    feature_names: List[str]
    shap_values: np.ndarray  # (n_samples, n_features)
    base_value: float
    expected_value: float
    feature_values: Optional[np.ndarray] = None


@dataclass
class GlobalExplanation:
    """Global feature importance from SHAP."""

    feature_importance: Dict[str, float]
    feature_ranking: List[str]
    summary_data: Optional[Dict] = None


@dataclass
class LocalExplanation:
    """Local explanation for individual prediction."""

    prediction: float
    base_value: float
    feature_contributions: Dict[str, float]
    top_positive: List[Dict[str, float]]
    top_negative: List[Dict[str, float]]


class ShapExplainer:
    """
    SHAP-based model explainability.

    Provides:
    - Global feature importance via mean |SHAP|
    - Local explanations for individual predictions
    - Summary and dependence plot data
    """

    def __init__(
        self,
        model: Any,
        background_samples: int = 100,
        algorithm: str = "auto",
    ):
        """
        Initialize SHAP explainer.

        Parameters
        ----------
        model : Any
            Fitted model (sklearn Pipeline or individual model).
        background_samples : int, default=100
            Number of background samples for kernel explainer.
        algorithm : str, default="auto"
            SHAP algorithm: "auto", "tree", "kernel", "linear".
        """
        self.model = model
        self.background_samples = background_samples
        self.algorithm = algorithm
        self._explainer = None
        self._background_data = None
        self._feature_names: List[str] = []

    def fit(self, X_background: pd.DataFrame) -> "ShapExplainer":
        """
        Initialize SHAP explainer with background data.

        Parameters
        ----------
        X_background : pd.DataFrame
            Background data for SHAP calculations.

        Returns
        -------
        self
        """
        try:
            import shap
        except ImportError:
            raise ImportError("shap package required. Install with: pip install shap")

        self._feature_names = list(X_background.columns)

        # Sample background data if needed
        if len(X_background) > self.background_samples:
            X_background = X_background.sample(
                n=self.background_samples, random_state=42
            )

        self._background_data = X_background

        # Create appropriate explainer
        model_to_explain = self._get_model_for_shap()

        if self.algorithm == "auto":
            # Try to use TreeExplainer for tree models
            try:
                self._explainer = shap.TreeExplainer(model_to_explain)
                self.algorithm = "tree"
            except Exception:
                # Fall back to kernel explainer
                self._explainer = shap.KernelExplainer(
                    self._predict_proba_wrapper,
                    X_background,
                )
                self.algorithm = "kernel"
        elif self.algorithm == "tree":
            self._explainer = shap.TreeExplainer(model_to_explain)
        elif self.algorithm == "linear":
            self._explainer = shap.LinearExplainer(
                model_to_explain, X_background
            )
        else:
            self._explainer = shap.KernelExplainer(
                self._predict_proba_wrapper,
                X_background,
            )

        return self

    def _get_model_for_shap(self) -> Any:
        """Extract the actual model from sklearn Pipeline if needed."""
        from sklearn.pipeline import Pipeline

        if isinstance(self.model, Pipeline):
            # Get the final estimator
            return self.model.named_steps.get("model", self.model[-1])
        return self.model

    def _predict_proba_wrapper(self, X: np.ndarray) -> np.ndarray:
        """Wrapper for predict_proba to work with SHAP."""
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict(X)

    def explain_global(
        self,
        X: pd.DataFrame,
        max_samples: int = 1000,
    ) -> GlobalExplanation:
        """
        Compute global feature importance via mean |SHAP|.

        Parameters
        ----------
        X : pd.DataFrame
            Data to explain.
        max_samples : int, default=1000
            Maximum samples to use for computation.

        Returns
        -------
        GlobalExplanation
            Global importance ranking.
        """
        if self._explainer is None:
            raise ValueError("Explainer not fitted. Call fit() first.")

        # Sample if needed
        if len(X) > max_samples:
            X = X.sample(n=max_samples, random_state=42)

        # Compute SHAP values
        shap_values = self._compute_shap_values(X)

        # For binary classification, use values for positive class
        if len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]

        # Compute mean absolute SHAP
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        # Create importance dict
        feature_importance = {
            name: float(imp)
            for name, imp in zip(self._feature_names, mean_abs_shap)
        }

        # Sort by importance
        feature_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        )
        feature_ranking = list(feature_importance.keys())

        # Summary data for visualization
        summary_data = {
            "shap_values": shap_values.tolist()[:100],  # Limit for JSON
            "feature_names": self._feature_names,
            "feature_values": X.values.tolist()[:100],
        }

        return GlobalExplanation(
            feature_importance=feature_importance,
            feature_ranking=feature_ranking,
            summary_data=summary_data,
        )

    def explain_local(
        self,
        X: pd.DataFrame,
        index: Optional[int] = None,
        top_n: int = 10,
    ) -> LocalExplanation:
        """
        Compute local explanation for a single prediction.

        Parameters
        ----------
        X : pd.DataFrame
            Single row or DataFrame with row to explain.
        index : Optional[int]
            If X has multiple rows, which one to explain.
        top_n : int, default=10
            Number of top features to include.

        Returns
        -------
        LocalExplanation
            Local explanation.
        """
        if self._explainer is None:
            raise ValueError("Explainer not fitted. Call fit() first.")

        # Get single row
        if index is not None:
            X = X.iloc[[index]]
        elif len(X) > 1:
            X = X.iloc[[0]]

        # Compute SHAP values
        shap_values = self._compute_shap_values(X)

        # Handle binary classification
        if len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]

        shap_values = shap_values[0]  # First row

        # Get base value
        if hasattr(self._explainer, "expected_value"):
            base_value = self._explainer.expected_value
            if isinstance(base_value, np.ndarray):
                base_value = base_value[1] if len(base_value) > 1 else base_value[0]
        else:
            base_value = 0.5

        # Get prediction
        if hasattr(self.model, "predict_proba"):
            prediction = float(self.model.predict_proba(X)[0, 1])
        else:
            prediction = float(self.model.predict(X)[0])

        # Feature contributions
        feature_contributions = {
            name: float(val)
            for name, val in zip(self._feature_names, shap_values)
        }

        # Sort by absolute value
        sorted_features = sorted(
            feature_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )

        # Top positive and negative contributions
        top_positive = [
            {"feature": k, "contribution": v}
            for k, v in sorted_features
            if v > 0
        ][:top_n]

        top_negative = [
            {"feature": k, "contribution": v}
            for k, v in sorted_features
            if v < 0
        ][:top_n]

        return LocalExplanation(
            prediction=prediction,
            base_value=float(base_value),
            feature_contributions=feature_contributions,
            top_positive=top_positive,
            top_negative=top_negative,
        )

    def _compute_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Compute SHAP values for given data."""
        try:
            import shap
        except ImportError:
            raise ImportError("shap package required")

        if self.algorithm == "tree":
            return self._explainer.shap_values(X)
        else:
            return self._explainer.shap_values(X.values)

    def get_summary_plot_data(
        self,
        X: pd.DataFrame,
        max_display: int = 20,
    ) -> Dict:
        """
        Get data for SHAP summary plot.

        Parameters
        ----------
        X : pd.DataFrame
            Data to explain.
        max_display : int, default=20
            Maximum features to display.

        Returns
        -------
        Dict
            Data for visualization.
        """
        if self._explainer is None:
            raise ValueError("Explainer not fitted. Call fit() first.")

        shap_values = self._compute_shap_values(X)

        if len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]

        # Get top features by mean |SHAP|
        mean_abs = np.abs(shap_values).mean(axis=0)
        top_indices = np.argsort(mean_abs)[::-1][:max_display]

        return {
            "shap_values": shap_values[:, top_indices].tolist(),
            "feature_values": X.iloc[:, top_indices].values.tolist(),
            "feature_names": [self._feature_names[i] for i in top_indices],
            "mean_abs_shap": mean_abs[top_indices].tolist(),
        }

    def get_dependence_plot_data(
        self,
        X: pd.DataFrame,
        feature: str,
        interaction_feature: Optional[str] = None,
    ) -> Dict:
        """
        Get data for SHAP dependence plot.

        Parameters
        ----------
        X : pd.DataFrame
            Data to explain.
        feature : str
            Main feature.
        interaction_feature : Optional[str]
            Feature for interaction coloring.

        Returns
        -------
        Dict
            Data for dependence plot.
        """
        if self._explainer is None:
            raise ValueError("Explainer not fitted. Call fit() first.")

        if feature not in self._feature_names:
            raise ValueError(f"Feature '{feature}' not found")

        shap_values = self._compute_shap_values(X)

        if len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]

        feature_idx = self._feature_names.index(feature)

        result = {
            "feature_values": X[feature].values.tolist(),
            "shap_values": shap_values[:, feature_idx].tolist(),
            "feature_name": feature,
        }

        if interaction_feature and interaction_feature in X.columns:
            result["interaction_values"] = X[interaction_feature].values.tolist()
            result["interaction_name"] = interaction_feature

        return result


def compute_feature_importance_from_shap(
    explainer: ShapExplainer,
    X: pd.DataFrame,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Compute feature importance ranking from SHAP.

    Parameters
    ----------
    explainer : ShapExplainer
        Fitted SHAP explainer.
    X : pd.DataFrame
        Data for importance calculation.
    top_n : int, default=20
        Number of top features.

    Returns
    -------
    pd.DataFrame
        Feature importance table.
    """
    global_exp = explainer.explain_global(X)

    records = []
    for rank, (feature, importance) in enumerate(
        global_exp.feature_importance.items(), 1
    ):
        if rank > top_n:
            break
        records.append({
            "rank": rank,
            "feature": feature,
            "mean_abs_shap": importance,
        })

    return pd.DataFrame(records)
