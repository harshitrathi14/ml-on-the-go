"""
Comprehensive Evaluation Metrics for Financial Risk Models.

Provides standard ML metrics plus financial-specific metrics:
- KS (Kolmogorov-Smirnov) statistic
- Gini coefficient
- Stability metrics across data splits
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    log_loss,
    brier_score_loss,
    classification_report,
)


@dataclass
class ModelMetrics:
    """Complete metrics for a model evaluation."""

    # Standard classification metrics
    accuracy: float
    precision: float
    recall: float
    f1: float

    # Probabilistic metrics
    roc_auc: float
    gini: float
    ks_statistic: float
    log_loss: float
    brier_score: float

    # Threshold-dependent
    confusion_matrix: List[List[int]]
    classification_report: Dict[str, Dict[str, float]]

    # Optional curve data
    roc_curve: Optional[Dict[str, List[float]]] = None
    ks_curve: Optional[Dict[str, List[float]]] = None


@dataclass
class StabilityMetrics:
    """Stability metrics across multiple data splits."""

    # Metric stability (std dev across splits)
    auc_stability: float
    ks_stability: float
    gini_stability: float

    # Performance degradation (train vs test/OOT)
    auc_degradation: Dict[str, float] = field(default_factory=dict)
    ks_degradation: Dict[str, float] = field(default_factory=dict)

    # Score distribution stability
    score_mean_by_split: Dict[str, float] = field(default_factory=dict)
    score_std_by_split: Dict[str, float] = field(default_factory=dict)


def compute_ks_statistic(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    return_curve: bool = False,
) -> Tuple[float, Optional[Dict[str, List[float]]]]:
    """
    Compute Kolmogorov-Smirnov statistic.

    KS measures the maximum separation between the cumulative distribution
    functions of positive and negative classes. Higher KS indicates better
    discrimination.

    Interpretation:
    - KS < 20: Poor model
    - 20 <= KS < 40: Average model
    - 40 <= KS < 50: Good model
    - 50 <= KS < 60: Very good model
    - KS >= 60: Excellent model (check for overfitting)

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_prob : np.ndarray
        Predicted probabilities for positive class.
    return_curve : bool, default=False
        If True, return KS curve data.

    Returns
    -------
    ks_stat : float
        KS statistic (0-100 scale).
    curve_data : Optional[Dict]
        KS curve data if return_curve=True.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    # Separate probabilities by class
    pos_probs = y_prob[y_true == 1]
    neg_probs = y_prob[y_true == 0]

    # Sort by probability
    all_probs = np.sort(np.unique(y_prob))

    # Compute cumulative distributions
    pos_cdf = np.array([np.mean(pos_probs <= p) for p in all_probs])
    neg_cdf = np.array([np.mean(neg_probs <= p) for p in all_probs])

    # KS is maximum difference
    ks_values = np.abs(pos_cdf - neg_cdf)
    ks_stat = np.max(ks_values) * 100  # Scale to 0-100

    curve_data = None
    if return_curve:
        curve_data = {
            "thresholds": all_probs.tolist(),
            "pos_cdf": pos_cdf.tolist(),
            "neg_cdf": neg_cdf.tolist(),
            "ks_values": ks_values.tolist(),
        }

    return ks_stat, curve_data


def compute_gini(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Compute Gini coefficient from ROC-AUC.

    Gini = 2 * AUC - 1

    Interpretation:
    - Gini close to 0: Random model
    - Gini close to 1: Perfect model
    - Gini around 0.4-0.6: Good credit risk model

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_prob : np.ndarray
        Predicted probabilities.

    Returns
    -------
    float
        Gini coefficient.
    """
    auc = roc_auc_score(y_true, y_prob)
    return 2 * auc - 1


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    include_curves: bool = True,
) -> ModelMetrics:
    """
    Compute comprehensive metrics suite.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_pred : np.ndarray
        Predicted binary labels.
    y_prob : np.ndarray
        Predicted probabilities.
    include_curves : bool, default=True
        Include ROC and KS curve data.

    Returns
    -------
    ModelMetrics
        Complete metrics object.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    # Standard metrics
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # Probabilistic metrics
    auc = float(roc_auc_score(y_true, y_prob))
    gini = float(compute_gini(y_true, y_prob))
    ks_stat, ks_curve_data = compute_ks_statistic(y_true, y_prob, return_curve=include_curves)
    ll = float(log_loss(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred).tolist()

    # Classification report
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    # ROC curve data
    roc_data = None
    if include_curves:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_data = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
        }

    return ModelMetrics(
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1=f1,
        roc_auc=auc,
        gini=gini,
        ks_statistic=ks_stat,
        log_loss=ll,
        brier_score=brier,
        confusion_matrix=cm,
        classification_report=report,
        roc_curve=roc_data,
        ks_curve=ks_curve_data,
    )


class MetricsCalculator:
    """
    Calculate metrics across multiple data splits.

    Useful for evaluating model stability and generalization.
    """

    def __init__(self, splits: Dict[str, Tuple[pd.DataFrame, pd.Series]]):
        """
        Initialize with data splits.

        Parameters
        ----------
        splits : Dict[str, Tuple[pd.DataFrame, pd.Series]]
            Dictionary mapping split names to (X, y) tuples.
        """
        self.splits = splits

    def evaluate(
        self,
        model: Any,
        feature_cols: Optional[List[str]] = None,
    ) -> Dict[str, ModelMetrics]:
        """
        Evaluate model on all splits.

        Parameters
        ----------
        model : Any
            Fitted model with predict and predict_proba methods.
        feature_cols : Optional[List[str]]
            Feature columns to use. If None, use all columns.

        Returns
        -------
        Dict[str, ModelMetrics]
            Metrics for each split.
        """
        results = {}

        for split_name, (X, y) in self.splits.items():
            if feature_cols:
                X = X[feature_cols]

            y_pred = model.predict(X)
            y_prob = model.predict_proba(X)[:, 1]

            metrics = compute_all_metrics(
                y_true=y.values,
                y_pred=y_pred,
                y_prob=y_prob,
                include_curves=True,
            )
            results[split_name] = metrics

        return results

    def compute_stability(
        self,
        metrics_by_split: Dict[str, ModelMetrics],
        baseline_split: str = "train",
    ) -> StabilityMetrics:
        """
        Compute stability metrics across splits.

        Parameters
        ----------
        metrics_by_split : Dict[str, ModelMetrics]
            Metrics for each split.
        baseline_split : str, default="train"
            Split to use as baseline for degradation calculation.

        Returns
        -------
        StabilityMetrics
            Stability analysis results.
        """
        # Extract key metrics
        aucs = [m.roc_auc for m in metrics_by_split.values()]
        ks_values = [m.ks_statistic for m in metrics_by_split.values()]
        ginis = [m.gini for m in metrics_by_split.values()]

        # Compute stability (lower is better)
        auc_stability = float(np.std(aucs))
        ks_stability = float(np.std(ks_values))
        gini_stability = float(np.std(ginis))

        # Compute degradation from baseline
        baseline_metrics = metrics_by_split.get(baseline_split)
        auc_degradation = {}
        ks_degradation = {}

        if baseline_metrics:
            for split_name, metrics in metrics_by_split.items():
                if split_name != baseline_split:
                    auc_degradation[split_name] = (
                        baseline_metrics.roc_auc - metrics.roc_auc
                    )
                    ks_degradation[split_name] = (
                        baseline_metrics.ks_statistic - metrics.ks_statistic
                    )

        return StabilityMetrics(
            auc_stability=auc_stability,
            ks_stability=ks_stability,
            gini_stability=gini_stability,
            auc_degradation=auc_degradation,
            ks_degradation=ks_degradation,
        )


def compute_lift_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, List[float]]:
    """
    Compute lift curve data.

    Lift = (positive rate in bin) / (overall positive rate)

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_prob : np.ndarray
        Predicted probabilities.
    n_bins : int, default=10
        Number of bins (deciles).

    Returns
    -------
    Dict[str, List[float]]
        Lift curve data with bins and lift values.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    # Sort by predicted probability (descending)
    sorted_indices = np.argsort(y_prob)[::-1]
    y_true_sorted = y_true[sorted_indices]

    # Overall positive rate
    overall_rate = np.mean(y_true)

    # Compute lift per decile
    bin_size = len(y_true) // n_bins
    lifts = []
    cumulative_lifts = []

    for i in range(n_bins):
        start_idx = i * bin_size
        end_idx = (i + 1) * bin_size if i < n_bins - 1 else len(y_true)

        bin_rate = np.mean(y_true_sorted[start_idx:end_idx])
        lift = bin_rate / overall_rate if overall_rate > 0 else 1.0
        lifts.append(float(lift))

        # Cumulative lift
        cumulative_rate = np.mean(y_true_sorted[:end_idx])
        cumulative_lift = cumulative_rate / overall_rate if overall_rate > 0 else 1.0
        cumulative_lifts.append(float(cumulative_lift))

    return {
        "decile": list(range(1, n_bins + 1)),
        "lift": lifts,
        "cumulative_lift": cumulative_lifts,
        "overall_rate": float(overall_rate),
    }


def compute_gains_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_points: int = 100,
) -> Dict[str, List[float]]:
    """
    Compute cumulative gains curve.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_prob : np.ndarray
        Predicted probabilities.
    n_points : int, default=100
        Number of points on the curve.

    Returns
    -------
    Dict[str, List[float]]
        Gains curve data.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    # Sort by predicted probability (descending)
    sorted_indices = np.argsort(y_prob)[::-1]
    y_true_sorted = y_true[sorted_indices]

    total_positives = np.sum(y_true)
    n_samples = len(y_true)

    percentages = np.linspace(0, 1, n_points)
    gains = []

    for pct in percentages:
        n = int(pct * n_samples)
        if n == 0:
            gains.append(0.0)
        else:
            captured = np.sum(y_true_sorted[:n])
            gains.append(float(captured / total_positives))

    return {
        "percentage_population": percentages.tolist(),
        "percentage_captured": gains,
    }


def compare_models(
    model_metrics: Dict[str, Dict[str, ModelMetrics]],
    sort_by: str = "roc_auc",
    split: str = "test",
) -> pd.DataFrame:
    """
    Compare multiple models across metrics.

    Parameters
    ----------
    model_metrics : Dict[str, Dict[str, ModelMetrics]]
        Metrics for each model, keyed by model name.
    sort_by : str, default="roc_auc"
        Metric to sort by.
    split : str, default="test"
        Split to use for comparison.

    Returns
    -------
    pd.DataFrame
        Comparison table.
    """
    records = []

    for model_name, split_metrics in model_metrics.items():
        if split not in split_metrics:
            continue

        metrics = split_metrics[split]
        records.append({
            "model": model_name,
            "accuracy": metrics.accuracy,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "roc_auc": metrics.roc_auc,
            "gini": metrics.gini,
            "ks_statistic": metrics.ks_statistic,
            "log_loss": metrics.log_loss,
            "brier_score": metrics.brier_score,
        })

    df = pd.DataFrame(records)
    if sort_by in df.columns:
        ascending = sort_by in ["log_loss", "brier_score"]
        df = df.sort_values(sort_by, ascending=ascending)

    return df.reset_index(drop=True)
