"""
Characteristic Stability Index (CSI) for Feature Drift Detection.

CSI monitors individual feature distributions for drift, similar to PSI
but applied to input features rather than model scores.

This is critical for credit risk models where feature drift can indicate
changes in customer population or data quality issues.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .psi import compute_psi, PSIResult


@dataclass
class CSIResult:
    """Result of CSI calculation for multiple features."""

    overall_status: str  # "stable", "warning", "alert"
    n_features: int
    n_drifting: int
    feature_results: Dict[str, PSIResult]
    drifting_features: List[str]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class CSIMonitor:
    """
    Monitor feature distribution stability.

    Tracks CSI (PSI applied to features) for detecting input drift.
    """

    def __init__(
        self,
        baseline_data: pd.DataFrame,
        features: Optional[List[str]] = None,
        n_bins: int = 10,
        strategy: str = "quantile",
        warning_threshold: float = 0.10,
        alert_threshold: float = 0.25,
    ):
        """
        Initialize CSI monitor.

        Parameters
        ----------
        baseline_data : pd.DataFrame
            Baseline feature data.
        features : Optional[List[str]]
            Features to monitor. If None, use all numeric columns.
        n_bins : int, default=10
            Number of bins for each feature.
        strategy : str, default="quantile"
            Binning strategy.
        warning_threshold : float, default=0.10
            CSI threshold for warnings.
        alert_threshold : float, default=0.25
            CSI threshold for alerts.
        """
        self.baseline = baseline_data.copy()
        self.n_bins = n_bins
        self.strategy = strategy
        self.warning_threshold = warning_threshold
        self.alert_threshold = alert_threshold
        self.history: List[CSIResult] = []

        # Determine features to monitor
        if features is None:
            self.features = self.baseline.select_dtypes(
                include=[np.number]
            ).columns.tolist()
        else:
            self.features = [f for f in features if f in self.baseline.columns]

    def check_all(self, current_data: pd.DataFrame) -> CSIResult:
        """
        Compute CSI for all monitored features.

        Parameters
        ----------
        current_data : pd.DataFrame
            Current feature data.

        Returns
        -------
        CSIResult
            CSI results for all features.
        """
        feature_results = {}
        drifting_features = []

        for feature in self.features:
            if feature not in current_data.columns:
                continue

            result = compute_psi(
                expected=self.baseline[feature].values,
                actual=current_data[feature].values,
                n_bins=self.n_bins,
                strategy=self.strategy,
            )
            feature_results[feature] = result

            if result.psi_value >= self.alert_threshold:
                drifting_features.append(feature)

        # Determine overall status
        n_drifting = len(drifting_features)
        drift_ratio = n_drifting / len(self.features) if self.features else 0

        if drift_ratio >= 0.2 or n_drifting >= 5:
            overall_status = "alert"
        elif n_drifting >= 1:
            overall_status = "warning"
        else:
            overall_status = "stable"

        result = CSIResult(
            overall_status=overall_status,
            n_features=len(self.features),
            n_drifting=n_drifting,
            feature_results=feature_results,
            drifting_features=drifting_features,
        )

        self.history.append(result)
        return result

    def check_feature(self, current_data: pd.DataFrame, feature: str) -> PSIResult:
        """
        Compute CSI for a single feature.

        Parameters
        ----------
        current_data : pd.DataFrame
            Current feature data.
        feature : str
            Feature to check.

        Returns
        -------
        PSIResult
            CSI result for the feature.
        """
        if feature not in self.baseline.columns:
            raise ValueError(f"Feature '{feature}' not in baseline")
        if feature not in current_data.columns:
            raise ValueError(f"Feature '{feature}' not in current data")

        return compute_psi(
            expected=self.baseline[feature].values,
            actual=current_data[feature].values,
            n_bins=self.n_bins,
            strategy=self.strategy,
        )

    def get_drifting_features(
        self,
        current_data: pd.DataFrame,
        threshold: Optional[float] = None,
    ) -> List[str]:
        """
        Get list of features with CSI above threshold.

        Parameters
        ----------
        current_data : pd.DataFrame
            Current feature data.
        threshold : Optional[float]
            CSI threshold. If None, use alert_threshold.

        Returns
        -------
        List[str]
            Features with drift.
        """
        threshold = threshold or self.alert_threshold
        drifting = []

        for feature in self.features:
            if feature not in current_data.columns:
                continue

            result = compute_psi(
                expected=self.baseline[feature].values,
                actual=current_data[feature].values,
                n_bins=self.n_bins,
                strategy=self.strategy,
                return_details=False,
            )

            if result.psi_value >= threshold:
                drifting.append(feature)

        return drifting

    def get_summary(self) -> pd.DataFrame:
        """
        Get summary of latest CSI check.

        Returns
        -------
        pd.DataFrame
            CSI values for all features.
        """
        if not self.history:
            return pd.DataFrame()

        latest = self.history[-1]
        records = []

        for feature, result in latest.feature_results.items():
            records.append({
                "feature": feature,
                "csi": result.psi_value,
                "status": result.status,
                "is_drifting": feature in latest.drifting_features,
            })

        df = pd.DataFrame(records)
        return df.sort_values("csi", ascending=False).reset_index(drop=True)

    def get_trend(self, feature: str) -> pd.DataFrame:
        """
        Get CSI trend for a specific feature.

        Parameters
        ----------
        feature : str
            Feature name.

        Returns
        -------
        pd.DataFrame
            CSI values over time.
        """
        records = []
        for result in self.history:
            if feature in result.feature_results:
                records.append({
                    "timestamp": result.timestamp,
                    "csi": result.feature_results[feature].psi_value,
                    "status": result.feature_results[feature].status,
                })

        return pd.DataFrame(records)

    def reset_baseline(self, new_baseline: pd.DataFrame) -> None:
        """Reset baseline and clear history."""
        self.__init__(
            baseline_data=new_baseline,
            features=self.features,
            n_bins=self.n_bins,
            strategy=self.strategy,
            warning_threshold=self.warning_threshold,
            alert_threshold=self.alert_threshold,
        )


def compute_feature_drift_report(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features: Optional[List[str]] = None,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Generate comprehensive feature drift report.

    Parameters
    ----------
    baseline_df : pd.DataFrame
        Baseline data.
    current_df : pd.DataFrame
        Current data.
    features : Optional[List[str]]
        Features to analyze.
    n_bins : int, default=10
        Number of bins.

    Returns
    -------
    pd.DataFrame
        Drift report with statistics.
    """
    if features is None:
        features = baseline_df.select_dtypes(include=[np.number]).columns.tolist()

    records = []
    for feature in features:
        if feature not in current_df.columns:
            continue

        baseline_vals = baseline_df[feature].dropna()
        current_vals = current_df[feature].dropna()

        result = compute_psi(
            expected=baseline_vals.values,
            actual=current_vals.values,
            n_bins=n_bins,
            return_details=False,
        )

        records.append({
            "feature": feature,
            "csi": result.psi_value,
            "status": result.status,
            "baseline_mean": float(baseline_vals.mean()),
            "current_mean": float(current_vals.mean()),
            "mean_shift": float(current_vals.mean() - baseline_vals.mean()),
            "baseline_std": float(baseline_vals.std()),
            "current_std": float(current_vals.std()),
            "std_ratio": float(current_vals.std() / max(baseline_vals.std(), 1e-8)),
            "baseline_missing_pct": float(baseline_df[feature].isna().mean() * 100),
            "current_missing_pct": float(current_df[feature].isna().mean() * 100),
        })

    df = pd.DataFrame(records)
    return df.sort_values("csi", ascending=False).reset_index(drop=True)
