"""
Population Stability Index (PSI) for Model Monitoring.

PSI measures the shift in score distribution between baseline and current
populations. It's a key metric for model monitoring in financial services.

PSI Interpretation:
- PSI < 0.10: No significant change
- 0.10 <= PSI < 0.25: Moderate shift (investigate)
- PSI >= 0.25: Significant shift (action required)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class PSIResult:
    """Result of PSI calculation."""

    psi_value: float
    status: str  # "stable", "moderate_shift", "significant_shift"
    n_bins: int
    bin_details: List[Dict]
    expected_distribution: List[float]
    actual_distribution: List[float]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


def compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
    strategy: str = "quantile",
    return_details: bool = True,
) -> PSIResult:
    """
    Compute Population Stability Index.

    PSI = Σ (Actual% - Expected%) × ln(Actual% / Expected%)

    Parameters
    ----------
    expected : np.ndarray
        Baseline/expected score distribution.
    actual : np.ndarray
        Current/actual score distribution.
    n_bins : int, default=10
        Number of bins for discretization.
    strategy : str, default="quantile"
        Binning strategy: "quantile" or "uniform".
    return_details : bool, default=True
        Return detailed bin-level PSI.

    Returns
    -------
    PSIResult
        PSI calculation result.
    """
    expected = np.asarray(expected).flatten()
    actual = np.asarray(actual).flatten()

    # Remove NaN values
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    # Create bins based on expected distribution
    if strategy == "quantile":
        bin_edges = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    else:
        bin_edges = np.linspace(expected.min(), expected.max(), n_bins + 1)

    # Ensure unique bin edges
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 3:
        bin_edges = np.array([expected.min(), expected.median(), expected.max()])

    # Handle edge cases
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    # Calculate distributions
    expected_counts = np.histogram(expected, bins=bin_edges)[0]
    actual_counts = np.histogram(actual, bins=bin_edges)[0]

    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    # Apply small value to avoid division by zero
    epsilon = 1e-6
    expected_pct = np.clip(expected_pct, epsilon, 1 - epsilon)
    actual_pct = np.clip(actual_pct, epsilon, 1 - epsilon)

    # Calculate PSI
    psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    total_psi = float(np.sum(psi_values))

    # Determine status
    if total_psi < 0.10:
        status = "stable"
    elif total_psi < 0.25:
        status = "moderate_shift"
    else:
        status = "significant_shift"

    # Prepare bin details
    bin_details = []
    if return_details:
        for i in range(len(expected_pct)):
            lower = float(bin_edges[i]) if not np.isinf(bin_edges[i]) else None
            upper = float(bin_edges[i + 1]) if not np.isinf(bin_edges[i + 1]) else None
            bin_details.append({
                "bin": i + 1,
                "lower": lower,
                "upper": upper,
                "expected_pct": float(expected_pct[i]),
                "actual_pct": float(actual_pct[i]),
                "psi_contribution": float(psi_values[i]),
            })

    return PSIResult(
        psi_value=total_psi,
        status=status,
        n_bins=len(expected_pct),
        bin_details=bin_details,
        expected_distribution=expected_pct.tolist(),
        actual_distribution=actual_pct.tolist(),
    )


class PSIMonitor:
    """
    Monitor score distribution stability over time.

    Maintains a baseline distribution and tracks PSI against new samples.
    """

    def __init__(
        self,
        baseline_scores: np.ndarray,
        n_bins: int = 10,
        strategy: str = "quantile",
        alert_threshold: float = 0.25,
    ):
        """
        Initialize PSI monitor with baseline.

        Parameters
        ----------
        baseline_scores : np.ndarray
            Baseline score distribution to compare against.
        n_bins : int, default=10
            Number of bins for PSI calculation.
        strategy : str, default="quantile"
            Binning strategy.
        alert_threshold : float, default=0.25
            PSI threshold for alerts.
        """
        self.baseline = np.asarray(baseline_scores).flatten()
        self.baseline = self.baseline[~np.isnan(self.baseline)]
        self.n_bins = n_bins
        self.strategy = strategy
        self.alert_threshold = alert_threshold
        self.history: List[PSIResult] = []

        # Pre-compute bin edges from baseline
        if strategy == "quantile":
            self._bin_edges = np.percentile(
                self.baseline, np.linspace(0, 100, n_bins + 1)
            )
        else:
            self._bin_edges = np.linspace(
                self.baseline.min(), self.baseline.max(), n_bins + 1
            )
        self._bin_edges = np.unique(self._bin_edges)
        self._bin_edges[0] = -np.inf
        self._bin_edges[-1] = np.inf

        # Pre-compute baseline distribution
        counts = np.histogram(self.baseline, bins=self._bin_edges)[0]
        self._baseline_pct = counts / len(self.baseline)
        self._baseline_pct = np.clip(self._baseline_pct, 1e-6, 1 - 1e-6)

    def check(self, current_scores: np.ndarray) -> PSIResult:
        """
        Check PSI against baseline.

        Parameters
        ----------
        current_scores : np.ndarray
            Current score distribution.

        Returns
        -------
        PSIResult
            PSI calculation result.
        """
        result = compute_psi(
            expected=self.baseline,
            actual=current_scores,
            n_bins=self.n_bins,
            strategy=self.strategy,
        )
        self.history.append(result)
        return result

    def get_trend(self) -> pd.DataFrame:
        """
        Get PSI trend over time.

        Returns
        -------
        pd.DataFrame
            History of PSI checks.
        """
        if not self.history:
            return pd.DataFrame()

        records = []
        for result in self.history:
            records.append({
                "timestamp": result.timestamp,
                "psi_value": result.psi_value,
                "status": result.status,
            })

        return pd.DataFrame(records)

    def is_drifting(self) -> bool:
        """Check if latest PSI indicates drift."""
        if not self.history:
            return False
        return self.history[-1].psi_value >= self.alert_threshold

    def reset_baseline(self, new_baseline: np.ndarray) -> None:
        """Reset baseline and clear history."""
        self.__init__(
            baseline_scores=new_baseline,
            n_bins=self.n_bins,
            strategy=self.strategy,
            alert_threshold=self.alert_threshold,
        )


def compute_psi_matrix(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Compute PSI for multiple features.

    Parameters
    ----------
    baseline_df : pd.DataFrame
        Baseline data.
    current_df : pd.DataFrame
        Current data.
    columns : Optional[List[str]]
        Columns to analyze. If None, use all numeric columns.
    n_bins : int, default=10
        Number of bins.

    Returns
    -------
    pd.DataFrame
        PSI values and status for each feature.
    """
    if columns is None:
        columns = baseline_df.select_dtypes(include=[np.number]).columns.tolist()

    results = []
    for col in columns:
        if col not in current_df.columns:
            continue

        result = compute_psi(
            expected=baseline_df[col].values,
            actual=current_df[col].values,
            n_bins=n_bins,
            return_details=False,
        )

        results.append({
            "feature": col,
            "psi": result.psi_value,
            "status": result.status,
        })

    df = pd.DataFrame(results)
    return df.sort_values("psi", ascending=False).reset_index(drop=True)
