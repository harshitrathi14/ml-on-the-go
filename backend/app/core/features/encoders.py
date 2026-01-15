"""
Feature Encoding Transformers for Credit Risk Modeling.

Provides WOE (Weight of Evidence), Target Encoding, and other
encoding strategies optimized for financial risk applications.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


@dataclass
class WOEBin:
    """Represents a single WOE bin."""

    bin_id: int
    lower: float
    upper: float
    count: int
    event_count: int
    non_event_count: int
    event_rate: float
    woe: float
    iv: float


class WOEEncoder(BaseEstimator, TransformerMixin):
    """
    Weight of Evidence (WOE) Encoder for binary classification.

    WOE encoding is widely used in credit scoring and risk modeling.
    It transforms categorical/numerical features into WOE values that
    represent the strength of each category in predicting the target.

    WOE = ln(% of Events / % of Non-Events)
    IV = sum((% Events - % Non-Events) * WOE)

    Parameters
    ----------
    min_samples : int, default=100
        Minimum samples per bin for numeric features.
    n_bins : int, default=10
        Number of bins for numeric features.
    regularization : float, default=0.5
        Laplace smoothing to avoid division by zero.
    handle_missing : str, default='separate'
        How to handle missing values: 'separate' creates a missing bin,
        'most_frequent' assigns to most frequent bin.

    Attributes
    ----------
    woe_maps_ : Dict[str, Dict]
        WOE mapping for each feature.
    iv_ : Dict[str, float]
        Information Value for each feature.
    bin_edges_ : Dict[str, np.ndarray]
        Bin edges for numeric features.
    """

    def __init__(
        self,
        min_samples: int = 100,
        n_bins: int = 10,
        regularization: float = 0.5,
        handle_missing: str = "separate",
    ):
        self.min_samples = min_samples
        self.n_bins = n_bins
        self.regularization = regularization
        self.handle_missing = handle_missing
        self.woe_maps_: Dict[str, Dict] = {}
        self.iv_: Dict[str, float] = {}
        self.bin_edges_: Dict[str, np.ndarray] = {}
        self._feature_types: Dict[str, str] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WOEEncoder":
        """
        Compute WOE values for each feature.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Binary target (0/1).

        Returns
        -------
        self
        """
        X = pd.DataFrame(X).copy()
        y = pd.Series(y).values

        # Global event/non-event counts
        total_events = np.sum(y == 1)
        total_non_events = np.sum(y == 0)

        for col in X.columns:
            if X[col].dtype == "object" or X[col].nunique() < self.n_bins:
                # Categorical feature
                self._feature_types[col] = "categorical"
                self._fit_categorical(X[col], y, col, total_events, total_non_events)
            else:
                # Numeric feature
                self._feature_types[col] = "numeric"
                self._fit_numeric(X[col], y, col, total_events, total_non_events)

        return self

    def _fit_categorical(
        self,
        col_data: pd.Series,
        y: np.ndarray,
        col_name: str,
        total_events: int,
        total_non_events: int,
    ) -> None:
        """Fit WOE for categorical feature."""
        woe_map = {}
        iv_total = 0.0

        # Handle missing values
        if self.handle_missing == "separate":
            col_data = col_data.fillna("__MISSING__")
        else:
            mode_val = col_data.mode().iloc[0] if len(col_data.mode()) > 0 else "UNKNOWN"
            col_data = col_data.fillna(mode_val)

        for category in col_data.unique():
            mask = col_data == category
            events = np.sum(y[mask] == 1)
            non_events = np.sum(y[mask] == 0)

            # Apply Laplace smoothing
            events_pct = (events + self.regularization) / (total_events + self.regularization)
            non_events_pct = (non_events + self.regularization) / (total_non_events + self.regularization)

            woe = np.log(events_pct / non_events_pct)
            iv = (events_pct - non_events_pct) * woe

            woe_map[category] = {
                "woe": woe,
                "iv": iv,
                "count": int(mask.sum()),
                "event_rate": events / max(mask.sum(), 1),
            }
            iv_total += iv

        self.woe_maps_[col_name] = woe_map
        self.iv_[col_name] = iv_total

    def _fit_numeric(
        self,
        col_data: pd.Series,
        y: np.ndarray,
        col_name: str,
        total_events: int,
        total_non_events: int,
    ) -> None:
        """Fit WOE for numeric feature with binning."""
        woe_map = {}
        iv_total = 0.0

        # Handle missing values
        missing_mask = col_data.isna()
        non_missing = col_data[~missing_mask]
        y_non_missing = y[~missing_mask]

        # Create bins using quantiles
        try:
            bin_edges = np.percentile(
                non_missing.dropna(),
                np.linspace(0, 100, self.n_bins + 1)
            )
            bin_edges = np.unique(bin_edges)
        except Exception:
            bin_edges = np.array([non_missing.min(), non_missing.max()])

        self.bin_edges_[col_name] = bin_edges

        # Assign bins
        binned = np.digitize(non_missing.values, bin_edges[1:-1])

        for bin_idx in range(len(bin_edges) - 1):
            mask = binned == bin_idx
            events = np.sum(y_non_missing[mask] == 1)
            non_events = np.sum(y_non_missing[mask] == 0)

            events_pct = (events + self.regularization) / (total_events + self.regularization)
            non_events_pct = (non_events + self.regularization) / (total_non_events + self.regularization)

            woe = np.log(events_pct / non_events_pct)
            iv = (events_pct - non_events_pct) * woe

            woe_map[bin_idx] = {
                "woe": woe,
                "iv": iv,
                "lower": bin_edges[bin_idx],
                "upper": bin_edges[bin_idx + 1] if bin_idx < len(bin_edges) - 1 else np.inf,
                "count": int(mask.sum()),
            }
            iv_total += iv

        # Handle missing values as separate bin
        if missing_mask.any() and self.handle_missing == "separate":
            events = np.sum(y[missing_mask] == 1)
            non_events = np.sum(y[missing_mask] == 0)

            events_pct = (events + self.regularization) / (total_events + self.regularization)
            non_events_pct = (non_events + self.regularization) / (total_non_events + self.regularization)

            woe = np.log(events_pct / non_events_pct)
            iv = (events_pct - non_events_pct) * woe

            woe_map["__MISSING__"] = {
                "woe": woe,
                "iv": iv,
                "count": int(missing_mask.sum()),
            }
            iv_total += iv

        self.woe_maps_[col_name] = woe_map
        self.iv_[col_name] = iv_total

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply WOE transformation.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.

        Returns
        -------
        pd.DataFrame
            WOE-transformed features.
        """
        X = pd.DataFrame(X).copy()
        result = pd.DataFrame(index=X.index)

        for col in X.columns:
            if col not in self.woe_maps_:
                continue

            if self._feature_types.get(col) == "categorical":
                result[col] = self._transform_categorical(X[col], col)
            else:
                result[col] = self._transform_numeric(X[col], col)

        return result

    def _transform_categorical(self, col_data: pd.Series, col_name: str) -> pd.Series:
        """Transform categorical feature to WOE."""
        woe_map = self.woe_maps_[col_name]

        # Handle missing
        if self.handle_missing == "separate":
            col_data = col_data.fillna("__MISSING__")

        # Map to WOE values
        default_woe = 0.0  # Neutral WOE for unknown categories
        return col_data.map(lambda x: woe_map.get(x, {}).get("woe", default_woe))

    def _transform_numeric(self, col_data: pd.Series, col_name: str) -> pd.Series:
        """Transform numeric feature to WOE."""
        woe_map = self.woe_maps_[col_name]
        bin_edges = self.bin_edges_.get(col_name, np.array([]))

        result = pd.Series(index=col_data.index, dtype=float)

        # Handle missing
        missing_mask = col_data.isna()
        if missing_mask.any():
            missing_woe = woe_map.get("__MISSING__", {}).get("woe", 0.0)
            result[missing_mask] = missing_woe

        # Bin non-missing values
        non_missing = col_data[~missing_mask]
        if len(non_missing) > 0 and len(bin_edges) > 1:
            binned = np.digitize(non_missing.values, bin_edges[1:-1])
            for idx, bin_idx in zip(non_missing.index, binned):
                result[idx] = woe_map.get(bin_idx, {}).get("woe", 0.0)

        return result

    def get_iv(self) -> Dict[str, float]:
        """
        Get Information Value for each feature.

        IV Interpretation:
        - < 0.02: Useless for prediction
        - 0.02 - 0.1: Weak predictor
        - 0.1 - 0.3: Medium predictor
        - 0.3 - 0.5: Strong predictor
        - > 0.5: Suspicious (check for overfitting)

        Returns
        -------
        Dict[str, float]
            IV values sorted by importance.
        """
        return dict(sorted(self.iv_.items(), key=lambda x: x[1], reverse=True))

    def get_woe_report(self, feature: str) -> pd.DataFrame:
        """Get detailed WOE report for a feature."""
        if feature not in self.woe_maps_:
            raise ValueError(f"Feature '{feature}' not found")

        woe_map = self.woe_maps_[feature]
        records = []
        for key, values in woe_map.items():
            record = {"bin": key, **values}
            records.append(record)

        return pd.DataFrame(records)


class TargetEncoder(BaseEstimator, TransformerMixin):
    """
    Target Encoder with regularization and cross-validation.

    Encodes categorical features using the mean of the target variable,
    with smoothing to prevent overfitting on rare categories.

    Parameters
    ----------
    smoothing : float, default=1.0
        Smoothing factor for regularization. Higher values give more
        weight to the global mean.
    n_folds : int, default=5
        Number of folds for cross-validation encoding during fit_transform.
    handle_missing : str, default='global_mean'
        How to handle missing values: 'global_mean' uses overall mean,
        'separate' treats missing as a category.

    Attributes
    ----------
    encoding_maps_ : Dict[str, Dict]
        Mapping of category to encoded value for each feature.
    global_mean_ : float
        Global target mean for regularization.
    """

    def __init__(
        self,
        smoothing: float = 1.0,
        n_folds: int = 5,
        handle_missing: str = "global_mean",
    ):
        self.smoothing = smoothing
        self.n_folds = n_folds
        self.handle_missing = handle_missing
        self.encoding_maps_: Dict[str, Dict[Any, float]] = {}
        self.global_mean_: float = 0.5

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TargetEncoder":
        """
        Fit the target encoder.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with categorical columns.
        y : pd.Series
            Binary target.

        Returns
        -------
        self
        """
        X = pd.DataFrame(X).copy()
        y = pd.Series(y).values

        self.global_mean_ = np.mean(y)

        for col in X.columns:
            self._fit_column(X[col], y, col)

        return self

    def _fit_column(self, col_data: pd.Series, y: np.ndarray, col_name: str) -> None:
        """Fit encoding for a single column."""
        encoding_map = {}

        # Handle missing
        if self.handle_missing == "separate":
            col_data = col_data.fillna("__MISSING__")

        for category in col_data.unique():
            mask = col_data == category
            n = mask.sum()
            category_mean = np.mean(y[mask])

            # Bayesian smoothing
            smoothed = (
                category_mean * n + self.global_mean_ * self.smoothing
            ) / (n + self.smoothing)

            encoding_map[category] = smoothed

        self.encoding_maps_[col_name] = encoding_map

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply target encoding.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.

        Returns
        -------
        pd.DataFrame
            Target-encoded features.
        """
        X = pd.DataFrame(X).copy()
        result = pd.DataFrame(index=X.index)

        for col in X.columns:
            if col not in self.encoding_maps_:
                continue

            col_data = X[col].copy()
            if self.handle_missing == "separate":
                col_data = col_data.fillna("__MISSING__")

            encoding_map = self.encoding_maps_[col]
            result[col] = col_data.map(
                lambda x: encoding_map.get(x, self.global_mean_)
            )

            # Fill any remaining NaN with global mean
            result[col] = result[col].fillna(self.global_mean_)

        return result

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Fit and transform with cross-validation to prevent leakage.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Binary target.

        Returns
        -------
        pd.DataFrame
            Target-encoded features (CV-encoded for training).
        """
        X = pd.DataFrame(X).copy()
        y = pd.Series(y).values

        self.global_mean_ = np.mean(y)

        # Initialize result with global mean
        result = pd.DataFrame(
            data=np.full((len(X), len(X.columns)), self.global_mean_),
            columns=X.columns,
            index=X.index,
        )

        # Cross-validation encoding
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train = y[train_idx]

            for col in X.columns:
                col_data = X_train[col].copy()
                if self.handle_missing == "separate":
                    col_data = col_data.fillna("__MISSING__")

                # Compute encoding from training fold
                for category in col_data.unique():
                    mask = col_data == category
                    n = mask.sum()
                    category_mean = np.mean(y_train[mask])

                    smoothed = (
                        category_mean * n + self.global_mean_ * self.smoothing
                    ) / (n + self.smoothing)

                    # Apply to validation fold
                    val_col = X_val[col].copy()
                    if self.handle_missing == "separate":
                        val_col = val_col.fillna("__MISSING__")

                    val_mask = val_col == category
                    result.loc[val_mask.index[val_mask], col] = smoothed

        # Full fit for transform on new data
        self.fit(X, y)

        return result


class SmartEncoder(BaseEstimator, TransformerMixin):
    """
    Adaptive encoder that selects best strategy per feature.

    Automatically chooses between:
    - One-hot encoding for low cardinality
    - WOE encoding for medium cardinality (with target)
    - Target encoding for high cardinality

    Parameters
    ----------
    low_cardinality_threshold : int, default=5
        Maximum unique values for one-hot encoding.
    high_cardinality_threshold : int, default=50
        Minimum unique values for target encoding.
    woe_regularization : float, default=0.5
        Regularization for WOE encoder.
    target_smoothing : float, default=1.0
        Smoothing for target encoder.
    """

    def __init__(
        self,
        low_cardinality_threshold: int = 5,
        high_cardinality_threshold: int = 50,
        woe_regularization: float = 0.5,
        target_smoothing: float = 1.0,
    ):
        self.low_cardinality_threshold = low_cardinality_threshold
        self.high_cardinality_threshold = high_cardinality_threshold
        self.woe_regularization = woe_regularization
        self.target_smoothing = target_smoothing

        self._encoders: Dict[str, Any] = {}
        self._strategies: Dict[str, str] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SmartEncoder":
        """Fit appropriate encoder for each feature."""
        X = pd.DataFrame(X).copy()

        for col in X.columns:
            n_unique = X[col].nunique()

            if n_unique <= self.low_cardinality_threshold:
                # One-hot encoding
                encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                encoder.fit(X[[col]])
                self._encoders[col] = encoder
                self._strategies[col] = "onehot"

            elif n_unique <= self.high_cardinality_threshold:
                # WOE encoding
                encoder = WOEEncoder(regularization=self.woe_regularization)
                encoder.fit(X[[col]], y)
                self._encoders[col] = encoder
                self._strategies[col] = "woe"

            else:
                # Target encoding
                encoder = TargetEncoder(smoothing=self.target_smoothing)
                encoder.fit(X[[col]], y)
                self._encoders[col] = encoder
                self._strategies[col] = "target"

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted encoders."""
        X = pd.DataFrame(X).copy()
        result_frames = []

        for col in X.columns:
            if col not in self._encoders:
                continue

            encoder = self._encoders[col]
            strategy = self._strategies[col]

            if strategy == "onehot":
                encoded = encoder.transform(X[[col]])
                feature_names = encoder.get_feature_names_out([col])
                encoded_df = pd.DataFrame(
                    encoded, columns=feature_names, index=X.index
                )
                result_frames.append(encoded_df)

            else:
                encoded_df = encoder.transform(X[[col]])
                result_frames.append(encoded_df)

        if result_frames:
            return pd.concat(result_frames, axis=1)
        return pd.DataFrame(index=X.index)

    def get_encoding_summary(self) -> Dict[str, str]:
        """Get summary of encoding strategies used."""
        return self._strategies.copy()
