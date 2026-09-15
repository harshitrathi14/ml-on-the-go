"""
Feature preprocessing shared by every model in the zoo.

The transformer is fitted on the training split only. Design choices a
credit modeller would make by default:

* numeric: median imputation plus a missing-indicator column whenever a
  variable has gaps (missingness is often predictive in credit data)
* low-cardinality categoricals: one-hot with rare levels pooled
* high-cardinality categoricals (pincode, branch, dealer): out-of-fold
  target encoding with smoothing, so they neither explode into thousands
  of columns nor leak the target
* the raw column list, dtypes and typical values are recorded as the
  input schema used later to score new applications
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ONE_HOT_MAX_LEVELS = 25


class TargetEncoder(BaseEstimator, TransformerMixin):
    """Smoothed target-mean encoding with out-of-fold fitting on train."""

    def __init__(self, smoothing: float = 20.0, n_splits: int = 5, random_state: int = 42):
        self.smoothing = smoothing
        self.n_splits = n_splits
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y=None) -> "TargetEncoder":
        X = pd.DataFrame(X)
        y = np.asarray(y, dtype=float)
        self.prior_ = float(y.mean())
        self.maps_: Dict[Any, Dict[str, float]] = {}
        for col in X.columns:
            self.maps_[col] = self._encode_map(X[col].astype("string").fillna("<missing>"), y)
        self.columns_ = list(X.columns)
        return self

    def fit_transform(self, X: pd.DataFrame, y=None) -> np.ndarray:
        X = pd.DataFrame(X)
        y = np.asarray(y, dtype=float)
        self.fit(X, y)
        out = np.empty((len(X), len(self.columns_)), dtype=float)
        # Out-of-fold values for the training rows: each row's encoding never
        # sees its own target.
        folds = KFold(n_splits=min(self.n_splits, max(len(X) // 50, 2)), shuffle=True,
                      random_state=self.random_state)
        for j, col in enumerate(self.columns_):
            values = X[col].astype("string").fillna("<missing>")
            col_out = np.full(len(X), self.prior_)
            for train_idx, hold_idx in folds.split(X):
                fold_map = self._encode_map(values.iloc[train_idx], y[train_idx])
                col_out[hold_idx] = values.iloc[hold_idx].map(fold_map).fillna(self.prior_).to_numpy(dtype=float)
            out[:, j] = col_out
        return out

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X = pd.DataFrame(X)
        out = np.empty((len(X), len(self.columns_)), dtype=float)
        for j, col in enumerate(self.columns_):
            values = X[col].astype("string").fillna("<missing>")
            out[:, j] = values.map(self.maps_[col]).fillna(self.prior_).to_numpy(dtype=float)
        return out

    def _encode_map(self, values: pd.Series, y: np.ndarray) -> Dict[str, float]:
        frame = pd.DataFrame({"v": values.to_numpy(), "y": y})
        grouped = frame.groupby("v")["y"].agg(["sum", "count"])
        smoothed = (grouped["sum"] + self.prior_ * self.smoothing) / (grouped["count"] + self.smoothing)
        return smoothed.to_dict()

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray([f"{c}__te" for c in self.columns_], dtype=object)


@dataclass
class InputSchema:
    """What a scoring request must look like, learnt from the training table."""

    numeric: List[str] = field(default_factory=list)
    categorical: List[str] = field(default_factory=list)
    medians: Dict[str, float] = field(default_factory=dict)
    ranges: Dict[str, List[float]] = field(default_factory=dict)
    categories: Dict[str, List[str]] = field(default_factory=dict)
    modes: Dict[str, str] = field(default_factory=dict)

    @property
    def columns(self) -> List[str]:
        return self.numeric + self.categorical

    def to_dict(self) -> Dict[str, Any]:
        return {
            "numeric": self.numeric, "categorical": self.categorical, "medians": self.medians,
            "ranges": self.ranges, "categories": self.categories, "modes": self.modes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InputSchema":
        return cls(**{k: data.get(k, [] if k in ("numeric", "categorical") else {}) for k in
                      ("numeric", "categorical", "medians", "ranges", "categories", "modes")})

    def coerce(self, records: pd.DataFrame) -> pd.DataFrame:
        """Shape arbitrary scoring rows into the training layout."""
        out = pd.DataFrame(index=records.index)
        for col in self.numeric:
            series = records[col] if col in records.columns else pd.Series(np.nan, index=records.index)
            if series.dtype == object or pd.api.types.is_string_dtype(series):
                series = series.astype(str).str.replace(r"[,\s₹$%]", "", regex=True)
            out[col] = pd.to_numeric(series, errors="coerce")
        for col in self.categorical:
            series = records[col] if col in records.columns else pd.Series(np.nan, index=records.index)
            out[col] = series.astype("string").str.strip()
        return out


def split_columns(df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, List[str]]:
    numeric, low_card, high_card = [], [], []
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            numeric.append(col)
        elif df[col].nunique(dropna=True) <= ONE_HOT_MAX_LEVELS:
            low_card.append(col)
        else:
            high_card.append(col)
    return {"numeric": numeric, "low_card": low_card, "high_card": high_card}


def build_preprocessor(groups: Dict[str, List[str]], scale: bool = True, seed: int = 42) -> ColumnTransformer:
    transformers = []
    if groups["numeric"]:
        steps = [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
        if scale:
            steps.append(("scaler", StandardScaler()))
        transformers.append(("num", Pipeline(steps), groups["numeric"]))
    if groups["low_card"]:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="<missing>")),
            ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01,
                                     max_categories=ONE_HOT_MAX_LEVELS, sparse_output=False)),
        ]), groups["low_card"]))
    if groups["high_card"]:
        transformers.append(("te", TargetEncoder(random_state=seed), groups["high_card"]))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=True)


def feature_names(preprocessor: ColumnTransformer) -> List[str]:
    try:
        names = list(preprocessor.get_feature_names_out())
    except Exception:  # pragma: no cover - older sklearn paths
        names = [f"f{i}" for i in range(preprocessor.transform(pd.DataFrame()).shape[1])]
    return [n.replace("num__", "").replace("cat__", "").replace("te__", "") for n in names]


def learn_schema(df: pd.DataFrame, groups: Dict[str, List[str]]) -> InputSchema:
    schema = InputSchema(numeric=list(groups["numeric"]),
                         categorical=list(groups["low_card"] + groups["high_card"]))
    for col in schema.numeric:
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().any():
            schema.medians[col] = float(series.median())
            schema.ranges[col] = [float(series.quantile(0.01)), float(series.quantile(0.99))]
    for col in schema.categorical:
        counts = df[col].astype("string").value_counts().head(60)
        schema.categories[col] = [str(v) for v in counts.index]
        if len(counts):
            schema.modes[col] = str(counts.index[0])
    return schema


def prepare_frame(df: pd.DataFrame, schema: InputSchema) -> pd.DataFrame:
    """Cast a frame (training or scoring) into the dtypes the pipeline expects."""
    out = df.copy()
    for col in schema.numeric:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
    for col in schema.categorical:
        out[col] = out[col].astype("string").astype(object).where(out[col].notna(), None)
    return out[schema.columns]
