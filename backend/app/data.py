from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class DatasetSplits:
    train: pd.DataFrame
    test: pd.DataFrame
    oot: pd.DataFrame
    etrc: pd.DataFrame


@dataclass
class DatasetSummary:
    n_rows: int
    n_features: int
    target_col: str
    class_balance: Dict[str, Dict[str, float]]
    feature_types: Dict[str, int]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def generate_synthetic_dataset(
    n_rows: int = 5000,
    n_features: int = 100,
    n_categorical: int = 10,
    seed: int = 42,
    decision_labels: Tuple[str, str] = ("default", "no_default"),
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    n_categorical = min(n_categorical, n_features)
    n_numeric = n_features - n_categorical

    numeric_data = rng.normal(loc=0, scale=1.0, size=(n_rows, n_numeric))
    df = pd.DataFrame(
        numeric_data, columns=[f"f_num_{idx:03d}" for idx in range(n_numeric)]
    )

    for idx in range(n_categorical):
        cardinality = rng.integers(3, 8)
        df[f"f_cat_{idx:03d}"] = rng.integers(0, cardinality, size=n_rows).astype(
            str
        )

    weights = rng.normal(size=n_numeric)
    base_score = numeric_data @ weights
    cat_effect = np.zeros(n_rows)
    for idx in range(n_categorical):
        cat_values = df[f"f_cat_{idx:03d}"].astype(int)
        cat_effect += (cat_values - cat_values.mean()) * rng.normal()

    noise = rng.normal(scale=0.5, size=n_rows)
    probability = _sigmoid(base_score * 0.2 + cat_effect * 0.05 + noise)
    target = (probability > np.quantile(probability, 0.65)).astype(int)

    label_positive, label_negative = decision_labels
    df["decision"] = np.where(target == 1, label_positive, label_negative)
    df["decision_binary"] = target

    return df


def split_dataset(
    df: pd.DataFrame,
    target_col: str = "decision_binary",
    seed: int = 42,
    train_size: float = 0.6,
    test_size: float = 0.2,
    oot_size: float = 0.1,
    etrc_size: float = 0.1,
) -> DatasetSplits:
    if not np.isclose(train_size + test_size + oot_size + etrc_size, 1.0):
        raise ValueError("Split sizes must sum to 1.0")

    train_df, temp_df = train_test_split(
        df,
        train_size=train_size,
        random_state=seed,
        stratify=df[target_col],
    )

    temp_ratio = 1 - train_size
    test_ratio = test_size / temp_ratio
    oot_ratio = oot_size / temp_ratio

    test_df, holdout_df = train_test_split(
        temp_df,
        train_size=test_ratio,
        random_state=seed,
        stratify=temp_df[target_col],
    )

    oot_df, etrc_df = train_test_split(
        holdout_df,
        train_size=oot_ratio / (oot_ratio + etrc_size / temp_ratio),
        random_state=seed,
        stratify=holdout_df[target_col],
    )

    return DatasetSplits(train=train_df, test=test_df, oot=oot_df, etrc=etrc_df)


def summarize_dataset(
    splits: DatasetSplits, target_col: str = "decision"
) -> DatasetSummary:
    full_df = pd.concat([splits.train, splits.test, splits.oot, splits.etrc], axis=0)
    feature_cols = [
        col for col in full_df.columns if col not in {target_col, "decision_binary"}
    ]

    feature_types = {
        "numeric": int(full_df[feature_cols].select_dtypes(include=[np.number]).shape[1]),
        "categorical": int(
            full_df[feature_cols].select_dtypes(exclude=[np.number]).shape[1]
        ),
    }

    class_balance: Dict[str, Dict[str, float]] = {}
    for split_name, split_df in {
        "train": splits.train,
        "test": splits.test,
        "oot": splits.oot,
        "etrc": splits.etrc,
    }.items():
        counts = split_df[target_col].value_counts(normalize=True).to_dict()
        class_balance[split_name] = {str(k): float(v) for k, v in counts.items()}

    return DatasetSummary(
        n_rows=int(full_df.shape[0]),
        n_features=len(feature_cols),
        target_col=target_col,
        class_balance=class_balance,
        feature_types=feature_types,
    )
