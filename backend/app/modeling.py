from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

import os


try:  # pragma: no cover - optional dependency
    from xgboost import XGBClassifier  # type: ignore

    HAS_XGBOOST = True
except Exception:  # pragma: no cover
    XGBClassifier = None
    HAS_XGBOOST = False

try:  # pragma: no cover - optional dependency
    from lightgbm import LGBMClassifier  # type: ignore

    HAS_LIGHTGBM = True
except Exception:  # pragma: no cover
    LGBMClassifier = None
    HAS_LIGHTGBM = False


ROC_MAX_POINTS = 200


@dataclass
class ModelMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float


@dataclass
class ModelResult:
    name: str
    metrics: Dict[str, ModelMetrics]
    confusion: Dict[str, List[List[int]]]
    roc_curve: Dict[str, Dict[str, List[float]]]
    feature_importance: List[Dict[str, float]]


@dataclass
class TrainingOutput:
    results: List[ModelResult]
    leaderboard: List[Dict[str, float]]


def _get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    if hasattr(preprocessor, "get_feature_names_out"):
        return list(preprocessor.get_feature_names_out())

    feature_names: List[str] = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == "remainder":
            continue
        if hasattr(transformer, "get_feature_names"):
            transformed = list(transformer.get_feature_names(cols))
        else:
            transformed = list(cols)
        feature_names.extend(transformed)
    return feature_names


def _make_encoder() -> OneHotEncoder:
    # High-cardinality text (pincodes, branch codes) would otherwise explode
    # into thousands of columns; rare levels are pooled into one bucket.
    return OneHotEncoder(
        handle_unknown="infrequent_if_exist", max_categories=40, sparse_output=False
    )


def _gpu_available() -> bool:
    """True when the process was given GPUs (CUDA_VISIBLE_DEVICES non-empty)."""
    devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if devices is None:
        return os.path.exists("/dev/nvidia0")
    return devices.strip() not in ("", "-1")


def _build_preprocessor(categorical_features: List[str], numeric_features: List[str]):
    encoder = _make_encoder()
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", encoder),
    ])
    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_pipeline, categorical_features))
    return ColumnTransformer(transformers=transformers)


def _make_models(seed: int = 42) -> Dict[str, object]:
    """
    Create all available ML models for training.

    Models included:
    - LogisticRegression: Interpretable baseline model
    - RandomForest: Robust ensemble model
    - GradientBoosting: Sklearn gradient boosting
    - XGBoost: High-performance gradient boosting (if installed)
    - LightGBM: Fast, memory-efficient gradient boosting (if installed)
    """
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=400,
            solver="lbfgs",
            random_state=seed,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=seed,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.1,
            max_depth=6,
            early_stopping=True,
            random_state=seed,
        ),
    }

    if HAS_XGBOOST and XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=seed,
            tree_method="hist",
            device="cuda" if _gpu_available() else "cpu",
            n_jobs=-1,
        )

    if HAS_LIGHTGBM and LGBMClassifier is not None:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=-1,
            num_leaves=31,
            min_child_samples=20,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )

    return models


def _extract_feature_importance(
    model: object, feature_names: List[str]
) -> List[Dict[str, float]]:
    if hasattr(model, "feature_importances_"):
        importances = getattr(model, "feature_importances_")
        ranking = np.argsort(np.abs(importances))[::-1]
        return [
            {"feature": feature_names[idx], "importance": float(importances[idx])}
            for idx in ranking[:15]
        ]

    if hasattr(model, "coef_"):
        coef = getattr(model, "coef_")[0]
        ranking = np.argsort(np.abs(coef))[::-1]
        return [
            {"feature": feature_names[idx], "importance": float(coef[idx])}
            for idx in ranking[:15]
        ]

    return []


def _evaluate_split(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
) -> Tuple[ModelMetrics, List[List[int]], Dict[str, List[float]]]:
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    metrics = ModelMetrics(
        accuracy=float(accuracy_score(y, pred)),
        precision=float(precision_score(y, pred, zero_division=0)),
        recall=float(recall_score(y, pred, zero_division=0)),
        f1=float(f1_score(y, pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y, proba)),
    )
    conf = confusion_matrix(y, pred).tolist()
    fpr, tpr, _ = roc_curve(y, proba)
    # On large splits the raw curve has one point per distinct score; keep the
    # payload (sent to the browser and back for reports) small.
    if len(fpr) > ROC_MAX_POINTS:
        keep = np.unique(np.linspace(0, len(fpr) - 1, ROC_MAX_POINTS).astype(int))
        fpr, tpr = fpr[keep], tpr[keep]
    roc_payload = {"fpr": [float(v) for v in fpr], "tpr": [float(v) for v in tpr]}
    return metrics, conf, roc_payload


def train_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    oot_df: pd.DataFrame,
    etrc_df: pd.DataFrame,
    target_col: str = "decision_binary",
    seed: int = 42,
) -> TrainingOutput:
    feature_cols = [
        col for col in train_df.columns if col not in {target_col, "decision"}
    ]
    # pandas 3 reads text as the "str" dtype, not "object", so test for numeric
    # and treat everything else (str, object, category, bool-like text) as categorical.
    numeric_features = [
        col for col in feature_cols if pd.api.types.is_numeric_dtype(train_df[col])
    ]
    categorical_features = [col for col in feature_cols if col not in numeric_features]

    results: List[ModelResult] = []
    leaderboard: List[Dict[str, float]] = []

    for name, model in _make_models(seed).items():
        preprocessor = _build_preprocessor(categorical_features, numeric_features)
        pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
        try:
            pipeline.fit(train_df[feature_cols], train_df[target_col])
        except Exception as exc:
            if name == "XGBoost" and getattr(model, "device", "cpu") == "cuda":
                # GPU not usable in this process (driver, memory): retry on CPU.
                model.set_params(device="cpu")
                pipeline.fit(train_df[feature_cols], train_df[target_col])
            else:
                raise exc

        feature_names = _get_feature_names(pipeline.named_steps["preprocess"])
        importance = _extract_feature_importance(
            pipeline.named_steps["model"], feature_names
        )

        metrics: Dict[str, ModelMetrics] = {}
        confusion: Dict[str, List[List[int]]] = {}
        roc_payload: Dict[str, Dict[str, List[float]]] = {}

        for split_name, split_df in {
            "train": train_df,
            "test": test_df,
            "oot": oot_df,
            "etrc": etrc_df,
        }.items():
            split_metrics, conf, roc_curve_payload = _evaluate_split(
                pipeline, split_df[feature_cols], split_df[target_col]
            )
            metrics[split_name] = split_metrics
            confusion[split_name] = conf
            roc_payload[split_name] = roc_curve_payload

        results.append(
            ModelResult(
                name=name,
                metrics=metrics,
                confusion=confusion,
                roc_curve=roc_payload,
                feature_importance=importance,
            )
        )

        leaderboard.append({"model": name, "roc_auc": metrics["test"].roc_auc})

    leaderboard = sorted(leaderboard, key=lambda x: x["roc_auc"], reverse=True)

    return TrainingOutput(results=results, leaderboard=leaderboard)
