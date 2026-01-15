# =============================================================================
# ML on the Go - Model Definitions
# =============================================================================
# Defines available models and their default configurations
# =============================================================================

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type
from enum import Enum

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


class ModelType(str, Enum):
    """Supported model types."""
    LOGISTIC = "logistic"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"


@dataclass
class ModelDefinition:
    """Definition of a model including its class and hyperparameter search space."""

    model_type: ModelType
    model_class: Type
    default_params: Dict[str, Any]
    param_space: Dict[str, Any]  # For Optuna
    description: str

    def create_model(self, **kwargs) -> Any:
        """Create a model instance with given parameters."""
        params = {**self.default_params, **kwargs}
        return self.model_class(**params)


# -----------------------------------------------------------------------------
# Model Registry
# -----------------------------------------------------------------------------

MODEL_REGISTRY: Dict[ModelType, ModelDefinition] = {
    ModelType.LOGISTIC: ModelDefinition(
        model_type=ModelType.LOGISTIC,
        model_class=LogisticRegression,
        default_params={
            "max_iter": 1000,
            "solver": "lbfgs",
            "random_state": 42,
            "n_jobs": -1,
        },
        param_space={
            "C": {"type": "float", "low": 0.001, "high": 100.0, "log": True},
            "penalty": {"type": "categorical", "choices": ["l1", "l2", "elasticnet", None]},
            "solver": {"type": "categorical", "choices": ["lbfgs", "liblinear", "saga"]},
            "l1_ratio": {"type": "float", "low": 0.0, "high": 1.0, "condition": "penalty == 'elasticnet'"},
        },
        description="Regularized logistic regression for interpretable credit scoring",
    ),

    ModelType.RANDOM_FOREST: ModelDefinition(
        model_type=ModelType.RANDOM_FOREST,
        model_class=RandomForestClassifier,
        default_params={
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": 42,
            "n_jobs": -1,
        },
        param_space={
            "n_estimators": {"type": "int", "low": 50, "high": 500},
            "max_depth": {"type": "int", "low": 3, "high": 20},
            "min_samples_split": {"type": "int", "low": 2, "high": 20},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
            "max_features": {"type": "categorical", "choices": ["sqrt", "log2", None]},
            "criterion": {"type": "categorical", "choices": ["gini", "entropy"]},
        },
        description="Random forest ensemble for robust predictions",
    ),

    ModelType.GRADIENT_BOOSTING: ModelDefinition(
        model_type=ModelType.GRADIENT_BOOSTING,
        model_class=GradientBoostingClassifier,
        default_params={
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 5,
            "random_state": 42,
        },
        param_space={
            "n_estimators": {"type": "int", "low": 50, "high": 500},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "max_depth": {"type": "int", "low": 2, "high": 10},
            "min_samples_split": {"type": "int", "low": 2, "high": 20},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
            "subsample": {"type": "float", "low": 0.6, "high": 1.0},
        },
        description="Gradient boosting for high-performance predictions",
    ),

    ModelType.XGBOOST: ModelDefinition(
        model_type=ModelType.XGBOOST,
        model_class=XGBClassifier,
        default_params={
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 6,
            "eval_metric": "auc",
            "use_label_encoder": False,
            "random_state": 42,
            "n_jobs": -1,
        },
        param_space={
            "n_estimators": {"type": "int", "low": 50, "high": 500},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "max_depth": {"type": "int", "low": 2, "high": 12},
            "min_child_weight": {"type": "int", "low": 1, "high": 10},
            "subsample": {"type": "float", "low": 0.5, "high": 1.0},
            "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0},
            "reg_alpha": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
            "reg_lambda": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
            "gamma": {"type": "float", "low": 0.0, "high": 5.0},
        },
        description="XGBoost for state-of-the-art gradient boosting",
    ),

    ModelType.LIGHTGBM: ModelDefinition(
        model_type=ModelType.LIGHTGBM,
        model_class=LGBMClassifier,
        default_params={
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": -1,
            "num_leaves": 31,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        },
        param_space={
            "n_estimators": {"type": "int", "low": 50, "high": 500},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "num_leaves": {"type": "int", "low": 15, "high": 127},
            "max_depth": {"type": "int", "low": -1, "high": 15},
            "min_child_samples": {"type": "int", "low": 5, "high": 100},
            "subsample": {"type": "float", "low": 0.5, "high": 1.0},
            "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0},
            "reg_alpha": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
            "reg_lambda": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
        },
        description="LightGBM for fast and memory-efficient gradient boosting",
    ),
}


def get_model_definition(model_type: ModelType) -> ModelDefinition:
    """Get model definition by type."""
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_type]


def get_available_models() -> List[ModelType]:
    """Get list of available model types."""
    return list(MODEL_REGISTRY.keys())


def create_model(model_type: ModelType, **kwargs) -> Any:
    """Create a model instance."""
    definition = get_model_definition(model_type)
    return definition.create_model(**kwargs)
