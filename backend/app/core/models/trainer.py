# =============================================================================
# ML on the Go - Model Training Orchestrator
# =============================================================================
# Training pipeline with cross-validation, tuning, and artifact management
# =============================================================================

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .definitions import ModelType, get_model_definition, create_model, get_available_models
from .tuning import HyperparameterTuner, TuningConfig, TuningResult
from ..evaluation.metrics import (
    compute_all_metrics,
    compute_ks_statistic,
    compute_gini,
    ModelMetrics,
)


logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    model_types: List[ModelType] = field(default_factory=lambda: [ModelType.LIGHTGBM])
    cv_folds: int = 5
    enable_tuning: bool = True
    tuning_trials: int = 50
    scoring: str = "roc_auc"
    random_state: int = 42
    save_artifacts: bool = True
    artifact_dir: str = "artifacts/models"
    early_stopping: bool = True
    compute_oof_predictions: bool = True


@dataclass
class TrainedModel:
    """Container for a trained model with metadata."""

    model_id: str
    model_type: ModelType
    model: Any
    params: Dict[str, Any]
    feature_names: List[str]
    training_timestamp: datetime
    training_duration: float
    cv_scores: List[float]
    metrics: Optional[ModelMetrics] = None
    tuning_result: Optional[TuningResult] = None


@dataclass
class TrainingResult:
    """Result of a training run."""

    experiment_id: str
    models: Dict[ModelType, TrainedModel]
    leaderboard: pd.DataFrame
    training_time: float
    config: TrainingConfig
    oof_predictions: Optional[Dict[ModelType, np.ndarray]] = None
    split_metrics: Optional[Dict[str, Dict[ModelType, ModelMetrics]]] = None


class ModelTrainer:
    """Orchestrates model training with tuning and evaluation."""

    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        self.tuner = HyperparameterTuner()

    def train(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        X_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_test: Optional[Union[np.ndarray, pd.Series]] = None,
        feature_names: Optional[List[str]] = None,
        config: Optional[TrainingConfig] = None,
    ) -> TrainingResult:
        """
        Train multiple models with optional hyperparameter tuning.

        Args:
            X_train: Training features
            y_train: Training target
            X_test: Optional test features for evaluation
            y_test: Optional test target
            feature_names: List of feature names
            config: Training configuration

        Returns:
            TrainingResult with trained models and metrics
        """
        config = config or self.config
        start_time = time.time()
        experiment_id = str(uuid.uuid4())[:8]

        logger.info(f"Starting training experiment {experiment_id}")
        logger.info(f"Models: {[m.value for m in config.model_types]}")
        logger.info(f"Training samples: {len(X_train)}, Features: {X_train.shape[1]}")

        # Convert to numpy if needed
        X_train_np = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        y_train_np = y_train.values if isinstance(y_train, pd.Series) else y_train

        if feature_names is None and isinstance(X_train, pd.DataFrame):
            feature_names = X_train.columns.tolist()
        elif feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X_train_np.shape[1])]

        # Train each model type
        trained_models: Dict[ModelType, TrainedModel] = {}
        oof_predictions: Dict[ModelType, np.ndarray] = {}

        for model_type in config.model_types:
            try:
                model_start = time.time()
                logger.info(f"Training {model_type.value}...")

                # Hyperparameter tuning
                if config.enable_tuning:
                    tuning_config = TuningConfig(
                        n_trials=config.tuning_trials,
                        cv_folds=config.cv_folds,
                        scoring=config.scoring,
                        random_state=config.random_state,
                    )
                    tuning_result = self.tuner.tune(
                        model_type, X_train_np, y_train_np, tuning_config
                    )
                    best_params = tuning_result.best_params
                    cv_scores = tuning_result.cv_results
                else:
                    tuning_result = None
                    best_params = get_model_definition(model_type).default_params
                    cv_scores = self._cross_validate(
                        model_type, best_params, X_train_np, y_train_np, config
                    )

                # Train final model on full training set
                model = create_model(model_type, **best_params)
                model.fit(X_train_np, y_train_np)

                # Compute out-of-fold predictions
                oof_proba = None
                if config.compute_oof_predictions:
                    oof_proba = self._compute_oof_predictions(
                        model_type, best_params, X_train_np, y_train_np, config
                    )
                    oof_predictions[model_type] = oof_proba

                # Compute metrics on test set if provided
                metrics = None
                if X_test is not None and y_test is not None:
                    X_test_np = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
                    y_test_np = y_test.values if isinstance(y_test, pd.Series) else y_test

                    y_pred = model.predict(X_test_np)
                    y_proba = model.predict_proba(X_test_np)[:, 1]
                    metrics = compute_all_metrics(y_test_np, y_pred, y_proba)

                model_duration = time.time() - model_start

                trained_model = TrainedModel(
                    model_id=f"{experiment_id}_{model_type.value}",
                    model_type=model_type,
                    model=model,
                    params=best_params,
                    feature_names=feature_names,
                    training_timestamp=datetime.now(),
                    training_duration=model_duration,
                    cv_scores=cv_scores,
                    metrics=metrics,
                    tuning_result=tuning_result,
                )

                trained_models[model_type] = trained_model

                logger.info(
                    f"{model_type.value} trained in {model_duration:.2f}s, "
                    f"CV AUC: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})"
                )

            except Exception as e:
                logger.error(f"Failed to train {model_type.value}: {e}")

        # Create leaderboard
        leaderboard = self._create_leaderboard(trained_models)

        # Save artifacts
        if config.save_artifacts:
            self._save_artifacts(trained_models, experiment_id, config.artifact_dir)

        total_time = time.time() - start_time
        logger.info(f"Training complete in {total_time:.2f}s")

        return TrainingResult(
            experiment_id=experiment_id,
            models=trained_models,
            leaderboard=leaderboard,
            training_time=total_time,
            config=config,
            oof_predictions=oof_predictions if oof_predictions else None,
        )

    def _cross_validate(
        self,
        model_type: ModelType,
        params: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        config: TrainingConfig,
    ) -> List[float]:
        """Perform cross-validation without tuning."""
        from sklearn.model_selection import cross_val_score

        model = create_model(model_type, **params)
        cv = StratifiedKFold(
            n_splits=config.cv_folds,
            shuffle=True,
            random_state=config.random_state,
        )

        scores = cross_val_score(model, X, y, cv=cv, scoring=config.scoring, n_jobs=-1)
        return scores.tolist()

    def _compute_oof_predictions(
        self,
        model_type: ModelType,
        params: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        config: TrainingConfig,
    ) -> np.ndarray:
        """Compute out-of-fold predictions for model stacking and analysis."""
        model = create_model(model_type, **params)
        cv = StratifiedKFold(
            n_splits=config.cv_folds,
            shuffle=True,
            random_state=config.random_state,
        )

        oof_proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
        return oof_proba

    def _create_leaderboard(
        self, trained_models: Dict[ModelType, TrainedModel]
    ) -> pd.DataFrame:
        """Create a leaderboard comparing all trained models."""
        rows = []

        for model_type, trained_model in trained_models.items():
            row = {
                "model": model_type.value,
                "model_id": trained_model.model_id,
                "cv_auc_mean": np.mean(trained_model.cv_scores),
                "cv_auc_std": np.std(trained_model.cv_scores),
                "training_time_s": trained_model.training_duration,
            }

            # Add test metrics if available
            if trained_model.metrics:
                row.update({
                    "test_auc": trained_model.metrics.auc_roc,
                    "test_ks": trained_model.metrics.ks_statistic,
                    "test_gini": trained_model.metrics.gini,
                    "test_precision": trained_model.metrics.precision,
                    "test_recall": trained_model.metrics.recall,
                    "test_f1": trained_model.metrics.f1,
                })

            rows.append(row)

        df = pd.DataFrame(rows)

        # Sort by CV AUC (descending)
        df = df.sort_values("cv_auc_mean", ascending=False).reset_index(drop=True)

        return df

    def _save_artifacts(
        self,
        trained_models: Dict[ModelType, TrainedModel],
        experiment_id: str,
        artifact_dir: str,
    ) -> None:
        """Save trained models and metadata."""
        artifact_path = Path(artifact_dir) / experiment_id
        artifact_path.mkdir(parents=True, exist_ok=True)

        for model_type, trained_model in trained_models.items():
            model_path = artifact_path / f"{model_type.value}_model.joblib"
            joblib.dump(trained_model.model, model_path)

            # Save metadata
            metadata = {
                "model_id": trained_model.model_id,
                "model_type": model_type.value,
                "params": trained_model.params,
                "feature_names": trained_model.feature_names,
                "training_timestamp": trained_model.training_timestamp.isoformat(),
                "training_duration": trained_model.training_duration,
                "cv_scores": trained_model.cv_scores,
            }

            if trained_model.metrics:
                metadata["test_metrics"] = {
                    "auc_roc": trained_model.metrics.auc_roc,
                    "ks_statistic": trained_model.metrics.ks_statistic,
                    "gini": trained_model.metrics.gini,
                    "precision": trained_model.metrics.precision,
                    "recall": trained_model.metrics.recall,
                    "f1": trained_model.metrics.f1,
                }

            metadata_path = artifact_path / f"{model_type.value}_metadata.joblib"
            joblib.dump(metadata, metadata_path)

        logger.info(f"Artifacts saved to {artifact_path}")

    def evaluate_on_splits(
        self,
        trained_models: Dict[ModelType, TrainedModel],
        splits: Dict[str, Tuple[np.ndarray, np.ndarray]],
    ) -> Dict[str, Dict[ModelType, ModelMetrics]]:
        """
        Evaluate trained models on multiple data splits.

        Args:
            trained_models: Dictionary of trained models
            splits: Dictionary of split name -> (X, y) tuples

        Returns:
            Dictionary of split_name -> model_type -> metrics
        """
        results: Dict[str, Dict[ModelType, ModelMetrics]] = {}

        for split_name, (X, y) in splits.items():
            logger.info(f"Evaluating on split: {split_name}")
            results[split_name] = {}

            X_np = X.values if isinstance(X, pd.DataFrame) else X
            y_np = y.values if isinstance(y, pd.Series) else y

            for model_type, trained_model in trained_models.items():
                y_pred = trained_model.model.predict(X_np)
                y_proba = trained_model.model.predict_proba(X_np)[:, 1]
                metrics = compute_all_metrics(y_np, y_pred, y_proba)
                results[split_name][model_type] = metrics

        return results


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: Optional[pd.DataFrame] = None,
    y_test: Optional[pd.Series] = None,
    model_types: Optional[List[str]] = None,
    enable_tuning: bool = True,
    tuning_trials: int = 50,
) -> TrainingResult:
    """
    Convenience function to train multiple models.

    Args:
        X_train: Training features
        y_train: Training target
        X_test: Optional test features
        y_test: Optional test target
        model_types: List of model type names (defaults to all)
        enable_tuning: Whether to enable hyperparameter tuning
        tuning_trials: Number of tuning trials per model

    Returns:
        TrainingResult with trained models
    """
    if model_types is None:
        types = [ModelType.LIGHTGBM, ModelType.XGBOOST, ModelType.LOGISTIC]
    else:
        types = [ModelType(t) for t in model_types]

    config = TrainingConfig(
        model_types=types,
        enable_tuning=enable_tuning,
        tuning_trials=tuning_trials,
    )

    trainer = ModelTrainer(config)
    return trainer.train(X_train, y_train, X_test, y_test)
