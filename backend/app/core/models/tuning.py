# =============================================================================
# ML on the Go - Hyperparameter Tuning with Optuna
# =============================================================================
# Bayesian optimization for model hyperparameters
# =============================================================================

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.model_selection import cross_val_score, StratifiedKFold

from .definitions import ModelType, get_model_definition, MODEL_REGISTRY


logger = logging.getLogger(__name__)


@dataclass
class TuningConfig:
    """Configuration for hyperparameter tuning."""

    n_trials: int = 50
    cv_folds: int = 5
    scoring: str = "roc_auc"
    direction: str = "maximize"
    timeout: Optional[int] = None  # Seconds
    n_jobs: int = -1
    random_state: int = 42
    show_progress_bar: bool = True
    early_stopping_rounds: int = 10
    min_improvement: float = 0.001


@dataclass
class TuningResult:
    """Result of hyperparameter tuning."""

    model_type: ModelType
    best_params: Dict[str, Any]
    best_score: float
    best_std: float
    cv_results: List[float]
    n_trials_completed: int
    study: Optional[optuna.Study] = None
    param_importances: Optional[Dict[str, float]] = None


class OptunaObjective:
    """Objective function for Optuna optimization."""

    def __init__(
        self,
        model_type: ModelType,
        X: np.ndarray,
        y: np.ndarray,
        config: TuningConfig,
    ):
        self.model_type = model_type
        self.X = X
        self.y = y
        self.config = config
        self.definition = get_model_definition(model_type)
        self.best_score = float("-inf") if config.direction == "maximize" else float("inf")
        self.no_improvement_count = 0

    def __call__(self, trial: optuna.Trial) -> float:
        """Evaluate a single trial."""
        params = self._suggest_params(trial)

        # Merge with default params
        full_params = {**self.definition.default_params, **params}

        # Handle solver-penalty compatibility for logistic regression
        if self.model_type == ModelType.LOGISTIC:
            full_params = self._fix_logistic_params(full_params)

        try:
            model = self.definition.model_class(**full_params)

            cv = StratifiedKFold(
                n_splits=self.config.cv_folds,
                shuffle=True,
                random_state=self.config.random_state,
            )

            scores = cross_val_score(
                model,
                self.X,
                self.y,
                cv=cv,
                scoring=self.config.scoring,
                n_jobs=self.config.n_jobs,
            )

            mean_score = np.mean(scores)

            # Check for improvement
            improved = (
                mean_score > self.best_score + self.config.min_improvement
                if self.config.direction == "maximize"
                else mean_score < self.best_score - self.config.min_improvement
            )

            if improved:
                self.best_score = mean_score
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1

            # Report intermediate value for pruning
            trial.report(mean_score, step=0)

            if trial.should_prune():
                raise optuna.TrialPruned()

            return mean_score

        except Exception as e:
            logger.warning(f"Trial failed with params {params}: {e}")
            raise optuna.TrialPruned()

    def _suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for a trial."""
        params = {}

        for param_name, param_config in self.definition.param_space.items():
            # Skip conditional parameters if condition not met
            if "condition" in param_config:
                if not self._check_condition(params, param_config["condition"]):
                    continue

            param_type = param_config["type"]

            if param_type == "int":
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                )
            elif param_type == "float":
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    log=param_config.get("log", False),
                )
            elif param_type == "categorical":
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    param_config["choices"],
                )
            elif param_type == "loguniform":
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    log=True,
                )

        return params

    def _check_condition(self, params: Dict[str, Any], condition: str) -> bool:
        """Check if a condition is met based on current params."""
        try:
            return eval(condition, {"__builtins__": {}}, params)
        except Exception:
            return False

    def _fix_logistic_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Fix logistic regression parameter compatibility."""
        params = params.copy()
        solver = params.get("solver", "lbfgs")
        penalty = params.get("penalty", "l2")

        # Handle solver-penalty compatibility
        if penalty == "l1" and solver not in ["liblinear", "saga"]:
            params["solver"] = "saga"
        elif penalty == "elasticnet" and solver != "saga":
            params["solver"] = "saga"
        elif penalty is None and solver == "liblinear":
            params["solver"] = "lbfgs"

        # Remove l1_ratio if not using elasticnet
        if penalty != "elasticnet" and "l1_ratio" in params:
            del params["l1_ratio"]

        return params


class HyperparameterTuner:
    """Hyperparameter tuning using Optuna."""

    def __init__(self, config: Optional[TuningConfig] = None):
        self.config = config or TuningConfig()

    def tune(
        self,
        model_type: ModelType,
        X: np.ndarray,
        y: np.ndarray,
        config: Optional[TuningConfig] = None,
    ) -> TuningResult:
        """Run hyperparameter tuning for a single model type."""
        config = config or self.config

        logger.info(f"Starting tuning for {model_type.value} with {config.n_trials} trials")

        # Create study
        sampler = TPESampler(seed=config.random_state)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=0)

        study = optuna.create_study(
            direction=config.direction,
            sampler=sampler,
            pruner=pruner,
        )

        # Create objective
        objective = OptunaObjective(model_type, X, y, config)

        # Early stopping callback
        def early_stopping_callback(study: optuna.Study, trial: optuna.FrozenTrial):
            if objective.no_improvement_count >= config.early_stopping_rounds:
                study.stop()

        # Run optimization
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study.optimize(
            objective,
            n_trials=config.n_trials,
            timeout=config.timeout,
            show_progress_bar=config.show_progress_bar,
            callbacks=[early_stopping_callback],
        )

        # Get best parameters
        best_params = study.best_params

        # Compute final CV score with best params
        definition = get_model_definition(model_type)
        full_params = {**definition.default_params, **best_params}

        if model_type == ModelType.LOGISTIC:
            full_params = objective._fix_logistic_params(full_params)

        model = definition.model_class(**full_params)
        cv = StratifiedKFold(
            n_splits=config.cv_folds,
            shuffle=True,
            random_state=config.random_state,
        )

        cv_scores = cross_val_score(
            model,
            X,
            y,
            cv=cv,
            scoring=config.scoring,
            n_jobs=config.n_jobs,
        )

        # Compute parameter importances
        try:
            param_importances = optuna.importance.get_param_importances(study)
        except Exception:
            param_importances = None

        logger.info(
            f"Tuning complete for {model_type.value}: "
            f"best_score={np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})"
        )

        return TuningResult(
            model_type=model_type,
            best_params=full_params,
            best_score=np.mean(cv_scores),
            best_std=np.std(cv_scores),
            cv_results=cv_scores.tolist(),
            n_trials_completed=len(study.trials),
            study=study,
            param_importances=param_importances,
        )

    def tune_multiple(
        self,
        model_types: List[ModelType],
        X: np.ndarray,
        y: np.ndarray,
        config: Optional[TuningConfig] = None,
    ) -> Dict[ModelType, TuningResult]:
        """Tune multiple model types and return results."""
        results = {}

        for model_type in model_types:
            try:
                results[model_type] = self.tune(model_type, X, y, config)
            except Exception as e:
                logger.error(f"Failed to tune {model_type.value}: {e}")

        return results


def quick_tune(
    model_type: ModelType,
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = 20,
    cv_folds: int = 3,
) -> TuningResult:
    """Quick tuning with reduced trials for rapid iteration."""
    config = TuningConfig(
        n_trials=n_trials,
        cv_folds=cv_folds,
        show_progress_bar=False,
    )
    tuner = HyperparameterTuner(config)
    return tuner.tune(model_type, X, y)
