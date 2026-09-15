"""
Model zoo and Optuna hyper-parameter search.

Each spec knows how to build an estimator from a parameter dict and how to
sample that dict from an Optuna trial. Tuning maximises stratified
cross-validated AUC on the training split with TPE sampling, median
pruning and an early-stop patience, under a per-model trial and time
budget set by the speed tier.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import optuna
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None
try:
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover
    CatBoostClassifier = None
try:
    from .transformer import FTTransformerClassifier, torch_available
except Exception:  # pragma: no cover
    FTTransformerClassifier = None

    def torch_available() -> bool:
        return False


def gpu_available() -> bool:
    devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if devices is None:
        return os.path.exists("/dev/nvidia0")
    return devices.strip() not in ("", "-1")


# Trials / seconds per model by speed tier.
TIERS = {
    "quick": {"trials": 4, "seconds": 40, "patience": 3, "folds": 2},
    "balanced": {"trials": 25, "seconds": 400, "patience": 8, "folds": 3},
    "thorough": {"trials": 60, "seconds": 1500, "patience": 15, "folds": 3},
}
N_JOBS = int(os.environ.get("JOB_THREADS", "16"))
TUNE_MAX_ROWS = 200_000
CV_FOLDS = 3


@dataclass
class ModelSpec:
    name: str
    family: str
    build: Callable[[Dict[str, Any], int], Any]
    space: Callable[[optuna.Trial], Dict[str, Any]]
    needs_scaling: bool = False
    max_rows: Optional[int] = None      # fit on a sample beyond this (slow learners)
    description: str = ""


def _logistic(p: Dict[str, Any], seed: int):
    return LogisticRegression(max_iter=2000, C=p.get("C", 1.0), class_weight=p.get("class_weight"),
                              solver="lbfgs", random_state=seed)


def _logistic_space(t: optuna.Trial):
    return {"C": t.suggest_float("C", 1e-3, 30, log=True),
            "class_weight": t.suggest_categorical("class_weight", [None, "balanced"])}


def _rf(p: Dict[str, Any], seed: int):
    return RandomForestClassifier(n_estimators=p.get("n_estimators", 300), max_depth=p.get("max_depth"),
                                  min_samples_leaf=p.get("min_samples_leaf", 2), max_features=p.get("max_features", "sqrt"),
                                  class_weight=p.get("class_weight"), n_jobs=N_JOBS, random_state=seed)


def _rf_space(t: optuna.Trial):
    return {"n_estimators": t.suggest_int("n_estimators", 150, 600, step=50),
            "max_depth": t.suggest_int("max_depth", 4, 24),
            "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 40, log=True),
            "max_features": t.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5]),
            "class_weight": t.suggest_categorical("class_weight", [None, "balanced_subsample"])}


def _et(p: Dict[str, Any], seed: int):
    return ExtraTreesClassifier(n_estimators=p.get("n_estimators", 300), max_depth=p.get("max_depth"),
                                min_samples_leaf=p.get("min_samples_leaf", 2), max_features=p.get("max_features", "sqrt"),
                                n_jobs=N_JOBS, random_state=seed)


def _hgb(p: Dict[str, Any], seed: int):
    return HistGradientBoostingClassifier(learning_rate=p.get("learning_rate", 0.08), max_iter=p.get("max_iter", 300),
                                          max_leaf_nodes=p.get("max_leaf_nodes", 31), min_samples_leaf=p.get("min_samples_leaf", 20),
                                          l2_regularization=p.get("l2_regularization", 0.0), early_stopping=True,
                                          validation_fraction=0.1, n_iter_no_change=20, random_state=seed)


def _hgb_space(t: optuna.Trial):
    return {"learning_rate": t.suggest_float("learning_rate", 0.02, 0.3, log=True),
            "max_iter": t.suggest_int("max_iter", 100, 800, step=50),
            "max_leaf_nodes": t.suggest_int("max_leaf_nodes", 8, 96, log=True),
            "min_samples_leaf": t.suggest_int("min_samples_leaf", 5, 200, log=True),
            "l2_regularization": t.suggest_float("l2_regularization", 1e-4, 10, log=True)}


def _xgb(p: Dict[str, Any], seed: int):
    return XGBClassifier(n_estimators=p.get("n_estimators", 400), learning_rate=p.get("learning_rate", 0.05),
                         max_depth=p.get("max_depth", 6), min_child_weight=p.get("min_child_weight", 1),
                         subsample=p.get("subsample", 0.8), colsample_bytree=p.get("colsample_bytree", 0.8),
                         reg_lambda=p.get("reg_lambda", 1.0), reg_alpha=p.get("reg_alpha", 0.0), gamma=p.get("gamma", 0.0),
                         tree_method="hist", device="cuda" if gpu_available() else "cpu", eval_metric="auc",
                         random_state=seed, n_jobs=N_JOBS, verbosity=0)


def _xgb_space(t: optuna.Trial):
    return {"n_estimators": t.suggest_int("n_estimators", 150, 1200, step=50),
            "learning_rate": t.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": t.suggest_int("max_depth", 3, 10),
            "min_child_weight": t.suggest_float("min_child_weight", 0.5, 30, log=True),
            "subsample": t.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": t.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 30, log=True),
            "reg_alpha": t.suggest_float("reg_alpha", 1e-3, 10, log=True),
            "gamma": t.suggest_float("gamma", 1e-3, 5, log=True)}


def _lgbm(p: Dict[str, Any], seed: int):
    return LGBMClassifier(n_estimators=p.get("n_estimators", 400), learning_rate=p.get("learning_rate", 0.05),
                          num_leaves=p.get("num_leaves", 31), min_child_samples=p.get("min_child_samples", 20),
                          subsample=p.get("subsample", 0.8), subsample_freq=1, colsample_bytree=p.get("colsample_bytree", 0.8),
                          reg_lambda=p.get("reg_lambda", 0.0), reg_alpha=p.get("reg_alpha", 0.0),
                          random_state=seed, n_jobs=N_JOBS, verbose=-1)


def _lgbm_space(t: optuna.Trial):
    return {"n_estimators": t.suggest_int("n_estimators", 150, 1200, step=50),
            "learning_rate": t.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": t.suggest_int("num_leaves", 8, 256, log=True),
            "min_child_samples": t.suggest_int("min_child_samples", 5, 300, log=True),
            "subsample": t.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": t.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 30, log=True),
            "reg_alpha": t.suggest_float("reg_alpha", 1e-3, 10, log=True)}


def _cat(p: Dict[str, Any], seed: int):
    return CatBoostClassifier(iterations=p.get("iterations", 500), learning_rate=p.get("learning_rate", 0.05),
                              depth=p.get("depth", 6), l2_leaf_reg=p.get("l2_leaf_reg", 3.0),
                              random_strength=p.get("random_strength", 1.0), bagging_temperature=p.get("bagging_temperature", 1.0),
                              border_count=128, random_seed=seed, verbose=False, thread_count=N_JOBS, allow_writing_files=False)


def _cat_space(t: optuna.Trial):
    return {"iterations": t.suggest_int("iterations", 200, 1200, step=100),
            "learning_rate": t.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "depth": t.suggest_int("depth", 4, 10),
            "l2_leaf_reg": t.suggest_float("l2_leaf_reg", 0.5, 30, log=True),
            "random_strength": t.suggest_float("random_strength", 0.1, 10, log=True),
            "bagging_temperature": t.suggest_float("bagging_temperature", 0.0, 3.0)}


def _mlp(p: Dict[str, Any], seed: int):
    return MLPClassifier(hidden_layer_sizes=p.get("hidden", (128, 64)), alpha=p.get("alpha", 1e-4),
                         learning_rate_init=p.get("lr", 1e-3), batch_size=256, max_iter=200,
                         early_stopping=True, n_iter_no_change=12, random_state=seed)


def _mlp_space(t: optuna.Trial):
    width = t.suggest_categorical("width", [64, 128, 256])
    depth = t.suggest_int("depth", 1, 3)
    return {"hidden": tuple([width] * depth), "alpha": t.suggest_float("alpha", 1e-6, 1e-1, log=True),
            "lr": t.suggest_float("lr", 1e-4, 1e-2, log=True)}


def _ft(p: Dict[str, Any], seed: int):
    return FTTransformerClassifier(d_model=p.get("d_model", 64), n_layers=p.get("n_layers", 3), n_heads=4,
                                   dropout=p.get("dropout", 0.1), lr=p.get("lr", 1e-3), weight_decay=p.get("weight_decay", 1e-5),
                                   epochs=p.get("epochs", 40), batch_size=512, random_state=seed)


def _ft_space(t: optuna.Trial):
    return {"d_model": t.suggest_categorical("d_model", [32, 64, 128]),
            "n_layers": t.suggest_int("n_layers", 1, 4),
            "dropout": t.suggest_float("dropout", 0.0, 0.3),
            "lr": t.suggest_float("lr", 1e-4, 5e-3, log=True),
            "weight_decay": t.suggest_float("weight_decay", 1e-6, 1e-3, log=True)}


def available_specs() -> List[ModelSpec]:
    specs = [
        ModelSpec("LogisticRegression", "logistic", _logistic, _logistic_space, needs_scaling=True,
                  description="Regularised logistic regression: the interpretable baseline every scorecard starts from."),
        ModelSpec("RandomForest", "forest", _rf, _rf_space, max_rows=600_000,
                  description="Bagged decision trees; robust to outliers and monotone transformations."),
        ModelSpec("ExtraTrees", "forest", _et, _rf_space, max_rows=600_000,
                  description="Extremely randomised trees; lower variance than a random forest on noisy data."),
        ModelSpec("HistGradientBoosting", "boosting", _hgb, _hgb_space,
                  description="scikit-learn histogram gradient boosting with native missing-value handling."),
    ]
    if XGBClassifier is not None:
        specs.append(ModelSpec("XGBoost", "boosting", _xgb, _xgb_space,
                               description="Gradient boosting with regularisation and GPU training."))
    if LGBMClassifier is not None:
        specs.append(ModelSpec("LightGBM", "boosting", _lgbm, _lgbm_space,
                               description="Leaf-wise gradient boosting; fast on wide, sparse data."))
    if CatBoostClassifier is not None:
        specs.append(ModelSpec("CatBoost", "boosting", _cat, _cat_space, max_rows=800_000,
                               description="Ordered boosting with strong defaults; resists overfitting on small data."))
    specs.append(ModelSpec("MLP", "neural", _mlp, _mlp_space, needs_scaling=True, max_rows=400_000,
                           description="Multi-layer perceptron on standardised features."))
    if FTTransformerClassifier is not None and torch_available():
        specs.append(ModelSpec("FTTransformer", "transformer", _ft, _ft_space, needs_scaling=True, max_rows=300_000,
                               description="Feature-Tokenizer Transformer (PyTorch): attention across tabular features."))
    return specs


# ---------------------------------------------------------------- tuning


def _sample(X: np.ndarray, y: np.ndarray, n: int, seed: int):
    if len(y) <= n:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=n, replace=False)
    return X[idx], y[idx]


def tune(spec: ModelSpec, X: np.ndarray, y: np.ndarray, tier: str, seed: int,
         progress: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """Return {"params", "cv_auc", "trials", "history"} for the best trial."""
    budget = TIERS.get(tier, TIERS["balanced"])
    X_t, y_t = _sample(X, y, TUNE_MAX_ROWS, seed)
    folds = StratifiedKFold(n_splits=budget.get("folds", CV_FOLDS), shuffle=True, random_state=seed)
    fold_idx = list(folds.split(X_t, y_t))
    history: List[Dict[str, Any]] = []
    state = {"best": -1.0, "since_best": 0}

    def objective(trial: optuna.Trial) -> float:
        params = spec.space(trial)
        scores = []
        for k, (tr, va) in enumerate(fold_idx):
            model = spec.build(params, seed)
            model.fit(X_t[tr], y_t[tr])
            p = model.predict_proba(X_t[va])[:, 1]
            scores.append(roc_auc_score(y_t[va], p))
            trial.report(float(np.mean(scores)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        value = trial.value if trial.value is not None else float("nan")
        history.append({"trial": trial.number, "cv_auc": None if np.isnan(value) else round(value, 5),
                        "state": trial.state.name.lower()})
        if trial.state == optuna.trial.TrialState.COMPLETE and value > state["best"] + 1e-5:
            state["best"], state["since_best"] = value, 0
        else:
            state["since_best"] += 1
        if progress:
            progress(f"{spec.name}: trial {trial.number + 1}/{budget['trials']} · best CV AUC {max(state['best'], 0):.4f}")
        if state["since_best"] >= budget["patience"]:
            study.stop()

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=min(5, budget["trials"] // 2 or 1)),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1))
    started = time.time()
    study.optimize(objective, n_trials=budget["trials"], timeout=budget["seconds"], callbacks=[callback],
                   gc_after_trial=True, catch=(Exception,))
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        # Everything failed or was pruned: fall back to library defaults.
        return {"params": {}, "cv_auc": None, "trials": len(study.trials), "history": history,
                "seconds": round(time.time() - started, 1)}
    # Rebuild the parameter dict through the space so derived values (e.g. MLP
    # hidden sizes) are reconstructed from the raw trial parameters.
    best = study.best_trial
    params = spec.space(optuna.trial.FixedTrial(best.params))
    return {"params": params, "cv_auc": round(float(best.value), 5), "trials": len(study.trials),
            "history": history, "seconds": round(time.time() - started, 1)}
