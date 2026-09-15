"""
Training orchestrator: modelling table -> tuned zoo -> champion -> Nu Score.

Cohorts (see data.split_cohorts): train / validation / calibration / test / oot.
 1. splits: chronological when a time column exists (rows sharing a date
    are never divided), otherwise stratified random with the OOT labelled
    as "not a true out-of-time"
 2. preprocessing fitted on train (imputation + indicators, one-hot, OOF
    target encoding); input schema learnt for scoring
 3. every model family: Optuna tuning on train CV, refit on full train
 4. evaluation on all five cohorts; champion chosen by a composite score that
    uses ONLY train / validation / calibration (test and OOT never influence
    selection)
 5. stacked blend of the best models (logistic meta-learner on OOF
    predictions) as the "AI-enhanced" probability
 6. Platt calibration and Nu Score bands fitted on the calibration cohort,
    evaluated on test / OOT
 7. SHAP on the champion, permutation importance where a model has none,
    PSI drift per cohort, bootstrap AUC interval, data checks and warnings
 8. model bundle saved for scoring; rows without an outcome are scored
    ("scoring only") and never enter any cohort
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from ..data import COHORTS, split_cohorts
from ..profiling import compute_iv
from . import diagnostics, drift, evaluate, glossary, nuscore, zoo
from .artifact import ModelBundle
from .explain import Explainer
from .preprocess import build_preprocessor, feature_names, learn_schema, prepare_frame, split_columns

logger = logging.getLogger(__name__)

MAX_TRAIN_ROWS = 1_000_000
IV_ROWS = 300_000
STACK_TOP_K = 4
PERMUTATION_ROWS = 600
PERMUTATION_FEATURES = 80
Progress = Callable[[str, Optional[Dict[str, Any]]], None]


def _noop(message: str, extra: Optional[Dict[str, Any]] = None) -> None:
    pass


def _sample_rows(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    return df if len(df) <= n else df.sample(n=n, random_state=seed)


def run(
    df: pd.DataFrame,
    target_col: str,
    time_col: Optional[str],
    tier: str = "balanced",
    seed: int = 42,
    artifact_dir: Optional[Path] = None,
    progress: Progress = _noop,
    summary_target: Optional[str] = None,
    extra_dataset: Optional[Dict[str, Any]] = None,
    unlabelled: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    started = time.time()
    summary_target = summary_target or target_col

    # ---------------------------------------------------------------- splits
    progress("Splitting data into train / validation / calibration / test / out-of-time")
    cohorts, split_info = split_cohorts(df, target_col=target_col, time_col=time_col, seed=seed)
    dataset: Dict[str, Any] = {"split_strategy": split_info["strategy"], "split": split_info, "tier": tier}
    if "time_ranges" in split_info:
        dataset["time_ranges"] = split_info["time_ranges"]

    drop = {target_col, summary_target, time_col} - {None}
    feature_cols = [c for c in df.columns if c not in drop]
    train_full = cohorts["train"]
    train_df = _sample_rows(train_full, MAX_TRAIN_ROWS, seed)
    if len(train_df) < len(train_full):
        dataset["train_rows_used"] = len(train_df)
    frames_raw = {"train": train_df, **{n: cohorts[n] for n in COHORTS if n != "train"}}
    dataset.update(_summary(frames_raw, feature_cols, target_col, summary_target))

    # ---------------------------------------------------------- preprocess
    progress("Fitting preprocessing (imputation, encodings) on the training cohort")
    groups = split_columns(train_df, feature_cols)
    schema = learn_schema(train_df, groups)
    frames = {name: prepare_frame(part, schema) for name, part in frames_raw.items()}
    ys = {name: part[target_col].to_numpy(dtype=int) for name, part in frames_raw.items()}

    pre_scaled = build_preprocessor(groups, scale=True, seed=seed)
    pre_raw = build_preprocessor(groups, scale=False, seed=seed)
    X_scaled = {"train": np.asarray(pre_scaled.fit_transform(frames["train"], ys["train"]), dtype=float)}
    X_raw = {"train": np.asarray(pre_raw.fit_transform(frames["train"], ys["train"]), dtype=float)}
    for name in COHORTS[1:]:
        X_scaled[name] = np.asarray(pre_scaled.transform(frames[name]), dtype=float)
        X_raw[name] = np.asarray(pre_raw.transform(frames[name]), dtype=float)
    names = feature_names(pre_raw)

    iv_frame = _sample_rows(train_df, IV_ROWS, seed)
    dataset["feature_iv"] = compute_iv(iv_frame[feature_cols], iv_frame[target_col])
    dataset["feature_glossary"] = glossary.build(feature_cols)

    # ------------------------------------------------------------------ zoo
    specs = zoo.available_specs()
    models: List[Dict[str, Any]] = []
    fitted: Dict[str, Any] = {}
    for i, spec in enumerate(specs):
        X = X_scaled if spec.needs_scaling else X_raw
        Xtr, ytr = X["train"], ys["train"]
        if spec.max_rows and len(ytr) > spec.max_rows:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(ytr), size=spec.max_rows, replace=False)
            Xtr, ytr = Xtr[idx], ytr[idx]
        progress(f"Tuning {spec.name} ({i + 1}/{len(specs)})", {"model": spec.name, "stage": "tuning"})
        try:
            tuned = zoo.tune(spec, Xtr, ytr, tier, seed,
                             progress=lambda msg: progress(msg, {"model": spec.name, "stage": "tuning"}))
            progress(f"Fitting {spec.name} with its best parameters", {"model": spec.name, "stage": "fitting"})
            t0 = time.time()
            model = spec.build(tuned["params"], seed)
            model.fit(Xtr, ytr)
            fit_seconds = time.time() - t0
        except Exception as exc:
            logger.exception("Model %s failed", spec.name)
            models.append({"name": spec.name, "family": spec.family, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            continue

        probs = {name: model.predict_proba(X[name])[:, 1] for name in COHORTS}
        threshold = evaluate.best_threshold(ys["validation"], probs["validation"])
        evals = {name: evaluate.evaluate_split(ys[name], probs[name], threshold) for name in COHORTS}
        importance = _importance(model, names)
        method = "native"
        if not importance:
            progress(f"Measuring permutation importance for {spec.name}", {"model": spec.name, "stage": "fitting"})
            importance = _permutation_importance(model, X["validation"], ys["validation"], names, seed)
            method = "permutation (validation AUC drop)"
        entry = {
            "name": spec.name, "family": spec.family, "status": "ok", "description": spec.description,
            "tuning": {k: tuned[k] for k in ("cv_auc", "trials", "seconds", "history")},
            "params": _jsonable(tuned["params"]),
            "fit_seconds": round(fit_seconds, 2),
            "evaluations": evals,
            "stability": evaluate.stability(evals, probs),
            "feature_importance": importance, "importance_method": method,
            "_probs": probs, "_scaled": spec.needs_scaling,
        }
        models.append(entry)
        fitted[spec.name] = model

    ok = [m for m in models if m["status"] == "ok"]
    if not ok:
        raise RuntimeError("Every model failed to train; see the job log.")

    # ------------------------------------------------------------ champion
    for m in ok:
        m["composite"] = evaluate.composite_score(m, ok)
    ok.sort(key=lambda m: m["composite"]["score"], reverse=True)
    champion = ok[0]
    progress(f"Champion: {champion['name']}. Building the AI-enhanced blend", {"stage": "blend"})

    # ----------------------------------------------------------- stacking
    blend, blend_probs = _stack(ok, fitted, X_scaled, X_raw, ys, seed)
    if blend is not None:
        threshold = evaluate.best_threshold(ys["validation"], blend_probs["validation"])
        evals = {name: evaluate.evaluate_split(ys[name], blend_probs[name], threshold) for name in COHORTS}
        blend_entry = {
            "name": "NuScore Blend", "family": "ensemble", "status": "ok",
            "description": f"Stacked blend of {', '.join(blend['members'])} via a logistic meta-learner on out-of-fold predictions.",
            "tuning": {"cv_auc": None, "trials": 0, "seconds": 0, "history": []},
            "params": {"members": blend["members"], "weights": [round(w, 4) for w in blend["weights"]]},
            "fit_seconds": blend["seconds"], "evaluations": evals,
            "stability": evaluate.stability(evals, blend_probs),
            "feature_importance": champion["feature_importance"], "importance_method": champion["importance_method"],
            "_probs": blend_probs, "_scaled": champion["_scaled"],
        }
        blend_entry["composite"] = evaluate.composite_score(blend_entry, ok + [blend_entry])
        models.append(blend_entry)
    final_probs = blend_probs if blend_probs is not None else champion["_probs"]

    # -------------------------------------------------------- Nu Score
    progress("Calibrating on the calibration cohort and learning Nu Score bands", {"stage": "nuscore"})
    calibrator = nuscore.Calibrator().fit(final_probs["calibration"], ys["calibration"])
    cal = {name: calibrator(final_probs[name]) for name in COHORTS}
    scores = {name: nuscore.to_score(cal[name]) for name in COHORTS}
    bands = nuscore.fit_bands(scores["calibration"], ys["calibration"])
    base_cal_test = calibrator(champion["_probs"]["test"])
    test_after = evaluate.evaluate_split(ys["test"], cal["test"], evaluate.best_threshold(ys["validation"], cal["validation"]))
    calibration_effect = {
        "brier_before": final_probs and evaluate.evaluate_split(ys["test"], final_probs["test"], 0.5)["metrics"]["brier"],
        "brier_after": test_after["metrics"]["brier"],
        "ece_before": evaluate.expected_calibration_error(ys["test"], np.clip(final_probs["test"], 1e-6, 1 - 1e-6)),
        "ece_after": test_after["metrics"]["ece"],
    }
    nu = {
        "scale": {"min": 0, "max": 1000, "anchor": nuscore.ANCHOR, "pdo": nuscore.PDO, "higher_is_safer": True},
        "bands": bands,
        "bands_fitted_on": "calibration",
        "band_table": {name: nuscore.band_table(scores[name], ys[name], bands, reference=scores["calibration"])
                       for name in ("calibration", "test", "oot")},
        "cutoff_curve": nuscore.cutoff_curve(scores["test"], ys["test"]),
        "histogram": nuscore.histogram(scores["test"], ys["test"]),
        "calibration": {name: evaluate.reliability_curve(ys[name], cal[name]) for name in ("test", "oot")},
        "reliability_before": evaluate.reliability_curve(ys["test"], np.clip(final_probs["test"], 1e-6, 1 - 1e-6)),
        "ece_after_calibration": {name: evaluate.expected_calibration_error(ys[name], cal[name]) for name in ("test", "oot")},
        "calibration_effect": calibration_effect,
        "test_after_calibration": test_after["metrics"],
        "base_model": champion["name"],
        "ai_adjustment": {
            "mean_abs_points": float(np.mean(np.abs(scores["test"] - nuscore.to_score(base_cal_test)))),
            "auc_base": champion["evaluations"]["test"]["metrics"]["roc_auc"],
            "auc_enhanced": evaluate.evaluate_split(ys["test"], final_probs["test"], 0.5)["metrics"]["roc_auc"],
        },
    }

    # ---------------------------------------------------- diagnostics
    progress("Running diagnostics: bootstrap interval, data checks, drift", {"stage": "shap"})
    checks = diagnostics.data_checks(train_df, frames_raw["test"], feature_cols)
    cohort_sizes = {n: int(len(frames_raw[n])) for n in COHORTS}
    cohort_positives = {n: int(ys[n].sum()) for n in COHORTS}
    diag = {
        **checks,
        "auc_bootstrap_95_interval": diagnostics.bootstrap_auc(ys["test"], final_probs["test"], seed),
        "bootstrap_note": "Row bootstrap on the fixed test predictor: conditional uncertainty, not a correction for selection, repeated borrowers or drift.",
        "cohort_sizes": cohort_sizes, "cohort_positives": cohort_positives,
    }
    profiles = drift.fit_profiles(frames["train"])
    score_baseline = drift.fit_distribution(scores["train"])
    drift_report = {
        name: drift.summarise(drift.feature_drift(profiles, frames[name]), drift.compare(score_baseline, scores[name]))
        for name in COHORTS[1:]
    }
    champion_for_warnings = dict(champion)
    warnings = diagnostics.warnings_for(champion_for_warnings, cohort_sizes, cohort_positives, split_info, checks, calibration_effect)

    # ------------------------------------------------------------- SHAP
    progress(f"Explaining {champion['name']} with SHAP", {"stage": "shap"})
    X_ch = X_scaled if champion["_scaled"] else X_raw
    shap_summary: Dict[str, Any] = {}
    try:
        explainer = Explainer(fitted[champion["name"]], X_ch["train"], names, seed=seed)
        shap_summary = explainer.global_summary(X_ch["test"], seed=seed, display=X_raw["test"])
    except Exception as exc:
        logger.exception("SHAP failed")
        shap_summary = {"error": f"{type(exc).__name__}: {exc}"}

    # ----------------------------------------------------------- bundle
    model_id = None
    unlabelled_scoring = None
    if artifact_dir is not None:
        progress("Saving the model bundle for scoring", {"stage": "save"})
        rng = np.random.default_rng(seed)
        bg = X_ch["train"][rng.choice(len(X_ch["train"]), size=min(500, len(X_ch["train"])), replace=False)]
        bundle_blend = None
        if blend is not None:
            scaled_by_name = {m["name"]: m["_scaled"] for m in ok}
            members = []
            for n in blend["members"]:
                if n == champion["name"]:
                    members.append((None, False))
                else:
                    members.append((fitted[n], scaled_by_name[n] != champion["_scaled"]))
            bundle_blend = {"members": members, "weights": blend["weights"], "intercept": blend["intercept"],
                            "names": blend["members"]}
        bundle = ModelBundle(pre_scaled if champion["_scaled"] else pre_raw, fitted[champion["name"]], calibrator,
                             schema, names, bands, bg,
                             meta={"champion": champion["name"], "target_col": summary_target,
                                   "test_auc": champion["evaluations"]["test"]["metrics"]["roc_auc"],
                                   "blend_members": blend["members"] if blend else [],
                                   "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "n_train": int(len(train_df)),
                                   "drift_profiles": profiles, "score_baseline": score_baseline,
                                   "glossary": dataset["feature_glossary"]},
                             blend=bundle_blend,
                             alt_preprocessor=pre_raw if champion["_scaled"] else pre_scaled)
        bundle.save(Path(artifact_dir) / "model")
        model_id = Path(artifact_dir).name

        if unlabelled is not None and len(unlabelled):
            progress(f"Scoring {len(unlabelled):,} rows without an outcome", {"stage": "save"})
            unlabelled_scoring = _score_unlabelled(bundle, unlabelled, feature_cols, Path(artifact_dir))

    # ----------------------------------------------------------- output
    results = []
    for m in models:
        if m["status"] != "ok":
            results.append({"name": m["name"], "family": m["family"], "status": "failed", "error": m["error"]})
            continue
        results.append({k: v for k, v in m.items() if not k.startswith("_")} | {
            # Compatibility with the report generator and older clients.
            "metrics": {s: _legacy_metrics(m["evaluations"][s]["metrics"]) for s in m["evaluations"]},
            "confusion": {s: m["evaluations"][s]["confusion"] for s in m["evaluations"]},
            "roc_curve": {s: m["evaluations"][s]["roc_curve"] for s in m["evaluations"]},
        })
    ranked = sorted([r for r in results if r["status"] == "ok"], key=lambda r: r["composite"]["score"], reverse=True)
    leaderboard = [{"model": r["name"], "roc_auc": r["evaluations"]["test"]["metrics"]["roc_auc"],
                    "validation_auc": r["evaluations"]["validation"]["metrics"]["roc_auc"],
                    "composite": r["composite"]["score"], "ks": r["evaluations"]["test"]["metrics"]["ks"],
                    "oot_auc": r["evaluations"]["oot"]["metrics"]["roc_auc"], "family": r["family"]} for r in ranked]

    dataset.update(extra_dataset or {})
    dataset["n_models"] = len(ranked)
    dataset["training_seconds"] = round(time.time() - started, 1)
    dataset["cohort_sizes"] = cohort_sizes
    return {
        "dataset": dataset,
        "leaderboard": leaderboard,
        "results": results,
        "champion": {"name": champion["name"], "composite": champion["composite"], "why": _why(champion, ranked),
                     "selection_policy": "composite of validation AUC, validation Brier, validation→calibration consistency, "
                                         "train→validation overfit gap and simplicity; test and OOT never used for selection"},
        "nuscore": nu,
        "shap": shap_summary,
        "diagnostics": diag,
        "drift": drift_report,
        "warnings": warnings,
        "unlabelled_scoring": unlabelled_scoring,
        "model_id": model_id,
        "schema": schema.to_dict(),
    }


# ------------------------------------------------------------------ helpers


def _summary(frames: Dict[str, pd.DataFrame], feature_cols: List[str], target_col: str, label_col: str) -> Dict[str, Any]:
    full = pd.concat(list(frames.values()), axis=0)
    numeric = sum(pd.api.types.is_numeric_dtype(full[c]) for c in feature_cols)
    balance = {}
    for name, part in frames.items():
        counts = part[label_col].value_counts(normalize=True) if label_col in part.columns else part[target_col].value_counts(normalize=True)
        balance[name] = {str(k): round(float(v), 4) for k, v in counts.items()}
    return {"n_rows": int(len(full)), "n_features": len(feature_cols), "target_col": label_col,
            "class_balance": balance, "feature_types": {"numeric": int(numeric), "categorical": int(len(feature_cols) - numeric)}}


def _stack(ok, fitted, X_scaled, X_raw, ys, seed):
    """Logistic meta-learner on out-of-fold predictions of the top models."""
    members = [m for m in ok[:STACK_TOP_K]]
    if len(members) < 2:
        return None, None
    t0 = time.time()
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    oof_cols, split_cols = [], {name: [] for name in COHORTS}
    for m in members:
        X = X_scaled if m["_scaled"] else X_raw
        Xtr, ytr = X["train"], ys["train"]
        if len(ytr) > 200_000:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(ytr), size=200_000, replace=False)
            Xtr, ytr = Xtr[idx], ytr[idx]
        try:
            est = _clone(fitted[m["name"]])
            oof = cross_val_predict(est, Xtr, ytr, cv=folds, method="predict_proba")[:, 1]
        except Exception:
            logger.exception("OOF failed for %s; excluded from blend", m["name"])
            continue
        oof_cols.append((m["name"], oof, ytr))
        for name in split_cols:
            split_cols[name].append(m["_probs"][name])
    if len(oof_cols) < 2:
        return None, None
    y_oof = oof_cols[0][2]
    meta_X = np.column_stack([_logit(c[1]) for c in oof_cols])
    meta = LogisticRegression(C=1.0, max_iter=1000).fit(meta_X, y_oof)
    weights = meta.coef_[0].tolist()
    probs = {}
    for name, cols in split_cols.items():
        stacked = np.column_stack([_logit(c) for c in cols[:len(oof_cols)]])
        probs[name] = meta.predict_proba(stacked)[:, 1]
    blend = {"members": [c[0] for c in oof_cols], "weights": weights, "intercept": float(meta.intercept_[0]),
             "seconds": round(time.time() - t0, 2)}
    return blend, probs


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _clone(model):
    from sklearn.base import clone
    return clone(model)


def _importance(model, names: List[str], top: int = 25) -> List[Dict[str, float]]:
    values = None
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        values = np.asarray(model.coef_, dtype=float).ravel()
    if values is None or len(values) != len(names):
        return []
    order = np.argsort(-np.abs(values))[:top]
    return [{"feature": names[i], "importance": float(values[i])} for i in order]


def _permutation_importance(model, X: np.ndarray, y: np.ndarray, names: List[str], seed: int,
                            top: int = 25) -> List[Dict[str, float]]:
    """Validation AUC drop when a column is shuffled (3 repeats), for models
    without native importances (MLP, transformer)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=min(PERMUTATION_ROWS, len(y)), replace=False)
    Xp, yp = X[idx], y[idx]
    if len(np.unique(yp)) < 2:
        return []
    baseline = roc_auc_score(yp, model.predict_proba(Xp)[:, 1])
    candidates = list(range(X.shape[1]))
    if len(candidates) > PERMUTATION_FEATURES:
        spread = np.nanstd(Xp, axis=0)
        candidates = list(np.argsort(-spread)[:PERMUTATION_FEATURES])
    rows = []
    for j in candidates:
        drops = []
        for _ in range(3):
            shuffled = Xp.copy()
            shuffled[:, j] = rng.permutation(shuffled[:, j])
            drops.append(baseline - roc_auc_score(yp, model.predict_proba(shuffled)[:, 1]))
        rows.append({"feature": names[j], "importance": float(np.mean(drops))})
    rows.sort(key=lambda r: -abs(r["importance"]))
    return rows[:top]


def _score_unlabelled(bundle: ModelBundle, unlabelled: pd.DataFrame, feature_cols: List[str], artifact_dir: Path) -> Dict[str, Any]:
    frame = unlabelled[[c for c in feature_cols if c in unlabelled.columns]]
    scored = bundle.score(frame, with_reasons=len(frame) <= 2000)
    out = unlabelled.copy()
    out["nu_score"] = [s["nu_score"] for s in scored]
    out["nu_band"] = [s["band"] for s in scored]
    out["probability"] = [s["probability"] for s in scored]
    out["role"] = "scoring_only"
    scored_dir = artifact_dir / "scored"
    scored_dir.mkdir(parents=True, exist_ok=True)
    file_id = uuid.uuid4().hex
    out.to_csv(scored_dir / f"{file_id}.csv", index=False)
    return {"rows": int(len(out)), "file_id": file_id,
            "band_counts": out["nu_band"].value_counts().to_dict(),
            "mean_score": round(float(out["nu_score"].mean()), 1),
            "note": "Rows whose outcome was blank were scored with the final model and never entered any cohort."}


def _legacy_metrics(m: Dict[str, Any]) -> Dict[str, float]:
    return {k: m[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc")}


def _why(champion: Dict[str, Any], ranked: List[Dict[str, Any]]) -> str:
    c = champion["composite"]["components"]
    parts = [f"Best composite score ({champion['composite']['score']:.3f}) across {len(ranked)} models on the selection cohorts"]
    strong = [k for k, v in sorted(c.items(), key=lambda kv: -kv[1]) if v >= 0.8][:3]
    if strong:
        parts.append("strongest on " + ", ".join(strong))
    s = champion["stability"]
    parts.append(f"validation AUC {champion['evaluations']['validation']['metrics']['roc_auc']:.3f}, "
                 f"held-out test AUC {champion['evaluations']['test']['metrics']['roc_auc']:.3f}, "
                 f"OOT drop {s['oot_drop']:+.3f}, overfit gap {s['overfit_gap']:.3f}")
    return "; ".join(parts) + "."


def _jsonable(params: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in params.items():
        if isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        elif isinstance(v, tuple):
            out[k] = list(v)
        else:
            out[k] = v
    return out
