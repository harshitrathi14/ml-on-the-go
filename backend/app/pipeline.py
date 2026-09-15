"""
End-to-end training pipelines shared by the HTTP endpoints and the job runner.

Each pipeline returns a JSON-serialisable dict shaped like TrainResponse:
{"dataset": {...}, "leaderboard": [...], "results": [...]}.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import polars as pl

from .data import generate_synthetic_dataset

TARGET_BINARY_COL = "__target_binary__"

# Rows loaded from a large upload. Evaluation uses every row loaded; model
# fitting is capped separately in engine.train so a 20 GB upload still
# trains in minutes.
MAX_ROWS = 3_000_000


class TargetError(ValueError):
    """Raised when the requested target column cannot be used."""


def binarize_target(
    df: pd.DataFrame,
    target_col: str,
    positive_class: Optional[str] = None,
    threshold: Optional[float] = None,
) -> pd.DataFrame:
    """Replace ``target_col`` with a 0/1 column named TARGET_BINARY_COL."""
    if target_col not in df.columns:
        raise TargetError(f"Column '{target_col}' not found in uploaded data.")

    series = df[target_col]
    unique_vals = series.dropna().unique()

    try:
        if set(unique_vals).issubset({0, 1, True, False, "0", "1"}):
            binary = series.map({True: 1, False: 0, 1: 1, 0: 0, "1": 1, "0": 0})
        elif positive_class is not None:
            binary = (series.astype(str) == str(positive_class)).astype(int)
        elif threshold is not None:
            binary = (pd.to_numeric(series, errors="coerce") > threshold).astype(int)
        else:
            numeric = pd.to_numeric(series, errors="coerce")
            if numeric.notna().sum() > len(series) * 0.8:
                binary = (numeric > float(numeric.median())).astype(int)
            else:
                most_common = series.value_counts().index[0]
                binary = (series != most_common).astype(int)
    except Exception as exc:
        raise TargetError(f"Could not binarize target column '{target_col}': {exc}") from exc

    # Rows with a missing target carry no label; drop them rather than guess.
    keep = binary.notna()
    binary = binary[keep].astype(int)
    if binary.nunique() != 2:
        raise TargetError(f"Target '{target_col}' could not be reduced to exactly 2 classes.")

    out = df.loc[keep].drop(columns=[target_col])
    out[TARGET_BINARY_COL] = binary
    return out


def describe_binarization(series: pd.Series, positive_class: Optional[str], threshold: Optional[float]) -> Dict[str, Any]:
    """How the target will be turned into 0/1 (for the UI to confirm)."""
    unique_vals = series.dropna().unique()
    if set(unique_vals).issubset({0, 1, True, False, "0", "1"}):
        return {"strategy": "already_binary", "positive_class": "1"}
    if positive_class is not None:
        return {"strategy": "positive_class", "positive_class": str(positive_class)}
    if threshold is not None:
        return {"strategy": "threshold", "threshold": float(threshold)}
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() > len(series) * 0.8:
        return {"strategy": "threshold", "threshold": float(numeric.median()),
                "note": "numeric target: values above the median count as positive"}
    most_common = str(series.value_counts().index[0])
    return {"strategy": "not_most_common", "positive_class": f"anything except '{most_common}'",
            "note": f"'{most_common}' is the most common value and is treated as negative"}


def _train_and_package(
    df: pd.DataFrame, target_col: str, seed: int, summary_target: str,
    time_col: Optional[str] = None, extra: Optional[Dict[str, Any]] = None,
    tier: str = "balanced", artifact_dir=None, progress=None, unlabelled=None,
) -> dict:
    from .engine import train as engine

    # The label column ("decision") duplicates the binary target; never a feature.
    if summary_target != target_col and summary_target in df.columns:
        df = df.rename(columns={summary_target: "__label__"})
        df["__label__"] = df["__label__"].astype(str)
        label_col = "__label__"
    else:
        label_col = target_col
    result = engine.run(
        df, target_col=target_col, time_col=time_col, tier=tier, seed=seed,
        artifact_dir=artifact_dir, progress=progress or (lambda m, e=None: None),
        summary_target=label_col, extra_dataset=extra, unlabelled=unlabelled,
    )
    if label_col == "__label__":
        result["dataset"]["target_col"] = summary_target
    return result


def run_synthetic(
    n_rows: int,
    n_features: int,
    n_categorical: int,
    seed: int,
    decision_labels: Tuple[str, str],
    default_rate: float = 0.18,
    tier: str = "balanced",
    artifact_dir=None,
    progress=None,
) -> dict:
    df = generate_synthetic_dataset(
        n_rows=n_rows,
        n_features=n_features,
        n_categorical=n_categorical,
        seed=seed,
        decision_labels=tuple(decision_labels),
        default_rate=default_rate,
    )
    return _train_and_package(df, "decision_binary", seed, summary_target="decision",
                              tier=tier, artifact_dir=artifact_dir, progress=progress)


# ------------------------------------------------------------ session data


def load_frame(data_path: str, seed: int, max_rows: int = MAX_ROWS) -> pd.DataFrame:
    lf = pl.scan_parquet(data_path)
    n = lf.select(pl.len()).collect().item()
    if n > max_rows:
        frame = lf.collect().sample(n=max_rows, seed=seed)
    else:
        frame = lf.collect()
    return frame.to_pandas()


def _parse_dates(df: pd.DataFrame, cols: List[str]) -> None:
    for col in cols:
        if col not in df.columns:
            continue
        series = df[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            continue
        if pd.api.types.is_integer_dtype(series):
            df[col] = pd.to_datetime(series.astype("Int64").astype(str), errors="coerce", format="%Y%m%d")
        else:
            df[col] = pd.to_datetime(series, errors="coerce", format="mixed")


def _derive_date_features(df: pd.DataFrame, date_cols: List[str], time_col: Optional[str]) -> None:
    """Dates cannot be fed to the models directly; keep month and, when a
    reference date exists, the gap in days to it."""
    ref = df[time_col] if time_col and time_col in df.columns else None
    for col in date_cols:
        if col == time_col or col not in df.columns:
            continue
        df[f"{col}__month"] = df[col].dt.month.astype("float")
        if ref is not None:
            df[f"{col}__days_before_ref"] = (ref - df[col]).dt.days.astype("float")
        else:
            df[f"{col}__year"] = df[col].dt.year.astype("float")
        df.drop(columns=[col], inplace=True)


def run_session(
    data_path: str,
    target_col: str,
    positive_class: Optional[str] = None,
    threshold: Optional[float] = None,
    exclude_cols: Optional[List[str]] = None,
    date_cols: Optional[List[str]] = None,
    time_col: Optional[str] = None,
    seed: int = 42,
    tier: str = "balanced",
    artifact_dir=None,
    progress=None,
) -> dict:
    if progress:
        progress("Loading the modelling table")
    df = load_frame(data_path, seed)
    exclude = [c for c in (exclude_cols or []) if c in df.columns and c != target_col]
    df = df.drop(columns=exclude)

    date_cols = [c for c in (date_cols or []) if c in df.columns]
    if time_col and time_col not in df.columns:
        time_col = None
    if time_col and time_col not in date_cols:
        date_cols.append(time_col)
    _parse_dates(df, date_cols)
    if time_col:
        df = df[df[time_col].notna()]
        if df[time_col].nunique() < 3:
            time_col = None
    _derive_date_features(df, date_cols, time_col)

    # Deterministic clean-up of text columns ("₹1,20,000", "25%", "yes"/"no",
    # "NA"), with a change log the user can review.
    from .engine.normalise import normalise_frame
    df, normalisation_log = normalise_frame(df, skip=(target_col, time_col) if time_col else (target_col,))

    # Rows without an outcome are scored later but never enter a cohort.
    blank = df[target_col].isna() | df[target_col].astype(str).str.strip().str.lower().isin({"", "na", "n/a", "null", "none", "nan"})
    unlabelled = df.loc[blank].drop(columns=[target_col]) if blank.any() else None
    df = df.loc[~blank]

    df = binarize_target(df, target_col, positive_class, threshold)
    # Text columns with one value per row (free text, IDs missed earlier)
    # carry no signal for one-hot models and make preprocessing slow.
    for col in list(df.columns):
        if col in (TARGET_BINARY_COL, time_col):
            continue
        if not pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 0.5 * len(df):
            df.drop(columns=[col], inplace=True)
            exclude.append(col)
    if unlabelled is not None:
        unlabelled = unlabelled.drop(columns=[c for c in exclude if c in unlabelled.columns])

    extra = {"excluded_columns": exclude, "time_col": time_col, "rows_loaded": int(len(df)),
             "unlabelled_rows": int(len(unlabelled)) if unlabelled is not None else 0,
             "target_col": target_col, "normalisation_log": normalisation_log}
    return _train_and_package(df, TARGET_BINARY_COL, seed, summary_target=TARGET_BINARY_COL,
                              time_col=time_col, extra=extra, tier=tier,
                              artifact_dir=artifact_dir, progress=progress, unlabelled=unlabelled)
