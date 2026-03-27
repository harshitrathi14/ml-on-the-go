"""
End-to-end runner: load sample CSVs → clean features → train ML models
→ write rich HTML + PDF reports into output/

Usage:
    python input/run_and_report.py

Outputs:
    output/<dataset>_report.html   — self-contained, layman-friendly HTML
    output/<dataset>_report.pdf    — A4 PDF with same content
"""

import os
import sys
import textwrap
from dataclasses import asdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend.app.data import split_dataset, summarize_dataset
from backend.app.modeling import train_models
from backend.app.reporting import generate_html_report, generate_pdf_report
from backend.app.ai.noop_client import NoOpAIClient

INPUT_DIR  = os.path.join(ROOT, "input")
OUTPUT_DIR = os.path.join(ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

AI = NoOpAIClient()


# ──────────────────────────────────────────────────────────────────────────
# Feature cleaning
# ──────────────────────────────────────────────────────────────────────────

def _looks_like_date(series: pd.Series) -> bool:
    """Return True if the column contains date/datetime strings."""
    sample = series.dropna().astype(str).head(20)
    if len(sample) == 0:
        return False
    try:
        parsed = pd.to_datetime(sample, infer_datetime_format=True, errors="coerce")
        return parsed.notna().mean() > 0.80
    except Exception:
        return False


def _looks_like_id(col: str, series: pd.Series) -> bool:
    """Return True if column is a row-level identifier (not useful for ML)."""
    name_lower = col.lower()
    # Name-based heuristics
    id_patterns = ["_id", "_key", "_uuid", "_code", "transaction_id",
                   "record_id", "row_id", "index", "snapshot_date"]
    for pat in id_patterns:
        if name_lower.endswith(pat) or name_lower == pat.lstrip("_"):
            return True
    # High-cardinality string where uniqueness > 20% of rows
    if series.dtype == object:
        uniqueness = series.nunique() / max(len(series), 1)
        if uniqueness > 0.20:
            return True
    return False


def clean_features(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, dict]:
    """
    Return (clean_df, drop_report) where clean_df has only good ML features
    plus the target column, and drop_report explains what was removed and why.
    """
    dropped: dict[str, str] = {}

    for col in list(df.columns):
        if col == target_col:
            continue

        # 1. Constant columns
        if df[col].nunique(dropna=False) <= 1:
            dropped[col] = "constant (single unique value)"
            continue

        # 2. Date / timestamp strings
        if df[col].dtype == object and _looks_like_date(df[col]):
            dropped[col] = "date/timestamp string — already extracted as numeric components"
            continue

        # 3. Identifier columns
        if _looks_like_id(col, df[col]):
            dropped[col] = "identifier column — not a predictive feature"
            continue

        # 4. Near-zero variance numerics  (std < 0.001% of mean)
        if pd.api.types.is_numeric_dtype(df[col]):
            std = df[col].std()
            mean = abs(df[col].mean()) + 1e-9
            if std / mean < 1e-5:
                dropped[col] = "near-zero variance numeric"
                continue

    clean = df.drop(columns=list(dropped.keys()), errors="ignore")
    return clean, dropped


# ──────────────────────────────────────────────────────────────────────────
# Feature metadata (for the report)
# ──────────────────────────────────────────────────────────────────────────

def compute_feature_meta(df: pd.DataFrame, feature_cols: list) -> list[dict]:
    """Return per-feature statistics used to enrich the report."""
    meta = []
    for col in feature_cols:
        s = df[col]
        null_pct = round(s.isna().mean() * 100, 1)
        if pd.api.types.is_numeric_dtype(s):
            meta.append({
                "column":   col,
                "type":     "numeric",
                "null_pct": null_pct,
                "mean":     round(s.mean(), 3) if s.notna().any() else None,
                "std":      round(s.std(), 3)  if s.notna().any() else None,
                "min":      round(s.min(), 3)  if s.notna().any() else None,
                "max":      round(s.max(), 3)  if s.notna().any() else None,
                "unique":   int(s.nunique()),
            })
        else:
            top = s.value_counts().index[0] if s.notna().any() else "N/A"
            meta.append({
                "column":   col,
                "type":     "categorical",
                "null_pct": null_pct,
                "unique":   int(s.nunique()),
                "top_val":  str(top),
                "mean":     None, "std": None, "min": None, "max": None,
            })
    return meta


# ──────────────────────────────────────────────────────────────────────────
# Target binarization
# ──────────────────────────────────────────────────────────────────────────

def binarize_target(df: pd.DataFrame, target_col: str,
                    positive_class=None) -> pd.DataFrame:
    series = df[target_col]
    unique = series.dropna().unique()
    if set(unique).issubset({0, 1, True, False, "0", "1"}):
        binary = series.map({True: 1, False: 0, 1: 1, 0: 0,
                             "1": 1, "0": 0}).astype(int)
    elif positive_class is not None:
        binary = (series.astype(str) == str(positive_class)).astype(int)
    else:
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().sum() > len(series) * 0.8:
            binary = (numeric > float(numeric.median())).astype(int)
        else:
            most_common = series.value_counts().index[0]
            binary = (series != most_common).astype(int)

    out = df.drop(columns=[target_col]).copy()
    out["__target__"] = binary.values
    return out


# ──────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────

def run_pipeline(csv_name: str, target_col: str,
                 positive_class=None, seed: int = 42,
                 label: str = "") -> None:

    print(f"\n{'─'*60}")
    print(f"  Dataset : {csv_name}")
    print(f"  Target  : {target_col}  (positive = {positive_class})")
    print(f"{'─'*60}")

    # ── load ─────────────────────────────────────────────────────────────
    df_raw = pd.read_csv(os.path.join(INPUT_DIR, csv_name))
    print(f"  Loaded  : {len(df_raw):,} rows × {len(df_raw.columns)} columns")

    # ── clean features ────────────────────────────────────────────────────
    df_clean, dropped = clean_features(df_raw, target_col)
    if dropped:
        print(f"  Dropped {len(dropped)} non-feature column(s):")
        for col, reason in dropped.items():
            print(f"    ✗  {col:35s} → {reason}")
    print(f"  Features kept: {len(df_clean.columns) - 1}")

    # ── binarize ──────────────────────────────────────────────────────────
    df_model = binarize_target(df_clean, target_col, positive_class)
    balance  = df_model["__target__"].value_counts(normalize=True).to_dict()
    print(f"  Class balance: { {k: f'{v:.1%}' for k,v in balance.items()} }")

    # ── feature metadata (before splitting) ──────────────────────────────
    feature_cols = [c for c in df_model.columns if c != "__target__"]
    feat_meta    = compute_feature_meta(df_model, feature_cols)

    # ── split ─────────────────────────────────────────────────────────────
    splits  = split_dataset(df_model, target_col="__target__", seed=seed)
    summary = summarize_dataset(splits, target_col="__target__")
    print(f"  Splits  : train={len(splits.train):,} | test={len(splits.test):,} "
          f"| oot={len(splits.oot):,} | etrc={len(splits.etrc):,}")

    # ── train ─────────────────────────────────────────────────────────────
    print("  Training 5 models …")
    output     = train_models(splits.train, splits.test, splits.oot, splits.etrc,
                              target_col="__target__", seed=seed)
    leaderboard = output.leaderboard
    results     = [asdict(r) for r in output.results]

    print(f"\n  Leaderboard:")
    for i, row in enumerate(leaderboard):
        print(f"    {i+1}. {row['model']:22s}  ROC-AUC = {row['roc_auc']:.4f}")

    # ── AI insights ───────────────────────────────────────────────────────
    ds_dict     = asdict(summary)
    ai_insights = AI.explain_model_results(leaderboard, results, ds_dict)

    # ── build extra context for reports ───────────────────────────────────
    report_meta = {
        "dataset_label":  label or csv_name.replace(".csv", "").replace("_", " ").title(),
        "target_col":     target_col,
        "positive_class": positive_class,
        "feature_meta":   feat_meta,
        "dropped_cols":   dropped,
        "class_balance":  balance,
        "csv_name":       csv_name,
    }

    # ── write reports ──────────────────────────────────────────────────────
    stem     = csv_name.replace(".csv", "")
    ds_dict["feature_meta"]  = feat_meta
    ds_dict["dropped_cols"]  = dropped
    ds_dict["report_meta"]   = report_meta

    html_path = os.path.join(OUTPUT_DIR, f"{stem}_report.html")
    html = generate_html_report(leaderboard, results, ds_dict, ai_insights)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n  ✓  HTML : {os.path.relpath(html_path, ROOT)}  ({len(html):,} bytes)")

    pdf_path = os.path.join(OUTPUT_DIR, f"{stem}_report.pdf")
    pdf = generate_pdf_report(leaderboard, results, ds_dict, ai_insights)
    with open(pdf_path, "wb") as f:
        f.write(pdf)
    print(f"  ✓  PDF  : {os.path.relpath(pdf_path, ROOT)}  ({len(pdf):,} bytes)")


# ──────────────────────────────────────────────────────────────────────────
# Jobs
# ──────────────────────────────────────────────────────────────────────────

JOBS = [
    dict(csv_name="small_shop_sales_2yr.csv",
         target_col="high_value_transaction",
         positive_class="1",
         label="Small Shop — 2-Year Sales Analysis"),

    dict(csv_name="online_seller_inventory_1yr.csv",
         target_col="stockout_risk",
         positive_class="1",
         label="Online Seller — Inventory Stockout Risk"),

    dict(csv_name="customer_churn_1yr.csv",
         target_col="churned",
         positive_class="1",
         label="Customer Churn Prediction"),
]

if __name__ == "__main__":
    print(textwrap.dedent("""
    ╔══════════════════════════════════════════════════════╗
    ║   ML on the Go — Full Pipeline Runner               ║
    ╚══════════════════════════════════════════════════════╝
    """))
    for job in JOBS:
        run_pipeline(**job)
    print(textwrap.dedent(f"""
    ══════════════════════════════════════════════════════
     All done — reports saved to:  output/
    ══════════════════════════════════════════════════════
    """))
