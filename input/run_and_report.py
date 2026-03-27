"""
End-to-end runner: load sample CSVs → train ML models → write HTML + PDF reports.

Usage:
    python input/run_and_report.py

Outputs go to:  output/
  ├── small_shop_sales_2yr_report.html
  ├── small_shop_sales_2yr_report.pdf
  ├── online_seller_inventory_1yr_report.html
  └── online_seller_inventory_1yr_report.pdf

No API key needed — uses the Python-only NoOpAIClient.
"""

import os
import sys
import textwrap
from dataclasses import asdict

import pandas as pd

# Make sure the project root is on sys.path
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
# Helpers
# ──────────────────────────────────────────────────────────────────────────

def binarize_target(df: pd.DataFrame, target_col: str,
                    positive_class=None) -> pd.DataFrame:
    """Return df with '__target__' column (int 0/1)."""
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


def run_pipeline(csv_name: str, target_col: str,
                 positive_class=None, seed: int = 42) -> None:
    """Load CSV, train models, write HTML + PDF into output/."""
    print(f"\n{'─'*60}")
    print(f"  Dataset : {csv_name}")
    print(f"  Target  : {target_col}  (positive class = {positive_class})")
    print(f"{'─'*60}")

    # ── load ──────────────────────────────────────────────────────────────
    csv_path = os.path.join(INPUT_DIR, csv_name)
    df = pd.read_csv(csv_path)
    print(f"  Loaded  : {len(df):,} rows × {len(df.columns)} columns")

    # Drop pure-ID columns (object dtype, near-unique)
    id_cols = [c for c in df.columns
               if df[c].dtype == object and df[c].nunique() > len(df) * 0.8]
    if id_cols:
        print(f"  Dropping ID-like columns: {id_cols}")
        df = df.drop(columns=id_cols)

    # Drop the target from features, binarize
    df = binarize_target(df, target_col, positive_class)
    class_balance = df["__target__"].value_counts(normalize=True).to_dict()
    print(f"  Class balance: { {k: f'{v:.1%}' for k,v in class_balance.items()} }")

    # ── split ──────────────────────────────────────────────────────────────
    splits  = split_dataset(df, target_col="__target__", seed=seed)
    summary = summarize_dataset(splits, target_col="__target__")
    print(f"  Splits  : train={len(splits.train)} | test={len(splits.test)} | "
          f"oot={len(splits.oot)} | etrc={len(splits.etrc)}")

    # ── train ──────────────────────────────────────────────────────────────
    print("  Training models (this may take ~60-90 seconds)…")
    output  = train_models(splits.train, splits.test, splits.oot, splits.etrc,
                           target_col="__target__", seed=seed)

    leaderboard = output.leaderboard
    results     = [asdict(r) for r in output.results]

    print("\n  Leaderboard:")
    for i, row in enumerate(leaderboard):
        print(f"    {i+1}. {row['model']:22s}  ROC-AUC = {row['roc_auc']:.4f}")

    # ── AI insights (Python-only, no API key) ────────────────────────────
    ai_insights = AI.explain_model_results(leaderboard, results, asdict(summary))

    # ── reports ───────────────────────────────────────────────────────────
    stem = csv_name.replace(".csv", "")
    ds_dict = asdict(summary)

    html_path = os.path.join(OUTPUT_DIR, f"{stem}_report.html")
    html = generate_html_report(leaderboard, results, ds_dict, ai_insights)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n  ✓  HTML report : {os.path.relpath(html_path, ROOT)}  ({len(html):,} bytes)")

    pdf_path = os.path.join(OUTPUT_DIR, f"{stem}_report.pdf")
    pdf = generate_pdf_report(leaderboard, results, ds_dict, ai_insights)
    with open(pdf_path, "wb") as f:
        f.write(pdf)
    print(f"  ✓  PDF  report : {os.path.relpath(pdf_path, ROOT)}  ({len(pdf):,} bytes)")


# ──────────────────────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────────────────────

JOBS = [
    dict(csv_name="small_shop_sales_2yr.csv",
         target_col="high_value_transaction",
         positive_class="1"),

    dict(csv_name="online_seller_inventory_1yr.csv",
         target_col="stockout_risk",
         positive_class="1"),

    dict(csv_name="customer_churn_1yr.csv",
         target_col="churned",
         positive_class="1"),
]

if __name__ == "__main__":
    print(textwrap.dedent("""
    ╔══════════════════════════════════════════════════════╗
    ║   ML on the Go — Sample Data End-to-End Runner      ║
    ╚══════════════════════════════════════════════════════╝
    """))

    for job in JOBS:
        run_pipeline(**job)

    print(textwrap.dedent(f"""
    ══════════════════════════════════════════════════════
     All done!  Reports saved to:  output/
    ══════════════════════════════════════════════════════
    """))
