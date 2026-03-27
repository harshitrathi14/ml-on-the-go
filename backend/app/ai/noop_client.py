"""
Python-only fallback AI client.
Returns static placeholder responses — no external calls, no API keys required.
Used when no AI provider is configured.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from .base import AIClient
from ._helpers import fallback_analysis, fallback_explain


class NoOpAIClient(AIClient):
    """
    Works without any API key or external service.
    /upload-csv still works; ai_analysis fields are clearly labelled as unavailable.
    """

    def analyze_csv_metadata(self, df: pd.DataFrame) -> dict:
        cols = list(df.columns)
        # Simple heuristic: pick the last column that looks like a binary/label col
        suggested = _guess_target(df)
        cardinality = int(df[suggested].nunique(dropna=True)) if suggested else 0
        if cardinality == 2:
            strategy = "already_binary"
            problem_type = "binary_classification"
            confidence = "medium"
        elif cardinality > 2 and df[suggested].dtype == object:
            strategy = "label_encode"
            problem_type = "binary_classification"
            confidence = "low"
        else:
            strategy = "threshold"
            problem_type = "regression"
            confidence = "low"

        return {
            "suggested_target_col": suggested or cols[-1],
            "confidence": confidence,
            "problem_type": problem_type,
            "binarization_strategy": {
                "strategy": strategy,
                "positive_class": None,
                "threshold": None,
                "notes": "Heuristic guess — no AI provider configured. Please verify.",
            },
            "data_quality_issues": _basic_quality_checks(df),
            "feature_notes": [
                "No AI provider configured — analysis is rule-based only.",
                f"Dataset has {len(cols)} columns and {len(df):,} rows.",
                "Configure ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, or set AI_PROVIDER=ollama/lmstudio for richer analysis.",
            ],
            "recommendation": (
                "No AI provider is configured. The target column was guessed heuristically. "
                "Please review the column selection and set up an AI provider for smarter recommendations."
            ),
        }

    def explain_model_results(
        self,
        leaderboard: List[dict],
        results: List[dict],
        dataset_summary: Optional[dict] = None,
    ) -> dict:
        best = leaderboard[0] if leaderboard else {}
        model_name = best.get("model", "N/A")
        auc = best.get("roc_auc", 0)
        return {
            "executive_summary": (
                f"Training completed with {len(leaderboard)} models. "
                f"Best model: {model_name} (ROC-AUC {auc:.3f}). "
                "Configure an AI provider for a richer analysis."
            ),
            "best_model_analysis": (
                f"{model_name} achieved the highest ROC-AUC of {auc:.3f}. "
                "No AI provider is configured, so a detailed analysis is unavailable."
            ),
            "feature_insights": [
                "Feature analysis requires an AI provider.",
                "Check the feature importance chart in the dashboard for top drivers.",
                "Configure ANTHROPIC_API_KEY or OPENAI_API_KEY for automated insights.",
            ],
            "risk_flags": _basic_risk_flags(leaderboard, results),
            "recommendations": [
                "Review all split metrics (train/test/OOT) for consistency.",
                "Investigate features with high importance for business plausibility.",
                "Configure an AI provider to get automated plain-English explanations.",
            ],
        }


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------


def _guess_target(df: pd.DataFrame) -> str:
    """Guess target column by name patterns then by low-cardinality last column."""
    target_hints = {"target", "label", "class", "y", "outcome", "default", "churn",
                    "fraud", "survived", "species", "diagnosis"}
    for col in df.columns:
        if col.lower() in target_hints:
            return col
    # Low-cardinality columns are often targets
    low_card = [c for c in df.columns if df[c].nunique(dropna=True) <= 10]
    if low_card:
        return low_card[-1]
    return df.columns[-1]


def _basic_quality_checks(df: pd.DataFrame) -> list:
    issues = []
    for col in df.columns:
        null_pct = float(df[col].isna().mean())
        if null_pct > 0.3:
            issues.append({
                "column": col,
                "issue_type": "high_nulls",
                "severity": "high",
                "description": f"{null_pct:.1%} missing values.",
            })
        elif df[col].nunique(dropna=True) == 1:
            issues.append({
                "column": col,
                "issue_type": "constant_column",
                "severity": "medium",
                "description": "Column has only one unique value.",
            })
    return issues


def _basic_risk_flags(leaderboard: list, results: list) -> list:
    flags = []
    if leaderboard:
        best_auc = leaderboard[0].get("roc_auc", 1.0)
        if best_auc < 0.65:
            flags.append(f"Low best model AUC ({best_auc:.3f}) — models may lack predictive power.")
    # Check for large train/test gap
    for r in results[:2]:
        train_auc = (r.get("metrics") or {}).get("train", {}).get("roc_auc")
        test_auc = (r.get("metrics") or {}).get("test", {}).get("roc_auc")
        if train_auc and test_auc and (train_auc - test_auc) > 0.1:
            flags.append(f"{r['name']}: train/test AUC gap of {train_auc - test_auc:.3f} suggests overfitting.")
            break
    if not flags:
        flags.append("No obvious risk flags detected (rule-based check only).")
    return flags
