"""
Shared helpers: metadata builders, prompt strings, JSON parsing, fallbacks.
Used by all provider-specific client implementations.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

ANALYZE_SYSTEM = """You are an expert data scientist and ML engineer.
Analyze the CSV metadata provided and return ONLY a valid JSON object — no prose, no markdown fences.
The JSON must conform exactly to this schema:
{
  "suggested_target_col": "<column name — MUST be one of the column names listed>",
  "confidence": "high" | "medium" | "low",
  "problem_type": "binary_classification" | "multiclass" | "regression" | "unknown",
  "binarization_strategy": {
    "strategy": "already_binary" | "threshold" | "top_k_classes" | "label_encode",
    "positive_class": "<string or null>",
    "threshold": <float or null>,
    "notes": "<short explanation>"
  },
  "data_quality_issues": [
    { "column": "<name>", "issue_type": "<type>", "severity": "high"|"medium"|"low", "description": "<detail>" }
  ],
  "feature_notes": ["<note1>", "<note2>", "<note3>"],
  "recommendation": "<1-2 sentence plain-English recommendation>"
}
Rules:
- suggested_target_col MUST be one of the provided column names.
- Flag any column where null_pct > 0.3 as a data quality issue with severity "high".
- Flag columns with cardinality == 1 (constant) as severity "medium".
- Keep feature_notes to exactly 3 items.
- Return ONLY the JSON object. No extra text."""

EXPLAIN_SYSTEM = """You are a senior ML risk analyst providing plain-English model explanations to a business audience.
Analyze the model results provided and return ONLY a valid JSON object — no prose, no markdown fences.
The JSON must conform exactly to this schema:
{
  "executive_summary": "<2-3 sentence overview of overall model performance>",
  "best_model_analysis": "<2-3 sentences on why the best model performed well and any caveats>",
  "feature_insights": ["<insight 1>", "<insight 2>", "<insight 3>"],
  "risk_flags": ["<flag 1>", "<flag 2>"],
  "recommendations": ["<action 1>", "<action 2>", "<action 3>"]
}
Rules:
- Use plain English; avoid jargon unless explained inline.
- feature_insights: exactly 3 items about the top predictive features.
- risk_flags: 1-3 genuine concerns (overfitting, low AUC, class imbalance, OOT degradation).
- recommendations: exactly 3 actionable next steps.
- Return ONLY the JSON object. No extra text."""

# ---------------------------------------------------------------------------
# Metadata / payload builders
# ---------------------------------------------------------------------------


def build_metadata(df: pd.DataFrame) -> Dict[str, Any]:
    """Build column-level statistics for the AI prompt. Never includes raw rows."""
    columns_meta = []
    for col in df.columns:
        series = df[col]
        null_pct = float(series.isna().mean())
        cardinality = int(series.nunique(dropna=True))
        dtype = str(series.dtype)
        samples = [_to_native(v) for v in series.dropna().head(5).tolist()]
        columns_meta.append(
            {
                "name": col,
                "dtype": dtype,
                "null_pct": round(null_pct, 4),
                "cardinality": cardinality,
                "sample_values": samples,
            }
        )
    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": columns_meta,
    }


def build_explain_payload(
    leaderboard: List[dict],
    results: List[dict],
    dataset_summary: Optional[dict],
) -> Dict[str, Any]:
    """Slim payload: top-2 models, key splits, key metrics, top-5 features."""
    top2_names = {row["model"] for row in leaderboard[:2]}
    slim_results = []
    for r in results:
        if r["name"] not in top2_names:
            continue
        metrics_slim: Dict[str, Any] = {}
        for split in ("train", "test", "oot"):
            m = (r.get("metrics") or {}).get(split, {})
            metrics_slim[split] = {
                k: round(float(v), 4)
                for k, v in m.items()
                if k in ("roc_auc", "f1", "ks_stat")
            }
        top_features = sorted(
            r.get("feature_importance") or [],
            key=lambda x: x.get("importance", 0),
            reverse=True,
        )[:5]
        slim_results.append(
            {"name": r["name"], "metrics": metrics_slim, "top_features": top_features}
        )
    return {
        "leaderboard": leaderboard[:5],
        "top_models": slim_results,
        "dataset_summary": dataset_summary,
    }


# ---------------------------------------------------------------------------
# JSON parsing + fallbacks
# ---------------------------------------------------------------------------


def safe_parse(raw: str, fallback: dict) -> dict:
    try:
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        return json.loads(text)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Failed to parse AI JSON response: %s", exc)
        return fallback


def fallback_analysis(df: pd.DataFrame) -> dict:
    cols = list(df.columns)
    return {
        "suggested_target_col": cols[-1],
        "confidence": "low",
        "problem_type": "unknown",
        "binarization_strategy": {
            "strategy": "label_encode",
            "positive_class": None,
            "threshold": None,
            "notes": "AI analysis unavailable; please select target manually.",
        },
        "data_quality_issues": [],
        "feature_notes": [
            "AI analysis unavailable.",
            "Please review columns manually.",
            "Select the correct target column before training.",
        ],
        "recommendation": "AI analysis could not be completed. Please review your data and select the target column manually.",
    }


def fallback_explain() -> dict:
    return {
        "executive_summary": "AI explanation unavailable. Review the leaderboard metrics directly.",
        "best_model_analysis": "Unable to generate automated analysis at this time.",
        "feature_insights": [
            "Feature insights unavailable.",
            "Review feature importance chart above.",
            "Consider consulting a domain expert.",
        ],
        "risk_flags": ["AI explanation service unavailable."],
        "recommendations": [
            "Review model metrics manually.",
            "Compare AUC scores across splits.",
            "Investigate any large train/test performance gaps.",
        ],
    }


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _to_native(val: Any) -> Any:
    try:
        import numpy as np

        if isinstance(val, np.integer):
            return int(val)
        if isinstance(val, np.floating):
            return float(val)
        if isinstance(val, np.bool_):
            return bool(val)
    except ImportError:
        pass
    return val
