"""
Anthropic Claude client implementation.
Requires: pip install anthropic>=0.25
Set ANTHROPIC_API_KEY in environment.
"""

from __future__ import annotations

import json
from typing import List, Optional

import pandas as pd

from .base import AIClient
from ._helpers import (
    ANALYZE_SYSTEM,
    EXPLAIN_SYSTEM,
    build_metadata,
    build_explain_payload,
    safe_parse,
    fallback_analysis,
    fallback_explain,
)

DEFAULT_MODEL = "claude-sonnet-4-6"


class ClaudeClient(AIClient):
    def __init__(self, api_key: str, model: Optional[str] = None) -> None:
        import anthropic  # lazy import

        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model or DEFAULT_MODEL

    def analyze_csv_metadata(self, df: pd.DataFrame) -> dict:
        metadata = build_metadata(df)
        prompt = f"Analyze this CSV metadata and return the JSON analysis:\n\n{json.dumps(metadata, indent=2)}"
        raw = self._call(prompt, ANALYZE_SYSTEM)
        result = safe_parse(raw, fallback_analysis(df))
        columns = list(df.columns)
        if result.get("suggested_target_col") not in columns:
            result["suggested_target_col"] = columns[-1]
            result["confidence"] = "low"
        return result

    def explain_model_results(
        self,
        leaderboard: List[dict],
        results: List[dict],
        dataset_summary: Optional[dict] = None,
    ) -> dict:
        payload = build_explain_payload(leaderboard, results, dataset_summary)
        prompt = f"Explain these ML model results:\n\n{json.dumps(payload, indent=2)}"
        raw = self._call(prompt, EXPLAIN_SYSTEM)
        return safe_parse(raw, fallback_explain())

    def _call(self, user_prompt: str, system_prompt: str) -> str:
        message = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return message.content[0].text
