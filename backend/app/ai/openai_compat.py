"""
OpenAI-compatible client — covers OpenAI, Gemini, Ollama, and LM Studio.

All four providers expose an OpenAI-compatible /v1/chat/completions API,
so one implementation covers them all.

Provider       | base_url                                                | key env var
---------------|--------------------------------------------------------|-----------------
openai         | (default)                                               | OPENAI_API_KEY
gemini         | https://generativelanguage.googleapis.com/v1beta/openai/| GEMINI_API_KEY
ollama         | http://localhost:11434/v1                               | none ("ollama")
lmstudio       | http://localhost:1234/v1                                | none ("lmstudio")

Requires: pip install openai>=1.0
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

# Sensible defaults per provider
_DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.0-flash",
    "ollama": "llama3.2",
    "lmstudio": "local-model",
}

_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


class OpenAICompatClient(AIClient):
    """
    Args:
        provider: one of "openai" | "gemini" | "ollama" | "lmstudio"
        api_key:  API key string (pass "ollama" or "lmstudio" as dummy if not needed)
        base_url: override the endpoint URL (auto-set for gemini/ollama/lmstudio)
        model:    override the model name
    """

    def __init__(
        self,
        provider: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        from openai import OpenAI  # lazy import

        self._provider = provider
        self._model = model or _DEFAULT_MODELS.get(provider, "gpt-4o-mini")

        # Resolve base URL
        resolved_url = base_url
        if resolved_url is None:
            if provider == "gemini":
                resolved_url = _GEMINI_BASE_URL
            elif provider == "ollama":
                resolved_url = "http://localhost:11434/v1"
            elif provider == "lmstudio":
                resolved_url = "http://localhost:1234/v1"

        # Resolve API key (Ollama/LMStudio don't need one)
        resolved_key = api_key
        if resolved_key is None and provider in ("ollama", "lmstudio"):
            resolved_key = provider  # openai SDK requires a non-empty string

        self._client = OpenAI(api_key=resolved_key, base_url=resolved_url)

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
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content or ""
