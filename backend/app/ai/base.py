"""
Abstract base class for AI provider clients.
All providers implement analyze_csv_metadata and explain_model_results;
complete_json is a generic structured call used by newer features
(source-role assignment, identifier matching) and falls back to the
caller-supplied default when the provider cannot answer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import pandas as pd


class AIClient(ABC):
    @abstractmethod
    def analyze_csv_metadata(self, df: pd.DataFrame) -> dict:
        """Return AI analysis of CSV schema. Must never send full rows."""
        ...

    @abstractmethod
    def explain_model_results(
        self,
        leaderboard: List[dict],
        results: List[dict],
        dataset_summary: Optional[dict] = None,
    ) -> dict:
        """Return plain-English explanation of training results."""
        ...

    def complete_json(self, system_prompt: str, user_prompt: str, fallback: dict) -> dict:
        """Ask the model for a JSON object; return ``fallback`` if unavailable."""
        return fallback
