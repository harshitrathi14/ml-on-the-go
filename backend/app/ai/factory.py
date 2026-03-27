"""
AI client factory.

Auto-detection order (when AI_PROVIDER env var is not set):
  1. ANTHROPIC_API_KEY  → ClaudeClient
  2. OPENAI_API_KEY     → OpenAICompatClient(openai)
  3. GEMINI_API_KEY     → OpenAICompatClient(gemini)
  4. fallback           → NoOpAIClient (Python-only, no external calls)

When AI_PROVIDER is set explicitly it overrides auto-detection.
Ollama and LM Studio are selected by setting AI_PROVIDER=ollama or AI_PROVIDER=lmstudio
(no API key needed; they run locally).
"""

from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import Settings
    from .base import AIClient

logger = logging.getLogger(__name__)


def get_ai_client(settings: "Settings") -> "AIClient":
    provider = (settings.ai_provider or "").lower().strip()
    model_override: Optional[str] = settings.ai_model

    # --- Explicit provider selection ---
    if provider == "claude":
        return _make_claude(settings.anthropic_api_key, model_override)

    if provider == "openai":
        return _make_openai_compat("openai", settings.openai_api_key, None, model_override)

    if provider == "gemini":
        return _make_openai_compat("gemini", settings.gemini_api_key, None, model_override)

    if provider == "ollama":
        return _make_openai_compat(
            "ollama", None, f"{settings.ollama_base_url.rstrip('/')}/v1", model_override
        )

    if provider == "lmstudio":
        return _make_openai_compat(
            "lmstudio", None, f"{settings.lmstudio_base_url.rstrip('/')}/v1", model_override
        )

    if provider == "none":
        logger.info("AI_PROVIDER=none — using Python-only NoOpAIClient")
        return _make_noop()

    # --- Auto-detection ---
    if settings.anthropic_api_key:
        logger.info("AI provider auto-detected: Claude (ANTHROPIC_API_KEY present)")
        return _make_claude(settings.anthropic_api_key, model_override)

    if settings.openai_api_key:
        logger.info("AI provider auto-detected: OpenAI (OPENAI_API_KEY present)")
        return _make_openai_compat("openai", settings.openai_api_key, None, model_override)

    if settings.gemini_api_key:
        logger.info("AI provider auto-detected: Gemini (GEMINI_API_KEY present)")
        return _make_openai_compat("gemini", settings.gemini_api_key, None, model_override)

    logger.info("No AI provider configured — using Python-only NoOpAIClient")
    return _make_noop()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _make_claude(api_key: Optional[str], model: Optional[str]) -> "AIClient":
    if not api_key:
        logger.warning("AI_PROVIDER=claude but ANTHROPIC_API_KEY not set — falling back to NoOp")
        return _make_noop()
    try:
        from .claude_client import ClaudeClient
        return ClaudeClient(api_key=api_key, model=model)
    except ImportError:
        logger.warning("anthropic package not installed — falling back to NoOpAIClient")
        return _make_noop()


def _make_openai_compat(
    provider: str,
    api_key: Optional[str],
    base_url: Optional[str],
    model: Optional[str],
) -> "AIClient":
    try:
        from .openai_compat import OpenAICompatClient
        return OpenAICompatClient(provider=provider, api_key=api_key, base_url=base_url, model=model)
    except ImportError:
        logger.warning("openai package not installed — falling back to NoOpAIClient")
        return _make_noop()


def _make_noop() -> "AIClient":
    from .noop_client import NoOpAIClient
    return NoOpAIClient()
