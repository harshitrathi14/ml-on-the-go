"""
Application configuration using Pydantic Settings.

Manages environment-specific settings for the ML platform.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "ML on the Go"
    app_version: str = "1.0.0"
    app_env: str = Field(default="development", alias="APP_ENV")
    debug: bool = False

    # API
    api_prefix: str = "/api/v1"
    cors_origins: List[str] = ["*"]

    # Data generation defaults
    default_n_rows: int = 100_000
    default_n_features: int = 100
    default_seed: int = 42
    default_rate: float = 0.15  # Default class imbalance

    # Model training
    default_cv_folds: int = 5
    default_tuning_trials: int = 50
    tuning_timeout: Optional[int] = 600  # seconds

    # Storage paths
    artifacts_path: str = "./artifacts"
    models_path: str = "./artifacts/models"
    data_path: str = "./artifacts/data"

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"  # json or text


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
