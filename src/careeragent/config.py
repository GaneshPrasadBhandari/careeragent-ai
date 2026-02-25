from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


def repo_root() -> Path:
    """
    Description: Resolve repository root from within src/ package.
    Layer: L0
    Input: None
    Output: Absolute Path to repo root
    """
    # src/careeragent/config.py -> src/careeragent -> src -> repo root
    return Path(__file__).resolve().parents[2]


def artifacts_root() -> Path:
    """
    Description: Canonical artifacts root required by CareerAgent-AI (local-first).
    Layer: L0
    Input: None
    Output: Path to src/careeragent/artifacts
    """
    root = Path(__file__).resolve().parents[0] / "artifacts"
    root.mkdir(parents=True, exist_ok=True)
    return root


class Settings(BaseSettings):
    """
    Description: Central configuration loader for CareerAgent-AI.
    Layer: L0
    Input: .env in repo root + environment variables
    Output: Strongly typed settings object
    """

    model_config = SettingsConfigDict(
        env_file=str(repo_root() / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Tracing
    langsmith_api_key: Optional[str] = None
    langsmith_project: str = "careeragent-ai"

    # LLM backends (local-first)
    ollama_base_url: Optional[str] = None

    # Search
    serper_api_key: Optional[str] = None

    # Notifications
    ntfy_topic: Optional[str] = None
    ntfy_auth_token: Optional[str] = None
    ntfy_endpoint: Optional[str] = None  # From number
    user_phone: Optional[str] = None    # To number for local testing

    # Runtime
    environment: str = "local"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Description: Cached settings accessor for FastAPI/Streamlit.
    Layer: L0
    Input: None
    Output: Settings
    """
    return Settings()
