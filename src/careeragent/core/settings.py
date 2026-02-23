from __future__ import annotations

from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Description: Global settings loaded from .env.
    Layer: L0
    Input: environment
    Output: typed settings
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Keys
    GEMINI_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    TAVILY_API_KEY: Optional[str] = None
    SERPER_API_KEY: Optional[str] = None

    # LangSmith
    LANGSMITH_API_KEY: Optional[str] = None
    LANGSMITH_PROJECT: str = "careeragent-ai-phase2"

    # Storage
    DATABASE_URL: str = "sqlite:///outputs/careeragent.db"

    # Qdrant
    QDRANT_URL: Optional[str] = None
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION: str = "careeragent_memory"

    # MCP
    MCP_SERVER_URL: Optional[str] = None

    # Jina Reader
    JINA_READER_PREFIX: str = "https://r.jina.ai/"

    # Limits
    MAX_HTTP_SECONDS: float = 40.0


def bootstrap_langsmith(s: Settings) -> None:
    """Description: Enable LangChain/LangSmith tracing via env.
    Layer: L0
    Input: Settings
    Output: environment flags
    """

    # We avoid hard dependency on langsmith package; LangChain uses env flags.
    import os

    if s.LANGSMITH_API_KEY:
        os.environ.setdefault("LANGSMITH_API_KEY", s.LANGSMITH_API_KEY)
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_PROJECT", s.LANGSMITH_PROJECT)
