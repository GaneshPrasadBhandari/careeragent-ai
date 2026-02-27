from __future__ import annotations

import os
from typing import Optional

from pydantic import BaseModel


class Settings(BaseModel):
    """Global settings loaded from environment."""

    GEMINI_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    TAVILY_API_KEY: Optional[str] = None
    SERPER_API_KEY: Optional[str] = None

    LANGSMITH_API_KEY: Optional[str] = None
    LANGSMITH_PROJECT: str = "careeragent-ai-phase2"

    DATABASE_URL: str = "sqlite:///outputs/careeragent.db"

    QDRANT_URL: Optional[str] = None
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION: str = "careeragent_memory"

    MCP_SERVER_URL: Optional[str] = None
    JINA_READER_PREFIX: str = "https://r.jina.ai/"

    NTFY_TOPIC: Optional[str] = None
    NTFY_BASE_URL: str = "https://ntfy.sh"

    GMAIL_TO_EMAIL: Optional[str] = None
    GMAIL_FROM_EMAIL: Optional[str] = None
    GMAIL_SERVICE_ACCOUNT_JSON: Optional[str] = None
    TWILIO_ACCOUNT_SID: Optional[str] = None
    TWILIO_AUTH_TOKEN: Optional[str] = None
    TWILIO_FROM_NUMBER: Optional[str] = None
    TWILIO_TO_NUMBER: Optional[str] = None

    MAX_HTTP_SECONDS: float = 40.0

    def __init__(self, **data):
        env_data = {
            k: os.getenv(k)
            for k in [
                "GEMINI_API_KEY",
                "OPENAI_API_KEY",
                "TAVILY_API_KEY",
                "SERPER_API_KEY",
                "LANGSMITH_API_KEY",
                "LANGSMITH_PROJECT",
                "DATABASE_URL",
                "QDRANT_URL",
                "QDRANT_API_KEY",
                "QDRANT_COLLECTION",
                "MCP_SERVER_URL",
                "JINA_READER_PREFIX",
                "NTFY_TOPIC",
                "NTFY_BASE_URL",
                "GMAIL_TO_EMAIL",
                "GMAIL_FROM_EMAIL",
                "GMAIL_SERVICE_ACCOUNT_JSON",
                "TWILIO_ACCOUNT_SID",
                "TWILIO_AUTH_TOKEN",
                "TWILIO_FROM_NUMBER",
                "TWILIO_TO_NUMBER",
                "MAX_HTTP_SECONDS",
            ]
        }
        env_data = {k: v for k, v in env_data.items() if v is not None}
        merged = {**env_data, **data}
        super().__init__(**merged)


def bootstrap_langsmith(s: Settings) -> None:
    if s.LANGSMITH_API_KEY:
        os.environ.setdefault("LANGSMITH_API_KEY", s.LANGSMITH_API_KEY)
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_PROJECT", s.LANGSMITH_PROJECT)
