from __future__ import annotations

from careeragent.core.settings import Settings, bootstrap_langsmith


def configure_runtime_env() -> Settings:
    """Load env-backed settings and activate LangSmith/LangChain tracing vars."""
    settings = Settings()
    bootstrap_langsmith(settings)
    return settings
