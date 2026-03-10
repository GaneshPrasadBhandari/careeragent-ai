import os

from careeragent.core.settings import Settings, bootstrap_langsmith


def test_bootstrap_langsmith_sets_both_tracing_flags(monkeypatch):
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    s = Settings(LANGSMITH_API_KEY="k", LANGSMITH_PROJECT="careeragent-ai", LANGCHAIN_PROJECT="legacy", LANGSMITH_TRACING="true")

    bootstrap_langsmith(s)

    assert "LANGSMITH_API_KEY" in os.environ
    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
    assert os.environ["LANGSMITH_PROJECT"] == "careeragent-ai"
    assert os.environ["LANGCHAIN_PROJECT"] == "careeragent-ai"
