import os
import importlib.util
import sys
from pathlib import Path


def _load_api_main_module():
    p = Path(__file__).resolve().parents[2] / "api_main.py"
    spec = importlib.util.spec_from_file_location("api_main_entry_failsafe", p)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["api_main_entry_failsafe"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_disable_langchain_tracing_due_to_langsmith_auth_error(monkeypatch) -> None:
    api_main = _load_api_main_module()
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")

    api_main._disable_langchain_tracing_due_to_error(Exception("langsmith.client unauthorized 401"))

    assert os.environ.get("LANGCHAIN_TRACING_V2") == "false"
    assert os.environ.get("LANGSMITH_TRACING") == "false"


def test_disable_langchain_tracing_ignores_unrelated_errors(monkeypatch) -> None:
    api_main = _load_api_main_module()
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")

    api_main._disable_langchain_tracing_due_to_error(RuntimeError("database unavailable"))

    assert os.environ.get("LANGCHAIN_TRACING_V2") == "true"
    assert os.environ.get("LANGSMITH_TRACING") == "true"
