import ast
import os
from pathlib import Path
from urllib.parse import quote_plus
import re


def _load_langsmith_status_func():
    src = Path('src/careeragent/api/main.py').read_text(encoding='utf-8')
    mod = ast.parse(src)
    fn_node = next(n for n in mod.body if isinstance(n, ast.FunctionDef) and n.name == '_langsmith_status')
    fn_mod = ast.Module(body=[fn_node], type_ignores=[])
    code = compile(fn_mod, filename='langsmith_status_extract', mode='exec')
    scope = {'os': os, 're': re, 'quote_plus': quote_plus}
    exec(code, scope)
    return scope['_langsmith_status']


def test_langsmith_status_uses_safe_project_link_without_invalid_workspace(monkeypatch):
    langsmith_status = _load_langsmith_status_func()
    monkeypatch.setenv('LANGCHAIN_TRACING_V2', 'true')
    monkeypatch.setenv('LANGSMITH_API_KEY', 'key')
    monkeypatch.setenv('LANGSMITH_PROJECT', 'careeragent-ai')
    monkeypatch.setenv('LANGSMITH_WORKSPACE_ID', 'careeragent-ai')

    status = langsmith_status('run_123')

    assert status['enabled'] is True
    assert status['workspace'] is None
    assert status['dashboard_url'] == 'https://smith.langchain.com/projects?name=careeragent-ai'
