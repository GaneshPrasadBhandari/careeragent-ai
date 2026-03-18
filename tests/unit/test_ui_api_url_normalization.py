import ast
from pathlib import Path
from urllib.parse import urlparse, urlunparse


def _load_function(path: str, name: str):
    src = Path(path).read_text(encoding='utf-8')
    mod = ast.parse(src)
    fn_node = next(node for node in mod.body if isinstance(node, ast.FunctionDef) and node.name == name)
    code = compile(ast.Module(body=[fn_node], type_ignores=[]), filename=path, mode='exec')
    scope = {
        'os': __import__('os'),
        'urlparse': urlparse,
        'urlunparse': urlunparse,
    }
    exec(code, scope)
    return scope[name]


def test_mission_control_normalizer_repairs_scheme_typos_and_trailing_slashes(monkeypatch):
    resolve = _load_function('app/ui/mission_control.py', '_resolve_api_base')
    monkeypatch.setenv('BACKEND_URL', 'https://careeragent-api.onrender.com')

    assert resolve('tips://careeragent-dashboard.onrender.com/health/') == 'https://careeragent-api.onrender.com'
    assert resolve('careeragent-dashboard.onrender.com/hunt/start/') == 'https://careeragent-api.onrender.com'
    assert resolve('localhost:8000/') == 'http://localhost:8000'


def test_dashboard_normalizer_repairs_scheme_typos_and_trailing_slashes(monkeypatch):
    normalize = _load_function('app/ui/dashboard.py', '_normalize_api_base')
    monkeypatch.setenv('API_URL', 'https://careeragent-api.onrender.com')

    assert normalize('tips://careeragent-dashboard.onrender.com/docs/') == 'https://careeragent-api.onrender.com'
    assert normalize('careeragent-dashboard.onrender.com/hunt/start/') == 'https://careeragent-api.onrender.com'
    assert normalize('127.0.0.1:8000/') == 'http://127.0.0.1:8000'
