import ast
from pathlib import Path


def _load_api_get_status():
    src = Path("app/ui/mission_control.py").read_text(encoding="utf-8")
    mod = ast.parse(src)
    fn = next(n for n in mod.body if isinstance(n, ast.FunctionDef) and n.name == "_api_get_status")
    fn_mod = ast.Module(body=[fn], type_ignores=[])
    code = compile(fn_mod, filename="mission_control_extract", mode="exec")

    class _Session(dict):
        pass

    class _St:
        def __init__(self):
            self.session_state = _Session()

    scope = {
        "st": _St(),
        "quote_plus": __import__("urllib.parse", fromlist=["quote_plus"]).quote_plus,
        "Optional": __import__("typing").Optional,
    }
    exec(code, scope)
    return scope


def test_api_get_status_uses_long_polling_and_tracks_heartbeat():
    scope = _load_api_get_status()
    called = {}

    def fake_api_get(_base, path, timeout=0):
        called["path"] = path
        called["timeout"] = timeout
        return {"status": "running", "last_heartbeat_at": "2026-01-01T00:00:10+00:00", "layers": []}

    scope["_api_get"] = fake_api_get
    out = scope["_api_get_status"]("https://api.example.com", "run_123")

    assert out is not None
    assert "wait_for_heartbeat=1" in called["path"]
    assert "max_wait_seconds=12" in called["path"]
    assert called["timeout"] == 15
    assert scope["st"].session_state["last_heartbeat_at"] == "2026-01-01T00:00:10+00:00"
