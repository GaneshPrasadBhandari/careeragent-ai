import ast
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def _load_funcs():
    src = Path("src/careeragent/api/main.py").read_text(encoding="utf-8")
    mod = ast.parse(src)
    wanted = {"_sanitize_run_id", "_coerce_iso_ts", "_state_rank", "_refresh_run_state"}
    nodes = [n for n in mod.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    fn_mod = ast.Module(body=nodes, type_ignores=[])
    code = compile(fn_mod, filename="api_main_extract", mode="exec")

    class HTTPException(Exception):
        def __init__(self, status_code, detail):
            self.status_code = status_code
            self.detail = detail

    class Log:
        def debug(self, *args, **kwargs):
            return None

    scope = {
        "json": json,
        "datetime": datetime,
        "timezone": timezone,
        "Path": Path,
        "tempfile": tempfile,
        "Any": __import__("typing").Any,
        "HTTPException": HTTPException,
        "log": Log(),
        "_runs": {},
        "LOGS_DIR": Path("."),
        "RUN_ID_SAFE_PATTERN": __import__("re").compile(r"[^a-zA-Z0-9_-]"),
        "fcntl": __import__("fcntl"),
    }
    exec(code, scope)
    return scope


def test_refresh_run_state_reads_tmp_state_backup(tmp_path):
    scope = _load_funcs()
    run_id = "abc123"
    scope["LOGS_DIR"] = tmp_path
    tmp_state = Path(tempfile.gettempdir()) / f"state_{run_id}.json"
    tmp_lock = tmp_state.with_suffix(".lock")
    tmp_state.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "status": "running",
                "progress_pct": 22.0,
                "agent_log": [{"msg": "tmp backup"}],
                "updated_at": "2026-01-01T00:15:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    try:
        out = scope["_refresh_run_state"](run_id)
        assert out["run_id"] == run_id
        assert out["progress_pct"] == 22.0
    finally:
        tmp_state.unlink(missing_ok=True)
        tmp_lock.unlink(missing_ok=True)


def test_refresh_run_state_sanitizes_run_id(tmp_path):
    scope = _load_funcs()
    scope["LOGS_DIR"] = tmp_path
    bad_run_id = "../abc123??"
    clean_run_id = "abc123"
    (tmp_path / f"state_{clean_run_id}.json").write_text(
        json.dumps(
            {
                "run_id": clean_run_id,
                "status": "running",
                "progress_pct": 7.0,
                "agent_log": [{"msg": "ok"}],
                "updated_at": "2026-01-01T00:01:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    out = scope["_refresh_run_state"](bad_run_id)
    assert out["run_id"] == clean_run_id

