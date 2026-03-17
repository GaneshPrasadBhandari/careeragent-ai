import ast
from pathlib import Path


def test_run_pipeline_heartbeat_interval_is_10_seconds() -> None:
    src = Path("src/careeragent/api/main.py").read_text(encoding="utf-8")
    assert "await asyncio.sleep(10)" in src


def test_status_endpoint_supports_long_polling_heartbeat_params() -> None:
    src = Path("src/careeragent/api/main.py").read_text(encoding="utf-8")
    mod = ast.parse(src)
    status_fn = next(n for n in mod.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "get_status")
    arg_names = [a.arg for a in status_fn.args.args]
    assert "wait_for_heartbeat" in arg_names
    assert "max_wait_seconds" in arg_names
    assert "since_heartbeat" in arg_names
