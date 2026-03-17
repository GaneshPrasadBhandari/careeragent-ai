import ast
import json
from datetime import datetime, timezone
from pathlib import Path


def _load_funcs():
    src = Path('src/careeragent/api/main.py').read_text(encoding='utf-8')
    mod = ast.parse(src)
    wanted = {'_sanitize_run_id', '_coerce_iso_ts', '_state_rank', '_refresh_run_state'}
    nodes = [n for n in mod.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    fn_mod = ast.Module(body=nodes, type_ignores=[])
    code = compile(fn_mod, filename='api_main_extract', mode='exec')

    class HTTPException(Exception):
        def __init__(self, status_code, detail):
            self.status_code = status_code
            self.detail = detail

    class Log:
        def debug(self, *args, **kwargs):
            return None

    scope = {
        'json': json,
        'datetime': datetime,
        'timezone': timezone,
        'Path': Path,
        'Any': __import__('typing').Any,
        'HTTPException': HTTPException,
        'log': Log(),
        '_runs': {},
        'LOGS_DIR': Path('.'),
        'RUN_ID_SAFE_PATTERN': __import__('re').compile(r'[^a-zA-Z0-9_-]'),
        'tempfile': __import__('tempfile'),
    }
    exec(code, scope)
    return scope, HTTPException


def test_refresh_prefers_more_advanced_disk_state(tmp_path):
    scope, _ = _load_funcs()
    run_id = 'r1'
    scope['LOGS_DIR'] = tmp_path
    scope['_runs'][run_id] = {
        'run_id': run_id,
        'progress_pct': 5.0,
        'agent_log': [{'msg': 'only l0'}],
        'created_at': '2026-01-01T00:00:00+00:00',
    }
    (tmp_path / f'state_{run_id}.json').write_text(json.dumps({
        'run_id': run_id,
        'progress_pct': 35.0,
        'agent_log': [{'msg': 'a'}, {'msg': 'b'}],
        'updated_at': '2026-01-01T00:10:00+00:00',
    }))

    out = scope['_refresh_run_state'](run_id)
    assert out['progress_pct'] == 35.0
    assert len(out['agent_log']) == 2


def test_refresh_prefers_more_recent_memory_when_disk_stale(tmp_path):
    scope, _ = _load_funcs()
    run_id = 'r2'
    scope['LOGS_DIR'] = tmp_path
    scope['_runs'][run_id] = {
        'run_id': run_id,
        'progress_pct': 45.0,
        'agent_log': [{'msg': 'a'}, {'msg': 'b'}, {'msg': 'c'}],
        'updated_at': '2026-01-01T00:11:00+00:00',
    }
    (tmp_path / f'state_{run_id}.json').write_text(json.dumps({
        'run_id': run_id,
        'progress_pct': 45.0,
        'agent_log': [{'msg': 'a'}],
        'updated_at': '2026-01-01T00:01:00+00:00',
    }))

    out = scope['_refresh_run_state'](run_id)
    assert out['updated_at'] == '2026-01-01T00:11:00+00:00'
    assert len(out['agent_log']) == 3


def test_refresh_raises_404_when_missing(tmp_path):
    scope, HTTPException = _load_funcs()
    scope['LOGS_DIR'] = tmp_path

    try:
        scope['_refresh_run_state']('missing')
        assert False, 'expected HTTPException'
    except HTTPException as exc:
        assert exc.status_code == 404
