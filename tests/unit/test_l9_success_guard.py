import ast
from pathlib import Path


def _load_guard():
    src = Path('src/careeragent/api/main.py').read_text(encoding='utf-8')
    mod = ast.parse(src)
    fn = next(n for n in mod.body if isinstance(n, ast.FunctionDef) and n.name == '_force_success_if_l9_reached')
    code = compile(ast.Module(body=[fn], type_ignores=[]), filename='api_main_extract', mode='exec')
    scope = {}
    exec(code, scope)
    return scope['_force_success_if_l9_reached']


def test_l9_ok_forces_completed_status():
    guard = _load_guard()
    state = {
        'status': 'failed',
        'pending_action': 'approve_followups',
        'progress_pct': 92.0,
        'layers': [{} for _ in range(10)],
    }
    state['layers'][9] = {'status': 'ok'}
    out = guard(state)
    assert out['status'] == 'completed'
    assert out['progress_pct'] == 100.0
    assert out['pending_action'] is None


def test_non_l9_state_is_unchanged():
    guard = _load_guard()
    state = {'status': 'running', 'progress_pct': 80.0, 'layers': [{} for _ in range(10)]}
    out = guard(state)
    assert out['status'] == 'running'
    assert out['progress_pct'] == 80.0
