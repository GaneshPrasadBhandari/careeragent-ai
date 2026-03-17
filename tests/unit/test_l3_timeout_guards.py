import ast
from pathlib import Path


def _read_const(path: str, name: str):
    mod = ast.parse(Path(path).read_text(encoding='utf-8'))
    for node in mod.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.get_source_segment(Path(path).read_text(encoding='utf-8'), node)
    return ''


def test_discovery_timeouts_are_tightened_for_non_hanging_flow():
    api_decl = _read_const('src/careeragent/api/main.py', 'DISCOVERY_TIMEOUT_SECONDS')
    scout_decl = _read_const('src/careeragent/managers/leadscout_service.py', 'SEARCH_TASK_TIMEOUT_SECONDS')
    assert '"90"' in api_decl
    assert '"45.0"' in scout_decl
