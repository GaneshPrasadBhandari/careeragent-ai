import ast
from pathlib import Path
from urllib.parse import urlparse


def _load_funcs():
    src = Path('src/careeragent/api/main.py').read_text(encoding='utf-8')
    mod = ast.parse(src)
    wanted = {'_clean_role_title', '_build_cover_letter_text', '_is_direct_application_url'}
    nodes = [n for n in mod.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    fn_mod = ast.Module(body=nodes, type_ignores=[])
    code = compile(fn_mod, filename='api_main_extract', mode='exec')
    scope = {'urlparse': urlparse, '_now': lambda: '2026-03-10T12:00:00Z'}
    exec(code, scope)
    return scope


def test_cover_letter_uses_hiring_manager_not_domain_in_salutation():
    funcs = _load_funcs()
    text = funcs['_build_cover_letter_text'](
        {'name': 'Ganesh', 'email': 'g@example.com', 'phone': '123', 'skills': ['python']},
        {'title': 'Software Engineer @ linkedin.com', 'company': 'linkedin.com', 'url': 'https://www.linkedin.com/jobs/search/?keywords=Software+Engineer'},
    )
    assert 'Dear Hiring Manager,' in text
    assert 'Dear linkedin.com' not in text
    assert 'position with' in text


def test_is_direct_application_url_filters_search_links():
    funcs = _load_funcs()
    is_direct = funcs['_is_direct_application_url']
    assert is_direct('https://www.linkedin.com/jobs/view/123456') is True
    assert is_direct('https://www.indeed.com/viewjob?jk=abc123') is True
    assert is_direct('https://www.linkedin.com/jobs/search/?keywords=software+engineer') is False
    assert is_direct('https://www.indeed.com/jobs?q=backend+developer') is False
