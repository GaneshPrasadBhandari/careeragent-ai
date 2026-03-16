import ast
import json
from pathlib import Path


def _load_functions():
    src = Path('app/ui/mission_control.py').read_text(encoding='utf-8')
    mod = ast.parse(src)
    wanted = {'_default_api_base', '_resolve_api_base', '_candidate_api_bases', '_api_health', '_api_start_hunt'}
    nodes = [n for n in mod.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    fn_mod = ast.Module(body=nodes, type_ignores=[])
    code = compile(fn_mod, filename='mission_control_extract', mode='exec')

    class _St:
        def __init__(self):
            self.errors = []
            self.infos = []
            self.toasts = []

        def error(self, message):
            self.errors.append(str(message))

        def info(self, message):
            self.infos.append(str(message))

        def toast(self, message):
            self.toasts.append(str(message))

    parse = __import__('urllib.parse', fromlist=['urlparse', 'urlunparse'])
    scope = {
        'json': json,
        'os': __import__('os'),
        'time': __import__('time'),
        'st': _St(),
        'Optional': __import__('typing').Optional,
        'urlparse': parse.urlparse,
        'urlunparse': parse.urlunparse,
    }
    exec(code, scope)
    return scope


def test_resolve_api_base_rewrites_dashboard_hostname_and_paths():
    scope = _load_functions()
    resolve = scope['_resolve_api_base']
    assert resolve('careeragent-dashboard.onrender.com/health') == 'https://careeragent-api.onrender.com'


def test_default_api_base_infers_api_from_frontend_service_name(monkeypatch):
    scope = _load_functions()
    default = scope['_default_api_base']
    monkeypatch.delenv('API_BASE_URL', raising=False)
    monkeypatch.delenv('API_URL', raising=False)
    monkeypatch.delenv('BACKEND_URL', raising=False)
    monkeypatch.delenv('API_HOSTPORT', raising=False)
    monkeypatch.setenv('RENDER_SERVICE_NAME', 'careeragent-frontend')
    monkeypatch.delenv('RENDER_EXTERNAL_URL', raising=False)

    assert default() == 'https://careeragent-api.onrender.com'




def test_resolve_api_base_uses_http_for_localhost_without_scheme():
    scope = _load_functions()
    resolve = scope['_resolve_api_base']
    assert resolve('localhost:8000') == 'http://localhost:8000'



def test_api_health_tries_raw_input_when_normalized_base_fails():
    scope = _load_functions()

    calls = []

    def fake_get(api_base, path, timeout=0):
        calls.append(api_base)
        if api_base == 'https://careeragent-api.onrender.com':
            return None
        if api_base == 'https://careeragent-dashboard.onrender.com':
            return {'status': 'ok'}
        return None

    scope['_api_get'] = fake_get
    assert scope['_api_health']('https://careeragent-dashboard.onrender.com') is True
    assert 'https://careeragent-api.onrender.com' in calls
    assert 'https://careeragent-dashboard.onrender.com' in calls
def test_api_health_retries_before_false(monkeypatch):
    scope = _load_functions()
    calls = {'n': 0}

    def fake_get(api_base, path, timeout=0):
        calls['n'] += 1
        return None if calls['n'] == 1 else {'status': 'ok'}

    scope['_api_get'] = fake_get
    assert scope['_api_health']('http://x') is True
    assert calls['n'] == 2


def test_api_start_hunt_retries_and_succeeds(monkeypatch):
    scope = _load_functions()

    class Resp:
        def __init__(self, code, payload=None, text=''):
            self.status_code = code
            self._payload = payload or {}
            self.text = text

        def json(self):
            return self._payload

    class RequestsStub:
        def __init__(self):
            self.calls = 0

        def get(self, *args, **kwargs):
            return Resp(200, payload={'status': 'ok'})

        def post(self, *args, **kwargs):
            self.calls += 1
            if self.calls < 3:
                return Resp(502, text='bad gateway')
            return Resp(200, payload={'run_id': 'run_ok'})

    scope['requests'] = RequestsStub()
    run_id = scope['_api_start_hunt']('http://api', b'data', 'resume.pdf', {'a': 1})
    assert run_id == 'run_ok'
    assert scope['requests'].calls == 3


def test_api_start_hunt_reports_error_after_retries(monkeypatch):
    scope = _load_functions()

    class Resp:
        def __init__(self, code, text=''):
            self.status_code = code
            self.text = text

        def json(self):
            return {}

    class RequestsStub:
        class exceptions:
            ConnectionError = RuntimeError

        def get(self, *args, **kwargs):
            return Resp(503, text='service unavailable')

        def post(self, *args, **kwargs):
            return Resp(503, text='service unavailable')

    scope['requests'] = RequestsStub()
    run_id = scope['_api_start_hunt']('http://api', b'data', 'resume.pdf', {'a': 1})
    assert run_id is None
    assert any('Backend error 503' in msg for msg in scope['st'].errors)


def test_api_start_hunt_handles_initializing_signal_then_succeeds():
    scope = _load_functions()

    class Resp:
        def __init__(self, code, payload=None, text=''):
            self.status_code = code
            self._payload = payload or {}
            self.text = text or json.dumps(self._payload)

        def json(self):
            return self._payload

    class RequestsStub:
        def __init__(self):
            self.calls = 0

        def get(self, *args, **kwargs):
            return Resp(200, payload={'status': 'ok'})

        def post(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return Resp(503, payload={'status': 'initializing', 'retry_after': 1})
            return Resp(200, payload={'run_id': 'run_after_warm'})

    scope['requests'] = RequestsStub()
    run_id = scope['_api_start_hunt']('http://api', b'data', 'resume.pdf', {'a': 1})
    assert run_id == 'run_after_warm'
    assert any('Warming Up' in msg for msg in scope['st'].infos)
