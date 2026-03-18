import ast
from pathlib import Path


def _load_function(name: str):
    src = Path('app/ui/mission_control.py').read_text(encoding='utf-8')
    mod = ast.parse(src)
    fn = next(n for n in mod.body if isinstance(n, ast.FunctionDef) and n.name == name)
    fn_mod = ast.Module(body=[fn], type_ignores=[])
    code = compile(fn_mod, filename='mission_control_extract', mode='exec')

    class _Session(dict):
        pass

    class _St:
        def __init__(self):
            self.session_state = _Session()
            self.info_messages = []
            self.toast_messages = []
            self.error_messages = []
            self.warning_messages = []

        def info(self, msg):
            self.info_messages.append(msg)

        def toast(self, msg):
            self.toast_messages.append(msg)

        def error(self, msg):
            self.error_messages.append(msg)

        def warning(self, msg):
            self.warning_messages.append(msg)

    scope = {
        'st': _St(),
        'json': __import__('json'),
        'time': __import__('time'),
        'requests': type('RequestsStub', (), {'exceptions': type('Exceptions', (), {'ConnectionError': Exception, 'Timeout': Exception})})(),
        'Optional': __import__('typing').Optional,
        '_candidate_api_bases': lambda api_base: [api_base],
        '_show_connection_guard': lambda: None,
    }
    exec(code, scope)
    return scope


def test_api_start_hunt_uses_cold_start_friendly_timeout():
    scope = _load_function('_api_start_hunt')
    called = {}

    class _Resp:
        status_code = 200
        text = '{"run_id":"run_123"}'

        def json(self):
            return {'run_id': 'run_123'}

    def fake_post(endpoint, files=None, data=None, timeout=0):
        called['endpoint'] = endpoint
        called['timeout'] = timeout
        return _Resp()

    scope['requests'].post = fake_post
    out = scope['_api_start_hunt']('https://api.example.com', b'resume-bytes', 'resume.pdf', {})

    assert out == 'run_123'
    assert called['endpoint'].endswith('/hunt/start')
    assert called['timeout'] == 30
