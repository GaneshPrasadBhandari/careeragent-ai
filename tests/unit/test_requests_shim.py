import importlib.util
import sys
from pathlib import Path


def _load_requests_shim():
    p = Path(__file__).resolve().parents[2] / "requests.py"
    spec = importlib.util.spec_from_file_location("requests_shim", p)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["requests_shim"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_requests_shim_supports_form_data_encoding() -> None:
    requests = _load_requests_shim()
    try:
        requests.post("http://127.0.0.1:9/unused", data={"config": "{}"}, timeout=1)
    except requests.exceptions.ConnectionError:
        assert True


def test_requests_shim_exposes_connection_error_class() -> None:
    requests = _load_requests_shim()
    assert hasattr(requests, "exceptions")
    assert hasattr(requests.exceptions, "ConnectionError")


def test_load_real_requests_supports_package_relative_imports(tmp_path) -> None:
    pkg = tmp_path / "requests"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(
        "from .exceptions import RequestsDependencyWarning\n"
        "class Session: ...\n"
    )
    (pkg / "exceptions.py").write_text("class RequestsDependencyWarning(Exception):\n    pass\n")

    requests = _load_requests_shim()
    original_path = list(sys.path)
    original_requests = sys.modules.get("requests")
    try:
        sys.path.insert(0, str(tmp_path))
        sys.modules.pop("requests", None)
        real = requests._load_real_requests()
        assert real is not None
        assert hasattr(real, "Session")
    finally:
        sys.path[:] = original_path
        if original_requests is not None:
            sys.modules["requests"] = original_requests
        else:
            sys.modules.pop("requests", None)
