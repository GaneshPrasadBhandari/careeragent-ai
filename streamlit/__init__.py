"""Lightweight local Streamlit compatibility shim for constrained environments.

This shim is only intended for non-interactive validation where the real
`streamlit` package is unavailable. It provides permissive no-op APIs used by
this repository's dashboards so scripts can execute without ImportError.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterable, List

session_state: dict[str, Any] = {
    "api_base": "http://127.0.0.1:8000",
    "view_mode": "Pilot View",
    "live_update": False,
    "refresh_sec": 2,
    "hunt_running": False,
}


class _Placeholder:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __getattr__(self, _name: str):
        def _fn(*_args, **_kwargs):
            return None

        return _fn


class _Sidebar(_Placeholder):
    pass


sidebar = _Sidebar()


def set_page_config(*_args, **_kwargs) -> None:
    return None


def markdown(*_args, **_kwargs) -> None:
    return None


def caption(*_args, **_kwargs) -> None:
    return None


def divider() -> None:
    return None


def info(*_args, **_kwargs) -> None:
    return None


def warning(*_args, **_kwargs) -> None:
    return None


def error(*_args, **_kwargs) -> None:
    return None


def success(*_args, **_kwargs) -> None:
    return None


def metric(*_args, **_kwargs) -> None:
    return None


def json(*_args, **_kwargs) -> None:
    return None


def dataframe(*_args, **_kwargs) -> None:
    return None


def bar_chart(*_args, **_kwargs) -> None:
    return None


def download_button(*_args, **_kwargs) -> bool:
    return False


def text_input(_label: str, value: str = "", key: str | None = None, **_kwargs) -> str:
    if key:
        session_state[key] = session_state.get(key, value)
        return str(session_state[key])
    return value


def text_area(_label: str, value: str = "", **_kwargs) -> str:
    return value


def selectbox(_label: str, options: Iterable[Any], index: int = 0, key: str | None = None, **_kwargs):
    options_list = list(options)
    out = options_list[index] if options_list else None
    if key:
        session_state[key] = session_state.get(key, out)
        return session_state[key]
    return out


def checkbox(_label: str, value: bool = False, key: str | None = None, **_kwargs) -> bool:
    if key:
        session_state[key] = bool(session_state.get(key, value))
        return bool(session_state[key])
    return bool(value)


def slider(_label: str, min_value=None, max_value=None, value=None, step=None, key: str | None = None, **_kwargs):
    out = value if value is not None else min_value
    if key:
        session_state[key] = session_state.get(key, out)
        return session_state[key]
    return out


def file_uploader(*_args, **_kwargs):
    return None


def button(*_args, **_kwargs) -> bool:
    return False


def columns(spec) -> List[_Placeholder]:
    if isinstance(spec, int):
        count = spec
    else:
        count = len(list(spec))
    return [_Placeholder() for _ in range(max(1, count))]


def tabs(labels: Iterable[str]) -> List[_Placeholder]:
    return [_Placeholder() for _ in list(labels)]


def expander(*_args, **_kwargs) -> _Placeholder:
    return _Placeholder()


def empty() -> _Placeholder:
    return _Placeholder()


def rerun() -> None:
    return None


@contextmanager
def spinner(*_args, **_kwargs):
    yield
