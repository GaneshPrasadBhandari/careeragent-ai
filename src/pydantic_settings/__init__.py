"""Compatibility bridge for pydantic-settings."""
from __future__ import annotations

import importlib.machinery
import importlib.util
from pathlib import Path
import sys
from typing import Any


def _path_without_local_src() -> list[str]:
    here = Path(__file__).resolve()
    src_dir = here.parents[1]
    repo_dir = src_dir.parent
    filtered: list[str] = []
    for entry in sys.path:
        try:
            resolved = Path(entry or ".").resolve()
        except Exception:
            filtered.append(entry)
            continue
        if resolved in {src_dir, repo_dir}:
            continue
        filtered.append(entry)
    return filtered


def _load_real() -> bool:
    spec = importlib.machinery.PathFinder.find_spec("pydantic_settings", _path_without_local_src())
    if spec is None or spec.loader is None or spec.origin is None:
        return False
    if Path(spec.origin).resolve() == Path(__file__).resolve():
        return False
    module = importlib.util.module_from_spec(spec)
    sys.modules[__name__] = module
    spec.loader.exec_module(module)
    globals().update(module.__dict__)
    return True


if not _load_real():
    from pydantic import BaseModel

    class SettingsConfigDict(dict):
        pass


    class BaseSettings(BaseModel):
        model_config = SettingsConfigDict()

        def __init__(self, **data: Any) -> None:
            super().__init__(**data)
