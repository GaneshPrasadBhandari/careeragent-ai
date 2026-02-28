"""Compatibility bridge for pydantic.

Prefers the installed third-party package. Falls back to a minimal local shim
only when pydantic is unavailable in the environment.
"""
from __future__ import annotations

from copy import deepcopy
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


def _load_real_pydantic() -> bool:
    spec = importlib.machinery.PathFinder.find_spec("pydantic", _path_without_local_src())
    if spec is None or spec.loader is None or spec.origin is None:
        return False
    if Path(spec.origin).resolve() == Path(__file__).resolve():
        return False

    module = importlib.util.module_from_spec(spec)
    sys.modules[__name__] = module
    spec.loader.exec_module(module)
    globals().update(module.__dict__)
    return True


if not _load_real_pydantic():
    class _FieldSpec:
        def __init__(self, default: Any = ..., default_factory: Any = None):
            self.default = default
            self.default_factory = default_factory


    def Field(default: Any = ..., *, default_factory: Any = None, **_: Any) -> Any:
        return _FieldSpec(default=default, default_factory=default_factory)


    class ConfigDict(dict):
        pass


    class BaseModel:
        def __init__(self, **data: Any) -> None:
            annotations = {}
            for cls in reversed(self.__class__.mro()):
                annotations.update(getattr(cls, "__annotations__", {}))

            for name in annotations:
                if name in data:
                    value = data[name]
                elif hasattr(self.__class__, name):
                    raw = getattr(self.__class__, name)
                    if isinstance(raw, _FieldSpec):
                        if raw.default_factory is not None:
                            value = raw.default_factory()
                        elif raw.default is ...:
                            value = None
                        else:
                            value = deepcopy(raw.default)
                    else:
                        value = deepcopy(raw)
                else:
                    value = None
                setattr(self, name, value)

            for key, value in data.items():
                if key not in annotations:
                    setattr(self, key, value)

        @classmethod
        def model_validate(cls, data: Any):
            if isinstance(data, cls):
                return data
            if not isinstance(data, dict):
                raise TypeError("model_validate expects dict-like input")
            return cls(**data)

        def model_dump(self, mode: str | None = None) -> dict[str, Any]:
            del mode

            def _ser(v: Any) -> Any:
                if isinstance(v, BaseModel):
                    return v.model_dump()
                if isinstance(v, list):
                    return [_ser(x) for x in v]
                if isinstance(v, tuple):
                    return [_ser(x) for x in v]
                if isinstance(v, dict):
                    return {k: _ser(x) for k, x in v.items()}
                return deepcopy(v)

            return {k: _ser(v) for k, v in self.__dict__.items() if not k.startswith("_")}

        def model_copy(self, *, deep: bool = False, update: dict[str, Any] | None = None):
            payload = self.model_dump()
            if update:
                payload.update(update)
            if deep:
                payload = deepcopy(payload)
            return self.__class__(**payload)

        def dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            del args, kwargs
            return self.model_dump()


    def create_model(__model_name: str, __base__: type[BaseModel] | None = None, **field_definitions: Any):
        base_cls = __base__ or BaseModel
        annotations: dict[str, Any] = {}
        attrs: dict[str, Any] = {}

        for field_name, field_spec in field_definitions.items():
            if isinstance(field_spec, tuple) and len(field_spec) == 2:
                field_type, default = field_spec
            else:
                field_type, default = field_spec, None

            annotations[field_name] = field_type
            attrs[field_name] = default

        attrs["__annotations__"] = annotations
        return type(__model_name, (base_cls,), attrs)
