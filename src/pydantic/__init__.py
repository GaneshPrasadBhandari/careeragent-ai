"""Minimal local fallback for pydantic APIs used in tests.

This module is only intended for environments where pydantic is unavailable.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any


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
    """Minimal runtime model factory compatible with pydantic.create_model.

    Supports the common field declaration forms used by libraries:
    - `name=type` (default `None`)
    - `name=(type, default)`
    """
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
