"""Lightweight fallback shim for environments without pydantic installed.
This implements only the minimal subset used in unit tests.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class _FieldSpec:
    default: Any = None
    default_factory: Callable[[], Any] | None = None


def Field(default: Any = None, default_factory: Callable[[], Any] | None = None, **_: Any) -> Any:
    return _FieldSpec(default=default, default_factory=default_factory)


class BaseModel:
    def __init__(self, **data: Any) -> None:
        annotations = getattr(self.__class__, "__annotations__", {})
        for key in annotations:
            if key in data:
                setattr(self, key, data[key])
                continue
            raw = getattr(self.__class__, key, None)
            if isinstance(raw, _FieldSpec):
                if raw.default_factory is not None:
                    setattr(self, key, raw.default_factory())
                else:
                    setattr(self, key, deepcopy(raw.default))
            else:
                setattr(self, key, deepcopy(raw))

    @classmethod
    def model_validate(cls, value: Any):
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(**value)
        raise TypeError("model_validate expects instance or dict")

    def model_dump(self) -> dict[str, Any]:
        annotations = getattr(self.__class__, "__annotations__", {})
        out: dict[str, Any] = {}
        for key in annotations:
            val = getattr(self, key)
            if isinstance(val, BaseModel):
                out[key] = val.model_dump()
            elif isinstance(val, list):
                out[key] = [x.model_dump() if isinstance(x, BaseModel) else x for x in val]
            else:
                out[key] = val
        return out

    def model_copy(self, deep: bool = False):
        data = self.model_dump()
        return self.__class__(**(deepcopy(data) if deep else data))
