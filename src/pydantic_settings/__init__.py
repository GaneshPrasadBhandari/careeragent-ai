"""Minimal local fallback for pydantic_settings APIs used in tests."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class SettingsConfigDict(dict):
    pass


class BaseSettings(BaseModel):
    model_config = SettingsConfigDict()

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
