"""CareerAgent package bootstrap.

This package defensively resolves `pydantic` from site-packages if a stray
local `src/pydantic` directory exists on a developer machine. That local
folder can shadow the real dependency and crash FastAPI imports.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from pathlib import Path


def _ensure_real_pydantic_loaded() -> None:
    if "pydantic" in sys.modules:
        return

    spec = importlib.util.find_spec("pydantic")
    origin = str(getattr(spec, "origin", "") or "") if spec else ""
    # If resolution is already fine (or pydantic not installed), do nothing.
    if not origin or "/src/pydantic" not in origin.replace('\\\\', '/'):
        return

    # Try to resolve from non-shadow paths only.
    project_root = Path(__file__).resolve().parents[1]  # .../src
    shadow_root = Path(origin).resolve().parents[1]  # .../src from shadowed import
    candidate_paths = [
        p for p in sys.path
        if p
        and Path(p).resolve() != project_root.resolve()
        and Path(p).resolve() != shadow_root.resolve()
    ]
    real_spec = importlib.machinery.PathFinder.find_spec("pydantic", candidate_paths)
    if not real_spec or not real_spec.loader:
        return

    module = importlib.util.module_from_spec(real_spec)
    # Register before execution so package-relative imports resolve correctly.
    sys.modules["pydantic"] = module
    real_spec.loader.exec_module(module)


_ensure_real_pydantic_loaded()

__all__ = []
