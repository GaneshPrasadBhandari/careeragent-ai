"""Compatibility entrypoint for users running `uvicorn api_main:app`.

Always forwards to the canonical FastAPI app at `careeragent.api.main` so
all HITL/job-selection/langsmith fixes are consistently available.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
_SRC = _REPO_ROOT / "src"
for p in (str(_REPO_ROOT), str(_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from careeragent.api.main import app  # noqa: E402,F401
