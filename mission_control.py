"""Compatibility entrypoint for users running `streamlit run mission_control.py`.

Always forwards to the canonical UI implementation in `app.ui.mission_control`.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
_SRC = _REPO_ROOT / "src"
for p in (str(_REPO_ROOT), str(_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.ui.mission_control import main  # noqa: E402

if __name__ == "__main__":
    main()
