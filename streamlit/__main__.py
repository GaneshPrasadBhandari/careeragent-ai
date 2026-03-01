from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> int:
    args = sys.argv[1:]
    if len(args) >= 2 and args[0] == "run":
        target = Path(args[1])
        if not target.exists():
            print(f"File not found: {target}")
            return 2
        runpy.run_path(str(target), run_name="__main__")
        print("[streamlit-shim] Executed script in compatibility mode.")
        return 0
    print("[streamlit-shim] Usage: python -m streamlit run <script>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
