from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import runpy
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _candidate_site_paths() -> list[str]:
    """Candidate import roots for a real Streamlit installation.

    Includes current interpreter paths (excluding repository-local paths that
    resolve to this shim package) and local `.venv` site-packages used by uv.
    """
    out: list[str] = []

    def _add(path_entry: str | Path | None) -> None:
        if path_entry is None:
            return
        entry = str(path_entry)
        if not entry:
            return
        if entry not in out:
            out.append(entry)

    for entry in sys.path:
        try:
            resolved = Path(entry or ".").resolve()
        except Exception:
            resolved = None
        if resolved and (resolved == REPO_ROOT or REPO_ROOT in resolved.parents):
            continue
        _add(entry)

    py_mm = f"python{sys.version_info.major}.{sys.version_info.minor}"
    for root in (REPO_ROOT / ".venv", REPO_ROOT / "venv"):
        _add(root / "lib" / py_mm / "site-packages")
        _add(root / "Lib" / "site-packages")

    env_site = os.getenv("VIRTUAL_ENV")
    if env_site:
        ve = Path(env_site)
        _add(ve / "lib" / py_mm / "site-packages")
        _add(ve / "Lib" / "site-packages")

    return out


def _resolve_real_streamlit_cli():
    """Resolve real `streamlit.web.cli` module spec if available."""
    return importlib.machinery.PathFinder.find_spec("streamlit.web.cli", _candidate_site_paths())


def _run_real_streamlit() -> int | None:
    spec = _resolve_real_streamlit_cli()
    if not spec or not spec.loader:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    main_fn = getattr(module, "main", None)
    if callable(main_fn):
        return int(main_fn() or 0)
    return None


def _run_venv_streamlit_executable() -> int | None:
    """Run real Streamlit console-script from local venv when available."""
    args = sys.argv[1:]
    if not args:
        return None
    candidates = [
        REPO_ROOT / ".venv" / "bin" / "streamlit",
        REPO_ROOT / "venv" / "bin" / "streamlit",
        REPO_ROOT / ".venv" / "Scripts" / "streamlit.exe",
        REPO_ROOT / "venv" / "Scripts" / "streamlit.exe",
    ]
    exe = next((c for c in candidates if c.exists()), None)
    if not exe:
        return None
    import subprocess
    try:
        proc = subprocess.run([str(exe), *args])
        return int(proc.returncode)
    except KeyboardInterrupt:
        return 130


def main() -> int:
    # Prefer real Streamlit unless explicitly forcing local shim mode.
    if os.getenv("CAREERAGENT_FORCE_STREAMLIT_SHIM", "").strip().lower() not in {"1", "true", "yes", "on"}:
        venv_code = _run_venv_streamlit_executable()
        if venv_code is not None:
            return venv_code
        real_code = _run_real_streamlit()
        if real_code is not None:
            return real_code

    # Fallback shim behavior for restricted environments.
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
