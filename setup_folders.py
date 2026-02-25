"""
CareerOS scaffold generator (safe / append-only)

- Creates required folders (exist_ok=True)
- Creates __init__.py where needed (only if missing)
- Creates .env (only if missing) and NEVER overwrites it
- Creates/updates .env.example (append-only, never overwrite)
- Updates .gitignore (append-only, never overwrite)
"""

from __future__ import annotations

from pathlib import Path
import os
from typing import Iterable, List


# -------------------------
# Helpers (append-only)
# -------------------------
def append_unique_lines(filepath: Path, lines: List[str]) -> None:
    """
    Append lines to a file only if they do not already exist.
    Never overwrites file contents.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.touch(exist_ok=True)

    existing = set(filepath.read_text().splitlines())
    with filepath.open("a", encoding="utf-8") as f:
        for line in lines:
            # Keep blank lines exactly as blank lines (optional formatting)
            if line == "":
                f.write("\n")
                continue
            if line not in existing:
                f.write(line + "\n")


def touch_if_missing(file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if not file_path.exists():
        file_path.touch()


# -------------------------
# Scaffold config
# -------------------------
def get_directories() -> List[str]:
    """
    Directory layout aligned with your CareerOS architecture layers
    and current repo structure. Safe to re-run.
    """
    return [
        "src/careeros/core",
        "src/careeros/core/exceptions",

        "src/careeros/intake",
        "src/careeros/parsing",
        "src/careeros/evidence",
        "src/careeros/jobs",
        "src/careeros/matching",
        "src/careeros/generation",
        "src/careeros/guardrails",
        "src/careeros/export",
        "src/careeros/audit",

        "apps/api",
        "apps/ui",

        "notebooks",

        "tests/unit",
        "tests/integration",

        "logs",
        "outputs",
        "artifacts/raw",
        "artifacts/processed",
        "exports",

        "scripts",
        "config",

        # docs paths you already use (safe to keep)
        "docs/roadmap",
        "docs/architecture",
        "docs/quality",
        "docs/runbooks",
    ]


def get_init_files(directories: Iterable[str]) -> List[Path]:
    """
    Create __init__.py in python packages only.
    For simplicity: any folder under src/ or tests/ becomes a package.
    """
    init_files: List[Path] = []
    for d in directories:
        if d.startswith("src/") or d.startswith("tests/"):
            init_files.append(Path(d) / "__init__.py")
    # Ensure top-level package init exists too
    init_files.append(Path("src/careeros/__init__.py"))
    return init_files


# -------------------------
# Hardening (append-only)
# -------------------------
def harden_gitignore() -> None:
    """
    Adds CareerOS-specific ignores without overwriting your existing .gitignore.
    """
    lines = [
        "",
        "# ===== CareerOS: runtime artifacts (do not commit) =====",
        "logs/",
        "outputs/",
        "artifacts/",
        "exports/",
        "",
        "# ===== CareerOS: MLflow local tracking (do not commit) =====",
        "mlruns/",
        "",
        "# ===== CareerOS: DVC cache/temp (do not commit) =====",
        ".dvc/cache/",
        ".dvc/tmp/",
        "",
        "# ===== CareerOS: Streamlit secrets/cache (do not commit) =====",
        ".streamlit/secrets.toml",
        ".streamlit/.cache/",
    ]
    append_unique_lines(Path(".gitignore"), lines)


def harden_env_example() -> None:
    """
    Adds required keys to .env.example in append-only mode.
    Never overwrites existing content.
    """
    lines = [
        "",
        "# ===== CareerOS Config Template (Safe to commit) =====",
        "# === LLM / GenAI ===",
        "OPENAI_API_KEY=",
        "HUGGINGFACE_API_KEY=",
        "",
        "# === Search / Scraping ===",
        "TAVILY_API_KEY=",
        "SERPER_API_KEY=",
        "",
        "# === Runtime ===",
        "ENV=dev",
        "DEMO_MODE=true",
        "ORCHESTRATION_MODE=pipeline   # pipeline | agents",
    ]
    append_unique_lines(Path(".env.example"), lines)


def create_env_if_missing() -> None:
    """
    Create .env only if missing. Never overwrite (protects real keys).
    """
    env_file = Path(".env")
    if env_file.exists():
        return

    env_contents = "\n".join(
        [
            "# Private Secrets (Do NOT commit to GitHub)",
            "# Fill real keys locally",
            "OPENAI_API_KEY=",
            "HUGGINGFACE_API_KEY=",
            "TAVILY_API_KEY=",
            "SERPER_API_KEY=",
            "",
            "ENV=dev",
            "DEMO_MODE=true",
            "ORCHESTRATION_MODE=pipeline",
            "",
        ]
    )
    env_file.write_text(env_contents, encoding="utf-8")


# -------------------------
# Main execution
# -------------------------
def create_project_structure() -> None:
    print("🏗️  Building/Updating CareerOS Project Structure (safe mode)...")

    directories = get_directories()

    # 1) Create directories (safe)
    for folder in directories:
        os.makedirs(folder, exist_ok=True)

    # 2) Create __init__.py files (only if missing)
    for init_file in get_init_files(directories):
        touch_if_missing(init_file)

    # 3) Create .env only if missing (protect secrets)
    create_env_if_missing()

    # 4) Hardening (append-only, never overwrite)
    harden_env_example()
    harden_gitignore()

    print(" CareerOS Structure Updated Successfully (no overwrites).")


def main() -> None:
    """
    This is the main entrypoint.
    When you run: `python setup_folders.py`
    it calls create_project_structure() which also hardens gitignore/env.example.
    """
    create_project_structure()


if __name__ == "__main__":
    main()
