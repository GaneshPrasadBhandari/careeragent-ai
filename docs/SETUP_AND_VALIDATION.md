# Setup and Validation Guide (pyproject-first)

This repository uses **`pyproject.toml` + `uv.lock`** as the package source of truth.
Do not use `requirements.txt`.

## 1) Create and sync environment

```bash
# from repo root
uv venv
source .venv/bin/activate

# core + dev tooling (pytest/ruff/mypy)
uv sync --dev
```

Optional extras:

```bash
uv sync --dev --extra ui --extra infra --extra ml --extra docs
```

## 2) If your network blocks package fetches

Use one of these:

```bash
# Option A: internal mirror
export UV_INDEX_URL="https://<your-company-mirror>/simple"
uv sync --dev

# Option B: pip mirror (fallback)
export PIP_INDEX_URL="https://<your-company-mirror>/simple"
pip install -e .
```

If `hatchling` cannot be downloaded, preinstall it from your mirror first:

```bash
pip install "hatchling>=1.21"
```

## 3) Quick sanity checks

```bash
python -V
uv --version
pytest -q tests/unit/test_hitl_contract.py
python -m compileall -q src app tests
```

## 4) Full quality gates

```bash
pytest -q
ruff check src tests app *.py
mypy src/careeragent --ignore-missing-imports
```

## 5) Known first-pass fixes already made

- Added `requests>=2.32` to project dependencies in `pyproject.toml` (used by env diagnostics script).
- Added backwards-compatible state helpers in `core/state.py`:
  - `_utc_now`, `_iso_utc`, `touch`, `add_artifact`, legacy-compatible `start_step`/`end_step`.
- Fixed a runtime bug in `orchestration/engine.py` where `drafts` variable was referenced out of scope.
