from __future__ import annotations

import asyncio
import importlib.util
import json
import os
from pathlib import Path
from typing import Dict, List


def _check_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _env_present(name: str) -> bool:
    return bool(os.getenv(name, "").strip())


async def _run_pipeline_smoke() -> Dict[str, object]:
    from careeragent.api import main as api_main

    run_id = "ops_check"
    api_main._runs[run_id] = api_main._build_initial_state(run_id, {})

    resume_path = Path("uploads") / "ops_check_resume.txt"
    resume_path.parent.mkdir(parents=True, exist_ok=True)
    resume_path.write_text(
        "John Doe\n"
        "Email: john@example.com\n"
        "Skills: Python, FastAPI, SQL, LangGraph, OpenAI\n"
        "Experience: Senior Engineer 2019-Present\n"
    )

    await api_main.run_pipeline(run_id, resume_path)
    state = api_main._runs[run_id]
    return {
        "run_status": state.get("status"),
        "progress_pct": state.get("progress_pct"),
        "layers": [
            {"id": layer["id"], "name": layer["name"], "status": layer["status"], "error": layer["error"]}
            for layer in state.get("layers", [])
        ],
        "jobs_discovered": state.get("jobs_discovered", 0),
        "jobs_scored": state.get("jobs_scored", 0),
        "jobs_applied": state.get("jobs_applied", 0),
        "errors": state.get("errors", []),
    }


def run_ops_check() -> Dict[str, object]:
    checks: Dict[str, object] = {}

    checks["env_keys"] = {
        "OPENAI_API_KEY": _env_present("OPENAI_API_KEY"),
        "GEMINI_API_KEY": _env_present("GEMINI_API_KEY"),
        "SERPER_API_KEY": _env_present("SERPER_API_KEY"),
        "TAVILY_API_KEY": _env_present("TAVILY_API_KEY"),
        "NTFY_TOPIC": _env_present("NTFY_TOPIC"),
    }

    checks["python_modules"] = {
        name: _check_module(name)
        for name in [
            "fastapi",
            "uvicorn",
            "streamlit",
            "requests",
            "aiohttp",
            "docx",
            "playwright",
            "pdfplumber",
        ]
    }

    checks["pipeline_smoke"] = asyncio.run(_run_pipeline_smoke())
    return checks


if __name__ == "__main__":
    report = run_ops_check()
    print(json.dumps(report, indent=2))
