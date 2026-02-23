from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from careeragent.api.run_manager_service import RunManagerService
from careeragent.core.settings import Settings, bootstrap_langsmith


def create_app() -> FastAPI:
    """Description: FastAPI entry.
    Layer: L8
    Input: HTTP requests
    Output: JSON responses
    """

    s = Settings()
    bootstrap_langsmith(s)

    app = FastAPI(title="CareerAgent-AI API", version="0.3")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"] ,
        allow_headers=["*"],
    )

    rm = RunManagerService(settings=s)

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {"ok": True, "service": "careeragent-ai", "langsmith_project": s.LANGSMITH_PROJECT}

    @app.post("/analyze")
    async def analyze(
        resume: UploadFile = File(...),
        preferences_json: str = Form(...),
    ) -> JSONResponse:
        prefs = json.loads(preferences_json) if preferences_json else {}
        file_bytes = await resume.read()
        run = rm.create_run(resume_filename=resume.filename, resume_bytes=file_bytes, preferences=prefs)
        return JSONResponse(run)

    @app.get("/status/{run_id}")
    def status(run_id: str) -> JSONResponse:
        st = rm.get_state(run_id)
        if not st:
            return JSONResponse({"error": "not_found"}, status_code=404)
        return JSONResponse(st)

    @app.post("/action/{run_id}")
    async def action(run_id: str, payload: Dict[str, Any]) -> JSONResponse:
        out = rm.handle_action(run_id, payload)
        code = 200 if "error" not in out else 400
        return JSONResponse(out, status_code=code)

    return app


app = create_app()
