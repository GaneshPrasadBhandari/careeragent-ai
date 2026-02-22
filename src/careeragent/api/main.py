# #FastAPI backend (Brain)

# from __future__ import annotations

# # --- CareerOS-style bootstrap so uvicorn can import src-layout reliably ---
# import sys
# from pathlib import Path

# ROOT_DIR = Path(__file__).resolve().parents[3]   # repo root
# SRC_DIR = ROOT_DIR / "src"
# if str(SRC_DIR) not in sys.path:
#     sys.path.insert(0, str(SRC_DIR))

# import json
# from typing import Any, Dict

# from fastapi import FastAPI, File, Form, HTTPException, UploadFile
# from pydantic import BaseModel, Field

# from careeragent.orchestration import ENGINE


# app = FastAPI(title="CareerAgent-AI Beta Brain", version="0.3.0")


# @app.get("/health")
# def health() -> Dict[str, Any]:
#     """
#     Description: Health endpoint for UI connectivity checks.
#     Layer: L0
#     Input: None
#     Output: ok + version
#     """
#     return {"status": "ok", "service": "careeragent-api", "version": "0.3.0"}


# class AnalyzeResponse(BaseModel):
#     """
#     Description: /analyze response.
#     Layer: L1
#     Input: resume upload + preferences
#     Output: run_id + status
#     """
#     run_id: str
#     status: str


# class ActionRequest(BaseModel):
#     """
#     Description: HITL action payload.
#     Layer: L5
#     Input: action_type + payload
#     Output: state transition
#     """
#     action_type: str = Field(..., description="approve_ranking | reject_ranking | approve_drafts | reject_drafts")
#     payload: Dict[str, Any] = Field(default_factory=dict)


# @app.post("/analyze", response_model=AnalyzeResponse)
# async def analyze(
#     resume: UploadFile = File(...),
#     preferences_json: str = Form(...),
# ) -> AnalyzeResponse:
#     """
#     Description: One-click analyze. Accepts PDF/TXT/DOCX resume + preferences. No manual resume text.
#     Layer: L1
#     Input: multipart form
#     Output: run_id
#     """
#     try:
#         prefs = json.loads(preferences_json)
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Invalid preferences_json: {e}")

#     data = await resume.read()
#     if not data:
#         raise HTTPException(status_code=400, detail="Empty resume upload.")

#     st = ENGINE.start_run(filename=resume.filename or "resume", data=data, prefs=prefs)
#     return AnalyzeResponse(run_id=st.run_id, status=st.status)


# @app.get("/status/{run_id}")
# def status(run_id: str) -> Dict[str, Any]:
#     """
#     Description: Poll current OrchestrationState from local SQLite store.
#     Layer: L1
#     Input: run_id
#     Output: state JSON dict
#     """
#     st = ENGINE.load(run_id)
#     if not st:
#         raise HTTPException(status_code=404, detail="run_id not found")
#     return st


# @app.post("/action/{run_id}")
# def action(run_id: str, req: ActionRequest) -> Dict[str, Any]:
#     """
#     Description: Submit HITL decision and continue automation.
#     Layer: L5
#     Input: action_type + payload
#     Output: updated state JSON
#     """
#     try:
#         st = ENGINE.submit_action(run_id=run_id, action_type=req.action_type, payload=req.payload)
#     except ValueError:
#         raise HTTPException(status_code=404, detail="run_id not found")
#     return st.model_dump()



from __future__ import annotations

import json
import os
from typing import Any, Dict

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from careeragent.api.request_models import AnalyzeResponse, ActionRequest
from careeragent.api.run_manager_service import RunManagerService

app = FastAPI(title="CareerAgent-AI Beta Brain", version="0.3.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RM = RunManagerService()


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "title": app.title, "version": app.version}


@app.get("/debug/whoami")
def whoami() -> Dict[str, Any]:
    return {
        "pid": os.getpid(),
        "api_main_file": __file__,
        "title": app.title,
        "version": app.version,
    }


def _extract_text(filename: str, data: bytes) -> str:
    name = (filename or "").lower()
    if name.endswith(".txt"):
        return data.decode("utf-8", errors="replace")
    if name.endswith(".pdf"):
        from pypdf import PdfReader  # type: ignore
        import io
        reader = PdfReader(io.BytesIO(data))
        return "\n".join([(pg.extract_text() or "") for pg in reader.pages])
    if name.endswith(".docx"):
        import docx  # type: ignore
        import io
        d = docx.Document(io.BytesIO(data))
        return "\n".join([p.text for p in d.paragraphs if p.text])
    return data.decode("utf-8", errors="replace")


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(resume: UploadFile = File(...), preferences_json: str = Form(...)) -> AnalyzeResponse:
    try:
        prefs = json.loads(preferences_json)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid preferences_json")

    data = await resume.read()
    text = _extract_text(resume.filename or "resume.txt", data)

    st = RM.create_run(
        resume_filename=resume.filename or "resume.txt",
        resume_text=text,
        resume_bytes=data,
        preferences=prefs,
    )
    RM.start_background(st["run_id"])
    return AnalyzeResponse(run_id=st["run_id"], status=st["status"])


@app.get("/status/{run_id}")
def status(run_id: str) -> Dict[str, Any]:
    st = RM.get_state(run_id)
    if not st:
        raise HTTPException(status_code=404, detail="run_id not found")
    return st


@app.post("/action/{run_id}")
async def action(run_id: str, req: ActionRequest) -> Dict[str, Any]:
    try:
        return await RM.handle_action(run_id=run_id, action_type=req.action_type, payload=req.payload)
    except ValueError:
        raise HTTPException(status_code=404, detail="run_id not found")
