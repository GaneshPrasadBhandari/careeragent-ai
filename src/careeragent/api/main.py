#FastAPI backend (Brain)

from __future__ import annotations

# --- CareerOS-style bootstrap so uvicorn can import src-layout reliably ---
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]   # repo root
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import json
from typing import Any, Dict

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from careeragent.orchestration import ENGINE


app = FastAPI(title="CareerAgent-AI Beta Brain", version="0.3.0")


@app.get("/health")
def health() -> Dict[str, Any]:
    """
    Description: Health endpoint for UI connectivity checks.
    Layer: L0
    Input: None
    Output: ok + version
    """
    return {"status": "ok", "service": "careeragent-api", "version": "0.3.0"}


class AnalyzeResponse(BaseModel):
    """
    Description: /analyze response.
    Layer: L1
    Input: resume upload + preferences
    Output: run_id + status
    """
    run_id: str
    status: str


class ActionRequest(BaseModel):
    """
    Description: HITL action payload.
    Layer: L5
    Input: action_type + payload
    Output: state transition
    """
    action_type: str = Field(..., description="approve_ranking | reject_ranking | approve_drafts | reject_drafts")
    payload: Dict[str, Any] = Field(default_factory=dict)


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    resume: UploadFile = File(...),
    preferences_json: str = Form(...),
) -> AnalyzeResponse:
    """
    Description: One-click analyze. Accepts PDF/TXT/DOCX resume + preferences. No manual resume text.
    Layer: L1
    Input: multipart form
    Output: run_id
    """
    try:
        prefs = json.loads(preferences_json)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid preferences_json: {e}")

    data = await resume.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty resume upload.")

    st = ENGINE.start_run(filename=resume.filename or "resume", data=data, prefs=prefs)
    return AnalyzeResponse(run_id=st.run_id, status=st.status)


@app.get("/status/{run_id}")
def status(run_id: str) -> Dict[str, Any]:
    """
    Description: Poll current OrchestrationState from local SQLite store.
    Layer: L1
    Input: run_id
    Output: state JSON dict
    """
    st = ENGINE.load(run_id)
    if not st:
        raise HTTPException(status_code=404, detail="run_id not found")
    return st


@app.post("/action/{run_id}")
def action(run_id: str, req: ActionRequest) -> Dict[str, Any]:
    """
    Description: Submit HITL decision and continue automation.
    Layer: L5
    Input: action_type + payload
    Output: updated state JSON
    """
    try:
        st = ENGINE.submit_action(run_id=run_id, action_type=req.action_type, payload=req.payload)
    except ValueError:
        raise HTTPException(status_code=404, detail="run_id not found")
    return st.model_dump()
