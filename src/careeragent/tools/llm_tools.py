from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

import httpx

from careeragent.core.settings import Settings


# Optional LangSmith tracing.
try:  # pragma: no cover
    from langsmith.run_helpers import traceable  # type: ignore
except Exception:  # pragma: no cover
    def traceable(*args, **kwargs):  # type: ignore
        def _wrap(fn):
            return fn

        return _wrap


class GeminiClient:
    """Description: Minimal Gemini REST client (no SDK dependency).
    Layer: L2-L7
    """

    def __init__(self, settings: Settings, model: str = "gemini-1.5-flash") -> None:
        self.s = settings
        self.model = model

    @traceable(name="gemini.generate_json")
    def generate_json(self, prompt: str, *, temperature: float = 0.2, max_tokens: int = 1200) -> Optional[Dict[str, Any]]:
        if not self.s.GEMINI_API_KEY:
            return None
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.s.GEMINI_API_KEY}"
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": int(max_tokens)},
        }
        with httpx.Client(timeout=self.s.MAX_HTTP_SECONDS) as client:
            r = client.post(url, json=payload)
        if r.status_code >= 400:
            return None
        j = r.json()
        txt = ((((j.get("candidates") or [{}])[0].get("content") or {}).get("parts") or [{}])[0].get("text") or "")
        m = re.search(r"\{.*\}", txt, flags=re.S)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None

    @traceable(name="gemini.generate_text")
    def generate_text(self, prompt: str, *, temperature: float = 0.4, max_tokens: int = 1400) -> Optional[str]:
        if not self.s.GEMINI_API_KEY:
            return None
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.s.GEMINI_API_KEY}"
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": int(max_tokens)},
        }
        with httpx.Client(timeout=self.s.MAX_HTTP_SECONDS) as client:
            r = client.post(url, json=payload)
        if r.status_code >= 400:
            return None
        j = r.json()
        return ((((j.get("candidates") or [{}])[0].get("content") or {}).get("parts") or [{}])[0].get("text") or "").strip() or None
