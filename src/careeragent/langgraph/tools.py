from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import httpx
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class ToolSettings(BaseSettings):
    """
    Description: Central settings for tools/APIs.
    Layer: L0
    Input: .env
    Output: typed settings
    """
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    SERPER_API_KEY: Optional[str] = None
    FIRECRAWL_API_KEY: Optional[str] = None

    MCP_SERVER_URL: Optional[str] = None
    MCP_API_KEY: Optional[str] = None

    OLLAMA_BASE_URL: Optional[str] = None
    OLLAMA_MODEL: str = "llama3.2"

    # Google Gemini (recommended default cloud LLM)
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-1.5-flash"

    # Tavily (Corrective RAG)
    TAVILY_API_KEY: Optional[str] = None

    OPENAI_API_KEY: Optional[str] = None  # optional


@dataclass
class ToolResult:
    """
    Description: Standard tool call output with confidence.
    Layer: L0
    """
    ok: bool
    confidence: float
    data: Any = None
    error: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


class MCPClient:
    """
    Description: MCP client for high-fidelity extraction/search when standard tools fail.
    Layer: L0-L4
    Input: method + payload
    Output: MCP response
    """

    def __init__(self, base_url: Optional[str], api_key: Optional[str]) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key

    def available(self) -> bool:
        return bool(self.base_url and self.api_key)

    async def invoke(self, *, tool: str, payload: Dict[str, Any], timeout: float = 30.0) -> ToolResult:
        if not self.available():
            return ToolResult(ok=False, confidence=0.0, error="MCP not configured")
        try:
            url = f"{self.base_url}/invoke"
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.post(url, headers=headers, json={"tool": tool, "payload": payload})
            if r.status_code >= 400:
                return ToolResult(ok=False, confidence=0.0, error=f"MCP {r.status_code}: {r.text[:200]}")
            return ToolResult(ok=True, confidence=0.85, data=r.json())
        except Exception as e:
            return ToolResult(ok=False, confidence=0.0, error=str(e))


async def serper_search(settings: ToolSettings, query: str, num: int = 10, tbs: Optional[str] = None) -> ToolResult:
    """
    Description: Serper search tool.
    Layer: L3
    """
    if not settings.SERPER_API_KEY:
        return ToolResult(ok=False, confidence=0.0, error="SERPER_API_KEY missing")
    try:
        headers = {"X-API-KEY": settings.SERPER_API_KEY, "Content-Type": "application/json"}
        body: Dict[str, Any] = {"q": query, "num": num}
        if tbs:
            body["tbs"] = tbs
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post("https://google.serper.dev/jobs", headers=headers, json=body)
        if r.status_code == 403:
            return ToolResult(ok=False, confidence=0.0, error="Serper quota/403", meta={"status": 403})
        if r.status_code >= 400:
            return ToolResult(ok=False, confidence=0.0, error=f"Serper {r.status_code}")
        data = r.json()
        jobs = (data.get("jobs") or [])
        out = [
            {
                "title":   x.get("title") or "",
                "link":    x.get("applyOptions", [{}])[0].get("link") or x.get("link") or "",
                "snippet": x.get("description") or x.get("snippet") or "",
                "company": x.get("companyName") or "",
                "location": x.get("location") or "",
                "posted":  x.get("detectedExtensions", {}).get("postedAt") or "",
                "source":  "serper_jobs",
            }
            for x in jobs
            if x.get("title")
        ]
        conf = 0.75 if len(out) >= 5 else (0.45 if out else 0.15)
        return ToolResult(ok=True, confidence=conf, data=out)
    except Exception as e:
        return ToolResult(ok=False, confidence=0.0, error=str(e))


async def firecrawl_scrape(settings: ToolSettings, url: str) -> ToolResult:
    """
    Description: Firecrawl scrape tool (fallback A/B).
    Layer: L4
    """
    if not settings.FIRECRAWL_API_KEY:
        return ToolResult(ok=False, confidence=0.0, error="FIRECRAWL_API_KEY missing")
    try:
        # Minimal Firecrawl HTTP call (works even without SDK)
        headers = {"Authorization": f"Bearer {settings.FIRECRAWL_API_KEY}", "Content-Type": "application/json"}
        payload = {"url": url, "formats": ["markdown", "text"]}
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post("https://api.firecrawl.dev/v1/scrape", headers=headers, json=payload)
        if r.status_code >= 400:
            return ToolResult(ok=False, confidence=0.0, error=f"Firecrawl {r.status_code}: {r.text[:200]}")
        j = r.json()
        text = (j.get("data", {}) or {}).get("text") or (j.get("data", {}) or {}).get("markdown") or ""
        conf = 0.85 if len(text) > 800 else 0.45
        return ToolResult(ok=True, confidence=conf, data={"text": text, "raw": j})
    except Exception as e:
        return ToolResult(ok=False, confidence=0.0, error=str(e))


async def requests_scrape(url: str) -> ToolResult:
    """
    Description: Plain HTTP scrape fallback.
    Layer: L4
    """
    try:
        async with httpx.AsyncClient(timeout=18.0, follow_redirects=True) as client:
            r = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code >= 400:
            return ToolResult(ok=False, confidence=0.0, error=f"HTTP {r.status_code}")
        html = r.text
        txt = re.sub(r"<(script|style)[^>]*>.*?</\\1>", " ", html, flags=re.S | re.I)
        txt = re.sub(r"<[^>]+>", " ", txt)
        txt = re.sub(r"\\s+", " ", txt).strip()
        conf = 0.6 if len(txt) > 1200 else 0.35
        return ToolResult(ok=True, confidence=conf, data={"text": txt[:20000]})
    except Exception as e:
        return ToolResult(ok=False, confidence=0.0, error=str(e))


async def ollama_generate(settings: ToolSettings, prompt: str) -> ToolResult:
    """
    Description: Local Ollama LLM call for summarization/reasoning/drafting.
    Layer: L2-L6
    """
    if not settings.OLLAMA_BASE_URL:
        return ToolResult(ok=False, confidence=0.0, error="OLLAMA_BASE_URL missing")
    try:
        url = settings.OLLAMA_BASE_URL.rstrip("/") + "/api/generate"
        payload = {"model": settings.OLLAMA_MODEL, "prompt": prompt, "stream": False}
        async with httpx.AsyncClient(timeout=45.0) as client:
            r = await client.post(url, json=payload)
        if r.status_code >= 400:
            return ToolResult(ok=False, confidence=0.0, error=f"Ollama {r.status_code}")
        text = (r.json().get("response") or "").strip()
        conf = 0.7 if len(text) > 40 else 0.35
        return ToolResult(ok=True, confidence=conf, data={"text": text, "model": settings.OLLAMA_MODEL})
    except Exception as e:
        return ToolResult(ok=False, confidence=0.0, error=str(e))


async def gemini_generate(settings: ToolSettings, prompt: str) -> ToolResult:
    """Description: Google Gemini REST call.
    Layer: L2-L6
    """
    if not settings.GEMINI_API_KEY:
        return ToolResult(ok=False, confidence=0.0, error="GEMINI_API_KEY missing")
    try:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{settings.GEMINI_MODEL}:generateContent"
            f"?key={settings.GEMINI_API_KEY}"
        )
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 900},
        }
        async with httpx.AsyncClient(timeout=45.0) as client:
            r = await client.post(url, json=payload)
        if r.status_code >= 400:
            return ToolResult(ok=False, confidence=0.0, error=f"Gemini {r.status_code}: {r.text[:180]}")
        j = r.json()
        text = (
            (((j.get("candidates") or [{}])[0].get("content") or {}).get("parts") or [{}])[0].get("text")
            or ""
        ).strip()
        conf = 0.85 if len(text) > 80 else 0.45
        return ToolResult(ok=True, confidence=conf, data={"text": text, "model": settings.GEMINI_MODEL})
    except Exception as e:
        return ToolResult(ok=False, confidence=0.0, error=str(e))


async def tavily_search(settings: ToolSettings, query: str, max_results: int = 5) -> ToolResult:
    """Description: Tavily search (Corrective RAG).
    Layer: L4-L6
    """
    if not settings.TAVILY_API_KEY:
        return ToolResult(ok=False, confidence=0.0, error="TAVILY_API_KEY missing")
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.post(
                "https://api.tavily.com/search",
                json={"api_key": settings.TAVILY_API_KEY, "query": query, "max_results": int(max_results), "include_answer": False},
            )
        if r.status_code >= 400:
            return ToolResult(ok=False, confidence=0.0, error=f"Tavily {r.status_code}: {r.text[:180]}")
        j = r.json()
        conf = 0.75 if (j.get("results") or []) else 0.35
        return ToolResult(ok=True, confidence=conf, data=j)
    except Exception as e:
        return ToolResult(ok=False, confidence=0.0, error=str(e))