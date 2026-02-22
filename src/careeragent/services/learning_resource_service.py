
from __future__ import annotations

from typing import Any, Dict, List, Optional
import httpx


class LearningResourceService:
    """
    Description: Fetch tutorials / YouTube / docs for missing skills using Serper.
    Layer: L9
    Input: skill list
    Output: list of learning links
    """

    SERPER_URL = "https://google.serper.dev/search"

    def __init__(self, *, serper_api_key: Optional[str]) -> None:
        self._key = (serper_api_key or "").strip()

    def available(self) -> bool:
        return bool(self._key)

    def search_links(self, *, query: str, num: int = 5) -> List[Dict[str, Any]]:
        if not self.available():
            return []
        headers = {"X-API-KEY": self._key, "Content-Type": "application/json"}
        with httpx.Client(timeout=25.0) as client:
            r = client.post(self.SERPER_URL, headers=headers, json={"q": query, "num": num})
        if r.status_code >= 400:
            return []
        organic = (r.json().get("organic") or [])
        out = []
        for it in organic[:num]:
            out.append({"title": it.get("title"), "link": it.get("link"), "snippet": it.get("snippet")})
        return out

    def build_learning_plan(self, *, missing_skills: List[str]) -> Dict[str, Any]:
        plan: Dict[str, Any] = {}
        for s in missing_skills[:12]:
            skill = str(s).strip()
            if not skill:
                continue
            plan[skill] = {
                "youtube": self.search_links(query=f"{skill} tutorial youtube", num=3),
                "docs": self.search_links(query=f"{skill} official documentation", num=3),
                "course": self.search_links(query=f"best course to learn {skill}", num=3),
            }
        return plan
