from __future__ import annotations

import hashlib
import re
import time
import urllib.parse
import urllib.robotparser
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx

from careeragent.core.settings import Settings


def canonical_url(url: str) -> str:
    """Description: Canonicalize URLs for dedup.
    Layer: L3
    Input: raw url
    Output: canonical url
    """
    try:
        u = urllib.parse.urlsplit(url)
        # strip tracking
        q = urllib.parse.parse_qsl(u.query, keep_blank_values=True)
        q = [(k, v) for (k, v) in q if not k.lower().startswith("utm_")]
        query = urllib.parse.urlencode(q)
        clean = urllib.parse.urlunsplit((u.scheme, u.netloc, u.path.rstrip("/"), query, ""))
        return clean
    except Exception:
        return url


def stable_key(url: str) -> str:
    """Description: Stable short key for filenames."""
    return hashlib.sha256(canonical_url(url).encode("utf-8")).hexdigest()[:12]


@dataclass
class RobotsDecision:
    allowed: bool
    reason: str


class RobotsGuard:
    """Description: robots.txt compliance.
    Layer: L9
    Input: url
    Output: allow/deny
    """

    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self._cache: Dict[str, urllib.robotparser.RobotFileParser] = {}

    def allowed(self, url: str, user_agent: str = "CareerAgentAI") -> RobotsDecision:
        try:
            parts = urllib.parse.urlsplit(url)
            base = f"{parts.scheme}://{parts.netloc}"
            if base not in self._cache:
                rp = urllib.robotparser.RobotFileParser()
                rp.set_url(base + "/robots.txt")
                rp.read()
                self._cache[base] = rp
            ok = self._cache[base].can_fetch(user_agent, url)
            return RobotsDecision(allowed=bool(ok), reason="ok" if ok else "blocked by robots.txt")
        except Exception:
            # If robots fails to load, be conservative: allow read-only fetch, block apply.
            return RobotsDecision(allowed=True, reason="robots check failed; allowed")


class SerperClient:
    """Description: Google Serper search client.
    Layer: L3
    """

    def __init__(self, settings: Settings) -> None:
        self.s = settings

    def search(self, query: str, num: int = 10) -> List[Dict[str, Any]]:
        if not self.s.SERPER_API_KEY:
            return []
        url = "https://google.serper.dev/search"
        headers = {"X-API-KEY": self.s.SERPER_API_KEY, "Content-Type": "application/json"}
        payload = {"q": query, "num": int(num)}
        with httpx.Client(timeout=self.s.MAX_HTTP_SECONDS) as client:
            r = client.post(url, headers=headers, json=payload)
        if r.status_code >= 400:
            return []
        data = r.json()
        out = []
        for item in (data.get("organic") or []):
            out.append(
                {
                    "title": item.get("title"),
                    "url": canonical_url(item.get("link") or ""),
                    "snippet": item.get("snippet"),
                }
            )
        return out


class TavilyClient:
    """Description: Tavily search client.
    Layer: L3
    """

    def __init__(self, settings: Settings) -> None:
        self.s = settings

    def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        if not self.s.TAVILY_API_KEY:
            return []
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": self.s.TAVILY_API_KEY,
            "query": query,
            "max_results": int(max_results),
            "search_depth": "advanced",
            "include_answer": False,
            "include_raw_content": False,
        }
        with httpx.Client(timeout=self.s.MAX_HTTP_SECONDS) as client:
            r = client.post(url, json=payload)
        if r.status_code >= 400:
            return []
        data = r.json()
        out = []
        for item in (data.get("results") or []):
            out.append(
                {
                    "title": item.get("title"),
                    "url": canonical_url(item.get("url") or ""),
                    "snippet": item.get("content") or item.get("snippet"),
                }
            )
        return out


class JinaReader:
    """Description: Jina Reader extraction.
    Layer: L3
    """

    def __init__(self, settings: Settings, robots: RobotsGuard) -> None:
        self.s = settings
        self.robots = robots

    def fetch_markdown(self, url: str) -> Tuple[Optional[str], Optional[str]]:
        dec = self.robots.allowed(url)
        if not dec.allowed:
            return None, dec.reason
        rurl = self.s.JINA_READER_PREFIX.rstrip("/") + "/" + url
        with httpx.Client(timeout=self.s.MAX_HTTP_SECONDS) as client:
            r = client.get(rurl, headers={"User-Agent": "CareerAgentAI"})
        if r.status_code >= 400:
            return None, f"jina status {r.status_code}"
        text = r.text
        # basic sanity
        if len(text.strip()) < 200:
            return text, "short"
        return text, None


def extract_explicit_location(snippet: str, title: str = "", body_head: str = "") -> Optional[str]:
    """Description: Extract explicit location from snippet/header.
    Layer: L3
    Input: snippet/title/head
    Output: location string or None
    """
    blob = "\n".join([snippet or "", title or "", body_head or ""]).strip()
    # Location: City, ST
    m = re.search(r"\bLocation\s*[:\-]\s*([^\n|]{2,80})", blob, flags=re.I)
    if m:
        return m.group(1).strip()
    # City, ST
    m2 = re.search(r"\b([A-Za-z][A-Za-z .'-]{2,}),\s*([A-Z]{2})\b", blob)
    if m2:
        return f"{m2.group(1).strip()}, {m2.group(2).strip()}"
    # United States
    if re.search(r"\b(United States|USA|U\.S\.)\b", blob, flags=re.I):
        return "United States"
    return None


def is_non_us_location(loc: str) -> bool:
    if not loc:
        return False
    l = loc.lower()
    # Hard non-US mentions
    for bad in ["nashik", "pune", "hyderabad", "chennai", "mumbai", "karachi", "moscow"]:
        if bad in l:
            return True
    return False


def is_outside_target_geo(url: str, allowed_geos: list) -> bool:
    """
    Checks if a URL belongs to a blocked top-level domain based on
    the user's allowed locations.
    """
    blocked_tlds = [".in", ".pk", ".ru"]
    # If allowed_geos includes 'US' or 'Remote', we block specific non-target TLDs
    if any(url.lower().endswith(tld) for tld in blocked_tlds):
        # Check if the blocked TLD is actually in our allowed list (unlikely for India/US mix)
        if not any(g.lower() in ["india", "in"] for g in allowed_geos):
            return True
    return False
