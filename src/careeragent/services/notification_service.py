from __future__ import annotations

from typing import Any, Dict, Optional

import httpx

from careeragent.core.settings import Settings


class NotificationService:
    """L7/L5 notifications via ntfy.sh."""

    def __init__(self, settings: Optional[Settings] = None, *, dry_run: bool = False) -> None:
        self._settings = settings or Settings()
        self._dry_run = dry_run

    def send_alert(self, *, message: str, title: str = "CareerAgent HITL Required", priority: str = "high") -> Dict[str, Any]:
        topic = self._settings.NTFY_TOPIC or "careeragent_alerts_ganesh"
        url = f"{self._settings.NTFY_BASE_URL.rstrip('/')}/{topic}"
        headers = {"Title": title, "Priority": priority}
        payload = message.encode("utf-8")

        if self._dry_run:
            return {"sent": False, "dry_run": True, "url": url, "message": message}

        try:
            r = httpx.post(url, data=payload, headers=headers, timeout=15.0)
            return {"sent": r.status_code < 300, "status_code": r.status_code, "url": url}
        except Exception as e:
            return {"sent": False, "error": str(e), "url": url}
