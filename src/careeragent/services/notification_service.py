from __future__ import annotations

import base64
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

from careeragent.core.settings import Settings


class NotificationService:
    """L7/L5 notifications via ntfy.sh and optional Gmail API."""

    def __init__(self, settings: Optional[Settings] = None, *, dry_run: bool = False) -> None:
        self._settings = settings or Settings()
        self._dry_run = dry_run

    def _send_gmail(self, *, subject: str, body: str) -> Dict[str, Any]:
        to_addr = self._settings.GMAIL_TO_EMAIL
        from_addr = self._settings.GMAIL_FROM_EMAIL
        sa_path = self._settings.GMAIL_SERVICE_ACCOUNT_JSON
        if not (to_addr and from_addr and sa_path):
            return {"sent": False, "skipped": True, "reason": "gmail_not_configured"}
        try:
            from google.oauth2 import service_account
            from googleapiclient.discovery import build

            creds = service_account.Credentials.from_service_account_file(
                str(Path(sa_path)),
                scopes=["https://www.googleapis.com/auth/gmail.send"],
                subject=from_addr,
            )
            service = build("gmail", "v1", credentials=creds, cache_discovery=False)

            message = MIMEText(body)
            message["to"] = to_addr
            message["from"] = from_addr
            message["subject"] = subject
            raw = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")

            if self._dry_run:
                return {"sent": False, "dry_run": True, "to": to_addr, "subject": subject}

            sent = service.users().messages().send(userId="me", body={"raw": raw}).execute()
            return {"sent": True, "id": sent.get("id"), "channel": "gmail"}
        except Exception as e:
            return {"sent": False, "channel": "gmail", "error": str(e)}

    def send_alert(self, *, message: str, title: str = "CareerAgent HITL Required", priority: str = "high") -> Dict[str, Any]:
        gmail_res = self._send_gmail(subject=title, body=message)

        topic = self._settings.NTFY_TOPIC or "careeragent_alerts_ganesh"
        url = f"{self._settings.NTFY_BASE_URL.rstrip('/')}/{topic}"
        headers = {"Title": title, "Priority": priority}
        payload = message.encode("utf-8")

        if self._dry_run:
            return {"sent": False, "dry_run": True, "url": url, "message": message, "gmail": gmail_res}

        try:
            r = httpx.post(url, data=payload, headers=headers, timeout=15.0)
            return {
                "sent": r.status_code < 300,
                "status_code": r.status_code,
                "url": url,
                "gmail": gmail_res,
            }
        except Exception as e:
            return {"sent": False, "error": str(e), "url": url, "gmail": gmail_res}
