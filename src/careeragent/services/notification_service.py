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

    def _send_gmail(self, *, subject: str, body: str, to_addr: str = "") -> Dict[str, Any]:
        to_addr = to_addr or self._settings.GMAIL_TO_EMAIL
        from_addr = self._settings.GMAIL_FROM_EMAIL
        sa_path = self._settings.GMAIL_SERVICE_ACCOUNT_JSON
        if not (to_addr and from_addr and sa_path):
            return {"sent": False, "skipped": True, "reason": "gmail_not_configured", "to": to_addr}
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

    def _send_resend(self, *, subject: str, body: str, to_addr: str = "") -> Dict[str, Any]:
        api_key = str(getattr(self._settings, "RESEND_API_KEY", "") or "").strip()
        from_addr = str((getattr(self._settings, "SENDER_EMAIL", "") or getattr(self._settings, "GMAIL_FROM_EMAIL", "") or "")).strip()
        to_addr = str((to_addr or getattr(self._settings, "GMAIL_TO_EMAIL", "") or getattr(self._settings, "SENDER_EMAIL", "") or "")).strip()
        if not (api_key and from_addr and to_addr):
            return {"sent": False, "skipped": True, "reason": "resend_not_configured", "to": to_addr}
        if self._dry_run:
            return {"sent": False, "dry_run": True, "to": to_addr, "channel": "resend"}
        try:
            r = httpx.post(
                "https://api.resend.com/emails",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"from": from_addr, "to": [to_addr], "subject": subject, "text": body},
                timeout=15.0,
            )
            return {"sent": r.status_code < 300, "status_code": r.status_code, "channel": "resend"}
        except Exception as e:
            return {"sent": False, "channel": "resend", "error": str(e)}

    def _send_sendgrid(self, *, subject: str, body: str, to_addr: str = "") -> Dict[str, Any]:
        api_key = str(getattr(self._settings, "SENDGRID_API_KEY", "") or "").strip()
        from_addr = str((getattr(self._settings, "SENDER_EMAIL", "") or getattr(self._settings, "GMAIL_FROM_EMAIL", "") or "")).strip()
        to_addr = str((to_addr or getattr(self._settings, "GMAIL_TO_EMAIL", "") or getattr(self._settings, "SENDER_EMAIL", "") or "")).strip()
        if not (api_key and from_addr and to_addr):
            return {"sent": False, "skipped": True, "reason": "sendgrid_not_configured", "to": to_addr}
        if self._dry_run:
            return {"sent": False, "dry_run": True, "to": to_addr, "channel": "sendgrid"}
        try:
            r = httpx.post(
                "https://api.sendgrid.com/v3/mail/send",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "personalizations": [{"to": [{"email": to_addr}]}],
                    "from": {"email": from_addr},
                    "subject": subject,
                    "content": [{"type": "text/plain", "value": body}],
                },
                timeout=15.0,
            )
            return {"sent": r.status_code < 300, "status_code": r.status_code, "channel": "sendgrid"}
        except Exception as e:
            return {"sent": False, "channel": "sendgrid", "error": str(e)}

    def _send_twilio_sms(self, *, body: str, to_number: str = "") -> Dict[str, Any]:
        sid = self._settings.TWILIO_ACCOUNT_SID
        token = self._settings.TWILIO_AUTH_TOKEN or getattr(self._settings, "TWILIO_CLIENT_SECRET", None)
        from_number = self._settings.TWILIO_FROM_NUMBER
        to_number = to_number or self._settings.TWILIO_TO_NUMBER
        if not (sid and token and from_number and to_number):
            return {"sent": False, "skipped": True, "reason": "twilio_not_configured", "to": to_number}
        if self._dry_run:
            return {"sent": False, "dry_run": True, "to": to_number, "channel": "sms"}
        try:
            url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
            r = httpx.post(
                url,
                data={"From": from_number, "To": to_number, "Body": body[:1200]},
                auth=(sid, token),
                timeout=15.0,
            )
            return {"sent": r.status_code < 300, "status_code": r.status_code, "channel": "sms"}
        except Exception as e:
            return {"sent": False, "channel": "sms", "error": str(e)}

    def send_alert(
        self,
        *,
        message: str,
        title: str = "CareerAgent HITL Required",
        priority: str = "high",
        to_email: str = "",
        to_phone: str = "",
        enable_email: bool = True,
        enable_sms: bool = True,
    ) -> Dict[str, Any]:
        email_results: list[Dict[str, Any]] = []
        sms_res: Dict[str, Any] = {"sent": False, "skipped": True, "reason": "sms_disabled"}

        if enable_email:
            gmail_res = self._send_gmail(subject=title, body=message, to_addr=to_email)
            email_results.append(gmail_res)
            if not gmail_res.get("sent"):
                resend_res = self._send_resend(subject=title, body=message, to_addr=to_email)
                email_results.append(resend_res)
                if not resend_res.get("sent"):
                    email_results.append(self._send_sendgrid(subject=title, body=message, to_addr=to_email))
        else:
            email_results.append({"sent": False, "skipped": True, "reason": "email_disabled"})

        if enable_sms:
            sms_res = self._send_twilio_sms(body=f"{title}: {message}", to_number=to_phone)

        topic = self._settings.NTFY_TOPIC or "careeragent_alerts_ganesh"
        url = f"{self._settings.NTFY_BASE_URL.rstrip('/')}/{topic}"
        headers = {"Title": title, "Priority": priority}
        payload = message.encode("utf-8")

        if self._dry_run:
            return {"sent": False, "dry_run": True, "url": url, "message": message, "email": email_results, "gmail": (email_results[0] if email_results else {}), "sms": sms_res}

        try:
            r = httpx.post(url, data=payload, headers=headers, timeout=15.0)
            return {
                "sent": r.status_code < 300,
                "status_code": r.status_code,
                "url": url,
                "email": email_results,
                "gmail": (email_results[0] if email_results else {}),
                "sms": sms_res,
            }
        except Exception as e:
            return {"sent": False, "error": str(e), "url": url, "email": email_results, "gmail": (email_results[0] if email_results else {}), "sms": sms_res}
