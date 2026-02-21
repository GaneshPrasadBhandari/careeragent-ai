from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from careeragent.config import get_settings
from careeragent.orchestration.state import OrchestrationState


class NotificationService:
    """
    Description: L7 notifications service using Twilio SMS for critical run status changes.
    Layer: L7
    Input: OrchestrationState transitions and quota/security events
    Output: SMS send attempts logged to state.meta['notifications']
    """

    def __init__(self, *, dry_run: bool = False) -> None:
        """
        Description: Initialize notification service.
        Layer: L0
        Input: dry_run flag
        Output: NotificationService
        """
        self._settings = get_settings()
        self._dry_run = bool(dry_run)

    def _twilio_ready(self) -> bool:
        """
        Description: Check Twilio credential presence.
        Layer: L0
        Input: Settings
        Output: bool
        """
        s = self._settings
        return bool(s.twilio_account_sid and s.twilio_auth_token and s.twilio_phone)

    def send_sms(self, *, to_phone: str, body: str) -> Dict[str, Any]:
        """
        Description: Send SMS via Twilio (SDK if installed, otherwise safe error).
        Layer: L7
        Input: to_phone + body
        Output: dict result
        """
        payload = {"to": to_phone, "from": self._settings.twilio_phone, "body": body}

        if self._dry_run or not self._twilio_ready():
            return {"sent": False, "dry_run": True, "reason": "dry_run_or_missing_twilio_config", "payload": payload}

        try:
            from twilio.rest import Client  # type: ignore
        except Exception as e:
            return {"sent": False, "dry_run": True, "reason": f"twilio_sdk_missing:{e}", "payload": payload}

        client = Client(self._settings.twilio_account_sid, self._settings.twilio_auth_token)
        msg = client.messages.create(to=to_phone, from_=self._settings.twilio_phone, body=body)
        return {"sent": True, "sid": getattr(msg, "sid", None), "payload": payload}

    def notify(
        self,
        *,
        state: OrchestrationState,
        event: Literal["needs_human_approval", "completed", "quota_error"],
        extra: Optional[Dict[str, Any]] = None,
        to_phone: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Description: Send critical status notification and log it in OrchestrationState.meta.
        Layer: L7
        Input: state + event + extra + optional to_phone override
        Output: result dict
        """
        extra = extra or {}
        to = (to_phone or self._settings.user_phone or "").strip()
        if not to:
            result = {"sent": False, "dry_run": True, "reason": "missing_user_phone", "event": event}
        else:
            if event == "needs_human_approval":
                body = f"CareerAgent-AI: Run {state.run_id} needs human approval. Open the dashboard to review."
            elif event == "completed":
                body = f"CareerAgent-AI: Run {state.run_id} completed successfully."
            else:
                provider = extra.get("provider", "unknown")
                body = f"CareerAgent-AI: Run {state.run_id} blocked due to API quota error ({provider})."
            result = self.send_sms(to_phone=to, body=body)

        state.meta.setdefault("notifications", [])
        state.meta["notifications"].append({"event": event, "to": to, "result": result, "extra": extra})
        state.touch()
        return result
