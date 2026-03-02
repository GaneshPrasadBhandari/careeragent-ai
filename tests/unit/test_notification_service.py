from careeragent.services.notification_service import NotificationService


class _FakeSettings:
    GMAIL_TO_EMAIL = ""
    GMAIL_FROM_EMAIL = ""
    GMAIL_SERVICE_ACCOUNT_JSON = ""
    TWILIO_ACCOUNT_SID = "sid"
    TWILIO_AUTH_TOKEN = "token"
    TWILIO_FROM_NUMBER = "+14155550123"
    TWILIO_TO_NUMBER = ""
    NTFY_TOPIC = "careeragent_test"
    NTFY_BASE_URL = "https://ntfy.sh"


def test_send_alert_accepts_runtime_recipients_in_dry_run() -> None:
    svc = NotificationService(settings=_FakeSettings(), dry_run=True)
    out = svc.send_alert(
        message="hello",
        title="status",
        to_email="user@example.com",
        to_phone="+14155550100",
    )
    assert out["dry_run"] is True
    assert out["gmail"].get("skipped") is True
    assert out["gmail"].get("to") == "user@example.com"
    assert out["sms"].get("to") == "+14155550100"
    assert out["sms"].get("dry_run") is True
