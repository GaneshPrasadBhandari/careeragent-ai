from careeragent.services.notification_service import NotificationService


class _FakeSettings:
    GMAIL_TO_EMAIL = ""
    GMAIL_FROM_EMAIL = ""
    GMAIL_SERVICE_ACCOUNT_JSON = ""
    TWILIO_ACCOUNT_SID = "sid"
    TWILIO_AUTH_TOKEN = ""
    TWILIO_CLIENT_SECRET = "oauth-token"
    TWILIO_FROM_NUMBER = "+14155550123"
    TWILIO_TO_NUMBER = ""
    RESEND_API_KEY = "re_key"
    SENDER_EMAIL = "sender@example.com"
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


def test_send_alert_falls_back_to_sender_email_when_recipient_missing() -> None:
    svc = NotificationService(settings=_FakeSettings(), dry_run=True)
    out = svc.send_alert(message="hello", title="status", to_email="", enable_sms=False, enable_email=True)
    resend_attempt = next((x for x in out["email"] if x.get("channel") == "resend"), {})
    assert resend_attempt.get("to") == "sender@example.com"


def test_twilio_client_secret_alias_is_used_for_sms() -> None:
    svc = NotificationService(settings=_FakeSettings(), dry_run=True)
    out = svc.send_alert(message="hello", title="status", to_phone="+14155550100", enable_email=False, enable_sms=True)
    assert out["sms"].get("dry_run") is True
    assert out["sms"].get("to") == "+14155550100"
