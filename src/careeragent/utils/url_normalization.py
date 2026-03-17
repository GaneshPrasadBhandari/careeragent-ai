from __future__ import annotations

from urllib.parse import urlparse


_SCHEME_TYPOS = {
    "htps://": "https://",
    "ttps://": "https://",
    "tps://": "https://",
    "tips://": "https://",
    "http//": "http://",
    "https//": "https://",
}


def normalize_api_base(raw: str, default: str = "http://127.0.0.1:8000") -> str:
    """Normalize user-entered API base URLs for dashboard/backend calls."""
    text = (raw or "").strip()
    if not text:
        return default.rstrip("/")

    lowered = text.lower()
    for typo, fixed in _SCHEME_TYPOS.items():
        if lowered.startswith(typo):
            text = fixed + text[len(typo):]
            break

    parsed = urlparse(text)
    if not parsed.scheme:
        text = f"https://{text}"
        parsed = urlparse(text)

    if parsed.scheme not in {"http", "https"}:
        return default.rstrip("/")

    if not parsed.netloc:
        return default.rstrip("/")

    return text.rstrip("/")

