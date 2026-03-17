from careeragent.utils.url_normalization import normalize_api_base


def test_normalize_typo_scheme_tips() -> None:
    assert normalize_api_base("tips://careeragent-api.onrender.com") == "https://careeragent-api.onrender.com"


def test_normalize_missing_scheme() -> None:
    assert normalize_api_base("careeragent-api.onrender.com/") == "https://careeragent-api.onrender.com"


def test_normalize_bad_value_falls_back_default() -> None:
    assert normalize_api_base("") == "http://127.0.0.1:8000"
    assert normalize_api_base("ftp://example.com") == "http://127.0.0.1:8000"
