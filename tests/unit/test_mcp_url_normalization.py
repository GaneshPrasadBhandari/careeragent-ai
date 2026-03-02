from careeragent.langgraph.tools import MCPClient


def test_mcp_client_normalizes_invoke_suffixes() -> None:
    c = MCPClient("https://example.com/mcp/invoke", "k")
    assert c.base_url == "https://example.com"


def test_mcp_client_disables_legacy_careeros_backend() -> None:
    c = MCPClient("https://careeros-backend-d9sc.onrender.com/mcp/invoke", "k")
    assert c.base_url == ""
    assert c.available() is False
