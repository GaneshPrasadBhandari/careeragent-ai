import sys
import types


try:
    from careeragent.langgraph.tools import MCPClient
except Exception:
    # In constrained envs with partial pydantic installs, provide a tiny
    # pydantic_settings shim so langgraph.tools can import for unit testing.
    shim = types.ModuleType("pydantic_settings")

    class SettingsConfigDict(dict):
        pass

    class BaseSettings:
        model_config = SettingsConfigDict()

        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    shim.BaseSettings = BaseSettings
    shim.SettingsConfigDict = SettingsConfigDict
    sys.modules["pydantic_settings"] = shim
    sys.modules.pop("careeragent.langgraph.tools", None)

    from careeragent.langgraph.tools import MCPClient


def test_mcp_client_normalizes_invoke_suffixes() -> None:
    c = MCPClient("https://example.com/mcp/invoke", "k")
    assert c.base_url == "https://example.com"


def test_mcp_client_disables_legacy_careeros_backend() -> None:
    c = MCPClient("https://careeros-backend-d9sc.onrender.com/mcp/invoke", "k")
    assert c.base_url == ""
    assert c.available() is False
