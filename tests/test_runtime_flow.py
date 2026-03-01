import asyncio
import sys


sys.path.insert(0, "src")


def test_core_mcp_client_tries_normalized_paths(monkeypatch):
    from careeragent.core.mcp_client import MCPClient
    from careeragent.core.settings import Settings

    called = []

    class _Resp:
        def __init__(self, status_code, payload=None, text=""):
            self.status_code = status_code
            self._payload = payload or {}
            self.text = text

        def json(self):
            return self._payload

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def post(self, url, json):
            called.append(url)
            if url.endswith("/invoke") and not url.endswith("/mcp/invoke"):
                return _Resp(404, text="not found")
            return _Resp(200, payload={"ok": True})

    import httpx

    monkeypatch.setattr(httpx, "Client", _Client)

    c = MCPClient(Settings(MCP_SERVER_URL="https://example.test/mcp"))
    out = c._remote_invoke("jobs.search", {"q": "python"})

    assert out is not None and out.ok is True
    assert called == [
        "https://example.test/invoke",
        "https://example.test/mcp/invoke",
    ]


def test_langgraph_tool_mcp_client_tries_normalized_paths(monkeypatch):
    from careeragent.langgraph.tools import MCPClient

    called = []

    class _Resp:
        def __init__(self, status_code, payload=None, text=""):
            self.status_code = status_code
            self._payload = payload or {}
            self.text = text

        def json(self):
            return self._payload

    class _AsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        async def post(self, url, headers=None, json=None):
            called.append(url)
            if url.endswith("/invoke") and not url.endswith("/mcp/invoke"):
                return _Resp(404, text="not found")
            return _Resp(200, payload={"ok": True})

    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", _AsyncClient)

    async def _run():
        c = MCPClient("https://example.test/mcp", "token")
        return await c.invoke(tool="jobs.search", payload={"q": "python"})

    out = asyncio.run(_run())

    assert out.ok is True
    assert called == [
        "https://example.test/invoke",
        "https://example.test/mcp/invoke",
    ]


def test_runtime_nodes_mcp_invoke_tries_normalized_paths(monkeypatch):
    from careeragent.langgraph.runtime_nodes import RuntimeSettings, mcp_invoke

    called = []

    class _Resp:
        def __init__(self, status_code, payload=None, text=""):
            self.status_code = status_code
            self._payload = payload or {}
            self.text = text

        def json(self):
            return self._payload

    class _AsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        async def post(self, url, headers=None, json=None):
            called.append(url)
            if url.endswith("/invoke") and not url.endswith("/mcp/invoke"):
                return _Resp(404, text="not found")
            return _Resp(200, payload={"ok": True})

    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", _AsyncClient)

    async def _run():
        s = RuntimeSettings(MCP_SERVER_URL="https://example.test/mcp", MCP_API_KEY="token")
        return await mcp_invoke(s, "jobs.search", {"q": "python"})

    ok, conf, data, err = asyncio.run(_run())

    assert ok is True
    assert conf == 0.85
    assert data == {"ok": True}
    assert err is None
    assert called == [
        "https://example.test/invoke",
        "https://example.test/mcp/invoke",
    ]
