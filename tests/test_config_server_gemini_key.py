"""/api/config tells the dashboard when the server's .env holds a Gemini key.

resolve_gemini falls back to GEMINI_API_KEY for any request without an
X-Gemini-Key header, but the dashboard only knew about a key pasted into the
browser, so a self-hoster with the key in .env was still told to set one and
could not start a job. The flag is self-host only; the key itself never leaves
the server.
"""
import asyncio

import httpx
import pytest

app_module = pytest.importorskip("app")


def _config():
    async def _do():
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
            return (await c.get("/api/config")).json()
    return asyncio.run(_do())


def test_a_key_in_the_server_env_is_reported(monkeypatch):
    monkeypatch.setattr(app_module, "BILLING_ENABLED", False)
    monkeypatch.setenv("GEMINI_API_KEY", "AQ.server-side-test-key")
    body = _config()
    assert body["serverGeminiKey"] is True
    assert "AQ.server-side-test-key" not in str(body)  # a flag, never the key


def test_no_key_or_a_blank_one_is_not(monkeypatch):
    monkeypatch.setattr(app_module, "BILLING_ENABLED", False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert _config()["serverGeminiKey"] is False
    monkeypatch.setenv("GEMINI_API_KEY", "   ")
    assert _config()["serverGeminiKey"] is False


def test_hosted_mode_never_reports_it(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(app_module, "BILLING_ENABLED", True)
    # Self-host loads no cloud module; get_config reads one setting from it.
    monkeypatch.setattr(app_module, "cloud", SimpleNamespace(
        settings=SimpleNamespace(google_auth_enabled=False)))
    monkeypatch.setenv("GEMINI_API_KEY", "AQ.server-side-test-key")
    assert _config()["serverGeminiKey"] is False
