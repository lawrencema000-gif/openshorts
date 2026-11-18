"""The OpenAI-compatible moment-picker backend (llm_backend.py).

Self-hosters asked for a pipeline that never calls Google. The transcript
passes go to any /chat/completions server when LLM_BASE_URL is set; these
tests pin the contract main.py relies on: same (parsed, cost) shape as the
Gemini stage, schema validation, and the response_format fallback ladder.
"""
import json
import sys
import types
from types import SimpleNamespace

import httpx
import pytest

import gemini_worker
import llm_backend


@pytest.fixture
def local(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://llm.test/v1")
    monkeypatch.setenv("LLM_MODEL", "qwen2.5:14b")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)


def _serve(handler, monkeypatch):
    """Route llm_backend's httpx client through an in-process handler."""
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(llm_backend, "_client",
                        lambda **kw: httpx.Client(transport=transport, **kw))


def _completion(payload, usage=None):
    return httpx.Response(200, json={
        "choices": [{"message": {"role": "assistant", "content": json.dumps(payload)}}],
        "usage": usage or {"prompt_tokens": 120, "completion_tokens": 40},
    })


# --- activation -----------------------------------------------------------

def test_inactive_without_a_base_url(monkeypatch):
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert llm_backend.active() is False
    assert llm_backend.describe() is None


def test_base_url_alone_activates_and_describes(local):
    assert llm_backend.active() is True
    assert llm_backend.describe() == {
        "provider": "openai", "model": "qwen2.5:14b", "baseUrl": "http://llm.test/v1"}


def test_explicit_gemini_provider_wins_over_base_url(local, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    assert llm_backend.active() is False


# --- generate_json ---------------------------------------------------------

def test_returns_validated_payload_and_zero_cost(local, monkeypatch):
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers.get("authorization")
        return _completion({"windows": [{"id": "w0", "start": 0, "end": 90, "score": 88, "reason": "hook"}]})

    _serve(handler, monkeypatch)
    parsed, cost = llm_backend.generate_json("prompt", gemini_worker.ScoreResponse)

    assert seen["url"] == "http://llm.test/v1/chat/completions"
    assert seen["body"]["model"] == "qwen2.5:14b"
    assert seen["body"]["response_format"]["type"] == "json_schema"
    assert seen["auth"] == "Bearer ollama"  # documented placeholder when no key is set
    assert parsed["windows"][0]["score"] == 88
    assert cost["total_cost"] == 0.0 and cost["local"] is True
    assert cost["input_tokens"] == 120 and cost["output_tokens"] == 40


def test_falls_back_to_json_object_when_schema_mode_is_rejected(local, monkeypatch):
    formats = []

    def handler(request):
        body = json.loads(request.content)
        fmt = (body.get("response_format") or {}).get("type")
        formats.append(fmt)
        if fmt == "json_schema":
            return httpx.Response(400, json={"error": {"message": "response_format json_schema not supported"}})
        return _completion({"windows": []})

    _serve(handler, monkeypatch)
    parsed, _ = llm_backend.generate_json("prompt", gemini_worker.ScoreResponse)
    assert formats == ["json_schema", "json_object"]
    assert parsed == {"windows": []}


def test_code_fenced_json_is_accepted(local, monkeypatch):
    """Small models wrap the object in ```json fences even when told not to."""
    def handler(request):
        text = "```json\n" + json.dumps({"windows": []}) + "\n```"
        return httpx.Response(200, json={"choices": [{"message": {"content": text}}], "usage": {}})

    _serve(handler, monkeypatch)
    parsed, _ = llm_backend.generate_json("prompt", gemini_worker.ScoreResponse)
    assert parsed == {"windows": []}


def test_schema_violation_raises_instead_of_leaking_into_the_pipeline(local, monkeypatch):
    _serve(lambda r: _completion({"windows": [{"id": "w0"}]}), monkeypatch)
    with pytest.raises(Exception) as exc:
        llm_backend.generate_json("prompt", gemini_worker.ScoreResponse)
    assert "validation error" in str(exc.value)


def test_server_errors_surface_with_status_and_url(local, monkeypatch):
    _serve(lambda r: httpx.Response(503, text="loading model"), monkeypatch)
    with pytest.raises(RuntimeError) as exc:
        llm_backend.generate_json("prompt", gemini_worker.ScoreResponse)
    assert "503" in str(exc.value) and "loading model" in str(exc.value)


# --- main.py routing (needs the heavy deps; skipped on minimal CI) ------------

def test_stage_routes_to_local_backend_and_retries_transient(local, monkeypatch):
    main = pytest.importorskip("main")
    monkeypatch.setattr(main.time, "sleep", lambda *_: None)
    calls = {"n": 0}

    def fake_generate(prompt, schema, model=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("ConnectError: connection refused")
        return {"windows": [{"id": "w0", "start": 0, "end": 1, "score": 50, "reason": ""}]}, {"total_cost": 0.0}

    monkeypatch.setattr(llm_backend, "generate_json", fake_generate)
    parsed, cost = main._run_gemini_stage(None, "qwen2.5:14b", "prompt", gemini_worker.ScoreResponse)
    assert calls["n"] == 2
    assert parsed["windows"][0]["score"] == 50
    assert cost["total_cost"] == 0.0


def test_score_batch_shrinks_for_local_models(local, monkeypatch):
    main = pytest.importorskip("main")
    monkeypatch.delenv("LLM_SCORE_BATCH", raising=False)
    assert main.score_batch_size() == 3
    monkeypatch.setenv("LLM_SCORE_BATCH", "5")
    assert main.score_batch_size() == 5
    monkeypatch.delenv("LLM_BASE_URL")
    monkeypatch.delenv("LLM_SCORE_BATCH")
    assert main.score_batch_size() == 8


# --- Claude (LLM_PROVIDER=anthropic) ------------------------------------------
# CI installs no anthropic SDK, so these run against a stand-in module that
# records what llm_backend asks for. The real SDK is checked in the image.

@pytest.fixture
def claude(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    for var in ("LLM_BASE_URL", "LLM_MODEL", "LLM_EFFORT", "LLM_MAX_TOKENS", "LLM_FALLBACKS"):
        monkeypatch.delenv(var, raising=False)


class _FakeStatusError(Exception):
    def __init__(self, message="", status_code=0):
        super().__init__(message)
        self.status_code = status_code


def _fake_anthropic(monkeypatch, response=None, raises=None):
    """Install a stand-in ``anthropic`` module; returns the recorded calls.
    ``raises`` gets the module and returns the exception to raise."""
    calls = []
    mod = types.ModuleType("anthropic")

    class APIStatusError(_FakeStatusError):
        pass

    class AuthenticationError(APIStatusError):
        pass

    class PermissionDeniedError(APIStatusError):
        pass

    class NotFoundError(APIStatusError):
        pass

    class RateLimitError(APIStatusError):
        pass

    class APIConnectionError(Exception):
        pass

    for cls in (APIStatusError, AuthenticationError, PermissionDeniedError,
                NotFoundError, RateLimitError, APIConnectionError):
        setattr(mod, cls.__name__, cls)

    def parse_for(namespace):
        def parse(**kwargs):
            calls.append((namespace, kwargs))
            if raises is not None:
                raise raises(mod)
            return response
        return parse

    class Anthropic:
        def __init__(self, **kwargs):
            self.messages = SimpleNamespace(parse=parse_for("messages"))
            self.beta = SimpleNamespace(messages=SimpleNamespace(parse=parse_for("beta")))

    mod.Anthropic = Anthropic
    monkeypatch.setitem(sys.modules, "anthropic", mod)
    return calls


def _claude_response(parsed, stop_reason="end_turn", model="claude-opus-5-5", stop_details=None):
    usage = SimpleNamespace(input_tokens=1000, output_tokens=200, cache_creation_input_tokens=0,
                            cache_read_input_tokens=0, iterations=None)
    return SimpleNamespace(parsed_output=parsed, stop_reason=stop_reason, stop_details=stop_details,
                           model=model, usage=usage)


def test_claude_needs_a_key_to_activate(claude, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    assert llm_backend.active() is False  # pipeline stays on Gemini until the key is added
    assert llm_backend.describe() is None
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    assert llm_backend.active() is True
    assert llm_backend.describe() == {
        "provider": "anthropic", "model": "claude-opus-5-5", "baseUrl": "https://api.anthropic.com"}
    assert llm_backend.small_context() is False
    assert llm_backend.label() == "Claude (claude-opus-5-5)"


def test_claude_parses_with_the_schema_and_prices_the_call(claude, monkeypatch):
    parsed = gemini_worker.ScoreResponse(
        windows=[{"id": "w0", "start": 0, "end": 90, "score": 88, "reason": "hook"}])
    calls = _fake_anthropic(monkeypatch, response=_claude_response(parsed))

    result, cost = llm_backend.generate_json("prompt", gemini_worker.ScoreResponse)

    namespace, kwargs = calls[0]
    assert namespace == "beta"  # Opus 5.5 has a server-side refusal fallback
    assert kwargs["betas"] == [llm_backend.FALLBACK_BETA] and kwargs["fallbacks"] == "default"
    assert kwargs["model"] == "claude-opus-5-5"
    assert kwargs["output_format"] is gemini_worker.ScoreResponse
    assert kwargs["output_config"] == {"effort": "medium"}
    assert kwargs["max_tokens"] == 16000
    assert kwargs["messages"] == [{"role": "user", "content": "prompt"}]
    assert "temperature" not in kwargs  # current Claude models reject non-default sampling
    assert result["windows"][0]["score"] == 88
    assert cost["input_tokens"] == 1000 and cost["output_tokens"] == 200
    assert cost["total_cost"] == pytest.approx(1000 * 4 / 1e6 + 200 * 20 / 1e6)
    assert cost["price_estimated"] is False and cost["local"] is False


def test_haiku_gets_no_fallback_and_effort_can_be_dropped(claude, monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "claude-haiku-5-5")
    monkeypatch.setenv("LLM_EFFORT", "none")
    calls = _fake_anthropic(monkeypatch, response=_claude_response(
        gemini_worker.ScoreResponse(windows=[]), model="claude-haiku-5-5"))

    _, cost = llm_backend.generate_json("prompt", gemini_worker.ScoreResponse)

    namespace, kwargs = calls[0]
    assert namespace == "messages"
    assert not {"fallbacks", "betas", "output_config"} & set(kwargs)
    assert cost["total_cost"] == pytest.approx(1000 * 0.10 / 1e6 + 200 * 0.50 / 1e6)


def test_claude_refusal_becomes_a_policy_block(claude, monkeypatch):
    """main._run_stage_split splits a batch on GeminiBlockedError; a Claude
    decline has to give the same signal, not a generic failure."""
    _fake_anthropic(monkeypatch, response=_claude_response(
        None, stop_reason="refusal", stop_details=SimpleNamespace(category="cyber")))
    with pytest.raises(gemini_worker.GeminiBlockedError) as exc:
        llm_backend.generate_json("prompt", gemini_worker.ScoreResponse)
    assert "cyber" in str(exc.value)


def test_claude_truncated_output_says_how_to_fix_it(claude, monkeypatch):
    _fake_anthropic(monkeypatch, response=_claude_response(None, stop_reason="max_tokens"))
    with pytest.raises(RuntimeError) as exc:
        llm_backend.generate_json("prompt", gemini_worker.ScoreResponse)
    assert "LLM_MAX_TOKENS" in str(exc.value)


@pytest.mark.parametrize("make_error, expected", [
    (lambda m: m.RateLimitError("slow down", status_code=429), "429"),           # retried by main
    (lambda m: m.APIStatusError("overloaded_error", status_code=529), "overloaded"),  # retried by main
    (lambda m: m.APIConnectionError("reset"), "ConnectError"),                   # retried by main
    (lambda m: m.AuthenticationError("bad key", status_code=401), "ANTHROPIC_API_KEY"),
    (lambda m: m.NotFoundError("no such model", status_code=404), "LLM_MODEL"),
])
def test_claude_errors_carry_the_markers_main_retries_on(claude, monkeypatch, make_error, expected):
    _fake_anthropic(monkeypatch, raises=make_error)
    with pytest.raises(RuntimeError) as exc:
        llm_backend.generate_json("prompt", gemini_worker.ScoreResponse)
    assert expected in str(exc.value)


def test_score_batch_stays_full_for_claude(claude, monkeypatch):
    main = pytest.importorskip("main")
    monkeypatch.delenv("LLM_SCORE_BATCH", raising=False)
    assert main.score_batch_size() == 8
