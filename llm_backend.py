"""Text-only LLM backend for the moment picker: Claude, or any OpenAI-compatible server.

Ollama, LM Studio, vLLM, llama.cpp's server, LocalAI and OpenRouter all speak
``POST {base}/chat/completions``. When ``LLM_BASE_URL`` is set (or
``LLM_PROVIDER=openai``), the transcript scoring and detail passes in
``main.get_viral_clips`` go here instead of Gemini, so a self-hosted install
can run the whole pipeline without a Google key.

``LLM_PROVIDER=anthropic`` (or ``claude``) sends the same passes to Claude
through the official ``anthropic`` SDK instead, once ``ANTHROPIC_API_KEY`` is
set. Claude is not reached through its OpenAI-compatible endpoint: that layer
returns 400 for the ``temperature`` and ``json_object``/non-strict
``json_schema`` requests the generic path sends to current Claude models, and
Anthropic does not recommend it for production. The SDK's ``messages.parse``
takes the same pydantic schema and returns a validated object. Knobs:
``LLM_MODEL`` (default ``claude-opus-5-5``), ``LLM_EFFORT`` (default
``medium``; ``none`` omits it for models without effort), ``LLM_MAX_TOKENS``
(default 16000) and ``LLM_FALLBACKS`` (default ``default``: server-side
refusal fallback on the models that have one; ``none`` disables it).

``LLM_QUOTA_FALLBACK=anthropic`` keeps Gemini as the provider and makes
Claude its overflow: only a call Gemini refuses for QUOTA (HTTP 429
RESOURCE_EXHAUSTED) goes to Claude. ``main._run_gemini_stage`` waits out a
per-minute limit on Gemini first, for as long as Google says; a spent daily
quota, or a model with no quota on the key (limit 0), sends that call and the
rest of the job's calls to Claude at once. Outages, policy blocks and bad
responses keep their Gemini handling, so they never spend Claude money. The
Claude knobs above (``LLM_MODEL``, ``LLM_EFFORT``, ...) apply to the overflow.

What stays on Gemini, because it needs a model that can look at frames or a
video file: the layout picker (``layout_picker.py``), the on-screen content
detector (``screencast_layout.py``) and the silent-video path
(``main.get_visual_clips``). Without a Gemini key those degrade the way they
already did: the first two return "none", the third fails the job with a clear
message. Nothing in this module is imported by them.

Structured output: the prompts already spell out the exact JSON shape, so a
plain ``json_object`` mode is enough for most models. The request first asks
for ``json_schema`` (Ollama, vLLM, llama.cpp and LM Studio enforce it); a
server that rejects that field gets the same request again with
``json_object``, then with no ``response_format`` at all. Whatever comes back
is validated with the same pydantic model Gemini's ``response_schema`` uses,
so ``main.py`` sees one shape regardless of provider.
"""
from __future__ import annotations

import json
import os
import re
from typing import Optional, Tuple, Type

import httpx
from pydantic import BaseModel

DEFAULT_MODEL = "llama3.1:8b"
DEFAULT_TIMEOUT = 600.0  # local models on CPU are slow; a scoring batch can take minutes

ANTHROPIC_PROVIDERS = ("anthropic", "claude")
DEFAULT_CLAUDE_MODEL = "claude-opus-5-5"
DEFAULT_CLAUDE_EFFORT = "medium"
# Non-streaming default: keeps a request under the SDK's HTTP timeout. Thinking
# counts toward it; the JSON itself is a few thousand tokens at most.
DEFAULT_CLAUDE_MAX_TOKENS = 16000
CLAUDE_API_URL = "https://api.anthropic.com"
# Server-side refusal fallback: a request a safety classifier declines is re-run
# on Anthropic's recommended fallback model inside the same call.
FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Models with a server-side fallback. Claude Haiku 5.5 has none, and its docs
# say to leave the parameter off; older models are not listed.
_FALLBACK_MODEL_PREFIXES = ("claude-opus-5", "claude-sonnet-5", "claude-fable-5", "claude-mythos-5")
# USD per million tokens (input, output), Claude API list prices. Output
# includes thinking. A model missing here is still used, its cost reported 0.
CLAUDE_PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),  # prompts up to 100K tokens; ours are far smaller
}


def provider() -> str:
    """``"anthropic"`` for Claude, ``"openai"`` when a compatible endpoint is
    configured, else ``"gemini"``."""
    explicit = (os.environ.get("LLM_PROVIDER") or "").strip().lower()
    if explicit in ANTHROPIC_PROVIDERS:
        return "anthropic"
    if explicit in ("openai", "ollama", "local", "openai-compatible"):
        return "openai"
    if explicit == "gemini":
        return "gemini"
    return "openai" if base_url() else "gemini"


def base_url() -> str:
    return (os.environ.get("LLM_BASE_URL") or "").strip().rstrip("/")


def _anthropic_key() -> str:
    return (os.environ.get("ANTHROPIC_API_KEY") or "").strip()


def model_name() -> str:
    if provider() == "anthropic":
        return claude_model()
    return (os.environ.get("LLM_MODEL") or "").strip() or DEFAULT_MODEL


def claude_model() -> str:
    """The Claude model, whether Claude is the provider or Gemini's overflow."""
    return (os.environ.get("LLM_MODEL") or "").strip() or DEFAULT_CLAUDE_MODEL


def quota_fallback() -> Optional[str]:
    """``"anthropic"`` when Claude takes the calls Gemini refuses for quota:
    ``LLM_QUOTA_FALLBACK=anthropic`` (or ``claude``), Gemini as the provider,
    and ``ANTHROPIC_API_KEY`` set. Otherwise ``None`` and Gemini's quota
    errors are handled as before."""
    wanted = (os.environ.get("LLM_QUOTA_FALLBACK") or "").strip().lower()
    if wanted in ANTHROPIC_PROVIDERS and provider() == "gemini" and _anthropic_key():
        return "anthropic"
    return None


def active() -> bool:
    """True when the moment picker should call Claude or the OpenAI-compatible
    server instead of Gemini. Claude without a key stays inactive, so the
    pipeline keeps working on Gemini until the key is added."""
    if provider() == "anthropic":
        return bool(_anthropic_key())
    return provider() == "openai" and bool(base_url())


def small_context() -> bool:
    """True for a local OpenAI-compatible server: those usually run a 4-8k
    context, so main.py sends fewer transcript windows per call. Claude and
    Gemini take the full batch."""
    return active() and provider() == "openai"


def label() -> str:
    """Who answers, for the job log."""
    if provider() == "anthropic":
        return f"Claude ({model_name()})"
    return f"local LLM at {base_url()}"


def describe() -> Optional[dict]:
    """What ``/api/config`` tells the dashboard, or ``None`` when inactive."""
    if not active():
        return None
    if provider() == "anthropic":
        return {"provider": "anthropic", "model": model_name(), "baseUrl": CLAUDE_API_URL}
    return {"provider": "openai", "model": model_name(), "baseUrl": base_url()}


def _timeout() -> float:
    try:
        return float(os.environ.get("LLM_TIMEOUT") or DEFAULT_TIMEOUT)
    except ValueError:
        return DEFAULT_TIMEOUT


def _headers() -> dict:
    # Ollama ignores the key but the OpenAI client convention (and vLLM with
    # --api-key) wants the header present; "ollama" is the documented placeholder.
    key = (os.environ.get("LLM_API_KEY") or "ollama").strip()
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _client(**kwargs) -> httpx.Client:
    """Factory so tests can swap in ``httpx.MockTransport``."""
    return httpx.Client(timeout=_timeout(), **kwargs)


def _response_formats(schema: Type[BaseModel]):
    name = getattr(schema, "__name__", "response").lower()
    yield {"type": "json_schema",
           "json_schema": {"name": name, "schema": schema.model_json_schema()}}
    yield {"type": "json_object"}
    yield None


def _is_format_rejection(resp: httpx.Response) -> bool:
    if resp.status_code not in (400, 422):
        return False
    body = resp.text.lower()
    return "response_format" in body or "json_schema" in body or "json_object" in body \
        or "format" in body


def generate_json(prompt: str, schema: Type[BaseModel], model: Optional[str] = None,
                  ) -> Tuple[dict, Optional[dict]]:
    """One chat completion that must come back as JSON matching ``schema``.

    Returns ``(parsed_dict, cost_analysis)`` in the exact shape
    ``main._run_gemini_stage`` returns, so the caller does not branch on the
    provider. Raises on HTTP errors, empty bodies and schema violations; the
    retry policy lives in the caller, same as for Gemini.
    """
    if provider() == "anthropic":
        return _generate_json_claude(prompt, schema, model or model_name())

    import gemini_worker  # local import: keeps this module free of the google SDK

    url = f"{base_url()}/chat/completions"
    model = model or model_name()
    messages = [
        {"role": "system", "content": "You answer with a single JSON object and nothing else."},
        {"role": "user", "content": prompt},
    ]
    last_rejection: Optional[str] = None
    with _client() as client:
        for fmt in _response_formats(schema):
            body = {"model": model, "messages": messages, "temperature": 0.2, "stream": False}
            if fmt is not None:
                body["response_format"] = fmt
            resp = client.post(url, json=body, headers=_headers())
            if fmt is not None and _is_format_rejection(resp):
                # The server does not know this response_format flavour; the
                # next loop iteration asks for a looser one.
                last_rejection = resp.text[:200]
                continue
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"LLM server {resp.status_code} from {url}: {resp.text[:300]}")
            data = resp.json()
            break
        else:
            raise RuntimeError(
                f"LLM server rejected every response_format variant: {last_rejection}")

    choices = data.get("choices") or []
    text = ""
    if choices:
        msg = choices[0].get("message") or {}
        text = msg.get("content") or ""
        if isinstance(text, list):  # some servers return content parts
            text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
    parsed = gemini_worker._parse_json_response_text(text)
    # Validate against the same schema Gemini enforces server-side, so a
    # local model that drops a field fails here with a readable error
    # (retried by the caller) instead of deep inside the clip pipeline.
    validated = schema.model_validate(parsed).model_dump()

    usage = data.get("usage") or {}
    cost = {
        "input_tokens": int(usage.get("prompt_tokens") or 0),
        "output_tokens": int(usage.get("completion_tokens") or 0),
        "thinking_tokens": 0,
        "input_cost": 0.0,
        "output_cost": 0.0,
        "total_cost": 0.0,
        "model": model,
        "price_estimated": False,
        "local": True,
    }
    return validated, cost


# --- Claude (official anthropic SDK) ----------------------------------------

_CLAUDE_SYSTEM = ("You pick and describe moments from video transcripts. "
                  "Answer with the JSON object the schema describes.")


def _claude_effort() -> Optional[str]:
    raw = os.environ.get("LLM_EFFORT")
    if raw is None:
        return DEFAULT_CLAUDE_EFFORT
    raw = raw.strip().lower()
    return None if raw in ("", "none", "off") else raw


def _claude_max_tokens() -> int:
    try:
        return int(os.environ.get("LLM_MAX_TOKENS") or DEFAULT_CLAUDE_MAX_TOKENS)
    except ValueError:
        return DEFAULT_CLAUDE_MAX_TOKENS


def _claude_uses_fallbacks(model: str) -> bool:
    if (os.environ.get("LLM_FALLBACKS") or "default").strip().lower() in ("", "none", "off"):
        return False
    return model.startswith(_FALLBACK_MODEL_PREFIXES)


def _claude_client():
    """Factory so tests can swap in a fake. Reads ANTHROPIC_API_KEY itself."""
    import anthropic  # local import: only the Claude path needs the SDK
    return anthropic.Anthropic(timeout=_timeout(), max_retries=2)


def _generate_json_claude(prompt: str, schema: Type[BaseModel], model: str,
                          ) -> Tuple[dict, Optional[dict]]:
    """``generate_json`` for Claude: structured output validated by the SDK.

    Errors are re-raised as RuntimeError text that main._run_gemini_stage
    already classifies: 429 / 5xx / overloaded / ConnectError are retried
    there (the SDK has retried them twice already), anything else fails the
    stage with the reason. A safety decline raises GeminiBlockedError, the
    same signal a Gemini policy block gives, so main._run_stage_split splits
    the batch instead of failing the job."""
    import anthropic
    import gemini_worker

    max_tokens = _claude_max_tokens()
    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "system": _CLAUDE_SYSTEM,
        "messages": [{"role": "user", "content": prompt}],
        "output_format": schema,
    }
    effort = _claude_effort()
    if effort:
        kwargs["output_config"] = {"effort": effort}
    client = _claude_client()
    try:
        if _claude_uses_fallbacks(model):
            response = client.beta.messages.parse(
                betas=[FALLBACK_BETA], fallbacks="default", **kwargs)
        else:
            response = client.messages.parse(**kwargs)
    except anthropic.AuthenticationError as e:
        raise RuntimeError(
            "Claude rejected ANTHROPIC_API_KEY (401). Check the key in .env; it must be a "
            "Claude Console API key (sk-ant-...), not a claude.ai login.") from e
    except anthropic.PermissionDeniedError as e:
        raise RuntimeError(f"Claude refused access to {model} (403): {e}") from e
    except anthropic.NotFoundError as e:
        raise RuntimeError(f"Claude model {model!r} not found (404); check LLM_MODEL.") from e
    except anthropic.RateLimitError as e:
        raise RuntimeError(f"Claude 429 rate limited: {e}") from e
    except anthropic.APIStatusError as e:
        if e.status_code >= 500:
            raise RuntimeError(f"Claude {e.status_code} server error (overloaded?): {e}") from e
        raise RuntimeError(f"Claude {e.status_code}: {e}") from e
    except anthropic.APIConnectionError as e:
        raise RuntimeError(f"ConnectError talking to Claude ({type(e).__name__}): {e}") from e

    stop_reason = getattr(response, "stop_reason", None)
    if stop_reason == "refusal":
        details = getattr(response, "stop_details", None)
        category = getattr(details, "category", None) if details is not None else None
        raise gemini_worker.GeminiBlockedError(
            f"Claude declined to analyze this content (refusal, category: {category or 'unspecified'}).")
    if stop_reason == "max_tokens":
        raise RuntimeError(
            f"Claude stopped at max_tokens ({max_tokens}) before finishing the JSON; "
            "raise LLM_MAX_TOKENS.")
    parsed = getattr(response, "parsed_output", None)
    if parsed is None:
        raise RuntimeError("Claude returned an empty response body (no structured output).")
    data = parsed.model_dump() if hasattr(parsed, "model_dump") else parsed
    validated = schema.model_validate(data).model_dump()
    return validated, _claude_cost(response, model)


def _claude_cost(response, model: str) -> dict:
    usage = getattr(response, "usage", None)

    def n(field):
        return int(getattr(usage, field, 0) or 0)

    input_tokens = n("input_tokens") + n("cache_creation_input_tokens") + n("cache_read_input_tokens")
    output_tokens = n("output_tokens")
    served_by = getattr(response, "model", None) or model
    iterations = getattr(usage, "iterations", None) or []
    fallback_ran = any(getattr(it, "type", None) == "fallback_message" for it in iterations)
    price = CLAUDE_PRICES.get(served_by) or CLAUDE_PRICES.get(model)
    input_cost = input_tokens * price[0] / 1_000_000 if price else 0.0
    output_cost = output_tokens * price[1] / 1_000_000 if price else 0.0
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,  # includes thinking; the API does not split it out
        "thinking_tokens": 0,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
        "model": served_by,
        "price_estimated": price is None or fallback_ran,
        "local": False,
    }


# --- Claude as the overflow for Gemini's quota (LLM_QUOTA_FALLBACK) ----------

def generate_json_quota_fallback(prompt: str, schema: Type[BaseModel],
                                 ) -> Tuple[dict, Optional[dict]]:
    """``generate_json`` on Claude, for a call Gemini refused for quota."""
    return _generate_json_claude(prompt, schema, claude_model())


# The Gemini API's 429 carries a google.rpc.QuotaFailure naming each quota it
# hit (quotaId "GenerateRequestsPerDayPerProjectPerModel-FreeTier", ...) and its
# limit (quotaValue), plus a RetryInfo delay. The SDK keeps that JSON on the
# error's ``details``; str(error) holds the same text, so an error re-raised as
# a plain message still classifies. Both quote styles are matched.
_QUOTA_ID = re.compile(r"""quotaId['"]?\s*:\s*['"]([^'"]+)['"]""")
_QUOTA_VALUE = re.compile(r"""quotaValue['"]?\s*:\s*['"]?(\d+)""")
_QUOTA_LIMIT = re.compile(r"\blimit:\s*(\d+)")
_RETRY_DELAY = re.compile(r"""retryDelay['"]?\s*:\s*['"](\d+(?:\.\d+)?)s""")
_RETRY_IN = re.compile(r"retry in (\d+(?:\.\d+)?)\s*s", re.IGNORECASE)


def _error_text(error) -> str:
    text = str(error)
    details = getattr(error, "details", None)
    if details is not None and not isinstance(details, str):
        try:
            text += " " + json.dumps(details, default=str)
        except (TypeError, ValueError):
            pass
    return text


def gemini_quota_kind(error) -> Optional[str]:
    """How a failed Gemini call ran out of quota, or ``None`` when it did not.

    ``"none"``: the model has no quota of that kind on this key (limit 0), so
    no wait helps. ``"daily"``: a per-day quota is spent; free-tier quotas
    come back at midnight Pacific time. ``"minute"``: a per-minute request or
    token limit, clear within a minute. ``"unknown"``: a 429 that does not say
    which quota it hit."""
    text = _error_text(error)
    if getattr(error, "code", None) != 429 and "RESOURCE_EXHAUSTED" not in text:
        return None
    limits = _QUOTA_VALUE.findall(text) + _QUOTA_LIMIT.findall(text)
    if "0" in limits:
        return "none"
    quota_ids = _QUOTA_ID.findall(text)
    if any("PerDay" in q for q in quota_ids):
        return "daily"
    if any("PerMinute" in q for q in quota_ids):
        return "minute"
    return "unknown"


def gemini_retry_after(error) -> Optional[float]:
    """Seconds Google asks a caller to wait (RetryInfo, else the message's
    "Please retry in Ns"), or ``None`` when it gives no delay."""
    text = _error_text(error)
    match = _RETRY_DELAY.search(text) or _RETRY_IN.search(text)
    return float(match.group(1)) if match else None
