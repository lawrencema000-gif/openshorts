"""Claude as the overflow for a free Gemini key (LLM_QUOTA_FALLBACK).

A self-hoster runs the moment picker on a free Gemini key and wants Claude
only once that key's quota runs out. These tests pin the rule:

- a spent daily quota, or a model with no free quota on the key, hands the
  call (and the rest of the job's calls) to Claude at once;
- a per-minute limit is waited out on Gemini first, with Google's own delay;
  only a limit that never clears sends that one call to Claude;
- every other Gemini failure keeps its old handling and never spends Claude
  money, and nothing changes when the overflow is not configured.
"""
import types

import pytest

import llm_backend

genai_errors = pytest.importorskip("google.genai.errors")

DAILY = "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
MINUTE = "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"
TOKENS_PER_MINUTE = "GenerateContentInputTokensPerModelPerMinute-FreeTier"


def _quota_error(quota_id, value="1000", delay="12s", model="gemini-3.1-flash-lite"):
    """The 429 the Gemini API returns, built with the SDK's own error class."""
    details = [{
        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
        "violations": [{
            "quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
            "quotaId": quota_id,
            "quotaDimensions": {"location": "global", "model": model},
            "quotaValue": value,
        }],
    }]
    if delay:
        details.append({"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay})
    return genai_errors.ClientError(429, {"error": {
        "code": 429,
        "message": ("You exceeded your current quota, please check your plan and billing details.\n"
                    "* Quota exceeded for metric: generativelanguage.googleapis.com/"
                    f"generate_content_free_tier_requests, limit: {value}, model: {model}\n"
                    f"Please retry in {delay or '1s'}."),
        "status": "RESOURCE_EXHAUSTED",
        "details": details,
    }})


def _real_429(quota_ids, limit, delay):
    """The shape the Gemini API actually sent on 2026-11-18 (gemini-3.1-pro on a
    free key): one violation per quota hit, no quotaValue, the limit only in
    the message, and a RetryInfo delay up to the quota's reset."""
    return genai_errors.ClientError(429, {"error": {
        "code": 429,
        "message": ("You exceeded your current quota, please check your plan and billing details. "
                    "For more information on this error, head to: https://aistudio.google.com/docs/rate-limits.\n"
                    "* Quota exceeded for metric: generativelanguage.googleapis.com/"
                    f"generate_content_free_tier_requests, limit: {limit}, model: gemini-3.1-flash-lite\n"
                    f"Please retry in {delay}."),
        "status": "RESOURCE_EXHAUSTED",
        "details": [
            {"@type": "type.googleapis.com/google.rpc.Help", "links": []},
            {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [
                {"description": "Generate content requests limit exceeded on Free Tier.",
                 "quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
                 "quotaId": quota_id,
                 "quotaDimensions": {"location": "global", "model": "gemini-3.1-flash-lite"}}
                for quota_id in quota_ids]},
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay},
        ],
    }})


def _overloaded():
    return genai_errors.ServerError(503, {"error": {
        "code": 503, "message": "The model is overloaded. Please try again later.",
        "status": "UNAVAILABLE"}})


@pytest.fixture
def overflow(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("LLM_QUOTA_FALLBACK", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)


# --- when the overflow is armed ----------------------------------------------

def test_overflow_keeps_gemini_first_and_needs_a_claude_key(overflow, monkeypatch):
    assert llm_backend.quota_fallback() == "anthropic"
    assert llm_backend.active() is False  # Gemini still answers first
    monkeypatch.setenv("LLM_QUOTA_FALLBACK", "claude")
    assert llm_backend.quota_fallback() == "anthropic"
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    assert llm_backend.quota_fallback() is None  # no key: plain Gemini, as before


def test_overflow_is_off_by_default_and_when_claude_is_primary(overflow, monkeypatch):
    monkeypatch.delenv("LLM_QUOTA_FALLBACK")
    assert llm_backend.quota_fallback() is None
    monkeypatch.setenv("LLM_QUOTA_FALLBACK", "anthropic")
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    assert llm_backend.quota_fallback() is None  # Claude already answers every call


def test_unset_provider_counts_as_gemini(overflow, monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER")
    assert llm_backend.quota_fallback() == "anthropic"


def test_overflow_model_defaults_to_opus_and_follows_llm_model(overflow, monkeypatch):
    assert llm_backend.claude_model() == "claude-opus-5-5"
    monkeypatch.setenv("LLM_MODEL", "claude-sonnet-5-5")
    assert llm_backend.claude_model() == "claude-sonnet-5-5"


# --- reading Gemini's 429 ----------------------------------------------------

@pytest.mark.parametrize("quota_id,value,kind", [
    (DAILY, "1000", "daily"),
    ("GenerateContentInputTokensPerModelPerDay-FreeTier", "1000000", "daily"),
    (MINUTE, "15", "minute"),
    (TOKENS_PER_MINUTE, "250000", "minute"),
    (MINUTE, "0", "none"),  # the model has no free quota on this key at all
    (DAILY, "0", "none"),
])
def test_quota_errors_are_classified(quota_id, value, kind):
    assert llm_backend.gemini_quota_kind(_quota_error(quota_id, value)) == kind


def test_the_real_reply_shape_classifies():
    # No quotaValue anywhere: the limit is only in the message.
    every_quota = [DAILY, "GenerateContentInputTokensPerModelPerDay-FreeTier", MINUTE, TOKENS_PER_MINUTE]
    no_quota = _real_429(every_quota, limit="0", delay="71928s")
    assert llm_backend.gemini_quota_kind(no_quota) == "none"
    assert llm_backend.gemini_retry_after(no_quota) == 71928.0
    assert llm_backend.gemini_quota_kind(_real_429([DAILY], limit="1000", delay="40000s")) == "daily"
    assert llm_backend.gemini_quota_kind(_real_429([MINUTE], limit="15", delay="31s")) == "minute"


def test_a_429_without_quota_details_is_unknown_quota():
    err = genai_errors.ClientError(429, {"error": {
        "code": 429, "message": "Resource exhausted. Please try again later.",
        "status": "RESOURCE_EXHAUSTED"}})
    assert llm_backend.gemini_quota_kind(err) == "unknown"


@pytest.mark.parametrize("err", [
    _overloaded(),
    genai_errors.ClientError(400, {"error": {
        "code": 400, "message": "API key not valid.", "status": "INVALID_ARGUMENT"}}),
    RuntimeError("Failed to parse Gemini JSON response"),
    RuntimeError("Gemini returned an empty response body"),
])
def test_other_failures_are_not_quota(err):
    assert llm_backend.gemini_quota_kind(err) is None


def test_classification_also_reads_an_error_re_raised_as_text():
    assert llm_backend.gemini_quota_kind(RuntimeError(str(_quota_error(DAILY)))) == "daily"
    assert llm_backend.gemini_quota_kind(RuntimeError(str(_quota_error(MINUTE)))) == "minute"


def test_retry_delay_comes_from_google():
    assert llm_backend.gemini_retry_after(_quota_error(MINUTE, delay="45s")) == 45.0
    assert llm_backend.gemini_retry_after(RuntimeError("Please retry in 33.2s.")) == 33.2
    assert llm_backend.gemini_retry_after(RuntimeError("no hint at all")) is None


# --- main._run_gemini_stage --------------------------------------------------

class _Resp:
    def __init__(self, parsed):
        self.parsed = parsed
        self.candidates = []
        self.usage_metadata = None
        self.prompt_feedback = None

    @property
    def text(self):
        return "{}"


class _Models:
    """Plays a script: an exception item is raised, anything else answered.
    Past the end of the script every call succeeds."""

    def __init__(self, *script):
        self.script = list(script)
        self.calls = 0

    def generate_content(self, **kwargs):
        self.calls += 1
        item = self.script.pop(0) if self.script else {"windows": [{"id": "w0", "score": 70}]}
        if isinstance(item, BaseException):
            raise item
        return _Resp(item)


def _client(models):
    return types.SimpleNamespace(models=models)


@pytest.fixture
def stage(overflow, monkeypatch):
    main = pytest.importorskip("main")
    seen = types.SimpleNamespace(sleeps=[], claude=[], main=main)
    monkeypatch.setattr(main.time, "sleep", lambda s: seen.sleeps.append(s))
    monkeypatch.setattr(main, "_GEMINI_QUOTA", {"spent": None})  # a fresh job

    def fake_claude(prompt, schema):
        seen.claude.append(prompt)
        return {"windows": [{"id": "w9", "score": 99}]}, {
            "input_tokens": 10, "output_tokens": 5, "total_cost": 0.01, "model": "claude-opus-5-5"}

    monkeypatch.setattr(llm_backend, "generate_json_quota_fallback", fake_claude)
    return seen


def test_spent_daily_quota_hands_the_rest_of_the_job_to_claude(stage):
    models = _Models(_quota_error(DAILY))
    parsed, cost = stage.main._run_gemini_stage(_client(models), "gemini-3.1-flash-lite", "p1", object)
    assert models.calls == 1  # no retries: a spent day does not come back in seconds
    assert stage.claude == ["p1"] and stage.sleeps == []
    assert parsed["windows"][0]["id"] == "w9"
    assert cost["model"] == "claude-opus-5-5"
    # The job's next call goes straight to Claude; Gemini is not asked again.
    stage.main._run_gemini_stage(_client(models), "gemini-3.1-flash-lite", "p2", object)
    assert models.calls == 1
    assert stage.claude == ["p1", "p2"]


def test_model_without_free_quota_goes_to_claude_at_once(stage):
    models = _Models(_quota_error(MINUTE, value="0"))
    stage.main._run_gemini_stage(_client(models), "gemini-3.1-pro-preview", "p", object)
    assert models.calls == 1 and stage.claude == ["p"] and stage.sleeps == []


def test_per_minute_limit_is_waited_out_on_gemini(stage):
    models = _Models(_quota_error(MINUTE, delay="45s"))
    parsed, _ = stage.main._run_gemini_stage(_client(models), "gemini-3.1-flash-lite", "p", object)
    assert models.calls == 2
    assert stage.claude == []
    assert len(stage.sleeps) == 1 and stage.sleeps[0] >= 45  # Google's delay, not the 5 s default
    assert parsed["windows"][0]["id"] == "w0"


def test_a_wait_of_hours_counts_as_spent_whatever_the_quota_is_called(stage):
    # A 429 that names no quota but tells us to come back tomorrow.
    err = genai_errors.ClientError(429, {"error": {
        "code": 429, "message": "Resource exhausted. Please retry in 40000s.",
        "status": "RESOURCE_EXHAUSTED"}})
    models = _Models(err)
    stage.main._run_gemini_stage(_client(models), "gemini-3.1-flash-lite", "p1", object)
    assert models.calls == 1 and stage.claude == ["p1"] and stage.sleeps == []
    stage.main._run_gemini_stage(_client(models), "gemini-3.1-flash-lite", "p2", object)
    assert models.calls == 1 and stage.claude == ["p1", "p2"]


def test_real_daily_reply_switches_without_waiting(stage):
    models = _Models(_real_429([DAILY], limit="1000", delay="40000s"))
    stage.main._run_gemini_stage(_client(models), "gemini-3.1-flash-lite", "p", object)
    assert models.calls == 1 and stage.claude == ["p"] and stage.sleeps == []


def test_per_minute_limit_that_never_clears_overflows_that_call_only(stage):
    models = _Models(*[_quota_error(MINUTE)] * 3)
    stage.main._run_gemini_stage(_client(models), "gemini-3.1-flash-lite", "p1", object)
    assert models.calls == 3 and stage.claude == ["p1"]
    # Not a spent day: the next call tries Gemini again (and gets through).
    stage.main._run_gemini_stage(_client(models), "gemini-3.1-flash-lite", "p2", object)
    assert models.calls == 4 and stage.claude == ["p1"]


def test_outages_never_spend_claude_money(stage):
    models = _Models(_overloaded(), _overloaded(), _overloaded())
    with pytest.raises(genai_errors.ServerError):
        stage.main._run_gemini_stage(_client(models), "gemini-3.1-flash-lite", "p", object)
    assert models.calls == 3 and stage.claude == []


def test_without_the_overflow_quota_errors_behave_as_before(stage, monkeypatch):
    monkeypatch.delenv("LLM_QUOTA_FALLBACK")
    models = _Models(*[_quota_error(DAILY)] * 3)
    with pytest.raises(genai_errors.ClientError):
        stage.main._run_gemini_stage(_client(models), "gemini-3.1-flash-lite", "p", object)
    assert models.calls == 3 and stage.claude == []
    assert stage.sleeps == [5, 10]  # the old backoff, untouched


# --- what the job reports ----------------------------------------------------

def test_cost_summary_names_every_model_that_answered(stage):
    costs = [
        {"input_tokens": 100, "output_tokens": 10, "total_cost": 0.001, "model": "gemini-3.1-flash-lite"},
        {"input_tokens": 200, "output_tokens": 20, "total_cost": 0.05, "model": "claude-opus-5-5"},
        {"input_tokens": 100, "output_tokens": 10, "total_cost": 0.001, "model": "gemini-3.1-flash-lite"},
    ]
    summary = stage.main._cost_summary(costs, "gemini-3.1-flash-lite")
    assert summary["model"] == "gemini-3.1-flash-lite + claude-opus-5-5"
    assert summary["total_cost"] == pytest.approx(0.052)
    assert summary["by_model"] == {"gemini-3.1-flash-lite": pytest.approx(0.002),
                                   "claude-opus-5-5": pytest.approx(0.05)}
    assert (summary["input_tokens"], summary["output_tokens"]) == (400, 40)


def test_cost_summary_for_one_model_keeps_the_old_shape(stage):
    costs = [{"input_tokens": 1, "output_tokens": 2, "total_cost": 0.5, "model": "gemini-3.1-flash-lite"},
             {"input_tokens": 1, "output_tokens": 2, "total_cost": 0.5}]  # a cost without a model
    summary = stage.main._cost_summary(costs, "gemini-3.1-flash-lite")
    assert summary["model"] == "gemini-3.1-flash-lite"
    assert summary["total_cost"] == pytest.approx(1.0)
