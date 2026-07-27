"""Moonshot (Kimi) as a judge provider — routed through polygate.

Moonshot is OpenAI wire-compatible, which used to be the one real hazard: an
``OpenAI`` client built without an explicit key reads ``OPENAI_API_KEY``, so a
keyless Moonshot judge could ship an OpenAI credential to ``api.moonshot.ai``.
Routing text judging through polygate removes that failure mode structurally —
polygate's Moonshot adapter reads ``MOONSHOT_API_KEY``, never ``OPENAI_API_KEY``.

These tests pin that the dispatch reaches polygate with the right
provider/model/key, that a keyless Moonshot judge refuses rather than borrowing a
key, that BYOK keys are forwarded, and that usage is counted. polygate itself is
faked so the tests stay hermetic (no network, no polygate install required).

Run directly:

    ../.workers-venv/Scripts/python.exe tests/test_moonshot_judge.py
"""
import os
import sys
import types

os.environ.setdefault("EVAL_JUDGE_THRESHOLD", "0.7")
os.environ.setdefault("EVAL_JUDGE_CACHE", "1")
os.environ.setdefault("EVAL_JUDGE_CACHE_TTL", "300")
os.environ.setdefault("EVAL_JUDGE_CACHE_MAX", "1000")
os.environ.setdefault("EVAL_JUDGE_PROVIDER", "openai")
os.environ.setdefault("EVAL_JUDGE_MODEL", "gpt-4o-mini")
os.environ.setdefault("CLICKHOUSE_PORT", "8123")
# Single attempt so the fake polygate below never needs a real Retry policy.
os.environ["EVAL_JUDGE_RETRY_ATTEMPTS"] = "1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── a fake `polygate` module so these tests are hermetic ──────────────────────

class MissingAPIKeyError(Exception):
    """Mirror of polygate.exceptions.MissingAPIKeyError."""


class _FakeUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens


class _FakeResponse:
    def __init__(self, content: str, usage: _FakeUsage):
        self.content = content
        self.role = "assistant"
        self.usage = usage


# Records every polygate.chat(**kwargs) the judge makes.
CALLS: list[dict] = []


def _default_chat(**kwargs):
    CALLS.append(kwargs)
    return _FakeResponse('{"score": 0.8, "reason": "ok"}', _FakeUsage(900, 40))


def install_fake_polygate(chat_impl=None):
    """(Re)install a stub `polygate` into sys.modules and reset the call log."""
    CALLS.clear()
    mod = types.ModuleType("polygate")
    mod.chat = chat_impl or _default_chat
    mod.MissingAPIKeyError = MissingAPIKeyError

    class Retry:  # only referenced when EVAL_JUDGE_RETRY_ATTEMPTS > 1
        def __init__(self, **kw):
            self.kw = kw

    mod.Retry = Retry

    exc_mod = types.ModuleType("polygate.exceptions")
    exc_mod.MissingAPIKeyError = MissingAPIKeyError

    sys.modules["polygate"] = mod
    sys.modules["polygate.exceptions"] = exc_mod
    return mod


install_fake_polygate()

from jobs.helper.judge import (  # noqa: E402
    DEFAULT_MODELS,
    PROVIDERS,
    JudgeUsage,
    LLMJudge,
)
from jobs.helper.vision import supported_kinds  # noqa: E402
from jobs.run import parse_judge_spec, parse_jury_specs  # noqa: E402


# ── registration ──────────────────────────────────────────────────────────────

def test_moonshot_is_a_judge_provider():
    """Without this, a stored Moonshot key would never be reachable."""
    assert "moonshot" in PROVIDERS
    assert DEFAULT_MODELS["moonshot"] == "kimi-k2-0711-preview"


def test_specs_parse():
    assert parse_judge_spec("moonshot:kimi-k2-0711-preview") == (
        "moonshot", "kimi-k2-0711-preview",
    )
    assert parse_jury_specs(["moonshot:kimi-k2-0711-preview"]) == [
        ("moonshot", "kimi-k2-0711-preview"),
    ]


# ── dispatch through polygate ──────────────────────────────────────────────────

def test_dispatch_reaches_polygate_with_the_moonshot_provider():
    install_fake_polygate()
    LLMJudge(provider="moonshot", model="kimi-k2-0711-preview",
             api_key="sk-moonshot-test")("grade this")
    assert CALLS, "__call__ did not route through polygate"
    assert CALLS[-1]["provider"] == "moonshot"
    assert CALLS[-1]["model"] == "kimi-k2-0711-preview"


def test_byok_key_reaches_the_call():
    install_fake_polygate()
    LLMJudge(provider="moonshot", model="kimi-k2-0711-preview",
             api_key="sk-org-own-key")("grade this")
    assert CALLS, "no call was made"
    assert CALLS[-1]["api_key"] == "sk-org-own-key", "judge did not use the org's key"
    assert CALLS[-1]["provider"] == "moonshot", "call went to the wrong provider"


# ── the key-leak guard ────────────────────────────────────────────────────────

def test_keyless_moonshot_refuses_instead_of_borrowing_the_openai_key():
    """polygate's Moonshot adapter reads MOONSHOT_API_KEY, never OPENAI_API_KEY.

    So with no BYOK key and no MOONSHOT_API_KEY, the call must refuse — and the
    OpenAI key present in the environment must never be forwarded to it.
    """
    real_env = os.environ.get("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = "sk-openai-must-not-leak"
    os.environ.pop("MOONSHOT_API_KEY", None)

    seen: dict = {}

    def _chat(**kwargs):
        seen.update(kwargs)
        # Mimic polygate: Moonshot resolves only MOONSHOT_API_KEY.
        if not kwargs.get("api_key") and not os.getenv("MOONSHOT_API_KEY"):
            raise MissingAPIKeyError("No moonshot API key provided.")
        return _FakeResponse("{}", _FakeUsage(0, 0))

    install_fake_polygate(_chat)
    try:
        judge = LLMJudge(provider="moonshot", model="kimi-k2-0711-preview")
        try:
            judge("grade this")
        except MissingAPIKeyError:
            pass
        else:
            raise AssertionError("keyless moonshot judge did not refuse")
        assert seen.get("api_key") in (None, ""), (
            "a key was forwarded to a call that should have refused"
        )
        assert seen.get("api_key") != "sk-openai-must-not-leak", (
            "the OpenAI key leaked into the Moonshot call"
        )
    finally:
        if real_env is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = real_env


# ── accounting and capability ─────────────────────────────────────────────────

def test_tokens_are_counted():
    """polygate normalizes usage, so BYOK spend stays visible."""
    install_fake_polygate()
    usage = JudgeUsage()
    LLMJudge(provider="moonshot", model="kimi-k2-0711-preview",
             api_key="sk-moonshot-test", usage=usage)("grade this")
    assert usage.drain() == (900, 40, 1)


def test_kimi_k2_is_text_only_so_vision_metrics_skip_it():
    """Not an oversight: Kimi K2 has no image path, and a not-applicable
    verdict beats a failed judge call."""
    assert supported_kinds("moonshot", ("image", "audio", "video")) == ()
    assert not LLMJudge(provider="moonshot", model="kimi-k2-0711-preview").supports_vision()


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {name}\n      {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR {name}\n      {type(exc).__name__}: {exc}")
    print("\n" + ("all green" if not failures else f"{failures} failing"))
    sys.exit(1 if failures else 0)
