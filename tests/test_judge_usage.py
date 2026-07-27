"""Judge-token accounting: the measurement every pricing model depends on.

Eval cost is driven by trace size, jury size, and judge model — none of which
the per-metric row count can see, which is why a 3-model jury over a 40-step
trajectory and a one-shot relevance check used to be indistinguishable after
the fact. These tests pin the accounting rules:

  * a panel's jurors accumulate into the *same* place as the primary
  * a cache hit spends nothing and must not be counted
  * draining puts a message's total on one row, so SUM is exact
  * a provider that omits usage under-reports instead of failing the eval

No network or AWS. Run directly:

    ../.workers-venv/Scripts/python.exe tests/test_judge_usage.py
"""
import os
import sys

os.environ.setdefault("EVAL_JUDGE_THRESHOLD", "0.7")
os.environ.setdefault("EVAL_JUDGE_CACHE", "1")
os.environ.setdefault("EVAL_JUDGE_CACHE_TTL", "300")
os.environ.setdefault("EVAL_JUDGE_CACHE_MAX", "1000")
os.environ.setdefault("EVAL_JUDGE_PROVIDER", "openai")
os.environ.setdefault("EVAL_JUDGE_MODEL", "gpt-4o-mini")
os.environ.setdefault("CLICKHOUSE_PORT", "8123")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jobs.helper.judge import (  # noqa: E402
    InMemoryCache,
    JudgeUsage,
    LLMJudge,
    PromptCache,
    _usage_anthropic,
    _usage_gemini,
    _usage_openai,
)
from jobs.run import OrgCredentials, _build_judge  # noqa: E402

ORG = "11111111-1111-1111-1111-111111111111"


class _Obj:
    """Minimal stand-in for a provider SDK response object."""
    def __init__(self, **kw):
        self.__dict__.update(kw)


# ── accumulation ──────────────────────────────────────────────────────────────

def test_usage_accumulates_across_calls():
    u = JudgeUsage()
    u.add(100, 20)
    u.add(50, 10)
    assert (u.input_tokens, u.output_tokens, u.calls) == (150, 30, 2)


def test_drain_returns_totals_and_resets():
    """The second metric row of a message must not repeat the first row's spend."""
    u = JudgeUsage()
    u.add(100, 20)
    assert u.drain() == (100, 20, 1)
    assert u.drain() == (0, 0, 0), "draining twice double-counted the message"


def test_sum_over_rows_equals_true_spend():
    """Simulates one message persisting three metric rows."""
    u = JudgeUsage()
    u.add(1000, 200)   # metric A judge call
    u.add(1500, 300)   # metric B
    u.add(2000, 400)   # metric C
    rows = [u.drain() for _ in range(3)]
    assert sum(r[0] for r in rows) == 4500
    assert sum(r[1] for r in rows) == 900
    assert sum(r[2] for r in rows) == 3


def test_negative_and_missing_counts_are_floored():
    u = JudgeUsage()
    u.add(None, None)
    u.add(-5, -5)
    assert (u.input_tokens, u.output_tokens) == (0, 0)
    assert u.calls == 2, "a call that reported no usage is still a call"


# ── provider extraction ───────────────────────────────────────────────────────

def test_openai_usage_shape():
    resp = _Obj(usage=_Obj(prompt_tokens=1200, completion_tokens=150))
    assert _usage_openai(resp) == (1200, 150)


def test_anthropic_usage_shape_includes_cache_tokens():
    """Cached input is billed too — omitting it would under-report real spend."""
    resp = _Obj(usage=_Obj(
        input_tokens=800, output_tokens=120,
        cache_read_input_tokens=400, cache_creation_input_tokens=100,
    ))
    assert _usage_anthropic(resp) == (1300, 120)


def test_gemini_usage_shape():
    resp = _Obj(usage_metadata=_Obj(prompt_token_count=900, candidates_token_count=80))
    assert _usage_gemini(resp) == (900, 80)


def test_missing_usage_object_does_not_raise():
    for fn in (_usage_openai, _usage_anthropic, _usage_gemini):
        assert fn(_Obj()) == (0, 0)


def test_record_never_raises():
    """Usage is telemetry; a broken accumulator must not fail the evaluation."""
    judge = LLMJudge(provider="openai", model="gpt-4o-mini")

    class _Exploding:
        def add(self, *_a, **_kw):
            raise RuntimeError("boom")

    judge.usage = _Exploding()
    judge._record(10, 5)  # must not propagate


# ── wiring ────────────────────────────────────────────────────────────────────

def test_panel_jurors_share_the_primarys_accumulator():
    """A jury's spend must land in one place, not scattered across judges."""
    shared = JudgeUsage()
    creds = OrgCredentials(ORG, {})
    primary = _build_judge(org_id=ORG, creds=creds, usage=shared)
    juror_a = _build_judge("anthropic", "claude-haiku-4-5", org_id=ORG, creds=creds, usage=shared)
    juror_b = _build_judge("gemini", "gemini-2.5-flash", org_id=ORG, creds=creds, usage=shared)

    assert primary.usage is shared
    assert juror_a.usage is shared and juror_b.usage is shared

    primary._record(1000, 100)
    juror_a._record(1000, 100)
    juror_b._record(1000, 100)
    assert shared.drain() == (3000, 300, 3), "jury spend did not aggregate"


def test_judge_without_a_shared_accumulator_still_has_one():
    judge = LLMJudge(provider="openai", model="gpt-4o-mini")
    judge._record(5, 5)
    assert judge.usage.drain() == (5, 5, 1)


def test_cache_hits_do_not_spend_tokens():
    """A verdict served from cache costs nothing and must not be billed."""
    usage = JudgeUsage()
    backend = InMemoryCache(max_size=10)
    calls = []

    def _underlying(prompt, **_kw):
        calls.append(prompt)
        usage.add(1000, 100)      # stands in for a real provider call
        return '{"score": 0.9}'

    cache = PromptCache(_underlying, model="openai:gpt-4o-mini",
                        backend=backend, ttl=300, namespace=ORG)

    cache("same prompt", temperature=0.0)
    cache("same prompt", temperature=0.0)

    assert len(calls) == 1, "cache did not serve the second call"
    assert usage.drain() == (1000, 100, 1), "a cache hit was counted as spend"


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
