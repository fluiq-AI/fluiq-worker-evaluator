"""Offline tests for judge-prompt org resolution + provenance capture.

No Postgres involved: the snapshots are injected directly, the way refresh()
would populate them, then resolution and capture are exercised through the
public render/captured_call surface.
"""
from __future__ import annotations

import os
import sys

import pytest

# config.py reads a few vars with float()/int() and will raise on import if they
# are unset. Seed the minimum before importing anything that pulls in config.
os.environ.setdefault("EVAL_JUDGE_THRESHOLD", "0.7")
os.environ.setdefault("EVAL_JUDGE_CACHE_TTL", "300")
os.environ.setdefault("EVAL_JUDGE_CACHE_MAX", "1000")
os.environ.setdefault("CLICKHOUSE_PORT", "8123")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jobs.helper import judge_prompts as jp
from jobs.helper.base import BaseEvaluator, EvalResult


ORG = "11111111-1111-1111-1111-111111111111"


@pytest.fixture(autouse=True)
def _clean_registry():
    """Snapshot and restore the module state around every test."""
    with jp._lock:
        eff, meta, org = dict(jp._effective), dict(jp._effective_meta), dict(jp._org_overrides)
    yield
    with jp._lock:
        jp._effective.clear();      jp._effective.update(eff)
        jp._effective_meta.clear(); jp._effective_meta.update(meta)
        jp._org_overrides.clear();  jp._org_overrides.update(org)
    jp.set_org(None)
    jp.end_capture()


def _install_platform_override(name: str, template: str, version: int = 3):
    with jp._lock:
        jp._effective[name] = template
        jp._effective_meta[name] = {"version": version, "overridden": True}


def _install_org_override(org_id: str, name: str, template: str, version: int = 2):
    with jp._lock:
        jp._org_overrides.setdefault(org_id, {})[name] = {
            "template": template, "version": version,
        }


def test_default_resolution_and_meta():
    jp.set_org(None)
    template, meta = jp._resolve("coherence")
    assert "coherent" in template
    assert meta["source"] == "default"


def test_platform_override_wins_over_default():
    _install_platform_override("coherence", "PLATFORM $answer")
    template, meta = jp._resolve("coherence")
    assert template == "PLATFORM $answer"
    assert meta["source"] == "platform" and meta["version"] == 3


def test_org_override_wins_over_platform_for_that_org_only():
    _install_platform_override("coherence", "PLATFORM $answer")
    _install_org_override(ORG, "coherence", "ORG $answer")

    jp.set_org(ORG)
    template, meta = jp._resolve("coherence")
    assert template == "ORG $answer"
    assert meta["source"] == "org" and meta["version"] == 2

    jp.set_org("22222222-2222-2222-2222-222222222222")
    template, meta = jp._resolve("coherence")
    assert template == "PLATFORM $answer"
    assert meta["source"] == "platform"


def test_render_records_capture_with_dedupe():
    _install_org_override(ORG, "coherence", "ORG $answer")
    jp.set_org(ORG)

    jp.begin_capture()
    first = jp.render("coherence", answer="hello")
    jp.render("coherence", answer="again")   # same name → counted, not duplicated
    jp.render("answer_relevancy", question="q", answer="a")
    used = jp.end_capture()

    assert first == "ORG hello"
    by_name = {u["name"]: u for u in used}
    assert set(by_name) == {"coherence", "answer_relevancy"}
    assert by_name["coherence"]["calls"] == 2
    assert by_name["coherence"]["source"] == "org"
    assert by_name["coherence"]["rendered"] == "ORG hello"
    assert by_name["answer_relevancy"]["source"] == "default"


def test_captured_call_attaches_to_result_details():
    class _Fake(BaseEvaluator):
        name = "fake"

        def evaluate(self, **_) -> EvalResult:
            jp.render("coherence", answer="x")
            return self._result(1.0, "ok", {"existing": True})

    result = jp.captured_call(_Fake().evaluate)
    assert result.details["existing"] is True
    prompts = result.details["judge_prompts"]
    assert len(prompts) == 1 and prompts[0]["name"] == "coherence"
    assert "rendered" in prompts[0]


def test_render_outside_capture_records_nothing():
    jp.render("coherence", answer="x")
    jp.begin_capture()
    used = jp.end_capture()
    assert used == []


def test_completeness_builds_and_evaluates_with_provenance():
    """Regression: 'completeness' is in the SDK/API supported set but had no
    worker evaluator, so warn-mode requests silently dropped it."""
    import json

    from jobs.helper.judge import LLMJudge
    from jobs.run import _build_evaluator

    def fn(prompt: str) -> str:
        assert "ANSWER:" in prompt
        return json.dumps({"score": 0.4, "missing": ["the second question"], "reason": "partial"})

    ev = _build_evaluator("completeness", LLMJudge(provider="openai", judge_fn=fn))
    assert ev.name == "ragas.completeness"

    result = jp.captured_call(ev.evaluate, question="a? and b?", answer="only a")
    assert result.score == 0.4
    assert result.details["missing"] == ["the second question"]
    (used,) = result.details["judge_prompts"]
    assert used["name"] == "completeness"


def test_reference_kwargs_route_to_reference_aware_metrics():
    """Dataset metric runs pass reference/contexts alongside question/answer/event
    (the exact kwargs auto_llm_eval builds); every metric evaluator must accept
    them, and reference-aware ones must actually grade against the reference."""
    import json

    from jobs.helper.hallucination import HallucinationEvaluator
    from jobs.helper.judge import LLMJudge
    from jobs.run import _build_evaluator

    seen = []

    def fn(prompt: str) -> str:
        seen.append(prompt)
        if "CLAIMS:" in prompt:
            return json.dumps({"verdicts": [{"claim": "c", "supported": True, "reason": ""}]})
        return json.dumps({"claims": ["c"], "score": 0.9, "reason": "ok",
                           "statements": ["s"], "verdicts": [{"statement": "s", "entailed": True}]})

    judge = LLMJudge(provider="openai", judge_fn=fn)
    kwargs = dict(
        question="q", answer="a", event={},
        reference="the expected output", contexts=["the expected output"],
    )
    ev = HallucinationEvaluator(judge=judge, threshold=0.7)
    result = ev.evaluate(**kwargs)
    assert result.score is not None
    # The reference (not general knowledge) drove the verification prompt.
    assert any("the expected output" in p for p in seen)

    # Every runnable metric must tolerate the full kwarg set.
    for name in ("faithfulness", "relevance", "toxicity", "coherence", "completeness"):
        _build_evaluator(name, judge).evaluate(**kwargs)


def test_rendered_is_capped():
    _install_org_override(ORG, "coherence", "PAD " * 3000 + "$answer")
    jp.set_org(ORG)
    jp.begin_capture()
    jp.render("coherence", answer="x")
    (rec,) = jp.end_capture()
    assert rec["truncated"] is True
    assert len(rec["rendered"]) == jp._RENDERED_CAP
