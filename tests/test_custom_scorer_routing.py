"""A client scorer slug resolves to either an LLM judge or a code scorer.

The client references both identically — ``fluiq.eval(custom_judges={slug: t})``
— so everything that decides which one runs lives in ``_run_custom_judges``. The
routing is worth pinning because getting it wrong is silent: a code scorer sent
down the judge path would be handed to a model as a prompt and score whatever
that model felt like.
"""
from __future__ import annotations

import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jobs import run as run_mod  # noqa: E402


class _FakeUsage:
    def drain(self):
        return (11, 22, 1)


class _FakeJudge:
    provider = "openai"
    model = "gpt-4o-mini"
    usage = _FakeUsage()

    def judge_json(self, _prompt):
        return {"score": 0.42, "reason": "because"}

    def judge_choice(self, _prompt, _labels):
        return {"choice": "Y", "reason": "it did"}


class _Persisted:
    """Captures what _run_custom_judges tried to write."""

    def __init__(self):
        self.rows = []

    async def __call__(self, org, prefix, trace_id, **kw):
        self.rows.append(kw)


@pytest.fixture
def wired(monkeypatch):
    """Patch persistence and the scorer lookup; return handles to both."""
    persisted = _Persisted()
    monkeypatch.setattr(run_mod, "_persist_eval_result", persisted)

    scorers = {}

    class _PG:
        async def fetch_custom_scorer(self, _org, slug):
            return scorers.get(slug)

    import db.postgres as pg_mod

    monkeypatch.setattr(pg_mod, "postgres_client", _PG())
    return persisted, scorers


def _run(custom, **kw):
    args = dict(
        organization_id="org-1",
        api_key_prefix="pk_",
        trace_id="t-1",
        judge=_FakeJudge(),
        question="why is my order late?",
        answer="Sorry — your refund is on the way.",
        context="apologise and mention the refund",
    )
    args.update(kw)
    asyncio.run(run_mod._run_custom_judges(custom, **args))


# ── Routing ──────────────────────────────────────────────────────────────────

def test_code_scorer_is_evaluated_as_code(wired):
    persisted, scorers = wired
    scorers["mentions-refund"] = ("code", "contains(output, 'refund')", {})

    _run({"mentions-refund": 0.5})

    assert len(persisted.rows) == 1
    row = persisted.rows[0]
    assert row["metric"] == "mentions-refund"
    assert row["result"].score == 1.0
    assert row["result"].details["scorer_kind"] == "code"


def test_judge_scorer_still_goes_to_the_model(wired):
    persisted, scorers = wired
    scorers["on-brand"] = ("judge", "Grade this: {{answer}}", {})

    _run({"on-brand": 0.5})

    row = persisted.rows[0]
    assert row["result"].score == pytest.approx(0.42)
    assert "custom_judge" in row["result"].details


def test_a_code_scorer_records_no_judge_model(wired):
    """It made no model call. Attributing one would misreport both the cost and
    the provenance of the score."""
    persisted, scorers = wired
    scorers["short-enough"] = ("code", "len(output) < 500", {})

    _run({"short-enough": 0.5})

    assert persisted.rows[0]["judge"] is None


def test_a_judge_scorer_keeps_its_judge(wired):
    persisted, scorers = wired
    scorers["on-brand"] = ("judge", "Grade: {{answer}}", {})

    _run({"on-brand": 0.5})

    assert persisted.rows[0]["judge"] is not None


def test_a_judge_with_choices_scores_from_its_table(wired):
    persisted, scorers = wired
    scorers["apologised"] = (
        "judge", "Did it apologise? {{answer}}",
        {"choices": [{"label": "Y", "score": 1.0}, {"label": "N", "score": 0.0}]},
    )

    _run({"apologised": 0.5})

    row = persisted.rows[0]
    assert row["result"].score == 1.0          # from the table, not the model
    assert row["result"].details["choice"] == "Y"


def test_a_judge_without_choices_is_unaffected(wired):
    """Config is optional; every scorer saved before choices existed has none."""
    persisted, scorers = wired
    scorers["on-brand"] = ("judge", "Grade {{answer}}", {})

    _run({"on-brand": 0.5})

    assert persisted.rows[0]["result"].score == pytest.approx(0.42)


def test_a_malformed_choice_set_skips_the_scorer(wired):
    """Falling back to free scoring would change what the scorer means without
    saying so, and the resulting number would look entirely legitimate."""
    persisted, scorers = wired
    scorers["broken"] = ("judge", "Grade {{answer}}", {"choices": [{"label": "only-one", "score": 1}]})
    scorers["fine"] = ("judge", "Grade {{answer}}", {})

    _run({"broken": 0.5, "fine": 0.5})

    assert [r["metric"] for r in persisted.rows] == ["fine"]


# ── What a code scorer can see ───────────────────────────────────────────────

def test_code_scorer_reads_the_generated_output(wired):
    persisted, scorers = wired
    scorers["s"] = ("code", "output == 'exact'", {})

    _run({"s": 0.5}, answer="exact")

    assert persisted.rows[0]["result"].score == 1.0


def test_code_scorer_reads_the_expected_output(wired):
    """`context` carries the reference on this path, and a code scorer sees it as
    `expected` — the name that means something to whoever writes the scorer."""
    persisted, scorers = wired
    scorers["s"] = ("code", "output == expected", {})

    _run({"s": 0.5}, answer="same", context="same")

    assert persisted.rows[0]["result"].score == 1.0


def test_code_scorer_reads_the_question_as_input(wired):
    persisted, scorers = wired
    scorers["s"] = ("code", "contains(input, 'late')", {})

    _run({"s": 0.5}, question="my order is late")

    assert persisted.rows[0]["result"].score == 1.0


def test_code_scorer_reads_example_metadata(wired):
    persisted, scorers = wired
    scorers["s"] = ("code", "metadata.get('tier') == 'gold'", {})

    _run({"s": 0.5}, example_metadata={"tier": "gold"})

    assert persisted.rows[0]["result"].score == 1.0


def test_metadata_defaults_to_empty_rather_than_none(wired):
    """A scorer that reads metadata must not explode on a trace that has none."""
    persisted, scorers = wired
    scorers["s"] = ("code", "metadata.get('tier') == 'gold'", {})

    _run({"s": 0.5})

    assert persisted.rows[0]["result"].score == 0.0


# ── Failure isolation ────────────────────────────────────────────────────────

def test_an_unknown_slug_is_skipped_not_fatal(wired):
    persisted, _scorers = wired
    _run({"never-saved": 0.5})
    assert persisted.rows == []


def test_a_broken_code_scorer_skips_only_itself(wired):
    """One bad scorer must not cost the other scorers on the same example."""
    persisted, scorers = wired
    scorers["broken"] = ("code", "this is not python(", {})
    scorers["fine"] = ("code", "True", {})

    _run({"broken": 0.5, "fine": 0.5})

    assert [r["metric"] for r in persisted.rows] == ["fine"]


def test_a_scorer_returning_the_wrong_type_skips_only_itself(wired):
    persisted, scorers = wired
    scorers["stringy"] = ("code", "'not a number'", {})
    scorers["fine"] = ("code", "0.5", {})

    _run({"stringy": 0.5, "fine": 0.5})

    assert [r["metric"] for r in persisted.rows] == ["fine"]


def test_threshold_decides_passed(wired):
    persisted, scorers = wired
    scorers["s"] = ("code", "0.6", {})

    _run({"s": 0.9})
    assert persisted.rows[0]["result"].passed is False

    persisted.rows.clear()
    _run({"s": 0.5})
    assert persisted.rows[0]["result"].passed is True


def test_empty_scorer_map_does_nothing(wired):
    persisted, _ = wired
    _run({})
    assert persisted.rows == []
