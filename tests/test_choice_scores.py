"""Choice scores: the judge picks a label, a table turns it into a number.

The value of this is that the number never comes from the model. So the tests
that matter are the ones about what happens when a model answers something the
author did not offer — that must be reported, never quietly scored.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jobs.helper import choice_scores  # noqa: E402
from jobs.helper.choice_scores import ChoiceError, parse_choices, resolve  # noqa: E402
from jobs.helper.custom_judge import CustomJudgeEvaluator  # noqa: E402
from jobs.helper.judge import _gemini_choice_schema, _openai_choice_schema  # noqa: E402

YN = [{"label": "Y", "score": 1.0}, {"label": "N", "score": 0.0}]


# ── Parsing a choice set ─────────────────────────────────────────────────────

def test_absent_choices_mean_free_scoring():
    """A scorer without choices must keep the behaviour it has always had."""
    for empty in (None, "", [], {}):
        assert parse_choices(empty) is None


def test_a_valid_set_round_trips():
    assert parse_choices(YN) == YN


def test_one_option_is_not_a_choice():
    with pytest.raises(ChoiceError, match="at least two"):
        parse_choices([{"label": "Y", "score": 1}])


def test_too_many_options_are_rejected():
    many = [{"label": f"L{i}", "score": 0.5} for i in range(choice_scores.MAX_CHOICES + 1)]
    with pytest.raises(ChoiceError, match="at most"):
        parse_choices(many)


def test_duplicate_labels_are_rejected():
    """Two options the judge cannot tell apart make the score meaningless."""
    with pytest.raises(ChoiceError, match="Duplicate"):
        parse_choices([{"label": "Y", "score": 1}, {"label": "y", "score": 0}])


def test_a_blank_label_is_rejected():
    with pytest.raises(ChoiceError, match="needs a label"):
        parse_choices([{"label": "  ", "score": 1}, {"label": "N", "score": 0}])


def test_a_non_numeric_score_is_rejected():
    with pytest.raises(ChoiceError, match="numeric score"):
        parse_choices([{"label": "Y", "score": "high"}, {"label": "N", "score": 0}])


def test_a_score_outside_the_unit_range_is_rejected():
    with pytest.raises(ChoiceError, match="must be 0 to 1"):
        parse_choices([{"label": "Y", "score": 5}, {"label": "N", "score": 0}])


def test_scores_need_not_be_unique():
    """Two labels can legitimately mean the same grade ("N/A" and "No")."""
    assert parse_choices([{"label": "a", "score": 0.5}, {"label": "b", "score": 0.5}])


# ── Resolving an answer ──────────────────────────────────────────────────────

def test_the_score_comes_from_the_table():
    assert resolve(YN, "Y") == (1.0, "Y")
    assert resolve(YN, "N") == (0.0, "N")


def test_matching_ignores_case_and_padding():
    """A judge answering "yes" for a "Yes" label has understood the question."""
    assert resolve([{"label": "Yes", "score": 1}, {"label": "No", "score": 0}], " yes ")[0] == 1.0


def test_an_off_menu_answer_raises_rather_than_scoring_zero():
    """Scoring it 0 would be indistinguishable from a genuine failing grade."""
    with pytest.raises(ChoiceError, match="not one of"):
        resolve(YN, "Maybe")


def test_an_empty_answer_raises():
    with pytest.raises(ChoiceError, match="no choice"):
        resolve(YN, "")


def test_the_error_names_the_options_that_were_offered():
    with pytest.raises(ChoiceError, match="Y, N"):
        resolve(YN, "unsure")


# ── The prompt the judge sees ────────────────────────────────────────────────

def test_the_options_are_stated_in_the_prompt():
    """Stated even where the provider enforces the enum: a model told what the
    options mean chooses better than one merely prevented from saying otherwise."""
    text = choice_scores.describe(YN)
    assert '"Y"' in text and '"N"' in text
    assert "choice" in text


# ── Provider schemas ─────────────────────────────────────────────────────────

def test_openai_schema_pins_the_enum_strictly():
    schema = _openai_choice_schema(["Y", "N"])
    inner = schema["json_schema"]
    assert inner["strict"] is True
    assert inner["schema"]["properties"]["choice"]["enum"] == ["Y", "N"]
    assert inner["schema"]["additionalProperties"] is False
    assert set(inner["schema"]["required"]) == {"choice", "reason"}


def test_gemini_schema_pins_the_enum():
    schema = _gemini_choice_schema(["Y", "N"])
    assert schema["properties"]["choice"]["enum"] == ["Y", "N"]
    # Gemini's OpenAPI subset has no additionalProperties; sending it errors.
    assert "additionalProperties" not in schema


def test_the_enum_survives_the_wrapper_chain():
    """In production ``_judge_fn`` is always set — the judge cache and the BYOK
    error reporter both wrap the provider call. If ``judge_choice`` dropped the
    labels when a wrapper was present, constrained decoding would work only on a
    machine with neither configured, which is to say never."""
    from jobs.helper.judge import LLMJudge

    seen = {}

    def wrapper(prompt, **params):
        seen.update(params)
        return '{"choice": "Y", "reason": "ok"}'

    judge = LLMJudge(provider="openai", model="gpt-4o-mini")
    judge._judge_fn = wrapper
    judge.judge_choice("grade this", ["Y", "N"])

    assert seen.get("choice_labels") == ["Y", "N"]


def test_a_prompt_only_hook_still_works():
    """Older/simpler hooks take just a prompt. The options are in the prompt text
    and the caller validates the answer, so this degrades rather than breaking."""
    from jobs.helper.judge import LLMJudge

    judge = LLMJudge(provider="openai", model="gpt-4o-mini")
    judge._judge_fn = lambda prompt: '{"choice": "N", "reason": "no"}'
    assert judge.judge_choice("grade", ["Y", "N"])["choice"] == "N"


# ── End to end through the evaluator ─────────────────────────────────────────

class _StubJudge:
    """Records what it was asked, answers what it was told to."""

    provider = "openai"
    model = "gpt-4o-mini"

    def __init__(self, answer):
        self.answer = answer
        self.seen_prompt = ""
        self.seen_labels = None

    def judge_choice(self, prompt, labels):
        self.seen_prompt, self.seen_labels = prompt, labels
        return self.answer

    def judge_json(self, prompt):
        self.seen_prompt = prompt
        return self.answer


def test_a_chosen_label_becomes_its_score():
    judge = _StubJudge({"choice": "Y", "reason": "it apologised"})
    ev = CustomJudgeEvaluator(
        judge=judge, template="Did it apologise? {{answer}}",
        name="apology", threshold=0.5, choices=YN,
    )
    result = ev.evaluate(answer="Sorry about that.")

    assert result.score == 1.0
    assert result.passed is True
    assert result.reason == "it apologised"
    assert result.details["choice"] == "Y"


def test_the_offered_labels_reach_the_provider():
    judge = _StubJudge({"choice": "N", "reason": "no"})
    CustomJudgeEvaluator(
        judge=judge, template="{{answer}}", name="x", choices=YN,
    ).evaluate(answer="hi")
    assert judge.seen_labels == ["Y", "N"]


def test_the_options_reach_the_prompt():
    judge = _StubJudge({"choice": "Y", "reason": ""})
    CustomJudgeEvaluator(
        judge=judge, template="Grade {{answer}}", name="x", choices=YN,
    ).evaluate(answer="hi")
    assert '"Y"' in judge.seen_prompt


def test_a_choice_set_replaces_the_free_score_contract():
    """Asking for a choice *and* a float invites the model to answer with a float."""
    judge = _StubJudge({"choice": "Y", "reason": ""})
    CustomJudgeEvaluator(
        judge=judge, template="Grade {{answer}}", name="x", choices=YN,
    ).evaluate(answer="hi")
    assert "float between 0 and 1" not in judge.seen_prompt


def test_an_off_menu_answer_surfaces_as_an_error():
    judge = _StubJudge({"choice": "Probably", "reason": ""})
    ev = CustomJudgeEvaluator(judge=judge, template="{{answer}}", name="x", choices=YN)
    with pytest.raises(ChoiceError):
        ev.evaluate(answer="hi")


def test_without_choices_the_judge_still_returns_a_free_score():
    judge = _StubJudge({"score": 0.42, "reason": "meh"})
    result = CustomJudgeEvaluator(
        judge=judge, template="Grade {{answer}}", name="x",
    ).evaluate(answer="hi")

    assert result.score == pytest.approx(0.42)
    assert "choice" not in result.details
    assert "float between 0 and 1" in judge.seen_prompt


def test_threshold_applies_to_the_mapped_score():
    partial = [
        {"label": "great", "score": 1.0},
        {"label": "ok", "score": 0.5},
        {"label": "bad", "score": 0.0},
    ]
    judge = _StubJudge({"choice": "ok", "reason": ""})
    ev = CustomJudgeEvaluator(
        judge=judge, template="{{answer}}", name="x", threshold=0.8, choices=partial,
    )
    result = ev.evaluate(answer="hi")
    assert result.score == 0.5
    assert result.passed is False


def test_the_choice_set_is_recorded_on_the_result():
    """A score you cannot see the derivation of is a score nobody trusts."""
    judge = _StubJudge({"choice": "Y", "reason": ""})
    result = CustomJudgeEvaluator(
        judge=judge, template="{{answer}}", name="x", choices=YN,
    ).evaluate(answer="hi")
    assert result.details["choices"] == YN
