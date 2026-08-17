"""Client-defined LLM-as-judge.

Runs a judge prompt the customer authored on the Prompts page (``kind='judge'``)
and referenced by slug in ``fluiq.eval(custom_judges={slug: threshold})``. The
template uses ``{{question}}`` / ``{{answer}}`` / ``{{context}}`` placeholders
(the legacy ``$answer`` form still substitutes) and is expected to return
``{"score": float, "reason": str}``.

The judge call goes through the same :class:`LLMJudge` (and response cache) as
the built-in evaluators, so custom judges honour the configured judge provider /
model and benefit from caching.
"""
from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence

from jobs.helper import choice_scores, judge_prompts
from jobs.helper.base import BaseEvaluator, EvalResult, _clamp_unit
from jobs.helper.judge import LLMJudge

_OUTPUT_CONTRACT = (
    '\n\nReturn ONLY a JSON object: '
    '{"score": <float between 0 and 1>, "reason": "<short explanation>"}'
)


def _ensure_output_contract(template: str) -> str:
    """Append the score/reason JSON contract unless the template already asks for it."""
    return template if "score" in template.lower() else template + _OUTPUT_CONTRACT


class CustomJudgeEvaluator(BaseEvaluator):
    """Evaluate an answer with a customer-authored judge prompt.

    With ``choices``, the judge picks one of a fixed set of written labels and
    the score comes from the author's own table — see
    :mod:`jobs.helper.choice_scores` for why that is more stable than asking for
    a number. Without them, it keeps the free 0..1 behaviour it has always had.
    """

    def __init__(
        self,
        judge: LLMJudge,
        template: str,
        name: str,
        threshold: float = 0.5,
        choices: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> None:
        super().__init__(threshold=threshold)
        self.judge = judge
        self.template = template or ""
        self.name = name
        self.choices: Optional[List[dict]] = (
            [dict(c) for c in choices] if choices else None
        )

    def _render(self, question: str, answer: str, context: str) -> str:
        # A choice set replaces the free-score contract rather than adding to it:
        # asking for both invites the model to answer with a number anyway.
        body = (
            self.template + choice_scores.describe(self.choices)
            if self.choices
            else _ensure_output_contract(self.template)
        )
        return judge_prompts.substitute(
            body,
            {
                "question": question or "",
                "answer":   answer or "",
                "context":  context or question or "",
            },
        )

    def evaluate(
        self,
        *,
        question: str = "",
        answer: str = "",
        context: str = "",
        **_: Any,
    ) -> EvalResult:
        rendered = self._render(question, answer, context)

        if self.choices:
            data = self.judge.judge_choice(
                rendered, choice_scores.labels_of(self.choices),
            )
            # Raises on an off-menu answer, which the caller reports rather than
            # scoring — a 0 there would be indistinguishable from a real fail.
            score, chosen = choice_scores.resolve(self.choices, data.get("choice"))
            reason = str(data.get("reason") or "")
            extra = {"choice": chosen, "choices": self.choices}
        else:
            data = self.judge.judge_json(rendered)
            score = _clamp_unit(data.get("score"))
            reason = str(data.get("reason") or "")
            extra = {}

        return self._result(
            score,
            reason=reason,
            details={
                "custom_judge": self.name,
                **extra,
                # Same provenance shape the registry-backed evaluators emit, so
                # the dashboard renders custom judges' prompts identically.
                "judge_prompts": [{
                    "name": self.name,
                    "source": "custom",
                    "version": None,
                    "calls": 1,
                    "truncated": len(rendered) > 6000,
                    "rendered": rendered[:6000],
                }],
            },
        )
