"""Client-defined LLM-as-judge.

Runs a judge prompt the customer authored on the Prompts page (``kind='judge'``)
and referenced by slug in ``fluiq.eval(custom_judges={slug: threshold})``. The
template uses ``string.Template`` ``$question`` / ``$answer`` / ``$context``
placeholders and is expected to return ``{"score": float, "reason": str}``.

The judge call goes through the same :class:`LLMJudge` (and response cache) as
the built-in evaluators, so custom judges honour the configured judge provider /
model and benefit from caching.
"""
from __future__ import annotations

from string import Template
from typing import Any

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
    """Evaluate an answer with a customer-authored judge prompt."""

    def __init__(
        self,
        judge: LLMJudge,
        template: str,
        name: str,
        threshold: float = 0.5,
    ) -> None:
        super().__init__(threshold=threshold)
        self.judge = judge
        self.template = template or ""
        self.name = name

    def evaluate(
        self,
        *,
        question: str = "",
        answer: str = "",
        context: str = "",
        **_: Any,
    ) -> EvalResult:
        rendered = Template(_ensure_output_contract(self.template)).safe_substitute(
            question=question or "",
            answer=answer or "",
            context=context or question or "",
        )
        data = self.judge.judge_json(rendered)
        score = _clamp_unit(data.get("score"))
        reason = str(data.get("reason") or "")
        return self._result(score, reason=reason, details={"custom_judge": self.name})
