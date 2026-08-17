"""Client-defined deterministic scorer, as an evaluator.

Wraps :mod:`jobs.helper.code_scorer` in the same :class:`BaseEvaluator` shape as
the LLM-judge evaluators, so a code scorer and a judge scorer are persisted,
thresholded, and rendered identically. From the client's side both are just
``fluiq.eval(custom_judges={slug: threshold})`` — the difference is only which
``kind`` the saved scorer has.

Unlike a judge this makes no model call, so it costs nothing, adds no latency
worth measuring, and returns the same answer every time.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

from jobs.helper.base import BaseEvaluator, EvalResult
from jobs.helper.code_scorer import ScorerError, compile_scorer, run_scorer


class CodeScorerEvaluator(BaseEvaluator):
    """Evaluate an answer with a customer-authored expression."""

    def __init__(self, source: str, name: str, threshold: float = 0.5) -> None:
        super().__init__(threshold=threshold)
        self.source = source or ""
        self.name = name
        # Compiled once here rather than per example: a batch run scores hundreds
        # of rows with the same scorer, and a syntax error should surface on the
        # first one rather than being re-discovered on every one.
        self.tree = compile_scorer(self.source)

    def evaluate(
        self,
        *,
        question: str = "",
        answer: str = "",
        expected: str = "",
        metadata: Optional[Mapping[str, Any]] = None,
        **_: Any,
    ) -> EvalResult:
        score, reason = run_scorer(
            self.source,
            output=answer or "",
            expected=expected or "",
            input=question or "",
            metadata=metadata or {},
            tree=self.tree,
        )
        return self._result(
            score,
            reason=reason,
            details={
                "custom_scorer": self.name,
                "scorer_kind": "code",
                # The source is carried for the same reason a judge carries its
                # rendered prompt: a score you cannot see the derivation of is a
                # score nobody trusts.
                "source": self.source[:6000],
                "truncated": len(self.source) > 6000,
            },
        )


__all__ = ["CodeScorerEvaluator", "ScorerError"]
