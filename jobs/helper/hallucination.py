from typing import Any, List, Optional

from jobs.helper.base import BaseEvaluator, EvalResult, _clamp_unit, _coerce_contexts
from jobs.helper.judge import LLMJudge


_CLAIMS_PROMPT = (
    "Extract every standalone factual claim from the ANSWER below. "
    "Return JSON: {{\"claims\": [\"claim 1\", \"claim 2\", ...]}}.\n\n"
    "ANSWER:\n{answer}"
)

_VERIFY_PROMPT = (
    "You are checking whether each CLAIM is supported by the REFERENCE. "
    "A claim is SUPPORTED only if the reference entails it; if the reference "
    "neither states nor implies it, mark it UNSUPPORTED. Speculation, added "
    "details, and contradictions are UNSUPPORTED.\n\n"
    "REFERENCE:\n{reference}\n\n"
    "CLAIMS:\n{claims}\n\n"
    "Return JSON: {{\"verdicts\": [{{\"claim\": str, \"supported\": bool, "
    "\"reason\": str}}]}}"
)


class HallucinationEvaluator(BaseEvaluator):
    """Hallucination detector. Score 1.0 = no hallucinations, 0.0 = all claims unsupported."""

    name = "hallucination"

    def __init__(self, judge: Optional[LLMJudge] = None, threshold: float = 0.7):
        super().__init__(threshold=threshold)
        self.judge = judge or LLMJudge()

    def evaluate(self, answer: str, question: Optional[str] = None, contexts: Any = None, reference: Optional[str] = None, **_: Any) -> EvalResult:
        if not answer or not answer.strip():
            return self._result(1.0, "empty answer; nothing to hallucinate", {"claims": [], "verdicts": []})

        reference_text = _build_reference(contexts, reference)
        if not reference_text:
            # No retrieval context — use LLM general knowledge to assess factual accuracy.
            data = self.judge.judge_json(
                "Evaluate whether the ANSWER contains any factual errors or hallucinations "
                "based on your general knowledge. Score 1.0 = fully accurate, 0.0 = completely "
                "hallucinated or wrong.\n\n"
                "Return JSON: {\"score\": float, \"reason\": str}.\n\n"
                + (f"QUESTION: {question}\n" if question else "")
                + f"ANSWER: {answer}"
            )
            score = _clamp_unit(data.get("score"), default=0.5)
            return self._result(score, str(data.get("reason") or ""), data)

        claims = self._extract_claims(answer)
        if not claims:
            return self._result(1.0, "no factual claims extracted from answer", {"claims": [], "verdicts": []})

        verdicts = self._verify_claims(reference_text, claims)
        if not verdicts:
            return self._result(0.0, "judge returned no verdicts", {"claims": claims, "verdicts": []})

        supported = sum(1 for v in verdicts if v.get("supported") is True)
        score = _clamp_unit(supported / len(verdicts))
        unsupported = [v for v in verdicts if v.get("supported") is not True]
        reason = (
            f"{supported}/{len(verdicts)} claims supported"
            if not unsupported
            else f"{len(unsupported)} unsupported claim(s) of {len(verdicts)}"
        )
        return self._result(score, reason, {
            "claims": claims,
            "verdicts": verdicts,
            "supported_count": supported,
            "total_claims": len(verdicts),
        })

    def _extract_claims(self, answer: str) -> List[str]:
        data = self.judge.judge_json(_CLAIMS_PROMPT.format(answer=answer))
        raw = data.get("claims") or []
        return [str(c).strip() for c in raw if isinstance(raw, list) and str(c).strip()]

    def _verify_claims(self, reference: str, claims: List[str]) -> List[dict]:
        prompt = _VERIFY_PROMPT.format(
            reference=reference,
            claims="\n".join(f"- {c}" for c in claims),
        )
        data = self.judge.judge_json(prompt)
        raw = data.get("verdicts") or []
        if not isinstance(raw, list):
            return []
        return [
            {
                "claim":     str(v.get("claim", "")).strip(),
                "supported": bool(v.get("supported")),
                "reason":    str(v.get("reason", "")).strip() or None,
            }
            for v in raw if isinstance(v, dict)
        ]


def _build_reference(contexts: Any, reference: Optional[str]) -> str:
    parts: List[str] = [c.strip() for c in _coerce_contexts(contexts) if c.strip()]
    if reference and reference.strip():
        parts.append(reference.strip())
    return "\n\n".join(parts)
