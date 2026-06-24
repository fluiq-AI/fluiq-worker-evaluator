from typing import Any, Dict, List, Optional

from jobs.helper import judge_prompts
from jobs.helper.base import BaseEvaluator, EvalResult, _clamp_unit, _coerce_contexts
from jobs.helper.judge import LLMJudge

class Faithfulness(BaseEvaluator):
    """RAGAS Faithfulness: fraction of answer claims entailed by retrieved contexts."""

    name = "ragas.faithfulness"

    def __init__(self, judge: Optional[LLMJudge] = None, threshold: float = 0.7):
        super().__init__(threshold=threshold)
        self.judge = judge or LLMJudge()

    def evaluate(self, question: str, answer: str, contexts: Any, **_: Any) -> EvalResult:
        ctx_list = _coerce_contexts(contexts)
        if not answer.strip():
            return self._result(1.0, "empty answer", {"statements": [], "verdicts": []})
        if not ctx_list:
            return self._result(0.0, "no contexts provided", {"statements": [], "verdicts": []})
        statements = _judge_list(
            self.judge,
            judge_prompts.render("faithfulness_statements", question=question, answer=answer),
            "statements",
        )
        if not statements:
            return self._result(1.0, "no statements extracted", {"statements": []})
        joined_ctx = "\n\n".join(ctx_list)
        verdicts = _judge_list(
            self.judge,
            judge_prompts.render(
                "faithfulness_verify",
                context=joined_ctx,
                statements="\n".join(f"- {s}" for s in statements),
            ),
            "verdicts",
        )
        if not verdicts:
            return self._result(0.0, "judge returned no verdicts", {"statements": statements})
        entailed = sum(1 for v in verdicts if isinstance(v, dict) and v.get("entailed") is True)
        score = _clamp_unit(entailed / len(verdicts))
        return self._result(
            score,
            f"{entailed}/{len(verdicts)} statements entailed by context",
            {"statements": statements, "verdicts": verdicts},
        )


class AnswerRelevancy(BaseEvaluator):
    """RAGAS Answer Relevancy: how directly the answer addresses the question."""

    name = "ragas.answer_relevancy"

    def __init__(self, judge: Optional[LLMJudge] = None, threshold: float = 0.7):
        super().__init__(threshold=threshold)
        self.judge = judge or LLMJudge()

    def evaluate(self, question: str, answer: str, **_: Any) -> EvalResult:
        if not answer.strip():
            return self._result(0.0, "empty answer")
        data = self.judge.judge_json(
            judge_prompts.render("answer_relevancy", question=question, answer=answer)
        )
        score = _clamp_unit(data.get("score"))
        if data.get("noncommittal") is True:
            score = 0.0
        return self._result(score, str(data.get("reason") or ""), data)


class ContextPrecision(BaseEvaluator):
    """RAGAS Context Precision: rank-weighted relevance of retrieved contexts."""

    name = "ragas.context_precision"

    def __init__(self, judge: Optional[LLMJudge] = None, threshold: float = 0.7):
        super().__init__(threshold=threshold)
        self.judge = judge or LLMJudge()

    def evaluate(self, question: str, contexts: Any, reference: Optional[str] = None, **_: Any) -> EvalResult:
        ctx_list = _coerce_contexts(contexts)
        if not ctx_list:
            return self._result(0.0, "no contexts provided")
        flags: List[bool] = []
        for c in ctx_list:
            data = self.judge.judge_json(
                judge_prompts.render(
                    "context_precision",
                    question=question,
                    context=c,
                    reference_clause=(" given the REFERENCE answer" if reference else ""),
                    reference_block=(f"REFERENCE: {reference}\n" if reference else ""),
                )
            )
            flags.append(bool(data.get("useful")))
        if not any(flags):
            return self._result(0.0, "no useful contexts", {"flags": flags})
        seen = 0
        precisions: List[float] = []
        for i, useful in enumerate(flags, start=1):
            if useful:
                seen += 1
                precisions.append(seen / i)
        score = _clamp_unit(sum(precisions) / max(seen, 1))
        return self._result(score, f"{seen}/{len(flags)} contexts useful", {"flags": flags})


class ContextRecall(BaseEvaluator):
    """RAGAS Context Recall: fraction of reference statements covered by contexts."""

    name = "ragas.context_recall"

    def __init__(self, judge: Optional[LLMJudge] = None, threshold: float = 0.7):
        super().__init__(threshold=threshold)
        self.judge = judge or LLMJudge()

    def evaluate(self, question: str, reference: str, contexts: Any, **_: Any) -> EvalResult:
        ctx_list = _coerce_contexts(contexts)
        if not reference.strip():
            return self._result(0.0, "no reference provided")
        if not ctx_list:
            return self._result(0.0, "no contexts provided")
        joined_ctx = "\n\n".join(ctx_list)
        verdicts = _judge_list(
            self.judge,
            judge_prompts.render(
                "context_recall",
                question=question,
                reference=reference,
                context=joined_ctx,
            ),
            "verdicts",
        )
        if not verdicts:
            return self._result(0.0, "judge returned no verdicts")
        attributed = sum(1 for v in verdicts if isinstance(v, dict) and v.get("attributed") is True)
        score = _clamp_unit(attributed / len(verdicts))
        return self._result(
            score,
            f"{attributed}/{len(verdicts)} reference statements covered",
            {"verdicts": verdicts},
        )

class Toxicity(BaseEvaluator):
    """Measures whether the generated answer contains toxic, harmful, abusive,
    hateful, or unsafe content."""

    name = "ragas.toxicity"

    def __init__(self, judge: Optional[LLMJudge] = None, threshold: float = 0.2):
        # lower is better for toxicity, but we normalize to "safe score"
        super().__init__(threshold=threshold)
        self.judge = judge or LLMJudge()

    def evaluate(
        self,
        answer: str,
        question: Optional[str] = None,
        **_: Any,
    ) -> EvalResult:
        if not answer.strip():
            return self._result(1.0, "empty answer")

        data = self.judge.judge_json(
            judge_prompts.render(
                "toxicity",
                answer=answer,
                question_block=judge_prompts.question_block(question),
            )
        )

        toxic = bool(data.get("toxic"))
        severity = float(data.get("severity", 0.0))
        severity = _clamp_unit(severity)

        # Convert toxicity severity into a safety score
        score = 1.0 - severity if toxic else 1.0

        return self._result(
            score,
            data.get("reason", "toxicity evaluation completed"),
            {
                "toxic": toxic,
                "severity": severity,
            },
        )


class Coherence(BaseEvaluator):
    """Measures logical flow, readability, and consistency of the answer."""

    name = "ragas.coherence"

    def __init__(self, judge: Optional[LLMJudge] = None, threshold: float = 0.7):
        super().__init__(threshold=threshold)
        self.judge = judge or LLMJudge()

    def evaluate(
        self,
        answer: str,
        question: Optional[str] = None,
        **_: Any,
    ) -> EvalResult:
        if not answer.strip():
            return self._result(0.0, "empty answer")

        data = self.judge.judge_json(
            judge_prompts.render(
                "coherence",
                answer=answer,
                question_block=judge_prompts.question_block(question),
            )
        )

        score = _clamp_unit(float(data.get("score", 0.0)))

        return self._result(
            score,
            data.get("reason", "coherence evaluation completed"),
            {
                "raw_score": score,
            },
        )

class Ragas:
    """Convenience runner that executes the four core RAGAS metrics."""

    def __init__(self, judge: Optional[LLMJudge] = None, threshold: float = 0.7):
        judge = judge or LLMJudge()
        self.faithfulness      = Faithfulness(judge=judge, threshold=threshold)
        self.answer_relevancy  = AnswerRelevancy(judge=judge, threshold=threshold)
        self.context_precision = ContextPrecision(judge=judge, threshold=threshold)
        self.context_recall    = ContextRecall(judge=judge, threshold=threshold)
        self.toxicity = Toxicity(judge=judge, threshold=threshold)
        self.coherence = Coherence(judge=judge, threshold=threshold)

    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: Any,
        reference: Optional[str] = None,
    ) -> Dict[str, EvalResult]:
        results: Dict[str, EvalResult] = {
            "faithfulness":      self.faithfulness.evaluate(question=question, answer=answer, contexts=contexts),
            "answer_relevancy":  self.answer_relevancy.evaluate(question=question, answer=answer),
            "context_precision": self.context_precision.evaluate(question=question, contexts=contexts, reference=reference),
            "toxicity": self.toxicity.evaluate(question=question, answer=answer),
            "coherence": self.coherence.evaluate(answer=answer, question=question)
        }
        if reference:
            results["context_recall"] = self.context_recall.evaluate(
                question=question, reference=reference, contexts=contexts,
            )
        return results


def _judge_list(judge: LLMJudge, prompt: str, key: str) -> List[Any]:
    data = judge.judge_json(prompt)
    raw = data.get(key) or []
    if not isinstance(raw, list):
        return []
    if key == "statements":
        return [str(s).strip() for s in raw if str(s).strip()]
    return [v for v in raw if isinstance(v, dict)]
