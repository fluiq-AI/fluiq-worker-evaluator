"""Layer 2.5 — Retrieval & Ranking Quality (LLM-as-judge + IR metrics).

A run can score well on tool selection and trajectory while its retriever
returned garbage, because nothing upstream of this layer looked at what came
back from the vector store. This layer closes that hole.

It grades three distinct failures that are usually collapsed into one number:

1. **Relevance** — were the returned documents actually about the query?
   Rank-weighted precision, the same shape as RAGAS ContextPrecision, so the
   number is comparable to the per-span metric users already see.
2. **Ranking** — were the *best* documents at the top? nDCG over graded
   relevance labels. A retriever that finds the right document and buries it at
   rank 8 has a real defect that precision alone does not express, and it is a
   different fix (reranker) from a recall problem (embeddings/chunking).
3. **Utilisation** — did the final answer actually use what was retrieved?
   Retrieving perfectly and then ignoring it is the RAG failure teams hit most,
   and it is only visible at run level, which is why it lives here and not in
   the per-span evaluator.

Cost note: the judge is called **once per retrieval step** and grades every
document in that step in a single pass. RAGAS ContextPrecision issues one call
per document; at agentic depth over a run with several retrievals that is a
pathological number of judge calls, so this layer deliberately batches.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from jobs.agentic.schema import Retrieval
from jobs.helper import judge_prompts
from jobs.helper.base import BaseEvaluator, EvalResult, _clamp_unit
from jobs.helper.judge import LLMJudge

# Graded relevance scale handed to the judge. Graded (not binary) labels are
# what nDCG needs — with binary labels nDCG degenerates and stops distinguishing
# "perfect doc at rank 3" from "merely adequate doc at rank 3".
MAX_GRADE = 3

# How much each sub-score contributes to the layer verdict. Relevance leads
# because a retriever that returns the wrong documents cannot be rescued by
# ordering them well. Utilisation is smallest because a low score there is often
# a *generator* fault rather than a retriever fault, and we do not want the
# retrieval layer to absorb the blame for it.
W_RELEVANCE = 0.5
W_NDCG = 0.3
W_UTILISATION = 0.2

# Documents are truncated before going to the judge: retrieved chunks can be
# large and we send several per call.
_DOC_CHARS = 700
_MAX_DOCS_JUDGED = 20


def _render_docs(retrieval: Retrieval) -> str:
    lines: List[str] = []
    for doc in retrieval.docs[:_MAX_DOCS_JUDGED]:
        text = (doc.text or "").strip().replace("\n", " ")
        if len(text) > _DOC_CHARS:
            text = text[:_DOC_CHARS] + "…"
        lines.append(f"[{doc.rank}] {text or '(empty document)'}")
    return "\n".join(lines) or "(the retriever returned nothing)"


def dcg(grades: List[int]) -> float:
    """Discounted cumulative gain with the standard exponential gain."""
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(grades))


def ndcg(grades: List[int]) -> Optional[float]:
    """Normalized DCG, or ``None`` when ranking cannot be judged.

    Returns ``None`` in two cases, and the distinction matters:

    - **fewer than two documents** — with one result any ordering is trivially
      perfect, so reporting 1.0 would silently inflate the layer score.
    - **no relevant documents at all** — that is a recall failure, already
      punished by the relevance term. Scoring ranking as 0 too would penalise
      the same defect twice.
    """
    if len(grades) < 2:
        return None
    ideal = dcg(sorted(grades, reverse=True))
    if ideal <= 0:
        return None
    return _clamp_unit(dcg(grades) / ideal)


def rank_weighted_precision(grades: List[int], useful_at: int = 1) -> float:
    """Mean precision@k over the ranks holding a useful document.

    Identical in shape to RAGAS ContextPrecision so the two are comparable;
    ``useful_at`` is the grade at or above which a document counts as useful.
    """
    if not grades:
        return 0.0
    seen = 0
    precisions: List[float] = []
    for i, g in enumerate(grades, start=1):
        if g >= useful_at:
            seen += 1
            precisions.append(seen / i)
    if not seen:
        return 0.0
    return _clamp_unit(sum(precisions) / seen)


class RetrievalQuality(BaseEvaluator):
    """Score what the retriever returned, how it ordered it, and whether the
    agent's answer used it."""

    name = "agentic.retrieval_quality"

    def __init__(self, judge: Optional[LLMJudge] = None, threshold: float = 0.7):
        super().__init__(threshold=threshold)
        self.judge = judge or LLMJudge()

    def evaluate(
        self,
        retrievals: Optional[List[Retrieval]] = None,
        goal: str = "",
        final_output: str = "",
        **_: Any,
    ) -> EvalResult:
        retrievals = [r for r in (retrievals or []) if r.docs]
        if not retrievals:
            # Nothing retrieved: this layer has no opinion. A neutral pass keeps
            # non-RAG runs from being dragged down by a metric that does not
            # apply to them — the orchestrator skips us entirely in that case,
            # so this is only a guard for direct callers.
            return self._result(1.0, "no retrieval steps to evaluate", {"per_retrieval": []})

        per_retrieval: List[Dict[str, Any]] = []
        relevance_scores: List[float] = []
        ndcg_scores: List[float] = []
        utilisations: List[float] = []

        for idx, retrieval in enumerate(retrievals):
            query = (retrieval.query or goal or "").strip()
            prompt = judge_prompts.render(
                "retrieval_quality",
                query=query or "(no query captured)",
                documents=_render_docs(retrieval),
                answer=(final_output or "").strip() or "(no final answer captured)",
                max_grade=str(MAX_GRADE),
            )
            data = self.judge.judge_json(prompt)

            grades = self._grades(data, len(retrieval.docs[:_MAX_DOCS_JUDGED]))
            relevance = rank_weighted_precision(grades)
            order_score = ndcg(grades)
            used = data.get("answer_uses_documents")
            utilisation = None if used is None else (1.0 if bool(used) else 0.0)

            relevance_scores.append(relevance)
            if order_score is not None:
                ndcg_scores.append(order_score)
            if utilisation is not None:
                utilisations.append(utilisation)

            per_retrieval.append({
                "index": idx,
                "integration": retrieval.integration,
                "target": retrieval.target,
                "query": query[:300],
                "doc_count": len(retrieval.docs),
                "grades": grades,
                "relevance": round(relevance, 4),
                "ndcg": None if order_score is None else round(order_score, 4),
                "answer_uses_documents": None if utilisation is None else bool(utilisation),
                "reason": str(data.get("reason") or ""),
            })

        return self._result(*self._blend(relevance_scores, ndcg_scores, utilisations, per_retrieval))

    # ── internals ────────────────────────────────────────────────────────────

    def _grades(self, data: Dict[str, Any], expected: int) -> List[int]:
        """Coerce the judge's per-document grades into a fixed-length int list.

        A judge that returns the wrong number of grades is common enough that
        padding with 0 (rather than raising) is the right call: a missing grade
        means "no evidence this document was relevant", which is exactly 0.
        """
        raw = data.get("grades")
        grades: List[int] = []
        if isinstance(raw, list):
            for item in raw[:expected]:
                if isinstance(item, dict):
                    item = item.get("grade")
                try:
                    grades.append(max(0, min(MAX_GRADE, int(item))))
                except (TypeError, ValueError):
                    grades.append(0)
        grades += [0] * (expected - len(grades))
        return grades

    def _blend(
        self,
        relevance: List[float],
        order: List[float],
        utilisation: List[float],
        per_retrieval: List[Dict[str, Any]],
    ) -> tuple:
        mean_rel = sum(relevance) / len(relevance) if relevance else 0.0
        mean_ndcg = sum(order) / len(order) if order else None
        mean_util = sum(utilisation) / len(utilisation) if utilisation else None

        # Renormalize over the terms we could actually measure, so a run whose
        # ranking is ungradeable (single-doc retrievals) is not implicitly
        # scored as if ranking were zero.
        weighted = W_RELEVANCE * mean_rel
        total_w = W_RELEVANCE
        if mean_ndcg is not None:
            weighted += W_NDCG * mean_ndcg
            total_w += W_NDCG
        if mean_util is not None:
            weighted += W_UTILISATION * mean_util
            total_w += W_UTILISATION
        score = _clamp_unit(weighted / total_w)

        bits = [f"relevance {mean_rel:.2f}"]
        bits.append("ranking n/a" if mean_ndcg is None else f"nDCG {mean_ndcg:.2f}")
        if mean_util is not None:
            bits.append("answer grounded in results" if mean_util >= 0.5
                        else "answer ignored the retrieved documents")
        reason = f"{len(per_retrieval)} retrieval(s): " + ", ".join(bits)

        return score, reason, {
            "per_retrieval": per_retrieval,
            "relevance": round(mean_rel, 4),
            "ndcg": None if mean_ndcg is None else round(mean_ndcg, 4),
            "utilisation": None if mean_util is None else round(mean_util, 4),
            "weights": {
                "relevance": W_RELEVANCE,
                "ndcg": W_NDCG if mean_ndcg is not None else None,
                "utilisation": W_UTILISATION if mean_util is not None else None,
            },
        }
