"""Offline tests for Layer 2.5 — retrieval & ranking quality.

Covers the IR maths directly (no judge needed) and the full normalize → evaluate
path with a stubbed judge. No network or API keys required. Run directly:

    ../.workers-venv/Scripts/python.exe tests/test_agentic_retrieval.py
"""
import os
import sys

# config.py reads a few vars with float()/int() and raises on import if unset.
os.environ.setdefault("EVAL_JUDGE_THRESHOLD", "0.7")
os.environ.setdefault("EVAL_JUDGE_CACHE_TTL", "300")
os.environ.setdefault("EVAL_JUDGE_CACHE_MAX", "1000")
os.environ.setdefault("CLICKHOUSE_PORT", "8123")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jobs.agentic.adapters import from_fluiq                      # noqa: E402
from jobs.agentic.retrieval import (                              # noqa: E402
    RetrievalQuality,
    dcg,
    ndcg,
    rank_weighted_precision,
)
from jobs.agentic.schema import Retrieval, RetrievedDoc           # noqa: E402


class StubJudge:
    """Returns a canned judge payload; records how many calls it received."""

    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def judge_json(self, _prompt):
        self.calls += 1
        return dict(self.payload)


def _vectorstore_event(texts, query="what is the refund window?", **over):
    """A Fluiq vector-store span in the shape emit_vector_trace produces."""
    event = {
        "type": "vectorstore",
        "integration": "pinecone",
        "trace_id": "t-1",
        "target": {"index": "kb"},
        "query": {"texts": [query], "top_k": len(texts)},
        "result": {"matches": {"items": [
            {"id": f"d{i}", "score": 0.9 - i * 0.1, "text": t}
            for i, t in enumerate(texts)
        ]}},
    }
    event.update(over)
    return event


# ── IR maths ─────────────────────────────────────────────────────────────────

def test_ndcg_rewards_correct_ordering():
    """Same documents, better order must score strictly higher."""
    best_first = ndcg([3, 2, 0])
    buried = ndcg([0, 2, 3])
    assert best_first == 1.0, best_first
    assert buried is not None and buried < best_first
    print(f"  ndcg ordering: ideal={best_first:.3f} buried={buried:.3f}")


def test_ndcg_is_none_when_ranking_cannot_be_judged():
    """Single-doc and zero-relevance cases must not fake a ranking score."""
    # One document: any order is trivially perfect, so 1.0 would inflate.
    assert ndcg([3]) is None
    # Nothing relevant: that is a recall failure, already caught by relevance.
    # Scoring ranking 0 as well would punish the same defect twice.
    assert ndcg([0, 0, 0]) is None
    print("  ndcg returns None for n<2 and for all-irrelevant results")


def test_dcg_discounts_by_position():
    assert dcg([3]) > dcg([0, 3]), "a top hit must beat the same hit at rank 2"
    print("  dcg discounts later positions")


def test_rank_weighted_precision_matches_ragas_shape():
    assert rank_weighted_precision([3, 3, 3]) == 1.0
    assert rank_weighted_precision([0, 0, 0]) == 0.0
    # Useful doc at rank 2 only: precision@2 = 1/2.
    assert abs(rank_weighted_precision([0, 2]) - 0.5) < 1e-9
    print("  rank-weighted precision behaves like ContextPrecision")


# ── Adapter ──────────────────────────────────────────────────────────────────

def test_adapter_extracts_ranked_docs():
    run = from_fluiq([_vectorstore_event(["refunds within 30 days", "office hours"])])
    assert len(run.retrievals) == 1, run.retrievals
    r = run.retrievals[0]
    assert r.query == "what is the refund window?"
    assert [d.rank for d in r.docs] == [0, 1]
    assert r.docs[0].text.startswith("refunds")
    assert r.integration == "pinecone" and r.target == "kb"
    print(f"  adapter: {len(r.docs)} ranked docs, target={r.target}")


def test_adapter_ignores_mutations_and_empty_results():
    """Upserts and empty result sets carry nothing to grade."""
    upsert = {"type": "vectorstore", "integration": "pinecone",
              "mutation": {"ids": {"count": 3}}}
    empty = _vectorstore_event([])
    run = from_fluiq([upsert, empty])
    assert run.retrievals == [], run.retrievals
    print("  adapter: mutations and empty results produce no retrieval step")


def test_non_retrieval_spans_are_untouched():
    llm = {"type": "llm", "model": "gpt-4o", "response": "hi"}
    run = from_fluiq([llm])
    assert run.retrievals == []
    assert run.steps[0].retrieval is None
    print("  adapter: llm spans unaffected")


# ── Evaluator ────────────────────────────────────────────────────────────────

def _retrieval(texts):
    return Retrieval(
        query="q",
        docs=[RetrievedDoc(rank=i, text=t) for i, t in enumerate(texts)],
    )


def test_good_retrieval_scores_high():
    judge = StubJudge({"grades": [3, 2], "answer_uses_documents": True, "reason": "ok"})
    res = RetrievalQuality(judge=judge).evaluate(
        retrievals=[_retrieval(["a", "b"])], final_output="an answer",
    )
    assert res.score > 0.9, res.model_dump()
    assert res.passed is True
    assert res.details["ndcg"] == 1.0
    print(f"  good retrieval: score={res.score:.3f}")


def test_buried_relevant_doc_is_penalised():
    """The retriever found the right doc but ranked it last."""
    good = RetrievalQuality(judge=StubJudge(
        {"grades": [3, 0, 0], "answer_uses_documents": True})).evaluate(
        retrievals=[_retrieval(["a", "b", "c"])], final_output="x")
    buried = RetrievalQuality(judge=StubJudge(
        {"grades": [0, 0, 3], "answer_uses_documents": True})).evaluate(
        retrievals=[_retrieval(["a", "b", "c"])], final_output="x")
    assert buried.score < good.score, (buried.score, good.score)
    assert buried.details["ndcg"] < good.details["ndcg"]
    print(f"  ranking penalty: top={good.score:.3f} buried={buried.score:.3f}")


def test_ignored_documents_lower_the_score():
    """Perfect retrieval that the answer ignored is still a RAG failure."""
    used = RetrievalQuality(judge=StubJudge(
        {"grades": [3, 3], "answer_uses_documents": True})).evaluate(
        retrievals=[_retrieval(["a", "b"])], final_output="x")
    ignored = RetrievalQuality(judge=StubJudge(
        {"grades": [3, 3], "answer_uses_documents": False})).evaluate(
        retrievals=[_retrieval(["a", "b"])], final_output="x")
    assert ignored.score < used.score, (ignored.score, used.score)
    print(f"  utilisation: used={used.score:.3f} ignored={ignored.score:.3f}")


def test_irrelevant_results_fail():
    res = RetrievalQuality(judge=StubJudge(
        {"grades": [0, 0], "answer_uses_documents": False})).evaluate(
        retrievals=[_retrieval(["a", "b"])], final_output="x")
    assert res.score == 0.0, res.model_dump()
    assert res.passed is False
    # Ranking is not gradeable here, so it must not appear as a 0 alongside it.
    assert res.details["ndcg"] is None
    print(f"  irrelevant results: score={res.score:.3f}, ndcg omitted")


def test_one_judge_call_per_retrieval_step():
    """Cost guard: batching, not one call per document."""
    judge = StubJudge({"grades": [3, 2, 1, 0], "answer_uses_documents": True})
    RetrievalQuality(judge=judge).evaluate(
        retrievals=[_retrieval(["a", "b", "c", "d"]), _retrieval(["e", "f"])],
        final_output="x",
    )
    assert judge.calls == 2, f"expected 2 calls for 2 retrievals, got {judge.calls}"
    print(f"  cost: {judge.calls} judge calls for 2 retrievals / 6 documents")


def test_short_grade_list_is_padded_not_crashed():
    """A judge that returns too few grades must degrade, not raise."""
    res = RetrievalQuality(judge=StubJudge({"grades": [3]})).evaluate(
        retrievals=[_retrieval(["a", "b", "c"])], final_output="x")
    assert res.details["per_retrieval"][0]["grades"] == [3, 0, 0]
    print("  malformed judge output padded with 0")


def test_no_retrievals_is_a_neutral_pass():
    res = RetrievalQuality(judge=StubJudge({})).evaluate(retrievals=[])
    assert res.score == 1.0 and res.passed is True
    print("  empty input is a neutral pass")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"Running {len(tests)} retrieval-layer tests\n")
    failed = 0
    for t in tests:
        try:
            print(f"- {t.__name__}")
            t()
        except AssertionError as e:
            failed += 1
            print(f"  FAILED: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
