"""Harvest pipeline: raw production traces → draft golden cases → promote.

Offline (injected events, no ClickHouse, no judge).

Run:  ../.workers-venv/Scripts/python.exe tests/test_harvest.py
"""
import json
import os
import sys
import tempfile

os.environ.setdefault("EVAL_JUDGE_THRESHOLD", "0.7")
os.environ.setdefault("EVAL_JUDGE_CACHE_TTL", "300")
os.environ.setdefault("EVAL_JUDGE_CACHE_MAX", "1000")
os.environ.setdefault("CLICKHOUSE_PORT", "8123")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jobs.calibration.golden import GoldenCase, load_golden
from jobs.calibration.harvest import harvest, group_by_run, promote_labeled, write_candidates


def _tc(name, args):
    return {"id": name + "1", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


# A plain LLM answer (no tools) — should yield single-shot drafts.
LLM_EVENTS = [
    {"type": "llm", "trace_id": "q1", "root_trace_id": "q1",
     "messages": [{"role": "user", "content": "What is the capital of France?"}],
     "response": "The capital of France is Paris."},
]

# A tool-using multi-agent run with a join — should yield agentic drafts.
AGENT_EVENTS = [
    {"integration": "LANGGRAPH", "type": "llm", "trace_id": "r", "root_trace_id": "r",
     "messages": [{"role": "user", "content": "Research and combine"}], "langgraph": {"langgraph_node": "router"},
     "tools": [{"name": "search", "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}}}]},
    {"type": "llm", "trace_id": "a", "parent_id": "r", "root_trace_id": "r", "langgraph": {"langgraph_node": "rx"},
     "response": "alpha", "tool_uses": [{"type": "tool_use", "id": "u1", "name": "search", "input": {"q": "X"}}]},
    {"type": "llm", "trace_id": "b", "parent_id": "r", "root_trace_id": "r", "langgraph": {"langgraph_node": "ry"},
     "response": "beta"},
    {"type": "llm", "trace_id": "j", "parent_ids": ["a", "b"], "root_trace_id": "r",
     "langgraph": {"langgraph_node": "synth"}, "response": "alpha + beta"},
]


def test_group_by_run():
    runs = group_by_run(LLM_EVENTS + AGENT_EVENTS)
    assert set(runs) == {"q1", "r"}
    assert len(runs["r"]) == 4


def test_harvest_classifies_runs():
    drafts = harvest(LLM_EVENTS + AGENT_EVENTS)
    by_metric = {}
    for d in drafts:
        by_metric.setdefault(d.metric, []).append(d)

    # single-shot from the plain LLM answer
    assert "hallucination" in by_metric and "relevance" in by_metric
    assert by_metric["hallucination"][0].inputs["answer"] == "The capital of France is Paris."

    # agentic drafts from the tool-using join run
    assert "agentic.tool_selection_quality" in by_metric
    assert "agentic.trajectory" in by_metric
    assert "agentic.coordination" in by_metric
    assert by_metric["agentic.coordination"][0].trace is not None

    # every draft is UNLABELLED
    assert all(d.expected_pass is None for d in drafts)


def test_unlabelled_drafts_are_never_loaded(tmp=None):
    # Writing drafts into a golden dir and loading it yields zero cases.
    with tempfile.TemporaryDirectory() as d:
        drafts = harvest(LLM_EVENTS)
        with open(os.path.join(d, "drafts.json"), "w", encoding="utf-8") as fh:
            json.dump([c.model_dump(mode="json") for c in drafts], fh)
        assert load_golden(d) == []          # unlabelled → skipped


def test_harvest_scrubs_pii():
    events = [
        {"type": "llm", "trace_id": "p1", "root_trace_id": "p1",
         "messages": [{"role": "user", "content": "My email is john@acme.com, help me."}],
         "response": "Sure John, I've noted john@acme.com."},
    ]
    drafts = harvest(events)                          # scrub on by default
    blob = json.dumps([d.model_dump(mode="json") for d in drafts])
    assert "john@acme.com" not in blob               # PII removed
    assert "<EMAIL_ADDRESS>" in blob
    assert any("scrubbed: EMAIL_ADDRESS" in (d.note or "") for d in drafts)

    # with scrubbing off, the raw value survives (opt-out path)
    raw = json.dumps([d.model_dump(mode="json") for d in harvest(events, scrub=False)])
    assert "john@acme.com" in raw


def test_promote_only_labelled():
    with tempfile.TemporaryDirectory() as d:
        drafts = harvest(LLM_EVENTS)          # 2 unlabelled drafts
        cand = os.path.join(d, "cand.json")   # kept outside the golden dir
        with open(cand, "w", encoding="utf-8") as fh:
            rows = [c.model_dump(mode="json") for c in drafts]
            rows[0]["expected_pass"] = True   # human labels one of them
            json.dump(rows, fh)

        golden_dir = os.path.join(d, "golden")
        os.makedirs(golden_dir)
        target = os.path.join(golden_dir, "production.json")
        stats = promote_labeled(cand, target)
        assert stats["total"] == 2 and stats["labelled"] == 1 and stats["promoted"] == 1

        loaded = load_golden(golden_dir)      # only the labelled one is usable
        assert len(loaded) == 1 and loaded[0].id.startswith("harvest-")

        # promoting again is idempotent (no duplicates)
        assert promote_labeled(cand, target)["promoted"] == 0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
