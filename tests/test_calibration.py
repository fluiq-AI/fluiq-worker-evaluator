"""Calibration harness tests — pure agreement math + an offline runner smoke.

Run:  ../.workers-venv/Scripts/python.exe tests/test_calibration.py
"""
import json
import os
import sys

os.environ.setdefault("EVAL_JUDGE_THRESHOLD", "0.7")
os.environ.setdefault("EVAL_JUDGE_CACHE_TTL", "300")
os.environ.setdefault("EVAL_JUDGE_CACHE_MAX", "1000")
os.environ.setdefault("CLICKHOUSE_PORT", "8123")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jobs.calibration.golden import load_golden
from jobs.calibration.metrics import agreement, score_agreement
from jobs.calibration.runner import CaseRecord, build_report, run_calibration
from jobs.helper.judge import LLMJudge


# ── pure agreement math ──────────────────────────────────────────────────────

def test_agreement_math():
    a = agreement([True, True, False, False], [True, False, False, False])
    assert a["n"] == 4
    assert a["accuracy"] == 0.75
    assert a["kappa"] == 0.5
    assert a["precision"] == 0.5
    assert a["recall"] == 1.0
    assert abs(a["f1"] - 0.6667) < 1e-3


def test_agreement_chance_corrected():
    # A judge that always says "pass" on imbalanced labels: high accuracy, κ≈0.
    a = agreement([True] * 9 + [True], [True] * 9 + [False])
    assert a["accuracy"] == 0.9
    assert a["kappa"] == 0.0            # no skill beyond chance


def test_score_agreement_math():
    s = score_agreement([0.9, 0.1], [1.0, 0.0])
    assert s["mae"] == 0.1
    assert s["pearson"] == 1.0


# ── report aggregation ───────────────────────────────────────────────────────

def test_build_report():
    records = [
        CaseRecord(metric="hallucination", predicted_pass=True, expected_pass=True),
        CaseRecord(metric="hallucination", predicted_pass=False, expected_pass=False),
        CaseRecord(metric="hallucination", predicted_pass=True, expected_pass=False),
        CaseRecord(metric="relevance", predicted_pass=True, expected_pass=True),
    ]
    report = build_report(records, min_accuracy=0.7)
    assert abs(report["per_metric"]["hallucination"]["classification"]["accuracy"] - 0.6667) < 1e-3
    assert report["per_metric"]["hallucination"]["calibrated"] is False   # 0.67 < 0.7
    assert report["per_metric"]["relevance"]["calibrated"] is True
    assert report["overall"]["n"] == 4
    assert report["passed"] is False


# ── golden loading + offline runner ──────────────────────────────────────────

def test_golden_loads():
    cases = load_golden()
    assert len(cases) >= 8
    metrics = {c.metric for c in cases}
    assert {"hallucination", "relevance", "toxicity", "coherence"} <= metrics
    assert all(c.expected_pass is not None for c in cases)


def test_runner_offline_smoke():
    # Stub judge → deterministic; we assert the report is well-formed, not its
    # accuracy (a fixed judge has no real skill).
    judge = LLMJudge(provider="openai", judge_fn=lambda _p: json.dumps({"score": 0.8, "reason": "stub"}))
    report = run_calibration(judge, load_golden(), threshold=0.7)
    assert report["overall"]["n"] >= 8
    assert "hallucination" in report["per_metric"]
    assert isinstance(report["passed"], bool)


# ── agentic (trace-level) golden cases ───────────────────────────────────────

def test_agentic_golden_loads():
    agentic = [c for c in load_golden() if c.kind == "agentic"]
    assert len(agentic) >= 6
    metrics = {c.metric for c in agentic}
    assert {"agentic.tool_selection_quality", "agentic.trajectory", "agentic.coordination"} <= metrics
    assert all(c.trace is not None and c.expected_pass is not None for c in agentic)


def test_runner_agentic_cases():
    # Stub judge answers all three agentic layers; assert the whole trace→eval
    # pipeline runs without error and reports each agentic metric.
    def fn(prompt):
        if "BRANCHES:" in prompt:
            return json.dumps({"score": 0.8, "per_branch": [{"incorporated": True}, {"incorporated": True}]})
        return json.dumps({"score": 0.8, "reason": "stub"})
    judge = LLMJudge(provider="openai", judge_fn=fn)

    cases = [c for c in load_golden() if c.kind == "agentic"]
    report = run_calibration(judge, cases, threshold=0.7)
    pm = report["per_metric"]
    assert {"agentic.tool_selection_quality", "agentic.trajectory", "agentic.coordination"} <= set(pm)
    assert all(m["errors"] == 0 for m in pm.values())   # every trace normalized + scored
    assert report["overall"]["n"] == len(cases)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
