"""Run the golden set through the real judges and report agreement.

`build_report` (pure) turns per-case predictions + labels into per-metric
agreement stats; `run_calibration` drives the evaluators to produce those
predictions. Run as a module for a one-shot report:

    python -m jobs.calibration.runner            # uses the configured judge
    python -m jobs.calibration.runner --stub     # offline smoke (fixed judge)
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from jobs.calibration.golden import GoldenCase, load_golden
from jobs.calibration.metrics import agreement, score_agreement
from jobs.helper.base import BaseEvaluator
from jobs.helper.hallucination import HallucinationEvaluator
from jobs.helper.judge import LLMJudge
from jobs.helper.ragas import (
    AnswerRelevancy, Coherence, ContextPrecision, ContextRecall, Faithfulness, Toxicity,
)

logger = logging.getLogger(__name__)

# Metric name → single-shot evaluator class. These are the judge-driven metrics
# where a human label is natural; agentic metrics (tool/trajectory) calibrate
# via full traces and are out of scope for this golden format.
_EVALUATORS = {
    "hallucination":     HallucinationEvaluator,
    "relevance":         AnswerRelevancy,
    "answer_relevancy":  AnswerRelevancy,
    "faithfulness":      Faithfulness,
    "toxicity":          Toxicity,
    "coherence":         Coherence,
    "context_precision": ContextPrecision,
    "context_recall":    ContextRecall,
}


class CaseRecord(BaseModel):
    metric: str
    predicted_pass: Optional[bool] = None
    expected_pass: Optional[bool] = None
    predicted_score: Optional[float] = None
    expected_score: Optional[float] = None
    error: Optional[str] = None


def build_evaluator(metric: str, judge: LLMJudge, threshold: float) -> BaseEvaluator:
    cls = _EVALUATORS.get((metric or "").lower())
    if cls is None:
        raise ValueError(f"No calibratable evaluator for metric {metric!r}")
    return cls(judge=judge, threshold=threshold)


# Agentic metrics score a whole trace, not a (question, answer) pair. Accept the
# bare or ``agentic.``-prefixed name.
_AGENTIC_METRICS = {
    "tool_selection_quality", "trajectory", "coordination", "deterministic",
}


def is_agentic_case(case) -> bool:
    m = (case.metric or "").lower().replace("agentic.", "")
    return case.kind == "agentic" or case.trace is not None or m in _AGENTIC_METRICS


def run_agentic_case(judge: LLMJudge, case, threshold: float):
    """Normalize the case's trace into an AgentRun and score it with the agentic
    layer named by ``metric``. Returns an EvalResult (has .passed / .score)."""
    from jobs.agentic import deterministic
    from jobs.agentic.adapters import normalize
    from jobs.agentic.coordination import MultiAgentEvaluator
    from jobs.agentic.graph import build_graph
    from jobs.agentic.tool_selection import ToolSelectionQuality
    from jobs.agentic.trajectory import TrajectoryEvaluator
    from jobs.helper.base import EvalResult

    run = normalize(case.trace or {})
    graph = build_graph(run)
    metric = (case.metric or "").lower().replace("agentic.", "")

    if metric == "tool_selection_quality":
        det = deterministic.check(run, graph)
        return ToolSelectionQuality(judge=judge, threshold=threshold).evaluate(
            goal=run.goal, tool_calls=run.tool_calls,
            available_tools=run.available_tools, deterministic=det,
        )
    if metric == "trajectory":
        return TrajectoryEvaluator(judge=judge, threshold=threshold).evaluate(
            goal=run.goal, steps=run.steps, final_output=run.final_output, graph=graph,
        )
    if metric == "coordination":
        return MultiAgentEvaluator(judge=judge, threshold=threshold).evaluate(run=run, graph=graph)
    if metric == "deterministic":
        det = deterministic.check(run, graph)
        return EvalResult(name="agentic.deterministic", score=det.score, passed=det.passed)
    raise ValueError(f"Unknown agentic metric {case.metric!r}")


def build_report(records: List[CaseRecord], min_accuracy: float = 0.7) -> Dict[str, Any]:
    """Pure: aggregate per-case records into per-metric + overall agreement."""
    by_metric: Dict[str, List[CaseRecord]] = defaultdict(list)
    for r in records:
        by_metric[r.metric].append(r)

    per_metric: Dict[str, Any] = {}
    all_pred: List[bool] = []
    all_exp: List[bool] = []

    for metric, recs in sorted(by_metric.items()):
        cls_pred = [r.predicted_pass for r in recs if r.expected_pass is not None and r.predicted_pass is not None]
        cls_exp = [r.expected_pass for r in recs if r.expected_pass is not None and r.predicted_pass is not None]
        sc_pred = [r.predicted_score for r in recs if r.expected_score is not None and r.predicted_score is not None]
        sc_exp = [r.expected_score for r in recs if r.expected_score is not None and r.predicted_score is not None]

        cls_stats = agreement(cls_pred, cls_exp)
        per_metric[metric] = {
            "cases": len(recs),
            "errors": sum(1 for r in recs if r.error),
            "classification": cls_stats,
            "score": score_agreement(sc_pred, sc_exp),
            "calibrated": cls_stats["n"] == 0 or cls_stats["accuracy"] >= min_accuracy,
        }
        all_pred.extend(bool(p) for p in cls_pred)
        all_exp.extend(bool(e) for e in cls_exp)

    overall = agreement(all_pred, all_exp)
    overall_ok = overall["n"] == 0 or overall["accuracy"] >= min_accuracy
    all_calibrated = all(m["calibrated"] for m in per_metric.values())
    return {
        "min_accuracy": min_accuracy,
        "overall": overall,
        "per_metric": per_metric,
        # A release gate: overall accuracy clears the bar AND every metric is
        # individually calibrated (so one weak metric can't hide behind others).
        "passed": overall_ok and all_calibrated,
    }


def run_calibration(
    judge: LLMJudge,
    cases: List[GoldenCase],
    threshold: float = 0.7,
    min_accuracy: float = 0.7,
) -> Dict[str, Any]:
    records: List[CaseRecord] = []
    for c in cases:
        rec = CaseRecord(metric=c.metric, expected_pass=c.expected_pass, expected_score=c.expected_score)
        try:
            if is_agentic_case(c):
                result = run_agentic_case(judge, c, threshold)
            else:
                result = build_evaluator(c.metric, judge, threshold).evaluate(**c.inputs)
            rec.predicted_pass = result.passed
            rec.predicted_score = float(result.score)
        except Exception as exc:
            rec.error = str(exc)
            logger.warning("[CALIBRATION] case %s failed: %s", c.id, exc)
        records.append(rec)
    return build_report(records, min_accuracy=min_accuracy)


def _print_report(report: Dict[str, Any]) -> None:
    o = report["overall"]
    print("=" * 64)
    print(f"CALIBRATION  overall: acc={o['accuracy']} kappa={o['kappa']} "
          f"n={o['n']}  passed={report['passed']} (min_acc={report['min_accuracy']})")
    print("-" * 64)
    for metric, m in report["per_metric"].items():
        c = m["classification"]
        s = m["score"]
        print(f"{metric:20} cases={m['cases']:2} acc={c['accuracy']} "
              f"kappa={c['kappa']} f1={c['f1']} mae={s['mae']} calibrated={m['calibrated']}")
    print("=" * 64)


def main() -> None:
    import sys
    import config

    cases = load_golden()
    if not cases:
        print("No golden cases found.")
        return

    if "--stub" in sys.argv:
        import json as _json
        judge = LLMJudge(provider="openai", judge_fn=lambda _p: _json.dumps({"score": 0.8, "reason": "stub"}))
    else:
        judge = LLMJudge(provider=config.JUDGE_PROVIDER, model=config.JUDGE_MODEL)

    report = run_calibration(judge, cases, threshold=config.JUDGE_THRESHOLD)
    _print_report(report)


if __name__ == "__main__":
    main()
