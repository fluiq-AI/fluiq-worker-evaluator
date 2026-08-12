"""Composes the agentic evaluation layers for one run.

* L1 deterministic (always, no LLM)
* L2 tool-selection judge (always)
* L3 trajectory judge (depth ``standard`` / ``deep``)
* L4 multi-agent panel wraps L2+L3 (depth ``deep`` — gated by default)

Depth is per-run so streaming can default cheap while offline/CI runs go deep.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from jobs.agentic import deterministic
from jobs.agentic.coordination import MultiAgentEvaluator, group_agents
from jobs.agentic.graph import build_graph
from jobs.agentic.panel import JudgePanel
from jobs.agentic.schema import AgentRun
from jobs.agentic.tool_selection import ToolSelectionQuality
from jobs.agentic.retrieval import RetrievalQuality
from jobs.agentic.trajectory import TrajectoryEvaluator
from jobs.helper import judge_prompts
from jobs.helper.base import EvalResult
from jobs.helper.judge import LLMJudge

# Which layer each metric belongs to (persisted on the CH ``layer`` column).
METRIC_LAYER: Dict[str, str] = {
    ToolSelectionQuality.name: "tool_selection",
    RetrievalQuality.name:     "retrieval",
    TrajectoryEvaluator.name:  "trajectory",
    MultiAgentEvaluator.name:  "coordination",
}


def evaluate_run(
    run: AgentRun,
    judge: LLMJudge,
    threshold: float = 0.7,
    *,
    depth: str = "standard",
    panel: Optional[JudgePanel] = None,
) -> Dict[str, Any]:
    """Run the agentic layers over a normalized :class:`AgentRun`.

    ``depth``: "fast" (L1+L2), "standard" (+L3), "deep" (+L4 panel on L2/L3).
    Returns per-metric :class:`EvalResult` objects plus a blended run verdict.
    """
    use_panel = depth == "deep" and panel is not None

    # Build the run's DAG once and share it: L1 uses it for branch-aware loop
    # detection, L3 for topological + fan-out/join-aware trajectory rendering.
    graph = build_graph(run)
    det_report = deterministic.check(run, graph)

    metrics: Dict[str, EvalResult] = {}

    # ── L2: tool selection ────────────────────────────────────────────────
    # Only judged when there's something to judge: either tools were actually
    # called, or tools were AVAILABLE but unused (so "should have used a tool
    # but answered from memory" is still caught as a selection issue). A run
    # with neither has a vacuous tool-selection verdict, so we skip the judge
    # call entirely and let L1/L3 carry the evaluation.
    if run.tool_calls or run.available_tools:
        def _build_tsq(j: LLMJudge) -> ToolSelectionQuality:
            return ToolSelectionQuality(judge=j, threshold=threshold)

        tsq_kwargs = dict(
            goal=run.goal,
            tool_calls=run.tool_calls,
            available_tools=run.available_tools,
            deterministic=det_report,
        )
        if use_panel:
            metrics[ToolSelectionQuality.name] = judge_prompts.captured_call(
                panel.evaluate, _build_tsq, tsq_kwargs,
            )
        else:
            metrics[ToolSelectionQuality.name] = judge_prompts.captured_call(
                _build_tsq(judge).evaluate, **tsq_kwargs,
            )

    # ── L2.5: retrieval & ranking ─────────────────────────────────────────
    # Runs at every depth, including "fast": a retriever that returned the wrong
    # documents invalidates everything downstream of it, so this is not a
    # refinement you defer to a deeper tier. Skipped entirely when the run did
    # no retrieval, which keeps non-RAG runs from paying for a judge call.
    retrievals = run.retrievals
    if retrievals:
        def _build_rq(j: LLMJudge) -> RetrievalQuality:
            return RetrievalQuality(judge=j, threshold=threshold)

        rq_kwargs = dict(
            retrievals=retrievals,
            goal=run.goal,
            final_output=run.final_output,
        )
        if use_panel:
            metrics[RetrievalQuality.name] = judge_prompts.captured_call(
                panel.evaluate, _build_rq, rq_kwargs,
            )
        else:
            metrics[RetrievalQuality.name] = judge_prompts.captured_call(
                _build_rq(judge).evaluate, **rq_kwargs,
            )

    # ── L3: trajectory (standard + deep) ──────────────────────────────────
    if depth in ("standard", "deep"):
        def _build_traj(j: LLMJudge) -> TrajectoryEvaluator:
            return TrajectoryEvaluator(judge=j, threshold=threshold)

        traj_kwargs = dict(
            goal=run.goal, steps=run.steps, final_output=run.final_output, graph=graph,
        )
        if use_panel:
            metrics[TrajectoryEvaluator.name] = judge_prompts.captured_call(
                panel.evaluate, _build_traj, traj_kwargs,
            )
        else:
            metrics[TrajectoryEvaluator.name] = judge_prompts.captured_call(
                _build_traj(judge).evaluate, **traj_kwargs,
            )

    # ── L5: multi-agent coordination (only for genuine multi-agent runs) ───
    if depth in ("standard", "deep"):
        is_multi_agent = len(group_agents(run)) >= 2 or any(
            graph.is_join(nid) for nid in graph.step_by_id
        )
        if is_multi_agent:
            def _build_coord(j: LLMJudge) -> MultiAgentEvaluator:
                return MultiAgentEvaluator(judge=j, threshold=threshold)

            coord_kwargs = dict(run=run, graph=graph)
            if use_panel:
                metrics[MultiAgentEvaluator.name] = judge_prompts.captured_call(
                    panel.evaluate, _build_coord, coord_kwargs,
                )
            else:
                metrics[MultiAgentEvaluator.name] = judge_prompts.captured_call(
                    _build_coord(judge).evaluate, **coord_kwargs,
                )

    # ── run-level verdict ─────────────────────────────────────────────────
    metric_scores = [m.score for m in metrics.values()]
    run_score = min([det_report.score, *metric_scores]) if metric_scores else det_report.score
    run_passed = det_report.passed and all(m.passed is not False for m in metrics.values())

    return {
        "run_id": run.run_id,
        "source": run.source,
        "integration": run.integration,
        "depth": depth,
        "tool_call_count": len(run.tool_calls),
        "retrieval_count": len(run.retrievals),
        "deterministic": det_report.model_dump(mode="json"),
        "metrics": metrics,
        "metric_layers": {name: METRIC_LAYER.get(name, "") for name in metrics},
        "run_score": run_score,
        "run_passed": run_passed,
    }
