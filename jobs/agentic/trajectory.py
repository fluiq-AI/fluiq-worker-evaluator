"""Layer 3 — Trajectory / run-level evaluation.

Steps back from individual tool calls and judges the *whole run*: did the agent
accomplish the user's goal, and did it get there efficiently and coherently?
This is the agentic analogue of an end-to-end correctness metric — the planner
decomposes the goal into sub-goals and checks which the trajectory satisfied
(action advancement/completion), while also flagging redundant or looping work.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from jobs.agentic.graph import AgentGraph
from jobs.agentic.schema import AgentRun, AgentStep
from jobs.helper import judge_prompts
from jobs.helper.base import BaseEvaluator, EvalResult, _clamp_unit, judge_score
from jobs.helper.judge import LLMJudge


def _render_step(step: AgentStep, prefix: str = "") -> List[str]:
    lines: List[str] = []
    if step.tool_calls:
        for c in step.tool_calls:
            tag = f"[mcp:{c.server}]" if c.kind == "mcp" else "[tool]"
            args = json.dumps(c.arguments, default=str)[:200]
            outcome = ""
            if c.error:
                outcome = f" -> ERROR {str(c.error)[:80]}"
            elif c.result is not None:
                outcome = f" -> {str(c.result)[:120]}"
            lines.append(f"{prefix}{step.order}. {tag} {c.name}({args}){outcome}")
    else:
        lines.append(f"{prefix}{step.order}. [{step.type}] {step.name or ''}".rstrip())
    return lines


def _render_trajectory(steps: List[AgentStep], graph: Optional[AgentGraph] = None) -> str:
    """Render steps for the judge. With a graph, order topologically and annotate
    fan-out / join (multi-parent) structure so the judge can reason about
    parallel branches and whether a join combined them; without one, fall back to
    a flat arrival-ordered list."""
    if graph is None:
        lines: List[str] = []
        for step in steps:
            lines.extend(_render_step(step))
        return "\n".join(lines) or "(no steps recorded)"

    lines = []
    for nid in graph.order:
        step = graph.step_by_id[nid]
        indent = "  " * min(graph.depth(nid), 6)
        marker = ""
        if graph.is_join(nid):
            parents = ", ".join(sorted(graph.parents.get(nid, ()))) or "?"
            marker = f"[JOIN <- {parents}] "
        elif graph.is_fan_out(nid):
            marker = "[FAN-OUT] "
        rendered = _render_step(step, prefix=indent)
        if marker and rendered:
            rendered[0] = indent + marker + rendered[0][len(indent):]
        lines.extend(rendered)
    return "\n".join(lines) or "(no steps recorded)"


class TrajectoryEvaluator(BaseEvaluator):
    """Score whether the run achieved the goal, efficiently and coherently."""

    name = "agentic.trajectory"

    def __init__(self, judge: Optional[LLMJudge] = None, threshold: float = 0.7):
        super().__init__(threshold=threshold)
        self.judge = judge or LLMJudge()

    def evaluate(
        self,
        goal: str,
        steps: List[AgentStep],
        final_output: str = "",
        graph: Optional[AgentGraph] = None,
        **_: Any,
    ) -> EvalResult:
        if not steps:
            return self._result(0.0, "no steps to evaluate", {"subgoals": []})

        prompt = judge_prompts.render(
            "trajectory_quality",
            goal=goal or "(no explicit goal captured)",
            trajectory=_render_trajectory(steps, graph),
            final_output=(final_output or "(no final output captured)")[:2000],
        )
        data = self.judge.judge_json(prompt)

        completion = _clamp_unit(data.get("goal_completion"), default=None) if data.get("goal_completion") is not None else None
        efficiency = (
            judge_score(data, key="efficiency_rating")
            if data.get("efficiency_rating") is not None
            else (_clamp_unit(data.get("efficiency")) if data.get("efficiency") is not None else None)
        )

        # Overall score: prefer explicit; else blend completion (weighted) with
        # efficiency. Completion dominates — an efficient run that fails the goal
        # is still a failure.
        if data.get("rating") is not None or data.get("score") is not None:
            score = judge_score(data)
        elif completion is not None:
            score = _clamp_unit(0.75 * completion + 0.25 * (efficiency if efficiency is not None else completion))
        else:
            score = 0.5

        subgoals = data.get("subgoals")
        subgoals = subgoals if isinstance(subgoals, list) else []

        details: Dict[str, Any] = {
            "goal_completion": completion,
            "efficiency": efficiency,
            "subgoals": subgoals,
        }
        if graph is not None:
            details["structure"] = graph.summary()

        return self._result(
            score,
            str(data.get("reason") or "trajectory evaluation completed"),
            details,
        )
