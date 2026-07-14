"""Layer 5 — multi-agent coordination (join / hand-off quality).

Where the trajectory judge (L3) asks "did the run reach the goal?", this asks the
question that only exists in a *multi-agent DAG*: **at each join / fan-in node,
did the aggregating agent correctly incorporate every incoming branch?** A
synthesizer that silently drops one branch's contribution is a coordination
failure the other layers can't see.

It also produces a cheap per-agent structural breakdown (steps / tool calls /
errors), so a multi-agent run gets an at-a-glance "who did what". The judge only
fires on join nodes, so single-agent and linear runs cost nothing here.
"""
from __future__ import annotations

import json
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from jobs.agentic.graph import AgentGraph
from jobs.agentic.schema import AgentRun, AgentStep
from jobs.helper import judge_prompts
from jobs.helper.base import BaseEvaluator, EvalResult, _clamp_unit
from jobs.helper.judge import LLMJudge

_MAX_JOINS_JUDGED = 8  # cap judge calls on pathological graphs


def group_agents(run: AgentRun) -> "OrderedDict[str, List[AgentStep]]":
    """Group steps by their logical agent. Agent-less steps fall into 'main'."""
    groups: "OrderedDict[str, List[AgentStep]]" = OrderedDict()
    for s in run.steps:
        groups.setdefault(s.agent or "main", []).append(s)
    return groups


def _agent_summary(groups: "OrderedDict[str, List[AgentStep]]") -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for agent, steps in groups.items():
        calls = sum(len(s.tool_calls) for s in steps)
        errors = sum(1 for s in steps for c in s.tool_calls if c.error)
        out.append({"agent": agent, "steps": len(steps), "tool_calls": calls, "errors": errors})
    return out


def _step_content(step: AgentStep) -> str:
    """A short textual description of what a step produced, for the judge."""
    if step.output:
        return step.output[:600]
    if step.tool_calls:
        return "called: " + "; ".join(
            f"{c.name}({json.dumps(c.arguments, default=str)[:120]})" for c in step.tool_calls
        )
    return "(no output)"


class MultiAgentEvaluator(BaseEvaluator):
    """Score hand-off / coordination quality across a multi-agent run."""

    name = "agentic.coordination"

    def __init__(self, judge: Optional[LLMJudge] = None, threshold: float = 0.7):
        super().__init__(threshold=threshold)
        self.judge = judge or LLMJudge()

    def evaluate(self, run: AgentRun, graph: AgentGraph, **_: Any) -> EvalResult:
        groups = group_agents(run)
        agents = _agent_summary(groups)
        joins = [nid for nid in graph.order if graph.is_join(nid)]

        # Not a multi-agent run → nothing to coordinate; neutral pass.
        if len(groups) < 2 and not joins:
            return self._result(
                1.0, "single-agent run; no coordination points",
                {"agents": agents, "joins": [], "num_agents": len(groups), "num_joins": 0},
            )

        per_join: List[Dict[str, Any]] = []
        incorporated = 0
        branch_total = 0

        for jid in joins[:_MAX_JOINS_JUDGED]:
            join_step = graph.step_by_id[jid]
            parent_ids = sorted(graph.parents.get(jid, ()))
            branches = [
                {"agent": graph.step_by_id[p].agent or p, "content": _step_content(graph.step_by_id[p])}
                for p in parent_ids
            ]
            if not branches:
                continue

            prompt = judge_prompts.render(
                "agent_coordination",
                join_agent=join_step.agent or jid,
                join_output=_step_content(join_step),
                branches="\n".join(
                    f"- BRANCH {i} (agent {b['agent']}): {b['content']}" for i, b in enumerate(branches)
                ),
            )
            data = self.judge.judge_json(prompt)

            verdicts_raw = data.get("per_branch")
            verdicts: List[Dict[str, Any]] = []
            for i, b in enumerate(branches):
                entry = verdicts_raw[i] if isinstance(verdicts_raw, list) and i < len(verdicts_raw) and isinstance(verdicts_raw[i], dict) else {}
                inc = bool(entry.get("incorporated", True))
                verdicts.append({"agent": b["agent"], "incorporated": inc, "reason": str(entry.get("reason") or "")})
                branch_total += 1
                if inc:
                    incorporated += 1

            per_join.append({
                "join": join_step.agent or jid,
                "branches": verdicts,
                "reason": str(data.get("reason") or ""),
            })

        score = _clamp_unit(incorporated / branch_total) if branch_total else 1.0
        reason = (
            f"{len(groups)} agents, {len(joins)} join(s): "
            f"{incorporated}/{branch_total} branch contributions incorporated"
            if branch_total else f"{len(groups)} agents, no join nodes to score"
        )
        return self._result(score, reason, {
            "agents": agents,
            "joins": per_join,
            "num_agents": len(groups),
            "num_joins": len(joins),
        })
