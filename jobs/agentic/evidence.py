"""Render one slice of a normalized run as text a custom judge can grade.

A customer-authored judge prompt has always been handed the final answer. That
is the right evidence for "is this answer grounded?" and the wrong evidence for
"did my agent pick the right tool?" — the tool calls are not in the answer, so
such a prompt was scoring something it could not see.

Each target here produces the evidence for one failure surface, in the same
order the run took, so a prompt asking about ordering (a reranker, a retry loop)
has the ordering in front of it.
"""
from __future__ import annotations

import json
from typing import Any, List

from jobs.agentic.schema import AgentRun

#: Kept well under the judge's context while still admitting a long run. A
#: truncated tail is marked rather than silently dropped, because a judge that
#: cannot see the end of a trajectory should say so instead of scoring it.
MAX_CHARS = 12000

TARGETS = ("output", "retrieval", "tools", "trajectory", "coordination")


def _short(value: Any, limit: int = 600) -> str:
    """A value as compact text, truncated with a visible marker."""
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            text = str(value)
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + f"… (+{len(text) - limit} chars)"


def _clip(lines: List[str]) -> str:
    out = "\n".join(lines).strip()
    if len(out) <= MAX_CHARS:
        return out
    return out[:MAX_CHARS] + "\n… (evidence truncated)"


def _tools(run: AgentRun) -> str:
    lines: List[str] = []
    if run.available_tools:
        lines.append("Tools the agent was allowed to call:")
        for spec in run.available_tools:
            kind = f" [{spec.kind}" + (f" via {spec.server}" if spec.server else "") + "]"
            lines.append(f"- {spec.name}{kind}: {spec.description or '(no description)'}")
        lines.append("")

    calls = run.tool_calls
    if not calls:
        # Stated rather than left blank: "called nothing" is a finding, and an
        # empty section reads as missing data instead of as the answer.
        lines.append("The agent called no tools.")
        return _clip(lines)

    lines.append("Calls the agent actually made, in order:")
    for i, call in enumerate(calls, 1):
        where = f" (MCP server: {call.server})" if call.server else ""
        lines.append(f"{i}. {call.name}{where}")
        lines.append(f"   arguments: {_short(call.arguments)}")
        if call.error:
            lines.append(f"   error: {_short(call.error, 300)}")
        elif call.result is not None:
            lines.append(f"   result: {_short(call.result)}")
    return _clip(lines)


def _retrieval(run: AgentRun) -> str:
    retrievals = run.retrievals
    if not retrievals:
        return "The agent retrieved nothing."
    lines: List[str] = []
    for i, r in enumerate(retrievals, 1):
        where = " / ".join(x for x in (r.integration, r.target) if x)
        lines.append(f"Retrieval {i}: {r.query or '(no query recorded)'}"
                     + (f"  [{where}]" if where else ""))
        if not r.docs:
            lines.append("  (no documents returned)")
        for doc in r.docs:
            score = f"  score={doc.score}" if doc.score is not None else ""
            # Rank is printed because order is half of what a retrieval judge
            # grades — the same documents in a different order is a real defect.
            lines.append(f"  [{doc.rank}]{score} {_short(doc.text, 400)}")
        lines.append("")
    return _clip(lines)


def _trajectory(run: AgentRun) -> str:
    lines: List[str] = [f"Goal: {run.goal or '(not recorded)'}", ""]
    if not run.steps:
        lines.append("(no steps recorded)")
        return _clip(lines)
    lines.append("Steps, in order:")
    for i, step in enumerate(run.steps, 1):
        label = step.agent or step.name or f"step {i}"
        lines.append(f"{i}. {label}")
        if step.output:
            lines.append(f"   said: {_short(step.output, 400)}")
        for call in step.tool_calls:
            lines.append(f"   called {call.name}({_short(call.arguments, 200)})")
        if step.retrieval is not None:
            lines.append(
                f"   retrieved {len(step.retrieval.docs)} doc(s) for "
                f"{_short(step.retrieval.query, 160)}"
            )
    lines.append("")
    lines.append(f"Final answer: {_short(run.final_output, 800)}")
    return _clip(lines)


def _coordination(run: AgentRun) -> str:
    agents = [s.agent for s in run.steps if getattr(s, "agent", None)]
    if not agents:
        return "This run involved a single agent."
    lines = ["Hand-offs between agents, in order:"]
    previous = None
    for i, step in enumerate(run.steps, 1):
        who = getattr(step, "agent", None)
        if not who:
            continue
        if previous and who != previous:
            lines.append(f"  → hand-off: {previous} → {who}")
        lines.append(f"{i}. {who}: {_short(step.output, 400)}")
        previous = who
    return _clip(lines)


def render(run: AgentRun, target: str) -> str:
    """Evidence for ``target``. Unknown targets fall back to the final answer,
    so a target this build does not know about grades something real rather
    than grading an empty string."""
    if target == "tools":
        return _tools(run)
    if target == "retrieval":
        return _retrieval(run)
    if target == "trajectory":
        return _trajectory(run)
    if target == "coordination":
        return _coordination(run)
    return run.final_output or ""
