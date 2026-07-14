"""Layer 2 — Tool Selection Quality (LLM-as-judge).

Given the user's goal, the tools the agent *could* call, and the calls it
*actually* made, judge whether each call was the right tool with sensible
arguments — the agentic analogue of RAGAS answer-relevancy. Deterministic
findings from Layer 1 are handed to the judge as context so it doesn't re-derive
schema problems and can focus on *appropriateness*.

Reuses :class:`BaseEvaluator` (score/passed/threshold) and the admin-editable
prompt registry, exactly like the existing hallucination/RAGAS metrics.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from jobs.agentic.deterministic import DeterministicReport
from jobs.agentic.schema import AgentRun, ToolCall, ToolSpec
from jobs.helper import judge_prompts
from jobs.helper.base import BaseEvaluator, EvalResult, _clamp_unit
from jobs.helper.judge import LLMJudge


def _render_tools(tools: List[ToolSpec]) -> str:
    if not tools:
        return "(the trace did not declare an explicit toolset)"
    lines = []
    for t in tools:
        desc = (t.description or "").strip().replace("\n", " ")
        params = list((t.parameters or {}).get("properties", {}).keys())
        tag = f" [mcp:{t.server}]" if t.kind == "mcp" else ""
        lines.append(f"- {t.name}{tag}: {desc} (args: {', '.join(params) or 'none'})")
    return "\n".join(lines)


def _render_calls(calls: List[ToolCall]) -> str:
    lines = []
    for i, c in enumerate(calls):
        tag = f" [mcp:{c.server}]" if c.kind == "mcp" else ""
        lines.append(f"{i}. {c.name}{tag} args={json.dumps(c.arguments, default=str)[:400]}")
    return "\n".join(lines) or "(no tool calls were made)"


def _render_flags(report: Optional[DeterministicReport]) -> str:
    if not report or not report.findings:
        return "(no deterministic issues found)"
    return "\n".join(
        f"- call {f.call_index} ({f.tool}): {f.code} — {f.message}"
        for f in report.findings
    )


class ToolSelectionQuality(BaseEvaluator):
    """Score whether the agent chose the right tools with sound arguments."""

    name = "agentic.tool_selection_quality"

    def __init__(self, judge: Optional[LLMJudge] = None, threshold: float = 0.7):
        super().__init__(threshold=threshold)
        self.judge = judge or LLMJudge()

    def evaluate(
        self,
        goal: str,
        tool_calls: List[ToolCall],
        available_tools: Optional[List[ToolSpec]] = None,
        deterministic: Optional[DeterministicReport] = None,
        **_: Any,
    ) -> EvalResult:
        if not tool_calls:
            # No tools used. Whether that's right depends on the task, but with
            # nothing to judge we return a neutral pass rather than a false fail.
            return self._result(1.0, "no tool calls to evaluate", {"per_call": []})

        prompt = judge_prompts.render(
            "tool_selection_quality",
            goal=goal or "(no explicit goal captured)",
            tools=_render_tools(available_tools or []),
            calls=_render_calls(tool_calls),
            flags=_render_flags(deterministic),
        )
        data = self.judge.judge_json(prompt)

        per_call_raw = data.get("per_call")
        per_call: List[Dict[str, Any]] = []
        if isinstance(per_call_raw, list):
            for i, call in enumerate(tool_calls):
                entry = per_call_raw[i] if i < len(per_call_raw) and isinstance(per_call_raw[i], dict) else {}
                per_call.append({
                    "index": i,
                    "tool": call.name,
                    "appropriate": bool(entry.get("appropriate", True)),
                    "reason": str(entry.get("reason") or ""),
                })

        # Prefer an explicit overall score; else derive from per-call verdicts.
        if data.get("score") is not None:
            score = _clamp_unit(data.get("score"))
        elif per_call:
            score = _clamp_unit(
                sum(1 for c in per_call if c["appropriate"]) / len(per_call)
            )
        else:
            score = 0.5

        # Blend in the deterministic score so schema-invalid calls can't be
        # rated "perfect" by a lenient judge. Judge is weighted higher.
        if deterministic is not None and deterministic.total_calls:
            score = _clamp_unit(0.7 * score + 0.3 * deterministic.score)

        return self._result(
            score,
            str(data.get("reason") or f"{sum(c['appropriate'] for c in per_call)}/{len(per_call) or len(tool_calls)} calls appropriate"),
            {
                "per_call": per_call,
                "deterministic_score": deterministic.score if deterministic else None,
            },
        )
