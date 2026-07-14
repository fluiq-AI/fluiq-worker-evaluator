"""Layer 1 — deterministic checks. No LLM, no cost, run first.

Catches the large class of agent failures that don't need a judge: calling a
tool that isn't allowed, arguments that don't match the tool's JSON schema,
tools that errored, and redundant/looping calls. Findings feed the orchestrator
and are attached to the tool-selection details so a judge call is never spent on
something a schema check already settled.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from jobs.agentic.graph import AgentGraph, build_graph
from jobs.agentic.schema import AgentRun, ToolCall, ToolSpec

# JSON-Schema type name → Python types. ``bool`` is excluded from the numeric
# types on purpose (``True`` is an ``int`` in Python but not a valid number arg).
_TYPE_MAP: Dict[str, tuple] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
    "null": (type(None),),
}

_SEVERITY_WEIGHT = {"error": 1.0, "warning": 0.4}


class Finding(BaseModel):
    call_index: int
    tool: str
    code: str          # unknown_tool | missing_required_arg | arg_type_mismatch | ...
    severity: str      # "error" | "warning"
    message: str
    detail: Dict[str, Any] = Field(default_factory=dict)


class DeterministicReport(BaseModel):
    findings: List[Finding] = Field(default_factory=list)
    total_calls: int = 0
    error_calls: int = 0     # calls with >=1 error-severity finding
    # 1.0 = every call clean; drops with each error/warning. Used as a cheap
    # gate and blended into the run verdict.
    score: float = 1.0

    @property
    def passed(self) -> bool:
        return self.error_calls == 0


def _matches_type(value: Any, json_type: Any) -> bool:
    types = json_type if isinstance(json_type, list) else [json_type]
    for t in types:
        py = _TYPE_MAP.get(t)
        if not py:
            return True  # unknown/None schema type → don't fail on it
        if t in ("integer", "number") and isinstance(value, bool):
            continue     # bool is not a valid number
        if isinstance(value, py):
            return True
    return False


def _validate_args(call: ToolCall, spec: Optional[ToolSpec], idx: int) -> List[Finding]:
    """Validate a call's arguments against its tool's JSON schema (best-effort)."""
    findings: List[Finding] = []

    if spec is None:
        # Unknown tool only counts when an allowlist actually exists (checked by
        # the caller); here we simply can't validate args, so skip.
        return findings

    schema = spec.parameters or {}
    props = schema.get("properties") if isinstance(schema, dict) else None
    if not isinstance(props, dict):
        return findings  # no declared schema → nothing to validate against

    required = schema.get("required") or []
    args = call.arguments or {}

    for key in required:
        if key not in args:
            findings.append(Finding(
                call_index=idx, tool=call.name, code="missing_required_arg",
                severity="error",
                message=f"required argument '{key}' is missing",
                detail={"argument": key},
            ))

    for key, value in args.items():
        if key.startswith("_"):  # synthetic keys from unparseable args
            continue
        if key not in props:
            if schema.get("additionalProperties") is False:
                findings.append(Finding(
                    call_index=idx, tool=call.name, code="unexpected_arg",
                    severity="error",
                    message=f"argument '{key}' is not in the tool schema",
                    detail={"argument": key},
                ))
            else:
                findings.append(Finding(
                    call_index=idx, tool=call.name, code="unknown_arg",
                    severity="warning",
                    message=f"argument '{key}' is not declared by the tool",
                    detail={"argument": key},
                ))
            continue
        expected = props[key].get("type") if isinstance(props[key], dict) else None
        if expected and not _matches_type(value, expected):
            findings.append(Finding(
                call_index=idx, tool=call.name, code="arg_type_mismatch",
                severity="error",
                message=f"argument '{key}' should be {expected}, got {type(value).__name__}",
                detail={"argument": key, "expected": expected, "got": type(value).__name__},
            ))
        enum = props[key].get("enum") if isinstance(props[key], dict) else None
        if isinstance(enum, list) and value not in enum:
            findings.append(Finding(
                call_index=idx, tool=call.name, code="arg_not_in_enum",
                severity="error",
                message=f"argument '{key}'={value!r} is not one of {enum}",
                detail={"argument": key, "enum": enum, "got": value},
            ))
    return findings


def _call_signature(call: ToolCall) -> str:
    return call.name + "::" + json.dumps(call.arguments, sort_keys=True, default=str)


def check(run: AgentRun, graph: Optional[AgentGraph] = None) -> DeterministicReport:
    """Run all Layer-1 checks over a normalized run.

    ``graph`` makes loop detection DAG-aware: an identical repeated call is only
    a loop when both calls lie on the *same path* (one is an ancestor of the
    other). Identical calls in sibling parallel branches are legitimate fan-out
    and are not flagged. Pass a prebuilt graph to avoid rebuilding it.
    """
    if graph is None:
        graph = build_graph(run)

    report = DeterministicReport(total_calls=len(run.tool_calls))
    has_allowlist = bool(run.available_tools)
    # signature -> list of (call_index, step_id) already seen
    seen_signatures: Dict[str, list[tuple[int, str]]] = {}
    calls_with_error: set[int] = set()

    idx = 0
    for step in run.steps:
        step_id = graph.id_for(step)
        for call in step.tool_calls:
            spec = run.tool_spec(call.name)

            if has_allowlist and spec is None:
                report.findings.append(Finding(
                    call_index=idx, tool=call.name, code="unknown_tool",
                    severity="error",
                    message=f"tool '{call.name}' is not in the declared toolset",
                    detail={"available": [s.name for s in run.available_tools]},
                ))
                calls_with_error.add(idx)

            for f in _validate_args(call, spec, idx):
                report.findings.append(f)
                if f.severity == "error":
                    calls_with_error.add(idx)

            if call.error:
                report.findings.append(Finding(
                    call_index=idx, tool=call.name, code="tool_execution_error",
                    severity="error",
                    message=f"tool call returned an error: {str(call.error)[:200]}",
                    detail={"error": call.error},
                ))
                calls_with_error.add(idx)

            sig = _call_signature(call)
            prior = seen_signatures.get(sig)
            if prior:
                # Loop only if a prior identical call shares this call's path.
                same_path = next(
                    ((pidx, pstep) for (pidx, pstep) in prior
                     if graph.same_lineage(step_id, pstep)),
                    None,
                )
                if same_path is not None:
                    report.findings.append(Finding(
                        call_index=idx, tool=call.name, code="duplicate_call",
                        severity="warning",
                        message="identical tool call repeated on the same path (possible loop)",
                        detail={"first_index": same_path[0]},
                    ))
                # else: identical call in a parallel branch — legitimate, skip.
                prior.append((idx, step_id))
            else:
                seen_signatures[sig] = [(idx, step_id)]

            idx += 1

    report.error_calls = len(calls_with_error)

    # Score: start at 1.0, subtract normalized penalty. A run with any error
    # finding can't score above ~0.6; warnings nudge it down more gently.
    if report.findings:
        penalty = sum(_SEVERITY_WEIGHT.get(f.severity, 0.5) for f in report.findings)
        denom = max(report.total_calls, 1) + penalty
        report.score = max(0.0, 1.0 - penalty / denom)
    return report
