"""Layer 0 — ingestion adapters.

Each adapter maps a trace source onto :class:`AgentRun`:

* :func:`from_fluiq`         — Fluiq SDK ``LogTrace`` envelope(s). Handles the
  OpenAI (``tool_calls``), Anthropic (``tool_uses``), Gemini
  (``function_calls``) and MCP (``mcp_calls``) shapes, plus the declared
  ``tools`` list used as the allowlist / arg-schema source.
* :func:`from_openinference` — OTel / OpenInference GenAI spans (BYO
  observability: Phoenix, Arize, LangSmith, Langfuse).
* :func:`from_raw`           — a caller-supplied, already-close-to-normal dict.

:func:`normalize` picks the adapter from the message ``source`` / shape so the
worker never has to care where a trace came from.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from jobs.agentic.schema import AgentRun, AgentStep, ToolCall, ToolSpec

logger = logging.getLogger(__name__)


# ── small helpers ────────────────────────────────────────────────────────────

def _as_dict(value: Any) -> Dict[str, Any]:
    """Coerce arguments that may arrive as a JSON string (OpenAI/MCP) or dict."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {"_value": parsed}
        except json.JSONDecodeError:
            return {"_raw": value}
    return {}


def _agent_key(event: Dict[str, Any]) -> Optional[str]:
    """The logical agent a step belongs to: the LangGraph node, an explicit
    agent/role field (A2A / CrewAI), else the function name. Agent-less steps
    fall back to a single 'main' bucket in the coordination evaluator."""
    lg = event.get("langgraph")
    if isinstance(lg, dict) and lg.get("langgraph_node"):
        return str(lg["langgraph_node"])
    for k in ("agent", "agent_name", "crew_agent", "role", "function"):
        v = event.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None


def _step_output(event: Dict[str, Any]) -> Optional[str]:
    resp = event.get("response") or event.get("output")
    if isinstance(resp, str) and resp.strip():
        return resp[:1000]
    return None


def _parent_ids(event: Dict[str, Any]) -> List[str]:
    """Explicit multi-parent list for DAG join nodes, when the SDK emits one.
    Falls back to empty (the single ``parent_id`` is used by AgentStep.parents)."""
    pids = event.get("parent_ids")
    if isinstance(pids, list):
        return [str(p) for p in pids if p]
    return []


def _latest_user_message(event: Dict[str, Any]) -> str:
    """Best-effort extraction of the user objective from an LLM event."""
    messages = event.get("messages") or event.get("contents") or event.get("input")
    if isinstance(messages, str):
        return messages
    if isinstance(messages, list):
        for msg in reversed(messages):
            if not isinstance(msg, dict):
                continue
            if msg.get("role") in ("user", "human"):
                content = msg.get("content")
                if isinstance(content, str) and content.strip():
                    return content
                if isinstance(content, list):  # anthropic block form
                    parts = [
                        b.get("text", "") for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    ]
                    if any(parts):
                        return "\n".join(p for p in parts if p)
    return ""


def _goal_from_events(events: List[Dict[str, Any]]) -> str:
    """The run-level objective, structure-aware.

    The old heuristic took the first user message found in ingestion order —
    which, on a multi-agent run, is the first *sub-agent's* task prompt (events
    arrive in completion order), so the trajectory judge scored the whole run
    against one narrow subtask and reliably failed it. Resolution order:

    1. The ROOT event's own input / user message — the outermost span carries
       the run's true objective (a LangGraph ``invoke`` state, a ``@trace``
       root's args, an ADK user turn).
    2. Multi-task runs whose root has no input of its own (CrewAI: ``kickoff()``
       returns output only): compose the objective from the task descriptions
       in execution order, so the judge evaluates the trajectory against the
       actual plan — including its final step.
    3. Fallback: the first user message found anywhere (single-agent runs,
       where it genuinely is the goal).
    """
    # (1) Root event's own input. Root = no parent, or the self-parented crew
    # span (root_trace_id is a message-level field, not in the event JSON).
    for event in events:
        pid, tid = event.get("parent_id"), event.get("trace_id")
        if pid and pid != tid:
            continue
        text = _latest_user_message(event)
        if text.strip():
            return text
        raw = event.get("input")
        if isinstance(raw, str) and raw.strip():
            return raw
        if isinstance(raw, (dict, list)) and raw:
            # A chat-shaped list with no user turn isn't a goal statement —
            # let the later fallbacks handle it instead of dumping messages.
            if isinstance(raw, list) and any(
                isinstance(m, dict) and "role" in m for m in raw
            ):
                continue
            try:
                return json.dumps(raw, default=str)[:2000]
            except Exception:
                pass

    # (2) Compose from the task plan.
    tasks = [
        str(e.get("function") or "").strip()
        for e in events
        if e.get("type") == "task" and str(e.get("function") or "").strip()
    ]
    if len(tasks) >= 2:
        steps = "; ".join(f"({i + 1}) {t[:160]}" for i, t in enumerate(tasks))
        return f"Multi-step objective, in order: {steps}"

    # (3) First user message anywhere.
    for event in events:
        text = _latest_user_message(event)
        if text.strip():
            return text
    return ""


# ── tool-spec (allowlist / schema) normalization ─────────────────────────────

def _normalize_tool_specs(tools: Any) -> List[ToolSpec]:
    specs: List[ToolSpec] = []
    if not isinstance(tools, list):
        return specs
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        ttype = tool.get("type")

        # Gemini: a tool wraps many function declarations.
        if "function_declarations" in tool:
            for decl in tool.get("function_declarations") or []:
                if isinstance(decl, dict) and decl.get("name"):
                    specs.append(ToolSpec(
                        name=decl["name"],
                        description=decl.get("description"),
                        parameters=_as_dict(decl.get("parameters")),
                    ))
            continue

        # OpenAI chat/responses: {"type":"function","function":{...}} or flat.
        if ttype == "function":
            fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
            if fn.get("name"):
                specs.append(ToolSpec(
                    name=fn["name"],
                    description=fn.get("description"),
                    parameters=_as_dict(fn.get("parameters")),
                ))
            continue

        # MCP server descriptor — tools may be nested or discovered separately.
        if ttype == "mcp":
            label = tool.get("server_label") or tool.get("name")
            for mt in tool.get("tools") or tool.get("allowed_tools") or []:
                if isinstance(mt, dict) and mt.get("name"):
                    specs.append(ToolSpec(
                        name=mt["name"],
                        description=mt.get("description"),
                        parameters=_as_dict(mt.get("input_schema") or mt.get("parameters")),
                        kind="mcp", server=label,
                    ))
                elif isinstance(mt, str):
                    specs.append(ToolSpec(name=mt, kind="mcp", server=label))
            continue

        # Anthropic: {"name","description","input_schema":{...}}.
        if tool.get("name"):
            specs.append(ToolSpec(
                name=tool["name"],
                description=tool.get("description"),
                parameters=_as_dict(tool.get("input_schema") or tool.get("parameters")),
            ))
    return specs


# ── tool-call normalization (per provider) ───────────────────────────────────

def _calls_from_openai(tool_calls: Any) -> List[ToolCall]:
    out: List[ToolCall] = []
    for tc in tool_calls or []:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        name = fn.get("name") or tc.get("name")
        if not name:
            continue
        out.append(ToolCall(
            id=tc.get("id"), name=name,
            arguments=_as_dict(fn.get("arguments") if "arguments" in fn else tc.get("arguments")),
            raw=tc,
        ))
    return out


def _calls_from_anthropic(tool_uses: Any) -> List[ToolCall]:
    out: List[ToolCall] = []
    for tu in tool_uses or []:
        if isinstance(tu, dict) and tu.get("name"):
            out.append(ToolCall(
                id=tu.get("id"), name=tu["name"],
                arguments=_as_dict(tu.get("input")), raw=tu,
            ))
    return out


def _calls_from_gemini(function_calls: Any) -> List[ToolCall]:
    out: List[ToolCall] = []
    for fc in function_calls or []:
        if isinstance(fc, dict) and fc.get("name"):
            out.append(ToolCall(
                id=fc.get("id"), name=fc["name"],
                arguments=_as_dict(fc.get("args")), raw=fc,
            ))
    return out


def _calls_from_mcp(mcp_calls: Any) -> List[ToolCall]:
    out: List[ToolCall] = []
    for item in mcp_calls or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") not in (None, "mcp_call"):  # skip list_tools/approval
            continue
        name = item.get("name")
        if not name:
            continue
        out.append(ToolCall(
            id=item.get("id"), name=name,
            arguments=_as_dict(item.get("arguments")),
            kind="mcp", server=item.get("server_label"),
            result=item.get("output"), error=item.get("error"),
            raw=item,
        ))
    return out


def _tool_calls_from_event(event: Dict[str, Any]) -> List[ToolCall]:
    calls: List[ToolCall] = []
    calls += _calls_from_openai(event.get("tool_calls"))
    calls += _calls_from_anthropic(event.get("tool_uses"))
    calls += _calls_from_gemini(event.get("function_calls"))
    calls += _calls_from_mcp(event.get("mcp_calls"))
    return calls


# ── Fluiq envelope ───────────────────────────────────────────────────────────

def from_fluiq(payload: Any) -> AgentRun:
    """Build an AgentRun from one ``LogTrace`` event or a list of events that
    share a ``root_trace_id`` (a full run)."""
    events: List[Dict[str, Any]] = payload if isinstance(payload, list) else [payload]
    events = [e for e in events if isinstance(e, dict)]

    run = AgentRun(source="fluiq")
    specs: List[ToolSpec] = []
    seen_specs: set[str] = set()

    for i, event in enumerate(events):
        run.integration = run.integration or event.get("integration") or event.get("type")
        run.run_id = run.run_id or event.get("root_trace_id") or event.get("trace_id")

        for spec in _normalize_tool_specs(event.get("tools")):
            if spec.name not in seen_specs:
                seen_specs.add(spec.name)
                specs.append(spec)

        response = event.get("response") or event.get("output")
        if isinstance(response, str) and response.strip():
            run.final_output = response

        run.steps.append(AgentStep(
            order=i,
            type=event.get("type") or "llm",
            name=event.get("function") or event.get("api"),
            agent=_agent_key(event),
            output=_step_output(event),
            trace_id=event.get("trace_id"),
            parent_id=event.get("parent_id"),
            parent_ids=_parent_ids(event),
            tool_calls=_tool_calls_from_event(event),
        ))

    run.goal = _goal_from_events(events)
    run.available_tools = specs
    return run


# ── OpenInference / OTel GenAI (BYO observability) ───────────────────────────

def from_openinference(spans: Any) -> AgentRun:
    """Minimal OpenInference/OTel-GenAI adapter.

    Reads the common attribute names emitted by Phoenix/Arize/LangSmith:
    ``tool.name`` / ``tool_call.function.name`` (+ ``.arguments``),
    ``llm.tools`` for the allowlist, and ``input.value`` for the objective.
    Deliberately lenient: unknown attributes pass through untouched.
    """
    span_list: List[Dict[str, Any]] = spans if isinstance(spans, list) else [spans]
    span_list = [s for s in span_list if isinstance(s, dict)]

    run = AgentRun(source="openinference")
    specs: List[ToolSpec] = []
    seen: set[str] = set()

    for i, span in enumerate(span_list):
        attrs = span.get("attributes") or span
        run.goal = run.goal or str(attrs.get("input.value") or attrs.get("llm.input") or "")
        out = attrs.get("output.value") or attrs.get("llm.output")
        if isinstance(out, str) and out.strip():
            run.final_output = out

        for spec in _normalize_tool_specs(attrs.get("llm.tools") or attrs.get("tools")):
            if spec.name not in seen:
                seen.add(spec.name)
                specs.append(spec)

        calls: List[ToolCall] = []
        name = attrs.get("tool_call.function.name") or attrs.get("tool.name")
        if name:
            args = attrs.get("tool_call.function.arguments") or attrs.get("tool.parameters")
            calls.append(ToolCall(id=span.get("span_id"), name=str(name), arguments=_as_dict(args)))
        # Some exporters put a list under ``llm.tool_calls``.
        calls += _calls_from_openai(attrs.get("llm.tool_calls"))

        if calls or span.get("span_kind") in ("TOOL", "LLM", None):
            # OTel span links express fan-in beyond the single parent_id.
            links = [
                l.get("span_id") for l in (span.get("links") or [])
                if isinstance(l, dict) and l.get("span_id")
            ]
            run.steps.append(AgentStep(
                order=i, type=str(span.get("span_kind") or "llm").lower(),
                name=span.get("name"), trace_id=span.get("span_id"),
                parent_id=span.get("parent_id"),
                parent_ids=[str(x) for x in links],
                tool_calls=calls,
            ))

    run.available_tools = specs
    return run


# ── raw / already-normalized ─────────────────────────────────────────────────

def from_raw(payload: Dict[str, Any]) -> AgentRun:
    """Accept a dict that is already close to :class:`AgentRun` (offline/CI)."""
    return AgentRun.model_validate(payload)


# ── dispatch ─────────────────────────────────────────────────────────────────

def normalize(message: Dict[str, Any]) -> AgentRun:
    """Route a Kafka message to the right adapter.

    Priority: explicit ``source`` → shape sniffing. Falls back to the Fluiq
    envelope (single ``event`` or ``events`` list).
    """
    source = (message.get("source") or "").lower()

    if source in ("openinference", "otel", "phoenix", "arize", "langsmith"):
        return from_openinference(message.get("spans") or message.get("event") or [])
    if source == "raw":
        return from_raw(message.get("agent_run") or message.get("event") or {})
    if message.get("agent_run"):
        return from_raw(message["agent_run"])
    if message.get("spans"):
        return from_openinference(message["spans"])

    # Default: Fluiq envelope. ``events`` (full run) beats single ``event``.
    payload = message.get("events") or message.get("event") or message
    return from_fluiq(payload)
