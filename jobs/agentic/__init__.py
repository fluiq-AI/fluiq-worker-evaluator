"""Agentic evaluation layer.

Turns any agent trace (Fluiq SDK envelope, OpenInference/OTel spans, or raw
JSON) into a normalized :class:`AgentRun` and scores it in layers:

* Layer 0 — normalization (:mod:`jobs.agentic.adapters`)
* Layer 1 — deterministic checks, no LLM (:mod:`jobs.agentic.deterministic`)
* Layer 2 — span-level LLM-judge evals (:mod:`jobs.agentic.tool_selection`)

The :mod:`jobs.agentic.orchestrator` composes the layers for one run. This is
the Phase 0→2 vertical slice; trajectory (Layer 3) and the multi-agent panel
(Layer 4) plug into the orchestrator later.
"""
from jobs.agentic.schema import AgentRun, AgentStep, ToolCall, ToolSpec

__all__ = ["AgentRun", "AgentStep", "ToolCall", "ToolSpec"]
