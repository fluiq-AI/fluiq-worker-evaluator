"""Normalized agent-run schema — the single input contract for agentic evals.

Every ingestion adapter (Fluiq envelope, OpenInference/OTel, raw JSON) produces
an :class:`AgentRun`. Everything downstream (deterministic checks, judges, the
orchestrator) reads only this shape, so adding a new trace source never touches
evaluator logic.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ToolSpec(BaseModel):
    """A tool the agent was *allowed* to call, with its argument JSON schema."""

    model_config = ConfigDict(extra="allow")

    name: str
    description: Optional[str] = None
    # JSON Schema for the arguments (``{"type": "object", "properties": {...},
    # "required": [...]}``). Empty when the source didn't declare tools.
    parameters: Dict[str, Any] = Field(default_factory=dict)
    kind: str = "tool"  # "tool" | "mcp"
    server: Optional[str] = None  # MCP server label, when kind == "mcp"


class ToolCall(BaseModel):
    """A single tool/MCP invocation the agent actually emitted."""

    model_config = ConfigDict(extra="allow")

    id: Optional[str] = None
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    kind: str = "tool"  # "tool" | "mcp"
    server: Optional[str] = None
    # Populated only when the trace also carried the tool's result/outcome.
    result: Optional[Any] = None
    error: Optional[str] = None
    # Raw un-normalized fragment, kept for debugging / provider-specific fields.
    raw: Optional[Dict[str, Any]] = None


class RetrievedDoc(BaseModel):
    """One document returned by a retriever, in the order the retriever ranked it."""

    model_config = ConfigDict(extra="allow")

    rank: int = 0              # 0-based position as returned; this is what nDCG grades
    id: Optional[str] = None
    text: str = ""
    score: Optional[float] = None   # the store's own similarity/distance, when given


class Retrieval(BaseModel):
    """A single retrieval step: the query that went in and the ranked docs that
    came back. Ranking is preserved because *order* is half of what we grade —
    returning the right documents in the wrong order is a real, separate defect
    from returning the wrong documents."""

    model_config = ConfigDict(extra="allow")

    query: str = ""
    docs: List[RetrievedDoc] = Field(default_factory=list)
    integration: Optional[str] = None   # "pinecone" | "chromadb" | ...
    target: Optional[str] = None        # index / collection name
    top_k: Optional[int] = None


class AgentStep(BaseModel):
    """One step of a run: an LLM turn that emitted zero or more tool calls, or a
    retrieval step carrying its ranked results.

    ``parent_id`` is the single causal parent the SDK records. ``parent_ids``
    carries the *multiple* parents of a DAG join / fan-in node (e.g. a
    synthesizer agent consuming several parallel branches, or OTel span links).
    Use :meth:`parents` to read the effective parent set — it falls back to the
    single ``parent_id`` when no explicit multi-parent list is present."""

    model_config = ConfigDict(extra="allow")

    order: int = 0
    type: str = "llm"  # "llm" | "tool" | "mcp" | "retrieval"
    name: Optional[str] = None
    agent: Optional[str] = None    # logical agent this step belongs to (node/role)
    output: Optional[str] = None   # this step's response/output text, when known
    trace_id: Optional[str] = None
    parent_id: Optional[str] = None
    parent_ids: List[str] = Field(default_factory=list)
    tool_calls: List[ToolCall] = Field(default_factory=list)
    # Present only on retrieval steps (``type == "retrieval"``).
    retrieval: Optional["Retrieval"] = None

    def parents(self) -> List[str]:
        """Effective parents: the explicit multi-parent list if present, else
        the single ``parent_id`` (or empty for a root)."""
        if self.parent_ids:
            return self.parent_ids
        return [self.parent_id] if self.parent_id else []


class AgentRun(BaseModel):
    """A normalized agent execution, ready to evaluate."""

    model_config = ConfigDict(extra="allow")

    run_id: Optional[str] = None
    source: str = "fluiq"  # "fluiq" | "openinference" | "raw"
    integration: Optional[str] = None
    goal: str = ""          # user objective / latest user message
    final_output: str = ""  # agent's final answer, when known
    available_tools: List[ToolSpec] = Field(default_factory=list)
    steps: List[AgentStep] = Field(default_factory=list)

    @property
    def tool_calls(self) -> List[ToolCall]:
        """All tool/MCP calls across every step, in order."""
        calls: List[ToolCall] = []
        for step in self.steps:
            calls.extend(step.tool_calls)
        return calls

    @property
    def retrievals(self) -> List["Retrieval"]:
        """Every retrieval step's payload, in run order."""
        return [s.retrieval for s in self.steps if s.retrieval is not None]

    def tool_spec(self, name: str) -> Optional[ToolSpec]:
        for spec in self.available_tools:
            if spec.name == name:
                return spec
        return None
