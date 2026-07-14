"""Offline end-to-end test + demo for the agentic eval vertical slice.

Runs Layer 0 (normalization) → Layer 1 (deterministic) → Layer 2 (tool-selection
judge, stubbed) across the OpenAI / Anthropic / Gemini / MCP / OpenInference
shapes. No network or API keys required. Run directly:

    ../.workers-venv/Scripts/python.exe tests/test_agentic_slice.py
"""
import json
import os
import sys

# config.py reads a few vars with float()/int() and will raise on import if they
# are unset. Seed the minimum before importing anything that pulls in config.
os.environ.setdefault("EVAL_JUDGE_THRESHOLD", "0.7")
os.environ.setdefault("EVAL_JUDGE_CACHE_TTL", "300")
os.environ.setdefault("EVAL_JUDGE_CACHE_MAX", "1000")
os.environ.setdefault("CLICKHOUSE_PORT", "8123")

# Make the evaluator package root importable when run as a plain script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jobs.agentic import deterministic
from jobs.agentic.adapters import from_fluiq, from_openinference, normalize
from jobs.agentic.graph import build_graph
from jobs.agentic.orchestrator import evaluate_run
from jobs.agentic.panel import JudgePanel
from jobs.agentic.trajectory import _render_trajectory
from jobs.helper.judge import LLMJudge


# ── stub judges (no network) ─────────────────────────────────────────────────

def _fake_judge() -> LLMJudge:
    """Answers both prompt types: marks the first tool call inappropriate, and
    returns a partial-completion trajectory verdict. Deterministic."""
    def fn(prompt: str) -> str:
        if "BRANCHES:" in prompt:  # coordination
            return json.dumps({
                "score": 0.5,
                "per_branch": [{"incorporated": True, "reason": "used"},
                               {"incorporated": False, "reason": "dropped"}],
                "reason": "stub coordination judge",
            })
        if "TRAJECTORY:" in prompt:
            return json.dumps({
                "score": 0.5, "goal_completion": 0.5, "efficiency": 0.6,
                "subgoals": [{"subgoal": "get weather", "achieved": True},
                             {"subgoal": "get stock price", "achieved": False}],
                "reason": "stub trajectory judge",
            })
        return json.dumps({
            "score": 0.5,
            "per_call": [{"appropriate": False, "reason": "a better tool existed"}],
            "reason": "stub judge",
        })
    return LLMJudge(provider="openai", judge_fn=fn)


def _fixed_judge(score: float, provider: str = "openai", model: str = "stub") -> LLMJudge:
    """A judge that always returns a fixed score (for panel aggregation tests)."""
    def fn(_prompt: str) -> str:
        return json.dumps({"score": score, "reason": f"fixed {score}"})
    return LLMJudge(provider=provider, model=model, judge_fn=fn)


# ── sample traces ────────────────────────────────────────────────────────────

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}, "units": {"type": "string", "enum": ["c", "f"]}},
            "required": ["location"],
        },
    },
}

OPENAI_EVENT = {
    "integration": "OPENAI", "type": "OPENAI", "api": "chat.completions",
    "trace_id": "t-openai", "root_trace_id": "run-1",
    "messages": [{"role": "user", "content": "What's the weather in Paris and the AAPL price?"}],
    "tools": [WEATHER_TOOL],
    "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": json.dumps({"location": "Paris"})}},
        {"id": "c2", "type": "function", "function": {"name": "get_weather", "arguments": json.dumps({"location": 123})}},   # bad type
        {"id": "c3", "type": "function", "function": {"name": "get_stock_price", "arguments": json.dumps({"ticker": "AAPL"})}},  # unknown tool
        {"id": "c4", "type": "function", "function": {"name": "get_weather", "arguments": json.dumps({})}},  # missing required
    ],
}

ANTHROPIC_EVENT = {
    "integration": "ANTHROPIC", "type": "ANTHROPIC", "trace_id": "t-anthropic",
    "messages": [{"role": "user", "content": [{"type": "text", "text": "Weather in Tokyo?"}]}],
    "tools": [{"name": "get_weather", "description": "weather",
               "input_schema": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}],
    "tool_uses": [{"type": "tool_use", "id": "tu1", "name": "get_weather", "input": {"location": "Tokyo"}}],
}

GEMINI_EVENT = {
    "integration": "GEMINI", "type": "GEMINI", "trace_id": "t-gemini",
    "contents": [{"role": "user", "parts": [{"text": "Weather in Berlin?"}]}],
    "tools": [{"function_declarations": [
        {"name": "get_weather", "description": "weather",
         "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}]}],
    "function_calls": [{"name": "get_weather", "args": {"location": "Berlin"}}],
}

MCP_EVENT = {
    "integration": "OPENAI", "type": "OPENAI", "trace_id": "t-mcp",
    "messages": [{"role": "user", "content": "List my repos"}],
    "mcp_calls": [{"type": "mcp_call", "id": "m1", "name": "list_repos",
                   "arguments": json.dumps({"owner": "me"}), "server_label": "github"}],
}

OPENINFERENCE_SPANS = [
    {"span_id": "s1", "span_kind": "LLM", "name": "chat",
     "attributes": {"input.value": "Weather in Rome?",
                    "llm.tools": [WEATHER_TOOL]}},
    {"span_id": "s2", "span_kind": "TOOL", "name": "get_weather",
     "attributes": {"tool.name": "get_weather", "tool.parameters": {"location": "Rome"}}},
]

# A DAG run: root fans out to two parallel branches (a, b) that each call the
# SAME tool, then a join node (j) with two parents merges them; k repeats j's
# call on the same path. Tests fan-out/join + branch-aware loop detection.
DAG_TOOLS = [
    {"type": "function", "function": {"name": "search",
     "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}}},
    {"type": "function", "function": {"name": "summarize",
     "parameters": {"type": "object", "properties": {}}}},
]

def _call(name, args):
    return {"id": name + "1", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}

DAG_EVENTS = [
    {"integration": "LANGGRAPH", "type": "llm", "trace_id": "r", "root_trace_id": "r",
     "messages": [{"role": "user", "content": "Compare X and Y and summarize"}], "tools": DAG_TOOLS},
    {"type": "llm", "trace_id": "a", "parent_id": "r", "root_trace_id": "r",
     "tool_calls": [_call("search", {"q": "X"})]},
    {"type": "llm", "trace_id": "b", "parent_id": "r", "root_trace_id": "r",
     "tool_calls": [_call("search", {"q": "X"})]},                       # parallel dup of a
    {"type": "llm", "trace_id": "j", "parent_ids": ["a", "b"], "root_trace_id": "r",
     "tool_calls": [_call("summarize", {})]},                            # join (2 parents)
    {"type": "llm", "trace_id": "k", "parent_id": "j", "root_trace_id": "r",
     "tool_calls": [_call("summarize", {})]},                            # same-path dup of j
]

# A genuine multi-agent run (named LangGraph nodes = agents) with a synthesizer
# join over two researcher branches — exercises Layer 5 coordination.
def _lg(node):
    return {"langgraph": {"langgraph_node": node}}

MA_EVENTS = [
    {"integration": "LANGGRAPH", "type": "llm", "trace_id": "r", "root_trace_id": "r",
     "messages": [{"role": "user", "content": "Research X and Y, then synthesize"}], **_lg("router")},
    {"type": "llm", "trace_id": "a", "parent_id": "r", "root_trace_id": "r",
     "response": "X findings: alpha", "tool_calls": [_call("search", {"q": "X"})], **_lg("researcher_x")},
    {"type": "llm", "trace_id": "b", "parent_id": "r", "root_trace_id": "r",
     "response": "Y findings: beta", "tool_calls": [_call("search", {"q": "Y"})], **_lg("researcher_y")},
    {"type": "llm", "trace_id": "j", "parent_ids": ["a", "b"], "root_trace_id": "r",
     "response": "Synthesis of alpha and beta", **_lg("synthesizer")},
]


# ── tests ────────────────────────────────────────────────────────────────────

def test_normalization_across_providers():
    assert [c.name for c in from_fluiq(OPENAI_EVENT).tool_calls][:1] == ["get_weather"]
    assert from_fluiq(ANTHROPIC_EVENT).tool_calls[0].arguments == {"location": "Tokyo"}
    assert from_fluiq(GEMINI_EVENT).tool_calls[0].arguments == {"location": "Berlin"}
    mcp = from_fluiq(MCP_EVENT).tool_calls[0]
    assert mcp.kind == "mcp" and mcp.server == "github" and mcp.arguments == {"owner": "me"}
    oi = from_openinference(OPENINFERENCE_SPANS)
    assert oi.goal == "Weather in Rome?" and oi.tool_calls[0].name == "get_weather"
    assert any(t.name == "get_weather" for t in oi.available_tools)


def test_deterministic_catches_defects():
    run = from_fluiq(OPENAI_EVENT)
    report = deterministic.check(run)
    codes = {f.code for f in report.findings}
    assert "arg_type_mismatch" in codes      # location=123
    assert "unknown_tool" in codes           # get_stock_price
    assert "missing_required_arg" in codes    # empty args
    assert report.error_calls == 3            # calls c2, c3, c4
    assert report.passed is False
    assert report.score < 0.7


def test_clean_run_passes_deterministic():
    report = deterministic.check(from_fluiq(ANTHROPIC_EVENT))
    assert report.passed is True and report.score == 1.0


def test_orchestrator_end_to_end():
    run = from_fluiq(OPENAI_EVENT)
    outcome = evaluate_run(run, _fake_judge(), threshold=0.7, depth="fast")
    tsq = outcome["metrics"]["agentic.tool_selection_quality"]
    assert 0.0 <= tsq.score <= 1.0
    assert outcome["run_passed"] is False              # deterministic errors present
    assert outcome["deterministic"]["error_calls"] == 3
    assert tsq.details["per_call"][0]["appropriate"] is False
    assert "agentic.trajectory" not in outcome["metrics"]   # fast depth = no L3


def test_standard_depth_adds_trajectory():
    run = from_fluiq(OPENAI_EVENT)
    outcome = evaluate_run(run, _fake_judge(), threshold=0.7, depth="standard")
    assert "agentic.trajectory" in outcome["metrics"]
    traj = outcome["metrics"]["agentic.trajectory"]
    assert traj.details["goal_completion"] == 0.5
    assert outcome["metric_layers"]["agentic.trajectory"] == "trajectory"


def test_deep_panel_always_convenes():
    # Assert on the trajectory metric — it returns the judge score directly (no
    # deterministic blend), so panel aggregation math is clean.
    run = from_fluiq(ANTHROPIC_EVENT)               # a clean run
    members = [_fixed_judge(0.8, model="a"), _fixed_judge(0.6, model="b"), _fixed_judge(0.7, model="c")]
    panel = JudgePanel(members=members, mode="always", threshold=0.7)
    outcome = evaluate_run(run, members[0], threshold=0.7, depth="deep", panel=panel)
    traj = outcome["metrics"]["agentic.trajectory"]
    p = traj.details["panel"]
    assert p["convened"] is True and p["votes_total"] == 3
    assert abs(traj.score - 0.7) < 1e-6             # mean(0.8, 0.6, 0.7)
    assert 0.0 <= p["agreement"] <= 1.0


def test_deep_panel_gated_skips_when_confident():
    run = from_fluiq(ANTHROPIC_EVENT)
    # primary is far above threshold (0.95 vs 0.7, margin 0.12) → jury not needed
    members = [_fixed_judge(0.95, model="a"), _fixed_judge(0.2, model="b")]
    panel = JudgePanel(members=members, mode="gated", gate_margin=0.12, threshold=0.7)
    outcome = evaluate_run(run, members[0], threshold=0.7, depth="deep", panel=panel)
    traj = outcome["metrics"]["agentic.trajectory"]
    assert traj.details["panel"]["convened"] is False
    assert abs(traj.score - 0.95) < 1e-6            # only primary counted


def _broken_judge(model: str = "dead") -> LLMJudge:
    """A judge whose every call fails (provider outage / bad key / 429)."""
    def fn(_prompt: str) -> str:
        raise RuntimeError("judge unavailable")
    return LLMJudge(provider="openai", model=model, judge_fn=fn)


def test_panel_dead_primary_does_not_dilute():
    # Primary fails; juror scores 0.9. The dead primary must not vote — the
    # aggregate is the juror's 0.9, not mean(0.5, 0.9), the metric name
    # survives, and the audit trail says "failed" rather than "scored 0.5".
    run = from_fluiq(ANTHROPIC_EVENT)
    members = [_broken_judge("a"), _fixed_judge(0.9, model="b")]
    panel = JudgePanel(members=members, mode="always", threshold=0.7)
    outcome = evaluate_run(run, members[0], threshold=0.7, depth="deep", panel=panel)
    traj = outcome["metrics"]["agentic.trajectory"]
    assert abs(traj.score - 0.9) < 1e-6
    assert traj.passed is True
    assert traj.name == "agentic.trajectory"
    p = traj.details["panel"]
    assert p["members"][0].get("failed") is True
    assert "score" not in p["members"][0]
    assert p["votes_total"] == 1


def test_panel_total_outage_is_neutral_not_fail():
    # Every judge down = infrastructure outage, not a failing agent run:
    # neutral verdict (passed=None), and the run still passes on deterministic.
    run = from_fluiq(ANTHROPIC_EVENT)
    members = [_broken_judge("a"), _broken_judge("b")]
    panel = JudgePanel(members=members, mode="always", threshold=0.7)
    outcome = evaluate_run(run, members[0], threshold=0.7, depth="deep", panel=panel)
    traj = outcome["metrics"]["agentic.trajectory"]
    assert traj.passed is None
    assert outcome["run_passed"] is True


def test_panel_verdict_follows_aggregate_score():
    # A 1-1 split with a failing mean must FAIL — the old tie rule passed it
    # while the displayed score (0.51) said otherwise.
    run = from_fluiq(ANTHROPIC_EVENT)
    members = [_fixed_judge(0.72, model="a"), _fixed_judge(0.30, model="b")]
    panel = JudgePanel(members=members, mode="gated", gate_margin=0.12, threshold=0.7)
    outcome = evaluate_run(run, members[0], threshold=0.7, depth="deep", panel=panel)
    traj = outcome["metrics"]["agentic.trajectory"]
    assert traj.details["panel"]["convened"] is True
    assert abs(traj.score - 0.51) < 1e-6
    assert traj.passed is False


def test_goal_is_run_level_not_first_subtask():
    # A CrewAI-style run: events arrive in completion order, so the FIRST event
    # is a sub-agent's LLM call. The run goal must be the composed task plan —
    # never the first subtask's prompt (that made the trajectory judge fail
    # every multi-agent run).
    events = [
        {"trace_id": "l1", "parent_id": "a1", "type": "llm", "integration": "OPENAI",
         "messages": [{"role": "user", "content": "Current Task: describe the weather of Lisbon."}],
         "response": "Mild and sunny."},
        {"trace_id": "t1", "parent_id": "c", "type": "task",
         "function": "In one sentence, describe the weather of Lisbon.", "output": "Mild and sunny."},
        {"trace_id": "t2", "parent_id": "c", "type": "task",
         "function": "Merge the notes into one paragraph.", "output": "Lisbon is mild..."},
        {"trace_id": "t3", "parent_id": "c", "type": "task",
         "function": "Name the weakest point of the paragraph.", "output": "Lacks data."},
        # crew root: self-parented, no input of its own (kickoff() output only)
        {"trace_id": "c", "parent_id": "c", "type": "crew",
         "function": "Crew(Analyst, Synthesizer, Editor)", "output": "Lacks data."},
    ]
    run = from_fluiq(events)
    assert run.goal.startswith("Multi-step objective")
    assert "weakest point" in run.goal          # the plan includes the FINAL step
    assert not run.goal.startswith("Current Task")
    assert run.final_output == "Lacks data."


def test_goal_prefers_root_event_input():
    # A LangGraph-style run: the container root carries the invoke() state —
    # that IS the run's objective, even though child node events came first.
    events = [
        {"trace_id": "n1", "parent_id": "g", "type": "chain",
         "input": {"topic": "Lisbon"}, "output": "notes"},
        {"trace_id": "g", "parent_id": None, "type": "chain", "integration": "LANGCHAIN",
         "input": {"topic": "Lisbon, Portugal", "findings": []}, "output": "report"},
    ]
    run = from_fluiq(events)
    assert "Lisbon, Portugal" in run.goal


def test_panel_gated_convenes_for_dead_primary():
    # In gated mode a dead primary escalates to the jury — the jurors are the
    # only path to a verdict at all.
    run = from_fluiq(ANTHROPIC_EVENT)
    members = [_broken_judge("a"), _fixed_judge(0.9, model="b")]
    panel = JudgePanel(members=members, mode="gated", gate_margin=0.12, threshold=0.7)
    outcome = evaluate_run(run, members[0], threshold=0.7, depth="deep", panel=panel)
    traj = outcome["metrics"]["agentic.trajectory"]
    assert traj.details["panel"]["convened"] is True
    assert abs(traj.score - 0.9) < 1e-6


def test_dag_graph_structure():
    g = build_graph(from_fluiq(DAG_EVENTS))
    assert g.roots == ["r"]
    assert g.is_fan_out("r") is True        # r -> a, b
    assert g.is_join("j") is True           # a, b -> j
    assert g.parents["j"] == {"a", "b"}
    order = g.order
    assert order.index("r") < order.index("a") < order.index("j") < order.index("k")
    assert order.index("r") < order.index("b") < order.index("j")
    assert g.same_lineage("j", "k") is True      # k descends from j
    assert g.same_lineage("a", "b") is False     # sibling parallel branches


def test_dag_loop_detection_is_branch_aware():
    report = deterministic.check(from_fluiq(DAG_EVENTS))
    dups = [f for f in report.findings if f.code == "duplicate_call"]
    # The two parallel search({q:X}) calls are NOT a loop; only summarize
    # repeated on the j->k path is flagged.
    assert len(dups) == 1
    assert dups[0].tool == "summarize"


def test_dag_trajectory_render_marks_structure():
    run = from_fluiq(DAG_EVENTS)
    rendered = _render_trajectory(run.steps, build_graph(run))
    assert "JOIN" in rendered and "FAN-OUT" in rendered
    assert rendered.index("search") < rendered.index("summarize")  # topological


def test_dag_orchestrator_reports_structure():
    outcome = evaluate_run(from_fluiq(DAG_EVENTS), _fake_judge(), threshold=0.7, depth="standard")
    struct = outcome["metrics"]["agentic.trajectory"].details["structure"]
    assert struct["is_dag"] is True and struct["joins"] == 1 and struct["fan_outs"] == 1


def test_multiagent_coordination():
    outcome = evaluate_run(from_fluiq(MA_EVENTS), _fake_judge(), threshold=0.7, depth="standard")
    coord = outcome["metrics"]["agentic.coordination"]
    d = coord.details
    assert d["num_agents"] == 4                      # router + 2 researchers + synthesizer
    assert d["num_joins"] == 1
    assert {"researcher_x", "researcher_y", "synthesizer"} <= {a["agent"] for a in d["agents"]}
    branches = d["joins"][0]["branches"]
    assert len(branches) == 2                        # a and b feed the synthesizer
    assert abs(coord.score - 0.5) < 1e-6             # stub: 1 of 2 incorporated
    assert outcome["metric_layers"]["agentic.coordination"] == "coordination"


def test_single_agent_has_no_coordination_metric():
    outcome = evaluate_run(from_fluiq(ANTHROPIC_EVENT), _fake_judge(), threshold=0.7, depth="standard")
    assert "agentic.coordination" not in outcome["metrics"]


# A multi-step reasoning run that never calls a tool: two LLM turns, no `tools`
# declared and no tool calls. This still has a trajectory to judge.
NO_TOOL_EVENTS = [
    {"integration": "OPENAI", "type": "llm", "trace_id": "r", "root_trace_id": "r",
     "messages": [{"role": "user", "content": "Reason step by step: is 91 prime?"}],
     "response": "Let me factor it."},
    {"type": "llm", "trace_id": "s2", "parent_id": "r", "root_trace_id": "r",
     "response": "91 = 7 x 13, so it is not prime."},
]


def test_no_tool_multistep_run_still_gets_trajectory():
    """A tool-less multi-step run is evaluated on L1 + L3; tool-selection is
    skipped (vacuous) rather than short-circuiting the whole run."""
    run = from_fluiq(NO_TOOL_EVENTS)
    assert not run.tool_calls and len(run.steps) == 2 and not run.available_tools
    outcome = evaluate_run(run, _fake_judge(), threshold=0.7, depth="standard")
    # Trajectory (tool-agnostic) runs; tool-selection is absent for a no-tool,
    # no-available-tools run.
    assert "agentic.trajectory" in outcome["metrics"]
    assert "agentic.tool_selection_quality" not in outcome["metrics"]
    assert outcome["tool_call_count"] == 0


def test_no_tool_run_skips_tool_selection_but_keeps_it_when_tools_available():
    """When tools were AVAILABLE but unused, tool-selection still runs so a
    missed-tool-use is catchable."""
    events = [dict(NO_TOOL_EVENTS[0], tools=[WEATHER_TOOL]), NO_TOOL_EVENTS[1]]
    run = from_fluiq(events)
    assert not run.tool_calls and run.available_tools
    outcome = evaluate_run(run, _fake_judge(), threshold=0.7, depth="standard")
    assert "agentic.tool_selection_quality" in outcome["metrics"]


def test_routing_picks_agentic():
    from app import _looks_agentic
    assert _looks_agentic({"event": OPENAI_EVENT}) is True
    assert _looks_agentic({"event": {"type": "vectorstore", "api": "query"}}) is False
    assert normalize({"event": ANTHROPIC_EVENT}).source == "fluiq"
    assert normalize({"source": "openinference", "spans": OPENINFERENCE_SPANS}).source == "openinference"


# ── demo runner ──────────────────────────────────────────────────────────────

def _demo():
    print("=" * 70)
    print("AGENTIC EVAL SLICE - demo run (OpenAI trace, 4 tool calls)")
    print("=" * 70)
    run = from_fluiq(OPENAI_EVENT)
    print(f"\nGoal: {run.goal}")
    print(f"Available tools: {[t.name for t in run.available_tools]}")
    print(f"Tool calls: {[(c.name, c.arguments) for c in run.tool_calls]}")

    report = deterministic.check(run)
    print(f"\n-- Layer 1 (deterministic) -- score={report.score:.3f} passed={report.passed}")
    for f in report.findings:
        print(f"   [{f.severity}] call {f.call_index} {f.tool}: {f.code} - {f.message}")

    outcome = evaluate_run(run, _fake_judge(), threshold=0.7, depth="standard")
    tsq = outcome["metrics"]["agentic.tool_selection_quality"]
    print(f"\n-- Layer 2 (tool-selection judge, stub) -- score={tsq.score:.3f}")
    for pc in tsq.details["per_call"]:
        print(f"   call {pc['index']} {pc['tool']}: appropriate={pc['appropriate']} - {pc['reason']}")

    traj = outcome["metrics"]["agentic.trajectory"]
    print(f"\n-- Layer 3 (trajectory judge, stub) -- score={traj.score:.3f} "
          f"completion={traj.details['goal_completion']} efficiency={traj.details['efficiency']}")
    for sg in traj.details["subgoals"]:
        print(f"   subgoal: {sg['subgoal']} -> achieved={sg['achieved']}")

    print(f"\n-- RUN VERDICT (standard) -- run_score={outcome['run_score']:.3f} passed={outcome['run_passed']}")

    # Layer 4 — multi-agent panel (3 stub jurors, always mode)
    members = [_fixed_judge(0.8, model="haiku"), _fixed_judge(0.6, model="flash"), _fixed_judge(0.7, model="mini")]
    panel = JudgePanel(members=members, mode="always", threshold=0.7)
    deep = evaluate_run(from_fluiq(ANTHROPIC_EVENT), members[0], threshold=0.7, depth="deep", panel=panel)
    dp = deep["metrics"]["agentic.trajectory"].details["panel"]
    print(f"\n-- Layer 4 (multi-agent panel, 3 jurors) -- convened={dp['convened']} "
          f"score={deep['metrics']['agentic.trajectory'].score:.3f} "
          f"agreement={dp['agreement']} votes_pass={dp['votes_pass']}/{dp['votes_total']}")
    for m in dp["members"]:
        print(f"   {m['role']:7} {m['provider']}:{m['model']} -> {m['score']:.2f}")
    print("=" * 70)


if __name__ == "__main__":
    # Run the assertions, then the demo.
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print()
    _demo()
