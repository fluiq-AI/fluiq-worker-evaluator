"""Grow the golden corpus from real production traces.

Production traces carry no human labels, so this is a *harvest → label → promote*
pipeline, not a one-shot import:

  1. **harvest** — pull real traces from ClickHouse, group them into runs, and
     turn each into a *draft* golden case (``expected_pass = null``). Single LLM
     answers become single-shot hallucination/relevance drafts; runs with tool
     calls / multiple steps / joins become agentic drafts (tool-selection /
     trajectory / coordination). Optionally attach the current judge's *suggested*
     verdict as a review hint — never as the label.
  2. **label** — a human reviews ``golden/candidates/*.json`` and fills in
     ``expected_pass`` (and redacts anything sensitive).
  3. **promote** — labelled candidates are merged into the active corpus; drafts
     without a label are skipped so they never pollute calibration.

The classification / promotion logic is pure and unit-tested; the ClickHouse pull
lives behind the ``main`` CLI.
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from jobs.calibration.golden import GoldenCase
from jobs.calibration.scrub import scrub_value
from jobs.helper.judge import LLMJudge

logger = logging.getLogger(__name__)

_CANDIDATES_DIR = os.path.join(os.path.dirname(__file__), "golden", "candidates")


def _run_id(event: Dict[str, Any]) -> Optional[str]:
    return event.get("root_trace_id") or event.get("trace_id")


def group_by_run(events: List[Dict[str, Any]]) -> "OrderedDict[str, List[Dict[str, Any]]]":
    """Group raw trace events into runs, keyed by root_trace_id."""
    runs: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
    for ev in events:
        if not isinstance(ev, dict):
            continue
        rid = _run_id(ev) or f"_anon_{len(runs)}"
        runs.setdefault(rid, []).append(ev)
    return runs


def _draft(metric: str, kind: str, rid: str, *, inputs=None, trace=None,
           suggested: Optional[Dict[str, Any]] = None,
           scrubbed: Optional[List[str]] = None) -> GoldenCase:
    note = "HARVESTED — review and set expected_pass."
    if scrubbed:
        note += f" [scrubbed: {', '.join(scrubbed)}]"
    if suggested is not None:
        note += f" (judge suggested: pass={suggested.get('passed')} score={suggested.get('score')})"
    return GoldenCase(
        id=f"harvest-{metric.replace('agentic.', '')}-{str(rid)[:8]}",
        metric=metric,
        kind=kind,
        inputs=inputs or {},
        trace=trace,
        expected_pass=None,          # unlabelled by design
        note=note,
    )


def harvest_run(rid: str, events: List[Dict[str, Any]], judge: Optional[LLMJudge] = None,
                threshold: float = 0.7, scrub: bool = True) -> List[GoldenCase]:
    """Classify one run into zero or more draft golden cases.

    PII/secrets are scrubbed (default on) *before* the run is parsed, so every
    derived draft — trace envelopes and single-shot inputs alike — is already
    sanitized; only redaction placeholders (not real values) land in the repo.
    """
    from jobs.agentic.adapters import from_fluiq
    from jobs.agentic.graph import build_graph

    scrubbed_entities: List[str] = []
    if scrub:
        events, scrubbed_entities = scrub_value(events)

    run = from_fluiq(events)
    graph = build_graph(run)
    drafts: List[GoldenCase] = []
    trace_env = {"events": events}
    _ent = scrubbed_entities or None

    has_tools = bool(run.tool_calls)
    is_multi_step = len(run.steps) >= 2
    has_join = any(graph.is_join(nid) for nid in graph.step_by_id)

    def _suggest(build):
        if judge is None:
            return None
        try:
            return build().model_dump(mode="json")
        except Exception:
            return None

    if has_tools:
        from jobs.agentic import deterministic
        from jobs.agentic.tool_selection import ToolSelectionQuality
        det = deterministic.check(run, graph)
        drafts.append(_draft(
            "agentic.tool_selection_quality", "agentic", rid, trace=trace_env, scrubbed=_ent,
            suggested=_suggest(lambda: ToolSelectionQuality(judge=judge, threshold=threshold).evaluate(
                goal=run.goal, tool_calls=run.tool_calls,
                available_tools=run.available_tools, deterministic=det)),
        ))
    if is_multi_step:
        from jobs.agentic.trajectory import TrajectoryEvaluator
        drafts.append(_draft(
            "agentic.trajectory", "agentic", rid, trace=trace_env, scrubbed=_ent,
            suggested=_suggest(lambda: TrajectoryEvaluator(judge=judge, threshold=threshold).evaluate(
                goal=run.goal, steps=run.steps, final_output=run.final_output, graph=graph)),
        ))
    if has_join:
        from jobs.agentic.coordination import MultiAgentEvaluator
        drafts.append(_draft(
            "agentic.coordination", "agentic", rid, trace=trace_env, scrubbed=_ent,
            suggested=_suggest(lambda: MultiAgentEvaluator(judge=judge, threshold=threshold).evaluate(
                run=run, graph=graph)),
        ))

    # A plain LLM answer (question + response, no tools) → single-shot drafts.
    if not has_tools and run.goal and run.final_output:
        inputs = {"question": run.goal, "answer": run.final_output}
        for metric in ("hallucination", "relevance"):
            drafts.append(_draft(metric, "single_shot", f"{rid}-{metric}", inputs=inputs, scrubbed=_ent))

    return drafts


def harvest(events: List[Dict[str, Any]], judge: Optional[LLMJudge] = None,
            threshold: float = 0.7, max_cases: Optional[int] = None,
            scrub: bool = True) -> List[GoldenCase]:
    """Turn a flat list of raw trace events into draft golden cases.

    ``scrub`` (default True) redacts PII/secrets from every case before it is
    written to disk."""
    drafts: List[GoldenCase] = []
    for rid, run_events in group_by_run(events).items():
        try:
            drafts.extend(harvest_run(rid, run_events, judge=judge, threshold=threshold, scrub=scrub))
        except Exception:
            logger.warning("[HARVEST] run %s failed to classify", rid, exc_info=True)
        if max_cases is not None and len(drafts) >= max_cases:
            return drafts[:max_cases]
    return drafts


def write_candidates(cases: List[GoldenCase], directory: Optional[str] = None) -> Optional[str]:
    """Write draft cases to a timestamped file under golden/candidates/."""
    if not cases:
        return None
    directory = directory or _CANDIDATES_DIR
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"harvested_{int(time.time())}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([c.model_dump(mode="json") for c in cases], fh, indent=2)
    return path


def promote_labeled(candidate_path: str, target_path: str) -> Dict[str, int]:
    """Merge *labelled* candidates (expected_pass or expected_score set) from a
    candidate file into a target golden file. Unlabelled drafts are skipped."""
    with open(candidate_path, "r", encoding="utf-8") as fh:
        rows = json.load(fh)

    labelled = [
        r for r in rows
        if isinstance(r, dict) and (r.get("expected_pass") is not None or r.get("expected_score") is not None)
    ]

    existing: List[Dict[str, Any]] = []
    if os.path.isfile(target_path):
        try:
            with open(target_path, "r", encoding="utf-8") as fh:
                existing = json.load(fh)
        except Exception:
            existing = []
    seen_ids = {r.get("id") for r in existing if isinstance(r, dict)}

    added = [r for r in labelled if r.get("id") not in seen_ids]
    if added:
        with open(target_path, "w", encoding="utf-8") as fh:
            json.dump(existing + added, fh, indent=2)

    return {"total": len(rows), "labelled": len(labelled), "promoted": len(added)}


def main() -> None:
    """CLI: pull real traces from ClickHouse and write draft candidates.

        python -m jobs.calibration.harvest [--limit N] [--hours H] [--suggest]
    """
    import asyncio
    import sys
    import config
    from db.clickhouse import clickhouse_eval_client

    argv = sys.argv[1:]

    def _arg(flag, default):
        return argv[argv.index(flag) + 1] if flag in argv else default

    limit = int(_arg("--limit", "200"))
    hours = int(_arg("--hours", "168"))
    judge = (LLMJudge(provider=config.JUDGE_PROVIDER, model=config.JUDGE_MODEL)
             if "--suggest" in argv else None)

    async def _run():
        await clickhouse_eval_client.start()
        try:
            events = await clickhouse_eval_client.fetch_recent_trace_events(limit=limit, since_hours=hours)
        finally:
            await clickhouse_eval_client.stop()
        return events

    from jobs.calibration.scrub import presidio_available

    scrub = "--no-scrub" not in argv
    events = asyncio.run(_run())
    drafts = harvest(events, judge=judge, threshold=config.JUDGE_THRESHOLD, scrub=scrub)
    path = write_candidates(drafts)
    print(f"Harvested {len(events)} events → {len(drafts)} draft cases "
          f"(scrub={'on' if scrub else 'OFF'}, presidio={'yes' if presidio_available() else 'regex-only'})")
    if path:
        print(f"Wrote {path}\nLabel expected_pass, then: promote_labeled(<file>, golden/<file>.json)")


if __name__ == "__main__":
    main()
