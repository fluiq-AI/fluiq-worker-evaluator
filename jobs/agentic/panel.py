"""Layer 4 — multi-agent judge panel (jury), with confidence gating.

Runs one metric through several judges and aggregates their verdicts. This is
the only place true "multi-agent" judging lives, and it is *gated*: a primary
judge runs first, and the rest of the panel is convened only when the primary's
verdict is near the pass/fail threshold (low confidence) — so full panel cost is
paid on the ~10-20% of evals where it changes the answer, not on every call.

Modes (``EVAL_PANEL_MODE``):
  * ``off``    — single judge, no panel (cheapest).
  * ``gated``  — primary + escalate near threshold (default; recommended).
  * ``always`` — full panel on every metric (max quality, max cost).

Every member's score/reason is kept in the result as an *eval-trace*, so a panel
score is auditable rather than a black box. All members are best-effort: a judge
whose provider isn't configured (missing API key) is skipped, and aggregation
uses whoever answered.
"""
from __future__ import annotations

import logging
import statistics
from typing import Any, Callable, Dict, List, Optional

from jobs.helper.base import BaseEvaluator, EvalResult
from jobs.helper.judge import LLMJudge

logger = logging.getLogger(__name__)

# Factory: given a judge, build the evaluator to run (e.g. ToolSelectionQuality).
BuildEvaluator = Callable[[LLMJudge], BaseEvaluator]


class JudgePanel:
    def __init__(
        self,
        members: List[LLMJudge],
        mode: str = "gated",
        gate_margin: float = 0.12,
        threshold: float = 0.7,
    ) -> None:
        self.members = [m for m in members if m is not None]
        self.mode = mode if mode in ("off", "gated", "always") else "gated"
        self.gate_margin = gate_margin
        self.threshold = threshold

    def _run_member(self, build: BuildEvaluator, judge: LLMJudge, kwargs: Dict[str, Any]) -> Optional[EvalResult]:
        try:
            return build(judge).evaluate(**kwargs)
        except Exception:
            logger.warning(
                "[PANEL] member judge failed provider=%s model=%s", judge.provider, judge.model,
                exc_info=True,
            )
            return None

    def evaluate(self, build: BuildEvaluator, kwargs: Dict[str, Any]) -> EvalResult:
        if not self.members:
            raise ValueError("JudgePanel needs at least one member judge")

        primary_judge = self.members[0]
        primary = self._run_member(build, primary_judge, kwargs)
        primary_failed = primary is None

        member_traces: List[Dict[str, Any]] = [{
            "role": "primary",
            "provider": primary_judge.provider,
            "model": primary_judge.model,
            **(
                {"failed": True, "reason": "judge unavailable"}
                if primary_failed
                else {"score": primary.score, "reason": primary.reason}
            ),
        }]

        # Convene the jury when configured to, when the primary is unsure — or
        # when the primary DIED: the jurors are then the only path to a real
        # verdict at all, so a gated panel escalates for a dead primary too.
        convene = self.mode == "always" or (
            self.mode == "gated"
            and len(self.members) > 1
            and (
                primary_failed
                or abs(primary.score - self.threshold) <= self.gate_margin
            )
        )

        if not convene:
            if primary_failed:
                # Single-judge path with no judge: neutral, never a verdict.
                return EvalResult(
                    name="agentic.panel", score=0.5, passed=None,
                    reason="judge unavailable",
                    details={"panel": {
                        "mode": self.mode, "convened": False,
                        "members": member_traces, "agreement": None,
                    }},
                )
            details = dict(primary.details or {})
            details["panel"] = {
                "mode": self.mode, "convened": False,
                "members": member_traces, "agreement": 1.0,
            }
            return EvalResult(
                name=primary.name, score=primary.score,
                passed=primary.passed, reason=primary.reason, details=details,
            )

        # Only GENUINE verdicts vote. A failed member is recorded in the trace
        # but never contributes a synthetic score — a dead primary must not
        # dilute (or manufacture) the jury's verdict.
        results: List[EvalResult] = [] if primary_failed else [primary]
        for judge in self.members[1:]:
            r = self._run_member(build, judge, kwargs)
            if r is None:
                member_traces.append({
                    "role": "juror",
                    "provider": judge.provider, "model": judge.model,
                    "failed": True, "reason": "judge unavailable",
                })
                continue
            results.append(r)
            member_traces.append({
                "role": "juror",
                "provider": judge.provider, "model": judge.model,
                "score": r.score, "reason": r.reason,
            })

        if not results:
            # Every judge failed — an infrastructure outage is not a failing
            # agent run. Surface neutral (passed=None) so run_passed ignores it.
            return EvalResult(
                name="agentic.panel", score=0.5, passed=None,
                reason="no judge available (all panel members failed)",
                details={"panel": {
                    "mode": self.mode, "convened": True,
                    "members": member_traces, "agreement": None,
                    "votes_pass": 0, "votes_total": 0,
                }},
            )

        scores = [r.score for r in results]
        agg_score = statistics.mean(scores)
        # Agreement: 1 when jurors are unanimous, → 0 as spread widens.
        spread = (statistics.pstdev(scores) if len(scores) > 1 else 0.0)
        agreement = max(0.0, 1.0 - 2.0 * spread)
        votes_pass = sum(1 for s in scores if s >= self.threshold)
        # The verdict follows the aggregate score, so `passed` can never
        # contradict the number the dashboard displays (a 1-1 split with a
        # failing mean used to "pass" on the tie rule). Votes stay in the
        # details as the per-member audit trail.
        passed = agg_score >= self.threshold

        base = results[0]
        base_details = dict(base.details or {})
        base_details["panel"] = {
            "mode": self.mode,
            "convened": True,
            "members": member_traces,
            "agreement": round(agreement, 3),
            "votes_pass": votes_pass,
            "votes_total": len(scores),
            "score_spread": round(spread, 3),
        }
        reason = (
            f"panel of {len(scores)}: mean={agg_score:.2f} "
            f"agreement={agreement:.2f} votes_pass={votes_pass}/{len(scores)}"
        )
        return EvalResult(
            name=base.name, score=agg_score, passed=passed,
            reason=reason, details=base_details,
        )


def build_panel(
    primary: LLMJudge,
    member_specs: List[tuple],
    mode: str,
    gate_margin: float,
    threshold: float,
    judge_factory: Callable[[str, str], LLMJudge],
) -> JudgePanel:
    """Assemble a panel: the configured primary judge plus jurors from
    ``member_specs`` (list of ``(provider, model)``). ``judge_factory`` builds a
    cached judge for a provider/model (so jurors share the response cache)."""
    members: List[LLMJudge] = [primary]
    if mode != "off":
        for provider, model in member_specs:
            if provider == primary.provider and model == primary.model:
                continue
            try:
                members.append(judge_factory(provider, model))
            except Exception:
                logger.warning("[PANEL] could not build juror %s:%s", provider, model, exc_info=True)
    return JudgePanel(members=members, mode=mode, gate_margin=gate_margin, threshold=threshold)
