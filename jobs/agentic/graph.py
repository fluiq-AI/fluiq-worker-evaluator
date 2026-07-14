"""Phase 1 — reconstruct the run's DAG from step parents.

An agent run is a *graph*, not a line: a step can fan out into parallel branches
and a later step can join several of them (multiple parents). The evaluator used
to flatten steps by arrival order, which (a) made legitimate parallel work look
like loops and (b) hid fan-out/join structure from the trajectory judge.

:class:`AgentGraph` builds adjacency from each step's effective parents
(:meth:`AgentStep.parents`), exposes a stable **topological order**, and answers
the two questions the evaluators need: *are these two steps on the same path?*
(loop detection) and *which nodes are joins / fan-outs?* (trajectory rendering).
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Optional, Set

from jobs.agentic.schema import AgentRun, AgentStep


class AgentGraph:
    def __init__(self, steps: List[AgentStep]) -> None:
        self.steps = steps

        # Assign a stable id per step (trace_id when present, else positional).
        self._id_by_step: Dict[int, str] = {}
        self.step_by_id: Dict[str, AgentStep] = {}
        for i, s in enumerate(steps):
            nid = s.trace_id or f"__step_{i}__"
            if nid in self.step_by_id:  # duplicate trace_id → disambiguate
                nid = f"{nid}#{i}"
            self._id_by_step[id(s)] = nid
            self.step_by_id[nid] = s

        ids: Set[str] = set(self.step_by_id)

        self.parents: Dict[str, Set[str]] = {}
        self.children: Dict[str, Set[str]] = defaultdict(set)
        for i, s in enumerate(steps):
            nid = self._id_by_step[id(s)]
            present = {p for p in s.parents() if p in ids and p != nid}
            self.parents[nid] = present
            for p in present:
                self.children[p].add(nid)

        self.roots: List[str] = [nid for nid in self.step_by_id if not self.parents.get(nid)]
        self.order: List[str] = self._toposort()
        self._anc_cache: Dict[str, Set[str]] = {}

    # ── ids ───────────────────────────────────────────────────────────────
    def id_for(self, step: AgentStep) -> str:
        return self._id_by_step[id(step)]

    # ── ordering ──────────────────────────────────────────────────────────
    def _toposort(self) -> List[str]:
        """Kahn's algorithm, stable by original step order. Any nodes left in a
        cycle are appended in arrival order so evaluation never stalls."""
        indeg = {nid: len(self.parents.get(nid, ())) for nid in self.step_by_id}
        arrival = {self._id_by_step[id(s)]: i for i, s in enumerate(self.steps)}
        ready = deque(sorted((nid for nid, d in indeg.items() if d == 0), key=arrival.get))
        out: List[str] = []
        seen: Set[str] = set()
        while ready:
            nid = ready.popleft()
            if nid in seen:
                continue
            seen.add(nid)
            out.append(nid)
            newly: List[str] = []
            for c in sorted(self.children.get(nid, ()), key=arrival.get):
                indeg[c] -= 1
                if indeg[c] == 0:
                    newly.append(c)
            ready.extend(newly)
        if len(out) < len(self.step_by_id):  # cycle remnants
            for nid in sorted(self.step_by_id, key=arrival.get):
                if nid not in seen:
                    out.append(nid)
        return out

    def ordered_steps(self) -> List[AgentStep]:
        return [self.step_by_id[nid] for nid in self.order]

    # ── lineage ───────────────────────────────────────────────────────────
    def ancestors(self, nid: str) -> Set[str]:
        cached = self._anc_cache.get(nid)
        if cached is not None:
            return cached
        acc: Set[str] = set()
        stack = list(self.parents.get(nid, ()))
        while stack:
            p = stack.pop()
            if p in acc:
                continue
            acc.add(p)
            stack.extend(self.parents.get(p, ()))
        self._anc_cache[nid] = acc
        return acc

    def same_lineage(self, a: str, b: str) -> bool:
        """True when a and b are the same node or one is an ancestor of the
        other — i.e. they lie on the *same path*. Two nodes in sibling parallel
        branches are NOT same-lineage."""
        if a == b:
            return True
        return b in self.ancestors(a) or a in self.ancestors(b)

    # ── classification ────────────────────────────────────────────────────
    def is_join(self, nid: str) -> bool:
        return len(self.parents.get(nid, ())) > 1

    def is_fan_out(self, nid: str) -> bool:
        return len(self.children.get(nid, ())) > 1

    def depth(self, nid: str) -> int:
        anc = self.ancestors(nid)
        return len(anc)  # cheap proxy for indentation; exact longest-path not needed

    def summary(self) -> Dict[str, object]:
        joins = [nid for nid in self.step_by_id if self.is_join(nid)]
        fan_outs = [nid for nid in self.step_by_id if self.is_fan_out(nid)]
        return {
            "nodes": len(self.step_by_id),
            "edges": sum(len(v) for v in self.parents.values()),
            "roots": len(self.roots),
            "joins": len(joins),
            "fan_outs": len(fan_outs),
            "is_dag": bool(joins) or bool(fan_outs),
        }


def build_graph(run: AgentRun) -> AgentGraph:
    return AgentGraph(run.steps)
