"""Golden-set case model + loader.

A golden case is one human-labelled example: the inputs a metric's evaluator
expects, plus the human verdict (``expected_pass`` and/or ``expected_score``).
Cases live as JSON lists under ``jobs/calibration/golden/`` and are loaded and
concatenated. Extend the corpus by dropping in more files — no code change.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "golden")


class GoldenCase(BaseModel):
    id: str
    metric: str
    # "single_shot" (default): score `inputs` with a single-shot metric.
    # "agentic": normalize `trace` into an AgentRun and score it with an agentic
    #            layer (tool_selection_quality / trajectory / coordination / deterministic).
    kind: str = "single_shot"
    inputs: Dict[str, Any] = Field(default_factory=dict)
    # For agentic cases: a trace envelope — {"events": [...]} / {"event": {...}} /
    # {"source": "openinference", "spans": [...]} — anything adapters.normalize accepts.
    trace: Optional[Dict[str, Any]] = None
    expected_pass: Optional[bool] = None
    expected_score: Optional[float] = None
    note: Optional[str] = None


def load_golden(path: Optional[str] = None) -> List[GoldenCase]:
    """Load every ``*.json`` golden file in the directory into GoldenCases."""
    directory = path or _GOLDEN_DIR
    cases: List[GoldenCase] = []
    if not os.path.isdir(directory):
        return cases
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(directory, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as fh:
                rows = json.load(fh)
            for row in rows if isinstance(rows, list) else []:
                try:
                    case = GoldenCase(**row)
                except Exception:
                    logger.warning("[CALIBRATION] bad golden case in %s: %r", fname, row)
                    continue
                # Skip unlabelled drafts (harvested candidates awaiting a human
                # label) so they can never pollute calibration accuracy.
                if case.expected_pass is None and case.expected_score is None:
                    continue
                cases.append(case)
        except Exception:
            logger.warning("[CALIBRATION] could not read golden file %s", fpath, exc_info=True)
    return cases
