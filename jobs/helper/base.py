import json
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class EvalResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    score: float = Field(ge=0.0, le=1.0)
    passed: Optional[bool] = None
    reason: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class BaseEvaluator(ABC):
    name: str = "evaluator"

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    @abstractmethod
    def evaluate(self, **kwargs: Any) -> EvalResult:
        ...

    def _result(
        self,
        score: float,
        reason: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> EvalResult:
        return EvalResult(
            name=self.name,
            score=float(score),
            passed=float(score) >= self.threshold,
            reason=reason,
            details=details,
        )


def _parse_json_object(raw: str) -> Dict[str, Any]:
    if not raw:
        return {}
    raw = raw.strip()
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {"value": data}
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            return data if isinstance(data, dict) else {"value": data}
        except json.JSONDecodeError:
            return {}
    return {}


def _coerce_contexts(contexts: Any) -> List[str]:
    if contexts is None:
        return []
    if isinstance(contexts, str):
        return [contexts]
    if isinstance(contexts, (list, tuple)):
        return [str(c) for c in contexts if c is not None]
    return [str(contexts)]


def _clamp_unit(value: Any, default: float = 0.0) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, f))


# ── Discrete judge scale ─────────────────────────────────────────────────────
#
# Judges rate on an integer 1..5 against a written rubric line per point, and
# the mapping to 0..1 happens here rather than in the model. Picking one of five
# labelled options is a classification task an LLM does reliably; emitting a
# calibrated real number is not. This is the Prometheus / MT-Bench convention —
# essentially every published judge uses a discrete scale — and it removes the
# score clustering that a free 0..1 float produces around the pass threshold.
LIKERT_MAX = 5


def likert_to_unit(rating: Any, default: float = 0.0) -> float:
    """Map an integer rating on 1..LIKERT_MAX onto 0..1.

    1 -> 0.00, 2 -> 0.25, 3 -> 0.50, 4 -> 0.75, 5 -> 1.00.
    """
    try:
        r = float(rating)
    except (TypeError, ValueError):
        return default
    r = max(1.0, min(float(LIKERT_MAX), r))
    return (r - 1.0) / (LIKERT_MAX - 1.0)


def judge_score(data: Dict[str, Any], default: float = 0.0, key: str = "rating") -> float:
    """Read a judge's verdict as 0..1, accepting both scales.

    Prefers the discrete ``rating`` the shipped prompts now ask for, and falls
    back to a raw 0..1 ``score``. The fallback is not politeness — an org that
    edited its prompt before the switch keeps that template forever, because
    the seeder never overwrites a row with ``is_overridden`` set. Those judges
    still answer with ``score``, and dropping the fallback would silently zero
    every one of them.
    """
    if data.get(key) is not None:
        return likert_to_unit(data.get(key), default=default)
    if data.get("score") is not None:
        return _clamp_unit(data.get("score"), default=default)
    return default
