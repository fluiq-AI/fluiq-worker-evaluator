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
