"""Pure agreement statistics (no I/O, no judge) — easy to unit-test.

`agreement` scores judge-vs-human pass/fail classification; `score_agreement`
scores continuous judge-vs-human scores. Cohen's kappa is the headline number:
it corrects raw accuracy for chance agreement, which matters when labels are
imbalanced (a judge that always says "pass" can score 90% accuracy yet κ≈0).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional


def agreement(predicted: List[bool], expected: List[bool]) -> Dict[str, float]:
    n = len(predicted)
    if n == 0:
        return {"n": 0, "accuracy": 0.0, "kappa": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    pred = [bool(p) for p in predicted]
    exp = [bool(e) for e in expected]

    correct = sum(1 for p, e in zip(pred, exp) if p == e)
    accuracy = correct / n

    # Cohen's kappa (binary)
    p_pred_true = sum(pred) / n
    p_exp_true = sum(exp) / n
    pe = p_pred_true * p_exp_true + (1 - p_pred_true) * (1 - p_exp_true)
    kappa = 1.0 if accuracy == 1.0 else (0.0 if pe >= 1.0 else (accuracy - pe) / (1 - pe))

    tp = sum(1 for p, e in zip(pred, exp) if p and e)
    fp = sum(1 for p, e in zip(pred, exp) if p and not e)
    fn = sum(1 for p, e in zip(pred, exp) if not p and e)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "n": n,
        "accuracy": round(accuracy, 4),
        "kappa": round(kappa, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def score_agreement(predicted: List[float], expected: List[float]) -> Dict[str, Optional[float]]:
    pairs = [(float(p), float(e)) for p, e in zip(predicted, expected)
             if p is not None and e is not None]
    n = len(pairs)
    if n == 0:
        return {"n": 0, "mae": None, "pearson": None}

    mae = sum(abs(p - e) for p, e in pairs) / n

    mp = sum(p for p, _ in pairs) / n
    me = sum(e for _, e in pairs) / n
    cov = sum((p - mp) * (e - me) for p, e in pairs)
    vp = sum((p - mp) ** 2 for p, _ in pairs)
    ve = sum((e - me) ** 2 for _, e in pairs)
    pearson = (cov / math.sqrt(vp * ve)) if vp > 0 and ve > 0 else None

    return {
        "n": n,
        "mae": round(mae, 4),
        "pearson": round(pearson, 4) if pearson is not None else None,
    }
