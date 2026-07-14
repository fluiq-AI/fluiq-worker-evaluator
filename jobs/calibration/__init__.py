"""Judge calibration — measure how well the evaluator's LLM-judges agree with
human labels on a held-out golden set.

Once the evaluator has more than one moving part (decomposition, juries, panels),
"is the judge actually right?" stops being obvious. This package answers it:
a labelled corpus (`golden/*.json`) + a runner that scores each case with the
real judge and reports agreement (accuracy, Cohen's κ, precision/recall, and
score MAE / correlation) per metric. Run it to gate releases and to detect judge
drift when the underlying model updates.
"""
from jobs.calibration.golden import GoldenCase, load_golden
from jobs.calibration.metrics import agreement, score_agreement

__all__ = ["GoldenCase", "load_golden", "agreement", "score_agreement"]
