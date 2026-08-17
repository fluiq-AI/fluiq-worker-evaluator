"""Choice scores: a judge picks a written label, and a table turns it into a number.

Asking a model for a calibrated float is asking it to do the one thing it is
worst at. Scores come back clustered around whatever threshold the author
mentioned, they move when the prompt is reworded, and two runs on identical
input disagree. Asking it to choose between written options is a classification
task — the thing models are reliably good at — and the number then comes from a
lookup the author controls, so it never varies at all.

This is the mechanism behind Braintrust's "Choice scores" (Y → 1, N → 0), and the
same insight already behind this repo's built-in Likert judges: the mapping lives
in code, not in the model.

A choice set is a list of ``{label, score}``. Labels must be unique and scores
must be in 0..1. The judge is offered exactly those labels, and an answer outside
the set is an error — coercing it would invent a grade.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

MAX_CHOICES = 12
MAX_LABEL_CHARS = 60


class ChoiceError(ValueError):
    """A choice set, or a judge's answer against one, is unusable."""


def parse_choices(raw: Any) -> Optional[List[Dict[str, Any]]]:
    """Validate a choice set. Returns None when there isn't one.

    None means "score this the ordinary way" — a scorer without choices keeps
    the free-score behaviour it has always had.
    """
    if raw in (None, "", [], {}):
        return None
    if not isinstance(raw, (list, tuple)):
        raise ChoiceError("Choices must be a list of {label, score}.")
    if len(raw) < 2:
        raise ChoiceError("A choice set needs at least two options to choose between.")
    if len(raw) > MAX_CHOICES:
        raise ChoiceError(f"A choice set can hold at most {MAX_CHOICES} options.")

    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, Mapping):
            raise ChoiceError("Each choice must be an object with a label and a score.")
        label = str(item.get("label") or "").strip()
        if not label:
            raise ChoiceError("Every choice needs a label.")
        if len(label) > MAX_LABEL_CHARS:
            raise ChoiceError(f"Labels must be {MAX_LABEL_CHARS} characters or fewer.")
        # Case-insensitive, because a judge that answers "yes" for a "Yes" label
        # has understood the question perfectly.
        key = label.lower()
        if key in seen:
            raise ChoiceError(f"Duplicate choice label: {label!r}.")
        seen.add(key)
        try:
            score = float(item.get("score"))
        except (TypeError, ValueError):
            raise ChoiceError(f"Choice {label!r} needs a numeric score.") from None
        if not 0.0 <= score <= 1.0:
            raise ChoiceError(f"Choice {label!r} has score {score:g}; must be 0 to 1.")
        out.append({"label": label, "score": score})
    return out


def labels_of(choices: Sequence[Mapping[str, Any]]) -> List[str]:
    return [str(c["label"]) for c in choices]


def describe(choices: Sequence[Mapping[str, Any]]) -> str:
    """The instruction appended to a judge prompt that has a choice set.

    Stated even when the provider enforces the enum: a model told what the
    options mean chooses better than one merely prevented from saying anything
    else.
    """
    options = "\n".join(f'  - "{c["label"]}"' for c in choices)
    return (
        "\n\nChoose exactly one of these options:\n"
        f"{options}\n\n"
        'Return ONLY a JSON object: {"choice": "<one of the options above>", '
        '"reason": "<short explanation>"}'
    )


def resolve(
    choices: Sequence[Mapping[str, Any]], answer: Any,
) -> Tuple[float, str]:
    """Map a judge's chosen label to its score. Returns ``(score, label)``.

    Raises when the answer is not one of the offered labels. That is deliberate:
    an unrecognised answer means the judge did not do the task, and scoring it 0
    would be indistinguishable from a genuine failing grade.
    """
    raw = "" if answer is None else str(answer).strip()
    if not raw:
        raise ChoiceError("Judge returned no choice.")
    lowered = raw.lower()
    for choice in choices:
        if str(choice["label"]).strip().lower() == lowered:
            return float(choice["score"]), str(choice["label"])
    raise ChoiceError(
        f"Judge answered {raw!r}, which is not one of: "
        f"{', '.join(labels_of(choices))}"
    )


__all__ = [
    "MAX_CHOICES",
    "MAX_LABEL_CHARS",
    "ChoiceError",
    "describe",
    "labels_of",
    "parse_choices",
    "resolve",
]
