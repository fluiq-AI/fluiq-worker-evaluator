"""PII / secret scrubbing for harvested golden candidates.

The evaluator is deployed **without** the security worker's heavy deps
(Presidio / spaCy / torch), so this scrubber is a dependency-free regex baseline
that always runs, covering the high-confidence patterns: emails, phones, US
SSNs, credit cards, IP addresses, and the same API-key / token patterns the
security worker's custom recognizers use. If Presidio *is* importable (e.g. run
in an image that has it), :func:`scrub_text` augments the regex pass with a full
Presidio analyze+anonymize for broader coverage (PERSON, LOCATION, IBAN, …).

Best-effort by design: scrubbing production traces before they land in the repo
is a safety net, not a guarantee — the candidate README still asks a human to
review and redact before promotion.
"""
from __future__ import annotations

import logging
import re
from typing import Any, List, Tuple

logger = logging.getLogger(__name__)

# Ordered most-specific first. Secret patterns mirror the security worker's
# custom recognizers (jobs/helper/pii.py) so redaction is consistent.
_PATTERNS: List[Tuple[str, "re.Pattern"]] = [
    ("ANTHROPIC_API_KEY", re.compile(r"sk-ant-[a-zA-Z0-9\-]{20,}")),
    ("OPENAI_API_KEY",    re.compile(r"sk-[a-zA-Z0-9]{20,}")),
    ("AWS_ACCESS_KEY",    re.compile(r"AKIA[0-9A-Z]{16}")),
    ("GITHUB_TOKEN",      re.compile(r"ghp_[a-zA-Z0-9]{36}")),
    ("STRIPE_LIVE_KEY",   re.compile(r"sk_live_[a-zA-Z0-9]{24}")),
    ("EMAIL_ADDRESS",     re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("US_SSN",            re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("CREDIT_CARD",       re.compile(
        r"\b(?:4\d{12}(?:\d{3})?|5[1-5]\d{14}|3[47]\d{13}|6(?:011|5\d{2})\d{12})\b")),
    ("PHONE_NUMBER",      re.compile(
        r"(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}\b")),
    ("IP_ADDRESS",        re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
]

# Presidio is optional and only used when installed.
try:
    from presidio_analyzer import AnalyzerEngine  # noqa: F401
    from presidio_anonymizer import AnonymizerEngine  # noqa: F401
    _PRESIDIO_AVAILABLE = True
except Exception:
    _PRESIDIO_AVAILABLE = False

_analyzer = None
_anonymizer = None
# Deliberately NARROW: only high-sensitivity entities. PERSON / LOCATION are
# excluded on purpose — for calibration cases they are usually the semantic
# content (e.g. "the capital of France is Paris"), and redacting them would
# destroy the case. The regex pass above already covers contact PII + secrets.
_PRESIDIO_ENTITIES = [
    "IBAN_CODE", "CRYPTO", "US_PASSPORT", "MEDICAL_LICENSE", "NRP", "US_BANK_NUMBER",
]


def _presidio_engines():
    global _analyzer, _anonymizer
    if _analyzer is None:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine
        _analyzer = AnalyzerEngine()
        _anonymizer = AnonymizerEngine()
    return _analyzer, _anonymizer


def scrub_text(text: Any, use_presidio: bool = True) -> Tuple[Any, List[str]]:
    """Redact PII/secrets from a string. Returns (scrubbed_text, entity_types)."""
    if not isinstance(text, str) or not text:
        return text, []
    found: List[str] = []
    for entity, pattern in _PATTERNS:
        if pattern.search(text):
            text = pattern.sub(f"<{entity}>", text)
            found.append(entity)

    if use_presidio and _PRESIDIO_AVAILABLE:
        try:
            analyzer, anonymizer = _presidio_engines()
            results = analyzer.analyze(text=text, entities=_PRESIDIO_ENTITIES, language="en")
            if results:
                text = anonymizer.anonymize(text=text, analyzer_results=results).text
                found.extend(sorted({r.entity_type for r in results}))
        except Exception:
            logger.warning("[SCRUB] presidio pass failed; regex-only", exc_info=True)

    return text, found


def scrub_value(value: Any, use_presidio: bool = True) -> Tuple[Any, List[str]]:
    """Recursively scrub every string inside a dict / list / scalar.
    Returns (scrubbed_value, sorted_unique_entity_types)."""
    found: set = set()

    def _walk(v: Any) -> Any:
        if isinstance(v, str):
            scrubbed, ents = scrub_text(v, use_presidio=use_presidio)
            found.update(ents)
            return scrubbed
        if isinstance(v, dict):
            return {k: _walk(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [_walk(x) for x in v]
        return v

    return _walk(value), sorted(found)


def presidio_available() -> bool:
    return _PRESIDIO_AVAILABLE
