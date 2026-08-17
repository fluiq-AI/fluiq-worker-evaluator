"""Test bootstrap for the evaluator suite.

``config`` reads its settings with ``os.getenv`` at import time, and several test
modules set those env vars at their own module scope. That works when a module
is run on its own, and breaks under ``pytest tests/``: whichever module imports
``config`` first freezes the values, so every module collected after it sees
whatever the first one happened to have set — leaving ``JUDGE_PROVIDER`` as None
and failing seven tests that pass individually.

conftest runs before any test module is imported, so setting the defaults here
fixes the ordering dependency once. The per-module ``setdefault`` calls stay
where they are and remain harmless: this simply gets there first.

Nothing here connects to anything. These are placeholders that let real code be
imported and exercised.
"""
from __future__ import annotations

import base64
import os
import secrets
import sys
from pathlib import Path

EVALUATOR_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EVALUATOR_ROOT))

_DEFAULTS = {
    # Judge selection — the one that actually breaks the suite when missing.
    "EVAL_JUDGE_PROVIDER":  "openai",
    "EVAL_JUDGE_MODEL":     "gpt-4o-mini",
    "EVAL_JUDGE_THRESHOLD": "0.7",
    "EVAL_JUDGE_CACHE":     "1",
    "EVAL_JUDGE_CACHE_TTL": "300",
    "EVAL_JUDGE_CACHE_MAX": "1000",
    # config int-coerces these, so absence is an import-time TypeError.
    "CLICKHOUSE_PORT":      "8123",
    "CLICKHOUSE_HOST":      "localhost",
}

for name, value in _DEFAULTS.items():
    os.environ.setdefault(name, value)

# Credential tests need a local encryption backend rather than KMS.
os.environ.setdefault("CREDENTIAL_ENCRYPTION_BACKEND", "local")
os.environ.setdefault(
    "CREDENTIAL_ENCRYPTION_LOCAL_KEY",
    base64.b64encode(secrets.token_bytes(32)).decode(),
)
