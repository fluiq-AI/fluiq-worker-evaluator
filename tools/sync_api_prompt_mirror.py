"""Generate (or check) the API's mirrored copy of the judge-prompt defaults.

The prompts live twice on purpose — the worker and the API are separate
deployables and cannot share code at runtime — but the duplicate was
hand-maintained and had already drifted (the API copy was missing
``retrieval_quality`` entirely). Generating it removes that failure mode: the
worker registry is the single source of truth and the API file is derived.

    python tools/sync_api_prompt_mirror.py           # rewrite the API file
    python tools/sync_api_prompt_mirror.py --check   # exit 1 if out of sync

``--check`` is the useful one in CI: it needs both trees checked out, which is
true in this monorepo-of-repos layout but not inside either deploy image, so it
is a pre-merge guard rather than a runtime assertion.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

# config.py reads several vars with float()/int() and raises on import if unset.
for _k, _v in [
    ("EVAL_JUDGE_THRESHOLD", "0.7"),
    ("EVAL_JUDGE_CACHE_TTL", "300"),
    ("EVAL_JUDGE_CACHE_MAX", "1000"),
    ("CLICKHOUSE_PORT", "8123"),
]:
    os.environ.setdefault(_k, _v)

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from jobs.helper.judge_prompts import _PROMPTS  # noqa: E402

# evaluator/ -> fluiq-workers/ -> source/ -> fluiq-api/...
DEFAULT_TARGET = (
    _HERE.parent.parent.parent
    / "fluiq-api" / "db_queues" / "postgresql" / "judge_prompt_defaults.py"
)

HEADER = '''"""Canonical default LLM-as-Judge prompts, mirrored for seeding from the API.

GENERATED FILE — do not edit by hand.

Source of truth: ``fluiq-workers/evaluator/jobs/helper/judge_prompts.py``
(``_PROMPTS``). Regenerate with::

    python tools/sync_api_prompt_mirror.py

from the evaluator repo, and check it in CI with ``--check``.

The text is duplicated because the two deployables cannot share code at
runtime: the worker uses its copy for fail-open rendering when Postgres is
unreachable, while the API uses this copy to seed/refresh the
``eval_judge_prompts`` table on startup so the Admin "Judge Prompts" tab is
populated as soon as the API is up, independent of whether the evaluator worker
has booted yet. Both seeders refresh ``default_template`` and carry unedited
rows forward, and neither clobbers a row an org has overridden.

Syntax: ``{{variable}}``, the product-wide placeholder standard. Only
``{{identifier}}`` is treated as a placeholder, so the literal JSON braces these
prompts are full of need no escaping. Prompts saved before the switch may still
use the legacy ``$var`` / ``${var}`` form; both still substitute at render time.
"""

JUDGE_PROMPT_DEFAULTS: list[dict] = [
'''

FOOTER = ']\n\n__all__ = ["JUDGE_PROMPT_DEFAULTS"]\n'


def render() -> str:
    out = [HEADER]
    for name, spec in _PROMPTS.items():
        out.append("    {\n")
        out.append("        %r: %r,\n" % ("name", name))
        out.append("        %r: %r,\n" % ("description", spec.get("description")))
        out.append("        %r: %r,\n" % ("required_vars", list(spec.get("required", []))))
        out.append("        %r: (\n            %r\n        ),\n" % ("template", spec["template"]))
        out.append("    },\n")
    out.append(FOOTER)
    return "".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit 1 when the mirror is stale instead of rewriting it")
    ap.add_argument("--target", type=pathlib.Path, default=DEFAULT_TARGET)
    args = ap.parse_args()

    generated = render()

    if not args.target.exists():
        if args.check:
            print(f"MISSING: {args.target}")
            return 1
        args.target.parent.mkdir(parents=True, exist_ok=True)

    current = args.target.read_text(encoding="utf-8") if args.target.exists() else ""

    if args.check:
        if current != generated:
            print("OUT OF SYNC: the API prompt mirror does not match the worker registry.")
            print("Regenerate with: python tools/sync_api_prompt_mirror.py")
            return 1
        print(f"in sync: {len(_PROMPTS)} prompts")
        return 0

    if current == generated:
        print(f"already in sync: {len(_PROMPTS)} prompts")
        return 0

    args.target.write_text(generated, encoding="utf-8")
    print(f"wrote {len(_PROMPTS)} prompts -> {args.target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
