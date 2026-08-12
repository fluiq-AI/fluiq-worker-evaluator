"""Guard the hand-duplicated judge prompts against drift.

The API keeps a mirrored copy of the prompt defaults because the two
deployables cannot share code at runtime. That duplicate had already drifted
once — the API copy was missing ``retrieval_quality`` — which would have meant
the Admin "Judge Prompts" tab silently lacked a prompt the evaluator was using.

The mirror is now generated from this repo's registry, so this test only has to
assert it was regenerated. It skips when the API tree is not checked out beside
this one, which is the case inside the evaluator's deploy image.

    ../.workers-venv/Scripts/python.exe tests/test_prompt_mirror_sync.py
"""
import os
import subprocess
import sys

os.environ.setdefault("EVAL_JUDGE_THRESHOLD", "0.7")
os.environ.setdefault("EVAL_JUDGE_CACHE_TTL", "300")
os.environ.setdefault("EVAL_JUDGE_CACHE_MAX", "1000")
os.environ.setdefault("CLICKHOUSE_PORT", "8123")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

_TOOL = os.path.join(_ROOT, "tools", "sync_api_prompt_mirror.py")
_API_MIRROR = os.path.abspath(os.path.join(
    _ROOT, "..", "..", "fluiq-api", "db_queues", "postgresql", "judge_prompt_defaults.py",
))


def test_api_mirror_is_in_sync():
    if not os.path.exists(_API_MIRROR):
        print("  SKIP: fluiq-api tree not present beside this repo")
        return
    proc = subprocess.run(
        [sys.executable, _TOOL, "--check"],
        cwd=_ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 0, (
        "API prompt mirror is stale. Regenerate with:\n"
        "    python tools/sync_api_prompt_mirror.py\n\n" + proc.stdout + proc.stderr
    )
    print(" ", proc.stdout.strip())


def test_every_prompt_declares_the_vars_it_uses():
    """A prompt that renders a placeholder it never declares fails at runtime
    only for the org that hits it, so catch it here instead."""
    import re
    from jobs.helper.judge_prompts import _PROMPTS

    problems = []
    for name, spec in _PROMPTS.items():
        used = set(re.findall(r"\{\{(\w+)\}\}", spec["template"]))
        used.discard("today")  # injected by the renderer, never passed in
        declared = set(spec.get("required", []))
        undeclared_and_unfilled = used - declared
        # Optional vars are legitimately undeclared (the renderer defaults them
        # to ""), so this only flags a declared-but-absent placeholder.
        missing = declared - used
        if missing:
            problems.append(f"{name}: declares {sorted(missing)} but never uses them")
        del undeclared_and_unfilled
    assert not problems, "\n".join(problems)
    print(f"  {len(_PROMPTS)} prompts: declared vars all used")


def test_scalar_prompts_ask_for_reasoning_before_score():
    """Models emit JSON keys in the order requested, so a score-first schema
    makes the reason a post-hoc rationalisation. Every prompt that asks for a
    float score must ask for its reasoning first."""
    from jobs.helper.judge_prompts import _PROMPTS

    offenders = []
    for name, spec in _PROMPTS.items():
        t = spec["template"]
        if '\\"score\\"' not in t and '"score"' not in t:
            continue
        score_at = t.index('"score"') if '"score"' in t else t.index('\\"score\\"')
        reason_at = t.find('"reason"')
        if reason_at == -1:
            reason_at = t.find('\\"reason\\"')
        if reason_at == -1 or reason_at > score_at:
            offenders.append(name)
    assert not offenders, f"score requested before reason in: {offenders}"
    print("  every scored prompt asks for reasoning first")


def test_scalar_prompts_have_midpoint_anchors():
    """Endpoint-only scales ('0 = bad, 1 = good') leave 0.6 undefined, which is
    what produces score clustering around the 0.7 pass threshold."""
    from jobs.helper.judge_prompts import _PROMPTS

    # Prompts that return a bare 0..1 quality score the user can threshold on.
    anchored = [
        "hallucination_no_context", "answer_relevancy", "toxicity",
        "coherence", "completeness", "trajectory_quality",
        "vision_faithfulness", "media_faithfulness",
    ]
    missing = [
        n for n in anchored
        if not ("0.75" in _PROMPTS[n]["template"] and "0.25" in _PROMPTS[n]["template"])
    ]
    assert not missing, f"no midpoint anchors in: {missing}"
    print(f"  {len(anchored)} scalar prompts carry 0.25/0.5/0.75 anchors")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"Running {len(tests)} prompt-registry tests\n")
    failed = 0
    for t in tests:
        try:
            print(f"- {t.__name__}")
            t()
        except AssertionError as e:
            failed += 1
            print(f"  FAILED: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
