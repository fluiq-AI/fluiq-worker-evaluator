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


def test_scalar_prompts_ask_for_reasoning_before_the_verdict():
    """Models emit JSON keys in the order requested, so a verdict-first schema
    makes the reason a post-hoc rationalisation. Every prompt that asks for a
    rating must ask for its reasoning first."""
    from jobs.helper.judge_prompts import _PROMPTS

    offenders = []
    for name, spec in _PROMPTS.items():
        t = spec["template"]
        verdict_at = min(
            (t.index(k) for k in ('\\"rating\\"', '\\"severity_rating\\"') if k in t),
            default=-1,
        )
        if verdict_at == -1:
            continue
        reason_at = t.find('\\"reason\\"')
        if reason_at == -1 or reason_at > verdict_at:
            offenders.append(name)
    assert not offenders, f"verdict requested before reason in: {offenders}"
    print("  every rated prompt asks for reasoning first")


def test_scalar_prompts_use_the_discrete_scale():
    """A free 0..1 float is a task LLMs do badly; picking one of five labelled
    options is one they do well. Every scalar prompt must ask for an integer
    rating against a rubric line per point, and none may still ask for a float
    score — base.judge_score() maps the rating to 0..1 in Python."""
    from jobs.helper.judge_prompts import _PROMPTS

    rated = [
        "hallucination_no_context", "answer_relevancy", "toxicity",
        "coherence", "completeness", "trajectory_quality",
        "vision_faithfulness", "media_faithfulness",
        "tool_selection_quality", "agent_coordination",
    ]
    problems = []
    for name in rated:
        t = _PROMPTS[name]["template"]
        if "int 1-5" not in t:
            problems.append(f"{name}: no integer rating requested")
        if '\\"score\\": float' in t:
            problems.append(f"{name}: still asks for a float score")
        # every point of the scale must carry a written rubric line
        for point in ("1 = ", "2 = ", "3 = ", "4 = ", "5 = "):
            if point not in t:
                problems.append(f"{name}: missing rubric line for {point.strip(' =')}")
    assert not problems, "\n".join(problems)
    print(f"  {len(rated)} prompts rate 1-5 against a rubric line per point")


def test_likert_mapping_is_exact():
    from jobs.helper.base import likert_to_unit

    assert likert_to_unit(1) == 0.0
    assert likert_to_unit(2) == 0.25
    assert likert_to_unit(3) == 0.5
    assert likert_to_unit(4) == 0.75
    assert likert_to_unit(5) == 1.0
    # out-of-range and junk must clamp, not explode
    assert likert_to_unit(9) == 1.0
    assert likert_to_unit(0) == 0.0
    assert likert_to_unit("not a number", default=0.5) == 0.5
    print("  1-5 maps onto 0.00/0.25/0.50/0.75/1.00")


def test_legacy_float_score_still_read():
    """An org that overrode its prompt before the switch keeps that template
    forever — the seeder never rewrites an is_overridden row — so those judges
    still answer with a float. Dropping the fallback would silently zero them."""
    from jobs.helper.base import judge_score

    assert judge_score({"rating": 4}) == 0.75           # new scale wins
    assert judge_score({"score": 0.62}) == 0.62         # legacy override
    assert judge_score({"rating": 5, "score": 0.1}) == 1.0
    assert judge_score({}, default=0.5) == 0.5          # neither present
    print("  legacy float scores from overridden prompts still parse")


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
