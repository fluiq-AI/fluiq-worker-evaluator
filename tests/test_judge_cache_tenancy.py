"""Regression tests: the judge response cache must not cross tenants.

``_judge_cache_backend`` is a single process-wide ``InMemoryCache`` shared by
every eval a worker handles. Before the namespace fix, its key was
``sha256(provider:model + prompt + params)`` — so two orgs whose rendered judge
prompts collided shared a verdict, serving one org's judged content (the trace
is embedded in the prompt) to the other. Once each org pays for its own judge
calls, the same hole also bills org A for org B's eval.

No network or API keys required. Run directly:

    ../.workers-venv/Scripts/python.exe tests/test_judge_cache_tenancy.py
"""
import os
import sys

# config.py reads a few vars with float()/int() and will raise on import if they
# are unset. Seed the minimum before importing anything that pulls in config.
os.environ.setdefault("EVAL_JUDGE_THRESHOLD", "0.7")
os.environ.setdefault("EVAL_JUDGE_CACHE", "1")
os.environ.setdefault("EVAL_JUDGE_CACHE_TTL", "300")
os.environ.setdefault("EVAL_JUDGE_CACHE_MAX", "1000")
os.environ.setdefault("EVAL_JUDGE_PROVIDER", "openai")
os.environ.setdefault("EVAL_JUDGE_MODEL", "gpt-4o-mini")
os.environ.setdefault("CLICKHOUSE_PORT", "8123")

# Make the evaluator package root importable when run as a plain script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jobs.helper.judge import InMemoryCache, PromptCache

ORG_A = "11111111-1111-1111-1111-111111111111"
ORG_B = "22222222-2222-2222-2222-222222222222"

PROMPT = "Rate the trajectory of this agent run. Goal: refund the customer."


def _counting_fn(reply: str, counter: list):
    """A stand-in judge that records how many times it was actually invoked."""
    def _fn(prompt: str, **params) -> str:
        counter.append(prompt)
        return reply
    return _fn


def _cache_for(org: str, backend: InMemoryCache, reply: str, counter: list) -> PromptCache:
    return PromptCache(
        _counting_fn(reply, counter),
        model="openai:gpt-4o-mini",
        backend=backend,
        ttl=300,
        namespace=org,
    )


def test_same_prompt_different_orgs_do_not_share_a_verdict():
    """The core leak: identical prompt + model, two tenants, one shared backend."""
    backend = InMemoryCache(max_size=100)
    calls_a, calls_b = [], []

    a = _cache_for(ORG_A, backend, '{"score": 0.9}', calls_a)
    b = _cache_for(ORG_B, backend, '{"score": 0.1}', calls_b)

    assert a(PROMPT, temperature=0.0) == '{"score": 0.9}'
    assert b(PROMPT, temperature=0.0) == '{"score": 0.1}', (
        "org B was served org A's cached verdict — cross-tenant leak"
    )
    assert len(calls_a) == 1 and len(calls_b) == 1, (
        "each org must reach its own judge exactly once"
    )


def test_same_org_still_caches():
    """The namespace must not defeat caching within a tenant."""
    backend = InMemoryCache(max_size=100)
    calls = []
    a = _cache_for(ORG_A, backend, '{"score": 0.9}', calls)

    a(PROMPT, temperature=0.0)
    a(PROMPT, temperature=0.0)

    assert len(calls) == 1, f"expected 1 judge call, got {len(calls)}"


def test_a_param_named_model_cannot_shadow_the_real_model():
    """Judge params must not overwrite the model discriminator in the key.

    ``prompt`` is a named parameter of ``__call__`` so it can never arrive via
    ``**params``, but ``model`` can. When params were splatted alongside the
    key fields, a param named ``model`` overrode the judge's real model and two
    genuinely different judges collapsed onto one cache entry.
    """
    backend = InMemoryCache(max_size=100)
    calls_small, calls_large = [], []

    small = PromptCache(
        _counting_fn("cheap verdict", calls_small),
        model="openai:gpt-4o-mini", backend=backend, ttl=300, namespace=ORG_A,
    )
    large = PromptCache(
        _counting_fn("expensive verdict", calls_large),
        model="openai:gpt-5", backend=backend, ttl=300, namespace=ORG_A,
    )

    assert small(PROMPT, model="spoof") == "cheap verdict"
    assert large(PROMPT, model="spoof") == "expensive verdict", (
        "a param named 'model' overrode the real model and collapsed two judges "
        "onto one cache entry"
    )
    assert len(calls_small) == 1 and len(calls_large) == 1


def test_no_backend_is_passthrough():
    calls = []
    a = PromptCache(_counting_fn("x", calls), model="m", backend=None, namespace=ORG_A)
    a(PROMPT)
    a(PROMPT)
    assert len(calls) == 2, "a disabled cache must never memoize"


def test_build_judge_refuses_to_cache_without_an_org():
    """An unattributed message must not write into the shared cache."""
    from jobs.run import _build_judge

    unattributed = _build_judge(org_id=None)
    assert unattributed._judge_fn is None, (
        "judge built without an org must bypass the shared cache entirely"
    )

    attributed = _build_judge(org_id=ORG_A)
    assert attributed._judge_fn is not None, (
        "judge built with an org should still use the cache (EVAL_JUDGE_CACHE=1)"
    )


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {name}\n      {exc}")
        except Exception as exc:  # noqa: BLE001 - surface import/wiring breakage
            failures += 1
            print(f"ERROR {name}\n      {type(exc).__name__}: {exc}")
    print("\n" + ("all green" if not failures else f"{failures} failing"))
    sys.exit(1 if failures else 0)
