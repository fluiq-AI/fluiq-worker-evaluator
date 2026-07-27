"""BYOK wiring: which provider key a judge is built on, and what happens when
the customer's key is unusable.

The load-bearing rule is that an unusable BYOK credential must *stop* the eval
rather than quietly falling back to Fluiq's managed key. Falling back would
bill Fluiq for usage sold as bring-your-own-key, and would hide the breakage
from the person who has to rotate the key.

No network or AWS. Run directly:

    ../.workers-venv/Scripts/python.exe tests/test_byok_judge_binding.py
"""
import base64
import os
import secrets
import sys

os.environ.setdefault("EVAL_JUDGE_THRESHOLD", "0.7")
os.environ.setdefault("EVAL_JUDGE_CACHE", "1")
os.environ.setdefault("EVAL_JUDGE_CACHE_TTL", "300")
os.environ.setdefault("EVAL_JUDGE_CACHE_MAX", "1000")
os.environ.setdefault("EVAL_JUDGE_PROVIDER", "openai")
os.environ.setdefault("EVAL_JUDGE_MODEL", "gpt-4o-mini")
os.environ.setdefault("CLICKHOUSE_PORT", "8123")
os.environ["CREDENTIAL_ENCRYPTION_BACKEND"] = "local"
os.environ["CREDENTIAL_ENCRYPTION_LOCAL_KEY"] = base64.b64encode(secrets.token_bytes(32)).decode()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jobs.helper import crypto  # noqa: E402
from jobs.run import BYOKUnavailable, OrgCredentials, _build_judge  # noqa: E402

ORG = "11111111-1111-1111-1111-111111111111"
CUSTOMER_KEY = "sk-customer-EXAMPLEnotarealkey0123456789"


def _row(provider: str, plaintext: str, org_id: str = ORG, status: str = "active") -> dict:
    """Build a credential row the way Postgres would hand it to us."""
    sealed = _seal(plaintext, org_id)
    return {
        "credential_id": "cred-1",
        "provider": provider,
        "status": status,
        "ciphertext": sealed[0],
        "nonce": sealed[1],
        "wrapped_dek": sealed[2],
        "key_version": crypto.KEY_VERSION,
    }


def _seal(plaintext: str, org_id: str):
    """Seal locally using the worker's own primitives (mirrors the API)."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    dek = secrets.token_bytes(32)
    dek_nonce = secrets.token_bytes(12)
    wrapped = dek_nonce + AESGCM(crypto._local_key()).encrypt(dek_nonce, dek, b"fluiq:dek")
    nonce = secrets.token_bytes(12)
    ct = AESGCM(dek).encrypt(
        nonce, plaintext.encode(), crypto._aad(org_id, crypto.KEY_VERSION)
    )
    return ct, nonce, wrapped


def test_no_credentials_uses_managed_key():
    creds = OrgCredentials(ORG, {})
    judge = _build_judge(org_id=ORG, creds=creds)
    assert judge._api_key is None, "an org without BYOK must fall through to managed keys"


def test_active_credential_is_bound_to_the_judge():
    creds = OrgCredentials(ORG, {"openai": _row("openai", CUSTOMER_KEY)})
    judge = _build_judge(org_id=ORG, creds=creds)
    assert judge._api_key == CUSTOMER_KEY


def test_credential_for_a_different_provider_is_not_used():
    """An Anthropic key must not be handed to an OpenAI judge."""
    creds = OrgCredentials(ORG, {"anthropic": _row("anthropic", CUSTOMER_KEY)})
    judge = _build_judge("openai", "gpt-4o-mini", org_id=ORG, creds=creds)
    assert judge._api_key is None


def test_invalid_credential_stops_the_eval_instead_of_falling_back():
    creds = OrgCredentials(
        ORG, {"openai": _row("openai", CUSTOMER_KEY, status="invalid")}
    )
    try:
        _build_judge(org_id=ORG, creds=creds)
    except BYOKUnavailable:
        return
    raise AssertionError(
        "an invalid customer key silently fell back to Fluiq's managed account"
    )


def test_undecryptable_credential_stops_the_eval():
    """A row sealed under a different org must not resolve to a managed key."""
    creds = OrgCredentials(ORG, {"openai": _row("openai", CUSTOMER_KEY, org_id="99999999-9999-9999-9999-999999999999")})
    try:
        _build_judge(org_id=ORG, creds=creds)
    except BYOKUnavailable:
        return
    raise AssertionError("an undecryptable credential did not stop the eval")


def test_byok_and_managed_do_not_share_a_cache_bucket():
    """A verdict paid for by the customer's key must not be served to a
    managed-key run of the same org, or vice versa.

    Asserted behaviorally — both judges are driven through the same shared
    cache and the provider dispatch is counted — rather than by inspecting the
    closure, so the test survives changes to how the judge is wrapped.
    """
    managed = _build_judge(org_id=ORG, creds=OrgCredentials(ORG, {}))
    byok = _build_judge(
        org_id=ORG, creds=OrgCredentials(ORG, {"openai": _row("openai", CUSTOMER_KEY)})
    )

    calls = []
    managed._call_openai = lambda p: (calls.append("managed"), '{"score": 0.1}')[1]
    byok._call_openai = lambda p: (calls.append("byok"), '{"score": 0.9}')[1]

    assert managed("same prompt") == '{"score": 0.1}'
    assert byok("same prompt") == '{"score": 0.9}', (
        "the BYOK judge was served the managed judge's cached verdict"
    )
    assert calls == ["managed", "byok"], "one run reused the other's cache entry"

    # And each still caches within its own bucket.
    managed("same prompt")
    assert calls == ["managed", "byok"], "caching broke for the managed judge"


def test_unresolved_credentials_are_reported():
    """A Postgres failure is 'unknown', not 'no credentials'."""
    assert OrgCredentials(ORG, None).resolved is False
    assert OrgCredentials(ORG, {}).resolved is True


def test_auth_errors_are_detected_conservatively():
    """A false positive disables a working customer key, so be strict."""
    from jobs.run import is_auth_error

    class AuthenticationError(Exception):
        pass

    class RateLimitError(Exception):
        status_code = 429

    class Boom(Exception):
        status_code = 401

    assert is_auth_error(AuthenticationError("bad key"))
    assert is_auth_error(Boom())
    assert not is_auth_error(RateLimitError()), "a 429 must not evict a good key"
    assert not is_auth_error(TimeoutError()), "a timeout must not evict a good key"
    assert not is_auth_error(ValueError("nope"))


def test_rejected_key_is_reported_once_per_message():
    """A jury of three must not write three invalidations for one key."""
    creds = OrgCredentials(ORG, {"openai": _row("openai", CUSTOMER_KEY)})
    # No loop attached: exercises the dedupe without touching Postgres.
    creds.note_auth_failure("openai", "rejected")
    creds.note_auth_failure("openai", "rejected")
    creds.note_auth_failure("openai", "rejected")
    assert creds._reported == {"openai"}, "a jury would have written three times"


def test_managed_key_failures_are_not_reported():
    """Fluiq's own key going bad is our problem, not a customer credential."""
    creds = OrgCredentials(ORG, {})
    creds.note_auth_failure("openai", "rejected")
    assert creds._reported == set(), "reported a failure for a key the org does not own"


def test_judge_spec_parsing():
    """Caller-supplied, so unusable input must degrade to the server default."""
    from jobs.run import parse_judge_spec

    assert parse_judge_spec("anthropic:claude-sonnet-5") == ("anthropic", "claude-sonnet-5")
    assert parse_judge_spec("  OpenAI : gpt-4o  ") == ("openai", "gpt-4o")
    # Anything unusable falls back rather than raising deep inside LLMJudge.
    assert parse_judge_spec(None) == (None, None)
    assert parse_judge_spec("") == (None, None)
    assert parse_judge_spec("no-colon") == (None, None)
    assert parse_judge_spec("anthropic:") == (None, None)
    assert parse_judge_spec("notaprovider:some-model") == (None, None)
    assert parse_judge_spec(42) == (None, None)


def test_jury_spec_parsing_drops_bad_entries():
    """A dropdown typo must not cost the customer a whole batch run."""
    from jobs.run import parse_jury_specs

    specs = parse_jury_specs([
        "anthropic:claude-haiku-4-5",
        "bogus:whatever",
        "openai:gpt-4o-mini",
        "",
    ])
    assert specs == [("anthropic", "claude-haiku-4-5"), ("openai", "gpt-4o-mini")]
    assert parse_jury_specs(None) == []
    assert parse_jury_specs([]) == []
    # Comma-separated string accepted too, matching the env-var form.
    assert parse_jury_specs("anthropic:a,openai:b") == [("anthropic", "a"), ("openai", "b")]


def test_selected_judge_is_used_over_the_server_default():
    creds = OrgCredentials(ORG, {})
    judge = _build_judge("anthropic", "claude-sonnet-5", org_id=ORG, creds=creds)
    assert judge.provider == "anthropic"
    assert judge.model == "claude-sonnet-5"


def test_selected_judge_uses_that_providers_byok_key():
    """Picking Anthropic must draw on the org's Anthropic key, not OpenAI's."""
    creds = OrgCredentials(ORG, {
        "anthropic": _row("anthropic", "sk-ant-customer"),
        "openai":    _row("openai", "sk-oai-customer"),
    })
    judge = _build_judge("anthropic", "claude-sonnet-5", org_id=ORG, creds=creds)
    assert judge._api_key == "sk-ant-customer"


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
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR {name}\n      {type(exc).__name__}: {exc}")
    print("\n" + ("all green" if not failures else f"{failures} failing"))
    sys.exit(1 if failures else 0)
