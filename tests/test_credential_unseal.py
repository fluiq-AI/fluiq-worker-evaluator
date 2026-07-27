"""Contract test: the evaluator must be able to unseal what the API sealed.

``jobs/helper/crypto.py`` is a deliberate mirror of
``fluiq-api/shared/crypto.py`` — the evaluator is a separate service with its
own image and venv, so there is no shared package to import. The risk of a
mirror is silent drift: change the AAD string or bump KEY_VERSION on one side
and every customer credential becomes undecryptable on the other, at runtime,
in production.

These tests pin the wire format so that drift fails here instead. The first
test imports *both* modules and round-trips a secret across them, which is the
real contract; the rest pin the individual constants so a failure says which
part moved.

No network or AWS. Run directly:

    ../.workers-venv/Scripts/python.exe tests/test_credential_unseal.py
"""
import base64
import os
import secrets
import sys

os.environ.setdefault("EVAL_JUDGE_THRESHOLD", "0.7")
os.environ.setdefault("EVAL_JUDGE_CACHE_TTL", "300")
os.environ.setdefault("EVAL_JUDGE_CACHE_MAX", "1000")
os.environ.setdefault("CLICKHOUSE_PORT", "8123")
os.environ["CREDENTIAL_ENCRYPTION_BACKEND"] = "local"
os.environ["CREDENTIAL_ENCRYPTION_LOCAL_KEY"] = base64.b64encode(secrets.token_bytes(32)).decode()

_HERE = os.path.dirname(os.path.abspath(__file__))
_EVALUATOR = os.path.dirname(_HERE)
_SOURCE = os.path.dirname(os.path.dirname(_EVALUATOR))
sys.path.insert(0, _EVALUATOR)

from jobs.helper import crypto as worker_crypto  # noqa: E402

ORG_A = "11111111-1111-1111-1111-111111111111"
ORG_B = "22222222-2222-2222-2222-222222222222"
KEY = "sk-proj-EXAMPLEnotarealkey0123456789abcdefXYZ"


def _load_api_crypto():
    """Import the API's crypto module by path, without importing its config."""
    import importlib.util
    path = os.path.join(_SOURCE, "fluiq-api", "shared", "crypto.py")
    if not os.path.isfile(path):
        return None
    # The API module does `import config`; satisfy it with a stub carrying only
    # the attributes crypto.py reads, so this test needs no API env at all.
    import types
    stub = types.ModuleType("config")
    stub.AWS_REGION = "us-east-2"
    stub.CREDENTIAL_ENCRYPTION_BACKEND = "local"
    stub.CREDENTIAL_ENCRYPTION_LOCAL_KEY = os.environ["CREDENTIAL_ENCRYPTION_LOCAL_KEY"]
    stub.CREDENTIAL_KMS_KEY_ID = None
    saved = sys.modules.get("config")
    sys.modules["config"] = stub
    try:
        spec = importlib.util.spec_from_file_location("_api_crypto", path)
        mod = importlib.util.module_from_spec(spec)
        # Must be in sys.modules *before* exec: @dataclass resolves the defining
        # module through sys.modules[cls.__module__] while the class body runs.
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        if saved is not None:
            sys.modules["config"] = saved
        else:
            sys.modules.pop("config", None)


def test_worker_unseals_what_the_api_sealed():
    """The contract. If this fails, BYOK is broken in production."""
    api = _load_api_crypto()
    if api is None:
        print("      (skipped: fluiq-api not present alongside the worker)")
        return

    sealed = api.seal(KEY, org_id=ORG_A)
    mirrored = worker_crypto.SealedSecret(
        ciphertext=sealed.ciphertext,
        nonce=sealed.nonce,
        wrapped_dek=sealed.wrapped_dek,
        key_version=sealed.key_version,
    )
    assert worker_crypto.unseal(mirrored, org_id=ORG_A) == KEY, (
        "the worker could not decrypt an API-sealed credential — the mirror has drifted"
    )


def test_org_binding_still_holds_across_the_boundary():
    from cryptography.exceptions import InvalidTag

    api = _load_api_crypto()
    if api is None:
        return

    sealed = api.seal(KEY, org_id=ORG_A)
    mirrored = worker_crypto.SealedSecret(
        ciphertext=sealed.ciphertext, nonce=sealed.nonce,
        wrapped_dek=sealed.wrapped_dek, key_version=sealed.key_version,
    )
    try:
        worker_crypto.unseal(mirrored, org_id=ORG_B)
    except InvalidTag:
        return
    raise AssertionError("worker decrypted org A's credential under org B")


def test_aad_format_is_pinned():
    """Pinned literal — changing it invalidates every stored credential."""
    assert worker_crypto._aad("org-123", 1) == b"fluiq:credential:v1:org-123"


def test_key_version_is_pinned():
    assert worker_crypto.KEY_VERSION == 1


def test_fingerprint_matches_the_api():
    api = _load_api_crypto()
    if api is None:
        return
    assert worker_crypto.fingerprint(KEY) == api.fingerprint(KEY)


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
