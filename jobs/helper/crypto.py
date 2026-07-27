"""Decrypt-only mirror of ``fluiq-api/shared/crypto.py``.

⚠️  KEEP IN SYNC with fluiq-api/shared/crypto.py. Sealing happens in the API;
this worker only ever unseals. The two files must agree exactly on three
things, or every credential written by the API becomes unreadable here:

    * ``KEY_VERSION``
    * the AAD string built by ``_aad`` (byte-for-byte)
    * the cipher and the wrapping scheme

``tests/test_credential_unseal.py`` pins all three. If you change the format on
either side, that test fails on this side — which is the point.

The evaluator is a separate service with its own image, venv, and config, the
same as ``config.py`` and ``db/`` already are, so this is a deliberate mirror
rather than a shared package: introducing one would mean a build and deploy
change across four services. The contract test is what keeps the copy honest.
"""
from __future__ import annotations

import base64
import hashlib
import logging
import os
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

# Must match fluiq-api/shared/crypto.py.
KEY_VERSION = 1
_NONCE_BYTES = 12
_DEK_BYTES = 32

_DEK_CACHE_TTL_SECONDS = 300.0
_DEK_CACHE_MAX = 512


class CredentialDecryptionUnavailable(RuntimeError):
    """No decryption backend configured — BYOK cannot be used on this worker."""


@dataclass(frozen=True)
class SealedSecret:
    ciphertext: bytes
    nonce: bytes
    wrapped_dek: bytes
    key_version: int = KEY_VERSION


def fingerprint(plaintext: str) -> str:
    """Must match the API's fingerprint(): sha256 hex, truncated to 16."""
    return hashlib.sha256(plaintext.encode()).hexdigest()[:16]


def _aad(org_id: str, key_version: int) -> bytes:
    """Must match the API's _aad() byte-for-byte."""
    return f"fluiq:credential:v{key_version}:{org_id}".encode()


def _aesgcm(key: bytes):
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    return AESGCM(key)


# ── DEK cache ────────────────────────────────────────────────────────────────

_dek_cache: dict[str, tuple[bytes, float]] = {}
_dek_lock = threading.Lock()


def _dek_cache_get(wrapped: bytes) -> Optional[bytes]:
    key = hashlib.sha256(wrapped).hexdigest()
    with _dek_lock:
        hit = _dek_cache.get(key)
        if hit is None:
            return None
        dek, expires_at = hit
        if time.monotonic() > expires_at:
            _dek_cache.pop(key, None)
            return None
        return dek


def _dek_cache_put(wrapped: bytes, dek: bytes) -> None:
    key = hashlib.sha256(wrapped).hexdigest()
    with _dek_lock:
        if len(_dek_cache) >= _DEK_CACHE_MAX:
            oldest = min(_dek_cache, key=lambda k: _dek_cache[k][1])
            _dek_cache.pop(oldest, None)
        _dek_cache[key] = (dek, time.monotonic() + _DEK_CACHE_TTL_SECONDS)


def clear_dek_cache() -> None:
    with _dek_lock:
        _dek_cache.clear()


# ── backends ─────────────────────────────────────────────────────────────────

def _backend() -> str:
    return (os.getenv("CREDENTIAL_ENCRYPTION_BACKEND") or "kms").lower()


@lru_cache(maxsize=1)
def _kms_client():
    import boto3
    return boto3.client("kms", region_name=os.getenv("AWS_REGION", "us-east-2"))


@lru_cache(maxsize=1)
def _local_key() -> bytes:
    raw = os.getenv("CREDENTIAL_ENCRYPTION_LOCAL_KEY")
    if not raw:
        raise CredentialDecryptionUnavailable(
            "CREDENTIAL_ENCRYPTION_BACKEND=local requires CREDENTIAL_ENCRYPTION_LOCAL_KEY"
        )
    key = base64.b64decode(raw, validate=True)
    if len(key) != _DEK_BYTES:
        raise CredentialDecryptionUnavailable(
            f"CREDENTIAL_ENCRYPTION_LOCAL_KEY must decode to {_DEK_BYTES} bytes"
        )
    return key


def is_configured() -> bool:
    try:
        if _backend() == "local":
            _local_key()
            return True
        return bool(os.getenv("CREDENTIAL_KMS_KEY_ID"))
    except Exception:
        return False


def _unwrap_dek(wrapped: bytes) -> bytes:
    cached = _dek_cache_get(wrapped)
    if cached is not None:
        return cached

    if _backend() == "local":
        nonce, body = wrapped[:_NONCE_BYTES], wrapped[_NONCE_BYTES:]
        dek = _aesgcm(_local_key()).decrypt(nonce, body, b"fluiq:dek")
    else:
        key_id = os.getenv("CREDENTIAL_KMS_KEY_ID")
        if not key_id:
            raise CredentialDecryptionUnavailable("CREDENTIAL_KMS_KEY_ID is not configured")
        dek = _kms_client().decrypt(CiphertextBlob=wrapped, KeyId=key_id)["Plaintext"]

    _dek_cache_put(wrapped, dek)
    return dek


def unseal(sealed: SealedSecret, *, org_id: str) -> str:
    """Recover a provider key. Raises on a tag mismatch — never returns garbage.

    Use the result to build a provider client and drop it. It must not be
    logged, persisted, attached to an eval result, or published to Kafka: the
    eval topic is PLAINTEXT on the wire.
    """
    dek = _unwrap_dek(sealed.wrapped_dek)
    plaintext = _aesgcm(dek).decrypt(
        sealed.nonce, sealed.ciphertext, _aad(str(org_id), sealed.key_version)
    )
    return plaintext.decode()
