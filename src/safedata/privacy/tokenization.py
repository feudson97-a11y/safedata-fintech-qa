"""Keyed, irreversible surrogate keys (HMAC-SHA256)."""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
from typing import Final, Self

from safedata.domain.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

HMAC_KEY_ENV_VAR: Final = "SAFEDATA_HMAC_KEY"
_MIN_KEY_BYTES: Final = 16
_TOKEN_HEX_LENGTH: Final = 16


class Tokenizer:
    """
    Deterministic surrogate keys. A plain SHA-256 of a card number is *not*
    anonymisation: the PAN space is small enough to brute-force. Keying the hash
    with a secret held outside the dataset defeats dictionary attacks while
    keeping tokens stable for joins and distinct counts.
    """

    __slots__ = ("_key",)

    def __init__(self, key: bytes) -> None:
        if len(key) < _MIN_KEY_BYTES:
            raise ConfigurationError(f"HMAC key must be at least {_MIN_KEY_BYTES} bytes")
        self._key = key

    @property
    def fingerprint(self) -> str:
        """Non-reversible key identifier for audit logs; never the key itself."""
        return hashlib.sha256(self._key).hexdigest()[:12]

    def token(self, value: str | None, prefix: str) -> str:
        normalised = (value or "").strip().lower()
        if not normalised:
            return ""
        digest = hmac.new(self._key, f"{prefix}|{normalised}".encode(), hashlib.sha256).hexdigest()
        return f"{prefix}_{digest[:_TOKEN_HEX_LENGTH]}"

    @classmethod
    def from_environment(cls, env_var: str = HMAC_KEY_ENV_VAR) -> Self:
        if raw := os.environ.get(env_var):
            return cls(raw.encode())
        logger.warning("%s not set: using an ephemeral key; tokens will NOT be stable across runs.", env_var)
        return cls(secrets.token_bytes(32))
