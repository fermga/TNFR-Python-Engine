"""Cryptographic primitives for TNFR security.

This module provides secure hashing and signing utilities used across the
TNFR codebase, particularly for cache integrity verification.
"""

from __future__ import annotations

import hashlib
import hmac
from typing import Callable

__all__ = [
    "create_hmac_signer",
    "create_hmac_validator",
]

def create_hmac_signer(secret: bytes | str) -> Callable[[bytes], bytes]:
    """Create an HMAC-SHA256 signer for data integrity validation.

    Parameters
    ----------
    secret : bytes or str
        The secret key for HMAC signing. If str, it will be encoded as UTF-8.

    Returns
    -------
    callable
        A function that takes payload bytes and returns an HMAC signature.
    """
    secret_bytes = secret if isinstance(secret, bytes) else secret.encode("utf-8")

    def signer(payload: bytes) -> bytes:
        return hmac.new(secret_bytes, payload, hashlib.sha256).digest()

    return signer

def create_hmac_validator(secret: bytes | str) -> Callable[[bytes, bytes], bool]:
    """Create an HMAC-SHA256 validator for data integrity validation.

    Parameters
    ----------
    secret : bytes or str
        The secret key for HMAC validation. Must match the signer's secret.
        If str, it will be encoded as UTF-8.

    Returns
    -------
    callable
        A function that takes (payload_bytes, signature) and returns True
        if the signature is valid.
    """
    secret_bytes = secret if isinstance(secret, bytes) else secret.encode("utf-8")

    def validator(payload: bytes, signature: bytes) -> bool:
        expected = hmac.new(secret_bytes, payload, hashlib.sha256).digest()
        return hmac.compare_digest(expected, signature)

    return validator
