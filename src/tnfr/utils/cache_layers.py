"""Cache storage layers for TNFR.

This module provides the storage backend implementations for the cache system,
including in-memory, shelve-based, and Redis-based layers. It also handles
secure serialization with HMAC signing to prevent tampering.
"""

from __future__ import annotations

import os
import pickle
import shelve
import threading
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, MutableMapping
from typing import Any

from ..errors import TNFRValueError, TNFRSecurityError, TNFRSecurityWarning
from ..security.crypto import create_hmac_signer, create_hmac_validator

__all__ = (
    "CacheLayer",
    "MappingCacheLayer",
    "ShelveCacheLayer",
    "RedisCacheLayer",
    "create_secure_shelve_layer",
    "create_secure_redis_layer",
)

_SIGNATURE_PREFIX = b"TNFRSIG1"
_SIGN_MODE_RAW = 0
_SIGN_MODE_PICKLE = 1
_SIGNATURE_HEADER_SIZE = len(_SIGNATURE_PREFIX) + 1 + 4

# Environment variable to control security warnings for pickle deserialization
_TNFR_ALLOW_UNSIGNED_PICKLE = "TNFR_ALLOW_UNSIGNED_PICKLE"


def create_secure_shelve_layer(
    path: str,
    secret: bytes | str | None = None,
    *,
    flag: str = "c",
    protocol: int | None = None,
    writeback: bool = False,
) -> ShelveCacheLayer:
    """Create a ShelveCacheLayer with HMAC signature validation enabled.

    This is the recommended way to create persistent cache layers that handle
    TNFR structures (EPI, NFR, NetworkX graphs). Signature validation protects
    against arbitrary code execution from tampered pickle data.

    Parameters
    ----------
    path : str
        Path to the shelve database file.
    secret : bytes, str, or None
        Secret key for HMAC signing. If None, reads from TNFR_CACHE_SECRET
        environment variable. In production, **always** set this via environment.
    flag : str, default='c'
        Shelve open flag ('r', 'w', 'c', 'n').
    protocol : int, optional
        Pickle protocol version. Defaults to pickle.HIGHEST_PROTOCOL.
    writeback : bool, default=False
        Enable shelve writeback mode.

    Returns
    -------
    ShelveCacheLayer
        A cache layer with signature validation enabled.

    Raises
    ------
    ValueError
        If no secret is provided and TNFR_CACHE_SECRET is not set.

    Examples
    --------
    >>> # In production, set environment variable:
    >>> # export TNFR_CACHE_SECRET="your-secure-random-key"
    >>>
    >>> layer = create_secure_shelve_layer("coherence.db")
    >>> # Or explicitly provide secret:
    >>> layer = create_secure_shelve_layer("coherence.db", secret=b"my-secret")
    """
    if secret is None:
        secret = os.environ.get("TNFR_CACHE_SECRET")
        if not secret:
            raise TNFRValueError(
                "Secret required for secure cache layer.",
                context={"env_var": "TNFR_CACHE_SECRET"},
                suggestion="Set TNFR_CACHE_SECRET environment variable or pass secret parameter."
            )

    signer = create_hmac_signer(secret)
    validator = create_hmac_validator(secret)

    return ShelveCacheLayer(
        path,
        flag=flag,
        protocol=protocol,
        writeback=writeback,
        signer=signer,
        validator=validator,
        require_signature=True,
    )


def create_secure_redis_layer(
    client: Any | None = None,
    secret: bytes | str | None = None,
    *,
    namespace: str = "tnfr:cache",
    protocol: int | None = None,
) -> RedisCacheLayer:
    """Create a RedisCacheLayer with HMAC signature validation enabled.

    This is the recommended way to create distributed cache layers for TNFR.
    Signature validation protects against arbitrary code execution if Redis
    is compromised or contains tampered data.

    Parameters
    ----------
    client : redis.Redis, optional
        Redis client instance. If None, creates default client.
    secret : bytes, str, or None
        Secret key for HMAC signing. If None, reads from TNFR_CACHE_SECRET
        environment variable.
    namespace : str, default='tnfr:cache'
        Redis key namespace prefix.
    protocol : int, optional
        Pickle protocol version.

    Returns
    -------
    RedisCacheLayer
        A cache layer with signature validation enabled.

    Raises
    ------
    ValueError
        If no secret is provided and TNFR_CACHE_SECRET is not set.

    Examples
    --------
    >>> # Set environment variable in production:
    >>> # export TNFR_CACHE_SECRET="your-secure-random-key"
    >>>
    >>> layer = create_secure_redis_layer()
    >>> # Or with explicit configuration:
    >>> import redis
    >>> client = redis.Redis(host='localhost', port=6379)
    >>> layer = create_secure_redis_layer(client, secret=b"my-secret")
    """
    if secret is None:
        secret = os.environ.get("TNFR_CACHE_SECRET")
        if not secret:
            raise TNFRValueError(
                "Secret required for secure cache layer.",
                context={"env_var": "TNFR_CACHE_SECRET"},
                suggestion="Set TNFR_CACHE_SECRET environment variable or pass secret parameter."
            )

    signer = create_hmac_signer(secret)
    validator = create_hmac_validator(secret)

    return RedisCacheLayer(
        client=client,
        namespace=namespace,
        signer=signer,
        validator=validator,
        require_signature=True,
        protocol=protocol,
    )


def _prepare_payload_bytes(value: Any, *, protocol: int) -> tuple[int, bytes]:
    """Return payload encoding mode and the bytes that should be signed."""

    if isinstance(value, (bytes, bytearray, memoryview)):
        return _SIGN_MODE_RAW, bytes(value)
    return _SIGN_MODE_PICKLE, pickle.dumps(value, protocol=protocol)


def _pack_signed_envelope(mode: int, payload: bytes, signature: bytes) -> bytes:
    """Pack payload and signature into a self-describing binary envelope."""

    if not (0 <= mode <= 255):  # pragma: no cover - defensive guard
        raise TNFRValueError(
            f"invalid payload mode: {mode}",
            context={"mode": mode},
            suggestion="Mode must be between 0 and 255."
        )
    signature_length = len(signature)
    if signature_length >= 2**32:  # pragma: no cover - defensive guard
        raise TNFRValueError(
            "signature too large to encode",
            context={"signature_length": signature_length},
            suggestion="Signature must be smaller than 4GB."
        )
    header = (
        _SIGNATURE_PREFIX
        + bytes([mode])
        + signature_length.to_bytes(4, byteorder="big", signed=False)
    )
    return header + signature + payload


def _is_signed_envelope(blob: bytes) -> bool:
    """Return ``True`` when *blob* represents a signed cache entry."""

    return blob.startswith(_SIGNATURE_PREFIX)


def _unpack_signed_envelope(blob: bytes) -> tuple[int, bytes, bytes]:
    """Return the ``(mode, signature, payload)`` triple encoded in *blob*."""

    if len(blob) < _SIGNATURE_HEADER_SIZE:
        raise TNFRSecurityError("signed payload header truncated")
    if not _is_signed_envelope(blob):
        raise TNFRSecurityError("missing signed payload marker")
    mode = blob[len(_SIGNATURE_PREFIX)]
    sig_start = len(_SIGNATURE_PREFIX) + 1
    sig_len = int.from_bytes(blob[sig_start : sig_start + 4], byteorder="big")
    payload_start = sig_start + 4 + sig_len
    if len(blob) < payload_start:
        raise TNFRSecurityError("signed payload signature truncated")
    signature = blob[sig_start + 4 : payload_start]
    payload = blob[payload_start:]
    return mode, signature, payload


def _decode_payload(mode: int, payload: bytes) -> Any:
    """Decode payload bytes depending on cache encoding *mode*."""

    if mode == _SIGN_MODE_RAW:
        return payload
    if mode == _SIGN_MODE_PICKLE:
        return pickle.loads(payload)  # nosec B301 - validated via signature
    raise TNFRSecurityError(f"unknown payload encoding mode: {mode}")


class CacheLayer(ABC):
    """Abstract interface implemented by storage backends orchestrated by :class:`CacheManager`."""

    @abstractmethod
    def load(self, name: str) -> Any:
        """Return the stored payload for ``name`` or raise :class:`KeyError`."""

    @abstractmethod
    def store(self, name: str, value: Any) -> None:
        """Persist ``value`` under ``name``."""

    @abstractmethod
    def delete(self, name: str) -> None:
        """Remove ``name`` from the backend if present."""

    @abstractmethod
    def clear(self) -> None:
        """Remove every entry maintained by the layer."""

    def close(self) -> None:  # pragma: no cover - optional hook
        """Release resources held by the backend."""


class MappingCacheLayer(CacheLayer):
    """In-memory cache layer backed by a mutable mapping."""

    def __init__(self, storage: MutableMapping[str, Any] | None = None) -> None:
        self._storage: MutableMapping[str, Any] = {} if storage is None else storage
        self._lock = threading.RLock()

    @property
    def storage(self) -> MutableMapping[str, Any]:
        """Return the mapping used to store cache entries."""

        return self._storage

    def load(self, name: str) -> Any:
        with self._lock:
            if name not in self._storage:
                raise KeyError(name)
            return self._storage[name]

    def store(self, name: str, value: Any) -> None:
        with self._lock:
            self._storage[name] = value

    def delete(self, name: str) -> None:
        with self._lock:
            self._storage.pop(name, None)

    def clear(self) -> None:
        with self._lock:
            self._storage.clear()


class ShelveCacheLayer(CacheLayer):
    """Persistent cache layer backed by :mod:`shelve`.

    .. warning::
        This layer uses :mod:`pickle` for serialization, which can deserialize
        arbitrary Python objects and execute code during deserialization.
        **Only use with trusted data** from controlled sources. Never load
        shelf files from untrusted origins without cryptographic verification.

        Pickle is required for TNFR's complex structures (NetworkX graphs, EPIs,
        coherence states, numpy arrays). For untrusted inputs, enable
        :term:`HMAC` or equivalent signing via ``signer``/``validator`` and
        set ``require_signature=True`` to reject tampered payloads.

    :param signer: Optional callable that receives payload bytes and returns a
        signature (for example ``lambda payload: hmac.new(key, payload,
        hashlib.sha256).digest()``).
    :param validator: Optional callable that receives ``(payload_bytes,
        signature)`` and returns ``True`` when the payload is trustworthy.
    :param require_signature: When ``True`` the cache operates in hardened
        mode, deleting entries whose signatures are missing or invalid and
        raising :class:`SecurityError`.
    """

    def __init__(
        self,
        path: str,
        *,
        flag: str = "c",
        protocol: int | None = None,
        writeback: bool = False,
        signer: Callable[[bytes], bytes] | None = None,
        validator: Callable[[bytes, bytes], bool] | None = None,
        require_signature: bool = False,
    ) -> None:
        # Validate cache file path to prevent path traversal
        from ..security import validate_file_path, PathTraversalError

        try:
            validated_path = validate_file_path(
                path,
                allow_absolute=True,
                allowed_extensions=None,  # Shelve creates multiple files with various extensions
            )
            self._path = str(validated_path)
        except (ValueError, PathTraversalError) as e:
            raise TNFRValueError(
                f"Invalid cache path {path!r}: {e}",
                context={"path": path, "error": str(e)},
            ) from e

        self._flag = flag
        self._protocol = pickle.HIGHEST_PROTOCOL if protocol is None else protocol
        # shelve module inherently uses pickle for serialization; security risks documented in class docstring
        self._shelf = shelve.open(
            self._path, flag=flag, protocol=self._protocol, writeback=writeback  # type: ignore
        )  # nosec B301
        self._lock = threading.RLock()
        self._signer = signer
        self._validator = validator
        self._require_signature = require_signature
        if require_signature and (signer is None or validator is None):
            raise TNFRValueError(
                "require_signature=True requires both signer and validator",
                context={"signer": signer, "validator": validator},
            )

        # Issue security warning when using unsigned pickle deserialization
        if not require_signature and os.environ.get(_TNFR_ALLOW_UNSIGNED_PICKLE) != "1":
            warnings.warn(
                f"ShelveCacheLayer at {path!r} uses pickle without signature validation. "
                "This can execute arbitrary code during deserialization. "
                "Use create_secure_shelve_layer() or set require_signature=True with signer/validator. "
                f"To suppress this warning, set {_TNFR_ALLOW_UNSIGNED_PICKLE}=1 environment variable.",
                TNFRSecurityWarning,
                stacklevel=2,
            )

    def load(self, name: str) -> Any:
        with self._lock:
            if name not in self._shelf:
                raise KeyError(name)
            entry = self._shelf[name]

        return self._decode_entry(name, entry)

    def store(self, name: str, value: Any) -> None:
        if self._signer is None:
            stored_value: Any = value
        else:
            mode, payload = _prepare_payload_bytes(value, protocol=self._protocol)
            signature = self._signer(payload)
            stored_value = _pack_signed_envelope(mode, payload, signature)
        with self._lock:
            self._shelf[name] = stored_value
            self._shelf.sync()

    def delete(self, name: str) -> None:
        with self._lock:
            try:
                del self._shelf[name]
            except KeyError:
                return
            self._shelf.sync()

    def clear(self) -> None:
        with self._lock:
            self._shelf.clear()
            self._shelf.sync()

    def close(self) -> None:  # pragma: no cover - exercised indirectly
        with self._lock:
            self._shelf.close()

    def _decode_entry(self, name: str, entry: Any) -> Any:
        if isinstance(entry, (bytes, bytearray, memoryview)):
            blob = bytes(entry)
            if _is_signed_envelope(blob):
                try:
                    mode, signature, payload = _unpack_signed_envelope(blob)
                except TNFRSecurityError:
                    self.delete(name)
                    raise
                validator = self._validator
                if validator is None:
                    if self._require_signature:
                        self.delete(name)
                        raise TNFRSecurityError(
                            "signature validation requested but no validator configured"
                        )
                else:
                    try:
                        valid = validator(payload, signature)
                    except Exception as exc:  # pragma: no cover - defensive
                        self.delete(name)
                        raise TNFRSecurityError("signature validator raised an exception") from exc
                    if not valid:
                        self.delete(name)
                        raise TNFRSecurityError(f"signature validation failed for cache entry {name!r}")
                try:
                    return _decode_payload(mode, payload)
                except Exception as exc:
                    self.delete(name)
                    raise TNFRSecurityError("signed payload decode failure") from exc
            if self._require_signature:
                self.delete(name)
                raise TNFRSecurityError(f"unsigned cache entry rejected: {name}")
            return blob
        if self._require_signature:
            self.delete(name)
            raise TNFRSecurityError(f"unsigned cache entry rejected: {name}")
        return entry


class RedisCacheLayer(CacheLayer):
    """Distributed cache layer backed by a Redis client.

    .. warning::
        This layer uses :mod:`pickle` for serialization, which can deserialize
        arbitrary Python objects and execute code during deserialization.
        **Only cache trusted data** from controlled TNFR nodes. Ensure Redis
        uses authentication (AUTH command or ACL for Redis 6.0+) and network
        access controls. Never cache untrusted user input or external data.

        If Redis is compromised or contains tampered data, pickle deserialization
        executes arbitrary code. Use TLS for connections and enable signature
        validation (``signer``/``validator`` with ``require_signature=True``)
        in high-assurance deployments.

    :param signer: Optional callable that produces a signature for payload bytes
        before they are written to Redis.
    :param validator: Optional callable that validates ``(payload_bytes,
        signature)`` during loads.
    :param require_signature: Enable hardened mode that deletes and rejects
        cache entries whose signatures are missing or invalid, raising
        :class:`SecurityError`.
    """

    def __init__(
        self,
        client: Any | None = None,
        *,
        namespace: str = "tnfr:cache",
        signer: Callable[[bytes], bytes] | None = None,
        validator: Callable[[bytes, bytes], bool] | None = None,
        require_signature: bool = False,
        protocol: int | None = None,
    ) -> None:
        if client is None:
            try:  # pragma: no cover - import guarded for optional dependency
                import redis  # type: ignore
            except Exception as exc:  # pragma: no cover - defensive import
                raise RuntimeError("redis-py is required to initialise RedisCacheLayer") from exc
            client = redis.Redis()
        self._client = client
        self._namespace = namespace.rstrip(":") or "tnfr:cache"
        self._lock = threading.RLock()
        self._signer = signer
        self._validator = validator
        self._require_signature = require_signature
        self._protocol = pickle.HIGHEST_PROTOCOL if protocol is None else protocol
        if require_signature and (signer is None or validator is None):
            raise TNFRValueError(
                "require_signature=True requires both signer and validator",
                context={"signer": signer, "validator": validator},
            )

        # Issue security warning when using unsigned pickle deserialization
        if not require_signature and os.environ.get(_TNFR_ALLOW_UNSIGNED_PICKLE) != "1":
            warnings.warn(
                f"RedisCacheLayer with namespace {namespace!r} uses pickle without signature validation. "
                "This can execute arbitrary code if Redis is compromised. "
                "Use create_secure_redis_layer() or set require_signature=True with signer/validator. "
                f"To suppress this warning, set {_TNFR_ALLOW_UNSIGNED_PICKLE}=1 environment variable.",
                TNFRSecurityWarning,
                stacklevel=2,
            )

    def _format_key(self, name: str) -> str:
        return f"{self._namespace}:{name}"

    def load(self, name: str) -> Any:
        key = self._format_key(name)
        with self._lock:
            value = self._client.get(key)
        if value is None:
            raise KeyError(name)
        if isinstance(value, (bytes, bytearray, memoryview)):
            blob = bytes(value)
            if _is_signed_envelope(blob):
                try:
                    mode, signature, payload = _unpack_signed_envelope(blob)
                except TNFRSecurityError:
                    self.delete(name)
                    raise
                validator = self._validator
                if validator is None:
                    if self._require_signature:
                        self.delete(name)
                        raise TNFRSecurityError(
                            "signature validation requested but no validator configured"
                        )
                else:
                    try:
                        valid = validator(payload, signature)
                    except Exception as exc:  # pragma: no cover - defensive
                        self.delete(name)
                        raise TNFRSecurityError("signature validator raised an exception") from exc
                    if not valid:
                        self.delete(name)
                        raise TNFRSecurityError(f"signature validation failed for cache entry {name!r}")
                try:
                    return _decode_payload(mode, payload)
                except Exception as exc:
                    self.delete(name)
                    raise TNFRSecurityError("signed payload decode failure") from exc
            if self._require_signature:
                self.delete(name)
                raise TNFRSecurityError(f"unsigned cache entry rejected: {name}")
            # pickle from trusted Redis; documented security warning in class docstring
            return pickle.loads(blob)  # nosec B301
        return value

    def store(self, name: str, value: Any) -> None:
        key = self._format_key(name)
        if self._signer is None:
            payload: Any = value
            if not isinstance(value, (bytes, bytearray, memoryview)):
                payload = pickle.dumps(value, protocol=self._protocol)
        else:
            mode, payload_bytes = _prepare_payload_bytes(value, protocol=self._protocol)
            signature = self._signer(payload_bytes)
            payload = _pack_signed_envelope(mode, payload_bytes, signature)
        with self._lock:
            self._client.set(key, payload)

    def delete(self, name: str) -> None:
        key = self._format_key(name)
        with self._lock:
            self._client.delete(key)

    def clear(self) -> None:
        pattern = f"{self._namespace}:*"
        with self._lock:
            if hasattr(self._client, "scan_iter"):
                keys = list(self._client.scan_iter(match=pattern))
            elif hasattr(self._client, "keys"):
                keys = list(self._client.keys(pattern))
            else:  # pragma: no cover - extremely defensive
                keys = []
            if keys:
                self._client.delete(*keys)
