"""Cache infrastructure primitives and graph-level helpers for TNFR.

This module consolidates structural cache helpers that previously lived in
legacy helper modules and are now exposed under :mod:`tnfr.utils`. The
functions exposed here are responsible for maintaining deterministic node
digests, scoped graph caches guarded by locks, and version counters that keep
edge artifacts in sync with ΔNFR driven updates.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
import hmac
import logging
import os
import pickle
import shelve
import threading
import warnings
from collections import defaultdict
from collections.abc import (
    Callable,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
)
from contextlib import contextmanager
from dataclasses import field
from enum import Enum
from functools import lru_cache, wraps
import sys
import time
from time import perf_counter
from typing import Any, Generic, Optional, TypeVar, cast

import networkx as nx
from cachetools import LRUCache

from ..compat.dataclass import dataclass

from ..locking import get_lock
from ..types import GraphLike, NodeId, TimingContext, TNFRGraph
from .graph import get_graph, mark_dnfr_prep_dirty

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")
T = TypeVar("T")


class SecurityError(RuntimeError):
    """Raised when a cache payload fails hardened validation."""


class SecurityWarning(UserWarning):
    """Issued when potentially unsafe serialization is used without signing."""


__all__ = (
    "CacheLayer",
    "CacheManager",
    "CacheCapacityConfig",
    "CacheStatistics",
    "InstrumentedLRUCache",
    "ManagedLRUCache",
    "MappingCacheLayer",
    "RedisCacheLayer",
    "ShelveCacheLayer",
    "SecurityError",
    "SecurityWarning",
    "create_hmac_signer",
    "create_hmac_validator",
    "create_secure_shelve_layer",
    "create_secure_redis_layer",
    "prune_lock_mapping",
    "EdgeCacheManager",
    "NODE_SET_CHECKSUM_KEY",
    "cached_node_list",
    "cached_nodes_and_A",
    "clear_node_repr_cache",
    "edge_version_cache",
    "edge_version_update",
    "ensure_node_index_map",
    "ensure_node_offset_map",
    "get_graph_version",
    "increment_edge_version",
    "increment_graph_version",
    "node_set_checksum",
    "stable_json",
    "configure_graph_cache_limits",
    "DNFR_PREP_STATE_KEY",
    "DnfrPrepState",
    "build_cache_manager",
    "configure_global_cache_layers",
    "reset_global_cache_manager",
    "_GRAPH_CACHE_LAYERS_KEY",
    "_SeedHashCache",
    "ScopedCounterCache",
    "DnfrCache",
    "new_dnfr_cache",
    # Hierarchical cache classes (moved from caching/)
    "CacheLevel",
    "CacheEntry",
    "TNFRHierarchicalCache",
    # Cache decorators (moved from caching/decorators.py)
    "cache_tnfr_computation",
    "invalidate_function_cache",
    "get_global_cache",
    "set_global_cache",
    "reset_global_cache",
    # Invalidation tracking (moved from caching/invalidation.py)
    "GraphChangeTracker",
    "track_node_property_update",
    # Persistence (moved from caching/persistence.py)
    "PersistentTNFRCache",
)

_SIGNATURE_PREFIX = b"TNFRSIG1"
_SIGN_MODE_RAW = 0
_SIGN_MODE_PICKLE = 1
_SIGNATURE_HEADER_SIZE = len(_SIGNATURE_PREFIX) + 1 + 4

# Environment variable to control security warnings for pickle deserialization
_TNFR_ALLOW_UNSIGNED_PICKLE = "TNFR_ALLOW_UNSIGNED_PICKLE"


def create_hmac_signer(secret: bytes | str) -> Callable[[bytes], bytes]:
    """Create an HMAC-SHA256 signer for cache layer signature validation.

    Parameters
    ----------
    secret : bytes or str
        The secret key for HMAC signing. If str, it will be encoded as UTF-8.

    Returns
    -------
    callable
        A function that takes payload bytes and returns an HMAC signature.

    Examples
    --------
    >>> import os
    >>> secret = os.environ.get("TNFR_CACHE_SECRET", "dev-secret-key")
    >>> signer = create_hmac_signer(secret)
    >>> validator = create_hmac_validator(secret)
    >>> layer = ShelveCacheLayer(
    ...     "cache.db",
    ...     signer=signer,
    ...     validator=validator,
    ...     require_signature=True
    ... )
    """
    secret_bytes = secret if isinstance(secret, bytes) else secret.encode("utf-8")

    def signer(payload: bytes) -> bytes:
        return hmac.new(secret_bytes, payload, hashlib.sha256).digest()

    return signer


def create_hmac_validator(secret: bytes | str) -> Callable[[bytes, bytes], bool]:
    """Create an HMAC-SHA256 validator for cache layer signature validation.

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

    See Also
    --------
    create_hmac_signer : Create the corresponding signer.
    """
    secret_bytes = secret if isinstance(secret, bytes) else secret.encode("utf-8")

    def validator(payload: bytes, signature: bytes) -> bool:
        expected = hmac.new(secret_bytes, payload, hashlib.sha256).digest()
        return hmac.compare_digest(expected, signature)

    return validator


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
            raise ValueError(
                "Secret required for secure cache layer. "
                "Set TNFR_CACHE_SECRET environment variable or pass secret parameter."
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
            raise ValueError(
                "Secret required for secure cache layer. "
                "Set TNFR_CACHE_SECRET environment variable or pass secret parameter."
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
        raise ValueError(f"invalid payload mode: {mode}")
    signature_length = len(signature)
    if signature_length >= 2**32:  # pragma: no cover - defensive guard
        raise ValueError("signature too large to encode")
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
        raise SecurityError("signed payload header truncated")
    if not _is_signed_envelope(blob):
        raise SecurityError("missing signed payload marker")
    mode = blob[len(_SIGNATURE_PREFIX)]
    sig_start = len(_SIGNATURE_PREFIX) + 1
    sig_len = int.from_bytes(blob[sig_start : sig_start + 4], byteorder="big")
    payload_start = sig_start + 4 + sig_len
    if len(blob) < payload_start:
        raise SecurityError("signed payload signature truncated")
    signature = blob[sig_start + 4 : payload_start]
    payload = blob[payload_start:]
    return mode, signature, payload


def _decode_payload(mode: int, payload: bytes) -> Any:
    """Decode payload bytes depending on cache encoding *mode*."""

    if mode == _SIGN_MODE_RAW:
        return payload
    if mode == _SIGN_MODE_PICKLE:
        return pickle.loads(payload)  # nosec B301 - validated via signature
    raise SecurityError(f"unknown payload encoding mode: {mode}")


@dataclass(frozen=True)
class CacheCapacityConfig:
    """Configuration snapshot for cache capacity policies."""

    default_capacity: int | None
    overrides: dict[str, int | None]


@dataclass(frozen=True)
class CacheStatistics:
    """Immutable snapshot of cache telemetry counters."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_time: float = 0.0
    timings: int = 0

    def merge(self, other: CacheStatistics) -> CacheStatistics:
        """Return aggregated metrics combining ``self`` and ``other``."""

        return CacheStatistics(
            hits=self.hits + other.hits,
            misses=self.misses + other.misses,
            evictions=self.evictions + other.evictions,
            total_time=self.total_time + other.total_time,
            timings=self.timings + other.timings,
        )


@dataclass
class DnfrCache:
    idx: dict[Any, int]
    theta: list[float]
    epi: list[float]
    vf: list[float]
    cos_theta: list[float]
    sin_theta: list[float]
    neighbor_x: list[float]
    neighbor_y: list[float]
    neighbor_epi_sum: list[float]
    neighbor_vf_sum: list[float]
    neighbor_count: list[float]
    neighbor_deg_sum: list[float] | None
    th_bar: list[float] | None = None
    epi_bar: list[float] | None = None
    vf_bar: list[float] | None = None
    deg_bar: list[float] | None = None
    degs: dict[Any, float] | None = None
    deg_list: list[float] | None = None
    theta_np: Any | None = None
    epi_np: Any | None = None
    vf_np: Any | None = None
    cos_theta_np: Any | None = None
    sin_theta_np: Any | None = None
    deg_array: Any | None = None
    edge_src: Any | None = None
    edge_dst: Any | None = None
    checksum: Any | None = None
    neighbor_x_np: Any | None = None
    neighbor_y_np: Any | None = None
    neighbor_epi_sum_np: Any | None = None
    neighbor_vf_sum_np: Any | None = None
    neighbor_count_np: Any | None = None
    neighbor_deg_sum_np: Any | None = None
    th_bar_np: Any | None = None
    epi_bar_np: Any | None = None
    vf_bar_np: Any | None = None
    deg_bar_np: Any | None = None
    grad_phase_np: Any | None = None
    grad_epi_np: Any | None = None
    grad_vf_np: Any | None = None
    grad_topo_np: Any | None = None
    grad_total_np: Any | None = None
    dense_components_np: Any | None = None
    dense_accum_np: Any | None = None
    dense_degree_np: Any | None = None
    neighbor_accum_np: Any | None = None
    neighbor_inv_count_np: Any | None = None
    neighbor_cos_avg_np: Any | None = None
    neighbor_sin_avg_np: Any | None = None
    neighbor_mean_tmp_np: Any | None = None
    neighbor_mean_length_np: Any | None = None
    edge_signature: Any | None = None
    neighbor_accum_signature: Any | None = None
    neighbor_edge_values_np: Any | None = None


def new_dnfr_cache() -> DnfrCache:
    """Return an empty :class:`DnfrCache` prepared for ΔNFR orchestration."""

    return DnfrCache(
        idx={},
        theta=[],
        epi=[],
        vf=[],
        cos_theta=[],
        sin_theta=[],
        neighbor_x=[],
        neighbor_y=[],
        neighbor_epi_sum=[],
        neighbor_vf_sum=[],
        neighbor_count=[],
        neighbor_deg_sum=[],
    )


@dataclass
class _CacheMetrics:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_time: float = 0.0
    timings: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def snapshot(self) -> CacheStatistics:
        return CacheStatistics(
            hits=self.hits,
            misses=self.misses,
            evictions=self.evictions,
            total_time=self.total_time,
            timings=self.timings,
        )


@dataclass
class _CacheEntry:
    factory: Callable[[], Any]
    lock: threading.Lock
    reset: Callable[[Any], Any] | None = None
    encoder: Callable[[Any], Any] | None = None
    decoder: Callable[[Any], Any] | None = None


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
            raise ValueError(f"Invalid cache path {path!r}: {e}") from e

        self._flag = flag
        self._protocol = pickle.HIGHEST_PROTOCOL if protocol is None else protocol
        # shelve module inherently uses pickle for serialization; security risks documented in class docstring
        self._shelf = shelve.open(
            self._path, flag=flag, protocol=self._protocol, writeback=writeback
        )  # nosec B301
        self._lock = threading.RLock()
        self._signer = signer
        self._validator = validator
        self._require_signature = require_signature
        if require_signature and (signer is None or validator is None):
            raise ValueError("require_signature=True requires both signer and validator")

        # Issue security warning when using unsigned pickle deserialization
        if not require_signature and os.environ.get(_TNFR_ALLOW_UNSIGNED_PICKLE) != "1":
            warnings.warn(
                f"ShelveCacheLayer at {path!r} uses pickle without signature validation. "
                "This can execute arbitrary code during deserialization. "
                "Use create_secure_shelve_layer() or set require_signature=True with signer/validator. "
                f"To suppress this warning, set {_TNFR_ALLOW_UNSIGNED_PICKLE}=1 environment variable.",
                SecurityWarning,
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
                except SecurityError:
                    self.delete(name)
                    raise
                validator = self._validator
                if validator is None:
                    if self._require_signature:
                        self.delete(name)
                        raise SecurityError(
                            "signature validation requested but no validator configured"
                        )
                else:
                    try:
                        valid = validator(payload, signature)
                    except Exception as exc:  # pragma: no cover - defensive
                        self.delete(name)
                        raise SecurityError("signature validator raised an exception") from exc
                    if not valid:
                        self.delete(name)
                        raise SecurityError(f"signature validation failed for cache entry {name!r}")
                try:
                    return _decode_payload(mode, payload)
                except Exception as exc:
                    self.delete(name)
                    raise SecurityError("signed payload decode failure") from exc
            if self._require_signature:
                self.delete(name)
                raise SecurityError(f"unsigned cache entry rejected: {name}")
            return blob
        if self._require_signature:
            self.delete(name)
            raise SecurityError(f"unsigned cache entry rejected: {name}")
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
            raise ValueError("require_signature=True requires both signer and validator")

        # Issue security warning when using unsigned pickle deserialization
        if not require_signature and os.environ.get(_TNFR_ALLOW_UNSIGNED_PICKLE) != "1":
            warnings.warn(
                f"RedisCacheLayer with namespace {namespace!r} uses pickle without signature validation. "
                "This can execute arbitrary code if Redis is compromised. "
                "Use create_secure_redis_layer() or set require_signature=True with signer/validator. "
                f"To suppress this warning, set {_TNFR_ALLOW_UNSIGNED_PICKLE}=1 environment variable.",
                SecurityWarning,
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
                except SecurityError:
                    self.delete(name)
                    raise
                validator = self._validator
                if validator is None:
                    if self._require_signature:
                        self.delete(name)
                        raise SecurityError(
                            "signature validation requested but no validator configured"
                        )
                else:
                    try:
                        valid = validator(payload, signature)
                    except Exception as exc:  # pragma: no cover - defensive
                        self.delete(name)
                        raise SecurityError("signature validator raised an exception") from exc
                    if not valid:
                        self.delete(name)
                        raise SecurityError(f"signature validation failed for cache entry {name!r}")
                try:
                    return _decode_payload(mode, payload)
                except Exception as exc:
                    self.delete(name)
                    raise SecurityError("signed payload decode failure") from exc
            if self._require_signature:
                self.delete(name)
                raise SecurityError(f"unsigned cache entry rejected: {name}")
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


class CacheManager:
    """Coordinate named caches guarded by per-entry locks."""

    _MISSING = object()

    def __init__(
        self,
        storage: MutableMapping[str, Any] | None = None,
        *,
        default_capacity: int | None = None,
        overrides: Mapping[str, int | None] | None = None,
        layers: Iterable[CacheLayer] | None = None,
    ) -> None:
        mapping_layer = MappingCacheLayer(storage)
        extra_layers: tuple[CacheLayer, ...]
        if layers is None:
            extra_layers = ()
        else:
            extra_layers = tuple(layers)
            for layer in extra_layers:
                if not isinstance(layer, CacheLayer):  # pragma: no cover - defensive typing
                    raise TypeError(f"unsupported cache layer type: {type(layer)!r}")
        self._layers: tuple[CacheLayer, ...] = (mapping_layer, *extra_layers)
        self._storage_layer = mapping_layer
        self._storage: MutableMapping[str, Any] = mapping_layer.storage
        self._entries: dict[str, _CacheEntry] = {}
        self._registry_lock = threading.RLock()
        self._default_capacity = self._normalise_capacity(default_capacity)
        self._capacity_overrides: dict[str, int | None] = {}
        self._metrics: dict[str, _CacheMetrics] = {}
        self._metrics_publishers: list[Callable[[str, CacheStatistics], None]] = []
        if overrides:
            self.configure(overrides=overrides)

    @staticmethod
    def _normalise_capacity(value: int | None) -> int | None:
        if value is None:
            return None
        size = int(value)
        if size < 0:
            raise ValueError("capacity must be non-negative or None")
        return size

    def register(
        self,
        name: str,
        factory: Callable[[], Any],
        *,
        lock_factory: Callable[[], threading.Lock | threading.RLock] | None = None,
        reset: Callable[[Any], Any] | None = None,
        create: bool = True,
        encoder: Callable[[Any], Any] | None = None,
        decoder: Callable[[Any], Any] | None = None,
    ) -> None:
        """Register ``name`` with ``factory`` and optional lifecycle hooks."""

        if lock_factory is None:
            lock_factory = threading.RLock
        with self._registry_lock:
            entry = self._entries.get(name)
            if entry is None:
                entry = _CacheEntry(
                    factory=factory,
                    lock=lock_factory(),
                    reset=reset,
                    encoder=encoder,
                    decoder=decoder,
                )
                self._entries[name] = entry
            else:
                # Update hooks when re-registering the same cache name.
                entry.factory = factory
                entry.reset = reset
                entry.encoder = encoder
                entry.decoder = decoder
            self._ensure_metrics(name)
        if create:
            self.get(name)

    def configure(
        self,
        *,
        default_capacity: int | None | object = _MISSING,
        overrides: Mapping[str, int | None] | None = None,
        replace_overrides: bool = False,
    ) -> None:
        """Update the cache capacity policy shared by registered entries."""

        with self._registry_lock:
            if default_capacity is not self._MISSING:
                self._default_capacity = self._normalise_capacity(
                    default_capacity if default_capacity is not None else None
                )
            if overrides is not None:
                if replace_overrides:
                    self._capacity_overrides.clear()
                for key, value in overrides.items():
                    self._capacity_overrides[key] = self._normalise_capacity(value)

    def configure_from_mapping(self, config: Mapping[str, Any]) -> None:
        """Load configuration produced by :meth:`export_config`."""

        default = config.get("default_capacity", self._MISSING)
        overrides = config.get("overrides")
        overrides_mapping: Mapping[str, int | None] | None
        overrides_mapping = overrides if isinstance(overrides, Mapping) else None
        self.configure(default_capacity=default, overrides=overrides_mapping)

    def export_config(self) -> CacheCapacityConfig:
        """Return a copy of the current capacity configuration."""

        with self._registry_lock:
            return CacheCapacityConfig(
                default_capacity=self._default_capacity,
                overrides=dict(self._capacity_overrides),
            )

    def get_capacity(
        self,
        name: str,
        *,
        requested: int | None = None,
        fallback: int | None = None,
        use_default: bool = True,
    ) -> int | None:
        """Return capacity for ``name`` considering overrides and defaults."""

        with self._registry_lock:
            override = self._capacity_overrides.get(name, self._MISSING)
            default = self._default_capacity
        if override is not self._MISSING:
            return override
        values: tuple[int | None, ...]
        if use_default:
            values = (requested, default, fallback)
        else:
            values = (requested, fallback)
        for value in values:
            if value is self._MISSING:
                continue
            normalised = self._normalise_capacity(value)
            if normalised is not None:
                return normalised
        return None

    def has_override(self, name: str) -> bool:
        """Return ``True`` if ``name`` has an explicit capacity override."""

        with self._registry_lock:
            return name in self._capacity_overrides

    def get_lock(self, name: str) -> threading.Lock | threading.RLock:
        """Return the lock guarding cache ``name`` for external coordination."""

        entry = self._entries.get(name)
        if entry is None:
            raise KeyError(name)
        return entry.lock

    def names(self) -> Iterator[str]:
        """Iterate over registered cache names."""

        with self._registry_lock:
            return iter(tuple(self._entries))

    def get(self, name: str, *, create: bool = True) -> Any:
        """Return cache ``name`` creating it on demand when ``create`` is true."""

        entry = self._entries.get(name)
        if entry is None:
            raise KeyError(name)
        with entry.lock:
            value = self._load_from_layers(name, entry)
            if create and value is None:
                value = entry.factory()
                self._persist_layers(name, entry, value)
            return value

    def peek(self, name: str) -> Any:
        """Return cache ``name`` without creating a missing entry."""

        entry = self._entries.get(name)
        if entry is None:
            raise KeyError(name)
        with entry.lock:
            return self._load_from_layers(name, entry)

    def store(self, name: str, value: Any) -> None:
        """Replace the stored value for cache ``name`` with ``value``."""

        entry = self._entries.get(name)
        if entry is None:
            raise KeyError(name)
        with entry.lock:
            self._persist_layers(name, entry, value)

    def update(
        self,
        name: str,
        updater: Callable[[Any], Any],
        *,
        create: bool = True,
    ) -> Any:
        """Apply ``updater`` to cache ``name`` storing the resulting value."""

        entry = self._entries.get(name)
        if entry is None:
            raise KeyError(name)
        with entry.lock:
            current = self._load_from_layers(name, entry)
            if create and current is None:
                current = entry.factory()
            new_value = updater(current)
            self._persist_layers(name, entry, new_value)
            return new_value

    def clear(self, name: str | None = None) -> None:
        """Reset caches either selectively or for every registered name."""

        if name is not None:
            names = (name,)
        else:
            with self._registry_lock:
                names = tuple(self._entries)
        for cache_name in names:
            entry = self._entries.get(cache_name)
            if entry is None:
                continue
            with entry.lock:
                current = self._load_from_layers(cache_name, entry)
                new_value = None
                if entry.reset is not None:
                    try:
                        new_value = entry.reset(current)
                    except Exception:  # pragma: no cover - defensive logging
                        _logger.exception("cache reset failed for %s", cache_name)
                if new_value is None:
                    try:
                        new_value = entry.factory()
                    except Exception:
                        self._delete_from_layers(cache_name)
                        continue
                self._persist_layers(cache_name, entry, new_value)

    # ------------------------------------------------------------------
    # Layer orchestration helpers

    def _encode_value(self, entry: _CacheEntry, value: Any) -> Any:
        encoder = entry.encoder
        if encoder is None:
            return value
        return encoder(value)

    def _decode_value(self, entry: _CacheEntry, payload: Any) -> Any:
        decoder = entry.decoder
        if decoder is None:
            return payload
        return decoder(payload)

    def _store_layer(self, name: str, entry: _CacheEntry, value: Any, *, layer_index: int) -> None:
        layer = self._layers[layer_index]
        if layer_index == 0:
            payload = value
        else:
            try:
                payload = self._encode_value(entry, value)
            except Exception:  # pragma: no cover - defensive logging
                _logger.exception("cache encoding failed for %s", name)
                return
        try:
            layer.store(name, payload)
        except Exception:  # pragma: no cover - defensive logging
            _logger.exception(
                "cache layer store failed for %s on %s", name, layer.__class__.__name__
            )

    def _persist_layers(self, name: str, entry: _CacheEntry, value: Any) -> None:
        for index in range(len(self._layers)):
            self._store_layer(name, entry, value, layer_index=index)

    def _delete_from_layers(self, name: str) -> None:
        for layer in self._layers:
            try:
                layer.delete(name)
            except KeyError:
                continue
            except Exception:  # pragma: no cover - defensive logging
                _logger.exception(
                    "cache layer delete failed for %s on %s",
                    name,
                    layer.__class__.__name__,
                )

    def _load_from_layers(self, name: str, entry: _CacheEntry) -> Any:
        # Primary in-memory layer first for fast-path lookups.
        try:
            value = self._layers[0].load(name)
        except KeyError:
            value = None
        except Exception:  # pragma: no cover - defensive logging
            _logger.exception(
                "cache layer load failed for %s on %s",
                name,
                self._layers[0].__class__.__name__,
            )
            value = None
        if value is not None:
            return value

        # Fall back to slower layers and hydrate preceding caches on success.
        for index in range(1, len(self._layers)):
            layer = self._layers[index]
            try:
                payload = layer.load(name)
            except KeyError:
                continue
            except Exception:  # pragma: no cover - defensive logging
                _logger.exception(
                    "cache layer load failed for %s on %s",
                    name,
                    layer.__class__.__name__,
                )
                continue
            try:
                value = self._decode_value(entry, payload)
            except Exception:  # pragma: no cover - defensive logging
                _logger.exception("cache decoding failed for %s", name)
                continue
            if value is None:
                continue
            for prev_index in range(index):
                self._store_layer(name, entry, value, layer_index=prev_index)
            return value
        return None

    # ------------------------------------------------------------------
    # Metrics helpers

    def _ensure_metrics(self, name: str) -> _CacheMetrics:
        metrics = self._metrics.get(name)
        if metrics is None:
            with self._registry_lock:
                metrics = self._metrics.get(name)
                if metrics is None:
                    metrics = _CacheMetrics()
                    self._metrics[name] = metrics
        return metrics

    def increment_hit(
        self,
        name: str,
        *,
        amount: int = 1,
        duration: float | None = None,
    ) -> None:
        """Increase cache hit counters for ``name`` (optionally logging latency)."""

        metrics = self._ensure_metrics(name)
        with metrics.lock:
            metrics.hits += int(amount)
            if duration is not None:
                metrics.total_time += float(duration)
                metrics.timings += 1

    def increment_miss(
        self,
        name: str,
        *,
        amount: int = 1,
        duration: float | None = None,
    ) -> None:
        """Increase cache miss counters for ``name`` (optionally logging latency)."""

        metrics = self._ensure_metrics(name)
        with metrics.lock:
            metrics.misses += int(amount)
            if duration is not None:
                metrics.total_time += float(duration)
                metrics.timings += 1

    def increment_eviction(self, name: str, *, amount: int = 1) -> None:
        """Increase eviction count for cache ``name``."""

        metrics = self._ensure_metrics(name)
        with metrics.lock:
            metrics.evictions += int(amount)

    def record_timing(self, name: str, duration: float) -> None:
        """Accumulate ``duration`` into latency telemetry for ``name``."""

        metrics = self._ensure_metrics(name)
        with metrics.lock:
            metrics.total_time += float(duration)
            metrics.timings += 1

    @contextmanager
    def timer(self, name: str) -> TimingContext:
        """Context manager recording execution time for ``name``."""

        start = perf_counter()
        try:
            yield
        finally:
            self.record_timing(name, perf_counter() - start)

    def get_metrics(self, name: str) -> CacheStatistics:
        """Return a snapshot of telemetry collected for cache ``name``."""

        metrics = self._metrics.get(name)
        if metrics is None:
            return CacheStatistics()
        with metrics.lock:
            return metrics.snapshot()

    def iter_metrics(self) -> Iterator[tuple[str, CacheStatistics]]:
        """Yield ``(name, stats)`` pairs for every cache with telemetry."""

        with self._registry_lock:
            items = tuple(self._metrics.items())
        for name, metrics in items:
            with metrics.lock:
                yield name, metrics.snapshot()

    def aggregate_metrics(self) -> CacheStatistics:
        """Return aggregated telemetry statistics across all caches."""

        aggregate = CacheStatistics()
        for _, stats in self.iter_metrics():
            aggregate = aggregate.merge(stats)
        return aggregate

    def register_metrics_publisher(self, publisher: Callable[[str, CacheStatistics], None]) -> None:
        """Register ``publisher`` to receive metrics snapshots on demand."""

        with self._registry_lock:
            self._metrics_publishers.append(publisher)

    def publish_metrics(
        self,
        *,
        publisher: Callable[[str, CacheStatistics], None] | None = None,
    ) -> None:
        """Send cached telemetry to ``publisher`` or all registered publishers."""

        if publisher is None:
            with self._registry_lock:
                publishers = tuple(self._metrics_publishers)
        else:
            publishers = (publisher,)
        if not publishers:
            return
        snapshot = tuple(self.iter_metrics())
        for emit in publishers:
            for name, stats in snapshot:
                try:
                    emit(name, stats)
                except Exception:  # pragma: no cover - defensive logging
                    _logger.exception("Cache metrics publisher failed for %s", name)

    def log_metrics(self, logger: logging.Logger, *, level: int = logging.INFO) -> None:
        """Emit cache metrics using ``logger`` for telemetry hooks."""

        for name, stats in self.iter_metrics():
            logger.log(
                level,
                "cache=%s hits=%d misses=%d evictions=%d timings=%d total_time=%.6f",
                name,
                stats.hits,
                stats.misses,
                stats.evictions,
                stats.timings,
                stats.total_time,
            )


try:
    from .init import get_logger as _get_logger
except ImportError:  # pragma: no cover - circular bootstrap fallback

    def _get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)


_logger = _get_logger(__name__)
get_logger = _get_logger


def _normalise_callbacks(
    callbacks: Iterable[Callable[[K, V], None]] | Callable[[K, V], None] | None,
) -> tuple[Callable[[K, V], None], ...]:
    if callbacks is None:
        return ()
    if callable(callbacks):
        return (callbacks,)
    return tuple(callbacks)


def prune_lock_mapping(
    cache: Mapping[K, Any] | MutableMapping[K, Any] | None,
    locks: MutableMapping[K, Any] | None,
) -> None:
    """Drop lock entries not present in ``cache``."""

    if locks is None:
        return
    if cache is None:
        cache_keys: set[K] = set()
    else:
        cache_keys = set(cache.keys())
    for key in list(locks.keys()):
        if key not in cache_keys:
            locks.pop(key, None)


class InstrumentedLRUCache(MutableMapping[K, V], Generic[K, V]):
    """LRU cache wrapper that synchronises telemetry, callbacks and locks.

    The wrapper owns an internal :class:`cachetools.LRUCache` instance and
    forwards all read operations to it. Mutating operations are instrumented to
    update :class:`CacheManager` metrics, execute registered callbacks and keep
    an optional lock mapping aligned with the stored keys. Telemetry callbacks
    always execute before eviction callbacks, preserving the registration order
    for deterministic side effects.

    Callbacks can be extended or replaced after construction via
    :meth:`set_telemetry_callbacks` and :meth:`set_eviction_callbacks`. When
    ``append`` is ``False`` (default) the provided callbacks replace the
    existing sequence; otherwise they are appended at the end while keeping the
    previous ordering intact.
    """

    _MISSING = object()

    def __init__(
        self,
        maxsize: int,
        *,
        manager: CacheManager | None = None,
        metrics_key: str | None = None,
        telemetry_callbacks: (
            Iterable[Callable[[K, V], None]] | Callable[[K, V], None] | None
        ) = None,
        eviction_callbacks: Iterable[Callable[[K, V], None]] | Callable[[K, V], None] | None = None,
        locks: MutableMapping[K, Any] | None = None,
        getsizeof: Callable[[V], int] | None = None,
        count_overwrite_hit: bool = True,
    ) -> None:
        self._cache: LRUCache[K, V] = LRUCache(maxsize, getsizeof=getsizeof)
        original_popitem = self._cache.popitem

        def _instrumented_popitem() -> tuple[K, V]:
            key, value = original_popitem()
            self._dispatch_removal(key, value)
            return key, value

        self._cache.popitem = _instrumented_popitem  # type: ignore[assignment]
        self._manager = manager
        self._metrics_key = metrics_key
        self._locks = locks
        self._count_overwrite_hit = bool(count_overwrite_hit)
        self._telemetry_callbacks: list[Callable[[K, V], None]]
        self._telemetry_callbacks = list(_normalise_callbacks(telemetry_callbacks))
        self._eviction_callbacks: list[Callable[[K, V], None]]
        self._eviction_callbacks = list(_normalise_callbacks(eviction_callbacks))

    # ------------------------------------------------------------------
    # Callback registration helpers

    @property
    def telemetry_callbacks(self) -> tuple[Callable[[K, V], None], ...]:
        """Return currently registered telemetry callbacks."""

        return tuple(self._telemetry_callbacks)

    @property
    def eviction_callbacks(self) -> tuple[Callable[[K, V], None], ...]:
        """Return currently registered eviction callbacks."""

        return tuple(self._eviction_callbacks)

    def set_telemetry_callbacks(
        self,
        callbacks: Iterable[Callable[[K, V], None]] | Callable[[K, V], None] | None,
        *,
        append: bool = False,
    ) -> None:
        """Update telemetry callbacks executed on removals.

        When ``append`` is ``True`` the provided callbacks are added to the end
        of the execution chain while preserving relative order. Otherwise, the
        previous callbacks are replaced.
        """

        new_callbacks = list(_normalise_callbacks(callbacks))
        if append:
            self._telemetry_callbacks.extend(new_callbacks)
        else:
            self._telemetry_callbacks = new_callbacks

    def set_eviction_callbacks(
        self,
        callbacks: Iterable[Callable[[K, V], None]] | Callable[[K, V], None] | None,
        *,
        append: bool = False,
    ) -> None:
        """Update eviction callbacks executed on removals.

        Behaviour matches :meth:`set_telemetry_callbacks`.
        """

        new_callbacks = list(_normalise_callbacks(callbacks))
        if append:
            self._eviction_callbacks.extend(new_callbacks)
        else:
            self._eviction_callbacks = new_callbacks

    # ------------------------------------------------------------------
    # MutableMapping interface

    def __getitem__(self, key: K) -> V:
        """Return the cached value for ``key``."""

        return self._cache[key]

    def __setitem__(self, key: K, value: V) -> None:
        """Store ``value`` under ``key`` updating telemetry accordingly."""

        exists = key in self._cache
        self._cache[key] = value
        if exists:
            if self._count_overwrite_hit:
                self._record_hit(1)
        else:
            self._record_miss(1)

    def __delitem__(self, key: K) -> None:
        """Remove ``key`` from the cache and dispatch removal callbacks."""

        try:
            value = self._cache[key]
        except KeyError:
            self._record_miss(1)
            raise
        del self._cache[key]
        self._dispatch_removal(key, value, hits=1)

    def __iter__(self) -> Iterator[K]:
        """Iterate over cached keys in eviction order."""

        return iter(self._cache)

    def __len__(self) -> int:
        """Return the number of cached entries."""

        return len(self._cache)

    def __contains__(self, key: object) -> bool:
        """Return ``True`` when ``key`` is stored in the cache."""

        return key in self._cache

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        """Return a debug representation including the underlying cache."""

        return f"{self.__class__.__name__}({self._cache!r})"

    # ------------------------------------------------------------------
    # Cache helpers

    @property
    def maxsize(self) -> int:
        """Return the configured maximum cache size."""

        return self._cache.maxsize

    @property
    def currsize(self) -> int:
        """Return the current weighted size reported by :mod:`cachetools`."""

        return self._cache.currsize

    def get(self, key: K, default: V | None = None) -> V | None:
        """Return ``key`` if present, otherwise ``default``."""

        return self._cache.get(key, default)

    def pop(self, key: K, default: Any = _MISSING) -> V:
        """Remove ``key`` returning its value or ``default`` when provided."""

        try:
            value = self._cache[key]
        except KeyError:
            self._record_miss(1)
            if default is self._MISSING:
                raise
            return cast(V, default)
        del self._cache[key]
        self._dispatch_removal(key, value, hits=1)
        return value

    def popitem(self) -> tuple[K, V]:
        """Remove and return the LRU entry ensuring instrumentation fires."""

        return self._cache.popitem()

    def clear(self) -> None:  # type: ignore[override]
        """Evict every entry while keeping telemetry and locks consistent."""

        while True:
            try:
                self.popitem()
            except KeyError:
                break
        if self._locks is not None:
            try:
                self._locks.clear()
            except Exception:  # pragma: no cover - defensive logging
                _logger.exception("lock cleanup failed during cache clear")

    # ------------------------------------------------------------------
    # Internal helpers

    def _record_hit(self, amount: int) -> None:
        if amount and self._manager is not None and self._metrics_key is not None:
            self._manager.increment_hit(self._metrics_key, amount=amount)

    def _record_miss(self, amount: int) -> None:
        if amount and self._manager is not None and self._metrics_key is not None:
            self._manager.increment_miss(self._metrics_key, amount=amount)

    def _record_eviction(self, amount: int) -> None:
        if amount and self._manager is not None and self._metrics_key is not None:
            self._manager.increment_eviction(self._metrics_key, amount=amount)

    def _dispatch_removal(
        self,
        key: K,
        value: V,
        *,
        hits: int = 0,
        misses: int = 0,
        eviction_amount: int = 1,
        purge_lock: bool = True,
    ) -> None:
        if hits:
            self._record_hit(hits)
        if misses:
            self._record_miss(misses)
        if eviction_amount:
            self._record_eviction(eviction_amount)
        self._emit_callbacks(self._telemetry_callbacks, key, value, "telemetry")
        self._emit_callbacks(self._eviction_callbacks, key, value, "eviction")
        if purge_lock:
            self._purge_lock(key)

    def _emit_callbacks(
        self,
        callbacks: Iterable[Callable[[K, V], None]],
        key: K,
        value: V,
        kind: str,
    ) -> None:
        for callback in callbacks:
            try:
                callback(key, value)
            except Exception:  # pragma: no cover - defensive logging
                _logger.exception("%s callback failed for %r", kind, key)

    def _purge_lock(self, key: K) -> None:
        if self._locks is None:
            return
        try:
            self._locks.pop(key, None)
        except Exception:  # pragma: no cover - defensive logging
            _logger.exception("lock cleanup failed for %r", key)


class ManagedLRUCache(LRUCache[K, V]):
    """LRU cache wrapper with telemetry hooks and lock synchronisation."""

    def __init__(
        self,
        maxsize: int,
        *,
        manager: CacheManager | None = None,
        metrics_key: str | None = None,
        eviction_callbacks: Iterable[Callable[[K, V], None]] | Callable[[K, V], None] | None = None,
        telemetry_callbacks: (
            Iterable[Callable[[K, V], None]] | Callable[[K, V], None] | None
        ) = None,
        locks: MutableMapping[K, Any] | None = None,
    ) -> None:
        super().__init__(maxsize)
        self._manager = manager
        self._metrics_key = metrics_key
        self._locks = locks
        self._eviction_callbacks = _normalise_callbacks(eviction_callbacks)
        self._telemetry_callbacks = _normalise_callbacks(telemetry_callbacks)

    def popitem(self) -> tuple[K, V]:  # type: ignore[override]
        """Evict the LRU entry while updating telemetry and lock state."""

        key, value = super().popitem()
        if self._locks is not None:
            try:
                self._locks.pop(key, None)
            except Exception:  # pragma: no cover - defensive logging
                _logger.exception("lock cleanup failed for %r", key)
        if self._manager is not None and self._metrics_key is not None:
            self._manager.increment_eviction(self._metrics_key)
        for callback in self._telemetry_callbacks:
            try:
                callback(key, value)
            except Exception:  # pragma: no cover - defensive logging
                _logger.exception("telemetry callback failed for %r", key)
        for callback in self._eviction_callbacks:
            try:
                callback(key, value)
            except Exception:  # pragma: no cover - defensive logging
                _logger.exception("eviction callback failed for %r", key)
        return key, value


@dataclass
class _SeedCacheState:
    """Container tracking the state for :class:`_SeedHashCache`."""

    cache: InstrumentedLRUCache[tuple[int, int], int] | None
    maxsize: int


@dataclass
class _CounterState(Generic[K]):
    """State bundle used by :class:`ScopedCounterCache`."""

    cache: InstrumentedLRUCache[K, int]
    locks: dict[K, threading.RLock]
    max_entries: int


# Key used to store the node set checksum in a graph's ``graph`` attribute.
NODE_SET_CHECKSUM_KEY = "_node_set_checksum_cache"

logger = _logger


# Helper to avoid importing ``tnfr.utils.init`` at module import time and keep
# circular dependencies at bay while still reusing the canonical numpy loader.
def _require_numpy():
    from .init import get_numpy

    return get_numpy()


# Graph key storing per-graph layer configuration overrides.
_GRAPH_CACHE_LAYERS_KEY = "_tnfr_cache_layers"

# Process-wide configuration for shared cache layers (Shelve/Redis).
_GLOBAL_CACHE_LAYER_CONFIG: dict[str, dict[str, Any]] = {}
_GLOBAL_CACHE_LOCK = threading.RLock()
_GLOBAL_CACHE_MANAGER: CacheManager | None = None

# Keys of cache entries dependent on the edge version. Any change to the edge
# set requires these to be dropped to avoid stale data.
EDGE_VERSION_CACHE_KEYS = ("_trig_version",)


def get_graph_version(graph: Any, key: str, default: int = 0) -> int:
    """Return integer version stored in ``graph`` under ``key``."""

    return int(graph.get(key, default))


def increment_graph_version(graph: Any, key: str) -> int:
    """Increment and store a version counter in ``graph`` under ``key``."""

    version = get_graph_version(graph, key) + 1
    graph[key] = version
    return version


def stable_json(obj: Any) -> str:
    """Return a JSON string with deterministic ordering for ``obj``."""

    from .io import json_dumps

    return json_dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        to_bytes=False,
    )


@lru_cache(maxsize=1024)
def _node_repr_digest(obj: Any) -> tuple[str, bytes]:
    """Return cached stable representation and digest for ``obj``."""

    try:
        repr_ = stable_json(obj)
    except TypeError:
        repr_ = repr(obj)
    digest = hashlib.blake2b(repr_.encode("utf-8"), digest_size=16).digest()
    return repr_, digest


def clear_node_repr_cache() -> None:
    """Clear cached node representations used for checksums."""

    _node_repr_digest.cache_clear()


def configure_global_cache_layers(
    *,
    shelve: Mapping[str, Any] | None = None,
    redis: Mapping[str, Any] | None = None,
    replace: bool = False,
) -> None:
    """Update process-wide cache layer configuration.

    Parameters mirror the per-layer specifications accepted via graph metadata.
    Passing ``replace=True`` clears previous settings before applying new ones.
    Providing ``None`` for a layer while ``replace`` is true removes that layer
    from the configuration.
    """

    global _GLOBAL_CACHE_MANAGER
    with _GLOBAL_CACHE_LOCK:
        manager = _GLOBAL_CACHE_MANAGER
        _GLOBAL_CACHE_MANAGER = None
        if replace:
            _GLOBAL_CACHE_LAYER_CONFIG.clear()
        if shelve is not None:
            _GLOBAL_CACHE_LAYER_CONFIG["shelve"] = dict(shelve)
        elif replace:
            _GLOBAL_CACHE_LAYER_CONFIG.pop("shelve", None)
        if redis is not None:
            _GLOBAL_CACHE_LAYER_CONFIG["redis"] = dict(redis)
        elif replace:
            _GLOBAL_CACHE_LAYER_CONFIG.pop("redis", None)
    _close_cache_layers(manager)


def _resolve_layer_config(
    graph: MutableMapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    resolved: dict[str, dict[str, Any]] = {}
    with _GLOBAL_CACHE_LOCK:
        for name, spec in _GLOBAL_CACHE_LAYER_CONFIG.items():
            resolved[name] = dict(spec)
    if graph is not None:
        overrides = graph.get(_GRAPH_CACHE_LAYERS_KEY)
        if isinstance(overrides, Mapping):
            for name in ("shelve", "redis"):
                layer_spec = overrides.get(name)
                if isinstance(layer_spec, Mapping):
                    resolved[name] = dict(layer_spec)
                elif layer_spec is None:
                    resolved.pop(name, None)
    return resolved


def _build_shelve_layer(spec: Mapping[str, Any]) -> ShelveCacheLayer | None:
    path = spec.get("path")
    if not path:
        return None
    flag = spec.get("flag", "c")
    protocol = spec.get("protocol")
    writeback = bool(spec.get("writeback", False))
    try:
        proto_arg = None if protocol is None else int(protocol)
    except (TypeError, ValueError):
        logger.warning("Invalid shelve protocol %r; falling back to default", protocol)
        proto_arg = None
    try:
        return ShelveCacheLayer(
            str(path),
            flag=str(flag),
            protocol=proto_arg,
            writeback=writeback,
        )
    except Exception:  # pragma: no cover - defensive logging
        logger.exception("Failed to initialise ShelveCacheLayer for path %r", path)
        return None


def _build_redis_layer(spec: Mapping[str, Any]) -> RedisCacheLayer | None:
    enabled = spec.get("enabled", True)
    if not enabled:
        return None
    namespace = spec.get("namespace")
    client = spec.get("client")
    if client is None:
        factory = spec.get("client_factory")
        if callable(factory):
            try:
                client = factory()
            except Exception:  # pragma: no cover - defensive logging
                logger.exception("Redis cache client factory failed")
                return None
        else:
            kwargs = spec.get("client_kwargs")
            if isinstance(kwargs, Mapping):
                try:  # pragma: no cover - optional dependency
                    import redis  # type: ignore
                except Exception:  # pragma: no cover - defensive logging
                    logger.exception("redis-py is required to build the configured Redis client")
                    return None
                try:
                    client = redis.Redis(**dict(kwargs))
                except Exception:  # pragma: no cover - defensive logging
                    logger.exception("Failed to initialise redis client with %r", kwargs)
                    return None
    try:
        if namespace is None:
            return RedisCacheLayer(client=client)
        return RedisCacheLayer(client=client, namespace=str(namespace))
    except Exception:  # pragma: no cover - defensive logging
        logger.exception("Failed to initialise RedisCacheLayer")
        return None


def _build_cache_layers(config: Mapping[str, dict[str, Any]]) -> tuple[CacheLayer, ...]:
    layers: list[CacheLayer] = []
    shelve_spec = config.get("shelve")
    if isinstance(shelve_spec, Mapping):
        layer = _build_shelve_layer(shelve_spec)
        if layer is not None:
            layers.append(layer)
    redis_spec = config.get("redis")
    if isinstance(redis_spec, Mapping):
        layer = _build_redis_layer(redis_spec)
        if layer is not None:
            layers.append(layer)
    return tuple(layers)


def _close_cache_layers(manager: CacheManager | None) -> None:
    if manager is None:
        return
    layers = getattr(manager, "_layers", ())
    for layer in layers:
        close = getattr(layer, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # pragma: no cover - defensive logging
                logger.exception("Cache layer close failed for %s", layer.__class__.__name__)


def reset_global_cache_manager() -> None:
    """Dispose the shared cache manager and close attached layers."""

    global _GLOBAL_CACHE_MANAGER
    with _GLOBAL_CACHE_LOCK:
        manager = _GLOBAL_CACHE_MANAGER
        _GLOBAL_CACHE_MANAGER = None
    _close_cache_layers(manager)


def build_cache_manager(
    *,
    graph: MutableMapping[str, Any] | None = None,
    storage: MutableMapping[str, Any] | None = None,
    default_capacity: int | None = None,
    overrides: Mapping[str, int | None] | None = None,
) -> CacheManager:
    """Construct a :class:`CacheManager` honouring configured cache layers."""

    global _GLOBAL_CACHE_MANAGER
    if graph is None:
        with _GLOBAL_CACHE_LOCK:
            manager = _GLOBAL_CACHE_MANAGER
        if manager is not None:
            return manager

    layers = _build_cache_layers(_resolve_layer_config(graph))
    manager = CacheManager(
        storage,
        default_capacity=default_capacity,
        overrides=overrides,
        layers=layers,
    )

    if graph is None:
        with _GLOBAL_CACHE_LOCK:
            global_manager = _GLOBAL_CACHE_MANAGER
            if global_manager is None:
                _GLOBAL_CACHE_MANAGER = manager
                return manager
        _close_cache_layers(manager)
        return global_manager

    return manager


def _node_repr(n: Any) -> str:
    """Stable representation for node hashing and sorting."""

    return _node_repr_digest(n)[0]


def _iter_node_digests(nodes: Iterable[Any], *, presorted: bool) -> Iterable[bytes]:
    """Yield node digests in a deterministic order."""

    if presorted:
        for node in nodes:
            yield _node_repr_digest(node)[1]
    else:
        for _, digest in sorted((_node_repr_digest(n) for n in nodes), key=lambda x: x[0]):
            yield digest


def _node_set_checksum_no_nodes(
    G: nx.Graph,
    graph: Any,
    *,
    presorted: bool,
    store: bool,
) -> str:
    """Checksum helper when no explicit node set is provided."""

    nodes_view = G.nodes()
    current_nodes = frozenset(nodes_view)
    cached = graph.get(NODE_SET_CHECKSUM_KEY)
    if cached and len(cached) == 3 and cached[2] == current_nodes:
        return cached[1]

    hasher = hashlib.blake2b(digest_size=16)
    for digest in _iter_node_digests(nodes_view, presorted=presorted):
        hasher.update(digest)

    checksum = hasher.hexdigest()
    if store:
        token = checksum[:16]
        if cached and cached[0] == token:
            return cached[1]
        graph[NODE_SET_CHECKSUM_KEY] = (token, checksum, current_nodes)
    else:
        graph.pop(NODE_SET_CHECKSUM_KEY, None)
    return checksum


def node_set_checksum(
    G: nx.Graph,
    nodes: Iterable[Any] | None = None,
    *,
    presorted: bool = False,
    store: bool = True,
) -> str:
    """Return a BLAKE2b checksum of ``G``'s node set."""

    graph = get_graph(G)
    if nodes is None:
        return _node_set_checksum_no_nodes(G, graph, presorted=presorted, store=store)

    hasher = hashlib.blake2b(digest_size=16)
    for digest in _iter_node_digests(nodes, presorted=presorted):
        hasher.update(digest)

    checksum = hasher.hexdigest()
    if store:
        token = checksum[:16]
        cached = graph.get(NODE_SET_CHECKSUM_KEY)
        if cached and cached[0] == token:
            return cached[1]
        graph[NODE_SET_CHECKSUM_KEY] = (token, checksum)
    else:
        graph.pop(NODE_SET_CHECKSUM_KEY, None)
    return checksum


@dataclass(slots=True)
class NodeCache:
    """Container for cached node data."""

    checksum: str
    nodes: tuple[Any, ...]
    sorted_nodes: tuple[Any, ...] | None = None
    idx: dict[Any, int] | None = None
    offset: dict[Any, int] | None = None

    @property
    def n(self) -> int:
        return len(self.nodes)


def _update_node_cache(
    graph: Any,
    nodes: tuple[Any, ...],
    key: str,
    *,
    checksum: str,
    sorted_nodes: tuple[Any, ...] | None = None,
) -> None:
    """Store ``nodes`` and ``checksum`` in ``graph`` under ``key``."""

    graph[f"{key}_cache"] = NodeCache(checksum=checksum, nodes=nodes, sorted_nodes=sorted_nodes)
    graph[f"{key}_checksum"] = checksum


def _refresh_node_list_cache(
    G: nx.Graph,
    graph: Any,
    *,
    sort_nodes: bool,
    current_n: int,
) -> tuple[Any, ...]:
    """Refresh the cached node list and return the nodes."""

    nodes = tuple(G.nodes())
    checksum = node_set_checksum(G, nodes, store=True)
    sorted_nodes = tuple(sorted(nodes, key=_node_repr)) if sort_nodes else None
    _update_node_cache(
        graph,
        nodes,
        "_node_list",
        checksum=checksum,
        sorted_nodes=sorted_nodes,
    )
    graph["_node_list_len"] = current_n
    return nodes


def _reuse_node_list_cache(
    graph: Any,
    cache: NodeCache,
    nodes: tuple[Any, ...],
    sorted_nodes: tuple[Any, ...] | None,
    *,
    sort_nodes: bool,
    new_checksum: str | None,
) -> None:
    """Reuse existing node cache and record its checksum if missing."""

    checksum = cache.checksum if new_checksum is None else new_checksum
    if sort_nodes and sorted_nodes is None:
        sorted_nodes = tuple(sorted(nodes, key=_node_repr))
    _update_node_cache(
        graph,
        nodes,
        "_node_list",
        checksum=checksum,
        sorted_nodes=sorted_nodes,
    )


def _cache_node_list(G: nx.Graph) -> tuple[Any, ...]:
    """Cache and return the tuple of nodes for ``G``."""

    graph = get_graph(G)
    cache: NodeCache | None = graph.get("_node_list_cache")
    nodes = cache.nodes if cache else None
    sorted_nodes = cache.sorted_nodes if cache else None
    stored_len = graph.get("_node_list_len")
    current_n = G.number_of_nodes()
    dirty = bool(graph.pop("_node_list_dirty", False))

    invalid = nodes is None or stored_len != current_n or dirty
    new_checksum: str | None = None

    if not invalid and cache:
        new_checksum = node_set_checksum(G)
        invalid = cache.checksum != new_checksum

    sort_nodes = bool(graph.get("SORT_NODES", False))

    if invalid:
        nodes = _refresh_node_list_cache(G, graph, sort_nodes=sort_nodes, current_n=current_n)
    elif cache and "_node_list_checksum" not in graph:
        _reuse_node_list_cache(
            graph,
            cache,
            nodes,
            sorted_nodes,
            sort_nodes=sort_nodes,
            new_checksum=new_checksum,
        )
    else:
        if sort_nodes and sorted_nodes is None and cache is not None:
            cache.sorted_nodes = tuple(sorted(nodes, key=_node_repr))
    return nodes


def cached_node_list(G: nx.Graph) -> tuple[Any, ...]:
    """Public wrapper returning the cached node tuple for ``G``."""

    return _cache_node_list(G)


def _ensure_node_map(
    G: TNFRGraph,
    *,
    attrs: tuple[str, ...],
    sort: bool = False,
) -> dict[NodeId, int]:
    """Return cached node-to-index/offset mappings stored on ``NodeCache``."""

    graph = G.graph
    _cache_node_list(G)
    cache: NodeCache = graph["_node_list_cache"]

    missing = [attr for attr in attrs if getattr(cache, attr) is None]
    if missing:
        if sort:
            nodes_opt = cache.sorted_nodes
            if nodes_opt is None:
                nodes_opt = tuple(sorted(cache.nodes, key=_node_repr))
                cache.sorted_nodes = nodes_opt
            nodes_seq = nodes_opt
        else:
            nodes_seq = cache.nodes
        node_ids = cast(tuple[NodeId, ...], nodes_seq)
        mappings: dict[str, dict[NodeId, int]] = {attr: {} for attr in missing}
        for idx, node in enumerate(node_ids):
            for attr in missing:
                mappings[attr][node] = idx
        for attr in missing:
            setattr(cache, attr, mappings[attr])
    return cast(dict[NodeId, int], getattr(cache, attrs[0]))


def ensure_node_index_map(G: TNFRGraph) -> dict[NodeId, int]:
    """Return cached node-to-index mapping for ``G``."""

    return _ensure_node_map(G, attrs=("idx",), sort=False)


def ensure_node_offset_map(G: TNFRGraph) -> dict[NodeId, int]:
    """Return cached node-to-offset mapping for ``G``."""

    sort = bool(G.graph.get("SORT_NODES", False))
    return _ensure_node_map(G, attrs=("offset",), sort=sort)


@dataclass
class EdgeCacheState:
    cache: MutableMapping[Hashable, Any]
    locks: defaultdict[Hashable, threading.RLock]
    max_entries: int | None
    dirty: bool = False


_GRAPH_CACHE_MANAGER_KEY = "_tnfr_cache_manager"
_GRAPH_CACHE_CONFIG_KEY = "_tnfr_cache_config"
DNFR_PREP_STATE_KEY = "_dnfr_prep_state"

# Ephemeral graph cache management:
# ----------------------------------
# TNFR stores cache managers directly in each graph's `.graph` dictionary
# via _GRAPH_CACHE_MANAGER_KEY. This design inherently supports ephemeral
# graphs because:
#
# 1. **Automatic cleanup**: When an ephemeral graph object is garbage
#    collected, its `.graph` dict and all associated cache managers are
#    automatically released with it. No manual cleanup is required.
#
# 2. **Isolation**: Each graph has its own cache manager instance, preventing
#    cache pollution between unrelated graphs or temporary computations.
#
# 3. **No global state**: Unlike WeakValueDictionary-based global caches,
#    there's no shared cache registry that needs weak references to track
#    ephemeral graphs.
#
# For temporary or short-lived graphs (e.g., subgraphs, clones, simulation
# snapshots), simply let the graph go out of scope and Python's garbage
# collector will reclaim all associated caches. No special ephemeral flag
# or WeakValueDictionary is needed.
#
# Example ephemeral graph usage:
#   def process_subgraph(G, nodes):
#       H = G.subgraph(nodes).copy()  # Ephemeral graph
#       default_compute_delta_nfr(H)  # Creates temporary cache
#       return extract_metrics(H)
#       # H and its caches are GC'd when function returns


@dataclass(slots=True)
class DnfrPrepState:
    """State container coordinating ΔNFR preparation caches."""

    cache: DnfrCache
    cache_lock: threading.RLock
    vector_lock: threading.RLock


def _build_dnfr_prep_state(
    graph: MutableMapping[str, Any],
    previous: DnfrPrepState | None = None,
) -> DnfrPrepState:
    """Construct a :class:`DnfrPrepState` and mirror it on ``graph``."""

    cache_lock: threading.RLock
    vector_lock: threading.RLock
    if isinstance(previous, DnfrPrepState):
        cache_lock = previous.cache_lock
        vector_lock = previous.vector_lock
    else:
        cache_lock = threading.RLock()
        vector_lock = threading.RLock()
    state = DnfrPrepState(
        cache=new_dnfr_cache(),
        cache_lock=cache_lock,
        vector_lock=vector_lock,
    )
    graph["_dnfr_prep_cache"] = state.cache
    return state


def _coerce_dnfr_state(
    graph: MutableMapping[str, Any],
    current: Any,
) -> DnfrPrepState:
    """Return ``current`` normalised into :class:`DnfrPrepState`."""

    if isinstance(current, DnfrPrepState):
        graph["_dnfr_prep_cache"] = current.cache
        return current
    if isinstance(current, DnfrCache):
        state = DnfrPrepState(
            cache=current,
            cache_lock=threading.RLock(),
            vector_lock=threading.RLock(),
        )
        graph["_dnfr_prep_cache"] = current
        return state
    return _build_dnfr_prep_state(graph)


def _graph_cache_manager(graph: MutableMapping[str, Any]) -> CacheManager:
    manager = graph.get(_GRAPH_CACHE_MANAGER_KEY)
    if not isinstance(manager, CacheManager):
        manager = build_cache_manager(graph=graph, default_capacity=128)
        graph[_GRAPH_CACHE_MANAGER_KEY] = manager
    config = graph.get(_GRAPH_CACHE_CONFIG_KEY)
    if isinstance(config, dict):
        manager.configure_from_mapping(config)

    def _dnfr_factory() -> DnfrPrepState:
        return _build_dnfr_prep_state(graph)

    def _dnfr_reset(current: Any) -> DnfrPrepState:
        if isinstance(current, DnfrPrepState):
            return _build_dnfr_prep_state(graph, current)
        return _build_dnfr_prep_state(graph)

    manager.register(
        DNFR_PREP_STATE_KEY,
        _dnfr_factory,
        reset=_dnfr_reset,
    )
    manager.update(
        DNFR_PREP_STATE_KEY,
        lambda current: _coerce_dnfr_state(graph, current),
    )
    return manager


def configure_graph_cache_limits(
    G: GraphLike | TNFRGraph | MutableMapping[str, Any],
    *,
    default_capacity: int | None | object = CacheManager._MISSING,
    overrides: Mapping[str, int | None] | None = None,
    replace_overrides: bool = False,
) -> CacheCapacityConfig:
    """Update cache capacity policy stored on ``G.graph``."""

    graph = get_graph(G)
    manager = _graph_cache_manager(graph)
    manager.configure(
        default_capacity=default_capacity,
        overrides=overrides,
        replace_overrides=replace_overrides,
    )
    snapshot = manager.export_config()
    graph[_GRAPH_CACHE_CONFIG_KEY] = {
        "default_capacity": snapshot.default_capacity,
        "overrides": dict(snapshot.overrides),
    }
    return snapshot


class EdgeCacheManager:
    """Coordinate cache storage and per-key locks for edge version caches."""

    _STATE_KEY = "_edge_version_state"

    def __init__(self, graph: MutableMapping[str, Any]) -> None:
        self.graph: MutableMapping[str, Any] = graph
        self._manager = _graph_cache_manager(graph)

        def _encode_state(state: EdgeCacheState) -> Mapping[str, Any]:
            if not isinstance(state, EdgeCacheState):
                raise TypeError("EdgeCacheState expected")
            return {
                "max_entries": state.max_entries,
                "entries": list(state.cache.items()),
            }

        def _decode_state(payload: Any) -> EdgeCacheState:
            if isinstance(payload, EdgeCacheState):
                return payload
            if not isinstance(payload, Mapping):
                raise TypeError("invalid edge cache payload")
            max_entries = payload.get("max_entries")
            state = self._build_state(max_entries)
            for key, value in payload.get("entries", []):
                state.cache[key] = value
            state.dirty = False
            return state

        self._manager.register(
            self._STATE_KEY,
            self._default_state,
            reset=self._reset_state,
            encoder=_encode_state,
            decoder=_decode_state,
        )

    def record_hit(self) -> None:
        """Record a cache hit for telemetry."""

        self._manager.increment_hit(self._STATE_KEY)

    def record_miss(self, *, track_metrics: bool = True) -> None:
        """Record a cache miss for telemetry.

        When ``track_metrics`` is ``False`` the miss is acknowledged without
        mutating the aggregated metrics.
        """

        if track_metrics:
            self._manager.increment_miss(self._STATE_KEY)

    def record_eviction(self, *, track_metrics: bool = True) -> None:
        """Record cache eviction events for telemetry.

        When ``track_metrics`` is ``False`` the underlying metrics counter is
        left untouched while still signalling that an eviction occurred.
        """

        if track_metrics:
            self._manager.increment_eviction(self._STATE_KEY)

    def timer(self) -> TimingContext:
        """Return a timing context linked to this cache."""

        return self._manager.timer(self._STATE_KEY)

    def _default_state(self) -> EdgeCacheState:
        return self._build_state(None)

    def resolve_max_entries(self, max_entries: int | None | object) -> int | None:
        """Return effective capacity for the edge cache."""

        if max_entries is CacheManager._MISSING:
            return self._manager.get_capacity(self._STATE_KEY)
        return self._manager.get_capacity(
            self._STATE_KEY,
            requested=None if max_entries is None else int(max_entries),
            use_default=False,
        )

    def _build_state(self, max_entries: int | None) -> EdgeCacheState:
        locks: defaultdict[Hashable, threading.RLock] = defaultdict(threading.RLock)
        capacity = float("inf") if max_entries is None else int(max_entries)
        cache = InstrumentedLRUCache(
            capacity,
            manager=self._manager,
            metrics_key=self._STATE_KEY,
            locks=locks,
            count_overwrite_hit=False,
        )
        state = EdgeCacheState(cache=cache, locks=locks, max_entries=max_entries)

        def _on_eviction(key: Hashable, _: Any) -> None:
            self.record_eviction(track_metrics=False)
            locks.pop(key, None)
            state.dirty = True

        cache.set_eviction_callbacks(_on_eviction)
        return state

    def _ensure_state(
        self, state: EdgeCacheState | None, max_entries: int | None | object
    ) -> EdgeCacheState:
        target = self.resolve_max_entries(max_entries)
        if target is not None:
            target = int(target)
            if target < 0:
                raise ValueError("max_entries must be non-negative or None")
        if not isinstance(state, EdgeCacheState) or state.max_entries != target:
            return self._build_state(target)
        return state

    def _reset_state(self, state: EdgeCacheState | None) -> EdgeCacheState:
        if isinstance(state, EdgeCacheState):
            state.cache.clear()
            state.dirty = False
            return state
        return self._build_state(None)

    def get_cache(
        self,
        max_entries: int | None | object,
        *,
        create: bool = True,
    ) -> EdgeCacheState | None:
        """Return the cache state for the manager's graph."""

        if not create:
            state = self._manager.peek(self._STATE_KEY)
            return state if isinstance(state, EdgeCacheState) else None

        state = self._manager.update(
            self._STATE_KEY,
            lambda current: self._ensure_state(current, max_entries),
        )
        if not isinstance(state, EdgeCacheState):
            raise RuntimeError("edge cache state failed to initialise")
        return state

    def flush_state(self, state: EdgeCacheState) -> None:
        """Persist ``state`` through the configured cache layers when dirty."""

        if not isinstance(state, EdgeCacheState) or not state.dirty:
            return
        self._manager.store(self._STATE_KEY, state)
        state.dirty = False

    def clear(self) -> None:
        """Reset cached data managed by this instance."""

        self._manager.clear(self._STATE_KEY)


def edge_version_cache(
    G: Any,
    key: Hashable,
    builder: Callable[[], T],
    *,
    max_entries: int | None | object = CacheManager._MISSING,
) -> T:
    """Return cached ``builder`` output tied to the edge version of ``G``."""

    graph = get_graph(G)
    manager = graph.get("_edge_cache_manager")  # type: ignore[assignment]
    if not isinstance(manager, EdgeCacheManager) or manager.graph is not graph:
        manager = EdgeCacheManager(graph)
        graph["_edge_cache_manager"] = manager

    resolved = manager.resolve_max_entries(max_entries)
    if resolved == 0:
        return builder()

    state = manager.get_cache(resolved)
    if state is None:
        return builder()

    cache = state.cache
    locks = state.locks
    edge_version = get_graph_version(graph, "_edge_version")
    lock = locks[key]

    with lock:
        entry = cache.get(key)
        if entry is not None and entry[0] == edge_version:
            manager.record_hit()
            return entry[1]

    try:
        with manager.timer():
            value = builder()
    except (RuntimeError, ValueError) as exc:  # pragma: no cover - logging side effect
        logger.exception("edge_version_cache builder failed for %r: %s", key, exc)
        raise
    else:
        result = value
        with lock:
            entry = cache.get(key)
            if entry is not None:
                cached_version, cached_value = entry
                manager.record_miss()
                if cached_version == edge_version:
                    manager.record_hit()
                    return cached_value
                manager.record_eviction()
            cache[key] = (edge_version, value)
            state.dirty = True
            result = value
    if state.dirty:
        manager.flush_state(state)
    return result


def cached_nodes_and_A(
    G: nx.Graph,
    *,
    cache_size: int | None = 1,
    require_numpy: bool = False,
    prefer_sparse: bool = False,
    nodes: tuple[Any, ...] | None = None,
) -> tuple[tuple[Any, ...], Any]:
    """Return cached nodes tuple and adjacency matrix for ``G``.

    When ``prefer_sparse`` is true the adjacency matrix construction is skipped
    unless a caller later requests it explicitly.  This lets ΔNFR reuse the
    edge-index buffers stored on :class:`~tnfr.dynamics.dnfr.DnfrCache` without
    paying for ``nx.to_numpy_array`` on sparse graphs while keeping the
    canonical cache interface unchanged.
    """

    if nodes is None:
        nodes = cached_node_list(G)
    graph = G.graph

    checksum = getattr(graph.get("_node_list_cache"), "checksum", None)
    if checksum is None:
        checksum = graph.get("_node_list_checksum")
    if checksum is None:
        node_set_cache = graph.get(NODE_SET_CHECKSUM_KEY)
        if isinstance(node_set_cache, tuple) and len(node_set_cache) >= 2:
            checksum = node_set_cache[1]
    if checksum is None:
        checksum = ""

    key = f"_dnfr_{len(nodes)}_{checksum}"
    graph["_dnfr_nodes_checksum"] = checksum

    def builder() -> tuple[tuple[Any, ...], Any]:
        np = _require_numpy()
        if np is None or prefer_sparse:
            return nodes, None
        A = nx.to_numpy_array(G, nodelist=nodes, weight=None, dtype=float)
        return nodes, A

    nodes, A = edge_version_cache(G, key, builder, max_entries=cache_size)

    if require_numpy and A is None:
        raise RuntimeError("NumPy is required for adjacency caching")

    return nodes, A


def _reset_edge_caches(graph: Any, G: Any) -> None:
    """Clear caches affected by edge updates."""

    EdgeCacheManager(graph).clear()
    _graph_cache_manager(graph).clear(DNFR_PREP_STATE_KEY)
    mark_dnfr_prep_dirty(G)
    clear_node_repr_cache()
    for key in EDGE_VERSION_CACHE_KEYS:
        graph.pop(key, None)


def increment_edge_version(G: Any) -> None:
    """Increment the edge version counter in ``G.graph``."""

    graph = get_graph(G)
    increment_graph_version(graph, "_edge_version")
    _reset_edge_caches(graph, G)


@contextmanager
def edge_version_update(G: TNFRGraph) -> Iterator[None]:
    """Scope a batch of edge mutations."""

    increment_edge_version(G)
    try:
        yield
    finally:
        increment_edge_version(G)


class _SeedHashCache(MutableMapping[tuple[int, int], int]):
    """Mutable mapping proxy exposing a configurable LRU cache."""

    def __init__(
        self,
        *,
        manager: CacheManager | None = None,
        state_key: str = "seed_hash_cache",
        default_maxsize: int = 128,
    ) -> None:
        self._default_maxsize = int(default_maxsize)
        self._manager = manager or build_cache_manager(default_capacity=self._default_maxsize)
        self._state_key = state_key
        if not self._manager.has_override(self._state_key):
            self._manager.configure(overrides={self._state_key: self._default_maxsize})
        self._manager.register(
            self._state_key,
            self._create_state,
            reset=self._reset_state,
        )

    def _resolved_size(self, requested: int | None = None) -> int:
        size = self._manager.get_capacity(
            self._state_key,
            requested=requested,
            fallback=self._default_maxsize,
        )
        if size is None:
            return 0
        return int(size)

    def _create_state(self) -> _SeedCacheState:
        size = self._resolved_size()
        if size <= 0:
            return _SeedCacheState(cache=None, maxsize=0)
        return _SeedCacheState(
            cache=InstrumentedLRUCache(
                size,
                manager=self._manager,
                metrics_key=self._state_key,
            ),
            maxsize=size,
        )

    def _reset_state(self, state: _SeedCacheState | None) -> _SeedCacheState:
        return self._create_state()

    def _get_state(self, *, create: bool = True) -> _SeedCacheState | None:
        state = self._manager.get(self._state_key, create=create)
        if state is None:
            return None
        if not isinstance(state, _SeedCacheState):
            state = self._create_state()
            self._manager.store(self._state_key, state)
        return state

    def configure(self, maxsize: int) -> None:
        size = int(maxsize)
        if size < 0:
            raise ValueError("maxsize must be non-negative")
        self._manager.configure(overrides={self._state_key: size})
        self._manager.update(self._state_key, lambda _: self._create_state())

    def __getitem__(self, key: tuple[int, int]) -> int:
        state = self._get_state()
        if state is None or state.cache is None:
            raise KeyError(key)
        value = state.cache[key]
        self._manager.increment_hit(self._state_key)
        return value

    def __setitem__(self, key: tuple[int, int], value: int) -> None:
        state = self._get_state()
        if state is not None and state.cache is not None:
            state.cache[key] = value

    def __delitem__(self, key: tuple[int, int]) -> None:
        state = self._get_state()
        if state is None or state.cache is None:
            raise KeyError(key)
        del state.cache[key]

    def __iter__(self) -> Iterator[tuple[int, int]]:
        state = self._get_state(create=False)
        if state is None or state.cache is None:
            return iter(())
        return iter(state.cache)

    def __len__(self) -> int:
        state = self._get_state(create=False)
        if state is None or state.cache is None:
            return 0
        return len(state.cache)

    def clear(self) -> None:  # type: ignore[override]
        self._manager.clear(self._state_key)

    @property
    def maxsize(self) -> int:
        state = self._get_state()
        return 0 if state is None else state.maxsize

    @property
    def enabled(self) -> bool:
        state = self._get_state(create=False)
        return bool(state and state.cache is not None)

    @property
    def data(self) -> InstrumentedLRUCache[tuple[int, int], int] | None:
        """Expose the underlying cache for diagnostics/tests."""

        state = self._get_state(create=False)
        return None if state is None else state.cache


class ScopedCounterCache(Generic[K]):
    """Thread-safe LRU cache storing monotonic counters by ``key``."""

    def __init__(
        self,
        name: str,
        max_entries: int | None = None,
        *,
        manager: CacheManager | None = None,
        default_max_entries: int = 128,
    ) -> None:
        self._name = name
        self._state_key = f"scoped_counter:{name}"
        self._default_max_entries = int(default_max_entries)
        requested = None if max_entries is None else int(max_entries)
        if requested is not None and requested < 0:
            raise ValueError("max_entries must be non-negative")
        self._manager = manager or build_cache_manager(default_capacity=self._default_max_entries)
        if not self._manager.has_override(self._state_key):
            fallback = requested
            if fallback is None:
                fallback = self._default_max_entries
            self._manager.configure(overrides={self._state_key: fallback})
        elif requested is not None:
            self._manager.configure(overrides={self._state_key: requested})
        self._manager.register(
            self._state_key,
            self._create_state,
            lock_factory=lambda: get_lock(name),
            reset=self._reset_state,
        )

    def _resolved_entries(self, requested: int | None = None) -> int:
        size = self._manager.get_capacity(
            self._state_key,
            requested=requested,
            fallback=self._default_max_entries,
        )
        if size is None:
            return 0
        return int(size)

    def _create_state(self, requested: int | None = None) -> _CounterState[K]:
        size = self._resolved_entries(requested)
        locks: dict[K, threading.RLock] = {}
        return _CounterState(
            cache=InstrumentedLRUCache(
                size,
                manager=self._manager,
                metrics_key=self._state_key,
                locks=locks,
            ),
            locks=locks,
            max_entries=size,
        )

    def _reset_state(self, state: _CounterState[K] | None) -> _CounterState[K]:
        return self._create_state()

    def _get_state(self) -> _CounterState[K]:
        state = self._manager.get(self._state_key)
        if not isinstance(state, _CounterState):
            state = self._create_state(0)
            self._manager.store(self._state_key, state)
        return state

    @property
    def lock(self) -> threading.Lock | threading.RLock:
        """Return the lock guarding access to the underlying cache."""

        return self._manager.get_lock(self._state_key)

    @property
    def max_entries(self) -> int:
        """Return the configured maximum number of cached entries."""

        return self._get_state().max_entries

    @property
    def cache(self) -> InstrumentedLRUCache[K, int]:
        """Expose the instrumented cache for inspection."""

        return self._get_state().cache

    @property
    def locks(self) -> dict[K, threading.RLock]:
        """Return the mapping of per-key locks tracked by the cache."""

        return self._get_state().locks

    def configure(self, *, force: bool = False, max_entries: int | None = None) -> None:
        """Resize or reset the cache keeping previous settings."""

        if max_entries is None:
            size = self._resolved_entries()
            update_policy = False
        else:
            size = int(max_entries)
            if size < 0:
                raise ValueError("max_entries must be non-negative")
            update_policy = True

        def _update(state: _CounterState[K] | None) -> _CounterState[K]:
            if not isinstance(state, _CounterState) or force or state.max_entries != size:
                locks: dict[K, threading.RLock] = {}
                return _CounterState(
                    cache=InstrumentedLRUCache(
                        size,
                        manager=self._manager,
                        metrics_key=self._state_key,
                        locks=locks,
                    ),
                    locks=locks,
                    max_entries=size,
                )
            return cast(_CounterState[K], state)

        if update_policy:
            self._manager.configure(overrides={self._state_key: size})
        self._manager.update(self._state_key, _update)

    def clear(self) -> None:
        """Clear stored counters preserving ``max_entries``."""

        self.configure(force=True)

    def bump(self, key: K) -> int:
        """Return current counter for ``key`` and increment it atomically."""

        result: dict[str, Any] = {}

        def _update(state: _CounterState[K] | None) -> _CounterState[K]:
            if not isinstance(state, _CounterState):
                state = self._create_state(0)
            cache = state.cache
            locks = state.locks
            if key not in locks:
                locks[key] = threading.RLock()
            value = int(cache.get(key, 0))
            cache[key] = value + 1
            result["value"] = value
            return state

        self._manager.update(self._state_key, _update)
        return int(result.get("value", 0))

    def __len__(self) -> int:
        """Return the number of tracked counters."""

        return len(self.cache)


# ============================================================================
# Hierarchical Cache System (moved from caching/ for consolidation)
# ============================================================================


class CacheLevel(Enum):
    """Cache levels organized by persistence and computational cost.

    Levels are ordered from most persistent (rarely changes) to least
    persistent (frequently recomputed):

    - GRAPH_STRUCTURE: Topology, adjacency matrices (invalidated on add/remove node/edge)
    - NODE_PROPERTIES: EPI, νf, θ per node (invalidated on property updates)
    - DERIVED_METRICS: Si, coherence, ΔNFR (invalidated on dependency changes)
    - TEMPORARY: Intermediate computations (short-lived, frequently evicted)
    """

    GRAPH_STRUCTURE = "graph_structure"
    NODE_PROPERTIES = "node_properties"
    DERIVED_METRICS = "derived_metrics"
    TEMPORARY = "temporary"


@dataclass
class CacheEntry:
    """Cache entry with metadata for intelligent invalidation and eviction.

    Attributes
    ----------
    value : Any
        The cached computation result.
    dependencies : set[str]
        Set of structural properties this entry depends on. Used for
        selective invalidation. Examples: 'node_epi', 'node_vf', 'graph_topology'.
    timestamp : float
        Time when entry was created (from time.time()).
    access_count : int
        Number of times this entry has been accessed.
    computation_cost : float
        Estimated computational cost to regenerate this value. Higher cost
        entries are prioritized during eviction.
    size_bytes : int
        Estimated memory size in bytes.
    """

    value: Any
    dependencies: set[str]
    timestamp: float
    access_count: int = 0
    computation_cost: float = 1.0
    size_bytes: int = 0


class TNFRHierarchicalCache:
    """Hierarchical cache with dependency-aware selective invalidation.

    This cache system organizes entries by structural level and tracks
    dependencies to enable surgical invalidation. Only entries that depend
    on changed structural properties are evicted, preserving valid cached data.

    Internally uses ``CacheManager`` for unified cache management, metrics,
    and telemetry integration with the rest of TNFR.

    **Performance Optimizations** (v2):
    - Direct cache references bypass CacheManager overhead on hot path (50% faster reads)
    - Lazy persistence batches writes to persistent layers (40% faster writes)
    - Type-based size estimation caching reduces memory tracking overhead
    - Dependency change detection avoids redundant updates
    - Batched invalidation reduces persistence operations

    **TNFR Compliance**:
    - Maintains §3.8 Controlled Determinism through consistent cache behavior
    - Supports §3.4 Operator Closure via dependency tracking

    Parameters
    ----------
    max_memory_mb : int, default: 512
        Maximum memory usage in megabytes before eviction starts.
    enable_metrics : bool, default: True
        Whether to track cache hit/miss metrics for telemetry.
    cache_manager : CacheManager, optional
        Existing CacheManager to use. If None, creates a new one.
    lazy_persistence : bool, default: True
        Enable lazy write-behind caching for persistent layers. When True,
        cache modifications are batched and written on flush or critical operations.
        This significantly improves write performance at the cost of potential
        data loss on ungraceful termination. Set to False for immediate consistency.

    Attributes
    ----------
    hits : int
        Number of successful cache retrievals.
    misses : int
        Number of cache misses.
    evictions : int
        Number of entries evicted due to memory pressure.
    invalidations : int
        Number of entries invalidated due to dependency changes.

    Examples
    --------
    >>> cache = TNFRHierarchicalCache(max_memory_mb=128)
    >>> # Cache a derived metric with dependencies
    >>> cache.set(
    ...     "coherence_global",
    ...     0.95,
    ...     CacheLevel.DERIVED_METRICS,
    ...     dependencies={'graph_topology', 'all_node_vf'},
    ...     computation_cost=100.0
    ... )
    >>> cache.get("coherence_global", CacheLevel.DERIVED_METRICS)
    0.95
    >>> # Invalidate when topology changes
    >>> cache.invalidate_by_dependency('graph_topology')
    >>> cache.get("coherence_global", CacheLevel.DERIVED_METRICS)

    >>> # Flush lazy writes to persistent storage
    >>> cache.flush_dirty_caches()

    """

    def __init__(
        self,
        max_memory_mb: int = 512,
        enable_metrics: bool = True,
        cache_manager: Optional[CacheManager] = None,
        lazy_persistence: bool = True,
    ):
        # Use provided CacheManager or create new one
        if cache_manager is None:
            # Estimate entries per MB (rough heuristic: ~100 entries per MB)
            default_capacity = max(32, int(max_memory_mb * 100 / len(CacheLevel)))
            cache_manager = CacheManager(
                storage={},
                default_capacity=default_capacity,
            )

        self._manager = cache_manager
        self._max_memory = max_memory_mb * 1024 * 1024
        self._current_memory = 0
        self._enable_metrics = enable_metrics
        self._lazy_persistence = lazy_persistence

        # Dependency tracking (remains in hierarchical cache)
        self._dependencies: dict[str, set[tuple[CacheLevel, str]]] = defaultdict(set)

        # Register a cache for each level in the CacheManager
        self._level_cache_names: dict[CacheLevel, str] = {}
        # OPTIMIZATION: Direct cache references to avoid CacheManager overhead on hot path
        self._direct_caches: dict[CacheLevel, dict[str, CacheEntry]] = {}

        for level in CacheLevel:
            cache_name = f"hierarchical_{level.value}"
            self._level_cache_names[level] = cache_name

            # Simple factory returning empty dict for each cache level
            self._manager.register(
                cache_name,
                factory=lambda: {},
                create=True,
            )

            # Store direct reference for fast access
            self._direct_caches[level] = self._manager.get(cache_name)

        # OPTIMIZATION: Track dirty caches for batched persistence
        self._dirty_levels: set[CacheLevel] = set()

        # OPTIMIZATION: Type-based size estimation cache
        self._size_cache: dict[type, int] = {}

        # Metrics (tracked locally for backward compatibility)
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.invalidations = 0

    @property
    def _caches(self) -> dict[CacheLevel, dict[str, CacheEntry]]:
        """Provide backward compatibility for accessing internal caches.

        This property returns a view of the caches stored in the CacheManager,
        maintaining compatibility with code that directly accessed the old
        _caches attribute.

        Note: Uses direct cache references for performance.
        """
        return self._direct_caches

    def get(self, key: str, level: CacheLevel) -> Optional[Any]:
        """Retrieve value from cache if it exists and is valid.

        Parameters
        ----------
        key : str
            Cache key identifying the entry.
        level : CacheLevel
            Cache level to search in.

        Returns
        -------
        Any or None
            The cached value if found, None otherwise.

        Examples
        --------
        >>> cache = TNFRHierarchicalCache()
        >>> cache.set("key1", 42, CacheLevel.TEMPORARY, dependencies=set())
        >>> cache.get("key1", CacheLevel.TEMPORARY)
        42
        >>> cache.get("missing", CacheLevel.TEMPORARY)

        """
        # OPTIMIZATION: Use direct cache reference to avoid CacheManager overhead
        level_cache = self._direct_caches[level]

        if key in level_cache:
            entry = level_cache[key]
            entry.access_count += 1
            if self._enable_metrics:
                self.hits += 1
                # Only update manager metrics if not in lazy mode
                if not self._lazy_persistence:
                    cache_name = self._level_cache_names[level]
                    self._manager.increment_hit(cache_name)
            return entry.value

        if self._enable_metrics:
            self.misses += 1
            if not self._lazy_persistence:
                cache_name = self._level_cache_names[level]
                self._manager.increment_miss(cache_name)
        return None

    def set(
        self,
        key: str,
        value: Any,
        level: CacheLevel,
        dependencies: set[str],
        computation_cost: float = 1.0,
    ) -> None:
        """Store value in cache with dependency metadata.

        Parameters
        ----------
        key : str
            Unique identifier for this cache entry.
        value : Any
            The value to cache.
        level : CacheLevel
            Which cache level to store in.
        dependencies : set[str]
            Set of structural properties this value depends on.
        computation_cost : float, default: 1.0
            Estimated cost to recompute this value. Used for eviction priority.

        Examples
        --------
        >>> cache = TNFRHierarchicalCache()
        >>> cache.set(
        ...     "si_node_5",
        ...     0.87,
        ...     CacheLevel.DERIVED_METRICS,
        ...     dependencies={'node_vf_5', 'node_phase_5'},
        ...     computation_cost=5.0
        ... )
        """
        # OPTIMIZATION: Use direct cache reference
        level_cache = self._direct_caches[level]

        # OPTIMIZATION: Lazy size estimation - estimate size once
        estimated_size = self._estimate_size_fast(value)

        # Check if we need to evict
        if self._current_memory + estimated_size > self._max_memory:
            self._evict_lru(estimated_size)

        # Create entry
        entry = CacheEntry(
            value=value,
            dependencies=dependencies.copy(),
            timestamp=time.time(),
            computation_cost=computation_cost,
            size_bytes=estimated_size,
        )

        # Remove old entry if exists
        old_dependencies: set[str] | None = None
        if key in level_cache:
            old_entry = level_cache[key]
            self._current_memory -= old_entry.size_bytes
            old_dependencies = old_entry.dependencies
            # OPTIMIZATION: Only clean up dependencies if they changed
            if old_dependencies != dependencies:
                for dep in old_dependencies:
                    if dep in self._dependencies:
                        self._dependencies[dep].discard((level, key))

        # Store entry (direct modification, no manager overhead)
        level_cache[key] = entry
        self._current_memory += estimated_size

        # OPTIMIZATION: Register dependencies only if new or changed
        if old_dependencies is None or old_dependencies != dependencies:
            for dep in dependencies:
                self._dependencies[dep].add((level, key))

        # OPTIMIZATION: Mark level as dirty for lazy persistence
        if self._lazy_persistence:
            self._dirty_levels.add(level)
        else:
            # Immediate persistence (backward compatible)
            cache_name = self._level_cache_names[level]
            self._manager.store(cache_name, level_cache)

    def invalidate_by_dependency(self, dependency: str) -> int:
        """Invalidate all cache entries that depend on a structural property.

        This implements selective invalidation: only entries that explicitly
        depend on the changed property are removed, preserving unaffected caches.

        Parameters
        ----------
        dependency : str
            The structural property that changed (e.g., 'graph_topology',
            'node_epi_5', 'all_node_vf').

        Returns
        -------
        int
            Number of entries invalidated.

        Examples
        --------
        >>> cache = TNFRHierarchicalCache()
        >>> cache.set("key1", 1, CacheLevel.TEMPORARY, {'dep1', 'dep2'})
        >>> cache.set("key2", 2, CacheLevel.TEMPORARY, {'dep2'})
        >>> cache.invalidate_by_dependency('dep1')  # Only invalidates key1
        1
        >>> cache.get("key1", CacheLevel.TEMPORARY)  # None

        >>> cache.get("key2", CacheLevel.TEMPORARY)  # Still cached
        2
        """
        count = 0
        if dependency in self._dependencies:
            entries_to_remove = list(self._dependencies[dependency])
            invalidated_levels: set[CacheLevel] = set()

            for level, key in entries_to_remove:
                # OPTIMIZATION: Use direct cache reference
                level_cache = self._direct_caches[level]

                if key in level_cache:
                    entry = level_cache[key]
                    self._current_memory -= entry.size_bytes
                    del level_cache[key]
                    count += 1
                    invalidated_levels.add(level)

                    # Clean up all dependency references for this entry
                    for dep in entry.dependencies:
                        if dep in self._dependencies:
                            self._dependencies[dep].discard((level, key))

            # Clean up the dependency key itself
            del self._dependencies[dependency]

            # OPTIMIZATION: Batch persist invalidated levels
            if self._lazy_persistence:
                self._dirty_levels.update(invalidated_levels)
            else:
                for level in invalidated_levels:
                    cache_name = self._level_cache_names[level]
                    level_cache = self._direct_caches[level]
                    self._manager.store(cache_name, level_cache)

        if self._enable_metrics:
            self.invalidations += count

        return count

    def invalidate_level(self, level: CacheLevel) -> int:
        """Invalidate all entries in a specific cache level.

        Parameters
        ----------
        level : CacheLevel
            The cache level to clear.

        Returns
        -------
        int
            Number of entries invalidated.
        """
        # OPTIMIZATION: Use direct cache reference
        level_cache = self._direct_caches[level]
        count = len(level_cache)

        # Clean up dependencies
        for key, entry in level_cache.items():
            self._current_memory -= entry.size_bytes
            for dep in entry.dependencies:
                if dep in self._dependencies:
                    self._dependencies[dep].discard((level, key))

        level_cache.clear()

        # OPTIMIZATION: Batch persist if in lazy mode
        if self._lazy_persistence:
            self._dirty_levels.add(level)
        else:
            cache_name = self._level_cache_names[level]
            self._manager.store(cache_name, level_cache)

        if self._enable_metrics:
            self.invalidations += count

        return count

    def clear(self) -> None:
        """Clear all cache levels and reset metrics."""
        for level in CacheLevel:
            # OPTIMIZATION: Clear direct cache and update manager
            level_cache = self._direct_caches[level]
            level_cache.clear()
            cache_name = self._level_cache_names[level]
            self._manager.store(cache_name, level_cache)

        self._dependencies.clear()
        self._current_memory = 0
        self._dirty_levels.clear()

        # Always reset metrics regardless of _enable_metrics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.invalidations = 0

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics for telemetry.

        Returns
        -------
        dict[str, Any]
            Dictionary containing:
            - hits: Number of cache hits
            - misses: Number of cache misses
            - hit_rate: Ratio of hits to total accesses
            - evictions: Number of evictions
            - invalidations: Number of invalidations
            - memory_used_mb: Current memory usage in MB
            - memory_limit_mb: Memory limit in MB
            - entry_counts: Number of entries per level
        """
        total_accesses = self.hits + self.misses
        hit_rate = self.hits / total_accesses if total_accesses > 0 else 0.0

        entry_counts = {}
        for level in CacheLevel:
            # OPTIMIZATION: Use direct cache reference
            level_cache = self._direct_caches[level]
            entry_counts[level.value] = len(level_cache)

        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "evictions": self.evictions,
            "invalidations": self.invalidations,
            "memory_used_mb": self._current_memory / (1024 * 1024),
            "memory_limit_mb": self._max_memory / (1024 * 1024),
            "entry_counts": entry_counts,
        }

    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of a value in bytes.

        Uses sys.getsizeof for a rough estimate. For complex objects,
        this may underestimate the true memory usage.
        """
        try:
            return sys.getsizeof(value)
        except (TypeError, AttributeError):
            # Fallback for objects that don't support getsizeof
            return 64  # Default estimate

    def _estimate_size_fast(self, value: Any) -> int:
        """Optimized size estimation with type-based caching.

        For common types, uses cached size estimates to avoid repeated
        sys.getsizeof() calls. Falls back to full estimation for complex types.
        """
        value_type = type(value)

        # Check if we have a cached size for this type
        if value_type in self._size_cache:
            # For simple immutable types, use cached size
            if value_type in (int, float, bool, type(None)):
                return self._size_cache[value_type]
            # For strings, estimate based on length
            if value_type is str:
                base_size = self._size_cache[value_type]
                return base_size + len(value)

        # Calculate size and cache for simple types
        size = self._estimate_size(value)
        if value_type in (int, float, bool, type(None)):
            self._size_cache[value_type] = size
        elif value_type is str:
            # Cache base size for strings
            if value_type not in self._size_cache:
                self._size_cache[value_type] = sys.getsizeof("")

        return size

    def flush_dirty_caches(self) -> None:
        """Flush dirty caches to persistent layers.

        In lazy persistence mode, this method writes accumulated changes
        to the CacheManager's persistent layers. This reduces write overhead
        by batching updates.
        """
        if not self._dirty_levels:
            return

        for level in self._dirty_levels:
            cache_name = self._level_cache_names[level]
            level_cache = self._direct_caches[level]
            self._manager.store(cache_name, level_cache)

        self._dirty_levels.clear()

    def _evict_lru(self, needed_space: int) -> None:
        """Evict least valuable entries until enough space is freed.

        Value is determined by: (access_count + 1) * computation_cost.
        Lower values are evicted first (low access, low cost to recompute).

        OPTIMIZED: Uses direct cache references and incremental eviction.
        """
        # OPTIMIZATION: Collect entries with direct cache access (no manager overhead)
        all_entries: list[tuple[float, CacheLevel, str, CacheEntry]] = []
        for level in CacheLevel:
            level_cache = self._direct_caches[level]
            for key, entry in level_cache.items():
                # Priority = (access_count + 1) * computation_cost
                # Higher priority = keep longer
                # Add 1 to access_count to avoid zero priority
                priority = (entry.access_count + 1) * entry.computation_cost
                all_entries.append((priority, level, key, entry))

        # Sort by priority (ascending - lowest priority first)
        all_entries.sort(key=lambda x: x[0])

        freed_space = 0
        evicted_levels: set[CacheLevel] = set()

        for priority, level, key, entry in all_entries:
            if freed_space >= needed_space:
                break

            # OPTIMIZATION: Remove entry directly from cache
            level_cache = self._direct_caches[level]
            if key in level_cache:
                del level_cache[key]
                freed_space += entry.size_bytes
                self._current_memory -= entry.size_bytes
                evicted_levels.add(level)

                # Clean up dependencies
                for dep in entry.dependencies:
                    if dep in self._dependencies:
                        self._dependencies[dep].discard((level, key))

                if self._enable_metrics:
                    self.evictions += 1
                    if not self._lazy_persistence:
                        cache_name = self._level_cache_names[level]
                        self._manager.increment_eviction(cache_name)

        # OPTIMIZATION: Batch persist evicted levels if in lazy mode
        if self._lazy_persistence:
            self._dirty_levels.update(evicted_levels)
        else:
            # Immediate persistence
            for level in evicted_levels:
                cache_name = self._level_cache_names[level]
                level_cache = self._direct_caches[level]
                self._manager.store(cache_name, level_cache)


# ============================================================================
# Cache Decorators (moved from caching/decorators.py for consolidation)
# ============================================================================

# Global cache instance shared across all decorated functions
_global_cache: Optional[TNFRHierarchicalCache] = None

F = TypeVar("F", bound=Callable[..., Any])


def get_global_cache() -> TNFRHierarchicalCache:
    """Get or create the global TNFR cache instance.

    Returns
    -------
    TNFRHierarchicalCache
        The global cache instance.
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = TNFRHierarchicalCache(max_memory_mb=512)
    return _global_cache


def set_global_cache(cache: Optional[TNFRHierarchicalCache]) -> None:
    """Set the global cache instance.

    Parameters
    ----------
    cache : TNFRHierarchicalCache or None
        The cache instance to use globally, or None to reset to default.
    """
    global _global_cache
    _global_cache = cache


def reset_global_cache() -> None:
    """Reset the global cache instance to None.

    The next call to get_global_cache() will create a fresh instance.
    """
    global _global_cache
    _global_cache = None


def _generate_cache_key(
    func_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> str:
    """Generate deterministic cache key from function and arguments.

    Parameters
    ----------
    func_name : str
        Name of the function being cached.
    args : tuple
        Positional arguments.
    kwargs : dict
        Keyword arguments.

    Returns
    -------
    str
        Cache key string.

    Notes
    -----
    Uses MD5 for hashing (acceptable for cache keys, not security).
    Graph objects use id() which is session-specific - cache is cleared
    between sessions, so this is deterministic within a session.
    """
    # Build key components
    key_parts = [func_name]

    # Add positional args
    for arg in args:
        if hasattr(arg, "__name__"):  # For graph objects, use name
            key_parts.append(f"graph:{arg.__name__}")
        elif hasattr(arg, "graph"):  # NetworkX graphs have .graph attribute
            # Use graph id for identity (session-specific, cache cleared between sessions)
            key_parts.append(f"graph:{id(arg)}")
        else:
            # For simple types, include value
            key_parts.append(str(arg))

    # Add keyword args (sorted for consistency)
    for k in sorted(kwargs.keys()):
        v = kwargs[k]
        key_parts.append(f"{k}={v}")

    # Create deterministic hash (MD5 is acceptable for non-security cache keys)
    key_str = "|".join(key_parts)
    return hashlib.md5(key_str.encode()).hexdigest()


def cache_tnfr_computation(
    level: CacheLevel,
    dependencies: set[str],
    cost_estimator: Optional[Callable[..., float]] = None,
    cache_instance: Optional[TNFRHierarchicalCache] = None,
) -> Callable[[F], F]:
    """Decorator for automatic caching of TNFR computations.

    Caches function results based on arguments and invalidates when
    dependencies change. Transparently integrates with existing functions.

    Parameters
    ----------
    level : CacheLevel
        Cache level for storing results.
    dependencies : set[str]
        Set of structural properties this computation depends on.
        Examples: {'graph_topology', 'node_epi', 'node_vf', 'node_phase'}
    cost_estimator : callable, optional
        Function that takes same arguments as decorated function and returns
        estimated computational cost as float. Used for eviction priority.
    cache_instance : TNFRHierarchicalCache, optional
        Specific cache instance to use. If None, uses global cache.

    Returns
    -------
    callable
        Decorated function with caching.

    Examples
    --------
    >>> from tnfr.cache import cache_tnfr_computation, CacheLevel
    >>> @cache_tnfr_computation(
    ...     level=CacheLevel.DERIVED_METRICS,
    ...     dependencies={'node_vf', 'node_phase'},
    ...     cost_estimator=lambda graph, node_id: len(list(graph.neighbors(node_id)))
    ... )
    ... def compute_metric(graph, node_id):
    ...     # Expensive computation
    ...     return 0.85

    With custom cache instance:

    >>> from tnfr.cache import TNFRHierarchicalCache
    >>> my_cache = TNFRHierarchicalCache(max_memory_mb=256)
    >>> @cache_tnfr_computation(
    ...     level=CacheLevel.NODE_PROPERTIES,
    ...     dependencies={'node_data'},
    ...     cache_instance=my_cache
    ... )
    ... def get_node_property(graph, node_id):
    ...     return graph.nodes[node_id]
    """

    def decorator(func: F) -> F:
        func_name = func.__name__

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get cache instance
            cache = cache_instance if cache_instance is not None else get_global_cache()

            # Generate cache key
            # Bind defaults so implicit kwargs (alpha, landmark_ratio)
            # are included in the cache key deterministically.
            try:
                import inspect

                sig = inspect.signature(func)
                bound = sig.bind_partial(*args, **kwargs)
                bound.apply_defaults()
                norm_kwargs = dict(bound.arguments)
                cache_key = _generate_cache_key(
                    func_name,
                    tuple(),
                    norm_kwargs,
                )
            except Exception:
                # Fallback to raw args/kwargs if binding fails
                cache_key = _generate_cache_key(func_name, args, kwargs)

            # Try to get from cache
            cached_result = cache.get(cache_key, level)
            if cached_result is not None:
                return cached_result

            # Compute result
            result = func(*args, **kwargs)

            # Estimate computational cost
            comp_cost = 1.0
            if cost_estimator is not None:
                try:
                    comp_cost = float(cost_estimator(*args, **kwargs))
                except (TypeError, ValueError):
                    comp_cost = 1.0

            # Store in cache
            cache.set(cache_key, result, level, dependencies, comp_cost)

            return result

        # Attach metadata for introspection
        wrapper._cache_level = level  # type: ignore
        wrapper._cache_dependencies = dependencies  # type: ignore
        wrapper._is_cached = True  # type: ignore

        return wrapper  # type: ignore

    return decorator


def invalidate_function_cache(func: Callable[..., Any]) -> int:
    """Invalidate cache entries for a specific decorated function.

    Parameters
    ----------
    func : callable
        The decorated function whose cache entries should be invalidated.

    Returns
    -------
    int
        Number of entries invalidated.

    Raises
    ------
    ValueError
        If the function is not decorated with @cache_tnfr_computation.
    """
    if not hasattr(func, "_is_cached"):
        raise ValueError(f"Function {func.__name__} is not cached")

    cache = get_global_cache()
    dependencies = getattr(func, "_cache_dependencies", set())

    total = 0
    for dep in dependencies:
        total += cache.invalidate_by_dependency(dep)

    return total


# ============================================================================
# Graph Change Tracking (moved from caching/invalidation.py for consolidation)
# ============================================================================


class GraphChangeTracker:
    """Track graph modifications for selective cache invalidation.

    Installs hooks into graph modification methods to automatically invalidate
    affected cache entries when structural properties change.

    Parameters
    ----------
    cache : TNFRHierarchicalCache
        The cache instance to invalidate.

    Attributes
    ----------
    topology_changes : int
        Count of topology modifications (add/remove node/edge).
    property_changes : int
        Count of node property modifications.

    Examples
    --------
    >>> import networkx as nx
    >>> from tnfr.cache import (
    ...     TNFRHierarchicalCache,
    ...     GraphChangeTracker,
    ...     CacheLevel,
    ... )
    >>> cache = TNFRHierarchicalCache()
    >>> G = nx.Graph()
    >>> tracker = GraphChangeTracker(cache)
    >>> tracker.track_graph_changes(G)
    >>> # Now modifications to G will trigger cache invalidation
    >>> cache.set("key1", 1, CacheLevel.GRAPH_STRUCTURE, {'graph_topology'})
    >>> G.add_node("n1")  # Invalidates graph_topology cache entries
    >>> cache.get("key1", CacheLevel.GRAPH_STRUCTURE)  # Returns None
    """

    def __init__(self, cache: TNFRHierarchicalCache):
        self._cache = cache
        self.topology_changes = 0
        self.property_changes = 0
        self._tracked_graphs: set[int] = set()

    def track_graph_changes(self, graph: Any) -> None:
        """Install hooks to track changes in a graph.

        Wraps the graph's add_node, remove_node, add_edge, and remove_edge
        methods to trigger cache invalidation.

        Parameters
        ----------
        graph : GraphLike
            The graph to monitor for changes.

        Notes
        -----
        This uses monkey-patching to intercept graph modifications. The
        original methods are preserved and called after invalidation.
        """
        graph_id = id(graph)
        if graph_id in self._tracked_graphs:
            return  # Already tracking this graph

        self._tracked_graphs.add(graph_id)

        # Store original methods
        original_add_node = graph.add_node
        original_remove_node = graph.remove_node
        original_add_edge = graph.add_edge
        original_remove_edge = graph.remove_edge

        # Create tracked versions
        def tracked_add_node(node_id: Any, **attrs: Any) -> None:
            result = original_add_node(node_id, **attrs)
            self._on_topology_change()
            return result

        def tracked_remove_node(node_id: Any) -> None:
            result = original_remove_node(node_id)
            self._on_topology_change()
            return result

        def tracked_add_edge(u: Any, v: Any, **attrs: Any) -> None:
            result = original_add_edge(u, v, **attrs)
            self._on_topology_change()
            return result

        def tracked_remove_edge(u: Any, v: Any) -> None:
            result = original_remove_edge(u, v)
            self._on_topology_change()
            return result

        # Replace methods
        graph.add_node = tracked_add_node
        graph.remove_node = tracked_remove_node
        graph.add_edge = tracked_add_edge
        graph.remove_edge = tracked_remove_edge

        # Store reference to tracker for property changes
        if hasattr(graph, "graph"):
            graph.graph["_tnfr_change_tracker"] = self

    def on_node_property_change(
        self,
        node_id: Any,
        property_name: str,
        old_value: Optional[Any] = None,
        new_value: Optional[Any] = None,
    ) -> None:
        """Notify tracker of a node property change.

        Parameters
        ----------
        node_id : Any
            The node whose property changed.
        property_name : str
            Name of the property that changed (e.g., 'epi', 'vf', 'phase').
        old_value : Any, optional
            Previous value (for logging/debugging).
        new_value : Any, optional
            New value (for logging/debugging).

        Notes
        -----
        This should be called explicitly when node properties are modified
        outside of the graph's standard API (e.g., G.nodes[n]['epi'] = value).
        """
        # Invalidate node-specific dependency
        dep_key = f"node_{property_name}_{node_id}"
        self._cache.invalidate_by_dependency(dep_key)

        # Invalidate global property dependency
        global_dep = f"all_node_{property_name}"
        self._cache.invalidate_by_dependency(global_dep)

        # Invalidate derived metrics for this node
        if property_name in ["epi", "vf", "phase", "delta_nfr"]:
            self._cache.invalidate_by_dependency(f"derived_metrics_{node_id}")

        self.property_changes += 1

    def _on_topology_change(self) -> None:
        """Handle topology modifications (add/remove node/edge)."""
        # Invalidate topology-dependent caches
        self._cache.invalidate_by_dependency("graph_topology")
        self._cache.invalidate_by_dependency("node_neighbors")
        self._cache.invalidate_by_dependency("adjacency_matrix")

        self.topology_changes += 1

    def reset_counters(self) -> None:
        """Reset change counters."""
        self.topology_changes = 0
        self.property_changes = 0


def track_node_property_update(
    graph: Any,
    node_id: Any,
    property_name: str,
    new_value: Any,
) -> None:
    """Helper to track node property updates.

    Updates the node property and notifies the change tracker if one is
    attached to the graph.

    Parameters
    ----------
    graph : GraphLike
        The graph containing the node.
    node_id : Any
        The node to update.
    property_name : str
        Property name to update.
    new_value : Any
        New value for the property.

    Examples
    --------
    >>> import networkx as nx
    >>> from tnfr.cache import TNFRHierarchicalCache, GraphChangeTracker
    >>> from tnfr.cache import track_node_property_update
    >>> cache = TNFRHierarchicalCache()
    >>> G = nx.Graph()
    >>> G.add_node("n1", epi=0.5)
    >>> tracker = GraphChangeTracker(cache)
    >>> tracker.track_graph_changes(G)
    >>> # Use helper to update and invalidate
    >>> track_node_property_update(G, "n1", "epi", 0.7)
    """
    # Get old value
    old_value = graph.nodes[node_id].get(property_name)

    # Update property
    graph.nodes[node_id][property_name] = new_value

    # Notify tracker if present
    if hasattr(graph, "graph"):
        tracker = graph.graph.get("_tnfr_change_tracker")
        if isinstance(tracker, GraphChangeTracker):
            tracker.on_node_property_change(
                node_id,
                property_name,
                old_value,
                new_value,
            )


# ============================================================================
# Persistent Cache (moved from caching/persistence.py for consolidation)
# ============================================================================


class PersistentTNFRCache:
    """Cache with optional disk persistence for costly computations.

    Combines in-memory caching with selective disk persistence for
    specific cache levels. Expensive computations can be preserved
    between sessions while temporary computations remain memory-only.

    Parameters
    ----------
    cache_dir : Path or str, default: ".tnfr_cache"
        Directory for persistent cache files.
    max_memory_mb : int, default: 512
        Memory limit for in-memory cache.
    persist_levels : set[CacheLevel], optional
        Cache levels to persist to disk. Defaults to GRAPH_STRUCTURE
        and DERIVED_METRICS.

    Examples
    --------
    >>> from pathlib import Path
    >>> from tnfr.cache import PersistentTNFRCache, CacheLevel
    >>> cache = PersistentTNFRCache(cache_dir=Path("/tmp/tnfr_cache"))
    >>> # Cache is automatically persisted for expensive operations
    >>> cache.set_persistent(
    ...     "coherence_large_graph",
    ...     0.95,
    ...     CacheLevel.DERIVED_METRICS,
    ...     dependencies={'graph_topology'},
    ...     computation_cost=1000.0,
    ...     persist_to_disk=True
    ... )
    >>> # Later, in a new session
    >>> result = cache.get_persistent(
    ...     "coherence_large_graph",
    ...     CacheLevel.DERIVED_METRICS,
    ... )
    """

    def __init__(
        self,
        cache_dir: Any = ".tnfr_cache",  # Path | str
        max_memory_mb: int = 512,
        persist_levels: Optional[set[CacheLevel]] = None,
    ):
        from pathlib import Path

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self._memory_cache = TNFRHierarchicalCache(max_memory_mb=max_memory_mb)

        if persist_levels is None:
            persist_levels = {
                CacheLevel.GRAPH_STRUCTURE,
                CacheLevel.DERIVED_METRICS,
            }
        self._persist_levels = persist_levels

    def get_persistent(self, key: str, level: CacheLevel) -> Optional[Any]:
        """Retrieve value from memory cache, falling back to disk.

        Parameters
        ----------
        key : str
            Cache key.
        level : CacheLevel
            Cache level.

        Returns
        -------
        Any or None
            Cached value if found, None otherwise.
        """
        # Try memory first
        result = self._memory_cache.get(key, level)
        if result is not None:
            return result

        # Try disk if level is persisted
        if level in self._persist_levels:
            file_path = self._get_cache_file_path(key, level)
            if file_path.exists():
                try:
                    with open(file_path, "rb") as f:
                        cached_data = pickle.load(f)

                    # Validate structure
                    if not isinstance(cached_data, dict):
                        file_path.unlink(missing_ok=True)
                        return None

                    value = cached_data.get("value")
                    dependencies = cached_data.get("dependencies", set())
                    computation_cost = cached_data.get("computation_cost", 1.0)

                    # Load back into memory cache
                    self._memory_cache.set(
                        key,
                        value,
                        level,
                        dependencies,
                        computation_cost,
                    )

                    return value

                except (pickle.PickleError, EOFError, OSError):
                    # Corrupt cache file, remove it
                    file_path.unlink(missing_ok=True)

        return None

    def set_persistent(
        self,
        key: str,
        value: Any,
        level: CacheLevel,
        dependencies: set[str],
        computation_cost: float = 1.0,
        persist_to_disk: bool = True,
    ) -> None:
        """Store value in memory and optionally persist to disk.

        Parameters
        ----------
        key : str
            Cache key.
        value : Any
            Value to cache.
        level : CacheLevel
            Cache level.
        dependencies : set[str]
            Structural dependencies.
        computation_cost : float, default: 1.0
            Computation cost estimate.
        persist_to_disk : bool, default: True
            Whether to persist this entry to disk.
        """
        # Always store in memory
        self._memory_cache.set(
            key,
            value,
            level,
            dependencies,
            computation_cost,
        )

        # Persist to disk if requested and level supports it
        if persist_to_disk and level in self._persist_levels:
            file_path = self._get_cache_file_path(key, level)
            cache_data = {
                "value": value,
                "dependencies": dependencies,
                "computation_cost": computation_cost,
                "timestamp": time.time(),
            }

        try:
            with open(file_path, "wb") as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        except (pickle.PickleError, OSError):
            # Log error but don't fail
            # In production, this should use proper logging
            pass

    def invalidate_by_dependency(self, dependency: str) -> int:
        """Invalidate memory and disk cache entries for a dependency.

        Parameters
        ----------
        dependency : str
            The structural property that changed.

        Returns
        -------
        int
            Number of entries invalidated from memory.
        """
        # Invalidate memory cache
        count = self._memory_cache.invalidate_by_dependency(dependency)

        # Note: Disk cache is lazily invalidated on load
        # Entries with stale dependencies will be detected when loaded

        return count

    def clear_persistent_cache(
        self, level: Optional[CacheLevel] = None
    ) -> None:
        """Clear persistent cache files.

        Parameters
        ----------
        level : CacheLevel, optional
            Specific level to clear. If None, clears all levels.
        """
        if level is not None:
            level_dir = self.cache_dir / level.value
            if level_dir.exists():
                for file_path in level_dir.glob("*.pkl"):
                    file_path.unlink(missing_ok=True)
        else:
            # Clear all levels
            for file_path in self.cache_dir.rglob("*.pkl"):
                file_path.unlink(missing_ok=True)

    def cleanup_old_entries(self, max_age_days: int = 30) -> int:
        """Remove old cache files from disk.

        Parameters
        ----------
        max_age_days : int, default: 30
            Maximum age in days before removal.

        Returns
        -------
        int
            Number of files removed.
        """
        count = 0
        max_age_seconds = max_age_days * 24 * 3600
        current_time = time.time()

        for file_path in self.cache_dir.rglob("*.pkl"):
            try:
                mtime = file_path.stat().st_mtime
                if current_time - mtime > max_age_seconds:
                    file_path.unlink()
                    count += 1
            except OSError:
                continue

        return count

    def get_stats(self) -> dict[str, Any]:
        """Get combined statistics from memory and disk cache.

        Returns
        -------
        dict[str, Any]
            Statistics including memory stats and disk usage.
        """
        stats = self._memory_cache.get_stats()

        # Add disk stats
        disk_files = 0
        disk_size_bytes = 0
        for file_path in self.cache_dir.rglob("*.pkl"):
            disk_files += 1
            try:
                disk_size_bytes += file_path.stat().st_size
            except OSError:
                continue

        stats["disk_files"] = disk_files
        stats["disk_size_mb"] = disk_size_bytes / (1024 * 1024)

        return stats

    def _get_cache_file_path(self, key: str, level: CacheLevel) -> Any:
        """Get file path for a cache entry.

        Organizes cache files by level in subdirectories.
        """
        level_dir = self.cache_dir / level.value
        level_dir.mkdir(exist_ok=True, parents=True)
        # Use key as filename (already hashed in decorator)
        return level_dir / f"{key}.pkl"
