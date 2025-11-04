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
import logging
import pickle
import shelve
import threading
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
from dataclasses import dataclass, field
from functools import lru_cache
from time import perf_counter
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

import networkx as nx
from cachetools import LRUCache

from ..locking import get_lock
from ..types import GraphLike, NodeId, TimingContext, TNFRGraph
from .graph import get_graph, mark_dnfr_prep_dirty

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")
T = TypeVar("T")

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
)

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
        coherence states, numpy arrays). For untrusted inputs, implement
        alternative serialization or HMAC-based integrity validation.
    """

    def __init__(
        self,
        path: str,
        *,
        flag: str = "c",
        protocol: int | None = None,
        writeback: bool = False,
    ) -> None:
        self._path = path
        self._flag = flag
        self._protocol = pickle.HIGHEST_PROTOCOL if protocol is None else protocol
        # shelve module inherently uses pickle for serialization; security risks documented in class docstring
        self._shelf = shelve.open(path, flag=flag, protocol=self._protocol, writeback=writeback)  # nosec B301
        self._lock = threading.RLock()

    def load(self, name: str) -> Any:
        with self._lock:
            if name not in self._shelf:
                raise KeyError(name)
            return self._shelf[name]

    def store(self, name: str, value: Any) -> None:
        with self._lock:
            self._shelf[name] = value
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


class RedisCacheLayer(CacheLayer):
    """Distributed cache layer backed by a Redis client.

    .. warning::
        This layer uses :mod:`pickle` for serialization, which can deserialize
        arbitrary Python objects and execute code during deserialization.
        **Only cache trusted data** from controlled TNFR nodes. Ensure Redis
        uses authentication (AUTH command or ACL for Redis 6.0+) and network
        access controls. Never cache untrusted user input or external data.

        If Redis is compromised or contains tampered data, pickle deserialization
        executes arbitrary code. Use TLS for connections and implement HMAC
        validation for critical deployments.
    """

    def __init__(self, client: Any | None = None, *, namespace: str = "tnfr:cache") -> None:
        if client is None:
            try:  # pragma: no cover - import guarded for optional dependency
                import redis  # type: ignore
            except Exception as exc:  # pragma: no cover - defensive import
                raise RuntimeError("redis-py is required to initialise RedisCacheLayer") from exc
            client = redis.Redis()
        self._client = client
        self._namespace = namespace.rstrip(":") or "tnfr:cache"
        self._lock = threading.RLock()

    def _format_key(self, name: str) -> str:
        return f"{self._namespace}:{name}"

    def load(self, name: str) -> Any:
        key = self._format_key(name)
        with self._lock:
            value = self._client.get(key)
        if value is None:
            raise KeyError(name)
        if isinstance(value, (bytes, bytearray, memoryview)):
            # pickle from trusted Redis; documented security warning in class docstring
            return pickle.loads(bytes(value))  # nosec B301
        return value

    def store(self, name: str, value: Any) -> None:
        key = self._format_key(name)
        payload = value
        if not isinstance(value, (bytes, bytearray, memoryview)):
            payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
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
                    "cache layer delete failed for %s on %s", name, layer.__class__.__name__
                )

    def _load_from_layers(self, name: str, entry: _CacheEntry) -> Any:
        # Primary in-memory layer first for fast-path lookups.
        try:
            value = self._layers[0].load(name)
        except KeyError:
            value = None
        except Exception:  # pragma: no cover - defensive logging
            _logger.exception(
                "cache layer load failed for %s on %s", name, self._layers[0].__class__.__name__
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
                    "cache layer load failed for %s on %s", name, layer.__class__.__name__
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

    def register_metrics_publisher(
        self, publisher: Callable[[str, CacheStatistics], None]
    ) -> None:
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
        eviction_callbacks: (
            Iterable[Callable[[K, V], None]] | Callable[[K, V], None] | None
        ) = None,
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
        eviction_callbacks: (
            Iterable[Callable[[K, V], None]] | Callable[[K, V], None] | None
        ) = None,
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
                logger.exception(
                    "Cache layer close failed for %s", layer.__class__.__name__
                )


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
        for _, digest in sorted(
            (_node_repr_digest(n) for n in nodes), key=lambda x: x[0]
        ):
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

    graph[f"{key}_cache"] = NodeCache(
        checksum=checksum, nodes=nodes, sorted_nodes=sorted_nodes
    )
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
        nodes = _refresh_node_list_cache(
            G, graph, sort_nodes=sort_nodes, current_n=current_n
        )
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
        self._manager = manager or build_cache_manager(
            default_capacity=self._default_maxsize
        )
        self._state_key = state_key
        if not self._manager.has_override(self._state_key):
            self._manager.configure(
                overrides={self._state_key: self._default_maxsize}
            )
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
        self._manager = manager or build_cache_manager(
            default_capacity=self._default_max_entries
        )
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
            if (
                not isinstance(state, _CounterState)
                or force
                or state.max_entries != size
            ):
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
