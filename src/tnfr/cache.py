"""Central cache registry infrastructure for TNFR services."""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
import pickle
import shelve
import threading
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import (
    Any,
    Callable,
    Generic,
    Hashable,
    Iterator,
    Mapping,
    MutableMapping,
    TypeVar,
    cast,
)

from cachetools import LRUCache

from .types import TimingContext

__all__ = [
    "CacheLayer",
    "CacheManager",
    "CacheCapacityConfig",
    "CacheStatistics",
    "InstrumentedLRUCache",
    "MappingCacheLayer",
    "RedisCacheLayer",
    "ShelveCacheLayer",
    "ManagedLRUCache",
    "prune_lock_mapping",
]


K = TypeVar("K", bound=Hashable)
V = TypeVar("V")

_logger = logging.getLogger(__name__)


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
    """Persistent cache layer backed by :mod:`shelve`."""

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
        self._shelf = shelve.open(path, flag=flag, protocol=self._protocol, writeback=writeback)
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
    """Distributed cache layer backed by a Redis client."""

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
            return pickle.loads(bytes(value))
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
                    logging.getLogger(__name__).exception(
                        "Cache metrics publisher failed for %s", name
                    )

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
