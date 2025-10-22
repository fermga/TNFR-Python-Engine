"""Central cache registry infrastructure for TNFR services."""

from __future__ import annotations

import logging
import threading
from collections.abc import Collection, Iterable, Iterator, Mapping, MutableMapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Generic, Hashable, Sequence, TypeVar, cast

from cachetools import LRUCache  # type: ignore[import-untyped]

from .types import TimingContext

__all__ = [
    "CacheManager",
    "CacheCapacityConfig",
    "CacheStatistics",
    "InstrumentedLRUCache",
    "LockMapCleaner",
]


K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


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


class CacheManager:
    """Coordinate named caches guarded by per-entry locks."""

    _MISSING = object()

    def __init__(
        self,
        storage: MutableMapping[str, Any] | None = None,
        *,
        default_capacity: int | None = None,
        overrides: Mapping[str, int | None] | None = None,
    ) -> None:
        self._storage: MutableMapping[str, Any]
        if storage is None:
            self._storage = {}
        else:
            self._storage = storage
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
    ) -> None:
        """Register ``name`` with ``factory`` and optional lifecycle hooks."""

        if lock_factory is None:
            lock_factory = threading.RLock
        with self._registry_lock:
            entry = self._entries.get(name)
            if entry is None:
                entry = _CacheEntry(factory=factory, lock=lock_factory(), reset=reset)
                self._entries[name] = entry
            else:
                # Update hooks when re-registering the same cache name.
                entry.factory = factory
                entry.reset = reset
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
        entry = self._entries.get(name)
        if entry is None:
            raise KeyError(name)
        return entry.lock

    def names(self) -> Iterator[str]:
        with self._registry_lock:
            return iter(tuple(self._entries))

    def get(self, name: str, *, create: bool = True) -> Any:
        entry = self._entries.get(name)
        if entry is None:
            raise KeyError(name)
        with entry.lock:
            value = self._storage.get(name)
            if create and value is None:
                value = entry.factory()
                self._storage[name] = value
            return value

    def peek(self, name: str) -> Any:
        return self.get(name, create=False)

    def store(self, name: str, value: Any) -> None:
        entry = self._entries.get(name)
        if entry is None:
            raise KeyError(name)
        with entry.lock:
            self._storage[name] = value

    def update(
        self,
        name: str,
        updater: Callable[[Any], Any],
        *,
        create: bool = True,
    ) -> Any:
        entry = self._entries.get(name)
        if entry is None:
            raise KeyError(name)
        with entry.lock:
            current = self._storage.get(name)
            if create and current is None:
                current = entry.factory()
            new_value = updater(current)
            self._storage[name] = new_value
            return new_value

    def clear(self, name: str | None = None) -> None:
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
                current = self._storage.get(cache_name)
                new_value = None
                if entry.reset is not None:
                    new_value = entry.reset(current)
                if new_value is None:
                    try:
                        new_value = entry.factory()
                    except Exception:
                        self._storage.pop(cache_name, None)
                        continue
                self._storage[cache_name] = new_value

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
        metrics = self._ensure_metrics(name)
        with metrics.lock:
            metrics.misses += int(amount)
            if duration is not None:
                metrics.total_time += float(duration)
                metrics.timings += 1

    def increment_eviction(self, name: str, *, amount: int = 1) -> None:
        metrics = self._ensure_metrics(name)
        with metrics.lock:
            metrics.evictions += int(amount)

    def record_timing(self, name: str, duration: float) -> None:
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
        metrics = self._metrics.get(name)
        if metrics is None:
            return CacheStatistics()
        with metrics.lock:
            return metrics.snapshot()

    def iter_metrics(self) -> Iterator[tuple[str, CacheStatistics]]:
        with self._registry_lock:
            items = tuple(self._metrics.items())
        for name, metrics in items:
            with metrics.lock:
                yield name, metrics.snapshot()

    def aggregate_metrics(self) -> CacheStatistics:
        aggregate = CacheStatistics()
        for _, stats in self.iter_metrics():
            aggregate = aggregate.merge(stats)
        return aggregate

    def register_metrics_publisher(
        self, publisher: Callable[[str, CacheStatistics], None]
    ) -> None:
        with self._registry_lock:
            self._metrics_publishers.append(publisher)

    def publish_metrics(
        self,
        *,
        publisher: Callable[[str, CacheStatistics], None] | None = None,
    ) -> None:
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


class LockMapCleaner(Generic[K]):
    """Utility coordinating lock lifecycle with cache events."""

    def __init__(self, locks: MutableMapping[K, Any]) -> None:
        self._locks = locks

    @property
    def locks(self) -> MutableMapping[K, Any]:
        """Return the mutable mapping storing per-entry locks."""

        return self._locks

    def on_remove(self, key: K, _value: Any | None = None) -> None:
        """Drop the lock associated with ``key`` if present."""

        self._locks.pop(key, None)

    def prune(self, keys: Iterable[K]) -> None:
        """Remove locks for keys no longer present in ``keys``."""

        if isinstance(keys, Collection):
            candidates: Collection[K] = keys
        else:
            candidates = tuple(keys)
        for lock_key in list(self._locks.keys()):
            if lock_key not in candidates:
                self._locks.pop(lock_key, None)

    def clear(self) -> None:
        """Drop all tracked locks."""

        self._locks.clear()


class InstrumentedLRUCache(LRUCache[K, V], Generic[K, V]):
    """``LRUCache`` variant instrumented with telemetry and lock hooks."""

    _MISSING = object()

    def __init__(
        self,
        maxsize: int,
        *,
        lock_cleaner: LockMapCleaner[K] | None = None,
        telemetry: Sequence[tuple[CacheManager, str]] | tuple[CacheManager, str] | None = None,
        on_evict: Iterable[Callable[[K, V], None]] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(maxsize)
        self._lock_cleaner = lock_cleaner
        self._logger = logger or logging.getLogger(__name__)
        self._evict_callbacks: list[Callable[[K, V], None]] = []

        if telemetry is not None:
            if isinstance(telemetry, tuple) and len(telemetry) == 2 and isinstance(
                telemetry[0], CacheManager
            ):
                items: Sequence[tuple[CacheManager, str]] = (telemetry,)
            else:
                items = tuple(telemetry)
            for manager, metrics_key in items:
                self.register_telemetry(manager, metrics_key)

        if on_evict is not None:
            for callback in on_evict:
                self.register_evict_callback(callback)

    def register_evict_callback(self, callback: Callable[[K, V], None]) -> None:
        """Register ``callback`` to be invoked on cache evictions."""

        self._evict_callbacks.append(callback)

    def register_telemetry(self, manager: CacheManager, metrics_key: str) -> None:
        """Register telemetry hook incrementing ``metrics_key`` on ``manager``."""

        def _emit(_key: K, _value: V) -> None:
            manager.increment_eviction(metrics_key)

        self._evict_callbacks.append(_emit)

    def _cleanup_lock(self, key: K) -> None:
        if self._lock_cleaner is None:
            return
        try:
            self._lock_cleaner.on_remove(key)
        except Exception:  # pragma: no cover - defensive logging
            self._logger.exception("lock cleanup failed for %r", key)

    def _emit_eviction(self, key: K, value: V) -> None:
        self._cleanup_lock(key)
        for callback in self._evict_callbacks:
            try:
                callback(key, value)
            except Exception:  # pragma: no cover - defensive logging
                self._logger.exception("LRU eviction callback failed for %r", key)

    def popitem(self) -> tuple[K, V]:  # type: ignore[override]
        key, value = super().popitem()
        self._emit_eviction(key, value)
        return key, value

    def pop(self, key: K, default: V | object = _MISSING) -> V:  # type: ignore[override]
        if default is self._MISSING:
            value = super().pop(key)
        else:
            sentinel = object()
            value = super().pop(key, sentinel)
            if value is sentinel:
                return cast(V, default)
        self._cleanup_lock(key)
        return cast(V, value)

    def __delitem__(self, key: K) -> None:  # type: ignore[override]
        super().__delitem__(key)
        self._cleanup_lock(key)

    def clear(self) -> None:  # type: ignore[override]
        super().clear()
        if self._lock_cleaner is not None:
            try:
                self._lock_cleaner.clear()
            except Exception:  # pragma: no cover - defensive logging
                self._logger.exception("lock cleanup failed during clear")
