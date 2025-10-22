"""Central cache registry infrastructure for TNFR services."""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Generic, Hashable, Iterator, Mapping, MutableMapping, TypeVar, cast

from cachetools import LRUCache

from .types import TimingContext

__all__ = [
    "CacheManager",
    "CacheCapacityConfig",
    "CacheStatistics",
    "InstrumentedLRUCache",
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
        telemetry_callbacks: Iterable[Callable[[K, V], None]]
        | Callable[[K, V], None]
        | None = None,
        eviction_callbacks: Iterable[Callable[[K, V], None]]
        | Callable[[K, V], None]
        | None = None,
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
        callbacks: Iterable[Callable[[K, V], None]]
        | Callable[[K, V], None]
        | None,
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
        callbacks: Iterable[Callable[[K, V], None]]
        | Callable[[K, V], None]
        | None,
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
        return self._cache[key]

    def __setitem__(self, key: K, value: V) -> None:
        exists = key in self._cache
        self._cache[key] = value
        if exists:
            if self._count_overwrite_hit:
                self._record_hit(1)
        else:
            self._record_miss(1)

    def __delitem__(self, key: K) -> None:
        try:
            value = self._cache[key]
        except KeyError:
            self._record_miss(1)
            raise
        del self._cache[key]
        self._dispatch_removal(key, value, hits=1)

    def __iter__(self) -> Iterator[K]:
        return iter(self._cache)

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, key: object) -> bool:
        return key in self._cache

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"{self.__class__.__name__}({self._cache!r})"

    # ------------------------------------------------------------------
    # Cache helpers

    @property
    def maxsize(self) -> int:
        return self._cache.maxsize

    @property
    def currsize(self) -> int:
        return self._cache.currsize

    def get(self, key: K, default: V | None = None) -> V | None:
        return self._cache.get(key, default)

    def pop(self, key: K, default: Any = _MISSING) -> V:
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
        return self._cache.popitem()

    def clear(self) -> None:  # type: ignore[override]
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
        eviction_callbacks: Iterable[Callable[[K, V], None]]
        | Callable[[K, V], None]
        | None = None,
        telemetry_callbacks: Iterable[Callable[[K, V], None]]
        | Callable[[K, V], None]
        | None = None,
        locks: MutableMapping[K, Any] | None = None,
    ) -> None:
        super().__init__(maxsize)
        self._manager = manager
        self._metrics_key = metrics_key
        self._locks = locks
        self._eviction_callbacks = _normalise_callbacks(eviction_callbacks)
        self._telemetry_callbacks = _normalise_callbacks(telemetry_callbacks)

    def popitem(self) -> tuple[K, V]:  # type: ignore[override]
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
