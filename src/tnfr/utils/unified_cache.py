"""TNFR Unified Cache System - Consolidated Memory Management.

CONSOLIDATION ACHIEVEMENT: This module unifies caching strategies across
TNFR codebase under a single coherent interface with monitoring and policies.

Unified Architecture:
- Centralized cache manager with named regions
- Consistent LRU eviction policies
- Memory usage monitoring and limits
- Unified hit/miss statistics
- Thread-safe operations

Theoretical Foundation:
Efficient caching preserves computational coherence by minimizing redundant
reorganization (ΔNFR) calculations, analogous to structural memory (ξ_C)
persistence in the nodal equation.

Consolidates:
- Ad-hoc caching in validation/telemetry systems
- Legacy cache implementations in utils/cache.py (interface wrapper)
- Scattered memoization decorators

Status: UNIFIED CACHE CONSOLIDATION
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator, MutableMapping
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

from ..types import CacheLevel, CacheStats

if TYPE_CHECKING:
    from .cache import CacheManager


# Lazy import wrapper to avoid circular dependencies with utils.cache
def cache_tnfr_computation(*args: Any, **kwargs: Any) -> Any:
    """Unified cache decorator that delegates to the robust implementation in utils.cache."""
    from .cache import cache_tnfr_computation as real_impl

    return real_impl(*args, **kwargs)


logger = logging.getLogger(__name__)

__all__ = [
    "CacheLevel",
    "cache_tnfr_computation",
    "TNFRUnifiedCacheSystem",
    "UnifiedLRUCache",
    "CacheStats",
]

K = TypeVar("K")
V = TypeVar("V")


def _normalise_callbacks(
    callbacks: Iterable[Callable[[K, V], None]] | Callable[[K, V], None] | None,
) -> tuple[Callable[[K, V], None], ...]:
    if callbacks is None:
        return ()
    if callable(callbacks):
        return (callbacks,)
    return tuple(callbacks)


class UnifiedLRUCache(MutableMapping[K, V], Generic[K, V]):
    """Thread-safe LRU Cache implementation for TNFR unified systems.

    Features:
    - Thread-safe operations (RLock)
    - LRU eviction policy
    - Telemetry and eviction callbacks
    - Integration with CacheManager (optional)
    - External lock synchronization
    """

    def __init__(
        self,
        maxsize: int = 1000,
        name: str = "default",
        manager: CacheManager | None = None,
        metrics_key: str | None = None,
        eviction_callbacks: (
            Iterable[Callable[[K, V], None]] | Callable[[K, V], None] | None
        ) = None,
        telemetry_callbacks: (
            Iterable[Callable[[K, V], None]] | Callable[[K, V], None] | None
        ) = None,
        locks: MutableMapping[K, Any] | None = None,
        getsizeof: Callable[[V], int] | None = None,
        count_overwrite_hit: bool = True,
    ) -> None:
        self.maxsize = maxsize
        self.name = name
        self._manager = manager
        self._metrics_key = metrics_key
        self._locks = locks
        self._getsizeof = getsizeof
        self._count_overwrite_hit = count_overwrite_hit
        self._currsize = 0

        self._cache: OrderedDict[K, V] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = CacheStats(max_size=maxsize)

        self._eviction_callbacks = list(_normalise_callbacks(eviction_callbacks))
        self._telemetry_callbacks = list(_normalise_callbacks(telemetry_callbacks))

    @property
    def telemetry_callbacks(self) -> tuple[Callable[[K, V], None], ...]:
        """Return currently registered telemetry callbacks."""
        with self._lock:
            return tuple(self._telemetry_callbacks)

    @property
    def eviction_callbacks(self) -> tuple[Callable[[K, V], None], ...]:
        """Return currently registered eviction callbacks."""
        with self._lock:
            return tuple(self._eviction_callbacks)

    def set_telemetry_callbacks(
        self,
        callbacks: Iterable[Callable[[K, V], None]] | Callable[[K, V], None] | None,
        *,
        append: bool = False,
    ) -> None:
        """Update telemetry callbacks executed on removals."""
        with self._lock:
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
        """Update eviction callbacks executed on removals."""
        with self._lock:
            new_callbacks = list(_normalise_callbacks(callbacks))
            if append:
                self._eviction_callbacks.extend(new_callbacks)
            else:
                self._eviction_callbacks = new_callbacks

    def __getitem__(self, key: K) -> V:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._stats.hits += 1
                if self._manager and self._metrics_key:
                    self._manager.increment_hit(self._metrics_key)
                return self._cache[key]

            self._stats.misses += 1
            if self._manager and self._metrics_key:
                self._manager.increment_miss(self._metrics_key)
            raise KeyError(key)

    def __setitem__(self, key: K, value: V) -> None:
        with self._lock:
            size = self._getsizeof(value) if self._getsizeof else 1

            if key in self._cache:
                old_value = self._cache[key]
                old_size = self._getsizeof(old_value) if self._getsizeof else 1
                self._currsize -= old_size
                self._cache.move_to_end(key)
                self._cache[key] = value
                self._currsize += size

                if self._count_overwrite_hit:
                    self._stats.hits += 1
                    if self._manager and self._metrics_key:
                        self._manager.increment_hit(self._metrics_key)
            else:
                self._cache[key] = value
                self._currsize += size
                self._stats.size += 1

            # Evict if full
            while self._currsize > self.maxsize:
                self.popitem()

    def __delitem__(self, key: K) -> None:
        with self._lock:
            value = self._cache.pop(key)
            size = self._getsizeof(value) if self._getsizeof else 1
            self._currsize -= size
            self._stats.size -= 1
            self._dispatch_removal(key, value)

    def __iter__(self) -> Iterator[K]:
        with self._lock:
            return iter(list(self._cache))

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    @property
    def currsize(self) -> int:
        """Return current size (items or weighted)."""
        with self._lock:
            return self._currsize

    def popitem(self) -> tuple[K, V]:
        """Remove and return the (key, value) pair least recently used."""
        with self._lock:
            # last=False pops FIFO (least recently used in OrderedDict)
            key, value = self._cache.popitem(last=False)
            size = self._getsizeof(value) if self._getsizeof else 1
            self._currsize -= size
            self._stats.size -= 1
            self._stats.evictions += 1

            if self._manager and self._metrics_key:
                self._manager.increment_eviction(self._metrics_key)

            self._dispatch_removal(key, value)
            return key, value

    def _dispatch_removal(self, key: K, value: V) -> None:
        """Handle cleanup/callbacks for removed items."""
        # Lock cleanup
        if self._locks is not None:
            try:
                self._locks.pop(key, None)
            except Exception:
                logger.exception("lock cleanup failed for %r", key)

        # Telemetry callbacks
        for callback in self._telemetry_callbacks:
            try:
                callback(key, value)
            except Exception:
                logger.exception("telemetry callback failed for %r", key)

        # Eviction callbacks
        for callback in self._eviction_callbacks:
            try:
                callback(key, value)
            except Exception:
                logger.exception("eviction callback failed for %r", key)

    def get(self, key: K, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def put(self, key: K, value: V) -> None:
        self[key] = value

    def clear(self) -> None:
        """Clear cache."""
        with self._lock:
            self._cache.clear()
            self._stats.size = 0
            # Don't reset hits/misses for historical stats

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            # Return copy
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                size=self._stats.size,
                max_size=self.maxsize,
                timings=self._stats.timings,
                total_time=self._stats.total_time,
            )


class TNFRUnifiedCacheSystem:
    """Unified Cache System - Centralized Memory Management.

    ARCHITECTURE: Provides named cache regions for different TNFR subsystems
    (validation, telemetry, computation) with unified monitoring.

    Usage:
        # Get global system
        cache_sys = get_unified_cache_system()

        # Get or create a region
        val_cache = cache_sys.get_region("validation", max_size=5000)

        # Use cache
        val_cache.put("key", result)
        item = val_cache.get("key")
    """

    def __init__(self) -> None:
        self._regions: dict[str, UnifiedLRUCache] = {}
        self._lock = threading.RLock()
        self._global_stats = {"start_time": time.time()}

        logger.info("Initialized unified cache system")

    def get_region(self, name: str, max_size: int = 1000) -> UnifiedLRUCache:
        """Get or create a named cache region."""
        with self._lock:
            if name not in self._regions:
                self._regions[name] = UnifiedLRUCache(maxsize=max_size, name=name)
                logger.debug(f"Created cache region '{name}' with size {max_size}")
            return self._regions[name]

    def clear_all(self) -> None:
        """Clear all cache regions."""
        with self._lock:
            for region in self._regions.values():
                region.clear()
            logger.info("Cleared all unified cache regions")

    def get_system_stats(self) -> dict[str, Any]:
        """Get aggregated statistics for all regions."""
        with self._lock:
            stats: dict[str, Any] = {
                "regions": {},
                "total_hits": 0,
                "total_misses": 0,
                "total_evictions": 0,
                "total_items": 0,
                "uptime": time.time() - cast(float, self._global_stats["start_time"]),
            }

            for name, region in self._regions.items():
                r_stats = region.get_stats()
                stats["regions"][name] = {
                    "hits": r_stats.hits,
                    "misses": r_stats.misses,
                    "hit_rate": r_stats.hit_rate,
                    "size": r_stats.size,
                    "max_size": r_stats.max_size,
                }
                stats["total_hits"] += r_stats.hits
                stats["total_misses"] += r_stats.misses
                stats["total_evictions"] += r_stats.evictions
                stats["total_items"] += r_stats.size

            total_reqs = stats["total_hits"] + stats["total_misses"]
            stats["global_hit_rate"] = (
                (stats["total_hits"] / total_reqs) if total_reqs > 0 else 0.0
            )

            return stats


# ============================================================================
# PUBLIC API
# ============================================================================

_unified_cache_system: TNFRUnifiedCacheSystem | None = None


def get_unified_cache_system() -> TNFRUnifiedCacheSystem:
    """Get global unified cache system instance."""
    global _unified_cache_system
    if _unified_cache_system is None:
        _unified_cache_system = TNFRUnifiedCacheSystem()
    return _unified_cache_system


def get_cache_region(name: str, max_size: int = 1000) -> UnifiedLRUCache:
    """Get a named cache region - convenience function."""
    return get_unified_cache_system().get_region(name, max_size)


def clear_unified_caches() -> None:
    """Clear all unified caches - convenience function."""
    if _unified_cache_system:
        _unified_cache_system.clear_all()
