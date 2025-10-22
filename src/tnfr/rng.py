"""Deterministic RNG helpers."""

from __future__ import annotations

import hashlib
import random
import struct
import threading
from collections.abc import Iterator, MutableMapping
from dataclasses import dataclass
from typing import Any, Generic, Hashable, TypeVar, cast


from cachetools import cached  # type: ignore[import-untyped]
from .constants import DEFAULTS, get_param
from .cache import CacheManager, InstrumentedLRUCache
from .utils import get_graph
from .locking import get_lock
from .types import GraphLike, TNFRGraph

MASK64 = 0xFFFFFFFFFFFFFFFF

_RNG_LOCK = get_lock("rng")
_DEFAULT_CACHE_MAXSIZE = int(DEFAULTS.get("JITTER_CACHE_SIZE", 128))
_CACHE_MAXSIZE = _DEFAULT_CACHE_MAXSIZE
_CACHE_LOCKED = False

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


@dataclass
class _SeedCacheState:
    cache: InstrumentedLRUCache[tuple[int, int], int] | None
    maxsize: int


@dataclass
class _CounterState(Generic[K]):
    cache: InstrumentedLRUCache[K, int]
    locks: dict[K, threading.RLock]
    max_entries: int


_RNG_CACHE_MANAGER = CacheManager(default_capacity=_DEFAULT_CACHE_MAXSIZE)


class _SeedHashCache(MutableMapping[tuple[int, int], int]):
    """Mutable mapping proxy exposing a configurable LRU cache."""

    def __init__(
        self,
        *,
        manager: CacheManager | None = None,
        state_key: str = "seed_hash_cache",
        default_maxsize: int = _DEFAULT_CACHE_MAXSIZE,
    ) -> None:
        self._manager = manager or _RNG_CACHE_MANAGER
        self._state_key = state_key
        self._default_maxsize = int(default_maxsize)
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
        default_max_entries: int = _DEFAULT_CACHE_MAXSIZE,
    ) -> None:
        self._name = name
        self._manager = manager or _RNG_CACHE_MANAGER
        self._state_key = f"scoped_counter:{name}"
        self._default_max_entries = int(default_max_entries)
        requested = None if max_entries is None else int(max_entries)
        if requested is not None and requested < 0:
            raise ValueError("max_entries must be non-negative")
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

    def configure(
        self, *, force: bool = False, max_entries: int | None = None
    ) -> None:
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
        return len(self.cache)


_seed_hash_cache = _SeedHashCache()


def _compute_seed_hash(seed_int: int, key_int: int) -> int:
    seed_bytes = struct.pack(
        ">QQ",
        seed_int & MASK64,
        key_int & MASK64,
    )
    return int.from_bytes(
        hashlib.blake2b(seed_bytes, digest_size=8).digest(), "big"
    )


@cached(cache=_seed_hash_cache, lock=_RNG_LOCK)
def _cached_seed_hash(seed_int: int, key_int: int) -> int:
    return _compute_seed_hash(seed_int, key_int)


def seed_hash(seed_int: int, key_int: int) -> int:
    """Return a 64-bit hash derived from ``seed_int`` and ``key_int``."""

    if _CACHE_MAXSIZE <= 0 or not _seed_hash_cache.enabled:
        return _compute_seed_hash(seed_int, key_int)
    return _cached_seed_hash(seed_int, key_int)


seed_hash.cache_clear = cast(Any, _cached_seed_hash).cache_clear  # type: ignore[attr-defined]
seed_hash.cache = _seed_hash_cache  # type: ignore[attr-defined]


def _sync_cache_size(G: TNFRGraph | GraphLike | None) -> None:
    """Synchronise cache size with ``G`` when needed."""

    global _CACHE_MAXSIZE
    if G is None or _CACHE_LOCKED:
        return
    size = get_cache_maxsize(G)
    with _RNG_LOCK:
        if size != _seed_hash_cache.maxsize:
            _seed_hash_cache.configure(size)
            _CACHE_MAXSIZE = _seed_hash_cache.maxsize


def make_rng(
    seed: int, key: int, G: TNFRGraph | GraphLike | None = None
) -> random.Random:
    """Return a ``random.Random`` for ``seed`` and ``key``.

    When ``G`` is provided, ``JITTER_CACHE_SIZE`` is read from ``G`` and the
    internal cache size is updated accordingly.
    """
    _sync_cache_size(G)
    seed_int = int(seed)
    key_int = int(key)
    return random.Random(seed_hash(seed_int, key_int))


def clear_rng_cache() -> None:
    """Clear cached seed hashes."""
    if _seed_hash_cache.maxsize <= 0 or not _seed_hash_cache.enabled:
        return
    seed_hash.cache_clear()  # type: ignore[attr-defined]


def get_cache_maxsize(G: TNFRGraph | GraphLike) -> int:
    """Return RNG cache maximum size for ``G``."""
    return int(get_param(G, "JITTER_CACHE_SIZE"))


def cache_enabled(G: TNFRGraph | GraphLike | None = None) -> bool:
    """Return ``True`` if RNG caching is enabled.

    When ``G`` is provided, the cache size is synchronised with
    ``JITTER_CACHE_SIZE`` stored in ``G``.
    """
    # Only synchronise the cache size with ``G`` when caching is enabled.  This
    # preserves explicit calls to :func:`set_cache_maxsize(0)` which are used in
    # tests to temporarily disable caching regardless of graph defaults.
    if _seed_hash_cache.maxsize > 0:
        _sync_cache_size(G)
    return _seed_hash_cache.maxsize > 0


def base_seed(G: TNFRGraph | GraphLike) -> int:
    """Return base RNG seed stored in ``G.graph``."""
    graph = get_graph(G)
    return int(graph.get("RANDOM_SEED", 0))


def _rng_for_step(seed: int, step: int) -> random.Random:
    """Return deterministic RNG for a simulation ``step``."""

    return make_rng(seed, step)


def set_cache_maxsize(size: int) -> None:
    """Update RNG cache maximum size.

    ``size`` must be a non-negative integer; ``0`` disables caching.
    Changing the cache size resets any cached seed hashes.
    If caching is disabled, ``clear_rng_cache`` has no effect.
    """

    global _CACHE_MAXSIZE, _CACHE_LOCKED
    new_size = int(size)
    if new_size < 0:
        raise ValueError("size must be non-negative")
    with _RNG_LOCK:
        _seed_hash_cache.configure(new_size)
        _CACHE_MAXSIZE = _seed_hash_cache.maxsize
    _CACHE_LOCKED = new_size != _DEFAULT_CACHE_MAXSIZE


__all__ = (
    "seed_hash",
    "make_rng",
    "get_cache_maxsize",
    "set_cache_maxsize",
    "base_seed",
    "cache_enabled",
    "clear_rng_cache",
    "ScopedCounterCache",
)
