"""Deterministic RNG helpers."""

from __future__ import annotations

import random
import hashlib
import struct
from collections.abc import Iterator, MutableMapping
from dataclasses import dataclass
from typing import Any, Generic, Hashable, TypeVar, cast


from cachetools import LRUCache, cached
from .constants import DEFAULTS, get_param
from .cache import CacheManager
from .utils import get_graph
from .locking import get_lock

MASK64 = 0xFFFFFFFFFFFFFFFF

_RNG_LOCK = get_lock("rng")
_DEFAULT_CACHE_MAXSIZE = int(DEFAULTS.get("JITTER_CACHE_SIZE", 128))
_CACHE_MAXSIZE = _DEFAULT_CACHE_MAXSIZE
_CACHE_LOCKED = False

K = TypeVar("K", bound=Hashable)


@dataclass
class _SeedCacheState:
    cache: LRUCache[tuple[int, int], int] | None
    maxsize: int


@dataclass
class _CounterState(Generic[K]):
    cache: LRUCache[K, int]
    max_entries: int


_RNG_CACHE_MANAGER = CacheManager()



class _SeedHashCache(MutableMapping[tuple[int, int], int]):
    """Mutable mapping proxy exposing a configurable LRU cache."""

    def __init__(
        self,
        maxsize: int,
        *,
        manager: CacheManager | None = None,
        state_key: str = "seed_hash_cache",
    ) -> None:
        self._manager = manager or _RNG_CACHE_MANAGER
        self._state_key = state_key
        initial_size = int(maxsize)
        self._manager.register(
            self._state_key,
            lambda: self._create_state(initial_size),
            reset=self._reset_state,
        )

    def _create_state(self, maxsize: int) -> _SeedCacheState:
        if maxsize <= 0:
            return _SeedCacheState(cache=None, maxsize=0)
        return _SeedCacheState(cache=LRUCache(maxsize=maxsize), maxsize=maxsize)

    def _reset_state(self, state: _SeedCacheState | None) -> _SeedCacheState:
        size = state.maxsize if isinstance(state, _SeedCacheState) else 0
        return self._create_state(size)

    def _get_state(self, *, create: bool = True) -> _SeedCacheState | None:
        state = self._manager.get(self._state_key, create=create)
        if state is None:
            return None
        if not isinstance(state, _SeedCacheState):
            state = self._create_state(0)
            self._manager.store(self._state_key, state)
        return state

    def configure(self, maxsize: int) -> None:
        size = int(maxsize)
        self._manager.update(self._state_key, lambda _: self._create_state(size))

    def __getitem__(self, key: tuple[int, int]) -> int:
        state = self._get_state()
        if state is None or state.cache is None:
            raise KeyError(key)
        return state.cache[key]

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
    def data(self) -> LRUCache[tuple[int, int], int] | None:
        """Expose the underlying cache for diagnostics/tests."""

        state = self._get_state(create=False)
        return None if state is None else state.cache


class ScopedCounterCache(Generic[K]):
    """Thread-safe LRU cache storing monotonic counters by ``key``."""

    def __init__(
        self,
        name: str,
        max_entries: int,
        *,
        manager: CacheManager | None = None,
    ) -> None:
        if max_entries < 0:
            raise ValueError("max_entries must be non-negative")
        self._name = name
        self._manager = manager or _RNG_CACHE_MANAGER
        self._state_key = f"scoped_counter:{name}"
        initial_size = int(max_entries)
        self._manager.register(
            self._state_key,
            lambda: self._create_state(initial_size),
            lock_factory=lambda: get_lock(name),
            reset=self._reset_state,
        )
        self.configure(force=True, max_entries=max_entries)

    def _create_state(self, max_entries: int) -> _CounterState[K]:
        return _CounterState(cache=LRUCache(maxsize=max_entries), max_entries=max_entries)

    def _reset_state(self, state: _CounterState[K] | None) -> _CounterState[K]:
        size = state.max_entries if isinstance(state, _CounterState) else 0
        return self._create_state(size)

    def _get_state(self) -> _CounterState[K]:
        state = self._manager.get(self._state_key)
        if not isinstance(state, _CounterState):
            state = self._create_state(0)
            self._manager.store(self._state_key, state)
        return state

    @property
    def lock(self):
        """Return the lock guarding access to the underlying cache."""

        return self._manager.get_lock(self._state_key)

    @property
    def max_entries(self) -> int:
        """Return the configured maximum number of cached entries."""

        return self._get_state().max_entries

    @property
    def cache(self) -> LRUCache[K, int]:
        """Expose the underlying ``LRUCache`` for inspection."""

        return self._get_state().cache

    def configure(
        self, *, force: bool = False, max_entries: int | None = None
    ) -> None:
        """Resize or reset the cache keeping previous settings."""

        size = self.max_entries if max_entries is None else int(max_entries)
        if size < 0:
            raise ValueError("max_entries must be non-negative")

        def _update(state: _CounterState[K] | None) -> _CounterState[K]:
            if not isinstance(state, _CounterState) or force or state.max_entries != size:
                return self._create_state(size)
            return cast(_CounterState[K], state)

        self._manager.update(self._state_key, _update)

    def clear(self) -> None:
        """Clear stored counters preserving ``max_entries``."""

        self.configure(force=True)

    def bump(self, key: K) -> int:
        """Return current counter for ``key`` and increment it atomically."""

        result: dict[str, int] = {}

        def _update(state: _CounterState[K] | None) -> _CounterState[K]:
            if not isinstance(state, _CounterState):
                state = self._create_state(0)
            cache = state.cache
            value = int(cache.get(key, 0))
            cache[key] = value + 1
            result["value"] = value
            return state

        self._manager.update(self._state_key, _update)
        return result.get("value", 0)

    def __len__(self) -> int:
        return len(self.cache)
_seed_hash_cache = _SeedHashCache(_CACHE_MAXSIZE)


@cached(cache=_seed_hash_cache, lock=_RNG_LOCK)
def seed_hash(seed_int: int, key_int: int) -> int:
    """Return a 64-bit hash derived from ``seed_int`` and ``key_int``."""

    seed_bytes = struct.pack(
        ">QQ",
        seed_int & MASK64,
        key_int & MASK64,
    )
    return int.from_bytes(
        hashlib.blake2b(seed_bytes, digest_size=8).digest(), "big"
    )


def _sync_cache_size(G: Any | None) -> None:
    """Synchronise cache size with ``G`` when needed."""

    global _CACHE_MAXSIZE
    if G is None or _CACHE_LOCKED:
        return
    size = get_cache_maxsize(G)
    with _RNG_LOCK:
        if size != _CACHE_MAXSIZE:
            _seed_hash_cache.configure(size)
            _CACHE_MAXSIZE = size


def make_rng(seed: int, key: int, G: Any | None = None) -> random.Random:
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
    if _CACHE_MAXSIZE <= 0 or not _seed_hash_cache.enabled:
        return
    seed_hash.cache_clear()


def get_cache_maxsize(G: Any) -> int:
    """Return RNG cache maximum size for ``G``."""
    return int(get_param(G, "JITTER_CACHE_SIZE"))


def cache_enabled(G: Any | None = None) -> bool:
    """Return ``True`` if RNG caching is enabled.

    When ``G`` is provided, the cache size is synchronised with
    ``JITTER_CACHE_SIZE`` stored in ``G``.
    """
    # Only synchronise the cache size with ``G`` when caching is enabled.  This
    # preserves explicit calls to :func:`set_cache_maxsize(0)` which are used in
    # tests to temporarily disable caching regardless of graph defaults.
    if _CACHE_MAXSIZE > 0:
        _sync_cache_size(G)
    return _CACHE_MAXSIZE > 0


def base_seed(G: Any) -> int:
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
        _CACHE_MAXSIZE = new_size
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
