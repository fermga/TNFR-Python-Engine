"""Deterministic RNG helpers."""

from __future__ import annotations

import random
import hashlib
import struct
from collections.abc import Iterator, MutableMapping
from typing import Any, Generic, Hashable, TypeVar


from cachetools import LRUCache, cached
from .constants import DEFAULTS, get_param
from .graph_utils import get_graph
from .locking import get_lock

MASK64 = 0xFFFFFFFFFFFFFFFF

_RNG_LOCK = get_lock("rng")
_DEFAULT_CACHE_MAXSIZE = int(DEFAULTS.get("JITTER_CACHE_SIZE", 128))
_CACHE_MAXSIZE = _DEFAULT_CACHE_MAXSIZE
_CACHE_LOCKED = False

K = TypeVar("K", bound=Hashable)


class _SeedHashCache(MutableMapping[tuple[int, int], int]):
    """Mutable mapping proxy exposing a configurable LRU cache."""

    def __init__(self, maxsize: int) -> None:
        self._maxsize = 0
        self._cache: LRUCache[tuple[int, int], int] | None = None
        self.configure(maxsize)

    def configure(self, maxsize: int) -> None:
        """Configure internal cache size, clearing previous entries."""

        self._maxsize = int(maxsize)
        if self._maxsize <= 0:
            self._cache = None
        else:
            self._cache = LRUCache(maxsize=self._maxsize)

    def __getitem__(self, key: tuple[int, int]) -> int:
        if self._cache is None:
            raise KeyError(key)
        return self._cache[key]

    def __setitem__(self, key: tuple[int, int], value: int) -> None:
        if self._cache is not None:
            self._cache[key] = value

    def __delitem__(self, key: tuple[int, int]) -> None:
        if self._cache is None:
            raise KeyError(key)
        del self._cache[key]

    def __iter__(self) -> Iterator[tuple[int, int]]:
        if self._cache is None:
            return iter(())
        return iter(self._cache)

    def __len__(self) -> int:
        if self._cache is None:
            return 0
        return len(self._cache)

    def clear(self) -> None:  # type: ignore[override]
        if self._cache is not None:
            self._cache.clear()

    @property
    def maxsize(self) -> int:
        return self._maxsize

    @property
    def enabled(self) -> bool:
        return self._cache is not None

    @property
    def data(self) -> LRUCache[tuple[int, int], int] | None:
        """Expose the underlying cache for diagnostics/tests."""

        return self._cache


class ScopedCounterCache(Generic[K]):
    """Thread-safe LRU cache storing monotonic counters by ``key``."""

    def __init__(self, name: str, max_entries: int) -> None:
        if max_entries < 0:
            raise ValueError("max_entries must be non-negative")
        self._lock = get_lock(name)
        self._max_entries = int(max_entries)
        self._cache: LRUCache[K, int] = LRUCache(maxsize=self._max_entries)

    @property
    def lock(self):
        """Return the lock guarding access to the underlying cache."""

        return self._lock

    @property
    def max_entries(self) -> int:
        """Return the configured maximum number of cached entries."""

        return self._max_entries

    @property
    def cache(self) -> LRUCache[K, int]:
        """Expose the underlying ``LRUCache`` for inspection."""

        return self._cache

    def configure(
        self, *, force: bool = False, max_entries: int | None = None
    ) -> None:
        """Resize or reset the cache keeping previous settings."""

        size = self._max_entries if max_entries is None else int(max_entries)
        if size < 0:
            raise ValueError("max_entries must be non-negative")
        with self._lock:
            if size != self._max_entries:
                self._max_entries = size
                force = True
            if force:
                self._cache = LRUCache(maxsize=self._max_entries)

    def clear(self) -> None:
        """Clear stored counters preserving ``max_entries``."""

        self.configure(force=True)

    def bump(self, key: K) -> int:
        """Return current counter for ``key`` and increment it atomically."""

        with self._lock:
            value = int(self._cache.get(key, 0))
            self._cache[key] = value + 1
            return value

    def __len__(self) -> int:
        return len(self._cache)


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
