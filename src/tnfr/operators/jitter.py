from __future__ import annotations
from typing import Any, TYPE_CHECKING

from cachetools import LRUCache

from ..cache import ensure_node_offset_map
from ..rng import (
    ScopedCounterCache,
    make_rng,
    base_seed,
    cache_enabled,
    clear_rng_cache as _clear_rng_cache,
    seed_hash,
)
from ..import_utils import get_nodonx

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from ..node import NodoProtocol

# Guarded by the cache lock to ensure thread-safe access. ``seq`` stores
# per-scope jitter sequence counters in an LRU cache bounded to avoid
# unbounded memory usage.
_JITTER_MAX_ENTRIES = 1024


class JitterCache:
    """Container for jitter-related caches."""

    def __init__(self, max_entries: int = _JITTER_MAX_ENTRIES) -> None:
        self._sequence = ScopedCounterCache("jitter", max_entries)
        self.settings: dict[str, Any] = {"max_entries": self._sequence.max_entries}

    @property
    def seq(self) -> LRUCache[tuple[int, int], int]:
        """Expose the sequence cache for tests and diagnostics."""

        return self._sequence.cache

    @property
    def lock(self):
        """Return the lock protecting the sequence cache."""

        return self._sequence.lock

    @property
    def max_entries(self) -> int:
        """Return the maximum number of cached jitter sequences."""

        return self._sequence.max_entries

    @max_entries.setter
    def max_entries(self, value: int) -> None:
        """Set the maximum number of cached jitter sequences."""

        self._sequence.configure(max_entries=int(value))
        self.settings["max_entries"] = self._sequence.max_entries

    def setup(
        self, force: bool = False, max_entries: int | None = None
    ) -> None:
        """Ensure jitter cache matches the configured size."""

        self._sequence.configure(force=force, max_entries=max_entries)
        self.settings["max_entries"] = self._sequence.max_entries

    def clear(self) -> None:
        """Clear cached RNGs and jitter state."""

        _clear_rng_cache()
        self._sequence.clear()

    def bump(self, key: tuple[int, int]) -> int:
        """Return current jitter sequence counter for ``key`` and increment it."""

        return self._sequence.bump(key)


class JitterCacheManager:
    """Manager exposing the jitter cache without global reassignment."""

    def __init__(self, cache: JitterCache | None = None) -> None:
        self.cache = cache or JitterCache()

    # Convenience passthrough properties
    @property
    def seq(self) -> LRUCache[tuple[int, int], int]:
        return self.cache.seq

    @property
    def settings(self) -> dict[str, Any]:
        return self.cache.settings

    @property
    def lock(self):
        return self.cache.lock

    @property
    def max_entries(self) -> int:
        """Return the maximum number of cached jitter entries."""
        return self.cache.max_entries

    @max_entries.setter
    def max_entries(self, value: int) -> None:
        """Set the maximum number of cached jitter entries."""
        self.cache.max_entries = value

    def setup(
        self, force: bool = False, max_entries: int | None = None
    ) -> None:
        """Ensure jitter cache matches the configured size.

        ``max_entries`` may be provided to explicitly resize the cache.
        When omitted the existing ``cache.max_entries`` is preserved.
        """
        if max_entries is not None:
            self.cache.setup(force=True, max_entries=max_entries)
        else:
            self.cache.setup(force=force)

    def clear(self) -> None:
        """Clear cached RNGs and jitter state."""
        self.cache.clear()

    def bump(self, key: tuple[int, int]) -> int:
        """Return and increment the jitter sequence counter for ``key``."""

        return self.cache.bump(key)


# Lazy manager instance
_JITTER_MANAGER: JitterCacheManager | None = None


def get_jitter_manager() -> JitterCacheManager:
    """Return the singleton jitter manager, initializing on first use."""
    global _JITTER_MANAGER
    if _JITTER_MANAGER is None:
        _JITTER_MANAGER = JitterCacheManager()
        _JITTER_MANAGER.setup(force=True)
    return _JITTER_MANAGER


def reset_jitter_manager() -> None:
    """Reset the global jitter manager (useful for tests)."""
    global _JITTER_MANAGER
    if _JITTER_MANAGER is not None:
        _JITTER_MANAGER.clear()
    _JITTER_MANAGER = None


def _node_offset(G, n) -> int:
    """Deterministic node index used for jitter seeds."""
    mapping = ensure_node_offset_map(G)
    return int(mapping.get(n, 0))


def _resolve_jitter_seed(node: NodoProtocol) -> tuple[int, int]:
    NodoNX = get_nodonx()
    if NodoNX is None:
        raise ImportError("NodoNX is unavailable")
    if isinstance(node, NodoNX):
        return _node_offset(node.G, node.n), id(node.G)
    uid = getattr(node, "_noise_uid", None)
    if uid is None:
        uid = id(node)
        setattr(node, "_noise_uid", uid)
    return int(uid), id(node)


def random_jitter(
    node: NodoProtocol,
    amplitude: float,
    manager: JitterCacheManager | None = None,
) -> float:
    """Return deterministic noise in ``[-amplitude, amplitude]`` for ``node``.

    Uses ``manager`` to track per-node jitter sequences. When ``manager`` is
    ``None`` the global manager from :func:`get_jitter_manager` is used.
    """
    if amplitude < 0:
        raise ValueError("amplitude must be positive")
    if amplitude == 0:
        return 0.0

    seed_root = base_seed(node.G)
    seed_key, scope_id = _resolve_jitter_seed(node)

    manager = manager or get_jitter_manager()

    cache_key = (seed_root, scope_id)
    seq = 0
    if cache_enabled(node.G):
        seq = manager.bump(cache_key)
    seed = seed_hash(seed_root, scope_id)
    rng = make_rng(seed, seed_key + seq, node.G)
    return rng.uniform(-amplitude, amplitude)


__all__ = [
    "JitterCache",
    "JitterCacheManager",
    "get_jitter_manager",
    "reset_jitter_manager",
    "random_jitter",
]
