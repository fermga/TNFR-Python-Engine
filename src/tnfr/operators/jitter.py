from __future__ import annotations
from typing import Any, TYPE_CHECKING
from weakref import WeakKeyDictionary, WeakSet
from cachetools import LRUCache

from ..helpers import ensure_node_offset_map
from ..rng import (
    make_rng,
    base_seed,
    cache_enabled,
    clear_rng_cache as _clear_rng_cache,
    seed_hash,
)
from ..locking import get_lock
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
        self.max_entries = max_entries
        self.seq: LRUCache[tuple[int, int], int] = LRUCache(maxsize=max_entries)
        self.graphs: WeakSet[Any] = WeakSet()
        self.settings: dict[str, Any] = {"max_entries": max_entries}
        self.lock = get_lock("jitter")

    def setup(self, force: bool = False) -> None:
        """Ensure ``seq`` matches the configured size."""
        max_entries = self.max_entries
        if force or self.settings.get("max_entries") != max_entries:
            self.seq = LRUCache(maxsize=max_entries)
            self.settings["max_entries"] = max_entries

    def clear(self) -> None:
        """Clear cached RNGs and jitter state."""
        with self.lock:
            _clear_rng_cache()
            self.setup(force=True)
            for G in list(self.graphs):
                cache = G.graph.get("_jitter_seed_hash")
                if cache is not None:
                    cache.clear()
            self.graphs.clear()


class JitterCacheManager:
    """Manager exposing the jitter cache without global reassignment."""

    def __init__(self, cache: JitterCache | None = None) -> None:
        self.cache = cache or JitterCache()

    # Convenience passthrough properties
    @property
    def seq(self) -> LRUCache[tuple[int, int], int]:
        return self.cache.seq

    @property
    def graphs(self) -> WeakSet[Any]:
        return self.cache.graphs

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

    def setup(self, force: bool = False, max_entries: int | None = None) -> None:
        """Ensure jitter cache matches the configured size.

        ``max_entries`` may be provided to explicitly resize the cache.
        When omitted the existing ``cache.max_entries`` is preserved.
        """
        if max_entries is not None:
            self.cache.max_entries = max_entries
        self.cache.setup(force)

    def clear(self) -> None:
        """Clear cached RNGs and jitter state."""
        self.cache.clear()


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


def _get_jitter_cache(
    node: NodoProtocol, manager: JitterCacheManager | None = None
) -> dict:
    """Return the jitter cache for ``node``.

    If the node cannot store attributes, fall back to a graph-level
    ``WeakKeyDictionary`` and ensure the graph is tracked in
    ``manager.graphs`` so its cache can be cleared when needed.
    """
    if manager is None:
        manager = get_jitter_manager()

    cache = getattr(node, "_jitter_seed_hash", None)
    if cache is not None:
        return cache

    try:
        cache = {}
        setattr(node, "_jitter_seed_hash", cache)
        return cache
    except AttributeError:
        graph_cache = node.graph.get("_jitter_seed_hash")
        if graph_cache is None:
            graph_cache = WeakKeyDictionary()
            node.graph["_jitter_seed_hash"] = graph_cache
            with manager.lock:
                manager.graphs.add(node.graph)
        cache = graph_cache.get(node)
        if cache is None:
            cache = {}
            graph_cache[node] = cache
        return cache


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
    cache = _get_jitter_cache(node, manager)

    cache_key = (seed_root, scope_id)
    seed = cache.get(cache_key)
    if seed is None:
        seed = seed_hash(seed_root, scope_id)
        cache[cache_key] = seed
    seq = 0
    if cache_enabled(node.G):
        with manager.lock:
            seq = manager.seq.get(cache_key, 0)
            manager.seq[cache_key] = seq + 1
    rng = make_rng(seed, seed_key + seq, node.G)
    return rng.uniform(-amplitude, amplitude)


__all__ = [
    "JitterCache",
    "JitterCacheManager",
    "get_jitter_manager",
    "reset_jitter_manager",
    "random_jitter",
    "_get_jitter_cache",
]
