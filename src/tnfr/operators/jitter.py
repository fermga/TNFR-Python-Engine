from __future__ import annotations
from typing import Any, TYPE_CHECKING
from weakref import WeakKeyDictionary, WeakSet
from cachetools import LRUCache

from ..helpers.cache import ensure_node_offset_map
from ..rng import (
    make_rng,
    base_seed,
    cache_enabled,
    clear_rng_cache as _clear_rng_cache,
    seed_hash,
)
from ..locking import get_lock
from ..import_utils import import_nodonx

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from ..node import NodoProtocol

# Guarded by ``JITTER_CACHE.lock`` to ensure thread-safe access.
# ``JITTER_CACHE.seq`` stores per-scope jitter sequence counters in an LRU cache
# bounded to avoid unbounded memory usage.
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


# Module-level singleton
JITTER_CACHE = JitterCache()

# Backward-compatibility aliases; updated by ``_update_cache_refs``.
_JITTER_SEQ = JITTER_CACHE.seq
_JITTER_GRAPHS = JITTER_CACHE.graphs
_JITTER_SETTINGS = JITTER_CACHE.settings


def _update_cache_refs(cache: JitterCache) -> None:
    global _JITTER_SEQ, _JITTER_GRAPHS, _JITTER_SETTINGS
    _JITTER_SEQ = cache.seq
    _JITTER_GRAPHS = cache.graphs
    _JITTER_SETTINGS = cache.settings


def setup_jitter_cache(
    force: bool = False, cache: JitterCache = JITTER_CACHE
) -> None:
    """Ensure jitter cache matches the configured size."""
    cache.max_entries = _JITTER_MAX_ENTRIES
    cache.setup(force)
    _update_cache_refs(cache)


# Initialize jitter cache on module import.
setup_jitter_cache(force=True)


def _node_offset(G, n) -> int:
    """Deterministic node index used for jitter seeds."""
    mapping = ensure_node_offset_map(G)
    return int(mapping.get(n, 0))


def clear_rng_cache(cache: JitterCache = JITTER_CACHE) -> None:
    """Clear cached RNGs and jitter state."""
    cache.clear()
    _update_cache_refs(cache)


def _resolve_jitter_seed(node: NodoProtocol) -> tuple[int, int]:
    NodoNX = import_nodonx()  # cache import to avoid repeated lookups
    if isinstance(node, NodoNX):
        return _node_offset(node.G, node.n), id(node.G)
    uid = getattr(node, "_noise_uid", None)
    if uid is None:
        uid = id(node)
        setattr(node, "_noise_uid", uid)
    return int(uid), id(node)


def _get_jitter_cache(
    node: NodoProtocol, cache: JitterCache = JITTER_CACHE
) -> dict:
    """Return the jitter cache for ``node``.

    If the node cannot store attributes, fall back to a graph-level
    ``WeakKeyDictionary`` and ensure the graph is tracked in
    ``_JITTER_GRAPHS`` so its cache can be cleared when needed.
    """

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
            with cache.lock:
                cache.graphs.add(node.graph)
        cache = graph_cache.get(node)
        if cache is None:
            cache = {}
            graph_cache[node] = cache
        return cache


def random_jitter(node: NodoProtocol, amplitude: float) -> float:
    """Return deterministic noise in ``[-amplitude, amplitude]`` for
    ``node``."""
    if amplitude < 0:
        raise ValueError("amplitude must be positive")
    if amplitude == 0:
        return 0.0

    seed_root = base_seed(node.G)
    seed_key, scope_id = _resolve_jitter_seed(node)

    cache = _get_jitter_cache(node)

    cache_key = (seed_root, scope_id)
    seed = cache.get(cache_key)
    if seed is None:
        seed = seed_hash(seed_root, scope_id)
        cache[cache_key] = seed
    seq = 0
    if cache_enabled(node.G):
        with JITTER_CACHE.lock:
            seq = JITTER_CACHE.seq.get(cache_key, 0)
            JITTER_CACHE.seq[cache_key] = seq + 1
    rng = make_rng(seed, seed_key + seq, node.G)
    return rng.uniform(-amplitude, amplitude)


__all__ = [
    "JitterCache",
    "JITTER_CACHE",
    "setup_jitter_cache",
    "clear_rng_cache",
    "random_jitter",
    "_get_jitter_cache",
    "_JITTER_SEQ",
    "_JITTER_GRAPHS",
    "_JITTER_SETTINGS",
]
