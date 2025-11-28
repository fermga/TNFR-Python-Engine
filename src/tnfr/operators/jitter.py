"""Jitter operators for reproducible phase perturbations."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, cast

from ..rng import base_seed, cache_enabled
from ..rng import clear_rng_cache as _clear_rng_cache
from ..rng import (
    make_rng,
    seed_hash,
)
from ..types import NodeId, TNFRGraph
from ..utils import (
    CacheManager,
    InstrumentedLRUCache,
    ScopedCounterCache,
    build_cache_manager,
    ensure_node_offset_map,
    get_nodenx,
)

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from ..node import NodeProtocol

# Guarded by the cache lock to ensure thread-safe access. ``seq`` stores
# per-scope jitter sequence counters in an instrumented LRU cache bounded to avoid
# unbounded memory usage.
_JITTER_MAX_ENTRIES = 1024


class JitterCache:
    """Container for jitter-related caches."""

    def __init__(
        self,
        max_entries: int = _JITTER_MAX_ENTRIES,
        *,
        manager: CacheManager | None = None,
    ) -> None:
        self._manager = manager or build_cache_manager()
        if not self._manager.has_override("scoped_counter:jitter"):
            self._manager.configure(overrides={"scoped_counter:jitter": int(max_entries)})
        self._sequence = ScopedCounterCache(
            "jitter",
            max_entries=None,
            manager=self._manager,
            default_max_entries=int(max_entries),
        )
        self._settings_key = "jitter_settings"
        self._manager.register(
            self._settings_key,
            lambda: {"max_entries": self._sequence.max_entries},
            reset=self._reset_settings,
        )

    def _reset_settings(self, settings: dict[str, Any] | None) -> dict[str, Any]:
        return {"max_entries": self._sequence.max_entries}

    def _refresh_settings(self) -> None:
        self._manager.update(
            self._settings_key,
            lambda _: {"max_entries": self._sequence.max_entries},
        )

    @property
    def manager(self) -> CacheManager:
        """Expose the cache manager backing this cache."""

        return self._manager

    @property
    def seq(self) -> InstrumentedLRUCache[tuple[int, int], int]:
        """Expose the instrumented sequence cache for tests and diagnostics."""

        return self._sequence.cache

    @property
    def lock(self) -> threading.Lock | threading.RLock:
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
        self._refresh_settings()

    @property
    def settings(self) -> dict[str, Any]:
        """Return jitter cache settings stored on the manager."""

        return cast(dict[str, Any], self._manager.get(self._settings_key))

    def setup(self, force: bool = False, max_entries: int | None = None) -> None:
        """Ensure jitter cache matches the configured size."""

        self._sequence.configure(force=force, max_entries=max_entries)
        self._refresh_settings()

    def clear(self) -> None:
        """Clear cached RNGs and jitter state."""

        _clear_rng_cache()
        self._sequence.clear()
        self._manager.clear(self._settings_key)

    def bump(self, key: tuple[int, int]) -> int:
        """Return current jitter sequence counter for ``key`` and increment it."""

        return self._sequence.bump(key)


class JitterCacheManager:
    """Manager exposing the jitter cache without global reassignment."""

    def __init__(
        self,
        cache: JitterCache | None = None,
        *,
        manager: CacheManager | None = None,
    ) -> None:
        if cache is not None:
            self.cache = cache
            self._manager = cache.manager
        else:
            self._manager = manager or build_cache_manager()
            self.cache = JitterCache(manager=self._manager)

    # Convenience passthrough properties
    @property
    def seq(self) -> InstrumentedLRUCache[tuple[int, int], int]:
        """Expose the underlying instrumented jitter sequence cache."""

        return self.cache.seq

    @property
    def settings(self) -> dict[str, Any]:
        """Return persisted jitter cache configuration."""

        return self.cache.settings

    @property
    def lock(self) -> threading.Lock | threading.RLock:
        """Return the lock associated with the jitter cache."""

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


def _node_offset(G: TNFRGraph, n: NodeId) -> int:
    """Deterministic node index used for jitter seeds."""
    mapping = ensure_node_offset_map(G)
    return int(mapping.get(n, 0))


def _resolve_jitter_seed(node: NodeProtocol) -> tuple[int, int]:
    node_nx_type = get_nodenx()
    if node_nx_type is None:
        raise ImportError("NodeNX is unavailable")
    if isinstance(node, node_nx_type):
        graph = cast(TNFRGraph, getattr(node, "G"))
        node_id = cast(NodeId, getattr(node, "n"))
        return _node_offset(graph, node_id), id(graph)
    uid = getattr(node, "_noise_uid", None)
    if uid is None:
        uid = id(node)
        setattr(node, "_noise_uid", uid)
    graph = cast(TNFRGraph | None, getattr(node, "G", None))
    scope = graph if graph is not None else node
    return int(uid), id(scope)


def random_jitter(
    node: NodeProtocol,
    amplitude: float,
) -> float:
    """Return deterministic noise in ``[-amplitude, amplitude]`` for ``node``.

    The per-node jitter sequences are tracked using the global manager
    returned by :func:`get_jitter_manager`.
    """
    if amplitude < 0:
        raise ValueError("amplitude must be positive")
    if amplitude == 0:
        return 0.0

    seed_root = base_seed(node.G)
    seed_key, scope_id = _resolve_jitter_seed(node)

    cache_key = (seed_root, scope_id, seed_key)
    seq = 0
    if cache_enabled(node.G):
        manager = get_jitter_manager()
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
