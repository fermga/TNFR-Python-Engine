"""Tests covering multi-layer cache orchestration."""

from __future__ import annotations

import fnmatch
from typing import Any

import networkx as nx

from tnfr.cache import CacheLayer, CacheManager, RedisCacheLayer, ShelveCacheLayer
from tnfr.utils import (
    _GRAPH_CACHE_LAYERS_KEY,
    build_cache_manager,
    configure_global_cache_layers,
    edge_version_cache,
    reset_global_cache_manager,
)


class _FlakyLayer(CacheLayer):
    """Cache layer that can simulate load/store failures."""

    def __init__(self, *, fail_on_store: bool = False, fail_on_load: bool = False) -> None:
        self.fail_on_store = fail_on_store
        self.fail_on_load = fail_on_load
        self._storage: dict[str, Any] = {}

    def load(self, name: str) -> Any:
        if self.fail_on_load:
            raise RuntimeError("layer load failure")
        if name not in self._storage:
            raise KeyError(name)
        return self._storage[name]

    def store(self, name: str, value: Any) -> None:
        if self.fail_on_store:
            raise RuntimeError("layer store failure")
        self._storage[name] = value

    def delete(self, name: str) -> None:
        self._storage.pop(name, None)

    def clear(self) -> None:
        self._storage.clear()


class _FakeRedis:
    """Minimal Redis client stub used for testing the distributed layer."""

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    def get(self, key: str) -> Any:
        return self._data.get(key)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def delete(self, *keys: str) -> None:
        for key in keys:
            self._data.pop(key, None)

    def scan_iter(self, match: str | None = None):
        pattern = "*" if match is None else match
        for key in list(self._data):
            if fnmatch.fnmatch(key, pattern):
                yield key

    def keys(self, pattern: str) -> list[str]:
        return [key for key in self._data if fnmatch.fnmatch(key, pattern)]


def _close_manager(manager: CacheManager | None) -> None:
    if manager is None:
        return
    layers = getattr(manager, "_layers", ())
    for layer in layers:
        close = getattr(layer, "close", None)
        if callable(close):
            close()


def test_cache_manager_layer_failover(tmp_path):
    shelf_layer = ShelveCacheLayer(str(tmp_path / "cache.db"))
    failing = _FlakyLayer(fail_on_store=True)
    manager = CacheManager(layers=(failing, shelf_layer))
    manager.register("demo", lambda: {"value": 0})

    manager.store("demo", {"value": 1})
    assert manager.get("demo") == {"value": 1}

    shelf_layer.close()

    failing_loader = _FlakyLayer(fail_on_load=True)
    restored_shelf = ShelveCacheLayer(str(tmp_path / "cache.db"))
    manager_restarted = CacheManager(layers=(failing_loader, restored_shelf))
    manager_restarted.register("demo", lambda: {"value": 0})

    restored = manager_restarted.get("demo")
    assert restored == {"value": 1}

    restored_shelf.close()


def test_build_cache_manager_hydrates_from_persistent_layers(tmp_path):
    fake_redis = _FakeRedis()
    shelf_path = tmp_path / "global-cache.db"
    manager: CacheManager | None = None
    reloaded: CacheManager | None = None
    configure_global_cache_layers(
        shelve={"path": str(shelf_path)},
        redis={"client": fake_redis, "namespace": "tests:cache"},
        replace=True,
    )
    reset_global_cache_manager()
    try:
        manager = build_cache_manager()
        manager.register("persistent", lambda: {"value": 0})
        manager.store("persistent", {"value": 99})
        with manager.timer("persistent"):
            pass
        manager.increment_hit("persistent")
        assert fake_redis.get("tests:cache:persistent") is not None
        stats = manager.get_metrics("persistent")
        assert stats.hits == 1
        assert stats.timings == 1

        reset_global_cache_manager()
        reloaded = build_cache_manager()
        reloaded.register("persistent", lambda: {"value": -1})
        assert reloaded.get("persistent") == {"value": 99}
        reloaded.increment_hit("persistent")
        with reloaded.timer("persistent"):
            pass
        reloaded_stats = reloaded.get_metrics("persistent")
        assert reloaded_stats.hits == 1
        assert reloaded_stats.timings == 1
    finally:
        configure_global_cache_layers(replace=True)
        reset_global_cache_manager()
        _close_manager(manager)
        _close_manager(reloaded)


def test_graph_cache_manager_uses_layer_overrides(tmp_path):
    fake_redis = _FakeRedis()
    shelf_path = tmp_path / "graph-cache.db"

    G = nx.Graph()
    G.add_nodes_from([0, 1])
    G.graph[_GRAPH_CACHE_LAYERS_KEY] = {
        "shelve": {"path": str(shelf_path)},
        "redis": {"client": fake_redis, "namespace": "graph-cache"},
    }

    calls = 0

    def builder() -> str:
        nonlocal calls
        calls += 1
        return "persisted"

    assert edge_version_cache(G, "alpha", builder) == "persisted"
    assert calls == 1

    first_manager = G.graph.get("_tnfr_cache_manager")
    _close_manager(first_manager)
    G.graph.pop("_edge_cache_manager", None)
    G.graph.pop("_tnfr_cache_manager", None)

    builder_called = False

    def second_builder() -> str:
        nonlocal builder_called
        builder_called = True
        return "should-not-run"

    assert edge_version_cache(G, "alpha", second_builder) == "persisted"
    assert builder_called is False

    second_manager = G.graph.get("_tnfr_cache_manager")
    _close_manager(second_manager)


def test_edge_cache_persists_across_layers(tmp_path):
    shelf_path = tmp_path / "edge.db"
    fake_redis = _FakeRedis()

    redis_layer = RedisCacheLayer(client=fake_redis)
    shelf_layer = ShelveCacheLayer(str(shelf_path))
    initial_manager = CacheManager(layers=(redis_layer, shelf_layer))

    G = nx.Graph()
    G.add_nodes_from([0, 1])
    G.graph["_tnfr_cache_manager"] = initial_manager

    calls = 0

    def builder() -> str:
        nonlocal calls
        calls += 1
        return "persisted"

    assert edge_version_cache(G, "alpha", builder) == "persisted"
    assert calls == 1

    shelf_layer.close()
    G.graph.pop("_edge_cache_manager", None)

    restarted_shelf = ShelveCacheLayer(str(shelf_path))
    restarted_manager = CacheManager(
        layers=(RedisCacheLayer(client=fake_redis), restarted_shelf)
    )
    G.graph["_tnfr_cache_manager"] = restarted_manager

    builder_called = False

    def second_builder() -> str:
        nonlocal builder_called
        builder_called = True
        return "should-not-run"

    assert edge_version_cache(G, "alpha", second_builder) == "persisted"
    assert builder_called is False
    restarted_shelf.close()
