"""Tests covering multi-layer cache orchestration."""

from __future__ import annotations

import fnmatch
from typing import Any

import networkx as nx

from tnfr.cache import CacheLayer, CacheManager, RedisCacheLayer, ShelveCacheLayer
from tnfr.utils.cache import edge_version_cache


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
