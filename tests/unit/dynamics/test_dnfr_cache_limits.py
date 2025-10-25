"""Tests for graph cache limit configuration persistence."""

from __future__ import annotations

from tnfr.cache import CacheCapacityConfig
from tnfr.utils.cache import (
    DNFR_PREP_STATE_KEY,
    _GRAPH_CACHE_MANAGER_KEY,
    _graph_cache_manager,
    configure_graph_cache_limits,
)


def test_configure_limits_store_snapshot(graph_canon) -> None:
    G = graph_canon()
    overrides = {DNFR_PREP_STATE_KEY: 17, "custom": None}

    snapshot = configure_graph_cache_limits(
        G,
        default_capacity=256,
        overrides=overrides,
        replace_overrides=True,
    )

    assert isinstance(snapshot, CacheCapacityConfig)
    assert snapshot.default_capacity == 256
    assert snapshot.overrides == overrides

    expected_config = {"default_capacity": 256, "overrides": overrides}
    assert G.graph["_tnfr_cache_config"] == expected_config

    manager = _graph_cache_manager(G.graph)
    assert manager.export_config() == snapshot
    assert manager.get_capacity(DNFR_PREP_STATE_KEY) == 17
    assert manager.get_capacity("unregistered") == 256


def test_manager_reloads_persisted_limits(graph_canon) -> None:
    G = graph_canon()
    overrides = {DNFR_PREP_STATE_KEY: 9}

    configure_graph_cache_limits(G, default_capacity=64, overrides=overrides)
    first_manager = _graph_cache_manager(G.graph)
    assert first_manager.get_capacity(DNFR_PREP_STATE_KEY) == 9

    G.graph.pop(_GRAPH_CACHE_MANAGER_KEY, None)

    reloaded_manager = _graph_cache_manager(G.graph)
    assert reloaded_manager is not first_manager
    assert reloaded_manager.get_capacity(DNFR_PREP_STATE_KEY) == 9
    assert reloaded_manager.get_capacity("other") == 64


def test_manager_ignores_non_mapping_config(graph_canon) -> None:
    G = graph_canon()

    baseline_manager = _graph_cache_manager(G.graph)
    baseline_snapshot = baseline_manager.export_config()

    G.graph["_tnfr_cache_config"] = object()
    G.graph.pop(_GRAPH_CACHE_MANAGER_KEY, None)

    reloaded_manager = _graph_cache_manager(G.graph)
    assert reloaded_manager.export_config() == baseline_snapshot
    assert reloaded_manager.get_capacity(DNFR_PREP_STATE_KEY) == baseline_snapshot.default_capacity
