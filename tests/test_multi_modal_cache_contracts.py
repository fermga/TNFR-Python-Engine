"""Contracts for the exact-state shared multimodal cache."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from tnfr.dynamics.multi_modal_cache import (
    CacheEntryType,
    CacheInvalidationTrigger,
    TNFRUnifiedMultiModalCache,
)


def _mixed_graph() -> nx.Graph:
    graph = nx.Graph(model="test")
    graph.add_nodes_from([0, "node", (2, "x")])
    graph.add_edge(0, "node", weight=1.0)
    graph.nodes[0]["EPI"] = 0.0
    graph.nodes["node"]["EPI"] = 1.0
    graph.nodes[(2, "x")]["EPI"] = 2.0
    return graph


def test_signature_tracks_full_state_and_weight_with_mixed_labels() -> None:
    graph = _mixed_graph()
    cache = TNFRUnifiedMultiModalCache()

    first = cache.compute_graph_signature(graph)
    graph.nodes[(2, "x")]["late_channel"] = np.arange(20.0)
    second = cache.compute_graph_signature(graph)
    graph.edges[0, "node"]["weight"] = 2.0
    third = cache.compute_graph_signature(graph)

    assert len({first, second, third}) == 3


def test_cache_result_is_detached_on_store_and_hit() -> None:
    graph = _mixed_graph()
    cache = TNFRUnifiedMultiModalCache()
    calls = 0

    def compute():
        nonlocal calls
        calls += 1
        return {"values": np.array([1.0, 2.0]), "nested": [{"x": 3}]}

    first = cache.get(CacheEntryType.NODAL_STATE, graph, computation_func=compute)
    first["values"][0] = 99.0
    first["nested"][0]["x"] = 99
    second = cache.get(CacheEntryType.NODAL_STATE, graph, computation_func=compute)
    second["values"][1] = 88.0
    third = cache.get(CacheEntryType.NODAL_STATE, graph, computation_func=compute)

    assert calls == 1
    np.testing.assert_allclose(third["values"], [1.0, 2.0])
    assert third["nested"] == [{"x": 3}]


def test_state_mutation_never_hits_previous_entry() -> None:
    graph = _mixed_graph()
    cache = TNFRUnifiedMultiModalCache()
    calls = 0

    def compute():
        nonlocal calls
        calls += 1
        return float(graph.nodes["node"]["EPI"])

    assert cache.get(
        CacheEntryType.NODAL_STATE,
        graph,
        computation_func=compute,
    ) == 1.0
    graph.nodes["node"]["EPI"] = 4.0
    assert cache.get(
        CacheEntryType.NODAL_STATE,
        graph,
        computation_func=compute,
    ) == 4.0
    assert calls == 2


def test_graph_scoped_invalidation_finds_old_signatures_after_mutation() -> None:
    graph = _mixed_graph()
    other = _mixed_graph()
    cache = TNFRUnifiedMultiModalCache()
    for subject in (graph, other):
        cache.get(
            CacheEntryType.NODAL_STATE,
            subject,
            computation_func=lambda: {"ok": True},
        )

    graph.nodes[0]["EPI"] = 8.0
    removed = cache.invalidate(
        CacheInvalidationTrigger.OPERATOR_APPLICATION,
        G=graph,
    )

    assert removed == 1
    assert len(cache._cache) == 1


def test_parameter_keys_are_array_shape_aware() -> None:
    graph = _mixed_graph()
    cache = TNFRUnifiedMultiModalCache()
    calls = 0

    def compute():
        nonlocal calls
        calls += 1
        return calls

    first = cache.get(
        CacheEntryType.FFT_OPERATION,
        graph,
        parameters={"kernel": np.array([1.0, 2.0])},
        computation_func=compute,
    )
    second = cache.get(
        CacheEntryType.FFT_OPERATION,
        graph,
        parameters={"kernel": np.array([[1.0, 2.0]])},
        computation_func=compute,
    )

    assert (first, second) == (1, 2)


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), True])
def test_cache_rejects_invalid_capacity(value) -> None:
    with pytest.raises(ValueError, match="positive finite"):
        TNFRUnifiedMultiModalCache(max_size_mb=value)


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf"), True])
def test_cache_rejects_invalid_importance(value) -> None:
    graph = _mixed_graph()
    cache = TNFRUnifiedMultiModalCache()

    with pytest.raises(ValueError, match="finite and nonnegative"):
        cache.get(
            CacheEntryType.NODAL_STATE,
            graph,
            computation_func=lambda: 1.0,
            mathematical_importance=value,
        )


def test_cache_statistics_distinguish_hits_from_unproved_cross_engine_reuse() -> None:
    graph = _mixed_graph()
    cache = TNFRUnifiedMultiModalCache()
    cache.get(CacheEntryType.NODAL_STATE, graph, computation_func=lambda: 1.0)
    cache.get(CacheEntryType.NODAL_STATE, graph, computation_func=lambda: 2.0)

    stats = cache.get_statistics()
    assert stats.cache_hit_count == 1
    assert stats.cross_engine_reuse_count == 0
