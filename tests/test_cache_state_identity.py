"""Exact-state and defensive-copy contracts for shared caches."""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest

from tnfr.constants.aliases import (
    ALIAS_DEPI,
    ALIAS_DNFR,
    ALIAS_EPI,
    ALIAS_THETA,
    ALIAS_VF,
)
from tnfr.dynamics.structural_cache import StructuralCoherenceCache
from tnfr.utils.cache import _compute_dependency_hash


def _graph() -> nx.Graph:
    graph = nx.path_graph(2)
    for node, epi in enumerate((0.0, 1.0)):
        graph.nodes[node].update(
            {
                ALIAS_EPI[0]: epi,
                ALIAS_VF[0]: 1.0,
                ALIAS_THETA[0]: 0.0,
                ALIAS_DNFR[0]: 0.2,
                ALIAS_DEPI[0]: 0.0,
            }
        )
    return graph


def test_dependency_hash_and_structural_cache_track_depi() -> None:
    graph = _graph()
    dependencies = {"graph_topology", "node_depi"}
    cache = StructuralCoherenceCache()

    first_hash = _compute_dependency_hash(graph, dependencies)
    first = cache.get_structural_fields(graph)
    for node in graph:
        graph.nodes[node][ALIAS_DEPI[0]] = 3.0
    second_hash = _compute_dependency_hash(graph, dependencies)
    second = cache.get_structural_fields(graph)

    assert first_hash != second_hash
    assert first.topology_hash != second.topology_hash
    assert second.coherence < first.coherence
    assert cache.misses == 2


def test_resonance_cache_owns_inputs_and_returns_read_only_copies() -> None:
    cache = StructuralCoherenceCache()
    frequencies = np.array([1.0, 2.0])
    amplitudes = np.array([0.25, 0.75])
    phases = np.array([0.1, 0.2])

    key = cache.cache_resonance_pattern(frequencies, amplitudes, phases)
    frequencies[:] = 99.0
    first = cache.get_resonance_pattern(key)

    assert first is not None
    np.testing.assert_allclose(first.frequencies, [1.0, 2.0])
    with pytest.raises(ValueError):
        first.amplitudes[0] = 9.0

    first.usage_count = 999
    second = cache.get_resonance_pattern(key)
    assert second is not None
    assert second.usage_count == 3


@pytest.mark.parametrize(
    ("frequencies", "amplitudes", "phases", "message"),
    [
        (
            np.array([1.0, math.nan]),
            np.ones(2),
            np.zeros(2),
            "finite",
        ),
        (
            np.ones(2),
            np.ones(3),
            np.zeros(2),
            "identical shapes",
        ),
        (
            np.array([True, False]),
            np.ones(2),
            np.zeros(2),
            "numeric",
        ),
        (
            np.ones((1, 2)),
            np.ones(2),
            np.zeros(2),
            "one-dimensional",
        ),
    ],
)
def test_resonance_cache_rejects_invalid_channels(
    frequencies,
    amplitudes,
    phases,
    message: str,
) -> None:
    cache = StructuralCoherenceCache()

    with pytest.raises(ValueError, match=message):
        cache.cache_resonance_pattern(frequencies, amplitudes, phases)

    assert cache._resonance_cache == {}
