"""Analytic checks for exact potential and explicitly requested approximations."""

from __future__ import annotations

import math
import random

import networkx as nx
import numpy as np
import pytest

from tnfr.alias import set_attr
from tnfr.config import get_precision_mode, set_precision_mode
from tnfr.constants.aliases import ALIAS_DNFR
from tnfr.physics.canonical import (
    _PHI_S_DISTANCE_CACHE,
    compute_structural_potential,
)
from tnfr.physics.vectorized_ops import (
    compute_phi_s_exact_vectorized,
    compute_phi_s_landmarks_vectorized,
)
from tnfr.physics.fields import classify_nodal_topology, path_integrated_gradient
from tnfr.utils.cache import reset_global_cache


@pytest.fixture(autouse=True)
def clear_potential_caches():
    previous_mode = get_precision_mode()
    reset_global_cache()
    _PHI_S_DISTANCE_CACHE.clear()
    yield
    set_precision_mode(previous_mode)
    reset_global_cache()
    _PHI_S_DISTANCE_CACHE.clear()


def _set_pressure(graph, pressure):
    for node, value in pressure.items():
        set_attr(graph.nodes[node], ALIAS_DNFR, value)


def test_large_default_potential_matches_analytic_path():
    graph = nx.path_graph(600)
    _set_pressure(graph, {node: 0.1 for node in graph})
    expected = {
        i: math.fsum(0.1 / (i - j) ** 2 for j in graph if i != j)
        for i in graph
    }
    assert compute_structural_potential(graph) == pytest.approx(expected, abs=1e-12)


def test_large_default_weighted_directed_signed_path():
    graph = nx.DiGraph()
    graph.add_weighted_edges_from((i, i + 1, 0.5) for i in range(500))
    pressure = {node: (-1.0) ** node * 0.1 for node in graph}
    _set_pressure(graph, pressure)
    expected = {
        i: math.fsum(pressure[j] / (0.5 * (j - i)) ** 2 for j in graph if j > i)
        for i in graph
    }
    assert compute_structural_potential(graph) == pytest.approx(expected, abs=1e-12)


def test_explicit_length_separates_geometry_from_transport_conductance():
    graph = nx.path_graph(3)
    graph.edges[0, 1].update(weight=100.0, length=0.5)
    graph.edges[1, 2].update(weight=0.25, length=1.5)
    _set_pressure(graph, {node: 1.0 for node in graph})

    potential = compute_structural_potential(graph)

    assert potential == pytest.approx(
        {
            0: 1.0 / 0.5**2 + 1.0 / 2.0**2,
            1: 1.0 / 0.5**2 + 1.0 / 1.5**2,
            2: 1.0 / 2.0**2 + 1.0 / 1.5**2,
        }
    )


def test_weight_remains_legacy_length_fallback():
    graph = nx.path_graph(2)
    graph.edges[0, 1]["weight"] = 2.0
    _set_pressure(graph, {0: 1.0, 1: 1.0})

    assert compute_structural_potential(graph) == {0: 0.25, 1: 0.25}


@pytest.mark.parametrize("mode", ["standard", "high", "research"])
@pytest.mark.parametrize("leaf_count", [3, 50])
def test_exact_potential_preserves_signed_residual_across_size_and_precision(mode, leaf_count):
    graph = nx.star_graph(leaf_count)
    pressure = {node: 0.0 for node in graph}
    pressure.update({1: 1e30, 2: 1.0, 3: -1e30})
    _set_pressure(graph, pressure)
    set_precision_mode(mode)
    # All leaves have unit distance to the center. Their exact contribution
    # is 1e30 + 1 - 1e30 = 1, beyond an uncompensated float64/80-bit sum.
    assert compute_structural_potential(graph)[0] == 1.0


def test_research_potential_keeps_compensation_when_longdouble_aliases_float64(monkeypatch):
    import tnfr.physics.canonical as canonical

    graph = nx.star_graph(50)
    pressure = {node: 0.0 for node in graph}
    pressure.update({1: 1e16, 2: 1.0, 3: -1e16})
    _set_pressure(graph, pressure)
    set_precision_mode("research")
    monkeypatch.setattr(canonical, "_get_precision_dtype", lambda: np.float64)
    assert compute_structural_potential(graph)[0] == 1.0


@pytest.mark.parametrize("dtype", [np.float64, np.longdouble])
@pytest.mark.parametrize("force_fallback", [False, True])
def test_direct_dense_potential_and_fallback_preserve_signed_residual(
    dtype, force_fallback, monkeypatch,
):
    graph = nx.star_graph(3)
    pressure = {0: 0.0, 1: 1e30, 2: 1.0, 3: -1e30}
    if force_fallback:
        def unavailable_floyd_warshall(*args, **kwargs):
            raise RuntimeError("Floyd-Warshall unavailable")

        monkeypatch.setattr(nx, "floyd_warshall_numpy", unavailable_floyd_warshall)
    result = compute_phi_s_exact_vectorized(graph, list(graph), pressure, 2.0, dtype=dtype)
    assert result[0] == 1.0


@pytest.mark.skipif(
    np.finfo(np.longdouble).eps >= np.finfo(float).eps,
    reason="longdouble has no extended mantissa on this platform",
)
def test_research_potential_keeps_extended_intermediate_range():
    graph = nx.star_graph(50)
    nx.set_edge_attributes(graph, 1e200, "weight")
    pressure = {node: 0.0 for node in graph}
    pressure[1] = 1e300
    _set_pressure(graph, pressure)
    set_precision_mode("research")
    # Squaring 1e200 overflows float64, but the final potential is 1e-100.
    assert compute_structural_potential(graph)[0] == pytest.approx(1e-100, rel=1e-14, abs=0)


def test_landmark_distance_uses_directed_path_through_landmark():
    graph = nx.DiGraph()
    graph.add_weighted_edges_from([(0, 1, 0.25), (1, 2, 0.25), (2, 3, 0.25)])
    nodes = list(graph)
    pressure = {node: 0.2 for node in graph}
    distances = {1: nx.single_source_dijkstra_path_length(graph, 1, weight="weight")}
    result = compute_phi_s_landmarks_vectorized(graph, nodes, pressure, 2.0, [1], distances)
    # Every reachable path from node 0 passes through landmark 1.
    assert result[0] == pytest.approx(0.2 / 0.25**2 + 0.2 / 0.5**2 + 0.2 / 0.75**2)
    # Sink 3 cannot reach the landmark or any other node.
    assert result[3] == 0.0


def test_landmark_potential_has_no_disconnected_leakage():
    graph = nx.Graph([(0, 1), (2, 3)])
    pressure = {0: 1.0, 1: 1.0, 2: 0.0, 3: 0.0}
    distances = {0: {0: 0, 1: 1}, 2: {2: 0, 3: 1}}
    result = compute_phi_s_landmarks_vectorized(
        graph, list(graph), pressure, 2.0, [0, 2], distances
    )
    assert result == {0: 1.0, 1: 1.0, 2: 0.0, 3: 0.0}


def test_validation_enforces_global_error_for_signed_pressure():
    graph = nx.path_graph(120)
    pressure = {node: (-1.0) ** node * 0.1 for node in graph}
    _set_pressure(graph, pressure)
    expected = {
        i: math.fsum(pressure[j] / (i - j) ** 2 for j in graph if i != j)
        for i in graph
    }
    result = compute_structural_potential(
        graph, landmark_ratio=0.025, validate=True,
        error_epsilon=0.0, max_refinements=0, sample_size=1,
    )
    actual = {node: result[node] for node in graph}
    assert actual == pytest.approx(expected, abs=1e-12)
    assert result["__phi_s_rmae__"] == 0.0
    assert result["__phi_s_fallback_exact__"] == 1.0


def test_explicit_landmarks_are_deterministic_and_do_not_consume_rng():
    graph = nx.path_graph(40)
    _set_pressure(graph, {node: 0.1 for node in graph})
    state = random.getstate()
    first = compute_structural_potential(graph, landmark_ratio=0.1)
    assert random.getstate() == state
    reset_global_cache()
    _PHI_S_DISTANCE_CACHE.clear()
    assert compute_structural_potential(graph, landmark_ratio=0.1) == first


def test_weighted_multigraph_default_uses_minimum_parallel_distance():
    graph = nx.MultiGraph()
    graph.add_weighted_edges_from((i, i + 1, 2.0) for i in range(59))
    graph.add_weighted_edges_from((i, i + 1, 0.5) for i in range(59))
    _set_pressure(graph, {node: 0.1 for node in graph})
    expected = {
        i: math.fsum(0.1 / (0.5 * (i - j)) ** 2 for j in graph if i != j)
        for i in graph
    }
    assert compute_structural_potential(graph) == pytest.approx(expected, abs=1e-12)


def test_unit_source_nodal_geometry_matches_weighted_potential():
    graph = nx.Graph()
    graph.add_weighted_edges_from([(0, 1, 0.5), (1, 2, 1.5)])
    _set_pressure(graph, {node: 1.0 for node in graph})
    assert classify_nodal_topology(graph)["centrality"] == pytest.approx(
        {0: 4.25, 1: 4.0 + 1.0 / 1.5**2, 2: 0.25 + 1.0 / 1.5**2}
    )


def test_path_integral_has_one_term_per_edge_and_zero_empty_path():
    graph = nx.path_graph(3)
    nx.set_node_attributes(graph, {0: 0.0, 1: 0.2, 2: 0.6}, "theta")
    # |∇φ|(0)=.2 and |∇φ|(1)=.3; the target's .4 is excluded.
    assert path_integrated_gradient(graph, 0, 2) == pytest.approx(0.5)
    assert path_integrated_gradient(graph, 1, 1) == 0.0


@pytest.mark.parametrize("epsilon", [-1.0, math.nan, math.inf])
def test_validation_rejects_invalid_error_tolerance(epsilon):
    with pytest.raises(ValueError, match="error_epsilon"):
        compute_structural_potential(
            nx.path_graph(4), landmark_ratio=0.1, validate=True,
            error_epsilon=epsilon,
        )


def test_validation_cannot_certify_nonfinite_pressure():
    graph = nx.path_graph(4)
    _set_pressure(graph, {node: math.nan for node in graph})
    with pytest.raises(ValueError, match="finite"):
        compute_structural_potential(graph, landmark_ratio=0.1, validate=True)


def test_zero_distance_exclusion_agrees_for_explicit_landmarks():
    graph = nx.Graph()
    graph.add_weighted_edges_from([(0, 1, 0.0), (1, 2, 0.5)])
    _set_pressure(graph, {node: 0.1 for node in graph})
    expected = {0: 0.4, 1: 0.4, 2: 0.8}
    assert compute_structural_potential(graph) == pytest.approx(expected)
    assert compute_structural_potential(graph, landmark_ratio=0.1) == pytest.approx(expected)


@pytest.mark.parametrize("first_length", [0.0, 1.0])
def test_validation_rejects_nonfinite_exact_output_on_both_fallback_paths(first_length):
    import numpy as np

    graph = nx.Graph()
    graph.add_weighted_edges_from([(0, 1, first_length), (1, 2, 1e-155)])
    _set_pressure(graph, {node: 1.0 for node in graph})
    # A unit source at distance 1e-155 contributes 1e310, beyond float64.
    # Neither the ordinary reference path nor the zero-distance compatibility
    # fallback may certify an infinite returned field as RMAE zero.
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="finite exact potentials"):
            compute_structural_potential(
                graph, landmark_ratio=0.1, validate=True, error_epsilon=0.0,
            )


def test_scalar_and_vectorized_landmark_paths_agree(monkeypatch):
    import tnfr.physics.canonical as canonical

    graph = nx.DiGraph()
    graph.add_weighted_edges_from((i, i + 1, 0.5) for i in range(9))
    _set_pressure(graph, {node: (-1.0)**node * 0.1 for node in graph})
    expected = compute_structural_potential(graph, landmark_ratio=0.3)
    reset_global_cache()
    monkeypatch.setattr(canonical, "_VECTORIZATION_AVAILABLE", False)
    assert compute_structural_potential(graph, landmark_ratio=0.3) == pytest.approx(expected)
