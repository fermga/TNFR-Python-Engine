"""Known exponential static-product profiles test the correlation distance contract."""

import math
from copy import deepcopy

import networkx as nx
import numpy as np
import pytest

from tnfr.operators.grammar_telemetry import warn_coherence_length_telemetry
from tnfr.physics import _coherence_fit as coherence_fit
from tnfr.physics import canonical
from tnfr.physics.telemetry import compute_structural_telemetry
from tnfr.physics.vectorized_ops import compute_coherence_length_vectorized
from tnfr.utils.cache import reset_global_cache


@pytest.fixture(autouse=True)
def _clear_field_cache():
    reset_global_cache()
    yield
    reset_global_cache()


@pytest.mark.parametrize("mode", ["standard", "high", "research"])
@pytest.mark.parametrize("size", [999, 1000, 1019])
def test_source_sample_boundary_is_independent_of_precision_label(mode, size):
    nodes = tuple(range(size))
    expected = nodes if size < 1000 else tuple(range(0, size, 20))
    assert coherence_fit.coherence_sources(nodes, mode) == expected


def _exponential_star(
    *, leaves=80, directed=False, explicit=True, reverse=False, zero=False
):
    """C0=1, Ci=exp(-ri/2), so Ci*Cj=exp(-dij/2) on a star.

    Repeated radii give enough pairs in each distance bin. This is a static
    synthetic field, not a simulated correlation or a fitted pressure law.
    """
    graph = nx.DiGraph() if directed else nx.Graph()
    nodes = ["center"] + [f"leaf-{i}" for i in range(leaves)]
    graph.add_nodes_from(reversed(nodes) if reverse else nodes)
    graph.nodes["center"]["delta_nfr"] = 0.0
    for i, node in enumerate(nodes[1:]):
        radius = 0.0 if zero and i == 0 else 1.0 + i % 4
        attributes = (
            {"weight": 0.5 + i % 3, "length": radius}
            if explicit
            else {
                "weight": radius,
            }
        )
        graph.add_edge("center", node, **attributes)
        graph.nodes[node]["delta_nfr"] = math.expm1(radius / 2.0)
    return graph


def _vectorized(graph, **kwargs):
    pressure = {node: graph.nodes[node]["delta_nfr"] for node in graph}
    return compute_coherence_length_vectorized(graph, list(graph), pressure, **kwargs)


@pytest.mark.parametrize("directed", (False, True))
def test_known_weighted_exponential_above_old_sampling_threshold_has_backend_parity(
    monkeypatch,
    directed,
):
    graph = _exponential_star(directed=directed)
    saved = deepcopy(dict(graph.nodes(data=True))), deepcopy(dict(graph.edges))
    vectorized = _vectorized(graph)
    monkeypatch.setattr(canonical, "_VECTORIZATION_AVAILABLE", False)
    streamed = canonical.estimate_coherence_length_with_provenance(graph)
    assert vectorized == pytest.approx(2.0, rel=2e-14)
    assert streamed.value == pytest.approx(vectorized, rel=2e-14)
    assert streamed.method == "autocorrelation_fit"
    assert "uncentered" in streamed.fit_quality
    assert "path-length units" in streamed.distance_weighting
    assert (
        "ordered outgoing" in streamed.sample_selection
        if directed
        else (streamed.sample_selection == "all unordered node pairs")
    )
    assert (dict(graph.nodes(data=True)), dict(graph.edges)) == saved


def test_directed_outgoing_pairs_do_not_disappear_when_center_is_ordered_last():
    forward = _exponential_star(directed=True)
    reversed_order = _exponential_star(directed=True, reverse=True)
    assert _vectorized(forward) == pytest.approx(2.0, rel=2e-14)
    assert _vectorized(reversed_order) == pytest.approx(2.0, rel=2e-14)


@pytest.mark.parametrize("explicit", (False, True))
def test_rescaling_declared_distances_rescales_fit_length_and_invalidates_cache(
    explicit,
):
    graph = _exponential_star(explicit=explicit)
    first = canonical.estimate_coherence_length_with_provenance(graph)
    key = "length" if explicit else "weight"
    for _, _, data in graph.edges(data=True):
        data[key] *= 3.0
    after = canonical.estimate_coherence_length_with_provenance(graph)
    assert first.method == after.method == "autocorrelation_fit"
    assert first.value == pytest.approx(2.0, rel=2e-14)
    assert after.value == pytest.approx(6.0, rel=2e-14)


@pytest.mark.parametrize("directed", (False, True))
def test_length_warning_uses_fit_geometry_and_is_invariant_to_length_units(directed):
    graph = _exponential_star(leaves=16, directed=directed)
    safe, before, _ = warn_coherence_length_telemetry(graph)
    expected_mean = 2.5 if directed else 80.0 / 17.0
    expected_diameter = 4.0 if directed else 8.0
    assert safe
    assert before["method"] == "autocorrelation_fit"
    assert before["xi_c"] == pytest.approx(2.0)
    assert before["mean_path_length"] == pytest.approx(expected_mean)
    assert before["diameter"] == pytest.approx(expected_diameter)

    for _, _, attributes in graph.edges(data=True):
        attributes["length"] *= 3.0
    still_safe, after, _ = warn_coherence_length_telemetry(graph)
    assert still_safe == safe
    assert after["severity"] == before["severity"]
    for key in ("xi_c", "mean_path_length", "diameter"):
        assert after[key] == pytest.approx(3.0 * before[key])


@pytest.mark.parametrize("size", [0, 1])
def test_missing_coherence_length_never_passes_a_warning_cut(size):
    safe, stats, message = warn_coherence_length_telemetry(nx.empty_graph(size))
    assert not safe
    assert math.isnan(stats["xi_c"])
    assert stats["method"] == "unavailable"
    assert stats["severity"] == "unavailable"
    assert "PASS" not in message


def test_spectral_fallback_is_not_compared_to_a_physical_path_length():
    graph = nx.path_graph(3)
    nx.set_edge_attributes(graph, 10.0, "length")
    safe, stats, message = warn_coherence_length_telemetry(graph)
    assert not safe
    assert math.isfinite(stats["xi_c"])
    assert stats["method"] == "spectral_gap"
    assert stats["severity"] == "not_comparable"
    assert math.isnan(stats["mean_path_length"])
    assert "dimensionless" in message


def test_explicit_metric_makes_fitted_length_independent_of_transport_conductance():
    graph = _exponential_star()
    before = canonical.estimate_coherence_length_with_provenance(graph)
    for _, _, data in graph.edges(data=True):
        data["weight"] *= 7.0
    after = canonical.estimate_coherence_length_with_provenance(graph)
    assert before == after
    assert after.value == pytest.approx(2.0, rel=2e-14)


def test_parallel_metric_lengths_use_minimum_not_conductance_sum():
    simple = _exponential_star()
    graph = nx.MultiGraph(simple)
    for source, target, data in simple.edges(data=True):
        graph.add_edge(source, target, length=data["length"] + 10.0, weight=100.0)
    assert _vectorized(graph) == pytest.approx(2.0, rel=2e-14)


def test_zero_distance_pair_is_omitted_from_the_pseudometric_fit():
    assert _vectorized(_exponential_star(zero=True)) == pytest.approx(2.0, rel=2e-14)


def test_large_lengths_do_not_allocate_bins_indexed_by_distance():
    graph = _exponential_star()
    for _, _, data in graph.edges(data=True):
        data["length"] *= 2.0**40
    assert _vectorized(graph) == pytest.approx(2.0**41, rel=2e-14)


def test_large_graph_source_sampling_is_backend_independent_and_disclosed(monkeypatch):
    graph = _exponential_star(leaves=1000)
    vectorized = _vectorized(graph)
    monkeypatch.setattr(canonical, "_VECTORIZATION_AVAILABLE", False)
    streamed = canonical.estimate_coherence_length_with_provenance(graph)
    assert vectorized == pytest.approx(2.0, rel=2e-14)
    assert streamed.value == pytest.approx(vectorized, rel=2e-14)
    assert "source IDs included in fit cache key" in streamed.sample_selection


def test_explicit_sample_materialization_allocates_only_selected_source_rows(
    monkeypatch,
):
    graph = _exponential_star(leaves=1000)
    nodes = tuple(graph)
    sources = coherence_fit.coherence_sources(nodes, "standard")
    pressure = {node: graph.nodes[node]["delta_nfr"] for node in graph}
    streamed = coherence_fit.fit_coherence_length(
        graph, nodes, pressure, sources=sources
    )
    allocated = []
    original_full = np.full

    def bounded_full(shape, *args, **kwargs):
        allocated.append(shape)
        assert shape == (len(sources), len(nodes))
        return original_full(shape, *args, **kwargs)

    monkeypatch.setattr(coherence_fit.np, "full", bounded_full)
    materialized = coherence_fit.fit_coherence_length(
        graph,
        nodes,
        pressure,
        sources=sources,
        materialize=True,
    )
    assert len(sources) < len(nodes)
    assert allocated == [(len(sources), len(nodes))]
    assert materialized == pytest.approx(2.0, rel=2e-14)
    assert materialized == pytest.approx(streamed, rel=2e-14)


def test_sampled_sources_still_index_declared_full_matrix_by_graph_order():
    graph = _exponential_star(leaves=12)
    nodes = tuple(graph)
    sources = (nodes[10], nodes[2])
    pressure = {node: graph.nodes[node]["delta_nfr"] for node in graph}
    matrix = nx.floyd_warshall_numpy(graph, nodelist=list(nodes), weight="length")
    for options in ({"materialize": True}, {"distance_matrix": matrix}):
        estimate = coherence_fit.fit_coherence_length(
            graph,
            nodes,
            pressure,
            sources=sources,
            **options,
        )
        assert estimate == pytest.approx(2.0, rel=2e-14)
    with pytest.raises(ValueError, match="square array"):
        coherence_fit.fit_coherence_length(
            graph,
            nodes,
            pressure,
            sources=sources,
            distance_matrix=matrix[:2],
        )


@pytest.mark.parametrize("invalid", (-1.0, math.inf, math.nan, True))
def test_invalid_graph_metric_is_not_hidden_by_the_spectral_fallback(invalid):
    graph = nx.path_graph(2)
    graph.edges[0, 1]["length"] = invalid
    with pytest.raises(ValueError, match="edge length"):
        canonical.estimate_coherence_length_with_provenance(graph)


def test_reachable_path_overflow_is_not_declared_unreachable():
    graph = nx.path_graph(3)
    nx.set_edge_attributes(graph, 1e308, "length")
    with pytest.raises(ValueError, match="path distance exceeds"):
        canonical.estimate_coherence_length_with_provenance(graph)


@pytest.mark.parametrize("corrupt", ("shape", "diagonal", "asymmetry", "text", "bool"))
def test_external_distance_matrix_domain_is_checked(corrupt):
    graph = _exponential_star(leaves=12)
    distances = nx.floyd_warshall_numpy(graph, nodelist=list(graph), weight="length")
    if corrupt == "shape":
        distances = distances[:-1]
    elif corrupt == "diagonal":
        distances[0, 0] = 1
    elif corrupt == "asymmetry":
        distances[0, 1] += 1
    elif corrupt == "text":
        distances = distances.astype(str)
    else:
        distances = distances.astype(bool)
    with pytest.raises(ValueError, match="distance_matrix"):
        _vectorized(graph, distance_matrix=distances)


def test_valid_declared_distance_matrix_is_not_mutated():
    graph = _exponential_star(leaves=12)
    distances = nx.floyd_warshall_numpy(graph, nodelist=list(graph), weight="length")
    saved = distances.copy()
    assert _vectorized(graph, distance_matrix=distances) == pytest.approx(
        2.0, rel=2e-14
    )
    assert np.array_equal(distances, saved)


def test_flat_field_uses_separately_identified_dimensionless_spectral_scale():
    graph = nx.path_graph(3)
    nx.set_edge_attributes(graph, 17.0, "length")
    result = canonical.estimate_coherence_length_with_provenance(graph)
    assert result.method == "spectral_gap"
    assert "dimensionless" in result.distance_weighting
    assert result.value == pytest.approx(1.0)


def test_outer_telemetry_cache_rebuilds_after_same_graph_node_reordering(monkeypatch):
    calls = _count_telemetry_coherence_calls(monkeypatch)
    graph = _exponential_star(leaves=12)
    first = compute_structural_telemetry(graph)
    assert compute_structural_telemetry(graph) == first
    assert len(calls) == 1
    nodes = [(node, dict(data)) for node, data in graph.nodes(data=True)]
    edges = [(u, v, dict(data)) for u, v, data in graph.edges(data=True)]
    graph.clear()
    graph.add_nodes_from(reversed(nodes))
    graph.add_edges_from(edges)
    second = compute_structural_telemetry(graph)
    assert second is not first
    assert tuple(second["grad_phi"]) == tuple(graph)
    assert second["xi_c"] == pytest.approx(_vectorized(graph), rel=2e-14)
    assert compute_structural_telemetry(graph) == second
    assert len(calls) == 2


def test_outer_telemetry_cache_binds_neighbor_order_even_with_fixed_node_order(
    monkeypatch,
):
    calls = _count_telemetry_coherence_calls(monkeypatch)
    graph = _exponential_star(leaves=12)
    first = compute_structural_telemetry(graph)
    nodes = tuple(graph)
    edges = [(u, v, dict(data)) for u, v, data in graph.edges(data=True)]
    old_neighbors = tuple(graph.neighbors("center"))
    graph.remove_edges_from(tuple(graph.edges()))
    graph.add_edges_from(reversed(edges))
    assert tuple(graph) == nodes
    assert tuple(graph.neighbors("center")) == tuple(reversed(old_neighbors))
    second = compute_structural_telemetry(graph)
    assert second is not first
    assert second["xi_c"] == pytest.approx(_vectorized(graph), rel=2e-14)
    assert len(calls) == 2


def test_outer_telemetry_cache_binds_the_coherence_numerical_path(monkeypatch):
    calls = _count_telemetry_coherence_calls(monkeypatch)
    graph = _exponential_star(leaves=12)
    first = compute_structural_telemetry(graph)
    monkeypatch.setattr(canonical, "_VECTORIZATION_AVAILABLE", False)
    second = compute_structural_telemetry(graph)
    assert second is not first
    assert second["xi_c"] == pytest.approx(first["xi_c"], rel=2e-14)
    assert len(calls) == 2


def _count_telemetry_coherence_calls(monkeypatch):
    from tnfr.physics import telemetry

    calls = []
    original = telemetry.estimate_coherence_length

    def counted(graph):
        calls.append(tuple(graph))
        return original(graph)

    monkeypatch.setattr(telemetry, "estimate_coherence_length", counted)
    return calls
