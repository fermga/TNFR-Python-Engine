"""Research snapshot statistics retain scalar, distance and applicability scope."""

import math

import networkx as nx
import pytest

from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_THETA
from tnfr.metrics.common import finite_population_std
from tnfr.physics.extended import (
    compute_phase_strain,
    compute_phase_vorticity,
    compute_reorganization_strain,
)


def _phase_pair(graph_type=nx.Graph, **edge_data):
    graph = graph_type()
    graph.add_edge(0, 1, **edge_data)
    graph.nodes[0][ALIAS_THETA[0]] = 0.0
    graph.nodes[1][ALIAS_THETA[0]] = 0.5
    return graph


def test_phase_strain_is_one_hop_spatial_variance_without_time():
    graph = nx.path_graph(3)
    for node, phase in enumerate((0.0, 0.2, 0.6)):
        graph.nodes[node][ALIAS_THETA[0]] = phase
    result = compute_phase_strain(graph)
    assert result == pytest.approx({0: 0.0, 1: 0.01, 2: 0.0}, abs=1e-16, rel=0)


@pytest.mark.parametrize("scale", [0, 2, -1, True, "1", math.nan])
def test_phase_strain_rejects_unimplemented_scale(scale):
    with pytest.raises(ValueError, match="scale"):
        compute_phase_strain(_phase_pair(), scale=scale)


@pytest.mark.parametrize(
    "edge_data,length", [({}, 1), ({"weight": 3}, 3), ({"length": 2, "weight": 0}, 2)]
)
def test_legacy_vorticity_can_be_nonzero_on_tree_and_reads_structural_length(
    edge_data, length
):
    graph = _phase_pair(**edge_data)
    assert nx.is_tree(graph)
    assert compute_phase_vorticity(graph) == {0: 0.5 / length, 1: -0.5 / length}
    # There is no cycle on this supplied support. Nonzero output is only a
    # signed neighbor statistic, not evidence of circulation or a defect.


def test_vorticity_parallel_lengths_use_shared_minimum_once_per_neighbor():
    graph = _phase_pair(nx.MultiDiGraph, length=4.0, weight=100.0)
    graph.add_edge(0, 1, length=2.0, weight=0.0)
    graph.add_node("isolate")
    assert compute_phase_vorticity(graph) == {0: 0.25, 1: 0.0, "isolate": 0.0}


@pytest.mark.parametrize("length", [0, -1, math.inf, math.nan, True, "1"])
def test_vorticity_rejects_undefined_or_invalid_inverse_distance(length):
    with pytest.raises(ValueError, match="length"):
        compute_phase_vorticity(_phase_pair(length=length))


def test_vorticity_preserves_finite_cancellation_of_large_inverse_distances():
    graph = nx.DiGraph()
    graph.add_edges_from(((0, 1), (0, 2)), length=math.ulp(0.0))
    for node, phase in enumerate((0.0, 0.5, -0.5)):
        graph.nodes[node][ALIAS_THETA[0]] = phase
    assert compute_phase_vorticity(graph) == {0: 0.0, 1: 0.0, 2: 0.0}
    graph.remove_edge(0, 2)
    with pytest.raises(ValueError, match="finite range"):
        compute_phase_vorticity(graph)


def test_reorganization_strain_keeps_missing_neighbor_pressure_as_zero():
    graph = nx.path_graph(3)
    graph.nodes[0][ALIAS_DNFR[0]] = 2.0
    # The neighbors of 1 have pressures 2 and the shared missing-value zero.
    assert compute_reorganization_strain(graph) == {0: 0.0, 1: 1.0, 2: 0.0}


@pytest.mark.parametrize(
    "bad_pressure,error",
    [
        (True, TypeError),
        ("2", TypeError),
        (math.nan, ValueError),
        (math.inf, ValueError),
    ],
)
def test_reorganization_strain_rejects_invalid_primary_alias(bad_pressure, error):
    graph = nx.path_graph(3)
    graph.nodes[0].update({ALIAS_DNFR[0]: bad_pressure, ALIAS_DNFR[1]: 2.0})
    with pytest.raises(error, match="pressure"):
        compute_reorganization_strain(graph)


@pytest.mark.parametrize(
    "pressures,expected",
    [
        ((1e308, -1e308), 1e308),
        ((1e308, 1e308), 0.0),
        ((1e308, math.nextafter(1e308, math.inf)), math.ulp(1e308) / 2),
        ((0.0, 2 * math.ulp(0.0)), math.ulp(0.0)),
        ((math.ulp(0.0), 3 * math.ulp(0.0)), math.ulp(0.0)),
    ],
)
def test_reorganization_standard_deviation_preserves_extreme_and_close_spreads(
    pressures, expected
):
    # For two samples the exact population standard deviation is |a-b|/2.
    assert finite_population_std(pressures) == expected
    graph = nx.path_graph(3)
    for node, pressure in zip((0, 2), pressures, strict=True):
        graph.nodes[node][ALIAS_DNFR[0]] = pressure
    assert compute_reorganization_strain(graph) == {0: 0.0, 1: expected, 2: 0.0}


def test_shared_population_std_retains_population_normalization_and_empty_zero():
    assert finite_population_std(()) == 0.0
    assert finite_population_std(value for value in (-1.0, 0.0, 1.0)) == math.sqrt(
        2 / 3
    )
