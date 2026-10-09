"""Independent combinatorial and scope controls for an auxiliary benchmark."""

import math

import networkx as nx
import pytest

from benchmarks import emergent_dimension_dynamics as dimension


@pytest.mark.parametrize("size", [0, 1, 2, 4])
def test_clique_vertex_count_is_not_simplex_dimension(size):
    graph = nx.complete_graph(size)
    assert dimension._max_clique_size(graph) == size
    readout = dimension._simplex_readout([0.0] * size)
    assert readout["max_clique_size"] == size
    assert readout["simplex_dimension"] == (size - 1 if size else None)


def test_triangle_free_cycle_keeps_clique_dimension_one():
    assert dimension._max_clique_size(nx.cycle_graph(5)) == 2


def test_selected_threshold_uses_circular_separation():
    graph = dimension._resonant_graph([0.0, 2 * math.pi - 0.01, math.pi])
    assert set(graph.edges) == {(0, 1)}
    assert dimension.PHASE_THRESHOLD == pytest.approx(math.pi / 6)


def test_three_auxiliary_controls_retain_explicit_preparation_and_scope():
    result = dimension.run_dimension_controls()
    assert result["status"] == "AUXILIARY_COMPARISON"
    assert result["canonical_operators_executed"] is False
    assert result["physical_dimension_selection_demonstrated"] is False
    assert result["averaging_factor"] == 0.4
    assert result["averaging_steps"] == 60
    assert [row["max_clique_size"] for row in result["accretion"]] == [1, 2, 3, 4, 5]
    assert [row["simplex_dimension"] for row in result["accretion"]] == [0, 1, 2, 3, 4]
    rejected = result["incompatible_append"]
    assert rejected["before"]["node_count"] == 4
    assert rejected["after"]["node_count"] == 5
    assert rejected["before"]["simplex_dimension"] == 3
    assert rejected["after"]["simplex_dimension"] == 3
    averaged = result["averaging"]
    assert averaged["before"]["node_count"] == averaged["after"]["node_count"] == 5
    assert averaged["before"]["max_clique_size"] == 4
    assert averaged["after"]["max_clique_size"] == 5
