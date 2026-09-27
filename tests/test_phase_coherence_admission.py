"""Phase-order observations retain selected support and raw scalar admission."""

import math
import subprocess
import sys
from copy import deepcopy
from fractions import Fraction

import networkx as nx
import numpy as np
import pytest

from tnfr.constants.aliases import ALIAS_THETA
from tnfr.gamma import kuramoto_R_psi
from tnfr.metrics import phase_coherence
from tnfr.metrics.local_coherence import compute_radius_structural_coherence
from tnfr.metrics.phase_compatibility import (
    compute_network_phase_alignment,
    compute_phase_coupling_strength,
    is_phase_compatible,
)


@pytest.mark.parametrize(
    "invalid",
    [None, True, np.bool_(False), "0", math.nan, math.inf, Fraction(1, 2**1075)],
)
@pytest.mark.parametrize("nodes", [1, 2])
def test_phase_observations_reject_invalid_authoritative_alias(invalid, nodes):
    graph = nx.path_graph(nodes)
    graph.nodes[0].update({ALIAS_THETA[0]: invalid, ALIAS_THETA[1]: 0.0})
    for observe in (
        phase_coherence.compute_global_phase_coherence,
        lambda g: phase_coherence.compute_phase_alignment(g, 0),
        lambda g: compute_network_phase_alignment(g, 0),
    ):
        with pytest.raises((TypeError, ValueError)):
            observe(graph)


@pytest.mark.parametrize(
    "radius", [True, False, np.bool_(True), np.bool_(False), -1, 0.5, "1", None]
)
def test_structural_and_phase_observations_share_radius_admission(radius):
    graph = nx.path_graph(2)
    for observe in (
        compute_radius_structural_coherence,
        phase_coherence.compute_phase_alignment,
        compute_network_phase_alignment,
    ):
        with pytest.raises((TypeError, ValueError), match="radius"):
            observe(graph, 0, radius)


def test_radius_admission_preserves_integer_index_protocol():
    class Radius:
        def __index__(self):
            return 1

    graph = nx.path_graph(3)
    graph.nodes[2][ALIAS_THETA[0]] = "outside radius one"
    for radius in (1, np.int64(1), Radius()):
        assert phase_coherence.compute_phase_alignment(graph, 0, radius) == 1.0


@pytest.mark.parametrize("radius", [0, 1, 2])
def test_missing_center_never_becomes_a_perfect_phase_observation(radius):
    with pytest.raises(KeyError):
        phase_coherence.compute_phase_alignment(nx.path_graph(2), "absent", radius)


def test_radius_scope_uses_outgoing_unique_support_and_keeps_missing_alias_zero():
    graph = nx.MultiDiGraph()
    graph.add_edges_from([(0, 1), (0, 1), (0, 0), (1, 2), (3, 0)])
    nx.set_edge_attributes(graph, 0.0, "weight")
    graph.nodes[1][ALIAS_THETA[1]] = math.pi
    graph.nodes[2][ALIAS_THETA[0]] = "invalid outside radius one"
    graph.nodes[3][ALIAS_THETA[0]] = "incoming node is outside outgoing support"
    before = deepcopy((graph.graph, dict(graph.nodes(data=True)), list(graph.edges)))
    assert phase_coherence.compute_phase_alignment(graph, 0, np.int64(0)) == 1.0
    assert phase_coherence.compute_phase_alignment(
        graph, 0, np.int64(1)
    ) == pytest.approx(0.0, abs=1e-15)
    with pytest.raises((TypeError, ValueError)):
        phase_coherence.compute_phase_alignment(graph, 0, 2)
    with pytest.raises((TypeError, ValueError)):
        phase_coherence.compute_global_phase_coherence(graph)
    assert (graph.graph, dict(graph.nodes(data=True)), list(graph.edges)) == before


def test_local_and_global_order_reuse_shared_kuramoto_magnitude_without_state_writes():
    graph = nx.path_graph([object(), "node", (1, 2), 7])
    for node, phase in zip(graph, (0.1, 0.3, -0.2, 0.5)):
        graph.nodes[node][ALIAS_THETA[0]] = phase
    before_metadata = deepcopy(graph.graph)
    before_nodes = {node: deepcopy(data) for node, data in graph.nodes(data=True)}
    before_edges = [(u, v, deepcopy(data)) for u, v, data in graph.edges(data=True)]
    center = next(iter(graph))
    local = phase_coherence.compute_phase_alignment(graph, center, 3)
    global_order = phase_coherence.compute_global_phase_coherence(graph)
    assert local == global_order
    assert graph.graph == before_metadata
    assert dict(graph.nodes(data=True)) == before_nodes
    assert list(graph.edges(data=True)) == before_edges
    assert global_order == pytest.approx(kuramoto_R_psi(graph)[0], abs=1e-15)
    assert 0.0 <= global_order <= 1.0


def test_cancelling_pattern_has_zero_magnitude_without_inventing_direction():
    graph = nx.path_graph(4)
    for node, phase in zip(graph, (0.0, 0.0, math.pi, -math.pi)):
        graph.nodes[node][ALIAS_THETA[0]] = phase
    assert phase_coherence.compute_global_phase_coherence(graph) == 0.0
    assert phase_coherence.compute_phase_alignment(graph, 0, 3) == 0.0


def test_empty_and_singleton_conventions_are_preserved():
    assert phase_coherence.compute_global_phase_coherence(nx.Graph()) == 1.0
    graph = nx.empty_graph(1)
    assert phase_coherence.compute_global_phase_coherence(graph) == 1.0
    assert phase_coherence.compute_phase_alignment(graph, 0) == 1.0


@pytest.mark.parametrize(
    "invalid", [True, np.bool_(False), "0", math.nan, math.inf, Fraction(1, 2**1075)]
)
@pytest.mark.parametrize("position", [0, 1])
def test_pair_affinity_rejects_raw_invalid_phase(position, invalid):
    phases = [0.0, 0.0]
    phases[position] = invalid
    with pytest.raises((TypeError, ValueError)):
        compute_phase_coupling_strength(*phases)


@pytest.mark.parametrize(
    "invalid",
    [
        True,
        np.bool_(False),
        "0.5",
        -0.1,
        1.1,
        math.nan,
        Fraction(1, 2**1075),
        Fraction(2**1075 + 1, 2**1075),
    ],
)
def test_compatibility_rejects_invalid_threshold_instead_of_returning_a_decision(
    invalid,
):
    with pytest.raises((TypeError, ValueError)):
        is_phase_compatible(0.0, 0.0, threshold=invalid)


def test_pair_affinity_keeps_wrapped_geometry_and_inclusive_unit_thresholds():
    assert compute_phase_coupling_strength(0.0, 0.0) == 1.0
    assert compute_phase_coupling_strength(0.0, math.pi / 2) == 0.5
    assert compute_phase_coupling_strength(0.0, math.pi) == 0.0
    assert compute_phase_coupling_strength(0.1, 2 * math.pi - 0.1) == pytest.approx(
        1 - 0.2 / math.pi
    )
    assert is_phase_compatible(0.0, 0.0, threshold=1.0)
    assert is_phase_compatible(0.0, math.pi, threshold=0.0)
    assert is_phase_compatible(0.0, math.pi / 2, threshold=0.5)
    assert not is_phase_compatible(0.0, 0.31 * math.pi, threshold=0.7)


def test_phase_order_cold_import_has_no_gamma_metrics_cycle():
    code = """
import networkx as nx
from tnfr.metrics.phase_coherence import compute_phase_alignment
from tnfr.metrics.phase_compatibility import compute_network_phase_alignment
graph = nx.path_graph(2)
assert compute_phase_alignment(graph, 0) == 1.0
assert compute_network_phase_alignment(graph, 0) == 1.0
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
