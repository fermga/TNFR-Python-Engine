"""Capacity/synchrony snapshots do not count observed oscillations."""

import math
from copy import deepcopy

import networkx as nx
import pytest

from tnfr.physics.structural_diffusion import compute_nodal_pulse
from tnfr.sdk.simple import Network


def _snapshot(capacities):
    graph = nx.cycle_graph(len(capacities))
    for node, capacity in enumerate(capacities):
        graph.nodes[node].update(EPI=0.5, nu_f=capacity, theta=0.0, delta_nfr=0.0)
    return graph


def test_every_positive_represented_capacity_counts_without_activity_cutoff():
    graph = _snapshot(
        (
            0.0,
            -0.0,
            math.ulp(0.0),
            math.nextafter(1e-9, 0.0),
            1e-9,
            math.nextafter(1e-9, math.inf),
            1.0,
        )
    )
    before = deepcopy(dict(graph.nodes(data=True)))

    result = compute_nodal_pulse(graph)

    assert result["n_pulsing"] == 5
    assert result["n_nodes"] == 7
    assert dict(graph.nodes(data=True)) == before


def test_sdk_count_is_capacity_presence_even_at_zero_nodal_rate():
    graph = _snapshot((math.ulp(0.0), 0.25, 0.5, 1.0))
    result = Network(graph).resonance()

    assert result["n_pulsing"] == 4
    assert all(
        data["nu_f"] * data["delta_nfr"] == 0.0 for _, data in graph.nodes(data=True)
    )
    assert result["phase_coherence"] == pytest.approx(1.0)
    assert result["mean_local_resonance"] == pytest.approx(1.0)

    # The same supplied phases retain their synchrony with no capacity.
    nx.set_node_attributes(graph, 0.0, "nu_f")
    inactive = Network(graph).resonance()
    assert inactive["n_pulsing"] == 0
    assert inactive["phase_coherence"] == result["phase_coherence"]
    assert inactive["mean_local_resonance"] == result["mean_local_resonance"]


def test_empty_snapshot_retains_legacy_schema_without_active_nodes():
    result = compute_nodal_pulse(nx.Graph())

    assert set(result) == {
        "mean_frequency",
        "frequency_spread",
        "phase_coherence",
        "mean_local_resonance",
        "resonance_gate",
        "n_pulsing",
        "n_nodes",
    }
    assert result["n_pulsing"] == result["n_nodes"] == 0
