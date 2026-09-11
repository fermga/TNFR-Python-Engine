"""SDK regressions for local and aggregate structural-coherence semantics."""

from __future__ import annotations

import networkx as nx
import pytest

from tnfr.sdk.simple import Network


def _two_node_network() -> Network:
    graph = nx.path_graph(2)
    graph.nodes[0].update(
        EPI=1.0,
        nu_f=1.0,
        phase=0.0,
        delta_nfr=0.0,
        dEPI_dt=0.0,
    )
    graph.nodes[1].update(
        EPI=1.0,
        nu_f=1.0,
        phase=0.2,
        delta_nfr=2.0,
        dEPI_dt=2.0,
    )
    return Network(graph)


def test_nodal_state_coherence_includes_predicted_nodal_rate() -> None:
    graph = nx.Graph()
    graph.add_node(
        "n",
        EPI=1.0,
        nu_f=2.0,
        phase=0.0,
        delta_nfr=1.0,
    )

    state = Network(graph).nodal_state("n")

    assert state.expected_depi_dt == pytest.approx(2.0)
    assert state.coherence == pytest.approx(1.0 / 4.0)


def test_nodal_equilibrium_checks_pressure_and_predicted_rate() -> None:
    graph = nx.Graph()
    graph.add_node(
        "n",
        EPI=1.0,
        nu_f=4.0,
        phase=0.0,
        delta_nfr=5.0e-4,
    )

    state = Network(graph).nodal_state("n", equilibrium_tolerance=1.0e-3)

    assert abs(state.delta_nfr) <= 1.0e-3
    assert abs(state.expected_depi_dt) > 1.0e-3
    assert state.equilibrium is False


def test_nodal_scan_distinguishes_total_from_mean_local_coherence() -> None:
    report = _two_node_network().nodal_scan()

    assert report.total_coherence() == pytest.approx(1.0 / 3.0)
    assert report.mean_local_coherence() == pytest.approx(3.0 / 5.0)
    assert "C=0.333" in report.summary()

    aggregate = report.to_dict()["aggregate"]
    assert aggregate["coherence"] == pytest.approx(1.0 / 3.0)
    assert aggregate["mean_local_coherence"] == pytest.approx(3.0 / 5.0)
    assert aggregate["mean_abs_dnfr"] == pytest.approx(1.0)
    assert aggregate["mean_abs_depi_dt"] == pytest.approx(1.0)
    assert aggregate["depi_dt_source"] == "nodal_equation_prediction"


def test_network_nfr_uses_the_same_aggregate_reduction_as_network_coherence() -> None:
    network = _two_node_network()

    result = network.nfr()

    assert result["coherence"] == pytest.approx(1.0 / 3.0)
    assert result["coherence"] == pytest.approx(network.coherence())
    assert result["mean_local_coherence"] == pytest.approx(3.0 / 5.0)
    assert result["mean_abs_dnfr"] == pytest.approx(1.0)
    assert result["mean_abs_depi_dt"] == pytest.approx(1.0)


def test_graph_nfr_observation_uses_canonical_equilibrium_tolerance() -> None:
    observation = _two_node_network().nfr_observation()

    assert observation.equilibrium_tolerance == pytest.approx(1.0e-3)