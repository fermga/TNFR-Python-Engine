"""SDK regressions for local and aggregate structural-coherence semantics."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import networkx as nx
import numpy as np
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


@pytest.mark.parametrize(
    "invalid",
    [
        -1.0,
        1.1,
        True,
        "0.5",
        float("nan"),
        Fraction(1, 10**400),
        Fraction(2**55 + 1, 2**55),
    ],
)
def test_local_coherence_reports_reject_invalid_samples_without_taking_magnitudes(
    invalid,
):
    report = _two_node_network().nodal_scan()
    report.nodes[0].coherence = invalid

    with pytest.raises((TypeError, ValueError)):
        report.mean_local_coherence()
    with pytest.raises((TypeError, ValueError)):
        report.to_dict()
    with pytest.raises((TypeError, ValueError)):
        report.nodes[0].to_dict()


def test_local_coherence_mean_preserves_representable_subnormal_samples():
    report = _two_node_network().nodal_scan()
    tiny = float.fromhex("0x0.0000000000001p-1022")
    report.nodes[0].coherence = np.float64(tiny)
    report.nodes[1].coherence = Fraction.from_float(tiny)

    assert report.mean_local_coherence() == tiny
    assert report.to_dict()["aggregate"]["mean_local_coherence"] == tiny


def test_string_key_collision_cannot_silently_drop_a_scanned_node():
    network = _two_node_network()
    nx.relabel_nodes(network.G, {0: 1, 1: "1"}, copy=False)
    report = network.nodal_scan()

    assert len(report.nodes) == 2
    with pytest.raises(ValueError, match="collide"):
        report.to_dict()


def test_mixed_noncolliding_node_labels_remain_complete_and_exports_are_detached():
    network = _two_node_network()
    nx.relabel_nodes(network.G, {0: (0, "a"), 1: "second"}, copy=False)
    report = network.nodal_scan()
    data = report.to_dict()

    assert len(data["nodes"]) == data["aggregate"]["count"] == 2
    assert {row["node"] for row in data["nodes"].values()} == set(network.G)
    data["nodes"]["second"]["coherence"] = 0.0
    data["aggregate"]["count"] = 0
    assert report.nodes["second"].coherence == pytest.approx(0.2)
    assert report.to_dict()["aggregate"]["count"] == 2


@pytest.mark.parametrize(
    "name,invalid",
    [
        ("equilibrium", "false"),
        ("active", 1),
        ("observed_crossed", "no"),
        ("evidence_available", np.nan),
    ],
)
def test_mutated_report_verdicts_cannot_be_promoted_by_python_truthiness(name, invalid):
    report = _two_node_network().nodal_scan()
    state = report.nodes[0]
    setattr(state, name, invalid)

    with pytest.raises(TypeError, match=name):
        state.to_dict()
    with pytest.raises(TypeError, match=name):
        report.to_dict()
    with pytest.raises(TypeError, match=name):
        report.summary()
    with pytest.raises(TypeError, match=name):
        report.near_equilibrium_nodes()


def test_unavailable_verdicts_stay_distinct_from_negative_verdicts():
    report = _two_node_network().nodal_scan()
    state = report.nodes[0]
    state.equilibrium = state.active = None
    state.near_bifurcation = state.predicted_crossed = None

    data = report.to_dict()
    assert data["nodes"]["0"]["equilibrium"] is None
    assert data["nodes"]["0"]["active"] is None
    assert data["nodes"]["0"]["predicted_crossed"] is None
    assert data["nodes"]["1"]["equilibrium"] is False
    assert data["aggregate"]["equilibrium_count"] == 0
    assert data["aggregate"]["equilibrium_unavailable_count"] == 1
    assert data["aggregate"]["active_unavailable_count"] == 1
    assert data["aggregate"]["bifurcation_unavailable_count"] == 1
    assert report.near_equilibrium_nodes() == []
    assert "equilibrium unavailable" in state.summary()
    assert "equilibrium_unavailable_count=1" in report.summary()

    # Reports are mutable snapshots; a later read must not reuse stale counts.
    state.equilibrium = np.bool_(True)
    assert report.near_equilibrium_nodes() == [state]
    assert state.to_dict()["equilibrium"] is True
    assert report.to_dict()["aggregate"]["equilibrium_count"] == 1
    assert report.to_dict()["aggregate"]["equilibrium_unavailable_count"] == 0


def test_prediction_alias_contradiction_is_not_serialized_as_two_valid_verdicts():
    report = _two_node_network().nodal_scan()
    report.nodes[0].near_bifurcation = not report.nodes[0].predicted_crossed

    with pytest.raises(ValueError, match="must agree"):
        report.nodes[0].to_dict()
    with pytest.raises(ValueError, match="must agree"):
        report.summary()


def test_legacy_prediction_construction_rejects_text_instead_of_asserting_a_crossing():
    state = _two_node_network().nodal_state(0)
    with pytest.raises(TypeError, match="predicted_crossed"):
        replace(state, near_bifurcation="false", predicted_crossed=None)
