"""Tetrad telemetry preserves complete samples and finite summary availability."""

import json
import math
from copy import deepcopy
from fractions import Fraction

import networkx as nx
import pytest

from tnfr.metrics import tetrad
from tnfr.sdk import Network, diagnose_network


def test_large_uniform_field_keeps_its_finite_mean_and_zero_spread():
    stats = tetrad._field_statistics({0: 1e308, 1: 1e308}, "low", False)
    assert stats["available"] is True
    assert stats["mean"] == stats["min"] == stats["max"] == 1e308
    assert stats["std"] == 0.0


def test_extreme_optional_statistics_do_not_discard_available_basic_statistics():
    stats = tetrad._field_statistics({0: -1e308, 1: 1e308}, "high", True)
    assert stats["mean"] == 0.0
    assert stats["std"] == 1e308
    assert stats["available"] is True
    assert stats["p50"] is None
    assert "p50" in stats["unavailable_statistics"]
    assert stats["histogram"] is None
    assert "histogram" in stats["unavailable_statistics"]
    json.dumps(stats, allow_nan=False)


@pytest.mark.parametrize("invalid", [True, "2.0", None, math.nan, Fraction(1, 10**400)])
def test_one_invalid_node_makes_the_whole_field_summary_unavailable(invalid):
    stats = tetrad._field_statistics({0: 1.0, 1: invalid}, "high", True)
    assert stats["available"] is False
    assert stats["mean"] is stats["min"] is stats["max"] is stats["std"] is None
    assert "no nodes were filtered" in stats["error"]["message"]
    assert "histogram" not in stats


def test_empty_field_is_unavailable_without_a_zero_mean():
    stats = tetrad._field_statistics({}, "low", False)
    assert stats["available"] is False
    assert stats["mean"] is None
    assert stats["error"]["message"] == "Empty field"


def test_ordinary_percentile_and_histogram_contract_is_preserved():
    stats = tetrad._field_statistics(
        dict(enumerate((1.0, 2.0, 3.0, 4.0, 5.0))), "high", True
    )
    assert stats["mean"] == 3.0
    assert stats["std"] == pytest.approx(math.sqrt(2.0))
    assert (stats["p25"], stats["p50"], stats["p75"]) == (2.0, 3.0, 4.0)
    assert (stats["p10"], stats["p90"], stats["p99"]) == pytest.approx((1.4, 4.6, 4.96))
    histogram = stats["histogram"]
    assert len(histogram["counts"]) == 20
    assert len(histogram["edges"]) == 21
    assert sum(histogram["counts"]) == 5
    assert (histogram["edges"][0], histogram["edges"][-1]) == (1.0, 5.0)
    assert "unavailable_statistics" not in stats


@pytest.mark.parametrize("nodes", [1, 2])
def test_tetrad_retains_unavailable_or_spectral_coherence_length_provenance(nodes):
    snapshot = tetrad.collect_tetrad_snapshot(
        nx.path_graph(nodes), include_histograms=False
    )
    if nodes == 1:
        assert snapshot["xi_c"] is None
        assert snapshot["xi_c_available"] is False
        assert snapshot["xi_c_provenance"]["method"] == "unavailable"
        assert snapshot["xi_c_error"] is not None
    else:
        assert snapshot["xi_c"] == pytest.approx(1 / math.sqrt(2))
        assert snapshot["xi_c_available"] is True
        assert snapshot["xi_c_provenance"]["method"] == "spectral_gap"
        assert "dimensionless" in snapshot["xi_c_provenance"]["distance_weighting"]
        assert snapshot["xi_c_error"] is None
    json.dumps(snapshot, allow_nan=False)


def test_estimator_failure_keeps_its_reason_without_discarding_local_fields(
    monkeypatch,
):
    def refuse(graph):
        raise ValueError("unsupported estimator support")

    monkeypatch.setattr(tetrad, "estimate_coherence_length_with_provenance", refuse)
    snapshot = tetrad.collect_tetrad_snapshot(
        nx.path_graph(2), include_histograms=False
    )
    assert snapshot["phi_s"]["available"] is True
    assert snapshot["xi_c_available"] is False
    assert snapshot["xi_c_provenance"] is None
    assert snapshot["xi_c_error"] == {
        "type": "ValueError",
        "message": "unsupported estimator support",
    }


def _cancelling_phase_graph():
    graph = nx.star_graph(4)
    for node, phase in zip(graph, (0.3, 0.0, 0.0, math.pi, -math.pi)):
        graph.nodes[node].update(EPI=1.0, nu_f=1.0, phase=phase, delta_nfr=0.0)
    return graph


def test_undefined_curvature_keeps_other_fields_and_sdk_evidence():
    graph = _cancelling_phase_graph()
    before_nodes = deepcopy(dict(graph.nodes(data=True)))
    before_edges = deepcopy(list(graph.edges(data=True)))
    snapshot = tetrad.collect_tetrad_snapshot(graph, include_histograms=False)
    diagnostic = diagnose_network(Network(graph))["tetrad"]
    assert snapshot["phi_s"]["available"] is True
    assert snapshot["phase_grad"]["available"] is True
    assert snapshot["xi_c_available"] is True
    curvature = snapshot["phase_curv"]
    assert curvature["available"] is False
    assert curvature["complete"] is False
    assert curvature["mean"] is curvature["std"] is None
    assert curvature["node_values"] == diagnostic["k_phi"]["value"]
    assert curvature["node_status"] == diagnostic["k_phi"]["node_status"]
    assert curvature["node_values"][0] is None
    assert all(value is not None for value in curvature["node_values"][1:])
    assert curvature["error"]["type"] == "UndefinedPhaseCurvatureError"
    assert snapshot["phase_grad"]["mean"] == pytest.approx(
        math.fsum(diagnostic["grad_phi"]["value"]) / graph.number_of_nodes()
    )
    assert dict(graph.nodes(data=True)) == before_nodes
    assert list(graph.edges(data=True)) == before_edges
    json.dumps(snapshot, allow_nan=False)


def test_tetrad_snapshot_reads_each_independent_owner_once(monkeypatch):
    calls = []
    for name in (
        "observe_phase_curvature",
        "compute_structural_potential",
        "estimate_coherence_length_with_provenance",
    ):
        original = getattr(tetrad, name)

        def counted(graph, original=original, name=name):
            calls.append(name)
            return original(graph)

        monkeypatch.setattr(tetrad, name, counted)
    tetrad.collect_tetrad_snapshot(_cancelling_phase_graph())
    assert len(calls) == len(set(calls)) == 3


def test_phase_evidence_is_detached_and_relabeling_preserves_ordered_status():
    original = _cancelling_phase_graph()
    labels = (object(), "center", 5, (1, 2), object())
    relabeled = nx.relabel_nodes(original, dict(zip(original, labels)), copy=True)
    first = tetrad.collect_tetrad_snapshot(original)
    second = tetrad.collect_tetrad_snapshot(relabeled)
    for name in first:
        if name != "phase_curv":
            assert first[name] == second[name]
    for name in first["phase_curv"]:
        if name != "error":
            assert first["phase_curv"][name] == second["phase_curv"][name]
    assert first["phase_curv"]["error"]["type"] == second["phase_curv"]["error"]["type"]
    retained = deepcopy(second)
    second["phase_curv"]["node_values"][1] = 99.0
    second["phase_curv"]["node_status"][0] = "forged"
    assert tetrad.collect_tetrad_snapshot(relabeled) == retained


@pytest.mark.parametrize("invalid", [True, "0", math.nan, Fraction(1, 10**400)])
def test_invalid_phase_preserves_independent_pressure_and_xi(invalid):
    graph = _cancelling_phase_graph()
    graph.nodes[0]["phase"] = invalid
    snapshot = tetrad.collect_tetrad_snapshot(graph)
    assert snapshot["phi_s"]["available"] is True
    assert snapshot["xi_c_available"] is True
    for name in ("phase_grad", "phase_curv"):
        assert snapshot[name]["available"] is False
        assert snapshot[name]["mean"] is None
        assert snapshot[name]["error"] is not None
    assert snapshot["phase_curv"]["node_values"] is None
    json.dumps(snapshot, allow_nan=False)


def test_pressure_failure_preserves_independent_phase_fields():
    graph = nx.path_graph(3)
    nx.set_node_attributes(graph, 0.2, "phase")
    graph.nodes[0]["delta_nfr"] = math.nan
    snapshot = tetrad.collect_tetrad_snapshot(graph)
    assert snapshot["phi_s"]["available"] is False
    assert snapshot["phi_s"]["mean"] is None
    assert snapshot["phi_s"]["error"] is not None
    assert snapshot["phase_grad"]["available"] is True
    assert snapshot["phase_curv"]["available"] is True
    assert snapshot["phase_grad"]["mean"] == 0.0
    json.dumps(snapshot, allow_nan=False)
