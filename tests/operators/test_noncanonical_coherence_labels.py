"""Semantic boundaries for noncanonical operator diagnostic scores."""

from __future__ import annotations

from dataclasses import fields

import networkx as nx
import pytest

from tnfr.errors import TNFRValueError
from tnfr.mathematics.epi import BEPIElement
from tnfr.operators.cascade import detect_cascade
from tnfr.operators.cycle_detection import CycleAnalysis, CycleDetector
from tnfr.operators.health_analyzer import (
    SequenceHealthAnalyzer,
    SequenceHealthMetrics,
)
from tnfr.operators.metrics_basic import reception_metrics
from tnfr.operators.network_analysis.source_detection import (
    detect_emission_sources,
)


def test_source_detection_separates_bounded_phase_score_from_activity() -> None:
    graph = nx.Graph()
    graph.add_node(0, EPI=0.1, nu_f=1.0, theta=0.0)
    graph.add_node(1, EPI=2.0, nu_f=3.0, theta=0.25)
    graph.add_edge(0, 1)

    [(source, phase_score, emission_activity)] = detect_emission_sources(
        graph, 0
    )

    assert source == 1
    assert 0.0 <= phase_score <= 1.0
    assert emission_activity == pytest.approx(6.0)
    assert emission_activity > 1.0


def test_source_activity_accepts_only_the_uniform_real_bepi_embedding() -> None:
    graph = nx.path_graph(2)
    graph.nodes[0].update(EPI=0.1, nu_f=1.0, theta=0.0)
    graph.nodes[1].update(
        EPI=BEPIElement((2.0, 2.0), (2.0, 2.0), (0.0, 1.0)),
        nu_f=3.0,
        theta=0.0,
    )

    assert detect_emission_sources(graph, 0)[0][2] == pytest.approx(6.0)

    graph.nodes[1]["EPI"] = BEPIElement(
        (2.0, 1.0), (2.0, 2.0), (0.0, 1.0)
    )
    with pytest.raises(TNFRValueError, match="uniform-real BEPI"):
        detect_emission_sources(graph, 0)


@pytest.mark.parametrize("invalid_frequency", [-1.0, float("inf"), True])
def test_source_activity_requires_canonical_finite_nonnegative_capacity(
    invalid_frequency,
) -> None:
    graph = nx.path_graph(2)
    graph.nodes[0].update(EPI=0.1, nu_f=1.0, theta=0.0)
    graph.nodes[1].update(
        EPI=1.0,
        nu_f=invalid_frequency,
        theta=0.0,
    )

    with pytest.raises(TNFRValueError):
        detect_emission_sources(graph, 0)


def test_reception_metrics_expose_activity_and_bounded_phase_names() -> None:
    graph = nx.Graph()
    graph.add_node(
        0,
        EPI=1.5,
        DeltaNFR=0.2,
        _reception_sources=[(1, 0.75, 6.0)],
    )
    graph.add_node(1, EPI=2.0, DeltaNFR=0.1)
    graph.add_edge(0, 1)

    metrics = reception_metrics(graph, 0, epi_before=0.5)

    assert metrics["total_source_emission_activity"] == pytest.approx(6.0)
    assert metrics["epi_delta_per_source_activity"] == pytest.approx(1.0 / 6.0)
    assert metrics["mean_phase_compatibility_score"] == pytest.approx(0.75)
    assert metrics["integration_efficiency"] == (
        metrics["epi_delta_per_source_activity"]
    )
    assert metrics["phase_compatibility_avg"] == (
        metrics["mean_phase_compatibility_score"]
    )
    assert metrics["coherence_received"] == metrics["delta_epi"]
    assert metrics["canonical_coherence_certified"] is False


def _cascade_graph() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph(
        thol_propagations=[
            {
                "source_node": 0,
                "propagations": [(1, 0.2), (2, 0.1)],
            }
        ]
    )
    graph.add_nodes_from((0, 1, 2, 99))
    graph.add_edge(0, 1, key="a", weight=-2.0)
    graph.add_edge(0, 1, key="b", weight=4.0)
    graph.add_edge(1, 0, key="c", weight=8.0)
    graph.add_edge(2, 99, key="outside", weight=1000.0)
    return graph


def test_cascade_reports_unbounded_directed_multiedge_magnitude() -> None:
    graph = _cascade_graph()

    analysis = detect_cascade(graph)

    assert analysis["mean_internal_edge_weight_magnitude"] == pytest.approx(
        14.0 / 3.0
    )
    assert analysis["mean_internal_edge_weight_magnitude"] > 1.0
    assert analysis["cascade_coherence"] == (
        analysis["mean_internal_edge_weight_magnitude"]
    )
    assert analysis["canonical_coherence_certified"] is False


def test_cascade_cache_fingerprint_includes_edge_weights() -> None:
    graph = _cascade_graph()
    before = detect_cascade(graph)["mean_internal_edge_weight_magnitude"]

    graph.edges[0, 1, "a"]["weight"] = -20.0
    after = detect_cascade(graph)["mean_internal_edge_weight_magnitude"]

    assert after != before
    assert after == pytest.approx(32.0 / 3.0)


def test_cascade_rejects_overflowing_unbounded_magnitude_sum() -> None:
    graph = _cascade_graph()
    for left, right, key in graph.edges(keys=True):
        if left in {0, 1, 2} and right in {0, 1, 2}:
            graph.edges[left, right, key]["weight"] = 1e308

    with pytest.raises(TNFRValueError, match="must remain finite"):
        detect_cascade(graph)


def test_cascade_rejects_invalid_edge_weight_instead_of_returning_zero() -> None:
    graph = _cascade_graph()
    graph.edges[0, 1, "a"]["weight"] = "invalid"

    with pytest.raises(TNFRValueError, match="finite real scalar"):
        detect_cascade(graph)


def test_sequence_health_stores_flow_quality_and_keeps_read_alias() -> None:
    metrics = SequenceHealthAnalyzer().analyze_health(
        ["emission", "reception", "coherence", "silence"]
    )

    assert "flow_quality_score" in {
        field.name for field in fields(SequenceHealthMetrics)
    }
    assert "coherence_index" not in {
        field.name for field in fields(SequenceHealthMetrics)
    }
    assert 0.0 <= metrics.flow_quality_score <= 1.0
    assert metrics.coherence_index == metrics.flow_quality_score

    metrics.coherence_index = 0.25
    assert metrics.flow_quality_score == pytest.approx(0.25)
    with pytest.raises(TNFRValueError, match=r"\[0, 1\]"):
        metrics.coherence_index = 1.1


def test_cycle_analysis_stores_integrity_score_and_keeps_read_alias() -> None:
    result = CycleDetector().analyze_potential_cycle(
        ["emission", "coherence", "transition", "resonance", "silence"],
        2,
    )

    assert "cycle_integrity_score" in {
        field.name for field in fields(CycleAnalysis)
    }
    assert "coherence_score" not in {
        field.name for field in fields(CycleAnalysis)
    }
    assert 0.0 <= result.cycle_integrity_score <= 1.0
    assert result.coherence_score == result.cycle_integrity_score

    result.coherence_score = 0.4
    assert result.cycle_integrity_score == pytest.approx(0.4)
    with pytest.raises(TNFRValueError, match=r"\[0, 1\]"):
        result.coherence_score = -0.1


def test_health_transition_penalty_is_centralized() -> None:
    analyzer = SequenceHealthAnalyzer()
    sequence = ["dissonance", "mutation", "silence"]
    problematic = [("dissonance", "mutation")]

    expected = analyzer._transition_quality_score(3, 1)

    assert expected == pytest.approx(0.75)
    assert analyzer._calculate_smoothness(sequence, problematic) == expected
