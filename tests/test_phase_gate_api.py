from __future__ import annotations

import math
from copy import deepcopy
from fractions import Fraction
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from tnfr.operators.grammar_dynamics import validate_sequence_incremental
from tnfr.operators.grammar_patterns import validate_sequence
from tnfr.operators.grammar_types import glyph_function_name
from tnfr.validation.phase_gate import (
    analyze_phase_gate,
    compare_against_global_baselines,
    compute_edge_gate_compliance,
    detect_phase_gate_violations,
    export_phase_gate_report,
    get_node_phase,
    node_phases,
    phase_histogram_entropy,
    prescribe_phase_gate_operators,
    rank_phase_stress_hotspots,
)


def _cycle_with_phases(phases: list[float]) -> nx.Graph:
    G = nx.cycle_graph(len(phases))
    for node, phase in enumerate(phases):
        G.nodes[node]["phase"] = phase
        G.nodes[node]["theta"] = phase
        G.nodes[node]["EPI"] = 1.0
        G.nodes[node]["delta_nfr"] = 0.05
        G.nodes[node]["dnfr"] = 0.05
        G.nodes[node]["glyph_history"] = []
    return G


def _assert_prescription_sequences_are_grammar_valid(
    G: nx.Graph, prescriptions
) -> None:
    fallback_node = next(iter(G.nodes()))
    for prescription in prescriptions:
        canonical = [
            glyph_function_name(code, default=code) for code in prescription.sequence
        ]
        full_result = validate_sequence(
            canonical,
            context={"initial_epi_nonzero": True},
        )
        assert full_result.passed, (prescription.sequence, full_result.message)

        node = prescription.target if prescription.scope == "node" else fallback_node
        incremental_results = validate_sequence_incremental(
            G,
            node,
            prescription.sequence,
        )
        assert all(result.allowed for result in incremental_results), (
            prescription.sequence,
            [
                (result.candidate, [(v.rule, v.severity) for v in result.violations])
                for result in incremental_results
            ],
        )


def test_phase_gate_api_distinguishes_local_scramble(tmp_path: Path):
    phases = [2.0 * math.pi * i / 24 for i in range(24)]
    smooth = _cycle_with_phases(phases)
    scrambled = _cycle_with_phases([phases[(i * 7) % 24] for i in range(24)])

    smooth_compliance = compute_edge_gate_compliance(smooth, gate=math.pi / 4.0)
    scrambled_compliance = compute_edge_gate_compliance(scrambled, gate=math.pi / 4.0)

    assert smooth_compliance.compliance_ratio == 1.0
    assert scrambled_compliance.compliance_ratio < 1.0
    assert len(detect_phase_gate_violations(scrambled, gate=math.pi / 4.0)) > 0

    smooth_features = compare_against_global_baselines(smooth, gate=math.pi / 4.0)
    scrambled_features = compare_against_global_baselines(scrambled, gate=math.pi / 4.0)

    assert (
        abs(smooth_features["global_order_r"] - scrambled_features["global_order_r"])
        < 1e-12
    )
    assert (
        scrambled_features["tnfr_mean_phase_gradient"]
        > smooth_features["tnfr_mean_phase_gradient"]
    )

    hotspots = rank_phase_stress_hotspots(scrambled, gate=math.pi / 4.0, top_n=3)
    assert len(hotspots) == 3
    assert hotspots[0].stress_score >= hotspots[-1].stress_score

    report = analyze_phase_gate(scrambled, gate=math.pi / 4.0, top_n=3)
    assert report.recommendation in {
        "couple_with_hotspot_monitoring",
        "hold_coupling_and_stabilize_hotspots",
    }
    assert report.operator_prescriptions
    assert report.operator_prescriptions[0].scope in {"network", "node"}
    assert any(
        "IL" in prescription.sequence for prescription in report.operator_prescriptions
    )
    _assert_prescription_sequences_are_grammar_valid(
        scrambled,
        report.operator_prescriptions,
    )

    smooth_prescriptions = prescribe_phase_gate_operators(smooth, gate=math.pi / 4.0)
    assert smooth_prescriptions[0].sequence == ("UM", "RA", "SHA")
    _assert_prescription_sequences_are_grammar_valid(smooth, smooth_prescriptions)

    json_path = export_phase_gate_report(scrambled, tmp_path / "phase_gate.json")
    md_path = export_phase_gate_report(scrambled, tmp_path / "phase_gate.md")
    html_path = export_phase_gate_report(scrambled, tmp_path / "phase_gate.html")
    assert json_path.exists()
    assert md_path.exists()
    assert html_path.exists()
    assert "operator_prescriptions" in json_path.read_text(encoding="utf-8")
    assert "TNFR canonical operator prescription" in md_path.read_text(encoding="utf-8")
    assert "TNFR canonical operator prescription" in html_path.read_text(
        encoding="utf-8"
    )


@pytest.mark.parametrize("edges", (False, True))
@pytest.mark.parametrize(
    "gate",
    (math.nan, math.inf, -1.0, True, np.bool_(False), "0.5", Fraction(1, 10**400)),
)
def test_gate_admission_precedes_empty_or_nonempty_graph_shortcuts(edges, gate):
    graph = nx.path_graph(2) if edges else nx.Graph()
    for evaluate in (compute_edge_gate_compliance, detect_phase_gate_violations):
        with pytest.raises((TypeError, ValueError)):
            evaluate(graph, gate=gate)


@pytest.mark.parametrize(
    "minimum", (math.nan, math.inf, -0.1, 1.1, True, "0.9", Fraction(1, 10**400))
)
def test_invalid_minimum_cannot_certify_empty_graph(minimum):
    graph = nx.Graph()
    result = compute_edge_gate_compliance(graph)
    with pytest.raises((TypeError, ValueError)):
        result.passed(minimum)
    with pytest.raises((TypeError, ValueError)):
        prescribe_phase_gate_operators(graph, min_compliance=minimum)
    with pytest.raises((TypeError, ValueError)):
        analyze_phase_gate(graph, min_compliance=minimum)


@pytest.mark.parametrize("phase", (None, True, math.nan, "0.0", Fraction(1, 10**400)))
def test_invalid_authoritative_phase_cannot_fall_back_to_compatible_alias(phase):
    graph = nx.path_graph(2)
    graph.nodes[0].update(theta=phase, phase=0.0)
    graph.nodes[1].update(theta=0.0, phase=0.0)
    with pytest.raises((TypeError, ValueError)):
        compute_edge_gate_compliance(graph)


def test_canonical_alias_order_missing_default_and_small_difference():
    graph = nx.path_graph(2)
    graph.nodes[0].update(theta=1e-18, phase=1.0)
    assert get_node_phase(graph, 1) == 0.0  # Declared missing-phase convention.
    assert node_phases(graph) == [1e-18, 0.0]
    result = compute_edge_gate_compliance(graph, gate=0.0)
    assert result.violation_count == 1
    assert result.violations[0].phase_difference == pytest.approx(1e-18, abs=0.0)


def test_overflowing_phase_difference_is_rejected():
    graph = nx.path_graph(2)
    graph.nodes[0]["theta"] = 1e308
    graph.nodes[1]["theta"] = -1e308
    with pytest.raises(ValueError, match="finite binary64"):
        compute_edge_gate_compliance(graph)


def test_explicit_phase_chart_is_shared_by_edges_and_field_readers():
    graph = _cycle_with_phases([0.0] * 4)
    for node in graph:
        graph.nodes[node]["measured"] = node / 10
    before = deepcopy(dict(graph.nodes(data=True)))
    selected = compare_against_global_baselines(graph, phase_keys=("measured",))
    canonical = compare_against_global_baselines(graph)
    assert selected["edge_diff_max"] == pytest.approx(0.3)
    assert selected["tnfr_mean_phase_gradient"] == pytest.approx(0.15)
    assert canonical["tnfr_mean_phase_gradient"] == 0.0
    assert dict(graph.nodes(data=True)) == before


@pytest.mark.parametrize("bins", (True, 0, -1, 2.5))
def test_histogram_policy_is_admitted_even_without_samples(bins):
    with pytest.raises((TypeError, ValueError)):
        phase_histogram_entropy([], bins=bins)


def test_report_export_rejects_invalid_gate_without_replacing_file(tmp_path):
    path = tmp_path / "retained.json"
    path.write_text("retained observation", encoding="utf-8")
    with pytest.raises(ValueError, match="finite"):
        export_phase_gate_report(nx.Graph(), path, gate=math.nan)
    assert path.read_text(encoding="utf-8") == "retained observation"
