from __future__ import annotations

import math
from pathlib import Path

import networkx as nx

from tnfr.operators.grammar_dynamics import validate_sequence_incremental
from tnfr.operators.grammar_patterns import validate_sequence
from tnfr.operators.grammar_types import glyph_function_name
from tnfr.validation.phase_gate import (
    analyze_phase_gate,
    compare_against_global_baselines,
    compute_edge_gate_compliance,
    detect_phase_gate_violations,
    export_phase_gate_report,
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
