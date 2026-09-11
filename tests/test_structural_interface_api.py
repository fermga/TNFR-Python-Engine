"""Tests for the TNFR Structural Interface orchestration API.

Covers graph construction, phase encoding, TNFR scoring, classical baselines,
evaluation metrics, report export, and — crucially — that every generated
operator prescription is grammar-valid under both the whole-sequence and the
incremental validators (Milestone 1 acceptance criterion).
"""

from __future__ import annotations

import json
import math

import networkx as nx
import pytest

from tnfr.operators.grammar_dynamics import validate_sequence_incremental
from tnfr.operators.grammar_patterns import validate_sequence
from tnfr.operators.grammar_types import glyph_function_name
from tnfr.validation.structural_interface import (
    StructuralInterfaceProblem,
    StructuralInterfaceScore,
    baseline_score_maps,
    build_knn_graph,
    encode_phase_from_binary_state,
    evaluate_interface_scores,
    export_structural_interface_report,
    interface_score_maps,
    local_state_disagreement,
    render_structural_interface_html,
    render_structural_interface_markdown,
    score_structural_interfaces,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _two_cluster_records() -> list[dict[str, object]]:
    """Two well-separated 2D clusters with binary state labels."""
    records: list[dict[str, object]] = []
    for i in range(6):
        records.append({"f1": 0.0 + 0.05 * i, "f2": 0.0, "state": "low"})
    for i in range(6):
        records.append({"f1": 5.0 + 0.05 * i, "f2": 5.0, "state": "high"})
    return records


def _assert_prescription_grammar_valid(G, node, sequence) -> None:
    """Assert a glyph-code sequence passes both grammar validators.

    Prescriptions target an already-active graph state, so the whole-sequence
    validator is given ``initial_epi_nonzero=True``.  The incremental validator
    reads live node state (EPI defaults to initialized, history empty).
    """
    function_names = [glyph_function_name(code) for code in sequence]
    whole = validate_sequence(function_names, context={"initial_epi_nonzero": True})
    assert whole.passed, (sequence, getattr(whole, "message", ""))

    steps = validate_sequence_incremental(G, node, list(sequence))
    assert all(step.allowed for step in steps), [
        (step.candidate, [v.message for v in step.violations]) for step in steps
    ]


# ---------------------------------------------------------------------------
# Graph construction & encoding
# ---------------------------------------------------------------------------


def test_build_knn_graph_basic() -> None:
    records = _two_cluster_records()
    G = build_knn_graph(
        records,
        ["f1", "f2"],
        k=3,
        node_attributes=["state"],
    )
    assert G.number_of_nodes() == len(records)
    assert G.number_of_edges() > 0
    # Every edge carries a non-negative feature distance.
    for _, _, data in G.edges(data=True):
        assert "distance" in data
        assert data["distance"] >= 0.0
    # State attribute copied onto nodes.
    assert G.nodes[0]["state"] == "low"
    assert G.nodes[len(records) - 1]["state"] == "high"


def test_build_knn_graph_empty() -> None:
    G = build_knn_graph([], ["f1"], k=3)
    assert G.number_of_nodes() == 0
    assert G.number_of_edges() == 0


def test_build_knn_graph_rejects_invalid_k() -> None:
    with pytest.raises(ValueError):
        build_knn_graph(_two_cluster_records(), ["f1", "f2"], k=0)


def test_encode_phase_from_binary_state() -> None:
    G = build_knn_graph(
        _two_cluster_records(), ["f1", "f2"], k=3, node_attributes=["state"]
    )
    encode_phase_from_binary_state(G, "state", positive_value="high")
    for node in G.nodes():
        phase = G.nodes[node]["phase"]
        if G.nodes[node]["state"] == "high":
            assert phase == pytest.approx(0.0)
        else:
            assert phase == pytest.approx(math.pi)
        # theta mirror is set for TNFR phase aliasing.
        assert G.nodes[node]["theta"] == pytest.approx(phase)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def test_local_state_disagreement_uniform_is_zero() -> None:
    G = nx.path_graph(5)
    for node in G.nodes():
        G.nodes[node]["state"] = "same"
    disagreement = local_state_disagreement(G, "state")
    assert set(disagreement.values()) == {0}


def test_local_state_disagreement_counts_conflicts() -> None:
    G = nx.path_graph(3)
    G.nodes[0]["state"] = "a"
    G.nodes[1]["state"] = "b"
    G.nodes[2]["state"] = "a"
    disagreement = local_state_disagreement(G, "state")
    # Middle node disagrees with both neighbours.
    assert disagreement[1] == 2
    assert disagreement[0] == 1
    assert disagreement[2] == 1


def test_baseline_score_maps_keys() -> None:
    G = build_knn_graph(
        _two_cluster_records(), ["f1", "f2"], k=3, node_attributes=["state"]
    )
    maps = baseline_score_maps(G, state_key="state")
    assert set(maps) == {
        "local_state_disagreement",
        "mean_neighbour_distance",
        "topology_degree",
        "constant_baseline",
    }
    for score_map in maps.values():
        assert set(score_map) == set(G.nodes())
        assert all(math.isfinite(v) for v in score_map.values())


# ---------------------------------------------------------------------------
# TNFR scoring
# ---------------------------------------------------------------------------


def test_score_structural_interfaces_finite_and_complete() -> None:
    G = build_knn_graph(
        _two_cluster_records(), ["f1", "f2"], k=3, node_attributes=["state"]
    )
    encode_phase_from_binary_state(G, "state", positive_value="high")
    scores = score_structural_interfaces(G)
    assert len(scores) == G.number_of_nodes()
    for score in scores:
        assert isinstance(score, StructuralInterfaceScore)
        assert math.isfinite(score.tnfr_stress)
        assert math.isfinite(score.phase_gradient)
        assert math.isfinite(score.abs_curvature)
        assert math.isfinite(score.structural_potential)
        assert score.incident_violation_count >= 0


def test_score_structural_interfaces_problem_object() -> None:
    G = build_knn_graph(
        _two_cluster_records(), ["f1", "f2"], k=3, node_attributes=["state"]
    )
    encode_phase_from_binary_state(G, "state", positive_value="high")
    problem = StructuralInterfaceProblem(graph=G, state_key="state", domain="unit-test")
    scores = score_structural_interfaces(problem)
    assert len(scores) == G.number_of_nodes()


def test_uniform_state_has_low_stress() -> None:
    """Well-separated clusters with internally uniform state have no interfaces."""
    G = build_knn_graph(
        _two_cluster_records(), ["f1", "f2"], k=2, node_attributes=["state"]
    )
    encode_phase_from_binary_state(G, "state", positive_value="high")
    scores = score_structural_interfaces(G)
    # With separated clusters, intra-cluster edges dominate -> no gate violations.
    assert all(score.incident_violation_count == 0 for score in scores)


# ---------------------------------------------------------------------------
# Grammar validity of prescriptions (Milestone 1 acceptance)
# ---------------------------------------------------------------------------


def test_all_prescriptions_grammar_valid_separated() -> None:
    G = build_knn_graph(
        _two_cluster_records(), ["f1", "f2"], k=3, node_attributes=["state"]
    )
    encode_phase_from_binary_state(G, "state", positive_value="high")
    scores = score_structural_interfaces(G)
    for score in scores:
        _assert_prescription_grammar_valid(G, score.node, score.prescription)


def test_all_prescriptions_grammar_valid_mixed() -> None:
    """Force phase conflicts to exercise the failed/hotspot prescription paths."""
    G = nx.Graph()
    # Star with a phase-conflicting hub forces gate violations.
    G.add_node(0, state="a", phase=0.0, theta=0.0)
    for leaf in range(1, 6):
        G.add_node(leaf, state="b", phase=math.pi, theta=math.pi)
        G.add_edge(0, leaf, distance=1.0)
    scores = score_structural_interfaces(G)
    assert any(score.incident_violation_count > 0 for score in scores)
    for score in scores:
        _assert_prescription_grammar_valid(G, score.node, score.prescription)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def test_evaluate_interface_scores_metrics() -> None:
    G = build_knn_graph(
        _two_cluster_records(), ["f1", "f2"], k=3, node_attributes=["state"]
    )
    encode_phase_from_binary_state(G, "state", positive_value="high")
    scores = score_structural_interfaces(G)
    tnfr_map = interface_score_maps(scores)
    baselines = baseline_score_maps(G, state_key="state")

    # NOTE: this target equals local disagreement, so it is a localization
    # sanity check (circular), not an external superiority claim.
    disagreement = local_state_disagreement(G, "state")
    labels = {node: disagreement[node] > 0 for node in G.nodes()}

    evaluation = evaluate_interface_scores(
        labels, {"tnfr_stress": tnfr_map, **baselines}
    )
    assert evaluation["total_nodes"] == G.number_of_nodes()
    names = {row["score"] for row in evaluation["score_comparison"]}
    assert "tnfr_stress" in names
    for row in evaluation["score_comparison"]:
        assert 0.0 <= row["auc"] <= 1.0
        assert 0.0 <= row["precision_at_review_count"] <= 1.0


def test_evaluate_handles_degenerate_labels() -> None:
    G = nx.path_graph(4)
    scores = {node: float(node) for node in G.nodes()}
    labels = {node: False for node in G.nodes()}  # no positives
    evaluation = evaluate_interface_scores(labels, {"score": scores})
    assert evaluation["review_node_count"] == 0
    row = evaluation["score_comparison"][0]
    assert row["auc"] == pytest.approx(0.5)
    assert row["precision_at_review_count"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Reporting / export
# ---------------------------------------------------------------------------


def _build_result_payload(G) -> dict[str, object]:
    scores = score_structural_interfaces(G)
    tnfr_map = interface_score_maps(scores)
    baselines = baseline_score_maps(G, state_key="state")
    disagreement = local_state_disagreement(G, "state")
    labels = {node: disagreement[node] > 0 for node in G.nodes()}
    evaluation = evaluate_interface_scores(
        labels, {"tnfr_stress": tnfr_map, **baselines}
    )
    top = sorted(scores, key=lambda s: s.tnfr_stress, reverse=True)[:5]
    return {
        "dataset": {
            "name": "unit-test",
            "sector": "generic",
            "samples": G.number_of_nodes(),
        },
        "graph": {
            "construction": "k-NN graph",
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
        },
        "task": {
            "target_definition": "local_state_disagreement > 0",
            "is_circular_target": True,
        },
        "evaluation": evaluation,
        "hotspots": [s.as_dict() for s in top],
        "honest_interpretation": "Circular target; localization sanity check only.",
    }


def test_render_markdown_and_html() -> None:
    G = build_knn_graph(
        _two_cluster_records(), ["f1", "f2"], k=3, node_attributes=["state"]
    )
    encode_phase_from_binary_state(G, "state", positive_value="high")
    result = _build_result_payload(G)
    markdown = render_structural_interface_markdown(result)
    assert "# TNFR Structural Interface Audit" in markdown
    assert "Score comparison" in markdown
    html = render_structural_interface_html(markdown)
    assert "<html" in html
    assert "<table>" in html


def test_export_structural_interface_report(tmp_path) -> None:
    G = build_knn_graph(
        _two_cluster_records(), ["f1", "f2"], k=3, node_attributes=["state"]
    )
    encode_phase_from_binary_state(G, "state", positive_value="high")
    result = _build_result_payload(G)
    paths = export_structural_interface_report(result, tmp_path, stem="demo")
    assert paths["json"].exists()
    assert paths["markdown"].exists()
    assert paths["html"].exists()
    # JSON round-trips.
    loaded = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert loaded["graph"]["nodes"] == G.number_of_nodes()
