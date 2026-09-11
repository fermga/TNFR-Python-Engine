"""High-level Mutation experiments abstain without fabricating evidence."""

from copy import deepcopy

import pytest

from tnfr.errors import TNFRValueError
from tnfr.sdk.fluent import TNFRNetwork


def _coherent_ring(*, nodes: int = 4, epi: float = 0.2) -> TNFRNetwork:
    return (
        TNFRNetwork()
        .add_nodes(
            nodes,
            epi_range=(epi, epi),
            vf_range=(0.6, 0.6),
            phase_range=(0.0, 0.0),
        )
        .connect_nodes(connection_pattern="ring")
    )


def test_direct_creative_mutation_remains_strict_without_evidence() -> None:
    network = _coherent_ring()
    before = deepcopy(dict(network.graph.nodes(data=True)))

    with pytest.raises(TNFRValueError, match="Mutation sequence preflight failed"):
        network.apply_sequence("creative_mutation")

    assert dict(network.graph.nodes(data=True)) == before
    assert "mutation_workflows" not in network.graph.graph


def test_high_level_workflow_abstains_into_exploration_without_history() -> None:
    network = _coherent_ring()

    results = network.apply_evidence_gated_mutation(repeat=2).measure()

    decision = results.mutation_workflows[0]
    assert decision["status"] == "mutation_abstained"
    assert decision["reason"] == "missing_history"
    assert decision["requested_sequence"] == "creative_mutation"
    assert decision["executed_sequence"] == "exploration"
    assert decision["cycles"] == 2
    assert all(report["source"] is None for report in decision["observations"])
    assert all(report["time_basis"] is None for report in decision["observations"])
    assert all("ZHIR" not in data["glyph_history"] for _, data in network.graph.nodes(data=True))
    assert all(
        key not in data
        for _, data in network.graph.nodes(data=True)
        for key in ("epi_time_history", "epi_history", "_epi_history")
    )
    assert results.to_dict()["mutation_workflows"] == results.mutation_workflows
    assert "0 applied, 1 abstained" in results.summary()


def test_legacy_evidence_is_labeled_when_it_explicitly_allows_mutation() -> None:
    network = _coherent_ring()
    for _, data in network.graph.nodes(data=True):
        data["epi_history"] = [0.0, 0.2]

    decision = (
        network.apply_evidence_gated_mutation().measure().mutation_workflows[0]
    )

    assert decision["status"] == "mutation_applied"
    assert decision["reason"] is None
    assert all(
        report["time_basis"] == "legacy_unit_operator_step"
        for report in decision["observations"]
    )
    assert all(
        report["physical_time_resolved"] is False
        for report in decision["observations"]
    )
    assert all("ZHIR" in data["glyph_history"] for _, data in network.graph.nodes(data=True))


def test_physical_crossing_abstains_when_prefix_would_stale_it() -> None:
    network = _coherent_ring()
    for _, data in network.graph.nodes(data=True):
        data["epi_time_history"] = [(0.0, 0.0), (1.0, 0.2)]

    decision = (
        network.apply_evidence_gated_mutation().measure().mutation_workflows[0]
    )

    assert decision["status"] == "mutation_abstained"
    assert decision["reason"] == "physical_mutation_evidence_would_be_stale"
    assert all(
        report["gate_satisfied"]
        and report["time_basis"] == "physical_time"
        and report["physical_time_resolved"]
        for report in decision["observations"]
    )
    assert all("ZHIR" not in data["glyph_history"] for _, data in network.graph.nodes(data=True))


def test_malformed_physical_history_is_an_error_not_an_abstention() -> None:
    network = _coherent_ring()
    for _, data in network.graph.nodes(data=True):
        data["epi_time_history"] = [(0.0, 0.0), (0.0, 0.2)]
    before_nodes = deepcopy(dict(network.graph.nodes(data=True)))
    before_graph = deepcopy(dict(network.graph.graph))

    with pytest.raises(TNFRValueError, match="Invalid evidence-gated Mutation"):
        network.apply_evidence_gated_mutation()

    assert dict(network.graph.nodes(data=True)) == before_nodes
    assert dict(network.graph.graph) == before_graph


def test_invalid_workflow_log_rejects_before_operator_execution() -> None:
    network = _coherent_ring()
    network.graph.graph["mutation_workflows"] = "not-a-list"
    before = deepcopy(dict(network.graph.nodes(data=True)))

    with pytest.raises(ValueError, match="mutation_workflows must be a list"):
        network.apply_evidence_gated_mutation()

    assert dict(network.graph.nodes(data=True)) == before


def test_adaptive_innovation_uses_the_evidence_gated_protocol() -> None:
    network = _coherent_ring(nodes=4)

    network.apply_adaptive_sequence()

    decision = network.graph.graph["mutation_workflows"][0]
    assert decision["requested_sequence"] == "innovation"
    assert decision["executed_sequence"] == "exploration"
    assert decision["status"] == "mutation_abstained"
