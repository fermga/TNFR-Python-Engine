"""Tests for the Paley partition planner."""

from __future__ import annotations

import networkx as nx
import pytest

from tnfr_factorization.partitioning import (  # type: ignore[import]
    PartitionPlannerConfig,
    aggregate_partition_metrics,
    annotate_partition_candidates,
    plan_paley_partitions,
)


def test_plan_paley_partitions_produces_multiple_chunks() -> None:
    graph = nx.cycle_graph(40)
    config = PartitionPlannerConfig(target_size=10, boundary_overlap=2, notes="test")
    result = plan_paley_partitions(
        graph,
        41,
        phi_s=0.5,
        phase_gradient=0.1,
        phase_curvature=0.2,
        coherence_length=2.0,
        config=config,
    )
    assert len(result.partitions) >= 4
    first_partition = result.partitions[0]
    assert first_partition.metadata["size"] <= config.target_size
    assert "range" in first_partition.metadata
    # Ensure boundary nodes are identified for overlapping edges
    assert any(first_partition.boundary_nodes)


def test_partition_aggregation_matches_parent_metrics() -> None:
    graph = nx.path_graph(20)
    result = plan_paley_partitions(
        graph,
        23,
        phi_s=0.4,
        phase_gradient=0.2,
        phase_curvature=0.3,
        coherence_length=1.5,
    )
    aggregation = aggregate_partition_metrics(
        result,
        parent_phi_s=0.4,
        parent_phase_gradient=0.2,
        parent_phase_curvature=0.3,
        parent_coherence_length=1.5,
        total_candidate_count=0,
    )
    assert aggregation.partition_count == len(result.partitions)
    assert aggregation.node_total == sum(len(p.node_indices) for p in result.partitions)
    assert aggregation.phi_s_ratio == pytest.approx(1.0, abs=0.5)


def test_partition_candidate_annotation_and_aggregation() -> None:
    graph = nx.path_graph(30)
    partitions = plan_paley_partitions(
        graph,
        31,
        phi_s=0.5,
        phase_gradient=0.1,
        phase_curvature=0.2,
        coherence_length=2.0,
    )
    annotate_partition_candidates(partitions, [5, 7, 11, 13])
    aggregation = aggregate_partition_metrics(
        partitions,
        parent_phi_s=0.5,
        parent_phase_gradient=0.1,
        parent_phase_curvature=0.2,
        parent_coherence_length=2.0,
        total_candidate_count=4,
    )
    assert aggregation.candidate_total == 4
    assert aggregation.candidate_ratio == pytest.approx(1.0)
    non_empty = [pid for pid, cands in aggregation.partition_candidates.items() if cands]
    assert non_empty, "Expected at least one partition to receive candidates"
