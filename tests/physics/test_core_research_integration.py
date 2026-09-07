"""Executable shared-hypothesis checks for the restricted S16 chain."""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest

from tnfr.physics import (
    CoreResearchIntegrationCertificate,
    EpiCoarseGrainingCertificate,
    EpiDiffusionReconstructionCertificate,
    HeterogeneousDiffusionStabilityCertificate,
    StructuralChannelScales,
    StructuralStateDistanceCertificate,
    certify_core_research_integration,
    structural_diffusion_operator,
)


SCALES = StructuralChannelScales(
    epi=1.0,
    frequency=1.0,
    phase=math.pi,
    pressure=1.0,
    epi_rate=1.0,
    edge_conductance=1.0,
    edge_length=1.0,
)
EXACT_PARTITION = ((0, 3), (1, 2))


def _state(epi, frequency, phase) -> nx.Graph:
    graph = nx.path_graph(4)
    graph.edges[0, 1]["weight"] = 2.0
    graph.edges[1, 2]["weight"] = 0.75
    graph.edges[2, 3]["weight"] = 2.0
    for node, value, rate, angle in zip(graph, epi, frequency, phase):
        graph.nodes[node].update(
            EPI=float(value),
            nu_f=float(rate),
            theta=float(angle),
        )

    nodes, laplacian = structural_diffusion_operator(graph)
    field = np.asarray([graph.nodes[node]["EPI"] for node in nodes], dtype=float)
    pressure = -(laplacian @ field)
    for node, value in zip(nodes, pressure):
        graph.nodes[node]["delta_nfr"] = float(value)
        graph.nodes[node]["dEPI_dt"] = float(graph.nodes[node]["nu_f"] * value)
    return graph


def _states() -> tuple[nx.Graph, nx.Graph]:
    left = _state(
        [2.0, -1.0, 3.0, 0.5],
        [0.5, 1.5, 1.5, 0.5],
        [0.0, 0.2, 0.4, 0.6],
    )
    right = _state(
        [1.5, -0.5, 2.0, 0.25],
        [0.8, 1.2, 1.2, 0.8],
        [0.1, 0.3, 0.5, 0.7],
    )
    return left, right


def test_connected_heterogeneous_states_pass_the_restricted_s16_chain():
    left, right = _states()

    result = certify_core_research_integration(
        left, right, EXACT_PARTITION, scales=SCALES
    )

    assert isinstance(result, CoreResearchIntegrationCertificate)
    assert isinstance(
        result.left_stability, HeterogeneousDiffusionStabilityCertificate
    )
    assert isinstance(
        result.right_stability, HeterogeneousDiffusionStabilityCertificate
    )
    assert isinstance(
        result.left_reconstruction, EpiDiffusionReconstructionCertificate
    )
    assert isinstance(
        result.right_reconstruction, EpiDiffusionReconstructionCertificate
    )
    assert isinstance(result.left_coarse_graining, EpiCoarseGrainingCertificate)
    assert isinstance(result.right_coarse_graining, EpiCoarseGrainingCertificate)
    assert isinstance(result.structural_distance, StructuralStateDistanceCertificate)

    assert result.joint_numerical_conditions_pass
    assert result.failed_conditions == ()
    assert all(passed for _, passed in result.numerical_conditions)
    assert result.structural_distance.distance > 0.0
    assert result.left_pure_epi_pressure_residual_linf == pytest.approx(0.0)
    assert result.right_pure_epi_pressure_residual_linf == pytest.approx(0.0)
    assert result.left_coarse_graining.nodal_closure_within_tolerance
    assert result.right_coarse_graining.nodal_closure_within_tolerance
    assert len({left.nodes[node]["nu_f"] for node in left}) > 1
    assert len({right.nodes[node]["nu_f"] for node in right}) > 1
    assert "Phase dynamics" in result.scope
    assert "changing support/topology" in result.scope
    assert "OPEN" in result.scope


def test_nonclosing_partition_returns_evidence_without_joint_promotion():
    left, right = _states()

    result = certify_core_research_integration(
        left,
        right,
        ((0, 1), (2, 3)),
        scales=SCALES,
    )

    assert result.left_stability.is_certified
    assert result.right_stability.is_certified
    assert result.left_reconstruction.reconstructs_absolute_epi
    assert result.right_reconstruction.reconstructs_absolute_epi
    assert not result.left_coarse_graining.nodal_closure_within_tolerance
    assert not result.right_coarse_graining.nodal_closure_within_tolerance
    assert not result.joint_numerical_conditions_pass
    assert {
        "left_partition_closure",
        "right_partition_closure",
        "left_nodal_flow_transport",
        "right_nodal_flow_transport",
    }.issubset(result.failed_conditions)


def test_pressure_outside_the_pure_epi_channel_blocks_joint_promotion():
    left, right = _states()
    right.nodes[0]["delta_nfr"] += 0.25
    right.nodes[0]["dEPI_dt"] = (
        right.nodes[0]["nu_f"] * right.nodes[0]["delta_nfr"]
    )

    result = certify_core_research_integration(
        left, right, EXACT_PARTITION, scales=SCALES
    )

    assert result.right_scaled_nodal_equation_residual == pytest.approx(0.0)
    assert result.right_scaled_pure_epi_pressure_residual > result.tolerance
    assert not result.joint_numerical_conditions_pass
    assert "right_pure_epi_pressure_consistency" in result.failed_conditions
    assert "right_nodal_equation_consistency" not in result.failed_conditions


def test_nodal_equation_residual_blocks_joint_promotion_independently():
    left, right = _states()
    right.nodes[0]["dEPI_dt"] += 0.25

    result = certify_core_research_integration(
        left, right, EXACT_PARTITION, scales=SCALES
    )

    assert result.right_scaled_pure_epi_pressure_residual == pytest.approx(0.0)
    assert result.right_scaled_nodal_equation_residual > result.tolerance
    assert not result.joint_numerical_conditions_pass
    assert "right_nodal_equation_consistency" in result.failed_conditions
    assert "right_pure_epi_pressure_consistency" not in result.failed_conditions


def test_changed_support_is_rejected_before_subcertificates_are_composed():
    left, right = _states()
    right.add_edge(0, 2, weight=1.0)

    with pytest.raises(ValueError, match="same bare edge support"):
        certify_core_research_integration(
            left, right, EXACT_PARTITION, scales=SCALES
        )


@pytest.mark.parametrize("tolerance", [0.0, 1.0, math.inf, math.nan, True])
def test_shared_relative_tolerance_must_fit_every_subcertificate(tolerance):
    left, right = _states()

    with pytest.raises(ValueError, match="open interval"):
        certify_core_research_integration(
            left,
            right,
            EXACT_PARTITION,
            scales=SCALES,
            tolerance=tolerance,
        )
