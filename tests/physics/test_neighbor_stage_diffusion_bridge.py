"""One-stage EN/RA represented-map to pure-EPI flow bridge."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import networkx as nx
import pytest

from tnfr.physics import (
    NeighborStageDiffusionBridgeCertificate,
    certify_reception_all_target_stage,
    certify_resonance_all_target_stage,
    compose_neighbor_stage_diffusion_stability,
)


def _graph(
    *,
    topology: nx.Graph | None = None,
    epi: tuple[float, ...] = (0.0, 0.2, 0.9),
) -> nx.Graph:
    graph = topology if topology is not None else nx.path_graph(len(epi))
    for node, value in zip(graph, epi, strict=True):
        graph.nodes[node].update(
            EPI=value,
            nu_f=1.0,
            theta=0.2 * node,
            delta_nfr=0.1,
            Si=0.8,
            EPI_kind="wave",
            glyph_history=["AL", "IL"],
        )
    graph.graph["GLYPH_FACTORS"] = {
        "EN_mix": 0.25,
        "RA_epi_diff": 0.25,
        "RA_vf_amplification": 0.25,
        "RA_phase_coupling": 0.5,
    }
    return graph


def test_en_stage_composes_with_positive_flow_without_runtime_affinity_claim() -> None:
    stage = certify_reception_all_target_stage(
        _graph(), fixed_support_declared=True
    )

    bridge = compose_neighbor_stage_diffusion_stability(stage, 1.0)

    assert isinstance(bridge, NeighborStageDiffusionBridgeCertificate)
    assert bridge.represented_model_bridge_certified
    assert bridge.finite_horizon_disagreement_bound_certified
    assert bridge.disagreement_contracts_over_declared_horizon
    assert bridge.hybrid_certificate is not None
    assert bridge.hybrid_certificate.flow_durations == (0.0, 1.0)
    assert not bridge.hybrid_certificate.repeat_schedule_declared
    assert bridge.repeated_schedule_disagreement_convergence_certified is None
    assert bridge.represented_stage_preserves_post_metric_weighted_mean_exactly
    assert not bridge.runtime_post_metric_weighted_mean_preserved_exactly
    assert bridge.exact_runtime_post_metric_weighted_mean_shift != 0
    assert bridge.hybrid_preserves_initial_weighted_mean
    assert not bridge.runtime_proposal_matches_represented_stage_exactly
    assert bridge.runtime_proposal_matches_represented_stage_within_tolerance
    assert bridge.runtime_pure_epi_pressure_defect_detected
    assert bridge.runtime_pure_epi_pressure_defect_norm > 0.0
    assert bridge.pressure_refresh_required_before_runtime_flow
    assert not bridge.stored_pressure_refresh_certified
    assert not bridge.global_binary64_runtime_affinity_certified
    assert bridge.failed_conditions == ()


def test_ra_post_metric_flow_remains_available_when_capacity_changes_metric() -> None:
    stage = certify_resonance_all_target_stage(
        _graph(epi=(0.0, 0.0, 0.8)),
        fixed_support_declared=True,
        fixed_phase_neighbor_sets_declared=False,
    )

    bridge = compose_neighbor_stage_diffusion_stability(stage, 1.0)

    assert bridge.represented_model_bridge_certified
    assert bridge.disagreement_contracts_over_declared_horizon
    assert not bridge.pre_post_metric_values_identical
    assert not bridge.pre_post_metric_exactly_proportional
    assert not bridge.represented_stage_preserves_post_metric_weighted_mean_exactly
    assert bridge.exact_runtime_post_metric_weighted_mean_shift != 0
    assert not bridge.hybrid_preserves_initial_weighted_mean
    assert bridge.repeated_schedule_disagreement_convergence_certified is None


def test_ra_phase_only_change_still_requires_runtime_pressure_refresh() -> None:
    stage = certify_resonance_all_target_stage(
        _graph(epi=(0.2, 0.2, 0.2)),
        fixed_support_declared=True,
        fixed_phase_neighbor_sets_declared=False,
    )

    bridge = compose_neighbor_stage_diffusion_stability(stage, 1.0)

    assert not bridge.runtime_pure_epi_pressure_defect_detected
    assert bridge.runtime_pure_epi_pressure_defect_norm == 0.0
    assert bridge.pressure_refresh_required_before_runtime_flow
    assert not bridge.stored_pressure_refresh_certified


def test_false_fixed_support_declaration_produces_explicit_abstention() -> None:
    stage = certify_reception_all_target_stage(
        _graph(), fixed_support_declared=False
    )

    bridge = compose_neighbor_stage_diffusion_stability(stage, 1.0)

    assert not bridge.represented_model_bridge_certified
    assert not bridge.finite_horizon_disagreement_bound_certified
    assert bridge.disagreement_contracts_over_declared_horizon is None
    assert bridge.hybrid_certificate is None
    assert "fixed_support_declared" in bridge.failed_conditions
    assert "all_local_snapshots_in_affine_model_domain" in bridge.failed_conditions


def test_atomic_ra_identity_rejection_produces_explicit_abstention() -> None:
    graph = _graph(topology=nx.path_graph(2), epi=(-0.8, 0.2))
    graph.graph["GLYPH_FACTORS"]["RA_epi_diff"] = 1.0
    stage = certify_resonance_all_target_stage(
        graph,
        fixed_support_declared=True,
        fixed_phase_neighbor_sets_declared=True,
    )

    bridge = compose_neighbor_stage_diffusion_stability(stage, 1.0)

    assert bridge.hybrid_certificate is None
    assert not bridge.represented_model_bridge_certified
    assert "runtime_stage_admissible" in bridge.failed_conditions
    assert "all_local_snapshots_in_affine_model_domain" in bridge.failed_conditions
    assert not bridge.runtime_pure_epi_pressure_defect_detected
    assert not bridge.pressure_refresh_required_before_runtime_flow


@pytest.mark.parametrize(
    "duration",
    [True, 0, -1, float("inf"), Fraction(1, 10**1000)],
)
def test_bridge_requires_a_representable_strictly_positive_duration(duration) -> None:
    stage = certify_reception_all_target_stage(
        _graph(), fixed_support_declared=True
    )

    with pytest.raises(ValueError, match="flow_duration"):
        compose_neighbor_stage_diffusion_stability(stage, duration)


def test_bridge_rejects_repeated_or_internally_inconsistent_stage_records() -> None:
    graph = _graph()
    repeated = certify_reception_all_target_stage(
        graph, fixed_support_declared=True, repetitions=2
    )
    with pytest.raises(ValueError, match="exactly one certified stage"):
        compose_neighbor_stage_diffusion_stability(repeated, 1.0)

    stage = certify_reception_all_target_stage(
        graph, fixed_support_declared=True
    )
    bad_step = replace(
        stage.steps[0],
        exact_post_metric_runtime_weighted_mean_shift=Fraction(99),
    )
    with pytest.raises(ValueError, match="weighted-mean shift"):
        compose_neighbor_stage_diffusion_stability(
            replace(stage, steps=(bad_step,)), 1.0
        )

    bad_flow = replace(
        stage.steps[0].post_diffusion_certificate,
        certified_exponential_rate_lower_bound=0.5,
    )
    with pytest.raises(ValueError, match="proof fields"):
        compose_neighbor_stage_diffusion_stability(
            replace(
                stage,
                steps=(replace(stage.steps[0], post_diffusion_certificate=bad_flow),),
            ),
            1.0,
        )


def test_bridge_recomputes_local_domain_instead_of_trusting_promoted_boolean() -> None:
    stage = certify_reception_all_target_stage(
        _graph(), fixed_support_declared=False
    )
    forged_local = replace(
        stage.steps[0].local_certificates[0],
        runtime_snapshot_in_affine_model_domain=True,
    )
    forged_step = replace(
        stage.steps[0],
        local_certificates=(
            forged_local,
            *stage.steps[0].local_certificates[1:],
        ),
    )

    with pytest.raises(ValueError, match="runtime-domain summary"):
        compose_neighbor_stage_diffusion_stability(
            replace(stage, steps=(forged_step,)), 1.0
        )
