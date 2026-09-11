"""Exact realization boundary for immutable pointwise Jacobi proposals."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from fractions import Fraction
import math
from typing import Any

import networkx as nx
import pytest

from tnfr.constants.aliases import (
    ALIAS_DNFR,
    ALIAS_EPI,
    ALIAS_THETA,
    ALIAS_VF,
)
from tnfr.operators.factor_contracts import resolve_runtime_operator_factors
from tnfr.operators.network_stage import (
    PointwiseStageProposal,
    _propose_pointwise,
    execute_pointwise_stage,
)
from tnfr.operators.transition import Transition
import tnfr.physics.pointwise_stage_stability as pointwise
from tnfr.physics.pointwise_stage_stability import (
    certify_pointwise_epi_jump_realization,
)
from tnfr.physics.structural_diffusion import (
    structural_diffusion_operator,
    verify_heterogeneous_diffusion_stability,
)
from tnfr.types import Glyph


_NOW = datetime(2032, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
_TIMESTAMP = _NOW.isoformat()
_ZHIR_TAU = 0.01
_FACTORS = {
    Glyph.AL: {"AL_boost": 0.125},
    Glyph.SHA: {"SHA_vf_factor": 0.5},
    Glyph.VAL: {"VAL_scale": 2.0},
    Glyph.NUL: {"NUL_scale": 0.5},
    Glyph.ZHIR: {"ZHIR_theta_shift_factor": 0.5},
    Glyph.NAV: {"NAV_eta": 0.25, "NAV_jitter": 0.0},
}


def _graph(glyph: Glyph) -> nx.Graph:
    graph = nx.path_graph(3)
    graph.graph.update(
        GLYPH_FACTORS=deepcopy(_FACTORS[glyph]),
        EDGE_AWARE_ENABLED=True,
        EPI_MIN=-8.0,
        EPI_MAX=8.0,
        CLIP_MODE="hard",
        NAV_RANDOM=False,
        NAV_STRICT=False,
        ZHIR_THRESHOLD_XI=0.05,
    )
    for node, epi in enumerate((0.125, 0.25, 0.375)):
        graph.nodes[node].update(
            {
                ALIAS_EPI[0]: epi,
                ALIAS_VF[0]: 1.0,
                ALIAS_DNFR[0]: 0.125 * (node + 1),
                ALIAS_THETA[0]: 0.125 * node,
                "EPI_kind": "wave",
                "glyph_history": ["AL", "IL", "OZ"],
                "epi_history": [epi - 0.25, epi - 0.125, epi],
            }
        )
    return graph


def _plain_graph_state(graph: nx.Graph) -> tuple[Any, ...]:
    return (
        tuple(
            (node, deepcopy(dict(data)))
            for node, data in graph.nodes(data=True)
        ),
        tuple(
            (left, right, deepcopy(dict(data)))
            for left, right, data in graph.edges(data=True)
        ),
        deepcopy(dict(graph.graph)),
        (
            hasattr(graph, "_last_operator_applied"),
            getattr(graph, "_last_operator_applied", None),
        ),
    )


def _stage_proposals(
    graph: nx.Graph,
    glyph: Glyph,
    *,
    targets: tuple[Any, ...] | None = None,
    nav_kwargs: dict[str, Any] | None = None,
    nav_seed: int | None = None,
    nav_offsets: dict[Any, int] | None = None,
    stage_timestamp: str = _TIMESTAMP,
    zhir_tau: Any = _ZHIR_TAU,
) -> tuple[PointwiseStageProposal, ...]:
    factors = resolve_runtime_operator_factors(
        graph.graph.get("GLYPH_FACTORS"), glyph, graph.graph
    )
    target_nodes = tuple(graph) if targets is None else targets
    execution_kwargs = dict(nav_kwargs or {})
    operator = Transition() if glyph is Glyph.NAV else None
    offsets = dict(nav_offsets or {})
    return tuple(
        _propose_pointwise(
            graph,
            node,
            operator,
            glyph,
            factors,
            timestamp=(
                stage_timestamp
                if glyph in (Glyph.AL, Glyph.SHA)
                else None
            ),
            tau=zhir_tau if glyph is Glyph.ZHIR else None,
            transition_now=_NOW if glyph is Glyph.NAV else None,
            resolved_seed=nav_seed,
            node_offset=offsets.get(node),
            execution_kwargs=execution_kwargs,
        )
        for node in target_nodes
    )


def _certificate(
    graph: nx.Graph,
    glyph: Glyph,
    proposals: tuple[PointwiseStageProposal, ...],
    *,
    fixed_support_declared: bool = True,
    nav_kwargs: dict[str, Any] | None = None,
    replay_nav: bool = True,
    nav_seed: int | None = None,
    nav_offsets: dict[Any, int] | None = None,
):
    kwargs: dict[str, Any] = {}
    if glyph in (Glyph.AL, Glyph.SHA):
        kwargs["stage_timestamp"] = _TIMESTAMP
    elif glyph is Glyph.ZHIR:
        kwargs["zhir_tau"] = _ZHIR_TAU
    elif glyph is Glyph.NAV and replay_nav:
        kwargs.update(
            nav_transition_now=_NOW,
            nav_resolved_seed=nav_seed,
            nav_node_offsets=dict(nav_offsets or {}),
            nav_execution_kwargs=dict(nav_kwargs or {}),
        )
    return certify_pointwise_epi_jump_realization(
        graph,
        proposals,
        fixed_support_declared=fixed_support_declared,
        **kwargs,
    )


@pytest.mark.parametrize(
    ("glyph", "gain", "metric_scale", "nav_kwargs"),
    (
        (Glyph.AL, Fraction(1), Fraction(1), None),
        (Glyph.SHA, Fraction(1), Fraction(2), None),
        (Glyph.VAL, Fraction(4), Fraction(1, 2), None),
        (Glyph.NUL, Fraction(1, 4), Fraction(2), None),
        (Glyph.ZHIR, Fraction(1), Fraction(1), None),
        (
            Glyph.NAV,
            Fraction(1),
            Fraction(1),
            {"vf_factor": 1.0, "phase_shift": 0.125},
        ),
    ),
)
def test_all_six_pointwise_glyphs_certify_all_three_levels(
    glyph: Glyph,
    gain: Fraction,
    metric_scale: Fraction,
    nav_kwargs: dict[str, Any] | None,
) -> None:
    graph = _graph(glyph)
    proposals = _stage_proposals(graph, glyph, nav_kwargs=nav_kwargs)
    certificate = _certificate(
        graph,
        glyph,
        proposals,
        nav_kwargs=nav_kwargs,
    )

    assert certificate.glyph == glyph.value
    assert certificate.proposal_builder_replayed_from_declared_inputs
    assert certificate.runtime_matches_represented_affine_exactly
    assert certificate.pre_post_metric_exactly_proportional
    assert certificate.exact_post_to_pre_metric_scale == metric_scale
    assert certificate.exact_common_metric_energy_gain_bound == gain
    assert certificate.runtime_epi_realization_certified
    assert certificate.pre_metric_affine_gain_certified
    assert certificate.pre_post_common_metric_bridge_certified
    assert certificate.operational_affine_gain_available
    assert certificate.supports_runtime_affine_gain
    assert certificate.supports_hybrid_common_metric_bridge
    assert certificate.failed_conditions == ()


@pytest.mark.parametrize("glyph", tuple(_FACTORS))
def test_certificate_is_read_only_for_graph_and_frozen_proposals(
    glyph: Glyph,
) -> None:
    graph = _graph(glyph)
    nav_kwargs = (
        {"vf_factor": 1.0, "phase_shift": 0.125}
        if glyph is Glyph.NAV
        else None
    )
    proposals = _stage_proposals(graph, glyph, nav_kwargs=nav_kwargs)
    before = _plain_graph_state(graph)
    certificate = _certificate(
        graph,
        glyph,
        proposals,
        nav_kwargs=nav_kwargs,
    )

    assert _plain_graph_state(graph) == before
    rebuilt = _stage_proposals(graph, glyph, nav_kwargs=nav_kwargs)
    assert tuple(rebuilt) == proposals
    with pytest.raises(ValueError):
        certificate.state_before[0] = 99.0


def test_fixed_support_is_only_a_level_c_hypothesis() -> None:
    graph = _graph(Glyph.AL)
    proposals = _stage_proposals(graph, Glyph.AL)
    certificate = _certificate(
        graph,
        Glyph.AL,
        proposals,
        fixed_support_declared=False,
    )

    assert certificate.runtime_epi_realization_certified
    assert certificate.pre_metric_affine_gain_certified
    assert not certificate.pre_post_common_metric_bridge_certified
    assert certificate.supports_runtime_affine_gain
    assert not certificate.supports_hybrid_common_metric_bridge
    assert certificate.failed_conditions == ("fixed_support_declared",)


def test_subset_stage_abstains_at_level_a() -> None:
    graph = _graph(Glyph.AL)
    proposals = _stage_proposals(graph, Glyph.AL)[:2]
    certificate = _certificate(graph, Glyph.AL, proposals)

    assert not certificate.runtime_epi_realization_certified
    assert not certificate.pre_metric_affine_gain_certified
    assert not certificate.pre_post_common_metric_bridge_certified
    assert (
        "all_graph_nodes_targeted_once"
        in certificate.failed_runtime_epi_realization_conditions
    )
    assert certificate.affine_jump_certificate is None


def test_level_a_does_not_require_a_diffusion_flow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _graph(Glyph.AL)
    proposals = _stage_proposals(graph, Glyph.AL)
    monkeypatch.setattr(
        pointwise,
        "_flow_certificate",
        lambda *_args: (None, "flow unavailable"),
    )

    certificate = _certificate(graph, Glyph.AL, proposals)

    assert certificate.runtime_epi_realization_certified
    assert not certificate.pre_metric_affine_gain_certified
    assert not certificate.pre_post_common_metric_bridge_certified
    assert certificate.affine_jump_certificate is None
    assert "pre_diffusion_flow_certified" in certificate.failed_conditions


@pytest.mark.parametrize("mode", ("hard", "soft"))
def test_emission_abstains_when_boundary_map_is_active_or_soft(
    mode: str,
) -> None:
    graph = _graph(Glyph.AL)
    graph.graph.update(CLIP_MODE=mode, EPI_MAX=0.4)
    proposals = _stage_proposals(graph, Glyph.AL)
    certificate = _certificate(graph, Glyph.AL, proposals)

    assert not certificate.runtime_epi_realization_certified
    assert certificate.affine_jump_certificate is None
    if mode == "soft":
        assert (
            "supported_hard_or_inactive_clip_policy"
            in certificate.failed_runtime_epi_realization_conditions
        )
    assert certificate.clip_intervention_nodes


def test_val_exactly_detects_hidden_edge_adaptation() -> None:
    graph = _graph(Glyph.VAL)
    # This effective scale differs from 2.0 by less than the runtime reporting
    # epsilon, but it is still a different represented affine coefficient.
    graph.graph.update(EPI_MAX=0.74, EDGE_AWARE_EPSILON=0.1)
    proposals = _stage_proposals(graph, Glyph.VAL)
    certificate = _certificate(graph, Glyph.VAL, proposals)

    assert certificate.edge_adaptation_nodes
    assert (
        "edge_adaptation_inactive"
        in certificate.failed_runtime_epi_realization_conditions
    )
    assert not certificate.supports_runtime_affine_gain


def test_sha_zero_capacity_keeps_levels_a_and_b_separate_from_c() -> None:
    graph = _graph(Glyph.SHA)
    graph.graph["GLYPH_FACTORS"] = {"SHA_vf_factor": 0.0}
    proposals = _stage_proposals(graph, Glyph.SHA)
    certificate = _certificate(graph, Glyph.SHA, proposals)

    assert certificate.zero_post_capacity_nodes == tuple(graph)
    assert certificate.runtime_epi_realization_certified
    assert certificate.pre_metric_affine_gain_certified
    assert not certificate.pre_post_common_metric_bridge_certified
    assert certificate.supports_runtime_affine_gain
    assert not certificate.supports_hybrid_common_metric_bridge
    assert "post_diffusion_flow_certified" in certificate.failed_conditions


def test_nav_missing_inputs_and_nonproportional_post_metric_are_distinct() -> None:
    graph = _graph(Glyph.NAV)
    graph.nodes[0][ALIAS_EPI[0]] = 0.75
    graph.nodes[1][ALIAS_EPI[0]] = 0.2
    graph.nodes[1][ALIAS_VF[0]] = 0.02
    graph.nodes[1]["latent"] = True
    nav_kwargs = {"vf_factor": 1.0, "phase_shift": 0.125}
    proposals = _stage_proposals(graph, Glyph.NAV, nav_kwargs=nav_kwargs)
    unbound = _certificate(
        graph,
        Glyph.NAV,
        proposals,
        nav_kwargs=nav_kwargs,
        replay_nav=False,
    )
    varying = _certificate(
        graph,
        Glyph.NAV,
        proposals,
        nav_kwargs=nav_kwargs,
    )

    assert not unbound.runtime_epi_realization_certified
    assert (
        "proposal_builder_replayed_from_declared_inputs"
        in unbound.failed_runtime_epi_realization_conditions
    )
    assert unbound.affine_jump_certificate is None
    assert varying.runtime_epi_realization_certified
    assert varying.pre_metric_affine_gain_certified
    assert not varying.pre_post_common_metric_bridge_certified
    assert not varying.pre_post_metric_exactly_proportional
    assert varying.affine_jump_certificate is not None
    assert varying.supports_runtime_affine_gain
    assert not varying.supports_hybrid_common_metric_bridge


def test_nul_pressure_diagnostics_are_separate_from_epi_gain() -> None:
    graph = _graph(Glyph.NUL)
    proposals = _stage_proposals(graph, Glyph.NUL)
    certificate = _certificate(graph, Glyph.NUL, proposals)

    assert certificate.pre_post_common_metric_bridge_certified
    assert certificate.exact_nul_stored_pressure_map_residual == (
        Fraction(0),
        Fraction(0),
        Fraction(0),
    )
    assert certificate.exact_nul_nodal_drive_residual == (
        Fraction(0),
        Fraction(0),
        Fraction(0),
    )
    assert (
        certificate.exact_nul_stored_pressure_minus_pure_epi_pressure
        is not None
    )
    assert certificate.nul_stored_pressure_defect_norm is not None
    assert certificate.nul_stored_pressure_defect_norm > 0.0
    assert certificate.nul_pressure_diagnostic_abstention_reason is None


@pytest.mark.parametrize("glyph", (Glyph.VAL, Glyph.NUL))
def test_scale_stage_without_edge_awareness_has_identity_epi_gain(
    glyph: Glyph,
) -> None:
    graph = _graph(glyph)
    graph.graph["EDGE_AWARE_ENABLED"] = False
    epi_before = tuple(graph.nodes[node][ALIAS_EPI[0]] for node in graph)
    vf_before = tuple(graph.nodes[node][ALIAS_VF[0]] for node in graph)
    proposals = _stage_proposals(graph, glyph)
    certificate = _certificate(graph, glyph, proposals)

    assert all(not proposal.payload.write_epi for proposal in proposals)
    assert certificate.runtime_proposed_state_after.tolist() == list(epi_before)
    assert certificate.represented_linear_map.tolist() == [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    assert tuple(proposal.payload.vf_after for proposal in proposals) != vf_before
    assert certificate.exact_common_metric_energy_gain_bound == Fraction(1)
    assert certificate.pre_post_common_metric_bridge_certified
    assert certificate.failed_conditions == ()


def test_binary64_rounding_residual_prevents_false_runtime_affinity() -> None:
    graph = _graph(Glyph.AL)
    graph.graph["GLYPH_FACTORS"] = {"AL_boost": 0.1}
    for node, epi in enumerate((0.2, 0.3, 0.4)):
        graph.nodes[node][ALIAS_EPI[0]] = epi
    proposals = _stage_proposals(graph, Glyph.AL)
    certificate = _certificate(graph, Glyph.AL, proposals)

    assert certificate.runtime_matches_represented_affine_within_tolerance
    assert not certificate.runtime_matches_represented_affine_exactly
    assert any(certificate.exact_runtime_minus_represented_affine)
    assert (
        "runtime_matches_represented_affine_exactly"
        in certificate.failed_runtime_epi_realization_conditions
    )
    assert certificate.affine_jump_certificate is None


@pytest.mark.parametrize("glyph", (Glyph.AL, Glyph.SHA, Glyph.ZHIR, Glyph.NAV))
def test_missing_exogenous_builder_inputs_cause_level_a_abstention(
    glyph: Glyph,
) -> None:
    graph = _graph(glyph)
    nav_kwargs = (
        {"vf_factor": 1.0, "phase_shift": 0.125}
        if glyph is Glyph.NAV
        else None
    )
    proposals = _stage_proposals(graph, glyph, nav_kwargs=nav_kwargs)
    certificate = certify_pointwise_epi_jump_realization(
        graph,
        proposals,
        fixed_support_declared=True,
    )

    assert not certificate.proposal_builder_replayed_from_declared_inputs
    assert certificate.proposal_replay_abstention_reason is not None
    assert not certificate.runtime_epi_realization_certified
    assert (
        "proposal_builder_replayed_from_declared_inputs"
        in certificate.failed_runtime_epi_realization_conditions
    )


@pytest.mark.parametrize(
    ("glyph", "declared"),
    (
        (Glyph.AL, {"stage_timestamp": "forged"}),
        (Glyph.SHA, {"stage_timestamp": "forged"}),
        (Glyph.ZHIR, {"zhir_tau": 0.02}),
    ),
)
def test_forged_exogenous_builder_inputs_reject_proposals(
    glyph: Glyph,
    declared: dict[str, Any],
) -> None:
    graph = _graph(glyph)
    proposals = _stage_proposals(graph, glyph)

    with pytest.raises(ValueError, match="tampered|stale"):
        certify_pointwise_epi_jump_realization(
            graph,
            proposals,
            fixed_support_declared=True,
            **declared,
        )


def test_explicit_none_zhir_tau_uses_the_declared_graph_default() -> None:
    graph = _graph(Glyph.ZHIR)
    proposals = _stage_proposals(graph, Glyph.ZHIR, zhir_tau=None)

    certificate = certify_pointwise_epi_jump_realization(
        graph,
        proposals,
        fixed_support_declared=True,
        zhir_tau=None,
    )

    assert certificate.proposal_builder_replayed_from_declared_inputs
    assert certificate.runtime_epi_realization_certified


def test_random_nav_replay_binds_seed_offsets_and_draws() -> None:
    graph = _graph(Glyph.NAV)
    graph.graph.update(
        GLYPH_FACTORS={"NAV_eta": 0.25, "NAV_jitter": 0.05},
        NAV_RANDOM=True,
    )
    nav_kwargs = {"vf_factor": 1.0, "phase_shift": 0.125}
    offsets = {node: 10 + node for node in graph}
    seed = 1459
    proposals = _stage_proposals(
        graph,
        Glyph.NAV,
        nav_kwargs=nav_kwargs,
        nav_seed=seed,
        nav_offsets=offsets,
    )
    certificate = _certificate(
        graph,
        Glyph.NAV,
        proposals,
        nav_kwargs=nav_kwargs,
        nav_seed=seed,
        nav_offsets=offsets,
    )

    assert all(proposal.payload.jitter_proposal is not None for proposal in proposals)
    assert certificate.pre_post_common_metric_bridge_certified
    missing_rng = certify_pointwise_epi_jump_realization(
        graph,
        proposals,
        fixed_support_declared=True,
        nav_transition_now=_NOW,
        nav_execution_kwargs=nav_kwargs,
    )
    assert not missing_rng.runtime_epi_realization_certified
    assert "resolved graph seed" in (
        missing_rng.proposal_replay_abstention_reason or ""
    )
    with pytest.raises(ValueError, match="tampered|stale"):
        _certificate(
            graph,
            Glyph.NAV,
            proposals,
            nav_kwargs=nav_kwargs,
            nav_seed=seed + 1,
            nav_offsets=offsets,
        )


def test_replayed_proposal_and_returned_certificate_reject_tampering() -> None:
    graph = _graph(Glyph.ZHIR)
    proposals = _stage_proposals(graph, Glyph.ZHIR)
    changed_payload = replace(
        proposals[0].payload,
        structural_acceleration=(
            proposals[0].payload.structural_acceleration + 1.0
        ),
    )
    changed = (
        replace(proposals[0], payload=changed_payload),
        proposals[1],
        proposals[2],
    )

    with pytest.raises(ValueError, match="tampered|stale"):
        _certificate(graph, Glyph.ZHIR, changed)

    certificate = _certificate(graph, Glyph.ZHIR, proposals)
    replacements = (
        replace(
            certificate,
            exact_common_metric_energy_gain_bound=Fraction(99),
        ),
        replace(certificate, stage_schedule="operator_major_gauss_seidel"),
        replace(certificate, fixed_support_declared=False),
        replace(
            certificate,
            proposal_builder_replayed_from_declared_inputs=False,
        ),
        replace(certificate, clip_intervention_nodes=(0,)),
        replace(certificate, pre_post_metric_exactly_proportional=False),
        replace(certificate, tolerance=1e-4),
        replace(certificate, scope="forged"),
    )
    for replaced in replacements:
        assert not replaced._proof_fields_are_intact()
        assert not replaced.supports_runtime_affine_gain
        assert "certificate_proof_fields_intact" in replaced.failed_conditions

    nested = certificate.affine_jump_certificate
    assert nested is not None
    forged_nested_certificates = (
        replace(nested, operator_name="Dissonance"),
        replace(nested, glyph="OZ"),
        replace(nested, primary_channel="DeltaNFR"),
        replace(nested, operator_scale="NETWORK"),
    )
    for forged_nested in forged_nested_certificates:
        assert forged_nested._proof_fields_are_intact()
        forged_outer = replace(
            certificate,
            affine_jump_certificate=forged_nested,
        )
        assert not forged_outer._proof_fields_are_intact()
        assert not forged_outer.supports_runtime_affine_gain

    original_conditions = certificate.runtime_epi_realization_conditions
    integer_condition = (
        (original_conditions[0][0], 1),
        *original_conditions[1:],
    )
    type_forged = (
        replace(certificate, runtime_epi_realization_certified=1),
        replace(certificate, fixed_support_declared=1),
        replace(
            certificate,
            exact_common_metric_energy_gain_bound=True,
        ),
        replace(certificate, exact_post_to_pre_metric_scale=True),
        replace(
            certificate,
            exact_represented_linear_map=(
                (True, Fraction(0), Fraction(0)),
                *certificate.exact_represented_linear_map[1:],
            ),
        ),
        replace(
            certificate,
            exact_runtime_minus_represented_affine=(
                False,
                *certificate.exact_runtime_minus_represented_affine[1:],
            ),
        ),
        replace(
            certificate,
            runtime_epi_realization_conditions=integer_condition,
        ),
    )
    for replaced in type_forged:
        assert not replaced._proof_fields_are_intact()
        assert not replaced.supports_runtime_affine_gain

    certificate.represented_linear_map.setflags(write=True)
    certificate.represented_linear_map[0, 0] = 2.0
    assert not certificate._proof_fields_are_intact()
    assert not certificate.supports_runtime_affine_gain


def test_pointwise_proof_stamp_rejects_hostile_equality_without_dispatch() -> None:
    class Probe:
        called = False

        def __eq__(self, _other):
            self.called = True
            raise SystemExit("proof-stamp equality must not be dispatched")

        def __bool__(self):
            self.called = True
            raise SystemExit("proof-stamp truthiness must not be dispatched")

    graph = _graph(Glyph.ZHIR)
    certificate = _certificate(
        graph, Glyph.ZHIR, _stage_proposals(graph, Glyph.ZHIR)
    )
    probe = Probe()
    original = object.__getattribute__(certificate, "_proof_stamp")
    object.__setattr__(certificate, "_proof_stamp", (probe, *original[1:]))

    assert not certificate._proof_fields_are_intact()
    assert not certificate.supports_runtime_affine_gain
    assert not probe.called


def test_pointwise_nested_semantic_equality_cannot_escape_integrity_check() -> None:
    class HostileName:
        def __eq__(self, _other):
            raise SystemExit("nested semantic equality must fail closed")

    graph = _graph(Glyph.ZHIR)
    certificate = _certificate(
        graph, Glyph.ZHIR, _stage_proposals(graph, Glyph.ZHIR)
    )
    nested = certificate.affine_jump_certificate
    assert nested is not None
    forged_nested = replace(nested, operator_name=HostileName())
    assert forged_nested._proof_fields_are_intact()
    forged_outer = replace(
        certificate,
        affine_jump_certificate=forged_nested,
    )

    assert not forged_outer._proof_fields_are_intact()
    assert not forged_outer.supports_runtime_affine_gain


def test_stale_factor_configuration_rejects_proposal_instead_of_refitting() -> None:
    graph = _graph(Glyph.VAL)
    proposals = _stage_proposals(graph, Glyph.VAL)
    graph.graph["GLYPH_FACTORS"] = {"VAL_scale": 1.5}

    with pytest.raises(ValueError, match="tampered|stale"):
        _certificate(graph, Glyph.VAL, proposals)


@pytest.mark.parametrize(
    ("glyph", "missing_field", "message"),
    (
        (Glyph.VAL, "epi_after", "required EPI fields"),
        (Glyph.NUL, "dnfr_after", "required pressure fields"),
    ),
)
def test_malformed_scale_payload_raises_without_runtime_asserts(
    glyph: Glyph,
    missing_field: str,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _graph(glyph)
    proposals = _stage_proposals(graph, glyph)
    malformed_payload = replace(proposals[0].payload, **{missing_field: None})
    malformed = (
        replace(proposals[0], payload=malformed_payload),
        proposals[1],
        proposals[2],
    )
    monkeypatch.setattr(
        pointwise,
        "_replay_proposals_from_declared_inputs",
        lambda *args, **kwargs: (True, None),
    )

    with pytest.raises(ValueError, match=message):
        _certificate(graph, glyph, malformed)


class _CopyReturnsSelf(nx.Graph):
    copy_calls = 0

    def copy(self, as_view: bool = False):
        type(self).copy_calls += 1
        return self


class _CopyReordersNodes(nx.Graph):
    copy_calls = 0

    def copy(self, as_view: bool = False):
        type(self).copy_calls += 1
        copied = nx.Graph()
        for node in reversed(tuple(self.nodes)):
            copied.add_node(node, **dict(self.nodes[node]))
        copied.add_edges_from(self.edges(data=True))
        copied.graph.update(dict(self.graph))
        return copied


def _hostile_graph(graph_type: type[nx.Graph], glyph: Glyph) -> nx.Graph:
    source = _graph(glyph)
    graph = graph_type()
    graph.graph.update(deepcopy(dict(source.graph)))
    graph.add_nodes_from(
        (node, deepcopy(dict(data)))
        for node, data in source.nodes(data=True)
    )
    graph.add_edges_from(
        (left, right, deepcopy(dict(data)))
        for left, right, data in source.edges(data=True)
    )
    graph_type.copy_calls = 0
    return graph


@pytest.mark.parametrize("graph_type", (_CopyReturnsSelf, _CopyReordersNodes))
def test_controlled_copy_ignores_hostile_copy_overrides(
    graph_type: type[nx.Graph],
) -> None:
    graph = _hostile_graph(graph_type, Glyph.AL)
    proposals = _stage_proposals(graph, Glyph.AL)
    before = _plain_graph_state(graph)
    certificate = _certificate(graph, Glyph.AL, proposals)

    assert graph_type.copy_calls == 0
    assert _plain_graph_state(graph) == before
    assert certificate.logical_copy_verified
    assert certificate.pre_flow_node_order_matches
    assert certificate.post_flow_node_order_matches
    assert certificate.pre_post_common_metric_bridge_certified


def test_logical_copy_preserves_base_multidigraph_structure_and_mappings() -> None:
    graph = nx.MultiDiGraph()
    graph.graph["config"] = {"mode": "test"}
    graph.add_node("b", payload=[1])
    graph.add_node("a", payload=[2])
    graph.add_edge("a", "b", key="second", weight=0.2)
    graph.add_edge("a", "b", key="first", weight=0.1)
    nodes = tuple(graph.nodes)
    edges = tuple(graph.edges(keys=True))
    copied = pointwise._logical_copy(graph)

    assert type(copied) is nx.MultiDiGraph
    assert tuple(copied.nodes) == nodes
    assert tuple(copied.edges(keys=True)) == edges
    assert copied.graph == graph.graph
    assert copied.graph is not graph.graph
    for node in nodes:
        assert copied.nodes[node] == graph.nodes[node]
        assert copied.nodes[node] is not graph.nodes[node]
    for left, right, key in edges:
        assert copied.edges[left, right, key] == graph.edges[left, right, key]
        assert copied.edges[left, right, key] is not graph.edges[left, right, key]
    copied.graph["new"] = True
    copied.nodes["a"]["new"] = True
    copied.edges["a", "b", "first"]["new"] = True
    assert "new" not in graph.graph
    assert "new" not in graph.nodes["a"]
    assert "new" not in graph.edges["a", "b", "first"]


def test_post_flow_identity_order_mismatch_blocks_only_level_c(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _graph(Glyph.AL)
    proposals = _stage_proposals(graph, Glyph.AL)
    calls = 0
    original = pointwise._flow_certificate

    def delegated_flow(candidate: nx.Graph, tolerance: float):
        nonlocal calls
        calls += 1
        if calls == 1:
            return original(candidate, tolerance)
        reversed_graph = nx.Graph()
        reversed_graph.graph.update(dict(candidate.graph))
        for node in reversed(tuple(candidate.nodes)):
            reversed_graph.add_node(node, **dict(candidate.nodes[node]))
        reversed_graph.add_edges_from(candidate.edges(data=True))
        return (
            verify_heterogeneous_diffusion_stability(
                reversed_graph,
                tolerance=tolerance,
            ),
            None,
        )

    monkeypatch.setattr(pointwise, "_flow_certificate", delegated_flow)
    certificate = _certificate(graph, Glyph.AL, proposals)

    assert certificate.runtime_epi_realization_certified
    assert certificate.pre_metric_affine_gain_certified
    assert not certificate.post_flow_node_order_matches
    assert not certificate.pre_post_common_metric_bridge_certified
    assert "post_flow_node_order_matches" in certificate.failed_conditions


def test_pre_flow_identity_order_mismatch_blocks_level_b(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _graph(Glyph.AL)
    proposals = _stage_proposals(graph, Glyph.AL)
    original = pointwise._flow_certificate
    calls = 0

    def delegated_flow(candidate: nx.Graph, tolerance: float):
        nonlocal calls
        calls += 1
        if calls > 1:
            return original(candidate, tolerance)
        reversed_graph = nx.Graph()
        reversed_graph.graph.update(dict(candidate.graph))
        for node in reversed(tuple(candidate.nodes)):
            reversed_graph.add_node(node, **dict(candidate.nodes[node]))
        reversed_graph.add_edges_from(candidate.edges(data=True))
        return (
            verify_heterogeneous_diffusion_stability(
                reversed_graph,
                tolerance=tolerance,
            ),
            None,
        )

    monkeypatch.setattr(pointwise, "_flow_certificate", delegated_flow)
    certificate = _certificate(graph, Glyph.AL, proposals)

    assert certificate.runtime_epi_realization_certified
    assert not certificate.pre_flow_node_order_matches
    assert not certificate.pre_metric_affine_gain_certified
    assert not certificate.pre_post_common_metric_bridge_certified
    assert "pre_flow_node_order_matches" in certificate.failed_conditions


def test_extreme_finite_scale_retains_exact_gain_when_display_overflows() -> None:
    graph = _graph(Glyph.VAL)
    scale = math.ldexp(1.0, 600)
    inverse_scale = math.ldexp(1.0, -600)
    graph.graph["GLYPH_FACTORS"] = {"VAL_scale": scale}
    for node in graph:
        graph.nodes[node][ALIAS_EPI[0]] = (node + 1) * inverse_scale
        graph.nodes[node][ALIAS_VF[0]] = inverse_scale
    proposals = _stage_proposals(graph, Glyph.VAL)
    certificate = _certificate(graph, Glyph.VAL, proposals)

    expected = Fraction.from_float(scale) ** 2
    assert certificate.runtime_epi_realization_certified
    assert certificate.affine_jump_certificate is not None
    assert certificate.exact_common_metric_energy_gain_bound == expected
    assert not certificate.operational_affine_gain_available
    assert certificate.pre_metric_affine_gain_certified
    assert certificate.pre_post_common_metric_bridge_certified
    assert certificate.supports_runtime_affine_gain
    assert certificate.supports_hybrid_common_metric_bridge
    assert certificate.failed_pre_metric_affine_gain_conditions == ()
    assert certificate.failed_common_metric_bridge_conditions == ()
    assert (
        "finite_binary64_affine_gain_display"
        in certificate.failed_operational_affine_gain_conditions
    )
    assert certificate.failed_conditions


def test_nul_pressure_matvec_is_exact_over_represented_fractions() -> None:
    graph = _graph(Glyph.NUL)
    graph[0][1]["weight"] = 0.1
    graph[1][2]["weight"] = 0.2
    for node, epi in enumerate((0.1, 0.2, 0.4)):
        graph.nodes[node][ALIAS_EPI[0]] = epi
    proposals = _stage_proposals(graph, Glyph.NUL)
    certificate = _certificate(graph, Glyph.NUL, proposals)

    post = pointwise._logical_copy(graph)
    for proposal in proposals:
        node = proposal.node
        post.nodes[node][ALIAS_EPI[0]] = proposal.payload.epi_after
        post.nodes[node][ALIAS_VF[0]] = proposal.payload.vf_after
        post.nodes[node][ALIAS_DNFR[0]] = proposal.payload.dnfr_after
    pressure_nodes, laplacian = structural_diffusion_operator(post)
    assert tuple(pressure_nodes) == tuple(graph)
    laplacian_q = tuple(
        tuple(Fraction.from_float(float(value)) for value in row)
        for row in laplacian
    )
    epi_q = tuple(
        Fraction.from_float(float(proposal.payload.epi_after))
        for proposal in proposals
    )
    pure_q = tuple(
        -sum(
            (coefficient * value for coefficient, value in zip(row, epi_q)),
            Fraction(0),
        )
        for row in laplacian_q
    )
    stored_q = tuple(
        Fraction.from_float(float(proposal.payload.dnfr_after))
        for proposal in proposals
    )
    expected = tuple(
        stored - pure
        for stored, pure in zip(stored_q, pure_q, strict=True)
    )
    rounded = tuple(
        Fraction.from_float(
            float(stored)
            + sum(float(a) * float(x) for a, x in zip(row, epi_q))
        )
        for stored, row in zip(stored_q, laplacian_q, strict=True)
    )

    assert (
        certificate.exact_nul_stored_pressure_minus_pure_epi_pressure
        == expected
    )
    assert expected != rounded
    assert certificate.nul_pressure_diagnostic_abstention_reason is None


def test_nul_pressure_diagnostic_abstention_is_explicit_and_nondecisive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _graph(Glyph.NUL)
    proposals = _stage_proposals(graph, Glyph.NUL)

    def unavailable(_graph: nx.Graph):
        raise ValueError("diagnostic unavailable")

    monkeypatch.setattr(
        pointwise,
        "structural_diffusion_operator",
        unavailable,
    )
    certificate = _certificate(graph, Glyph.NUL, proposals)

    assert certificate.pre_post_common_metric_bridge_certified
    assert (
        certificate.exact_nul_stored_pressure_minus_pure_epi_pressure
        is None
    )
    assert certificate.nul_stored_pressure_defect_norm is None
    assert certificate.nul_pressure_diagnostic_abstention_reason is not None
    assert "diagnostic unavailable" in (
        certificate.nul_pressure_diagnostic_abstention_reason
    )
    assert certificate._proof_fields_are_intact()


def test_nul_pressure_node_order_mismatch_has_a_sealed_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _graph(Glyph.NUL)
    proposals = _stage_proposals(graph, Glyph.NUL)
    original = pointwise.structural_diffusion_operator

    def reordered(candidate: nx.Graph):
        nodes, laplacian = original(candidate)
        return tuple(reversed(tuple(nodes))), laplacian

    monkeypatch.setattr(
        pointwise,
        "structural_diffusion_operator",
        reordered,
    )
    certificate = _certificate(graph, Glyph.NUL, proposals)

    assert certificate.pre_post_common_metric_bridge_certified
    assert (
        certificate.exact_nul_stored_pressure_minus_pure_epi_pressure
        is None
    )
    assert "node order changed" in (
        certificate.nul_pressure_diagnostic_abstention_reason or ""
    )
    assert certificate._proof_fields_are_intact()

@pytest.mark.parametrize(
    ("glyph", "operator_type", "execution_kwargs"),
    (
        (Glyph.AL, "Emission", {}),
        (Glyph.SHA, "Silence", {}),
        (Glyph.VAL, "Expansion", {}),
        (Glyph.NUL, "Contraction", {}),
        (Glyph.ZHIR, "Mutation", {"tau": _ZHIR_TAU}),
        (Glyph.NAV, "Transition", {}),
    ),
)
def test_executor_opt_in_certifies_its_own_frozen_proposals(
    glyph: Glyph,
    operator_type: str,
    execution_kwargs: dict[str, Any],
) -> None:
    from tnfr.operators.definitions import (
        Contraction,
        Emission,
        Expansion,
        Mutation,
        Silence,
    )

    operator_types = {
        "Emission": Emission,
        "Silence": Silence,
        "Expansion": Expansion,
        "Contraction": Contraction,
        "Mutation": Mutation,
        "Transition": Transition,
    }
    graph = _graph(glyph)
    nodes = (2, 0, 1)
    epi_before = tuple(_raw for _raw in (
        float(graph.nodes[node][ALIAS_EPI[0]]) for node in graph
    ))

    result = execute_pointwise_stage(
        graph,
        operator_types[operator_type](),
        nodes,
        epi_jump_fixed_support_declared=True,
        **execution_kwargs,
    )

    certificate = result.pointwise_epi_jump_certificate
    assert certificate is not None
    assert certificate.glyph == glyph.value
    assert certificate.nodes == tuple(graph.nodes)
    assert certificate.target_nodes == nodes
    assert tuple(certificate.state_before) == epi_before
    assert certificate.stage_schedule == "two_phase_jacobi"
    assert certificate.proposal_builder_replayed_from_declared_inputs
    assert certificate.runtime_epi_realization_certified


def test_executor_certificate_is_opt_in() -> None:
    from tnfr.operators.definitions import Expansion

    graph = _graph(Glyph.VAL)
    result = execute_pointwise_stage(graph, Expansion(), tuple(graph))

    assert result.pointwise_epi_jump_certificate is None


def test_executor_certificate_request_validates_domain_before_mutation() -> None:
    from tnfr.operators.definitions import Coherence, Expansion

    graph = _graph(Glyph.VAL)
    before = _plain_graph_state(graph)
    with pytest.raises(TypeError, match="bool or None"):
        execute_pointwise_stage(
            graph,
            Expansion(),
            tuple(graph),
            epi_jump_fixed_support_declared=1,
        )
    assert _plain_graph_state(graph) == before

    with pytest.raises(ValueError, match="supports AL, SHA, VAL, NUL, ZHIR and NAV"):
        execute_pointwise_stage(
            graph,
            Coherence(),
            tuple(graph),
            epi_jump_fixed_support_declared=True,
        )
    assert _plain_graph_state(graph) == before


def test_executor_certificate_failure_rolls_back_complete_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tnfr.operators.definitions import Emission

    graph = _graph(Glyph.AL)
    before = _plain_graph_state(graph)

    def fail_certificate(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("certificate failure")

    monkeypatch.setattr(
        pointwise,
        "certify_pointwise_epi_jump_realization",
        fail_certificate,
    )
    with pytest.raises(RuntimeError, match="certificate failure"):
        execute_pointwise_stage(
            graph,
            Emission(),
            tuple(graph),
            epi_jump_fixed_support_declared=True,
        )

    assert _plain_graph_state(graph) == before
