"""Runtime binding and finite composition of executed glyph-stage gains."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import networkx as nx
import pytest

from tnfr.operators.event_runtime import (
    ExecutedGlyphStage,
    ObservedRepresentedEPIScheduleComposition,
    execute_operator_event_schedule,
)
from tnfr.operators.event_timing import build_operator_event_schedule


def _graph() -> nx.Graph:
    graph = nx.path_graph(2)
    graph.graph.update(
        _t=0.0,
        DT_MIN=0.0,
        GAMMA={"type": "none"},
        EPI_MIN=-10.0,
        EPI_MAX=10.0,
        NAV_RANDOM=False,
        RANDOM_SEED=7,
        GLYPH_FACTORS={"NAV_eta": 0.25, "NAV_jitter": 0.0},
    )
    for node, epi, pressure in zip(
        graph,
        (0.25, -0.25),
        (-0.5, 0.5),
        strict=True,
    ):
        graph.nodes[node].update(
            EPI=epi,
            epi_kind="test",
            nu_f=1.0,
            theta=0.0,
            delta_nfr=pressure,
            latent=False,
            glyph_history=[],
        )
    return graph


def _refresh_pure_epi_pressure(graph: nx.Graph) -> None:
    left = float(graph.nodes[0]["EPI"])
    right = float(graph.nodes[1]["EPI"])
    graph.nodes[0]["delta_nfr"] = right - left
    graph.nodes[1]["delta_nfr"] = left - right


def test_stage_opt_in_binds_transition_to_both_adjacent_exact_flows() -> None:
    graph = _graph()
    graph.graph["compute_delta_nfr"] = _refresh_pure_epi_pressure
    schedule = build_operator_event_schedule(
        ("transition",),
        start_time=0.0,
        flow_durations=(0.125, 0.125),
    )

    result = execute_operator_event_schedule(
        graph,
        schedule,
        include_stage_certificates=True,
    )

    assert result.flow_certification_requested
    assert result.stage_certification_requested
    assert len(result.flow_interval_evidence) == 2
    assert len(result.glyph_stage_evidence) == 1
    stage = result.glyph_stage_evidence[0]
    assert isinstance(stage, ExecutedGlyphStage)
    assert stage.certificate_kind == "pointwise"
    assert stage.certificate_abstention_reason is None
    assert stage.endpoint_capture_complete
    assert stage.exact_runtime_endpoint_bound
    assert stage.exact_common_metric_bridge
    assert stage.represented_affine_gain_bound_at_observed_endpoint_certified
    assert stage.exact_energy_gain_upper_bound == Fraction(1)
    assert stage.pre_interval_index == 0
    assert stage.post_interval_index == 1
    assert stage.pre_flow_evidence is result.flow_interval_evidence[0]
    assert stage.post_flow_evidence is result.flow_interval_evidence[1]
    assert stage.pre_flow_endpoint_continuous is True
    assert stage.post_flow_endpoint_continuous is True
    assert stage.pre_flow_metric_compatible is True
    assert stage.post_flow_metric_compatible is True
    assert result.all_glyph_stages_represented_affine is True

    mixed = result.represented_epi_schedule_composition
    assert isinstance(mixed, ObservedRepresentedEPIScheduleComposition)
    assert mixed.represented_affine_composition_gain_certified
    assert mixed.exact_operation_energy_gain_factors == (
        Fraction(9, 16),
        Fraction(1),
        Fraction(9, 16),
    )
    assert mixed.exact_energy_gain_upper_bound == Fraction(81, 256)
    assert mixed.represented_map_global_disagreement_contraction_certified
    assert mixed.failed_conditions == ()
    assert not mixed.solver_accuracy_certified
    assert not mixed.full_multichannel_stability_certified
    assert not mixed.future_or_repeated_schedule_stability_certified
    assert "glyph_stage_evidence" not in graph.graph
    assert "represented_epi_schedule_composition" not in graph.graph


def test_stage_opt_in_is_disabled_without_changing_the_default_path() -> None:
    schedule = build_operator_event_schedule(
        ("transition",),
        start_time=0.0,
        flow_durations=(0.0, 0.0),
    )

    result = execute_operator_event_schedule(_graph(), schedule)

    assert not result.stage_certification_requested
    assert result.glyph_stage_evidence == ()
    assert result.represented_epi_schedule_composition is None
    assert result.all_glyph_stages_represented_affine is None


@pytest.mark.parametrize("value", [None, 0, 1.0, "yes"])
def test_stage_certificate_flag_requires_a_strict_bool(value: object) -> None:
    graph = _graph()
    schedule = build_operator_event_schedule(
        ("transition",),
        start_time=0.0,
        flow_durations=(0.0, 0.0),
    )

    with pytest.raises(TypeError, match="include_stage_certificates"):
        execute_operator_event_schedule(
            graph,
            schedule,
            include_stage_certificates=value,  # type: ignore[arg-type]
        )


def test_callback_epi_mutation_is_observed_but_blocks_stage_gain() -> None:
    graph = _graph()

    def mutate_epi(live_graph: nx.Graph) -> None:
        live_graph.nodes[0]["EPI"] = float(live_graph.nodes[0]["EPI"]) + 1.0

    graph.graph["compute_delta_nfr"] = mutate_epi
    schedule = build_operator_event_schedule(
        ("transition",),
        start_time=0.0,
        flow_durations=(0.0, 0.0),
    )

    result = execute_operator_event_schedule(
        graph,
        schedule,
        include_stage_certificates=True,
    )

    stage = result.glyph_stage_evidence[0]
    assert stage.endpoint_capture_complete
    assert not stage.exact_runtime_endpoint_bound
    assert not stage.represented_affine_gain_bound_at_observed_endpoint_certified
    assert stage.exact_energy_gain_upper_bound is None
    assert stage.certificate_abstention_reason is not None
    assert "glyph_certificate_endpoint_mismatch" in (
        stage.certificate_abstention_reason
    )
    assert result.pressure_refresh_callback_invocations == 1
    assert result.all_glyph_stages_represented_affine is False
    mixed = result.represented_epi_schedule_composition
    assert mixed is not None
    assert not mixed.represented_affine_composition_gain_certified
    assert "all_glyph_stages_represented_affine" in mixed.failed_conditions


def test_unsupported_glyphs_abstain_without_invalidating_the_schedule() -> None:
    graph = _graph()
    schedule = build_operator_event_schedule(
        ("emission", "coupling", "coherence", "silence"),
        start_time=0.0,
        flow_durations=(0.0, 0.0, 0.0, 0.0, 0.0),
    )

    result = execute_operator_event_schedule(
        graph,
        schedule,
        include_stage_certificates=True,
    )

    assert [stage.event.glyph.value for stage in result.glyph_stage_evidence] == [
        "AL",
        "UM",
        "IL",
        "SHA",
    ]
    coupling, coherence = result.glyph_stage_evidence[1:3]
    assert coupling.certificate is None
    assert coherence.certificate is None
    assert coupling.certificate_abstention_reason == (
        "epi_jump_certificate_unavailable_for_glyph:UM"
    )
    assert coherence.certificate_abstention_reason == (
        "epi_jump_certificate_unavailable_for_glyph:IL"
    )
    assert result.all_glyph_stages_represented_affine is False
    assert result.represented_epi_schedule_composition is not None
    composition = result.represented_epi_schedule_composition
    assert not composition.represented_affine_composition_gain_certified


def test_reception_certificate_is_bound_to_the_executor_proposal_endpoint() -> None:
    graph = _graph()
    schedule = build_operator_event_schedule(
        ("emission", "reception", "transition"),
        start_time=0.0,
        flow_durations=(0.0, 0.0, 0.0, 0.0),
    )

    result = execute_operator_event_schedule(
        graph,
        schedule,
        include_stage_certificates=True,
        suppress_birth_warnings=True,
    )

    stage = result.glyph_stage_evidence[1]
    assert stage.event.glyph.value == "EN"
    assert stage.certificate_kind == "neighbor"
    assert stage.certificate is not None
    assert stage.endpoint_capture_complete
    assert stage.exact_runtime_endpoint_bound
    step = stage.certificate.steps[0]
    assert tuple(float(value) for value in step.state_before) == stage.left.epi
    assert tuple(float(value) for value in step.runtime_accepted_state_after) == (
        stage.right.epi
    )


def test_zero_flow_transition_does_not_invent_strict_contraction() -> None:
    schedule = build_operator_event_schedule(
        ("transition",),
        start_time=0.0,
        flow_durations=(0.0, 0.0),
    )

    result = execute_operator_event_schedule(
        _graph(),
        schedule,
        include_stage_certificates=True,
    )

    mixed = result.represented_epi_schedule_composition
    assert mixed is not None
    assert mixed.represented_affine_composition_gain_certified
    assert mixed.exact_energy_gain_upper_bound == 1
    assert not mixed.represented_map_global_disagreement_contraction_certified


def test_stage_gain_claim_fails_closed_when_a_sealed_field_changes() -> None:
    schedule = build_operator_event_schedule(
        ("transition",),
        start_time=0.0,
        flow_durations=(0.0, 0.0),
    )
    result = execute_operator_event_schedule(
        _graph(),
        schedule,
        include_stage_certificates=True,
    )
    stage = result.glyph_stage_evidence[0]

    assert stage._proof_fields_are_intact()
    assert stage.represented_affine_gain_bound_at_observed_endpoint_certified

    altered = replace(stage, exact_runtime_endpoint_bound=False)

    assert not altered._proof_fields_are_intact()
    assert not altered.represented_affine_gain_bound_at_observed_endpoint_certified
    with pytest.raises(ValueError, match="proof fields are not intact"):
        replace(result, glyph_stage_evidence=(altered,))
