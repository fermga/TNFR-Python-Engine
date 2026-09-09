"""Executor-bound tests for event-local held-pressure ZHIR comparisons."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import networkx as nx
import pytest

from tnfr.operators.event_runtime import (
    ExecutedNodalFlowInterval,
    execute_operator_event_schedule,
)
from tnfr.operators.event_timing import build_operator_event_schedule
from tnfr.physics.event_refinement import (
    EventLocalZHIRHeldPressureComparison,
    compare_event_local_zhir_held_pressure_subdivision,
    observe_event_local_zhir_prejump,
)
from tnfr.physics.mutation_trigger import certify_mutation_trigger


def _graph(
    *,
    epi: tuple[float, ...],
    pressure: tuple[float, ...],
    substeps: int,
    duration: float = 1.0,
    start_time: float = 0.0,
) -> nx.Graph:
    graph = nx.empty_graph(len(epi))
    graph.graph.update(
        _t=start_time,
        DT_MIN=0.0 if substeps == 1 else duration / substeps,
        GAMMA={"type": "none"},
        EPI_MIN=-1.0e300,
        EPI_MAX=1.0e300,
    )
    for node, epi_value, pressure_value in zip(
        graph,
        epi,
        pressure,
        strict=True,
    ):
        graph.nodes[node].update(
            EPI=epi_value,
            nu_f=1.0,
            theta=0.0,
            delta_nfr=pressure_value,
        )
    return graph


def _executed_flow(
    *,
    epi: tuple[float, ...],
    pressure: tuple[float, ...],
    substeps: int,
    duration: float = 1.0,
    start_time: float = 0.0,
):
    graph = _graph(
        epi=epi,
        pressure=pressure,
        substeps=substeps,
        duration=duration,
        start_time=start_time,
    )
    schedule = build_operator_event_schedule(
        (),
        start_time=start_time,
        flow_durations=(duration,),
    )
    result = execute_operator_event_schedule(
        graph,
        schedule,
        include_flow_certificates=True,
        suppress_birth_warnings=True,
    )
    flow = result.flow_interval_evidence[0]
    assert flow.runtime_bound_binary64_held_pressure_interval_identified
    assert flow.resolved_substeps == substeps
    return flow


def _zhir_event(*, duration: float = 1.0, start_time: float = 0.0):
    schedule = build_operator_event_schedule(
        ("mutation",),
        start_time=start_time,
        flow_durations=(duration, 0.0),
    )
    return schedule.events[0]


def _observation(
    *,
    epi: tuple[float, ...],
    pressure: tuple[float, ...],
    substeps: int,
    xi: float,
    duration: float = 1.0,
    start_time: float = 0.0,
):
    return observe_event_local_zhir_prejump(
        _executed_flow(
            epi=epi,
            pressure=pressure,
            substeps=substeps,
            duration=duration,
            start_time=start_time,
        ),
        _zhir_event(duration=duration, start_time=start_time),
        xi=xi,
    )


def test_observation_separates_rational_secant_from_actual_zhir_arithmetic() -> None:
    observation = _observation(
        epi=(-2.0**-53,),
        pressure=(1.0000000000000002,),
        substeps=1,
        xi=1.0,
    )

    assert observation.pre_jump_observation_certified
    assert observation.exact_rational_physical_secants[0] > Fraction(1)
    assert observation.binary64_observed_gate_rates == (1.0,)
    assert observation.exact_binary64_observed_gate_rates == (Fraction(1),)
    assert observation.exact_rational_minus_binary64_gate_rates[0] > 0
    assert not observation.rational_and_binary64_gate_rates_equal
    assert observation.binary64_observed_strict_gate_decisions == (False,)
    assert "common schedule-execution provenance" in observation.scope


@pytest.mark.parametrize(
    ("epi", "pressure", "substeps", "xi"),
    [
        ((-2.0**-53,), (1.0000000000000002,), 1, 1.0),
        ((0.25, -0.25), (2.0, -2.0), 2, 1.0),
        ((1.0e16,), (4.0,), 4, 1.0),
    ],
)
def test_observation_reproduces_the_canonical_mutation_trigger(
    epi: tuple[float, ...],
    pressure: tuple[float, ...],
    substeps: int,
    xi: float,
) -> None:
    flow = _executed_flow(
        epi=epi,
        pressure=pressure,
        substeps=substeps,
    )
    observation = observe_event_local_zhir_prejump(
        flow,
        _zhir_event(),
        xi=xi,
    )
    certificate = flow.certificate
    assert certificate is not None

    trigger_results = tuple(
        certify_mutation_trigger(
            current_epi=right_epi,
            nu_f=nu_f,
            delta_nfr=delta_nfr,
            xi=xi,
            epi_time_history=(
                (flow.interval.start_time, left_epi),
                (flow.interval.end_time, right_epi),
            ),
        )
        for left_epi, right_epi, nu_f, delta_nfr in zip(
            certificate.left.epi,
            certificate.right.epi,
            certificate.left.nu_f,
            certificate.left.delta_nfr,
            strict=True,
        )
    )

    assert tuple(
        result.observed_depi_dt.hex()
        for result in trigger_results
        if result.observed_depi_dt is not None
    ) == tuple(rate.hex() for rate in observation.binary64_observed_gate_rates)
    assert tuple(
        result.threshold_gate_satisfied for result in trigger_results
    ) == observation.binary64_observed_strict_gate_decisions


def test_strict_margin_certifies_an_invariant_actual_gate_decision() -> None:
    baseline = _observation(
        epi=(0.25, -0.25),
        pressure=(2.0, -2.0),
        substeps=1,
        xi=1.0,
    )
    candidate = _observation(
        epi=(0.25, -0.25),
        pressure=(2.0, -2.0),
        substeps=2,
        xi=1.0,
    )

    result = compare_event_local_zhir_held_pressure_subdivision(
        baseline,
        candidate,
    )

    assert isinstance(result, EventLocalZHIRHeldPressureComparison)
    assert result.comparison_kind == "internal_held_pressure_subdivision"
    assert result.baseline_substeps == 1
    assert result.candidate_substeps == 2
    assert result.observed_strict_gate_decisions_agree
    assert result.strict_threshold_separation
    assert result.event_local_gate_invariance_certified
    assert not result.physical_pressure_reevaluated_partition_established
    assert not result.diffusion_modal_decisions_applicable
    assert not result.solver_accuracy_certified
    assert not result.solver_order_certified
    assert not result.physical_refinement_equivalence_certified
    assert not result.future_or_repeated_behavior_certified
    assert not result.u4_readiness_certified
    assert not result.adaptive_policy_certified


def test_threshold_contact_abstains_even_when_observed_decisions_agree() -> None:
    baseline = _observation(
        epi=(1.0e16,),
        pressure=(4.0,),
        substeps=1,
        xi=8.0,
    )
    candidate = _observation(
        epi=(1.0e16,),
        pressure=(4.0,),
        substeps=4,
        xi=8.0,
    )

    result = compare_event_local_zhir_held_pressure_subdivision(
        baseline,
        candidate,
    )

    assert result.baseline_binary64_observed_gate_rates == (4.0,)
    assert result.candidate_binary64_observed_gate_rates == (0.0,)
    assert result.baseline_observed_strict_gate_decisions == (False,)
    assert result.candidate_observed_strict_gate_decisions == (False,)
    assert result.observed_strict_gate_decisions_agree
    assert result.exact_binary64_gate_rate_difference_bounds == (Fraction(4),)
    assert result.baseline_exact_threshold_distances == (Fraction(4),)
    assert not result.strict_threshold_separation
    assert not result.event_local_gate_invariance_certified


def test_different_actual_gate_decisions_cannot_be_promoted() -> None:
    baseline = _observation(
        epi=(1.0e16,),
        pressure=(4.0,),
        substeps=1,
        xi=1.0,
    )
    candidate = _observation(
        epi=(1.0e16,),
        pressure=(4.0,),
        substeps=4,
        xi=1.0,
    )

    result = compare_event_local_zhir_held_pressure_subdivision(
        baseline,
        candidate,
    )

    assert result.baseline_observed_strict_gate_decisions == (True,)
    assert result.candidate_observed_strict_gate_decisions == (False,)
    assert not result.observed_strict_gate_decisions_agree
    assert not result.event_local_gate_invariance_certified


def test_comparison_requires_different_substep_counts() -> None:
    first = _observation(epi=(0.0,), pressure=(2.0,), substeps=1, xi=1.0)
    second = _observation(epi=(0.0,), pressure=(2.0,), substeps=1, xi=1.0)

    with pytest.raises(ValueError, match="different_substeps"):
        compare_event_local_zhir_held_pressure_subdivision(first, second)


def test_comparison_rejects_different_initial_states() -> None:
    baseline = _observation(epi=(0.0,), pressure=(2.0,), substeps=1, xi=1.0)
    candidate = _observation(epi=(1.0,), pressure=(2.0,), substeps=2, xi=1.0)

    with pytest.raises(ValueError, match="exact_initial_epi"):
        compare_event_local_zhir_held_pressure_subdivision(baseline, candidate)


def test_observation_rejects_a_jump_at_a_different_coordinate() -> None:
    flow = _executed_flow(epi=(0.0,), pressure=(2.0,), substeps=1)

    with pytest.raises(ValueError, match="flow endpoint"):
        observe_event_local_zhir_prejump(
            flow,
            _zhir_event(duration=1.0, start_time=1.0),
            xi=1.0,
        )


def test_observation_rejects_unsealed_flow_evidence() -> None:
    flow = _executed_flow(epi=(0.0,), pressure=(2.0,), substeps=1)
    forged = ExecutedNodalFlowInterval(
        interval=flow.interval,
        certificate=flow.certificate,
        abstention_reason=flow.abstention_reason,
        integrator_name=flow.integrator_name,
        integrator_provenance_certified=flow.integrator_provenance_certified,
        resolved_method=flow.resolved_method,
        resolved_substeps=flow.resolved_substeps,
        gamma_is_none=flow.gamma_is_none,
        clipping_applied=flow.clipping_applied,
        extended_dynamics_requested=flow.extended_dynamics_requested,
    )

    assert not forged._proof_fields_are_intact()
    with pytest.raises(ValueError, match="unsealed"):
        observe_event_local_zhir_prejump(forged, _zhir_event(), xi=1.0)


def test_observation_rejects_empty_support_and_negative_xi() -> None:
    empty_flow = _executed_flow(epi=(), pressure=(), substeps=1)
    with pytest.raises(ValueError, match="at least one node"):
        observe_event_local_zhir_prejump(empty_flow, _zhir_event(), xi=1.0)

    flow = _executed_flow(epi=(0.0,), pressure=(2.0,), substeps=1)
    with pytest.raises(ValueError, match="nonnegative"):
        observe_event_local_zhir_prejump(flow, _zhir_event(), xi=-1.0)


def test_observation_and_comparison_seals_fail_closed_after_tampering() -> None:
    baseline = _observation(epi=(0.0,), pressure=(2.0,), substeps=1, xi=0.0)
    candidate = _observation(epi=(0.0,), pressure=(2.0,), substeps=2, xi=0.0)
    comparison = compare_event_local_zhir_held_pressure_subdivision(
        baseline,
        candidate,
    )

    signed_zero_tamper = replace(baseline, xi=-0.0)
    assert not signed_zero_tamper.pre_jump_observation_certified
    with pytest.raises(ValueError, match="tampered"):
        compare_event_local_zhir_held_pressure_subdivision(
            signed_zero_tamper,
            candidate,
        )

    supporting_field_tamper = replace(
        baseline,
        exact_pressure=(Fraction(3),),
    )
    assert not supporting_field_tamper.pre_jump_observation_certified

    decision_promotion = replace(
        baseline,
        binary64_observed_strict_gate_decisions=(False,),
    )
    assert not decision_promotion.pre_jump_observation_certified
    with pytest.raises(ValueError, match="tampered"):
        compare_event_local_zhir_held_pressure_subdivision(
            decision_promotion,
            candidate,
        )

    forged_claim = replace(
        comparison,
        _event_local_gate_invariance_certified=False,
    )
    assert not forged_claim.event_local_gate_invariance_certified

    rejected = compare_event_local_zhir_held_pressure_subdivision(
        _observation(
            epi=(1.0e16,),
            pressure=(4.0,),
            substeps=1,
            xi=1.0,
        ),
        _observation(
            epi=(1.0e16,),
            pressure=(4.0,),
            substeps=4,
            xi=1.0,
        ),
    )
    assert not rejected.event_local_gate_invariance_certified
    promoted_claim = replace(
        rejected,
        _event_local_gate_invariance_certified=True,
    )
    assert not promoted_claim.event_local_gate_invariance_certified
