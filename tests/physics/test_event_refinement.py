"""Executor-bound tests for held-pressure and physical ZHIR comparisons."""

from __future__ import annotations

from dataclasses import asdict, fields, replace
from fractions import Fraction

import networkx as nx
import pytest

from tnfr.dynamics.integrators import AbstractIntegrator
from tnfr.errors import TNFRValueError
from tnfr.operators.event_runtime import (
    ExecutedNodalFlowInterval,
    execute_operator_event_schedule,
)
from tnfr.operators.event_timing import (
    build_operator_event_schedule,
    build_physical_flow_partition,
)
from tnfr.physics.event_refinement import (
    ExecutedEventLocalZHIRPhysicalPrejumpObservation,
    EventLocalZHIRHeldPressureComparison,
    EventLocalZHIRPhysicalRefinementComparison,
    compare_event_local_zhir_held_pressure_subdivision,
    compare_event_local_zhir_physical_refinement,
    observe_executed_event_local_zhir_physical_prejump,
    observe_event_local_zhir_physical_prejump,
    observe_event_local_zhir_prejump,
)
from tnfr.physics.mutation_trigger import certify_mutation_trigger


def _assert_slotless_claim_boundaries(
    value: object,
    false_claims: tuple[str, ...],
) -> None:
    """Reject constructor and low-level mutation of fixed negative claims."""

    dataclass_fields = {item.name for item in fields(value)}
    slots = set(getattr(type(value), "__slots__", ()))
    for name in false_claims:
        assert getattr(value, name) is False
        assert name not in dataclass_fields
        assert name not in slots
        assert isinstance(type(value).__dict__.get(name), property)
        with pytest.raises(AttributeError):
            object.__setattr__(value, name, True)
        with pytest.raises(TypeError):
            replace(value, **{name: True})

    original_scope = getattr(value, "scope")
    assert "scope" not in dataclass_fields
    assert "scope" not in slots
    assert isinstance(type(value).__dict__.get("scope"), property)
    with pytest.raises(AttributeError):
        object.__setattr__(value, "scope", "forged broader theorem")
    with pytest.raises(TypeError):
        replace(value, scope="forged broader theorem")
    assert getattr(value, "scope") == original_scope


def _assert_historical_claim_fields(
    value: object,
    false_claims: tuple[str, ...],
) -> None:
    """Preserve published fields while making raw claim tampering fail closed."""

    field_names = {item.name for item in fields(value)}
    slots = set(getattr(type(value), "__slots__", ()))
    serialized = asdict(value)
    representation = repr(value)
    expected = {name: False for name in false_claims}
    expected["scope"] = getattr(value, "scope")

    assert value._proof_fields_are_intact()
    for name, expected_value in expected.items():
        assert name in field_names
        assert name in slots
        assert serialized[name] == expected_value
        assert f"{name}=" in representation
        with pytest.raises(ValueError, match="init=False"):
            replace(value, **{name: "forged"})

        raw = object.__getattribute__(value, name)
        promoted = True if expected_value is False else "forged broader theorem"
        object.__setattr__(value, name, promoted)
        assert getattr(value, name) == expected_value
        assert not value._proof_fields_are_intact()
        object.__setattr__(value, name, raw)
        assert value._proof_fields_are_intact()


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
        integrator_provenance_certified=(
            flow.integrator_provenance_certified
        ),
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


def test_refinement_proof_stamps_reject_hostile_equality_without_dispatch() -> None:
    calls: list[str] = []

    class HostileEquality:
        def __eq__(self, other: object) -> bool:
            del other
            calls.append("equality dispatched")
            raise SystemExit("proof comparison must fail closed")

    baseline = _observation(epi=(0.0,), pressure=(2.0,), substeps=1, xi=0.0)
    candidate = _observation(epi=(0.0,), pressure=(2.0,), substeps=2, xi=0.0)
    comparison = compare_event_local_zhir_held_pressure_subdivision(
        baseline,
        candidate,
    )

    for record in (baseline, comparison):
        object.__setattr__(
            record,
            "_proof_stamp",
            ("injected", HostileEquality()),
        )
        assert not record._proof_fields_are_intact()

    assert calls == []


def _pure_epi_k2_graph() -> nx.Graph:
    graph = nx.path_graph(2)
    graph.graph.update(
        _t=0.0,
        DT_MIN=0.0,
        GAMMA={"type": "none"},
        EPI_MIN=-1.0e300,
        EPI_MAX=1.0e300,
    )
    graph.nodes[0].update(EPI=1.0, nu_f=1.0, theta=0.0, delta_nfr=-2.0)
    graph.nodes[1].update(EPI=-1.0, nu_f=1.0, theta=0.0, delta_nfr=2.0)

    def refresh_pressure(live_graph: nx.Graph) -> None:
        left = float(live_graph.nodes[0]["EPI"])
        right = float(live_graph.nodes[1]["EPI"])
        live_graph.nodes[0]["delta_nfr"] = right - left
        live_graph.nodes[1]["delta_nfr"] = left - right

    graph.graph["compute_delta_nfr"] = refresh_pressure
    return graph


def _executed_physical_zhir_result(
    *,
    include_stage_certificates: bool = True,
    include_physical_partition: bool = True,
):
    """Execute one accepted ZHIR after a two-segment physical flow."""

    graph = nx.path_graph(2)
    graph.graph.update(
        _t=0.0,
        DT_MIN=0.0,
        GAMMA={"type": "none"},
        EPI_MIN=-1.0e300,
        EPI_MAX=1.0e300,
        ZHIR_THRESHOLD_XI=0.0,
        DNFR_WEIGHTS={
            "phase": 0.0,
            "epi": 1.0,
            "vf": 0.0,
            "topo": 0.0,
        },
    )
    for node, epi in zip(graph, (1.0, -1.0), strict=True):
        graph.nodes[node].update(
            EPI=epi,
            epi_kind="test",
            nu_f=1.0,
            theta=0.0,
            delta_nfr=0.0,
            glyph_history=[],
        )

    def refresh(live_graph: nx.Graph) -> None:
        pressure = 1.0 if float(live_graph.graph["_t"]) < 0.25 else 2.0
        for node in live_graph:
            live_graph.nodes[node]["delta_nfr"] = pressure

    graph.graph["compute_delta_nfr"] = refresh
    schedule = build_operator_event_schedule(
        (
            "emission",
            "coherence",
            "dissonance",
            "mutation",
            "coherence",
            "silence",
        ),
        start_time=0.0,
        flow_durations=(0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0),
    )
    partitions = (
        (
            build_physical_flow_partition(
                schedule.intervals[3],
                (0.25, 0.25),
            ),
        )
        if include_physical_partition
        else ()
    )
    return execute_operator_event_schedule(
        graph,
        schedule,
        include_stage_certificates=include_stage_certificates,
        physical_flow_partitions=partitions,
        suppress_birth_warnings=True,
    )


def test_executed_physical_zhir_observation_binds_one_real_execution() -> None:
    result = _executed_physical_zhir_result()

    observation = observe_executed_event_local_zhir_physical_prejump(
        result,
        event_index=3,
    )

    assert type(observation) is ExecutedEventLocalZHIRPhysicalPrejumpObservation
    assert observation.common_execution_provenance_certified
    assert observation.execution_result is result
    assert observation.scheduled_event is result.schedule.events[3]
    assert observation.executed_event is result.events[3]
    assert observation.glyph_stage is result.glyph_stage_evidence[3]
    assert observation.partition_evidence is (
        result.physical_flow_partition_evidence[0]
    )
    assert observation.physical_observation.partition_evidence is (
        observation.partition_evidence
    )
    assert observation.nodes is result.target_nodes
    assert observation.mutation_decision_observations is (
        observation.glyph_stage.mutation_decision_observations
    )
    assert all(
        certificate is decision.trigger_certificate
        for certificate, decision in zip(
            observation.trigger_certificates,
            observation.mutation_decision_observations,
            strict=True,
        )
    )
    assert observation.physical_observation.terminal_binary64_observed_gate_rates == (
        2.0,
        2.0,
    )
    assert all(
        certificate.threshold_gate_satisfied
        for certificate in observation.trigger_certificates
    )
    assert not observation.solver_accuracy_certified
    assert not observation.solver_order_certified
    assert not observation.mesh_convergence_certified
    assert not observation.u4_readiness_certified
    assert not observation.adaptive_policy_certified
    assert not observation.future_or_repeated_behavior_certified
    assert "one executor-sealed schedule execution" in observation.scope


def test_executed_physical_zhir_observation_requires_both_evidence_layers() -> None:
    without_stages = _executed_physical_zhir_result(
        include_stage_certificates=False,
    )
    without_partition = _executed_physical_zhir_result(
        include_physical_partition=False,
    )

    with pytest.raises(ValueError, match="stage certificates"):
        observe_executed_event_local_zhir_physical_prejump(
            without_stages,
            event_index=3,
        )
    with pytest.raises(ValueError, match="immediately preceding physical"):
        observe_executed_event_local_zhir_physical_prejump(
            without_partition,
            event_index=3,
        )
    with pytest.raises(ValueError, match="canonical ZHIR"):
        observe_executed_event_local_zhir_physical_prejump(
            without_partition,
            event_index=2,
        )


def test_executed_physical_zhir_observation_rejects_reordered_links() -> None:
    result = _executed_physical_zhir_result()
    observation = observe_executed_event_local_zhir_physical_prejump(
        result,
        event_index=3,
    )

    assert not replace(
        observation,
        mutation_decision_observations=tuple(
            reversed(observation.mutation_decision_observations)
        ),
    ).common_execution_provenance_certified
    assert not replace(
        observation,
        trigger_certificates=tuple(reversed(observation.trigger_certificates)),
    ).common_execution_provenance_certified
    assert not replace(
        observation,
        glyph_stage=result.glyph_stage_evidence[2],
    ).common_execution_provenance_certified


def test_executed_physical_zhir_observation_rejects_tampered_execution() -> None:
    result = _executed_physical_zhir_result()
    partition = result.physical_flow_partition_evidence[0]
    object.__setattr__(
        result,
        "physical_flow_partition_evidence",
        (partition, partition),
    )

    with pytest.raises(ValueError, match="tampered"):
        observe_executed_event_local_zhir_physical_prejump(
            result,
            event_index=3,
        )


def test_executed_physical_zhir_claims_are_slotless_read_only() -> None:
    observation = observe_executed_event_local_zhir_physical_prejump(
        _executed_physical_zhir_result(),
        event_index=3,
    )

    _assert_slotless_claim_boundaries(
        observation,
        (
            "solver_accuracy_certified",
            "solver_order_certified",
            "mesh_convergence_certified",
            "u4_readiness_certified",
            "adaptive_policy_certified",
            "future_or_repeated_behavior_certified",
        ),
    )
    with pytest.raises(AttributeError):
        object.__setattr__(
            observation,
            "common_execution_provenance_certified",
            False,
        )


def test_executed_physical_zhir_observer_is_exported_by_its_module() -> None:
    import tnfr.physics.event_refinement as event_refinement

    assert (
        event_refinement.ExecutedEventLocalZHIRPhysicalPrejumpObservation
        is ExecutedEventLocalZHIRPhysicalPrejumpObservation
    )
    assert (
        event_refinement.observe_executed_event_local_zhir_physical_prejump
        is observe_executed_event_local_zhir_physical_prejump
    )
    assert "ExecutedEventLocalZHIRPhysicalPrejumpObservation" in (
        event_refinement.__all__
    )
    assert "observe_executed_event_local_zhir_physical_prejump" in (
        event_refinement.__all__
    )


def _pure_epi_path3_graph() -> nx.Graph:
    graph = nx.path_graph(3)
    graph.graph.update(
        _t=0.0,
        DT_MIN=0.0,
        GAMMA={"type": "none"},
        EPI_MIN=-1.0e300,
        EPI_MAX=1.0e300,
    )
    for node, epi, capacity in zip(
        graph,
        (1.0, 0.0, -1.0),
        (1.0, 2.0, 3.0),
        strict=True,
    ):
        graph.nodes[node].update(
            EPI=epi,
            nu_f=capacity,
            theta=0.0,
            delta_nfr=0.0,
        )

    def refresh_pressure(live_graph: nx.Graph) -> None:
        epi = {
            node: float(live_graph.nodes[node]["EPI"])
            for node in live_graph
        }
        for node in live_graph:
            neighbors = tuple(live_graph.neighbors(node))
            mean = sum(epi[neighbor] for neighbor in neighbors) / len(neighbors)
            live_graph.nodes[node]["delta_nfr"] = mean - epi[node]

    graph.graph["compute_delta_nfr"] = refresh_pressure
    refresh_pressure(graph)
    return graph


class _HeldEulerWithOptionalCapacityPermutation(AbstractIntegrator):
    """Custom held-pressure Euler flow outside trusted runtime provenance."""

    def __init__(self, *, permute_capacity_after_first_call: bool) -> None:
        self.calls = 0
        self.permute_capacity_after_first_call = permute_capacity_after_first_call

    def integrate(
        self,
        graph,
        *,
        dt,
        t,
        method,
        n_jobs,
    ) -> None:
        del method, n_jobs
        self.calls += 1
        duration = float(dt)
        for node in graph:
            data = graph.nodes[node]
            data["EPI"] = float(data["EPI"]) + duration * (
                float(data["nu_f"]) * float(data["delta_nfr"])
            )
        if self.permute_capacity_after_first_call and self.calls == 1:
            capacities = tuple(float(graph.nodes[node]["nu_f"]) for node in graph)
            for node, capacity in zip(graph, reversed(capacities), strict=True):
                graph.nodes[node]["nu_f"] = capacity
        graph.graph["_t"] = float(t) + duration


def _physical_k2_pair(
    *,
    xi: float,
    duration: float = 0.5,
    segment_durations: tuple[float, ...] = (0.25, 0.25),
    dt_min: float = 0.0,
):
    baseline_graph = _pure_epi_k2_graph()
    physical_graph = _pure_epi_k2_graph()
    baseline_graph.graph["DT_MIN"] = dt_min
    physical_graph.graph["DT_MIN"] = dt_min
    flow_schedule = build_operator_event_schedule(
        (),
        start_time=0.0,
        flow_durations=(duration,),
    )
    partition = build_physical_flow_partition(
        flow_schedule.intervals[0],
        segment_durations,
    )
    baseline_result = execute_operator_event_schedule(
        baseline_graph,
        flow_schedule,
        include_flow_certificates=True,
        suppress_birth_warnings=True,
    )
    physical_result = execute_operator_event_schedule(
        physical_graph,
        flow_schedule,
        physical_flow_partitions=(partition,),
        suppress_birth_warnings=True,
    )
    event = _zhir_event(duration=duration)
    baseline = observe_event_local_zhir_prejump(
        baseline_result.flow_interval_evidence[0],
        event,
        xi=xi,
    )
    physical = observe_event_local_zhir_physical_prejump(
        physical_result.physical_flow_partition_evidence[0],
        event,
        xi=xi,
    )
    return baseline, physical


def test_physical_refinement_separates_actual_and_parent_zhir_windows() -> None:
    baseline, physical = _physical_k2_pair(xi=1.25)

    assert physical.physical_pre_jump_observation_certified
    assert physical.binary64_terminal_epi == (0.25, -0.25)
    assert physical.exact_terminal_epi == (Fraction(1, 4), Fraction(-1, 4))
    assert physical.whole_parent_exact_rational_physical_secants == (
        Fraction(-3, 2),
        Fraction(3, 2),
    )
    assert physical.terminal_exact_rational_physical_secants == (
        Fraction(-1),
        Fraction(1),
    )
    assert physical.whole_parent_binary64_gate_rates == (-1.5, 1.5)
    assert physical.terminal_binary64_observed_gate_rates == (-1.0, 1.0)
    assert physical.whole_parent_strict_gate_decisions == (False, True)
    assert physical.terminal_observed_strict_gate_decisions == (False, False)
    assert not physical.terminal_window_equals_parent_horizon
    assert not physical.common_jump_execution_certified
    assert not physical.solver_accuracy_certified
    assert not physical.solver_order_certified
    assert not physical.mesh_convergence_certified
    assert not physical.future_or_repeated_behavior_certified
    assert not physical.u4_readiness_certified

    result = compare_event_local_zhir_physical_refinement(baseline, physical)

    assert isinstance(result, EventLocalZHIRPhysicalRefinementComparison)
    assert result.comparison_kind == "pressure_reevaluated_physical_partition"
    assert result.baseline_binary64_gate_rates == (-2.0, 2.0)
    assert result.exact_whole_parent_gate_rate_differences == (
        Fraction(1, 2),
        Fraction(-1, 2),
    )
    assert result.exact_terminal_gate_rate_differences == (
        Fraction(1),
        Fraction(-1),
    )
    assert result.whole_parent_gate_decisions_agree
    assert result.fixed_horizon_gate_agreement_certified
    assert not result.terminal_gate_decisions_agree
    assert not result.observation_windows_equal
    assert not result.actual_terminal_gate_invariance_certified
    assert result.modal_comparison_applicable
    assert result.baseline_runtime_substeps == 1
    assert result.baseline_held_interval_duration == 0.5
    assert result.common_binary64_modal_decay_rates == (2.0,)
    assert result.baseline_held_interval_binary64_modal_multipliers == (0.0,)
    assert result.baseline_held_interval_modal_stable is True
    assert result.physical_segment_modal_decisions == (True, True)
    assert result.physical_refined_composite_binary64_modal_multipliers == (0.25,)
    assert result.physical_refined_composite_modal_stable is True
    assert result.modal_stability_decisions_agree is True
    assert not result.modal_equivalence_certified
    assert not result.solver_accuracy_certified
    assert not result.solver_order_certified
    assert not result.mesh_convergence_certified
    assert not result.future_or_repeated_behavior_certified
    assert not result.u4_readiness_certified
    assert not result.adaptive_policy_certified


def test_physical_refinement_can_change_the_same_horizon_gate_decision() -> None:
    baseline, physical = _physical_k2_pair(xi=1.75)

    result = compare_event_local_zhir_physical_refinement(baseline, physical)

    assert baseline.binary64_observed_strict_gate_decisions == (False, True)
    assert physical.whole_parent_strict_gate_decisions == (False, False)
    assert not result.whole_parent_gate_decisions_agree
    assert not result.fixed_horizon_gate_agreement_certified
    assert not result.actual_terminal_gate_invariance_certified


def test_physical_refinement_can_stabilize_the_held_interval_modal_map() -> None:
    baseline, physical = _physical_k2_pair(
        xi=0.0,
        duration=1.25,
        segment_durations=(0.625, 0.625),
    )

    result = compare_event_local_zhir_physical_refinement(baseline, physical)

    assert baseline.binary64_endpoint_epi == (-1.5, 1.5)
    assert physical.binary64_terminal_epi == (0.0625, -0.0625)
    assert result.modal_comparison_applicable
    assert result.baseline_held_interval_binary64_modal_multipliers == (-1.5,)
    assert result.baseline_held_interval_binary64_maximum_modal_factor == 1.5
    assert result.baseline_held_interval_modal_stable is False
    assert result.physical_segment_modal_decisions == (True, True)
    assert result.physical_refined_composite_binary64_modal_multipliers == (
        0.0625,
    )
    assert result.physical_refined_composite_binary64_maximum_modal_factor == 0.0625
    assert result.physical_refined_composite_modal_stable is True
    assert result.modal_stability_decisions_agree is False
    assert not result.modal_equivalence_certified


def test_held_modal_factor_uses_parent_duration_across_internal_substeps() -> None:
    baseline, physical = _physical_k2_pair(
        xi=0.0,
        dt_min=0.125,
    )

    result = compare_event_local_zhir_physical_refinement(baseline, physical)

    assert baseline.substeps == 4
    assert tuple(
        flow.resolved_substeps
        for flow in physical.partition_evidence.segment_flow_evidence
    ) == (2, 2)
    assert result.modal_comparison_applicable
    assert result.baseline_held_interval_binary64_modal_multipliers == (0.0,)
    assert result.physical_refined_composite_binary64_modal_multipliers == (
        0.25,
    )


def test_physical_modal_comparison_abstains_outside_pure_epi_pressure() -> None:
    baseline_graph = _pure_epi_k2_graph()
    physical_graph = _pure_epi_k2_graph()

    def constant_pressure(live_graph: nx.Graph) -> None:
        live_graph.nodes[0]["delta_nfr"] = -1.0
        live_graph.nodes[1]["delta_nfr"] = 1.0

    for graph in (baseline_graph, physical_graph):
        graph.nodes[0]["delta_nfr"] = -1.0
        graph.nodes[1]["delta_nfr"] = 1.0
        graph.graph["compute_delta_nfr"] = constant_pressure
    schedule = build_operator_event_schedule(
        (), start_time=0.0, flow_durations=(0.5,)
    )
    partition = build_physical_flow_partition(
        schedule.intervals[0], (0.25, 0.25)
    )
    baseline_result = execute_operator_event_schedule(
        baseline_graph,
        schedule,
        include_flow_certificates=True,
        suppress_birth_warnings=True,
    )
    physical_result = execute_operator_event_schedule(
        physical_graph,
        schedule,
        physical_flow_partitions=(partition,),
        suppress_birth_warnings=True,
    )
    event = _zhir_event(duration=0.5)
    baseline = observe_event_local_zhir_prejump(
        baseline_result.flow_interval_evidence[0], event, xi=0.0
    )
    physical = observe_event_local_zhir_physical_prejump(
        physical_result.physical_flow_partition_evidence[0], event, xi=0.0
    )

    result = compare_event_local_zhir_physical_refinement(baseline, physical)

    assert not result.modal_comparison_applicable
    assert result.modal_abstention_reason == (
        "physical_segment_modal_diagnostics_unavailable"
    )
    assert result.baseline_held_interval_modal_stable is None
    assert result.physical_refined_composite_modal_stable is None
    assert result.modal_stability_decisions_agree is None


def _path3_custom_physical_comparison(*, permute_capacity: bool):
    duration = 0.25
    baseline_graph = _pure_epi_path3_graph()
    physical_graph = _pure_epi_path3_graph()
    physical_graph.graph["integrator"] = (
        _HeldEulerWithOptionalCapacityPermutation(
            permute_capacity_after_first_call=permute_capacity,
        )
    )
    schedule = build_operator_event_schedule(
        (),
        start_time=0.0,
        flow_durations=(duration,),
    )
    partition = build_physical_flow_partition(
        schedule.intervals[0],
        (0.125, 0.125),
    )
    baseline_result = execute_operator_event_schedule(
        baseline_graph,
        schedule,
        include_flow_certificates=True,
        suppress_birth_warnings=True,
    )
    physical_result = execute_operator_event_schedule(
        physical_graph,
        schedule,
        physical_flow_partitions=(partition,),
        suppress_birth_warnings=True,
    )
    event = _zhir_event(duration=duration)
    baseline = observe_event_local_zhir_prejump(
        baseline_result.flow_interval_evidence[0],
        event,
        xi=0.0,
    )
    physical = observe_event_local_zhir_physical_prejump(
        physical_result.physical_flow_partition_evidence[0],
        event,
        xi=0.0,
    )
    return physical, compare_event_local_zhir_physical_refinement(
        baseline,
        physical,
    )


def test_physical_execution_rejects_and_rolls_back_capacity_permutation() -> None:
    graph = _pure_epi_path3_graph()
    integrator = _HeldEulerWithOptionalCapacityPermutation(
        permute_capacity_after_first_call=True,
    )
    graph.graph["integrator"] = integrator
    schedule = build_operator_event_schedule(
        (),
        start_time=0.0,
        flow_durations=(0.25,),
    )
    partition = build_physical_flow_partition(
        schedule.intervals[0],
        (0.125, 0.125),
    )
    before = tuple(
        (
            float(graph.nodes[node]["EPI"]),
            float(graph.nodes[node]["nu_f"]),
            float(graph.nodes[node]["delta_nfr"]),
        )
        for node in graph
    )

    with pytest.raises(TNFRValueError, match="may update only EPI"):
        execute_operator_event_schedule(
            graph,
            schedule,
            physical_flow_partitions=(partition,),
            suppress_birth_warnings=True,
        )

    assert graph.graph["_t"] == 0.0
    assert integrator.calls == 0
    assert tuple(
        (
            float(graph.nodes[node]["EPI"]),
            float(graph.nodes[node]["nu_f"]),
            float(graph.nodes[node]["delta_nfr"]),
        )
        for node in graph
    ) == before


def test_physical_modal_comparison_requires_trusted_segment_runtime() -> None:
    physical, result = _path3_custom_physical_comparison(
        permute_capacity=False,
    )

    assert all(
        not flow.integrator_provenance_certified
        for flow in physical.partition_evidence.segment_flow_evidence
    )
    assert not result.modal_comparison_applicable
    assert result.modal_abstention_reason == (
        "physical_segment_held_pressure_runtime_unidentified"
    )


def test_physical_observation_and_comparison_seals_fail_closed() -> None:
    baseline, physical = _physical_k2_pair(xi=1.75)
    result = compare_event_local_zhir_physical_refinement(baseline, physical)

    altered_observation = replace(
        physical,
        terminal_observed_strict_gate_decisions=(False, True),
    )
    assert not altered_observation.physical_pre_jump_observation_certified
    with pytest.raises(ValueError, match="tampered"):
        compare_event_local_zhir_physical_refinement(
            baseline,
            altered_observation,
        )

    promoted = replace(result, _fixed_horizon_gate_agreement_certified=True)
    assert not promoted.fixed_horizon_gate_agreement_certified
    assert not promoted.actual_terminal_gate_invariance_certified


def test_refinement_claim_boundaries_preserve_compatible_fixed_surfaces() -> None:
    held_baseline = _observation(
        epi=(0.25, -0.25),
        pressure=(2.0, -2.0),
        substeps=1,
        xi=1.0,
    )
    held_candidate = _observation(
        epi=(0.25, -0.25),
        pressure=(2.0, -2.0),
        substeps=2,
        xi=1.0,
    )
    held_comparison = compare_event_local_zhir_held_pressure_subdivision(
        held_baseline,
        held_candidate,
    )
    physical_baseline, physical = _physical_k2_pair(xi=1.25)
    physical_comparison = compare_event_local_zhir_physical_refinement(
        physical_baseline,
        physical,
    )

    _assert_historical_claim_fields(held_baseline, ())
    _assert_historical_claim_fields(
        held_comparison,
        (
            "physical_pressure_reevaluated_partition_established",
            "diffusion_modal_decisions_applicable",
            "diffusion_modal_decisions_established",
            "solver_accuracy_certified",
            "solver_order_certified",
            "physical_refinement_equivalence_certified",
            "future_or_repeated_behavior_certified",
            "u4_readiness_certified",
            "adaptive_policy_certified",
        ),
    )
    _assert_slotless_claim_boundaries(
        physical,
        (
            "common_jump_execution_certified",
            "u4_readiness_certified",
            "solver_accuracy_certified",
            "solver_order_certified",
            "mesh_convergence_certified",
            "future_or_repeated_behavior_certified",
        ),
    )
    _assert_slotless_claim_boundaries(
        physical_comparison,
        (
            "modal_equivalence_certified",
            "common_execution_provenance_certified",
            "solver_accuracy_certified",
            "solver_order_certified",
            "mesh_convergence_certified",
            "future_or_repeated_behavior_certified",
            "u4_readiness_certified",
            "adaptive_policy_certified",
        ),
    )

    observation_fields = tuple(item.name for item in fields(held_baseline))
    observation_tail = observation_fields[
        observation_fields.index("endpoint_subtraction_matches_declared_duration") :
    ]
    assert observation_tail == (
        "endpoint_subtraction_matches_declared_duration",
        "scope",
        "_proof_stamp",
        "binary64_initial_epi",
        "binary64_endpoint_epi",
        "binary64_capacity",
        "binary64_pressure",
    )
    comparison_fields = tuple(item.name for item in fields(held_comparison))
    assert comparison_fields[-11:] == (
        "physical_pressure_reevaluated_partition_established",
        "diffusion_modal_decisions_applicable",
        "diffusion_modal_decisions_established",
        "solver_accuracy_certified",
        "solver_order_certified",
        "physical_refinement_equivalence_certified",
        "future_or_repeated_behavior_certified",
        "u4_readiness_certified",
        "adaptive_policy_certified",
        "scope",
        "_proof_stamp",
    )


def test_held_observation_preserves_signed_zero_in_its_comparison_inputs() -> None:
    observation = _observation(
        epi=(-0.0,),
        pressure=(-0.0,),
        substeps=1,
        xi=0.0,
    )

    assert observation.binary64_initial_epi[0].hex() == (-0.0).hex()
    assert observation.binary64_pressure[0].hex() == (-0.0).hex()
    assert not replace(
        observation,
        binary64_initial_epi=(0.0,),
    ).pre_jump_observation_certified


@pytest.mark.parametrize(
    (
        "field",
        "baseline_epi",
        "candidate_epi",
        "baseline_pressure",
        "candidate_pressure",
    ),
    (
        ("binary64_initial_epi", (0.0,), (-0.0,), (0.0,), (0.0,)),
        ("binary64_pressure", (0.0,), (0.0,), (0.0,), (-0.0,)),
    ),
)
def test_held_comparison_rejects_binary64_signed_zero_mismatch(
    field: str,
    baseline_epi: tuple[float, ...],
    candidate_epi: tuple[float, ...],
    baseline_pressure: tuple[float, ...],
    candidate_pressure: tuple[float, ...],
) -> None:
    baseline = _observation(
        epi=baseline_epi,
        pressure=baseline_pressure,
        substeps=1,
        xi=0.0,
    )
    candidate = _observation(
        epi=candidate_epi,
        pressure=candidate_pressure,
        substeps=2,
        xi=0.0,
    )

    with pytest.raises(ValueError, match=field):
        compare_event_local_zhir_held_pressure_subdivision(
            baseline,
            candidate,
        )
