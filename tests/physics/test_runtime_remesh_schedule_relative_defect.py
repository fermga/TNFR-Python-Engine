"""Finite causal promotion of the relative-defect REMESH envelope."""

from __future__ import annotations

from collections import deque
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import networkx as nx
import pytest

import tnfr.physics.runtime_remesh_schedule_relative_defect as runtime_module
from tnfr.errors import TNFRValueError
from tnfr.operators.event_remesh_causal_runtime import (
    EventRemeshCycleExecutionSpec,
    ExecutedEventRemeshCycleSequence,
    execute_event_remesh_cycle_sequence,
)
from tnfr.operators.event_timing import build_operator_event_schedule
from tnfr.physics.remesh_history_stability import (
    certify_uniform_remesh_history_stability,
)
from tnfr.physics.remesh_schedule_policy_stability import (
    certify_uniform_remesh_schedule_policy_stability,
)
from tnfr.physics.remesh_schedule_relative_defect_stability import (
    UniformRemeshScheduleRelativeDefectStabilityCertificate,
    certify_uniform_remesh_schedule_relative_defect_stability,
)
from tnfr.physics.runtime_remesh_schedule_relative_defect import (
    RuntimeRemeshScheduleRelativeDefectBlockObservation,
    observe_executed_event_remesh_relative_defect_block,
)
from tnfr.physics.runtime_remesh_schedule_stability import (
    RuntimeRemeshScheduleBoundaryObservation,
)


def _set_pure_epi_pressure(graph: nx.Graph) -> None:
    values = {node: float(graph.nodes[node]["EPI"]) for node in graph}
    for node in graph:
        neighbours = tuple(graph.neighbors(node))
        graph.nodes[node]["delta_nfr"] = (
            sum(values[item] for item in neighbours) / len(neighbours)
            - values[node]
        )


def _graph(
    *,
    current: tuple[float, float] = (2.0, 0.0),
    past: tuple[float, float] = (0.0, 2.0),
    alpha: float = 0.5,
) -> nx.Graph:
    graph = nx.path_graph(2)
    graph.graph.update(
        _t=0.0,
        RANDOM_SEED=23,
        _gamma_spec={"type": "none"},
        REMESH_TAU_GLOBAL=1,
        REMESH_TAU_LOCAL=1,
        REMESH_ALPHA=alpha,
        REMESH_ALPHA_HARD=True,
        REMESH_LOG_EVENTS=False,
        EPI_MIN=-10.0,
        EPI_MAX=10.0,
        CLIP_MODE="hard",
        DT_MIN=0.0,
        compute_delta_nfr=_set_pure_epi_pressure,
    )
    for node, epi in enumerate(current):
        graph.nodes[node].update(
            EPI=epi,
            nu_f=1.0,
            theta=0.0,
            delta_nfr=0.0,
            glyph_history=[],
        )
    graph.graph["_epi_hist"] = deque(
        [{node: value for node, value in enumerate(past)}],
        maxlen=64,
    )
    _set_pure_epi_pressure(graph)
    return graph


def _specs(
    count: int,
    *,
    duration: float = 0.125,
) -> tuple[EventRemeshCycleExecutionSpec, ...]:
    result: list[EventRemeshCycleExecutionSpec] = []
    start = 0.0
    for _index in range(count):
        schedule = build_operator_event_schedule(
            (),
            start_time=start,
            flow_durations=(duration,),
        )
        result.append(EventRemeshCycleExecutionSpec(schedule))
        start = schedule.end_time
    return tuple(result)


def _execution(
    *,
    count: int,
    current: tuple[float, float] = (2.0, 0.0),
    past: tuple[float, float] = (0.0, 2.0),
    alpha: float = 0.5,
    duration: float = 0.125,
    metric_weights: tuple[float, float] = (1.0, 1.0),
) -> ExecutedEventRemeshCycleSequence:
    return execute_event_remesh_cycle_sequence(
        _graph(current=current, past=past, alpha=alpha),
        _specs(count, duration=duration),
        metric_weights=metric_weights,
    )


def _relative_certificate(
    *,
    alpha: float | Fraction = Fraction(1, 2),
    tau_local: int = 1,
    tau_global: int = 1,
    q: Fraction = Fraction(9, 16),
    eta: Fraction = Fraction(0),
) -> UniformRemeshScheduleRelativeDefectStabilityCertificate:
    remesh = certify_uniform_remesh_history_stability(
        alpha=alpha,
        tau_local=tau_local,
        tau_global=tau_global,
    )
    policy = certify_uniform_remesh_schedule_policy_stability(remesh, q)
    return certify_uniform_remesh_schedule_relative_defect_stability(
        policy,
        eta,
    )


def _matrix_vector(
    matrix: tuple[tuple[Fraction, ...], ...],
    vector: tuple[Fraction, ...],
) -> tuple[Fraction, ...]:
    return tuple(
        sum(
            (
                entry * value
                for entry, value in zip(row, vector, strict=True)
            ),
            Fraction(0),
        )
        for row in matrix
    )


def _jensen_input_envelope(
    boundary: RuntimeRemeshScheduleBoundaryObservation,
) -> Fraction:
    transition = boundary.exact_transition
    return sum(
        (
            coefficient * transition.exact_history_energies[delay]
            for delay, coefficient in (
                transition.certificate.combined_delay_coefficients
            )
        ),
        Fraction(0),
    )


@pytest.fixture(scope="module")
def dyadic_three_cycle_execution() -> ExecutedEventRemeshCycleSequence:
    return _execution(count=3)


@pytest.fixture(scope="module")
def dyadic_certificate(
) -> UniformRemeshScheduleRelativeDefectStabilityCertificate:
    return _relative_certificate()


@pytest.fixture(scope="module")
def dyadic_observation(
    dyadic_three_cycle_execution: ExecutedEventRemeshCycleSequence,
    dyadic_certificate: UniformRemeshScheduleRelativeDefectStabilityCertificate,
) -> RuntimeRemeshScheduleRelativeDefectBlockObservation:
    return observe_executed_event_remesh_relative_defect_block(
        dyadic_three_cycle_execution,
        dyadic_certificate,
    )


@pytest.fixture(scope="module")
def positive_defect_execution() -> ExecutedEventRemeshCycleSequence:
    return _execution(
        count=2,
        current=(-2.0, -1.0),
        past=(-2.0, -1.0),
        alpha=0.4,
    )


@pytest.fixture(scope="module")
def zero_energy_execution() -> ExecutedEventRemeshCycleSequence:
    return _execution(
        count=2,
        current=(1.0, 1.0),
        past=(1.0, 1.0),
    )


def test_direct_module_and_stub_expose_runtime_relative_defect_api() -> None:
    assert runtime_module.__all__ == (
        "RuntimeRemeshScheduleRelativeDefectBlockObservation",
        "observe_executed_event_remesh_relative_defect_block",
    )
    package = Path(runtime_module.__file__).parent
    stub = (
        package / "runtime_remesh_schedule_relative_defect.pyi"
    ).read_text(encoding="utf-8")
    assert "class RuntimeRemeshScheduleRelativeDefectBlockObservation" in stub
    assert "def observe_executed_event_remesh_relative_defect_block" in stub


def test_zero_defect_dyadic_block_is_bound_to_causal_execution(
    dyadic_three_cycle_execution: ExecutedEventRemeshCycleSequence,
    dyadic_certificate: UniformRemeshScheduleRelativeDefectStabilityCertificate,
    dyadic_observation: RuntimeRemeshScheduleRelativeDefectBlockObservation,
) -> None:
    execution = dyadic_three_cycle_execution
    certificate = dyadic_certificate
    observation = dyadic_observation

    assert type(observation) is (
        RuntimeRemeshScheduleRelativeDefectBlockObservation
    )
    assert observation.source_execution is execution
    assert observation.relative_defect_certificate is certificate
    assert observation.block_observation.source_execution is execution
    assert observation.boundaries == execution.runtime_telescope.boundaries
    assert all(
        observed is expected
        for observed, expected in zip(
            observation.boundaries,
            execution.runtime_telescope.boundaries,
            strict=True,
        )
    )
    assert observation.start_boundary == 0
    assert observation.boundary_count == 2
    assert observation.exact_pre_schedule_energy_defects == (
        Fraction(0),
        Fraction(0),
    )
    assert observation.exact_relative_energy_defect_ratios == (
        Fraction(0),
        Fraction(0),
    )
    assert observation.exact_relative_energy_defect_slacks == (
        Fraction(0),
        Fraction(0),
    )
    assert observation.exact_schedule_energy_gain_upper_bounds == (
        Fraction(9, 16),
        Fraction(9, 16),
    )
    assert observation.relative_defect_block_observation_certified
    assert all(passed for _name, passed in observation.conditions)


def test_construction_and_queries_validate_dependencies_once_without_cache(
    dyadic_three_cycle_execution: ExecutedEventRemeshCycleSequence,
    dyadic_certificate: UniformRemeshScheduleRelativeDefectStabilityCertificate,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counts = {"source": 0, "certificate": 0, "block": 0}
    block_type = runtime_module.RuntimeRemeshScheduleBlockMarginObservation
    original_source_check = runtime_module._execution_is_intact
    original_certificate_check = runtime_module._certificate_is_intact
    original_block_check = (
        block_type._proof_fields_are_intact_after_source_validation
    )

    def source_check(value: object) -> bool:
        counts["source"] += 1
        return original_source_check(value)

    def certificate_check(value: object) -> bool:
        counts["certificate"] += 1
        return original_certificate_check(value)

    def block_check(self: object) -> bool:
        counts["block"] += 1
        return original_block_check(self)

    monkeypatch.setattr(runtime_module, "_execution_is_intact", source_check)
    monkeypatch.setattr(
        runtime_module,
        "_certificate_is_intact",
        certificate_check,
    )
    monkeypatch.setattr(
        block_type,
        "_proof_fields_are_intact_after_source_validation",
        block_check,
    )

    observation = observe_executed_event_remesh_relative_defect_block(
        dyadic_three_cycle_execution,
        dyadic_certificate,
    )
    assert counts == {"source": 1, "certificate": 1, "block": 1}

    counts.update(source=0, certificate=0, block=0)
    assert observation.relative_defect_block_observation_certified
    assert counts == {"source": 1, "certificate": 1, "block": 1}

    # No trust survives the top-level query: the next query validates afresh.
    assert observation.relative_defect_block_observation_certified
    assert counts == {"source": 2, "certificate": 2, "block": 2}


def test_each_boundary_uses_Jensen_input_envelope_before_the_schedule(
    dyadic_certificate: UniformRemeshScheduleRelativeDefectStabilityCertificate,
    dyadic_observation: RuntimeRemeshScheduleRelativeDefectBlockObservation,
) -> None:
    certificate = dyadic_certificate
    observation = dyadic_observation
    eta = certificate.pre_schedule_relative_energy_defect_upper_bound
    q_effective = certificate.exact_effective_head_energy_gain_upper_bound

    for index, boundary in enumerate(observation.boundaries):
        balance = boundary.schedule_balance
        jensen = _jensen_input_envelope(boundary)
        ideal = boundary.exact_transition.exact_next_energy
        bounded = balance.exact_runtime_bounded_head_energy
        scheduled = balance.exact_scheduled_head_energy
        defect = bounded - ideal

        assert observation.exact_remesh_input_energy_upper_bounds[index] == jensen
        assert observation.exact_ideal_remesh_head_energies[index] == ideal
        assert observation.exact_runtime_bounded_head_energies[index] == bounded
        assert observation.exact_pre_schedule_energy_defects[index] == defect
        assert defect <= eta * jensen
        assert observation.exact_relative_energy_defect_slacks[index] == (
            eta * jensen - defect
        )
        assert observation.exact_effective_head_energy_upper_bounds[index] == (
            q_effective * jensen
        )
        assert observation.exact_scheduled_head_energies[index] == scheduled
        assert scheduled <= q_effective * jensen


def test_every_history_energy_vector_obeys_the_effective_matrix_envelope(
    dyadic_certificate: UniformRemeshScheduleRelativeDefectStabilityCertificate,
    dyadic_observation: RuntimeRemeshScheduleRelativeDefectBlockObservation,
) -> None:
    certificate = dyadic_certificate
    observation = dyadic_observation
    matrix = certificate.effective_head_energy_domination_matrix

    for input_vector, output_vector, recorded_envelope in zip(
        observation.exact_input_history_energy_vectors,
        observation.exact_output_history_energy_vectors,
        observation.exact_energy_envelope_vectors,
        strict=True,
    ):
        expected_envelope = _matrix_vector(matrix, input_vector)
        assert recorded_envelope == expected_envelope
        assert all(
            observed <= upper
            for observed, upper in zip(
                output_vector,
                expected_envelope,
                strict=True,
            )
        )
        assert output_vector[1:] == input_vector[:-1]

    assert dict(observation.conditions)[
        "every_history_energy_vector_satisfies_effective_envelope"
    ]


def test_endpoint_bound_uses_only_complete_universal_blocks(
    dyadic_three_cycle_execution: ExecutedEventRemeshCycleSequence,
    dyadic_certificate: UniformRemeshScheduleRelativeDefectStabilityCertificate,
    dyadic_observation: RuntimeRemeshScheduleRelativeDefectBlockObservation,
) -> None:
    execution = dyadic_three_cycle_execution
    certificate = dyadic_certificate
    prefix = observe_executed_event_remesh_relative_defect_block(
        execution,
        certificate,
        boundary_count=1,
    )
    whole = dyadic_observation

    assert certificate.universal_block_horizon == 2
    assert prefix.exact_finite_endpoint_energy_gain_upper_bound == 1
    assert prefix.exact_finite_endpoint_energy_upper_bound == (
        prefix.exact_augmented_energy_before
    )
    assert prefix.exact_augmented_energy_after <= (
        prefix.exact_finite_endpoint_energy_upper_bound
    )
    assert prefix.exact_finite_endpoint_energy_gain_upper_bound <= 1
    assert prefix.exact_finite_endpoint_energy_gain_upper_bound == 1

    assert whole.exact_finite_endpoint_energy_gain_upper_bound == Fraction(9, 16)
    assert whole.exact_finite_endpoint_energy_upper_bound == (
        Fraction(9, 16) * whole.exact_augmented_energy_before
    )
    assert whole.exact_augmented_energy_after <= (
        whole.exact_finite_endpoint_energy_upper_bound
    )
    assert dict(whole.conditions)["finite_endpoint_energy_bound_satisfied"]
    assert whole.exact_augmented_energy_before > 0
    assert whole.exact_finite_endpoint_energy_gain_upper_bound < 1


def test_positive_binary64_defect_is_accepted_at_exact_minimum_eta(
    positive_defect_execution: ExecutedEventRemeshCycleSequence,
) -> None:
    execution = positive_defect_execution
    boundary = execution.runtime_telescope.boundaries[0]
    jensen = _jensen_input_envelope(boundary)
    ideal = boundary.exact_transition.exact_next_energy
    bounded = boundary.schedule_balance.exact_runtime_bounded_head_energy
    defect = bounded - ideal
    eta = defect / jensen
    certificate = _relative_certificate(alpha=0.4, eta=eta)
    observation = observe_executed_event_remesh_relative_defect_block(
        execution,
        certificate,
    )

    assert jensen > 0
    assert defect > 0
    assert ideal <= jensen
    assert certificate.exact_effective_head_energy_gain_upper_bound < 1
    assert observation.exact_remesh_input_energy_upper_bounds == (jensen,)
    assert observation.exact_pre_schedule_energy_defects == (defect,)
    assert observation.exact_relative_energy_defect_ratios == (eta,)
    assert observation.exact_relative_energy_defect_slacks == (Fraction(0),)
    assert observation.relative_defect_block_observation_certified


def test_positive_defect_rejects_one_exact_rational_step_below_minimum_eta(
    positive_defect_execution: ExecutedEventRemeshCycleSequence,
) -> None:
    boundary = positive_defect_execution.runtime_telescope.boundaries[0]
    jensen = _jensen_input_envelope(boundary)
    defect = (
        boundary.schedule_balance.exact_runtime_bounded_head_energy
        - boundary.exact_transition.exact_next_energy
    )
    minimum_eta = defect / jensen
    eta_below = Fraction(
        minimum_eta.numerator - 1,
        minimum_eta.denominator,
    )
    certificate = _relative_certificate(alpha=0.4, eta=eta_below)

    with pytest.raises(TNFRValueError, match="relative|defect|eta"):
        observe_executed_event_remesh_relative_defect_block(
            positive_defect_execution,
            certificate,
        )


def test_schedule_gain_larger_than_declared_q_is_rejected(
    dyadic_three_cycle_execution: ExecutedEventRemeshCycleSequence,
) -> None:
    certificate = _relative_certificate(q=Fraction(1, 2))

    with pytest.raises(TNFRValueError, match="schedule|gain|q"):
        observe_executed_event_remesh_relative_defect_block(
            dyadic_three_cycle_execution,
            certificate,
        )


def test_mismatched_remesh_configuration_is_rejected(
    dyadic_three_cycle_execution: ExecutedEventRemeshCycleSequence,
) -> None:
    certificate = _relative_certificate(
        alpha=Fraction(1, 3),
        tau_local=1,
        tau_global=1,
    )

    with pytest.raises(TNFRValueError, match="REMESH|remesh|configuration"):
        observe_executed_event_remesh_relative_defect_block(
            dyadic_three_cycle_execution,
            certificate,
        )


def test_effective_gain_one_remains_a_zero_margin_runtime_boundary(
    dyadic_three_cycle_execution: ExecutedEventRemeshCycleSequence,
) -> None:
    certificate = _relative_certificate(eta=Fraction(7, 9))
    observation = observe_executed_event_remesh_relative_defect_block(
        dyadic_three_cycle_execution,
        certificate,
    )

    assert certificate.exact_effective_head_energy_gain_upper_bound == 1
    assert observation.exact_finite_endpoint_energy_gain_upper_bound == 1
    assert observation.exact_augmented_energy_after < (
        observation.exact_augmented_energy_before
    )
    assert observation.finite_energy_nonincrease_sufficiently_certified
    assert not observation.finite_strict_energy_contraction_sufficiently_certified
    assert not certificate.geometric_spatial_disagreement_convergence_certified


def test_zero_energy_uses_no_ratio_and_is_preserved(
    zero_energy_execution: ExecutedEventRemeshCycleSequence,
) -> None:
    certificate = _relative_certificate(eta=Fraction(7, 9))
    observation = observe_executed_event_remesh_relative_defect_block(
        zero_energy_execution,
        certificate,
    )

    assert observation.exact_remesh_input_energy_upper_bounds == (Fraction(0),)
    assert observation.exact_ideal_remesh_head_energies == (Fraction(0),)
    assert observation.exact_runtime_bounded_head_energies == (Fraction(0),)
    assert observation.exact_pre_schedule_energy_defects == (Fraction(0),)
    assert observation.exact_relative_energy_defect_ratios == (None,)
    assert observation.exact_relative_energy_defect_slacks == (Fraction(0),)
    assert observation.exact_scheduled_head_energies == (Fraction(0),)
    assert observation.exact_augmented_energy_before == 0
    assert observation.exact_augmented_energy_after == 0
    assert observation.exact_finite_endpoint_energy_upper_bound == 0
    assert observation.zero_energy_preservation_observed


@pytest.mark.parametrize(
    ("start", "count", "error"),
    (
        (True, None, TypeError),
        (0, True, TypeError),
        (-1, None, TNFRValueError),
        (2, None, TNFRValueError),
        (0, 0, TNFRValueError),
        (1, 2, TNFRValueError),
    ),
)
def test_invalid_or_empty_boundary_ranges_are_rejected(
    start: int,
    count: int | None,
    error: type[Exception],
    dyadic_three_cycle_execution: ExecutedEventRemeshCycleSequence,
) -> None:
    with pytest.raises(error):
        observe_executed_event_remesh_relative_defect_block(
            dyadic_three_cycle_execution,
            _relative_certificate(),
            start_boundary=start,
            boundary_count=count,
        )


def test_equal_value_boundary_copy_cannot_replace_causal_identity(
    dyadic_observation: RuntimeRemeshScheduleRelativeDefectBlockObservation,
) -> None:
    observation = dyadic_observation
    copied = replace(observation.boundaries[0])
    forged = replace(
        observation,
        boundaries=(copied, observation.boundaries[1]),
        _proof_stamp=(),
    )
    object.__setattr__(
        forged,
        "_proof_stamp",
        runtime_module._proof_stamp_from_values(
            runtime_module._observation_values(forged)
        ),
    )
    resealed = forged

    assert copied.boundary_observation_certified
    assert not resealed.relative_defect_block_observation_certified


def test_nested_boundary_tampering_invalidates_existing_observation() -> None:
    execution = _execution(count=2)
    certificate = _relative_certificate()
    observation = observe_executed_event_remesh_relative_defect_block(
        execution,
        certificate,
    )
    boundary = execution.runtime_telescope.boundaries[0]
    object.__setattr__(boundary, "exact_energy_drop", Fraction(99))

    assert not boundary.boundary_observation_certified
    assert not execution._proof_fields_are_intact()
    assert not observation.relative_defect_block_observation_certified


def test_nested_relative_defect_certificate_tampering_invalidates_observation(
    dyadic_three_cycle_execution: ExecutedEventRemeshCycleSequence,
) -> None:
    certificate = _relative_certificate()
    observation = observe_executed_event_remesh_relative_defect_block(
        dyadic_three_cycle_execution,
        certificate,
    )
    object.__setattr__(
        certificate,
        "pre_schedule_relative_energy_defect_upper_bound",
        Fraction(1, 3),
    )

    assert not certificate.relative_defect_stability_certificate_certified
    assert not observation.relative_defect_block_observation_certified


@pytest.mark.parametrize(
    ("field_name", "forged_value"),
    (
        ("exact_finite_endpoint_energy_gain_upper_bound", Fraction(0)),
        ("exact_finite_endpoint_energy_upper_bound", Fraction(0)),
        ("exact_relative_energy_defect_slacks", (Fraction(99),)),
        ("conditions", (("forged", True),)),
    ),
)
def test_private_outer_reseal_cannot_promote_changed_derivatives(
    field_name: str,
    forged_value: object,
    dyadic_observation: RuntimeRemeshScheduleRelativeDefectBlockObservation,
) -> None:
    observation = dyadic_observation
    forged = replace(
        observation,
        **{field_name: forged_value},
        _proof_stamp=(),
    )
    object.__setattr__(
        forged,
        "_proof_stamp",
        runtime_module._proof_stamp_from_values(
            runtime_module._observation_values(forged)
        ),
    )
    resealed = forged

    assert not resealed.relative_defect_block_observation_certified


def test_wrong_public_input_types_are_rejected(
    dyadic_three_cycle_execution: ExecutedEventRemeshCycleSequence,
) -> None:
    with pytest.raises(TypeError, match="ExecutedEventRemeshCycleSequence"):
        observe_executed_event_remesh_relative_defect_block(
            object(),  # type: ignore[arg-type]
            _relative_certificate(),
        )
    with pytest.raises(TypeError, match="relative|certificate"):
        observe_executed_event_remesh_relative_defect_block(
            dyadic_three_cycle_execution,
            object(),  # type: ignore[arg-type]
        )


def test_finite_adapter_keeps_future_and_solver_claims_false(
    dyadic_observation: RuntimeRemeshScheduleRelativeDefectBlockObservation,
) -> None:
    observation = dyadic_observation

    assert "finite" in observation.scope.lower()
    assert "Jensen" in runtime_module.__doc__
    assert not observation.runtime_forward_invariant_class_certified
    assert not observation.repeated_binary64_runtime_stability_certified
    assert not observation.future_runtime_stability_certified
    assert not observation.solver_accuracy_certified
    assert not observation.solver_order_certified
    assert not observation.full_tnfr_stability_certified
