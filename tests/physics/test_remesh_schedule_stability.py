"""Exact algebraic REMESH-to-schedule augmented-energy balance."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

import tnfr.physics.remesh_schedule_stability as _module
from tnfr.errors import TNFRValueError
from tnfr.physics.remesh_history_stability import (
    certify_uniform_remesh_history_stability,
    observe_uniform_remesh_history_transition,
)
from tnfr.physics.remesh_schedule_stability import (
    RemeshScheduleHistoryStabilityObservation,
    observe_remesh_schedule_history_transition,
)


def _transition(*, constant_history: bool = False):
    certificate = certify_uniform_remesh_history_stability(
        alpha=Fraction(1, 2),
        tau_local=1,
        tau_global=1,
    )
    history = (
        (
            (Fraction(1, 2), Fraction(3, 2)),
            (Fraction(1, 2), Fraction(3, 2)),
        )
        if constant_history
        else ((Fraction(2), Fraction(0)), (Fraction(0), Fraction(2)))
    )
    return observe_uniform_remesh_history_transition(
        certificate,
        history,
        (Fraction(1), Fraction(1)),
    )


def test_public_physics_facade_and_stub_expose_schedule_balance() -> None:
    import tnfr.physics as physics

    assert {
        "RemeshScheduleHistoryStabilityObservation",
        "observe_remesh_schedule_history_transition",
    } <= set(physics.__all__)
    assert (
        physics.RemeshScheduleHistoryStabilityObservation
        is RemeshScheduleHistoryStabilityObservation
    )
    assert (
        physics.observe_remesh_schedule_history_transition
        is observe_remesh_schedule_history_transition
    )

    package = Path(physics.__file__).parent
    stub = (package / "remesh_schedule_stability.pyi").read_text(
        encoding="utf-8"
    )
    assert "class RemeshScheduleHistoryStabilityObservation" in stub
    assert "def observe_remesh_schedule_history_transition" in stub


def test_exact_balance_with_strict_schedule_contraction() -> None:
    transition = _transition()
    ideal = transition.exact_next_field
    observation = observe_remesh_schedule_history_transition(
        transition,
        ideal,
        ideal,
        (1, 1),
        Fraction(0),
    )

    assert type(observation) is RemeshScheduleHistoryStabilityObservation
    assert observation.transition_observation_certified
    assert observation.exact_energy_balance_certified
    assert observation.schedule_energy_gain_bound_certified
    assert observation.exact_runtime_raw_head_energy == Fraction(1, 4)
    assert observation.exact_runtime_bounded_head_energy == Fraction(1, 4)
    assert observation.exact_scheduled_head_energy == 0
    assert observation.exact_raw_spatial_energy_defect == 0
    assert observation.exact_clipping_spatial_energy_defect == 0
    assert observation.exact_schedule_spatial_energy_defect == Fraction(-1, 4)
    assert observation.exact_schedule_augmented_energy_defect == Fraction(-1, 7)
    assert observation.exact_total_augmented_energy_defect == Fraction(-1, 7)
    assert observation.exact_augmented_energy_before == 1
    assert observation.exact_augmented_energy_after == Fraction(3, 7)
    assert observation.exact_energy_drop == Fraction(4, 7)
    assert observation.exact_schedule_energy_gain_slack == 0
    assert observation.exact_schedule_augmented_energy_gain_slack == 0
    assert observation.exact_gain_based_energy_drop_lower_bound == Fraction(4, 7)
    assert observation.exact_stationary_history_barycenter_drift == (
        Fraction(2, 7),
        Fraction(-2, 7),
    )
    assert observation.energy_nonincrease_observed
    assert observation.energy_nonincrease_sufficiently_certified
    assert not observation.stationary_history_barycenter_preserved_observed
    assert not observation.runtime_provenance_certified
    assert not observation.repeated_stability_certified
    assert not observation.future_stability_certified


def test_loose_schedule_gain_exposes_exact_slack_and_lower_bound() -> None:
    transition = _transition()
    ideal = transition.exact_next_field
    observation = observe_remesh_schedule_history_transition(
        transition,
        ideal,
        ideal,
        (1, 1),
        Fraction(1, 2),
    )

    assert observation.exact_energy_drop == Fraction(4, 7)
    assert observation.exact_schedule_energy_gain_slack == Fraction(1, 8)
    assert observation.exact_schedule_augmented_energy_gain_slack == Fraction(1, 14)
    assert observation.exact_gain_based_energy_drop_lower_bound == Fraction(1, 2)
    assert observation.exact_energy_drop == (
        observation.exact_gain_based_energy_drop_lower_bound
        + observation.exact_schedule_augmented_energy_gain_slack
    )


def test_expansive_schedule_can_be_paid_by_jensen_dissipation() -> None:
    transition = _transition()
    ideal = transition.exact_next_field
    observation = observe_remesh_schedule_history_transition(
        transition,
        ideal,
        ideal,
        (0, 2),
        Fraction(4),
    )

    assert observation.exact_scheduled_head_energy == 1
    assert observation.exact_schedule_spatial_energy_defect == Fraction(3, 4)
    assert observation.exact_schedule_augmented_energy_defect == Fraction(3, 7)
    assert observation.exact_total_augmented_energy_defect == Fraction(3, 7)
    assert observation.exact_augmented_energy_after == 1
    assert observation.exact_energy_drop == 0
    assert observation.exact_schedule_energy_gain_slack == 0
    assert observation.exact_schedule_contraction_augmented_margin == Fraction(-3, 7)
    assert observation.exact_gain_based_energy_drop_lower_bound == 0
    assert observation.energy_nonincrease_observed
    assert observation.energy_nonincrease_sufficiently_certified


def test_soft_like_bounded_expansion_is_observed_without_promotion() -> None:
    transition = _transition(constant_history=True)
    ideal = transition.exact_next_field
    observation = observe_remesh_schedule_history_transition(
        transition,
        ideal,
        (-1, 3),
        (-1, 3),
        Fraction(1),
    )

    assert transition.exact_jensen_dissipation == 0
    assert observation.exact_runtime_raw_head_energy == Fraction(1, 4)
    assert observation.exact_runtime_bounded_head_energy == 4
    assert observation.exact_scheduled_head_energy == 4
    assert observation.exact_clipping_spatial_energy_defect == Fraction(15, 4)
    assert observation.exact_clipping_augmented_energy_defect == Fraction(15, 7)
    assert observation.exact_schedule_augmented_energy_defect == 0
    assert observation.exact_total_augmented_energy_defect == Fraction(15, 7)
    assert observation.exact_augmented_energy_after == Fraction(67, 28)
    assert observation.exact_energy_drop == Fraction(-15, 7)
    assert observation.exact_gain_based_energy_drop_lower_bound == Fraction(-15, 7)
    assert observation.exact_stationary_history_barycenter_drift == (
        Fraction(-6, 7),
        Fraction(6, 7),
    )
    assert observation.transition_observation_certified
    assert observation.exact_energy_balance_certified
    assert not observation.energy_nonincrease_observed
    assert not observation.energy_nonincrease_sufficiently_certified


@pytest.mark.parametrize(
    "gain",
    (True, Fraction(-1), float("nan"), float("inf"), "1"),
)
def test_invalid_schedule_gain_is_rejected(gain: object) -> None:
    transition = _transition()
    ideal = transition.exact_next_field

    with pytest.raises(TNFRValueError, match="schedule_energy_gain_upper_bound"):
        observe_remesh_schedule_history_transition(
            transition,
            ideal,
            ideal,
            (1, 1),
            gain,  # type: ignore[arg-type]
        )


def test_violated_schedule_gain_is_rejected() -> None:
    transition = _transition()
    ideal = transition.exact_next_field

    with pytest.raises(TNFRValueError, match="violates"):
        observe_remesh_schedule_history_transition(
            transition,
            ideal,
            ideal,
            (0, 2),
            Fraction(3),
        )
    with pytest.raises(TNFRValueError, match="violates"):
        observe_remesh_schedule_history_transition(
            transition,
            ideal,
            (1, 1),
            (0, 2),
            Fraction(100),
        )


def test_real_inputs_are_rationalized_exactly() -> None:
    transition = _transition()
    ideal = transition.exact_next_field
    observation = observe_remesh_schedule_history_transition(
        transition,
        tuple(float(item) for item in ideal),
        tuple(float(item) for item in ideal),
        (1.0, 1.0),
        0.1,
    )

    assert observation.exact_schedule_energy_gain_upper_bound == Fraction.from_float(
        0.1
    )


def test_outer_observation_fails_closed_after_direct_tampering() -> None:
    transition = _transition()
    ideal = transition.exact_next_field
    observation = observe_remesh_schedule_history_transition(
        transition,
        ideal,
        ideal,
        (1, 1),
        Fraction(0),
    )
    object.__setattr__(observation, "exact_energy_drop", Fraction(99))

    assert not observation.transition_observation_certified
    assert not observation.exact_energy_balance_certified
    assert not observation.energy_nonincrease_observed
    assert observation.failed_conditions == (
        "remesh_schedule_history_transition_proof_fields_intact",
    )


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    (
        ("exact_energy_drop", Fraction(99)),
        ("exact_schedule_augmented_energy_gain_slack", Fraction(99)),
        (
            "exact_stationary_history_barycenter_drift",
            (Fraction(99), Fraction(99)),
        ),
    ),
)
def test_private_reseal_cannot_promote_changed_derived_field(
    field_name: str,
    replacement: object,
) -> None:
    transition = _transition()
    ideal = transition.exact_next_field
    observation = observe_remesh_schedule_history_transition(
        transition,
        ideal,
        ideal,
        (1, 1),
        Fraction(0),
    )
    object.__setattr__(observation, field_name, replacement)
    values = _module._observation_values(observation)
    object.__setattr__(
        observation,
        "_proof_stamp",
        _module._proof_stamp_from_values(values),
    )

    assert not observation.transition_observation_certified
    assert not observation.exact_energy_balance_certified


class _ExplodingEquality:
    def __eq__(self, other: object) -> bool:
        raise RuntimeError("caller equality must not be invoked")


def test_hostile_private_reseal_fails_closed_without_equality_dispatch() -> None:
    transition = _transition()
    ideal = transition.exact_next_field
    observation = observe_remesh_schedule_history_transition(
        transition,
        ideal,
        ideal,
        (1, 1),
        Fraction(0),
    )
    object.__setattr__(observation, "exact_energy_drop", _ExplodingEquality())
    values = _module._observation_values(observation)
    object.__setattr__(
        observation,
        "_proof_stamp",
        _module._proof_stamp_from_values(values),
    )

    assert not observation.transition_observation_certified
    assert not observation.energy_nonincrease_observed


def test_outer_observation_fails_closed_after_nested_tampering() -> None:
    transition = _transition()
    ideal = transition.exact_next_field
    observation = observe_remesh_schedule_history_transition(
        transition,
        ideal,
        ideal,
        (1, 1),
        Fraction(0),
    )
    object.__setattr__(transition, "exact_energy_drop", Fraction(99))

    assert not transition.transition_observation_certified
    assert not observation.transition_observation_certified
    assert not observation.schedule_energy_gain_bound_certified
    with pytest.raises(TNFRValueError, match="unsealed, tampered"):
        observe_remesh_schedule_history_transition(
            transition,
            ideal,
            ideal,
            (1, 1),
            Fraction(0),
        )


def test_wrong_transition_type_and_head_width_are_rejected() -> None:
    with pytest.raises(TypeError, match="UniformRemeshHistoryTransitionObservation"):
        observe_remesh_schedule_history_transition(
            object(), (), (), (), Fraction(1)  # type: ignore[arg-type]
        )

    transition = _transition()
    with pytest.raises(TNFRValueError, match="transition width"):
        observe_remesh_schedule_history_transition(
            transition,
            (0,),
            (0,),
            (0,),
            Fraction(1),
        )
