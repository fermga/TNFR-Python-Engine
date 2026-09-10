r"""Exact augmented-energy balance for one REMESH-to-schedule transition.

The uniform REMESH companion produces an ideal head ``y``.  This module
accepts three caller-supplied heads derived after that model step: a raw head
``b``, a bounded head ``z`` and a scheduled head ``s``.  In the fixed positive
diagonal metric of the companion observation it separates

``E(b) - E(y)``, ``E(z) - E(b)`` and ``E(s) - E(z)``

before lifting each defect by the stationary head weight ``pi[0]``.  The
resulting identity is algebraic.  It neither identifies the supplied heads
with an executable runtime nor certifies repeated or future behavior.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field, fields
from fractions import Fraction
from numbers import Real
from typing import Any

from ..errors import TNFRValueError
from ..utils._structural_signature import proof_stamps_are_identical
from .remesh_history_stability import (
    ExactVector,
    UniformRemeshHistoryTransitionObservation,
    _centered_energy,
    _exact_scalar,
    _exact_vector,
    _weighted_history_field,
)

__all__ = (
    "RemeshScheduleHistoryStabilityObservation",
    "observe_remesh_schedule_history_transition",
)

_PROOF_VERSION = "remesh_schedule_history_stability_v1"
_SCOPE = (
    "One exact algebraic transition from an intact uniform REMESH companion "
    "observation through supplied raw, bounded and scheduled heads in the "
    "same positive diagonal metric. Signed spatial and stationary-weighted "
    "augmented energy defects, the schedule-gain slack, a sufficient energy-"
    "drop lower bound and stationary-history barycenter drift are retained. "
    "The supplied heads are not identified with an executable runtime. "
    "Runtime provenance, repetition, future stability, changing parameters "
    "or support, solver accuracy and full TNFR stability are not certified."
)

_CONDITION_NAMES = (
    "nested_remesh_transition_intact",
    "heads_share_transition_width",
    "spatial_energy_defects_telescope",
    "augmented_defects_are_pi0_weighted",
    "augmented_energy_defects_telescope",
    "schedule_energy_gain_bound_satisfied",
    "schedule_energy_gain_slack_nonnegative",
    "exact_augmented_energy_balance",
    "gain_based_lower_bound_identity",
    "exact_drop_equals_lower_bound_plus_slack",
    "stationary_history_barycenter_drift_identity",
)


def _vector_subtract(left: ExactVector, right: ExactVector) -> ExactVector:
    return tuple(a - b for a, b in zip(left, right, strict=True))


def _scale_vector(scale: Fraction, value: ExactVector) -> ExactVector:
    return tuple(scale * item for item in value)


def _strict_exact_vector(value: Any, width: int) -> bool:
    return bool(
        type(value) is tuple
        and len(value) == width
        and all(type(item) is Fraction for item in value)
    )


def _strict_conditions(value: Any) -> bool:
    return bool(
        type(value) is tuple
        and len(value) == len(_CONDITION_NAMES)
        and all(
            type(item) is tuple
            and len(item) == 2
            and type(item[0]) is str
            and type(item[1]) is bool
            and item[0] == _CONDITION_NAMES[index]
            for index, item in enumerate(value)
        )
    )


def _transition_stamp_or_none(value: Any) -> tuple[Any, ...] | None:
    if type(value) is not UniformRemeshHistoryTransitionObservation:
        return None
    try:
        stamp = object.__getattribute__(value, "_proof_stamp")
    except BaseException:
        return None
    return stamp if type(stamp) is tuple else None


def _proof_stamp_from_values(values: dict[str, Any]) -> tuple[Any, ...]:
    """Build a compact closed token without copying the nested proof tree."""

    return (
        _PROOF_VERSION,
        _transition_stamp_or_none(values["transition"]),
        tuple(
            (name, values[name])
            for name in _FIELD_NAMES
            if name not in {"transition", "_proof_stamp"}
        ),
    )


def _observation_values(
    value: "RemeshScheduleHistoryStabilityObservation",
) -> dict[str, Any]:
    if type(value) is not RemeshScheduleHistoryStabilityObservation:
        raise TypeError("observation must have its canonical result type")
    return {
        item.name: object.__getattribute__(value, item.name)
        for item in fields(RemeshScheduleHistoryStabilityObservation)
    }


@dataclass(frozen=True, slots=True)
class RemeshScheduleHistoryStabilityObservation:
    """One sealed exact REMESH-head and schedule-head energy balance."""

    transition: UniformRemeshHistoryTransitionObservation
    exact_runtime_raw_head: ExactVector
    exact_runtime_bounded_head: ExactVector
    exact_scheduled_head: ExactVector
    exact_runtime_raw_head_centered: ExactVector
    exact_runtime_bounded_head_centered: ExactVector
    exact_scheduled_head_centered: ExactVector
    exact_runtime_raw_head_energy: Fraction
    exact_runtime_bounded_head_energy: Fraction
    exact_scheduled_head_energy: Fraction
    exact_schedule_energy_gain_upper_bound: Fraction
    exact_schedule_energy_gain_slack: Fraction
    exact_schedule_augmented_energy_gain_slack: Fraction
    exact_raw_spatial_energy_defect: Fraction
    exact_clipping_spatial_energy_defect: Fraction
    exact_schedule_spatial_energy_defect: Fraction
    exact_total_spatial_energy_defect: Fraction
    exact_raw_augmented_energy_defect: Fraction
    exact_clipping_augmented_energy_defect: Fraction
    exact_schedule_augmented_energy_defect: Fraction
    exact_total_augmented_energy_defect: Fraction
    exact_augmented_energy_before: Fraction
    exact_augmented_energy_after: Fraction
    exact_energy_drop: Fraction
    exact_schedule_contraction_augmented_margin: Fraction
    exact_gain_based_energy_drop_lower_bound: Fraction
    exact_stationary_history_barycenter_before: ExactVector
    exact_stationary_history_barycenter_after: ExactVector
    exact_stationary_history_barycenter_drift: ExactVector
    conditions: tuple[tuple[str, bool], ...]
    _proof_stamp: tuple[Any, ...] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        values = _observation_values(self)
        if not proof_stamps_are_identical(
            object.__getattribute__(self, "_proof_stamp"),
            _proof_stamp_from_values(values),
        ):
            raise ValueError(
                "REMESH/schedule history proof fields are inconsistent"
            )
        _validate_observation(self)

    def _proof_fields_are_intact(self) -> bool:
        try:
            self.__post_init__()
        except BaseException:
            return False
        return True

    @property
    def scope(self) -> str:
        return _SCOPE

    @property
    def transition_observation_certified(self) -> bool:
        return self._proof_fields_are_intact()

    @property
    def failed_conditions(self) -> tuple[str, ...]:
        if not self._proof_fields_are_intact():
            return ("remesh_schedule_history_transition_proof_fields_intact",)
        return tuple(name for name, passed in self.conditions if not passed)

    @property
    def exact_energy_balance_certified(self) -> bool:
        return self._proof_fields_are_intact()

    @property
    def schedule_energy_gain_bound_certified(self) -> bool:
        return self._proof_fields_are_intact()

    @property
    def energy_nonincrease_observed(self) -> bool:
        return bool(self._proof_fields_are_intact() and self.exact_energy_drop >= 0)

    @property
    def energy_nonincrease_sufficiently_certified(self) -> bool:
        return bool(
            self._proof_fields_are_intact()
            and self.exact_gain_based_energy_drop_lower_bound >= 0
        )

    @property
    def stationary_history_barycenter_preserved_observed(self) -> bool:
        return bool(
            self._proof_fields_are_intact()
            and all(
                item == 0
                for item in self.exact_stationary_history_barycenter_drift
            )
        )

    @property
    def runtime_provenance_certified(self) -> bool:
        return False

    @property
    def repeated_stability_certified(self) -> bool:
        return False

    @property
    def future_stability_certified(self) -> bool:
        return False


_FIELD_NAMES = tuple(
    item.name for item in fields(RemeshScheduleHistoryStabilityObservation)
)


def _validate_observation(
    value: RemeshScheduleHistoryStabilityObservation,
) -> None:
    """Recheck the closed exact payload without trusting derived booleans."""

    transition = object.__getattribute__(value, "transition")
    if type(transition) is not UniformRemeshHistoryTransitionObservation:
        raise TypeError(
            "transition must be a UniformRemeshHistoryTransitionObservation"
        )
    if not transition.transition_observation_certified:
        raise ValueError("nested REMESH history transition is not intact")

    width = len(transition.exact_metric_weights)
    vectors = (
        value.exact_runtime_raw_head,
        value.exact_runtime_bounded_head,
        value.exact_scheduled_head,
        value.exact_runtime_raw_head_centered,
        value.exact_runtime_bounded_head_centered,
        value.exact_scheduled_head_centered,
        value.exact_stationary_history_barycenter_before,
        value.exact_stationary_history_barycenter_after,
        value.exact_stationary_history_barycenter_drift,
    )
    if width <= 0 or any(not _strict_exact_vector(item, width) for item in vectors):
        raise TypeError("all exact vectors must match the transition width")
    scalar_names = (
        "exact_runtime_raw_head_energy",
        "exact_runtime_bounded_head_energy",
        "exact_scheduled_head_energy",
        "exact_schedule_energy_gain_upper_bound",
        "exact_schedule_energy_gain_slack",
        "exact_schedule_augmented_energy_gain_slack",
        "exact_raw_spatial_energy_defect",
        "exact_clipping_spatial_energy_defect",
        "exact_schedule_spatial_energy_defect",
        "exact_total_spatial_energy_defect",
        "exact_raw_augmented_energy_defect",
        "exact_clipping_augmented_energy_defect",
        "exact_schedule_augmented_energy_defect",
        "exact_total_augmented_energy_defect",
        "exact_augmented_energy_before",
        "exact_augmented_energy_after",
        "exact_energy_drop",
        "exact_schedule_contraction_augmented_margin",
        "exact_gain_based_energy_drop_lower_bound",
    )
    if any(type(getattr(value, name)) is not Fraction for name in scalar_names):
        raise TypeError("all exact scalar diagnostics must be Fractions")
    if value.exact_schedule_energy_gain_upper_bound < 0:
        raise ValueError("schedule energy gain must be nonnegative")
    if not _strict_conditions(value.conditions) or not all(
        passed for _, passed in value.conditions
    ):
        raise ValueError("REMESH/schedule conditions are inconsistent")

    metric = transition.exact_metric_weights
    raw_centered, raw_energy = _centered_energy(
        value.exact_runtime_raw_head, metric
    )
    bounded_centered, bounded_energy = _centered_energy(
        value.exact_runtime_bounded_head, metric
    )
    scheduled_centered, scheduled_energy = _centered_energy(
        value.exact_scheduled_head, metric
    )
    if (
        value.exact_runtime_raw_head_centered != raw_centered
        or value.exact_runtime_bounded_head_centered != bounded_centered
        or value.exact_scheduled_head_centered != scheduled_centered
        or value.exact_runtime_raw_head_energy != raw_energy
        or value.exact_runtime_bounded_head_energy != bounded_energy
        or value.exact_scheduled_head_energy != scheduled_energy
    ):
        raise ValueError("head energy diagnostics are inconsistent")

    pi_zero = transition.certificate.stationary_distribution[0]
    ideal_energy = transition.exact_next_energy
    raw_spatial = raw_energy - ideal_energy
    clipping_spatial = bounded_energy - raw_energy
    schedule_spatial = scheduled_energy - bounded_energy
    total_spatial = scheduled_energy - ideal_energy
    spatial = (raw_spatial, clipping_spatial, schedule_spatial, total_spatial)
    observed_spatial = (
        value.exact_raw_spatial_energy_defect,
        value.exact_clipping_spatial_energy_defect,
        value.exact_schedule_spatial_energy_defect,
        value.exact_total_spatial_energy_defect,
    )
    augmented = tuple(pi_zero * item for item in spatial)
    observed_augmented = (
        value.exact_raw_augmented_energy_defect,
        value.exact_clipping_augmented_energy_defect,
        value.exact_schedule_augmented_energy_defect,
        value.exact_total_augmented_energy_defect,
    )
    if observed_spatial != spatial or observed_augmented != augmented:
        raise ValueError("energy defect diagnostics are inconsistent")

    q = value.exact_schedule_energy_gain_upper_bound
    slack = q * bounded_energy - scheduled_energy
    augmented_slack = pi_zero * slack
    contraction_margin = pi_zero * (Fraction(1) - q) * bounded_energy
    lower_bound = (
        transition.exact_jensen_dissipation
        - augmented[0]
        - augmented[1]
        + contraction_margin
    )
    scheduled_post_history = (
        value.exact_scheduled_head,
    ) + transition.exact_history[:-1]
    barycenter_before = transition.exact_stationary_history_barycenter
    barycenter_after = _weighted_history_field(
        transition.certificate.stationary_distribution,
        scheduled_post_history,
    )
    barycenter_drift = _vector_subtract(barycenter_after, barycenter_before)
    augmented_after = (
        transition.exact_augmented_energy_after + augmented[3]
    )
    energy_drop = transition.exact_augmented_energy_before - augmented_after
    expected = (
        value.exact_schedule_energy_gain_slack == slack,
        value.exact_schedule_augmented_energy_gain_slack == augmented_slack,
        value.exact_schedule_contraction_augmented_margin == contraction_margin,
        value.exact_gain_based_energy_drop_lower_bound == lower_bound,
        value.exact_augmented_energy_before
        == transition.exact_augmented_energy_before,
        value.exact_augmented_energy_after == augmented_after,
        value.exact_energy_drop == energy_drop,
        value.exact_stationary_history_barycenter_before == barycenter_before,
        value.exact_stationary_history_barycenter_after == barycenter_after,
        value.exact_stationary_history_barycenter_drift == barycenter_drift,
        slack >= 0,
        energy_drop == lower_bound + augmented_slack,
    )
    if not all(expected):
        raise ValueError("REMESH/schedule derived diagnostics are inconsistent")


def observe_remesh_schedule_history_transition(
    transition: UniformRemeshHistoryTransitionObservation,
    runtime_raw_head: Iterable[Real],
    runtime_bounded_head: Iterable[Real],
    scheduled_head: Iterable[Real],
    schedule_energy_gain_upper_bound: Real,
) -> RemeshScheduleHistoryStabilityObservation:
    """Observe one exact algebraic REMESH-to-schedule history transition."""

    if type(transition) is not UniformRemeshHistoryTransitionObservation:
        raise TypeError(
            "transition must be a UniformRemeshHistoryTransitionObservation"
        )
    if not transition.transition_observation_certified:
        raise TNFRValueError("transition is unsealed, tampered, or inconsistent")

    raw = _exact_vector(runtime_raw_head, "runtime_raw_head")
    bounded = _exact_vector(runtime_bounded_head, "runtime_bounded_head")
    scheduled = _exact_vector(scheduled_head, "scheduled_head")
    width = len(transition.exact_metric_weights)
    if any(len(item) != width for item in (raw, bounded, scheduled)):
        raise TNFRValueError("all heads must match the transition width")
    q = _exact_scalar(
        schedule_energy_gain_upper_bound,
        "schedule_energy_gain_upper_bound",
    )
    if q < 0:
        raise TNFRValueError(
            "schedule_energy_gain_upper_bound must be nonnegative"
        )

    metric = transition.exact_metric_weights
    raw_centered, raw_energy = _centered_energy(raw, metric)
    bounded_centered, bounded_energy = _centered_energy(bounded, metric)
    scheduled_centered, scheduled_energy = _centered_energy(scheduled, metric)
    if scheduled_energy > q * bounded_energy:
        raise TNFRValueError(
            "scheduled head violates schedule_energy_gain_upper_bound"
        )

    pi_zero = transition.certificate.stationary_distribution[0]
    ideal_energy = transition.exact_next_energy
    raw_spatial = raw_energy - ideal_energy
    clipping_spatial = bounded_energy - raw_energy
    schedule_spatial = scheduled_energy - bounded_energy
    total_spatial = scheduled_energy - ideal_energy
    raw_augmented = pi_zero * raw_spatial
    clipping_augmented = pi_zero * clipping_spatial
    schedule_augmented = pi_zero * schedule_spatial
    total_augmented = pi_zero * total_spatial
    augmented_after = transition.exact_augmented_energy_after + total_augmented
    energy_drop = transition.exact_augmented_energy_before - augmented_after

    slack = q * bounded_energy - scheduled_energy
    augmented_slack = pi_zero * slack
    contraction_margin = pi_zero * (Fraction(1) - q) * bounded_energy
    lower_bound = (
        transition.exact_jensen_dissipation
        - raw_augmented
        - clipping_augmented
        + contraction_margin
    )
    post_history = (scheduled,) + transition.exact_history[:-1]
    barycenter_before = transition.exact_stationary_history_barycenter
    barycenter_after = _weighted_history_field(
        transition.certificate.stationary_distribution,
        post_history,
    )
    barycenter_drift = _vector_subtract(barycenter_after, barycenter_before)

    conditions = (
        ("nested_remesh_transition_intact", True),
        ("heads_share_transition_width", True),
        (
            "spatial_energy_defects_telescope",
            total_spatial == raw_spatial + clipping_spatial + schedule_spatial,
        ),
        (
            "augmented_defects_are_pi0_weighted",
            (raw_augmented, clipping_augmented, schedule_augmented, total_augmented)
            == tuple(
                pi_zero * item
                for item in (
                    raw_spatial,
                    clipping_spatial,
                    schedule_spatial,
                    total_spatial,
                )
            ),
        ),
        (
            "augmented_energy_defects_telescope",
            total_augmented
            == raw_augmented + clipping_augmented + schedule_augmented,
        ),
        (
            "schedule_energy_gain_bound_satisfied",
            scheduled_energy <= q * bounded_energy,
        ),
        ("schedule_energy_gain_slack_nonnegative", slack >= 0),
        (
            "exact_augmented_energy_balance",
            energy_drop
            == transition.exact_jensen_dissipation
            - raw_augmented
            - clipping_augmented
            - schedule_augmented,
        ),
        (
            "gain_based_lower_bound_identity",
            lower_bound
            == transition.exact_jensen_dissipation
            - raw_augmented
            - clipping_augmented
            + contraction_margin,
        ),
        (
            "exact_drop_equals_lower_bound_plus_slack",
            energy_drop == lower_bound + augmented_slack,
        ),
        (
            "stationary_history_barycenter_drift_identity",
            barycenter_drift
            == _scale_vector(
                pi_zero,
                _vector_subtract(scheduled, transition.exact_next_field),
            ),
        ),
    )
    if not all(passed for _, passed in conditions):
        failed = ", ".join(name for name, passed in conditions if not passed)
        raise RuntimeError(f"REMESH/schedule history balance failed: {failed}")

    values: dict[str, Any] = {
        "transition": transition,
        "exact_runtime_raw_head": raw,
        "exact_runtime_bounded_head": bounded,
        "exact_scheduled_head": scheduled,
        "exact_runtime_raw_head_centered": raw_centered,
        "exact_runtime_bounded_head_centered": bounded_centered,
        "exact_scheduled_head_centered": scheduled_centered,
        "exact_runtime_raw_head_energy": raw_energy,
        "exact_runtime_bounded_head_energy": bounded_energy,
        "exact_scheduled_head_energy": scheduled_energy,
        "exact_schedule_energy_gain_upper_bound": q,
        "exact_schedule_energy_gain_slack": slack,
        "exact_schedule_augmented_energy_gain_slack": augmented_slack,
        "exact_raw_spatial_energy_defect": raw_spatial,
        "exact_clipping_spatial_energy_defect": clipping_spatial,
        "exact_schedule_spatial_energy_defect": schedule_spatial,
        "exact_total_spatial_energy_defect": total_spatial,
        "exact_raw_augmented_energy_defect": raw_augmented,
        "exact_clipping_augmented_energy_defect": clipping_augmented,
        "exact_schedule_augmented_energy_defect": schedule_augmented,
        "exact_total_augmented_energy_defect": total_augmented,
        "exact_augmented_energy_before": transition.exact_augmented_energy_before,
        "exact_augmented_energy_after": augmented_after,
        "exact_energy_drop": energy_drop,
        "exact_schedule_contraction_augmented_margin": contraction_margin,
        "exact_gain_based_energy_drop_lower_bound": lower_bound,
        "exact_stationary_history_barycenter_before": barycenter_before,
        "exact_stationary_history_barycenter_after": barycenter_after,
        "exact_stationary_history_barycenter_drift": barycenter_drift,
        "conditions": conditions,
    }
    result = RemeshScheduleHistoryStabilityObservation(
        **values,
        _proof_stamp=_proof_stamp_from_values(values),
    )
    if not result.transition_observation_certified:
        raise RuntimeError(
            "constructed REMESH/schedule history observation is inconsistent"
        )
    return result
