r"""Exact temporal stability for the uniform delayed REMESH recurrence.

For one fixed node support, one fixed positive diagonal spatial metric ``H``
and one uniform coefficient ``alpha``, this module studies the exact rational
recurrence

    x[k + 1] = beta x[k] + gamma x[k - tau_local]
                 + delta x[k - tau_global],

where ``beta = (1 - alpha)^2``, ``gamma = alpha (1 - alpha)`` and
``delta = alpha``.  Coincident delays are combined before the augmented
temporal companion is built.

The result is a theorem for the unclipped exact recurrence.  It does not
identify a binary64 runtime execution, allow changing coefficients or metric,
or infer spatial consensus or zero TNFR pressure.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass, field, fields, replace
from fractions import Fraction
from numbers import Integral, Rational, Real
from typing import Any

from ..errors import TNFRValueError
from ..utils._structural_signature import (
    proof_stamps_are_identical,
    structural_proof_signature,
)

__all__ = (
    "UniformRemeshHistoryStabilityCertificate",
    "UniformRemeshHistoryTransitionObservation",
    "certify_uniform_remesh_history_stability",
    "observe_uniform_remesh_history_transition",
)

ExactVector = tuple[Fraction, ...]
ExactMatrix = tuple[ExactVector, ...]
ExactHistory = tuple[ExactVector, ...]

_CERTIFICATE_PROOF_VERSION = "uniform_remesh_history_stability_v1"
_TRANSITION_PROOF_VERSION = "uniform_remesh_history_transition_v1"
_SCOPE = (
    "Exact rational stability for one fixed uniform-alpha, unclipped delayed "
    "REMESH recurrence on fixed ordered support and one fixed positive "
    "diagonal spatial metric. The temporal companion and its stationary "
    "measure prove a Jensen Lyapunov inequality. Strict temporal mixing for "
    "0 < alpha < 1 gives coordinatewise convergence to the stationary "
    "history barycenter. This does not certify a binary64 runtime, changing "
    "history rules, spatial consensus, DeltaNFR = 0, or full TNFR stability."
)


def _proof_stamp(
    value: Any,
    expected_type: type[Any],
    version: str,
) -> tuple[Any, ...]:
    if type(value) is not expected_type:
        raise TypeError("proof value must have its canonical result type")
    payload = tuple(
        (item.name, object.__getattribute__(value, item.name))
        for item in fields(expected_type)
        if item.name != "_proof_stamp"
    )
    return (version, structural_proof_signature(payload))


def _seal(value: Any, expected_type: type[Any], version: str) -> Any:
    return replace(
        value,
        _proof_stamp=_proof_stamp(value, expected_type, version),
    )


def _sealed(value: Any, expected_type: type[Any], version: str) -> bool:
    try:
        expected = _proof_stamp(value, expected_type, version)
        observed = object.__getattribute__(value, "_proof_stamp")
    except BaseException:
        return False
    return proof_stamps_are_identical(observed, expected)


def _strict_exact_vector(
    value: Any,
    *,
    width: int | None = None,
) -> bool:
    """Recognize the closed exact-vector representation without coercion."""

    return bool(
        type(value) is tuple
        and (width is None or len(value) == width)
        and all(type(item) is Fraction for item in value)
    )


def _strict_exact_history(
    value: Any,
    *,
    length: int | None = None,
    width: int | None = None,
) -> bool:
    """Recognize a nonempty rectangular exact history without dispatch."""

    return bool(
        type(value) is tuple
        and value
        and (length is None or len(value) == length)
        and all(_strict_exact_vector(row, width=width) for row in value)
    )


def _structural_values_are_identical(left: Any, right: Any) -> bool:
    """Compare closed proof values without caller-owned equality protocols."""

    try:
        return proof_stamps_are_identical(
            structural_proof_signature(left),
            structural_proof_signature(right),
        )
    except BaseException:
        return False


def _exact_scalar(value: Any, label: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TNFRValueError(f"{label} must be a finite real scalar")
    if isinstance(value, Rational):
        return Fraction(value)
    try:
        source_nonzero = bool(value != 0)
        floating = float(value)
    except (OverflowError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise TNFRValueError(f"{label} must be a finite real scalar") from exc
    if not math.isfinite(floating):
        raise TNFRValueError(f"{label} must be finite")
    if floating == 0.0 and source_nonzero:
        raise TNFRValueError(
            f"{label} contains a nonzero value below binary64 range"
        )
    return Fraction.from_float(floating)


def _positive_delay(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TNFRValueError(f"{label} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise TNFRValueError(f"{label} must be a positive integer")
    return result


def _materialize_iterable(value: Any, label: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise TNFRValueError(f"{label} must be an iterable sequence")
    try:
        return tuple(value)
    except (TypeError, ValueError) as exc:
        raise TNFRValueError(f"{label} must be an iterable sequence") from exc


def _exact_vector(value: Any, label: str) -> ExactVector:
    items = _materialize_iterable(value, label)
    if not items:
        raise TNFRValueError(f"{label} must be nonempty")
    return tuple(
        _exact_scalar(item, f"{label}[{index}]")
        for index, item in enumerate(items)
    )


def _node_order(nodes: Iterable[Hashable] | None, width: int) -> tuple[Hashable, ...]:
    if nodes is None:
        return tuple(range(width))
    ordered = _materialize_iterable(nodes, "nodes")
    if len(ordered) != width:
        raise TNFRValueError("nodes must match the spatial vector width")
    try:
        unique = set(ordered)
    except TypeError as exc:
        raise TNFRValueError("nodes must contain hashable identifiers") from exc
    if len(unique) != len(ordered):
        raise TNFRValueError("nodes must contain unique identifiers")
    try:
        tuple(structural_proof_signature(node) for node in ordered)
    except Exception as exc:
        raise TNFRValueError("nodes contain unreadable structural state") from exc
    return ordered


def _sealed_node_order_is_valid(value: Any, width: int) -> bool:
    """Recheck the public node-order contract without node hash/equality calls."""

    if type(value) is not tuple or len(value) != width:
        return False
    try:
        signatures = tuple(structural_proof_signature(node) for node in value)
        hash_slots = tuple(
            type.__getattribute__(type(node), "__hash__") for node in value
        )
    except BaseException:
        return False
    return bool(
        all(slot is not None for slot in hash_slots)
        and len(set(signatures)) == len(signatures)
    )


def _left_matrix_vector(vector: ExactVector, matrix: ExactMatrix) -> ExactVector:
    return tuple(
        sum(
            (vector[row] * matrix[row][column] for row in range(len(matrix))),
            Fraction(0),
        )
        for column in range(len(matrix))
    )


@dataclass(frozen=True, slots=True)
class _TemporalModel:
    beta: Fraction
    gamma: Fraction
    delta: Fraction
    combined_delay_coefficients: tuple[tuple[int, Fraction], ...]
    active_delays: tuple[int, ...]
    active_max_delay: int
    companion_matrix: ExactMatrix
    stationary_denominator: Fraction
    stationary_distribution: ExactVector
    conditions: tuple[tuple[str, bool], ...]


def _temporal_model(
    alpha: Fraction,
    tau_local: int,
    tau_global: int,
) -> _TemporalModel:
    one_minus = Fraction(1) - alpha
    beta = one_minus * one_minus
    gamma = alpha * one_minus
    delta = alpha
    by_delay: dict[int, Fraction] = {0: beta}
    by_delay[tau_local] = by_delay.get(tau_local, Fraction(0)) + gamma
    by_delay[tau_global] = by_delay.get(tau_global, Fraction(0)) + delta
    combined = tuple(
        (delay, coefficient)
        for delay, coefficient in sorted(by_delay.items())
        if coefficient > 0
    )
    active_delays = tuple(delay for delay, _ in combined)
    maximum = max(active_delays)
    coefficients = dict(combined)
    dimension = maximum + 1
    first_row = tuple(
        coefficients.get(delay, Fraction(0)) for delay in range(dimension)
    )
    companion = (first_row,) + tuple(
        tuple(
            Fraction(1) if column == row - 1 else Fraction(0)
            for column in range(dimension)
        )
        for row in range(1, dimension)
    )
    denominator = (
        Fraction(1) + gamma * tau_local + delta * tau_global
    )
    stationary = (Fraction(1, 1) / denominator,) + tuple(
        (
            (gamma if index <= tau_local else Fraction(0))
            + (delta if index <= tau_global else Fraction(0))
        )
        / denominator
        for index in range(1, dimension)
    )
    row_stochastic = all(
        all(entry >= 0 for entry in row)
        and sum(row, Fraction(0)) == 1
        for row in companion
    )
    stationary_invariant = (
        _left_matrix_vector(stationary, companion) == stationary
    )
    strict_mixing = Fraction(0) < alpha < Fraction(1)
    conditions = (
        ("canonical_coefficient_partition", beta + gamma + delta == 1),
        (
            "canonical_coefficients_nonnegative",
            beta >= 0 and gamma >= 0 and delta >= 0,
        ),
        (
            "coincident_delays_combined",
            len(active_delays) == len(set(active_delays)),
        ),
        (
            "active_delay_support_exact",
            bool(combined)
            and all(coefficient > 0 for _, coefficient in combined)
            and sum((coefficient for _, coefficient in combined), Fraction(0))
            == 1,
        ),
        ("temporal_companion_row_stochastic", row_stochastic),
        (
            "stationary_distribution_normalized_positive",
            bool(stationary)
            and all(entry > 0 for entry in stationary)
            and sum(stationary, Fraction(0)) == 1,
        ),
        ("stationary_distribution_left_invariant", stationary_invariant),
        (
            "strict_mixing_companion_primitive",
            not strict_mixing
            or (
                beta > 0
                and maximum > 0
                and coefficients[maximum] > 0
            ),
        ),
        (
            "alpha_zero_identity_map",
            alpha != 0
            or (combined == ((0, Fraction(1)),) and maximum == 0),
        ),
        (
            "alpha_one_pure_delay_map",
            alpha != 1
            or (
                combined == ((tau_global, Fraction(1)),)
                and maximum == tau_global
            ),
        ),
    )
    return _TemporalModel(
        beta=beta,
        gamma=gamma,
        delta=delta,
        combined_delay_coefficients=combined,
        active_delays=active_delays,
        active_max_delay=maximum,
        companion_matrix=companion,
        stationary_denominator=denominator,
        stationary_distribution=stationary,
        conditions=conditions,
    )


@dataclass(frozen=True, slots=True)
class UniformRemeshHistoryStabilityCertificate:
    """Sealed exact theorem data for one uniform delayed recurrence."""

    alpha: Fraction
    tau_local: int
    tau_global: int
    beta: Fraction
    gamma: Fraction
    delta: Fraction
    combined_delay_coefficients: tuple[tuple[int, Fraction], ...]
    active_delays: tuple[int, ...]
    active_max_delay: int
    companion_matrix: ExactMatrix
    stationary_denominator: Fraction
    stationary_distribution: ExactVector
    conditions: tuple[tuple[str, bool], ...]
    _proof_stamp: tuple[Any, ...] = field(default=(), repr=False, compare=False)

    def _proof_fields_are_intact(self) -> bool:
        try:
            # Authenticate the immutable payload before performing arithmetic
            # on any field.  A caller can bypass ``frozen=True`` through
            # ``object.__setattr__``; an untrusted replacement must therefore
            # be rejected without invoking its numeric protocol.
            if not _sealed(
                self,
                UniformRemeshHistoryStabilityCertificate,
                _CERTIFICATE_PROOF_VERSION,
            ):
                return False
            if (
                type(self.alpha) is not Fraction
                or not Fraction(0) <= self.alpha <= Fraction(1)
                or type(self.tau_local) is not int
                or self.tau_local <= 0
                or type(self.tau_global) is not int
                or self.tau_global <= 0
            ):
                return False
            model = _temporal_model(
                self.alpha,
                self.tau_local,
                self.tau_global,
            )
            observed_payload = tuple(
                (
                    item.name,
                    object.__getattribute__(self, item.name),
                )
                for item in fields(_TemporalModel)
            )
            expected_payload = tuple(
                (
                    item.name,
                    object.__getattribute__(model, item.name),
                )
                for item in fields(_TemporalModel)
            )
            return _structural_values_are_identical(
                observed_payload,
                expected_payload,
            )
        except BaseException:
            return False

    @property
    def scope(self) -> str:
        return _SCOPE

    @property
    def stability_certificate_certified(self) -> bool:
        return bool(
            self._proof_fields_are_intact()
            and all(passed for _, passed in self.conditions)
        )

    @property
    def failed_conditions(self) -> tuple[str, ...]:
        if not self._proof_fields_are_intact():
            return ("remesh_history_proof_fields_intact",)
        return tuple(name for name, passed in self.conditions if not passed)

    @property
    def companion_row_stochastic_certified(self) -> bool:
        return self.stability_certificate_certified

    @property
    def stationary_distribution_certified(self) -> bool:
        return self.stability_certificate_certified

    @property
    def jensen_lyapunov_nonincrease_certified(self) -> bool:
        return self.stability_certificate_certified

    @property
    def equality_characterization_certified(self) -> bool:
        return self.stability_certificate_certified

    @property
    def alpha_zero_identity_map_certified(self) -> bool:
        return bool(self.stability_certificate_certified and self.alpha == 0)

    @property
    def alpha_one_pure_delay_map_certified(self) -> bool:
        return bool(self.stability_certificate_certified and self.alpha == 1)

    @property
    def alpha_one_augmented_energy_conservation_certified(self) -> bool:
        return self.alpha_one_pure_delay_map_certified

    @property
    def pure_delay_period(self) -> int | None:
        """Return the companion permutation order ``m + 1`` for ``alpha = 1``.

        A particular history orbit can have any least period dividing this
        order, including period one for a temporally constant history.
        """

        if not self.alpha_one_pure_delay_map_certified:
            return None
        return self.active_max_delay + 1

    @property
    def periodic_temporal_cycles_possible(self) -> bool:
        return self.alpha_one_pure_delay_map_certified

    @property
    def strict_mixing_companion_primitive_certified(self) -> bool:
        return bool(
            self.stability_certificate_certified
            and Fraction(0) < self.alpha < Fraction(1)
        )

    @property
    def strict_mixing_pointwise_temporal_convergence_certified(self) -> bool:
        return self.strict_mixing_companion_primitive_certified

    @property
    def spatial_consensus_certified(self) -> bool:
        return False

    @property
    def zero_pressure_equilibrium_certified(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class UniformRemeshHistoryTransitionObservation:
    """One exact transition and its stationary-weighted Jensen balance."""

    certificate: UniformRemeshHistoryStabilityCertificate
    nodes: tuple[Hashable, ...]
    exact_metric_weights: ExactVector
    exact_history: ExactHistory
    exact_centered_history: ExactHistory
    exact_history_energies: ExactVector
    exact_next_field: ExactVector
    exact_next_centered_field: ExactVector
    exact_next_energy: Fraction
    exact_augmented_energy_before: Fraction
    exact_augmented_energy_after: Fraction
    exact_energy_drop: Fraction
    exact_jensen_dissipation: Fraction
    active_centered_fields: tuple[tuple[int, ExactVector], ...]
    active_centered_fields_pairwise_equal: bool
    lyapunov_nonincreasing: bool
    lyapunov_equality: bool
    exact_stationary_history_barycenter: ExactVector
    exact_post_transition_stationary_history_barycenter: ExactVector
    exact_strict_mixing_temporal_limit: ExactVector | None
    conditions: tuple[tuple[str, bool], ...]
    _proof_stamp: tuple[Any, ...] = field(default=(), repr=False, compare=False)

    def _proof_fields_are_intact(self) -> bool:
        try:
            certificate = self.certificate
            if type(certificate) is not UniformRemeshHistoryStabilityCertificate:
                return False
            if not _sealed(
                self,
                UniformRemeshHistoryTransitionObservation,
                _TRANSITION_PROOF_VERSION,
            ):
                return False
            if not certificate.stability_certificate_certified:
                return False
            nodes = self.nodes
            metric = self.exact_metric_weights
            history = self.exact_history
            if type(nodes) is not tuple or not nodes:
                return False
            width = len(nodes)
            if (
                not _sealed_node_order_is_valid(nodes, width)
                or not _strict_exact_vector(metric, width=width)
                or not metric
                or any(weight <= 0 for weight in metric)
                or not _strict_exact_history(
                    history,
                    length=certificate.active_max_delay + 1,
                    width=width,
                )
            ):
                return False
            expected = _derive_transition_model(
                certificate,
                history,
                metric,
            )
            observed_payload = tuple(
                (
                    item.name,
                    object.__getattribute__(self, item.name),
                )
                for item in fields(_TransitionModel)
            )
            expected_payload = tuple(
                (
                    item.name,
                    object.__getattribute__(expected, item.name),
                )
                for item in fields(_TransitionModel)
            )
            return _structural_values_are_identical(
                observed_payload,
                expected_payload,
            )
        except BaseException:
            return False

    @property
    def scope(self) -> str:
        return _SCOPE

    @property
    def transition_observation_certified(self) -> bool:
        return bool(
            self._proof_fields_are_intact()
            and all(passed for _, passed in self.conditions)
        )

    @property
    def failed_conditions(self) -> tuple[str, ...]:
        if not self._proof_fields_are_intact():
            return ("remesh_history_transition_proof_fields_intact",)
        return tuple(name for name, passed in self.conditions if not passed)

    @property
    def exact_dissipation_identity_certified(self) -> bool:
        return self.transition_observation_certified

    @property
    def equality_iff_active_centered_fields_agree_certified(self) -> bool:
        return self.transition_observation_certified

    @property
    def spatial_consensus_certified(self) -> bool:
        return False

    @property
    def zero_pressure_equilibrium_certified(self) -> bool:
        return False


def certify_uniform_remesh_history_stability(
    *,
    alpha: Real,
    tau_local: Integral,
    tau_global: Integral,
) -> UniformRemeshHistoryStabilityCertificate:
    """Build the exact companion/stationary-measure stability certificate."""

    alpha_q = _exact_scalar(alpha, "alpha")
    if not Fraction(0) <= alpha_q <= Fraction(1):
        raise TNFRValueError("alpha must be in [0, 1]")
    local = _positive_delay(tau_local, "tau_local")
    global_ = _positive_delay(tau_global, "tau_global")
    model = _temporal_model(alpha_q, local, global_)
    value = UniformRemeshHistoryStabilityCertificate(
        alpha=alpha_q,
        tau_local=local,
        tau_global=global_,
        beta=model.beta,
        gamma=model.gamma,
        delta=model.delta,
        combined_delay_coefficients=model.combined_delay_coefficients,
        active_delays=model.active_delays,
        active_max_delay=model.active_max_delay,
        companion_matrix=model.companion_matrix,
        stationary_denominator=model.stationary_denominator,
        stationary_distribution=model.stationary_distribution,
        conditions=model.conditions,
    )
    result = _seal(
        value,
        UniformRemeshHistoryStabilityCertificate,
        _CERTIFICATE_PROOF_VERSION,
    )
    if not result.stability_certificate_certified:
        raise RuntimeError("constructed REMESH history certificate is inconsistent")
    return result


def _centered_energy(
    field_value: ExactVector,
    metric: ExactVector,
) -> tuple[ExactVector, Fraction]:
    total = sum(metric, Fraction(0))
    mean = sum(
        (
            weight * value
            for weight, value in zip(metric, field_value, strict=True)
        ),
        Fraction(0),
    ) / total
    centered = tuple(value - mean for value in field_value)
    energy = sum(
        (
            weight * value * value
            for weight, value in zip(metric, centered, strict=True)
        ),
        Fraction(0),
    ) / 2
    return centered, energy


def _weighted_history_field(
    weights: ExactVector,
    history: ExactHistory,
) -> ExactVector:
    return tuple(
        sum(
            (
                temporal_weight * history[index][coordinate]
                for index, temporal_weight in enumerate(weights)
            ),
            Fraction(0),
        )
        for coordinate in range(len(history[0]))
    )


def _pairwise_active_fields_equal(
    active: tuple[tuple[int, ExactVector], ...],
) -> bool:
    return all(
        left_field == right_field
        for left_index, (_, left_field) in enumerate(active)
        for _, right_field in active[left_index + 1 :]
    )


@dataclass(frozen=True, slots=True)
class _TransitionModel:
    exact_centered_history: ExactHistory
    exact_history_energies: ExactVector
    exact_next_field: ExactVector
    exact_next_centered_field: ExactVector
    exact_next_energy: Fraction
    exact_augmented_energy_before: Fraction
    exact_augmented_energy_after: Fraction
    exact_energy_drop: Fraction
    exact_jensen_dissipation: Fraction
    active_centered_fields: tuple[tuple[int, ExactVector], ...]
    active_centered_fields_pairwise_equal: bool
    lyapunov_nonincreasing: bool
    lyapunov_equality: bool
    exact_stationary_history_barycenter: ExactVector
    exact_post_transition_stationary_history_barycenter: ExactVector
    exact_strict_mixing_temporal_limit: ExactVector | None
    conditions: tuple[tuple[str, bool], ...]


def _derive_transition_model(
    certificate: UniformRemeshHistoryStabilityCertificate,
    exact_history: ExactHistory,
    metric: ExactVector,
) -> _TransitionModel:
    """Derive every transition field from its canonical mathematical inputs."""

    width = len(metric)
    coefficients = certificate.combined_delay_coefficients
    next_field = tuple(
        sum(
            (
                coefficient * exact_history[delay][coordinate]
                for delay, coefficient in coefficients
            ),
            Fraction(0),
        )
        for coordinate in range(width)
    )
    centered_and_energy = tuple(
        _centered_energy(row, metric) for row in exact_history
    )
    centered_history = tuple(item[0] for item in centered_and_energy)
    history_energies = tuple(item[1] for item in centered_and_energy)
    next_centered, next_energy = _centered_energy(next_field, metric)
    temporal_weights = certificate.stationary_distribution
    before = sum(
        (
            weight * energy
            for weight, energy in zip(
                temporal_weights,
                history_energies,
                strict=True,
            )
        ),
        Fraction(0),
    )
    post_energies = (next_energy,) + history_energies[:-1]
    after = sum(
        (
            weight * energy
            for weight, energy in zip(
                temporal_weights,
                post_energies,
                strict=True,
            )
        ),
        Fraction(0),
    )
    active = tuple(
        (delay, centered_history[delay]) for delay, _ in coefficients
    )
    dissipation = Fraction(0)
    pi_zero = temporal_weights[0]
    for left_index, (left_delay, left_coefficient) in enumerate(coefficients):
        left = centered_history[left_delay]
        for right_delay, right_coefficient in coefficients[left_index + 1 :]:
            right = centered_history[right_delay]
            squared_distance = sum(
                (
                    weight * (left_value - right_value) ** 2
                    for weight, left_value, right_value in zip(
                        metric,
                        left,
                        right,
                        strict=True,
                    )
                ),
                Fraction(0),
            )
            dissipation += (
                pi_zero
                * left_coefficient
                * right_coefficient
                * squared_distance
                / 2
            )
    drop = before - after
    active_equal = _pairwise_active_fields_equal(active)
    stationary_barycenter = _weighted_history_field(
        temporal_weights,
        exact_history,
    )
    post_history = (next_field,) + exact_history[:-1]
    post_barycenter = _weighted_history_field(temporal_weights, post_history)
    strict_limit = (
        stationary_barycenter
        if Fraction(0) < certificate.alpha < Fraction(1)
        else None
    )
    weighted_centered_next = tuple(
        sum(
            (
                coefficient * centered_history[delay][coordinate]
                for delay, coefficient in coefficients
            ),
            Fraction(0),
        )
        for coordinate in range(width)
    )
    conditions = (
        (
            "center_projection_commutes_with_remesh",
            next_centered == weighted_centered_next,
        ),
        (
            "stationary_history_barycenter_preserved",
            stationary_barycenter == post_barycenter,
        ),
        ("exact_jensen_dissipation_identity", drop == dissipation),
        ("augmented_energy_nonincreasing", drop >= 0),
        (
            "equality_iff_active_centered_fields_agree",
            (drop == 0) == active_equal,
        ),
    )
    return _TransitionModel(
        exact_centered_history=centered_history,
        exact_history_energies=history_energies,
        exact_next_field=next_field,
        exact_next_centered_field=next_centered,
        exact_next_energy=next_energy,
        exact_augmented_energy_before=before,
        exact_augmented_energy_after=after,
        exact_energy_drop=drop,
        exact_jensen_dissipation=dissipation,
        active_centered_fields=active,
        active_centered_fields_pairwise_equal=active_equal,
        lyapunov_nonincreasing=drop >= 0,
        lyapunov_equality=drop == 0,
        exact_stationary_history_barycenter=stationary_barycenter,
        exact_post_transition_stationary_history_barycenter=post_barycenter,
        exact_strict_mixing_temporal_limit=strict_limit,
        conditions=conditions,
    )


def observe_uniform_remesh_history_transition(
    certificate: UniformRemeshHistoryStabilityCertificate,
    history: Iterable[Iterable[Real]],
    metric_weights: Iterable[Real],
    *,
    nodes: Iterable[Hashable] | None = None,
) -> UniformRemeshHistoryTransitionObservation:
    """Observe one exact recurrence step from ``(x[k], ..., x[k-m])``.

    The history must have exactly ``m + 1`` equally shaped spatial fields,
    where ``m`` is the maximum delay with a positive combined coefficient.
    """

    if type(certificate) is not UniformRemeshHistoryStabilityCertificate:
        raise TypeError(
            "certificate must be a UniformRemeshHistoryStabilityCertificate"
        )
    if not certificate.stability_certificate_certified:
        raise TNFRValueError("certificate is unsealed, tampered, or inconsistent")
    history_items = _materialize_iterable(history, "history")
    required = certificate.active_max_delay + 1
    if len(history_items) != required:
        raise TNFRValueError(
            f"history must contain exactly {required} fields ordered newest first"
        )
    exact_history = tuple(
        _exact_vector(row, f"history[{index}]")
        for index, row in enumerate(history_items)
    )
    width = len(exact_history[0])
    if any(len(row) != width for row in exact_history):
        raise TNFRValueError("history fields must have one common spatial shape")
    node_order = _node_order(nodes, width)
    metric = _exact_vector(metric_weights, "metric_weights")
    if len(metric) != width:
        raise TNFRValueError("metric_weights must match the spatial vector width")
    if any(weight <= 0 for weight in metric):
        raise TNFRValueError("metric_weights must be strictly positive")

    model = _derive_transition_model(certificate, exact_history, metric)
    value = UniformRemeshHistoryTransitionObservation(
        certificate=certificate,
        nodes=node_order,
        exact_metric_weights=metric,
        exact_history=exact_history,
        exact_centered_history=model.exact_centered_history,
        exact_history_energies=model.exact_history_energies,
        exact_next_field=model.exact_next_field,
        exact_next_centered_field=model.exact_next_centered_field,
        exact_next_energy=model.exact_next_energy,
        exact_augmented_energy_before=model.exact_augmented_energy_before,
        exact_augmented_energy_after=model.exact_augmented_energy_after,
        exact_energy_drop=model.exact_energy_drop,
        exact_jensen_dissipation=model.exact_jensen_dissipation,
        active_centered_fields=model.active_centered_fields,
        active_centered_fields_pairwise_equal=(
            model.active_centered_fields_pairwise_equal
        ),
        lyapunov_nonincreasing=model.lyapunov_nonincreasing,
        lyapunov_equality=model.lyapunov_equality,
        exact_stationary_history_barycenter=(
            model.exact_stationary_history_barycenter
        ),
        exact_post_transition_stationary_history_barycenter=(
            model.exact_post_transition_stationary_history_barycenter
        ),
        exact_strict_mixing_temporal_limit=(
            model.exact_strict_mixing_temporal_limit
        ),
        conditions=model.conditions,
    )
    result = _seal(
        value,
        UniformRemeshHistoryTransitionObservation,
        _TRANSITION_PROOF_VERSION,
    )
    if not result.transition_observation_certified:
        raise RuntimeError("constructed REMESH history transition is inconsistent")
    return result
