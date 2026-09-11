r"""Finite causal runtime verification of the common-``q`` defect theorem.

The pure relative-defect certificate assumes that every REMESH/schedule
boundary obeys

``E_H(z_j) - E_H(y_j) <= eta * J_j`` and ``q_j <= q``,

where ``y_j`` is the ideal REMESH head, ``z_j`` is its bounded binary64
realization and ``J_j`` is the Jensen head budget computed from the incoming
history energies.  This module checks those hypotheses on one nonempty,
contiguous block selected from an executor-owned runtime telescope.  It then
checks the exact entrywise envelope and the finite endpoint estimate

``V_after <= q_eff**floor(n / L) * V_before``.

All binary64 values enter through already rationalized runtime observations.
The result concerns only the selected, causally executed block.  It does not
establish a forward-invariant runtime class, repeated or future binary64
stability, solver accuracy or order, or full TNFR stability.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from fractions import Fraction
from typing import Any

from ..errors import TNFRValueError
from ..operators.event_remesh_causal_runtime import (
    ExecutedEventRemeshCycleSequence,
)
from ..utils._structural_signature import proof_stamps_are_identical
from ._exact_linear_algebra import ExactSquareMatrix
from .remesh_history_stability import ExactVector
from .remesh_schedule_relative_defect_stability import (
    UniformRemeshScheduleRelativeDefectStabilityCertificate,
)
from .runtime_remesh_schedule_block_margin import (
    RuntimeRemeshScheduleBlockMarginObservation,
    _execution_is_intact,
    _observe_executed_event_remesh_block_margin_after_source_validation,
)
from .runtime_remesh_schedule_stability import (
    RuntimeRemeshScheduleBoundaryObservation,
)

__all__ = (
    "RuntimeRemeshScheduleRelativeDefectBlockObservation",
    "observe_executed_event_remesh_relative_defect_block",
)


_PROOF_VERSION = "runtime_remesh_schedule_relative_defect_block_v1"
_SCOPE = (
    "One exact, nonempty contiguous block selected from one intact causal "
    "event/REMESH execution. The observation verifies, at every recorded "
    "boundary, the common-q schedule hypothesis, the declared relative "
    "pre-schedule spatial-energy defect bound and the q_eff companion "
    "envelope, then verifies its finite endpoint bound. It binds the source "
    "execution, block observation, selected boundaries and pure certificate "
    "by process-local identity. It does not establish a forward-invariant "
    "runtime class, repeated or future binary64 stability, a global runtime "
    "gain, solver accuracy or order, mesh convergence, adaptive grammar or "
    "full TNFR stability."
)
_CONDITION_NAMES = (
    "source_execution_intact",
    "source_causal_execution_provenance_intact",
    "relative_defect_certificate_intact",
    "runtime_block_selection_and_provenance_intact",
    "runtime_block_objects_bound_by_identity",
    "fixed_remesh_model_matches_certificate",
    "fixed_exact_metric_and_history_dimension",
    "every_boundary_energy_telemetry_exact_nonnegative",
    "every_remesh_input_budget_matches_companion_head",
    "every_ideal_remesh_head_satisfies_jensen_bound",
    "every_pre_schedule_energy_defect_identity",
    "every_pre_schedule_relative_energy_defect_bound",
    "every_runtime_schedule_gain_bounded_by_common_q",
    "every_recorded_schedule_head_satisfies_its_gain_bound",
    "every_effective_head_energy_bound",
    "every_history_energy_vector_satisfies_effective_envelope",
    "recorded_history_energy_vectors_advance_contiguously",
    "augmented_energies_match_stationary_history_weights",
    "finite_endpoint_gain_matches_exact_theorem",
    "finite_endpoint_energy_bound_satisfied",
)


def _raw_proof_stamp(value: Any) -> tuple[Any, ...] | None:
    try:
        stamp = object.__getattribute__(value, "_proof_stamp")
    except BaseException:
        return None
    return stamp if type(stamp) is tuple else None


def _certificate_is_intact(value: Any) -> bool:
    if type(value) is not UniformRemeshScheduleRelativeDefectStabilityCertificate:
        return False
    try:
        return bool(
            UniformRemeshScheduleRelativeDefectStabilityCertificate
            ._proof_fields_are_intact(value)
            is True
        )
    except BaseException:
        return False


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


def _strict_exact_vector(value: Any, dimension: int) -> bool:
    return bool(
        type(value) is tuple
        and len(value) == dimension
        and all(type(item) is Fraction for item in value)
    )


def _strict_exact_matrix(value: Any, dimension: int) -> bool:
    return bool(
        type(value) is tuple
        and len(value) == dimension
        and all(_strict_exact_vector(row, dimension) for row in value)
    )


def _matrix_vector_action(
    matrix: ExactSquareMatrix,
    vector: ExactVector,
) -> ExactVector:
    dimension = len(vector)
    if not _strict_exact_matrix(matrix, dimension):
        raise TNFRValueError("energy-envelope matrix is not exact and square")
    if not _strict_exact_vector(vector, dimension):
        raise TNFRValueError("history-energy vector is not exact and complete")
    return tuple(
        sum(
            (
                matrix[row][column] * vector[column]
                for column in range(dimension)
            ),
            Fraction(0),
        )
        for row in range(dimension)
    )


def _weighted_energy(
    weights: ExactVector,
    energies: ExactVector,
) -> Fraction:
    if not _strict_exact_vector(weights, len(energies)):
        raise TNFRValueError("stationary history weights are invalid")
    return sum(
        (
            weight * energy
            for weight, energy in zip(weights, energies, strict=True)
        ),
        Fraction(0),
    )


def _remesh_models_match(left: Any, right: Any) -> bool:
    """Compare validated REMESH models without relying on object identity."""

    try:
        return bool(
            type(left.alpha) is Fraction
            and type(right.alpha) is Fraction
            and left.alpha == right.alpha
            and type(left.tau_local) is int
            and type(right.tau_local) is int
            and left.tau_local == right.tau_local
            and type(left.tau_global) is int
            and type(right.tau_global) is int
            and left.tau_global == right.tau_global
            and left.combined_delay_coefficients
            == right.combined_delay_coefficients
            and left.companion_matrix == right.companion_matrix
            and left.stationary_distribution == right.stationary_distribution
        )
    except BaseException:
        return False


def _observation_values(
    value: "RuntimeRemeshScheduleRelativeDefectBlockObservation",
) -> dict[str, Any]:
    if type(value) is not RuntimeRemeshScheduleRelativeDefectBlockObservation:
        raise TypeError("observation must have its canonical result type")
    return {
        item.name: object.__getattribute__(value, item.name)
        for item in fields(RuntimeRemeshScheduleRelativeDefectBlockObservation)
        if item.name != "_proof_stamp"
    }


def _proof_stamp_from_values(values: dict[str, Any]) -> tuple[Any, ...]:
    source = values["source_execution"]
    certificate = values["relative_defect_certificate"]
    block = values["block_observation"]
    boundaries = values["boundaries"]
    boundary_tokens = (
        tuple(
            ("runtime-boundary", id(item), _raw_proof_stamp(item))
            for item in boundaries
        )
        if type(boundaries) is tuple
        else ("invalid-boundary-container",)
    )
    telemetry_names = tuple(
        item.name
        for item in fields(RuntimeRemeshScheduleRelativeDefectBlockObservation)
        if item.name
        not in {
            "source_execution",
            "relative_defect_certificate",
            "block_observation",
            "boundaries",
            "_proof_stamp",
        }
    )
    return (
        _PROOF_VERSION,
        ("source-execution", id(source), _raw_proof_stamp(source)),
        (
            "relative-defect-certificate",
            id(certificate),
            _raw_proof_stamp(certificate),
        ),
        ("runtime-block", id(block), _raw_proof_stamp(block)),
        boundary_tokens,
        tuple((name, values[name]) for name in telemetry_names),
    )


def _seal(
    value: "RuntimeRemeshScheduleRelativeDefectBlockObservation",
) -> "RuntimeRemeshScheduleRelativeDefectBlockObservation":
    """Seal current fields; semantic rederivation remains authoritative."""

    values = _observation_values(value)
    return replace(value, _proof_stamp=_proof_stamp_from_values(values))


def _derive_values(
    execution: ExecutedEventRemeshCycleSequence,
    certificate: UniformRemeshScheduleRelativeDefectStabilityCertificate,
    start_boundary: int,
    boundary_count: int | None,
    *,
    block_override: RuntimeRemeshScheduleBlockMarginObservation | None = None,
    source_already_validated: bool = False,
    certificate_already_validated: bool = False,
    block_already_validated: bool = False,
) -> dict[str, Any]:
    if type(execution) is not ExecutedEventRemeshCycleSequence:
        raise TypeError(
            "execution must be an ExecutedEventRemeshCycleSequence"
        )
    if (
        type(certificate)
        is not UniformRemeshScheduleRelativeDefectStabilityCertificate
    ):
        raise TypeError(
            "certificate must be a "
            "UniformRemeshScheduleRelativeDefectStabilityCertificate"
        )
    if not source_already_validated and not _execution_is_intact(execution):
        raise TNFRValueError(
            "executed cycle sequence is unsealed, tampered, or inconsistent"
        )
    if (
        not certificate_already_validated
        and not _certificate_is_intact(certificate)
    ):
        raise TNFRValueError(
            "relative-defect certificate is unsealed, tampered, or inconsistent"
        )

    if block_override is None:
        block = (
            _observe_executed_event_remesh_block_margin_after_source_validation(
                execution,
                start_boundary=start_boundary,
                boundary_count=boundary_count,
            )
        )
    else:
        block = block_override
        if (
            type(block) is not RuntimeRemeshScheduleBlockMarginObservation
            or block.source_execution is not execution
            or block.start_boundary != start_boundary
            or block.boundary_count != boundary_count
        ):
            raise TNFRValueError(
                "stored runtime block is not the selected intact causal block"
            )
        if (
            not block_already_validated
            and not block._proof_fields_are_intact_after_source_validation()
        ):
            raise TNFRValueError(
                "stored runtime block is not the selected intact causal block"
            )

    boundaries = block.boundaries
    if (
        type(boundaries) is not tuple
        or not boundaries
        or any(
            type(item) is not RuntimeRemeshScheduleBoundaryObservation
            for item in boundaries
        )
    ):
        raise TNFRValueError("runtime block has no intact selected boundary")

    remesh = certificate.remesh_certificate
    eta = certificate.pre_schedule_relative_energy_defect_upper_bound
    q = certificate.schedule_energy_gain_upper_bound
    q_eff = certificate.effective_head_energy_gain_upper_bound
    dimension = certificate.history_length
    horizon = certificate.universal_block_horizon
    companion = certificate.remesh_companion_matrix
    envelope_matrix = certificate.effective_head_energy_domination_matrix
    stationary = remesh.stationary_distribution
    coefficients = remesh.combined_delay_coefficients
    common_metric = boundaries[0].exact_common_normalized_metric
    if (
        type(eta) is not Fraction
        or type(q) is not Fraction
        or type(q_eff) is not Fraction
        or type(dimension) is not int
        or type(horizon) is not int
        or dimension <= 0
        or horizon != dimension
        or not _strict_exact_matrix(companion, dimension)
        or not _strict_exact_matrix(envelope_matrix, dimension)
        or not _strict_exact_vector(stationary, dimension)
        or any(weight <= 0 for weight in stationary)
        or sum(stationary, Fraction(0)) != 1
        or not _strict_exact_vector(common_metric, len(common_metric))
        or not common_metric
        or any(weight <= 0 for weight in common_metric)
        or sum(common_metric, Fraction(0)) != 1
    ):
        raise TNFRValueError(
            "relative-defect certificate or runtime metric has invalid exact dimensions"
        )

    input_vectors: list[ExactVector] = []
    output_vectors: list[ExactVector] = []
    budgets: list[Fraction] = []
    ideal_energies: list[Fraction] = []
    bounded_energies: list[Fraction] = []
    defects: list[Fraction] = []
    ratios: list[Fraction | None] = []
    slacks: list[Fraction] = []
    schedule_gains: list[Fraction] = []
    effective_head_bounds: list[Fraction] = []
    scheduled_energies: list[Fraction] = []
    envelope_vectors: list[ExactVector] = []

    remesh_matches: list[bool] = []
    exact_telemetry: list[bool] = []
    budget_identities: list[bool] = []
    jensen_bounds: list[bool] = []
    defect_identities: list[bool] = []
    relative_bounds: list[bool] = []
    gain_bounds: list[bool] = []
    schedule_head_bounds: list[bool] = []
    effective_bounds: list[bool] = []
    entrywise_envelopes: list[bool] = []
    augmented_matches: list[bool] = []

    for boundary in boundaries:
        transition = boundary.exact_transition
        schedule_balance = boundary.schedule_balance
        runtime_remesh = transition.certificate
        remesh_matches.append(_remesh_models_match(runtime_remesh, remesh))

        energies = transition.exact_history_energies
        if not _strict_exact_vector(energies, dimension):
            raise TNFRValueError(
                "runtime transition history energies are not exact and complete"
            )
        if any(energy < 0 for energy in energies):
            raise TNFRValueError("runtime transition history energy is negative")
        input_vectors.append(energies)

        try:
            budget = sum(
                (
                    coefficient * energies[delay]
                    for delay, coefficient in coefficients
                ),
                Fraction(0),
            )
        except (IndexError, TypeError) as exc:
            raise TNFRValueError(
                "REMESH delay coefficients do not address the history energy vector"
            ) from exc
        ideal = transition.exact_next_energy
        # The bridge's scalar energy uses its input metric normalization.  The
        # schedule balance has already recomputed the same bounded field in the
        # telescope's authoritative common normalized metric, which is also the
        # metric used by ``ideal``, ``scheduled`` and every history energy here.
        bounded = schedule_balance.exact_runtime_bounded_head_energy
        scheduled = schedule_balance.exact_scheduled_head_energy
        q_j = boundary.schedule_composition.exact_energy_gain_upper_bound
        scalar_values = (budget, ideal, bounded, scheduled, q_j)
        scalar_telemetry_exact = all(
            type(value) is Fraction for value in scalar_values
        )
        if not scalar_telemetry_exact:
            raise TNFRValueError("runtime boundary energy telemetry must be exact")
        if any(value < 0 for value in scalar_values):
            raise TNFRValueError(
                "runtime boundary energies and schedule gain must be nonnegative"
            )

        defect = bounded - ideal
        slack = eta * budget - defect
        ratio = None if budget == 0 else defect / budget
        output = (scheduled,) + energies[:-1]
        remesh_action = _matrix_vector_action(companion, energies)
        envelope = _matrix_vector_action(envelope_matrix, energies)
        effective_bound = q_eff * budget

        budgets.append(budget)
        ideal_energies.append(ideal)
        bounded_energies.append(bounded)
        defects.append(defect)
        ratios.append(ratio)
        slacks.append(slack)
        schedule_gains.append(q_j)
        effective_head_bounds.append(effective_bound)
        scheduled_energies.append(scheduled)
        output_vectors.append(output)
        envelope_vectors.append(envelope)

        exact_telemetry.append(scalar_telemetry_exact)
        budget_identities.append(
            budget == remesh_action[0]
            and budget
            == sum(
                (
                    coefficient * energies[delay]
                    for delay, coefficient in coefficients
                ),
                Fraction(0),
            )
        )
        jensen_bounds.append(ideal <= budget)
        defect_identities.append(
            defect == bounded - ideal
            and slack == eta * budget - defect
            and ratio == (None if budget == 0 else defect / budget)
        )
        relative_bounds.append(defect <= eta * budget)
        gain_bounds.append(q_j <= q)
        schedule_head_bounds.append(scheduled <= q_j * bounded)
        effective_bounds.append(
            effective_bound == q_eff * budget
            and q_eff == q * (Fraction(1) + eta)
            and scheduled <= effective_bound
        )
        entrywise_envelopes.append(
            all(
                observed <= upper
                for observed, upper in zip(output, envelope, strict=True)
            )
        )
        augmented_matches.append(
            _weighted_energy(stationary, energies)
            == boundary.exact_augmented_energy_before
            and _weighted_energy(stationary, output)
            == boundary.exact_augmented_energy_after
        )

    input_tuple = tuple(input_vectors)
    output_tuple = tuple(output_vectors)
    continuous = all(
        left == right
        for left, right in zip(
            output_tuple[:-1],
            input_tuple[1:],
            strict=True,
        )
    )
    before = block.exact_augmented_energy_before
    after = block.exact_augmented_energy_after
    # The certificate was deeply validated once at this call boundary.  Use
    # its exact theorem formula directly so this same derivation does not
    # recursively validate it again.
    endpoint_gain = q_eff ** (block.boundary_count // horizon)
    endpoint_upper = endpoint_gain * before
    endpoint_gain_exact = bool(
        type(endpoint_gain) is Fraction
        and endpoint_gain == q_eff ** (block.boundary_count // horizon)
    )
    endpoint_satisfied = bool(
        type(before) is Fraction
        and type(after) is Fraction
        and before >= 0
        and after >= 0
        and after <= endpoint_upper
    )

    conditions = (
        ("source_execution_intact", True),
        (
            "source_causal_execution_provenance_intact",
            True,
        ),
        ("relative_defect_certificate_intact", True),
        (
            "runtime_block_selection_and_provenance_intact",
            True,
        ),
        (
            "runtime_block_objects_bound_by_identity",
            block.source_execution is execution
            and len(boundaries) == block.boundary_count
            and all(
                observed is expected
                for observed, expected in zip(
                    boundaries,
                    execution.runtime_telescope.boundaries[
                        block.start_boundary : block.start_boundary
                        + block.boundary_count
                    ],
                    strict=True,
                )
            ),
        ),
        ("fixed_remesh_model_matches_certificate", all(remesh_matches)),
        (
            "fixed_exact_metric_and_history_dimension",
            all(
                boundary.exact_common_normalized_metric == common_metric
                and len(boundary.exact_transition.exact_history_energies)
                == dimension
                for boundary in boundaries
            ),
        ),
        (
            "every_boundary_energy_telemetry_exact_nonnegative",
            all(exact_telemetry),
        ),
        (
            "every_remesh_input_budget_matches_companion_head",
            all(budget_identities),
        ),
        (
            "every_ideal_remesh_head_satisfies_jensen_bound",
            all(jensen_bounds),
        ),
        (
            "every_pre_schedule_energy_defect_identity",
            all(defect_identities),
        ),
        (
            "every_pre_schedule_relative_energy_defect_bound",
            all(relative_bounds),
        ),
        (
            "every_runtime_schedule_gain_bounded_by_common_q",
            all(gain_bounds),
        ),
        (
            "every_recorded_schedule_head_satisfies_its_gain_bound",
            all(schedule_head_bounds),
        ),
        ("every_effective_head_energy_bound", all(effective_bounds)),
        (
            "every_history_energy_vector_satisfies_effective_envelope",
            all(entrywise_envelopes),
        ),
        (
            "recorded_history_energy_vectors_advance_contiguously",
            continuous,
        ),
        (
            "augmented_energies_match_stationary_history_weights",
            all(augmented_matches)
            and _weighted_energy(stationary, input_tuple[0]) == before
            and _weighted_energy(stationary, output_tuple[-1]) == after,
        ),
        (
            "finite_endpoint_gain_matches_exact_theorem",
            endpoint_gain_exact,
        ),
        (
            "finite_endpoint_energy_bound_satisfied",
            endpoint_satisfied,
        ),
    )
    if not _strict_conditions(conditions) or not all(
        passed for _name, passed in conditions
    ):
        failed = ", ".join(name for name, passed in conditions if not passed)
        raise TNFRValueError(
            f"runtime REMESH relative-defect block failed: {failed}"
        )

    return {
        "source_execution": execution,
        "relative_defect_certificate": certificate,
        "block_observation": block,
        "start_boundary": block.start_boundary,
        "boundary_count": block.boundary_count,
        "boundaries": boundaries,
        "exact_common_normalized_metric": common_metric,
        "exact_stationary_history_weights": stationary,
        "exact_input_history_energy_vectors": input_tuple,
        "exact_output_history_energy_vectors": output_tuple,
        "exact_remesh_input_energy_upper_bounds": tuple(budgets),
        "exact_ideal_remesh_head_energies": tuple(ideal_energies),
        "exact_runtime_bounded_head_energies": tuple(bounded_energies),
        "exact_pre_schedule_energy_defects": tuple(defects),
        "exact_relative_energy_defect_ratios": tuple(ratios),
        "exact_relative_energy_defect_slacks": tuple(slacks),
        "exact_schedule_energy_gain_upper_bounds": tuple(schedule_gains),
        "exact_effective_head_energy_upper_bounds": tuple(
            effective_head_bounds
        ),
        "exact_scheduled_head_energies": tuple(scheduled_energies),
        "exact_energy_envelope_vectors": tuple(envelope_vectors),
        "exact_augmented_energy_before": before,
        "exact_augmented_energy_after": after,
        "exact_finite_endpoint_energy_gain_upper_bound": endpoint_gain,
        "exact_finite_endpoint_energy_upper_bound": endpoint_upper,
        "conditions": conditions,
    }


@dataclass(frozen=True, slots=True)
class RuntimeRemeshScheduleRelativeDefectBlockObservation:
    """Sealed common-``q`` relative-defect proof for one runtime block."""

    source_execution: ExecutedEventRemeshCycleSequence = field(
        repr=False,
        compare=False,
    )
    relative_defect_certificate: (
        UniformRemeshScheduleRelativeDefectStabilityCertificate
    ) = field(repr=False, compare=False)
    block_observation: RuntimeRemeshScheduleBlockMarginObservation = field(
        repr=False,
        compare=False,
    )
    start_boundary: int
    boundary_count: int
    boundaries: tuple[RuntimeRemeshScheduleBoundaryObservation, ...] = field(
        repr=False,
        compare=False,
    )
    exact_common_normalized_metric: ExactVector
    exact_stationary_history_weights: ExactVector
    exact_input_history_energy_vectors: tuple[ExactVector, ...]
    exact_output_history_energy_vectors: tuple[ExactVector, ...]
    exact_remesh_input_energy_upper_bounds: tuple[Fraction, ...]
    exact_ideal_remesh_head_energies: tuple[Fraction, ...]
    exact_runtime_bounded_head_energies: tuple[Fraction, ...]
    exact_pre_schedule_energy_defects: tuple[Fraction, ...]
    exact_relative_energy_defect_ratios: tuple[Fraction | None, ...]
    exact_relative_energy_defect_slacks: tuple[Fraction, ...]
    exact_schedule_energy_gain_upper_bounds: tuple[Fraction, ...]
    exact_effective_head_energy_upper_bounds: tuple[Fraction, ...]
    exact_scheduled_head_energies: tuple[Fraction, ...]
    exact_energy_envelope_vectors: tuple[ExactVector, ...]
    exact_augmented_energy_before: Fraction
    exact_augmented_energy_after: Fraction
    exact_finite_endpoint_energy_gain_upper_bound: Fraction
    exact_finite_endpoint_energy_upper_bound: Fraction
    conditions: tuple[tuple[str, bool], ...]
    _proof_stamp: tuple[Any, ...] = field(default=(), repr=False, compare=False)

    def _proof_fields_are_intact_after_dependencies_validation(self) -> bool:
        """Rederive this proof after same-call dependency validation.

        No result from this private path is retained.  The caller must have
        deeply validated the source execution, the relative-defect certificate
        and the stored block immediately beforehand.
        """

        try:
            current = _observation_values(self)
            current_stamp = _proof_stamp_from_values(current)
            if not proof_stamps_are_identical(
                object.__getattribute__(self, "_proof_stamp"),
                current_stamp,
            ):
                return False
            execution = object.__getattribute__(self, "source_execution")
            certificate = object.__getattribute__(
                self,
                "relative_defect_certificate",
            )
            expected = _derive_values(
                execution,
                certificate,
                object.__getattribute__(self, "start_boundary"),
                object.__getattribute__(self, "boundary_count"),
                block_override=object.__getattribute__(
                    self,
                    "block_observation",
                ),
                source_already_validated=True,
                certificate_already_validated=True,
                block_already_validated=True,
            )
            expected_stamp = _proof_stamp_from_values(expected)
            return proof_stamps_are_identical(
                object.__getattribute__(self, "_proof_stamp"),
                expected_stamp,
            )
        except BaseException:
            return False

    def _proof_fields_are_intact(self) -> bool:
        try:
            source = object.__getattribute__(self, "source_execution")
            certificate = object.__getattribute__(
                self,
                "relative_defect_certificate",
            )
            block = object.__getattribute__(self, "block_observation")
            if not _execution_is_intact(source):
                return False
            if not _certificate_is_intact(certificate):
                return False
            if (
                type(block) is not RuntimeRemeshScheduleBlockMarginObservation
                or block.source_execution is not source
                or not block._proof_fields_are_intact_after_source_validation()
            ):
                return False
            return self._proof_fields_are_intact_after_dependencies_validation()
        except BaseException:
            return False

    @property
    def scope(self) -> str:
        return _SCOPE

    @property
    def relative_defect_block_observation_certified(self) -> bool:
        return self._proof_fields_are_intact()

    @property
    def failed_conditions(self) -> tuple[str, ...]:
        if not self._proof_fields_are_intact():
            return (
                "runtime_remesh_schedule_relative_defect_proof_fields_intact",
            )
        return tuple(name for name, passed in self.conditions if not passed)

    @property
    def same_graph_execution_provenance_certified(self) -> bool:
        return self._proof_fields_are_intact()

    @property
    def whole_sequence_graph_state_atomic(self) -> bool:
        return self._proof_fields_are_intact()

    @property
    def every_boundary_relative_defect_bound_verified(self) -> bool:
        return self._proof_fields_are_intact()

    @property
    def runtime_relative_defect_bound_verified(self) -> bool:
        """Report verification only for this finite selected runtime block."""

        return self._proof_fields_are_intact()

    @property
    def runtime_schedule_maps_verified(self) -> bool:
        """Report represented-gain verification only for this finite block."""

        return self._proof_fields_are_intact()

    @property
    def exact_finite_energy_envelope_certified(self) -> bool:
        return self._proof_fields_are_intact()

    @property
    def exact_finite_endpoint_bound_certified(self) -> bool:
        return self._proof_fields_are_intact()

    @property
    def finite_energy_nonincrease_sufficiently_certified(self) -> bool:
        return bool(
            self._proof_fields_are_intact()
            and self.exact_finite_endpoint_energy_gain_upper_bound <= 1
        )

    @property
    def finite_strict_energy_contraction_sufficiently_certified(self) -> bool:
        return bool(
            self._proof_fields_are_intact()
            and self.exact_augmented_energy_before > 0
            and self.exact_finite_endpoint_energy_gain_upper_bound < 1
        )

    @property
    def zero_energy_preservation_observed(self) -> bool:
        return bool(
            self._proof_fields_are_intact()
            and self.exact_augmented_energy_before == 0
            and self.exact_augmented_energy_after == 0
        )

    @property
    def uniform_runtime_relative_defect_class_certified(self) -> bool:
        return False

    @property
    def runtime_forward_invariant_class_certified(self) -> bool:
        return False

    @property
    def runtime_forward_invariance_certified(self) -> bool:
        return False

    @property
    def repeated_runtime_stability_certified(self) -> bool:
        return False

    @property
    def repeated_binary64_runtime_stability_certified(self) -> bool:
        return False

    @property
    def binary64_runtime_stability_certified(self) -> bool:
        return False

    @property
    def future_runtime_stability_certified(self) -> bool:
        return False

    @property
    def future_stability_certified(self) -> bool:
        return False

    @property
    def runtime_global_gain_certified(self) -> bool:
        return False

    @property
    def solver_accuracy_certified(self) -> bool:
        return False

    @property
    def solver_order_certified(self) -> bool:
        return False

    @property
    def mesh_convergence_certified(self) -> bool:
        return False

    @property
    def adaptive_grammar_certified(self) -> bool:
        return False

    @property
    def full_tnfr_stability_certified(self) -> bool:
        return False


def observe_executed_event_remesh_relative_defect_block(
    execution: ExecutedEventRemeshCycleSequence,
    certificate: UniformRemeshScheduleRelativeDefectStabilityCertificate,
    *,
    start_boundary: int = 0,
    boundary_count: int | None = None,
) -> RuntimeRemeshScheduleRelativeDefectBlockObservation:
    """Verify the common-``q`` relative-defect theorem on a causal block."""

    values = _derive_values(
        execution,
        certificate,
        start_boundary,
        boundary_count,
    )
    result = RuntimeRemeshScheduleRelativeDefectBlockObservation(
        **values,
        _proof_stamp=_proof_stamp_from_values(values),
    )
    # ``_derive_values`` just validated every authoritative dependency and the
    # nested block in this same stack.  Rederive only the outer proof here;
    # public queries validate all dependencies afresh.
    if not result._proof_fields_are_intact_after_dependencies_validation():
        raise RuntimeError(
            "constructed runtime REMESH relative-defect proof is inconsistent"
        )
    return result
