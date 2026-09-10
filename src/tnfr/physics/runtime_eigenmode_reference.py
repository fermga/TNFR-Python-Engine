r"""Bind exact reversible eigenmode references to executed Euler partitions.

The adapter treats every captured binary64 scalar as its exact rational value.
For one refreshed segment with observed state ``z``, stored pressure ``p64``
and exact reference pressure ``p* = -L_rw z``, it separates

``rho = p64 - p*``
``eta = z_next - z - h diag(nu) p64``
``epsilon = h diag(nu) rho + eta``.

Consequently, with ``A = diag(nu) L_rw`` and ``M_h = I - h A``, the
observed endpoint satisfies ``z_next = M_h z + epsilon`` exactly over the
rationalized represented values.  Defects are propagated with the complete
matrix ``M_h``; they are not assumed to remain in the initial eigenmode.
The reported H-energy intervals bound the unrecentered endpoint error through
``E_H(e) = 1/2 sum_i H_i e_i**2``.

This is a finite, offline binding of individually executor-certified physical
partitions.  It does not establish binary64 mesh convergence, solver order,
common causal provenance across partitions, glyph or REMESH dynamics, or
future TNFR stability.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Set
from dataclasses import dataclass, field, fields, replace
from fractions import Fraction
from typing import Any

from ..errors import TNFRValueError
from ..operators.event_runtime import (
    ExecutedPressureRefreshedFlowPartition,
)
from ..utils._structural_signature import (
    binary64_vectors_are_identical,
    proof_stamps_are_identical,
    structural_proof_signature,
)
from .reversible_eigenmode_reference import (
    ReversibleSingleEigenmodeEulerReferenceCertificate,
    certify_reversible_single_eigenmode_euler_reference,
)

__all__ = (
    "ExecutedReversibleSingleEigenmodeEulerPartitionObservation",
    "ExecutedReversibleSingleEigenmodeEulerReferenceObservation",
    "observe_executed_reversible_single_eigenmode_euler_reference",
)

ExactVector = tuple[Fraction, ...]
ExactMatrix = tuple[ExactVector, ...]
ExactPartition = tuple[Fraction, ...]

_PARTITION_PROOF_VERSION = (
    "executed_reversible_single_eigenmode_euler_partition_v1"
)
_REFERENCE_PROOF_VERSION = (
    "executed_reversible_single_eigenmode_euler_reference_v1"
)
_SCOPE = (
    "Finite offline binding of individually executor-certified, explicitly "
    "pressure-refreshed Euler partitions to one exact rational reversible "
    "single-eigenmode reference. Captured binary64 coefficients and states "
    "are interpreted as exact Fractions. Exact pressure-realization and "
    "held-input execution residuals are separated and propagated through "
    "the complete Euler matrices. Rational continuous-runtime endpoint "
    "intervals and Linf/H-energy bounds cover only the supplied finite "
    "partitions. Binary64 asymptotic or mesh convergence, solver accuracy "
    "or order, common causal provenance across partitions, future or "
    "repeated stability, mixed initial modes, directed or changing generators, "
    "glyphs, REMESH and full TNFR stability remain outside scope."
)

_PARTITION_CONDITION_NAMES = (
    "reference_certificate_intact",
    "executor_partition_evidence_intact",
    "runtime_partition_matches_reference_inputs",
    "trusted_binary64_held_pressure_replays",
    "binary64_pure_epi_pressure_refreshed",
    "exact_reference_boundary_recurrence",
    "exact_pressure_execution_defect_decomposition",
    "complete_matrix_endpoint_defect_propagation",
    "rational_continuous_runtime_error_enclosure",
)

_REFERENCE_CONDITION_NAMES = (
    "reference_derived_from_executor_evidence",
    "every_runtime_partition_binding_intact",
    "common_ordered_node_support",
    "common_exact_initial_epi",
    "common_exact_conductance",
    "common_exact_capacity",
    "runtime_partition_family_matches_reference",
    "exact_runtime_defect_decompositions",
    "rational_continuous_runtime_error_enclosures",
)


@dataclass(frozen=True, slots=True)
class _ExecutionFacts:
    execution: ExecutedPressureRefreshedFlowPartition
    nodes: tuple[Any, ...]
    segment_durations: ExactPartition
    runtime_boundary_epi: tuple[ExactVector, ...]


def _raw_stamp_or_none(value: Any) -> tuple[Any, ...] | None:
    try:
        stamp = object.__getattribute__(value, "_proof_stamp")
    except BaseException:
        return None
    return stamp if type(stamp) is tuple else None


def _nested_stamp(value: Any, expected_type: type[Any]) -> Any:
    if type(value) is not expected_type:
        return structural_proof_signature(value)
    return (
        "tnfr-nested-proof-stamp-v1",
        expected_type.__module__,
        expected_type.__qualname__,
        _raw_stamp_or_none(value),
    )


def _strict_exact_vector(value: Any, width: int) -> bool:
    return bool(
        type(value) is tuple
        and len(value) == width
        and all(type(item) is Fraction for item in value)
    )


def _strict_conditions(value: Any, names: tuple[str, ...]) -> bool:
    return bool(
        type(value) is tuple
        and len(value) == len(names)
        and all(
            type(item) is tuple
            and len(item) == 2
            and type(item[0]) is str
            and type(item[1]) is bool
            and item[0] == names[index]
            for index, item in enumerate(value)
        )
    )


def _materialize_executions(value: Any) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, bytearray, Mapping, Set)):
        raise TypeError("partitions must be an ordered iterable sequence")
    try:
        result = tuple(value)
    except TypeError as exc:
        raise TypeError("partitions must be an ordered iterable sequence") from exc
    if not result:
        raise TNFRValueError("partitions must contain runtime partition evidence")
    if any(
        type(item) is not ExecutedPressureRefreshedFlowPartition
        for item in result
    ):
        raise TypeError(
            "partitions must contain ExecutedPressureRefreshedFlowPartition "
            "evidence"
        )
    return result


def _support_signature(nodes: tuple[Any, ...]) -> tuple[Any, ...]:
    if type(nodes) is not tuple or not nodes:
        raise TNFRValueError("runtime partition nodes must be a nonempty tuple")
    tokens = tuple(structural_proof_signature(node) for node in nodes)
    if len(set(tokens)) != len(tokens):
        raise TNFRValueError(
            "runtime partition support contains duplicate structural nodes"
        )
    return tokens


def _matrix_vector(matrix: ExactMatrix, vector: ExactVector) -> ExactVector:
    return tuple(
        sum(
            (
                coefficient * value
                for coefficient, value in zip(row, vector, strict=True)
            ),
            Fraction(0),
        )
        for row in matrix
    )


def _euler_matrix(generator: ExactMatrix, duration: Fraction) -> ExactMatrix:
    size = len(generator)
    return tuple(
        tuple(
            (Fraction(1) if row == column else Fraction(0))
            - duration * generator[row][column]
            for column in range(size)
        )
        for row in range(size)
    )


def _vector_difference(left: ExactVector, right: ExactVector) -> ExactVector:
    return tuple(
        left_value - right_value
        for left_value, right_value in zip(left, right, strict=True)
    )


def _vector_sum(left: ExactVector, right: ExactVector) -> ExactVector:
    return tuple(
        left_value + right_value
        for left_value, right_value in zip(left, right, strict=True)
    )


def _max_abs(value: ExactVector) -> Fraction:
    return max((abs(item) for item in value), default=Fraction(0))


def _interval_min_abs(lower: Fraction, upper: Fraction) -> Fraction:
    if lower <= 0 <= upper:
        return Fraction(0)
    return min(abs(lower), abs(upper))


def _execution_facts(
    execution: ExecutedPressureRefreshedFlowPartition,
) -> _ExecutionFacts:
    if type(execution) is not ExecutedPressureRefreshedFlowPartition:
        raise TypeError(
            "runtime partition evidence must have its canonical executor type"
        )
    if not execution._proof_fields_are_intact():
        raise TNFRValueError("runtime partition evidence is not intact")

    segments = execution.partition.segments
    boundaries = execution.boundary_observations
    flows = execution.segment_flow_evidence
    if not segments or len(boundaries) != len(segments) + 1:
        raise TNFRValueError("runtime partition evidence is incomplete")
    nodes = boundaries[0].after.nodes
    support = _support_signature(nodes)
    width = len(nodes)

    for boundary in boundaries:
        snapshot = boundary.after
        if (
            _support_signature(snapshot.nodes) != support
            or not _strict_exact_vector(snapshot.exact_epi, width)
            or not _strict_exact_vector(snapshot.exact_nu_f, width)
            or not _strict_exact_vector(snapshot.exact_delta_nfr, width)
            or not binary64_vectors_are_identical(
                snapshot.delta_nfr,
                snapshot.binary64_pure_epi_pressure,
            )
        ):
            raise TNFRValueError(
                "runtime boundaries must retain support and refreshed "
                "binary64 pure-EPI pressure"
            )

    for flow in flows:
        certificate = flow.certificate
        held_replay_identified = bool(
            object.__getattribute__(
                flow,
                "integrator_provenance_certified",
            )
            is True
            and certificate is not None
            and object.__getattribute__(
                certificate,
                "_binary64_held_pressure_runtime_identified",
            )
            is True
        )
        if not held_replay_identified:
            raise TNFRValueError(
                "every segment requires a trusted binary64 held-pressure "
                "Euler replay"
            )
        if certificate is None:
            raise TNFRValueError("runtime flow certificate is absent")
        if (
            certificate.exact_nodal_equation_residual is None
            or certificate.exact_pressure_residual is None
        ):
            raise TNFRValueError(
                "runtime flow residual evidence is incomplete"
            )

    return _ExecutionFacts(
        execution=execution,
        nodes=nodes,
        segment_durations=tuple(
            segment.exact_duration for segment in segments
        ),
        runtime_boundary_epi=tuple(
            boundary.after.exact_epi for boundary in boundaries
        ),
    )


def _require_common_sources(facts: tuple[_ExecutionFacts, ...]) -> None:
    first = facts[0]
    first_snapshot = first.execution.boundary_observations[0].after
    support = _support_signature(first.nodes)
    initial = first.runtime_boundary_epi[0]
    capacity = first_snapshot.exact_nu_f
    conductance = first_snapshot.conductance

    for item in facts:
        if _support_signature(item.nodes) != support:
            raise TNFRValueError(
                "runtime partitions require common ordered node support"
            )
        if item.runtime_boundary_epi[0] != initial:
            raise TNFRValueError(
                "runtime partitions require one common exact initial EPI"
            )
        for boundary in item.execution.boundary_observations:
            snapshot = boundary.after
            if snapshot.exact_nu_f != capacity:
                raise TNFRValueError(
                    "runtime partitions require common fixed capacity"
                )
            if snapshot.conductance != conductance:
                raise TNFRValueError(
                    "runtime partitions require common fixed conductance"
                )
        for flow in item.execution.segment_flow_evidence:
            certificate = flow.certificate
            if certificate is None:
                raise TNFRValueError("runtime flow certificate is absent")
            for snapshot in (certificate.left, certificate.right):
                if (
                    _support_signature(snapshot.nodes) != support
                    or snapshot.exact_nu_f != capacity
                    or snapshot.conductance != conductance
                ):
                    raise TNFRValueError(
                        "runtime flow source differs from the common generator"
                    )


def _reference_from_facts(
    facts: tuple[_ExecutionFacts, ...],
) -> ReversibleSingleEigenmodeEulerReferenceCertificate:
    _require_common_sources(facts)
    first = facts[0]
    snapshot = first.execution.boundary_observations[0].after
    return certify_reversible_single_eigenmode_euler_reference(
        snapshot.conductance,
        nu_f=snapshot.exact_nu_f,
        initial_epi=first.runtime_boundary_epi[0],
        partitions=tuple(item.segment_durations for item in facts),
    )


@dataclass(frozen=True, slots=True)
class ExecutedReversibleSingleEigenmodeEulerPartitionObservation:
    """One executor-bound row of a reversible eigenmode partition family."""

    reference_certificate: ReversibleSingleEigenmodeEulerReferenceCertificate = (
        field(repr=False)
    )
    execution: ExecutedPressureRefreshedFlowPartition = field(repr=False)
    partition_index: int
    nodes: tuple[Any, ...]
    exact_segment_durations: ExactPartition
    exact_runtime_boundary_epi: tuple[ExactVector, ...]
    exact_reference_boundary_epi: tuple[ExactVector, ...]
    exact_pressure_realization_residuals: tuple[ExactVector, ...]
    exact_held_input_execution_residuals: tuple[ExactVector, ...]
    exact_local_runtime_defects: tuple[ExactVector, ...]
    exact_endpoint_runtime_defect: ExactVector
    exact_endpoint_runtime_defect_linf: Fraction
    exact_runtime_minus_continuous_endpoint_lower_bound: ExactVector
    exact_runtime_minus_continuous_endpoint_upper_bound: ExactVector
    exact_runtime_continuous_linf_error_lower_bound: Fraction
    exact_runtime_continuous_linf_error_upper_bound: Fraction
    exact_runtime_continuous_h_energy_error_lower_bound: Fraction
    exact_runtime_continuous_h_energy_error_upper_bound: Fraction
    all_segment_exact_affine_maps_identified: bool
    conditions: tuple[tuple[str, bool], ...]
    _proof_stamp: tuple[Any, ...] = field(default=(), repr=False, compare=False)

    def _proof_fields_are_intact(self) -> bool:
        try:
            current = _partition_values(self)
            observed_stamp = object.__getattribute__(self, "_proof_stamp")
            if not proof_stamps_are_identical(
                observed_stamp,
                _partition_stamp(current),
            ):
                return False
            reference = object.__getattribute__(self, "reference_certificate")
            execution = object.__getattribute__(self, "execution")
            if (
                type(reference)
                is not ReversibleSingleEigenmodeEulerReferenceCertificate
                or not reference.reference_certificate_certified
                or type(execution)
                is not ExecutedPressureRefreshedFlowPartition
            ):
                return False
            expected = _derive_partition_values(
                reference,
                object.__getattribute__(self, "partition_index"),
                _execution_facts(execution),
                reference_already_validated=True,
            )
            return proof_stamps_are_identical(
                observed_stamp,
                _partition_stamp(expected),
            )
        except BaseException:
            return False

    @property
    def scope(self) -> str:
        return _SCOPE

    @property
    def runtime_partition_binding_certified(self) -> bool:
        return self._proof_fields_are_intact()

    @property
    def failed_conditions(self) -> tuple[str, ...]:
        if not self._proof_fields_are_intact():
            return ("runtime_partition_proof_fields_intact",)
        return tuple(name for name, passed in self.conditions if not passed)

    @property
    def exact_runtime_defect_decomposition_certified(self) -> bool:
        return self._proof_fields_are_intact()

    @property
    def exact_continuous_error_enclosure_certified(self) -> bool:
        return self._proof_fields_are_intact()


_PARTITION_FIELD_NAMES = tuple(
    item.name
    for item in fields(
        ExecutedReversibleSingleEigenmodeEulerPartitionObservation
    )
    if item.name != "_proof_stamp"
)


def _partition_values(
    value: ExecutedReversibleSingleEigenmodeEulerPartitionObservation,
) -> dict[str, Any]:
    if (
        type(value)
        is not ExecutedReversibleSingleEigenmodeEulerPartitionObservation
    ):
        raise TypeError("partition observation must have its canonical type")
    return {
        name: object.__getattribute__(value, name)
        for name in _PARTITION_FIELD_NAMES
    }


def _partition_field_signature(name: str, value: Any) -> Any:
    if name == "reference_certificate":
        return _nested_stamp(
            value,
            ReversibleSingleEigenmodeEulerReferenceCertificate,
        )
    if name == "execution":
        return _nested_stamp(value, ExecutedPressureRefreshedFlowPartition)
    return structural_proof_signature(value)


def _partition_stamp(values: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _PARTITION_PROOF_VERSION,
        tuple(
            (name, _partition_field_signature(name, values[name]))
            for name in _PARTITION_FIELD_NAMES
        ),
    )


def _derive_partition_values(
    reference: ReversibleSingleEigenmodeEulerReferenceCertificate,
    partition_index: int,
    facts: _ExecutionFacts,
    *,
    reference_already_validated: bool = False,
) -> dict[str, Any]:
    if (
        type(reference)
        is not ReversibleSingleEigenmodeEulerReferenceCertificate
        or (
            not reference_already_validated
            and not reference.reference_certificate_certified
        )
    ):
        raise TNFRValueError("exact eigenmode reference is not intact")
    if (
        type(partition_index) is not int
        or partition_index < 0
        or partition_index >= len(reference.exact_partitions)
    ):
        raise TNFRValueError("partition_index is outside the reference family")

    execution = facts.execution
    durations = facts.segment_durations
    runtime_boundaries = facts.runtime_boundary_epi
    flows = execution.segment_flow_evidence
    width = len(reference.exact_initial_epi)
    support = _support_signature(facts.nodes)
    if (
        durations != reference.exact_partitions[partition_index]
        or len(runtime_boundaries) != len(durations) + 1
        or runtime_boundaries[0] != reference.exact_initial_epi
    ):
        raise TNFRValueError(
            "runtime partition does not match its exact reference partition"
        )

    for boundary in execution.boundary_observations:
        snapshot = boundary.after
        if (
            _support_signature(snapshot.nodes) != support
            or snapshot.exact_nu_f != reference.exact_nu_f
            or snapshot.conductance != reference.exact_conductance
        ):
            raise TNFRValueError(
                "runtime partition changed support, capacity, or conductance"
            )

    reference_boundaries: list[ExactVector] = [reference.exact_initial_epi]
    running_factor = Fraction(1)
    for factor in reference.exact_euler_segment_factors[partition_index]:
        running_factor *= factor
        reference_boundaries.append(
            tuple(
                reference.exact_weighted_mean
                + running_factor * mode_value
                for mode_value in reference.exact_centered_mode
            )
        )
    if reference_boundaries[-1] != reference.exact_euler_endpoints[partition_index]:
        raise RuntimeError("exact reference boundary recurrence is inconsistent")

    pressure_residuals: list[ExactVector] = []
    execution_residuals: list[ExactVector] = []
    local_defects: list[ExactVector] = []
    propagated = (Fraction(0),) * width

    for index, (duration, flow) in enumerate(
        zip(durations, flows, strict=True)
    ):
        certificate = flow.certificate
        if certificate is None:
            raise TNFRValueError("runtime flow certificate is absent")
        if (
            certificate.exact_duration != duration
            or _support_signature(certificate.left.nodes) != support
            or _support_signature(certificate.right.nodes) != support
            or certificate.left.exact_epi != runtime_boundaries[index]
            or certificate.right.exact_epi != runtime_boundaries[index + 1]
            or certificate.left.exact_nu_f != reference.exact_nu_f
            or certificate.right.exact_nu_f != reference.exact_nu_f
            or certificate.left.conductance != reference.exact_conductance
            or certificate.right.conductance != reference.exact_conductance
        ):
            raise TNFRValueError(
                "runtime flow certificate differs from the reference source"
            )
        rho = certificate.exact_pressure_residual
        eta = certificate.exact_nodal_equation_residual
        if (
            not _strict_exact_vector(rho, width)
            or not _strict_exact_vector(eta, width)
        ):
            raise TNFRValueError("runtime flow residual evidence is incomplete")
        local = tuple(
            duration * capacity * pressure_residual + execution_residual
            for capacity, pressure_residual, execution_residual in zip(
                reference.exact_nu_f,
                rho,
                eta,
                strict=True,
            )
        )
        matrix = _euler_matrix(reference.exact_generator, duration)
        expected_runtime_right = _vector_sum(
            _matrix_vector(matrix, runtime_boundaries[index]),
            local,
        )
        if expected_runtime_right != runtime_boundaries[index + 1]:
            raise RuntimeError(
                "pressure and held-input residuals do not reconstruct runtime"
            )
        expected_reference_right = _matrix_vector(
            matrix,
            reference_boundaries[index],
        )
        if expected_reference_right != reference_boundaries[index + 1]:
            raise RuntimeError("reference boundary lost its exact Euler map")
        propagated = _vector_sum(_matrix_vector(matrix, propagated), local)
        pressure_residuals.append(rho)
        execution_residuals.append(eta)
        local_defects.append(local)

    endpoint_defect = _vector_difference(
        runtime_boundaries[-1],
        reference_boundaries[-1],
    )
    if propagated != endpoint_defect:
        raise RuntimeError(
            "complete-matrix runtime defect propagation is inconsistent"
        )

    continuous_lower = reference.exact_continuous_endpoint_lower_bound
    continuous_upper = reference.exact_continuous_endpoint_upper_bound
    signed_lower = tuple(
        runtime - upper
        for runtime, upper in zip(
            runtime_boundaries[-1],
            continuous_upper,
            strict=True,
        )
    )
    signed_upper = tuple(
        runtime - lower
        for runtime, lower in zip(
            runtime_boundaries[-1],
            continuous_lower,
            strict=True,
        )
    )
    if any(lower > upper for lower, upper in zip(signed_lower, signed_upper)):
        raise RuntimeError("continuous-runtime coordinate enclosure is inverted")
    coordinate_minima = tuple(
        _interval_min_abs(lower, upper)
        for lower, upper in zip(signed_lower, signed_upper, strict=True)
    )
    coordinate_maxima = tuple(
        max(abs(lower), abs(upper))
        for lower, upper in zip(signed_lower, signed_upper, strict=True)
    )
    linf_lower = max(coordinate_minima, default=Fraction(0))
    linf_upper = max(coordinate_maxima, default=Fraction(0))
    metric = reference.exact_reversible_metric
    energy_lower = sum(
        (
            weight * value * value
            for weight, value in zip(metric, coordinate_minima, strict=True)
        ),
        Fraction(0),
    ) / 2
    energy_upper = sum(
        (
            weight * value * value
            for weight, value in zip(metric, coordinate_maxima, strict=True)
        ),
        Fraction(0),
    ) / 2
    exact_affine = all(
        object.__getattribute__(
            flow,
            "integrator_provenance_certified",
        )
        is True
        and flow.certificate is not None
        and flow.certificate.explicit_euler_map_identified
        for flow in flows
    )
    conditions = (
        ("reference_certificate_intact", True),
        ("executor_partition_evidence_intact", True),
        ("runtime_partition_matches_reference_inputs", True),
        ("trusted_binary64_held_pressure_replays", True),
        ("binary64_pure_epi_pressure_refreshed", True),
        ("exact_reference_boundary_recurrence", True),
        ("exact_pressure_execution_defect_decomposition", True),
        ("complete_matrix_endpoint_defect_propagation", True),
        ("rational_continuous_runtime_error_enclosure", True),
    )
    if not _strict_conditions(conditions, _PARTITION_CONDITION_NAMES):
        raise RuntimeError("runtime partition conditions are inconsistent")

    return {
        "reference_certificate": reference,
        "execution": execution,
        "partition_index": partition_index,
        "nodes": facts.nodes,
        "exact_segment_durations": durations,
        "exact_runtime_boundary_epi": runtime_boundaries,
        "exact_reference_boundary_epi": tuple(reference_boundaries),
        "exact_pressure_realization_residuals": tuple(pressure_residuals),
        "exact_held_input_execution_residuals": tuple(execution_residuals),
        "exact_local_runtime_defects": tuple(local_defects),
        "exact_endpoint_runtime_defect": endpoint_defect,
        "exact_endpoint_runtime_defect_linf": _max_abs(endpoint_defect),
        "exact_runtime_minus_continuous_endpoint_lower_bound": signed_lower,
        "exact_runtime_minus_continuous_endpoint_upper_bound": signed_upper,
        "exact_runtime_continuous_linf_error_lower_bound": linf_lower,
        "exact_runtime_continuous_linf_error_upper_bound": linf_upper,
        "exact_runtime_continuous_h_energy_error_lower_bound": energy_lower,
        "exact_runtime_continuous_h_energy_error_upper_bound": energy_upper,
        "all_segment_exact_affine_maps_identified": exact_affine,
        "conditions": conditions,
    }


def _build_partition(
    reference: ReversibleSingleEigenmodeEulerReferenceCertificate,
    partition_index: int,
    facts: _ExecutionFacts,
) -> ExecutedReversibleSingleEigenmodeEulerPartitionObservation:
    values = _derive_partition_values(
        reference,
        partition_index,
        facts,
        reference_already_validated=True,
    )
    return ExecutedReversibleSingleEigenmodeEulerPartitionObservation(
        **values,
        _proof_stamp=_partition_stamp(values),
    )


def _seal_partition(
    value: ExecutedReversibleSingleEigenmodeEulerPartitionObservation,
) -> ExecutedReversibleSingleEigenmodeEulerPartitionObservation:
    """Reseal current fields; validation still rederives authoritative values."""

    values = _partition_values(value)
    return replace(value, _proof_stamp=_partition_stamp(values))


@dataclass(frozen=True, slots=True)
class ExecutedReversibleSingleEigenmodeEulerReferenceObservation:
    """Sealed finite family linking executor evidence to one exact reference."""

    reference_certificate: ReversibleSingleEigenmodeEulerReferenceCertificate = (
        field(repr=False)
    )
    partition_observations: tuple[
        ExecutedReversibleSingleEigenmodeEulerPartitionObservation,
        ...,
    ]
    nodes: tuple[Any, ...]
    conditions: tuple[tuple[str, bool], ...]
    _proof_stamp: tuple[Any, ...] = field(default=(), repr=False, compare=False)

    def _proof_fields_are_intact(self) -> bool:
        try:
            current = _reference_values(self)
            observed_stamp = object.__getattribute__(self, "_proof_stamp")
            if not proof_stamps_are_identical(
                observed_stamp,
                _reference_stamp(current),
            ):
                return False
            reference = object.__getattribute__(self, "reference_certificate")
            rows = object.__getattribute__(self, "partition_observations")
            if (
                type(reference)
                is not ReversibleSingleEigenmodeEulerReferenceCertificate
                or not reference.reference_certificate_certified
                or type(rows) is not tuple
                or not rows
                or any(
                    type(row)
                    is not ExecutedReversibleSingleEigenmodeEulerPartitionObservation
                    for row in rows
                )
                or any(
                    object.__getattribute__(row, "reference_certificate")
                    is not reference
                    for row in rows
                )
            ):
                return False
            executions = tuple(
                object.__getattribute__(row, "execution") for row in rows
            )
            expected = _derive_reference_values(
                executions,
                reference_override=reference,
                row_overrides=rows,
            )
            return proof_stamps_are_identical(
                observed_stamp,
                _reference_stamp(expected),
            )
        except BaseException:
            return False

    @property
    def scope(self) -> str:
        return _SCOPE

    @property
    def runtime_reference_binding_certified(self) -> bool:
        return self._proof_fields_are_intact()

    @property
    def failed_conditions(self) -> tuple[str, ...]:
        if not self._proof_fields_are_intact():
            return ("runtime_reference_proof_fields_intact",)
        return tuple(name for name, passed in self.conditions if not passed)

    @property
    def exact_runtime_defect_decomposition_certified(self) -> bool:
        return self._proof_fields_are_intact()

    @property
    def exact_continuous_error_enclosures_certified(self) -> bool:
        return self._proof_fields_are_intact()

    @property
    def all_observed_endpoints_equal_exact_euler_reference(self) -> bool:
        return bool(
            self._proof_fields_are_intact()
            and all(
                not any(row.exact_endpoint_runtime_defect)
                for row in self.partition_observations
            )
        )

    @property
    def binary64_asymptotic_convergence_certified(self) -> bool:
        return False

    @property
    def runtime_mesh_convergence_certified(self) -> bool:
        return False

    @property
    def solver_accuracy_certified(self) -> bool:
        return False

    @property
    def solver_order_certified(self) -> bool:
        return False

    @property
    def future_or_repeated_stability_certified(self) -> bool:
        return False

    @property
    def common_causal_execution_provenance_certified(self) -> bool:
        return False

    @property
    def glyph_or_remesh_dynamics_certified(self) -> bool:
        return False

    @property
    def full_tnfr_stability_certified(self) -> bool:
        return False


_REFERENCE_FIELD_NAMES = tuple(
    item.name
    for item in fields(
        ExecutedReversibleSingleEigenmodeEulerReferenceObservation
    )
    if item.name != "_proof_stamp"
)


def _reference_values(
    value: ExecutedReversibleSingleEigenmodeEulerReferenceObservation,
) -> dict[str, Any]:
    if (
        type(value)
        is not ExecutedReversibleSingleEigenmodeEulerReferenceObservation
    ):
        raise TypeError("runtime reference observation must have canonical type")
    return {
        name: object.__getattribute__(value, name)
        for name in _REFERENCE_FIELD_NAMES
    }


def _reference_field_signature(name: str, value: Any) -> Any:
    if name == "reference_certificate":
        return _nested_stamp(
            value,
            ReversibleSingleEigenmodeEulerReferenceCertificate,
        )
    if name == "partition_observations":
        if type(value) is not tuple:
            return structural_proof_signature(value)
        return (
            "tnfr-nested-proof-sequence-v1",
            tuple(
                _nested_stamp(
                    item,
                    ExecutedReversibleSingleEigenmodeEulerPartitionObservation,
                )
                for item in value
            ),
        )
    return structural_proof_signature(value)


def _reference_stamp(values: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _REFERENCE_PROOF_VERSION,
        tuple(
            (name, _reference_field_signature(name, values[name]))
            for name in _REFERENCE_FIELD_NAMES
        ),
    )


def _derive_reference_values(
    executions: tuple[ExecutedPressureRefreshedFlowPartition, ...],
    *,
    reference_override: (
        ReversibleSingleEigenmodeEulerReferenceCertificate | None
    ) = None,
    row_overrides: tuple[
        ExecutedReversibleSingleEigenmodeEulerPartitionObservation,
        ...,
    ]
    | None = None,
) -> dict[str, Any]:
    facts = tuple(_execution_facts(execution) for execution in executions)
    canonical_reference = _reference_from_facts(facts)
    if reference_override is None:
        reference = canonical_reference
    else:
        reference = reference_override
        if (
            type(reference)
            is not ReversibleSingleEigenmodeEulerReferenceCertificate
            or not reference.reference_certificate_certified
            or not proof_stamps_are_identical(
                _raw_stamp_or_none(reference),
                _raw_stamp_or_none(canonical_reference),
            )
        ):
            raise TNFRValueError(
                "stored exact reference is forged, stale, or unbound"
            )

    if row_overrides is None:
        rows = tuple(
            _build_partition(reference, index, fact)
            for index, fact in enumerate(facts)
        )
    else:
        rows = row_overrides
        if (
            type(rows) is not tuple
            or len(rows) != len(facts)
            or any(
                type(row)
                is not ExecutedReversibleSingleEigenmodeEulerPartitionObservation
                for row in rows
            )
        ):
            raise TNFRValueError("stored runtime partition rows are invalid")
        for index, (row, fact) in enumerate(zip(rows, facts, strict=True)):
            if (
                object.__getattribute__(row, "reference_certificate")
                is not reference
                or object.__getattribute__(row, "execution")
                is not fact.execution
                or object.__getattribute__(row, "partition_index") != index
            ):
                raise TNFRValueError(
                    "stored runtime partition row is forged, stale, or unbound"
                )
            observed_row_stamp = _raw_stamp_or_none(row)
            if not proof_stamps_are_identical(
                observed_row_stamp,
                _partition_stamp(_partition_values(row)),
            ):
                raise TNFRValueError(
                    "stored runtime partition row changed after sealing"
                )
            expected = _derive_partition_values(
                reference,
                index,
                fact,
                reference_already_validated=True,
            )
            if not proof_stamps_are_identical(
                observed_row_stamp,
                _partition_stamp(expected),
            ):
                raise TNFRValueError(
                    "stored runtime partition row changed its derived values"
                )

    nodes = facts[0].nodes
    support = _support_signature(nodes)
    conditions = (
        ("reference_derived_from_executor_evidence", True),
        ("every_runtime_partition_binding_intact", True),
        (
            "common_ordered_node_support",
            all(_support_signature(item.nodes) == support for item in facts),
        ),
        (
            "common_exact_initial_epi",
            all(
                item.runtime_boundary_epi[0] == reference.exact_initial_epi
                for item in facts
            ),
        ),
        ("common_exact_conductance", True),
        ("common_exact_capacity", True),
        (
            "runtime_partition_family_matches_reference",
            reference.exact_partitions
            == tuple(item.segment_durations for item in facts),
        ),
        ("exact_runtime_defect_decompositions", True),
        ("rational_continuous_runtime_error_enclosures", True),
    )
    if not _strict_conditions(conditions, _REFERENCE_CONDITION_NAMES) or not all(
        passed for _, passed in conditions
    ):
        raise RuntimeError("runtime reference conditions are inconsistent")
    return {
        "reference_certificate": reference,
        "partition_observations": rows,
        "nodes": nodes,
        "conditions": conditions,
    }


def _seal_reference(
    value: ExecutedReversibleSingleEigenmodeEulerReferenceObservation,
) -> ExecutedReversibleSingleEigenmodeEulerReferenceObservation:
    """Reseal current fields; validation still rederives authoritative values."""

    values = _reference_values(value)
    return replace(value, _proof_stamp=_reference_stamp(values))


def observe_executed_reversible_single_eigenmode_euler_reference(
    partitions: Iterable[ExecutedPressureRefreshedFlowPartition],
) -> ExecutedReversibleSingleEigenmodeEulerReferenceObservation:
    """Bind an ordered finite runtime mesh family to one exact reference.

    The first refreshed boundary supplies the exact represented conductance,
    capacity and initial EPI. Every later partition must reproduce those
    inputs. Every segment must satisfy ``0 < mu*h < 1`` and multiple
    partitions must form a strict proper-subdivision chain. The supplied
    partitions are paired offline; their individual executor provenance does
    not create one shared causal execution.
    """

    executions = _materialize_executions(partitions)
    values = _derive_reference_values(executions)
    result = ExecutedReversibleSingleEigenmodeEulerReferenceObservation(
        **values,
        _proof_stamp=_reference_stamp(values),
    )
    if not proof_stamps_are_identical(
        _raw_stamp_or_none(result),
        _reference_stamp(_reference_values(result)),
    ):
        raise RuntimeError("constructed runtime eigenmode binding is inconsistent")
    return result
