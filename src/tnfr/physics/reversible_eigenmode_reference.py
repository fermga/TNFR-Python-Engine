r"""Exact Euler reference theorem for one reversible diffusion eigenmode.

For a fixed symmetric nonnegative conductance matrix ``W`` with zero diagonal,
positive row strengths ``d`` and positive nodal capacities ``nu``, define

``A = diag(nu) (I - D**-1 W)`` and ``H = diag(d_i / nu_i)``.

Then ``H A = D - W`` is symmetric.  If the centered initial field
``v = x0 - mean_H(x0) * 1`` is nonzero and satisfies the exact rational
identity ``A v = mu v`` with ``mu > 0``, the pure-EPI nodal equation has the
closed solution

``x(t) = mean_H(x0) * 1 + exp(-mu*t) * v``.

For a positive pressure-refreshed Euler partition ``P = (h_j)`` of ``T`` with
``0 < mu*h_j < 1``, its modal factor is
``g_P = product_j(1 - mu*h_j)`` and

``0 <= exp(-mu*T) - g_P``
``   <= mu**2/2 * sum_j(h_j**2)``
``   <= mu**2/2 * T * max_j(h_j)``.

Proper positive subdivision strictly raises ``g_P`` and strictly lowers the
quadratic bound.  Consequently, for any admissible family with fixed data and
``h_max -> 0``, the exact-real Euler endpoints converge to the exact modal
solution.  This conditional theorem is separate from binary64 execution and
does not cover mixed modes, glyphs, REMESH, changing generators or full TNFR
dynamics.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Set
from dataclasses import dataclass, field, fields, replace
from fractions import Fraction
from typing import Any

from .._exact_time import exp_unit_bounds
from ..errors import TNFRValueError
from ..utils._structural_signature import (
    proof_stamps_are_identical,
    structural_proof_signature,
)

__all__ = (
    "ReversibleSingleEigenmodeEulerReferenceCertificate",
    "certify_reversible_single_eigenmode_euler_reference",
)

ExactVector = tuple[Fraction, ...]
ExactMatrix = tuple[ExactVector, ...]
ExactPartition = tuple[Fraction, ...]
ExactPartitionFamily = tuple[ExactPartition, ...]

_PROOF_VERSION = "reversible_single_eigenmode_euler_reference_v1"
_MAX_RATIONAL_EXPONENT = 4096
_SCOPE = (
    "Exact rational reference theorem for one nonuniform eigenmode of a fixed "
    "connected reversible pure-EPI generator with symmetric nonnegative "
    "zero-diagonal conductance, positive row strengths and positive capacity. "
    "The stored positive partitions with equal total duration lie in "
    "0 < mu*h < 1 and form a strict proper-subdivision chain when more than "
    "one is supplied. "
    "Rational exponential enclosures and exact first-order h_max bounds "
    "certify the endpoint solution and conditional exact-real convergence as "
    "h_max tends to zero. The rational enclosure has the representational "
    "cutoff mu*T <= 4096. Mixed modes, irrational eigenpairs outside the "
    "Fraction domain, directed or changing generators, binary64 execution or "
    "asymptotics, glyphs, REMESH, clipping, solver-order claims and full TNFR "
    "stability are outside scope."
)

_CONDITION_NAMES = (
    "square_zero_diagonal_conductance",
    "symmetric_nonnegative_conductance",
    "connected_positive_strength_support",
    "positive_capacity",
    "exact_reversible_metric_identity",
    "nonuniform_centered_initial_mode",
    "exact_positive_single_eigenmode_identity",
    "positive_equal_duration_partitions",
    "strict_positive_modal_step_domain",
    "strict_proper_subdivision_chain",
    "rational_continuous_factor_enclosure",
    "exact_euler_modal_recurrence",
    "quadratic_factor_error_bound",
    "hmax_factor_error_bound",
    "continuous_endpoint_enclosure",
    "linf_error_bounds",
    "h_energy_error_bounds",
    "strict_refinement_improvement_when_applicable",
)


def _materialize(value: Any, label: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, bytearray, Mapping, Set)):
        raise TypeError(f"{label} must be an iterable sequence")
    try:
        return tuple(value)
    except TypeError as exc:
        raise TypeError(f"{label} must be an iterable sequence") from exc


def _materialize_exact_vector(value: Any, label: str) -> ExactVector:
    items = _materialize(value, label)
    if not items:
        raise TNFRValueError(f"{label} must be nonempty")
    if any(type(item) is not Fraction for item in items):
        raise TNFRValueError(f"{label} must contain exact Fraction values")
    return items


def _materialize_exact_matrix(value: Any, label: str) -> ExactMatrix:
    rows = _materialize(value, label)
    if not rows:
        raise TNFRValueError(f"{label} must be nonempty")
    return tuple(
        _materialize_exact_vector(row, f"{label}[{index}]")
        for index, row in enumerate(rows)
    )


def _materialize_exact_partitions(value: Any) -> ExactPartitionFamily:
    rows = _materialize(value, "partitions")
    if not rows:
        raise TNFRValueError("partitions must contain at least one partition")
    return tuple(
        _materialize_exact_vector(row, f"partitions[{index}]")
        for index, row in enumerate(rows)
    )


def _strict_exact_vector(value: Any, width: int | None = None) -> bool:
    return bool(
        type(value) is tuple
        and (width is None or len(value) == width)
        and all(type(item) is Fraction for item in value)
    )


def _strict_exact_matrix(value: Any, width: int | None = None) -> bool:
    if type(value) is not tuple or not value:
        return False
    size = len(value) if width is None else width
    return bool(
        len(value) == size
        and all(_strict_exact_vector(row, size) for row in value)
    )


def _strict_exact_partitions(value: Any) -> bool:
    return bool(
        type(value) is tuple
        and value
        and all(bool(row) and _strict_exact_vector(row) for row in value)
    )


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


def _weighted_mean(values: ExactVector, metric: ExactVector) -> Fraction:
    total = sum(metric, Fraction(0))
    return sum(
        (
            weight * value
            for weight, value in zip(metric, values, strict=True)
        ),
        Fraction(0),
    ) / total


def _centered_energy(values: ExactVector, metric: ExactVector) -> Fraction:
    center = _weighted_mean(values, metric)
    return sum(
        (
            weight * (value - center) ** 2
            for weight, value in zip(metric, values, strict=True)
        ),
        Fraction(0),
    ) / 2


def _max_abs(values: ExactVector) -> Fraction:
    return max((abs(value) for value in values), default=Fraction(0))


def _negative_exp_bounds(exponent: Fraction) -> tuple[Fraction, Fraction]:
    """Enclose ``exp(-exponent)`` with a bounded integer exponent."""

    if type(exponent) is not Fraction or exponent < 0:
        raise TNFRValueError("exponential exponent must be a nonnegative Fraction")
    if exponent > _MAX_RATIONAL_EXPONENT:
        raise TNFRValueError(
            "exact rational exponential enclosure supports exponent <= 4096"
        )
    integer = exponent.numerator // exponent.denominator
    remainder = exponent - integer
    unit_lower, unit_upper = exp_unit_bounds(Fraction(1))
    positive_lower = unit_lower**integer
    positive_upper = unit_upper**integer
    if remainder:
        remainder_lower, remainder_upper = exp_unit_bounds(remainder)
        positive_lower *= remainder_lower
        positive_upper *= remainder_upper
    if positive_lower <= 0 or positive_upper < positive_lower:
        raise RuntimeError("rational exponential enclosure is inconsistent")
    return 1 / positive_upper, 1 / positive_lower


def _partition_boundaries(partition: ExactPartition) -> tuple[Fraction, ...]:
    boundaries = [Fraction(0)]
    for duration in partition:
        boundaries.append(boundaries[-1] + duration)
    return tuple(boundaries)


def _proper_positive_subdivision(
    coarser: ExactPartition,
    finer: ExactPartition,
) -> bool:
    """Whether ``finer`` strictly subdivides the positive ``coarser`` mesh."""

    if not coarser or not finer or any(value <= 0 for value in (*coarser, *finer)):
        return False
    coarse_boundaries = _partition_boundaries(coarser)
    fine_boundaries = _partition_boundaries(finer)
    return bool(
        coarse_boundaries[0] == fine_boundaries[0]
        and coarse_boundaries[-1] == fine_boundaries[-1]
        and len(fine_boundaries) > len(coarse_boundaries)
        and set(coarse_boundaries).issubset(fine_boundaries)
    )


def _connected_support(conductance: ExactMatrix) -> bool:
    size = len(conductance)
    reached = {0}
    frontier = [0]
    while frontier:
        source = frontier.pop()
        for target, weight in enumerate(conductance[source]):
            if target not in reached and weight > 0:
                reached.add(target)
                frontier.append(target)
    return len(reached) == size


def _endpoint_enclosure(
    mean: Fraction,
    mode: ExactVector,
    factor_lower: Fraction,
    factor_upper: Fraction,
) -> tuple[ExactVector, ExactVector]:
    lower: list[Fraction] = []
    upper: list[Fraction] = []
    for value in mode:
        candidates = (
            mean + factor_lower * value,
            mean + factor_upper * value,
        )
        lower.append(min(candidates))
        upper.append(max(candidates))
    return tuple(lower), tuple(upper)


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


@dataclass(frozen=True, slots=True)
class ReversibleSingleEigenmodeEulerReferenceCertificate:
    """Sealed exact theorem and finite partition-family evaluation."""

    exact_conductance: ExactMatrix
    exact_nu_f: ExactVector
    exact_initial_epi: ExactVector
    exact_partitions: ExactPartitionFamily
    exact_degrees: ExactVector
    exact_reversible_metric: ExactVector
    exact_normalized_reversible_metric: ExactVector
    exact_generator: ExactMatrix
    exact_weighted_mean: Fraction
    exact_centered_mode: ExactVector
    exact_mode_eigenvalue: Fraction
    exact_mode_residual: ExactVector
    exact_mode_linf_norm: Fraction
    exact_mode_h_energy: Fraction
    exact_total_duration: Fraction
    exact_continuous_factor_lower_bound: Fraction
    exact_continuous_factor_upper_bound: Fraction
    exact_continuous_endpoint_lower_bound: ExactVector
    exact_continuous_endpoint_upper_bound: ExactVector
    exact_partition_hmax: ExactVector
    exact_scaled_partitions: ExactPartitionFamily
    exact_euler_segment_factors: ExactPartitionFamily
    exact_euler_factors: ExactVector
    exact_euler_endpoints: tuple[ExactVector, ...]
    exact_factor_error_lower_bounds: ExactVector
    exact_factor_error_upper_bounds: ExactVector
    exact_quadratic_factor_error_upper_bounds: ExactVector
    exact_hmax_factor_error_upper_bounds: ExactVector
    exact_linf_error_lower_bounds: ExactVector
    exact_linf_error_upper_bounds: ExactVector
    exact_linf_quadratic_error_upper_bounds: ExactVector
    exact_linf_hmax_error_upper_bounds: ExactVector
    exact_h_energy_error_lower_bounds: ExactVector
    exact_h_energy_error_upper_bounds: ExactVector
    exact_h_energy_quadratic_error_upper_bounds: ExactVector
    exact_h_energy_hmax_error_upper_bounds: ExactVector
    exact_euler_factor_improvements: ExactVector
    exact_quadratic_bound_improvements: ExactVector
    exact_linf_quadratic_bound_improvements: ExactVector
    exact_h_energy_quadratic_bound_improvements: ExactVector
    conditions: tuple[tuple[str, bool], ...]
    _proof_stamp: tuple[Any, ...] = field(default=(), repr=False, compare=False)

    def _proof_fields_are_intact(self) -> bool:
        """Rederive every theorem field from the four exact source inputs."""

        try:
            current = _certificate_values(self)
            observed_stamp = object.__getattribute__(self, "_proof_stamp")
            if not proof_stamps_are_identical(
                observed_stamp,
                _stamp_from_values(current),
            ):
                return False
            expected = _derive_values(
                object.__getattribute__(self, "exact_conductance"),
                object.__getattribute__(self, "exact_nu_f"),
                object.__getattribute__(self, "exact_initial_epi"),
                object.__getattribute__(self, "exact_partitions"),
            )
            return proof_stamps_are_identical(
                observed_stamp,
                _stamp_from_values(expected),
            )
        except BaseException:
            return False

    @property
    def scope(self) -> str:
        return _SCOPE

    @property
    def reference_certificate_certified(self) -> bool:
        return self._proof_fields_are_intact()

    @property
    def failed_conditions(self) -> tuple[str, ...]:
        if not self._proof_fields_are_intact():
            return ("reversible_single_eigenmode_proof_fields_intact",)
        return tuple(name for name, passed in self.conditions if not passed)

    @property
    def exact_modal_solution_enclosure_certified(self) -> bool:
        return self._proof_fields_are_intact()

    @property
    def exact_real_euler_error_bound_certified(self) -> bool:
        return self._proof_fields_are_intact()

    @property
    def exact_linf_error_bound_certified(self) -> bool:
        return self._proof_fields_are_intact()

    @property
    def exact_h_energy_error_bound_certified(self) -> bool:
        return self._proof_fields_are_intact()

    @property
    def strict_proper_subdivision_improvement_certified(self) -> bool:
        return bool(self._proof_fields_are_intact() and len(self.exact_partitions) > 1)

    @property
    def conditional_exact_real_partition_convergence_certified(self) -> bool:
        """Certify the h_max limit implication, not an observed runtime limit."""
        return self._proof_fields_are_intact()

    @property
    def binary64_asymptotic_convergence_certified(self) -> bool:
        return False

    @property
    def arbitrary_or_mixed_mode_initial_data_certified(self) -> bool:
        return False

    @property
    def directed_or_nonreversible_generator_certified(self) -> bool:
        return False

    @property
    def changing_generator_or_metric_certified(self) -> bool:
        return False

    @property
    def glyph_or_remesh_dynamics_certified(self) -> bool:
        return False

    @property
    def solver_order_certified(self) -> bool:
        return False

    @property
    def full_tnfr_stability_certified(self) -> bool:
        return False


_FIELD_NAMES = tuple(
    item.name
    for item in fields(ReversibleSingleEigenmodeEulerReferenceCertificate)
    if item.name != "_proof_stamp"
)


def _certificate_values(
    value: ReversibleSingleEigenmodeEulerReferenceCertificate,
) -> dict[str, Any]:
    if type(value) is not ReversibleSingleEigenmodeEulerReferenceCertificate:
        raise TypeError("certificate must have its canonical result type")
    return {
        name: object.__getattribute__(value, name)
        for name in _FIELD_NAMES
    }


def _stamp_from_values(values: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _PROOF_VERSION,
        structural_proof_signature(
            tuple((name, values[name]) for name in _FIELD_NAMES)
        ),
    )


def _derive_values(
    conductance: ExactMatrix,
    nu_f: ExactVector,
    initial_epi: ExactVector,
    partitions: ExactPartitionFamily,
) -> dict[str, Any]:
    if not _strict_exact_matrix(conductance):
        raise TNFRValueError("conductance must be a nonempty exact square matrix")
    size = len(conductance)
    if size < 2:
        raise TNFRValueError("conductance must contain at least two nodes")
    if not _strict_exact_vector(nu_f, size):
        raise TNFRValueError(
            "nu_f must be an exact Fraction vector matching conductance"
        )
    if not _strict_exact_vector(initial_epi, size):
        raise TNFRValueError(
            "initial_epi must be an exact Fraction vector matching conductance"
        )
    if not _strict_exact_partitions(partitions):
        raise TNFRValueError("partitions must be a nonempty exact partition family")

    zero_diagonal = all(conductance[index][index] == 0 for index in range(size))
    if not zero_diagonal:
        raise TNFRValueError("conductance diagonal must be exactly zero")
    nonnegative = all(value >= 0 for row in conductance for value in row)
    if not nonnegative:
        raise TNFRValueError("conductance entries must be nonnegative")
    symmetric = all(
        conductance[left][right] == conductance[right][left]
        for left in range(size)
        for right in range(left)
    )
    if not symmetric:
        raise TNFRValueError("conductance must be exactly symmetric")

    degrees = tuple(sum(row, Fraction(0)) for row in conductance)
    if any(value <= 0 for value in degrees):
        raise TNFRValueError("every conductance row must have positive strength")
    connected = _connected_support(conductance)
    if not connected:
        raise TNFRValueError("positive conductance support must be connected")
    if any(value <= 0 for value in nu_f):
        raise TNFRValueError("every capacity must be positive")

    metric = tuple(
        degree / capacity
        for degree, capacity in zip(degrees, nu_f, strict=True)
    )
    metric_total = sum(metric, Fraction(0))
    normalized_metric = tuple(value / metric_total for value in metric)
    generator = tuple(
        tuple(
            nu_f[row]
            * (
                (Fraction(1) if row == column else Fraction(0))
                - conductance[row][column] / degrees[row]
            )
            for column in range(size)
        )
        for row in range(size)
    )
    reversible_identity = all(
        metric[left] * generator[left][right]
        == metric[right] * generator[right][left]
        for left in range(size)
        for right in range(size)
    )
    if not reversible_identity:
        raise RuntimeError("derived generator lost exact H self-adjointness")

    mean = _weighted_mean(initial_epi, metric)
    mode = tuple(value - mean for value in initial_epi)
    if not any(value != 0 for value in mode):
        raise TNFRValueError("initial_epi must contain a nonuniform mode")
    action = _matrix_vector(generator, mode)
    pivot = next(index for index, value in enumerate(mode) if value != 0)
    eigenvalue = action[pivot] / mode[pivot]
    residual = tuple(
        observed - eigenvalue * value
        for observed, value in zip(action, mode, strict=True)
    )
    if eigenvalue <= 0 or any(value != 0 for value in residual):
        raise TNFRValueError(
            "the centered initial field is not one exact positive eigenmode"
        )
    if _weighted_mean(mode, metric) != 0:
        raise RuntimeError("derived mode is not exactly H-centered")

    totals = tuple(sum(partition, Fraction(0)) for partition in partitions)
    if any(duration <= 0 for partition in partitions for duration in partition):
        raise TNFRValueError("every partition duration must be positive")
    if any(total != totals[0] for total in totals[1:]):
        raise TNFRValueError("all partitions must have one exact total duration")
    total_duration = totals[0]
    if total_duration <= 0:
        raise TNFRValueError("partition duration must be positive")
    scaled = tuple(
        tuple(eigenvalue * duration for duration in partition)
        for partition in partitions
    )
    if any(value <= 0 or value >= 1 for row in scaled for value in row):
        raise TNFRValueError("every exact modal step mu*h must lie in (0, 1)")
    subdivisions = tuple(
        _proper_positive_subdivision(coarser, finer)
        for coarser, finer in zip(partitions, partitions[1:])
    )
    if not all(subdivisions):
        raise TNFRValueError(
            "successive partitions must form a strict proper-subdivision chain"
        )

    # Reject unsupported exponential ranges before materializing potentially
    # large exact products for every partition.
    exp_lower, exp_upper = _negative_exp_bounds(eigenvalue * total_duration)
    segment_factors = tuple(
        tuple(Fraction(1) - value for value in row) for row in scaled
    )
    euler_factors = tuple(
        math.prod(row, start=Fraction(1)) for row in segment_factors
    )
    hmax = tuple(max(partition) for partition in partitions)
    quadratic = tuple(
        sum((value * value for value in row), Fraction(0)) / 2
        for row in scaled
    )
    hmax_bounds = tuple(
        eigenvalue * eigenvalue * total_duration * value / 2
        for value in hmax
    )
    if any(
        bound > hmax_bound
        for bound, hmax_bound in zip(quadratic, hmax_bounds, strict=True)
    ):
        raise RuntimeError("quadratic Euler bounds exceed their h_max bounds")

    factor_error_lower = tuple(
        max(Fraction(0), exp_lower - factor) for factor in euler_factors
    )
    factor_error_upper = tuple(
        min(exp_upper - factor, bound)
        for factor, bound in zip(euler_factors, quadratic, strict=True)
    )
    if any(
        lower < 0 or upper < lower
        for lower, upper in zip(
            factor_error_lower,
            factor_error_upper,
            strict=True,
        )
    ):
        raise RuntimeError("rational factor-error enclosure is inconsistent")

    continuous_lower, continuous_upper = _endpoint_enclosure(
        mean,
        mode,
        exp_lower,
        exp_upper,
    )
    euler_endpoints = tuple(
        tuple(mean + factor * value for value in mode)
        for factor in euler_factors
    )
    mode_linf = _max_abs(mode)
    mode_energy = _centered_energy(mode, metric)
    linf_lower = tuple(mode_linf * value for value in factor_error_lower)
    linf_upper = tuple(mode_linf * value for value in factor_error_upper)
    linf_quadratic = tuple(mode_linf * value for value in quadratic)
    linf_hmax = tuple(mode_linf * value for value in hmax_bounds)
    energy_lower = tuple(mode_energy * value * value for value in factor_error_lower)
    energy_upper = tuple(mode_energy * value * value for value in factor_error_upper)
    energy_quadratic = tuple(mode_energy * value * value for value in quadratic)
    energy_hmax = tuple(mode_energy * value * value for value in hmax_bounds)

    factor_improvements = tuple(
        finer - coarser
        for coarser, finer in zip(euler_factors, euler_factors[1:])
    )
    quadratic_improvements = tuple(
        coarser - finer
        for coarser, finer in zip(quadratic, quadratic[1:])
    )
    linf_improvements = tuple(
        coarser - finer
        for coarser, finer in zip(
            linf_quadratic,
            linf_quadratic[1:],
        )
    )
    energy_improvements = tuple(
        coarser - finer
        for coarser, finer in zip(
            energy_quadratic,
            energy_quadratic[1:],
        )
    )
    strict_improvement = bool(
        not factor_improvements
        or (
            all(value > 0 for value in factor_improvements)
            and all(value > 0 for value in quadratic_improvements)
            and all(value > 0 for value in linf_improvements)
            and all(value > 0 for value in energy_improvements)
        )
    )
    if not strict_improvement:
        raise RuntimeError("proper subdivision did not improve the exact bounds")

    conditions = (
        ("square_zero_diagonal_conductance", zero_diagonal),
        ("symmetric_nonnegative_conductance", nonnegative and symmetric),
        (
            "connected_positive_strength_support",
            connected and all(value > 0 for value in degrees),
        ),
        ("positive_capacity", all(value > 0 for value in nu_f)),
        ("exact_reversible_metric_identity", reversible_identity),
        ("nonuniform_centered_initial_mode", mode_linf > 0),
        (
            "exact_positive_single_eigenmode_identity",
            eigenvalue > 0 and all(value == 0 for value in residual),
        ),
        (
            "positive_equal_duration_partitions",
            total_duration > 0 and all(total == total_duration for total in totals),
        ),
        (
            "strict_positive_modal_step_domain",
            all(0 < value < 1 for row in scaled for value in row),
        ),
        ("strict_proper_subdivision_chain", all(subdivisions)),
        (
            "rational_continuous_factor_enclosure",
            0 < exp_lower <= exp_upper,
        ),
        (
            "exact_euler_modal_recurrence",
            all(
                endpoint
                == tuple(mean + factor * value for value in mode)
                for endpoint, factor in zip(
                    euler_endpoints,
                    euler_factors,
                    strict=True,
                )
            ),
        ),
        (
            "quadratic_factor_error_bound",
            all(
                upper <= bound
                for upper, bound in zip(
                    factor_error_upper,
                    quadratic,
                    strict=True,
                )
            ),
        ),
        (
            "hmax_factor_error_bound",
            all(
                bound <= h_bound
                for bound, h_bound in zip(quadratic, hmax_bounds, strict=True)
            ),
        ),
        (
            "continuous_endpoint_enclosure",
            all(
                lower <= upper
                for lower, upper in zip(
                    continuous_lower,
                    continuous_upper,
                    strict=True,
                )
            ),
        ),
        (
            "linf_error_bounds",
            all(
                lower <= upper <= theorem
                for lower, upper, theorem in zip(
                    linf_lower,
                    linf_upper,
                    linf_quadratic,
                    strict=True,
                )
            ),
        ),
        (
            "h_energy_error_bounds",
            all(
                lower <= upper <= theorem
                for lower, upper, theorem in zip(
                    energy_lower,
                    energy_upper,
                    energy_quadratic,
                    strict=True,
                )
            ),
        ),
        (
            "strict_refinement_improvement_when_applicable",
            strict_improvement,
        ),
    )
    if not _strict_conditions(conditions) or not all(
        passed for _, passed in conditions
    ):
        raise RuntimeError("single-eigenmode theorem conditions are inconsistent")

    return {
        "exact_conductance": conductance,
        "exact_nu_f": nu_f,
        "exact_initial_epi": initial_epi,
        "exact_partitions": partitions,
        "exact_degrees": degrees,
        "exact_reversible_metric": metric,
        "exact_normalized_reversible_metric": normalized_metric,
        "exact_generator": generator,
        "exact_weighted_mean": mean,
        "exact_centered_mode": mode,
        "exact_mode_eigenvalue": eigenvalue,
        "exact_mode_residual": residual,
        "exact_mode_linf_norm": mode_linf,
        "exact_mode_h_energy": mode_energy,
        "exact_total_duration": total_duration,
        "exact_continuous_factor_lower_bound": exp_lower,
        "exact_continuous_factor_upper_bound": exp_upper,
        "exact_continuous_endpoint_lower_bound": continuous_lower,
        "exact_continuous_endpoint_upper_bound": continuous_upper,
        "exact_partition_hmax": hmax,
        "exact_scaled_partitions": scaled,
        "exact_euler_segment_factors": segment_factors,
        "exact_euler_factors": euler_factors,
        "exact_euler_endpoints": euler_endpoints,
        "exact_factor_error_lower_bounds": factor_error_lower,
        "exact_factor_error_upper_bounds": factor_error_upper,
        "exact_quadratic_factor_error_upper_bounds": quadratic,
        "exact_hmax_factor_error_upper_bounds": hmax_bounds,
        "exact_linf_error_lower_bounds": linf_lower,
        "exact_linf_error_upper_bounds": linf_upper,
        "exact_linf_quadratic_error_upper_bounds": linf_quadratic,
        "exact_linf_hmax_error_upper_bounds": linf_hmax,
        "exact_h_energy_error_lower_bounds": energy_lower,
        "exact_h_energy_error_upper_bounds": energy_upper,
        "exact_h_energy_quadratic_error_upper_bounds": energy_quadratic,
        "exact_h_energy_hmax_error_upper_bounds": energy_hmax,
        "exact_euler_factor_improvements": factor_improvements,
        "exact_quadratic_bound_improvements": quadratic_improvements,
        "exact_linf_quadratic_bound_improvements": linf_improvements,
        "exact_h_energy_quadratic_bound_improvements": energy_improvements,
        "conditions": conditions,
    }


def _seal(
    value: ReversibleSingleEigenmodeEulerReferenceCertificate,
) -> ReversibleSingleEigenmodeEulerReferenceCertificate:
    values = _certificate_values(value)
    return replace(value, _proof_stamp=_stamp_from_values(values))


def certify_reversible_single_eigenmode_euler_reference(
    conductance: Iterable[Iterable[Fraction]],
    *,
    nu_f: Iterable[Fraction],
    initial_epi: Iterable[Fraction],
    partitions: Iterable[Iterable[Fraction]],
) -> ReversibleSingleEigenmodeEulerReferenceCertificate:
    """Certify one exact reversible eigenmode over an ordered mesh family.

    Every scalar input must be an exact :class:`fractions.Fraction`. The
    function materializes each iterable once, derives ``A``, ``H`` and the
    eigenvalue from those values, and rejects projection of mixed-mode data.
    """

    exact_conductance = _materialize_exact_matrix(conductance, "conductance")
    exact_nu_f = _materialize_exact_vector(nu_f, "nu_f")
    exact_initial_epi = _materialize_exact_vector(initial_epi, "initial_epi")
    exact_partitions = _materialize_exact_partitions(partitions)
    values = _derive_values(
        exact_conductance,
        exact_nu_f,
        exact_initial_epi,
        exact_partitions,
    )
    result = ReversibleSingleEigenmodeEulerReferenceCertificate(
        **values,
        _proof_stamp=_stamp_from_values(values),
    )
    if not result.reference_certificate_certified:
        raise RuntimeError("constructed single-eigenmode certificate is inconsistent")
    return result
