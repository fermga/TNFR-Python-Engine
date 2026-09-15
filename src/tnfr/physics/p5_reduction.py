r"""Exact nested P5 observation geometry and uniform REMESH transport.

The reflection quotient ``((0,4),(1,3),(2,))`` is a genuine three-node
pure-EPI network. Its coordinates ``(a,p,c)`` close under diffusion and
uniform unclipped REMESH. Observing only ``(a,b)=(a,(2*p+c)/3)`` loses the
contrast ``u=c-p`` and produces the derived memory kernel.

These are fixed-model algebraic observations. They neither replace nodal
diffusion by REMESH nor certify binary64 execution, clipping or full TNFR
state closure. Proof and scope: ``theory/DERIVED_EPI_MEMORY.md`` section 9.
"""

from __future__ import annotations

from collections.abc import Mapping, Set
from dataclasses import dataclass
from fractions import Fraction

from .._exact_time import exact_or_represented_real as _rational
from ..mathematics.krylov import exact_rank
from .hybrid_operator_stability import _exact_matrix_product
from .remesh_history_stability import (
    UniformRemeshHistoryStabilityCertificate,
    UniformRemeshHistoryTransitionObservation,
    certify_uniform_remesh_history_stability,
    observe_uniform_remesh_history_transition,
)

__all__ = [
    "P5ReducedState",
    "P5ReductionGeometry",
    "P5RemeshReduction",
    "reduce_p5_state",
    "p5_reduction_geometry",
    "observe_p5_remesh_reduction",
]

Vector = tuple[Fraction, ...]
Matrix = tuple[Vector, ...]


def _vector(values, name: str) -> Vector:
    if isinstance(values, (str, bytes, bytearray, Mapping, Set)):
        raise TypeError(f"{name} must be an ordered sequence")
    try:
        items = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be an ordered sequence") from exc
    if not items:
        raise ValueError(f"{name} must be nonempty")
    return tuple(_rational(item, name) for item in items)


def _matrix(rows) -> Matrix:
    return tuple(tuple(Fraction(value) for value in row) for row in rows)


def _transpose(matrix: Matrix) -> Matrix:
    return tuple(zip(*matrix, strict=True))


def _diagonal(values: Vector) -> Matrix:
    return tuple(
        tuple(value if i == j else Fraction(0) for j in range(len(values)))
        for i, value in enumerate(values)
    )


@dataclass(frozen=True)
class P5ReducedState:
    """A detached exact observation of five real EPI coordinates."""

    epi: Vector
    orbit_epi: Vector
    memory_epi: Vector
    discarded_epi: Vector
    conserved_mean: Fraction
    contrast_coefficients: tuple[Fraction, Fraction]
    max_reference_contrast: Fraction


def reduce_p5_state(epi) -> P5ReducedState:
    """Observe the fixed P5 coordinates, without evolving or reading a graph.

    ``orbit_epi=(a,p,c)`` are three genuine quotient EPI coordinates.
    ``memory_epi=(a,b,u)`` is a change of chart, with consensus direction
    ``(1,1,0)``; it must not be treated as three ordinary TNFR nodes.
    """
    values = _vector(epi, "epi")
    if len(values) != 5:
        raise ValueError("epi must contain five P5 coordinates")
    a = (values[0] + values[4]) / 2
    p = (values[1] + values[3]) / 2
    c = values[2]
    b, u = (2 * p + c) / 3, c - p
    r, s = (values[0] - values[4]) / 2, (values[1] - values[3]) / 2
    contrast = a - b
    c1, c2 = 2 * contrast / 3 - 4 * u / 9, contrast / 3 + 4 * u / 9
    amplitude = abs(contrast)
    if c2 and 0 < -c1 / (2 * c2) < 1:
        amplitude = max(amplitude, abs(-c1 * c1 / (4 * c2)))
    return P5ReducedState(
        epi=values,
        orbit_epi=(a, p, c),
        memory_epi=(a, b, u),
        discarded_epi=(r, s, Fraction(0), -s, -r),
        conserved_mean=(a + 3 * b) / 4,
        contrast_coefficients=(c1, c2),
        max_reference_contrast=amplitude,
    )


@dataclass(frozen=True)
class P5ReductionGeometry:
    """Exact fixed-path geometry, with a scoped minimal linear-state result."""

    capacity: Fraction
    micro_generator: Matrix
    micro_metric: Vector
    orbit_projection: Matrix
    orbit_lift: Matrix
    orbit_generator: Matrix
    orbit_metric: Vector
    memory_projection: Matrix
    memory_lift: Matrix
    memory_generator: Matrix
    memory_metric: Vector
    visible_projection: Matrix
    minimal_linear_dimension: int


def p5_reduction_geometry(capacity=1) -> P5ReductionGeometry:
    """Derive the reflection quotient and memory chart from fixed P5 support.

    Minimality concerns autonomous linear states containing both original
    macro outputs, for all micro initial states. It is not a claim of global
    nonlinear state completeness, or of full microscopic observability.
    """
    nu = _rational(capacity, "capacity")
    if nu <= 0:
        raise ValueError("capacity must be positive")
    degrees = (1, 2, 2, 2, 1)
    micro = tuple(
        tuple(
            nu if i == j else -nu / degrees[i] if abs(i - j) == 1 else Fraction(0)
            for j in range(5)
        )
        for i in range(5)
    )
    blocks = ((0, 4), (1, 3), (2,))
    lift = _matrix([[int(i in block) for block in blocks] for i in range(5)])
    projection = tuple(
        tuple(Fraction(int(i in block), len(block)) for i in range(5))
        for block in blocks
    )
    product = _exact_matrix_product
    orbit = product(product(projection, micro), lift)
    chart = _matrix([[1, 0, 0], [0, Fraction(2, 3), Fraction(1, 3)], [0, -1, 1]])
    inverse = _matrix([[1, 0, 0], [0, 1, Fraction(-1, 3)], [0, 1, Fraction(2, 3)]])
    memory_projection = product(chart, projection)
    memory_lift = product(lift, inverse)
    metric = tuple(Fraction(degree) / nu for degree in degrees)
    orbit_metric = tuple(sum(metric[i] for i in block) for block in blocks)
    memory_metric_matrix = product(
        product(_transpose(memory_lift), _diagonal(metric)), memory_lift
    )
    memory_metric = tuple(memory_metric_matrix[i][i] for i in range(3))
    visible = memory_projection[:2]
    dimension = exact_rank(visible + product(visible, micro))
    if (
        product(projection, micro) != product(orbit, projection)
        or product(micro, lift) != product(lift, orbit)
        or product(projection, lift) != _diagonal((Fraction(1),) * 3)
        or memory_metric_matrix != _diagonal(memory_metric)
        or dimension != 3
    ):
        raise RuntimeError("fixed P5 reduction identities are inconsistent")
    return P5ReductionGeometry(
        capacity=nu,
        micro_generator=micro,
        micro_metric=metric,
        orbit_projection=projection,
        orbit_lift=lift,
        orbit_generator=orbit,
        orbit_metric=orbit_metric,
        memory_projection=memory_projection,
        memory_lift=memory_lift,
        memory_generator=product(product(chart, orbit), inverse),
        memory_metric=memory_metric,
        visible_projection=visible,
        minimal_linear_dimension=dimension,
    )


@dataclass(frozen=True)
class P5RemeshReduction:
    """Exact finite-history comparison reusing existing REMESH certificates.

    The nested certificates establish the fixed recurrence and its temporal
    energy balance. They do not bind supplied history to a live executor.
    ``augmented_energy_split_residual`` contains before/after/drop differences
    between the fine result and the sum of orbit and discarded results.
    """

    geometry: P5ReductionGeometry
    certificate: UniformRemeshHistoryStabilityCertificate
    fine: UniformRemeshHistoryTransitionObservation
    orbit: UniformRemeshHistoryTransitionObservation
    discarded: UniformRemeshHistoryTransitionObservation
    projected_next: Vector
    commutation_residual: Vector
    augmented_energy_split_residual: Vector


def observe_p5_remesh_reduction(
    history, *, alpha, tau_local=1, tau_global=2, capacity=1
) -> P5RemeshReduction:
    """Compare one exact REMESH recurrence on fine and quotient histories.

    History is ordered newest first and has exactly active_max_delay+1 rows.
    Runtime history sufficiency and advancement are separate contracts. This
    observer excludes clipping, binary64 operation order and time-varying
    support/capacity; even exact commutation does not replace diffusion by echo.
    """
    geometry = p5_reduction_geometry(capacity)
    alpha_q = _rational(alpha, "alpha")
    if not 0 < alpha_q <= 1:
        raise ValueError("alpha must be in (0, 1]")
    certificate = certify_uniform_remesh_history_stability(
        alpha=alpha_q, tau_local=tau_local, tau_global=tau_global
    )
    if isinstance(history, (str, bytes, bytearray, Mapping, Set)):
        raise TypeError("history must be an ordered sequence of P5 fields")
    states = tuple(reduce_p5_state(row) for row in history)
    fine = observe_uniform_remesh_history_transition(
        certificate, (state.epi for state in states), geometry.micro_metric
    )
    orbit = observe_uniform_remesh_history_transition(
        certificate, (state.orbit_epi for state in states), geometry.orbit_metric
    )
    discarded = observe_uniform_remesh_history_transition(
        certificate, (state.discarded_epi for state in states), geometry.micro_metric
    )
    projected = reduce_p5_state(fine.exact_next_field).orbit_epi
    residual = tuple(
        left - right
        for left, right in zip(projected, orbit.exact_next_field, strict=True)
    )
    split = tuple(
        getattr(fine, name) - getattr(orbit, name) - getattr(discarded, name)
        for name in (
            "exact_augmented_energy_before",
            "exact_augmented_energy_after",
            "exact_energy_drop",
        )
    )
    if any(residual) or any(split):
        raise RuntimeError(
            "exact P5 REMESH reduction lost its transport or energy split"
        )
    return P5RemeshReduction(
        geometry=geometry,
        certificate=certificate,
        fine=fine,
        orbit=orbit,
        discarded=discarded,
        projected_next=projected,
        commutation_residual=residual,
        augmented_energy_split_residual=split,
    )
