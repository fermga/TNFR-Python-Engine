r"""Exact nested P5 observation geometry and uniform REMESH transport.

The reflection quotient ``((0,4),(1,3),(2,))`` is a genuine three-node
pure-EPI network. Its coordinates ``(a,p,c)`` close under diffusion and
uniform unclipped REMESH. Observing only ``(a,b)=(a,(2*p+c)/3)`` loses the
contrast ``u=c-p`` and produces the derived memory kernel.

For the separate fixed pure-EPI flow, retaining the hidden quadratic data
``(r*r,r*s,s*s)`` completes the reflection-invariant observation. It identifies
five-coordinate states up to reversal and has a closed induced rate. This
nonlinear observation is not a three-node state or an additional dynamics.

These are fixed-model algebraic observations. They neither replace nodal
diffusion by REMESH nor certify binary64 execution, clipping or full TNFR
state closure. Proof and scope: ``theory/DERIVED_EPI_MEMORY.md`` section 9.
"""

from __future__ import annotations

from collections.abc import Mapping, Set
from dataclasses import dataclass
from fractions import Fraction
from math import isqrt

from .._exact_time import exact_or_represented_real as _rational
from ..mathematics.krylov import exact_rank
from ._cycle_algebra import dot
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
    "P5ReflectionInvariants",
    "reduce_p5_state",
    "p5_reduction_geometry",
    "observe_p5_remesh_reduction",
    "observe_p5_reflection_invariants",
    "decode_p5_reflection_invariants",
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
class P5ReflectionInvariants:
    """Exact fixed-P5 observation and rates, complete up to path reflection.

    ``quadratic_invariants=(r*r,r*s,s*s)`` and ``orbit_epi=(a,p,c)`` retain
    the reflection orbit of the supplied scalar state. ``hidden_epi=(r,s)``
    records the original orientation for checking the induced rate; it is
    not an additional invariant. The chosen ``representative_epi`` decodes
    the invariant data, not an executed or newly selected physical state.

    ``jacobian_rank`` is the derivative rank of the six-output quadratic
    observation: five off the reflection-fixed set and three at r=s=0.
    It is not the local topological dimension of the quotient. The canonical
    representative is not asserted to be a globally smooth coordinate chart.
    """

    geometry: P5ReductionGeometry
    reduced: P5ReducedState
    orbit_epi: Vector
    hidden_epi: Vector
    quadratic_invariants: Vector
    orbit_rate: Vector
    hidden_rate: Vector
    invariant_rate: Vector
    representative_epi: Vector
    jacobian_rank: int
    exact_identity_checks: tuple[str, ...]
    scope: str


def _rational_square_root(value: Fraction) -> Fraction:
    """Return a nonnegative rational root, rejecting an algebraic-only lift."""
    numerator = isqrt(value.numerator)
    denominator = isqrt(value.denominator)
    if (
        numerator * numerator != value.numerator
        or denominator * denominator != value.denominator
    ):
        raise ValueError(
            "reflection invariants admit a real lift but this decoder requires "
            "an exact rational square root"
        )
    return Fraction(numerator, denominator)


def decode_p5_reflection_invariants(orbit_epi, quadratic_invariants) -> Vector:
    """Decode the canonical rational representative of one reflection orbit.

    Both ordered inputs have three finite exact/represented real coordinates.
    For J=(J00,J01,J11), require J00,J11>=0 and J00*J11=J01**2 exactly.
    These conditions characterize a positive-semidefinite rank-at-most-one
    real matrix. Rational decoding additionally requires an exact rational
    square root; a valid real cone point without that lift raises ValueError
    rather than being rounded by a floating square root.

    Choose r>0 when J00>0 and s=J01/r; otherwise choose r=0 and s>=0.
    Return (a+r,p+s,c,p-s,a-r). The all-zero hidden case decodes uniquely.
    This algebraic convention selects neither a physical orientation nor a
    smooth global chart, and supplies no evolution or graph realization.
    """
    orbit = _vector(orbit_epi, "orbit_epi")
    quadratic = _vector(quadratic_invariants, "quadratic_invariants")
    if len(orbit) != 3 or len(quadratic) != 3:
        raise ValueError(
            "orbit_epi and quadratic_invariants must each contain three coordinates"
        )
    j00, j01, j11 = quadratic
    if j00 < 0 or j11 < 0:
        raise ValueError("quadratic invariant diagonal entries must be nonnegative")
    if j00 * j11 != j01 * j01:
        raise ValueError(
            "quadratic invariants must satisfy the exact rank-one cone identity"
        )
    if j00:
        r = _rational_square_root(j00)
        s = j01 / r
    else:
        r, s = Fraction(0), _rational_square_root(j11)
    a, p, c = orbit
    return a + r, p + s, c, p - s, a - r


def observe_p5_reflection_invariants(epi, capacity=1) -> P5ReflectionInvariants:
    """Observe a closed nonlinear reflection quotient of fixed P5 diffusion.

    The declared fine model has fixed unit-conductance, unit-length P5 support,
    common positive capacity, pure EPI coefficient one, and equal held primitive
    phases. There are no other pressure sources or phase/capacity/support laws.
    No graph, solver, xi estimator, REMESH history or runtime event is executed.

    Reuse the existing even coordinates (a,p,c) and discarded coordinates
    (r,s,0,-s,-r). The latter satisfy r'=nu*(s-r), s'=nu*(r/2-s), hence
    J' = nu*(2*(J01-J00), J00/2+J11-2*J01, J01-2*J11).
    The even rate uses the existing orbit generator. Exact checks compare
    both sectors and the quadratic chain rule to the existing fine generator,
    verify the cone constraint and decode to the original state or reversal.
    These are induced identities on supplied states, not a new feedback law
    or a certificate of persistence, full tetrad dynamics or physical emergence.
    """
    reduced = reduce_p5_state(epi)
    geometry = p5_reduction_geometry(capacity)
    nu = geometry.capacity
    r, s = reduced.discarded_epi[:2]
    quadratic = (r * r, r * s, s * s)
    j00, j01, j11 = quadratic
    orbit_rate = tuple(-dot(row, reduced.orbit_epi) for row in geometry.orbit_generator)
    hidden_rate = (nu * (s - r), nu * (r / 2 - s))
    invariant_rate = (
        2 * nu * (j01 - j00),
        nu * (j00 / 2 + j11 - 2 * j01),
        nu * (j01 - 2 * j11),
    )
    representative = decode_p5_reflection_invariants(reduced.orbit_epi, quadratic)
    representative_reduced = reduce_p5_state(representative)
    decoded_r, decoded_s = representative_reduced.discarded_epi[:2]
    fine_rate = tuple(-dot(row, reduced.epi) for row in geometry.micro_generator)
    rate_parts = reduce_p5_state(fine_rate)
    fine_r_rate, fine_s_rate = rate_parts.discarded_epi[:2]
    chain_rule = (
        2 * r * fine_r_rate,
        s * fine_r_rate + r * fine_s_rate,
        2 * s * fine_s_rate,
    )
    zero = Fraction(0)
    jacobian = (
        *geometry.orbit_projection,
        (r, zero, zero, zero, -r),
        (s / 2, r / 2, zero, -r / 2, -s / 2),
        (zero, s, zero, -s, zero),
    )
    differential_rank = exact_rank(jacobian)
    checks = {
        "even rate is the projection of the fine nodal rate": orbit_rate
        == rate_parts.orbit_epi,
        "hidden rate is the odd part of the fine nodal rate": hidden_rate
        == (fine_r_rate, fine_s_rate),
        "quadratic rate obeys the fine nodal chain rule": invariant_rate == chain_rule,
        "quadratic data lie on the nonnegative rank-one cone": j00 >= 0
        and j11 >= 0
        and j00 * j11 == j01 * j01,
        "quadratic rate is tangent to the rank-one cone": invariant_rate[0] * j11
        + j00 * invariant_rate[2]
        - 2 * j01 * invariant_rate[1]
        == 0,
        "decoded representative retains even and quadratic data": representative_reduced.orbit_epi
        == reduced.orbit_epi
        and (decoded_r**2, decoded_r * decoded_s, decoded_s**2) == quadratic,
        "decoded representative is the original state or its reflection": representative
        in (reduced.epi, reduced.epi[::-1]),
        "fine generator commutes with reflection": all(
            geometry.micro_generator[i][j] == geometry.micro_generator[4 - i][4 - j]
            for i in range(5)
            for j in range(5)
        ),
        "quadratic observation differential rank": differential_rank
        == (5 if r or s else 3),
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError(f"exact P5 reflection-invariant identities failed: {failed}")
    return P5ReflectionInvariants(
        geometry=geometry,
        reduced=reduced,
        orbit_epi=reduced.orbit_epi,
        hidden_epi=(r, s),
        quadratic_invariants=quadratic,
        orbit_rate=orbit_rate,
        hidden_rate=hidden_rate,
        invariant_rate=invariant_rate,
        representative_epi=representative,
        jacobian_rank=differential_rank,
        exact_identity_checks=tuple(checks),
        scope=(
            "Exact induced reflection-invariant observation of fixed unit-support/unit-length "
            "P5 pure-EPI diffusion with common positive capacity and equal held primitive phases. "
            "Five-coordinate scalar states are retained up to reflection; the six output "
            "coordinates obey one cone constraint. Jacobian rank is not local topological "
            "dimension. No source, phase/capacity/support law, solver, xi reimplementation, "
            "REMESH/runtime execution, persistence or physical emergence is certified."
        ),
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
