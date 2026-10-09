"""Prospective Taylor-coefficient predictions for one prepared joint response.

This instrument advances auxiliary amplitude coefficients with the same explicit
Euler arithmetic as the production relational step. It never advances a graph.
Matching the time discretization separates amplitude approximation from the
leading Euler bias; refinement alone is not an exact-ODE error certificate.

The polynomial formulas expand the ideal positive-winding paired-C5 lock.
Their c, s and 1/pi are supplied by represented binary64 trigonometric constants,
and the shared exact split is applied to that rational specialization. Neither
this specialization nor a Decimal replay certifies exact irrational constants.
The production engine's represented reference can have a tiny nonzero drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
from fractions import Fraction as Q
from functools import lru_cache
import math

from tnfr.dynamics._euler_kernel import euler_update

from benchmarks.relational_local_composition import (
    _paired_ring_geometry,
    analyze_local_memory,
)


EPSILON = Q(1, 64)
HORIZON = Q(1, 4)
STEP_COUNTS = (64, 128, 256)
NODES = tuple(range(10))
_SUPPORT = _paired_ring_geometry(Q(1))[0]
EDGES = tuple((i, j) for i in NODES for j in NODES if i < j and _SUPPORT[i][j])
REFERENCE_PHASE = tuple(math.tau * (i % 5) / 5 for i in NODES)
INITIAL_FORM = tuple(float(EPSILON * value) for value in (0, 1, -1, -1, 1) + (0,) * 5)
INITIAL_PHASE = tuple(
    value + (float(EPSILON) if i == 0 else 0.0)
    for i, value in enumerate(REFERENCE_PHASE)
)
Number = float | Decimal
Vector = tuple[Number, ...]
Matrix = tuple[Vector, ...]


@dataclass(frozen=True)
class MemoryPredictionGeometry:
    """Fixed support and represented coefficient specialization, without state."""

    ring_cosine: Number
    ring_sine: Number
    inverse_pi: Number
    epi_weight: Number
    phase_weight: Number
    storage_scale: Number
    neighbors: tuple[tuple[tuple[int, Number, Number], ...], ...]
    form_laplacian: Matrix
    resultant_strength: Vector
    visible_projection: Matrix
    visible_lift: Matrix
    hidden_projection: Matrix
    hidden_lift: Matrix
    visible_generator: Matrix
    hidden_generator: Matrix
    visible_energy: Matrix
    hidden_energy: Matrix
    output_map: Matrix
    initial_direction: Vector
    decimal_precision: int | None
    coefficient_constants: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class PolynomialCoefficients:
    """Full-state Q2=one half D2F and Q3=one sixth D3F on a direction."""

    quadratic: Vector
    cubic: Vector


@dataclass(frozen=True)
class MemoryPrediction:
    """Euler amplitude expansion; all phase coordinates are lock deviations."""

    steps: int
    timestep: Number
    horizon: Number
    epsilon: Number
    linear_visible: Vector
    memory_visible: Vector
    direct_visible: Vector
    hidden_second_order: Vector
    linear_output: Vector
    memory_output: Vector
    direct_output: Vector
    y1: Vector
    h2: Vector
    y3: Vector
    y3_direct: Vector
    decimal_precision: int | None
    coefficient_constants: tuple[tuple[str, str], ...]
    arithmetic_scope: str
    scope: str = (
        "Fixed matched-Euler amplitude prediction from analytic ideal-lock Taylor "
        "formulas with represented c, s and inverse_pi. No measured response, "
        "coefficient fit, nonlinear graph solver, exact-ODE enclosure, exact "
        "irrational certificate or validated nonlinear remainder constant."
    )


def _precision(value):
    if value is not None and (type(value) is not int or value != 50):
        raise ValueError("decimal_precision must be None (binary64) or 50")
    return value


def _number(value, precision):
    if isinstance(value, bool) or not isinstance(value, (int, float, Q, Decimal)):
        raise TypeError("coefficient inputs must be finite real scalars")
    if precision is None:
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("coefficient inputs must be finite")
        if value and not result:
            raise ValueError("a nonzero coefficient input is lost in binary64")
        return result
    if isinstance(value, Q):
        result = Decimal(value.numerator) / Decimal(value.denominator)
    elif isinstance(value, float):
        result = Decimal.from_float(value)
    else:
        result = Decimal(value)
    if not result.is_finite():
        raise ValueError("coefficient inputs must be finite")
    return result


@lru_cache(maxsize=1)
def _exact_geometry():
    c, s, rho = math.cos(math.tau / 5), math.sin(math.tau / 5), 1 / math.pi
    split = analyze_local_memory(ring_cosine=Q(c), inverse_pi=Q(rho))
    b, _, _, strength = _paired_ring_geometry(Q(c))
    return split, b, strength, c, s, rho


def prepare_geometry(*, decimal_precision=None) -> MemoryPredictionGeometry:
    """Prepare the one fixed coefficient geometry without reading a response.

    Decimal50 starts from the same exact represented c, s and inverse_pi as
    binary64. Derived matrix entries are divided at fifty decimal digits, so
    its comparison also includes binary64 matrix materialization error. It
    is an evaluation-rounding control, not a bound on trigonometric error.
    """
    precision = _precision(decimal_precision)
    with localcontext() as context:
        context.prec = precision or 50
        context.rounding = ROUND_HALF_EVEN
        split, b, strength, c, s, rho = _exact_geometry()

        def number(value):
            return _number(value, precision)

        def matrix(values):
            return tuple(tuple(map(number, row)) for row in values)

        neighbors = []
        for i, row in enumerate(b):
            entries = []
            for j, value in enumerate(row):
                if i == j or not value:
                    continue
                same_ring = i // 5 == j // 5
                sine = s if (j - i) % 5 == 1 else -s
                entries.append(
                    (j, number(c if same_ring else 1), number(sine if same_ring else 0))
                )
            neighbors.append(tuple(entries))
        initial = (0, 1, -1, -1, 1, 0, 0, 0, 0, 0) + (1,) + (0,) * 9
        return MemoryPredictionGeometry(
            ring_cosine=number(c),
            ring_sine=number(s),
            inverse_pi=number(rho),
            epi_weight=number(Q(1, 2)),
            phase_weight=number(Q(1, 2)),
            storage_scale=number(1),
            neighbors=tuple(neighbors),
            form_laplacian=matrix(b),
            resultant_strength=tuple(map(number, strength)),
            visible_projection=matrix(split.visible_projection),
            visible_lift=matrix(split.visible_lift),
            hidden_projection=matrix(split.hidden_projection),
            hidden_lift=matrix(split.hidden_lift),
            visible_generator=matrix(split.visible_generator),
            hidden_generator=matrix(split.hidden_generator),
            visible_energy=matrix(split.visible_energy),
            hidden_energy=matrix(split.hidden_energy),
            output_map=matrix(split.composition.output_map),
            initial_direction=tuple(map(number, initial)),
            decimal_precision=precision,
            coefficient_constants=tuple(
                (name, value.hex())
                for name, value in (("c", c), ("s", s), ("inverse_pi", rho))
            ),
        )


def _matvec(matrix, vector):
    zero = vector[0] * 0
    return tuple(
        sum((a * b for a, b in zip(row, vector, strict=True)), zero) for row in matrix
    )


def _direction(geometry, values):
    result = tuple(_number(value, geometry.decimal_precision) for value in values)
    if len(result) != 20:
        raise ValueError("a direction must contain ten form and ten phase deviations")
    return result


def _node_terms(geometry, direction):
    form, phase = direction[:10], direction[10:]
    zero = form[0] * 0
    result = []
    for i, neighbors in enumerate(geometry.neighbors):
        strength = geometry.resultant_strength[i]
        q = sum((form[i] - form[j] for j, _, _ in neighbors), zero)
        a1 = a2 = b1 = b2 = b3 = zero
        for j, cosine, sine in neighbors:
            delta = phase[j] - phase[i]
            a1 -= sine * delta
            a2 -= cosine * delta * delta / 2
            b1 += cosine * delta
            b2 -= sine * delta * delta / 2
            b3 -= cosine * delta * delta * delta / 6
        result.append((q,) + tuple(value / strength for value in (a1, a2, b1, b2, b3)))
    return tuple(result)


def _polynomials(geometry, direction):
    quadratic_form, quadratic_phase, cubic_form, cubic_phase = [], [], [], []
    w_rho = geometry.phase_weight * geometry.inverse_pi
    k_rho = w_rho / geometry.storage_scale
    for i, (q, a1, a2, b1, b2, b3) in enumerate(_node_terms(geometry, direction)):
        alpha2 = b2 - a1 * b1
        alpha3 = b3 - a1 * b2 + (a1 * a1 - a2) * b1 - b1 * b1 * b1 / 3
        mobility = k_rho / geometry.resultant_strength[i]
        quadratic_form.append(w_rho * alpha2)
        quadratic_phase.append(-mobility * q * a1)
        cubic_form.append(w_rho * alpha3)
        cubic_phase.append(mobility * q * (a1 * a1 - a2 - b1 * b1 / 3))
    return PolynomialCoefficients(
        tuple(quadratic_form + quadratic_phase), tuple(cubic_form + cubic_phase)
    )


def evaluate_polynomials(geometry, direction) -> PolynomialCoefficients:
    """Evaluate analytic homogeneous Q2 and Q3; no finite differences or fit."""
    with localcontext() as context:
        context.prec = geometry.decimal_precision or 50
        context.rounding = ROUND_HALF_EVEN
        return _polynomials(geometry, _direction(geometry, direction))


def _quadratic_cross(geometry, left, right):
    left_terms, right_terms = _node_terms(geometry, left), _node_terms(geometry, right)
    form, phase = [], []
    w_rho = geometry.phase_weight * geometry.inverse_pi
    k_rho = w_rho / geometry.storage_scale
    zero = left[0] * 0
    for i, (lterm, rterm) in enumerate(zip(left_terms, right_terms, strict=True)):
        q, a1, _, b1, _, _ = lterm
        other_q, other_a1, _, other_b1, _, _ = rterm
        mixed_sine = sum(
            (
                sine * (left[10 + j] - left[10 + i]) * (right[10 + j] - right[10 + i])
                for j, _, sine in geometry.neighbors[i]
            ),
            zero,
        )
        strength = geometry.resultant_strength[i]
        form.append(w_rho * (-mixed_sine / strength - a1 * other_b1 - other_a1 * b1))
        phase.append(-k_rho * (q * other_a1 + other_q * a1) / strength)
    return tuple(form + phase)


def quadratic_cross(geometry, left, right) -> Vector:
    """Return Q2(left+right)-Q2(left)-Q2(right), with no extra factor half."""
    with localcontext() as context:
        context.prec = geometry.decimal_precision or 50
        context.rounding = ROUND_HALF_EVEN
        return _quadratic_cross(
            geometry, _direction(geometry, left), _direction(geometry, right)
        )


def _add(left, right):
    return tuple(a + b for a, b in zip(left, right, strict=True))


def _advance(values, rates, timestep):
    return tuple(
        euler_update(value, timestep, rate)
        for value, rate in zip(values, rates, strict=True)
    )


def predict(steps, *, decimal_precision=None) -> MemoryPrediction:
    """Predict the reserved prepared-even response on one declared Euler grid.

    Every coefficient RHS uses the same old snapshot. The memory-omitted
    control keeps the direct cubic term but excludes the bilinear feedback
    from h2. No endpoint, full-state history or graph field is consumed.
    """
    if type(steps) is not int or steps not in STEP_COUNTS:
        raise ValueError(f"steps must be one of the fixed grids {STEP_COUNTS}")
    precision = _precision(decimal_precision)
    with localcontext() as context:
        context.prec = precision or 50
        context.rounding = ROUND_HALF_EVEN
        geometry = prepare_geometry(decimal_precision=precision)
        epsilon = _number(EPSILON, precision)
        horizon = _number(HORIZON, precision)
        timestep = horizon / steps
        zero = _number(0, precision)
        y1 = _matvec(geometry.visible_projection, geometry.initial_direction)
        h2, y3, direct = (zero,) * 8, (zero,) * 10, (zero,) * 10
        for _ in range(steps):
            visible = _matvec(geometry.visible_lift, y1)
            hidden = _matvec(geometry.hidden_lift, h2)
            terms = _polynomials(geometry, visible)
            cross = _quadratic_cross(geometry, visible, hidden)
            linear_rate = _matvec(geometry.visible_generator, y1)
            hidden_rate = _add(
                _matvec(geometry.hidden_generator, h2),
                _matvec(geometry.hidden_projection, terms.quadratic),
            )
            cubic_rate = _add(
                _matvec(geometry.visible_generator, y3),
                _matvec(geometry.visible_projection, _add(terms.cubic, cross)),
            )
            direct_rate = _add(
                _matvec(geometry.visible_generator, direct),
                _matvec(geometry.visible_projection, terms.cubic),
            )
            y1, h2, y3, direct = (
                _advance(y1, linear_rate, timestep),
                _advance(h2, hidden_rate, timestep),
                _advance(y3, cubic_rate, timestep),
                _advance(direct, direct_rate, timestep),
            )
        linear = tuple(epsilon * value for value in y1)
        memory = _add(linear, tuple(epsilon**3 * value for value in y3))
        instantaneous = _add(linear, tuple(epsilon**3 * value for value in direct))
        return MemoryPrediction(
            steps=steps,
            timestep=timestep,
            horizon=horizon,
            epsilon=epsilon,
            linear_visible=linear,
            memory_visible=memory,
            direct_visible=instantaneous,
            hidden_second_order=tuple(epsilon**2 * value for value in h2),
            linear_output=_matvec(geometry.output_map, linear),
            memory_output=_matvec(geometry.output_map, memory),
            direct_output=_matvec(geometry.output_map, instantaneous),
            y1=y1,
            h2=h2,
            y3=y3,
            y3_direct=direct,
            decimal_precision=precision,
            coefficient_constants=geometry.coefficient_constants,
            arithmetic_scope=(
                "binary64 matrix materialization and coefficient arithmetic"
                if precision is None
                else "Decimal50 coefficient replay from identical exact represented constants"
            ),
        )
