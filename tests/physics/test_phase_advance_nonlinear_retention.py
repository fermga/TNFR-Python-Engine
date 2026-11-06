"""Eight constructed controls for finite nonlinear pulse retention.

The tests check exact ingredients of a local coordinate-change and comparison
proof. They integrate no trajectory and provide no evaluated amplitude cutoff.
The two-dimensional chart and Jordan examples below are explicitly algebraic
controls, not replacement TNFR laws or approximations used as observations.
"""

from fractions import Fraction as Q
from itertools import product

import mpmath as mp
import pytest
import sympy as sp

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._interval_taylor import Jet
from tnfr.mathematics._rational_interval import I, pi_interval, sqrt
from tnfr.physics._sine_flow import _held_sine_neighbor_field

R = sp.sqrt(5)
C = (R - 1) / 4
S = sp.sqrt((5 + R) / 8)
# Keep positive scales symbolic in matrix algebra; specialize them to the
# actual twist for every comparison with the original engine field.
ROOT_C = sp.symbols("root_c", positive=True)
SIN_ALPHA = sp.symbols("sin_alpha", positive=True)
EDGES = tuple(
    sorted(
        {
            tuple(sorted((2 * i + a, 2 * ((i + 1) % 5) + b)))
            for i in range(5)
            for a, b in product((0, 1), repeat=2)
        }
    )
)


def _clean(value):
    return sp.simplify(sp.expand(value))


def _zero(matrix):
    assert sp.Matrix(matrix).applyfunc(_clean) == sp.zeros(*matrix.shape)


def _physical(value):
    return value.subs({ROOT_C: sp.sqrt(C), SIN_ALPHA: S})


def _contains(interval, expression):
    # High precision is a cross-check of the independent symbolic value;
    # exact identities and inequalities carry the mathematical argument.
    with mp.workdps(90):
        expected = mp.mpf(str(sp.N(expression, 85)))
        lower = mp.mpf(interval.lo.numerator) / interval.lo.denominator
        upper = mp.mpf(interval.hi.numerator) / interval.hi.denominator
        if interval.lo == interval.hi:
            assert abs(expected - lower) < mp.mpf("1e-80")
        else:
            assert lower <= expected <= upper
        assert interval.width < Q(1, 2**65)


def _bilinear(left, right):
    """Actual quadratic coefficient B in (q,x), in tau=sqrt(c)*t/pi."""
    result = sp.zeros(20, 1)
    for i, j in EDGES:
        sign = 1 if (j // 2 - i // 2) % 5 == 1 else -1
        current = (
            -sign
            * SIN_ALPHA
            * (left[j] - left[i])
            * (right[j] - right[i])
            / (8 * ROOT_C)
        )
        result[10 + i] += current
        result[10 + j] -= current
    return result


@pytest.fixture(scope="module")
def geometry():
    fine, plus, minus = sp.zeros(10), sp.zeros(10, 5), sp.zeros(10, 5)
    for i, j in EDGES:
        fine[i, i] += 1
        fine[j, j] += 1
        fine[i, j] -= 1
        fine[j, i] -= 1
    for i in range(5):
        plus[2 * i, i] = plus[2 * i + 1, i] = 1
        minus[2 * i, i], minus[2 * i + 1, i] = 1, -1
    centered = sp.eye(10) - sp.ones(10) / 10
    internal = minus * minus.T / 2
    collective = centered - internal
    collective_laplacian = fine - 4 * internal
    low = ((5 + R) * collective - collective_laplacian) / (2 * R)
    high = (collective_laplacian - (5 - R) * collective) / (2 * R)
    spatial = (internal, low, high)
    laplace_values = (4, 5 - R, 5 + R)
    oscillators = tuple(
        sp.Matrix(((1, -sign * sp.I / ROOT_C), (sign * sp.I * ROOT_C, 1))) / 2
        for sign in (1, -1)
    )
    projectors = tuple(
        sp.kronecker_product(oscillator, projector)
        for projector in spatial
        for oscillator in oscillators
    )
    frequencies = tuple(
        sp.Rational(1, 4) * sign * value for value in laplace_values for sign in (1, -1)
    )
    generator = sp.BlockMatrix(
        [[sp.zeros(10), fine / (4 * ROOT_C)], [-ROOT_C * fine / 4, sp.zeros(10)]]
    ).as_explicit()
    return dict(
        L=fine,
        plus=plus,
        minus=minus,
        centered=centered,
        spatial=spatial,
        laplace_values=laplace_values,
        oscillators=oscillators,
        projectors=projectors,
        frequencies=frequencies,
        A=generator,
    )


def test_shared_field_has_the_declared_complete_quadratic_jet(geometry):
    neighbors = tuple(
        tuple((j, 1, Q(1)) for j in range(10) if tuple(sorted((i, j))) in EDGES)
        for i in range(10)
    )
    model = RelationalExchangeModel(
        1, epi_weight=0, phase_weight=1, phase_domain="regular"
    )
    field = _held_sine_neighbor_field(model, (4,) * 10, (Q(1),) * 10, neighbors)
    pi, root = pi_interval(), sqrt((sqrt(I(5)) - 1) / 4)
    directions = []
    for seed in (1, 3, 5):
        values = [Q((seed * i + 7) % 17 - 8, 16) for i in range(20)]
        for start in (0, 10):
            mean = sum(values[start : start + 10]) / 10
            values[start : start + 10] = [
                value - mean for value in values[start : start + 10]
            ]
        direction = sp.Matrix(values)
        directions.append(direction)
        state = tuple(Jet((0, values[10 + i], 0)) for i in range(10)) + tuple(
            Jet((2 * pi * Q(i // 2, 5), values[i], 0)) for i in range(10)
        )
        original = field(state)
        rates = tuple(value * (pi / root) for value in original[10:] + original[:10])
        linear = geometry["A"] * direction
        quadratic = _bilinear(direction, direction)
        for i, rate in enumerate(rates):
            _contains(rate.coeffs[0], 0)
            _contains(rate.coeffs[1], _physical(linear[i]))
            _contains(rate.coeffs[2], _physical(quadratic[i]))
        assert all(value == 0 for value in quadratic[:10])
        assert _clean(sum(quadratic[10:])) == 0
    left, right = directions[:2]
    _zero(_bilinear(left, right) - _bilinear(right, left))
    _zero(
        _bilinear(left + right, left + right)
        - _bilinear(left, left)
        - 2 * _bilinear(left, right)
        - _bilinear(right, right)
    )
    for origin in (
        sp.ones(10, 1).col_join(sp.zeros(10, 1)),
        sp.zeros(10, 1).col_join(sp.ones(10, 1)),
    ):
        _zero(_bilinear(origin, left))


def test_six_centered_projectors_retain_every_internal_and_collective_direction(
    geometry,
):
    spatial = geometry["spatial"]
    for i, projector in enumerate(spatial):
        _zero(geometry["L"] * projector - geometry["laplace_values"][i] * projector)
        assert _clean(sp.trace(projector)) == (5 if i == 0 else 2)
        for j, other in enumerate(spatial):
            _zero(projector * other - (projector if i == j else sp.zeros(10)))
    for i, oscillator in enumerate(geometry["oscillators"]):
        for j, other in enumerate(geometry["oscillators"]):
            _zero(oscillator * other - (oscillator if i == j else sp.zeros(2)))
    total = sp.zeros(20)
    for i, (projector, frequency) in enumerate(
        zip(geometry["projectors"], geometry["frequencies"])
    ):
        _zero(geometry["A"] * projector - sp.I * frequency * projector)
        _zero(projector.conjugate() - geometry["projectors"][i ^ 1])
        assert _clean(sp.trace(projector)) == (5 if i < 2 else 2)
        total += projector
    _zero(total - sp.diag(geometry["centered"], geometry["centered"]))
    assert _clean(sp.trace(total)) == 18


def test_all_quadratic_denominators_and_the_actual_homological_identity(geometry):
    integer_parts = ((4, 0), (-4, 0), (5, -1), (-5, 1), (5, 1), (-5, -1))
    lower_r, upper_r = Q(223, 100), Q(56, 25)
    assert lower_r**2 < 5 < upper_r**2
    frequencies = geometry["frequencies"]
    count = 0
    coefficient = sp.symbols("coefficient")
    for j, k, ell in product(range(6), repeat=3):
        rational = integer_parts[j][0] + integer_parts[k][0] - integer_parts[ell][0]
        radical = integer_parts[j][1] + integer_parts[k][1] - integer_parts[ell][1]
        endpoints = sorted(
            ((rational + radical * lower_r) / 4, (rational + radical * upper_r) / 4)
        )
        assert endpoints[0] > Q(1, 10) or endpoints[1] < -Q(1, 10)
        denominator = sp.I * (frequencies[j] + frequencies[k] - frequencies[ell])
        assert _clean(denominator - sp.I * (rational + radical * R) / 4) == 0
        assert _clean(denominator * (coefficient / denominator) - coefficient) == 0
        conjugate = sp.I * (
            frequencies[j ^ 1] + frequencies[k ^ 1] - frequencies[ell ^ 1]
        )
        assert _clean(sp.conjugate(denominator) - conjugate) == 0
        count += 1
    assert count == 216
    # Check a nonzero actual modal coupling using the full A and B, rather
    # than only testing a scalar denominator against its own definition.
    probe = (geometry["minus"] * sp.Matrix((1, 2, 3, 4, 5))).col_join(sp.zeros(10, 1))
    source = geometry["projectors"][0] * probe
    output = geometry["projectors"][2]
    target = (output * _bilinear(source, source)).applyfunc(_clean)
    assert any(value != 0 for value in target)
    denominator = sp.I * (2 * frequencies[0] - frequencies[2])
    correction = target / denominator
    polarized = (
        output
        * (
            _bilinear(geometry["A"] * source, source)
            + _bilinear(source, geometry["A"] * source)
        )
        / denominator
    )
    _zero(polarized - geometry["A"] * correction - target)
    assert _clean(sum(correction[:10])) == _clean(sum(correction[10:])) == 0
    _zero(
        _bilinear(source.conjugate(), source.conjugate())
        - _bilinear(source, source).conjugate()
    )


def test_local_chart_cancels_quadratic_terms_and_has_a_cubic_remainder():
    a, b, scale = sp.symbols("a b scale", real=True)
    point = sp.Matrix((a, b))
    linear = sp.Matrix(((0, 1), (-1, 0)))
    correction = sp.Matrix((a**2 + 2 * a * b, a * b - b**2)) / 16
    # Independently expanded coefficients of DQ(w)*A*w-A*Q(w).
    quadratic = sp.Matrix((-2 * a**2 + a * b + 3 * b**2, 4 * a * b + b**2)) / 16
    _zero(correction.jacobian(point) * linear * point - linear * correction - quadratic)
    image = scale * point + scale**2 * correction
    substitution = dict(zip(point, image))
    pushed = linear * image + quadratic.subs(substitution, simultaneous=True)
    pulled_linear = (
        (sp.eye(2) + correction.jacobian(point).subs({a: scale * a, b: scale * b}))
        * scale
        * linear
        * point
    )
    residual = pushed - pulled_linear
    for degree in (0, 1, 2):
        _zero(residual.applyfunc(lambda value: sp.expand(value).coeff(scale, degree)))
    assert any(_clean(sp.expand(value).coeff(scale, 3)) != 0 for value in residual)
    # For this explicit bilinear map ||Q||_infinity<=3/16 on the unit ball.
    # The general proof uses 2*||Q||*r<=1/2 and hence Lipschitz3/2, inverse2.
    for first, second in product((-1, 1), repeat=2):
        derivative = correction.jacobian(point).subs({a: first, b: second})
        assert max(
            sum(abs(value) for value in row) for row in derivative.tolist()
        ) <= sp.Rational(3, 8)
    assert 1 + Q(3, 8) < Q(3, 2) and 1 / (1 - Q(3, 8)) < 2
    u, error, strength = sp.symbols("u error strength", nonnegative=True)
    cubic = strength * u**3
    remainder = cubic.subs(u, u + error) - cubic - sp.diff(cubic, u) * error
    assert _clean(remainder - strength * (3 * u * error**2 + error**3)) == 0
    assert (
        _clean(3 * strength * (u + error) * error**2 - remainder)
        == 2 * strength * error**3
    )


def test_uniform_slow_time_propagator_bound_keeps_neutral_shear(geometry):
    scaling = sp.diag(sp.eye(10), ROOT_C * sp.eye(10))
    inverse = sp.diag(sp.eye(10), sp.eye(10) / ROOT_C)
    canonical = inverse * geometry["A"] * scaling
    _zero(canonical.T + canonical)
    weight = sp.diag(sp.eye(10), sp.eye(10) / ROOT_C**2)
    _zero(geometry["A"].T * weight + weight * geometry["A"])
    assert C.is_positive is True and (1 - C).is_positive is True
    ca = sp.sqrt(20 / C)
    assert _clean(ca**2 - 20 / C) == 0
    epsilon, time, horizon, third, bound = sp.symbols(
        "epsilon time T_star L3 M", positive=True
    )
    perturbation = third * bound**2 / 2
    majorant = ca * sp.exp(ca * perturbation * epsilon**2 * time)
    assert (
        _clean(sp.diff(majorant, time) - ca * perturbation * epsilon**2 * majorant) == 0
    )
    assert majorant.subs(time, 0) == ca
    assert (
        _clean(
            majorant.subs(time, horizon / epsilon**2)
            - ca * sp.exp(ca * third * bound**2 * horizon / 2)
        )
        == 0
    )
    nilpotent = sp.Matrix(((0, 1), (0, 0)))
    transition = sp.eye(2) + epsilon**2 * time * nilpotent
    _zero(sp.diff(transition, time) - epsilon**2 * nilpotent * transition)
    assert (
        transition.subs(time, horizon / epsilon**2) == sp.eye(2) + horizon * nilpotent
    )
    assert sp.diff(transition[0, 1], time) == epsilon**2


def test_bootstrap_constants_give_an_eta_times_amplitude_initial_radius():
    k, remainder, epsilon, distance, time = sp.symbols(
        "K C epsilon delta time", positive=True
    )
    majorant = 2 * k * distance / (1 - 4 * k**2 * remainder * epsilon * distance * time)
    assert (
        _clean(sp.diff(majorant, time) - 2 * k * remainder * epsilon * majorant**2) == 0
    )
    assert majorant.subs(time, 0) == 2 * k * distance
    assert majorant.subs(distance, 0) == 0
    eta, horizon = sp.symbols("eta T_star", positive=True)
    endpoint = majorant.subs({distance: eta * epsilon, time: horizon / epsilon**2})
    assert (
        _clean(
            endpoint / (k * eta * epsilon)
            - 2 / (1 - 4 * k**2 * remainder * eta * horizon)
        )
        == 0
    )
    fraction = sp.symbols("fraction", nonnegative=True)
    assert (
        _clean(
            sp.Rational(8, 3)
            - 2 / (1 - fraction / 4)
            - 8 * (1 - fraction) / (3 * (4 - fraction))
        )
        == 0
    )
    # fraction<=1 is the admitted eta<=1/(16*K^2*C*T_star).
    assert Q(8, 3) < 4
    assert _clean((4 * k * eta * epsilon).subs(eta, 1 / (8 * k)) - epsilon / 2) == 0
    assert (
        _clean(
            (4 * k**2 * remainder * eta * horizon).subs(
                eta, 1 / (16 * k**2 * remainder * horizon)
            )
            - sp.Rational(1, 4)
        )
        == 0
    )
    assert sp.Rational(3, 2) * 4 * k * distance == 6 * k * distance
    assert (
        _clean(
            (12 * k * eta / sp.sqrt(C)).subs(eta, sp.sqrt(C) / (24 * k))
            - sp.Rational(1, 2)
        )
        == 0
    )


def test_common_phase_comparison_retains_free_means_and_controls_activity(geometry):
    mean_phase = geometry["plus"] * sp.Matrix((1, -1, 0, 0, 0))
    mean_form = geometry["plus"] * sp.Matrix((0, 1, -1, 0, 0))
    collective = mean_phase.col_join(mean_form)
    assert sum(mean_phase) == sum(mean_form) == 0
    assert max(abs(value) for value in collective) == 1
    _zero(geometry["minus"].T * mean_phase)
    _zero(geometry["minus"].T * mean_form)
    assert geometry["A"] * collective != sp.zeros(20, 1)
    # The independent mean directions cannot be removed by fitting each
    # internal phase. The theorem selects one common temporal shift only.
    radius, horizontal, vertical = sp.symbols("radius horizontal vertical", real=True)
    error_squared = horizontal**2 + vertical**2
    current_squared = (radius + horizontal) ** 2 + vertical**2
    assert (
        _clean(
            error_squared * current_squared
            - radius**2 * vertical**2
            - (radius * horizontal + error_squared) ** 2
        )
        == 0
    )
    # With ||error||<=radius/2 the real part stays positive, so the identity
    # controls the actual angle by arcsin(||error||/radius), with no lift jump.
    k, delta, epsilon = sp.symbols("K delta epsilon", positive=True)
    reference_radius = sp.sqrt(C / 2) * epsilon
    internal_error = sp.sqrt(2) * 6 * k * delta
    ratio = 12 * k * delta / (sp.sqrt(C) * epsilon)
    assert _clean(internal_error / reference_radius - ratio) == 0
    assert _clean((reference_radius / 2) ** 2 - C * epsilon**2 / 8) == 0
    angle, shift = sp.symbols("angle shift", real=True)
    relative = (angle + shift) - (2 * angle + shift)
    assert _clean(relative + angle) == 0


def test_tick_crossings_require_margins_and_keep_the_original_clock():
    # A small allowed error can change an endpoint counter. Values here are
    # exact phase/π readings, not an artificial trajectory.
    reference, nearby = sp.Rational(11, 12), sp.Rational(13, 12)
    assert nearby - reference == sp.Rational(1, 6)
    assert sp.floor(reference) == 0 and sp.floor(nearby) == 1
    for base in (Q(-7, 6), Q(-1, 6), Q(5, 6), Q(13, 6)):
        for error in (-Q(1, 3), Q(1, 3)):
            assert abs(sp.floor(base + error) - sp.floor(base)) <= 1
    speed, tolerance, crossing = sp.symbols("speed tolerance crossing", positive=True)
    phase_error = sp.symbols("phase_error", real=True)
    displaced = crossing - phase_error / speed
    assert _clean(speed * (displaced - crossing) + phase_error) == 0
    assert (
        _clean(displaced.subs(phase_error, tolerance) - crossing + tolerance / speed)
        == 0
    )
    assert (
        _clean(displaced.subs(phase_error, -tolerance) - crossing - tolerance / speed)
        == 0
    )
    collar, sinc = sp.symbols("c_r sinc_D", positive=True)
    lower_speed_tau = collar * sinc / sp.sqrt(C)
    original_time_error = sp.pi * (tolerance / lower_speed_tau) / sp.sqrt(C)
    assert _clean(original_time_error - sp.pi * tolerance / (collar * sinc)) == 0
    horizon, epsilon = sp.symbols("T_star epsilon", positive=True)
    original_horizon = sp.pi * horizon / (sp.sqrt(C) * epsilon**2)
    assert _clean(original_horizon * sp.sqrt(C) / sp.pi - horizon / epsilon**2) == 0
