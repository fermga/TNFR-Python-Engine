"""Eight finite controls for the complete free-collective interaction law.

All polynomial coefficients are exact. Formal amplitude jets evaluate the
shared field, not trajectories. The normal form retains nine complex modes;
its conserved actions are not asserted to be exact full-flow invariants.
"""

from collections import Counter
from fractions import Fraction as Q
from itertools import combinations_with_replacement, product
from math import factorial

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
ZETA = C + sp.I * S
TANGENT = S / C
EDGES = tuple(
    sorted(
        {
            tuple(sorted((2 * i + a, 2 * ((i + 1) % 5) + b)))
            for i in range(5)
            for a, b in product((0, 1), repeat=2)
        }
    )
)
ZERO_POWER = (0,) * 18
SPATIAL_MODES = (0, 0, 0, 0, 0, 1, 4, 2, 3)
NORMS = (2,) * 5 + (10,) * 4


def _clean(value):
    return sp.simplify(sp.expand(value))


def _contains(interval, expression):
    with mp.workdps(90):
        expected = mp.mpf(str(sp.N(expression, 85)))
        lower = mp.mpf(interval.lo.numerator) / interval.lo.denominator
        upper = mp.mpf(interval.hi.numerator) / interval.hi.denominator
        if interval.lo == interval.hi:
            assert abs(expected - lower) < mp.mpf("1e-80")
        else:
            assert lower <= expected <= upper
        assert interval.width < Q(1, 2**65)


def _add_term(polynomial, powers, coefficient):
    if not coefficient:
        return
    value = polynomial.get(powers, coefficient - coefficient) + coefficient
    if value:
        polynomial[powers] = value
    else:
        polynomial.pop(powers, None)


def _sum_polynomials(left, right, multiplier):
    result = dict(left)
    for powers, coefficient in right.items():
        _add_term(result, powers, multiplier * coefficient)
    return result


def _linear_power(linear, degree, field):
    result = {}
    indices = tuple(sorted(linear))
    for choice in combinations_with_replacement(indices, degree):
        counts = Counter(choice)
        powers = [0] * 18
        multiplicity = factorial(degree)
        coefficient = field.one
        for index, count in counts.items():
            powers[index] = count
            multiplicity //= factorial(count)
            coefficient *= linear[index] ** count
        _add_term(result, tuple(powers), field.convert(multiplicity) * coefficient)
    return result


def _differentiate(polynomial, index, field):
    result = {}
    for powers, coefficient in polynomial.items():
        if powers[index]:
            lowered = list(powers)
            lowered[index] -= 1
            result[tuple(lowered)] = coefficient * field.convert(powers[index])
    return result


def _multiply(left, right):
    result = {}
    for first, a in left.items():
        for second, b in right.items():
            powers = tuple(x + y for x, y in zip(first, second))
            _add_term(result, powers, a * b)
    return result


def _poisson_over_i(left, right, field):
    # {a_j,bar(a_j)}=2*i/||v_j||^2 for the unnormalized spatial basis.
    result = {}
    for index, norm in enumerate(NORMS):
        la, lb = (_differentiate(left, j, field) for j in (index, index + 9))
        ra, rb = (_differentiate(right, j, field) for j in (index, index + 9))
        factor = field.convert(sp.Rational(2, norm))
        result = _sum_polynomials(result, _multiply(la, rb), factor)
        result = _sum_polynomials(result, _multiply(lb, ra), -factor)
    return result


def _frequency_parts(powers):
    net = [powers[j] - powers[j + 9] for j in range(9)]
    internal, low, high = sum(net[:5]), sum(net[5:7]), sum(net[7:])
    return 4 * internal + 5 * low + 5 * high, -low + high


def _frequency(powers, field, radical):
    rational, irrational = _frequency_parts(powers)
    return (
        field.convert(sp.Rational(rational, 4))
        + field.convert(sp.Rational(irrational, 4)) * radical
    )


def _conjugate(coefficient, field, zeta):
    value = field.zero
    for rational in coefficient.to_list():
        value = value * zeta**4 + field.convert(rational)
    return value


def _intensity_power(first, second=None):
    powers = [0] * 18
    powers[first] += 1
    powers[first + 9] += 1
    if second is not None:
        powers[second] += 1
        powers[second + 9] += 1
    return tuple(powers)


@pytest.fixture(scope="module")
def algebra():
    field = sp.QQ.algebraic_field(ZETA)
    zeta = field.from_sympy(ZETA)
    radical = field.one + field.convert(2) * (zeta + zeta**4)
    tangent_squared = field.convert(5) + field.convert(2) * radical
    basis = [[field.zero for _ in range(9)] for _ in range(10)]
    for node in range(10):
        basis[node][node // 2] = field.convert(1 if node % 2 == 0 else -1)
        for mode in range(5, 9):
            basis[node][mode] = zeta ** ((SPATIAL_MODES[mode] * (node // 2)) % 5)
    h3, h4 = {}, {}
    for left, right in EDGES:
        sign = 1 if (right // 2 - left // 2) % 5 == 1 else -1
        linear = {}
        for index in range(9):
            value = (basis[right][index] - basis[left][index]) * field.convert(
                sp.Rational(1, 2)
            )
            if value:
                linear[index] = value
                linear[index + 9] = _conjugate(value, field, zeta)
        h3 = _sum_polynomials(
            h3, _linear_power(linear, 3, field), field.convert(sp.Rational(-sign, 24))
        )
        h4 = _sum_polynomials(
            h4, _linear_power(linear, 4, field), field.convert(sp.Rational(-1, 96))
        )
    divided = {
        powers: coefficient / _frequency(powers, field, radical)
        for powers, coefficient in h3.items()
    }
    k4 = _sum_polynomials(
        h4,
        _poisson_over_i(h3, divided, field),
        tangent_squared * field.convert(sp.Rational(1, 2)),
    )
    resonant = {p: a for p, a in k4.items() if _frequency_parts(p) == (0, 0)}
    return dict(
        field=field,
        basis=basis,
        zeta=zeta,
        radical=radical,
        tangent_squared=tangent_squared,
        h3=h3,
        h4=h4,
        divided=divided,
        k4=k4,
        resonant=resonant,
    )


def test_actual_full_cubic_jet_and_hamiltonian_normalization():
    neighbors = tuple(
        tuple((j, 1, Q(1)) for j in range(10) if tuple(sorted((i, j))) in EDGES)
        for i in range(10)
    )
    model = RelationalExchangeModel(
        1, epi_weight=0, phase_weight=1, phase_domain="regular"
    )
    field = _held_sine_neighbor_field(model, (4,) * 10, (Q(1),) * 10, neighbors)
    q = sp.symbols("q0:10", real=True)
    y = sp.symbols("y0:10", real=True)
    h2, h3, h4 = sp.S.Zero, sp.S.Zero, sp.S.Zero
    for left, right in EDGES:
        gap, form = q[right] - q[left], y[right] - y[left]
        sign = 1 if (right // 2 - left // 2) % 5 == 1 else -1
        h2 += (gap**2 + form**2) / 8
        h3 -= sign * TANGENT * gap**3 / 24
        h4 -= gap**4 / 96
    rows = tuple(sp.diff(h2, yi) for yi in y) + tuple(
        -sp.diff(h2 + h3 + h4, qi) for qi in q
    )
    pi, curvature = pi_interval(), (sqrt(I(5)) - 1) / 4
    root = sqrt(curvature)
    parameter = sp.symbols("parameter", real=True)
    for seed in (1, 3):
        values = [Q((seed * i + 7) % 17 - 8, 16) for i in range(20)]
        for start in (0, 10):
            mean = sum(values[start : start + 10]) / 10
            values[start : start + 10] = [x - mean for x in values[start : start + 10]]
        states = tuple(
            Jet((0, root * values[10 + i], 0, 0)) for i in range(10)
        ) + tuple(Jet((2 * pi * Q(i // 2, 5), values[i], 0, 0)) for i in range(10))
        original = field(states)
        rates = tuple(rate * pi / root for rate in original[10:]) + tuple(
            rate * pi / curvature for rate in original[:10]
        )
        substitution = dict(zip(q + y, [parameter * value for value in values]))
        for rate, row in zip(rates, rows):
            polynomial = sp.expand(row.subs(substitution))
            for degree in range(4):
                _contains(rate.coeffs[degree], polynomial.coeff(parameter, degree))
        assert all(_clean(sum(rows[part : part + 10])) == 0 for part in (0, 10))


def test_all_frequency_sectors_and_quartic_resonances_are_retained(algebra):
    field, zeta, basis = algebra["field"], algebra["zeta"], algebra["basis"]
    neighbors = tuple(
        tuple(j for j in range(10) if tuple(sorted((i, j))) in EDGES) for i in range(10)
    )
    eigenvalues = (
        [field.convert(4)] * 5
        + [field.convert(5) - algebra["radical"]] * 2
        + [field.convert(5) + algebra["radical"]] * 2
    )
    for mode in range(9):
        assert sum((basis[node][mode] for node in range(10)), field.zero) == field.zero
        for node in range(10):
            laplacian_value = sum(
                (
                    basis[node][mode] - basis[neighbor][mode]
                    for neighbor in neighbors[node]
                ),
                field.zero,
            )
            assert laplacian_value == eigenvalues[mode] * basis[node][mode]
        for other in range(9):
            inner = sum(
                (
                    _conjugate(basis[node][mode], field, zeta) * basis[node][other]
                    for node in range(10)
                ),
                field.zero,
            )
            assert inner == (
                field.convert(NORMS[mode]) if mode == other else field.zero
            )
    # Nine independent centered spatial vectors exhaust its dimension; both
    # oscillator orientations retain all eighteen canonical coordinates.
    assert len(basis[0]) == 10 - 1
    sectors = ((4, 0), (-4, 0), (5, -1), (-5, 1), (5, 1), (-5, -1))
    multiplicities = (5, 5, 2, 2, 2, 2)
    assert sum(multiplicities) == 18
    lower, upper = Q(223, 100), Q(56, 25)
    assert lower**2 < 5 < upper**2
    for choice in product(range(6), repeat=3):
        rational = sum(sectors[j][0] for j in choice)
        irrational = sum(sectors[j][1] for j in choice)
        assert (rational, irrational) != (0, 0)
    count = 0
    for choice in product(range(6), repeat=4):
        rational = sum(sectors[j][0] for j in choice)
        irrational = sum(sectors[j][1] for j in choice)
        net = tuple(choice.count(2 * j) - choice.count(2 * j + 1) for j in range(3))
        resonant = (rational, irrational) == (0, 0)
        assert resonant == (net == (0, 0, 0))
        if resonant:
            count += 1
        else:
            endpoints = sorted(
                (
                    (rational + irrational * lower) / 4,
                    (rational + irrational * upper) / 4,
                )
            )
            assert endpoints[0] > 0 or endpoints[1] < 0
    # Six arrangements within one band, twenty-four for each band pair.
    assert count == 3 * 6 + 3 * 24
    assert abs(-5) + abs(2) + abs(2) == 9


def test_canonical_lie_sign_and_cubic_homological_equation(algebra):
    field, radical = algebra["field"], algebra["radical"]
    assert algebra["h3"] and algebra["divided"]
    for powers, value in algebra["h3"].items():
        assert _frequency(powers, field, radical) * algebra["divided"][powers] == value
    # An independently solved canonical cubic example fixes the pullback sign.
    q, p, strength, epsilon = sp.symbols("q p strength epsilon", real=True)
    h2, h3 = (q**2 + p**2) / 2, strength * q**3
    generator = -strength * (q**2 * p + 2 * p**3 / 3)

    def bracket(f, g):
        return sp.diff(f, q) * sp.diff(g, p) - sp.diff(f, p) * sp.diff(g, q)

    assert _clean(bracket(h2, generator) + h3) == 0
    point = sp.Matrix((q, p))
    vector = sp.Matrix((sp.diff(generator, p), -sp.diff(generator, q)))
    image = (
        epsilon * point
        + epsilon**2 * vector
        + epsilon**3 * vector.jacobian(point) * vector / 2
    )
    pulled = (h2 + h3).subs(dict(zip(point, image)), simultaneous=True)
    assert _clean(sp.expand(pulled).coeff(epsilon, 3)) == 0
    assert _clean(sp.expand(pulled).coeff(epsilon, 4) - bracket(h3, generator) / 2) == 0


def test_complete_resonant_hamiltonian_keeps_symmetries_and_three_actions(algebra):
    field, zeta, resonant = algebra["field"], algebra["zeta"], algebra["resonant"]
    assert resonant and any(sum(p[5:9]) + sum(p[14:]) > 0 for p in resonant)
    transformed = {}
    for powers, value in resonant.items():
        assert sum(powers) == 4 and _frequency_parts(powers) == (0, 0)
        conjugate_powers = powers[9:] + powers[:9]
        assert resonant[conjugate_powers] == _conjugate(value, field, zeta)
        for site in range(5):
            assert (powers[site] + powers[site + 9]) % 2 == 0
        for sector in (range(5), range(5, 7), range(7, 9)):
            assert sum(powers[j] - powers[j + 9] for j in sector) == 0
        shifted = list(powers)
        for offset in (0, 9):
            for site in range(5):
                shifted[offset + (site + 1) % 5] = powers[offset + site]
        momentum = sum(
            SPATIAL_MODES[j] * (powers[j] - powers[j + 9]) for j in range(5, 9)
        )
        _add_term(transformed, tuple(shifted), value * zeta ** (momentum % 5))
        internal_degree = sum(powers[:5]) + sum(powers[9:14])
        if internal_degree == 0:
            assert all(powers[j] == powers[j + 9] for j in range(5, 9))
        if internal_degree == 2:
            assert sum(powers[j] * powers[j + 9] for j in range(5)) == 1
    assert transformed == resonant
    # The positive sector actions use weights ||v||^2/2: one and five.
    assert tuple(Q(norm, 2) for norm in NORMS) == (Q(1),) * 5 + (Q(5),) * 4


def test_zero_collective_polynomial_recovers_the_existing_internal_generator(algebra):
    field, resonant = algebra["field"], algebra["resonant"]
    adjacency, difference = sp.zeros(5), sp.zeros(5)
    for site in range(5):
        adjacency[site, (site + 1) % 5] = adjacency[site, (site - 1) % 5] = 1
        difference[site, (site + 1) % 5], difference[site, (site - 1) % 5] = 1, -1
    laplacian, origin = 2 * sp.eye(5) - adjacency, sp.ones(5) / 5
    inverse = (laplacian + origin).inv() - origin
    n = sp.eye(5) + adjacency - (5 + 2 * R) * difference * inverse * difference
    m = (
        adjacency
        - (5 + 2 * R)
        * difference
        * (laplacian**2 - 16 * sp.eye(5)).inv()
        * laplacian
        * difference
    )
    expected = {}
    for i, j in product(range(5), repeat=2):
        _add_term(
            expected, _intensity_power(i, j), field.from_sympy(_clean(-n[i, j] / 32))
        )
        powers = [0] * 18
        powers[i + 9], powers[j] = 2, 2
        _add_term(expected, tuple(powers), field.from_sympy(_clean(-m[i, j] / 64)))
    actual = {p: a for p, a in resonant.items() if not any(p[5:9] + p[14:])}
    assert actual == expected
    assert sum(actual.values(), field.zero) == field.convert(sp.Rational(-5, 8))


def test_actual_collective_standing_wave_detunes_local_pulses(algebra):
    field, zeta, resonant = algebra["field"], algebra["zeta"], algebra["resonant"]
    a2 = algebra["tangent_squared"]
    for mode, first in ((1, 5), (2, 7)):
        cosine = (zeta**mode + zeta ** ((-mode) % 5)) * field.convert(sp.Rational(1, 2))
        ell = field.one - cosine
        standing, traveling = {}, {}
        for powers, value in resonant.items():
            if sum(powers[:5]) + sum(powers[9:14]) != 2:
                continue
            active = (first, first + 1, first + 9, first + 10)
            if any(
                powers[j]
                for j in tuple(range(5, 9)) + tuple(range(14, 18))
                if j not in active
            ):
                continue
            internal = powers[:5] + (0,) * 4 + powers[9:14] + (0,) * 4
            _add_term(standing, internal, value * field.convert(sp.Rational(1, 4)))
            if powers[first + 1] == powers[first + 10] == 0:
                _add_term(traveling, internal, value)
        rates = []
        for site in range(5):
            cos_double = (
                zeta ** ((2 * mode * site) % 5) + zeta ** ((-2 * mode * site) % 5)
            ) * field.convert(sp.Rational(1, 2))
            denominator = field.convert(2) + ell
            detuning = (
                ell
                * field.convert(sp.Rational(1, 8))
                * (
                    field.one
                    + a2 / denominator
                    - (cosine * (field.one + a2) + a2 / denominator) * cos_double
                )
            )
            assert standing[_intensity_power(site)] == -detuning
            uniform = (
                ell * field.convert(sp.Rational(1, 4)) * (field.one + a2 / denominator)
            )
            assert traveling[_intensity_power(site)] == -uniform
            rates.append(detuning)
        assert rates[0] != rates[1]
        # Mixed terms are diagonal |A_i|^2: their Hamiltonian row changes
        # phase directly, while the existing internal interaction can later
        # convert changed relative phases into intensity exchange.
        assert set(standing) == {_intensity_power(site) for site in range(5)}


def test_quartic_elimination_and_interaction_window_error_scaling(algebra):
    field, radical = algebra["field"], algebra["radical"]
    residual = {}
    assert len(algebra["k4"]) > len(algebra["resonant"])
    for powers, value in algebra["k4"].items():
        frequency = _frequency(powers, field, radical)
        if frequency:
            # S4=K4/(i*frequency), so {H2,S4}=-K4 off resonance.
            assert frequency * (value / frequency) == value
        else:
            residual[powers] = value
    assert residual == algebra["resonant"]
    epsilon, time, rate, remainder = sp.symbols("epsilon T L M", positive=True)
    bound = epsilon * remainder * (sp.exp(rate * time) - 1) / rate
    assert _clean(sp.diff(bound, time) - rate * bound - epsilon * remainder) == 0
    assert bound.subs(time, 0) == 0
    physical = epsilon * bound
    assert (
        _clean(physical / epsilon**2 - remainder * (sp.exp(rate * time) - 1) / rate)
        == 0
    )
    # A degree-five Hamiltonian remainder gives a degree-four field residual;
    # dividing z=epsilon*a and T=epsilon^2*tau leaves order epsilon.
    assert _clean(epsilon**4 / epsilon / epsilon**2 - epsilon) == 0


def test_reconstructed_pulse_readings_keep_activity_and_original_clock():
    epsilon, error, minimum = sp.symbols("epsilon E rho", positive=True)
    leading_radius = sp.sqrt(C) * minimum * epsilon
    state_error = sp.sqrt(2) * error * epsilon**2
    ratio = sp.sqrt(2 / C) * error * epsilon / minimum
    assert _clean(state_error / leading_radius - ratio) == 0
    assert _clean((leading_radius / 2) ** 2 - C * minimum**2 * epsilon**2 / 4) == 0
    sigma = sp.symbols("sigma", real=True)
    denominator = sp.cos(sigma) ** 2 + C * sp.sin(sigma) ** 2
    angular_speed = sp.sqrt(C) / denominator
    assert _clean(angular_speed * denominator - sp.sqrt(C)) == 0
    assert C.is_positive is True and (1 - C).is_positive is True
    horizon, time = sp.symbols("T_star t", positive=True)
    slow_time = epsilon**2 * sp.sqrt(C) * time / sp.pi
    original_horizon = sp.pi * horizon / (sp.sqrt(C) * epsilon**2)
    assert _clean(slow_time.subs(time, original_horizon) - horizon) == 0
    # A zero local amplitude leaves phase undefined despite positive total
    # sector action; the nonzero local-amplitude premise is indispensable.
    reference, perturbed = sp.Rational(11, 12), sp.Rational(13, 12)
    assert perturbed - reference == sp.Rational(1, 6)
    assert sp.floor(reference) == 0 and sp.floor(perturbed) == 1
    speed, tolerance, crossing = sp.symbols("m_tau b crossing", positive=True)
    phase_error = sp.symbols("phase_error", real=True)
    displaced = crossing - phase_error / speed
    assert _clean(speed * (displaced - crossing) + phase_error) == 0
    original_tick_bound = sp.pi * tolerance / (sp.sqrt(C) * speed)
    assert _clean(original_tick_bound * sp.sqrt(C) / sp.pi - tolerance / speed) == 0
    amplitudes = (0, 1, 0, 0, 0)
    assert sum(value**2 for value in amplitudes) > 0 and min(amplitudes) == 0
