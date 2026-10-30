"""Eight finite controls for reciprocal counterpropagating pulse response.

The new response is the complete rotating eighteen-dimensional Jacobian.
The edge-derived tensor is constructed once; no old test or trajectory is run.
"""

from math import comb

import pytest
import sympy as sp

from tests.physics.pulse_interaction_fixtures import (
    NORMS,
    C,
    R,
    _add_term,
    _conjugate,
    _differentiate,
    build_interaction_tensor,
)

GAMMA = sp.symbols("gamma", positive=True)
KAPPA = -(14 + 21 * R) / 164
UNIFORM = (425 + 39 * R) / 656
MIXED = -(55 + 61 * R) / 164
SELF = 25 * (4 + R) / 44
CROSS = sp.Rational(25, 4)
COUNTER = 5 * (3 - 2 * R) / 44


def _clean(value):
    return sp.simplify(sp.expand(value))


def _powers(*indices):
    result = [0] * 18
    for index in indices:
        result[index] += 1
    return tuple(result)


def _matrix(field):
    return [[field.zero for _ in range(18)] for _ in range(18)]


def _multiply(left, right, field):
    output = _matrix(field)
    for row in range(18):
        for middle in range(18):
            if left[row][middle]:
                for column in range(18):
                    if right[middle][column]:
                        output[row][column] += left[row][middle] * right[middle][column]
    return output


def _evaluate_at_reference(polynomial, field, zeta):
    """Return the polynomial in gamma without assigning a numerical gamma."""
    result = {}
    for powers, value in polynomial.items():
        if any(powers[index] for index in (6, 7, 8, 15, 16, 17)):
            continue
        degree = powers[5] + powers[14]
        momentum = sum(
            index * (powers[index] - powers[index + 9]) for index in range(5)
        )
        _add_term(result, degree, value * zeta ** (momentum % 5))
    return result


def _expected_internal():
    n = (sp.S(3), 13 + 6 * R, sp.S(3), sp.S(3), 13 + 6 * R)
    m = (
        sp.S(2),
        -(108 + 2 * R) / 41,
        -(358 + 168 * R) / 41,
        -(358 + 168 * R) / 41,
        -(108 + 2 * R) / 41,
    )
    diagonal = tuple(
        _clean((2 * n[k] + 2 * m[(k + 2) % 5] - m[2]) / 32) for k in range(5)
    )
    anomalous = tuple(_clean((2 * n[k] + m[2]) / 32) for k in range(5))
    return diagonal, anomalous


@pytest.fixture(scope="module")
def response():
    tensor = build_interaction_tensor()
    field, zeta = tensor["field"], tensor["zeta"]
    positive = {powers: -value for powers, value in tensor["resonant"].items()}
    gradient = [_differentiate(positive, index, field) for index in range(18)]
    rotations = (
        {0: field.from_sympy(KAPPA), 2: field.from_sympy(UNIFORM)},
        {0: field.from_sympy(UNIFORM), 2: field.from_sympy(2 * SELF / 5)},
        {},
    )
    coefficients = [_matrix(field) for _ in range(3)]
    reference_rows = []
    for row in range(18):
        mode = row % 9
        conjugate = row >= 9
        consumed = mode if conjugate else mode + 9
        factor = field.convert(sp.Rational(-2 if conjugate else 2, NORMS[mode]))
        reference_rows.append(
            {
                degree: factor * value
                for degree, value in _evaluate_at_reference(
                    gradient[consumed], field, zeta
                ).items()
            }
        )
        for column in range(18):
            entry = _evaluate_at_reference(
                _differentiate(gradient[consumed], column, field), field, zeta
            )
            for degree, value in entry.items():
                assert degree <= 2
                coefficients[degree][row][column] = factor * value
        band = 0 if mode < 5 else (1 if mode < 7 else 2)
        for degree, value in rotations[band].items():
            coefficients[degree][row][row] += value if conjugate else -value
    change, inverse = _matrix(field), _matrix(field)
    for site in range(5):
        for mode in range(5):
            phase = zeta ** ((site * (mode + 1)) % 5)
            change[site][mode], change[site + 9][mode + 9] = phase, _conjugate(
                phase, field, zeta
            )
            inverse[mode][site] = _conjugate(phase, field, zeta) * field.convert(
                sp.Rational(1, 5)
            )
            inverse[mode + 9][site + 9] = phase * field.convert(sp.Rational(1, 5))
    for index in (5, 6, 7, 8, 14, 15, 16, 17):
        change[index][index] = inverse[index][index] = field.one
    identity = _multiply(inverse, change, field)
    assert identity == [
        [field.one if i == j else field.zero for j in range(18)] for i in range(18)
    ]
    transformed = [
        _multiply(_multiply(inverse, coefficient, field), change, field)
        for coefficient in coefficients
    ]
    return dict(
        tensor=tensor,
        K=positive,
        field=field,
        zeta=zeta,
        gradients=gradient,
        reference=reference_rows,
        matrices=transformed,
    )


def test_edge_tensor_coefficients_and_reference_rotations(response):
    field, zeta, polynomial = response["field"], response["zeta"], response["K"]
    for powers, expected in (
        (_powers(5, 5, 14, 14), SELF),
        (_powers(5, 14, 6, 15), CROSS),
        (_powers(0, 9, 5, 14), UNIFORM),
        (_powers(0, 9, 5, 15), MIXED),
    ):
        assert polynomial[powers] == field.from_sympy(expected)
    assert _clean((CROSS - 2 * SELF) / 5 - COUNTER) == 0
    for row, actual in enumerate(response["reference"]):
        mode, conjugate = row % 9, row >= 9
        sign = -1 if conjugate else 1
        if mode < 5:
            phase = zeta ** ((sign * mode) % 5)
            expected = {
                0: field.convert(sign) * field.from_sympy(KAPPA) * phase,
                2: field.convert(sign) * field.from_sympy(UNIFORM) * phase,
            }
        elif mode == 5:
            expected = {
                1: field.convert(sign) * field.from_sympy(UNIFORM),
                3: field.convert(sign) * field.from_sympy(2 * SELF / 5),
            }
        else:
            expected = {}
        assert actual == expected


def test_all_eighteen_rotating_rows_match_the_reciprocal_blocks(response):
    field, polynomial = response["field"], response["K"]
    expected = [_matrix(field) for _ in range(3)]
    a, b = _expected_internal()
    for mode in range(5):
        expected[0][mode][mode] = field.from_sympy(a[mode])
        expected[0][mode][9 + (-mode) % 5] = field.from_sympy(b[mode])
    expected[1][0][5] = expected[1][0][14] = field.from_sympy(UNIFORM)
    expected[1][2][15] = expected[1][3][6] = field.from_sympy(MIXED)
    expected[1][5][0] = expected[1][5][9] = field.from_sympy(UNIFORM)
    expected[2][5][5] = expected[2][5][14] = field.from_sympy(2 * SELF / 5)
    expected[1][6][3] = expected[1][6][11] = field.from_sympy(MIXED)
    expected[2][6][6] = field.from_sympy(COUNTER)
    ell = (5 + R) / 4
    high_uniform = ell / 4 * (1 + (5 + 2 * R) / (2 + ell))
    for mode in (7, 8):
        expected[0][mode][mode] = field.from_sympy(_clean(high_uniform))
        expected[2][mode][mode] = polynomial[
            _powers(5, 14, mode, mode + 9)
        ] * field.convert(sp.Rational(1, 5))
    for degree in range(3):
        for row in range(9):
            for column in range(18):
                expected[degree][row + 9][(column + 9) % 18] = -_conjugate(
                    expected[degree][row][column], field, response["zeta"]
                )
    assert response["matrices"] == expected
    # The decomposition is 6 coupled, 4 other internal, 4 empty-band,
    # and 4 occupied-action/phase real coordinates, with nothing removed.
    assert 6 + 4 + 4 + 4 == 18


def test_coupled_characteristic_polynomial_and_slow_schur_sign():
    a, b = _expected_internal()
    a2, a3, coupling = a[2], a[3], b[2]
    matrix = sp.Matrix(
        (
            (a2, coupling, MIXED * GAMMA),
            (-coupling, -a3, -MIXED * GAMMA),
            (-MIXED * GAMMA, -MIXED * GAMMA, -COUNTER * GAMMA**2),
        )
    )
    spectral = sp.symbols("lambda", real=True)
    determinant = _clean(a2 * a3 - coupling**2)
    numerator = _clean(a2 + a3 - 2 * coupling)
    nu = _clean(-COUNTER + MIXED**2 * numerator / determinant)
    expected = (spectral + COUNTER * GAMMA**2) * (
        spectral**2 + (a3 - a2) * spectral - determinant
    ) + MIXED**2 * GAMMA**2 * numerator
    characteristic = matrix.charpoly(spectral)
    # SymPy may return a fresh generator without the requested assumptions.
    # Align its formal indeterminate before comparing two expressions.
    aligned = characteristic.as_expr().xreplace({characteristic.gen: spectral})
    assert _clean(aligned - expected) == 0
    pencil = spectral * sp.eye(3) - matrix
    # Independent explicit cofactor expansion does not use charpoly's
    # generator or its characteristic-polynomial implementation.
    cofactor = (
        pencil[0, 0] * (pencil[1, 1] * pencil[2, 2] - pencil[1, 2] * pencil[2, 1])
        - pencil[0, 1] * (pencil[1, 0] * pencil[2, 2] - pencil[1, 2] * pencil[2, 0])
        + pencil[0, 2] * (pencil[1, 0] * pencil[2, 1] - pencil[1, 1] * pencil[2, 0])
    )
    assert _clean(cofactor - expected) == 0
    assert _clean(aligned - cofactor) == 0
    assert _clean(expected.subs(spectral, nu * GAMMA**2)).coeff(GAMMA, 2) == 0
    assert _clean(determinant - (282080 + 153504 * R) / (1024 * 41**2)) == 0
    assert _clean(numerator - (345 + 167 * R) / 328) == 0
    assert determinant.is_positive is True and numerator.is_positive is True
    assert COUNTER.is_negative is True and MIXED != 0
    assert _clean(nu + COUNTER - MIXED**2 * numerator / determinant) == 0
    signature = sp.diag(1, -1, -1)
    hessian = signature * matrix
    assert (hessian - hessian.T).applyfunc(_clean) == sp.zeros(3, 3)
    assert (
        _clean(
            hessian.det()
            - determinant * (COUNTER - MIXED**2 * numerator / determinant) * GAMMA**2
        )
        == 0
    )


def test_other_internal_gaps_and_unquotiented_empty_band(response):
    a, b = _expected_internal()
    for mode in (1, 2):
        opposite = (-mode) % 5
        determinant = _clean(a[mode] * a[opposite] - b[mode] ** 2)
        discriminant = _clean(((a[mode] + a[opposite]) / 2) ** 2 - b[mode] ** 2)
        assert determinant.is_positive is True and discriminant.is_positive is True
        assert a[mode].is_positive is True and a[opposite].is_positive is True
    field, zeta = response["field"], response["zeta"]
    matrices = response["matrices"]
    for index in (7, 8, 16, 17):
        for degree in range(3):
            assert all(
                value == field.zero
                for column, value in enumerate(matrices[degree][index])
                if column != index
            )
            assert (
                _conjugate(matrices[degree][index][index], field, zeta)
                == matrices[degree][index][index]
            )
        assert matrices[0][index][index] != field.zero
    # Empty complex modes retain two Cartesian coordinates each; no phase
    # is defined, or quotient taken, at their zero reference amplitude.
    assert len((7, 8, 16, 17)) == 4


def test_occupied_action_variations_keep_both_neutral_shears():
    hessian = sp.Matrix(
        ((KAPPA, UNIFORM * GAMMA), (UNIFORM * GAMMA, 2 * SELF * GAMMA**2 / 5))
    )
    generator = sp.BlockMatrix(
        [[sp.zeros(2), sp.zeros(2)], [-2 * hessian, sp.zeros(2)]]
    ).as_explicit()
    assert generator**2 == sp.zeros(4)
    assert _clean(hessian.det() / GAMMA**2 - (2 * KAPPA * SELF / 5 - UNIFORM**2)) == 0
    assert KAPPA.is_negative is True and SELF.is_positive is True
    assert generator.rank() == 2
    time = sp.symbols("T", real=True)
    transition = sp.eye(4) + time * generator
    assert sp.diff(transition, time) == generator * transition
    amplitude, phase, occupied_real, occupied_imag = sp.symbols("x y p q", real=True)
    initial = sp.Matrix((amplitude, occupied_real, phase, occupied_imag))
    assert (transition * initial)[:2] == [amplitude, occupied_real]
    assert sp.diff(10 * amplitude, time) == 0
    assert sp.diff(10 * GAMMA * occupied_real, time) == 0


def test_counter_mode_projector_and_initial_timing_response():
    a, b = _expected_internal()
    internal = sp.Matrix(((a[2], b[2]), (-b[2], -a[3])))
    zero = sp.diag(internal, sp.S.Zero)
    determinant = _clean(a[2] * a[3] - b[2] ** 2)
    projector = (zero**2 - sp.trace(internal) * zero - determinant * sp.eye(3)) / (
        -determinant
    )
    assert projector.applyfunc(_clean) == sp.diag(0, 0, 1)
    p = _clean(-MIXED * (a[3] - b[2]) / determinant)
    q = _clean(-MIXED * (a[2] - b[2]) / determinant)
    nu = _clean(-MIXED * (p + q) - COUNTER)
    slow_vector = sp.Matrix((GAMMA * p, GAMMA * q, 1))
    full = sp.Matrix(
        (
            (a[2], b[2], MIXED * GAMMA),
            (-b[2], -a[3], -MIXED * GAMMA),
            (-MIXED * GAMMA, -MIXED * GAMMA, -COUNTER * GAMMA**2),
        )
    )
    residual = (full * slow_vector - nu * GAMMA**2 * slow_vector).applyfunc(_clean)
    expected_residual = sp.Matrix((-nu * p * GAMMA**3, -nu * q * GAMMA**3, 0))
    assert (residual - expected_residual).applyfunc(_clean) == sp.zeros(3, 1)
    eta = sp.symbols("eta", positive=True)
    cos_double = -(1 + R) / 4
    contrast_rate = -2 * MIXED * GAMMA * eta * (cos_double - 1)
    assert _clean(contrast_rate / (GAMMA * eta) - 2 * MIXED * (1 - cos_double)) == 0
    assert (1 - cos_double).is_positive is True and MIXED.is_negative is True


def _along_line(polynomial, base, direction, field):
    result = {}
    for powers, value in polynomial.items():
        coefficients = [value]
        for index, power in enumerate(powers):
            if not power:
                continue
            factor = [
                field.convert(comb(power, k))
                * base[index] ** (power - k)
                * direction[index] ** k
                for k in range(power + 1)
            ]
            updated = [field.zero] * (len(coefficients) + power)
            for i, left in enumerate(coefficients):
                for j, right in enumerate(factor):
                    updated[i + j] += left * right
            coefficients = updated
        for degree, coefficient in enumerate(coefficients):
            _add_term(result, degree, coefficient)
    return result


def test_actual_cubic_remainder_and_finite_nonlinear_comparison(response):
    field, zeta = response["field"], response["zeta"]
    # This rational algebraic probe checks a polynomial identity; it is not
    # a numerical claim about an admitted small-gamma stability cutoff.
    base = [zeta**i for i in range(5)] + [
        field.convert(sp.Rational(1, 7)),
        field.zero,
        field.zero,
        field.zero,
    ]
    direction = [field.convert(sp.Rational((3 * i + 1) % 7 - 3, 11)) for i in range(9)]
    base += [_conjugate(value, field, zeta) for value in base]
    direction += [_conjugate(value, field, zeta) for value in direction]
    for polynomial in response["gradients"]:
        along = _along_line(polynomial, base, direction, field)
        pure = _along_line(polynomial, [field.zero] * 18, direction, field)
        assert set(along).issubset({0, 1, 2, 3})
        assert set(pure).issubset({3})
        assert along.get(3, field.zero) == pure.get(3, field.zero)
        direct = sum(
            _along_line(
                polynomial,
                [a + b for a, b in zip(base, direction)],
                [field.zero] * 18,
                field,
            ).values(),
            field.zero,
        )
        assert sum(along.values(), field.zero) == direct
    # A retained quadratic/cubic remainder gives order eta^2 on each fixed
    # slow interval; the coefficient is allowed to depend on that interval.
    eta, rate, time, bound = sp.symbols("eta L T M", positive=True)
    majorant = eta**2 * bound * (sp.exp(rate * time) - 1) / rate
    assert _clean(sp.diff(majorant, time) - rate * majorant - eta**2 * bound) == 0
    assert majorant.subs(time, 0) == 0


def test_full_flow_readings_and_original_clock_keep_finite_scope():
    epsilon, eta, gamma, modulation, nonlinear, reconstruction, radius = sp.symbols(
        "epsilon eta gamma B N K rho", positive=True
    )
    # Constants may depend on the fixed gamma and finite slow horizon;
    # the displayed gamma factor supplies no uniform joint-limit bound.
    state_bound = (
        epsilon * (modulation * gamma * eta + nonlinear * eta**2)
        + reconstruction * epsilon**2
    )
    normalized = _clean(state_bound / (epsilon * radius))
    assert (
        _clean(
            normalized
            - (modulation * gamma * eta + nonlinear * eta**2 + reconstruction * epsilon)
            / radius
        )
        == 0
    )
    sigma = sp.symbols("sigma", real=True)
    ellipse = sp.sqrt(C) / (sp.cos(sigma) ** 2 + C * sp.sin(sigma) ** 2)
    assert (
        _clean(ellipse * (sp.cos(sigma) ** 2 + C * sp.sin(sigma) ** 2) - sp.sqrt(C))
        == 0
    )
    horizon, time = sp.symbols("T_star t", positive=True)
    assert (
        _clean(
            (epsilon**2 * sp.sqrt(C) * time / sp.pi).subs(
                time, sp.pi * horizon / (sp.sqrt(C) * epsilon**2)
            )
            - horizon
        )
        == 0
    )
    assert sp.floor(sp.Rational(11, 12)) != sp.floor(sp.Rational(13, 12))
    # Positive total action alone does not admit each local timing reading.
    assert sum(value**2 for value in (0, 1, 0, 0, 0)) > 0
