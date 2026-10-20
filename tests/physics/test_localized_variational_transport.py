"""Static admission of finite chronological variational transport.

Only algebraic controls and primitive rational budgets are evaluated here.
The canonical profile, its retained Fourier sum, the correction, and a
nonlinear response are not evaluated. Error bounds are upper allowances,
never lower bounds on an actual correction or a retention verdict.
"""

from fractions import Fraction as Q
from itertools import product
from math import factorial

import pytest
import sympy as sp

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.mathematics._rational_interval import pi_interval
from tnfr.physics.relational_sine_class_cubic_response import _exponential_tail


@pytest.fixture(scope="module", autouse=True)
def no_source_or_response_evaluation():
    with forbid_organized_response_evaluation():
        yield


def _primitive_budgets():
    """Coarse strict bounds from integrands, not evaluated profile constants."""
    size = 1024
    height = Q(81, 2 * size)
    width = Q(9, 2 * size)
    edge = Q(49, 64)
    c0_ratio = edge + width / 4
    gap_ratio = c0_ratio + height**2 / 8
    return size, height, width, edge, c0_ratio, gap_ratio


def _fourier_tail_budgets():
    """Geometric majorants at the declared cutoff, without retained terms."""
    size = 1024
    nome, mean, spacing, cutoff = Q(1, 20), Q(1, 32), Q(44, 7 * size), 24
    tail = 4 * mean * nome ** (cutoff + 1) / (1 - nome)
    derivative_tail = (
        4
        * mean
        * spacing
        * nome ** (cutoff + 1)
        * (cutoff + 1 - cutoff * nome)
        / (1 - nome) ** 2
    )
    first = 4 * mean * spacing * nome / (1 - nome) ** 2
    second = 4 * mean * spacing**2 * nome * (1 + nome) / (1 - nome) ** 3
    return tail, derivative_tail, first, second


def test_chronological_integrals_retain_noncommuting_order():
    time, earlier, later = sp.symbols("t s u", nonnegative=True)
    first = sp.Matrix([[0, 1], [0, 0]])
    second = sp.Matrix([[0, 0], [1, 0]])
    coefficient = first + time * second
    level_one = sp.integrate(coefficient.subs(time, earlier), (earlier, 0, time))
    level_two = sp.integrate(
        sp.integrate(
            (first + later * second) * (first + earlier * second),
            (earlier, 0, later),
        ),
        (later, 0, time),
    )
    expected = (
        time**2 * first**2 / 2
        + time**3 * (first * second / 6 + second * first / 3)
        + time**4 * second**2 / 8
    )
    assert sp.simplify(level_two - expected) == sp.zeros(2)
    assert sp.simplify(sp.diff(level_two, time) - coefficient * level_one) == sp.zeros(
        2
    )
    # Exponentiating the integrated coefficient loses the ordered commutator.
    unordered_second_term = level_one**2 / 2
    assert sp.simplify(
        level_two
        - unordered_second_term
        - time**3 * (second * first - first * second) / 12
    ) == sp.zeros(2)
    assert (level_two - unordered_second_term).subs(time, 1) != sp.zeros(2)


def test_simplex_series_is_entire_and_keeps_repeated_frequency_limits():
    time, early, late = sp.symbols("t s u", nonnegative=True)
    frequencies = sp.symbols("rho0:3")
    # Expanding each exponential on the ordered simplex gives this exact
    # monomial integral; no frequency difference has been divided out.
    for degree in range(4):
        coefficient = 0
        for powers in product(range(degree + 1), repeat=3):
            if sum(powers) != degree:
                continue
            left, middle, right = powers
            integrand = (
                (time - late) ** left
                * (late - early) ** middle
                * early**right
                / (factorial(left) * factorial(middle) * factorial(right))
            )
            integral = sp.integrate(
                sp.integrate(integrand, (early, 0, late)), (late, 0, time)
            )
            assert (
                sp.simplify(integral - time ** (degree + 2) / factorial(degree + 2))
                == 0
            )
            coefficient += sp.prod(
                value**power for value, power in zip(frequencies, powers)
            )
        repeated = sp.Symbol("rho")
        collapsed = coefficient.subs(dict.fromkeys(frequencies, repeated))
        assert sp.expand(collapsed - sp.binomial(degree + 2, 2) * repeated**degree) == 0
    # The coincident-node series is t^2*exp(rho*t)/2, including rho=0.
    assert (
        sp.simplify(sp.binomial(6, 2) / sp.factorial(6) - 1 / (2 * sp.factorial(4)))
        == 0
    )
    first_frequency, second_frequency = sp.symbols("a b")
    divided_form = (
        sp.exp(second_frequency * time) - sp.exp(first_frequency * time)
    ) / (second_frequency - first_frequency)
    repeated_limit = sp.limit(divided_form, second_frequency, first_frequency)
    assert repeated_limit == time * sp.exp(first_frequency * time)
    assert repeated_limit.subs(first_frequency, 0) == time


def test_factorial_volterra_bounds_do_not_require_a_contraction():
    time, integration_time, rate = sp.symbols("t s L", nonnegative=True)
    bound = sp.Integer(1)
    forced_bound = time
    for order in range(1, 6):
        bound = sp.integrate(
            rate * bound.subs(time, integration_time), (integration_time, 0, time)
        )
        forced_bound = sp.integrate(
            rate * forced_bound.subs(time, integration_time),
            (integration_time, 0, time),
        )
        assert sp.simplify(bound - (rate * time) ** order / sp.factorial(order)) == 0
        assert (
            sp.simplify(
                forced_bound - time * (rate * time) ** order / sp.factorial(order + 1)
            )
            == 0
        )
    # LT=2 is outside the usual sup-norm contraction test. The factorial
    # bound still converges and the shared positive tail is finite.
    assert Q(2) > 1
    partial = sum((Q(2) ** order / factorial(order) for order in range(13)), Q(0))
    tail = _exponential_tail(Q(2), 13)
    assert 7 < partial < partial + tail < 8
    assert tail < Q(1, 100000)


def test_actual_jacobian_couples_modes_and_annihilates_both_mean_directions():
    angle = sp.Symbol("a", real=True)
    size = 4
    incidence = sp.eye(size)
    for row in range(size):
        incidence[row, (row + 1) % size] = -1
    # These gaps come from the real lifted phase [a,0,0,a].
    phase = sp.Matrix([angle, 0, 0, angle])
    assert incidence * phase == sp.Matrix([angle, 0, -angle, 0])
    edge_coefficient = sp.diag(1 - sp.cos(angle), 0, 1 - sp.cos(angle), 0)
    jacobian = incidence.T * edge_coefficient * incidence / 2
    basis = sp.Matrix(size, size, lambda row, mode: sp.I ** (row * mode) / 2)
    transformed = sp.simplify(basis.conjugate().T * jacobian * basis)
    symbols = [1 - sp.I**mode for mode in range(size)]
    weights = [1 - sp.cos(angle), 0, 1 - sp.cos(angle), 0]
    for output, input_mode in product(range(size), repeat=2):
        weight = (
            sum(
                weights[row] * sp.I ** (-(output - input_mode) * row)
                for row in range(size)
            )
            / size
        )
        expected = sp.conjugate(symbols[output]) * symbols[input_mode] * weight / 2
        assert sp.simplify(transformed[output, input_mode] - expected) == 0
    assert transformed[1, 3].subs(angle, sp.pi / 2) != 0
    assert transformed[:, 0] == sp.zeros(size, 1)
    assert transformed[0, :] == sp.zeros(1, size)
    zeros = sp.zeros(size)
    coupled = zeros.row_join(jacobian).col_join(zeros.row_join(zeros))
    mean = sp.ones(size, 1)
    assert coupled * mean.col_join(sp.zeros(size, 1)) == sp.zeros(2 * size, 1)
    assert coupled * sp.zeros(size, 1).col_join(mean) == sp.zeros(2 * size, 1)
    # Contractivity belongs to J_sigma, not automatically to J_sigma+B.
    principal = incidence.T * incidence / 2
    test_state = sp.Matrix([1, -1, 0, 0])
    production = -(test_state.T * principal * test_state)[0] / 8 + (
        test_state.T * jacobian * test_state
    )[0].subs(angle, sp.pi / 2)
    assert production > 0


def test_finite_correction_defect_keeps_source_jacobian_and_last_level_separate():
    time, integration_time = sp.symbols("t s", real=True)
    approximate = sp.Matrix([[0, 1], [time, 0]])
    jacobian_error = sp.Matrix([[time, 0], [0, 1]])
    source_error = sp.Matrix([time**2, -time])
    approximate_source = sp.Matrix([1, 0])
    levels = [-time * approximate_source]
    for _ in range(2):
        levels.append(
            sp.integrate(
                approximate.subs(time, integration_time)
                * levels[-1].subs(time, integration_time),
                (integration_time, 0, time),
            )
        )
    finite = sum(levels, sp.zeros(2, 1))
    exact_jacobian = approximate + jacobian_error
    exact_source = approximate_source + source_error
    actual_defect = sp.diff(finite, time) - exact_jacobian * finite + exact_source
    separated = source_error - jacobian_error * finite - approximate * levels[-1]
    assert sp.simplify(actual_defect - separated) == sp.zeros(2, 1)
    assert finite.subs(time, 0) == sp.zeros(2, 1)
    assert approximate * levels[-1] != sp.zeros(2, 1)
    # A finite chronological sum is not an exact forced solution merely
    # because its initial condition and lower levels satisfy their equations.
    assert sp.simplify(
        actual_defect - source_error + jacobian_error * finite
    ) != sp.zeros(2, 1)


def test_primitive_jacobian_budget_and_declared_dyson_order_have_small_tails():
    size, height, width, edge, _, gap_ratio = _primitive_budgets()
    carrier = 1 - sp.sqrt(2) / 2
    nonlinear = carrier**2 / 8
    assert sp.sqrt(2) < sp.Rational(17, 12)
    assert nonlinear > sp.Rational(1, 96)
    assert sp.sqrt(2 * carrier) < sp.Rational(edge.numerator, edge.denominator)
    harmonic_sum = (
        (2 - sp.sqrt(2)) / 32 + sp.Rational(1, 96) + (3 * sp.sqrt(2) - 2) / 672
    )
    assert harmonic_sum < sp.Rational(1, 16)
    # K<=pi/sqrt(2)<9/4 implies zeta<9/(2N), b<81/(2N).
    pi = pi_interval()
    assert pi.hi**2 < Q(81, 8)
    assert height < Q(1, 25)
    assert width == Q(9, 2 * size)
    # Exact m=1/2 correlation b^2*T=4/(3g) must be used here;
    # independent marginal upper bounds on b and T would lose this margin.
    beta = 128 * gap_ratio**2
    assert beta < 76
    operator_tail = _exponential_tail(Q(76), 257)
    forced_factor_tail = _exponential_tail(Q(76), 258) / 76
    assert operator_tail < Q(1, 2**80)
    assert forced_factor_tail < Q(1, 2**88)
    index = sp.Symbol("n", integer=True, nonnegative=True)
    argument = sp.Symbol("x", positive=True)
    term = argument**index / sp.factorial(index)
    assert sp.simplify(term.subs(index, index + 1) / term - argument / (index + 1)) == 0
    assert Q(76, 258) < 1
    # A small successive ratio alone is insufficient if the first omitted
    # term is still large: this prevents the earlier LT<256 shortcut.
    assert Q(256) ** 513 / factorial(513) > 1


def test_nome_tail_and_derivative_budgets_follow_from_geometric_series():
    pi = pi_interval()
    assert pi.lo > 3
    assert sum((Q(3) ** order / factorial(order) for order in range(9)), Q(0)) > 20
    # Positive exponential terms prove exp(pi)>exp(3)>20, so q_e<1/20.
    assert pi.hi * 9 / 1024 < Q(1, 32)
    assert 2 * pi.hi / 1024 < Q(44, 7 * 1024) < Q(1, 160)
    nome = sp.Symbol("q", positive=True)
    cutoff = sp.Symbol("M", integer=True, positive=True)
    geometric_tail = nome ** (cutoff + 1) / (1 - nome)
    weighted = nome * sp.diff(geometric_tail, nome)
    assert (
        sp.simplify(
            weighted
            - nome ** (cutoff + 1) * (cutoff + 1 - cutoff * nome) / (1 - nome) ** 2
        )
        == 0
    )
    full_first = nome * sp.diff(1 / (1 - nome), nome)
    full_second = nome * sp.diff(full_first, nome)
    assert sp.simplify(full_second - nome * (1 + nome) / (1 - nome) ** 3) == 0
    # The exact positive dn coefficients are bounded by 2*f0*q^|n|;
    # only this geometric majorant is evaluated, not any retained coefficient.
    tail, derivative_tail, first, second = _fourier_tail_budgets()
    assert tail < Q(1, 2**110)
    assert derivative_tail < tail / 4
    assert first < Q(1, 2**14)
    assert second < Q(1, 2**20)
    size, height, _, edge, _, _ = _primitive_budgets()
    coefficient_gap = edge * height + first + height**3 / 8
    assert coefficient_gap < Q(1, 16)
    # Means and retained terms stay separate from this discarded tail.
    assert 4 * Q(1, 32) * Q(1, 20) / (1 - Q(1, 20)) > tail
    assert size == 1024


def test_cosine_and_signed_residual_tails_have_distinct_admitted_orders():
    tail, derivative_tail, first, _ = _fourier_tail_budgets()
    _, height, _, _, _, _ = _primitive_budgets()
    amplitude = Q(1, 16)
    cosine_tail = amplitude**18 / factorial(18) / (1 - amplitude**2 / (19 * 20))
    sine_tail = amplitude**19 / factorial(19) / (1 - amplitude**2 / (20 * 21))
    assert cosine_tail < Q(1, 2**122)
    assert 2 * cosine_tail < Q(1, 2**122)
    assert sine_tail < Q(1, 2**130)
    # In the coefficient l1 algebra, D and D^T/2 have norms at most 2
    # and 1. The reconstruction phase error is <2*epsilon_F, so its
    # edge error is <4*epsilon_F. The cosine Jacobian costs another 2.
    sinh_majorant = amplitude / (1 - amplitude**2 / 6)
    assert 8 * sinh_majorant < 1
    jacobian_tail = 8 * sinh_majorant * tail + 2 * cosine_tail
    assert jacobian_tail < Q(1, 2**109)
    # Independently differentiate F^3-F_M^3. This retains the discarded
    # profile derivative and the derivative of the retained profile.
    full, retained, full_derivative, retained_derivative = sp.symbols("F P dF dP")
    assert (
        sp.expand(
            full**3
            - retained**3
            - (full - retained) * (full**2 + full * retained + retained**2)
        )
        == 0
    )
    assert (
        sp.expand(
            3 * full**2 * full_derivative
            - 3 * retained**2 * retained_derivative
            - 3 * full**2 * (full_derivative - retained_derivative)
            - 3 * (full - retained) * (full + retained) * retained_derivative
        )
        == 0
    )
    cubic_error = 3 * height**2 * tail
    cubic_derivative_error = 3 * height**2 * derivative_tail + 6 * height * first * tail
    state_error = 2 * (tail + cubic_error / 16)
    derivative_error = 2 * (derivative_tail + cubic_derivative_error / 16) + 2 * (
        tail + 3 * cubic_error / 16
    )
    assert state_error < 3 * tail
    assert derivative_error < 3 * tail
    cosh_minus_one = amplitude**2 / (1 - amplitude**2)
    nonlinear_error = cosh_minus_one * 4 * tail
    assert nonlinear_error < tail
    # ||J_sigma||<=2+2sigma<3; the source is derivative minus tangent
    # minus nonlinear force. Cancellation is retained in the finite core.
    coefficient_residual_error = derivative_error + 3 * state_error + nonlinear_error
    assert coefficient_residual_error < 32 * tail
    source_tail = 32 * (32 * tail + sine_tail)  # sqrt(N)=32.
    assert source_tail < Q(1, 2**99)
    # Sine degree17 is not differentiated to silently obtain the cosine
    # budget: the independently retained cosine degree16 omits degree18.
    variable = sp.Symbol("a")
    polynomial = sum(
        (-1) ** j * variable ** (2 * j + 1) / sp.factorial(2 * j + 1) for j in range(9)
    )
    assert sp.Poly(sp.diff(polynomial, variable), variable).degree() == 16
    assert cosine_tail != sine_tail


def test_phase_projection_and_quadratic_source_cannot_be_replaced_by_a_norm_floor():
    angle, increment = sp.symbols("a h", real=True)
    edge_remainder = (
        sp.sin(angle + increment) - sp.sin(angle) - sp.cos(angle) * increment
    )
    assert edge_remainder.subs(increment, 0) == 0
    assert sp.diff(edge_remainder, increment).subs(increment, 0) == 0
    assert sp.diff(edge_remainder, increment, 2).subs(increment, 0) == -sp.sin(angle)
    # A nonzero form-only correction has zero phase gap and zero remaining
    # nonlinear source. Its full norm cannot supply a positive lower bound.
    correction = sp.Matrix([1, -1, 0, 0])
    phase_projection = sp.Matrix([[0, 0, 1, 0], [0, 0, 0, 1]])
    assert correction.dot(correction) == 2
    assert phase_projection * correction == sp.zeros(2, 1)
    # Even a nonzero constant phase correction has zero edge increments;
    # keeping the mean coordinate is compatible with zero phase-gap work.
    incidence = sp.Matrix([[1, -1], [-1, 1]])
    assert incidence * sp.ones(2, 1) == sp.zeros(2, 1)


def test_zero_radius_metric_gain_has_a_primitive_uniform_bound():
    size, height, width, edge, c0_ratio, gap_ratio = _primitive_budgets()
    horizon = Q(8 * size**2, 27)
    c0 = c0_ratio * height
    # W(F)=b and W(F')<2^-14, with b>24/N, give this ratio.
    z0_ratio = Q(25, 32)
    assert edge + Q(1, 2**14) / Q(24, size) < z0_ratio
    z0 = z0_ratio * height
    c1_time = (
        c0 + Q(3, 4) * (edge * width * height / 4 + width**2 * height / 2) * horizon
    )
    z1_time = z0 + Q(3, 4) * (edge * Q(1, 2**14) + Q(1, 2**20)) * horizon
    correction = (c0**2 + 3 * z0**2) / 8
    derivative_time = (c0 * c1_time + 3 * z0 * z1_time) / 4
    gap, harmonic_gap = gap_ratio * height, height**3 / 8
    # Correlated Z0^2*T <= (Z0/b)^2*128 controls the near resonance.
    near_time = Q(5, 3) * z0_ratio**2 * 128 / 64
    remainder_time = gap**4 * horizon / 24 + (gap + c0) * harmonic_gap * horizon / 2
    sigma_time = Q(16, 5 * size**2) * horizon
    exponent_time = (
        near_time
        + remainder_time
        + derivative_time / 2
        + correction * (2 * sigma_time + 76)
    ) / (1 - correction)
    assert correction < Q(1, 1024)
    assert exponent_time < 5
    assert (1 + correction) / (1 - correction) < Q(21, 20) ** 2
    # Positive Taylor coefficients and the shared tail give e<3, hence
    # gamma*exp(alpha*T)<(21/20)*3^5<256 without an evaluated response.
    exp_one_upper = Q(5, 2) + _exponential_tail(Q(1), 3)
    assert exp_one_upper < 3
    assert Q(21, 20) * 3**5 < 256
    assert horizon < 2**19


def test_published_metric_bounds_and_original_scalar_residual_are_admitted():
    size, height, width, edge, c0_ratio, gap_ratio = _primitive_budgets()
    # The published W0,W1 bounds precede the stronger positive-Fourier
    # estimate. Verify that their original constants also close the gain.
    assert sp.sqrt(3) > sp.Rational(3, 2)
    w0_ratio = 1 + size * width / 12
    w1_ratio = size * width**2 / 6
    assert w0_ratio <= Q(11, 8)
    assert w1_ratio < Q(1, 256)
    assert c0_ratio < 1
    assert gap_ratio < edge + Q(1, 512)
    assert gap_ratio < 1
    assert edge * Q(11, 8) + Q(1, 256) < Q(9, 8)
    assert (1 + 3 * Q(9, 8) ** 2) / 8 < 1
    assert height**2 < Q(1, 625)
    # b/zeta<9 and d>1/4 retain the width/time correlation in C1*T.
    c1_time = Q(9, 1) / (6 * Q(1, 4)) + height / (3 * Q(1, 4)) + height
    z1_time = 2 * (size * height / (6 * Q(1, 4) * Q(3, 2)) + Q(11, 8) * height)
    assert c1_time < Q(25, 4)
    assert z1_time < 37
    derivative_time = (Q(33, 800) * Q(25, 4) + 3 * Q(9, 200) * 37) / 4
    assert derivative_time < Q(4, 3)
    carrier = sp.Symbol("lambda", positive=True)
    near_time = 5 * carrier * sp.Rational(9, 8) ** 2 * 4 / (3 * (carrier**2 / 8)) / 64
    assert sp.simplify(near_time - 135 / (128 * carrier)) == 0
    assert near_time.subs(carrier, sp.Rational(2, 7)) == sp.Rational(945, 256)
    assert 128 * height**2 / 6 < Q(1, 20)
    assert Q(945, 256) + Q(1, 20) < Q(81, 20)
    published_beta = Q(224, 3) * Q(385, 384) ** 2
    assert published_beta < Q(151, 2) < 76
    assert Q(78, 625) < Q(1, 8)
    numerator = Q(81, 20) + Q(2, 3) + Q(1, 8)
    assert numerator == Q(581, 120)
    assert numerator * Q(625, 624) < 5
    assert Q(626, 624) < Q(21, 20) ** 2
    # Independently bound every term of (42). A1<=2*A0 follows from
    # |v-sin(l*k)|<=2; no true signed defect is bounded from below.
    actual_carrier = 1 - sp.sqrt(2) / 2
    assert sp.Rational(2, 7) < actual_carrier < sp.Rational(1, 3)
    assert actual_carrier**2 / 8 < sp.Rational(1, 72)
    assert sp.sqrt(2) / 4 < sp.Rational(1, 2)
    b1, b2, b3 = width * height / 4, width**2 * height / 2, Q(9, 8) * width**3 * height
    c0, gap = c0_ratio * height, gap_ratio * height
    slow_derivative = b2 / 2 + height**3 / 72
    scalar_residual = (
        b3 / 6
        + 3 * Q(1, 8) * height**2 * b1
        + b1 * c0**2 / 2
        + (
            3 * height**2 * slow_derivative
            + 3 * height * b1**2
            + Q(3, 2) * height**2 * b2
        )
        / 16
        + height**3 * gap**2 / 16
        + gap**5 / 120
        + 2 * Q(16, 5 * size**2) * (height + height**3 / 16)
    )
    assert scalar_residual < 1
    assert 32 * scalar_residual < 32


def test_input_error_and_chronological_tail_are_separate_uniform_allowances():
    tail, derivative_tail, first, _ = _fourier_tail_budgets()
    size, height, _, _, _, gap_ratio = _primitive_budgets()
    horizon = Q(8 * size**2, 27)
    amplitude = Q(1, 16)
    cosine_tail = amplitude**18 / factorial(18) / (1 - amplitude**2 / (19 * 20))
    sine_tail = amplitude**19 / factorial(19) / (1 - amplitude**2 / (20 * 21))
    jacobian_tail = 8 * amplitude / (1 - amplitude**2 / 6) * tail + 2 * cosine_tail
    source_tail = 32 * (32 * tail + sine_tail)
    # A coarse residual norm bound uses the two real rows, their temporal
    # derivatives, actual tangent and nonlinear coefficient norm separately.
    state_norm = 64 * (height + height**3 / 16)
    rate_norm = 64 * (first + 3 * height**2 * first / 16 + height + 3 * height**3 / 16)
    nonlinear_norm = 32 * amplitude**3 / 6 / (1 - amplitude**2 / 20)
    residual_bound = rate_norm + 3 * state_norm + nonlinear_norm
    assert residual_bound < 32
    approximate_residual_bound = residual_bound + source_tail
    assert approximate_residual_bound < 33 < 64
    # The approximating Jacobian also satisfies the declared beta limit.
    assert 128 * gap_ratio**2 + jacobian_tail * horizon < 76
    gain = Q(256)
    assert gain * jacobian_tail * horizon < Q(1, 2)
    # E<=M*T*eps_r+M*T*eps_B*(M*R*T+E) absorbs only the small
    # input perturbation. It assumes no contraction of the full V map.
    error, source, jacobian, metric_gain, end_time, residual = sp.symbols(
        "E eps_r eps_B M T R", positive=True
    )
    comparison = metric_gain * end_time * source + metric_gain * end_time * jacobian * (
        metric_gain * residual * end_time + error
    )
    solved = sp.solve(sp.Eq(error, comparison), error)[0]
    assert (
        sp.simplify(
            solved
            - (
                metric_gain * end_time * source
                + metric_gain**2 * residual * end_time**2 * jacobian
            )
            / (1 - metric_gain * end_time * jacobian)
        )
        == 0
    )
    source_factor = 2 * gain * horizon
    jacobian_factor = 2 * gain**2 * residual_bound * horizon**2
    assert source_factor < 2**28
    assert jacobian_factor < 2**60
    representation_error = source_factor * source_tail + jacobian_factor * jacobian_tail
    assert approximate_residual_bound * horizon < 2**25
    forced_tail = (
        approximate_residual_bound * horizon * _exponential_tail(Q(76), 258) / 76
    )
    assert forced_tail < Q(1, 2**63)
    assert representation_error + forced_tail < Q(1, 2**48)
    assert representation_error + forced_tail < Q(1, 2**40)
    assert derivative_tail > 0
    # These are defects of a finite comparison representation. None is an
    # assertion that the retained correction or nonlinear state is small.
