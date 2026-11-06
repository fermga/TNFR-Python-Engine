"""Static controls for full-state relative storage and carrier resonance.

These checks evaluate identities and counterexamples, never source preparation,
an evolution trajectory, or a claim of retained or acquired organization.
"""

from fractions import Fraction as Q

import mpmath as mp
import pytest
import sympy as sp

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.mathematics._rational_interval import I, pi_interval, sin


@pytest.fixture(scope="module", autouse=True)
def no_source_or_response_evaluation():
    with forbid_organized_response_evaluation():
        yield


def _cycle_incidence(size):
    incidence = sp.zeros(size)
    for j in range(size):
        incidence[j, j] = 1
        incidence[j, (j + 1) % size] = -1
    return incidence


def test_bregman_derivative_keeps_both_residual_rows_and_actual_loss():
    # Arbitrary node values have no carrier-band restriction. Differentiate
    # the storage itself before substituting either complete evolution row.
    size = 3
    incidence = _cycle_incidence(size)
    laplacian = incidence.T * incidence
    sigma = sp.Symbol("sigma", positive=True)
    reference_form = sp.Matrix(sp.symbols("x:3", real=True))
    reference_phase = sp.Matrix(sp.symbols("theta:3", real=True))
    form_error = sp.Matrix(sp.symbols("u:3", real=True))
    phase_error = sp.Matrix(sp.symbols("eta:3", real=True))
    form_residual = sp.Matrix(sp.symbols("rx:3", real=True))
    phase_residual = sp.Matrix(sp.symbols("rt:3", real=True))
    reference_gap = incidence * reference_phase
    error_gap = incidence * phase_error
    reference_gradient = incidence.T * reference_gap.applyfunc(sp.sin)
    actual_gradient = incidence.T * (reference_gap + error_gap).applyfunc(sp.sin)
    gradient_error = actual_gradient - reference_gradient
    reference_hessian = (
        incidence.T * sp.diag(*reference_gap.applyfunc(sp.cos)) * incidence
    )
    relative_storage = (form_error.T * laplacian * form_error)[0] / 2 + sum(
        -sp.cos(a + b) + sp.cos(a) - sp.sin(a) * b
        for a, b in zip(reference_gap, error_gap)
    )
    reference_phase_rate = laplacian * reference_form / 2 + phase_residual
    form_error_rate = (
        -sigma * laplacian * form_error / 2 - gradient_error / 2 - form_residual
    )
    phase_error_rate = laplacian * form_error / 2 - phase_residual
    differentiated = sum(
        sp.diff(relative_storage, variable) * rate
        for variables, rates in (
            (form_error, form_error_rate),
            (phase_error, phase_error_rate),
            (reference_phase, reference_phase_rate),
        )
        for variable, rate in zip(variables, rates)
    )
    loss = -sigma * (laplacian * form_error).dot(laplacian * form_error) / 2
    moving_curvature = (gradient_error - reference_hessian * phase_error).dot(
        reference_phase_rate
    )
    residual_work = -form_error.dot(laplacian * form_residual) - gradient_error.dot(
        phase_residual
    )
    assert sp.expand(differentiated - loss - moving_curvature - residual_work) == 0
    # Neither row's residual nor the moving-curvature term vanishes generally.
    assert sp.expand(form_error.dot(laplacian * form_residual)) != 0
    assert sp.expand(gradient_error.dot(phase_residual)) != 0
    assert sp.expand(moving_curvature) != 0
    assert sp.expand(loss) != 0


def test_means_are_independent_error_coordinates_invisible_to_relative_storage():
    incidence = _cycle_incidence(4)
    laplacian = incidence.T * incidence
    ones = sp.ones(4, 1)
    form = sp.Matrix(sp.symbols("x:4", real=True))
    phase = sp.Matrix(sp.symbols("theta:4", real=True))
    residual_form = sp.Matrix(sp.symbols("rx:4", real=True))
    residual_phase = sp.Matrix(sp.symbols("rt:4", real=True))
    sigma = sp.Symbol("sigma", positive=True)
    current = incidence.T * (incidence * phase).applyfunc(sp.sin)
    reference_form_rate = -sigma * laplacian * form / 2 - current / 2 + residual_form
    reference_phase_rate = laplacian * form / 2 + residual_phase
    assert sp.expand(sum(reference_form_rate) - sum(residual_form)) == 0
    assert sp.expand(sum(reference_phase_rate) - sum(residual_phase)) == 0
    # The actual unforced means are constant, hence error means have the
    # negative residual means as rates; mean-free residuals are a premise.
    mean_form, mean_phase = sp.symbols("mean_x mean_theta", real=True)
    mean_error_storage = 4 * (mean_form**2 + mean_phase**2) / 2
    mean_error_rate = (
        -sp.diff(mean_error_storage, mean_form) * sum(residual_form) / 4
        - sp.diff(mean_error_storage, mean_phase) * sum(residual_phase) / 4
    )
    assert (
        sp.expand(
            mean_error_rate
            + mean_form * sum(residual_form)
            + mean_phase * sum(residual_phase)
        )
        == 0
    )
    assert incidence * (mean_form * ones) == sp.zeros(4, 1)
    assert incidence * (mean_phase * ones) == sp.zeros(4, 1)
    assert sp.expand((mean_form * ones).dot(mean_form * ones)) == 4 * mean_form**2
    assert sp.expand(mean_error_storage) != 0


def test_curvature_coercivity_needs_the_intervening_chart_and_the_cycle_gap():
    reference, error, fraction = sp.symbols("a b s", real=True)
    potential_along_segment = 1 - sp.cos(reference + fraction * error)
    assert sp.diff(potential_along_segment, fraction, 2) == error**2 * sp.cos(
        reference + fraction * error
    )
    remainder = (
        -sp.cos(reference + error) + sp.cos(reference) - sp.sin(reference) * error
    )
    # A regular point on the boundary of the acute chart does not provide
    # a positive Hessian floor or a positive Bregman remainder on both sides.
    assert remainder.subs({reference: sp.pi / 2, error: sp.pi / 6}) == (
        sp.Rational(1, 2) - sp.pi / 6
    )
    assert sp.Rational(1, 2) - sp.pi / 6 < 0
    assert sp.diff(remainder, error, 2).subs({reference: sp.pi / 2, error: 0}) == 0
    # Low-frequency errors are admitted. Edge storage cannot be converted
    # to a node norm using the carrier's much larger eigenvalue.
    with mp.workdps(60):
        size = 1024
        lowest_mode = [mp.cos(2 * mp.pi * j / size) for j in range(size)]
        norm_squared = sum(value**2 for value in lowest_mode)
        edge_squared = sum(
            (lowest_mode[j] - lowest_mode[(j + 1) % size]) ** 2 for j in range(size)
        )
        gap = 2 - 2 * mp.cos(2 * mp.pi / size)
        carrier_eigenvalue = 2 - 2 * mp.cos(mp.pi / 4)
        assert abs(sum(lowest_mode)) < mp.mpf("1e-55")
        assert abs(edge_squared - gap * norm_squared) < mp.mpf("1e-55")
        assert edge_squared < carrier_eigenvalue * norm_squared / 1000


def _fast_carrier_curvature(relative_quadrature):
    character = sp.Symbol("v", nonzero=True)
    height, error_height = sp.symbols("b c", positive=True)
    root = (1 + sp.I) / sp.sqrt(2)
    frequency = 1 - sp.sqrt(2) / 2
    reference_phase = height * (character - 1 / character) / (2 * sp.I)
    error_phase = (
        error_height
        * (relative_quadrature * character - 1 / (relative_quadrature * character))
        / (2 * sp.I)
    )
    reference_gap = sp.expand(
        reference_phase - reference_phase.subs(character, root * character)
    )
    error_gap = sp.expand(error_phase - error_phase.subs(character, root * character))
    gap_rate = sp.I * frequency * character * sp.diff(reference_gap, character)
    # The coefficient of squared reference height in the exact moving
    # curvature is -a*a'*b_error^2/2; higher sine terms remain in the theorem.
    quadratic = sp.expand(-reference_gap * gap_rate * error_gap**2 / 2)
    return character, height, error_height, frequency, quadratic


@pytest.mark.parametrize("sign", (-1, 1))
def test_same_carrier_error_leaves_a_sign_changing_fast_resonance(sign):
    quadrature = (1 + sign * sp.I) / sp.sqrt(2)
    character, height, error_height, frequency, quadratic = _fast_carrier_curvature(
        quadrature
    )
    # Every nonconstant mode here is below the eight-site carrier alias.
    # Therefore the C1024 spatial sum equals 1024 times the constant term.
    coefficients = {
        mode: sp.simplify(quadratic.coeff(character, mode)) for mode in range(-4, 5)
    }
    root = (1 + sp.I) / sp.sqrt(2)
    for mode, coefficient in coefficients.items():
        if coefficient != 0 and mode != 0:
            assert sp.simplify(sum(root ** (mode * j) for j in range(8))) == 0
    expected_mean = -sign * frequency**3 * height**2 * error_height**2 / 4
    assert sp.simplify(coefficients[0] - expected_mean) == 0
    assert coefficients[0] != 0
    assert (
        sp.simplify(sp.I * frequency * character * sp.diff(coefficients[0], character))
        == 0
    )
    baseline_storage = 1024 * frequency * error_height**2
    nonlinear_coefficient = frequency**2 / 8
    assert (
        sp.simplify(
            1024 * coefficients[0] / baseline_storage
            + sign * 2 * nonlinear_coefficient * height**2
        )
        == 0
    )


def test_periodic_fast_correction_cannot_cancel_a_resonant_constant_term():
    root = (1 - sp.I) / sp.sqrt(2)
    character, _, _, frequency, quadratic = _fast_carrier_curvature(root)
    coefficients = sp.symbols("q:9")
    correction = sum(
        coefficient * character**mode
        for mode, coefficient in zip(range(-4, 5), coefficients)
    )
    correction_derivative = sp.expand(
        sp.I * frequency * character * sp.diff(correction, character)
    )
    assert correction_derivative.coeff(character, 0) == 0
    assert sp.simplify(quadratic.coeff(character, 0)) > 0
    # This is the same-carrier frequency denominator, not the third-harmonic
    # denominator used to construct the reference's nonresonant corrector.
    carrier_frequency = 1 - sp.cos(sp.pi / 4)
    assert 2 * carrier_frequency - carrier_frequency - carrier_frequency == 0


def test_actual_loss_survives_carrier_averaging_and_is_not_phase_dissipation():
    character = sp.Symbol("v", nonzero=True)
    height, sigma = sp.symbols("c sigma", positive=True)
    root = (1 + sp.I) / sp.sqrt(2)
    form = height * (character + 1 / character) / 2
    phase = height * (character - 1 / character) / (2 * sp.I)

    def laplacian(value):
        return sp.expand(
            2 * value
            - value.subs(character, character * root)
            - value.subs(character, character / root)
        )

    frequency = 1 - sp.sqrt(2) / 2
    assert sp.simplify(laplacian(form) - 2 * frequency * form) == 0
    assert sp.simplify(laplacian(phase) - 2 * frequency * phase) == 0
    loss = sp.expand(-sigma * laplacian(form) ** 2 / 2)
    assert sp.simplify(loss.coeff(character, 0) + sigma * frequency**2 * height**2) == 0
    baseline_storage = frequency * height**2
    assert (
        sp.simplify(loss.coeff(character, 0) / baseline_storage + sigma * frequency)
        == 0
    )
    # A full-state perturbation with zero instantaneous form error has no
    # instantaneous dissipative loss, even with nonzero phase error.
    phase_only_form = sp.Integer(0)
    assert -sigma * laplacian(phase_only_form) ** 2 / 2 == 0
    assert laplacian(phase) != 0


def test_distinct_neighboring_carriers_produce_the_small_divisor_correction():
    # Unit spatial characters on C1024: modes 129 and 127 sum to twice
    # the reference's mode 128. Independent conjugate symbols retain both
    # real rows without introducing a Fourier-band assumption in the model.
    spatial, fast = sp.symbols("w f", nonzero=True)
    gap_height, frequency = sp.symbols("A lambda", positive=True)
    edge_q, edge_r, conjugate_q, conjugate_r = sp.symbols("dq dr dbq dbr")
    mode_q, mode_r, conjugate_mode_q, conjugate_mode_r = sp.symbols("zq zr zbq zbr")
    reference_gap = (
        gap_height * (fast * spatial**128 - 1 / (fast * spatial**128)) / (2 * sp.I)
    )
    reference_rate = sp.I * frequency * fast * sp.diff(reference_gap, fast)
    phase_error_gap = (
        edge_q * mode_q * spatial**129
        + edge_r * mode_r * spatial**127
        - conjugate_q * conjugate_mode_q / spatial**129
        - conjugate_r * conjugate_mode_r / spatial**127
    ) / (2 * sp.I)
    moving_term = sp.expand(-reference_gap * reference_rate * phase_error_gap**2 / 2)
    product = edge_q * edge_r * mode_q * mode_r / fast**2
    conjugate_product = (
        conjugate_q * conjugate_r * conjugate_mode_q * conjugate_mode_r * fast**2
    )
    expected_mean = (
        -(gap_height**2) * frequency * (product - conjugate_product) / (16 * sp.I)
    )
    assert sp.simplify(moving_term.coeff(spatial, 0) - expected_mean) == 0
    # No nonzero spatial exponent aliases to zero modulo the actual size.
    for term in sp.Add.make_args(moving_term):
        mode = term.as_powers_dict().get(spatial, 0)
        assert mode == 0 or mode % 1024 != 0
    detuning = sp.Symbol("Delta", nonzero=True, real=True)
    product_symbol, conjugate_symbol = sp.symbols("P Pbar")
    correction = (
        -(gap_height**2)
        * frequency
        * (product_symbol + conjugate_symbol)
        / (16 * detuning)
    )
    correction_rate = (
        sp.diff(correction, product_symbol) * sp.I * detuning * product_symbol
        - sp.diff(correction, conjugate_symbol) * sp.I * detuning * conjugate_symbol
    )
    forcing = (
        -(gap_height**2) * frequency * (product_symbol - conjugate_symbol) / (16 * sp.I)
    )
    assert sp.expand(correction_rate + forcing) == 0
    carrier, sideband = sp.symbols("k h", real=True)

    def dispersion(angle):
        return 1 - sp.cos(angle)

    exact_detuning = (
        dispersion(carrier + sideband)
        + dispersion(carrier - sideband)
        - 2 * dispersion(carrier)
    )
    assert (
        sp.simplify(
            sp.expand_trig(exact_detuning)
            - 2 * sp.cos(carrier) * (1 - sp.cos(sideband))
        )
        == 0
    )


def test_pair_correction_ratio_retains_full_storage_normalization():
    eigenvalue_q, eigenvalue_r, amplitude_q, amplitude_r = sp.symbols(
        "lambda_q lambda_r a_q a_r", positive=True
    )
    storage = eigenvalue_q * amplitude_q**2 + eigenvalue_r * amplitude_r**2
    largest_product = (
        2 * sp.sqrt(eigenvalue_q * eigenvalue_r) * amplitude_q * amplitude_r
    )
    assert (
        sp.expand(
            storage**2
            - largest_product**2
            - (eigenvalue_q * amplitude_q**2 - eigenvalue_r * amplitude_r**2) ** 2
        )
        == 0
    )
    balanced = {
        amplitude_q: 1 / sp.sqrt(eigenvalue_q),
        amplitude_r: 1 / sp.sqrt(eigenvalue_r),
    }
    assert storage.subs(balanced) == 2
    assert largest_product.subs(balanced) == 2
    height, eigenvalue, detuning = sp.symbols("b lambda Delta", positive=True)
    # |d_q d_r|=2 sqrt(lambda_q lambda_r); using a unitary transform
    # gives the same ratio, with no extra N or factor of two.
    squared_reference_gap = 2 * eigenvalue * height**2
    ratio = squared_reference_gap * eigenvalue / (8 * detuning)
    nonlinear = eigenvalue**2 / 8
    assert sp.simplify(ratio - 2 * nonlinear * height**2 / detuning) == 0


def test_fixed_cycle_period_relation_defeats_uniformly_small_sideband_correction():
    size = sp.Symbol("N", positive=True)
    elliptic_period, dispersion, nonlinear = sp.symbols("K d g", positive=True)
    physical_height_squared = (
        8 * elliptic_period**2 * dispersion / (nonlinear * size**2)
    )
    sideband_detuning = 8 * dispersion * sp.sin(sp.pi / size) ** 2
    ratio = 2 * nonlinear * physical_height_squared / sideband_detuning
    assert (
        sp.simplify(
            ratio - 2 * elliptic_period**2 / (size**2 * sp.sin(sp.pi / size) ** 2)
        )
        == 0
    )
    # K=pi is a possible elliptic period value in (pi/2,infinity).
    # This is a static obstruction fixture, not an evaluated profile source.
    pi = pi_interval()
    correction_ratio = 2 * pi**2 / (1024**2 * sin(pi / 1024) ** 2)
    assert correction_ratio.lo > 2
    frequency = 1 - sp.sqrt(2) / 2
    actual_dispersion = sp.sqrt(2) / 4
    actual_nonlinear = frequency**2 / 8
    assert sp.simplify(81 - 2 * actual_dispersion / actual_nonlinear) > 0
    assert pi.hi < Q(7, 2)
    height_upper = Q(2) * Q(7, 2) * 9 / 1024
    assert height_upper < Q(1, 16)
    assert 2 * frequency < 1  # D_k<1, hence leading gaps are also <1/16.
    # On balanced amplitudes and the unfavorable product phase, H0=2
    # while its canonical nonresonant correction is -2*ratio.
    corrected_storage = I(2) * (1 - correction_ratio)
    assert corrected_storage.hi < -2


def test_fixed_congruence_split_retains_near_modes_and_moving_coefficient_work():
    time = sp.Symbol("t", real=True)
    skew = sp.diag(sp.Matrix([[0, -1], [1, 0]]), sp.Matrix([[0, -3], [3, 0]]))
    one_basis = sp.Matrix([[1, 1], [-sp.I, sp.I]]) / sp.sqrt(2)
    basis = sp.diag(one_basis, one_basis)
    frequencies = (1, -1, 3, -3)
    omega, cutoff = 2, 2
    assert basis.conjugate().T * basis == sp.eye(4)
    assert skew * basis == basis * sp.diag(*(sp.I * value for value in frequencies))
    constant = sp.Matrix(4, 4, lambda j, k: j + k + 1 + sp.I * (j * k + 1))
    slope = sp.Matrix(4, 4, lambda j, k: (j + 1) * (k + 1) + sp.I * (j + k + 2))
    matrix = constant + time * slope
    congruence = sp.simplify(basis.T * matrix * basis)
    assert congruence.T == congruence
    assert sp.simplify(basis.conjugate().T * matrix * basis - congruence) != sp.zeros(4)
    near, far, correction_coefficients = sp.zeros(4), sp.zeros(4), sp.zeros(4)
    retained_divisors, removed_divisors = set(), set()
    for j in range(4):
        for k in range(4):
            divisor = omega + frequencies[j] + frequencies[k]
            if abs(divisor) <= cutoff:
                near[j, k] = congruence[j, k]
                retained_divisors.add(divisor)
            else:
                far[j, k] = congruence[j, k]
                correction_coefficients[j, k] = -2 * congruence[j, k] / (sp.I * divisor)
                removed_divisors.add(divisor)
    assert {0, -cutoff, cutoff}.issubset(retained_divisors)
    assert all(abs(value) > cutoff for value in removed_divisors)
    assert near + far == congruence
    assert (
        near.T == near
        and far.T == far
        and correction_coefficients.T == correction_coefficients
    )
    assert all(any(near[j, k] != 0 for k in range(4)) for j in range(4))
    assert all(any(far[j, k] != 0 for k in range(4)) for j in range(4))

    def inverse_congruence(value):
        return sp.simplify(basis.conjugate() * value * basis.conjugate().T)

    def real_part(value):
        return value.applyfunc(lambda entry: sp.expand_complex(entry).as_real_imag()[0])

    def squared_frobenius(value):
        return sp.simplify(sum(entry * sp.conjugate(entry) for entry in value))

    assert (inverse_congruence(near + far) - matrix).applyfunc(sp.simplify) == sp.zeros(
        4
    )
    assert squared_frobenius(congruence) == squared_frobenius(matrix)
    correction = inverse_congruence(correction_coefficients)
    far_matrix = inverse_congruence(far)
    phase = sp.cos(omega * time) + sp.I * sp.sin(omega * time)
    metric = real_part(phase * correction)
    moving_coefficient = real_part(phase * correction.diff(time))
    identity_error = (
        metric.diff(time)
        + skew.T * metric
        + metric * skew
        + 2 * real_part(phase * far_matrix)
        - moving_coefficient
    )
    assert identity_error.applyfunc(sp.simplify) == sp.zeros(4)
    assert metric.T == metric and moving_coefficient.T == moving_coefficient
    # Freezing the envelope coefficient would silently discard this term.
    assert moving_coefficient.subs(time, 0) != sp.zeros(4)
    assert (
        sp.simplify(
            squared_frobenius(correction) - squared_frobenius(correction_coefficients)
        )
        == 0
    )
    # Entrywise division gives the claimed Frobenius estimate on this
    # independent matrix, including its derivative under the same fixed mask.
    for evaluated_time in (0, sp.Rational(2, 3)):
        assert (
            4 * squared_frobenius(far) / cutoff**2
            - squared_frobenius(correction_coefficients)
        ).subs(time, evaluated_time) > 0
    assert (
        4 * squared_frobenius(far.diff(time)) / cutoff**2
        - squared_frobenius(correction_coefficients.diff(time))
        > 0
    )


def test_corrected_energy_derivative_requires_the_half_factor_and_all_forcing_rows():
    state = sp.Matrix(sp.symbols("y:2", real=True))
    forcing = sp.Matrix(sp.symbols("f:2", real=True))
    residual = sp.Matrix(sp.symbols("r:2", real=True))
    frequency = sp.Symbol("lambda", positive=True)
    skew = sp.Matrix([[0, -frequency], [frequency, 0]])
    a, b, c, d, e, f, h, j, k = sp.symbols("a b c d e f h j k", real=True)
    metric = sp.Matrix([[a, b], [b, c]])
    removed = sp.Matrix([[d, e], [e, f]])
    retained = sp.Matrix([[h, j], [j, k]])
    slope = sp.Matrix(
        [[sp.Symbol("s0"), sp.Symbol("s1")], [sp.Symbol("s1"), sp.Symbol("s2")]]
    )
    metric_rate = -skew.T * metric - metric * skew - 2 * removed + slope
    state_rate = skew * state + forcing - residual
    base_rate = state.dot((removed + retained) * state)
    corrected_rate = (
        base_rate + state.dot(metric_rate * state) / 2 + state.dot(metric * state_rate)
    )
    expected_rate = (
        state.dot(retained * state)
        + state.dot(slope * state) / 2
        + state.dot(metric * (forcing - residual))
    )
    assert sp.expand(corrected_rate - expected_rate) == 0
    # Neither the metric/reference derivative nor either sign of the
    # forcing/residual interaction is removed by skew cancellation.
    assert sp.expand(state.dot(slope * state)) != 0
    assert sp.expand(state.dot(metric * residual)) != 0
    wrong_normalization = (
        base_rate + state.dot(metric_rate * state) + 2 * state.dot(metric * state_rate)
    )
    assert sp.expand(wrong_normalization - expected_rate) != 0


def test_positive_corrected_metric_has_a_strict_coercivity_requirement():
    state = sp.Matrix(sp.symbols("y:2", real=True))
    kappa, correction_bound = sp.Rational(3, 4), sp.Rational(1, 4)
    base_hessian = sp.diag(kappa, 1)
    correction = sp.diag(-correction_bound, correction_bound)
    storage = state.dot((base_hessian + correction) * state) / 2
    lower_difference = sp.expand(
        storage - (kappa - correction_bound) * state.dot(state) / 2
    )
    upper_difference = sp.expand(
        (1 + correction_bound) * state.dot(state) / 2 - storage
    )
    assert lower_difference == sp.Rational(3, 8) * state[1] ** 2
    assert upper_difference == sp.Rational(3, 8) * state[0] ** 2
    assert base_hessian + sp.diag(-kappa, kappa) == sp.diag(0, 1 + kappa)
    too_large_correction = sp.diag(
        -kappa - sp.Rational(1, 10), kappa + sp.Rational(1, 10)
    )
    direction = sp.Matrix([1, 0])
    assert direction.dot(
        (base_hessian + too_large_correction) * direction
    ) == -sp.Rational(1, 10)


def test_sqrt_twice_energy_comparison_keeps_residual_and_exponent_factors():
    q, coercivity, growth, residual, correction = sp.symbols(
        "q delta c r k", positive=True
    )
    largest_norm = q / sp.sqrt(coercivity)
    energy_rate = growth * largest_norm**2 + (1 + correction) * largest_norm * residual
    # E_c=q^2/2 gives E_c'=q*q', with no additional factor of two.
    q_rate = sp.simplify(energy_rate / q)
    alpha, beta = growth / coercivity, (1 + correction) / sp.sqrt(coercivity)
    assert sp.simplify(q_rate - alpha * q - beta * residual) == 0
    time, initial = sp.symbols("t q0", positive=True)
    majorant = (
        sp.exp(alpha * time) * initial
        + beta * residual * (sp.exp(alpha * time) - 1) / alpha
    )
    assert (
        sp.simplify(sp.diff(majorant, time) - alpha * majorant - beta * residual) == 0
    )
    assert majorant.subs(time, 0) == initial
    # The initial estimate follows from the upper metric bound, while the
    # exit radius uses the lower bound. Interchanging them is not sound.
    initial_norm, radius = sp.symbols("y0 rho", positive=True)
    initial_bound = sp.sqrt(1 + correction) * initial_norm
    exit_bound = radius * sp.sqrt(coercivity)
    assert initial_bound**2 / 2 == (1 + correction) * initial_norm**2 / 2
    assert sp.simplify(exit_bound**2 / coercivity) == radius**2
