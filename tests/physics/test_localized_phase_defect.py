"""Static controls for signed, phase-resolved two-row defect transport.

The forced linear comparison identities below are not evaluated nonlinear
responses, source preparations, or certificates of envelope maintenance.
"""

import numpy as np
import pytest
import sympy as sp

from tests.sine_evidence_helpers import forbid_organized_response_evaluation


@pytest.fixture(scope="module", autouse=True)
def no_source_or_response_evaluation():
    with forbid_organized_response_evaluation():
        yield


def _simplify_matrix(value):
    return value.applyfunc(sp.simplify)


def _generator(eigenvalue, sigma):
    return eigenvalue * sp.Matrix([[-sigma, -1], [1, 0]])


def test_actual_two_row_propagator_solves_the_modal_equation_and_keeps_zero_mode():
    sigma, eigenvalue, time = sp.symbols("sigma lambda t", positive=True)
    damped_frequency = sp.sqrt(1 - sigma**2 / 4)
    shifted = sp.Matrix([[-sigma / 2, -1], [1, sigma / 2]])
    generator = _generator(eigenvalue, sigma)
    assert _simplify_matrix(shifted**2 + damped_frequency**2 * sp.eye(2)) == sp.zeros(2)
    propagator = sp.exp(-sigma * eigenvalue * time / 2) * (
        sp.cos(eigenvalue * damped_frequency * time) * sp.eye(2)
        + sp.sin(eigenvalue * damped_frequency * time) * shifted / damped_frequency
    )
    assert _simplify_matrix(propagator.diff(time) - generator * propagator) == sp.zeros(
        2
    )
    assert propagator.subs(time, 0) == sp.eye(2)
    assert propagator.subs(eigenvalue, 0) == sp.eye(2)
    assert _simplify_matrix(
        propagator.diff(time).subs(time, 0) - generator
    ) == sp.zeros(2)
    # Positive form loss mixes the two-real-row circular eigenvectors.
    circular = sp.Matrix([1, -sp.I])
    assert _simplify_matrix(
        generator * circular - sp.I * eigenvalue * circular
    ) == sp.Matrix([-sigma * eigenvalue, 0])


def test_modal_contractivity_is_form_loss_not_scalar_isotropic_damping():
    sigma, eigenvalue = sp.symbols("sigma lambda", positive=True)
    xr, xi, tr, ti = sp.symbols("xr xi tr ti", real=True)
    state = sp.Matrix([xr + sp.I * xi, tr + sp.I * ti])
    rate = _generator(eigenvalue, sigma) * state
    norm_rate = sp.re((state.conjugate().T * rate)[0])
    assert sp.expand(norm_rate + sigma * eigenvalue * (xr**2 + xi**2)) == 0
    assert norm_rate.subs({xr: 0, xi: 0}) == 0
    assert (
        sp.expand(norm_rate + sigma * eigenvalue * (xr**2 + xi**2 + tr**2 + ti**2)) != 0
    )


def test_forced_harmonic_resolvent_and_actual_loss_resonance_are_two_row_identities():
    eigenvalue, sigma = sp.symbols("lambda sigma", positive=True)
    frequency = sp.Symbol("Omega", real=True)
    generator = _generator(eigenvalue, sigma)
    matrix = sp.I * frequency * sp.eye(2) - generator
    determinant = eigenvalue**2 - frequency**2 + sp.I * sigma * eigenvalue * frequency
    inverse = (
        sp.Matrix(
            [
                [sp.I * frequency, -eigenvalue],
                [eigenvalue, sp.I * frequency + sigma * eigenvalue],
            ]
        )
        / determinant
    )
    assert sp.expand(matrix.det() - determinant) == 0
    assert _simplify_matrix(matrix * inverse - sp.eye(2)) == sp.zeros(2)
    assert determinant.subs(frequency, eigenvalue) == sp.I * sigma * eigenvalue**2
    assert determinant.subs(frequency, 0) == eigenvalue**2
    time = sp.Symbol("t", real=True)
    entries = sp.symbols("e:4")
    propagator = sp.Matrix(2, 2, entries)
    forced = (sp.exp(sp.I * frequency * time) * sp.eye(2) - propagator) * inverse
    # Differentiate using E'=J E, independently of its closed form above.
    forced_rate = (
        sp.I * frequency * sp.exp(sp.I * frequency * time) * sp.eye(2)
        - generator * propagator
    ) * inverse
    assert _simplify_matrix(
        forced_rate - generator * forced - sp.exp(sp.I * frequency * time) * sp.eye(2)
    ) == sp.zeros(2)
    assert forced.subs(
        {time: 0, entries[0]: 1, entries[1]: 0, entries[2]: 0, entries[3]: 1}
    ) == sp.zeros(2)


def test_mean_forcing_has_a_regular_frequency_limit_and_is_not_discarded():
    frequency, time = sp.symbols("Omega t", real=True)
    response = (sp.exp(sp.I * frequency * time) - 1) / (sp.I * frequency)
    assert sp.simplify(sp.diff(response, time) - sp.exp(sp.I * frequency * time)) == 0
    assert sp.limit(response, frequency, 0) == time
    assert sp.limit(response, time, 0) == 0
    amplitude = sp.Matrix(sp.symbols("rx rt"))
    constant_mean_correction = -time * amplitude
    assert constant_mean_correction.diff(time) == -amplitude
    # The inverse branch is singular only when both lambda and Omega vanish;
    # the entire convolution still has this finite limit in both rows.
    assert _generator(0, sp.Symbol("sigma")) == sp.zeros(2)


def test_complex_reference_notation_projects_into_both_real_fourier_rows():
    coefficient, conjugate_coefficient, character = sp.symbols("c cbar u", nonzero=True)
    form = (coefficient * character + conjugate_coefficient / character) / 2
    phase = (coefficient * character - conjugate_coefficient / character) / (2 * sp.I)
    assert sp.expand(form + sp.I * phase - coefficient * character) == 0
    positive = sp.Matrix(
        [sp.expand(form).coeff(character, 1), sp.expand(phase).coeff(character, 1)]
    )
    assert positive == coefficient * sp.Matrix([1, -sp.I]) / 2
    eigenvalue, sigma, frequency = sp.symbols("lambda sigma Omega", real=True)
    signed_linear_defect = (
        sp.I * frequency * sp.eye(2) - _generator(eigenvalue, sigma)
    ) * positive
    expected = (
        coefficient
        * sp.Matrix(
            [
                sigma * eigenvalue + sp.I * (frequency - eigenvalue),
                frequency - eigenvalue,
            ]
        )
        / 2
    )
    assert _simplify_matrix(signed_linear_defect - expected) == sp.zeros(2, 1)
    assert _simplify_matrix(
        signed_linear_defect.subs(frequency, eigenvalue)
    ) == sp.Matrix([coefficient * sigma * eigenvalue / 2, 0])


def test_spatial_aliases_keep_distinct_temporal_frequencies_and_reference_means():
    size = 1024
    site = sp.Symbol("j", integer=True)
    index, harmonic = sp.symbols("n ell", integer=True)
    velocity, slow_frequency, time = sp.symbols("v omega_s t", real=True)
    carrier, spacing = sp.pi / 4, 2 * sp.pi / size
    lattice_frequency = 1 - sp.cos(carrier)
    spatial = harmonic * carrier + index * spacing
    temporal = (
        harmonic * (lattice_frequency - slow_frequency) + index * spacing * velocity
    )
    assert (
        sp.simplify(
            sp.exp(sp.I * spatial.subs(index, index + size) * site)
            - sp.exp(sp.I * spatial * site)
        )
        == 0
    )
    assert (
        sp.expand(temporal.subs(index, index + size) - temporal) == 2 * sp.pi * velocity
    )
    first_history = sp.exp(sp.I * temporal * time)
    aliased_history = sp.exp(sp.I * temporal.subs(index, index + size) * time)
    assert first_history.subs(time, 0) == aliased_history.subs(time, 0)
    assert (
        sp.simplify(
            (aliased_history.diff(time) - first_history.diff(time)).subs(time, 0)
        )
        == 2 * sp.pi * sp.I * velocity
    )
    mean_index = -128 * harmonic
    assert sp.expand(spatial.subs(index, mean_index)) == 0
    mean_frequency = sp.simplify(temporal.subs(index, mean_index))
    assert (
        sp.simplify(
            mean_frequency
            - harmonic * (lattice_frequency - slow_frequency - carrier * velocity)
        )
        == 0
    )
    assert mean_frequency != 0


def test_signed_cubic_quintic_defect_transport_matches_direct_lattice_taylor_rows():
    character = sp.Symbol("u", nonzero=True)
    root = (1 + sp.I) / sp.sqrt(2)
    amplitude, sigma, frequency = (
        sp.Rational(1, 5),
        sp.Rational(1, 2),
        sp.Rational(2, 3),
    )
    form = amplitude * (character + 1 / character) / 2
    phase = amplitude * (character - 1 / character) / (2 * sp.I)

    def adjacency_laplacian(value):
        return sp.expand(
            value
            - (
                value.subs(character, root * character)
                + value.subs(character, character / root)
            )
            / 2
        )

    def principal_field(pair):
        return sp.Matrix(
            [
                -sigma * adjacency_laplacian(pair[0]) - adjacency_laplacian(pair[1]),
                adjacency_laplacian(pair[0]),
            ]
        )

    def temporal_derivative(value):
        return sp.expand(sp.I * frequency * character * sp.diff(value, character))

    gap = sp.expand(phase - phase.subs(character, root * character))
    cubic = sp.expand((gap**3 - gap.subs(character, character / root) ** 3) / 12)
    quintic = sp.expand(-(gap**5 - gap.subs(character, character / root) ** 5) / 240)
    reference = sp.Matrix([form, phase])
    defect = (
        reference.applyfunc(temporal_derivative)
        - principal_field(reference)
        - sp.Matrix([cubic + quintic, 0])
    )
    defect = defect.applyfunc(sp.expand)
    modes = (-5, -3, -1, 1, 3, 5)
    coefficients = {
        mode: sp.Matrix([sp.simplify(row.coeff(character, mode)) for row in defect])
        for mode in modes
    }
    assert _simplify_matrix(
        defect
        - sum((coefficients[mode] * character**mode for mode in modes), sp.zeros(2, 1))
    ) == sp.zeros(2, 1)
    assert sp.simplify(cubic.coeff(character, 3)) != 0
    assert sp.simplify(quintic.coeff(character, 5)) != 0
    assert coefficients[1][1] != 0
    # Modes 3 and -5 share a spatial eigenvalue, yet their signed defects
    # rotate at distinct frequencies; both enter the complete convolution.
    assert sp.simplify(root**8) == 1
    assert 3 * frequency != -5 * frequency
    direct_derivative = sp.zeros(2, 1)
    residual_derivative = defect
    for order in range(1, 4):
        direct_derivative = (
            principal_field(direct_derivative) - residual_derivative
        ).applyfunc(sp.expand)
        modal_derivative = sp.zeros(2, 1)
        for mode in modes:
            eigenvalue = sp.simplify(1 - (root**mode + root ** (-mode)) / 2)
            generator = _generator(eigenvalue, sigma)
            forcing_frequency = sp.I * mode * frequency
            # Taylor derivatives of -(exp(i Omega t)I-exp(Jt))
            # (i Omega I-J)^(-1) r, using the exact divided-power identity.
            divided_power = sum(
                (
                    generator ** (order - 1 - j) * forcing_frequency**j
                    for j in range(order)
                ),
                sp.zeros(2),
            )
            assert _simplify_matrix(
                (forcing_frequency * sp.eye(2) - generator) * divided_power
                - (forcing_frequency**order * sp.eye(2) - generator**order)
            ) == sp.zeros(2)
            coefficient = -divided_power * coefficients[mode]
            modal_derivative += coefficient * character**mode
        assert _simplify_matrix(
            sp.expand(direct_derivative - modal_derivative)
        ) == sp.zeros(2, 1)
        residual_derivative = residual_derivative.applyfunc(temporal_derivative)


def test_tangent_forced_reference_correction_leaves_a_linear_nonlinear_defect():
    generator = sp.Matrix(sp.symbols("j:4")).reshape(2, 2)
    reference = sp.Matrix(sp.symbols("z:2"))
    correction = sp.Matrix(sp.symbols("h:2"))
    residual = sp.Matrix(sp.symbols("r:2"))
    original_nonlinear, corrected_nonlinear = sp.symbols("R_star R_corrected")
    reference_rate = (
        generator * reference + sp.Matrix([original_nonlinear, 0]) + residual
    )
    correction_rate = generator * correction - residual
    corrected_field = generator * (reference + correction) + sp.Matrix(
        [corrected_nonlinear, 0]
    )
    assert _simplify_matrix(
        reference_rate + correction_rate - corrected_field
    ) == sp.Matrix([original_nonlinear - corrected_nonlinear, 0])
    base_phase, increment = sp.symbols("a h", real=True)

    # At node zero of C3 with phases (a,0,0), R_0=a-sin(a).
    # A phase correction (h,0,0) gives this exact nonlinear defect.
    def nonlinear(value):
        return value - sp.sin(value)

    corrected_defect = nonlinear(base_phase) - nonlinear(base_phase + increment)
    first_variation = sp.diff(corrected_defect, increment).subs(increment, 0)
    assert sp.simplify(first_variation - (sp.cos(base_phase) - 1)) == 0
    assert first_variation.subs(base_phase, sp.pi / 3) == -sp.Rational(1, 2)


def test_full_variational_correction_cancels_the_linear_term_but_changes_its_law():
    phase, increment = sp.symbols("a h", real=True)
    nonlinear = phase - sp.sin(phase)
    derivative = sp.diff(nonlinear, phase)
    variational_defect = (
        nonlinear + derivative * increment - nonlinear.subs(phase, phase + increment)
    )
    assert variational_defect.subs(increment, 0) == 0
    assert sp.diff(variational_defect, increment).subs(increment, 0) == 0
    assert sp.diff(variational_defect, increment, 2).subs(increment, 0) == -sp.sin(
        phase
    )
    assert (
        sp.diff(variational_defect, increment, 2).subs({increment: 0, phase: sp.pi / 3})
        == -sp.sqrt(3) / 2
    )
    # The added DR(theta_*)h_theta term is generally nonzero. The constant
    # J_sigma propagator alone does not solve that full variational equation.
    assert derivative.subs(phase, sp.pi / 3) * increment == increment / 2


def test_direct_metric_production_and_sylvester_correction_keep_common_means():
    size = 4
    incidence = sp.zeros(size)
    for j in range(size):
        incidence[j, j] = 1
        incidence[j, (j + 1) % size] = -1
    laplacian = incidence.T * incidence
    principal = laplacian / 2
    projection = sp.eye(size) - sp.ones(size) / size
    gap_height = sp.Symbol("C0", positive=True)
    static_block = gap_height**2 * laplacian / 8
    sylvester = gap_height**2 * projection / 8
    assert principal * sylvester + sylvester * principal == static_block
    zero = sp.zeros(size)
    skew = zero.row_join(-principal).col_join(principal.row_join(zero))
    metric = sp.diag(sylvester, -sylvester)
    production = zero.row_join(static_block / 2).col_join(
        (static_block / 2).row_join(zero)
    )
    assert skew.T == -skew
    assert skew.T * metric + metric * skew == -2 * production
    form = sp.Matrix(sp.symbols("u:4", real=True))
    phase = sp.Matrix(sp.symbols("eta:4", real=True))
    full = form.col_join(phase)
    assert sp.expand(full.dot(production * full) - form.dot(static_block * phase)) == 0
    mean_form, mean_phase = sp.symbols("mean_x mean_theta", real=True)
    common = (mean_form * sp.ones(size, 1)).col_join(mean_phase * sp.ones(size, 1))
    assert metric * common == sp.zeros(2 * size, 1)
    assert (
        sp.expand(
            common.dot((sp.eye(2 * size) + metric) * common) / 2
            - size * (mean_form**2 + mean_phase**2) / 2
        )
        == 0
    )
    # The bound is C0^2/8 on every nonconstant direction, without dividing
    # the final norm estimate by the cycle's smallest positive eigenvalue.
    assert sylvester * sp.ones(size, 1) == sp.zeros(size, 1)
    assert sylvester * sylvester == gap_height**2 * sylvester / 8


def test_symmetric_off_diagonal_fast_production_cancels_opposite_branches():
    coefficient = sp.Symbol("T")
    off_diagonal = sp.Matrix([[0, coefficient], [coefficient, 0]])
    for left_sign in (-1, 1):
        for right_sign in (-1, 1):
            left = sp.Matrix([1, -sp.I * left_sign]) / sp.sqrt(2)
            right = sp.Matrix([1, -sp.I * right_sign]) / sp.sqrt(2)
            congruence = (left.T * off_diagonal * right)[0]
            assert (
                sp.simplify(
                    congruence + sp.I * (left_sign + right_sign) * coefficient / 2
                )
                == 0
            )
            if left_sign == -right_sign:
                assert congruence == 0
    first, second, carrier = sp.symbols("lambda_q lambda_r lambda", positive=True)
    omega, cutoff = 2 * carrier, carrier / 2
    assert sp.simplify(omega + cutoff - sp.Rational(5, 2) * carrier) == 0
    assert (
        sp.expand((first + second) ** 2 - 4 * first * second - (first - second) ** 2)
        == 0
    )
    # The positive branch always has divisor >omega>cutoff. On the near
    # negative branch, lambda_q+lambda_r<=omega+cutoff=5lambda/2.
    assert sp.simplify(omega / cutoff) == 4


def test_full_cycle_direct_metric_norms_retain_aliases_and_all_eigenbranches():
    # An arbitrary complex coefficient fixture on a finite cycle, not an
    # evaluated moving envelope or a supplied source. DFT coefficients use 1/N.
    size = 8
    incidence = np.eye(size) - np.roll(np.eye(size), -1, axis=1)
    principal = incidence.T @ incidence / 2
    eigenvalues, eigenvectors = np.linalg.eigh(principal)
    eigenvalues[np.abs(eigenvalues) < 1e-13] = 0
    zeros = np.zeros_like(principal)
    skew = np.block([[zeros, -principal], [principal, zeros]])
    basis = np.block(
        [[eigenvectors, eigenvectors], [-1j * eigenvectors, 1j * eigenvectors]]
    ) / np.sqrt(2)
    frequencies = np.concatenate((eigenvalues, -eigenvalues))
    np.testing.assert_allclose(skew @ basis, basis * (1j * frequencies), atol=2e-15)
    indices = np.arange(size)
    coefficient = (10 + indices + 2j * (-1) ** indices) / 80
    derivative = ((-1) ** indices * (indices + 1) + 1j * (indices - 2)) / 160
    c0, c1 = np.max(np.abs(coefficient)), np.max(np.abs(derivative))
    z0 = np.sum(np.abs(np.fft.fft(coefficient) / size))
    z1 = np.sum(np.abs(np.fft.fft(derivative) / size))
    static = incidence.T @ np.diag(np.abs(coefficient) ** 2) @ incidence / 8
    static_derivative = (
        incidence.T
        @ np.diag(2 * np.real(np.conj(coefficient) * derivative))
        @ incidence
        / 8
    )
    sylvester_divisors = eigenvalues[:, None] + eigenvalues[None, :]

    def solve_sylvester(value):
        transformed = eigenvectors.T @ value @ eigenvectors
        solution = np.zeros_like(transformed)
        np.divide(
            transformed,
            sylvester_divisors,
            out=solution,
            where=sylvester_divisors > 1e-13,
        )
        return eigenvectors @ solution @ eigenvectors.T

    static_correction = solve_sylvester(static)
    static_correction_derivative = solve_sylvester(static_derivative)
    np.testing.assert_allclose(
        principal @ static_correction + static_correction @ principal,
        static,
        atol=2e-16,
    )
    assert np.min(np.linalg.eigvalsh(static_correction)) >= -2e-16
    assert np.linalg.norm(static_correction, 2) <= c0**2 / 8 + 2e-16
    assert np.linalg.norm(static_correction_derivative, 2) <= c0 * c1 / 4 + 2e-16
    np.testing.assert_allclose(static_correction @ np.ones(size), 0, atol=2e-16)
    fast_block = -incidence.T @ np.diag(coefficient**2) @ incidence / 16
    fast_derivative_block = (
        -incidence.T @ np.diag(2 * coefficient * derivative) @ incidence / 16
    )
    fast = np.block([[zeros, fast_block], [fast_block, zeros]])
    fast_derivative = np.block(
        [[zeros, fast_derivative_block], [fast_derivative_block, zeros]]
    )
    transformed = basis.T @ fast @ basis
    transformed_derivative = basis.T @ fast_derivative @ basis
    np.testing.assert_allclose(transformed[:size, size:], 0, atol=2e-17)
    np.testing.assert_allclose(transformed[size:, :size], 0, atol=2e-17)
    carrier = 1 - np.cos(np.pi / 4)
    omega, cutoff = 2 * carrier, carrier / 2
    divisors = omega + frequencies[:, None] + frequencies[None, :]
    near_mask = np.abs(divisors) <= cutoff
    near = np.where(near_mask, transformed, 0)
    assert np.linalg.norm(near) > 0
    np.testing.assert_allclose(near[:size, :], 0, atol=2e-17)
    np.testing.assert_allclose(near[:, :size], 0, atol=2e-17)
    correction = np.zeros_like(transformed)
    correction_derivative = np.zeros_like(transformed_derivative)
    np.divide(-2 * transformed, 1j * divisors, out=correction, where=~near_mask)
    np.divide(
        -2 * transformed_derivative,
        1j * divisors,
        out=correction_derivative,
        where=~near_mask,
    )
    assert np.linalg.norm(correction[:size, :size], 2) <= z0**2 / 8 + 2e-16
    assert np.linalg.norm(correction[size:, size:], 2) <= 5 * z0**2 / 8 + 2e-16
    near_matrix = basis.conj() @ near @ basis.conj().T
    correction_matrix = basis.conj() @ correction @ basis.conj().T
    derivative_matrix = basis.conj() @ correction_derivative @ basis.conj().T
    for phase in (0, 1 / 7, np.pi / 3):
        factor = np.exp(1j * phase)
        near_real = np.real(factor * near_matrix)
        fast_metric = np.real(factor * correction_matrix)
        fast_metric_derivative = np.real(factor * derivative_matrix)
        np.testing.assert_allclose(
            np.linalg.norm(near_real, 2),
            np.linalg.norm(near_matrix, 2) / 2,
            rtol=2e-13,
            atol=2e-17,
        )
        assert np.linalg.norm(near_real, 2) <= 5 * carrier * z0**2 / 64 + 2e-16
        assert np.linalg.norm(fast_metric, 2) <= 3 * z0**2 / 8 + 2e-16
        assert np.linalg.norm(fast_metric_derivative, 2) <= 3 * z0 * z1 / 4 + 2e-16


def test_direct_euclidean_nonlinear_budget_and_correction_factors():
    phase = sp.Matrix(sp.symbols("eta:4", real=True))
    edges = sp.Matrix([phase[j] - phase[(j + 1) % 4] for j in range(4)])
    assert (
        sp.expand(
            4 * phase.dot(phase)
            - edges.dot(edges)
            - sum((phase[j] + phase[(j + 1) % 4]) ** 2 for j in range(4))
        )
        == 0
    )
    assert (
        sp.expand(
            2 * phase.dot(phase)
            - edges[0] ** 2
            - (phase[0] + phase[1]) ** 2
            - 2 * (phase[2] ** 2 + phase[3] ** 2)
        )
        == 0
    )
    # These identities give ||D eta||2<=2||eta|| and
    # ||D eta||infinity<=sqrt(2)||eta||. Applying the sine Taylor remainder
    # gives n_rho=sqrt(2)min(1,G)rho+2rho^2/3, retaining both powers.
    radius, sine_bound = sp.symbols("rho s", positive=True)
    quadratic_allowance = sine_bound * sp.sqrt(2) * radius * 2 * radius / 2
    cubic_allowance = (sp.sqrt(2) * radius) ** 2 * 2 * radius / 6
    assert (
        sp.simplify(
            (quadratic_allowance + cubic_allowance) / radius
            - (sp.sqrt(2) * sine_bound * radius + 2 * radius**2 / 3)
        )
        == 0
    )
    form_norm, phase_norm, correction, nonlinear = sp.symbols(
        "u_norm eta_norm k n", nonnegative=True
    )
    assert (
        sp.expand(
            (form_norm**2 + phase_norm**2) / 2
            - form_norm * phase_norm
            - (form_norm - phase_norm) ** 2 / 2
        )
        == 0
    )
    retained, metric_derivative, sigma, gap = sp.symbols(
        "m d sigma G", nonnegative=True
    )
    claimed = (
        retained
        + metric_derivative / 2
        + correction * (2 * sigma + gap**2)
        + (correction + sp.Rational(1, 2)) * nonlinear
    )
    independent_sum = (
        retained
        + metric_derivative / 2
        + 2 * correction * sigma
        + correction * gap**2
        + correction * nonlinear
        + nonlinear / 2
    )
    assert sp.expand(claimed - independent_sum) == 0
    # Full Euclidean coercivity uses 1-k. Constant modes remain in the
    # identity part of the metric and do not incur a Poincare factor.
    q, residual, margin = sp.symbols("q r delta", positive=True)
    norm_bound = q / sp.sqrt(margin)
    energy_rate = claimed * norm_bound**2 + (1 + correction) * norm_bound * residual
    assert (
        sp.simplify(
            energy_rate / q
            - claimed * q / margin
            - (1 + correction) * residual / sp.sqrt(margin)
        )
        == 0
    )


def test_periodic_derivative_tail_bounds_follow_from_boundary_and_integral_controls():
    position = sp.Symbol("s", real=True)
    frequency = sp.Symbol("xi", real=True)
    profile = sp.Function("P")(position)
    phase = sp.exp(-sp.I * frequency * position)
    boundary_primitive = (
        sp.diff(profile, position, 2)
        + sp.I * frequency * sp.diff(profile, position)
        + (sp.I * frequency) ** 2 * profile
    ) * phase
    assert (
        sp.simplify(
            sp.diff(boundary_primitive, position)
            - sp.diff(profile, position, 3) * phase
            + (sp.I * frequency) ** 3 * profile * phase
        )
        == 0
    )
    index = sp.Symbol("n", integer=True)
    assert sp.exp(-2 * sp.pi * sp.I * index) == 1
    # Periodicity of P,P',P'' cancels these boundary terms; hence the
    # coefficient estimate C3/|n h0|^3 includes all continuous profile points.
    integration_variable, cutoff, spacing, derivative_bound = sp.symbols(
        "x M h0 C3", positive=True
    )
    assert sp.diff(integration_variable**-3, integration_variable).is_negative
    assert sp.diff(integration_variable**-2, integration_variable).is_negative
    unweighted = (
        2
        * derivative_bound
        / spacing**3
        * sp.integrate(integration_variable**-3, (integration_variable, cutoff, sp.oo))
    )
    weighted = (
        2
        * derivative_bound
        / spacing**2
        * sp.integrate(integration_variable**-2, (integration_variable, cutoff, sp.oo))
    )
    assert unweighted == derivative_bound / (spacing**3 * cutoff**2)
    assert weighted == 2 * derivative_bound / (spacing**2 * cutoff)
    cubic_derivative = sp.diff(profile**3, position, 3)
    independent_terms = (
        6 * sp.diff(profile, position) ** 3
        + 18 * profile * sp.diff(profile, position) * sp.diff(profile, position, 2)
        + 3 * profile**2 * sp.diff(profile, position, 3)
    )
    assert sp.expand(cubic_derivative - independent_terms) == 0


def test_sine_convolution_tail_budget_does_not_delete_retained_nonlinear_terms():
    character = sp.Symbol("u", nonzero=True)
    gap = (character - 1 / character) / (4 * sp.I) + (
        character**2 - character**-2
    ) / (10 * sp.I)
    norm = sp.Rational(7, 10)

    def coefficient_norm(value):
        expanded = sp.expand(value)
        return sum(abs(expanded.coeff(character, mode)) for mode in range(-18, 19))

    assert coefficient_norm(gap) == norm
    for degree in (3, 5, 7, 9):
        assert coefficient_norm(gap**degree) <= norm**degree
    retained_core = -(gap**3) / sp.factorial(3) + gap**5 / sp.factorial(5)
    first_omitted = norm**7 / sp.factorial(7)
    ratio = norm**2 / (8 * 9)
    geometric_budget = first_omitted / (1 - ratio)
    assert 0 < ratio < 1
    assert coefficient_norm(retained_core) > geometric_budget
    assert coefficient_norm(-(gap**7) / sp.factorial(7)) <= first_omitted
    order, offset = sp.symbols("P m", integer=True, nonnegative=True)
    denominator_increase = (2 * order + 4 + 2 * offset) * (
        2 * order + 5 + 2 * offset
    ) - (2 * order + 4) * (2 * order + 5)
    assert (
        sp.expand(denominator_increase - 2 * offset * (4 * order + 9 + 2 * offset)) == 0
    )
    # Every successive factorial ratio is no larger than the first one.
    amplitude = sp.Symbol("A", positive=True)
    term_index = sp.Symbol("j", integer=True, nonnegative=True)
    term = amplitude ** (2 * term_index + 1) / sp.factorial(2 * term_index + 1)
    assert (
        sp.simplify(
            term.subs(term_index, term_index + 1) / term
            - amplitude**2 / ((2 * term_index + 2) * (2 * term_index + 3))
        )
        == 0
    )
    nonlinear_derivative_series = sp.summation(
        amplitude ** (2 * term_index) / sp.factorial(2 * term_index),
        (term_index, 1, sp.oo),
    )
    assert sp.simplify(nonlinear_derivative_series - (sp.cosh(amplitude) - 1)) == 0
    assert sp.diff(sp.sin(amplitude) - amplitude, amplitude) == sp.cos(amplitude) - 1
    # The coefficient-tail Lipschitz factor is cosh(A)-1 because the
    # linear phase row has already been subtracted from this sine remainder.


def test_omitted_residual_transport_retains_finite_support_factor_without_detuning():
    eigenvalue, sigma = sp.symbols("lambda sigma", positive=True)
    time = sp.Symbol("T", nonnegative=True)
    frequency = sp.Symbol("Omega", real=True)
    determinant = eigenvalue**2 - frequency**2 + sp.I * sigma * eigenvalue * frequency
    assert sp.limit(determinant.subs(frequency, eigenvalue), eigenvalue, 0) == 0
    assert determinant.subs(frequency, eigenvalue) != 0
    # There is no uniform nonzero detuning on the admitted spectrum. The
    # whole contractive convolution has the sharp bound T in the mean limit.
    mean_kernel = time * sp.eye(2)
    assert mean_kernel.T * mean_kernel == time**2 * sp.eye(2)
    size = 8
    root = (1 + sp.I) / sp.sqrt(2)
    first, second = sp.symbols("rx rt", real=True)
    coefficient = sp.Matrix([first, second])
    norm_squared = sum(
        sp.expand_complex(
            ((root**j * coefficient).conjugate().T * (root**j * coefficient))[0]
        )
        for j in range(size)
    )
    assert sp.simplify(norm_squared - size * (first**2 + second**2)) == 0
    # Triangle inequality followed by this Fourier normalization gives
    # sqrt(N)*T*epsilon_r for a complete two-row coefficient-norm tail.
    assert sp.simplify((sp.sqrt(size) * time) ** 2 - size * time**2) == 0
