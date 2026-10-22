"""Static eligibility controls for a phase-sensitive envelope mechanism.

These checks neither instantiate a new preparation nor evaluate a trajectory.
An envelope is a conditional reference, not an installed TNFR evolution law.
"""

from fractions import Fraction as Q

import mpmath as mp
import pytest
import sympy as sp

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I, pi_interval, sin
from tnfr.physics.relational_sine_comparison import _sine_rates


@pytest.fixture(scope="module", autouse=True)
def no_source_or_response_evaluation():
    with forbid_organized_response_evaluation():
        yield


def _simplify(value):
    return sp.simplify(sp.expand(value))


def test_cubic_harmonics_follow_from_actual_incidence_and_both_real_phase_components():
    # u is the fast character exp(i*(k*j+lambda*t)), not a state coordinate.
    u, b, conjugate_b = sp.symbols("u b conjugate_b", nonzero=True)
    root = (1 + sp.I) / sp.sqrt(2)
    phase = (b * u - conjugate_b / u) / (2 * sp.I)
    edge = sp.expand(phase - phase.subs(u, root * u))
    cubic = sp.expand((edge**3 - edge.subs(u, u / root) ** 3) / 12)
    coefficients = {mode: _simplify(cubic.coeff(u, mode)) for mode in (-3, -1, 1, 3)}
    eigenvalue = 1 - sp.sqrt(2) / 2
    resonant = eigenvalue**2 / 8
    third = (sp.sqrt(2) - 1) / 48
    expected = {
        1: -sp.I * resonant * b**2 * conjugate_b,
        -1: sp.I * resonant * b * conjugate_b**2,
        3: -sp.I * third * b**3,
        -3: sp.I * third * conjugate_b**3,
    }
    assert all(_simplify(coefficients[k] - value) == 0 for k, value in expected.items())
    assert _simplify(cubic - sum(coefficients[k] * u**k for k in coefficients)) == 0
    assert (
        coefficients[-1] != 0
    )  # Discarding the conjugate fundamental loses a real row.
    assert coefficients[-3] != 0


def test_nonresonant_correctors_cancel_with_the_actual_lattice_frequencies():
    eigenvalue = 1 - sp.sqrt(2) / 2
    third_eigenvalue = 1 + sp.sqrt(2) / 2
    resonant, third = eigenvalue**2 / 8, (sp.sqrt(2) - 1) / 48
    forcing = {-1: sp.I * resonant, 3: -sp.I * third, -3: sp.I * third}
    frequencies = {-1: eigenvalue, 3: third_eigenvalue, -3: third_eigenvalue}
    derived = {
        mode: _simplify(value / (sp.I * (mode * eigenvalue - frequencies[mode])))
        for mode, value in forcing.items()
    }
    expected = {
        -1: -(2 - sp.sqrt(2)) / 32,
        3: sp.Rational(1, 96),
        -3: -(3 * sp.sqrt(2) - 2) / 672,
    }
    assert all(_simplify(derived[k] - value) == 0 for k, value in expected.items())
    assert 3 * eigenvalue - third_eigenvalue != 0
    assert derived[3] != sp.conjugate(derived[-3])
    # The resonant fundamental cannot be removed by a finite harmonic inverse.
    assert resonant > 0
    wrong_same_frequency = _simplify(
        sp.I * (3 * eigenvalue - eigenvalue) * derived[3] - forcing[3]
    )
    assert wrong_same_frequency != 0


def test_third_harmonic_resonance_and_cycle_periodicity_are_independent_admissions():
    cosine = sp.Symbol("c", real=True)
    mismatch = sp.expand(3 * (1 - cosine) - (1 - (4 * cosine**3 - 3 * cosine)))
    assert (
        sp.expand(mismatch - 2 * (cosine - 1) * (2 * cosine**2 + 2 * cosine - 1)) == 0
    )
    assert sp.simplify(mismatch.subs(cosine, (sp.sqrt(3) - 1) / 2)) == 0
    assert sp.simplify(mismatch.subs(cosine, sp.sqrt(2) / 2)) != 0
    assert Q(1024, 8).denominator == 1  # k=pi/4 is a C1024 character.
    assert Q(1024, 6).denominator != 1  # k=pi/3 is not.
    assert (
        sp.simplify(sp.exp(3 * sp.I * sp.pi / 4) - sp.exp(-3 * sp.I * sp.pi / 4)) != 0
    )
    assert sp.exp(sp.I * sp.pi) == sp.exp(-sp.I * sp.pi)  # pi/3 has a third-mode alias.


def test_two_real_rows_and_group_transport_fix_the_slow_equation_sign():
    ax, atheta, current, sigma = sp.symbols("Ax Atheta S sigma", real=True)
    actual = -sigma * ax + current / 2 + sp.I * ax
    complex_tangent = sp.I * (ax + sp.I * atheta) - sigma * ax
    assert sp.expand(actual - complex_tangent) == atheta + current / 2

    eps, k = sp.symbols("eps k", real=True)
    b, bx, bxx, bxxx = sp.symbols("B Bx Bxx Bxxx")
    shifted_right = b + eps * bx + eps**2 * bxx / 2 + eps**3 * bxxx / 6
    shifted_left = b - eps * bx + eps**2 * bxx / 2 - eps**3 * bxxx / 6
    laplacian = (
        b
        - (
            (sp.cos(k) + sp.I * sp.sin(k)) * shifted_right
            + (sp.cos(k) - sp.I * sp.sin(k)) * shifted_left
        )
        / 2
    )
    expanded = sp.expand(sp.I * laplacian)
    assert expanded.coeff(eps, 1) == sp.sin(k) * bx
    assert expanded.coeff(eps, 2) == -sp.I * sp.cos(k) * bxx / 2
    assert expanded.coeff(eps, 3) == sp.sin(k) * bxxx / 6
    # X=eps*(j+sin(k)*tau): positive carrier has negative physical group speed.
    assert -sp.sin(sp.pi / 4) < 0 < -sp.sin(-sp.pi / 4)


def test_focusing_sech_requires_positive_curvature_and_has_soft_frequency_shift():
    position = sp.Symbol("X", real=True)
    amplitude, width, dispersion, cubic = sp.symbols("b kappa P c", positive=True)
    profile = amplitude / sp.cosh(width * position)
    identity = (
        sp.diff(profile, position, 2)
        - width**2 * profile
        + 2 * width**2 * profile**3 / amplitude**2
    )
    assert sp.simplify(sp.expand_trig(identity * sp.cosh(width * position) ** 3)) == 0
    width_squared = cubic * amplitude**2 / (2 * dispersion)
    slow_frequency = cubic * amplitude**2 / 2
    residual = (dispersion * width_squared - slow_frequency) * profile + (
        cubic - 2 * dispersion * width_squared / amplitude**2
    ) * profile**3
    assert sp.expand(residual) == 0
    # B=profile*exp(-i*Omega*T) gives i*B_T=Omega*B. A plus sign in
    # that phase would raise the carrier frequency and fail this equation.
    time = sp.Symbol("T", real=True)
    envelope = profile * sp.exp(-sp.I * slow_frequency * time)
    assert sp.simplify(sp.I * sp.diff(envelope, time) - slow_frequency * envelope) == 0
    assert sp.cos(sp.pi / 4) / 2 > 0
    assert sp.cos(sp.pi / 2) / 2 == 0
    assert sp.cos(3 * sp.pi / 4) / 2 < 0
    assert (cubic * amplitude**2 / (2 * dispersion)).is_positive
    assert (cubic * amplitude**2 / (-2 * dispersion)).is_negative


def test_stationary_envelope_first_integral_does_not_make_a_sech_periodic():
    position = sp.Symbol("X", real=True)
    dispersion, cubic, frequency, length = sp.symbols("P c Omega L", positive=True)
    profile = sp.Function("f")(position)
    energy = (
        dispersion * sp.diff(profile, position) ** 2 / 2
        + cubic * profile**4 / 4
        - frequency * profile**2 / 2
    )
    derivative = sp.diff(energy, position)
    expected = sp.diff(profile, position) * (
        dispersion * sp.diff(profile, position, 2)
        + cubic * profile**3
        - frequency * profile
    )
    assert sp.expand(derivative - expected) == 0
    # Endpoint values of a centered sech match, but derivatives have opposite
    # nonzero signs. Simply wrapping it does not give a smooth periodic reference.
    sech = 1 / sp.cosh(position)
    assert (
        sp.simplify(sech.subs(position, length / 2) - sech.subs(position, -length / 2))
        == 0
    )
    derivative_jump = sp.simplify(
        sp.diff(sech, position).subs(position, length / 2)
        - sp.diff(sech, position).subs(position, -length / 2)
    )
    assert derivative_jump.is_negative


def test_fixed_positive_loss_enters_both_the_slow_equation_and_contrarotating_corrector():
    eps, sigma, eigenvalue = sp.symbols("eps sigma lambda", positive=True)
    b, conjugate_b, correction = sp.symbols("B conjugate_B C")
    gamma = sigma / eps**2  # Bookkeeping; the actual loss is not changed.
    resonant_rate = -gamma * eigenvalue * b / 2
    assert sp.simplify(eps**3 * resonant_rate + sigma * eigenvalue * eps * b / 2) == 0
    derived = sp.solve(
        sp.Eq(
            -2 * sp.I * eigenvalue * correction, -gamma * eigenvalue * conjugate_b / 2
        ),
        correction,
    )[0]
    assert sp.simplify(derived + sp.I * gamma * conjugate_b / 4) == 0
    assert (
        sp.simplify(
            (-2 * sp.I * eigenvalue) * sp.conjugate(-sp.I) * gamma * conjugate_b / 4
            + gamma * eigenvalue * conjugate_b / 2
        )
        != 0
    )
    # Keeping a conservative envelope instead leaves this positive-size
    # damping residual; no cancellation or small-sigma limit is implicit.
    assert (sigma / eps**2).is_positive


def test_actual_shared_field_retains_damping_and_the_declared_clock():
    model = RelationalExchangeModel(
        storage_scale=1,
        epi_weight=Q(1, 4),
        phase_weight=Q(3, 4),
        phase_domain="regular",
    )
    forms = tuple(Q(j - 3, 50) for j in range(8))
    phases = tuple(Q((-1) ** j * (j + 1), 100) for j in range(8))
    gradient = tuple(2 * forms[j] - forms[j - 1] - forms[(j + 1) % 8] for j in range(8))
    currents = tuple(
        sin(I(phases[j - 1] - phases[j])) + sin(I(phases[(j + 1) % 8] - phases[j]))
        for j in range(8)
    )
    rates = _sine_rates(model, (2,) * 8, gradient, (1,) * 8, currents)
    pi = pi_interval()
    clock, sigma = pi / Q(model.phase_weight), pi * Q(model.epi_weight) / Q(
        model.phase_weight
    )
    for j in range(8):
        difference = clock * rates["form_rates"][j] - currents[j] / 2
        expected = -sigma * gradient[j] / 2
        assert (difference - expected).contains(0)
        assert (difference - expected).abs_max < Q(1, 10**32)
        phase_difference = clock * rates["phase_rates"][j] - gradient[j] / 2
        assert phase_difference.contains(0) and phase_difference.abs_max < Q(1, 10**32)
    assert not (clock * rates["form_rates"][0] - currents[0] / 2).contains(0)


def test_phase_sensitive_remainder_lipschitz_bound_and_residual_shadow_majorant():
    size = 5
    values = sp.symbols("v:5", real=True)
    gaps = [values[j] - values[(j + 1) % size] for j in range(size)]
    squares = sum(value**2 for value in values)
    assert (
        sp.expand(
            4 * squares
            - sum(gap**2 for gap in gaps)
            - sum((values[j] + values[(j + 1) % size]) ** 2 for j in range(size))
        )
        == 0
    )
    # DR=(1/2)D^T diag(1-cos(edge phase))D. If |edge phase|<=H,
    # the diagonal is <=H^2/2 and the resulting operator norm is <=H^2.
    h = sp.Symbol("H", positive=True)
    weights = sp.symbols("w:5", nonnegative=True)
    slack = h**2 * squares - sum(w * gap**2 / 2 for w, gap in zip(weights, gaps))
    decomposition = h**2 * (4 * squares - sum(gap**2 for gap in gaps)) / 4 + sum(
        (h**2 / 4 - w / 2) * gap**2 for w, gap in zip(weights, gaps)
    )
    assert sp.expand(slack - decomposition) == 0
    # Under the independently checked error-radius bootstrap, the positive
    # majorant dominates E'<=L E+rho; this does not assert a radius closes.
    time = sp.Symbol("t", nonnegative=True)
    initial, rate, lipschitz = sp.symbols("E0 rho L", positive=True)
    majorant = (initial + rate * time) * sp.exp(lipschitz * time)
    assert sp.simplify(sp.diff(majorant, time) - lipschitz * majorant) == rate * sp.exp(
        lipschitz * time
    )
    assert sp.simplify(majorant.subs(time, 0)) == initial


def test_edge_local_storage_has_antisymmetric_flux_and_fixed_region_loss():
    size = 5
    incidence = sp.zeros(size)
    for j in range(size):
        incidence[j, j], incidence[j, (j + 1) % size] = 1, -1
    laplacian = incidence.T * incidence
    forms = sp.Matrix(sp.symbols("x:5", real=True))
    phases = sp.Matrix(sp.symbols("theta:5", real=True))
    sigma = sp.Symbol("sigma", positive=True)
    form_gaps, phase_gaps = incidence * forms, incidence * phases
    currents = phase_gaps.applyfunc(sp.sin)
    form_rate = -sigma * laplacian * forms / 2 - incidence.T * currents / 2
    phase_rate = laplacian * forms / 2
    form_gap_rate, phase_gap_rate = incidence * form_rate, incidence * phase_rate
    local_rate = [
        form_gaps[j] * form_gap_rate[j] + currents[j] * phase_gap_rate[j]
        for j in range(size)
    ]

    def flux(j, k):
        return (currents[j] * form_gaps[k] - form_gaps[j] * currents[k]) / 2 + sigma * (
            form_gaps[j] ** 2 - form_gaps[k] ** 2
        ) / 4

    def neighbors(j):
        return ((j - 1) % size, (j + 1) % size)

    losses = [
        sigma * sum((form_gaps[j] - form_gaps[k]) ** 2 for k in neighbors(j)) / 4
        for j in range(size)
    ]
    for j in range(size):
        assert all(sp.expand(flux(j, k) + flux(k, j)) == 0 for k in neighbors(j))
        assert (
            sp.expand(local_rate[j] + sum(flux(j, k) for k in neighbors(j)) + losses[j])
            == 0
        )
    assert (
        sp.expand(
            sum(local_rate) + sigma * (laplacian * forms).dot(laplacian * forms) / 2
        )
        == 0
    )

    region = {0, 1, 2}  # The selected edge region is fixed throughout the identity.
    boundary_flux = sum(
        flux(j, k) for j in region for k in neighbors(j) if k not in region
    )
    assert (
        sp.expand(sum(local_rate[j] + losses[j] for j in region) + boundary_flux) == 0
    )
    omitted_damping_flux = (
        sigma
        * sum(
            form_gaps[j] ** 2 - form_gaps[k] ** 2
            for j in region
            for k in neighbors(j)
            if k not in region
        )
        / 4
    )
    assert sp.expand(omitted_damping_flux) != 0


def test_periodic_profile_first_integral_and_both_turning_amplitudes():
    coordinate = sp.Symbol("X", real=True)
    rho, kappa, dispersion = sp.symbols("rho kappa d", positive=True)
    modulus = sp.Symbol("m", real=True)  # The profile premise is 0<m<1.
    profile = sp.Function("f")(coordinate)
    acceleration = (
        kappa**2 * (2 - modulus) * profile - (2 * kappa**2 / rho**2) * profile**3
    )
    squared_speed = (
        kappa**2
        / rho**2
        * (rho**2 - profile**2)
        * (profile**2 - rho**2 * (1 - modulus))
    )
    first_integral = sp.diff(profile, coordinate) ** 2 - squared_speed
    # Differentiate without dividing by f': the identity includes turning points.
    derivative = sp.diff(first_integral, coordinate).subs(
        sp.diff(profile, coordinate, 2), acceleration
    )
    assert sp.expand(derivative) == 0
    peak, trough = rho, rho * sp.sqrt(1 - modulus)
    assert sp.simplify(squared_speed.subs(profile, peak)) == 0
    assert sp.simplify(squared_speed.subs(profile, trough)) == 0
    assert sp.simplify(acceleration.subs(profile, peak)) == -(kappa**2) * modulus * rho
    assert (
        sp.simplify(
            acceleration.subs(profile, trough)
            - kappa**2 * modulus * rho * sp.sqrt(1 - modulus)
        )
        == 0
    )
    nonlinear = 2 * dispersion * kappa**2 / rho**2
    frequency = dispersion * kappa**2 * (2 - modulus)
    assert (
        sp.expand(
            dispersion * acceleration + nonlinear * profile**3 - frequency * profile
        )
        == 0
    )
    # Reversing the cubic sign fails even though it retains the linear frequency.
    assert sp.expand(
        dispersion * acceleration - nonlinear * profile**3 - frequency * profile
    )


def test_periodic_plane_envelope_residual_against_both_full_site_rows():
    # A smooth, nonlocalized method fixture, evaluated only at fixed phases.
    # This numerical control supplements the exact harmonic identities above;
    # it neither integrates the field nor certifies a retained organization.
    with mp.workdps(60):
        size, epsilon, amplitude = 64, mp.mpf(1) / 8, mp.mpf(1) / 2
        carrier, sigma = mp.pi / 4, mp.pi / (2**20 - 1)
        spatial_frequency = 2 * mp.pi / (size * epsilon)
        lattice_frequency = 1 - mp.cos(carrier)
        velocity, dispersion = mp.sin(carrier), mp.cos(carrier) / 2
        nonlinear = lattice_frequency**2 / 8
        slow_frequency = -dispersion * spatial_frequency**2 + nonlinear * amplitude**2
        frequency = (
            lattice_frequency
            + epsilon * spatial_frequency * velocity
            - epsilon**2 * slow_frequency
        )
        correctors = {
            -1: -(2 - mp.sqrt(2)) / 32,
            3: mp.mpf(1) / 96,
            -3: -(3 * mp.sqrt(2) - 2) / 672,
        }
        coefficients = {1: epsilon * amplitude}
        coefficients.update(
            {
                mode: epsilon**3 * coefficient * amplitude**3
                for mode, coefficient in correctors.items()
            }
        )
        bounds = [amplitude * spatial_frequency**order for order in range(4)]
        b0, b1, b2, b3 = bounds
        slow_bound = abs(dispersion) * b2 + nonlinear * b0**3
        a0 = sum(abs(value) for value in correctors.values())
        a1 = sum(
            abs(value) * abs(velocity - mp.sin(mode * carrier))
            for mode, value in correctors.items()
        )
        edge_factor = abs(1 - mp.exp(1j * carrier))
        gap_bound = (
            epsilon * (edge_factor * b0 + epsilon * b1) + 2 * epsilon**3 * a0 * b0**3
        )
        residual_bound = (
            epsilon**4
            * (
                b3 / 6
                + 3 * a1 * b0**2 * b1
                + b1 * (edge_factor * b0 + epsilon * b1) ** 2 / 2
            )
            + epsilon**5
            * a0
            * (3 * b0**2 * slow_bound + 3 * b0 * b1**2 + mp.mpf(3) * b0**2 * b2 / 2)
            + epsilon**3 * a0 * b0**3 * gap_bound**2
            + gap_bound**5 / 120
            + 2 * sigma * (epsilon * b0 + epsilon**3 * a0 * b0**3)
        )
        assert all(bound > 0 for bound in bounds) and slow_frequency != 0
        assert abs(slow_frequency) * b0 <= slow_bound
        site_frequency = carrier + epsilon * spatial_frequency
        assert abs(mp.exp(1j * size * site_frequency) - 1) < mp.mpf("1e-55")
        for offset in (mp.mpf(0), mp.mpf(1) / 7, mp.mpf(2) / 5):
            states = [
                sum(
                    value * mp.exp(1j * mode * (site_frequency * j + offset))
                    for mode, value in coefficients.items()
                )
                for j in range(size)
            ]
            rates = [
                sum(
                    1j
                    * mode
                    * frequency
                    * value
                    * mp.exp(1j * mode * (site_frequency * j + offset))
                    for mode, value in coefficients.items()
                )
                for j in range(size)
            ]
            forms, phases = [value.real for value in states], [
                value.imag for value in states
            ]
            laplacian = [
                2 * forms[j] - forms[j - 1] - forms[(j + 1) % size] for j in range(size)
            ]
            currents = [
                mp.sin(phases[j - 1] - phases[j])
                + mp.sin(phases[(j + 1) % size] - phases[j])
                for j in range(size)
            ]
            actual = [
                -sigma * laplacian[j] / 2 + currents[j] / 2 + 1j * laplacian[j] / 2
                for j in range(size)
            ]
            residuals = [rates[j] - actual[j] for j in range(size)]
            assert max(abs(value) for value in residuals) < residual_bound
            assert (
                mp.sqrt(sum(abs(value) ** 2 for value in residuals))
                < mp.sqrt(size) * residual_bound
            )
            # Keeping just the form row cannot pass this full-state control.
            missing_phase_row = [rates[j] - actual[j].real for j in range(size)]
            assert (
                mp.sqrt(sum(abs(value) ** 2 for value in missing_phase_row))
                > mp.sqrt(size) * residual_bound
            )


def test_acute_coarse_certificate_obstruction_retains_actual_positive_loss():
    height, modulus, frequency, sigma = sp.symbols("b m lambda sigma", positive=True)
    nonlinear = frequency**2 / 8
    phase_time = 2 / (nonlinear * height**2 * (2 - modulus))
    edge_factor_squared = 2 * frequency
    assert (
        sp.simplify(
            edge_factor_squared * height**2 * phase_time
            - 32 / (frequency * (2 - modulus))
        )
        == 0
    )
    # k=pi/4 and the additional acute certificate supply the domain bounds;
    # these are not assumptions about every possible maintenance mechanism.
    actual_frequency = 1 - sp.sqrt(2) / 2
    assert Q(1, 4) < actual_frequency < Q(1, 3)
    assert actual_frequency**2 / 8 < Q(1, 72)
    assert edge_factor_squared.subs(frequency, actual_frequency) > Q(1, 4)
    pi = pi_interval()
    assert 3 < pi.lo < pi.hi < 4
    # G>=D_k*b, G+2r<pi/2 imply r<1 and b<4. With 0<m<1,
    # T_phi*b>1/(g*b), while L_*T_phi>=32/[lambda*(2-m)].
    height_upper = Q(4) / (2 * Q(1, 2))
    radius_upper = Q(4) / 4
    time_height_lower = 1 / (Q(1, 72) * height_upper)
    exponent_lower = 32 / (Q(1, 3) * 2)
    damping_prefactor = 2 * sp.sqrt(1024) * time_height_lower
    assert damping_prefactor == 1152
    assert (damping_prefactor * sigma).subs(sigma, 0) == 0
    # The true law has sigma=pi/(2^20-1), not zero damping or a tunable clock.
    loss_lower = Q(3, 2**20)
    assert pi.lo / (2**20 - 1) > loss_lower
    accumulated_residual_lower = Q(damping_prefactor) * loss_lower
    assert accumulated_residual_lower > Q(1, 400)
    exponential_lower = 1 + exponent_lower + exponent_lower**2 / 2
    assert exponential_lower / 400 == Q(1201, 400)
    assert exponential_lower / 400 > radius_upper
