"""Static full-law and interval controls for prepared finite transport."""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q

import numpy as np
import pytest
import sympy as sp

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I, pi_interval, sin
from tnfr.physics import _localized_identity_transport as owner
from tnfr.physics.relational_sine_comparison import _sine_rates


@pytest.fixture(scope="module", autouse=True)
def no_source_or_response_evaluation():
    with forbid_organized_response_evaluation():
        yield


def _model(**changes):
    return RelationalExchangeModel(
        **(
            dict(
                storage_scale=1,
                epi_weight=Q(1, 2**20),
                phase_weight=1 - Q(1, 2**20),
                phase_domain="regular",
            )
            | changes
        )
    )


def _bound(**changes):
    return owner._bound_localized_identity_transport(
        **(
            dict(
                reference_model=_model(),
                node_count=1024,
                fejer_order=16,
                amplitude=Q(1, 2**12),
                initial_state_error_bound=Q(1, 1000 * 2**12),
                localization_radius=96,
                horizon=256,
            )
            | changes
        )
    )


def _cycle_laplacian(size):
    matrix = sp.zeros(size)
    for left in range(size):
        right = (left + 1) % size
        for row, column, sign in (
            (left, left, 1),
            (right, right, 1),
            (left, right, -1),
            (right, left, -1),
        ):
            matrix[row, column] += sign
    return matrix


def _near_zero(interval):
    assert interval.contains(0)
    assert interval.abs_max < Q(1, 10**32)


def test_actual_complete_rows_and_clock_give_the_coupled_cycle_tangent():
    size = 8
    form = tuple(Q(j * j - 7, 11) for j in range(size))
    phase = tuple(Q((-1) ** j * (j + 1), 13) for j in range(size))
    gradient, phase_current = [Q(0)] * size, [Q(0)] * size
    for left in range(size):
        right = (left + 1) % size
        difference, phase_difference = (
            form[left] - form[right],
            phase[right] - phase[left],
        )
        gradient[left] += difference
        gradient[right] -= difference
        phase_current[left] += phase_difference
        phase_current[right] -= phase_difference
    model = _model()
    loss, exchange = Q(1, 2**20), 1 - Q(1, 2**20)
    perturbation = sp.Symbol("perturbation", real=True)
    for j in range(size):
        sine_current = sum(
            sp.sin(perturbation * (phase[other] - phase[j]))
            for other in ((j - 1) % size, (j + 1) % size)
        )
        assert (
            sp.diff(sine_current, perturbation).subs(perturbation, 0)
            == phase_current[j]
        )
    rates = _sine_rates(model, (2,) * size, gradient, (Q(1),) * size, phase_current)
    conversion = pi_interval() / exchange
    for j in range(size):
        expected_form = -pi_interval() * loss * gradient[j] / (2 * exchange)
        expected_form += Q(phase_current[j], 2)
        _near_zero(rates["form_rates"][j] * conversion - expected_form)
        _near_zero(rates["phase_rates"][j] * conversion - Q(gradient[j], 2))
    assert any(gradient)
    assert any(not rate.contains(0) for rate in rates["phase_rates"])
    assert sum(gradient) == sum(phase_current) == 0
    # Changing the clock in only the form row fails the complete model.
    assert not (rates["phase_rates"][0] - Q(gradient[0], 2)).contains(0)


def test_joint_tangent_is_contractive_and_missing_phase_feedback_breaks_it():
    size = 4
    laplacian = _cycle_laplacian(size)
    a = laplacian / 2
    damping = sp.Symbol("d", positive=True)
    x = sp.Matrix(sp.symbols("x:4", real=True))
    theta = sp.Matrix(sp.symbols("theta:4", real=True))
    form_rate, phase_rate = -damping * a * x - a * theta, a * x
    derivative = (x.T * form_rate + theta.T * phase_rate)[0]
    loss = damping * sum((x[j] - x[(j + 1) % size]) ** 2 for j in range(size)) / 2
    assert sp.expand(derivative + loss) == 0
    assert sp.expand((x.T * form_rate)[0] + loss) != 0
    # Ring translations preserve every complete linear row, not only its spectrum.
    translation = sp.zeros(size)
    for j in range(size):
        translation[j, (j + 1) % size] = 1
    assert translation * laplacian == laplacian * translation


def test_nonlinear_rows_preserve_integer_translation_reflection_and_storage_balance():
    size = 8
    form = tuple(Q(j - 3, 17) for j in range(size))
    phase = tuple(Q((-1) ** j * (j + 1), 31) for j in range(size))
    model = _model()

    def evaluate(forms, phases):
        gradient = tuple(
            2 * forms[j] - forms[(j - 1) % size] - forms[(j + 1) % size]
            for j in range(size)
        )
        current = tuple(
            sin(phases[(j - 1) % size] - phases[j])
            + sin(phases[(j + 1) % size] - phases[j])
            for j in range(size)
        )
        rates = _sine_rates(model, (2,) * size, gradient, (Q(1),) * size, current)
        return gradient, current, rates

    gradient, current, rates = evaluate(form, phase)
    for permutation in (
        tuple((j + 1) % size for j in range(size)),
        tuple(-j % size for j in range(size)),
    ):
        _, _, transformed = evaluate(
            tuple(form[j] for j in permutation), tuple(phase[j] for j in permutation)
        )
        for channel in ("form_rates", "phase_rates"):
            for j, original in enumerate(permutation):
                _near_zero(transformed[channel][j] - rates[channel][original])
    storage_rate = sum(
        (
            gradient[j] * rates["form_rates"][j] - current[j] * rates["phase_rates"][j]
            for j in range(size)
        ),
        I(0),
    )
    loss = Q(1, 2**21) * sum(value**2 for value in gradient)
    _near_zero(storage_rate + loss)
    assert loss > 0
    assert not storage_rate.contains(0)


def test_exact_cycle_fourier_modes_normalize_the_whole_form_phase_packet():
    size, order, carrier = 8, 2, 2
    root = (1 + sp.I) / sp.sqrt(2)
    laplacian = _cycle_laplacian(size)
    modes = range(-order + 1, order)
    coefficients = {m: sp.Rational(order - abs(m), order) for m in modes}
    packet = sp.zeros(size, 1)
    for m, coefficient in coefficients.items():
        mode = sp.Matrix([sp.expand(root ** ((carrier + m) * j)) for j in range(size)])
        frequency = 1 - sp.cos(2 * sp.pi * (carrier + m) / size)
        assert all(
            sp.simplify(value) == 0 for value in laplacian * mode / 2 - frequency * mode
        )
        assert all(sp.simplify(value * sp.conjugate(value)) == 1 for value in mode)
        packet += coefficient * mode
    norm_squared = sp.simplify(
        sum(sp.expand_complex(value * sp.conjugate(value)) for value in packet)
    )
    coefficient_square_sum = sum(value**2 for value in coefficients.values())
    assert norm_squared == size * coefficient_square_sum
    assert sp.simplify(sum(packet)) == 0
    assert any(sp.simplify(sp.re(value)) != 0 for value in packet)
    assert any(sp.simplify(sp.im(value)) != 0 for value in packet)
    # This coupled tangent uses lambda(L/2), not sqrt(lambda) of an auxiliary wave.
    frequency = 1 - sp.cos(sp.pi / 4)
    assert sp.simplify(frequency**2 - frequency) != 0


def test_a_single_mode_cannot_pass_the_localized_mass_requirement():
    node_count, radius = 1024, 96
    distances = tuple(min(j, node_count - j) for j in range(node_count))
    window_count = sum(distance <= radius for distance in distances)
    # Every site has equal |x+i*theta|^2 in the single-mode negative control.
    inside_fraction = Q(window_count, node_count)
    assert inside_fraction < Q(24, 25)
    assert 1 - inside_fraction > _bound().tail_tolerance


def test_dispersion_orientation_retains_carrier_rotation_and_cubic_defect():
    k, q, j, time = sp.symbols("k q j time", real=True)
    frequency = 1 - sp.cos(k)
    assert sp.diff(frequency, k).subs(k, sp.pi / 2) == 1
    assert sp.diff(frequency, k).subs(k, -sp.pi / 2) == -1
    assert sp.diff(frequency, k, 2).subs(k, sp.pi / 2) == 0
    assert sp.simplify(frequency.subs(k, sp.pi / 2 + q) - 1 - q) == sp.sin(q) - q
    translated_phase = (sp.pi / 2 + q) * (j + time)
    carrier_corrected = translated_phase + (1 - sp.pi / 2) * time
    assert sp.expand(carrier_corrected - ((sp.pi / 2 + q) * j + (1 + q) * time)) == 0
    assert sp.diff(translated_phase - carrier_corrected, time) != 0
    # The envelope coordinate j+time places its center at -time.
    assert sp.solve(j + time, j) == [-time]


def test_cubic_full_law_remainder_and_bootstrap_constants_follow_from_edges():
    u, s = sp.symbols("u s", nonnegative=True)
    taylor_remainder = -sp.integrate((u - s) ** 2 * sp.cos(s) / 2, (s, 0, u))
    assert sp.simplify(taylor_remainder - (sp.sin(u) - u)) == 0
    assert sp.integrate((u - s) ** 2 / 2, (s, 0, u)) == u**3 / 6
    theta = sp.Matrix(sp.symbols("theta:4", real=True))
    laplacian = _cycle_laplacian(4)
    spectral_slack = (theta.T * (4 * sp.eye(4) - laplacian) * theta)[0]
    assert (
        sp.expand(
            spectral_slack - sum((theta[j] + theta[(j + 1) % 4]) ** 2 for j in range(4))
        )
        == 0
    )
    # ||D^T||<=2, the row has factor 1/2, and ||D theta||<=2||theta||.
    remainder_constant = Q(2, 2) * Q(1, 6) * 2**3
    amplitude, horizon = sp.symbols("A T", positive=True)
    integral = sp.integrate(
        sp.Rational(remainder_constant) * (2 * amplitude) ** 3, (s, 0, horizon)
    )
    assert integral / amplitude == sp.Rational(32, 3) * horizon * amplitude**2
    assert remainder_constant * 2**3 > remainder_constant  # the 2A bootstrap matters


def test_fejer_norm_and_two_sided_tail_bound_are_independently_derived():
    order = sp.Symbol("Q", integer=True, positive=True)
    m = sp.Symbol("m", integer=True)
    coefficient_sum = 1 + 2 * sp.summation((1 - m / order) ** 2, (m, 1, order - 1))
    assert sp.simplify(coefficient_sum - (2 * order**2 + 1) / (3 * order)) == 0
    assert sp.simplify(coefficient_sum - 2 * order / 3) == 1 / (3 * order)
    radius, n, distance = sp.symbols("R N d", positive=True)
    # Fejer numerator <=1 and sin(pi*d/N)>=2d/N yield this nodal mass bound.
    nodal_majorant = n**3 / (16 * order**2 * (2 * order / 3) * distance**4)
    tail = 2 * sp.integrate(nodal_majorant, (distance, radius - 1, sp.oo))
    assert sp.simplify(tail - n**3 / (16 * order**3 * (radius - 1) ** 3)) == 0
    assert sp.simplify(tail - tail / 2) != 0  # omitting one spatial tail is invalid


def test_fixed_prospective_budget_closes_without_evaluating_any_response():
    report = _bound()
    n, order, amplitude, time = 1024, 16, Q(1, 2**12), 256
    loss, exchange = Q(1, 2**20), 1 - Q(1, 2**20)
    pi = pi_interval()
    damping = 2 * pi.hi * loss * time / exchange
    bandwidth = 2 * pi.hi * Q(order - 1, n)
    dispersion = time * bandwidth**3 / 6
    nonlinear = Q(4, 3) * time * (2 * amplitude) ** 3 / amplitude
    total = Q(1, 1000) + damping + dispersion + nonlinear
    assert report.relative_identity_error_upper_bound == total < Q(1, 25)
    assert report.relative_damping_error_upper_bound == damping
    assert report.relative_dispersion_error_upper_bound == dispersion
    assert report.relative_nonlinear_error_upper_bound == nonlinear
    assert report.fejer_coefficient_square_sum == sum(
        Q(order - abs(m), order) ** 2 for m in range(1 - order, order)
    )
    tail = Q(n**3, 16 * order**3 * 95**3)
    assert report.reference_tail_fraction_upper_bound == tail < Q(1, 49)
    coarse_actual_tail = ((Q(1, 7) + Q(1, 25)) / (1 - Q(1, 25))) ** 2
    assert report.actual_tail_fraction_upper_bound < coarse_actual_tail < Q(1, 25)
    assert report.state_norm_lower_bound > Q(24, 25) * amplitude
    assert report.state_norm_upper_bound < Q(26, 25) * amplitude
    assert report.endpoint_displacement == -time
    assert abs(report.endpoint_displacement) > 2 * report.localization_radius
    assert report.original_horizon_bounds.contains(time * pi.midpoint / exchange)
    assert report.status == "certified"
    assert not report.unavailable_reasons


def test_failure_is_unavailable_and_keeps_the_independent_obligations():
    nominal = _bound(initial_state_error_bound=0)
    remaining = nominal.relative_identity_error_upper_bound
    at_bootstrap = _bound(initial_state_error_bound=nominal.amplitude * (1 - remaining))
    assert at_bootstrap.bootstrap_margin == 0
    assert not at_bootstrap.bootstrap_certified
    assert at_bootstrap.actual_tail_fraction_upper_bound is None
    assert at_bootstrap.state_norm_lower_bound is None
    assert at_bootstrap.state_norm_upper_bound is None
    at_identity = _bound(
        initial_state_error_bound=nominal.amplitude * (Q(1, 25) - remaining)
    )
    assert at_identity.bootstrap_certified
    assert not at_identity.identity_tolerance_met
    assert at_identity.relative_identity_error_upper_bound == Q(1, 25)
    wide = _bound(localization_radius=200)
    assert wide.identity_tolerance_met and wide.tail_tolerance_met
    assert not wide.endpoint_windows_disjoint
    assert wide.status == "not_certified"
    narrow = _bound(localization_radius=2)
    assert narrow.identity_tolerance_met and narrow.endpoint_windows_disjoint
    assert not narrow.tail_tolerance_met


def test_exact_tiny_primitive_state_errors_and_amplitude_are_retained():
    tiny = Q(1, 10**500)
    report = _bound(amplitude=tiny, initial_state_error_bound=tiny / 1000)
    assert report.amplitude == tiny
    assert report.initial_state_error_bound == tiny / 1000
    assert report.relative_initial_error == Q(1, 1000)
    assert report.state_norm_lower_bound > 0
    with pytest.raises(FrozenInstanceError):
        report.amplitude = 1


@pytest.mark.parametrize("parameter", ["node_count", "fejer_order", "horizon"])
@pytest.mark.parametrize("bad", [True, Q(16), 16.0, np.int64(16), "16", None])
def test_discrete_primitives_require_declared_ordinary_integer_admission(
    parameter, bad
):
    with pytest.raises(TypeError, match="ordinary Python integer"):
        _bound(**{parameter: bad})


@pytest.mark.parametrize(
    "parameter", ["amplitude", "initial_state_error_bound", "localization_radius"]
)
@pytest.mark.parametrize("bad", [True, float("nan"), float("inf"), "0.1", None, 1j])
def test_real_primitive_admission_precedes_arithmetic(parameter, bad):
    with pytest.raises((TypeError, ValueError)):
        _bound(**{parameter: bad})


@pytest.mark.parametrize(
    "changes",
    [
        {"node_count": 4},
        {"node_count": 1023},
        {"fejer_order": 1},
        {"fejer_order": 257},
        {"horizon": 0},
        {"horizon": 513},
        {"amplitude": 0},
        {"initial_state_error_bound": -Q(1, 10**500)},
        {"localization_radius": 1},
        {"localization_radius": 512},
    ],
)
def test_unsupported_analytic_domains_reject(changes):
    with pytest.raises(ValueError):
        _bound(**changes)


@pytest.mark.parametrize(
    "changes", [{"storage_scale": 2}, {"epi_weight": 0}, {"phase_domain": "acute"}]
)
def test_different_model_premises_do_not_inherit_the_certificate(changes):
    with pytest.raises(ValueError):
        _bound(reference_model=_model(**changes))


def test_actual_model_fields_are_readmitted_without_silent_renormalization():
    forged = _model()
    object.__setattr__(forged, "epi_weight", True)
    with pytest.raises(TypeError):
        _bound(reference_model=forged)
    forged = _model()
    object.__setattr__(forged, "epi_weight", Q(1, 10**500))
    report = _bound(reference_model=forged)
    assert report.form_loss_coefficient == Q(1, 10**500)
    assert report.exchange_coefficient == 1 - Q(1, 2**20)
    assert report.relative_damping_error_upper_bound > 0
    with pytest.raises((TypeError, ValueError)):
        _bound(reference_model=replace(report, status="certified"))


def test_tiny_positive_clock_retains_exact_law_and_a_finite_positive_horizon():
    exchange = Q(1, 10**500)
    model = _model()
    object.__setattr__(model, "phase_weight", exchange)
    object.__setattr__(model, "epi_weight", exchange / (2**20 - 1))

    report = _bound(reference_model=model)

    assert report.exchange_coefficient == exchange > 0
    assert report.form_loss_coefficient == exchange / (2**20 - 1)
    assert (
        report.relative_damping_error_upper_bound
        == _bound().relative_damping_error_upper_bound
    )
    assert report.status == "certified"
    pi = pi_interval()
    assert report.clock_rate_bounds.contains(exchange / pi.lo)
    assert report.clock_rate_bounds.contains(exchange / pi.hi)
    # Zero here is only the outward grid resolution, not an admitted zero clock.
    assert report.clock_rate_bounds.lo == 0 < report.clock_rate_bounds.hi
    assert report.original_horizon_bounds.lo > 0
    assert report.original_horizon_bounds.contains(report.horizon * pi.lo / exchange)
    assert report.original_horizon_bounds.contains(report.horizon * pi.hi / exchange)


def test_tiny_exchange_with_unscaled_loss_fails_the_budget_without_arithmetic_failure():
    model = _model()
    exchange = Q(1, 10**500)
    object.__setattr__(model, "phase_weight", exchange)

    report = _bound(reference_model=model)

    assert report.exchange_coefficient == exchange
    assert report.status == "not_certified"
    assert not report.bootstrap_certified
    assert report.relative_damping_error_upper_bound > 1
    assert report.original_horizon_bounds.lo > 0
    assert report.actual_tail_fraction_upper_bound is None
