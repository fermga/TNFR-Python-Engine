"""Independent static controls for the weighted, oriented C9 interface."""

from fractions import Fraction as Q

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.physics import _oriented_interface_constitutive as owner


@pytest.fixture(scope="module", autouse=True)
def no_response_or_frozen_execution():
    with forbid_sine_regeneration():
        yield


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 120
    return context


def _mp(mp, value):
    return mp.mpf(value.numerator) / value.denominator


def _contains(mp, interval, value):
    assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)
    assert _mp(mp, interval.width) < mp.mpf("1e-28")


def _potential(mp, angle, deformation, frequency=9):
    return (
        1
        - mp.cos(angle)
        + deformation * mp.cos(angle) * (1 - mp.cos(frequency * angle)) ** 2
    )


def _current(mp, angle, deformation, frequency=9):
    high = 1 - mp.cos(frequency * angle)
    return mp.sin(angle) + deformation * (
        -mp.sin(angle) * high**2
        + 2 * frequency * mp.cos(angle) * high * mp.sin(frequency * angle)
    )


def _weighted_support(mixing):
    cycles = tuple(
        (9 * component + local, 9 * component + (local + 1) % 9, Q(1))
        for component in range(2)
        for local in range(9)
    )
    cross = tuple(
        (local, 9 + (local + offset) % 9, weight)
        for local in range(9)
        for offset, weight in ((0, 1 - mixing), (1, mixing))
        if weight
    )
    return cycles + cross, cross


@pytest.mark.parametrize("mixing", (Q(0), Q(1, 3), Q(1, 2), Q(1)))
@pytest.mark.parametrize("deformation", (Q(-1, 324), Q(0), Q(1, 324)))
def test_weighted_fine_rows_storage_loss_and_registered_chart(mixing, deformation, mp):
    forms, phasor = (Q(3, 7), Q(-2, 5)), (Q(3, 5), Q(4, 5))
    report = owner._observe_oriented_constitutive_pair(
        mixing=mixing,
        deformation=deformation,
        form=forms,
        relative_phase_phasor=phasor,
    )
    geometry = report.geometry
    edges, cross = _weighted_support(mixing)
    assert sorted(geometry.weighted_edges) == sorted(edges)
    degrees = tuple(
        sum(
            (weight for first, second, weight in edges if node in (first, second)), Q(0)
        )
        for node in range(18)
    )
    assert geometry.weighted_degrees == degrees == (Q(3),) * 18
    assert geometry.support_degrees == (3 if mixing in (0, 1) else 4,) * 18
    assert geometry.clock_factor == Q(1, 3)
    assert report.law.deformation == deformation
    assert report.form == forms
    assert report.relative_phase_phasor == phasor

    alpha, delta, origin = 2 * mp.pi / 9, mp.atan2(4, 3), mp.pi / 17
    s, e, gamma = _mp(mp, mixing), _mp(mp, deformation), 1 / (1023 * mp.pi)
    c, d = 1 - s + s * mp.cos(alpha), s * mp.sin(alpha)
    rho = mp.mpf(1) if mixing in (0, 1) else mp.sqrt(c**2 + d**2)
    beta = mp.atan2(d, c)
    scale, exchange = mp.sqrt(rho), gamma * mp.sqrt(rho)
    for interval, expected in (
        (geometry.real_multiplier_bounds, c),
        (geometry.imaginary_multiplier_bounds, d),
        (geometry.amplitude_bounds, rho),
        (geometry.phase_origin_shift_bounds, beta),
        (geometry.form_scale_bounds, scale),
        (geometry.gamma_bounds, gamma),
        (geometry.effective_gamma_bounds, exchange),
        (geometry.internal_storage_bounds, 18 * (1 - mp.cos(alpha))),
        (geometry.aligned_attachment_work_bounds, 9 * s * (1 - mp.cos(alpha))),
    ):
        _contains(mp, interval, expected)
    for interval, expected in zip(
        geometry.phase_origin_phasor_bounds, (mp.cos(beta), mp.sin(beta))
    ):
        _contains(mp, interval, expected)

    values = tuple(forms[node // 9] for node in range(18))
    phases = tuple(
        origin + (node // 9) * delta + alpha * (node % 9 - 4) for node in range(18)
    )
    gradient, current = [Q(0)] * 18, [mp.mpf(0)] * 18
    energy, contact_work = mp.mpf(0), mp.mpf(0)
    for first, second, weight in edges:
        difference = values[first] - values[second]
        gradient[first] += weight * difference
        gradient[second] -= weight * difference
        gap = phases[second] - phases[first]
        flow = _mp(mp, weight) * _current(mp, gap, e)
        current[first] += flow
        current[second] -= flow
        edge_energy = _mp(mp, weight) * (
            _mp(mp, difference**2 / 2) + _potential(mp, gap, e)
        )
        energy += edge_energy
        if (first, second, weight) in cross:
            contact_work += edge_energy
    rate_from_rows = mp.mpf(0)
    for node, (grad, flow) in enumerate(zip(gradient, current)):
        form_rate = (-_mp(mp, grad) + gamma * flow) / 3
        phase_rate = gamma * _mp(mp, grad) / 3
        _contains(mp, report.form_rate_bounds[node // 9], form_rate)
        _contains(mp, report.phase_rate_bounds[node // 9], phase_rate)
        rate_from_rows += _mp(mp, grad) * form_rate - flow * phase_rate
    expected_loss = sum((grad**2 / 3 for grad in gradient), Q(0))
    assert report.loss == expected_loss == 6 * (forms[1] - forms[0]) ** 2
    assert report.storage_rate == -expected_loss
    _contains(mp, report.storage_rate_from_rows_bounds, rate_from_rows)
    _contains(mp, report.full_storage_bounds, energy)
    _contains(mp, report.attachment_work_bounds, contact_work)

    inherited_current = (1 - s) * _current(mp, delta, e) + s * _current(
        mp, delta + alpha, e
    )
    inherited_potential = (1 - s) * _potential(mp, delta, e) + s * _potential(
        mp, delta + alpha, e
    )
    _contains(mp, report.current_bounds, inherited_current)
    _contains(mp, report.contact_potential_bounds, inherited_potential)
    _contains(mp, report.chart_current_bounds, inherited_current / rho)
    _contains(mp, report.chart_contact_potential_bounds, inherited_potential / rho)
    for interval, expected in zip(
        report.chart_relative_phase_phasor_bounds,
        (mp.cos(delta + beta), mp.sin(delta + beta)),
    ):
        _contains(mp, interval, expected)
    z = tuple(_mp(mp, value) / scale for value in forms)
    dz = z[1] - z[0]
    chart_rates = (dz + exchange * inherited_current / rho, -exchange * dz)
    for component, sign in ((0, 1), (1, -1)):
        _contains(mp, report.chart_form_bounds[component], z[component])
        _contains(mp, report.chart_form_rate_bounds[component], sign * chart_rates[0])
        _contains(mp, report.chart_phase_rate_bounds[component], sign * chart_rates[1])
        # The chart does not redefine the original form measurement.
        assert mp.almosteq(scale * z[component], _mp(mp, forms[component]))
    assert mp.almosteq(
        energy,
        18 * (1 - mp.cos(alpha)) + 9 * rho * (dz**2 / 2 + inherited_potential / rho),
    )
    assert mp.almosteq(_mp(mp, report.loss) * 3, 18 * rho * dz**2)
    expected_profile = deformation == 0 or mixing in (0, 1)
    assert report.profile_preserved is expected_profile
    assert report.profile_preserved_over_open_interfaces is (deformation == 0)


@pytest.mark.parametrize("mixing", (Q(1, 7), Q(1, 2), Q(6, 7)))
def test_oriented_profile_coefficients_keep_both_quadratures(mixing, mp):
    report = owner._observe_oriented_constitutive_pair(
        mixing=mixing,
        deformation=Q(1, 324),
        form=(0, 0),
        relative_phase_phasor=(0, 1),
    )
    alpha, s = 2 * mp.pi / 9, _mp(mp, mixing)
    c, d = 1 - s + s * mp.cos(alpha), s * mp.sin(alpha)
    rho, beta = mp.sqrt(c**2 + d**2), mp.atan2(d, c)
    coefficients = dict(report.law.current_coefficients)
    assert len(report.inherited_harmonic_coefficients) == len(coefficients)
    for n, sine_bound, cosine_bound in report.inherited_harmonic_coefficients:
        b, sign = _mp(mp, coefficients[n]), 1 if n % 9 == 1 else -1
        sine = b * (c * mp.cos(n * beta) + sign * d * mp.sin(n * beta)) / rho
        cosine = b * (-c * mp.sin(n * beta) + sign * d * mp.cos(n * beta)) / rho
        _contains(mp, sine_bound, sine)
        _contains(mp, cosine_bound, cosine)
    # Frequencies 1 and 8 would demand exp(i*9*beta)=1. The actual branch
    # obeys 0<9*beta<2*pi at every strictly interior interface.
    assert 0 < 9 * beta < 2 * mp.pi
    assert abs(mp.exp(9j * beta) - 1) > mp.mpf("0.1")
    assert report.profile_preserved is False


def test_midpoint_alone_does_not_select_sine(mp):
    """A distinct admitted family survives the midpoint but not an interval."""
    alpha, beta, rho = 2 * mp.pi / 9, mp.pi / 9, mp.cos(mp.pi / 9)
    e, frequency = mp.mpf(1) / 1296, 18
    assert 2 * frequency**2 * e == mp.mpf("0.5")
    for k in (0, 1, 2):
        angle = k * alpha
        for derivative in range(4):
            perturbation = mp.diff(
                lambda value: mp.cos(value) * (1 - mp.cos(18 * value)) ** 2,
                angle,
                derivative,
            )
            assert abs(perturbation) < mp.mpf("1e-105")
    for angle in (mp.mpf("0.13"), mp.mpf("0.71"), mp.pi / 2):
        averaged = (
            _current(mp, angle, e, frequency)
            + _current(mp, angle + alpha, e, frequency)
        ) / 2
        inherited = rho * _current(mp, angle + beta, e, frequency)
        assert abs(averaged - inherited) < mp.mpf("1e-110")
        averaged_storage = (
            _potential(mp, angle, e, frequency)
            + _potential(mp, angle + alpha, e, frequency)
        ) / 2
        assert abs(
            averaged_storage
            - (1 - rho + rho * _potential(mp, angle + beta, e, frequency))
        ) < mp.mpf("1e-110")
    # This is a genuinely nonsine current, despite matching all target jets.
    assert abs(_current(mp, mp.pi / 2, e, frequency) - 1) > mp.mpf("0.001")


@pytest.mark.parametrize("deformation", (Q(-1, 324), Q(0), Q(1, 324)))
def test_aligned_attachment_has_nonzero_current_and_retained_work(deformation, mp):
    report = owner._observe_oriented_constitutive_pair(
        mixing=Q(1, 2),
        deformation=deformation,
        form=(0, 0),
        relative_phase_phasor=(1, 0),
    )
    alpha, gamma = 2 * mp.pi / 9, 1 / (1023 * mp.pi)
    _contains(mp, report.current_bounds, mp.sin(alpha) / 2)
    _contains(mp, report.form_rate_bounds[0], gamma * mp.sin(alpha) / 6)
    assert report.form_rate_bounds[0].lo > 0
    assert report.form_rate_bounds[1].hi < 0
    assert report.phase_rate_bounds[0].contains(0)
    assert report.phase_rate_bounds[1].contains(0)
    _contains(mp, report.attachment_work_bounds, 9 * (1 - mp.cos(alpha)) / 2)
    assert report.attachment_work_bounds.lo > 0
    assert report.chart_contact_potential_bounds.lo > 0
    assert report.loss == 0
    # Zero instantaneous loss does not make the newly attached state an equilibrium.


@pytest.mark.parametrize("deformation", (Q(-1, 324), Q(0), Q(1, 324)))
def test_source_work_transforms_with_both_rows_and_full_potential(deformation, mp):
    mixing, forms = Q(2, 5), (Q(3, 7), Q(-2, 5))
    form_input, phase_input = (Q(2, 7), Q(-1, 5)), (Q(-3, 8), Q(1, 9))
    report = owner._observe_oriented_constitutive_pair(
        mixing=mixing,
        deformation=deformation,
        form=forms,
        relative_phase_phasor=(Q(3, 5), Q(4, 5)),
    )
    alpha, delta = 2 * mp.pi / 9, mp.atan2(4, 3)
    e, s = _mp(mp, deformation), _mp(mp, mixing)
    edges, _ = _weighted_support(mixing)
    fine_source_work = mp.mpf(0)
    for first, second, weight in edges:
        left, right = first // 9, second // 9
        gap = (right - left) * delta + (second % 9 - first % 9) * alpha
        fine_source_work += _mp(mp, weight) * (
            _mp(
                mp,
                (forms[left] - forms[right]) * (form_input[left] - form_input[right]),
            )
            + _current(mp, gap, e) * _mp(mp, phase_input[right] - phase_input[left])
        )
    u = forms[1] - forms[0]
    current = (1 - s) * _current(mp, delta, e) + s * _current(mp, delta + alpha, e)
    expected_work = 9 * (
        _mp(mp, u * (form_input[1] - form_input[0]))
        + current * _mp(mp, phase_input[1] - phase_input[0])
    )
    assert abs(fine_source_work - expected_work) < mp.mpf("1e-110")
    raw_dx = tuple(rate + f for rate, f in zip(report.form_rate_bounds, form_input))
    raw_dphase = tuple(
        rate + g for rate, g in zip(report.phase_rate_bounds, phase_input)
    )
    raw_chain_rule = 9 * (
        u * (raw_dx[1] - raw_dx[0])
        + report.current_bounds * (raw_dphase[1] - raw_dphase[0])
    )
    expected_rate = -6 * _mp(mp, u**2) + expected_work
    _contains(mp, raw_chain_rule, expected_rate)

    geometry = report.geometry
    chart_dx = tuple(
        rate + 3 * f / geometry.form_scale_bounds
        for rate, f in zip(report.chart_form_rate_bounds, form_input)
    )
    chart_dphase = tuple(
        rate + 3 * g for rate, g in zip(report.chart_phase_rate_bounds, phase_input)
    )
    chart_z = report.chart_form_bounds[1] - report.chart_form_bounds[0]
    chart_chain_rule = (
        9
        * geometry.amplitude_bounds
        * (
            chart_z * (chart_dx[1] - chart_dx[0])
            + report.chart_current_bounds * (chart_dphase[1] - chart_dphase[0])
        )
    )
    _contains(mp, chart_chain_rule, 3 * expected_rate)
    # The form source and phase source scale differently; the clock acts on both.


@pytest.mark.parametrize("mixing", (Q(1, 10**80), 1 - Q(1, 10**80)))
def test_nonzero_tiny_contact_weights_keep_the_actual_support(mixing, mp):
    report = owner._observe_oriented_constitutive_pair(
        mixing=mixing,
        deformation=Q(1, 324),
        form=(0, 0),
        relative_phase_phasor=(1, 0),
    )
    assert report.geometry.mixing == mixing
    assert len(report.geometry.weighted_edges) == 36
    assert report.geometry.support_degrees == (4,) * 18
    assert report.geometry.weighted_degrees == (Q(3),) * 18
    assert report.profile_preserved is False
    expected = 9 * _mp(mp, mixing) * (1 - mp.cos(2 * mp.pi / 9))
    _contains(mp, report.geometry.aligned_attachment_work_bounds, expected)


@pytest.mark.parametrize(
    "bad", (True, "0.5", float("nan"), float("inf"), Q(-1, 10**80), 1 + Q(1, 10**80))
)
def test_mixing_rejects_invalid_and_exactly_out_of_domain_values(bad):
    with pytest.raises((TypeError, ValueError)):
        owner._derive_oriented_interface(mixing=bad)


@pytest.mark.parametrize(
    "change",
    (
        {"deformation": True},
        {"deformation": Q(1, 324) + Q(1, 10**80)},
        {"form": (0,)},
        {"form": (0, True)},
        {"form": (0, float("inf"))},
        {"relative_phase_phasor": (1, 1)},
        {"relative_phase_phasor": (True, 0)},
    ),
)
def test_observation_re_admits_every_primitive(change):
    inputs = dict(
        mixing=Q(1, 2), deformation=0, form=(0, 0), relative_phase_phasor=(1, 0)
    )
    with pytest.raises((TypeError, ValueError)):
        owner._observe_oriented_constitutive_pair(**(inputs | change))


def test_report_or_trajectory_controls_do_not_replace_primitives():
    for extra in ({"geometry": object()}, {"source": object()}, {"horizon": 1}):
        with pytest.raises(TypeError):
            owner._observe_oriented_constitutive_pair(
                mixing=Q(1, 2),
                deformation=0,
                form=(0, 0),
                relative_phase_phasor=(1, 0),
                **extra,
            )
