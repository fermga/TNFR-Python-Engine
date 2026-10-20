"""Independent static controls for the fixed acquired-parent intervention."""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.physics import _organized_parent_restoring_response as owner
from tnfr.physics import _repeated_organized_composition as composition
from tnfr.physics._sine_preparation import _prepared_heat_error_radii

_ERROR = Q(1, 10**10)
_RECOVERED = Q(1, 10**70)


@pytest.fixture(scope="module", autouse=True)
def no_scientific_regeneration():
    with forbid_sine_regeneration():
        yield


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 240
    return context


def _mp(mp, value):
    return mp.mpf(value.numerator) / value.denominator


def _contains(mp, interval, expected):
    assert _mp(mp, interval.lo) <= expected <= _mp(mp, interval.hi)
    assert _mp(mp, interval.width) < mp.mpf("1e-24") * max(1, abs(expected))


def _assess(**changes):
    return owner._assess_organized_parent_restoring_response(
        **(
            dict(form_error_bounds=(_ERROR,) * 4, phase_error_bounds=(_ERROR,) * 4)
            | changes
        )
    )


@pytest.fixture(scope="module")
def response():
    return _assess()


def _means(values, size):
    return tuple(
        sum(values[start : start + size]) / size for start in range(0, 36, size)
    )


def test_fine_intervention_descends_to_eight_means_and_parent_identity_map():
    forms = tuple(
        Q(g + a + 1, 20) + Q(j - 4, 10000)
        for g in range(2)
        for a in range(2)
        for j in range(9)
    )
    phases = tuple(
        Q(2 * g - 1, 2000)
        + Q(2 * a - 1, 200)
        + Q((g + 1) * (a + 2), 100000) * (j * j - Q(68, 3))
        for g in range(2)
        for a in range(2)
        for j in range(9)
    )
    after_x, after_theta = owner._apply_parent_registration_intervention(
        form=forms, phase_offsets=phases
    )
    assert after_x == forms
    assert _means(after_theta, 18) == _means(phases, 18)
    assert _means(after_x, 18) == _means(forms, 18)
    assert tuple(a - b for a, b in zip(_means(after_theta, 9), _means(phases, 9))) == (
        -Q(1, 4),
        Q(1, 4),
        Q(0),
        Q(0),
    )
    # Every internal circular edge, including the cycle seam, is unchanged.
    for start in range(0, 36, 9):
        for j in range(9):
            left, right = start + j, start + (j + 1) % 9
            assert (
                after_theta[right] - after_theta[left] == phases[right] - phases[left]
            )
    before = composition._observe_repeated_composition(
        interface="matching", form=forms, phase_offsets=phases
    )
    after = composition._observe_repeated_composition(
        interface="matching", form=after_x, phase_offsets=after_theta
    )
    assert before.parent_form_means == after.parent_form_means
    assert before.parent_phase_offset_means == after.parent_phase_offset_means
    assert before.parent_phase_rate_bounds == after.parent_phase_rate_bounds
    assert before.parent_form_rate_bounds != after.parent_form_rate_bounds


@pytest.mark.parametrize("field", ("form", "phase_offsets"))
@pytest.mark.parametrize("bad", (True, "0", float("nan"), float("inf")))
def test_intervention_admits_each_fine_primitive(field, bad):
    state = dict(form=(Q(0),) * 36, phase_offsets=(Q(0),) * 36)
    values = list(state[field])
    values[-1] = bad
    state[field] = values
    with pytest.raises((TypeError, ValueError)):
        owner._apply_parent_registration_intervention(**state)


@pytest.mark.parametrize("field", ("form", "phase_offsets"))
@pytest.mark.parametrize("bad", ((), (0,) * 35, (0,) * 37, "0" * 36, {0}))
def test_intervention_requires_both_ordered_fine_vectors(field, bad):
    state = dict(form=(Q(0),) * 36, phase_offsets=(Q(0),) * 36)
    state[field] = bad
    with pytest.raises((TypeError, ValueError)):
        owner._apply_parent_registration_intervention(**state)


def test_mixed_internal_mode_matches_all_six_relative_rows_and_clock(mp):
    # Nonzero forms and the mixed binary mode distinguish this state from
    # the two separated pendula used before the intervention.
    u, v, w = Q(1, 7000), -Q(1, 8000), Q(1, 9000)
    delta, inner, mixed = Q(1, 1000), Q(13, 50), Q(1, 4)
    forms = tuple(
        Q(1, 3)
        + ((2 * g - 1) * u + (2 * a - 1) * v + (2 * g - 1) * (2 * a - 1) * w) / 2
        for g in range(2)
        for a in range(2)
        for _ in range(9)
    )
    phases = tuple(
        Q(1, 7)
        + (
            (2 * g - 1) * delta
            + (2 * a - 1) * inner
            - (2 * g - 1) * (2 * a - 1) * mixed
        )
        / 2
        for g in range(2)
        for a in range(2)
        for _ in range(9)
    )
    observed = composition._observe_repeated_composition(
        interface="matching", form=forms, phase_offsets=phases
    )
    dx, dt = observed.child_form_rate_bounds, observed.child_phase_rate_bounds
    du = observed.parent_form_rate_bounds[1] - observed.parent_form_rate_bounds[0]
    dd = observed.parent_phase_rate_bounds[1] - observed.parent_phase_rate_bounds[0]
    dv = (dx[1] - dx[0] + dx[3] - dx[2]) / 2
    dw = (dx[3] - dx[2] - dx[1] + dx[0]) / 2
    db = (dt[1] - dt[0] + dt[3] - dt[2]) / 2
    dk = (dt[1] - dt[0] - dt[3] + dt[2]) / 2
    gamma = 1 / (1023 * mp.pi)
    uu, vv, ww, d, b, k = (_mp(mp, value) for value in (u, v, w, delta, inner, mixed))
    _contains(mp, du, -(uu + gamma * mp.sin(d) * mp.cos(k)) / 2)
    _contains(mp, dv, -(vv + gamma * mp.sin(b) * mp.cos(k)) / 2)
    _contains(mp, dw, -ww + gamma * (mp.cos(b) + mp.cos(d)) * mp.sin(k) / 2)
    _contains(mp, dd, gamma * uu / 2)
    _contains(mp, db, gamma * vv / 2)
    _contains(mp, dk, -gamma * ww)
    eta = observed.geometry.gamma_bounds**2
    y_d, y_b, z = -uu / gamma, -vv / gamma, 2 * ww / gamma
    for phase_rate, expected in ((dd, -y_d), (db, -y_b), (dk, -z)):
        _contains(mp, 2 * phase_rate / eta, expected)
    _contains(
        mp,
        -2 * du / observed.geometry.gamma_bounds**3,
        (mp.sin(d) * mp.cos(k) - y_d) / gamma**2,
    )
    _contains(
        mp,
        -2 * dv / observed.geometry.gamma_bounds**3,
        (mp.sin(b) * mp.cos(k) - y_b) / gamma**2,
    )
    _contains(
        mp,
        4 * dw / observed.geometry.gamma_bounds**3,
        2 * ((mp.cos(b) + mp.cos(d)) * mp.sin(k) - z) / gamma**2,
    )


def test_source_and_every_original_fine_direction_are_carried(response):
    report = response
    assert report.acquisition.relative_preparation_dimension == 64
    assert len(report.acquisition.child_sources) == 4
    for child in report.acquisition.child_sources:
        assert child.form_error_bound == child.phase_error_bound == _ERROR
        assert child.scaled_handoff_time == 100 + 10**13
    assert report.source_certified and report.admitted
    assert report.disruption_certified and report.child_identities_retained
    assert (
        report.recovery_certified
        and report.prediction_certified
        and report.contrast_certified
    )
    assert not report.unresolved_conditions
    with pytest.raises(FrozenInstanceError):
        report.source_certified = False


def test_eight_state_forecast_and_finite_contrast_have_independent_margins(
    response, mp
):
    gamma = 1 / (1023 * mp.pi)
    epsilon, a = gamma**2, 1 + 2 * gamma**2
    # Lower comparison uses the moving mixed mode, not its initial current.
    integral = (1 - mp.exp(-4 * a)) / (192 * a)
    assert integral > mp.mpf(1) / 200
    assert mp.sin(mp.mpf(1) / (1200 * mp.e)) > mp.mpf(3) / 10000
    assert response.interaction_delay_integral_lower_bound >= Q(1, 200)
    assert response.gradient_phase_delay_lower_bound >= Q(3, 2 * 10**6)
    z = _mp(mp, response.full_phase_error_upper_bound)
    r = _mp(mp, response.full_form_error_upper_bound)
    fine = 2 * _mp(mp, _RECOVERED) * (1 + gamma) * mp.exp(32)
    assert fine <= z < mp.mpf("1e-50")
    assert 2 * _mp(mp, _RECOVERED) + 24 * gamma * fine <= r < mp.mpf("1e-50")
    lower = _mp(mp, response.paired_outer_gap_contrast_lower_bound)
    assert mp.mpf("1e-6") < lower <= mp.mpf(3) / (2 * 10**6) - epsilon / 5 - 4 * z
    prediction = _mp(mp, response.eight_coordinate_phase_prediction_error_upper_bound)
    assert 2 * z + 18 * (z + gamma * r) <= prediction < mp.mpf("1e-7")


def test_finite_recovery_uses_both_internal_modes(response, mp):
    epsilon = 1 / (1023 * mp.pi) ** 2
    initial_inner = mp.mpf(1) / 4 + mp.exp(-mp.mpf(23) / 24 * (4 - epsilon)) / 2
    inner = initial_inner * mp.exp(-mp.mpf(19) / 20 * (4 - epsilon))
    mixed = mp.mpf(1) / 4 * mp.exp(-mp.mpf(19) / 10 * (4 - epsilon / 2))
    assert inner <= _mp(mp, response.recovered_inner_mode_upper_bound)
    assert mixed <= _mp(mp, response.recovered_mixed_mode_upper_bound)
    assert response.post_event_disruption_margin > 0
    assert response.parent_gap_at_recovery_upper_bound < Q(1, 4)
    assert response.parent_phase_residual_at_recovery_upper_bound < Q(1, 100)
    assert response.parent_form_residual_at_recovery_upper_bound < Q(1, 100)


@pytest.mark.parametrize(
    "dependency", ("sources_certified", "entry_certified", "child_identities_retained")
)
def test_failed_actual_acquisition_dependency_withholds_response(
    dependency, response, monkeypatch
):
    source = replace(response.acquisition, **{dependency: False})
    monkeypatch.setattr(
        owner, "_assess_organized_parent_acquisition", lambda **kwargs: source
    )
    report = _assess()
    assert not report.source_certified and not report.admitted
    assert report.status == "unavailable" and report.unresolved_conditions


def test_unrelated_old_readout_verdict_is_not_an_entry_premise(response, monkeypatch):
    source = replace(
        response.acquisition,
        prediction_certified=False,
        admitted=False,
        status="unavailable",
    )
    monkeypatch.setattr(
        owner, "_assess_organized_parent_acquisition", lambda **kwargs: source
    )
    report = _assess()
    assert report.source_certified and report.admitted


@pytest.mark.parametrize(
    "extra",
    (
        "acquisition",
        "source_report",
        "event_amplitude",
        "recovery_time",
        "readout_time",
        "trajectory",
    ),
)
def test_frozen_source_event_and_horizons_cannot_be_replaced(extra):
    with pytest.raises(TypeError):
        _assess(**{extra: 0})


def test_event_work_retains_signed_linear_reached_state_error(response, mp):
    report = response
    assert (
        report.event_work_error_upper_bound
        == 18 * report.acquisition.full_phase_error_upper_bound
    )
    assert 0 < report.event_work_bounds.lo <= report.event_work_bounds.hi < 3
    # The reached child means are not constrained to equal the moving
    # reference. A tiny mean shift changes this jump work at FIRST order.
    b, d = mp.mpf(1) / 100, mp.mpf(1) / 1000
    radius = _mp(mp, report.acquisition.full_phase_error_upper_bound)
    nominal = 9 * (mp.cos(b) - mp.cos(b + mp.mpf(1) / 2)) + 18 * mp.cos(d) * (
        1 - mp.cos(mp.mpf(1) / 4)
    )
    errors = []
    for sign in (-1, 1):
        shift = sign * radius / 12
        before = tuple(
            mp.mpf(2 * g - 1) * d / 2
            + mp.mpf(2 * a - 1) * b / 2
            + (mp.mpf(2 * a - 1) * shift if g == 0 else 0)
            for g in range(2)
            for a in range(2)
        )
        after = (
            before[0] - mp.mpf(1) / 4,
            before[1] + mp.mpf(1) / 4,
            before[2],
            before[3],
        )
        contacts = ((0, 1), (2, 3), (0, 2), (1, 3))
        work = 9 * sum(
            mp.cos(before[j] - before[i]) - mp.cos(after[j] - after[i])
            for i, j in contacts
        )
        correction = work - nominal
        assert radius / 10 < abs(correction) < 18 * radius
        errors.append(correction)
    assert errors[0] < 0 < errors[1]


def test_heat_integral_arithmetic_preserves_subgrid_initial_error():
    initial = Q(2, 10**1000)
    gamma, growth, factor = Q(1, 3000), Q(3**32), Q(24)
    form, phase = _prepared_heat_error_radii(
        initial_form_error=initial,
        initial_phase_error=initial,
        gamma_upper=gamma,
        phase_growth_upper=growth,
        form_convolution_factor=factor,
    )
    expected_phase = initial * (1 + gamma) * growth
    assert phase == expected_phase > 0
    assert form == initial + gamma * factor * expected_phase > 0


@pytest.mark.parametrize(
    "field",
    (
        "initial_form_error",
        "initial_phase_error",
        "gamma_upper",
        "phase_growth_upper",
        "form_convolution_factor",
    ),
)
@pytest.mark.parametrize("bad", (True, "0", float("nan"), -Q(1, 10**100)))
def test_heat_integral_admits_all_primitives_even_for_zero_error(field, bad):
    arguments = dict(
        initial_form_error=0,
        initial_phase_error=0,
        gamma_upper=Q(1, 3000),
        phase_growth_upper=1,
        form_convolution_factor=24,
    )
    arguments[field] = bad
    with pytest.raises((TypeError, ValueError)):
        _prepared_heat_error_radii(**arguments)


def test_heat_integral_rejects_growth_smaller_than_one():
    with pytest.raises(ValueError):
        _prepared_heat_error_radii(
            initial_form_error=0,
            initial_phase_error=0,
            gamma_upper=Q(1, 3000),
            phase_growth_upper=Q(999, 1000),
            form_convolution_factor=24,
        )
