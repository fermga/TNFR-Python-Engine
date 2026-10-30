"""Independent finite-tail and actual-source constitutive separation controls."""

from fractions import Fraction as Q

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.physics import _sine_constitutive_discrimination as owner

ERROR = Q(1, 10**10)
READOUT = Q(1, 10**54)


@pytest.fixture(scope="module", autouse=True)
def no_response_or_frozen_execution():
    with forbid_sine_regeneration():
        yield


def _assess(**changes):
    inputs = dict(
        form_error_bound=ERROR, phase_error_bound=ERROR, per_reading_error_bound=READOUT
    )
    return owner._assess_carried_constitutive_discrimination(**(inputs | changes))


@pytest.fixture(scope="module")
def report():
    return _assess()


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 120
    return context


def _mp(mp, value):
    return mp.mpf(value.numerator) / value.denominator


def test_real_variation_and_complete_defect_constants_have_independent_bounds():
    gamma, time, amplitude = Q(1, 3000), Q(1, 100), Q(7, 10000)
    linear_growth = 1 / (1 - 3 * time)
    assert linear_growth < Q(11, 10)
    second = Q(11, 10) * 2 * Q(11, 5) ** 2 / 3
    third = Q(11, 10) * (
        Q(4, 3) * Q(11, 5) ** 3 / 4 + Q(88, 5) * gamma**2 * time**2 / 6
    )
    assert second < 4 and third < 4
    assert 18 * gamma**2 * time**2 + 18 * gamma**4 * time**4 < 1
    assert Q(243, 4) * gamma**2 * time**2 < 1
    fourth_defect = Q(11, 50) * (2 + Q(56, 3) * Q(9, 4) ** 4)
    assert fourth_defect < 106
    replacement = Q(11, 10) * Q(4, 3) * Q(9, 4) * (Q(11, 5) ** 2 + 2 * Q(11, 5) + 4) / 5
    transport = Q(33, 10) * Q(4, 3) * 2**3 / 20
    recoupling = Q(11, 10) * 4 * Q(11, 5) * 2 / 6
    cubic_tail = replacement + transport + recoupling * gamma**2 * time
    assert cubic_tail < 11
    assert 106 * gamma * amplitude < 1
    assert 3 * (cubic_tail + fourth_defect * gamma * amplitude) < 36
    assert Q(1, 2560) - 36 * Q(1, 10**6) > Q(1, 3000)


def test_independent_onset_and_own_nonlinear_tail_enclose_each_law(report, mp):
    gamma, amplitude, horizon = 1 / (1023 * mp.pi), mp.mpf(7) / 10000, mp.mpf(1) / 10**6
    cosine = mp.cos(2 * mp.pi / 9)
    for row in report.observations:
        lam = _mp(mp, row.deformation)
        leading = (16 * lam - 15 - cosine) * gamma**4 * amplitude**3 * horizon**4 / 512
        lower, upper = map(lambda value: _mp(mp, value), row.leading_response_bounds)
        assert lower <= leading <= upper
        assert upper - lower < abs(leading) * mp.mpf("1e-28")
        tail = 36 * gamma**4 * amplitude**3 * horizon**5
        assert (
            0
            <= _mp(mp, row.nonlinear_remainder_upper_bound) - tail
            < tail * mp.mpf("1e-28")
        )
        assert row.current_fourth_derivative_upper_bound == (1 if lam == 0 else 28)
        assert row.nominal_bound_admitted
        assert all(margin > 0 for _, margin in row.nominal_bound_margins)
        nominal_lower, nominal_upper = map(
            lambda value: _mp(mp, value), row.nominal_response_bounds
        )
        assert nominal_lower <= leading - tail and leading + tail <= nominal_upper


def test_carry_and_readout_errors_are_added_for_all_four_histories(report):
    for row in report.observations:
        assert row.handoff.status == "certified_constitutive_handoff"
        assert all(branch.identity_certified for branch in row.handoff.branches)
        assert row.handoff.source.form_error_bound == ERROR
        assert row.handoff.source.phase_error_bound == ERROR
        assert row.handoff.endpoint_radius == Q(1, 10**70)
        carried = 8 * Q(1, 10**70)
        reading = 4 * READOUT
        assert row.carried_state_error_upper_bound == carried
        assert row.mixed_readout_error_upper_bound == reading
        lower, upper = row.nominal_response_bounds
        assert row.actual_response_bounds == (lower - carried, upper + carried)
        assert row.measured_response_bounds == (
            lower - carried - reading,
            upper + carried + reading,
        )
        # Exact tiny endpoints must not be expanded to the absolute interval grid.
        assert all(
            0 < abs(value) < Q(1, 10**49) for value in row.measured_response_bounds
        )


def test_entire_actual_source_and_error_sets_have_opposite_measured_signs(report):
    negative, positive = report.observations
    assert negative.measured_response_bounds[1] < -Q(8, 10**52)
    assert positive.measured_response_bounds[0] > Q(8, 10**52)
    assert report.common_nominal_magnitude_lower_bound > Q(9, 10**52)
    assert report.record_separation_lower_bound == (
        positive.measured_response_bounds[0] - negative.measured_response_bounds[1]
    )
    assert report.record_separation_lower_bound > Q(16, 10**52)
    assert report.status == "certified_opposite_carried_responses"
    assert not report.unresolved_conditions
    assert report.original_observation_duration == Q(1024, 1023 * 10**6)


def test_excess_readout_error_is_unavailable_not_an_observational_collision():
    result = _assess(per_reading_error_bound=Q(1, 10**40))
    assert result.status == "unavailable"
    assert result.record_separation_lower_bound < 0
    assert all(row.handoff.handoff_certified for row in result.observations)
    assert "lambda_1_strict_measured_sign_unavailable" in result.unresolved_conditions


def test_exact_zero_sign_margin_is_not_certified(report):
    positive = report.observations[1]
    allowance = positive.actual_response_bounds[0] / 4
    result = _assess(per_reading_error_bound=allowance)
    assert result.observations[1].signed_measured_margin == 0
    assert not result.observations[1].certified
    assert result.status == "unavailable"


def test_failed_actual_acquisition_cannot_be_replaced_by_a_nominal_sign():
    result = _assess(phase_error_bound=Q(1, 10))
    assert result.status == "unavailable"
    assert all(row.nominal_bound_admitted for row in result.observations)
    assert all(not row.certified for row in result.observations)
    assert "lambda_1_actual_handoff_unavailable" in result.unresolved_conditions


def test_subgrid_readout_allowance_keeps_its_exact_association():
    tiny = Q(1, 2**600)
    result = _assess(per_reading_error_bound=tiny)
    assert result.per_reading_error_bound == tiny
    assert all(
        row.mixed_readout_error_upper_bound == 4 * tiny for row in result.observations
    )


@pytest.mark.parametrize(
    "field,bad",
    [
        ("form_error_bound", True),
        ("form_error_bound", -1),
        ("phase_error_bound", float("nan")),
        ("phase_error_bound", "0"),
        ("per_reading_error_bound", False),
        ("per_reading_error_bound", float("inf")),
        ("per_reading_error_bound", -Q(1, 2**500)),
    ],
)
def test_raw_budgets_are_admitted_before_recovery(monkeypatch, field, bad):
    def forbidden(**kwargs):
        pytest.fail("invalid budget reached the source or law calculator")

    monkeypatch.setattr(owner, "_assess_constitutive_handoff", forbidden)
    with pytest.raises((TypeError, ValueError)):
        _assess(**{field: bad})


def test_fixed_comparison_cannot_accept_a_report_or_a_tuned_horizon(report):
    for change in (
        {"source": report.observations[0].handoff},
        {"horizon": Q(1, 8)},
        {"amplitude": Q(1, 100)},
    ):
        with pytest.raises(TypeError):
            _assess(**change)
