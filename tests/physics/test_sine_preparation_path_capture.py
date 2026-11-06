"""Independent source, passage, contraction and full-state capture controls."""

from fractions import Fraction as Q

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.physics import _sine_preparation_path_capture as owner


@pytest.fixture(scope="module", autouse=True)
def no_trajectory_or_reference_execution():
    with forbid_sine_regeneration():
        yield


def _assess(**changes):
    inputs = dict(
        parameter_interval=(Q(3, 4), Q(4, 5)),
        form_error_bound=Q(1, 10**10),
        phase_error_bound=Q(1, 10**10),
    )
    return owner._assess_preparation_path_capture(**(inputs | changes))


@pytest.fixture(scope="module")
def report():
    return _assess()


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 100
    return context


def _mp(mp, value):
    return mp.mpf(value.numerator) / value.denominator


def _contains(mp, interval, expected):
    assert _mp(mp, interval.lo) <= expected <= _mp(mp, interval.hi)


def test_initial_mixed_edges_follow_the_original_source_and_exact_cancellation(
    report, mp
):
    gamma = 1 / (1023 * mp.pi)
    scale = mp.mpf(2046) / 9 * (mp.mpf(355) / 113) ** 2
    kappa = mp.mpf(1) / (27 * 512**2)
    dipole = [int(i == 3) - int(i == 4) for i in range(9)]
    # The source-to-y identity has no homogeneous z(T) remainder.
    source = report.source
    exact_error = (
        source.scaled_initial_error_norm_upper_bound
        + source.initial_phase_error_norm_upper_bound
        + 200 * source.eta_bounds.hi * source.forcing_norm_bounds.hi
    )
    assert report.initial_mixed_phase_error_norm_upper_bound == exact_error
    for parameter in (Q(3, 4), Q(31, 40), Q(4, 5)):
        s = _mp(mp, parameter)
        denominator = 1 + kappa * s * s
        p = (
            9 * 512 * (1 - kappa * s * s + 2 * kappa * s)
            - (2 * s - 1 + kappa * s * s) / 512
        ) / denominator
        q = (3 * (1 - kappa * s * s + 2 * kappa * s) + (2 * s - 1 + kappa * s * s)) / (
            2 * denominator
        )
        proxy = [
            gamma * scale * (p * dipole[i] + q * (i - 4)) - 2 * mp.pi * 512 * dipole[i]
            for i in range(9)
        ]
        for sign in (-1, 1):
            for index, interval in enumerate(report.initial_ordinary_gap_bounds):
                _contains(
                    mp,
                    interval,
                    proxy[index + 1] - proxy[index] + sign * _mp(mp, exact_error),
                )
            _contains(
                mp,
                report.initial_closing_coordinate_bounds,
                proxy[8] - proxy[0] - 3 * mp.pi + sign * _mp(mp, exact_error),
            )
    assert report.initial_coupled_lower_margin > Q(7, 1000)
    assert report.initial_closing_lower_margin > Q(348, 1000)
    assert report.initial_closing_upper_margin > Q(15, 10000)


def test_uniform_hidden_form_and_mixed_error_retain_every_later_time(report, mp):
    source = report.source
    gamma = 1 / (1023 * mp.pi)
    steady = gamma**2 * mp.sqrt(18) * 5
    z = report.uniform_scaled_form_norm_upper_bound
    assert z == max(
        source.scaled_form_radius_upper_bound,
        5 * source.eta_bounds.hi * source.forcing_norm_bounds.hi,
    )
    assert _mp(mp, z) >= steady
    assert report.mixed_field_error_upper_bound == 2 * z < Q(1, 200000)
    # Any semigroup convex combination stays below max(initial,steady),
    # independent of a particular trajectory or a finite time grid.
    initial = _mp(mp, source.scaled_form_radius_upper_bound)
    for weight in (Q(0), Q(3, 7), Q(1)):
        assert _mp(mp, weight) * initial + (1 - _mp(mp, weight)) * steady <= _mp(mp, z)


def test_dini_boundary_cancellation_includes_endpoint_and_interior_minima(report, mp):
    epsilon = _mp(mp, report.mixed_field_error_upper_bound)
    for q in (mp.mpf(-7) / 4, mp.mpf(-1), mp.mpf(0), mp.mpf(63) / 100):
        minimum = mp.mpf(9) / 10 + q / 2
        remaining = (3 * mp.pi + q - minimum) / 7
        assert minimum <= remaining < mp.mpf(13) / 10
        for location in range(8):
            gaps = [remaining] * 8
            gaps[location] = minimum
            sines = [mp.sin(value) for value in gaps] + [mp.sin(q)]
            derivative = (
                sines[location - 1] - 2 * sines[location] + sines[location + 1]
            ) / 2
            q_derivative = sines[8] - (sines[0] + sines[7]) / 2
            # Separate extremal perturbations conservatively enlarge the
            # actual correlated error set and must still give the inequality.
            lower = (derivative - epsilon) - (q_derivative + epsilon) / 2
            assert lower >= -3 * epsilon / 2 - mp.mpf("1e-95")


def test_analytic_passage_speed_and_prospective_hitting_budget(report, mp):
    values = (
        mp.sin(mp.mpf(607) / 500) - mp.sin(mp.mpf(63) / 100),
        mp.sin(mp.mpf(649) / 1000),
        mp.sin(mp.mpf(1) / 2),
    )
    for lower, value in zip(report.stage_one_sine_speed_lower_bounds, values):
        assert 0 <= value - _mp(mp, lower) < mp.mpf("1e-28")
    speed = min(values) - _mp(mp, report.mixed_field_error_upper_bound)
    assert (
        0
        <= speed - _mp(mp, report.stage_one_passage_speed_lower_bound)
        < mp.mpf("1e-28")
    )
    assert report.stage_one_passage_speed_lower_bound > Q(3, 10)
    assert report.stage_one_passage_time_upper_bound == Q(119, 15) < 8
    assert report.stage_one_coupled_lower_bound > Q(899, 1000)
    assert report.stage_one_ordinary_lower_bound > Q(3, 125)


def test_acute_tube_and_contraction_bound_have_independent_values(report, mp):
    epsilon, z = map(
        lambda value: _mp(mp, value),
        (
            report.mixed_field_error_upper_bound,
            report.uniform_scaled_form_norm_upper_bound,
        ),
    )
    lower = mp.mpf(7) / 4 - mp.pi
    upper = mp.mpf(13) / 10 + 8 * epsilon
    _contains(mp, report.stage_two_initial_gap_bounds, lower)
    _contains(mp, report.stage_two_initial_gap_bounds, upper)
    cap = max(abs(lower), abs(upper)) + 128 * epsilon + z
    assert (
        0
        <= _mp(mp, report.stage_two_actual_gap_absolute_upper_bound) - cap
        < mp.mpf("1e-28")
    )
    assert cap < mp.mpf(7) / 5
    curvature = mp.cos(mp.mpf(7) / 5)
    assert curvature > mp.mpf(1) / 6
    assert (
        0
        <= curvature - _mp(mp, report.stage_two_curvature_lower_bound)
        < mp.mpf("1e-28")
    )
    distance_squared = 45 * (mp.mpf(7) / 5 + 2 * mp.pi / 9) ** 2
    assert distance_squared < 225
    assert (
        0
        <= _mp(mp, report.stage_two_initial_distance_squared_upper_bound)
        - distance_squared
        < mp.mpf("1e-28")
    )
    _contains(mp, report.stage_two_decay_bounds, mp.exp(-mp.mpf(128) / 30))
    radius = 15 * mp.exp(-mp.mpf(128) / 30) + 30 * epsilon
    assert (
        0
        <= _mp(mp, report.stage_two_phase_distance_upper_bound) - radius
        < mp.mpf("1e-28")
    )
    assert (
        report.actual_phase_distance_at_capture_upper_bound
        == report.stage_two_phase_distance_upper_bound
        + report.uniform_scaled_form_norm_upper_bound
    )


def test_complete_storage_and_both_clock_rows_close_the_capture(report, mp):
    z = _mp(mp, report.uniform_scaled_form_norm_upper_bound)
    gamma = 1 / (1023 * mp.pi)
    radius = _mp(mp, report.actual_phase_distance_at_capture_upper_bound)
    expected = 9 * (1 - mp.cos(2 * mp.pi / 9)) + radius**2 + z**2 / gamma**2
    assert 0 <= _mp(mp, report.total_storage_upper_bound) - expected < mp.mpf("1e-28")
    barrier = 9 - 8 * mp.cos(3 * mp.pi / 16)
    assert (
        0
        <= barrier - _mp(mp, report.sector_boundary_storage_lower_bound)
        < mp.mpf("1e-28")
    )
    assert report.capture_margin > Q(19, 100)
    assert report.total_storage_upper_bound < Q(2156, 1000)
    assert report.slow_time_deadline == 136
    tau = 200 + 136 / gamma**2
    _contains(mp, report.scaled_time_deadline_bounds, tau)
    _contains(mp, report.original_time_deadline_bounds, tau * 1024 / 1023)


def test_entire_original_family_is_captured_without_discarding_saddle_exceptions(
    report,
):
    assert report.source.parameter_interval == (Q(3, 4), Q(4, 5))
    assert report.source.winding_at_time == 2
    assert report.source.captured_winding is None
    assert report.source.preparation_dimension == 16
    assert report.passage_certified and report.capture_certified
    assert report.captured_winding == 1
    assert report.source_policy_certified
    assert all(margin > 0 for _, margin in report.sufficient_margins)
    assert not report.unresolved_conditions


@pytest.mark.parametrize("errors", [(0, 0), (0, Q(1, 10**10)), (Q(1, 10**10), 0)])
def test_actual_selector_also_covers_nominal_and_lower_dimensional_sources(errors):
    selected = _assess(form_error_bound=errors[0], phase_error_bound=errors[1])
    assert selected.captured_winding == 1
    assert selected.capture_certified


@pytest.mark.parametrize(
    "changes",
    [
        {"parameter_interval": (0, Q(1, 8))},
        {"parameter_interval": (Q(63, 64), 1)},
        {"parameter_interval": (0, 1)},
        {"phase_error_bound": Q(1, 10)},
        {"form_error_bound": 100},
    ],
)
def test_failed_selector_premises_remain_unavailable(changes):
    result = _assess(**changes)
    assert result.status == "unavailable"
    assert not result.capture_certified
    assert result.captured_winding is None
    assert result.unresolved_conditions


def test_zero_margin_never_certifies_capture(monkeypatch):
    original = _assess()
    # Make the admitted coupled lower constant exactly its observed source bound.
    source = original.source
    endpoint = (
        min(value.lo for value in original.initial_ordinary_gap_bounds)
        - original.initial_closing_coordinate_bounds.hi / 2
    )
    monkeypatch.setattr(owner, "_GAP_LOWER", endpoint)
    result = _assess()
    assert result.initial_coupled_lower_margin == 0
    assert not result.capture_certified
    assert "initial_coupled_gap_lower_bound" in result.unresolved_conditions
    assert result.source.parameter_interval == source.parameter_interval


@pytest.mark.parametrize(
    "changes",
    [
        {"parameter_interval": (True, 1)},
        {"parameter_interval": (0, 1 + Q(1, 2**300))},
        {"form_error_bound": True},
        {"form_error_bound": -1},
        {"phase_error_bound": float("nan")},
        {"phase_error_bound": "0"},
    ],
)
def test_source_admission_precedes_continuation_arithmetic(monkeypatch, changes):
    def forbidden():
        pytest.fail("invalid source reached continuation arithmetic")

    monkeypatch.setattr(owner, "pi_interval", forbidden)
    with pytest.raises((TypeError, ValueError)):
        _assess(**changes)


def test_capture_does_not_accept_an_incoming_certificate_or_supplied_horizon(report):
    with pytest.raises(TypeError):
        owner._assess_preparation_path_capture(source=report.source)
    with pytest.raises(TypeError):
        _assess(scaled_time=300)
