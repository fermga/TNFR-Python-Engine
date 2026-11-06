"""Independent analytical controls; no record or source history is evaluated."""

from fractions import Fraction as Q
from math import exp, expm1, factorial

import pytest

from tnfr.mathematics._exact_linear_algebra import (
    exact_matrix_product,
    exact_symmetric_semidefinite,
)
from tnfr.physics import _organized_law_phase_transport as transport
from tnfr.physics._organized_computable_prediction import _ForecastStage
from tnfr.physics._organized_law_observation import _bound_organized_observation_law


@pytest.fixture(scope="module")
def report():
    return transport._bound_organized_law_phase_transport(deformation=1)


def test_sample_domain_ends_before_the_unobserved_last_ramp(report):
    # Five blocks have three paired plateaus each; only fourteen intervening
    # ramps are bracketed by observations. The last return ramp is excluded.
    assert len(report.observation_times) == 5 * 3 * 2
    assert report.observation_times[0] == 0
    assert len(report.cell_durations) == 29
    assert report.cell_durations.count(Q(1)) == 14
    assert report.cell_durations.count(Q(1, 10**5)) == 15
    assert report.command_variation == (4 * (1 + 2 + 1) + 1 + 2) * Q(1, 10)
    assert report.command_weighted_variation == report.command_variation
    assert report.cell_cube_sum == 14 + 15 * Q(1, 10**5) ** 3
    assert report.historical_start == 200
    assert report.historical_end == 214 + 15 * Q(1, 10**5) < 215 < 216
    assert report.observation_times[-1] == sum(report.cell_durations)


def test_exact_c5_operator_controls_the_green_and_noise_factors(report):
    incidence = tuple(
        tuple(Q(i == (j + 1) % 5) - Q(i == j) for j in range(5)) for i in range(5)
    )
    laplacian = exact_matrix_product(incidence, tuple(zip(*incidence)))
    generator = tuple(tuple(value / 4 for value in row) for row in laplacian)
    assert exact_symmetric_semidefinite(generator)
    assert exact_symmetric_semidefinite(
        tuple(tuple(Q(i == j) - generator[i][j] for j in range(5)) for i in range(5))
    )
    edge_phase = exact_matrix_product(tuple(zip(*incidence)), generator)
    assert all(sum(value**2 for value in row) == Q(5, 4) for row in edge_phase)
    assert report.edge_phase_row_squared_norm == Q(5, 4)
    # Two endpoint errors each have norm <=sqrt(5)*epsilon. Each bridge
    # weight has spectral norm <=1; the edge row contributes sqrt(5)/2.
    assert 2**2 * 5 * report.edge_phase_row_squared_norm == 5**2
    assert sum((Q(1, factorial(k)) for k in range(5)), Q(0)) > Q(8, 3)
    assert 1 / (1 - Q(3, 8)) == report.green_kernel_factor


@pytest.mark.parametrize("mu", (0.0, 0.125, 0.5, 1.0))
@pytest.mark.parametrize("t,s", ((0.2, 0.7), (0.7, 0.2), (0.4, 0.4)))
def test_scalar_green_bound_and_exact_bridge_for_constant_forcing(mu, t, s, report):
    # Synthetic scalar modes, independent of the protocol or a TNFR history.
    green_zero = min(t, s) * (1 - max(t, s))
    if mu == 0:
        green = green_zero
        bridge_weight = t
        endpoint = 2 + 3
        truth = 2 + 3 * t
    else:
        green = (
            -expm1(-mu * min(t, s))
            * -expm1(-mu * (1 - max(t, s)))
            * exp(-mu * max(0, t - s))
            / (-mu * expm1(-mu))
        )
        bridge_weight = expm1(-mu * t) / expm1(-mu)
        endpoint = 2 * exp(-mu) - 3 * expm1(-mu) / mu
        truth = 2 * exp(-mu * t) - 3 * expm1(-mu * t) / mu
    assert 0 <= green <= float(report.green_kernel_factor) * green_zero
    assert 0 <= bridge_weight <= 1
    assert (1 - bridge_weight) * 2 + bridge_weight * endpoint == pytest.approx(truth)


def test_cubic_command_integral_and_rational_bound_are_independent(report):
    # int_0^h s(h-s)/2 * |6*a*s*(h-s)/h^3| ds = |a|*h^2/10.
    for duration in (Q(1), Q(1, 3)):
        integral = (
            3
            / duration**3
            * (
                duration**2 * duration**3 / 3
                - 2 * duration * duration**4 / 4
                + duration**5 / 5
            )
        )
        assert integral == duration**2 / 10
    assert Q(19, 6) ** 2 > 10
    law = _bound_organized_observation_law(deformation=1)
    expected_factor = (
        report.green_kernel_factor
        * Q(19, 6)
        / 4
        / 10
        * law.gamma_upper**2
        * law.current_lipschitz
    )
    assert report.command_phase_error == expected_factor * report.command_variation


def test_source_noise_prefixes_and_current_variation_remain_separate(report):
    law = _bound_organized_observation_law(deformation=1)
    assert report.source_form_error_upper == law.source_form_error_upper
    assert (
        report.sample_error_upper == report.reading_error + law.source_form_error_upper
    )
    assert report.carrier_form_upper == (
        law.initial_form_upper + law.gamma_upper * law.current_upper * law.duration / 2
    )
    assert report.endpoint_noise_phase_error == (
        5 * law.gamma_upper * report.sample_error_upper * report.observation_times[-1]
    )
    assert report.endogenous_phase_error == (
        law.gamma_upper**3
        * law.current_lipschitz
        * report.carrier_form_upper
        * report.cell_cube_sum
        / 3
    )
    assert report.historical_phase_increment_error == sum(
        (
            report.endpoint_noise_phase_error,
            report.command_phase_error,
            report.endogenous_phase_error,
        )
    )
    assert all(
        a < b
        for a, b in zip(report.cell_end_error_bounds, report.cell_end_error_bounds[1:])
    )
    assert report.cell_end_error_bounds[-1] == report.historical_phase_increment_error
    assert report.historical_phase_increment_error < Q(28, 10**9)
    assert (
        report.absolute_initial_phase_drift_upper
        > 600 * report.historical_phase_increment_error
    )


@pytest.mark.parametrize("value", (True, False, float("nan"), float("inf"), -1, 2))
def test_invalid_law_domain_rejects_before_materializing_schedule(monkeypatch, value):
    def forbidden():
        raise AssertionError("invalid deformation reached schedule construction")

    monkeypatch.setattr(transport, "_forecast_stages", forbidden)
    with pytest.raises((TypeError, ValueError)):
        transport._bound_organized_law_phase_transport(deformation=value)


def test_nonconstant_reading_plateau_is_not_admitted_by_its_label(monkeypatch):
    formation = _ForecastStage(Q(0), Q(200), "formation")
    forged = _ForecastStage(Q(200), Q(1, 10**5), "plateau", 0, Q(0), Q(1, 10))
    monkeypatch.setattr(transport, "_forecast_stages", lambda: (formation, forged))
    with pytest.raises(ArithmeticError, match="zero command-form offset"):
        transport._bound_organized_law_phase_transport(deformation=0)
