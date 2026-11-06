"""Preparation-based finite star predictions and complete-field controls."""

import json
from fractions import Fraction as Q

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from tnfr.physics.phase_response import assess_phase_information_response
from tnfr.sdk import export_to_json, relational_report_to_dict

LEFT = ((Q(1), Q(0)),) * 3 + ((Q(3, 5), Q(4, 5)),) * 3
RIGHT = ((Q(4, 5), Q(3, 5)),) * 5 + ((Q(4, 5), Q(-3, 5)),)
BUDGET = dict(
    duration=Q(1, 16), preparation_error=Q(1, 100000), observation_error=Q(1, 100000)
)


@pytest.fixture(scope="module")
def report():
    return assess_phase_information_response(LEFT, RIGHT, **BUDGET)


def test_declared_finite_budget_separates_both_complete_laws(report):
    assert report.discrimination_certified and report.status == "certified"
    assert report.reasons == ()
    assert report.information.degree == 6
    assert report.information.first_resultants == ((Q(24, 5), Q(12, 5)),) * 2
    assert report.root_initial_rate_values == ((Q(2, 5),) * 2, (Q(82, 125), Q(68, 125)))
    assert report.form_contrast_centers == (0, Q(-7, 1000))
    assert report.finite_time_error_upper_bounds == (Q(1, 3072), Q(1, 384))
    assert report.preparation_error_upper_bounds == (Q(29, 1280000), Q(49, 1600000))
    assert report.observation_error_upper_bound == Q(1, 50000)
    assert 0 < report.separation_margin_lower_bound <= Q(25453, 6400000)
    sine, cubic = report.form_contrast_prediction_bounds
    assert cubic.hi < sine.lo
    assert sine.contains(0) and cubic.hi < 0
    assert report.clock == "tau=t/pi"


def test_independent_full_fourteen_coordinate_flow_respects_preparation_error(report):
    """A numerical consistency control, not the proof of the analytic enclosure."""
    h, rho, sigma = (
        float(BUDGET[name])
        for name in ("duration", "preparation_error", "observation_error")
    )
    count = 7
    neighbors = (tuple(range(1, count)),) + ((0,),) * 6

    def field(_time, state, eta):
        x, phase = state[:count], state[count:]
        dx, dphase = np.empty(count), np.empty(count)
        for i, row in enumerate(neighbors):
            gaps = phase[list(row)] - phase[i]
            dx[i] = np.mean(np.sin(gaps) + eta * np.sin(gaps) ** 3)
            dphase[i] = x[i] - np.mean(x[list(row)])
        return np.concatenate((dx, dphase))

    for model_index, eta in enumerate((0, 1)):
        readings = []
        for trial, phasors in enumerate((LEFT, RIGHT)):
            ideal_phase = np.array(
                [0.0] + [np.arctan2(float(s), float(c)) for c, s in phasors]
            )
            signs = np.array([(-1) ** (i + trial) for i in range(count)])
            # Stay inside the admitted cube after numerical angle materialization.
            initial = np.concatenate((rho * signs / 2, ideal_phase - rho * signs / 2))
            result = solve_ivp(
                field,
                (0, h),
                initial,
                args=(eta,),
                method="DOP853",
                rtol=5e-13,
                atol=1e-14,
            )
            assert result.success
            endpoint = result.y[:, -1]
            ideal_rate = sum(s + eta * s**3 for _, s in phasors) / 6
            per_root_bound = (
                report.finite_time_error_upper_bounds[model_index]
                + report.preparation_error_upper_bounds[model_index]
            ) / 2
            assert abs(endpoint[0] - h * float(ideal_rate)) < float(per_root_bound)
            assert np.max(np.abs(endpoint[count:] - initial[count:])) > 1e-5
            assert np.ptp(endpoint[1:count]) > 1e-3
            readings.append(endpoint[0] + (-1) ** trial * sigma)
        bounds = report.form_contrast_prediction_bounds[model_index]
        assert float(bounds.lo) < readings[1] - readings[0] < float(bounds.hi)


def test_sine_equal_initial_current_does_not_imply_equal_future_response():
    third_derivatives = []
    for phasors in (LEFT, RIGHT):
        # Differentiate all star rows: x_leaf'=-s, x_root'=mean(s),
        # theta_root''=2*mean(s), theta_leaf''=-s-mean(s).
        mean = sum(s for _, s in phasors) / 6
        third_derivatives.append(sum(c * (-s - 3 * mean) for c, s in phasors) / 6)
    assert third_derivatives == [Q(-6, 5), Q(-32, 25)]
    assert third_derivatives[1] - third_derivatives[0] == Q(-2, 25)


def test_signed_trial_order_is_preserved_and_inputs_are_consumed_once(report):
    swapped = assess_phase_information_response(
        iter(RIGHT), (iter(row) for row in LEFT), **BUDGET
    )
    assert swapped.form_contrast_centers == tuple(
        -value for value in report.form_contrast_centers
    )
    assert swapped.form_contrast_radii == report.form_contrast_radii
    assert swapped.form_contrast_prediction_bounds == tuple(
        -value for value in report.form_contrast_prediction_bounds
    )
    assert swapped.discrimination_certified


@pytest.mark.parametrize(
    "changes",
    (
        {"observation_error": Q(1, 100)},
        {"preparation_error": Q(1, 100)},
        {"duration": 1},
    ),
)
def test_failed_sufficient_budget_is_unavailable_not_model_rejection(changes):
    result = assess_phase_information_response(LEFT, RIGHT, **(BUDGET | changes))
    assert not result.discrimination_certified
    assert result.status == "unavailable"
    assert result.reasons == ("finite_response_intervals_not_separated",)
    assert result.separation_margin_lower_bound <= 0


def test_equal_preparations_and_subgrid_signal_do_not_create_false_separation():
    same = assess_phase_information_response(LEFT, LEFT, **BUDGET)
    assert same.form_contrast_centers == (0, 0)
    assert not same.discrimination_certified
    tiny = Q(1, 10**400)
    result = assess_phase_information_response(LEFT, RIGHT, duration=tiny, epsilon=tiny)
    assert result.duration == result.information.epsilon == tiny
    assert result.form_contrast_centers[1] == Q(-14, 125) * tiny**2
    assert result.form_contrast_prediction_bounds[1].contains(0)
    assert result.status == "unavailable"


@pytest.mark.parametrize(
    "changes",
    (
        {"duration": 0},
        {"duration": True},
        {"duration": float("inf")},
        {"epsilon": 0},
        {"epsilon": -1},
        {"epsilon": False},
        {"preparation_error": -1},
        {"preparation_error": float("nan")},
        {"observation_error": -1},
        {"observation_error": True},
    ),
)
def test_invalid_budgets_reject_before_prediction(changes):
    with pytest.raises((TypeError, ValueError)):
        assess_phase_information_response(LEFT, RIGHT, **(BUDGET | changes))


def test_unequal_information_and_invalid_phasors_are_rejected():
    with pytest.raises(ValueError, match="equal first resultants"):
        assess_phase_information_response(LEFT, ((1, 0),) * 6, **BUDGET)
    with pytest.raises(ValueError, match="equal degree"):
        assess_phase_information_response(LEFT, RIGHT[:-1], **BUDGET)
    with pytest.raises(ValueError, match="unit norm"):
        assess_phase_information_response(((Q(1, 2), Q(1, 2)),), ((1, 0),), **BUDGET)


def test_sdk_export_retains_exact_model_clock_preparation_and_error_budget(
    report, tmp_path
):
    direct = report.to_dict()
    payload = relational_report_to_dict(report)
    assert direct["schema"] == "tnfr.phase-information-response.v1"
    assert payload["report_type"] == "PhaseInformationResponse"
    assert payload["report"] == direct["report"]
    assert payload["report"]["duration"] == {"numerator": 1, "denominator": 16}
    assert payload["report"]["information"]["left_phasors"][3] == [
        {"numerator": 3, "denominator": 5},
        {"numerator": 4, "denominator": 5},
    ]
    destination = tmp_path / "phase-information-response.json"
    export_to_json(payload, destination)
    assert json.loads(destination.read_text(encoding="utf-8")) == payload
    payload["report"]["form_contrast_centers"][1]["numerator"] = 99
    assert report.form_contrast_centers[1] == Q(-7, 1000)
