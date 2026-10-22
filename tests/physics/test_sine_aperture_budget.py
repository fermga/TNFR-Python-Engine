"""Independent response-free conditioning, boundary and ambiguity controls."""

import json
import subprocess
from decimal import Decimal
from fractions import Fraction as Q
from inspect import Parameter, signature

import numpy as np
import pytest

from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.physics import relational_sine_aperture_budget as owner
from tnfr.physics.relational_sine_two_port_inference import _inference_geometry
from tnfr.sdk.relational_reports import relational_report_to_dict
from tnfr.utils.io import json_loads


def _arguments():
    return dict(
        probe_duration=Q(1, 2**24),
        form_radius=Q(1, 2**48),
        phase_radius=Q(1, 2**48),
        readout_error_bound=Q(1, 2**90),
        averaged_reading_halfwidth_bound=Q(1, 2**90),
        clock_rate_derivative_bound=Q(1, 2**22),
    )


@pytest.fixture(scope="module", autouse=True)
def no_observation_execution():
    from tnfr.mathematics import _validated_taylor
    from tnfr.physics import (
        relational_sine_aperture_inference,
        relational_sine_aperture_readout,
        relational_sine_clock_inference,
        relational_sine_curvature_inference,
        relational_sine_two_port_inference,
        relational_sine_two_port_readout,
        relational_sine_two_pulse_inference,
    )

    def forbidden(*args, **kwargs):
        pytest.fail("a response-free budget must not run a response, inverse or worker")

    with pytest.MonkeyPatch.context() as patch:
        for module in (
            relational_sine_aperture_inference,
            relational_sine_aperture_readout,
            relational_sine_clock_inference,
            relational_sine_curvature_inference,
            relational_sine_two_port_inference,
            relational_sine_two_port_readout,
            relational_sine_two_pulse_inference,
        ):
            for name in vars(module):
                if name.startswith(("infer_sine_", "bound_sine_")):
                    patch.setattr(module, name, forbidden)
        for name in ("flow_jets", "picard_tube", "validated_box_taylor_step"):
            patch.setattr(_validated_taylor, name, forbidden)
        patch.setattr(subprocess, "run", forbidden)
        patch.setattr(subprocess, "Popen", forbidden)
        yield


@pytest.fixture(scope="module")
def report():
    return owner.assess_sine_aperture_budget(**_arguments())


def test_six_mandatory_budget_primitives_exclude_observations_and_targets(report):
    parameters = signature(owner.assess_sine_aperture_budget).parameters
    assert set(parameters) == set(_arguments()) and len(parameters) == 6
    assert all(
        item.kind == Parameter.KEYWORD_ONLY and item.default == Parameter.empty
        for item in parameters.values()
    )
    for extra in (
        "averaged_reading_bounds",
        "source_state",
        "reference_envelope",
        "clock_profile",
        "actual_angle_width_target",
    ):
        with pytest.raises(TypeError):
            owner.assess_sine_aperture_budget(**_arguments(), **{extra: report})
    assert report.phase_increments == (Q(1, 4), Q(3, 4))
    assert report.readout_gain_bounds == (1, 2) and report.clock_rate_bounds == (
        Q(1, 2),
        2,
    )


def test_source_guard_is_valid_for_every_edge_of_the_fixed_affine_chart():
    chart = _inference_geometry()
    pi = pi_interval()
    assert pi.hi < Q(355, 113)
    assert 11 - Q(7, 2) * Q(355, 113) == Q(1, 226) > Q(1, 256)
    for p, b, c in chart["edge_angle_affine_coefficients"]:
        edge = p * pi + b * I(Q(11, 8), Q(3, 2)) + c * I(Q(2, 3), Q(1))
        assert pi.lo / 2 - edge.abs_max > Q(1, 256)
    assert chart["named_cycle_periods"] == (2, 1, 0)


def test_independent_moment_projection_recovers_separated_error_budgets(report):
    matrix = (
        (Q(11, 6), Q(-7, 6), Q(1, 3), Q(0)),
        (Q(-1, 24), Q(13, 12), Q(-1, 24), Q(0)),
        (Q(1, 3), Q(-7, 6), Q(11, 6), Q(0)),
        (Q(-1, 3), Q(7, 6), Q(-11, 6), Q(2)),
    )
    h, hs, g = report.probe_duration, 2 * report.probe_duration, Q(1, 3069)
    q0 = report.form_radius + 7 * g * (2 * hs)
    speed = 2 * q0 + 7 * g
    clock = tuple(
        2 * speed * report.clock_rate_derivative_bound * h**2 * c
        for c in (Q(7, 108), Q(13, 108), Q(7, 108), Q(5, 12))
    )
    for values, result in (
        ((report.readout_error_bound,) * 4, report.virtual_sensor_error_radii),
        (
            (report.averaged_reading_halfwidth_bound,) * 4,
            report.virtual_numerical_error_radii,
        ),
        (clock, report.virtual_clock_error_radii),
    ):
        assert result == tuple(
            sum(abs(w) * value for w, value in zip(row, values)) for row in matrix
        )
    assert report.virtual_reconstruction_error_radii == (
        2 * report.third_derivative_bound * hs**3 / 108,
        2 * report.third_derivative_bound * hs**3 / 2304,
        2 * report.third_derivative_bound * hs**3 / 108,
        2 * report.third_derivative_bound * hs**3 / 108
        + 2 * report.second_derivative_bound * hs**2 / 6,
    )
    assert report.virtual_point_error_radii == tuple(
        sum(values)
        for values in zip(
            report.virtual_sensor_error_radii,
            report.virtual_numerical_error_radii,
            report.virtual_clock_error_radii,
            report.virtual_reconstruction_error_radii,
        )
    )
    r0, rm, rh, _ = report.virtual_point_error_radii
    assert report.curvature_numerator_diameter_bound == 8 * (r0 + 2 * rm + rh) / h**2
    assert (
        report.curvature_numerator_diameter_bound
        == report.curvature_observation_error_term
        + report.curvature_clock_drift_term
        + report.curvature_reconstruction_term
    )


def test_specialized_full_flow_error_keeps_its_own_norm_and_sine_bounds(report):
    h, g, x, y = (
        report.probe_duration,
        Q(1, 3069),
        report.form_radius,
        report.phase_radius,
    )
    total, hs = 4 * h, 2 * h
    q = (x + g * total * (7 + 2 * y)) / (1 - 4 * g * g * total**2)
    error = 2 * hs * q + 2 * g * hs * y + 4 * g * g * total * hs * q
    assert report.specialized_form_norm_bound == q
    assert q != report.whole_window_form_norm_bound
    assert report.finite_response_error_bound == error
    assert (
        report.transformed_error_radius
        == 36000
        * (
            2 * error
            + report.virtual_point_error_radii[2]
            + report.virtual_point_error_radii[3]
        )
        / hs
    )
    sine = min(Q(7), 5 + 2 * y + 8 * g * hs * q)
    m3 = 8 * (1 + 2 * g * g) * q + 4 * g * (1 + g * g) * sine + 8 * g**3 * q * q
    assert report.specialized_sine_norm_bound == sine
    assert report.specialized_third_derivative_bound == m3
    assert (
        report.normalized_curvature_error_bound
        == 2 * (4 * (1 + g * g) * x + 4 * g * y) + 2 * h * m3
    )


def test_analytic_reference_budget_is_eligible_and_meets_all_fixed_targets(report):
    assert report.source_chart_eligible and report.transformed_radius_eligible
    assert report.curvature_quotients_eligible and report.sufficient_bound_eligible
    assert report.curvature_numerator_lower_candidate > 0
    assert report.corrected_curvature_numerator_margin > 0
    assert report.transformed_error_radius < Q(1, 11700)
    assert report.nominal_angle_width_upper_bound < Q(1, 1450)
    assert report.effective_gain_width_upper_bound < Q(1, 2900)
    assert (
        report.status == "certified_sufficient_budget"
        and report.sufficient_resolution_certified
    )
    for name in ("actual_angle", "effective_gain", "readout_gain", "mean_clock"):
        assert getattr(report, name + "_width_upper_bound") == getattr(
            report, name + "_width_candidate"
        )
        assert getattr(report, name + "_width_upper_bound") < getattr(
            report, name + "_width_target"
        )
    assert report.ineligibility_reasons == report.unmet_resolution_targets == ()


def test_positive_horizon_interval_with_larger_errors_uses_uniform_bounds():
    # Bound the whole interval analytically: increasing terms use H_plus,
    # inverse powers use H_minus. This is not a finite horizon grid.
    lower, upper = Q(3, 2**26), Q(1, 2**24)
    args = {
        **_arguments(),
        "readout_error_bound": Q(1, 2**80),
        "averaged_reading_halfwidth_bound": Q(1, 2**80),
    }
    high = owner.assess_sine_aperture_budget(**args)
    g, k0 = Q(1, 3069), Q(1, 18000)
    epsilon = args["readout_error_bound"] + args["averaged_reading_halfwidth_bound"]
    radius = 18000 * (
        8 * high.specialized_form_norm_bound
        + 8 * g * args["phase_radius"]
        + 64 * g * g * upper * high.specialized_form_norm_bound
        + 26 * epsilon / (3 * lower)
        + Q(226, 81)
        * high.structural_readout_speed_bound
        * args["clock_rate_derivative_bound"]
        * upper
        + Q(8, 27) * high.third_derivative_bound * upper**2
        + Q(4, 3) * high.second_derivative_bound * upper
    )
    diameter = (
        72 * epsilon / lower**2
        + high.curvature_clock_drift_term
        + high.curvature_reconstruction_term
    )
    d = high.normalized_curvature_error_bound
    numerator = (k0 / 2 - d) / 2 - diameter
    assert numerator > 0 and numerator / 4 - d > 0
    assert radius < Q(1, 11700)
    wj = 4 * radius
    wb = 2 * radius / (Q(1, 4) - 2 * radius)
    relative = 11 * g * wb / (8 * k0)
    product = 2 * wj + relative + 2 * wj * relative
    wrate = 2 * diameter / k0 + (2 + d / k0) * product + 2 * d / k0
    wgain = 2 * wj + 4 * wrate
    assert wb + args["phase_radius"] / 4 < Q(1, 1450)
    assert wj < Q(1, 2900)
    assert wrate < Q(3, 200) < high.mean_clock_width_target
    assert wgain < Q(3, 50) < high.readout_gain_width_target
    assert lower < upper and high.sufficient_resolution_certified


def test_shortening_horizon_increases_noise_amplification_and_can_admit_witness():
    args = _arguments()
    original = owner.assess_sine_aperture_budget(**args)
    shorter = owner.assess_sine_aperture_budget(
        **{**args, "probe_duration": args["probe_duration"] / 2}
    )
    assert (
        shorter.curvature_observation_error_term
        == 4 * original.curvature_observation_error_term
    )
    assert (
        shorter.curvature_reconstruction_term < original.curvature_reconstruction_term
    )
    tiny = owner.assess_sine_aperture_budget(**{**args, "probe_duration": Q(1, 2**60)})
    assert not original.noise_overlap_witness_admitted
    assert (
        tiny.noise_overlap_witness_admitted and not tiny.sufficient_resolution_certified
    )


def test_numeric_halfwidth_and_sensor_error_have_same_budget_but_distinct_witness():
    args = _arguments()
    h, g = args["probe_duration"], Q(1, 3069)
    threshold = Q(245, 8) * g * h**2 + Q(1365, 32) * g**3 * h**3
    sensor = owner.assess_sine_aperture_budget(
        **{
            **args,
            "readout_error_bound": threshold,
            "averaged_reading_halfwidth_bound": 0,
        }
    )
    numerical = owner.assess_sine_aperture_budget(
        **{
            **args,
            "readout_error_bound": 0,
            "averaged_reading_halfwidth_bound": threshold,
        }
    )
    assert sensor.combined_average_error_bound == numerical.combined_average_error_bound
    assert sensor.virtual_point_error_radii == numerical.virtual_point_error_radii
    assert sensor.transformed_error_radius == numerical.transformed_error_radius
    assert (
        sensor.noise_overlap_witness_admitted
        and not numerical.noise_overlap_witness_admitted
    )
    assert not sensor.sufficient_resolution_certified


def test_noise_overlap_threshold_follows_integrated_full_flow_remainders(report):
    # Independent remainder integration for the same-J gain/rate pair. The
    # largest average of s^2 and s^3 is over [H,2H]; no response is selected.
    h, g = report.probe_duration, Q(1, 3069)
    pairs = ((Q(3, 2), Q(1)), (Q(1), Q(3, 2)))
    assert len({gain * rate for gain, rate in pairs}) == 1
    mean_square = ((2 * h) ** 3 - h**3) / (3 * h)
    mean_cube = ((2 * h) ** 4 - h**4) / (4 * h)
    sum_remainders = sum(
        gain * (7 * g * rate**2 * mean_square + Q(14, 3) * g**3 * rate**3 * mean_cube)
        for gain, rate in pairs
    )
    assert report.noise_overlap_sensor_error_threshold == sum_remainders / 2
    assert report.noise_overlap_gain_rate_pairs == pairs
    assert report.noise_overlap_common_effective_gain == Q(3, 2)
    assert (
        abs(pairs[0][0] - pairs[1][0])
        == abs(pairs[0][1] - pairs[1][1])
        == report.noise_overlap_required_gain_and_mean_clock_width
    )


def test_noise_witness_equality_and_independence_from_whole_ball_chart_guard(report):
    threshold = report.noise_overlap_sensor_error_threshold
    for noise, expected in ((threshold, True), (threshold - Q(1, 2**400), False)):
        result = owner.assess_sine_aperture_budget(
            **{**_arguments(), "readout_error_bound": noise, "phase_radius": Q(1)}
        )
        assert result.noise_overlap_witness_admitted is expected
        assert not result.source_chart_eligible and not result.sufficient_bound_eligible
        assert result.actual_angle_width_upper_bound is None
        assert result.status == "not_certified"


@pytest.mark.parametrize("guard", ("radius", "numerator", "corrected_numerator"))
def test_exact_eligibility_boundaries_abstain_without_fabricating_bounds(guard):
    args = {
        **_arguments(),
        "form_radius": 0,
        "phase_radius": 0,
        "readout_error_bound": 0,
        "averaged_reading_halfwidth_bound": 0,
        "clock_rate_derivative_bound": 0,
    }
    base = owner.assess_sine_aperture_budget(**args)
    h = base.probe_duration
    if guard == "radius":
        epsilon = (Q(1, 8) - base.transformed_error_radius) * h / 156000
    else:
        d = base.normalized_curvature_error_bound
        target = Q(1, 72000) - (d / 2 if guard == "numerator" else 9 * d / 2)
        epsilon = (target - base.curvature_numerator_diameter_bound) * h**2 / 72
    assert epsilon > 0
    result = owner.assess_sine_aperture_budget(
        **{**args, "readout_error_bound": epsilon}
    )
    if guard == "radius":
        assert (
            result.transformed_radius_margin == 0
            and not result.transformed_radius_eligible
        )
        assert result.nominal_angle_width_candidate is None
    elif guard == "numerator":
        assert result.curvature_numerator_lower_candidate == 0
    else:
        assert result.corrected_curvature_numerator_margin == 0
    assert (
        not result.sufficient_bound_eligible
        and not result.sufficient_resolution_certified
    )
    assert (
        result.effective_gain_width_upper_bound
        is result.mean_clock_width_upper_bound
        is None
    )
    assert result.ineligibility_reasons


def test_source_equality_unavailable_and_large_valid_budgets_remain_exact():
    for phase in (Q(1, 256), Q(10**400)):
        result = owner.assess_sine_aperture_budget(
            **{**_arguments(), "phase_radius": phase}
        )
        assert not result.source_chart_eligible
        assert result.sufficient_resolution_certified is False
        assert result.nominal_angle_width_upper_bound is None
        assert isinstance(result.curvature_numerator_lower_candidate, Q)
        assert (
            "source_phase_radius_not_below_fixed_chart_margin"
            in result.ineligibility_reasons
        )


def test_subgrid_horizon_and_error_are_not_rounded_to_zero():
    result = owner.assess_sine_aperture_budget(
        **{
            **_arguments(),
            "probe_duration": Q(1, 2**400),
            "readout_error_bound": Q(1, 2**900),
        }
    )
    assert result.probe_duration > 0 and I(result.probe_duration).contains(0)
    assert (
        result.readout_error_bound > 0
        and result.noise_overlap_sensor_error_threshold > 0
    )
    assert (
        result.status == "not_certified" and result.mean_clock_width_upper_bound is None
    )


def test_method_only_horizon_obstruction_is_not_an_identifiability_claim():
    g, k0 = Q(1, 3069), Q(1, 18000)
    # R>=C*H, whereas the angle target requires R<1/8200.
    c = 4368000 * g
    cap = 1 / (8200 * c)
    assert cap == Q(1023, 11939200000)
    # Wrho>=144*epsilon/(k0 H^2); combine the two necessary
    # inequalities for this sufficient certificate, without a horizon grid.
    epsilon_cap = k0 * cap**2 / 9216
    assert epsilon_cap == Q(116281, 2627380162068480000000000000)
    assert 2 * Q(1, 2**75) > epsilon_cap
    assert Q(80, 3) * 7 * g * Q(1, 2**16) / k0 > Q(1, 64)
    assert 144000 * Q(1, 2**30) > Q(1, 8200)
    # The separate exact overlap witness need not hold merely because a
    # sufficient method is obstructed at every H.
    result = owner.assess_sine_aperture_budget(
        **{
            **_arguments(),
            "readout_error_bound": Q(1, 2**75),
            "averaged_reading_halfwidth_bound": Q(1, 2**75),
            "form_radius": 0,
            "phase_radius": 0,
            "clock_rate_derivative_bound": 0,
        }
    )
    assert not result.sufficient_resolution_certified
    assert not result.noise_overlap_witness_admitted


@pytest.mark.parametrize("name", tuple(_arguments()))
@pytest.mark.parametrize(
    "value",
    (
        True,
        np.bool_(False),
        float("nan"),
        float("inf"),
        Q(-1, 10**400),
        Decimal("1e-400"),
    ),
)
def test_invalid_budget_primitives_are_rejected(name, value):
    with pytest.raises((TypeError, ValueError)):
        owner.assess_sine_aperture_budget(**{**_arguments(), name: value})


@pytest.mark.parametrize("horizon", (Q(0), Q(1, 4) + Q(1, 2**400)))
def test_horizon_work_domain_has_exact_endpoints(horizon):
    with pytest.raises(ValueError):
        owner.assess_sine_aperture_budget(**{**_arguments(), "probe_duration": horizon})
    assert owner.assess_sine_aperture_budget(
        **{**_arguments(), "probe_duration": Q(1, 4)}
    ).probe_duration == Q(1, 4)


def test_sdk_serialization_preserves_fraction_and_unavailable_widths(report):
    projected = json_loads(
        json.dumps(relational_report_to_dict(report), allow_nan=False)
    )
    assert projected["schema"] == "tnfr.relational-report.v1"
    assert projected["report_type"] == "SineApertureBudget"
    direct = report.to_dict()
    assert direct["schema"] == "tnfr.sine-aperture-budget.v1"
    assert direct["report"]["probe_duration"] == {"numerator": 1, "denominator": 2**24}
    unavailable = owner.assess_sine_aperture_budget(
        **{**_arguments(), "phase_radius": Q(1, 256)}
    )
    body = json_loads(json.dumps(unavailable.to_dict(), allow_nan=False))["report"]
    assert body["actual_angle_width_upper_bound"] is None
    assert body["mean_clock_width_upper_bound"] is None
    assert body["sufficient_bound_eligible"] is False
