"""Independent fixed-neighbor source and observation-policy controls.

Only exact static geometry and scalar inequalities execute. Complete-law
flows, time-coefficient producers and archived workers remain forbidden.
"""

import itertools
from decimal import Decimal
from fractions import Fraction as Q
from types import SimpleNamespace

import numpy as np
import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.mathematics._rational_interval import I
from tnfr.research import sine_class_neighbor_forward as owner

EPS, G, H, PULSE = Q(1, 10**32), Q(1, 3000), Q(1, 8), Q(7, 10000)
DELTA, WIDTH, SMALL = Q(1, 10**30), Q(1, 10**30), Q(1, 2**600)
ELL, D = 1 - 2 * G * H, 1 - 2 * G**2 * H**2


def _initialization(amplitude):
    return 4 * G * H * EPS / (ELL * D) * (G * amplitude / D + EPS / ELL)


SOURCE = sum(_initialization(a) for a in (0, PULSE, PULSE, 2 * PULSE))


@pytest.fixture(scope="module", autouse=True)
def no_scientific_execution():
    with forbid_sine_regeneration():
        yield


@pytest.fixture(scope="module")
def prediction():
    return owner.neighbor_forward_prediction()


def test_reference_lifts_cover_exact_target_and_separate_actual_family():
    source = owner.neighbor_forward_sources()
    assert source["classes"] == (1, 2, 1)
    pi_lo, pi_hi = source["pi_bounds"]
    assert Q(25, 8) < pi_lo < pi_hi < Q(22, 7)
    assert pi_hi - pi_lo < Q(1, 2**250)
    residual = tuple(EPS / 2 * (int(j == 4) - int(j == 0)) for j in range(9))
    assert sum(residual) == 0 and sum(v**2 for v in residual) < EPS**2
    for node in range(27):
        winding, local = source["classes"][node // 9], node % 9
        factor = Q(2 * winding * (local - 4), 9)
        corners = factor * pi_lo, factor * pi_hi
        target = min(corners), max(corners)
        assert source["reference_form_bounds"][node] == (0, 0)
        assert source["reference_phase_bounds"][node] == target
        assert source["actual_form_bounds"][node] == (-EPS, EPS)
        assert source["actual_phase_bounds"][node] == (
            target[0] - EPS,
            target[1] + EPS,
        )
        for endpoint in corners:
            assert target[0] - EPS <= endpoint + residual[local] <= target[1] + EPS
        rounded = I(*target)
        assert rounded.lo <= target[0] <= target[1] <= rounded.hi
    assert source["reference_phase_bounds"][9][1] < -1
    assert source["reference_phase_bounds"][13] == (0, 0)
    # The numerical rectangle also covers inadmissible nonzero-sum corners.
    assert 9 * EPS != 0 and 9 * EPS**2 > EPS**2
    # Original component zero sums do not imply zero joined weighted mean.
    assert sum(d * residual[4] for d in (1, 2, 1)) == 2 * EPS


def test_fixed_inputs_select_distinct_ports_without_consuming_prediction(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("forward inputs must not consume a prediction")

    monkeypatch.setattr(owner, "neighbor_forward_prediction", forbidden)
    source, inputs = owner.neighbor_forward_sources(), owner.neighbor_forward_inputs()
    assert set(inputs) == {
        "initial_form_bounds",
        "initial_phase_bounds",
        "first_probe_amplitude",
        "second_probe_amplitude",
        "delay",
        "total_duration",
        "time_step",
        "order",
        "max_steps",
        "first_probe_node",
        "second_probe_node",
        "readout_node",
    }
    assert inputs["initial_form_bounds"] == source["reference_form_bounds"]
    assert inputs["initial_phase_bounds"] == source["reference_phase_bounds"]
    assert inputs["first_probe_amplitude"] == inputs["second_probe_amplitude"] == PULSE
    assert inputs["delay"] == 0 and inputs["total_duration"] == H
    assert inputs["time_step"] == Q(1, 128)
    assert type(inputs["order"]) is int and inputs["order"] == 16
    assert type(inputs["max_steps"]) is int and inputs["max_steps"] == 64
    assert tuple(
        inputs[key] for key in ("first_probe_node", "second_probe_node", "readout_node")
    ) == (4, 22, 13)
    lengths = (inputs["delay"],) * 2 + (H - inputs["delay"],) * 4
    assert sum(length / inputs["time_step"] for length in lengths) == 64


def test_common_source_transfer_charges_all_four_nonlinear_defects(prediction):
    assert prediction.mediator_class == 2
    assert prediction.donor_amplitude == prediction.receiver_amplitude == PULSE
    assert prediction.horizon == H and prediction.endpoint_radius == EPS
    assert prediction.readout_error_bound == DELTA
    assert prediction.intervention_nodes == (4, 22) and prediction.readout_node == 13
    assert prediction.conditional_protocol_sufficient
    assert ELL == Q(11999, 12000) and D == Q(287999999, 288000000)
    terms = tuple(_initialization(a) for a in (0, PULSE, PULSE, 2 * PULSE))
    assert terms[0] > 0
    assert SOURCE == 16 * G * H * EPS / (ELL * D) * (G * PULSE / D + EPS / ELL)
    assert SOURCE < Q(1556, 10**45)
    assert prediction.nonlinear_source_error_upper_bound == SOURCE
    assert tuple(
        row.total_input_variation for row in prediction.per_history_fidelity
    ) == (0, PULSE, PULSE, 2 * PULSE)
    weights = (1, -1, -1, 1)
    common_linear_source = Q(17, 19)
    outputs = tuple(
        common_linear_source + weight * error for weight, error in zip(weights, terms)
    )
    assert sum(weight * value for weight, value in zip(weights, outputs)) == SOURCE
    assert SOURCE < 4 * EPS / ELL
    policy = owner.neighbor_forward_policy()
    assert tuple(
        policy[key]
        for key in (
            "endpoint_radius",
            "gamma_upper",
            "total_duration",
            "readout_error_bound",
        )
    ) == (EPS, G, H, DELTA)
    assert policy["source_error_upper_bound"] == SOURCE
    assert policy["actual_width_allowance"] == WIDTH
    assert policy["reading_count"] == policy["independent_additive_reading_count"] == 4
    assert policy["mixed_coefficients"] == (1, -1, -1, 1)


def test_sensor_error_corners_distinguish_four_readings_from_two_record_sets():
    weights = (1, -1, -1, 1)
    errors = tuple(
        sum(w * e for w, e in zip(weights, corner))
        for corner in itertools.product((-DELTA, DELTA), repeat=4)
    )
    assert min(errors) == -4 * DELTA and max(errors) == 4 * DELTA
    assert -8 * DELTA + max(errors) == min(errors)
    assert -8 * DELTA - SMALL + max(errors) < min(errors)
    assert -4 * DELTA - SMALL + max(errors) < 0
    assert -4 * DELTA - SMALL + max(errors) > min(errors)


def test_full_forward_transport_is_once_and_does_not_add_amplitude_error(prediction):
    reference = (Q(-2, 10**29) - SMALL, Q(-2, 10**29) + SMALL)
    result = owner.assess_neighbor_forward_bounds(reference)
    assert result["reference_bounds"] == reference
    assert result["source_error_upper_bound"] == SOURCE
    assert result["actual_bounds"] == (reference[0] - SOURCE, reference[1] + SOURCE)
    assert result["reference_width"] == 2 * SMALL
    assert result["actual_width"] == 2 * SMALL + 2 * SOURCE
    assert result["decision"]["true_bounds"] == result["actual_bounds"]
    assert result["decision"]["recorded_bounds"] == (
        reference[0] - SOURCE - 4 * DELTA,
        reference[1] + SOURCE + 4 * DELTA,
    )
    assert (
        result["decision"]["null_separation_margin"]
        == -reference[1] - SOURCE - 8 * DELTA
    )
    assert prediction.higher_amplitude_error_upper_bound > SOURCE
    assert result["all_conditions_met"] and result["status"] == "all_conditions_met"


@pytest.mark.parametrize(
    "pair",
    (
        True,
        (0,),
        (0, 1, 2),
        (1, 0),
        (False, 0),
        (np.bool_(False), 0),
        (float("nan"), 0),
        (0, float("inf")),
        (Decimal("0"), 0),
        ({"numerator": True, "denominator": 1}, 0),
        ({"numerator": 0, "denominator": 0}, 0),
        ({"numerator": 0, "denominator": -1}, 0),
        ({"numerator": 0, "denominator": 1, "extra": 0}, 0),
    ),
)
def test_invalid_reference_scalars_are_not_coerced(pair, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid scalar reached the static prediction")

    monkeypatch.setattr(owner, "neighbor_forward_prediction", forbidden)
    with pytest.raises((ValueError, TypeError)):
        owner.assess_neighbor_forward_bounds(pair)


def test_exact_fraction_records_keep_subgrid_precision():
    lo, hi = Q(-2, 10**29) - SMALL, Q(-2, 10**29) + SMALL
    records = tuple(
        {"numerator": x.numerator, "denominator": x.denominator} for x in (lo, hi)
    )
    assert owner.assess_neighbor_forward_bounds(
        records
    ) == owner.assess_neighbor_forward_bounds((lo, hi))


def test_prediction_contains_its_own_errors_but_never_clips_forward_interval(
    prediction,
):
    direct = prediction.static_cone.direct_cubic_heat_bounds
    correction = (
        3 * PULSE**3 * (Q(4, 15) * G**6 * H**6 / D**4 + Q(1, 63) * G**8 * H**8 / D**5)
    )
    fifth = sum(
        256 * G**6 * a**5 * H / (D * (1 - 4 * G**2 * a**2))
        for a in (0, PULSE, PULSE, 2 * PULSE)
    )
    expected = direct[0] - correction - fifth, direct[1] + correction + fifth
    reference = (expected[0] - 1, expected[1] + 1)
    result = owner.assess_neighbor_forward_bounds(reference)
    assert result["nominal_prediction_bounds"] == expected
    assert result["actual_prediction_bounds"] == (
        expected[0] - SOURCE,
        expected[1] + SOURCE,
    )
    assert result["reference_bounds"] == reference
    assert result["actual_bounds"] == (reference[0] - SOURCE, reference[1] + SOURCE)
    assert result["nominal_prediction_overlap"] and result["actual_prediction_overlap"]
    assert result["status"] == "numerically_unresolved"


@pytest.mark.parametrize("side", (-1, 0, 1))
def test_closed_nominal_overlap_is_independent_of_wider_actual_overlap(side):
    nominal = owner.assess_neighbor_forward_bounds(None)["nominal_prediction_bounds"]
    point = nominal[0] + side * SMALL
    result = owner.assess_neighbor_forward_bounds((point, point))
    assert result["nominal_prediction_overlap"] is (side >= 0)
    # Source expansion can make actual bands overlap despite nominal conflict.
    assert result["actual_prediction_overlap"]
    assert result["reference_bounds"] == (point, point)
    assert (result["status"] == "consistency_conflict") is (side < 0)


@pytest.mark.parametrize("side", (-1, 0, 1))
def test_actual_width_ceiling_is_closed_and_source_inclusive(side):
    center = Q(-2, 10**29)
    reference_width = WIDTH - 2 * SOURCE + side * SMALL
    reference = center - reference_width / 2, center + reference_width / 2
    result = owner.assess_neighbor_forward_bounds(reference)
    assert result["actual_width"] == WIDTH + side * SMALL
    assert result["width_within_allowance"] is (side <= 0)
    assert result["decision"]["null_excluded"]
    assert (result["status"] == "numerically_unresolved") is (side > 0)
    assert result["all_conditions_met"] is (side <= 0)


@pytest.mark.parametrize("noise_multiplier", (4, 8))
@pytest.mark.parametrize("side", (-1, 0, 1))
def test_sign_and_independent_additive_thresholds_are_strict(noise_multiplier, side):
    # Place the transported upper endpoint exactly at each noise threshold.
    point = -noise_multiplier * DELTA - SOURCE - side * SMALL
    result = owner.assess_neighbor_forward_bounds((point, point))
    decision = result["decision"]
    assert (
        decision["recorded_bounds"][1] == (4 - noise_multiplier) * DELTA - side * SMALL
    )
    assert result["additive_recorded_bounds"] == (-4 * DELTA, 4 * DELTA)
    if noise_multiplier == 4:
        assert decision["recorded_sign_margin"] == side * SMALL
        assert decision["recorded_sign"] is (side > 0)
    else:
        assert decision["null_separation_margin"] == side * SMALL
        assert decision["null_excluded"] is (side > 0)
    # A useful sign cannot bypass an incompatible analytic prediction.
    assert result["status"] == "consistency_conflict"
    assert not result["all_conditions_met"]


def test_unresolved_discrimination_is_not_a_numerical_failure(monkeypatch):
    policy = owner.neighbor_forward_policy()
    monkeypatch.setattr(
        owner,
        "neighbor_forward_policy",
        lambda: policy | {"readout_error_bound": Q(3, 10**30)},
    )
    result = owner.assess_neighbor_forward_bounds((Q(-2, 10**29),) * 2)
    assert result["nominal_prediction_overlap"] and result["width_within_allowance"]
    assert (
        result["decision"]["recorded_sign"] and not result["decision"]["null_excluded"]
    )
    assert result["status"] == "discrimination_unresolved"
    assert not result["all_conditions_met"]


def test_missing_observation_stays_unavailable_with_only_static_bounds():
    result = owner.assess_neighbor_forward_bounds(None)
    assert all(
        result[key] is None
        for key in (
            "reference_bounds",
            "actual_bounds",
            "reference_width",
            "actual_width",
            "nominal_prediction_overlap",
            "actual_prediction_overlap",
            "width_within_allowance",
            "decision",
        )
    )
    assert result["source_error_upper_bound"] == SOURCE
    assert result["nominal_prediction_bounds"] is not None
    assert result["actual_prediction_bounds"] is not None
    assert result["status"] == "unavailable" and not result["all_conditions_met"]


@pytest.mark.parametrize("complete", (False, True))
def test_report_consumer_reconstructs_selectors_and_ignores_cached_verdict(
    complete, monkeypatch
):
    reference = I(Q(-2, 10**29))
    evidence = SimpleNamespace(
        complete=complete, mixed_bounds=reference, raw_mixed_bounds=I(1)
    )
    report = SimpleNamespace(
        status="fabricated_pass", mixed_readout_bounds=I(100), all_conditions_met=True
    )
    calls = []

    def reconstruct(received, admitted, **selectors):
        assert received is report
        assert selectors == {
            "first_probe_node": 4,
            "second_probe_node": 22,
            "readout_node": 13,
        }
        form, phase, a, b, delay, total, step, order, cap = admitted
        expected = owner.neighbor_forward_sources()
        assert form == (I(0),) * 27
        assert phase == tuple(I(*pair) for pair in expected["reference_phase_bounds"])
        assert (a, b, delay, total, step, order, cap) == (
            PULSE,
            PULSE,
            0,
            H,
            Q(1, 128),
            16,
            64,
        )
        calls.append(True)
        return evidence

    monkeypatch.setattr(owner, "_reconstruct_class_readout", reconstruct)
    result = owner.assess_neighbor_forward_report(report)
    assert len(calls) == 1 and result["evidence"] is evidence
    expected_pair = (reference.lo, reference.hi) if complete else None
    assert result["comparison"] == owner.assess_neighbor_forward_bounds(expected_pair)
    assert result["all_conditions_met"] is complete
    assert result["status"] == ("all_conditions_met" if complete else "unavailable")


def test_invalid_report_evidence_cannot_fall_back_to_cached_passing_fields(monkeypatch):
    def reject(*args, **kwargs):
        raise ValueError("source or retained step mismatch")

    monkeypatch.setattr(owner, "_reconstruct_class_readout", reject)
    with pytest.raises(ValueError, match="retained step"):
        owner.assess_neighbor_forward_report(
            SimpleNamespace(status="admitted", all_conditions_met=True)
        )
