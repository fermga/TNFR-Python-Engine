"""Independent source, width and observation algebra before class comparison.

No complete sine response, amplitude coefficient or archived helper executes.
The heat example demonstrates interval wrapping, not a full-sine impossibility
result or a prediction of the new producer's achieved width.
"""

import itertools
from fractions import Fraction as Q
from math import factorial

import pytest

from tnfr.physics import relational_sine_class_cubic_response as cubic
from tnfr.physics import relational_sine_class_readout as readout
from tnfr.research import sine_class_comparison_protocol as protocol


@pytest.fixture(autouse=True)
def no_scientific_evaluation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("protocol algebra must not evaluate scientific responses")

    monkeypatch.setattr(cubic, "_class_cubic_coefficients", forbidden)
    monkeypatch.setattr(readout, "bound_sine_class_four_history_readout", forbidden)
    monkeypatch.setattr(readout, "_full_sine_field", forbidden)
    monkeypatch.setattr(readout, "validated_box_taylor_step", forbidden)


def test_nominal_references_and_actual_covers_keep_both_classes_and_correlations():
    source = protocol.canonical_comparison_sources()
    eps = Q(1, 10**32)
    assert source["classes"] == ((1, 1, 1), (1, 2, 1))
    pi_lower, pi_upper = source["pi_bounds"]
    assert Q(3) < pi_lower < pi_upper < Q(22, 7)
    assert pi_upper - pi_lower < Q(1, 2**250)
    # This nonzero error family has exact component sums and stays strictly
    # inside each original Euclidean ball. It is not reset to the reference.
    deviation = tuple(
        eps / 2 if j == 4 else -eps / 2 if j == 0 else Q(0) for j in range(9)
    )
    assert sum(deviation) == 0
    assert sum(value**2 for value in deviation) < eps**2
    for model, classes in enumerate(source["classes"]):
        reference_x = source["reference_form_bounds"][model]
        reference_theta = source["reference_phase_bounds"][model]
        actual_x = source["actual_form_bounds"][model]
        actual_theta = source["actual_phase_bounds"][model]
        assert (
            len(reference_x)
            == len(reference_theta)
            == len(actual_x)
            == len(actual_theta)
            == 27
        )
        for component, winding in enumerate(classes):
            for local in range(9):
                node = 9 * component + local
                coefficient = Q(2 * winding * (local - 4), 9)
                candidates = (coefficient * pi_lower, coefficient * pi_upper)
                target_bounds = min(candidates), max(candidates)
                assert reference_x[node] == (0, 0)
                assert reference_theta[node] == target_bounds
                assert actual_x[node] == (-eps, eps)
                assert actual_theta[node] == (
                    target_bounds[0] - eps,
                    target_bounds[1] + eps,
                )
                for target in candidates:
                    assert (
                        actual_theta[node][0]
                        <= target + deviation[local]
                        <= actual_theta[node][1]
                    )
                assert actual_x[node][0] <= deviation[local] <= actual_x[node][1]
        # Exact component zero sums do not force the joined degree-weighted
        # mean to vanish: contacts add weights (1,2,1) at the central nodes.
        extra_charge = sum(weight * deviation[4] for weight in (1, 2, 1))
        assert extra_charge == 2 * eps != 0

    # The rectangular corner is included numerically but is not an acquired
    # source with the required component sum or Euclidean norm. Phase corners
    # also include target-pi enclosure uncertainty; they need not be epsilon
    # from the exact common target for every independent corner selection.
    assert 9 * eps != 0
    assert 9 * eps**2 > eps**2
    assert (
        source["actual_phase_bounds"][1][0][1]
        - source["reference_phase_bounds"][1][0][0]
        > eps
    )


def test_packet_changes_only_the_admitted_scale_and_uses_one_joint_attempt_budget():
    source = protocol.canonical_comparison_sources()
    packet = protocol.comparison_producer_inputs()
    assert set(packet) == {
        "initial_form_bounds",
        "initial_phase_bounds",
        "first_probe_amplitude",
        "second_probe_amplitude",
        "delay",
        "total_duration",
        "time_step",
        "order",
        "max_steps",
    }
    assert packet["initial_form_bounds"] == source["reference_form_bounds"]
    assert packet["initial_phase_bounds"] == source["reference_phase_bounds"]
    assert (
        packet["first_probe_amplitude"]
        == packet["second_probe_amplitude"]
        == Q(7, 10000)
    )
    assert packet["first_probe_amplitude"] / Q(1, 2000) == Q(7, 5)
    assert packet["delay"] == 1 and packet["total_duration"] == 2
    assert packet["time_step"] == Q(1, 128) and packet["order"] == 16
    segment_lengths = (packet["delay"],) * 2 + (
        packet["total_duration"] - packet["delay"],
    ) * 4
    attempts_one_class = sum(
        duration / packet["time_step"] for duration in segment_lengths
    )
    assert attempts_one_class == 768
    assert packet["max_steps"] == 2 * attempts_one_class == 1536


def test_full_source_transfer_has_matching_jumps_and_is_added_only_once():
    policy = protocol.comparison_observation_policy()
    eps, g, total = (
        policy["endpoint_radius"],
        policy["gamma_upper"],
        policy["total_duration"],
    )
    assert (eps, g, total) == (Q(1, 10**32), Q(1, 3000), Q(2))
    denominator = 1 - 2 * g * total
    endpoint_error = eps / denominator
    # Heat contraction and real sine Lipschitz give X<=eps+2g integral Y,
    # Y<=eps+2g integral X. A uniform supremum is therefore <=eps/denominator.
    assert endpoint_error == eps + 2 * g * total * endpoint_error
    actual_before, reference_before, pulse = Q(3, 17), Q(5, 29), Q(7, 10000)
    assert (actual_before + pulse) - (
        reference_before + pulse
    ) == actual_before - reference_before
    # The same cancellation holds at the delayed event after each trajectory
    # has carried its own full state; it does not identify the preevent states.
    weights = (1, -1, -1, 1, -1, 1, 1, -1)
    worst_endpoint_errors = tuple(weight * endpoint_error for weight in weights)
    source_error = sum(
        weight * error for weight, error in zip(weights, worst_endpoint_errors)
    )
    assert source_error == 8 * endpoint_error
    reference = (Q(-1925, 10**32), Q(-1924, 10**32))
    result = protocol.assess_reference_contrast(reference, **policy)
    assert result["source_error_upper_bound"] == source_error
    assert result["actual_bounds"] == (
        reference[0] - source_error,
        reference[1] + source_error,
    )
    assert result["reference_width"] == reference[1] - reference[0]
    assert result["actual_width"] == result["reference_width"] + 2 * source_error
    assert result["decision"].true_bounds == result["actual_bounds"]
    assert result["decision"].recorded_bounds == (
        result["actual_bounds"][0] - 8 * policy["readout_error_bound"],
        result["actual_bounds"][1] + 8 * policy["readout_error_bound"],
    )
    # No cubic or higher-amplitude error enters an independently validated
    # full-law reference. Those errors belong to the separate prediction.
    assert result["status"] == "discrimination_certified"


def test_scalar_heat_wrapping_survives_small_step_without_reference_transport():
    eps, h, steps = Q(1, 10**32), Q(1, 128), 128
    # For z'=-2z and symmetric interval [-r,r], natural interval coefficients
    # have radii r*2^j/j!: all signs disappear. This is a scalar algebraic
    # counterexample, not execution of either prospective class history.
    multiplier = sum((2 * h) ** degree / factorial(degree) for degree in range(17))
    prefix_growth = multiplier**steps
    suffix_increment_radius = eps * prefix_growth * (prefix_growth - 1)
    cross_class_radius = 8 * suffix_increment_radius
    assert cross_class_radius > Q(37, 10**31)
    # Prefix cancellation helps but does not cancel independently propagated
    # suffix interval uncertainty. Refining a fixed-order step approaches
    # 8eps*(exp(4)-exp(2)), whereas the true decay contracts the source radius.
    policy = protocol.comparison_observation_policy()
    prediction_margin = (
        -policy["prediction_open_bounds"][1] - 16 * policy["readout_error_bound"]
    )
    assert cross_class_radius > prediction_margin > 0
    assert (
        8 * eps / (1 - 2 * policy["gamma_upper"] * policy["total_duration"])
        < prediction_margin / 30
    )
    # This motivates retaining source dependence analytically; it proves no
    # lower bound on the actual complete-sine solver's final enclosure width.


def test_eight_original_errors_and_noisy_null_have_distinct_exact_allowances():
    delta = Q(1, 64)
    weights = (1, -1, -1, 1, -1, 1, 1, -1)
    statistics = {
        sum(weight * error for weight, error in zip(weights, errors))
        for errors in itertools.product((-delta, delta), repeat=8)
    }
    assert min(statistics) == -8 * delta and max(statistics) == 8 * delta
    # At true contrast -16delta the separately noisy zero alternative still
    # touches. A recorded sign alone does not exclude its record set.
    assert -16 * delta + max(statistics) == min(statistics)
    assert -17 * delta + max(statistics) < min(statistics)
    assert -9 * delta + max(statistics) < 0
    assert -9 * delta + max(statistics) > min(statistics)


@pytest.mark.parametrize(
    ("upper_multiplier", "expected"),
    (
        (0, (False, False, False)),
        (-1, (True, False, False)),
        (-8, (True, False, False)),
        (-9, (True, True, False)),
        (-16, (True, True, False)),
        (-17, (True, True, True)),
    ),
)
def test_strict_true_recorded_and_noisy_null_thresholds(upper_multiplier, expected):
    policy = protocol.comparison_observation_policy()
    policy["prediction_open_bounds"] = (-1, 1)
    policy["reference_prediction_open_bounds"] = (-1, 1)
    delta = policy["readout_error_bound"]
    source_error = (
        8
        * policy["endpoint_radius"]
        / (1 - 2 * policy["gamma_upper"] * policy["total_duration"])
    )
    reference_point = upper_multiplier * delta - source_error
    result = protocol.assess_reference_contrast(
        (reference_point, reference_point), **policy
    )
    decision = result["decision"]
    assert result["actual_bounds"][1] == upper_multiplier * delta
    assert (
        decision.true_sign,
        decision.recorded_sign,
        decision.null_excluded,
    ) == expected


def test_actual_width_budget_is_independent_and_uses_closed_equality():
    policy = protocol.comparison_observation_policy()
    policy["prediction_open_bounds"] = (-1, 1)
    policy["reference_prediction_open_bounds"] = (-1, 1)
    source = (
        8
        * policy["endpoint_radius"]
        / (1 - 2 * policy["gamma_upper"] * policy["total_duration"])
    )
    allowance = policy["width_allowance"]
    reference_width = allowance - 2 * source
    assert 0 < reference_width < allowance
    upper = -20 * policy["readout_error_bound"]
    admitted = protocol.assess_reference_contrast(
        (upper - reference_width, upper), **policy
    )
    assert admitted["actual_width"] == allowance
    assert admitted["width_within_allowance"]
    assert admitted["status"] == "discrimination_certified"
    enlarged = protocol.assess_reference_contrast(
        (upper - reference_width - Q(1, 10**40), upper), **policy
    )
    assert enlarged["decision"].null_excluded
    assert enlarged["theorem_open_interval_overlap"]
    assert not enlarged["width_within_allowance"]
    assert enlarged["status"] == "resolution_not_certified"


@pytest.mark.parametrize(
    ("reference", "overlap", "contained"),
    (
        ((-5, -4), False, False),
        ((-3, -2), False, False),
        ((-4, -4), False, False),
        ((-3, -3), False, False),
        ((-4, -3), True, False),
        ((Q(-7, 2), Q(-13, 4)), True, True),
    ),
)
def test_open_prediction_is_only_checked_never_intersected(
    reference, overlap, contained
):
    policy = protocol.comparison_observation_policy()
    policy.update(
        endpoint_radius=0,
        reference_prediction_open_bounds=(-4, -3),
        prediction_open_bounds=(-4, -3),
        width_allowance=10,
    )
    result = protocol.assess_reference_contrast(reference, **policy)
    assert result["actual_bounds"] == reference
    assert result["reference_theorem_open_interval_overlap"] is overlap
    assert result["reference_theorem_open_interval_contains_band"] is contained
    assert result["theorem_open_interval_overlap"] is overlap
    assert result["theorem_open_interval_contains_actual_band"] is contained
    if not overlap:
        assert result["status"] == "consistency_conflict"


def test_missing_reference_retains_unavailability_without_passing_flags():
    result = protocol.assess_reference_contrast(
        None, **protocol.comparison_observation_policy()
    )
    assert result["status"] == "unavailable"
    for key in (
        "reference_bounds",
        "actual_bounds",
        "reference_width",
        "actual_width",
        "decision",
        "width_within_allowance",
        "reference_theorem_open_interval_overlap",
        "reference_theorem_open_interval_contains_band",
        "theorem_open_interval_overlap",
        "theorem_open_interval_contains_actual_band",
    ):
        assert result[key] is None
    assert result["source_error_upper_bound"] > 0


@pytest.mark.parametrize("invalid", (True, False, float("nan"), float("inf"), 0.5))
def test_policy_and_reference_require_exact_finite_nonboolean_primitives(invalid):
    policy = protocol.comparison_observation_policy()
    for key in (
        "endpoint_radius",
        "gamma_upper",
        "total_duration",
        "readout_error_bound",
        "width_allowance",
    ):
        malformed = dict(policy, **{key: invalid})
        with pytest.raises((TypeError, ValueError)):
            protocol.assess_reference_contrast(None, **malformed)
    for position in (0, 1):
        reference = [Q(-2), Q(-1)]
        reference[position] = invalid
        with pytest.raises((TypeError, ValueError)):
            protocol.assess_reference_contrast(reference, **policy)
        for key in ("reference_prediction_open_bounds", "prediction_open_bounds"):
            prediction = list(policy[key])
            prediction[position] = invalid
            with pytest.raises((TypeError, ValueError)):
                protocol.assess_reference_contrast(
                    None, **dict(policy, **{key: prediction})
                )


@pytest.mark.parametrize("conflicting_reference", (True, False))
def test_either_prediction_conflict_is_retained_without_narrowing(
    conflicting_reference,
):
    policy = protocol.comparison_observation_policy()
    reference = (Q(-1, 2), Q(-1, 2))
    policy.update(
        endpoint_radius=0,
        reference_prediction_open_bounds=(0, 1) if conflicting_reference else (-1, 1),
        prediction_open_bounds=(-1, 1) if conflicting_reference else (0, 1),
        width_allowance=1,
    )
    result = protocol.assess_reference_contrast(reference, **policy)
    assert result["reference_bounds"] == result["actual_bounds"] == reference
    assert (
        result["reference_theorem_open_interval_overlap"] is not conflicting_reference
    )
    assert result["theorem_open_interval_overlap"] is conflicting_reference
    assert result["status"] == "consistency_conflict"


@pytest.mark.parametrize(
    "key", ("reference_prediction_open_bounds", "prediction_open_bounds")
)
@pytest.mark.parametrize("bad_pair", ((Q(1), Q(0)), (Q(1), Q(1))))
def test_each_open_prediction_requires_strict_order_even_without_a_response(
    key, bad_pair
):
    policy = dict(protocol.comparison_observation_policy(), **{key: bad_pair})
    with pytest.raises(ValueError):
        protocol.assess_reference_contrast(None, **policy)
