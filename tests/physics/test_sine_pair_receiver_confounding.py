"""Exact fine-law controls for a finite constitutive ambiguity certificate."""

from fractions import Fraction as Q

import pytest

from tests.physics._sine_pair_oracles import (
    EDGES,
    NEIGHBORS,
    ONE,
    ORIENTATIONS,
    ZERO,
    _conj,
    _fine_jets,
    _independent_current_derivative_bounds,
    _independent_receiver_remainder,
    _integrated_current_remainder,
    _mul,
    _preparation,
    _receiver_jets,
)
from tnfr.physics.relational_sine_scale import assess_sine_pair_receiver_confounding
from tnfr.sdk import export_to_json, relational_report_to_dict
from tnfr.utils.io import json_loads

ROTATION = (Q(399, 401), Q(40, 401))
HORIZON = Q(1, 20)

EPSILON_UPPER = Q(1, 10)


def _assess(rotation=ROTATION, horizon=HORIZON, epsilon=EPSILON_UPPER):
    return assess_sine_pair_receiver_confounding(
        phase_rotation=rotation, horizon_tau=horizon, epsilon_upper=epsilon
    )


@pytest.fixture(scope="module")
def frozen_confounding():
    return _assess()


@pytest.mark.parametrize("rotation", (ROTATION, (Q(3, 5), Q(4, 5)), (Q(0), Q(1))))
def test_each_endpoint_uses_the_original_cubic_fine_rows(rotation):
    report = _assess(rotation=rotation)
    assert report.coefficient_endpoints == (Q(0), EPSILON_UPPER)
    for index, epsilon in enumerate(report.coefficient_endpoints):
        coefficients = _receiver_jets(rotation, ORIENTATIONS[0], epsilon)
        assert coefficients[0] == coefficients[2] == coefficients[4] == 0
        assert report.orientation_a_linear_coefficients[index] == coefficients[1]
        assert report.orientation_a_cubic_coefficients[index] == coefficients[3]
        assert report.orientation_a_centers[index] == (
            coefficients[1] * HORIZON + coefficients[3] * HORIZON**3
        )
    reference = _receiver_jets(rotation, ORIENTATIONS[1], Q(0))
    assert report.orientation_b_reference_linear_coefficient == reference[1]
    assert report.orientation_b_reference_cubic_coefficient == reference[3]
    assert (
        report.orientation_b_reference_center
        == reference[1] * HORIZON + reference[3] * HORIZON**3
    )
    assert report.comparison_initial_phasors == tuple(
        _preparation(rotation, orientation) for orientation in ORIENTATIONS
    )


@pytest.mark.parametrize("epsilon", (Q(1, 100), EPSILON_UPPER, Q(1)))
def test_current_derivative_bounds_follow_collected_sine_cosine_polynomials(epsilon):
    report = _assess(epsilon=epsilon)
    fields = (
        "current_value_upper_bounds",
        "current_first_derivative_upper_bounds",
        "current_second_derivative_upper_bounds",
        "current_third_derivative_upper_bounds",
    )
    for index, coefficient in enumerate((Q(0), epsilon)):
        for name, expected in zip(
            fields, _independent_current_derivative_bounds(coefficient)
        ):
            assert getattr(report, name)[index] == expected


@pytest.mark.parametrize("horizon", (HORIZON, Q(2)))
def test_nominal_remainders_count_all_pressure_chain_rule_terms_and_edges(horizon):
    report = _assess(horizon=horizon)
    expected = tuple(
        _independent_receiver_remainder(horizon, epsilon)
        for epsilon in (Q(0), EPSILON_UPPER)
    )
    assert report.orientation_a_remainder_bounds == expected
    assert report.orientation_b_reference_remainder == expected[0]
    # At epsilon zero, the full-pressure and pair-current proofs agree.
    assert expected[0] == 2 * _integrated_current_remainder(horizon)
    for center, bound, error in zip(
        report.endpoint_difference_centers, report.endpoint_difference_bounds, expected
    ):
        radius = error + expected[0]
        assert bound.lo <= center - radius <= center + radius <= bound.hi


def test_frozen_strict_sign_reversal_certifies_an_interior_collision(
    frozen_confounding,
):
    report = frozen_confounding
    assert report.endpoint_difference_signs == (1, -1)
    assert report.endpoint_difference_bounds[0].lo > 0
    assert report.endpoint_difference_bounds[1].hi < 0
    assert report.status == "certified_collision_exists"
    assert report.unavailable_reason is None
    assert report.collision_parameter_open_interval == (Q(0), EPSILON_UPPER)
    for a, difference in zip(
        report.orientation_a_centers, report.endpoint_difference_centers
    ):
        assert difference == a - report.orientation_b_reference_center


def test_conjugation_reverses_both_endpoint_signs_but_keeps_the_existence_result(
    frozen_confounding,
):
    reflected = _assess(rotation=_conj(ROTATION))
    assert reflected.endpoint_difference_centers == tuple(
        -value for value in frozen_confounding.endpoint_difference_centers
    )
    assert reflected.endpoint_difference_bounds == tuple(
        -value for value in frozen_confounding.endpoint_difference_bounds
    )
    assert reflected.endpoint_difference_signs == (-1, 1)
    assert reflected.status == "certified_collision_exists"
    assert reflected.collision_parameter_open_interval == (Q(0), EPSILON_UPPER)


def test_insufficient_parameter_range_is_unavailable_not_a_no_collision_verdict():
    report = _assess(epsilon=Q(1, 100))
    assert report.endpoint_difference_bounds[0].lo > 0
    assert 0 in report.endpoint_difference_bounds[1]
    assert report.endpoint_difference_signs == (1, 0)
    assert report.status == "unavailable"
    assert (
        report.unavailable_reason
        == "endpoint_contrasts_do_not_have_strict_opposite_signs"
    )
    assert report.collision_parameter_open_interval is None


@pytest.mark.parametrize("rotation", ((Q(1), Q(0)), (Q(-1), Q(0))))
def test_zero_nominal_contrast_is_not_a_strict_endpoint_bracket(rotation):
    report = _assess(rotation=rotation)
    assert report.endpoint_difference_centers == (Q(0), Q(0))
    assert report.endpoint_difference_signs == (0, 0)
    assert report.status == "unavailable"
    assert report.collision_parameter_open_interval is None
    # The sufficient strict-bracket reader does not turn an inconclusive
    # interval into either an equilibrium or a no-collision assertion.
    for orientation in ORIENTATIONS:
        x, z = _fine_jets(
            _preparation(rotation, orientation), order=1, epsilon=EPSILON_UPPER
        )
        assert all(node[1] == ZERO for node in z)
        if rotation == ONE or orientation == ORIENTATIONS[0]:
            assert all(node[1] == 0 for node in x)
        else:
            # At q=-1 the imaginary selected pair still moves internally.
            assert any(node[1] != 0 for node in x)


def test_tiny_exact_coefficient_range_remains_distinct_but_cannot_supply_a_false_bracket():
    tiny = Q(1, 2**1100)
    report = _assess(epsilon=tiny)
    assert report.epsilon_upper == tiny > 0
    assert report.current_value_upper_bounds[1] > report.current_value_upper_bounds[0]
    assert (
        report.orientation_a_linear_coefficients[1]
        != report.orientation_a_linear_coefficients[0]
    )
    assert report.endpoint_difference_signs == (1, 1)
    assert report.status == "unavailable"


def test_tiny_exact_horizon_keeps_distinct_centers_but_outward_zero_bounds_abstain():
    tiny = Q(1, 2**1100)
    report = _assess(horizon=tiny)
    assert report.horizon_tau == tiny > 0
    assert report.endpoint_difference_centers[0] > 0
    assert report.endpoint_difference_centers[1] < 0
    assert report.endpoint_difference_signs == (0, 0)
    assert report.status == "unavailable"
    assert report.collision_parameter_open_interval is None


@pytest.mark.parametrize("orientation", (*ORIENTATIONS, (Q(3, 5), Q(4, 5))))
def test_stationary_cancellation_control_survives_the_complete_cubic_law(orientation):
    phasors = _preparation(ROTATION, orientation, control=True)
    x, z = _fine_jets(phasors, order=1, epsilon=EPSILON_UPPER)
    assert all(node[1] == 0 for node in x)
    assert all(node[1] == ZERO for node in z)


def test_the_alternative_conserves_its_own_storage_but_not_cosine_storage():
    forms = tuple(Q((i + 1) ** 2, 37) for i in range(10))
    phasors = _preparation(ROTATION, ORIENTATIONS[0])
    x, _ = _fine_jets(phasors, order=1, forms=forms, epsilon=EPSILON_UPPER)
    phase_rates = tuple(
        forms[i] - sum((forms[j] for j in NEIGHBORS[i]), Q(0)) / 4 for i in range(10)
    )
    form_work = sum(
        ((forms[i] - forms[j]) * (x[i][1] - x[j][1]) for i, j in EDGES), Q(0)
    )
    cubic_phase_work, sine_phase_work = Q(0), Q(0)
    for i, j in EDGES:
        cosine, sine = _mul(_conj(phasors[i]), phasors[j])
        potential = 1 - cosine + EPSILON_UPPER * (1 - cosine) ** 2 * (2 + cosine) / 3
        assert potential >= 0
        # Differentiate the actual supplied potential with respect to gap.
        derivative = sine * (1 + EPSILON_UPPER * (1 - cosine**2))
        assert derivative == sine + EPSILON_UPPER * sine**3
        cubic_phase_work += derivative * (phase_rates[j] - phase_rates[i])
        sine_phase_work += sine * (phase_rates[j] - phase_rates[i])
    assert form_work != 0
    assert form_work + cubic_phase_work == 0
    assert form_work + sine_phase_work != 0


def test_consensus_local_response_agrees_while_nonlinear_response_differs():
    phasors = (ONE,) * 10
    forms = tuple(Q(i * i - 3 * i, 31) for i in range(10))
    baseline_x, _ = _fine_jets(phasors, forms=forms)
    alternative_x, _ = _fine_jets(phasors, forms=forms, epsilon=EPSILON_UPPER)
    assert any(row[2] != 0 for row in baseline_x)
    assert all(left[:4] == right[:4] for left, right in zip(baseline_x, alternative_x))
    assert any(left[4] != right[4] for left, right in zip(baseline_x, alternative_x))


@pytest.mark.parametrize("parameter", ("horizon", "epsilon"))
@pytest.mark.parametrize(
    "bad", (0, -1, Q(-1, 2**1100), True, False, float("nan"), float("inf"), "0.1", 1j)
)
def test_scalar_parameters_require_finite_positive_nonboolean_values(parameter, bad):
    with pytest.raises((TypeError, ValueError)):
        _assess(**{parameter: bad})


@pytest.mark.parametrize(
    "rotation",
    ((True, 0), (float("inf"), 0), (1, Q(1, 2**1100)), (0.6, 0.8), {0, 1}, (1,)),
)
def test_declared_rotation_requires_ordered_exact_unit_phasors(rotation):
    with pytest.raises((TypeError, ValueError)):
        _assess(rotation=rotation)


def test_sdk_preserves_endpoint_evidence_and_unavailable_collision_parameter(
    tmp_path, frozen_confounding
):
    for name, report in (
        ("certificate", frozen_confounding),
        ("unavailable", _assess(epsilon=Q(1, 100))),
    ):
        direct = report.to_dict()
        assert direct["schema"] == "tnfr.sine-pair-receiver-confounding.v1"
        projected = relational_report_to_dict(report)
        assert projected["report"] == direct["report"]
        assert projected["report"]["epsilon_upper"] == {
            "numerator": report.epsilon_upper.numerator,
            "denominator": report.epsilon_upper.denominator,
        }
        if name == "unavailable":
            assert projected["report"]["collision_parameter_open_interval"] is None
        target = tmp_path / (name + ".json")
        export_to_json(projected, target)
        assert json_loads(target.read_text(encoding="utf-8")) == projected
