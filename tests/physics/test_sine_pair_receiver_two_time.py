"""Two-time coefficient exclusion from independent complete fine-law algebra."""

from fractions import Fraction as Q
from typing import get_type_hints

import pytest

from tests.physics._sine_pair_oracles import (
    ORIENTATIONS,
    _independent_receiver_remainder,
    _preparation,
    _preparation_bound,
    _receiver_jets,
)
from tnfr.physics import relational_sine_pair as pair
from tnfr.physics import relational_sine_scale as scale
from tnfr.sdk import export_to_json, relational_report_to_dict
from tnfr.utils.io import json_loads

ROTATION = (Q(399, 401), Q(40, 401))

HORIZONS = (Q(1, 40), Q(1, 20))
UPPER = Q(1, 10)
ERROR = Q(1, 10**11)


def _assess(
    rotation=ROTATION,
    horizons=HORIZONS,
    upper=UPPER,
    form=ERROR,
    phase=ERROR,
    readout=ERROR,
):
    return pair.assess_sine_pair_receiver_two_time(
        phase_rotation=rotation,
        horizons_tau=horizons,
        epsilon_upper=upper,
        form_error_bound=form,
        phase_error_bound=phase,
        readout_error_bound=readout,
    )


def _contrast_coefficients(rotation):
    """Recover the coefficient polynomial from original-node Taylor jets."""
    reference = _receiver_jets(rotation, ORIENTATIONS[1], Q(0))
    samples = tuple(
        _receiver_jets(rotation, ORIENTATIONS[0], epsilon) for epsilon in range(3)
    )
    linear = tuple(sample[1] - reference[1] for sample in samples)
    cubic = tuple(sample[3] - reference[3] for sample in samples)
    assert linear[0] == 0 and linear[2] == 2 * linear[1]
    quadratic = (cubic[2] - 2 * cubic[1] + cubic[0]) / 2
    return -linear[1], (cubic[0], cubic[1] - cubic[0] - quadratic, quadratic)


def _affine_oracle(
    horizon, coefficients, upper=UPPER, form=ERROR, phase=ERROR, readout=ERROR
):
    """Build necessary inequalities from full-law jets and error majorants."""
    c, (d0, d1, d2) = coefficients
    remainders = tuple(
        _independent_receiver_remainder(horizon, epsilon) for epsilon in (0, upper)
    )
    preparations = tuple(
        _preparation_bound(horizon, epsilon, form, phase) for epsilon in (0, upper)
    )
    radii = tuple(
        error + remainders[0] + initial + preparations[0] + 2 * readout
        for error, initial in zip(remainders, preparations, strict=True)
    )
    chord = (radii[1] - radii[0]) / upper
    n = d0 * horizon**3 - radii[0]
    a = c * horizon - d1 * horizon**3 + chord
    u = d0 * horizon**3 + radii[0]
    b = c * horizon - (d1 + d2 * upper) * horizon**3 - chord
    return remainders, preparations, radii, chord, (n, a, u, b)


@pytest.fixture(scope="module")
def coefficients():
    return _contrast_coefficients(ROTATION)


@pytest.fixture(scope="module")
def frozen_two_time():
    return _assess()


@pytest.mark.parametrize("rotation", (ROTATION, (Q(3, 5), Q(4, 5))))
def test_coefficient_polynomial_is_the_complete_fine_receiver_jet(rotation):
    report = _assess(rotation=rotation)
    c, cubic = _contrast_coefficients(rotation)
    assert report.contrast_linear_epsilon_coefficient == c
    assert report.contrast_cubic_coefficients == cubic
    assert report.comparison_initial_phasors == tuple(
        _preparation(rotation, orientation) for orientation in ORIENTATIONS
    )
    assert report.control_initial_phasors == tuple(
        _preparation(rotation, orientation, control=True)
        for orientation in ORIENTATIONS
    )
    # An additional coefficient checks the independently interpolated polynomial.
    epsilon = Q(3, 37)
    alternative = _receiver_jets(rotation, ORIENTATIONS[0], epsilon)
    reference = _receiver_jets(rotation, ORIENTATIONS[1], Q(0))
    assert alternative[1] - reference[1] == -c * epsilon
    assert alternative[3] - reference[3] == sum(
        (value * epsilon**power for power, value in enumerate(cubic)), Q(0)
    )


@pytest.mark.parametrize("form,phase", ((0, 0), (ERROR, 0), (0, ERROR), (ERROR, ERROR)))
def test_affine_bounds_retain_both_complete_laws_and_separate_errors(
    coefficients, form, phase
):
    report = _assess(form=form, phase=phase)
    for index, horizon in enumerate(HORIZONS):
        remainder, preparation, radii, chord, (n, a, u, b) = _affine_oracle(
            horizon, coefficients, form=form, phase=phase
        )
        assert report.remainder_upper_bounds_by_time[index] == remainder
        assert report.preparation_error_bounds_by_time[index] == preparation
        assert report.reference_contrast_radii[index] == radii[0]
        assert report.uncertainty_chord_slopes[index] == chord
        assert report.lower_affine_intercepts[index] == n
        assert report.lower_affine_epsilon_slopes[index] == a
        assert report.upper_affine_intercepts[index] == u
        assert report.upper_affine_epsilon_slopes[index] == b
        if a > 0 and b > 0:
            bounds = report.necessary_coefficient_bounds[index]
            assert bounds.lo <= n / a <= u / b <= bounds.hi


@pytest.mark.parametrize("epsilon", (Q(0), UPPER / 3, UPPER * Q(7, 11), UPPER))
def test_chords_enclose_complete_law_bounds_between_coefficient_endpoints(
    coefficients, epsilon
):
    # Finite controls of the analytic convex-chord proof, not a grid certificate.
    c, (d0, d1, d2) = coefficients
    for horizon in HORIZONS:
        _, _, _, _, (n, a, u, b) = _affine_oracle(horizon, coefficients)
        center = (
            -c * epsilon * horizon + (d0 + d1 * epsilon + d2 * epsilon**2) * horizon**3
        )
        radius = sum(
            _independent_receiver_remainder(horizon, law)
            + _preparation_bound(horizon, law, ERROR, ERROR)
            + ERROR
            for law in (Q(0), epsilon)
        )
        assert n - a * epsilon <= center - radius
        assert center + radius <= u - b * epsilon


def test_frozen_joint_certificate_uses_one_shared_coefficient(frozen_two_time):
    report = frozen_two_time
    first, second = report.necessary_coefficient_bounds
    assert report.coefficient_endpoints == (0, UPPER)
    assert 0 < first.lo < first.hi < second.lo < second.hi < UPPER
    assert report.status == "certified_disjoint"
    assert report.unavailable_reason is None
    assert report.coefficient_gap_lower_bound == second.lo - first.hi > 0
    assert report.horizons_tau == HORIZONS
    assert report.nodes == tuple(range(10))
    assert report.receiver_nodes == (2, 3) and report.receiver_pair_index == 1
    assert report.initial_forms == (Q(0),) * 10
    assert report.held_capacities == (Q(1),) * 10
    assert report.clock == "tau=t/pi"
    assert (
        report.form_error_bound
        == report.phase_error_bound
        == report.readout_error_bound
        == ERROR
    )
    # Allowing a different epsilon per reading leaves each necessary interval
    # nonempty. Their exclusion concerns the same constant coefficient only;
    # membership in either outer interval does not prove a realizable collision.
    assert first.lo <= first.midpoint <= first.hi
    assert second.lo <= second.midpoint <= second.hi
    assert first.midpoint != second.midpoint
    # The existing full-law existence proof does certify an actual nominal
    # collision separately at each horizon. The two-time theorem excludes
    # using one coefficient for both, even after admitting the error budgets.
    for horizon in HORIZONS:
        single_time = pair.assess_sine_pair_receiver_confounding(
            phase_rotation=ROTATION, horizon_tau=horizon, epsilon_upper=UPPER
        )
        assert single_time.status == "certified_collision_exists"
        assert single_time.endpoint_difference_signs == (1, -1)


def test_readout_witness_cancels_coefficient_in_the_necessary_inequalities(
    frozen_two_time,
):
    report = frozen_two_time
    weights = report.joint_readout_weights
    assert weights[0] < 0 < weights[1]
    assert sum(map(abs, weights)) == 1
    n = report.lower_affine_intercepts[1]
    a = report.lower_affine_epsilon_slopes[1]
    u = report.upper_affine_intercepts[0]
    b = report.upper_affine_epsilon_slopes[0]
    assert weights[0] * b + weights[1] * a == 0
    exact_margin = weights[0] * u + weights[1] * n
    assert 0 < report.joint_readout_gap_lower_bound <= exact_margin
    for epsilon in (Q(0), UPPER / 2, UPPER):
        assert (
            weights[0] * (u - b * epsilon) + weights[1] * (n - a * epsilon)
            == exact_margin
        )


def test_readout_noise_broadens_intervals_without_changing_law(
    coefficients, frozen_two_time
):
    noisy = _assess(readout=Q(1, 100))
    assert (
        noisy.contrast_cubic_coefficients == frozen_two_time.contrast_cubic_coefficients
    )
    assert (
        noisy.preparation_error_bounds_by_time
        == frozen_two_time.preparation_error_bounds_by_time
    )
    assert noisy.status == "unavailable"
    assert (
        noisy.unavailable_reason == "necessary_coefficient_intervals_overlap_or_touch"
    )
    assert noisy.coefficient_gap_lower_bound < 0
    assert noisy.joint_readout_weights is None
    assert noisy.joint_readout_gap_lower_bound is None


def test_exactly_touching_necessary_intervals_cannot_certify(coefficients):
    rows = tuple(
        _affine_oracle(h, coefficients, form=0, phase=0, readout=0)[-1]
        for h in HORIZONS
    )
    n2, a2, _, _ = rows[1]
    _, _, u1, b1 = rows[0]
    error = (n2 * b1 - u1 * a2) / (2 * (a2 + b1))
    assert error > 0
    assert (n2 - 2 * error) / a2 == (u1 + 2 * error) / b1
    report = _assess(form=0, phase=0, readout=error)
    first, second = report.necessary_coefficient_bounds
    # Outward materialization may widen the exact touching point, never separate it.
    assert second.lo <= first.hi
    assert report.coefficient_gap_lower_bound <= 0
    assert report.status == "unavailable"
    assert (
        report.unavailable_reason == "necessary_coefficient_intervals_overlap_or_touch"
    )


def test_nonpositive_affine_slope_is_unavailable_before_division():
    report = _assess(horizons=(Q(1, 8), Q(1, 4)))
    assert min(report.upper_affine_epsilon_slopes) <= 0
    assert report.necessary_coefficient_bounds is None
    assert report.coefficient_gap_lower_bound is None
    assert report.joint_readout_weights is None
    assert report.status == "unavailable"
    assert report.unavailable_reason == "coefficient_interval_slope_not_positive"


def test_tiny_exact_horizons_abstain_after_outward_materialization():
    tiny = Q(1, 2**1100)
    report = _assess(horizons=(tiny, 2 * tiny), form=0, phase=0, readout=0)
    assert report.horizons_tau == (tiny, 2 * tiny)
    assert all(value > 0 for value in report.lower_affine_intercepts)
    first, second = report.necessary_coefficient_bounds
    assert first == second
    assert report.coefficient_gap_lower_bound <= 0
    assert report.status == "unavailable"


def test_coefficient_certificate_does_not_require_resolved_optional_readout_gap():
    horizon = Q(1, 2**50)
    report = _assess(horizons=(horizon, 2 * horizon), form=0, phase=0, readout=0)
    first, second = report.necessary_coefficient_bounds
    assert report.coefficient_gap_lower_bound == second.lo - first.hi > 0
    assert report.status == "certified_disjoint"
    assert report.joint_readout_weights is not None
    # The coefficient gap and weighted form gap have different scales and units.
    # This latter positive exact margin lies below the outward dyadic128 grid.
    exact_margin = (
        report.joint_readout_weights[0] * report.upper_affine_intercepts[0]
        + report.joint_readout_weights[1] * report.lower_affine_intercepts[1]
    )
    assert 0 < exact_margin < Q(1, 2**128)
    assert report.joint_readout_gap_lower_bound == 0


def test_fraction_subclasses_and_tiny_budgets_are_preserved_before_arithmetic():
    class RationalInput(Q):
        pass

    tiny = RationalInput(1, 2**1100)
    report = _assess(form=tiny, phase=tiny, readout=tiny)
    assert (
        report.form_error_bound
        == report.phase_error_bound
        == report.readout_error_bound
        == tiny
    )
    assert type(report.form_error_bound) is Q
    assert all(
        value > tiny for row in report.preparation_error_bounds_by_time for value in row
    )


def test_exact_horizon_below_rational_amplification_boundary_is_not_rounded():
    maximum = Q(1, 4) - Q(1, 2**1100)  # epsilon_upper=1 gives L=4.
    report = _assess(horizons=(maximum / 2, maximum), upper=Q(1))
    assert report.horizons_tau[1] == maximum < Q(1, 4)
    assert report.status == "unavailable"


@pytest.mark.parametrize("field", ("form", "phase", "readout"))
@pytest.mark.parametrize(
    "bad", (-1, Q(-1, 2**1100), True, False, float("nan"), float("inf"), "0", 1j)
)
def test_error_budgets_require_nonnegative_original_real_scalars(field, bad):
    with pytest.raises((TypeError, ValueError)):
        _assess(**{field: bad})


@pytest.mark.parametrize(
    "bad", (0, -1, True, False, float("nan"), float("inf"), "0.1", 1j)
)
def test_coefficient_range_requires_a_positive_original_real_scalar(bad):
    with pytest.raises((TypeError, ValueError)):
        _assess(upper=bad)


@pytest.mark.parametrize(
    "horizons",
    (
        (),
        (Q(1, 40),),
        (*HORIZONS, Q(1, 10)),
        HORIZONS[::-1],
        (HORIZONS[0],) * 2,
        (0, HORIZONS[1]),
        (-1, HORIZONS[1]),
        (True, HORIZONS[1]),
        (float("nan"), HORIZONS[1]),
        (HORIZONS[0], float("inf")),
        ("0.025", HORIZONS[1]),
        (1j, HORIZONS[1]),
        set(HORIZONS),
        (Q(1, 4), Q(1, 2)),
    ),
)
def test_horizons_require_two_ordered_positive_scalars_in_the_majorant_domain(horizons):
    with pytest.raises((TypeError, ValueError)):
        _assess(horizons=horizons)


@pytest.mark.parametrize(
    "rotation",
    (
        (True, 0),
        (float("inf"), 0),
        (1, Q(1, 2**1100)),
        (0.6, 0.8),
        {0, 1},
        (1,),
        (1, 0),
        (0, 1),
        (Q(3, 5), Q(-4, 5)),
    ),
)
def test_rotation_requires_exact_unit_data_in_the_declared_positive_sector(rotation):
    with pytest.raises((TypeError, ValueError)):
        _assess(rotation=rotation)


def test_ordered_generators_are_consumed_once(frozen_two_time):
    assert (
        _assess(rotation=(value for value in ROTATION), horizons=(h for h in HORIZONS))
        == frozen_two_time
    )


def test_previous_single_time_collision_remains_valid():
    report = pair.assess_sine_pair_receiver_confounding(
        phase_rotation=ROTATION, horizon_tau=HORIZONS[1], epsilon_upper=UPPER
    )
    assert report.status == "certified_collision_exists"
    assert report.endpoint_difference_signs == (1, -1)
    assert report.endpoint_difference_centers[0] == Q(397, 771844800)
    assert report.endpoint_difference_centers[1] == Q(
        -39091977466163, 19957561355531524800
    )


def test_direct_sdk_export_and_compatibility_keep_exact_evidence(
    tmp_path, frozen_two_time
):
    assert scale.SinePairReceiverTwoTime is pair.SinePairReceiverTwoTime
    assert (
        scale.assess_sine_pair_receiver_two_time
        is pair.assess_sine_pair_receiver_two_time
    )
    for name in ("SinePairReceiverTwoTime", "assess_sine_pair_receiver_two_time"):
        assert name in pair.__all__ and name in scale.__all__
        assert getattr(pair, name).__module__ == pair.__name__
        get_type_hints(getattr(pair, name))
    for name, report in (
        ("certificate", frozen_two_time),
        ("unavailable", _assess(readout=1)),
    ):
        direct = report.to_dict()
        assert direct["schema"] == "tnfr.sine-pair-receiver-two-time.v1"
        projected = relational_report_to_dict(report)
        assert projected["report_type"] == "SinePairReceiverTwoTime"
        assert projected["report"] == direct["report"]
        assert projected["report"]["horizons_tau"] == [
            {"numerator": horizon.numerator, "denominator": horizon.denominator}
            for horizon in HORIZONS
        ]
        target = tmp_path / (name + ".json")
        export_to_json(projected, target)
        assert json_loads(target.read_text(encoding="utf-8")) == projected
