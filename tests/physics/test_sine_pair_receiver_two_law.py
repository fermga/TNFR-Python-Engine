"""Independent fine-node controls for two unknown conservative-law coefficients."""

from fractions import Fraction as Q
from typing import get_type_hints

import pytest

from tests.physics._sine_pair_oracles import (
    ORIENTATIONS,
    _cubic_coefficients,
    _fine_jets,
    _initial_rate_bound,
    _preparation,
    _preparation_bound,
    _receiver_jets,
    _refined_remainder,
    _value,
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
    return pair.assess_sine_pair_receiver_two_law(
        phase_rotation=rotation,
        horizons_tau=horizons,
        epsilon_upper=upper,
        form_error_bound=form,
        phase_error_bound=phase,
        readout_error_bound=readout,
    )


@pytest.fixture(scope="module")
def frozen_two_law():
    return _assess()


@pytest.mark.parametrize("rotation", (ROTATION, (Q(3, 5), Q(4, 5))))
def test_both_receiver_polynomials_are_derived_from_the_complete_fine_laws(rotation):
    report = _assess(rotation=rotation)
    coefficients = tuple(
        _cubic_coefficients(rotation, orientation) for orientation in ORIENTATIONS
    )
    assert report.nominal_cubic_coefficients == coefficients
    for orientation, cubic, extrema in zip(
        ORIENTATIONS, coefficients, report.nominal_cubic_coefficient_ranges, strict=True
    ):
        assert all(value > 0 for value in cubic)
        assert extrema == (_value(cubic, 0), _value(cubic, UPPER))
        epsilon = Q(3, 37)
        jets = _receiver_jets(rotation, orientation, epsilon)
        assert jets[0] == jets[2] == jets[4] == 0
        assert jets[3] == _value(cubic, epsilon)
    assert report.comparison_initial_phasors == tuple(
        _preparation(rotation, orientation) for orientation in ORIENTATIONS
    )
    assert report.control_initial_phasors == tuple(
        _preparation(rotation, orientation, control=True)
        for orientation in ORIENTATIONS
    )


@pytest.mark.parametrize(
    "rotation,upper", ((ROTATION, UPPER), ((Q(3, 5), Q(4, 5)), Q(2)))
)
def test_uniform_initial_rate_bound_consumes_all_fine_nodes(rotation, upper):
    report = _assess(rotation=rotation, upper=upper)
    expected = tuple(
        _initial_rate_bound(rotation, orientation, upper)
        for orientation in ORIENTATIONS
    )
    assert report.nominal_initial_rate_bounds == expected
    if upper == 2:
        # The imaginary selected pair dominates here; the environmental pair
        # alone would underestimate B's full nominal initial rate.
        x, _ = _fine_jets(
            _preparation(rotation, ORIENTATIONS[1]), order=1, epsilon=upper
        )
        assert abs(x[0][1]) > abs(x[2][1])
        assert expected[1] == abs(x[0][1]) > expected[0]
    for epsilon in (upper / 3, upper * Q(7, 11)):
        for index, orientation in enumerate(ORIENTATIONS):
            x, _ = _fine_jets(
                _preparation(rotation, orientation), order=1, epsilon=epsilon
            )
            assert max(abs(node[1]) for node in x) <= expected[index]


@pytest.mark.parametrize(
    "rotation,upper", ((ROTATION, UPPER), ((Q(3, 5), Q(4, 5)), Q(2)))
)
def test_remainder_integrates_initial_rate_continuation_and_all_pressure_derivatives(
    rotation, upper
):
    report = _assess(rotation=rotation, upper=upper)
    initial = tuple(
        _initial_rate_bound(rotation, orientation, upper)
        for orientation in ORIENTATIONS
    )
    for time_index, horizon in enumerate(HORIZONS):
        for orientation_index, bound in enumerate(initial):
            growth, rate, error = _refined_remainder(horizon, upper, bound)
            assert (
                report.form_growth_bounds_by_time[time_index][orientation_index]
                == growth
            )
            assert (
                report.form_rate_bounds_by_time[time_index][orientation_index] == rate
            )
            assert (
                report.remainder_upper_bounds_by_time[time_index][orientation_index]
                == error
            )
            assert 0 < bound < growth < rate


@pytest.mark.parametrize("form,phase", ((0, 0), (ERROR, 0), (0, ERROR), (ERROR, ERROR)))
def test_joint_errors_retain_separate_preparation_channels_and_every_reading(
    form, phase
):
    report = _assess(form=form, phase=phase)
    preparations = tuple(_preparation_bound(h, UPPER, form, phase) for h in HORIZONS)
    assert report.preparation_error_bounds_by_time == preparations
    for index, orientation in enumerate(ORIENTATIONS):
        initial = _initial_rate_bound(ROTATION, orientation, UPPER)
        radius = sum(
            abs(weight)
            * (_refined_remainder(h, UPPER, initial)[2] + preparation + ERROR)
            for weight, h, preparation in zip(
                report.readout_weights, HORIZONS, preparations, strict=True
            )
        )
        assert report.joint_error_radii[index] == radius


def test_fixed_observable_cancels_each_law_linear_term_separately(frozen_two_law):
    report = frozen_two_law
    assert report.readout_weights == (Q(-2, 3), Q(1, 3))
    assert sum(map(abs, report.readout_weights)) == 1
    assert (
        sum(w * h for w, h in zip(report.readout_weights, HORIZONS, strict=True)) == 0
    )
    factor = sum(
        w * h**3 for w, h in zip(report.readout_weights, HORIZONS, strict=True)
    )
    assert report.cubic_time_factor == factor > 0
    # Coefficients differ between hypotheses, but remain fixed across times.
    for orientation, epsilon in zip(ORIENTATIONS, (UPPER / 3, UPPER), strict=True):
        jets = _receiver_jets(ROTATION, orientation, epsilon)
        weighted = sum(
            w * (jets[1] * h + jets[3] * h**3)
            for w, h in zip(report.readout_weights, HORIZONS, strict=True)
        )
        assert weighted == factor * jets[3]
    early = _receiver_jets(ROTATION, ORIENTATIONS[0], Q(0))
    late = _receiver_jets(ROTATION, ORIENTATIONS[0], UPPER)
    time_varying_linear = sum(
        w * h * jets[1]
        for w, h, jets in zip(
            report.readout_weights, HORIZONS, (early, late), strict=True
        )
    )
    assert time_varying_linear != 0  # Outside the fixed-coefficient premise.


def test_frozen_bounds_use_independent_coefficient_extrema_and_outward_separation(
    frozen_two_law,
):
    report = frozen_two_law
    ranges = tuple(
        (
            _value(_cubic_coefficients(ROTATION, orientation), 0),
            _value(_cubic_coefficients(ROTATION, orientation), UPPER),
        )
        for orientation in ORIENTATIONS
    )
    factor = HORIZONS[0] * HORIZONS[1] * (HORIZONS[1] - HORIZONS[0])
    error = sum(report.joint_error_radii)
    lower = factor * (ranges[0][0] - ranges[1][1]) - error
    upper = factor * (ranges[0][1] - ranges[1][0]) + error
    assert (
        report.joint_difference_bounds.lo
        <= lower
        <= upper
        <= report.joint_difference_bounds.hi
    )
    assert report.joint_difference_bounds.lo > 0
    assert report.status == "certified_disjoint" and report.unavailable_reason is None
    assert report.coefficient_endpoints == (0, UPPER)
    assert report.phase_rotation == ROTATION and report.horizons_tau == HORIZONS
    assert (
        report.initial_forms == (Q(0),) * 10 and report.held_capacities == (Q(1),) * 10
    )
    assert report.nodes == tuple(range(10)) and report.receiver_nodes == (2, 3)
    assert report.clock == "tau=t/pi"
    # Both independent corner pairs are admitted. Correlating alpha=beta
    # would omit the true opposing extrema used by this certificate.
    for alpha, beta in ((Q(0), UPPER), (UPPER, Q(0))):
        center = factor * (
            _receiver_jets(ROTATION, ORIENTATIONS[0], alpha)[3]
            - _receiver_jets(ROTATION, ORIENTATIONS[1], beta)[3]
        )
        assert center - error in report.joint_difference_bounds
        assert center + error in report.joint_difference_bounds


def test_increased_readout_error_abstains_without_recomputing_nominal_evidence(
    frozen_two_law,
):
    report = _assess(readout=Q(1, 100))
    assert (
        report.nominal_cubic_coefficients == frozen_two_law.nominal_cubic_coefficients
    )
    assert (
        report.remainder_upper_bounds_by_time
        == frozen_two_law.remainder_upper_bounds_by_time
    )
    assert (
        report.preparation_error_bounds_by_time
        == frozen_two_law.preparation_error_bounds_by_time
    )
    assert report.status == "unavailable"
    assert report.unavailable_reason == "joint_readout_difference_not_strictly_positive"
    assert 0 in report.joint_difference_bounds


def test_exact_zero_lower_bound_is_not_a_strict_certificate():
    coefficients = tuple(
        _cubic_coefficients(ROTATION, orientation) for orientation in ORIENTATIONS
    )
    factor = HORIZONS[0] * HORIZONS[1] * (HORIZONS[1] - HORIZONS[0])
    nominal_lower = factor * (
        _value(coefficients[0], 0) - _value(coefficients[1], UPPER)
    )
    weights = (-HORIZONS[1] / sum(HORIZONS), HORIZONS[0] / sum(HORIZONS))
    truncation = sum(
        abs(weight)
        * _refined_remainder(
            h, UPPER, _initial_rate_bound(ROTATION, orientation, UPPER)
        )[2]
        for orientation in ORIENTATIONS
        for weight, h in zip(weights, HORIZONS, strict=True)
    )
    noise = (nominal_lower - truncation) / 2
    assert noise > 0
    report = _assess(form=0, phase=0, readout=noise)
    assert report.joint_difference_bounds.lo == 0
    assert report.status == "unavailable"


def test_tiny_exact_positive_margin_cannot_pass_after_outward_rounding():
    horizon = Q(1, 2**50)
    report = _assess(horizons=(horizon, 2 * horizon), form=0, phase=0, readout=0)
    nominal = report.cubic_time_factor * (
        report.nominal_cubic_coefficient_ranges[0][0]
        - report.nominal_cubic_coefficient_ranges[1][1]
    )
    assert 0 < nominal - sum(report.joint_error_radii) < Q(1, 2**128)
    assert report.joint_difference_bounds.lo == 0
    assert report.status == "unavailable"


def test_tiny_exact_scalar_budgets_preserve_original_values():
    class RationalInput(Q):
        pass

    tiny = RationalInput(1, 2**1100)
    report = _assess(form=tiny, phase=tiny, readout=tiny, upper=tiny)
    assert (
        report.epsilon_upper
        == report.form_error_bound
        == report.phase_error_bound
        == report.readout_error_bound
        == tiny
        > 0
    )
    assert type(report.epsilon_upper) is Q
    assert all(value > tiny for value in report.preparation_error_bounds_by_time)


@pytest.mark.parametrize("field", ("form", "phase", "readout"))
@pytest.mark.parametrize(
    "bad", (-1, Q(-1, 2**1100), True, False, float("nan"), float("inf"), "0", 1j)
)
def test_original_error_budgets_require_nonnegative_finite_real_scalars(field, bad):
    with pytest.raises((TypeError, ValueError)):
        _assess(**{field: bad})


@pytest.mark.parametrize(
    "bad", (0, -1, True, False, float("nan"), float("inf"), "0.1", 1j)
)
def test_coefficient_upper_bound_requires_a_positive_finite_real_scalar(bad):
    with pytest.raises((TypeError, ValueError)):
        _assess(upper=bad)


@pytest.mark.parametrize(
    "horizons",
    (
        (),
        (HORIZONS[0],),
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
def test_horizons_require_ordered_positive_original_scalars_in_the_majorant_domain(
    horizons,
):
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
def test_rotation_requires_exact_unit_phasors_in_the_proved_sector(rotation):
    with pytest.raises((TypeError, ValueError)):
        _assess(rotation=rotation)


def test_ordered_generators_are_captured_once(frozen_two_law):
    assert (
        _assess(rotation=(value for value in ROTATION), horizons=(h for h in HORIZONS))
        == frozen_two_law
    )


def test_exact_preparation_boundary_is_not_decided_by_float_conversion():
    maximum = Q(1, 4) - Q(1, 2**1100)
    report = _assess(horizons=(maximum / 2, maximum), upper=Q(1))
    assert report.horizons_tau[1] == maximum < Q(1, 4)
    assert report.status == "unavailable"


def test_previous_two_time_certificate_keeps_its_outward_gap():
    report = pair.assess_sine_pair_receiver_two_time(
        phase_rotation=ROTATION,
        horizons_tau=HORIZONS,
        epsilon_upper=UPPER,
        form_error_bound=ERROR,
        phase_error_bound=ERROR,
        readout_error_bound=ERROR,
    )
    assert report.status == "certified_disjoint"
    assert report.coefficient_gap_lower_bound == Q(
        241391045683833871486739624526496365, 2**128
    )


def test_compatibility_and_exact_sdk_export(tmp_path, frozen_two_law):
    for name in ("SinePairReceiverTwoLaw", "assess_sine_pair_receiver_two_law"):
        assert getattr(scale, name) is getattr(pair, name)
        assert name in scale.__all__ and name in pair.__all__
        assert getattr(pair, name).__module__ == pair.__name__
        get_type_hints(getattr(pair, name))
    for name, report in (
        ("certificate", frozen_two_law),
        ("unavailable", _assess(readout=1)),
    ):
        direct = report.to_dict()
        assert direct["schema"] == "tnfr.sine-pair-receiver-two-law.v1"
        projected = relational_report_to_dict(report)
        assert projected["report_type"] == "SinePairReceiverTwoLaw"
        assert projected["report"] == direct["report"]
        assert projected["report"]["readout_weights"] == [
            {"numerator": value.numerator, "denominator": value.denominator}
            for value in report.readout_weights
        ]
        target = tmp_path / (name + ".json")
        export_to_json(projected, target)
        assert json_loads(target.read_text(encoding="utf-8")) == projected
