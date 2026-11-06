"""External mean-form discrimination from independent complete-law controls."""

from fractions import Fraction as Q

import pytest

from tests.physics._sine_pair_oracles import (
    NEIGHBORS,
    ONE,
    ORIENTATIONS,
    PAIRS,
    _conj,
    _current_coefficient,
    _fine_jets,
    _integrated_current_remainder,
    _preparation,
)
from tnfr.physics.relational_sine_scale import assess_sine_pair_receiver_readout
from tnfr.sdk import export_to_json, relational_report_to_dict
from tnfr.utils.io import json_loads

ROTATION = (Q(399, 401), Q(40, 401))
HORIZON = Q(1, 20)

ERROR = Q(1, 10**8)


def _assess(rotation=ROTATION, horizon=HORIZON, form=0, phase=0, readout=0):
    return assess_sine_pair_receiver_readout(
        phase_rotation=rotation,
        horizon_tau=horizon,
        form_error_bound=form,
        phase_error_bound=phase,
        readout_error_bound=readout,
    )


def _receiver_coefficients(rotation, orientation):
    x, z = _fine_jets(_preparation(rotation, orientation))
    means = tuple(sum((x[i][k] for i in PAIRS[1]), Q(0)) / 2 for k in range(5))
    return means, x, z


@pytest.fixture(scope="module")
def frozen_readout():
    return _assess(form=ERROR, phase=ERROR, readout=ERROR)


@pytest.mark.parametrize("rotation", (ROTATION, (Q(3, 5), Q(4, 5)), (Q(0), Q(1))))
def test_absolute_receiver_polynomials_use_both_actual_incoming_connections(rotation):
    report = _assess(rotation=rotation)
    for index, orientation in enumerate(ORIENTATIONS):
        coefficients, _, z = _receiver_coefficients(rotation, orientation)
        assert coefficients[0] == coefficients[2] == coefficients[4] == 0
        for degree in range(4):
            assert (degree + 1) * coefficients[degree + 1] == sum(
                (_current_coefficient(z, degree, 1, source) for source in (0, 2)),
                Q(0),
            )
        # The hidden pair has no initial current. The receiver's other
        # connection supplies its nonzero linear drift and cannot be omitted.
        assert _current_coefficient(z, 0, 1, 0) == 0
        assert _current_coefficient(z, 0, 1, 2) == coefficients[1] != 0
        assert report.nominal_linear_coefficient == coefficients[1]
        assert report.nominal_cubic_coefficients[index] == coefficients[3]
        assert report.nominal_readout_centers[index] == (
            coefficients[1] * HORIZON + coefficients[3] * HORIZON**3
        )
        assert report.comparison_initial_phasors[index] == _preparation(
            rotation, orientation
        )
    assert report.nominal_readout_centers[0] != report.nominal_readout_centers[1]


@pytest.mark.parametrize("horizon", (HORIZON, Q(1, 8)))
def test_each_receiver_remainder_adds_both_integrated_current_majorants(horizon):
    report = _assess(horizon=horizon)
    # Each current majorant is independently assembled from the full
    # phasor product rule, then integrated with the factorial Taylor kernel.
    # This sum is over two incoming connections of ONE trajectory, rather
    # than a copied remainder for a contrast of two orientations.
    expected = sum((_integrated_current_remainder(horizon) for _ in (0, 2)), Q(0))
    assert report.nominal_remainder_upper_bound == expected
    for center, bounds in zip(
        report.nominal_readout_centers, report.nominal_readout_bounds
    ):
        assert bounds.lo <= center - expected <= center + expected <= bounds.hi


def _comparison_series_form_bound(horizon, form, phase):
    """Solve the two-channel supremum comparison via an independent series.

    Fine sine differences are globally Lipschitz in each phase argument;
    the phase row is linear in all forms. Their original fine-edge row
    sums produce the two off-diagonal comparison coefficients.
    """
    form_from_phase = horizon * max(Q(2 * len(row), 4) for row in NEIGHBORS)
    phase_from_form = horizon * max(1 + Q(len(row), 4) for row in NEIGHBORS)
    term = (Q(form), Q(phase))
    partial = [Q(0), Q(0)]
    for _ in range(8):
        partial[0] += term[0]
        partial[1] += term[1]
        term = (form_from_phase * term[1], phase_from_form * term[0])
    # After eight powers, the off-diagonal matrix returns to a scalar
    # multiple of identity; sum the remaining geometric blocks exactly.
    block_ratio = (form_from_phase * phase_from_form) ** 4
    assert block_ratio < 1
    return partial[0] / (1 - block_ratio)


@pytest.mark.parametrize(
    "form,phase", ((ERROR, 0), (0, ERROR), (ERROR, ERROR), (ERROR, ERROR / 7))
)
@pytest.mark.parametrize("horizon", (HORIZON, Q(1, 4)))
def test_preparation_bound_retains_separate_form_and_phase_channels(
    form, phase, horizon
):
    report = _assess(horizon=horizon, form=form, phase=phase)
    expected = _comparison_series_form_bound(horizon, form, phase)
    assert report.propagated_preparation_error_upper_bound == expected
    assert report.form_error_amplification_upper_bound == _comparison_series_form_bound(
        horizon, 1, 0
    )
    assert (
        report.phase_error_amplification_upper_bound
        == _comparison_series_form_bound(horizon, 0, 1)
    )
    assert report.readout_uncertainty_radius == expected
    assert report.form_error_bound == form
    assert report.phase_error_bound == phase


def test_common_form_origin_changes_absolute_readout_and_its_budget_is_retained():
    phasors = _preparation(ROTATION, ORIENTATIONS[0])
    baseline_x, baseline_z = _fine_jets(phasors)
    shifted_x, shifted_z = _fine_jets(phasors, forms=(ERROR,) * 10)
    assert shifted_z == baseline_z
    for unshifted, shifted in zip(baseline_x, shifted_x):
        assert shifted[0] == unshifted[0] + ERROR
        assert shifted[1:] == unshifted[1:]
    # Exact common-form symmetry shifts the one-time reading by ERROR at
    # every time. The bound therefore includes preparation origin error;
    # it must not secretly center each candidate or discard that mode.
    report = _assess(form=ERROR)
    assert report.propagated_preparation_error_upper_bound >= ERROR
    for center, bounds in zip(
        report.nominal_readout_centers, report.expanded_readout_bounds
    ):
        assert center - ERROR in bounds
        assert center + ERROR in bounds


def test_frozen_positive_error_budgets_still_give_disjoint_reading_intervals(
    frozen_readout,
):
    report = frozen_readout
    first, second = report.expanded_readout_bounds
    assert report.status == "certified_disjoint"
    assert report.unavailable_reason is None
    assert report.higher_readout_orientation == "real_antipodal"
    assert report.receiver_pair_index == 1
    assert report.receiver_nodes == (2, 3)
    assert first.lo > second.hi
    assert report.readout_gap_lower_bound == first.lo - second.hi > 0
    assert (
        report.form_error_bound
        == report.phase_error_bound
        == report.readout_error_bound
        == ERROR
    )
    radius = report.propagated_preparation_error_upper_bound + ERROR
    assert report.readout_uncertainty_radius == radius > 0
    for center, bounds in zip(
        report.nominal_readout_centers, report.expanded_readout_bounds
    ):
        assert bounds.lo <= center - report.nominal_remainder_upper_bound - radius
        assert center + report.nominal_remainder_upper_bound + radius <= bounds.hi


def test_readout_noise_is_added_after_flow_uncertainty_and_does_not_change_centers():
    exact = _assess()
    noisy = _assess(readout=Q(1, 100))
    assert noisy.nominal_readout_centers == exact.nominal_readout_centers
    assert noisy.propagated_preparation_error_upper_bound == 0
    assert noisy.readout_uncertainty_radius == Q(1, 100)
    assert noisy.status == "unavailable"
    assert noisy.unavailable_reason == "receiver_readout_intervals_overlap_or_touch"
    assert noisy.higher_readout_orientation is None
    assert noisy.readout_gap_lower_bound < 0


def test_touching_outward_intervals_do_not_certify_discrimination():
    # The dyadic horizon makes both independently derived centers dyadic.
    # Choose the readout budget to make the certified intervals touch
    # exactly, rather than relying on a rounded nearly-zero comparison.
    horizon, rotation = Q(3, 32), (Q(0), Q(1))
    centers = tuple(
        values[0][1] * horizon + values[0][3] * horizon**3
        for values in (_receiver_coefficients(rotation, d) for d in ORIENTATIONS)
    )
    remainder = 2 * _integrated_current_remainder(horizon)
    error = abs(centers[0] - centers[1]) / 2 - remainder
    assert error > 0
    report = _assess(rotation=rotation, horizon=horizon, readout=error)
    assert report.expanded_readout_bounds[0].hi == report.expanded_readout_bounds[1].lo
    assert report.readout_gap_lower_bound == 0
    assert report.status == "unavailable"
    assert report.higher_readout_orientation is None


def test_conjugation_reverses_readout_order_without_changing_discrimination(
    frozen_readout,
):
    reflected = _assess(
        rotation=_conj(ROTATION), form=ERROR, phase=ERROR, readout=ERROR
    )
    assert reflected.status == "certified_disjoint"
    assert reflected.higher_readout_orientation == "imaginary_antipodal"
    assert reflected.nominal_readout_centers == tuple(
        -value for value in frozen_readout.nominal_readout_centers
    )
    assert reflected.expanded_readout_bounds == tuple(
        -bounds for bounds in frozen_readout.expanded_readout_bounds
    )
    assert reflected.readout_gap_lower_bound == frozen_readout.readout_gap_lower_bound


def test_control_is_exact_only_at_its_symmetric_center_and_uncertainty_is_visible(
    frozen_readout,
):
    exact = _assess()
    assert exact.symmetry_control_readout_centers == (Q(0), Q(0))
    assert exact.symmetry_control_uncertainty_radius == 0
    for bounds in exact.symmetry_control_readout_bounds:
        assert bounds.lo == bounds.hi == 0
    uncertain = frozen_readout
    assert uncertain.symmetry_control_uncertainty_radius > 0
    assert (
        uncertain.symmetry_control_readout_bounds[0]
        == uncertain.symmetry_control_readout_bounds[1]
    )
    for bounds in uncertain.symmetry_control_readout_bounds:
        assert bounds.lo < 0 < bounds.hi
        assert -ERROR in bounds and ERROR in bounds
    for orientation, phasors in zip(ORIENTATIONS, uncertain.control_initial_phasors):
        assert phasors == _preparation(ROTATION, orientation, control=True)


def test_unbroken_environment_preserves_unavailable_orientation_discrimination():
    report = _assess(rotation=ONE, form=ERROR, phase=ERROR, readout=ERROR)
    assert report.nominal_readout_centers == (Q(0), Q(0))
    assert report.expanded_readout_bounds[0] == report.expanded_readout_bounds[1]
    assert report.status == "unavailable"


def test_tiny_positive_budgets_are_kept_exact_before_outward_interval_rounding():
    tiny = Q(1, 2**1100)
    report = _assess(form=tiny, phase=tiny, readout=tiny)
    assert (
        report.form_error_bound
        == report.phase_error_bound
        == report.readout_error_bound
        == tiny
        > 0
    )
    assert report.propagated_preparation_error_upper_bound > tiny
    assert report.readout_uncertainty_radius > 2 * tiny
    assert report.status == "certified_disjoint"


def test_tiny_horizon_does_not_turn_exact_center_separation_into_a_certificate():
    tiny = Q(1, 2**1100)
    report = _assess(horizon=tiny)
    assert report.horizon_tau == tiny > 0
    assert report.nominal_readout_centers[0] != report.nominal_readout_centers[1]
    assert report.expanded_readout_bounds[0] == report.expanded_readout_bounds[1]
    assert report.status == "unavailable"


def test_exact_horizon_just_below_boundary_is_not_rounded_to_the_rejected_endpoint():
    horizon = Q(1, 2) - Q(1, 2**1100)
    report = _assess(horizon=horizon)
    assert report.horizon_tau == horizon < Q(1, 2)
    assert report.form_error_amplification_upper_bound > 2**1097
    assert report.status == "unavailable"


@pytest.mark.parametrize(
    "horizon", (0, -1, Q(-1, 2**1100), Q(1, 2), Q(1, 2) + Q(1, 2**1100), 1)
)
def test_horizon_requires_the_strict_declared_majorant_domain(horizon):
    with pytest.raises(ValueError):
        _assess(horizon=horizon)


@pytest.mark.parametrize("field", ("form", "phase", "readout"))
@pytest.mark.parametrize(
    "bad", (-1, Q(-1, 2**1100), True, False, float("nan"), float("inf"), "0", 1j)
)
def test_each_error_budget_requires_a_nonnegative_physical_real(field, bad):
    with pytest.raises((TypeError, ValueError)):
        _assess(**{field: bad})


@pytest.mark.parametrize("bad", (True, False, float("nan"), float("inf"), "0.05", 1j))
def test_horizon_rejects_nonphysical_scalar_values(bad):
    with pytest.raises((TypeError, ValueError)):
        _assess(horizon=bad)


@pytest.mark.parametrize(
    "rotation", ((True, 0), (float("nan"), 0), (1, Q(1, 2**1100)), (0.6, 0.8), {0, 1})
)
def test_receiver_reader_reuses_exact_rotation_admission(rotation):
    with pytest.raises((TypeError, ValueError)):
        _assess(rotation=rotation)


def test_sdk_export_keeps_budgets_bounds_and_nominal_control_separate(
    tmp_path, frozen_readout
):
    direct = frozen_readout.to_dict()
    assert direct["schema"] == "tnfr.sine-pair-receiver-readout.v1"
    projected = relational_report_to_dict(frozen_readout)
    assert projected["report"] == direct["report"]
    assert projected["report"]["form_error_bound"] == {
        "numerator": 1,
        "denominator": 10**8,
    }
    radius = frozen_readout.propagated_preparation_error_upper_bound
    assert projected["report"]["propagated_preparation_error_upper_bound"] == {
        "numerator": radius.numerator,
        "denominator": radius.denominator,
    }
    target = tmp_path / "receiver-readout.json"
    export_to_json(projected, target)
    assert json_loads(target.read_text(encoding="utf-8")) == projected
    assert (
        projected["report"]["symmetry_control_readout_bounds"]
        != projected["report"]["symmetry_control_readout_centers"]
    )
