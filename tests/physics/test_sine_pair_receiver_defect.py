"""Independent inhomogeneous-flow controls for the fixed receiver experiment."""

from fractions import Fraction as Q
from math import factorial
from typing import get_type_hints

import pytest

from tests.physics._sine_pair_oracles import (
    EDGES,
    NEIGHBORS,
    ORIENTATIONS,
    _conj,
    _cubic_coefficients,
    _fine_jets,
    _independent_current_derivative_bounds,
    _initial_rate_bound,
    _mul,
    _preparation,
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

DEFECT = Q(1, 10**6)


def _reference(
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


def _assess(
    form_defect=DEFECT,
    phase_defect=DEFECT,
    rotation=ROTATION,
    horizons=HORIZONS,
    upper=UPPER,
    form=ERROR,
    phase=ERROR,
    readout=ERROR,
):
    return pair.assess_sine_pair_receiver_defect(
        phase_rotation=rotation,
        horizons_tau=horizons,
        epsilon_upper=upper,
        form_error_bound=form,
        phase_error_bound=phase,
        readout_error_bound=readout,
        form_rate_defect_bound=form_defect,
        phase_rate_defect_bound=phase_defect,
    )


def _inhomogeneous_response_factors(horizon, upper):
    """Integrate powers of the two-channel fine-row comparison matrix.

    The true constant-source comparison series has factorial denominators.
    Keeping 1 for even matrix powers and 2 for odd ones majorizes its form
    component term by term. Eight pairs plus their geometric tail give the
    exact rational factors used by the finite proof, without an exponential.
    """
    lipschitz = _independent_current_derivative_bounds(upper)[1]
    form_from_phase = lipschitz * max(Q(2 * len(row), 4) for row in NEIGHBORS)
    phase_from_form = max(1 + Q(len(row), 4) for row in NEIGHBORS)
    block_ratio = form_from_phase * phase_from_form * horizon**2
    assert 0 <= block_ratio < 1
    factors = []
    for source in ((Q(1), Q(0)), (Q(0), Q(1))):
        term = source
        majorant_sum = Q(0)
        true_partial_sum = Q(0)
        for power in range(16):
            numerator = term[0] * horizon ** (power + 1)
            true_term = numerator / factorial(power + 1)
            majorant_term = numerator / (1 if power % 2 == 0 else 2)
            assert 0 <= true_term <= majorant_term
            true_partial_sum += true_term
            majorant_sum += majorant_term
            term = (form_from_phase * term[1], phase_from_form * term[0])
        bound = majorant_sum / (1 - block_ratio**8)
        assert true_partial_sum <= bound
        factors.append(bound)
    return tuple(factors)


@pytest.fixture(scope="module")
def frozen_defect():
    return _assess()


@pytest.mark.parametrize("upper", (UPPER, Q(2)))
def test_response_factors_majorize_the_integrated_two_channel_series(upper):
    report = _assess(upper=upper)
    for index, horizon in enumerate(HORIZONS):
        form, phase = _inhomogeneous_response_factors(horizon, upper)
        assert report.form_defect_response_factors_by_time[index] == form
        assert report.phase_defect_response_factors_by_time[index] == phase
        assert report.defect_error_bounds_by_time[index] == DEFECT * (form + phase)
        # Form-rate defects enter directly; phase-rate defects first pass
        # through the phase-to-form row and therefore carry a second power.
        assert form >= horizon
        assert phase >= _independent_current_derivative_bounds(upper)[1] * horizon**2


@pytest.mark.parametrize(
    "form_defect,phase_defect",
    ((0, 0), (DEFECT, 0), (0, DEFECT), (DEFECT, DEFECT), (DEFECT / 7, DEFECT * 3)),
)
def test_defect_channels_remain_separate_from_preparation_and_measurement(
    form_defect, phase_defect
):
    report = _assess(form_defect=form_defect, phase_defect=phase_defect)
    reference = _reference()
    assert report.reference_certificate == reference
    assert report.form_rate_defect_bound == form_defect
    assert report.phase_rate_defect_bound == phase_defect
    expected = tuple(
        form_defect * factors[0] + phase_defect * factors[1]
        for factors in (_inhomogeneous_response_factors(h, UPPER) for h in HORIZONS)
    )
    assert report.defect_error_bounds_by_time == expected
    weighted = sum(
        abs(w) * error
        for w, error in zip(reference.readout_weights, expected, strict=True)
    )
    assert report.joint_defect_radius == weighted
    assert report.joint_error_radii == tuple(
        radius + weighted for radius in reference.joint_error_radii
    )


@pytest.mark.parametrize("readout", (ERROR, Q(1, 100)))
def test_zero_defects_preserve_the_earlier_interval_and_status_exactly(readout):
    reference = _reference(readout=readout)
    report = _assess(form_defect=0, phase_defect=0, readout=readout)
    assert report.reference_certificate == reference
    assert report.defect_error_bounds_by_time == (0, 0)
    assert report.joint_defect_radius == 0
    assert report.joint_error_radii == reference.joint_error_radii
    assert report.joint_difference_bounds == reference.joint_difference_bounds
    assert report.status == reference.status
    assert report.unavailable_reason == reference.unavailable_reason


def test_frozen_defect_certificate_uses_the_direct_exact_difference(frozen_defect):
    report = frozen_defect
    reference = report.reference_certificate
    a, b = reference.nominal_cubic_coefficient_ranges
    total_error = sum(report.joint_error_radii)
    lower = reference.cubic_time_factor * (a[0] - b[1]) - total_error
    upper = reference.cubic_time_factor * (a[1] - b[0]) + total_error
    assert (
        report.joint_difference_bounds.lo
        <= lower
        <= upper
        <= report.joint_difference_bounds.hi
    )
    assert report.status == "certified_disjoint"
    assert report.unavailable_reason is None
    assert 0 < report.joint_difference_bounds.lo < reference.joint_difference_bounds.lo
    assert report.form_rate_defect_bound == report.phase_rate_defect_bound == DEFECT


def test_uniform_form_ramp_is_an_exact_nonstationary_allowed_departure(frozen_defect):
    # For any nominal solution, x_i(t)+s(t) has the same phase row and
    # circular currents when the common shift is s(t)=delta_x*t²/(2*h_max).
    # Its derivative adds the admissible ramp delta_x*t/h_max to every form row.
    coefficient = DEFECT / (2 * HORIZONS[1])
    forms = tuple(Q((i + 1) ** 2, 37) for i in range(10))
    phasors = _preparation(ROTATION, ORIENTATIONS[0])
    nominal_x, nominal_z = _fine_jets(phasors, order=1, forms=forms, epsilon=UPPER)
    offsets = []
    for horizon, error_bound in zip(
        HORIZONS, frozen_defect.defect_error_bounds_by_time, strict=True
    ):
        offset = coefficient * horizon**2
        shifted_x, shifted_z = _fine_jets(
            phasors,
            order=1,
            forms=tuple(value + offset for value in forms),
            epsilon=UPPER,
        )
        assert all(
            left[1] == right[1]
            for left, right in zip(nominal_x, shifted_x, strict=True)
        )
        assert nominal_z == shifted_z
        residual = 2 * coefficient * horizon
        assert 0 < residual <= DEFECT
        assert 0 < offset <= error_bound
        offsets.append(offset)
    weighted_offset = sum(
        w * offset
        for w, offset in zip(
            frozen_defect.reference_certificate.readout_weights, offsets, strict=True
        )
    )
    assert 0 < weighted_offset <= frozen_defect.joint_defect_radius
    # Thus the two-time weights cancel a constant rate's linear drift,
    # but do not cancel every permitted continuous residual history.
    assert (
        sum(
            w * DEFECT * h
            for w, h in zip(
                frozen_defect.reference_certificate.readout_weights,
                HORIZONS,
                strict=True,
            )
        )
        == 0
    )
    control = _preparation(ROTATION, ORIENTATIONS[0], control=True)
    x, z = _fine_jets(control, order=1, forms=(offsets[-1],) * 10, epsilon=UPPER)
    assert all(row[1] == 0 for row in x)
    assert all(row[1] == (0, 0) for row in z)
    # Adding the same ramp to this stationary nominal control makes every
    # form move, despite retaining zero pressure and all phase differences.
    assert 2 * coefficient * HORIZONS[1] == DEFECT > 0


@pytest.mark.parametrize("channel", ("form", "phase"))
def test_complete_row_defects_have_explicit_storage_work(channel):
    forms = tuple(Q((i + 1) ** 2, 37) for i in range(10))
    phasors = _preparation(ROTATION, ORIENTATIONS[0])
    x, _ = _fine_jets(phasors, order=1, forms=forms, epsilon=UPPER)
    gradient = tuple(
        sum((forms[i] - forms[j] for j in row), Q(0)) for i, row in enumerate(NEIGHBORS)
    )
    phase_rates = tuple(value / 4 for value in gradient)
    form_defects = tuple(
        DEFECT if channel == "form" and i == 0 else Q(0) for i in range(10)
    )
    phase_defects = tuple(
        DEFECT if channel == "phase" and i == 2 else Q(0) for i in range(10)
    )
    defective_form_rates = tuple(
        row[1] + defect for row, defect in zip(x, form_defects, strict=True)
    )
    defective_phase_rates = tuple(
        rate + defect for rate, defect in zip(phase_rates, phase_defects, strict=True)
    )
    direct_storage_rate = Q(0)
    for i, j in EDGES:
        sine = _mul(_conj(phasors[i]), phasors[j])[1]
        current = sine + UPPER * sine**3
        direct_storage_rate += (forms[i] - forms[j]) * (
            defective_form_rates[i] - defective_form_rates[j]
        )
        direct_storage_rate += current * (
            defective_phase_rates[j] - defective_phase_rates[i]
        )
    residual_work = sum(
        q * rx - 4 * row[1] * rp
        for q, row, rx, rp in zip(gradient, x, form_defects, phase_defects, strict=True)
    )
    assert direct_storage_rate == residual_work != 0
    assert sum(defective_form_rates) == sum(form_defects)


def test_large_defects_do_not_inherit_the_nominal_certificate():
    report = _assess(form_defect=Q(1, 100))
    assert report.reference_certificate.status == "certified_disjoint"
    assert report.status == "unavailable"
    assert report.unavailable_reason == "joint_readout_difference_not_strictly_positive"
    assert 0 in report.joint_difference_bounds


def test_exact_touching_defect_budget_is_unavailable():
    weights = (-HORIZONS[1] / sum(HORIZONS), HORIZONS[0] / sum(HORIZONS))
    factor = HORIZONS[0] * HORIZONS[1] * (HORIZONS[1] - HORIZONS[0])
    nominal = factor * (
        _value(_cubic_coefficients(ROTATION, ORIENTATIONS[0]), 0)
        - _value(_cubic_coefficients(ROTATION, ORIENTATIONS[1]), UPPER)
    )
    truncation = sum(
        abs(weight)
        * _refined_remainder(
            h, UPPER, _initial_rate_bound(ROTATION, orientation, UPPER)
        )[2]
        for orientation in ORIENTATIONS
        for weight, h in zip(weights, HORIZONS, strict=True)
    )
    response = sum(
        abs(w) * _inhomogeneous_response_factors(h, UPPER)[0]
        for w, h in zip(weights, HORIZONS, strict=True)
    )
    defect = (nominal - truncation) / (2 * response)
    assert defect > 0
    report = _assess(form_defect=defect, phase_defect=0, form=0, phase=0, readout=0)
    assert report.joint_difference_bounds.lo == 0
    assert report.status == "unavailable"


def test_tiny_exact_defect_budgets_are_retained():
    class RationalInput(Q):
        pass

    tiny = RationalInput(1, 2**1100)
    report = _assess(form_defect=tiny, phase_defect=tiny)
    assert report.form_rate_defect_bound == report.phase_rate_defect_bound == tiny > 0
    assert type(report.form_rate_defect_bound) is Q
    assert all(value > 0 for value in report.defect_error_bounds_by_time)
    assert report.joint_defect_radius > 0


@pytest.mark.parametrize("field", ("form_defect", "phase_defect"))
@pytest.mark.parametrize(
    "bad",
    (-1, Q(-1, 2**1100), True, False, float("nan"), float("inf"), "0", 1j, object()),
)
def test_original_defect_scalars_reject_before_building_nominal_evidence(
    monkeypatch, field, bad
):
    def forbidden(**kwargs):
        raise AssertionError("invalid defect reached the nominal producer")

    monkeypatch.setattr(pair, "assess_sine_pair_receiver_two_law", forbidden)
    with pytest.raises((TypeError, ValueError)):
        _assess(**{field: bad})


@pytest.mark.parametrize(
    "kwargs",
    (
        {"rotation": (0.6, 0.8)},
        {"horizons": HORIZONS[::-1]},
        {"upper": True},
        {"form": float("nan")},
        {"phase": -1},
        {"readout": "0"},
    ),
)
def test_nominal_primitives_are_freshly_admitted(kwargs):
    with pytest.raises((TypeError, ValueError)):
        _assess(**kwargs)


def test_ordered_nominal_generators_are_consumed_once(frozen_defect):
    assert (
        _assess(rotation=(value for value in ROTATION), horizons=(h for h in HORIZONS))
        == frozen_defect
    )


def test_defect_api_does_not_accept_a_supplied_reference_report(frozen_defect):
    with pytest.raises(TypeError, match="reference_certificate"):
        pair.assess_sine_pair_receiver_defect(
            reference_certificate=frozen_defect.reference_certificate
        )


def test_compatibility_and_nested_exact_export(tmp_path, frozen_defect):
    for name in ("SinePairReceiverDefect", "assess_sine_pair_receiver_defect"):
        assert getattr(scale, name) is getattr(pair, name)
        assert name in scale.__all__ and name in pair.__all__
        assert getattr(pair, name).__module__ == pair.__name__
        get_type_hints(getattr(pair, name))
    for name, report in (
        ("certificate", frozen_defect),
        ("unavailable", _assess(form_defect=1)),
    ):
        direct = report.to_dict()
        assert direct["schema"] == "tnfr.sine-pair-receiver-defect.v1"
        projected = relational_report_to_dict(report)
        assert projected["report_type"] == "SinePairReceiverDefect"
        assert projected["report"] == direct["report"]
        assert (
            projected["report"]["reference_certificate"]
            == report.reference_certificate.to_dict()["report"]
        )
        assert projected["report"]["form_rate_defect_bound"] == {
            "numerator": report.form_rate_defect_bound.numerator,
            "denominator": report.form_rate_defect_bound.denominator,
        }
        target = tmp_path / (name + ".json")
        export_to_json(projected, target)
        assert json_loads(target.read_text(encoding="utf-8")) == projected
