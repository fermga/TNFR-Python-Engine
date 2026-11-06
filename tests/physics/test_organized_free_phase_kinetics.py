"""Independent primitive-phase controls without evolving a scientific source."""

from collections import UserDict
from dataclasses import FrozenInstanceError
from fractions import Fraction as Q
from math import factorial

import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.physics import _organized_collective_observation as geometry
from tnfr.physics import _organized_free_phase_kinetics as owner

_PI = pi_interval()
_CARRIER = tuple(2 * _PI.midpoint * (j - 4) / 9 for j in range(9))
_ZERO_FORM = (Q(0),) * 18


@pytest.fixture(scope="module", autouse=True)
def no_scientific_execution():
    with forbid_organized_response_evaluation():
        yield


def _phases(a=Q(0), b=Q(0)):
    return tuple(value + origin for origin in (a, b) for value in _CARRIER)


def _read(**changes):
    inputs = dict(
        sample_times=(Q(0), Q(1)),
        form_snapshots=(_ZERO_FORM, _ZERO_FORM),
        phase_snapshots=(_phases(), _phases()),
        form_error_bound=0,
        phase_error_bound=0,
        exposure_time_error_bound=0,
        velocity_lipschitz_bound=0,
    )
    inputs.update(changes)
    return owner._read_organized_free_phase_kinetic_record(**inputs)


def _cos_reference(value):
    partial = sum(
        (Q((-1) ** k) * value ** (2 * k) / factorial(2 * k) for k in range(32)), Q(0)
    )
    omitted = value**64 / factorial(64)
    return partial, partial + omitted


def test_pair_adapter_demodulates_primitive_phases_and_reuses_child_kernel(monkeypatch):
    seen = []
    original = geometry._observe_child

    def child(**kwargs):
        seen.append(kwargs["child"])
        return original(**kwargs)

    monkeypatch.setattr(geometry, "_observe_child", child)
    forms = tuple(Q(i - 8, 7) for i in range(18))
    offsets = (Q(0),) * 8 + (Q(1),)
    phases = tuple(
        2 * _PI * (j - 4) / 9 + offsets[j] for _ in range(2) for j in range(9)
    )
    report = geometry._observe_marked_c9_pair(form=forms, phase=phases)
    assert seen == [0, 1]
    assert report.available
    for index, frame in enumerate(report.children):
        assert frame.nodes == tuple(range(9 * index, 9 * index + 9))
        assert frame.form_mean == sum(forms[9 * index : 9 * index + 9]) / 9
        assert frame.phase_mean_lift_bounds.contains(Q(1, 9))
        # Arg(8+exp(i)) is different from the arithmetic collar mean 1/9.
        assert frame.phase_anchor_lift_bounds.hi < frame.phase_mean_lift_bounds.lo
        for error, value in zip(frame.centered_phase_bounds, offsets):
            assert error.contains(value - Q(1, 9))


def test_pair_gauge_and_independent_turns_preserve_circular_observation():
    gauge = Q(7, 5)
    turns = tuple((7 * i) % 5 - 2 for i in range(18))
    phases = tuple(
        2 * _PI * (i % 9 - 4) / 9 + gauge + 2 * turns[i] * _PI for i in range(18)
    )
    report = geometry._observe_marked_c9_pair(form=_ZERO_FORM, phase=phases)
    assert report.available
    for c, frame in enumerate(report.children):
        expected = I(gauge) + 2 * turns[9 * c] * _PI
        assert frame.phase_mean_lift_bounds.contains(expected)
        assert all(value.contains(0) for value in frame.centered_phase_bounds)


def test_stationary_and_nonzero_consistent_records_remain_inconclusive():
    stationary = _read()
    assert stationary.status == "inconclusive"
    assert stationary.primitive_phase_speed_bound is None
    assert stationary.cells[0].squared_chord_defect_bounds.contains(0)
    assert not stationary.cells[0].mean_increment_available
    assert stationary.cells[0].product_squared_chord_defect_bounds.contains(0)
    times = (Q(1, 7), Q(2, 5), Q(6, 7))
    gamma = 1 / (1023 * _PI)
    forms = ((Q(1),) * 9 + (Q(0),) * 9,) * len(times)
    report = _read(
        sample_times=times,
        form_snapshots=forms,
        phase_snapshots=tuple(_phases(gamma.midpoint * t) for t in times),
        phase_error_bound=Q(1, 10**30),
    )
    assert report.status == "inconclusive"
    for cell, left, right in zip(report.cells, times, times[1:]):
        assert cell.integrated_velocity_bounds.contains((right - left) * gamma)
        assert cell.circular_residual_bounds.contains(0)
        assert cell.squared_chord_defect_bounds.contains(0)
        assert cell.product_squared_chord_defect_bounds.contains(0)


def test_independent_phase_motion_with_zero_form_excludes_kinetic_identity():
    motion = Q(1, 5)
    report = _read(phase_snapshots=(_phases(), _phases(motion)))
    cell = report.cells[0]
    assert report.status == cell.status == "incompatible_kinetic_record"
    assert cell.integrated_velocity_bounds == I(0)
    assert cell.circular_residual_bounds.contains(motion)
    lower_cos, upper_cos = _cos_reference(motion)
    assert 0 < cell.squared_chord_defect_bounds.lo <= 2 * (1 - upper_cos)
    assert 2 * (1 - lower_cos) <= cell.squared_chord_defect_bounds.hi
    assert cell.squared_chord_defect_bounds.lo > motion**2 / 2
    product_lower, product_upper = _cos_reference(9 * motion)
    assert cell.product_circular_residual_bounds.contains(9 * motion)
    assert 0 < cell.product_squared_chord_defect_bounds.lo <= 2 * (1 - product_upper)
    assert 2 * (1 - product_lower) <= cell.product_squared_chord_defect_bounds.hi


def test_lipschitz_trapezoid_constant_is_sharp_for_an_unseen_triangle():
    # v(s)=min(s,1-s) has unit Lipschitz constant, zero endpoint values and
    # integral 1/4. No kinetic equation is used to bound that integral.
    midpoint, height = Q(1, 2), Q(1, 2)
    exact_area = midpoint * height / 2 + (1 - midpoint) * height / 2
    assert exact_area == Q(1, 4)
    report = _read(
        phase_snapshots=(_phases(), _phases(exact_area)),
        velocity_lipschitz_bound=1,
    )
    cell = report.cells[0]
    assert cell.integrated_velocity_bounds.contains(exact_area)
    assert cell.integrated_velocity_bounds.contains(-exact_area)
    assert cell.squared_chord_defect_bounds.contains(0)
    assert report.status == "inconclusive"
    # Removing the independent between-sample allowance produces a different
    # hypothesis and a separated contradiction for exactly the same readings.
    assert (
        _read(phase_snapshots=report.phase_snapshots).status
        == "incompatible_kinetic_record"
    )


def test_phase_and_velocity_jitter_bounds_use_separate_independent_speeds():
    eps, eta, xi, lv, lt = Q(1, 100), Q(1, 200), Q(1, 300), Q(2, 7), Q(3, 8)
    report = _read(
        form_error_bound=eps,
        phase_error_bound=eta,
        exposure_time_error_bound=xi,
        velocity_lipschitz_bound=lv,
        primitive_phase_speed_bound=lt,
    )
    expected_phase = eta + lt * xi
    expected_velocity = 2 * report.gamma_bounds.hi * eps + lv * xi
    assert report.effective_phase_error_bound == expected_phase
    assert report.velocity_sample_error_upper_bound == expected_velocity
    for sample, frame in zip(report.phase_snapshots, report.snapshots):
        for value, bound in zip(sample, frame.primitive_phase_bounds):
            assert bound.contains(value - expected_phase)
            assert bound.contains(value + expected_phase)
    for velocity in report.velocity_bounds:
        assert velocity.contains(-expected_velocity)
        assert velocity.contains(expected_velocity)
    assert report.cells[0].integrated_velocity_bounds.contains(
        expected_velocity + lv / 4
    )


def test_independent_full_turn_is_hidden_and_cannot_verify_the_law():
    turn = 2 * _PI.midpoint
    report = _read(
        phase_snapshots=(_phases(), _phases(turn)),
        phase_error_bound=Q(1, 10**30),
    )
    cell = report.cells[0]
    assert cell.phase_increment_bounds.lo > 6
    assert cell.circular_residual_bounds.lo > 6
    assert cell.squared_chord_defect_bounds.contains(0)
    assert report.status == "inconclusive"
    assert all("verified" not in item.status for item in report.cells)


def test_arbitrary_common_snapshot_gauges_do_not_change_the_relative_test():
    motion = Q(1, 5)
    origin = Q(19, 7)
    shifted = _read(
        phase_snapshots=(
            _phases(origin, origin),
            _phases(origin + 8 + motion, origin + 8),
        )
    )
    assert shifted.status == "incompatible_kinetic_record"
    assert shifted.cells[0].circular_residual_bounds.contains(motion)


def test_unavailable_collar_keeps_signed_means_and_propagates_no_phase_value():
    report = _read(phase_error_bound=2)
    assert report.status == "inconclusive"
    assert report.relative_phase_bounds == (None, None)
    assert all(not frame.available for frame in report.snapshots)
    assert all(frame.children[0].form_mean == 0 for frame in report.snapshots)
    cell = report.cells[0]
    assert cell.integrated_velocity_bounds == I(0)
    assert cell.phase_increment_bounds is None
    assert cell.circular_residual_bounds is None
    assert cell.squared_chord_defect_bounds is None
    assert not cell.mean_increment_available
    assert cell.mean_increment_unavailable_reason == "endpoint_marked_collar_unresolved"
    assert cell.product_squared_chord_defect_bounds.contains(0)


def test_global_product_detects_a_contradiction_without_a_local_collar():
    phases = (_ZERO_FORM, (Q(1, 5),) + (Q(0),) * 17)
    report = _read(phase_snapshots=phases)
    cell = report.cells[0]
    assert report.status == "incompatible_kinetic_record"
    assert cell.phase_increment_bounds is None
    assert not cell.mean_increment_available
    assert cell.product_circular_residual_bounds.contains(Q(1, 5))
    assert cell.product_squared_chord_defect_bounds.lo > 0


@pytest.mark.parametrize("speed", (None, Q(7)))
def test_hidden_single_node_turn_cannot_make_local_mean_a_false_contradiction(speed):
    gamma = 1 / (1023 * _PI)
    advance = 2 * _PI.midpoint / 9
    contrast = advance / gamma.midpoint
    form = (contrast,) * 9 + (Q(0),) * 9
    final = list(_phases())
    final[1] += 2 * _PI.midpoint
    report = _read(
        form_snapshots=(form, form),
        phase_snapshots=(_phases(), tuple(final)),
        phase_error_bound=Q(1, 10**28),
        primitive_phase_speed_bound=speed,
    )
    cell = report.cells[0]
    assert all(frame.available for frame in report.snapshots)
    assert cell.product_squared_chord_defect_bounds.contains(0)
    assert cell.squared_chord_defect_bounds.lo > 0
    assert not cell.mean_increment_available
    assert report.status == "inconclusive"
    if speed is not None:
        assert max(cell.mean_lift_margin_lower_bounds) < 0


def test_independent_no_slip_bound_resolves_a_product_phase_alias():
    advance = 2 * _PI.midpoint / 9
    arguments = dict(
        phase_snapshots=(_phases(), _phases(advance)),
        phase_error_bound=Q(1, 10**28),
    )
    unresolved = _read(**arguments)
    assert unresolved.cells[0].product_squared_chord_defect_bounds.contains(0)
    assert unresolved.cells[0].squared_chord_defect_bounds.lo > 0
    assert unresolved.status == "inconclusive"
    resolved = _read(**arguments, primitive_phase_speed_bound=1)
    cell = resolved.cells[0]
    assert cell.product_squared_chord_defect_bounds.contains(0)
    assert cell.mean_increment_available
    assert min(cell.mean_lift_margin_lower_bounds) > 1
    assert cell.mean_increment_unavailable_reason is None
    assert resolved.status == "incompatible_kinetic_record"


def test_collective_checks_do_not_verify_zero_mean_internal_phase_dynamics():
    final = list(_phases())
    final[1] += Q(1, 5)
    final[2] -= Q(1, 5)
    report = _read(
        phase_snapshots=(_phases(), tuple(final)),
        primitive_phase_speed_bound=Q(1, 5),
    )
    cell = report.cells[0]
    assert report.phase_snapshots[0] != report.phase_snapshots[1]
    assert cell.mean_increment_available
    assert cell.product_squared_chord_defect_bounds.contains(0)
    assert cell.squared_chord_defect_bounds.contains(0)
    assert report.status == "inconclusive"


def test_tiny_exact_forms_and_fixed_width_payloads_are_preserved_and_immutable():
    np = pytest.importorskip("numpy")
    tiny = -Q(1, 10**500)
    row = (tiny, Q(np.int64(1), np.int64(7))) + (Q(0),) * 16
    report = _read(form_snapshots=(row, row))
    assert report.form_snapshots[0][0] == tiny
    assert report.form_snapshots[0][1] == Q(1, 7)
    for value in report.form_snapshots[0]:
        assert type(value.numerator) is type(value.denominator) is int
    for object_, field, value in (
        (report, "status", "verified"),
        (report.cells[0], "status", "verified"),
        (report.snapshots[0], "available", False),
    ):
        with pytest.raises(FrozenInstanceError):
            setattr(object_, field, value)


@pytest.mark.parametrize("field", ("form_snapshots", "phase_snapshots"))
@pytest.mark.parametrize(
    "invalid", (True, None, "0", 1j, float("nan"), float("inf"), I(0))
)
def test_all_record_scalars_are_admitted_before_geometry(monkeypatch, field, invalid):
    def forbidden(**kwargs):
        pytest.fail("geometry evaluated before all primitive values were admitted")

    monkeypatch.setattr(owner, "_observe_marked_c9_pair", forbidden)
    row = (Q(0),) * 17 + (invalid,)
    with pytest.raises((TypeError, ValueError)):
        _read(
            **{field: ((_ZERO_FORM if field == "form_snapshots" else _phases()), row)}
        )


@pytest.mark.parametrize(
    "field",
    (
        "form_error_bound",
        "phase_error_bound",
        "exposure_time_error_bound",
        "velocity_lipschitz_bound",
        "primitive_phase_speed_bound",
    ),
)
@pytest.mark.parametrize("invalid", (True, -Q(1, 10**90), "0", float("nan")))
def test_invalid_error_and_speed_bounds_reject_before_geometry(
    monkeypatch, field, invalid
):
    monkeypatch.setattr(
        owner, "_observe_marked_c9_pair", lambda **_: pytest.fail("early geometry")
    )
    with pytest.raises((TypeError, ValueError)):
        _read(**{field: invalid})


def test_nonzero_jitter_needs_phase_speed_but_exact_times_do_not():
    with pytest.raises(ValueError, match="requires primitive_phase_speed_bound"):
        _read(exposure_time_error_bound=Q(1, 100))
    assert (
        _read(phase_snapshots=(_phases(), _phases(Q(1, 5)))).primitive_phase_speed_bound
        is None
    )


@pytest.mark.parametrize(
    "times",
    (
        (),
        (Q(0),),
        (Q(0), Q(0)),
        (Q(1), Q(0)),
        (Q(-1), Q(1)),
        (Q(0), Q(2)),
        (True, Q(1)),
        tuple(Q(i, 128) for i in range(129)),
    ),
)
def test_invalid_or_unordered_nominal_times_reject(times):
    with pytest.raises((TypeError, ValueError)):
        _read(sample_times=times)


@pytest.mark.parametrize("field", ("sample_times", "form_snapshots", "phase_snapshots"))
@pytest.mark.parametrize(
    "invalid", ({0, 1}, frozenset((0, 1)), {0: 0}, UserDict({0: 0}), "01")
)
def test_mapping_and_unordered_outer_inputs_reject(field, invalid):
    with pytest.raises(TypeError):
        _read(**{field: invalid})


def test_snapshot_count_and_marked_coordinate_order_are_not_inferred():
    for field in ("form_snapshots", "phase_snapshots"):
        for rows in (
            (_ZERO_FORM,),
            (_ZERO_FORM,) * 3,
            ((0,) * 17, _ZERO_FORM),
            (_ZERO_FORM, set(range(18))),
        ):
            with pytest.raises((TypeError, ValueError)):
                _read(**{field: rows})
    # Reversing a winding-one marked frame destroys the prescribed collar;
    # the observer must not silently sort or relabel its channels.
    reversed_frame = tuple(reversed(_CARRIER)) * 2
    report = _read(phase_snapshots=(reversed_frame,) * 2)
    assert report.status == "inconclusive"
    assert (
        report.cells[0].mean_increment_unavailable_reason
        == "endpoint_marked_collar_unresolved"
    )


def test_count_work_limit_stops_an_unbounded_time_iterator():
    seen = []

    def times():
        for i in range(1000):
            seen.append(i)
            yield Q(i, 1000)

    with pytest.raises(ValueError, match="128"):
        _read(sample_times=times())
    assert len(seen) == 129


def test_report_objects_cannot_replace_primitive_records():
    report = _read()
    for field in (
        "sample_times",
        "form_snapshots",
        "phase_snapshots",
        "form_error_bound",
    ):
        with pytest.raises((TypeError, ValueError)):
            _read(**{field: report})


@pytest.mark.parametrize("field", ("form", "phase"))
def test_pair_adapter_admits_every_primitive_before_child_geometry(monkeypatch, field):
    monkeypatch.setattr(
        geometry, "_observe_child", lambda **_: pytest.fail("early geometry")
    )
    arguments = dict(form=_ZERO_FORM, phase=_phases())
    arguments[field] = (Q(0),) * 17 + (True,)
    with pytest.raises((TypeError, ValueError)):
        geometry._observe_marked_c9_pair(**arguments)
