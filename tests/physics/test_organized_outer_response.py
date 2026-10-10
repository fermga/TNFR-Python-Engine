"""Independent static checks of the frozen acquired-organization response."""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.physics import _organized_outer_response as owner


_ERROR = Q(1, 10**10)
_Q = Q(1, 10**70)
_EPSILON = Q(1, 1000)
_DELAY = Q(1, 100)
_INTERNAL = tuple((9 * c + j, 9 * c + (j + 1) % 9) for c in range(5) for j in range(9))
_OUTER = tuple((9 * c + j, 9 * ((c + 1) % 5) + j) for c in range(5) for j in range(9))
_EDGES = _INTERNAL + _OUTER


@pytest.fixture(scope="module", autouse=True)
def no_trajectory_or_frozen_execution():
    with forbid_sine_regeneration():
        yield


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 220
    return context


def _mp(mp, value):
    return mp.mpf(value.numerator) / value.denominator


def _contains(mp, interval, expected):
    assert _mp(mp, interval.lo) <= expected <= _mp(mp, interval.hi)


def _assess(**changes):
    return owner._assess_organized_outer_response(
        **(
            dict(form_error_bounds=(_ERROR,) * 5, phase_error_bounds=(_ERROR,) * 5)
            | changes
        )
    )


@pytest.fixture(scope="module")
def report():
    return _assess()


def _fields(mp, form, phase):
    """Complete edge sums, independent of every certificate field."""
    gradient, current = [mp.mpf(0)] * 45, [mp.mpf(0)] * 45
    storage = mp.mpf(0)
    for left, right in _EDGES:
        gap = form[left] - form[right]
        angle = phase[right] - phase[left]
        gradient[left] += gap
        gradient[right] -= gap
        current[left] += mp.sin(angle)
        current[right] -= mp.sin(angle)
        storage += gap**2 / 2 + 1 - mp.cos(angle)
    gamma = 1 / (1023 * mp.pi)
    dx = tuple((-a + gamma * b) / 4 for a, b in zip(gradient, current))
    dt = tuple(gamma * a / 4 for a in gradient)
    return dx, dt, storage


def _phase(mp, outer, residual=None):
    gamma = 1 / (1023 * mp.pi)
    if residual is None:
        residual = (mp.mpf(0),) * 45
    return tuple(
        2 * mp.pi * (j - 4) / 9
        + (gamma * 4092 * (c - 2) if outer else 0)
        + residual[9 * c + j]
        for c in range(5)
        for j in range(9)
    )


def _probe_phase(mp, phase):
    return tuple(
        value + _mp(mp, _EPSILON) * (int(i // 9 == 0) - mp.mpf(1) / 5)
        for i, value in enumerate(phase)
    )


def _mean(values, child):
    return sum(values[9 * child : 9 * child + 9]) / 9


def test_supplied_probe_preserves_forms_means_and_all_internal_gaps():
    form = tuple(Q((7 * i) % 19 - 9, 64) for i in range(45))
    phase = tuple(Q((11 * i) % 23 - 11, 256) for i in range(45))
    after_x, after_t = owner._apply_outer_response_probe(form=form, phase_offsets=phase)
    assert after_x == form
    assert sum(after_t) == sum(phase)
    assert tuple(_mean(after_t, c) - _mean(phase, c) for c in range(5)) == (
        4 * _EPSILON / 5,
        -_EPSILON / 5,
        -_EPSILON / 5,
        -_EPSILON / 5,
        -_EPSILON / 5,
    )
    assert _mean(after_x, 1) - _mean(form, 1) == 0
    for left, right in _INTERNAL:
        assert after_t[right] - after_t[left] == phase[right] - phase[left]
    changes = tuple(
        after_t[right] - after_t[left] - phase[right] + phase[left]
        for left, right in _OUTER
    )
    assert changes.count(-_EPSILON) == changes.count(_EPSILON) == 9
    assert changes.count(Q(0)) == 27


def test_jump_does_not_project_out_any_fine_source_coordinate():
    base = (Q(0),) * 45
    base_after = owner._apply_outer_response_probe(form=base, phase_offsets=base)
    # Two rows, five children and eight independent relative directions.
    # The affine event is the identity on every one of these eighty modes.
    for channel in range(2):
        for child in range(5):
            for slot in range(8):
                perturbation = [Q(0)] * 45
                perturbation[9 * child + slot] = _Q / 2
                perturbation[9 * child + 8] = -_Q / 2
                state = [base, base]
                state[channel] = tuple(perturbation)
                actual = owner._apply_outer_response_probe(
                    form=state[0], phase_offsets=state[1]
                )
                for row in range(2):
                    difference = tuple(
                        a - b for a, b in zip(actual[row], base_after[row])
                    )
                    assert difference == (
                        tuple(perturbation) if row == channel else base
                    )


@pytest.mark.parametrize("outer", (False, True))
def test_receiver_initial_increment_and_signed_jump_match_all_ninety_rows(
    mp, report, outer
):
    # A nonstationary point in the source enclosure; it is not asserted to
    # be a particular recovered trajectory. Fine modes retain child means.
    q = _mp(mp, _Q)
    residual = tuple(q * (i % 9 - 4) * (i // 9 + 1) / 100 for i in range(45))
    x = tuple(q * (i % 9 - 4) * (5 - i // 9) / 100 for i in range(45))
    phase = _phase(mp, outer, residual)
    probed = _probe_phase(mp, phase)
    before_dx, before_dt, before_storage = _fields(mp, x, phase)
    after_dx, after_dt, after_storage = _fields(mp, x, probed)
    expected_slope = _mean(tuple(a - b for a, b in zip(after_dx, before_dx)), 1)
    gamma, epsilon = 1 / (1023 * mp.pi), _mp(mp, _EPSILON)
    direct = (
        gamma
        / 36
        * sum(
            mp.sin(phase[j] - phase[9 + j] + epsilon) - mp.sin(phase[j] - phase[9 + j])
            for j in range(9)
        )
    )
    assert abs(expected_slope - direct) < mp.mpf("1e-200")
    assert before_dt == after_dt
    assert max(abs(value) for value in before_dt) > 0
    name = "outer_one" if outer else "outer_zero"
    _contains(mp, getattr(report, name + "_response_slope_bounds"), expected_slope)
    work = after_storage - before_storage
    _contains(mp, getattr(report, name + "_event_work_bounds"), work)
    assert abs(work) < _mp(
        mp, getattr(report, name + "_event_work_absolute_upper_bound")
    )
    assert abs(work) < mp.mpf(".001")
    assert work < 0 if outer else work > 0


def test_formed_source_is_not_replaced_by_its_limiting_outer_twist(mp, report):
    gamma, epsilon = 1 / (1023 * mp.pi), _mp(mp, _EPSILON)
    reached_slope = gamma * (mp.sin(epsilon - 4 / mp.pi) + mp.sin(4 / mp.pi)) / 4
    limiting_slope = (
        gamma * (mp.sin(epsilon - 2 * mp.pi / 5) + mp.sin(2 * mp.pi / 5)) / 4
    )
    actual = report.outer_one_response_slope_bounds
    _contains(mp, actual, reached_slope)
    assert not _mp(mp, actual.lo) <= limiting_slope <= _mp(mp, actual.hi)
    nominal_phase = _phase(mp, True)
    dx, _, _ = _fields(mp, (mp.mpf(0),) * 45, nominal_phase)
    assert abs(_mean(dx, 0)) > mp.mpf("1e-6")
    assert abs(reached_slope - limiting_slope) > mp.mpf("1e-10")


def test_probe_finite_nonlinear_remainders_and_four_raw_errors_are_retained(mp, report):
    gamma, epsilon, h = 1 / (1023 * mp.pi), _mp(mp, _EPSILON), _mp(mp, _DELAY)
    for name in ("outer_zero", "outer_one"):
        x = _mp(mp, getattr(report, name + "_source_form_upper_bound"))
        reference = (
            2 * gamma * epsilon * h**2
            + 2 * gamma**2 * epsilon * (x + gamma * h) * h**2
            + mp.mpf(4) * gamma**3 * epsilon * h**3 / 3
        )
        remainder = getattr(report, name + "_finite_remainder_upper_bound")
        assert reference <= _mp(mp, remainder) < mp.mpf("7e-11")
        slope = getattr(report, name + "_response_slope_bounds")
        response = getattr(report, name + "_response_bounds")
        assert response.lo <= _DELAY * slope.lo - remainder
        assert response.hi >= _DELAY * slope.hi + remainder
        assert response.width > 2 * remainder > 0
    contrast = report.contrast_bounds
    assert (
        contrast.lo
        <= report.outer_zero_response_bounds.lo - report.outer_one_response_bounds.hi
    )
    assert (
        contrast.hi
        >= report.outer_zero_response_bounds.hi - report.outer_one_response_bounds.lo
    )
    observed = report.observed_contrast_bounds
    assert observed.lo <= contrast.lo - Q(4, 10**12)
    assert observed.hi >= contrast.hi + Q(4, 10**12)
    assert observed.lo > Q(1, 10**10)
    assert report.response_contrast_margin == observed.lo - Q(1, 10**10)
    # A signed extremal error on each raw endpoint attains the complete
    # allowance; averaging does not remove arbitrary common/adversarial noise.
    noises = (Q(1, 10**12), -Q(1, 10**12), -Q(1, 10**12), Q(1, 10**12))
    assert noises[0] - noises[1] - noises[2] + noises[3] == Q(4, 10**12)


def test_complete_vector_field_has_nonzero_curvature_within_the_finite_allowance(
    mp, report
):
    # This is a static Jacobian action, not a numerical integration. At the
    # ideal control point the receiver increment has R''(0)=-R'(0), so
    # extrapolating the initial slope alone already loses a nonzero term.
    zero = (mp.mpf(0),) * 45
    base = _phase(mp, False)
    probed = _probe_phase(mp, base)
    increment_derivatives = []
    gamma = 1 / (1023 * mp.pi)
    for phase in (base, probed):
        dx, dt, _ = _fields(mp, zero, phase)
        lap_dx, ds = [mp.mpf(0)] * 45, [mp.mpf(0)] * 45
        for left, right in _EDGES:
            difference = dx[left] - dx[right]
            current_rate = mp.cos(phase[right] - phase[left]) * (dt[right] - dt[left])
            lap_dx[left] += difference
            lap_dx[right] -= difference
            ds[left] += current_rate
            ds[right] -= current_rate
        ddx = tuple((-lap + gamma * rate) / 4 for lap, rate in zip(lap_dx, ds))
        step = mp.mpf("1e-70")
        advanced_x = tuple(a + mp.j * step * b for a, b in zip(zero, dx))
        advanced_phase = tuple(a + mp.j * step * b for a, b in zip(phase, dt))
        numerical_action = tuple(
            mp.im(v) / step for v in _fields(mp, advanced_x, advanced_phase)[0]
        )
        assert max(abs(a - b) for a, b in zip(numerical_action, ddx)) < mp.mpf("1e-130")
        increment_derivatives.append((_mean(dx, 1), _mean(ddx, 1)))
    slope = increment_derivatives[1][0] - increment_derivatives[0][0]
    curvature = increment_derivatives[1][1] - increment_derivatives[0][1]
    assert abs(curvature + slope) < mp.mpf("1e-200")
    assert abs(curvature) > 0
    correction = abs(curvature) * _mp(mp, _DELAY) ** 2 / 2
    assert 0 < correction < _mp(mp, report.outer_zero_finite_remainder_upper_bound)


@pytest.mark.parametrize("field", ("form", "phase_offsets"))
@pytest.mark.parametrize(
    "bad", (True, "0", None, float("nan"), float("inf"), complex(0))
)
def test_probe_admits_every_fine_primitive_before_applying_jump(field, bad):
    state = dict(form=(Q(0),) * 45, phase_offsets=(Q(0),) * 45)
    state[field] = (Q(0),) * 44 + (bad,)
    with pytest.raises((TypeError, ValueError)):
        owner._apply_outer_response_probe(**state)


@pytest.mark.parametrize("field", ("form", "phase_offsets"))
@pytest.mark.parametrize("bad", ((), (Q(0),) * 44, (Q(0),) * 46, "0" * 45, {0}))
def test_probe_requires_both_ordered_complete_fine_vectors(field, bad):
    state = dict(form=(Q(0),) * 45, phase_offsets=(Q(0),) * 45)
    state[field] = bad
    with pytest.raises((TypeError, ValueError)):
        owner._apply_outer_response_probe(**state)


@pytest.mark.parametrize("field", ("form_error_bounds", "phase_error_bounds"))
@pytest.mark.parametrize(
    "bad", (True, "0", None, float("nan"), float("inf"), complex(0), -Q(1, 10**1000))
)
def test_original_allowances_are_admitted_before_any_source_rebuild(
    field, bad, monkeypatch
):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid original primitive reached source rebuilding")

    monkeypatch.setattr(owner, "_assess_organized_outer_winding", forbidden)
    with pytest.raises((TypeError, ValueError)):
        _assess(**{field: (_ERROR,) * 4 + (bad,)})


@pytest.mark.parametrize("field", ("form_error_bounds", "phase_error_bounds"))
@pytest.mark.parametrize("index", range(5))
def test_larger_original_allowances_do_not_retune_the_protocol(
    field, index, monkeypatch
):
    def forbidden(*args, **kwargs):
        pytest.fail("outside-protocol primitive reached source rebuilding")

    monkeypatch.setattr(owner, "_assess_organized_outer_winding", forbidden)
    values = [_ERROR] * 5
    values[index] += Q(1, 10**1000)
    with pytest.raises(ValueError):
        _assess(**{field: tuple(values)})


@pytest.mark.parametrize("field", ("form_error_bounds", "phase_error_bounds"))
@pytest.mark.parametrize("bad", ((), (_ERROR,) * 4, (_ERROR,) * 6, "00000", {_ERROR}))
def test_original_error_shapes_are_not_truncated_or_coerced(field, bad, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid shape reached source rebuilding")

    monkeypatch.setattr(owner, "_assess_organized_outer_winding", forbidden)
    with pytest.raises((TypeError, ValueError)):
        _assess(**{field: bad})


@pytest.mark.parametrize(
    "extra",
    (
        "source",
        "acquisition",
        "epsilon",
        "delay",
        "clock",
        "horizon",
        "noise",
        "trajectory",
    ),
)
def test_cached_reports_and_retuned_choices_cannot_replace_original_primitives(extra):
    with pytest.raises(TypeError):
        _assess(**{extra: 0})


def test_reports_are_immutable(report):
    with pytest.raises(FrozenInstanceError):
        report.admitted = False


def test_frozen_source_clock_four_readouts_and_all_eighty_directions(report):
    assert report.protocol_revision == "50ce12a5"
    assert report.acquisition.protocol_revision == "82825de5"
    assert report.source_organization_order == (1, 0)
    assert report.acquisition.relative_preparation_dimension == 80
    assert len(report.acquisition.child_sources) == 5
    assert len({id(child) for child in report.acquisition.child_sources}) == 5
    assert report.form_error_bounds == report.phase_error_bounds == (_ERROR,) * 5
    assert report.source_scaled_time == report.acquisition.elapsed_scaled_time == 200
    assert report.probe_amplitude == _EPSILON
    assert report.elapsed_probe_scaled_time == _DELAY
    assert report.elapsed_probe_original_time == _DELAY * Q(1024, 1023)
    assert report.receiver_child == 1
    assert report.raw_readout_error_upper_bound == Q(1, 10**12)
    assert report.raw_readout_error_contributions == (Q(1, 10**12),) * 4
    assert report.increment_readout_error_upper_bound == Q(2, 10**12)
    assert report.contrast_readout_error_upper_bound == Q(4, 10**12)
    assert report.minimum_observed_contrast == Q(1, 10**10)
    assert report.probe_work_budget == Q(1, 1000)
    assert report.source_full_form_error_upper_bound > 0
    assert report.source_full_phase_error_upper_bound > 0
    assert report.outer_zero_source_form_upper_bound > 0
    assert (
        report.outer_zero_source_form_upper_bound
        == report.source_full_form_error_upper_bound
    )
    assert all(value > 0 for _, value in report.sufficient_margins)
    for flag in (
        "source_certified",
        "conditional_response_certified",
        "work_certified",
        "child_identities_retained",
        "outer_identities_retained",
        "finite_response_certified",
        "contrast_certified",
        "post_probe_capture_certified",
        "subsequent_maintenance_certified",
        "admitted",
    ):
        assert getattr(report, flag)
    assert not report.unresolved_conditions


@pytest.mark.parametrize("outer", (False, True))
def test_all_five_internal_and_nine_outer_cycles_survive_the_actual_jump(
    mp, report, outer
):
    q = _mp(mp, _Q)
    residual = tuple(q * (i % 9 - 4) * (i // 9 + 1) / 100 for i in range(45))
    phase = _phase(mp, outer, residual)
    probed = _probe_phase(mp, phase)
    index = 0 if outer else 1
    source_boxes, probed_boxes = (
        report.source_outer_gap_bounds[index],
        report.probed_outer_gap_bounds[index],
    )
    assert len(source_boxes) == len(probed_boxes) == 45
    for current, boxes in ((phase, source_boxes), (probed, probed_boxes)):
        for child in range(5):
            gaps = tuple(
                (current[9 * child + (j + 1) % 9] - current[9 * child + j] + mp.pi)
                % (2 * mp.pi)
                - mp.pi
                for j in range(9)
            )
            assert all(0 < gap < mp.pi / 2 for gap in gaps)
            assert abs(sum(gaps) - 2 * mp.pi) < mp.mpf("1e-200")
        for slot in range(9):
            gaps = tuple(
                (
                    current[9 * ((child + 1) % 5) + slot]
                    - current[9 * child + slot]
                    + mp.pi
                )
                % (2 * mp.pi)
                - mp.pi
                for child in range(5)
            )
            assert all(abs(gap) < mp.pi / 2 for gap in gaps)
            assert abs(sum(gaps) - int(outer) * 2 * mp.pi) < mp.mpf("1e-200")
            for child, gap in enumerate(gaps):
                _contains(mp, boxes[5 * slot + child], gap)


def test_full_post_jump_storage_uses_signed_work_and_separate_sector_barriers(report):
    acquired = report.acquisition
    assert report.outer_one_event_work_bounds.hi < 0
    assert report.outer_zero_event_work_bounds.lo > 0
    assert report.outer_one_post_probe_excess_upper_bound == (
        acquired.endpoint_excess_upper_bound + report.outer_one_event_work_bounds.hi
    )
    assert report.outer_zero_post_probe_excess_upper_bound == (
        acquired.control.initial_excess_upper_bound
        + report.outer_zero_event_work_bounds.hi
    )
    assert (
        report.outer_one_post_probe_capture_margin
        == (
            acquired.boundary_energy_lower_bound
            - report.outer_one_post_probe_excess_upper_bound
        )
        > acquired.capture_margin
        > 0
    )
    assert (
        acquired.control.capture_margin
        > report.outer_zero_post_probe_capture_margin
        > 0
    )
    assert report.outer_zero_post_probe_capture_margin == (
        acquired.control.boundary_energy_lower_bound
        - report.outer_zero_post_probe_excess_upper_bound
    )
    assert report.outer_one_event_work_absolute_upper_bound > 0
    assert report.outer_zero_event_work_absolute_upper_bound > 0
    assert acquired.attachment_nominal_work == Q(1507001760)
    assert acquired.attachment_work_upper_bound > acquired.attachment_nominal_work


@pytest.mark.parametrize(
    "field", ("admitted", "capture_certified", "fine_error_certified")
)
def test_unavailable_actual_source_cannot_inherit_a_conditional_response(
    field, report, monkeypatch
):
    altered = replace(report.acquisition, **{field: False})
    monkeypatch.setattr(
        owner, "_assess_organized_outer_winding", lambda **kwargs: altered
    )
    failed = _assess()
    assert failed.conditional_response_certified
    assert failed.observed_contrast_bounds == report.observed_contrast_bounds
    assert not failed.source_certified and not failed.admitted
    assert not failed.finite_response_certified and not failed.contrast_certified
    assert not failed.child_identities_retained and not failed.outer_identities_retained
    assert not failed.subsequent_maintenance_certified
    assert failed.status == "unavailable" and failed.unresolved_conditions


def test_unavailable_paired_control_is_not_assumed_stationary(report, monkeypatch):
    control = replace(report.acquisition.control, capture_certified=False)
    altered = replace(report.acquisition, control=control)
    monkeypatch.setattr(
        owner, "_assess_organized_outer_winding", lambda **kwargs: altered
    )
    failed = _assess()
    assert failed.conditional_response_certified
    assert not failed.source_certified and not failed.admitted
    assert not failed.finite_response_certified and not failed.contrast_certified
    assert "actual_outer_zero_capture_not_certified" in failed.unresolved_conditions


def test_strict_work_budget_does_not_replace_response_or_identity_checks(
    report, monkeypatch
):
    monkeypatch.setattr(
        owner, "_WORK_BUDGET", report.outer_zero_event_work_absolute_upper_bound
    )
    monkeypatch.setattr(
        owner, "_assess_organized_outer_winding", lambda **kwargs: report.acquisition
    )
    failed = _assess()
    assert failed.source_certified and failed.finite_response_certified
    assert failed.contrast_certified and failed.subsequent_maintenance_certified
    assert not failed.work_certified and not failed.admitted
    assert "outer_zero_absolute_probe_work" in failed.unresolved_conditions


def test_unsupported_noise_resolution_is_unavailable_without_loosening_identity(
    report, monkeypatch
):
    monkeypatch.setattr(owner, "_RAW_ERROR", Q(1, 10**8))
    monkeypatch.setattr(
        owner, "_assess_organized_outer_winding", lambda **kwargs: report.acquisition
    )
    failed = _assess()
    assert failed.source_certified and failed.finite_response_certified
    assert failed.work_certified and failed.subsequent_maintenance_certified
    assert not failed.contrast_certified and not failed.admitted
    assert failed.observed_contrast_bounds.lo < 0


def test_post_jump_capture_must_be_readmitted_even_with_valid_previous_identity(
    report, monkeypatch
):
    altered = replace(report.acquisition, endpoint_excess_upper_bound=Q(1))
    monkeypatch.setattr(
        owner, "_assess_organized_outer_winding", lambda **kwargs: altered
    )
    failed = _assess()
    assert failed.source_certified and failed.contrast_certified
    assert failed.work_certified
    assert not failed.subsequent_maintenance_certified and not failed.admitted
    assert "outer_one_complete_post_probe_capture" in failed.unresolved_conditions


def test_primitive_allowance_pairs_are_normalized_then_sent_to_one_fresh_source(
    report, monkeypatch
):
    calls = []
    forms = tuple(_ERROR * Q(c + 1, 5) for c in range(5))
    phases = tuple(_ERROR * Q(5 - c, 5) for c in range(5))

    def recorded(**kwargs):
        calls.append(kwargs)
        # Wiring only: the fixture supplies no fresh evidence for these subsets.
        return report.acquisition

    monkeypatch.setattr(owner, "_assess_organized_outer_winding", recorded)
    result = _assess(form_error_bounds=forms, phase_error_bounds=phases)
    assert calls == [dict(form_error_bounds=forms, phase_error_bounds=phases)]
    assert result.form_error_bounds == forms and result.phase_error_bounds == phases


@pytest.mark.parametrize("field", ("gamma_upper", "initial_form_upper_bound"))
@pytest.mark.parametrize(
    "bad", (True, "0", None, float("nan"), float("inf"), -Q(1, 10**1000))
)
def test_finite_remainder_admits_represented_real_nonnegative_primitives(field, bad):
    arguments = dict(gamma_upper=Q(1, 1000), initial_form_upper_bound=Q(1, 100))
    arguments[field] = bad
    with pytest.raises((TypeError, ValueError)):
        owner._outer_response_finite_remainder(**arguments)


def test_finite_remainder_has_a_checked_domain_and_retains_background_dependence():
    first = owner._outer_response_finite_remainder(
        gamma_upper=Q(1, 1000), initial_form_upper_bound=Q(0)
    )
    second = owner._outer_response_finite_remainder(
        gamma_upper=Q(1, 1000), initial_form_upper_bound=Q(1)
    )
    assert second > first > 0
    with pytest.raises(ValueError):
        owner._outer_response_finite_remainder(
            gamma_upper=Q(100), initial_form_upper_bound=Q(0)
        )
