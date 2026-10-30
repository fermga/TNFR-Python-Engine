"""Unequal conductance controls for the existing detached bare-sine law.

All static edge sums below distinguish weighted form transport from bare sine
pressure. The explicit nonclosure pair lies in the admitted error neighborhood;
it is not asserted to be two evaluated endpoints of the source preparation.
"""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q
from itertools import repeat

import mpmath
import numpy as np
import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics import integrators, runtime
from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.operators import network_stage, word_execution
from tnfr.operators.definitions_base import Operator
from tnfr.physics import _sine_flow as flow_owner
from tnfr.physics import _sine_mixed_conductance as owner

ERROR = Q(1, 10**10)
NOISE = Q(1, 10**6)
ETA = Q(2, 10**69)
DELTA = Q(1, 1024)
WEIGHTS = (Q(3, 4), Q(5, 4)) + (Q(1),) * 7
SCALE = Q(6, 5)


def _assess(**changes):
    values = dict(
        contact_conductances=WEIGHTS,
        form_error_bound=ERROR,
        phase_error_bound=ERROR,
        observation_error_bound=NOISE,
    )
    values.update(changes)
    return owner._assess_sine_mixed_conductance(**values)


def _mp(context, value):
    value = Q(value)
    return context.mpf(value.numerator) / value.denominator


def _contains(context, interval, value):
    assert _mp(context, interval.lo) <= value <= _mp(context, interval.hi)
    assert _mp(context, interval.hi - interval.lo) < context.mpf("1e-30")


def _geometry(weights=WEIGHTS):
    p = tuple(
        next(k for k in range(9) if (2 * (k - 4) - (j - 4)) % 9 == 0) for j in range(9)
    )
    internal = tuple(
        (9 * r + j, 9 * r + (j + 1) % 9, Q(1)) for r in (0, 1) for j in range(9)
    )
    contact = tuple((j, 9 + p[j], weight) for j, weight in enumerate(weights))
    return p, internal, contact


def _state(context, forms=None, errors=None, gap=DELTA):
    forms = (Q(0),) * 18 if forms is None else forms
    errors = (Q(0),) * 18 if errors is None else errors
    phase = tuple(
        (1 + i // 9) * 2 * context.pi * (i % 9 - 4) / 9
        + (-1 if i < 9 else 1) * _mp(context, gap) / 2
        + _mp(context, errors[i])
        for i in range(18)
    )
    return tuple(_mp(context, x) for x in forms), phase


def _interval_state(forms, errors=None, gap=DELTA):
    errors = (Q(0),) * 18 if errors is None else errors
    phase = tuple(
        2 * pi_interval() * Q((1 + i // 9) * (i % 9 - 4), 9)
        + (-1 if i < 9 else 1) * gap / 2
        + errors[i]
        for i in range(18)
    )
    return tuple(I(value) for value in forms) + phase


def _rows(context, forms, phase, weights=WEIGHTS, joined=True):
    _, internal, contact = _geometry(weights)
    edges = internal + (contact if joined else ())
    gradient, sine, weighted_sine = (
        [context.mpf(0) for _ in range(18)] for _ in range(3)
    )
    strengths = [Q(0)] * 18
    energy = work = context.mpf(0)
    for i, j, raw_weight in edges:
        weight = _mp(context, raw_weight)
        difference = forms[i] - forms[j]
        gradient[i] += weight * difference
        gradient[j] -= weight * difference
        current = context.sin(phase[j] - phase[i])
        sine[i] += current
        sine[j] -= current
        weighted_sine[i] += weight * current
        weighted_sine[j] -= weight * current
        strengths[i] += raw_weight
        strengths[j] += raw_weight
        cost = weight * difference**2 / 2 + 1 - context.cos(phase[j] - phase[i])
        energy += cost
        if (i, j, raw_weight) in contact:
            work += cost
    gamma, degree = 1 / (1023 * context.pi), 3 if joined else 2
    dx = tuple((-g + gamma * s) / degree for g, s in zip(gradient, sine))
    dt = tuple(gamma * g / degree for g in gradient)
    return (
        dx,
        dt,
        tuple(gradient),
        tuple(sine),
        tuple(weighted_sine),
        tuple(strengths),
        energy,
        work,
    )


def _matrix(edges, context=None):
    zero = Q(0) if context is None else context.mpf(0)
    result = [[zero for _ in range(18)] for _ in range(18)]
    for i, j, weight in edges:
        result[i][i] += weight
        result[j][j] += weight
        result[i][j] -= weight
        result[j][i] -= weight
    return tuple(tuple(row) for row in result)


def _multiply(left, right):
    return tuple(
        tuple(sum(a * b for a, b in zip(row, column)) for column in zip(*right))
        for row in left
    )


@pytest.fixture(scope="module", autouse=True)
def no_trajectory_operator_or_source_replay():
    static_rate_factory = flow_owner._sine_rate_evaluator

    def forbidden(*args, **kwargs):
        pytest.fail("conductance controls cannot evolve or replay a source")

    with forbid_organized_response_evaluation(), pytest.MonkeyPatch.context() as patch:
        for object_, name in (
            (runtime, "step"),
            (runtime, "run"),
            (runtime, "_update_nodes"),
            (integrators.DefaultIntegrator, "integrate"),
            (Operator, "__call__"),
            (network_stage, "execute_coupling_stage"),
            (network_stage, "execute_neighbor_stage"),
            (network_stage, "execute_pointwise_stage"),
            (network_stage, "execute_operator_major_stage"),
            (word_execution, "execute_network_operator_stage"),
        ):
            patch.setattr(object_, name, forbidden)
        # This protocol admits the static held-field factory. Keep every
        # trajectory, full-field producer and restoration guard in force.
        with pytest.MonkeyPatch.context() as static_patch:
            static_patch.setattr(
                flow_owner, "_sine_rate_evaluator", static_rate_factory
            )
            yield


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 200
    return context


@pytest.fixture(scope="module")
def report():
    return _assess()


def test_marked_weights_bare_degrees_and_strengths_remain_distinct(report):
    p, internal, contact = _geometry()
    assert report.contact_conductances == WEIGHTS
    assert report.weighted_edges == internal + contact
    assert report.bare_degrees == (3,) * 18
    expected = [Q(2)] * 18
    for j, weight in enumerate(WEIGHTS):
        expected[j] += weight
        expected[9 + p[j]] += weight
    assert report.conductance_strengths == tuple(expected)
    assert len(set(expected)) == 3
    assert report.mean_contact_conductance == 1
    assert report.contact_conductance_variance == Q(1, 72)
    assert not report.carrier_compatible
    assert report.actual_source_aliases_certified


def test_compiled_shared_field_matches_weighted_gradient_and_bare_sine(mp):
    forms = tuple(
        Q((i % 9) - 4, 1000) + (Q(1, 100) if i < 9 else Q(-1, 100)) for i in range(18)
    )
    x, theta = _state(mp, forms)
    dx, dt, gradient, sine, weighted_sine, strengths, *_ = _rows(mp, x, theta)
    field = owner._mixed_conductance_field(contact_conductances=WEIGHTS)
    actual = field(_interval_state(forms))
    assert len(actual) == 36
    for interval, expected in zip(actual, dx + dt):
        _contains(mp, interval, expected)
    gamma = 1 / (1023 * mp.pi)
    weighted_sine_alternative = tuple(
        (-g + gamma * s) / 3 for g, s in zip(gradient, weighted_sine)
    )
    strength_loss_alternative = tuple(
        -g / _mp(mp, d) + gamma * s / 3 for g, s, d in zip(gradient, sine, strengths)
    )
    assert max(abs(a - b) for a, b in zip(dx, weighted_sine_alternative)) > mp.mpf(
        "1e-8"
    )
    assert max(abs(a - b) for a, b in zip(dx, strength_loss_alternative)) > mp.mpf(
        "1e-5"
    )
    # Both returned rows use tau. Original time multiplies both by1023/1024.
    for interval, expected in zip(actual, dx + dt):
        _contains(mp, interval * Q(1023, 1024), expected * 1023 / 1024)


def test_initial_sine_pressure_is_independent_of_contact_conductance(mp):
    zeros = (Q(0),) * 18
    x, theta = _state(mp)
    expected_x, expected_theta, *_ = _rows(mp, x, theta)
    for weights in ((Q(3, 4),) * 9, (Q(5, 4),) * 9, WEIGHTS):
        actual = owner._mixed_conductance_field(contact_conductances=weights)(
            _interval_state(zeros)
        )
        for interval, expected in zip(actual, expected_x + expected_theta):
            _contains(mp, interval, expected)
    assert expected_x[0] > 0 and expected_x[9] < 0
    assert all(value == 0 for value in expected_theta)


def test_full_weighted_storage_loss_and_unweighted_phase_work(mp, report):
    forms = tuple(ETA * Q((i % 9) - 4, 20) for i in range(18))
    errors = tuple(value * (1 if i < 9 else -1) for i, value in enumerate(forms))
    x, theta = _state(mp, forms, errors)
    dx, dt, gradient, sine, _, _, energy, work = _rows(mp, x, theta)
    cut = _rows(mp, x, theta, joined=False)
    assert abs(energy - cut[-2] - work) < mp.mpf("1e-190")
    nominal = 9 * (1 - mp.cos(_mp(mp, DELTA)))
    _contains(mp, report.nominal_attachment_work_bounds, nominal)
    assert 0 < work - nominal < _mp(mp, Q(9, 2) * ETA**2)
    assert report.attachment_work_correction_upper_bound == Q(9, 2) * ETA**2
    assert abs(
        sum(g * a - s * b for g, s, a, b in zip(gradient, sine, dx, dt))
        + sum(g**2 for g in gradient) / 3
    ) < mp.mpf("1e-190")
    assert abs(sum(dx)) < mp.mpf("1e-190") and abs(sum(dt)) < mp.mpf("1e-190")
    # No phase weight belongs in the event work, even for uniform 5/4.
    zero_x, nominal_theta = _state(mp)
    weighted = _rows(mp, zero_x, nominal_theta, (Q(5, 4),) * 9)
    assert abs(weighted[-1] - nominal) < mp.mpf("1e-190")
    assert abs(weighted[-1] - mp.mpf(5) * nominal / 4) > mp.mpf("1e-7")


def test_box_comparison_is_coefficientwise_and_phase_rescaling_is_exact(report):
    _, internal, contact = _geometry()
    edges = internal + contact
    unit = tuple((i, j, Q(1)) for i, j, _ in edges)
    h, base = _matrix(edges), _matrix(unit)
    lower_remainder = _matrix(tuple((i, j, weight - Q(3, 4)) for i, j, weight in edges))
    upper_remainder = _matrix(tuple((i, j, Q(5, 4) - weight) for i, j, weight in edges))
    assert all(
        weight - Q(3, 4) >= 0 and Q(5, 4) - weight >= 0 for _, _, weight in edges
    )
    assert h == tuple(
        tuple(Q(3, 4) * base[i][j] + lower_remainder[i][j] for j in range(18))
        for i in range(18)
    )
    assert h == tuple(
        tuple(Q(5, 4) * base[i][j] - upper_remainder[i][j] for j in range(18))
        for i in range(18)
    )
    # Each remainder is a sum of nonnegative weighted edge squares. The
    # argument applies to every point in the box, not just its tested members.
    assert Q(1, 20) / Q(5, 4) / SCALE**2 == Q(1, 36)
    assert (Q(4, 3) / SCALE**2) == Q(25, 27) < 1
    assert SCALE / 3300 == Q(1, 2750) and SCALE / 3000 == Q(1, 2500)
    assert Q(3, 4) * Q(2, 5) == Q(3, 10)


def test_rescaled_metric_derivative_from_complete_noncommuting_rows(mp, report):
    forms = tuple(Q((i % 9) - 4, 10000) * (1 if i < 9 else -1) for i in range(18))
    errors = tuple(
        Q((i % 9) ** 2 - Q(68, 3), 100000) * (1 if i < 9 else -1) for i in range(18)
    )
    x, theta = _state(mp, forms, errors, gap=Q(0))
    _, target = _state(mp, gap=Q(0))
    y = tuple(a - b for a, b in zip(theta, target))
    dx, dy, hx, sine, *_ = _rows(mp, x, theta)
    scale, gamma = _mp(mp, SCALE), 1 / (1023 * mp.pi)
    b, db, coupling = (
        tuple(scale * value for value in y),
        tuple(scale * value for value in dy),
        scale * gamma,
    )
    metric = (
        mp.mpf(3) / 4 * sum(a * a + c * c + 2 * coupling * a * c for a, c in zip(x, b))
    )
    derivative = (
        mp.mpf(3)
        / 2
        * sum(
            a * da + c * dc + coupling * (da * c + a * dc)
            for a, c, da, dc in zip(x, b, dx, db)
        )
    )
    assert derivative < -_mp(mp, report.common_metric_decay_rate) * metric
    _, internal, contact = _geometry()
    secants = []
    for i, j, _ in internal + contact:
        base, displacement = target[j] - target[i], y[j] - y[i]
        weight = (
            (mp.sin(base + displacement) - mp.sin(base)) / displacement
            if displacement
            else mp.cos(base)
        )
        secants.append((i, j, weight / scale**2))
    k = _matrix(secants, mp)
    h = _matrix(tuple((i, j, _mp(mp, w)) for i, j, w in internal + contact), mp)
    kb = tuple(sum(a * c for a, c in zip(row, b)) for row in k)
    identity = (
        -(1 - coupling**2) * sum(a * c for a, c in zip(x, hx)) / 2
        - coupling * sum(a * c for a, c in zip(x, kb)) / 2
        - coupling**2 * sum(a * c for a, c in zip(b, kb)) / 2
    )
    assert abs(derivative - identity) < mp.mpf("1e-190")
    assert all(abs(value + s / scale) < mp.mpf("1e-190") for value, s in zip(kb, sine))
    hk, kh = _multiply(h, k), _multiply(k, h)
    assert max(
        abs(a - c) for row, other in zip(hk, kh) for a, c in zip(row, other)
    ) > mp.mpf("1e-3")


def test_fixed_return_and_recorded_contrast_cover_the_whole_box(
    report, record_property
):
    rate = Q(1, 3300) ** 2 / (250 * (1 + Q(1, 2500)))
    ratio = Q(2501, 2499)
    norm = Q(9, 2) * DELTA**2 + 4 * ETA**2
    assert report.common_metric_decay_rate == rate
    assert report.recovery_exponent_lower_bound == rate * 2 * 10**10 > 7
    assert 1 + 1 + Q(1, 2) + Q(1, 6) + Q(1, 24) == Q(65, 24) > Q(27, 10)
    assert Q(27, 10) ** 7 > 1024
    assert report.decay_upper_bound == Q(1, 1024)
    assert report.phase_scale == SCALE
    assert report.common_metric_lower_bound == Q(3, 4) * (1 - Q(1, 2500))
    assert report.common_metric_upper_bound == Q(3, 4) * (1 + Q(1, 2500))
    assert report.common_metric_upper_bound / report.common_metric_lower_bound == ratio
    assert report.initial_joint_norm_squared_upper_bound == norm
    assert report.initial_scaled_norm_squared_upper_bound == SCALE**2 * norm
    returned = ratio * SCALE**2 * norm / 1024
    assert report.returned_joint_norm_squared_upper_bound == returned < DELTA**2 / 128
    assert (
        report.returned_registered_gap_squared_upper_bound
        == 2 * returned
        < DELTA**2 / 64
    )
    assert report.returned_mean_registration_squared_upper_bound == Q(2, 9) * returned
    assert report.post_attachment_excess_upper_bound == Q(13, 4) * norm
    assert report.radius_margin > 0 and report.storage_margin > 0
    assert report.joint_return_margin > 0 and report.registration_margin > 0
    assert report.initial_identity_absent and report.joint_identity_certified
    assert report.connected_registration_bounds == (-DELTA / 8, DELTA / 8)
    assert report.disconnected_registration_bounds == (DELTA, DELTA)
    lower = 7 * DELTA / 8 - 2 * NOISE
    assert report.recorded_contrast_bounds == (lower, 9 * DELTA / 8 + 2 * NOISE)
    assert report.contrast_margin == lower - DELTA / 2 > 0
    assert (
        report.response_certified
        and report.certified
        and not report.unresolved_conditions
    )
    record_property(
        "returned_physical_norm_squared_over_delta_squared", str(returned / DELTA**2)
    )
    record_property("uniform_recovery_exponent_lower_bound", str(rate * 2 * 10**10))
    record_property("recorded_contrast_lower_bound", str(lower))


def test_uniform_members_and_box_extremes_use_same_uniform_capture():
    reports = tuple(
        _assess(contact_conductances=(weight,) * 9)
        for weight in (Q(3, 4), Q(1), Q(5, 4))
    )
    assert all(row.certified and row.carrier_compatible for row in reports)
    assert len({row.returned_joint_norm_squared_upper_bound for row in reports}) == 1
    assert all(row.contact_conductance_variance == 0 for row in reports)
    assert all(
        row.returned_weighted_correlation_squared_upper_bound == 0 for row in reports
    )
    assert all(row.returned_weighted_correlation_upper_bound == 0 for row in reports)
    assert all(
        row.returned_relative_phase_row_defect_upper_bound == 0 for row in reports
    )
    assert all(not row.actual_source_aliases_certified for row in reports)


def test_exact_aggregate_rows_expose_form_weight_correlation(mp, report):
    forms = tuple(Q((i % 9) - 4, 1000) * (1 if i < 9 else -1) for i in range(18))
    errors = tuple(Q((i % 9) - 4, 100000) for i in range(18))
    x, theta = _state(mp, forms, errors)
    dx, dt, *_ = _rows(mp, x, theta)
    p, _, _ = _geometry()
    differences = tuple(x[9 + p[j]] - x[j] for j in range(9))
    gaps = tuple(
        theta[9 + p[j]] - theta[j] - 2 * mp.pi * ((2 * (p[j] - 4) - (j - 4)) // 9)
        for j in range(9)
    )
    u, jmean = sum(differences) / 9, sum(gaps) / 9
    meanw = sum(_mp(mp, w) for w in WEIGHTS) / 9
    chi = sum((_mp(mp, w) - meanw) * (d - u) for w, d in zip(WEIGHTS, differences)) / 9
    gamma = 1 / (1023 * mp.pi)
    du, dj = sum(dx[9:]) / 9 - sum(dx[:9]) / 9, sum(dt[9:]) / 9 - sum(dt[:9]) / 9
    assert abs(
        du
        + 2 * (meanw * u + chi) / 3
        + 2 * gamma * sum(mp.sin(gap) for gap in gaps) / 27
    ) < mp.mpf("1e-190")
    assert abs(dj - 2 * gamma * (meanw * u + chi) / 3) < mp.mpf("1e-190")
    variance = _mp(mp, report.contact_conductance_variance)
    assert chi**2 <= 2 * variance * sum(value**2 for value in x) / 9
    _, target = _state(mp, gap=Q(0))
    y2 = sum((a - b) ** 2 for a, b in zip(theta, target))
    form_defect = du + 2 * meanw * u / 3 + 2 * gamma * mp.sin(jmean) / 3
    phase_defect = dj - 2 * gamma * meanw * u / 3
    assert abs(form_defect) <= 2 * abs(chi) / 3 + 2 * gamma * y2 / 27
    assert abs(phase_defect - 2 * gamma * chi / 3) < mp.mpf("1e-190")
    assert chi != 0 and abs(phase_defect) > mp.mpf("1e-10")


def test_subgrid_nonclosure_pair_has_equal_means_energy_and_norms(mp, record_property):
    p, _, _ = _geometry()
    amplitude = ETA / 4
    plus = [Q(0)] * 18
    plus[9 + p[0]], plus[9 + p[1]] = amplitude, -amplitude
    minus = tuple(-value for value in plus)
    xp, theta = _state(mp, tuple(plus))
    xm, same_theta = _state(mp, minus)
    a, b = _rows(mp, xp, theta), _rows(mp, xm, same_theta)
    assert theta == same_theta and sum(plus[:9]) == sum(plus[9:]) == 0
    assert sum(value**2 for value in plus) == sum(value**2 for value in minus) < ETA**2
    assert abs(a[-2] - b[-2]) < mp.mpf("1e-190")
    dup = sum(a[0][9:]) / 9 - sum(a[0][:9]) / 9
    dum = sum(b[0][9:]) / 9 - sum(b[0][:9]) / 9
    djp = sum(a[1][9:]) / 9 - sum(a[1][:9]) / 9
    djm = sum(b[1][9:]) / 9 - sum(b[1][:9]) / 9
    form_difference = -4 * amplitude * (WEIGHTS[0] - WEIGHTS[1]) / 27
    phase_difference = -_mp(mp, form_difference) / (1023 * mp.pi)
    assert abs(dup - dum - _mp(mp, form_difference)) < mp.mpf("1e-185")
    assert abs(djp - djm - phase_difference) < mp.mpf("1e-185")
    assert dup - dum > 0 and djp - djm < 0
    uniform_a, uniform_b = _rows(mp, xp, theta, (Q(1),) * 9), _rows(
        mp, xm, theta, (Q(1),) * 9
    )
    for row in (0, 1):
        assert abs(
            sum(uniform_a[row][9:])
            - sum(uniform_a[row][:9])
            - sum(uniform_b[row][9:])
            + sum(uniform_b[row][:9])
        ) < mp.mpf("1e-185")
    record_property("static_witness_form_rate_difference", str(form_difference))
    record_property("explicit_static_witness_is_claimed_reached", "false")


def test_returned_defects_retain_positive_subgrid_spread_and_quadratic_phase(report):
    returned = report.returned_joint_norm_squared_upper_bound
    square = Q(2, 9) * report.contact_conductance_variance * returned
    assert report.returned_weighted_correlation_squared_upper_bound == square
    assert report.returned_weighted_correlation_upper_bound**2 >= square > 0
    upper = report.returned_weighted_correlation_upper_bound
    assert (
        report.returned_relative_form_row_defect_upper_bound
        == Q(2, 3) * upper + Q(2, 27) * Q(1, 3000) * returned
    )
    assert (
        report.returned_relative_phase_row_defect_upper_bound
        == Q(2, 3) * Q(1, 3000) * upper
    )
    tiny = Q(1, 10**100)
    small = _assess(contact_conductances=(Q(1) + tiny,) + (Q(1),) * 8)
    assert small.contact_conductance_variance > 0
    assert small.returned_weighted_correlation_upper_bound > 0
    assert small.returned_relative_phase_row_defect_upper_bound > 0
    assert not small.carrier_compatible


def test_only_precontact_source_and_waiting_are_consumed(monkeypatch):
    original = owner._assess_sine_mixed_capture
    calls = []

    def rejected_unit_return(**kwargs):
        calls.append(kwargs)
        return replace(
            original(**kwargs),
            joint_identity_certified=False,
            response_certified=False,
            certified=False,
            returned_joint_norm_squared_upper_bound=Q(10**100),
            status="unavailable",
        )

    monkeypatch.setattr(owner, "_assess_sine_mixed_capture", rejected_unit_return)
    actual = _assess()
    assert len(calls) == 1
    assert (
        actual.source_certified
        and actual.isolated_waiting_certified
        and actual.certified
    )
    assert not actual.precontact_reference.certified
    assert actual.returned_joint_norm_squared_upper_bound < DELTA**2 / 128


def test_missing_actual_source_or_waiting_and_zero_error_openness_do_not_pass(
    monkeypatch,
):
    original = owner._assess_sine_mixed_capture
    for field in ("source_certified", "isolated_waiting_certified"):
        with monkeypatch.context() as patch:
            patch.setattr(
                owner,
                "_assess_sine_mixed_capture",
                lambda **kwargs: replace(original(**kwargs), **{field: False}),
            )
            result = _assess()
            assert not result.certified and result.status == "unavailable"
    for changes in (dict(form_error_bound=0), dict(phase_error_bound=0)):
        result = _assess(**changes)
        assert result.certified
        assert not result.actual_source_aliases_certified


def test_fresh_reference_geometry_and_recovery_class_association_are_checked(
    monkeypatch,
):
    original = owner._assess_sine_mixed_capture

    def corrupt_geometry(**kwargs):
        reference = original(**kwargs)
        return replace(
            reference,
            geometry=replace(reference.geometry, matching_permutation=tuple(range(9))),
        )

    def corrupt_class(**kwargs):
        reference = original(**kwargs)
        return replace(
            reference,
            recoveries=(
                replace(reference.recoveries[0], target_winding=2),
                reference.recoveries[1],
            ),
        )

    for replacement in (corrupt_geometry, corrupt_class):
        with monkeypatch.context() as patch:
            patch.setattr(owner, "_assess_sine_mixed_capture", replacement)
            result = _assess()
            assert not result.certified and not result.actual_source_aliases_certified
            assert result.status == "unavailable" and result.unresolved_conditions


def test_nonnegative_outside_family_is_unavailable_and_field_rejects():
    for weights in (
        (Q(0),) * 9,
        (Q(3, 4) - Q(1, 10**100),) + (Q(1),) * 8,
        (Q(5, 4) + Q(1, 10**100),) + (Q(1),) * 8,
    ):
        report = _assess(contact_conductances=weights)
        assert report.contact_conductances == weights
        assert not report.conductances_within_protocol and not report.certified
        assert report.status == "unavailable" and report.unresolved_conditions
        with pytest.raises(ValueError):
            owner._mixed_conductance_field(contact_conductances=weights)
    for field, value in (
        ("form_error_bound", ERROR + Q(1, 10**100)),
        ("phase_error_bound", ERROR + Q(1, 10**100)),
        ("observation_error_bound", NOISE + Q(1, 10**100)),
    ):
        assert not _assess(**{field: value}).certified


def test_all_raw_admission_precedes_source_geometry_metric_or_field(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid primitive reached numerical or source work")

    for name in (
        "_assess_sine_mixed_capture",
        "_derive_mixed_constitutive_pair",
        "_held_sine_neighbor_field",
        "_sine_common_quadratic_coefficients",
    ):
        monkeypatch.setattr(owner, name, forbidden)
    invalid_weights = (
        (),
        WEIGHTS[:-1],
        WEIGHTS + (Q(1),),
        repeat(Q(1)),
        {1: 2},
        "123456789",
        (True,) + (Q(1),) * 8,
        (np.bool_(False),) + (Q(1),) * 8,
        (Q(-1),) + (Q(1),) * 8,
        (float("nan"),) + (Q(1),) * 8,
        (float("inf"),) + (Q(1),) * 8,
        (1j,) + (Q(1),) * 8,
    )
    for value in invalid_weights:
        with pytest.raises((TypeError, ValueError, OverflowError)):
            _assess(contact_conductances=value)
        with pytest.raises((TypeError, ValueError, OverflowError)):
            owner._mixed_conductance_field(contact_conductances=value)
    for field in ("form_error_bound", "phase_error_bound", "observation_error_bound"):
        for value in (True, np.bool_(False), Q(-1), float("nan"), float("inf"), None):
            with pytest.raises((TypeError, ValueError, OverflowError)):
                _assess(**{field: value})


def test_one_shot_and_fixed_width_inputs_normalize_before_association(report):
    result = _assess(
        contact_conductances=(
            Q(np.int64(value.numerator), np.int64(value.denominator))
            for value in WEIGHTS
        ),
        form_error_bound=Q(np.int64(1), np.int64(10**10)),
        phase_error_bound=Q(np.int64(1), np.int64(10**10)),
        observation_error_bound=Q(np.int64(1), np.int64(10**6)),
    )
    assert result == report
    swapped = _assess(contact_conductances=tuple(reversed(WEIGHTS)))
    assert swapped.mean_contact_conductance == report.mean_contact_conductance
    assert swapped.contact_conductance_variance == report.contact_conductance_variance
    assert swapped.weighted_edges != report.weighted_edges
    assert (
        swapped.returned_joint_norm_squared_upper_bound
        == report.returned_joint_norm_squared_upper_bound
    )


def test_frozen_report_and_field_accept_no_cached_certificate_or_protocol_override(
    report,
):
    with pytest.raises(FrozenInstanceError):
        report.certified = False
    for extra in (
        dict(precontact_reference=report.precontact_reference),
        dict(recovery_duration=Q(2 * 10**10)),
        dict(phase_scale=SCALE),
        dict(geometry=report.precontact_reference.geometry),
    ):
        with pytest.raises(TypeError):
            _assess(**extra)
    field = owner._mixed_conductance_field(contact_conductances=WEIGHTS)
    for state in ((0,) * 35, (0,) * 37):
        with pytest.raises(ValueError):
            field(state)
