"""Independent complete-product checks for the frozen outer-winding gate."""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q

import mpmath
import numpy as np
import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I
from tnfr.physics import _organized_outer_winding as owner


_ERROR = Q(1, 10**10)
_Q = Q(1, 10**70)
_INTERNAL = tuple(
    (9 * c + j, 9 * c + (j + 1) % 9, Q(1)) for c in range(5) for j in range(9)
)
_OUTER = tuple(
    (9 * c + j, 9 * ((c + 1) % 5) + j, Q(1)) for c in range(5) for j in range(9)
)
_EDGES = _INTERNAL + _OUTER
_SHIFTS = tuple(Q(4092 * (c - 2)) for c in range(5))


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


def _contains(mp, interval, value):
    assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)


def _assess(**changes):
    return owner._assess_organized_outer_winding(
        **(
            dict(form_error_bounds=(_ERROR,) * 5, phase_error_bounds=(_ERROR,) * 5)
            | changes
        )
    )


@pytest.fixture(scope="module")
def report():
    return _assess()


def _fields(mp, form, phase, edges=_EDGES):
    """Sum the actual edge law, independently of every report field."""
    gradient, current = [mp.mpf(0)] * len(form), [mp.mpf(0)] * len(form)
    storage = mp.mpf(0)
    for a, b, weight in edges:
        weight = _mp(mp, weight)
        difference, angle = form[a] - form[b], phase[b] - phase[a]
        gradient[a] += weight * difference
        gradient[b] -= weight * difference
        current[a] += weight * mp.sin(angle)
        current[b] -= weight * mp.sin(angle)
        storage += weight * (difference**2 / 2 + 1 - mp.cos(angle))
    return gradient, current, storage


def _phase(mp, offsets):
    return tuple(
        2 * mp.pi * (i % 9 - 4) / 9 + _mp(mp, v) for i, v in enumerate(offsets)
    )


def _cycle_gaps(mp, phase, *, outer, index):
    nodes = (
        tuple(9 * c + index for c in range(5))
        if outer
        else tuple(9 * index + j for j in range(9))
    )
    return tuple(
        (phase[nodes[(k + 1) % len(nodes)]] - phase[a] + mp.pi) % (2 * mp.pi) - mp.pi
        for k, a in enumerate(nodes)
    )


def test_support_is_the_complete_cartesian_product_with_uniform_clock(report):
    geometry = report.geometry
    assert set(geometry.weighted_edges) == set(_EDGES)
    assert len(geometry.weighted_edges) == len(set(_EDGES)) == 90
    assert geometry.node_order == tuple(range(45))
    assert geometry.held_capacities == (Q(1),) * 45
    assert geometry.weighted_degrees == (Q(4),) * 45
    assert geometry.clock_factor == Q(1, 4)
    assert report.form_error_bounds == report.phase_error_bounds == (_ERROR,) * 5
    assert len(report.child_sources) == 5
    assert len({id(child) for child in report.child_sources}) == 5
    for child in report.child_sources:
        assert child.form_error_bound == child.phase_error_bound == _ERROR
        assert child.deformation == 0 and child.recovery_certified
        assert child.endpoint_form_norm_squared_upper_bound < _Q**2
        assert child.endpoint_phase_norm_squared_upper_bound < _Q**2
        assert child.formation_time == 100
        assert child.relaxation_duration == 10**13
    assert report.protocol_revision == "82825de5"
    assert report.relative_preparation_dimension == 80
    assert report.source_formation_time == 200
    assert report.source_relaxation_duration == 2 * 10**13
    assert report.source_handoff_time == 2 * (100 + 10**13)
    assert report.elapsed_scaled_time == 200
    assert report.elapsed_original_time == 200 * Q(1024, 1023)
    assert report.primitive_scaled_time == 100
    assert report.initial_child_form_origins == _SHIFTS
    assert report.initial_child_phase_origins == (Q(0),) * 5


def test_both_carrier_rows_transform_to_the_primitive_c5_clock(mp):
    forms = tuple(Q(v, 7) for v in (-4, 7, 1, -9, 5))
    origins = tuple(Q(v, 11) for v in (3, -2, 1, 4, -6))
    x = tuple(_mp(mp, forms[c]) for c in range(5) for _ in range(9))
    phase = tuple(
        2 * mp.pi * (j - 4) / 9 + _mp(mp, origins[c])
        for c in range(5)
        for j in range(9)
    )
    gradient, current, storage = _fields(mp, x, phase)
    primitive_edges = tuple((c, (c + 1) % 5, Q(1)) for c in range(5))
    primitive_gradient, primitive_current, primitive_storage = _fields(
        mp,
        tuple(_mp(mp, v) for v in forms),
        tuple(_mp(mp, v) for v in origins),
        primitive_edges,
    )
    gamma = 1 / (1023 * mp.pi)
    observation = owner._observe_organized_outer_winding(
        form=tuple(v for v in forms for _ in range(9)),
        phase_offsets=tuple(v for v in origins for _ in range(9)),
    )
    for i in range(45):
        c = i // 9
        assert abs(gradient[i] - primitive_gradient[c]) < mp.mpf("1e-200")
        assert abs(current[i] - primitive_current[c]) < mp.mpf("1e-200")
        dx = (-gradient[i] + gamma * current[i]) / 4
        dt = gamma * gradient[i] / 4
        # sigma=tau/2 transforms both rows: d/dsigma = 2*d/dtau.
        assert abs(
            2 * dx - (-primitive_gradient[c] + gamma * primitive_current[c]) / 2
        ) < mp.mpf("1e-200")
        assert abs(2 * dt - gamma * primitive_gradient[c] / 2) < mp.mpf("1e-200")
        _contains(mp, observation.form_tau_rate_bounds[i], dx)
        _contains(mp, observation.phase_tau_rate_bounds[i], dt)
    expected = 45 * (1 - mp.cos(2 * mp.pi / 9)) + 9 * primitive_storage
    assert abs(storage - expected) < mp.mpf("1e-200")
    _contains(mp, observation.storage_bounds, expected)


def test_complete_off_carrier_storage_derivative_and_conserved_means(mp):
    forms = tuple(Q((7 * i) % 19 - 9, 64) for i in range(45))
    offsets = tuple(Q((11 * i) % 23 - 11, 256) for i in range(45))
    x, phase = tuple(_mp(mp, value) for value in forms), _phase(mp, offsets)
    gradient, current, storage = _fields(mp, x, phase)
    gamma = 1 / (1023 * mp.pi)
    dx = tuple((-a + gamma * b) / 4 for a, b in zip(gradient, current))
    dt = tuple(gamma * a / 4 for a in gradient)
    loss = sum(value**2 for value in gradient) / 4
    assert abs(sum(dx)) < mp.mpf("1e-200")
    assert abs(sum(dt)) < mp.mpf("1e-200")
    assert abs(sum(dx[:9])) > mp.mpf(".01")
    assert abs(sum(dt[:9])) > mp.mpf("1e-6")
    direct = sum(a * u - b * v for a, b, u, v in zip(gradient, current, dx, dt))
    assert abs(direct + loss) < mp.mpf("1e-200")
    step = mp.mpf("1e-80")
    advanced_x = tuple(a + mp.j * step * b for a, b in zip(x, dx))
    advanced_t = tuple(a + mp.j * step * b for a, b in zip(phase, dt))
    numerical_derivative = mp.im(_fields(mp, advanced_x, advanced_t)[2]) / step
    assert abs(numerical_derivative + loss) < mp.mpf("1e-150")
    observation = owner._observe_organized_outer_winding(
        form=forms, phase_offsets=offsets
    )
    _contains(mp, observation.storage_bounds, storage)
    assert abs(_mp(mp, observation.storage_tau_rate) + loss) < mp.mpf("1e-200")
    assert observation.balance_residual_bounds.contains(Q(0))
    for i in range(45):
        _contains(mp, observation.form_tau_rate_bounds[i], dx[i])
        _contains(mp, observation.phase_tau_rate_bounds[i], dt[i])
        _contains(
            mp, observation.form_original_rate_bounds[i], mp.mpf(1023) * dx[i] / 1024
        )
        _contains(
            mp, observation.phase_original_rate_bounds[i], mp.mpf(1023) * dt[i] / 1024
        )


def test_equal_child_means_discard_actual_phase_currents(mp):
    # These generic fine states test the mean-closure obstruction. They are
    # not claimed to be reached by the small recovered-source family.
    form = (Q(0),) * 45
    first = (Q(0),) * 45
    second = (Q(1, 2),) + (-Q(1, 16),) * 8 + (Q(0),) * 36
    assert all(sum(second[9 * c : 9 * c + 9], Q(0)) == 0 for c in range(5))
    nominal = owner._observe_organized_outer_winding(form=form, phase_offsets=first)
    changed = owner._observe_organized_outer_winding(form=form, phase_offsets=second)
    _, first_current, _ = _fields(mp, (mp.mpf(0),) * 45, _phase(mp, first))
    _, second_current, _ = _fields(mp, (mp.mpf(0),) * 45, _phase(mp, second))
    assert abs(sum(first_current[:9])) < mp.mpf("1e-200")
    assert abs(sum(second_current[:9])) > mp.mpf(".03")
    nominal_mean_rate = sum((value for value in nominal.form_tau_rate_bounds[:9])) / 9
    changed_mean_rate = sum((value for value in changed.form_tau_rate_bounds[:9])) / 9
    assert nominal_mean_rate.hi < changed_mean_rate.lo


def test_target_is_stationary_and_full_fine_hessian_has_only_global_null(mp, report):
    phase = tuple(
        2 * mp.pi * ((i % 9 - 4) / mp.mpf(9) + (i // 9 - 2) / mp.mpf(5))
        for i in range(45)
    )
    gradient, current, energy = _fields(mp, (mp.mpf(0),) * 45, phase)
    assert not any(gradient)
    assert max(abs(value) for value in current) < mp.mpf("1e-200")
    assert abs(sum(phase)) < mp.mpf("1e-200")
    expected = 45 * (1 - mp.cos(2 * mp.pi / 9)) + 45 * (1 - mp.cos(2 * mp.pi / 5))
    assert abs(energy - expected) < mp.mpf("1e-200")
    _contains(mp, report.target_storage_bounds, expected)
    hessian = np.zeros((45, 45))
    for a, b, _ in _EDGES:
        coefficient = float(mp.cos(phase[b] - phase[a]))
        hessian[a, a] += coefficient
        hessian[b, b] += coefficient
        hessian[a, b] -= coefficient
        hessian[b, a] -= coefficient
    eigenvalues = np.linalg.eigvalsh(hessian)
    assert abs(eigenvalues[0]) < 1e-12
    assert eigenvalues[1] > 0.3
    for outer, count in ((False, 5), (True, 9)):
        for index in range(count):
            gaps = _cycle_gaps(mp, phase, outer=outer, index=index)
            assert all(0 < value < mp.pi / 2 for value in gaps)
            assert abs(sum(gaps) - 2 * mp.pi) < mp.mpf("1e-200")


def test_independent_cycle_face_bounds_retain_both_product_multiplicities(mp, report):
    alpha, beta = 2 * mp.pi / 9, 2 * mp.pi / 5
    internal = 9 * (1 - mp.cos(alpha))
    outer = 5 * (1 - mp.cos(beta))
    positive_internal = 1 + 8 * (1 - mp.cos(3 * mp.pi / 16))
    negative_internal = 1 + 8 * (1 - mp.cos(5 * mp.pi / 16))
    positive_outer = 1 + 4 * (1 - mp.cos(3 * mp.pi / 8))
    _contains(mp, report.internal_boundary_excess_bounds, positive_internal - internal)
    _contains(mp, report.outer_boundary_excess_bounds, positive_outer - outer)
    assert (
        0
        < positive_outer - outer
        < positive_internal - internal
        < negative_internal - internal
    )
    bound = _mp(mp, report.boundary_energy_lower_bound)
    assert 0 < bound <= positive_outer - outer
    assert 5 * internal + 9 * outer + bound <= 5 * internal + 8 * outer + positive_outer


def test_all_180_signed_faces_respect_full_cycle_compatibility(mp, report):
    # Coherent product witnesses preserve all overlapping edge/cycle
    # correlations. Simultaneous faces are allowed: no sharp-product
    # attainment of the separate-cycle lower bound is asserted.
    alpha, beta = 2 * mp.pi / 9, 2 * mp.pi / 5
    baseline = 45 * (1 - mp.cos(alpha)) + 45 * (1 - mp.cos(beta))
    bound = _mp(mp, report.boundary_energy_lower_bound)
    internal_faces, outer_faces, impossible_outer_faces = set(), set(), set()
    for size, outer, cycle_count in ((9, False, 5), (5, True, 9)):
        for seam in range(size):
            for sign in (-1, 1):
                if outer and sign < 0:
                    # Even the largest four remaining acute gaps cannot
                    # close a winding-one C5 with a negative boundary edge.
                    assert -mp.pi / 2 + (size - 1) * mp.pi / 2 < 2 * mp.pi
                    impossible_outer_faces.update(
                        (index, seam, sign) for index in range(cycle_count)
                    )
                    continue
                face = sign * mp.pi / 2
                remainder = (2 * mp.pi - face) / (size - 1)
                gaps = [remainder] * size
                gaps[seam] = face
                lifts = [mp.mpf(0)]
                for gap in gaps[:-1]:
                    lifts.append(lifts[-1] + gap)
                lifts = [value - sum(lifts) / size for value in lifts]
                phase = tuple(
                    (alpha * (j - 4) + lifts[c] if outer else lifts[j] + beta * (c - 2))
                    for c in range(5)
                    for j in range(9)
                )
                for check_outer, count in ((False, 5), (True, 9)):
                    for index in range(count):
                        observed = _cycle_gaps(
                            mp, phase, outer=check_outer, index=index
                        )
                        assert all(
                            abs(value) <= mp.pi / 2 + mp.mpf("1e-200")
                            for value in observed
                        )
                        assert abs(sum(observed) - 2 * mp.pi) < mp.mpf("1e-200")
                excess = _fields(mp, (mp.mpf(0),) * 45, phase)[2] - baseline
                assert excess >= bound
                faces = outer_faces if outer else internal_faces
                faces.update((index, seam, sign) for index in range(cycle_count))
    assert len(internal_faces) == 90
    assert len(outer_faces) == len(impossible_outer_faces) == 45
    assert len(internal_faces) + len(outer_faces) + len(impossible_outer_faces) == 180
    assert report.signed_internal_boundary_faces == len(internal_faces)
    assert report.signed_outer_boundary_faces == len(outer_faces) + len(
        impossible_outer_faces
    )
    assert report.infeasible_outer_boundary_faces == len(impossible_outer_faces)
    assert (
        len(report.signed_face_feasible)
        == len(report.signed_face_excess_lower_bounds)
        == 180
    )
    assert all(report.signed_face_feasible[:90])
    assert report.signed_face_feasible[90:] == (False, True) * 45
    for face, possible in zip(
        report.signed_face_excess_lower_bounds, report.signed_face_feasible
    ):
        assert (face is not None) == possible
        if possible:
            assert face >= report.boundary_energy_lower_bound


def test_actual_initial_residuals_keep_zero_mean_correlations_and_positive_work(
    mp, report
):
    # An asymmetric member of the reached-state envelope, not an asserted
    # individual recovered trajectory. All five independent child residuals
    # have their actual initial zero means; neither row is reset to a carrier.
    form_residual = tuple(
        _Q * Q((c + 1) * (j - 4), 100) for c in range(5) for j in range(9)
    )
    phase_residual = tuple(
        _Q * Q((5 - c) * (3 * (j - 4) ** 2 - 20), 1200)
        for c in range(5)
        for j in range(9)
    )
    for values in (form_residual, phase_residual):
        for c in range(5):
            part = values[9 * c : 9 * c + 9]
            assert sum(part, Q(0)) == 0
            assert 0 < sum((value**2 for value in part), Q(0)) < _Q**2
    forms = tuple(_SHIFTS[i // 9] + value for i, value in enumerate(form_residual))
    x, phase = tuple(_mp(mp, value) for value in forms), _phase(mp, phase_residual)
    nominal = 9 * sum(
        ((_SHIFTS[(c + 1) % 5] - _SHIFTS[c]) ** 2 / 2 for c in range(5)), Q(0)
    )
    linear = sum(
        (
            (_SHIFTS[b // 9] - _SHIFTS[a // 9]) * (form_residual[b] - form_residual[a])
            for a, b, _ in _OUTER
        ),
        Q(0),
    )
    assert linear == 0
    work = _fields(mp, x, phase, _OUTER)[2]
    internal = _fields(mp, x, phase, _INTERNAL)[2]
    baseline = 45 * (1 - mp.cos(2 * mp.pi / 9))
    assert report.attachment_nominal_work == nominal == Q(1507001760)
    assert (
        0
        < work - _mp(mp, nominal)
        < _mp(mp, report.attachment_residual_work_upper_bound)
    )
    assert report.attachment_residual_work_upper_bound > 0
    assert report.attachment_residual_work_upper_bound < Q(1, 10**130)
    assert work < _mp(mp, report.attachment_work_upper_bound) < 2 * 10**9
    assert 0 < internal - baseline < 20 * _mp(mp, _Q) ** 2
    assert work + internal < _mp(mp, report.initial_storage_upper_bound) < 2 * 10**9
    for c in range(5):
        gaps = _cycle_gaps(mp, phase, outer=False, index=c)
        assert all(0 < value < mp.pi / 2 for value in gaps)
        assert abs(sum(gaps) - 2 * mp.pi) < mp.mpf("1e-200")
    for j in range(9):
        gaps = _cycle_gaps(mp, phase, outer=True, index=j)
        assert max(abs(value) for value in gaps) < 2 * _mp(mp, _Q)
        assert abs(sum(gaps)) < mp.mpf("1e-200")


def test_initial_child_correlations_do_not_discard_any_relative_direction():
    # Two rows times five children times eight free zero-mean coordinates.
    # Each basis direction has strict norm below q, independent of the
    # particular recovery enclosure or a projected mean representation.
    basis = []
    for channel in range(2):
        for c in range(5):
            for j in range(8):
                direction = [Q(0)] * 90
                direction[45 * channel + 9 * c + j] = _Q / 2
                direction[45 * channel + 9 * c + 8] = -_Q / 2
                assert sum(direction, Q(0)) == 0
                assert sum((value**2 for value in direction), Q(0)) < _Q**2
                basis.append(direction)
    integer_basis = np.array([[int(2 * value / _Q) for value in row] for row in basis])
    assert np.linalg.matrix_rank(integer_basis) == 80


def test_full_sources_enter_and_capture_with_strict_complete_storage_margin(report):
    assert report.source_certified
    assert report.initial_outer_zero_winding_certified
    assert report.child_winding_retained_during_transit
    assert report.primitive_entry.admitted
    assert report.primitive_entry.scaled_time == 100
    assert report.primitive_entry.source.epi == _SHIFTS
    assert report.primitive_entry.source.phase == (Q(0),) * 5
    assert report.endpoint_nominal_excess_upper_bound > 0
    assert report.endpoint_residual_excess_upper_bound > 0
    assert (
        report.endpoint_excess_upper_bound >= report.endpoint_nominal_excess_upper_bound
    )
    assert (
        report.capture_margin
        == report.boundary_energy_lower_bound - report.endpoint_excess_upper_bound
        > 0
    )
    assert report.outer_entry_certified and report.capture_certified
    assert report.converges_to_target
    assert not report.unresolved_conditions
    assert report.target_child_windings == (1,) * 5
    assert report.target_outer_windings == (1,) * 9
    assert len(report.endpoint_oriented_outer_phase_gap_bounds) == 45
    assert report.outer_winding_retained and report.child_winding_retained


def test_each_actual_outer_slot_cycle_is_enclosed_with_its_closing_branch(mp, report):
    gamma = 1 / (1023 * mp.pi)
    nominal = tuple(gamma * _mp(mp, value) for value in _SHIFTS)
    expected = tuple(
        nominal[(c + 1) % 5] - nominal[c] + (2 * mp.pi if c == 4 else 0)
        for c in range(5)
    )
    bounds = report.endpoint_oriented_outer_phase_gap_bounds
    for slot in range(9):
        part = bounds[5 * slot : 5 * slot + 5]
        for box, value in zip(part, expected):
            _contains(mp, box, value)
            assert 0 < _mp(mp, box.lo) < _mp(mp, box.hi) < mp.pi / 2
        # Intervals bound actual coherent gaps; their endpoints must not be
        # independently summed and asserted to be exact winding evidence.
        assert sum(_mp(mp, box.lo) for box in part) < 2 * mp.pi
        assert sum(_mp(mp, box.hi) for box in part) > 2 * mp.pi
    assert expected[-1] != expected[0]


def test_paired_control_keeps_actual_residuals_and_traps_zero_outer_winding(mp, report):
    control = report.control
    assert control.form_origins == control.phase_origins == (Q(0),) * 5
    assert control.initial_excess_upper_bound == 40 * _Q**2 > 0
    assert (
        control.capture_margin
        == control.boundary_energy_lower_bound - control.initial_excess_upper_bound
        > 0
    )
    assert control.source_certified and control.capture_certified
    assert control.child_winding_retained and control.outer_zero_winding_retained
    assert control.converges_to_target
    assert control.target_child_windings == (1,) * 5
    assert control.target_outer_windings == (0,) * 9
    q = _mp(mp, _Q)
    form = tuple(q * (i % 9 - 4) * (i // 9 + 1) / 100 for i in range(45))
    phase = tuple(2 * mp.pi * (i % 9 - 4) / 9 for i in range(45))
    gradient, current, storage = _fields(mp, form, phase)
    baseline = 45 * (1 - mp.cos(2 * mp.pi / 9))
    assert 0 < storage - baseline < _mp(mp, control.initial_excess_upper_bound)
    gamma = 1 / (1023 * mp.pi)
    assert max(abs((-a + gamma * b) / 4) for a, b in zip(gradient, current)) > q / 100
    assert max(abs(gamma * a / 4) for a in gradient) > gamma * q / 100
    for slot in range(9):
        assert abs(sum(_cycle_gaps(mp, phase, outer=True, index=slot))) < mp.mpf(
            "1e-200"
        )


def test_endpoint_phase_first_variation_cannot_be_dropped_or_childwise_recentered(
    mp, report
):
    # A point in the admitted carrier endpoint enclosure plus a globally
    # mean-zero fine residual. It is not asserted to be a reached trajectory
    # point. The post-contact error bound retains these child mean offsets.
    gamma, q = 1 / (1023 * mp.pi), _mp(mp, _Q)
    phase = tuple(
        2 * mp.pi * (j - 4) / 9 + gamma * _mp(mp, _SHIFTS[c])
        for c in range(5)
        for j in range(9)
    )
    residual = tuple(
        (-q / 2 if c == 0 else q / 2 if c == 4 else mp.mpf(0))
        for c in range(5)
        for _ in range(9)
    )
    assert abs(sum(residual)) < mp.mpf("1e-200")
    assert (
        sum(value**2 for value in residual)
        < _mp(mp, report.full_phase_error_upper_bound) ** 2
    )
    assert sum(residual[:9]) != 0
    zero = (mp.mpf(0),) * 45
    baseline = _fields(mp, zero, phase)[2]
    changed = _fields(mp, zero, tuple(a + b for a, b in zip(phase, residual)))[2]
    excess = changed - baseline
    assert excess > q / 10
    assert excess > 1000 * q**2
    assert excess < _mp(mp, report.endpoint_residual_excess_upper_bound)


def test_global_error_transfer_uses_full_product_spectrum_and_both_rows(mp, report):
    spectrum = sorted(
        4 - 2 * mp.cos(2 * mp.pi * j / 9) - 2 * mp.cos(2 * mp.pi * c / 5)
        for j in range(9)
        for c in range(5)
    )
    assert abs(spectrum[0]) < mp.mpf("1e-200")
    assert spectrum[1] > mp.mpf(1) / 3
    assert spectrum[-1] < 8
    # The heat semigroup and its complement are contractions. Integrating
    # the reciprocal phase row uses the complement, so no factor exp(T)
    # and no childwise mean conservation is needed after contact.
    for elapsed in (mp.mpf(0), mp.mpf(1), mp.mpf(200)):
        heat = tuple(mp.exp(-value * elapsed / 4) for value in spectrum)
        assert max(heat) <= 1 + mp.mpf("1e-200")
        assert max(1 - value for value in heat) <= 1
    gamma = 1 / (1023 * mp.pi)
    q = _mp(mp, _Q)
    initial = mp.sqrt(5) * q
    exact_phase_envelope = (initial + gamma * initial) * mp.exp(400 * gamma**2)
    actual_phase_bound = _mp(mp, report.full_phase_error_upper_bound)
    actual_form_bound = _mp(mp, report.full_form_error_upper_bound)
    assert exact_phase_envelope < actual_phase_bound < 7 * q
    integrated_form_envelope = initial + 8 * gamma * actual_phase_bound / spectrum[1]
    assert integrated_form_envelope < actual_form_bound < 4 * q
    assert report.full_phase_error_upper_bound < Q(1, 10**68)
    assert report.full_form_error_upper_bound < Q(1, 10**68)


def test_five_distinct_original_allowance_pairs_reach_five_fresh_recoveries(
    report, monkeypatch
):
    calls = []
    forms = tuple(_ERROR * Q(c + 1, 5) for c in range(5))
    phases = tuple(_ERROR * Q(5 - c, 5) for c in range(5))

    def recorded(**kwargs):
        calls.append(kwargs)
        # This is a wiring test; the returned fixture is not new scientific
        # evidence for the deliberately changed primitive input pairs.
        return report.child_sources[len(calls) - 1]

    monkeypatch.setattr(owner, "_assess_constitutive_recovery", recorded)
    result = _assess(form_error_bounds=forms, phase_error_bounds=phases)
    assert result.form_error_bounds == forms
    assert result.phase_error_bounds == phases
    assert calls == [
        dict(deformation=0, form_error_bound=form, phase_error_bound=phase)
        for form, phase in zip(forms, phases)
    ]


@pytest.mark.parametrize(
    "field,value",
    (
        ("recovery_certified", False),
        ("endpoint_form_norm_squared_upper_bound", None),
        ("endpoint_phase_norm_squared_upper_bound", None),
        ("endpoint_form_norm_squared_upper_bound", _Q**2),
        ("endpoint_phase_norm_squared_upper_bound", _Q**2),
    ),
)
def test_one_unavailable_or_nonstrict_child_suppresses_actual_outcomes(
    field, value, report, monkeypatch
):
    calls = []

    def failed_child(**kwargs):
        index = len(calls)
        calls.append(kwargs)
        child = report.child_sources[index]
        return replace(child, **{field: value}) if index == 3 else child

    monkeypatch.setattr(owner, "_assess_constitutive_recovery", failed_child)
    failed = _assess()
    assert len(calls) == 5
    assert not failed.source_certified
    assert not failed.initial_outer_zero_winding_certified
    assert not failed.child_winding_retained_during_transit
    assert not failed.outer_entry_certified
    assert not failed.capture_certified
    assert not failed.converges_to_target
    assert failed.status == "unavailable" and failed.unresolved_conditions
    assert failed.endpoint_excess_upper_bound == report.endpoint_excess_upper_bound
    assert not failed.control.source_certified
    assert not failed.control.capture_certified
    assert not failed.control.child_winding_retained
    assert not failed.control.outer_zero_winding_retained
    assert not failed.control.converges_to_target


def test_cached_primitive_values_cannot_replace_fresh_state_and_law_admission(
    report, monkeypatch
):
    source = replace(
        report.primitive_entry.source,
        storage=I(-100),
        form_storage=Q(-1),
        phase_storage=I(-1),
        form_gradient=(Q(0),) * 5,
    )
    monkeypatch.setattr(owner, "_primitive_outer_source", lambda: source)
    rebuilt = _assess()
    assert rebuilt.capture_certified
    assert (
        rebuilt.primitive_entry.capture.storage_bounds
        == report.primitive_entry.capture.storage_bounds
    )
    assert rebuilt.capture_margin == report.capture_margin


def test_different_primitive_law_cannot_inherit_its_predecessor_capture(
    report, monkeypatch
):
    source = replace(
        report.primitive_entry.source,
        reference_model=RelationalExchangeModel(1, phase_domain="regular"),
    )
    monkeypatch.setattr(owner, "_primitive_outer_source", lambda: source)
    failed = _assess()
    assert failed.source_certified
    assert not failed.primitive_law_and_clock_certified
    assert not failed.outer_entry_certified
    assert not failed.capture_certified
    assert not failed.converges_to_target
    assert (
        "primitive_C5_complete_law_or_clock_not_certified"
        in failed.unresolved_conditions
    )


def test_strict_attachment_budget_is_a_separate_actual_premise(report, monkeypatch):
    monkeypatch.setattr(owner, "_BUDGET", report.attachment_nominal_work)
    failed = _assess()
    assert failed.source_certified and failed.conditional_capture_certified
    assert not failed.attachment_budget_certified
    assert failed.attachment_work_margin < 0 and failed.initial_storage_margin < 0
    assert not failed.outer_entry_certified and not failed.capture_certified
    assert not failed.converges_to_target
    assert failed.control.capture_certified
    assert failed.status == "unavailable"


def test_failed_full_product_domain_never_inherits_the_carrier_flag(
    report, monkeypatch
):
    geometry = replace(report.geometry, outer_target_angle_bounds=I(0))
    monkeypatch.setattr(owner, "_derive_organized_outer_winding", lambda: geometry)
    failed = _assess()
    assert failed.source_certified and failed.primitive_entry.admitted
    assert not failed.fine_error_certified and not failed.barrier_certified
    assert not failed.capture_certified
    assert not failed.outer_winding_retained and not failed.child_winding_retained
    assert not failed.converges_to_target and not failed.control.capture_certified
    assert failed.status == "unavailable"


@pytest.mark.parametrize("field", ("form_error_bounds", "phase_error_bounds"))
@pytest.mark.parametrize(
    "bad", (True, "0", None, float("nan"), float("inf"), complex(0), -Q(1, 10**1000))
)
def test_all_ten_primitives_are_admitted_before_source_or_interval_work(
    field, bad, monkeypatch
):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid primitive reached source or geometry construction")

    monkeypatch.setattr(owner, "_assess_constitutive_recovery", forbidden)
    monkeypatch.setattr(owner, "_derive_organized_outer_winding", forbidden)
    with pytest.raises((TypeError, ValueError)):
        _assess(**{field: (_ERROR,) * 4 + (bad,)})


@pytest.mark.parametrize("field", ("form_error_bounds", "phase_error_bounds"))
@pytest.mark.parametrize("bad", ((), (_ERROR,) * 4, (_ERROR,) * 6, "00000", {_ERROR}))
def test_preparation_shapes_are_not_coerced_or_truncated(field, bad, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid shape reached source construction")

    monkeypatch.setattr(owner, "_assess_constitutive_recovery", forbidden)
    with pytest.raises((TypeError, ValueError)):
        _assess(**{field: bad})


@pytest.mark.parametrize("field", ("form_error_bounds", "phase_error_bounds"))
@pytest.mark.parametrize("index", range(5))
def test_larger_than_frozen_allowance_rejects_before_any_child_rebuild(
    field, index, monkeypatch
):
    def forbidden(*args, **kwargs):
        pytest.fail("outside-frozen-family allowance reached source construction")

    monkeypatch.setattr(owner, "_assess_constitutive_recovery", forbidden)
    monkeypatch.setattr(owner, "_derive_organized_outer_winding", forbidden)
    values = [_ERROR] * 5
    values[index] += Q(1, 10**1000)
    with pytest.raises(ValueError):
        _assess(**{field: tuple(values)})


@pytest.mark.parametrize(
    "extra",
    (
        "source",
        "barrier",
        "primitive_entry",
        "horizon",
        "clock",
        "form_shifts",
        "phase_origins",
        "trajectory",
    ),
)
def test_reports_and_retuned_protocol_values_cannot_replace_primitives(extra):
    with pytest.raises(TypeError):
        _assess(**{extra: 0})


def test_reports_are_immutable(report):
    with pytest.raises(FrozenInstanceError):
        report.capture_certified = False
