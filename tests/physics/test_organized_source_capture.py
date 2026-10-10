"""Independent full-state and disconnected controls for the five-C9 source."""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q

import mpmath
import numpy as np
import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.physics import _organized_source_capture as owner


_ERROR = Q(1, 10**10)
_CYCLES = tuple(
    (9 * child + j, 9 * child + (j + 1) % 9, Q(1))
    for child in range(5)
    for j in range(9)
)
_HOST_CONTACTS = tuple(
    (9 * a + j, 9 * b + j, Q(1))
    for a, b in ((0, 1), (2, 3), (0, 2), (1, 3))
    for j in range(9)
)
_PORT = tuple((j, 36 + j, Q(1)) for j in range(9))
_EDGES = _CYCLES + _HOST_CONTACTS + _PORT


@pytest.fixture(scope="module", autouse=True)
def no_response_or_frozen_execution():
    with forbid_sine_regeneration():
        yield


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 220
    return context


def _mp(mp, value):
    return mp.mpf(value.numerator) / value.denominator


def _contains(mp, bounds, value):
    assert _mp(mp, bounds.lo) <= value <= _mp(mp, bounds.hi)


def _assess(**changes):
    return owner._assess_organized_source_capture(
        **(
            dict(
                host_form_error_bounds=(_ERROR,) * 4,
                host_phase_error_bounds=(_ERROR,) * 4,
                donor_form_error_bound=_ERROR,
                donor_phase_error_bound=_ERROR,
            )
            | changes
        )
    )


@pytest.fixture(scope="module")
def report():
    return _assess()


def _fields(mp, form, phase, edges):
    gradient = [mp.mpf(0)] * 45
    current = [mp.mpf(0)] * 45
    energy = mp.mpf(0)
    for a, b, weight in edges:
        w = _mp(mp, weight)
        difference, angle = form[a] - form[b], phase[b] - phase[a]
        gradient[a] += w * difference
        gradient[b] -= w * difference
        current[a] += w * mp.sin(angle)
        current[b] -= w * mp.sin(angle)
        energy += w * (difference**2 / 2 + 1 - mp.cos(angle))
    return gradient, current, energy


def test_actual_source_and_uniform_clock_are_retained(report):
    source = report.source
    assert source.relative_preparation_dimension == 80
    assert source.host.acquisition_slow_time == 4
    assert source.host_form_error_bounds == (_ERROR,) * 4
    assert source.host_phase_error_bounds == (_ERROR,) * 4
    assert source.donor_form_error_bound == _ERROR
    assert source.donor_phase_error_bound == _ERROR
    assert source.donor_formation_tau == 200
    assert source.donor_recovery_duration_tau == 2 * 10**13
    assert source.donor_handoff_tau == 2 * (100 + 10**13)
    assert source.donor_phase_origin == Q(1, 8)
    geometry = report.barrier.geometry
    assert set(geometry.weighted_edges) == set(_EDGES)
    assert len(geometry.weighted_edges) == 90
    assert geometry.node_order == tuple(range(45))
    assert geometry.held_capacities == (Q(1),) * 45
    assert geometry.clock_factor == Q(1, 4)
    assert geometry.weighted_degrees == (5,) * 9 + (4,) * 27 + (3,) * 9
    assert source.attachment_work_lower_bound > 0
    assert source.attachment_work_upper_bound > source.attachment_work_lower_bound


def test_target_is_a_full_fine_stationary_state_with_the_conserved_mean(report, mp):
    barrier = report.barrier
    alpha, mean = 2 * mp.pi / 9, mp.mpf(1) / 40
    theta = tuple(alpha * (j % 9 - 4) + mean for j in range(45))
    x = (mp.mpf(0),) * 45
    gradient, current, energy = _fields(mp, x, theta, _EDGES)
    assert barrier.target_form == (Q(0),) * 45
    assert barrier.target_phase_offsets == (Q(1, 40),) * 45
    for bounds, value in zip(barrier.target_phase_lift_bounds, theta):
        _contains(mp, bounds, value)
    assert max(abs(value) for value in gradient) == 0
    assert max(abs(value) for value in current) < mp.mpf("1e-200")
    assert abs(sum(theta) / 45 - mean) < mp.mpf("1e-200")
    _contains(mp, barrier.target_storage_bounds, energy)
    assert abs(energy - 45 * (1 - mp.cos(alpha))) < mp.mpf("1e-200")
    assert report.source.conserved_phase_total / 5 == Q(1, 40)
    assert report.source.conserved_form_total == 0
    for start in range(0, 45, 9):
        angles = tuple(
            theta[start + (j + 1) % 9] - theta[start + j] + (2 * mp.pi if j == 8 else 0)
            for j in range(9)
        )
        assert all(0 < angle < mp.pi / 2 for angle in angles)
        assert abs(sum(angles) - 2 * mp.pi) < mp.mpf("1e-200")


def test_full_fine_hessian_has_only_the_global_phase_null_direction(report, mp):
    alpha = 2 * mp.pi / 9
    hessian = np.zeros((45, 45))
    for a, b, weight in _EDGES:
        coefficient = float(weight) * (float(mp.cos(alpha)) if a // 9 == b // 9 else 1)
        hessian[a, a] += coefficient
        hessian[b, b] += coefficient
        hessian[a, b] -= coefficient
        hessian[b, a] -= coefficient
    spectrum = np.linalg.eigvalsh(hessian)
    assert abs(spectrum[0]) < 1e-12
    assert spectrum[1] > 0.3
    assert report.barrier.target_certified


def test_cycle_face_minima_are_independently_attained_and_bound_both_signs(report, mp):
    barrier = report.barrier
    alpha = 2 * mp.pi / 9
    baseline = 9 * (1 - mp.cos(alpha))
    positive_angles = (mp.pi / 2,) + (3 * mp.pi / 16,) * 8
    negative_angles = (-mp.pi / 2,) + (5 * mp.pi / 16,) * 8
    positive = sum(1 - mp.cos(value) for value in positive_angles)
    negative = sum(1 - mp.cos(value) for value in negative_angles)
    for angles in (positive_angles, negative_angles):
        assert abs(sum(angles) - 2 * mp.pi) < mp.mpf("1e-200")
        assert all(-mp.pi / 2 <= angle <= mp.pi / 2 for angle in angles)
    _contains(mp, barrier.cycle_storage_minimum_bounds, baseline)
    _contains(mp, barrier.positive_cycle_face_storage_bounds, positive)
    _contains(mp, barrier.negative_cycle_face_storage_bounds, negative)
    _contains(mp, barrier.cycle_boundary_excess_bounds, positive - baseline)
    assert 0 < positive - baseline < 1 < negative - baseline
    # A zero-sum redistribution of the eight free angles increases storage;
    # these vectors are full cycle configurations, not fixture-bound literals.
    for sign_angles, minimum in (
        (positive_angles, positive),
        (negative_angles, negative),
    ):
        for amount in (mp.mpf(".01"), mp.mpf(".2")):
            changed = list(sign_angles)
            changed[1] += amount
            changed[2] -= amount
            assert all(-mp.pi / 2 <= angle <= mp.pi / 2 for angle in changed)
            assert sum(1 - mp.cos(value) for value in changed) > minimum
    assert barrier.matching_boundary_excess == 1
    assert 0 < _mp(mp, barrier.boundary_energy_lower_bound) <= positive - baseline
    _contains(mp, barrier.cell_edge_gap_bounds, mp.pi / 2)


def test_every_signed_internal_face_retains_all_cycle_sums_and_full_boundary(
    report, mp
):
    alpha = 2 * mp.pi / 9
    baseline = 45 * (1 - mp.cos(alpha))
    zero = (mp.mpf(0),) * 45
    covered = set()
    for child in range(5):
        for seam in range(9):
            for sign in (-1, 1):
                boundary_gap = sign * mp.pi / 2
                remainder = (2 * mp.pi - boundary_gap) / 8
                angles = [remainder] * 9
                angles[seam] = boundary_gap
                lifts = [mp.mpf(0)]
                for gap in angles[:-1]:
                    lifts.append(lifts[-1] + gap)
                residual = [value - alpha * j for j, value in enumerate(lifts)]
                residual_mean = sum(residual) / 9
                residual = [value - residual_mean for value in residual]
                theta = tuple(
                    alpha * (node % 9 - 4)
                    + mp.mpf(1) / 40
                    + (residual[node % 9] if node // 9 == child else 0)
                    for node in range(45)
                )
                # This whole fine state lies on the chosen boundary face;
                # its neighboring cycles and matching contacts stay admitted.
                for a, b, _ in _EDGES:
                    gap = theta[b] - theta[a]
                    if a // 9 == b // 9 and a % 9 == 8:
                        gap += 2 * mp.pi
                    assert abs(gap) <= mp.pi / 2 + mp.mpf("1e-200")
                excess = _fields(mp, zero, theta, _EDGES)[2] - baseline
                assert excess >= _mp(mp, report.barrier.boundary_energy_lower_bound)
                covered.add((child, seam, sign))
    assert len(covered) == report.barrier.signed_internal_boundary_faces == 90
    assert (
        2 * len(_HOST_CONTACTS + _PORT)
        == report.barrier.signed_matching_boundary_faces
        == 90
    )


def test_source_excess_is_rebuilt_at_entry_instead_of_initial_assembly(report):
    source = report.source
    host, gamma = source.host, source.geometry.gamma_bounds.hi
    rx, rt = host.full_form_error_upper_bound, host.full_phase_error_upper_bound
    gaps = (
        host.nominal_inner_gap_at_entry_bounds.hi,
        host.nominal_outer_gap_at_entry_bounds.hi,
    )
    scale = 1 + 2 * (source.geometry.gamma_bounds**2).hi
    nominal = 9 * sum(((gamma * scale * gap) ** 2 + gap**2 for gap in gaps), Q(0))
    full = 2 * (rx**2 + rt**2) + 9 * sum(
        ((gamma * scale * gap + 2 * rx) ** 2 + (gap + 2 * rt) ** 2 for gap in gaps),
        Q(0),
    )
    donor = 4 * source.donor_recovered_norm_bound**2
    assert report.source_host_nominal_excess_upper_bound == nominal
    assert report.source_host_excess_upper_bound == full
    assert report.source_host_residual_excess_upper_bound == full - nominal > 0
    assert report.source_donor_excess_upper_bound == donor > 0
    assert report.pre_attachment_excess_upper_bound == full + donor
    assert (
        report.initial_excess_upper_bound
        == full + donor + source.attachment_work_upper_bound
    )
    assert report.initial_excess_upper_bound < Q(1, 10)
    assert (
        report.capture_margin
        == report.barrier.boundary_energy_lower_bound
        - report.initial_excess_upper_bound
        > 0
    )
    assert (
        report.source_certified and report.source_in_cell and report.barrier_certified
    )
    assert (
        report.capture_certified
        and report.winding_retained
        and report.converges_to_target
    )
    assert not report.unresolved_conditions


def test_full_source_envelope_keeps_asymmetric_residuals_and_linear_work(report, mp):
    # This is a state in the admitted endpoint envelope, not a claim that this
    # chosen vector is attained by the original preparation. The inequalities
    # must hold for it as well as for the narrower actual reached family.
    source = report.source
    host = source.host
    alpha, gamma = 2 * mp.pi / 9, 1 / (1023 * mp.pi)
    inner = _mp(mp, host.nominal_inner_gap_at_entry_bounds.hi)
    outer = _mp(mp, host.nominal_outer_gap_at_entry_bounds.hi)
    rx, rt = _mp(mp, host.full_form_error_upper_bound), _mp(
        mp, host.full_phase_error_upper_bound
    )
    q = _mp(mp, source.donor_recovered_norm_bound)
    scale = 1 + 2 * gamma**2
    u, v = -gamma * scale * outer / 2, -gamma * scale * inner / 2
    nominal_x, nominal_t = [], []
    for child in range(4):
        g, a = divmod(child, 2)
        nominal_x.extend(((2 * g - 1) * u / 2 + (2 * a - 1) * v / 2,) * 9)
        nominal_t.extend(
            alpha * (j - 4) + (2 * g - 1) * outer / 2 + (2 * a - 1) * inner / 2
            for j in range(9)
        )
    nominal_x.extend((mp.mpf(0),) * 9)
    nominal_t.extend(alpha * (j - 4) + mp.mpf(1) / 8 for j in range(9))
    form, phase = list(nominal_x), list(nominal_t)
    form[0] += rx / 4
    form[9] -= rx / 4
    phase[0] -= rt / 4
    phase[9] += rt / 4
    for j in range(9):
        form[36 + j] += q * (j - 4) / 20
        phase[36 + j] += q * ((j - 4) ** 2 - mp.mpf(20) / 3) / 80
    for start, stop, bound_x, bound_t in ((0, 36, rx, rt), (36, 45, q, q)):
        delta_x = tuple(form[i] - nominal_x[i] for i in range(start, stop))
        delta_t = tuple(phase[i] - nominal_t[i] for i in range(start, stop))
        assert abs(sum(delta_x)) < mp.mpf("1e-200")
        assert abs(sum(delta_t)) < mp.mpf("1e-200")
        assert sum(value**2 for value in delta_x) <= bound_x**2
        assert sum(value**2 for value in delta_t) <= bound_t**2
    # The host residual has nonzero individual child means, which are no
    # longer conserved after host assembly; a childwise reset would erase it.
    assert sum(form[:9]) != sum(nominal_x[:9])
    assert sum(phase[:9]) != sum(nominal_t[:9])
    host_edges = _CYCLES[:36] + _HOST_CONTACTS
    nominal_host_energy = _fields(mp, nominal_x, nominal_t, host_edges)[2]
    actual_host_energy = _fields(mp, form, phase, host_edges)[2]
    linear_residual = actual_host_energy - nominal_host_energy
    assert linear_residual > rt / 1000
    assert linear_residual > 1000 * (rx**2 + rt**2)
    assert linear_residual < _mp(mp, report.source_host_residual_excess_upper_bound)
    host_excess = actual_host_energy - 36 * (1 - mp.cos(alpha))
    assert 0 < host_excess < _mp(mp, report.source_host_excess_upper_bound)
    donor_excess = _fields(mp, form, phase, _CYCLES[36:])[2] - 9 * (1 - mp.cos(alpha))
    assert 0 < donor_excess < _mp(mp, report.source_donor_excess_upper_bound)
    contact_work = _fields(mp, form, phase, _PORT)[2]
    assert (
        _mp(mp, source.attachment_work_lower_bound)
        < contact_work
        < _mp(mp, source.attachment_work_upper_bound)
    )
    total_excess = _fields(mp, form, phase, _EDGES)[2] - 45 * (1 - mp.cos(alpha))
    assert 0 < total_excess < _mp(mp, report.initial_excess_upper_bound)
    for a, b, _ in _EDGES:
        angle = phase[b] - phase[a]
        if a // 9 == b // 9 and a % 9 == 8:
            angle += 2 * mp.pi
        assert -mp.pi / 2 < angle < mp.pi / 2


def test_disconnected_control_keeps_component_means_and_excludes_joint_target(
    report, mp
):
    control = report.control
    assert set(control.weighted_edges) == set(_CYCLES + _HOST_CONTACTS)
    assert len(control.weighted_edges) == 81
    assert control.components == (tuple(range(36)), tuple(range(36, 45)))
    assert control.component_form_means == (Q(0), Q(0))
    assert control.component_phase_offset_means == (Q(0), Q(1, 8))
    assert control.common_target_phase_offset == Q(1, 40)
    assert control.target_component_mean_mismatches == (-Q(1, 40), Q(1, 10))
    assert control.clock_factor == Q(1, 4)
    assert control.joined_alignment_excluded
    assert report.initial_donor_host_phase_difference == Q(1, 8)
    assert report.asymptotic_donor_host_phase_difference == 0
    assert control.donor_host_phase_difference == Q(1, 8)
    # Test conservation on an asymmetric off-carrier state, not just on the
    # already stationary endpoint. Pairwise cancellation belongs to support.
    form = tuple(mp.mpf((7 * i) % 19 - 9) / 64 for i in range(45))
    phase = tuple(
        2 * mp.pi * (i % 9 - 4) / 9 + mp.mpf((11 * i) % 23 - 11) / 256
        for i in range(45)
    )
    gamma = 1 / (1023 * mp.pi)
    gradient, current, _ = _fields(mp, form, phase, _CYCLES + _HOST_CONTACTS)
    dx = tuple((-lap + gamma * sine) / 4 for lap, sine in zip(gradient, current))
    dt = tuple(gamma * lap / 4 for lap in gradient)
    for nodes in control.components:
        assert abs(sum(dx[i] for i in nodes)) < mp.mpf("1e-200")
        assert abs(sum(dt[i] for i in nodes)) < mp.mpf("1e-200")
    # The same two sums are generally exchanged through the joined port.
    joined_gradient, joined_current, _ = _fields(mp, form, phase, _EDGES)
    assert abs(sum(joined_gradient[:36])) > mp.mpf(".01")
    assert abs(sum(joined_current[:36])) > mp.mpf(".01")


def test_complete_fine_loss_includes_both_rows_and_actual_uniform_factor(mp):
    x = tuple(mp.mpf((7 * i) % 19 - 9) / 64 for i in range(45))
    phase = tuple(
        2 * mp.pi * (i % 9 - 4) / 9 + mp.mpf((11 * i) % 23 - 11) / 256
        for i in range(45)
    )
    gamma = 1 / (1023 * mp.pi)
    for edges in (_EDGES, _CYCLES + _HOST_CONTACTS):
        gradient, current, _ = _fields(mp, x, phase, edges)
        dx = tuple((-lap + gamma * sine) / 4 for lap, sine in zip(gradient, current))
        dt = tuple(gamma * lap / 4 for lap in gradient)
        loss = sum(lap**2 for lap in gradient) / 4
        direct = sum(
            lap * form_rate - sine * phase_rate
            for lap, sine, form_rate, phase_rate in zip(gradient, current, dx, dt)
        )
        assert abs(direct + loss) < mp.mpf("1e-200")
        # Independently differentiate storage, without using its gradient row.
        step = mp.mpf("1e-80")
        advanced_x = tuple(value + mp.j * step * rate for value, rate in zip(x, dx))
        advanced_t = tuple(value + mp.j * step * rate for value, rate in zip(phase, dt))
        derivative = mp.im(_fields(mp, advanced_x, advanced_t, edges)[2]) / step
        assert abs(derivative + loss) < mp.mpf("1e-150")
        assert loss > 0


@pytest.mark.parametrize(
    "field",
    (
        "host_form_error_bounds",
        "host_phase_error_bounds",
        "donor_form_error_bound",
        "donor_phase_error_bound",
    ),
)
@pytest.mark.parametrize(
    "bad", (True, "0", None, float("nan"), float("inf"), complex(0), -Q(1, 10**100))
)
def test_all_invalid_allowances_reject_before_source_or_barrier(
    field, bad, monkeypatch
):
    def forbidden(**kwargs):
        pytest.fail("invalid allowance reached source or barrier construction")

    monkeypatch.setattr(owner, "_assess_organized_source_attachment", forbidden)
    monkeypatch.setattr(owner, "_derive_organized_source_capture_barrier", forbidden)
    if field.startswith("host_"):
        bad = (Q(0), Q(0), Q(0), bad)
    with pytest.raises((TypeError, ValueError)):
        _assess(**{field: bad})


@pytest.mark.parametrize("field", ("host_form_error_bounds", "host_phase_error_bounds"))
@pytest.mark.parametrize("bad", ((), (0,) * 3, (0,) * 5, "0000", {0}))
def test_host_shapes_are_not_coerced_or_truncated(field, bad):
    with pytest.raises((TypeError, ValueError)):
        _assess(**{field: bad})


@pytest.mark.parametrize(
    "extra",
    (
        {"source": object()},
        {"barrier": object()},
        {"horizon": 1},
        {"clock": 1},
        {"donor_phase_origin": Q(1, 16)},
    ),
)
def test_reports_new_horizons_and_retuned_origins_cannot_replace_primitives(extra):
    with pytest.raises(TypeError):
        _assess(**extra)


def test_failed_source_keeps_candidate_bounds_but_suppresses_actual_outcomes(
    report, monkeypatch
):
    unavailable = replace(report.source, admitted=False, sources_certified=False)
    monkeypatch.setattr(
        owner, "_assess_organized_source_attachment", lambda **kwargs: unavailable
    )
    failed = _assess()
    assert failed.barrier_certified
    assert not failed.source_certified
    assert failed.conditional_source_in_cell
    assert not failed.source_in_cell
    assert not failed.capture_certified
    assert not failed.winding_retained
    assert not failed.converges_to_target
    assert not failed.control.joined_alignment_excluded
    assert failed.asymptotic_donor_host_phase_difference is None
    assert failed.status == "unavailable"
    assert failed.unresolved_conditions


def test_failed_barrier_does_not_inherit_source_admission(report, monkeypatch):
    unavailable = replace(report.barrier, barrier_certified=False)
    monkeypatch.setattr(
        owner, "_derive_organized_source_capture_barrier", lambda: unavailable
    )
    failed = _assess()
    assert failed.source_certified
    assert not failed.barrier_certified
    assert not failed.capture_certified
    assert not failed.winding_retained
    assert not failed.converges_to_target
    assert failed.unresolved_conditions


def test_large_original_allowance_remains_unavailable_without_changing_source():
    failed = _assess(donor_form_error_bound=1, donor_phase_error_bound=1)
    assert failed.source.donor.form_error_bound == 1
    assert failed.source.donor.phase_error_bound == 1
    assert failed.barrier_certified and failed.conditional_source_in_cell
    assert not failed.source_certified and not failed.source_in_cell
    assert not failed.capture_certified and not failed.winding_retained
    assert not failed.converges_to_target
    assert failed.asymptotic_donor_host_phase_difference is None
    assert not failed.control.joined_alignment_excluded
    assert failed.status == "unavailable"


def test_reports_are_immutable(report):
    with pytest.raises(FrozenInstanceError):
        report.capture_certified = False
    with pytest.raises(FrozenInstanceError):
        report.barrier.target_certified = False
