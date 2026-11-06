"""Independent controls for original-source formation on connected C9 support."""

import importlib
from fractions import Fraction as Q

import mpmath
import numpy as np
import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.physics import _organized_joint_acquisition as owner
from tnfr.physics import _organized_network_capture as capture_owner

_ERROR = Q(1, 10**10)
_EQUAL = (Q(1, 2),) * 3
_UNEQUAL = (Q(1, 4), Q(1, 2), Q(3, 4))
_CONTACTS = ((0, 1), (1, 2), (2, 0))
_RAMP = Q(2046, 9) * Q(355, 113) ** 2


@pytest.fixture(scope="module", autouse=True)
def no_response_or_isolated_source_execution():
    def forbidden(*args, **kwargs):
        pytest.fail("connected preparation consumed an isolated-source verdict")

    with forbid_sine_regeneration(), pytest.MonkeyPatch.context() as patch:
        for module_name, function in (
            ("_sine_constitutive_recovery", "_assess_constitutive_recovery"),
            ("_sine_class_constitutive_acquisition", "_assess_constitutive_class_pair"),
            (
                "_acquired_organized_network_capture",
                "_assess_acquired_organized_triangle",
            ),
            ("_organized_assembly_work", "_assess_organized_assembly_work"),
        ):
            module = importlib.import_module("tnfr.physics." + module_name)
            patch.setattr(module, function, forbidden)
            if hasattr(owner, function):
                patch.setattr(owner, function, forbidden)
        yield


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 100
    return context


def _mp(mp, value):
    return mp.mpf(value.numerator) / value.denominator


def _contains(mp, interval, value):
    assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)


def _assess(**changes):
    inputs = dict(mixing=_UNEQUAL, form_error_bound=_ERROR, phase_error_bound=_ERROR)
    return owner._assess_organized_joint_acquisition(**(inputs | changes))


@pytest.fixture(scope="module")
def reports():
    return tuple(_assess(mixing=weights) for weights in (_EQUAL, _UNEQUAL))


def _edges(mixing):
    internal = tuple(
        (9 * r + j, 9 * r + (j + 1) % 9, Q(1)) for r in range(3) for j in range(9)
    )
    cross = tuple(
        (9 * a + j, 9 * b + (j + offset) % 9, weight)
        for (a, b), s in zip(_CONTACTS, mixing)
        for j in range(9)
        for offset, weight in ((0, 1 - s), (1, s))
    )
    return internal + cross


def _storage(mp, edges, form, phase):
    return sum(
        _mp(mp, weight)
        * ((form[b] - form[a]) ** 2 / 2 + 1 - mp.cos(phase[b] - phase[a]))
        for a, b, weight in edges
    )


def _rows(mp, edges, form, phase):
    gradient, current = [mp.mpf(0)] * 27, [mp.mpf(0)] * 27
    for a, b, weight in edges:
        weight = _mp(mp, weight)
        form_flow = weight * (form[a] - form[b])
        sine_flow = weight * mp.sin(phase[b] - phase[a])
        gradient[a] += form_flow
        gradient[b] -= form_flow
        current[a] += sine_flow
        current[b] -= sine_flow
    gamma = 1 / (1023 * mp.pi)
    form_rate = [(-q + gamma * s) / 4 for q, s in zip(gradient, current)]
    phase_rate = [gamma * q / 4 for q in gradient]
    return gradient, current, form_rate, phase_rate


def test_fixed_connected_preparation_has_its_own_deadline_and_capture(reports):
    for report in reports:
        assert report.capture_geometry.geometry.weighted_edges == _edges(
            report.capture_geometry.geometry.mixing
        )
        assert report.capture_geometry.geometry.weighted_degrees == (4,) * 27
        assert report.nominal_initial_forms == tuple(
            _RAMP * (node % 9 - 4) for node in range(27)
        )
        assert report.nominal_initial_phases == (0,) * 27
        assert report.form_error_bound == report.phase_error_bound == _ERROR
        assert report.acquisition_time == 200
        assert report.original_acquisition_time == Q(204800, 1023)
        assert report.held_capacities == (1,) * 27
        assert report.relative_preparation_dimension == 48
        assert report.joined_mean_leaf_dimension == 52
        assert report.nominal_initial_windings == (0, 0, 0)
        assert report.target_windings == (1, 1, 1)
        assert report.source_budget_certified
        assert report.initial_zero_winding_certified
        assert report.entry_certified and report.formation_certified
        assert report.winding_retained and report.converges_to_target
        assert not report.unresolved_conditions
        assert report.capture_margin > 0


def test_complete_initial_cost_and_gradient_come_from_all_fine_edges(reports):
    for report in reports:
        form = tuple(_RAMP * (node % 9 - 4) for node in range(27))
        laplacian_form = [Q(0)] * 27
        internal = contact = Q(0)
        for a, b, weight in _edges(report.capture_geometry.geometry.mixing):
            difference = form[a] - form[b]
            laplacian_form[a] += weight * difference
            laplacian_form[b] -= weight * difference
            if a // 9 == b // 9:
                internal += weight * difference**2 / 2
            else:
                contact += weight * difference**2 / 2
        assert internal == report.nominal_initial_internal_form_storage
        assert contact == report.nominal_initial_contact_form_storage
        assert contact == internal / 2 > 0
        assert internal + contact == report.nominal_initial_form_storage
        assert report.nominal_initial_form_storage == 162 * _RAMP**2
        gradient_norm = sum(abs(value) for value in laplacian_form)
        assert report.initial_form_gradient_l1 == gradient_norm
        assert report.initial_storage_correction_upper_bound == (
            gradient_norm * _ERROR + 108 * (2 * _ERROR**2)
        )
        assert report.initial_storage_upper_bound == (
            internal + contact + report.initial_storage_correction_upper_bound
        )
        assert 815_000_000 < report.initial_storage_upper_bound < 816_000_000
        assert report.source_budget_margin == 10**9 - report.initial_storage_upper_bound
    assert reports[0].initial_form_gradient_l1 < reports[1].initial_form_gradient_l1


def test_asymmetric_original_error_states_obey_the_complete_storage_bound(reports, mp):
    error = _mp(mp, _ERROR)
    form_errors = [error * (node % 9 - 4) / (4 + node // 9) for node in range(27)]
    phase = [
        error * ((node % 9 - 4) ** 2 - mp.mpf(20) / 3) / (10 + node // 9)
        for node in range(27)
    ]
    form = [_mp(mp, _RAMP) * (node % 9 - 4) + form_errors[node] for node in range(27)]
    for start in (0, 9, 18):
        assert abs(sum(form_errors[start : start + 9])) < mp.mpf("1e-90")
        assert abs(sum(phase[start : start + 9])) < mp.mpf("1e-90")
    assert max(abs(value) for value in form_errors + phase) <= error
    for report in reports:
        actual = _storage(
            mp, _edges(report.capture_geometry.geometry.mixing), form, phase
        )
        nominal = _mp(mp, report.nominal_initial_form_storage)
        assert actual > nominal
        assert actual < _mp(mp, report.initial_storage_upper_bound)
        assert actual - nominal > mp.mpf("1e-6")


def test_full_support_gap_and_degree_bound_admit_the_global_semigroup(reports):
    for report in reports:
        laplacian = np.zeros((27, 27))
        for a, b, weight in _edges(report.capture_geometry.geometry.mixing):
            value = float(weight)
            laplacian[a, a] += value
            laplacian[b, b] += value
            laplacian[a, b] -= value
            laplacian[b, a] -= value
        spectrum = np.linalg.eigvalsh(laplacian / 4)
        assert abs(spectrum[0]) < 1e-12
        assert spectrum[1] > float(report.semigroup_gap_lower_bound) > 0
        assert spectrum[-1] <= 2
        assert report.semigroup_gap_lower_bound == Q(1, 10)
        # Inter-constituent means are retained fine directions, not quotiented out.
        vector = np.repeat((1.0, -1.0, 0.0), 9)
        assert np.allclose(laplacian @ vector, 3 * vector)


def test_outward_transport_matches_independent_high_precision_bounds(reports, mp):
    gamma, alpha = 1 / (1023 * mp.pi), 2 * mp.pi / 9
    error = _mp(mp, _ERROR)
    ramp = _mp(mp, _RAMP)
    forcing = mp.sqrt(27)
    nominal_norm = mp.sqrt(180) * ramp
    decay = mp.exp(-20)
    proxy = mp.sqrt(180) * abs(gamma * ramp - alpha)
    bx = decay * (nominal_norm + forcing * error) + 10 * gamma * forcing
    by = (
        proxy
        + forcing * (error + gamma * error)
        + gamma * bx
        + 200 * gamma**2 * forcing
    )
    assert proxy > mp.mpf("1e-6")  # The rational ramp never silently becomes pi.
    for report in reports:
        _contains(mp, report.exponential_decay_bounds, decay)
        _contains(mp, report.forcing_norm_bounds, forcing)
        _contains(mp, report.proxy_target_distance_bounds, proxy)
        xbound = _mp(mp, report.endpoint_form_norm_upper_bound)
        ybound = _mp(mp, report.endpoint_aligned_phase_norm_upper_bound)
        assert bx <= xbound < bx + mp.mpf("1e-20")
        assert by <= ybound < by + mp.mpf("1e-20")
        assert Q(1, 100) < report.endpoint_form_norm_upper_bound < Q(1, 50)
        assert Q(1, 10000) < report.endpoint_aligned_phase_norm_upper_bound < Q(1, 5000)
        assert report.endpoint_scaled_form_norm_upper_bound > 0
        assert report.endpoint_cell_margin > 1
        assert report.endpoint_acute_margin > Q(4, 5)


def test_unequal_entry_retains_first_variation_for_moving_child_means(reports, mp):
    alpha, epsilon = 2 * mp.pi / 9, mp.mpf(".00001")
    aligned = [alpha * (node % 9 - 4) for node in range(27)]
    zero = [mp.mpf(0)] * 27
    for report in reports:
        edges = _edges(report.capture_geometry.geometry.mixing)
        _, current, _, _ = _rows(mp, edges, zero, aligned)
        gradient_norm = mp.sqrt(sum(value**2 for value in current))
        if report.capture_geometry.geometry.mixing == _EQUAL:
            assert gradient_norm < mp.mpf("1e-90")
            assert report.endpoint_linear_phase_storage_upper_bound == 0
            continue
        _contains(
            mp, report.capture_geometry.aligned_gradient_norm_bounds, gradient_norm
        )
        assert gradient_norm > 1
        phase = [
            value - epsilon * s / gradient_norm for value, s in zip(aligned, current)
        ]
        assert abs(sum(phase)) < mp.mpf("1e-90")
        assert abs(sum(phase[:9])) > epsilon
        actual_change = _storage(mp, edges, zero, phase) - _storage(
            mp, edges, zero, aligned
        )
        assert actual_change > 4 * epsilon**2  # A quadratic-only handoff is invalid.
        assert actual_change < gradient_norm * epsilon + 4 * epsilon**2
        assert actual_change < _mp(mp, report.endpoint_linear_phase_storage_upper_bound)
    assert reports[0].endpoint_excess_storage_upper_bound < Q(11, 10000)
    assert reports[1].endpoint_excess_storage_upper_bound < Q(263, 1000)
    for report in reports:
        expected = (
            report.capture_geometry.nominal_excess_bounds.hi
            + report.endpoint_form_storage_upper_bound
            + report.endpoint_linear_phase_storage_upper_bound
            + report.endpoint_quadratic_phase_storage_upper_bound
        )
        assert report.endpoint_excess_storage_upper_bound == expected
        assert report.capture_margin == Q(196, 625) - expected


def test_complete_rows_give_integrated_lift_identity_and_only_global_means(mp):
    form = [mp.mpf(node % 9 - 4) / 7 + mp.mpf(node // 9 - 1) / 5 for node in range(27)]
    phase = [mp.sin(mp.mpf(node + 1) / 3) for node in range(27)]
    mean = sum(phase) / 27
    phase = [value - mean for value in phase]
    gamma = 1 / (1023 * mp.pi)
    for mixing in (_EQUAL, _UNEQUAL):
        q, s, xdot, tdot = _rows(mp, _edges(mixing), form, phase)
        assert abs(sum(xdot)) < mp.mpf("1e-90")
        assert abs(sum(tdot)) < mp.mpf("1e-90")
        assert any(
            abs(sum(tdot[start : start + 9])) > mp.mpf("1e-6") for start in (0, 9, 18)
        )
        for dx, dt, current in zip(xdot, tdot, s):
            assert abs(dt + gamma * dx - gamma**2 * current / 4) < mp.mpf("1e-90")
        storage_rate = sum(a * b - c * d for a, b, c, d in zip(q, xdot, s, tdot))
        loss = -sum(value**2 for value in q) / 4
        assert abs(storage_rate - loss) < mp.mpf("1e-90")
        # Restoring the original clock multiplies both rows and the loss.
        clock = mp.mpf(1023) / 1024
        original_rate = sum(
            a * clock * b - c * clock * d for a, b, c, d in zip(q, xdot, s, tdot)
        )
        assert abs(original_rate - clock * loss) < mp.mpf("1e-90")


def test_winding_uses_primitive_phase_lifts_and_endpoint_seam_integers(mp):
    alpha = 2 * mp.pi / 9
    initial = [_mp(mp, _ERROR) * (node % 9 - 4) / (5 + node // 9) for node in range(27)]
    endpoint = [
        alpha * (node % 9 - 4)
        + mp.mpf(".00001") * ((2, -1, -1)[node // 9] + (node % 9 - 4) / mp.mpf(10))
        for node in range(27)
    ]
    for phase, expected in ((initial, 0), (endpoint, 1)):
        for start in (0, 9, 18):
            raw = [phase[start + (j + 1) % 9] - phase[start + j] for j in range(9)]
            assert abs(sum(raw)) < mp.mpf("1e-90")
            wrapped = [mp.atan2(mp.sin(value), mp.cos(value)) for value in raw]
            assert abs(sum(wrapped) / (2 * mp.pi) - expected) < mp.mpf("1e-90")
            if expected:
                assert raw[-1] < -mp.pi
                assert max(abs(value) for value in wrapped) < 1


@pytest.mark.parametrize(
    "changes",
    (
        {"mixing": (True, Q(1, 2), Q(1, 2))},
        {"mixing": (0, 0, 0)},
        {"mixing": (Q(1, 2),) * 2},
        {"mixing": (Q(1, 2), Q(1, 2), Q(1, 2) + Q(1, 10**80))},
        {"form_error_bound": True},
        {"form_error_bound": -Q(1, 10**80)},
        {"phase_error_bound": float("nan")},
        {"phase_error_bound": float("inf")},
        {"phase_error_bound": "0.0"},
    ),
)
def test_invalid_primitives_reject_before_support_or_transit(changes, monkeypatch):
    def forbidden(**kwargs):
        pytest.fail("invalid primitive reached support or transit construction")

    monkeypatch.setattr(owner, "_derive_organized_triangle_capture", forbidden)
    monkeypatch.setattr(owner, "_prepared_transit_radii", forbidden)
    with pytest.raises((TypeError, ValueError)):
        _assess(**changes)


def test_large_allowances_are_unavailable_without_borrowing_isolated_recovery():
    report = _assess(form_error_bound=1000, phase_error_bound=2)
    assert report.form_error_bound == 1000 and report.phase_error_bound == 2
    assert report.capture_geometry.target_certified
    assert report.capture_geometry.barrier_certified
    assert not report.source_budget_certified
    assert not report.initial_zero_winding_certified
    assert not report.entry_certified and not report.formation_certified
    assert not report.winding_retained and not report.converges_to_target
    assert report.status == "unavailable"
    assert "connected_preparation_budget_not_certified" in report.unresolved_conditions
    assert "initial_zero_winding_not_certified" in report.unresolved_conditions
    assert "endpoint_energy_entry_not_certified" in report.unresolved_conditions


def test_static_boundary_failure_does_not_become_source_capture(monkeypatch):
    monkeypatch.setattr(
        capture_owner, "_edge_resistances", lambda geometry: (Q(1),) * 81
    )
    report = _assess()
    assert report.source_budget_certified and report.initial_zero_winding_certified
    assert report.capture_geometry.target_certified
    assert not report.capture_geometry.barrier_certified
    assert not report.entry_certified and not report.formation_certified
    assert not report.winding_retained and not report.converges_to_target
    assert "full_state_boundary_not_certified" in report.unresolved_conditions


@pytest.mark.parametrize(
    "errors, dimension", (((0, 0), 0), ((0, _ERROR), 24), ((Q(1, 10**100), 0), 24))
)
def test_exact_and_subgrid_allowances_keep_their_preparation_dimension(
    errors, dimension
):
    rx, rt = errors
    report = _assess(form_error_bound=rx, phase_error_bound=rt)
    assert report.form_error_bound == rx and report.phase_error_bound == rt
    assert report.relative_preparation_dimension == dimension
    assert report.formation_certified
    if rx or rt:
        assert report.initial_storage_correction_upper_bound > 0
        assert report.initial_storage_upper_bound > report.nominal_initial_form_storage
    else:
        assert report.initial_storage_upper_bound == report.nominal_initial_form_storage


@pytest.mark.parametrize(
    "extra",
    (
        {"recovery": object()},
        {"capture_geometry": object()},
        {"horizon": 1},
        {"scaled_time": 100},
        {"source_budget": 10**12},
    ),
)
def test_reports_deadlines_and_budgets_cannot_replace_frozen_primitives(extra):
    with pytest.raises(TypeError):
        _assess(**extra)
