"""Independent full-state controls for capture of three acquired C9 objects."""

from dataclasses import replace
from fractions import Fraction as Q

import mpmath
import numpy as np
import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.physics import _acquired_organized_network_capture as owner
from tnfr.physics import _organized_network_capture as capture_owner

_ERROR = Q(1, 10**10)
_EPS = Q(1, 10**70)
_EQUAL = (Q(1, 2),) * 3
_UNEQUAL = (Q(1, 4), Q(1, 2), Q(3, 4))
_CONTACTS = ((0, 1), (1, 2), (2, 0))


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


def _contains(mp, interval, value):
    assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)


def _assess(**changes):
    inputs = dict(mixing=_UNEQUAL, form_error_bound=_ERROR, phase_error_bound=_ERROR)
    return owner._assess_acquired_organized_triangle(**(inputs | changes))


@pytest.fixture(scope="module")
def reports():
    return tuple(_assess(mixing=weights) for weights in (_EQUAL, _UNEQUAL))


def _edges(mixing):
    internal = tuple(
        (9 * r + j, 9 * r + (j + 1) % 9, Q(1), 1) for r in range(3) for j in range(9)
    )
    cross = tuple(
        (9 * a + j, 9 * b + (j + offset) % 9, weight, offset)
        for (a, b), s in zip(_CONTACTS, mixing)
        for j in range(9)
        for offset, weight in ((0, 1 - s), (1, s))
    )
    return internal + cross


def _target(mp, mixing):
    alpha = 2 * mp.pi / 9
    if mixing == _EQUAL:
        return (mp.mpf(0),) * 3

    def currents(first, second):
        origins = (first, second, -first - second)
        return tuple(
            (1 - _mp(mp, s)) * mp.sin(origins[b] - origins[a])
            + _mp(mp, s) * mp.sin(origins[b] - origins[a] + alpha)
            for (a, b), s in zip(_CONTACTS, mixing)
        )

    def residual(first, second):
        current = currents(first, second)
        return current[0] - current[2], current[1] - current[0]

    first, second = mp.findroot(residual, (-mp.mpf(".12"), mp.mpf(".058")))
    return (first, second, -first - second)


@pytest.fixture(scope="module")
def targets(mp):
    return tuple(_target(mp, mixing) for mixing in (_EQUAL, _UNEQUAL))


def test_fresh_isolated_source_is_carried_into_actual_three_object_support(reports):
    for report in reports:
        assert report.recovery.recovery_certified
        assert report.recovery.deformation == 0
        assert report.recovery.endpoint_radius == _EPS
        assert report.recovery.scaled_handoff_time == 100 + 10**13
        assert report.recovery.form_error_bound == _ERROR
        assert report.recovery.phase_error_bound == _ERROR
        assert report.geometry.weighted_degrees == (4,) * 27
        assert len(report.geometry.weighted_edges) == 81
        assert report.source_certified
        assert report.target_certified
        assert report.barrier_certified
        assert report.capture_certified
        assert report.winding_retained
        assert report.converges_to_target
        assert not report.unresolved_conditions
        assert report.capture_margin > 0


def test_independent_full_fine_target_criticality_and_registered_gaps(
    reports, targets, mp
):
    alpha = 2 * mp.pi / 9
    for report, origins in zip(reports, targets):
        assert abs(sum(origins)) < mp.mpf("1e-200")
        for enclosure, value in zip(report.target_phase_origins_bounds, origins):
            _contains(mp, enclosure, value)
        delta = tuple(origins[b] - origins[a] for a, b in _CONTACTS)
        for enclosure, value in zip(report.target_contact_gaps_bounds, delta):
            _contains(mp, enclosure, value)
        current = [mp.mpf(0)] * 27
        potential = mp.mpf(0)
        for (a, b, w, offset), enclosure in zip(
            _edges(report.geometry.mixing), report.target_fine_gap_bounds
        ):
            gap = origins[b // 9] - origins[a // 9] + offset * alpha
            _contains(mp, enclosure, gap)
            assert abs(gap) < mp.mpf(".88")
            flow = _mp(mp, w) * mp.sin(gap)
            current[a] += flow
            current[b] -= flow
            potential += _mp(mp, w) * (1 - mp.cos(gap))
        assert max(abs(value) for value in current) < mp.mpf("1e-200")
        aligned = mp.mpf("40.5") * (1 - mp.cos(alpha))
        nominal_excess = 0 if report.geometry.mixing == _EQUAL else aligned - potential
        _contains(mp, report.nominal_excess_bounds, nominal_excess)
        if report.geometry.mixing == _EQUAL:
            assert max(abs(value) for value in origins) < mp.mpf("1e-200")
            assert report.target_root_bounds is None
        else:
            assert origins[0] < -mp.mpf(".12")
            assert origins[1] > mp.mpf(".058")
            assert mp.mpf(".261") < aligned - potential < mp.mpf(".262")
            left, right = report.target_root_endpoint_sign_bounds
            assert left.hi < 0 < right.lo or right.hi < 0 < left.lo


def test_target_full_transverse_hessian_and_fine_graph_gap(reports, targets, mp):
    alpha = 2 * mp.pi / 9
    for report, origins in zip(reports, targets):
        laplacian, hessian = np.zeros((27, 27)), np.zeros((27, 27))
        for a, b, weight, offset in _edges(report.geometry.mixing):
            gap = origins[b // 9] - origins[a // 9] + offset * alpha
            for matrix, factor in (
                (laplacian, float(weight)),
                (hessian, float(weight) * float(mp.cos(gap))),
            ):
                matrix[a, a] += factor
                matrix[b, b] += factor
                matrix[a, b] -= factor
                matrix[b, a] -= factor
        fine_spectrum = np.linalg.eigvalsh(laplacian)
        phase_spectrum = np.linalg.eigvalsh(hessian)
        assert abs(fine_spectrum[0]) < 1e-12
        assert abs(phase_spectrum[0]) < 1e-12
        assert fine_spectrum[1] > float(report.laplacian_gap_lower_bound)
        assert phase_spectrum[1] > float(report.target_hessian_lower_bound) > 0
        assert report.laplacian_gap_lower_bound == Q(2, 5)
        assert report.target_hessian_lower_bound <= Q(2, 5)


def test_all_edge_resistances_come_from_the_actual_weighted_fine_laplacian(reports, mp):
    for report in reports:
        edges = _edges(report.geometry.mixing)
        laplacian = mp.matrix(27)
        for a, b, weight, _ in edges:
            weight = _mp(mp, weight)
            laplacian[a, a] += weight
            laplacian[b, b] += weight
            laplacian[a, b] -= weight
            laplacian[b, a] -= weight
        grounded = laplacian[:26, :26] ** -1
        for (a, b, _, _), exact in zip(edges, report.edge_effective_resistances):
            vector = mp.matrix(26, 1)
            if a < 26:
                vector[a] = 1
            if b < 26:
                vector[b] = -1
            resistance = (vector.T * grounded * vector)[0]
            assert abs(resistance - _mp(mp, exact)) < mp.mpf("1e-200")
            assert 0 < exact < Q(3, 5)
        assert len(report.edge_effective_resistances) == len(edges)
        # The degree-four fine graph, not the three-node surrogate, supplies
        # the resistance controlling a boundary in any of the 27 fine nodes.
        assert max(report.edge_effective_resistances) > Q(1, 2)


@pytest.fixture
def resistance_solves(monkeypatch):
    cache = capture_owner._support_edge_resistances
    cache.cache_clear()
    inverse = capture_owner.exact_matrix_inverse
    matrices = []

    def counted(matrix):
        matrices.append(matrix)
        return inverse(matrix)

    monkeypatch.setattr(capture_owner, "exact_matrix_inverse", counted)
    yield matrices
    cache.cache_clear()


@pytest.mark.parametrize("mixing", (_EQUAL, _UNEQUAL))
def test_resistance_cache_preserves_exact_values_and_original_edge_order(
    mixing, resistance_solves
):
    cold = capture_owner._derive_organized_triangle_capture(mixing=mixing)
    warm = capture_owner._derive_organized_triangle_capture(mixing=mixing)
    assert warm == cold
    assert warm is not cold and warm.geometry is not cold.geometry
    assert len(resistance_solves) == 1
    reordered = replace(
        cold.geometry,
        nodes=tuple(reversed(cold.geometry.nodes)),
        weighted_edges=tuple(
            (b, a, float(weight))
            for a, b, weight in reversed(cold.geometry.weighted_edges)
        ),
    )
    assert capture_owner._edge_resistances(reordered) == tuple(
        reversed(cold.edge_effective_resistances)
    )
    assert len(resistance_solves) == 1


def test_resistance_cache_keys_actual_weights_and_support_and_is_bounded(
    resistance_solves,
):
    geometry = capture_owner._derive_organized_triangle(mixing=_EQUAL)
    original = capture_owner._edge_resistances(geometry)
    doubled = replace(
        geometry,
        weighted_edges=tuple((a, b, 2 * w) for a, b, w in geometry.weighted_edges),
    )
    assert capture_owner._edge_resistances(doubled) == tuple(r / 2 for r in original)
    assert len(resistance_solves) == 2
    reduced = replace(geometry, weighted_edges=geometry.weighted_edges[1:])
    removed_edge = capture_owner._edge_resistances(reduced)
    # Removing conductance cannot reduce the remaining effective resistances.
    assert all(new >= old for new, old in zip(removed_edge, original[1:]))
    assert any(new > old for new, old in zip(removed_edge, original[1:]))
    assert len(resistance_solves) == 3
    assert capture_owner._support_edge_resistances.cache_info().currsize == 2
    assert capture_owner._edge_resistances(geometry) == original
    assert len(resistance_solves) == 4  # The oldest support was evicted.


@pytest.mark.parametrize(
    "changes",
    (
        {"nodes": (False, *range(1, 27))},
        {"edge": (0, True, Q(1))},
        {"edge": (0, 1, True)},
        {"edge": (0, 1, float("nan"))},
        {"edge": (0, 1, float("inf"))},
        {"edge": (0, 1, "1")},
        {"edge": (0, 1, 0)},
        {"edge": (0, 1, -1)},
        {"edge": (0, 27, 1)},
        {"edge": (0, 0, 1)},
        {"edge": (1, 2, 1)},  # Duplicate an existing undirected edge.
    ),
)
def test_invalid_support_cannot_enter_a_warm_resistance_cache(
    changes, resistance_solves
):
    geometry = capture_owner._derive_organized_triangle(mixing=_EQUAL)
    capture_owner._edge_resistances(geometry)
    previous = capture_owner._support_edge_resistances.cache_info()
    if "edge" in changes:
        changes = {"weighted_edges": (changes["edge"], *geometry.weighted_edges[1:])}
    with pytest.raises((TypeError, ValueError)):
        capture_owner._edge_resistances(replace(geometry, **changes))
    assert capture_owner._support_edge_resistances.cache_info() == previous
    assert len(resistance_solves) == 1


def test_warm_resistances_cannot_supply_a_target_or_boundary_verdict(
    resistance_solves, monkeypatch
):
    warm = _assess()
    assert warm.capture_certified
    target = capture_owner._target
    with monkeypatch.context() as patch:
        patch.setattr(capture_owner, "_target", lambda g: (*target(g)[:-1], False))
        failed_target = _assess()
    assert not failed_target.target_certified and not failed_target.capture_certified
    monkeypatch.setattr(capture_owner, "_RESISTANCE", Q(1, 2))
    failed_boundary = _assess()
    assert failed_boundary.target_certified
    assert not failed_boundary.barrier_certified
    assert not failed_boundary.capture_certified
    assert len(resistance_solves) == 1


def test_nonacute_trap_uses_bregman_and_radial_bounds_not_positive_hessian(reports, mp):
    dmax, limit = mp.mpf(".88"), mp.mpf(2)
    coefficient = (mp.cos(dmax) - mp.cos(limit) - (limit - dmax) * mp.sin(dmax)) / (
        limit - dmax
    ) ** 2
    assert coefficient > mp.mpf(3) / 20
    assert mp.cos(limit) < 0
    assert limit + dmax < mp.pi
    for target in (-dmax, -mp.mpf(".2"), mp.mpf(0), dmax):
        for actual in (-limit, -mp.mpf("1.8"), mp.mpf(".4"), limit):
            change = actual - target
            bregman = mp.cos(target) - mp.cos(actual) - mp.sin(target) * change
            assert bregman >= coefficient * change**2 - mp.mpf("1e-200")
            assert change * (mp.sin(actual) - mp.sin(target)) > 0
    barrier = Q(3, 20) * (Q(2) - Q(22, 25)) ** 2 / Q(3, 5)
    assert barrier == Q(196, 625)
    for report in reports:
        _contains(mp, report.bregman_coefficient_bounds, coefficient)
        assert report.boundary_energy_lower_bound == barrier
        assert report.initial_excess_upper_bound < Q(27, 100) < barrier
        assert report.capture_margin == barrier - report.initial_excess_upper_bound


def test_unreset_asymmetric_fine_errors_keep_exact_tiny_work_account(reports, mp):
    alpha, eps = 2 * mp.pi / 9, _mp(mp, _EPS)
    form = [eps * (node % 9 - 4) / (20 + 7 * (node // 9)) for node in range(27)]
    phase = [
        eps * ((node % 9 - 4) ** 2 - mp.mpf(20) / 3) / (80 + 11 * (node // 9))
        for node in range(27)
    ]
    for start in (0, 9, 18):
        assert abs(sum(form[start : start + 9])) < mp.mpf("1e-200")
        assert abs(sum(phase[start : start + 9])) < mp.mpf("1e-200")
        assert sum(value**2 for value in form[start : start + 9]) < eps**2
        assert sum(value**2 for value in phase[start : start + 9]) < eps**2
    nominal_work = mp.mpf("13.5") * (1 - mp.cos(alpha))
    for report in reports:
        total, work = mp.mpf(0), mp.mpf(0)
        for index, (a, b, weight, offset) in enumerate(_edges(report.geometry.mixing)):
            gap = offset * alpha + phase[b] - phase[a]
            energy = _mp(mp, weight) * ((form[b] - form[a]) ** 2 / 2 + 1 - mp.cos(gap))
            total += energy
            if index >= 27:
                work += energy
        residual = work - nominal_work
        low, high = report.attachment_residual_work_bounds
        assert 0 < residual < _mp(mp, high)
        assert _mp(mp, low) <= residual
        assert high < Q(1, 10**137)
        assert report.attachment_work_bounds.width > high
        _contains(mp, report.attachment_work_bounds, work)
        aligned_energy = 27 * (1 - mp.cos(alpha)) + nominal_work
        assert 0 < total - aligned_energy < _mp(mp, report.source_excess_upper_bound)


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
def test_invalid_or_unsupported_primitives_reject_before_source(changes, monkeypatch):
    def forbidden(**kwargs):
        pytest.fail("invalid primitive reached recovery or support construction")

    monkeypatch.setattr(owner, "_assess_constitutive_recovery", forbidden)
    monkeypatch.setattr(owner, "_derive_organized_triangle_capture", forbidden)
    with pytest.raises((TypeError, ValueError)):
        _assess(**changes)


def test_failed_source_does_not_inherit_the_static_target_verdict(reports, monkeypatch):
    source = replace(reports[1].recovery, recovery_certified=False)
    monkeypatch.setattr(owner, "_assess_constitutive_recovery", lambda **kw: source)
    report = _assess()
    assert report.target_certified
    assert not report.source_certified
    assert not report.capture_certified
    assert not report.winding_retained
    assert not report.converges_to_target
    assert report.joint_error_squared_upper_bound is None
    assert report.status == "unavailable"
    assert report.unresolved_conditions


def test_unproved_large_source_is_unavailable_without_changing_its_preparation():
    report = _assess(form_error_bound=1, phase_error_bound=1)
    assert report.recovery.form_error_bound == 1
    assert report.recovery.phase_error_bound == 1
    assert report.target_certified and report.barrier_certified
    assert not report.source_certified and not report.capture_certified
    assert report.joint_error_squared_upper_bound is None
    assert "original_source_recovery_not_certified" in report.unresolved_conditions


def test_unproved_resistance_does_not_pass_the_full_state_boundary(monkeypatch):
    monkeypatch.setattr(
        capture_owner, "_edge_resistances", lambda geometry: (Q(1),) * 81
    )
    report = _assess()
    assert report.source_certified and report.target_certified
    assert not report.barrier_certified and not report.capture_certified
    assert "full_state_boundary_not_certified" in report.unresolved_conditions


@pytest.mark.parametrize(
    "extra", ({"recovery": object()}, {"geometry": object()}, {"horizon": 1})
)
def test_reports_and_new_horizons_cannot_replace_primitive_premises(extra):
    with pytest.raises(TypeError):
        _assess(**extra)
