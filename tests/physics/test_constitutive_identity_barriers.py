"""Independent first-boundary costs and own-law proof-domain controls."""

from fractions import Fraction as Q

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.physics import _constitutive_identity_barriers as owner

_LIMIT = Q(1, 10000)


@pytest.fixture(scope="module", autouse=True)
def no_response_or_frozen_execution():
    with forbid_sine_regeneration():
        yield


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 100
    return context


def _mp(mp, value):
    return mp.mpf(value.numerator) / value.denominator


def _contains(mp, bounds, value):
    assert _mp(mp, bounds.lo) <= value <= _mp(mp, bounds.hi)
    assert _mp(mp, bounds.width) < mp.mpf("1e-30")


def _potential(mp, angle, deformation):
    return (
        1 - mp.cos(angle) + deformation * mp.cos(angle) * (1 - mp.cos(9 * angle)) ** 2
    )


@pytest.fixture(scope="module")
def reports():
    return {
        e: owner._assess_constitutive_identity_barriers(deformation=e)
        for e in (-_LIMIT, Q(0), _LIMIT)
    }


@pytest.mark.parametrize("deformation", (-_LIMIT, Q(0), _LIMIT))
def test_boundary_minimum_matches_independent_potential_and_strict_proof_margins(
    deformation, reports, mp
):
    report, e = reports[deformation], _mp(mp, deformation)

    def potential(angle):
        return _potential(mp, angle, e)

    remainder = 8 * potential(mp.pi / 8)
    expected = potential(mp.pi) + remainder
    baseline = 2 + 8 * (1 - mp.cos(mp.pi / 8))
    slope = mp.diff(
        lambda value: _potential(mp, mp.pi, value)
        + 8 * _potential(mp, mp.pi / 8, value),
        e,
    )
    target = 9 * potential(2 * mp.pi / 9)
    _contains(mp, report.boundary_storage_bounds, expected)
    _contains(mp, report.sine_boundary_storage_bounds, baseline)
    _contains(mp, report.barrier_slope_bounds, slope)
    _contains(mp, report.barrier_change_bounds, expected - baseline)
    _contains(mp, report.target_storage_bounds, target)
    _contains(mp, report.donor_target_margin_bounds, expected - target)
    assert 23 < slope < 28
    size = abs(e)
    outside_gap_floor = 1 - mp.cos(5 * mp.pi / 12) - 4 * size
    assert (
        outside_gap_floor - remainder
        > _mp(mp, report.large_gap_exclusion_lower_bound)
        > 0
    )
    curvature_floor = mp.cos(5 * mp.pi / 12) - 328 * size
    assert curvature_floor > _mp(mp, report.remaining_gap_convexity_lower_bound) > 0
    opposite_floor = mp.mpf(9) / 4 - 32 * size
    assert (
        opposite_floor - remainder
        > _mp(mp, report.opposite_face_exclusion_lower_bound)
        > 0
    )
    assert report.geometric_minimum_certified
    assert report.status == "certified_geometric_boundary_minimum"
    assert not report.unresolved_conditions
    assert not report.source_admitted
    assert not report.trajectory_evaluated


@pytest.mark.parametrize("deformation", (-_LIMIT, _LIMIT))
@pytest.mark.parametrize("winding,boundary_sign", ((0, -1), (0, 1), (1, 1)))
def test_registered_boundary_minimizers_and_nonuniform_cost_controls(
    deformation, winding, boundary_sign, reports, mp
):
    e = _mp(mp, deformation)
    first = boundary_sign * mp.pi
    rest = (2 * mp.pi * winding - first) / 8
    gaps = [first] + [rest] * 8
    assert abs(sum(gaps) - 2 * mp.pi * winding) < mp.mpf("1e-95")
    assert all(abs(value) < mp.pi for value in gaps[1:])
    baseline = sum(_potential(mp, gap, e) for gap in gaps)
    _contains(mp, reports[deformation].boundary_storage_bounds, baseline)
    # Independent cost controls exercise the strictly convex face, not a
    # sampled substitute for the global exclusion/Jensen proof.
    for amplitude in (mp.mpf("0.0001"), mp.mpf("0.04")):
        altered = gaps.copy()
        altered[1] += amplitude
        altered[2] -= amplitude
        assert sum(_potential(mp, gap, e) for gap in altered) > baseline
    if winding == 1:
        opposite = [-mp.pi] + [3 * mp.pi / 8] * 8
        assert sum(_potential(mp, gap, e) for gap in opposite) > baseline + 1


def test_distinct_barriers_keep_identical_target_and_shift_both_costs(reports, mp):
    lower, upper = reports[-_LIMIT], reports[_LIMIT]
    assert lower.target_storage_bounds == upper.target_storage_bounds
    barrier_difference = upper.boundary_storage_bounds - lower.boundary_storage_bounds
    margin_difference = (
        upper.donor_target_margin_bounds - lower.donor_target_margin_bounds
    )
    c = mp.cos(mp.pi / 8)
    expected = 2 * _mp(mp, _LIMIT) * (8 * c * (1 + c) ** 2 - 4)
    _contains(mp, barrier_difference, expected)
    _contains(mp, margin_difference, expected)
    assert barrier_difference.lo > Q(23, 5000)
    assert margin_difference.lo > Q(23, 5000)


@pytest.mark.parametrize("deformation", (-_LIMIT, _LIMIT))
def test_actual_equal_network_target_keeps_own_law_storage_and_two_field_jets(
    deformation, mp
):
    e = _mp(mp, deformation)
    phases = [2 * mp.pi * (j - 4) / 9 for _ in range(3) for j in range(9)]
    edges = [
        (9 * r + j, 9 * r + (j + 1) % 9, mp.mpf(1)) for r in range(3) for j in range(9)
    ] + [
        (9 * a + j, 9 * b + (j + offset) % 9, mp.mpf("0.5"))
        for a, b in ((0, 1), (1, 2), (2, 0))
        for j in range(9)
        for offset in (0, 1)
    ]
    current, storage = [mp.mpf(0)] * 27, mp.mpf(0)

    def own(angle):
        return _potential(mp, angle, e)

    def baseline(angle):
        return 1 - mp.cos(angle)

    for a, b, weight in edges:
        gap = phases[b] - phases[a]
        # These are the actual fine-edge gaps, including closing lifts.
        for order in range(4):
            assert abs(mp.diff(own, gap, order) - mp.diff(baseline, gap, order)) < (
                mp.mpf("1e-85")
            )
        flow = weight * mp.diff(own, gap)
        current[a] += flow
        current[b] -= flow
        storage += weight * own(gap)
    assert max(map(abs, current)) < mp.mpf("1e-90")
    expected = mp.mpf(81) / 2 * (1 - mp.cos(2 * mp.pi / 9))
    assert abs(storage - expected) < mp.mpf("1e-90")


@pytest.mark.parametrize("deformation", (Q(-1, 324), Q(1, 324), _LIMIT + Q(1, 10**30)))
def test_valid_law_outside_boundary_proof_has_no_fabricated_minimum(deformation):
    report = owner._assess_constitutive_identity_barriers(deformation=deformation)
    assert report.law.potential_relative_lower_bound >= Q(1, 2)
    assert not report.geometric_minimum_certified
    assert report.status == "unavailable"
    assert report.boundary_storage_bounds is None
    assert report.barrier_change_bounds is None
    assert report.donor_target_margin_bounds is None
    assert report.unresolved_conditions


@pytest.mark.parametrize(
    "deformation", (True, "0", float("nan"), float("inf"), Q(1, 323), Q(-1, 323))
)
def test_invalid_primitive_law_rejects_before_trigonometry(deformation, monkeypatch):
    def forbidden(*args):
        pytest.fail("invalid primitive reached the boundary calculation")

    monkeypatch.setattr(owner, "pi_interval", forbidden)
    monkeypatch.setattr(owner, "cos", forbidden)
    with pytest.raises((ValueError, TypeError)):
        owner._assess_constitutive_identity_barriers(deformation=deformation)
