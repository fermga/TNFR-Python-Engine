"""Independent endpoint controls for recovered C9 assembly work."""

from dataclasses import replace
from fractions import Fraction as Q

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.physics import _organized_assembly_work as owner

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
    return owner._assess_organized_assembly_work(**(inputs | changes))


@pytest.fixture(scope="module")
def reports():
    return tuple(_assess(mixing=weights) for weights in (_EQUAL, _UNEQUAL))


def _edges(mixing, *, contacts=True):
    internal = tuple(
        (9 * r + j, 9 * r + (j + 1) % 9, Q(1)) for r in range(3) for j in range(9)
    )
    cross = tuple(
        (9 * a + j, 9 * b + (j + offset) % 9, weight)
        for (a, b), s in zip(_CONTACTS, mixing)
        for j in range(9)
        for offset, weight in ((0, 1 - s), (1, s))
    )
    return internal + cross if contacts else internal


def _storage(mp, edges, form, phase):
    return sum(
        _mp(mp, weight)
        * ((form[b] - form[a]) ** 2 / 2 + 1 - mp.cos(phase[b] - phase[a]))
        for a, b, weight in edges
    )


def _source(mp):
    alpha, eps = 2 * mp.pi / 9, _mp(mp, _EPS)
    form = [eps * (node % 9 - 4) / (20 + 7 * (node // 9)) for node in range(27)]
    residual = [
        eps * ((node % 9 - 4) ** 2 - mp.mpf(20) / 3) / (80 + 11 * (node // 9))
        for node in range(27)
    ]
    phase = [alpha * (node % 9 - 4) + residual[node] for node in range(27)]
    return form, phase, residual


def test_rebuilds_original_recovery_and_actual_fixed_support(reports):
    for report in reports:
        assert report.recovery.recovery_certified
        assert report.recovery.deformation == 0
        assert report.recovery.endpoint_radius == _EPS
        assert report.recovery.scaled_handoff_time == 100 + 10**13
        assert report.recovery.form_error_bound == _ERROR
        assert report.recovery.phase_error_bound == _ERROR
        assert report.geometry.weighted_edges == _edges(report.geometry.mixing)
        assert report.source_certified and report.bound_certified
        assert report.zero_supply_excluded
        assert report.required_work_lower_bound == report.conditional_work_lower_bound
        assert report.required_work_lower_bound > 1
        assert not report.unresolved_conditions


def test_contact_floor_matches_independent_pair_minimizers(reports, mp):
    alpha = 2 * mp.pi / 9
    baseline = 27 * (1 - mp.cos(alpha))
    for report in reports:
        floor = mp.mpf(0)
        total_origin_shift = mp.mpf(0)
        for s, enclosure in zip(report.geometry.mixing, report.contact_floor_bounds):
            s = _mp(mp, s)
            shift = mp.atan2(s * mp.sin(alpha), 1 - s + s * mp.cos(alpha))
            # A single contact can put its uniform winding-one tail at the
            # argument of the head's weighted adjacent-node resultant.
            pair_cost = 9 * (
                (1 - s) * (1 - mp.cos(shift)) + s * (1 - mp.cos(alpha - shift))
            )
            _contains(mp, enclosure, pair_cost)
            assert pair_cost > 9 * s * (1 - s) * (1 - mp.cos(alpha))
            floor += pair_cost
            total_origin_shift += shift
        # Those independent contact minimizers cannot all be installed
        # around this cycle; this certificate does not claim a sharp minimum.
        assert abs(total_origin_shift - mp.pi / 3) < mp.mpf("1e-200")
        _contains(mp, report.total_contact_floor_bounds, floor)
        _contains(mp, report.endpoint_storage_floor_bounds, baseline + floor)
        assert _mp(mp, report.conditional_work_lower_bound) <= floor
        assert report.conditional_work_lower_bound == (
            report.total_contact_floor_bounds.lo - report.source_excess_upper_bound
        )


def test_actual_source_storage_keeps_subgrid_excess(reports, mp):
    form, phase, residual = _source(mp)
    eps = _mp(mp, _EPS)
    for start in (0, 9, 18):
        for values in (form, residual):
            part = values[start : start + 9]
            assert abs(sum(part)) < mp.mpf("1e-200")
            assert sum(value**2 for value in part) < eps**2
    baseline = 27 * (1 - mp.cos(2 * mp.pi / 9))
    storage = _storage(mp, _edges(_EQUAL, contacts=False), form, phase)
    assert storage > baseline
    for report in reports:
        excess = report.source_excess_upper_bound
        assert 0 < storage - baseline < _mp(mp, excess)
        assert excess == 12 * _EPS**2
        assert excess < report.source_storage_bounds.width
        assert (
            report.total_contact_floor_bounds.lo - report.conditional_work_lower_bound
            == excess
        )
        _contains(mp, report.source_storage_bounds, storage)


def test_nonregistered_endpoints_with_negative_internal_and_nonacute_contact_gaps(
    reports, mp
):
    alpha = 2 * mp.pi / 9
    origins = (mp.mpf("2.8"), mp.mpf("-.9"), mp.mpf(".37"))
    phase = []
    for component, origin in enumerate(origins):
        # The nine signed acute gaps sum to one full turn. The displaced
        # gap occurs at a different labelled edge in each constituent.
        perturbations = [mp.mpf(".15")] * 9
        perturbations[2 * component] = -mp.mpf("1.2")
        local = [origin]
        for perturbation in perturbations[:-1]:
            local.append(local[-1] + alpha + perturbation)
        phase.extend(local)
        gaps = [alpha + value for value in perturbations]
        assert min(gaps) < 0
        assert max(abs(gap) for gap in gaps) < mp.pi / 2
        assert abs(sum(gaps) - 2 * mp.pi) < mp.mpf("1e-200")
    form = [mp.mpf(".07") * (node % 9 - 4) + origins[node // 9] for node in range(27)]
    initial_form, initial_phase, _ = _source(mp)
    before = _storage(mp, _edges(_EQUAL, contacts=False), initial_form, initial_phase)
    for report in reports:
        edges = _edges(report.geometry.mixing)
        assert any(mp.cos(phase[b] - phase[a]) < 0 for a, b, _ in edges[27:])
        zero_form = [mp.mpf(0)] * 27
        for index, enclosure in enumerate(report.contact_floor_bounds):
            contact = edges[27 + 18 * index : 27 + 18 * (index + 1)]
            phase_cost = _storage(mp, contact, zero_form, phase)
            assert phase_cost > _mp(mp, enclosure.hi)
        after = _storage(mp, edges, form, phase)
        assert after > _mp(mp, report.endpoint_storage_floor_bounds.hi)
        assert after - before > _mp(mp, report.required_work_lower_bound)
        remaining = _storage(
            mp, _edges(report.geometry.mixing, contacts=False), form, phase
        )
        deletion = remaining - after
        assert deletion < -_mp(mp, report.total_contact_floor_bounds.hi)


def test_identity_and_recovered_source_energy_are_necessary_scope_restrictions(
    reports, mp
):
    zero = [mp.mpf(0)] * 27
    aligned = [2 * mp.pi * (node % 9 - 4) / 9 for node in range(27)]
    initial_form, initial_phase, _ = _source(mp)
    before = _storage(mp, _edges(_EQUAL, contacts=False), initial_form, initial_phase)
    energized = zero.copy()
    energized[0] = mp.mpf(100)
    unrecovered_before = _storage(
        mp, _edges(_EQUAL, contacts=False), energized, aligned
    )
    for report in reports:
        joined = _edges(report.geometry.mixing)
        # Erasing all three winding-one identities permits a passive reset.
        assert _storage(mp, joined, zero, zero) - before < 0
        # Keeping winding one can also be passive when a large source form
        # contrast is available. The recovered-source energy bound is crucial.
        assert _storage(mp, joined, zero, aligned) - unrecovered_before < 0


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
    monkeypatch.setattr(owner, "_derive_organized_triangle", forbidden)
    with pytest.raises((TypeError, ValueError)):
        _assess(**changes)


def test_unproved_source_does_not_promote_conditional_work_to_obstruction(
    reports, monkeypatch
):
    source = replace(reports[1].recovery, recovery_certified=False)
    monkeypatch.setattr(owner, "_assess_constitutive_recovery", lambda **kw: source)
    report = _assess()
    assert report.bound_certified
    assert report.conditional_work_lower_bound > 0
    assert not report.source_certified
    assert not report.zero_supply_excluded
    assert report.required_work_lower_bound is None
    assert report.status == "unavailable"
    assert report.unresolved_conditions


def test_large_original_preparation_cannot_borrow_recovered_bound():
    report = _assess(form_error_bound=1, phase_error_bound=1)
    assert report.recovery.form_error_bound == 1
    assert report.recovery.phase_error_bound == 1
    assert not report.source_certified
    assert not report.zero_supply_excluded
    assert report.required_work_lower_bound is None


def test_source_radius_must_admit_the_internal_acute_baseline(reports, monkeypatch):
    source = replace(reports[1].recovery, endpoint_radius=Q(1, 2))
    monkeypatch.setattr(owner, "_assess_constitutive_recovery", lambda **kw: source)
    report = _assess()
    assert report.recovery.recovery_certified
    assert not report.source_certified
    assert not report.zero_supply_excluded
    assert report.required_work_lower_bound is None
    assert (
        "original_source_internal_acuteness_not_certified"
        in report.unresolved_conditions
    )


@pytest.mark.parametrize(
    "extra",
    (
        {"recovery": object()},
        {"geometry": object()},
        {"endpoint": object()},
        {"work_reservoir": 100},
        {"horizon": 1},
    ),
)
def test_reports_endpoints_and_reservoirs_do_not_replace_primitive_premises(extra):
    with pytest.raises(TypeError):
        _assess(**extra)
