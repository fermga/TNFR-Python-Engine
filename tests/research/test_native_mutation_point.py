"""Independent readmission controls, separate from the frozen source proposal.

Only the earlier formal capacity/phase frame and zero or coordinate pressure
rows are substituted. These controls identify exact fields and admission
boundaries; they are neither searched preparations nor native observations.
The archived 24-run proposer and its selected points are not imported here.
"""

import builtins
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q
from itertools import combinations, product
from math import factorial
from pathlib import Path

import pytest

from tests.research import test_native_mutation_feasibility as prior
from tnfr.research import _native_mutation_feasibility as shared
from tnfr.research import _native_mutation_point as point

no_execution = prior.no_execution
calculus_guard = prior.calculus_guard
renewal_guard = prior.renewal_guard

PAIRS = tuple(product(prior.BRANCHES, ("stable", "transition")))
PRESSURES = ((Q(0),) * 4,) + tuple(tuple(Q(i == j) for i in range(4)) for j in range(4))


def _fine(result, eta=Q(0)):
    return prior._fine_coefficients(
        result.spec, (*result.pressure_coordinates, eta), prior._defaults()
    )


def _owners(values, permitted):
    peak = max(map(abs, values))
    return tuple(
        owner for owner in permitted if peak > 0 and owner[1] * values[owner[0]] == peak
    )


def _interval_status(bounds, threshold, *, lower, strict):
    """Independent closed/open interval logic, not the owner classifier."""
    left, right = bounds
    if threshold is None:
        return "not_required"
    if lower:
        if left > threshold or (not strict and left == threshold):
            return "certified"
        if right < threshold or (strict and right == threshold):
            return "rejected"
    else:
        if right < threshold or (not strict and right == threshold):
            return "certified"
        if left > threshold or (strict and left == threshold):
            return "rejected"
    return "unresolved"


@pytest.fixture(scope="module")
def points(renewal_guard):
    targets, forbidden = renewal_guard
    assert all(getattr(owner, name) is forbidden for owner, name in targets)
    return {
        (first, second, i): point.check_native_mutation_point(
            prior._spec(first, second), pressure
        )
        for first, second in PAIRS
        for i, pressure in enumerate(PRESSURES)
    }


def test_ten_constituent_law_and_both_held_flows_are_rebuilt(points):
    for result in points.values():
        fine = _fine(result)
        p, q = (values[::2] for values in fine["pressure"])
        assert (
            result.pressure
            == p
            == (
                *result.pressure_coordinates,
                -sum(result.pressure_coordinates),
            )
        )
        assert result.successor_pressure == q
        assert sum(p) == sum(q) == 0
        assert result.form_offsets == tuple(values[::2] for values in fine["forms"])
        assert result.first_phase == fine["phase"][1][::2]
        assert result.mutation_phase == fine["phase"][2][::2]
        assert result.terminal_phase == fine["phase"][3][::2]
        assert result.pressure_peaks == tuple(
            max(map(abs, values)) for values in (p, q)
        )
        # The second declared controller cannot change pressure already
        # prepared before Mutation and that controller's action.
    for first, i in product(prior.BRANCHES, range(len(PRESSURES))):
        assert (
            points[first, "stable", i].successor_pressure
            == points[first, "transition", i].successor_pressure
        )


def test_one_offset_midpoint_uses_all_fifteen_boundaries(points):
    for result in points.values():
        offsets = tuple(value for boundary in result.form_offsets for value in boundary)
        lower, upper = -1 - min(offsets), 1 - max(offsets)
        assert result.common_offset_interval == (lower, upper)
        assert result.common_offset == (lower + upper) / 2
        expected = tuple(
            tuple(value + result.common_offset for value in boundary)
            for boundary in result.form_offsets
        )
        assert result.form_boundaries == expected
        actual = tuple(value for boundary in expected for value in boundary)
        assert all(-1 <= value <= 1 for value in actual) == (lower <= upper)
        assert max(actual) + min(actual) == 0
        # A failed interval is retained as failed, not clipped or replaced
        # by unrelated offsets for the two endpoints.
        if lower > upper:
            assert result.status == "rejected"
            assert min(actual) < -1 and max(actual) > 1


def test_actual_peak_ties_and_every_selected_cell_row_match_fine_fields(points):
    for result in points.values():
        p, q = result.pressure, result.successor_pressure
        source = _owners(p, prior.SOURCE_OWNERS)
        following = _owners(q, prior.NEXT_OWNERS)
        assert result.source_peak_owners == source
        assert result.successor_peak_owners == following
        if result.selected_cell_key is None:
            assert result.cell_unavailability
            assert not result.row_checks
            continue
        assert result.selected_cell_key in {
            shared.CellKey(a, b) for a, b in product(source, following)
        }
        fine = _fine(result, result.common_offset)
        expected = prior._expected_residuals(
            prior._defaults(), fine, result.selected_cell_key
        )
        assert {row.label for row in result.row_checks} == set(expected)
        for row in result.row_checks:
            residual, strict, unit = expected[row.label]
            assert (row.residual, row.strict, row.unit) == (residual, strict, unit)
            assert row.residual == row.lhs - row.rhs
            assert row.satisfied == (residual < 0 if strict else residual <= 0)


def _assert_both_stability_masks(result, fine):
    """Compare signed rows to direct Si on an explicitly available cell."""
    law = prior._defaults()
    gamma = law["stability"][2]
    assert result.selected_cell_key is not None
    base = tuple(
        prior._si_base(law, fine["capacity"], phases)[::2]
        for phases in fine["phase"][:2]
    )
    mutated_base = prior._si_base(law, fine["capacity"], fine["phase"][2])[::2]
    assert mutated_base != base[1]
    rows = {row.label: row for row in result.row_checks}
    for prefix, values, maxima, baseline in zip(
        ("source", "successor"),
        (result.pressure, result.successor_pressure),
        result.pressure_peaks,
        base,
    ):
        assert maxima > 0
        si = tuple(
            value - gamma * abs(pressure) / maxima
            for value, pressure in zip(baseline, values)
        )
        for i in range(1, 5):
            signed = [
                rows[f"{prefix}.il.b{i}.{sign}"] for sign in ("positive", "negative")
            ]
            assert all(row.satisfied for row in signed) == (si[i] >= law["hi"])
        assert rows[f"{prefix}.target.si_low"].satisfied == (si[0] <= law["lo"])


def test_timestamped_secants_are_derived_from_associated_form_samples(points):
    law = prior._defaults()
    times = (Q(1), Q(17, 16), Q(9, 8))
    for result in points.values():
        expected = tuple(
            tuple((time, result.form_boundaries[j][i]) for j, time in enumerate(times))
            for i in range(5)
        )
        assert result.temporal_histories == expected
        secants = tuple(
            (samples[1][1] - samples[0][1]) / prior.DT for samples in expected
        )
        assert result.first_secants == secants
        assert secants == tuple(
            result.spec.capacities[i]
            * (law["f"] if i == 0 else law["ell"])
            * result.pressure[i]
            for i in range(5)
        )
        assert result.premises.times == times
        assert result.premises.node_ordinals == (3, 4, 5)
        assert result.premises.runtime_ordinals == (0, 1, 2)
        assert result.premises.traces[2][0] == ("AL", "IL", "EN", "OZ", "ZHIR")


def test_fixed_cosine_budget_and_actual_two_controller_bounds(points):
    grid = Q(1, 1 << 128)
    for result in points.values():
        phases = (result.spec.phases, result.mutation_phase)
        for check, phase in zip(result.controller_checks, phases):
            assert (check.terms, check.bits) == (12, 128)
            assert (
                check.method
                == "rational_machin_pi_reduction_alternating_cosine_lipschitz_v1"
            )
            obligation = check.obligation
            gaps = tuple(a - b for a, b in combinations(phase, 2))
            assert tuple(map(abs, obligation.phase_differences)) == tuple(
                map(abs, gaps)
            )
            assert len(check.cosine_enclosures) == 10
            for gap, (lower, upper) in zip(gaps, check.cosine_enclosures):
                angle = abs(gap)
                assert angle < 3
                partial = sum(
                    Q((-1) ** j, factorial(2 * j)) * angle ** (2 * j) for j in range(12)
                )
                remainder = angle**24 / factorial(24)
                assert lower <= partial <= partial + remainder <= upper
                assert partial - lower < grid
                assert upper - partial - remainder < grid
            bounds = tuple(
                (Q(5) + 2 * sum(pair[i] for pair in check.cosine_enclosures)) / 25
                for i in (0, 1)
            )
            assert check.r_squared_interval == bounds
            expected = (
                _interval_status(
                    bounds,
                    obligation.lower_squared,
                    lower=True,
                    strict=obligation.lower_strict,
                ),
                _interval_status(
                    bounds,
                    obligation.upper_squared,
                    lower=False,
                    strict=obligation.upper_strict,
                ),
            )
            assert (check.lower_status, check.upper_status) == expected
            status = (
                "rejected"
                if "rejected" in expected
                else "unresolved" if "unresolved" in expected else "certified"
            )
            assert check.status == status


@pytest.mark.parametrize("lower,strict", tuple(product((False, True), repeat=2)))
def test_controller_equality_straddling_and_rejection_are_distinct(
    points, lower, strict
):
    original = points["stable", "stable", 0].controller_checks[0].obligation
    cut = Q(1, 2)
    obligation = replace(
        original,
        lower_squared=cut if lower else None,
        upper_squared=None if lower else cut,
        lower_strict=strict,
        upper_strict=strict,
    )
    for bounds in (
        (Q(0), Q(1, 4)),
        (cut, cut),
        (Q(3, 4), Q(1)),
        (Q(1, 4), Q(3, 4)),
        (Q(1, 4), cut),
        (cut, Q(3, 4)),
    ):
        expected = _interval_status(bounds, cut, lower=lower, strict=strict)
        left, right, status = point._controller_status(obligation, bounds)
        assert (left if lower else right) == expected
        assert (right if lower else left) == "not_required"
        assert status == expected
    with pytest.raises(ValueError, match="ordered"):
        point._controller_status(obligation, (Q(1), Q(0)))


def test_source_peak_ties_and_selected_rows_are_wired_without_a_new_source(
    monkeypatch, renewal_guard
):
    original_prepare = point._prepare
    mock_q = (Q(-2), Q(2), Q(0), Q(0), Q(0))

    def prepared(spec):
        values = list(original_prepare(spec))
        # A deliberately replaced successor field exercises canonical row
        # selection. It is not a law-consistent preparation or witness.
        values[4] = tuple(
            shared.AffineExpression((Q(0),) * 5, value) for value in mock_q
        )
        return tuple(values)

    monkeypatch.setattr(point, "_prepare", prepared)
    result = point.check_native_mutation_point(prior._spec(), PRESSURES[1])
    assert result.source_peak_owners == ((0, 1), (4, -1))
    assert result.successor_peak_owners == ((1, 1),)
    assert result.pressure_peaks == (1, 2)
    assert result.selected_cell_key == shared.CellKey((0, 1), (1, 1))
    fine = _fine(result, result.common_offset)
    fine["pressure"] = (fine["pressure"][0], prior._paired(mock_q))
    expected = prior._expected_residuals(
        prior._defaults(), fine, result.selected_cell_key
    )
    assert {row.label for row in result.row_checks} == set(expected)
    assert len(result.row_checks) > 60
    for row in result.row_checks:
        assert (row.residual, row.strict, row.unit) == expected[row.label]
    _assert_both_stability_masks(result, fine)


def _mock_components(monkeypatch, q0, controller_status, rate=Q(1, 4)):
    """Replace full-law components explicitly for admission wiring only."""
    original_prepare = point._prepare

    def prepared(spec):
        values = list(original_prepare(spec))
        values[4] = tuple(
            shared.AffineExpression((Q(0),) * 5, value)
            for value in (q0, -q0, Q(0), Q(0), Q(0))
        )
        values[5] = tuple(
            tuple(
                shared.AffineExpression((Q(0),) * 4 + (Q(1),), offset) for _ in range(5)
            )
            for offset in (Q(0), prior.DT * rate, prior.DT * rate)
        )
        values[-1] = ()
        return tuple(values)

    monkeypatch.setattr(point, "_prepare", prepared)
    monkeypatch.setattr(
        point,
        "_cell",
        lambda prepared, key: shared.CompiledCell(key, (), "mocked wiring only"),
    )
    monkeypatch.setattr(
        point,
        "_check_controller",
        lambda obligation: point.ControllerPointCheck(
            obligation,
            (),
            (Q(0), Q(1)),
            controller_status,
            "not_required",
            controller_status,
        ),
    )


@pytest.mark.parametrize("controller_status", ("certified", "rejected", "unresolved"))
@pytest.mark.parametrize("q0", (Q(-1, 2**120), Q(0), Q(1, 2**120)))
def test_strict_negative_pressure_and_controller_status_assembly_are_wiring_only(
    monkeypatch, renewal_guard, controller_status, q0
):
    _mock_components(monkeypatch, q0, controller_status)
    result = point.check_native_mutation_point(prior._spec(), PRESSURES[1])
    check = next(
        check
        for check in result.parameter_checks
        if check.label == "point.successor_pressure"
    )
    assert (check.lhs, check.rhs, check.relation) == (q0, 0, "<")
    assert check.satisfied == (q0 < 0)
    expected = (
        "rejected"
        if q0 >= 0 or controller_status == "rejected"
        else (
            "unresolved"
            if controller_status == "unresolved"
            else "conditionally_admitted_ideal"
        )
    )
    assert result.status == expected
    assert bool(result.unresolved_checks) == (controller_status == "unresolved")
    if q0 >= 0:
        assert "point.successor_pressure" in result.failed_checks
    # This test replaces physical rows and controllers deliberately. Its
    # success branch tests report assembly, not existence of a joint source.


@pytest.mark.parametrize("distance", (Q(-1, 2**120), Q(0), Q(1, 2**120)))
def test_secant_threshold_is_strict_and_consumes_the_samples(
    monkeypatch, renewal_guard, distance
):
    rate = prior._defaults()["xi"] + distance
    _mock_components(monkeypatch, Q(-1), "certified", rate)
    result = point.check_native_mutation_point(prior._spec(), PRESSURES[1])
    check = next(
        check
        for check in result.parameter_checks
        if check.label == "point.acquired_secant"
    )
    samples = result.temporal_histories[0]
    assert (
        result.first_secants[0]
        == (samples[1][1] - samples[0][1]) / (samples[1][0] - samples[0][0])
        == rate
    )
    # The intentionally replaced sample row differs from the unmodified
    # nodal product: the verifier must read its associated samples, never a
    # cached rate or separately supplied trigger observation.
    assert rate != result.spec.capacities[0] * result.law.oz_factor * result.pressure[0]
    assert (check.lhs, check.rhs, check.relation) == (result.law.xi, rate, "<")
    assert check.satisfied == (distance > 0)
    assert (result.status == "conditionally_admitted_ideal") == (distance > 0)


def test_actual_zero_successor_never_implies_negative_zero_mutation(renewal_guard):
    result = point.check_native_mutation_point(
        prior._spec(capacities=(Q(1),) * 5, phases=prior.ZERO), PRESSURES[0]
    )
    assert result.pressure == result.successor_pressure == prior.ZERO
    assert result.pressure_peaks == (0, 0)
    assert result.source_peak_owners == result.successor_peak_owners == ()
    assert result.selected_cell_key is None
    assert "point.successor_pressure" in result.failed_checks
    assert "point.successor_peak" in result.failed_checks
    assert result.status == "rejected"


def test_normalization_and_fresh_defaults_are_not_cached(points, monkeypatch):
    original = points["stable", "stable", 1]
    equivalent = prior._spec(
        capacities=list(map(float, prior.CAPACITIES)),
        phases=[float(value + Q(7, 4)) for value in prior.PHASES],
    )
    rebuilt = point.check_native_mutation_point(
        equivalent, list(map(float, PRESSURES[1]))
    )
    assert rebuilt == original and rebuilt.law is not original.law
    cfg = prior.CoreDefaults()
    cfg.GLYPH_FACTORS["ZHIR_theta_shift"] = 1
    monkeypatch.setattr(shared, "CoreDefaults", lambda: cfg)
    with pytest.raises(ValueError):
        point.check_native_mutation_point(original.spec, original.pressure_coordinates)


@pytest.mark.parametrize(
    "coordinates",
    [
        None,
        (),
        (0, 0, 0),
        (0, 0, 0, 0, 0),
        {0, 1, 2, 3},
        bytearray(4),
        (True, 0, 0, 0),
        (float("nan"), 0, 0, 0),
        (float("inf"), 0, 0, 0),
        (complex(1), 0, 0, 0),
    ],
)
def test_invalid_pressure_coordinates_reject_before_cosine_work(
    coordinates, monkeypatch, renewal_guard
):
    _, forbidden = renewal_guard
    monkeypatch.setattr(point, "certified_cosine_bounds", forbidden)
    with pytest.raises((TypeError, ValueError)):
        point.check_native_mutation_point(prior._spec(), coordinates)


@pytest.mark.parametrize(
    "change",
    [
        {"capacities": (True, 2, 3, 4, 5)},
        {"phases": (0, 0, float("nan"), 0, 0)},
        {"first_branch": "invented"},
    ],
)
def test_invalid_source_parameters_reject_before_cosine_work(
    change, monkeypatch, renewal_guard
):
    _, forbidden = renewal_guard
    monkeypatch.setattr(point, "certified_cosine_bounds", forbidden)
    with pytest.raises((TypeError, ValueError)):
        point.check_native_mutation_point(
            replace(prior._spec(), **change), PRESSURES[0]
        )


def test_reports_files_and_native_producers_cannot_supply_point_evidence(
    points, renewal_guard
):
    original = points["stable", "stable", 0]
    forged = replace(original, pressure=prior.ZERO, status="admitted")
    for value in (original, forged, {"spec": original.spec}):
        with pytest.raises(TypeError):
            point.check_native_mutation_point(value, PRESSURES[0])
    _, forbidden = renewal_guard
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(builtins, "open", forbidden)
        patch.setattr(Path, "open", forbidden)
        patch.setattr(Path, "read_text", forbidden)
        patch.setattr(Path, "read_bytes", forbidden)
        patch.setattr(prior.guards.domain.gain, "_source", forbidden)
        rebuilt = point.check_native_mutation_point(forged.spec, PRESSURES[0])
    assert rebuilt == original and rebuilt.law is not original.law
    with pytest.raises(FrozenInstanceError):
        rebuilt.status = "admitted"
    with pytest.raises(FrozenInstanceError):
        rebuilt.controller_checks[0].status = "certified"
