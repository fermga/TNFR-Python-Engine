"""Exact mask projections, independent of joint source feasibility.

The named scalar boxes below are normalized-pressure algebra controls, not
capacity/phase sources. They retain source strictness and successor signed-zero
scope. Wiring uses only the already frozen coefficient frame and its two
symmetries. No solver, native response or source search is evaluated.
"""

import builtins
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q
from itertools import product
from pathlib import Path

import pytest

from tests.research import test_native_mutation_feasibility as prior
from tnfr.research import _native_mutation_feasibility as compiler

no_execution = prior.no_execution
calculus_guard = prior.calculus_guard
renewal_guard = prior.renewal_guard
compiled = prior.compiled


def _interval_exists(lower, upper, lower_open):
    return lower < upper or (lower == upper and not lower_open)


def _sum_interval_test(boundary, peak):
    """Intersect t with the sum interval of the remaining pressure ratios."""
    budgets = boundary.surrounding_budgets
    target, cut = boundary.target_ratio_lower, boundary.pressure_cut
    if any(value < 0 for value in budgets):
        return False
    if peak is None:
        return boundary.stability_bases[0] <= prior._defaults()["lo"]
    block, sign = peak
    if boundary.boundary == "source":
        lower, upper, strict_cut = max(target, cut), Q(1), True
        if block == 0:
            lower, upper = max(lower, Q(1)), min(upper, Q(1))
            return sum(budgets) >= 1 and _interval_exists(lower, upper, lower == cut)
        residual_center = -sign
    else:
        lower, upper, strict_cut = target, cut, False
        residual_center = sign
    if budgets[block - 1] < 1:
        return False
    radius = sum(value for i, value in enumerate(budgets, start=1) if i != block)
    lower = max(lower, residual_center - radius)
    upper = min(upper, residual_center + radius)
    return _interval_exists(lower, upper, strict_cut and lower == cut)


def _boundary(law, name, target, budgets):
    gamma = law.stability[2]
    bases = (law.si_low + gamma * target,) + tuple(
        law.si_high + gamma * value for value in budgets
    )
    return compiler._mask_boundary(law, bases, name)


def _owner(boundary, peak):
    return next(owner for owner in boundary.owners if owner.peak == peak)


def _compatible(boundary, peak):
    return _owner(boundary, peak).status == "projection_compatible"


def _independent_bases(spec, phases):
    alpha, beta, gamma = prior._defaults()["stability"]
    capacities = tuple(Q(value) for value in spec.capacities)
    fine_capacity, fine_phase = prior._paired(capacities), prior._paired(phases)
    values = tuple(
        alpha * fine_capacity[n] / max(fine_capacity)
        + beta
        * (
            1
            - abs(sum(fine_phase[j] for j in prior._neighbors(n)) / 4 - fine_phase[n])
            / prior.PI
        )
        + gamma
        for n in range(10)
    )
    return values[::2]


@pytest.fixture(scope="module")
def projections(renewal_guard):
    return {
        (first, second): compiler.project_native_mutation_masks(
            prior._spec(first, second)
        )
        for first, second in product(prior.BRANCHES, repeat=2)
    }


@pytest.fixture(scope="module")
def algebraic_boxes(projections):
    """Frozen finite endpoint cases; these are not physical parameter slices."""
    law = projections["stable", "stable"].law
    c = law.pressure_cut
    gap = (1 - c) / 4
    boxes = (
        ("all_zero", Q(0), (Q(0),) * 4),
        ("target_sum_equal", Q(1), (Q(1, 4),) * 4),
        ("target_sum_short", Q(1), (Q(1, 8), Q(1, 4), Q(1, 4), Q(1, 4))),
        ("source_ratio_equal", Q(0), (Q(1), Q(1), c, Q(0))),
        ("source_upper_equal", c + gap, (Q(1), Q(1), c + gap, Q(0))),
        ("source_upper_short", c + 2 * gap, (Q(1), Q(1), c + gap, Q(0))),
        ("source_opposite_tie", Q(1), (Q(1), Q(0), Q(0), Q(0))),
        ("next_positive_equal", c, (Q(1), 1 - c, Q(0), Q(0))),
        ("next_positive_short", c, (Q(1), 1 - c - gap, Q(0), Q(0))),
        ("next_negative_equal", c, (Q(1), Q(1), c, Q(0))),
        ("next_negative_zero_target", Q(0), (Q(1), Q(1), Q(0), Q(0))),
        ("target_ratio_above", c + gap, (Q(1),) * 4),
        ("target_ratio_impossible", Q(1) + gap, (Q(1),) * 4),
        ("negative_budget", Q(0), (Q(1), Q(1), Q(1), -gap)),
        ("peak_unavailable", Q(0), (Q(1) - gap,) * 4),
        ("budget_clipped_above", Q(0), (Q(2), Q(1), Q(1), Q(1))),
    )
    return law, {
        (boundary, name): _boundary(law, boundary, target, budgets)
        for boundary in ("source", "successor")
        for name, target, budgets in boxes
    }


def test_six_projection_cases_equal_independent_sum_interval_elimination(
    algebraic_boxes,
):
    _, boxes = algebraic_boxes
    seen = set()
    for boundary in boxes.values():
        for owner in boundary.owners:
            seen.add(owner.case)
            expected = _sum_interval_test(boundary, owner.peak)
            assert (owner.status == "projection_compatible") == expected
            assert all(check.relation in ("<", "<=") for check in owner.conditions)
        assert (boundary.status == "projection_compatible") == any(
            _sum_interval_test(boundary, owner.peak) for owner in boundary.owners
        )
    assert seen == {
        "source_target",
        "source_positive",
        "source_negative",
        "successor_positive",
        "successor_negative",
        "successor_zero",
    }


def test_strict_source_cut_and_weak_successor_endpoints_are_distinct(algebraic_boxes):
    _, boxes = algebraic_boxes
    assert not _compatible(boxes["source", "source_ratio_equal"], (1, 1))
    assert _compatible(boxes["source", "source_upper_equal"], (1, 1))
    assert not _compatible(boxes["source", "source_upper_short"], (1, 1))
    assert _compatible(boxes["source", "source_opposite_tie"], (1, -1))
    assert _compatible(boxes["source", "target_sum_equal"], (0, 1))
    assert not _compatible(boxes["source", "target_sum_short"], (0, 1))
    assert _compatible(boxes["successor", "next_positive_equal"], (1, 1))
    assert not _compatible(boxes["successor", "next_positive_short"], (1, 1))
    assert _compatible(boxes["successor", "next_negative_equal"], (1, -1))
    assert _compatible(boxes["successor", "next_negative_zero_target"], (1, -1))
    assert not _compatible(boxes["successor", "target_ratio_above"], (1, 1))
    assert boxes["source", "target_ratio_impossible"].status == "excluded"
    strict = [
        check
        for check in _owner(boxes["source", "source_ratio_equal"], (1, 1)).conditions
        if check.relation == "<"
    ]
    assert len(strict) == 1 and strict[0].lhs == strict[0].rhs


def test_zero_peak_and_negative_budgets_retain_the_actual_masks(algebraic_boxes):
    law, boxes = algebraic_boxes
    assert _compatible(boxes["successor", "all_zero"], None)
    assert not _compatible(boxes["successor", "next_positive_equal"], None)
    for boundary in ("source", "successor"):
        negative = boxes[boundary, "negative_budget"]
        assert negative.surrounding_budgets[-1] < 0
        assert all(owner.status == "excluded" for owner in negative.owners)
    no_peak = boxes["successor", "peak_unavailable"]
    assert all(
        owner.status == "excluded" for owner in no_peak.owners if owner.peak is not None
    )
    assert _compatible(no_peak, None)
    assert boxes["source", "budget_clipped_above"].surrounding_budgets[0] == 1
    # Negative raw target lower bounds clamp at zero; unlike a negative
    # surrounding budget, this does not reject the pressure-free mask.
    bases = (law.si_low - law.stability[2],) + (law.si_high,) * 4
    below = compiler._mask_boundary(law, bases, "successor")
    assert below.target_ratio_lower == 0 and _compatible(below, None)


def test_parameter_projection_rebuilds_both_boundaries_and_keeps_obligations(
    projections, compiled
):
    for pair, result in projections.items():
        full = compiled[pair]
        assert result.spec == full.spec
        assert result.law == full.law and result.law is not full.law
        assert result.premises == full.premises
        assert result.phase_geometry == full.phase_geometry
        assert result.controller_obligations == full.controller_obligations
        assert all(
            item.status == "unevaluated" for item in result.controller_obligations
        )
        assert result.source.stability_bases == _independent_bases(
            result.spec, result.spec.phases
        )
        assert result.successor.stability_bases == _independent_bases(
            result.spec, full.first_phase
        )
        assert (
            tuple(owner.peak for owner in result.source.owners) == prior.SOURCE_OWNERS
        )
        assert tuple(owner.peak for owner in result.successor.owners) == (
            *prior.NEXT_OWNERS,
            None,
        )
        for boundary in (result.source, result.successor):
            gamma = result.law.stability[2]
            assert boundary.surrounding_budgets == tuple(
                min(Q(1), (s - result.law.si_high) / gamma)
                for s in boundary.stability_bases[1:]
            )
            assert boundary.target_ratio_lower == max(
                Q(0), (boundary.stability_bases[0] - result.law.si_low) / gamma
            )
            assert boundary.pressure_cut == result.law.pressure_cut
        assert result.status in ("excluded", "projection_compatible")
        assert result.scope == "separate normalized-mask/peak/zero-sum subsystems only"
        assert (result.status == "projection_compatible") == (
            result.source.status == result.successor.status == "projection_compatible"
        )
        # These coefficient frames have not been admitted as joint sources.
        # Passing a projection cannot overwrite any geometry or R obligation.
        assert len(full.cells) == 81


def _assert_affine_row(row, terms):
    coefficients = tuple(
        sum(factor * expression.coefficients[j] for factor, expression in terms)
        for j in range(5)
    )
    constant = sum(factor * expression.constant for factor, expression in terms)
    assert row.coefficients == coefficients and row.rhs == -constant


def test_mask_terms_match_every_canonical_full_cell_without_removing_rows(
    projections, compiled
):
    for pair, result in projections.items():
        full = compiled[pair]
        gamma = result.law.stability[2]
        for cell in full.cells:
            rows = {row.label: row for row in cell.rows}
            source_block, source_sign = cell.key.source_peak
            peak0 = full.pressure[source_block]
            _assert_affine_row(
                rows["source.target.si_low"],
                (
                    (
                        source_sign
                        * (result.source.stability_bases[0] - result.law.si_low),
                        peak0,
                    ),
                    (-gamma, full.pressure[0]),
                ),
            )
            for i in range(1, 5):
                for sign, name in ((1, "positive"), (-1, "negative")):
                    _assert_affine_row(
                        rows[f"source.il.b{i}.{name}"],
                        (
                            (sign * gamma, full.pressure[i]),
                            (
                                source_sign
                                * (
                                    result.law.si_high
                                    - result.source.stability_bases[i]
                                ),
                                peak0,
                            ),
                        ),
                    )
            if cell.key.successor_peak is None:
                zero = rows["successor.zero.target.si_low"]
                assert zero.coefficients == prior.ZERO
                assert (
                    zero.rhs == result.law.si_low - result.successor.stability_bases[0]
                )
                for i in range(1, 5):
                    assert (
                        rows[f"successor.zero.il.b{i}"].rhs
                        == result.successor.stability_bases[i] - result.law.si_high
                    )
            else:
                successor_block, successor_sign = cell.key.successor_peak
                peak1 = full.successor_pressure[successor_block]
                _assert_affine_row(
                    rows["successor.target.si_low"],
                    (
                        (
                            successor_sign
                            * (result.successor.stability_bases[0] - result.law.si_low),
                            peak1,
                        ),
                        (gamma, full.successor_pressure[0]),
                    ),
                )
                for i in range(1, 5):
                    for sign, name in ((1, "positive"), (-1, "negative")):
                        _assert_affine_row(
                            rows[f"successor.il.b{i}.{name}"],
                            (
                                (sign * gamma, full.successor_pressure[i]),
                                (
                                    successor_sign
                                    * (
                                        result.law.si_high
                                        - result.successor.stability_bases[i]
                                    ),
                                    peak1,
                                ),
                            ),
                        )


def test_translation_and_reflection_reindex_the_same_projection(projections):
    original = projections["stable", "stable"]
    translated = compiler.project_native_mutation_masks(
        prior._spec(
            capacities=list(map(float, prior.CAPACITIES)),
            phases=[float(value + Q(7, 4)) for value in prior.PHASES],
        )
    )
    assert translated == original
    order = (0, 4, 3, 2, 1)
    reflected = compiler.project_native_mutation_masks(
        prior._spec(
            capacities=tuple(prior.CAPACITIES[i] for i in order),
            phases=tuple(prior.PHASES[i] for i in order),
        )
    )
    for left, right in (
        (original.source, reflected.source),
        (original.successor, reflected.successor),
    ):
        assert right.stability_bases == tuple(left.stability_bases[i] for i in order)
        assert right.surrounding_budgets == tuple(reversed(left.surrounding_budgets))
        assert right.target_ratio_lower == left.target_ratio_lower
        for owner in left.owners:
            mapped = (
                None if owner.peak is None else (order[owner.peak[0]], owner.peak[1])
            )
            assert _owner(right, mapped).status == owner.status


@pytest.mark.parametrize(
    "change",
    [
        {"capacities": (True, 2, 3, 4, 5)},
        {"capacities": (0, 2, 3, 4, 5)},
        {"capacities": (float("nan"), 2, 3, 4, 5)},
        {"capacities": {1, 2, 3, 4, 5}},
        {"phases": (0, 1, 2, 3)},
        {"phases": (0, 0, float("inf"), 0, 0)},
        {"first_branch": "assumed"},
    ],
)
def test_projection_readmits_raw_primitives(change, renewal_guard):
    with pytest.raises((TypeError, ValueError)):
        compiler.project_native_mutation_masks(replace(prior._spec(), **change))


def test_projection_cannot_consume_a_caller_report_or_artifact(
    projections, renewal_guard
):
    original = projections["stable", "stable"]
    fake = replace(original, status="feasible")
    for value in (original, fake, {"spec": original.spec}):
        with pytest.raises(TypeError):
            compiler.project_native_mutation_masks(value)
    _, forbidden = renewal_guard
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(builtins, "open", forbidden)
        patch.setattr(Path, "open", forbidden)
        patch.setattr(Path, "read_text", forbidden)
        patch.setattr(Path, "read_bytes", forbidden)
        patch.setattr(prior.guards.domain.gain, "_source", forbidden)
        rebuilt = compiler.project_native_mutation_masks(fake.spec)
    assert rebuilt == original and rebuilt.law is not original.law
    with pytest.raises(FrozenInstanceError):
        rebuilt.status = "feasible"
    with pytest.raises(FrozenInstanceError):
        rebuilt.source.owners[0].status = "feasible"
