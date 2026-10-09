"""Independent finite-state oracles for exact carried viability certificates."""

from dataclasses import replace
from fractions import Fraction as F
from itertools import product

import pytest

from tnfr.dynamics._euler_kernel import NodalRemainderState, advance_nodal_remainder
from tnfr.physics.c6_carried_viability import (
    C6CarriedViabilityBox,
    derive_c6_carried_viability,
)
from tnfr.physics.c6_pressure_lattice import (
    _observe_rebuilt_c6_pressure_lattice,
    derive_c6_pressure_lattice,
)

DELTA = F(1, 2**54)
LOW, HIGH = float(F(1, 2) - 4 * DELTA), float(F(1, 2) - 3 * DELTA)
FIRST = (LOW, HIGH, LOW, HIGH, LOW, HIGH)
SECOND = (HIGH, LOW, HIGH, LOW, HIGH, LOW)
THIRD = (HIGH, HIGH, LOW, HIGH, LOW, HIGH)
SPACINGS = (DELTA, DELTA / 2, DELTA, DELTA, DELTA, DELTA / 2)


@pytest.fixture(scope="module")
def reference():
    # Zero phases make the phase contribution vanish without modifying the
    # positive represented coefficient accepted by the canonical owner.
    return derive_c6_pressure_lattice(
        phase=(0.0,) * 6, epi_weight=1.0, phase_weight=1.0
    )


def _state(epi, first=DELTA / 4, fifth=DELTA / 4):
    return NodalRemainderState(
        epi, (F(0), first, F(0), F(0), F(0), fifth), 0.375, 0.625
    )


def _step(reference, state):
    return advance_nodal_remainder(
        state,
        timestep=1.0,
        capacity=(1.0,) * 6,
        pressure=_observe_rebuilt_c6_pressure_lattice(reference, state.epi).pressure,
    )


def _box(epi):
    return C6CarriedViabilityBox(
        epi,
        tuple(F(x) - (DELTA / 4 if i in (1, 5) else 0) for i, x in enumerate(epi)),
        tuple(F(x) + (DELTA / 4 if i in (1, 5) else 0) for i, x in enumerate(epi)),
    )


def _singleton(state):
    return C6CarriedViabilityBox(state.epi, state.exact_epi, state.exact_epi)


def _enumerate_boxes(result):
    """Enumerate the tiny declared domains; never call the viability descent."""
    points = set()
    for box in result.retained_boxes:
        coordinates = []
        for lower, upper, origin, spacing in zip(
            box.lower,
            box.upper,
            result.affine_origin,
            result.coordinate_spacings,
            strict=True,
        ):
            first = ((lower - origin) / spacing).__ceil__()
            last = ((upper - origin) / spacing).__floor__()
            coordinates.append(
                tuple(origin + spacing * index for index in range(first, last + 1))
            )
        for exact in product(*coordinates):
            assert (
                exact not in points
            ), "a returned partition double-counts one exact state"
            points.add(exact)
            state = NodalRemainderState(
                box.epi, tuple(x - F(y) for x, y in zip(exact, box.epi)), 0.375, 0.625
            )
            assert state.exact_epi == exact
    return points


def _eight_states():
    return tuple(
        _state(row, a, b)
        for row in (FIRST, SECOND)
        for a, b in product((-DELTA / 4, DELTA / 4), repeat=2)
    )


def test_nonzero_carry_fixed_point_matches_independent_eight_state_permutation(
    reference,
):
    states = _eight_states()
    points = {state.exact_epi for state in states}
    assert len(points) == 8
    assert {_step(reference, state).after.exact_epi for state in states} == points
    assert all(
        _step(reference, _step(reference, state).after).after == state
        for state in states
    )
    result = derive_c6_carried_viability(
        reference,
        state=_state(FIRST),
        epi_states=(FIRST, SECOND, THIRD),
        timestep=1.0,
        initial_boxes=(_box(FIRST), _box(SECOND)),
    )
    assert result.status == "fixed_point" and result.conditional_boundedness_certified
    assert result.coordinate_spacings == SPACINGS
    assert tuple(record.point_count for record in result.iterations) == (8, 8)
    assert _enumerate_boxes(result) == points
    assert not result.asymptotic_convergence_certified


def test_off_coset_geometric_overlap_does_not_duplicate_or_discard_states(reference):
    original = _box(FIRST)
    split = F(FIRST[1])
    left = replace(original, upper=original.upper[:1] + (split,) + original.upper[2:])
    right = replace(original, lower=original.lower[:1] + (split,) + original.lower[2:])
    # The shared real hyperplane is not a member of the incoming affine coset.
    assert ((split - _state(FIRST).exact_epi[1]) / SPACINGS[1]).denominator == 2
    result = derive_c6_carried_viability(
        reference,
        state=_state(FIRST),
        epi_states=(FIRST, SECOND, THIRD),
        timestep=1.0,
        initial_boxes=(left, right, _box(SECOND)),
    )
    assert result.status == "fixed_point"
    assert _enumerate_boxes(result) == {state.exact_epi for state in _eight_states()}
    assert all(record.point_count == 8 for record in result.iterations)


def test_actual_coset_overlap_is_rejected_before_cardinality_can_certify_it(reference):
    original = _box(FIRST)
    split = _state(FIRST).exact_epi[1]
    left = replace(original, upper=original.upper[:1] + (split,) + original.upper[2:])
    right = replace(original, lower=original.lower[:1] + (split,) + original.lower[2:])
    with pytest.raises(ValueError, match="disjoint"):
        derive_c6_carried_viability(
            reference,
            state=_state(FIRST),
            epi_states=(FIRST, SECOND, THIRD),
            timestep=1.0,
            initial_boxes=(left, right, _box(SECOND)),
        )


def test_visible_two_step_return_with_wrong_carry_is_excluded_at_exact_depth_two(
    reference,
):
    initial = _state(THIRD)
    middle = _step(reference, initial).after
    final = _step(reference, middle).after
    assert final.epi == initial.epi
    assert final.exact_epi != initial.exact_epi
    result = derive_c6_carried_viability(
        reference,
        state=initial,
        epi_states=(initial.epi, middle.epi),
        timestep=1.0,
        initial_boxes=(_singleton(initial), _singleton(middle)),
    )
    assert result.status == "origin_excluded" and result.exclusion_step_bound == 2
    assert tuple(record.point_count for record in result.iterations) == (2, 1, 0)
    assert tuple(record.origin_retained for record in result.iterations) == (
        True,
        True,
        False,
    )
    assert not result.conditional_invariance_certified


def test_outgoing_odd_target_tie_cannot_be_treated_as_a_closed_cell(reference):
    initial = NodalRemainderState(FIRST, (DELTA / 2,) + (F(0),) * 5, 0.375, 0.625)
    assert initial.exact_epi[0] == (F(LOW) + F(HIGH)) / 2
    step = _step(reference, initial)
    # LOW is even; HIGH is odd. The translated tie rounds past HIGH.
    assert step.after.epi[0] == float(F(HIGH) + DELTA)
    result = derive_c6_carried_viability(
        reference,
        state=initial,
        epi_states=(FIRST, SECOND),
        timestep=1.0,
        initial_boxes=(_singleton(initial),),
    )
    assert result.origin_exit_certified and result.exclusion_step_bound == 1
    assert result.retained_boxes == ()


def test_exact_band_failure_is_rejected_even_when_rounding_would_hide_it(reference):
    initial = NodalRemainderState(
        FIRST,
        (F(0), -DELTA / 4, F(0), F(0), F(0), F(0)),
        LOW,
        HIGH,
    )
    candidate = initial.exact_epi[1] - DELTA
    assert candidate < F(LOW) and float(candidate) == LOW
    with pytest.raises(ValueError):
        _step(reference, initial)
    result = derive_c6_carried_viability(
        reference,
        state=initial,
        epi_states=(FIRST, SECOND),
        timestep=1.0,
        initial_boxes=(_singleton(initial),),
    )
    assert result.origin_exit_certified and result.exclusion_step_bound == 1
    assert not result.conditional_boundedness_certified


def test_incomplete_second_descent_retains_last_complete_partition_without_fixed_point(
    reference,
):
    inputs = dict(
        state=_state(FIRST),
        epi_states=(FIRST, SECOND, THIRD),
        timestep=1.0,
        initial_boxes=(_box(FIRST), _box(SECOND), _singleton(_state(THIRD))),
    )
    complete = derive_c6_carried_viability(reference, **inputs)
    assert complete.status == "fixed_point"
    assert tuple(record.point_count for record in complete.iterations) == (9, 8, 8)
    interrupted = derive_c6_carried_viability(
        reference, **inputs, max_work_items=complete.work_items - 1
    )
    assert interrupted.status == "resource_limit"
    assert tuple(record.point_count for record in interrupted.iterations) == (9, 8)
    assert _enumerate_boxes(interrupted) == {
        state.exact_epi for state in _eight_states()
    }
    assert not interrupted.conditional_invariance_certified
    assert (
        not interrupted.origin_exit_certified
        and interrupted.exclusion_step_bound is None
    )


def test_forged_reference_caches_cannot_change_the_independent_finite_map(reference):
    forged = replace(
        reference,
        sources=(99.0,) * 6,
        rows=(),
        epi_quantum=F(7),
        gradient_quantum=F(11),
        every_pressure_has_positive=True,
        every_pressure_has_negative=True,
        no_cartesian_trap=True,
    )
    result = derive_c6_carried_viability(
        forged,
        state=_state(FIRST),
        epi_states=(FIRST, SECOND, THIRD),
        timestep=1.0,
        initial_boxes=(_box(FIRST), _box(SECOND)),
    )
    assert result.reference == reference
    assert result.status == "fixed_point"
    assert _enumerate_boxes(result) == {state.exact_epi for state in _eight_states()}


def test_invalid_primitive_phase_is_not_replaced_by_a_previously_valid_cache(reference):
    forged = replace(
        reference, source=replace(reference.source, phase=(float("nan"),) * 6)
    )
    with pytest.raises(ValueError):
        derive_c6_carried_viability(
            forged,
            state=_state(FIRST),
            epi_states=(FIRST, SECOND, THIRD),
            timestep=1.0,
            initial_boxes=(_box(FIRST), _box(SECOND)),
        )
