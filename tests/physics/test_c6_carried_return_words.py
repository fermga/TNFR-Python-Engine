"""Nodal-kernel and finite-integer oracles for guarded word repetition."""

import math
import random
from dataclasses import replace
from fractions import Fraction as F
from itertools import product

import pytest

from tnfr.dynamics._euler_kernel import NodalRemainderState, advance_nodal_remainder
from tnfr.physics import c6_carried_return as owner
from tnfr.physics.c6_carried_viability import _dbm_close
from tnfr.physics.c6_pressure_lattice import derive_c6_pressure_lattice

EPI = (0.5,) * 6
STATE = NodalRemainderState(EPI, (F(0),) * 6, 0.375, 0.625)
TIMESTEP = 2.0**-56


@pytest.fixture(scope="module")
def reference():
    return derive_c6_pressure_lattice(
        phase=(0.0, 0.1, 0.0, 0.0, 0.0, 0.0), epi_weight=1.0, phase_weight=1.0
    )


def derive(reference, **changes):
    args = dict(
        state=STATE,
        epi_states=(EPI,),
        timestep=TIMESTEP,
        transient_epi_states=(),
        word_edge_indices=(0,),
        max_intersections=1,
    )
    args.update(changes)
    return owner.derive_c6_carried_return_word_budget(reference, **args)


@pytest.fixture(scope="module")
def finite(reference):
    return derive(reference)


def test_nonzero_nodal_pressure_has_a_sharp_finite_cell_repetition_budget(finite):
    assert finite.return_envelope.relation_complete
    assert finite.return_envelope.status == "resource_limit"
    assert finite.status == "finite_repetition_budget" and finite.word_budget_certified
    assert finite.net_shift == (1, -2, 1, 0, 0, 0)
    assert finite.word_physical_steps == 1
    assert finite.maximum_repetitions == 188 and finite.first_empty_repetition == 189
    assert finite.origin_in_word_source
    assert finite.tested_repetitions[0] == (1, True)
    assert (188, True) in finite.tested_repetitions and (
        189,
        False,
    ) in finite.tested_repetitions
    assert finite.work_items == 2 + len(finite.tested_repetitions) - 1
    assert not finite.conditional_word_identity_certified
    assert (
        not finite.conditional_boundedness_certified
        and not finite.conditional_invariance_certified
    )
    assert (
        not finite.future_runtime_certified
        and not finite.asymptotic_convergence_certified
    )
    assert not finite.actual_origin_reachability_certified


def test_last_source_witness_repeats_the_actual_shared_nodal_kernel(finite):
    envelope = finite.return_envelope
    grid, point = envelope.grid_quantum, finite.maximum_repetition_source_point
    current = NodalRemainderState(EPI, tuple(grid * v for v in point), 0.375, 0.625)
    assert any(current.remainder)
    for ordinal in range(finite.maximum_repetitions):
        step = advance_nodal_remainder(
            current,
            timestep=TIMESTEP,
            capacity=(1.0,) * 6,
            pressure=envelope.pressures[0],
        )
        assert step.after.epi == EPI and not any(step.nodal_balance_residual)
        expected = tuple(
            grid * (point[i] + (ordinal + 1) * finite.net_shift[i]) for i in range(6)
        )
        assert step.after.remainder == expected
        current = step.after
    after_budget = advance_nodal_remainder(
        current,
        timestep=TIMESTEP,
        capacity=(1.0,) * 6,
        pressure=envelope.pressures[0],
    )
    assert after_budget.after.epi != EPI


def test_more_envelope_refinement_changes_the_declared_word_class(reference, finite):
    refined = derive(reference, max_intersections=2)
    assert refined.maximum_repetitions == 187
    assert (
        refined.return_envelope.retained_zones != finite.return_envelope.retained_zones
    )
    retained = refined.return_envelope.retained_zones[0].bounds
    assert all(
        refined.source_bounds[i][j] <= retained[i][j]
        for i in range(7)
        for j in range(7)
    )
    assert refined.source_bounds != retained
    assert not refined.conditional_boundedness_certified


@pytest.mark.parametrize("length", [2, 3, 7])
def test_grouping_same_edges_uses_complete_word_displacement(reference, finite, length):
    result = derive(reference, word_edge_indices=(0,) * length)
    assert result.maximum_repetitions == finite.maximum_repetitions // length
    assert result.word_physical_steps == length
    assert result.net_shift == tuple(length * v for v in finite.net_shift)
    assert result.first_empty_repetition == result.maximum_repetitions + 1


def test_empty_complete_word_does_not_require_a_terminal_first_guard(reference, finite):
    result = derive(
        reference, word_edge_indices=(0,) * (finite.maximum_repetitions + 1)
    )
    assert result.status == "empty_word" and result.word_budget_certified
    assert result.maximum_repetitions == 0 and result.first_empty_repetition == 1
    assert result.source_bounds is None and not result.origin_in_word_source
    assert result.tested_repetitions == ((1, False),)
    assert result.maximum_repetition_source_point is None


def test_zero_displacement_is_a_conditional_word_identity():
    reference = derive_c6_pressure_lattice(
        phase=(0.0,) * 6, epi_weight=1.0, phase_weight=1.0
    )
    result = derive(reference, word_edge_indices=(0,) * 3)
    assert result.status == "zero_shift_word_identity"
    assert result.word_budget_certified and result.conditional_word_identity_certified
    assert result.net_shift == (0,) * 6 and result.origin_in_word_source
    assert result.maximum_repetitions is None and result.first_empty_repetition is None
    assert (
        result.source_bounds is not None
        and result.pairwise_repetition_upper_bound is None
    )
    assert result.tested_repetitions == ((1, True),)
    assert (
        not result.future_runtime_certified
        and not result.conditional_boundedness_certified
    )


def test_incomplete_relation_supplies_no_word_certificate(reference):
    other = (math.nextafter(0.5, math.inf), 0.5, 0.5, 0.5, 0.5, 0.5)
    result = derive(reference, epi_states=(EPI, other))
    assert (
        result.status == "construction_resource_limit"
        and not result.word_budget_certified
    )
    assert not result.return_envelope.relation_complete
    assert result.word_edges == () and result.tested_repetitions == ()
    assert result.maximum_repetitions is None and result.source_bounds is None
    assert result.work_items == 0 and result.origin_in_word_source is None


@pytest.mark.parametrize("budget", [1, 2, 3, 4])
def test_composition_or_search_stop_never_publishes_a_maximum(reference, budget):
    result = derive(reference, max_work_items=budget)
    assert result.status == "word_resource_limit" and not result.word_budget_certified
    assert result.work_items == budget
    assert result.maximum_repetitions is None and result.first_empty_repetition is None
    assert (
        result.maximum_repetition_source_bounds is None
        and result.maximum_repetition_source_point is None
    )
    if budget == 1:
        assert result.source_bounds is None and result.origin_in_word_source is None
    else:
        assert result.source_bounds is not None and result.tested_repetitions[0] == (
            1,
            True,
        )


def test_exact_work_boundary_completes_without_a_further_budget_charge(
    reference, finite
):
    exact = derive(reference, max_work_items=finite.work_items)
    assert (
        exact.status == finite.status
        and exact.maximum_repetitions == finite.maximum_repetitions
    )
    assert exact.work_items == exact.max_work_items


def test_origin_membership_uses_rn_row_as_well_as_zero_coordinates(reference):
    other = (math.nextafter(0.5, math.inf), 0.5, 0.5, 0.5, 0.5, 0.5)
    envelope = owner.derive_c6_carried_return_envelope(
        reference,
        state=STATE,
        epi_states=(EPI, other),
        timestep=TIMESTEP,
        transient_epi_states=(),
        max_intersections=4,
    )
    index = next(
        i
        for i, edge in enumerate(envelope.return_relation)
        if edge.source_epi == edge.target_epi == other
    )
    result = derive(
        reference,
        epi_states=(EPI, other),
        max_intersections=4,
        word_edge_indices=(index,),
    )
    assert result.word_budget_certified and not result.origin_in_word_source


def test_nonclosed_and_noncomposable_word_labels_are_rejected(reference):
    other = (math.nextafter(0.5, math.inf), 0.5, 0.5, 0.5, 0.5, 0.5)
    envelope = owner.derive_c6_carried_return_envelope(
        reference,
        state=STATE,
        epi_states=(EPI, other),
        timestep=TIMESTEP,
        transient_epi_states=(),
        max_intersections=4,
    )
    index = next(
        i
        for i, edge in enumerate(envelope.return_relation)
        if edge.source_epi != edge.target_epi
    )
    with pytest.raises(ValueError, match="composable and closed"):
        derive(
            reference,
            epi_states=(EPI, other),
            max_intersections=4,
            word_edge_indices=(index,),
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"word_edge_indices": ()},
        {"word_edge_indices": []},
        {"word_edge_indices": (True,)},
        {"word_edge_indices": (-1,)},
        {"word_edge_indices": (F(0),)},
        {"word_edge_indices": (1,)},
        {"word_edge_indices": (0, 0), "max_word_length": 1},
        {"max_word_length": 0},
        {"max_word_length": True},
        {"max_work_items": 0},
        {"max_work_items": 1.5},
    ],
)
def test_invalid_indices_and_resource_guards_are_rejected(reference, changes):
    with pytest.raises((TypeError, ValueError)):
        derive(reference, **changes)


def test_forged_pressure_reference_is_rebuilt_before_word_selection(reference, finite):
    forged = replace(
        reference, sources=(44.0,) * 6, rows=(), epi_quantum=F(5), gradient_quantum=F(7)
    )
    result = derive(forged)
    assert result.return_envelope.reference == reference
    assert (
        result.word_edges == finite.word_edges and result.net_shift == finite.net_shift
    )
    assert (
        result.source_bounds == finite.source_bounds
        and result.maximum_repetitions == 188
    )


def contains(bounds, point):
    return bounds is not None and all(
        point[i] - point[j] <= bounds[i][j] for i in range(7) for j in range(7)
    )


@pytest.mark.parametrize("displacement", tuple(product((-1, 0, 1), repeat=2)))
def test_repeat_formula_matches_brute_force_integer_trajectories(displacement):
    rng = random.Random(20260917)
    universe = [tuple(p) + (0,) * 5 for p in product(range(-2, 3), repeat=2)]
    shift = tuple(displacement) + (0,) * 4
    for _ in range(12):
        points = rng.sample(universe, rng.randint(1, len(universe)))
        bounds = tuple(
            tuple(max(p[i] - p[j] for p in points) for j in range(7)) for i in range(7)
        )
        assert _dbm_close(bounds) == bounds
        for count in range(1, 7):
            repeated = owner._repeated_return_word_source(bounds, shift, count)
            exact = {
                point
                for point in universe
                if all(
                    contains(
                        bounds,
                        tuple(point[i] + ordinal * ((*shift, 0)[i]) for i in range(7)),
                    )
                    for ordinal in range(count)
                )
            }
            represented = {point for point in universe if contains(repeated, point)}
            assert represented == exact
            assert (repeated is None) == (not exact)


def test_joint_closure_can_be_stronger_than_every_individual_pair_width():
    points = [
        p + (0,) * 3
        for p in product(range(3), repeat=4)
        if p[0] <= p[1] <= p[2] <= p[3]
    ]
    bounds = tuple(
        tuple(max(p[i] - p[j] for p in points) for j in range(7)) for i in range(7)
    )
    shift = (1, 0, 1, 0, 0, 0)
    vector = (*shift, 0)
    upper = 1 + min(
        (bounds[i][j] + bounds[j][i]) // abs(vector[i] - vector[j])
        for i in range(7)
        for j in range(i)
        if vector[i] != vector[j]
    )
    assert upper == 3
    assert owner._repeated_return_word_source(bounds, shift, 2) is not None
    assert owner._repeated_return_word_source(bounds, shift, 3) is None
