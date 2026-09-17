"""Universal exact-box coverage has positive, negative and undecided outcomes."""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as F
from itertools import product
import math

import pytest

from tnfr.dynamics._euler_kernel import (
    NodalRemainderState, advance_nodal_remainder, initialize_nodal_remainder,
)
from tnfr.physics.c6_carried_viability import (
    C6CarriedViabilityBox, derive_c6_carried_viability,
)
from tnfr.physics.c6_pressure_lattice import (
    derive_c6_pressure_lattice, _observe_rebuilt_c6_pressure_lattice,
)
from tnfr.physics.nodal_remainder import derive_nodal_remainder_itinerary

GRID = F(1, 2**3222)


@pytest.fixture(scope="module")
def reference():
    return derive_c6_pressure_lattice(phase=(0.,) * 6, epi_weight=1., phase_weight=1.)


def _state(row, carry=F(0)):
    return NodalRemainderState(row, (carry,) * 6, .375, .625)


def _box(row, exact=None):
    values = tuple(map(F, row)) if exact is None else exact
    return C6CarriedViabilityBox(row, values, values)


def _advance(reference, state):
    pressure = _observe_rebuilt_c6_pressure_lattice(reference, state.epi).pressure
    return advance_nodal_remainder(state, timestep=1., capacity=(1.,) * 6, pressure=pressure)


def _swap_rows():
    first = (.4375, .5625) * 3
    return first, first[::-1]


def _transient_candidate(reference):
    rows = _swap_rows() + ((.4375, .4375, .4375, .4375, .4375, .5625),)
    return derive_c6_carried_viability(
        reference, state=_state(rows[0]), epi_states=rows, timestep=1.,
        initial_boxes=tuple(_box(row) for row in rows),
    )


def test_zero_pressure_whole_rn_cell_is_an_exact_conditional_fixed_point(reference):
    row = (.5,) * 6
    state = _state(row)
    result = derive_c6_carried_viability(reference, state=state, epi_states=(row,), timestep=1.)
    assert result.status == "fixed_point"
    assert result.conditional_invariance_certified and result.conditional_boundedness_certified
    assert not result.origin_exit_certified and result.exclusion_step_bound is None
    assert result.coordinate_spacings == (GRID,) * 6
    assert result.pressures == ((0.,) * 6,)
    assert result.iterations[0].point_count == result.iterations[1].point_count
    assert result.iterations[1].origin_retained and len(result.retained_boxes) == 1
    assert result.state == state and result.affine_origin == state.exact_epi
    assert not result.future_runtime_certified and not result.asymptotic_convergence_certified


def test_nonzero_canonical_two_cycle_proves_invariance_without_claiming_convergence(reference):
    rows = _swap_rows()
    state = _state(rows[0], GRID)
    result = derive_c6_carried_viability(reference, state=state, epi_states=rows, timestep=1.)
    assert result.status == "fixed_point" and result.iterations[-1].point_count == 2
    assert result.coordinate_spacings == (F(1, 8),) * 6
    first = _advance(reference, state)
    second = _advance(reference, first.after)
    assert first.after.epi == rows[1] and second.after == state
    assert first.after.remainder == state.remainder == (GRID,) * 6
    assert first.pressure != (0.,) * 6 and second.pressure != first.pressure
    expected = {state.exact_epi, first.after.exact_epi}
    assert {box.lower for box in result.retained_boxes} == expected
    assert all(box.lower == box.upper for box in result.retained_boxes)
    assert not result.asymptotic_convergence_certified


def test_exact_descending_partition_can_remove_transients_then_certify_a_cycle(reference):
    result = _transient_candidate(reference)
    assert result.status == "fixed_point"
    assert tuple(record.point_count for record in result.iterations) == (3, 2, 2)
    assert tuple(record.box_count for record in result.iterations) == (3, 2, 2)
    assert {box.epi for box in result.retained_boxes} == set(_swap_rows())
    assert all(record.origin_retained for record in result.iterations)


def test_two_descents_exclude_origin_with_a_verified_two_step_witness(reference):
    origin = _state((.5, .5, .5, .5, .5, .625))
    first = _advance(reference, origin)
    second = _advance(reference, first.after)
    rows = (origin.epi, first.after.epi)
    result = derive_c6_carried_viability(
        reference, state=origin, epi_states=rows, timestep=1.,
        initial_boxes=(_box(origin.epi), _box(first.after.epi)),
    )
    assert result.status == "origin_excluded" and result.origin_exit_certified
    assert result.exclusion_step_bound == 2
    assert tuple(record.point_count for record in result.iterations) == (2, 1, 0)
    assert tuple(record.origin_retained for record in result.iterations) == (True, True, False)
    assert result.retained_boxes == () and second.after.epi not in rows
    assert not result.conditional_invariance_certified


def test_one_step_leaving_the_declared_band_is_also_an_exact_exclusion(reference):
    rows = _swap_rows()
    result = derive_c6_carried_viability(reference, state=_state(rows[0]), epi_states=rows, timestep=4.)
    assert result.status == "origin_excluded" and result.exclusion_step_bound == 1
    pressure = result.pressures[0]
    candidate = tuple(value + 4 * F(p) for value, p in zip(result.state.exact_epi, pressure))
    assert any(not F(.375) <= value <= F(.625) for value in candidate)


def test_limit_during_second_descent_exposes_only_the_first_complete_partition(reference):
    complete = _transient_candidate(reference)
    limited = derive_c6_carried_viability(
        reference, state=complete.state, epi_states=complete.epi_states, timestep=1.,
        initial_boxes=complete.initial_boxes, max_work_items=complete.work_items - 1,
    )
    assert limited.status == "resource_limit"
    assert tuple(record.point_count for record in limited.iterations) == (3, 2)
    assert len(limited.retained_boxes) == 2
    assert limited.work_items == limited.max_work_items
    assert not limited.conditional_invariance_certified and not limited.origin_exit_certified


def test_limit_before_any_complete_descent_retains_the_validated_candidate(reference):
    rows = _swap_rows()
    limited = derive_c6_carried_viability(
        reference, state=_state(rows[0]), epi_states=rows, timestep=1., max_work_items=1,
    )
    assert limited.status == "resource_limit" and len(limited.iterations) == 1
    assert limited.retained_boxes == limited.initial_boxes
    assert limited.iterations[0].point_count == 2 and limited.iterations[0].origin_retained


def test_complete_small_affine_domain_matches_independent_pointwise_preimages(reference):
    first = (math.nextafter(.5, -math.inf), .5) * 3
    rows, state = (first, first[::-1]), _state(first)
    result = derive_c6_carried_viability(reference, state=state, epi_states=rows, timestep=.25, max_boxes=100)

    def indices(box):
        lower = tuple(int((a - x) / g) for a, x, g in zip(box.lower, result.affine_origin, result.coordinate_spacings))
        upper = tuple(int((b - x) / g) for b, x, g in zip(box.upper, result.affine_origin, result.coordinate_spacings))
        return product(*(range(a, b + 1) for a, b in zip(lower, upper)))

    source = {point: box.epi for box in result.initial_boxes for point in indices(box)}
    translations = {}
    for row in rows:
        pressure = _observe_rebuilt_c6_pressure_lattice(reference, row).pressure
        translations[row] = tuple(int(F(.25) * F(p) / g) for p, g in zip(pressure, result.coordinate_spacings))
    image = {point: tuple(a + d for a, d in zip(point, translations[row])) for point, row in source.items()}
    first_kernel = {point for point, target in image.items() if target in source}
    second_kernel = {point for point in first_kernel if image[point] in first_kernel}
    assert tuple(map(len, (source, first_kernel, second_kernel))) == (18522, 3458, 254)
    assert tuple(record.point_count for record in result.iterations) == (18522, 3458, 254)
    assert {point for box in result.retained_boxes for point in indices(box)} == second_kernel
    assert (0,) * 6 in first_kernel and (0,) * 6 not in second_kernel
    assert result.origin_exit_certified and result.exclusion_step_bound == 2


def test_piece_limit_discards_a_complete_but_unadmitted_partial_partition(reference):
    first = (math.nextafter(.5, -math.inf), .5) * 3
    result = derive_c6_carried_viability(
        reference, state=_state(first), epi_states=(first, first[::-1]), timestep=.25, max_boxes=2,
    )
    assert result.status == "resource_limit" and len(result.iterations) == 1
    assert result.retained_boxes == result.initial_boxes
    assert result.iterations[0].box_count == 2 and result.iterations[0].point_count == 18522
    assert not result.origin_exit_certified and not result.conditional_invariance_certified


def test_even_and_odd_rn_ties_are_read_from_the_shared_grid_owner(reference):
    for visible in (.5, math.nextafter(.5, -math.inf)):
        row = (visible,) * 6
        result = derive_c6_carried_viability(reference, state=_state(row), epi_states=(row,), timestep=1.)
        cells = derive_nodal_remainder_itinerary(
            epi_states=(row, row), timesteps=(0.,), capacities=((1.,) * 6,), pressures=((0.,) * 6,),
            epi_lower=.375, epi_upper=.625,
        ).coordinates
        box = result.retained_boxes[0]
        assert box.lower == tuple(cell.first_grid_index * GRID for cell in cells)
        assert box.upper == tuple(cell.last_grid_index * GRID for cell in cells)
        previous, following = math.nextafter(visible, -math.inf), math.nextafter(visible, math.inf)
        lower_tie, upper_tie = (F(previous) + F(visible)) / 2, (F(visible) + F(following)) / 2
        if visible == .5:
            assert box.lower[0] == lower_tie and box.upper[0] == upper_tie
        else:
            assert box.lower[0] == lower_tie + GRID and box.upper[0] == upper_tie - GRID


def test_candidate_cells_are_clipped_to_the_actual_state_band(reference):
    row = (.5,) * 6
    state = replace(_state(row), epi_lower=.5)
    result = derive_c6_carried_viability(reference, state=state, epi_states=(row,), timestep=1.)
    assert result.status == "fixed_point" and result.retained_boxes[0].lower == (F(.5),) * 6


def test_adjacent_same_cell_boxes_coalesce_without_filling_a_gap(reference):
    row = (.5,) * 6
    base = _box(row)
    first = replace(base, lower=(F(.5) - GRID,) + base.lower[1:])
    second = replace(base, lower=(F(.5) + GRID,) + base.lower[1:],
                     upper=(F(.5) + GRID,) + base.upper[1:])
    result = derive_c6_carried_viability(
        reference, state=_state(row), epi_states=(row,), timestep=1., initial_boxes=(first, second),
    )
    assert result.status == "fixed_point" and len(result.retained_boxes) == 1
    assert result.iterations[-1].point_count == 3
    separated = replace(second, lower=(F(.5) + 2 * GRID,) + base.lower[1:],
                        upper=(F(.5) + 2 * GRID,) + base.upper[1:])
    result = derive_c6_carried_viability(
        reference, state=_state(row), epi_states=(row,), timestep=1., initial_boxes=(first, separated),
    )
    assert result.status == "fixed_point" and len(result.retained_boxes) == 2
    assert result.iterations[-1].point_count == 3


def test_forged_derived_reference_fields_are_rebuilt(reference):
    row = (.5,) * 6
    forged = replace(reference, sources=(F(100),) * 6, epi_quantum=F(1), gradient_quantum=F(1), rows=())
    result = derive_c6_carried_viability(forged, state=_state(row), epi_states=(row,), timestep=1.)
    assert result.reference == reference and result.pressures == ((0.,) * 6,)
    assert result.status == "fixed_point"


@pytest.mark.parametrize("name,value", (("max_work_items", 0), ("max_work_items", True),
                                        ("max_work_items", 1.), ("max_boxes", -1), ("max_boxes", True)))
def test_resource_limits_are_positive_actual_integers(reference, name, value):
    row = (.5,) * 6
    with pytest.raises(ValueError, match="positive exact integer"):
        derive_c6_carried_viability(reference, state=_state(row), epi_states=(row,), timestep=1., **{name: value})


def test_initial_size_limit_is_checked_before_search(reference):
    rows = _swap_rows()
    with pytest.raises(ValueError, match="max_boxes"):
        derive_c6_carried_viability(reference, state=_state(rows[0]), epi_states=rows, timestep=1., max_boxes=1)


def test_duplicate_rows_and_overlapping_boxes_are_rejected(reference):
    row = (.5,) * 6
    with pytest.raises(ValueError, match="distinct"):
        derive_c6_carried_viability(reference, state=_state(row), epi_states=(row, row), timestep=1.)
    with pytest.raises(ValueError, match="disjoint"):
        derive_c6_carried_viability(
            reference, state=_state(row), epi_states=(row,), timestep=1., initial_boxes=(_box(row), _box(row)),
        )


def test_initial_box_must_contain_the_unmodified_origin(reference):
    rows = _swap_rows()
    with pytest.raises(ValueError, match="unchanged supplied origin"):
        derive_c6_carried_viability(
            reference, state=_state(rows[0]), epi_states=rows, timestep=1., initial_boxes=(_box(rows[1]),),
        )


def test_custom_bounds_cannot_include_an_odd_rounding_tie(reference):
    visible = math.nextafter(.5, -math.inf)
    row = (visible,) * 6
    midpoint = (F(visible) + F(.5)) / 2
    box = replace(_box(row), upper=(midpoint,) + (F(visible),) * 5)
    with pytest.raises(ValueError, match="legal RN cell"):
        derive_c6_carried_viability(reference, state=_state(row), epi_states=(row,), timestep=1., initial_boxes=(box,))


def test_custom_bounds_require_exact_fractions_and_matching_cells(reference):
    row = (.5,) * 6
    with pytest.raises(TypeError, match="exact Fraction"):
        derive_c6_carried_viability(
            reference, state=_state(row), epi_states=(row,), timestep=1.,
            initial_boxes=(replace(_box(row), lower=row),),
        )
    other = (.5625,) * 6
    with pytest.raises(ValueError, match="declared visible row"):
        derive_c6_carried_viability(
            reference, state=_state(row), epi_states=(row,), timestep=1., initial_boxes=(_box(other),),
        )


def test_noncanonical_state_or_nonpositive_timestep_is_rejected(reference):
    row = (.5,) * 6
    with pytest.raises(TypeError, match="exact NodalRemainderState"):
        derive_c6_carried_viability(reference, state=None, epi_states=(row,), timestep=1.)
    with pytest.raises(ValueError, match="positive timestep"):
        derive_c6_carried_viability(reference, state=_state(row), epi_states=(row,), timestep=0.)
    state = initialize_nodal_remainder(row, epi_lower=.25, epi_upper=.75)
    with pytest.raises(ValueError, match="reference slab"):
        derive_c6_carried_viability(reference, state=state, epi_states=(row,), timestep=1.)


def test_returned_certificate_is_frozen(reference):
    row = (.5,) * 6
    result = derive_c6_carried_viability(reference, state=_state(row), epi_states=(row,), timestep=1.)
    with pytest.raises(FrozenInstanceError):
        result.status = "origin_excluded"
