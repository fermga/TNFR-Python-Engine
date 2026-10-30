"""Independent finite counterexamples for carried forward-envelope claims."""

from fractions import Fraction as F
from itertools import product

import pytest

from tnfr.dynamics._euler_kernel import NodalRemainderState, advance_nodal_remainder
from tnfr.physics.c6_carried_viability import derive_c6_carried_forward_envelope
from tnfr.physics.c6_pressure_lattice import (
    _observe_rebuilt_c6_pressure_lattice,
    derive_c6_pressure_lattice,
)

DELTA = F(1, 2**54)
LOW = float(F(1, 2) - 4 * DELTA)
HIGH = float(F(LOW) + DELTA)
ROWS = tuple(product((LOW, HIGH), repeat=6))


@pytest.fixture(scope="module")
def reference():
    return derive_c6_pressure_lattice(
        phase=(0.0,) * 6, epi_weight=1.0, phase_weight=1.0
    )


def _state(epi=(LOW,) * 6, carry=(F(0),) * 6, band=(0.375, 0.625)):
    return NodalRemainderState(epi, carry, *band)


def _derive(reference, state=None, rows=ROWS, timestep=1.0, **kwargs):
    return derive_c6_carried_forward_envelope(
        reference,
        state=_state() if state is None else state,
        epi_states=rows,
        timestep=timestep,
        derive_pair_barriers=False,
        **kwargs,
    )


def _contains(result, zones, exact):
    """Read public inequalities at an independently specified rational point."""
    indices = tuple(
        (x - origin) / result.grid_quantum
        for x, origin in zip(exact, result.affine_origin, strict=True)
    )
    if any(value.denominator != 1 for value in indices):
        return False
    point = tuple(map(int, indices)) + (0,)
    row = tuple(map(float, exact))
    zone = next((zone for zone in zones if zone.epi == row), None)
    return zone is not None and all(
        point[i] - point[j] <= zone.bounds[i][j] for i in range(7) for j in range(7)
    )


def _small_image(point):
    # Coordinates are (X-LOW)/(DELTA/2). LOW is even, HIGH is odd;
    # precisely x=2 displays HIGH on {-1,0,1,2}.
    visible = tuple(int(x == 2) for x in point)
    return tuple(
        point[i] + visible[(i - 1) % 6] + visible[(i + 1) % 6] - 2 * visible[i]
        for i in range(6)
    )


def test_missing_cartesian_corner_cannot_be_hidden_by_complete_axis_projections(
    reference,
):
    assert (LOW,) * 6 in ROWS[:-1]
    assert all({row[i] for row in ROWS[:-1]} == {LOW, HIGH} for i in range(6))
    with pytest.raises(ValueError, match="complete Cartesian"):
        _derive(reference, rows=ROWS[:-1])


def test_full_visible_product_with_an_actual_grid_hole_is_rejected(reference):
    skipped = float(F(LOW) + 2 * DELTA)
    rows = tuple(product((LOW, skipped), repeat=6))
    # At h=1 the common grid is DELTA, so the omitted middle float is
    # reachable on that grid and cannot be hidden by checking outer facets.
    with pytest.raises(ValueError, match="contiguous affine-grid"):
        _derive(reference, rows=rows)


def test_visible_gap_without_any_omitted_affine_grid_point_is_allowed(reference):
    skipped = float(F(LOW) + 2 * DELTA)
    result = _derive(
        reference,
        rows=tuple(product((LOW, skipped), repeat=6)),
        timestep=2.0,
        max_intersections=1,
    )
    assert result.grid_quantum == 2 * DELTA
    assert len(result.initial_zones) == 64
    assert not _contains(result, result.initial_zones, (F(HIGH),) * 6)
    assert result.status == "resource_limit"
    assert not result.conditional_invariance_certified


def test_six_distinct_affine_bases_preserve_exact_cell_membership(reference):
    carry = tuple(DELTA * F(k, 8) for k in (-2, -1, 0, 1, 2, 3))
    state = _state(carry=carry)
    result = _derive(reference, state=state, max_intersections=1)
    assert result.grid_quantum == DELTA / 2
    assert result.affine_origin == state.exact_epi
    assert len({value % result.grid_quantum for value in carry}) == 4
    coordinates = tuple(
        tuple(
            origin + index * result.grid_quantum
            for index in range(-4, 5)
            if float(origin + index * result.grid_quantum) in (LOW, HIGH)
        )
        for origin in state.exact_epi
    )
    assert all(len(axis) == 4 for axis in coordinates)
    for exact in product(*coordinates):
        assert _contains(result, result.initial_zones, exact)
    off_coset = (state.exact_epi[0] + DELTA / 8,) + state.exact_epi[1:]
    assert tuple(map(float, off_coset)) in ROWS
    assert not _contains(result, result.initial_zones, off_coset)


def test_even_and_odd_rounding_ties_are_not_interchangeable(reference):
    result = _derive(reference, max_intersections=1)
    for offset, admitted in (
        (-DELTA / 2, True),
        (DELTA / 2, True),
        (3 * DELTA / 2, False),
    ):
        exact = (F(LOW) + offset,) + (F(LOW),) * 5
        assert _contains(result, result.initial_zones, exact) is admitted
    # A legal even-cell tie jumps beyond the odd cell, although its image
    # lies on the geometric outer midpoint of the two-cell family.
    epi = (LOW, HIGH, LOW, LOW, LOW, HIGH)
    initial = _state(epi, (DELTA / 2,) + (F(0),) * 5)
    pressure = _observe_rebuilt_c6_pressure_lattice(reference, epi).pressure
    step = advance_nodal_remainder(
        initial, timestep=1.0, capacity=(1.0,) * 6, pressure=pressure
    )
    assert step.after.exact_epi[0] == F(HIGH) + DELTA / 2
    assert step.after.epi[0] == float(F(HIGH) + DELTA)
    assert not _contains(result, result.initial_zones, step.after.exact_epi)


def test_stationary_relational_abstraction_with_fourteen_outgoing_points_is_not_invariant(
    reference,
):
    # First bind the independent integer formula to all canonical source rows.
    for bits in product((0, 1), repeat=6):
        row = tuple(HIGH if bit else LOW for bit in bits)
        pressure = _observe_rebuilt_c6_pressure_lattice(reference, row).pressure
        expected = tuple(
            DELTA / 2 * (bits[(i - 1) % 6] + bits[(i + 1) % 6] - 2 * bits[i])
            for i in range(6)
        )
        assert tuple(map(F, pressure)) == expected
    result = _derive(reference)
    domain = set(product(range(-1, 3), repeat=6))
    retained = {
        point
        for point in domain
        if _contains(
            result,
            result.retained_zones,
            tuple(F(LOW) + DELTA / 2 * x for x in point),
        )
    }
    images = {_small_image(point) for point in retained}
    assert len(retained) == 821
    assert len(images - domain) == 14
    assert images.intersection(domain) <= retained
    assert result.status == "stationary_outer_envelope"
    assert result.completed_image_layers == 7
    assert result.iterations[-1].outgoing_facets == 6
    assert not result.conditional_invariance_certified
    assert not result.conditional_boundedness_certified
    assert not result.origin_exit_certified
    assert not result.future_runtime_certified
    assert not result.asymptotic_convergence_certified


def test_empty_compatible_past_domain_does_not_relabel_itself_as_an_origin_exit_proof(
    reference,
):
    row = (LOW, HIGH) * 3
    result = _derive(reference, state=_state(row), rows=(row,))
    assert result.status == "empty_core"
    assert result.retained_zones == ()
    assert result.completed_image_layers == 1
    assert not result.iterations[-1].origin_retained
    assert not result.conditional_invariance_certified
    assert not result.conditional_boundedness_certified
    assert not result.origin_exit_certified
    assert result.entry_state is None and not result.entry_steps


@pytest.mark.parametrize("narrow_band", (False, True))
def test_proved_invariant_core_does_not_certify_an_origin_that_fails_to_enter(
    reference, narrow_band
):
    values = tuple(float(F(LOW) + i * DELTA) for i in range(3))
    epi = tuple(values[i] for i in (0, 2, 0, 0, 0, 0))
    band = (values[0], values[-1]) if narrow_band else (0.375, 0.625)
    initial = _state(epi, band=band)
    result = _derive(
        reference, state=initial, rows=tuple(product(values, repeat=6)), timestep=2.0
    )
    assert result.status == "invariant_core"
    assert result.conditional_invariance_certified
    assert tuple(record.zone_count for record in result.iterations) == (729, 45, 9, 3)
    assert not result.iterations[-1].origin_retained
    assert result.entry_state is None
    assert not result.conditional_boundedness_certified
    assert not result.origin_exit_certified
    assert result.state is initial and result.state.remainder == (F(0),) * 6
    if narrow_band:
        assert result.entry_failure == "band_exit" and result.entry_steps == ()
    else:
        assert result.entry_failure == "candidate_exit" and len(result.entry_steps) == 1
        step = result.entry_steps[0]
        assert step.before is initial
        assert step.after.exact_epi == tuple(
            F(LOW) + i * DELTA for i in (2, -2, 2, 0, 0, 0)
        )
        assert step.after.epi not in result.epi_states
