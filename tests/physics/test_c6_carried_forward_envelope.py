"""Independent finite-state oracles for conditional forward envelopes."""

from dataclasses import replace
from fractions import Fraction as F
from itertools import product

import pytest

from tnfr.dynamics._euler_kernel import NodalRemainderState, advance_nodal_remainder
from tnfr.physics import c6_carried_viability as owner
from tnfr.physics.c6_pressure_lattice import (
    _observe_rebuilt_c6_pressure_lattice,
    derive_c6_pressure_lattice,
)

GRID = F(1, 2**3222)
DELTA = F(1, 2**54)
VALUES = tuple(float(F(1, 2) + (i - 4) * DELTA) for i in range(3))
TRIPLE_ROWS = tuple(product(VALUES, repeat=6))
ORIGIN_INDICES = (1, 2, 2, 1, 0, 0)


@pytest.fixture(scope="module")
def reference():
    return derive_c6_pressure_lattice(
        phase=(0.0,) * 6, epi_weight=1.0, phase_weight=1.0
    )


def _state(indices=ORIGIN_INDICES):
    return NodalRemainderState(
        tuple(VALUES[i] for i in indices), (F(0),) * 6, 0.375, 0.625
    )


def _step(reference, state, timestep):
    return advance_nodal_remainder(
        state,
        timestep=timestep,
        capacity=(1.0,) * 6,
        pressure=_observe_rebuilt_c6_pressure_lattice(reference, state.epi).pressure,
    )


def _integer_image(point):
    # At h=2 the exact pure EPI nodal update on this uniform binary64
    # binade is the integer linear map left + right - current.
    return tuple(point[(i - 1) % 6] + point[(i + 1) % 6] - point[i] for i in range(6))


def _finite_layers():
    """Enumerate 729 physical states, without the DBM implementation."""
    domain = set(product(range(3), repeat=6))
    layers = [domain]
    outgoing = []
    while True:
        image = {_integer_image(point) for point in layers[-1]}
        outgoing.append(image - domain)
        if image <= domain:
            return layers, outgoing
        following = image & domain
        assert following < layers[-1]
        layers.append(following)


def _contains(result, zones, state):
    point = tuple(
        (x - a) / result.grid_quantum
        for x, a in zip(state.exact_epi, result.affine_origin, strict=True)
    )
    if any(value.denominator != 1 for value in point):
        return False
    extended = point + (F(0),)
    return any(
        zone.epi == state.epi
        and all(
            extended[i] - extended[j] <= zone.bounds[i][j]
            for i in range(7)
            for j in range(7)
        )
        for zone in zones
    )


def _singleton_indices(result, zones):
    """Read the actual singleton sets; do not call an envelope helper."""
    points = set()
    for zone in zones:
        low = tuple(-zone.bounds[6][i] for i in range(6))
        high = tuple(zone.bounds[i][6] for i in range(6))
        assert low == high
        exact = tuple(
            a + result.grid_quantum * index
            for a, index in zip(result.affine_origin, low, strict=True)
        )
        assert tuple(float(value) for value in exact) == zone.epi
        indices = tuple((x - F(VALUES[0])) / DELTA for x in exact)
        assert all(value.denominator == 1 for value in indices)
        point = tuple(map(int, indices))
        assert point not in points
        points.add(point)
    return points


@pytest.fixture(scope="module")
def triple(reference):
    return owner.derive_c6_carried_forward_envelope(
        reference,
        state=_state(),
        epi_states=TRIPLE_ROWS,
        timestep=2.0,
        derive_pair_barriers=False,
    )


def test_zero_pressure_core_certifies_stationarity_with_unchanged_nonzero_carry(
    reference,
):
    row = (0.5,) * 6
    state = NodalRemainderState(row, tuple(i * GRID for i in range(6)), 0.375, 0.625)
    result = owner.derive_c6_carried_forward_envelope(
        reference,
        state=state,
        epi_states=(row,),
        timestep=1.0,
        derive_pair_barriers=False,
    )
    assert result.status == "invariant_core"
    assert (
        result.conditional_invariance_certified
        and result.conditional_boundedness_certified
    )
    assert result.grid_quantum == GRID and result.affine_origin == state.exact_epi
    assert result.initial_zones == result.retained_zones
    assert (
        result.entry_state == state
        and result.entry_steps == ()
        and result.entry_failure is None
    )
    assert result.completed_image_layers == 0 and result.intersections == 0
    assert result.pressures == ((0.0,) * 6,)
    assert not result.origin_exit_certified
    assert (
        not result.future_runtime_certified
        and not result.asymptotic_convergence_certified
    )


def test_positive_nonstationary_core_preserves_the_actual_carried_step(reference):
    rows = tuple(product(VALUES[:2], repeat=6))
    state = _state((0, 1, 0, 1, 0, 1))
    result = owner.derive_c6_carried_forward_envelope(
        reference,
        state=state,
        epi_states=rows,
        timestep=0.5,
        derive_pair_barriers=False,
    )
    assert (
        result.status == "invariant_core" and result.conditional_boundedness_certified
    )
    assert result.grid_quantum == DELTA / 4 and result.completed_image_layers == 0
    assert result.entry_state == state and not result.entry_steps
    first = _step(reference, state, 0.5)
    assert first.after != state and first.pressure != (0.0,) * 6
    midpoint = (F(VALUES[0]) + F(VALUES[1])) / 2
    assert first.after.exact_epi == (midpoint,) * 6
    assert first.after.remainder == (DELTA / 2,) * 6
    assert _contains(result, result.retained_zones, first.after)
    assert _step(reference, first.after, 0.5).after == first.after
    assert not result.asymptotic_convergence_certified


def test_every_canonical_pressure_matches_the_independent_729_state_integer_map(
    reference, triple
):
    assert triple.grid_quantum == DELTA
    for point, row, pressure in zip(
        product(range(3), repeat=6), TRIPLE_ROWS, triple.pressures, strict=True
    ):
        target = _integer_image(point)
        expected = tuple(
            DELTA * (after - before)
            for before, after in zip(point, target, strict=True)
        )
        assert tuple(2 * F(value) for value in pressure) == expected
        fresh = _observe_rebuilt_c6_pressure_lattice(reference, row).pressure
        assert fresh == pressure
        state = _state(point)
        step = advance_nodal_remainder(
            state, timestep=2.0, capacity=(1.0,) * 6, pressure=fresh
        )
        assert step.after.exact_epi == tuple(
            F(VALUES[0]) + DELTA * value for value in target
        )


def test_exact_forward_layers_match_the_complete_independent_finite_oracle(triple):
    layers, outgoing = _finite_layers()
    assert tuple(map(len, layers)) == (729, 45, 9, 3)
    assert tuple(map(len, outgoing)) == (380, 36, 6, 0)
    assert tuple(item.zone_count for item in triple.iterations) == tuple(
        map(len, layers)
    )
    assert tuple(item.origin_retained for item in triple.iterations) == (
        True,
        False,
        False,
        False,
    )
    assert tuple(item.outgoing_facets for item in triple.iterations) == (
        1296,
        108,
        12,
        0,
    )
    assert _singleton_indices(triple, triple.initial_zones) == layers[0]
    assert _singleton_indices(triple, triple.retained_zones) == layers[-1]
    assert triple.status == "invariant_core" and triple.completed_image_layers == 3
    assert (
        triple.conditional_invariance_certified
        and triple.conditional_boundedness_certified
    )
    assert not _contains(triple, triple.retained_zones, triple.state)


def test_origin_outside_core_is_bound_by_a_fresh_proof_derived_entry(reference, triple):
    assert len(triple.entry_steps) == triple.completed_image_layers == 3
    current = triple.state
    for retained in triple.entry_steps:
        fresh = _step(reference, current, 2.0)
        assert retained == fresh and retained.before == current
        current = fresh.after
        assert _contains(triple, triple.initial_zones, current)
    assert current == triple.entry_state
    assert current.exact_epi == (F(VALUES[1]),) * 6
    assert current.remainder == (F(0),) * 6 and triple.entry_failure is None
    assert _contains(triple, triple.retained_zones, current)
    assert triple.entry_steps[0].pressure != (0.0,) * 6
    assert all(step.pressure == (0.0,) * 6 for step in triple.entry_steps[1:])
    # The proper invariant core and the finite entry prefix form an invariant
    # set containing the original state, even though the core alone does not.
    invariant = _singleton_indices(triple, triple.retained_zones) | {ORIGIN_INDICES}
    assert {_integer_image(point) for point in invariant} <= invariant


def test_incomplete_image_discards_partial_targets_and_retains_the_last_complete_layer(
    reference, triple
):
    limited = owner.derive_c6_carried_forward_envelope(
        reference,
        state=_state(),
        epi_states=TRIPLE_ROWS,
        timestep=2.0,
        derive_pair_barriers=False,
        max_intersections=triple.intersections - 1,
    )
    layers, _ = _finite_layers()
    assert limited.status == "resource_limit"
    assert limited.completed_image_layers == 2
    assert tuple(item.zone_count for item in limited.iterations) == (729, 45, 9)
    assert _singleton_indices(limited, limited.retained_zones) == layers[2]
    assert limited.intersections == limited.max_intersections == 182
    assert limited.entry_state is None and limited.entry_steps == ()
    assert limited.entry_failure is None
    assert (
        not limited.conditional_invariance_certified
        and not limited.conditional_boundedness_certified
    )
    assert not limited.origin_exit_certified


def test_exactly_completed_intersection_budget_still_permits_the_inclusion_proof(
    reference, triple
):
    exact_budget = owner.derive_c6_carried_forward_envelope(
        reference,
        state=_state(),
        epi_states=TRIPLE_ROWS,
        timestep=2.0,
        derive_pair_barriers=False,
        max_intersections=triple.intersections,
    )
    assert exact_budget.status == "invariant_core"
    assert exact_budget.retained_zones == triple.retained_zones
    assert exact_budget.entry_steps == triple.entry_steps
    assert exact_budget.intersections == exact_budget.max_intersections == 183


def test_limit_before_a_complete_image_keeps_the_validated_source(reference):
    result = owner.derive_c6_carried_forward_envelope(
        reference,
        state=_state(),
        epi_states=TRIPLE_ROWS,
        timestep=2.0,
        derive_pair_barriers=False,
        max_intersections=1,
    )
    assert result.status == "resource_limit" and result.completed_image_layers == 0
    assert result.retained_zones == result.initial_zones
    assert len(result.retained_zones) == 729 and len(result.iterations) == 1
    assert result.entry_steps == () and result.entry_state is None


def test_reference_caches_are_rebuilt_and_entry_pressure_is_refreshed(
    reference, triple, monkeypatch
):
    forged = replace(
        reference,
        sources=(F(99),) * 6,
        epi_quantum=F(3),
        gradient_quantum=F(7),
        rows=(),
        every_pressure_has_positive=True,
    )
    calls = []
    canonical = owner._observe_rebuilt_c6_pressure_lattice

    def record(ref, row):
        calls.append((ref, row))
        return canonical(ref, row)

    monkeypatch.setattr(owner, "_observe_rebuilt_c6_pressure_lattice", record)
    result = owner.derive_c6_carried_forward_envelope(
        forged,
        state=_state(),
        epi_states=TRIPLE_ROWS,
        timestep=2.0,
        derive_pair_barriers=False,
    )
    assert result.reference == reference and result.pressures == triple.pressures
    assert (
        result.retained_zones == triple.retained_zones
        and result.entry_steps == triple.entry_steps
    )
    assert len(calls) == len(TRIPLE_ROWS) + len(result.entry_steps)
    assert all(ref == reference for ref, _row in calls)
    assert tuple(row for _ref, row in calls[-3:]) == tuple(
        step.before.epi for step in result.entry_steps
    )


def test_derived_pair_intervals_are_universally_protected_on_the_declared_source(
    reference,
):
    result = owner.derive_c6_carried_forward_envelope(
        reference,
        state=_state(),
        epi_states=TRIPLE_ROWS,
        timestep=2.0,
        derive_pair_barriers=True,
    )
    assert len(result.pair_barriers) == 15
    for barrier in result.pair_barriers:
        assert barrier.lower <= 0 <= barrier.upper
        for point in product(range(3), repeat=6):
            relative = tuple(x - a for x, a in zip(point, ORIGIN_INDICES, strict=True))
            value = relative[barrier.first] - relative[barrier.second]
            if barrier.lower <= value <= barrier.upper:
                target = _integer_image(point)
                after = tuple(
                    x - a for x, a in zip(target, ORIGIN_INDICES, strict=True)
                )
                assert (
                    barrier.lower
                    <= after[barrier.first] - after[barrier.second]
                    <= barrier.upper
                )
    assert _contains(result, result.initial_zones, result.state)


@pytest.mark.parametrize(
    "argument,value,error",
    (
        ("max_intersections", 0, ValueError),
        ("max_intersections", True, ValueError),
        ("max_cells", 0, ValueError),
        ("max_cells", 1.0, ValueError),
        ("derive_pair_barriers", 1, TypeError),
        ("derive_pair_barriers", None, TypeError),
    ),
)
def test_resource_and_policy_parameters_have_strict_primitive_types(
    reference, argument, value, error
):
    with pytest.raises(error):
        owner.derive_c6_carried_forward_envelope(
            reference,
            state=_state(),
            epi_states=TRIPLE_ROWS,
            timestep=2.0,
            **{argument: value},
        )
