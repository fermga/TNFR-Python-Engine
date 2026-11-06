"""Finite integer oracles for last-return memory, plus canonical source checks."""

from copy import deepcopy
from dataclasses import replace
from fractions import Fraction as F
from itertools import product

import pytest

from tnfr.dynamics._euler_kernel import NodalRemainderState, advance_nodal_remainder
from tnfr.physics import c6_carried_return as owner
from tnfr.physics.c6_carried_viability import C6CarriedForwardZone
from tnfr.physics.c6_pressure_lattice import derive_c6_pressure_lattice

POINTS = tuple((x, y, 0, 0, 0, 0) for x, y in product(range(-3, 5), repeat=2))
ZERO = (0,) * 6


def _hull(points):
    points = tuple((*point, 0) for point in points)
    return (
        None
        if not points
        else tuple(
            tuple(max(p[i] - p[j] for p in points) for j in range(7)) for i in range(7)
        )
    )


def _members(bounds):
    if bounds is None:
        return set()
    return {
        p
        for p in POINTS
        if all(
            (*p, 0)[i] - (*p, 0)[j] <= bounds[i][j] for i in range(7) for j in range(7)
        )
    }


def _add(point, shift):
    return tuple(a + b for a, b in zip(point, shift, strict=True))


def _rectangle(xlow, xhigh, ylow, yhigh):
    return _hull(
        (x, y, 0, 0, 0, 0)
        for x, y in product(
            range(xlow, xhigh + 1),
            range(ylow, yhigh + 1),
        )
    )


@pytest.fixture(scope="module")
def geometry():
    # Synthetic guarded integer translations test the private geometry only.
    # Public-entry tests below independently reconstruct canonical nodal data.
    rows = tuple((0.4 + i / 100,) + (0.5,) * 5 for i in range(5))
    areas = (
        _rectangle(-2, 2, 0, 0),
        _rectangle(-2, 3, -2, 3),
        _rectangle(-2, 3, -2, 3),
        _rectangle(-2, 3, 0, 0),
        _hull(((1, 0, 0, 0, 0, 0),)),
    )
    diagonal = _hull(
        p
        for p in POINTS
        if -2 <= p[0] <= 2 and -2 <= p[1] <= 2 and abs(p[0] - p[1]) <= 1
    )
    specs = (
        (0, None, 0, _hull((ZERO,)), ZERO),
        (0, None, 1, _hull((ZERO,)), (1, 0, 0, 0, 0, 0)),
        (1, None, 1, diagonal, (1, 1, 0, 0, 0, 0)),
        (1, None, 2, _rectangle(1, 3, 1, 3), (-1, 0, 0, 0, 0, 0)),
        (2, None, 1, _rectangle(0, 2, 1, 3), (0, -1, 0, 0, 0, 0)),
        (3, None, 3, _rectangle(-2, 2, 0, 0), (1, 0, 0, 0, 0, 0)),
        (0, 4, 2, _rectangle(-2, 2, 0, 0), (2, 0, 0, 0, 0, 0)),
    )
    edges = tuple(
        owner.C6CarriedReturnTransition(
            rows[a],
            None if m is None else rows[m],
            rows[b],
            guard,
            None if m is None else areas[m],
            _hull(_add(p, shift) for p in _members(guard)),
            shift,
        )
        for a, m, b, guard, shift in specs
    )
    state = NodalRemainderState(rows[0], (F(0),) * 6, 0.375, 0.625)
    zones = tuple(
        C6CarriedForwardZone(row, area) for row, area in zip(rows, areas, strict=True)
    )
    return owner.C6CarriedReturnEnvelope(
        None,
        state,
        1.0,
        rows,
        ((1.0, 0.0, 0.0, 0.0, 0.0, 0.0),) + ((0.0,) * 6,) * 4,
        F(1),
        state.exact_epi,
        (rows[4],),
        zones,
        zones,
        edges,
        (),
        len(edges),
        True,
        True,
        (),
        "fixed_point",
        0,
        0,
        1,
        len(rows),
    )


def _derive(geometry, maximum=10_000, arcs=10_000):
    return owner._derive_return_memory_envelope(
        geometry,
        max_memory_work=maximum,
        max_memory_arcs=arcs,
    )


def _oracle(geometry):
    areas = {z.epi: _members(z.bounds) for z in geometry.retained_zones}
    edges = geometry.return_relation
    shifts = dict(
        zip(
            geometry.epi_states,
            (tuple(map(int, p)) for p in geometry.pressures),
            strict=True,
        )
    )
    guards, roots = [], []
    for edge in edges:
        guard = {
            p
            for p in _members(edge.source_guard) & areas[edge.source_epi]
            if _add(p, edge.shift) in areas[edge.target_epi]
        }
        if edge.intermediate_epi is not None:
            guard = {
                p
                for p in guard
                if _add(p, shifts[edge.source_epi]) in areas[edge.intermediate_epi]
            }
        guards.append(guard)
        roots.append(
            {_add(ZERO, edge.shift)}
            if edge.source_epi == geometry.state.epi and ZERO in guard
            else set()
        )
    initial = [
        {_add(p, edge.shift) for p in guard}
        for edge, guard in zip(edges, guards, strict=True)
    ]
    pairs = tuple(
        (i, j)
        for i, edge in enumerate(edges)
        for j, following in enumerate(edges)
        if edge.target_epi == following.source_epi and initial[i] & guards[j]
    )
    current = initial
    while True:
        following = [set(root) for root in roots]
        for i, j in pairs:
            following[j].update(_add(p, edges[j].shift) for p in current[i] & guards[j])
        following = [_members(_hull(points)) for points in following]
        if following == current:
            return guards, roots, initial, pairs, tuple(map(_hull, following))
        current = following


def test_full_pair_geometry_and_stationary_descent_match_independent_integer_oracle(
    geometry,
):
    result = _derive(geometry)
    guards, roots, initial, pairs, final = _oracle(geometry)
    assert result.source_guards == tuple(map(_hull, guards))
    assert result.root_endpoint_zones == tuple(map(_hull, roots))
    assert result.initial_endpoint_zones == tuple(map(_hull, initial))
    assert result.pair_arcs == pairs
    assert result.retained_endpoint_zones == final
    assert result.status == "fixed_point" and not result.pending_memories
    assert result.strict_updates > 1 and result.completed_visits > len(initial)
    assert result.retained_endpoint_zones[5] is None  # Unreachable advancing cycle.
    assert result.initialization_complete and result.pair_relation_complete
    assert result.domain_confined_origin_histories_covered
    assert (
        not result.conditional_invariance_certified
        and not result.conditional_boundedness_certified
    )
    assert (
        not result.future_runtime_certified
        and not result.asymptotic_convergence_certified
    )


def test_transient_guard_is_clipped_at_its_own_time_and_origin_injection_persists(
    geometry,
):
    result = _derive(geometry)
    assert _members(geometry.return_relation[6].source_guard) != {ZERO}
    assert _members(result.source_guards[6]) == {ZERO}
    assert _members(result.root_endpoint_zones[6]) == {(2, 0, 0, 0, 0, 0)}
    assert all(
        _members(root) <= _members(zone)
        for root, zone in zip(
            result.root_endpoint_zones,
            result.retained_endpoint_zones,
            strict=True,
        )
    )


def test_every_budget_prefix_retains_all_finite_origin_histories_without_partial_updates(
    geometry,
):
    complete = _derive(geometry)
    guards, roots, _, pairs, _ = _oracle(geometry)
    reached = [set(root) for root in roots]
    while True:
        following = [set(p) for p in reached]
        for i, j in pairs:
            following[j].update(
                _add(p, geometry.return_relation[j].shift)
                for p in reached[i] & guards[j]
            )
        if following == reached:
            break
        reached = following
    previous = None
    for maximum in range(1, complete.memory_work + 1):
        limited = _derive(geometry, maximum)
        assert limited.memory_work <= maximum
        if not limited.initialization_complete:
            assert not limited.domain_confined_origin_histories_covered
            assert (
                limited.source_guards
                == limited.retained_endpoint_zones
                == limited.pair_arcs
                == ()
            )
            continue
        assert all(
            p <= _members(z)
            for p, z in zip(reached, limited.retained_endpoint_zones, strict=True)
        )
        if previous is not None:
            assert all(
                _members(a) <= _members(b)
                for a, b in zip(
                    limited.retained_endpoint_zones,
                    previous,
                    strict=True,
                )
            )
        previous = limited.retained_endpoint_zones
        if not limited.pair_relation_complete:
            assert (
                limited.pair_arcs == ()
                and limited.retained_endpoint_zones == limited.initial_endpoint_zones
            )
            assert limited.descent_work == limited.completed_visits == 0
        elif limited.status != "fixed_point":
            assert limited.pending_memories
            next_cost = max(
                1, sum(j == limited.pending_memories[0] for _, j in limited.pair_arcs)
            )
            assert limited.memory_work + next_cost > maximum
    assert previous == complete.retained_endpoint_zones


def test_arc_cap_discards_partial_graph_and_retains_initial_cover(geometry):
    result = _derive(geometry, arcs=1)
    assert (
        result.status == "memory_arc_resource_limit" and result.initialization_complete
    )
    assert not result.pair_relation_complete and result.pair_arcs == ()
    assert result.retained_endpoint_zones == result.initial_endpoint_zones
    assert result.descent_work == result.completed_visits == 0


def test_empty_or_incomplete_relation_is_handled_without_promoting_an_incomplete_cover(
    geometry,
):
    empty = _derive(replace(geometry, return_relation=()))
    assert empty.status == "fixed_point" and empty.pair_relation_complete
    assert (
        empty.domain_confined_origin_histories_covered
        and empty.retained_endpoint_zones == ()
    )
    incomplete = _derive(replace(geometry, relation_complete=False))
    assert incomplete.status == "return_construction_resource_limit"
    assert (
        not incomplete.initialization_complete and not incomplete.pair_relation_complete
    )
    assert (
        incomplete.memory_work == 0
        and not incomplete.domain_confined_origin_histories_covered
    )


@pytest.fixture(scope="module")
def canonical():
    delta = F(1, 2**54)
    values = tuple(float(F(1, 2) + (i - 4) * delta) for i in range(2))
    rows = tuple(product(values, repeat=6))
    state = NodalRemainderState(rows[7], (F(0),) * 6, 0.375, 0.625)
    reference = derive_c6_pressure_lattice(
        phase=(0.0,) * 6, epi_weight=1.0, phase_weight=1.0
    )
    args = dict(state=state, epi_states=rows, timestep=2.0, transient_epi_states=())
    return (
        reference,
        args,
        owner.derive_c6_carried_return_memory_envelope(reference, **args),
    )


def test_public_entry_rebuilds_canonical_source_and_covers_the_shared_nodal_path(
    canonical,
):
    reference, args, proof = canonical
    before = deepcopy((reference, args))
    forged = replace(reference, sources=(17.0,) * 6, rows=(), epi_quantum=F(7))
    rebuilt = owner.derive_c6_carried_return_memory_envelope(forged, **args)
    assert rebuilt == proof and (reference, args) == before
    assert (
        proof.status == "fixed_point" and proof.return_envelope.reference == reference
    )
    state = args["state"]
    envelope = proof.return_envelope
    index = {row: i for i, row in enumerate(envelope.epi_states)}
    observed = 0
    for _ in range(20):
        if state.epi not in index:
            break
        step = advance_nodal_remainder(
            state,
            timestep=2.0,
            capacity=(1.0,) * 6,
            pressure=envelope.pressures[index[state.epi]],
        )
        assert not any(step.nodal_balance_residual)
        if step.after.epi not in index:
            break
        endpoint = tuple(
            int((q - o) / envelope.grid_quantum)
            for q, o in zip(
                step.after.exact_epi,
                envelope.affine_origin,
                strict=True,
            )
        )
        candidates = [
            zone
            for edge, zone in zip(
                envelope.return_relation, proof.retained_endpoint_zones, strict=True
            )
            if edge.source_epi == state.epi
            and edge.target_epi == step.after.epi
            and zone is not None
        ]
        assert any(
            all(
                (*endpoint, 0)[i] - (*endpoint, 0)[j] <= zone[i][j]
                for i in range(7)
                for j in range(7)
            )
            for zone in candidates
        )
        state = step.after
        observed += 1
    assert observed >= 1


@pytest.mark.parametrize(
    "name,value",
    (
        ("max_memory_work", 0),
        ("max_memory_work", True),
        ("max_memory_arcs", -1),
        ("max_memory_arcs", 1.5),
    ),
)
def test_public_memory_guards_are_strict_positive_integers(canonical, name, value):
    reference, args, _ = canonical
    with pytest.raises((TypeError, ValueError)):
        owner.derive_c6_carried_return_memory_envelope(
            reference, **args, **{name: value}
        )


def test_canonical_and_memory_budgets_are_independent(canonical):
    reference, args, complete = canonical
    limited = owner.derive_c6_carried_return_memory_envelope(
        reference, **args, max_intersections=1
    )
    assert not limited.return_envelope.relation_complete and limited.memory_work == 0
    limited = owner.derive_c6_carried_return_memory_envelope(
        reference, **args, max_memory_work=1
    )
    assert limited.return_envelope == complete.return_envelope
    assert (
        limited.status == "memory_initialization_resource_limit"
        and limited.memory_work == 1
    )
