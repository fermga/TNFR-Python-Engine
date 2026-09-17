"""Independent finite-state and finite-set oracles for exact return unions."""

from dataclasses import replace
from fractions import Fraction as F
from itertools import product

import pytest

from tnfr.dynamics._euler_kernel import NodalRemainderState, advance_nodal_remainder
from tnfr.physics import c6_carried_return as owner
from tnfr.physics.c6_carried_viability import C6CarriedForwardZone, _Work, _dbm_join
from tnfr.physics.c6_pressure_lattice import derive_c6_pressure_lattice


DELTA = F(1, 2**54)
ORIGIN = (-20,) * 6
FIXED = (-19,) * 6
CHAIN = (
    (-20, -20, -20, -20, -19, -19),
    (-19, -20, -20, -19, -20, -20),
    (-21, -19, -19, -21, -19, -19),
    (-17, -21, -21, -17, -21, -21),
)
POINTS = (ORIGIN, FIXED, *CHAIN)


def row(point):
    return tuple(float(F(1, 2) + value * DELTA) for value in point)


def state(point=ORIGIN):
    return NodalRemainderState(row(point), (F(0),) * 6, .375, .625)


def target(point, origin=ORIGIN):
    vector = tuple(a-b for a, b in zip(point, origin, strict=True)) + (0,)
    return C6CarriedForwardZone(row(point), tuple(tuple(a-b for b in vector) for a in vector))


def image(point):
    return tuple(point[i-1] + point[(i+1) % 6] - point[i] for i in range(6))


@pytest.fixture(scope="module")
def reference():
    return derive_c6_pressure_lattice(phase=(0.,) * 6, epi_weight=1., phase_weight=1.)


def derive(reference, targets=(CHAIN[-1],), origin=ORIGIN, **changes):
    args = dict(state=state(origin), epi_states=tuple(map(row, POINTS)), timestep=2.,
                transient_epi_states=(row(CHAIN[1]),),
                target_region_groups=tuple((target(point, origin),) for point in targets))
    args.update(changes)
    return owner.derive_c6_carried_return_union_exclusions(reference, **args)


@pytest.fixture(scope="module")
def result(reference):
    return derive(reference, targets=(CHAIN[-1], CHAIN[1], FIXED))


def test_chain_oracle_matches_shared_nodal_integrator(reference):
    envelope = derive(reference, origin=CHAIN[0]).return_envelope
    for before, after in zip(CHAIN, CHAIN[1:]):
        assert image(before) == after
        pressure = envelope.pressures[envelope.epi_states.index(row(before))]
        step = advance_nodal_remainder(state(before), timestep=2., capacity=(1.,)*6, pressure=pressure)
        assert step.after == state(after) and not any(step.nodal_balance_residual)
    assert image(CHAIN[-1]) not in POINTS


def test_empty_and_stationary_complete_union_certificates(result):
    for query in result.queries[:2]:
        assert query.status == "empty_complete_layer" and query.completed_depth == 0
        assert query.initial_base_zones == query.retained_zones == ()
        assert query.initialization_complete and query.origin_path_within_domain_excluded
    stationary = result.queries[2]
    assert stationary.status == "stationary_complete_layer" and stationary.completed_depth == 1
    assert stationary.initial_base_zones == stationary.retained_zones == (target(FIXED),)
    assert stationary.origin_path_within_domain_excluded
    assert result.intersections == sum(q.intersections for q in result.queries)
    assert result.subsumptions == sum(q.subsumptions for q in result.queries)
    assert result.common_grid_relaxes_coordinate_cosets
    assert not result.conditional_boundedness_certified and not result.conditional_invariance_certified
    assert not result.asymptotic_convergence_certified and not result.future_runtime_certified
    assert not stationary.actual_origin_reachability_certified


def test_equal_zone_counts_do_not_replace_semantic_equality(reference, result):
    queried = derive(reference, max_intersections=result.return_envelope.construction_intersections)
    query = queried.queries[0]
    assert queried.return_envelope.relation_complete
    assert query.status == "empty_complete_layer" and query.completed_depth == 3
    assert tuple(layer.zone_count for layer in query.iterations) == (1, 1, 1, 0)
    assert query.initial_base_zones == (target(CHAIN[-1]),)


def test_origin_collision_after_two_guarded_returns_is_not_promoted(reference):
    query = derive(reference, origin=CHAIN[0]).queries[0]
    assert query.status == "origin_not_excluded" and query.completed_depth == 2
    assert tuple(layer.origin_present for layer in query.iterations) == (False, False, True)
    assert not query.origin_path_within_domain_excluded and not query.actual_origin_reachability_certified


def test_origin_in_initial_target_stops_the_query(reference):
    query = derive(reference, targets=(ORIGIN,)).queries[0]
    assert query.completed_depth == 0 and query.iterations[0].origin_present
    assert query.status == "origin_not_excluded" and not query.origin_path_within_domain_excluded


def test_transient_first_image_without_any_later_return_is_not_lost(reference):
    queried = derive(reference, targets=(CHAIN[1],), origin=CHAIN[0],
                     epi_states=tuple(map(row, (ORIGIN, CHAIN[0], CHAIN[1]))))
    query = queried.queries[0]
    assert not any(edge.source_epi == row(CHAIN[0]) for edge in queried.return_envelope.return_relation)
    assert query.status == "origin_not_excluded" and query.completed_depth == 0
    assert len(query.transient_target_preimages) == 1
    edge = query.transient_target_preimages[0]
    assert edge.source_guard == target(CHAIN[0], CHAIN[0]).bounds
    assert edge.target_guard == target(CHAIN[1], CHAIN[0]).bounds


def test_two_step_return_keeps_the_actual_intermediate_gate(reference):
    envelope = derive(reference, origin=CHAIN[0]).return_envelope
    edges = [edge for edge in envelope.return_relation if edge.source_epi == row(CHAIN[0])]
    assert len(edges) == 1
    edge = edges[0]
    assert edge.intermediate_epi == row(CHAIN[1]) and edge.target_epi == row(CHAIN[2])
    assert edge.source_guard == target(CHAIN[0], CHAIN[0]).bounds
    assert edge.intermediate_guard == target(CHAIN[1], CHAIN[0]).bounds
    assert edge.target_guard == target(CHAIN[2], CHAIN[0]).bounds
    assert edge.shift == tuple(a-b for a, b in zip(CHAIN[2], CHAIN[0], strict=True))


def test_mixed_targets_keep_transient_pullbacks_and_original_quantification(reference, result):
    supplied = (target(CHAIN[-1]), target(CHAIN[1]))
    queried = derive(reference, target_region_groups=(supplied,),
                     max_intersections=result.return_envelope.construction_intersections)
    query = queried.queries[0]
    assert query.target_regions == supplied
    assert set(query.initial_base_zones) == {target(CHAIN[-1]), target(CHAIN[0])}
    assert len(query.transient_target_preimages) == 1
    assert query.completed_depth == 3 and query.origin_path_within_domain_excluded


def test_partial_relation_and_initialization_do_not_publish_absence(reference, result):
    partial = derive(reference, max_intersections=1).queries[0]
    assert partial.status == "construction_resource_limit"
    assert partial.completed_depth is None and partial.iterations == ()
    assert not partial.initialization_complete and not partial.origin_path_within_domain_excluded
    partial = derive(reference, targets=(CHAIN[1],), query_max_intersections=1,
                     max_intersections=result.return_envelope.construction_intersections).queries[0]
    assert partial.status == "initialization_resource_limit"
    assert partial.initial_base_zones == partial.retained_zones == partial.transient_target_preimages == ()
    assert partial.iterations == () and not partial.origin_path_within_domain_excluded


def test_partial_backward_layer_preserves_last_complete_union(reference, result):
    budget = result.return_envelope.construction_intersections
    full = derive(reference, max_intersections=budget)
    partial = derive(reference, max_intersections=budget,
                     query_max_intersections=full.intersections-1).queries[0]
    assert partial.status == "resource_limit" and partial.completed_depth == 1
    assert partial.retained_zones == (target(CHAIN[2]),)
    assert not partial.origin_path_within_domain_excluded


def test_initial_zone_limit_discards_the_complete_unpublished_union(reference, result):
    supplied = (target(CHAIN[-1]), target(CHAIN[1]))
    query = derive(reference, target_region_groups=(supplied,), query_max_zones=1,
                   max_intersections=result.return_envelope.construction_intersections).queries[0]
    assert query.status == "initialization_zone_resource_limit"
    assert query.target_regions == supplied and query.retained_zones == ()
    assert query.iterations == () and not query.origin_path_within_domain_excluded


def test_initial_subsumption_stop_discards_only_the_unfinished_query(reference, result):
    supplied = (target(CHAIN[0]), target(CHAIN[1]))
    queried = derive(reference, target_region_groups=(supplied, supplied), query_max_subsumptions=1,
                     max_intersections=result.return_envelope.construction_intersections)
    first, second = queried.queries
    assert first.initialization_complete and first.origin_path_within_domain_excluded
    assert first.subsumptions == queried.subsumptions == 1
    assert second.status == "initialization_subsumption_resource_limit"
    assert not second.initialization_complete and not second.origin_path_within_domain_excluded
    assert second.initial_base_zones == second.retained_zones == second.transient_target_preimages == ()
    assert second.iterations == () and second.subsumptions == 0


def test_shared_intersection_budget_advances_independent_queries_round_robin(reference, result):
    queried = derive(reference, targets=(CHAIN[-1], CHAIN[-1]), query_max_intersections=4,
                     max_intersections=result.return_envelope.construction_intersections)
    assert queried.intersections == sum(q.intersections for q in queried.queries) == 4
    assert tuple(q.completed_depth for q in queried.queries) == (1, 1)
    assert all(q.status == "resource_limit" and not q.origin_path_within_domain_excluded
               for q in queried.queries)


def test_zone_cap_inside_a_backward_layer_discards_already_inserted_pieces(reference):
    alternative = tuple(a+b for a, b in zip(CHAIN[2], (1, 1, 0, -1, -1, 0), strict=True))
    assert image(alternative) == CHAIN[-1]
    rows = tuple(map(row, (*POINTS, alternative)))
    envelope = derive(reference, epi_states=rows).return_envelope
    query = derive(reference, epi_states=rows, query_max_zones=1,
                   max_intersections=envelope.construction_intersections).queries[0]
    assert query.status == "zone_resource_limit" and query.completed_depth == 0
    assert query.initial_base_zones == query.retained_zones == (target(CHAIN[-1]),)
    assert not query.origin_path_within_domain_excluded


def box(low, high):
    low = tuple(low) + (0,)*(7-len(low))
    high = tuple(high) + (0,)*(7-len(high))
    return tuple(tuple(0 if i == j else high[i]-low[j] for j in range(7)) for i in range(7))


def members(bounds):
    result = set()
    for first, second in product(range(-2, 3), repeat=2):
        point = (first, second, 0, 0, 0, 0, 0)
        if all(point[i]-point[j] <= bounds[i][j] for i in range(7) for j in range(7)):
            result.add((first, second))
    return result


def test_exact_union_preserves_a_hole_that_a_hull_turns_into_a_spurious_past():
    key = row(ORIGIN)
    first, second, origin = box((-2, 0), (-1, 0)), box((1, 0), (2, 0)), box((0, 0), (0, 0))
    union = owner._ReturnUnion((key,), 8, _Work(100))
    union.insert(key, first)
    union.insert(key, second)
    assert union.count == 2 and not union.contains_origin(key)
    assert set().union(*(members(z.bounds) for z in union.public())) == {(-2, 0), (-1, 0), (1, 0), (2, 0)}
    assert members(_dbm_join(first, second)) & members(origin) == {(0, 0)}
    assert all(not (members(zone.bounds) & members(origin)) for zone in union.public())


def test_subsumption_is_exact_under_an_independent_finite_set_oracle():
    key, union, expected = row(ORIGIN), owner._ReturnUnion((row(ORIGIN),), 25, _Work(10000)), set()
    boxes = (box((-2, -2), (-1, 1)), box((0, 0), (2, 2)), box((-1, -1), (1, 1)),
             box((0, 0), (0, 0)), box((-2, -2), (2, 2)))
    for value in boxes:
        expected |= members(value)
        union.insert(key, value)
        assert set().union(*(members(z.bounds) for z in union.public())) == expected
    assert union.count == 1 and union.public()[0].bounds == boxes[-1]


def test_subsumption_resource_stop_keeps_the_original_antichain_atomic():
    key = row(ORIGIN)
    union = owner._ReturnUnion((key,), 8, _Work(1))
    first, second = box((-2,), (-1,)), box((1,), (2,))
    union.insert(key, first)
    before = union.public()
    with pytest.raises(owner._UnionLimit) as raised:
        union.insert(key, second)
    assert raised.value.status == "subsumption_resource_limit"
    assert union.public() == before and union.count == 1


@pytest.mark.parametrize("changes", [
    {"query_max_intersections": 0}, {"query_max_intersections": True},
    {"query_max_zones": 0}, {"query_max_zones": 1.5},
    {"query_max_subsumptions": 0}, {"query_max_subsumptions": True},
    {"target_region_groups": ()}, {"target_region_groups": ((),)},
    {"target_region_groups": ((target(ORIGIN), target(ORIGIN)),)},
])
def test_invalid_resource_guards_and_target_groups_are_rejected(reference, changes):
    with pytest.raises((TypeError, ValueError)):
        derive(reference, **changes)


def test_unclosed_and_noninteger_targets_are_rejected(reference):
    item = target(ORIGIN)
    unclosed = [list(values) for values in item.bounds]
    unclosed[0][1] += 1
    with pytest.raises(ValueError, match="closed difference bounds"):
        derive(reference, target_region_groups=((replace(item, bounds=tuple(map(tuple, unclosed))),),))
    rational = tuple(tuple(F(v) for v in values) for values in item.bounds)
    with pytest.raises(TypeError, match="exact integers"):
        derive(reference, target_region_groups=((replace(item, bounds=rational),),))


def test_forged_reference_caches_cannot_change_the_union_map(reference, result):
    forged = replace(reference, sources=(99.,)*6, rows=(), epi_quantum=F(3), gradient_quantum=F(5))
    queried = derive(forged, targets=(CHAIN[-1], CHAIN[1], FIXED))
    assert queried.return_envelope.reference == reference
    assert queried.return_envelope.pressures == result.return_envelope.pressures
    assert queried.queries == result.queries
