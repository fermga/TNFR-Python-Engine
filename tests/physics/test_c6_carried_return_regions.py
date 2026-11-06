"""Exact finite diffusion-chain oracles for guarded return target queries."""

from dataclasses import replace
from fractions import Fraction as F

import pytest

from tnfr.dynamics._euler_kernel import NodalRemainderState, advance_nodal_remainder
from tnfr.physics import c6_carried_return as owner
from tnfr.physics.c6_carried_viability import C6CarriedForwardZone
from tnfr.physics.c6_pressure_lattice import derive_c6_pressure_lattice

DELTA = F(1, 2**54)
ORIGIN = (-20,) * 6
OTHER_FIXED = (-19,) * 6
CHAIN = (
    (-20, -20, -20, -20, -19, -19),
    (-19, -20, -20, -19, -20, -20),
    (-21, -19, -19, -21, -19, -19),
    (-17, -21, -21, -17, -21, -21),
)
POINTS = (ORIGIN, OTHER_FIXED, *CHAIN)


def _row(point):
    return tuple(float(F(1, 2) + value * DELTA) for value in point)


def _state(point=ORIGIN):
    return NodalRemainderState(_row(point), (F(0),) * 6, 0.375, 0.625)


def _target(point, origin=ORIGIN):
    vector = tuple(a - b for a, b in zip(point, origin, strict=True)) + (0,)
    return C6CarriedForwardZone(
        _row(point), tuple(tuple(a - b for b in vector) for a in vector)
    )


def _image(point):
    return tuple(point[i - 1] + point[(i + 1) % 6] - point[i] for i in range(6))


@pytest.fixture(scope="module")
def reference():
    return derive_c6_pressure_lattice(
        phase=(0.0,) * 6, epi_weight=1.0, phase_weight=1.0
    )


def _derive(reference, targets=(CHAIN[-1],), origin=ORIGIN, **changes):
    args = dict(
        state=_state(origin),
        epi_states=tuple(map(_row, POINTS)),
        timestep=2.0,
        transient_epi_states=(_row(CHAIN[1]),),
        target_region_groups=tuple((_target(point, origin),) for point in targets),
    )
    args.update(changes)
    return owner.derive_c6_carried_return_region_exclusions(reference, **args)


@pytest.fixture(scope="module")
def result(reference):
    return _derive(reference, targets=(CHAIN[-1], CHAIN[1], OTHER_FIXED))


def test_finite_chain_is_the_actual_shared_nodal_map(reference):
    envelope = owner.derive_c6_carried_return_envelope(
        reference,
        state=_state(CHAIN[0]),
        epi_states=tuple(map(_row, POINTS)),
        timestep=2.0,
        transient_epi_states=(_row(CHAIN[1]),),
    )
    assert envelope.grid_quantum == DELTA
    for first, second in zip(CHAIN, CHAIN[1:]):
        assert _image(first) == second
        pressure = envelope.pressures[envelope.epi_states.index(_row(first))]
        step = advance_nodal_remainder(
            _state(first), timestep=2.0, capacity=(1.0,) * 6, pressure=pressure
        )
        assert step.after == _state(second) and not any(step.nodal_balance_residual)
    assert _image(CHAIN[-1]) not in POINTS


def test_refined_envelope_excludes_whole_transient_targets_at_depth_zero(result):
    assert result.return_envelope.status == "fixed_point"
    for query, point in zip(result.queries[:2], (CHAIN[-1], CHAIN[1]), strict=True):
        assert query.target_regions == (_target(point),)
        assert query.initialization_complete and query.status == "empty_complete_layer"
        assert query.initial_base_zones == query.retained_zones == ()
        assert query.completed_depth == query.past_exclusion_depth == 0
        assert query.origin_path_within_domain_excluded
    assert result.intersections == sum(q.intersections for q in result.queries)


def test_unreachable_fixed_target_closes_by_exact_stationary_layer(result):
    query = result.queries[2]
    assert query.status == "stationary_complete_layer" and query.completed_depth == 1
    assert query.initial_base_zones == query.retained_zones == (_target(OTHER_FIXED),)
    assert all(not layer.origin_present for layer in query.iterations)
    assert query.origin_path_within_domain_excluded
    assert not query.actual_origin_reachability_certified
    assert result.common_grid_relaxes_coordinate_cosets
    assert (
        not result.conditional_invariance_certified
        and not result.conditional_boundedness_certified
    )
    assert (
        not result.future_runtime_certified
        and not result.asymptotic_convergence_certified
    )


def test_backward_layers_with_equal_counts_are_not_mistaken_for_equality(
    reference, result
):
    limited = _derive(
        reference, max_intersections=result.return_envelope.construction_intersections
    )
    query = limited.queries[0]
    assert limited.return_envelope.status == "resource_limit"
    assert limited.return_envelope.relation_complete
    assert query.status == "empty_complete_layer" and query.completed_depth == 3
    assert tuple(item.zone_count for item in query.iterations) == (1, 1, 1, 0)
    assert query.origin_path_within_domain_excluded
    assert query.initial_base_zones == (_target(CHAIN[-1]),)


def test_original_origin_can_reach_target_only_after_two_complete_returns(reference):
    result = _derive(reference, origin=CHAIN[0])
    query = result.queries[0]
    assert query.status == "origin_not_excluded" and query.completed_depth == 2
    assert tuple(item.origin_present for item in query.iterations) == (
        False,
        False,
        True,
    )
    assert not query.origin_path_within_domain_excluded
    assert not query.actual_origin_reachability_certified


def test_transient_target_without_subsequent_return_is_still_checked(reference):
    points = (ORIGIN, CHAIN[0], CHAIN[1])
    result = _derive(
        reference,
        targets=(CHAIN[1],),
        origin=CHAIN[0],
        epi_states=tuple(map(_row, points)),
    )
    query = result.queries[0]
    assert not any(
        e.source_epi == _row(CHAIN[0]) for e in result.return_envelope.return_relation
    )
    assert query.status == "origin_not_excluded" and query.completed_depth == 0
    assert query.initial_base_zones == (_target(CHAIN[0], CHAIN[0]),)
    assert len(query.transient_target_preimages) == 1
    edge = query.transient_target_preimages[0]
    assert edge.source_epi == _row(CHAIN[0]) and edge.target_epi == _row(CHAIN[1])
    assert edge.intermediate_epi is None and edge.intermediate_guard is None
    assert edge.source_guard == _target(CHAIN[0], CHAIN[0]).bounds
    assert edge.target_guard == _target(CHAIN[1], CHAIN[0]).bounds
    assert not query.origin_path_within_domain_excluded


def test_transient_target_pullback_precedes_return_predecessor_layers(
    reference, result
):
    queried = _derive(
        reference,
        targets=(CHAIN[1],),
        max_intersections=result.return_envelope.construction_intersections,
    )
    query = queried.queries[0]
    assert query.status == "empty_complete_layer" and query.completed_depth == 1
    assert query.initial_base_zones == (_target(CHAIN[0]),)
    assert len(query.transient_target_preimages) == 1
    assert query.target_regions == (_target(CHAIN[1]),)


def test_mixed_target_group_preserves_both_direct_and_transient_pieces(
    reference, result
):
    targets = (_target(CHAIN[-1]), _target(CHAIN[1]))
    queried = _derive(
        reference,
        target_region_groups=(targets,),
        max_intersections=result.return_envelope.construction_intersections,
    )
    query = queried.queries[0]
    assert query.target_regions == targets and query.origin_path_within_domain_excluded
    assert set(query.initial_base_zones) == {_target(CHAIN[-1]), _target(CHAIN[0])}
    assert query.completed_depth == 3


def test_partial_relation_never_supplies_an_absence_certificate(reference):
    queried = _derive(reference, max_intersections=1)
    query = queried.queries[0]
    assert not queried.return_envelope.relation_complete
    assert (
        query.status == "construction_resource_limit"
        and not query.initialization_complete
    )
    assert query.completed_depth is None and query.past_exclusion_depth is None
    assert query.iterations == () and not query.origin_path_within_domain_excluded
    assert query.target_regions == (_target(CHAIN[-1]),)
    assert queried.intersections == 0


def test_partial_target_initialization_publishes_no_partial_layer(reference, result):
    queried = _derive(
        reference,
        targets=(CHAIN[1],),
        query_max_intersections=1,
        max_intersections=result.return_envelope.construction_intersections,
    )
    query = queried.queries[0]
    assert query.status == "initialization_resource_limit"
    assert not query.initialization_complete and query.completed_depth is None
    assert (
        query.initial_base_zones
        == query.retained_zones
        == query.transient_target_preimages
        == ()
    )
    assert queried.intersections == query.intersections == 1
    assert not query.origin_path_within_domain_excluded


def test_partial_backward_layer_retains_last_complete_layer(reference, result):
    budget = result.return_envelope.construction_intersections
    full = _derive(reference, max_intersections=budget)
    limited = _derive(
        reference,
        max_intersections=budget,
        query_max_intersections=full.intersections - 1,
    )
    query = limited.queries[0]
    assert query.status == "resource_limit" and query.completed_depth == 1
    assert query.retained_zones == (_target(CHAIN[2]),)
    assert not query.origin_path_within_domain_excluded
    assert tuple(item.zone_count for item in query.iterations) == (1, 1)


def test_interruption_inside_a_layer_discards_its_already_found_piece(reference):
    kernel = (1, 1, 0, -1, -1, 0)
    alternative = tuple(a + b for a, b in zip(CHAIN[2], kernel, strict=True))
    assert alternative != CHAIN[2] and _image(alternative) == CHAIN[3]
    rows = tuple(map(_row, (*POINTS, alternative)))
    envelope = owner.derive_c6_carried_return_envelope(
        reference,
        state=_state(),
        epi_states=rows,
        timestep=2.0,
        transient_epi_states=(_row(CHAIN[1]),),
    )
    queried = _derive(
        reference,
        epi_states=rows,
        max_intersections=envelope.construction_intersections,
        query_max_intersections=2,
    )
    query = queried.queries[0]
    assert query.status == "resource_limit" and query.completed_depth == 0
    assert query.retained_zones == query.initial_base_zones == (_target(CHAIN[-1]),)
    assert query.intersections == queried.intersections == 2
    assert not query.origin_path_within_domain_excluded


def test_queries_share_one_round_robin_budget(reference, result):
    queried = _derive(
        reference,
        targets=(CHAIN[-1], CHAIN[-1]),
        max_intersections=result.return_envelope.construction_intersections,
        query_max_intersections=4,
    )
    assert queried.intersections == 4
    assert sum(q.intersections for q in queried.queries) == 4
    assert tuple(q.completed_depth for q in queried.queries) == (1, 1)
    assert all(
        q.status == "resource_limit" and not q.origin_path_within_domain_excluded
        for q in queried.queries
    )


def test_original_origin_target_is_not_excluded(reference):
    query = _derive(reference, targets=(ORIGIN,)).queries[0]
    assert query.status == "origin_not_excluded" and query.completed_depth == 0
    assert (
        query.iterations[0].origin_present
        and not query.origin_path_within_domain_excluded
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"target_region_groups": ()},
        {"target_region_groups": []},
        {"target_region_groups": ((),)},
        {"target_region_groups": (None,)},
        {"target_region_groups": ((_target(ORIGIN), _target(ORIGIN)),)},
        {"query_max_intersections": 0},
        {"query_max_intersections": True},
    ],
)
def test_invalid_query_containers_and_guards_are_rejected(reference, changes):
    with pytest.raises((TypeError, ValueError)):
        _derive(reference, **changes)


def test_target_must_lie_inside_original_validated_domain(reference):
    outside = _target((-18,) * 6)
    with pytest.raises(ValueError, match="domain rows"):
        _derive(reference, target_region_groups=((outside,),))
    target = _target(ORIGIN)
    widened = tuple(
        tuple(v + int(i != j) for j, v in enumerate(row))
        for i, row in enumerate(target.bounds)
    )
    with pytest.raises(ValueError, match="inside the declared domain"):
        _derive(reference, target_region_groups=((replace(target, bounds=widened),),))


def test_noninteger_and_unclosed_target_bounds_are_rejected(reference):
    target = _target(ORIGIN)
    fractional = tuple(tuple(F(v) for v in row) for row in target.bounds)
    with pytest.raises(TypeError, match="exact integers"):
        _derive(
            reference, target_region_groups=((replace(target, bounds=fractional),),)
        )
    unclosed = [list(row) for row in target.bounds]
    unclosed[0][1] += 1
    with pytest.raises(ValueError, match="closed difference bounds"):
        _derive(
            reference,
            target_region_groups=(
                (replace(target, bounds=tuple(map(tuple, unclosed))),),
            ),
        )


def test_forged_reference_caches_are_rebuilt_for_return_queries(reference, result):
    forged = replace(
        reference,
        sources=(11.0,) * 6,
        rows=(),
        epi_quantum=F(7),
        gradient_quantum=F(13),
    )
    fresh = _derive(forged, targets=(CHAIN[-1], CHAIN[1], OTHER_FIXED))
    assert fresh.return_envelope.reference == reference
    assert fresh.return_envelope.pressures == result.return_envelope.pressures
    assert fresh.queries == result.queries
