"""Independent finite predecessor oracles for guarded last-return memory."""

from dataclasses import replace
from fractions import Fraction as F

import pytest

from tnfr.dynamics._euler_kernel import NodalRemainderState, advance_nodal_remainder
from tnfr.physics import c6_carried_return as owner
from tnfr.physics.c6_carried_viability import C6CarriedForwardZone
from tnfr.physics.c6_pressure_lattice import derive_c6_pressure_lattice


def _hull(values):
    points = tuple((value, 0, 0, 0, 0, 0, 0) for value in values)
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
        value
        for value in range(-1, 12)
        if all(
            (value, 0, 0, 0, 0, 0, 0)[i] - (value, 0, 0, 0, 0, 0, 0)[j] <= bounds[i][j]
            for i in range(7)
            for j in range(7)
        )
    }


@pytest.fixture(scope="module")
def memory():
    # This private algebra fixture distinguishes original zero from later
    # states in the same RN mode. It asserts no canonical pressure realization.
    rows = ((0.4,) * 6, (0.5,) * 6, (0.6,) * 6)
    specs = ((0, 1, (0,), 1), (1, 0, (1,), 1), (0, 0, (2,), 0), (0, 0, (10,), 0))
    edges = tuple(
        owner.C6CarriedReturnTransition(
            rows[a],
            None,
            rows[b],
            _hull(values),
            None,
            _hull(v + shift for v in values),
            (shift, 0, 0, 0, 0, 0),
        )
        for a, b, values, shift in specs
    )
    zones = tuple(
        C6CarriedForwardZone(row, _hull(values))
        for row, values in zip(
            rows,
            (range(11), (1,), (3, 11)),
            strict=True,
        )
    )
    # Two distinct terminal pieces permit a nonempty contribution followed by
    # a disjoint one, exercising joins without assuming every meet succeeds.
    terminals = tuple(
        owner.C6CarriedReturnTransition(
            rows[0],
            None,
            rows[2],
            _hull((value,)),
            None,
            _hull((value + 1,)),
            (1, 0, 0, 0, 0, 0),
        )
        for value in (2, 10)
    )
    state = NodalRemainderState(rows[0], (F(0),) * 6, 0.375, 0.625)
    env = owner.C6CarriedReturnEnvelope(
        None,
        state,
        1.0,
        rows,
        ((0.0,) * 6,) * 3,
        F(1),
        state.exact_epi,
        (rows[2],),
        zones,
        zones,
        edges,
        terminals,
        6,
        True,
        True,
        (),
        "fixed_point",
        0,
        0,
        1,
        3,
    )
    return owner._derive_return_memory_envelope(
        env, max_memory_work=1000, max_memory_arcs=100
    )


def _targets(memory, *entries):
    return tuple(
        C6CarriedForwardZone(memory.return_envelope.epi_states[index], _hull(values))
        for index, values in entries
    )


def _query(memory, targets, maximum=1000):
    return owner._derive_return_memory_region_queries(memory, (targets,), maximum)


def _oracle(memory, targets):
    envelope, edges = memory.return_envelope, memory.return_envelope.return_relation
    zones = [*_members_per_zone(memory.retained_endpoint_zones), {0}]
    sentinel = len(edges)
    graph = []
    for i, j in memory.pair_arcs:
        for value in zones[i] & _members(memory.source_guards[j]):
            image = value + edges[j].shift[0]
            if image in zones[j]:
                graph.append((i, j, value, image))
    graph.extend(
        (sentinel, j, 0, edges[j].shift[0])
        for j, root in enumerate(memory.root_endpoint_zones)
        if root is not None
    )
    current = []
    for i, zone in enumerate(zones):
        row = envelope.state.epi if i == sentinel else edges[i].target_epi
        selected = set()
        for target in targets:
            target_values = _members(target.bounds)
            if target.epi == row:
                selected.update(zone & target_values)
            for edge in envelope.intermediate_transitions:
                if edge.source_epi == row and edge.target_epi == target.epi:
                    selected.update(
                        v
                        for v in zone & _members(edge.source_guard)
                        if v + edge.shift[0] in target_values
                    )
        current.append(_members(_hull(selected)))
    layers = [tuple(map(_hull, current))]
    for _ in range(30):
        if 0 in current[sentinel]:
            return layers, "origin_not_excluded"
        if not any(current):
            return layers, "empty_complete_layer"
        following = [set() for _ in zones]
        for i, j, value, image in graph:
            if image in current[j]:
                following[i].add(value)
        following = [_members(_hull(values)) for values in following]
        layers.append(tuple(map(_hull, following)))
        if following == current:
            return layers, "stationary_complete_layer"
        current = following
    raise AssertionError("the finite oracle failed to terminate")


def _members_per_zone(zones):
    return tuple(map(_members, zones))


@pytest.mark.parametrize(
    "entries",
    (
        ((0, (0,)),),
        ((0, (2,)),),
        ((0, (10,)),),
        ((2, (3,)),),
        ((2, (11,)),),
        ((0, (2,)), (2, (11,))),
    ),
)
def test_complete_layers_match_independent_enumeration_including_terminal_pieces(
    memory, entries
):
    targets = _targets(memory, *entries)
    expected, status = _oracle(memory, targets)
    result = _query(memory, targets)
    query = result.queries[0]
    assert result.query_relation_complete and query.initialization_complete
    assert (
        query.initial_endpoint_zones == expected[0]
        and query.retained_endpoint_zones == expected[-1]
    )
    assert query.status == status and query.completed_depth == len(expected) - 1
    assert tuple(layer.zone_count for layer in query.iterations) == tuple(
        sum(zone is not None for zone in layer) for layer in expected
    )
    assert query.origin_path_within_domain_excluded == (status != "origin_not_excluded")
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


def test_return_to_origin_rn_mode_is_distinct_from_original_zero_sentinel(memory):
    reached = _query(memory, _targets(memory, (0, (2,)))).queries[0]
    unreachable = _query(memory, _targets(memory, (0, (10,)))).queries[0]
    assert reached.status == "origin_not_excluded" and reached.completed_depth == 2
    assert (
        not reached.iterations[0].origin_present
        and reached.iterations[-1].origin_present
    )
    assert (
        unreachable.status == "stationary_complete_layer"
        and unreachable.completed_depth == 1
    )
    assert unreachable.origin_path_within_domain_excluded


def test_each_query_budget_retains_only_complete_layers(memory):
    targets = _targets(memory, (0, (2,)))
    complete = _query(memory, targets)
    expected, _ = _oracle(memory, targets)
    saw_layer_stop = False
    for maximum in range(1, complete.queries[0].intersections + 1):
        result = _query(memory, targets, maximum)
        query = result.queries[0]
        assert (
            query.intersections <= maximum
            and result.query_relation_intersections <= maximum
        )
        if not result.query_relation_complete:
            assert not query.initialization_complete and not query.iterations
        elif not query.initialization_complete:
            assert query.initial_endpoint_zones == query.retained_endpoint_zones == ()
        else:
            assert query.retained_endpoint_zones == expected[query.completed_depth]
            if query.status == "resource_limit":
                saw_layer_stop = True
                assert not query.origin_path_within_domain_excluded
        if not query.initialization_complete:
            assert (
                not query.origin_path_within_domain_excluded
                and query.completed_depth is None
            )
    assert saw_layer_stop


def test_partial_target_initialization_discards_every_piece(memory):
    targets = _targets(memory, (0, (2,)), (1, (1,)), (2, (3, 11)))
    complete = _query(memory, targets)
    saw_initial_stop = False
    for maximum in range(
        complete.query_relation_intersections, complete.queries[0].intersections
    ):
        result = _query(memory, targets, maximum)
        query = result.queries[0]
        if result.query_relation_complete and not query.initialization_complete:
            saw_initial_stop = True
            assert query.status == "memory_query_initialization_resource_limit"
            assert (
                query.initial_endpoint_zones
                == query.retained_endpoint_zones
                == query.iterations
                == ()
            )
            assert (
                not query.origin_path_within_domain_excluded
                and query.intersections == maximum
            )
    assert saw_initial_stop


def test_each_whole_target_gets_its_own_budget(memory):
    targets = _targets(memory, (0, (2,)))
    complete = _query(memory, targets)
    maximum = complete.queries[0].intersections - 1
    result = owner._derive_return_memory_region_queries(
        memory, (targets, targets), maximum
    )
    assert result.queries[0] == result.queries[1]
    assert all(
        q.status == "resource_limit" and q.intersections == maximum
        for q in result.queries
    )


def test_incomplete_pair_graph_cannot_supply_any_query_exclusion(memory):
    for incomplete in (
        replace(memory, pair_relation_complete=False, pair_arcs=()),
        replace(
            memory,
            initialization_complete=False,
            pair_relation_complete=False,
            retained_endpoint_zones=(),
            pair_arcs=(),
        ),
    ):
        result = _query(incomplete, _targets(memory, (0, (10,))))
        assert (
            not result.query_relation_complete
            and result.query_relation_intersections == 0
        )
        query = result.queries[0]
        assert query.status == "memory_query_construction_resource_limit"
        assert (
            not query.initialization_complete
            and not query.origin_path_within_domain_excluded
        )


def test_shared_graph_helper_matches_exact_finite_transition_guards_and_work(memory):
    edges = memory.return_envelope.return_relation
    zones = _members_per_zone(memory.retained_endpoint_zones)
    expected, required = [], 0
    for i, j in memory.pair_arcs:
        required += 1
        if not zones[i] or not zones[j]:
            continue
        source = zones[i] & _members(memory.source_guards[j])
        if source:
            required += 1
            source = {x for x in source if x + edges[j].shift[0] in zones[j]}
        if source:
            expected.append((i, j, _hull(source), edges[j].shift))
    for j, root in enumerate(memory.root_endpoint_zones):
        required += 1
        if root is not None:
            expected.append((len(edges), j, _hull((0,)), edges[j].shift))
    graph, complete, work = owner._return_memory_query_graph(memory, required)
    assert complete and graph == tuple(expected) and work == required
    for maximum in range(1, required):
        graph, complete, work = owner._return_memory_query_graph(memory, maximum)
        assert graph == () and not complete and work == maximum


@pytest.mark.parametrize(
    "entries",
    (
        ((0, (0,)),),
        ((0, (2,)),),
        ((2, (3,)),),
        ((0, (2,)), (2, (11,))),
        ((0, (2,)), (1, (1,)), (2, (3, 11))),
    ),
)
def test_shared_initialization_helper_preserves_exact_arrays_and_atomic_budget(
    memory, entries
):
    targets = _targets(memory, *entries)
    expected = _oracle(memory, targets)[0][0]
    envelope = memory.return_envelope
    edges = envelope.return_relation
    zones = (*_members_per_zone(memory.retained_endpoint_zones), {0})
    required = 0
    for i, zone in enumerate(zones):
        if not zone:
            required += 1
            continue
        row = envelope.state.epi if i == len(edges) else edges[i].target_epi
        for target in targets:
            required += 1 + (row == target.epi)
            for edge in envelope.intermediate_transitions:
                if edge.source_epi == row and edge.target_epi == target.epi:
                    required += 1 + bool(zone & _members(edge.source_guard))
    initial, complete, work = owner._return_memory_query_initialization(
        memory, targets, required
    )
    assert complete and initial == expected and work == required
    for maximum in range(1, required):
        initial, complete, work = owner._return_memory_query_initialization(
            memory, targets, maximum
        )
        assert initial == () and not complete and work == maximum


def test_shared_helpers_keep_initial_cover_and_graph_completeness_as_separate_premises(
    memory,
):
    targets = _targets(memory, (0, (2,)))
    incomplete_graph = replace(memory, pair_relation_complete=False, pair_arcs=())
    assert owner._return_memory_query_graph(incomplete_graph, 1000) == ((), False, 0)
    assert owner._return_memory_query_initialization(
        incomplete_graph, targets, 1000
    ) == (owner._return_memory_query_initialization(memory, targets, 1000))
    incomplete_cover = replace(
        incomplete_graph, initialization_complete=False, retained_endpoint_zones=()
    )
    assert owner._return_memory_query_initialization(
        incomplete_cover, targets, 1000
    ) == ((), False, 0)


DELTA = F(1, 2**54)
ORIGIN = (-20,) * 6
OTHER = (-19,) * 6
CHAIN = (
    (-20, -20, -20, -20, -19, -19),
    (-19, -20, -20, -19, -20, -20),
    (-21, -19, -19, -21, -19, -19),
    (-17, -21, -21, -17, -21, -21),
)


def _row(point):
    return tuple(float(F(1, 2) + v * DELTA) for v in point)


def _state(point):
    return NodalRemainderState(_row(point), (F(0),) * 6, 0.375, 0.625)


def _target(point, origin):
    vector = tuple(a - b for a, b in zip(point, origin, strict=True)) + (0,)
    return C6CarriedForwardZone(
        _row(point), tuple(tuple(a - b for b in vector) for a in vector)
    )


@pytest.fixture(scope="module")
def reference():
    return derive_c6_pressure_lattice(
        phase=(0.0,) * 6, epi_weight=1.0, phase_weight=1.0
    )


def _canonical(
    reference,
    origin=ORIGIN,
    points=(ORIGIN, OTHER, *CHAIN),
    targets=(CHAIN[-1],),
    **changes,
):
    args = dict(
        state=_state(origin),
        epi_states=tuple(map(_row, points)),
        timestep=2.0,
        transient_epi_states=(_row(CHAIN[1]),),
        target_region_groups=tuple((_target(p, origin),) for p in targets),
    )
    args.update(changes)
    return owner.derive_c6_carried_return_memory_region_exclusions(reference, **args)


def test_canonical_full_targets_exclude_by_empty_or_exact_stationary_layers(reference):
    result = _canonical(reference, targets=(CHAIN[-1], OTHER, ORIGIN))
    assert tuple(q.status for q in result.queries) == (
        "empty_complete_layer",
        "stationary_complete_layer",
        "origin_not_excluded",
    )
    assert tuple(q.completed_depth for q in result.queries) == (0, 1, 0)
    assert tuple(q.origin_path_within_domain_excluded for q in result.queries) == (
        True,
        True,
        False,
    )


def test_nonstationary_memory_cover_allows_complete_backward_exclusion(reference):
    full = _canonical(reference)
    envelope = full.memory_envelope.return_envelope
    coarse = _canonical(
        reference, max_intersections=envelope.construction_intersections
    )
    memory = coarse.memory_envelope
    result = _canonical(
        reference,
        max_intersections=envelope.construction_intersections,
        max_memory_work=memory.initialization_work + memory.pair_construction_work,
    )
    assert result.memory_envelope.status == "memory_resource_limit"
    query = result.queries[0]
    assert query.status == "empty_complete_layer" and query.completed_depth == 2
    assert tuple(q.zone_count for q in query.iterations) == (1, 1, 0)
    assert query.origin_path_within_domain_excluded


def test_terminal_transient_without_any_completed_origin_return_is_still_detected(
    reference,
):
    result = _canonical(
        reference,
        origin=CHAIN[0],
        points=(ORIGIN, CHAIN[0], CHAIN[1]),
        targets=(CHAIN[1],),
    )
    memory, query = result.memory_envelope, result.queries[0]
    assert all(
        edge.source_epi != _row(CHAIN[0])
        for edge in memory.return_envelope.return_relation
    )
    pressure = memory.return_envelope.pressures[
        memory.return_envelope.epi_states.index(_row(CHAIN[0]))
    ]
    step = advance_nodal_remainder(
        _state(CHAIN[0]), timestep=2.0, capacity=(1.0,) * 6, pressure=pressure
    )
    assert step.after == _state(CHAIN[1]) and not any(step.nodal_balance_residual)
    assert query.status == "origin_not_excluded" and query.completed_depth == 0
    assert query.initial_endpoint_zones[-1] == _hull((0,))
    assert not query.origin_path_within_domain_excluded


@pytest.mark.parametrize(
    "changes",
    (
        {"query_max_intersections": True},
        {"query_max_intersections": 0},
        {"target_region_groups": ()},
        {"target_region_groups": []},
        {"target_region_groups": ((),)},
    ),
)
def test_public_query_validation_is_strict(reference, changes):
    with pytest.raises((TypeError, ValueError)):
        _canonical(reference, **changes)
