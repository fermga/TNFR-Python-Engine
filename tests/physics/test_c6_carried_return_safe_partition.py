"""Finite integer-set oracles for cuts justified by whole-target exclusion."""
from dataclasses import replace
from fractions import Fraction as F
from itertools import product

import pytest

from tnfr.dynamics._euler_kernel import NodalRemainderState
from tnfr.physics import c6_carried_return as owner
from tnfr.physics.c6_carried_viability import C6CarriedForwardZone, _Work, _dbm_close


def _hull(points):
    values = tuple((*point, 0, 0, 0, 0, 0) for point in points)
    return tuple(tuple(max(point[i] - point[j] for point in values) for j in range(7)) for i in range(7))


def _members(bounds, coordinates=range(-4, 10)):
    if bounds is None:
        return set()
    return {(x, y) for x, y in product(coordinates, repeat=2)
            if all((x, y, 0, 0, 0, 0, 0)[i] - (x, y, 0, 0, 0, 0, 0)[j] <= bounds[i][j]
                   for i in range(7) for j in range(7))}


def _union(zones):
    return set().union(*(_members(zone) for zone in zones))


@pytest.mark.parametrize("bad_points", (
    ((0, 0),), ((2, 2),), ((3, 3),),
    tuple(product(range(-2, 3), repeat=2)), tuple(product(range(-3, 4), repeat=2)),
    tuple((x, y) for x, y in product(range(-2, 3), repeat=2) if 0 <= x-y <= 1),
))
def test_exact_integer_difference_matches_independent_grid_and_relative_basis(bad_points):
    bounds = _hull(product(range(-2, 3), repeat=2))
    bad = _hull(bad_points)
    pieces, basis = owner._return_safe_difference(bounds, bad, _Work(10_000))
    expected = _members(bounds) - _members(bad)
    observed = tuple(_members(piece) for piece in pieces)
    assert set().union(*observed) == expected
    assert sum(map(len, observed)) == len(expected)
    assert all(piece and _dbm_close(zone) == zone for zone, piece in zip(pieces, observed, strict=True))
    # The selected basis is relative to the source zone; its size need not be
    # globally minimum and its lexicographic decomposition need not be unique.
    if _members(bounds) & _members(bad):
        inside = {(x, y) for x, y in _members(bounds)
                  if all((x, y, 0, 0, 0, 0, 0)[i] - (x, y, 0, 0, 0, 0, 0)[j] <= bound
                         for i, j, bound in basis)}
        assert inside == _members(bounds) & _members(bad)


def test_center_hole_cannot_be_replaced_by_its_unchanged_hull():
    bounds, bad = _hull(product(range(-1, 2), repeat=2)), _hull(((0, 0),))
    pieces, _ = owner._return_safe_difference(bounds, bad, _Work(10_000))
    remaining = _union(pieces)
    assert len(remaining) == 8 and (0, 0) not in remaining
    assert _hull(remaining) == bounds
    assert (0, 0) in _members(_hull(remaining))


def test_sequential_subtractions_keep_exact_union_and_integer_boundary():
    bounds = _hull(product(range(-2, 3), repeat=2))
    forbidden = (_hull(((0, 0),)), _hull(((1, -1), (1, 0), (1, 1))), _hull(((0, 0), (1, 0))))
    pieces, expected = (bounds,), _members(bounds)
    work = _Work(100_000)
    for bad in forbidden:
        pieces = tuple(piece for zone in pieces for piece in owner._return_safe_difference(zone, bad, work)[0])
        expected -= _members(bad)
        assert _union(pieces) == expected
        assert sum(len(_members(piece)) for piece in pieces) == len(expected)
    # Exactly adjacent integer points remain: complementing <= b requires
    # >= b+1, not >= b, otherwise a forbidden boundary would survive.
    assert (0, 1) in expected and (1, 1) not in expected and (2, 1) in expected


@pytest.fixture(scope="module")
def memory():
    # A private algebra fixture, with no claim of a canonical pressure law.
    # Origin (0,0) returns to (4,0). A stationary row covers [4,6]x[-1,1],
    # including unreachable interior points that exact cuts must distinguish.
    rows = ((.4,) * 6, (.5,) * 6, (.6,) * 6)
    zero = _hull(((0, 0),))
    box = _hull(product(range(4, 7), range(-1, 2)))
    edges = (
        owner.C6CarriedReturnTransition(rows[0], None, rows[1], zero, None, _hull(((4, 0),)), (4, 0, 0, 0, 0, 0)),
        owner.C6CarriedReturnTransition(rows[1], None, rows[1], box, None, box, (0,) * 6),
    )
    terminal_specs = ((0, (0, 0), (0, 7)), (1, (4, 0), (4, 5)),
                      (1, (5, -1), (5, 4)), (1, (5, 1), (5, 6)), (1, (5, 0), (5, 5)))
    terminals = tuple(owner.C6CarriedReturnTransition(
        rows[source], None, rows[2], _hull((before,)), None, _hull((after,)),
        (after[0]-before[0], after[1]-before[1], 0, 0, 0, 0),
    ) for source, before, after in terminal_specs)
    zones = (C6CarriedForwardZone(rows[0], zero), C6CarriedForwardZone(rows[1], box),
             C6CarriedForwardZone(rows[2], _hull(after for _source, _before, after in terminal_specs)))
    state = NodalRemainderState(rows[0], (F(0),) * 6, .375, .625)
    envelope = owner.C6CarriedReturnEnvelope(
        None, state, 1., rows, ((4., 0., 0., 0., 0., 0.), (0.,) * 6, (0.,) * 6), F(1), state.exact_epi,
        (rows[2],), zones, zones, edges, terminals, 7, True, True, (), "fixed_point", 0, 0, 1, 3,
    )
    return owner._derive_return_memory_envelope(envelope, max_memory_work=1000, max_memory_arcs=100)


def _target(memory, row, *points):
    return (C6CarriedForwardZone(memory.return_envelope.epi_states[row], _hull(points)),)


def _derive(memory, groups=None, **changes):
    if groups is None:
        groups = (_target(memory, 1, (5, 0)),)
    kwargs = dict(exclusion_query_max_intersections=10_000, max_partition_pieces=100,
                  max_partition_construction_work=100_000, max_partition_arcs=1000, max_partition_work=100_000)
    kwargs.update(changes)
    return owner._derive_return_safe_partition(memory, groups, **kwargs)


@pytest.fixture(scope="module")
def partition(memory):
    return _derive(memory)


def test_verified_center_cut_preserves_every_other_endpoint_and_origin_image(memory, partition):
    assert partition.initialization_complete and partition.relation_complete
    assert all(query.origin_path_within_domain_excluded for query in partition.excluded_target_queries)
    for index, zone in enumerate(memory.retained_endpoint_zones):
        seeds = tuple(piece for (old_index, _ordinal), piece in zip(partition.vertices, partition.seed_zones, strict=True)
                      if old_index == index)
        assert _union(seeds) == _members(zone) - {(5, 0)}
        assert sum(len(_members(piece)) for piece in seeds) == len(_union(seeds))
    assert _union(partition.root_zones) == {(4, 0)}
    assert all(_members(after) <= _members(seed) for after, seed in zip(partition.retained_zones, partition.seed_zones, strict=True))
    assert (5, 0) not in _union(partition.retained_zones)
    for flag in ("conditional_invariance_certified", "conditional_boundedness_certified",
                 "future_runtime_certified", "asymptotic_convergence_certified"):
        assert getattr(partition, flag) is False


def test_two_disconnected_terminal_preimages_do_not_authorize_cutting_their_join(memory):
    # One closed terminal target has two disjoint admissible preimage pieces.
    # Remove the separate central terminal edge for this specific fixture.
    memory = replace(memory, return_envelope=replace(
        memory.return_envelope, intermediate_transitions=memory.return_envelope.intermediate_transitions[:-1],
    ))
    groups = (_target(memory, 2, (5, 4), (5, 6)),)
    partition = _derive(memory, groups)
    assert partition.initialization_complete and partition.relation_complete
    seeds = tuple(zone for (index, _ordinal), zone in zip(partition.vertices, partition.seed_zones, strict=True) if index == 1)
    expected = _members(memory.retained_endpoint_zones[1]) - {(5, -1), (5, 1)}
    assert _union(seeds) == expected and (5, 0) in _union(seeds)
    original_query = partition.excluded_target_queries[0]
    assert (5, 0) in _members(original_query.initial_endpoint_zones[1])
    assert original_query.origin_path_within_domain_excluded


@pytest.mark.parametrize("targets", ((0, ((0, 0),)), (1, ((4, 0),)), (2, ((0, 7),)), (2, ((4, 5),))))
def test_origin_sentinel_and_terminal_only_visits_survive_safe_partition(memory, partition, targets):
    row, points = targets
    supplied = _target(memory, row, *points)
    result = owner._derive_return_safe_partition_region_queries(partition, (supplied,), 10_000)
    query = result.queries[0]
    assert query.target_regions == supplied
    assert query.initialization_complete
    assert query.status == "origin_not_excluded"
    assert not query.origin_path_within_domain_excluded and not query.actual_origin_reachability_certified
    assert query.iterations[-1].origin_present


def test_cut_terminal_target_remains_whole_while_its_safe_preimage_is_empty(memory, partition):
    supplied = _target(memory, 2, (5, 5))
    query = owner._derive_return_safe_partition_region_queries(partition, (supplied,), 10_000).queries[0]
    assert query.target_regions == supplied
    assert query.origin_path_within_domain_excluded and not query.actual_origin_reachability_certified


@pytest.mark.parametrize("change", ("origin", "unproved_resource"))
def test_unverified_positive_targets_cannot_authorize_partition(memory, change):
    partition = (_derive(memory, (_target(memory, 0, (0, 0)),)) if change == "origin"
                 else _derive(memory, exclusion_query_max_intersections=1))
    assert partition.status == "exclusion_not_verified"
    assert not partition.initialization_complete and not partition.relation_complete
    assert partition.vertices == partition.seed_zones == partition.partition_arcs == ()
    assert not all(query.origin_path_within_domain_excluded for query in partition.excluded_target_queries)
    result = owner._derive_return_safe_partition_region_queries(partition, (_target(memory, 2, (4, 5)),), 10_000)
    assert all(not query.origin_path_within_domain_excluded for query in result.queries)


@pytest.mark.parametrize("changes", ({"max_partition_construction_work": 1}, {"max_partition_pieces": 1}))
def test_partial_partition_initialization_is_discarded_atomically(memory, changes):
    partition = _derive(memory, **changes)
    assert not partition.initialization_complete and not partition.relation_complete
    assert partition.vertices == partition.seed_zones == partition.partition_arcs == ()
    assert not partition.conditional_boundedness_certified


def test_partial_arc_graph_cannot_be_used_as_a_complete_query_operator(memory):
    partition = _derive(memory, max_partition_arcs=1)
    assert not partition.relation_complete and not partition.partition_arcs
    query = owner._derive_return_safe_partition_region_queries(partition, (_target(memory, 2, (4, 5)),), 10_000).queries[0]
    assert not query.origin_path_within_domain_excluded


def test_partial_forward_descent_preserves_the_safe_seed_cover(memory):
    partition = _derive(memory, max_partition_work=1)
    assert partition.initialization_complete and partition.relation_complete
    assert all(_members(after) <= _members(seed) for after, seed in zip(partition.retained_zones, partition.seed_zones, strict=True))
    assert (5, 0) not in _union(partition.retained_zones)
    assert _union(partition.root_zones) == {(4, 0)}
    assert not partition.conditional_boundedness_certified


def _enumerated_graph(partition):
    memory = partition.memory_envelope
    edges = memory.return_envelope.return_relation
    zones = tuple(map(_members, partition.retained_zones))
    graph = []
    for a, (i, _ordinal) in enumerate(partition.vertices):
        for b, (j, _other) in enumerate(partition.vertices):
            if edges[i].target_epi != edges[j].source_epi:
                continue
            shift = edges[j].shift[:2]
            for point in zones[a] & _members(memory.source_guards[j]):
                image = (point[0]+shift[0], point[1]+shift[1])
                if image in zones[b]:
                    graph.append((a, b, point, image))
    for b, (j, _ordinal) in enumerate(partition.vertices):
        edge = edges[j]
        if edge.source_epi == memory.return_envelope.state.epi and (0, 0) in _members(memory.source_guards[j]):
            image = edge.shift[:2]
            if image in zones[b]:
                graph.append((len(zones), b, (0, 0), image))
    return graph


def test_partition_graph_is_complete_against_independent_point_enumeration(partition):
    observed, complete, _ = owner._return_safe_partition_query_graph(partition, 10_000)
    assert complete
    actual = {(a, b, p, (p[0]+shift[0], p[1]+shift[1]))
              for a, b, guard, shift in observed for p in _members(guard)}
    assert actual == set(_enumerated_graph(partition))


def _query_oracle(partition, targets):
    envelope = partition.memory_envelope.return_envelope
    rows = tuple(envelope.return_relation[i].target_epi for i, _ in partition.vertices) + (envelope.state.epi,)
    zones = tuple(map(_members, partition.retained_zones)) + ({(0, 0)},)
    current = []
    for row, points in zip(rows, zones, strict=True):
        selected = set()
        for target in targets:
            members = _members(target.bounds)
            if row == target.epi:
                selected |= points & members
            for edge in envelope.intermediate_transitions:
                if edge.source_epi == row and edge.target_epi == target.epi:
                    selected |= {point for point in points & _members(edge.source_guard)
                                 if (point[0]+edge.shift[0], point[1]+edge.shift[1]) in members}
        current.append(_hull(selected) if selected else None)
    graph = _enumerated_graph(partition)
    layers = [tuple(current)]
    for _ in range(10):
        if (0, 0) in _members(current[-1]):
            return layers, "origin_not_excluded"
        if not any(zone is not None for zone in current):
            return layers, "empty_complete_layer"
        following = [set() for _ in current]
        for a, b, before, after in graph:
            if after in _members(current[b]):
                following[a].add(before)
        following = tuple(_hull(points) if points else None for points in following)
        layers.append(following)
        if following == tuple(current):
            return layers, "stationary_complete_layer"
        current = following
    raise AssertionError("the finite oracle failed to terminate")


@pytest.mark.parametrize("row,point", ((0, (0, 0)), (1, (4, 0)), (1, (6, -1)),
                                       (2, (0, 7)), (2, (4, 5)), (2, (5, 5))))
def test_complete_query_layers_match_independent_finite_state_oracle(memory, partition, row, point):
    targets = _target(memory, row, point)
    layers, status = _query_oracle(partition, targets)
    query = owner._derive_return_safe_partition_region_queries(partition, (targets,), 10_000).queries[0]
    assert query.status == status
    assert query.initial_endpoint_zones == layers[0]
    assert query.retained_endpoint_zones == layers[-1]
    assert len(query.iterations) == len(layers)
    assert tuple(step.zone_count for step in query.iterations) == tuple(sum(zone is not None for zone in layer) for layer in layers)
    assert query.origin_path_within_domain_excluded == (status != "origin_not_excluded")


def test_public_owner_rebuilds_canonical_diffusion_and_rechecks_proposed_exclusion():
    from tnfr.physics.c6_pressure_lattice import derive_c6_pressure_lattice

    delta, origin, fixed = F(1, 2**54), (-20,) * 6, (-19,) * 6
    chain = ((-20, -20, -20, -20, -19, -19), (-19, -20, -20, -19, -20, -20),
             (-21, -19, -19, -21, -19, -19), (-17, -21, -21, -17, -21, -21))

    def row(point):
        return tuple(float(F(1, 2) + value*delta) for value in point)

    def target(point):
        vector = tuple(value-start for value, start in zip(point, origin, strict=True)) + (0,)
        return (C6CarriedForwardZone(row(point), tuple(tuple(a-b for b in vector) for a in vector)),)

    reference = derive_c6_pressure_lattice(phase=(0.,) * 6, epi_weight=1., phase_weight=1.)
    state = NodalRemainderState(row(origin), (F(0),) * 6, .375, .625)
    result = owner.derive_c6_carried_return_safe_partition_region_exclusions(
        reference, state=state, epi_states=tuple(map(row, (origin, fixed, *chain))), timestep=2.,
        transient_epi_states=(row(chain[1]),), excluded_target_region_groups=(target(fixed),),
        target_region_groups=(target(origin), target(fixed), target(chain[1])),
    )
    assert result.partition.memory_envelope.return_envelope.state == state
    assert result.partition.initialization_complete and result.partition.relation_complete
    assert result.partition.excluded_target_queries[0].origin_path_within_domain_excluded
    assert not result.queries[0].origin_path_within_domain_excluded
    assert all(query.origin_path_within_domain_excluded for query in result.queries[1:])
    assert all(not query.actual_origin_reachability_certified for query in result.queries)
    assert not result.partition.conditional_boundedness_certified
