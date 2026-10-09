"""Finite set oracles for advisory priority and authenticated history covers."""

from dataclasses import replace
from fractions import Fraction as F
from itertools import product

import pytest

from tnfr.dynamics._euler_kernel import NodalRemainderState
from tnfr.physics import c6_carried_return as owner
from tnfr.physics.c6_carried_viability import C6CarriedForwardZone
from tnfr.physics.c6_pressure_lattice import derive_c6_pressure_lattice


def _hull(points):
    values = tuple((*point, 0, 0, 0, 0, 0) for point in points)
    return tuple(
        tuple(max(point[i] - point[j] for point in values) for j in range(7))
        for i in range(7)
    )


def _members(bounds):
    if bounds is None:
        return set()
    return {
        (x, y)
        for x, y in product(range(-1, 10), range(-2, 9))
        if all(
            (x, y, 0, 0, 0, 0, 0)[i] - (x, y, 0, 0, 0, 0, 0)[j] <= bounds[i][j]
            for i in range(7)
            for j in range(7)
        )
    }


@pytest.fixture(scope="module")
def memory():
    # Private finite transition algebra. No canonical nodal-source assertion.
    # The actual origin reaches x=4,5,6; a stationary outer box also retains
    # unreachable x=7,8 and y!=0 states that valid covers may remove.
    rows = ((0.4,) * 6, (0.5,) * 6, (0.6,) * 6)
    zero = _hull(((0, 0),))
    wide = _hull(product(range(4, 9), range(-1, 2)))
    specs = (
        (0, ((0, 0),), (4, 0)),
        (1, ((4, 0), (5, 0)), (1, 0)),
        (1, tuple(product(range(4, 9), range(-1, 2))), (0, 0)),
    )
    edges = tuple(
        owner.C6CarriedReturnTransition(
            rows[source],
            None,
            rows[1],
            _hull(points),
            None,
            _hull((x + shift[0], y + shift[1]) for x, y in points),
            (*shift, 0, 0, 0, 0),
        )
        for source, points, shift in specs
    )
    terminals = tuple(
        owner.C6CarriedReturnTransition(
            rows[source],
            None,
            rows[2],
            _hull((before,)),
            None,
            _hull((after,)),
            (after[0] - before[0], after[1] - before[1], 0, 0, 0, 0),
        )
        for source, before, after in (
            (0, (0, 0), (0, 7)),
            (1, (5, 0), (5, 5)),
            (1, (8, 0), (8, 5)),
        )
    )
    zones = (
        C6CarriedForwardZone(rows[0], zero),
        C6CarriedForwardZone(rows[1], wide),
        C6CarriedForwardZone(rows[2], _hull(((0, 7), (5, 5), (8, 5)))),
    )
    state = NodalRemainderState(rows[0], (F(0),) * 6, 0.375, 0.625)
    envelope = owner.C6CarriedReturnEnvelope(
        None,
        state,
        1.0,
        rows,
        ((4.0, 0.0, 0.0, 0.0, 0.0, 0.0), (0.0,) * 6, (0.0,) * 6),
        F(1),
        state.exact_epi,
        (rows[2],),
        zones,
        zones,
        edges,
        terminals,
        len(edges) + len(terminals),
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
        envelope, max_memory_work=10_000, max_memory_arcs=1000
    )


def _target(memory, row=2, point=(8, 5)):
    return (
        C6CarriedForwardZone(memory.return_envelope.epi_states[row], _hull((point,))),
    )


def _derive(memory, **changes):
    arguments = dict(
        exclusion_query_max_intersections=10_000,
        max_partition_pieces=1000,
        max_partition_construction_work=100_000,
        max_partition_arcs=5000,
        max_partition_work=100_000,
        max_cover_work=100_000,
    )
    arguments.update(changes)
    return owner._derive_return_safe_cover(memory, (_target(memory),), **arguments)


def _exact_graph(partition):
    """Enumerate every seed-admitted point edge without the stored arc list."""
    memory = partition.memory_envelope
    edges = memory.return_envelope.return_relation
    zones = tuple(map(_members, partition.seed_zones))
    graph = set()
    for a, (i, _) in enumerate(partition.vertices):
        for b, (j, _) in enumerate(partition.vertices):
            edge = edges[j]
            if edges[i].target_epi != edge.source_epi:
                continue
            for point in zones[a] & _members(memory.source_guards[j]):
                image = (point[0] + edge.shift[0], point[1] + edge.shift[1])
                if image in zones[b]:
                    graph.add((a, b, point, image))
    return graph


def _reachable(partition):
    current = {
        (i, point)
        for i, root in enumerate(partition.root_zones)
        for point in _members(root)
    }
    graph = _exact_graph(partition)
    while True:
        following = current | {
            (b, after) for a, b, before, after in graph if (a, before) in current
        }
        if following == current:
            return current
        current = following


def _image_oracle(partition, zones):
    members = tuple(map(_members, zones))
    images = [set(_members(root)) for root in partition.root_zones]
    for a, b, before, after in _exact_graph(partition):
        if before in members[a]:
            images[b].add(after)
    return tuple(_hull(points) if points else None for points in images)


def _reachable_cover(partition):
    reached = _reachable(partition)
    return tuple(
        (
            _hull(point for index, point in reached if index == i)
            if any(index == i for index, _ in reached)
            else None
        )
        for i in range(len(partition.vertices))
    )


def test_private_finite_fixture_has_a_real_successor_beyond_the_root(memory):
    cover = _derive(memory)
    reached = _reachable(cover.partition)
    assert {point for _, point in reached} == {(4, 0), (5, 0), (6, 0)}
    assert {
        point for root in cover.partition.root_zones for point in _members(root)
    } == {(4, 0)}
    actual = {
        (a, b, point, (point[0] + shift[0], point[1] + shift[1]))
        for a, b, guard, shift in cover.partition.partition_arcs
        for point in _members(guard)
    }
    assert actual == _exact_graph(cover.partition)


@pytest.fixture(scope="module")
def cover(memory):
    return _derive(memory)


def test_default_fifo_preserves_previous_partition_payload_and_has_no_classification_charge(
    memory, cover
):
    previous = owner._derive_return_safe_partition(
        memory,
        (_target(memory),),
        exclusion_query_max_intersections=10_000,
        max_partition_pieces=1000,
        max_partition_construction_work=100_000,
        max_partition_arcs=5000,
        max_partition_work=100_000,
    )
    assert cover.partition == previous
    assert cover.schedule.policy == "fifo" and cover.schedule.classification_work == 0
    assert cover.check.certified and cover.domain_confined_origin_histories_covered
    assert cover.check.image_zones == _image_oracle(previous, cover.retained_zones)
    for name in (
        "conditional_invariance_certified",
        "conditional_boundedness_certified",
        "actual_origin_reachability_certified",
        "future_runtime_certified",
        "asymptotic_convergence_certified",
    ):
        assert getattr(cover, name) is False


def test_null_hints_and_missing_memory_hints_charge_every_seed_without_cutting(
    memory, cover
):
    supplied = ((0, None),)
    result = _derive(memory, priority_memory_regions=supplied)
    assert result.schedule.priority_memory_regions == supplied
    assert result.schedule.validation_work == 1
    assert result.schedule.classification_work == len(result.partition.vertices)
    assert result.schedule.priority_flags == (False,) * len(result.partition.vertices)
    assert result.partition.seed_zones == cover.partition.seed_zones
    assert result.partition.root_zones == cover.partition.root_zones
    assert result.partition.partition_arcs == cover.partition.partition_arcs
    assert result.retained_zones == cover.retained_zones


@pytest.mark.parametrize("maximum", (1, 3, 5, 6, 9, 13, 30))
def test_advisory_priorities_preserve_all_finite_origin_paths_under_bounded_work(
    memory, cover, maximum
):
    wide = memory.retained_endpoint_zones[2]
    result = _derive(
        memory, priority_memory_regions=((2, wide),), max_partition_work=maximum
    )
    partition = result.partition
    assert partition.seed_zones == cover.partition.seed_zones
    assert partition.bad_pieces == cover.partition.bad_pieces
    assert partition.partition_arcs == cover.partition.partition_arcs
    assert partition.root_zones == cover.partition.root_zones
    assert all(
        point in _members(partition.retained_zones[index])
        for index, point in _reachable(partition)
    )
    assert result.schedule.classification_work <= maximum
    if result.schedule.classification_complete:
        flags = tuple(
            bool(_members(seed) & _members(wide)) if index == 2 else False
            for (index, _), seed in zip(
                partition.vertices, partition.seed_zones, strict=True
            )
        )
        assert result.schedule.priority_flags == flags
        assert partition.forward_work <= maximum
        pending = result.schedule.priority_pending + result.schedule.ordinary_pending
        assert len(set(pending)) == len(pending)
        assert all(flags[v] for v in result.schedule.priority_pending)
        assert all(not flags[v] for v in result.schedule.ordinary_pending)
        assert partition.pending_vertices == pending
        assert result.check.certified
    else:
        assert result.schedule.status == "priority_resource_limit"
        assert result.schedule.priority_flags == ()
        assert result.schedule.ordinary_pending == tuple(range(len(partition.vertices)))
        assert not result.check.certified and result.retained_zones == ()


def test_priority_classification_keeps_hint_order_and_charges_null_visits(memory):
    wide = memory.retained_endpoint_zones[2]
    result = _derive(memory, priority_memory_regions=((2, None), (2, wide), (2, wide)))
    n2 = sum(index == 2 for index, _ in result.partition.vertices)
    assert result.schedule.classification_work == len(result.partition.vertices) + n2
    assert result.schedule.validation_work == 5
    assert result.schedule.priority_memory_regions == ((2, None), (2, wide), (2, wide))


def _algebra_partition(partition):
    # Only a private arithmetic context: never submitted as public provenance.
    wide = _hull(product(range(3), range(3)))
    guard = _hull(product(range(2), range(3)))
    zero = _hull(((0, 0),))
    arcs = ((0, 1, guard, (1, 0, 0, 0, 0, 0)),)
    return replace(
        partition,
        vertices=((0, 0), (1, 0)),
        seed_zones=(wide, wide),
        root_zones=(zero, None),
        retained_zones=(wide, wide),
        partition_arcs=arcs,
    )


@pytest.mark.parametrize("flags", ((False, True), (True, False)))
def test_exact_shared_worklist_matches_finite_reachability_for_every_small_budget(
    cover, flags
):
    partition = _algebra_partition(cover.partition)
    wide = partition.seed_zones[0]
    reachable = {(0, (0, 0)), (1, (1, 0))}
    finals = []
    for maximum in range(2, 14):
        result = owner._return_partition_descent(
            partition.seed_zones,
            partition.root_zones,
            partition.partition_arcs,
            maximum,
            priority_flags=flags,
            initial_work=2,
        )
        assert all(
            point in _members(result.retained_zones[index])
            for index, point in reachable
        )
        assert result.work <= maximum
        assert set(result.pending) == set(result.priority_pending) | set(
            result.ordinary_pending
        )
        assert len(result.pending) == len(set(result.pending))
        finals.append(result)
    assert tuple(map(_members, finals[-1].retained_zones)) == ({(0, 0)}, {(1, 0)})
    if flags == (False, True):
        # The first urgent visit costs two: one incoming edge and seed clip.
        # A remaining budget of one must not commit its root-only partial image.
        stopped = finals[1]
        assert stopped.work == 2 and stopped.visits == 0
        assert stopped.retained_zones == (wide, wide)
        assert stopped.priority_pending == (1,) and stopped.ordinary_pending == (0,)


def test_cover_gate_requires_complete_image_not_merely_roots_and_safe_seeds(cover):
    partition = cover.partition
    candidate = partition.root_zones
    assert all(
        _members(root) <= _members(zone) <= _members(seed)
        for root, zone, seed in zip(
            partition.root_zones, candidate, partition.seed_zones, strict=True
        )
    )
    expected = _image_oracle(partition, candidate)
    assert any(
        _members(image) - _members(zone)
        for image, zone in zip(expected, candidate, strict=True)
    )
    check = owner._check_return_safe_cover(partition, candidate, 10_000)
    assert check.status == "image_not_included" and not check.certified
    assert check.image_zones == expected and check.violating_vertices


def test_gate_accepts_a_strict_image_subset_and_same_index_cover_intersection(cover):
    partition = _algebra_partition(cover.partition)
    left = (_hull(product(range(2), range(3))), _hull(product(range(1, 3), range(3))))
    right = (_hull(product(range(3), range(2))), _hull(product(range(3), range(2))))
    checks = [
        owner._check_return_safe_cover(partition, value, 10_000)
        for value in (left, right)
    ]
    assert all(
        check.certified and check.image_zones != check.candidate_zones
        for check in checks
    )
    intersection = tuple(
        _hull(_members(a) & _members(b)) for a, b in zip(left, right, strict=True)
    )
    check = owner._check_return_safe_cover(partition, intersection, 10_000)
    assert check.certified
    expected = ({(0, 0), (0, 1), (1, 0), (1, 1)}, {(1, 0), (1, 1), (2, 0), (2, 1)})
    assert tuple(map(_members, intersection)) == expected
    wrong_indices = tuple(reversed(intersection))
    assert not owner._check_return_safe_cover(
        partition, wrong_indices, 10_000
    ).certified


def test_exact_reachable_candidate_is_verified_without_replacing_the_discovery_partition(
    memory, cover
):
    candidate = _reachable_cover(cover.partition)
    result = _derive(memory, candidate_retained_zones=candidate)
    assert (
        result.schedule.policy == "candidate_only"
        and result.schedule.status == "not_run"
    )
    assert result.schedule.classification_work == 0
    assert result.partition.completed_visits == result.partition.forward_work == 0
    assert result.check.certified and result.retained_zones == candidate
    assert result.check.image_zones == _image_oracle(result.partition, candidate)
    assert result.partition.retained_zones == result.partition.seed_zones


@pytest.mark.parametrize("kind", ("root", "seed", "incomplete_source"))
def test_failed_cover_inclusion_never_promotes_the_candidate(cover, kind):
    partition = cover.partition
    candidate = list(partition.seed_zones)
    if kind == "root":
        index = next(
            i for i, root in enumerate(partition.root_zones) if root is not None
        )
        candidate[index] = None
        expected = "root_not_included"
    elif kind == "seed":
        candidate[0] = _hull(((4, 0), (3, 0)))
        expected = "outside_safe_seeds"
    else:
        partition = replace(partition, relation_complete=False)
        expected = "source_construction_incomplete"
    check = owner._check_return_safe_cover(partition, tuple(candidate), 10_000)
    assert not check.certified and check.status == expected
    forged = replace(cover, partition=partition, check=check)
    assert (
        forged.retained_zones == ()
        and not forged.domain_confined_origin_histories_covered
    )


def test_cover_budget_abstains_in_the_middle_of_the_image_and_final_inclusion_check(
    cover,
):
    partition = cover.partition
    complete = owner._check_return_safe_cover(partition, partition.seed_zones, 10_000)
    for maximum in (1, 4 * len(partition.vertices) + 1, complete.work - 1):
        stopped = owner._check_return_safe_cover(
            partition, partition.seed_zones, maximum
        )
        assert stopped.status == "cover_resource_limit" and not stopped.certified
        assert stopped.work == maximum and stopped.image_zones == ()


def _malformed_bounds(kind):
    rows = [list(row) for row in _hull(((0, 0),))]
    rows[0][1] = {"bool": False, "float": 0.0, "infinite": float("inf"), "unclosed": 1}[
        kind
    ]
    return tuple(map(tuple, rows))


@pytest.mark.parametrize("kind", ("bool", "float", "infinite", "unclosed"))
def test_candidate_and_hint_dbms_require_closed_exact_integer_geometry(
    memory, cover, kind
):
    bounds = _malformed_bounds(kind)
    with pytest.raises((TypeError, ValueError)):
        _derive(memory, priority_memory_regions=((0, bounds),))
    candidate = (bounds,) + cover.partition.seed_zones[1:]
    with pytest.raises((TypeError, ValueError)):
        owner._check_return_safe_cover(cover.partition, candidate, 10_000)


@pytest.mark.parametrize("candidate", ([], (), (None,), "bounds"))
def test_candidate_shape_is_checked_against_canonical_vertex_indexing(cover, candidate):
    with pytest.raises(ValueError, match="candidate"):
        owner._check_return_safe_cover(cover.partition, candidate, 10_000)


@pytest.mark.parametrize(
    "regions",
    (
        [],
        "unbounded",
        ((True, None),),
        ((1.0, None),),
        ((-1, None),),
        ((0,),),
        ([0, None],),
        ((0, ((0,),)),),
    ),
)
def test_malformed_priority_hints_reject_before_source_reconstruction(
    monkeypatch, regions
):
    def forbidden(*_args, **_kwargs):
        raise AssertionError(
            "invalid hints must be rejected before canonical source reconstruction"
        )

    monkeypatch.setattr(owner, "derive_c6_carried_return_memory_envelope", forbidden)
    with pytest.raises((TypeError, ValueError)):
        owner.derive_c6_carried_return_safe_cover(
            None,
            state=None,
            epi_states=(),
            timestep=1.0,
            transient_epi_states=(),
            excluded_target_region_groups=(),
            priority_memory_regions=regions,
        )


def test_non_tuple_hint_stream_is_never_materialized():
    class Unbounded:
        def __iter__(self):
            raise AssertionError("an unbounded hint stream must not be consumed")

    with pytest.raises(ValueError, match="tuple"):
        owner.derive_c6_carried_return_safe_cover(
            None,
            state=None,
            epi_states=(),
            timestep=1.0,
            transient_epi_states=(),
            excluded_target_region_groups=(),
            priority_memory_regions=Unbounded(),
        )


@pytest.mark.parametrize(
    "changes",
    (
        {"max_priority_regions": True},
        {"max_priority_regions": 0},
        {"max_cover_work": False},
        {"max_cover_work": 1.0},
        {"max_priority_regions": 1, "priority_memory_regions": ((0, None), (0, None))},
    ),
)
def test_new_resource_limits_reject_noninteger_boolean_or_overfull_inputs(
    memory, changes
):
    with pytest.raises(ValueError):
        _derive(memory, **changes)


def test_out_of_range_priority_memory_and_candidate_hint_mix_are_rejected(
    memory, cover
):
    with pytest.raises(ValueError, match="canonical return"):
        _derive(
            memory,
            priority_memory_regions=(
                (len(memory.return_envelope.return_relation), None),
            ),
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        _derive(
            memory,
            priority_memory_regions=((0, None),),
            candidate_retained_zones=cover.partition.seed_zones,
        )


@pytest.mark.parametrize(
    "row,point", ((0, (0, 0)), (1, (6, 0)), (2, (5, 5)), (2, (0, 7)))
)
def test_complete_queries_preserve_direct_terminal_and_origin_sentinel_visits(
    memory, cover, row, point
):
    targets = _target(memory, row, point)
    result = owner._derive_return_safe_cover_region_queries(cover, (targets,), 10_000)
    query = result.queries[0]
    assert result.query_relation_complete and query.initialization_complete
    assert query.target_regions == targets
    assert query.status == "origin_not_excluded" and query.iterations[-1].origin_present
    assert (
        not query.origin_path_within_domain_excluded
        and not query.actual_origin_reachability_certified
    )


def test_failed_gate_and_partial_query_budget_cannot_authorize_exclusion(memory, cover):
    supplied = _target(memory, 2, (5, 5))
    failed = _derive(memory, candidate_retained_zones=cover.partition.root_zones)
    assert failed.check.status == "image_not_included"
    for checked, maximum in ((failed, 10_000), (cover, 1)):
        result = owner._derive_return_safe_cover_region_queries(
            checked, (supplied,), maximum
        )
        assert not result.query_relation_complete
        assert not result.queries[0].origin_path_within_domain_excluded
        assert not result.queries[0].initialization_complete


@pytest.fixture(scope="module")
def canonical():
    row = (0.5,) * 6
    reference = derive_c6_pressure_lattice(
        phase=(0.0,) * 6, epi_weight=1.0, phase_weight=1.0
    )
    state = NodalRemainderState(row, (F(0),) * 6, 0.375, 0.625)
    excluded = (C6CarriedForwardZone(row, _hull(((1, 0),))),)
    args = dict(
        state=state,
        epi_states=(row,),
        timestep=0.0625,
        transient_epi_states=(),
        excluded_target_region_groups=(excluded,),
        max_partition_work=10_000,
    )
    return reference, args, owner.derive_c6_carried_return_safe_cover(reference, **args)


def test_public_boundary_rebuilds_forged_source_caches(canonical):
    reference, args, result = canonical
    forged = replace(reference, sources=(17.0,) * 6, rows=(), epi_quantum=F(7))
    rebuilt = owner.derive_c6_carried_return_safe_cover(forged, **args)
    assert result.check.certified and rebuilt == result
    assert rebuilt.partition.memory_envelope.return_envelope.reference == reference


@pytest.mark.parametrize(
    "field", ("partition", "partition_arcs", "seed_zones", "root_zones", "cover")
)
def test_public_query_rejects_external_geometry_and_forged_certificate_dataclasses(
    canonical, field
):
    reference, args, result = canonical
    forged = replace(
        result,
        partition=replace(
            result.partition,
            partition_arcs=(),
            retained_zones=(None,) * len(result.partition.vertices),
        ),
    )
    supplied = (
        forged if field == "cover" else forged.partition if field == "partition" else ()
    )
    target = (C6CarriedForwardZone(args["state"].epi, _hull(((0, 0),))),)
    with pytest.raises(TypeError, match="unexpected keyword"):
        owner.derive_c6_carried_return_safe_cover_region_exclusions(
            reference,
            **args,
            target_region_groups=(target,),
            **{field: supplied},
        )


def test_public_candidate_is_rechecked_against_rebuilt_root_and_query_observations(
    canonical,
):
    reference, args, result = canonical
    targets = (C6CarriedForwardZone(args["state"].epi, _hull(((0, 0),))),)
    original = owner.derive_c6_carried_return_safe_cover_region_exclusions(
        reference,
        **args,
        candidate_retained_zones=result.retained_zones,
        target_region_groups=(targets,),
    )
    assert original.cover.check.certified
    assert original.queries[0].status == "origin_not_excluded"
    forged = owner.derive_c6_carried_return_safe_cover_region_exclusions(
        reference,
        **args,
        candidate_retained_zones=(None,) * len(result.partition.vertices),
        target_region_groups=(targets,),
    )
    assert (
        not forged.cover.check.certified
        and not forged.queries[0].origin_path_within_domain_excluded
    )


@pytest.mark.parametrize(
    "options",
    (
        {"priority_flags": [False, True]},
        {"priority_flags": (0, 1)},
        {"priority_flags": (False,)},
        {"initial_work": True},
        {"initial_work": -1},
        {"initial_work": 11},
        {"initial_work": 1.0},
    ),
)
def test_shared_scheduler_rejects_malformed_flags_and_classification_work(
    cover, options
):
    partition = _algebra_partition(cover.partition)
    with pytest.raises(ValueError):
        owner._return_partition_descent(
            partition.seed_zones,
            partition.root_zones,
            partition.partition_arcs,
            10,
            **options,
        )


def test_complete_relation_does_not_authorize_partial_target_initialization(
    memory, cover
):
    group = (
        _target(memory, 0, (0, 0))
        + _target(memory, 1, (6, 0))
        + _target(memory, 2, (5, 5))
    )
    complete = owner._derive_return_safe_cover_region_queries(cover, (group,), 10_000)
    maximum = complete.query_relation_intersections
    stopped = owner._derive_return_safe_cover_region_queries(cover, (group,), maximum)
    query = stopped.queries[0]
    assert stopped.query_relation_complete and not query.initialization_complete
    assert query.target_regions == group and query.intersections == maximum
    assert query.status == "memory_query_initialization_resource_limit"
    assert query.initial_endpoint_zones == query.retained_endpoint_zones == ()
    assert not query.origin_path_within_domain_excluded


def test_partial_backward_layer_retains_only_the_last_complete_image(memory, cover):
    targets = _target(memory, 1, (6, 0))
    first = owner._derive_return_safe_cover_region_queries(cover, (targets,), 21)
    later = owner._derive_return_safe_cover_region_queries(cover, (targets,), 24)
    a, b = first.queries[0], later.queries[0]
    assert first.query_relation_complete and later.query_relation_complete
    assert a.initialization_complete and b.initialization_complete
    assert a.status == b.status == "resource_limit"
    assert a.intersections == 21 and b.intersections == 24
    assert (
        a.iterations == b.iterations
        and a.retained_endpoint_zones == b.retained_endpoint_zones
    )
    assert (
        not a.origin_path_within_domain_excluded
        and not b.origin_path_within_domain_excluded
    )
    complete = owner._derive_return_safe_cover_region_queries(cover, (targets,), 10_000)
    assert complete.queries[0].status == "origin_not_excluded"


def test_incomplete_canonical_relation_never_creates_a_cover_or_query_certificate(
    memory,
):
    result = _derive(memory, max_partition_arcs=1)
    assert (
        not result.partition.relation_complete and result.partition.partition_arcs == ()
    )
    assert result.check.status == "source_construction_incomplete"
    assert (
        not result.domain_confined_origin_histories_covered
        and result.retained_zones == ()
    )
    query = owner._derive_return_safe_cover_region_queries(
        result, (_target(memory),), 10_000
    ).queries[0]
    assert not query.origin_path_within_domain_excluded
