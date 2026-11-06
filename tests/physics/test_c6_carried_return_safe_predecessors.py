"""Finite path oracles for exact excluded-memory predecessor cuts."""

from dataclasses import asdict
from fractions import Fraction as F
from itertools import product

import pytest

from tnfr.dynamics._euler_kernel import NodalRemainderState
from tnfr.physics import c6_carried_return as owner
from tnfr.physics.c6_carried_viability import C6CarriedForwardZone


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


def _memory(branches=False):
    # Private integer transition algebra, not a canonical nodal realization.
    # The origin stays at (4,0); the separate x>=5 component is unreachable.
    rows = ((0.4,) * 6, (0.5,) * 6, (0.6,) * 6)
    box = _hull(product(range(4, 9), range(-1, 2)))
    specs = (
        (0, ((0, 0),), (4, 0)),
        (1, tuple(product(range(4, 9), range(-1, 2))), (0, 0)),
    )
    if branches:
        specs += ((1, ((7, -1),), (1, 1)), (1, ((7, 1),), (1, -1)))
    else:
        specs += ((1, ((5, 0), (6, 0), (7, 0)), (1, 0)),)
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
    terminal_specs = ((0, (0, 0), (0, 7)), (1, (4, 0), (4, 5)), (1, (8, 0), (8, 5)))
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
        for source, before, after in terminal_specs
    )
    zones = (
        C6CarriedForwardZone(rows[0], _hull(((0, 0),))),
        C6CarriedForwardZone(rows[1], box),
        C6CarriedForwardZone(
            rows[2], _hull(after for _source, _before, after in terminal_specs)
        ),
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


@pytest.fixture(scope="module")
def memory():
    return _memory()


def _target(memory, row=2, point=(8, 5)):
    return (
        C6CarriedForwardZone(memory.return_envelope.epi_states[row], _hull((point,))),
    )


def _derive(memory, depth=1, groups=None, **changes):
    kwargs = dict(
        exclusion_query_max_intersections=10_000,
        max_partition_pieces=1000,
        max_partition_construction_work=100_000,
        max_partition_arcs=5000,
        max_partition_work=100_000,
    )
    kwargs.update(changes)
    return owner._derive_return_safe_partition(
        memory,
        groups or (_target(memory),),
        excluded_predecessor_depth=depth,
        **kwargs,
    )


def _exact_predecessor_oracle(memory, targets, depth):
    envelope = memory.return_envelope
    edges = envelope.return_relation
    zones = tuple(map(_members, memory.retained_endpoint_zones))
    bad = [set() for _ in zones]
    for i, points in enumerate(zones):
        row = edges[i].target_epi
        for target in targets:
            if row == target.epi:
                bad[i] |= points & _members(target.bounds)
            for terminal in envelope.intermediate_transitions:
                if terminal.source_epi == row and terminal.target_epi == target.epi:
                    bad[i] |= {
                        p
                        for p in points & _members(terminal.source_guard)
                        if (p[0] + terminal.shift[0], p[1] + terminal.shift[1])
                        in _members(target.bounds)
                    }
    total, layers = [set(points) for points in bad], [tuple(map(set, bad))]
    for _ in range(depth):
        following = [set() for _ in zones]
        for i, points in enumerate(zones):
            for j, edge in enumerate(edges):
                if edges[i].target_epi == edge.source_epi:
                    following[i] |= {
                        p
                        for p in points & _members(memory.source_guards[j])
                        if (p[0] + edge.shift[0], p[1] + edge.shift[1]) in bad[j]
                    }
        for old, added in zip(total, following, strict=True):
            old.update(added)
        layers.append(tuple(map(set, following)))
        bad = following
    return tuple(total), tuple(layers)


@pytest.mark.parametrize("depth", (0, 1, 2, 3, 8))
def test_complete_predecessor_cuts_match_independent_finite_path_enumeration(
    memory, depth
):
    partition = _derive(memory, depth)
    assert partition.initialization_complete and partition.relation_complete
    expected, _layers = _exact_predecessor_oracle(memory, _target(memory), depth)
    observed = [set() for _ in expected]
    for piece in partition.bad_pieces:
        observed[piece.memory_index].update(_members(piece.bounds))
    assert tuple(observed) == expected
    for index, old in enumerate(memory.retained_endpoint_zones):
        seeds = tuple(
            zone
            for (i, _ordinal), zone in zip(
                partition.vertices, partition.seed_zones, strict=True
            )
            if i == index
        )
        union = set().union(*map(_members, seeds))
        assert union == _members(old) - expected[index]
        assert sum(len(_members(zone)) for zone in seeds) == len(union)
    assert set().union(*map(_members, partition.root_zones)) == {(4, 0)}
    assert (
        not partition.conditional_boundedness_certified
        and not partition.future_runtime_certified
    )
    if depth:
        assert (
            partition.predecessor_complete
            and partition.excluded_predecessor_depth == depth
        )


def test_every_generated_piece_replays_its_exact_guarded_path_to_the_original_exclusion(
    memory,
):
    partition = _derive(memory, 3)
    edges = memory.return_envelope.return_relation
    extra = [
        piece
        for piece in partition.bad_pieces
        if hasattr(piece, "successor_memory_indices")
    ]
    assert extra and any(len(piece.successor_memory_indices) == 3 for piece in extra)
    for piece in extra:
        original = partition.bad_pieces[piece.original_bad_piece_index]
        assert not hasattr(original, "successor_memory_indices")
        assert original.exclusion_index == piece.exclusion_index
        assert original.target_index == piece.target_index
        assert original.terminal_transition_index == piece.terminal_transition_index
        for start in _members(piece.bounds):
            current, index = start, piece.memory_index
            for successor in piece.successor_memory_indices:
                edge = edges[successor]
                assert edges[index].target_epi == edge.source_epi
                assert current in _members(memory.source_guards[successor])
                current = (current[0] + edge.shift[0], current[1] + edge.shift[1])
                assert current in _members(memory.retained_endpoint_zones[successor])
                index = successor
            assert index == original.memory_index and current in _members(
                original.bounds
            )
            terminal = memory.return_envelope.intermediate_transitions[
                original.terminal_transition_index
            ]
            assert current in _members(terminal.source_guard)
            image = (current[0] + terminal.shift[0], current[1] + terminal.shift[1])
            assert image in _members(_target(memory)[original.target_index].bounds)


def test_disconnected_exact_predecessors_do_not_authorize_their_joined_gap():
    memory = _memory(branches=True)
    partition = _derive(memory, 1)
    expected, _ = _exact_predecessor_oracle(memory, _target(memory), 1)
    assert (
        (7, -1) in expected[1] and (7, 1) in expected[1] and (7, 0) not in expected[1]
    )
    actual_bad = set().union(
        *(
            _members(piece.bounds)
            for piece in partition.bad_pieces
            if piece.memory_index == 1
        )
    )
    seeds = set().union(
        *(
            _members(zone)
            for (i, _ordinal), zone in zip(
                partition.vertices, partition.seed_zones, strict=True
            )
            if i == 1
        )
    )
    assert actual_bad == expected[1]
    assert (7, 0) in seeds and (7, 0) in _members(_hull(actual_bad))


def test_default_and_explicit_zero_preserve_b59_type_payload_and_work(memory):
    kwargs = dict(
        exclusion_query_max_intersections=10_000,
        max_partition_pieces=1000,
        max_partition_construction_work=100_000,
        max_partition_arcs=5000,
        max_partition_work=100_000,
    )
    default = owner._derive_return_safe_partition(memory, (_target(memory),), **kwargs)
    explicit = _derive(memory, 0)
    assert type(default) is owner.C6CarriedReturnSafePartition and type(
        explicit
    ) is type(default)
    assert default == explicit and asdict(default) == asdict(explicit)
    assert not hasattr(default, "predecessor_complete")


@pytest.mark.parametrize(
    "row,point", ((0, (0, 0)), (1, (4, 0)), (2, (0, 7)), (2, (4, 5)))
)
def test_origin_and_terminal_only_visits_remain_admitted_after_three_predecessor_layers(
    memory, row, point
):
    partition = _derive(memory, 3)
    target = _target(memory, row, point)
    query = owner._derive_return_safe_partition_region_queries(
        partition, (target,), 10_000
    ).queries[0]
    assert query.target_regions == target and query.status == "origin_not_excluded"
    assert (
        not query.origin_path_within_domain_excluded
        and not query.actual_origin_reachability_certified
    )


@pytest.mark.parametrize(
    "changes", ({"max_partition_construction_work": 1}, {"max_partition_pieces": 1})
)
def test_interrupted_generation_never_publishes_a_partial_safe_partition(
    memory, changes
):
    partition = _derive(memory, 2, **changes)
    assert not partition.initialization_complete and not partition.relation_complete
    assert (
        partition.vertices
        == partition.seed_zones
        == partition.root_zones
        == partition.partition_arcs
        == ()
    )
    assert not partition.conditional_boundedness_certified
    query = owner._derive_return_safe_partition_region_queries(
        partition, (_target(memory),), 10_000
    ).queries[0]
    assert not query.origin_path_within_domain_excluded


def test_unverified_origin_target_cannot_authorize_any_predecessor_cut(memory):
    partition = _derive(memory, 2, (_target(memory, 0, (0, 0)),))
    assert partition.status == "exclusion_not_verified"
    assert not partition.initialization_complete and not partition.predecessor_complete
    assert not partition.bad_pieces and not partition.vertices


@pytest.mark.parametrize("value", (-1, True, 1.0, "1", None))
def test_invalid_depth_rejected_before_canonical_reconstruction(monkeypatch, value):
    def forbidden(*_a, **_k):
        raise AssertionError(
            "invalid depth must be rejected before rebuilding the canonical source"
        )

    monkeypatch.setattr(owner, "derive_c6_carried_return_memory_envelope", forbidden)
    with pytest.raises((TypeError, ValueError), match="depth"):
        owner.derive_c6_carried_return_safe_partition(
            None,
            state=None,
            epi_states=(),
            timestep=1.0,
            transient_epi_states=(),
            excluded_target_region_groups=(),
            excluded_predecessor_depth=value,
        )


def test_completed_predecessor_layers_cover_exact_futures_without_joining(memory):
    partition = _derive(memory, 3)
    original = [
        piece
        for piece in partition.bad_pieces
        if not hasattr(piece, "successor_memory_indices")
    ]
    observed = [set() for _ in memory.retained_endpoint_zones]
    for piece in original:
        observed[piece.memory_index].update(_members(piece.bounds))
    assert [layer.depth for layer in partition.predecessor_layers] == [1, 2, 3]
    for layer in partition.predecessor_layers:
        assert layer.generated_piece_count >= len(layer.pieces)
        assert layer.work >= layer.intersections + layer.subsumption_checks
        for piece in layer.pieces:
            assert len(piece.successor_memory_indices) == layer.depth
            observed[piece.memory_index].update(_members(piece.bounds))
        expected, _ = _exact_predecessor_oracle(memory, _target(memory), layer.depth)
        assert tuple(observed) == expected


def test_interruption_inside_second_layer_keeps_only_first_complete_evidence_and_abstains(
    monkeypatch, memory
):
    starts = []
    implementation = owner._return_safe_predecessor_pieces

    def observe_start(source, bad, depth, work, limit, progress):
        starts.append(work.count)
        return implementation(source, bad, depth, work, limit, progress)

    monkeypatch.setattr(owner, "_return_safe_predecessor_pieces", observe_start)
    complete = _derive(memory, 3)
    first, second = complete.predecessor_layers[:2]
    guard = (
        starts[0]
        + complete.predecessor_relation_intersections
        + first.work
        + second.work
        - 1
    )
    interrupted = _derive(memory, 3, max_partition_construction_work=guard)
    assert interrupted.predecessor_layers == (first,)
    assert not interrupted.predecessor_complete
    assert not interrupted.initialization_complete and not interrupted.relation_complete
    assert (
        interrupted.bad_pieces
        == interrupted.vertices
        == interrupted.seed_zones
        == interrupted.partition_arcs
        == ()
    )
    assert interrupted.memory_envelope == memory
    query = owner._derive_return_safe_partition_region_queries(
        interrupted, (_target(memory),), 10_000
    ).queries[0]
    assert not query.origin_path_within_domain_excluded
