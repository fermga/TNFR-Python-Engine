"""Exact short-return envelopes for a fixed canonical carried C6 map.

Transient RN cells are eliminated only when their complete one-step relation
has no internal edge. Return guards preserve their unjoined intermediate
geometry. Every finite origin path confined to the declared domain is covered;
domain confinement itself is not proved here.
"""

from collections import Counter, deque
from dataclasses import dataclass, replace
from fractions import Fraction as F
import math

from ..dynamics._euler_kernel import NodalRemainderState, _binary64_tuple
from .c6_carried_viability import (
    C6CarriedForwardZone, C6CarriedRegionExclusion, C6CarriedRegionIteration,
    _GRID, _Limit, _Work, _dbm_close, _dbm_contains_origin,
    _dbm_intersection, _dbm_join, _dbm_subset, _positive_integer,
    _predecessor_domain, _prepare_carried_family,
)
from .c6_pressure_lattice import C6PressureLatticeReference

__all__ = [
    "C6CarriedReturnTransition", "C6CarriedReturnIteration",
    "C6CarriedReturnEnvelope", "derive_c6_carried_return_envelope",
    "C6CarriedReturnRegionExclusion", "C6CarriedReturnRegionExclusions",
    "derive_c6_carried_return_region_exclusions",
    "C6CarriedReturnUnionExclusion", "C6CarriedReturnUnionExclusions",
    "derive_c6_carried_return_union_exclusions",
    "C6CarriedReturnCountWitness", "C6CarriedReturnCountRelaxation",
    "derive_c6_carried_return_count_relaxation",
    "C6CarriedReturnWordBudget", "derive_c6_carried_return_word_budget",
    "C6CarriedReturnMemoryEnvelope", "derive_c6_carried_return_memory_envelope",
]

_Bounds = tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class C6CarriedReturnTransition:
    """One exact direct or two-step relation in origin-relative grid units.

All guards use their respective time coordinates. ``shift`` is the full
six-coordinate displacement from source to target. An intermediate guard
exists exactly when the path contains one transient cell.
"""

    source_epi: tuple[float, ...]
    intermediate_epi: tuple[float, ...] | None
    target_epi: tuple[float, ...]
    source_guard: _Bounds
    intermediate_guard: _Bounds | None
    target_guard: _Bounds
    shift: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class C6CarriedReturnIteration:
    """A complete origin-containing return layer and its intermediate cover."""

    ordinal: int
    base_zone_count: int
    transient_zone_count: int
    origin_retained: bool


@dataclass(frozen=True, slots=True)
class C6CarriedReturnEnvelope:
    """A conditional path envelope, never an unconditional trapping theorem.

An interrupted construction retains the full supplied domain. Once the
relation is complete, each accepted base layer contains all domain-confined
origin returns. Transient coverage includes first images with no subsequent
return. Joining these images can lose their predecessor correlation, so even
an exact return fixed point does not assert one-step forward inclusion.
"""

    reference: C6PressureLatticeReference
    state: NodalRemainderState
    timestep: float
    epi_states: tuple[tuple[float, ...], ...]
    pressures: tuple[tuple[float, ...], ...]
    grid_quantum: F
    affine_origin: tuple[F, ...]
    transient_epi_states: tuple[tuple[float, ...], ...]
    domain_zones: tuple[C6CarriedForwardZone, ...]
    retained_zones: tuple[C6CarriedForwardZone, ...]
    return_relation: tuple[C6CarriedReturnTransition, ...]
    intermediate_transitions: tuple[C6CarriedReturnTransition, ...]
    ordinary_transition_count: int
    transient_isolation_certified: bool
    relation_complete: bool
    iterations: tuple[C6CarriedReturnIteration, ...]
    status: str
    construction_intersections: int
    closure_intersections: int
    max_intersections: int
    max_cells: int

    @property
    def intersections(self) -> int:
        return self.construction_intersections + self.closure_intersections

    @property
    def domain_confined_origin_paths_covered(self) -> bool:
        return True

    @property
    def clipped_return_inclusion_certified(self) -> bool:
        return self.relation_complete

    @property
    def clipped_forward_inclusion_certified(self) -> bool:
        return False

    @property
    def conditional_invariance_certified(self) -> bool:
        return False

    @property
    def conditional_boundedness_certified(self) -> bool:
        return False

    @property
    def future_runtime_certified(self) -> bool:
        return False

    @property
    def asymptotic_convergence_certified(self) -> bool:
        return False


def _move(bounds, shift):
    vector = (*shift, 0)
    return tuple(tuple(bounds[i][j] + vector[i] - vector[j] for j in range(7)) for i in range(7))


def _intersection(first, second, work):
    # The budget includes each examined pair, including disjoint boxes.
    work.charge()
    if any(first[i][6] < -second[6][i] or second[i][6] < -first[6][i] for i in range(6)):
        return None
    return _dbm_intersection(first, second)


@dataclass(frozen=True, slots=True)
class C6CarriedReturnMemoryEnvelope:
    """Descending endpoint cover indexed by the last complete return record.

The unchanged origin is a separate persistent zero sentinel. Completed-return
histories confined to the source domain are covered after every complete
update; neither graph stationarity nor this cover proves domain confinement.
Intermediate states, including paths without a subsequent return, remain in
``return_envelope.intermediate_transitions`` and require separate observation.
An interrupted pair construction publishes no partial pair graph.
"""

    return_envelope: C6CarriedReturnEnvelope
    source_guards: tuple[_Bounds | None, ...]
    initial_endpoint_zones: tuple[_Bounds | None, ...]
    retained_endpoint_zones: tuple[_Bounds | None, ...]
    root_endpoint_zones: tuple[_Bounds | None, ...]
    pair_arcs: tuple[tuple[int, int], ...]
    initialization_complete: bool
    pair_relation_complete: bool
    pending_memories: tuple[int, ...]
    status: str
    initialization_work: int
    pair_construction_work: int
    descent_work: int
    completed_visits: int
    strict_updates: int
    max_memory_work: int
    max_memory_arcs: int

    @property
    def memory_work(self) -> int:
        return self.initialization_work + self.pair_construction_work + self.descent_work

    @property
    def domain_confined_origin_histories_covered(self) -> bool:
        return self.initialization_complete

    @property
    def conditional_invariance_certified(self) -> bool:
        return False

    @property
    def conditional_boundedness_certified(self) -> bool:
        return False

    @property
    def future_runtime_certified(self) -> bool:
        return False

    @property
    def asymptotic_convergence_certified(self) -> bool:
        return False


def _derive_return_memory_envelope(envelope, *, max_memory_work, max_memory_arcs):
    """Internal geometry kernel; public callers first rebuild the nodal source."""
    work = _Work(max_memory_work)
    guards, initial, roots, pairs = [], [], [], []
    retained = {zone.epi: zone.bounds for zone in envelope.retained_zones}
    edges = envelope.return_relation
    initialized = graph_complete = False
    initialization_work = pair_work = visits = updates = 0
    current, pending = [], deque()
    status = "return_construction_resource_limit"

    def meet(first, second):
        if first is None or second is None:
            work.charge()
            return None
        return _intersection(first, second, work)

    if envelope.relation_complete:
        status = "memory_initialization_resource_limit"
        shifts = {}
        for row, pressure in zip(envelope.epi_states, envelope.pressures, strict=True):
            shift = tuple(F(envelope.timestep) * F(p) / envelope.grid_quantum for p in pressure)
            if any(value.denominator != 1 for value in shift):
                raise RuntimeError("return memory requires exact integral nodal shifts")
            shifts[row] = tuple(map(int, shift))
        try:
            for edge in edges:
                guard = meet(edge.source_guard, retained.get(edge.source_epi))
                target = retained.get(edge.target_epi)
                guard = meet(guard, None if target is None else _move(target, tuple(-v for v in edge.shift)))
                if edge.intermediate_epi is not None:
                    middle = retained.get(edge.intermediate_epi)
                    guard = meet(guard, None if middle is None else _move(
                        middle, tuple(-v for v in shifts[edge.source_epi]),
                    ))
                # Charge every origin check, including incompatible source labels.
                root = meet(guard, ((0,) * 7,) * 7 if edge.source_epi == envelope.state.epi else None)
                guards.append(guard)
                initial.append(None if guard is None else _move(guard, edge.shift))
                roots.append(None if root is None else _move(root, edge.shift))
            initialized = True
        except _Limit:
            guards, initial, roots = [], [], []
        initialization_work = work.count
        if initialized:
            current = list(initial)
            pending = deque(range(len(edges)))
            status = "memory_pair_resource_limit"
            by_source = {}
            for j, edge in enumerate(edges):
                by_source.setdefault(edge.source_epi, []).append(j)
            try:
                for i, zone in enumerate(initial):
                    # Empty memories are charged too: construction is bounded.
                    if zone is None:
                        work.charge()
                        continue
                    for j in by_source.get(edges[i].target_epi, ()):
                        if meet(zone, guards[j]) is not None:
                            if len(pairs) == max_memory_arcs:
                                status = "memory_arc_resource_limit"
                                raise _Limit
                            pairs.append((i, j))
                graph_complete = True
            except _Limit:
                pairs = []
            pair_work = work.count - initialization_work
    if graph_complete:
        predecessors = [[] for _ in edges]
        successors = [[] for _ in edges]
        for i, j in pairs:
            predecessors[j].append(i)
            successors[i].append(j)
        queued = set(pending)
        status = "memory_resource_limit"
        while pending:
            j = pending[0]
            required = max(1, len(predecessors[j]))
            # Reserve a complete visit, including null predecessors. No partial
            # hull is retained and an unvisited memory remains in the queue.
            if work.count + required > max_memory_work:
                break
            pending.popleft()
            queued.remove(j)
            after = roots[j]
            if not predecessors[j]:
                work.charge()
            for i in predecessors[j]:
                piece = meet(current[i], guards[j])
                if piece is not None:
                    after = _dbm_join(after, _move(piece, edges[j].shift))
            if not _dbm_subset(after, current[j]):
                raise RuntimeError("return memory lost its descending endpoint inclusion")
            visits += 1
            if after != current[j]:
                current[j] = after
                updates += 1
                for k in successors[j]:
                    if k not in queued:
                        pending.append(k)
                        queued.add(k)
        if not pending:
            status = "fixed_point"
    return C6CarriedReturnMemoryEnvelope(
        envelope, tuple(guards), tuple(initial), tuple(current), tuple(roots),
        tuple(pairs), initialized, graph_complete, tuple(pending), status,
        initialization_work, pair_work, work.count - initialization_work - pair_work,
        visits, updates, max_memory_work, max_memory_arcs,
    )


def derive_c6_carried_return_memory_envelope(
    reference: C6PressureLatticeReference, *, state: NodalRemainderState,
    epi_states: tuple[tuple[float, ...], ...], timestep: float,
    transient_epi_states: tuple[tuple[float, ...], ...],
    domain_zones: tuple[C6CarriedForwardZone, ...] | None = None,
    max_intersections: int = 250_000, max_cells: int = 4096,
    max_memory_work: int = 500_000, max_memory_arcs: int = 250_000,
) -> C6CarriedReturnMemoryEnvelope:
    """Rebuild canonical returns, then refine their last-return endpoint cover.

Memory records distinguish full guarded returns, including their transient
middle cell. Initial guards intersect every retained cell at its proper time.
A complete compatibility graph precedes descent. Every accepted worklist
update recomputes all predecessor images and preserves the original first
images. The two memory guards bound construction, graph storage and descent
independently of ``max_intersections``, which bounds canonical reconstruction.
Resource exhaustion leaves only complete covers and complete updates. Empty
initialization geometry is explicitly incomplete and supports no exclusion.

The partition is a proof device: it introduces no physical parameter, changes
no pressure or carried step, and asserts no actual reachability or stability.
"""
    maximum = _positive_integer(max_memory_work, "max_memory_work")
    arc_limit = _positive_integer(max_memory_arcs, "max_memory_arcs")
    envelope = derive_c6_carried_return_envelope(
        reference, state=state, epi_states=epi_states, timestep=timestep,
        transient_epi_states=transient_epi_states, domain_zones=domain_zones,
        max_intersections=max_intersections, max_cells=max_cells,
    )
    return _derive_return_memory_envelope(
        envelope, max_memory_work=maximum, max_memory_arcs=arc_limit,
    )


def derive_c6_carried_return_envelope(
    reference: C6PressureLatticeReference, *, state: NodalRemainderState,
    epi_states: tuple[tuple[float, ...], ...], timestep: float,
    transient_epi_states: tuple[tuple[float, ...], ...],
    domain_zones: tuple[C6CarriedForwardZone, ...] | None = None,
    max_intersections: int = 250_000, max_cells: int = 4096,
) -> C6CarriedReturnEnvelope:
    """Derive exact one/two-step returns and origin-injected descending hulls.

The unchanged origin must be outside the selected transient rows. Every
transient-to-transient domain edge is forbidden: a discovered edge raises
``ValueError``. All source pressures, RN cells, carried origin and the common
increment grid are rebuilt. Transient selection is a proof partition, not a
physical parameter or a modification of the nodal evolution.

One shared guard bounds ordinary-edge construction, return construction,
closure and reconstruction of transient images. Incomplete construction
returns the original domain with ``relation_complete=False``. An incomplete
closure layer is discarded in full. Only exact base-matrix equality gives
``fixed_point``; neither that status nor a resource stop proves confinement.
"""
    maximum = _positive_integer(max_intersections, "max_intersections")
    cell_limit = _positive_integer(max_cells, "max_cells")
    ref, origin, h, rows, pressures, areas, complete = _prepare_carried_family(
        reference, state=state, epi_states=epi_states, timestep=timestep,
        row_limit=cell_limit, row_limit_label="max_cells",
    )
    if type(transient_epi_states) is not tuple:
        raise TypeError("transient_epi_states must be a tuple of distinct declared rows")
    transient = tuple(_binary64_tuple(row, "transient row") for row in transient_epi_states)
    if len(set(transient)) != len(transient) or any(row not in rows for row in transient):
        raise ValueError("transient rows must be distinct members of epi_states")
    if state.epi in transient:
        raise ValueError("the unchanged origin must be outside the transient rows")
    denominator = math.lcm(*(v.denominator for row in areas for v in row))
    grid = F(math.gcd(*(int(v * denominator) for row in areas for v in row)), denominator) or _GRID
    exact_shifts = tuple(tuple(v / grid for v in row) for row in areas)
    if any(v.denominator != 1 for row in exact_shifts for v in row):
        raise RuntimeError("the return envelope lost its exact increment lattice")
    shifts = dict(zip(rows, (tuple(map(int, row)) for row in exact_shifts), strict=True))
    domain = _predecessor_domain(rows, complete, origin, grid, domain_zones)
    cells = {zone.epi: zone.bounds for zone in domain}
    if state.epi not in cells or not _dbm_contains_origin(cells[state.epi]):
        raise ValueError("the declared return domain must contain the unchanged supplied origin")
    transient_set = set(transient)
    base = tuple(row for row in cells if row not in transient_set)
    middle = tuple(row for row in cells if row in transient_set)
    records = [C6CarriedReturnIteration(0, len(base), len(middle), True)]
    work = _Work(maximum)
    ordinary, relations, intermediates = [], [], []
    isolated = relation_complete = False
    retained = cells
    construction = 0
    status = "construction_resource_limit"
    try:
        for source, zone in cells.items():
            image = _move(zone, shifts[source])
            for target, target_zone in cells.items():
                piece = _intersection(image, target_zone, work)
                if piece is None:
                    continue
                if source in transient_set and target in transient_set:
                    raise ValueError("the transient domain contains a transient-to-transient edge")
                guard = _move(piece, tuple(-v for v in shifts[source]))
                ordinary.append(C6CarriedReturnTransition(
                    source, None, target, guard, None, piece, shifts[source],
                ))
        isolated = True
        intermediates = [edge for edge in ordinary
                         if edge.source_epi not in transient_set and edge.target_epi in transient_set]
        relations = [edge for edge in ordinary
                     if edge.source_epi not in transient_set and edge.target_epi not in transient_set]
        exits = {row: [] for row in middle}
        for edge in ordinary:
            if edge.source_epi in transient_set:
                exits[edge.source_epi].append(edge)
        for first in intermediates:
            for second in exits[first.target_epi]:
                mid_guard = _intersection(first.target_guard, second.source_guard, work)
                if mid_guard is None:
                    continue
                source_guard = _move(mid_guard, tuple(-v for v in first.shift))
                target_guard = _move(mid_guard, second.shift)
                total = tuple(a + b for a, b in zip(first.shift, second.shift, strict=True))
                relations.append(C6CarriedReturnTransition(
                    first.source_epi, first.target_epi, second.target_epi,
                    source_guard, mid_guard, target_guard, total,
                ))
        relation_complete = True
    except _Limit:
        # Partial relations carry no closure authority and are not published.
        relations = []
        intermediates = []
    construction = work.count
    if relation_complete:
        status = "resource_limit"
        low_zones = {row: cells[row] for row in base}
        while True:
            following = {row: None for row in base}
            following[state.epi] = ((0,) * 7,) * 7
            try:
                for edge in relations:
                    zone = low_zones[edge.source_epi]
                    if zone is None:
                        # Charge even empty relation entries to bound traversal.
                        work.charge()
                        continue
                    piece = _intersection(zone, edge.source_guard, work)
                    if piece is not None:
                        following[edge.target_epi] = _dbm_join(
                            following[edge.target_epi], _move(piece, edge.shift),
                        )
                high_zones = {row: None for row in middle}
                for edge in intermediates:
                    zone = following[edge.source_epi]
                    if zone is None:
                        work.charge()
                        continue
                    piece = _intersection(zone, edge.source_guard, work)
                    if piece is not None:
                        high_zones[edge.target_epi] = _dbm_join(
                            high_zones[edge.target_epi], _move(piece, edge.shift),
                        )
            except _Limit:
                break
            combined = following | high_zones
            if (not _dbm_contains_origin(following[state.epi])
                    or any(not _dbm_subset(after, retained[row]) for row, after in combined.items())):
                raise RuntimeError("the return envelope lost its origin or descending inclusion")
            retained = combined
            records.append(C6CarriedReturnIteration(
                len(records), sum(z is not None for z in following.values()),
                sum(z is not None for z in high_zones.values()), True,
            ))
            if following == low_zones:
                status = "fixed_point"
                break
            low_zones = following
    retained_zones = tuple(C6CarriedForwardZone(row, retained[row])
                           for row in cells if retained[row] is not None)
    return C6CarriedReturnEnvelope(
        ref, state, h, rows, pressures, grid, origin, transient, domain, retained_zones,
        tuple(relations), tuple(intermediates), len(ordinary), isolated, relation_complete,
        tuple(records), status, construction, work.count - construction, maximum, cell_limit,
    )


@dataclass(frozen=True, slots=True)
class C6CarriedReturnRegionExclusion(C6CarriedRegionExclusion):
    """One whole-target exclusion under complete guarded short returns.

    Original supplied targets remain unchanged in ``target_regions``. The
    depth-zero backward layer combines retained base targets with every
    low-to-transient preimage, including targets without a subsequent return.
    An interrupted initialization publishes no partial backward layer.
    """

    initial_base_zones: tuple[C6CarriedForwardZone, ...]
    transient_target_preimages: tuple[C6CarriedReturnTransition, ...]
    initialization_complete: bool

    @property
    def completed_depth(self) -> int | None:
        return self.iterations[-1].depth if self.iterations else None


@dataclass(frozen=True, slots=True)
class C6CarriedReturnRegionExclusions:
    """Canonical return envelope and independent whole-target past queries."""

    return_envelope: C6CarriedReturnEnvelope
    queries: tuple[C6CarriedReturnRegionExclusion, ...]
    intersections: int
    query_max_intersections: int

    @property
    def common_grid_relaxes_coordinate_cosets(self) -> bool:
        return True

    @property
    def conditional_invariance_certified(self) -> bool:
        return False

    @property
    def conditional_boundedness_certified(self) -> bool:
        return False

    @property
    def future_runtime_certified(self) -> bool:
        return False

    @property
    def asymptotic_convergence_certified(self) -> bool:
        return False


def _return_target_groups(groups, domain, maximum):
    """Validate target subsets against the already rebuilt exact RN domain."""
    if type(groups) is not tuple or not groups or len(groups) > maximum:
        raise ValueError("target_region_groups must be a nonempty tuple within max_cells")
    cells = {zone.epi: zone.bounds for zone in domain}
    result = []
    for supplied in groups:
        if type(supplied) is not tuple or not supplied or len(supplied) > len(cells):
            raise ValueError("each target group must be a nonempty tuple of distinct domain rows")
        seen, targets = set(), []
        for zone in supplied:
            if type(zone) is not C6CarriedForwardZone:
                raise TypeError("target regions must be C6CarriedForwardZone values")
            row = _binary64_tuple(zone.epi, "target region epi")
            if row not in cells or row in seen:
                raise ValueError("target regions must use distinct declared domain rows")
            bounds = zone.bounds
            if type(bounds) is not tuple or len(bounds) != 7 or any(
                type(values) is not tuple or len(values) != 7
                or any(type(value) is not int for value in values) for values in bounds
            ):
                raise TypeError("target bounds must be seven-by-seven tuples of exact integers")
            if any(bounds[i][i] != 0 for i in range(7)) or _dbm_close(bounds) != bounds:
                raise ValueError("target bounds must be nonempty closed difference bounds")
            if not _dbm_subset(bounds, cells[row]):
                raise ValueError("every target region must stay inside the declared domain")
            seen.add(row)
            targets.append(C6CarriedForwardZone(row, bounds))
        result.append(tuple(targets))
    return tuple(result)


def _return_target_pieces(targets, envelope, retained, work):
    """Yield every unjoined direct target or guarded transient preimage."""
    transient = set(envelope.transient_epi_states)
    for target in targets:
        if target.epi not in retained:
            continue
        clipped = _intersection(target.bounds, retained[target.epi], work)
        if clipped is None:
            continue
        if target.epi not in transient:
            yield target.epi, clipped, None
            continue
        for edge in envelope.intermediate_transitions:
            if edge.target_epi != target.epi or edge.source_epi not in retained:
                continue
            guard = tuple(tuple(min(edge.source_guard[i][j], retained[edge.source_epi][i][j])
                                for j in range(7)) for i in range(7))
            piece = _intersection(guard, _move(clipped, tuple(-v for v in edge.shift)), work)
            if piece is not None:
                preimage = C6CarriedReturnTransition(
                    edge.source_epi, None, target.epi, piece, None,
                    _move(piece, edge.shift), edge.shift,
                )
                yield edge.source_epi, piece, preimage


def _guarded_return_edges(envelope, retained):
    """Clip exact return source guards to the retained path envelope."""
    return tuple(
        (edge, tuple(tuple(min(edge.source_guard[i][j], retained[edge.source_epi][i][j])
                           for j in range(7)) for i in range(7)))
        for edge in envelope.return_relation
        if edge.source_epi in retained and edge.target_epi in retained
    )


def derive_c6_carried_return_region_exclusions(
    reference: C6PressureLatticeReference, *, state: NodalRemainderState,
    epi_states: tuple[tuple[float, ...], ...], timestep: float,
    transient_epi_states: tuple[tuple[float, ...], ...],
    target_region_groups: tuple[tuple[C6CarriedForwardZone, ...], ...],
    domain_zones: tuple[C6CarriedForwardZone, ...] | None = None,
    max_intersections: int = 250_000, query_max_intersections: int = 500_000,
    max_cells: int = 4096,
) -> C6CarriedReturnRegionExclusions:
    """Exclude whole targets by exact predecessor layers on guarded returns.

    The canonical return relation and an origin-preserving envelope are
    derived once. Every supplied target is retained verbatim after validation
    inside the original domain. Intersecting targets with the retained
    envelope preserves all domain-confined origin visits. Each transient
    target is pulled back through every first-step edge, even if that edge
    has no subsequent return; intermediate guards are never replaced by a
    separately joined transient hull.

    Complete backward layers are joined per base RN cell and checked for
    origin membership, including depth zero. Empty layers or exact equality
    of consecutive complete layers exclude the target, provided every prior
    layer excluded the origin. Equal counts, partial layers, resource limits
    and origin membership in an outer hull never certify exclusion. Layers
    need not be nested. Depth counts return transitions, not physical steps.
    """
    maximum = _positive_integer(query_max_intersections, "query_max_intersections")
    envelope = derive_c6_carried_return_envelope(
        reference, state=state, epi_states=epi_states, timestep=timestep,
        transient_epi_states=transient_epi_states, domain_zones=domain_zones,
        max_intersections=max_intersections, max_cells=max_cells,
    )
    groups = _return_target_groups(target_region_groups, envelope.domain_zones, envelope.max_cells)
    retained = {zone.epi: zone.bounds for zone in envelope.retained_zones}
    transient = set(envelope.transient_epi_states)
    base = tuple(zone.epi for zone in envelope.domain_zones if zone.epi not in transient)
    work = _Work(maximum)
    queries = []

    def public(zones):
        return tuple(C6CarriedForwardZone(row, zones[row]) for row in base if zones[row] is not None)

    for targets in groups:
        query = dict(targets=targets, zones={row: None for row in base}, initial=(),
                     preimages=(), records=[], status="construction_resource_limit",
                     intersections=0, initialized=False)
        queries.append(query)
        if not envelope.relation_complete:
            continue
        zones, preimages = {row: None for row in base}, []
        before = work.count
        try:
            for row, piece, preimage in _return_target_pieces(targets, envelope, retained, work):
                zones[row] = _dbm_join(zones[row], piece)
                if preimage is not None:
                    preimages.append(preimage)
        except _Limit:
            query["status"] = "initialization_resource_limit"
            query["intersections"] = work.count - before
            continue
        query["intersections"] = work.count - before
        present = _dbm_contains_origin(zones[state.epi])
        count = sum(zone is not None for zone in zones.values())
        query.update(zones=zones, initial=public(zones), preimages=tuple(preimages),
                     records=[C6CarriedRegionIteration(0, count, present)], initialized=True,
                     status="origin_not_excluded" if present else "active" if count else "empty_complete_layer")

    guarded = _guarded_return_edges(envelope, retained)
    while any(query["status"] == "active" for query in queries) and work.count < maximum:
        for query in queries:
            if query["status"] != "active":
                continue
            following = {row: None for row in base}
            before = work.count
            try:
                for edge, guard in guarded:
                    target = query["zones"][edge.target_epi]
                    if target is None:
                        continue
                    piece = _intersection(guard, _move(target, tuple(-v for v in edge.shift)), work)
                    if piece is not None:
                        following[edge.source_epi] = _dbm_join(following[edge.source_epi], piece)
            except _Limit:
                query["intersections"] += work.count - before
                query["status"] = "resource_limit"
                break
            query["intersections"] += work.count - before
            if any(not _dbm_subset(zone, retained.get(row)) for row, zone in following.items()):
                raise RuntimeError("a return predecessor hull escaped its validated envelope")
            present = _dbm_contains_origin(following[state.epi])
            count = sum(zone is not None for zone in following.values())
            query["records"].append(C6CarriedRegionIteration(len(query["records"]), count, present))
            if present:
                query["status"] = "origin_not_excluded"
            elif not count:
                query["status"] = "empty_complete_layer"
            elif following == query["zones"]:
                query["status"] = "stationary_complete_layer"
            query["zones"] = following
    results = tuple(C6CarriedReturnRegionExclusion(
        query["targets"], public(query["zones"]), tuple(query["records"]),
        "resource_limit" if query["status"] == "active" else query["status"],
        query["intersections"], query["initial"], query["preimages"], query["initialized"],
    ) for query in queries)
    return C6CarriedReturnRegionExclusions(envelope, results, work.count, maximum)


@dataclass(frozen=True, slots=True)
class C6CarriedReturnUnionExclusion(C6CarriedReturnRegionExclusion):
    """Whole-target past query preserving a finite union per base RN cell.

    Repeated EPI rows in the zone tuples denote separate closed integer
    DBMs. Only exact pairwise subsumption removes pieces; no convex hull is
    introduced in the backward layers. The inherited forward envelope and
    common-grid relaxation still prevent an origin collision from certifying
    a realized trajectory. Interrupted layers are never published.
    """

    subsumptions: int


@dataclass(frozen=True, slots=True)
class C6CarriedReturnUnionExclusions(C6CarriedReturnRegionExclusions):
    """Independent union queries sharing bounded intersection/comparison work."""

    queries: tuple[C6CarriedReturnUnionExclusion, ...]
    subsumptions: int
    query_max_subsumptions: int
    query_max_zones: int


class _UnionLimit(Exception):
    def __init__(self, status):
        self.status = status


class _ReturnUnion:
    """An exact antichain of DBMs with bounded pairwise comparison work."""

    def __init__(self, rows, maximum, work):
        self.zones = {row: [] for row in rows}
        self.count = 0
        self.maximum = maximum
        self.work = work

    def insert(self, row, piece):
        values, removed = self.zones[row], []
        try:
            for index, old in enumerate(values):
                self.work.charge()
                if _dbm_subset(piece, old):
                    return
                self.work.charge()
                if _dbm_subset(old, piece):
                    removed.append(index)
        except _Limit:
            raise _UnionLimit("subsumption_resource_limit") from None
        count = self.count + 1 - len(removed)
        if count > self.maximum:
            raise _UnionLimit("zone_resource_limit")
        remove = set(removed)
        self.zones[row] = [old for i, old in enumerate(values) if i not in remove] + [piece]
        self.count = count

    def public(self):
        return tuple(C6CarriedForwardZone(row, piece)
                     for row, values in self.zones.items() for piece in sorted(values))

    def contains_origin(self, row):
        return any(_dbm_contains_origin(piece) for piece in self.zones[row])


def _derive_return_union_queries(envelope, groups, maximum, zone_limit, comparison_limit):
    """Run union queries after canonical envelope and target validation."""
    retained = {zone.epi: zone.bounds for zone in envelope.retained_zones}
    transient = set(envelope.transient_epi_states)
    base = tuple(zone.epi for zone in envelope.domain_zones if zone.epi not in transient)
    work, comparisons = _Work(maximum), _Work(comparison_limit)
    queries = []
    for targets in groups:
        query = dict(targets=targets, zones=None, initial=(), preimages=(), records=[],
                     status="construction_resource_limit", intersections=0,
                     subsumptions=0, initialized=False)
        queries.append(query)
        if not envelope.relation_complete:
            continue
        zones, preimages = _ReturnUnion(base, zone_limit, comparisons), []
        before, compared = work.count, comparisons.count
        try:
            for row, piece, preimage in _return_target_pieces(targets, envelope, retained, work):
                zones.insert(row, piece)
                if preimage is not None:
                    preimages.append(preimage)
        except (_Limit, _UnionLimit) as error:
            suffix = error.status if isinstance(error, _UnionLimit) else "resource_limit"
            query["status"] = "initialization_" + suffix
            query["intersections"] = work.count - before
            query["subsumptions"] = comparisons.count - compared
            continue
        present = zones.contains_origin(envelope.state.epi)
        query.update(zones=zones, initial=zones.public(), preimages=tuple(preimages),
                     intersections=work.count-before, subsumptions=comparisons.count-compared,
                     records=[C6CarriedRegionIteration(0, zones.count, present)], initialized=True,
                     status="origin_not_excluded" if present else "active" if zones.count else "empty_complete_layer")

    guarded = _guarded_return_edges(envelope, retained)
    while any(q["status"] == "active" for q in queries) and work.count < maximum:
        for query in queries:
            if query["status"] != "active":
                continue
            following = _ReturnUnion(base, zone_limit, comparisons)
            before, compared = work.count, comparisons.count
            try:
                for edge, guard in guarded:
                    for target in query["zones"].zones[edge.target_epi]:
                        piece = _intersection(guard, _move(target, tuple(-v for v in edge.shift)), work)
                        if piece is not None:
                            following.insert(edge.source_epi, piece)
            except (_Limit, _UnionLimit) as error:
                query["intersections"] += work.count-before
                query["subsumptions"] += comparisons.count-compared
                query["status"] = error.status if isinstance(error, _UnionLimit) else "resource_limit"
                continue
            query["intersections"] += work.count-before
            query["subsumptions"] += comparisons.count-compared
            present = following.contains_origin(envelope.state.epi)
            query["records"].append(C6CarriedRegionIteration(len(query["records"]), following.count, present))
            if present:
                query["status"] = "origin_not_excluded"
            elif not following.count:
                query["status"] = "empty_complete_layer"
            elif following.public() == query["zones"].public():
                query["status"] = "stationary_complete_layer"
            query["zones"] = following
    results = tuple(C6CarriedReturnUnionExclusion(
        q["targets"], q["zones"].public() if q["zones"] is not None else (), tuple(q["records"]),
        "resource_limit" if q["status"] == "active" else q["status"], q["intersections"],
        q["initial"], q["preimages"], q["initialized"], q["subsumptions"],
    ) for q in queries)
    return C6CarriedReturnUnionExclusions(
        envelope, results, work.count, maximum, comparisons.count, comparison_limit, zone_limit,
    )


def derive_c6_carried_return_union_exclusions(
    reference: C6PressureLatticeReference, *, state: NodalRemainderState,
    epi_states: tuple[tuple[float, ...], ...], timestep: float,
    transient_epi_states: tuple[tuple[float, ...], ...],
    target_region_groups: tuple[tuple[C6CarriedForwardZone, ...], ...],
    domain_zones: tuple[C6CarriedForwardZone, ...] | None = None,
    max_intersections: int = 250_000, query_max_intersections: int = 500_000,
    query_max_zones: int = 5000, query_max_subsumptions: int = 1_000_000,
    max_cells: int = 4096,
) -> C6CarriedReturnUnionExclusions:
    """Exclude whole target regions using exact unjoined return predecessors.

    Source pressures, RN cells, the carried origin, transient isolation and
    guarded returns are rebuilt by the shared canonical envelope owner.
    Original target regions are validated and retained verbatim. A transient
    target contributes every individual one-step preimage, even without a
    subsequent return. Backward layers preserve their finite unions: a DBM
    is removed only when another retained DBM contains it.

    A complete empty layer or identical consecutive canonical unions excludes
    every domain-confined origin path to the supplied targets, provided all
    completed layers exclude the original origin. The comparison is sufficient
    for semantic equality; equal zone counts alone do not suffice. Source-domain
    confinement itself, actual reachability and global stability remain open.

    Queries advance round-robin under shared intersection and DBM-comparison
    budgets, with a separate zone cap per query layer. Counts include work in
    discarded partial layers. Interrupted initialization publishes no layer;
    later interruptions retain only the last complete layer. Resource bounds
    and proof partitions introduce no physical parameter or trajectory horizon.
    """
    maximum = _positive_integer(query_max_intersections, "query_max_intersections")
    zones = _positive_integer(query_max_zones, "query_max_zones")
    comparisons = _positive_integer(query_max_subsumptions, "query_max_subsumptions")
    envelope = derive_c6_carried_return_envelope(
        reference, state=state, epi_states=epi_states, timestep=timestep,
        transient_epi_states=transient_epi_states, domain_zones=domain_zones,
        max_intersections=max_intersections, max_cells=max_cells,
    )
    groups = _return_target_groups(target_region_groups, envelope.domain_zones, envelope.max_cells)
    return _derive_return_union_queries(envelope, groups, maximum, zones, comparisons)


@dataclass(frozen=True, slots=True)
class C6CarriedReturnCountWitness:
    """Integral counts for an abstract mode walk, with chronology unverified."""

    target_epi: tuple[float, ...]
    displacement: tuple[int, ...]
    coordinate_coefficients: tuple[int, ...]
    zero_circulation_multiplier: int
    edge_counts: tuple[int, ...]

    @property
    def actual_origin_reachability_certified(self) -> bool:
        return False

    @property
    def joint_guard_satisfaction_certified(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class C6CarriedReturnCountRelaxation:
    """Exact boundary of the global incidence/displacement count model.

    The six balanced signed generators span the complete product lattice
    with ``coordinate_divisors``. A strictly positive zero-displacement
    circulation converts every integral signed count solution into positive
    counts. Strong connectivity then supplies an abstract Euler walk with
    the prescribed source, destination and displacement. The RN guards along
    that walk are not checked simultaneously: no actual state reachability,
    unsafe-region reachability or stability conclusion follows.
    """

    return_envelope: C6CarriedReturnEnvelope
    coordinate_divisors: tuple[int, ...]
    positive_circulation: tuple[int, ...]
    coordinate_generators: tuple[tuple[int, ...], ...]
    origin_to_mode_paths: tuple[tuple[tuple[float, ...], tuple[int, ...]], ...]
    mode_to_origin_paths: tuple[tuple[tuple[float, ...], tuple[int, ...]], ...]

    @property
    def exact_cycle_displacement_lattice_certified(self) -> bool:
        return True

    @property
    def nonnegative_counts_cover_coordinate_cosets(self) -> bool:
        return True

    @property
    def abstract_mode_walk_exists_for_each_coordinate_coset_point(self) -> bool:
        return True

    @property
    def joint_guard_satisfaction_certified(self) -> bool:
        return False

    @property
    def actual_origin_reachability_certified(self) -> bool:
        return False

    @property
    def conditional_invariance_certified(self) -> bool:
        return False

    @property
    def conditional_boundedness_certified(self) -> bool:
        return False

    @property
    def future_runtime_certified(self) -> bool:
        return False

    @property
    def asymptotic_convergence_certified(self) -> bool:
        return False

    def construct_counts(
        self, *, target_epi: tuple[float, ...], displacement: tuple[int, ...],
    ) -> C6CarriedReturnCountWitness:
        """Construct positive integer counts without executing the mode walk.

        ``displacement`` is measured from the unchanged carried origin in
        the envelope's common-grid units. It need not lie inside the target
        RN cell; this count model does not enforce endpoint or path geometry.
        The finite construction verifies incidence and displacement exactly.
        """
        target = _binary64_tuple(target_epi, "target_epi")
        vector = _exact_count_vector(displacement, 6, "displacement")
        paths = dict(self.origin_to_mode_paths)
        if target not in paths:
            raise ValueError("target_epi must be a certified base mode")
        edges = self.return_envelope.return_relation
        path_counts = Counter(paths[target])
        baseline = tuple(sum(edges[i].shift[j] for i in paths[target]) for j in range(6))
        delta = tuple(a - b for a, b in zip(vector, baseline, strict=True))
        if any(value % divisor for value, divisor in zip(delta, self.coordinate_divisors, strict=True)):
            raise ValueError("the requested displacement is outside the derived coordinate coset")
        coefficients = tuple(value // divisor for value, divisor in zip(delta, self.coordinate_divisors, strict=True))
        signed = tuple(
            path_counts[i] + sum(c * row[i] for c, row in zip(
                coefficients, self.coordinate_generators, strict=True,
            ))
            for i in range(len(edges))
        )
        multiplier = max(0, *((1 - value + zero - 1) // zero
                              for value, zero in zip(signed, self.positive_circulation, strict=True)))
        counts = tuple(value + multiplier * zero for value, zero in
                       zip(signed, self.positive_circulation, strict=True))
        if not all(value > 0 for value in counts):
            raise RuntimeError("the positive circulation failed to remove a count sign")
        _verify_count_identity(edges, counts, self.return_envelope.state.epi, target, vector)
        return C6CarriedReturnCountWitness(target, vector, coefficients, multiplier, counts)


def _exact_count_vector(values, length, label):
    if (type(values) is not tuple or len(values) != length
            or any(type(value) is not int for value in values)):
        raise TypeError(f"{label} must be a tuple of {length} exact integers")
    return values


def _verify_count_identity(edges, counts, origin, target, displacement):
    balance = Counter()
    total = [0] * 6
    for edge, count in zip(edges, counts, strict=True):
        balance[edge.source_epi] += count
        balance[edge.target_epi] -= count
        for i in range(6):
            total[i] += count * edge.shift[i]
    expected = Counter()
    expected[origin] += 1
    expected[target] -= 1
    if any(balance[row] != expected[row] for row in set(balance) | set(expected)):
        raise ValueError("integer edge counts do not satisfy the required mode incidence balance")
    if tuple(total) != displacement:
        raise ValueError("integer edge counts do not have the required exact nodal displacement")


def _certify_return_count_relaxation(envelope, positive_circulation, coordinate_generators):
    """Verify integer candidates after the canonical envelope has been rebuilt."""
    if not envelope.relation_complete or not envelope.transient_isolation_certified:
        raise ValueError("count certification requires a completely constructed isolated-transient relation")
    edges = envelope.return_relation
    circulation = _exact_count_vector(positive_circulation, len(edges), "positive_circulation")
    if not circulation or any(value <= 0 for value in circulation):
        raise ValueError("the zero-displacement circulation must be strictly positive on every return edge")
    if type(coordinate_generators) is not tuple or len(coordinate_generators) != 6:
        raise TypeError("coordinate_generators must contain six exact integer count tuples")
    generators = tuple(_exact_count_vector(row, len(edges), "coordinate generator")
                       for row in coordinate_generators)
    increments = tuple(tuple(F(envelope.timestep) * F(value) / envelope.grid_quantum for value in row)
                       for row in envelope.pressures)
    if any(value.denominator != 1 for row in increments for value in row):
        raise RuntimeError("canonical nodal shifts lost the envelope's exact increment grid")
    divisors = tuple(math.gcd(*(int(row[i]) for row in increments)) for i in range(6))
    if any(value <= 0 for value in divisors):
        raise ValueError("the full coordinate-lattice count theorem requires a positive increment gcd in every coordinate")
    if any(edge.shift[i] % divisors[i] for edge in edges for i in range(6)):
        raise RuntimeError("a return displacement escaped the canonical coordinate increment lattice")
    transient = set(envelope.transient_epi_states)
    base = tuple(zone.epi for zone in envelope.domain_zones if zone.epi not in transient)
    origin = envelope.state.epi
    if origin not in base:
        raise ValueError("the count relation must contain the unchanged origin as a base mode")
    outgoing, incoming = {row: [] for row in base}, {row: [] for row in base}
    for index, edge in enumerate(edges):
        if edge.source_epi not in outgoing or edge.target_epi not in outgoing:
            raise ValueError("every return endpoint must belong to the declared base modes")
        outgoing[edge.source_epi].append((edge.target_epi, index))
        incoming[edge.target_epi].append((edge.source_epi, index))

    def tree(adjacency, backward):
        paths, queue = {origin: ()}, deque((origin,))
        while queue:
            row = queue.popleft()
            for other, index in adjacency[row]:
                if other not in paths:
                    paths[other] = (index,) + paths[row] if backward else paths[row] + (index,)
                    queue.append(other)
        if len(paths) != len(base):
            raise ValueError("the complete base-mode return graph must be strongly connected")
        return tuple((row, paths[row]) for row in base)

    forward, backward = tree(outgoing, False), tree(incoming, True)
    _verify_count_identity(edges, circulation, origin, origin, (0,) * 6)
    for axis, counts in enumerate(generators):
        target_displacement = tuple(divisors[axis] * int(i == axis) for i in range(6))
        _verify_count_identity(edges, counts, origin, origin, target_displacement)
    return C6CarriedReturnCountRelaxation(envelope, divisors, circulation, generators, forward, backward)


def derive_c6_carried_return_count_relaxation(
    reference: C6PressureLatticeReference, *, state: NodalRemainderState,
    epi_states: tuple[tuple[float, ...], ...], timestep: float,
    transient_epi_states: tuple[tuple[float, ...], ...],
    positive_circulation: tuple[int, ...], coordinate_generators: tuple[tuple[int, ...], ...],
    domain_zones: tuple[C6CarriedForwardZone, ...] | None = None,
    max_intersections: int = 250_000, max_cells: int = 4096,
) -> C6CarriedReturnCountRelaxation:
    """Certify the limitations of global nonnegative return-edge counts.

    Rebuild the fixed C6 source, carried origin, RN cells and complete return
    relation before checking the supplied integer candidates. All six nodal
    coordinate gcds must be positive; a stationary coordinate is explicitly
    outside this theorem. Incomplete construction is rejected. Signed cycle
    generators and a positive zero-displacement circulation are proof data,
    never physical parameters or additional evolution rules.

    The resulting universal statement concerns only mode incidence and total
    nodal displacement. Even a constructed positive count vector need not
    admit any ordering whose carried states satisfy all RN guards. This API
    certifies neither actual origin reachability nor boundedness.
    """
    envelope = derive_c6_carried_return_envelope(
        reference, state=state, epi_states=epi_states, timestep=timestep,
        transient_epi_states=transient_epi_states, domain_zones=domain_zones,
        max_intersections=max_intersections, max_cells=max_cells,
    )
    return _certify_return_count_relaxation(envelope, positive_circulation, coordinate_generators)


@dataclass(frozen=True, slots=True)
class C6CarriedReturnWordBudget:
    """Exact repetition budget for one closed guarded return word.

    Sharpness concerns the declared common-grid word class, not reachability
    from the retained origin or its finer coordinate cosets. ``None`` for
    the maximum means either an unfinished proof or a zero-shift identity;
    ``status`` distinguishes those cases. The latter identity is conditional
    on entering the certified word source and following this fixed map.
    """

    return_envelope: C6CarriedReturnEnvelope
    word_edge_indices: tuple[int, ...]
    word_edges: tuple[C6CarriedReturnTransition, ...]
    word_physical_steps: int
    source_bounds: _Bounds | None
    net_shift: tuple[int, ...]
    status: str
    maximum_repetitions: int | None
    first_empty_repetition: int | None
    maximum_repetition_source_bounds: _Bounds | None
    maximum_repetition_source_point: tuple[int, ...] | None
    pairwise_repetition_upper_bound: int | None
    tested_repetitions: tuple[tuple[int, bool], ...]
    origin_in_word_source: bool | None
    work_items: int
    max_work_items: int
    max_word_length: int

    @property
    def word_budget_certified(self) -> bool:
        return self.status in ("finite_repetition_budget", "zero_shift_word_identity", "empty_word")

    @property
    def conditional_word_identity_certified(self) -> bool:
        return self.status == "zero_shift_word_identity"

    @property
    def common_grid_relaxes_coordinate_cosets(self) -> bool:
        return True

    @property
    def actual_origin_reachability_certified(self) -> bool:
        return False

    @property
    def conditional_invariance_certified(self) -> bool:
        return False

    @property
    def conditional_boundedness_certified(self) -> bool:
        return False

    @property
    def future_runtime_certified(self) -> bool:
        return False

    @property
    def asymptotic_convergence_certified(self) -> bool:
        return False


def _return_word_indices(indices, maximum):
    if type(indices) is not tuple or not indices or len(indices) > maximum:
        raise ValueError("word_edge_indices must be a nonempty tuple within max_word_length")
    if any(type(index) is not int or index < 0 for index in indices):
        raise ValueError("word_edge_indices must contain nonnegative exact integers")
    return indices


def _repeated_return_word_source(bounds, shift, count):
    """Intersect all translated word guards without enumerating repetitions."""
    vector = (*shift, 0)
    return _dbm_close(tuple(tuple(
        bounds[i][j] - (count-1)*max(0, vector[i]-vector[j])
        for j in range(7)) for i in range(7)))


def _derive_return_word_budget(
    envelope, word_edge_indices, *, max_word_length=4096, max_work_items=10_000,
):
    """Word proof core for an envelope freshly derived by a canonical owner."""
    word_limit = _positive_integer(max_word_length, "max_word_length")
    maximum = _positive_integer(max_work_items, "max_work_items")
    indices = _return_word_indices(word_edge_indices, word_limit)
    work = _Work(maximum)
    edges, source, shift, tested = (), None, (0,)*6, []
    status, repetitions, first_empty = "construction_resource_limit", None, None
    last_source, point, upper, origin_present = None, None, None, None

    def result():
        return C6CarriedReturnWordBudget(
            envelope, indices, edges, sum(1 + (edge.intermediate_epi is not None) for edge in edges),
            source, shift, status, repetitions, first_empty, last_source, point, upper,
            tuple(tested), origin_present, work.count, maximum, word_limit,
        )

    if not envelope.relation_complete:
        return result()
    if any(index >= len(envelope.return_relation) for index in indices):
        raise ValueError("word_edge_indices must index the complete rebuilt return relation")
    edges = tuple(envelope.return_relation[index] for index in indices)
    if any(edge.target_epi != edges[(i+1) % len(edges)].source_epi for i, edge in enumerate(edges)):
        raise ValueError("the return word must be composable and closed in its RN-cell labels")
    shift = tuple(sum(edge.shift[i] for edge in edges) for i in range(6))
    retained = {zone.epi: zone.bounds for zone in envelope.retained_zones}
    prefix, joint = (0,)*6, None
    try:
        for ordinal, edge in enumerate(edges):
            if edge.source_epi not in retained or edge.target_epi not in retained:
                joint = None
                break
            guard = _intersection(edge.source_guard, retained[edge.source_epi], work)
            if guard is None:
                joint = None
                break
            guard = _intersection(guard, _move(retained[edge.target_epi], tuple(-v for v in edge.shift)), work)
            if guard is None:
                joint = None
                break
            translated = _move(guard, tuple(-v for v in prefix))
            joint = translated if ordinal == 0 else _intersection(joint, translated, work)
            if joint is None:
                break
            prefix = tuple(a+b for a, b in zip(prefix, edge.shift, strict=True))
    except _Limit:
        status = "word_resource_limit"
        return result()
    source = joint
    origin_present = edges[0].source_epi == envelope.state.epi and _dbm_contains_origin(source)
    if source is None:
        status, repetitions, first_empty = "empty_word", 0, 1
        tested.append((1, False))
        return result()
    tested.append((1, True))
    if not any(shift):
        status = "zero_shift_word_identity"
        return result()
    vector = (*shift, 0)
    upper = 1 + min((source[i][j]+source[j][i]) // abs(vector[i]-vector[j])
                    for i in range(7) for j in range(i) if vector[i] != vector[j])
    cache = {1: source}

    def check(count):
        work.charge()
        value = _repeated_return_word_source(source, shift, count)
        cache[count] = value
        tested.append((count, value is not None))
        return value

    lower, higher = 1, upper+1
    try:
        if check(higher) is not None:
            raise RuntimeError("a repeated return word exceeded its exact pairwise displacement bound")
        while higher-lower > 1:
            middle = (higher+lower)//2
            if check(middle) is None:
                higher = middle
            else:
                lower = middle
    except _Limit:
        status = "word_resource_limit"
        return result()
    repetitions, first_empty, last_source = lower, higher, cache[lower]
    point = tuple(last_source[i][6] for i in range(6))
    augmented = (*point, 0)
    if any(augmented[i]-augmented[j] > last_source[i][j] for i in range(7) for j in range(7)):
        raise RuntimeError("the last feasible return-word source lost its integer witness")
    status = "finite_repetition_budget"
    return result()


def derive_c6_carried_return_word_budget(
    reference: C6PressureLatticeReference, *, state: NodalRemainderState,
    epi_states: tuple[tuple[float, ...], ...], timestep: float,
    transient_epi_states: tuple[tuple[float, ...], ...], word_edge_indices: tuple[int, ...],
    domain_zones: tuple[C6CarriedForwardZone, ...] | None = None,
    max_intersections: int = 250_000, max_cells: int = 4096,
    max_word_length: int = 4096, max_work_items: int = 10_000,
) -> C6CarriedReturnWordBudget:
    """Derive the sharp common-grid repetition budget of one closed word.

    Rebuild the canonical return envelope before selecting the ordered edge
    indices. Every word guard is intersected with the retained source zone
    and the translated retained target zone. Its exact intermediate RN guard
    already belongs to the return edge. Let ``B`` be the intersection of
    those guards translated by their preceding cumulative shifts, and let
    ``A`` be the complete word displacement. No extra copy of the first edge
    guard is required after the final edge.

    The source of ``n`` consecutive repetitions has difference bounds
    ``B[i][j]-(n-1)*max(0,A[i]-A[j])``, followed by exact integer closure,
    with ``A[6]=0``. Nonzero displacement gives a finite pairwise upper bound;
    exact monotone search identifies the last nonempty and first empty source.
    A nonempty zero-displacement word is an identity on its conditional source
    class. Empty word sources exclude even one complete repetition.

    The word/composition budget counts every examined guard intersection and
    every repetition closure, separately from envelope construction. Resource
    stops publish no repetition maximum. Sharpness is relative to the common
    grid and declared envelope; neither the witness nor the result establishes
    reachability from the original carried state or complete-runtime stability.
    """
    word_limit = _positive_integer(max_word_length, "max_word_length")
    work_limit = _positive_integer(max_work_items, "max_work_items")
    _return_word_indices(word_edge_indices, word_limit)
    envelope = derive_c6_carried_return_envelope(
        reference, state=state, epi_states=epi_states, timestep=timestep,
        transient_epi_states=transient_epi_states, domain_zones=domain_zones,
        max_intersections=max_intersections, max_cells=max_cells,
    )
    return _derive_return_word_budget(
        envelope, word_edge_indices, max_word_length=word_limit, max_work_items=work_limit,
    )


@dataclass(frozen=True, slots=True)
class C6CarriedReturnMemoryRegionExclusion:
    """Complete backward hull layers indexed by last-return memory.

The final array entry is the unchanged origin sentinel. Earlier entries use
the endpoint classes declared by the enclosing memory or safe partition.
A resource stop keeps the last complete layer and makes no exclusion claim.
"""

    target_regions: tuple[C6CarriedForwardZone, ...]
    initial_endpoint_zones: tuple[_Bounds | None, ...]
    retained_endpoint_zones: tuple[_Bounds | None, ...]
    iterations: tuple[C6CarriedRegionIteration, ...]
    initialization_complete: bool
    status: str
    intersections: int

    @property
    def origin_path_within_domain_excluded(self) -> bool:
        return self.initialization_complete and self.status in (
            "empty_complete_layer", "stationary_complete_layer",
        )

    @property
    def actual_origin_reachability_certified(self) -> bool:
        return False

    @property
    def completed_depth(self) -> int | None:
        return self.iterations[-1].depth if self.iterations else None


@dataclass(frozen=True, slots=True)
class C6CarriedReturnMemoryRegionExclusions:
    """Canonical memory envelope and separately bounded whole-target queries."""

    memory_envelope: C6CarriedReturnMemoryEnvelope
    queries: tuple[C6CarriedReturnMemoryRegionExclusion, ...]
    query_relation_complete: bool
    query_relation_intersections: int
    query_max_intersections: int

    @property
    def common_grid_relaxes_coordinate_cosets(self) -> bool:
        return True

    @property
    def conditional_invariance_certified(self) -> bool:
        return False

    @property
    def conditional_boundedness_certified(self) -> bool:
        return False

    @property
    def future_runtime_certified(self) -> bool:
        return False

    @property
    def asymptotic_convergence_certified(self) -> bool:
        return False


def _return_memory_query_graph(memory, maximum):
    """Build the complete clipped memory graph, including its origin sentinel.

The returned work count includes skipped empty arcs and every first-image
check. Exhaustion publishes an empty graph with ``complete=False``.
"""
    envelope = memory.return_envelope
    edges = envelope.return_relation
    sentinel = len(edges)
    zero = ((0,) * 7,) * 7
    zones = (*memory.retained_endpoint_zones, zero) if memory.initialization_complete else ()
    work = _Work(maximum)
    graph, complete = [], False
    if memory.pair_relation_complete:
        try:
            for i, j in memory.pair_arcs:
                if zones[i] is None or zones[j] is None:
                    work.charge()
                    continue
                guard = _intersection(zones[i], memory.source_guards[j], work)
                if guard is not None:
                    guard = _intersection(guard, _move(zones[j], tuple(-v for v in edges[j].shift)), work)
                if guard is not None:
                    graph.append((i, j, guard, edges[j].shift))
            for j, root in enumerate(memory.root_endpoint_zones):
                work.charge()
                if root is not None:
                    if not _dbm_subset(root, zones[j]):
                        raise RuntimeError("memory query lost an original first return")
                    graph.append((sentinel, j, zero, edges[j].shift))
            complete = True
        except _Limit:
            graph = []
    return tuple(graph), complete, work.count


def _return_memory_query_initialization(memory, targets, maximum):
    """Initialize whole targets in every endpoint class and the zero sentinel.

Terminal transient steps are included even without a subsequent return. This
helper needs a complete memory initialization; query-graph completeness is a
separate premise. Partial target arrays are discarded on budget exhaustion.
"""
    if not memory.initialization_complete:
        return (), False, 0
    envelope = memory.return_envelope
    rows = tuple(edge.target_epi for edge in envelope.return_relation) + (envelope.state.epi,)
    zones = (*memory.retained_endpoint_zones, ((0,) * 7,) * 7)
    return _return_history_query_initialization(envelope, rows, zones, targets, maximum)


def _return_history_target_pieces(envelope, rows, zones, targets, work):
    """Yield individual direct/terminal pieces, without joining their geometry."""
    terminal_by_rows = {}
    for ordinal, edge in enumerate(envelope.intermediate_transitions):
        terminal_by_rows.setdefault((edge.source_epi, edge.target_epi), []).append((ordinal, edge))
    for i, (row, zone) in enumerate(zip(rows, zones, strict=True)):
        if zone is None:
            work.charge()
            continue
        for target_index, target in enumerate(targets):
            work.charge()
            if row == target.epi:
                piece = _intersection(zone, target.bounds, work)
                if piece is not None:
                    yield i, piece, target_index, None
            for ordinal, edge in terminal_by_rows.get((row, target.epi), ()):
                guard = _intersection(zone, edge.source_guard, work)
                if guard is not None:
                    piece = _intersection(guard, _move(target.bounds, tuple(-v for v in edge.shift)), work)
                    if piece is not None:
                        yield i, piece, target_index, ordinal


def _return_history_query_initialization(envelope, rows, zones, targets, maximum):
    work = _Work(maximum)
    following = [None] * len(zones)
    try:
        for i, piece, _target, _terminal in _return_history_target_pieces(envelope, rows, zones, targets, work):
            following[i] = _dbm_join(following[i], piece)
    except _Limit:
        return (), False, work.count
    return tuple(following), True, work.count


def _return_history_query_result(targets, initial, initialized, initial_work, incoming, maximum, complete):
    """Shared complete-layer predecessor kernel for explicitly indexed histories."""
    work = _Work(maximum)
    work.count = initial_work
    current, records = initial, []
    status = "memory_query_construction_resource_limit"
    if complete:
        status = "active" if initialized else "memory_query_initialization_resource_limit"
    while status == "active":
        present = _dbm_contains_origin(current[-1])
        count = sum(zone is not None for zone in current)
        records.append(C6CarriedRegionIteration(len(records), count, present))
        if present:
            status = "origin_not_excluded"
            break
        if not count:
            status = "empty_complete_layer"
            break
        following = [None] * len(current)
        try:
            work.charge()
            for j, target in enumerate(current):
                if target is None:
                    continue
                for i, guard, shift in incoming.get(j, ()):
                    piece = _intersection(guard, _move(target, tuple(-v for v in shift)), work)
                    if piece is not None:
                        following[i] = _dbm_join(following[i], piece)
        except _Limit:
            status = "resource_limit"
            break
        after = tuple(following)
        if after == current:
            records.append(C6CarriedRegionIteration(len(records), count, False))
            status = "stationary_complete_layer"
        current = after
    return C6CarriedReturnMemoryRegionExclusion(
        targets, initial, current, tuple(records), initialized, status, work.count,
    )


def _derive_return_memory_region_queries(memory, groups, maximum):
    graph, complete, graph_work = _return_memory_query_graph(memory, maximum)
    incoming = {}
    for i, j, guard, shift in graph:
        incoming.setdefault(j, []).append((i, guard, shift))
    results = []
    for targets in groups:
        initial, initialized, initial_work = (), False, 0
        if complete:
            initial, initialized, initial_work = _return_memory_query_initialization(memory, targets, maximum)
        results.append(_return_history_query_result(
            targets, initial, initialized, initial_work, incoming, maximum, complete,
        ))
    return C6CarriedReturnMemoryRegionExclusions(memory, tuple(results), complete, graph_work, maximum)


def derive_c6_carried_return_memory_region_exclusions(
    reference: C6PressureLatticeReference, *, state: NodalRemainderState,
    epi_states: tuple[tuple[float, ...], ...], timestep: float,
    transient_epi_states: tuple[tuple[float, ...], ...],
    target_region_groups: tuple[tuple[C6CarriedForwardZone, ...], ...],
    domain_zones: tuple[C6CarriedForwardZone, ...] | None = None,
    max_intersections: int = 250_000, max_cells: int = 4096,
    max_memory_work: int = 500_000, max_memory_arcs: int = 250_000,
    query_max_intersections: int = 250_000,
) -> C6CarriedReturnMemoryRegionExclusions:
    """Exclude whole targets through joint backward memory layers.

The canonical memory envelope is rebuilt once. Its query graph and each
target have separate budgets of query_max_intersections. Targets include
every compatible memory and the distinct origin sentinel, plus terminal
transient steps with no subsequent return. Only complete origin-free layers
establish empty or stationary exclusion. Domain confinement stays unproved.
"""
    maximum = _positive_integer(query_max_intersections, "query_max_intersections")
    memory = derive_c6_carried_return_memory_envelope(
        reference, state=state, epi_states=epi_states, timestep=timestep,
        transient_epi_states=transient_epi_states, domain_zones=domain_zones,
        max_intersections=max_intersections, max_cells=max_cells,
        max_memory_work=max_memory_work, max_memory_arcs=max_memory_arcs,
    )
    groups = _return_target_groups(target_region_groups, memory.return_envelope.domain_zones, max_cells)
    return _derive_return_memory_region_queries(memory, groups, maximum)


__all__.extend([
    "C6CarriedReturnMemoryRegionExclusion", "C6CarriedReturnMemoryRegionExclusions",
    "derive_c6_carried_return_memory_region_exclusions",
])


@dataclass(frozen=True, slots=True)
class C6CarriedReturnExcludedPiece:
    """One exact direct target or one-step terminal preimage, never its hull."""

    memory_index: int
    exclusion_index: int
    target_index: int
    terminal_transition_index: int | None
    bounds: _Bounds


@dataclass(frozen=True, slots=True)
class C6CarriedReturnPredecessorPiece(C6CarriedReturnExcludedPiece):
    """An exact guarded word ending in one original excluded piece.

The word lists future canonical return-record indices, in execution order.
It never replaces a union of different paths by their convex hull.
"""

    original_bad_piece_index: int
    successor_memory_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class C6CarriedReturnPredecessorLayer:
    """One completed, subset-pruned layer of exact forbidden future paths."""

    depth: int
    pieces: tuple[C6CarriedReturnPredecessorPiece, ...]
    generated_piece_count: int
    intersections: int
    subsumption_checks: int
    work: int


@dataclass(frozen=True, slots=True)
class C6CarriedReturnSafeSubtraction:
    """A relative facet basis and its exact disjoint integer complement."""

    memory_index: int
    bad_piece_index: int
    source_bounds: _Bounds
    relative_facets: tuple[tuple[int, int, int], ...]
    safe_pieces: tuple[_Bounds, ...]


@dataclass(frozen=True, slots=True)
class C6CarriedReturnSafePartition:
    """Certified safe pieces of last-return histories with persistent seeds.

Requested exclusions are reverified against the unchanged origin before any
piece is removed. Vertices refer to existing canonical return records; they
are proof classes, not new physical edges. Coverage is conditional on the
original declared domain. A resource stop retains only complete updates.
"""

    memory_envelope: C6CarriedReturnMemoryEnvelope
    excluded_target_queries: tuple[C6CarriedReturnMemoryRegionExclusion, ...]
    exclusion_query_relation_complete: bool
    exclusion_query_relation_intersections: int
    bad_pieces: tuple[C6CarriedReturnExcludedPiece, ...]
    subtractions: tuple[C6CarriedReturnSafeSubtraction, ...]
    vertices: tuple[tuple[int, int], ...]
    seed_zones: tuple[_Bounds, ...]
    root_zones: tuple[_Bounds | None, ...]
    retained_zones: tuple[_Bounds | None, ...]
    partition_arcs: tuple[tuple[int, int, _Bounds, tuple[int, ...]], ...]
    initialization_complete: bool
    relation_complete: bool
    status: str
    construction_work: int
    forward_work: int
    completed_visits: int
    strict_updates: int
    pending_vertices: tuple[int, ...]
    exclusion_query_max_intersections: int
    max_partition_pieces: int
    max_partition_construction_work: int
    max_partition_arcs: int
    max_partition_work: int

    @property
    def domain_confined_origin_histories_covered(self) -> bool:
        return self.initialization_complete

    @property
    def conditional_invariance_certified(self) -> bool:
        return False

    @property
    def conditional_boundedness_certified(self) -> bool:
        return False

    @property
    def actual_origin_reachability_certified(self) -> bool:
        return False

    @property
    def future_runtime_certified(self) -> bool:
        return False

    @property
    def asymptotic_convergence_certified(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class C6CarriedReturnPredecessorPartition(C6CarriedReturnSafePartition):
    """Safe partition extended by completely derived exact predecessor words.

The original excluded pieces remain the unchanged prefix of ``bad_pieces``.
Incomplete predecessor construction authorizes no partition initialization;
any completed layers retained in that result are informational only.
"""

    excluded_predecessor_depth: int
    predecessor_relation_intersections: int
    predecessor_intersections: int
    predecessor_layers: tuple[C6CarriedReturnPredecessorLayer, ...]
    predecessor_complete: bool


def _return_predecessor_depth(value):
    if type(value) is not int:
        raise TypeError("excluded_predecessor_depth must be a nonnegative exact integer")
    if value < 0:
        raise ValueError("excluded_predecessor_depth must be a nonnegative exact integer")
    return value


def _return_optional_intersection(first, second, work):
    if first is None or second is None:
        work.charge()
        return None
    return _intersection(first, second, work)


def _return_safe_difference(bounds, bad, work):
    """Subtract one exact integer DBM using a reduced relative facet basis.

Each piece violates the first unsatisfied facet: integer strictness reverses
``x_i-x_j<=b`` into ``x_j-x_i<=-b-1``. The pieces are disjoint by construction.
Callers must discard the entire subtraction if its work guard interrupts it.
"""
    bad = _intersection(bounds, bad, work)
    if bad is None:
        return (bounds,), ()
    facets = [(i, j, bad[i][j]) for i in range(7) for j in range(7) if bad[i][j] < bounds[i][j]]

    def compose(selected):
        work.charge()
        candidate = [list(row) for row in bounds]
        for i, j, value in selected:
            candidate[i][j] = min(candidate[i][j], value)
        return _dbm_close(tuple(map(tuple, candidate)))

    for facet in tuple(facets):
        reduced = [item for item in facets if item != facet]
        if compose(reduced) == bad:
            facets = reduced
    if compose(facets) != bad:
        raise RuntimeError("safe subtraction lost its exact relative bad-region basis")
    remaining, pieces = bounds, []
    for i, j, value in facets:
        if remaining is None:
            break
        work.charge()
        outside = [list(row) for row in remaining]
        outside[j][i] = min(outside[j][i], -value - 1)
        piece = _dbm_close(tuple(map(tuple, outside)))
        if piece is not None:
            pieces.append(piece)
        work.charge()
        inside = [list(row) for row in remaining]
        inside[i][j] = min(inside[i][j], value)
        remaining = _dbm_close(tuple(map(tuple, inside)))
    if remaining != bad:
        raise RuntimeError("safe subtraction did not retain exactly its forbidden intersection")
    return tuple(pieces), tuple(facets)


def _return_safe_predecessor_pieces(memory, bad_pieces, depth, work, piece_limit, progress):
    """Append only complete exact-path layers; all work shares the caller's guard."""
    graph, complete, count = _return_memory_query_graph(memory, work.maximum - work.count)
    work.count += count
    progress["relation_intersections"] = count
    if not complete:
        raise _Limit
    incoming = {}
    for source, target, guard, shift in graph:
        incoming.setdefault(target, []).append((source, guard, shift))
    previous = tuple((index, (), piece) for index, piece in enumerate(bad_pieces))
    known = {}
    for piece in bad_pieces:
        known.setdefault(piece.memory_index, []).append(piece.bounds)
    for ordinal in range(1, depth + 1):
        start = work.count
        work.charge()
        layer, generated, intersections, comparisons = {}, 0, 0, 0
        for original_index, suffix, successor in previous:
            for source, guard, shift in incoming.get(successor.memory_index, ()):
                piece = _intersection(guard, _move(successor.bounds, tuple(-v for v in shift)), work)
                intersections += 1
                progress["intersections"] += 1
                if piece is None:
                    continue
                generated += 1
                if source == len(memory.return_envelope.return_relation):
                    raise RuntimeError("an excluded predecessor word contains the original zero sentinel")
                redundant = False
                for old in known.get(source, ()):
                    work.charge()
                    comparisons += 1
                    if _dbm_subset(piece, old):
                        redundant = True
                        break
                if redundant:
                    continue
                bucket = layer.setdefault(source, [])
                superseded = []
                for index, old in enumerate(bucket):
                    work.charge()
                    comparisons += 1
                    if _dbm_subset(piece, old.bounds):
                        redundant = True
                        break
                    work.charge()
                    comparisons += 1
                    if _dbm_subset(old.bounds, piece):
                        superseded.append(index)
                if redundant:
                    continue
                bucket[:] = [old for index, old in enumerate(bucket) if index not in superseded]
                bucket.append(C6CarriedReturnPredecessorPiece(
                    source, successor.exclusion_index, successor.target_index,
                    successor.terminal_transition_index, piece, original_index,
                    (successor.memory_index, *suffix),
                ))
                if len(bad_pieces) + sum(map(len, layer.values())) > piece_limit:
                    progress["piece_limit"] = True
                    raise _Limit
        retained = tuple(piece for source in sorted(layer) for piece in layer[source])
        progress["layers"].append(C6CarriedReturnPredecessorLayer(
            ordinal, retained, generated, intersections, comparisons, work.count - start,
        ))
        bad_pieces.extend(retained)
        for piece in retained:
            known.setdefault(piece.memory_index, []).append(piece.bounds)
        previous = tuple((piece.original_bad_piece_index, piece.successor_memory_indices, piece)
                         for piece in retained)
        if not retained:
            break
    progress["complete"] = True


@dataclass(frozen=True, slots=True)
class _ReturnPartitionDescent:
    retained_zones: tuple[_Bounds | None, ...]
    priority_pending: tuple[int, ...]
    ordinary_pending: tuple[int, ...]
    work: int
    visits: int
    updates: int
    priority_visits: int
    ordinary_visits: int

    @property
    def pending(self):
        return self.priority_pending + self.ordinary_pending


def _return_partition_descent(seeds, roots, arcs, maximum, *, priority_flags=(), initial_work=0):
    """Refine complete safe geometry with atomic FIFO or fixed-priority visits."""
    if type(priority_flags) is not tuple or priority_flags and (
        len(priority_flags) != len(seeds) or any(type(flag) is not bool for flag in priority_flags)
    ):
        raise ValueError("priority_flags must be empty or contain one boolean per safe vertex")
    if type(initial_work) is not int or not 0 <= initial_work <= maximum:
        raise ValueError("initial classification work must stay within the forward budget")
    flags = priority_flags or (False,) * len(seeds)
    predecessors, successors = [[] for _ in seeds], [[] for _ in seeds]
    for a, b, guard, shift in arcs:
        predecessors[b].append((a, guard, shift))
        successors[a].append(b)
    urgent = deque(v for v, flag in enumerate(flags) if flag)
    ordinary = deque(v for v, flag in enumerate(flags) if not flag)
    queued = set(range(len(seeds)))
    current = list(seeds)
    work = _Work(maximum)
    work.count = initial_work
    visits = updates = priority_visits = ordinary_visits = 0
    while urgent or ordinary:
        queue = urgent if urgent else ordinary
        b = queue[0]
        required = len(predecessors[b]) + 1
        if work.count + required > maximum:
            break
        queue.popleft()
        queued.remove(b)
        after = roots[b]
        for a, guard, shift in predecessors[b]:
            piece = _return_optional_intersection(current[a], guard, work)
            if piece is not None:
                after = _dbm_join(after, _move(piece, shift))
        after = _return_optional_intersection(after, seeds[b], work)
        if not _dbm_subset(after, current[b]):
            raise RuntimeError("the safe partition lost descending fixed-seed inclusion")
        visits += 1
        priority_visits += flags[b]
        ordinary_visits += not flags[b]
        if after != current[b]:
            current[b] = after
            updates += 1
            for target in successors[b]:
                if target not in queued:
                    (urgent if flags[target] else ordinary).append(target)
                    queued.add(target)
    return _ReturnPartitionDescent(
        tuple(current), tuple(urgent), tuple(ordinary), work.count, visits, updates,
        priority_visits, ordinary_visits,
    )


def _derive_return_safe_partition(
    memory, groups, *, exclusion_query_max_intersections=250_000,
    max_partition_pieces=5000, max_partition_construction_work=100_000,
    max_partition_arcs=100_000, max_partition_work=500_000,
    excluded_predecessor_depth=0, _construction_only=False,
):
    depth = _return_predecessor_depth(excluded_predecessor_depth)
    limits = tuple(_positive_integer(value, name) for name, value in (
        ("exclusion_query_max_intersections", exclusion_query_max_intersections),
        ("max_partition_pieces", max_partition_pieces),
        ("max_partition_construction_work", max_partition_construction_work),
        ("max_partition_arcs", max_partition_arcs), ("max_partition_work", max_partition_work),
    ))
    exclusion_limit, piece_limit, construction_limit, arc_limit, forward_limit = limits
    envelope = memory.return_envelope
    groups = _return_target_groups(groups, envelope.domain_zones, envelope.max_cells)
    exclusions = _derive_return_memory_region_queries(memory, groups, exclusion_limit)
    work = _Work(construction_limit)
    bad_pieces, subtractions, vertices, seeds, roots, arcs = [], [], [], [], [], []
    initialized = complete = False
    status = "exclusion_not_verified"
    queue, current = deque(), []
    forward = visits = updates = 0
    edges = envelope.return_relation
    predecessor = {"relation_intersections": 0, "intersections": 0, "layers": [],
                   "complete": False, "piece_limit": False}
    if exclusions.query_relation_complete and all(q.origin_path_within_domain_excluded for q in exclusions.queries):
        status = "partition_construction_resource_limit"
        rows = tuple(edge.target_epi for edge in edges) + (envelope.state.epi,)
        zones = (*memory.retained_endpoint_zones, ((0,) * 7,) * 7)
        by_memory = [[] for _ in edges]
        try:
            bad_by_memory = [[] for _ in edges]
            for exclusion_index, targets in enumerate(groups):
                for i, piece, target, terminal in _return_history_target_pieces(envelope, rows, zones, targets, work):
                    if i == len(edges):
                        raise RuntimeError("a certified exclusion contains its original zero sentinel")
                    if len(bad_pieces) == piece_limit:
                        status = "partition_piece_resource_limit"
                        raise _Limit
                    bad_by_memory[i].append(len(bad_pieces))
                    bad_pieces.append(C6CarriedReturnExcludedPiece(i, exclusion_index, target, terminal, piece))
            if depth:
                original_count = len(bad_pieces)
                _return_safe_predecessor_pieces(memory, bad_pieces, depth, work, piece_limit, predecessor)
                for index in range(original_count, len(bad_pieces)):
                    bad_by_memory[bad_pieces[index].memory_index].append(index)
            for i, zone in enumerate(memory.retained_endpoint_zones):
                work.charge()
                if zone is None:
                    continue
                pieces = (zone,)
                for bad_index in bad_by_memory[i]:
                    following = []
                    for source in pieces:
                        parts, basis = _return_safe_difference(source, bad_pieces[bad_index].bounds, work)
                        if parts != (source,):
                            subtractions.append(C6CarriedReturnSafeSubtraction(i, bad_index, source, basis, parts))
                        following.extend(parts)
                        if len(following) + len(vertices) > piece_limit:
                            status = "partition_piece_resource_limit"
                            raise _Limit
                    pieces = tuple(following)
                for ordinal, piece in enumerate(pieces):
                    if len(vertices) == piece_limit:
                        status = "partition_piece_resource_limit"
                        raise _Limit
                    by_memory[i].append(len(vertices))
                    vertices.append((i, ordinal))
                    seeds.append(piece)
            roots = [None] * len(vertices)
            for i, root in enumerate(memory.root_endpoint_zones):
                work.charge()
                if root is None:
                    continue
                if any(root[k][6] != -root[6][k] for k in range(6)):
                    raise RuntimeError("a canonical first return must be a single carried point")
                hits = []
                for v in by_memory[i]:
                    roots[v] = _intersection(root, seeds[v], work)
                    if roots[v] is not None:
                        hits.append(roots[v])
                if hits != [root]:
                    raise RuntimeError("the safe partition lost or duplicated an original first return")
            initialized = True
            current = list(seeds)
            queue = deque(range(len(vertices)))
            for i, j in memory.pair_arcs:
                work.charge()
                for a in by_memory[i]:
                    for b in by_memory[j]:
                        guard = _intersection(seeds[a], memory.source_guards[j], work)
                        guard = _return_optional_intersection(guard, _move(seeds[b], tuple(-v for v in edges[j].shift)), work)
                        if guard is not None:
                            if len(arcs) == arc_limit:
                                status = "partition_arc_resource_limit"
                                raise _Limit
                            arcs.append((a, b, guard, edges[j].shift))
            complete = True
        except _Limit:
            if predecessor["piece_limit"]:
                status = "partition_piece_resource_limit"
            arcs = []
            if not initialized:
                bad_pieces, subtractions, vertices, seeds, roots = [], [], [], [], []
    if complete:
        status = "partition_constructed"
        if not _construction_only:
            descent = _return_partition_descent(seeds, roots, arcs, forward_limit)
            current, queue = descent.retained_zones, descent.pending
            forward, visits, updates = descent.work, descent.visits, descent.updates
            status = "partition_resource_limit" if queue else "fixed_point"
    result = (
        memory, exclusions.queries, exclusions.query_relation_complete, exclusions.query_relation_intersections,
        tuple(bad_pieces), tuple(subtractions), tuple(vertices), tuple(seeds), tuple(roots), tuple(current),
        tuple(arcs), initialized, complete, status, work.count, forward, visits, updates, tuple(queue), *limits,
    )
    if depth:
        return C6CarriedReturnPredecessorPartition(
            *result, depth, predecessor["relation_intersections"], predecessor["intersections"],
            tuple(predecessor["layers"]), predecessor["complete"],
        )
    return C6CarriedReturnSafePartition(*result)


@dataclass(frozen=True, slots=True)
class C6CarriedReturnSafePartitionRegionExclusions:
    """Whole-target queries on explicitly indexed safe-piece histories."""

    partition: C6CarriedReturnSafePartition
    queries: tuple[C6CarriedReturnMemoryRegionExclusion, ...]
    query_relation_complete: bool
    query_relation_intersections: int
    query_max_intersections: int


def _return_safe_partition_query_graph(partition, maximum):
    work = _Work(maximum)
    graph, complete = [], False
    if partition.relation_complete:
        zones = partition.retained_zones
        edges = partition.memory_envelope.return_envelope.return_relation
        try:
            for a, b, guard, shift in partition.partition_arcs:
                if zones[a] is None or zones[b] is None:
                    work.charge()
                    continue
                piece = _intersection(zones[a], guard, work)
                if piece is not None:
                    piece = _intersection(piece, _move(zones[b], tuple(-v for v in shift)), work)
                if piece is not None:
                    graph.append((a, b, piece, shift))
            for v, root in enumerate(partition.root_zones):
                work.charge()
                if root is not None:
                    if not _dbm_subset(root, zones[v]):
                        raise RuntimeError("safe-partition queries lost an original first image")
                    graph.append((len(zones), v, ((0,) * 7,) * 7, edges[partition.vertices[v][0]].shift))
            complete = True
        except _Limit:
            graph = []
    return tuple(graph), complete, work.count


def _derive_return_safe_partition_region_queries(partition, groups, maximum):
    maximum = _positive_integer(maximum, "query_max_intersections")
    envelope = partition.memory_envelope.return_envelope
    groups = _return_target_groups(groups, envelope.domain_zones, envelope.max_cells)
    graph, complete, graph_work = _return_safe_partition_query_graph(partition, maximum)
    incoming = {}
    for a, b, guard, shift in graph:
        incoming.setdefault(b, []).append((a, guard, shift))
    rows = tuple(envelope.return_relation[i].target_epi for i, _ in partition.vertices) + (envelope.state.epi,)
    zones = (*partition.retained_zones, ((0,) * 7,) * 7)
    queries = []
    for targets in groups:
        initial, initialized, initial_work = (), False, 0
        if complete:
            initial, initialized, initial_work = _return_history_query_initialization(
                envelope, rows, zones, targets, maximum,
            )
        queries.append(_return_history_query_result(
            targets, initial, initialized, initial_work, incoming, maximum, complete,
        ))
    return C6CarriedReturnSafePartitionRegionExclusions(partition, tuple(queries), complete, graph_work, maximum)


def derive_c6_carried_return_safe_partition(
    reference: C6PressureLatticeReference, *, state: NodalRemainderState,
    epi_states: tuple[tuple[float, ...], ...], timestep: float,
    transient_epi_states: tuple[tuple[float, ...], ...],
    excluded_target_region_groups: tuple[tuple[C6CarriedForwardZone, ...], ...],
    domain_zones: tuple[C6CarriedForwardZone, ...] | None = None,
    max_intersections: int = 250_000, max_cells: int = 4096,
    max_memory_work: int = 500_000, max_memory_arcs: int = 250_000,
    exclusion_query_max_intersections: int = 250_000,
    max_partition_pieces: int = 5000, max_partition_construction_work: int = 100_000,
    max_partition_arcs: int = 100_000, max_partition_work: int = 500_000,
    excluded_predecessor_depth: int = 0,
) -> C6CarriedReturnSafePartition:
    """Reverify exclusions, subtract exact pieces, and refine persistent seeds.

No caller-supplied success flag or joined predecessor region authorizes a cut.
Direct targets, individual terminal preimages and optional exact guarded
predecessor words of completely verified origin-free queries are removed.
Depth zero preserves the direct/terminal construction. All bounds are computational resource guards;
the canonical pressures, origin, RN rules and physical shifts are unchanged.
Incomplete exclusion, construction and refinement remain distinct outcomes.
"""
    depth = _return_predecessor_depth(excluded_predecessor_depth)
    memory = derive_c6_carried_return_memory_envelope(
        reference, state=state, epi_states=epi_states, timestep=timestep,
        transient_epi_states=transient_epi_states, domain_zones=domain_zones,
        max_intersections=max_intersections, max_cells=max_cells,
        max_memory_work=max_memory_work, max_memory_arcs=max_memory_arcs,
    )
    return _derive_return_safe_partition(
        memory, excluded_target_region_groups, exclusion_query_max_intersections=exclusion_query_max_intersections,
        max_partition_pieces=max_partition_pieces, max_partition_construction_work=max_partition_construction_work,
        max_partition_arcs=max_partition_arcs, max_partition_work=max_partition_work,
        excluded_predecessor_depth=depth,
    )


def derive_c6_carried_return_safe_partition_region_exclusions(
    reference: C6PressureLatticeReference, *, state: NodalRemainderState,
    epi_states: tuple[tuple[float, ...], ...], timestep: float,
    transient_epi_states: tuple[tuple[float, ...], ...],
    excluded_target_region_groups: tuple[tuple[C6CarriedForwardZone, ...], ...],
    target_region_groups: tuple[tuple[C6CarriedForwardZone, ...], ...],
    domain_zones: tuple[C6CarriedForwardZone, ...] | None = None,
    max_intersections: int = 250_000, max_cells: int = 4096,
    max_memory_work: int = 500_000, max_memory_arcs: int = 250_000,
    exclusion_query_max_intersections: int = 250_000,
    max_partition_pieces: int = 5000, max_partition_construction_work: int = 100_000,
    max_partition_arcs: int = 100_000, max_partition_work: int = 500_000,
    query_max_intersections: int = 250_000,
    excluded_predecessor_depth: int = 0,
) -> C6CarriedReturnSafePartitionRegionExclusions:
    """Query every original target history after one verified partition rebuild."""
    maximum = _positive_integer(query_max_intersections, "query_max_intersections")
    depth = _return_predecessor_depth(excluded_predecessor_depth)
    partition = derive_c6_carried_return_safe_partition(
        reference, state=state, epi_states=epi_states, timestep=timestep,
        transient_epi_states=transient_epi_states, excluded_target_region_groups=excluded_target_region_groups,
        domain_zones=domain_zones, max_intersections=max_intersections, max_cells=max_cells,
        max_memory_work=max_memory_work, max_memory_arcs=max_memory_arcs,
        exclusion_query_max_intersections=exclusion_query_max_intersections,
        max_partition_pieces=max_partition_pieces, max_partition_construction_work=max_partition_construction_work,
        max_partition_arcs=max_partition_arcs, max_partition_work=max_partition_work,
        excluded_predecessor_depth=depth,
    )
    return _derive_return_safe_partition_region_queries(partition, target_region_groups, maximum)


__all__.extend([
    "C6CarriedReturnExcludedPiece", "C6CarriedReturnSafeSubtraction", "C6CarriedReturnSafePartition",
    "C6CarriedReturnSafePartitionRegionExclusions", "derive_c6_carried_return_safe_partition",
    "derive_c6_carried_return_safe_partition_region_exclusions",
    "C6CarriedReturnPredecessorPiece", "C6CarriedReturnPredecessorLayer",
    "C6CarriedReturnPredecessorPartition",
])


@dataclass(frozen=True, slots=True)
class C6CarriedReturnCoverSchedule:
    """Computational discovery metadata; advisory regions authorize no cut."""

    policy: str
    priority_memory_regions: tuple[tuple[int, _Bounds | None], ...]
    max_priority_regions: int
    validation_work: int
    classification_complete: bool
    classification_work: int
    priority_flags: tuple[bool, ...]
    priority_pending: tuple[int, ...]
    ordinary_pending: tuple[int, ...]
    priority_visits: int
    ordinary_visits: int
    status: str


@dataclass(frozen=True, slots=True)
class C6CarriedReturnCoverCheck:
    """A complete local image check on canonically reconstructed safe seeds."""

    candidate_zones: tuple[_Bounds | None, ...]
    image_zones: tuple[_Bounds | None, ...]
    status: str
    validation_work: int
    image_work: int
    max_work: int
    violating_vertices: tuple[int, ...]

    @property
    def certified(self) -> bool:
        return self.status == "accepted"

    @property
    def work(self) -> int:
        return self.validation_work + self.image_work


@dataclass(frozen=True, slots=True)
class C6CarriedReturnSafeCover:
    """A rebuilt canonical partition and its checked conditional history cover.

The public derivation authenticates the canonical source and exclusion premises
before checking the candidate. A frozen value object is not an authorization
token: public queries rebuild from primitive inputs, never from this object.
"""

    partition: C6CarriedReturnSafePartition
    schedule: C6CarriedReturnCoverSchedule
    check: C6CarriedReturnCoverCheck

    @property
    def retained_zones(self) -> tuple[_Bounds | None, ...]:
        return self.check.candidate_zones if self.check.certified else ()

    @property
    def domain_confined_origin_histories_covered(self) -> bool:
        return self.check.certified

    @property
    def conditional_invariance_certified(self) -> bool:
        return False

    @property
    def conditional_boundedness_certified(self) -> bool:
        return False

    @property
    def actual_origin_reachability_certified(self) -> bool:
        return False

    @property
    def future_runtime_certified(self) -> bool:
        return False

    @property
    def asymptotic_convergence_certified(self) -> bool:
        return False


def _return_cover_bounds(bounds, label, work):
    work.charge()
    if bounds is None:
        return
    if type(bounds) is not tuple or len(bounds) != 7 or any(
        type(row) is not tuple or len(row) != 7 or any(type(value) is not int for value in row)
        for row in bounds
    ):
        raise TypeError(f"{label} must be None or seven-by-seven tuples of exact integers")
    work.charge()
    if any(bounds[i][i] != 0 for i in range(7)) or _dbm_close(bounds) != bounds:
        raise ValueError(f"{label} must be nonempty closed difference bounds")


def _return_priority_regions(regions, maximum):
    maximum = _positive_integer(maximum, "max_priority_regions")
    if type(regions) is not tuple or len(regions) > maximum:
        raise ValueError("priority_memory_regions must be a tuple within max_priority_regions")
    work = _Work(2 * maximum)
    for item in regions:
        if type(item) is not tuple or len(item) != 2 or type(item[0]) is not int or item[0] < 0:
            raise TypeError("each priority region must be an exact nonnegative memory index and bounds")
        _return_cover_bounds(item[1], "priority bounds", work)
    return regions, work.count


def _return_priority_flags(partition, regions):
    by_memory = {}
    for memory, bounds in regions:
        by_memory.setdefault(memory, []).append(bounds)
    work = _Work(partition.max_partition_work)
    flags = []
    try:
        for (memory, _), seed in zip(partition.vertices, partition.seed_zones, strict=True):
            choices = by_memory.get(memory, ())
            priority = False
            if not choices:
                work.charge()
            for bounds in choices:
                if _return_optional_intersection(seed, bounds, work) is not None:
                    priority = True
                    break
            flags.append(priority)
    except _Limit:
        return (), False, work.count
    return tuple(flags), True, work.count


def _check_return_safe_cover(partition, candidate, maximum):
    """Check an image inequality on a private, already authenticated context.

This arithmetic helper does not authenticate caller-supplied partitions. The
public owner reconstructs those premises before it invokes this helper.
"""
    maximum = _positive_integer(maximum, "max_cover_work")
    if not partition.initialization_complete or not partition.relation_complete:
        return C6CarriedReturnCoverCheck((), (), "source_construction_incomplete", 0, 0, maximum, ())
    if type(candidate) is not tuple or len(candidate) != len(partition.vertices):
        raise ValueError("candidate_retained_zones must be a tuple with one zone per canonical safe vertex")
    work = _Work(maximum)
    image_work = 0
    try:
        lost_roots, escaped_seeds = [], []
        for v, (zone, root, seed) in enumerate(zip(candidate, partition.root_zones, partition.seed_zones, strict=True)):
            _return_cover_bounds(zone, "candidate bounds", work)
            work.charge()
            if not _dbm_subset(root, zone):
                lost_roots.append(v)
            work.charge()
            if not _dbm_subset(zone, seed):
                escaped_seeds.append(v)
        if lost_roots or escaped_seeds:
            status = "root_not_included" if lost_roots else "outside_safe_seeds"
            return C6CarriedReturnCoverCheck(
                candidate, (), status, work.count, 0, maximum, tuple(lost_roots or escaped_seeds),
            )
        following = list(partition.root_zones)
        for a, b, guard, shift in partition.partition_arcs:
            piece = _return_optional_intersection(candidate[a], guard, work)
            image_work += 1
            if piece is not None:
                following[b] = _dbm_join(following[b], _move(piece, shift))
        for v, seed in enumerate(partition.seed_zones):
            following[v] = _return_optional_intersection(following[v], seed, work)
            image_work += 1
        violations = []
        for v, (image, zone) in enumerate(zip(following, candidate, strict=True)):
            work.charge()
            if not _dbm_subset(image, zone):
                violations.append(v)
        status = "image_not_included" if violations else "accepted"
        return C6CarriedReturnCoverCheck(
            candidate, tuple(following), status, work.count - image_work, image_work, maximum, tuple(violations),
        )
    except _Limit:
        return C6CarriedReturnCoverCheck(
            candidate, (), "cover_resource_limit", work.count - image_work, image_work, maximum, (),
        )


def _derive_return_safe_cover(
    memory, groups, *, exclusion_query_max_intersections=250_000,
    max_partition_pieces=5000, max_partition_construction_work=100_000,
    max_partition_arcs=100_000, max_partition_work=500_000, excluded_predecessor_depth=0,
    priority_memory_regions=(), candidate_retained_zones=None,
    max_priority_regions=5000, max_cover_work=100_000, _validated_priority=None,
):
    maximum = _positive_integer(max_cover_work, "max_cover_work")
    region_limit = _positive_integer(max_priority_regions, "max_priority_regions")
    regions, validation = (_return_priority_regions(priority_memory_regions, region_limit)
                           if _validated_priority is None else _validated_priority)
    if candidate_retained_zones is not None and regions:
        raise ValueError("a supplied retained candidate and priority regions are mutually exclusive")
    if any(index >= len(memory.return_envelope.return_relation) for index, _ in regions):
        raise ValueError("priority indices must identify canonical return memories")
    partition = _derive_return_safe_partition(
        memory, groups, exclusion_query_max_intersections=exclusion_query_max_intersections,
        max_partition_pieces=max_partition_pieces, max_partition_construction_work=max_partition_construction_work,
        max_partition_arcs=max_partition_arcs, max_partition_work=max_partition_work,
        excluded_predecessor_depth=excluded_predecessor_depth, _construction_only=True,
    )
    policy = "candidate_only" if candidate_retained_zones is not None else "priority_fifo" if regions else "fifo"
    schedule = C6CarriedReturnCoverSchedule(
        policy, regions, region_limit, validation, False, 0, (), (), (), 0, 0, "source_construction_incomplete",
    )
    if not partition.relation_complete or not partition.initialization_complete:
        check = _check_return_safe_cover(partition, (), maximum)
        return C6CarriedReturnSafeCover(partition, schedule, check)
    if candidate_retained_zones is not None:
        schedule = replace(schedule, classification_complete=True, status="not_run")
        check = _check_return_safe_cover(partition, candidate_retained_zones, maximum)
        return C6CarriedReturnSafeCover(partition, schedule, check)
    flags, complete, classification = (_return_priority_flags(partition, regions) if regions
                                       else ((False,) * len(partition.vertices), True, 0))
    schedule = replace(schedule, classification_complete=complete, classification_work=classification,
                       priority_flags=flags)
    if not complete:
        partition = replace(partition, forward_work=classification, status="priority_resource_limit")
        schedule = replace(schedule, status="priority_resource_limit",
                           ordinary_pending=tuple(range(len(partition.vertices))))
        check = C6CarriedReturnCoverCheck((), (), "priority_resource_limit", 0, 0, maximum, ())
        return C6CarriedReturnSafeCover(partition, schedule, check)
    descent = _return_partition_descent(
        partition.seed_zones, partition.root_zones, partition.partition_arcs,
        partition.max_partition_work, priority_flags=flags, initial_work=classification,
    )
    status = "partition_resource_limit" if descent.pending else "fixed_point"
    partition = replace(partition, retained_zones=descent.retained_zones, status=status,
                        forward_work=descent.work, completed_visits=descent.visits,
                        strict_updates=descent.updates, pending_vertices=descent.pending)
    schedule = replace(schedule, priority_pending=descent.priority_pending, ordinary_pending=descent.ordinary_pending,
                       priority_visits=descent.priority_visits, ordinary_visits=descent.ordinary_visits, status=status)
    check = _check_return_safe_cover(partition, descent.retained_zones, maximum)
    return C6CarriedReturnSafeCover(partition, schedule, check)


@dataclass(frozen=True, slots=True)
class C6CarriedReturnSafeCoverRegionExclusions:
    """Whole-target results obtained only after a rebuilt cover passes its gate."""

    cover: C6CarriedReturnSafeCover
    queries: tuple[C6CarriedReturnMemoryRegionExclusion, ...]
    query_relation_complete: bool
    query_relation_intersections: int
    query_max_intersections: int


def _derive_return_safe_cover_region_queries(cover, groups, maximum):
    """Reuse a same-invocation checked cover, never an external proof object."""
    partition = replace(cover.partition, retained_zones=cover.retained_zones,
                        relation_complete=cover.check.certified)
    queries = _derive_return_safe_partition_region_queries(partition, groups, maximum)
    return C6CarriedReturnSafeCoverRegionExclusions(
        cover, queries.queries, queries.query_relation_complete,
        queries.query_relation_intersections, queries.query_max_intersections,
    )


def derive_c6_carried_return_safe_cover(
    reference: C6PressureLatticeReference, *, state: NodalRemainderState,
    epi_states: tuple[tuple[float, ...], ...], timestep: float,
    transient_epi_states: tuple[tuple[float, ...], ...],
    excluded_target_region_groups: tuple[tuple[C6CarriedForwardZone, ...], ...],
    domain_zones: tuple[C6CarriedForwardZone, ...] | None = None,
    max_intersections: int = 250_000, max_cells: int = 4096,
    max_memory_work: int = 500_000, max_memory_arcs: int = 250_000,
    exclusion_query_max_intersections: int = 250_000,
    max_partition_pieces: int = 5000, max_partition_construction_work: int = 100_000,
    max_partition_arcs: int = 100_000, max_partition_work: int = 500_000,
    excluded_predecessor_depth: int = 0,
    priority_memory_regions: tuple[tuple[int, _Bounds | None], ...] = (),
    candidate_retained_zones: tuple[_Bounds | None, ...] | None = None,
    max_priority_regions: int = 5000, max_cover_work: int = 100_000,
) -> C6CarriedReturnSafeCover:
    """Rebuild canonical premises, optionally discover, then certify a cover.

Advisory regions change only scheduling. Their bounded validation is recorded
separately; seed classification consumes the same forward budget as updates.
A supplied candidate skips discovery but must pass strict exact DBM validation,
root/seed inclusion and a complete seed-clipped image check. No external seeds,
arcs, partition dataclass or success flags authorize a positive result.
The certificate covers domain-confined histories, never domain confinement.
"""
    depth = _return_predecessor_depth(excluded_predecessor_depth)
    maximum = _positive_integer(max_cover_work, "max_cover_work")
    regions = _return_priority_regions(priority_memory_regions, max_priority_regions)
    if candidate_retained_zones is not None and priority_memory_regions:
        raise ValueError("a supplied retained candidate and priority regions are mutually exclusive")
    memory = derive_c6_carried_return_memory_envelope(
        reference, state=state, epi_states=epi_states, timestep=timestep,
        transient_epi_states=transient_epi_states, domain_zones=domain_zones,
        max_intersections=max_intersections, max_cells=max_cells,
        max_memory_work=max_memory_work, max_memory_arcs=max_memory_arcs,
    )
    return _derive_return_safe_cover(
        memory, excluded_target_region_groups, exclusion_query_max_intersections=exclusion_query_max_intersections,
        max_partition_pieces=max_partition_pieces, max_partition_construction_work=max_partition_construction_work,
        max_partition_arcs=max_partition_arcs, max_partition_work=max_partition_work,
        excluded_predecessor_depth=depth, priority_memory_regions=priority_memory_regions,
        candidate_retained_zones=candidate_retained_zones, max_priority_regions=max_priority_regions,
        max_cover_work=maximum, _validated_priority=regions,
    )


def derive_c6_carried_return_safe_cover_region_exclusions(
    reference: C6PressureLatticeReference, *,
    target_region_groups: tuple[tuple[C6CarriedForwardZone, ...], ...],
    query_max_intersections: int = 250_000, **cover_arguments,
) -> C6CarriedReturnSafeCoverRegionExclusions:
    """Rebuild and check one cover, then query unchanged whole target groups.

All other arguments are primitive inputs of
``derive_c6_carried_return_safe_cover``. A supplied proof dataclass is rejected
by that function's explicit signature; only exact candidate matrices may be
submitted for verification against reconstructed canonical premises.
"""
    maximum = _positive_integer(query_max_intersections, "query_max_intersections")
    cover = derive_c6_carried_return_safe_cover(reference, **cover_arguments)
    return _derive_return_safe_cover_region_queries(cover, target_region_groups, maximum)


__all__.extend([
    "C6CarriedReturnCoverSchedule", "C6CarriedReturnCoverCheck", "C6CarriedReturnSafeCover",
    "C6CarriedReturnSafeCoverRegionExclusions", "derive_c6_carried_return_safe_cover",
    "derive_c6_carried_return_safe_cover_region_exclusions",
])
