"""Exact correlated-domain analysis for a fixed canonical carried C6 map.

Backward viability, forward envelopes, exact point predecessors and labeled
region predecessor bounds share canonical pressure and RN-cell preparation.
Each result distinguishes domain inclusion, origin membership and finite
temporal evidence. Resource limits retain only completely proved layers.
"""

from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction as F
from itertools import combinations, product
import math

from ..dynamics._euler_kernel import (
    NODAL_REMAINDER_DENOMINATOR_BITS, NodalRemainderState, NodalRemainderStep,
    advance_nodal_remainder,
    _binary64_tuple, _finite_binary64, _validate_nodal_remainder_state,
)
from .c6_pressure_lattice import (
    C6PressureLatticeReference, _observe_rebuilt_c6_pressure_lattice, _rebuild_lattice,
)
from .nodal_remainder import derive_nodal_remainder_itinerary

__all__ = [
    "C6CarriedViabilityBox", "C6CarriedViabilityIteration", "C6CarriedViability",
    "derive_c6_carried_viability", "C6CarriedForwardZone", "C6CarriedPairBarrier",
    "C6CarriedForwardIteration", "C6CarriedForwardEnvelope",
    "derive_c6_carried_forward_envelope",
    "C6CarriedPredecessorLayer", "C6CarriedPredecessors", "derive_c6_carried_predecessors",
    "C6CarriedRegionIteration", "C6CarriedRegionExclusion", "C6CarriedRegionExclusions",
    "derive_c6_carried_region_exclusions",
    "C6CarriedReachableIteration", "C6CarriedReachableEnvelope",
    "derive_c6_carried_reachable_envelope",
]

_GRID = F(1, 2**NODAL_REMAINDER_DENOMINATOR_BITS)


@dataclass(frozen=True, slots=True)
class C6CarriedViabilityBox:
    """Closed reconstructed-coordinate bounds in one displayed RN cell.

    The represented domain additionally intersects the derived coordinate
    affine cosets through the supplied origin. Bounds are exact Fractions;
    they are geometric constraints, not modifications of the nodal map.
    """

    epi: tuple[float, ...]
    lower: tuple[F, ...]
    upper: tuple[F, ...]


@dataclass(frozen=True, slots=True)
class C6CarriedViabilityIteration:
    ordinal: int
    box_count: int
    point_count: int
    origin_retained: bool


@dataclass(frozen=True, slots=True)
class C6CarriedViability:
    """A complete finite-set proof or an explicitly undecided partial search.

    fixed_point: every state in retained_boxes, on the reported affine
    cosets, remains there under arbitrary repetition of this fixed C6
    source, timestep and unit-capacity carried map. The supplied origin
    belongs to it. origin_excluded: that origin must leave initial_boxes
    or the declared band within exclusion_step_bound transitions.
    resource_limit: no infinite-time or origin-exit conclusion follows.
    No result certifies a live graph, later phase/event policy or convergence.
    """

    reference: C6PressureLatticeReference
    state: NodalRemainderState
    timestep: float
    epi_states: tuple[tuple[float, ...], ...]
    pressures: tuple[tuple[float, ...], ...]
    coordinate_spacings: tuple[F, ...]
    affine_origin: tuple[F, ...]
    initial_boxes: tuple[C6CarriedViabilityBox, ...]
    retained_boxes: tuple[C6CarriedViabilityBox, ...]
    iterations: tuple[C6CarriedViabilityIteration, ...]
    status: str
    work_items: int
    max_work_items: int
    max_boxes: int

    @property
    def conditional_invariance_certified(self) -> bool:
        return self.status == "fixed_point"

    @property
    def conditional_boundedness_certified(self) -> bool:
        return self.conditional_invariance_certified

    @property
    def origin_exit_certified(self) -> bool:
        return self.status == "origin_excluded"

    @property
    def exclusion_step_bound(self) -> int | None:
        return self.iterations[-1].ordinal if self.origin_exit_certified else None

    @property
    def future_runtime_certified(self) -> bool:
        return False

    @property
    def asymptotic_convergence_certified(self) -> bool:
        return False


class _Limit(Exception):
    pass


class _Work:
    def __init__(self, maximum):
        self.maximum, self.count = maximum, 0

    def charge(self):
        if self.count == self.maximum:
            raise _Limit
        self.count += 1


def _intersection(first, second):
    low = tuple(max(a, b) for a, b in zip(first[0], second[0], strict=True))
    high = tuple(min(a, b) for a, b in zip(first[1], second[1], strict=True))
    return None if any(a > b for a, b in zip(low, high, strict=True)) else (low, high)


def _volume(pieces):
    return sum(math.prod(b - a + 1 for a, b in zip(low, high, strict=True)) for _row, low, high in pieces)


def _origin_present(pieces):
    return any(all(a <= 0 <= b for a, b in zip(low, high, strict=True)) for _row, low, high in pieces)


def _coalesce(pieces, work):
    """Merge adjacent boxes only when every other coordinate agrees."""
    while True:
        previous = len(pieces)
        for axis in range(6):
            groups = defaultdict(list)
            for row, low, high in pieces:
                work.charge()
                groups[(row, low[:axis] + low[axis + 1:], high[:axis] + high[axis + 1:])].append(
                    (low[axis], high[axis]),
                )
            following = []
            for (row, rest_low, rest_high), intervals in groups.items():
                merged = []
                for lo, hi in sorted(intervals):
                    if merged and lo <= merged[-1][1]:
                        raise RuntimeError("the exact viability partition unexpectedly overlaps")
                    if merged and lo == merged[-1][1] + 1:
                        merged[-1] = (merged[-1][0], hi)
                    else:
                        merged.append((lo, hi))
                following.extend((row, rest_low[:axis] + (lo,) + rest_low[axis:],
                                  rest_high[:axis] + (hi,) + rest_high[axis:]) for lo, hi in merged)
            pieces = following
        if len(pieces) == previous:
            return pieces


class _BoxIndex:
    """Linear-memory exact interval index; no floating spatial comparisons."""

    def __init__(self, boxes):
        self.boxes = boxes
        self.orders = tuple(
            tuple(sorted((box[bound][axis], i) for i, box in enumerate(boxes)))
            for bound in range(2) for axis in range(6)
        )

    def candidates(self, low, high, work):
        work.charge()
        selected = None
        for index, order in enumerate(self.orders):
            axis = index % 6
            cut = (bisect_right(order, (high[axis], len(self.boxes))) if index < 6
                   else bisect_left(order, (low[axis], -1)))
            bounds = (0, cut) if index < 6 else (cut, len(order))
            if selected is None or bounds[1] - bounds[0] < selected[2] - selected[1]:
                selected = (order, *bounds)
        order, first, last = selected
        for position in range(first, last):
            work.charge()
            box = self.boxes[order[position][1]]
            if _intersection((low, high), box) is not None:
                yield box


def _positive_integer(value, label):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive exact integer")
    return value


def _exact_bounds(values, label):
    if type(values) is not tuple or len(values) != 6 or any(type(value) is not F for value in values):
        raise TypeError(f"{label} must contain six exact Fraction bounds")
    return values


def _prepare_carried_family(reference, *, state, epi_states, timestep, row_limit, row_limit_label):
    """Rebuild the shared fixed source and its exact band-clipped RN cells."""
    ref = _rebuild_lattice(reference)
    if type(state) is not NodalRemainderState:
        raise TypeError("state must be an exact NodalRemainderState")
    origin = _validate_nodal_remainder_state(state)
    if len(origin) != 6:
        raise ValueError("the carried state must contain six coordinates")
    if not ref.source.epi_lower <= state.epi_lower <= state.epi_upper <= ref.source.epi_upper:
        raise ValueError("the carried state band must lie inside the reference slab")
    h = _finite_binary64(timestep, "timestep")
    if h <= 0:
        raise ValueError("viability requires a positive timestep")
    if type(epi_states) is not tuple or not epi_states or len(epi_states) > row_limit:
        raise ValueError(f"epi_states must be a nonempty tuple within {row_limit_label}")
    rows = tuple(_binary64_tuple(row, "epi_states row") for row in epi_states)
    if len(set(rows)) != len(rows) or any(len(row) != 6 for row in rows):
        raise ValueError("the visible rows must be distinct six-coordinate tuples")
    observations = tuple(_observe_rebuilt_c6_pressure_lattice(ref, row) for row in rows)
    pressures = tuple(item.pressure for item in observations)
    areas = tuple(tuple(F(h) * F(value) for value in row) for row in pressures)
    cells = tuple(derive_nodal_remainder_itinerary(
        epi_states=(row, row), timesteps=(0.,), capacities=((1.,) * 6,), pressures=((0.,) * 6,),
        epi_lower=state.epi_lower, epi_upper=state.epi_upper,
    ).coordinates for row in rows)
    complete = tuple(C6CarriedViabilityBox(
        row, tuple(cell.first_grid_index * _GRID for cell in row_cells),
        tuple(cell.last_grid_index * _GRID for cell in row_cells),
    ) for row, row_cells in zip(rows, cells, strict=True))
    return ref, origin, h, rows, pressures, areas, complete


def derive_c6_carried_viability(
    reference: C6PressureLatticeReference, *, state: NodalRemainderState,
    epi_states: tuple[tuple[float, ...], ...], timestep: float,
    initial_boxes: tuple[C6CarriedViabilityBox, ...] | None = None,
    max_work_items: int = 500_000, max_boxes: int = 30_000,
) -> C6CarriedViability:
    """Compute exact complete descents, stopping on a proof or resource limit.

    Each coordinate spacing is the rational gcd of its canonical nodal
    areas over the declared visible family; a stationary coordinate uses
    the shared 2^-3222 grid. These affine cosets contain the unchanged
    origin and are preserved whenever a declared cell supplies pressure.
    They are necessary numerical provenance, not an extra physical field.

    Omitted initial_boxes means all legal RN cells on these cosets. Custom
    boxes must stay inside their exact band-clipped RN cell and must be
    disjoint on the cosets. A step is one exact source-cell translation.
    Intersecting with translated target boxes gives a disjoint partition
    of K intersect F^-1(K). Because K is finite, equal exact point counts
    prove set equality, rather than merely convergence of a sampled hull.
    """
    maximum = _positive_integer(max_work_items, "max_work_items")
    box_limit = _positive_integer(max_boxes, "max_boxes")
    ref, origin, h, rows, pressures, areas, complete = _prepare_carried_family(
        reference, state=state, epi_states=epi_states, timestep=timestep,
        row_limit=box_limit, row_limit_label="max_boxes",
    )
    denominator = math.lcm(*(value.denominator for row in areas for value in row))
    spacings = tuple(F(math.gcd(*(int(row[i] * denominator) for row in areas)), denominator) or _GRID
                     for i in range(6))
    translations = tuple(tuple(int(value / spacing) for value, spacing in zip(row, spacings, strict=True))
                         for row in areas)
    if any(value != step * spacing for row, steps in zip(areas, translations, strict=True)
           for value, step, spacing in zip(row, steps, spacings, strict=True)):
        raise RuntimeError("the candidate pressure family lost its exact affine increment lattice")
    supplied = complete if initial_boxes is None else initial_boxes
    if type(supplied) is not tuple or not supplied or len(supplied) > box_limit:
        raise ValueError("initial_boxes must be a nonempty tuple within max_boxes")
    pieces = []
    for box in supplied:
        if type(box) is not C6CarriedViabilityBox:
            raise TypeError("each initial box must be an exact C6CarriedViabilityBox")
        visible = _binary64_tuple(box.epi, "box.epi")
        if visible not in rows:
            raise ValueError("every initial box must use a declared visible row")
        index = rows.index(visible)
        low, high = _exact_bounds(box.lower, "box.lower"), _exact_bounds(box.upper, "box.upper")
        if any(not cell_lo <= lo <= hi <= cell_hi for cell_lo, lo, hi, cell_hi in zip(
                complete[index].lower, low, high, complete[index].upper, strict=True)):
            raise ValueError("initial box bounds must remain inside the exact legal RN cell")
        lower = tuple(((value - x) / g).__ceil__() for value, x, g in zip(low, origin, spacings, strict=True))
        upper = tuple(((value - x) / g).__floor__() for value, x, g in zip(high, origin, spacings, strict=True))
        if all(a <= b for a, b in zip(lower, upper, strict=True)):
            pieces.append((index, lower, upper))
    grouped = defaultdict(list)
    for index, low, high in pieces:
        if any(_intersection((low, high), box) is not None for box in grouped[index]):
            raise ValueError("initial boxes must be disjoint on the derived affine cosets")
        grouped[index].append((low, high))
    if not _origin_present(pieces):
        raise ValueError("the initial candidate must contain the unchanged supplied origin")
    initial = tuple(pieces)
    count = _volume(pieces)
    records = [C6CarriedViabilityIteration(0, len(pieces), count, True)]
    work, status = _Work(maximum), "resource_limit"
    while True:
        try:
            by_row = defaultdict(list)
            for index, low, high in pieces:
                work.charge()
                by_row[index].append((low, high))
            indices = {index: _BoxIndex(boxes) for index, boxes in by_row.items()}
            following = []
            for source, source_boxes in by_row.items():
                source_result = []
                displacement = translations[source]
                for low, high in source_boxes:
                    image_low = tuple(a + d for a, d in zip(low, displacement, strict=True))
                    image_high = tuple(b + d for b, d in zip(high, displacement, strict=True))
                    for target in indices.values():
                        for target_low, target_high in target.candidates(image_low, image_high, work):
                            shifted = (tuple(a - d for a, d in zip(target_low, displacement, strict=True)),
                                       tuple(b - d for b, d in zip(target_high, displacement, strict=True)))
                            result = _intersection((low, high), shifted)
                            if result is None:
                                raise RuntimeError("the exact target index returned an impossible preimage")
                            source_result.append((source, *result))
                            if len(source_result) > box_limit:
                                raise _Limit
                following.extend(_coalesce(source_result, work))
                if len(following) > box_limit:
                    raise _Limit
            next_count = _volume(following)
            if next_count > count:
                raise RuntimeError("the descending viability set gained grid points")
            present = _origin_present(following)
            records.append(C6CarriedViabilityIteration(len(records), len(following), next_count, present))
            pieces = following
            if not present:
                status = "origin_excluded"
                break
            if next_count == count:
                status = "fixed_point"
                break
            count = next_count
        except _Limit:
            break

    def public(boxes):
        return tuple(C6CarriedViabilityBox(
            rows[index], tuple(x + g * i for x, g, i in zip(origin, spacings, low, strict=True)),
            tuple(x + g * i for x, g, i in zip(origin, spacings, high, strict=True)),
        ) for index, low, high in boxes)

    return C6CarriedViability(
        ref, state, h, rows, pressures, spacings, origin, public(initial), public(pieces),
        tuple(records), status, work.count, maximum, box_limit,
    )


@dataclass(frozen=True, slots=True)
class C6CarriedForwardZone:
    """Closed integer difference bounds x_i-x_j<=bounds[i][j].

    X_i=affine_origin_i+grid_quantum*x_i for i<6. Index 6 is the
    fixed zero coordinate. The visible row identifies the exact RN cell.
    """

    epi: tuple[float, ...]
    bounds: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class C6CarriedPairBarrier:
    """A pair interval preserved whenever its source remains in the cube."""

    first: int
    second: int
    lower: int
    upper: int


@dataclass(frozen=True, slots=True)
class C6CarriedForwardIteration:
    ordinal: int
    zone_count: int
    outgoing_facets: int
    origin_retained: bool


@dataclass(frozen=True, slots=True)
class C6CarriedForwardEnvelope:
    """Complete abstract image layers with separately certified origin entry.

    invariant_core denotes a nonempty forward-invariant retained union under
    the fixed source, positive timestep and unit-capacity carried map. Actual
    origin boundedness additionally requires entry_state. A missing origin in
    a past-compatible envelope never implies that the origin will escape.
    Other statuses carry no infinite-time claim. In particular an unchanged
    abstraction with outgoing states is not an invariant-domain certificate.
    """

    reference: C6PressureLatticeReference
    state: NodalRemainderState
    timestep: float
    epi_states: tuple[tuple[float, ...], ...]
    pressures: tuple[tuple[float, ...], ...]
    grid_quantum: F
    affine_origin: tuple[F, ...]
    pair_barriers: tuple[C6CarriedPairBarrier, ...]
    initial_zones: tuple[C6CarriedForwardZone, ...]
    retained_zones: tuple[C6CarriedForwardZone, ...]
    iterations: tuple[C6CarriedForwardIteration, ...]
    status: str
    intersections: int
    max_intersections: int
    max_cells: int
    entry_state: NodalRemainderState | None
    entry_steps: tuple[NodalRemainderStep, ...]
    entry_failure: str | None

    @property
    def conditional_invariance_certified(self) -> bool:
        return self.status == "invariant_core" and bool(self.retained_zones)

    @property
    def conditional_boundedness_certified(self) -> bool:
        return self.conditional_invariance_certified and self.entry_state is not None

    @property
    def completed_image_layers(self) -> int:
        return self.iterations[-1].ordinal

    @property
    def origin_exit_certified(self) -> bool:
        return False

    @property
    def future_runtime_certified(self) -> bool:
        return False

    @property
    def asymptotic_convergence_certified(self) -> bool:
        return False


def _dbm_close(bounds):
    result = [list(row) for row in bounds]
    for middle in range(7):
        for first in range(7):
            for last in range(7):
                result[first][last] = min(
                    result[first][last], result[first][middle] + result[middle][last],
                )
    if any(result[i][i] < 0 for i in range(7)):
        return None
    return tuple(map(tuple, result))


def _dbm_box(lower, upper):
    low, high = lower + (0,), upper + (0,)
    return tuple(tuple(0 if i == j else high[i] - low[j] for j in range(7)) for i in range(7))


def _dbm_intersection(first, second):
    return _dbm_close(tuple(tuple(min(a, b) for a, b in zip(x, y, strict=True))
                            for x, y in zip(first, second, strict=True)))


def _dbm_join(first, second):
    """The componentwise maximum of closed DBMs is their least DBM hull."""
    if first is None:
        return second
    return tuple(tuple(max(a, b) for a, b in zip(x, y, strict=True))
                 for x, y in zip(first, second, strict=True))


def _dbm_contains_origin(bounds):
    return bounds is not None and all(value >= 0 for row in bounds for value in row)


def _dbm_subset(first, second):
    return first is None or second is not None and all(
        a <= b for x, y in zip(first, second, strict=True) for a, b in zip(x, y, strict=True)
    )


def _conditional_pair_barriers(boxes, translations):
    barriers = []
    for first, second in combinations(range(6), 2):
        projections = tuple((low[first] - high[second], high[first] - low[second],
                             translations[row][first] - translations[row][second])
                            for row, low, high in boxes)
        lower = upper = 0
        while True:
            previous = lower, upper
            for low, high, increment in projections:
                if high < lower or upper < low:
                    continue
                if increment < 0:
                    lower = min(lower, low + increment)
                elif increment > 0:
                    upper = max(upper, high + increment)
            if (lower, upper) == previous:
                break
        # Endpoints come from a finite list of cell endpoints plus increments;
        # this closure has no physical horizon or tolerance parameter.
        for low, high, increment in projections:
            if max(low, lower) <= min(high, upper) and (
                max(low, lower) + increment < lower or min(high, upper) + increment > upper
            ):
                raise RuntimeError("the conditional pair interval failed exact forward inclusion")
        barriers.append(C6CarriedPairBarrier(first, second, lower, upper))
    return tuple(barriers)


def _forward_outgoing(zones, translations, lower, upper):
    count = 0
    for zone, shift in zip(zones, translations, strict=True):
        if zone is not None:
            count += sum(-zone[6][i] + shift[i] < lower[i] for i in range(6))
            count += sum(zone[i][6] + shift[i] > upper[i] for i in range(6))
    return count


def _forward_contains(state, zones, rows, origin, grid):
    values = tuple((x - initial) / grid for x, initial in zip(state.exact_epi, origin, strict=True))
    if any(value.denominator != 1 for value in values):
        return False
    point = tuple(map(int, values)) + (0,)
    return any(zone is not None and row == state.epi and all(
        point[i] - point[j] <= zone[i][j] for i in range(7) for j in range(7)
    ) for row, zone in zip(rows, zones, strict=True))


def derive_c6_carried_forward_envelope(
    reference: C6PressureLatticeReference, *, state: NodalRemainderState,
    epi_states: tuple[tuple[float, ...], ...], timestep: float,
    derive_pair_barriers: bool = True, max_intersections: int = 250_000,
    max_cells: int = 4096,
) -> C6CarriedForwardEnvelope:
    """Bound compatible-past images with one exact relational zone per cell.

    The common grid is the rational gcd of all canonical nodal increments
    (the shared encoding grid when all vanish), with six separate affine
    bases taken from the supplied reconstructed origin. The finite RN family
    must exhaust a contiguous Cartesian box on that grid. Visible gaps with
    no affine-grid points are harmless; gaps containing grid points fail.

    Optional pair intervals contain the origin and each preserve their own
    bound for every source in the original cube. Starting from their
    intersection D, define R_0=D and R_(n+1)=alpha_D(F(R_n)), where alpha_D
    intersects exact translated source zones with each destination cell and
    takes its least DBM hull. Monotonicity gives R_(n+1) subset R_n. If the
    complete image F(R_n) stays in the original cube, the independently
    protected pair intervals imply F(R_n) subset D, hence
    F(R_n) subset R_(n+1) subset R_n. Equality of abstract layers is unnecessary.

    Only complete image layers are retained on resource exhaustion. A
    nonempty invariant core certifies origin boundedness directly when it
    contains that origin; otherwise at most the proved layer ordinal's
    shared nodal steps are replayed to bind entry. The replay is attempted
    only after an invariant core is proved, never to extend an undecided
    horizon. Neither the envelope nor its optional entry is live provenance.
    """
    maximum = _positive_integer(max_intersections, "max_intersections")
    cell_limit = _positive_integer(max_cells, "max_cells")
    if type(derive_pair_barriers) is not bool:
        raise TypeError("derive_pair_barriers must be an exact boolean")
    ref, origin, h, rows, pressures, areas, complete = _prepare_carried_family(
        reference, state=state, epi_states=epi_states, timestep=timestep,
        row_limit=cell_limit, row_limit_label="max_cells",
    )
    denominator = math.lcm(*(value.denominator for row in areas for value in row))
    grid = F(math.gcd(*(int(value * denominator) for row in areas for value in row)), denominator) or _GRID
    exact_translations = tuple(tuple(value / grid for value in row) for row in areas)
    if any(value.denominator != 1 for row in exact_translations for value in row):
        raise RuntimeError("the common affine grid does not preserve all canonical nodal increments")
    translations = tuple(tuple(map(int, row)) for row in exact_translations)
    boxes = []
    for index, box in enumerate(complete):
        low = tuple(((value - initial) / grid).__ceil__()
                    for value, initial in zip(box.lower, origin, strict=True))
        high = tuple(((value - initial) / grid).__floor__()
                     for value, initial in zip(box.upper, origin, strict=True))
        if all(a <= b for a, b in zip(low, high, strict=True)):
            boxes.append((index, low, high))
    if not _origin_present(boxes):
        raise ValueError("the forward candidate must contain the unchanged supplied origin")
    intervals = tuple(tuple(sorted({(low[i], high[i]) for _row, low, high in boxes})) for i in range(6))
    if any(any(first[1] + 1 != second[0] for first, second in zip(axis, axis[1:])) for axis in intervals):
        raise ValueError("the forward RN family must have contiguous affine-grid intervals on every coordinate")
    actual_boxes = {tuple(zip(low, high, strict=True)) for _row, low, high in boxes}
    if math.prod(map(len, intervals)) != len(actual_boxes) or actual_boxes != set(product(*intervals)):
        raise ValueError("the forward RN family must cover its complete Cartesian affine-grid domain")
    lower = tuple(axis[0][0] for axis in intervals)
    upper = tuple(axis[-1][1] for axis in intervals)
    barriers = _conditional_pair_barriers(boxes, translations) if derive_pair_barriers else ()
    source_cells = [None] * len(rows)
    for row, low, high in boxes:
        zone = [list(bound) for bound in _dbm_box(low, high)]
        for barrier in barriers:
            i, j = barrier.first, barrier.second
            zone[i][j] = min(zone[i][j], barrier.upper)
            zone[j][i] = min(zone[j][i], -barrier.lower)
        source_cells[row] = _dbm_close(zone)
    initial = zones = tuple(source_cells)
    if not any(_dbm_contains_origin(zone) for zone in zones):
        raise RuntimeError("derived pair barriers unexpectedly excluded their supplied origin")
    records = []
    work = _Work(maximum)
    status = "resource_limit"
    while True:
        count = sum(zone is not None for zone in zones)
        outgoing = _forward_outgoing(zones, translations, lower, upper)
        records.append(C6CarriedForwardIteration(
            len(records), count, outgoing, any(_dbm_contains_origin(zone) for zone in zones),
        ))
        if not count:
            status = "empty_core"
            break
        if not outgoing:
            status = "invariant_core"
            break
        following = [None] * len(rows)
        try:
            for zone, added in zip(zones, translations, strict=True):
                if zone is None:
                    continue
                shift = added + (0,)
                translated = tuple(tuple(zone[i][j] + shift[i] - shift[j] for j in range(7))
                                   for i in range(7))
                for target, cell in enumerate(initial):
                    if cell is None or any(translated[i][6] < -cell[6][i]
                                           or cell[i][6] < -translated[6][i] for i in range(6)):
                        continue
                    work.charge()
                    piece = _dbm_intersection(translated, cell)
                    if piece is not None:
                        following[target] = _dbm_join(following[target], piece)
        except _Limit:
            break
        following = tuple(following)
        if not all(_dbm_subset(after, before) for after, before in zip(following, zones, strict=True)):
            raise RuntimeError("the monotone forward envelope failed its exact descending identity")
        if following == zones:
            status = "stationary_outer_envelope"
            break
        zones = following
    entry_state, entry_failure = None, None
    entry_steps = []
    if status == "invariant_core":
        if _forward_contains(state, zones, rows, origin, grid):
            entry_state = state
        else:
            current = state
            for _ordinal in range(records[-1].ordinal):
                pressure = _observe_rebuilt_c6_pressure_lattice(ref, current.epi).pressure
                candidate = tuple(value + F(h) * F(p) for value, p in zip(current.exact_epi, pressure, strict=True))
                if any(not F(state.epi_lower) <= value <= F(state.epi_upper) for value in candidate):
                    entry_failure = "band_exit"
                    break
                step = advance_nodal_remainder(current, timestep=h, capacity=(1.,) * 6, pressure=pressure)
                entry_steps.append(step)
                current = step.after
                if not _forward_contains(current, initial, rows, origin, grid):
                    entry_failure = "candidate_exit"
                    break
            if entry_failure is None:
                if not _forward_contains(current, zones, rows, origin, grid):
                    raise RuntimeError("the exact proof-derived entry is absent from its forward envelope")
                entry_state = current

    def public(values):
        return tuple(C6CarriedForwardZone(row, zone)
                     for row, zone in zip(rows, values, strict=True) if zone is not None)

    return C6CarriedForwardEnvelope(
        ref, state, h, rows, pressures, grid, origin, barriers, public(initial), public(zones),
        tuple(records), status, work.count, maximum, cell_limit, entry_state, tuple(entry_steps), entry_failure,
    )


@dataclass(frozen=True, slots=True)
class C6CarriedPredecessorLayer:
    """One completely enumerated exact-depth predecessor frontier.

    For depth>0, successor_indices[j] locates the unique forward successor
    of states[j] in the preceding layer. The target at depth zero has no link.
    Frontiers at different depths need not be nested or disjoint.
    """

    depth: int
    states: tuple[NodalRemainderState, ...]
    successor_indices: tuple[int, ...]


def _predecessor_path(layers, depth, index):
    path = [layers[depth].states[index]]
    while depth:
        index = layers[depth].successor_indices[index]
        depth -= 1
        path.append(layers[depth].states[index])
    return tuple(path)


@dataclass(frozen=True, slots=True)
class C6CarriedPredecessors:
    """Finite exact past feasibility inside a declared fixed-map domain.

    past_excluded means the first empty complete frontier occurs at depth n:
    this target has no n-step past wholly inside the domain and derived
    coordinate cosets. It therefore cannot occur at time>=n during a path
    confined to that domain. Earlier transient visits are not excluded.
    depth_limit and resource_limit give no absence conclusion beyond their
    retained complete frontiers. A nonempty past is not origin reachability;
    only an exact origin match and its forward links provide that evidence.
    This point-specific result says nothing about other points on a facet.
    """

    reference: C6PressureLatticeReference
    state: NodalRemainderState
    target: NodalRemainderState
    timestep: float
    epi_states: tuple[tuple[float, ...], ...]
    pressures: tuple[tuple[float, ...], ...]
    coordinate_spacings: tuple[F, ...]
    grid_quantum: F
    affine_origin: tuple[F, ...]
    domain_zones: tuple[C6CarriedForwardZone, ...]
    layers: tuple[C6CarriedPredecessorLayer, ...]
    status: str
    row_checks: int
    max_row_checks: int
    max_depth: int
    max_cells: int

    @property
    def completed_depth(self) -> int:
        return self.layers[-1].depth

    @property
    def frontier_counts(self) -> tuple[int, ...]:
        return tuple(len(layer.states) for layer in self.layers)

    @property
    def past_exclusion_depth(self) -> int | None:
        return self.completed_depth if self.status == "past_excluded" else None

    @property
    def maximum_compatible_past_depth(self) -> int | None:
        depth = self.past_exclusion_depth
        return None if depth is None else depth - 1

    @property
    def origin_reachability_depths(self) -> tuple[int, ...]:
        return tuple(layer.depth for layer in self.layers if self.state in layer.states)

    @property
    def finite_origin_reachability_certified(self) -> bool:
        return bool(self.origin_reachability_depths)

    @property
    def origin_path_within_domain_excluded(self) -> bool:
        """No finite origin-to-target path stays wholly in the declared domain."""
        return self.status == "past_excluded" and not self.origin_reachability_depths

    @property
    def witness_path(self) -> tuple[NodalRemainderState, ...]:
        depth = self.completed_depth - int(not self.layers[-1].states)
        return _predecessor_path(self.layers, depth, 0)

    @property
    def origin_path(self) -> tuple[NodalRemainderState, ...]:
        for layer in self.layers:
            if self.state in layer.states:
                return _predecessor_path(self.layers, layer.depth, layer.states.index(self.state))
        return ()

    @property
    def whole_domain_exclusion_certified(self) -> bool:
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


def _predecessor_domain(rows, complete, origin, grid, supplied):
    cells = {}
    for row, box in zip(rows, complete, strict=True):
        low = tuple(((value - x) / grid).__ceil__() for value, x in zip(box.lower, origin, strict=True))
        high = tuple(((value - x) / grid).__floor__() for value, x in zip(box.upper, origin, strict=True))
        if all(a <= b for a, b in zip(low, high, strict=True)):
            cells[row] = _dbm_box(low, high)
    if supplied is None:
        return tuple(C6CarriedForwardZone(row, bounds) for row, bounds in cells.items())
    if type(supplied) is not tuple or not supplied or len(supplied) > len(rows):
        raise ValueError("domain_zones must be a nonempty tuple with at most one zone per declared row")
    seen, result = set(), []
    for zone in supplied:
        if type(zone) is not C6CarriedForwardZone:
            raise TypeError("each domain zone must be an exact C6CarriedForwardZone")
        row = _binary64_tuple(zone.epi, "domain zone epi")
        if row not in cells or row in seen:
            raise ValueError("domain zones must use distinct declared rows with nonempty affine-grid RN cells")
        bounds = zone.bounds
        if type(bounds) is not tuple or len(bounds) != 7 or any(
            type(values) is not tuple or len(values) != 7 or any(type(x) is not int for x in values)
            for values in bounds
        ):
            raise TypeError("domain bounds must be a seven-by-seven tuple of exact integers")
        if any(bounds[i][i] != 0 for i in range(7)) or _dbm_close(bounds) != bounds:
            raise ValueError("domain bounds must be nonempty closed difference bounds with zero diagonal")
        if not _dbm_subset(bounds, cells[row]):
            raise ValueError("each domain zone must stay inside its exact legal RN cell")
        seen.add(row)
        result.append(C6CarriedForwardZone(row, bounds))
    return tuple(result)


def derive_c6_carried_predecessors(
    reference: C6PressureLatticeReference, *, state: NodalRemainderState,
    target: NodalRemainderState, epi_states: tuple[tuple[float, ...], ...],
    timestep: float, max_depth: int, domain_zones: tuple[C6CarriedForwardZone, ...] | None = None,
    max_row_checks: int = 32_768, max_cells: int = 4096,
) -> C6CarriedPredecessors:
    """Enumerate every exact predecessor of one target at each completed depth.

    For each declared row r and target X, its only possible predecessor in
    that row is Y=X-h*p(r). The canonical source, exact nodal area, nearest-even
    cell, declared band, coordinate cosets and optional relational domain are
    independently rebuilt or validated. Every admitted link is then checked
    by the shared carried nodal kernel, including the complete remainder.

    Optional zones use the derived common increment-gcd grid and the six
    reconstructed coordinates of state as their affine bases. They declare
    a domain; neither their temporal provenance nor invariance is inferred.
    The domain need not be Cartesian, but must contain both state and target.
    The whole-layer row-check budget is computational, not a physical horizon.
    A resource interruption discards the entire incomplete frontier. Equal
    frontier sizes never justify termination or an existence claim at later
    depths; only the explicitly requested finite depths are examined.
    """
    if type(max_depth) is not int or max_depth < 0:
        raise ValueError("max_depth must be a nonnegative exact integer")
    maximum = _positive_integer(max_row_checks, "max_row_checks")
    cell_limit = _positive_integer(max_cells, "max_cells")
    ref, origin, h, rows, pressures, areas, complete = _prepare_carried_family(
        reference, state=state, epi_states=epi_states, timestep=timestep,
        row_limit=cell_limit, row_limit_label="max_cells",
    )
    if type(target) is not NodalRemainderState:
        raise TypeError("target must be an exact NodalRemainderState")
    exact_target = _validate_nodal_remainder_state(target)
    if len(exact_target) != 6 or (target.epi_lower, target.epi_upper) != (state.epi_lower, state.epi_upper):
        raise ValueError("target must have six coordinates and the same declared band as state")
    denominator = math.lcm(*(value.denominator for row in areas for value in row))
    spacings = tuple(F(math.gcd(*(int(row[i] * denominator) for row in areas)), denominator) or _GRID
                     for i in range(6))
    grid = F(math.gcd(*(int(value * denominator) for row in areas for value in row)), denominator) or _GRID
    if any((value / spacing).denominator != 1 for row in areas
           for value, spacing in zip(row, spacings, strict=True)):
        raise RuntimeError("the predecessor pressure family lost its exact affine increment lattice")
    domain = _predecessor_domain(rows, complete, origin, grid, domain_zones)
    by_row = {zone.epi: zone.bounds for zone in domain}

    def contains(exact, row):
        if row not in by_row or any(
            ((x - initial) / spacing).denominator != 1
            for x, initial, spacing in zip(exact, origin, spacings, strict=True)
        ):
            return False
        indices = tuple((x - initial) / grid for x, initial in zip(exact, origin, strict=True))
        if any(value.denominator != 1 for value in indices):
            return False
        point, bounds = tuple(map(int, indices)) + (0,), by_row[row]
        return all(point[i] - point[j] <= bounds[i][j] for i in range(7) for j in range(7))

    if not contains(origin, state.epi):
        raise ValueError("the declared predecessor domain must contain the unchanged supplied origin")
    if not contains(exact_target, target.epi):
        raise ValueError("target must belong to the declared domain and the origin coordinate cosets")
    layers = [C6CarriedPredecessorLayer(0, (target,), ())]
    work, status = _Work(maximum), "depth_limit"
    for depth in range(1, max_depth + 1):
        following, links, seen = [], [], set()
        try:
            for successor_index, successor in enumerate(layers[-1].states):
                exact_successor = successor.exact_epi
                for row, pressure, area, cell in zip(rows, pressures, areas, complete, strict=True):
                    work.charge()
                    exact = tuple(x - added for x, added in zip(exact_successor, area, strict=True))
                    if any(not low <= value <= high for low, value, high in zip(
                            cell.lower, exact, cell.upper, strict=True)) or not contains(exact, row):
                        continue
                    predecessor = NodalRemainderState(
                        row, tuple(x - F(y) for x, y in zip(exact, row, strict=True)),
                        state.epi_lower, state.epi_upper,
                    )
                    step = advance_nodal_remainder(
                        predecessor, timestep=h, capacity=(1.,) * 6, pressure=pressure,
                    )
                    if step.after != successor:
                        raise RuntimeError("an admitted exact predecessor failed shared nodal replay")
                    if exact in seen:
                        raise RuntimeError("a deterministic carried predecessor has conflicting successor links")
                    seen.add(exact)
                    following.append(predecessor)
                    links.append(successor_index)
        except _Limit:
            status = "resource_limit"
            break
        layers.append(C6CarriedPredecessorLayer(depth, tuple(following), tuple(links)))
        if not following:
            status = "past_excluded"
            break
    return C6CarriedPredecessors(
        ref, state, target, h, rows, pressures, spacings, grid, origin, domain, tuple(layers), status,
        work.count, maximum, max_depth, cell_limit,
    )


@dataclass(frozen=True, slots=True)
class C6CarriedRegionIteration:
    """One complete abstract predecessor layer for a labeled target region."""

    depth: int
    zone_count: int
    origin_present: bool


@dataclass(frozen=True, slots=True)
class C6CarriedRegionExclusion:
    """Whole-region origin-path exclusion through an exact outer construction.

    A complete empty layer proves no past of its depth reaches any target
    point. Exact equality of two consecutive abstract layers proves that
    all later abstract layers repeat, even though earlier layers need not
    be nested. Either result, with the origin absent from every completed
    layer, excludes every origin-to-target path confined to the declared
    domain. Origin membership in a hull is inconclusive and stops the query;
    it is never evidence of an actual path. Resource exhaustion is likewise
    inconclusive, and retains only the last completely constructed layer.
    """

    target_regions: tuple[C6CarriedForwardZone, ...]
    retained_zones: tuple[C6CarriedForwardZone, ...]
    iterations: tuple[C6CarriedRegionIteration, ...]
    status: str
    intersections: int

    @property
    def completed_depth(self) -> int:
        return self.iterations[-1].depth

    @property
    def past_exclusion_depth(self) -> int | None:
        return self.completed_depth if self.status == "empty_complete_layer" else None

    @property
    def origin_path_within_domain_excluded(self) -> bool:
        return self.status in ("empty_complete_layer", "stationary_complete_layer") and not any(
            record.origin_present for record in self.iterations
        )

    @property
    def actual_origin_reachability_certified(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class C6CarriedRegionExclusions:
    """Shared fixed-map premises for independently labeled region exclusions.

    Integer coordinates use the common gcd of all nodal increments and six
    separate affine bases from state. This relaxes finer coordinate cosets:
    it can prove absence, but cannot certify existence of an actual past.
    No domain invariance, live provenance or general stability follows from
    excluding any selected collection of target regions.
    """

    reference: C6PressureLatticeReference
    state: NodalRemainderState
    timestep: float
    epi_states: tuple[tuple[float, ...], ...]
    pressures: tuple[tuple[float, ...], ...]
    grid_quantum: F
    affine_origin: tuple[F, ...]
    domain_zones: tuple[C6CarriedForwardZone, ...]
    queries: tuple[C6CarriedRegionExclusion, ...]
    intersections: int
    max_intersections: int
    max_cells: int

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


def derive_c6_carried_region_exclusions(
    reference: C6PressureLatticeReference, *, state: NodalRemainderState,
    epi_states: tuple[tuple[float, ...], ...], timestep: float,
    target_regions: tuple[tuple[C6CarriedForwardZone, ...], ...],
    domain_zones: tuple[C6CarriedForwardZone, ...] | None = None,
    max_intersections: int = 250_000, max_cells: int = 4096,
) -> C6CarriedRegionExclusions:
    """Bound all compatible pasts of each declared target region.

    For source cell i with canonical exact nodal increment a_i, form each
    D_i intersect (P_j-a_i), then take the least closed difference-bound
    hull per source cell. This contains every exact predecessor on the
    common affine grid; all branch pressure, RN-cell and band premises are
    rebuilt. Targets and the optional domain are declared closed integer
    zones within those cells, with at most one zone per visible row.

    Queries advance round-robin under one intersection budget. They stop at
    a complete empty layer, exact equality with the preceding layer, or
    origin membership in an outer layer. The latter is inconclusive. A
    partially computed layer is discarded on resource interruption; other
    queries retain their independently completed layers. This construction
    has no trajectory horizon, floating tolerance or new physical parameter.
    """
    maximum = _positive_integer(max_intersections, "max_intersections")
    cell_limit = _positive_integer(max_cells, "max_cells")
    ref, origin, h, rows, pressures, areas, complete = _prepare_carried_family(
        reference, state=state, epi_states=epi_states, timestep=timestep,
        row_limit=cell_limit, row_limit_label="max_cells",
    )
    denominator = math.lcm(*(value.denominator for row in areas for value in row))
    grid = F(math.gcd(*(int(value * denominator) for row in areas for value in row)), denominator) or _GRID
    exact_shifts = tuple(tuple(value / grid for value in row) for row in areas)
    if any(value.denominator != 1 for row in exact_shifts for value in row):
        raise RuntimeError("the region pressure family lost its exact common increment lattice")
    shifts = tuple(tuple(map(int, row)) + (0,) for row in exact_shifts)
    domain = _predecessor_domain(rows, complete, origin, grid, domain_zones)
    by_row = {zone.epi: zone.bounds for zone in domain}
    if state.epi not in by_row or not _dbm_contains_origin(by_row[state.epi]):
        raise ValueError("the declared predecessor domain must contain the unchanged supplied origin")
    if type(target_regions) is not tuple or not target_regions or len(target_regions) > cell_limit:
        raise ValueError("target_regions must be a nonempty tuple of queries within max_cells")
    cells = tuple(by_row.get(row) for row in rows)
    queries = []
    for supplied in target_regions:
        targets = _predecessor_domain(rows, complete, origin, grid, supplied)
        if supplied is None:
            raise TypeError("each target query must be an explicit nonempty tuple of zones")
        if any(zone.epi not in by_row or not _dbm_subset(zone.bounds, by_row[zone.epi]) for zone in targets):
            raise ValueError("every target region must stay inside the declared domain")
        target_by_row = {zone.epi: zone.bounds for zone in targets}
        zones = tuple(target_by_row.get(row) for row in rows)
        present = any(_dbm_contains_origin(zone) for zone in zones)
        queries.append(dict(
            targets=targets, zones=zones, records=[C6CarriedRegionIteration(0, len(targets), present)],
            status="origin_not_excluded" if present else "active", intersections=0,
        ))
    work = _Work(maximum)
    while any(query["status"] == "active" for query in queries) and work.count < maximum:
        for query in queries:
            if query["status"] != "active":
                continue
            following = [None] * len(rows)
            targets = tuple(zone for zone in query["zones"] if zone is not None)
            previous_work = work.count
            try:
                for source, (cell, shift) in enumerate(zip(cells, shifts, strict=True)):
                    if cell is None:
                        continue
                    for target in targets:
                        translated = tuple(tuple(target[i][j] - shift[i] + shift[j] for j in range(7))
                                           for i in range(7))
                        if any(translated[i][6] < -cell[6][i]
                               or cell[i][6] < -translated[6][i] for i in range(6)):
                            continue
                        work.charge()
                        piece = _dbm_intersection(cell, translated)
                        if piece is not None:
                            following[source] = _dbm_join(following[source], piece)
            except _Limit:
                query["intersections"] += work.count - previous_work
                query["status"] = "resource_limit"
                break
            query["intersections"] += work.count - previous_work
            following = tuple(following)
            if any(not _dbm_subset(zone, cell) or zone is not None and _dbm_close(zone) != zone
                   for zone, cell in zip(following, cells, strict=True)):
                raise RuntimeError("an abstract predecessor hull escaped its exact source domain")
            present = any(_dbm_contains_origin(zone) for zone in following)
            count = sum(zone is not None for zone in following)
            query["records"].append(C6CarriedRegionIteration(len(query["records"]), count, present))
            if present:
                query["status"] = "origin_not_excluded"
            elif not count:
                query["status"] = "empty_complete_layer"
            elif following == query["zones"]:
                query["status"] = "stationary_complete_layer"
            query["zones"] = following

    def public(zones):
        return tuple(C6CarriedForwardZone(row, zone)
                     for row, zone in zip(rows, zones, strict=True) if zone is not None)

    results = tuple(C6CarriedRegionExclusion(
        query["targets"], public(query["zones"]), tuple(query["records"]),
        "resource_limit" if query["status"] == "active" else query["status"], query["intersections"],
    ) for query in queries)
    return C6CarriedRegionExclusions(
        ref, state, h, rows, pressures, grid, origin, domain, results, work.count, maximum, cell_limit,
    )


@dataclass(frozen=True, slots=True)
class C6CarriedReachableIteration:
    """One complete origin-containing forward hull inside the declared domain."""

    ordinal: int
    zone_count: int
    origin_retained: bool


@dataclass(frozen=True, slots=True)
class C6CarriedReachableEnvelope:
    """An exact outer envelope for every origin path confined to a domain.

    Each retained layer contains the unchanged origin and satisfies
    F(R) intersect D subset R. This clipping is to the declared domain D,
    not necessarily the original RN cube or the physical band. Actual
    trajectories are not assumed to remain in D. In particular an exact
    abstract fixed point does not certify unbounded-time trapping.
    """

    reference: C6PressureLatticeReference
    state: NodalRemainderState
    timestep: float
    epi_states: tuple[tuple[float, ...], ...]
    pressures: tuple[tuple[float, ...], ...]
    grid_quantum: F
    affine_origin: tuple[F, ...]
    domain_zones: tuple[C6CarriedForwardZone, ...]
    retained_zones: tuple[C6CarriedForwardZone, ...]
    iterations: tuple[C6CarriedReachableIteration, ...]
    status: str
    intersections: int
    max_intersections: int
    max_cells: int

    @property
    def domain_confined_paths_covered(self) -> bool:
        return True

    @property
    def clipped_forward_inclusion_certified(self) -> bool:
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


def derive_c6_carried_reachable_envelope(
    reference: C6PressureLatticeReference, *, state: NodalRemainderState,
    epi_states: tuple[tuple[float, ...], ...], timestep: float,
    domain_zones: tuple[C6CarriedForwardZone, ...] | None = None,
    max_intersections: int = 250_000, max_cells: int = 4096,
) -> C6CarriedReachableEnvelope:
    """Close a canonical carried domain under origin-containing image hulls.

    R_0=D and R_next=hull_per_RN_cell({origin} union (F(R) intersect D)).
    Monotonicity makes the complete layers descend, while explicit origin
    injection preserves every finite origin path staying in D. The method
    stops only at exact matrix equality or its computational guard. It
    executes no trajectory and never promotes clipping to domain invariance.
    """
    maximum = _positive_integer(max_intersections, "max_intersections")
    cell_limit = _positive_integer(max_cells, "max_cells")
    ref, origin, h, rows, pressures, areas, complete = _prepare_carried_family(
        reference, state=state, epi_states=epi_states, timestep=timestep,
        row_limit=cell_limit, row_limit_label="max_cells",
    )
    denominator = math.lcm(*(value.denominator for row in areas for value in row))
    grid = F(math.gcd(*(int(value * denominator) for row in areas for value in row)), denominator) or _GRID
    exact_shifts = tuple(tuple(value / grid for value in row) for row in areas)
    if any(value.denominator != 1 for row in exact_shifts for value in row):
        raise RuntimeError("the reachable envelope lost its exact common increment lattice")
    shifts = tuple(tuple(map(int, row)) + (0,) for row in exact_shifts)
    domain = _predecessor_domain(rows, complete, origin, grid, domain_zones)
    by_row = {zone.epi: zone.bounds for zone in domain}
    if state.epi not in by_row or not _dbm_contains_origin(by_row[state.epi]):
        raise ValueError("the declared reachable domain must contain the unchanged supplied origin")
    origin_index = rows.index(state.epi)
    cells = tuple(by_row.get(row) for row in rows)
    zones = cells
    records = [C6CarriedReachableIteration(0, len(domain), True)]
    work = _Work(maximum)
    status = "resource_limit"
    while True:
        following = [None] * len(rows)
        following[origin_index] = ((0,) * 7,) * 7
        try:
            for zone, shift in zip(zones, shifts, strict=True):
                if zone is None:
                    continue
                image = tuple(tuple(zone[i][j] + shift[i] - shift[j] for j in range(7)) for i in range(7))
                for target, cell in enumerate(cells):
                    if cell is None or any(image[i][6] < -cell[6][i]
                                           or cell[i][6] < -image[6][i] for i in range(6)):
                        continue
                    work.charge()
                    piece = _dbm_intersection(image, cell)
                    if piece is not None:
                        following[target] = _dbm_join(following[target], piece)
        except _Limit:
            break
        following = tuple(following)
        if (not _dbm_contains_origin(following[origin_index])
                or any(not _dbm_subset(after, before) for after, before in zip(following, zones, strict=True))):
            raise RuntimeError("the reachable envelope lost its origin or descending inclusion")
        records.append(C6CarriedReachableIteration(len(records), sum(z is not None for z in following), True))
        if following == zones:
            status = "fixed_point"
            break
        zones = following
    retained = tuple(C6CarriedForwardZone(row, zone)
                     for row, zone in zip(rows, zones, strict=True) if zone is not None)
    return C6CarriedReachableEnvelope(
        ref, state, h, rows, pressures, grid, origin, domain, retained,
        tuple(records), status, work.count, maximum, cell_limit,
    )
