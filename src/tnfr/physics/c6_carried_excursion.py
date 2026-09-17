"""Exact origin-to-region exclusions through monotone C6 excursions.

A linear proof coordinate increases on a declared collection of RN cells.
Every entry from the other cells is bounded against the target regions.
Only the finite initial visit can then require shared-kernel replay. The
coordinate is a certificate, not a new pressure or a physical parameter.
"""

from dataclasses import dataclass
from fractions import Fraction as F
import math

import networkx as nx

from ..dynamics._euler_kernel import (
    NodalRemainderState, NodalRemainderStep, _binary64_tuple,
    _validate_nodal_remainder_state, advance_nodal_remainder,
)
from .c6_carried_viability import (
    C6CarriedForwardZone, _GRID, _dbm_close, _dbm_contains_origin,
    _dbm_intersection, _dbm_subset, _positive_integer, _predecessor_domain,
    _prepare_carried_family,
)
from .c6_pressure_lattice import C6PressureLatticeReference, _observe_rebuilt_c6_pressure_lattice

__all__ = [
    "C6CarriedLinearExtremum", "C6CarriedExcursionIngress",
    "C6CarriedExcursionExclusion", "derive_c6_carried_excursion_exclusion",
    "C6CarriedExcursionTransition", "C6CarriedModeExcursionExclusion",
    "derive_c6_carried_mode_excursion_exclusion",
]


@dataclass(frozen=True, slots=True)
class C6CarriedLinearExtremum:
    """An exact integer-zone extremum with matching primal and dual evidence.

    For upper extrema, dual_flow balances (weights, -sum(weights)) over the
    seven DBM indices. For lower extrema it balances their negatives. Every
    flow (i,j,amount) uses the inequality k_i-k_j<=bounds[i][j]. Its signed
    cost equals value, and point attains value inside the original zone.
    """

    zone: C6CarriedForwardZone
    weights: tuple[int, ...]
    sense: str
    value: int
    point: tuple[int, ...]
    dual_flow: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True, slots=True)
class C6CarriedExcursionIngress:
    """One complete outside-to-active image intersection and its lower bound."""

    source_epi: tuple[float, ...]
    target_epi: tuple[float, ...]
    intersection: C6CarriedForwardZone
    lower: C6CarriedLinearExtremum


@dataclass(frozen=True, slots=True)
class C6CarriedExcursionExclusion:
    """Scoped target exclusion, or an explicit unresolved proof gate.

    Successful exclusion concerns finite paths from the unchanged supplied
    origin which stay in domain_zones. It neither proves that domain invariant
    nor excludes visits after leaving and returning to it. The common grid
    relaxes finer coordinate cosets, so the universal bounds are conservative.
    The retained prefix, when needed, uses actual exact carried states and
    fresh canonical pressure. A domain/band departure settles only the absence
    of a wholly domain-confined origin-to-target path.
    """

    reference: C6PressureLatticeReference
    state: NodalRemainderState
    timestep: float
    epi_states: tuple[tuple[float, ...], ...]
    pressures: tuple[tuple[float, ...], ...]
    grid_quantum: F
    affine_origin: tuple[F, ...]
    active_epi_states: tuple[tuple[float, ...], ...]
    weights: tuple[int, ...]
    domain_zones: tuple[C6CarriedForwardZone, ...]
    target_regions: tuple[C6CarriedForwardZone, ...]
    minimum_drift: int | None
    target_upper: int
    ingress_lower: int | None
    ingress: tuple[C6CarriedExcursionIngress, ...]
    domain_extrema: tuple[tuple[C6CarriedLinearExtremum, C6CarriedLinearExtremum], ...]
    target_extrema: tuple[C6CarriedLinearExtremum, ...]
    prefix_deadline: int | None
    max_prefix_steps: int
    max_cells: int
    prefix_steps: tuple[NodalRemainderStep, ...]
    endpoint: NodalRemainderState
    endpoint_potential: int | None
    terminal_exact_candidate: tuple[F, ...] | None
    exact_total_area: tuple[F, ...]
    nodal_balance_residual: tuple[F, ...]
    status: str

    @property
    def origin_path_within_domain_excluded(self) -> bool:
        return self.status in (
            "initially_inactive", "initially_above_targets", "initial_visit_ended",
            "cleared_initial_budget", "prefix_left_domain", "prefix_band_exit",
        )

    @property
    def actual_target_reached(self) -> bool:
        return self.status == "target_reached"

    @property
    def observed_prefix_steps(self) -> int:
        return len(self.prefix_steps)

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


@dataclass(frozen=True, slots=True)
class C6CarriedExcursionTransition:
    """One complete active-cell transition and its constant potential change.

    intersection is expressed in the target coordinates. Translating it by
    the negative source increment gives every admitted source for this edge.
    """

    source_epi: tuple[float, ...]
    target_epi: tuple[float, ...]
    intersection: C6CarriedForwardZone
    potential_increment: int


@dataclass(frozen=True, slots=True)
class C6CarriedModeExcursionExclusion(C6CarriedExcursionExclusion):
    """A common linear coordinate with exact offsets for the active RN cells.

    V_c(k)=weights dot k+cell_offsets[c]. The inherited extrema retain their
    linear primal/dual evidence; add the appropriate cell offset to obtain
    V. target_upper, ingress_lower and reported prefix potentials already
    include those offsets. An inactive endpoint has endpoint_potential=None.
    minimum_drift=None means no active-to-active domain transition exists:
    each active visit then contains at most one domain point, so checking an
    initial visit requires at most one step and invents no positive drift.
    """

    cell_offsets: tuple[tuple[tuple[float, ...], int], ...]
    initial_potential: int | None
    active_transitions: tuple[C6CarriedExcursionTransition, ...]


def _linear_extremum(zone, weights, sense):
    """Solve a finite integer transport dual and check an attaining primal."""
    original = weights + (-sum(weights),)
    signed = original if sense == "upper" else tuple(-value for value in original)
    graph = nx.DiGraph()
    for i, amount in enumerate(signed):
        if amount:
            graph.add_node(i, demand=-amount)
    for i, source in enumerate(signed):
        for j, sink in enumerate(signed):
            if source > 0 > sink:
                graph.add_edge(i, j, weight=zone.bounds[i][j], capacity=min(source, -sink))
    allocation = nx.min_cost_flow(graph) if graph else {}
    flow = tuple((i, j, amount) for i, row in sorted(allocation.items())
                 for j, amount in sorted(row.items()) if amount)
    balance = [0] * 7
    cost = 0
    tight = [list(row) for row in zone.bounds]
    for i, j, amount in flow:
        if type(amount) is not int or amount <= 0:
            raise RuntimeError("the linear-bound dual lost positive integer flow")
        balance[i] += amount
        balance[j] -= amount
        cost += amount * zone.bounds[i][j]
        tight[j][i] = min(tight[j][i], -zone.bounds[i][j])
    if tuple(balance) != signed:
        raise RuntimeError("the linear-bound dual lost its exact coefficient balance")
    closed = _dbm_close(tight)
    if closed is None:
        raise RuntimeError("the linear-bound dual has no matching primal point")
    point = tuple(closed[i][6] for i in range(6)) + (0,)
    if any(point[i] - point[j] > zone.bounds[i][j] for i in range(7) for j in range(7)):
        raise RuntimeError("the linear-bound primal escaped its declared zone")
    value = cost if sense == "upper" else -cost
    if sum(weight * coordinate for weight, coordinate in zip(weights, point)) != value:
        raise RuntimeError("the linear-bound primal and dual objectives differ")
    return C6CarriedLinearExtremum(zone, weights, sense, value, point[:6], flow)


def _translated_zone(bounds, shift):
    extended = shift + (0,)
    return tuple(tuple(bounds[i][j] + extended[i] - extended[j] for j in range(7)) for i in range(7))


def _contains_coordinates(bounds, coordinates):
    values = coordinates + (0,)
    return all(values[i] - values[j] <= bounds[i][j] for i in range(7) for j in range(7))


def derive_c6_carried_excursion_exclusion(
    reference: C6PressureLatticeReference, *, state: NodalRemainderState,
    epi_states: tuple[tuple[float, ...], ...], timestep: float,
    active_epi_states: tuple[tuple[float, ...], ...], weights: tuple[int, ...],
    target_regions: tuple[C6CarriedForwardZone, ...],
    domain_zones: tuple[C6CarriedForwardZone, ...] | None = None,
    max_prefix_steps: int = 4096, max_cells: int = 4096,
) -> C6CarriedExcursionExclusion:
    """Exclude targets using positive drift during each visit to active cells.

    Integer coordinates satisfy X=origin+grid*k. Fresh canonical increments
    give d=min_active(weights dot increment/grid). Target zones have upper
    bound M; every outside-to-active domain transition has lower bound L.
    If d>0 and L>M (or no ingress exists), no later active visit reaches a
    target. An initially active origin has potential zero. Its only possible
    target source ordinals are 0 through floor(M/d); the exact shared replay
    stops as soon as that initial visit ends, its potential exceeds M, a target
    is reached, the domain is left, or the computational prefix guard is hit.

    All RN, band, source, affine-grid and relational-domain premises are rebuilt.
    Proof weights change neither the physical source nor the actual nodal map.
    """
    return _derive_excursion(
        reference, state=state, epi_states=epi_states, timestep=timestep,
        active_epi_states=active_epi_states, weights=weights, target_regions=target_regions,
        domain_zones=domain_zones, max_prefix_steps=max_prefix_steps, max_cells=max_cells,
        mode_dependent=False, cell_offsets=None,
    )


def derive_c6_carried_mode_excursion_exclusion(
    reference: C6PressureLatticeReference, *, state: NodalRemainderState,
    epi_states: tuple[tuple[float, ...], ...], timestep: float,
    active_epi_states: tuple[tuple[float, ...], ...], weights: tuple[int, ...],
    target_regions: tuple[C6CarriedForwardZone, ...],
    cell_offsets: tuple[tuple[tuple[float, ...], int], ...] | None = None,
    domain_zones: tuple[C6CarriedForwardZone, ...] | None = None,
    max_prefix_steps: int = 4096, max_cells: int = 4096,
) -> C6CarriedModeExcursionExclusion:
    """Verify a common-gradient, cell-offset excursion exclusion.

    Source, RN, grid, extrema and the exact prefix kernel are shared with the
    single-coordinate owner. An offset tuple must cover every active row
    exactly once; None supplies zero offsets. Rational proof coefficients
    can be multiplied by one positive common denominator before calling.

    Every admitted active edge c->d has the exact change
    weights dot a_c+b_d-b_c. A positive minimum, separated ingress and an
    initial potential V0 give the derived deadline floor((M-V0)/d)+1.
    The final transition out of the active family needs no drift bound.
    With no internal edges an initial visit needs at most one checked step.
    """
    return _derive_excursion(
        reference, state=state, epi_states=epi_states, timestep=timestep,
        active_epi_states=active_epi_states, weights=weights, target_regions=target_regions,
        domain_zones=domain_zones, max_prefix_steps=max_prefix_steps, max_cells=max_cells,
        mode_dependent=True, cell_offsets=cell_offsets,
    )


def _derive_excursion(
    reference, *, state, epi_states, timestep, active_epi_states, weights,
    target_regions, domain_zones, max_prefix_steps, max_cells, mode_dependent, cell_offsets,
):
    maximum = _positive_integer(max_prefix_steps, "max_prefix_steps")
    cell_limit = _positive_integer(max_cells, "max_cells")
    if type(weights) is not tuple or len(weights) != 6 or any(type(value) is not int for value in weights):
        raise TypeError("weights must be a six-tuple of exact integers")
    ref, origin, h, rows, pressures, areas, complete = _prepare_carried_family(
        reference, state=state, epi_states=epi_states, timestep=timestep,
        row_limit=cell_limit, row_limit_label="max_cells",
    )
    denominator = math.lcm(*(value.denominator for row in areas for value in row))
    grid = F(math.gcd(*(int(value * denominator) for row in areas for value in row)), denominator) or _GRID
    exact_shifts = tuple(tuple(value / grid for value in row) for row in areas)
    if any(value.denominator != 1 for row in exact_shifts for value in row):
        raise RuntimeError("the excursion source lost its exact increment lattice")
    shifts = tuple(tuple(map(int, row)) for row in exact_shifts)
    domain = _predecessor_domain(rows, complete, origin, grid, domain_zones)
    by_row = {zone.epi: zone for zone in domain}
    if state.epi not in by_row or not _dbm_contains_origin(by_row[state.epi].bounds):
        raise ValueError("the declared excursion domain must contain the unchanged supplied origin")
    if type(active_epi_states) is not tuple or not active_epi_states or len(active_epi_states) > cell_limit:
        raise ValueError("active_epi_states must be a nonempty tuple within max_cells")
    active = tuple(_binary64_tuple(row, "active epi row") for row in active_epi_states)
    if len(set(active)) != len(active) or any(row not in by_row for row in active):
        raise ValueError("active rows must be distinct nonempty declared domain rows")
    offset_by_row = {row: 0 for row in active}
    if mode_dependent and cell_offsets is not None:
        if type(cell_offsets) is not tuple or len(cell_offsets) != len(active):
            raise ValueError("cell_offsets must be a tuple covering every active row exactly once")
        seen = set()
        for item in cell_offsets:
            if type(item) is not tuple or len(item) != 2 or type(item[1]) is not int:
                raise TypeError("each cell offset must pair an exact epi tuple and an exact integer")
            row = _binary64_tuple(item[0], "cell offset epi row")
            if row not in offset_by_row or row in seen:
                raise ValueError("cell_offsets must identify distinct active rows")
            seen.add(row)
            offset_by_row[row] = item[1]
    if target_regions is None:
        raise TypeError("target_regions must be an explicit nonempty tuple of zones")
    targets = _predecessor_domain(rows, complete, origin, grid, target_regions)
    if any(zone.epi not in active or not _dbm_subset(zone.bounds, by_row[zone.epi].bounds) for zone in targets):
        raise ValueError("every target region must stay within one active domain zone")
    position = {row: index for index, row in enumerate(rows)}
    transitions = []
    if mode_dependent:
        for source in active:
            image = _translated_zone(by_row[source].bounds, shifts[position[source]])
            for target in active:
                cell = by_row[target].bounds
                if any(image[i][6] < -cell[6][i] or cell[i][6] < -image[6][i] for i in range(6)):
                    continue
                piece = _dbm_intersection(image, cell)
                if piece is not None:
                    change = (sum(w * a for w, a in zip(weights, shifts[position[source]]))
                              + offset_by_row[target] - offset_by_row[source])
                    transitions.append(C6CarriedExcursionTransition(
                        source, target, C6CarriedForwardZone(target, piece), change,
                    ))
        drift = min((item.potential_increment for item in transitions), default=None)
    else:
        drift = min(sum(w * a for w, a in zip(weights, shifts[position[row]])) for row in active)
    domain_extrema = tuple((_linear_extremum(by_row[row], weights, "lower"),
                            _linear_extremum(by_row[row], weights, "upper")) for row in active)
    target_extrema = tuple(_linear_extremum(zone, weights, "upper") for zone in targets)
    target_upper = max(item.value + offset_by_row[item.zone.epi] for item in target_extrema)
    ingress = []
    for source in domain:
        if source.epi in active:
            continue
        image = _translated_zone(source.bounds, shifts[position[source.epi]])
        for row in active:
            cell = by_row[row].bounds
            if any(image[i][6] < -cell[6][i] or cell[i][6] < -image[6][i] for i in range(6)):
                continue
            piece = _dbm_intersection(image, cell)
            if piece is not None:
                zone = C6CarriedForwardZone(row, piece)
                ingress.append(C6CarriedExcursionIngress(
                    source.epi, row, zone, _linear_extremum(zone, weights, "lower"),
                ))
    ingress = tuple(ingress)
    ingress_lower = min((item.lower.value + offset_by_row[item.target_epi] for item in ingress), default=None)
    initial_potential = offset_by_row.get(state.epi)
    if drift is not None and drift <= 0:
        deadline = None
    elif state.epi not in active or initial_potential > target_upper:
        deadline = 0
    elif drift is None:
        deadline = 1
    else:
        deadline = max(0, (target_upper - initial_potential) // drift + 1)
    steps, current, total, terminal = [], state, (F(0),) * 6, None

    def coordinates(value):
        exact = _validate_nodal_remainder_state(value)
        result = tuple((x - start) / grid for x, start in zip(exact, origin, strict=True))
        if any(item.denominator != 1 for item in result):
            raise RuntimeError("a carried excursion prefix left its exact increment coset")
        return tuple(map(int, result))

    def finish(status):
        endpoint_coordinates = coordinates(current)
        residual = tuple(x - start - area for x, start, area in zip(current.exact_epi, origin, total, strict=True))
        if any(residual):
            raise RuntimeError("the excursion prefix lost its complete nodal telescope")
        endpoint_value = sum(w * x for w, x in zip(weights, endpoint_coordinates))
        if mode_dependent:
            endpoint_value = endpoint_value + offset_by_row[current.epi] if current.epi in active else None
        arguments = (
            ref, state, h, rows, pressures, grid, origin, active, weights, domain, targets,
            drift, target_upper, ingress_lower, ingress, domain_extrema, target_extrema,
            deadline, maximum, cell_limit, tuple(steps), current,
            endpoint_value, terminal, total, residual, status,
        )
        if mode_dependent:
            return C6CarriedModeExcursionExclusion(
                *arguments, cell_offsets=tuple((row, offset_by_row[row]) for row in active),
                initial_potential=initial_potential, active_transitions=tuple(transitions),
            )
        return C6CarriedExcursionExclusion(*arguments)

    if any(zone.epi == state.epi and _dbm_contains_origin(zone.bounds) for zone in targets):
        return finish("target_reached")
    if drift is not None and drift <= 0:
        return finish("nonpositive_drift")
    if ingress_lower is not None and ingress_lower <= target_upper:
        return finish("ingress_gap_not_strict")
    if state.epi not in active:
        return finish("initially_inactive")
    if initial_potential > target_upper:
        return finish("initially_above_targets")
    while True:
        point = coordinates(current)
        if current.epi not in by_row or not _contains_coordinates(by_row[current.epi].bounds, point):
            return finish("prefix_left_domain")
        if any(zone.epi == current.epi and _contains_coordinates(zone.bounds, point) for zone in targets):
            return finish("target_reached")
        if current.epi not in active:
            return finish("initial_visit_ended")
        potential = sum(w * x for w, x in zip(weights, point)) + offset_by_row[current.epi]
        if potential > target_upper:
            return finish("cleared_initial_budget")
        if len(steps) >= deadline:
            raise RuntimeError("the excursion prefix contradicted its strict drift deadline")
        if len(steps) == maximum:
            return finish("prefix_resource_limit")
        pressure = _observe_rebuilt_c6_pressure_lattice(ref, current.epi).pressure
        index = position[current.epi]
        if pressure != pressures[index]:
            raise RuntimeError("a refreshed excursion pressure differs from its declared canonical row")
        added = tuple(F(h) * F(value) for value in pressure)
        candidate = tuple(x + area for x, area in zip(current.exact_epi, added, strict=True))
        if any(not F(state.epi_lower) <= x <= F(state.epi_upper) for x in candidate):
            terminal = candidate
            return finish("prefix_band_exit")
        step = advance_nodal_remainder(current, timestep=h, capacity=(1.,) * 6, pressure=pressure)
        if step.after.exact_epi != candidate or step.exact_increment != added or any(step.nodal_balance_residual):
            raise RuntimeError("the shared excursion step lost its exact nodal area")
        next_point = coordinates(step.after)
        next_row = step.after.epi
        if mode_dependent:
            if next_row in active and _contains_coordinates(by_row[next_row].bounds, next_point):
                change = (sum(w * (y - x) for w, x, y in zip(weights, point, next_point))
                          + offset_by_row[next_row] - offset_by_row[current.epi])
                if drift is None or change < drift:
                    raise RuntimeError("an active excursion transition violated its derived edge drift")
        elif sum(w * (y - x) for w, x, y in zip(weights, point, next_point)) < drift:
            raise RuntimeError("an actual excursion step violated its derived minimum drift")
        steps.append(step)
        total = tuple(old + area for old, area in zip(total, added, strict=True))
        current = step.after
