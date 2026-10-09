"""Exact finite visibility-cell escape for the fixed carried C6 pressure map.

Pairwise cell transitions are existential. Their graph overapproximates
complete carried trajectories; witnesses on neighboring edges need not use
compatible carries. Acyclicity after removing self-loops nevertheless gives
a sound finite escape theorem when each held pressure is nonzero.
"""

from dataclasses import dataclass
from fractions import Fraction as F

from ..dynamics._euler_kernel import (
    NODAL_REMAINDER_DENOMINATOR_BITS,
    NodalRemainderState,
    _binary64_tuple,
    _finite_binary64,
)
from .c6_pressure_lattice import (
    C6PressureLatticeReference,
    _observe_rebuilt_c6_pressure_lattice,
    _rebuild_lattice,
)
from .nodal_remainder import (
    NodalRemainderCellHorizon,
    derive_nodal_remainder_cell_horizon,
    derive_nodal_remainder_itinerary,
)

__all__ = ["C6CarriedCellGraph", "derive_c6_carried_cell_graph"]

_CAPACITY = (1.0,) * 6


@dataclass(frozen=True, slots=True)
class C6CarriedCellGraph:
    """A conditional graph of all legal-carry transitions within finite cells.

    A finite deadline means every trajectory starting in this cell family
    must leave the family or fail the declared band by that step, provided
    phase, support, capacity, coefficients and timestep retain this fixed
    numerical map. It is not a claim that the complete band is exited.
    A nontrivial graph cycle or a stationary cell makes this sufficient
    escape test undecided; neither establishes a carried periodic orbit.
    No entry into this hypothetical family from the saved state is proved.
    """

    reference: C6PressureLatticeReference
    timestep: float
    epi_states: tuple[tuple[float, ...], ...]
    pressures: tuple[tuple[float, ...], ...]
    adjacency: tuple[tuple[bool, ...], ...]
    edge_witnesses: tuple[tuple[NodalRemainderState | None, ...], ...]
    infeasible_coordinate: tuple[tuple[int | None, ...], ...]
    maximal_residence: tuple[NodalRemainderCellHorizon, ...]
    nonself_topological_order: tuple[int, ...] | None
    family_exit_step_bounds: tuple[int, ...] | None
    maximum_steps_until_family_exit_or_band_failure: int | None

    @property
    def finite_family_escape_certified(self) -> bool:
        return self.maximum_steps_until_family_exit_or_band_failure is not None

    @property
    def carried_cycle_in_family_excluded(self) -> bool:
        return self.finite_family_escape_certified

    @property
    def whole_band_exit_certified(self) -> bool:
        return False

    @property
    def future_runtime_certified(self) -> bool:
        return False

    @property
    def reachable_from_saved_state_certified(self) -> bool:
        return False


def _itinerary(reference, source, target, timestep, pressure):
    return derive_nodal_remainder_itinerary(
        epi_states=(source, target),
        timesteps=(timestep,),
        capacities=(_CAPACITY,),
        pressures=(pressure,),
        epi_lower=reference.source.epi_lower,
        epi_upper=reference.source.epi_upper,
    )


def _maximal_residence(reference, epi, timestep, pressure):
    # The zero-area itinerary exposes the exact legal grid endpoints of
    # each source cell, including band clipping and nearest-even ties.
    cells = _itinerary(reference, epi, epi, 0.0, pressure).coordinates
    scale = 2**NODAL_REMAINDER_DENOMINATOR_BITS
    exact = tuple(
        F(cell.first_grid_index if p >= 0 else cell.last_grid_index, scale)
        for cell, p in zip(cells, pressure, strict=True)
    )
    state = NodalRemainderState(
        epi,
        tuple(value - F(visible) for value, visible in zip(exact, epi, strict=True)),
        reference.source.epi_lower,
        reference.source.epi_upper,
    )
    horizon = derive_nodal_remainder_cell_horizon(
        state=state,
        timestep=timestep,
        capacity=_CAPACITY,
        pressure=pressure,
    )
    # The chosen carry maximizes every directional distance at once.
    # This grid-width calculation verifies maximality, not only a witness.
    for cell, increment, limit in zip(
        cells, horizon.exact_increment, horizon.coordinate_step_limits, strict=True
    ):
        displacement = increment * scale
        if displacement.denominator != 1:
            raise RuntimeError("a shared nodal increment escaped the legal dyadic grid")
        maximum = (
            (cell.last_grid_index - cell.first_grid_index)
            // abs(displacement.numerator)
            if displacement
            else None
        )
        if maximum != limit:
            raise RuntimeError(
                "the extremal legal carry did not maximize cell residence"
            )
    return horizon


def _topological_order(adjacency):
    count = len(adjacency)
    indegree = [
        sum(adjacency[j][i] for j in range(count) if j != i) for i in range(count)
    ]
    ready = [i for i, value in enumerate(indegree) if not value]
    order = []
    while ready:
        source = ready.pop(0)
        order.append(source)
        for target, edge in enumerate(adjacency[source]):
            if target != source and edge:
                indegree[target] -= 1
                if not indegree[target]:
                    ready.append(target)
                    ready.sort()
    return tuple(order) if len(order) == count else None


def derive_c6_carried_cell_graph(
    reference: C6PressureLatticeReference,
    *,
    epi_states: tuple[tuple[float, ...], ...],
    timestep: float,
) -> C6CarriedCellGraph:
    """Certify all pairwise transitions of a supplied finite visible family.

    Rebuild the fixed C6 source and refresh its actual binary64 pressure
    at every supplied visible row. The shared inverse itinerary tests all
    N squared edges on the complete 2^-3222 carry grid, with no carry reset.
    For each row the directional extremal legal carry simultaneously
    maximizes all coordinate residence limits. The shared analytic horizon
    then bounds every possible self-loop run under pressure refresh.

    Remove self-loops. If the remaining graph is a DAG and all residence
    limits are finite, reverse topological induction gives
    D_i = first_exit_i + max(D_j for nonself successors j), taking max=0
    at a sink. Adjacent existential edges need not compose: enlarging the
    allowed path set can only make this upper deadline more conservative.
    No new trajectory from the research endpoint is executed or presumed.
    """
    ref = _rebuild_lattice(reference)
    h = _finite_binary64(timestep, "timestep")
    if h <= 0:
        raise ValueError("the cell graph requires a positive timestep")
    if type(epi_states) is not tuple or not epi_states:
        raise ValueError("epi_states must be a nonempty ordered tuple")
    rows = tuple(
        _binary64_tuple(row, f"epi_states[{i}]") for i, row in enumerate(epi_states)
    )
    if len(set(rows)) != len(rows):
        raise ValueError("the visible cell family must not contain duplicate rows")
    pressures = tuple(
        _observe_rebuilt_c6_pressure_lattice(ref, row).pressure for row in rows
    )
    adjacency, witnesses, exclusions = [], [], []
    for source, pressure in zip(rows, pressures, strict=True):
        edges = tuple(_itinerary(ref, source, target, h, pressure) for target in rows)
        adjacency.append(tuple(edge.feasible for edge in edges))
        witnesses.append(tuple(edge.witness_initial for edge in edges))
        exclusions.append(
            tuple(
                next(
                    (i for i, cell in enumerate(edge.coordinates) if not cell.feasible),
                    None,
                )
                for edge in edges
            )
        )
    graph = tuple(adjacency)
    residence = tuple(
        _maximal_residence(ref, row, h, pressure)
        for row, pressure in zip(rows, pressures, strict=True)
    )
    order = _topological_order(graph)
    bounds = None
    if order is not None and all(
        item.first_exit_step is not None for item in residence
    ):
        values = [0] * len(rows)
        for source in reversed(order):
            remaining = max(
                (
                    values[j]
                    for j, edge in enumerate(graph[source])
                    if j != source and edge
                ),
                default=0,
            )
            values[source] = residence[source].first_exit_step + remaining
        bounds = tuple(values)
    return C6CarriedCellGraph(
        ref,
        h,
        rows,
        pressures,
        graph,
        tuple(witnesses),
        tuple(exclusions),
        residence,
        order,
        bounds,
        max(bounds) if bounds is not None else None,
    )
