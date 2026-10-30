"""Obstruction to trapping by finite unions of complete carried C6 cells.

The visible states may be arbitrarily correlated. The obstruction concerns
the independent complete admissible carry cell attached to each visible
state. A candidate that restricts carry jointly with shape remains open.
No saved trajectory is changed or claimed to visit the supplied witnesses.
"""

from dataclasses import dataclass
from fractions import Fraction as F

from ..dynamics._euler_kernel import (
    NODAL_REMAINDER_ABSOLUTE_BOUND,
    NODAL_REMAINDER_DENOMINATOR_BITS,
    NodalRemainderState,
    NodalRemainderStep,
    _binary64_tuple,
    _validate_nodal_remainder_state,
    advance_nodal_remainder,
)
from .c6_carried_closure import C6CarriedClosure, derive_c6_carried_closure
from .c6_pressure_lattice import (
    _observe_rebuilt_c6_pressure_lattice,
    _pressure_at_gradient_index,
)
from .nodal_remainder import derive_nodal_remainder_itinerary

__all__ = [
    "C6CarriedCompleteCellObstruction",
    "derive_c6_carried_complete_cell_obstruction",
    "C6CarriedCompleteCellEscape",
    "observe_c6_carried_complete_cell_escape",
]


@dataclass(frozen=True, slots=True)
class C6CarriedCompleteCellObstruction:
    """A class-wide obstruction, not a trajectory escape deadline.

    Consider any nonempty finite family of displayed EPI tuples in the
    declared band whose six gradient indices lie in the displayed bounds.
    Attach *every* legal carry encoding to every tuple. Such a union cannot
    be forward invariant under the fixed source, unit capacity and timestep.

    Choose a tuple maximizing the sum of its displayed coordinates and
    the greatest admissible dyadic input in each of its six rounding cells.
    At least one pressure is positive by the lattice sign theorem. Every
    negative increment is smaller than the minimum cell width, so no
    displayed coordinate decreases. The represented h*p increments lie
    on the shared dyadic grid; each positive one therefore moves beyond
    the greatest admissible grid point. Positive coordinates either increase
    strictly or leave the declared band. The candidate therefore escapes
    in one step for this supplied witness. This is an existence claim,
    not escape for every carry or for the saved carried trajectory.

    Gradient bounds are necessary consequences of the spatial closure,
    but this larger full-cell class need not satisfy its energy bound at
    every carry. The theorem does not exclude its energy-trimmed subcells.
    """

    closure: C6CarriedClosure
    grid_quantum: F
    minimum_cell_grid_width: F
    pressure_lower: tuple[F, ...]
    pressure_upper: tuple[F, ...]
    maximum_negative_increment: F

    @property
    def finite_complete_cell_union_invariance_excluded(self) -> bool:
        return True

    @property
    def correlated_carry_region_excluded(self) -> bool:
        return False

    @property
    def saved_trajectory_escape_certified(self) -> bool:
        return False

    @property
    def future_runtime_certified(self) -> bool:
        return False


def _closure(value):
    if type(value) is not C6CarriedClosure:
        raise TypeError("closure must be a C6CarriedClosure")
    tube = value.base_tube
    return derive_c6_carried_closure(
        tube.contraction.profile,
        state=tube.state,
        timestep=tube.contraction.timestep,
    )


def derive_c6_carried_complete_cell_obstruction(
    closure,
) -> C6CarriedCompleteCellObstruction:
    """Derive uniform sign and held-negative-cell bounds from existing owners.

    Positive coefficients and monotone nearest-even rounding make the
    represented pressure monotone in its integer gradient. Evaluate both
    endpoints of each rebuilt B37 interval through the already-proved
    source-plus-rounded-product identity. No pressure samples are fitted.

    The band endpoints are represented, its width is at least one lattice
    quantum delta, and it lies in the lattice slab. Every band-clipped
    rounding cell has real width at least delta/2. The two possible open
    endpoints remove at most two shared dyadic-grid quanta. The explicit
    carry bound is inactive on these cells because their half width is no
    larger than the slab's delta. This conservative lower width also
    covers cells touching either physical band endpoint.
    """
    closed = _closure(closure)
    tube = closed.base_tube
    lattice = tube.contraction.profile.lattice
    if not lattice.every_pressure_has_positive:
        raise ValueError("the fixed lattice must exclude all-nonpositive pressure")
    lower, upper = F(tube.state.epi_lower), F(tube.state.epi_upper)
    delta = lattice.epi_quantum
    grid = F(1, 2**NODAL_REMAINDER_DENOMINATOR_BITS)
    if upper - lower < delta:
        raise ValueError(
            "the displayed band must span at least one EPI lattice quantum"
        )
    if delta > NODAL_REMAINDER_ABSOLUTE_BOUND:
        raise ValueError("the slab must leave the shared carry bound inactive")
    minimum_width = delta / 2 - 2 * grid
    pressure_lower = tuple(
        F(_pressure_at_gradient_index(lattice, i, m))
        for i, m in enumerate(closed.gradient_index_lower)
    )
    pressure_upper = tuple(
        F(_pressure_at_gradient_index(lattice, i, m))
        for i, m in enumerate(closed.gradient_index_upper)
    )
    h = F(tube.contraction.timestep)
    maximum = h * max(F(0), *(-p for p in pressure_lower))
    if not maximum < minimum_width:
        raise ValueError(
            "negative nodal increments must be smaller than each admissible cell width"
        )
    return C6CarriedCompleteCellObstruction(
        closed,
        grid,
        minimum_width,
        pressure_lower,
        pressure_upper,
        maximum,
    )


@dataclass(frozen=True, slots=True)
class C6CarriedCompleteCellEscape:
    """One hypothetical full-cell witness, with an unchanged fixed source.

    The witness is intentionally selected from the complete carry fiber.
    Its carry is an existence input, not a continuation of a saved state.
    A band-failure witness is recorded as exact candidate coordinates;
    the shared nodal kernel is only invoked for an admissible endpoint.
    """

    obstruction: C6CarriedCompleteCellObstruction
    epi_states: tuple[tuple[float, ...], ...]
    selected_index: int
    state: NodalRemainderState
    pressure: tuple[float, ...]
    exact_increment: tuple[F, ...]
    exact_candidate: tuple[F, ...]
    endpoint: NodalRemainderStep | None
    band_failure: bool
    increased_coordinates: tuple[int, ...]
    source_visible_sum: F
    endpoint_visible_sum: F | None

    @property
    def family_invariance_excluded(self) -> bool:
        return True

    @property
    def saved_trajectory_escape_certified(self) -> bool:
        return False


def observe_c6_carried_complete_cell_escape(
    obstruction,
    *,
    epi_states: tuple[tuple[float, ...], ...],
) -> C6CarriedCompleteCellEscape:
    """Construct a legal counterexample to a supplied complete-cell family.

    This verifies every supplied visible state's pressure and gradient
    bounds. It then uses the existing zero-area inverse itinerary to get
    exact grid extrema, including RN tie parity and physical-band clipping.
    No graph write, saved-carry reset, or new trajectory search is performed.
    """
    if type(obstruction) is not C6CarriedCompleteCellObstruction:
        raise TypeError("obstruction must be a C6CarriedCompleteCellObstruction")
    bound = derive_c6_carried_complete_cell_obstruction(obstruction.closure)
    if type(epi_states) is not tuple or not epi_states:
        raise ValueError("a nonempty ordered tuple of displayed cells is required")
    states = tuple(_binary64_tuple(epi, "epi_states row") for epi in epi_states)
    if len(set(states)) != len(states):
        raise ValueError("displayed cells must be distinct")
    tube = bound.closure.base_tube
    lattice = tube.contraction.profile.lattice
    low, high = tube.state.epi_lower, tube.state.epi_upper
    readings = tuple(
        _observe_rebuilt_c6_pressure_lattice(lattice, epi) for epi in states
    )
    for reading in readings:
        if any(not low <= x <= high for x in reading.epi):
            raise ValueError(
                "every displayed cell must lie in the closure's declared band"
            )
        if any(
            not lo <= m <= hi
            for lo, m, hi in zip(
                bound.closure.gradient_index_lower,
                reading.gradient_indices,
                bound.closure.gradient_index_upper,
                strict=True,
            )
        ):
            raise ValueError(
                "every displayed cell must satisfy the rebuilt gradient bounds"
            )
        if any(
            not lo <= F(p) <= hi
            for lo, p, hi in zip(
                bound.pressure_lower,
                reading.pressure,
                bound.pressure_upper,
                strict=True,
            )
        ):
            raise RuntimeError(
                "the canonical pressure lost its monotone interval bound"
            )
    selected = max(range(len(states)), key=lambda i: sum(map(F, states[i]), F(0)))
    epi, pressure = states[selected], readings[selected].pressure
    cells = derive_nodal_remainder_itinerary(
        epi_states=(epi, epi),
        timesteps=(0.0,),
        capacities=((1.0,) * 6,),
        pressures=(pressure,),
        epi_lower=low,
        epi_upper=high,
    ).coordinates
    exact = tuple(cell.last_grid_index * bound.grid_quantum for cell in cells)
    state = NodalRemainderState(
        epi, tuple(x - F(v) for x, v in zip(exact, epi, strict=True)), low, high
    )
    if _validate_nodal_remainder_state(state) != exact:
        raise RuntimeError("the complete-cell extremum lost its shared nodal encoding")
    added = tuple(F(tube.contraction.timestep) * F(p) for p in pressure)
    candidate = tuple(x + a for x, a in zip(exact, added, strict=True))
    for cell, a in zip(cells, added, strict=True):
        width = (cell.last_grid_index - cell.first_grid_index) * bound.grid_quantum
        if width < bound.minimum_cell_grid_width or a < -width:
            raise RuntimeError(
                "the complete-cell width failed its uniform negative-step bound"
            )
    failure = any(not F(low) <= x <= F(high) for x in candidate)
    increased = tuple(
        i
        for i, (a, x, v) in enumerate(zip(added, candidate, epi, strict=True))
        if a > 0 and float(x) > v
    )
    if not any(a > 0 for a in added) or any(
        float(x) < v for x, v in zip(candidate, epi, strict=True)
    ):
        raise RuntimeError(
            "the maximal-carry witness lost its one-sided visible escape"
        )
    endpoint = None
    visible_sum = None
    before_sum = sum(map(F, epi), F(0))
    if not failure:
        endpoint = advance_nodal_remainder(
            state,
            timestep=tube.contraction.timestep,
            capacity=(1.0,) * 6,
            pressure=pressure,
        )
        visible_sum = sum(map(F, endpoint.after.epi), F(0))
        if not increased or visible_sum <= before_sum or endpoint.after.epi in states:
            raise RuntimeError(
                "the maximal visible-sum witness failed to escape its family"
            )
    return C6CarriedCompleteCellEscape(
        bound,
        states,
        selected,
        state,
        pressure,
        added,
        candidate,
        endpoint,
        failure,
        increased,
        before_sum,
        visible_sum,
    )
