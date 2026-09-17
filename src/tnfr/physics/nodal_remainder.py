"""Exact prefix balances and feasible remainder-carrying nodal itineraries.

This observer executes supplied numeric inputs through the shared nodal
kernel and independently checks their nearest-even output cells. It does
not refresh pressure, write a graph or authenticate retained runtime data.
The inverse itinerary certificate intersects translated cells on the
shared dyadic grid and replays an admissible witness when one exists.
"""

from dataclasses import dataclass
from fractions import Fraction

from ..dynamics._euler_kernel import (
    NODAL_REMAINDER_ABSOLUTE_BOUND,
    NODAL_REMAINDER_DENOMINATOR_BITS,
    NodalRemainderState,
    NodalRemainderStep,
    _binary64_tuple,
    _finite_binary64,
    _remainder_band,
    _require_remainder_rounding,
    _validate_nodal_remainder_state,
    advance_nodal_remainder,
)
from .binary64_nodal_flow import Binary64AdditionCell, _rounding_cell

__all__ = [
    "NodalRemainderPrefix", "NodalRemainderSequence",
    "observe_nodal_remainder_sequence",
    "NodalRemainderCellHorizon", "derive_nodal_remainder_cell_horizon",
    "NodalRemainderCellExit", "observe_nodal_remainder_cell_exit",
    "NodalRemainderItineraryCell", "NodalRemainderItinerary",
    "derive_nodal_remainder_itinerary",
]


def _mean(values):
    return sum(values, Fraction(0)) / len(values)


@dataclass(frozen=True, slots=True)
class NodalRemainderCellHorizon:
    """Maximal unchanged visible prefix of a constant supplied nodal input.

    None limits denote an unbounded stationary prefix, not a missing test.
    At the first exit, the exact update can leave the source rounding cell,
    the declared EPI band, or both. A cell exit is not a band-exit theorem.
    No graph, phase producer, event or future pressure constancy is sealed.
    """

    state: NodalRemainderState
    timestep: float
    capacity: tuple[float, ...]
    pressure: tuple[float, ...]
    exact_increment: tuple[Fraction, ...]
    mean_increment: Fraction
    source_cells: tuple[Binary64AdditionCell, ...]
    coordinate_step_limits: tuple[int | None, ...]
    max_unchanged_steps: int | None
    first_exit_step: int | None
    unchanged_endpoint: tuple[Fraction, ...] | None
    first_exit_exact: tuple[Fraction, ...] | None
    first_exit_leaves_cell: tuple[bool, ...]
    first_exit_leaves_band: tuple[bool, ...]


def derive_nodal_remainder_cell_horizon(
    *, state: NodalRemainderState, timestep: float,
    capacity: tuple[float, ...], pressure: tuple[float, ...],
) -> NodalRemainderCellHorizon:
    """Derive an exact integer prefix without iterating a held nodal flow.

    Let a_i=h*nu_i*p_i and X_i=F(x_i)+r_i. The largest admissible unchanged
    prefix consists of integers n>=0 for which X_i+n*a_i belongs both to
    RN's source cell C(x_i) and to the declared closed EPI band, for every
    coordinate. The directional distance divided by abs(a_i) gives each
    bound: floor for a closed endpoint, ceil minus one for an open tie.
    Linearity makes these endpoint tests sufficient for every earlier step.

    If pressure is a deterministic function of visible EPI and fixed other
    inputs, and the supplied p equals that function at x, induction gives
    the same p at every refresh up to the first exit. This conditional
    statement does not establish that a graph's phases, capacity, support,
    callback or event schedule remain fixed. With a nonzero increment the
    visible state cannot remain unchanged indefinitely, even when ordinary
    rounded Euler stalls. Pressure may change at the first cell exit; its
    earlier signed mean source cannot be extrapolated beyond that boundary.
    """
    exact = _validate_nodal_remainder_state(state)
    h = _finite_binary64(timestep, "timestep")
    capacities = _binary64_tuple(capacity, "capacity")
    pressures = _binary64_tuple(pressure, "pressure")
    if len(capacities) != len(exact) or len(pressures) != len(exact):
        raise ValueError("capacity and pressure tuples must match the EPI dimensions")
    if h < 0 or any(value < 0 for value in capacities):
        raise ValueError("timestep and capacity must be nonnegative")
    increment = tuple(Fraction(h) * Fraction(nu) * Fraction(p)
                      for nu, p in zip(capacities, pressures, strict=True))
    lower, upper = Fraction(state.epi_lower), Fraction(state.epi_upper)
    cells = tuple(_rounding_cell(x, value) for x, value in zip(state.epi, exact, strict=True))
    limits = []
    for value, added, cell in zip(exact, increment, cells, strict=True):
        if not added:
            limits.append(None)
            continue
        if added > 0:
            endpoint = min(cell.upper, upper)
            closed = endpoint < cell.upper or cell.even_significand
            ratio = (endpoint - value) / added
        else:
            endpoint = max(cell.lower, lower)
            closed = endpoint > cell.lower or cell.even_significand
            ratio = (value - endpoint) / -added
        # ceil(ratio)-1 == (numerator-1)//denominator for rational ratio.
        limit = (ratio.numerator if closed else ratio.numerator - 1) // ratio.denominator
        if limit < 0:
            raise RuntimeError("validated carried state is outside its initial admissible cell")
        limits.append(limit)
    finite_limits = tuple(value for value in limits if value is not None)
    maximum = min(finite_limits) if finite_limits else None
    first_exit = maximum + 1 if maximum is not None else None
    endpoint = (tuple(value + maximum * added for value, added in zip(exact, increment, strict=True))
                if maximum is not None else None)
    after = (tuple(value + first_exit * added for value, added in zip(exact, increment, strict=True))
             if first_exit is not None else None)
    leaves_cell = tuple(
        not _rounding_cell(x, value).contains_exact_input
        for x, value in zip(state.epi, after, strict=True)
    ) if after is not None else (False,) * len(exact)
    leaves_band = tuple(not lower <= value <= upper for value in after) if after is not None else (False,) * len(exact)
    if after is not None and not any(leaves_cell + leaves_band):
        raise RuntimeError("first carried-cell exit failed its exact boundary check")
    return NodalRemainderCellHorizon(
        state, h, capacities, pressures, increment, _mean(increment), cells,
        tuple(limits), maximum, first_exit, endpoint, after, leaves_cell, leaves_band,
    )


@dataclass(frozen=True, slots=True)
class NodalRemainderPrefix:
    """One supplied-input prefix and its signed final-cell mean bounds."""

    ordinal: int
    cumulative_nodal_area: tuple[Fraction, ...]
    visible_change: tuple[Fraction, ...]
    reconstructed_change: tuple[Fraction, ...]
    carry_transfer: tuple[Fraction, ...]
    identity_residual: tuple[Fraction, ...]
    mean_nodal_area: Fraction
    mean_visible_change: Fraction
    mean_reconstructed_change: Fraction
    mean_carry_transfer: Fraction
    mean_identity_residual: Fraction
    mean_rounding_defect: Fraction
    mean_rounding_lower_bound: Fraction
    mean_rounding_upper_bound: Fraction
    output_cells: tuple[Binary64AdditionCell, ...]


@dataclass(frozen=True, slots=True)
class NodalRemainderSequence:
    """A finite carried sequence, with no graph or pressure-law provenance.

    The uniform bound controls visible change minus the exact accumulated
    supplied nodal area. It does not bound that area or pressure-realization
    error, and is not a solver convergence or full-runtime stability claim.
    """

    initial: NodalRemainderState
    steps: tuple[NodalRemainderStep, ...]
    prefixes: tuple[NodalRemainderPrefix, ...]
    endpoint: NodalRemainderState
    uniform_mean_rounding_bound: Fraction


def observe_nodal_remainder_sequence(
    *, initial: NodalRemainderState, timesteps: tuple[float, ...],
    capacities: tuple[tuple[float, ...], ...],
    pressures: tuple[tuple[float, ...], ...],
) -> NodalRemainderSequence:
    """Execute a nonempty prescribed schedule and verify every exact prefix.

    For a_k=h_k*nu_k*p_k interpreted as exact represented coefficients,
    X_N-X_0=sum(a_k) and x_N-x_0=sum(a_k)+r_0-r_N. The displayed arithmetic
    mean defect therefore lies between mean(r_0) minus the upper/lower
    final-cell remainder limits. Closed bounds remain valid for odd cells
    whose tie endpoints are excluded. Its magnitude is at most
    abs(mean(r_0))+2^-53 inside the kernel's positive unit band, independent
    of the supplied finite length. Zero initial carry and balanced total
    nodal area imply exact reconstructed-mean conservation, with at most
    2^-53 displayed-mean drift. Under heterogeneous capacities, zero mean
    pressure alone does not balance the nodal area.

    These are numerical representation identities for declared inputs.
    Discarding carry between prefixes would introduce an additional jump.
    Neither independently initialized sequences nor changed-state pressure
    refreshes may be spliced into this balance without accounting for it.
    """
    if not isinstance(initial, NodalRemainderState):
        raise TypeError("initial must be a NodalRemainderState")
    exact_initial = initial.exact_epi
    for values, label in ((timesteps, "timesteps"), (capacities, "capacities"),
                          (pressures, "pressures")):
        if type(values) is not tuple:
            raise TypeError(f"{label} must be an ordered tuple")
    if not timesteps or len(capacities) != len(timesteps) or len(pressures) != len(timesteps):
        raise ValueError("the prescribed schedule must have nonempty matching lengths")
    visible_initial = tuple(Fraction.from_float(value) for value in initial.epi)
    initial_mean_carry = _mean(initial.remainder)
    bound = abs(initial_mean_carry) + NODAL_REMAINDER_ABSOLUTE_BOUND
    area = (Fraction(0),) * len(exact_initial)
    current = initial
    steps, prefixes = [], []
    for ordinal, (h, capacity, pressure) in enumerate(zip(timesteps, capacities, pressures, strict=True), 1):
        step = advance_nodal_remainder(current, timestep=h, capacity=capacity, pressure=pressure)
        current = step.after
        exact = current.exact_epi
        visible = tuple(Fraction.from_float(value) for value in current.epi)
        area = tuple(total + added for total, added in zip(area, step.exact_increment, strict=True))
        visible_change = tuple(y - x for x, y in zip(visible_initial, visible, strict=True))
        reconstructed_change = tuple(y - x for x, y in zip(exact_initial, exact, strict=True))
        transfer = tuple(x - y for x, y in zip(initial.remainder, current.remainder, strict=True))
        residual = tuple(change - total - carried for change, total, carried
                         in zip(visible_change, area, transfer, strict=True))
        cells = tuple(_rounding_cell(value, coordinate)
                      for value, coordinate in zip(current.epi, exact, strict=True))
        remainder_lower = tuple(cell.lower - value for cell, value in zip(cells, visible, strict=True))
        remainder_upper = tuple(cell.upper - value for cell, value in zip(cells, visible, strict=True))
        lower = initial_mean_carry - _mean(remainder_upper)
        upper = initial_mean_carry - _mean(remainder_lower)
        mean_area, mean_visible = _mean(area), _mean(visible_change)
        defect = mean_visible - mean_area
        if (any(residual) or reconstructed_change != area
                or not all(cell.contains_exact_input for cell in cells)
                or not lower <= defect <= upper or abs(defect) > bound):
            raise RuntimeError("carried nodal prefix failed its exact area or rounding-cell balance")
        prefixes.append(NodalRemainderPrefix(
            ordinal, area, visible_change, reconstructed_change, transfer, residual,
            mean_area, mean_visible, _mean(reconstructed_change), _mean(transfer),
            _mean(residual), defect, lower, upper, cells,
        ))
        steps.append(step)
    return NodalRemainderSequence(initial, tuple(steps), tuple(prefixes), current, bound)


@dataclass(frozen=True, slots=True)
class NodalRemainderCellExit:
    """One budgeted first-cell exit under constant supplied nodal inputs.

    The sequence preserves the initial carry and is checked against the
    analytic horizon. Its inputs are not authenticated as the output of
    a pressure producer, graph or operator schedule.
    """

    horizon: NodalRemainderCellHorizon
    sequence: NodalRemainderSequence


def observe_nodal_remainder_cell_exit(
    *, state: NodalRemainderState, timestep: float,
    capacity: tuple[float, ...], pressure: tuple[float, ...], step_budget: int,
) -> NodalRemainderCellExit:
    """Replay one analytically bounded cell exit without discarding carry.

    Require a positive integer computational budget. Derive the existing
    exact horizon first, rejecting a stationary prefix, an exit outside
    the declared EPI band or a horizon beyond the budget before allocating
    repeated input rows or executing any numerical step. No clipping or
    partial prefix is substituted when these conditions fail.

    The shared sequence owner then advances the unchanged supplied inputs.
    Every pre-update visible row and every intermediate output must remain
    in the initial cell; the last pre-update and final exact coordinates
    must match the analytic horizon. A caller that wants fresh generated
    pressure must separately bind the source and refresh it at the exit.
    The budget is a work limit, not a physical parameter or future bound.
    """
    if type(step_budget) is not int:
        raise TypeError("step_budget must be a positive integer")
    if step_budget <= 0:
        raise ValueError("step_budget must be positive")
    horizon = derive_nodal_remainder_cell_horizon(
        state=state, timestep=timestep, capacity=capacity, pressure=pressure,
    )
    count = horizon.first_exit_step
    if count is None:
        raise ValueError("the supplied nodal inputs have no finite cell exit")
    if any(horizon.first_exit_leaves_band):
        raise ValueError("the first cell exit leaves the declared EPI band")
    if count > step_budget:
        raise ValueError("the first cell exit exceeds step_budget")
    sequence = observe_nodal_remainder_sequence(
        initial=horizon.state, timesteps=(horizon.timestep,) * count,
        capacities=(horizon.capacity,) * count, pressures=(horizon.pressure,) * count,
    )
    if (len(sequence.steps) != count
            or any(step.before.epi != horizon.state.epi for step in sequence.steps)
            or any(step.after.epi != horizon.state.epi for step in sequence.steps[:-1])
            or sequence.steps[-1].before.exact_epi != horizon.unchanged_endpoint
            or sequence.steps[-1].after.exact_epi != horizon.first_exit_exact
            or sequence.endpoint.exact_epi != horizon.first_exit_exact):
        raise RuntimeError("shared carried replay differs from the analytic first cell exit")
    return NodalRemainderCellExit(horizon, sequence)


@dataclass(frozen=True, slots=True)
class NodalRemainderItineraryCell:
    """An initial-coordinate interval and its admissible dyadic grid indices.

    The exact interval retains nearest-even endpoint openness. The first
    and last indices refer to multiples of 2^-3222, the shared kernel's
    numerical representation limit. An interval can have real interior
    but no admissible grid point; ``feasible`` tests the latter condition.
    Empty intervals and reversed index limits are retained diagnostically.
    """

    lower: Fraction
    upper: Fraction
    lower_closed: bool
    upper_closed: bool
    first_grid_index: int
    last_grid_index: int
    feasible: bool


@dataclass(frozen=True, slots=True)
class NodalRemainderItinerary:
    """Existential feasibility of one supplied visible nodal itinerary.

    The pressure and capacity rows are supplied arithmetic inputs. This
    observer authenticates neither their production nor the reachability
    of its witness. Adjacent feasible transitions need not share one carry.
    A closed visible itinerary is a true conditional carried cycle only
    when its total nodal area vanishes in every coordinate, not merely in
    arithmetic mean. Repetition then also requires the same supplied input
    schedule; graph events and pressure refresh rules are not certified.
    """

    epi_states: tuple[tuple[float, ...], ...]
    timesteps: tuple[float, ...]
    capacities: tuple[tuple[float, ...], ...]
    pressures: tuple[tuple[float, ...], ...]
    epi_lower: float
    epi_upper: float
    coordinates: tuple[NodalRemainderItineraryCell, ...]
    cumulative_nodal_area: tuple[tuple[Fraction, ...], ...]
    total_nodal_area: tuple[Fraction, ...]
    feasible: bool
    visible_closed: bool
    conditional_carried_cycle: bool
    zero_initial_carry_feasible: bool
    witness_initial: NodalRemainderState | None
    witness_sequence: NodalRemainderSequence | None

    @property
    def pressure_provenance_certified(self) -> bool:
        return False

    @property
    def runtime_provenance_certified(self) -> bool:
        return False


def _itinerary_rows(rows, label):
    if type(rows) is not tuple:
        raise TypeError(f"{label} must be an ordered tuple of binary64 rows")
    return tuple(_binary64_tuple(row, f"{label}[{index}]") for index, row in enumerate(rows))


def derive_nodal_remainder_itinerary(
    *, epi_states: tuple[tuple[float, ...], ...], timesteps: tuple[float, ...],
    capacities: tuple[tuple[float, ...], ...], pressures: tuple[tuple[float, ...], ...],
    epi_lower: float = .05, epi_upper: float = 1.0,
) -> NodalRemainderItinerary:
    """Intersect exact translated output cells for a nonempty itinerary.

    For N declared steps let A_0=0 and
    A_(k+1)=A_k+Fraction(h_k)*Fraction(nu_k)*Fraction(p_k), coordinatewise.
    An initial reconstructed value X_0 produces all supplied visible rows
    exactly when X_0 lies in every translated interval
    (C(x_k) intersect [epi_lower,epi_upper])-A_k. At coincident boundaries
    all contributing inclusion flags must hold. The exact cells come from
    the shared nearest-even owner; no rounded accumulation is substituted.

    Every legal carry has denominator dividing 2^3222, and every displayed
    float lies on that grid. The intersection is therefore tested on this
    exact grid, including strict ties, rather than only as a real interval.
    Every declared three-factor area preserves the grid. This proves both
    necessity and sufficiency for the shared numerical representation.

    When feasible, choose each initial grid point nearest the displayed
    initial value by clamping its integer index to the admissible limits.
    This witness is a detached existence certificate, not a new graph
    preparation. Replay it through the shared sequence owner and require
    every visible row to match. If the word closes visibly, the exact
    carry also closes if and only if total_nodal_area is the zero vector.
    No pressure-generation law, physical event or runtime word is inferred.
    """
    _require_remainder_rounding()
    lower, upper = _remainder_band(epi_lower, epi_upper)
    states = _itinerary_rows(epi_states, "epi_states")
    steps = _binary64_tuple(timesteps, "timesteps")
    capacity_rows = _itinerary_rows(capacities, "capacities")
    pressure_rows = _itinerary_rows(pressures, "pressures")
    count = len(steps)
    if len(states) != count + 1 or len(capacity_rows) != count or len(pressure_rows) != count:
        raise ValueError("the itinerary requires N steps and N+1 visible EPI rows")
    dimension = len(states[0])
    if any(len(row) != dimension for row in states + capacity_rows + pressure_rows):
        raise ValueError("every itinerary row must have the same nodal dimension")
    if any(h < 0 for h in steps) or any(nu < 0 for row in capacity_rows for nu in row):
        raise ValueError("timestep and capacity must be nonnegative")
    if any(not lower <= value <= upper for row in states for value in row):
        raise ValueError("every visible EPI must lie in the declared positive band")
    zero = (Fraction(0),) * dimension
    area = [zero]
    for h, capacities_at_step, pressures_at_step in zip(steps, capacity_rows, pressure_rows, strict=True):
        area.append(tuple(
            total + Fraction(h) * Fraction(nu) * Fraction(p)
            for total, nu, p in zip(area[-1], capacities_at_step, pressures_at_step, strict=True)
        ))
    exact_lower, exact_upper = Fraction(lower), Fraction(upper)
    scale = 2**NODAL_REMAINDER_DENOMINATOR_BITS
    coordinate_cells = []
    for coordinate in range(dimension):
        interval_lower = interval_upper = None
        lower_closed = upper_closed = True
        for row, accumulated in zip(states, area, strict=True):
            visible = row[coordinate]
            cell = _rounding_cell(visible, Fraction(visible))
            lo = max(cell.lower, exact_lower)
            hi = min(cell.upper, exact_upper)
            lo_closed = lo > cell.lower or cell.even_significand
            hi_closed = hi < cell.upper or cell.even_significand
            lo -= accumulated[coordinate]
            hi -= accumulated[coordinate]
            if interval_lower is None or lo > interval_lower:
                interval_lower, lower_closed = lo, lo_closed
            elif lo == interval_lower:
                lower_closed = lower_closed and lo_closed
            if interval_upper is None or hi < interval_upper:
                interval_upper, upper_closed = hi, hi_closed
            elif hi == interval_upper:
                upper_closed = upper_closed and hi_closed
        scaled_lower, scaled_upper = interval_lower * scale, interval_upper * scale
        first = -((-scaled_lower.numerator) // scaled_lower.denominator)
        last = scaled_upper.numerator // scaled_upper.denominator
        if not lower_closed and scaled_lower.denominator == 1:
            first += 1
        if not upper_closed and scaled_upper.denominator == 1:
            last -= 1
        coordinate_cells.append(NodalRemainderItineraryCell(
            interval_lower, interval_upper, lower_closed, upper_closed, first, last, first <= last,
        ))
    coordinates = tuple(coordinate_cells)
    feasible = all(cell.feasible for cell in coordinates)
    initial_indices = tuple(int(Fraction(value) * scale) for value in states[0])
    zero_carry_feasible = all(
        cell.first_grid_index <= index <= cell.last_grid_index
        for cell, index in zip(coordinates, initial_indices, strict=True)
    )
    witness = sequence = None
    if feasible:
        initial_exact = tuple(
            Fraction(min(max(index, cell.first_grid_index), cell.last_grid_index), scale)
            for cell, index in zip(coordinates, initial_indices, strict=True)
        )
        witness = NodalRemainderState(
            states[0], tuple(value - Fraction(visible)
                             for value, visible in zip(initial_exact, states[0], strict=True)),
            lower, upper,
        )
        _validate_nodal_remainder_state(witness)
        sequence = observe_nodal_remainder_sequence(
            initial=witness, timesteps=steps, capacities=capacity_rows, pressures=pressure_rows,
        )
        if tuple(step.after.epi for step in sequence.steps) != states[1:]:
            raise RuntimeError("the feasible itinerary witness failed shared-kernel replay")
    visible_closed = states[0] == states[-1]
    carried_cycle = feasible and visible_closed and not any(area[-1])
    return NodalRemainderItinerary(
        states, steps, capacity_rows, pressure_rows, lower, upper, coordinates,
        tuple(area), area[-1], feasible, visible_closed, carried_cycle,
        zero_carry_feasible, witness, sequence,
    )
