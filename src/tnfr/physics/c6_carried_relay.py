"""Exact local relay strips and finite signed budgets for carried C6.

The relay coefficients come from four refreshed canonical pressure rows.
Their rotations describe the existing nodal map; they add no oscillator,
pressure correction or physical parameter. The two-relay certificate holds
the other four displayed nodes fixed. The locality extension holds only the
required pressure neighborhood and permits the remaining nodes to vary.
Signed corrected coordinates bound each domain without a long period search.
"""

from dataclasses import dataclass
from fractions import Fraction as F
import math

from ..dynamics._euler_kernel import (
    NODAL_REMAINDER_DENOMINATOR_BITS, NodalRemainderState, NodalRemainderStep,
    _validate_nodal_remainder_state, advance_nodal_remainder,
)
from ._cycle_algebra import Vector
from .binary64_nodal_flow import _rounding_cell
from .c6_carried_balance import _closure
from .c6_carried_closure import C6CarriedClosure, derive_c6_carried_closure
from .c6_carried_tube import C6CarriedBandHorizon, derive_c6_carried_band_horizon
from .c6_pressure_lattice import _observe_rebuilt_c6_pressure_lattice

__all__ = [
    "C6CarriedRelayAxis", "C6CarriedRelay", "derive_c6_carried_relay",
    "C6CarriedRelayPoint", "C6CarriedRelayExit", "observe_c6_carried_relay_exit",
    "C6CarriedLocalRelayBudget", "derive_c6_carried_local_relay_budget",
    "C6CarriedLocalRelayPoint", "C6CarriedLocalRelayExit", "observe_c6_carried_local_relay_exit",
]


@dataclass(frozen=True, slots=True)
class C6CarriedRelayAxis:
    """One exact two-cell rotation, including its nearest-even endpoint rule.

    For u=X_node-facet the upper cell subtracts decrement=a and the lower
    cell adds increment=b. The invariant strip is [-a,b) when the upper
    cell owns the tie, and (-a,b] when the lower cell owns it.
    """

    node: int
    upper_visible: float
    lower_visible: float
    facet: F
    decrement: F
    increment: F
    length: F
    initial_offset: F
    lower_closed: bool
    upper_closed: bool
    correction: Vector


@dataclass(frozen=True, slots=True)
class C6CarriedRelay:
    """Conditional product-strip inclusion and a finite signed exit bound.

    The two selected coordinates stay in their strips as long as the four
    other displayed coordinates retain the source values. The full six-node
    region is not invariant: a nonzero compensated drift forces another cell
    exit by deadline, or earlier band failure. Initial membership refers to
    the supplied carried state, without changing any remainder.
    """

    closure: C6CarriedClosure
    relay_nodes: tuple[int, int]
    visible_rows: tuple[tuple[float, ...], ...]
    pressure_rows: tuple[tuple[float, ...], ...]
    axes: tuple[C6CarriedRelayAxis, ...]
    base_increment: Vector
    drift: Vector
    correction_lower: Vector
    correction_upper: Vector
    nonrelay_nodes: tuple[int, ...]
    exit_deadlines: tuple[int | None, ...]
    deadline: int
    deadline_nodes: tuple[int, ...]

    @property
    def initial_membership_verified(self) -> bool:
        return True

    @property
    def conditional_product_relay_inclusion(self) -> bool:
        return True

    @property
    def full_region_invariant(self) -> bool:
        return False

    @property
    def future_runtime_certified(self) -> bool:
        return False


def _inside_strip(axis, value):
    return ((value > -axis.decrement or axis.lower_closed and value == -axis.decrement)
            and (value < axis.increment or axis.upper_closed and value == axis.increment))


def derive_c6_carried_relay(
    closure: C6CarriedClosure, *, relay_nodes: tuple[int, int] = (0, 3),
) -> C6CarriedRelay:
    """Bind four pressure cells and derive their exact affine rotation budget.

    Each selected source value is the upper member of an adjacent pair.
    Opposite C6 nodes have disjoint pressure-response supports. Recompute all
    four rows and require their exact additive decomposition, positive inward
    increments, correct nearest-even ties and initial strip membership.

    Writing d_s=h*(p_single_s-p_base), L_s=a_s+b_s and k_s=d_s/L_s gives
    v=h*p_base+sum_s(k_s*a_s). Until another displayed coordinate changes,
    X(n)=X(0)+n*v+sum_s k_s*(u_s(n)-u_s(0)). The two relay entries of v vanish.
    Bounds on each u_s supply exact rational lower/upper corrections. For
    each other coordinate with nonzero v, choose the first integer for which
    even the favorable correction is strictly beyond its current RN cell.
    The minimum is a proof-derived deadline, not a fitted simulation horizon.
    """
    if (type(relay_nodes) is not tuple or len(relay_nodes) != 2
            or any(type(node) is not int or not 0 <= node < 6 for node in relay_nodes)
            or (relay_nodes[1] - relay_nodes[0]) % 6 != 3):
        raise ValueError("relay_nodes must be an exact tuple of two opposite ordered C6 nodes")
    bound = _closure(closure)
    state = bound.base_tube.state
    exact = _validate_nodal_remainder_state(state)
    reference = bound.base_tube.contraction.profile.lattice
    h = F(bound.base_tube.contraction.timestep)
    if h <= 0:
        raise ValueError("the relay requires a strictly positive nodal timestep")
    lower_values = tuple(math.nextafter(state.epi[node], -math.inf) for node in relay_nodes)
    rows = tuple(tuple(
        lower_values[relay_nodes.index(node)]
        if node in relay_nodes and mask & (1 << relay_nodes.index(node)) else value
        for node, value in enumerate(state.epi)
    ) for mask in range(4))
    observations = tuple(_observe_rebuilt_c6_pressure_lattice(reference, row) for row in rows)
    pressures = tuple(observation.pressure for observation in observations)
    increments = tuple(tuple(h * F(value) for value in row) for row in pressures)
    base = increments[0]
    differences = tuple(tuple(value - initial for value, initial in zip(
        increments[1 << index], base, strict=True,
    )) for index in range(2))
    if any(increments[mask][node] != base[node] + sum(
        (differences[index][node] for index in range(2) if mask & (1 << index)), F(0),
    ) for mask in range(4) for node in range(6)):
        raise ValueError("the four canonical pressure rows do not decompose into independent relays")
    scale = 2**NODAL_REMAINDER_DENOMINATOR_BITS
    if any((value * scale).denominator != 1 for row in increments for value in row):
        raise RuntimeError("a canonical relay increment left the shared dyadic encoding grid")
    axes = []
    for index, node in enumerate(relay_nodes):
        a, b = -base[node], increments[1 << index][node]
        if a <= 0 or b <= 0:
            raise ValueError("each adjacent cell pair must have strictly inward relay pressures")
        length = a + b
        if differences[1 - index][node] != 0 or differences[index][node] != length:
            raise ValueError("a selected relay coordinate depends on the opposite switch")
        upper, lower = state.epi[node], lower_values[index]
        facet = (F(upper) + F(lower)) / 2
        upper_cell, lower_cell = _rounding_cell(upper, facet), _rounding_cell(lower, facet)
        if (upper_cell.lower != facet or lower_cell.upper != facet
                or upper_cell.even_significand == lower_cell.even_significand):
            raise ValueError("the adjacent relay cells require complementary nearest-even tie ownership")
        # Strict containment at the two outer boundaries avoids an additional
        # third-cell endpoint case; the central facet retains its exact tie.
        if not lower_cell.lower < facet - a < facet + b < upper_cell.upper:
            raise ValueError("the relay strip must fit strictly inside the two adjacent RN cells")
        if not F(state.epi_lower) <= facet - a < facet + b <= F(state.epi_upper):
            raise ValueError("the complete relay strip must remain inside the declared band")
        axis = C6CarriedRelayAxis(
            node, upper, lower, facet, a, b, length, exact[node] - facet,
            upper_cell.even_significand, lower_cell.even_significand,
            tuple(value / length for value in differences[index]),
        )
        if not _inside_strip(axis, axis.initial_offset):
            raise ValueError("the actual incoming carry lies outside a derived relay strip")
        axes.append(axis)
    axes = tuple(axes)
    drift = tuple(base[node] + sum((axis.correction[node] * axis.decrement for axis in axes), F(0))
                  for node in range(6))
    if any(drift[node] != 0 for node in relay_nodes):
        raise RuntimeError("the relay decomposition retained a spurious drift in a rotation coordinate")
    correction_lower = tuple(sum((min(
        axis.correction[node] * (-axis.decrement - axis.initial_offset),
        axis.correction[node] * (axis.increment - axis.initial_offset),
    ) for axis in axes), F(0)) for node in range(6))
    correction_upper = tuple(sum((max(
        axis.correction[node] * (-axis.decrement - axis.initial_offset),
        axis.correction[node] * (axis.increment - axis.initial_offset),
    ) for axis in axes), F(0)) for node in range(6))
    nonrelay = tuple(node for node in range(6) if node not in relay_nodes)
    deadlines = [None] * 6
    for node in nonrelay:
        cell = _rounding_cell(state.epi[node], exact[node])
        if drift[node] > 0:
            ratio = (cell.upper - exact[node] - correction_lower[node]) / drift[node]
        elif drift[node] < 0:
            ratio = (exact[node] + correction_upper[node] - cell.lower) / -drift[node]
        else:
            continue
        deadlines[node] = ratio.__floor__() + 1
        if deadlines[node] <= 0:
            raise RuntimeError("an admitted source lost its positive signed exit deadline")
    finite = tuple(value for value in deadlines if value is not None)
    if not finite:
        raise ValueError("the local relay family has no nonzero non-relay drift for a finite exit proof")
    deadline = min(finite)
    return C6CarriedRelay(
        bound, relay_nodes, rows, pressures, axes, base, drift, correction_lower, correction_upper,
        nonrelay, tuple(deadlines), deadline,
        tuple(node for node, value in enumerate(deadlines) if value == deadline),
    )


@dataclass(frozen=True, slots=True)
class C6CarriedRelayPoint:
    """One formula value, with exact switch counts and the vector identity."""

    ordinal: int
    relay_offsets: Vector
    lower_visits: tuple[int, ...]
    exact_epi: Vector
    drift_identity_residual: Vector


def _relay_offset_and_count(axis, ordinal):
    """Use the shared tie-aware rotation for a product or local relay."""
    if axis.lower_closed:
        offset = -axis.decrement + ((axis.initial_offset + axis.decrement
                                     - ordinal * axis.decrement) % axis.length)
    else:
        offset = axis.increment - ((axis.increment - axis.initial_offset
                                    + ordinal * axis.decrement) % axis.length)
    count = (offset - axis.initial_offset + ordinal * axis.decrement) / axis.length
    if not _inside_strip(axis, offset) or count.denominator != 1 or not 0 <= count <= ordinal:
        raise RuntimeError("the exact relay rotation lost its legal strip or integer switch count")
    return offset, count.numerator


def _formula(relay, ordinal):
    rotations = tuple(_relay_offset_and_count(axis, ordinal) for axis in relay.axes)
    offsets = tuple(value[0] for value in rotations)
    visits = tuple(value[1] for value in rotations)
    initial = _validate_nodal_remainder_state(relay.closure.base_tube.state)
    exact = tuple(initial[node] + ordinal * relay.base_increment[node] + sum((
        count * axis.length * axis.correction[node]
        for count, axis in zip(visits, relay.axes, strict=True)
    ), F(0)) for node in range(6))
    decomposed = tuple(initial[node] + ordinal * relay.drift[node] + sum((
        axis.correction[node] * (offset - axis.initial_offset)
        for offset, axis in zip(offsets, relay.axes, strict=True)
    ), F(0)) for node in range(6))
    residual = tuple(value - other for value, other in zip(exact, decomposed, strict=True))
    if any(residual) or any(exact[axis.node] - axis.facet != offset
                            for axis, offset in zip(relay.axes, offsets, strict=True)):
        raise RuntimeError("the relay switch-count formula lost its full-vector drift identity")
    return C6CarriedRelayPoint(ordinal, tuple(offsets), tuple(visits), exact, residual)


@dataclass(frozen=True, slots=True)
class C6CarriedRelayExit:
    """First conditional exit of another visible coordinate, within the theorem bound.

    All admitted transitions use the shared carried nodal kernel. A candidate
    outside the physical band is retained without constructing an invalid
    state. This is a conditional numerical continuation, not a live graph word.
    """

    relay: C6CarriedRelay
    points: tuple[C6CarriedRelayPoint, ...]
    steps: tuple[NodalRemainderStep, ...]
    exit_step: int
    exiting_nodes: tuple[int, ...]
    endpoint: NodalRemainderState | None
    exact_endpoint: Vector
    total_nodal_area: Vector
    mean_area: F
    nodal_balance_residual: Vector
    band_failure: bool

    @property
    def future_runtime_certified(self) -> bool:
        return False

    @property
    def indefinite_trapping_certified(self) -> bool:
        return False


def observe_c6_carried_relay_exit(
    relay: C6CarriedRelay, *, step_budget: int = 1024,
) -> C6CarriedRelayExit:
    """Rebuild the theorem and verify only its bounded first non-relay exit.

    The exact modular formula determines the state at each inspected integer;
    its pressure word is checked against the four rederived canonical rows and
    the shared integrator. The work budget must cover the analytic deadline.
    No cached drift, endpoint, pressure or deadline is trusted as input.
    """
    if type(relay) is not C6CarriedRelay:
        raise TypeError("relay must be a C6CarriedRelay")
    if type(step_budget) is not int or step_budget <= 0:
        raise ValueError("step_budget must be a positive exact integer")
    bound = derive_c6_carried_relay(relay.closure, relay_nodes=relay.relay_nodes)
    if bound.deadline > step_budget:
        raise ValueError("the derived relay exit deadline exceeds step_budget")
    state = bound.closure.base_tube.state
    initial = _validate_nodal_remainder_state(state)
    h = bound.closure.base_tube.contraction.timestep
    points, steps, area = [_formula(bound, 0)], [], (F(0),) * 6
    for ordinal in range(1, bound.deadline + 1):
        try:
            mask = bound.visible_rows.index(state.epi)
        except ValueError as exc:
            raise RuntimeError("the pre-exit source is outside the four certified relay cells") from exc
        pressure = bound.pressure_rows[mask]
        added = tuple(F(h) * F(value) for value in pressure)
        candidate = tuple(value + increment for value, increment in zip(state.exact_epi, added, strict=True))
        point = _formula(bound, ordinal)
        if point.exact_epi != candidate:
            raise RuntimeError("the next canonical nodal area differs from the exact relay formula")
        points.append(point)
        area = tuple(value + increment for value, increment in zip(area, added, strict=True))
        failure = any(not F(state.epi_lower) <= value <= F(state.epi_upper) for value in candidate)
        endpoint = None
        if not failure:
            step = advance_nodal_remainder(state, timestep=h, capacity=(1.,) * 6, pressure=pressure)
            if (step.after.exact_epi != candidate or step.exact_increment != added
                    or any(step.nodal_balance_residual)):
                raise RuntimeError("the shared carried relay step lost its exact nodal area")
            steps.append(step)
            endpoint = step.after
        leaving = tuple(node for node in bound.nonrelay_nodes if not _rounding_cell(
            bound.visible_rows[0][node], candidate[node],
        ).contains_exact_input)
        if leaving or failure:
            residual = tuple(value - start - increment for value, start, increment in zip(
                candidate, initial, area, strict=True,
            ))
            if any(residual):
                raise RuntimeError("the finite relay passage lost its full-vector nodal telescope")
            return C6CarriedRelayExit(
                bound, tuple(points), tuple(steps), ordinal, leaving, endpoint,
                candidate, area, sum(area, F(0)) / 6, residual, failure,
            )
        state = endpoint
    raise RuntimeError("the exact relay passage contradicted its signed exit deadline")


@dataclass(frozen=True, slots=True)
class C6CarriedLocalRelayBudget:
    """A local corrected-coordinate budget that survives other cell changes.

    Only the selected relay and adjacent budget pressure rows are fixed by
    the retained displayed neighborhood. Other rows remain fully refreshed.
    The supplied state is checked for membership; its temporal relationship
    to the anchor is not inferred. The separate band horizon can close the
    numerical-band premise through the signed exit deadline.
    """

    anchor: C6CarriedRelay
    closure: C6CarriedClosure
    band_horizon: C6CarriedBandHorizon
    relay_node: int
    budget_node: int
    held_nodes: tuple[int, ...]
    free_nodes: tuple[int, ...]
    dependency_nodes: tuple[tuple[int, ...], ...]
    local_pressure_rows: tuple[tuple[float, ...], ...]
    upper_visible: float
    lower_visible: float
    facet: F
    decrement: F
    increment: F
    length: F
    initial_offset: F
    lower_closed: bool
    upper_closed: bool
    budget_correction: F
    budget_drift: F
    corrected_initial: F
    correction_lower: F
    correction_upper: F
    deadline: int
    band_covers_deadline: bool

    @property
    def initial_membership_verified(self) -> bool:
        return True

    @property
    def incoming_state_lineage_certified(self) -> bool:
        return False

    @property
    def free_coordinates_restricted_to_anchor_cells(self) -> bool:
        return False

    @property
    def future_runtime_certified(self) -> bool:
        return False


def derive_c6_carried_local_relay_budget(
    relay: C6CarriedRelay, *, state: NodalRemainderState,
    relay_node: int = 0, budget_node: int = 1,
) -> C6CarriedLocalRelayBudget:
    """Lift one anchor relay by the exact locality of the canonical C6 rows.

    For a fixed phase source, row i depends only on the displayed triple
    (i-1,i,i+1): its exact integer gradient feeds the existing product and
    assembly rounding. Holding the union of the two selected triples,
    except the relay coordinate, therefore fixes both pressure functions
    for every admitted value of the other coordinates. This is a support
    identity, not an extrapolation from a finite sample of free states.

    Let k=(h*p_budget_lower-h*p_budget_upper)/(a+b). The coordinate
    W=X_budget-k*X_relay has the constant exact increment
    v=h*p_budget_upper+k*a, even while the free displayed nodes change.
    The relay strip bounds its oscillatory correction; a strictly signed
    v gives a finite deadline for a held-node change or earlier band loss.
    The existing band-horizon owner checks the latter premise separately.
    """
    if type(relay) is not C6CarriedRelay:
        raise TypeError("relay must be a C6CarriedRelay")
    if type(state) is not NodalRemainderState:
        raise TypeError("state must be an exact NodalRemainderState")
    if (type(relay_node) is not int or type(budget_node) is not int
            or not 0 <= relay_node < 6 or not 0 <= budget_node < 6
            or (budget_node - relay_node) % 6 not in (1, 5)):
        raise ValueError("relay_node and budget_node must be exact adjacent C6 indices")
    anchor = derive_c6_carried_relay(relay.closure, relay_nodes=relay.relay_nodes)
    if relay_node not in anchor.relay_nodes:
        raise ValueError("relay_node must select a certified anchor axis")
    axis_index = anchor.relay_nodes.index(relay_node)
    axis = anchor.axes[axis_index]
    source = anchor.closure.base_tube.state
    exact = _validate_nodal_remainder_state(state)
    if (state.epi_lower, state.epi_upper) != (source.epi_lower, source.epi_upper):
        raise ValueError("the incoming state must retain the exact anchor band")
    dependencies = tuple(tuple(sorted(((node - 1) % 6, node, (node + 1) % 6)))
                         for node in (relay_node, budget_node))
    held = tuple(sorted((set(dependencies[0]) | set(dependencies[1])) - {relay_node}))
    free = tuple(node for node in range(6) if node not in held and node != relay_node)
    if any(state.epi[node] != source.epi[node] for node in held):
        raise ValueError("the incoming displayed neighborhood differs from the held anchor values")
    if state.epi[relay_node] not in (axis.upper_visible, axis.lower_visible):
        raise ValueError("the incoming relay value must belong to the anchor adjacent pair")
    offset = exact[relay_node] - axis.facet
    if not _inside_strip(axis, offset):
        raise ValueError("the incoming relay carry lies outside the anchor invariant strip")
    reference = anchor.closure.base_tube.contraction.profile.lattice
    upper_row = tuple(axis.upper_visible if node == relay_node else value
                      for node, value in enumerate(state.epi))
    lower_row = tuple(axis.lower_visible if node == relay_node else value
                      for node, value in enumerate(state.epi))
    observations = tuple(_observe_rebuilt_c6_pressure_lattice(reference, row)
                         for row in (upper_row, lower_row))
    pressures = tuple(tuple(observation.pressure[node] for node in (relay_node, budget_node))
                      for observation in observations)
    anchor_pressures = tuple(tuple(anchor.pressure_rows[mask][node] for node in (relay_node, budget_node))
                             for mask in (0, 1 << axis_index))
    if pressures != anchor_pressures:
        raise RuntimeError("the canonical local pressure rows lost their displayed-support identity")
    h = F(anchor.closure.base_tube.contraction.timestep)
    correction = h * (F(pressures[1][1]) - F(pressures[0][1])) / axis.length
    drift = h * F(pressures[0][1]) + correction * axis.decrement
    if drift == 0:
        raise ValueError("the local corrected budget needs a nonzero signed drift")
    lower_correction = min(correction * (-axis.decrement - offset), correction * (axis.increment - offset))
    upper_correction = max(correction * (-axis.decrement - offset), correction * (axis.increment - offset))
    cell = _rounding_cell(state.epi[budget_node], exact[budget_node])
    if drift > 0:
        ratio = (cell.upper - exact[budget_node] - lower_correction) / drift
    else:
        ratio = (exact[budget_node] + upper_correction - cell.lower) / -drift
    deadline = ratio.__floor__() + 1
    if deadline <= 0:
        raise RuntimeError("the admitted local relay lost its positive signed exit deadline")
    closure = derive_c6_carried_closure(
        anchor.closure.base_tube.contraction.profile, state=state,
        timestep=anchor.closure.base_tube.contraction.timestep,
    )
    horizon = derive_c6_carried_band_horizon(closure.base_tube)
    covered = horizon.tube_initially_admitted and (
        horizon.unbounded_conditional_prefix
        or horizon.maximum_steps is not None and horizon.maximum_steps >= deadline
    )
    return C6CarriedLocalRelayBudget(
        anchor, closure, horizon, relay_node, budget_node, held, free, dependencies,
        pressures, axis.upper_visible, axis.lower_visible, axis.facet,
        axis.decrement, axis.increment, axis.length, offset, axis.lower_closed, axis.upper_closed,
        correction, drift, exact[budget_node] - correction * exact[relay_node],
        lower_correction, upper_correction, deadline, covered,
    )


@dataclass(frozen=True, slots=True)
class C6CarriedLocalRelayPoint:
    """One full state with two locally predicted coordinates and an exact budget."""

    ordinal: int
    relay_offset: F
    lower_visits: int
    exact_epi: Vector
    expected_relay: F
    expected_budget: F
    corrected_budget: F
    corrected_budget_residual: F


def _local_point(budget, ordinal, exact):
    offset, visits = _relay_offset_and_count(budget, ordinal)
    initial = _validate_nodal_remainder_state(budget.closure.base_tube.state)
    expected_relay = budget.facet + offset
    expected_budget = (initial[budget.budget_node] + ordinal * budget.budget_drift
                       + budget.budget_correction * (offset - budget.initial_offset))
    corrected = exact[budget.budget_node] - budget.budget_correction * exact[budget.relay_node]
    residual = corrected - budget.corrected_initial - ordinal * budget.budget_drift
    if (exact[budget.relay_node] != expected_relay or exact[budget.budget_node] != expected_budget
            or residual != 0):
        raise RuntimeError("the refreshed nodal state differs from the exact local relay budget")
    return C6CarriedLocalRelayPoint(
        ordinal, offset, visits, exact, expected_relay, expected_budget, corrected, residual,
    )


@dataclass(frozen=True, slots=True)
class C6CarriedLocalRelayExit:
    """First held-neighborhood exit under fully refreshed six-node pressure.

    The free coordinates can switch cells throughout the passage. The
    formula is certified only through the step whose source still has the
    held neighborhood, including its outgoing endpoint. Neither source
    lineage nor later graph-owned execution is supplied by this observer.
    """

    budget: C6CarriedLocalRelayBudget
    points: tuple[C6CarriedLocalRelayPoint, ...]
    steps: tuple[NodalRemainderStep, ...]
    exit_step: int
    exiting_nodes: tuple[int, ...]
    endpoint: NodalRemainderState | None
    exact_endpoint: Vector
    total_nodal_area: Vector
    mean_area: F
    nodal_balance_residual: Vector
    band_failure: bool

    @property
    def incoming_state_lineage_certified(self) -> bool:
        return False

    @property
    def future_runtime_certified(self) -> bool:
        return False

    @property
    def indefinite_trapping_certified(self) -> bool:
        return False


def observe_c6_carried_local_relay_exit(
    budget: C6CarriedLocalRelayBudget, *, step_budget: int = 1024,
) -> C6CarriedLocalRelayExit:
    """Rebuild local primitives and stop at the first proved neighborhood gate.

    Every step refreshes the complete canonical pressure from the actual
    displayed tuple. Only the two locally fixed rows are compared with
    the certificate. The remaining rows are neither frozen nor projected.
    All six exact nodal areas telescope, with preserved incoming carry.
    """
    if type(budget) is not C6CarriedLocalRelayBudget:
        raise TypeError("budget must be a C6CarriedLocalRelayBudget")
    if type(step_budget) is not int or step_budget <= 0:
        raise ValueError("step_budget must be a positive exact integer")
    bound = derive_c6_carried_local_relay_budget(
        budget.anchor, state=budget.closure.base_tube.state,
        relay_node=budget.relay_node, budget_node=budget.budget_node,
    )
    if bound.deadline > step_budget:
        raise ValueError("the local relay exit deadline exceeds step_budget")
    state = bound.closure.base_tube.state
    initial = _validate_nodal_remainder_state(state)
    reference = bound.closure.base_tube.contraction.profile.lattice
    h = bound.closure.base_tube.contraction.timestep
    points, steps, area = [_local_point(bound, 0, initial)], [], (F(0),) * 6
    for ordinal in range(1, bound.deadline + 1):
        if any(state.epi[node] != bound.closure.base_tube.state.epi[node] for node in bound.held_nodes):
            raise RuntimeError("a local relay step was attempted after its held-neighborhood exit")
        value = state.epi[bound.relay_node]
        if value not in (bound.upper_visible, bound.lower_visible):
            raise RuntimeError("the local relay left its proved adjacent pair before the neighborhood gate")
        branch = int(value == bound.lower_visible)
        observation = _observe_rebuilt_c6_pressure_lattice(reference, state.epi)
        pressure = observation.pressure
        if tuple(pressure[node] for node in (bound.relay_node, bound.budget_node)) != bound.local_pressure_rows[branch]:
            raise RuntimeError("a freshly generated pressure differs from the exact local row identity")
        added = tuple(F(h) * F(value) for value in pressure)
        current = _validate_nodal_remainder_state(state)
        candidate = tuple(value + increment for value, increment in zip(current, added, strict=True))
        points.append(_local_point(bound, ordinal, candidate))
        area = tuple(value + increment for value, increment in zip(area, added, strict=True))
        failure = any(not F(state.epi_lower) <= value <= F(state.epi_upper) for value in candidate)
        if failure and bound.band_covers_deadline:
            raise RuntimeError("the local passage contradicted its independent numerical-band certificate")
        endpoint = None
        if not failure:
            step = advance_nodal_remainder(state, timestep=h, capacity=(1.,) * 6, pressure=pressure)
            if (step.after.exact_epi != candidate or step.exact_increment != added
                    or any(step.nodal_balance_residual)):
                raise RuntimeError("the shared local relay step lost its exact nodal area")
            steps.append(step)
            endpoint = step.after
        leaving = tuple(node for node in bound.held_nodes if not _rounding_cell(
            bound.closure.base_tube.state.epi[node], candidate[node],
        ).contains_exact_input)
        if leaving or failure:
            residual = tuple(value - start - increment for value, start, increment in zip(
                candidate, initial, area, strict=True,
            ))
            if any(residual):
                raise RuntimeError("the local relay passage lost its full-vector nodal telescope")
            return C6CarriedLocalRelayExit(
                bound, tuple(points), tuple(steps), ordinal, leaving, endpoint,
                candidate, area, sum(area, F(0)) / 6, residual, failure,
            )
        state = endpoint
    raise RuntimeError("the local relay passage contradicted its signed exit deadline")
