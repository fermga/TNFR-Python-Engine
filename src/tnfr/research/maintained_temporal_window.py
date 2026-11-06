"""Primitive-only state-box and true-history bounds for the positive sine law.

These detached bounds hold capacity, support and conductance fixed in the
declared structural clock. They rebuild the complete rows and global bounds;
no cached report, trajectory, observed history or event supplies a premise.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from ..mathematics._rational_interval import I, cos, pi_interval, sin
from .maintained_parent_um_work import StateBounds, _bound_state
from .maintained_successor_chamber import _admit_edges
from .maintained_successor_motion import MotionBounds, _bound_motion, _horizon


@dataclass(frozen=True)
class NodeHistoryWindow:
    """Whole-window bounds, without numerical sample or live admission claims.

    Edge differences retain their supplied principal branches and canonical
    left-to-right orientation. Node pressure and acceleration instead sum
    neighbor-minus-target differences. A positive principal margin certifies
    that edge throughout the window; a nonpositive one only abstains.
    ``third_form_derivative_bound`` is a global magnitude bound for this law,
    not a measured derivative or a numerical sample-error allowance.
    """

    motion: MotionBounds
    target_index: int
    horizon: Q
    edge_phase_differences: tuple[I, ...]
    edge_phase_velocities: tuple[I, ...]
    pressure: I
    form_rate: I
    form_acceleration: I
    form: I
    principal_margins: tuple[Q, ...]
    retained_edge_indices: tuple[int, ...]
    third_form_derivative_bound: Q


@dataclass(frozen=True)
class NodeHistoryBox:
    """Conditional constitutive ranges on a supplied complete state box.

    These fields certify no trajectory or time window. A consumer must
    independently establish that its whole-time state enclosure belongs to
    this box under the same held law. Rectangular ranges discard coordinate
    correlations; their endpoints need not occur jointly or along a solution.
    Edge differences retain the supplied principal branches and canonical
    orientation. Node rows use neighbor-minus-target differences instead.
    """

    state: StateBounds
    target_index: int
    edge_phase_differences: tuple[I, ...]
    edge_phase_velocities: tuple[I, ...]
    form: I
    pressure: I
    form_rate: I
    form_acceleration: I
    principal_margins: tuple[Q, ...]


def _target_rows(
    capacities, degrees, mobility, edges, differences, velocities, target_index
):
    """Evaluate target rows only from freshly admitted constituent bounds."""
    current, acceleration = I(0), I(0)
    for (left, right), gap, velocity in zip(
        edges, differences, velocities, strict=True
    ):
        if target_index not in (left, right):
            continue
        if target_index == right:
            gap, velocity = -gap, -velocity
        current += sin(gap)
        acceleration += cos(gap) * velocity
    pressure = current / (pi_interval() * degrees[target_index])
    return (
        pressure,
        capacities[target_index] * pressure,
        mobility[target_index] * acceleration,
    )


def bound_node_history_box(
    *,
    epi,
    phases,
    capacities,
    edges,
    conductances,
    shifts,
    target_index,
):
    """Bound direct form, pressure and acceleration on a primitive state box.

    All form and phase coordinates are retained as exact rational scalars or
    outward rational intervals. Capacity and conductance intervals represent
    a family of positive held coefficients, not changing coefficients within
    a trajectory. Admission rebuilds complete rows and the margins of every
    supplied original branch. Nonpositive rectangular margins are permitted:
    they abstain from retention and do not establish a crossing. A separate
    valid certificate can retain a branch even when this box is too wide.
    No cached derivatives or trajectory verdicts are accepted.

    The acceleration is the chain-rule range
    ``M_i*sum(cos(theta_j-theta_i)*(theta_dot_j-theta_dot_i))``, using
    ``theta_dot=M*L_W*x`` on the same box. No time evolution or source-to-box
    reachability is asserted: the supplied box needs independent whole-time
    enclosure evidence before these ranges describe a true history window.
    """
    if type(target_index) is not int:
        raise TypeError("target_index requires an exact integer node index")
    epi, phases, capacities, conductances = (
        tuple(I.coerce(value) for value in values)
        for values in (epi, phases, capacities, conductances)
    )
    edges, shifts = _admit_edges(len(epi), edges), tuple(shifts)
    if not 0 <= target_index < len(epi):
        raise ValueError("target_index must identify a supplied node")
    if len(phases) != len(epi) or len(capacities) != len(epi):
        raise ValueError("one phase and capacity are required per node")
    if len(conductances) != len(edges) or len(shifts) != len(edges):
        raise ValueError("one conductance and branch are required per edge")
    if any(type(shift) is not int for shift in shifts):
        raise TypeError("edge branches require exact integer shifts")
    state = _bound_state(epi, phases, capacities, edges, conductances)
    if not state.positive_mean_metric:
        raise ValueError("history box bounds require positive degree at every node")
    pi = pi_interval()
    mobility = tuple(
        capacity / (pi * degree)
        for capacity, degree in zip(capacities, state.degrees, strict=True)
    )
    differences = tuple(
        phases[right] - phases[left] - 2 * shift * pi
        for (left, right), shift in zip(edges, shifts, strict=True)
    )
    velocities = tuple(
        state.phase_rate[right] - state.phase_rate[left] for left, right in edges
    )
    pressure, form_rate, acceleration = _target_rows(
        capacities,
        state.degrees,
        mobility,
        edges,
        differences,
        velocities,
        target_index,
    )
    return NodeHistoryBox(
        state=state,
        target_index=target_index,
        edge_phase_differences=differences,
        edge_phase_velocities=velocities,
        form=epi[target_index],
        pressure=pressure,
        form_rate=form_rate,
        form_acceleration=acceleration,
        principal_margins=tuple(pi.lo - gap.abs_max for gap in differences),
    )


def bound_node_history_window(
    *,
    epi,
    phases,
    capacities,
    edges,
    conductances,
    shifts,
    horizon,
    target_index,
):
    """Enclose a target's true history from admitted primitive coordinates.

    Coordinates and held coefficients are exact rational scalars or outward
    rational intervals. The shared motion owner admits their dimensions,
    positive capacities/conductances and original strict principal branches.
    A zero horizon returns instantaneous bounds. The complete unforced law is
    ``x_dot=M*S(theta)``, ``theta_dot=M*L_W*x`` with ``M=nu/(pi*degree)``.

    For an edge, its global acceleration magnitude B gives
    ``delta(t) in delta(0)+v(0)*[0,T]+[-B,B]*T**2/2`` and
    ``v(t) in v(0)+[-B*T,B*T]``. The actual sine/cosine chain rule then bounds
    pressure and form acceleration over the entire window. By the positive
    unit-mass three-sample kernel, every genuine three-sample acceleration
    wholly in this window lies in ``form_acceleration``. No samples are
    generated or installed here, and their observation errors remain separate.
    """
    horizon = _horizon(horizon)
    if type(target_index) is not int:
        raise TypeError("target_index requires an exact integer node index")
    epi, edges, shifts = tuple(epi), tuple(edges), tuple(shifts)
    if not 0 <= target_index < len(epi):
        raise ValueError("target_index must identify a supplied node")
    motion = _bound_motion(
        epi=epi,
        phases=phases,
        capacities=capacities,
        edges=edges,
        conductances=conductances,
        shifts=shifts,
        horizon=horizon,
    )
    pi = pi_interval()
    elapsed = I(0, horizon)
    differences, velocities, margins = [], [], []
    jerk_sum = I(0)
    for index, ((left, right), shift) in enumerate(zip(edges, shifts, strict=True)):
        initial_gap = motion.phases[right] - motion.phases[left] - 2 * shift * pi
        initial_velocity = motion.edge_phase_velocity[index]
        acceleration_bound = motion.edge_acceleration_magnitudes[index].hi
        gap = (
            initial_gap
            + initial_velocity * elapsed
            + I(-acceleration_bound, acceleration_bound) * horizon**2 / 2
        )
        velocity = initial_velocity + I(
            -acceleration_bound * horizon, acceleration_bound * horizon
        )
        differences.append(gap)
        velocities.append(velocity)
        margins.append(
            max(
                pi.lo - gap.abs_max,
                motion.horizon_remaining_margins[index].lo,
            )
        )
        if target_index not in (left, right):
            continue
        # Differentiate M_i*sum cos(delta_ij)*delta_dot_ij. The sine and
        # cosine magnitudes are at most one; both edge bounds are global.
        jerk_sum += (
            motion.edge_speed_bounds[index] ** 2
            + motion.edge_acceleration_magnitudes[index]
        )
    mobility = motion.mobility[target_index]
    pressure, form_rate, acceleration = _target_rows(
        motion.capacities,
        motion.degrees,
        motion.mobility,
        edges,
        differences,
        velocities,
        target_index,
    )
    form = (
        motion.epi[target_index]
        + motion.form_rate[target_index] * elapsed
        + acceleration * I(0, horizon**2 / 2)
    )
    return NodeHistoryWindow(
        motion=motion,
        target_index=target_index,
        horizon=horizon,
        edge_phase_differences=tuple(differences),
        edge_phase_velocities=tuple(velocities),
        pressure=pressure,
        form_rate=form_rate,
        form_acceleration=acceleration,
        form=form,
        principal_margins=tuple(margins),
        retained_edge_indices=tuple(
            index for index, margin in enumerate(margins) if margin > 0
        ),
        third_form_derivative_bound=(mobility * jerk_sum).hi,
    )
