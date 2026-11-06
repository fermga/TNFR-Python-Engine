"""Static complete-law motion bounds for the hypothetical maintained successor.

The law holds the selected support, conductances and positive capacities fixed:
``x_dot=M*S(theta)``, ``theta_dot=M*L_W*x``, with zero forcing. Source derivatives
are instantaneous observations. Energy/Cauchy bounds apply to every later state
of that same law; an unresolved branch margin does not establish an exit.
No operator, trajectory, source producer or horizon search is executed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real
from ..mathematics._rational_interval import I, cos, pi_interval, sin, sqrt
from .maintained_parent_um_work import _bound_state
from .maintained_successor_chamber import (
    SuccessorChamber,
    _admit_edges,
    predict_successor_chamber,
)


@dataclass(frozen=True)
class MotionBounds:
    """One admitted fixed law and its source, in the chamber's node/edge order.

    ``edge_phase_acceleration`` is signed and instantaneous. In contrast,
    ``edge_acceleration_magnitudes`` and ``edge_speed_bounds`` enclose
    nonnegative bounds valid throughout the fixed law. The latter use conserved
    source storage, not an assumed phase tube. Positive remaining margins
    certify retention through the requested horizon; other entries abstain.
    """

    epi: tuple[I, ...]
    phases: tuple[I, ...]
    capacities: tuple[I, ...]
    conductances: tuple[I, ...]
    degrees: tuple[int, ...]
    mobility: tuple[I, ...]
    form_rate: tuple[I, ...]
    phase_rate: tuple[I, ...]
    phase_acceleration: tuple[I, ...]
    edge_phase_velocity: tuple[I, ...]
    edge_phase_acceleration: tuple[I, ...]
    edge_acceleration_magnitudes: tuple[I, ...]
    form_storage: I
    phase_storage: I
    storage: I
    storage_rate: I
    edge_energy_coefficients: tuple[I, ...]
    edge_speed_bounds: tuple[I, ...]
    principal_margins: tuple[I, ...]
    horizon_remaining_margins: tuple[I, ...]
    retained_edge_indices: tuple[int, ...]


@dataclass(frozen=True)
class SuccessorMotionPrediction:
    """Ideal reset and conditional output allowance remain distinct laws."""

    chamber: SuccessorChamber
    ideal: MotionBounds
    conditional: MotionBounds
    horizon: Q


def _horizon(value):
    value = exact_or_represented_real(value, "motion horizon")
    if value < 0:
        raise ValueError("motion horizon must be nonnegative")
    return value


def _bound_motion(*, epi, phases, capacities, edges, conductances, shifts, horizon):
    """Bound a fixed positive law from primitive exact or interval coordinates.

    Private algebra controls may supply a smaller graph. No cached derivative,
    storage, branch verdict or source association is accepted as a premise.
    The incidence orientation is -1 at the left endpoint and +1 at the right.
    """
    epi, phases, capacities, conductances = (
        tuple(I.coerce(value) for value in values)
        for values in (epi, phases, capacities, conductances)
    )
    count = len(epi)
    edges = _admit_edges(count, edges)
    shifts = tuple(shifts)
    if len(phases) != count or len(capacities) != count:
        raise ValueError("one phase and capacity are required per node")
    if len(conductances) != len(edges) or len(shifts) != len(edges):
        raise ValueError("one conductance and branch are required per edge")
    if any(type(shift) is not int for shift in shifts):
        raise TypeError("edge branches require exact integer shifts")
    horizon = _horizon(horizon)
    state = _bound_state(epi, phases, capacities, edges, conductances)
    if not state.positive_mean_metric:
        raise ValueError("motion bounds require positive degree at every node")
    pi = pi_interval()
    mobility = tuple(
        value / (pi * degree)
        for value, degree in zip(capacities, state.degrees, strict=True)
    )
    strength = [I(0) for _ in epi]
    acceleration = [I(0) for _ in epi]
    # Sparse rows of A=M*L_W*M suffice; no edge-by-edge dense matrix is built.
    acceleration_rows = [{} for _ in epi]
    form_storage, phase_storage, storage_rate = I(0), I(0), I(0)
    differences = []
    for (left, right), weight, shift in zip(edges, conductances, shifts, strict=True):
        strength[left] += weight
        strength[right] += weight
        form_gap = epi[right] - epi[left]
        difference = phases[right] - phases[left] - 2 * shift * pi
        differences.append(difference)
        form_storage += weight * form_gap**2 / 2
        phase_storage += 1 - cos(difference)
        storage_rate += (
            weight * form_gap * (state.form_rate[right] - state.form_rate[left])
        )
        storage_rate += sin(difference) * (
            state.phase_rate[right] - state.phase_rate[left]
        )
        form_rate_gradient = weight * (state.form_rate[left] - state.form_rate[right])
        acceleration[left] += mobility[left] * form_rate_gradient
        acceleration[right] -= mobility[right] * form_rate_gradient
        for node, neighbor in ((left, right), (right, left)):
            row = acceleration_rows[node]
            row[node] = row.get(node, I(0)) + mobility[node] ** 2 * weight
            row[neighbor] = -mobility[node] * mobility[neighbor] * weight
    storage = form_storage + phase_storage
    coefficients, speed_bounds, acceleration_magnitudes = [], [], []
    for (left, right), weight in zip(edges, conductances, strict=True):
        coefficient = (
            mobility[left] ** 2 * strength[left]
            + mobility[right] ** 2 * strength[right]
            + 2 * mobility[left] * mobility[right] * weight
        )
        coefficients.append(coefficient)
        speed_bounds.append(sqrt(2 * storage * coefficient))
        left_row, right_row = acceleration_rows[left], acceleration_rows[right]
        acceleration_magnitudes.append(
            sum(
                (
                    abs(right_row.get(node, I(0)) - left_row.get(node, I(0)))
                    * state.degrees[node]
                    for node in sorted(left_row.keys() | right_row.keys())
                ),
                I(0),
            )
        )
    margins = tuple(pi - abs(value) for value in differences)
    if any(margin.lo <= 0 for margin in margins):
        raise ValueError("source must certify every strict principal branch")
    remaining = tuple(
        margin - horizon * speed
        for margin, speed in zip(margins, speed_bounds, strict=True)
    )
    return MotionBounds(
        epi=epi,
        phases=phases,
        capacities=capacities,
        conductances=conductances,
        degrees=state.degrees,
        mobility=mobility,
        form_rate=state.form_rate,
        phase_rate=state.phase_rate,
        phase_acceleration=tuple(acceleration),
        edge_phase_velocity=tuple(
            state.phase_rate[b] - state.phase_rate[a] for a, b in edges
        ),
        edge_phase_acceleration=tuple(
            acceleration[b] - acceleration[a] for a, b in edges
        ),
        edge_acceleration_magnitudes=tuple(acceleration_magnitudes),
        form_storage=form_storage,
        phase_storage=phase_storage,
        storage=storage,
        storage_rate=storage_rate,
        edge_energy_coefficients=tuple(coefficients),
        edge_speed_bounds=tuple(speed_bounds),
        principal_margins=margins,
        horizon_remaining_margins=remaining,
        retained_edge_indices=tuple(
            index for index, value in enumerate(remaining) if value.lo > 0
        ),
    )


def predict_successor_motion(graph, *, horizon=Q(1, 2)):
    """Re-admit the primitive hypothetical reset and bound its complete motion.

    A supplied graph must satisfy the existing source/reset contracts. Reports
    and cached verdicts do not bypass that admission. The conditional result
    assumes the previous reset's per-output allowance; it does not certify an
    actual implementation response. Every retained edge uses its original lift.
    """
    horizon = _horizon(horizon)
    chamber = predict_successor_chamber(graph)

    def bound(phases, capacities, conductances):
        return _bound_motion(
            epi=chamber.reset.epi,
            phases=phases,
            capacities=capacities,
            edges=chamber.edges,
            conductances=conductances,
            shifts=chamber.ideal.shifts,
            horizon=horizon,
        )

    return SuccessorMotionPrediction(
        chamber=chamber,
        ideal=bound(
            chamber.ideal.phases, chamber.reset.capacity_after, chamber.conductances
        ),
        conditional=bound(
            chamber.conditional.phases,
            chamber.conditional_capacities,
            chamber.conditional_conductances,
        ),
        horizon=horizon,
    )
