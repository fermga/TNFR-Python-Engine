"""Lifted reservoir memory and conditional full-state local attraction.

ORGANIZED_PARENT_RESERVOIR_MEMORY.md owns the frozen comparison and proof.
This observer re-admits a supplied complete state. It does not establish that
an acquired source enters its sufficient neighborhood, or give an entry time.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._rational_interval import I, cos, pi_interval, sin
from ._organized_parent_reservoir_exchange import (
    _CLOCK_FACTOR,
    _observe_admitted_parent_reservoir_exchange,
    _ParentReservoirObservation,
)
from ._repeated_organized_composition import _vector

_PHASE_RADIUS = Q(1, 8)
_ENERGY_BARRIER = Q(1, 768)
_REGISTRATION_GAP = Q(1, 4)


def _leakage(value):
    if type(value) is not str or value not in ("zero", "original"):
        raise ValueError("leakage must be 'zero' or 'original'")
    return value


def _equilibrium_gap_bounds(retained_quantity):
    """Enclose the unique root of sin(B)+2B=J/3 for exact |J|<=3.

    Monotonicity has global derivative at least one. A sine enclosure that
    meets zero therefore narrows the root by its residual; it never chooses
    a root side or asserts equality from overlapping intervals. At most 64
    bisections use the shared outward rational intervals; precision left
    unresolved by that fixed calculation is retained in the output.
    """
    retained = _exact(retained_quantity, "retained_quantity")
    if abs(retained) > 3:
        raise ValueError("equilibrium classification requires abs(J)<=3")
    if retained == 0:
        return I(0)
    magnitude = abs(retained)
    # For 0<=B<=1/2, 0<=sin(B)<=B gives this analytic initial bracket.
    lower, upper = magnitude / 9, magnitude / 6
    for _ in range(64):
        middle = (lower + upper) / 2
        residual = sin(I(middle)) + 2 * middle - magnitude / 3
        if residual.lo > 0:
            upper = middle
        elif residual.hi < 0:
            lower = middle
        else:
            lower = max(lower, middle - max(residual.hi, Q(0)))
            upper = min(upper, middle + max(-residual.lo, Q(0)))
            break
    result = I(lower, upper)
    return result if retained > 0 else -result


@dataclass(frozen=True)
class _ParentReservoirMemory:
    """Conditional equilibrium and trapping evidence from complete primitives.

    Distances use all 36 lifted offsets and the Euclidean norm. Relative
    energy encloses the signed difference of the original joint storages;
    its lower interval endpoint is not clipped when cancellation crosses zero.
    Registration classifies the target geometry, separately from local entry.
    """

    observation: _ParentReservoirObservation
    leakage: str
    retained_quantity: Q
    retained_tau_rate_bounds: I
    retained_slow_rate_bounds: I
    retained_original_rate_bounds: I
    form_sum: Q
    phase_offset_sum: Q
    chart_margins: tuple[tuple[str, Q], ...]
    common_modes_admitted: bool
    chart_admitted: bool
    retained_range_admitted: bool
    equilibrium_available: bool
    equilibrium_phase_gap_bounds: I | None
    equilibrium_reservoir_bounds: I | None
    equilibrium_phase_offsets_bounds: tuple[I, ...] | None
    equilibrium_storage_bounds: I | None
    registration_threshold_bounds: I
    equilibrium_registration: str
    state_parent_registration: tuple[str, str]
    phase_distance_squared_bounds: I | None
    relative_energy_bounds: I | None
    phase_ball_margin: Q | None
    energy_barrier_margin: Q | None
    phase_ball_admitted: bool
    energy_barrier_admitted: bool
    trap_certified: bool
    attraction_certified: bool
    status: str
    unresolved_conditions: tuple[str, ...]
    phase_ball_radius: Q = _PHASE_RADIUS
    trap_energy_barrier: Q = _ENERGY_BARRIER
    phase_coercivity_lower_bound: Q = Q(1, 12)
    parent_registration_gap: Q = _REGISTRATION_GAP
    protocol_revision: str = "53cc7b68"
    clock: str = "tau=(1023/1024)*t; s=gamma^2*tau/2"
    scope: tuple[str, ...] = (
        "all_73_complete_state_primitives_admitted_before_joint_observation",
        "fixed_connected_port_only_zero_or_original_positive_leakage",
        "exact_zero_common_form_and_lifted_phase_means_without_projection",
        "retained_quantity_depends_on_the_declared_continuous_phase_lift",
        "zero_leakage_attraction_is_to_one_target_on_its_actual_fixed_J_leaf",
        "independent_reservoir_perturbations_change_the_zero_leakage_target",
        "original_leakage_does_not_conserve_J_and_has_the_zero_target",
        "strict_phase_ball_and_signed_energy_upper_bound_are_both_required",
        "all_fine_directions_are_retained_in_the_local_trapping_argument",
        "interval_overlap_is_unresolved_not_exact_registration_boundary",
        "asymptotic_attraction_has_no_finite_rate_deadline_or_source_entry_claim",
        "no_trajectory_or_physical_reservoir_identification",
    )


def _assess_parent_reservoir_memory(
    *, form, phase_offsets, reservoir_state, leakage
) -> _ParentReservoirMemory:
    """Classify a supplied full state without reducing or resetting its means."""
    selected = _leakage(leakage)
    values = _vector(form, "form", 36)
    offsets = _vector(phase_offsets, "phase_offsets", 36)
    reservoir = _exact(reservoir_state, "reservoir_state")
    observation = _observe_admitted_parent_reservoir_exchange(
        form=values,
        phase_offsets=offsets,
        reservoir_state=reservoir,
        port="connected",
        leakage=selected,
    )
    network = observation.network
    geometry = network.geometry
    signs = tuple(2 * (node // 9 % 2) - 1 for node in range(36))
    retained = reservoir + sum(
        (sign * value / 3 for sign, value in zip(signs, offsets)), Q(0)
    )
    rate = -observation.reservoir_decay_bounds * reservoir
    form_sum, phase_sum = sum(values, Q(0)), sum(offsets, Q(0))
    means = form_sum == 0 and phase_sum == 0
    internal = tuple(
        geometry.alpha_bounds + offsets[b] - offsets[a]
        for a, b, _ in geometry.internal_edges
    )
    contacts = tuple(
        offsets[b] - offsets[a]
        for a, b, _ in geometry.parent_edges + geometry.outer_edges
    )
    margins = (
        ("child_positive_gap", min(gap.lo for gap in internal)),
        ("child_acute_gap", pi_interval().lo / 2 - max(gap.hi for gap in internal)),
        ("contact_acute_gap", pi_interval().lo / 2 - max(map(abs, contacts))),
    )
    chart = all(margin > 0 for _, margin in margins)
    in_range = selected == "original" or abs(retained) <= 3
    available = means and in_range
    threshold = Q(3, 2) + 3 * sin(I(_REGISTRATION_GAP))

    def parent_registration(parent):
        if not chart:
            return "unavailable"
        maximum = max(
            abs(offsets[18 * parent + 9 + j] - offsets[18 * parent + j])
            for j in range(9)
        )
        if maximum < _REGISTRATION_GAP:
            return "registered"
        if maximum > _REGISTRATION_GAP:
            return "unregistered"
        return "boundary"

    gap = target_r = target_offsets = target_storage = distance = energy = None
    phase_margin = energy_margin = None
    registration = "unavailable"
    if available:
        gap = I(0) if selected == "original" else _equilibrium_gap_bounds(retained)
        target_r = I(0) if gap == I(0) else 3 * sin(gap)
        target_offsets = tuple(sign * gap / 2 for sign in signs)
        target_storage = (
            36 * (1 - cos(geometry.alpha_bounds))
            + 18 * (1 - cos(gap))
            + target_r**2 / 2
        )
        distance = sum(
            ((offset - target) ** 2 for offset, target in zip(offsets, target_offsets)),
            I(0),
        )
        energy = observation.joint_storage_bounds - target_storage
        phase_margin = _PHASE_RADIUS**2 - distance.hi
        energy_margin = _ENERGY_BARRIER - energy.hi
        if selected == "original" or abs(retained) < threshold.lo:
            registration = "registered"
        elif abs(retained) > threshold.hi:
            registration = "unregistered"
        else:
            registration = "unresolved"
    phase_admitted = phase_margin is not None and phase_margin > 0
    energy_admitted = energy_margin is not None and energy_margin > 0
    trap = available and chart and phase_admitted and energy_admitted
    conditions = (
        ("zero_common_modes", means),
        ("acute_winding_and_contact_chart", chart),
        ("zero_leakage_retained_range", in_range),
        ("strict_initial_phase_ball", phase_admitted),
        ("strict_joint_energy_barrier", energy_admitted),
    )
    return _ParentReservoirMemory(
        observation=observation,
        leakage=selected,
        retained_quantity=retained,
        retained_tau_rate_bounds=rate,
        retained_slow_rate_bounds=2 / geometry.gamma_bounds**2 * rate,
        retained_original_rate_bounds=_CLOCK_FACTOR * rate,
        form_sum=form_sum,
        phase_offset_sum=phase_sum,
        chart_margins=margins,
        common_modes_admitted=means,
        chart_admitted=chart,
        retained_range_admitted=in_range,
        equilibrium_available=available,
        equilibrium_phase_gap_bounds=gap,
        equilibrium_reservoir_bounds=target_r,
        equilibrium_phase_offsets_bounds=target_offsets,
        equilibrium_storage_bounds=target_storage,
        registration_threshold_bounds=threshold,
        equilibrium_registration=registration,
        state_parent_registration=tuple(parent_registration(p) for p in range(2)),
        phase_distance_squared_bounds=distance,
        relative_energy_bounds=energy,
        phase_ball_margin=phase_margin,
        energy_barrier_margin=energy_margin,
        phase_ball_admitted=phase_admitted,
        energy_barrier_admitted=energy_admitted,
        trap_certified=trap,
        attraction_certified=trap,
        status="certified_local_attraction" if trap else "local_attraction_unavailable",
        unresolved_conditions=tuple(name for name, passed in conditions if not passed),
    )
