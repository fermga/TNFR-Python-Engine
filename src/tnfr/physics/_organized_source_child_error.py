"""Prospective child-swap error and incoming energy-crossing bounds.

The averaged state is an explicitly approximate preparation, distinct from
strict restriction to the exact child-swap sector. The shared complete fine
law supplies initial observations. Scalar comparison bounds control its
mathematical incoming-law extensions without evaluating a trajectory or hit.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._comparison_flow import scalar_affine_upper
from ..mathematics._rational_interval import I, cos
from ._organized_source_child_quotient import (
    _derive_organized_source_child_quotient,
    _OrganizedSourceChildQuotientState,
    _restrict_organized_source_child_quotient,
)
from ._organized_source_port import (
    _observe_organized_source_port,
    _OrganizedSourceObservation,
)
from ._repeated_organized_composition import _vector


@dataclass(frozen=True)
class _OrganizedSourceChildErrorBound:
    """Inspectable bounds, not a sampled response or an executed guard.

    ``projected_state_error_bound`` bounds each form and phase-offset error
    between the projected fine incoming solution and the exact-sector core
    initialized at ``projected_state``. ``energy_error_bound`` compares the
    original, unprojected fine solution's energy with the core's energy
    throughout the same horizon. ``crossing_time_error_bound``
    is present only when both initial signs, both positive loss floors and the
    strict terminal bracket are certified. It is a difference bound between
    two first hits, never either evaluated hitting time.
    """

    form: tuple[Q, ...]
    phase_offsets: tuple[Q, ...]
    capacities: tuple[Q, ...]
    horizon: Q
    guard_level: Q
    projected_state: _OrganizedSourceChildQuotientState
    fine_initial: _OrganizedSourceObservation
    core_initial: _OrganizedSourceObservation
    hidden_form_coordinates: tuple[Q, ...]
    hidden_phase_offset_coordinates: tuple[Q, ...]
    initial_hidden_radius: Q
    gamma_upper: Q
    lipschitz_bound: Q
    quadratic_defect_coefficient: Q
    growth_upper: Q
    convolution_upper: Q
    hidden_radius_bound: Q
    initial_form_radius: Q
    form_radius_bound: Q
    projected_state_error_bound: Q
    energy_error_bound: Q
    gradient_drift_bound: Q
    fine_initial_excess_bounds: I
    core_initial_excess_bounds: I
    fine_loss_floor: Q
    core_loss_floor: Q
    fine_initial_guard_margin: Q
    core_initial_guard_margin: Q
    terminal_guard_margin: Q
    crossing_certified: bool
    crossing_time_error_bound: Q | None
    unavailable_reasons: tuple[str, ...]
    scope: tuple[str, ...] = (
        "all_primitives_readmitted_before_projection_geometry_or_observation",
        "fixed_shared_forty_five_node_ninety_unit_edge_support",
        "unit_incoming_capacities_required_not_inferred_or_replaced",
        "declared_real_phase_offsets_global_sine_law_without_acute_cell_admission",
        "orbit_averaging_is_explicitly_approximate_not_strict_sector_restriction",
        "both_form_and_phase_rows_use_the_same_complete_tau_clock_law",
        "quadratic_projected_state_and_energy_errors_on_incoming_law_extensions",
        "positive_loss_and_strict_sign_margins_are_required_for_first_hit_bounds",
        "comparison_work_cap_L_times_horizon_at_most_one_is_numerical_policy",
        "no_trajectory_evaluated_hit_guard_selector_or_executed_event",
        "no_acute_retention_word_context_runtime_admission_or_postword_claim",
    )


def _bound_organized_source_child_error(
    *, form, phase_offsets, capacities, horizon, guard_level
):
    """Bound departure from the child-swap core and a fixed incoming guard.

    Normalize every primitive before geometry or equality checks. Average
    only the swapped children, then pass that exactly symmetric preparation
    through the existing strict restriction. The original primitive state
    is retained and is never presented as belonging to the exact sector.

    With hidden initial sup radius h0, g>=gamma, L=5*(1+g)/2 and C=5*g/4,
    the comparison gives H=h0*exp(L*T) and projected error
    e<=C*H**2*phi(L,T). The maximum principle supplies
    X=X0+5*g*T/4. Ninety-edge accounting then bounds the energy discrepancy
    by 108*H**2+(360*X+180)*e, without assuming an acute phase domain.

    Initial gradients and a uniform drift allowance yield positive loss
    floors when possible. Both initial energies must be strictly above the
    supplied guard, and the core terminal upper bound plus the energy error
    must be strictly below it. Failure leaves the bounds available with
    explicit reasons; it does not assert that either crossing fails to exist.
    """
    values = _vector(form, "form", 45)
    offsets = _vector(phase_offsets, "phase_offsets", 45)
    mobilities = _vector(capacities, "capacities", 45)
    duration = _exact(horizon, "horizon")
    level = _exact(guard_level, "guard_level")
    if any(value != 1 for value in mobilities):
        raise ValueError("child error bound requires unit incoming capacities")
    if duration <= 0:
        raise ValueError("child error bound requires a strictly positive horizon")

    geometry = _derive_organized_source_child_quotient()
    gamma = geometry.fine_geometry.gamma_bounds.hi
    lipschitz = 5 * (1 + gamma) / 2
    coefficient = 5 * gamma / 4
    if lipschitz * duration > 1:
        raise ValueError("child error comparison requires L*horizon <= 1")

    def average(row):
        coordinates = tuple(
            sum((row[node] for node in block), Q(0)) / len(block)
            for block in geometry.blocks
        )
        return tuple(coordinates[a] for a in geometry.node_blocks)

    projected = _restrict_organized_source_child_quotient(
        form=average(values),
        phase_offsets=average(offsets),
        capacities=mobilities,
    )
    pairs = tuple(block for block in geometry.blocks if len(block) == 2)
    hidden_form = tuple((values[a] - values[b]) / 2 for a, b in pairs)
    hidden_phase = tuple((offsets[a] - offsets[b]) / 2 for a, b in pairs)
    hidden_initial = max(abs(value) for value in hidden_form + hidden_phase)
    growth = scalar_affine_upper(1, lipschitz, duration, 0)
    convolution = scalar_affine_upper(0, lipschitz, duration, 1)
    # Keep all source-dependent scaling exact outside the dyadic comparison
    # kernel. Exact symmetry stays zero, and arbitrarily tiny nonzero hidden
    # radii retain their quadratic scaling instead of a grid-sized floor.
    hidden_bound = hidden_initial * growth
    state_error = coefficient * hidden_bound**2 * convolution
    initial_radius = max(abs(value) for value in values)
    form_bound = initial_radius + 5 * gamma * duration / 4
    energy_error = 108 * hidden_bound**2 + (360 * form_bound + 180) * state_error
    drift = (25 * initial_radius + 25 * gamma / 2) * duration
    drift += 125 * gamma * duration**2 / 8

    fine = _observe_organized_source_port(form=values, phase_offsets=offsets)
    core = _observe_organized_source_port(
        form=projected.form, phase_offsets=projected.phase_offsets
    )
    carrier_storage = 45 * (1 - cos(geometry.fine_geometry.alpha_bounds))
    fine_excess = fine.joint_storage_bounds - carrier_storage
    core_excess = core.joint_storage_bounds - carrier_storage

    def loss_floor(gradient):
        return sum((max(abs(q) - drift, Q(0)) ** 2 for q in gradient), Q(0)) / 4

    fine_floor = loss_floor(fine.form_gradient)
    core_floor = loss_floor(core.form_gradient)
    fine_margin = fine_excess.lo - level
    core_margin = core_excess.lo - level
    terminal_margin = level - (core_excess.hi - core_floor * duration + energy_error)
    reasons = []
    for margin, reason in (
        (fine_margin, "fine_initial_energy_not_certified_strictly_above_guard"),
        (core_margin, "core_initial_energy_not_certified_strictly_above_guard"),
        (fine_floor, "fine_loss_floor_not_strictly_positive"),
        (core_floor, "core_loss_floor_not_strictly_positive"),
        (terminal_margin, "terminal_energy_not_certified_strictly_below_guard"),
    ):
        if margin <= 0:
            reasons.append(reason)
    certified = not reasons
    time_error = energy_error / min(fine_floor, core_floor) if certified else None
    return _OrganizedSourceChildErrorBound(
        form=values,
        phase_offsets=offsets,
        capacities=mobilities,
        horizon=duration,
        guard_level=level,
        projected_state=projected,
        fine_initial=fine,
        core_initial=core,
        hidden_form_coordinates=hidden_form,
        hidden_phase_offset_coordinates=hidden_phase,
        initial_hidden_radius=hidden_initial,
        gamma_upper=gamma,
        lipschitz_bound=lipschitz,
        quadratic_defect_coefficient=coefficient,
        growth_upper=growth,
        convolution_upper=convolution,
        hidden_radius_bound=hidden_bound,
        initial_form_radius=initial_radius,
        form_radius_bound=form_bound,
        projected_state_error_bound=state_error,
        energy_error_bound=energy_error,
        gradient_drift_bound=drift,
        fine_initial_excess_bounds=fine_excess,
        core_initial_excess_bounds=core_excess,
        fine_loss_floor=fine_floor,
        core_loss_floor=core_floor,
        fine_initial_guard_margin=fine_margin,
        core_initial_guard_margin=core_margin,
        terminal_guard_margin=terminal_margin,
        crossing_certified=certified,
        crossing_time_error_bound=time_error,
        unavailable_reasons=tuple(reasons),
    )
