"""Detached regional storage and conditional identity-retention budgets.

ORGANIZED_REGIONAL_RETENTION.md owns the first-exit theorem. The observer
evaluates the actual degree-four sine law at supplied complete state; the
assessment rebuilds release bounds and tests a supplied all-prefix budget.
Neither operation observes a trajectory or authenticates accumulated work.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._rational_interval import I, cos, pi_interval, sin
from ..types import require_finite_real_scalar_epi
from ._organized_network_compatibility import (
    _EQUAL_CONTROL,
    _derive_organized_triangle,
    _OrganizedTriangle,
)
from ._organized_receiver_slow_admission import (
    _assess_organized_receiver_slow_admission,
    _OrganizedReceiverSlowAdmission,
)
from ._sine_preparation import _prepared_form_costs
from .relational_observations import _ordered

_FIRST_EXIT_BARRIER = Q(9, 4)


@dataclass(frozen=True)
class _OrganizedRegionalRetention:
    """Source bounds and an optional conditional donor-budget inequality.

    A nonnegative supplied prefix upper bound concerns each donor's net
    boundary work minus accumulated internal dissipation after release.
    Passing the inequality does not establish that the source realizes it.
    Receiver reserve is necessary for a first exit, not sufficient for
    acquisition or evidence of its direction or occurrence.
    """

    preparation: _OrganizedReceiverSlowAdmission
    release_storage_upper_bounds: tuple[Q, Q, Q]
    net_work_reserves: tuple[Q, Q, Q]
    donor_net_work_prefix_upper_bound: Q | None
    donor_budget_margin: Q | None
    donor_budget_inequality_passes: bool | None
    source_admitted: bool
    release_windings: tuple[int, int, int] | None
    status: str
    unresolved_conditions: tuple[str, ...]
    first_exit_barrier: Q = _FIRST_EXIT_BARRIER
    clock: str = "tau=(1023/1024)*t"
    accumulated_work_observed: bool = False
    donor_retention_measured: bool = False
    receiver_acquisition: str = "unresolved"
    scope: tuple[str, ...] = (
        "fixed_equal_contact_two_ramp_source_rebuilt_from_primitive_errors",
        "regional_internal_storage_excludes_incident_contact_storage",
        "C9_first_exit_barrier_for_initial_winding_zero_or_one",
        "supplied_net_work_budget_must_hold_at_every_prefix_after_release",
        "strict_donor_budget_is_conditional_not_measured_retention",
        "receiver_net_work_reserve_is_necessary_not_sufficient_for_first_exit",
        "no_trajectory_event_selector_work_measurement_or_physical_identification",
    )


def _assess_organized_regional_retention(
    *, form_error_bound, phase_error_bound, donor_net_work_prefix_upper_bound=None
) -> _OrganizedRegionalRetention:
    """Rebuild the unchanged source before testing a declared prefix budget."""
    rx = _exact(form_error_bound, "form_error_bound")
    rt = _exact(phase_error_bound, "phase_error_bound")
    budget = (
        None
        if donor_net_work_prefix_upper_bound is None
        else _exact(
            donor_net_work_prefix_upper_bound, "donor_net_work_prefix_upper_bound"
        )
    )
    if rx < 0 or rt < 0:
        raise ValueError("preparation error bounds must be nonnegative")
    if budget is not None and budget < 0:
        raise ValueError(
            "the net-work prefix bound includes zero and must be nonnegative"
        )
    source = _assess_organized_receiver_slow_admission(
        form_error_bound=rx, phase_error_bound=rt
    )
    correction = (
        2 * source.release_form_norm_upper_bound**2
        + 2 * source.release_phase_error_norm_upper_bound**2
    )
    donor = (9 * (1 - cos(2 * pi_interval() / 9))).hi + correction
    release = (donor, donor, correction)
    reserves = tuple(_FIRST_EXIT_BARRIER - value for value in release)
    admitted = (
        source.status == "certified_uniform_slow_admission"
        and source.release_windings == (1, 1, 0)
        and min(reserves) > 0
    )
    margin = None if budget is None else min(reserves[:2]) - budget
    passes = None if budget is None or not admitted else margin > 0
    unresolved = tuple(
        reason
        for failed, reason in (
            (not admitted, "source_release_or_positive_reserve_not_certified"),
            (budget is None, "all_prefix_net_boundary_work_bound_not_supplied"),
            (passes is False, "strict_donor_budget_inequality_not_satisfied"),
        )
        if failed
    )
    return _OrganizedRegionalRetention(
        preparation=source,
        release_storage_upper_bounds=release,
        net_work_reserves=reserves,
        donor_net_work_prefix_upper_bound=budget,
        donor_budget_margin=margin,
        donor_budget_inequality_passes=passes,
        source_admitted=admitted,
        release_windings=source.release_windings,
        status="conditional_donor_budget_passes" if passes else "unavailable",
        unresolved_conditions=unresolved,
    )


@dataclass(frozen=True)
class _OrganizedRegionWork:
    """One internal store and the actual incident power at supplied state."""

    region: int
    nodes: tuple[int, ...]
    form_gradient: tuple[Q, ...]
    contact_form_gradient: tuple[Q, ...]
    internal_current_bounds: tuple[I, ...]
    contact_current_bounds: tuple[I, ...]
    form_storage: Q
    phase_storage_bounds: I
    storage_bounds: I
    dissipation: Q
    boundary_power_bounds: I
    storage_rate_bounds: I
    storage_rate_from_rows_bounds: I
    balance_residual_bounds: I


@dataclass(frozen=True)
class _OrganizedRegionalWork:
    """Static full-law balance; no temporal work or retention is observed.

    Form follows the shared finite signed-scalar materialization contract,
    including uniform-real BEPI. Phase values are admitted continuous lifts,
    not means inferred from circular observations. All calculations use the
    returned normalized primitive state.
    """

    geometry: _OrganizedTriangle
    form: tuple[Q, ...]
    phase_lifts: tuple[Q, ...]
    form_gradient: tuple[Q, ...]
    phase_current_bounds: tuple[I, ...]
    form_rate_bounds: tuple[I, ...]
    phase_rate_bounds: tuple[I, ...]
    regions: tuple[_OrganizedRegionWork, ...]
    form_storage: Q
    phase_storage_bounds: I
    storage_bounds: I
    contact_form_storage: Q
    contact_phase_storage_bounds: I
    contact_storage_bounds: I
    contact_storage_rate_from_rows_bounds: I
    storage_partition_residual_bounds: I
    dissipation: Q
    storage_rate: Q
    storage_rate_from_rows_bounds: I
    balance_residual_bounds: I
    rate_partition_residual_bounds: I
    clock: str = "tau=(1023/1024)*t"
    accumulated_work_observed: bool = False
    scope: tuple[str, ...] = (
        "supplied_complete_27_node_state_on_fixed_equal_contact_support",
        "held_unit_capacity_actual_degree_four_sine_law_in_tau_clock",
        "both_actual_rows_and_all_internal_and_contact_storage_retained",
        "regional_boundary_power_has_no_prescribed_sign",
        "instantaneous_balance_is_not_accumulated_work_or_a_closed_regional_law",
        "no_source_acquisition_retention_event_or_trajectory_verdict",
    )


def _observe_organized_regional_work(*, form, phase) -> _OrganizedRegionalWork:
    """Admit all primitive values before constructing support or doing algebra."""
    raw_form = _ordered(form, "form", limit=28)
    raw_phase = _ordered(phase, "phase", limit=28)
    if len(raw_form) != 27 or len(raw_phase) != 27:
        raise ValueError("form and phase require 27 values in registered node order")
    values = tuple(
        _exact(require_finite_real_scalar_epi(value, f"form[{i}]"), f"form[{i}]")
        for i, value in enumerate(raw_form)
    )
    phases = tuple(_exact(value, f"phase[{i}]") for i, value in enumerate(raw_phase))
    geometry = _derive_organized_triangle(mixing=_EQUAL_CONTROL)
    edges = geometry.weighted_edges
    internal_edges = tuple(edge for edge in edges if edge[0] // 9 == edge[1] // 9)
    costs, gradient = _prepared_form_costs(form=values, weighted_edges=edges)
    internal_costs, internal_gradient = _prepared_form_costs(
        form=values, weighted_edges=internal_edges
    )
    contact_gradient = tuple(q - ell for q, ell in zip(gradient, internal_gradient))
    internal_current, contact_current = [I(0)] * 27, [I(0)] * 27
    internal_potential = [I(0)] * 3
    contact_potential = I(0)
    for a, b, weight in edges:
        angle = I(phases[b] - phases[a])
        current = weight * sin(angle)
        potential = weight * (1 - cos(angle))
        internal = a // 9 == b // 9
        channel = internal_current if internal else contact_current
        channel[a] += current
        channel[b] -= current
        if internal:
            internal_potential[a // 9] += potential
        else:
            contact_potential += potential
    current = tuple(u + v for u, v in zip(internal_current, contact_current))
    gamma, k = geometry.gamma_bounds, geometry.clock_factor
    dx = tuple(k * (-q + gamma * s) for q, s in zip(gradient, current))
    dtheta = tuple(k * gamma * q for q in gradient)
    regional = []
    for region in range(3):
        nodes = tuple(range(9 * region, 9 * (region + 1)))
        ell = tuple(internal_gradient[i] for i in nodes)
        boundary = tuple(contact_gradient[i] for i in nodes)
        u = tuple(internal_current[i] for i in nodes)
        v = tuple(contact_current[i] for i in nodes)
        dissipation = k * sum((value**2 for value in ell), Q(0))
        power = k * (
            -sum((a * b for a, b in zip(ell, boundary)), Q(0))
            + gamma
            * sum((a * d - c * b for a, b, c, d in zip(ell, boundary, u, v)), I(0))
        )
        direct = sum(
            (
                internal_gradient[i] * dx[i] - internal_current[i] * dtheta[i]
                for i in nodes
            ),
            I(0),
        )
        form_cost = sum(
            (
                cost
                for (a, _, _), cost in zip(internal_edges, internal_costs)
                if a // 9 == region
            ),
            Q(0),
        )
        rate = power - dissipation
        regional.append(
            _OrganizedRegionWork(
                region=region,
                nodes=nodes,
                form_gradient=ell,
                contact_form_gradient=boundary,
                internal_current_bounds=u,
                contact_current_bounds=v,
                form_storage=form_cost,
                phase_storage_bounds=internal_potential[region],
                storage_bounds=form_cost + internal_potential[region],
                dissipation=dissipation,
                boundary_power_bounds=power,
                storage_rate_bounds=rate,
                storage_rate_from_rows_bounds=direct,
                balance_residual_bounds=direct - rate,
            )
        )
    form_cost = sum(costs, Q(0))
    contact_form = form_cost - sum(internal_costs, Q(0))
    phase_cost = sum(internal_potential, contact_potential)
    storage = form_cost + phase_cost
    contact_storage = contact_form + contact_potential
    loss = k * sum((value**2 for value in gradient), Q(0))
    direct = sum(
        (q * x - s * theta for q, s, x, theta in zip(gradient, current, dx, dtheta)),
        I(0),
    )
    contact_direct = sum(
        (
            b * x - v * theta
            for b, v, x, theta in zip(contact_gradient, contact_current, dx, dtheta)
        ),
        I(0),
    )
    return _OrganizedRegionalWork(
        geometry=geometry,
        form=values,
        phase_lifts=phases,
        form_gradient=gradient,
        phase_current_bounds=current,
        form_rate_bounds=dx,
        phase_rate_bounds=dtheta,
        regions=tuple(regional),
        form_storage=form_cost,
        phase_storage_bounds=phase_cost,
        storage_bounds=storage,
        contact_form_storage=contact_form,
        contact_phase_storage_bounds=contact_potential,
        contact_storage_bounds=contact_storage,
        contact_storage_rate_from_rows_bounds=contact_direct,
        storage_partition_residual_bounds=storage
        - contact_storage
        - sum((r.storage_bounds for r in regional), I(0)),
        dissipation=loss,
        storage_rate=-loss,
        storage_rate_from_rows_bounds=direct,
        balance_residual_bounds=direct + loss,
        rate_partition_residual_bounds=direct
        - contact_direct
        - sum((r.storage_rate_from_rows_bounds for r in regional), I(0)),
    )
