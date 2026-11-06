"""Required work for assembly preserving three acquired acute C9 identities.

ORGANIZED_ASSEMBLY_WORK.md owns the endpoint theorem. This detached
assessment quantifies over the declared endpoint class; it does not admit
an individual reset, select an event or transfer post-reset capture.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._rational_interval import I, pi_interval
from ._organized_network_compatibility import (
    _EQUAL_CONTROL,
    _UNEQUAL_CONTROL,
    _derive_organized_triangle,
    _mixings,
    _OrganizedTriangle,
)
from ._sine_constitutive_recovery import (
    _assess_constitutive_recovery,
    _ConstitutiveRecovery,
)


@dataclass(frozen=True)
class _OrganizedAssemblyWork:
    """Conditional storage floor and source-dependent work obstruction.

    Floor intervals enclose expressions that lower-bound storage, not all
    possible endpoint values. The exact subgrid source correction is kept
    separately. An unavailable source has no actual required-work verdict.
    """

    recovery: _ConstitutiveRecovery
    geometry: _OrganizedTriangle
    contact_floor_bounds: tuple[I, ...]
    total_contact_floor_bounds: I
    source_excess_upper_bound: Q
    source_storage_bounds: I
    endpoint_storage_floor_bounds: I
    conditional_work_lower_bound: Q
    required_work_lower_bound: Q | None
    source_certified: bool
    bound_certified: bool
    zero_supply_excluded: bool
    status: str
    unresolved_conditions: tuple[str, ...]
    endpoint_windings: tuple[int, ...] = (1, 1, 1)
    endpoint_domain: str = "strictly_acute_internal_circular_gaps_on_each_marked_C9"
    scope: tuple[str, ...] = (
        "original_three_recovered_sources_with_all_fine_errors_retained",
        "arbitrary_finite_signed_forms_and_circular_phases_at_reset_endpoint",
        "no_endpoint_mean_registered_family_or_cross_edge_acuteness_requirement",
        "original_unit_internal_edges_and_fixed_final_oriented_contacts_retained",
        "held_unit_capacity_and_same_graph_dependent_sine_storage",
        "nonsharp_endpoint_floor_not_a_minimizer_or_post_reset_capture",
        "passive_finite_schedules_cannot_evade_same_endpoint_storage_gap",
        "no_reservoir_operator_admission_event_selection_or_physical_energy_claim",
    )


def _assess_organized_assembly_work(
    *, mixing, form_error_bound, phase_error_bound
) -> _OrganizedAssemblyWork:
    """Rebuild primitive source and support before certifying required work.

    For each directed contact, minimization over a tail phase and convexity
    on the head's acute winding-one cycle give 9*(1-rho_s). Summing these
    relaxed floors does not claim that their minimizers coexist on a cycle.
    """
    weights = _mixings(mixing)
    rx = _exact(form_error_bound, "form_error_bound")
    rt = _exact(phase_error_bound, "phase_error_bound")
    if rx < 0 or rt < 0:
        raise ValueError("preparation error bounds must be nonnegative")
    if weights not in (_EQUAL_CONTROL, _UNEQUAL_CONTROL):
        raise ValueError("assembly work requires one of the two fixed contact controls")
    recovery = _assess_constitutive_recovery(
        deformation=Q(0), form_error_bound=rx, phase_error_bound=rt
    )
    geometry = _derive_organized_triangle(mixing=weights)
    floors = tuple(9 * (1 - contact.amplitude_bounds) for contact in geometry.contacts)
    contact_floor = sum(floors, I(0))
    epsilon = recovery.endpoint_radius
    # On three disjoint unit C9s, L <= 4I and the twist is critical.
    # Half x^T L x + the phase Taylor remainder <= 12*epsilon**2.
    source_excess = 12 * epsilon**2
    baseline = geometry.internal_storage_bounds
    source_bounds = baseline + I(0, source_excess)
    endpoint_floor = baseline + contact_floor
    # Cancel the shared exact baseline analytically before interval rounding.
    # Keep the subgrid source excess as a separate exact rational subtraction.
    conditional = contact_floor.lo - source_excess
    source_acute = 2 * pi_interval().hi / 9 + 2 * epsilon < pi_interval().lo / 2
    source_ok = recovery.recovery_certified and source_acute
    bound_ok = all(value.lo > 0 for value in floors) and conditional > 0
    unresolved = []
    if not recovery.recovery_certified:
        unresolved.append("original_source_recovery_not_certified")
    if not source_acute:
        unresolved.append("original_source_internal_acuteness_not_certified")
    if not bound_ok:
        unresolved.append("positive_required_work_not_certified")
    certified = source_ok and bound_ok
    return _OrganizedAssemblyWork(
        recovery=recovery,
        geometry=geometry,
        contact_floor_bounds=floors,
        total_contact_floor_bounds=contact_floor,
        source_excess_upper_bound=source_excess,
        source_storage_bounds=source_bounds,
        endpoint_storage_floor_bounds=endpoint_floor,
        conditional_work_lower_bound=conditional,
        required_work_lower_bound=conditional if certified else None,
        source_certified=source_ok,
        bound_certified=bound_ok,
        zero_supply_excluded=certified,
        status="certified_positive_assembly_work" if certified else "unavailable",
        unresolved_conditions=tuple(unresolved),
    )
