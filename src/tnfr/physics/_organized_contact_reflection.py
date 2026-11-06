"""Detached illustration of one analytically defined reflected contact saddle.

ORGANIZED_CONTACT_REFLECTION_OBSTRUCTION.md proves the reflection-sector
obstruction and the exact cubic identities. Small enclosing residuals below
audit those identities; they do not establish stationarity on their own.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from ..mathematics._rational_interval import I, arg, pi_interval, sqrt
from ._organized_contact_maintenance import _derive_organized_contact_maintenance
from ._organized_external_interaction import _OrganizedExternalInteractionGeometry
from ._repeated_organized_composition import _bounded_sine_edge_fields

_BISECTIONS = 128
_ROOT_DOMAIN = (Q(-1, 20), Q(0))


def _polynomial(z):
    return ((16 * z + 36) * z + 24) * z + 1


@dataclass(frozen=True)
class _OrganizedContactReflection:
    geometry: _OrganizedExternalInteractionGeometry
    root_bracket: tuple[Q, Q]
    root_endpoint_values: tuple[Q, Q]
    root_derivative_bounds: I
    root_bounds: I
    circle_coordinate_bounds: tuple[tuple[I, I], ...]
    child_origin_bounds: tuple[I, ...]
    form: tuple[Q, ...]
    phase_lift_bounds: tuple[I, ...]
    internal_gap_bounds: I
    outer_gap_bounds: tuple[I, ...]
    winding_sum_bounds: tuple[I, ...]
    phase_current_bounds: tuple[I, ...]
    edge_hessian_bounds: tuple[I, ...]
    centered_cut: tuple[Q, ...]
    cut_quadratic_bounds: I
    algebraic_cut_quadratic_bounds: I
    root_isolated: bool
    stationary_identity_certified: bool
    stationary_residuals_consistent: bool
    marked_windings_certified: bool
    stationary_instability_certified: bool
    bisections: int = _BISECTIONS
    protocol_revision: str = "9c7c3ef4"
    source_status: str = "unavailable_source_entry_and_maintenance_not_assessed"
    scope: tuple[str, ...] = (
        "one_analytic_stationary_example_not_a_numerical_existence_search",
        "cubic_unit_circle_and_sine_identities_prove_stationarity",
        "zero_enclosing_residuals_only_audit_the_analytic_identity",
        "all_fifty_four_forms_and_phases_and_fifteen_marked_cycles_retained",
        "centered_cut_is_a_full_relative_phase_hessian_witness",
        "negative_curvature_implies_stationary_two_row_instability",
        "two_conserved_common_means_remain_neutral",
        "example_does_not_prove_the_general_reflection_sector_obstruction",
        "no_source_trajectory_event_or_fundamental_physical_identification",
    )


def _derive_organized_contact_reflection():
    """Enclose the single declared algebraic root with exactly 128 bisections."""
    lower, upper = _ROOT_DOMAIN
    for _ in range(_BISECTIONS):
        midpoint = (lower + upper) / 2
        if _polynomial(midpoint) <= 0:
            lower = midpoint
        else:
            upper = midpoint
    endpoint_values = (_polynomial(lower), _polynomial(upper))
    domain = I(*_ROOT_DOMAIN)
    derivative = 24 * (2 * domain + 1) * (domain + 1)
    isolated = derivative.lo > 0 and endpoint_values[0] < 0 < endpoint_values[1]
    z = I(lower, upper)
    sine_magnitude = sqrt(1 - z**2)
    factor = 1 + 2 * z
    a_circle = (z, -sine_magnitude)
    b_circle = (-(2 + z) * factor / 2, sine_magnitude * factor / 2)
    a, b = arg(*a_circle), arg(*b_circle)
    origins = (I(0), b, a, -a, -b, I(0))
    geometry = _derive_organized_contact_maintenance().interaction_geometry
    pi, alpha = pi_interval(), geometry.alpha_bounds
    outer_gaps = (b, 2 * pi + a - b, -2 * a - 2 * pi, 2 * pi + a - b, b)
    offsets = tuple(origin for origin in origins for _ in range(9))
    phases = tuple(alpha * (j - 4) + origins[c] for c in range(6) for j in range(9))
    current, potential, _ = _bounded_sine_edge_fields(
        phase_offsets=offsets,
        weighted_edges=geometry.weighted_edges,
        alpha_bounds=alpha,
    )
    hessian = tuple(1 - value for value in potential)
    cut = (Q(5, 6),) * 9 + (Q(-1, 6),) * 45
    quadratic = sum(
        (
            weight * (cut[right] - cut[left]) ** 2
            for (left, right, _), weight in zip(geometry.weighted_edges, hessian)
        ),
        I(0),
    )
    algebraic_quadratic = 9 * (-1 - 5 * z - 2 * z**2)
    # P(z)=0 makes the b pair a unit vector; its sine identities cancel
    # every wheel current. Equal internal twists cancel the C9 currents.
    # The proof owns these exact identities, not interval overlap with zero.
    stationary = isolated and a_circle[1].hi < 0 < b_circle[1].lo
    residuals = all(value.contains(0) for value in current)
    # With these strict branches the symbolic five-gap sum is exactly 2*pi.
    # Likewise each internal cycle has nine copies of alpha=2*pi/9.
    winding = (
        stationary
        and 0 < alpha.lo <= alpha.hi < pi.lo
        and all(-pi.lo < gap.lo <= gap.hi < pi.lo for gap in outer_gaps)
    )
    return _OrganizedContactReflection(
        geometry=geometry,
        root_bracket=(lower, upper),
        root_endpoint_values=endpoint_values,
        root_derivative_bounds=derivative,
        root_bounds=z,
        circle_coordinate_bounds=(a_circle, b_circle),
        child_origin_bounds=origins,
        form=(Q(0),) * 54,
        phase_lift_bounds=phases,
        internal_gap_bounds=alpha,
        outer_gap_bounds=outer_gaps,
        winding_sum_bounds=(9 * alpha,) * 6 + (sum(outer_gaps, I(0)),) * 9,
        phase_current_bounds=current,
        edge_hessian_bounds=hessian,
        centered_cut=cut,
        cut_quadratic_bounds=quadratic,
        algebraic_cut_quadratic_bounds=algebraic_quadratic,
        root_isolated=isolated,
        stationary_identity_certified=stationary,
        stationary_residuals_consistent=residuals,
        marked_windings_certified=winding,
        stationary_instability_certified=(
            stationary and residuals and quadratic.hi < 0
        ),
    )
