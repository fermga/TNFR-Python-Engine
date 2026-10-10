"""Own-law C9 boundary minima for the composition-compatible counterfamily.

CONSTITUTIVE_IDENTITY_BARRIERS.md proves the minimum on a registered winding
cell boundary, not a dynamically reached activation cost. No preparation,
transport, prefix work or capture certificate is imported from the sine law.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from ..mathematics._rational_interval import I, cos, pi_interval
from ._composition_compatible_constitutive import (
    _CompositionCompatibleLaw,
    _derive_composition_compatible_law,
)

_BOUNDARY_PROOF_LIMIT = Q(1, 10000)


@dataclass(frozen=True)
class _ConstitutiveIdentityBarriers:
    """Geometric minimum and target margin with their own-law proof domain.

    ``boundary_storage_bounds`` applies to both declared initial windings.
    It is also the flat receiver's geometric boundary cost. The donor margin
    starts at an exact isolated twist with zero form, not at an acquired
    release family. Missing minima remain unavailable outside the proof.
    """

    law: _CompositionCompatibleLaw
    sine_boundary_storage_bounds: I
    barrier_slope_bounds: I
    target_storage_bounds: I
    large_gap_exclusion_lower_bound: Q
    remaining_gap_convexity_lower_bound: Q
    opposite_face_exclusion_lower_bound: Q
    boundary_storage_bounds: I | None
    barrier_change_bounds: I | None
    donor_target_margin_bounds: I | None
    geometric_minimum_certified: bool
    status: str
    unresolved_conditions: tuple[str, ...]
    proof_deformation_limit: Q = _BOUNDARY_PROOF_LIMIT
    initial_windings: tuple[int, int] = (0, 1)
    source_admitted: bool = False
    trajectory_evaluated: bool = False
    scope: tuple[str, ...] = (
        "unit_C9_registered_first_contact_boundary_with_all_gaps_in_closed_branch",
        "exact_geometric_minimum_not_an_observed_or_sufficient_passage_cost",
        "same_own_law_barrier_for_initial_winding_zero_and_one",
        "target_margin_uses_exact_zero_form_twist_not_a_source_release_bound",
        "constitutive_storage_changes_do_not_change_the_supplied_phase_row",
        "no_transferred_source_defect_acquisition_retention_or_capture_certificate",
    )


def _assess_constitutive_identity_barriers(*, deformation):
    """Rebuild the primitive law and admit the narrower boundary theorem."""
    law = _derive_composition_compatible_law(deformation=deformation)
    e = law.deformation
    size = abs(e)
    # Rational proof margins use cos(pi/8)>23/25 and
    # 1/4<cos(5*pi/12)<1/3, as derived by the theorem owner.
    exclusion = Q(2, 75) - 36 * size
    convexity = Q(1, 4) - 328 * size
    opposite = Q(9, 4) - Q(16, 25) - 64 * size
    pi = pi_interval()
    cosine = cos(pi / 8)
    baseline = 2 + 8 * (1 - cosine)
    slope = 8 * cosine * (1 + cosine) ** 2 - 4
    target = 9 * (1 - cos(2 * pi / 9))
    admitted = (
        size <= _BOUNDARY_PROOF_LIMIT
        and min(exclusion, convexity, opposite) > 0
        and slope.lo > 23
    )
    change = e * slope if admitted else None
    boundary = baseline + change if admitted else None
    return _ConstitutiveIdentityBarriers(
        law=law,
        sine_boundary_storage_bounds=baseline,
        barrier_slope_bounds=slope,
        target_storage_bounds=target,
        large_gap_exclusion_lower_bound=exclusion,
        remaining_gap_convexity_lower_bound=convexity,
        opposite_face_exclusion_lower_bound=opposite,
        boundary_storage_bounds=boundary,
        barrier_change_bounds=change,
        donor_target_margin_bounds=boundary - target if admitted else None,
        geometric_minimum_certified=admitted,
        status="certified_geometric_boundary_minimum" if admitted else "unavailable",
        unresolved_conditions=(
            () if admitted else ("own_law_boundary_minimum_not_certified",)
        ),
    )
