"""Identity retention from protected weighted-star subgraphs of actual support.

The complete source law and all represented coordinates are re-admitted. A
fixed lower storage barrier survives arbitrary additional nonnegative edges
and positive held capacities; no ideal contact footprint is substituted.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction as Q
from typing import Any

from ..dynamics.relational import RelationalExchangeModel
from ..mathematics._rational_interval import INTERVAL_METHOD, I, cos, sqrt
from ._sine_admission import _sine_model_coefficients
from ._sine_weighted_star import CONDUCTANCES, EDGES
from .relational_observations import _ordered
from .relational_sine_comparison import _validate_comparison_labels
from .relational_sine_support import (
    SineSupportExchangeComparison,
    _rebuild_sine_support_comparison,
)

__all__ = (
    "SineWeightedStarSupportRetention",
    "assess_sine_weighted_star_support_retention",
)

_BARRIER = Q(35, 64)


@dataclass(frozen=True)
class SineWeightedStarSupportRetention:
    """Actual-support storage criterion for two supplied marked identities.

    Each star order is center, three leaves, boundary. Phase observations use
    cosine of actual radian differences, with no invented rational-turn lift.
    An unavailable certificate is not identity loss; other arguments can
    retain states outside this sufficient energy sublevel.
    """

    source: SineSupportExchangeComparison
    stars: tuple[tuple[Any, ...], tuple[Any, ...]]
    protected_edges: tuple[tuple[Any, Any], ...]
    protected_weight_minima: tuple[Q, ...]
    protected_weights: tuple[Q, ...]
    protected_edge_checks: tuple[bool, ...]
    internal_form_gaps: tuple[tuple[Q, ...], tuple[Q, ...]]
    internal_phase_cosine_bounds: tuple[tuple[I, ...], tuple[I, ...]]
    source_identity_status: str
    source_identity_admitted: bool
    full_storage_bounds: I
    storage_margin: Q
    internal_form_gap_upper_bound: I
    internal_phase_cosine_lower_bound: Q
    all_time_identity_retention_certified: bool
    status: str
    unavailable_reasons: tuple[str, ...]
    form_radius: Q = Q(1, 2)
    phase_identity: str = "principal_circular_internal_gap_strictly_below_pi_over_two"
    identity_storage_boundary_lower_bound: Q = _BARRIER
    clock: str = "t"
    arithmetic_method: str = INTERVAL_METHOD
    temporal_scope: str = "all_real_times_under_the_admitted_unchanged_held_law"
    scope: tuple[str, ...] = (
        "two_disjoint_marked_five_node_stars_cover_the_complete_ten_node_source",
        "original_star_edges_present_at_or_above_the_declared_positive_weights",
        "additional_nonnegative_conductances_retain_their_bare_phase_incidence",
        "positive_held_capacities_zero_loss_unit_exchange_and_storage_scale",
        "complete_original_clock_source_rebuilt_from_primitive_state_and_law",
        "cosine_observations_use_actual_radian_coordinates_and_circular_identity",
        "protected_subgraph_barrier_is_a_lower_bound_not_a_sharp_joined_threshold",
        "retention_flag_uses_only_this_sufficient_storage_criterion",
        "higher_storage_or_unresolved_bounds_do_not_prove_identity_loss",
        "no_word_admission_event_execution_trajectory_acquisition_or_physical_identity",
    )

    def to_dict(self):
        if type(self) is not SineWeightedStarSupportRetention:
            raise TypeError("a SineWeightedStarSupportRetention is required")
        fresh = assess_sine_weighted_star_support_retention(
            self.source, stars=self.stars
        )
        _validate_comparison_labels(fresh.source)
        from ..sdk.relational_reports import _project

        return {
            "schema": "tnfr.sine-weighted-star-support-retention.v1",
            "report": _project(fresh),
        }


def _marked_stars(nodes, stars):
    if len(nodes) != 10:
        raise ValueError("star retention requires exactly ten source nodes")
    raw = _ordered(stars, "stars", limit=3)
    if len(raw) != 2:
        raise ValueError("stars must contain exactly two marked stars")
    raw = tuple(_ordered(star, "marked star", limit=6) for star in raw)
    if any(len(star) != 5 for star in raw):
        raise ValueError(
            "each marked star must contain center, three leaves and boundary"
        )
    positions = {node: index for index, node in enumerate(nodes)}
    try:
        indices = tuple(tuple(positions[node] for node in star) for star in raw)
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "every marked node must belong to the admitted source"
        ) from exc
    if len({index for star in indices for index in star}) != 10:
        raise ValueError("marked stars must partition all ten source nodes")
    # Retain the admitted source labels after resolving equivalent graph keys.
    return tuple(tuple(nodes[index] for index in star) for star in indices), indices


def assess_sine_weighted_star_support_retention(
    source, *, stars
) -> SineWeightedStarSupportRetention:
    """Rebuild a complete actual support and apply the protected-star barrier.

    Missing or weakened original edges violate this theorem's premises and
    raise; they are not an observed failure of maintenance. Finite enclosures
    that do not resolve the strict storage inequality explicitly abstain.
    """
    admitted = _rebuild_sine_support_comparison(source)
    if _sine_model_coefficients(admitted.reference_model) != (0, 1, 1):
        raise ValueError("star retention requires zero loss and unit exchange and beta")
    if any(value <= 0 for value in admitted.capacity):
        raise ValueError("star retention requires strictly positive held capacities")
    # The consumed coefficients are admitted above; retain their normalized
    # declaration rather than relying on equality with a supplied model cache.
    admitted = replace(
        admitted,
        reference_model=RelationalExchangeModel(
            1, epi_weight=0, phase_weight=1, phase_domain="regular"
        ),
    )
    stars, indices = _marked_stars(admitted.nodes, stars)
    protected = tuple((star[i], star[j]) for star in stars for i, j in EDGES)
    minima = tuple(Q(value) for value in CONDUCTANCES * 2)
    weights = {
        frozenset(edge): weight
        for edge, weight in zip(admitted.edges, admitted.conductances)
    }
    if any(frozenset(edge) not in weights for edge in protected):
        raise ValueError("every protected star edge must remain present")
    actual_weights = tuple(weights[frozenset(edge)] for edge in protected)
    checks = tuple(value >= minimum for value, minimum in zip(actual_weights, minima))
    if not all(checks):
        raise ValueError("protected star conductances must meet their original minima")

    form_gaps = tuple(
        tuple(abs(admitted.epi[star[0]] - admitted.epi[leaf]) for leaf in star[1:4])
        for star in indices
    )
    phase_cosines = tuple(
        tuple(
            cos(I(admitted.phase[leaf] - admitted.phase[star[0]])) for leaf in star[1:4]
        )
        for star in indices
    )
    form_admitted = all(gap < Q(1, 2) for row in form_gaps for gap in row)
    phase_admitted = all(value.lo > 0 for row in phase_cosines for value in row)
    phase_excluded = any(value.hi <= 0 for row in phase_cosines for value in row)
    identity = form_admitted and phase_admitted
    identity_status = (
        "admitted"
        if identity
        else "excluded" if not form_admitted or phase_excluded else "unavailable"
    )
    upper = admitted.storage.hi
    if upper < 0:
        raise ArithmeticError("nonnegative full storage has a negative upper bound")
    margin = _BARRIER - upper
    if margin > 0 and identity_status == "excluded":
        raise ArithmeticError(
            "strict storage sublevel contradicts its identity boundary"
        )
    certified = identity and margin > 0
    reasons = tuple(
        reason
        for failed, reason in (
            (not identity, "source_internal_identity_not_certified"),
            (margin <= 0, "strict_full_storage_margin_not_certified"),
        )
        if failed
    )
    return SineWeightedStarSupportRetention(
        source=admitted,
        stars=stars,
        protected_edges=protected,
        protected_weight_minima=minima,
        protected_weights=actual_weights,
        protected_edge_checks=checks,
        internal_form_gaps=form_gaps,
        internal_phase_cosine_bounds=phase_cosines,
        source_identity_status=identity_status,
        source_identity_admitted=identity,
        full_storage_bounds=admitted.storage,
        storage_margin=margin,
        internal_form_gap_upper_bound=sqrt(I(Q(16, 35) * upper)),
        internal_phase_cosine_lower_bound=1 - upper,
        all_time_identity_retention_certified=certified,
        status="certified_identity_retention" if certified else "unavailable",
        unavailable_reasons=reasons,
    )
