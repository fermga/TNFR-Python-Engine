"""All-time parity bounds for actual formed C9 networks and their surrogate.

This detached assessor retains the original source families, the supplied
contact work and the degree-weighted common modes. It installs no flow and
does not extend the earlier finite-window certificate by changing its horizon.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real
from ..mathematics._exact_linear_algebra import exact_symmetric_semidefinite
from ..mathematics._rational_interval import INTERVAL_METHOD, I, cos, pi_interval, sqrt
from ._sine_formed_contact import _unprobed_handoff, _UnprobedHandoff
from ._sine_port_bounds import _joined_port_bounds, _JoinedPortBounds
from ._sine_port_geometry import _central_port_geometry, _PortGeometry
from .relational_sine_port_composition import _parameters, _support

__all__ = ("SinePortRelaxation", "assess_sine_port_relaxation")

_ODD_GAP = Q(7, 30)
_REFINED_COSINE = Q(1, 6)


@dataclass(frozen=True)
class _ParityTrackingBounds:
    """Rational invariant rectangle for admitted weighted parity errors.

    The comparison coordinates are y=phase+gamma*form and z=gamma*form,
    using their weighted norms. They are not independent physical state rows.
    The matrix is [[-a,2],[2,-b]]; nonpositive determinant cannot give this bound.
    """

    gap_lower_bound: Q
    forcing_upper_bound: Q
    initial_norm_upper_bound: Q
    phase_damping: Q
    scaled_form_damping: Q
    determinant: Q
    initial_joint_norm_upper_bound: Q
    initial_scaled_form_norm_upper_bound: Q
    joint_initial_correction: Q
    scaled_form_initial_correction: Q
    joint_norm_upper_bound: Q | None
    scaled_form_norm_upper_bound: Q | None
    phase_norm_upper_bound: Q | None
    certified: bool


def _parity_tracking_bounds(*, gap, forcing, initial_norm, gamma_upper):
    """Use already admitted rational coefficients, without report premises."""
    a = _REFINED_COSINE * gap
    b = gap / gamma_upper**2 - 2
    determinant = a * b - 4
    y0, z0 = (1 + gamma_upper) * initial_norm, gamma_upper * initial_norm
    p, q = max(Q(0), -a * y0 + 2 * z0), max(Q(0), 2 * y0 - b * z0)
    y = z = phase = None
    if determinant > 0:
        y = y0 + (b * (forcing + p) + 2 * (forcing + q)) / determinant
        z = z0 + (2 * (forcing + p) + a * (forcing + q)) / determinant
        phase = y + z
    return _ParityTrackingBounds(
        gap_lower_bound=gap,
        forcing_upper_bound=forcing,
        initial_norm_upper_bound=initial_norm,
        phase_damping=a,
        scaled_form_damping=b,
        determinant=determinant,
        initial_joint_norm_upper_bound=y0,
        initial_scaled_form_norm_upper_bound=z0,
        joint_initial_correction=p,
        scaled_form_initial_correction=q,
        joint_norm_upper_bound=y,
        scaled_form_norm_upper_bound=z,
        phase_norm_upper_bound=phase,
        certified=determinant > 0,
    )


def _normalized_gap_certificate(geometry, gap):
    """Check the exact even weighted quotient; callers also bound odd modes."""
    weights = geometry.layer_masses
    mass, size = sum(weights, Q(0)), len(weights)
    witness = tuple(
        tuple(
            geometry.joined_laplacian[i][j]
            - gap * ((weights[i] if i == j else Q(0)) - weights[i] * weights[j] / mass)
            for j in range(size)
        )
        for i in range(size)
    )
    return exact_symmetric_semidefinite(witness)


@dataclass(frozen=True)
class SinePortRelaxation:
    """All-time coordinate-error envelopes with separate channel resolution.

    The phase and form bounds include the permanent actual-minus-nominal common
    mode floors. A valid envelope need not resolve a requested fraction of the
    supplied origin span. Failure of that sufficient test is not an observed
    tracking error or a failure of the original dynamics. Envelope/channel
    flags describe mathematical bounds; status additionally requires the
    separately declared supplied-work policy.
    """

    classes: tuple[int, ...]
    contacts: tuple[tuple[int, int], ...]
    phase_origins: tuple[Q, ...]
    formation_time: Q
    relaxation_duration: Q
    form_error_bound: Q
    phase_error_bound: Q
    endpoint_radius: Q
    radius: Q
    decay_power: int
    work_allowance: Q
    normalized_gap_lower_bound: Q
    phase_resolution_fraction: Q
    form_resolution_fraction: Q
    geometry: _PortGeometry
    unprobed_handoff: _UnprobedHandoff
    joined_bounds: _JoinedPortBounds
    gamma_bounds: I
    origin_span: Q
    phase_allowance: Q
    form_allowance_bounds: I
    normalized_gap_certified: bool
    edge_deviation_bounds: I | None
    refined_cosine_bounds: I | None
    refined_chart_certified: bool
    edge_disagreement_norm_squared_upper_bound: Q | None
    edge_disagreement_norm_bounds: I | None
    odd_bounds: _ParityTrackingBounds | None
    even_bounds: _ParityTrackingBounds | None
    form_mean_error_floor: Q | None
    phase_mean_error_floor: Q | None
    all_time_form_error_upper_bound: Q | None
    all_time_phase_error_upper_bound: Q | None
    form_resolution_margin_bounds: I | None
    phase_resolution_margin_bounds: I | None
    all_time_envelopes_certified: bool
    form_resolution_certified: bool
    phase_resolution_certified: bool
    joint_resolution_certified: bool
    status: str
    unavailable_reasons: tuple[str, ...]
    resolution_limitations: tuple[str, ...]
    odd_gap_lower_bound: Q = _ODD_GAP
    refined_cosine_lower_bound: Q = _REFINED_COSINE
    clock: str = "tau=e*t; e=1023/1024; estimates cover all postcontact tau>=0"
    interval_method: str = INTERVAL_METHOD
    gap_method: str = "exact_psd_Lhat_minus_gap_times_weighted_mean_free_metric"
    implementation_component_cap: int = 16
    scope: tuple[str, ...] = (
        "same_degree_aware_thirty_or_general_ten_m_coordinate_surrogate",
        "fresh_original_zero_sum_source_errors_and_unprobed_handoff_no_ideal_reset",
        "full_and_surrogate_acute_trapping_before_parity_estimates",
        "reflection_odd_modes_generated_by_internal_nonlinearity_not_discarded",
        "weighted_even_and_odd_error_rectangles_include_initial_exchange",
        "actual_nominal_conserved_mean_difference_retained_as_permanent_floor",
        "uniform_each_fine_coordinate_bounds_for_all_uninterrupted_postcontact_times",
        "origin_span_resolution_is_declared_mathematical_policy_not_a_sensor",
        "insufficient_channel_bound_is_not_observed_tracking_failure",
        "supplied_work_and_fixed_support_no_occurrence_rule_solver_or_physical_identity",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {"schema": "tnfr.sine-port-relaxation.v1", "report": _project(self)}


def assess_sine_port_relaxation(
    *,
    classes,
    contacts,
    phase_origins,
    formation_time,
    relaxation_duration,
    form_error_bound,
    phase_error_bound,
    endpoint_radius,
    radius,
    decay_power,
    work_allowance,
    normalized_gap_lower_bound,
    phase_resolution_fraction,
    form_resolution_fraction,
) -> SinePortRelaxation:
    """Rebuild an all-time tracking certificate from original primitives.

    Support admission shares the composition evaluator's simple unit contacts
    and one-to-sixteen component policy. Require finite nonnegative source times,
    errors and work allowance, positive endpoint radius, 0<radius<=1/12, ordinary
    integer decay_power in0..4096, and 0<normalized_gap_lower_bound<=7/30.
    The proposed gap is checked, not assumed. Resolution fractions lie in(0,1].
    No contact duration or incoming report is admitted. Existing source and
    relaxation exponential work caps still apply. Missing theorem premises or
    unresolved margins produce explicit unavailable or partial results.
    """
    kinds, edges, origins = _support(classes, contacts, phase_origins)
    raw = dict(
        formation_time=formation_time,
        relaxation_duration=relaxation_duration,
        form_error_bound=form_error_bound,
        phase_error_bound=phase_error_bound,
        endpoint_radius=endpoint_radius,
        radius=radius,
        work_allowance=work_allowance,
        normalized_gap_lower_bound=normalized_gap_lower_bound,
        phase_resolution_fraction=phase_resolution_fraction,
        form_resolution_fraction=form_resolution_fraction,
    )
    values = {
        name: exact_or_represented_real(value, name) for name, value in raw.items()
    }
    if any(value < 0 for value in values.values()):
        raise ValueError(
            "source times, budgets and resolution parameters must be nonnegative"
        )
    if not 0 < values["radius"] <= Q(1, 12) or values["endpoint_radius"] <= 0:
        raise ValueError("require 0 < radius <= 1/12 and positive endpoint_radius")
    if not 0 < values["normalized_gap_lower_bound"] <= _ODD_GAP:
        raise ValueError("normalized_gap_lower_bound must be positive and at most7/30")
    if any(
        not 0 < values[key] <= 1
        for key in ("phase_resolution_fraction", "form_resolution_fraction")
    ):
        raise ValueError("resolution fractions must lie in(0,1]")
    if type(decay_power) is not int or not 0 <= decay_power <= 4096:
        raise ValueError("decay_power must be an ordinary integer zero through4096")
    n, eps = len(kinds), values["endpoint_radius"]
    geometry = _central_port_geometry(n, edges)
    gap = values["normalized_gap_lower_bound"]
    gap_ok = _normalized_gap_certificate(geometry, gap)
    handoff = _unprobed_handoff(
        formation_time=values["formation_time"],
        relaxation_duration=values["relaxation_duration"],
        form_error_bound=values["form_error_bound"],
        phase_error_bound=values["phase_error_bound"],
        radius=values["radius"],
        endpoint_radius=eps,
        decay_power=decay_power,
    )
    joined = _joined_port_bounds(
        geometry=geometry,
        phase_origins=origins,
        endpoint_radius=eps,
        radius=values["radius"],
        work_allowance=values["work_allowance"],
        handoff_available=all(handoff.handoff_certified_by_class),
    )
    gamma, _ = _parameters(kinds)
    span = max(origins) - min(origins)
    phase_allowance = values["phase_resolution_fraction"] * span
    form_allowance = gamma * (values["form_resolution_fraction"] * span)
    edge_radius = refined_cosine = b2 = b_norm = odd = even = None
    mean_floor = form_error = phase_error = form_margin = phase_margin = None
    refined = envelopes = phase_resolved = form_resolved = False
    if joined.identity_certified:
        energy = joined.joined_excess_storage_upper_bound
        edge_radius = sqrt(I(2 * energy / joined.joined_cosine_bounds.lo))
        refined_cosine = cos(4 * pi_interval() / 9 + edge_radius)
        refined = refined_cosine.lo >= _REFINED_COSINE
        if gap_ok and refined:
            root_two = sqrt(I(2))
            b2 = 2 * energy / _REFINED_COSINE
            b_norm = sqrt(I(b2))
            odd_forcing = root_two.hi * b2 / 2
            # Scale the exact tiny preparation radius after square-root
            # enclosure; rounding eps**2 first would lose its useful size.
            odd_initial = sqrt(I(2 * n)).hi * eps
            even_initial = sqrt(I((2 + max(geometry.contact_degrees)) * n)).hi * eps
            odd = _parity_tracking_bounds(
                gap=_ODD_GAP,
                forcing=odd_forcing,
                initial_norm=odd_initial,
                gamma_upper=gamma.hi,
            )
            if odd.certified:
                even_forcing = (
                    2 * b_norm.hi * odd.phase_norm_upper_bound
                    + root_two.hi * b_norm.hi * b2 / 6
                )
                even = _parity_tracking_bounds(
                    gap=gap,
                    forcing=even_forcing,
                    initial_norm=even_initial,
                    gamma_upper=gamma.hi,
                )
                if even.certified:
                    mean_floor = Q(2 * len(edges), 18 * n + 2 * len(edges)) * eps
                    phase_error = (
                        sqrt(
                            I(
                                odd.phase_norm_upper_bound**2
                                + even.phase_norm_upper_bound**2
                            )
                        ).hi
                        / root_two.lo
                        + mean_floor
                    )
                    form_error = (
                        sqrt(
                            I(
                                odd.scaled_form_norm_upper_bound**2
                                + even.scaled_form_norm_upper_bound**2
                            )
                        ).hi
                        / (gamma.lo * root_two.lo)
                        + mean_floor
                    )
                    phase_margin = I(phase_allowance - phase_error)
                    form_margin = form_allowance - form_error
                    envelopes = True
                    phase_resolved = phase_margin.lo > 0
                    form_resolved = form_margin.lo > 0
    reasons = tuple(
        reason
        for condition, reason in (
            (
                handoff.formation_certificate.status == "certified_two_formed_classes",
                "formation_unavailable",
            ),
            (
                all(handoff.handoff_certified_by_class),
                "unprobed_endpoint_budget_not_certified",
            ),
            (joined.supported, "connected_multi_component_identity_not_supported"),
            (joined.identity_certified, "whole_network_identity_not_certified"),
            (
                joined.work_within_allowance,
                "supplied_contact_work_allowance_not_certified",
            ),
            (gap_ok, "normalized_gap_not_certified"),
            (refined, "refined_acute_chart_not_certified"),
            (envelopes, "parity_tracking_envelopes_not_certified"),
        )
        if not condition
    )
    limitations = tuple(
        reason
        for condition, reason in (
            (phase_resolved, "phase_resolution_not_certified"),
            (form_resolved, "form_resolution_not_certified"),
        )
        if not condition
    )
    status = "unavailable"
    if not reasons:
        status = (
            "full"
            if phase_resolved and form_resolved
            else (
                "phase_only"
                if phase_resolved
                else "form_only" if form_resolved else "envelopes_only"
            )
        )
    return SinePortRelaxation(
        **values,
        classes=kinds,
        contacts=edges,
        phase_origins=origins,
        decay_power=decay_power,
        geometry=geometry,
        unprobed_handoff=handoff,
        joined_bounds=joined,
        gamma_bounds=gamma,
        origin_span=span,
        phase_allowance=phase_allowance,
        form_allowance_bounds=form_allowance,
        normalized_gap_certified=gap_ok,
        edge_deviation_bounds=edge_radius,
        refined_cosine_bounds=refined_cosine,
        refined_chart_certified=refined,
        edge_disagreement_norm_squared_upper_bound=b2,
        edge_disagreement_norm_bounds=b_norm,
        odd_bounds=odd,
        even_bounds=even,
        form_mean_error_floor=mean_floor,
        phase_mean_error_floor=mean_floor,
        all_time_form_error_upper_bound=form_error,
        all_time_phase_error_upper_bound=phase_error,
        form_resolution_margin_bounds=form_margin,
        phase_resolution_margin_bounds=phase_margin,
        all_time_envelopes_certified=envelopes,
        form_resolution_certified=form_resolved,
        phase_resolution_certified=phase_resolved,
        joint_resolution_certified=phase_resolved and form_resolved,
        status=status,
        unavailable_reasons=reasons,
        resolution_limitations=limitations,
    )
