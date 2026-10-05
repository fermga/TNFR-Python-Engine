"""Analytic preparation-to-capture bounds for the complete sine law.

The globally smooth law supplies a weighted Duhamel estimate through nonacute
passage. The resulting full-state endpoint enclosure is passed to the shared
sector capture owner without a trajectory run, fabricated observation or
equilibrium target. Supplied form information, support and coefficients remain
preparation premises.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real
from ..dynamics.relational import RelationalExchangeModel
from ..mathematics._rational_interval import INTERVAL_METHOD, I, sqrt
from ._sine_preparation import _sine_preparation
from .phase_cycle_geometry import PhaseCycleGeometry
from .relational_sine_comparison import (
    SineExchangeComparison,
    _validate_comparison_labels,
)
from .relational_sine_pattern import SineRelativePattern
from .relational_sine_recovery import SineSectorCapture, _certify_sine_sector_set
from .reversible_eigenmode_reference import _negative_exp_bounds

__all__ = ("SinePreparedEntry", "certify_sine_prepared_entry")


@dataclass(frozen=True)
class SinePreparedEntry:
    """Conditional same-law acquisition from a complete preparation set.

    The node intervals are rectangular outer bounds. Tighter edge intervals
    retain the same weighted-norm correlation and each member's conserved
    means; they
    need not contain every point of the larger Cartesian product of node
    intervals. Capture is proved for the enclosed actual endpoint set.
    Pattern endpoints are centered separately by each member's conserved
    means; their absolute endpoints and common origins remain unavailable.
    ``phase_radius_upper_bound`` bounds the dynamic Duhamel correction only.
    Node and edge phase intervals additionally include the initial form and
    phase residual errors.
    """

    source: SineExchangeComparison | SineRelativePattern
    reference_model: RelationalExchangeModel
    geometry: PhaseCycleGeometry
    scaled_time: Q
    horizon: Q
    coefficient_ratio: Q
    weighted_form_mean: Q | None
    common_initial_phase: Q | None
    initial_form_storage: Q | None
    initial_cycle_periods: tuple[int, ...] | None
    mobility: tuple[Q, ...]
    metric_weights: tuple[Q, ...]
    weighted_gap_lower_bound: Q
    exponential_decay_bounds: I
    form_to_phase_scale_bounds: I
    feedback_strength_bounds: I
    scaled_initial_form_bounds: tuple[I, ...]
    scaled_initial_norm_bounds: I
    forcing_norm_bounds: I
    scaled_form_radius_upper_bound: Q
    phase_radius_upper_bound: Q
    endpoint_form_bounds: tuple[I, ...] | None
    endpoint_phase_bounds: tuple[I, ...] | None
    endpoint_form_edge_gap_bounds: tuple[I, ...]
    endpoint_phase_edge_gap_bounds: tuple[I, ...]
    capture: SineSectorCapture
    hypothesis_failures: tuple[str, ...]
    unresolved_conditions: tuple[str, ...]
    status: str
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "exact_phase_flat_source_or_original_complete_correlated_residual_preparation_set",
        "same_complete_normalized_sine_law_positive_held_capacity_and_loss_throughout",
        "scaled_clock_tau_equals_epi_weight_times_structural_time_all_rows_transformed",
        "full_support_weighted_mean_and_exact_rational_reversible_quotient_gap",
        "global_sine_current_bound_and_weighted_Duhamel_estimate_allow_nonacute_transit",
        "shared_rational_negative_exponential_with_exponent_at_most_4096",
        "full_node_endpoint_boxes_and_tighter_correlated_weighted_norm_edge_bounds",
        "each_actual_trajectory_retains_its_own_initial_weighted_form_and_phase_means",
        "unknown_common_origins_make_absolute_pattern_endpoints_unavailable",
        "weighted_centering_contracts_residual_norm_and_retains_node_and_edge_error_bounds",
        "node_box_is_an_outer_projection_not_an_independent_initial_observation_set",
        "shared_all_face_total_storage_capture_without_supplied_equilibrium_coordinates",
        "acquisition_requires_a_nonzero_period_after_initial_zero_phase_periods",
        "supplied_coefficients_time_and_form_profile_are_not_retuned_or_optimized",
        "unavailable_is_not_impossibility_and_capture_alone_is_not_nonzero_acquisition",
        "source_dataclasses_and_projection_do_not_authenticate_capture_provenance",
        "no_solver_trajectory_forcing_event_autonomous_preparation_or_physical_identity",
    )
    centered_endpoint_form_bounds: tuple[I, ...] = ()
    centered_endpoint_phase_bounds: tuple[I, ...] = ()
    endpoint_coordinate_scope: str = "exact_source_absolute_and_centered_endpoints"
    mean_scope: str = "exact_captured_conserved_weighted_means"
    nominal_weighted_form_mean: Q | None = None
    nominal_weighted_phase_mean: Q | None = None
    form_error_bounds: tuple[Q, ...] = ()
    phase_error_bounds: tuple[Q, ...] = ()
    centered_form_error_bounds: tuple[Q, ...] = ()
    centered_phase_error_bounds: tuple[Q, ...] = ()
    nominal_scaled_initial_norm_bounds: I | None = None
    scaled_initial_error_norm_upper_bound: Q | None = None
    initial_form_storage_bounds: I | None = None
    initial_phase_storage_bounds: I | None = None
    initial_storage_bounds: I | None = None
    initial_phase_edge_gap_bounds: tuple[I, ...] = ()
    initial_acute_margin_bounds: tuple[I, ...] = ()
    initial_zero_winding_certified: bool = False

    @property
    def admitted(self):
        return self.status == "admitted"

    def to_dict(self):
        from ..sdk.relational_reports import _project

        _validate_comparison_labels(self.source)
        return {
            "schema": "tnfr.relational-sine-prepared-entry.v1",
            "report": _project(self),
        }


def certify_sine_prepared_entry(
    source: SineExchangeComparison | SineRelativePattern,
    *,
    scaled_time,
    edge_turn_offsets,
) -> SinePreparedEntry:
    """Bound an analytic full-state transit, then certify nonzero-sector capture.

    An exact comparison retains the original identical-phase contract. A
    relative pattern supplies its original per-node form and phase residual
    radii, exact held capacities and arbitrary common origins. Its initial
    zero winding is certified from all raw edge intervals, without choosing
    wrap branches. All complete states in the declared set must pass.
    The supplied nonnegative time is
    ``tau=e*t`` using the reference model's effective positive form weight e.
    The shared exact exponential work limit is ``lambda*tau<=4096``; graph
    admission retains the shared 32-node/50-edge budget. Unsupported input
    domains raise. Failure of a sufficient endpoint/capture comparison returns
    unavailable without asserting nonexistence or changing the supplied time.

    For ``K_i=nu_i/d_i``, ``H=K^-1``, ``alpha=w/(beta*pi*e)`` and
    ``z=alpha*(x-weighted_mean(x))``, the scaled law is
    ``z'=-K*L*z+eta*K*S``, ``theta'=K*L*z``, with ``eta=beta*alpha**2``.
    Thus ``theta+z-v`` has derivative ``eta*K*S``, where v=z(0), and
    ``||K*S||_H<=sqrt(sum(nu_i*d_i))`` globally. Weighted contraction of
    the diffusion semigroup supplies the documented endpoint radii. This is
    a comparison estimate for the actual full sine law, not pure diffusion
    substituted for that law.

    For a pattern, weighted centering contracts the form residual norm.
    The phase endpoint also retains its initial phase and form residuals;
    common origins cancel, but the family does not have a single known
    conserved mean or a stationary trajectory merely because its nominal
    form is uniform. The source's projected relative boxes are not consumed.
    """
    preparation = _sine_preparation(source)
    admitted, geometry = preparation.admitted, preparation.geometry
    uncertain = preparation.uncertain
    form, phase = preparation.form, preparation.phase
    form_errors, phase_errors = preparation.form_errors, preparation.phase_errors
    model = preparation.model
    e, w = preparation.e, preparation.w
    weights, mobility = preparation.weights, preparation.mobility
    mean, phase_mean = preparation.mean, preparation.phase_mean
    centered_form_errors = preparation.centered_form_errors
    centered_phase_errors = preparation.centered_phase_errors
    gap, pi = preparation.gap, preparation.pi
    alpha, inverse_alpha, eta = (
        preparation.alpha,
        preparation.inverse_alpha,
        preparation.eta,
    )
    nominal_initial_norm = preparation.nominal_initial_norm
    initial_error_norm = preparation.initial_error_norm
    initial_norm, forcing = preparation.initial_norm, preparation.forcing
    scaled_nominal, scaled_initial = (
        preparation.scaled_nominal,
        preparation.scaled_initial,
    )
    if not uncertain and any(value != phase[0] for value in phase[1:]):
        raise ValueError("prepared entry requires identical exact initial phase lifts")
    tau = exact_or_represented_real(scaled_time, "scaled_time")
    if tau < 0:
        raise ValueError("scaled_time must be nonnegative")
    horizon = tau / e
    decay = I(*_negative_exp_bounds(gap * tau))
    scaled_form_radius = decay.hi * initial_norm.hi + eta.hi * forcing.hi * min(
        tau, Q(1) / gap
    )
    phase_radius = scaled_form_radius + eta.hi * tau * forcing.hi
    centered_form_bounds, centered_phase_bounds = [], []
    absolute_form_bounds, absolute_phase_bounds = [], []
    for index, value in enumerate(scaled_nominal):
        node_gain = sqrt(I(mobility[index])).hi
        form_radius = inverse_alpha.hi * node_gain * scaled_form_radius
        angle_radius = (
            node_gain * phase_radius
            + centered_phase_errors[index]
            + alpha.hi * centered_form_errors[index]
        )
        centered_form_bounds.append(I(-form_radius, form_radius))
        centered_phase_bounds.append(
            phase[index] - phase_mean + value + I(-angle_radius, angle_radius)
        )
        if not uncertain:
            absolute_form_bounds.append(I(mean - form_radius, mean + form_radius))
            absolute_phase_bounds.append(
                phase[0] + value + I(-angle_radius, angle_radius)
            )
    form_bounds = None if uncertain else tuple(absolute_form_bounds)
    phase_bounds = None if uncertain else tuple(absolute_phase_bounds)
    form_edges, phase_edges = [], []
    for left, right in geometry.edges:
        edge_gain = sqrt(I(mobility[left] + mobility[right])).hi
        form_radius = inverse_alpha.hi * edge_gain * scaled_form_radius
        angle_radius = (
            edge_gain * phase_radius
            + phase_errors[left]
            + phase_errors[right]
            + alpha.hi * (form_errors[left] + form_errors[right])
        )
        form_edges.append(I(-form_radius, form_radius))
        # Cancel the common origin and combine the exact form difference
        # before enclosing its shared mathematical-pi coefficient.
        phase_edges.append(
            phase[right]
            - phase[left]
            + alpha * (form[right] - form[left])
            + I(-angle_radius, angle_radius)
        )
    initial_form_storage = preparation.initial_form_storage_bounds
    initial_phase_storage = preparation.initial_phase_storage_bounds
    initial_phase_gaps = preparation.initial_phase_gaps
    initial_acute_margins = tuple(pi / 2 - abs(value) for value in initial_phase_gaps)
    initial_zero = all(value.lo > 0 for value in initial_acute_margins)
    mean_scope = (
        "unobserved_absolute_common_origins_each_family_member_has_its_own_conserved_means"
        if uncertain
        else "exact_initial_means_of_enclosed_trajectory_not_every_point_of_outer_node_box"
    )
    capture = _certify_sine_sector_set(
        source=source,
        geometry=geometry,
        model=model,
        capacity_bounds=tuple(I(value) for value in admitted.capacity),
        exact_held_capacity=admitted.capacity,
        form_edge_gap_bounds=tuple(form_edges),
        phase_edge_gap_bounds=tuple(phase_edges),
        edge_turn_offsets=edge_turn_offsets,
        uncertainty_scope=(
            "analytic_weighted_Duhamel_endpoint_of_original_residual_family_with_memberwise_conserved_means"
            if uncertain
            else "analytic_weighted_Duhamel_endpoint_with_correlated_norm_bounds_and_conserved_mean_leaf"
        ),
        observation_time=horizon,
        weighted_form_mean=None if uncertain else mean,
        weighted_phase_mean=None if uncertain else phase_mean,
        weighted_mean_scope=mean_scope,
    )
    unresolved = tuple(
        reason
        for condition, reason in (
            (initial_zero, "whole_initial_set_zero_winding_not_certified"),
            (capture.admitted, "analytic_endpoint_sector_capture_not_certified"),
            (any(capture.cycle_periods), "nonzero_captured_sector_required"),
        )
        if not condition
    )
    return SinePreparedEntry(
        source=source,
        reference_model=model,
        geometry=geometry,
        scaled_time=tau,
        horizon=horizon,
        coefficient_ratio=w / e,
        weighted_form_mean=None if uncertain else mean,
        common_initial_phase=None if uncertain else phase[0],
        initial_form_storage=preparation.initial_form_storage,
        initial_cycle_periods=(0,) * geometry.cycle_rank if initial_zero else None,
        mobility=mobility,
        metric_weights=weights,
        weighted_gap_lower_bound=gap,
        exponential_decay_bounds=decay,
        form_to_phase_scale_bounds=alpha,
        feedback_strength_bounds=eta,
        scaled_initial_form_bounds=scaled_initial,
        scaled_initial_norm_bounds=initial_norm,
        forcing_norm_bounds=forcing,
        scaled_form_radius_upper_bound=scaled_form_radius,
        phase_radius_upper_bound=phase_radius,
        endpoint_form_bounds=form_bounds,
        endpoint_phase_bounds=phase_bounds,
        endpoint_form_edge_gap_bounds=tuple(form_edges),
        endpoint_phase_edge_gap_bounds=tuple(phase_edges),
        capture=capture,
        hypothesis_failures=capture.hypothesis_failures,
        unresolved_conditions=unresolved,
        status=(
            "unavailable" if capture.hypothesis_failures or unresolved else "admitted"
        ),
        centered_endpoint_form_bounds=tuple(centered_form_bounds),
        centered_endpoint_phase_bounds=tuple(centered_phase_bounds),
        endpoint_coordinate_scope=(
            "centered_by_each_member_conserved_weighted_means_absolute_origins_unavailable"
            if uncertain
            else "exact_source_absolute_and_centered_endpoints"
        ),
        mean_scope=mean_scope,
        nominal_weighted_form_mean=mean,
        nominal_weighted_phase_mean=phase_mean,
        form_error_bounds=form_errors,
        phase_error_bounds=phase_errors,
        centered_form_error_bounds=centered_form_errors,
        centered_phase_error_bounds=centered_phase_errors,
        nominal_scaled_initial_norm_bounds=nominal_initial_norm,
        scaled_initial_error_norm_upper_bound=initial_error_norm,
        initial_form_storage_bounds=initial_form_storage,
        initial_phase_storage_bounds=initial_phase_storage,
        initial_storage_bounds=preparation.initial_storage_bounds,
        initial_phase_edge_gap_bounds=tuple(initial_phase_gaps),
        initial_acute_margin_bounds=initial_acute_margins,
        initial_zero_winding_certified=initial_zero,
    )
