"""Finite slow-time comparison with the complete, prepared sine dynamics.

This reader bounds a mathematical reference flow; it neither runs that flow
nor replaces full-state execution. Every uncertain member has its own reference
initialized with its retained form. The scalar decay bounds a matrix-semigroup
norm on the weighted-mean-zero subspace, not a vector-valued initial-layer
approximation or decay of the common mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real
from ..dynamics.relational import RelationalExchangeModel
from ..mathematics._rational_interval import INTERVAL_METHOD, I, sqrt
from ._sine_preparation import _sine_preparation
from .phase_cycle_geometry import PhaseCycleGeometry, PhaseCycleState
from .relational_sine_comparison import (
    SineExchangeComparison,
    _validate_comparison_labels,
)
from .relational_sine_pattern import SineRelativePattern
from .relational_sine_recovery import (
    SineSectorCapture,
    _certify_sine_sector_set,
    _target_admission,
)
from .reversible_eigenmode_reference import _MAX_RATIONAL_EXPONENT, _negative_exp_bounds

__all__ = (
    "SineSlowPhaseBound",
    "SineSlowCapture",
    "bound_sine_slow_phase",
    "certify_sine_slow_capture",
)


def _decay_over_interval(exponent: I) -> I:
    """Enclose decay, using a monotone tail beyond the exact work budget."""
    cutoff = Q(_MAX_RATIONAL_EXPONENT)
    lower = Q(0) if exponent.hi > cutoff else _negative_exp_bounds(exponent.hi)[0]
    upper = _negative_exp_bounds(min(exponent.lo, cutoff))[1]
    return I(lower, upper)


@dataclass(frozen=True)
class SineSlowPhaseBound:
    """Pointwise and whole-horizon error with the initial transient retained.

    ``composite_phase_error_upper_bound`` compares theta to
    psi(sigma)-exp(-A*tau)z(0). ``phase_error_upper_bound`` compares theta to
    psi(sigma) without that transient. ``joint_error_upper_bound`` compares
    theta+z to psi. None supplies a numerical reference trajectory or a basin
    certificate. Absolute reference initial coordinates are unavailable for
    relative sources; their centered intervals are correlated outer bounds.
    """

    source: SineExchangeComparison | SineRelativePattern
    reference_model: RelationalExchangeModel
    geometry: PhaseCycleGeometry
    slow_time: Q
    scaled_time_bounds: I
    horizon_bounds: I
    coefficient_ratio: Q
    mobility: tuple[Q, ...]
    metric_weights: tuple[Q, ...]
    weighted_gap_lower_bound: Q
    phase_lipschitz_bound: Q
    forcing_norm_bounds: I
    form_to_phase_scale_bounds: I
    feedback_strength_bounds: I
    scaled_initial_norm_bounds: I
    initial_form_norm_upper_bound: Q
    initial_storage_bounds: I
    exponential_decay_bounds: I
    exponential_growth_bounds: I
    decay_tail_enclosure_used: bool
    composite_phase_error_upper_bound: Q
    uniform_composite_phase_error_upper_bound: Q
    phase_error_upper_bound: Q
    joint_error_upper_bound: Q
    scaled_form_remainder_upper_bound: Q
    scaled_form_norm_upper_bound: Q
    form_remainder_upper_bound: Q
    form_norm_upper_bound: Q
    phase_potential_error_upper_bound: Q
    phase_edge_error_upper_bounds: tuple[Q, ...]
    reference_initial_phase_bounds: tuple[I, ...] | None
    centered_reference_initial_phase_bounds: tuple[I, ...]
    weighted_form_mean: Q | None
    weighted_phase_mean: Q | None
    reference_scope: str
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "complete_positive_loss_sine_law_fixed_support_and_positive_held_capacity",
        "slow_clock_sigma_equals_eta_times_tau_tau_equals_e_times_structural_time",
        "reference_dpsi_dsigma_equals_KS_psi_with_initial_theta_plus_scaled_centered_form",
        "each_source_member_has_its_own_reference_and_conserved_mean_leaf",
        "global_lift_comparison_allows_nonacute_phases_without_changing_the_law",
        "composite_corrector_is_full_matrix_exp_minus_A_tau_times_initial_z",
        "scalar_decay_bounds_the_corrector_norm_not_its_vector_components",
        "uniform_composite_bound_covers_zero_through_declared_slow_horizon",
        "pointwise_phase_error_retains_the_initial_fast_transient",
        "original_form_remainder_retains_inverse_scaling_and_preparation_budget",
        "exact_rational_growth_exponent_at_most_4096_reciprocal_before_dyadic_rounding",
        "monotone_decay_tail_bounds_do_not_shorten_the_declared_horizon",
        "dyadic_enclosure_floor_can_limit_small_parameter_numerical_sharpness",
        "no_reference_solver_capture_basin_event_or_physical_identification",
        "source_dataclasses_and_projection_do_not_authenticate_provenance",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        _validate_comparison_labels(self.source)
        return {
            "schema": "tnfr.relational-sine-slow-phase.v1",
            "report": _project(self),
        }

    def certify_capture(self, *, target_phase_turns):
        """Recompute the comparison before a critical-reference capture check."""
        return certify_sine_slow_capture(
            self.source, slow_time=self.slow_time, target_phase_turns=target_phase_turns
        )


def bound_sine_slow_phase(source, *, slow_time) -> SineSlowPhaseBound:
    """Bound full-law deviation from its initially shifted phase reference.

    Supply an exact comparison or the original residual preparation family,
    and a finite nonnegative slow time sigma. Mathematical pi makes the fast
    and structural horizons generally irrational; returned intervals enclose
    them without replacing them by midpoints. The supplied law must have
    positive loss, exchange, storage scale and exact held capacities. Shared
    geometry admission retains its 32-node/50-edge budget. Unsupported domains
    and growth exponents above 4096 raise; a large error bound is still a bound,
    not an assertion that the approximation is accurate or that capture occurs.
    """
    return _slow_phase_preparation(source, slow_time=slow_time)[1]


def _slow_phase_preparation(source, *, slow_time):
    """Retain the shared preparation for consumers of the analytic bound."""
    sigma = exact_or_represented_real(slow_time, "slow_time")
    if sigma < 0:
        raise ValueError("slow_time must be nonnegative")
    p = _sine_preparation(source)
    ell = 2 * max(p.admitted.capacity)
    # Invert raw positive Fractions, never a rounded interval containing zero.
    growth_lower, growth_upper = _negative_exp_bounds(ell * sigma)
    growth = I(1 / growth_upper, 1 / growth_lower)
    # eta may have an outward lower endpoint zero. These exact coefficient
    # formulas retain the positive law without dividing by that enclosure.
    tau = (sigma * p.beta * p.e**2 / p.w**2) * p.pi**2
    horizon = (sigma * p.beta * p.e / p.w**2) * p.pi**2
    exponent = p.gap * tau
    decay = _decay_over_interval(exponent)
    z0, forcing, eta = p.initial_norm.hi, p.forcing.hi, p.eta.hi
    composite = (
        eta * (ell * z0 + forcing) * (growth.hi - decay.lo) / (p.gap + p.eta.lo * ell)
    )
    # This coarser all-time envelope proves fixed-slow-horizon O(eta) even
    # through the initial layer. The pointwise bound vanishes at sigma=0.
    uniform_composite = eta * (ell * z0 + forcing) * growth.hi / p.gap
    kernel_time = min(tau.hi, (1 - decay.lo) / p.gap)
    scaled_remainder = eta * forcing * kernel_time
    form_remainder = ((p.w / p.e) / p.pi).hi * forcing * kernel_time
    norm = sqrt(I(sum((m * x**2 for m, x in zip(p.weights, p.centered)), Q(0))))
    error = sqrt(I(sum((m * r**2 for m, r in zip(p.weights, p.form_errors)), Q(0))))
    form_norm = norm.hi + error.hi
    phase_error = decay.hi * z0 + composite
    centered_reference = tuple(
        phase - p.phase_mean + scaled + I(-radius, radius)
        for phase, scaled, radius in zip(
            p.phase,
            p.scaled_nominal,
            (
                t + p.alpha.hi * x
                for t, x in zip(p.centered_phase_errors, p.centered_form_errors)
            ),
        )
    )
    return p, SineSlowPhaseBound(
        source=source,
        reference_model=p.model,
        geometry=p.geometry,
        slow_time=sigma,
        scaled_time_bounds=tau,
        horizon_bounds=horizon,
        coefficient_ratio=p.w / p.e,
        mobility=p.mobility,
        metric_weights=p.weights,
        weighted_gap_lower_bound=p.gap,
        phase_lipschitz_bound=ell,
        forcing_norm_bounds=p.forcing,
        form_to_phase_scale_bounds=p.alpha,
        feedback_strength_bounds=p.eta,
        scaled_initial_norm_bounds=p.initial_norm,
        initial_form_norm_upper_bound=form_norm,
        initial_storage_bounds=p.initial_storage_bounds,
        exponential_decay_bounds=decay,
        exponential_growth_bounds=growth,
        decay_tail_enclosure_used=exponent.hi > _MAX_RATIONAL_EXPONENT,
        composite_phase_error_upper_bound=composite,
        uniform_composite_phase_error_upper_bound=uniform_composite,
        phase_error_upper_bound=phase_error,
        joint_error_upper_bound=composite + scaled_remainder,
        scaled_form_remainder_upper_bound=scaled_remainder,
        scaled_form_norm_upper_bound=decay.hi * z0 + scaled_remainder,
        form_remainder_upper_bound=form_remainder,
        form_norm_upper_bound=decay.hi * form_norm + form_remainder,
        phase_potential_error_upper_bound=min(
            Q(2 * len(p.geometry.edges)), forcing * phase_error
        ),
        phase_edge_error_upper_bounds=tuple(
            sqrt(I(p.mobility[i] + p.mobility[j])).hi * phase_error
            for i, j in p.geometry.edges
        ),
        reference_initial_phase_bounds=(
            None
            if p.uncertain
            else tuple(t + z for t, z in zip(p.phase, p.scaled_nominal))
        ),
        centered_reference_initial_phase_bounds=centered_reference,
        weighted_form_mean=None if p.uncertain else p.mean,
        weighted_phase_mean=None if p.uncertain else p.phase_mean,
        reference_scope=(
            "memberwise_references_with_unobserved_origins_not_one_nominal_trajectory"
            if p.uncertain
            else "exact_preparation_reference_with_retained_initial_form"
        ),
    )


@dataclass(frozen=True)
class SineSlowCapture:
    """Full-state capture from a proved stationary-reference neighborhood.

    The target establishes a reference enclosure by exact sine criticality
    and flow dependence. It is never assigned to a live state. The capture
    set retains correlated phase/form norm bounds as well as outer edge boxes;
    not every corner of those boxes satisfies the retained storage bounds.
    """

    source: SineExchangeComparison | SineRelativePattern
    slow_phase: SineSlowPhaseBound
    target_geometry: PhaseCycleState
    target_phase_turns: tuple[Q, ...]
    target_centered_phase_bounds: tuple[I, ...]
    target_edge_turns: tuple[Q, ...]
    target_phase_storage_bounds: I
    nominal_initial_reference_distance_bounds: I
    initial_reference_uncertainty_upper_bound: Q
    initial_reference_distance_upper_bound: Q
    reference_distance_upper_bound: Q
    actual_phase_distance_upper_bound: Q
    correlated_form_storage_upper_bound: Q
    correlated_phase_storage_upper_bound: Q
    capture: SineSectorCapture
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "complete_preparation_and_slow_bound_recomputed_from_primitives",
        "exact_rational_turn_target_revalidated_by_shared_acute_sine_cancellation",
        "stationary_reference_neighborhood_from_initial_mismatch_and_global_Lipschitz_bound",
        "each_actual_member_uses_its_own_mean_matched_target_and_reference",
        "actual_phase_bound_includes_fast_transient_not_only_composite_error",
        "critical_target_cancels_linear_phase_storage_before_quadratic_bound",
        "full_original_form_storage_is_included_in_the_capture_budget",
        "endpoint_set_intersects_outer_edge_boxes_with_proved_correlated_storage_bounds",
        "integer_edge_offsets_derived_from_supplied_target_lifts_without_source_unwrapping",
        "shared_all_face_barrier_proves_same_sector_retention_and_convergence",
        "analytic_endpoint_clock_is_the_outer_slow_report_horizon_bounds_not_an_observation",
        "target_is_a_proof_reference_not_a_dynamically_selected_or_injected_state",
        "unavailable_sufficient_bounds_are_not_instability_or_impossibility",
        "capture_is_not_automatically_acquisition_initial_winding_is_not_assumed",
        "no_supplied_response_endpoint_new_solver_law_event_or_physical_identity",
    )

    @property
    def admitted(self):
        return self.capture.admitted

    @property
    def status(self):
        return self.capture.status

    def to_dict(self):
        from ..sdk.relational_reports import _project

        _validate_comparison_labels(self.source)
        return {
            "schema": "tnfr.relational-sine-slow-capture.v1",
            "report": _project(self),
        }


def certify_sine_slow_capture(
    source, *, slow_time, target_phase_turns
) -> SineSlowCapture:
    """Certify full capture using an independently proved phase reference set.

    Exact target turns are supplied in source node order. The shared target
    owner proves strictly acute sine criticality. The reference flow's initial
    distance from this stationary geometry, with the original residual errors,
    bounds its later distance without a reference solver or supplied endpoint.
    The full-law comparison adds the actual phase transient and original form
    storage. Strict acute and complete boundary margins must both pass.

    Source/law/clock/target malformation raises. Conservative failed margins
    return unavailable. The slow report retains generally irrational endpoint
    time bounds; the nested capture is neither a timestamped observation nor
    a sampled forecast. Report fields are not trusted as incoming evidence.
    """
    p, slow = _slow_phase_preparation(source, slow_time=slow_time)
    target, _, _, _, _, fields = _target_admission(
        p.admitted, cycle=None, winding=None, target_phase_turns=target_phase_turns
    )
    target_mean = sum((m * q for m, q in zip(p.normalized_weights, target)), Q(0))
    centered_target = tuple(2 * (q - target_mean) * p.pi for q in target)
    mismatch = tuple(
        phase - p.phase_mean + z - reference
        for phase, z, reference in zip(p.phase, p.scaled_nominal, centered_target)
    )
    nominal_distance = sqrt(
        sum((m * value**2 for m, value in zip(p.weights, mismatch)), I(0))
    )
    phase_uncertainty = sqrt(
        I(sum((m * r**2 for m, r in zip(p.weights, p.phase_errors)), Q(0)))
    ).hi
    uncertainty = phase_uncertainty + p.initial_error_norm
    initial_distance = nominal_distance.hi + uncertainty
    reference_distance = slow.exponential_growth_bounds.hi * initial_distance
    distance = reference_distance + slow.phase_error_upper_bound
    curvature = slow.phase_lipschitz_bound
    form_storage = curvature * slow.form_norm_upper_bound**2 / 2
    target_storage = fields["target_phase_storage_bounds"]
    phase_storage = target_storage.hi + curvature * distance**2 / 2
    offsets, turns, phase_gaps, form_gaps = [], [], [], []
    for i, j in p.geometry.edges:
        raw = target[j] - target[i]
        principal = (raw + Q(1, 2)) % 1 - Q(1, 2)
        offset = principal - raw
        if offset.denominator != 1:
            raise ArithmeticError("target phase offsets must be integers")
        offsets.append(int(offset))
        turns.append(principal)
        gain = sqrt(I(p.mobility[i] + p.mobility[j])).hi
        phase_radius, form_radius = gain * distance, gain * slow.form_norm_upper_bound
        phase_gaps.append(2 * raw * p.pi + I(-phase_radius, phase_radius))
        form_gaps.append(I(-form_radius, form_radius))
    capture = _certify_sine_sector_set(
        source=source,
        geometry=p.geometry,
        model=p.model,
        capacity_bounds=tuple(I(v) for v in p.admitted.capacity),
        exact_held_capacity=p.admitted.capacity,
        form_edge_gap_bounds=tuple(form_gaps),
        phase_edge_gap_bounds=tuple(phase_gaps),
        edge_turn_offsets=tuple(offsets),
        uncertainty_scope="analytic_slow_handoff_endpoint_with_memberwise_reference_and_correlated_critical_storage_bounds",
        weighted_form_mean=None if p.uncertain else p.mean,
        weighted_phase_mean=None if p.uncertain else p.phase_mean,
        weighted_mean_scope=(
            "unobserved_absolute_origins_each_member_has_its_own_conserved_means"
            if p.uncertain
            else "actual_analytic_endpoint_retains_exact_initial_means"
        ),
        correlated_form_storage_upper_bound=form_storage,
        correlated_phase_storage_upper_bound=phase_storage,
    )
    return SineSlowCapture(
        source=source,
        slow_phase=slow,
        target_geometry=fields["target_geometry"],
        target_phase_turns=target,
        target_centered_phase_bounds=centered_target,
        target_edge_turns=tuple(turns),
        target_phase_storage_bounds=target_storage,
        nominal_initial_reference_distance_bounds=nominal_distance,
        initial_reference_uncertainty_upper_bound=uncertainty,
        initial_reference_distance_upper_bound=initial_distance,
        reference_distance_upper_bound=reference_distance,
        actual_phase_distance_upper_bound=distance,
        correlated_form_storage_upper_bound=form_storage,
        correlated_phase_storage_upper_bound=phase_storage,
        capture=capture,
    )
