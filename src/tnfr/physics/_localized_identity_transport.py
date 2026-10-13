"""Detached transport and source budgets for a full-state sine-law packet.

Proofs: LOCALIZED_IDENTITY_TRANSPORT.md#localized-transport-error and
LOCALIZED_TRANSPORT_SOURCES.md. Prepared transport and phase-flat source
exclusion have separate premises; neither report supplies an actual source.
This arithmetic evaluates no trajectory, source, observed response or native
runtime. A failed sufficient bound does not disprove finite transport.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._rational_interval import INTERVAL_METHOD, I, pi_interval, sqrt
from ._sine_admission import _sine_model_coefficients

_IDENTITY_TOLERANCE = Q(1, 25)
_TAIL_TOLERANCE = Q(1, 25)


def _admit_cycle_window(node_count, fejer_order, horizon):
    """Admit the shared finite cycle, disjoint carrier bands and clock window."""
    for name, value in (
        ("node_count", node_count),
        ("fejer_order", fejer_order),
        ("horizon", horizon),
    ):
        if type(value) is not int:
            raise TypeError(f"{name} must be an ordinary Python integer")
    if node_count < 8 or node_count % 4:
        raise ValueError("node_count must be at least eight and divisible by four")
    if fejer_order < 2 or 4 * (fejer_order - 1) >= node_count:
        raise ValueError("fejer_order requires 2 <= Q and Q-1 < N/4")
    if not 0 < horizon <= node_count // 2:
        raise ValueError("horizon requires 0 < T <= N/2")


def _admit_cycle_law(reference_model):
    """Re-admit the common positive-loss complete law without renormalization."""
    coefficients = _sine_model_coefficients(reference_model, positive_loss=True)
    if coefficients[2] != 1:
        raise ValueError("localized transport requires unit storage_scale")
    return coefficients


def _cycle_clock_bounds(exchange, horizon, pi):
    """Divide admitted exact positive rates before outward grid projection."""
    # A subgrid exchange has a zero lower enclosure endpoint, not a zero clock.
    return (
        I(exchange / pi.hi, exchange / pi.lo),
        I(horizon * pi.lo / exchange, horizon * pi.hi / exchange),
    )


def _cycle_cubic_error_upper_bound(state_norm_upper_bound, horizon):
    """Integrate the cycle sine defect inside an independently closed norm ball.

    Callers admit exact nonnegative radius and horizon, both complete rows,
    beta one and the contractive tangent before applying this arithmetic.
    The result alone does not certify continued membership in the norm ball.
    """
    return Q(4, 3) * horizon * state_norm_upper_bound**3


@dataclass(frozen=True)
class _LocalizedIdentityTransportBound:
    """Conditional whole-window comparison with a prescribed moving reference.

    The joint Euclidean state is ``z=x+i*theta`` relative to the uniform zero
    background, with real phase lifts. Its modulus is a localization diagnostic,
    not physical energy. The Fejer carrier has nominal norm ``amplitude`` and
    wave index N/4. Its reference center is ``-tau`` and its explicit internal
    rotation remains part of the predicted full state, not a phase-origin gauge.

    The initial error bounds every form and phase coordinate jointly, including
    common means. Positive loss, fixed unit capacity, unit cycle conductances
    and unit geometric edge lengths are supplied. No preparation, formation or
    physical interpretation is authenticated by this report.

    The error fields are candidate bootstrap majorants; they bound actual
    full-state error only when ``bootstrap_certified`` is true. Actual norm
    and tail conclusions remain unavailable when that proof does not close.
    """

    node_count: int
    fejer_order: int
    amplitude: Q
    initial_state_error_bound: Q
    localization_radius: Q
    horizon: int
    form_loss_coefficient: Q
    exchange_coefficient: Q
    storage_scale: Q
    carrier_mode: int
    fejer_coefficient_square_sum: Q
    clock_rate_bounds: I
    original_horizon_bounds: I
    relative_initial_error: Q
    relative_damping_error_upper_bound: Q
    relative_dispersion_error_upper_bound: Q
    relative_nonlinear_error_upper_bound: Q
    relative_identity_error_upper_bound: Q
    identity_error_upper_bound: Q
    bootstrap_margin: Q
    reference_tail_fraction_upper_bound: Q
    actual_tail_fraction_upper_bound: Q | None
    state_norm_lower_bound: Q | None
    state_norm_upper_bound: Q | None
    endpoint_displacement: int
    endpoint_window_separation_margin: Q
    bootstrap_certified: bool
    identity_tolerance_met: bool
    tail_tolerance_met: bool
    endpoint_windows_disjoint: bool
    unavailable_reasons: tuple[str, ...]
    status: str
    identity_tolerance: Q = _IDENTITY_TOLERANCE
    tail_tolerance: Q = _TAIL_TOLERANCE
    interval_method: str = INTERVAL_METHOD
    conditional_on_supplied_preparation: bool = True
    scope: tuple[str, ...] = (
        "complete_normalized_sine_rows_with_authoritative_model_coefficients",
        "fixed_simple_cycle_with_unit_capacity_conductance_and_geometric_length",
        "positive_form_loss_unit_storage_scale_and_no_inputs_or_events",
        "tau_equals_exchange_times_original_time_over_pi_in_both_rows",
        "nominal_Fejer_packet_and_directional_carrier_are_prepared_not_formed",
        "arbitrary_joint_initial_error_retains_all_fine_and_common_mean_components",
        "identity_compares_the_complete_signed_form_and_real_phase_lift_reference",
        "reference_internal_rotation_is_not_a_nonlinear_symmetry_or_phase_gauge",
        "fractional_center_is_an_observation_not_an_exact_graph_translation",
        "integer_cycle_translations_preserve_support_and_complete_dynamics",
        "tail_and_signal_bounds_hold_throughout_the_declared_finite_interval",
        "localization_diagnostic_is_not_storage_or_a_physical_probability",
        "not_certified_means_sufficient_bounds_failed_not_transport_impossible",
        "no_source_authentication_trajectory_response_or_physical_identification",
    )


def _bound_localized_identity_transport(
    *,
    reference_model,
    node_count,
    fejer_order,
    amplitude,
    initial_state_error_bound,
    localization_radius,
    horizon,
) -> _LocalizedIdentityTransportBound:
    """Bound a finite prepared transport claim without evaluating a response.

    N, Q and T are ordinary Python integers: N is divisible by four, Q>=2,
    Q-1<N/4, and 0<T<=N/2. The exact or represented real primitives require
    A>0, eps>=0 and 1<R<N/2. Unit storage and positive loss/exchange are admitted
    from the supplied regular reference model without renormalizing its mix.

    The packet has Fourier coefficients ``1-|m|/Q`` for ``|m|<Q`` around mode
    N/4, normalized to joint norm A. The moving reference retains carrier
    rotation and has center -tau. The returned relative error is
    eps/A + 2*pi*(e/w)*T + T*(2*pi*(Q-1)/N)**3/6 + (32/3)*T*A**2.
    A strict bound below one closes the 2A nonlinear bootstrap. Otherwise
    actual-state norm/tail conclusions are unavailable; no bound is clipped.

    Certification additionally requires the fixed protocol's strict 1/25
    identity and tail tolerances and separated endpoint windows (2R<T).
    These are sufficient finite observation conditions, not selected laws.
    """
    _admit_cycle_window(node_count, fejer_order, horizon)
    amplitude, error, radius = (
        _exact(value, name)
        for name, value in (
            ("amplitude", amplitude),
            ("initial_state_error_bound", initial_state_error_bound),
            ("localization_radius", localization_radius),
        )
    )
    if amplitude <= 0 or error < 0:
        raise ValueError("require positive amplitude and nonnegative initial error")
    if not 1 < radius < Q(node_count, 2):
        raise ValueError("localization_radius requires 1 < R < N/2")
    loss, exchange, beta = _admit_cycle_law(reference_model)

    pi = pi_interval()
    initial = error / amplitude
    damping = 2 * pi.hi * loss * horizon / exchange
    dispersion = Q(horizon, 6) * (2 * pi.hi * Q(fejer_order - 1, node_count)) ** 3
    nonlinear = _cycle_cubic_error_upper_bound(2 * amplitude, horizon) / amplitude
    relative_error = initial + damping + dispersion + nonlinear
    identity_error = amplitude * relative_error
    bootstrap_margin = 1 - relative_error
    bootstrap = bootstrap_margin > 0
    reference_tail = Q(node_count**3, 16 * fejer_order**3) / (radius - 1) ** 3
    actual_tail = (
        ((sqrt(I(reference_tail)).hi + relative_error) / bootstrap_margin) ** 2
        if bootstrap
        else None
    )
    identity_met = bootstrap and relative_error < _IDENTITY_TOLERANCE
    tail_met = actual_tail is not None and actual_tail < _TAIL_TOLERANCE
    separation = horizon - 2 * radius
    disjoint = separation > 0
    reasons = []
    if not bootstrap:
        reasons.append("nonlinear_bootstrap_not_closed")
    if not identity_met:
        reasons.append("identity_tolerance_not_certified")
    if not tail_met:
        reasons.append("tail_tolerance_not_certified")
    if not disjoint:
        reasons.append("endpoint_windows_not_separated")
    clock_rate, original_horizon = _cycle_clock_bounds(exchange, horizon, pi)
    return _LocalizedIdentityTransportBound(
        node_count=node_count,
        fejer_order=fejer_order,
        amplitude=amplitude,
        initial_state_error_bound=error,
        localization_radius=radius,
        horizon=horizon,
        form_loss_coefficient=loss,
        exchange_coefficient=exchange,
        storage_scale=beta,
        carrier_mode=node_count // 4,
        fejer_coefficient_square_sum=Q(2 * fejer_order**2 + 1, 3 * fejer_order),
        clock_rate_bounds=clock_rate,
        original_horizon_bounds=original_horizon,
        relative_initial_error=initial,
        relative_damping_error_upper_bound=damping,
        relative_dispersion_error_upper_bound=dispersion,
        relative_nonlinear_error_upper_bound=nonlinear,
        relative_identity_error_upper_bound=relative_error,
        identity_error_upper_bound=identity_error,
        bootstrap_margin=bootstrap_margin,
        reference_tail_fraction_upper_bound=reference_tail,
        actual_tail_fraction_upper_bound=actual_tail,
        state_norm_lower_bound=amplitude - identity_error if bootstrap else None,
        state_norm_upper_bound=amplitude + identity_error if bootstrap else None,
        endpoint_displacement=-horizon,
        endpoint_window_separation_margin=separation,
        bootstrap_certified=bootstrap,
        identity_tolerance_met=identity_met,
        tail_tolerance_met=tail_met,
        endpoint_windows_disjoint=disjoint,
        unavailable_reasons=tuple(reasons),
        status="certified" if not reasons else "not_certified",
    )


@dataclass(frozen=True)
class _PhaseFlatDirectionalSourceBound:
    """Finite exclusion from a norm-bounded phase-flat source class.

    The source is within the supplied joint error of (u,0), where u is any
    real form vector with norm at most M. It need not be reflection symmetric,
    localized or mean zero. The target class consists of norm-a joint states
    z=x+i*theta supported in either of the two opposite carrier bands. This is
    a necessary property of the prepared packet, not a new formation target.

    Delta is a candidate nonlinear bootstrap bound. Actual distance and its
    exclusion margin are unavailable unless delta<M. A failed sufficient
    bound supplies neither a formation claim nor an all-future obstruction.
    """

    node_count: int
    fejer_order: int
    source_form_norm_bound: Q
    target_amplitude: Q
    initial_state_error_bound: Q
    horizon: int
    form_loss_coefficient: Q
    exchange_coefficient: Q
    storage_scale: Q
    carrier_mode: int
    clock_rate_bounds: I
    original_horizon_bounds: I
    nonlinear_error_upper_bound: Q
    tangent_error_upper_bound: Q
    bootstrap_margin: Q
    bootstrap_certified: bool
    one_over_sqrt_two_bounds: I
    tangent_target_distance_lower_bound: Q
    target_distance_lower_bound: Q | None
    target_tube_radius: Q
    exclusion_margin: Q | None
    unavailable_reasons: tuple[str, ...]
    status: str
    identity_tolerance: Q = _IDENTITY_TOLERANCE
    interval_method: str = INTERVAL_METHOD
    conditional_on_supplied_source_class: bool = True
    scope: tuple[str, ...] = (
        "same_complete_positive_loss_sine_law_unit_cycle_and_held_unit_capacities",
        "tau_equals_exchange_times_original_time_over_pi_in_both_rows",
        "source_class_is_arbitrary_real_form_with_zero_nominal_phase_lift",
        "joint_initial_error_retains_every_form_phase_and_common_mean_component",
        "no_reflection_symmetry_spatial_profile_or_target_reversed_source_assumed",
        "actual_damped_tangent_preserves_equal_opposite_carrier_band_norms",
        "target_class_has_fixed_norm_and_only_one_of_two_disjoint_carrier_bands",
        "target_rotations_translations_and_either_direction_are_covered_by_the_band_bound",
        "strict_exclusion_applies_throughout_the_declared_finite_window",
        "not_certified_is_failed_sufficiency_not_positive_formation_evidence",
        "no_obstruction_to_all_mobile_patterns_or_arbitrarily_late_nonlinear_bias",
        "no_source_authentication_trajectory_response_or_physical_identification",
    )


def _bound_phase_flat_directional_source(
    *,
    reference_model,
    node_count,
    fejer_order,
    source_form_norm_bound,
    target_amplitude,
    initial_state_error_bound,
    horizon,
) -> _PhaseFlatDirectionalSourceBound:
    """Bound distance from a source class to either one-sided carrier band.

    Require positive exact or represented M and a, nonnegative error eps,
    and the same finite cycle, band, time and model domains as prepared
    transport. No initial state or target profile is calculated. The nominal
    source is (u,0) for arbitrary real ||u||<=M; its actual positive-loss tangent
    retains equal opposite-band norms, hence distance at least a/sqrt(2) from
    every norm-a target in either band. This does not require spatial symmetry.

    Delta=eps+(32/3)*T*M**3 closes the full nonlinear 2M bootstrap only when
    delta<M. Then a/sqrt(2)-delta bounds the actual distance at every time in
    [0,T]. Exclusion requires this lower bound strictly above a/25.
    """
    _admit_cycle_window(node_count, fejer_order, horizon)
    source_norm, amplitude, error = (
        _exact(value, name)
        for name, value in (
            ("source_form_norm_bound", source_form_norm_bound),
            ("target_amplitude", target_amplitude),
            ("initial_state_error_bound", initial_state_error_bound),
        )
    )
    if source_norm <= 0 or error < 0 or amplitude <= 0:
        raise ValueError("require positive source/target norms and nonnegative error")
    loss, exchange, beta = _admit_cycle_law(reference_model)
    pi = pi_interval()
    clock_rate, original_horizon = _cycle_clock_bounds(exchange, horizon, pi)
    nonlinear = _cycle_cubic_error_upper_bound(2 * source_norm, horizon)
    delta = error + nonlinear
    margin = source_norm - delta
    bootstrap = margin > 0
    reciprocal_root = sqrt(I(Q(1, 2)))
    tangent_distance = amplitude * reciprocal_root.lo
    target_distance = tangent_distance - delta if bootstrap else None
    tube = amplitude * _IDENTITY_TOLERANCE
    exclusion_margin = target_distance - tube if target_distance is not None else None
    reasons = []
    if not bootstrap:
        reasons.append("nonlinear_bootstrap_not_closed")
    elif exclusion_margin <= 0:
        reasons.append("target_tube_exclusion_not_certified")
    return _PhaseFlatDirectionalSourceBound(
        node_count=node_count,
        fejer_order=fejer_order,
        source_form_norm_bound=source_norm,
        target_amplitude=amplitude,
        initial_state_error_bound=error,
        horizon=horizon,
        form_loss_coefficient=loss,
        exchange_coefficient=exchange,
        storage_scale=beta,
        carrier_mode=node_count // 4,
        clock_rate_bounds=clock_rate,
        original_horizon_bounds=original_horizon,
        nonlinear_error_upper_bound=nonlinear,
        tangent_error_upper_bound=delta,
        bootstrap_margin=margin,
        bootstrap_certified=bootstrap,
        one_over_sqrt_two_bounds=reciprocal_root,
        tangent_target_distance_lower_bound=tangent_distance,
        target_distance_lower_bound=target_distance,
        target_tube_radius=tube,
        exclusion_margin=exclusion_margin,
        unavailable_reasons=tuple(reasons),
        status="excluded" if not reasons else "not_certified",
    )


@dataclass(frozen=True)
class _CycleConsensusBasinBound:
    """Conditional all-time capture from a supplied small real-lift state.

    The primitive norm bounds both signed form and retained real phase lifts
    about a supplied constant background. It is not merely a bound on storage
    or on wrapped edge gaps: either of those can also describe nonzero winding.
    Actual conserved form and lifted phase means, including source residuals,
    determine the eventual constant state. No zero-mean preparation is assumed.

    This report supplies no convergence deadline, mobile template certificate,
    finite-lived formation exclusion, actual source or physical identification.
    Failed sufficiency leaves the consensus basin unavailable.
    """

    node_count: int
    initial_state_norm_upper_bound: Q
    form_loss_coefficient: Q
    exchange_coefficient: Q
    storage_scale: Q
    initial_storage_upper_bound: Q
    acute_boundary_storage_lower_bound: Q
    acute_barrier_margin: Q
    edge_cosine_lower_bound: Q | None
    consensus_basin_certified: bool
    unavailable_reasons: tuple[str, ...]
    status: str
    conditional_on_supplied_preparation: bool = True
    scope: tuple[str, ...] = (
        "complete_positive_loss_normalized_sine_rows_and_no_inputs_or_events",
        "fixed_simple_unit_cycle_with_held_unit_capacity_and_unit_storage_scale",
        "initial_joint_norm_uses_signed_form_and_real_phase_lifts_about_constants",
        "supplied_initial_lift_is_not_replaced_by_wrapped_edge_storage",
        "strict_storage_barrier_retains_the_zero_winding_acute_cell_for_all_time",
        "consensus_is_on_each_actual_conserved_form_and_lifted_phase_mean_leaf",
        "tau_equals_exchange_times_original_time_over_pi_in_both_rows",
        "no_uniform_convergence_deadline_or_finite_packet_template_transfer",
        "no_exclusion_of_finite_lived_transient_organizations",
        "not_certified_is_failed_sufficiency_not_observed_nonformation",
        "no_source_response_trajectory_or_physical_identification",
    )


def _bound_cycle_consensus_basin(
    *, reference_model, node_count, initial_state_norm_upper_bound
) -> _CycleConsensusBasinBound:
    """Apply the small-state zero-sector corollary without evaluating a flow.

    Proof: LOCALIZED_ORGANIZATION_GEOMETRY.md. Require an ordinary integer N>=3,
    the admitted positive-loss sine law with beta=1, and an exact or represented
    nonnegative joint norm M about a supplied constant form/real-phase background.
    The cycle has held unit capacities, conductances and geometric lengths.

    Storage satisfies H<=2*M**2 initially and is nonincreasing under both
    complete rows. If 2*M**2<1, then 2*M<sqrt(2)<pi/2 also admits the initial
    zero-winding acute chart. Every boundary face costs at least one. Its
    forward-invariant storage sublevel has edge cosines at least 1-2*M**2
    and contains only the consensus critical geometry. The existing full-law
    sector theorem gives convergence on each actual conserved-mean leaf.

    Only sufficient scalar arithmetic is performed. No phase state is read,
    so the caller must retain the real-lift norm premise and support model;
    a small storage observation alone cannot replace that premise.
    """
    if type(node_count) is not int:
        raise TypeError("node_count must be an ordinary Python integer")
    if node_count < 3:
        raise ValueError("node_count must be at least three for a simple cycle")
    radius = _exact(initial_state_norm_upper_bound, "initial_state_norm_upper_bound")
    if radius < 0:
        raise ValueError("initial_state_norm_upper_bound must be nonnegative")
    loss, exchange, beta = _admit_cycle_law(reference_model)
    storage = 2 * radius**2
    margin = 1 - storage
    certified = margin > 0
    return _CycleConsensusBasinBound(
        node_count=node_count,
        initial_state_norm_upper_bound=radius,
        form_loss_coefficient=loss,
        exchange_coefficient=exchange,
        storage_scale=beta,
        initial_storage_upper_bound=storage,
        acute_boundary_storage_lower_bound=Q(1),
        acute_barrier_margin=margin,
        edge_cosine_lower_bound=margin if certified else None,
        consensus_basin_certified=certified,
        unavailable_reasons=(
            () if certified else ("acute_storage_barrier_not_certified",)
        ),
        status="certified" if certified else "not_certified",
    )
