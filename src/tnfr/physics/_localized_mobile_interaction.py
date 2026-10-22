"""Detached identity and cubic-error budgets for two supplied moving packets.

The complete-law proof is owned by LOCALIZED_MOBILE_INTERACTION.md. This
calculator evaluates neither a trajectory nor the cubic Duhamel predictor.
Its preparation is two joint error balls with the exact relation AB=A+B.
No report authenticates that relation or detects an interaction response.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._rational_interval import INTERVAL_METHOD, I, atan, pi_interval, sqrt
from ._localized_identity_transport import (
    _IDENTITY_TOLERANCE,
    _TAIL_TOLERANCE,
    _admit_cycle_law,
    _admit_cycle_window,
    _cycle_clock_bounds,
    _cycle_cubic_error_upper_bound,
)


@dataclass(frozen=True)
class _LocalizedMobileInteractionBound:
    """Conditional preparation budgets, with no evaluated endpoint signal.

    The two nominal Fejer packets each have norm a and disjoint opposite
    carrier bands. Arbitrary joint errors of norm eps retain common means.
    The joint initial state is exactly the sum of those same two states.
    Band observations remain defined during overlap; a unique spatial center
    is not asserted there. All candidate bounds require their own bootstrap.
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
    carrier_modes: tuple[int, int]
    initial_centers: tuple[int, int]
    endpoint_centers: tuple[int, int]
    clock_rate_bounds: I
    original_horizon_bounds: I
    sqrt_two_bounds: I
    linear_relative_error_upper_bound: Q
    joint_identity_error_upper_bound: Q
    joint_relative_identity_error_upper_bound: Q
    identity_bootstrap_margin: Q
    identity_bootstrap_certified: bool
    reference_tail_fraction_upper_bound: Q
    band_tail_fraction_upper_bound: Q | None
    joint_state_norm_lower_bound: Q | None
    joint_state_norm_upper_bound: Q | None
    band_state_norm_lower_bound: Q | None
    endpoint_window_separation_margin: Q
    prepared_identity_certified: bool
    single_template_relative_error_upper_bound: Q
    single_template_bootstrap_certified: bool
    nominal_packet_sup_norm_upper_bound: Q
    single_state_sup_norm_upper_bound: Q | None
    paired_state_sup_norm_upper_bound: Q | None
    single_state_radius: Q
    paired_state_radius: Q
    single_state_bootstrap_margin: Q
    paired_state_bootstrap_margin: Q
    single_sensitivity_denominator: Q
    paired_sensitivity_denominator: Q
    fifth_order_remainder_majorant: Q
    nominal_cubic_remainder_upper_bound: Q | None
    shared_preparation_sensitivity_upper_bound: Q | None
    interaction_proxy_error_upper_bound: Q | None
    cubic_remainder_certified: bool
    unavailable_reasons: tuple[str, ...]
    status: str
    interaction_norm_bounds: None = None
    identity_tolerance: Q = _IDENTITY_TOLERANCE
    tail_tolerance: Q = _TAIL_TOLERANCE
    interval_method: str = INTERVAL_METHOD
    conditional_on_supplied_preparation: bool = True
    scope: tuple[str, ...] = (
        "complete_positive_loss_sine_rows_fixed_unit_cycle_and_unit_capacities",
        "both_evolution_rows_use_tau_equals_exchange_times_original_time_over_pi",
        "two_opposite_Fejer_carriers_have_supplied_centers_and_prepared_full_states",
        "two_joint_error_balls_include_all_fine_and_common_mean_components",
        "paired_initial_state_is_exactly_the_sum_of_the_same_two_single_states",
        "correlated_shared_preparation_is_a_premise_not_a_copying_protocol",
        "identity_uses_the_joint_full_state_and_declared_band_components",
        "cubic_predictor_uses_the_actual_damped_tangent_not_a_lossless_substitute",
        "linear_response_cancels_in_the_full_state_superposition_defect",
        "fifth_order_remainder_and_preparation_sensitivity_have_distinct_budgets",
        "no_cubic_predictor_trajectory_or_endpoint_response_is_evaluated",
        "no_interaction_detection_formation_source_authentication_or_physical_claim",
    )


def _bound_localized_mobile_interaction(
    *,
    reference_model,
    node_count,
    fejer_order,
    amplitude,
    initial_state_error_bound,
    localization_radius,
    horizon,
) -> _LocalizedMobileInteractionBound:
    """Bound paired identity and prospective cubic accuracy from fresh primitives.

    The cycle, law and integer horizon use the prepared-transport admission;
    T must additionally be even. The opposite reference centers start at
    +/-T/2, meet at T/2 and exchange places at T. A>0, eps>=0 and 1<R<N/2.
    Separated endpoint windows and the fixed 1/25 identity/tail tolerances
    are sufficient verdicts, not additional admission requirements.

    The cubic approximation is I3(T)=integral U(T-s)[R3(vA+vB)-R3(vA)
    -R3(vB)] ds, with U the actual positive-loss tangent. No value of I3 is
    consumed or produced. Its error certificate uses radii 9A/8 and 3A/2
    enclosing the two single families and their exact shared sum. Template
    bounds also control their coordinate sup norms, sharpening the cubic
    derivative and fifth remainder independently of spatial overlap. Failure
    leaves actual error fields unavailable rather than treating them as zero.
    """
    _admit_cycle_window(node_count, fejer_order, horizon)
    if horizon % 2:
        raise ValueError("paired packet horizon must be even")
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
    pi, root_two = pi_interval(), sqrt(I(2))
    linear_error = 2 * pi.hi * loss * horizon / exchange
    linear_error += Q(horizon, 6) * (2 * pi.hi * Q(fejer_order - 1, node_count)) ** 3
    joint_error = 2 * error + root_two.hi * amplitude * linear_error
    joint_error += _cycle_cubic_error_upper_bound(2 * amplitude, horizon)
    relative_error = joint_error / amplitude
    relative_joint_error = relative_error / root_two.lo
    identity_margin = (2 - root_two.hi) * amplitude - joint_error
    identity_bootstrap = identity_margin > 0
    reference_tail = Q(node_count**3, 16 * fejer_order**3) / (radius - 1) ** 3
    band_tail = (
        ((sqrt(I(reference_tail)).hi + relative_error) / (1 - relative_error)) ** 2
        if identity_bootstrap
        else None
    )
    separation = horizon - 2 * radius
    identity = (
        identity_bootstrap
        and relative_joint_error < _IDENTITY_TOLERANCE
        and band_tail < _TAIL_TOLERANCE
        and separation > 0
    )

    single_template_error = error / amplitude + linear_error
    single_template_error += (
        _cycle_cubic_error_upper_bound(2 * amplitude, horizon) / amplitude
    )
    single_template_bootstrap = single_template_error < 1
    coefficient_square_sum = Q(2 * fejer_order**2 + 1, 3 * fejer_order)
    packet_sup = (
        amplitude * fejer_order / sqrt(I(node_count * coefficient_square_sum)).lo
    )
    single_sup = packet_sup + amplitude * single_template_error
    paired_sup = 2 * packet_sup + joint_error
    templates = identity_bootstrap and single_template_bootstrap
    single_radius, paired_radius = Q(9, 8) * amplitude, Q(3, 2) * amplitude
    single_margin = single_radius - amplitude - error
    single_margin -= _cycle_cubic_error_upper_bound(single_radius, horizon)
    paired_margin = paired_radius - root_two.hi * amplitude - 2 * error
    paired_margin -= _cycle_cubic_error_upper_bound(paired_radius, horizon)
    single_denominator = 1 - 4 * single_sup**2 * horizon
    paired_denominator = 1 - 4 * paired_sup**2 * horizon
    fifth_majorant = (Q(8, 3) * horizon**2 + Q(4, 15) * horizon) * (
        paired_sup**4 * paired_radius + 2 * single_sup**4 * single_radius
    )
    cubic = (
        templates
        and single_margin > 0
        and paired_margin > 0
        and single_denominator > 0
        and paired_denominator > 0
    )
    sensitivity = (
        2 * error * (1 / paired_denominator - 1 + 1 / single_denominator - 1)
        if cubic
        else None
    )
    reasons = []
    if not identity_bootstrap:
        reasons.append("joint_identity_bootstrap_not_closed")
    if not identity:
        reasons.append("prepared_identity_tolerances_not_certified")
    if not cubic:
        reasons.append("cubic_error_bootstrap_not_closed")
    clock_rate, original_horizon = _cycle_clock_bounds(exchange, horizon, pi)
    return _LocalizedMobileInteractionBound(
        node_count=node_count,
        fejer_order=fejer_order,
        amplitude=amplitude,
        initial_state_error_bound=error,
        localization_radius=radius,
        horizon=horizon,
        form_loss_coefficient=loss,
        exchange_coefficient=exchange,
        storage_scale=beta,
        carrier_modes=(node_count // 4, -node_count // 4),
        initial_centers=(horizon // 2, -horizon // 2),
        endpoint_centers=(-horizon // 2, horizon // 2),
        clock_rate_bounds=clock_rate,
        original_horizon_bounds=original_horizon,
        sqrt_two_bounds=root_two,
        linear_relative_error_upper_bound=linear_error,
        joint_identity_error_upper_bound=joint_error,
        joint_relative_identity_error_upper_bound=relative_joint_error,
        identity_bootstrap_margin=identity_margin,
        identity_bootstrap_certified=identity_bootstrap,
        reference_tail_fraction_upper_bound=reference_tail,
        band_tail_fraction_upper_bound=band_tail,
        joint_state_norm_lower_bound=(
            root_two.lo * amplitude - joint_error if identity_bootstrap else None
        ),
        joint_state_norm_upper_bound=(
            root_two.hi * amplitude + joint_error if identity_bootstrap else None
        ),
        band_state_norm_lower_bound=(
            amplitude - joint_error if identity_bootstrap else None
        ),
        endpoint_window_separation_margin=separation,
        prepared_identity_certified=identity,
        single_template_relative_error_upper_bound=single_template_error,
        single_template_bootstrap_certified=single_template_bootstrap,
        nominal_packet_sup_norm_upper_bound=packet_sup,
        single_state_sup_norm_upper_bound=single_sup if templates else None,
        paired_state_sup_norm_upper_bound=paired_sup if templates else None,
        single_state_radius=single_radius,
        paired_state_radius=paired_radius,
        single_state_bootstrap_margin=single_margin,
        paired_state_bootstrap_margin=paired_margin,
        single_sensitivity_denominator=single_denominator,
        paired_sensitivity_denominator=paired_denominator,
        fifth_order_remainder_majorant=fifth_majorant,
        nominal_cubic_remainder_upper_bound=fifth_majorant if cubic else None,
        shared_preparation_sensitivity_upper_bound=sensitivity,
        interaction_proxy_error_upper_bound=(
            fifth_majorant + sensitivity if cubic else None
        ),
        cubic_remainder_certified=cubic,
        unavailable_reasons=tuple(reasons),
        status="certified" if not reasons else "not_certified",
    )


@dataclass(frozen=True)
class _LocalizedMobileGeometryBound:
    """Conditional band geometry for the joint, tangent and separate families.

    The nested budget is rebuilt from primitives by this calculator, never
    consumed from an incoming report. Moment centers and proximity masses
    discard state information; they define neither a state quotient nor an
    autonomous law. Each optional conclusion retains its own hypotheses.
    """

    budget: _LocalizedMobileInteractionBound
    proximity_distance: Q
    common_relative_error_majorant: Q
    common_family_bootstrap_certified: bool
    band_norm_lower_bound: Q | None
    common_tail_fraction_upper_bound: Q | None
    mass_tail_bound_certified: bool
    reference_first_moment_ratio: Q
    first_moment_error_majorant: Q
    moment_nonvanishing_margin: Q
    moment_nonvanishing_certified: bool
    center_angle_error_upper_bound: Q | None
    center_position_error_upper_bound: Q | None
    relative_branch_margin: Q | None
    relative_branch_certified: bool
    relative_lifetime_additive_error_upper_bound: Q | None
    initial_endpoint_distance_lower_bound: Q | None
    coincident_reference_mass_lower_bound: Q | None
    initial_endpoint_mass_upper_bound: Q | None
    unavailable_reasons: tuple[str, ...]
    interval_method: str = INTERVAL_METHOD
    conditional_on_supplied_preparation: bool = True
    scope: tuple[str, ...] = (
        "inherits_the_freshly_admitted_complete_law_clock_and_shared_preparation",
        "form_and_phase_are_two_real_channels_with_one_shared_phase_lift_background",
        "band_projectors_are_observations_not_invariant_dynamical_subsystems",
        "common_error_covers_joint_nonlinear_tangent_and_sum_of_separate_flows",
        "moment_centers_require_nonvanishing_first_circular_moments",
        "relative_lifetime_bound_is_interval_width_over_two_plus_reported_correction",
        "lifetime_refers_to_one_contiguous_residence_on_the_certified_relative_branch",
        "proximity_mass_uses_normalized_band_intensities_and_cyclic_vertex_distance",
        "equal_band_intensities_do_not_identify_the_complete_state_or_its_evolution",
        "no_state_response_or_trajectory_is_observed_or_evaluated",
        "no_formation_detection_source_authentication_or_physical_identification",
    )


def _bound_localized_mobile_geometry(
    *,
    reference_model,
    node_count,
    fejer_order,
    amplitude,
    initial_state_error_bound,
    localization_radius,
    proximity_distance,
    horizon,
) -> _LocalizedMobileGeometryBound:
    """Bound circular centers and proximity masses from fresh preparations.

    Existing packet admission applies, with an additional exact distance
    0<=d<=N/2. The common error covers the actual joint flow, its actual
    damped tangent and the sum of the two separately evolved preparations.
    Real linearity suffices; complex linearity or closed carrier bands are
    not assumed. The original interaction report and its verdict are intact.

    A band moment is sum_j exp(2*pi*i*j/N)*abs(y_j)**2. Its center is
    unavailable if the perturbation disk can contain zero. A positive
    relative branch margin gives residence time <=W/2+2*delta in any fixed
    lifted interval of width W. It does not give an instantaneous velocity.
    Proximity bounds refer to the two nominal centers coinciding at T/2
    or having cyclic distance T at the initial and final endpoints.
    """
    distance = _exact(proximity_distance, "proximity_distance")
    budget = _bound_localized_mobile_interaction(
        reference_model=reference_model,
        node_count=node_count,
        fejer_order=fejer_order,
        amplitude=amplitude,
        initial_state_error_bound=initial_state_error_bound,
        localization_radius=localization_radius,
        horizon=horizon,
    )
    if not 0 <= distance <= Q(budget.node_count, 2):
        raise ValueError("proximity_distance requires 0 <= d <= N/2")

    a, radius, order = budget.amplitude, budget.localization_radius, budget.fejer_order
    common_error = budget.joint_identity_error_upper_bound / a
    common_error += _cycle_cubic_error_upper_bound(2 * a, budget.horizon) / a
    family_bootstrap = (
        budget.identity_bootstrap_certified
        and budget.single_template_bootstrap_certified
    )
    norm_available = family_bootstrap and common_error < 1
    tail = (
        (
            (sqrt(I(budget.reference_tail_fraction_upper_bound)).hi + common_error)
            / (1 - common_error)
        )
        ** 2
        if norm_available
        else None
    )
    mass_tail = tail is not None and tail < 1
    kappa = Q(2 * order**2 - 2, 2 * order**2 + 1)
    moment_error = common_error * (2 + common_error)
    moment_margin = kappa - moment_error
    moment = norm_available and moment_margin > 0
    angle = center_error = branch_margin = None
    if moment:
        pi = pi_interval()
        ratio = moment_error / kappa
        denominator = sqrt(I(1 - ratio**2))
        # The exact denominator is positive. A zero lower enclosure means
        # insufficient grid resolution, not an admissible division by zero.
        angle = (
            min(pi.hi / 2, atan(I(ratio / denominator.lo)).hi)
            if denominator.lo > 0
            else pi.hi / 2
        )
        center_error = budget.node_count * angle / (2 * pi.lo)
        branch_margin = Q(budget.node_count, 2) - budget.horizon - 2 * center_error
    branch = branch_margin is not None and branch_margin > 0
    middle_mass = (1 - tail) ** 2 if mass_tail and distance >= 2 * radius else None
    endpoint_mass = (
        1 - (1 - tail) ** 2
        if mass_tail and budget.horizon > 2 * radius + distance
        else None
    )
    reasons = []
    if not budget.identity_bootstrap_certified:
        reasons.append("joint_template_bootstrap_not_closed")
    if not budget.single_template_bootstrap_certified:
        reasons.append("single_template_bootstrap_not_closed")
    if not norm_available:
        reasons.append("common_nonzero_band_norm_not_certified")
    if not mass_tail:
        reasons.append("common_mass_tail_below_one_not_certified")
    if not moment:
        reasons.append("first_moment_nonvanishing_not_certified")
    if not branch:
        reasons.append("relative_lift_branch_not_certified")
    if distance < 2 * radius:
        reasons.append("proximity_distance_does_not_cover_coincident_windows")
    if budget.horizon <= 2 * radius + distance:
        reasons.append("endpoint_reference_windows_not_strictly_farther_than_proximity")
    return _LocalizedMobileGeometryBound(
        budget=budget,
        proximity_distance=distance,
        common_relative_error_majorant=common_error,
        common_family_bootstrap_certified=family_bootstrap,
        band_norm_lower_bound=a * (1 - common_error) if norm_available else None,
        common_tail_fraction_upper_bound=tail,
        mass_tail_bound_certified=mass_tail,
        reference_first_moment_ratio=kappa,
        first_moment_error_majorant=moment_error,
        moment_nonvanishing_margin=moment_margin,
        moment_nonvanishing_certified=moment,
        center_angle_error_upper_bound=angle,
        center_position_error_upper_bound=center_error,
        relative_branch_margin=branch_margin,
        relative_branch_certified=branch,
        relative_lifetime_additive_error_upper_bound=(
            2 * center_error if branch else None
        ),
        initial_endpoint_distance_lower_bound=(
            max(Q(0), budget.horizon - 2 * center_error) if moment else None
        ),
        coincident_reference_mass_lower_bound=middle_mass,
        initial_endpoint_mass_upper_bound=endpoint_mass,
        unavailable_reasons=tuple(reasons),
    )


@dataclass(frozen=True)
class _LocalizedMobileRelativeRateBound:
    """Conditional instantaneous rates of the actual joint-flow band centers.

    The decomposition compares the nonlinear field with its damped tangent
    at the same complete state. It does not compare histories, identify a
    closed relative-coordinate law or evaluate any prepared response.
    """

    geometry: _LocalizedMobileGeometryBound
    sigma_bounds: I
    conservative_error_upper_bound: Q | None
    damping_error_upper_bound: Q | None
    same_state_nonlinear_correction_upper_bound: Q | None
    per_center_rate_error_upper_bound: Q | None
    positive_band_center_rate_bounds: I | None
    negative_band_center_rate_bounds: I | None
    relative_center_rate_bounds: I | None
    band_rate_bounds_available: bool
    relative_rate_bounds_available: bool
    negative_relative_rate_margin: Q | None
    strictly_decreasing_relative_center_certified: bool
    unavailable_reasons: tuple[str, ...]
    interval_method: str = INTERVAL_METHOD
    conditional_on_supplied_preparation: bool = True
    scope: tuple[str, ...] = (
        "inherits_fresh_primitive_complete_law_clock_and_correlated_preparation",
        "actual_joint_nonlinear_flow_only_not_the_sum_of_separate_flow_derivatives",
        "both_real_rows_use_tau_equals_exchange_times_original_time_over_pi",
        "positive_and_negative_carrier_band_center_velocities_are_near_minus_and_plus_one",
        "relative_center_is_positive_band_center_minus_negative_band_center",
        "same_state_damped_tangent_and_nonlinear_remainder_have_separate_bounds",
        "nonlinear_correction_is_unsigned_and_need_not_point_toward_capture",
        "adjacent_frequency_differences_control_the_conservative_moment_rate",
        "opposite_band_conjugation_in_the_damping_term_is_retained",
        "relative_principal_rate_requires_the_certified_lift_branch",
        "failed_negative_rate_certificate_does_not_assert_reversal_or_formation",
        "no_state_response_or_trajectory_is_evaluated_no_closed_observation_law",
    )


def _bound_localized_mobile_relative_rate(
    *,
    reference_model,
    node_count,
    fejer_order,
    amplitude,
    initial_state_error_bound,
    localization_radius,
    proximity_distance,
    horizon,
) -> _LocalizedMobileRelativeRateBound:
    """Bound actual joint-flow center rates without evaluating a full state.

    Fresh primitive admission and the existing geometry bootstrap provide
    ||Pz||<=a(1+h), |M(Pz)|>=a^2*m and ||z||<2a. For the normalized complete
    sine law, R(z) obeys ||R||<=32*a^3/3. The bound retains the actual
    positive-loss tangent; it is not the lossless packet reference velocity.

    The conservative, dissipative and same-state nonlinear error estimates
    are per center. Each requires certified nonzero moments. The relative
    principal rate additionally requires a certified relative branch; a
    negative upper endpoint is sufficient for strict decrease. Failure of
    either condition leaves that conclusion unavailable, never reversed.
    """
    geometry = _bound_localized_mobile_geometry(
        reference_model=reference_model,
        node_count=node_count,
        fejer_order=fejer_order,
        amplitude=amplitude,
        initial_state_error_bound=initial_state_error_bound,
        localization_radius=localization_radius,
        proximity_distance=proximity_distance,
        horizon=horizon,
    )
    budget, pi = geometry.budget, pi_interval()
    loss_ratio = budget.form_loss_coefficient / budget.exchange_coefficient
    sigma = I(pi.lo * loss_ratio, pi.hi * loss_ratio)
    available = (
        geometry.common_family_bootstrap_certified
        and geometry.moment_nonvanishing_certified
    )
    conservative = damping = nonlinear = per_center = None
    positive = negative = relative = margin = None
    if available:
        norm_ratio = 1 + geometry.common_relative_error_majorant
        moment_margin = geometry.moment_nonvanishing_margin
        norm_squared_over_moment = norm_ratio**2 / moment_margin
        frequency_error = (Q(1, 6) + Q((2 * budget.fejer_order - 3) ** 2, 2)) * (
            pi.hi / budget.node_count
        ) ** 2
        conservative = frequency_error * norm_squared_over_moment
        damping = 2 * budget.node_count * loss_ratio * norm_squared_over_moment
        nonlinear = (
            Q(budget.node_count, 1)
            / pi.lo
            * Q(32, 3)
            * budget.amplitude**2
            * norm_ratio
            / moment_margin
        )
        per_center = conservative + damping + nonlinear
        positive = I(-1 - per_center, -1 + per_center)
        negative = I(1 - per_center, 1 + per_center)
        if geometry.relative_branch_certified:
            relative = I(-2 - 2 * per_center, -2 + 2 * per_center)
            margin = 2 - 2 * per_center
    relative_available = relative is not None
    decreasing = relative_available and relative.hi < 0
    reasons = []
    if not available:
        reasons.append("joint_family_nonzero_moments_not_certified")
    if not geometry.relative_branch_certified:
        reasons.append("relative_lift_branch_not_certified")
    if relative_available and not decreasing:
        reasons.append("strictly_negative_relative_rate_not_certified")
    return _LocalizedMobileRelativeRateBound(
        geometry=geometry,
        sigma_bounds=sigma,
        conservative_error_upper_bound=conservative,
        damping_error_upper_bound=damping,
        same_state_nonlinear_correction_upper_bound=nonlinear,
        per_center_rate_error_upper_bound=per_center,
        positive_band_center_rate_bounds=positive,
        negative_band_center_rate_bounds=negative,
        relative_center_rate_bounds=relative,
        band_rate_bounds_available=available,
        relative_rate_bounds_available=relative_available,
        negative_relative_rate_margin=margin,
        strictly_decreasing_relative_center_certified=decreasing,
        unavailable_reasons=tuple(reasons),
    )
