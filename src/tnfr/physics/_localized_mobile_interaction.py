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
from ..mathematics._rational_interval import INTERVAL_METHOD, I, pi_interval, sqrt
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
