"""Detached sufficient budgets for a localized full-state sine-law packet.

Proof: LOCALIZED_IDENTITY_TRANSPORT.md#localized-transport-error.
The prescribed Fejer packet, directional carrier and complete law are premises.
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
    loss, exchange, beta = _sine_model_coefficients(reference_model, positive_loss=True)
    if beta != 1:
        raise ValueError("localized transport requires unit storage_scale")

    pi = pi_interval()
    initial = error / amplitude
    damping = 2 * pi.hi * loss * horizon / exchange
    dispersion = Q(horizon, 6) * (2 * pi.hi * Q(fejer_order - 1, node_count)) ** 3
    nonlinear = Q(32, 3) * horizon * amplitude**2
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
    # Divide exact positive coefficients before outward interval rounding. A
    # subgrid exchange rate may have a zero lower enclosure endpoint, but is
    # neither a zero clock nor an interval denominator that contains zero.
    clock_rate = I(exchange / pi.hi, exchange / pi.lo)
    original_horizon = I(horizon * pi.lo / exchange, horizon * pi.hi / exchange)
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
