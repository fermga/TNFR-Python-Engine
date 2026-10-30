"""Acquired winding-dependent collective response on two fixed C9 interfaces.

The same sine law, registered relative phase and final support are compared
between homogeneous winding-one and winding-two pairs. Original independent
sources are recovered without resetting their fine state. The fixed protocol
and proof belong to ORGANIZED_PATTERN_CONSTITUTIVE_INHERITANCE.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._rational_interval import I, cos, pi_interval, sin, sqrt
from ._sine_constitutive_inheritance import (
    _ConstitutiveInheritance,
    _derive_constitutive_inheritance,
)
from ._sine_constitutive_recovery import (
    _assess_constitutive_recovery,
    _ConstitutiveRecovery,
)
from .reversible_eigenmode_reference import _negative_exp_bounds

_SOURCE_ALLOWANCE = Q(1, 10**10)
_READOUT_ALLOWANCE = Q(1, 10**10)
_RECOVERED_RADIUS = Q(1, 10**70)
_WAITING_RADIUS = 20 * _RECOVERED_RADIUS
_PHASE_GAP = Q(1, 1024)
_DURATION = Q(1, 10)
_TARGET = Q(1, 10**9)
_RADIUS = Q(1, 12)
_COERCIVITY = Q(1, 100)
_BARRIER = _COERCIVITY * _RADIUS**2


@dataclass(frozen=True)
class _SineWindingInterfaceResponse:
    """Conditional endpoint bounds for both acquired classes on one support.

    Every response is the first child's mean form at the fixed duration,
    also its increment because the admitted initial mean is exactly zero.
    Tuple endpoints remain exact rational bounds; the positive feedback loss
    and tiny fine-state allowances are kept outside interval-grid rounding.
    Candidate bounds do not certify an unsupported source or observation.
    """

    interface: str
    laws: tuple[_ConstitutiveInheritance, _ConstitutiveInheritance]
    exponential_decay_bounds: tuple[Q, Q]
    duhamel_gain_bounds: I
    phase_bootstrap_margin: Q
    feedback_upper_bounds: tuple[Q, Q]
    ideal_response_bounds: tuple[tuple[Q, Q], tuple[Q, Q]]
    actual_response_bounds: tuple[tuple[Q, Q], tuple[Q, Q]]
    recorded_response_bounds: tuple[tuple[Q, Q], tuple[Q, Q]]
    ideal_contrast_bounds: tuple[Q, Q]
    actual_contrast_bounds: tuple[Q, Q]
    recorded_contrast_bounds: tuple[Q, Q]
    fine_mean_error_upper_bound: Q
    fine_contrast_error_upper_bound: Q
    nominal_attachment_work_bounds: tuple[I, I]
    attachment_work_correction_upper_bound: Q
    initial_joint_norm_squared_upper_bound: Q
    post_attachment_excess_upper_bound: Q
    phase_hessian_lower_bound: Q
    full_graph_gap_lower_bound: Q
    storage_coercivity_lower_bound: Q
    storage_barrier: Q
    radius_margin: Q
    storage_margin: Q
    identity_certified: bool
    response_certified: bool
    status: str
    unresolved_conditions: tuple[str, ...]
    duration: Q = _DURATION
    original_duration: Q = _DURATION * Q(1024, 1023)
    relative_phase_gap: Q = _PHASE_GAP
    observation: str = "mean form of component zero; class one minus class two"
    scope: tuple[str, ...] = (
        "same_interface_complete_sine_law_and_probe_in_both_class_alternatives",
        "homogeneous_equal_winding_pair_only_not_heterogeneous_class_closure",
        "class_registration_and_interface_remain_constitutive_information",
        "carrier_endpoint_bounds_retain_one_sided_nonlinear_phase_feedback",
        "actual_source_residuals_are_retained_without_mean_or_phase_reset",
        "nominal_support_work_plus_exact_nonnegative_residual_correction",
        "all_future_winding_identity_after_one_supplied_support_event",
        "finite_response_not_a_repeated_probe_or_outer_identity_certificate",
    )


@dataclass(frozen=True)
class _SineWindingResponse:
    """Fresh preparation, indefinite isolated waiting and finite response.

    Each recovery bounds either of two independently prepared constituents
    of that class. Common phase origins are supplied from their original
    isolated preparations; this calculation introduces no later phase jump.
    Waiting is any additional finite unforced isolated duration. The finite
    joined response does not replace that isolated persistence premise.
    """

    form_error_bound: Q
    phase_error_bound: Q
    observation_error_bound: Q
    recoveries: tuple[_ConstitutiveRecovery, _ConstitutiveRecovery]
    interfaces: tuple[_SineWindingInterfaceResponse, _SineWindingInterfaceResponse]
    source_allowances_within_protocol: bool
    observation_allowance_within_protocol: bool
    source_certified: bool
    isolated_waiting_certified: bool
    isolated_excess_upper_bound: Q
    isolated_waiting_storage_margin: Q
    certified: bool
    status: str
    unresolved_conditions: tuple[str, ...]
    recovered_radius: Q = _RECOVERED_RADIUS
    waiting_radius: Q = _WAITING_RADIUS
    identity_radius: Q = _RADIUS
    isolated_coercivity_lower_bound: Q = _COERCIVITY
    isolated_storage_barrier: Q = _BARRIER
    source_allowance_limit: Q = _SOURCE_ALLOWANCE
    observation_allowance_limit: Q = _READOUT_ALLOWANCE
    response_target: Q = _TARGET
    class_order: tuple[int, int] = (1, 2)
    interface_order: tuple[str, str] = ("matching", "symmetric")
    initial_form_means: tuple[Q, Q] = (Q(0), Q(0))
    supplied_phase_origins: tuple[Q, Q] = (-_PHASE_GAP / 2, _PHASE_GAP / 2)
    deformation: Q = Q(0)
    clock: str = "tau=(1023/1024)*t; isolated factor1/2, joined factor1/(2+q)"
    scope: tuple[str, ...] = (
        "original_two_ramp_classes_with_separate_zero_sum_fine_errors",
        "both_classes_recovered_at_unchanged_time100_plus_dwell1e13",
        "arbitrary_further_isolated_waiting_preserves_a_full_state_allowance",
        "common_phase_origins_supplied_from_initial_preparation_not_a_later_jump",
        "support_only_attachment_preserves_all_actual_fine_coordinates",
        "source_costs_and_attachment_work_can_differ_between_class_alternatives",
        "positive_attachment_work_requires_external_supply_not_prior_loss_credit",
        "registered_collective_form_phase_and_acquired_class_are_distinct_data",
        "matching_control_is_finite_residual_blindness_not_exact_fine_equality",
        "mathematical_readout_error_has_no_laboratory_sensor_or_clock_bridge",
        "no_trajectory_event_execution_cached_verdict_or_autonomous_assembly",
    )


def _assess_sine_winding_response(
    *, form_error_bound, phase_error_bound, observation_error_bound
):
    """Assess the fixed two-class protocol from fresh primitive allowances.

    All three raw values admit before any source, trigonometric or exponential
    calculation. Nonnegative values outside the declared 1e-10 limits retain
    candidate bounds but cannot certify this fixed protocol. No source report,
    dwell, probe amplitude, duration or interface replacement is an input.
    """
    rx = _exact(form_error_bound, "form_error_bound")
    rt = _exact(phase_error_bound, "phase_error_bound")
    noise = _exact(observation_error_bound, "observation_error_bound")
    if min(rx, rt, noise) < 0:
        raise ValueError("source and observation error bounds must be nonnegative")

    source_limits = max(rx, rt) <= _SOURCE_ALLOWANCE
    noise_limit = noise <= _READOUT_ALLOWANCE
    recoveries = tuple(
        _assess_constitutive_recovery(
            deformation=0,
            form_error_bound=rx,
            phase_error_bound=rt,
            target_winding=winding,
        )
        for winding in (1, 2)
    )
    source_ok = source_limits and all(row.recovery_certified for row in recoveries)
    isolated_excess = 4 * _RECOVERED_RADIUS**2
    waiting_margin = _BARRIER - isolated_excess
    waiting_geometry = all(
        row.source.phase_hessian_lower_bound is not None
        and row.source.phase_hessian_lower_bound > Q(1, 20)
        and row.source.coercivity_lower_bound is not None
        and row.source.coercivity_lower_bound > _COERCIVITY
        and row.radius == _RADIUS
        and row.endpoint_radius == _RECOVERED_RADIUS
        for row in recoveries
    )
    waiting_ok = source_ok and waiting_geometry and waiting_margin > 0

    # These are the actual complete 18-node phase/mean-leaf constants. They
    # do not reuse the earlier class-one acquired-inheritance verdict.
    cosine_lower = cos(4 * pi_interval() / 9 + sqrt(I(2)) * _RADIUS).lo
    initial_norm = Q(9, 2) * _PHASE_GAP**2 + 4 * _WAITING_RADIUS**2
    excess = 5 * initial_norm
    radius_margin = _RADIUS**2 - initial_norm
    storage_margin = _BARRIER - excess
    fine_error = Q(10, 7) * _WAITING_RADIUS
    contrast_error = 2 * fine_error
    responses = []
    for interface in ("matching", "symmetric"):
        laws = tuple(
            _derive_constitutive_inheritance(
                interface=interface, deformation=0, target_winding=winding
            )
            for winding in (1, 2)
        )
        gamma, kappa, q = (
            laws[0].gamma_bounds,
            laws[0].collective_clock_factor,
            laws[0].cross_degree,
        )
        harmonics = tuple(row.first_harmonic_multiplier_bounds for row in laws)
        decay = _negative_exp_bounds(2 * kappa * _DURATION)
        gain = gamma * sin(I(_PHASE_GAP)) * (1 - I(*decay)) / 2
        leading = tuple(gain * harmonic for harmonic in harmonics)
        feedback = tuple(
            Q(2, 3)
            * gamma.hi**3
            * kappa**3
            * harmonic.hi**2
            * _PHASE_GAP
            * _DURATION**3
            for harmonic in harmonics
        )
        bootstrap_margin = Q(1, 2) - (
            2 * gamma.hi**2 * kappa**2 * max(r.hi for r in harmonics) * _DURATION**2
        )
        ideal = tuple((row.lo - error, row.hi) for row, error in zip(leading, feedback))
        actual = tuple((lo - fine_error, hi + fine_error) for lo, hi in ideal)
        recorded = tuple((lo - noise, hi + noise) for lo, hi in actual)
        if interface == "matching":
            # Equal carrier laws and initial data, not independent endpoint
            # interval subtraction, prove this exact nominal cancellation.
            ideal_contrast = (Q(0), Q(0))
        else:
            difference = gain * (harmonics[0] - harmonics[1])
            ideal_contrast = (
                difference.lo - feedback[0],
                difference.hi + feedback[1],
            )
        actual_contrast = (
            ideal_contrast[0] - contrast_error,
            ideal_contrast[1] + contrast_error,
        )
        recorded_contrast = (
            actual_contrast[0] - 2 * noise,
            actual_contrast[1] + 2 * noise,
        )
        work = tuple(9 * q * (1 - r * cos(I(_PHASE_GAP))) for r in harmonics)
        geometry_ok = (
            cosine_lower > Q(1, 20)
            and 2 - 2 * laws[0].alpha_cosine_bounds.hi > Q(2, 5)
            and all(row.degree == 2 + q for row in laws)
            and all(row.gamma_bounds == gamma for row in laws)
        )
        identity = (
            waiting_ok and geometry_ok and radius_margin > 0 and storage_margin > 0
        )
        finite_ok = (
            identity
            and 2 + 2 * gamma.hi < 3
            and bootstrap_margin > 0
            and all(r.lo > 0 for r in harmonics)
        )
        separated = (
            recorded_contrast[0] > _TARGET
            if interface == "symmetric"
            else max(abs(value) for value in recorded_contrast) < _TARGET
        )
        response_ok = finite_ok and noise_limit and separated
        reasons = tuple(
            label
            for passed, label in (
                (waiting_ok, "isolated_actual_source_waiting_not_certified"),
                (geometry_ok, "joined_full_state_geometry_not_certified"),
                (radius_margin > 0, "joined_initial_radius_not_certified"),
                (storage_margin > 0, "joined_storage_barrier_not_certified"),
                (finite_ok, "complete_finite_response_bound_not_certified"),
                (noise_limit, "observation_allowance_exceeds_fixed_protocol"),
                (separated, "fixed_interface_contrast_target_not_certified"),
            )
            if not passed
        )
        responses.append(
            _SineWindingInterfaceResponse(
                interface=interface,
                laws=laws,
                exponential_decay_bounds=decay,
                duhamel_gain_bounds=gain,
                phase_bootstrap_margin=bootstrap_margin,
                feedback_upper_bounds=feedback,
                ideal_response_bounds=ideal,
                actual_response_bounds=actual,
                recorded_response_bounds=recorded,
                ideal_contrast_bounds=ideal_contrast,
                actual_contrast_bounds=actual_contrast,
                recorded_contrast_bounds=recorded_contrast,
                fine_mean_error_upper_bound=fine_error,
                fine_contrast_error_upper_bound=contrast_error,
                nominal_attachment_work_bounds=work,
                attachment_work_correction_upper_bound=4 * q * _WAITING_RADIUS**2,
                initial_joint_norm_squared_upper_bound=initial_norm,
                post_attachment_excess_upper_bound=excess,
                phase_hessian_lower_bound=Q(1, 20),
                full_graph_gap_lower_bound=Q(2, 5),
                storage_coercivity_lower_bound=_COERCIVITY,
                storage_barrier=_BARRIER,
                radius_margin=radius_margin,
                storage_margin=storage_margin,
                identity_certified=identity,
                response_certified=response_ok,
                status="certified_winding_response" if response_ok else "unavailable",
                unresolved_conditions=reasons,
            )
        )
    certified = all(row.response_certified for row in responses)
    reasons = tuple(
        label
        for passed, label in (
            (source_limits, "source_allowance_exceeds_fixed_protocol"),
            (source_ok, "both_original_class_recoveries_not_certified"),
            (waiting_ok, "isolated_waiting_persistence_not_certified"),
            (noise_limit, "observation_allowance_exceeds_fixed_protocol"),
        )
        if not passed
    ) + tuple(
        f"{row.interface}:{reason}"
        for row in responses
        for reason in row.unresolved_conditions
    )
    return _SineWindingResponse(
        form_error_bound=rx,
        phase_error_bound=rt,
        observation_error_bound=noise,
        recoveries=recoveries,
        interfaces=tuple(responses),
        source_allowances_within_protocol=source_limits,
        observation_allowance_within_protocol=noise_limit,
        source_certified=source_ok,
        isolated_waiting_certified=waiting_ok,
        isolated_excess_upper_bound=isolated_excess,
        isolated_waiting_storage_margin=waiting_margin,
        certified=certified,
        status="certified_organized_winding_response" if certified else "unavailable",
        unresolved_conditions=reasons,
    )
