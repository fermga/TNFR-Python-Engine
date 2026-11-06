"""Uniform full-law admission of the two-ramp receiver's slow phase motion.

ORGANIZED_RECEIVER_SLOW_ADMISSION.md owns the exact phase/velocity closure,
retained structural energy and all-future force-balance defect theorem.
No reference trajectory, donor-retention corridor or joined entry is evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._rational_interval import I, pi_interval, sqrt
from ._organized_receiver_mechanism import (
    _assess_organized_receiver_mechanism,
    _OrganizedReceiverMechanism,
)
from ._sine_preparation import _prepared_transit_radii, _prepared_velocity_defect
from .reversible_eigenmode_reference import _negative_exp_bounds

_RELEASE_TIME = Q(400)
_DEFECT_BUDGET = Q(11, 10**6)
_RATE_FLOOR = Q(3, 20)
_LIPSCHITZ = Q(2)


@dataclass(frozen=True)
class _OrganizedReceiverSlowAdmission:
    """Actual velocity control and its source-specific release conditions.

    The velocity discrepancy is uniformly bounded after release. It does
    not bound accumulated phase error without an admitted comparison tube.
    Phase-rate signs and windings refer only to the release instant.
    """

    preparation: _OrganizedReceiverMechanism
    nominal_initial_slow_velocity_bounds: tuple[I, ...]
    nominal_form_gradient_norm_bounds: I
    initial_defect_norm_upper_bound: Q
    damping_lower_bound: Q
    defect_decay_bounds: I
    asymptotic_defect_upper_bound: Q
    uniform_velocity_defect_upper_bound: Q
    velocity_defect_margin: Q
    uniform_form_rate_norm_upper_bound: Q
    release_form_decay_bounds: I
    release_scaled_form_norm_upper_bound: Q
    release_form_norm_upper_bound: Q
    release_phase_error_norm_upper_bound: Q
    bridge_phase_error_norm_upper_bound: Q
    bridge_donor_acute_margin: Q
    release_donor_acute_margin: Q
    release_receiver_zero_winding_margin: Q
    release_phase_rate_bounds: tuple[I, ...]
    release_receiver_signed_rate_lower_bound: Q
    receiver_rate_margin: Q
    release_windings: tuple[int, int, int] | None
    uniform_defect_certified: bool
    donor_release_bridge_certified: bool
    receiver_phase_motion_certified: bool
    status: str
    unresolved_conditions: tuple[str, ...]
    release_time: Q = _RELEASE_TIME
    original_release_time: Q = _RELEASE_TIME * Q(1024, 1023)
    velocity_defect_budget: Q = _DEFECT_BUDGET
    receiver_signed_rate_floor: Q = _RATE_FLOOR
    global_phase_lipschitz_bound: Q = _LIPSCHITZ
    structural_energy_ratio: Q = Q(1, 4)
    phase_mean_leaf_dimension: int = 26
    first_order_exact_closure: bool = False
    eventual_receiver_acquisition: str = "unresolved"
    eventual_donor_retention: str = "unresolved"
    joined_capture: str = "unresolved"
    slow_clock: str = "s=gamma**2*tau; tau=(1023/1024)*t"
    scope: tuple[str, ...] = (
        "unchanged_two_ramp_source_rebuilt_from_primitive_error_allowances",
        "exact_second_order_phase_law_on_global_mean_zero_leaf",
        "initial_phase_velocity_retains_supplied_form_and_all_errors",
        "structural_inertial_storage_is_H_over_four_not_physical_mass",
        "uniform_velocity_defect_for_all_times_after_release",
        "actual_receiver_phase_rate_signs_only_at_release_not_a_future_monotonicity_claim",
        "phase_accumulation_requires_separate_tube_and_residual_control",
        "donor_history_and_complete_form_phase_capture_remain_unresolved",
        "no_reference_trajectory_or_reserved_response_evaluated",
    )


def _assess_organized_receiver_slow_admission(
    *, form_error_bound, phase_error_bound
) -> _OrganizedReceiverSlowAdmission:
    """Rebuild source and certify slow velocity without accepting cached reports."""
    rx = _exact(form_error_bound, "form_error_bound")
    rt = _exact(phase_error_bound, "phase_error_bound")
    if rx < 0 or rt < 0:
        raise ValueError("preparation error bounds must be nonnegative")
    source = _assess_organized_receiver_mechanism(
        form_error_bound=rx, phase_error_bound=rt
    )
    gamma = source.capture_geometry.geometry.gamma_bounds
    epsilon = source.slow_clock_factor_bounds
    gap = source.semigroup_gap_lower_bound
    norm = source.forcing_norm_bounds
    gradient_norm = sqrt(
        I(sum(value**2 for value in source.nominal_initial_form_gradient))
    )
    inverse_gamma = 1 / gamma
    initial_defect = (
        gradient_norm.hi * inverse_gamma.hi / 4
        + 2 * norm.hi * rx * inverse_gamma.hi
        + _LIPSCHITZ * norm.hi * rt
    )
    damping, decay, floor, defect = _prepared_velocity_defect(
        time=_RELEASE_TIME,
        gap_lower=gap,
        feedback_upper=epsilon.hi,
        lipschitz_upper=_LIPSCHITZ,
        forcing_upper=norm.hi,
        initial_defect_upper=initial_defect,
    )
    form_decay = I(*_negative_exp_bounds(gap * _RELEASE_TIME))
    scaled_form, phase_remainder = _prepared_transit_radii(
        time=_RELEASE_TIME,
        decay_upper=form_decay.hi,
        gap_lower=gap,
        initial_norm_upper=source.scaled_initial_norm_upper_bound,
        feedback_upper=epsilon.hi,
        forcing_upper=norm.hi,
    )
    phase_radius = (
        source.proxy_reference_distance_bounds.hi
        + norm.hi * (rt + gamma.hi * rx)
        + phase_remainder
    )
    bridge_radius = (
        source.proxy_reference_distance_bounds.hi
        + norm.hi * (rt + gamma.hi * rx)
        + source.layer_scaled_form_norm_upper_bound
        + epsilon.hi * _RELEASE_TIME * norm.hi
    )
    form_radius = inverse_gamma.hi * scaled_form
    phase_row_error = sqrt(I(2)).hi * phase_radius + defect
    rates = tuple(
        current / 4 + I(-phase_row_error, phase_row_error)
        for current in source.reference_current_bounds
    )
    oriented = min(
        -rates[18 + j].hi if j < 4 else rates[18 + j].lo for j in range(9) if j != 4
    )
    pi = pi_interval()
    donor_margin = pi.lo / 2 - (2 * pi / 9).hi - sqrt(I(2)).hi * phase_radius
    bridge_margin = pi.lo / 2 - (2 * pi / 9).hi - sqrt(I(2)).hi * bridge_radius
    receiver_margin = pi.lo - sqrt(I(2)).hi * phase_radius
    defect_margin = _DEFECT_BUDGET - defect
    rate_margin = oriented - _RATE_FLOOR
    release_ok = donor_margin > 0 and receiver_margin > 0
    defect_ok = defect_margin > 0
    motion_ok = rate_margin > 0
    bridge_ok = bridge_margin > 0
    unresolved = tuple(
        reason
        for passed, reason in (
            (release_ok, "release_donor_one_receiver_zero_not_certified"),
            (bridge_ok, "donor_identity_bridge_not_certified"),
            (defect_ok, "uniform_velocity_defect_budget_not_certified"),
            (motion_ok, "receiver_phase_motion_floor_not_certified"),
        )
        if not passed
    )
    return _OrganizedReceiverSlowAdmission(
        preparation=source,
        nominal_initial_slow_velocity_bounds=tuple(
            inverse_gamma * value / 4 for value in source.nominal_initial_form_gradient
        ),
        nominal_form_gradient_norm_bounds=gradient_norm,
        initial_defect_norm_upper_bound=initial_defect,
        damping_lower_bound=damping,
        defect_decay_bounds=decay,
        asymptotic_defect_upper_bound=floor,
        uniform_velocity_defect_upper_bound=defect,
        velocity_defect_margin=defect_margin,
        uniform_form_rate_norm_upper_bound=gamma.hi * defect,
        release_form_decay_bounds=form_decay,
        release_scaled_form_norm_upper_bound=scaled_form,
        release_form_norm_upper_bound=form_radius,
        release_phase_error_norm_upper_bound=phase_radius,
        bridge_phase_error_norm_upper_bound=bridge_radius,
        bridge_donor_acute_margin=bridge_margin,
        release_donor_acute_margin=donor_margin,
        release_receiver_zero_winding_margin=receiver_margin,
        release_phase_rate_bounds=rates,
        release_receiver_signed_rate_lower_bound=oriented,
        receiver_rate_margin=rate_margin,
        release_windings=(1, 1, 0) if release_ok else None,
        uniform_defect_certified=defect_ok,
        donor_release_bridge_certified=bridge_ok,
        receiver_phase_motion_certified=motion_ok,
        status="certified_uniform_slow_admission" if not unresolved else "unavailable",
        unresolved_conditions=unresolved,
    )
