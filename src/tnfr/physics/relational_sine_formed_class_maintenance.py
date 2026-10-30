"""Uniform return under a supplied repeated probe of the fixed formed C9 classes.

Fresh formation and first-probe evidence defines the neighborhoods. A strict
nonlinear Lyapunov estimate returns both classes into half those neighborhoods
before every later jump. No trajectory or sequence of sampled cycles is run.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real
from ..mathematics._rational_interval import INTERVAL_METHOD, I
from ._sine_lyapunov import (
    _sine_lyapunov_coefficients,
    _sine_lyapunov_initial_upper,
    _sine_lyapunov_return_squared,
)
from .relational_sine_formed_classes import (
    SineFormedClassResponse,
    assess_sine_formed_class_response,
)
from .reversible_eigenmode_reference import _MAX_RATIONAL_EXPONENT, _negative_exp_bounds

__all__ = ("SineFormedClassMaintenance", "assess_sine_formed_class_maintenance")


@dataclass(frozen=True)
class SineFormedClassMaintenance:
    """Conditional repeated response and identity in invariant neighborhoods.

    The nested reference is rebuilt from primitives within this invocation.
    Its Euclidean pre-probe form and target-phase radii define ``K_k``; its
    supplied phase jump defines ``J(K_k)``. The reported squared return bounds
    must fit strictly inside one quarter of the respective squared radii,
    proving return into half ``K_k``. The same readout and work bounds then
    apply separately to every cycle without accumulated readout-error budgets.

    This is not an exact periodic orbit or convergence while probes continue.
    Target recovery after interventions stop follows the nested recovery
    admission. Repeated work is externally supplied and need not have a finite
    cumulative budget. No report-fed calculation or cached verdict is accepted.
    """

    reference_certificate: SineFormedClassResponse
    common_dwell: Q
    original_common_dwell: Q
    cosine_lower_bound: Q | None
    gamma_squared_bounds: I
    phase_stiffness_lower_bound: Q | None
    phase_stiffness_upper_bound: Q
    lyapunov_position_lower_coefficient: Q | None
    lyapunov_position_upper_coefficient: Q
    lyapunov_decay_rate: Q | None
    post_probe_lyapunov_upper_bounds: tuple[Q, Q]
    return_energy_thresholds: tuple[Q, Q] | None
    exponential_decay_bounds: I | None
    returned_lyapunov_upper_bounds: tuple[Q, Q] | None
    returned_form_norm_squared_upper_bounds: tuple[Q, Q] | None
    returned_phase_norm_squared_upper_bounds: tuple[Q, Q] | None
    return_energy_margin_bounds: tuple[I, I] | None
    form_return_margin_bounds: tuple[I, I] | None
    phase_return_margin_bounds: tuple[I, I] | None
    return_certified_by_class: tuple[bool, bool]
    status: str
    unavailable_reasons: tuple[str, ...]
    semigroup_gap_lower_bound: Q = Q(1, 5)
    semigroup_rate_upper_bound: Q = Q(2)
    lyapunov_cross_coefficient: Q = Q(1, 20)
    return_radius_factor: Q = Q(1, 2)
    clock: str = "tau=e*t; e=1023/1024"
    intervention_times: str = "probe_time+n*common_dwell; n=0,1,2,..."
    readout_times: str = "probe_time+n*common_dwell+probe_duration; n=0,1,2,..."
    interval_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "fresh_primitive_formation_and_first_probe_assessment_no_incoming_report",
        "same_fixed_positive_loss_C9_law_support_capacities_and_zero_mean_leaf",
        "same_supplied_phase_jump_at_every_declared_intervention",
        "pre_probe_K_is_product_of_Euclidean_form_and_target_phase_norm_balls",
        "nonlinear_return_from_every_member_of_J_K_into_half_K",
        "strict_post_jump_barriers_retain_winding_through_every_dwell",
        "same_correlated_recorded_contrast_and_signed_work_bounds_per_cycle",
        "readout_error_is_bounded_separately_per_readout_not_state_feedback",
        "cumulative_supplied_work_after_N_probes_is_N_times_per_probe_interval",
        "invariant_neighborhoods_not_an_exact_periodic_orbit",
        "target_convergence_when_interventions_stop_not_while_they_continue",
        "no_practical_recovery_speed_or_finite_total_work_budget_claim",
        "unavailable_is_not_instability_or_failed_maintenance",
        "no_trajectory_solver_probe_count_scan_autonomous_event_or_physical_identity",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {
            "schema": "tnfr.sine-formed-class-maintenance.v1",
            "report": _project(self),
        }


def assess_sine_formed_class_maintenance(
    *,
    formation_time,
    probe_time,
    probe_duration,
    phase_increment,
    form_error_bound,
    phase_error_bound,
    readout_error_bound,
    radius,
    common_dwell,
) -> SineFormedClassMaintenance:
    """Assess a common repeated probe with a nonlinear uniform return bound.

    All nine arguments are required exact/represented reals. The eight initial
    experiment inputs retain the formed-class response domain. ``common_dwell``
    must be strictly larger than ``probe_duration`` so each readout precedes
    the next jump. Its exponential work cap is ``lyapunov_decay_rate*dwell <=
    4096``, not the initial fast semigroup gap times dwell. Missing positive
    acute geometry yields unavailable return bounds; it does not supply a
    decay rate. All primitive admission precedes the fresh prerequisite call.
    """
    inputs = {
        "formation_time": formation_time,
        "probe_time": probe_time,
        "probe_duration": probe_duration,
        "phase_increment": phase_increment,
        "form_error_bound": form_error_bound,
        "phase_error_bound": phase_error_bound,
        "readout_error_bound": readout_error_bound,
        "radius": radius,
        "common_dwell": common_dwell,
    }
    values = {
        name: exact_or_represented_real(value, name) for name, value in inputs.items()
    }
    dwell = values.pop("common_dwell")
    tf, tp, h = (
        values[name] for name in ("formation_time", "probe_time", "probe_duration")
    )
    if tf < 0 or tp < tf or tp / 5 > _MAX_RATIONAL_EXPONENT:
        raise ValueError("times require 0 <= formation_time <= probe_time <= 20480")
    if h < 0 or 2 * h > _MAX_RATIONAL_EXPONENT:
        raise ValueError("probe_duration requires 0 <= 2*probe_duration <=4096")
    if dwell <= h:
        raise ValueError("common_dwell must be strictly larger than probe_duration")
    if (
        min(
            values[name]
            for name in (
                "phase_increment",
                "form_error_bound",
                "phase_error_bound",
                "readout_error_bound",
            )
        )
        < 0
    ):
        raise ValueError("phase increment and error bounds must be nonnegative")
    if values["radius"] <= 0:
        raise ValueError("radius must be strictly positive")

    reference = assess_sine_formed_class_response(**values)
    eta = reference.formation_certificate.eta_bounds
    gap, rate = Q(1, 5), Q(2)
    coercivities = reference.coercivity_lower_bounds_by_class
    cosine = (
        min(coercivities) / gap
        if all(value is not None and value > 0 for value in coercivities)
        else None
    )
    mu, upper_stiffness, lower_position, upper_position, lyapunov_rate = (
        _sine_lyapunov_coefficients(
            eta_lower=eta.lo,
            eta_upper=eta.hi,
            cosine_lower=cosine,
            gap_lower=gap,
            rate_upper=rate,
        )
    )
    forms = reference.warmup_form_norm_upper_bounds
    phases = reference.warmup_target_phase_radius_upper_bounds
    initial_values = tuple(
        _sine_lyapunov_initial_upper(
            eta_upper=eta.hi,
            rate_upper=rate,
            gap_lower=gap,
            position_upper=upper_position,
            form_norm_upper=x,
            phase_norm_upper=theta,
        )
        for x, theta in zip(forms, reference.post_probe_phase_radius_upper_bounds)
    )
    thresholds = decay = returned = form_squares = phase_squares = None
    energy_margins = form_margins = phase_margins = None
    flags = (False, False)
    trapped = all(reference.recovery_certified_by_class)
    if lyapunov_rate is not None and trapped:
        exponent = lyapunov_rate * dwell
        if exponent > _MAX_RATIONAL_EXPONENT:
            raise ValueError("lyapunov_decay_rate*common_dwell must not exceed4096")
        decay = I(*_negative_exp_bounds(exponent))
        thresholds = tuple(
            min(eta.lo * gap * x**2 / 16, lower_position * theta**2 / (4 * rate))
            for x, theta in zip(forms, phases)
        )
        returned_rows = tuple(
            _sine_lyapunov_return_squared(
                initial_upper=value,
                decay_upper=decay.hi,
                eta_lower=eta.lo,
                gap_lower=gap,
                rate_upper=rate,
                position_lower=lower_position,
            )
            for value in initial_values
        )
        returned, form_squares, phase_squares = tuple(zip(*returned_rows))
        energy_margins = tuple(I(q - value) for q, value in zip(thresholds, returned))
        form_margins = tuple(
            I(x**2 / 4 - value) for x, value in zip(forms, form_squares)
        )
        phase_margins = tuple(
            I(theta**2 / 4 - value) for theta, value in zip(phases, phase_squares)
        )
        flags = tuple(
            form_margins[i].lo > 0 and phase_margins[i].lo > 0 for i in (0, 1)
        )
    reasons = tuple(
        reason
        for condition, reason in (
            (
                reference.status == "certified_formed_class_response",
                "formed_class_response_prerequisite_unavailable",
            ),
            (lyapunov_rate is not None, "positive_common_acute_decay_rate_unavailable"),
            (trapped, "post_probe_recovery_prerequisite_unavailable"),
        )
        if not condition
    ) + tuple(
        f"{name}:strict_half_neighborhood_return_not_certified"
        for name, flag in zip(("winding_one", "winding_two"), flags)
        if not flag
    )
    return SineFormedClassMaintenance(
        reference_certificate=reference,
        common_dwell=dwell,
        original_common_dwell=dwell / reference.formation_certificate.form_loss,
        cosine_lower_bound=cosine,
        gamma_squared_bounds=eta,
        phase_stiffness_lower_bound=mu,
        phase_stiffness_upper_bound=upper_stiffness,
        lyapunov_position_lower_coefficient=lower_position,
        lyapunov_position_upper_coefficient=upper_position,
        lyapunov_decay_rate=lyapunov_rate,
        post_probe_lyapunov_upper_bounds=initial_values,
        return_energy_thresholds=thresholds,
        exponential_decay_bounds=decay,
        returned_lyapunov_upper_bounds=returned,
        returned_form_norm_squared_upper_bounds=form_squares,
        returned_phase_norm_squared_upper_bounds=phase_squares,
        return_energy_margin_bounds=energy_margins,
        form_return_margin_bounds=form_margins,
        phase_return_margin_bounds=phase_margins,
        return_certified_by_class=flags,
        status="certified_repeated_probe_maintenance" if not reasons else "unavailable",
        unavailable_reasons=reasons,
    )
