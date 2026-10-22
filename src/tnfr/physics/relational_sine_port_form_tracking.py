"""Cancellation-aware all-time form bounds for the same reduced C9 network.

The fixed-reference heat filters retain the fast/slow cancellation before
taking norms. Neither commuting operators nor a derivative of the unknown
forcing is assumed. Original preparations and phase bounds are rebuilt.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from ..mathematics._rational_interval import INTERVAL_METHOD, I, sqrt
from .relational_sine_port_relaxation import (
    SinePortRelaxation,
    assess_sine_port_relaxation,
)

__all__ = ("SinePortFormTracking", "assess_sine_port_form_tracking")


@dataclass(frozen=True)
class _HeatFormBounds:
    """Rational filter gains on one admitted centered parity subspace."""

    gap_lower_bound: Q
    spectral_lower_bound: Q
    spectral_upper_bound: Q
    spectral_ratio_upper_bound: Q
    dyadic_band_count: int
    heat_integral_upper_bound: Q
    derivative_filter_gain_upper_bound: Q
    initial_norm_upper_bound: Q
    forcing_upper_bound: Q
    loop_gain_upper_bound: Q
    loop_margin: Q
    initial_form_contribution_upper_bound: Q
    initial_joint_contribution_upper_bound: Q
    forced_contribution_upper_bound: Q
    form_norm_upper_bound: Q | None
    certified: bool


def _heat_form_bounds(*, gap, initial_norm, forcing, gamma_upper):
    """Use admitted rational spectral and forcing bounds, without a report.

    For mu<=B0<=M on the mean-free subspace, integral ||B0 exp(-B0 t)||
    is at most 1+log(M/mu)/exp(1). The exact dyadic count, log(2)<7/10 and
    1/exp(1)<3/8 give the rational bound below. The form-scale gamma cancels
    symbolically before interval endpoints are substituted.
    """
    lower, upper = gap / 6, Q(2)
    ratio = upper / lower
    bands = max(0, ratio.numerator.bit_length() - ratio.denominator.bit_length())
    if Q(2**bands) < ratio:
        bands += 1
    heat = 1 + Q(21, 80) * bands
    gain = 1 + heat
    loop = gamma_upper**2 * upper * gain / gap
    margin = 1 - loop
    initial_joint = gamma_upper * upper * (1 + gamma_upper) * initial_norm / gap
    forced = gamma_upper * gain * forcing / gap
    form = (initial_norm + initial_joint + forced) / margin if margin > 0 else None
    return _HeatFormBounds(
        gap_lower_bound=gap,
        spectral_lower_bound=lower,
        spectral_upper_bound=upper,
        spectral_ratio_upper_bound=ratio,
        dyadic_band_count=bands,
        heat_integral_upper_bound=heat,
        derivative_filter_gain_upper_bound=gain,
        initial_norm_upper_bound=initial_norm,
        forcing_upper_bound=forcing,
        loop_gain_upper_bound=loop,
        loop_margin=margin,
        initial_form_contribution_upper_bound=initial_norm,
        initial_joint_contribution_upper_bound=initial_joint,
        forced_contribution_upper_bound=forced,
        form_norm_upper_bound=form,
        certified=margin > 0,
    )


@dataclass(frozen=True)
class SinePortFormTracking:
    """Same-source phase evidence and a separate cancellation-aware form bound.

    The nested baseline is freshly reconstructed from all fourteen primitives.
    Its previous form envelope and verdict remain explicit. Mathematical channel
    flags are separate from the supplied-work policy required by overall status.
    No incoming baseline report or cached trajectory is accepted.
    """

    baseline_certificate: SinePortRelaxation
    bridge_hessian_variation_upper_bound: Q | None
    even_bridge_forcing_upper_bound: Q | None
    odd_heat_bounds: _HeatFormBounds | None
    even_heat_bounds: _HeatFormBounds | None
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
    interval_method: str = INTERVAL_METHOD
    method: str = (
        "fixed_reference_heat_filter_total_variation_and_finite_supremum_small_gain"
    )
    scope: tuple[str, ...] = (
        "same_complete_law_support_surrogate_source_and_resolution_primitives",
        "fresh_baseline_phase_parity_bounds_not_incoming_report_premises",
        "fixed_reference_phase_hessian_internal_cosines_and_unit_bridge_weights",
        "noncommuting_diffusion_and_phase_operators_keep_convolution_order",
        "time_varying_bridge_hessian_retained_as_explicit_even_forcing",
        "bounded_forcing_no_time_derivative_or_commuting_mode_assumption",
        "initial_form_and_joint_coordinate_exchange_retained",
        "uniform_all_time_fine_coordinate_envelopes_with_conserved_mean_floors",
        "declared_resolution_policy_not_observed_error_or_physical_sensor",
        "no_trajectory_solver_ideal_reset_probe_or_support_occurrence_rule",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {"schema": "tnfr.sine-port-form-tracking.v1", "report": _project(self)}


def assess_sine_port_form_tracking(
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
) -> SinePortFormTracking:
    """Rebuild the source and refine its all-time form comparison bound.

    All mandatory primitive domains and work caps are owned by
    assess_sine_port_relaxation and are applied before any new arithmetic. The
    same phase envelope and declared origin-span allowances are retained. New
    form gains require positive finite-supremum loop margins on both parities;
    missing hypotheses return unavailable rather than an inferred small error.
    """
    baseline = assess_sine_port_relaxation(
        classes=classes,
        contacts=contacts,
        phase_origins=phase_origins,
        formation_time=formation_time,
        relaxation_duration=relaxation_duration,
        form_error_bound=form_error_bound,
        phase_error_bound=phase_error_bound,
        endpoint_radius=endpoint_radius,
        radius=radius,
        decay_power=decay_power,
        work_allowance=work_allowance,
        normalized_gap_lower_bound=normalized_gap_lower_bound,
        phase_resolution_fraction=phase_resolution_fraction,
        form_resolution_fraction=form_resolution_fraction,
    )
    bridge_variation = bridge_forcing = odd = even = error = margin = None
    envelopes = form_resolved = False
    if baseline.all_time_envelopes_certified:
        # Odd port values vanish. Only the even comparison sees the change
        # from the exact bridge Hessian to its fixed unit-weight reference.
        bridge_variation = baseline.edge_disagreement_norm_squared_upper_bound
        bridge_forcing = bridge_variation * baseline.even_bounds.phase_norm_upper_bound
        odd = _heat_form_bounds(
            gap=baseline.odd_bounds.gap_lower_bound,
            initial_norm=baseline.odd_bounds.initial_norm_upper_bound,
            forcing=baseline.odd_bounds.forcing_upper_bound,
            gamma_upper=baseline.gamma_bounds.hi,
        )
        even = _heat_form_bounds(
            gap=baseline.even_bounds.gap_lower_bound,
            initial_norm=baseline.even_bounds.initial_norm_upper_bound,
            forcing=baseline.even_bounds.forcing_upper_bound + bridge_forcing,
            gamma_upper=baseline.gamma_bounds.hi,
        )
        if odd.certified and even.certified:
            error = (
                sqrt(I(odd.form_norm_upper_bound**2 + even.form_norm_upper_bound**2)).hi
                / sqrt(I(2)).lo
                + baseline.form_mean_error_floor
            )
            margin = baseline.form_allowance_bounds - error
            envelopes = True
            form_resolved = margin.lo > 0
    phase_resolved = baseline.phase_resolution_certified
    reasons = baseline.unavailable_reasons + (
        () if envelopes else ("heat_form_envelopes_not_certified",)
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
    return SinePortFormTracking(
        baseline_certificate=baseline,
        bridge_hessian_variation_upper_bound=bridge_variation,
        even_bridge_forcing_upper_bound=bridge_forcing,
        odd_heat_bounds=odd,
        even_heat_bounds=even,
        form_mean_error_floor=baseline.form_mean_error_floor,
        phase_mean_error_floor=baseline.phase_mean_error_floor,
        all_time_form_error_upper_bound=error,
        all_time_phase_error_upper_bound=baseline.all_time_phase_error_upper_bound,
        form_resolution_margin_bounds=margin,
        phase_resolution_margin_bounds=baseline.phase_resolution_margin_bounds,
        all_time_envelopes_certified=envelopes,
        form_resolution_certified=form_resolved,
        phase_resolution_certified=phase_resolved,
        joint_resolution_certified=phase_resolved and form_resolved,
        status=status,
        unavailable_reasons=reasons,
        resolution_limitations=limitations,
    )
