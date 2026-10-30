"""Source-inclusive parent registration on the fixed matching 36-node law.

ORGANIZED_PARENT_ACQUISITION.md owns the frozen protocol and analytic proof.
The actual fine source is compared with a separable two-pendulum family;
neither a numerical trajectory nor a reset to that family is performed.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._rational_interval import I, atan, cos, pi_interval, sin
from ._repeated_organized_composition import (
    _derive_repeated_composition,
    _RepeatedCompositionGeometry,
    _vector,
)
from ._sine_constitutive_recovery import (
    _assess_constitutive_recovery,
    _ConstitutiveRecovery,
)
from ._sine_preparation import _prepared_heat_error_radii
from .reversible_eigenmode_reference import _negative_exp_bounds

_RECOVERED_RADIUS = Q(1, 10**70)
_ENTRY_TIME, _READOUT_TIME = Q(4), Q(5)
_INNER_GAP, _OUTER_GAP = Q(1, 2), Q(1, 20)
_RATE = Q(23, 24)
_RESIDUAL_RADIUS = Q(1, 100)
_IDENTITY_GAP = Q(1, 4)
_PREDICTION_ERROR = Q(1, 10**5)
_RESOLVED_DECLINE = Q(1, 10**4)
_ORIGINS = tuple(
    Q(2 * parent - 1, 40) + Q(2 * child - 1, 4)
    for parent in range(2)
    for child in range(2)
)


def _predict_parent_gap(phase_gap) -> I:
    """Enclose ``2*atan(exp(-1)*tan(A/2))`` for an admitted ``0<=A<=1/20``.

    This is detached scalar arithmetic. It certifies no preparation, reached
    state or response; the acquisition assessment owns its source-specific
    error over the actual slow-time interval from four to five.
    """
    value = _exact(phase_gap, "phase_gap")
    if not 0 <= value <= _OUTER_GAP:
        raise ValueError("phase_gap requires 0 <= phase_gap <= 1/20")
    if value == 0:
        return I(0)
    half = I(value / 2)
    return 2 * atan(I(*_negative_exp_bounds(Q(1))) * sin(half) / cos(half))


def _nominal_gap_bounds(initial, slow_time, eta_upper):
    """Evaluate the proved fixed-source pendulum envelopes, not its trajectory."""
    lower = _negative_exp_bounds((1 + 2 * eta_upper) * slow_time)[0]
    upper = _negative_exp_bounds(_RATE * (slow_time - eta_upper))[1]
    return I(initial * lower, initial * upper)


@dataclass(frozen=True)
class _OrganizedParentAcquisition:
    """Fresh preparation, conditional bounds and source-inclusive decisions.

    Conditional candidates remain visible when a source or strict margin is
    unavailable. Tiny residual and work allowances remain exact rationals;
    interval-rounded nominal storage does not replace those allowances.
    """

    form_error_bounds: tuple[Q, ...]
    phase_error_bounds: tuple[Q, ...]
    child_sources: tuple[_ConstitutiveRecovery, ...]
    geometry: _RepeatedCompositionGeometry
    relative_preparation_dimension: int
    original_preparation_storage_upper_bound: Q
    full_phase_error_upper_bound: Q
    full_form_error_upper_bound: Q
    form_laplacian_gap_lower_bound: Q
    nominal_inner_gap_at_entry_bounds: I
    nominal_outer_gap_at_entry_bounds: I
    nominal_outer_gap_at_readout_bounds: I
    nominal_outer_gap_ratio_upper_bound: Q
    actual_outer_gap_at_entry_bounds: I
    actual_outer_gap_at_readout_bounds: I
    parent_form_residual_upper_bound: Q
    parent_phase_residual_upper_bound: Q
    parent_gap_upper_bound: Q
    parent_form_mean_absolute_upper_bound_at_entry: Q
    initial_unregistered_margin: Q
    child_positive_gap_margin: Q
    child_acute_gap_margin: Q
    outer_gap_decline_lower_bound: Q
    phase_prediction_error_upper_bound: Q
    parent_law_phase_error_upper_bound: Q
    attachment_nominal_work_bounds: I
    attachment_residual_work_bounds: tuple[Q, Q]
    attachment_work_bounds: I
    complete_initial_nominal_storage_bounds: I
    complete_initial_storage_residual_bounds: tuple[Q, Q]
    complete_initial_storage_bounds: I
    elapsed_scaled_acquisition_time_bounds: I
    elapsed_scaled_readout_time_bounds: I
    elapsed_original_acquisition_time_bounds: I
    elapsed_original_readout_time_bounds: I
    antipodal_control_parent_gap_bounds: I
    antipodal_control_nominal_work_bounds: I
    antipodal_control_residual_work_bounds: tuple[Q, Q]
    sufficient_margins: tuple[tuple[str, Q], ...]
    sources_certified: bool
    source_initially_unregistered: bool
    conditional_entry_certified: bool
    conditional_prediction_certified: bool
    entry_certified: bool
    child_identities_retained: bool
    parent_identity_retained: bool
    prediction_certified: bool
    admitted: bool
    status: str
    unresolved_conditions: tuple[str, ...]
    recovered_child_norm_bound: Q = _RECOVERED_RADIUS
    acquisition_slow_time: Q = _ENTRY_TIME
    readout_slow_time: Q = _READOUT_TIME
    initial_child_phase_origins: tuple[Q, ...] = _ORIGINS
    held_capacities: tuple[Q, ...] = (Q(1),) * 36
    conserved_global_form_mean: Q = Q(0)
    conserved_global_phase_offset_mean: Q = Q(0)
    entry_residual_radius: Q = _RESIDUAL_RADIUS
    parent_identity_gap: Q = _IDENTITY_GAP
    prediction_error_budget: Q = _PREDICTION_ERROR
    minimum_resolved_decline: Q = _RESOLVED_DECLINE
    protocol_revision: str = "da12bdb6"
    clock: str = "tau=(1023/1024)*t; s=gamma^2*tau/2; s=0 at attachment"
    antipodal_control_scope: tuple[str, ...] = (
        "exact_pi_intraparent_gap_is_an_identity_not_inferred_from_interval_sine",
        "paired_identical_recovered_fine_children_within_each_parent",
        "correlated_source_not_the_four_independent_positive_source_families",
        "intraparent_antipodal_difference_invariant_for_all_future_times",
        "parent_registration_never_enters_the_quarter_radian_identity_domain",
        "child_winding_retention_only_through_slow_time_five_by_full_error_bound",
        "no_open_neighborhood_nonentry_or_antipodal_stability_claim",
    )
    scope: tuple[str, ...] = (
        "four_original_isolated_class_one_sources_rebuilt_from_eight_primitives",
        "each_recovered_child_retains_its_full_fine_uncertainty_and_zero_means",
        "supplied_origins_and_both_contact_layers_do_not_reset_recovered_state",
        "all_seventy_two_coordinates_retained_by_global_heat_integral_comparison",
        "only_global_means_conserved_after_attachment",
        "two_pendulum_family_is_a_reference_not_an_exact_reduction_of_the_source",
        "entry_and_parent_identity_retained_from_slow_time_four_through_five",
        "finite_prediction_uses_carried_source_correlation_not_a_generic_centiradian_ball",
        "nonlinear_parent_predictor_starts_from_actual_parent_forms_and_phase_gap",
        "static_attachment_work_is_not_funded_by_prior_isolated_dissipation",
        "failed_sufficient_bound_is_unavailable_not_a_nonentry_obstruction",
        "no_trajectory_event_selector_autonomous_assembly_or_physical_identification",
    )


def _assess_organized_parent_acquisition(
    *, form_error_bounds, phase_error_bounds
) -> _OrganizedParentAcquisition:
    """Rebuild the four children before assessing the unchanged analytic protocol.

    All eight finite nonnegative primitive allowances are admitted before the
    first recovery call. No report, alternative horizon, source offset or law
    can replace the fixed preparation and complete-law premises.
    """
    rx = _vector(form_error_bounds, "form_error_bounds", 4)
    rt = _vector(phase_error_bounds, "phase_error_bounds", 4)
    if min(*rx, *rt) < 0:
        raise ValueError("original preparation errors must be nonnegative")
    children = tuple(
        _assess_constitutive_recovery(
            deformation=0, form_error_bound=a, phase_error_bound=b
        )
        for a, b in zip(rx, rt)
    )
    geometry = _derive_repeated_composition(interface="matching")
    gamma, pi = geometry.gamma_bounds, pi_interval()
    eta = gamma**2
    radius = _RECOVERED_RADIUS
    source_flags = tuple(
        child.recovery_certified
        and child.endpoint_form_norm_squared_upper_bound is not None
        and child.endpoint_phase_norm_squared_upper_bound is not None
        and child.endpoint_form_norm_squared_upper_bound < radius**2
        and child.endpoint_phase_norm_squared_upper_bound < radius**2
        for child in children
    )
    source_ok = all(source_flags)
    # e^(4s) <= e^20 < 3^20 through s=5. Keep sub-grid source errors exact.
    form_error, phase_error = _prepared_heat_error_radii(
        initial_form_error=2 * radius,
        initial_phase_error=2 * radius,
        gamma_upper=gamma.hi,
        phase_growth_upper=Q(3**20),
        form_convolution_factor=Q(24),
    )
    inner = _nominal_gap_bounds(_INNER_GAP, _ENTRY_TIME, eta.hi)
    outer = _nominal_gap_bounds(_OUTER_GAP, _ENTRY_TIME, eta.hi)
    readout = _nominal_gap_bounds(_OUTER_GAP, _READOUT_TIME, eta.hi)
    a = 1 + 2 * eta.hi
    # Both envelopes share the same initial gap; retain that correlation.
    ratio_exponent = _RATE * (_READOUT_TIME - eta.hi) - a * _ENTRY_TIME
    ratio_upper = _negative_exp_bounds(ratio_exponent)[1]
    parent_phase = inner.hi / 2 + 2 * phase_error
    parent_form = gamma.hi * a * inner.hi / 2 + 2 * form_error
    parent_gap = inner.hi + 2 * phase_error
    actual_outer = I(outer.lo - 2 * phase_error, outer.hi + 2 * phase_error)
    actual_readout = I(readout.lo - 2 * phase_error, readout.hi + 2 * phase_error)
    unregistered = _INNER_GAP - 2 * radius - _IDENTITY_GAP
    child_positive = geometry.alpha_bounds.lo - 2 * phase_error
    child_acute = pi.lo / 2 - geometry.alpha_bounds.hi - 2 * phase_error
    decline = Q(1, 2400) - 4 * phase_error
    prediction_error = eta.hi / 10 + 4 * phase_error
    parent_law_error = 8 * phase_error + 6 * gamma.hi * form_error
    entry_margins = (
        ("initial_parent_registration_absent", unregistered),
        ("full_form_spectral_gap", (2 - 2 * cos(geometry.alpha_bounds)).lo - Q(1, 3)),
        ("pendulum_small_parameter", Q(1, 10**7) - eta.hi),
        ("child_positive_winding_gap", child_positive),
        ("child_acute_winding_gap", child_acute),
        ("parent_form_entry", _RESIDUAL_RADIUS - parent_form),
        ("parent_phase_entry", _RESIDUAL_RADIUS - parent_phase),
        ("maintained_parent_registration", _IDENTITY_GAP - parent_gap),
    )
    prediction_margins = (
        ("nominal_outer_gap_floor", outer.lo - Q(1, 1200)),
        ("nominal_outer_gap_halving", Q(1, 2) - ratio_upper),
        ("actual_prediction_argument_positive", actual_outer.lo),
        ("actual_prediction_argument_below_source_gap", _OUTER_GAP - actual_outer.hi),
        ("source_specific_phase_prediction", _PREDICTION_ERROR - prediction_error),
        ("actual_mean_parent_law_prediction", _PREDICTION_ERROR - parent_law_error),
        ("resolved_actual_phase_decline", decline - _RESOLVED_DECLINE),
    )
    conditional_entry = all(value > 0 for _, value in entry_margins)
    conditional_prediction = conditional_entry and all(
        value > 0 for _, value in prediction_margins
    )
    entry = source_ok and conditional_entry
    prediction = source_ok and conditional_prediction
    margins = entry_margins + prediction_margins
    unresolved = tuple(
        f"child_{index}_original_recovery_not_certified"
        for index, passed in enumerate(source_flags)
        if not passed
    ) + tuple(name for name, value in margins if value <= 0)
    work = 18 * (1 - cos(I(_INNER_GAP))) + 18 * (1 - cos(I(_OUTER_GAP)))
    work_residual = (Q(0), 16 * radius**2)
    initial_storage = 36 * (1 - cos(geometry.alpha_bounds)) + work
    storage_residual = (Q(0), 32 * radius**2)
    tau_entry, tau_readout = 2 * _ENTRY_TIME / eta, 2 * _READOUT_TIME / eta
    return _OrganizedParentAcquisition(
        form_error_bounds=rx,
        phase_error_bounds=rt,
        child_sources=children,
        geometry=geometry,
        relative_preparation_dimension=sum(
            child.source.preparation_dimension for child in children
        ),
        original_preparation_storage_upper_bound=sum(
            (child.source.initial_storage_upper_bounds[0] for child in children), Q(0)
        ),
        full_phase_error_upper_bound=phase_error,
        full_form_error_upper_bound=form_error,
        form_laplacian_gap_lower_bound=Q(1, 3),
        nominal_inner_gap_at_entry_bounds=inner,
        nominal_outer_gap_at_entry_bounds=outer,
        nominal_outer_gap_at_readout_bounds=readout,
        nominal_outer_gap_ratio_upper_bound=ratio_upper,
        actual_outer_gap_at_entry_bounds=actual_outer,
        actual_outer_gap_at_readout_bounds=actual_readout,
        parent_form_residual_upper_bound=parent_form,
        parent_phase_residual_upper_bound=parent_phase,
        parent_gap_upper_bound=parent_gap,
        parent_form_mean_absolute_upper_bound_at_entry=gamma.hi * a * outer.hi / 2
        + form_error,
        initial_unregistered_margin=unregistered,
        child_positive_gap_margin=child_positive,
        child_acute_gap_margin=child_acute,
        outer_gap_decline_lower_bound=decline,
        phase_prediction_error_upper_bound=prediction_error,
        parent_law_phase_error_upper_bound=parent_law_error,
        attachment_nominal_work_bounds=work,
        attachment_residual_work_bounds=work_residual,
        attachment_work_bounds=work + I(*work_residual),
        complete_initial_nominal_storage_bounds=initial_storage,
        complete_initial_storage_residual_bounds=storage_residual,
        complete_initial_storage_bounds=initial_storage + I(*storage_residual),
        elapsed_scaled_acquisition_time_bounds=tau_entry,
        elapsed_scaled_readout_time_bounds=tau_readout,
        elapsed_original_acquisition_time_bounds=tau_entry / Q(1023, 1024),
        elapsed_original_readout_time_bounds=tau_readout / Q(1023, 1024),
        antipodal_control_parent_gap_bounds=pi,
        antipodal_control_nominal_work_bounds=36 + 18 * (1 - cos(I(_OUTER_GAP))),
        antipodal_control_residual_work_bounds=(Q(0), 8 * radius**2),
        sufficient_margins=margins,
        sources_certified=source_ok,
        source_initially_unregistered=source_ok and unregistered > 0,
        conditional_entry_certified=conditional_entry,
        conditional_prediction_certified=conditional_prediction,
        entry_certified=entry,
        child_identities_retained=source_ok and child_positive > 0 and child_acute > 0,
        parent_identity_retained=entry,
        prediction_certified=prediction,
        admitted=prediction,
        status=(
            "certified_organized_parent_acquisition" if prediction else "unavailable"
        ),
        unresolved_conditions=unresolved,
    )
