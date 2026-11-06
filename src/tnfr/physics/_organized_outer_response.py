"""Finite supplied-probe response of the two acquired outer organizations.

ORGANIZED_OUTER_RESPONSE.md owns the frozen protocol and full nonlinear
argument. Every assessment rebuilds the original acquisition, including its
actual nonstationary zero-form control. No trajectory, source reset, runtime
operator or autonomous event selector is installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from ..mathematics._rational_interval import I, cos, pi_interval, sin
from ._organized_outer_winding import (
    _assess_organized_outer_winding,
    _OrganizedOuterWinding,
)
from ._repeated_organized_composition import _vector
from ._sine_preparation import _prepared_degree_four_probe_remainder

_EPSILON = Q(1, 1000)
_DELAY = Q(1, 100)
_ALLOWANCE = Q(1, 10**10)
_RAW_ERROR = Q(1, 10**12)
_MINIMUM_CONTRAST = Q(1, 10**10)
_WORK_BUDGET = Q(1, 1000)
_PHASE_JUMP = tuple(_EPSILON * (int(node // 9 == 0) - Q(1, 5)) for node in range(45))


def _apply_outer_response_probe(*, form, phase_offsets):
    """Apply the declared detached jump after admitting both complete rows.

    Signed forms, every internal phase gap and the global lifted-phase mean
    are preserved. Admission of the source, signed work, sector and later
    evolution belongs to the assessor, not to this algebraic intervention.
    """
    values = _vector(form, "form", 45)
    offsets = _vector(phase_offsets, "phase_offsets", 45)
    return values, tuple(value + jump for value, jump in zip(offsets, _PHASE_JUMP))


def _outer_response_event_work_bounds(*, opening_gap_bounds, closing_gap_bounds):
    """Enclose signed work on all eighteen changed outer edges.

    The supplied actual gap intervals apply to every slot of the marked
    opening 0->1 and closing 4->0 interfaces. The trigonometric identity
    retains the cancellation between interfaces before interval evaluation.
    It admits neither source provenance nor occurrence of the supplied jump.
    """
    opening = I.coerce(opening_gap_bounds)
    closing = I.coerce(closing_gap_bounds)
    return 9 * (
        (1 - cos(I(_EPSILON))) * (cos(opening) + cos(closing))
        + sin(I(_EPSILON)) * (sin(closing) - sin(opening))
    )


def _outer_response_finite_remainder(*, gamma_upper, initial_form_upper_bound):
    """Bound the fixed finite increment after the caller admits the full law.

    The proof uses all degree-four rows, heat contraction, the global sine
    derivative bound and the declared jump. Both background motion and
    nonlinear probe feedback are retained. This is not an ODE approximation
    or a bound transferred from a tangent model.
    """
    return _prepared_degree_four_probe_remainder(
        gamma_upper=gamma_upper,
        initial_form_upper_bound=initial_form_upper_bound,
        phase_jump_upper_bound=_EPSILON,
        elapsed_time=_DELAY,
    )


@dataclass(frozen=True)
class _OrganizedOuterResponse:
    """Actual-source finite response with separate work and identity decisions."""

    form_error_bounds: tuple[Q, ...]
    phase_error_bounds: tuple[Q, ...]
    acquisition: _OrganizedOuterWinding
    source_full_form_error_upper_bound: Q
    source_full_phase_error_upper_bound: Q
    source_outer_gap_bounds: tuple[tuple[I, ...], tuple[I, ...]]
    probed_outer_gap_bounds: tuple[tuple[I, ...], tuple[I, ...]]
    outer_one_source_form_upper_bound: Q
    outer_zero_source_form_upper_bound: Q
    outer_one_event_work_bounds: I
    outer_zero_event_work_bounds: I
    outer_one_event_work_absolute_upper_bound: Q
    outer_zero_event_work_absolute_upper_bound: Q
    outer_one_post_probe_excess_upper_bound: Q
    outer_zero_post_probe_excess_upper_bound: Q
    outer_one_post_probe_capture_margin: Q
    outer_zero_post_probe_capture_margin: Q
    outer_one_response_slope_bounds: I
    outer_zero_response_slope_bounds: I
    outer_one_finite_remainder_upper_bound: Q
    outer_zero_finite_remainder_upper_bound: Q
    outer_one_response_bounds: I
    outer_zero_response_bounds: I
    contrast_bounds: I
    observed_contrast_bounds: I
    response_contrast_margin: Q
    internal_positive_gap_margin: Q
    internal_acute_gap_margin: Q
    outer_one_post_probe_acute_margin: Q
    outer_zero_post_probe_acute_margin: Q
    finite_comparison_domain_margin: Q
    sufficient_margins: tuple[tuple[str, Q], ...]
    source_certified: bool
    conditional_response_certified: bool
    work_certified: bool
    child_identities_retained: bool
    outer_identities_retained: bool
    finite_response_certified: bool
    contrast_certified: bool
    post_probe_capture_certified: bool
    subsequent_maintenance_certified: bool
    admitted: bool
    status: str
    unresolved_conditions: tuple[str, ...]
    source_organization_order: tuple[int, int] = (1, 0)
    receiver_child: int = 1
    probe_phase_increments: tuple[Q, ...] = _PHASE_JUMP
    probe_amplitude: Q = _EPSILON
    source_scaled_time: Q = Q(200)
    elapsed_probe_scaled_time: Q = _DELAY
    elapsed_probe_original_time: Q = _DELAY * Q(1024, 1023)
    raw_readout_error_upper_bound: Q = _RAW_ERROR
    raw_readout_error_contributions: tuple[Q, ...] = (_RAW_ERROR,) * 4
    increment_readout_error_upper_bound: Q = 2 * _RAW_ERROR
    contrast_readout_error_upper_bound: Q = 4 * _RAW_ERROR
    minimum_observed_contrast: Q = _MINIMUM_CONTRAST
    probe_work_budget: Q = _WORK_BUDGET
    protocol_revision: str = "50ce12a5"
    clock: str = "tau=(1023/1024)*t; h=tau-200"
    scope: tuple[str, ...] = (
        "all_ten_original_allowances_admitted_before_fresh_source_rebuilding",
        "actual_outer_one_and_zero_sources_at_two_hundred_not_limiting_targets",
        "same_original_isolated_errors_not_identical_later_reached_fine_states",
        "control_fine_errors_transported_to_readout_not_assumed_stationary",
        "all_ninety_continuous_rows_and_eighty_original_relative_directions",
        "zero_total_child_uniform_phase_jump_with_exact_unchanged_forms_and_support",
        "signed_work_of_both_changed_interfaces_without_credit_for_previous_loss",
        "source_preserving_probed_minus_unprobed_comparison_for_each_organization",
        "finite_nonlinear_increment_retains_background_motion_and_probe_feedback",
        "four_arbitrary_raw_readout_errors_without_independence_assumption",
        "all_five_internal_and_nine_respective_outer_windings_retained",
        "complete_post_jump_storage_gives_separate_all_future_maintenance",
        "organization_contrast_does_not_identify_winding_as_a_sufficient_state",
        "no_trajectory_frozen_producer_operator_selection_or_physical_identification",
        "no_arbitrary_scale_fractality_or_resonance_spectrum_claim",
    )


def _assess_organized_outer_response(*, form_error_bounds, phase_error_bounds):
    """Assess the fixed probe on freshly rebuilt original source families.

    Smaller primitive allowances select subsets of the frozen source family.
    A failed sufficient condition remains unavailable and does not establish
    failure of response, formation or identity. No caller report or tunable
    amplitude, delay, work budget or resolution replaces this protocol.
    """
    rx = _vector(form_error_bounds, "form_error_bounds", 5)
    rt = _vector(phase_error_bounds, "phase_error_bounds", 5)
    if min(*rx, *rt) < 0:
        raise ValueError("original preparation errors must be nonnegative")
    if max(*rx, *rt) > _ALLOWANCE:
        raise ValueError("original preparation errors must not exceed frozen 1e-10")
    acquisition = _assess_organized_outer_winding(
        form_error_bounds=rx, phase_error_bounds=rt
    )
    source_ok = (
        acquisition.admitted
        and acquisition.capture_certified
        and acquisition.control.capture_certified
        and acquisition.fine_error_certified
    )
    gamma = acquisition.geometry.gamma_bounds
    form_error = acquisition.full_form_error_upper_bound
    phase_error = acquisition.full_phase_error_upper_bound
    one_gaps = acquisition.endpoint_oriented_outer_phase_gap_bounds
    # The same initial full-norm allowances and global heat comparison apply
    # around the stationary nominal control. Its actual residuals still move.
    zero_gaps = (I(-2 * phase_error, 2 * phase_error),) * 45
    changes = tuple(
        _PHASE_JUMP[b] - _PHASE_JUMP[a] for a, b, _ in acquisition.geometry.outer_edges
    )
    one_probed = tuple(gap + change for gap, change in zip(one_gaps, changes))
    zero_probed = tuple(gap + change for gap, change in zip(zero_gaps, changes))
    one_work = _outer_response_event_work_bounds(
        opening_gap_bounds=one_gaps[0], closing_gap_bounds=one_gaps[4]
    )
    zero_work = _outer_response_event_work_bounds(
        opening_gap_bounds=zero_gaps[0], closing_gap_bounds=zero_gaps[4]
    )
    one_excess = acquisition.endpoint_excess_upper_bound + one_work.hi
    zero_excess = acquisition.control.initial_excess_upper_bound + zero_work.hi
    one_capture = acquisition.boundary_energy_lower_bound - one_excess
    zero_capture = acquisition.control.boundary_energy_lower_bound - zero_excess
    pi = pi_interval()
    inner = acquisition.transit_internal_phase_gap_bounds
    inner_positive = inner.lo
    inner_acute = pi.lo / 2 - inner.hi
    one_acute = pi.lo / 2 - max(box.abs_max for box in one_probed)
    zero_acute = pi.lo / 2 - max(box.abs_max for box in zero_probed)
    one_x = (
        max(box.abs_max for box in acquisition.primitive_entry.endpoint_form_bounds)
        + form_error
    )
    zero_x = form_error
    finite_domain = Q(1, 2) - 2 * gamma.hi**2 * _DELAY**2

    # Receiver one has one changed incident gap. The shared identity avoids
    # subtracting two interval sine values with the same uncertain argument.
    def slope(opening):
        return gamma * sin(I(_EPSILON / 2)) * cos(opening - _EPSILON / 2) / 2

    one_slope, zero_slope = slope(one_gaps[0]), slope(zero_gaps[0])
    one_remainder = _outer_response_finite_remainder(
        gamma_upper=gamma.hi, initial_form_upper_bound=one_x
    )
    zero_remainder = _outer_response_finite_remainder(
        gamma_upper=gamma.hi, initial_form_upper_bound=zero_x
    )
    one_response = _DELAY * one_slope + I(-one_remainder, one_remainder)
    zero_response = _DELAY * zero_slope + I(-zero_remainder, zero_remainder)
    contrast = zero_response - one_response
    observed = contrast + I(-4 * _RAW_ERROR, 4 * _RAW_ERROR)
    contrast_margin = observed.lo - _MINIMUM_CONTRAST
    work_margins = (
        ("outer_one_absolute_probe_work", _WORK_BUDGET - one_work.abs_max),
        ("outer_zero_absolute_probe_work", _WORK_BUDGET - zero_work.abs_max),
    )
    identity_margins = (
        ("internal_positive_gap", inner_positive),
        ("internal_acute_gap", inner_acute),
        ("outer_one_post_probe_acute_cell", one_acute),
        ("outer_zero_post_probe_acute_cell", zero_acute),
        ("outer_one_complete_post_probe_capture", one_capture),
        ("outer_zero_complete_post_probe_capture", zero_capture),
    )
    margins = (
        work_margins
        + identity_margins
        + (
            ("finite_nonlinear_comparison_domain", finite_domain),
            ("resolved_observed_finite_response_contrast", contrast_margin),
        )
    )
    work_ok = all(value > 0 for _, value in work_margins)
    identity_ok = all(value > 0 for _, value in identity_margins)
    conditional = all(value > 0 for _, value in margins)
    source_conditions = (
        ("original_outer_source_not_admitted", acquisition.admitted),
        ("actual_outer_one_capture_not_certified", acquisition.capture_certified),
        (
            "actual_outer_zero_capture_not_certified",
            acquisition.control.capture_certified,
        ),
        ("both_actual_source_errors_not_certified", acquisition.fine_error_certified),
    )
    unresolved = tuple(
        name for name, passed in source_conditions if not passed
    ) + tuple(name for name, value in margins if value <= 0)
    admitted = source_ok and conditional
    return _OrganizedOuterResponse(
        form_error_bounds=rx,
        phase_error_bounds=rt,
        acquisition=acquisition,
        source_full_form_error_upper_bound=form_error,
        source_full_phase_error_upper_bound=phase_error,
        source_outer_gap_bounds=(one_gaps, zero_gaps),
        probed_outer_gap_bounds=(one_probed, zero_probed),
        outer_one_source_form_upper_bound=one_x,
        outer_zero_source_form_upper_bound=zero_x,
        outer_one_event_work_bounds=one_work,
        outer_zero_event_work_bounds=zero_work,
        outer_one_event_work_absolute_upper_bound=one_work.abs_max,
        outer_zero_event_work_absolute_upper_bound=zero_work.abs_max,
        outer_one_post_probe_excess_upper_bound=one_excess,
        outer_zero_post_probe_excess_upper_bound=zero_excess,
        outer_one_post_probe_capture_margin=one_capture,
        outer_zero_post_probe_capture_margin=zero_capture,
        outer_one_response_slope_bounds=one_slope,
        outer_zero_response_slope_bounds=zero_slope,
        outer_one_finite_remainder_upper_bound=one_remainder,
        outer_zero_finite_remainder_upper_bound=zero_remainder,
        outer_one_response_bounds=one_response,
        outer_zero_response_bounds=zero_response,
        contrast_bounds=contrast,
        observed_contrast_bounds=observed,
        response_contrast_margin=contrast_margin,
        internal_positive_gap_margin=inner_positive,
        internal_acute_gap_margin=inner_acute,
        outer_one_post_probe_acute_margin=one_acute,
        outer_zero_post_probe_acute_margin=zero_acute,
        finite_comparison_domain_margin=finite_domain,
        sufficient_margins=margins,
        source_certified=source_ok,
        conditional_response_certified=conditional,
        work_certified=source_ok and work_ok,
        child_identities_retained=source_ok and identity_ok,
        outer_identities_retained=source_ok and identity_ok,
        finite_response_certified=source_ok and finite_domain > 0,
        contrast_certified=source_ok and finite_domain > 0 and contrast_margin > 0,
        post_probe_capture_certified=source_ok and identity_ok,
        subsequent_maintenance_certified=source_ok and identity_ok,
        admitted=admitted,
        status="certified_organized_outer_response" if admitted else "unavailable",
        unresolved_conditions=unresolved,
    )
