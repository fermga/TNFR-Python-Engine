"""Finite constitutive discrimination from actually acquired C9 objects.

The proof owner is ORGANIZED_PATTERN_CONSTITUTIVE_FREEDOM.md, section
organized-pattern-finite-carried-response. Every call rebuilds both laws'
acquisition, recovery and event bounds. Static onset algebra and an own-law
real nonlinear remainder enclose a prospective observation; no response,
trajectory, time-series producer or frozen evidence is evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ._sine_class_constitutive_comparison import (
    _ConstitutiveOnset,
    _current_derivative_bound,
    _derive_constitutive_onset,
)
from ._sine_constitutive_handoff import (
    _assess_constitutive_handoff,
    _ConstitutiveHandoff,
)

_AMPLITUDE = Q(7, 10000)
_HORIZON = Q(1, 10**6)
_WINDOW = Q(1, 100)
_GAMMA_UPPER = Q(1, 3000)
_GAMMA_LOWER = Q(1, 3300)


@dataclass(frozen=True)
class _CarriedLawObservation:
    """Exact rational record bounds conditional on the fresh handoff.

    Bounds below the shared interval grid remain numerator/denominator pairs
    of rational endpoints. The four raw readings each have the supplied
    absolute error; no independence or cancellation of those errors is assumed.
    Candidate formulas on failed premises do not certify a reached record.
    """

    deformation: Q
    handoff: _ConstitutiveHandoff
    onset: _ConstitutiveOnset
    current_fourth_derivative_upper_bound: Q
    nominal_bound_margins: tuple[tuple[str, Q], ...]
    nominal_bound_admitted: bool
    leading_response_bounds: tuple[Q, Q]
    nonlinear_remainder_upper_bound: Q
    carried_state_error_upper_bound: Q
    mixed_readout_error_upper_bound: Q
    nominal_response_bounds: tuple[Q, Q]
    actual_response_bounds: tuple[Q, Q]
    measured_response_bounds: tuple[Q, Q]
    signed_measured_margin: Q
    certified: bool


@dataclass(frozen=True)
class _CarriedConstitutiveDiscrimination:
    """Prospective separation of complete actual-source record sets.

    This is an analytic admission with an identity observation map, not a
    reserved response, sensor calibration or laboratory feasibility claim.
    The same law-independent preparation description and supplied events
    are used in both models; reached states and event work need not coincide.
    """

    form_error_bound: Q
    phase_error_bound: Q
    per_reading_error_bound: Q
    observations: tuple[_CarriedLawObservation, _CarriedLawObservation]
    common_nominal_magnitude_lower_bound: Q
    record_separation_lower_bound: Q
    status: str
    unresolved_conditions: tuple[str, ...]
    deformation_order: tuple[int, int] = (0, 1)
    amplitude: Q = _AMPLITUDE
    observation_duration: Q = _HORIZON
    original_observation_duration: Q = _HORIZON * Q(1024, 1023)
    readout_node: int = 13
    observation: str = "x13_both-x13_donor-x13_receiver+x13_neither"
    clock: str = "tau=(1023/1024)*t"
    scope: tuple[str, ...] = (
        "original_class_one_sources_with_all_zero_sum_latent_errors_retained",
        "fresh_own_law_acquisition_and_uninterrupted_recovery_without_reset",
        "supplied_two_bridge_attachment_and_four_paired_form_input_histories",
        "actual_support_and_input_work_with_retained_degree_weighted_means",
        "full_scaled_state_used_only_for_real_remainder_inequalities",
        "own_harmonic_derivatives_not_transferred_sine_complex_tails",
        "exact_static_onset_and_full_nonlinear_finite_time_remainder",
        "arbitrary_carried_symmetry_breaking_errors_bounded_by_global_stability",
        "four_raw_readout_errors_added_without_independence_assumptions",
        "identity_form_observation_with_supplied_mathematical_error_allowance",
        "no_trajectory_coefficient_producer_or_reserved_response_evaluation",
        "no_physical_sensor_clock_bridge_or_experimental_law_selection",
    )


def _assess_carried_constitutive_discrimination(
    *, form_error_bound, phase_error_bound, per_reading_error_bound
) -> _CarriedConstitutiveDiscrimination:
    """Admit raw budgets before rebuilding the fixed two-law comparison."""
    rx = _exact(form_error_bound, "form_error_bound")
    rt = _exact(phase_error_bound, "phase_error_bound")
    readout = _exact(per_reading_error_bound, "per_reading_error_bound")
    if min(rx, rt, readout) < 0:
        raise ValueError("source and reading error bounds must be nonnegative")

    rows = []
    reasons = []
    for deformation in (Q(0), Q(1)):
        handoff = _assess_constitutive_handoff(
            deformation=deformation, form_error_bound=rx, phase_error_bound=rt
        )
        onset = _derive_constitutive_onset(
            mediator_class=1,
            deformation=deformation,
            donor_amplitude=_AMPLITUDE,
            receiver_amplitude=_AMPLITUDE,
        )
        gamma = onset.geometry.gamma_bounds
        fourth = _current_derivative_bound(deformation, 4)
        # Non-strict jet bounds include the exactly saturated contact jet.
        jets_ok = (
            all(
                max(abs(value.lo), abs(value.hi)) <= 1
                for value in onset.target_current_third_derivative_bounds
            )
            and fourth <= 28
        )
        margins = (
            ("positive_horizon", _HORIZON),
            ("within_real_remainder_window", _WINDOW - _HORIZON),
            ("gamma_upper", _GAMMA_UPPER - gamma.hi),
            ("gamma_lower", gamma.lo - _GAMMA_LOWER),
            (
                "target_cosine_upper",
                Q(4, 5) - onset.geometry.channel_cosine_bounds[1].hi,
            ),
            ("scaled_field_lipschitz", 3 - (2 + 2 * gamma.hi**2 * Q(245, 243))),
            ("unscaled_field_lipschitz", 3 - (2 + 2 * gamma.hi * Q(245, 243))),
            ("carried_stability_factor", Q(1, 2) - 3 * _HORIZON),
            ("branch_tail_coefficient", 1 - 106 * gamma.hi * _AMPLITUDE),
            (
                "common_positive_signal_coefficient",
                Q(1, 2560) - 36 * _HORIZON - Q(1, 3000),
            ),
        )
        admitted = jets_ok and all(value > 0 for _, value in margins)
        leading = tuple(
            value * _HORIZON**4 for value in onset.mixed_time_four_coefficient_bounds
        )
        tail = 36 * gamma.hi**4 * _AMPLITUDE**3 * _HORIZON**5
        carried = 8 * handoff.endpoint_radius
        reading = 4 * readout
        nominal = leading[0] - tail, leading[1] + tail
        actual = nominal[0] - carried, nominal[1] + carried
        measured = actual[0] - reading, actual[1] + reading
        signed = -measured[1] if deformation == 0 else measured[0]
        certified = (
            admitted
            and handoff.status == "certified_constitutive_handoff"
            and signed > 0
        )
        if not admitted:
            reasons.append(f"lambda_{int(deformation)}_nominal_remainder_unavailable")
        if handoff.status != "certified_constitutive_handoff":
            reasons.append(f"lambda_{int(deformation)}_actual_handoff_unavailable")
        if signed <= 0:
            reasons.append(
                f"lambda_{int(deformation)}_strict_measured_sign_unavailable"
            )
        rows.append(
            _CarriedLawObservation(
                deformation=deformation,
                handoff=handoff,
                onset=onset,
                current_fourth_derivative_upper_bound=fourth,
                nominal_bound_margins=margins,
                nominal_bound_admitted=admitted,
                leading_response_bounds=leading,
                nonlinear_remainder_upper_bound=tail,
                carried_state_error_upper_bound=carried,
                mixed_readout_error_upper_bound=reading,
                nominal_response_bounds=nominal,
                actual_response_bounds=actual,
                measured_response_bounds=measured,
                signed_measured_margin=signed,
                certified=certified,
            )
        )
    separation = (
        rows[1].measured_response_bounds[0] - rows[0].measured_response_bounds[1]
    )
    return _CarriedConstitutiveDiscrimination(
        form_error_bound=rx,
        phase_error_bound=rt,
        per_reading_error_bound=readout,
        observations=tuple(rows),
        common_nominal_magnitude_lower_bound=_AMPLITUDE**3
        * _HORIZON**4
        / (3000 * 3300**4),
        record_separation_lower_bound=separation,
        status=(
            "certified_opposite_carried_responses"
            if all(row.certified for row in rows)
            else "unavailable"
        ),
        unresolved_conditions=tuple(reasons),
    )
