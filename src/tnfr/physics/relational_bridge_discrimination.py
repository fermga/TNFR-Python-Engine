"""Finite bridge observations distinguish two supplied phase-storage laws.

The shared two-C6 owner admits the support, symbolic target, common nodal
preparation and exact response jets. Full-field derivative bounds turn the
same preparations into a finite observation budget without a trajectory,
nonlinear shell closure or installation of an alternative runtime law.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from typing import Any

from .._exact_time import exact_or_represented_real, exp_unit_bounds
from ..mathematics._interval_taylor import Jet, sinc
from ..mathematics._rational_interval import INTERVAL_METHOD, I, pi_interval
from .relational_observations import _ordered
from .relational_sine_comparison import SineExchangeComparison
from .relational_sine_resonance import (
    BridgeStorageFamilyAssessment,
    assess_bridge_storage_family,
)

__all__ = (
    "BridgeFiniteLawDiscrimination",
    "BridgeClockLawDiscrimination",
    "assess_bridge_finite_law_discrimination",
    "assess_bridge_clock_law_discrimination",
)


@dataclass(frozen=True)
class BridgeFiniteLawDiscrimination:
    """Analytic prediction intervals for two noisy finite form responses.

    Each experiment starts within preparation_error in every form and lifted
    phase coordinate of its own symbolic preparation. observation_error bounds
    each measured bridge-form difference, not each individual node sensor.
    The projected boxes enclose these exact cubes; their extra outward
    materialization width is not a newly admitted preparation tolerance.
    """

    source: SineExchangeComparison
    families: tuple[BridgeStorageFamilyAssessment, BridgeStorageFamilyAssessment]
    left_cycle: tuple[Any, ...]
    right_cycle: tuple[Any, ...]
    bridge: tuple[Any, Any]
    target_phase_turns: tuple[Q, ...]
    nodal_form_preparation: tuple[Q, ...]
    positive_initial_form: tuple[Q, ...]
    negative_initial_form: tuple[Q, ...]
    initial_phase_bounds: tuple[I, ...]
    positive_initial_box: tuple[I, ...]
    negative_initial_box: tuple[I, ...]
    amplitude: Q
    duration: Q
    preparation_error: Q
    observation_error: Q
    growth_factor_upper_bound: Q
    ideal_full_coordinate_radius_upper_bound: Q
    actual_full_coordinate_radius_upper_bound: Q
    fourth_bridge_derivative_upper_bound: Q
    finite_time_error_upper_bound: Q
    preparation_error_upper_bound: Q
    observation_error_upper_bound: Q
    total_response_error_upper_bound: Q
    ideal_curvature_values: tuple[Q, Q]
    curvature_prediction_bounds: tuple[I, I]
    phase_chamber_margin_lower_bound: Q
    phase_chamber_certified: bool
    separation_margin_lower_bound: Q
    discrimination_certified: bool
    status: str
    reasons: tuple[str, ...]
    clock: str = "tau=t/pi"
    observable: str = "C=(2*a-u_plus(h)+u_minus(h))/(a*h^2)"
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "source_anchors_support_and_reference_law_not_the_declared_preparations",
        "two_supplied_complete_storage_laws_eta_zero_and_one_over_one_hundred",
        "fixed_two_C6_support_aligned_bridge_unit_capacities_zero_loss_and_no_inputs_or_events",
        "all_twenty_four_coordinates_retained_without_nonlinear_shell_projection",
        "target_phase_is_exactly_two_pi_times_declared_rational_turns",
        "actual_phase_representation_error_must_be_included_in_preparation_error",
        "independent_full_coordinate_error_cubes_for_the_two_opposite_form_preparations",
        "initial_boxes_are_outward_projections_not_larger_certified_source_cubes",
        "one_noisy_bridge_form_reading_at_duration_for_each_preparation",
        "observation_error_applies_to_each_bridge_gap_not_each_node_sensor",
        "form_reversal_converts_the_two_ideal_responses_to_a_central_time_difference",
        "finite_amplitude_initial_curvature_is_exact_not_a_tangent_amplitude_approximation",
        "whole_window_growth_and_derivative_bounds_use_each_complete_nonlinear_field",
        "duration_and_amplitude_are_declared_in_the_model_clock_and_units",
        "no_clock_rescaling_invariance_or_independent_laboratory_clock_bridge_claimed",
        "disjoint_prediction_intervals_distinguish_supplied_models_not_select_a_physical_law",
        "no_trajectory_frozen_producer_replay_law_installation_or_formation_claim",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project, _validate_label_groups

        self.source.to_dict()
        for family in self.families:
            family.to_dict()
        _validate_label_groups(self.left_cycle, self.right_cycle, self.bridge)
        return {
            "schema": "tnfr.bridge-finite-law-discrimination.v1",
            "report": _project(self),
        }


def _admit_bridge_experiment(source, left_cycle, right_cycle, target_phase_turns):
    """Rebuild both complete laws and their shared finite preparation maps."""
    left = _ordered(left_cycle, "left_cycle", limit=7)
    right = _ordered(right_cycle, "right_cycle", limit=7)
    turns = _ordered(target_phase_turns, "target_phase_turns", limit=13)
    families = tuple(
        assess_bridge_storage_family(
            source,
            left_cycle=left,
            right_cycle=right,
            target_phase_turns=turns,
            epsilon=coefficient,
        )
        for coefficient in (Q(0), Q(1, 100))
    )
    first, second = families
    if (
        first.nodal_bridge_preparation != second.nodal_bridge_preparation
        or first.bridge_observation_rows != second.bridge_observation_rows
        or first.target_phase_turns != second.target_phase_turns
    ):
        raise ArithmeticError(
            "constitutive members do not retain the common experiment"
        )
    count = len(first.source.nodes)
    lift = tuple(row[0] for row in first.nodal_bridge_preparation[:count])
    if count != 12 or any(abs(value) != Q(1, 2) for value in lift):
        raise ArithmeticError(
            "finite comparison requires the admitted two-C6 bridge lift"
        )
    centers = tuple(-family.second_jet[0][0] for family in families)
    if centers != (Q(2, 3), Q(2, 3) + Q(1, 200)):
        raise ArithmeticError(
            "complete bridge curvature differs from its exact identity"
        )
    return families, lift, centers


def _bridge_growth_bounds(amplitude, duration, preparation_error):
    """Full-field bounds shared by both finite observation protocols."""
    if duration <= 0 or 2 * duration > 1:
        raise ValueError("finite bridge comparison requires 0<2*duration<=1")
    _, exponential_upper = exp_unit_bounds(2 * duration)
    growth = I(exponential_upper).hi
    ideal_radius = amplitude * growth / 2
    actual_radius = ideal_radius + preparation_error * growth
    form_derivative = (
        128 * Q(43, 40) * ideal_radius**3
        + 192 * Q(103, 100) * ideal_radius**2
        + 32 * ideal_radius
    )
    phase_derivative = 64 * Q(103, 100) * ideal_radius**2 + 32 * ideal_radius
    return growth, ideal_radius, actual_radius, form_derivative, phase_derivative


def _bridge_form_preparations(amplitude, preparation_error, family, lift):
    positive = tuple(amplitude * value for value in lift)
    negative = tuple(-value for value in positive)
    phase = tuple(2 * pi_interval() * value for value in family.target_phase_turns)
    error = I(-preparation_error, preparation_error)
    phase_box = tuple(value + error for value in phase)
    positive_box = (
        tuple(
            I(value - preparation_error, value + preparation_error)
            for value in positive
        )
        + phase_box
    )
    negative_box = (
        tuple(
            I(value - preparation_error, value + preparation_error)
            for value in negative
        )
        + phase_box
    )
    return positive, negative, phase, positive_box, negative_box


def assess_bridge_finite_law_discrimination(
    source,
    *,
    left_cycle,
    right_cycle,
    target_phase_turns,
    amplitude,
    duration,
    preparation_error,
    observation_error,
) -> BridgeFiniteLawDiscrimination:
    """Enclose the finite response of sine and its fixed cubic-current member.

    The two full laws have j_eta(delta)=sin(delta)+eta*sin(delta)^3 and its
    own phase potential, with eta=0 or 1/100. The phase row is unchanged.
    Their common symbolic critical target and preparation vector are rebuilt
    by assess_bridge_storage_family, never taken from cached caller bounds.

    Write a=amplitude, h=duration, rho=preparation_error, sigma=observation_error.
    The ideal states have x=+/-a*p and the same target phase. Since form
    reversal reverses time, u_minus(h)=-u_plus(-h), making the stated observable
    a central difference with exact center -u_plus''(0)/a=2/3+eta/2.

    For either law |j'|<=1 globally: c*(1+3*eta-3*eta*c^2) is increasing for
    c in [0,1] when eta<=1/6, and this polynomial is odd in c. Thus the full
    field is 2-Lipschitz in maximum norm. The harmonic representation of j bounds its
    second and third derivatives by 103/100 and 43/40. With
    R=(a/2)*exp(2h), differentiating the complete rows bounds the fourth
    bridge-form derivative by 128*(43/40)*R^3+192*(103/100)*R^2+32*R.

    The central-difference remainder is U4*h^2/(12*a), the two independent
    source errors contribute at most 4*rho*exp(2h)/(a*h^2), and the two bridge
    reading errors at most 2*sigma/(a*h^2). No sampled derivative is assumed.
    The sufficient chamber additionally requires 2*(R+rho*exp(2h))<pi/12.
    Unsupported numerical duration (2h>1) raises; failed sufficient margins
    return unavailable without declaring the laws observationally equivalent.
    """
    a, h, rho, sigma = (
        exact_or_represented_real(value, name)
        for name, value in (
            ("amplitude", amplitude),
            ("duration", duration),
            ("preparation_error", preparation_error),
            ("observation_error", observation_error),
        )
    )
    if a <= 0 or h <= 0 or rho < 0 or sigma < 0:
        raise ValueError(
            "positive amplitude/duration and nonnegative preparation/observation errors required"
        )
    if 2 * h > 1:
        raise ValueError("finite bridge comparison requires 0<2*duration<=1")
    families, lift, centers = _admit_bridge_experiment(
        source, left_cycle, right_cycle, target_phase_turns
    )
    first = families[0]
    growth, ideal_radius, actual_radius, derivative, _ = _bridge_growth_bounds(
        a, h, rho
    )
    denominator = a * h**2
    time_error = derivative * h**2 / (12 * a)
    source_error = 4 * rho * growth / denominator
    reading_error = 2 * sigma / denominator
    total_error = time_error + source_error + reading_error
    predictions = tuple(
        I(center - total_error, center + total_error) for center in centers
    )
    pi = pi_interval()
    chamber_margin = pi.lo / 12 - 2 * actual_radius
    separation_margin = predictions[1].lo - predictions[0].hi
    chamber_certified = chamber_margin > 0
    discrimination_certified = chamber_certified and separation_margin > 0
    positive, negative, phase, positive_box, negative_box = _bridge_form_preparations(
        a, rho, first, lift
    )
    reasons = tuple(
        reason
        for passed, reason in (
            (chamber_certified, "whole_window_phase_chamber_not_certified"),
            (separation_margin > 0, "finite_observation_intervals_not_separated"),
        )
        if not passed
    )
    return BridgeFiniteLawDiscrimination(
        source=first.source,
        families=families,
        left_cycle=first.left_cycle,
        right_cycle=first.right_cycle,
        bridge=first.bridge,
        target_phase_turns=first.target_phase_turns,
        nodal_form_preparation=lift,
        positive_initial_form=positive,
        negative_initial_form=negative,
        initial_phase_bounds=phase,
        positive_initial_box=positive_box,
        negative_initial_box=negative_box,
        amplitude=a,
        duration=h,
        preparation_error=rho,
        observation_error=sigma,
        growth_factor_upper_bound=growth,
        ideal_full_coordinate_radius_upper_bound=ideal_radius,
        actual_full_coordinate_radius_upper_bound=actual_radius,
        fourth_bridge_derivative_upper_bound=derivative,
        finite_time_error_upper_bound=time_error,
        preparation_error_upper_bound=source_error,
        observation_error_upper_bound=reading_error,
        total_response_error_upper_bound=total_error,
        ideal_curvature_values=centers,
        curvature_prediction_bounds=predictions,
        phase_chamber_margin_lower_bound=chamber_margin,
        phase_chamber_certified=chamber_certified,
        separation_margin_lower_bound=separation_margin,
        discrimination_certified=discrimination_certified,
        status="certified" if discrimination_certified else "unavailable",
        reasons=reasons,
    )


@dataclass(frozen=True)
class BridgeClockLawDiscrimination:
    """Finite ratio prediction with one unknown common constant clock scale.

    Three forward experiments share a sampling duration: the two opposite
    form preparations and one positive phase preparation. The phase reading
    is a signed continuous bridge lift relative to the aligned target, never
    an absolute circular distance. Preparation and observation errors apply
    independently to each experiment in the declared coordinate units.
    Curvature and denominator bounds are normalized by the same unknown
    positive a*h^2; their ratio is formed directly from measured decrements.
    """

    source: SineExchangeComparison
    families: tuple[BridgeStorageFamilyAssessment, BridgeStorageFamilyAssessment]
    left_cycle: tuple[Any, ...]
    right_cycle: tuple[Any, ...]
    bridge: tuple[Any, Any]
    bridge_target_turn_offset: int
    target_phase_turns: tuple[Q, ...]
    nodal_form_preparation: tuple[Q, ...]
    positive_initial_form: tuple[Q, ...]
    negative_initial_form: tuple[Q, ...]
    initial_phase_bounds: tuple[I, ...]
    positive_initial_box: tuple[I, ...]
    negative_initial_box: tuple[I, ...]
    phase_initial_form: tuple[Q, ...]
    phase_initial_phase_bounds: tuple[I, ...]
    phase_initial_box: tuple[I, ...]
    amplitude: Q
    sampling_duration: Q
    clock_scale_bounds: tuple[Q, Q]
    structural_duration_bounds: tuple[Q, Q]
    preparation_error: Q
    observation_error: Q
    growth_factor_upper_bound: Q
    ideal_full_coordinate_radius_upper_bound: Q
    actual_full_coordinate_radius_upper_bound: Q
    fourth_form_bridge_derivative_upper_bound: Q
    fourth_phase_bridge_derivative_upper_bound: Q
    form_time_error_coefficient: Q
    phase_time_error_coefficient: Q
    preparation_error_coefficient: Q
    observation_error_coefficient: Q
    form_clock_endpoint_finite_time_errors: tuple[Q, Q]
    phase_clock_endpoint_finite_time_errors: tuple[Q, Q]
    clock_endpoint_preparation_errors: tuple[Q, Q]
    clock_endpoint_observation_errors: tuple[Q, Q]
    form_clock_endpoint_error_bounds: tuple[Q, Q]
    phase_clock_endpoint_error_bounds: tuple[Q, Q]
    form_response_error_upper_bound: Q
    phase_response_error_upper_bound: Q
    sinc_amplitude_bounds: I
    ideal_form_curvature_values: tuple[Q, Q]
    ideal_phase_curvature_bounds: tuple[I, I]
    form_curvature_prediction_bounds: tuple[I, I]
    phase_curvature_prediction_bounds: tuple[I, I]
    denominator_lower_bounds: tuple[Q, Q]
    denominator_positive: bool
    ratio_prediction_bounds: tuple[I, I] | None
    phase_chamber_margin_lower_bound: Q
    phase_chamber_certified: bool
    separation_margin_lower_bound: Q | None
    discrimination_certified: bool
    status: str
    reasons: tuple[str, ...]
    clock: str = "tau=k*s; k is one constant shared by all three experiments"
    observable: str = "R=(2*a-u_plus(h)+u_minus(h))/(2*(a-v(h)))"
    phase_observable: str = (
        "v=theta_bridge_head-theta_bridge_tail-2*pi*bridge_target_turn_offset"
    )
    error_bound_method: str = (
        "convex_complete_A_h_squared_plus_B_over_h_squared_endpoints"
    )
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "source_anchors_support_and_reference_law_not_any_declared_preparation",
        "same_two_C6_unit_support_and_supplied_eta_zero_or_one_over_one_hundred_complete_law",
        "all_twenty_four_coordinates_retained_with_independent_preparation_error_per_experiment",
        "two_opposite_form_preparations_and_one_positive_phase_preparation",
        "symbolic_target_angles_are_not_replaced_by_binary_floating_values",
        "actual_phase_representation_error_must_consume_the_preparation_error_budget",
        "initial_boxes_enclose_exact_declared_cubes_not_additional_source_tolerances",
        "phase_reading_is_a_signed_continuous_bridge_lift_about_the_target_integer_turn_offset",
        "whole_window_chamber_keeps_the_relative_bridge_lift_on_its_zero_principal_branch",
        "observation_error_bounds_each_measured_bridge_gap_in_its_declared_units",
        "clock_scale_is_positive_constant_and_common_to_all_three_experiments",
        "same_actual_h_squared_cancels_algebraically_before_interval_division",
        "normalized_curvature_intervals_are_proof_coordinates_not_measured_known_clock_values",
        "finite_time_dependence_remains_uniformly_enclosed_over_the_clock_interval",
        "frozen_upper_duration_derivative_bounds_give_convex_complete_clock_error_budgets",
        "strict_positive_denominator_required_before_forming_a_ratio",
        "phase_initial_curvature_uses_exact_finite_amplitude_current_not_its_consensus_limit",
        "no_clock_calibration_interexperiment_clock_drift_or_unknown_readout_gain_is_certified",
        "no_trajectory_runtime_law_installation_physical_selection_or_formation_claim",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project, _validate_label_groups

        self.source.to_dict()
        for family in self.families:
            family.to_dict()
        _validate_label_groups(self.left_cycle, self.right_cycle, self.bridge)
        return {
            "schema": "tnfr.bridge-clock-law-discrimination.v1",
            "report": _project(self),
        }


def assess_bridge_clock_law_discrimination(
    source,
    *,
    left_cycle,
    right_cycle,
    target_phase_turns,
    amplitude,
    sampling_duration,
    clock_scale_bounds,
    preparation_error,
    observation_error,
) -> BridgeClockLawDiscrimination:
    """Enclose a noisy response ratio for an interval of common clock scales.

    Let h=k*sampling_duration with one positive constant k shared by all
    experiments. The two form preparations give D_F=2a-u_plus(h)+u_minus(h).
    The phase preparation x=0, theta=theta_*+a*p is fixed by form reversal,
    so its relative bridge phase v(t) is even. It gives D_P=2(a-v(h)) and
    exact initial curvature (8/9)*j_eta(a)/a. The requested observable is
    D_F/D_P; its common a*h^2 factor cancels before interval arithmetic.

    The complete-field bounds at h_hi fix G, R, U4 and V4. In particular
    V4=64*(103/100)*R^2+32*R follows from the linear full phase row. With
    A=U4/(12a) or V4/(12a), B=(4*rho*G+2*sigma)/a, each normalized
    response error is at most A*h^2+B/h^2. Its maximum on the clock interval
    occurs at an endpoint by convexity in h^2. Summing separate maxima would
    unnecessarily discard that dependence. No exact finite-time clock
    invariance is asserted: the remaining time dependence is enclosed.

    Inputs admit exact or represented reals before arithmetic. The sufficient
    numerical domain is 0<a<=1 and 0<2*h_hi<=1; the sinc series is regular
    even for sub-grid amplitudes. Failed chamber, denominator or separation
    guards yield unavailable, without declaring a physical equivalence.
    """
    a, sampling, rho, sigma = (
        exact_or_represented_real(value, name)
        for name, value in (
            ("amplitude", amplitude),
            ("sampling_duration", sampling_duration),
            ("preparation_error", preparation_error),
            ("observation_error", observation_error),
        )
    )
    if not 0 < a <= 1 or sampling <= 0 or rho < 0 or sigma < 0:
        raise ValueError(
            "require 0<amplitude<=1, positive sampling duration, and nonnegative errors"
        )
    raw_scales = _ordered(clock_scale_bounds, "clock_scale_bounds", limit=3)
    if len(raw_scales) != 2:
        raise ValueError(
            "clock_scale_bounds must contain two ordered positive endpoints"
        )
    scales = tuple(
        exact_or_represented_real(value, "clock_scale_bounds") for value in raw_scales
    )
    if not 0 < scales[0] <= scales[1]:
        raise ValueError(
            "clock_scale_bounds must contain two ordered positive endpoints"
        )
    durations = tuple(sampling * value for value in scales)
    growth, ideal_radius, actual_radius, form_fourth, phase_fourth = (
        _bridge_growth_bounds(a, durations[1], rho)
    )
    families, lift, form_centers = _admit_bridge_experiment(
        source, left_cycle, right_cycle, target_phase_turns
    )
    first = families[0]
    positive, negative, phase, positive_box, negative_box = _bridge_form_preparations(
        a, rho, first, lift
    )
    phase_form = (Q(0),) * len(lift)
    phase_initial = tuple(value + a * shift for value, shift in zip(phase, lift))
    phase_box = (I(-rho, rho),) * len(lift) + tuple(
        value + I(-rho, rho) for value in phase_initial
    )
    phase_row = first.bridge_observation_rows[1][len(lift) :]
    target_offset = sum(
        (entry * turn for entry, turn in zip(phase_row, first.target_phase_turns)), Q(0)
    )
    if target_offset.denominator != 1:
        raise ArithmeticError("aligned bridge target lacks an integral turn offset")
    form_a, phase_a = form_fourth / (12 * a), phase_fourth / (12 * a)
    source_b, observation_b = 4 * rho * growth / a, 2 * sigma / a
    inverse_time_b = source_b + observation_b

    def clock_endpoints(coefficient):
        return tuple(
            coefficient * time**2 + inverse_time_b / time**2 for time in durations
        )

    form_endpoints, phase_endpoints = clock_endpoints(form_a), clock_endpoints(phase_a)
    form_error, phase_error = max(form_endpoints), max(phase_endpoints)
    sinc_a = sinc(Jet.constant(a, 0)).coeffs[0]
    phase_centers = tuple(
        Q(8, 9) * (sinc_a + family.epsilon * a**2 * sinc_a**3) for family in families
    )
    form_predictions = tuple(
        I(center - form_error, center + form_error) for center in form_centers
    )
    phase_predictions = tuple(
        I(center.lo - phase_error, center.hi + phase_error) for center in phase_centers
    )
    denominator_lower_bounds = tuple(value.lo for value in phase_predictions)
    denominator_positive = min(denominator_lower_bounds) > 0
    ratios = (
        tuple(
            numerator / denominator
            for numerator, denominator in zip(form_predictions, phase_predictions)
        )
        if denominator_positive
        else None
    )
    gap = None if ratios is None else ratios[1].lo - ratios[0].hi
    chamber_margin = pi_interval().lo / 12 - 2 * actual_radius
    chamber_certified = chamber_margin > 0
    separated = gap is not None and gap > 0
    certified = chamber_certified and denominator_positive and separated
    reasons = tuple(
        reason
        for passed, reason in (
            (chamber_certified, "whole_window_phase_chamber_not_certified"),
            (denominator_positive, "phase_decrement_denominator_not_strictly_positive"),
            (separated, "finite_ratio_intervals_not_separated"),
        )
        if not passed
    )
    return BridgeClockLawDiscrimination(
        source=first.source,
        families=families,
        left_cycle=first.left_cycle,
        right_cycle=first.right_cycle,
        bridge=first.bridge,
        bridge_target_turn_offset=target_offset.numerator,
        target_phase_turns=first.target_phase_turns,
        nodal_form_preparation=lift,
        positive_initial_form=positive,
        negative_initial_form=negative,
        initial_phase_bounds=phase,
        positive_initial_box=positive_box,
        negative_initial_box=negative_box,
        phase_initial_form=phase_form,
        phase_initial_phase_bounds=phase_initial,
        phase_initial_box=phase_box,
        amplitude=a,
        sampling_duration=sampling,
        clock_scale_bounds=scales,
        structural_duration_bounds=durations,
        preparation_error=rho,
        observation_error=sigma,
        growth_factor_upper_bound=growth,
        ideal_full_coordinate_radius_upper_bound=ideal_radius,
        actual_full_coordinate_radius_upper_bound=actual_radius,
        fourth_form_bridge_derivative_upper_bound=form_fourth,
        fourth_phase_bridge_derivative_upper_bound=phase_fourth,
        form_time_error_coefficient=form_a,
        phase_time_error_coefficient=phase_a,
        preparation_error_coefficient=source_b,
        observation_error_coefficient=observation_b,
        form_clock_endpoint_finite_time_errors=tuple(
            form_a * time**2 for time in durations
        ),
        phase_clock_endpoint_finite_time_errors=tuple(
            phase_a * time**2 for time in durations
        ),
        clock_endpoint_preparation_errors=tuple(
            source_b / time**2 for time in durations
        ),
        clock_endpoint_observation_errors=tuple(
            observation_b / time**2 for time in durations
        ),
        form_clock_endpoint_error_bounds=form_endpoints,
        phase_clock_endpoint_error_bounds=phase_endpoints,
        form_response_error_upper_bound=form_error,
        phase_response_error_upper_bound=phase_error,
        sinc_amplitude_bounds=sinc_a,
        ideal_form_curvature_values=form_centers,
        ideal_phase_curvature_bounds=phase_centers,
        form_curvature_prediction_bounds=form_predictions,
        phase_curvature_prediction_bounds=phase_predictions,
        denominator_lower_bounds=denominator_lower_bounds,
        denominator_positive=denominator_positive,
        ratio_prediction_bounds=ratios,
        phase_chamber_margin_lower_bound=chamber_margin,
        phase_chamber_certified=chamber_certified,
        separation_margin_lower_bound=gap,
        discrimination_certified=certified,
        status="certified" if certified else "unavailable",
        reasons=reasons,
    )
