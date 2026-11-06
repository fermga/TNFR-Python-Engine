"""Prospective endpoint information budgets for a small phase preparation.

Proof: ORGANIZED_SMALL_RESOURCE_INFORMATION.md#organized-small-resource-information.
The fixed two-C9 complete law and nominal preparation are supplied. This
instrument evaluates neither candidate response nor an observed reading.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._rational_interval import INTERVAL_METHOD, I
from ._organized_free_phase_probe import (
    _CHILD_SIZE,
    _CONTACT_DENOMINATOR,
    _FINE_DECAY_DENOMINATOR,
    _FINE_ENERGY_LIMIT,
    _fine_energy_upper_bound,
    _free_probe_current_bounds,
)
from .relational_sine_port_composition import _parameters


@dataclass(frozen=True)
class _OrganizedSmallResourceInformationBound:
    """Conditional information limits for one raw relative mean-form reading.

    Preparation bounds apply separately to each candidate law: they need not
    describe common nuisance values. Fine form/phase bounds are Euclidean
    norms of the eighteen separately child-centered residuals. The initial
    mean contrast has magnitude at most its error bound; relative phase
    differs from ``phase_offset`` by at most the registration error.

    The observation is Y=(1+g)*r(1+e_s)+e_read, with each signed nuisance
    bounded by its supplied magnitude. Timing uses the independently declared
    clock s=a*tau/2, not an unknown clock rate. A guaranteed separation is
    prospective; it identifies neither an observed law nor a physical sensor.
    ``current_coefficients`` records the lambda=1 endpoint used for the uniform
    current majorants; both endpoint laws remain candidates.
    Nonidentifiability uses a common nominal reading witness from reading
    noise alone, not overlap of conservative response enclosures.
    """

    phase_offset: Q
    form_error_bound: Q
    phase_error_bound: Q
    initial_form_contrast_error_bound: Q
    phase_registration_error_bound: Q
    endpoint_reading_error_bound: Q
    exposure_time_error_bound: Q
    relative_gain_error_bound: Q
    current_coefficients: tuple[tuple[int, Q], ...]
    current_derivative_bounds: tuple[Q, Q, Q]
    gamma_bounds: I
    contact_conductance_bounds: I
    nominal_horizon_bounds: I
    exposure_window: tuple[Q, Q]
    fine_decay_rate_in_scaled_clock: Q
    fine_energy_upper_bound: Q
    fine_energy_margin: Q
    fine_current_defect_upper_bound: Q
    nominal_gap_lower_bound: Q
    nominal_gap_upper_bound: Q
    phase_comparison_seed_upper_bound: Q
    endpoint_phase_error_upper_bound: Q
    window_phase_error_upper_bound: Q
    endpoint_preparation_error_upper_bound: Q
    window_form_upper_bound: Q
    window_form_rate_upper_bound: Q
    exposure_error_upper_bound: Q
    gain_error_upper_bound: Q
    observation_error_upper_bound: Q
    separation_margin: Q
    nominal_reading_overlap_margin: Q
    nominal_attachment_work_lower_bound: Q
    nominal_attachment_work_upper_bound: Q
    attachment_work_upper_bound: Q
    status: str
    nominal_gap_orientation: str = "sine_endpoint_minus_alternative_endpoint_positive"
    interval_method: str = INTERVAL_METHOD
    conditional_on_preparation_and_observation_bounds: bool = True
    scope: tuple[str, ...] = (
        "two_supplied_marked_winding_one_C9_children_with_nine_matching_contacts",
        "fixed_gamma_matching_weight_and_both_uniform_quarter_rows",
        "held_unit_capacities_no_input_jumps_or_later_support_changes",
        "nominal_child_mean_forms_zero_and_relative_phase_offset_positive",
        "fine_norm_bounds_retain_all_child_centered_form_and_phase_residuals",
        "each_candidate_has_its_own_allowed_preparation_and_observation_nuisances",
        "clock_is_s_equals_a_tau_over_two_without_unknown_rate_calibration",
        "one_raw_relative_mean_form_reading_at_nominal_s_equals_one",
        "multiplicative_gain_uncertainty_changes_readout_not_the_evolution_law",
        "supplied_bounds_do_not_authenticate_source_identity_or_sensor_capability",
        "guaranteed_separation_is_prospective_not_a_selected_observed_law",
        "nonidentifiability_uses_a_nominal_midpoint_reading_noise_witness",
        "not_certified_is_neither_separation_nor_a_proved_ambiguity_regime",
        "attachment_work_is_separate_from_source_formation_and_physical_cost",
        "no_trajectory_source_replay_response_evaluation_or_physical_admission",
    )


def _bound_organized_small_resource_information(
    *,
    phase_offset,
    form_error_bound,
    phase_error_bound,
    initial_form_contrast_error_bound,
    phase_registration_error_bound,
    endpoint_reading_error_bound,
    exposure_time_error_bound,
    relative_gain_error_bound,
) -> _OrganizedSmallResourceInformationBound:
    """Bound distinguishability without calculating either endpoint response.

    Every input is a required finite exact or represented real. The nominal
    phase lies in (0,1/100], all error bounds are nonnegative, and exposure
    jitter is at most 1/2 in s units. Gain uncertainty may be arbitrarily large;
    it can simply make separation uncertifiable. The shared sufficient fine
    energy must be strictly below 1/40000. Rejection at that boundary means
    this theorem is unavailable, not that the child identities are lost.

    Both nominal candidate endpoints are negative, with sine minus alternative
    bounded between gamma*d**3/20 and gamma*d**3/6. Preparation error is bounded
    for each law separately. A strict positive gap after both observation
    allowances guarantees nonoverlap. Reading noise of at least half the
    nominal gap upper bound admits their actual midpoint under both laws.
    No report, cached verdict or observed response replaces these primitives.
    """
    names_and_values = (
        ("phase_offset", phase_offset),
        ("form_error_bound", form_error_bound),
        ("phase_error_bound", phase_error_bound),
        ("initial_form_contrast_error_bound", initial_form_contrast_error_bound),
        ("phase_registration_error_bound", phase_registration_error_bound),
        ("endpoint_reading_error_bound", endpoint_reading_error_bound),
        ("exposure_time_error_bound", exposure_time_error_bound),
        ("relative_gain_error_bound", relative_gain_error_bound),
    )
    d, rho_x, rho_theta, eps_r, eps_delta, eps_read, eta, gain = (
        _exact(value, name) for name, value in names_and_values
    )
    if not 0 < d <= Q(1, 100):
        raise ValueError("phase_offset requires the proved interval (0,1/100]")
    if min(rho_x, rho_theta, eps_r, eps_delta, eps_read, eta, gain) < 0:
        raise ValueError("preparation and observation error bounds must be nonnegative")
    if eta > Q(1, 2):
        raise ValueError("exposure_time_error_bound must be at most 1/2 in s units")

    gamma, _ = _parameters(())
    gu, g_squared = gamma.hi, gamma.hi**2
    coefficients, derivative_bounds = _free_probe_current_bounds(Q(1))
    _, j1, j2 = derivative_bounds
    contact = gamma**2 / _CONTACT_DENOMINATOR
    decay = 2 * _CONTACT_DENOMINATOR / _FINE_DECAY_DENOMINATOR
    fine = _fine_energy_upper_bound(gu, rho_x, rho_theta)
    if fine >= _FINE_ENERGY_LIMIT:
        raise ValueError("fine error budget requires F0_upper < 1/40000")
    denominator_endpoint = 1 - g_squared * j1
    denominator_window = 1 - g_squared * j1 * (1 + eta)
    if denominator_window <= 0:
        raise ArithmeticError("fixed-model phase comparison bound is unresolved")

    fine_current = 4 * j2 * fine / _CHILD_SIZE
    phase_initial = eps_delta + gu * eps_r + g_squared * fine_current / decay
    phase_endpoint = phase_initial / denominator_endpoint
    phase_window = phase_initial / denominator_window
    preparation = (
        Q(2, 5) * eps_r
        + Q(2, 3) * gu * j1 * phase_endpoint
        + Q(2, 5) * gu * fine_current / (decay - 1)
    )
    form_upper = eps_r + gu * j1 * (d + phase_window) + gu * fine_current / decay
    form_rate = (
        eps_r + 2 * gu * j1 * (d + phase_window) + gu * fine_current * (1 + 1 / decay)
    )
    exposure_error, gain_error = eta * form_rate, gain * form_upper
    observation_error = preparation + exposure_error + gain_error + eps_read
    gap_lower, gap_upper = gamma.lo * d**3 / 20, gu * d**3 / 6
    margin, overlap_margin = gap_lower - 2 * observation_error, 2 * eps_read - gap_upper
    if margin > 0:
        status = "guaranteed_separation"
    elif overlap_margin >= 0:
        status = "observation_class_nonidentifiable"
    else:
        status = "not_certified"
    work_upper = contact.hi * (
        Q(_CHILD_SIZE, 2) * (eps_r**2 + j1 * (d + eps_delta) ** 2)
        + rho_x**2
        + j1 * rho_theta**2
    )
    return _OrganizedSmallResourceInformationBound(
        phase_offset=d,
        form_error_bound=rho_x,
        phase_error_bound=rho_theta,
        initial_form_contrast_error_bound=eps_r,
        phase_registration_error_bound=eps_delta,
        endpoint_reading_error_bound=eps_read,
        exposure_time_error_bound=eta,
        relative_gain_error_bound=gain,
        current_coefficients=coefficients,
        current_derivative_bounds=derivative_bounds,
        gamma_bounds=gamma,
        contact_conductance_bounds=contact,
        nominal_horizon_bounds=2 / contact,
        exposure_window=(1 - eta, 1 + eta),
        fine_decay_rate_in_scaled_clock=decay,
        fine_energy_upper_bound=fine,
        fine_energy_margin=_FINE_ENERGY_LIMIT - fine,
        fine_current_defect_upper_bound=fine_current,
        nominal_gap_lower_bound=gap_lower,
        nominal_gap_upper_bound=gap_upper,
        phase_comparison_seed_upper_bound=phase_initial,
        endpoint_phase_error_upper_bound=phase_endpoint,
        window_phase_error_upper_bound=phase_window,
        endpoint_preparation_error_upper_bound=preparation,
        window_form_upper_bound=form_upper,
        window_form_rate_upper_bound=form_rate,
        exposure_error_upper_bound=exposure_error,
        gain_error_upper_bound=gain_error,
        observation_error_upper_bound=observation_error,
        separation_margin=margin,
        nominal_reading_overlap_margin=overlap_margin,
        nominal_attachment_work_lower_bound=4 * contact.lo * d**2,
        nominal_attachment_work_upper_bound=5 * contact.hi * d**2,
        attachment_work_upper_bound=work_upper,
        status=status,
    )
