"""Conditional bounds for the fixed freely reacting two-C9 phase probe.

The preparation, zero child means of the fine residuals, winding profiles and
support event are supplied premises. This instrument evaluates no trajectory,
reached source or observed record. Proof: ORGANIZED_FREE_PHASE_PROBE.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._rational_interval import INTERVAL_METHOD, I
from ._phase_boundary_information import _admit_fourier_current
from ._sine_class_constitutive_comparison import _current_fourier_coefficients
from .relational_sine_port_composition import _parameters

_CHILD_SIZE = 9
_INITIAL_SCALED_FORM = Q(12)
_CONTACT_DENOMINATOR = Q(1000)
_FINE_ENERGY_LIMIT = Q(1, 40000)


@dataclass(frozen=True)
class _OrganizedFreePhaseProbe:
    """Sufficient bounds conditional on the independently supplied preparation.

    Error bounds are combined Euclidean norms over the eighteen within-child
    residuals, not coordinatewise allowances. Both child residual means vanish
    separately in each channel. The reference has equal lifted phase origins
    and forms +6/gamma and -6/gamma; each child has its marked winding-one C9
    profile. The report does not verify those premises from a physical state.

    ``form_contrast_bounds`` encloses X_A-X_B throughout 0 <= tau <= T.
    Scaled quantities use v=gamma*(X_A-X_B) and s=a*tau/2. The phase endpoint
    lower bound concerns the continuous lifted Phi_A-Phi_B at s=1, not a
    wrapped angle. Decay rates and ``horizon_bounds`` use structural tau.
    """

    deformation: Q
    form_error_bound: Q
    phase_error_bound: Q
    current_coefficients: tuple[tuple[int, Q], ...]
    current_derivative_bounds: tuple[Q, Q, Q]
    gamma_bounds: I
    contact_conductance_bounds: I
    horizon_bounds: I
    initial_form_contrast_bounds: I
    form_contrast_bounds: I
    fine_decay_rate_bounds: I
    fine_energy_upper_bound: Q
    fine_energy_margin: Q
    phase_residual_norm_squared_upper_bound: Q
    pointwise_current_defect_upper_bound: Q
    harmonic_defect_upper_bound: Q
    attachment_work_upper_bound: Q
    scaled_form_lower_bound: Q = Q(11, 3)
    scaled_form_upper_bound: Q = _INITIAL_SCALED_FORM
    endpoint_phase_lower_bound: Q = Q(34, 5)
    fine_energy_limit: Q = _FINE_ENERGY_LIMIT
    conditional_on_preparation: bool = True
    interval_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "supplied_two_marked_winding_one_C9_profiles_with_matching_slot_contacts",
        "primitive_error_norms_and_zero_child_means_are_preparation_premises",
        "initial_global_form_mean_zero_and_relative_phase_lift_zero",
        "held_unit_capacities_and_both_uniform_quarter_rows_in_structural_tau",
        "gamma_is_1_over_1023_pi_and_contact_conductance_is_gamma_squared_over_1000",
        "free_post_contact_evolution_without_phase_resets_clamping_or_inputs",
        "fine_energy_and_phase_residual_bounds_decay_at_the_reported_tau_rate",
        "pointwise_current_defect_bound_has_the_same_exponential_decay_factor",
        "harmonic_defect_is_the_bias_bound_for_the_derivative_free_full_turn_identity",
        "attachment_work_is_separate_from_prior_preparation_storage_and_loss",
        "conditional_query_access_not_a_verified_reached_source_or_measured_record",
        "no_autonomous_preparation_contact_occurrence_or_physical_admission",
    )


def _derive_organized_free_phase_probe(
    *, deformation, form_error_bound, phase_error_bound
) -> _OrganizedFreePhaseProbe:
    """Rebuild the fixed probe bounds from primitive law and error allowances.

    The deformation must be in [0,1]; both error allowances are nonnegative
    exact or represented reals. Their sufficient initial fine-energy bound
    must be strictly below 1/40000. Rejection outside that budget means this
    theorem supplies no certificate; it does not establish failed access or
    lost identity. No cached report, source verdict or measured response is
    consumed. Complete-law and preparation premises remain explicit above.
    """
    coefficient = _exact(deformation, "deformation")
    rho_x = _exact(form_error_bound, "form_error_bound")
    rho_theta = _exact(phase_error_bound, "phase_error_bound")
    if not 0 <= coefficient <= 1:
        raise ValueError("deformation requires the proved interval [0,1]")
    if min(rho_x, rho_theta) < 0:
        raise ValueError("fine form and phase error bounds must be nonnegative")

    coefficients = _admit_fourier_current(_current_fourier_coefficients(coefficient))
    derivative_bounds = tuple(
        sum((order**derivative * abs(scale) for order, scale in coefficients), Q(0))
        for derivative in range(3)
    )
    current_bound, slope_bound, curvature_bound = derivative_bounds
    gamma, _ = _parameters(())
    gamma_squared = gamma**2
    contact = gamma_squared / _CONTACT_DENOMINATOR
    # This is a sufficient bound for the fixed model, not an extra adjustable
    # kinetic parameter. The stricter half-bound supports v > 11/3.
    if gamma.hi**2 * current_bound >= Q(1, 2):
        raise ArithmeticError("fixed probe kinetic bound is unresolved")

    scaled_form_error = gamma.hi * rho_x
    fine_energy = ((rho_theta + scaled_form_error) ** 2 + scaled_form_error**2) / 2
    if fine_energy >= _FINE_ENERGY_LIMIT:
        raise ValueError("fine error budget requires F0_upper < 1/40000")

    # Nine new matching edges carry the mean contrast 12/gamma. Since
    # a=gamma**2/1000, their exact reference attachment work cancels gamma.
    reference_work = _CHILD_SIZE * _INITIAL_SCALED_FORM**2 / (2 * _CONTACT_DENOMINATOR)
    return _OrganizedFreePhaseProbe(
        deformation=coefficient,
        form_error_bound=rho_x,
        phase_error_bound=rho_theta,
        current_coefficients=coefficients,
        current_derivative_bounds=derivative_bounds,
        gamma_bounds=gamma,
        contact_conductance_bounds=contact,
        horizon_bounds=2 / contact,
        initial_form_contrast_bounds=_INITIAL_SCALED_FORM / gamma,
        form_contrast_bounds=I(Q(11, 3) / gamma.hi, _INITIAL_SCALED_FORM / gamma.lo),
        fine_decay_rate_bounds=gamma_squared / 32,
        fine_energy_upper_bound=fine_energy,
        fine_energy_margin=_FINE_ENERGY_LIMIT - fine_energy,
        phase_residual_norm_squared_upper_bound=4 * fine_energy,
        pointwise_current_defect_upper_bound=4 * curvature_bound * fine_energy / 9,
        harmonic_defect_upper_bound=8 * curvature_bound * fine_energy / 9,
        attachment_work_upper_bound=reference_work
        + contact.hi * (rho_x**2 + slope_bound * rho_theta**2),
    )
