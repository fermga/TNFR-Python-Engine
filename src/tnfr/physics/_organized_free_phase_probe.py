"""Conditional free-probe and prior form-preparation bounds for two C9s.

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
_FINE_DECAY_DENOMINATOR = Q(32)


def _fine_energy_upper_bound(gamma_upper, form_error, phase_error):
    """Bound the shared fine functional from normalized nonnegative primitives."""
    scaled_form = gamma_upper * form_error
    return ((phase_error + scaled_form) ** 2 + scaled_form**2) / 2


def _free_probe_current_bounds(coefficient):
    """Rebuild the retained family's coefficients and derivative majorants."""
    coefficients = _admit_fourier_current(_current_fourier_coefficients(coefficient))
    bounds = tuple(
        sum((order**derivative * abs(scale) for order, scale in coefficients), Q(0))
        for derivative in range(3)
    )
    return coefficients, bounds


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

    coefficients, derivative_bounds = _free_probe_current_bounds(coefficient)
    current_bound, slope_bound, curvature_bound = derivative_bounds
    gamma, _ = _parameters(())
    gamma_squared = gamma**2
    contact = gamma_squared / _CONTACT_DENOMINATOR
    # This is a sufficient bound for the fixed model, not an extra adjustable
    # kinetic parameter. The stricter half-bound supports v > 11/3.
    if gamma.hi**2 * current_bound >= Q(1, 2):
        raise ArithmeticError("fixed probe kinetic bound is unresolved")

    fine_energy = _fine_energy_upper_bound(gamma.hi, rho_x, rho_theta)
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
        fine_decay_rate_bounds=gamma_squared / _FINE_DECAY_DENOMINATOR,
        fine_energy_upper_bound=fine_energy,
        fine_energy_margin=_FINE_ENERGY_LIMIT - fine_energy,
        phase_residual_norm_squared_upper_bound=4 * fine_energy,
        pointwise_current_defect_upper_bound=4 * curvature_bound * fine_energy / 9,
        harmonic_defect_upper_bound=8 * curvature_bound * fine_energy / 9,
        attachment_work_upper_bound=reference_work
        + contact.hi * (rho_x**2 + slope_bound * rho_theta**2),
    )


@dataclass(frozen=True)
class _OrganizedFreePhasePreparationBound:
    """Necessary contrast-resource bounds, without source authentication.

    Continuous and jump budgets are unweighted eighteen-node L1 form
    variation, not storage work. ``not_excluded`` is not reachability,
    formation, identity retention or an available-input certificate.
    """

    initial_form_contrast_bound: Q
    continuous_form_variation_bound: Q
    jump_form_variation_bound: Q
    total_form_variation_bound: Q
    current_coefficients: tuple[tuple[int, Q], ...]
    current_bound: Q
    gamma_bounds: I
    current_forcing_bounds: I
    initial_scaled_form_upper_bound: Q
    unforced_scaled_form_upper_bound: Q
    source_scaled_form_upper_bound: Q
    scaled_form_upper_bound: Q
    target_margin: Q
    necessary_total_form_variation_lower_bound: Q
    status: str
    target_scaled_form: Q = _INITIAL_SCALED_FORM
    interval_method: str = INTERVAL_METHOD
    conditional_on_complete_law_and_resource_bounds: bool = True
    scope: tuple[str, ...] = (
        "two_fixed_marked_nine_node_children_with_unit_internal_cycle_edges",
        "nine_matching_contacts_share_one_supplied_nonnegative_weight_a_of_tau",
        "locally_integrable_contact_weight_with_arbitrary_nonnegative_total_exposure",
        "held_unit_capacities_fixed_gamma_and_both_uniform_quarter_rows",
        "uniform_over_the_retained_constitutive_family_zero_to_one",
        "additive_integrable_form_inputs_and_finite_form_jumps_keep_the_same_nodes",
        "no_other_phase_capacity_or_internal_support_events_are_admitted",
        "source_bounds_are_unweighted_full_nodal_L1_form_variation_not_work",
        "source_budgets_do_not_follow_from_net_form_change_or_stored_energy",
        "initial_mean_contrast_and_all_source_bounds_are_supplied_not_authenticated",
        "unforced_envelope_uses_the_global_current_bound_without_a_fine_state_certificate",
        "resource_bound_is_necessary_for_reachability_not_sufficient",
        "no_source_execution_identity_formation_autonomy_or_physical_sensor_claim",
    )


def _bound_organized_free_phase_preparation(
    *,
    initial_form_contrast_bound,
    continuous_form_variation_bound,
    jump_form_variation_bound,
) -> _OrganizedFreePhasePreparationBound:
    """Bound preparation contrast under every common matching-contact history.

    Supply nonnegative exact/represented bounds on |X_A(0)-X_B(0)|,
    integral sum_i |Gamma_i| d tau, and sum_events sum_i |Delta x_i|.
    Missing bounds are rejected; they never mean a zero resource baseline.
    All three are primitives, not previously computed reports or work budgets.

    Internal edges cancel from the mean row, giving
    v_tau=-(a/2)*(v+gamma**2*Jbar)+b_v for v=gamma*(X_A-X_B).
    The source projection obeys |b_v|<=gamma*sum_i|Gamma_i|/9;
    each form jump has the same projection bound. Since a>=0 and
    |Jbar|<=J0, the unforced interval of radius max(|v(0)|,gamma**2*J0)
    is invariant. Source variation adds at most its undiscounted projection.
    The resulting envelope holds at every time, without an exponential,
    a trajectory, a contact schedule or an assumed source fine-state tube.
    """
    initial, continuous, jumps = tuple(
        _exact(value, name)
        for name, value in (
            ("initial_form_contrast_bound", initial_form_contrast_bound),
            ("continuous_form_variation_bound", continuous_form_variation_bound),
            ("jump_form_variation_bound", jump_form_variation_bound),
        )
    )
    if min(initial, continuous, jumps) < 0:
        raise ValueError("contrast and form variation bounds must be nonnegative")

    coefficients, derivative_bounds = _free_probe_current_bounds(Q(1))
    current_bound = derivative_bounds[0]
    gamma, _ = _parameters(())
    forcing = gamma**2 * current_bound
    initial_scaled = gamma.hi * initial
    unforced = max(initial_scaled, forcing.hi)
    total = continuous + jumps
    source = gamma.hi * total / _CHILD_SIZE
    upper = unforced + source
    margin = _INITIAL_SCALED_FORM - upper
    necessary = _CHILD_SIZE * max(Q(0), _INITIAL_SCALED_FORM - unforced) / gamma.hi
    return _OrganizedFreePhasePreparationBound(
        initial_form_contrast_bound=initial,
        continuous_form_variation_bound=continuous,
        jump_form_variation_bound=jumps,
        total_form_variation_bound=total,
        current_coefficients=coefficients,
        current_bound=current_bound,
        gamma_bounds=gamma,
        current_forcing_bounds=forcing,
        initial_scaled_form_upper_bound=initial_scaled,
        unforced_scaled_form_upper_bound=unforced,
        source_scaled_form_upper_bound=source,
        scaled_form_upper_bound=upper,
        target_margin=margin,
        necessary_total_form_variation_lower_bound=necessary,
        status="excluded_by_resource_bound" if margin > 0 else "not_excluded",
    )
