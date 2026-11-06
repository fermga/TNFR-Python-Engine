"""Uniform own-law admission of the unchanged two-ramp receiver source.

The constitutive receiver proof admits every current in the fixed interval,
including its source transport and actual phase-velocity defect. Shared
preparation kernels carry inequalities, never a sine preparation verdict.
No trajectory, eventual winding, or capture outcome is evaluated here.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._rational_interval import I, pi_interval, sqrt
from ._constitutive_identity_barriers import _BOUNDARY_PROOF_LIMIT
from ._organized_network_capture import _LAPLACIAN_GAP
from ._organized_network_compatibility import _EQUAL_CONTROL, _derive_organized_triangle
from ._sine_preparation import (
    _prepared_form_costs,
    _prepared_transit_radii,
    _prepared_velocity_defect,
)
from .relational_sine_formed_classes import _SOURCE_SCALE
from .reversible_eigenmode_reference import _negative_exp_bounds

_LAYER_TIME = Q(200)
_RELEASE_TIME = Q(400)
_PHASE_BUDGET = Q(21, 100000)
_DEFECT_BUDGET = Q(11, 10**6)
_FORM_BUDGET = Q(1, 50)


@dataclass(frozen=True)
class _ConstitutiveReceiverSource:
    """Fresh primitive-source bounds uniform on a symmetric law interval.

    The retained form and velocity bounds hold at every time after release.
    The phase ball applies at release, not to an entire future trajectory.
    Constituent error sums vanish initially; only global means persist.
    """

    deformation_bound: Q
    deformation_interval: tuple[Q, Q]
    form_error_bound: Q
    phase_error_bound: Q
    nodes: tuple[int, ...]
    weighted_edges: tuple[tuple[int, int, Q], ...]
    weighted_degrees: tuple[Q, ...]
    nominal_initial_forms: tuple[Q, ...]
    nominal_initial_form_gradient: tuple[Q, ...]
    nominal_initial_internal_form_storage: Q
    nominal_initial_contact_form_storage: Q
    nominal_initial_form_storage: Q
    initial_form_gradient_l1: Q
    nominal_form_gradient_norm_bounds: I
    initial_potential_deformation_upper_bound: Q
    initial_storage_correction_upper_bound: Q
    initial_storage_bounds: I
    gamma_bounds: I
    slow_clock_factor_bounds: I
    edge_current_absolute_upper_bound: Q
    edge_current_derivative_absolute_upper_bound: Q
    forcing_norm_bounds: I
    global_phase_lipschitz_bound: Q
    semigroup_gap_lower_bound: Q
    scaled_initial_norm_upper_bound: Q
    reference_phase_bounds: tuple[I, ...]
    proxy_reference_distance_bounds: I
    layer_form_decay_bounds: I
    layer_scaled_form_norm_upper_bound: Q
    layer_form_norm_upper_bound: Q
    layer_phase_error_norm_upper_bound: Q
    layer_donor_acute_margin: Q
    layer_receiver_zero_winding_margin: Q
    early_receiver_edge_upper_bound: Q
    early_receiver_zero_winding_margin: Q
    bridge_phase_error_norm_upper_bound: Q
    bridge_donor_acute_margin: Q
    release_form_decay_bounds: I
    release_scaled_form_norm_upper_bound: Q
    release_form_norm_upper_bound: Q
    release_phase_error_norm_upper_bound: Q
    release_donor_acute_margin: Q
    release_receiver_zero_winding_margin: Q
    initial_defect_norm_upper_bound: Q
    damping_lower_bound: Q
    defect_decay_bounds: I
    asymptotic_defect_upper_bound: Q
    uniform_velocity_defect_upper_bound: Q
    uniform_form_rate_norm_upper_bound: Q
    release_phase_margin: Q
    release_form_margin: Q
    velocity_defect_margin: Q
    initial_windings: tuple[int, int, int] | None
    layer_windings: tuple[int, int, int] | None
    release_windings: tuple[int, int, int] | None
    early_receiver_zero_winding_certified: bool
    donor_release_bridge_certified: bool
    uniform_defect_certified: bool
    release_phase_budget_certified: bool
    release_form_budget_certified: bool
    status: str
    unresolved_conditions: tuple[str, ...]
    proof_deformation_limit: Q = _BOUNDARY_PROOF_LIMIT
    layer_time: Q = _LAYER_TIME
    release_time: Q = _RELEASE_TIME
    release_phase_budget: Q = _PHASE_BUDGET
    velocity_defect_budget: Q = _DEFECT_BUDGET
    release_form_budget: Q = _FORM_BUDGET
    nominal_source_scale: Q = _SOURCE_SCALE
    nominal_initial_phases: tuple[Q, ...] = (Q(0),) * 27
    held_capacities: tuple[Q, ...] = (Q(1),) * 27
    phase_mean_leaf_dimension: int = 26
    initial_constituent_error_sums: tuple[Q, ...] = (Q(0),) * 3
    clock: str = "tau=(1023/1024)*t; u=gamma**2*(tau-400)"
    edge_potential: str = "1-cos(d)+lambda*cos(d)*(1-cos(9*d))**2"
    relative_preparation_dimension: int = 48
    first_order_exact_closure: bool = False
    trajectory_evaluated: bool = False
    eventual_receiver_acquisition: str = "unresolved"
    eventual_donor_retention: str = "unresolved"
    joined_capture: str = "unresolved"
    scope: tuple[str, ...] = (
        "same_two_ramp_full_54_coordinate_source_and_all_initial_error_directions",
        "fixed_equal_degree_four_support_held_unit_capacity_no_inputs_or_events",
        "uniform_own_current_derivative_source_cost_and_transport_bounds",
        "initial_constituent_error_means_zero_only_global_means_conserved",
        "receiver_zero_through_release_and_donors_one_on_initial_layer_bridge",
        "release_phase_ball_and_all_future_form_and_actual_velocity_defect",
        "no_transferred_sine_source_verdict_or_numerical_trajectory",
        "no_eventual_winding_capture_or_physical_identification",
    )


def _assess_constitutive_receiver_source(
    *, deformation_bound, form_error_bound, phase_error_bound
) -> _ConstitutiveReceiverSource:
    """Admit primitives before rebuilding the fixed support and own-law bounds."""
    size = _exact(deformation_bound, "deformation_bound")
    rx = _exact(form_error_bound, "form_error_bound")
    rt = _exact(phase_error_bound, "phase_error_bound")
    if not 0 <= size <= _BOUNDARY_PROOF_LIMIT:
        raise ValueError("deformation_bound must lie in [0, 1/10000]")
    if rx < 0 or rt < 0:
        raise ValueError("preparation error bounds must be nonnegative")

    geometry = _derive_organized_triangle(mixing=_EQUAL_CONTROL)
    forms = tuple(
        _SOURCE_SCALE * (node % 9 - 4) if node < 18 else Q(0) for node in geometry.nodes
    )
    costs, gradient = _prepared_form_costs(
        form=forms, weighted_edges=geometry.weighted_edges
    )
    internal = sum(
        (
            cost
            for (a, b, _), cost in zip(geometry.weighted_edges, costs)
            if a // 9 == b // 9
        ),
        Q(0),
    )
    nominal_cost = sum(costs, Q(0))
    gradient_l1 = sum(map(abs, gradient), Q(0))
    gradient_norm = sqrt(I(sum(value**2 for value in gradient)))
    total_weight = sum((weight for _, _, weight in geometry.weighted_edges), Q(0))
    # |P(d)| <= 9**4*d**4/4 and every initial edge gap is at most 2*rt.
    deformation_cost = total_weight * 4 * 9**4 * size * rt**4
    correction = gradient_l1 * rx + 108 * (rx**2 + rt**2) + deformation_cost
    storage = I(nominal_cost - gradient_l1 * rx, nominal_cost + correction)

    pi = pi_interval()
    gamma = geometry.gamma_bounds
    epsilon = gamma**2
    root = sqrt(I(27))
    root_two = sqrt(I(2))
    alpha = 2 * pi / 9
    current_bound = 1 + Q(57, 2) * size
    derivative_bound = 1 + 328 * size
    forcing = root * current_bound
    lipschitz = 2 * derivative_bound
    # The admitted equal support has L >= (2/5)I on the global mean-zero leaf.
    gap = _LAPLACIAN_GAP * geometry.clock_factor
    initial_norm = gamma.hi * (sqrt(I(sum(x**2 for x in forms))).hi + root.hi * rx)
    proxy = sqrt(I(120)) * abs(gamma * _SOURCE_SCALE - alpha)
    shift = root.hi * (rt + gamma.hi * rx)

    def transit(time):
        decay = I(*_negative_exp_bounds(gap * time))
        form, remainder = _prepared_transit_radii(
            time=time,
            decay_upper=decay.hi,
            gap_lower=gap,
            initial_norm_upper=initial_norm,
            feedback_upper=epsilon.hi,
            forcing_upper=forcing.hi,
        )
        return decay, form, proxy.hi + shift + remainder

    layer_decay, layer_form, layer_phase = transit(_LAYER_TIME)
    release_decay, release_form, release_phase = transit(_RELEASE_TIME)
    bridge_phase = (
        proxy.hi + shift + layer_form + epsilon.hi * _RELEASE_TIME * forcing.hi
    )

    def donor_margin(radius):
        return pi.lo / 2 - alpha.hi - root_two.hi * radius

    def receiver_margin(radius):
        return pi.lo - root_two.hi * radius

    # Commuting Markov heat kernels give a receiver source-edge bound of four
    # at every time. The actual nonlinear history is retained through release.
    if not 4 * 20**15 < 17**17:
        raise ArithmeticError("receiver heat-shape bound failed")
    early_edge = (
        4 * gamma.hi * _SOURCE_SCALE
        + 2 * rt
        + 4 * gamma.hi * rx
        + 4 * epsilon.hi * _RELEASE_TIME * current_bound
    )
    initial_defect = (
        gradient_norm.hi * (1 / gamma).hi / 4
        + 2 * root.hi * rx * (1 / gamma).hi
        + lipschitz * root.hi * rt
    )
    damping, defect_decay, floor, defect = _prepared_velocity_defect(
        time=_RELEASE_TIME,
        gap_lower=gap,
        feedback_upper=epsilon.hi,
        lipschitz_upper=lipschitz,
        forcing_upper=forcing.hi,
        initial_defect_upper=initial_defect,
    )
    release_form_norm = release_form * (1 / gamma).hi
    initial_ok = pi.lo > 2 * rt
    layer_ok = min(donor_margin(layer_phase), receiver_margin(layer_phase)) > 0
    release_ok = min(donor_margin(release_phase), receiver_margin(release_phase)) > 0
    early_ok = initial_ok and early_edge < pi.lo
    bridge_ok = layer_ok and donor_margin(bridge_phase) > 0
    phase_margin = _PHASE_BUDGET - release_phase
    form_margin = _FORM_BUDGET - release_form_norm
    defect_margin = _DEFECT_BUDGET - defect
    conditions = (
        (initial_ok, "initial_zero_winding_not_certified"),
        (layer_ok, "initial_layer_windings_not_certified"),
        (early_ok, "receiver_zero_through_release_not_certified"),
        (bridge_ok, "donor_release_bridge_not_certified"),
        (release_ok, "release_windings_not_certified"),
        (phase_margin > 0, "release_phase_budget_not_certified"),
        (form_margin > 0, "release_form_budget_not_certified"),
        (defect_margin > 0, "uniform_velocity_defect_budget_not_certified"),
    )
    unresolved = tuple(reason for passed, reason in conditions if not passed)
    return _ConstitutiveReceiverSource(
        deformation_bound=size,
        deformation_interval=(-size, size),
        form_error_bound=rx,
        phase_error_bound=rt,
        nodes=geometry.nodes,
        weighted_edges=geometry.weighted_edges,
        weighted_degrees=geometry.weighted_degrees,
        nominal_initial_forms=forms,
        nominal_initial_form_gradient=gradient,
        nominal_initial_internal_form_storage=internal,
        nominal_initial_contact_form_storage=nominal_cost - internal,
        nominal_initial_form_storage=nominal_cost,
        initial_form_gradient_l1=gradient_l1,
        nominal_form_gradient_norm_bounds=gradient_norm,
        initial_potential_deformation_upper_bound=deformation_cost,
        initial_storage_correction_upper_bound=correction,
        initial_storage_bounds=storage,
        gamma_bounds=gamma,
        slow_clock_factor_bounds=epsilon,
        edge_current_absolute_upper_bound=current_bound,
        edge_current_derivative_absolute_upper_bound=derivative_bound,
        forcing_norm_bounds=forcing,
        global_phase_lipschitz_bound=lipschitz,
        semigroup_gap_lower_bound=gap,
        scaled_initial_norm_upper_bound=initial_norm,
        reference_phase_bounds=tuple(
            alpha * (node % 9 - 4) if node < 18 else I(0) for node in geometry.nodes
        ),
        proxy_reference_distance_bounds=proxy,
        layer_form_decay_bounds=layer_decay,
        layer_scaled_form_norm_upper_bound=layer_form,
        layer_form_norm_upper_bound=layer_form * (1 / gamma).hi,
        layer_phase_error_norm_upper_bound=layer_phase,
        layer_donor_acute_margin=donor_margin(layer_phase),
        layer_receiver_zero_winding_margin=receiver_margin(layer_phase),
        early_receiver_edge_upper_bound=early_edge,
        early_receiver_zero_winding_margin=pi.lo - early_edge,
        bridge_phase_error_norm_upper_bound=bridge_phase,
        bridge_donor_acute_margin=donor_margin(bridge_phase),
        release_form_decay_bounds=release_decay,
        release_scaled_form_norm_upper_bound=release_form,
        release_form_norm_upper_bound=release_form_norm,
        release_phase_error_norm_upper_bound=release_phase,
        release_donor_acute_margin=donor_margin(release_phase),
        release_receiver_zero_winding_margin=receiver_margin(release_phase),
        initial_defect_norm_upper_bound=initial_defect,
        damping_lower_bound=damping,
        defect_decay_bounds=defect_decay,
        asymptotic_defect_upper_bound=floor,
        uniform_velocity_defect_upper_bound=defect,
        uniform_form_rate_norm_upper_bound=gamma.hi * defect,
        release_phase_margin=phase_margin,
        release_form_margin=form_margin,
        velocity_defect_margin=defect_margin,
        initial_windings=(0, 0, 0) if initial_ok else None,
        layer_windings=(1, 1, 0) if layer_ok else None,
        release_windings=(1, 1, 0) if release_ok else None,
        early_receiver_zero_winding_certified=early_ok,
        donor_release_bridge_certified=bridge_ok,
        uniform_defect_certified=defect_margin > 0,
        release_phase_budget_certified=phase_margin > 0,
        release_form_budget_certified=form_margin > 0,
        relative_preparation_dimension=24 * (int(rx > 0) + int(rt > 0)),
        status=(
            "certified_uniform_constitutive_source" if not unresolved else "unavailable"
        ),
        unresolved_conditions=unresolved,
    )
