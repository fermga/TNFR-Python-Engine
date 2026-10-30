"""Shared exact preparation for conditional full-law sine estimates.

This owner revalidates primitive source data and computes the common weighted
geometry, coefficient bounds and initial norm. It neither evolves the source
nor decides a theorem's horizon, phase preparation or endpoint admission.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real
from ..dynamics.relational import RelationalExchangeModel
from ..mathematics._rational_interval import (
    INTERVAL_METHOD,
    I,
    cos,
    pi_interval,
    sin,
    sqrt,
)
from ._sine_admission import _sector_source_admission, _sine_model_coefficients
from .phase_cycle_geometry import PhaseCycleGeometry, _rebuild
from .relational_observations import _ordered
from .relational_sine_comparison import SineExchangeComparison
from .relational_sine_pattern import SineRelativePattern
from .reversible_eigenmode_reference import _MAX_RATIONAL_EXPONENT, _negative_exp_bounds
from .structural_diffusion import _exact_flow_gap_from_rationals


def _prepared_coordinate_vector(values, label, size):
    """Admit a registered ordered state or observation row exactly.

    The caller declares the coordinate count and meaning. This helper retains
    represented real values without float materialization, consumes at most
    one excess item, and supplies no state-class or provenance certificate.
    """
    raw = _ordered(values, label, limit=size + 1)
    if len(raw) != size:
        raise ValueError(f"{label} requires {size} registered coordinates")
    return tuple(
        exact_or_represented_real(value, f"{label}[{i}]") for i, value in enumerate(raw)
    )


def _prepared_form_costs(*, form, weighted_edges):
    """Return exact edge storage and its nodal gradient for admitted data.

    Callers supply finite rational forms and validated nonnegative weighted
    edges indexed in that form tuple. Costs retain contact edges; grouping,
    source-error allowances and preparation budgets belong to the consumer.
    No source or trajectory verdict is inferred by this algebraic kernel.
    """
    gradient = [Q(0)] * len(form)
    costs = []
    for a, b, weight in weighted_edges:
        difference = form[a] - form[b]
        gradient[a] += weight * difference
        gradient[b] -= weight * difference
        costs.append(weight * difference**2 / 2)
    return tuple(costs), tuple(gradient)


def _prepared_heat_error_radii(
    *,
    initial_form_error,
    initial_phase_error,
    gamma_upper,
    phase_growth_upper,
    form_convolution_factor,
):
    """Return full form/phase error radii from an admitted heat-integral proof.

    Callers prove the complete reciprocal rows, mean-zero form difference,
    spectral/Lipschitz bounds and the whole-time growth factor. Identical
    prescribed additive form inputs cancel from the compared rows; estimates
    for a common input do not transfer to different input recipes. Identical
    additive jumps preserve the difference only when applied to both compared
    states. No event, source, clock or horizon is selected by this arithmetic.
    Exact rational errors remain exact even below the interval grid.
    """
    form, phase, gamma, growth, convolution = (
        exact_or_represented_real(value, name)
        for name, value in (
            ("initial_form_error", initial_form_error),
            ("initial_phase_error", initial_phase_error),
            ("gamma_upper", gamma_upper),
            ("phase_growth_upper", phase_growth_upper),
            ("form_convolution_factor", form_convolution_factor),
        )
    )
    if min(form, phase, gamma, convolution) < 0 or growth < 1:
        raise ValueError("heat error bounds require nonnegative inputs and growth >= 1")
    phase_radius = (phase + gamma * form) * growth
    return form + gamma * convolution * phase_radius, phase_radius


def _prepared_degree_four_probe_remainder(
    *, gamma_upper, initial_form_upper_bound, phase_jump_upper_bound, elapsed_time
):
    """Bound a finite matched phase-probe increment for the quarter-rate law.

    The caller admits degree-four unit support, both reciprocal smooth-sine
    rows, held unit capacity, an unchanged initial form and an initial edge
    phase jump bounded by ``phase_jump_upper_bound``. Heat contraction and
    the global sine derivative retain background motion and probe feedback.
    The result bounds the coordinate maximum error from the initial slope;
    it neither selects a protocol nor certifies its state, work or identity.
    """
    gamma, initial, jump, time = (
        exact_or_represented_real(value, name)
        for name, value in (
            ("gamma_upper", gamma_upper),
            ("initial_form_upper_bound", initial_form_upper_bound),
            ("phase_jump_upper_bound", phase_jump_upper_bound),
            ("elapsed_time", elapsed_time),
        )
    )
    if min(gamma, initial, jump, time) < 0:
        raise ValueError("finite response bounds require nonnegative inputs")
    if 2 * gamma**2 * time**2 > Q(1, 2):
        raise ValueError("finite response comparison is outside its domain")
    return (
        2 * gamma * jump * time**2
        + 2 * gamma**2 * jump * (initial + gamma * time) * time**2
        + Q(4, 3) * gamma**3 * jump * time**3
    )


@dataclass(frozen=True)
class _PreparedBridgeExchange:
    """Conditional finite exchange bounds, without source authentication.

    Initial form and phase differences are host-port minus donor-port form
    and donor-port minus host-port phase, respectively. The phase endpoints
    belong to one supplied continuous real lift. The separately named initial
    mean rate and its integrated remainder determine the increment bounds;
    the other rate fields enclose the complete declared window.

    ``observed_sign`` concerns the connected-minus-disconnected donor-mean
    form increment after all four reading errors. ``retention_status`` is
    independent: it concerns only the supplied old-edge chart collars. Their
    association with actual edges, cycles and initial winding is a caller
    premise. Neither field admits a graph, source, event or stored report.
    """

    initial_form_difference_bounds: tuple[Q, Q]
    initial_phase_gap_bounds: tuple[Q, Q]
    bridge_weight_bounds: tuple[Q, Q]
    gamma_bounds: tuple[Q, Q]
    initial_form_upper_bound: Q
    weighted_degree_upper_bound: Q
    elapsed_time: Q
    donor_node_count: int
    reading_error_bounds: tuple[Q, Q, Q, Q]
    old_edge_chart_margins: tuple[Q, ...] | None
    effective_initial_form_difference_bounds: tuple[Q, Q]
    form_upper_bound: Q
    form_rate_upper_bound: Q
    phase_rate_upper_bound: Q
    form_displacement_upper_bound: Q
    phase_displacement_upper_bound: Q
    form_difference_bounds: I
    phase_gap_bounds: I
    initial_sine_bounds: I
    sine_bounds: I
    initial_donor_mean_form_rate_bounds: I
    donor_total_form_rate_bounds: I
    host_total_form_rate_bounds: I
    donor_mean_form_rate_bounds: I
    donor_total_phase_rate_bounds: I
    host_total_phase_rate_bounds: I
    donor_mean_phase_rate_bounds: I
    increment_remainder_upper_bound: Q
    true_increment_bounds: I
    reading_error_upper_bound: Q
    observed_increment_bounds: I
    observed_sign: str
    retention_margins: tuple[Q, ...] | None
    retention_status: str
    interval_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "two_disjoint_finite_undirected_constituents_and_one_positive_held_bridge",
        "identical_full_initial_state_in_the_connected_and_disconnected_branches",
        "same_nonnegative_weighted_support_in_both_uniform_quarter_sine_rows",
        "held_unit_capacities_fixed_gamma_and_declared_structural_clock",
        "no_subsequent_input_event_or_conductance_change",
        "raw_intervals_and_global_background_bounds_are_supplied_not_authenticated",
        "arbitrary_nonzero_constituent_means_and_hidden_states_are_retained",
        "donor_mean_normalization_is_explicit_host_outputs_are_totals_not_means",
        "four_reading_errors_allow_arbitrary_correlations",
        "old_chart_retention_requires_independent_edge_cycle_and_lift_association",
        "indeterminate_bounds_are_not_observed_dynamical_failures",
        "no_event_admission_work_reservoir_coarse_closure_or_trajectory_claim",
    )


def _prepared_old_edge_chart_margins(values):
    """Normalize optional old-edge collars without authenticating their edges."""
    if values is None:
        return None
    raw = _ordered(values, "old_edge_chart_margins", limit=4097)
    if not 1 <= len(raw) <= 4096:
        raise ValueError("old_edge_chart_margins requires 1..4096 entries")
    return tuple(
        exact_or_represented_real(value, f"old_edge_chart_margins[{index}]")
        for index, value in enumerate(raw)
    )


def _admit_bridge_exchange_inputs(
    *,
    initial_form_difference_bounds,
    initial_phase_gap_bounds,
    bridge_weight_bounds,
    gamma_bounds,
    initial_form_upper_bound,
    weighted_degree_upper_bound,
    elapsed_time,
    donor_node_count,
    reading_error_bounds,
    old_edge_chart_margins,
    allow_zero_weight=False,
):
    """Normalize common raw bridge premises without interval construction."""
    a0, d0, weight, gamma = (
        _prepared_coordinate_vector(raw, label, 2)
        for raw, label in (
            (initial_form_difference_bounds, "initial_form_difference_bounds"),
            (initial_phase_gap_bounds, "initial_phase_gap_bounds"),
            (bridge_weight_bounds, "bridge_weight_bounds"),
            (gamma_bounds, "gamma_bounds"),
        )
    )
    initial, degree, time = (
        exact_or_represented_real(raw, label)
        for raw, label in (
            (initial_form_upper_bound, "initial_form_upper_bound"),
            (weighted_degree_upper_bound, "weighted_degree_upper_bound"),
            (elapsed_time, "elapsed_time"),
        )
    )
    errors = _prepared_coordinate_vector(
        reading_error_bounds, "reading_error_bounds", 4
    )
    margins = _prepared_old_edge_chart_margins(old_edge_chart_margins)
    if type(donor_node_count) is not int:
        raise TypeError("donor_node_count must be an ordinary Python integer")
    if donor_node_count <= 0:
        raise ValueError("donor_node_count must be positive")
    for bounds, label in (
        (a0, "initial_form_difference_bounds"),
        (d0, "initial_phase_gap_bounds"),
        (weight, "bridge_weight_bounds"),
        (gamma, "gamma_bounds"),
    ):
        if bounds[0] > bounds[1]:
            raise ValueError(f"{label} lower endpoint must not exceed its upper")
    if weight[0] < 0 or (weight[0] == 0 and not allow_zero_weight):
        if allow_zero_weight:
            raise ValueError("bridge_weight_bounds must be nonnegative")
        raise ValueError("bridge_weight_bounds must be strictly positive")
    if gamma[0] < 0:
        raise ValueError("gamma_bounds must be nonnegative")
    if min(initial, degree, time, *errors) < 0:
        raise ValueError("background, horizon and reading errors must be nonnegative")
    if degree < weight[1]:
        raise ValueError(
            "weighted_degree_upper_bound must cover the bridge upper bound"
        )
    if margins is not None and min(margins) < 0:
        raise ValueError("old_edge_chart_margins must be nonnegative")
    effective_a0 = (max(a0[0], -2 * initial), min(a0[1], 2 * initial))
    if effective_a0[0] > effective_a0[1]:
        raise ValueError(
            "initial form difference and global form ceiling are inconsistent"
        )

    return (
        a0,
        d0,
        weight,
        gamma,
        initial,
        degree,
        time,
        donor_node_count,
        errors,
        margins,
        effective_a0,
    )


def _prepared_bridge_motion(*, initial, mobility_degree, gamma_upper, time):
    """Exact full-row motion arithmetic after primitive and law admission.

    ``mobility_degree`` bounds nu_max times the actual weighted degree. It
    is a rate coefficient, never a replacement for the support degree.
    The last result is integral_0^h (2*X(s)+2*g*T(s)) ds.
    """
    k, g = mobility_degree, gamma_upper
    form_ceiling = initial + k * g * time / 4
    form_rate = k * form_ceiling / 2 + k * g / 4
    phase_rate = g * k * form_ceiling / 2
    form_displacement = (k * initial / 2 + k * g / 4) * time + k**2 * g * time**2 / 16
    phase_displacement = g * k * initial * time / 2 + k**2 * g**2 * time**2 / 16
    variation_integral = (
        k * (initial * (1 + g**2) + g / 2) * time**2 / 2
        + k**2 * g * (1 + g**2) * time**3 / 24
    )
    return (
        form_ceiling,
        form_rate,
        phase_rate,
        form_displacement,
        phase_displacement,
        variation_integral,
    )


def _prepared_bridge_parameter_growth(
    *, maximum_capacity, initial, mobility_degree, gamma_upper, time
):
    """Shared complete-row first-parameter sensitivity coefficients.

    Callers admit all primitive and law premises before this numerical owner.
    It returns alpha, alpha*h, G>=exp(alpha*h), b0 and b1, so the complete
    state sensitivity to one fixed edge weight is at most
    G*(b0*s+b1*s**2/2) on 0<=s<=h. The same arithmetic bounds a finite
    matched difference after multiplying by the edge conductance.
    The exponent limit is a numerical work domain, not a theorem horizon.
    """
    g = gamma_upper
    growth_rate = g * mobility_degree / 2
    growth_exponent = growth_rate * time
    if growth_exponent > _MAX_RATIONAL_EXPONENT:
        raise ValueError("matched bridge growth exponent must not exceed 4096")
    growth = 1 / _negative_exp_bounds(growth_exponent)[0]
    forcing_constant = maximum_capacity * ((1 + g) * initial / 2 + g / 4)
    forcing_linear = maximum_capacity * (1 + g) * mobility_degree * g / 8
    return growth_rate, growth_exponent, growth, forcing_constant, forcing_linear


def _prepared_bridge_exchange(
    *,
    initial_form_difference_bounds,
    initial_phase_gap_bounds,
    bridge_weight_bounds,
    gamma_bounds,
    initial_form_upper_bound,
    weighted_degree_upper_bound,
    elapsed_time,
    donor_node_count,
    reading_error_bounds,
    old_edge_chart_margins=None,
) -> _PreparedBridgeExchange:
    """Enclose a matched finite bridge increment from raw primitive bounds.

    Callers prove two initially disjoint finite undirected constituents, one
    held positive bridge of conductance w, identical complete initial states
    on both branches, and held unit capacities. On each branch's own support
    the complete law is x'=(-L*x+gamma*S(theta))/4, theta'=gamma*L*x/4, with
    the same symmetric nonnegative weights in L and S. Gamma is held fixed
    and nonnegative; time uses one declared structural clock. The disconnected
    donor mean is then constant. No missing support, state, clock, history,
    event or law premise is reconstructed from these scalar inputs.

    Supply ordered endpoint pairs for a=x_host-x_donor, the continuously
    lifted d=theta_donor-theta_host, w and gamma. The initial global form
    ceiling B0, joined maximum weighted-degree ceiling K, horizon h and four
    reading-error radii are nonnegative. The conservative admission domain
    requires K>=w_upper and a nonempty intersection of the supplied a box
    with [-2*B0,2*B0]. Both the original and this effective box are retained.
    The donor averaging count must be a positive ordinary Python integer;
    a host mean additionally needs its own count and is not returned here.

    Optional old-edge margins are distances from the associated initial edge
    lifts to their prescribed chart boundaries. A requested sequence must
    contain 1..4096 entries (an instrument work limit), all nonnegative.
    None means unrequested. Zero or an exhausted margin is inconclusive.
    Scalar collars cannot establish initial cycles or their winding numbers.

    All primitives are admitted before interval/trigonometric work. Exact
    rational values are preserved; other real scalars use shared represented
    admission. No incoming interval object, cached report or passing flag is
    consumed. With g=gamma_upper, heat contraction gives B(s)=B0+K*g*s/4.
    Integrated complete-row bounds give displacements X(h), T(h). Integrating
    the unit sine Lipschitz estimate yields E, so the true mean increment is
    in h*R0+[-E,E], with R0 the initial bridge mean-rate interval. Requested
    charts are retained only when every margin-2*T(h) is strictly positive.
    """
    a0, d0, weight, gamma, initial, degree, time, n, errors, margins, effective_a0 = (
        _admit_bridge_exchange_inputs(
            initial_form_difference_bounds=initial_form_difference_bounds,
            initial_phase_gap_bounds=initial_phase_gap_bounds,
            bridge_weight_bounds=bridge_weight_bounds,
            gamma_bounds=gamma_bounds,
            initial_form_upper_bound=initial_form_upper_bound,
            weighted_degree_upper_bound=weighted_degree_upper_bound,
            elapsed_time=elapsed_time,
            donor_node_count=donor_node_count,
            reading_error_bounds=reading_error_bounds,
            old_edge_chart_margins=old_edge_chart_margins,
        )
    )
    (
        form_ceiling,
        form_rate,
        phase_rate,
        form_displacement,
        phase_displacement,
        variation_integral,
    ) = _prepared_bridge_motion(
        initial=initial, mobility_degree=degree, gamma_upper=gamma[1], time=time
    )
    remainder = weight[1] / (4 * n) * variation_integral
    error = sum(errors, Q(0))
    residual_margins = (
        None
        if margins is None
        else tuple(value - 2 * phase_displacement for value in margins)
    )
    retention = (
        "not_requested"
        if residual_margins is None
        else "certified" if min(residual_margins) > 0 else "indeterminate"
    )

    # Interval work starts only after complete primitive and domain admission.
    w_box, gamma_box = I(*weight), I(*gamma)
    initial_sine = sin(I(*d0))
    initial_mean_rate = w_box * (I(*effective_a0) - gamma_box * initial_sine) / (4 * n)
    form_box = I(
        effective_a0[0] - 2 * form_displacement,
        effective_a0[1] + 2 * form_displacement,
    )
    phase_box = I(d0[0] - 2 * phase_displacement, d0[1] + 2 * phase_displacement)
    sine = sin(phase_box)
    donor_form_rate = w_box * (form_box - gamma_box * sine) / 4
    donor_phase_rate = -gamma_box * w_box * form_box / 4
    increment = time * initial_mean_rate + I(-remainder, remainder)
    observed = increment + I(-error, error)
    sign = (
        "negative"
        if observed.hi < 0
        else "positive" if observed.lo > 0 else "indeterminate"
    )
    return _PreparedBridgeExchange(
        initial_form_difference_bounds=a0,
        initial_phase_gap_bounds=d0,
        bridge_weight_bounds=weight,
        gamma_bounds=gamma,
        initial_form_upper_bound=initial,
        weighted_degree_upper_bound=degree,
        elapsed_time=time,
        donor_node_count=n,
        reading_error_bounds=errors,
        old_edge_chart_margins=margins,
        effective_initial_form_difference_bounds=effective_a0,
        form_upper_bound=form_ceiling,
        form_rate_upper_bound=form_rate,
        phase_rate_upper_bound=phase_rate,
        form_displacement_upper_bound=form_displacement,
        phase_displacement_upper_bound=phase_displacement,
        form_difference_bounds=form_box,
        phase_gap_bounds=phase_box,
        initial_sine_bounds=initial_sine,
        sine_bounds=sine,
        initial_donor_mean_form_rate_bounds=initial_mean_rate,
        donor_total_form_rate_bounds=donor_form_rate,
        host_total_form_rate_bounds=-donor_form_rate,
        donor_mean_form_rate_bounds=donor_form_rate / n,
        donor_total_phase_rate_bounds=donor_phase_rate,
        host_total_phase_rate_bounds=-donor_phase_rate,
        donor_mean_phase_rate_bounds=donor_phase_rate / n,
        increment_remainder_upper_bound=remainder,
        true_increment_bounds=increment,
        reading_error_upper_bound=error,
        observed_increment_bounds=observed,
        observed_sign=sign,
        retention_margins=residual_margins,
        retention_status=retention,
    )


@dataclass(frozen=True)
class _PreparedCapacityBridgeExchange:
    """Two conditional constituent readings under positive held capacities.

    Unprefixed increment fields concern the arithmetic donor mean contrast.
    The initial rate is the matched connected-minus-disconnected rate; the
    whole-window contrast retains both branches' internal motion. Fields
    prefixed ``inverse_capacity_`` use sum(x_i/nu_i)/sum(1/nu_i) instead,
    with an independent four-reading error model. Its unnormalized totals
    are reciprocal; ordinary constituent totals need not be.

    Bounds are conditional raw premises, never state/history authentication.
    Chart retention is independent of either reading's strict sign.
    """

    initial_form_difference_bounds: tuple[Q, Q]
    initial_phase_gap_bounds: tuple[Q, Q]
    bridge_weight_bounds: tuple[Q, Q]
    gamma_bounds: tuple[Q, Q]
    initial_form_upper_bound: Q
    weighted_degree_upper_bound: Q
    elapsed_time: Q
    donor_node_count: int
    reading_error_bounds: tuple[Q, Q, Q, Q]
    old_edge_chart_margins: tuple[Q, ...] | None
    capacity_bounds: tuple[Q, Q]
    donor_port_capacity_bounds: tuple[Q, Q]
    donor_internal_capacity_contrast_upper_bound: Q | None
    fallback_donor_internal_capacity_contrast_upper_bound: Q
    effective_donor_internal_capacity_contrast_upper_bound: Q
    inverse_capacity_mass_bounds: tuple[Q, Q] | None
    effective_inverse_capacity_mass_bounds: tuple[Q, Q]
    inverse_capacity_reading_error_bounds: tuple[Q, Q, Q, Q] | None
    effective_initial_form_difference_bounds: tuple[Q, Q]
    mobility_degree_upper_bound: Q
    form_upper_bound: Q
    form_rate_upper_bound: Q
    phase_rate_upper_bound: Q
    form_displacement_upper_bound: Q
    phase_displacement_upper_bound: Q
    rate_variation_integral_upper_bound: Q
    form_difference_bounds: I
    phase_gap_bounds: I
    initial_sine_bounds: I
    sine_bounds: I
    initial_donor_mean_form_rate_bounds: I
    internal_mean_rate_contrast_upper_bound: Q
    donor_mean_form_rate_contrast_bounds: I
    increment_remainder_upper_bound: Q
    true_increment_bounds: I
    reading_error_upper_bound: Q
    observed_increment_bounds: I
    observed_sign: str
    inverse_capacity_initial_donor_mean_form_rate_bounds: I
    inverse_capacity_donor_total_form_rate_bounds: I
    inverse_capacity_host_total_form_rate_bounds: I
    inverse_capacity_donor_mean_form_rate_bounds: I
    inverse_capacity_donor_total_phase_rate_bounds: I
    inverse_capacity_host_total_phase_rate_bounds: I
    inverse_capacity_donor_mean_phase_rate_bounds: I
    inverse_capacity_increment_remainder_upper_bound: Q
    inverse_capacity_true_increment_bounds: I
    inverse_capacity_reading_error_upper_bound: Q | None
    inverse_capacity_observed_increment_bounds: I | None
    inverse_capacity_observed_sign: str
    retention_margins: tuple[Q, ...] | None
    retention_status: str
    interval_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "two_disjoint_finite_undirected_constituents_and_one_positive_held_bridge",
        "identical_full_initial_state_and_capacities_in_both_matched_branches",
        "same_symmetric_nonnegative_weights_in_both_uniform_quarter_sine_rows",
        "strictly_positive_held_capacity_multiplies_both_complete_nodal_rows",
        "fixed_nonnegative_gamma_and_clock_without_later_input_event_or_weight_change",
        "global_capacity_bounds_apply_to_every_node_in_both_constituents",
        "arithmetic_contrast_retains_internal_motion_of_both_branches",
        "inverse_capacity_mean_uses_each_members_held_mass_and_separate_errors",
        "only_inverse_capacity_constituent_totals_have_reciprocal_bridge_currents",
        "raw_scalar_premises_do_not_authenticate_support_source_or_capacity_association",
        "old_chart_retention_requires_independent_edge_cycle_and_lift_association",
        "no_zero_capacity_limit_event_work_coarse_closure_or_trajectory_claim",
    )


@dataclass(frozen=True, kw_only=True)
class _PreparedMatchedCapacityBridgeExchange(_PreparedCapacityBridgeExchange):
    """Shared-preparation refinement with exact conductance-scaled endpoints.

    ``matched_state_difference_upper_bound`` bounds the sum of the form and
    lifted-phase maximum differences in the declared coordinate normalization.
    This comparison norm is not a new structural observable. The exact tuple
    endpoints scale conductance after interval coefficient construction. They
    retain continuity below the fixed dyadic display grid; inherited interval
    fields are outward-rounded displays, while signs use the exact endpoints.
    The selected internal rate and remainder are the smaller of the old valid
    bound and the new matched bound. Measurement errors remain independent.
    """

    difference_growth_rate_upper_bound: Q
    difference_growth_exponent: Q
    difference_growth_upper_bound: Q
    difference_forcing_constant_upper_bound: Q
    difference_forcing_linear_upper_bound: Q
    matched_state_difference_upper_bound: Q
    integrated_matched_state_difference_upper_bound: Q
    matched_internal_mean_rate_upper_bound: Q
    matched_internal_increment_remainder_upper_bound: Q
    matched_increment_remainder_upper_bound: Q
    independent_increment_remainder_upper_bound: Q
    exact_true_increment_bounds: tuple[Q, Q]
    exact_observed_increment_bounds: tuple[Q, Q]
    inverse_capacity_exact_true_increment_bounds: tuple[Q, Q]
    inverse_capacity_exact_observed_increment_bounds: tuple[Q, Q] | None
    scope: tuple[str, ...] = (
        "two_disjoint_finite_undirected_constituents_and_one_nonnegative_held_bridge",
        "identical_full_initial_state_and_capacities_in_both_matched_branches",
        "same_symmetric_nonnegative_weights_in_both_uniform_quarter_sine_rows",
        "zero_weight_is_absent_from_both_rows_unlike_native_bare_phase_support",
        "strictly_positive_held_capacity_multiplies_both_complete_nodal_rows",
        "fixed_nonnegative_gamma_and_clock_without_later_input_event_or_weight_change",
        "global_capacity_bounds_apply_to_every_node_in_both_constituents",
        "matched_full_state_difference_retains_both_internal_dynamic_rows",
        "inverse_capacity_mean_uses_each_members_held_mass_and_separate_errors",
        "only_inverse_capacity_constituent_totals_have_reciprocal_bridge_currents",
        "exact_endpoint_conductance_scaling_precedes_dyadic_display_rounding",
        "strict_sign_uses_exact_endpoints_and_is_not_uniform_on_families_containing_zero",
        "raw_scalar_premises_do_not_authenticate_support_source_or_capacity_association",
        "old_chart_retention_requires_independent_edge_cycle_and_lift_association",
        "no_zero_capacity_limit_event_work_coarse_closure_or_trajectory_claim",
    )


def _prepared_capacity_bridge_exchange(
    *,
    initial_form_difference_bounds,
    initial_phase_gap_bounds,
    bridge_weight_bounds,
    gamma_bounds,
    initial_form_upper_bound,
    weighted_degree_upper_bound,
    elapsed_time,
    donor_node_count,
    reading_error_bounds,
    capacity_bounds,
    donor_port_capacity_bounds,
    donor_internal_capacity_contrast_upper_bound=None,
    inverse_capacity_mass_bounds=None,
    inverse_capacity_reading_error_bounds=None,
    old_edge_chart_margins=None,
) -> _PreparedCapacityBridgeExchange:
    """Enclose positive-capacity exchange with independent-branch motion bounds.

    Retains the original positive-bridge contract and numerical results. See
    the shared implementation for complete raw-premise and observation domains.
    """
    return _prepared_capacity_bridge_exchange_impl(
        initial_form_difference_bounds=initial_form_difference_bounds,
        initial_phase_gap_bounds=initial_phase_gap_bounds,
        bridge_weight_bounds=bridge_weight_bounds,
        gamma_bounds=gamma_bounds,
        initial_form_upper_bound=initial_form_upper_bound,
        weighted_degree_upper_bound=weighted_degree_upper_bound,
        elapsed_time=elapsed_time,
        donor_node_count=donor_node_count,
        reading_error_bounds=reading_error_bounds,
        capacity_bounds=capacity_bounds,
        donor_port_capacity_bounds=donor_port_capacity_bounds,
        donor_internal_capacity_contrast_upper_bound=donor_internal_capacity_contrast_upper_bound,
        inverse_capacity_mass_bounds=inverse_capacity_mass_bounds,
        inverse_capacity_reading_error_bounds=inverse_capacity_reading_error_bounds,
        old_edge_chart_margins=old_edge_chart_margins,
    )


def _prepared_matched_capacity_bridge_exchange(
    *,
    initial_form_difference_bounds,
    initial_phase_gap_bounds,
    bridge_weight_bounds,
    gamma_bounds,
    initial_form_upper_bound,
    weighted_degree_upper_bound,
    elapsed_time,
    donor_node_count,
    reading_error_bounds,
    capacity_bounds,
    donor_port_capacity_bounds,
    donor_internal_capacity_contrast_upper_bound=None,
    inverse_capacity_mass_bounds=None,
    inverse_capacity_reading_error_bounds=None,
    old_edge_chart_margins=None,
) -> _PreparedMatchedCapacityBridgeExchange:
    """Refine the arithmetic contrast using matched complete-state dynamics.

    All positive-capacity premises of the original reader remain. Only this
    opt-in reader admits a nonnegative bridge interval, including exact zero.
    Both rows use weighted support, so their matched states agree at zero
    weight. This limit does not transfer to a bare-neighbor phase law.

    The off-branch Markov semigroup and sine Lipschitz bound compare form and
    phase together. With k=M*K, g=gamma_upper, alpha=g*k/2, and B0 the initial
    form ceiling, set b0=M*((1+g)*B0/2+g/4), b1=M*(1+g)*k*g/8. A shared exact
    exponential bound G>=exp(alpha*h) gives Y=w_upper*G*(b0*h+b1*h**2/2)
    and H=w_upper*G*(b0*h**2/2+b1*h**3/6). These bound the sum of complete
    state differences and its time integral. The internal arithmetic rate
    is bounded by Lambda*max(1,g)*Y/(2*nD), and its increment by the same
    coefficient times H. The bridge contribution keeps the previous bound.
    Taking the smaller of the old and new valid allowances cannot worsen it.

    The implementation domain alpha*h<=4096 applies even at zero weight;
    it is the shared exponential owner's work limit, not a theorem horizon.
    Every primitive and derived domain is admitted before exponential or
    interval work. Exact rational endpoint tuples multiply raw conductance
    after interval coefficient evaluation. They retain arbitrary subgrid
    scaling and exact zero; display intervals keep their dyadic rounding.
    Signs use exact endpoints. Each reading's four errors are added afterward
    and need not vanish at disconnection. Old-chart retention stays separate.
    """
    return _prepared_capacity_bridge_exchange_impl(
        initial_form_difference_bounds=initial_form_difference_bounds,
        initial_phase_gap_bounds=initial_phase_gap_bounds,
        bridge_weight_bounds=bridge_weight_bounds,
        gamma_bounds=gamma_bounds,
        initial_form_upper_bound=initial_form_upper_bound,
        weighted_degree_upper_bound=weighted_degree_upper_bound,
        elapsed_time=elapsed_time,
        donor_node_count=donor_node_count,
        reading_error_bounds=reading_error_bounds,
        capacity_bounds=capacity_bounds,
        donor_port_capacity_bounds=donor_port_capacity_bounds,
        donor_internal_capacity_contrast_upper_bound=donor_internal_capacity_contrast_upper_bound,
        inverse_capacity_mass_bounds=inverse_capacity_mass_bounds,
        inverse_capacity_reading_error_bounds=inverse_capacity_reading_error_bounds,
        old_edge_chart_margins=old_edge_chart_margins,
        matched_response=True,
    )


def _prepared_capacity_bridge_exchange_impl(
    *,
    initial_form_difference_bounds,
    initial_phase_gap_bounds,
    bridge_weight_bounds,
    gamma_bounds,
    initial_form_upper_bound,
    weighted_degree_upper_bound,
    elapsed_time,
    donor_node_count,
    reading_error_bounds,
    capacity_bounds,
    donor_port_capacity_bounds,
    donor_internal_capacity_contrast_upper_bound=None,
    inverse_capacity_mass_bounds=None,
    inverse_capacity_reading_error_bounds=None,
    old_edge_chart_margins=None,
    matched_response=False,
) -> _PreparedCapacityBridgeExchange:
    """Enclose finite bridge contrasts under declared positive capacities.

    The common bridge premises and signs are those of the unit-capacity
    reader. Here both complete rows additionally multiply by the same held
    positive diagonal D: x'=D*(-L*x+gamma*S(theta))/4 and
    theta'=gamma*D*L*x/4. The matched branches share their entire initial
    form, phase and capacity vectors. The arithmetic disconnected donor
    mean need not be constant; its internal motion remains in this contrast.

    ``capacity_bounds=(m,M)`` bounds EVERY node and requires m>0. The positive
    donor-port interval must be wholly contained in this global interval,
    a conservative raw-input domain. K is still an actual weighted-degree
    ceiling, whereas M*K is the derived mobility-degree coefficient.
    Optional Lambda bounds sum_internal,D w_ij*abs(nu_i-nu_j). Its effective
    value is the smaller of the supplied bound and (M-m)*nD*K/2; omission
    uses that fallback. A sharper supplied Lambda is an additional premise.

    The inverse-capacity reading uses R_D=sum_D(1/nu_i). An optional ordered
    mass interval must itself be strictly positive; it is intersected with
    [nD/M,nD/m], retaining raw and effective endpoints. A disjoint interval
    rejects. These scalar bounds do not prove a realized capacity vector.
    Exact reciprocal endpoints precede outward rounding, so positive masses
    below the interval grid remain admissible. Theoretical weighted bounds
    remain available without weighted errors; the corresponding observed
    interval is then None and its sign is ``not_requested``, never zero-error.

    With k=M*K, g=gamma_upper and F=integral_0^h(2X+2gT), the arithmetic
    remainder is (nu_port_upper*w_upper+2*Lambda)*F/(4*nD). The two-Lambda
    term retains both matched branches. The inverse-capacity mean remainder
    is w_upper*F/(4*R_D_lower). Each observed contrast adds only its own four
    reading errors, allowing arbitrary correlations. Initial slopes and
    whole-window rate fields have distinct meanings. Requested old-chart
    collars use the same strict margin>2*T(h) check, independently of signs.
    No graph, event, source report or cached verdict is accepted here.
    """
    capacities = _prepared_coordinate_vector(capacity_bounds, "capacity_bounds", 2)
    port = _prepared_coordinate_vector(
        donor_port_capacity_bounds, "donor_port_capacity_bounds", 2
    )
    contrast = (
        None
        if donor_internal_capacity_contrast_upper_bound is None
        else exact_or_represented_real(
            donor_internal_capacity_contrast_upper_bound,
            "donor_internal_capacity_contrast_upper_bound",
        )
    )
    mass = (
        None
        if inverse_capacity_mass_bounds is None
        else _prepared_coordinate_vector(
            inverse_capacity_mass_bounds, "inverse_capacity_mass_bounds", 2
        )
    )
    weighted_errors = (
        None
        if inverse_capacity_reading_error_bounds is None
        else _prepared_coordinate_vector(
            inverse_capacity_reading_error_bounds,
            "inverse_capacity_reading_error_bounds",
            4,
        )
    )
    a0, d0, weight, gamma, initial, degree, time, n, errors, margins, effective_a0 = (
        _admit_bridge_exchange_inputs(
            initial_form_difference_bounds=initial_form_difference_bounds,
            initial_phase_gap_bounds=initial_phase_gap_bounds,
            bridge_weight_bounds=bridge_weight_bounds,
            gamma_bounds=gamma_bounds,
            initial_form_upper_bound=initial_form_upper_bound,
            weighted_degree_upper_bound=weighted_degree_upper_bound,
            elapsed_time=elapsed_time,
            donor_node_count=donor_node_count,
            reading_error_bounds=reading_error_bounds,
            old_edge_chart_margins=old_edge_chart_margins,
            allow_zero_weight=matched_response,
        )
    )
    for bounds, label in (
        (capacities, "capacity_bounds"),
        (port, "donor_port_capacity_bounds"),
        (mass, "inverse_capacity_mass_bounds"),
    ):
        if bounds is not None and (bounds[0] <= 0 or bounds[0] > bounds[1]):
            raise ValueError(f"{label} requires ordered strictly positive endpoints")
    if not capacities[0] <= port[0] <= port[1] <= capacities[1]:
        raise ValueError("donor_port_capacity_bounds must lie within capacity_bounds")
    if contrast is not None and contrast < 0:
        raise ValueError(
            "donor_internal_capacity_contrast_upper_bound must be nonnegative"
        )
    if weighted_errors is not None and min(weighted_errors) < 0:
        raise ValueError("inverse_capacity_reading_error_bounds must be nonnegative")
    physical_mass = (Q(n) / capacities[1], Q(n) / capacities[0])
    effective_mass = (
        physical_mass
        if mass is None
        else (max(mass[0], physical_mass[0]), min(mass[1], physical_mass[1]))
    )
    if effective_mass[0] > effective_mass[1]:
        raise ValueError(
            "inverse capacity mass and global capacity bounds are inconsistent"
        )
    fallback = (capacities[1] - capacities[0]) * n * degree / 2
    effective_contrast = fallback if contrast is None else min(contrast, fallback)
    mobility_degree = capacities[1] * degree
    g = gamma[1]
    (
        form_ceiling,
        form_rate,
        phase_rate,
        form_displacement,
        phase_displacement,
        variation_integral,
    ) = _prepared_bridge_motion(
        initial=initial, mobility_degree=mobility_degree, gamma_upper=g, time=time
    )
    remainder = (
        (port[1] * weight[1] + 2 * effective_contrast) * variation_integral / (4 * n)
    )
    weighted_remainder = weight[1] * variation_integral / (4 * effective_mass[0])
    internal_rate = (
        effective_contrast * (form_displacement + g * phase_displacement) / n
    )
    error = sum(errors, Q(0))
    weighted_error = None if weighted_errors is None else sum(weighted_errors, Q(0))
    residual_margins = (
        None
        if margins is None
        else tuple(value - 2 * phase_displacement for value in margins)
    )
    retention = (
        "not_requested"
        if residual_margins is None
        else "certified" if min(residual_margins) > 0 else "indeterminate"
    )

    if matched_response:
        # Admit every raw and derived domain before the first numerical bound.
        (
            growth_rate,
            growth_exponent,
            growth,
            forcing_constant,
            forcing_linear,
        ) = _prepared_bridge_parameter_growth(
            maximum_capacity=capacities[1],
            initial=initial,
            mobility_degree=mobility_degree,
            gamma_upper=g,
            time=time,
        )
        difference = (
            weight[1]
            * growth
            * (forcing_constant * time + forcing_linear * time**2 / 2)
        )
        integrated_difference = (
            weight[1]
            * growth
            * (forcing_constant * time**2 / 2 + forcing_linear * time**3 / 6)
        )
        internal_coefficient = effective_contrast * max(Q(1), g) / (2 * n)
        matched_internal_rate = internal_coefficient * difference
        matched_internal_remainder = internal_coefficient * integrated_difference
        matched_remainder = (
            port[1] * weight[1] * variation_integral / (4 * n)
            + matched_internal_remainder
        )

    # Every primitive and its domain is admitted before interval/trigonometry.
    w_box, gamma_box, port_box = I(*weight), I(*gamma), I(*port)
    initial_sine = sin(I(*d0))
    initial_current = w_box * (I(*effective_a0) - gamma_box * initial_sine)
    initial_rate = port_box * initial_current / (4 * n)
    weighted_initial_rate = initial_current * I(
        1 / (4 * effective_mass[1]), 1 / (4 * effective_mass[0])
    )
    form_box = I(
        effective_a0[0] - 2 * form_displacement,
        effective_a0[1] + 2 * form_displacement,
    )
    phase_box = I(d0[0] - 2 * phase_displacement, d0[1] + 2 * phase_displacement)
    sine = sin(phase_box)
    weighted_total_form_rate = w_box * (form_box - gamma_box * sine) / 4
    weighted_total_phase_rate = -gamma_box * w_box * form_box / 4
    inverse_mass = I(1 / effective_mass[1], 1 / effective_mass[0])
    mean_rate_contrast = port_box * weighted_total_form_rate / n + I(
        -internal_rate, internal_rate
    )
    increment = time * initial_rate + I(-remainder, remainder)
    observed = increment + I(-error, error)
    sign = (
        "negative"
        if observed.hi < 0
        else "positive" if observed.lo > 0 else "indeterminate"
    )
    weighted_increment = time * weighted_initial_rate + I(
        -weighted_remainder, weighted_remainder
    )
    weighted_observed = (
        None
        if weighted_error is None
        else weighted_increment + I(-weighted_error, weighted_error)
    )
    weighted_sign = (
        "not_requested"
        if weighted_observed is None
        else (
            "negative"
            if weighted_observed.hi < 0
            else "positive" if weighted_observed.lo > 0 else "indeterminate"
        )
    )
    report = _PreparedCapacityBridgeExchange(
        initial_form_difference_bounds=a0,
        initial_phase_gap_bounds=d0,
        bridge_weight_bounds=weight,
        gamma_bounds=gamma,
        initial_form_upper_bound=initial,
        weighted_degree_upper_bound=degree,
        elapsed_time=time,
        donor_node_count=n,
        reading_error_bounds=errors,
        old_edge_chart_margins=margins,
        capacity_bounds=capacities,
        donor_port_capacity_bounds=port,
        donor_internal_capacity_contrast_upper_bound=contrast,
        fallback_donor_internal_capacity_contrast_upper_bound=fallback,
        effective_donor_internal_capacity_contrast_upper_bound=effective_contrast,
        inverse_capacity_mass_bounds=mass,
        effective_inverse_capacity_mass_bounds=effective_mass,
        inverse_capacity_reading_error_bounds=weighted_errors,
        effective_initial_form_difference_bounds=effective_a0,
        mobility_degree_upper_bound=mobility_degree,
        form_upper_bound=form_ceiling,
        form_rate_upper_bound=form_rate,
        phase_rate_upper_bound=phase_rate,
        form_displacement_upper_bound=form_displacement,
        phase_displacement_upper_bound=phase_displacement,
        rate_variation_integral_upper_bound=variation_integral,
        form_difference_bounds=form_box,
        phase_gap_bounds=phase_box,
        initial_sine_bounds=initial_sine,
        sine_bounds=sine,
        initial_donor_mean_form_rate_bounds=initial_rate,
        internal_mean_rate_contrast_upper_bound=internal_rate,
        donor_mean_form_rate_contrast_bounds=mean_rate_contrast,
        increment_remainder_upper_bound=remainder,
        true_increment_bounds=increment,
        reading_error_upper_bound=error,
        observed_increment_bounds=observed,
        observed_sign=sign,
        inverse_capacity_initial_donor_mean_form_rate_bounds=weighted_initial_rate,
        inverse_capacity_donor_total_form_rate_bounds=weighted_total_form_rate,
        inverse_capacity_host_total_form_rate_bounds=-weighted_total_form_rate,
        inverse_capacity_donor_mean_form_rate_bounds=weighted_total_form_rate
        * inverse_mass,
        inverse_capacity_donor_total_phase_rate_bounds=weighted_total_phase_rate,
        inverse_capacity_host_total_phase_rate_bounds=-weighted_total_phase_rate,
        inverse_capacity_donor_mean_phase_rate_bounds=weighted_total_phase_rate
        * inverse_mass,
        inverse_capacity_increment_remainder_upper_bound=weighted_remainder,
        inverse_capacity_true_increment_bounds=weighted_increment,
        inverse_capacity_reading_error_upper_bound=weighted_error,
        inverse_capacity_observed_increment_bounds=weighted_observed,
        inverse_capacity_observed_sign=weighted_sign,
        retention_margins=residual_margins,
        retention_status=retention,
    )
    if not matched_response:
        return report

    def scaled_rate_endpoints(coefficient):
        """Multiply raw conductance after bounded coefficient arithmetic."""
        products = tuple(
            conductance * endpoint
            for conductance in weight
            for endpoint in (coefficient.lo, coefficient.hi)
        )
        return min(products), max(products)

    def exact_sign(endpoints):
        return (
            "negative"
            if endpoints[1] < 0
            else "positive" if endpoints[0] > 0 else "indeterminate"
        )

    # The dyadic grid belongs to coefficient evaluation and display only. In
    # particular, never reconstruct these endpoints from an I(weight) result.
    current_coefficient = I(*effective_a0) - gamma_box * initial_sine
    initial_exact = scaled_rate_endpoints(port_box * current_coefficient / (4 * n))
    weighted_initial_exact = scaled_rate_endpoints(
        current_coefficient
        * I(1 / (4 * effective_mass[1]), 1 / (4 * effective_mass[0]))
    )
    selected_remainder = min(remainder, matched_remainder)
    selected_internal_rate = min(internal_rate, matched_internal_rate)
    exact_increment = (
        time * initial_exact[0] - selected_remainder,
        time * initial_exact[1] + selected_remainder,
    )
    exact_observed = (exact_increment[0] - error, exact_increment[1] + error)
    weighted_exact_increment = (
        time * weighted_initial_exact[0] - weighted_remainder,
        time * weighted_initial_exact[1] + weighted_remainder,
    )
    weighted_exact_observed = (
        None
        if weighted_error is None
        else (
            weighted_exact_increment[0] - weighted_error,
            weighted_exact_increment[1] + weighted_error,
        )
    )
    values = vars(report).copy()
    values.update(
        initial_donor_mean_form_rate_bounds=I(*initial_exact),
        internal_mean_rate_contrast_upper_bound=selected_internal_rate,
        donor_mean_form_rate_contrast_bounds=port_box * weighted_total_form_rate / n
        + I(-selected_internal_rate, selected_internal_rate),
        increment_remainder_upper_bound=selected_remainder,
        true_increment_bounds=I(*exact_increment),
        observed_increment_bounds=I(*exact_observed),
        observed_sign=exact_sign(exact_observed),
        inverse_capacity_initial_donor_mean_form_rate_bounds=I(*weighted_initial_exact),
        inverse_capacity_true_increment_bounds=I(*weighted_exact_increment),
        inverse_capacity_observed_increment_bounds=(
            None if weighted_exact_observed is None else I(*weighted_exact_observed)
        ),
        inverse_capacity_observed_sign=(
            "not_requested"
            if weighted_exact_observed is None
            else exact_sign(weighted_exact_observed)
        ),
    )
    # The subclass declares its own nonnegative-weight scope.
    del values["scope"]
    return _PreparedMatchedCapacityBridgeExchange(
        **values,
        difference_growth_rate_upper_bound=growth_rate,
        difference_growth_exponent=growth_exponent,
        difference_growth_upper_bound=growth,
        difference_forcing_constant_upper_bound=forcing_constant,
        difference_forcing_linear_upper_bound=forcing_linear,
        matched_state_difference_upper_bound=difference,
        integrated_matched_state_difference_upper_bound=integrated_difference,
        matched_internal_mean_rate_upper_bound=matched_internal_rate,
        matched_internal_increment_remainder_upper_bound=matched_internal_remainder,
        matched_increment_remainder_upper_bound=matched_remainder,
        independent_increment_remainder_upper_bound=remainder,
        exact_true_increment_bounds=exact_increment,
        exact_observed_increment_bounds=exact_observed,
        inverse_capacity_exact_true_increment_bounds=weighted_exact_increment,
        inverse_capacity_exact_observed_increment_bounds=weighted_exact_observed,
    )


@dataclass(frozen=True)
class _PreparedTwoContactComposition:
    """Magnitude certificate for four branches with the same complete state.

    The defect is m12-m1-m2+m0 at one common final time. Both fixed normalized
    constituent form readings have norm at most one, so they share a true
    bound, not necessarily the same value. Each has its own four-error budget.
    Exact endpoints precede the dyadic display; a symmetric envelope supplies
    no strict sign. Scalar data do not authenticate support or preparation.
    ``exact_mixed_zero=True`` certifies zero under the premises; False is
    inconclusive, since the universal bound can overestimate an exact zero.
    """

    first_contact_weight_bounds: tuple[Q, Q]
    second_contact_weight_bounds: tuple[Q, Q]
    gamma_bounds: tuple[Q, Q]
    capacity_bounds: tuple[Q, Q]
    initial_form_upper_bound: Q
    weighted_degree_upper_bound: Q
    elapsed_time: Q
    reading_error_bounds: tuple[Q, Q, Q, Q]
    inverse_capacity_reading_error_bounds: tuple[Q, Q, Q, Q] | None
    old_edge_chart_margins: tuple[Q, ...] | None
    mobility_degree_upper_bound: Q
    form_upper_bound: Q
    form_displacement_upper_bound: Q
    phase_displacement_upper_bound: Q
    difference_growth_rate_upper_bound: Q
    difference_growth_exponent: Q
    difference_growth_upper_bound: Q
    difference_forcing_constant_upper_bound: Q
    difference_forcing_linear_upper_bound: Q
    parameter_sensitivity_upper_bound: Q
    integrated_parameter_sensitivity_upper_bound: Q
    integrated_squared_parameter_sensitivity_upper_bound: Q
    contact_derivative_mixed_coefficient_upper_bound: Q
    sine_curvature_mixed_coefficient_upper_bound: Q
    mixed_sensitivity_upper_bound: Q
    contact_weight_product_upper_bound: Q
    mixed_state_defect_upper_bound: Q
    exact_true_defect_bounds: tuple[Q, Q]
    true_defect_bounds: I
    reading_error_upper_bound: Q
    exact_observed_defect_bounds: tuple[Q, Q]
    observed_defect_bounds: I
    inverse_capacity_exact_true_defect_bounds: tuple[Q, Q]
    inverse_capacity_true_defect_bounds: I
    inverse_capacity_reading_error_upper_bound: Q | None
    inverse_capacity_exact_observed_defect_bounds: tuple[Q, Q] | None
    inverse_capacity_observed_defect_bounds: I | None
    exact_mixed_zero: bool
    retention_margins: tuple[Q, ...] | None
    retention_status: str
    interval_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "four_fixed_support_branches_share_the_same_complete_initial_state_and_clock",
        "two_fixed_undirected_contact_terms_have_held_nonnegative_conductances",
        "same_symmetric_nonnegative_weights_in_both_uniform_quarter_sine_rows",
        "strictly_positive_held_capacities_and_fixed_nonnegative_gamma",
        "no_subsequent_input_event_contact_motion_or_conductance_change",
        "joined_degree_bound_covers_every_intermediate_contact_parameter",
        "global_capacity_bounds_apply_to_every_node_in_every_branch",
        "full_mixed_state_retains_contact_derivatives_and_sine_curvature",
        "arithmetic_and_inverse_capacity_means_use_fixed_nonempty_constituents",
        "normalized_positive_weight_readings_share_a_bound_not_an_actual_value",
        "four_final_readings_have_separate_errors_without_assumed_cancellation",
        "exact_conductance_product_precedes_dyadic_display_rounding",
        "symmetric_magnitude_bound_is_not_a_signed_response_or_mean_closure",
        "scalar_bounds_do_not_authenticate_contact_endpoints_support_or_preparation",
        "old_chart_retention_requires_independent_edge_cycle_and_lift_association",
        "changing_generators_can_have_mixed_response_even_without_sine_curvature",
        "no_native_bare_phase_transfer_formation_occurrence_or_physical_claim",
    )


def _prepared_two_contact_composition(
    *,
    first_contact_weight_bounds,
    second_contact_weight_bounds,
    gamma_bounds,
    capacity_bounds,
    initial_form_upper_bound,
    weighted_degree_upper_bound,
    elapsed_time,
    reading_error_bounds,
    inverse_capacity_reading_error_bounds=None,
    old_edge_chart_margins=None,
) -> _PreparedTwoContactComposition:
    """Bound joint response minus the two separate matched contact responses.

    Callers establish a finite fixed node set, symmetric nonnegative old
    conductances, and two fixed undirected contact contributions. The four
    branches add neither, each alone, or both contacts to that same support.
    Every branch and intermediate parameter value starts from the same full
    form and lifted-phase state and uses the same held positive capacities,
    gamma>=0 and structural clock. The complete law is
    x'=D*(-L*x+gamma*S(theta))/4, theta'=gamma*D*L*x/4, with identical
    weighted support in both rows and no later input or event.

    B0 bounds every initial absolute form; K bounds every joined weighted
    degree at the upper corner, hence throughout the parameter rectangle.
    Scalar admission requires K>=max(w1_upper,w2_upper), but does not prove
    the actual summed degree at a shared port. Distinct ports need not have
    a degree as large as the sum. Endpoint and old-path associations remain
    caller premises; the bound covers all layouts satisfying those premises.

    For k=M*K and g=gamma_upper, the shared growth owner gives G, b0, b1.
    Write P(s)=b0*s+b1*s**2/2, A1=integral(P), A2=integral(P**2).
    Each full-state parameter sensitivity is bounded by G*P(s). Its mixed
    derivative is bounded by C=M*(1+g)*G**2*A1+g*k*G**3*A2. The first term
    retains both contact-field derivatives; the second retains sine
    curvature. Double parameter integration bounds the complete four-branch
    defect by w1_upper*w2_upper*C. A nonzero defect at gamma=0 is compatible
    with linear state evolution because the generators differ by branch.

    A fixed nonempty arithmetic constituent form mean and its independently
    normalized inverse-capacity mean each have observation norm one. Their
    true contrasts therefore share the symmetric bound. No unused count,
    inverse mass or port state is required, and no actual equality of the
    readings is inferred. Four final-time errors have coefficients(+,-,-,+)
    for (m12,m1,m2,m0); each observation adds its own sum of radii. Reused
    baselines need their own association and do not imply error cancellation.
    Omitted weighted errors leave that observed interval unavailable.

    All raw domains precede exponential and interval work, including the
    shared alpha*h<=4096 numerical work limit even at zero conductance.
    Exact rational conductance products multiply C last. Exact endpoints
    retain subgrid scaling and zero whenever either weight is zero; dyadic
    displays are separate. Old-edge chart collars retain the absolute
    margin>2*T(h) test on every branch, independently of composition accuracy.
    """
    first, second, gamma, capacities = (
        _prepared_coordinate_vector(raw, name, 2)
        for raw, name in (
            (first_contact_weight_bounds, "first_contact_weight_bounds"),
            (second_contact_weight_bounds, "second_contact_weight_bounds"),
            (gamma_bounds, "gamma_bounds"),
            (capacity_bounds, "capacity_bounds"),
        )
    )
    initial, degree, time = (
        exact_or_represented_real(raw, name)
        for raw, name in (
            (initial_form_upper_bound, "initial_form_upper_bound"),
            (weighted_degree_upper_bound, "weighted_degree_upper_bound"),
            (elapsed_time, "elapsed_time"),
        )
    )
    errors = _prepared_coordinate_vector(
        reading_error_bounds, "reading_error_bounds", 4
    )
    weighted_errors = (
        None
        if inverse_capacity_reading_error_bounds is None
        else _prepared_coordinate_vector(
            inverse_capacity_reading_error_bounds,
            "inverse_capacity_reading_error_bounds",
            4,
        )
    )
    margins = _prepared_old_edge_chart_margins(old_edge_chart_margins)
    for bounds, name in (
        (first, "first_contact_weight_bounds"),
        (second, "second_contact_weight_bounds"),
        (gamma, "gamma_bounds"),
    ):
        if bounds[0] < 0 or bounds[0] > bounds[1]:
            raise ValueError(f"{name} requires ordered nonnegative endpoints")
    if capacities[0] <= 0 or capacities[0] > capacities[1]:
        raise ValueError("capacity_bounds requires ordered strictly positive endpoints")
    if min(initial, degree, time, *errors) < 0:
        raise ValueError("background, horizon and reading errors must be nonnegative")
    if degree < max(first[1], second[1]):
        raise ValueError(
            "weighted_degree_upper_bound must cover both contact upper bounds"
        )
    if weighted_errors is not None and min(weighted_errors) < 0:
        raise ValueError("inverse_capacity_reading_error_bounds must be nonnegative")
    if margins is not None and min(margins) < 0:
        raise ValueError("old_edge_chart_margins must be nonnegative")

    maximum, g = capacities[1], gamma[1]
    mobility_degree = maximum * degree
    (
        form_ceiling,
        _,
        _,
        form_displacement,
        phase_displacement,
        _,
    ) = _prepared_bridge_motion(
        initial=initial, mobility_degree=mobility_degree, gamma_upper=g, time=time
    )
    (
        growth_rate,
        growth_exponent,
        growth,
        b0,
        b1,
    ) = _prepared_bridge_parameter_growth(
        maximum_capacity=maximum,
        initial=initial,
        mobility_degree=mobility_degree,
        gamma_upper=g,
        time=time,
    )
    integral = b0 * time**2 / 2 + b1 * time**3 / 6
    squared_integral = (
        b0**2 * time**3 / 3 + b0 * b1 * time**4 / 4 + b1**2 * time**5 / 20
    )
    contact_coefficient = maximum * (1 + g) * growth**2 * integral
    curvature_coefficient = g * mobility_degree * growth**3 * squared_integral
    mixed_coefficient = contact_coefficient + curvature_coefficient
    weight_product = first[1] * second[1]
    defect = weight_product * mixed_coefficient
    error = sum(errors, Q(0))
    weighted_error = None if weighted_errors is None else sum(weighted_errors, Q(0))
    exact_true = (-defect, defect)
    exact_observed = (-defect - error, defect + error)
    weighted_exact_observed = (
        None
        if weighted_error is None
        else (-defect - weighted_error, defect + weighted_error)
    )
    residual_margins = (
        None
        if margins is None
        else tuple(value - 2 * phase_displacement for value in margins)
    )
    retention = (
        "not_requested"
        if residual_margins is None
        else "certified" if min(residual_margins) > 0 else "indeterminate"
    )
    # Exact coefficient arithmetic and conductance scaling precede display.
    true_display = I(*exact_true)
    return _PreparedTwoContactComposition(
        first_contact_weight_bounds=first,
        second_contact_weight_bounds=second,
        gamma_bounds=gamma,
        capacity_bounds=capacities,
        initial_form_upper_bound=initial,
        weighted_degree_upper_bound=degree,
        elapsed_time=time,
        reading_error_bounds=errors,
        inverse_capacity_reading_error_bounds=weighted_errors,
        old_edge_chart_margins=margins,
        mobility_degree_upper_bound=mobility_degree,
        form_upper_bound=form_ceiling,
        form_displacement_upper_bound=form_displacement,
        phase_displacement_upper_bound=phase_displacement,
        difference_growth_rate_upper_bound=growth_rate,
        difference_growth_exponent=growth_exponent,
        difference_growth_upper_bound=growth,
        difference_forcing_constant_upper_bound=b0,
        difference_forcing_linear_upper_bound=b1,
        parameter_sensitivity_upper_bound=growth * (b0 * time + b1 * time**2 / 2),
        integrated_parameter_sensitivity_upper_bound=growth * integral,
        integrated_squared_parameter_sensitivity_upper_bound=growth**2
        * squared_integral,
        contact_derivative_mixed_coefficient_upper_bound=contact_coefficient,
        sine_curvature_mixed_coefficient_upper_bound=curvature_coefficient,
        mixed_sensitivity_upper_bound=mixed_coefficient,
        contact_weight_product_upper_bound=weight_product,
        mixed_state_defect_upper_bound=defect,
        exact_true_defect_bounds=exact_true,
        true_defect_bounds=true_display,
        reading_error_upper_bound=error,
        exact_observed_defect_bounds=exact_observed,
        observed_defect_bounds=I(*exact_observed),
        inverse_capacity_exact_true_defect_bounds=exact_true,
        inverse_capacity_true_defect_bounds=true_display,
        inverse_capacity_reading_error_upper_bound=weighted_error,
        inverse_capacity_exact_observed_defect_bounds=weighted_exact_observed,
        inverse_capacity_observed_defect_bounds=(
            None if weighted_exact_observed is None else I(*weighted_exact_observed)
        ),
        exact_mixed_zero=defect == 0,
        retention_margins=residual_margins,
        retention_status=retention,
    )


def _prepared_transit_radii(
    *,
    time,
    decay_upper,
    gap_lower,
    initial_norm_upper,
    feedback_upper,
    forcing_upper,
):
    """Bound scaled form and the integrated phase remainder of a prepared flow.

    Callers admit the complete positive-loss law, its weighted mean-free
    semigroup gap, nonnegative time and bounds before using this kernel.
    Arguments are exact rational bounds. The phase radius is measured from
    the actual initial scaled form plus initial phase; a nominal-source
    comparison must retain its initial form and phase errors separately.
    These are norm bounds for the actual coupled flow, not endpoint boxes
    or a capture verdict. No source, law or horizon is cached here.
    """
    scaled_form_radius = (
        decay_upper * initial_norm_upper
        + feedback_upper * forcing_upper * min(time, Q(1) / gap_lower)
    )
    phase_radius = scaled_form_radius + feedback_upper * time * forcing_upper
    return scaled_form_radius, phase_radius


def _prepared_velocity_defect(
    *,
    time,
    gap_lower,
    feedback_upper,
    lipschitz_upper,
    forcing_upper,
    initial_defect_upper,
):
    """Bound slow velocity error after an admitted fast relaxation time.

    Callers supply exact rational nonnegative bounds for the complete rows
    a'=-A*a+epsilon*F(theta), theta'=A*a on a preserved mean-zero leaf.
    The generator gap and global derivative bound must satisfy g>epsilon*K.
    The resulting decreasing envelope bounds theta_s-F(theta), s=epsilon*t,
    for every later time. It is not a phase trajectory or capture bound.
    """
    if (
        min(time, feedback_upper, lipschitz_upper, forcing_upper, initial_defect_upper)
        < 0
    ):
        raise ValueError("velocity defect requires nonnegative bounds and time")
    damping = gap_lower - feedback_upper * lipschitz_upper
    if damping <= 0:
        raise ValueError("velocity defect requires gap greater than feedback stiffness")
    decay = I(*_negative_exp_bounds(damping * time))
    floor = feedback_upper * lipschitz_upper * forcing_upper / damping
    return damping, decay, floor, decay.hi * initial_defect_upper + floor


def _prepared_duhamel_bounds(
    *,
    time,
    decay_upper,
    gap_lower,
    rate_upper,
    forcing_upper,
    initial_norm_upper,
    scaled_form_error_upper,
    phase_error_upper,
    feedback_upper,
):
    """Bound a prepared scaled flow around its nominal Poisson profile.

    Callers admit a self-adjoint mean-free generator with the supplied gap,
    upper rate and global forcing/Lipschitz bounds. All arguments are exact
    nonnegative bounds. The returned values enclose the weighted scaled-form
    remainder and phase distance from the nominal initial scaled form.
    This estimate retains the full transient and forcing history.
    """
    initial_error = scaled_form_error_upper + phase_error_upper
    history = feedback_upper * forcing_upper * (time + 1 / gap_lower)
    remainder = decay_upper * (
        initial_norm_upper
        + scaled_form_error_upper
        + feedback_upper * forcing_upper / gap_lower
    ) + feedback_upper * rate_upper * (
        initial_norm_upper * time * decay_upper + (initial_error + history) / gap_lower
    )
    phase_error = decay_upper * initial_norm_upper + initial_error + history
    return remainder, phase_error


@dataclass(frozen=True)
class _SineDomain:
    geometry: PhaseCycleGeometry
    model: RelationalExchangeModel
    capacity: tuple[Q, ...]
    degrees: tuple[int, ...]
    e: Q
    w: Q
    beta: Q
    weights: tuple[Q, ...]
    mobility: tuple[Q, ...]
    normalized_weights: tuple[Q, ...]
    gap: Q
    pi: I
    alpha: I
    inverse_alpha: I
    eta: I
    forcing: I


def _sine_domain(geometry, reference_model, capacity) -> _SineDomain:
    """Admit held coefficients and weighted geometry without a nodal state.

    The support declares unit sine coupling. Rebuilding its geometry retains
    the shared 32-node/50-edge budget and rejects forged derived fields. Model
    coefficients are their authoritative stored values, without a second
    normalization. No form, phase, common origin or representative is supplied.
    """
    geometry = _rebuild(geometry)
    e, w, beta = _sine_model_coefficients(reference_model, positive_loss=True)
    size = len(geometry.nodes)
    raw_capacity = _ordered(capacity, "capacity", limit=size + 1)
    if len(raw_capacity) != size:
        raise ValueError("capacity must contain one value per support node")
    capacity = tuple(
        exact_or_represented_real(value, f"capacity[{i}]")
        for i, value in enumerate(raw_capacity)
    )
    if any(value <= 0 for value in capacity):
        raise ValueError(
            "analytic sine domain requires strictly positive held capacities"
        )
    adjacency = [set() for _ in geometry.nodes]
    for left, right in geometry.edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    degrees = tuple(map(len, adjacency))
    weights = tuple(Q(degree) / nu for degree, nu in zip(degrees, capacity))
    mobility = tuple(Q(1) / value for value in weights)
    total_weight = sum(weights, Q(0))
    normalized_weights = tuple(weight / total_weight for weight in weights)
    laplacian = tuple(
        tuple(
            Q(len(adjacency[i]) if i == j else -int(j in adjacency[i]))
            for j in range(size)
        )
        for i in range(size)
    )
    gap, mean_preserved, consensus_preserved, uniform_fixed = (
        _exact_flow_gap_from_rationals(laplacian, mobility, weights)
    )
    if gap <= 0 or not all((mean_preserved, consensus_preserved, uniform_fixed)):
        raise ArithmeticError(
            "the complete weighted diffusion quotient failed admission"
        )
    pi = pi_interval()
    alpha = (w / (beta * e)) / pi
    inverse_alpha = (beta * e / w) * pi
    eta = (w**2 / (beta * e**2)) / pi**2
    forcing = sqrt(I(sum((nu * degree for nu, degree in zip(capacity, degrees)), Q(0))))
    return _SineDomain(
        geometry=geometry,
        model=reference_model,
        capacity=capacity,
        degrees=degrees,
        e=e,
        w=w,
        beta=beta,
        weights=weights,
        mobility=mobility,
        normalized_weights=normalized_weights,
        gap=gap,
        pi=pi,
        alpha=alpha,
        inverse_alpha=inverse_alpha,
        eta=eta,
        forcing=forcing,
    )


@dataclass(frozen=True)
class _SinePreparation:
    """Shared prepared rows; report consumers retain their admitted source.

    ``admitted`` is None only for an internal producer using freshly admitted
    exact rows and domain directly. It is never absent on the report-consuming
    ``_sine_preparation`` path, and it supplies no arithmetic premises.
    """

    admitted: SineExchangeComparison | SineRelativePattern | None
    geometry: PhaseCycleGeometry
    uncertain: bool
    form: tuple[Q, ...]
    phase: tuple[Q, ...]
    form_errors: tuple[Q, ...]
    phase_errors: tuple[Q, ...]
    model: RelationalExchangeModel
    e: Q
    w: Q
    beta: Q
    weights: tuple[Q, ...]
    mobility: tuple[Q, ...]
    normalized_weights: tuple[Q, ...]
    mean: Q
    phase_mean: Q
    centered: tuple[Q, ...]
    centered_form_errors: tuple[Q, ...]
    centered_phase_errors: tuple[Q, ...]
    gap: Q
    pi: I
    alpha: I
    inverse_alpha: I
    eta: I
    nominal_initial_norm: I
    initial_error_norm: Q
    initial_norm: I
    forcing: I
    scaled_nominal: tuple[I, ...]
    scaled_initial: tuple[I, ...]
    initial_form_storage: Q | None
    initial_form_storage_bounds: I
    initial_phase_storage_bounds: I
    initial_storage_bounds: I
    initial_phase_gaps: tuple[I, ...]


def _sine_preparation(source) -> _SinePreparation:
    """Admit a complete exact source or its original relative residual set.

    Exact and nominal phase lifts may be nonuniform. Each consuming theorem
    owns any stronger phase premise. Means here are nominal for relative sources;
    they do not recover their unknown common origins or identify one mean leaf
    for the entire residual family. The shared 32-node/50-edge geometry budget
    and strictly positive held capacities, form loss and exchange apply.
    """
    if not isinstance(source, (SineExchangeComparison, SineRelativePattern)):
        raise TypeError("a SineExchangeComparison or SineRelativePattern is required")
    admitted, geometry = _sector_source_admission(source)
    uncertain = isinstance(admitted, SineRelativePattern)
    if uncertain:
        form, phase = admitted.nominal_form, admitted.nominal_phase
        form_errors, phase_errors = (
            admitted.form_error_bounds,
            admitted.phase_error_bounds,
        )
    else:
        form, phase = admitted.epi, admitted.phase
        form_errors = phase_errors = (Q(0),) * len(admitted.nodes)
    domain = _sine_domain(geometry, admitted.reference_model, admitted.capacity)
    return _sine_preparation_from_rows(
        domain,
        form=form,
        phase=phase,
        form_errors=form_errors,
        phase_errors=phase_errors,
        admitted=admitted,
        uncertain=uncertain,
    )


def _sine_preparation_from_rows(
    domain: _SineDomain,
    *,
    form: tuple[Q, ...],
    phase: tuple[Q, ...],
    form_errors: tuple[Q, ...],
    phase_errors: tuple[Q, ...],
    admitted: SineExchangeComparison | SineRelativePattern | None = None,
    uncertain: bool = True,
) -> _SinePreparation:
    """Compute preparation data from one freshly admitted domain and rows.

    Private callers own prior scalar, row-size and error-sign admission and
    must pass normalized exact rows in the domain's node order. ``domain``
    comes from a fresh ``_sine_domain`` call; a cached domain or incoming
    report cannot substitute for its model/support/capacity admission.

    The report adapter above preserves its normalized source association and
    exact-versus-relative metadata. A fixed internal producer may omit that
    association after directly admitting its primitive rows, sharing one
    domain within an invocation without building discarded observation fields.
    The association and ``uncertain`` metadata do not alter the arithmetic.
    """
    geometry, model = domain.geometry, domain.model
    e, w, beta = domain.e, domain.w, domain.beta
    weights, mobility = domain.weights, domain.mobility
    normalized_weights = domain.normalized_weights
    gap, pi = domain.gap, domain.pi
    alpha, inverse_alpha, eta = domain.alpha, domain.inverse_alpha, domain.eta
    forcing = domain.forcing
    mean = sum(
        (weight * value for weight, value in zip(normalized_weights, form)), Q(0)
    )
    phase_mean = sum(
        (weight * value for weight, value in zip(normalized_weights, phase)), Q(0)
    )
    centered = tuple(value - mean for value in form)

    def centered_errors(errors):
        average = sum(
            (weight * value for weight, value in zip(normalized_weights, errors)), Q(0)
        )
        return tuple(
            radius + average - 2 * weight * radius
            for radius, weight in zip(errors, normalized_weights)
        )

    centered_form_errors = centered_errors(form_errors)
    centered_phase_errors = centered_errors(phase_errors)
    nominal_initial_norm = alpha * sqrt(
        I(sum((weight * value**2 for weight, value in zip(weights, centered)), Q(0)))
    )
    initial_error_norm = (
        alpha
        * sqrt(
            I(
                sum(
                    (
                        weight * radius**2
                        for weight, radius in zip(weights, form_errors)
                    ),
                    Q(0),
                )
            )
        )
    ).hi
    # Weighted centering is an orthogonal projection. Its residual norm is
    # at most the uncentered weighted radius norm; no common offset is added.
    initial_norm = (
        nominal_initial_norm
        if not any(form_errors)
        else I(
            max(Q(0), nominal_initial_norm.lo - initial_error_norm),
            nominal_initial_norm.hi + initial_error_norm,
        )
    )
    scaled_nominal = tuple(alpha * value for value in centered)
    scaled_initial = tuple(
        alpha * I(value - radius, value + radius)
        for value, radius in zip(centered, centered_form_errors)
    )
    initial_form_storage = initial_phase_storage = I(0)
    initial_phase_gaps = []
    for left, right in geometry.edges:
        form_error = form_errors[left] + form_errors[right]
        phase_error = phase_errors[left] + phase_errors[right]
        form_gap = form[right] - form[left]
        phase_gap = phase[right] - phase[left]
        initial_form_storage += I(form_gap - form_error, form_gap + form_error) ** 2 / 2
        phase_interval = I(phase_gap - phase_error, phase_gap + phase_error)
        initial_phase_gaps.append(phase_interval)
        initial_phase_storage += 1 - cos(phase_interval)
    exact_form_storage = sum(
        ((form[right] - form[left]) ** 2 / 2 for left, right in geometry.edges),
        Q(0),
    )
    return _SinePreparation(
        admitted=admitted,
        geometry=geometry,
        uncertain=uncertain,
        form=form,
        phase=phase,
        form_errors=form_errors,
        phase_errors=phase_errors,
        model=model,
        e=e,
        w=w,
        beta=beta,
        weights=weights,
        mobility=mobility,
        normalized_weights=normalized_weights,
        mean=mean,
        phase_mean=phase_mean,
        centered=centered,
        centered_form_errors=centered_form_errors,
        centered_phase_errors=centered_phase_errors,
        gap=gap,
        pi=pi,
        alpha=alpha,
        inverse_alpha=inverse_alpha,
        eta=eta,
        nominal_initial_norm=nominal_initial_norm,
        initial_error_norm=initial_error_norm,
        initial_norm=initial_norm,
        forcing=forcing,
        scaled_nominal=scaled_nominal,
        scaled_initial=scaled_initial,
        initial_form_storage=None if any(form_errors) else exact_form_storage,
        initial_form_storage_bounds=initial_form_storage,
        initial_phase_storage_bounds=initial_phase_storage,
        initial_storage_bounds=initial_form_storage + beta * initial_phase_storage,
        initial_phase_gaps=tuple(initial_phase_gaps),
    )
