"""Degree-aware component composition and conditional full-lift error bounds.

Three one-port C9 interfaces are connected by the two existing bridges. The
spatial factors below are exact rationals; gamma and target trigonometric
factors remain enclosures of the original law. No kernel, response or time
coefficient is evaluated. Error budgets concern complete lifted component
states after both boundary rows have been recomposed, not port tolerance alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real
from ..mathematics._rational_interval import I
from ..mathematics.linear_observation import (
    LinearCoordinateMemory,
    derive_coordinate_memory,
)
from ._sine_class_collective_interface import (
    _GAMMA_UPPER,
    _MAX_HORIZON,
    _bound_collective_interface,
    _CollectiveInterface,
    _CollectiveInterfaceBound,
    _derive_collective_interface,
)


@dataclass(frozen=True)
class _ComponentInterface:
    """One internal ring with the final joined mass normalization."""

    node_indices: tuple[int, ...]
    full_state_indices: tuple[int, ...]
    hidden_state_indices: tuple[int, ...]
    port_node: int
    joined_degrees: tuple[int, ...]
    spatial_partition: LinearCoordinateMemory
    internal_visible_generator_bounds: tuple[tuple[I, ...], ...]
    internal_edge_indices: tuple[int, ...]
    port_local_state_indices: tuple[int, int] = (4, 13)
    internal_generator_rule: str = "J_r=[[-A_r,-gamma*c_r*A_r],[gamma*A_r,0]]"
    initialization_rule: str = (
        "B_r*exp(M_r*t)*h_r(0-); all sixteen source coordinates retained; "
        "linearly annihilated modes remain relevant to nonlinear source error"
    )
    memory_rule: str = "K_r(t)=B_r*exp(M_r*t)*C_r"


@dataclass(frozen=True)
class _InterfaceComposition:
    """Structural composition of the existing support, not new attachments.

    Each spatial partition is a rational factor, not the physical two-row
    kernel. In incidence orientation, jx=-N*N.T*x_ports and
    js=-N*sin(N.T*theta_ports). Their central inputs are
    ((jx+gamma*js)/d, -gamma*jx/d). Both rows are indispensable.
    """

    collective: _CollectiveInterface
    components: tuple[_ComponentInterface, ...]
    contact_incidence: tuple[tuple[Q, ...], ...]
    contact_laplacian: tuple[tuple[Q, ...], ...]
    normalized_contact_laplacian: tuple[tuple[Q, ...], ...]
    internal_visible_generator_bounds: tuple[tuple[I, ...], ...]
    contact_visible_generator_bounds: tuple[tuple[I, ...], ...]
    recomposed_visible_generator_bounds: tuple[tuple[I, ...], ...]
    contact_cubic_scalar_bound: I
    contact_port_edges: tuple[tuple[int, int], ...] = ((0, 1), (1, 2))
    central_degrees: tuple[int, ...] = (3, 4, 3)
    local_charge_rate_rule: str = "(d^T*x_r,d^T*theta_r)'=(jx+gamma*js,-gamma*jx)"
    local_storage_supply_rule: str = (
        "((a_p-gamma*b_p)*jx+gamma*a_p*js)/d_p; a=L_r*x, b=grad V_r"
    )
    exact_bridge_rule: str = "jx=-N*N.T*x_ports; js=-N*sin(N.T*theta_ports)"
    contact_quadratic_term: str = (
        "zero at the aligned target; not at arbitrary actual phases"
    )
    scope: tuple[str, ...] = (
        "same_twenty_seven_nodes_and_two_existing_bridges_no_support_event",
        "internal_rings_retain_joined_central_degrees_three_four_three",
        "all_component_form_and_phase_boundary_inputs_follow_the_complete_law",
        "internal_and_bridge_storage_are_each_counted_once",
        "each_component_retains_its_full_hidden_initialization_and_nonlinear_feedback",
        "order_by_order_cubic_composition_includes_bridge_cubic_terms",
        "formal_cubic_bridge_truncation_has_a_nonzero_higher_order_residual",
        "no_isolated_degree_two_kernel_or_port_only_Markov_closure_is_substituted",
        "interval_coefficients_enclose_named_exact_transcendentals_not_rationalized_laws",
        "no_response_time_coefficient_or_kernel_exponential_is_evaluated",
    )


def _derive_interface_composition(mediator_class) -> _InterfaceComposition:
    """Partition existing internal spatial blocks before adjoining bridge rows."""
    collective = _derive_collective_interface(mediator_class)
    parameters = collective.parameter_bounds
    gamma = parameters.gamma
    components = []
    internal = [[I(0) for _ in range(6)] for _ in range(6)]
    for component in range(3):
        nodes = tuple(range(9 * component, 9 * component + 9))
        part = collective.phase_component_partitions[component].generator
        matrix = tuple(tuple(part[i][j] for j in nodes) for i in nodes)
        spatial = derive_coordinate_memory(matrix, (4,))
        degree = parameters.degrees[nodes[4]]
        cosine = parameters.edge_cosines[9 * component]
        local = (
            (I(Q(-2, degree)), -gamma * cosine * Q(2, degree)),
            (gamma * Q(2, degree), I(0)),
        )
        for row in range(2):
            for column in range(2):
                internal[component + 3 * row][component + 3 * column] = local[row][
                    column
                ]
        states = nodes + tuple(i + 27 for i in nodes)
        components.append(
            _ComponentInterface(
                node_indices=nodes,
                full_state_indices=states,
                hidden_state_indices=tuple(
                    i for i in states if i not in collective.port_indices
                ),
                port_node=nodes[4],
                joined_degrees=tuple(parameters.degrees[i] for i in nodes),
                spatial_partition=spatial,
                internal_visible_generator_bounds=local,
                internal_edge_indices=tuple(range(9 * component, 9 * component + 9)),
            )
        )
    incidence = ((Q(1), Q(0)), (Q(-1), Q(1)), (Q(0), Q(-1)))
    laplacian = tuple(
        tuple(sum((a * b for a, b in zip(row, other)), Q(0)) for other in incidence)
        for row in incidence
    )
    normalized = tuple(
        tuple(value / parameters.degrees[collective.port_nodes[i]] for value in row)
        for i, row in enumerate(laplacian)
    )
    bridge = tuple(
        tuple(
            (
                I(-normalized[i][j])
                if i < 3 and j < 3
                else (
                    -gamma * normalized[i][j - 3]
                    if i < 3
                    else gamma * normalized[i - 3][j] if j < 3 else I(0)
                )
            )
            for j in range(6)
        )
        for i in range(6)
    )
    internal = tuple(map(tuple, internal))
    joined = tuple(
        tuple(a + b for a, b in zip(row, other)) for row, other in zip(internal, bridge)
    )
    return _InterfaceComposition(
        collective=collective,
        components=tuple(components),
        contact_incidence=incidence,
        contact_laplacian=laplacian,
        normalized_contact_laplacian=normalized,
        internal_visible_generator_bounds=internal,
        contact_visible_generator_bounds=bridge,
        recomposed_visible_generator_bounds=joined,
        contact_cubic_scalar_bound=-gamma / 6,
    )


@dataclass(frozen=True)
class _InterfaceCompositionError:
    """Conditional error after complete-state reconstruction and feedback.

    Each component supplies an absolutely continuous full eighteen-coordinate
    lift between the same events. The residual bounds hold uniformly for the
    recomposed full law, including any bridge approximation. Initial errors
    and any unmatched jump errors must be admitted separately; this contract
    requires matching jumps and therefore introduces no event defect.
    """

    horizon: Q
    form_initial_error: Q
    phase_initial_error: Q
    form_residual_bound: Q
    phase_residual_bound: Q
    integrated_form_residual_bound: Q
    integrated_phase_residual_bound: Q
    feedback_factor: Q
    feedback_denominator: Q
    form_error_upper_bound: Q
    phase_error_upper_bound: Q
    port_pressure_error_upper_bounds: tuple[Q, ...]
    gamma_upper_bound: Q = _GAMMA_UPPER
    scope: tuple[str, ...] = (
        "conditional_on_full_eighteen_coordinate_component_lifts_not_port_tolerances",
        "residuals_are_supremum_bounds_after_both_boundary_rows_are_recomposed",
        "exact_bridge_currents_add_no_residual_but_approximated_bridge_terms_must_be_charged",
        "identical_hybrid_events_preserve_error_across_jumps",
        "initial_hidden_form_and_phase_errors_are_retained_in_the_supremum_budgets",
        "heat_contraction_uses_the_actual_joined_normalized_Laplacian",
        "no_residual_certificate_is_inferred_from_an_observed_or_cached_passing_flag",
        "no_identity_work_acquisition_or_observation_separation_verdict_is_supplied",
    )


def _bound_interface_composition(
    *,
    horizon,
    form_initial_error,
    phase_initial_error,
    form_residual_bound,
    phase_residual_bound,
) -> _InterfaceCompositionError:
    """Solve the two exact rational heat-feedback inequalities."""
    values = {
        key: exact_or_represented_real(value, key)
        for key, value in dict(
            horizon=horizon,
            form_initial_error=form_initial_error,
            phase_initial_error=phase_initial_error,
            form_residual_bound=form_residual_bound,
            phase_residual_bound=phase_residual_bound,
        ).items()
    }
    if not 0 <= values["horizon"] <= _MAX_HORIZON:
        raise ValueError("composition horizon must lie in [0,2]")
    if any(value < 0 for key, value in values.items() if key != "horizon"):
        raise ValueError("initial error and residual bounds must be nonnegative")
    h = values["horizon"]
    rx, ry = h * values["form_residual_bound"], h * values["phase_residual_bound"]
    ax, ay = values["form_initial_error"] + rx, values["phase_initial_error"] + ry
    alpha = 2 * _GAMMA_UPPER * h
    denominator = 1 - alpha**2
    form, phase = (ax + alpha * ay) / denominator, (ay + alpha * ax) / denominator
    return _InterfaceCompositionError(
        **values,
        integrated_form_residual_bound=rx,
        integrated_phase_residual_bound=ry,
        feedback_factor=alpha,
        feedback_denominator=denominator,
        form_error_upper_bound=form,
        phase_error_upper_bound=phase,
        port_pressure_error_upper_bounds=tuple(2 * d * form for d in (3, 4, 3)),
    )


@dataclass(frozen=True)
class _CubicInterfaceComposition:
    """Explicit residual witness for the recomposed cubic hierarchy.

    This describes z_s+z1+z2+z3 with exact retained linear source z_s. It is
    not a Taylor calculation about a reset actual state. The complete phase
    row is linear, so its residual vanishes; the form remainder includes all
    internal and contact sine terms beyond the declared hierarchy.
    """

    fidelity: _CollectiveInterfaceBound
    first_phase_upper_bound: Q
    second_form_upper_bound: Q
    second_phase_upper_bound: Q
    third_form_upper_bound: Q
    third_phase_upper_bound: Q
    linear_source_phase_upper_bound: Q
    higher_phase_upper_bound: Q
    quadratic_residual_upper_bound: Q
    cubic_residual_upper_bound: Q
    sine_tail_residual_upper_bound: Q
    composition_error: _InterfaceCompositionError
    source_coordinates: str = (
        "pre-input residuals about declared common origins retained exactly"
    )
    scope: tuple[str, ...] = (
        "component_cubic_hierarchies_are_composed_order_by_order_with_bridge_cubic_forcing",
        "second_order_hidden_feedback_is_retained_even_when_central_quadratic_output_vanishes",
        "linear_source_has_arbitrary_admitted_hidden_initialization",
        "source_nonlinearity_and_omitted_bridge_terms_are_in_the_explicit_form_residual",
        "initial_state_and_supplied_form_events_match_exactly",
        "generic_residual_bound_does_not_replace_the_sharper_existing_parity_fidelity_bound",
        "no_selected_response_or_time_coefficient_is_computed",
    )


def _bound_cubic_interface_composition(
    *, total_input_variation, horizon, endpoint_radius
) -> _CubicInterfaceComposition:
    """Build an analytic full-lift defect certificate from admitted primitives."""
    fidelity = _bound_collective_interface(
        total_input_variation=total_input_variation,
        horizon=horizon,
        endpoint_radius=endpoint_radius,
    )
    a, h, eps = (
        fidelity.total_input_variation,
        fidelity.horizon,
        fidelity.endpoint_radius,
    )
    g, ell, d = (
        _GAMMA_UPPER,
        fidelity.flow_comparison_margin,
        fidelity.phase_bootstrap_margin,
    )
    y1 = g * a / d
    x2 = 2 * g * h * y1**2 / d
    y2 = 2 * g * h * x2
    x3 = h * (4 * g * y1 * y2 + Q(4, 3) * g * y1**3) / d
    y3 = 2 * g * h * x3
    ys = eps / ell
    w = y2 + y3 + ys
    quadratic = g * (4 * y1 * (y3 + ys) + 2 * w**2)
    cubic = Q(4, 3) * g * (3 * y1**2 * w + 3 * y1 * w**2 + w**3)
    tail = Q(2, 3) * g * (y1 + w) ** 4
    error = _bound_interface_composition(
        horizon=h,
        form_initial_error=Q(0),
        phase_initial_error=Q(0),
        form_residual_bound=quadratic + cubic + tail,
        phase_residual_bound=Q(0),
    )
    return _CubicInterfaceComposition(
        fidelity=fidelity,
        first_phase_upper_bound=y1,
        second_form_upper_bound=x2,
        second_phase_upper_bound=y2,
        third_form_upper_bound=x3,
        third_phase_upper_bound=y3,
        linear_source_phase_upper_bound=ys,
        higher_phase_upper_bound=w,
        quadratic_residual_upper_bound=quadratic,
        cubic_residual_upper_bound=cubic,
        sine_tail_residual_upper_bound=tail,
        composition_error=error,
    )
