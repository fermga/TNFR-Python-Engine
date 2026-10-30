"""Structural nonlinear port memory and response-free approximation bounds.

The fixed three-C9 model retains six central form/phase coordinates and 48
hidden coordinates. Rational spatial partitions and oriented edge factors
specify its causal amplitude hierarchy; no memory kernel or trajectory is
evaluated. Scalar enclosures describe the named exact trigonometric model,
not an independently rationalized generator.
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
from ._sine_class_repeated_interaction import (
    _bound_repeated_interaction,
    _RepeatedInteraction,
)
from .relational_sine_class_cubic_response import (
    _cubic_parameters,
    _CubicParameters,
    _higher_amplitude_remainder,
)
from .relational_sine_class_mediation import _EDGES, _NODES
from .relational_sine_class_memory import _normalized_laplacian

_PORTS = (4, 13, 22)
_GAMMA_UPPER = Q(1, 3000)
_MAX_INPUT_VARIATION = Q(7, 5000)
_MAX_HORIZON = Q(2)


@dataclass(frozen=True)
class _CollectiveInterface:
    """Spatial factors of the original-coordinate six-port Volterra hierarchy.

    Each memory object partitions a rational spatial matrix only. The actual
    two-channel blocks E,B,C,M come from J=[[-A,-gamma*C],[gamma*A,0]], where
    C is the sum of the three internal matrices times their target cosines
    and the contact matrix. A spatial object's kernel_at_zero is not the
    physical two-channel kernel. The latter remains B*exp(M*t)*C.

    Edge phase differences and normalized current columns factor the original
    Q and T maps; scalar coefficients below include gamma. They supply the
    quadratic force Q(z1,z1) and cubic force 2Q(z1,z2)+T(z1,z1,z1).
    """

    mediator_class: int
    parameter_bounds: _CubicParameters
    spatial_partition: LinearCoordinateMemory
    phase_component_partitions: tuple[LinearCoordinateMemory, ...]
    port_indices: tuple[int, ...]
    hidden_indices: tuple[int, ...]
    reflection_indices: tuple[int, ...]
    edge_phase_difference_rows: tuple[tuple[Q, ...], ...]
    normalized_edge_current_columns: tuple[tuple[Q, ...], ...]
    quadratic_edge_scalar_bounds: tuple[I, ...]
    cubic_edge_scalar_bounds: tuple[I, ...]
    port_laplacian_observer: tuple[tuple[Q, ...], ...]
    port_jump_columns: tuple[tuple[Q, ...], ...]
    port_jump_quadratic_matrix: tuple[tuple[Q, ...], ...]
    hidden_path_node_orders: tuple[tuple[int, ...], ...]
    normalized_grounded_path_matrix: tuple[tuple[Q, ...], ...]
    port_nodes: tuple[int, ...] = _PORTS
    full_dimension: int = 54
    visible_dimension: int = 6
    hidden_dimension: int = 48
    nominal_amplitude_reflection_eigenvalues: tuple[int, ...] = (1, -1, 1)
    phase_component_order: tuple[str, ...] = (
        "donor_internal",
        "mediator_internal",
        "receiver_internal",
        "contacts",
    )
    linear_generator_rule: str = "J=[[-A,-gamma*C],[gamma*A,0]]"
    first_hidden_equation: str = "h1(t)=integral_0^t exp(M*(t-s))*C*v1(s) ds"
    second_hidden_equation: str = (
        "h2(t)=integral_0^t exp(M*(t-s))*Q_H(z1(s),z1(s)) ds; v2=0"
    )
    third_visible_equation: str = (
        "v3'=E*v3+integral_0^t K(t-s)*v3(s) ds+F3_V+"
        "integral_0^t B*exp(M*(t-s))*F3_H(s) ds; "
        "F3=2*Q(z1,(0,h2))+T(z1,z1,z1)"
    )
    hidden_mode_formula: str = (
        "lambda_m=1-cos(m*pi/9), m=1..8; "
        "M_component,m=lambda_m*[[-1,-gamma*c_component],[gamma,0]]"
    )
    initialization_equation: str = (
        "v_init(t)=Pi*exp(J*t)*z0; z0=z(0-) before input events"
    )
    clock: str = "tau=e*t; e=1023/1024"
    scope: tuple[str, ...] = (
        "spatial_partition_is_exact_rational_geometry_not_a_rationalized_physical_J",
        "trigonometric_interval_factors_enclose_the_named_exact_model_coefficients",
        "six_ports_retain_form_and_phase_at_all_three_central_nodes",
        "forty_eight_hidden_coordinates_and_their_initial_source_are_retained_in_memory",
        "nominal_central_quadratic_output_vanishes_but_hidden_quadratic_feedback_remains",
        "reflection_parity_applies_to_nominal_central_inputs_not_arbitrary_actual_sources",
        "exact_linear_initialization_is_retained_with_separate_nonlinear_source_error",
        "signed_hybrid_form_events_at_ports_are_not_replaced_by_duration_forcing",
        "full_boundary_pressure_requires_the_hidden_memory_not_only_current_port_state",
        "no_kernel_evaluation_coefficient_generation_solver_or_state_reset_is_performed",
        "no_instantaneous_nonlinear_Markov_closure_or_new_support_connection_is_certified",
    )


def _derive_collective_interface(mediator_class) -> _CollectiveInterface:
    """Rebuild fixed geometry and target factors without a finite response call."""
    if type(mediator_class) is not int or mediator_class not in (1, 2):
        raise ValueError("mediator class must be ordinary integer one or two")
    parameters = _cubic_parameters(mediator_class)
    degrees = parameters.degrees
    form = _normalized_laplacian(_EDGES, degrees)
    spatial = derive_coordinate_memory(form, _PORTS)
    components = tuple(_EDGES[9 * part : 9 * (part + 1)] for part in range(3)) + (
        _EDGES[27:],
    )
    phase = tuple(
        derive_coordinate_memory(_normalized_laplacian(edges, degrees), _PORTS)
        for edges in components
    )
    visible = _PORTS + tuple(node + 27 for node in _PORTS)
    hidden = tuple(i for i in range(54) if i not in visible)
    reflection = tuple(9 * (node // 9) + 8 - node % 9 for node in _NODES)
    differences = tuple(
        tuple(Q(int(node == right) - int(node == left)) for node in _NODES)
        for left, right in _EDGES
    )
    currents = tuple(
        tuple(
            Q(int(node == left) - int(node == right), degrees[node]) for node in _NODES
        )
        for left, right in _EDGES
    )
    observer = tuple(
        tuple(degrees[node] * value for value in form[node]) for node in _PORTS
    )
    paths = tuple(
        tuple(9 * component + local for local in (5, 6, 7, 8, 0, 1, 2, 3))
        for component in range(3)
    )
    path = tuple(tuple(form[i][j] for j in paths[0]) for i in paths[0])
    if any(
        tuple(tuple(form[i][j] for j in nodes) for i in nodes) != path
        for nodes in paths[1:]
    ) or any(
        form[i][j]
        for part in range(3)
        for other in range(part + 1, 3)
        for i in paths[part]
        for j in paths[other]
    ):
        raise ArithmeticError(
            "fixed hidden support must retain three identical grounded paths"
        )
    return _CollectiveInterface(
        mediator_class=mediator_class,
        parameter_bounds=parameters,
        spatial_partition=spatial,
        phase_component_partitions=phase,
        port_indices=visible,
        hidden_indices=hidden,
        reflection_indices=reflection + tuple(node + 27 for node in reflection),
        edge_phase_difference_rows=differences,
        normalized_edge_current_columns=currents,
        quadratic_edge_scalar_bounds=tuple(
            -parameters.gamma * s / 2 for s in parameters.edge_sines
        ),
        cubic_edge_scalar_bounds=tuple(
            -parameters.gamma * c / 6 for c in parameters.edge_cosines
        ),
        port_laplacian_observer=observer,
        port_jump_columns=tuple(
            tuple(Q(int(index == node)) for index in range(54)) for node in _PORTS
        ),
        port_jump_quadratic_matrix=tuple(
            tuple(row[node] for node in _PORTS) for row in observer
        ),
        hidden_path_node_orders=paths,
        normalized_grounded_path_matrix=path,
    )


@dataclass(frozen=True)
class _CollectiveInterfaceBound:
    """Conditional finite-window port fidelity with retained linear source.

    Source radii bound pre-input ``z(0-)`` form/phase residual coordinates about
    declared common origins. The origins are retained exactly. They may be the
    original preparation origins or actual branch means, which need not be
    small. Total input variation includes time-zero impulses at central ports.
    """

    total_input_variation: Q
    horizon: Q
    endpoint_radius: Q
    flow_comparison_margin: Q
    phase_bootstrap_margin: Q
    nominal_fifth_order_error_upper_bound: Q
    nonlinear_initialization_error_upper_bound: Q
    form_error_upper_bound: Q
    phase_error_upper_bound: Q
    port_pressure_error_upper_bounds: tuple[Q, ...]
    gamma_upper_bound: Q = _GAMMA_UPPER
    port_nodes: tuple[int, ...] = _PORTS
    scope: tuple[str, ...] = (
        "nominal_interface_retains_complete_first_and_third_amplitude_memory_terms",
        "fifth_order_bound_applies_to_the_full_reflection_even_projection",
        "exact_linear_source_Pi_exp_J_t_z0_is_retained_not_dropped_or_estimated_as_zero",
        "initialization_error_bounds_only_the_additional_nonlinear_source_coupling",
        "source_radius_bounds_residual_maximum_coordinates_about_declared_common_origins",
        "original_origins_or_actual_branch_means_are_retained_exactly_without_recentering_claims",
        "pressure_error_multiplies_form_fidelity_by_the_actual_port_Laplacian_row_norm",
        "finite_horizon_accuracy_does_not_supply_identity_work_or_return_admission",
        "no_interface_approximation_is_extended_through_the_long_relaxation_dwell",
    )


def _bound_collective_interface(
    *, total_input_variation, horizon, endpoint_radius
) -> _CollectiveInterfaceBound:
    """Bound fixed-model interface error from three independent scalar premises."""
    values = {
        key: exact_or_represented_real(value, key)
        for key, value in dict(
            total_input_variation=total_input_variation,
            horizon=horizon,
            endpoint_radius=endpoint_radius,
        ).items()
    }
    amplitude, duration, eps = (
        values[key] for key in ("total_input_variation", "horizon", "endpoint_radius")
    )
    if not 0 <= amplitude <= _MAX_INPUT_VARIATION:
        raise ValueError("total input variation must lie in [0,7/5000]")
    if not 0 <= duration <= _MAX_HORIZON:
        raise ValueError("interface horizon must lie in [0,2]")
    if eps < 0:
        raise ValueError("endpoint radius must be nonnegative")
    g = _GAMMA_UPPER
    ell, denominator = 1 - 2 * g * duration, 1 - 2 * g**2 * duration**2
    nominal = _higher_amplitude_remainder(amplitude, duration)
    initialization = (
        4
        * g
        * duration
        * eps
        / (ell * denominator)
        * (g * amplitude / denominator + eps / ell)
    )
    form = nominal + initialization
    return _CollectiveInterfaceBound(
        **values,
        flow_comparison_margin=ell,
        phase_bootstrap_margin=denominator,
        nominal_fifth_order_error_upper_bound=nominal,
        nonlinear_initialization_error_upper_bound=initialization,
        form_error_upper_bound=form,
        phase_error_upper_bound=2 * g * duration * form,
        port_pressure_error_upper_bounds=tuple(
            2 * degree * form for degree in (3, 4, 3)
        ),
    )


@dataclass(frozen=True)
class _RepeatedCollectiveInterface:
    """Fixed-word consequence of supplied cubic evidence and causal fidelity.

    The nested repeated certificate supplies work, means and return arithmetic.
    Its nonlinear source allowance bounds the retained linear initialization;
    the additional nonlinear initialization error below is charged separately.
    Neither input coefficient provenance nor actual branch premises are proved.
    """

    base_cubic_bounds: tuple[Q, Q]
    scaled_cubic_bounds: tuple[Q, Q]
    per_history_fidelity: tuple[_CollectiveInterfaceBound, ...]
    nominal_fifth_order_contrast_error_upper_bound: Q
    nonlinear_initialization_contrast_error_upper_bound: Q
    repeated: _RepeatedInteraction
    full_response_via_interface_bounds: tuple[Q, Q]
    recorded_full_response_via_interface_bounds: tuple[Q, Q]
    interface_separation_margin: Q
    interface_separation_sufficient: bool
    conditional_repeated_interface_sufficient: bool
    status: str
    amplitude_scale: Q = Q(7, 5)
    history_order: tuple[str, ...] = ("neither", "first_only", "second_only", "both")
    scope: tuple[str, ...] = (
        "base_cubic_interval_is_a_supplied_complete_coefficient_premise_with_time_error",
        "positive_common_scale_cubes_the_entire_admitted_base_interval",
        "all_eight_initialization_errors_include_the_zero_input_histories",
        "linear_source_and_tangent_memory_allowances_remain_separate_from_nonlinear_defect",
        "mean_compatibility_and_entry_into_the_relative_return_family_remain_required",
        "causal_interface_is_used_only_during_each_two_unit_word_not_during_the_dwell",
        "no_new_coefficient_response_or_source_acquisition_is_evaluated",
    )


def _bound_repeated_collective_interface(
    *, base_cubic_lower, base_cubic_upper
) -> _RepeatedCollectiveInterface:
    """Combine supplied cubic endpoints with the fixed scaled repeated word."""
    lower = exact_or_represented_real(base_cubic_lower, "base_cubic_lower")
    upper = exact_or_represented_real(base_cubic_upper, "base_cubic_upper")
    if lower > upper:
        raise ValueError("base cubic endpoints must be ordered")
    scale, amplitude, eps = Q(7, 5), Q(7, 10000), Q(1, 10**32)
    scaled = scale**3 * lower, scale**3 * upper
    fidelity = tuple(
        _bound_collective_interface(
            total_input_variation=value, horizon=Q(2), endpoint_radius=eps
        )
        for value in (Q(0), amplitude, amplitude, 2 * amplitude)
    )
    fifth = 2 * sum(
        (row.nominal_fifth_order_error_upper_bound for row in fidelity), Q(0)
    )
    initialization = 2 * sum(
        (row.nonlinear_initialization_error_upper_bound for row in fidelity), Q(0)
    )
    repeated = _bound_repeated_interaction(
        nominal_contrast_lower=scaled[0] - fifth,
        nominal_contrast_upper=scaled[1] + fifth,
    )
    true = (
        repeated.true_contrast_bounds[0] - initialization,
        repeated.true_contrast_bounds[1] + initialization,
    )
    recorded = (
        repeated.recorded_contrast_bounds[0] - initialization,
        repeated.recorded_contrast_bounds[1] + initialization,
    )
    margin = repeated.separation_margin - initialization
    separated = margin > 0
    sufficient = repeated.conditional_repeatability_sufficient and separated
    return _RepeatedCollectiveInterface(
        base_cubic_bounds=(lower, upper),
        scaled_cubic_bounds=scaled,
        per_history_fidelity=fidelity,
        nominal_fifth_order_contrast_error_upper_bound=fifth,
        nonlinear_initialization_contrast_error_upper_bound=initialization,
        repeated=repeated,
        full_response_via_interface_bounds=true,
        recorded_full_response_via_interface_bounds=recorded,
        interface_separation_margin=margin,
        interface_separation_sufficient=separated,
        conditional_repeated_interface_sufficient=sufficient,
        status=(
            "conditional_repeated_interface_sufficient" if sufficient else "bounds_only"
        ),
    )
