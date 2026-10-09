"""Causal elimination of the full acquired-class mediator and finite errors.

Rational spatial blocks retain all 27 nodes and both nodal rows. The exact
generator is specified by these blocks and the named transcendental scalars;
outward scalar enclosures do not replace that generator by a rational model.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real
from ..mathematics._exact_linear_algebra import (
    exact_matrix_product,
    exact_symmetric_semidefinite,
)
from ..mathematics._rational_interval import INTERVAL_METHOD, I
from ..mathematics.linear_observation import derive_coordinate_memory
from ._sine_port_geometry import _central_port_geometry
from .relational_sine_class_mediation import (
    _CLASSES,
    _CONTACTS,
    _EDGES,
    _NODES,
    _mediation_reference_bounds,
)
from .relational_sine_port_composition import _parameters

__all__ = (
    "SineClassMediatedMemory",
    "SineClassMediatedMemoryBound",
    "derive_sine_class_mediated_memory",
    "bound_sine_class_mediated_memory",
)

Matrix = tuple[tuple[Q, ...], ...]
_VISIBLE = tuple(range(9)) + tuple(range(18, 27))
_HIDDEN = tuple(range(9, 18))


def _add(left, right):
    return tuple(
        tuple(a + b for a, b in zip(row, other)) for row, other in zip(left, right)
    )


def _zero(matrix):
    return all(value == 0 for row in matrix for value in row)


def _row_norm(matrix):
    return max(sum(map(abs, row), Q(0)) for row in matrix)


def _normalized_laplacian(edges, degrees):
    matrix = [[Q(0) for _ in degrees] for _ in degrees]
    for left, right in edges:
        for node, neighbor in ((left, right), (right, left)):
            matrix[node][node] += Q(1, degrees[node])
            matrix[node][neighbor] -= Q(1, degrees[node])
    return tuple(map(tuple, matrix))


@dataclass(frozen=True)
class SineClassMediatedMemory:
    """Exact structural factors of a 36-visible/18-hidden causal law.

    Spatial blocks refer to A=K L with visible nodes (donor,receiver) and
    hidden nodes (mediator). Each channel keeps that spatial order. Write
    P=A_VH, Q=A_HV, N=N_HH, and C_HH=c_k N+W_HH. Then

        E=[[-A_VV,-gamma C_VV],[gamma A_VV,0]],
        B=[[-P,-gamma P],[gamma P,0]],
        C=[[-Q,-gamma Q],[gamma Q,0]],
        D_k=[[-A_HH,-gamma C_HH],[gamma A_HH,0]].

    C_VV=c_1 N_VV+W_VV. Eliminating hidden coordinates gives the kernel
    B exp(D_k t) C AND the source B exp(D_k t) h0. The scalar 2x2 kernel
    factors multiply each entry of the corresponding rational spatial map
    in channel-major order. They enclose coefficients of the exact law,
    not entries of an independently supplied rational surrogate.
    """

    nodes: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]
    degrees: tuple[int, ...]
    visible_nodes: tuple[int, ...]
    hidden_nodes: tuple[int, ...]
    visible_indices: tuple[int, ...]
    hidden_indices: tuple[int, ...]
    normalized_laplacian_vv: Matrix
    normalized_laplacian_vh: Matrix
    normalized_laplacian_hv: Matrix
    normalized_laplacian_hh: Matrix
    outer_phase_interior_vv: Matrix
    visible_phase_contact_vv: Matrix
    mediator_phase_interior_hh: Matrix
    hidden_phase_contact_hh: Matrix
    gamma_bounds: I
    eta_bounds: I
    class_cosine_bounds: tuple[I, I]
    class_cosine_difference_bounds: I
    kernel_at_zero_spatial_matrix: Matrix
    kernel_at_zero_scalar_bounds: tuple[tuple[I, ...], ...]
    kernel_derivative_difference_spatial_matrix: Matrix
    kernel_derivative_difference_scalar_bounds: tuple[tuple[I, ...], ...]
    receiver_donor_kernel_derivative_difference_bounds: I
    hidden_midpoint_map: Matrix
    grounded_hidden_laplacian: Matrix
    grounded_hidden_laplacian_positive: bool
    midpoint_equations_verified: bool
    quasistatic_form_matrix: Matrix
    quasistatic_phase_constant_matrix: Matrix
    quasistatic_phase_cosine_matrix: Matrix
    hidden_to_visible_norm_upper_bound: Q
    visible_to_hidden_norm_upper_bound: Q
    classes: tuple[tuple[int, ...], ...] = _CLASSES
    full_state_dimension: int = 54
    visible_state_dimension: int = 36
    hidden_state_dimension: int = 18
    full_generator_norm_upper_bound: Q = Q(3)
    hidden_generator_norm_upper_bound: Q = Q(3)
    clock: str = "tau=e*t; e=1023/1024"
    phase_coordinates: str = "y=theta-Theta; original unscaled radians"
    memory_equation: str = (
        "v'=E*v+B*exp(D_k*t)*h0+integral_0^t B*exp(D_k*(t-s))*C*v(s) ds"
    )
    nonlinear_residual: str = "r_V(t)+integral_0^t B*exp(D_k*(t-s))*r_H(s) ds"
    quasistatic_equation: str = (
        "v'=(E+B*T)*v; T=diag(hidden_midpoint_map,hidden_midpoint_map)"
    )
    nonlinear_midpoint_branch: str = (
        "x_H=1*(x_Dp+x_Rp)/2; y_H=1*(y_Dp+y_Rp)/2; retain full visible sine rows"
    )
    interval_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "fixed_full_three_C9_two_contact_support_actual_central_degrees_3_4_3",
        "all_donor_receiver_coordinates_visible_all_mediator_coordinates_hidden",
        "exact_rational_spatial_partition_not_a_transcendental_generator_rationalization",
        "kernel_at_zero_is_class_blind_first_kernel_derivative_retains_class",
        "individual_effective_trajectories_keep_full_hidden_initialization",
        "nonlinear_elimination_retains_visible_and_hidden_residual_forcing",
        "grounded_positive_hidden_blocks_give_unique_stationary_midpoint_map",
        "full_nonlinear_midpoint_branch_is_stationary_but_generally_not_invariant",
        "full_hidden_state_retention_does_not_assert_a_minimal_realization",
        "quasistatic_comparator_is_class_blind_not_an_installed_or_universally_valid_law",
        "no_memory_decay_positivity_or_arbitrary_instantaneous_closure_theorem",
        "no_source_acquisition_identity_work_or_physical_identification_verdict",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {
            "schema": "tnfr.sine-class-mediated-memory.v1",
            "report": _project(self),
        }


def derive_sine_class_mediated_memory() -> SineClassMediatedMemory:
    """Rebuild fixed-model spatial factors without a response or source input.

    The shared rational partitioner is applied to the spatial A only. Its
    returned B*C is a spatial factor, not the complete two-channel kernel.
    Full physical blocks and moments retain their explicit gamma/cosine
    factors; no exponential, numerical inverse or trajectory is evaluated.
    """
    degrees = tuple(sum(node in edge for edge in _EDGES) for node in _NODES)
    form = _normalized_laplacian(_EDGES, degrees)
    internal = _normalized_laplacian(_EDGES[:27], degrees)
    contact = _normalized_laplacian(_EDGES[27:], degrees)
    partition = derive_coordinate_memory(form, _VISIBLE)

    def block(matrix, rows, columns):
        return tuple(tuple(matrix[i][j] for j in columns) for i in rows)

    avv, p, q, ahh = (
        partition.visible_generator,
        partition.hidden_to_visible,
        partition.visible_to_hidden,
        partition.hidden_generator,
    )
    nvv = block(internal, _VISIBLE, _VISIBLE)
    nhh = block(internal, _HIDDEN, _HIDDEN)
    wvv = block(contact, _VISIBLE, _VISIBLE)
    whh = block(contact, _HIDDEN, _HIDDEN)
    product = exact_matrix_product
    first = product(product(p, nhh), q)
    midpoint = tuple(
        tuple(Q(1, 2) if j in (4, 13) else Q(0) for j in range(18)) for _ in range(9)
    )
    midpoint_valid = (
        _zero(_add(product(ahh, midpoint), q))
        and _zero(product(nhh, midpoint))
        and _zero(_add(product(whh, midpoint), q))
    )
    grounded = tuple(
        tuple(degrees[node] * value for value in row) for node, row in zip(_HIDDEN, ahh)
    )
    positive = exact_symmetric_semidefinite(grounded, strict=True)
    gamma, cosines = _parameters((1, 2))
    eta, difference = gamma**2, cosines[0] - cosines[1]
    if not midpoint_valid or not positive or min(cosine.lo for cosine in cosines) <= 0:
        raise ArithmeticError("fixed grounded hidden midpoint identity failed")
    correction = product(p, midpoint)
    return SineClassMediatedMemory(
        nodes=_NODES,
        edges=_EDGES,
        degrees=degrees,
        visible_nodes=_VISIBLE,
        hidden_nodes=_HIDDEN,
        visible_indices=_VISIBLE + tuple(i + 27 for i in _VISIBLE),
        hidden_indices=_HIDDEN + tuple(i + 27 for i in _HIDDEN),
        normalized_laplacian_vv=avv,
        normalized_laplacian_vh=p,
        normalized_laplacian_hv=q,
        normalized_laplacian_hh=ahh,
        outer_phase_interior_vv=nvv,
        visible_phase_contact_vv=wvv,
        mediator_phase_interior_hh=nhh,
        hidden_phase_contact_hh=whh,
        gamma_bounds=gamma,
        eta_bounds=eta,
        class_cosine_bounds=cosines,
        class_cosine_difference_bounds=difference,
        kernel_at_zero_spatial_matrix=partition.kernel_at_zero,
        kernel_at_zero_scalar_bounds=((1 - eta, gamma), (-gamma, -eta)),
        kernel_derivative_difference_spatial_matrix=first,
        kernel_derivative_difference_scalar_bounds=(
            (eta * difference, I(0)),
            (-(gamma**3) * difference, I(0)),
        ),
        receiver_donor_kernel_derivative_difference_bounds=eta
        * difference
        * first[13][4],
        hidden_midpoint_map=midpoint,
        grounded_hidden_laplacian=grounded,
        grounded_hidden_laplacian_positive=positive,
        midpoint_equations_verified=midpoint_valid,
        quasistatic_form_matrix=_add(avv, correction),
        quasistatic_phase_constant_matrix=_add(wvv, correction),
        quasistatic_phase_cosine_matrix=nvv,
        hidden_to_visible_norm_upper_bound=(1 + gamma.hi) * _row_norm(p),
        visible_to_hidden_norm_upper_bound=(1 + gamma.hi) * _row_norm(q),
    )


@dataclass(frozen=True)
class SineClassMediatedMemoryBound:
    """Conditional finite reduction bound on a supplied reached-state ball.

    In each class, probe and baseline begin at the SAME full reached state;
    every component form and target-phase Euclidean norm is <=endpoint_radius.
    No formation/acquisition is assessed by this API. Individual causal laws
    retain hidden initial data; only their paired linear response cancels it.
    Prior identity and supplied event-work obligations are not discharged here.
    """

    probe_amplitude: Q
    contact_duration: Q
    endpoint_radius: Q
    readout_error_bound: Q
    memory: SineClassMediatedMemory
    bootstrap_margin: Q
    probe_form_norm_upper_bound: Q
    baseline_form_norm_upper_bound: Q
    probe_tangent_error_upper_bound: Q
    baseline_tangent_error_upper_bound: Q
    paired_class_reduction_error_upper_bound: Q
    whole_window_kernel_norm_upper_bound: Q
    hidden_initial_source_norm_upper_bound: Q
    leading_contrast_bounds: I
    linear_tail_upper_bound: Q
    tangent_class_contrast_bounds: I
    nonlinear_class_contrast_bounds: I
    readout_contrast_error_upper_bound: Q
    recorded_class_contrast_bounds: I
    quasistatic_tangent_recorded_contrast_bounds: I
    quasistatic_nonlinear_recorded_contrast_bounds: I
    quasistatic_exclusion_margin_bounds: I
    response_certified: bool
    quasistatic_comparator_excluded: bool
    status: str
    interval_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "conditional_per_component_reached_form_and_target_phase_norm_budgets",
        "no_acquisition_identity_or_work_certificate_and_no_cached_report_input",
        "same_full_reached_state_for_probe_and_baseline_within_each_class",
        "different_classes_may_have_independent_reached_residuals",
        "both_effective_trajectories_retain_hidden_initial_state_and_memory",
        "paired_linear_responses_cancel_their_entire_common_initial_state_exactly",
        "source_matched_nonlinear_error_is_two_times_probe_plus_baseline_error",
        "no_extra_initial_state_error_is_added_after_proved_paired_cancellation",
        "four_scalar_reading_errors_have_total_bound_four_delta",
        "selected_stationary_hidden_comparator_is_class_blind_not_all_local_laws",
        "nonlinear_midpoint_comparator_retains_independent_visible_initial_errors",
        "neither_stationary_comparator_is_an_admitted_fast_limit_or_identity_certificate",
        "bounds_do_not_assert_memory_decay_or_physical_identification",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {
            "schema": "tnfr.sine-class-mediated-memory-bound.v1",
            "report": _project(self),
        }


def bound_sine_class_mediated_memory(
    *,
    probe_amplitude,
    contact_duration,
    endpoint_radius,
    readout_error_bound,
) -> SineClassMediatedMemoryBound:
    """Bound the source-matched nonlinear remainder without acquiring a source.

    All four primitives are mandatory finite exact-or-represented reals.
    Require nonnegative amplitude, endpoint radius and scalar readout error,
    and 0<=contact_duration<=1/4. Zero budgets are admitted. Each initial
    component has both Euclidean errors <=endpoint_radius relative to its
    aligned class target. This is a conditional premise, never inferred from
    a saved certificate. No solver, source handoff or observation is consumed.
    """
    values = {
        name: exact_or_represented_real(value, name)
        for name, value in (
            ("probe_amplitude", probe_amplitude),
            ("contact_duration", contact_duration),
            ("endpoint_radius", endpoint_radius),
            ("readout_error_bound", readout_error_bound),
        )
    }
    if min(values.values()) < 0 or values["contact_duration"] > Q(1, 4):
        raise ValueError("require nonnegative budgets and 0<=contact_duration<=1/4")
    a, h, eps, delta = (
        values[key]
        for key in (
            "probe_amplitude",
            "contact_duration",
            "endpoint_radius",
            "readout_error_bound",
        )
    )
    memory = derive_sine_class_mediated_memory()
    reference = _mediation_reference_bounds(_central_port_geometry(3, _CONTACTS), a, h)
    g = reference.gamma.hi
    denominator = 1 - 2 * g**2 * h**2

    def error(amplitude):
        form = (amplitude + eps + 2 * g * h * eps) / denominator
        bound = (
            2 * g * eps**2 * h
            + 4 * g**2 * eps * form * h**2
            + Q(8, 3) * g**3 * form**2 * h**3
        ) / (1 - 3 * h)
        return form, bound

    probe_form, probe_error = error(a)
    baseline_form, baseline_error = error(Q(0))
    reduction = 2 * (probe_error + baseline_error)
    tangent = reference.leading + I(-reference.tail, reference.tail)
    nonlinear = tangent + I(-reduction, reduction)
    noise = 4 * delta
    recorded = nonlinear + I(-noise, noise)
    static_tangent = I(-noise, noise)
    static_error = 4 * eps / (1 - 3 * h) + noise
    static = I(-static_error, static_error)
    margin = recorded - static.hi
    positive = bool(a > 0 and h > 0 and recorded.lo > 0)
    excluded = bool(positive and margin.lo > 0)
    return SineClassMediatedMemoryBound(
        **values,
        memory=memory,
        bootstrap_margin=denominator,
        probe_form_norm_upper_bound=probe_form,
        baseline_form_norm_upper_bound=baseline_form,
        probe_tangent_error_upper_bound=probe_error,
        baseline_tangent_error_upper_bound=baseline_error,
        paired_class_reduction_error_upper_bound=reduction,
        whole_window_kernel_norm_upper_bound=(
            memory.hidden_to_visible_norm_upper_bound
            * memory.visible_to_hidden_norm_upper_bound
            / (1 - 3 * h)
        ),
        hidden_initial_source_norm_upper_bound=memory.hidden_to_visible_norm_upper_bound
        * eps
        / (1 - 3 * h),
        leading_contrast_bounds=reference.leading,
        linear_tail_upper_bound=reference.tail,
        tangent_class_contrast_bounds=tangent,
        nonlinear_class_contrast_bounds=nonlinear,
        readout_contrast_error_upper_bound=noise,
        recorded_class_contrast_bounds=recorded,
        quasistatic_tangent_recorded_contrast_bounds=static_tangent,
        quasistatic_nonlinear_recorded_contrast_bounds=static,
        quasistatic_exclusion_margin_bounds=margin,
        response_certified=positive,
        quasistatic_comparator_excluded=excluded,
        status="certified_conditional_contrast" if excluded else "bounds_only",
    )
