"""Prepared doubled-C5 replica pulses and their detached response reports.

The exact invariant preparation, variational blocks, finite work responses,
stiffness screen and asymptotic splitting retain their separate evidence scopes.
No report installs a runtime law or identifies a physical realization.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real
from ..dynamics.relational import RelationalExchangeModel
from ..mathematics._interval_taylor import MAX_ORDER, Jet
from ..mathematics._interval_taylor import cos as jet_cos
from ..mathematics._interval_taylor import sin as jet_sin
from ..mathematics._phase_resultant_chamber import (
    certified_cosine_bounds,
    certified_sine_bounds,
)
from ..mathematics._rational_interval import (
    INTERVAL_METHOD,
    I,
    cos,
    pi_interval,
    sin,
    sqrt,
)
from ..mathematics._validated_taylor import ValidatedTaylorStep, validated_taylor_step
from ._sine_admission import _sine_model_coefficients
from .relational_observations import _ordered
from .relational_sine_comparison import _sine_rates

__all__ = (
    "SineReplicaPulseAssessment",
    "SineReplicaPulseVariation",
    "SineReplicaPulseWorkResponse",
    "SineReplicaPulseFiniteWorkResponse",
    "SineReplicaStiffnessTraceCurve",
    "SineReplicaPulseSplitting",
    "assess_sine_replica_pulse",
    "assess_sine_replica_pulse_variation",
    "assess_sine_replica_pulse_work_response",
    "assess_sine_replica_pulse_finite_work_response",
    "assess_sine_replica_stiffness_trace_curve",
    "assess_sine_replica_pulse_splitting",
)


@dataclass(frozen=True)
class SineReplicaPulseAssessment:
    """Conditional nonlinear pulse on a declared exact doubled-C5 family.

    Every pair has the same signed u and delta, positive held capacity nu,
    uniform form mean and symbolic phase mean 2*pi*j/5. Common form and phase
    origins are immaterial. No represented graph has been checked against that
    irrational target or invariant preparation. The period returns the labeled
    fine state; half of it returns the unordered-pair state through a simultaneous
    member swap. This does not make any fine node disappear.
    """

    model: RelationalExchangeModel
    form_half_difference: Q
    phase_half_difference: Q
    capacity: Q
    target_phase_turns: tuple[Q, ...]
    phase_chart_margin_bounds: I
    twist_cosine_bounds: I
    resultant_magnitude_bounds: I
    internal_form_squared: Q
    form_phase_correlation_bounds: I
    internal_form_rate_bounds: I
    internal_phase_rate_bounds: I
    internal_energy_bounds: I
    normalized_energy_bounds: I
    separatrix_energy_bounds: I
    full_storage_bounds: I
    natural_angular_frequency_squared_bounds: I
    energy_regime: str
    status: str
    nonlinear_periodic_exchange_certified: bool
    small_amplitude_period_bounds: I
    small_amplitude_unordered_period_bounds: I
    period_bounds: I | None
    unordered_period_bounds: I | None
    period_unavailable_reasons: tuple[str, ...]
    acute_energy_threshold_bounds: I
    all_fine_edges_acute_margin_bounds: I
    all_fine_edges_acute_status: str
    graph_membership_certified: bool = False
    symbolic_family_invariant: bool = True
    collective_means_constant: bool = True
    law: str = "normalized_sine_reciprocal_exchange"
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "declared_exact_complete_two_replica_unit_C5_not_a_captured_graph",
        "symbolic_phase_mean_turns_j_over_5_and_uniform_form_mean_modulo_common_origins",
        "identical_signed_internal_form_and_phase_half_differences_in_every_pair",
        "common_positive_held_capacity_explicit_zero_loss_no_input_or_support_event",
        "exact_invariant_preparation_not_inferred_from_rounded_target_residuals",
        "full_nonlinear_internal_exchange_not_a_tangent_oscillator_or_added_law",
        "positive_subseparatrix_internal_energy_keeps_the_pair_chart_for_all_time",
        "zero_internal_form_with_nonzero_admitted_phase_proves_libration_from_the_chart",
        "labeled_period_returns_fine_members_unordered_period_is_half_through_pair_swaps",
        "finite_amplitude_period_is_strictly_above_its_small_amplitude_limit",
        "all_fine_edge_acuteness_is_a_separate_stronger_energy_condition",
        "over_barrier_and_unresolved_energy_receive_no_libration_certificate",
        "no_capture_membership_formation_attraction_generic_internal_sync_or_trajectory",
        "structural_model_time_without_a_laboratory_clock_or_physical_identification",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {
            "schema": "tnfr.relational-sine-replica-pulse.v1",
            "report": _project(self),
        }


def assess_sine_replica_pulse(
    *, reference_model, form_half_difference, phase_half_difference, capacity
) -> SineReplicaPulseAssessment:
    """Assess an exact prepared family without certifying a rounded graph.

    Complete doubled-C5 support, phase means 2*pi*j/5, uniform form mean and
    identical signed internal coordinates are explicit premises. The shared
    normalized sine rows reduce to u'=-a*nu*cos(2*pi/5)*cos(delta)*sin(delta)
    and delta'=b*nu*u. Their conserved internal energy is
    H=u**2+beta*cos(2*pi/5)*sin(delta)**2. A strict 0<H<beta*cos(2*pi/5)
    proves nonlinear libration of every retained pair, with constant means.

    Scalar admission preserves exact rationals. The input phase half-gap is a
    supplied real chart coordinate, not an automatically unwrapped difference.
    Period bounds reuse the existing P2 elliptic-integrand inequality; no
    trajectory or elliptic-function solver is installed or evaluated.
    """
    from .relational_sine_resonance import _elliptic_libration_period_bounds

    e, w, beta = _sine_model_coefficients(reference_model)
    if e != 0:
        raise ValueError("replica pulse assessment requires explicit zero form loss")
    u = exact_or_represented_real(form_half_difference, "form_half_difference")
    delta = exact_or_represented_real(phase_half_difference, "phase_half_difference")
    nu = exact_or_represented_real(capacity, "capacity")
    if nu <= 0:
        raise ValueError("capacity must be strictly positive")
    pi = pi_interval()
    chart_margin = pi / 2 - abs(delta)
    if chart_margin.lo <= 0:
        raise ValueError("phase_half_difference needs certified absolute value < pi/2")
    twist_cosine = cos(2 * pi / 5)
    sine = I(*certified_sine_bounds(delta))
    cosine = I(*certified_cosine_bounds(delta))
    sine_squared = sine**2
    ratio = I(u**2 / beta) / twist_cosine + sine_squared
    energy = I(u**2) + beta * twist_cosine * sine_squared
    barrier = beta * twist_cosine
    rates = _sine_rates(
        reference_model, (1,), (u,), (nu,), (-twist_cosine * cosine * sine,)
    )
    if u == 0 and delta == 0:
        energy_regime = "zero_energy"
        status = "equilibrium"
    elif u == 0 or ratio.hi < 1:
        # The admitted chart makes every nonzero delta have nonzero sine,
        # independently of whether its tiny energy lower bound is resolved.
        # With u=0 it also proves sin(delta)**2<1 despite outward rounding.
        energy_regime = "libration_energy"
        status = "libration_certified"
    elif ratio.lo > 1:
        energy_regime = "over_barrier_energy"
        status = "over_barrier_out_of_scope"
    else:
        energy_regime = "unresolved"
        status = "energy_classification_unresolved"

    base_period = 2 * pi**2 * sqrt(beta) / sqrt(twist_cosine)
    exact_scale = 1 / (w * nu)
    small_period = I(exact_scale * base_period.lo, exact_scale * base_period.hi)
    period = None
    unavailable = ("no_certified_nonstationary_libration",)
    if status == "libration_certified":
        margin = cosine**2 if u == 0 else I(1 - ratio.hi, 1 - ratio.lo)
        period, unavailable = _elliptic_libration_period_bounds(small_period, margin)
    acute_threshold = sin(pi / 20) ** 2
    acute_margin = acute_threshold - ratio
    if status in ("libration_certified", "equilibrium"):
        if acute_margin.lo > 0:
            acute_status = "certified"
        elif acute_margin.hi < 0:
            acute_status = "excluded"
        else:
            acute_status = "unresolved"
    else:
        acute_status = "unavailable_without_libration"
    return SineReplicaPulseAssessment(
        model=reference_model,
        form_half_difference=u,
        phase_half_difference=delta,
        capacity=nu,
        target_phase_turns=tuple(Q(j, 5) for j in range(5)),
        phase_chart_margin_bounds=chart_margin,
        twist_cosine_bounds=twist_cosine,
        resultant_magnitude_bounds=cosine,
        internal_form_squared=u**2,
        form_phase_correlation_bounds=u * sine,
        internal_form_rate_bounds=rates["form_rates"][0],
        internal_phase_rate_bounds=rates["phase_rates"][0],
        internal_energy_bounds=energy,
        normalized_energy_bounds=ratio,
        separatrix_energy_bounds=barrier,
        full_storage_bounds=20 * beta * (1 - twist_cosine) + 20 * energy,
        natural_angular_frequency_squared_bounds=(
            (w**2 * nu**2 / beta) * twist_cosine / pi**2
        ),
        energy_regime=energy_regime,
        status=status,
        nonlinear_periodic_exchange_certified=status == "libration_certified",
        small_amplitude_period_bounds=small_period,
        small_amplitude_unordered_period_bounds=small_period / 2,
        period_bounds=period,
        unordered_period_bounds=period / 2 if period is not None else None,
        period_unavailable_reasons=unavailable,
        acute_energy_threshold_bounds=acute_threshold,
        all_fine_edges_acute_margin_bounds=acute_margin,
        all_fine_edges_acute_status=acute_status,
    )


@dataclass(frozen=True)
class SineReplicaPulseVariation:
    """Complete instantaneous variational blocks of the exact prepared family.

    For Fourier convention exp(+2*pi*i*k*j/5), mode zero uses ordinary real
    (dX,dTheta,du,ddelta). For representatives k=1,2 the coordinates are
    (dX_hat,dTheta_hat,i*du_hat,i*ddelta_hat). Their real and imaginary parts
    separately obey the same real 4-by-4 block, retaining both conjugate modes.
    Multiplicities (1,2,2) therefore retain all twenty real fine directions.

    Blocks are evaluated at the declared initial internal state. The identity
    applies along the exact reference while its pair chart persists, but no
    propagator is computed. Periodic-reference availability is inherited from
    the pulse owner; instantaneous variation also exists at equilibrium or a
    reference whose energy classification is unavailable or outside libration.
    Neither coefficient periodicity nor the zero-amplitude spectrum decides
    finite-amplitude orbital stability.
    """

    pulse: SineReplicaPulseAssessment
    mode_indices: tuple[int, ...]
    mode_turns: tuple[Q, ...]
    mode_multiplicities: tuple[int, ...]
    laplacian_eigenvalue_bounds: tuple[I, ...]
    oriented_mode_sine_bounds: tuple[I, ...]
    mode_blocks: tuple[tuple[tuple[I, ...], ...], ...]
    dimensionless_mode_blocks: tuple[tuple[tuple[I, ...], ...], ...]
    form_normalization_bounds: I
    dimensionless_clock_rate_bounds: I
    symplectic_form: tuple[tuple[Q, ...], ...]
    hamiltonian_structure_residuals: tuple[tuple[tuple[I, ...], ...], ...]
    half_return_swap: tuple[tuple[Q, ...], ...]
    zero_amplitude_collective_frequency_squared_bounds: tuple[I, ...]
    zero_amplitude_internal_frequency_squared_bounds: I
    periodic_reference_certified: bool
    equilibrium_reference: bool
    half_return_swap_covariance_certified: bool
    full_real_dimension: int = 20
    variation_identity_certified: bool = True
    orbital_stability_status: str = "not_assessed"
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "exact_prepared_family_and_scalar_admission_reused_from_the_pulse_owner",
        "full_twenty_dimensional_fine_linearization_not_only_the_symmetric_pulse_tangent",
        "representative_Fourier_modes_zero_one_two_with_real_multiplicities_one_two_two",
        "positive_Fourier_exponential_and_i_times_internal_coordinates_for_nonzero_modes",
        "mode_blocks_are_initial_coefficients_not_finite_time_transition_maps",
        "dimensionless_forms_divide_by_sqrt_beta_cos_alpha_and_tau_equals_Omega_t",
        "dimensionless_blocks_depend_on_instantaneous_delta_not_energy_parameter_alone",
        "instantaneous_variation_identity_does_not_require_a_certified_periodic_reference",
        "periodic_reference_and_half_return_covariance_require_the_separate_pulse_certificate",
        "half_return_conjugates_by_member_swap_instead_of_making_every_block_half_periodic",
        "fixed_linear_symplectic_identity_is_not_a_new_microscopic_constitutive_law",
        "zero_amplitude_frequency_bounds_describe_the_equilibrium_limit_only",
        "no_monodromy_multipliers_resonance_scan_or_finite_amplitude_stability_certificate",
        "no_graph_membership_solver_trajectory_support_event_or_runtime_law_change",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {
            "schema": "tnfr.relational-sine-replica-pulse-variation.v1",
            "report": _project(self),
        }


def _replica_variation_blocks(
    *,
    internal_cosine,
    internal_sine,
    double_internal_cosine,
    laplacian,
    mode_sines,
    stiffness,
    mobility,
    coupling,
):
    """Share the admitted real Fourier blocks between interval and jet readers."""
    if isinstance(internal_cosine, Jet):

        def constant(value):
            return Jet.constant(value, internal_cosine.order)

    else:
        constant = I.coerce
    stiffness, mobility, coupling = map(constant, (stiffness, mobility, coupling))
    zero = constant(0)
    result = []
    for eigenvalue, mode_sine in zip(laplacian, mode_sines):
        cross = -coupling * internal_cosine * internal_sine * mode_sine
        collective_phase = -stiffness * internal_cosine**2 * eigenvalue / 2
        internal_phase = -stiffness * (
            double_internal_cosine + internal_sine**2 * eigenvalue / 2
        )
        result.append(
            (
                (zero, collective_phase, zero, cross),
                (mobility * eigenvalue / 2, zero, zero, zero),
                (zero, cross, zero, internal_phase),
                (zero, zero, mobility, zero),
            )
        )
    return tuple(result)


def assess_sine_replica_pulse_variation(
    *, reference_model, form_half_difference, phase_half_difference, capacity
) -> SineReplicaPulseVariation:
    """Reduce the full fine variational field without integrating any block.

    Reusing pulse admission keeps the symbolic irrational target distinct from
    a rounded graph. The exact two-replica symmetry separates the full field
    into three real block types. Nonzero Fourier representatives use a fixed
    complex coordinate change whose real and imaginary parts supply independent
    copies of the displayed real block; no fine perturbation is discarded.
    """
    pulse = assess_sine_replica_pulse(
        reference_model=reference_model,
        form_half_difference=form_half_difference,
        phase_half_difference=phase_half_difference,
        capacity=capacity,
    )
    pi = pi_interval()
    modes = (0, 1, 2)
    turns = tuple(Q(k, 5) for k in modes)
    cosines = tuple(cos(2 * turn * pi) for turn in turns)
    sines = tuple(sin(2 * turn * pi) for turn in turns)
    laplacian = tuple(2 - 2 * cosine for cosine in cosines)
    internal_cosine = pulse.resultant_magnitude_bounds
    internal_sine = I(*certified_sine_bounds(pulse.phase_half_difference))
    double_internal_cosine = I(
        *certified_cosine_bounds(2 * pulse.phase_half_difference)
    )
    twist_sine = sin(2 * pi / 5)
    # Unit form/current directions expose the existing a*nu and b*nu rates.
    coefficients = _sine_rates(
        reference_model, (1,), (Q(1),), (pulse.capacity,), (I(1),)
    )
    a_nu = coefficients["form_rates"][0]
    b_nu = coefficients["phase_rates"][0]

    def assemble_blocks(stiffness, mobility, coupling):
        return _replica_variation_blocks(
            internal_cosine=internal_cosine,
            internal_sine=internal_sine,
            double_internal_cosine=double_internal_cosine,
            laplacian=laplacian,
            mode_sines=sines,
            stiffness=stiffness,
            mobility=mobility,
            coupling=coupling,
        )

    blocks = assemble_blocks(a_nu * pulse.twist_cosine_bounds, b_nu, a_nu * twist_sine)
    dimensionless_blocks = assemble_blocks(
        I(1), I(1), twist_sine / pulse.twist_cosine_bounds
    )
    symplectic = tuple(
        tuple(map(Q, row))
        for row in ((0, -1, 0, 0), (1, 0, 0, 0), (0, 0, 0, -1), (0, 0, 1, 0))
    )
    swap = tuple(
        tuple(Q((1 if i < 2 else -1) if i == j else 0) for j in range(4))
        for i in range(4)
    )
    structure_residuals = tuple(
        tuple(
            tuple(
                sum(
                    (
                        block[k][i] * symplectic[k][j] + symplectic[i][k] * block[k][j]
                        for k in range(4)
                    ),
                    I(0),
                )
                for j in range(4)
            )
            for i in range(4)
        )
        for block in blocks
    )
    return SineReplicaPulseVariation(
        pulse=pulse,
        mode_indices=modes,
        mode_turns=turns,
        mode_multiplicities=(1, 2, 2),
        laplacian_eigenvalue_bounds=laplacian,
        oriented_mode_sine_bounds=sines,
        mode_blocks=blocks,
        dimensionless_mode_blocks=dimensionless_blocks,
        form_normalization_bounds=sqrt(pulse.separatrix_energy_bounds),
        dimensionless_clock_rate_bounds=sqrt(
            pulse.natural_angular_frequency_squared_bounds
        ),
        symplectic_form=symplectic,
        hamiltonian_structure_residuals=structure_residuals,
        half_return_swap=swap,
        zero_amplitude_collective_frequency_squared_bounds=tuple(
            pulse.natural_angular_frequency_squared_bounds * eigenvalue**2 / 4
            for eigenvalue in laplacian
        ),
        zero_amplitude_internal_frequency_squared_bounds=(
            pulse.natural_angular_frequency_squared_bounds
        ),
        periodic_reference_certified=pulse.nonlinear_periodic_exchange_certified,
        equilibrium_reference=pulse.status == "equilibrium",
        half_return_swap_covariance_certified=pulse.nonlinear_periodic_exchange_certified,
    )


@dataclass(frozen=True)
class SineReplicaPulseWorkResponse:
    """Finite-time infinitesimal work response along a declared moving family.

    The two distributed form impulses are fC=(cos(alpha*j),cos(alpha*j)) and
    fI=(sin(alpha*j),-sin(alpha*j)), alpha=2*pi/5, in ordered fine pairs.
    Their conjugate work outputs are fC^T L delta_x and fI^T L delta_x.
    Both use the same declared moving background, with its unperturbed motion
    subtracted. This is not a finite-amplitude kick or a frequency transfer.
    """

    reference: SineReplicaPulseVariation
    scaled_duration: Q
    taylor_order: int
    initial_state_bounds: tuple[I, ...]
    work_output_coefficients: tuple[I, I]
    response_bounds: tuple[tuple[I, I], tuple[I, I]] | None
    antisymmetric_response_bounds: I | None
    response_sign: int | None
    directional_response_certified: bool
    step: ValidatedTaylorStep | None
    failed_tube: tuple[I, ...] | None
    status: str
    unavailable_reasons: tuple[str, ...]
    mode_index: int = 1
    clock: str = "tau=t/pi"
    arithmetic_method: str = INTERVAL_METHOD
    state_order: tuple[str, ...] = (
        "u",
        "delta",
        "collective_impulse_dX",
        "collective_impulse_dTheta",
        "collective_impulse_i_du",
        "collective_impulse_i_ddelta",
        "internal_impulse_dX",
        "internal_impulse_dTheta",
        "internal_impulse_i_du",
        "internal_impulse_i_ddelta",
    )
    scope: tuple[str, ...] = (
        "exact_prepared_doubled_C5_family_rebuilt_from_primitive_model_and_internal_state",
        "symbolic_phase_means_two_pi_j_over_five_not_a_captured_graph_membership_claim",
        "fixed_real_k1_fourier_copy_with_two_distributed_same_type_form_impulses",
        "fine_port_norm_squared_five_and_work_output_coefficients_twenty_times_diag_lambda_over_two_one",
        "full_fine_variation_reduction_reused_no_internal_motion_discarded",
        "pulse_and_variational_columns_evolve_jointly_no_frozen_instantaneous_generator",
        "strict_whole_time_Picard_tube_and_Taylor_remainder_under_declared_numerical_budget",
        "response_sign_zero_means_interval_overlaps_zero_not_exact_reciprocity",
        "original_model_time_t_equals_pi_tau_not_the_separate_Omega_t_normalization",
        "response_is_derivative_at_zero_kick_with_unperturbed_motion_subtracted",
        "no_finite_kick_error_monodromy_stability_hall_or_physical_identification",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {
            "schema": "tnfr.relational-sine-replica-pulse-work-response.v1",
            "report": _project(self),
        }


def assess_sine_replica_pulse_work_response(
    *,
    reference_model,
    form_half_difference,
    phase_half_difference,
    capacity,
    scaled_duration,
    order=10,
) -> SineReplicaPulseWorkResponse:
    """Bound two matched work responses on one exact moving background.

    The fixed k=1 real block retains a collective cosine form perturbation and
    an internal sine form perturbation. Pulse motion plus both four-coordinate
    variation columns are enclosed together by the shared validated kernel.
    Duration uses tau=t/pi; w, beta and positive held capacity are retained.
    Numerical failure returns unavailable evidence without changing the budget.
    """
    duration = exact_or_represented_real(scaled_duration, "scaled_duration")
    if duration <= 0:
        raise ValueError("scaled_duration must be strictly positive")
    if type(order) is not int or not 1 <= order <= MAX_ORDER:
        raise ValueError("Taylor order outside the shared jet domain")
    reference = assess_sine_replica_pulse_variation(
        reference_model=reference_model,
        form_half_difference=form_half_difference,
        phase_half_difference=phase_half_difference,
        capacity=capacity,
    )
    pulse = reference.pulse
    _, weight, beta = _sine_model_coefficients(pulse.model)
    exchange = weight * pulse.capacity
    mobility = exchange / beta
    pi = pi_interval()
    twist_cosine = pulse.twist_cosine_bounds
    twist_sine = sin(2 * pi / 5)
    eigenvalue = reference.laplacian_eigenvalue_bounds[1]
    mode_sine = reference.oriented_mode_sine_bounds[1]
    stiffness = exchange * twist_cosine
    coupling = exchange * twist_sine
    initial = tuple(
        I(value)
        for value in (
            pulse.form_half_difference,
            pulse.phase_half_difference,
            1,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
        )
    )

    def field(state):
        u, delta = state[:2]
        if isinstance(delta, Jet):
            sine, cosine, double_cosine = (
                jet_sin(delta),
                jet_cos(delta),
                jet_cos(2 * delta),
            )
        else:
            sine, cosine, double_cosine = sin(delta), cos(delta), cos(2 * delta)
        block = _replica_variation_blocks(
            internal_cosine=cosine,
            internal_sine=sine,
            double_internal_cosine=double_cosine,
            laplacian=(eigenvalue,),
            mode_sines=(mode_sine,),
            stiffness=stiffness,
            mobility=mobility,
            coupling=coupling,
        )[0]
        rates = [sine * cosine * -stiffness, u * mobility]
        for start in (2, 6):
            vector = state[start : start + 4]
            rates.extend(
                sum((value * coefficient for value, coefficient in zip(vector, row)), 0)
                for row in block
            )
        return tuple(rates)

    def domain(tube):
        return (pi.lo / 2 - tube[1].abs_max,)

    step, failed, reason = validated_taylor_step(
        initial,
        duration,
        field,
        domain,
        order=order,
        domain_failure="whole_time_pair_chart_not_certified",
    )
    outputs = (10 * eigenvalue, I(20))
    response = contrast = response_sign = None
    if step is not None:
        response = tuple(
            tuple(outputs[row] * step.endpoint[start + 2 * row] for start in (2, 6))
            for row in range(2)
        )
        contrast = response[0][1] - response[1][0]
        response_sign = 1 if contrast.lo > 0 else -1 if contrast.hi < 0 else 0
    return SineReplicaPulseWorkResponse(
        reference=reference,
        scaled_duration=duration,
        taylor_order=order,
        initial_state_bounds=initial,
        work_output_coefficients=outputs,
        response_bounds=response,
        antisymmetric_response_bounds=contrast,
        response_sign=response_sign,
        directional_response_certified=response_sign in (-1, 1),
        step=step,
        failed_tube=failed,
        status="certified_finite_response" if step is not None else "unavailable",
        unavailable_reasons=() if reason is None else (reason,),
    )


@dataclass(frozen=True)
class SineReplicaPulseFiniteWorkResponse:
    """Four complete nonlinear preparations under one fixed response protocol.

    Unlike the tangent report, each preparation changes all ten fine forms by
    a nonzero distributed impulse and evolves all twenty fine coordinates.
    The symbolic irrational phases and probe masks are enclosed, never replaced
    by rounded graph values. Interval boxes may conservatively include states
    outside that correlated symbolic preparation.
    """

    tangent: SineReplicaPulseWorkResponse
    probe_amplitude: Q
    taylor_order: int
    neighbors: tuple[tuple[int, ...], ...]
    edges: tuple[tuple[int, int], ...]
    base_state_bounds: tuple[I, ...]
    port_form_directions: tuple[tuple[I, ...], tuple[I, ...]]
    initial_boxes: tuple[tuple[I, ...], ...]
    steps: tuple[ValidatedTaylorStep | None, ...]
    failed_tubes: tuple[tuple[I, ...] | None, ...]
    failure_reasons: tuple[str | None, ...]
    response_bounds: tuple[tuple[I, I], tuple[I, I]] | None
    antisymmetric_response_bounds: I | None
    response_sign: int | None
    numerical_directional_response_certified: bool
    flow_third_derivative_upper_bound: Q
    finite_probe_error_upper_bound: Q
    analytic_tangent_lower_bound: Q
    analytic_contrast_lower_bound: Q
    analytic_directional_response_certified: bool
    predicted_antisymmetric_response_bounds: I | None
    fine_edge_margin_lower_bound: Q
    status: str
    unavailable_reasons: tuple[str, ...]
    scaled_duration: Q = Q(1, 16)
    form_half_difference: Q = Q(1, 32)
    phase_half_difference: Q = Q(1, 32)
    capacity: Q = Q(1)
    mode_index: int = 1
    clock: str = "tau=t/pi"
    preparation_order: tuple[str, ...] = (
        "collective_positive",
        "collective_negative",
        "internal_positive",
        "internal_negative",
    )
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "fixed_complete_doubled_C5_unit_capacity_zero_loss_w_beta_one_smooth_sine_protocol",
        "ordered_fine_pairs_2j_2j_plus_one_all_four_edges_per_neighbor_pair",
        "symbolic_phases_two_pi_j_over_five_plus_minus_one_over_32_and_symbolic_probe_masks",
        "four_complete_twenty_coordinate_nonlinear_flows_no_Fourier_reduced_kick_evolution",
        "same_type_collective_cosine_and_signed_internal_sine_form_impulses",
        "conjugate_work_outputs_f_transpose_L_x_with_centered_difference_over_twice_epsilon",
        "shared_original_time_sine_rows_multiplied_by_outward_pi_for_tau_clock",
        "fixed_horizon_one_over_16_and_requested_order_no_retry_or_budget_change",
        "analytic_finite_probe_error_uses_global_full_flow_third_variation_bound",
        "analytic_prediction_and_independent_numerical_response_keep_separate_availability",
        "response_sign_zero_means_interval_overlaps_zero_not_exact_reciprocity",
        "no_graph_capture_measurement_noise_physical_clock_or_hall_identification",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {
            "schema": "tnfr.relational-sine-replica-pulse-finite-work-response.v1",
            "report": _project(self),
        }


def assess_sine_replica_pulse_finite_work_response(
    *, probe_amplitude, order=10
) -> SineReplicaPulseFiniteWorkResponse:
    """Evaluate the fixed four-kick protocol and its prior analytic prediction.

    The declared family is u=delta=1/32 on the complete doubled C5, with unit
    capacity, w=beta=1, zero loss and tau horizon 1/16. Only a positive probe
    amplitude at most 2**-20 and a shared Taylor order are admitted. These are
    a conditional experiment's supplied parameters, not selected physical laws.
    All four trials retain their evidence even if another trial is unavailable.
    """
    from .relational_sine_forecast import _sine_flow

    epsilon = exact_or_represented_real(probe_amplitude, "probe_amplitude")
    if not 0 < epsilon <= Q(1, 1 << 20):
        raise ValueError("probe_amplitude must lie in (0, 2**-20]")
    if type(order) is not int or not 1 <= order <= MAX_ORDER:
        raise ValueError("Taylor order outside the shared jet domain")
    model = RelationalExchangeModel(
        1, epi_weight=0, phase_weight=1, phase_domain="regular"
    )
    half_difference, duration = Q(1, 32), Q(1, 16)
    tangent = assess_sine_replica_pulse_work_response(
        reference_model=model,
        form_half_difference=half_difference,
        phase_half_difference=half_difference,
        capacity=1,
        scaled_duration=duration,
        order=order,
    )
    pi = pi_interval()
    phases = tuple(2 * pi * Q(j, 5) for j in range(5))
    signs = (1, -1)
    base = tuple(I(sign * half_difference) for _ in range(5) for sign in signs) + tuple(
        phase + sign * half_difference for phase in phases for sign in signs
    )
    ports = (
        tuple(cos(phase) for phase in phases for _ in signs),
        tuple(sign * sin(phase) for phase in phases for sign in signs),
    )
    edges = tuple(
        sorted(
            tuple(sorted((2 * j + member, 2 * ((j + 1) % 5) + neighbor)))
            for j in range(5)
            for member in range(2)
            for neighbor in range(2)
        )
    )
    neighbors = tuple(
        tuple(
            sorted(
                right if left == i else left
                for left, right in edges
                if i in (left, right)
            )
        )
        for i in range(10)
    )
    initial = tuple(
        tuple(base[i] + sign * epsilon * port[i] for i in range(10)) + base[10:]
        for port in ports
        for sign in signs
    )

    def field(state):
        held_capacity = (
            Jet.constant(1, state[0].order) if isinstance(state[0], Jet) else I(1)
        )
        original_rates = _sine_flow(
            tuple(state) + (held_capacity,),
            neighbors=neighbors,
            visible_capacity=(Q(1),) * 9,
            model=model,
        )
        return tuple(rate * pi for rate in original_rates[:-1])

    # The full sine law is smooth on real phase lifts. Acute identity is
    # covered separately by the analytic margin; it is not a solver domain.
    def domain(_tube):
        return (Q(1),)

    steps, failed_tubes, failure_reasons = [], [], []
    for box in initial:
        step, failed, reason = validated_taylor_step(
            box, duration, field, domain, order=order
        )
        steps.append(step)
        failed_tubes.append(failed)
        failure_reasons.append(reason)

    response = contrast = response_sign = None
    if all(step is not None for step in steps):
        entries = []
        for port in ports:
            row = []
            for column in range(2):
                positive = steps[2 * column].endpoint
                negative = steps[2 * column + 1].endpoint
                work = sum(
                    (
                        (port[i] - port[j])
                        * (positive[i] - positive[j] - negative[i] + negative[j])
                        for i, j in edges
                    ),
                    I(0),
                )
                # Preserve exact tiny amplitudes: dividing by I(2*epsilon)
                # could manufacture a denominator containing zero.
                row.append(work * (1 / (2 * epsilon)))
            entries.append(tuple(row))
        response = tuple(entries)
        contrast = response[0][1] - response[1][0]
        response_sign = 1 if contrast.lo > 0 else -1 if contrast.hi < 0 else 0

    # Full-field derivative comparison: exp(2*h) <= E = 1/(1-2*h).
    # Each centered work readout has l1 norm <=40; their difference adds
    # two central-difference remainders, each divided by 3!.
    growth = 1 / (1 - 2 * duration)
    derivative_bound = 4 * growth * (growth - 1) * (2 * growth - 1)
    error = Q(80, 6) * derivative_bound * epsilon**2
    # Independent ordered-Volterra lower bound for the same fixed pulse and
    # two ports. These are proved bounds, not fitted endpoint coefficients.
    stiffness_norm = Q(3, 8)
    leading = Q(33, 800000) * duration**5
    tail = (
        2
        * stiffness_norm**3
        * duration**6
        / (720 * (1 - stiffness_norm * duration**2 / 56))
    )
    lower = 20 * (leading - tail)
    predicted = (
        tangent.antisymmetric_response_bounds + I(-error, error)
        if tangent.antisymmetric_response_bounds is not None
        else None
    )
    names = SineReplicaPulseFiniteWorkResponse.preparation_order
    reasons = tuple(
        f"{name}: {reason}"
        for name, reason in zip(names, failure_reasons)
        if reason is not None
    )
    return SineReplicaPulseFiniteWorkResponse(
        tangent=tangent,
        probe_amplitude=epsilon,
        taylor_order=order,
        neighbors=neighbors,
        edges=edges,
        base_state_bounds=base,
        port_form_directions=ports,
        initial_boxes=initial,
        steps=tuple(steps),
        failed_tubes=tuple(failed_tubes),
        failure_reasons=tuple(failure_reasons),
        response_bounds=response,
        antisymmetric_response_bounds=contrast,
        response_sign=response_sign,
        numerical_directional_response_certified=response_sign in (-1, 1),
        flow_third_derivative_upper_bound=derivative_bound,
        finite_probe_error_upper_bound=error,
        analytic_tangent_lower_bound=lower,
        analytic_contrast_lower_bound=lower - error,
        analytic_directional_response_certified=lower > error,
        predicted_antisymmetric_response_bounds=predicted,
        fine_edge_margin_lower_bound=Q(13, 60) - 2 * growth * epsilon,
        status="certified_finite_response" if response is not None else "unavailable",
        unavailable_reasons=reasons,
    )


@dataclass(frozen=True)
class SineReplicaStiffnessTraceCurve:
    """Necessary three-observation screen, not a physical-model admission.

    Inputs enclose trace and determinant of a mass-normalized two-mode
    stiffness, with the same independently declared constant kinetic mass,
    clock and coordinate convention. Correlations can be conservatively
    discarded by the input boxes. An unresolved residual never proves the
    complete law, autonomous preparation or a physical correspondence.
    """

    trace_bounds: tuple[I, ...]
    determinant_bounds: tuple[I, ...]
    geometry_coefficient_bounds: I
    trace_difference_product_bounds: I
    divided_difference_numerator_bounds: I
    template_residual_bounds: I
    affine_obstruction_bounds: I
    trace_separation_certified: bool
    template_curve_status: str
    affine_family_status: str
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "necessary_curve_of_declared_doubled_C5_k1_conservative_pulse_stiffness",
        "trace_and_determinant_of_M_inverse_K_not_unweighted_K_if_mass_is_nonidentity",
        "constant_positive_definite_mass_and_fixed_work_compatible_coordinates",
        "constant_clock_scaling_preserves_curvature_coefficient",
        "cross_multiplied_interval_predicate_no_division_by_small_trace_gaps",
        "geometry_coefficient_is_not_a_universal_TNFR_or_physical_constant",
        "affine_family_is_symmetric_J0_plus_rho_t_J1_with_fixed_matrices",
        "not_excluded_is_not_exact_equality_sufficiency_or_physical_admission",
        "no_phase_clock_fit_trajectory_dataset_score_or_runtime_mutation",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {
            "schema": "tnfr.relational-sine-replica-stiffness-trace-curve.v1",
            "report": _project(self),
        }


def assess_sine_replica_stiffness_trace_curve(
    *, trace_bounds, determinant_bounds
) -> SineReplicaStiffnessTraceCurve:
    """Screen three declared invariant observations, retaining uncertainty.

    Each ordered input contains three admitted real scalars or outward
    intervals. Separated trace boxes and a residual excluding zero reject
    this template. Strictly positive affine obstruction rejects a constant
    symmetric affine one-control stiffness family. Other outcomes abstain
    from claiming either model is established.
    """
    from .relational_sine_regional import _interval

    rows = []
    for values, label in (
        (trace_bounds, "trace_bounds"),
        (determinant_bounds, "determinant_bounds"),
    ):
        values = _ordered(values, label, limit=4)
        if len(values) != 3:
            raise ValueError(f"{label} requires exactly three observations")
        rows.append(
            tuple(_interval(value, f"{label}[{i}]") for i, value in enumerate(values))
        )
    traces, determinants = rows
    t0, t1, t2 = traces
    v0, v1, v2 = determinants
    product = (t2 - t1) * (t1 - t0) * (t2 - t0)
    numerator = (v2 - v1) * (t1 - t0) - (v1 - v0) * (t2 - t1)
    coefficient = (1160 + 480 * sqrt(5)) / 1089
    residual = numerator - coefficient * product
    obstruction = (4 * numerator - product) * product
    separated = product.lo > 0 or product.hi < 0
    return SineReplicaStiffnessTraceCurve(
        trace_bounds=traces,
        determinant_bounds=determinants,
        geometry_coefficient_bounds=coefficient,
        trace_difference_product_bounds=product,
        divided_difference_numerator_bounds=numerator,
        template_residual_bounds=residual,
        affine_obstruction_bounds=obstruction,
        trace_separation_certified=separated,
        template_curve_status=(
            "unresolved_trace_separation"
            if not separated
            else "excluded" if residual.lo > 0 or residual.hi < 0 else "not_excluded"
        ),
        affine_family_status=(
            "unresolved_trace_separation"
            if not separated
            else "excluded" if obstruction.lo > 0 else "not_excluded"
        ),
    )


@dataclass(frozen=True)
class SineReplicaPulseSplitting:
    """Asymptotic internal-return splitting, with no numerical amplitude radius.

    ``reference`` is the explicitly declared zero-amplitude equilibrium, not an
    observed or evaluated finite preparation. For m=H/(beta*cos(alpha)) tending
    to zero through positive values, use the fixed-period clock
    sigma=pi*Omega*t/(2*K(m)). In a parity-adapted analytic internal basis the
    effective return logarithmic generator log(M_internal)/(2*pi) is
    m*G+O(m**2), where G=[[0,Q/2],[-P/2,0]]. This is not a constant raw
    instantaneous variational row. The eigenvalues of G have squared value
    -P*Q/4. Coefficient pairs encode exact a+b*sqrt(5), not rounded rational
    replacements for the irrational geometry.

    The leading labeled and swap-correct half-return log-multiplier magnitudes
    are respectively 2*pi*m*sqrt(abs(-P*Q/4)) and half that value. The reported
    slope bounds enclose their coefficients of m, not finite log multipliers.
    Their sign type is real for the hyperbolic mode and imaginary for the
    elliptic mode.
    Analytic perturbation proves the classification on some positive interval;
    no explicit upper endpoint, remainder constant, finite multiplier or
    selected-amplitude verdict is provided. In particular, the stationary
    reference itself is not declared unstable by this positive-amplitude result.
    """

    reference: SineReplicaPulseVariation
    mode_indices: tuple[int, ...]
    mode_multiplicities: tuple[int, ...]
    p_exact_coefficients: tuple[tuple[Q, Q], ...]
    q_exact_coefficients: tuple[tuple[Q, Q], ...]
    p_bounds: tuple[I, ...]
    q_bounds: tuple[I, ...]
    slow_generators: tuple[tuple[tuple[I, ...], ...], ...]
    squared_slow_exponent_bounds: tuple[I, ...]
    leading_exponent_magnitude_bounds: tuple[I | None, ...]
    labeled_log_multiplier_slope_magnitude_bounds: tuple[I | None, ...]
    unordered_log_multiplier_slope_magnitude_bounds: tuple[I | None, ...]
    mode_classifications: tuple[str, ...]
    existential_amplitude_interval_certified: bool
    sufficiently_small_nonlinear_orbital_instability_certified: bool
    amplitude_upper_bound: Q | None = None
    finite_amplitude_remainder_bound: Q | None = None
    finite_preparation_assessed: bool = False
    return_multipliers_computed: bool = False
    coefficient_basis: str = "a_plus_b_sqrt5"
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "same_declared_conservative_doubled_C5_family_and_shared_scalar_admission",
        "reference_is_the_exact_zero_amplitude_template_not_a_measured_preparation",
        "positive_amplitude_m_equals_H_over_beta_cos_alpha_tends_to_zero",
        "fixed_period_clock_includes_the_first_elliptic_period_correction",
        "collective_sector_is_retained_in_the_internal_resonant_solvability_coefficients",
        "real_spatial_modes_one_and_two_each_have_multiplicity_two",
        "isolated_internal_symplectic_return_pair_and_analytic_remainder_give_existential_interval",
        "mode_one_hyperbolicity_implies_orbital_instability_of_sufficiently_small_nonzero_pulses",
        "mode_two_ellipticity_alone_does_not_establish_nonlinear_orbital_stability",
        "equilibrium_reference_is_not_given_a_finite_amplitude_instability_verdict",
        "no_explicit_amplitude_radius_remainder_constant_or_finite_preparation_certificate",
        "no_monodromy_solver_multiplier_sample_scan_trajectory_or_native_runtime_change",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {
            "schema": "tnfr.relational-sine-replica-pulse-splitting.v1",
            "report": _project(self),
        }


def assess_sine_replica_pulse_splitting(
    *, reference_model, capacity
) -> SineReplicaPulseSplitting:
    """Enclose the proved leading splitting without selecting an amplitude.

    The shared exact equilibrium template supplies model, capacity, support
    and Fourier provenance. The C5-specific quadratic coefficients below are
    the evaluated harmonic-solvability formulas from the scale proof, including
    collective backreaction and the varying-period clock correction. They are
    not empirical fits, finite return samples or a new algebraic-field model.
    Their resolved signs support an existential sufficiently-small interval;
    no finite source amplitude is accepted or inferred by this API.
    """
    reference = assess_sine_replica_pulse_variation(
        reference_model=reference_model,
        form_half_difference=Q(0),
        phase_half_difference=Q(0),
        capacity=capacity,
    )
    # The exact C5 expressions are retained as a+b*sqrt(5). Interval evaluation
    # uses the shared radical enclosure; no irrational coefficient is replaced
    # by a rationalized floating-point value in the proof's model.
    p_exact = ((Q(95, 164), Q(1, 164)), (Q(55, 41), Q(21, 41)))
    q_exact = ((Q(-479, 164), Q(-245, 164)), (Q(14, 41), Q(21, 41)))
    radical = sqrt(5)
    p_bounds = tuple(a + b * radical for a, b in p_exact)
    q_bounds = tuple(a + b * radical for a, b in q_exact)
    generators = tuple(
        ((I(0), q / 2), (-p / 2, I(0))) for p, q in zip(p_bounds, q_bounds)
    )
    squared = tuple(-p * q / 4 for p, q in zip(p_bounds, q_bounds))
    classifications = tuple(
        "hyperbolic" if value.lo > 0 else "elliptic" if value.hi < 0 else "unresolved"
        for value in squared
    )
    magnitudes = tuple(
        (
            sqrt(value)
            if kind == "hyperbolic"
            else sqrt(-value) if kind == "elliptic" else None
        )
        for value, kind in zip(squared, classifications)
    )
    pi = pi_interval()
    return SineReplicaPulseSplitting(
        reference=reference,
        mode_indices=(1, 2),
        mode_multiplicities=(2, 2),
        p_exact_coefficients=p_exact,
        q_exact_coefficients=q_exact,
        p_bounds=p_bounds,
        q_bounds=q_bounds,
        slow_generators=generators,
        squared_slow_exponent_bounds=squared,
        leading_exponent_magnitude_bounds=magnitudes,
        labeled_log_multiplier_slope_magnitude_bounds=tuple(
            2 * pi * value if value is not None else None for value in magnitudes
        ),
        unordered_log_multiplier_slope_magnitude_bounds=tuple(
            pi * value if value is not None else None for value in magnitudes
        ),
        mode_classifications=classifications,
        existential_amplitude_interval_certified=all(
            kind != "unresolved" for kind in classifications
        ),
        sufficiently_small_nonlinear_orbital_instability_certified=(
            classifications[0] == "hyperbolic"
        ),
    )
