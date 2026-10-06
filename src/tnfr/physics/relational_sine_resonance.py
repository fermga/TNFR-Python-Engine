"""Declared resonance and static storage controls on full nodal support.

The nonlinear phase/form exchange and work identity remain owned by the shared
sine field. This reader differentiates that same law about an exact cycle
target and evaluates specified input/output transfer functions. A second reader
reuses full-support recovery to certify a collocated work-port maximum without
computing its frequency. A mediated reader retains both complete C5 patterns
and their intermediary to distinguish remote response from silent sectors.
A separate P2 reader classifies nonlinear reversible exchange using the
already admitted zero-loss boundary. The conservative P3 reader reuses exact
coordinate memory after a declared clock change, retaining hidden initial
state. A nonlinear family reader checks compact invariant slabs for the
almost-everywhere recurrence theorem, without certifying a chosen moving
state. A static bridge control additionally declares the supplied U_epsilon
sine-plus-cubic storage to separate critical geometry from its response.
None installs forcing, an oscillator, an event or a native law.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from numbers import Integral
from typing import TYPE_CHECKING, Any

from .._exact_time import exact_or_represented_real
from ..dynamics.relational import RelationalExchangeModel
from ..mathematics._exact_linear_algebra import (
    exact_matrix_inverse,
    exact_matrix_product,
    exact_symmetric_semidefinite,
)
from ..mathematics._rational_interval import INTERVAL_METHOD, I, cos, pi_interval, sqrt
from ._sine_admission import _sine_model_coefficients
from .relational_observations import _ordered
from .relational_sine_comparison import (
    SineExchangeComparison,
    _sine_rates,
    _validate_comparison_labels,
    bound_relational_sine_exchange,
)

if TYPE_CHECKING:
    from ..mathematics.linear_observation import (
        LinearCoordinateMemory,
        LinearObservation,
    )
    from .phase_cycle_geometry import CircularPhaseState, PhaseCycleState
    from .relational_sine_recovery import SineCycleRecovery, SinePatternRecovery

__all__ = (
    "SineCycleResonance",
    "SineModeGain",
    "SineRecoveryResonance",
    "SineMediatedResponse",
    "SinePairPulseAssessment",
    "SinePathMemoryAssessment",
    "SineRecurrenceAssessment",
    "SineBridgeChannelAssessment",
    "BridgeStorageFamilyAssessment",
    "SineC5LeafSaddle",
    "assess_sine_cycle_resonance",
    "certify_sine_recovery_resonance",
    "assess_sine_mediated_response",
    "assess_sine_pair_pulse",
    "assess_sine_path_memory",
    "assess_sine_recurrence",
    "assess_sine_bridge_channels",
    "assess_bridge_storage_family",
    "assess_sine_c5_leaf_saddle",
)


@dataclass(frozen=True)
class SineModeGain:
    """Harmonic gains for one supplied angular frequency in the model clock.

    Input is additive modal form rate, not pressure or capacity modulation.
    Outputs are form amplitude, phase amplitude and the work-conjugate
    output lambda_m times form amplitude. No harmonic input is executed.
    """

    mode: SineCycleResonance
    angular_frequency: Q
    denominator_squared_bounds: I
    form_gain_bounds: I | None
    phase_gain_bounds: I | None
    work_gain_bounds: I | None
    unavailable_reasons: tuple[str, ...]
    status: str
    arithmetic_method: str = INTERVAL_METHOD

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {"schema": "tnfr.relational-sine-mode-gain.v1", "report": _project(self)}


@dataclass(frozen=True)
class SineCycleResonance:
    """Exact template and bounded linear-response quantities on a full C_n.

    The target has uniform form and phase turns k*j/n, modulo common origins.
    Each real Fourier direction of the declared nonzero mode uses unit
    Euclidean normalization. Its form-rate input is paired either with its
    form/phase amplitude or with the work-conjugate output lambda_m*X.
    Complex poles and a positive-frequency phase-output peak are separate
    conditions. The natural frequency is also the form/work peak frequency;
    it need not be the frequency of an oscillatory free response.
    """

    model: RelationalExchangeModel
    node_count: int
    mode_index: int
    winding: int
    capacity: Q
    target_edge_turn: Q
    mode_turn: Q
    laplacian_eigenvalue_bounds: I
    twist_cosine_bounds: I
    modal_rate_bounds: I
    modal_generator_bounds: tuple[tuple[I, I], tuple[I, I]]
    coupling_product_bounds: I
    feedback_bounds: I
    constitutive_ratio: Q
    damping_ratio_squared_bounds: I | None
    normalized_pole_discriminant_bounds: I
    phase_peak_margin_bounds: I
    pole_status: str
    phase_peak_status: str
    damping_coefficient_bounds: I
    stiffness_bounds: I
    natural_angular_frequency_bounds: I
    damped_angular_frequency_bounds: I | None
    phase_peak_angular_frequency_bounds: I | None
    form_peak_gain_bounds: I | None
    phase_dc_gain_bounds: I | None
    phase_peak_gain_bounds: I | None
    work_peak_gain: Q
    unavailable_reasons: tuple[str, ...]
    law: str = "normalized_sine_reciprocal_exchange"
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "declared_complete_simple_unit_cycle_not_an_observed_or_constructed_graph",
        "exact_uniform_form_and_rational_turn_acute_twist_or_consensus_target",
        "symbolic_opposite_sine_current_cancellation_at_the_target",
        "held_common_positive_capacity_and_positive_epi_weight_in_an_explicit_regular_model",
        "analytic_nonzero_Fourier_mode_no_floating_eigensolver_or_mode_scan",
        "modal_rows_differentiate_the_shared_full_sine_law_not_the_native_Arg_law",
        "input_is_additive_form_rate_in_one_unit_Euclidean_mode",
        "outputs_are_form_phase_and_work_conjugate_lambda_times_form_amplitudes",
        "form_and_work_gains_have_a_positive_frequency_peak_even_with_real_poles",
        "complex_poles_and_positive_frequency_phase_peak_require_different_inequalities",
        "natural_frequency_is_not_an_unconditional_observed_free_oscillation_rate",
        "capacity_and_modal_geometry_scale_time_but_capacity_is_not_itself_frequency",
        "constant_form_and_phase_origin_modes_are_excluded_from_this_stable_mode_report",
        "zero_in_an_outward_enclosure_does_not_replace_an_exact_positive_capacity",
        "unresolved_numerical_bounds_are_not_critical_damping_or_a_zero_response",
        "linear_hypothetical_response_not_finite_amplitude_nonlinear_resonance_or_formation",
        "no_forcing_installation_trajectory_runtime_dispatch_or_physical_clock_identification",
    )

    @property
    def form_peak_angular_frequency_bounds(self):
        return self.natural_angular_frequency_bounds

    @property
    def work_peak_angular_frequency_bounds(self):
        return self.natural_angular_frequency_bounds

    def gain(self, angular_frequency) -> SineModeGain:
        """Rebuild the declared mode, then enclose its three harmonic gains."""
        if self.law != "normalized_sine_reciprocal_exchange":
            raise ValueError("mode must declare the complete normalized sine law")
        mode = assess_sine_cycle_resonance(
            model=self.model,
            node_count=self.node_count,
            mode_index=self.mode_index,
            capacity=self.capacity,
            winding=self.winding,
        )
        frequency = exact_or_represented_real(angular_frequency, "angular_frequency")
        if frequency < 0:
            raise ValueError("angular_frequency must be nonnegative")
        denominator = (mode.stiffness_bounds - frequency**2) ** 2 + (
            mode.damping_coefficient_bounds * frequency
        ) ** 2
        reasons = []
        if denominator.lo <= 0:
            reasons.append("positive_transfer_denominator_not_resolved")
            form = work = I(0) if not frequency else None
            phase = None
        else:
            form = sqrt(I(frequency**2) / denominator)
            phase = sqrt(mode.modal_generator_bounds[1][0] ** 2 / denominator)
            work = mode.laplacian_eigenvalue_bounds * form
        return SineModeGain(
            mode=mode,
            angular_frequency=frequency,
            denominator_squared_bounds=denominator,
            form_gain_bounds=form,
            phase_gain_bounds=phase,
            work_gain_bounds=work,
            unavailable_reasons=tuple(reasons),
            status="unavailable" if reasons else "available",
        )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {
            "schema": "tnfr.relational-sine-cycle-resonance.v1",
            "report": _project(self),
        }


def _exact_integer(value, label):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{label} must be an exact nonboolean integer")
    return int(value)


def _positive_ratio(numerator, denominator, reason, unavailable):
    if denominator.lo <= 0:
        unavailable.append(reason)
        return None
    return numerator / denominator


def assess_sine_cycle_resonance(
    *, model, node_count, mode_index, capacity, winding=0
) -> SineCycleResonance:
    """Differentiate the complete sine model and identify each response port.

    For lambda=2-2*cos(2*pi*m/n), c=cos(2*pi*k/n), r=nu*lambda/2,
    a=w/pi and b=w/(beta*pi), the modal matrix is
    [[-e*r,-a*c*r],[b*r,0]]. Additive form-rate input gives transfer
    X/u=s/D(s), Theta/u=b*r/D(s), with D=s^2+e*r*s+a*b*c*r^2.
    Work output is lambda*X; its peak gain is exactly 2/(e*nu).

    Integer cycle declarations avoid graph allocation, including for large n.
    Counts, winding and mode are exact nonboolean integers; positive capacity
    preserves rational inputs through the shared scalar boundary. Bounds use
    mathematical pi and trigonometry, not eigenvalues of a rounded generator.
    """
    e, w, beta = _sine_model_coefficients(model)
    if e <= 0:
        raise ValueError("positive epi_weight is required for stable harmonic response")
    count = _exact_integer(node_count, "node_count")
    mode = _exact_integer(mode_index, "mode_index")
    winding = _exact_integer(winding, "winding")
    if count < 3:
        raise ValueError("node_count must be at least three")
    if not 1 <= mode < count:
        raise ValueError("mode_index must be a nonzero Fourier mode below node_count")
    if 4 * abs(winding) >= count:
        raise ValueError("winding must satisfy 4*abs(winding)<node_count")
    capacity = exact_or_represented_real(capacity, "capacity")
    if capacity <= 0:
        raise ValueError("capacity must be strictly positive")

    pi = pi_interval()
    raw_eigenvalue = 2 - 2 * cos(Q(2 * min(mode, count - mode), count) * pi)
    raw_cosine = cos(Q(2 * abs(winding), count) * pi)
    # The exact integer geometry proves these ranges independently of the
    # numerical enclosure. Intersect known ranges; do not infer strict signs
    # from an interval whose lower endpoint remains zero.
    eigenvalue = I(max(Q(0), raw_eigenvalue.lo), min(Q(4), raw_eigenvalue.hi))
    cosine = I(max(Q(0), raw_cosine.lo), min(Q(1), raw_cosine.hi))
    rate = capacity * eigenvalue / 2
    product = I(w**2 / beta) / pi**2
    feedback = product * cosine
    x_direction = _sine_rates(model, (2,), (eigenvalue,), (capacity,), (I(0),))
    phase_direction = _sine_rates(
        model, (2,), (I(0),), (capacity,), (-cosine * eigenvalue,)
    )
    matrix = (
        (x_direction["form_rates"][0], phase_direction["form_rates"][0]),
        (x_direction["phase_rates"][0], phase_direction["phase_rates"][0]),
    )
    damping = -matrix[0][0]
    stiffness = feedback * rate**2
    discriminant = I(e**2) - 4 * feedback
    phase_margin = 2 * feedback - e**2
    natural = rate * sqrt(feedback)
    unavailable = []
    if discriminant.hi < 0:
        pole_status = "complex_conjugate_poles"
        damped = rate * sqrt(-discriminant / 4)
    elif discriminant.lo > 0:
        pole_status, damped = "distinct_real_poles", None
    else:
        pole_status, damped = "unresolved", None
        unavailable.append("pole_classification_unresolved")
    damping_ratio = _positive_ratio(
        I(e**2), 4 * feedback, "damping_ratio_denominator_not_resolved", unavailable
    )
    form_peak = _positive_ratio(
        I(1), damping, "form_peak_gain_denominator_not_resolved", unavailable
    )
    phase_dc = _positive_ratio(
        matrix[1][0], stiffness, "phase_dc_gain_denominator_not_resolved", unavailable
    )
    if phase_margin.lo > 0:
        phase_status = "positive_frequency_peak"
        phase_frequency = rate * sqrt(phase_margin / 2)
        phase_peak = _positive_ratio(
            matrix[1][0],
            damping * rate * sqrt(-discriminant / 4),
            "phase_peak_gain_denominator_not_resolved",
            unavailable,
        )
    elif phase_margin.hi <= 0:
        phase_status, phase_frequency, phase_peak = "dc_maximum", I(0), phase_dc
    else:
        phase_status, phase_frequency, phase_peak = "unresolved", None, None
        unavailable.append("phase_peak_classification_unresolved")
    return SineCycleResonance(
        model=model,
        node_count=count,
        mode_index=mode,
        winding=winding,
        capacity=capacity,
        target_edge_turn=Q(winding, count),
        mode_turn=Q(mode, count),
        laplacian_eigenvalue_bounds=eigenvalue,
        twist_cosine_bounds=cosine,
        modal_rate_bounds=rate,
        modal_generator_bounds=matrix,
        coupling_product_bounds=product,
        feedback_bounds=feedback,
        constitutive_ratio=beta * (e / w) ** 2,
        damping_ratio_squared_bounds=damping_ratio,
        normalized_pole_discriminant_bounds=discriminant,
        phase_peak_margin_bounds=phase_margin,
        pole_status=pole_status,
        phase_peak_status=phase_status,
        damping_coefficient_bounds=damping,
        stiffness_bounds=stiffness,
        natural_angular_frequency_bounds=natural,
        damped_angular_frequency_bounds=damped,
        phase_peak_angular_frequency_bounds=phase_frequency,
        form_peak_gain_bounds=form_peak,
        phase_dc_gain_bounds=phase_dc,
        phase_peak_gain_bounds=phase_peak,
        work_peak_gain=2 / (e * capacity),
        unavailable_reasons=tuple(unavailable),
    )


@dataclass(frozen=True)
class SineRecoveryResonance:
    """A collocated work-port resonance theorem on complete recovered support.

    The hypothetical form-rate input is f*u with f=K*(e_i-e_j). The measured
    work-conjugate output is y=f.T*L*delta_x, so supplied power is u*y. For
    each admitted capacity member the target linearization has a positive
    finite-frequency maximum and the displayed global gain ceiling. No peak
    frequency or gain curve is computed. Unknown capacities retain a family
    of systems and, when a port capacity is unknown, a family of probe pairs.
    """

    recovery: SineCycleRecovery | SinePatternRecovery
    port: tuple[Any, Any]
    port_indices: tuple[int, int]
    degrees: tuple[int, ...]
    exact_mobility: tuple[Q | None, ...]
    mobility_bounds: tuple[I, ...]
    exact_input_coefficients: tuple[Q | None, ...]
    input_coefficients_bounds: tuple[I, ...]
    exact_work_output_coefficients: tuple[Q | None, ...]
    work_output_coefficients_bounds: tuple[I, ...]
    work_gain_upper_bound: Q
    capacity_family: bool
    input_output_family: bool
    positive_frequency_peak_certified: bool = True
    peak_angular_frequency_bounds: None = None
    law: str = "normalized_sine_reciprocal_exchange"
    scope: tuple[str, ...] = (
        "shared_recovery_certificate_rebuilt_and_compared_before_consumption",
        "complete_supplied_connected_unit_support_and_exact_acute_critical_target",
        "positive_held_capacity_and_epi_weight_on_each_admitted_capacity_member",
        "noncommuting_form_and_phase_Hessians_are_retained_on_the_full_mean_quotient",
        "input_f_is_K_times_a_declared_two_node_difference_no_edge_is_added",
        "work_output_is_f_transpose_L_delta_x_and_supplied_power_is_input_times_output",
        "weighted_mean_compatible_input_removes_the_two_common_origin_modes",
        "strictly_positive_real_harmonic_response_and_zero_low_and_high_frequency_limits",
        "positive_finite_frequency_maximum_exists_but_no_peak_frequency_is_computed",
        "global_gain_ceiling_is_the_port_mobility_sum_divided_by_epi_weight",
        "uncertain_port_capacity_defines_a_family_of_capacity_weighted_probe_pairs",
        "target_linear_response_not_finite_amplitude_resonance_or_forced_capture",
        "rebuilding_is_not_authentication_of_observations_or_proof_of_a_reserved_trajectory",
        "no_forcing_installation_frequency_scan_eigensolver_or_runtime_law_change",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        _validate_comparison_labels(self.recovery)
        return {
            "schema": "tnfr.relational-sine-recovery-resonance.v1",
            "report": _project(self),
        }


def _admit_sine_recovery(recovery):
    """Rebuild the complete supplied certificate through its shared owner."""
    from .relational_sine_recovery import (
        SineCycleRecovery,
        SinePatternRecovery,
        certify_sine_cycle_recovery,
        certify_sine_pattern_recovery,
    )

    if isinstance(recovery, SineCycleRecovery):
        rebuilt = certify_sine_cycle_recovery(
            recovery.source,
            cycle=recovery.cycle,
            winding=recovery.winding,
            radius=recovery.radius,
            phase_turns=recovery.phase_turns,
        )
    elif isinstance(recovery, SinePatternRecovery):
        rebuilt = certify_sine_pattern_recovery(
            recovery.source,
            target_phase_turns=recovery.target_phase_turns,
            radius=recovery.radius,
            phase_turns=recovery.phase_turns,
        )
    else:
        raise TypeError(
            "a shared sine cycle or pattern recovery certificate is required"
        )
    if rebuilt != recovery:
        raise ValueError("recovery does not match its rebuilt shared certificate")
    if not rebuilt.admitted:
        raise ValueError("an admitted sine recovery certificate is required")
    return rebuilt


def _recovery_support(recovery):
    positions = {node: i for i, node in enumerate(recovery.nodes)}
    neighbors = [[] for _ in recovery.nodes]
    for first, second in recovery.edges:
        i, j = positions[first], positions[second]
        neighbors[i].append(j)
        neighbors[j].append(i)
    return positions, tuple(tuple(row) for row in neighbors)


def certify_sine_recovery_resonance(recovery, *, port) -> SineRecoveryResonance:
    """Reuse an admitted full-support target for a declared collocated probe.

    Rebuild through the recovery owner, applying its source, target, capacity,
    uncertainty and radius hypotheses rather than trusting a status field.
    The supplied report must match the rebuilt result and be admitted. The
    target's unforced recovery does not promise a nonlinear response to an
    installed probe: the result concerns its infinitesimal transfer function.
    """
    rebuilt = _admit_sine_recovery(recovery)
    port = _ordered(port, "port", limit=3)
    if len(port) != 2:
        raise ValueError("port must contain exactly two distinct supplied nodes")
    positions, neighbors = _recovery_support(rebuilt)
    try:
        left, right = (positions[node] for node in port)
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "port nodes must belong to the complete supplied support"
        ) from exc
    if left == right:
        raise ValueError("port nodes must be distinct")
    degrees = tuple(map(len, neighbors))
    exact = tuple(
        nu / degree if nu is not None else None
        for nu, degree in zip(rebuilt.exact_held_capacity, degrees)
    )
    mobility = tuple(
        I(value) if value is not None else bounds / degree
        for value, bounds, degree in zip(exact, rebuilt.capacity_bounds, degrees)
    )
    coefficients = [I(0)] * len(degrees)
    exact_coefficients = [Q(0)] * len(degrees)
    coefficients[left], coefficients[right] = mobility[left], -mobility[right]
    exact_coefficients[left] = exact[left]
    exact_coefficients[right] = -exact[right] if exact[right] is not None else None
    output = tuple(
        degree * coefficients[i] - sum((coefficients[j] for j in row), I(0))
        for i, (degree, row) in enumerate(zip(degrees, neighbors))
    )
    exact_output = tuple(
        (
            degree * exact_coefficients[i]
            - sum((exact_coefficients[j] for j in row), Q(0))
            if all(exact_coefficients[j] is not None for j in (i, *row))
            else None
        )
        for i, (degree, row) in enumerate(zip(degrees, neighbors))
    )
    upper = tuple(
        value if value is not None else bounds.hi
        for value, bounds in zip(exact, mobility)
    )
    e = Q(rebuilt.reference_model.effective_weights[0])
    return SineRecoveryResonance(
        recovery=rebuilt,
        port=port,
        port_indices=(left, right),
        degrees=degrees,
        exact_mobility=exact,
        mobility_bounds=mobility,
        exact_input_coefficients=tuple(exact_coefficients),
        input_coefficients_bounds=tuple(coefficients),
        exact_work_output_coefficients=exact_output,
        work_output_coefficients_bounds=output,
        work_gain_upper_bound=(upper[left] + upper[right]) / e,
        capacity_family=any(value is None for value in exact),
        input_output_family=exact[left] is None or exact[right] is None,
    )


@dataclass(frozen=True)
class SineMediatedResponse:
    """Remote tangent response of two complete C5 patterns and an intermediary.

    The donor direction is f=K*p, with p equal to one at its contact port and
    minus one quarter at its four other nodes. The receiver contrast has the
    same relative weights on the other cycle. Markov entries are C*J**j*(f,0)
    for j=0,1,2, equivalently initial derivatives of the unforced tangent
    response. Exact bridge geometry gives zero remote DC response and a
    strictly negative phase entry at order two. Consequently its phase gain
    has a positive-frequency maximum and its unforced phase response changes
    sign. These statements give neither a frequency nor a reversal time.
    """

    recovery: SineCycleRecovery | SinePatternRecovery
    donor_cycle: tuple[Any, ...]
    receiver_cycle: tuple[Any, ...]
    mediator: Any
    donor_indices: tuple[int, ...]
    receiver_indices: tuple[int, ...]
    mediator_index: int
    degrees: tuple[int, ...]
    mobility: tuple[Q, ...]
    donor_contrast: tuple[Q, ...]
    input_direction: tuple[Q, ...]
    receiver_contrast: tuple[Q, ...]
    form_state_derivatives_bounds: tuple[tuple[I, ...], ...]
    phase_state_derivatives_bounds: tuple[tuple[I, ...], ...]
    form_markov_bounds: tuple[I, ...]
    phase_markov_bounds: tuple[I, ...]
    port_mobility_product: Q
    phase_order_two_pi_numerator: Q
    form_order_two_bounds: I
    mediator_generator_bounds: tuple[tuple[I, ...], ...]
    donor_to_mediator_bounds: tuple[tuple[I, ...], ...]
    mediator_to_receiver_bounds: tuple[tuple[I, ...], ...]
    memory_kernel_at_zero_bounds: tuple[tuple[I, ...], ...]
    donor_odd_tangent_sector_silent: bool
    receiver_odd_tangent_readout_silent: bool
    form_dc_gain: Q = Q(0)
    phase_dc_gain: Q = Q(0)
    phase_order_two_sign: int = -1
    positive_frequency_phase_peak_certified: bool = True
    phase_impulse_sign_reversal_certified: bool = True
    peak_angular_frequency_bounds: None = None
    sign_reversal_time_bounds: None = None
    law: str = "normalized_sine_reciprocal_exchange"
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "shared_recovery_rebuilt_and_compared_before_target_response",
        "exactly_two_complete_disjoint_C5_cycles_and_one_live_intermediary",
        "each_cycle_first_node_is_its_only_intermediary_contact",
        "all_held_capacities_exact_and_positive_no_hidden_coordinate_removed",
        "full_sine_tangent_at_exact_acute_critical_target_not_captured_nominal_state",
        "input_is_K_times_zero_sum_donor_port_relative_contrast",
        "receiver_phase_and_form_readouts_are_port_relative_to_other_four_nodes",
        "Markov_entries_are_derivatives_not_Taylor_coefficients_divided_by_factorials",
        "same_entries_describe_initial_form_perturbation_and_hypothetical_rate_input",
        "zero_remote_DC_response_follows_from_zero_receiver_bridge_flux",
        "nonzero_phase_response_has_zero_low_and_high_frequency_limits",
        "strict_negative_phase_onset_and_zero_time_integral_force_later_positive_response",
        "memory_blocks_keep_form_then_phase_coordinates_at_the_three_contact_nodes",
        "cross_memory_kernel_is_J_Rh_exp_J_hh_t_J_hD_only_its_zero_lag_is_evaluated",
        "hidden_initial_state_and_other_visible_history_are_separate_memory_terms",
        "silence_flags_are_exact_sufficient_tangent_reflection_certificates",
        "false_silence_flag_means_this_symmetry_proof_is_unavailable_not_transmission",
        "no_peak_frequency_reversal_time_amplitude_or_finite_nonlinear_response_bound",
        "no_forcing_installation_trajectory_support_event_or_pattern_formation",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        _validate_comparison_labels(self.recovery)
        return {
            "schema": "tnfr.relational-sine-mediated-response.v1",
            "report": _project(self),
        }


def _principal_turn_distance(left, right):
    return abs((right - left + Q(1, 2)) % 1 - Q(1, 2))


def _cycle_reflection_preserved(cycle, neighbors, capacity, target):
    """Check a declared reflection against K and the exact cosine edge colors."""
    permutation = list(range(len(neighbors)))
    for position, index in enumerate(cycle):
        permutation[index] = cycle[-position % len(cycle)]
    for i, row in enumerate(neighbors):
        reflected = permutation[i]
        if capacity[i] != capacity[reflected]:
            return False
        if {permutation[j] for j in row} != set(neighbors[reflected]):
            return False
        for j in row:
            if _principal_turn_distance(target[i], target[j]) != (
                _principal_turn_distance(target[reflected], target[permutation[j]])
            ):
                return False
    return True


def _sine_target_tangent_action(model, neighbors, capacity, edge_cosines, x, theta):
    """Apply the shared sine rates to exact-target directional gradients."""
    gradient = tuple(
        sum((x[i] - x[j] for j in row), I(0)) for i, row in enumerate(neighbors)
    )
    currents = tuple(
        sum(
            (weight * (theta[j] - theta[i]) for j, weight in zip(row, edge_cosines[i])),
            I(0),
        )
        for i, row in enumerate(neighbors)
    )
    rates = _sine_rates(model, tuple(map(len, neighbors)), gradient, capacity, currents)
    return tuple(rates["form_rates"]), tuple(rates["phase_rates"])


@dataclass(frozen=True)
class SineBridgeChannelAssessment:
    """Energy-normalized bridge response at a declared conservative target.

    The two outputs are the bridge form contrast and sqrt(beta) times its
    phase perturbation. Their transpose preparation is admissible in the
    nodal edge cut spaces because the aligned edge is a bridge. The
    returned jets use tau=w*t/pi, zero hidden preparation and the full visible
    unit ball. They are not jets at the captured source's observed state.
    """

    source: SineExchangeComparison
    target_phase_turns: tuple[Q, ...]
    target_geometry: PhaseCycleState
    bridge: tuple[Any, Any]
    bridge_edge_index: int
    mobility: tuple[Q, ...]
    storage_scale: Q
    clock_rate_pi_numerator: Q
    nodal_bridge_lift: tuple[Q, ...]
    bridge_incidence_row: tuple[Q, ...]
    edge_cosine_bounds: tuple[I, ...]
    first_jet_bounds: tuple[tuple[I, I], tuple[I, I]]
    second_jet_diagonal_bounds: tuple[I, I]
    channel_difference_bounds: I
    channel_difference_is_positive: bool
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "declared_acute_exact_critical_target_not_the_observed_source_state",
        "conservative_sine_law_full_unit_support_positive_held_capacity",
        "aligned_cut_edge_has_an_admissible_orthonormal_energy_preparation",
        "same_bridge_form_and_phase_outputs_with_transpose_preparation",
        "tau_equals_w_t_over_pi_and_both_output_channels_are_energy_normalized",
        "weighted_common_origins_removed_in_the_displayed_nodal_lift",
        "zero_hidden_preparation_for_homogeneous_jets_not_all_hidden_states",
        "nonzero_hidden_state_retains_a_separate_observed_source",
        "exact_positive_difference_iff_an_incident_nonbridge_target_gap_is_nonzero",
        "interval_zero_lower_endpoint_does_not_override_exact_structural_positivity",
        "no_matrix_square_root_numerical_approximation_is_treated_as_exact",
        "initial_jet_is_skew_geometric_distinction_does_not_derive_loss",
        "no_trajectory_support_birth_attraction_or_physical_identification",
    )

    def to_dict(self):
        """Project the declared target and jets without authenticating the source."""
        from ..sdk.relational_reports import _project

        _validate_comparison_labels(self.source)
        return {
            "schema": "tnfr.relational-sine-bridge-channels.v1",
            "report": _project(self),
        }


def assess_sine_bridge_channels(
    source, *, target_phase_turns, bridge
) -> SineBridgeChannelAssessment:
    """Distinguish channels using an aligned bridge in an actual critical target.

    With incidence E, mobility K, cosine weights c and A=E^T K E, the edge
    energy chart has cross-block D=A diag(sqrt(c))/sqrt(beta). The nodal
    cut-space constraints are retained: only an aligned cut edge admits the
    declared pair of coordinate-unit preparations. For its index b, the
    homogeneous visible response has T'(0)=A_bb R/sqrt(beta) and
    T''(0)=diag(-sum(c_e*A_be^2),-sum(A_be^2))/beta. The positive difference
    sum((1-c_e)*A_be^2)/beta is evaluated without matrix square roots.

    The exact rational target proves acuteness and sine criticality through
    existing geometry admission. A normalized source association is retained,
    but cached rates, storage and verdicts never supply this calculation.
    """
    from ._sine_admission import _admit_sine_source
    from .relational_sine_recovery import _target_admission

    if not isinstance(source, SineExchangeComparison):
        raise TypeError("bridge channels require a SineExchangeComparison")
    admitted, _ = _admit_sine_source(source)
    loss, exchange, beta = _sine_model_coefficients(admitted.reference_model)
    if loss != 0 or exchange <= 0 or beta <= 0:
        raise ValueError(
            "bridge channels require zero loss and positive exchange/storage"
        )
    if any(value <= 0 for value in admitted.capacity):
        raise ValueError("bridge channels require strictly positive held capacities")
    target, _, _, _, _, fields = _target_admission(
        admitted, cycle=None, winding=None, target_phase_turns=target_phase_turns
    )
    state = fields["target_geometry"]
    geometry = state.geometry
    turns = state.edge_turns
    if any(abs(turn) >= Q(1, 4) for turn in turns):
        raise ValueError("the declared critical target must be strictly acute")
    selected = _ordered(bridge, "bridge", limit=3)
    positions = {node: i for i, node in enumerate(admitted.nodes)}
    try:
        indices = tuple(sorted(positions[node] for node in selected))
    except (KeyError, TypeError) as exc:
        raise ValueError("bridge must contain two nodes from the source") from exc
    if len(indices) != 2 or indices not in geometry.edges:
        raise ValueError("bridge must be an edge of the full source support")
    edge_index = geometry.edges.index(indices)
    if edge_index not in geometry.bridge_edge_indices:
        raise ValueError("the declared edge must be a cut edge of the full support")
    if turns[edge_index] != 0:
        raise ValueError("the declared bridge must have aligned target phases")
    mobility = tuple(nu / d for nu, d in zip(admitted.capacity, admitted.degrees))
    incidence = geometry.incidence
    row = tuple(
        sum(
            (
                incidence[i][edge_index] * mobility[i] * incidence[i][j]
                for i in range(len(admitted.nodes))
            ),
            Q(0),
        )
        for j in range(len(geometry.edges))
    )
    pi = pi_interval()
    cosines = tuple(I(1) if turn == 0 else cos((2 * turn) * pi) for turn in turns)
    # The exact acute-turn premise bounds every cosine by [0, 1]. Keep
    # unresolved strict positivity separate from its dyadic enclosure.
    cosines = tuple(I(max(Q(0), value.lo), min(Q(1), value.hi)) for value in cosines)
    squares = tuple(value**2 / beta for value in row)
    phase_second = -sum(squares, Q(0))
    form_lower = -sum((s * c.hi for s, c in zip(squares, cosines)), Q(0))
    form_upper = -sum((s * c.lo for s, c in zip(squares, cosines)), Q(0))
    difference = I(
        sum((s * (1 - c.hi) for s, c in zip(squares, cosines)), Q(0)),
        sum((s * (1 - c.lo) for s, c in zip(squares, cosines)), Q(0)),
    )
    # Divide exact rationals before interval materialization. In particular a
    # positive sub-grid beta never becomes a zero interval denominator.
    angular_rate = sqrt(I(row[edge_index] ** 2 / beta))
    neighbors = [[] for _ in admitted.nodes]
    for j, (left, right) in enumerate(geometry.edges):
        if j != edge_index:
            neighbors[left].append(right)
            neighbors[right].append(left)
    reached, pending = {indices[0]}, [indices[0]]
    while pending:
        for node in neighbors[pending.pop()]:
            if node not in reached:
                reached.add(node)
                pending.append(node)
    weights = tuple(1 / value for value in mobility)
    total = sum(weights, Q(0))
    left_mass = sum((weights[i] for i in reached), Q(0))
    lift = tuple(
        -(total - left_mass) / total if i in reached else left_mass / total
        for i in range(len(admitted.nodes))
    )
    return SineBridgeChannelAssessment(
        source=admitted,
        target_phase_turns=target,
        target_geometry=state,
        bridge=tuple(admitted.nodes[i] for i in indices),
        bridge_edge_index=edge_index,
        mobility=mobility,
        storage_scale=beta,
        clock_rate_pi_numerator=exchange,
        nodal_bridge_lift=lift,
        bridge_incidence_row=row,
        edge_cosine_bounds=cosines,
        first_jet_bounds=((I(0), -angular_rate), (angular_rate, I(0))),
        second_jet_diagonal_bounds=(I(form_lower, form_upper), I(phase_second)),
        channel_difference_bounds=difference,
        channel_difference_is_positive=any(
            coefficient != 0 and turn != 0 for coefficient, turn in zip(row, turns)
        ),
    )


def _admit_two_cycle_bridge(source, *, left_cycle, right_cycle, target_phase_turns):
    """Rebuild the exact unit two-C6 support, target and primitive association."""
    if not isinstance(source, SineExchangeComparison):
        raise TypeError("two-cycle bridge readers require a SineExchangeComparison")
    left = _ordered(left_cycle, "left_cycle", limit=7)
    right = _ordered(right_cycle, "right_cycle", limit=7)
    if len(left) != 6 or len(right) != 6:
        raise ValueError("left_cycle and right_cycle must each contain six nodes")
    channels = assess_sine_bridge_channels(
        source, target_phase_turns=target_phase_turns, bridge=(left[0], right[0])
    )
    admitted = channels.source
    positions = {node: i for i, node in enumerate(admitted.nodes)}
    try:
        cycles = tuple(
            tuple(positions[node] for node in cycle) for cycle in (left, right)
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("cycle nodes must belong to the admitted source") from exc
    if len(positions) != 12 or len(set(cycles[0] + cycles[1])) != 12:
        raise ValueError("two disjoint six-node cycles must cover the full source")
    expected = {
        tuple(sorted((cycle[i], cycle[(i + 1) % 6])))
        for cycle in cycles
        for i in range(6)
    }
    bridge = tuple(sorted((cycles[0][0], cycles[1][0])))
    expected.add(bridge)
    geometry = channels.target_geometry.geometry
    if set(geometry.edges) != expected:
        raise ValueError("full support must be exactly the two cycles and their bridge")
    if (
        channels.storage_scale != 1
        or channels.clock_rate_pi_numerator != 1
        or any(value != 1 for value in admitted.capacity)
    ):
        raise ValueError(
            "two-cycle bridge requires unit capacity, storage and exchange"
        )
    if any(
        abs(turn) != Q(1, 6)
        for edge, turn in zip(geometry.edges, channels.target_geometry.edge_turns)
        if edge != bridge
    ):
        raise ValueError("every internal target edge must have absolute turn one sixth")
    return channels, left, right, cycles


def _conservative_full_tangent(edges, mobility, edge_curvatures):
    """Assemble exact conservative nodal matrices from admitted edge curvatures."""
    size = len(mobility)
    if len(edges) != len(edge_curvatures):
        raise ValueError("every admitted edge requires one phase curvature")
    laplacian = [[Q(0) for _ in range(size)] for _ in range(size)]
    hessian = [[Q(0) for _ in range(size)] for _ in range(size)]
    for (i, j), curvature in zip(edges, edge_curvatures):
        for matrix, coefficient in ((laplacian, Q(1)), (hessian, curvature)):
            matrix[i][i] += coefficient
            matrix[j][j] += coefficient
            matrix[i][j] -= coefficient
            matrix[j][i] -= coefficient
    laplacian = tuple(tuple(row) for row in laplacian)
    hessian = tuple(tuple(row) for row in hessian)
    zero = (Q(0),) * size
    full = tuple(
        zero + tuple(-mobility[i] * value for value in row)
        for i, row in enumerate(hessian)
    ) + tuple(
        tuple(mobility[i] * value for value in row) + zero
        for i, row in enumerate(laplacian)
    )
    metric = tuple(row + zero for row in laplacian) + tuple(
        zero + row for row in hessian
    )
    return laplacian, hessian, full, metric


def _bridge_full_tangent(channels, internal_phase_curvature):
    """Preserve the bridge adapter over the common exact nodal assembly."""
    edges = channels.target_geometry.geometry.edges
    curvatures = tuple(
        Q(1) if index == channels.bridge_edge_index else internal_phase_curvature
        for index in range(len(edges))
    )
    return _conservative_full_tangent(edges, channels.mobility, curvatures)


@dataclass(frozen=True)
class SineC5LeafSaddle:
    """Exact hypothetical saddle and its full-support conservative tangent.

    The source supplies admitted support and law, not an observed saddle.
    Odd coordinates are positive at cycle positions 0, 1 and their leaves,
    negative at their reflected partners, and zero at the two fixed nodes.
    Direction intervals enclose one correlated algebraic eigenvector; arbitrary
    independent values from those intervals need not be eigenvectors.
    """

    source: SineExchangeComparison
    cycle: tuple[Any, ...]
    cycle_indices: tuple[int, ...]
    contacts: tuple[tuple[Any, Any], ...]
    target_epi: tuple[Q, ...]
    target_phase_turns: tuple[Q, ...]
    target_geometry: CircularPhaseState
    target_storage: Q
    reflection_indices: tuple[int, ...]
    odd_reconstruction: tuple[tuple[Q, ...], ...]
    odd_restriction: tuple[tuple[Q, ...], ...]
    edge_phase_curvatures: tuple[Q, ...]
    form_laplacian: tuple[tuple[Q, ...], ...]
    phase_hessian: tuple[tuple[Q, ...], ...]
    full_tangent_generator: tuple[tuple[Q, ...], ...]
    full_energy_metric: tuple[tuple[Q, ...], ...]
    odd_phase_from_form: tuple[tuple[Q, ...], ...]
    odd_negative_form_from_phase: tuple[tuple[Q, ...], ...]
    odd_phase_acceleration: tuple[tuple[Q, ...], ...]
    odd_phase_hessian: tuple[tuple[Q, ...], ...]
    odd_leaf_schur_complement: tuple[tuple[Q, ...], ...]
    odd_characteristic_coefficients: tuple[Q, ...]
    positive_root_bracket: tuple[Q, Q]
    positive_root_endpoint_values: tuple[Q, Q]
    positive_root_count: int
    growth_rate_bounds: I
    unstable_phase_numerator_coefficients: tuple[tuple[Q, ...], ...]
    unstable_phase_denominator_coefficients: tuple[Q, ...]
    unstable_direction_bounds: tuple[I, ...]
    unstable_cycle_gap_direction_bounds: tuple[I, ...]
    odd_phase_hessian_inertia: tuple[int, int, int]
    even_phase_hessian_inertia: tuple[int, int, int]
    relative_phase_hessian_inertia: tuple[int, int, int]
    relative_hyperbolic_pairs: int
    relative_oscillatory_pairs: int
    neutral_origin_modes: int
    status: str = "certified"
    clock: str = "tau=t/pi"
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "source_primitives_anchor_support_and_law_not_an_observed_saddle",
        "fixed_complete_unit_C5_with_five_private_leaves_unit_capacity_e0_w1_beta1",
        "declared_exact_turn_saddle_has_zero_form_and_aligned_private_contacts",
        "full_nodal_matrices_and_signed_involution_are_rebuilt_before_reduction",
        "odd_basis_uses_declared_cycle_positions_independently_of_source_node_order",
        "exact_characteristic_polynomial_and_Descartes_sign_count_no_float_eigensolver",
        "positive_root_bracket_has_exact_rational_endpoint_signs",
        "direction_bounds_share_one_algebraic_root_not_independent_state_errors",
        "hyperbolic_pair_is_a_local_tangent_result_not_global_sector_crossing",
        "no_trajectory_formation_retention_preparation_selection_or_physical_identification",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project, _validate_label_groups

        _validate_comparison_labels(self.source)
        _validate_label_groups(
            self.cycle, *self.contacts, self.target_geometry.geometry.nodes
        )
        return {"schema": "tnfr.sine-c5-leaf-saddle.v1", "report": _project(self)}


def assess_sine_c5_leaf_saddle(source, *, cycle) -> SineC5LeafSaddle:
    """Derive the full saddle tangent and isolate its unique growing direction.

    The declared receiver/leaf phases are (i-2)/6 exact turns. Four receiver
    curvatures equal 1/2, the closing one -1/2, and contact curvatures one.
    On the odd reflection sector, theta'=A*x and x'=-B*theta. The exact
    phase acceleration matrix -A*B has one positive algebraic eigenvalue.
    No supplied source state is projected to, or asserted to occupy, this target.
    """
    from ..mathematics.krylov import exact_rank
    from .phase_cycle_geometry import (
        _c5_sine_templates,
        _derive,
        reconstruct_circular_phase_state,
    )
    from .relational_sine_partition import _private_leaf_support
    from .relational_sine_symmetry import assess_sine_involution_reduction

    source, edges, indices, pairs, contacts = _private_leaf_support(source, cycle)
    size = len(source.nodes)
    target = [Q(0)] * size
    permutation = list(range(size))
    for position, (node, leaf) in enumerate(contacts):
        target[node] = target[leaf] = Q(position - 2, 6)
        permutation[node] = contacts[4 - position][0]
        permutation[leaf] = contacts[4 - position][1]
    target = tuple(target)
    geometry = _derive(source.nodes, edges)

    def principal(turn):
        return (turn + Q(1, 2)) % 1 - Q(1, 2)

    turns = tuple(principal(target[j] - target[i]) for i, j in geometry.edges)
    cycle_turns = tuple(principal(target[j] - target[i]) for i, j in pairs)
    if cycle_turns not in _c5_sine_templates()[0]:
        raise ArithmeticError(
            "the declared saddle must satisfy exact C5 branch closure"
        )
    state = reconstruct_circular_phase_state(geometry, edge_turns=turns)
    if any(state.symbolic_sine_coefficients):
        raise ArithmeticError("the declared target's nodal sine currents must cancel")
    curvature_by_turn = {Q(0): Q(1), Q(1, 6): Q(1, 2), Q(1, 3): -Q(1, 2)}
    curvatures = tuple(curvature_by_turn[abs(turn)] for turn in turns)
    mobility = tuple(Q(1, degree) for degree in source.degrees)
    laplacian, hessian, full, metric = _conservative_full_tangent(
        geometry.edges, mobility, curvatures
    )

    odd = assess_sine_involution_reduction(
        source, permutation_indices=tuple(permutation), sign=-1
    )
    even = assess_sine_involution_reduction(
        source, permutation_indices=tuple(permutation), sign=1
    )
    if not odd.family_invariance_certified or not even.family_invariance_certified:
        raise ArithmeticError(
            "the admitted support must preserve both reflection sectors"
        )
    representatives = (indices[0], indices[1], contacts[0][1], contacts[1][1])
    columns = tuple(
        next(j for j, value in enumerate(odd.reconstruction_matrix[i]) if value)
        for i in representatives
    )
    reconstruction = tuple(
        tuple(
            odd.reconstruction_matrix[i][column]
            * odd.reconstruction_matrix[representative][column]
            for representative, column in zip(representatives, columns)
        )
        for i in range(size)
    )
    restriction = tuple(
        tuple(Q(i == representative) for i in range(size))
        for representative in representatives
    )
    product = exact_matrix_product

    def transpose(matrix):
        return tuple(zip(*matrix))

    kl = tuple(tuple(mobility[i] * x for x in row) for i, row in enumerate(laplacian))
    kh = tuple(tuple(mobility[i] * x for x in row) for i, row in enumerate(hessian))
    a = product(product(restriction, kl), reconstruction)
    b = product(product(restriction, kh), reconstruction)
    if product(kl, reconstruction) != product(reconstruction, a) or product(
        kh, reconstruction
    ) != product(reconstruction, b):
        raise ArithmeticError(
            "the full tangent must preserve the derived odd coordinates"
        )
    acceleration = tuple(tuple(-x for x in row) for row in product(a, b))
    odd_hessian = product(product(transpose(reconstruction), hessian), reconstruction)
    odd_form = product(product(transpose(reconstruction), laplacian), reconstruction)
    leaf = tuple(tuple(row[2:]) for row in odd_hessian[2:])
    cross = tuple(tuple(row[2:]) for row in odd_hessian[:2])
    correction = product(product(cross, exact_matrix_inverse(leaf)), transpose(cross))
    schur = tuple(
        tuple(odd_hessian[i][j] - correction[i][j] for j in range(2)) for i in range(2)
    )
    even_hessian = product(
        product(transpose(even.reconstruction_matrix), hessian),
        even.reconstruction_matrix,
    )
    if (
        not exact_symmetric_semidefinite(odd_form, strict=True)
        or not exact_symmetric_semidefinite(leaf, strict=True)
        or schur[0][0] * schur[1][1] - schur[0][1] ** 2 >= 0
        or not exact_symmetric_semidefinite(even_hessian)
        or exact_rank(even_hessian) != 5
    ):
        raise ArithmeticError(
            "the exact Hessian congruence must have one negative direction"
        )

    # Newton's trace identities derive the quartic from the actual full-row
    # reduction; a displayed polynomial is not accepted as cached evidence.
    coefficients, traces = [Q(1)], []
    power = tuple(tuple(Q(i == j) for j in range(4)) for i in range(4))
    for order in range(1, 5):
        power = product(power, acceleration)
        traces.append(sum((power[i][i] for i in range(4)), Q(0)))
        coefficients.append(
            -sum(
                (coefficients[order - j] * traces[j - 1] for j in range(1, order + 1)),
                Q(0),
            )
            / order
        )
    coefficients = tuple(coefficients)
    signs = tuple((x > 0) - (x < 0) for x in coefficients if x)
    sign_changes = sum(x != y for x, y in zip(signs, signs[1:]))
    bracket = (Q(19709, 250000), Q(78837, 1000000))

    def polynomial(value):
        total = Q(0)
        for coefficient in coefficients:
            total = total * value + coefficient
        return total

    endpoint_values = tuple(polynomial(value) for value in bracket)
    if sign_changes != 1 or not endpoint_values[0] < 0 < endpoint_values[1]:
        raise ArithmeticError(
            "exact Descartes count and bracket must isolate one positive root"
        )
    # Ascending coefficients of D*v, with
    # D=(6*lambda+8)*(18*lambda**2+43*lambda+5)>0 at the positive root.
    # Verify the exact numerator identity modulo the derived quartic before
    # enclosing its algebraic direction; interval residual overlap is not a proof.
    numerators = tuple(
        tuple(map(Q, row))
        for row in (
            (40, 374, 402, 108),
            (8, 38, 24, 0),
            (34, 297, 126, 0),
            (5, -3, -18, 0),
        )
    )
    for i in range(4):
        residual = tuple(
            sum(
                (
                    acceleration[i][j] * (numerators[j][k] if k < 4 else Q(0))
                    for j in range(4)
                ),
                Q(0),
            )
            - (numerators[i][k - 1] if k else Q(0))
            for k in range(5)
        )
        if residual != tuple(residual[-1] * x for x in reversed(coefficients)):
            raise ArithmeticError(
                "the correlated direction must solve the exact algebraic eigenproblem"
            )
    lam = I(*bracket)
    growth = sqrt(lam)
    second = (4 * lam + 1) / (18 * lam**2 + 43 * lam + 5)
    phase_direction = (
        I(1),
        second,
        (7 - second) / (6 * lam + 8),
        (10 * second - 1) / (6 * lam + 8),
    )
    form_direction = tuple(
        -sum(
            (coefficient * value for coefficient, value in zip(row, phase_direction)),
            I(0),
        )
        / growth
        for row in b
    )

    def lift(values):
        return tuple(
            sum((coefficient * value for coefficient, value in zip(row, values)), I(0))
            for row in reconstruction
        )

    full_phase = lift(phase_direction)
    gap_directions = tuple(full_phase[j] - full_phase[i] for i, j in pairs)
    return SineC5LeafSaddle(
        source=source,
        cycle=tuple(source.nodes[i] for i in indices),
        cycle_indices=indices,
        contacts=tuple((source.nodes[i], source.nodes[j]) for i, j in contacts),
        target_epi=(Q(0),) * size,
        target_phase_turns=target,
        target_geometry=state,
        target_storage=sum((1 - curvature for curvature in curvatures), Q(0)),
        reflection_indices=tuple(permutation),
        odd_reconstruction=reconstruction,
        odd_restriction=restriction,
        edge_phase_curvatures=curvatures,
        form_laplacian=laplacian,
        phase_hessian=hessian,
        full_tangent_generator=full,
        full_energy_metric=metric,
        odd_phase_from_form=a,
        odd_negative_form_from_phase=b,
        odd_phase_acceleration=acceleration,
        odd_phase_hessian=odd_hessian,
        odd_leaf_schur_complement=schur,
        odd_characteristic_coefficients=coefficients,
        positive_root_bracket=bracket,
        positive_root_endpoint_values=endpoint_values,
        positive_root_count=sign_changes,
        growth_rate_bounds=growth,
        unstable_phase_numerator_coefficients=numerators,
        unstable_phase_denominator_coefficients=numerators[0],
        unstable_direction_bounds=lift(form_direction) + full_phase,
        unstable_cycle_gap_direction_bounds=gap_directions,
        odd_phase_hessian_inertia=(3, 1, 0),
        even_phase_hessian_inertia=(5, 0, 1),
        relative_phase_hessian_inertia=(8, 1, 0),
        relative_hyperbolic_pairs=1,
        relative_oscillatory_pairs=8,
        neutral_origin_modes=2,
    )


@dataclass(frozen=True)
class BridgeStorageFamilyAssessment:
    """Static comparison under an explicitly supplied alternative phase storage.

    The law is U(delta)=1-cos(delta)+epsilon*(2/3-cos(delta)+cos(delta)^3/3),
    with zero loss, unit held capacity and beta=w=1. Its nodal rows in
    tau=t/pi are x'=-K grad_theta sum(U), theta'=K L x. The reference sine
    comparison supplies admitted primitives and support, not an observation
    of this alternative law or a tangent perturbation.

    Full matrices use source node order: form coordinates followed by phase
    coordinates. Bridge outputs and their energy-transpose nodal preparation
    are identical for every admitted epsilon. The local storage threshold is
    a sufficient first-exit bound, conditional on preparation inside the
    stated chamber; no captured source state is checked against that premise.
    """

    source: SineExchangeComparison
    law: str
    epsilon: Q
    left_cycle: tuple[Any, ...]
    right_cycle: tuple[Any, ...]
    target_phase_turns: tuple[Q, ...]
    target_geometry: PhaseCycleState
    target_epi: tuple[Q, ...]
    bridge: tuple[Any, Any]
    mobility: tuple[Q, ...]
    edge_phase_curvatures: tuple[Q, ...]
    form_laplacian: tuple[tuple[Q, ...], ...]
    phase_hessian: tuple[tuple[Q, ...], ...]
    full_tangent_generator: tuple[tuple[Q, ...], ...]
    full_energy_metric: tuple[tuple[Q, ...], ...]
    bridge_observation_rows: tuple[tuple[Q, ...], ...]
    nodal_bridge_preparation: tuple[tuple[Q, ...], ...]
    first_jet: tuple[tuple[Q, ...], ...]
    second_jet: tuple[tuple[Q, ...], ...]
    channel_difference: Q
    criticality_status: str
    local_positivity_status: str
    target_storage: Q
    local_phase_radius_turns: Q
    local_curvature_lower_bounds: I
    local_excess_storage_threshold_bounds: I
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "supplied_sine_plus_cubic_storage_family_not_selected_by_the_nodal_identity",
        "reference_sine_source_supplies_primitives_not_an_alternative_law_observation",
        "exact_two_C6_unit_support_with_one_aligned_bridge_and_unit_capacities",
        "zero_form_target_internal_abs_turn_one_sixth_and_zero_loss_beta_w_one",
        "tau_equals_t_over_pi_with_fixed_support_capacity_and_no_forcing_or_events",
        "critical_target_and_positive_relative_Hessian_do_not_imply_attraction",
        "full_nodal_energy_and_tangent_are_recomputed_for_the_declared_storage",
        "same_bridge_outputs_and_energy_transpose_preparation_for_every_epsilon",
        "channel_difference_is_signed_not_a_universal_positive_sine_bound",
        "local_phase_radius_bounds_every_edge_perturbation_from_the_declared_target",
        "conditional_local_storage_barrier_not_global_capture_or_an_observed_state_test",
        "strict_excess_below_threshold_lower_endpoint_is_a_sufficient_numeric_cutoff",
        "common_form_and_phase_origins_remain_quotient_directions",
        "no_trajectory_fit_default_change_runtime_law_or_physical_identification",
    )

    def to_dict(self):
        """Project the declared alternative law and its exact static comparison."""
        from ..sdk.relational_reports import _project, _validate_label_groups

        _validate_comparison_labels(self.source)
        _validate_label_groups(
            self.left_cycle,
            self.right_cycle,
            self.bridge,
            self.target_geometry.geometry.nodes,
        )
        return {"schema": "tnfr.bridge-storage-family.v1", "report": _project(self)}


def assess_bridge_storage_family(
    source, *, left_cycle, right_cycle, target_phase_turns, epsilon
) -> BridgeStorageFamilyAssessment:
    """Compare bridge jets and local storage under a supplied U_epsilon law.

    Epsilon is any finite represented or exact nonnegative real; it is neither
    fitted nor inferred from the source. Full support and target admission
    match the conservative bridge-memory reader. U'' at the internal target
    gaps is 1/2+9*epsilon/8, while the aligned bridge retains unit curvature.
    Odd equal-gap cancellation proves criticality for the entire family.

    The local chamber has edge perturbations strictly below pi/12. Within it,
    U'' >= m=cos(5*pi/12)>0 and excess storage is bounded below by
    (sum(form_edge^2)+m*sum(phase_edge^2))/2. For an initial state inside that
    chamber, conserved own-law excess below m*pi^2/288 prevents a first exit.
    The returned interval encloses that threshold; a strict comparison with
    its lower endpoint is sufficient. No initial state is assessed here.
    """
    coefficient = exact_or_represented_real(epsilon, "epsilon")
    if coefficient < 0:
        raise ValueError("epsilon must be nonnegative")
    channels, left, right, _ = _admit_two_cycle_bridge(
        source,
        left_cycle=left_cycle,
        right_cycle=right_cycle,
        target_phase_turns=target_phase_turns,
    )
    state = channels.target_geometry
    geometry = state.geometry
    size = len(channels.source.nodes)
    internal = Q(1, 2) + Q(9, 8) * coefficient
    curvatures = tuple(
        Q(1) if index == channels.bridge_edge_index else internal
        for index in range(len(geometry.edges))
    )
    # Every nonzero current is a common positive factor times the sine sign.
    # Recheck nodal cancellation rather than importing a cached law verdict.
    signs = tuple(Q((turn > 0) - (turn < 0)) for turn in state.edge_turns)
    if any(
        sum((entry * sign for entry, sign in zip(row, signs)), Q(0))
        for row in geometry.incidence
    ):
        raise ArithmeticError("alternative-law target currents do not cancel")
    laplacian, hessian, full, metric = _bridge_full_tangent(channels, internal)
    zero = (Q(0),) * size
    row = tuple(
        Q(entries[channels.bridge_edge_index]) for entries in geometry.incidence
    )
    observation = (row + zero, zero + row)
    lift = channels.nodal_bridge_lift
    preparation = tuple((value, Q(0)) for value in lift) + tuple(
        (Q(0), value) for value in lift
    )
    transpose = tuple(zip(*preparation))
    product = exact_matrix_product
    identity = ((Q(1), Q(0)), (Q(0), Q(1)))
    if (
        product(observation, preparation) != identity
        or product(transpose, metric) != observation
    ):
        raise ArithmeticError("the declared bridge preparation is not energy-transpose")
    weighted = product(metric, full)
    if any(
        weighted[i][j] != -weighted[j][i]
        for i in range(2 * size)
        for j in range(2 * size)
    ):
        raise ArithmeticError(
            "the alternative-law nodal tangent does not conserve storage"
        )
    first = product(product(observation, full), preparation)
    second = product(product(product(observation, full), full), preparation)
    pi = pi_interval()
    curvature_bound = cos(Q(5, 12) * pi)
    return BridgeStorageFamilyAssessment(
        source=channels.source,
        law="normalized_sine_cubic_reciprocal_exchange",
        epsilon=coefficient,
        left_cycle=left,
        right_cycle=right,
        target_phase_turns=channels.target_phase_turns,
        target_geometry=state,
        target_epi=(Q(0),) * size,
        bridge=channels.bridge,
        mobility=channels.mobility,
        edge_phase_curvatures=curvatures,
        form_laplacian=laplacian,
        phase_hessian=hessian,
        full_tangent_generator=full,
        full_energy_metric=metric,
        bridge_observation_rows=observation,
        nodal_bridge_preparation=preparation,
        first_jet=first,
        second_jet=second,
        channel_difference=second[0][0] - second[1][1],
        criticality_status="proved_by_equal_gap_odd_cancellation",
        local_positivity_status="positive_edge_curvatures_on_connected_support",
        target_storage=sum(
            (Q(0) if turn == 0 else Q(1, 2) + Q(5, 24) * coefficient)
            for turn in state.edge_turns
        ),
        local_phase_radius_turns=Q(1, 24),
        local_curvature_lower_bounds=curvature_bound,
        local_excess_storage_threshold_bounds=curvature_bound * pi**2 / 288,
    )


def assess_sine_mediated_response(
    recovery, *, donor_cycle, receiver_cycle, mediator
) -> SineMediatedResponse:
    """Certify remote response for a declared complete two-cycle target.

    The two ordered five-node cycles are disjoint and their first nodes are
    joined only through the distinct mediator. No extra nodes or edges are
    allowed. Capacities must be exact held scalars, including arbitrarily
    small positive rationals; interval-capacity families are not admitted by
    this specialized reader. Shared recovery supplies the target's strict
    acuity, criticality and stable mean quotient. Its observed source is not
    silently substituted for the exact target being differentiated.
    """
    rebuilt = _admit_sine_recovery(recovery)
    positions, neighbors = _recovery_support(rebuilt)
    donor_cycle = _ordered(donor_cycle, "donor_cycle", limit=6)
    receiver_cycle = _ordered(receiver_cycle, "receiver_cycle", limit=6)
    if len(donor_cycle) != 5 or len(receiver_cycle) != 5:
        raise ValueError("donor_cycle and receiver_cycle must each contain five nodes")
    try:
        donor = tuple(positions[node] for node in donor_cycle)
        receiver = tuple(positions[node] for node in receiver_cycle)
        hidden = positions[mediator]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "cycle and mediator nodes must belong to the recovery"
        ) from exc
    if len(positions) != 11 or len(set((*donor, *receiver, hidden))) != 11:
        raise ValueError(
            "two disjoint C5 cycles and one distinct mediator are required"
        )
    expected_edges = {
        tuple(sorted((cycle[i], cycle[(i + 1) % 5])))
        for cycle in (donor, receiver)
        for i in range(5)
    }
    expected_edges.update(
        tuple(sorted((hidden, port))) for port in (donor[0], receiver[0])
    )
    supplied_edges = {
        tuple(sorted((i, j))) for i, row in enumerate(neighbors) for j in row
    }
    if supplied_edges != expected_edges:
        raise ValueError(
            "full support must be exactly the two cycles and their bridges"
        )
    capacity = rebuilt.exact_held_capacity
    if any(value is None for value in capacity):
        raise ValueError(
            "mediated response requires exact held capacities at every node"
        )
    degrees = tuple(map(len, neighbors))
    mobility = tuple(nu / degree for nu, degree in zip(capacity, degrees))
    target = rebuilt.target_phase_turns
    if any(
        _principal_turn_distance(target[hidden], target[port]) != 0
        for port in (donor[0], receiver[0])
    ):
        raise ValueError("the exact acute critical target must have zero bridge turns")

    def contrast(cycle):
        result = [Q(0)] * len(neighbors)
        for index in cycle[1:]:
            result[index] = Q(-1, 4)
        result[cycle[0]] = Q(1)
        return tuple(result)

    donor_contrast, receiver_contrast = contrast(donor), contrast(receiver)
    direction = tuple(k * value for k, value in zip(mobility, donor_contrast))
    pi = pi_interval()
    edge_cosines = tuple(
        tuple(
            I(1) if distance == 0 else cos(2 * distance * pi)
            for distance in (
                _principal_turn_distance(target[i], target[j]) for j in row
            )
        )
        for i, row in enumerate(neighbors)
    )
    forms = [tuple(I(value) for value in direction)]
    phases = [(I(0),) * len(neighbors)]
    for _ in range(2):
        x_rate, theta_rate = _sine_target_tangent_action(
            rebuilt.reference_model,
            neighbors,
            capacity,
            edge_cosines,
            forms[-1],
            phases[-1],
        )
        forms.append(x_rate)
        phases.append(theta_rate)

    def basis_images(index):
        unit = tuple(I(int(i == index)) for i in range(len(neighbors)))
        zero = (I(0),) * len(neighbors)
        return tuple(
            _sine_target_tangent_action(
                rebuilt.reference_model, neighbors, capacity, edge_cosines, x, theta
            )
            for x, theta in ((unit, zero), (zero, unit))
        )

    def block(images, index):
        return tuple(
            tuple(image[coordinate][index] for image in images)
            for coordinate in range(2)
        )

    donor_images, mediator_images = basis_images(donor[0]), basis_images(hidden)
    donor_to_mediator = block(donor_images, hidden)
    mediator_to_receiver = block(mediator_images, receiver[0])
    memory_kernel = tuple(
        tuple(
            sum(
                (
                    mediator_to_receiver[i][k] * donor_to_mediator[k][j]
                    for k in range(2)
                ),
                I(0),
            )
            for j in range(2)
        )
        for i in range(2)
    )

    def observe(vectors):
        return tuple(
            sum((r * value for r, value in zip(receiver_contrast, row)), I(0))
            for row in vectors
        )

    e, w = map(Q, rebuilt.reference_model.effective_weights)
    beta = Q(rebuilt.reference_model.storage_scale)
    product = mobility[donor[0]] * mobility[hidden] * mobility[receiver[0]]
    return SineMediatedResponse(
        recovery=rebuilt,
        donor_cycle=donor_cycle,
        receiver_cycle=receiver_cycle,
        mediator=mediator,
        donor_indices=donor,
        receiver_indices=receiver,
        mediator_index=hidden,
        degrees=degrees,
        mobility=mobility,
        donor_contrast=donor_contrast,
        input_direction=direction,
        receiver_contrast=receiver_contrast,
        form_state_derivatives_bounds=tuple(forms),
        phase_state_derivatives_bounds=tuple(phases),
        form_markov_bounds=observe(forms),
        phase_markov_bounds=observe(phases),
        port_mobility_product=product,
        phase_order_two_pi_numerator=-e * w * product / beta,
        form_order_two_bounds=(e**2 - w**2 / (beta * pi**2)) * product,
        mediator_generator_bounds=block(mediator_images, hidden),
        donor_to_mediator_bounds=donor_to_mediator,
        mediator_to_receiver_bounds=mediator_to_receiver,
        memory_kernel_at_zero_bounds=memory_kernel,
        donor_odd_tangent_sector_silent=_cycle_reflection_preserved(
            donor, neighbors, capacity, target
        ),
        receiver_odd_tangent_readout_silent=_cycle_reflection_preserved(
            receiver, neighbors, capacity, target
        ),
    )


@dataclass(frozen=True)
class SinePairPulseAssessment:
    """Nonlinear P2 libration or an explicit boundary of that certificate.

    Coordinates follow captured node order: u=x0-x1, delta=theta1-theta0.
    With sigma=nu0+nu1, the existing sine rows imply
    u'=-e*sigma*u+(w/pi)*sigma*sin(delta) and
    delta'=-(w/(beta*pi))*sigma*u. The unchanged storage is
    E=u**2/2+beta*(1-cos(delta)), with E'=-e*sigma*u**2.

    At e=0, sigma>0 and 0<E<2*beta the full captured state follows a
    nonlinear periodic libration, including when exactly one capacity is
    zero. Its period lies strictly above the small-amplitude limit. This
    is a continuous theorem for the supplied preparation, not spontaneous
    motion from an equilibrium, an attracting cycle or numerical recurrence.
    """

    comparison: SineExchangeComparison
    form_contrast: Q
    phase_contrast: Q
    capacity_sum: Q
    exact_energy: Q | None
    normalized_energy_bounds: I
    separatrix_energy: Q
    relative_form_rate_bounds: I
    relative_phase_rate_bounds: I
    initial_continuous_loss: Q
    energy_regime: str
    status: str
    nonlinear_periodic_exchange_certified: bool
    nonstationary_recurrence_excluded: bool
    small_amplitude_period_bounds: I | None
    period_bounds: I | None
    period_unavailable_reasons: tuple[str, ...]
    law: str = "normalized_sine_reciprocal_exchange"
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "complete_mutual_singleton_unit_support_from_shared_sine_capture",
        "held_nonnegative_capacities_and_explicit_absent_Gamma_no_events",
        "zero_epi_weight_is_a_supplied_constitutive_boundary_not_a_new_default",
        "signed_form_and_raw_real_phase_contrasts_follow_captured_node_order",
        "shared_work_and_rate_owner_retains_the_exact_declared_continuous_loss",
        "libration_returns_both_form_and_real_phase_lifts_not_only_a_diagnostic",
        "one_zero_capacity_anchors_its_node_without_preventing_relative_libration",
        "both_zero_capacities_freeze_the_complete_pair",
        "positive_epi_weight_excludes_nonstationary_recurrence_even_at_zero_initial_loss",
        "equal_captured_phases_retain_exact_rational_energy_for_boundary_decisions",
        "period_bounds_use_elliptic_integrand_inequalities_without_an_elliptic_solver",
        "period_exists_independently_of_whether_its_numeric_enclosure_is_resolved",
        "period_is_continuous_structural_model_time_not_a_laboratory_clock_bridge",
        "equilibrium_does_not_start_a_pulse_and_energy_levels_are_not_attracting",
        "rotation_separatrix_and_unresolved_energy_have_no_libration_certificate",
        "no_trajectory_finite_step_recurrence_forcing_or_native_runtime_change",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        _validate_comparison_labels(self.comparison)
        return {
            "schema": "tnfr.relational-sine-pair-pulse.v1",
            "report": _project(self),
        }


def _elliptic_libration_period_bounds(small_period, parameter_margin):
    """Enclose T from T0 and the positive elliptic-parameter margin 1-m.

    Conditional on a separately proved nonstationary libration with 0<m<1,
    pi/2 < K(m) <= pi/(2*sqrt(1-m)) gives T0 < T <= T0/sqrt(1-m).
    This shared integrand inequality performs no elliptic solve and does not
    establish libration or accept an unresolved denominator as positive.
    """
    denominator = sqrt(parameter_margin)
    if denominator.lo > 0:
        return I(small_period.lo, small_period.hi / denominator.lo), ()
    return None, ("positive_period_bound_denominator_not_resolved",)


def assess_sine_pair_pulse(graph, *, reference_model) -> SinePairPulseAssessment:
    """Assess a supplied P2 state without selecting a lossless law for it.

    The shared sine reader owns state, support, capacity, coefficient and
    absent-input admission. Positive loss excludes nonstationary recurrence;
    it is never changed to zero here. Only a declared zero-loss model with
    strictly sub-barrier nonzero energy receives the libration certificate.
    Over-barrier rotations and the separatrix retain explicit scope limits.
    """
    comparison = bound_relational_sine_exchange(graph, reference_model=reference_model)
    if (
        len(comparison.nodes) != 2
        or len(comparison.edges) != 1
        or comparison.degrees != (1, 1)
    ):
        raise ValueError("pair pulse assessment requires complete P2 unit support")
    u = comparison.epi[0] - comparison.epi[1]
    delta = comparison.phase[1] - comparison.phase[0]
    sigma = sum(comparison.capacity, Q(0))
    e, w = map(Q, reference_model.effective_weights)
    beta = Q(reference_model.storage_scale)
    exact_energy = comparison.form_storage if delta == 0 else None
    exact_ratio = exact_energy / beta if exact_energy is not None else None
    # Divide exact form storage before outward projection. This keeps a tiny
    # positive storage scale from becoming an artificial interval denominator.
    phase_energy = I(
        max(Q(0), comparison.phase_storage.lo),
        min(Q(2), comparison.phase_storage.hi),
    )
    ratio = I(comparison.form_storage / beta) + phase_energy
    positive_energy = u != 0 or ratio.lo > 0
    if exact_ratio is not None:
        if exact_ratio == 0:
            energy_regime = "zero_energy"
        elif exact_ratio < 2:
            energy_regime = "libration_energy"
        elif exact_ratio == 2:
            energy_regime = "separatrix_energy"
        else:
            energy_regime = "rotation_energy"
    elif positive_energy and ratio.hi < 2:
        energy_regime = "libration_energy"
    elif ratio.lo > 2:
        energy_regime = "rotation_energy"
    else:
        energy_regime = "unresolved"

    if sigma == 0:
        status = "frozen"
    elif u == 0 and delta == 0:
        status = "equilibrium"
    elif e > 0:
        status = "dissipative_recurrence_excluded"
    elif energy_regime == "libration_energy":
        status = "libration_certified"
    elif energy_regime == "rotation_energy":
        status = "rotation_out_of_scope"
    elif energy_regime == "separatrix_energy":
        status = "separatrix_out_of_scope"
    else:
        status = "energy_classification_unresolved"

    small_period = period = None
    unavailable = []
    if sigma > 0 and e == 0:
        base = pi_interval() ** 2 * sqrt(beta)
        scale = 2 / (w * sigma)
        small_period = I(scale * base.lo, scale * base.hi)
    if status == "libration_certified":
        margin = (
            I(1 - exact_ratio / 2)
            if exact_ratio is not None
            else I(1 - ratio.hi / 2, 1 - ratio.lo / 2)
        )
        period, reasons = _elliptic_libration_period_bounds(small_period, margin)
        unavailable.extend(reasons)
    else:
        unavailable.append("no_certified_nonstationary_libration")
    return SinePairPulseAssessment(
        comparison=comparison,
        form_contrast=u,
        phase_contrast=delta,
        capacity_sum=sigma,
        exact_energy=exact_energy,
        normalized_energy_bounds=ratio,
        separatrix_energy=2 * beta,
        relative_form_rate_bounds=comparison.form_rates[0] - comparison.form_rates[1],
        relative_phase_rate_bounds=comparison.phase_rates[1]
        - comparison.phase_rates[0],
        initial_continuous_loss=comparison.continuous_loss,
        energy_regime=energy_regime,
        status=status,
        nonlinear_periodic_exchange_certified=status == "libration_certified",
        nonstationary_recurrence_excluded=(
            e > 0 or status in ("frozen", "equilibrium")
        ),
        small_amplitude_period_bounds=small_period,
        period_bounds=period,
        period_unavailable_reasons=tuple(unavailable),
    )


@dataclass(frozen=True)
class SinePathMemoryAssessment:
    """Exact conservative consensus-tangent memory and live nonlinear flux.

    The captured P3 state supplies the initial *linear* deviations from an
    explicitly declared uniform reference. It need not be at that reference.
    No error bound identifies the resulting linear evolution with the future
    nonlinear trajectory. Both endpoint form/phase pairs are observed, while
    the intermediary form/phase pair is retained in the memory identity.

    Exact matrices use tau=w*t/pi. With M=nu*D^-1*L, the reference tangent is
    [[0,-M],[M/beta,0]] without rounding mathematical pi. Its hidden block D
    satisfies D**2=-omega**2*I, omega**2=nu**2/beta. Therefore the memory is
    cos(omega*tau)*K0+sin(omega*tau)/omega*K1, where K0 is the shared
    coordinate-memory kernel at zero and K1 its reported derivative. Hidden
    forcing has the same two coefficients Bh0 and BDh0. No exponential or
    memory convolution is evaluated.

    The two nonlinear edge rates instead use the original structural time
    and the actual captured state. Their opposite flux redistributes the
    original edge storage without a dissipative sink or added reservoir.
    """

    comparison: SineExchangeComparison
    path: tuple[Any, Any, Any]
    path_indices: tuple[int, int, int]
    consensus_form: Q
    consensus_phase: Q
    common_capacity: Q
    clock_rate_pi_numerator: Q
    clock_rate_bounds: I
    initial_tangent_state: tuple[Q, ...]
    coordinate_memory: LinearCoordinateMemory
    linear_observation: LinearObservation
    hidden_initial_state: tuple[Q, ...]
    hidden_initial_forcing: tuple[Q, ...]
    hidden_initial_forcing_derivative: tuple[Q, ...]
    kernel_derivative_at_zero: tuple[tuple[Q, ...], ...]
    hidden_frequency_squared: Q
    full_tangent_period_bounds: I
    tangent_motion_nonstationary: bool
    nonlinear_edge_storage_bounds: tuple[I, I]
    nonlinear_edge_rate_bounds: tuple[I, I]
    nonlinear_transfer_bounds: I
    nonlinear_edge_rate_residual_bounds: tuple[I, I]
    law: str = "normalized_sine_reciprocal_exchange"
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "complete_unit_P3_with_declared_middle_node_and_common_positive_held_capacity",
        "explicit_zero_epi_weight_no_Gamma_input_support_event_or_new_default",
        "uniform_consensus_reference_is_supplied_not_inferred_from_current_state",
        "reference_form_and_phase_origins_are_declared_exact_or_represented_reals",
        "full_tangent_state_order_is_all_forms_then_all_real_phase_deviations",
        "tau_equals_w_t_over_pi_exact_clock_change_not_rationalized_pi",
        "coordinate_memory_and_linear_observation_reuse_shared_exact_matrix_owners",
        "endpoints_are_visible_and_both_intermediary_coordinates_are_retained",
        "memory_and_hidden_forcing_coefficients_are_derivatives_in_normalized_time_tau",
        "oscillatory_hidden_memory_is_not_instantaneous_damping_or_a_decaying_kernel",
        "hidden_initial_forcing_is_retained_without_a_zero_state_assumption",
        "all_state_visible_linear_closure_requires_two_additional_coordinates",
        "full_tangent_return_period_need_not_be_a_particular_state_fundamental_period",
        "linear_recurrence_is_not_a_finite_amplitude_nonlinear_period_certificate",
        "nonlinear_edge_storage_and_work_use_the_live_state_and_original_structural_time",
        "positive_and_negative_boundary_work_are_transfer_not_full_system_loss",
        "no_trajectory_matrix_exponential_memory_solver_or_damping_approximation",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        _validate_comparison_labels(self.comparison)
        return {
            "schema": "tnfr.relational-sine-path-memory.v1",
            "report": _project(self),
        }


def assess_sine_path_memory(
    graph, *, reference_model, mediator, consensus_form=0, consensus_phase=0
) -> SinePathMemoryAssessment:
    """Retain conservative P3 memory rather than postulating effective loss.

    The complete shared capture must be a three-node path with the declared
    mediator at its center, positive common held capacity and explicit e=0.
    The graph's state is used in the live nonlinear edge balances and as a
    declared initial perturbation of the separate consensus tangent. Those
    are different statements, even though both consume the same coordinates.
    Endpoint order follows capture order; the middle node is never removed
    from the model. The exact clock change multiplies every tangent row.
    """
    from ..mathematics._exact_linear_algebra import exact_matrix_product
    from ..mathematics.linear_observation import (
        derive_coordinate_memory,
        derive_linear_observation,
    )
    from .relational_sine_comparison import _comparison_neighbors

    reference_x = exact_or_represented_real(consensus_form, "consensus_form")
    reference_theta = exact_or_represented_real(consensus_phase, "consensus_phase")
    comparison = bound_relational_sine_exchange(graph, reference_model=reference_model)
    e, w = map(Q, reference_model.effective_weights)
    if e != 0:
        raise ValueError("conservative path memory requires explicitly zero EPI weight")
    if len(comparison.nodes) != 3 or len(comparison.edges) != 2:
        raise ValueError("conservative path memory requires complete P3 unit support")
    positions = {node: i for i, node in enumerate(comparison.nodes)}
    try:
        hidden = positions[mediator]
    except (KeyError, TypeError) as exc:
        raise ValueError("mediator must belong to the complete captured path") from exc
    neighbors = _comparison_neighbors(comparison)
    if comparison.degrees[hidden] != 2:
        raise ValueError("mediator must be the middle node of the complete P3 path")
    endpoints = tuple(i for i in range(3) if i != hidden)
    if any(comparison.degrees[i] != 1 for i in endpoints):
        raise ValueError("P3 endpoints must each have exactly one neighbor")
    nu = comparison.capacity[0]
    if nu <= 0 or any(value != nu for value in comparison.capacity):
        raise ValueError(
            "conservative P3 memory requires positive common held capacity"
        )
    beta = Q(reference_model.storage_scale)
    transport = tuple(
        tuple(
            nu * (len(row) if i == j else -int(j in row)) / len(row) for j in range(3)
        )
        for i, row in enumerate(neighbors)
    )
    zero = (Q(0),) * 3
    generator = tuple(
        zero + tuple(-value for value in row) for row in transport
    ) + tuple(tuple(value / beta for value in row) + zero for row in transport)
    visible = endpoints + tuple(i + 3 for i in endpoints)
    memory = derive_coordinate_memory(generator, visible)
    output_rows = tuple(tuple(Q(int(i == j)) for j in range(6)) for i in visible)
    observation = derive_linear_observation(generator, output_rows)
    initial = tuple(value - reference_x for value in comparison.epi) + tuple(
        value - reference_theta for value in comparison.phase
    )
    hidden_initial = tuple(initial[i] for i in memory.hidden_indices)
    hidden_column = tuple((value,) for value in hidden_initial)
    bh = exact_matrix_product(memory.hidden_to_visible, hidden_column)
    bd = exact_matrix_product(memory.hidden_to_visible, memory.hidden_generator)
    bdh = exact_matrix_product(bd, hidden_column)
    kernel_derivative = exact_matrix_product(bd, memory.visible_to_hidden)
    full_rate = exact_matrix_product(generator, tuple((value,) for value in initial))
    # Scale exact held capacity after enclosing the transcendental numerator,
    # so a positive sub-grid capacity remains a valid structural clock scale.
    base_period = pi_interval() ** 2 * sqrt(beta)
    period_scale = 2 / (w * nu)
    period = I(period_scale * base_period.lo, period_scale * base_period.hi)

    edge_storage, edge_rates = [], []
    for index in endpoints:
        r = comparison.epi[index] - comparison.epi[hidden]
        real, imaginary = comparison.relative_resultant[index]
        edge_storage.append(r**2 / 2 + beta * (1 - real))
        # The endpoint's imaginary resultant is sin(theta_h-theta_endpoint).
        # Hence the derivative of its cosine cost uses the opposite sign
        # when multiplying (theta_endpoint'-theta_h').
        edge_rates.append(
            r * (comparison.form_rates[index] - comparison.form_rates[hidden])
            - beta
            * imaginary
            * (comparison.phase_rates[index] - comparison.phase_rates[hidden])
        )
    left, right = endpoints
    r = comparison.epi[left] - comparison.epi[hidden]
    s = comparison.epi[right] - comparison.epi[hidden]
    sin_alpha = comparison.relative_resultant[left][1]
    sin_gamma = comparison.relative_resultant[right][1]
    transfer = (nu * w / 2) * (r * sin_gamma - s * sin_alpha) / pi_interval()
    return SinePathMemoryAssessment(
        comparison=comparison,
        path=(
            comparison.nodes[left],
            comparison.nodes[hidden],
            comparison.nodes[right],
        ),
        path_indices=(left, hidden, right),
        consensus_form=reference_x,
        consensus_phase=reference_theta,
        common_capacity=nu,
        clock_rate_pi_numerator=w,
        clock_rate_bounds=I(w) / pi_interval(),
        initial_tangent_state=initial,
        coordinate_memory=memory,
        linear_observation=observation,
        hidden_initial_state=hidden_initial,
        hidden_initial_forcing=tuple(row[0] for row in bh),
        hidden_initial_forcing_derivative=tuple(row[0] for row in bdh),
        kernel_derivative_at_zero=kernel_derivative,
        hidden_frequency_squared=nu**2 / beta,
        full_tangent_period_bounds=period,
        tangent_motion_nonstationary=any(row[0] != 0 for row in full_rate),
        nonlinear_edge_storage_bounds=tuple(edge_storage),
        nonlinear_edge_rate_bounds=tuple(edge_rates),
        nonlinear_transfer_bounds=transfer,
        nonlinear_edge_rate_residual_bounds=(
            edge_rates[0] - transfer,
            edge_rates[1] + transfer,
        ),
    )


@dataclass(frozen=True)
class SineRecurrenceAssessment:
    """A finite invariant nonlinear family, not a selected-orbit prediction.

    For e=0, fixed positive capacities and full connected unit support, the
    exact divergence is zero and both storage and the weighted form mean
    are invariant. The caller declares E<=energy_ceiling and a positive-width
    mean slab. With phases interpreted on circles, these bounds define a
    compact invariant family of finite positive volume. Poincare recurrence
    holds almost everywhere in that family. A captured nonstationary state
    is not certified recurrent merely because it belongs to the family.

    The n-1 path-length bound gives |x_i-m|<=sqrt(2*(n-1)*energy_ceiling).
    Its outward coordinate box witnesses compactness without computing a
    spectrum, all-pairs distances, a recurrence time or a trajectory.
    """

    comparison: SineExchangeComparison
    energy_ceiling: Q
    form_mean_bounds: tuple[Q, Q]
    form_mean_weights: tuple[Q, ...]
    normalized_mean_weights: tuple[Q, ...]
    weighted_form_mean: Q
    weighted_mean_rate_residual_bounds: I
    divergence: Q
    path_length_upper_bound: int
    form_radius_bounds: I
    form_coordinate_bounds: I
    captured_energy_exact: Q | None
    snapshot_family_membership: str
    snapshot_membership_reasons: tuple[str, ...]
    snapshot_motion_status: str
    individual_recurrence_status: str
    almost_everywhere_recurrence_certified: bool = True
    finite_positive_family_volume_certified: bool = True
    law: str = "normalized_sine_reciprocal_exchange"
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "shared_complete_finite_connected_unit_support_with_at_least_two_nodes",
        "explicit_zero_epi_weight_strictly_positive_held_capacities_no_inputs_or_events",
        "caller_declares_positive_energy_ceiling_and_positive_width_mean_slab",
        "nonlinear_energy_and_weighted_form_mean_are_exact_invariants",
        "divergence_minus_epi_weight_times_total_capacity_is_exactly_zero",
        "weighted_mean_rate_residual_is_computed_from_shared_interval_rates",
        "compact_family_lives_in_real_form_times_circular_phase_not_unbounded_phase_lifts",
        "n_minus_one_is_a_conservative_path_length_bound_not_a_computed_diameter",
        "family_has_finite_positive_product_Lebesgue_form_and_circle_Haar_volume",
        "almost_everywhere_recurrence_is_a_family_theorem_not_chosen_state_membership",
        "probability_one_requires_a_preparation_absolutely_continuous_in_that_ambient_family",
        "no_transfer_to_an_exact_energy_surface_fixed_mean_surface_or_point_preparation",
        "stationary_snapshot_has_only_trivial_recurrence_not_a_nonstationary_pulse",
        "nonstationary_and_numerically_unresolved_snapshots_receive_no_individual_verdict",
        "recurrence_is_arbitrarily_close_return_not_exact_periodicity_or_a_common_clock",
        "no_recurrence_time_amplitude_pattern_identity_or_attraction_certificate",
        "no_sampling_trajectory_solver_law_selection_or_native_runtime_change",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        _validate_comparison_labels(self.comparison)
        return {
            "schema": "tnfr.relational-sine-recurrence.v1",
            "report": _project(self),
        }


def assess_sine_recurrence(
    graph, *, reference_model, energy_ceiling, form_mean_bounds
) -> SineRecurrenceAssessment:
    """Admit a declared compact nonlinear family and report snapshot limits.

    Energy and mean bounds specify the analyzed family, not new dynamics or
    automatically chosen margins around the captured state. Outside or
    unresolved snapshot membership does not invalidate the family theorem.
    Even confirmed membership does not prove recurrence of that exact state.
    The only individual positive verdict here is a proved stationary state.
    """
    maximum = exact_or_represented_real(energy_ceiling, "energy_ceiling")
    if maximum <= 0:
        raise ValueError("energy_ceiling must be strictly positive")
    raw_mean = _ordered(form_mean_bounds, "form_mean_bounds", limit=3)
    if len(raw_mean) != 2:
        raise ValueError("form_mean_bounds must contain exactly two ordered endpoints")
    lower, upper = (
        exact_or_represented_real(value, "form_mean_bounds") for value in raw_mean
    )
    if lower >= upper:
        raise ValueError("form_mean_bounds must have strictly positive width")
    comparison = bound_relational_sine_exchange(graph, reference_model=reference_model)
    e = Q(reference_model.effective_weights[0])
    if e != 0:
        raise ValueError("nonlinear recurrence admission requires zero EPI weight")
    if any(nu <= 0 for nu in comparison.capacity):
        raise ValueError(
            "nonlinear recurrence admission requires positive held capacities"
        )
    weights = tuple(
        Q(degree) / nu for degree, nu in zip(comparison.degrees, comparison.capacity)
    )
    weight_sum = sum(weights, Q(0))
    normalized = tuple(value / weight_sum for value in weights)
    mean = sum((rho * x for rho, x in zip(normalized, comparison.epi)), Q(0))
    mean_rate = sum(
        (rho * rate for rho, rate in zip(normalized, comparison.form_rates)), I(0)
    )
    length_bound = len(comparison.nodes) - 1
    radius = sqrt(2 * length_bound * maximum)
    coordinate_box = I(lower - radius.hi, upper + radius.hi)
    equal_phases = all(value == comparison.phase[0] for value in comparison.phase)
    exact_energy = comparison.form_storage if equal_phases else None
    reasons = []
    if not lower <= mean <= upper:
        membership = "outside"
        reasons.append("weighted_mean_outside_declared_slab")
    elif exact_energy is not None:
        membership = "inside" if exact_energy <= maximum else "outside"
        if membership == "outside":
            reasons.append("exact_energy_exceeds_declared_ceiling")
    elif comparison.storage.hi <= maximum:
        membership = "inside"
    elif comparison.storage.lo > maximum:
        membership = "outside"
        reasons.append("energy_lower_bound_exceeds_declared_ceiling")
    else:
        membership = "unresolved"
        reasons.append("energy_ceiling_membership_not_resolved")
    if any(value != 0 for value in comparison.form_gradient):
        motion = "nonstationary"
    elif equal_phases:
        motion = "stationary"
    elif any(
        imaginary.lo > 0 or imaginary.hi < 0
        for _, imaginary in comparison.relative_resultant
    ):
        motion = "nonstationary"
    else:
        motion = "unresolved"
    return SineRecurrenceAssessment(
        comparison=comparison,
        energy_ceiling=maximum,
        form_mean_bounds=(lower, upper),
        form_mean_weights=weights,
        normalized_mean_weights=normalized,
        weighted_form_mean=mean,
        weighted_mean_rate_residual_bounds=mean_rate,
        divergence=-e * sum(comparison.capacity, Q(0)),
        path_length_upper_bound=length_bound,
        form_radius_bounds=radius,
        form_coordinate_bounds=coordinate_box,
        captured_energy_exact=exact_energy,
        snapshot_family_membership=membership,
        snapshot_membership_reasons=tuple(reasons),
        snapshot_motion_status=motion,
        individual_recurrence_status=(
            "trivial_stationary_recurrence"
            if motion == "stationary"
            else "unavailable_for_chosen_state"
        ),
    )
