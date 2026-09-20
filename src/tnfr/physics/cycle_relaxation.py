"""Conditional continuous relaxation on a fixed acute weighted cycle.

The supplied equal-capacity averaged-sine phase law preserves its acute gap
sector. Its two-neighbor phasor source drives the canonical weighted EPI row.
This evaluator bounds that exact-real model from detached represented initial
data; it neither advances the graph nor certifies a binary64 trajectory.
A detached comparison relates that supplied sine response to a separately
declared phase-pressure response without selecting or installing either law.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Set
from dataclasses import dataclass
from fractions import Fraction
from itertools import islice
from numbers import Integral

from .._exact_time import exact_or_represented_real
from ..mathematics._phase_midpoint import _affine_interval, _oriented_turn, _pi_bounds
from ..operators._phase_gate import resolve_u3_phase_limits
from .forcing_realization import (
    NonEpiForcingObservation,
    _validated_forcing_decomposition,
    capture_non_epi_forcing,
)
from .hybrid_operator_stability import _exact_sqrt_upper
from .reversible_eigenmode_reference import _negative_exp_bounds
from .structural_diffusion import (
    _exact_flow_gap_from_rationals,
    _exact_real_laplacian_gap_lower_bound,
)

__all__ = [
    "CycleRelaxationEnvelope",
    "CycleRelaxationSample",
    "CycleRestoringResponseComparison",
    "CycleCapacityForcingBudget",
    "CycleCapacityForcingSample",
    "bound_cycle_capacity_forcing",
    "bound_cycle_relaxation",
    "compare_cycle_restoring_responses",
]


@dataclass(frozen=True)
class CycleRelaxationSample:
    """Rational upper envelopes at one declared continuous model time.

    The form disagreement uses ``sum(s_i*(x_i-m_s)**2)``. The two mean
    bounds concern displacement from the initial mean and distance to the
    limiting mean respectively; neither supplies a fitted limiting value.
    """

    time: Fraction
    gap_deviation_squared_upper: Fraction
    phase_source_squared_upper: Fraction
    epi_disagreement_squared_upper: Fraction
    mean_displacement_squared_upper: Fraction
    mean_tail_squared_upper: Fraction
    duhamel_integral_upper: Fraction


@dataclass(frozen=True)
class CycleRelaxationEnvelope:
    """Exact coefficient theorem, affine-pi admission and separate capture.

    Gap arrays follow the caller's cycle order: edge i goes from cycle node
    i to node i+1. ``gap_affine`` contains (rational, pi coefficient), with
    mathematical pi, not the binary64 phase-wrap period. Strengths and the
    captured vectors instead follow ``capture.snapshot.nodes``. Squared
    quantities avoid introducing rounded square roots into the bounds.

    The capture retains production phase pressure, fresh assembly defects
    and stale stored-pressure residuals independently. They are not silently
    substituted for the exact-real midpoint source of this theorem.
    """

    capture: NonEpiForcingObservation
    cycle_indices: tuple[int, ...]
    cycle_order: tuple
    coupling_strength: Fraction
    capacity: Fraction
    epi_weight: Fraction
    phase_weight: Fraction
    effective_phase_gate: Fraction
    gap_affine: tuple[tuple[Fraction, int], ...]
    gap_enclosures: tuple[tuple[Fraction, Fraction], ...]
    winding: int
    mean_gap_pi_coefficient: Fraction
    gap_deviation_enclosures: tuple[tuple[Fraction, Fraction], ...]
    phase_radius_upper: Fraction
    cosine_lower_bound: Fraction
    phase_laplacian_gap_lower_bound: Fraction
    phase_decay_rate_lower_bound: Fraction
    epi_decay_rate_lower_bound: Fraction
    strengths: tuple[Fraction, ...]
    weighted_mean: Fraction
    initial_epi_disagreement_squared: Fraction
    initial_gap_deviation_squared_upper: Fraction
    phase_source_prefactor_squared_upper: Fraction
    form_forcing_prefactor_squared_upper: Fraction
    mean_rate_prefactor_squared_upper: Fraction
    all_time_epi_disagreement_squared_upper: Fraction
    mean_limit_offset_upper: Fraction
    all_time_epi_interval: tuple[Fraction, Fraction]
    samples: tuple[CycleRelaxationSample, ...]
    scope: tuple[str, ...] = (
        "conditional_exact_real_continuous_model_from_represented_initial_state",
        "fixed_simple_cycle_positive_symmetric_actual_conductances",
        "supplied_equal_positive_capacity_and_averaged_sine_coupling_law",
        "strict_acute_gaps_and_fixed_full_UM_gate_admission_in_exact_model",
        "canonical_two_neighbor_midpoint_phase_source_and_fresh_EPI_diffusion",
        "capacity_and_topology_channels_vanish_on_the_admitted_fixed_cycle",
        "unrestricted_scalar_chart_without_Gamma_events_or_controllers",
        "runtime_configuration_beyond_the_captured_coefficients_not_admitted",
        "no_binary64_solver_convergence_future_runtime_or_autonomous_birth_claim",
        "3_to_12_nodes_and_257_times_are_evaluation_budgets_not_physical_limits",
        "shared_exponential_work_limit_4096_and_outward_64_bit_exponent_grid",
    )

    def _phase_orbit_coordinates(self):
        """Split the admitted lift into a common offset and centered shape."""
        prefixes = [(Fraction(0), Fraction(0))]
        for rational, coefficient in self.gap_affine[:-1]:
            previous = prefixes[-1]
            prefixes.append(
                (
                    previous[0] + rational,
                    previous[1] + coefficient - self.mean_gap_pi_coefficient,
                )
            )
        count = len(prefixes)
        mean = tuple(sum(row[j] for row in prefixes) / count for j in (0, 1))
        offset = (
            self.capture.phase[self.cycle_indices[0]] + mean[0],
            mean[1],
        )
        shape = tuple((a - mean[0], b - mean[1]) for a, b in prefixes)
        return offset, shape

    @property
    def initial_phase_offset_affine(self) -> tuple[Fraction, Fraction]:
        """Return alpha(0) as rational + coefficient*pi in the chosen lift.

        The supplied equal-capacity sine law gives alpha(t)=alpha(0)+capacity*t.
        This ordered-support chart coordinate is not a global phasor direction
        or a target chosen from an unordered phase multiset.
        """
        return self._phase_orbit_coordinates()[0]

    @property
    def initial_phase_shape_affine(self) -> tuple[tuple[Fraction, Fraction], ...]:
        """Return centered lifted deformation from the regular winding orbit.

        In cycle order, theta_lift_i=alpha+i*mean_gap+h_i. Each h_i is an
        affine-pi pair; sum(h)=0 and h_(i+1)-h_i equals the gap deviation,
        including the closing edge. This is not a quotient by phase reversal.
        """
        return self._phase_orbit_coordinates()[1]

    @property
    def initial_phase_shape_enclosures(
        self,
    ) -> tuple[tuple[Fraction, Fraction], ...]:
        """Enclose the initial lifted shape using the shared exact pi owner."""
        pi_bounds = _pi_bounds()
        return tuple(
            _affine_interval(rational, coefficient, pi_bounds)
            for rational, coefficient in self.initial_phase_shape_affine
        )

    @property
    def initial_phase_curvature_affine(self) -> tuple[tuple[Fraction, Fraction], ...]:
        """Return ideal initial curvature as affine-pi pairs in cycle order.

        On the admitted acute two-neighbor chart,
        ``K_phi_i=-(delta_i-delta_(i-1))/2=(L_cycle*h)_i/2``. The Laplacian
        is the unweighted support operator, distinct from weighted form
        transport. Full curvature and the retained winding reconstruct h
        through ``h=2*L_cycle^+*K_phi``; curvature alone loses the sector
        and common phase offset. These are exact-model coordinates derived
        from admitted gaps, not rounded production curvature observations.
        """
        return tuple(
            tuple(
                -(
                    exact_or_represented_real(right, "gap coefficient")
                    - exact_or_represented_real(left, "gap coefficient")
                )
                / 2
                for right, left in zip(row, self.gap_affine[i - 1], strict=True)
            )
            for i, row in enumerate(self.gap_affine)
        )

    @property
    def phase_orbit_distance_squared_upper(self) -> tuple[Fraction, ...]:
        """Bound lifted squared shape norms at the existing sample times.

        Poincare gives ||h||^2 <= ||gap-mean_gap||^2/lambda_2(L_cycle).
        The represented rational lower spectral bound makes the returned
        upper bound conservative. Circular distance to the regular-twist
        orbit is no larger; binary64 trajectory error remains separate.
        """
        return tuple(
            sample.gap_deviation_squared_upper / self.phase_laplacian_gap_lower_bound
            for sample in self.samples
        )


@dataclass(frozen=True)
class CycleRestoringResponseComparison:
    """Conditional exact bounds for two separately supplied phase responses.

    On the admitted acute cycle, ``J_i=(sin(delta_i)-sin(delta_(i-1)))/2``
    and ``g_i=(delta_i-delta_(i-1))/(2*pi)`` obey ``g_i=m_i*J_i``.
    The multiplier is the reciprocal pi-scaled sine divided difference;
    where both responses vanish its continuous extension is used instead
    of dividing zero observations. The two declared phase laws are
    ``theta'=kappa+K*J`` and ``theta'=kappa+K*g``, with the same supplied
    common free rate. Both ideal responses telescope to zero over the nodes.

    Response mobility multiplies the sine current in the phase row; it is
    not a new capacity in the nodal EPI equation. The pressure class rate
    uses the general positive-response theorem; its sharper linear rate
    uses the exact two-neighbor pressure reduction. No new phase law is
    derived from the nodal product or installed in the engine.

    For the pressure candidate only, ``h'=K*g`` makes
    ``m_s-kappa*w*s.T*h/(K*sum(s))`` constant. The limiting form mean is
    supplied as an affine-pi pair and its rational enclosure. This requires
    the declared fixed positive capacity/conductance and unforced, unclipped
    scalar form law throughout; it is neither a sine-law endpoint nor a
    transferred clipping or binary64 trajectory certificate.
    """

    coupling_strength: Fraction
    cosine_lower_bound: Fraction
    phase_laplacian_gap_lower_bound: Fraction
    pressure_to_current_ratio_bounds: tuple[Fraction, Fraction]
    current_response_mobility_bounds: tuple[Fraction, Fraction]
    pressure_response_mobility_bounds: tuple[Fraction, Fraction]
    current_decay_rate_lower_bound: Fraction
    pressure_class_decay_rate_lower_bound: Fraction
    pressure_linear_decay_rate_lower_bound: Fraction
    pressure_response_mean_limit_affine: tuple[Fraction, Fraction]
    pressure_response_mean_limit_enclosure: tuple[Fraction, Fraction]
    scope: tuple[str, ...] = (
        "conditional_exact_real_arithmetic_from_a_declared_cycle_envelope",
        "original_fixed_acute_cycle_capacity_and_pressure_premises_retained",
        "supplied_common_free_phase_rate_and_positive_K_for_both_candidates",
        "sine_current_and_phase_pressure_are_separately_supplied_phase_laws",
        "both_ideal_candidate_corrections_have_zero_sum_so_alpha_dot_equals_kappa",
        "generic_nonuniform_positive_response_can_add_a_common_phase_drift",
        "ratio_uses_continuous_extension_at_equal_adjacent_gaps",
        "response_mobility_is_not_nodal_form_capacity",
        "pressure_candidate_mean_limit_uses_fixed_capacity_and_actual_strengths",
        "mean_limit_requires_unforced_unclipped_form_without_events_or_controllers",
        "sine_envelope_samples_and_clipping_bounds_are_not_pressure_candidate_bounds",
        "coefficient_checks_do_not_authenticate_envelope_or_live_provenance",
        "no_new_solver_controller_graph_write_or_binary64_identity",
        "no_autonomous_formation_or_unique_constitutive_law_claim",
    )


@dataclass(frozen=True)
class CycleCapacityForcingSample:
    """Upper budgets at a declared physical time in the conditional model.

    Capacity exposure bounds ``integral ||D_cycle*nu||_2 dt``; the phase
    source is the unit midpoint channel, and the complete non-EPI source
    also retains the capacity contrast channel. Form energy is the actual
    conductance Dirichlet energy, not a moving weighted-mean variance.
    Integrated source bounds and ``epi_interval`` cover the whole physical
    prefix [0,time] in the exact unclipped model, by the transport maximum
    principle. They are not bounds on a binary64 solver's accumulated error.
    """

    time: Fraction
    capacity_exposure_upper: Fraction
    gap_deviation_norm_upper: Fraction
    phase_source_norm_upper: Fraction
    non_epi_source_norm_upper: Fraction
    epi_dirichlet_energy_upper: Fraction
    gap_deviation_integral_upper: Fraction
    non_epi_source_integral_upper: Fraction
    epi_interval: tuple[Fraction, Fraction]


@dataclass(frozen=True)
class CycleCapacityForcingBudget:
    """Prospective interval-capacity tube for the supplied local sine law.

    The retained envelope supplies initial form/phase, actual fixed support,
    pressure coefficients and positive K. Its held capacity is replaced by
    the displayed interval premise at every physical time. The common-rate
    sine reference is reused only for phase-shape comparison; its sampled
    form bounds, limiting mean and clipping conclusions are not transferred.

    A failed strict tube margin returns ``admitted=False`` and no samples.
    Positive admission is conditional on the complete declared model, not
    evidence that an engine writer or future schedule obeys those premises.
    """

    capacity_bounds: tuple[Fraction, Fraction]
    phase_radius: Fraction
    initial_phase_radius_upper: Fraction
    capacity_difference_norm_upper: Fraction
    initial_gap_deviation_norm_upper: Fraction
    phase_decay_rate_lower_bound: Fraction
    phase_gap_tail_upper: Fraction
    reference_gap_deviation_upper: Fraction
    phase_tube_radius_upper: Fraction
    all_time_gap_norm_upper: Fraction
    transport_laplacian_gap_lower_bound: Fraction
    initial_epi_dirichlet_energy: Fraction
    epi_energy_decay_rate_lower_bound: Fraction
    non_epi_source_norm_upper: Fraction
    epi_energy_drive_upper: Fraction
    epi_energy_tail_upper: Fraction
    admitted: bool
    admission_failure: str | None
    samples: tuple[CycleCapacityForcingSample, ...]
    scope: tuple[str, ...] = (
        "conditional_exact_real_model_from_retained_initial_phase_form_support",
        "fixed_actual_symmetric_positive_cycle_conductance_and_pressure_weights",
        "held_template_capacity_replaced_by_declared_positive_interval_at_all_times",
        "supplied_constant_positive_K_averaged_sine_phase_law_theta_dot_equals_nu_plus_KJ",
        "capacity_may_be_measurable_or_jump_without_direct_phase_or_form_jumps",
        "strict_acute_U3_tube_from_centered_shape_comparison_with_existing_reference",
        "capacity_pressure_retained_and_topology_pressure_zero_on_fixed_cycle",
        "unforced_unclipped_scalar_form_row_without_Gamma_other_events_or_controllers",
        "common_actual_conductance_Dirichlet_energy_not_fixed_capacity_mean_variance",
        "finite_prefix_absolute_form_interval_from_transport_maximum_principle",
        "no_capacity_homogenization_zero_tail_or_integrable_total_exposure_claim",
        "no_all_time_form_bound_mean_or_phase_offset_convergence_or_autonomous_maintenance",
        "no_graph_write_kernel_recapture_solver_or_future_binary64_execution_certificate",
        "arithmetic_consistency_does_not_authenticate_caller_created_template_or_live_state",
    )


def compare_cycle_restoring_responses(
    envelope: CycleRelaxationEnvelope,
) -> CycleRestoringResponseComparison:
    """Compare existing sine response with a conditional pressure response.

    Reuse the envelope's positive coupling ``K``, acute cosine lower bound
    ``c`` and cycle spectral lower bound ``lambda``. The shared exact pi
    enclosure gives ``1/pi <= m_i <= 1/(pi*c)`` without a floating-point
    trigonometric evaluation. Multiplication by K bounds the pressure
    candidate's response mobility. The general positive-response class
    gives rate ``a_min*c*lambda/2``; the pressure candidate's exact linear
    gap law improves that to ``K*lambda/(2*pi)``.

    This read-only comparison does not resample or relabel the original
    sine envelope. Its additional pressure-candidate endpoint is
    ``m_s(0)-kappa*w*s.T*h(0)/(K*sum(s))``, with the actual strengths
    aligned from capture order to the retained cycle order. The exact-model
    limiting mean is affine in mathematical pi, with a rational enclosure;
    it introduces neither a fitted target nor another phase integration.
    The unrestricted scalar row excludes Gamma, clipping, later events,
    capacity changes and controllers. The sine envelope's clipping bounds
    are not certified for this alternative candidate.

    All derived coefficients are rational. Consumed inputs
    use the shared exact-or-represented scalar boundary and must satisfy
    the envelope's rate identity. These checks establish only arithmetic
    consistency, not authenticity of caller-created or replaced reports,
    current graph admission, an executed phase law or future stability.
    """
    if not isinstance(envelope, CycleRelaxationEnvelope):
        raise TypeError("envelope must be a CycleRelaxationEnvelope")
    coupling = exact_or_represented_real(
        envelope.coupling_strength, "coupling_strength"
    )
    cosine = exact_or_represented_real(
        envelope.cosine_lower_bound, "cosine_lower_bound"
    )
    phase_gap = exact_or_represented_real(
        envelope.phase_laplacian_gap_lower_bound,
        "phase_laplacian_gap_lower_bound",
    )
    current_rate = exact_or_represented_real(
        envelope.phase_decay_rate_lower_bound, "phase_decay_rate_lower_bound"
    )
    if coupling <= 0:
        raise ValueError("coupling_strength must be positive")
    if not 0 < cosine <= 1:
        raise ValueError("cosine_lower_bound must lie in (0,1]")
    if phase_gap <= 0:
        raise ValueError("phase_laplacian_gap_lower_bound must be positive")
    if current_rate != coupling * cosine * phase_gap / 2:
        raise ValueError(
            "phase_decay_rate_lower_bound is inconsistent with K*c*lambda/2"
        )

    pi_lower, pi_upper = _pi_bounds()
    ratio_bounds = (1 / pi_upper, 1 / (pi_lower * cosine))
    pressure_mobility = tuple(coupling * value for value in ratio_bounds)
    mean_limit = _pressure_response_mean_limit(envelope, coupling)
    return CycleRestoringResponseComparison(
        coupling_strength=coupling,
        cosine_lower_bound=cosine,
        phase_laplacian_gap_lower_bound=phase_gap,
        pressure_to_current_ratio_bounds=ratio_bounds,
        current_response_mobility_bounds=(coupling, coupling),
        pressure_response_mobility_bounds=pressure_mobility,
        current_decay_rate_lower_bound=current_rate,
        pressure_class_decay_rate_lower_bound=pressure_mobility[0]
        * cosine
        * phase_gap
        / 2,
        pressure_linear_decay_rate_lower_bound=pressure_mobility[0] * phase_gap / 2,
        pressure_response_mean_limit_affine=mean_limit,
        pressure_response_mean_limit_enclosure=_affine_interval(
            *mean_limit, (pi_lower, pi_upper)
        ),
    )


def _cycle_response_inputs(envelope):
    """Check the shared initial scalar state and affine phase coordinates.

    These checks retain coefficient, vector and coordinate consistency;
    they do not authenticate a caller-created envelope or a live state.
    """
    capacity = exact_or_represented_real(envelope.capacity, "capacity")
    epi_weight = exact_or_represented_real(envelope.epi_weight, "epi_weight")
    phase_weight = exact_or_represented_real(envelope.phase_weight, "phase_weight")
    mean = exact_or_represented_real(envelope.weighted_mean, "weighted_mean")
    if capacity <= 0 or epi_weight <= 0:
        raise ValueError("capacity and epi_weight must be positive")
    if phase_weight < 0:
        raise ValueError("phase_weight must be nonnegative")
    source = envelope.capture.snapshot
    size = len(source.nodes)
    indices = envelope.cycle_indices
    if (
        not 3 <= size <= 12
        or len(indices) != size
        or any(isinstance(i, bool) or not isinstance(i, Integral) for i in indices)
        or set(indices) != set(range(size))
        or tuple(source.nodes[i] for i in indices) != tuple(envelope.cycle_order)
    ):
        raise ValueError(
            "cycle_indices must align every captured node with cycle_order"
        )
    strengths = tuple(
        exact_or_represented_real(value, "strength") for value in envelope.strengths
    )
    if len(strengths) != size or any(value <= 0 for value in strengths):
        raise ValueError("strengths must contain one positive value per captured node")
    epi = tuple(
        exact_or_represented_real(value, "captured EPI") for value in source.epi
    )
    total_strength = sum(strengths, Fraction(0))
    if (
        len(epi) != size
        or mean
        != sum(
            (strength * value for strength, value in zip(strengths, epi)), Fraction(0)
        )
        / total_strength
    ):
        raise ValueError(
            "weighted_mean is inconsistent with strengths and captured EPI"
        )
    if len(envelope.gap_affine) != size or any(
        len(row) != 2 for row in envelope.gap_affine
    ):
        raise ValueError("gap_affine must contain one affine-pi pair per cycle node")
    gaps = tuple(
        tuple(exact_or_represented_real(value, "gap coefficient") for value in row)
        for row in envelope.gap_affine
    )
    mean_gap = exact_or_represented_real(
        envelope.mean_gap_pi_coefficient, "mean_gap_pi_coefficient"
    )
    if (
        isinstance(envelope.winding, bool)
        or not isinstance(envelope.winding, Integral)
        or mean_gap != Fraction(2 * int(envelope.winding), size)
        or sum(row[0] for row in gaps) != 0
        or sum(row[1] for row in gaps) != size * mean_gap
    ):
        raise ValueError(
            "gap_affine and mean_gap_pi_coefficient must retain the winding"
        )
    shape = tuple(
        tuple(exact_or_represented_real(value, "shape coefficient") for value in row)
        for row in envelope.initial_phase_shape_affine
    )
    if any(sum(row[j] for row in shape) for j in (0, 1)) or any(
        shape[(i + 1) % size][j] - shape[i][j] != gaps[i][j] - (mean_gap if j else 0)
        for i in range(size)
        for j in (0, 1)
    ):
        raise ValueError(
            "initial phase shape must be centered and reconstruct the gaps"
        )
    return capacity, epi_weight, phase_weight, mean, strengths, shape


def _pressure_response_mean_limit(envelope, coupling):
    """Derive one conditional affine-pi label from shared checked inputs."""
    capacity, _, phase_weight, mean, strengths, shape = _cycle_response_inputs(envelope)
    factor = capacity * phase_weight / (coupling * sum(strengths, Fraction(0)))
    weighted_shape = tuple(
        sum(
            (
                strengths[index] * shape[i][j]
                for i, index in enumerate(envelope.cycle_indices)
            ),
            Fraction(0),
        )
        for j in (0, 1)
    )
    return mean - factor * weighted_shape[0], -factor * weighted_shape[1]


def _ordered_cycle(source, cycle_order):
    if isinstance(cycle_order, (str, bytes, bytearray, Mapping, Set)):
        raise TypeError("cycle_order must be an ordered sequence of graph nodes")
    try:
        order = tuple(islice(iter(cycle_order), 13))
    except TypeError as exc:
        raise TypeError("cycle_order must be an ordered sequence") from exc
    size = len(source.nodes)
    if len(order) != size or len(set(order)) != size or set(order) != set(source.nodes):
        raise ValueError("cycle_order must contain each graph node exactly once")
    indices = {node: i for i, node in enumerate(source.nodes)}
    cycle = tuple(indices[node] for node in order)
    expected = {i: {cycle[j - 1], cycle[(j + 1) % size]} for j, i in enumerate(cycle)}
    if any(set(row) != expected[i] for i, row in enumerate(source.support_neighbors)):
        raise ValueError("support must be exactly the declared full cycle")
    conductance = {(i, j): value for i, j, value in source.conductance}
    if set(conductance) != {(i, j) for i, row in expected.items() for j in row}:
        raise ValueError("every cycle edge must have positive transport conductance")
    # Symmetry and positivity already have the shared snapshot validator as
    # their owner; no second weighted adjacency convention is introduced.
    return order, cycle


def _times(values):
    if isinstance(values, (str, bytes, bytearray, Mapping, Set)):
        raise TypeError("times must be an ordered sequence")
    try:
        raw = tuple(islice(iter(values), 258))
    except TypeError as exc:
        raise TypeError("times must be an ordered sequence") from exc
    if not 1 <= len(raw) <= 257:
        raise ValueError("times must contain 1 to 257 entries")
    times = tuple(exact_or_represented_real(value, "time") for value in raw)
    if any(value < 0 for value in times) or any(
        right < left for left, right in zip(times, times[1:])
    ):
        raise ValueError("times must be nonnegative and nondecreasing")
    return times


def _duhamel_upper(first_rate, second_rate, time, first_exp, second_exp):
    """Enclose the convolution of two decays without unstable cancellation."""
    if not time:
        return Fraction(0)
    slower_exp = first_exp if first_rate <= second_rate else second_exp
    simple = time * slower_exp[1]
    if first_rate == second_rate:
        return simple
    faster_exp = second_exp if first_rate <= second_rate else first_exp
    difference_bound = (slower_exp[1] - faster_exp[0]) / abs(first_rate - second_rate)
    return min(simple, difference_bound)


def _decay_bounds(exponent):
    """Reuse the shared exponential owner on outward dyadic exponents.

    Mathematical pi enclosures can give rates with large denominators. Round
    the nonnegative exponent down/up on a 2**-64 grid, and use decreasing
    exp(-x) to reverse the endpoints. This only widens an exact enclosure;
    64 bits is an arithmetic budget, not a physical approximation premise.
    The shared exponent <=4096 work limit remains unchanged.
    """
    if exponent < 0 or exponent > 4096:
        # Retain the shared validation and error contract.
        return _negative_exp_bounds(exponent)
    scale = 1 << 64
    numerator = exponent.numerator * scale
    lower_index, remainder = divmod(numerator, exponent.denominator)
    lower_exponent = Fraction(lower_index, scale)
    upper_exponent = Fraction(lower_index + bool(remainder), scale)
    lower_bounds = _negative_exp_bounds(lower_exponent)
    if not remainder:
        return lower_bounds
    return _negative_exp_bounds(upper_exponent)[0], lower_bounds[1]


def _relaxation_upper(initial, tail, rate, time):
    """Enclose tail+(initial-tail)*exp(-rate*time) with its signed factor."""
    if not time or initial == tail:
        return initial
    lower, upper = _decay_bounds(rate * time)
    difference = initial - tail
    return tail + difference * (upper if difference > 0 else lower)


def _integrated_relaxation_upper(initial, tail, rate, time):
    """Enclose the integral of the nonnegative exponential envelope."""
    if not time or initial == tail:
        return initial * time
    lower, upper = _decay_bounds(rate * time)
    difference = initial - tail
    # A positive coefficient needs the upper integral (lower exponential);
    # a negative coefficient needs the lower integral (upper exponential).
    decay = lower if difference > 0 else upper
    integrated = tail * time + difference * (1 - decay) / rate
    return min(integrated, max(initial, tail) * time)


def bound_cycle_capacity_forcing(
    envelope: CycleRelaxationEnvelope,
    *,
    capacity_bounds,
    phase_radius,
    times,
) -> CycleCapacityForcingBudget:
    """Bound a prospectively declared capacity interval on the retained cycle.

    Replace held capacity by arbitrary measurable ``lower<=nu_i(t)<=upper``
    in ``theta'=nu+K*J`` and ``x'=diag(nu)*(-e*L_W*x+w*g-f*L_U*nu)``.
    The supplied K and pressure weights stay fixed; capacity jumps do not
    directly change phase or form. This is an exact-model comparison, not
    identification of a native controller, numerical solver or writer trace.

    Let width=upper-lower, B=sqrt(n)*width and rho=phase_radius. On the acute
    chart, gamma=K*c*lambda_C/2 with c<=cos(rho). Centering the difference
    from the same-initial-phase constant-capacity sine reference removes
    common rotation. Strong monotonicity bounds its norm by
    sqrt(n)*width/(2*gamma), hence each gap differs by at most
    sqrt(n/2)*width/gamma. Adding that margin to the reference's invariant
    initial gap radius gives the strict prospective tube test. No extra
    bootstrap time, trajectory or selected capacity feedback is introduced.

    Once admitted, q=||delta-mean(delta)|| obeys q'<=-gamma*q+B. Its
    nonzero tail controls phase forcing; the capacity pressure contributes
    at most f*B. With actual conductance Laplacian B_W and strengths s,
    E=x.T*B_W*x/2 satisfies E'<=-r*E+drive, where
    r=e*lower*lambda_2(B_W)/max(s) and
    drive=max(s)*upper*(w*q_max/pi+f*B)^2/(2*e).
    Shared rational square-root, spectral and exponential owners enclose
    all displayed bounds. No fixed-capacity weighted mean is conserved here.

    Integrating the nonnegative source envelope gives A(t). Positivity of
    transport then bounds every form coordinate on the entire prefix by
    [min(x0)-upper*A(t), max(x0)+upper*A(t)]. This supports a sufficient
    finite clipping-inactivity check against separately declared rails;
    neither an all-time scalar bound nor future runtime admission follows.
    """
    if not isinstance(envelope, CycleRelaxationEnvelope):
        raise TypeError("envelope must be a CycleRelaxationEnvelope")
    if isinstance(capacity_bounds, (str, bytes, bytearray, Mapping, Set)):
        raise TypeError("capacity_bounds must be an ordered lower/upper pair")
    try:
        raw_bounds = tuple(islice(iter(capacity_bounds), 3))
    except TypeError as exc:
        raise TypeError("capacity_bounds must be an ordered lower/upper pair") from exc
    if len(raw_bounds) != 2:
        raise ValueError("capacity_bounds must contain exactly lower and upper")
    lower, upper = tuple(
        exact_or_represented_real(value, "capacity bound") for value in raw_bounds
    )
    if not 0 < lower <= upper:
        raise ValueError("capacity bounds must satisfy 0 < lower <= upper")
    rho = exact_or_represented_real(phase_radius, "phase_radius")
    evaluation_times = _times(times)
    coupling = exact_or_represented_real(
        envelope.coupling_strength, "coupling_strength"
    )
    gate = exact_or_represented_real(
        envelope.effective_phase_gate, "effective_phase_gate"
    )
    if coupling <= 0:
        raise ValueError("coupling_strength must be positive")
    pi_bounds = _pi_bounds()
    pi_lower = pi_bounds[0]
    if not 0 < rho < min(gate, pi_lower / 2):
        raise ValueError(
            "phase_radius must be strictly inside both the U3 gate and pi/2"
        )
    capacity, e, w, _, strengths, _ = _cycle_response_inputs(envelope)
    source, _, weights = _validated_forcing_decomposition(envelope.capture)
    _, cycle = _ordered_cycle(source, envelope.cycle_order)
    if cycle != tuple(envelope.cycle_indices):
        raise ValueError("cycle_indices differ from the captured support order")
    if any(value != capacity for value in source.capacity):
        raise ValueError("initial template must retain its declared common capacity")
    if e != weights["epi"] or w != weights["phase"]:
        raise ValueError(
            "template pressure weights differ from the captured coefficients"
        )
    f = weights["vf"]
    size = len(cycle)
    phases = tuple(
        exact_or_represented_real(value, "captured phase")
        for value in envelope.capture.phase
    )
    if len(phases) != size:
        raise ValueError("captured phases must match the support")
    gap_intervals = []
    deviations = []
    mean_coefficient = Fraction(2 * envelope.winding, size)
    for j, i in enumerate(cycle):
        difference = phases[cycle[(j + 1) % size]] - phases[i]
        turn = _oriented_turn(difference, pi_bounds)
        if turn is None or envelope.gap_affine[j] != (difference, 2 * turn):
            raise ValueError(
                "template gaps must equal the admitted captured phase lift"
            )
        gap_intervals.append(_affine_interval(difference, 2 * turn, pi_bounds))
        deviations.append(
            _affine_interval(difference, 2 * turn - mean_coefficient, pi_bounds)
        )
    rho0 = max(abs(value) for row in gap_intervals for value in row)
    q_squared = sum(
        (max(abs(value) for value in row) ** 2 for row in deviations), Fraction(0)
    )
    if rho0 != exact_or_represented_real(
        envelope.phase_radius_upper, "phase_radius_upper"
    ) or q_squared != exact_or_represented_real(
        envelope.initial_gap_deviation_squared_upper,
        "initial_gap_deviation_squared_upper",
    ):
        raise ValueError("template initial radius and gap bounds are inconsistent")

    phase_laplacian = tuple(
        tuple(
            Fraction(2 if i == j else -1 if j in source.support_neighbors[i] else 0)
            for j in range(size)
        )
        for i in range(size)
    )
    phase_gap, phase_uniform = _exact_real_laplacian_gap_lower_bound(phase_laplacian)
    transport = [[Fraction(0) for _ in range(size)] for _ in range(size)]
    actual_strengths = [Fraction(0) for _ in range(size)]
    for i, j, weight in source.conductance:
        actual_strengths[i] += weight
        transport[i][i] += weight
        transport[i][j] -= weight
    if tuple(actual_strengths) != strengths:
        raise ValueError("template strengths differ from its actual conductance")
    transport_gap, transport_uniform = _exact_real_laplacian_gap_lower_bound(
        tuple(tuple(row) for row in transport)
    )
    if not phase_uniform or not transport_uniform or min(phase_gap, transport_gap) <= 0:
        raise ValueError(
            "fixed-cycle phase and transport gaps must be strictly positive"
        )
    width = upper - lower
    capacity_difference = _exact_sqrt_upper(size * width**2)
    gamma = coupling * (1 - 2 * rho / pi_lower) * phase_gap / 2
    reference_deviation = _exact_sqrt_upper(Fraction(size, 2) * width**2) / gamma
    tube_radius = rho0 + reference_deviation
    admitted = tube_radius < rho
    q0 = _exact_sqrt_upper(q_squared)
    q_tail = capacity_difference / gamma
    q_max = max(q0, q_tail)
    forcing = w * q_max / pi_lower + f * capacity_difference
    energy_rate = e * lower * transport_gap / max(strengths)
    energy_drive = max(strengths) * upper * forcing**2 / (2 * e)
    energy_tail = energy_drive / energy_rate
    samples = []
    if admitted:
        for time in evaluation_times:
            q = _relaxation_upper(q0, q_tail, gamma, time)
            integrated_q = _integrated_relaxation_upper(q0, q_tail, gamma, time)
            integrated_source = (
                w * integrated_q / pi_lower + f * capacity_difference * time
            )
            form_offset = upper * integrated_source
            samples.append(
                CycleCapacityForcingSample(
                    time=time,
                    capacity_exposure_upper=capacity_difference * time,
                    gap_deviation_norm_upper=q,
                    phase_source_norm_upper=q / pi_lower,
                    non_epi_source_norm_upper=w * q / pi_lower
                    + f * capacity_difference,
                    epi_dirichlet_energy_upper=_relaxation_upper(
                        source.dirichlet_energy, energy_tail, energy_rate, time
                    ),
                    gap_deviation_integral_upper=integrated_q,
                    non_epi_source_integral_upper=integrated_source,
                    epi_interval=(
                        min(source.epi) - form_offset,
                        max(source.epi) + form_offset,
                    ),
                )
            )
    return CycleCapacityForcingBudget(
        capacity_bounds=(lower, upper),
        phase_radius=rho,
        initial_phase_radius_upper=rho0,
        capacity_difference_norm_upper=capacity_difference,
        initial_gap_deviation_norm_upper=q0,
        phase_decay_rate_lower_bound=gamma,
        phase_gap_tail_upper=q_tail,
        reference_gap_deviation_upper=reference_deviation,
        phase_tube_radius_upper=tube_radius,
        all_time_gap_norm_upper=q_max,
        transport_laplacian_gap_lower_bound=transport_gap,
        initial_epi_dirichlet_energy=source.dirichlet_energy,
        epi_energy_decay_rate_lower_bound=energy_rate,
        non_epi_source_norm_upper=forcing,
        epi_energy_drive_upper=energy_drive,
        epi_energy_tail_upper=energy_tail,
        admitted=admitted,
        admission_failure=(
            None if admitted else "prospective_reference_gap_tube_reaches_phase_radius"
        ),
        samples=tuple(samples),
    )


def bound_cycle_relaxation(
    graph, cycle_order, *, coupling_strength, times
) -> CycleRelaxationEnvelope:
    """Bound acute-sector maintenance and the driven form relaxation.

    On the fixed ordered cycle let delta_i be the exact wrapped forward gap.
    The declared continuous law is ``delta'=-K/2*L_cycle*sin(delta)``. A common
    positive capacity supplies the free angular rate and the EPI mobility.
    Initial gaps must be certified strictly acute and within the configured
    UM gate; their invariant interval then retains all edges in this model.

    Fresh pressure is ``-e*L_rw*x+w*g`` with
    ``g_i=(delta_i-delta_(i-1))/(2*pi)``. Actual positive conductances determine
    L_rw and strengths s. The exact quotient owner supplies the form decay
    rate in the strength metric; no unit-cycle replacement is made.
    The full configured mix is retained: fixed common capacity and the
    cycle's constant unique-support degree make the capacity/topology
    gradients identically zero, regardless of their coefficients.

    Every mathematical pi branch and initial enclosure reuses the certified
    midpoint owner. The positive rational cosine lower bound follows from
    ``cos(rho)>=1-2*rho/pi`` on the acute interval. Transcendental estimates
    and a floating eigensolver are absent from the proof arithmetic.

    Gamma, later events, clipping, capacity changes, controllers, binary64
    phase/pressure realization and numerical integration errors are outside
    this continuous conditional theorem. Their graph configuration is not
    audited. The source graph, histories and caches remain unchanged.
    """
    if graph.is_directed() or graph.is_multigraph() or not 3 <= len(graph) <= 12:
        raise ValueError(
            "cycle relaxation requires a simple undirected 3 to 12 node graph"
        )
    if any(left == right for left, right in graph.edges()):
        raise ValueError("cycle relaxation excludes self loops")
    evaluation_times = _times(times)
    coupling = exact_or_represented_real(coupling_strength, "coupling_strength")
    if coupling <= 0:
        raise ValueError("coupling_strength must be positive")
    capture = capture_non_epi_forcing(graph)
    source = capture.snapshot
    order, cycle = _ordered_cycle(source, cycle_order)
    size = len(cycle)
    capacity = source.capacity[0]
    if capacity <= 0 or any(value != capacity for value in source.capacity):
        raise ValueError("capacity must be common and strictly positive")
    weights = dict(capture.normalized_weights)
    e, w = weights["epi"], weights["phase"]
    if e <= 0:
        raise ValueError("cycle relaxation requires a positive EPI weight")
    if any(not 0 <= phase < Fraction.from_float(math.tau) for phase in capture.phase):
        raise ValueError("phases must be canonical represented values in [0,2*pi)")
    _, gate_float = resolve_u3_phase_limits(graph.graph, operator_code="UM")
    gate = Fraction.from_float(gate_float)
    pi_bounds = _pi_bounds()
    pi_lower = pi_bounds[0]
    affine, enclosures = [], []
    winding = 0
    for j, i in enumerate(cycle):
        difference = capture.phase[cycle[(j + 1) % size]] - capture.phase[i]
        turn = _oriented_turn(difference, pi_bounds)
        if turn is None:
            raise ValueError(
                "every cyclic gap must have a certified strictly acute lift"
            )
        interval = _affine_interval(difference, 2 * turn, pi_bounds)
        if max(abs(value) for value in interval) > gate:
            raise ValueError(
                "the configured UM gate does not admit every exact cyclic gap"
            )
        affine.append((difference, 2 * turn))
        enclosures.append(interval)
        winding += turn
    mean_coefficient = Fraction(2 * winding, size)
    deviations = tuple(
        _affine_interval(rational, coefficient - mean_coefficient, pi_bounds)
        for rational, coefficient in affine
    )
    q2 = sum((max(abs(value) for value in row) ** 2 for row in deviations), Fraction(0))
    rho = max(abs(value) for row in enclosures for value in row)
    cosine_lower = 1 - 2 * rho / pi_lower
    if cosine_lower <= 0:
        raise ValueError(
            "the rational pi enclosure did not resolve a positive acute margin"
        )
    phase_laplacian = tuple(
        tuple(
            Fraction(2 if i == j else (-1 if j in source.support_neighbors[i] else 0))
            for j in range(size)
        )
        for i in range(size)
    )
    phase_gap, uniform_preserved = _exact_real_laplacian_gap_lower_bound(
        phase_laplacian
    )
    if phase_gap <= 0 or not uniform_preserved:
        raise RuntimeError("the cycle phase gap certificate failed")
    gamma = coupling * cosine_lower * phase_gap / 2
    strengths = [Fraction(0) for _ in range(size)]
    laplacian = [[Fraction(0) for _ in range(size)] for _ in range(size)]
    for i, j, weight in source.conductance:
        strengths[i] += weight
        laplacian[i][i] += weight
        laplacian[i][j] -= weight
    strengths = tuple(strengths)
    beta, mean_preserved, consensus_preserved, uniform_preserved = (
        _exact_flow_gap_from_rationals(
            tuple(tuple(row) for row in laplacian),
            tuple(capacity * e / strength for strength in strengths),
            strengths,
        )
    )
    if beta <= 0 or not all((mean_preserved, consensus_preserved, uniform_preserved)):
        raise RuntimeError("the exact weighted EPI gap certificate failed")
    total_strength = sum(strengths, Fraction(0))
    mean = (
        sum(
            (strength * value for strength, value in zip(strengths, source.epi)),
            Fraction(0),
        )
        / total_strength
    )
    d0 = sum(
        (
            strength * (value - mean) ** 2
            for strength, value in zip(strengths, source.epi)
        ),
        Fraction(0),
    )
    a2 = q2 / pi_lower**2
    c2 = (capacity * w) ** 2 * max(strengths) * a2
    strength_gradient2 = sum(
        (
            (strengths[i] - strengths[cycle[(j + 1) % size]]) ** 2
            for j, i in enumerate(cycle)
        ),
        Fraction(0),
    )
    m2 = (
        (capacity * w) ** 2
        * strength_gradient2
        * q2
        / (4 * pi_lower**2 * total_strength**2)
    )
    driven_global = c2 / beta**2
    global_disagreement = (
        d0 + driven_global if not d0 or not driven_global else 2 * (d0 + driven_global)
    )
    mean_offset = _exact_sqrt_upper(m2) / gamma
    local_disagreement = _exact_sqrt_upper(global_disagreement / min(strengths))
    epi_interval = (
        mean - mean_offset - local_disagreement,
        mean + mean_offset + local_disagreement,
    )
    samples = []
    for time in evaluation_times:
        phase_exp = _decay_bounds(gamma * time)
        epi_exp = _decay_bounds(beta * time)
        phase_squared_exp = _decay_bounds(2 * gamma * time)[1]
        epi_squared_exp = _decay_bounds(2 * beta * time)[1]
        integral = _duhamel_upper(beta, gamma, time, epi_exp, phase_exp)
        homogeneous = d0 * epi_squared_exp
        driven = c2 * integral**2
        form_upper = (
            homogeneous + driven
            if not homogeneous or not driven
            else 2 * (homogeneous + driven)
        )
        samples.append(
            CycleRelaxationSample(
                time=time,
                gap_deviation_squared_upper=q2 * phase_squared_exp,
                phase_source_squared_upper=a2 * phase_squared_exp,
                epi_disagreement_squared_upper=form_upper,
                mean_displacement_squared_upper=m2 * ((1 - phase_exp[0]) / gamma) ** 2,
                mean_tail_squared_upper=m2 * phase_squared_exp / gamma**2,
                duhamel_integral_upper=integral,
            )
        )
    return CycleRelaxationEnvelope(
        capture=capture,
        cycle_indices=cycle,
        cycle_order=order,
        coupling_strength=coupling,
        capacity=capacity,
        epi_weight=e,
        phase_weight=w,
        effective_phase_gate=gate,
        gap_affine=tuple(affine),
        gap_enclosures=tuple(enclosures),
        winding=winding,
        mean_gap_pi_coefficient=mean_coefficient,
        gap_deviation_enclosures=deviations,
        phase_radius_upper=rho,
        cosine_lower_bound=cosine_lower,
        phase_laplacian_gap_lower_bound=phase_gap,
        phase_decay_rate_lower_bound=gamma,
        epi_decay_rate_lower_bound=beta,
        strengths=strengths,
        weighted_mean=mean,
        initial_epi_disagreement_squared=d0,
        initial_gap_deviation_squared_upper=q2,
        phase_source_prefactor_squared_upper=a2,
        form_forcing_prefactor_squared_upper=c2,
        mean_rate_prefactor_squared_upper=m2,
        all_time_epi_disagreement_squared_upper=global_disagreement,
        mean_limit_offset_upper=mean_offset,
        all_time_epi_interval=epi_interval,
        samples=tuple(samples),
    )
