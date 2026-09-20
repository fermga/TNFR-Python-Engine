"""Conditional continuous relaxation on a fixed acute weighted cycle.

The supplied equal-capacity averaged-sine phase law preserves its acute gap
sector. Its two-neighbor phasor source drives the canonical weighted EPI row.
This evaluator bounds that exact-real model from detached represented initial
data; it neither advances the graph nor certifies a binary64 trajectory.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Set
from dataclasses import dataclass
from fractions import Fraction
from itertools import islice

from .._exact_time import exact_or_represented_real
from ..mathematics._phase_midpoint import _affine_interval, _oriented_turn, _pi_bounds
from ..operators._phase_gate import resolve_u3_phase_limits
from .forcing_realization import NonEpiForcingObservation, capture_non_epi_forcing
from .hybrid_operator_stability import _exact_sqrt_upper
from .reversible_eigenmode_reference import _negative_exp_bounds
from .structural_diffusion import (
    _exact_flow_gap_from_rationals,
    _exact_real_laplacian_gap_lower_bound,
)

__all__ = [
    "CycleRelaxationEnvelope",
    "CycleRelaxationSample",
    "bound_cycle_relaxation",
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
