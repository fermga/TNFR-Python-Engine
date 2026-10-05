"""Replica structure and phase-pair observations for the complete sine law.

The supplied fine graph and all of its coordinates remain present. Block means
are an additional description, not replacement nodes or an installed quotient
law. Their nonlinear rates retain internal form and phase organization. Pure
phase grouping, local complete-law transitions and conservative finite-window
enclosures have separate admission and evidence scopes.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from typing import Any

from .._exact_time import exact_or_represented_real
from ..dynamics.relational import RelationalExchangeModel
from ..mathematics._phase_resultant_chamber import (
    certified_cosine_bounds,
    certified_sine_bounds,
    relative_resultant_bounds,
)
from ..mathematics._rational_interval import (
    INTERVAL_METHOD,
    I,
    cos,
    pi_interval,
    sin,
    sqrt,
)
from ._sine_admission import _sine_model_coefficients
from .phase_cycle_geometry import PhaseCycleState
from .relational_observations import _ordered
from .relational_sine_comparison import (
    SineExchangeComparison,
    SineMobilityComparison,
    SineMobilityRelativeBalance,
    _comparison_neighbors,
    _sine_phase_rate_numerators,
    _sine_rates,
    _validate_comparison_labels,
    bound_relational_sine_exchange,
)

__all__ = (
    "SineReplicaScaleAssessment",
    "SineReplicaPersistenceAssessment",
    "SineMobilityGeometryAssessment",
    "SineReplicaCapacityAssessment",
    "SineReplicaEquilibriaAssessment",
    "PhasePairObservation",
    "JointPairObservation",
    "SineStatePairingAssessment",
    "SinePairingTransitionAssessment",
    "SinePairingMobilityAssessment",
    "SinePairingWindowAssessment",
    "SineJointPairingWindowAssessment",
    "SineJointPairingProjection",
    "SinePairSupportSymmetryAssessment",
    "SineMixedPairStateAssessment",
    "SinePairEmissionAssessment",
    "SinePairEmissionOutcome",
    "SineReplicaPulseAssessment",
    "SineReplicaPulseVariation",
    "SineReplicaPulseSplitting",
    "assess_sine_replica_scale",
    "assess_sine_replica_persistence",
    "assess_sine_mobility_geometry",
    "assess_sine_replica_capacity",
    "assess_sine_replica_equilibria",
    "observe_phase_pairs",
    "observe_joint_pairs",
    "assess_sine_state_pairing",
    "assess_sine_pairing_transition",
    "assess_sine_pairing_mobility",
    "assess_sine_pairing_window",
    "assess_sine_joint_pairing_window",
    "assess_sine_pair_support_symmetry",
    "assess_sine_mixed_pair_state",
    "assess_sine_pair_emission",
    "assess_sine_replica_pulse",
    "assess_sine_replica_pulse_variation",
    "assess_sine_replica_pulse_splitting",
)


@dataclass(frozen=True)
class SineReplicaScaleAssessment:
    """Detached full-state replica coordinates and ideal instantaneous rates.

    Phase coordinates are split into exact rational radian and full-turn parts.
    ``phase_turns`` are supplied lifts in captured fine node order; no branch is
    inferred. The admitted local pair chart makes cos(delta) the nonnegative
    circular resultant magnitude. The chart need not persist outside the exact
    synchronized sector. Rate and storage residuals are computed enclosures;
    their containment of zero is distinct from the all-state algebraic proof.

    The unordered-pair coordinates are R=cos(delta), U=u**2 and Q=u*sin(delta).
    Together with the means they close on the realized local pair chart and
    identify states up to independent within-pair swaps. Their constraint does
    not remove continuous state dimension. ``all_state_coarse_closure_obstructed``
    concerns means alone, not this complete invariant description. The ordered
    fine provenance is retained, so the whole report is not swap invariant.

    ``poisson_coefficients`` holds the four independent entries identified by
    ``poisson_coefficient_pairs`` in each pair's inherited bracket. Reverse
    entries have opposite signs; other entries and cross-pair brackets vanish.
    The retained full-storage gradient generates all five coordinate rates.
    Its residuals compare with the direct fine-field pushforward, while the
    Casimir residuals evaluate the bracket on the existing constraint gradient.
    These are outward checks of the conditional algebra, not admission of
    arbitrary invariant tuples, a new Hamiltonian or a time discretization.
    """

    comparison: SineExchangeComparison
    pairs: tuple[tuple[Any, Any], ...]
    base_edges: tuple[tuple[int, int], ...]
    base_neighbors: tuple[tuple[int, ...], ...]
    base_degrees: tuple[int, ...]
    pair_capacity: tuple[Q, ...]
    phase_turns: tuple[int, ...]
    form_means: tuple[Q, ...]
    form_half_differences: tuple[Q, ...]
    phase_mean_radian_parts: tuple[Q, ...]
    phase_mean_turn_parts: tuple[Q, ...]
    phase_half_difference_radian_parts: tuple[Q, ...]
    phase_half_difference_turn_parts: tuple[Q, ...]
    phase_half_difference_bounds: tuple[I, ...]
    phase_chart_margins: tuple[I, ...]
    resultant_magnitude_bounds: tuple[I, ...]
    resultant_magnitude_rate_bounds: tuple[I, ...]
    internal_form_squared: tuple[Q, ...]
    form_phase_correlation_bounds: tuple[I, ...]
    internal_restoring_coefficients: tuple[I, ...]
    internal_phase_coefficients: tuple[I, ...]
    internal_form_squared_rates: tuple[I, ...]
    form_phase_correlation_rates: tuple[I, ...]
    closed_resultant_magnitude_rates: tuple[I, ...]
    closed_internal_form_squared_rates: tuple[I, ...]
    closed_form_phase_correlation_rates: tuple[I, ...]
    resultant_rate_closure_residual: tuple[I, ...]
    internal_form_squared_rate_closure_residual: tuple[I, ...]
    form_phase_correlation_rate_closure_residual: tuple[I, ...]
    internal_constraint_residual: tuple[I, ...]
    internal_constraint_rate_residual: tuple[I, ...]
    phase_current_factors: tuple[I, ...]
    mean_form_rates: tuple[I, ...]
    mean_phase_rates: tuple[I, ...]
    internal_form_rates: tuple[I, ...]
    internal_phase_rates: tuple[I, ...]
    factored_mean_form_rates: tuple[I, ...]
    factored_mean_phase_rates: tuple[I, ...]
    factored_internal_form_rates: tuple[I, ...]
    factored_internal_phase_rates: tuple[I, ...]
    mean_form_factorization_residual: tuple[I, ...]
    mean_phase_factorization_residual: tuple[I, ...]
    internal_form_factorization_residual: tuple[I, ...]
    internal_phase_factorization_residual: tuple[I, ...]
    naive_form_rates: tuple[I, ...]
    naive_phase_rates: tuple[I, ...]
    coarse_form_defect: tuple[I, ...]
    coarse_phase_defect: tuple[I, ...]
    coarse_form_storage: Q
    coarse_phase_storage: I
    coarse_storage: I
    internal_form_storage: Q
    internal_phase_storage_correction: I
    storage_decomposition_residual: I
    unordered_storage_bounds: I
    unordered_storage_rate_bounds: I
    unordered_storage_residual: I
    unordered_storage_rate_residual: I
    source_in_synchronized_submanifold: bool
    same_law_reduced_flow_certified_for_source: bool
    synchronized_submanifold_invariant: bool = True
    all_state_coarse_closure_obstructed: bool = True
    unordered_pair_state_closure_certified: bool = True
    unordered_pair_state_identifies_swap_orbits: bool = True
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "complete_supplied_double_replica_support_no_fiber_edges_all_four_cross_edges",
        "conservative_normalized_sine_law_exact_zero_loss_positive_held_paired_capacity",
        "all_fine_coordinates_support_and_internal_differences_remain_present",
        "explicit_pair_chart_and_supplied_phase_turns_not_automatic_unwrapping",
        "local_circular_pair_chart_is_not_a_future_chart_certificate",
        "same_law_inheritance_on_the_exact_synchronized_invariant_submanifold",
        "block_means_do_not_close_the_full_nonlinear_state_family",
        "means_and_realized_R_U_Q_close_while_the_local_pair_chart_persists",
        "invariants_identify_unordered_pairs_not_fewer_continuous_degrees_of_freedom",
        "no_arbitrary_invariant_tuple_admission_or_division_at_synchronized_phase",
        "zero_snapshot_defect_does_not_certify_synchronized_invariant_membership",
        "internal_coherence_factors_are_derived_current_factors_not_new_live_edge_weights",
        "storage_correction_may_be_signed_not_a_declared_reservoir_or_dissipation",
        "no_graph_write_node_deletion_input_support_event_solver_or_native_dispatch",
        "no_pattern_formation_selection_universal_scale_law_or_physical_identification",
        "sparse_Poisson_bracket_is_inherited_from_the_same_conservative_fine_law",
        "full_fine_storage_generates_the_unordered_rates_on_the_realized_pair_chart",
        "Casimir_annihilation_does_not_admit_arbitrary_independent_invariant_intervals",
    )
    poisson_coordinate_order: tuple[str, ...] = ("X", "Theta", "R", "U", "Q")
    poisson_coefficient_pairs: tuple[tuple[str, str], ...] = (
        ("X", "Theta"),
        ("R", "U"),
        ("R", "Q"),
        ("U", "Q"),
    )
    poisson_coefficients: tuple[tuple[I, ...], ...] = ()
    unordered_storage_gradients: tuple[tuple[I, ...], ...] = ()
    poisson_generated_rates: tuple[tuple[I, ...], ...] = ()
    poisson_rate_residuals: tuple[tuple[I, ...], ...] = ()
    poisson_casimir_residuals: tuple[tuple[I, ...], ...] = ()

    def to_dict(self):
        from ..sdk.relational_reports import _project

        _validate_replica_labels(self)
        return {
            "schema": "tnfr.relational-sine-replica-scale.v1",
            "report": _project(self),
        }


def _validate_replica_labels(replica):
    from ..sdk.relational_reports import _validate_label

    _validate_comparison_labels(replica.comparison)
    for pair in replica.pairs:
        for node in pair:
            _validate_label(node)


class _ReplicaAdmissionError(ValueError):
    """Expected refusal of supplied paired support, capacity or lift chart."""


def _replica_partition(comparison, pairs):
    """Admit an explicit ordered pair partition without imposing a support law."""
    nodes = comparison.nodes
    size = len(nodes)
    raw = _ordered(pairs, "pairs", limit=size + 1)
    if 2 * len(raw) != size:
        raise ValueError("pairs must partition all fine nodes into pairs")
    pairs = tuple(_ordered(pair, "each pair", limit=3) for pair in raw)
    if any(len(pair) != 2 for pair in pairs):
        raise ValueError("each ordered pair must contain exactly two fine nodes")
    flat = tuple(node for pair in pairs for node in pair)
    if len(set(flat)) != size or set(flat) != set(nodes):
        raise ValueError("pairs must contain every captured fine node exactly once")
    positions = {node: i for i, node in enumerate(nodes)}
    indices = tuple(tuple(positions[node] for node in pair) for pair in pairs)
    return pairs, indices


def _replica_support(comparison, pairs, *, require_equal_capacity=True):
    """Admit the explicit paired partition and complete bipartite fibers."""
    pairs, indices = _replica_partition(comparison, pairs)
    if len(pairs) < 2:
        raise _ReplicaAdmissionError("replica support requires at least two pairs")
    block = {node: i for i, pair in enumerate(pairs) for node in pair}
    edge_counts = {}
    for left, right in comparison.edges:
        i, j = sorted((block[left], block[right]))
        if i == j:
            raise _ReplicaAdmissionError(
                "within-pair edges are outside the replica support domain"
            )
        edge_counts[i, j] = edge_counts.get((i, j), 0) + 1
    if any(count != 4 for count in edge_counts.values()):
        raise _ReplicaAdmissionError(
            "each active base edge must contain all four fine cross-edges"
        )
    edges = tuple(sorted(edge_counts))
    neighbors = [[] for _ in pairs]
    for i, j in edges:
        neighbors[i].append(j)
        neighbors[j].append(i)
    # Shared capture already proved that the fine support is connected. A full
    # partition with complete bipartite fibers inherits connected base support.
    degrees = tuple(map(len, neighbors))
    capacity = []
    for i, j in indices:
        nu, other = comparison.capacity[i], comparison.capacity[j]
        if nu <= 0 or other <= 0 or (require_equal_capacity and nu != other):
            raise _ReplicaAdmissionError(
                "each pair requires "
                + ("the same " if require_equal_capacity else "")
                + "strictly positive held capacity"
            )
        capacity.append((nu + other) / 2)
    return pairs, indices, edges, tuple(map(tuple, neighbors)), degrees, tuple(capacity)


def _pi_shifted_trig(radians, pi_coefficient):
    """Retain exact integer-pi parity before interval trigonometry."""
    sign = -1 if pi_coefficient % 2 else 1
    return (
        sign * I(*certified_sine_bounds(radians)),
        sign * I(*certified_cosine_bounds(radians)),
    )


def _replica_phase_turns(phase_turns, size):
    turns = (
        (0,) * size
        if phase_turns is None
        else _ordered(phase_turns, "phase_turns", limit=size + 1)
    )
    if len(turns) != size or any(type(value) is not int for value in turns):
        raise ValueError(
            "phase_turns must contain one nonboolean integer per fine node"
        )
    return turns


def _apply_replica_poisson(coefficients, gradient):
    """Apply the inherited sparse skew block in (X, Theta, R, U, Q) order."""
    xt, ru, rq, uq = coefficients
    x, theta, r, u, q = gradient
    return (
        xt * theta,
        -xt * x,
        ru * u + rq * q,
        -ru * r + uq * q,
        -rq * r - uq * u,
    )


def _pair_chart_coordinates(comparison, indices, phase_turns):
    """Observe signed pair coordinates without imposing a replica support law."""
    size = len(comparison.nodes)
    turns = _replica_phase_turns(phase_turns, size)

    epi, phase = comparison.epi, comparison.phase
    means = tuple((epi[i] + epi[j]) / 2 for i, j in indices)
    internal = tuple((epi[i] - epi[j]) / 2 for i, j in indices)
    mean_radians = tuple((phase[i] + phase[j]) / 2 for i, j in indices)
    mean_turns = tuple(Q(turns[i] + turns[j], 2) for i, j in indices)
    internal_radians = tuple((phase[i] - phase[j]) / 2 for i, j in indices)
    internal_turns = tuple(Q(turns[i] - turns[j], 2) for i, j in indices)
    pi = pi_interval()
    delta = tuple(
        I(raw) + (2 * turn) * pi for raw, turn in zip(internal_radians, internal_turns)
    )
    chart_margins = tuple(pi / 2 - abs(value) for value in delta)
    if any(margin.lo <= 0 for margin in chart_margins):
        raise _ReplicaAdmissionError(
            "supplied pair phase lifts need certified absolute half-gaps < pi/2"
        )
    internal_trig = tuple(
        _pi_shifted_trig(raw, int(2 * turn))
        for raw, turn in zip(internal_radians, internal_turns)
    )
    sine_delta = tuple(sine for sine, _ in internal_trig)
    coherence = tuple(cosine for _, cosine in internal_trig)
    return dict(
        turns=turns,
        means=means,
        internal=internal,
        mean_radians=mean_radians,
        mean_turns=mean_turns,
        internal_radians=internal_radians,
        internal_turns=internal_turns,
        delta=delta,
        chart_margins=chart_margins,
        sine_delta=sine_delta,
        coherence=coherence,
    )


def _replica_coordinates(
    comparison, pairs, phase_turns, *, require_equal_capacity=True
):
    """Shared single-capture support, signed coordinates and supplied lift chart."""
    pairs, indices, edges, neighbors, degrees, capacity = _replica_support(
        comparison, pairs, require_equal_capacity=require_equal_capacity
    )
    chart = _pair_chart_coordinates(comparison, indices, phase_turns)
    radians, turns = chart["mean_radians"], chart["mean_turns"]
    edge_trig = tuple(
        _pi_shifted_trig(radians[j] - radians[i], int(2 * (turns[j] - turns[i])))
        for i, j in edges
    )
    return dict(
        pairs=pairs,
        indices=indices,
        edges=edges,
        neighbors=neighbors,
        degrees=degrees,
        capacity=capacity,
        **chart,
        edge_trig=edge_trig,
    )


def assess_sine_replica_scale(graph, *, reference_model, pairs, phase_turns=None):
    """Assess conservative same-law inheritance without removing fine nodes.

    ``pairs`` orders every fine node exactly once as (plus, minus). Each active
    block edge must contain all four unit cross-edges; paired capacities agree
    exactly and are positive. The reference must explicitly have zero form loss.

    ``phase_turns`` supplies integer full turns in captured fine node order.
    Their exact lifted within-pair gap must be strictly shorter than pi. A
    missing or uncertified chart is rejected, never repaired by unwrapping.
    On this chart, X,Theta are pair means and u,delta their half differences.
    The resultant magnitude R=cos(delta) changes the collective phase current
    through R_i*R_j. These factors are calculated from retained internal state.

    Fine rates and work come from the shared comparison owner. Independent
    trigonometric factorization is checked against their means/differences.
    The naive base law is exact on u=delta=0 for all time, but a zero observed
    defect outside that invariant sector supplies no autonomous reduced model.
    Retaining R=cos(delta), U=u**2 and Q=u*sin(delta) instead gives an exact
    constrained description modulo pair swaps, with no singular division at
    R=1. This pushforward is assessed only on the captured realizable source;
    it neither replaces the fine state nor admits arbitrary proposed tuples.
    """
    comparison = bound_relational_sine_exchange(graph, reference_model=reference_model)
    if Q(reference_model.effective_weights[0]) != 0:
        raise ValueError(
            "replica inheritance assessment requires explicit zero form loss"
        )
    data = _replica_coordinates(comparison, pairs, phase_turns)
    (
        pairs,
        indices,
        edges,
        neighbors,
        degrees,
        capacity,
        turns,
        means,
        internal,
        mean_radians,
        mean_turns,
        internal_radians,
        internal_turns,
        delta,
        chart_margins,
        sine_delta,
        coherence,
        edge_trig,
    ) = data.values()
    factors = tuple(coherence[i] * coherence[j] for i, j in edges)
    naive_currents = [I(0) for _ in pairs]
    factored_currents = [I(0) for _ in pairs]
    internal_currents = [I(0) for _ in pairs]
    restoring_currents = [I(0) for _ in pairs]
    coarse_phase_storage = I(0)
    internal_phase_storage = I(0)
    beta = Q(reference_model.storage_scale)
    for (i, j), (sine, cosine), factor in zip(edges, edge_trig, factors):
        naive_currents[i] += sine
        naive_currents[j] -= sine
        factored_currents[i] += factor * sine
        factored_currents[j] -= factor * sine
        internal_currents[i] -= sine_delta[i] * coherence[j] * cosine
        internal_currents[j] -= sine_delta[j] * coherence[i] * cosine
        restoring_currents[i] += coherence[j] * cosine
        restoring_currents[j] += coherence[i] * cosine
        coarse_phase_storage += 1 - cosine
        internal_phase_storage += 4 * beta * cosine * (1 - factor)
    gradient = tuple(
        sum((means[i] - means[j] for j in row), Q(0)) for i, row in enumerate(neighbors)
    )
    naive = _sine_rates(reference_model, degrees, gradient, capacity, naive_currents)
    factored_mean = _sine_rates(
        reference_model, degrees, gradient, capacity, factored_currents
    )
    factored_internal = _sine_rates(
        reference_model,
        degrees,
        tuple(degree * value for degree, value in zip(degrees, internal)),
        capacity,
        internal_currents,
    )
    # The same normalized row maps the restoring sums and a unit internal
    # form direction to A_i and c_i. No extra constitutive law is introduced.
    coefficients = _sine_rates(
        reference_model, degrees, degrees, capacity, restoring_currents
    )
    restoring = coefficients["form_rates"]
    internal_phase_coefficients = coefficients["phase_rates"]
    mean_form = tuple(
        (comparison.form_rates[i] + comparison.form_rates[j]) / 2 for i, j in indices
    )
    mean_phase = tuple(
        (comparison.phase_rates[i] + comparison.phase_rates[j]) / 2 for i, j in indices
    )
    internal_form = tuple(
        (comparison.form_rates[i] - comparison.form_rates[j]) / 2 for i, j in indices
    )
    internal_phase = tuple(
        (comparison.phase_rates[i] - comparison.phase_rates[j]) / 2 for i, j in indices
    )
    coherence_rates = tuple(
        -sine * rate for sine, rate in zip(sine_delta, internal_phase)
    )
    squared = tuple(value**2 for value in internal)
    correlation = tuple(value * sine for value, sine in zip(internal, sine_delta))
    squared_rates = tuple(
        2 * value * rate for value, rate in zip(internal, internal_form)
    )
    correlation_rates = tuple(
        form_rate * sine + value * cosine * phase_rate
        for form_rate, sine, value, cosine, phase_rate in zip(
            internal_form, sine_delta, internal, coherence, internal_phase
        )
    )
    closed_coherence_rates = tuple(
        -c * q for c, q in zip(internal_phase_coefficients, correlation)
    )
    closed_squared_rates = tuple(-2 * a * q for a, q in zip(restoring, correlation))
    closed_correlation_rates = tuple(
        -a * (1 - r**2) + c * u_squared * r
        for a, r, c, u_squared in zip(
            restoring, coherence, internal_phase_coefficients, squared
        )
    )
    constraint_residual = tuple(
        q**2 - u_squared * (1 - r**2)
        for q, u_squared, r in zip(correlation, squared, coherence)
    )
    constraint_rate_residual = tuple(
        2 * q * q_rate - u_rate * (1 - r**2) + 2 * u_squared * r * r_rate
        for q, q_rate, u_rate, r, u_squared, r_rate in zip(
            correlation,
            closed_correlation_rates,
            closed_squared_rates,
            coherence,
            squared,
            closed_coherence_rates,
        )
    )
    # Fine degree is 2*d_i. Taking means/half-differences adds another 1/2,
    # so the inherited coordinate mobility is b*nu_i/(4*d_i), not b*nu_i/d_i.
    poisson = tuple(
        (-kappa, -2 * kappa * q, -kappa * (1 - r**2), -2 * kappa * u_squared * r)
        for c, degree, r, u_squared, q in zip(
            internal_phase_coefficients, degrees, coherence, squared, correlation
        )
        for kappa in (c / (4 * degree),)
    )
    storage_gradients = tuple(
        (I(4 * q), -4 * beta * current, -4 * beta * restoring_sum, I(2 * degree), I(0))
        for q, current, restoring_sum, degree in zip(
            gradient, factored_currents, restoring_currents, degrees
        )
    )
    poisson_rates = tuple(
        _apply_replica_poisson(block, grad)
        for block, grad in zip(poisson, storage_gradients)
    )
    direct_rates = tuple(
        zip(mean_form, mean_phase, coherence_rates, squared_rates, correlation_rates)
    )
    poisson_residuals = tuple(
        tuple(generated - direct for generated, direct in zip(row, fine_row))
        for row, fine_row in zip(poisson_rates, direct_rates)
    )
    casimir_residuals = tuple(
        _apply_replica_poisson(
            block, (I(0), I(0), 2 * u_squared * r, -(1 - r**2), 2 * q)
        )
        for block, r, u_squared, q in zip(poisson, coherence, squared, correlation)
    )
    coarse_form_storage = sum(((means[i] - means[j]) ** 2 / 2 for i, j in edges), Q(0))
    coarse_storage = coarse_form_storage + beta * coarse_phase_storage
    internal_form_storage = 2 * sum(
        (degree * value**2 for degree, value in zip(degrees, internal)), Q(0)
    )
    unordered_storage = I(
        4 * coarse_form_storage
        + 2 * sum((degree * value for degree, value in zip(degrees, squared)), Q(0))
    )
    unordered_storage_rate = 2 * sum(
        (degree * rate for degree, rate in zip(degrees, closed_squared_rates)), I(0)
    )
    for (i, j), (sine, cosine), factor in zip(edges, edge_trig, factors):
        unordered_storage += 4 * beta * (1 - factor * cosine)
        unordered_storage_rate += 4 * (means[i] - means[j]) * (
            factored_mean["form_rates"][i] - factored_mean["form_rates"][j]
        ) + 4 * beta * (
            factor
            * sine
            * (factored_mean["phase_rates"][j] - factored_mean["phase_rates"][i])
            - cosine
            * (
                closed_coherence_rates[i] * coherence[j]
                + coherence[i] * closed_coherence_rates[j]
            )
        )
    synchronized = all(value == 0 for value in internal) and all(
        raw == 0 and turn == 0 for raw, turn in zip(internal_radians, internal_turns)
    )

    def residual(left, right):
        return tuple(x - y for x, y in zip(left, right))

    return SineReplicaScaleAssessment(
        comparison=comparison,
        pairs=pairs,
        base_edges=edges,
        base_neighbors=neighbors,
        base_degrees=degrees,
        pair_capacity=capacity,
        phase_turns=turns,
        form_means=means,
        form_half_differences=internal,
        phase_mean_radian_parts=mean_radians,
        phase_mean_turn_parts=mean_turns,
        phase_half_difference_radian_parts=internal_radians,
        phase_half_difference_turn_parts=internal_turns,
        phase_half_difference_bounds=delta,
        phase_chart_margins=chart_margins,
        resultant_magnitude_bounds=coherence,
        resultant_magnitude_rate_bounds=coherence_rates,
        internal_form_squared=squared,
        form_phase_correlation_bounds=correlation,
        internal_restoring_coefficients=restoring,
        internal_phase_coefficients=internal_phase_coefficients,
        internal_form_squared_rates=squared_rates,
        form_phase_correlation_rates=correlation_rates,
        closed_resultant_magnitude_rates=closed_coherence_rates,
        closed_internal_form_squared_rates=closed_squared_rates,
        closed_form_phase_correlation_rates=closed_correlation_rates,
        resultant_rate_closure_residual=residual(
            coherence_rates, closed_coherence_rates
        ),
        internal_form_squared_rate_closure_residual=residual(
            squared_rates, closed_squared_rates
        ),
        form_phase_correlation_rate_closure_residual=residual(
            correlation_rates, closed_correlation_rates
        ),
        internal_constraint_residual=constraint_residual,
        internal_constraint_rate_residual=constraint_rate_residual,
        phase_current_factors=factors,
        mean_form_rates=mean_form,
        mean_phase_rates=mean_phase,
        internal_form_rates=internal_form,
        internal_phase_rates=internal_phase,
        factored_mean_form_rates=factored_mean["form_rates"],
        factored_mean_phase_rates=factored_mean["phase_rates"],
        factored_internal_form_rates=factored_internal["form_rates"],
        factored_internal_phase_rates=factored_internal["phase_rates"],
        mean_form_factorization_residual=residual(
            mean_form, factored_mean["form_rates"]
        ),
        mean_phase_factorization_residual=residual(
            mean_phase, factored_mean["phase_rates"]
        ),
        internal_form_factorization_residual=residual(
            internal_form, factored_internal["form_rates"]
        ),
        internal_phase_factorization_residual=residual(
            internal_phase, factored_internal["phase_rates"]
        ),
        naive_form_rates=naive["form_rates"],
        naive_phase_rates=naive["phase_rates"],
        coarse_form_defect=residual(mean_form, naive["form_rates"]),
        coarse_phase_defect=residual(mean_phase, naive["phase_rates"]),
        coarse_form_storage=coarse_form_storage,
        coarse_phase_storage=coarse_phase_storage,
        coarse_storage=coarse_storage,
        internal_form_storage=internal_form_storage,
        internal_phase_storage_correction=internal_phase_storage,
        storage_decomposition_residual=(
            comparison.storage
            - 4 * coarse_storage
            - internal_form_storage
            - internal_phase_storage
        ),
        unordered_storage_bounds=unordered_storage,
        unordered_storage_rate_bounds=unordered_storage_rate,
        unordered_storage_residual=comparison.storage - unordered_storage,
        unordered_storage_rate_residual=comparison.storage_rate
        - unordered_storage_rate,
        poisson_coefficients=poisson,
        unordered_storage_gradients=storage_gradients,
        poisson_generated_rates=poisson_rates,
        poisson_rate_residuals=poisson_residuals,
        poisson_casimir_residuals=casimir_residuals,
        source_in_synchronized_submanifold=synchronized,
        same_law_reduced_flow_certified_for_source=synchronized,
    )


@dataclass(frozen=True)
class SineReplicaPersistenceAssessment:
    """Whole-state conservative trapping with independently active constituents.

    The declared open family has centered full-state norm below ``radius``,
    excess storage below ``excess_ceiling``, form mean in the open supplied
    slab, and no synchronized pair tip. Its invariant finite positive ambient
    volume supports almost-everywhere recurrence, not recurrence of the
    captured point. The captured absolute form mean is retained separately.

    Source trapping can hold independently of membership in that smaller
    family. Each exact nontip pair then has nonzero unordered internal velocity
    at every finite time. This does not give an amplitude floor. Angular bounds
    use cos(D)<=sinc(D), D=radius/sqrt(2), in the normalized signed internal
    plane (delta,u/sqrt(beta)). A full angular turn is not a state return or a
    common period. Interval speed bounds can touch zero for tiny exact positive
    capacity without negating the conditional strict-rotation theorem.
    """

    replica: SineReplicaScaleAssessment
    winding: int
    target_phase_turns: tuple[Q, ...]
    radius: Q
    excess_ceiling: Q
    form_mean_bounds: tuple[Q, Q]
    weighted_form_mean: Q
    weighted_mean_rate_residual_bounds: I
    divergence: Q
    family_form_coordinate_bounds: I
    spectral_gap_lower_bound: Q
    spectral_gap_method: str
    target_storage_bounds: I
    radius_angle_bounds: I
    acute_radius_margin_bounds: I
    cosine_lower_bound: Q | None
    coercivity_lower_bound: Q | None
    barrier_lower_bound: Q | None
    form_norm_squared_upper_bound: Q
    phase_norm_squared_upper_bound: Q
    norm_squared_upper_bound: Q
    excess_storage_upper_bound: Q
    norm_margin: Q
    energy_margin: Q | None
    family_barrier_margin: Q | None
    source_family_excess_margin: Q
    source_mean_margins: tuple[Q, Q]
    pair_source_status: tuple[str, ...]
    pair_all_time_activity_certified: tuple[bool, ...]
    pair_all_time_circulation_certified: tuple[bool, ...]
    family_admitted: bool
    source_set_trapping_certified: bool
    source_family_membership: str
    source_joint_persistence_certified: bool
    family_almost_everywhere_recurrence_certified: bool
    family_unresolved_conditions: tuple[str, ...]
    source_trapping_unresolved_conditions: tuple[str, ...]
    source_family_membership_reasons: tuple[str, ...]
    internal_phase_radius_bounds: I | None
    normalized_angular_speed_lower_bound: Q | None
    internal_angular_speed_bounds: I | None
    internal_full_turn_time_bounds: I | None
    individual_recurrence_status: str = "unavailable_for_chosen_state"
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "complete_doubled_C5_ordered_pairs_exact_target_turns_and_common_positive_capacity",
        "explicit_conservative_sine_law_same_single_capture_no_forcing_or_support_event",
        "shared_full_support_norm_cancelled_energy_and_acute_first_exit_barrier",
        "family_is_open_radius_excess_and_mean_slab_with_every_pair_tip_removed",
        "captured_absolute_mean_is_known_not_inferred_from_a_relative_gauge",
        "synchronized_tips_and_their_complements_are_invariant_by_fine_flow_uniqueness",
        "trapped_nontip_pairs_have_nonzero_unordered_velocity_at_every_finite_time",
        "no_uniform_positive_amplitude_floor_or_attraction_to_a_prepared_pulse",
        "almost_everywhere_recurrence_in_full_form_circle_volume_not_every_chosen_state",
        "no_transfer_to_exact_energy_mean_or_synchronized_preparation_surfaces",
        "angular_speed_uses_cos_D_below_sinc_D_with_no_new_clock_or_rate_law",
        "full_internal_plane_turn_times_are_not_periods_or_state_return_times",
        "positive_numeric_speed_lower_bound_may_be_unresolved_despite_exact_positive_capacity",
        "no_selected_state_recurrence_time_Floquet_radius_or_physical_identity",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        _validate_replica_labels(self.replica)
        return {
            "schema": "tnfr.relational-sine-replica-persistence.v1",
            "report": _project(self),
        }


def _replica_family_arguments(excess_ceiling, form_mean_bounds, winding):
    ceiling = exact_or_represented_real(excess_ceiling, "excess_ceiling")
    if ceiling <= 0:
        raise ValueError("excess_ceiling must be strictly positive")
    raw_mean = _ordered(form_mean_bounds, "form_mean_bounds", limit=3)
    if len(raw_mean) != 2:
        raise ValueError("form_mean_bounds must contain exactly two ordered endpoints")
    lower, upper = (
        exact_or_represented_real(value, "form_mean_bounds") for value in raw_mean
    )
    if lower >= upper:
        raise ValueError("form_mean_bounds must have strictly positive width")
    if type(winding) is not int or abs(winding) != 1:
        raise ValueError("winding must be the nonboolean integer -1 or 1")
    return ceiling, lower, upper


def _replica_cycle_target(comparison, pairs, edges, *, winding):
    """Shared exact target in the caller's complete oriented doubled-C5 order."""
    expected_edges = tuple(sorted({tuple(sorted((i, (i + 1) % 5))) for i in range(5)}))
    if len(pairs) != 5 or edges != expected_edges:
        raise _ReplicaAdmissionError(
            "ordered pairs must describe the complete doubled C5"
        )
    if type(winding) is not int or 4 * abs(winding) >= len(pairs):
        raise ValueError("target winding must be an integer with 4*abs(winding)<5")
    target_by_node = {
        node: Q(winding * j, len(pairs))
        for j, pair in enumerate(pairs)
        for node in pair
    }
    return tuple(target_by_node[node] for node in comparison.nodes)


def _replica_target_bounds(comparison, pairs, edges, turns, *, radius, winding):
    from .relational_sine_recovery import _local_sine_geometry

    target = _replica_cycle_target(comparison, pairs, edges, winding=winding)
    return _local_sine_geometry(
        comparison, radius=radius, phase_turns=turns, target_phase_turns=target
    )


@dataclass(frozen=True)
class SineReplicaEquilibriumTarget:
    """A symbolic critical family, modulo common form and circular phase origins.

    Exact turns and reconstructed sine cancellation concern this supplied
    family, not the captured graph's rounded phase attributes. The retained
    reconstruction is used only for circular topology and sine balance; its
    separate equal-capacity phase-law interpretation is not imported.
    """

    winding: int
    target_phase_turns: tuple[Q, ...]
    target_edge_turns: tuple[Q, ...]
    target_geometry: PhaseCycleState
    edge_cosine_bounds: tuple[I, ...]
    storage_bounds: I


@dataclass(frozen=True)
class SineReplicaEquilibriaAssessment:
    """Exhaustive acute equilibrium families and separate captured-state evidence.

    Positive held capacities make zero phase rates imply uniform form. Strict
    edge acuteness and zero form rates then force the structural twins to share
    a circular phase. The remaining cycle currents and circular closure give
    exactly the integer windings with 4*abs(k)<5. No local pair lift is assumed
    to prove this classification.

    Captured phase coordinates are exact rational radians after shared engine
    materialization. They cannot represent a nonzero irrational twist exactly.
    Thus, within the certified acute sector, only identical captured phases
    and uniform form certify equilibrium. A small residual is never promoted
    to equality. Nonacute or unresolved phases do not become a global exclusion
    theorem; nonuniform form alone excludes equilibrium on connected support.
    """

    comparison: SineExchangeComparison
    pairs: tuple[tuple[Any, Any], ...]
    base_edges: tuple[tuple[int, int], ...]
    targets: tuple[SineReplicaEquilibriumTarget, ...]
    source_edge_cosine_bounds: tuple[I, ...]
    source_acute_status: str
    source_form_uniform: bool
    source_raw_phase_uniform: bool
    source_equilibrium_status: str
    source_equilibrium_reasons: tuple[str, ...]
    source_equilibrium_winding: int | None
    acute_equilibrium_classification_certified: bool = True
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "complete_supplied_doubled_C5_conservative_normalized_sine_law",
        "arbitrary_strictly_positive_held_fine_capacities_no_pair_equality_required",
        "all_form_and_phase_rows_must_vanish_for_full_equilibrium",
        "exhaustive_only_inside_the_strictly_acute_fine_edge_sector",
        "structural_twins_are_supplied_support_not_a_phase_observer_fallback",
        "exact_symbolic_targets_have_uniform_form_and_arbitrary_common_origins",
        "exact_turn_reconstruction_and_sine_cancellation_reuse_the_geometry_owner",
        "captured_rational_radians_are_distinct_from_symbolic_pi_turn_targets",
        "no_small_residual_equilibrium_tolerance_or_rounded_twist_promotion",
        "no_winding_preference_stability_formation_selection_or_physical_identification",
        "no_graph_write_pair_lift_guess_trajectory_or_native_runtime_dispatch",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project, _validate_label

        _validate_replica_labels(self)
        for target in self.targets:
            for node in target.target_geometry.geometry.nodes:
                _validate_label(node)
        return {
            "schema": "tnfr.relational-sine-replica-equilibria.v1",
            "report": _project(self),
        }


def assess_sine_replica_equilibria(
    graph, *, reference_model, pairs
) -> SineReplicaEquilibriaAssessment:
    """Classify acute full equilibria without imposing a target on the source.

    ``pairs`` supplies the oriented structural C5 order, not an observed
    grouping. The source may lie outside the acute sector; the symbolic family
    classification is still meaningful, while per-source admission abstains.
    The existing phase-only pairing observer remains independent.
    """
    from .relational_sine_recovery import _target_geometry

    comparison = bound_relational_sine_exchange(graph, reference_model=reference_model)
    if Q(reference_model.effective_weights[0]) != 0:
        raise _ReplicaAdmissionError(
            "equilibrium classification requires explicit zero form loss"
        )
    pairs, _, edges, _, _, _ = _replica_support(
        comparison, pairs, require_equal_capacity=False
    )
    # Shared target admission proves the full doubled-C5 order before the
    # integer closure inequality generates the exhaustive candidate list.
    _replica_cycle_target(comparison, pairs, edges, winding=0)
    n, pi, beta = len(pairs), pi_interval(), Q(reference_model.storage_scale)
    windings = tuple(k for k in range(-(n // 4), n // 4 + 1) if 4 * abs(k) < n)
    targets = []
    for winding in windings:
        target = _replica_cycle_target(comparison, pairs, edges, winding=winding)
        geometry, _, turns = _target_geometry(comparison, target)
        cosine = tuple(cos((2 * turn) * pi) for turn in turns)
        targets.append(
            SineReplicaEquilibriumTarget(
                winding,
                target,
                turns,
                geometry,
                cosine,
                beta * sum((1 - value for value in cosine), I(0)),
            )
        )
    positions = {node: i for i, node in enumerate(comparison.nodes)}
    source_cosine = tuple(
        I(
            *certified_cosine_bounds(
                comparison.phase[positions[right]] - comparison.phase[positions[left]]
            )
        )
        for left, right in comparison.edges
    )
    acute_status = (
        "fully_acute"
        if all(value.lo > 0 for value in source_cosine)
        else (
            "outside_fully_acute_sector"
            if any(value.hi <= 0 for value in source_cosine)
            else "unresolved"
        )
    )
    form_uniform = len(set(comparison.epi)) == 1
    phase_uniform = len(set(comparison.phase)) == 1
    if not form_uniform:
        status, reasons, source_winding = (
            "certified_not_equilibrium",
            ("connected_positive_capacity_phase_row_requires_uniform_form",),
            None,
        )
    elif phase_uniform:
        status, reasons, source_winding = (
            "certified_consensus_equilibrium",
            ("exact_captured_form_and_raw_phase_uniformity",),
            0,
        )
    elif acute_status == "fully_acute":
        status, reasons, source_winding = (
            "certified_not_equilibrium",
            ("acute_classification_excludes_nonuniform_rational_radian_phases",),
            None,
        )
    else:
        status, reasons, source_winding = (
            "unavailable",
            ("nonacute_or_unresolved_phase_sector_not_globally_classified",),
            None,
        )
    return SineReplicaEquilibriaAssessment(
        comparison=comparison,
        pairs=pairs,
        base_edges=edges,
        targets=tuple(targets),
        source_edge_cosine_bounds=source_cosine,
        source_acute_status=acute_status,
        source_form_uniform=form_uniform,
        source_raw_phase_uniform=phase_uniform,
        source_equilibrium_status=status,
        source_equilibrium_reasons=reasons,
        source_equilibrium_winding=source_winding,
    )


def assess_sine_replica_persistence(
    graph,
    *,
    reference_model,
    pairs,
    radius,
    excess_ceiling,
    form_mean_bounds,
    winding=1,
    phase_turns=None,
) -> SineReplicaPersistenceAssessment:
    """Assess an explicit open persistent family without evolving the graph.

    ``pairs`` declares the oriented C5 order; both constituents in pair j have
    exact target phase turns winding*j/5, modulo a common origin. Captured
    phases remain represented inputs and need not equal that irrational target.
    Bounds select an analysis family, not constitutive parameters. A failed
    sufficient geometric check abstains; it does not establish instability.
    Capacity equality concerns the captured represented model, not discarded
    raw-input precision or a tolerance-based approximation to another law.
    """
    ceiling, lower, upper = _replica_family_arguments(
        excess_ceiling, form_mean_bounds, winding
    )
    replica = assess_sine_replica_scale(
        graph, reference_model=reference_model, pairs=pairs, phase_turns=phase_turns
    )
    if len(set(replica.pair_capacity)) != 1:
        raise ValueError("persistence admission requires common positive held capacity")
    comparison = replica.comparison
    common, target_fields, failures, geometry_unresolved, source_unresolved = (
        _replica_target_bounds(
            comparison,
            replica.pairs,
            replica.base_edges,
            replica.phase_turns,
            radius=radius,
            winding=winding,
        )
    )
    target = common["target_phase_turns"]
    if failures:  # The replica owner already admits strict positive capacity.
        raise ValueError("; ".join(failures))
    radius = common["radius"]
    barrier = common["barrier_lower_bound"]
    family_margin = None if barrier is None else barrier - ceiling
    family_unresolved = geometry_unresolved + (
        ()
        if family_margin is not None and family_margin > 0
        else ("strict_family_excess_barrier_not_certified",)
    )
    trapping_unresolved = geometry_unresolved + source_unresolved
    family_admitted = not family_unresolved
    trapped = not trapping_unresolved
    # The regular doubled cycle and common capacity make rho=d/nu uniform.
    mean = sum(comparison.epi, Q(0)) / len(comparison.nodes)
    mean_rate = sum(comparison.form_rates, I(0)) / len(comparison.nodes)
    mean_margins = mean - lower, upper - mean
    statuses = tuple(
        "synchronized_tip" if u == raw == turn == 0 else "exact_nontip"
        for u, raw, turn in zip(
            replica.form_half_differences,
            replica.phase_half_difference_radian_parts,
            replica.phase_half_difference_turn_parts,
        )
    )
    active = tuple(status == "exact_nontip" for status in statuses)
    source_margin = ceiling - common["excess_storage_upper_bound"]
    reasons = []
    if min(mean_margins) <= 0:
        reasons.append("captured_mean_outside_open_slab")
    if not all(active):
        reasons.append("captured_source_has_synchronized_pair_tip")
    if reasons:
        membership = "outside"
    elif common["norm_margin"] > 0 and source_margin > 0:
        membership = "certified_inside"
    else:
        membership = "unresolved"
        reasons.append("strict_relative_family_membership_not_certified")
    phase_radius = angular_factor = angular_speed = turn_time = None
    if not geometry_unresolved:
        pi = pi_interval()
        beta_root = sqrt(I(Q(reference_model.storage_scale)))
        phase_radius = radius / sqrt(I(2))
        angular_factor = common["cosine_lower_bound"] * cos(phase_radius).lo
        w = Q(reference_model.effective_weights[1])
        clock_numerator = w * replica.pair_capacity[0]
        if beta_root.lo > 0:
            frequency = clock_numerator / (pi * beta_root)
            angular_speed = I(frequency.lo * angular_factor, frequency.hi)
        # Divide by the exact positive capacity before interval projection.
        # This retains a finite upper bound even for sub-quantum capacities.
        base_turn = pi**2 * beta_root
        turn_scale = 2 / clock_numerator
        minimum_turn = I(turn_scale * base_turn.lo, turn_scale * base_turn.hi)
        turn_time = I(minimum_turn.lo, minimum_turn.hi / angular_factor)
    return SineReplicaPersistenceAssessment(
        replica=replica,
        winding=winding,
        target_phase_turns=target,
        radius=radius,
        excess_ceiling=ceiling,
        form_mean_bounds=(lower, upper),
        weighted_form_mean=mean,
        weighted_mean_rate_residual_bounds=mean_rate,
        divergence=Q(0),
        family_form_coordinate_bounds=I(lower - radius, upper + radius),
        spectral_gap_lower_bound=common["spectral_gap_lower_bound"],
        spectral_gap_method=target_fields["spectral_gap_method"],
        target_storage_bounds=Q(reference_model.storage_scale)
        * target_fields["target_phase_storage_bounds"],
        radius_angle_bounds=common["radius_angle_bounds"],
        acute_radius_margin_bounds=common["acute_radius_margin_bounds"],
        cosine_lower_bound=common["cosine_lower_bound"],
        coercivity_lower_bound=common["coercivity_lower_bound"],
        barrier_lower_bound=barrier,
        form_norm_squared_upper_bound=common["form_norm_squared_upper_bound"],
        phase_norm_squared_upper_bound=common["phase_norm_squared_upper_bound"],
        norm_squared_upper_bound=common["norm_squared_upper_bound"],
        excess_storage_upper_bound=common["excess_storage_upper_bound"],
        norm_margin=common["norm_margin"],
        energy_margin=common["energy_margin"],
        family_barrier_margin=family_margin,
        source_family_excess_margin=source_margin,
        source_mean_margins=mean_margins,
        pair_source_status=statuses,
        pair_all_time_activity_certified=tuple(trapped and value for value in active),
        pair_all_time_circulation_certified=tuple(
            trapped and value for value in active
        ),
        family_admitted=family_admitted,
        source_set_trapping_certified=trapped,
        source_family_membership=membership,
        source_joint_persistence_certified=trapped and all(active),
        family_almost_everywhere_recurrence_certified=family_admitted,
        family_unresolved_conditions=family_unresolved,
        source_trapping_unresolved_conditions=trapping_unresolved,
        source_family_membership_reasons=tuple(reasons),
        internal_phase_radius_bounds=phase_radius,
        normalized_angular_speed_lower_bound=angular_factor,
        internal_angular_speed_bounds=angular_speed,
        internal_full_turn_time_bounds=turn_time,
    )


@dataclass(frozen=True)
class SineMobilityGeometryAssessment:
    """Conservative relative trapping, separate from invariant-measure claims.

    The source uses the same doubled-C5 target, local pair lifts and acute
    excess-storage barrier as the sine family. Source trapping needs neither
    membership in the smaller open family nor nonzero internal displacement.
    That family's synchronized tips are removed; their invariance follows
    from swap symmetry and uniqueness, not a sine-specific circulation bound.

    Recurrence at epsilon zero concerns the bounded relative family and its
    Euclidean measure. At positive epsilon an equivalent finite invariant
    measure on this relative family is still unproved. Neither status proves
    return of a selected state or the removed common origins.
    """

    balance: SineMobilityRelativeBalance
    pairs: tuple[tuple[Any, Any], ...]
    phase_turns: tuple[int, ...]
    winding: int
    target_phase_turns: tuple[Q, ...]
    radius: Q
    excess_ceiling: Q
    spectral_gap_lower_bound: Q
    spectral_gap_method: str
    target_storage_bounds: I
    radius_angle_bounds: I
    acute_radius_margin_bounds: I
    cosine_lower_bound: Q | None
    coercivity_lower_bound: Q | None
    barrier_lower_bound: Q | None
    form_norm_squared_upper_bound: Q
    phase_norm_squared_upper_bound: Q
    norm_squared_upper_bound: Q
    excess_storage_upper_bound: Q
    norm_margin: Q
    energy_margin: Q | None
    family_barrier_margin: Q | None
    source_family_excess_margin: Q
    pair_source_status: tuple[str, ...]
    relative_coordinate_deviation_bound: I
    family_admitted: bool
    source_set_trapping_certified: bool
    source_relative_family_membership: str
    family_unresolved_conditions: tuple[str, ...]
    source_trapping_unresolved_conditions: tuple[str, ...]
    source_family_membership_reasons: tuple[str, ...]
    relative_family_recurrence_status: str
    invariant_measure_status: str
    family_excludes_synchronized_tips: bool = True
    synchronized_tip_sets_invariant: bool = True
    internal_circulation_status: str = "not_assessed_for_changed_mobility"
    individual_recurrence_status: str = "unavailable_for_chosen_state"
    full_state_recurrence_status: str = "not_assessed_removed_origins"
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "same_supplied_doubled_C5_exact_target_turns_and_common_positive_held_capacity",
        "declared_current_squared_reciprocal_mobility_both_full_rows_zero_loss",
        "shared_acute_norm_cancelled_excess_storage_and_first_exit_barrier",
        "common_form_and_phase_origins_removed_without_clock_redefinition",
        "relative_family_is_bounded_radius_and_excess_sublevel_with_tips_removed",
        "tip_complements_remain_invariant_by_swap_symmetry_and_smooth_uniqueness",
        "tip_exclusion_does_not_certify_nonzero_internal_velocity_or_circulation",
        "source_relative_trapping_is_independent_of_stricter_open_family_membership",
        "no_finite_absolute_mean_slab_or_sine_rate_bound_is_imported",
        "epsilon_zero_relative_recurrence_is_almost_everywhere_not_chosen_state_return",
        "positive_epsilon_needs_finite_invariant_measure_equivalent_to_relative_volume",
        "no_failure_of_recurrence_inferred_from_unproved_measure_or_nonzero_divergence",
        "no_new_target_radius_search_trajectory_solver_or_physical_identification",
    )

    @property
    def comparison(self):
        return self.balance.comparison

    @property
    def mobility(self):
        return self.balance.mobility

    def to_dict(self):
        from ..sdk.relational_reports import _project, _validate_label

        _validate_replica_labels(self)
        _validate_label(self.balance.reference_node)
        return {
            "schema": "tnfr.relational-sine-mobility-geometry.v1",
            "report": _project(self),
        }


def assess_sine_mobility_geometry(
    graph,
    *,
    reference_model,
    pairs,
    radius,
    excess_ceiling,
    epsilon,
    winding=1,
    phase_turns=None,
) -> SineMobilityGeometryAssessment:
    """Reuse the exact relative-energy barrier with the declared alternative.

    The first member of the first supplied pair selects the relative-coordinate
    reference. Captured phase differences remain raw circular representatives;
    ``phase_turns`` separately supplies the barrier chart. No absolute form
    mean interval, unchanged sine trajectory, angular speed or period follows.
    """
    ceiling = exact_or_represented_real(excess_ceiling, "excess_ceiling")
    if ceiling <= 0:
        raise ValueError("excess_ceiling must be strictly positive")
    if type(winding) is not int or abs(winding) != 1:
        raise ValueError("winding must be the nonboolean integer -1 or 1")
    comparison = bound_relational_sine_exchange(graph, reference_model=reference_model)
    mobility = comparison.with_current_squared_mobility(epsilon=epsilon)
    coordinates = _replica_coordinates(comparison, pairs, phase_turns)
    if len(set(comparison.capacity)) != 1:
        raise ValueError("mobility geometry requires common positive held capacity")
    pairs, turns = coordinates["pairs"], coordinates["turns"]
    common, target_fields, failures, geometry_unresolved, source_unresolved = (
        _replica_target_bounds(
            comparison,
            pairs,
            coordinates["edges"],
            turns,
            radius=radius,
            winding=winding,
        )
    )
    if failures:
        raise ValueError("; ".join(failures))
    barrier, radius = common["barrier_lower_bound"], common["radius"]
    family_margin = None if barrier is None else barrier - ceiling
    family_unresolved = geometry_unresolved + (
        ()
        if family_margin is not None and family_margin > 0
        else ("strict_family_excess_barrier_not_certified",)
    )
    trapping_unresolved = geometry_unresolved + source_unresolved
    statuses = tuple(
        "synchronized_tip" if u == raw == turn == 0 else "exact_nontip"
        for u, raw, turn in zip(
            coordinates["internal"],
            coordinates["internal_radians"],
            coordinates["internal_turns"],
        )
    )
    source_margin = ceiling - common["excess_storage_upper_bound"]
    if "synchronized_tip" in statuses:
        membership, reasons = "outside", ("captured_source_has_synchronized_pair_tip",)
    elif common["norm_margin"] > 0 and source_margin > 0:
        membership, reasons = "certified_inside", ()
    else:
        membership, reasons = "unresolved", (
            "strict_relative_family_membership_not_certified",
        )
    family_admitted = not family_unresolved
    if not family_admitted:
        recurrence = "unavailable_family_not_admitted"
    elif mobility.epsilon == 0:
        recurrence = "certified_almost_everywhere_in_relative_family"
    else:
        recurrence = "unavailable_invariant_measure_unproved"
    return SineMobilityGeometryAssessment(
        balance=mobility.relative_balance(reference_node=pairs[0][0]),
        pairs=pairs,
        phase_turns=turns,
        winding=winding,
        target_phase_turns=common["target_phase_turns"],
        radius=radius,
        excess_ceiling=ceiling,
        spectral_gap_lower_bound=common["spectral_gap_lower_bound"],
        spectral_gap_method=target_fields["spectral_gap_method"],
        target_storage_bounds=Q(reference_model.storage_scale)
        * target_fields["target_phase_storage_bounds"],
        radius_angle_bounds=common["radius_angle_bounds"],
        acute_radius_margin_bounds=common["acute_radius_margin_bounds"],
        cosine_lower_bound=common["cosine_lower_bound"],
        coercivity_lower_bound=common["coercivity_lower_bound"],
        barrier_lower_bound=barrier,
        form_norm_squared_upper_bound=common["form_norm_squared_upper_bound"],
        phase_norm_squared_upper_bound=common["phase_norm_squared_upper_bound"],
        norm_squared_upper_bound=common["norm_squared_upper_bound"],
        excess_storage_upper_bound=common["excess_storage_upper_bound"],
        norm_margin=common["norm_margin"],
        energy_margin=common["energy_margin"],
        family_barrier_margin=family_margin,
        source_family_excess_margin=source_margin,
        pair_source_status=statuses,
        relative_coordinate_deviation_bound=sqrt(I(2)) * radius,
        family_admitted=family_admitted,
        source_set_trapping_certified=not trapping_unresolved,
        source_relative_family_membership=membership,
        family_unresolved_conditions=family_unresolved,
        source_trapping_unresolved_conditions=trapping_unresolved,
        source_family_membership_reasons=reasons,
        relative_family_recurrence_status=recurrence,
        invariant_measure_status=(
            "relative_Euclidean_volume"
            if mobility.epsilon == 0
            else "equivalent_finite_invariant_measure_not_supplied"
        ),
    )


@dataclass(frozen=True)
class SineReplicaCapacityFamily:
    """Open full-state identity family; synchronized pair tips are included."""

    winding: int
    target_phase_turns: tuple[Q, ...]
    radius: Q
    excess_ceiling: Q
    form_mean_bounds: tuple[Q, Q]
    family_form_coordinate_bounds: I
    spectral_gap_lower_bound: Q
    spectral_gap_method: str
    target_storage_bounds: I
    acute_radius_margin_bounds: I
    cosine_lower_bound: Q | None
    barrier_lower_bound: Q | None
    norm_squared_upper_bound: Q
    excess_storage_upper_bound: Q
    norm_margin: Q
    energy_margin: Q | None
    family_barrier_margin: Q | None
    source_family_excess_margin: Q
    source_mean_margins: tuple[Q, Q]
    family_admitted: bool
    source_set_trapping_certified: bool
    source_family_membership: str
    family_almost_everywhere_recurrence_certified: bool
    family_unresolved_conditions: tuple[str, ...]
    source_trapping_unresolved_conditions: tuple[str, ...]
    source_family_membership_reasons: tuple[str, ...]
    individual_recurrence_status: str = "unavailable_for_chosen_state"


@dataclass(frozen=True)
class SineReplicaCapacityAssessment:
    """Captured positive held-capacity replica rows with capacity correlations.

    v=(nu_plus+nu_minus)/2 and eta=(nu_plus-nu_minus)/2 are exact properties of
    captured represented values. Retained E=eta**2, P=eta*u and T=eta*sin(delta)
    close jointly with X,Theta,R,U,Q. Their Gram constraints describe captured
    realizable states; arbitrary proposed tuples are not admitted. A joint swap
    of member states and capacities is a symmetry; swapping states alone while
    holding unequal capacities in place need not preserve collective rates.

    Optional family evidence uses the same complete fine-state energy barrier
    and the actual degree/capacity weighted mean. It includes synchronized tips
    and certifies no all-time internal activity, circulation or selected-state
    recurrence. The two sparse ordered Poisson coefficients multiply the within
    (X,Theta)/(u,delta) pairs and cross (X,delta)/(u,Theta) pairs respectively;
    both are inherited from the same fine storage, not a new Hamiltonian.
    """

    comparison: SineExchangeComparison
    pairs: tuple[tuple[Any, Any], ...]
    base_edges: tuple[tuple[int, int], ...]
    base_neighbors: tuple[tuple[int, ...], ...]
    base_degrees: tuple[int, ...]
    pair_capacity_means: tuple[Q, ...]
    pair_capacity_half_differences: tuple[Q, ...]
    phase_turns: tuple[int, ...]
    form_means: tuple[Q, ...]
    form_half_differences: tuple[Q, ...]
    phase_mean_radian_parts: tuple[Q, ...]
    phase_mean_turn_parts: tuple[Q, ...]
    phase_half_difference_radian_parts: tuple[Q, ...]
    phase_half_difference_turn_parts: tuple[Q, ...]
    phase_half_difference_bounds: tuple[I, ...]
    phase_chart_margins: tuple[I, ...]
    resultant_magnitude_bounds: tuple[I, ...]
    internal_form_squared: tuple[Q, ...]
    form_phase_correlation_bounds: tuple[I, ...]
    capacity_half_difference_squared: tuple[Q, ...]
    capacity_form_correlation: tuple[Q, ...]
    capacity_phase_correlation_bounds: tuple[I, ...]
    ordered_direct_rates: tuple[tuple[I, ...], ...]
    ordered_generated_rates: tuple[tuple[I, ...], ...]
    ordered_rate_residuals: tuple[tuple[I, ...], ...]
    ordered_poisson_coefficients: tuple[tuple[I, I], ...]
    ordered_storage_gradients: tuple[tuple[I, ...], ...]
    ordered_poisson_rate_residuals: tuple[tuple[I, ...], ...]
    capacity_aware_direct_rates: tuple[tuple[I, ...], ...]
    capacity_aware_generated_rates: tuple[tuple[I, ...], ...]
    capacity_aware_rate_residuals: tuple[tuple[I, ...], ...]
    capacity_aware_constraint_residuals: tuple[tuple[I, ...], ...]
    normalized_form_mean_weights: tuple[Q, ...]
    weighted_form_mean: Q
    weighted_mean_rate_residual_bounds: I
    arithmetic_mean_rate_bounds: I
    divergence: Q
    family: SineReplicaCapacityFamily | None
    ordered_coordinate_order: tuple[str, ...] = ("X", "Theta", "u", "delta")
    capacity_aware_coordinate_order: tuple[str, ...] = (
        "X",
        "Theta",
        "R",
        "U",
        "Q",
        "P",
        "T",
        "E",
    )
    capacity_aware_constraint_names: tuple[str, ...] = (
        "Q_squared_minus_U_one_minus_R_squared",
        "P_squared_minus_E_U",
        "T_squared_minus_E_one_minus_R_squared",
        "P_T_minus_E_Q",
        "P_Q_minus_U_T",
        "Q_T_minus_P_one_minus_R_squared",
    )
    capacity_aware_closure_certified: bool = True
    all_time_internal_activity_status: str = "not_certified_by_this_reader"
    angular_circulation_status: str = "not_certified_by_this_reader"
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "single_full_fine_capture_complete_bipartite_replica_support_and_explicit_local_chart",
        "conservative_sine_law_with_arbitrary_strictly_positive_held_fine_capacities",
        "exact_relations_of_captured_represented_values_not_discarded_raw_input_precision",
        "ordered_cross_coupling_and_Poisson_coefficients_retain_capacity_state_correlations",
        "capacity_tagged_unordered_closure_is_on_the_realized_rank_one_positive_Gram_image",
        "joint_member_state_capacity_swaps_not_fixed_capacity_state_swaps",
        "no_division_by_capacity_contrast_or_arbitrary_invariant_tuple_admission",
        "full_storage_and_degree_over_capacity_weighted_mean_remain_conserved",
        "optional_C5_family_includes_tips_and_is_separate_from_internal_activity_claims",
        "family_recurrence_is_almost_everywhere_not_a_selected_state_return",
        "no_capacity_evolution_new_law_trajectory_controller_or_physical_identification",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        _validate_replica_labels(self)
        return {
            "schema": "tnfr.relational-sine-replica-capacity.v1",
            "report": _project(self),
        }


def _capacity_family(comparison, data, mean, *, radius, ceiling, lower, upper, winding):
    common, target_fields, failures, geometry_unresolved, source_unresolved = (
        _replica_target_bounds(
            comparison,
            data["pairs"],
            data["edges"],
            data["turns"],
            radius=radius,
            winding=winding,
        )
    )
    if failures:
        raise ValueError("; ".join(failures))
    barrier = common["barrier_lower_bound"]
    family_margin = None if barrier is None else barrier - ceiling
    family_unresolved = geometry_unresolved + (
        ()
        if family_margin is not None and family_margin > 0
        else ("strict_family_excess_barrier_not_certified",)
    )
    source_margin = ceiling - common["excess_storage_upper_bound"]
    mean_margins = mean - lower, upper - mean
    if min(mean_margins) <= 0:
        membership, reasons = "outside", ("weighted_mean_outside_open_slab",)
    elif common["norm_margin"] > 0 and source_margin > 0:
        membership, reasons = "certified_inside", ()
    else:
        membership, reasons = "unresolved", (
            "strict_relative_family_membership_not_certified",
        )
    form_radius = sqrt(I(2)) * common["radius"]
    return SineReplicaCapacityFamily(
        winding=winding,
        target_phase_turns=common["target_phase_turns"],
        radius=common["radius"],
        excess_ceiling=ceiling,
        form_mean_bounds=(lower, upper),
        family_form_coordinate_bounds=I(lower - form_radius.hi, upper + form_radius.hi),
        spectral_gap_lower_bound=common["spectral_gap_lower_bound"],
        spectral_gap_method=target_fields["spectral_gap_method"],
        target_storage_bounds=Q(comparison.reference_model.storage_scale)
        * target_fields["target_phase_storage_bounds"],
        acute_radius_margin_bounds=common["acute_radius_margin_bounds"],
        cosine_lower_bound=common["cosine_lower_bound"],
        barrier_lower_bound=barrier,
        norm_squared_upper_bound=common["norm_squared_upper_bound"],
        excess_storage_upper_bound=common["excess_storage_upper_bound"],
        norm_margin=common["norm_margin"],
        energy_margin=common["energy_margin"],
        family_barrier_margin=family_margin,
        source_family_excess_margin=source_margin,
        source_mean_margins=mean_margins,
        family_admitted=not family_unresolved,
        source_set_trapping_certified=not (geometry_unresolved + source_unresolved),
        source_family_membership=membership,
        family_almost_everywhere_recurrence_certified=not family_unresolved,
        family_unresolved_conditions=family_unresolved,
        source_trapping_unresolved_conditions=geometry_unresolved + source_unresolved,
        source_family_membership_reasons=reasons,
    )


def _optional_replica_family(radius, excess_ceiling, form_mean_bounds, winding):
    requested = tuple(
        value is not None for value in (radius, excess_ceiling, form_mean_bounds)
    )
    if any(requested) and not all(requested):
        raise ValueError("supply radius, excess_ceiling and form_mean_bounds together")
    return (
        _replica_family_arguments(excess_ceiling, form_mean_bounds, winding)
        if all(requested)
        else None
    )


def assess_sine_replica_capacity(
    graph,
    *,
    reference_model,
    pairs,
    phase_turns=None,
    radius=None,
    excess_ceiling=None,
    form_mean_bounds=None,
    winding=1,
) -> SineReplicaCapacityAssessment:
    """Retain exact capacity contrast without extending symmetric-pair theorems.

    Supply all three optional family bounds, or none. The instantaneous rows
    admit any connected complete replica support; the optional family requires
    the five ordered pairs of doubled C5. All evidence uses one captured fine
    state. Source capacities may be unequal; none is adapted or approximated
    as equal by a threshold. Shared staging still defines represented values.
    """
    family_arguments = _optional_replica_family(
        radius, excess_ceiling, form_mean_bounds, winding
    )
    comparison = bound_relational_sine_exchange(graph, reference_model=reference_model)
    return _assess_sine_replica_capacity_comparison(
        comparison,
        pairs=pairs,
        phase_turns=phase_turns,
        radius=radius,
        family_arguments=family_arguments,
        winding=winding,
    )


def _assess_sine_replica_capacity_comparison(
    comparison,
    *,
    pairs,
    phase_turns=None,
    radius=None,
    family_arguments=None,
    winding=1,
):
    """Reuse one admitted full capture without another graph read."""
    reference_model = comparison.reference_model
    if Q(reference_model.effective_weights[0]) != 0:
        raise _ReplicaAdmissionError(
            "capacity replica assessment requires explicit zero form loss"
        )
    data = _replica_coordinates(
        comparison, pairs, phase_turns, require_equal_capacity=False
    )
    indices, degrees, v = data["indices"], data["degrees"], data["capacity"]
    means, u, sine_delta, r = (
        data["means"],
        data["internal"],
        data["sine_delta"],
        data["coherence"],
    )
    eta = tuple(
        (comparison.capacity[i] - comparison.capacity[j]) / 2 for i, j in indices
    )
    size = len(indices)
    neighbor_sine, neighbor_cosine = [I(0) for _ in indices], [I(0) for _ in indices]
    for (i, j), (sine, cosine) in zip(data["edges"], data["edge_trig"]):
        neighbor_sine[i] += r[j] * sine
        neighbor_sine[j] -= r[i] * sine
        neighbor_cosine[i] += r[j] * cosine
        neighbor_cosine[j] += r[i] * cosine
    q = tuple(
        sum((means[i] - means[j] for j in row), Q(0))
        for i, row in enumerate(data["neighbors"])
    )
    per_capacity = _sine_rates(
        reference_model,
        degrees,
        q,
        (Q(1),) * size,
        tuple(r_i * value for r_i, value in zip(r, neighbor_sine)),
    )
    restoring = _sine_rates(
        reference_model, degrees, degrees, (Q(1),) * size, neighbor_cosine
    )
    f, k, a, b = (
        per_capacity["form_rates"],
        per_capacity["phase_rates"],
        restoring["form_rates"],
        restoring["phase_rates"],
    )
    ordered = tuple(
        (
            mean * f_i - contrast * a_i * sine,
            mean * k_i + b_i * contrast * u_i,
            contrast * f_i - mean * a_i * sine,
            contrast * k_i + b_i * mean * u_i,
        )
        for mean, contrast, u_i, sine, f_i, k_i, a_i, b_i in zip(
            v, eta, u, sine_delta, f, k, a, b
        )
    )
    direct = tuple(
        (
            (comparison.form_rates[i] + comparison.form_rates[j]) / 2,
            (comparison.phase_rates[i] + comparison.phase_rates[j]) / 2,
            (comparison.form_rates[i] - comparison.form_rates[j]) / 2,
            (comparison.phase_rates[i] - comparison.phase_rates[j]) / 2,
        )
        for i, j in indices
    )
    variance = tuple(value**2 for value in u)
    correlation = tuple(value * sine for value, sine in zip(u, sine_delta))
    e = tuple(value**2 for value in eta)
    p = tuple(contrast * value for contrast, value in zip(eta, u))
    t = tuple(contrast * sine for contrast, sine in zip(eta, sine_delta))
    generated, pushed, constraints = [], [], []
    for i in range(size):
        ri, ui, qi, pi, ti, ei, vi = (
            r[i],
            variance[i],
            correlation[i],
            p[i],
            t[i],
            e[i],
            v[i],
        )
        fi, ki, ai, bi = f[i], k[i], a[i], b[i]
        generated.append(
            (
                vi * fi - ai * ti,
                vi * ki + bi * pi,
                -ki * ti - bi * vi * qi,
                2 * fi * pi - 2 * vi * ai * qi,
                fi * ti - vi * ai * (1 - ri**2) + ki * ri * pi + bi * vi * ui * ri,
                ei * fi - vi * ai * ti,
                ri * (ei * ki + bi * vi * pi),
                I(0),
            )
        )
        x_rate, theta_rate, u_rate, delta_rate = direct[i]
        pushed.append(
            (
                x_rate,
                theta_rate,
                -sine_delta[i] * delta_rate,
                2 * u[i] * u_rate,
                sine_delta[i] * u_rate + u[i] * ri * delta_rate,
                eta[i] * u_rate,
                eta[i] * ri * delta_rate,
                I(0),
            )
        )
        constraints.append(
            (
                qi**2 - ui * (1 - ri**2),
                I(pi**2 - ei * ui),
                ti**2 - ei * (1 - ri**2),
                pi * ti - ei * qi,
                pi * qi - ui * ti,
                qi * ti - pi * (1 - ri**2),
            )
        )
    beta = Q(reference_model.storage_scale)
    gradients = tuple(
        (I(4 * q_i), -4 * beta * ri * si, I(4 * degree * u_i), 4 * beta * sine * ci)
        for q_i, ri, si, degree, u_i, sine, ci in zip(
            q, r, neighbor_sine, degrees, u, sine_delta, neighbor_cosine
        )
    )
    poisson = tuple(
        (-bi * vi / (4 * degree), -bi * contrast / (4 * degree))
        for bi, vi, contrast, degree in zip(b, v, eta, degrees)
    )
    poisson_rates = tuple(
        (c * y + d * delta, -c * x - d * u_grad, d * y + c * delta, -d * x - c * u_grad)
        for (c, d), (x, y, u_grad, delta) in zip(poisson, gradients)
    )
    rho = tuple(
        Q(degree) / nu for degree, nu in zip(comparison.degrees, comparison.capacity)
    )
    rho_sum = sum(rho, Q(0))
    weights = tuple(value / rho_sum for value in rho)
    mean = sum((weight * value for weight, value in zip(weights, comparison.epi)), Q(0))
    family = None
    if family_arguments is not None:
        ceiling, lower, upper = family_arguments
        family = _capacity_family(
            comparison,
            data,
            mean,
            radius=radius,
            ceiling=ceiling,
            lower=lower,
            upper=upper,
            winding=winding,
        )

    def residual(rows):
        return tuple(
            tuple(x - y for x, y in zip(row, expected))
            for row, expected in zip(rows, direct)
        )

    return SineReplicaCapacityAssessment(
        comparison=comparison,
        pairs=data["pairs"],
        base_edges=data["edges"],
        base_neighbors=data["neighbors"],
        base_degrees=degrees,
        pair_capacity_means=v,
        pair_capacity_half_differences=eta,
        phase_turns=data["turns"],
        form_means=means,
        form_half_differences=u,
        phase_mean_radian_parts=data["mean_radians"],
        phase_mean_turn_parts=data["mean_turns"],
        phase_half_difference_radian_parts=data["internal_radians"],
        phase_half_difference_turn_parts=data["internal_turns"],
        phase_half_difference_bounds=data["delta"],
        phase_chart_margins=data["chart_margins"],
        resultant_magnitude_bounds=r,
        internal_form_squared=variance,
        form_phase_correlation_bounds=correlation,
        capacity_half_difference_squared=e,
        capacity_form_correlation=p,
        capacity_phase_correlation_bounds=t,
        ordered_direct_rates=direct,
        ordered_generated_rates=ordered,
        ordered_rate_residuals=residual(ordered),
        ordered_poisson_coefficients=poisson,
        ordered_storage_gradients=gradients,
        ordered_poisson_rate_residuals=residual(poisson_rates),
        capacity_aware_direct_rates=tuple(pushed),
        capacity_aware_generated_rates=tuple(generated),
        capacity_aware_rate_residuals=tuple(
            tuple(x - y for x, y in zip(row, expected))
            for row, expected in zip(generated, pushed)
        ),
        capacity_aware_constraint_residuals=tuple(constraints),
        normalized_form_mean_weights=weights,
        weighted_form_mean=mean,
        weighted_mean_rate_residual_bounds=sum(
            (weight * rate for weight, rate in zip(weights, comparison.form_rates)),
            I(0),
        ),
        arithmetic_mean_rate_bounds=sum(comparison.form_rates, I(0))
        / len(comparison.nodes),
        divergence=Q(0),
        family=family,
    )


@dataclass(frozen=True)
class PhasePairObservation:
    """Strict circular nearest-partner observation from phases alone.

    Squared chord distance is monotone in circular separation; no inverse angle,
    fitted threshold, graph edge, capacity, form or encoded label is consulted.
    A candidate exists only if every node has a certified unique nearest partner
    and all choices are mutual. Captured order arranges the already determined
    unordered pairs for output; it never breaks a tie or supplies a cycle order.
    """

    nodes: tuple[Any, ...]
    phases: tuple[Q, ...]
    distance_indices: tuple[tuple[int, int], ...]
    squared_chord_bounds: tuple[I, ...]
    nearest_partner_indices: tuple[int | None, ...]
    nearest_separation_margins: tuple[Q | None, ...]
    node_statuses: tuple[str, ...]
    candidate_pairs: tuple[tuple[Any, Any], ...] | None
    status: str
    reasons: tuple[str, ...]
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "exact_or_represented_circular_phase_inputs_only_labels_are_bookkeeping",
        "squared_chord_2_minus_2_cos_gap_uses_mathematical_trigonometric_bounds",
        "strict_unique_nearest_and_mutual_complete_matching_no_greedy_repair",
        "ties_overlapping_enclosures_and_unmatched_nodes_abstain",
        "output_pair_layout_is_not_a_target_cycle_order_or_phase_lift",
        "64_node_limit_is_a_quadratic_arithmetic_budget_not_a_physical_scale",
        "no_support_law_future_pairing_closure_or_birth_claim_from_observation_alone",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project, _validate_label

        for node in self.nodes:
            _validate_label(node)
        if self.candidate_pairs is not None:
            for pair in self.candidate_pairs:
                for node in pair:
                    _validate_label(node)
        return {"schema": "tnfr.phase-pairs.v1", "report": _project(self)}


def _nearest_bound_candidate(size, distance, i):
    """Choose a min-upper candidate; only strict lower-bound separation admits it."""
    others = tuple(j for j in range(size) if j != i)
    j = min(others, key=lambda k: distance(i, k).hi) if others else None
    if j is None:
        return None, None, False
    margin = min(
        (distance(i, k).lo - distance(i, j).hi for k in others if k != j),
        default=None,
    )
    return j, margin, margin is None or margin > 0


def _phase_nearest_group(phases, distance, i):
    """A strict singleton or exact nearest tie, never an interval-overlap tie."""
    j, margin, strict = _nearest_bound_candidate(len(phases), distance, i)
    if j is None:
        return None, None
    if strict:
        return (j,), margin
    others = tuple(j for j in range(len(phases)) if j != i)
    group = tuple(
        k for k in others if abs(phases[k] - phases[i]) == abs(phases[j] - phases[i])
    )
    members = set(group)
    if len(group) < 2 or not all(
        distance(i, j).hi < distance(i, k).lo for k in others if k not in members
    ):
        return None, None
    members = set(group)
    margin = min(
        (distance(i, k).lo - distance(i, j).hi for k in others if k not in members),
        default=None,
    )
    return group, margin


def observe_phase_pairs(*, nodes, phases) -> PhasePairObservation:
    """Observe a complete phase-only mutual pairing or explicitly abstain.

    Integer/Fraction inputs are exact; other real inputs follow shared
    represented-real admission. This standalone observation performs no graph
    capture and admits no measurement-error model. At most 64 nodes are read.
    """
    nodes = _ordered(nodes, "nodes", limit=65)
    if not nodes or len(nodes) > 64:
        raise ValueError("phase pairing requires 1 to 64 nodes within its work budget")
    if len(set(nodes)) != len(nodes):
        raise ValueError("nodes must contain distinct hashable labels")
    raw = _ordered(phases, "phases", limit=len(nodes) + 1)
    if len(raw) != len(nodes):
        raise ValueError("phases must contain one value per node")
    phases = tuple(exact_or_represented_real(value, "phases") for value in raw)
    size = len(nodes)
    indices = tuple((i, j) for i in range(size) for j in range(i + 1, size))
    distances = tuple(
        2 * (1 - I(*certified_cosine_bounds(phases[j] - phases[i]))) for i, j in indices
    )
    lookup = {pair: value for pair, value in zip(indices, distances)}

    def distance(i, j):
        return lookup[min(i, j), max(i, j)]

    nearest, margins, statuses = [], [], []
    for i in range(size):
        # Any certifiable nearest must minimize the upper bound. Choosing one
        # candidate costs O(n); the strict comparison still rejects every tie.
        group, margin = _phase_nearest_group(phases, distance, i)
        if group is not None and len(group) == 1:
            nearest.append(group[0])
            margins.append(margin)
            statuses.append("certified_unique_nearest")
            continue
        nearest.append(None)
        margins.append(None)
        statuses.append(
            "exact_nearest_tie"
            if group is not None
            else "unresolved_nearest_comparison"
        )
    reasons = []
    if size % 2:
        reasons.append("odd_node_count_cannot_form_complete_pairs")
    if "exact_nearest_tie" in statuses:
        reasons.append("exact_nearest_tie")
    if "unresolved_nearest_comparison" in statuses:
        reasons.append("strict_nearest_comparison_unresolved")
    if any(j is not None and nearest[j] != i for i, j in enumerate(nearest)):
        reasons.append("nearest_relation_not_mutual")
    candidate = (
        None
        if reasons
        else tuple((nodes[i], nodes[j]) for i, j in enumerate(nearest) if i < j)
    )
    return PhasePairObservation(
        nodes=nodes,
        phases=phases,
        distance_indices=indices,
        squared_chord_bounds=distances,
        nearest_partner_indices=tuple(nearest),
        nearest_separation_margins=tuple(margins),
        node_statuses=tuple(statuses),
        candidate_pairs=candidate,
        status="unavailable" if reasons else "certified",
        reasons=tuple(reasons),
    )


@dataclass(frozen=True)
class JointPairObservation:
    """Strict nearest pairing in the supplied storage-induced joint distance.

    Squared distance is (x_i-x_j)**2+2*beta*(1-cos(theta_i-theta_j)).
    Its form and circular-phase terms use the declared storage scale. This
    observer does not infer that scale, support, dynamics or a physical metric.
    Strict mutual nearest choices are required; no label breaks a tie.
    """

    nodes: tuple[Any, ...]
    forms: tuple[Q, ...]
    phases: tuple[Q, ...]
    storage_scale: Q
    distance_indices: tuple[tuple[int, int], ...]
    squared_joint_distance_bounds: tuple[I, ...]
    nearest_partner_indices: tuple[int | None, ...]
    nearest_separation_margins: tuple[Q | None, ...]
    candidate_pairs: tuple[tuple[Any, Any], ...] | None
    status: str
    reasons: tuple[str, ...]
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "declared_positive_storage_scale_weights_signed_form_and_circular_phase",
        "exact_or_represented_scalar_inputs_not_a_measured_error_model",
        "strict_mutual_nearest_choices_no_label_tiebreak_or_matching_repair",
        "nonnegative_distance_bounds_use_the_known_squared_metric_domain",
        "64_node_limit_is_a_quadratic_work_budget_not_a_physical_scale",
        "no_support_sufficient_state_future_identity_or_operator_occurrence_claim",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project, _validate_label

        for node in self.nodes:
            _validate_label(node)
        for pair in self.candidate_pairs or ():
            for node in pair:
                _validate_label(node)
        return {"schema": "tnfr.joint-pairs.v1", "report": _project(self)}


def observe_joint_pairs(*, nodes, forms, phases, storage_scale) -> JointPairObservation:
    """Observe nearest pairs from form and circular phase, without a graph.

    ``forms`` accepts signed exact/represented real scalars only. This
    numerical observer does not accept BEPI containers or project a complex
    form to its magnitude. Graph-based callers retain the engine's separate
    signed-scalar materialization contract when supplying these values.
    """
    nodes = _ordered(nodes, "nodes", limit=65)
    if not nodes or len(nodes) > 64:
        raise ValueError("joint pairing requires 1 to 64 nodes within its work budget")
    if len(set(nodes)) != len(nodes):
        raise ValueError("nodes must contain distinct hashable labels")
    size = len(nodes)
    forms = _ordered(forms, "forms", limit=size + 1)
    phases = _ordered(phases, "phases", limit=size + 1)
    if len(forms) != size or len(phases) != size:
        raise ValueError("forms and phases must each contain one value per node")
    forms = tuple(exact_or_represented_real(value, "forms") for value in forms)
    phases = tuple(exact_or_represented_real(value, "phases") for value in phases)
    beta = exact_or_represented_real(storage_scale, "storage_scale")
    if beta <= 0:
        raise ValueError("storage_scale must be positive")
    indices = tuple((i, j) for i in range(size) for j in range(i + 1, size))
    distances = []
    for i, j in indices:
        raw = I((forms[j] - forms[i]) ** 2) + 2 * beta * (
            1 - I(*certified_cosine_bounds(phases[j] - phases[i]))
        )
        distances.append(I(max(Q(0), raw.lo), max(Q(0), raw.hi)))
    distances = tuple(distances)
    nearest, margins, resolved, pairs, status = _strict_pairing_from_bounds(
        nodes, indices, distances
    )
    reasons = (
        ()
        if pairs is not None
        else (
            ("strict_nearest_choices_are_not_mutual",)
            if resolved
            else ("one_or_more_joint_nearest_choices_unresolved",)
        )
    )
    return JointPairObservation(
        nodes=nodes,
        forms=forms,
        phases=phases,
        storage_scale=beta,
        distance_indices=indices,
        squared_joint_distance_bounds=distances,
        nearest_partner_indices=nearest,
        nearest_separation_margins=margins,
        candidate_pairs=pairs,
        status=status,
        reasons=reasons,
    )


@dataclass(frozen=True)
class SinePairingSide:
    """Strict mathematical nearest choices on an unspecified one-sided interval.

    A numerical observer need not resolve arbitrarily small positive margins.
    Exact coefficient margins retain shared-factor correlations even when the
    displayed derivative enclosure touches zero through outward rounding.
    """

    time_direction: int
    nearest_partner_indices: tuple[int | None, ...]
    node_statuses: tuple[str, ...]
    tied_rate_factor_bounds: tuple[I | None, ...]
    tied_rate_coefficient_margins: tuple[Q | None, ...]
    tied_derivative_margin_bounds: tuple[I | None, ...]
    candidate_pairs: tuple[tuple[Any, Any], ...] | None
    status: str
    reasons: tuple[str, ...]
    support_admission_status: str
    support_admission_reasons: tuple[str, ...]
    local_interval_existence_certified: bool
    certified_time_horizon: None = None


def _pairing_support_admission(comparison, pairs):
    """Independent complete-fiber/positive-capacity check after phase grouping."""
    if pairs is None:
        return "not_attempted", ()
    try:
        _replica_support(comparison, pairs, require_equal_capacity=False)
    except _ReplicaAdmissionError as error:
        return "rejected", (str(error),)
    return "admitted", ()


def _phase_nearest_relation(nodes, nearest):
    """Aggregate only complete strict nearest choices, without repairing them."""
    resolved = all(j is not None for j in nearest)
    mutual = resolved and all(nearest[j] == i for i, j in enumerate(nearest))
    pairs = (
        tuple((nodes[i], nodes[j]) for i, j in enumerate(nearest) if i < j)
        if mutual
        else None
    )
    status = (
        "certified_matching"
        if mutual
        else ("certified_nonmutual" if resolved else "unavailable")
    )
    return resolved, pairs, status


def _strict_pairing_from_bounds(nodes, indices, distances):
    """Share strict nearest admission for complete supplied distance bounds."""
    lookup = dict(zip(indices, distances))
    nearest, margins = [], []
    for i in range(len(nodes)):
        j, margin, strict = _nearest_bound_candidate(
            len(nodes), lambda i, j: lookup[min(i, j), max(i, j)], i
        )
        nearest.append(j if strict else None)
        margins.append(margin if strict else None)
    resolved, pairs, status = _phase_nearest_relation(nodes, nearest)
    return tuple(nearest), tuple(margins), resolved, pairs, status


@dataclass(frozen=True)
class SinePairingTransitionAssessment:
    """Complete-law chord rates and sufficient local changes of phase pairing.

    Shared phase rates have the exact form theta'_i=N_i/pi. For a tied nearest
    group with common absolute rational gap d, each chord derivative shares
    the factor 2*sin(d)/pi and has exact coefficient sign(gap)*(N_j-N_i).
    Strict coefficient order and a resolved factor sign prove a one-sided
    split; independent subtraction of rounded rates is unnecessary. Unresolved
    distance comparisons never become exact ties, and equal first derivatives
    never become a higher-order decision.
    """

    comparison: SineExchangeComparison
    observation: PhasePairObservation
    phase_rate_numerators: tuple[Q, ...]
    squared_chord_rate_bounds: tuple[I, ...]
    exact_nearest_groups: tuple[tuple[int, ...] | None, ...]
    outside_group_distance_margins: tuple[Q | None, ...]
    backward: SinePairingSide
    forward: SinePairingSide
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "one_complete_normalized_sine_comparison_capture_and_its_declared_clock",
        "phase_only_snapshot_observer_is_distinct_from_complete_law_time_derivatives",
        "exact_phase_rate_numerators_reuse_the_shared_sine_phase_row_owner",
        "strict_distance_margins_and_transverse_exact_ties_use_smooth_flow_continuity",
        "local_interval_existence_is_not_a_numerically_certified_time_horizon",
        "strict_mathematical_matching_does_not_guarantee_finite_precision_observer_resolution",
        "all_nodes_need_certified_choices_before_matching_or_nonmutuality_is_asserted",
        "paired_support_admission_is_separate_and_uses_only_the_same_fixed_capture",
        "support_admission_covers_complete_bipartite_fibers_and_positive_held_capacities",
        "no_future_pair_lift_collective_closure_or_protected_family_is_inferred",
        "no_graph_write_support_birth_controller_trajectory_or_native_dispatch",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project, _validate_label

        _validate_comparison_labels(self.comparison)
        for node in self.observation.nodes:
            _validate_label(node)
        for pair in self.observation.candidate_pairs or ():
            for node in pair:
                _validate_label(node)
        for side in (self.backward, self.forward):
            for pair in side.candidate_pairs or ():
                for node in pair:
                    _validate_label(node)
        return {
            "schema": "tnfr.relational-sine-pairing-transition.v1",
            "report": _project(self),
        }


def assess_sine_pairing_transition(
    graph, *, reference_model
) -> SinePairingTransitionAssessment:
    """Assess local mathematical pairing changes without evaluating a trajectory.

    The source may have any support admitted by the sine comparison. A complete
    candidate matching is checked against paired support only after its phase
    ordering has been proved. No prescribed grouping or lift is substituted.
    """
    comparison = bound_relational_sine_exchange(graph, reference_model=reference_model)
    observation = observe_phase_pairs(nodes=comparison.nodes, phases=comparison.phase)
    numerators = comparison.phase_rate_numerators()
    phases, nodes, pi = comparison.phase, comparison.nodes, pi_interval()
    rates = tuple(
        2
        * I(*certified_sine_bounds(phases[j] - phases[i]))
        * (numerators[j] - numerators[i])
        / pi
        for i, j in observation.distance_indices
    )
    lookup = dict(zip(observation.distance_indices, observation.squared_chord_bounds))
    groups_and_margins = tuple(
        _phase_nearest_group(phases, lambda i, j: lookup[min(i, j), max(i, j)], i)
        for i in range(len(nodes))
    )
    groups = tuple(group for group, _ in groups_and_margins)
    margins = tuple(margin for _, margin in groups_and_margins)

    def side(direction):
        nearest, statuses, factors, coefficient_margins, derivative_margins = (
            [],
            [],
            [],
            [],
            [],
        )
        for i, group in enumerate(groups):
            partner, factor, coefficient_margin, derivative_margin = (
                None,
                None,
                None,
                None,
            )
            if group is None:
                status = "distance_comparison_unresolved"
            elif len(group) == 1:
                partner, status = group[0], "strict_nearest_retained_by_continuity"
            else:
                gap = abs(phases[group[0]] - phases[i])
                factor = 2 * I(*certified_sine_bounds(gap)) / pi
                sign = 1 if factor.lo > 0 else (-1 if factor.hi < 0 else 0)
                coefficients = {
                    j: (1 if phases[j] >= phases[i] else -1)
                    * (numerators[j] - numerators[i])
                    for j in group
                }
                ordered = {
                    j: direction * sign * value for j, value in coefficients.items()
                }
                candidate = min(group, key=lambda j: ordered[j])
                coefficient_margin = min(
                    ordered[j] - ordered[candidate] for j in group if j != candidate
                )
                if sign and coefficient_margin > 0:
                    partner, status = (
                        candidate,
                        "exact_tie_split_by_strict_derivative_order",
                    )
                    derivative_margin = abs(factor) * coefficient_margin
                else:
                    status = "first_derivative_order_unresolved"
                    coefficient_margin = None
            nearest.append(partner)
            statuses.append(status)
            factors.append(factor)
            coefficient_margins.append(coefficient_margin)
            derivative_margins.append(derivative_margin)
        resolved, candidate_pairs, status = _phase_nearest_relation(nodes, nearest)
        reasons = (
            ()
            if candidate_pairs is not None
            else (
                ("strict_local_nearest_choices_are_not_mutual",)
                if resolved
                else ("one_or_more_local_nearest_choices_unresolved",)
            )
        )
        support_status, support_reasons = _pairing_support_admission(
            comparison, candidate_pairs
        )
        return SinePairingSide(
            time_direction=direction,
            nearest_partner_indices=tuple(nearest),
            node_statuses=tuple(statuses),
            tied_rate_factor_bounds=tuple(factors),
            tied_rate_coefficient_margins=tuple(coefficient_margins),
            tied_derivative_margin_bounds=tuple(derivative_margins),
            candidate_pairs=candidate_pairs,
            status=status,
            reasons=reasons,
            support_admission_status=support_status,
            support_admission_reasons=support_reasons,
            local_interval_existence_certified=resolved,
        )

    return SinePairingTransitionAssessment(
        comparison=comparison,
        observation=observation,
        phase_rate_numerators=numerators,
        squared_chord_rate_bounds=rates,
        exact_nearest_groups=groups,
        outside_group_distance_margins=margins,
        backward=side(-1),
        forward=side(1),
    )


def _margin_sine_terms(phases, numerators, triple):
    """Retain shared rational sine arguments before evaluating chord rates."""
    observer, alternative, preferred = triple
    terms = {}
    for target, orientation in ((alternative, 1), (preferred, -1)):
        gap = phases[target] - phases[observer]
        if not gap:
            continue
        coefficient = (
            2
            * orientation
            * (1 if gap > 0 else -1)
            * (numerators[target] - numerators[observer])
        )
        angle = abs(gap)
        terms[angle] = terms.get(angle, 0) + coefficient
    return terms


def _evaluate_margin_terms(terms):
    return (
        sum(
            (
                coefficient * I(*certified_sine_bounds(angle))
                for angle, coefficient in terms.items()
            ),
            I(0),
        )
        / pi_interval()
    )


def _subtract_margin_terms(left, right):
    terms = dict(left)
    for angle, coefficient in right.items():
        terms[angle] = terms.get(angle, 0) - coefficient
    return terms


@dataclass(frozen=True)
class SinePairingMobilityAssessment:
    """Two declared chord-margin rates under one frozen mobility alternative.

    Each triple is (observer, alternative, preferred), and its margin is
    D(observer,alternative)-D(observer,preferred). The normalized contrast is
    (rate[0]-rate[1])/(rate[0]+rate[1]); it remains unavailable when its sum
    is not separated from zero. Rational source sine coefficients preserve
    exact cancellations before evaluation, rather than interpreting interval
    overlap as equality. Actual ties are separate from prospective rates.
    """

    mobility: SineMobilityComparison
    observation: PhasePairObservation
    margin_triples: tuple[tuple[Any, Any, Any], ...]
    margin_indices: tuple[tuple[int, int, int], ...]
    initial_margin_bounds: tuple[I, ...]
    initial_exact_ties: tuple[bool, ...]
    sine_margin_rate_bounds: tuple[I, ...]
    mobility_margin_rate_bounds: tuple[I, ...]
    mobility_margin_correction_bounds: tuple[I, ...]
    sine_contrast_numerator_bounds: I
    mobility_contrast_numerator_bounds: I
    sine_contrast_denominator_bounds: I
    mobility_contrast_denominator_bounds: I
    sine_normalized_contrast_bounds: I | None
    mobility_normalized_contrast_bounds: I | None
    sine_contrast_status: str
    mobility_contrast_status: str
    source_margin_rate_difference_exact_zero: bool
    certified_time_horizon: None = None
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "one_shared_capture_and_two_explicitly_supplied_margin_triples",
        "triples_order_observer_alternative_preferred_for_squared_chord_margin",
        "finite_nonnegative_declared_epsilon_not_fitted_to_evaluated_response",
        "complete_reciprocal_alternative_changes_both_form_and_phase_rows",
        "exact_captured_initial_ties_do_not_follow_from_interval_overlap",
        "shared_sine_argument_cancellation_precedes_source_rate_evaluation",
        "normalized_rate_contrast_is_invariant_under_a_common_nonzero_clock_scale",
        "denominator_separation_is_required_for_each_normalized_contrast",
        "instantaneous_margins_not_a_generic_future_matching_or_uniform_horizon",
        "no_transfer_of_sine_window_recurrence_or_pulse_certificates",
        "no_new_source_search_solver_native_dispatch_or_physical_identification",
    )

    @property
    def comparison(self):
        return self.mobility.comparison

    def to_dict(self):
        from ..sdk.relational_reports import _project, _validate_label

        _validate_comparison_labels(self.comparison)
        for triple in self.margin_triples:
            for node in triple:
                _validate_label(node)
        for node in self.observation.nodes:
            _validate_label(node)
        for pair in self.observation.candidate_pairs or ():
            for node in pair:
                _validate_label(node)
        return {
            "schema": "tnfr.relational-sine-pairing-mobility.v1",
            "report": _project(self),
        }


def assess_sine_pairing_mobility(
    graph, *, reference_model, epsilon, margin_triples
) -> SinePairingMobilityAssessment:
    """Compare two declared initial-rate margins without changing the source law.

    Exactly two ordered (observer, alternative, preferred) triples are needed
    for the contrast. Nodes in each triple must be distinct captured labels.
    The observation owner's 64-node budget applies. No exact tie, nearest
    relation, common capacity or support pattern is silently imposed; these
    remain hypotheses of any theorem consuming the report.
    """
    comparison = bound_relational_sine_exchange(graph, reference_model=reference_model)
    mobility = comparison.with_current_squared_mobility(epsilon=epsilon)
    observation = observe_phase_pairs(nodes=comparison.nodes, phases=comparison.phase)
    raw = _ordered(margin_triples, "margin_triples", limit=3)
    if len(raw) != 2:
        raise ValueError("margin_triples requires exactly two declared triples")
    triples = tuple(_ordered(row, "each margin triple", limit=4) for row in raw)
    positions = {node: i for i, node in enumerate(comparison.nodes)}
    if any(
        len(row) != 3
        or len(set(row)) != 3
        or any(node not in positions for node in row)
        for row in triples
    ):
        raise ValueError("each margin triple needs three distinct captured node labels")
    indices = tuple(tuple(positions[node] for node in row) for row in triples)
    phases = comparison.phase
    ties = tuple(
        abs(phases[a] - phases[i]) == abs(phases[p] - phases[i]) for i, a, p in indices
    )
    distances = dict(
        zip(observation.distance_indices, observation.squared_chord_bounds)
    )
    margins = tuple(
        (
            I(0)
            if tied
            else distances[min(i, a), max(i, a)] - distances[min(i, p), max(i, p)]
        )
        for (i, a, p), tied in zip(indices, ties)
    )
    numerators = comparison.phase_rate_numerators()
    correction_numerators = tuple(
        n * f for n, f in zip(numerators, mobility.mobility_corrections)
    )
    source_terms = tuple(_margin_sine_terms(phases, numerators, row) for row in indices)
    correction_terms = tuple(
        _margin_sine_terms(phases, correction_numerators, row) for row in indices
    )
    source_rates = tuple(_evaluate_margin_terms(row) for row in source_terms)
    corrections = tuple(_evaluate_margin_terms(row) for row in correction_terms)
    alternative_rates = tuple(
        rate + correction for rate, correction in zip(source_rates, corrections)
    )
    source_difference_terms = _subtract_margin_terms(*source_terms)
    source_difference = _evaluate_margin_terms(source_difference_terms)
    difference = source_difference + _evaluate_margin_terms(
        _subtract_margin_terms(*correction_terms)
    )
    source_sum = sum(source_rates, I(0))
    alternative_sum = sum(alternative_rates, I(0))

    def contrast(numerator, denominator):
        if denominator.contains(0):
            return None, "denominator_not_separated_from_zero"
        return numerator / denominator, "available"

    sine_contrast, sine_status = contrast(source_difference, source_sum)
    changed_contrast, changed_status = contrast(difference, alternative_sum)
    return SinePairingMobilityAssessment(
        mobility=mobility,
        observation=observation,
        margin_triples=triples,
        margin_indices=indices,
        initial_margin_bounds=margins,
        initial_exact_ties=ties,
        sine_margin_rate_bounds=source_rates,
        mobility_margin_rate_bounds=alternative_rates,
        mobility_margin_correction_bounds=corrections,
        sine_contrast_numerator_bounds=source_difference,
        mobility_contrast_numerator_bounds=difference,
        sine_contrast_denominator_bounds=source_sum,
        mobility_contrast_denominator_bounds=alternative_sum,
        sine_normalized_contrast_bounds=sine_contrast,
        mobility_normalized_contrast_bounds=changed_contrast,
        sine_contrast_status=sine_status,
        mobility_contrast_status=changed_status,
        source_margin_rate_difference_exact_zero=all(
            value == 0 for value in source_difference_terms.values()
        ),
    )


@dataclass(frozen=True)
class SinePairingWindowAssessment:
    """Whole-window nearest relation for a declared full initial uncertainty box.

    All error radii are independent initial errors in captured node order at
    structural time zero. Held capacities and support are exact. Bounds
    apply to captured centers plus these exact radii; initial_*_bounds are
    outward display enclosures, not an independently enlarged admitted box.
    Global bounds on the complete conservative sine field control phase
    acceleration, so the affine center plus remainder encloses every admitted
    nonlinear solution.
    This is an analytic enclosure, not a numerical trajectory or a new solver.
    """

    comparison: SineExchangeComparison
    form_error_bounds: tuple[Q, ...]
    phase_error_bounds: tuple[Q, ...]
    initial_form_bounds: tuple[I, ...]
    initial_phase_bounds: tuple[I, ...]
    window: tuple[Q, Q]
    phase_rate_numerators: tuple[Q, ...]
    form_speed_upper_bounds: tuple[Q, ...]
    initial_phase_rate_error_bounds: tuple[Q, ...]
    phase_acceleration_upper_bounds: tuple[Q, ...]
    phase_remainder_upper_bounds: tuple[Q, ...]
    phase_window_bounds: tuple[I, ...]
    distance_indices: tuple[tuple[int, int], ...]
    phase_gap_window_bounds: tuple[I, ...]
    squared_chord_window_bounds: tuple[I, ...]
    chord_bound_methods: tuple[str, ...]
    nearest_partner_indices: tuple[int | None, ...]
    nearest_separation_margins: tuple[Q | None, ...]
    candidate_pairs: tuple[tuple[Any, Any], ...] | None
    status: str
    reasons: tuple[str, ...]
    whole_window_nearest_relation_certified: bool
    support_admission_status: str
    support_admission_reasons: tuple[str, ...]
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "complete_conservative_normalized_sine_law_explicit_zero_form_loss",
        "independent_initial_errors_in_all_fine_form_and_real_phase_coordinates",
        "fixed_support_and_held_captured_capacities_no_input_or_events",
        "global_full_field_speed_and_acceleration_bounds_retain_nonlinear_feedback",
        "initial_box_is_supplied_analysis_evidence_not_measurement_authentication",
        "whole_closed_future_window_not_only_a_center_or_endpoint",
        "exact_phase_rate_numerators_and_common_time_factor_retained_in_pair_gaps",
        "monotone_chord_endpoint_bounds_only_on_proved_absolute_gap_range_below_pi",
        "generic_trigonometric_fallback_elsewhere_without_automatic_unwrapping",
        "all_nodes_and_all_competitors_needed_before_complete_relation_certification",
        "wide_enclosures_are_unavailable_not_a_refuted_prediction",
        "strict_mathematical_relation_is_not_a_finite_precision_sensor_contract",
        "support_admission_is_independent_of_phase_observation",
        "no_graph_write_solver_trajectory_support_birth_or_permanent_formation_claim",
    )

    def joint_observation(self):
        """Add form to this unchanged phase-window observation.

        Use the same captured complete field, initial uncertainty box and
        declared window. Form speed bounds provide a sufficient joint tube;
        no dynamics, matching policy or existing phase-only verdict changes.
        """
        return _joint_pairing_projection(self)

    def to_dict(self):
        from ..sdk.relational_reports import _project, _validate_label

        _validate_comparison_labels(self.comparison)
        for pair in self.candidate_pairs or ():
            for node in pair:
                _validate_label(node)
        return {
            "schema": "tnfr.relational-sine-pairing-window.v1",
            "report": _project(self),
        }


@dataclass(frozen=True)
class SineJointPairingProjection:
    """Joint form/phase observation of one already admitted phase window.

    Distances are squared, using the source law's beta. The initial center,
    complete initial error box and whole future window have separate nearest
    observations. ``same_pairing_as_initial_box`` is available only when both
    bounded observations give complete mutual pairings; it does not claim
    persistence during a gap before the declared window. Instantaneous Jdot
    belongs to the exact captured center, not every uncertain preparation.

    Support admission is the existing strict replica contract, not a test of
    every possible symmetry or of physical pattern identity. This report
    neither authenticates a manually supplied dataclass nor upgrades the
    independent K2,2 reference theorem to another support. All pair-indexed
    arrays follow ``phase_window.distance_indices`` in captured node order.
    """

    phase_window: SinePairingWindowAssessment
    initial_observation: JointPairObservation
    source_joint_distance_rate_bounds: tuple[I, ...]
    form_remainder_bounds: tuple[Q, ...]
    form_gap_window_bounds: tuple[I, ...]
    initial_box_joint_distance_bounds: tuple[I, ...]
    initial_box_nearest_partner_indices: tuple[int | None, ...]
    initial_box_nearest_separation_margins: tuple[Q | None, ...]
    initial_box_candidate_pairs: tuple[tuple[Any, Any], ...] | None
    initial_box_status: str
    initial_box_reasons: tuple[str, ...]
    joint_distance_window_bounds: tuple[I, ...]
    nearest_partner_indices: tuple[int | None, ...]
    nearest_separation_margins: tuple[Q | None, ...]
    candidate_pairs: tuple[tuple[Any, Any], ...] | None
    status: str
    reasons: tuple[str, ...]
    whole_window_pairing_certified: bool
    same_pairing_as_initial_box: bool | None
    support_admission_status: str
    support_admission_reasons: tuple[str, ...]
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "same_captured_conservative_sine_field_initial_full_box_and_declared_window",
        "form_tube_uses_captured_global_speed_bound_without_a_new_evolution_kernel",
        "phase_chord_bounds_are_the_existing_sharp_phase_window_enclosures",
        "squared_joint_distance_uses_both_signed_form_and_circular_phase",
        "instantaneous_joint_distance_rates_use_both_complete_captured_rows",
        "initial_exact_center_initial_box_and_whole_window_observations_remain_distinct",
        "strict_mutual_nearest_admission_checks_every_competing_partner",
        "unavailable_pairings_do_not_compare_equal_or_prove_observation_failure",
        "equal_initial_and_window_pairings_do_not_certify_any_unobserved_time_gap",
        "support_admission_is_strict_replica_only_not_a_complete_quotient_test",
        "phase_projection_and_joint_observation_are_not_interchangeable_identity_claims",
        "no_dataclass_authentication_K2_2_theorem_transfer_or_physical_identification",
        "no_graph_reread_solver_source_change_event_or_existing_verdict_rewrite",
    )

    @property
    def comparison(self):
        return self.phase_window.comparison

    @property
    def window(self):
        return self.phase_window.window

    def to_dict(self):
        from ..sdk.relational_reports import _project, _validate_label

        _validate_comparison_labels(self.comparison)
        for node in self.initial_observation.nodes:
            _validate_label(node)
        for pairs in (
            self.phase_window.candidate_pairs,
            self.initial_observation.candidate_pairs,
            self.initial_box_candidate_pairs,
            self.candidate_pairs,
        ):
            for pair in pairs or ():
                for node in pair:
                    _validate_label(node)
        return {
            "schema": "tnfr.relational-sine-joint-pairing-projection.v1",
            "report": _project(self),
        }


def _joint_pairing_projection(phase_window):
    """Reuse one admitted full-field window to observe another state projection."""
    comparison = phase_window.comparison
    nodes, x, theta = comparison.nodes, comparison.epi, comparison.phase
    beta = Q(comparison.reference_model.storage_scale)
    initial = observe_joint_pairs(
        nodes=nodes, forms=x, phases=theta, storage_scale=beta
    )
    indices = phase_window.distance_indices
    numerators = comparison.phase_rate_numerators()
    pi = pi_interval()
    rates = tuple(
        2 * (x[j] - x[i]) * (comparison.form_rates[j] - comparison.form_rates[i])
        + 2
        * beta
        * I(*certified_sine_bounds(theta[j] - theta[i]))
        * (I(numerators[j] - numerators[i]) / pi)
        for i, j in indices
    )
    form_errors, phase_errors = (
        phase_window.form_error_bounds,
        phase_window.phase_error_bounds,
    )
    radii = tuple(
        error + speed * phase_window.window[1]
        for error, speed in zip(form_errors, phase_window.form_speed_upper_bounds)
    )
    form_gaps = tuple(
        I(x[j] - x[i] - radii[i] - radii[j], x[j] - x[i] + radii[i] + radii[j])
        for i, j in indices
    )
    initial_bounds, window_bounds = [], []
    for k, (i, j) in enumerate(indices):
        form_error = form_errors[i] + form_errors[j]
        phase_error = phase_errors[i] + phase_errors[j]
        initial_form = I(x[j] - x[i] - form_error, x[j] - x[i] + form_error)
        initial_phase = I(
            theta[j] - theta[i] - phase_error, theta[j] - theta[i] + phase_error
        )
        chord, _ = _phase_gap_chord_bound(initial_phase)
        raw_initial = initial_form**2 + beta * chord
        raw_window = (
            form_gaps[k] ** 2 + beta * phase_window.squared_chord_window_bounds[k]
        )
        initial_bounds.append(I(max(Q(0), raw_initial.lo), max(Q(0), raw_initial.hi)))
        window_bounds.append(I(max(Q(0), raw_window.lo), max(Q(0), raw_window.hi)))
    initial_bounds, window_bounds = tuple(initial_bounds), tuple(window_bounds)
    (
        initial_nearest,
        initial_margins,
        initial_resolved,
        initial_pairs,
        initial_status,
    ) = _strict_pairing_from_bounds(nodes, indices, initial_bounds)
    nearest, margins, resolved, pairs, status = _strict_pairing_from_bounds(
        nodes, indices, window_bounds
    )

    def reasons(resolved, pairs):
        if pairs is not None:
            return ()
        return (
            (
                "strict_joint_nearest_choices_are_not_mutual"
                if resolved
                else "one_or_more_strict_joint_nearest_choices_unresolved"
            ),
        )

    same = None
    if initial_pairs is not None and pairs is not None:
        same = {frozenset(pair) for pair in initial_pairs} == {
            frozenset(pair) for pair in pairs
        }
    support_status, support_reasons = _pairing_support_admission(comparison, pairs)
    return SineJointPairingProjection(
        phase_window=phase_window,
        initial_observation=initial,
        source_joint_distance_rate_bounds=rates,
        form_remainder_bounds=radii,
        form_gap_window_bounds=form_gaps,
        initial_box_joint_distance_bounds=initial_bounds,
        initial_box_nearest_partner_indices=initial_nearest,
        initial_box_nearest_separation_margins=initial_margins,
        initial_box_candidate_pairs=initial_pairs,
        initial_box_status=initial_status,
        initial_box_reasons=reasons(initial_resolved, initial_pairs),
        joint_distance_window_bounds=window_bounds,
        nearest_partner_indices=nearest,
        nearest_separation_margins=margins,
        candidate_pairs=pairs,
        status=status,
        reasons=reasons(resolved, pairs),
        whole_window_pairing_certified=pairs is not None,
        same_pairing_as_initial_box=same,
        support_admission_status=support_status,
        support_admission_reasons=support_reasons,
    )


def _phase_gap_chord_bound(gap):
    """Tight monotone scalar bounds where proved, shared global bounds elsewhere."""
    absolute = abs(gap)
    if absolute.hi <= pi_interval().lo:
        _, upper_at_lower = certified_cosine_bounds(absolute.lo)
        lower_at_upper, _ = certified_cosine_bounds(absolute.hi)
        return (
            I(2 * (1 - upper_at_lower), 2 * (1 - lower_at_upper)),
            "monotone_scalar_endpoints_on_absolute_gap_within_pi",
        )
    return 2 * (1 - cos(gap)), "shared_interval_cosine_without_lift_inference"


def assess_sine_pairing_window(
    graph,
    *,
    reference_model,
    form_error_bounds,
    phase_error_bounds,
    window_start,
    window_end,
) -> SinePairingWindowAssessment:
    """Bound phase pairing for an entire future window and full initial box.

    Require 0<=window_start<window_end. Radius vectors are mandatory; no exact
    pair synchronization or form preparation is silently substituted. An
    unavailable enclosure says nothing about failure of the actual grouping.
    """
    from .relational_sine_pattern import _error_radii

    start = exact_or_represented_real(window_start, "window_start")
    end = exact_or_represented_real(window_end, "window_end")
    if not 0 <= start < end:
        raise ValueError("require 0<=window_start<window_end")
    comparison = bound_relational_sine_exchange(graph, reference_model=reference_model)
    if Q(reference_model.effective_weights[0]) != 0:
        raise ValueError("pairing window requires explicit zero form loss")
    size = len(comparison.nodes)
    if size > 64:
        raise ValueError(
            "phase pairing permits at most 64 nodes within its work budget"
        )
    form_errors = _error_radii(form_error_bounds, size, "form_error_bounds")
    phase_errors = _error_radii(phase_error_bounds, size, "phase_error_bounds")
    neighbors = _comparison_neighbors(comparison)
    nu = comparison.capacity
    pi = pi_interval()
    w, beta = Q(reference_model.effective_weights[1]), Q(reference_model.storage_scale)
    a, b = w / pi.lo, w / (beta * pi.lo)
    speed = tuple(a * capacity for capacity in nu)
    rate_error = tuple(
        b
        * nu[i]
        * (form_errors[i] + sum((form_errors[j] for j in row), Q(0)) / len(row))
        for i, row in enumerate(neighbors)
    )
    acceleration = tuple(
        b * nu[i] * (speed[i] + sum((speed[j] for j in row), Q(0)) / len(row))
        for i, row in enumerate(neighbors)
    )
    remainder = tuple(
        phase_errors[i] + rate_error[i] * end + acceleration[i] * end**2 / 2
        for i in range(size)
    )
    numerators, time = comparison.phase_rate_numerators(), I(start, end)
    phase_bounds = tuple(
        I(center) + I(numerators[i]) / pi * time + I(-remainder[i], remainder[i])
        for i, center in enumerate(comparison.phase)
    )
    indices = tuple((i, j) for i in range(size) for j in range(i + 1, size))
    gaps = tuple(
        I(comparison.phase[j] - comparison.phase[i])
        + I(numerators[j] - numerators[i]) / pi * time
        + I(-remainder[i] - remainder[j], remainder[i] + remainder[j])
        for i, j in indices
    )
    chord_evidence = tuple(_phase_gap_chord_bound(gap) for gap in gaps)
    chords = tuple(bound for bound, _ in chord_evidence)
    nearest, margins, resolved, pairs, status = _strict_pairing_from_bounds(
        comparison.nodes, indices, chords
    )
    reasons = (
        ()
        if pairs is not None
        else (
            ("whole_window_nearest_choices_are_not_mutual",)
            if resolved
            else ("one_or_more_whole_window_nearest_choices_unresolved",)
        )
    )
    support_status, support_reasons = _pairing_support_admission(comparison, pairs)
    return SinePairingWindowAssessment(
        comparison=comparison,
        form_error_bounds=form_errors,
        phase_error_bounds=phase_errors,
        initial_form_bounds=tuple(
            I(value - error, value + error)
            for value, error in zip(comparison.epi, form_errors)
        ),
        initial_phase_bounds=tuple(
            I(value - error, value + error)
            for value, error in zip(comparison.phase, phase_errors)
        ),
        window=(start, end),
        phase_rate_numerators=numerators,
        form_speed_upper_bounds=speed,
        initial_phase_rate_error_bounds=rate_error,
        phase_acceleration_upper_bounds=acceleration,
        phase_remainder_upper_bounds=remainder,
        phase_window_bounds=phase_bounds,
        distance_indices=indices,
        phase_gap_window_bounds=gaps,
        squared_chord_window_bounds=chords,
        chord_bound_methods=tuple(method for _, method in chord_evidence),
        nearest_partner_indices=tuple(nearest),
        nearest_separation_margins=tuple(margins),
        candidate_pairs=pairs,
        status=status,
        reasons=reasons,
        whole_window_nearest_relation_certified=resolved,
        support_admission_status=support_status,
        support_admission_reasons=support_reasons,
    )


@dataclass(frozen=True)
class SineJointPairingWindowAssessment:
    """Joint identity around one exact synchronized two-region exchange.

    The supplied K2,2 reference has equal initial forms and synchronized
    members in each pair. Its interpair squared joint distance J* is constant.
    Independent fine-node errors use the full real-lift norm
    r0**2=sum_i(dx_i**2+beta*dtheta_i**2); no perturbation is constrained to
    remain synchronized. The global complete-field Lipschitz bound gives
    r(t)**2<=r0**2*exp(2*L*T). The sufficient budget is J*>8*r(T)**2.

    Exact reference identity, resolved interval comparisons and a perturbed
    finite-window certificate are separate. An unavailable numerical bound
    does not refute the reference theorem or the actual perturbed identity.
    """

    replica: SineReplicaScaleAssessment
    form_error_bounds: tuple[Q, ...]
    phase_error_bounds: tuple[Q, ...]
    window: tuple[Q, Q]
    reference_phase_gap_radian_part: Q
    reference_phase_gap_turn_part: Q
    reference_phase_gap_bounds: I
    reference_joint_separation_squared_bounds: I
    initial_scaled_error_squared: Q
    lipschitz_upper: Q
    growth_exponent_upper: Q
    squared_growth_factor_upper: Q | None
    propagated_scaled_error_squared_upper: Q | None
    identification_budget_margin: Q | None
    distance_indices: tuple[tuple[int, int], ...]
    joint_distance_window_bounds: tuple[I, ...] | None
    nearest_partner_indices: tuple[int | None, ...]
    nearest_separation_margins: tuple[Q | None, ...]
    candidate_pairs: tuple[tuple[Any, Any], ...] | None
    status: str
    reasons: tuple[str, ...]
    reference_identity_certified: bool
    whole_window_identity_certified: bool
    reference_period_bounds: I | None
    period_unavailable_reasons: tuple[str, ...]
    window_covers_reference_period: bool
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "conservative_normalized_sine_law_fixed_unit_K2_2_common_positive_capacity",
        "exact_synchronized_equal_form_reference_and_explicit_subpi_interpair_lift",
        "storage_induced_joint_distance_not_a_uniquely_derived_physical_metric",
        "reference_within_pair_distance_zero_cross_distance_constant_by_inherited_energy",
        "independent_initial_errors_in_all_four_signed_forms_and_all_four_real_phase_lifts",
        "full_scaled_Euclidean_lift_norm_controls_embedded_circular_point_distance",
        "global_complete_field_Lipschitz_bound_not_an_embedded_field_or_linearization",
        "finite_closed_window_budget_and_strict_nearest_comparisons_are_both_required",
        "reference_all_time_identity_does_not_transfer_to_permanent_perturbed_identity",
        "zero_error_bypasses_exponential_evaluation_without_removing_reference_checks",
        "unresolved_interval_gap_or_overflow_is_abstention_not_a_counterexample",
        "reference_period_bound_is_not_a_period_or_recurrence_of_perturbed_solutions",
        "supplied_partition_support_and_preparation_are_not_formation_or_attraction",
        "no_graph_write_solver_operator_selector_new_law_or_physical_identification",
    )

    @property
    def comparison(self):
        return self.replica.comparison

    @property
    def pairs(self):
        return self.replica.pairs

    def to_dict(self):
        from ..sdk.relational_reports import _project, _validate_label

        _validate_replica_labels(self)
        for pair in self.candidate_pairs or ():
            for node in pair:
                _validate_label(node)
        return {
            "schema": "tnfr.relational-sine-joint-pairing-window.v1",
            "report": _project(self),
        }


def assess_sine_joint_pairing_window(
    graph,
    *,
    reference_model,
    pairs,
    form_error_bounds,
    phase_error_bounds,
    window_end,
    phase_turns=None,
) -> SineJointPairingWindowAssessment:
    """Bound joint nearest identity through a declared reference exchange.

    The graph supplies an exact reference, not a perturbed sample to be fitted.
    Error vectors declare independent initial radii about every captured form
    and supplied continuous phase lift. Their Euclidean norm controls the
    circular observation without identifying phase with a globally real angle.
    The same law, support, capacity and structural clock are held throughout
    [0, window_end]. All perturbations evolve under the full four-node law.
    """
    from math import isfinite

    from .._exact_time import exp_upper_float
    from .relational_sine_pattern import _error_radii
    from .relational_sine_resonance import _elliptic_libration_period_bounds

    end = exact_or_represented_real(window_end, "window_end")
    if end <= 0:
        raise ValueError("window_end must be positive")
    replica = assess_sine_replica_scale(
        graph, reference_model=reference_model, pairs=pairs, phase_turns=phase_turns
    )
    comparison = replica.comparison
    if len(comparison.nodes) != 4 or len(replica.pairs) != 2:
        raise ValueError("joint identity window requires two replica pairs on K2,2")
    if len(set(comparison.capacity)) != 1:
        raise ValueError("joint identity window requires common positive held capacity")
    if not replica.source_in_synchronized_submanifold:
        raise ValueError(
            "joint identity window requires an exact synchronized reference"
        )
    if len(set(replica.form_means)) != 1:
        raise ValueError("joint identity reference must have equal initial forms")

    form_errors = _error_radii(form_error_bounds, 4, "form_error_bounds")
    phase_errors = _error_radii(phase_error_bounds, 4, "phase_error_bounds")
    raw_gap = replica.phase_mean_radian_parts[1] - replica.phase_mean_radian_parts[0]
    gap_turns = replica.phase_mean_turn_parts[1] - replica.phase_mean_turn_parts[0]
    pi = pi_interval()
    gap = I(raw_gap) + (2 * gap_turns) * pi
    if abs(gap).hi >= pi.lo:
        raise ValueError("reference interpair phase lift must be certified below pi")
    nonzero = raw_gap != 0 or gap_turns != 0
    _, cosine = _pi_shifted_trig(raw_gap, int(2 * gap_turns))
    beta = Q(reference_model.storage_scale)
    w = Q(reference_model.effective_weights[1])
    nu = comparison.capacity[0]
    raw_separation = 2 * beta * (1 - cosine)
    separation = I(max(Q(0), raw_separation.lo), max(Q(0), raw_separation.hi))
    initial_error = sum(
        (x**2 + beta * theta**2 for x, theta in zip(form_errors, phase_errors)), Q(0)
    )
    # Divide exact beta before the square root: a tiny positive scale must not
    # become an artificial zero interval denominator.
    lipschitz = 2 * w * nu * sqrt(1 / beta).hi / pi.lo
    exponent = 2 * lipschitz * end
    amplification = None
    propagated = Q(0) if initial_error == 0 else None
    if initial_error:
        upper = exp_upper_float(exponent)
        if isfinite(upper):
            amplification = Q(upper)
            propagated = initial_error * amplification
    margin = None if propagated is None else separation.lo - 8 * propagated

    indices = tuple((i, j) for i in range(4) for j in range(i + 1, 4))
    distances = None
    nearest, nearest_margins, candidate = (None,) * 4, (None,) * 4, None
    if propagated is not None:
        radius = sqrt(2 * propagated)
        reference_distance = sqrt(separation)
        lower = max(Q(0), reference_distance.lo - radius.hi)
        cross = I(lower**2, (reference_distance.hi + radius.hi) ** 2)
        within = I(0, 2 * propagated)
        by_node = {node: k for k, pair in enumerate(replica.pairs) for node in pair}
        distances = tuple(
            (
                within
                if by_node[comparison.nodes[i]] == by_node[comparison.nodes[j]]
                else cross
            )
            for i, j in indices
        )
        nearest, nearest_margins, _, candidate, _ = _strict_pairing_from_bounds(
            comparison.nodes, indices, distances
        )
    certified = nonzero and margin is not None and margin > 0 and candidate is not None
    reasons = []
    if not nonzero:
        reasons.append("zero_reference_amplitude")
    if propagated is None:
        reasons.append("squared_error_amplification_not_representable")
    elif separation.lo <= 0 and nonzero:
        reasons.append("positive_reference_separation_not_resolved_numerically")
    elif margin <= 0 and nonzero:
        reasons.append("uncertainty_exceeds_sufficient_identification_budget")
    elif candidate is None and nonzero:
        reasons.append("strict_joint_nearest_comparison_unresolved")

    period = None
    period_reasons = ("zero_reference_amplitude",)
    if nonzero:
        base = pi**2 * sqrt(beta)
        small_period = I(base.lo / (w * nu), base.hi / (w * nu))
        period, period_reasons = _elliptic_libration_period_bounds(
            small_period, (1 + cosine) / 2
        )
    return SineJointPairingWindowAssessment(
        replica=replica,
        form_error_bounds=form_errors,
        phase_error_bounds=phase_errors,
        window=(Q(0), end),
        reference_phase_gap_radian_part=raw_gap,
        reference_phase_gap_turn_part=gap_turns,
        reference_phase_gap_bounds=gap,
        reference_joint_separation_squared_bounds=separation,
        initial_scaled_error_squared=initial_error,
        lipschitz_upper=lipschitz,
        growth_exponent_upper=exponent,
        squared_growth_factor_upper=amplification,
        propagated_scaled_error_squared_upper=propagated,
        identification_budget_margin=margin,
        distance_indices=indices,
        joint_distance_window_bounds=distances,
        nearest_partner_indices=nearest,
        nearest_separation_margins=nearest_margins,
        candidate_pairs=candidate if certified else None,
        status="certified" if certified else "unavailable",
        reasons=tuple(reasons),
        reference_identity_certified=nonzero,
        whole_window_identity_certified=certified,
        reference_period_bounds=period,
        period_unavailable_reasons=period_reasons,
        window_covers_reference_period=period is not None and end >= period.hi,
    )


@dataclass(frozen=True)
class SinePairSwapWitness:
    """One exact state-only swap on retained support, not a graph relabeling."""

    pair_index: int
    permutation_indices: tuple[int, ...]
    swapped_form: tuple[Q, ...]
    swapped_phase: tuple[Q, ...]
    swapped_form_gradient: tuple[Q, ...]
    swapped_phase_rate_numerators: tuple[Q, ...]
    swapped_form_rate_bounds: tuple[I, ...]
    swapped_phase_rate_bounds: tuple[I, ...]
    form_equivariance_residual: tuple[I, ...]
    phase_equivariance_numerator_residual: tuple[Q, ...]
    source_pair_mean_phase_rate_numerators: tuple[Q, ...]
    swapped_pair_mean_phase_rate_numerators: tuple[Q, ...]
    pair_mean_phase_rate_numerator_difference: tuple[Q, ...]
    pair_mean_phase_rate_difference_bounds: tuple[I, ...]
    source_orbit_equal: bool
    collective_phase_rate_obstruction_certified: bool


@dataclass(frozen=True)
class SinePairSupportSymmetryAssessment:
    """Exact fixed-support symmetry criterion, with an optional actual witness.

    For the admitted common positive capacity, each independent member swap
    commutes with the full sine field precisely when it is a graph
    automorphism. External neighbor sets must agree; a within-pair edge is
    allowed. The linear phase row proves necessity from its off-diagonal
    support. Sine superposition and preserved degrees prove sufficiency.

    ``phase_row_commutator_witnesses`` retain one (row, column, numerator)
    entry of G[P(i),P(j)]-G[i,j], where theta'=G*x/pi, per broken swap.
    No dense group enumeration, tolerance or observed zero residual proves the
    all-state criterion. ``witness`` compares pair means of supplied continuous
    real phase lifts as a local observable; it does not identify these means
    with primitive phase or infer a global circular chart.
    """

    comparison: SineExchangeComparison
    pairs: tuple[tuple[Any, Any], ...]
    pair_indices: tuple[tuple[int, int], ...]
    within_pair_edge_indices: tuple[int, ...]
    cross_block_edge_counts: tuple[tuple[int, int, int], ...]
    incomplete_cross_blocks: tuple[tuple[int, int, int], ...]
    external_neighbor_difference_indices: tuple[tuple[int, ...], ...]
    pair_swap_symmetry: tuple[bool, ...]
    phase_row_commutator_witnesses: tuple[tuple[int, int, Q] | None, ...]
    independent_pair_swaps_equivariant: bool
    unordered_pair_quotient_status: str
    strict_replica_admission_status: str
    strict_replica_admission_reasons: tuple[str, ...]
    witness: SinePairSwapWitness | None
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "complete_conservative_normalized_sine_law_common_positive_held_capacity",
        "fixed_simple_connected_unit_support_and_declared_ordered_pair_partition",
        "independent_whole_member_form_and_phase_swaps_on_fixed_support",
        "all_state_equivariance_is_distinct_from_one_captured_residual",
        "complete_or_empty_cross_blocks_allow_independent_internal_pair_edges",
        "equitable_neighbor_counts_alone_do_not_prove_swap_equivariance",
        "strict_no_internal_edge_replica_formulas_keep_their_own_admission",
        "optional_witness_is_an_exact_hypothetical_state_not_an_executed_event",
        "raw_pair_multiset_identity_proves_the_same_orbit_without_interval_equality",
        "zero_chosen_witness_difference_does_not_rescue_failed_all_state_closure",
        "unordered_orbits_remove_member_labels_not_internal_degrees_of_freedom",
        "no_graph_write_support_selection_trajectory_or_native_dispatch",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        _validate_replica_labels(self)
        return {
            "schema": "tnfr.relational-sine-pair-support-symmetry.v1",
            "report": _project(self),
        }


def assess_sine_pair_support_symmetry(
    graph, *, reference_model, pairs, witness_pair=None
) -> SinePairSupportSymmetryAssessment:
    """Check independent member swaps and optionally compare one source swap.

    ``witness_pair`` is an explicit nonboolean index in ``pairs``. No failed
    swap is selected automatically, and no graph or capacity is swapped. The
    support criterion is meaningful even when the chosen state has zero rates.
    """
    comparison = bound_relational_sine_exchange(graph, reference_model=reference_model)
    return _assess_sine_pair_support_symmetry(comparison, pairs, witness_pair)


def _assess_sine_pair_support_symmetry(comparison, pairs, witness_pair=None):
    """Reuse one authoritative capture for symmetry and dependent observations."""
    reference_model = comparison.reference_model
    if Q(reference_model.effective_weights[0]) != 0:
        raise ValueError("pair support symmetry requires explicit zero form loss")
    if len(set(comparison.capacity)) != 1 or comparison.capacity[0] <= 0:
        raise ValueError("pair support symmetry requires common positive held capacity")
    size = len(comparison.nodes)
    if size > 64:
        raise ValueError(
            "pair support symmetry permits at most 64 nodes within its work budget"
        )
    pairs, indices = _replica_partition(comparison, pairs)
    if witness_pair is not None and (
        type(witness_pair) is not int or not 0 <= witness_pair < len(pairs)
    ):
        raise ValueError("witness_pair must be a nonboolean pair index")
    neighbors = _comparison_neighbors(comparison)
    neighbor_sets = tuple(set(row) for row in neighbors)
    differences = tuple(
        tuple(sorted((neighbor_sets[i] - {j}) ^ (neighbor_sets[j] - {i})))
        for i, j in indices
    )
    symmetry = tuple(not row for row in differences)
    commutators = []
    for (i, j), row in zip(indices, differences):
        if not row:
            commutators.append(None)
            continue
        column = row[0]
        basis_gradient = tuple(
            Q(
                comparison.degrees[k] * int(k == column)
                - int(column in neighbor_sets[k])
            )
            for k in range(size)
        )
        basis_numerators = _sine_phase_rate_numerators(
            reference_model, comparison.degrees, basis_gradient, comparison.capacity
        )
        commutators.append((i, column, basis_numerators[j] - basis_numerators[i]))
    counts = tuple(
        (
            i,
            j,
            sum(
                int(right in neighbor_sets[left])
                for left in indices[i]
                for right in indices[j]
            ),
        )
        for i in range(len(pairs))
        for j in range(i + 1, len(pairs))
    )
    internal = tuple(k for k, (i, j) in enumerate(indices) if j in neighbor_sets[i])
    strict_status, strict_reasons = _pairing_support_admission(comparison, pairs)
    witness = None
    if witness_pair is not None:
        permutation = list(range(size))
        left, right = indices[witness_pair]
        permutation[left], permutation[right] = right, left
        epi = tuple(comparison.epi[j] for j in permutation)
        phase = tuple(comparison.phase[j] for j in permutation)
        gradient = tuple(
            sum((epi[i] - epi[j] for j in row), Q(0)) for i, row in enumerate(neighbors)
        )
        numerators = _sine_phase_rate_numerators(
            reference_model, comparison.degrees, gradient, comparison.capacity
        )
        currents = tuple(
            I(*imag) for _, imag in relative_resultant_bounds(phase, neighbors)
        )
        rates = _sine_rates(
            reference_model, comparison.degrees, gradient, comparison.capacity, currents
        )
        source_numerators = comparison.phase_rate_numerators()
        source_means = tuple(
            (source_numerators[i] + source_numerators[j]) / 2 for i, j in indices
        )
        swapped_means = tuple((numerators[i] + numerators[j]) / 2 for i, j in indices)
        mean_difference = tuple(
            right - left for left, right in zip(source_means, swapped_means)
        )
        orbit_equal = all(
            sorted((comparison.epi[k], comparison.phase[k]) for k in pair)
            == sorted((epi[k], phase[k]) for k in pair)
            for pair in indices
        )
        witness = SinePairSwapWitness(
            pair_index=witness_pair,
            permutation_indices=tuple(permutation),
            swapped_form=epi,
            swapped_phase=phase,
            swapped_form_gradient=gradient,
            swapped_phase_rate_numerators=numerators,
            swapped_form_rate_bounds=rates["form_rates"],
            swapped_phase_rate_bounds=rates["phase_rates"],
            form_equivariance_residual=tuple(
                rates["form_rates"][i] - comparison.form_rates[j]
                for i, j in enumerate(permutation)
            ),
            phase_equivariance_numerator_residual=tuple(
                numerators[i] - source_numerators[j] for i, j in enumerate(permutation)
            ),
            source_pair_mean_phase_rate_numerators=source_means,
            swapped_pair_mean_phase_rate_numerators=swapped_means,
            pair_mean_phase_rate_numerator_difference=mean_difference,
            pair_mean_phase_rate_difference_bounds=tuple(
                I(value) / pi_interval() for value in mean_difference
            ),
            source_orbit_equal=orbit_equal,
            collective_phase_rate_obstruction_certified=orbit_equal
            and any(mean_difference),
        )
    equivariant = all(symmetry)
    return SinePairSupportSymmetryAssessment(
        comparison=comparison,
        pairs=pairs,
        pair_indices=indices,
        within_pair_edge_indices=internal,
        cross_block_edge_counts=counts,
        incomplete_cross_blocks=tuple(item for item in counts if item[2] not in (0, 4)),
        external_neighbor_difference_indices=differences,
        pair_swap_symmetry=symmetry,
        phase_row_commutator_witnesses=tuple(commutators),
        independent_pair_swaps_equivariant=equivariant,
        unordered_pair_quotient_status=(
            "certified_by_independent_swap_symmetry"
            if equivariant
            else "obstructed_by_fixed_support"
        ),
        strict_replica_admission_status=strict_status,
        strict_replica_admission_reasons=strict_reasons,
        witness=witness,
    )


@dataclass(frozen=True)
class SineMixedPairCoordinates:
    """Retained exact coordinate expressions, with outward trig evaluations.

    An unordered row carries R=cos(delta), U=u**2 and Q=u*sin(delta).
    Its interval evaluations are not independent exact coordinates or an
    uncertainty inverse. An ordered row retains signed u and delta instead.
    Phase parts represent radians + 2*pi*turns in the supplied local lift.
    """

    mode: str
    form_mean: Q
    phase_mean_radian_part: Q
    phase_mean_turn_part: Q
    form_half_difference: Q | None
    phase_half_difference_radian_part: Q | None
    phase_half_difference_turn_part: Q | None
    resultant_magnitude_bounds: I | None
    internal_form_squared: Q | None
    form_phase_correlation_bounds: I | None
    reconstruction_stratum: str


@dataclass(frozen=True)
class SineMixedPairRates:
    """Fine-field pushforward; each exact phase numerator has divisor pi."""

    form_mean_bounds: I
    phase_mean_numerator: Q
    form_half_difference_bounds: I | None
    phase_half_difference_numerator: Q | None
    resultant_magnitude_rate_bounds: I | None
    internal_form_squared_rate_bounds: I | None
    form_phase_correlation_rate_bounds: I | None


@dataclass(frozen=True)
class SineMixedPairStateAssessment:
    """Sufficient realized pair state modulo the admitted independent swaps.

    The exact mixed observation identifies exactly the subgroup generated by
    the individually valid pair swaps, not every automorphism of the graph.
    It retains all continuous internal degrees of freedom. The theorem and
    reconstruction recipes apply to exact realized coordinates, not arbitrary
    tuples chosen within the displayed interval enclosures. Full captured
    member state and supplied lifts remain provenance, not reduced coordinates.
    """

    support_symmetry: SinePairSupportSymmetryAssessment
    phase_turns: tuple[int, ...]
    coordinates: tuple[SineMixedPairCoordinates, ...]
    rates: tuple[SineMixedPairRates, ...]
    phase_chart_margins: tuple[I, ...]
    exact_mixed_coordinates_identify_allowed_swap_orbits: bool = True
    mixed_coordinate_rates_well_defined: bool = True
    independent_interval_coordinates_admitted: bool = False
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "complete_conservative_normalized_sine_law_common_positive_held_capacity",
        "one_captured_fixed_simple_connected_unit_support_and_ordered_pair_partition",
        "signed_member_to_attachment_coordinates_retained_for_nonsymmetric_pairs",
        "unordered_R_U_Q_only_for_individually_equivariant_whole_member_swaps",
        "exact_realized_coordinates_identify_the_allowed_independent_swap_orbits",
        "not_a_quotient_by_all_graph_automorphisms_or_a_continuous_dimension_reduction",
        "supplied_phase_lifts_and_strict_pair_chart_are_not_inferred_from_proximity",
        "local_chart_validity_is_not_a_future_chart_or_grouping_certificate",
        "instantaneous_rates_are_the_shared_full_fine_field_pushforward",
        "reconstruction_is_piecewise_at_equal_phase_without_dividing_by_zero",
        "interval_projections_are_not_an_exact_inverse_or_independent_state_admission",
        "ordered_synchronized_tips_need_not_be_invariant_on_asymmetric_support",
        "strict_replica_law_admission_and_existing_certificates_remain_unchanged",
        "no_new_pressure_event_force_macro_runtime_or_physical_identification",
    )

    @property
    def comparison(self):
        return self.support_symmetry.comparison

    @property
    def pairs(self):
        return self.support_symmetry.pairs

    def to_dict(self):
        from ..sdk.relational_reports import _project

        _validate_replica_labels(self)
        return {
            "schema": "tnfr.relational-sine-mixed-pair-state.v1",
            "report": _project(self),
        }


def assess_sine_mixed_pair_state(
    graph, *, reference_model, pairs, phase_turns=None
) -> SineMixedPairStateAssessment:
    """Retain attachment information exactly where independent swaps fail.

    Symmetry admission and coordinate observation share one fine capture.
    ``phase_turns`` supplies integer turns in that capture's node order; every
    lifted pair gap must be certified shorter than pi. Ordered internal state
    is retained for each asymmetric pair, even when its current u=delta=0.
    Means and unordered invariants suffice for each truly symmetric pair.

    Reconstruction on an unordered pair uses delta=acos(R), u=Q/sqrt(1-R**2)
    when R<1; at R=1 use delta=0 and u=sqrt(U). These representatives differ
    only by an admitted simultaneous form/phase member swap. This exact recipe
    explains sufficiency; this reader does not invert interval evaluations.
    Rates use actual fine rows, never incomplete-block replica formulas.
    """
    comparison = bound_relational_sine_exchange(graph, reference_model=reference_model)
    symmetry = _assess_sine_pair_support_symmetry(comparison, pairs)
    chart = _pair_chart_coordinates(comparison, symmetry.pair_indices, phase_turns)
    numerators = comparison.phase_rate_numerators()
    pi = pi_interval()
    coordinates, rates = [], []
    for k, ((i, j), unordered) in enumerate(
        zip(symmetry.pair_indices, symmetry.pair_swap_symmetry)
    ):
        internal = chart["internal"][k]
        sine, cosine = chart["sine_delta"][k], chart["coherence"][k]
        radian, turn = chart["internal_radians"][k], chart["internal_turns"][k]
        equal_phase = radian == 0 and turn == 0
        if not unordered:
            stratum = "ordered"
        elif not equal_phase:
            stratum = "unordered_split_phase"
        elif internal:
            stratum = "unordered_equal_phase"
        else:
            stratum = "unordered_tip"
        coordinates.append(
            SineMixedPairCoordinates(
                mode="unordered" if unordered else "ordered",
                form_mean=chart["means"][k],
                phase_mean_radian_part=chart["mean_radians"][k],
                phase_mean_turn_part=chart["mean_turns"][k],
                form_half_difference=None if unordered else internal,
                phase_half_difference_radian_part=None if unordered else radian,
                phase_half_difference_turn_part=None if unordered else turn,
                resultant_magnitude_bounds=cosine if unordered else None,
                internal_form_squared=internal**2 if unordered else None,
                form_phase_correlation_bounds=internal * sine if unordered else None,
                reconstruction_stratum=stratum,
            )
        )
        internal_form = (comparison.form_rates[i] - comparison.form_rates[j]) / 2
        internal_phase = (numerators[i] - numerators[j]) / 2
        rates.append(
            SineMixedPairRates(
                form_mean_bounds=(comparison.form_rates[i] + comparison.form_rates[j])
                / 2,
                phase_mean_numerator=(numerators[i] + numerators[j]) / 2,
                form_half_difference_bounds=None if unordered else internal_form,
                phase_half_difference_numerator=None if unordered else internal_phase,
                resultant_magnitude_rate_bounds=(
                    -sine * (I(internal_phase) / pi) if unordered else None
                ),
                internal_form_squared_rate_bounds=(
                    2 * internal * internal_form if unordered else None
                ),
                form_phase_correlation_rate_bounds=(
                    internal_form * sine + internal * cosine * (I(internal_phase) / pi)
                    if unordered
                    else None
                ),
            )
        )
    return SineMixedPairStateAssessment(
        support_symmetry=symmetry,
        phase_turns=chart["turns"],
        coordinates=tuple(coordinates),
        rates=tuple(rates),
        phase_chart_margins=chart["chart_margins"],
    )


@dataclass(frozen=True)
class SinePairEmissionOutcome:
    """One supplied AL form projection; all pair phase coordinates are held.

    ``target_member_indices`` uses positions 0 and 1 in the selected pair,
    not graph node identifiers. Ordered endpoint forms are retained as fine
    provenance. The collective observation is X, U and Q, together with the
    unchanged Theta and R in the source state's selected coordinate row.
    """

    target_member_indices: tuple[int, ...]
    form_pair: tuple[Q, Q]
    effective_form_increments: tuple[Q, Q]
    form_mean: Q
    internal_form_squared: Q
    form_phase_correlation_bounds: I


@dataclass(frozen=True)
class SinePairEmissionAssessment:
    """Structural AL projection on one actual unordered-pair source orbit.

    The first and second member outcomes test whether a supplied single-port
    action has the same collective endpoint for the two fine representatives.
    Exact member multisets, not overlapping interval observations, decide this
    source-orbit question. Equality at a synchronized tip or at two actual
    no-ops is not all-state closure of a labeled single-member action.

    Applying the same admitted scalar AL map to both members commutes with
    their swap. This covers clipping but only the structural form projection;
    it does not certify grammar, configured public preconditions, lifecycle
    metadata, pressure refresh, event timing or subsequent flow admission.
    """

    source_state: SineMixedPairStateAssessment
    pair_index: int
    boost: Q
    clip_policy: tuple[Q, Q, str, Q]
    member_proposals: tuple[Q, Q]
    first_member: SinePairEmissionOutcome
    second_member: SinePairEmissionOutcome
    whole_pair: SinePairEmissionOutcome
    single_member_source_orbit_equal: bool
    single_member_source_orbit_status: str
    whole_pair_structural_descent_certified: bool = True
    runtime_admission_certified: bool = False
    event_occurrence_derived: bool = False
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "one_captured_mixed_pair_state_and_selected_independent_swap_symmetry",
        "explicit_positive_represented_AL_boost_and_shared_operator_boundary_policy",
        "all_constituent_structural_proposals_must_be_admitted_before_reporting",
        "binary64_AL_endpoints_retained_as_exact_represented_rationals",
        "source_phase_capacity_support_and_other_pairs_held_at_the_jump",
        "source_orbit_equality_uses_exact_form_and_supplied_phase_lift_members",
        "interval_overlap_never_certifies_equal_collective_endpoints",
        "single_member_source_exception_is_not_all_state_collective_descent",
        "whole_pair_same_scalar_map_commutes_with_the_admitted_member_swap",
        "ordered_endpoint_provenance_is_not_a_retained_macro_port_mark",
        "no_grammar_lifecycle_history_callback_or_full_runtime_admission_certificate",
        "no_derived_trigger_clock_support_change_or_event_occurrence",
        "no_post_event_loss_storage_passivity_or_future_flow_certificate",
        "no_graph_write_runtime_dispatch_or_physical_identification",
    )

    @property
    def comparison(self):
        return self.source_state.comparison

    @property
    def pairs(self):
        return self.source_state.pairs

    def form_increment(self, *, outcome):
        """Assess the actual full-form change of one supplied AL projection.

        ``outcome`` must be ``first_member``, ``second_member`` or
        ``whole_pair``. All nontarget nodes retain zero increment. This uses
        the same captured sine invariant owner as general endpoint checks;
        it neither executes AL nor promotes a zero change to reachability.
        """
        if not isinstance(outcome, str) or outcome not in (
            "first_member",
            "second_member",
            "whole_pair",
        ):
            raise ValueError(
                "outcome must be first_member, second_member or whole_pair"
            )
        selected = getattr(self, outcome)
        increments = [Q(0)] * len(self.comparison.nodes)
        indices = self.source_state.support_symmetry.pair_indices[self.pair_index]
        for node, delta in zip(indices, selected.effective_form_increments):
            increments[node] = delta
        return self.comparison.assess_form_increment(increments=tuple(increments))

    def to_dict(self):
        from ..sdk.relational_reports import _project

        _validate_replica_labels(self)
        return {
            "schema": "tnfr.relational-sine-pair-emission.v1",
            "report": _project(self),
        }


def assess_sine_pair_emission(
    graph, *, reference_model, pairs, pair_index, boost, phase_turns=None
) -> SinePairEmissionAssessment:
    """Compare actual AL form projections on one supplied unordered pair.

    ``boost`` is explicit and overrides no live graph configuration: it is an
    input to these hypothetical projections only. Its admission is the shared
    registered AL factor contract. The existing scalar AL proposal owner
    supplies rounding, clipping and the nondecrease postcondition. If either
    member proposal rejects, the whole observation rejects without writes.

    The selected pair must admit an independent whole-member swap in the
    existing mixed-state theorem. Other supplied pairs may retain ordered
    attachment information. The source carries all phase lifts and fine
    coordinates; no hidden port is inferred from a graph label or phase order.
    """
    from ..operators import _operator_epi_clip_policy
    from ..operators.al_sha_stage_proposals import emission_epi_proposal
    from ..operators.factor_contracts import validate_glyph_factor

    admitted_boost = validate_glyph_factor("AL_boost", boost)
    attributes = dict(graph.graph)
    clip = _operator_epi_clip_policy(attributes)
    source = assess_sine_mixed_pair_state(
        graph,
        reference_model=reference_model,
        pairs=pairs,
        phase_turns=phase_turns,
    )
    if type(pair_index) is not int or not 0 <= pair_index < len(source.pairs):
        raise ValueError("pair_index must be a nonboolean pair index")
    if not source.support_symmetry.pair_swap_symmetry[pair_index]:
        raise ValueError("selected pair must admit an independent member swap")

    i, j = source.support_symmetry.pair_indices[pair_index]
    comparison = source.comparison
    before = (comparison.epi[i], comparison.epi[j])
    proposed = tuple(
        Q(emission_epi_proposal(attributes, value, admitted_boost)[1])
        for value in before
    )
    radians = (comparison.phase[i] - comparison.phase[j]) / 2
    turn_difference = source.phase_turns[i] - source.phase_turns[j]
    sine_delta, _ = _pi_shifted_trig(radians, turn_difference)

    def outcome(targets):
        after = tuple(proposed[k] if k in targets else before[k] for k in (0, 1))
        internal = (after[0] - after[1]) / 2
        return SinePairEmissionOutcome(
            target_member_indices=targets,
            form_pair=after,
            effective_form_increments=tuple(after[k] - before[k] for k in (0, 1)),
            form_mean=(after[0] + after[1]) / 2,
            internal_form_squared=internal**2,
            form_phase_correlation_bounds=internal * sine_delta,
        )

    first, second, both = outcome((0,)), outcome((1,)), outcome((0, 1))

    def members(candidate):
        return sorted(
            (candidate.form_pair[k], comparison.phase[node], source.phase_turns[node])
            for k, node in enumerate((i, j))
        )

    equal = members(first) == members(second)
    tip = before[0] == before[1] and radians == 0 and turn_difference == 0
    status = (
        "equal_at_source_synchronized_tip"
        if tip
        else "equal_at_source_both_noops" if equal else "obstructed_at_source"
    )
    return SinePairEmissionAssessment(
        source_state=source,
        pair_index=pair_index,
        boost=Q(admitted_boost),
        clip_policy=(Q(clip[0]), Q(clip[1]), clip[2], Q(clip[3])),
        member_proposals=proposed,
        first_member=first,
        second_member=second,
        whole_pair=both,
        single_member_source_orbit_equal=equal,
        single_member_source_orbit_status=status,
    )


@dataclass(frozen=True)
class SineStatePairingAssessment:
    """Independent phase observation, then support/law admission on one capture."""

    comparison: SineExchangeComparison
    observation: PhasePairObservation
    pair_order: tuple[tuple[Any, Any], ...] | None
    pair_order_source: str
    phase_turns: tuple[int, ...]
    capacity: SineReplicaCapacityAssessment | None
    capacity_admission_status: str
    capacity_admission_reasons: tuple[str, ...]
    all_time_pairing_persistence_certified: bool
    scope: tuple[str, ...] = (
        "observation_consumes_only_the_single_capture_phase_projection",
        "candidate_is_never_replaced_by_topological_twins_or_supplied_pairs",
        "paired_support_capacity_and_conservative_law_admission_are_independent",
        "caller_supplies_cycle_order_and_lifts_before_target_family_evidence",
        "no_automatic_unwrap_from_circular_proximity_or_support",
        "strict_full_state_trapping_can_preserve_observed_partners_for_all_times",
        "instantaneous_matching_alone_does_not_prove_future_identity_or_closure",
        "no_selected_state_recurrence_internal_activity_or_new_links_are_inferred",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project, _validate_label

        _validate_comparison_labels(self.comparison)
        for node in self.observation.nodes:
            _validate_label(node)
        for pair in self.observation.candidate_pairs or ():
            for node in pair:
                _validate_label(node)
        for pair in self.pair_order or ():
            for node in pair:
                _validate_label(node)
        if self.capacity is not None:
            _validate_replica_labels(self.capacity)
        return {
            "schema": "tnfr.relational-sine-state-pairing.v1",
            "report": _project(self),
        }


def assess_sine_state_pairing(
    graph,
    *,
    reference_model,
    pair_order=None,
    phase_turns=None,
    radius=None,
    excess_ceiling=None,
    form_mean_bounds=None,
    winding=1,
) -> SineStatePairingAssessment:
    """Observe phase partners and separately admit their existing collective law.

    A supplied pair_order must equal the observed unordered partition. It is
    mandatory when requesting target-dependent family bounds; no cycle order
    is inferred from support or phase values. Valid circular matches can fail
    the supplied real-lift chart. Only expected law/support/chart refusals are
    retained as rejection; malformed caller input and arithmetic errors raise.
    """
    family_arguments = _optional_replica_family(
        radius, excess_ceiling, form_mean_bounds, winding
    )
    if family_arguments is not None:
        if pair_order is None:
            raise ValueError(
                "an explicit pair_order is required for target family assessment"
            )
        radius = exact_or_represented_real(radius, "radius")
        if radius <= 0:
            raise ValueError("radius must be strictly positive")
    comparison = bound_relational_sine_exchange(graph, reference_model=reference_model)
    turns = _replica_phase_turns(phase_turns, len(comparison.nodes))
    provided = None
    if pair_order is not None:
        raw = _ordered(pair_order, "pair_order", limit=len(comparison.nodes) + 1)
        provided = tuple(
            _ordered(pair, "each pair_order entry", limit=3) for pair in raw
        )
        flat = tuple(node for pair in provided for node in pair)
        if (
            any(len(pair) != 2 for pair in provided)
            or len(flat) != len(comparison.nodes)
            or len(set(flat)) != len(flat)
            or set(flat) != set(comparison.nodes)
        ):
            raise ValueError("pair_order must partition all captured nodes into pairs")
    observation = observe_phase_pairs(nodes=comparison.nodes, phases=comparison.phase)
    candidate = observation.candidate_pairs
    if (
        candidate is not None
        and provided is not None
        and {frozenset(pair) for pair in provided}
        != {frozenset(pair) for pair in candidate}
    ):
        raise ValueError("pair_order must match the phase-observed unordered partition")
    selected = provided if provided is not None else candidate
    order_source = (
        "explicit"
        if provided is not None
        else (
            "capture_order_display_only"
            if candidate is not None
            else "unavailable_no_partition"
        )
    )
    capacity = None
    if candidate is None:
        status, reasons = "not_attempted", ("phase_pairing_not_certified",)
    else:
        try:
            capacity = _assess_sine_replica_capacity_comparison(
                comparison,
                pairs=selected,
                phase_turns=turns,
                radius=radius,
                family_arguments=family_arguments,
                winding=winding,
            )
        except _ReplicaAdmissionError as exc:
            status, reasons = "rejected", (str(exc),)
        else:
            status, reasons = "admitted", ()
    persistent = (
        capacity is not None
        and capacity.family is not None
        and capacity.family.source_set_trapping_certified
    )
    return SineStatePairingAssessment(
        comparison=comparison,
        observation=observation,
        pair_order=selected,
        pair_order_source=order_source,
        phase_turns=turns,
        capacity=capacity,
        capacity_admission_status=status,
        capacity_admission_reasons=reasons,
        all_time_pairing_persistence_certified=persistent,
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
        result = []
        for eigenvalue, mode_sine in zip(laplacian, sines):
            cross = -coupling * internal_cosine * internal_sine * mode_sine
            collective_phase = -stiffness * internal_cosine**2 * eigenvalue / 2
            internal_phase = -stiffness * (
                double_internal_cosine + internal_sine**2 * eigenvalue / 2
            )
            result.append(
                (
                    (I(0), collective_phase, I(0), cross),
                    (mobility * eigenvalue / 2, I(0), I(0), I(0)),
                    (I(0), cross, I(0), internal_phase),
                    (I(0), I(0), mobility, I(0)),
                )
            )
        return tuple(result)

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
