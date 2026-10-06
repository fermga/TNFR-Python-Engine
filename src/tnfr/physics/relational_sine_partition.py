"""Conditional phase-offset partitions of the complete conservative sine law.

The source supplies a fixed support and a declared law. Its captured state need
not belong to the proposed family ``x_i=X_a, theta_i=Theta_a+psi_i``. Exact
normalized neighbor counts and complex phase moments test invariance for every
collective state, not a single prepared sample. Rational-turn sine expressions
prove sufficient identities; an undecided identity remains unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from math import isfinite

from .._exact_time import exact_or_represented_real, exp_upper_float
from ..mathematics._rational_interval import (
    INTERVAL_METHOD,
    I,
    cos,
    pi_interval,
    sin,
    sqrt,
)
from ._sine_admission import _admit_sine_source, _sine_model_coefficients
from .epi_memory import _exact_partition
from .phase_cycle_geometry import C5_PHASE_SECTOR_BARRIER, _sine_turn_term
from .relational_observations import _ordered
from .relational_sine_comparison import (
    SineExchangeComparison,
    _validate_comparison_labels,
)

__all__ = (
    "SinePhaseOffsetPartition",
    "SinePhaseOffsetState",
    "SineMovingPatternWindow",
    "SineCollectivePulseBalance",
    "SineContactAveraging",
    "assess_sine_phase_offset_partition",
    "assess_sine_moving_pattern_window",
    "observe_sine_collective_pulse",
    "assess_sine_contact_averaging",
)

Expression = tuple[tuple[Q, Q], ...]


def _rational_turns(values, label, size):
    values = _ordered(values, label, limit=size + 1)
    if len(values) != size:
        raise ValueError(f"{label} must contain {size} values in declared order")
    if any(type(value) not in (int, Q) for value in values):
        raise TypeError(f"{label} accepts only integers or Fractions in turns")
    return tuple(Q(value) for value in values)


def _sine_expression(terms):
    """Collect admitted rational-turn terms using the shared exact identities."""
    result = {}
    for value, coefficient in terms:
        term = _sine_turn_term(value, fold_circle=True)
        if term is not None:
            turn, sign = term
            result[turn] = result.get(turn, Q(0)) + coefficient * sign
    return tuple((turn, value) for turn, value in sorted(result.items()) if value)


def _difference(left, right):
    return _sine_expression((*left, *((turn, -value) for turn, value in right)))


def _expression_bound(expression, cache):
    result = I(0)
    for turn, coefficient in expression:
        if turn not in cache:
            cache[turn] = I(1) if turn == Q(1, 4) else sin(2 * pi_interval() * turn)
        result += coefficient * cache[turn]
    return result


def _expression_status(expression, cache):
    if not expression:
        return "certified"
    bound = _expression_bound(expression, cache)
    return "excluded" if bound.lo > 0 or bound.hi < 0 else "unresolved"


def _moment_status(cosine_difference, sine_difference, cache):
    statuses = tuple(
        _expression_status(value, cache)
        for value in (cosine_difference, sine_difference)
    )
    if "excluded" in statuses:
        return "excluded"
    return "certified" if statuses == ("certified", "certified") else "unresolved"


def _validate_partition_labels(report):
    from ..sdk.relational_reports import _validate_label

    _validate_comparison_labels(report.source)
    for block in report.blocks:
        for node in block:
            _validate_label(node)


@dataclass(frozen=True)
class SinePhaseOffsetPartition:
    """A sufficient exact invariant-family certificate or explicit abstention.

    ``normalized_counts`` and moment rows are indexed by full source node, then
    target block. Internal real moments are retained but never compared.
    Quotient moment diagonals are zero: certified internal sine torque vanishes.
    Nonempty symbolic expressions are not, by themselves, obstructions.
    """

    source: SineExchangeComparison
    blocks: tuple[tuple[object, ...], ...]
    block_indices: tuple[tuple[int, ...], ...]
    node_blocks: tuple[int, ...]
    phase_offset_turns: tuple[Q, ...]
    normalized_counts: tuple[tuple[Q, ...], ...]
    row_cosine_expressions: tuple[tuple[Expression, ...], ...]
    row_sine_expressions: tuple[tuple[Expression, ...], ...]
    row_cosine_bounds: tuple[tuple[I, ...], ...]
    row_sine_bounds: tuple[tuple[I, ...], ...]
    cross_count_residuals: tuple[tuple[Q, ...], ...]
    cross_moment_statuses: tuple[tuple[str, ...], ...]
    internal_current_expressions: tuple[Expression, ...]
    internal_current_bounds: tuple[I, ...]
    internal_current_statuses: tuple[str, ...]
    count_compatibility_certified: bool
    moment_compatibility_certified: bool
    internal_balance_certified: bool
    invariance_certified: bool
    status: str
    reasons: tuple[str, ...]
    quotient_counts: tuple[tuple[Q, ...], ...] | None
    quotient_cosine_expressions: tuple[tuple[Expression, ...], ...] | None
    quotient_sine_expressions: tuple[tuple[Expression, ...], ...] | None
    clock: str = "tau=t/pi"
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "source_anchors_complete_fixed_support_and_conservative_unit_sine_law",
        "captured_source_state_is_not_asserted_to_belong_to_the_declared_family",
        "all_collective_states_share_forms_and_fixed_phase_offsets_within_blocks",
        "cross_counts_and_complex_moments_and_zero_internal_current_are_required",
        "internal_real_moment_equality_is_not_required",
        "symbolic_cancellation_proves_identity_interval_overlap_does_not",
        "unresolved_exact_identities_do_not_disprove_invariance",
        "exact_invariance_is_not_attraction_formation_or_finite_time_entry",
        "no_capacity_support_event_forcing_or_new_law_is_installed",
    )

    def evaluate(self, collective_form, collective_phase_turns):
        """Evaluate an admitted family state after rebuilding this certificate."""
        return _evaluate_phase_offset_state(
            self, collective_form, collective_phase_turns
        )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        _validate_partition_labels(self)
        return {
            "schema": "tnfr.sine-phase-offset-partition.v1",
            "report": _project(self),
        }


@dataclass(frozen=True)
class SinePhaseOffsetState:
    """Detached full and quotient rates at a declared invariant-family state.

    The clock is ``tau=t/pi`` for every rate. Internal stores remain constant
    along the admitted family, while cross storage and collective coordinates
    can evolve. Rate residual intervals are arithmetic checks; equality is
    proved by the rebuilt all-state partition criterion, never zero overlap.
    """

    partition: SinePhaseOffsetPartition
    collective_form: tuple[Q, ...]
    collective_phase_turns: tuple[Q, ...]
    fine_form: tuple[Q, ...]
    fine_phase_turns: tuple[Q, ...]
    block_form_rates: tuple[I, ...]
    block_phase_rates: tuple[Q, ...]
    full_form_rates: tuple[I, ...]
    full_phase_rates: tuple[Q, ...]
    form_rate_residual_bounds: tuple[I, ...]
    phase_rate_residuals: tuple[Q, ...]
    full_row_equality_certified: bool
    internal_form_storage: tuple[Q, ...]
    internal_phase_storage_bounds: tuple[I, ...]
    cross_form_storage: Q
    cross_phase_storage_bounds: I
    full_form_storage: Q
    full_phase_storage_bounds: I
    full_storage_bounds: I
    clock: str = "tau=t/pi"
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "original_source_blocks_offsets_and_consumed_law_are_readmitted",
        "cached_partition_verdicts_rows_and_quotient_coefficients_are_not_evidence",
        "full_rates_use_actual_fine_edges_and_declared_collective_coordinates",
        "phase_rates_are_radian_phase_derivatives_with_respect_to_tau_not_turn_rates",
        "all_state_moment_proof_establishes_full_to_quotient_row_equality",
        "internal_form_storage_is_zero_and_internal_phase_storage_is_inherited",
        "moving_collective_state_and_live_cross_storage_do_not_imply_formation",
        "this_is_an_instantaneous_evaluation_not_a_solver_step_or_forecast",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        _validate_partition_labels(self.partition)
        return {"schema": "tnfr.sine-phase-offset-state.v1", "report": _project(self)}


def assess_sine_phase_offset_partition(source, *, blocks, phase_offset_turns):
    """Certify a phase-offset family for all collective coordinates, if proved.

    The complete law must have zero loss and unit exchange, beta and held
    capacities. The ordered partition has between two and ``n-1`` nonempty
    blocks and contains every source node exactly once. Offsets are declared
    exact rational turns in full source order;
    rounded radians or floating turns do not stand for exact geometric angles.
    Every normalized row is rebuilt from admitted primitive support. The
    symbolic test is sufficient but deliberately incomplete.
    """
    if not isinstance(source, SineExchangeComparison):
        raise TypeError("an exact SineExchangeComparison is required")
    admitted, edges = _admit_sine_source(source)
    if _sine_model_coefficients(admitted.reference_model) != (0, 1, 1) or any(
        value != 1 for value in admitted.capacity
    ):
        raise ValueError(
            "partition requires zero loss and unit capacity, exchange and beta"
        )
    nodes = admitted.nodes
    blocks = _exact_partition(nodes, blocks)
    offsets = _rational_turns(phase_offset_turns, "phase_offset_turns", len(nodes))
    positions = {node: i for i, node in enumerate(nodes)}
    block_indices = tuple(tuple(positions[node] for node in block) for block in blocks)
    node_blocks = [0] * len(nodes)
    for a, indices in enumerate(block_indices):
        for i in indices:
            node_blocks[i] = a
    node_blocks = tuple(node_blocks)
    neighbors = [[] for _ in nodes]
    for i, j in edges:
        neighbors[i].append(j)
        neighbors[j].append(i)
    counts, cosine_rows, sine_rows = [], [], []
    for i, row in enumerate(neighbors):
        by_block = tuple(
            tuple(j for j in row if node_blocks[j] == b) for b in range(len(blocks))
        )
        scale = Q(1, admitted.degrees[i])
        counts.append(tuple(len(group) * scale for group in by_block))
        cosine_rows.append(
            tuple(
                _sine_expression(
                    (offsets[j] - offsets[i] + Q(1, 4), scale) for j in group
                )
                for group in by_block
            )
        )
        sine_rows.append(
            tuple(
                _sine_expression((offsets[j] - offsets[i], scale) for j in group)
                for group in by_block
            )
        )
    counts, cosine_rows, sine_rows = tuple(counts), tuple(cosine_rows), tuple(sine_rows)
    cache = {}
    cosine_bounds = tuple(
        tuple(_expression_bound(expr, cache) for expr in row) for row in cosine_rows
    )
    sine_bounds = tuple(
        tuple(_expression_bound(expr, cache) for expr in row) for row in sine_rows
    )
    count_residuals, moment_statuses = [], []
    for i, a in enumerate(node_blocks):
        representative = block_indices[a][0]
        count_residuals.append(
            tuple(
                Q(0) if a == b else counts[i][b] - counts[representative][b]
                for b in range(len(blocks))
            )
        )
        moment_statuses.append(
            tuple(
                (
                    "not_required"
                    if a == b
                    else _moment_status(
                        _difference(cosine_rows[i][b], cosine_rows[representative][b]),
                        _difference(sine_rows[i][b], sine_rows[representative][b]),
                        cache,
                    )
                )
                for b in range(len(blocks))
            )
        )
    count_residuals, moment_statuses = tuple(count_residuals), tuple(moment_statuses)
    internal = tuple(sine_rows[i][a] for i, a in enumerate(node_blocks))
    internal_bounds = tuple(sine_bounds[i][a] for i, a in enumerate(node_blocks))
    internal_statuses = tuple(_expression_status(expr, cache) for expr in internal)
    count_ok = not any(value for row in count_residuals for value in row)
    moment_ok = all(
        value in ("certified", "not_required")
        for row in moment_statuses
        for value in row
    )
    internal_ok = all(value == "certified" for value in internal_statuses)
    certified = count_ok and moment_ok and internal_ok
    excluded = (
        not count_ok
        or "excluded" in internal_statuses
        or any("excluded" in row for row in moment_statuses)
    )
    status = "certified" if certified else "excluded" if excluded else "unavailable"
    reasons = []
    if not count_ok:
        reasons.append("normalized_cross_counts_differ_within_a_block")
    if not moment_ok:
        reasons.append("cross_complex_moment_equality_excluded_or_unresolved")
    if not internal_ok:
        reasons.append("zero_internal_sine_current_excluded_or_unresolved")
    quotient_counts = (
        tuple(counts[row[0]] for row in block_indices) if certified else None
    )
    quotient_cosine = (
        tuple(
            tuple(() if a == b else cosine_rows[row[0]][b] for b in range(len(blocks)))
            for a, row in enumerate(block_indices)
        )
        if certified
        else None
    )
    quotient_sine = (
        tuple(
            tuple(() if a == b else sine_rows[row[0]][b] for b in range(len(blocks)))
            for a, row in enumerate(block_indices)
        )
        if certified
        else None
    )
    return SinePhaseOffsetPartition(
        source=source,
        blocks=blocks,
        block_indices=block_indices,
        node_blocks=node_blocks,
        phase_offset_turns=offsets,
        normalized_counts=counts,
        row_cosine_expressions=cosine_rows,
        row_sine_expressions=sine_rows,
        row_cosine_bounds=cosine_bounds,
        row_sine_bounds=sine_bounds,
        cross_count_residuals=count_residuals,
        cross_moment_statuses=moment_statuses,
        internal_current_expressions=internal,
        internal_current_bounds=internal_bounds,
        internal_current_statuses=internal_statuses,
        count_compatibility_certified=count_ok,
        moment_compatibility_certified=moment_ok,
        internal_balance_certified=internal_ok,
        invariance_certified=certified,
        status=status,
        reasons=tuple(reasons),
        quotient_counts=quotient_counts,
        quotient_cosine_expressions=quotient_cosine,
        quotient_sine_expressions=quotient_sine,
    )


def _evaluate_phase_offset_state(report, collective_form, collective_phase_turns):
    reference = assess_sine_phase_offset_partition(
        report.source,
        blocks=report.blocks,
        phase_offset_turns=report.phase_offset_turns,
    )
    if not reference.invariance_certified:
        raise ValueError("evaluation requires a certified phase-offset partition")
    source, edges = _admit_sine_source(reference.source)
    size = len(reference.blocks)
    raw_form = _ordered(collective_form, "collective_form", limit=size + 1)
    if len(raw_form) != size:
        raise ValueError("collective_form must contain one value per block")
    form = tuple(
        exact_or_represented_real(value, f"collective_form[{i}]")
        for i, value in enumerate(raw_form)
    )
    turns = _rational_turns(collective_phase_turns, "collective_phase_turns", size)
    fine_form = tuple(form[a] for a in reference.node_blocks)
    fine_turns = tuple(
        value + turns[a]
        for value, a in zip(reference.phase_offset_turns, reference.node_blocks)
    )
    q = [Q(0)] * len(source.nodes)
    sine_terms = [[] for _ in source.nodes]
    internal_form = [Q(0)] * size
    internal_phase = [I(0) for _ in reference.blocks]
    cross_form, cross_phase = Q(0), I(0)
    cache = {}
    for i, j in edges:
        difference = fine_form[i] - fine_form[j]
        q[i] += difference
        q[j] -= difference
        sine_terms[i].append((fine_turns[j] - fine_turns[i], Q(1, source.degrees[i])))
        sine_terms[j].append((fine_turns[i] - fine_turns[j], Q(1, source.degrees[j])))
        form_storage = difference * difference / 2
        phase_storage = I(1) - _expression_bound(
            _sine_expression(((fine_turns[j] - fine_turns[i] + Q(1, 4), Q(1)),)), cache
        )
        if reference.node_blocks[i] == reference.node_blocks[j]:
            a = reference.node_blocks[i]
            internal_form[a] += form_storage
            internal_phase[a] += phase_storage
        else:
            cross_form += form_storage
            cross_phase += phase_storage
    full_phase = tuple(value / degree for value, degree in zip(q, source.degrees))
    full_form = tuple(
        _expression_bound(_sine_expression(terms), cache) for terms in sine_terms
    )
    # Representatives evaluate the quotient form rows after the all-state
    # coefficient identities prove every fine row in that block agrees.
    block_form = tuple(full_form[indices[0]] for indices in reference.block_indices)
    block_phase = tuple(
        sum(
            (coefficient * (form[a] - form[b]) for b, coefficient in enumerate(row)),
            Q(0),
        )
        for a, row in enumerate(reference.quotient_counts)
    )
    phase_residuals = tuple(
        value - block_phase[a] for value, a in zip(full_phase, reference.node_blocks)
    )
    if any(phase_residuals):
        raise RuntimeError("certified partition lost exact phase-row equality")
    form_residuals = tuple(
        value - block_form[a] for value, a in zip(full_form, reference.node_blocks)
    )
    if any(not value.contains(0) for value in form_residuals):
        raise RuntimeError("certified partition lost form-row enclosure consistency")
    full_form_storage = sum(internal_form, cross_form)
    full_phase_storage = sum(internal_phase, cross_phase)
    return SinePhaseOffsetState(
        partition=reference,
        collective_form=form,
        collective_phase_turns=turns,
        fine_form=fine_form,
        fine_phase_turns=fine_turns,
        block_form_rates=block_form,
        block_phase_rates=block_phase,
        full_form_rates=full_form,
        full_phase_rates=full_phase,
        form_rate_residual_bounds=form_residuals,
        phase_rate_residuals=phase_residuals,
        full_row_equality_certified=True,
        internal_form_storage=tuple(internal_form),
        internal_phase_storage_bounds=tuple(internal_phase),
        cross_form_storage=cross_form,
        cross_phase_storage_bounds=cross_phase,
        full_form_storage=full_form_storage,
        full_phase_storage_bounds=full_phase_storage,
        full_storage_bounds=full_form_storage + full_phase_storage,
    )


@dataclass(frozen=True)
class SineMovingPatternWindow:
    """Finite Bregman-storage control around a moving C5/private-leaf family.

    Error radii refer to every fine form and continuous phase lift, in source
    order. The propagated energy controls full-support edge contrasts, not
    absolute coordinates. Conserved weighted mean errors remain independently
    bounded by their input budgets. All times and rates use ``tau=t/pi``.

    The arithmetic candidate ``propagated_relative_storage_upper_bound`` is
    a whole-window guarantee only when ``whole_window_retention_certified``.
    Failed sufficient conditions are unavailable, not dynamical exclusions.
    """

    reference: SinePhaseOffsetState
    cycle_indices: tuple[int, ...]
    contact_indices: tuple[tuple[int, int], ...]
    orientation: int
    form_error_bounds: tuple[Q, ...]
    phase_error_bounds: tuple[Q, ...]
    scaled_horizon: Q
    original_horizon_bounds: I
    receiver_phase_radius: Q
    contact_phase_radius: Q
    reference_contact_bound: Q
    reference_form_gap: Q
    reference_contact_turn: Q
    reference_contact_phase_bounds: I
    reference_libration_energy_bounds: I
    reference_energy_ceiling_bounds: I
    reference_energy_guard_margin_lower_bound: Q
    reference_contact_initial_margin_lower_bound: Q
    receiver_chart_margin_lower_bound: Q
    contact_chart_margin_lower_bound: Q
    initial_receiver_phase_error_bounds: tuple[Q, ...]
    initial_contact_phase_error_bounds: tuple[Q, ...]
    receiver_curvature_lower_bound: Q
    contact_curvature_lower_bound: Q
    initial_relative_storage_upper_bound: Q
    relative_storage_barrier: Q | None
    growth_rate_upper_bound: Q | None
    growth_factor_upper_bound: Q | None
    propagated_relative_storage_upper_bound: Q | None
    retention_margin_lower_bound: Q | None
    initial_chart_certified: bool
    reference_contact_envelope_certified: bool
    whole_window_retention_certified: bool
    edge_form_error_upper_bound: Q | None
    receiver_phase_error_upper_bound: Q | None
    contact_phase_error_upper_bound: Q | None
    receiver_storage_excess_upper_bound: Q | None
    form_mean_error_upper_bound: Q
    phase_mean_error_upper_bound: Q
    initial_full_storage_bounds: I
    phase_flat_acquisition_status: str
    status: str
    reasons: tuple[str, ...]
    clock: str = "tau=t/pi"
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "rebuilt_exact_phase_offset_reference_same_complete_conservative_unit_law",
        "ordered_induced_C5_and_five_private_leaves_all_states_remain_live",
        "independent_errors_cover_all_ten_fine_forms_and_continuous_phase_lifts",
        "relative_Bregman_storage_uses_full_form_and_phase_edge_differences",
        "central_contact_libration_and_initial_error_chart_are_separate_premises",
        "strict_first_exit_budget_certifies_relative_storage_on_the_whole_window",
        "same_unchanged_law_certifies_both_forward_and_backward_scaled_horizon",
        "edge_bounds_do_not_bound_absolute_coordinates_or_remove_mean_errors",
        "receiver_internal_storage_excess_is_bounded_by_full_relative_storage",
        "initial_total_storage_box_is_an_enclosure_not_an_independent_reservoir",
        "admitted_receiver_chart_retains_the_correlated_C5_phase_minimum_in_the_energy_lower_bound",
        "sub_barrier_storage_excludes_zero_winding_origins_via_the_wider_phase_sector",
        "phase_flat_acquisition_status_retains_its_compatible_restricted_conclusion",
        "not_excluded_acquisition_is_only_absence_of_this_energy_obstruction",
        "unavailable_retention_does_not_disprove_the_actual_dynamics",
        "no_trajectory_formation_attraction_new_law_or_reference_state_replacement",
    )
    zero_winding_acquisition_status: str = "not_excluded"

    def to_dict(self):
        from ..sdk.relational_reports import _project

        _validate_partition_labels(self.reference.partition)
        return {
            "schema": "tnfr.sine-moving-pattern-window.v1",
            "report": _project(self),
        }


def _private_leaf_support(source, cycle):
    """Admit one complete conservative unit C5/private-leaf source and cycle."""
    if not isinstance(source, SineExchangeComparison):
        raise TypeError("an exact SineExchangeComparison is required")
    source, edges = _admit_sine_source(source)
    if _sine_model_coefficients(source.reference_model) != (0, 1, 1) or any(
        value != 1 for value in source.capacity
    ):
        raise ValueError(
            "private-leaf analysis requires zero loss and unit capacity, exchange and beta"
        )
    cycle = _ordered(cycle, "cycle", limit=6)
    positions = {node: i for i, node in enumerate(source.nodes)}
    try:
        cycle = tuple(positions[node] for node in cycle)
    except (KeyError, TypeError) as exc:
        raise ValueError("cycle must contain known source nodes") from exc
    if len(source.nodes) != 10 or len(cycle) != 5 or len(set(cycle)) != 5:
        raise ValueError("source must contain an ordered C5 and five private leaves")
    leaves = set(range(10)) - set(cycle)
    pairs = tuple(zip(cycle, cycle[1:] + cycle[:1]))
    internal = {tuple(sorted(edge)) for edge in pairs}
    neighbors = [set() for _ in source.nodes]
    for i, j in edges:
        neighbors[i].add(j)
        neighbors[j].add(i)
    if any(len(neighbors[i]) != 3 for i in cycle) or any(
        len(neighbors[i]) != 1 for i in leaves
    ):
        raise ValueError("receiver degrees must be three and private-leaf degrees one")
    contacts = tuple((i, next(iter(neighbors[i] & leaves), -1)) for i in cycle)
    if (
        len({j for _, j in contacts}) != 5
        or -1 in {j for _, j in contacts}
        or set(edges) != internal | {tuple(sorted(edge)) for edge in contacts}
    ):
        raise ValueError(
            "full support must contain exactly the ordered C5 and its private contacts"
        )
    return source, edges, cycle, pairs, contacts


def _moving_pattern_reference(reference):
    """Rebuild the full family and admit exactly the C5/private-leaf geometry."""
    if not isinstance(reference, SinePhaseOffsetState) or not isinstance(
        reference.partition, SinePhaseOffsetPartition
    ):
        raise TypeError("a SinePhaseOffsetState with its partition is required")
    reference = reference.partition.evaluate(
        reference.collective_form, reference.collective_phase_turns
    )
    partition = reference.partition
    if tuple(map(len, partition.blocks)) != (5, 5):
        raise ValueError(
            "moving pattern requires an ordered C5 and five private leaves"
        )
    source, edges, _, pairs, contacts = _private_leaf_support(
        partition.source, partition.blocks[0]
    )

    def principal(turn):
        return (turn + Q(1, 2)) % 1 - Q(1, 2)

    turns = reference.fine_phase_turns
    gaps = tuple(principal(turns[j] - turns[i]) for i, j in pairs)
    if gaps not in ((Q(1, 5),) * 5, (-Q(1, 5),) * 5):
        raise ValueError(
            "receiver phase offsets must have one uniform signed fifth turn"
        )
    contact_turns = tuple(principal(turns[j] - turns[i]) for i, j in contacts)
    if len(set(contact_turns)) != 1:
        raise ValueError("all private contacts must have the same principal phase lift")
    return (
        reference,
        source,
        edges,
        pairs,
        contacts,
        (1 if gaps[0] > 0 else -1),
        contact_turns[0],
    )


def assess_sine_moving_pattern_window(
    reference,
    *,
    form_error_bounds,
    phase_error_bounds,
    scaled_horizon,
    receiver_phase_radius,
    contact_phase_radius,
    reference_contact_bound,
) -> SineMovingPatternWindow:
    """Bound a full uncertainty box near a live conservative moving pattern.

    The first declared block is the ordered receiver C5; the second contains
    its five private leaves. The reference has uniform signed fifth-turn
    receiver gaps and a common matching-contact phase. A strict central
    libration energy guard and an independently admitted initial phase-error
    chart support the relative-storage first-exit proof. Every radius and
    ``scaled_horizon`` is positive, while per-node errors may be zero.

    This theorem controls a supplied neighborhood for both time directions;
    it does not certify first acquisition. The exact phase-flat source energy
    obstruction is separate from the finite sufficient retention inequality.
    """
    from .relational_sine_pattern import _error_radii

    reference, source, edges, cycle_edges, contacts, orientation, contact_turn = (
        _moving_pattern_reference(reference)
    )
    end, rho_r, rho_c, gamma = tuple(
        exact_or_represented_real(value, label)
        for value, label in (
            (scaled_horizon, "scaled_horizon"),
            (receiver_phase_radius, "receiver_phase_radius"),
            (contact_phase_radius, "contact_phase_radius"),
            (reference_contact_bound, "reference_contact_bound"),
        )
    )
    if min(end, rho_r, rho_c, gamma) <= 0:
        raise ValueError(
            "horizon, phase radii and reference contact bound must be positive"
        )
    fx = _error_radii(form_error_bounds, 10, "form_error_bounds")
    pt = _error_radii(phase_error_bounds, 10, "phase_error_bounds")
    pi = pi_interval()
    alpha, phi = 2 * pi / 5, 2 * contact_turn * pi
    u = reference.collective_form[0] - reference.collective_form[1]
    contact_cosine = _expression_bound(
        _sine_expression(((contact_turn + Q(1, 4), Q(1)),)), {}
    )
    raw_h = u**2 / 2 + 1 - contact_cosine
    h = I(max(Q(0), raw_h.lo), max(Q(0), raw_h.hi))
    ceiling = 1 - cos(I(gamma))
    energy_margin, contact_margin = ceiling.lo - h.hi, gamma - phi.abs_max
    receiver_margin = pi.lo / 2 - alpha.hi - rho_r
    chart_contact_margin = pi.lo / 2 - gamma - rho_c
    k_r, k_c = cos(alpha + rho_r).lo, cos(I(gamma + rho_c)).lo
    receiver_errors = tuple(pt[i] + pt[j] for i, j in cycle_edges)
    contact_errors = tuple(pt[i] + pt[j] for i, j in contacts)
    receiver_chart = receiver_margin > 0 and all(
        value < rho_r for value in receiver_errors
    )
    initial_chart = (
        receiver_chart
        and chart_contact_margin > 0
        and all(value < rho_c for value in contact_errors)
    )
    envelope = chart_contact_margin > 0 and contact_margin >= 0 and energy_margin > 0
    d0 = sum(((fx[i] + fx[j]) ** 2 + (pt[i] + pt[j]) ** 2 for i, j in edges), Q(0)) / 2
    barrier = rate = amplification = propagated = margin = None
    if receiver_margin > 0 and chart_contact_margin > 0 and k_r > 0 and k_c > 0:
        barrier = min(k_r * rho_r**2, k_c * rho_c**2) / 2
        rate = 4 * sqrt(2 * h).hi / (3 * k_c)
        if d0 == 0:
            propagated = Q(0)
        else:
            upper = exp_upper_float(rate * end)
            if isfinite(upper):
                amplification = Q(upper)
                propagated = d0 * amplification
        if propagated is not None:
            margin = barrier - propagated
    certified = initial_chart and envelope and margin is not None and margin > 0
    initial_storage = I(0)
    form_lower, contact_phase_lower = Q(0), Q(0)
    contact_keys = {tuple(sorted(edge)) for edge in contacts}
    for i, j in edges:
        form_gap = reference.fine_form[j] - reference.fine_form[i]
        form_radius, phase_radius = fx[i] + fx[j], pt[i] + pt[j]
        phase_turn = (
            reference.fine_phase_turns[j] - reference.fine_phase_turns[i] + Q(1, 2)
        ) % 1 - Q(1, 2)
        phase_gap = 2 * phase_turn * pi + I(-phase_radius, phase_radius)
        form_storage = I(form_gap - form_radius, form_gap + form_radius) ** 2 / 2
        phase_storage = 1 - cos(phase_gap)
        initial_storage += form_storage + phase_storage
        form_lower += form_storage.lo
        if (i, j) in contact_keys:
            contact_phase_lower += phase_storage.lo
    if receiver_chart:
        # Fixed winding and strict acuteness retain the exact correlated
        # receiver phase minimum; independent edge boxes forget that relation.
        correlated_lower = (
            reference.internal_phase_storage_bounds[0].lo
            + form_lower
            + contact_phase_lower
        )
        initial_storage = I(
            max(initial_storage.lo, correlated_lower), initial_storage.hi
        )
    reasons = []
    if not initial_chart:
        reasons.append(
            "initial_error_box_or_declared_radii_do_not_certify_the_acute_chart"
        )
    if not envelope:
        reasons.append("central_reference_contact_libration_envelope_unavailable")
    if propagated is None:
        reasons.append("finite_relative_storage_growth_bound_unavailable")
    elif margin is not None and margin <= 0:
        reasons.append("relative_storage_budget_does_not_certify_retention")
    total_degree = sum(source.degrees)
    acquisition_status = (
        "excluded"
        if receiver_chart and initial_storage.hi < C5_PHASE_SECTOR_BARRIER
        else "not_excluded"
    )
    return SineMovingPatternWindow(
        reference=reference,
        cycle_indices=reference.partition.block_indices[0],
        contact_indices=contacts,
        orientation=orientation,
        form_error_bounds=fx,
        phase_error_bounds=pt,
        scaled_horizon=end,
        original_horizon_bounds=pi * end,
        receiver_phase_radius=rho_r,
        contact_phase_radius=rho_c,
        reference_contact_bound=gamma,
        reference_form_gap=u,
        reference_contact_turn=contact_turn,
        reference_contact_phase_bounds=phi,
        reference_libration_energy_bounds=h,
        reference_energy_ceiling_bounds=ceiling,
        reference_energy_guard_margin_lower_bound=energy_margin,
        reference_contact_initial_margin_lower_bound=contact_margin,
        receiver_chart_margin_lower_bound=receiver_margin,
        contact_chart_margin_lower_bound=chart_contact_margin,
        initial_receiver_phase_error_bounds=receiver_errors,
        initial_contact_phase_error_bounds=contact_errors,
        receiver_curvature_lower_bound=k_r,
        contact_curvature_lower_bound=k_c,
        initial_relative_storage_upper_bound=d0,
        relative_storage_barrier=barrier,
        growth_rate_upper_bound=rate,
        growth_factor_upper_bound=amplification,
        propagated_relative_storage_upper_bound=propagated,
        retention_margin_lower_bound=margin,
        initial_chart_certified=initial_chart,
        reference_contact_envelope_certified=envelope,
        whole_window_retention_certified=certified,
        edge_form_error_upper_bound=sqrt(2 * propagated).hi if certified else None,
        receiver_phase_error_upper_bound=(
            sqrt(2 * propagated / k_r).hi if certified else None
        ),
        contact_phase_error_upper_bound=(
            sqrt(2 * propagated / k_c).hi if certified else None
        ),
        receiver_storage_excess_upper_bound=propagated if certified else None,
        form_mean_error_upper_bound=sum(
            (d * r for d, r in zip(source.degrees, fx)), Q(0)
        )
        / total_degree,
        phase_mean_error_upper_bound=sum(
            (d * r for d, r in zip(source.degrees, pt)), Q(0)
        )
        / total_degree,
        initial_full_storage_bounds=initial_storage,
        phase_flat_acquisition_status=acquisition_status,
        zero_winding_acquisition_status=acquisition_status,
        status="certified" if certified else "unavailable",
        reasons=tuple(reasons),
    )


@dataclass(frozen=True)
class SineCollectivePulseBalance:
    """Actual mean-contact feedback and a conditional exact initial energy jet.

    The declared integer offsets select continuous contact lifts. Changing
    them can change the collective energy and signed remainder while keeping
    physical full storage unchanged. All derivatives use ``tau=t/pi``.
    """

    source: SineExchangeComparison
    cycle: tuple[object, ...]
    cycle_indices: tuple[int, ...]
    contact_indices: tuple[tuple[int, int], ...]
    contact_turn_offsets: tuple[int, ...]
    receiver_form_mean: Q
    environment_form_mean: Q
    form_gap: Q
    contact_phase_bounds: tuple[I, ...]
    mean_contact_phase_bounds: I
    contact_phase_deviation_bounds: tuple[I, ...]
    contact_resultant_real_bounds: I
    contact_resultant_imag_bounds: I
    mean_contact_sine_bounds: I
    form_gap_rate_bounds: I
    mean_contact_phase_rate: Q
    contact_phase_rates: tuple[Q, ...]
    contact_rate_deviations: tuple[Q, ...]
    contact_rate_variance: Q
    contact_rate_third_central_moment: Q
    collective_energy_bounds: I
    collective_energy_rate_bounds: I
    feedback_energy_rate_bounds: I
    energy_rate_factorization_residual_bounds: I
    full_storage_bounds: I
    collective_storage_bounds: I
    remainder_storage_bounds: I
    remainder_storage_rate_bounds: I
    flat_phase_jet_available: bool
    first_three_energy_derivatives: tuple[Q, ...] | None
    fourth_energy_derivative: Q | None
    clock: str = "tau=t/pi"
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "actual_complete_conservative_unit_C5_private_leaf_state_not_a_family_substitution",
        "all_consumed_primitive_state_law_and_support_are_readmitted",
        "contact_offsets_declare_continuous_lifts_without_automatic_edge_wrapping",
        "collective_means_do_not_close_without_retained_contact_phase_deviations",
        "energy_rate_factorization_is_exact_algebra_not_an_interval_zero_test",
        "full_storage_is_conserved_but_five_times_collective_energy_need_not_be",
        "remainder_storage_is_signed_and_is_not_assumed_a_positive_reservoir",
        "flat_phase_jet_requires_equal_exact_source_lifts_and_zero_declared_offsets",
        "fourth_derivative_is_instantaneous_not_a_finite_time_transfer_certificate",
        "receiver_phase_derivatives_are_centered_rows_in_cycle_order_and_scaled_tau",
        "internal_and_contact_phase_rates_use_the_live_complete_support",
        "vanishing_relative_phase_derivatives_do_not_certify_persistent_rigidity",
        "no_trajectory_law_installation_capture_or_physical_identification",
    )
    receiver_internal_phase_rates: tuple[Q, ...] = ()
    receiver_contact_phase_rates: tuple[Q, ...] = ()
    receiver_relative_phase_rates: tuple[Q, ...] = ()
    receiver_relative_phase_acceleration_bounds: tuple[I, ...] = ()
    receiver_relative_phase_jerk_bounds: tuple[I, ...] = ()

    def to_dict(self):
        from ..sdk.relational_reports import _project, _validate_label

        _validate_comparison_labels(self.source)
        for node in self.cycle:
            _validate_label(node)
        return {
            "schema": "tnfr.sine-collective-pulse-balance.v1",
            "report": _project(self),
        }


def _mixed_phase_trig(radian_part, turn_part):
    """Bound trigonometry while retaining exact rational-turn cancellations."""
    if radian_part == 0:
        cache = {}
        return (
            _expression_bound(_sine_expression(((turn_part, Q(1)),)), cache),
            _expression_bound(_sine_expression(((turn_part + Q(1, 4), Q(1)),)), cache),
        )
    # Trig evaluation can reduce exact full turns without changing the
    # separately retained continuous lift used to define the observation.
    reduced_turn = (turn_part + Q(1, 2)) % 1 - Q(1, 2)
    angle = I(radian_part) + 2 * reduced_turn * pi_interval()
    return sin(angle), cos(angle)


def observe_sine_collective_pulse(
    source, *, cycle, contact_turn_offsets
) -> SineCollectivePulseBalance:
    """Rebuild instantaneous collective feedback from the actual fine state.

    ``cycle`` supplies five receiver labels in traversal order. The five
    integer offsets correspond to their matched private leaves in that same
    order, selecting ``phi_i=theta_leaf-theta_receiver-2*pi*k_i``. These are
    explicit continuous-lift choices, not inferred principal branches.

    Mean forms a,b give u=a-b; the full nodal rows imply u'=4/3 mean(sin phi_i)
    and mean(phi)'=-4u/3. Their pendulum energy h is generally not conserved:
    its feedback depends on the actual deviations from the mean contact lift.
    The optional exact fourth derivative concerns only a common initial phase
    and zero offsets. No finite-time conclusion follows from that derivative.

    Receiver phase rates subtract their current receiver mean. Their exact
    internal/contact components are Lq/3 and P(x_R-x_Q)/3, with q=P x_R.
    The acceleration and jerk intervals differentiate the complete live field;
    they are instantaneous radian derivatives with respect to tau, not a
    finite-time Taylor enclosure or a test of invariant collective motion.
    """
    from .relational_sine_comparison import (
        _comparison_from_state,
        _sine_state_from_rows,
    )
    from .relational_sine_entry import _conservative_kl_action, _unit_laplacian_action

    original = source
    source, edges, cycle_indices, _, contacts = _private_leaf_support(source, cycle)
    offsets = _ordered(contact_turn_offsets, "contact_turn_offsets", limit=6)
    if len(offsets) != 5 or any(type(value) is not int for value in offsets):
        raise ValueError("contact_turn_offsets must contain five exact integers")
    pi = pi_interval()
    a = sum((source.epi[i] for i in cycle_indices), Q(0)) / 5
    b = sum((source.epi[j] for _, j in contacts), Q(0)) / 5
    u = a - b
    raw_phase = tuple(source.phase[j] - source.phase[i] for i, j in contacts)
    raw_mean, offset_mean = sum(raw_phase, Q(0)) / 5, Q(sum(offsets), 5)
    lifts = tuple(
        I(value) - 2 * offset * pi for value, offset in zip(raw_phase, offsets)
    )
    mean_phase = I(raw_mean) - 2 * offset_mean * pi
    deviations = tuple(
        I(value - raw_mean) - 2 * (offset - offset_mean) * pi
        for value, offset in zip(raw_phase, offsets)
    )
    contact_trig = tuple(_mixed_phase_trig(value, Q(0)) for value in raw_phase)
    sine_mean = sum((pair[0] for pair in contact_trig), I(0)) / 5
    centered_trig = tuple(
        _mixed_phase_trig(value - raw_mean, offset_mean - offset)
        for value, offset in zip(raw_phase, offsets)
    )
    real = sum((pair[1] for pair in centered_trig), I(0)) / 5
    imag = sum((pair[0] for pair in centered_trig), I(0)) / 5
    mean_sine, mean_cosine = _mixed_phase_trig(raw_mean, -offset_mean)
    u_rate, mean_rate = 4 * sine_mean / 3, -4 * u / 3
    h = u**2 / 2 + 1 - mean_cosine
    h_rate = 4 * u * (sine_mean - mean_sine) / 3
    feedback = 4 * u * ((real - 1) * mean_sine + imag * mean_cosine) / 3
    phase_rates = _conservative_kl_action(source.epi, edges, source.degrees)
    contact_rates = tuple(phase_rates[j] - phase_rates[i] for i, j in contacts)
    if sum(contact_rates, Q(0)) / 5 != mean_rate:
        raise RuntimeError("full phase rows lost the mean-contact identity")
    rate_deviations = tuple(value - mean_rate for value in contact_rates)
    variance = sum((value**2 for value in rate_deviations), Q(0)) / 5
    third = sum((value**3 for value in rate_deviations), Q(0)) / 5

    cycle_edges = tuple((i, (i + 1) % 5) for i in range(5))

    def laplacian(values):
        return _unit_laplacian_action(values, cycle_edges)

    q = tuple(source.epi[i] - a for i in cycle_indices)
    internal_rates = tuple(value / 3 for value in laplacian(q))
    contact_relative_rates = tuple(
        (source.epi[i] - source.epi[j] - u) / 3 for i, j in contacts
    )
    relative_rates = tuple(
        internal + contact
        for internal, contact in zip(internal_rates, contact_relative_rates)
    )
    receiver_rate_mean = sum((phase_rates[i] for i in cycle_indices), Q(0)) / 5
    if relative_rates != tuple(
        phase_rates[i] - receiver_rate_mean for i in cycle_indices
    ):
        raise RuntimeError("full phase rows lost the centered receiver identity")

    internal_sines = [I(0)] * 5
    internal_sine_derivatives = [I(0)] * 5
    for i, j in cycle_edges:
        sine, cosine = _mixed_phase_trig(
            source.phase[cycle_indices[j]] - source.phase[cycle_indices[i]], Q(0)
        )
        sine_derivative = cosine * (relative_rates[j] - relative_rates[i])
        internal_sines[i] += sine
        internal_sines[j] -= sine
        internal_sine_derivatives[i] += sine_derivative
        internal_sine_derivatives[j] -= sine_derivative
    centered_contact_sines = tuple(pair[0] - sine_mean for pair in contact_trig)
    contact_sine_derivatives = tuple(
        pair[1] * rate for pair, rate in zip(contact_trig, contact_rates)
    )
    contact_sine_derivative_mean = sum(contact_sine_derivatives, I(0)) / 5
    centered_contact_sine_derivatives = tuple(
        value - contact_sine_derivative_mean for value in contact_sine_derivatives
    )

    def receiver_phase_derivative(internal, contact):
        # Differentiating the exact relative row gives [(L+I)s+(L+4I)z]/9.
        # For the jerk, s'=-L_cos v and z'=P(cos(phi)*phi') retain the
        # actual full-field contact rates instead of freezing the environment.
        return tuple(
            (li + value + lc + 4 * external) / 9
            for li, value, lc, external in zip(
                laplacian(internal), internal, laplacian(contact), contact
            )
        )

    relative_acceleration = receiver_phase_derivative(
        internal_sines, centered_contact_sines
    )
    relative_jerk = receiver_phase_derivative(
        internal_sine_derivatives, centered_contact_sine_derivatives
    )
    neighbors = [[] for _ in source.nodes]
    for i, j in edges:
        neighbors[i].append(j)
        neighbors[j].append(i)
    rebuilt = _comparison_from_state(
        _sine_state_from_rows(
            source.nodes,
            source.edges,
            source.epi,
            source.phase,
            source.capacity,
            tuple(tuple(row) for row in neighbors),
        ),
        source.reference_model,
    )
    flat = not any(offsets) and all(value == source.phase[0] for value in source.phase)
    fourth = 16 * u**2 * variance / 3 - 4 * u * third / 3 if flat else None
    return SineCollectivePulseBalance(
        source=original,
        cycle=tuple(source.nodes[i] for i in cycle_indices),
        cycle_indices=cycle_indices,
        contact_indices=contacts,
        contact_turn_offsets=offsets,
        receiver_form_mean=a,
        environment_form_mean=b,
        form_gap=u,
        contact_phase_bounds=lifts,
        mean_contact_phase_bounds=mean_phase,
        contact_phase_deviation_bounds=deviations,
        contact_resultant_real_bounds=real,
        contact_resultant_imag_bounds=imag,
        mean_contact_sine_bounds=sine_mean,
        form_gap_rate_bounds=u_rate,
        mean_contact_phase_rate=mean_rate,
        contact_phase_rates=contact_rates,
        contact_rate_deviations=rate_deviations,
        contact_rate_variance=variance,
        contact_rate_third_central_moment=third,
        collective_energy_bounds=h,
        collective_energy_rate_bounds=h_rate,
        feedback_energy_rate_bounds=feedback,
        energy_rate_factorization_residual_bounds=h_rate - feedback,
        full_storage_bounds=rebuilt.storage,
        collective_storage_bounds=5 * h,
        remainder_storage_bounds=rebuilt.storage - 5 * h,
        remainder_storage_rate_bounds=-5 * h_rate,
        flat_phase_jet_available=flat,
        first_three_energy_derivatives=(Q(0),) * 3 if flat else None,
        fourth_energy_derivative=fourth,
        receiver_internal_phase_rates=internal_rates,
        receiver_contact_phase_rates=contact_relative_rates,
        receiver_relative_phase_rates=relative_rates,
        receiver_relative_phase_acceleration_bounds=relative_acceleration,
        receiver_relative_phase_jerk_bounds=relative_jerk,
    )


@dataclass(frozen=True)
class SineContactAveraging:
    """Finite phase-organization obstruction from fast live contact motion.

    The bound applies to the full conservative law, not prescribed contact
    oscillations. Its transformed receiver storage is an analytical estimate,
    not a new dynamical law or a replacement for full-system storage.
    """

    source: SineExchangeComparison
    cycle: tuple[object, ...]
    cycle_indices: tuple[int, ...]
    contact_indices: tuple[tuple[int, int], ...]
    scaled_horizon: Q
    receiver_form: Q
    environment_form: Q
    form_gap: Q
    contact_speed_lower_bound: Q
    contact_acceleration_upper_bound: Q
    integrated_contact_current_upper_bound: Q | None
    receiver_phase_storage_upper_bound: Q | None
    phase_sector_barrier: Q
    initial_full_storage_bounds: I
    speed_separation_certified: bool
    whole_window_acute_winding_excluded: bool
    status: str
    reasons: tuple[str, ...]
    clock: str = "tau=t/pi"
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "complete_conservative_unit_C5_private_leaf_law_with_every_node_live",
        "uniform_initial_forms_within_each_block_and_equal_receiver_phase_lifts",
        "arbitrary_initial_contact_phases_and_both_signs_of_form_gap",
        "primitive_source_support_law_and_full_storage_are_readmitted_and_rebuilt",
        "contact_speed_and_acceleration_are_bounds_of_the_actual_full_field",
        "integrated_current_bound_uses_finite_time_integration_by_parts",
        "transformed_storage_bounds_receiver_phase_without_a_new_constitutive_law",
        "strict_running_C5_phase_barrier_excludes_only_acute_unit_winding",
        "unavailable_is_not_admission_of_formation_or_failure_of_the_model",
        "no_all_time_exclusion_damping_solver_trajectory_or_source_selection",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project, _validate_label

        _validate_comparison_labels(self.source)
        for node in self.cycle:
            _validate_label(node)
        return {
            "schema": "tnfr.sine-contact-averaging.v1",
            "report": _project(self),
        }


def assess_sine_contact_averaging(
    source, *, cycle, scaled_horizon
) -> SineContactAveraging:
    """Bound finite receiver phase storage under uniform-block preparation.

    The clock is ``tau=t/pi``. Let ``u`` be the signed initial receiver-minus-
    leaf form gap and ``T`` the positive horizon. The complete field gives
    contact acceleration at most four and initial contact rate ``-4*u/3``.
    Thus ``M=4*abs(u)/3-4*T>0`` bounds contact speed away from zero and
    ``eps=2/M+4*T/M**2`` bounds every integrated contact sine current.

    Removing this integrated current from receiver form yields an exact
    driven storage balance. Its nodal norm estimates give the whole-window
    receiver phase bound ``640*T**2*eps**2/81``. Only a strict bound below
    the shared C5 sector barrier excludes acute unit winding. Initial leaf
    phases remain arbitrary, and no phase profile or forcing is prescribed
    after preparation. No conclusion about nonacute winding is supplied.
    """
    original = source
    source, _, indices, _, contacts = _private_leaf_support(source, cycle)
    end = exact_or_represented_real(scaled_horizon, "scaled_horizon")
    if end <= 0:
        raise ValueError("scaled_horizon must be strictly positive")
    receiver_form = source.epi[indices[0]]
    leaf_form = source.epi[contacts[0][1]]
    if any(source.epi[i] != receiver_form for i in indices) or any(
        source.epi[j] != leaf_form for _, j in contacts
    ):
        raise ValueError("initial form must be uniform within each block")
    if any(source.phase[i] != source.phase[indices[0]] for i in indices):
        raise ValueError("initial receiver phases must have equal represented lifts")
    labels = tuple(source.nodes[i] for i in indices)
    balance = observe_sine_collective_pulse(
        source, cycle=labels, contact_turn_offsets=(0,) * 5
    )
    gap = receiver_form - leaf_form
    speed = 4 * abs(gap) / 3 - 4 * end
    current = phase_storage = None
    if speed > 0:
        current = 2 / speed + 4 * end / speed**2
        phase_storage = 640 * end**2 * current**2 / 81
    excluded = phase_storage is not None and phase_storage < C5_PHASE_SECTOR_BARRIER
    if speed <= 0:
        reasons = ("contact_speed_separation_not_certified",)
    elif not excluded:
        reasons = ("phase_storage_bound_does_not_exclude_acute_winding",)
    else:
        reasons = ()
    return SineContactAveraging(
        source=original,
        cycle=labels,
        cycle_indices=indices,
        contact_indices=contacts,
        scaled_horizon=end,
        receiver_form=receiver_form,
        environment_form=leaf_form,
        form_gap=gap,
        contact_speed_lower_bound=speed,
        contact_acceleration_upper_bound=Q(4),
        integrated_contact_current_upper_bound=current,
        receiver_phase_storage_upper_bound=phase_storage,
        phase_sector_barrier=C5_PHASE_SECTOR_BARRIER,
        initial_full_storage_bounds=balance.full_storage_bounds,
        speed_separation_certified=speed > 0,
        whole_window_acute_winding_excluded=excluded,
        status="excluded" if excluded else "unavailable",
        reasons=reasons,
    )
