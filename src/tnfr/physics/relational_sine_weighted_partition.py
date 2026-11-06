"""Fixed-offset collective closure for weighted conservative sine exchange.

This detached reader keeps the original clock and the two different channels:
conductance weights enter the form Laplacian, whereas the sine channel counts
bare neighbors. Positive held capacities may differ between nodes. The exact
all-state criterion has an intentionally incomplete symbolic implementation;
an unresolved rational-turn identity is unavailable, never approximately true.
No graph, runtime operator, event selector or trajectory is constructed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real, finite_represented_real
from ..mathematics._rational_interval import INTERVAL_METHOD, I, pi_interval
from ._sine_admission import _sine_model_coefficients
from .epi_memory import _exact_partition
from .relational_observations import _ordered
from .relational_sine_comparison import (
    _sine_phase_rate_numerators,
    _validate_comparison_labels,
)
from .relational_sine_partition import (
    Expression,
    _difference,
    _expression_bound,
    _expression_status,
    _mixed_phase_trig,
    _moment_status,
    _rational_turns,
    _sine_expression,
)
from .relational_sine_support import (
    SineSupportExchangeComparison,
    _rebuild_sine_support_comparison,
    _support_comparison_from_rows,
    _support_topology,
    _support_values,
)

__all__ = (
    "SineWeightedPhaseOffsetPartition",
    "SineWeightedPhaseOffsetState",
    "SineWeightedRadianState",
    "SineWeightedCapacityReset",
    "assess_sine_weighted_phase_offset_partition",
    "assess_sine_weighted_capacity_reset",
)


def _export(schema, report, source):
    from ..sdk.relational_reports import _project

    _validate_comparison_labels(source)
    return {"schema": schema, "report": _project(report)}


@dataclass(frozen=True)
class SineWeightedPhaseOffsetPartition:
    """An admitted conditional family, an obstruction, or explicit abstention.

    Rows are indexed by fine source node and target block. With ``r=nu/d``,
    ``weighted_rows`` are r times the cross conductance sums; complex moment
    rows are r times the *bare* offset phasor sums. Internal real moments are
    retained but not compared. The captured source coordinates need not belong
    to the family. Every consuming operation rebuilds the primitive premises.
    """

    source: SineSupportExchangeComparison
    blocks: tuple[tuple[object, ...], ...]
    block_indices: tuple[tuple[int, ...], ...]
    node_blocks: tuple[int, ...]
    phase_offset_turns: tuple[Q, ...]
    mobility: tuple[Q, ...]
    inverse_capacity_weights: tuple[Q, ...]
    block_mean_weights: tuple[Q, ...]
    weighted_rows: tuple[tuple[Q, ...], ...]
    row_cosine_expressions: tuple[tuple[Expression, ...], ...]
    row_sine_expressions: tuple[tuple[Expression, ...], ...]
    row_cosine_bounds: tuple[tuple[I, ...], ...]
    row_sine_bounds: tuple[tuple[I, ...], ...]
    cross_weighted_residuals: tuple[tuple[Q, ...], ...]
    cross_moment_statuses: tuple[tuple[str, ...], ...]
    internal_current_expressions: tuple[Expression, ...]
    internal_current_bounds: tuple[I, ...]
    internal_current_statuses: tuple[str, ...]
    weighted_compatibility_certified: bool
    moment_compatibility_certified: bool
    internal_balance_certified: bool
    invariance_certified: bool
    status: str
    reasons: tuple[str, ...]
    quotient_weighted_rows: tuple[tuple[Q, ...], ...] | None
    quotient_cosine_expressions: tuple[tuple[Expression, ...], ...] | None
    quotient_sine_expressions: tuple[tuple[Expression, ...], ...] | None
    cross_bare_counts: tuple[tuple[int, ...], ...]
    cross_conductance_sums: tuple[tuple[Q, ...], ...]
    cross_cosine_expressions: tuple[tuple[Expression, ...], ...]
    cross_sine_expressions: tuple[tuple[Expression, ...], ...]
    clock: str = "t"
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "fixed_connected_simple_undirected_support_positive_conductances_and_capacity",
        "zero_loss_unit_exchange_and_storage_scale_original_clock_in_both_rows",
        "cross_weighted_rows_cross_bare_complex_moments_zero_internal_current",
        "exact_all_state_criterion_with_incomplete_symbolic_identity_decision",
        "interval_overlap_never_certifies_an_exact_coefficient_identity",
        "source_coordinates_not_asserted_to_belong_to_the_proposed_family",
        "no_attraction_formation_event_occurrence_or_physical_identification",
    )

    def evaluate(self, collective_form, collective_phase_turns):
        """Evaluate both complete and collective rows, without evolving either."""
        return _evaluate(self, collective_form, collective_phase_turns)

    def evaluate_radians(self, collective_form, collective_phase_radians):
        """Evaluate a zero-offset family at supplied real radian phase lifts.

        The existing turn-valued evaluator retains its separate domain. This
        method admits exact or represented real radians without dividing by pi.
        Source membership is checked separately from all-state family closure.
        """
        return _evaluate_radians(self, collective_form, collective_phase_radians)

    def observe_radian_state(self):
        """Require the captured source to be in its exact zero-offset family.

        Both signed forms and declared radian lifts must agree within every
        block. No averaging, principal-phase projection or tolerance replaces
        membership; a valid partition need not contain its supplied source.
        """
        reference = _zero_offset_partition(self)
        source = reference.source
        forms = tuple(source.epi[block[0]] for block in reference.block_indices)
        phases = tuple(source.phase[block[0]] for block in reference.block_indices)
        if any(
            source.epi[i] != forms[a] or source.phase[i] != phases[a]
            for i, a in enumerate(reference.node_blocks)
        ):
            raise ValueError(
                "captured source forms and radian lifts must agree within each block"
            )
        return _evaluate_radians(reference, forms, phases)

    def capacity_reset(self, capacity_after):
        """Assess supplied represented primary capacities, not a runtime event."""
        return assess_sine_weighted_capacity_reset(self, capacity_after=capacity_after)

    def to_dict(self):
        fresh = _rebuild_partition(self)
        return _export(
            "tnfr.sine-weighted-phase-offset-partition.v1", fresh, fresh.source
        )


@dataclass(frozen=True)
class SineWeightedPhaseOffsetState:
    """Independent fine/coarse rows and inherited storage at one family state.

    Phase rates are radian derivatives with respect to original ``t``, not
    turn rates. Phase means use the explicitly supplied real lifts in turns,
    including fixed offsets. Interval residuals only check arithmetic; exact
    equality follows from the independently rebuilt all-state criterion.
    ``phase_rate_residuals`` and ``phase_mean_rate_numerator`` compare exact
    radian-rate numerators before the common ``1/pi`` factor.
    """

    partition: SineWeightedPhaseOffsetPartition
    collective_form: tuple[Q, ...]
    collective_phase_turns: tuple[Q, ...]
    fine_form: tuple[Q, ...]
    fine_phase_turns: tuple[Q, ...]
    block_form_rates: tuple[I, ...]
    block_phase_rates: tuple[I, ...]
    full_form_rates: tuple[I, ...]
    full_phase_rates: tuple[I, ...]
    full_form_rate_numerator_expressions: tuple[Expression, ...]
    block_phase_rate_numerators: tuple[Q, ...]
    full_phase_rate_numerators: tuple[Q, ...]
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
    quotient_form_storage: Q
    quotient_phase_storage_bounds: I
    storage_residual_bounds: I
    mean_weights: tuple[Q, ...]
    form_mean: Q
    phase_mean_turns: Q
    collective_phase_mean_turns: Q
    phase_offset_mean_turns: Q
    form_mean_rate_bounds: I
    phase_mean_rate_numerator: Q
    clock: str = "t"
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "fine_rows_rebuilt_from_actual_edges_independently_of_quotient_rows",
        "original_t_clock_with_one_over_pi_in_both_complete_rows",
        "inherited_weighted_form_and_bare_phase_storage_no_cross_channel_identification",
        "inverse_capacity_degree_weights_and_declared_lifted_phase_means",
        "no_solver_step_sampled_history_operator_admission_or_future_certificate",
    )

    def to_dict(self):
        fresh = _evaluate(
            self.partition, self.collective_form, self.collective_phase_turns
        )
        return _export(
            "tnfr.sine-weighted-phase-offset-state.v1", fresh, fresh.partition.source
        )


@dataclass(frozen=True)
class SineWeightedRadianState:
    """Exact zero-offset family membership and complete radian-state rows.

    ``full_source`` is the fine law independently evaluated at the reconstructed
    coordinates; ``partition.source`` retains the supplied source. Their exact
    admitted coordinate equality determines ``source_membership_certified``.
    Equality does not authenticate the source or prove acquisition. Full-row
    equality follows from the rebuilt all-state criterion, not overlapping
    numerical intervals. Means retain the explicitly supplied radian lifts.
    """

    partition: SineWeightedPhaseOffsetPartition
    collective_form: tuple[Q, ...]
    collective_phase_radians: tuple[Q, ...]
    full_source: SineSupportExchangeComparison
    source_membership_certified: bool
    quotient_bare_rows: tuple[tuple[Q, ...], ...]
    block_form_rates: tuple[I, ...]
    block_phase_rates: tuple[I, ...]
    block_phase_rate_numerators: tuple[Q, ...]
    full_phase_rate_numerators: tuple[Q, ...]
    form_rate_residual_bounds: tuple[I, ...]
    phase_rate_residuals: tuple[Q, ...]
    full_row_equality_certified: bool
    quotient_form_storage: Q
    quotient_phase_storage_bounds: I
    storage_residual_bounds: I
    mean_weights: tuple[Q, ...]
    form_mean: Q
    phase_mean_radians: Q
    form_mean_rate_bounds: I
    phase_mean_rate_numerator: Q
    clock: str = "t"
    phase_rate_units: str = "radians_per_original_structural_time"
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "certified_zero_offset_partition_rebuilt_from_primitive_source_and_law",
        "exact_or_represented_real_radian_lifts_without_rational_turn_conversion",
        "source_membership_is_exact_form_and_phase_lift_equality_not_authentication",
        "fine_state_and_complete_rows_rebuilt_by_the_shared_weighted_support_owner",
        "independent_collective_rows_use_cross_bare_counts_and_conductance_sums",
        "inherited_storage_and_inverse_capacity_degree_weighted_lifted_means",
        "exact_family_closure_not_a_description_of_arbitrary_off_family_errors",
        "no_solver_event_admission_acquired_history_formation_or_physical_identity",
    )

    def to_dict(self):
        """Re-admit source and coarse coordinates instead of exporting caches."""
        if type(self) is not SineWeightedRadianState:
            raise TypeError("a SineWeightedRadianState is required")
        fresh = _evaluate_radians(
            self.partition, self.collective_form, self.collective_phase_radians
        )
        return _export("tnfr.sine-weighted-radian-state.v1", fresh, fresh.full_source)


@dataclass(frozen=True)
class SineWeightedCapacityReset:
    """Primary capacity-only closure preservation for actual represented values.

    All forms, phases, offsets, support and conductances are held across this
    hypothetical reset. Thus inherited storage has zero jump, while mean
    weights and the vector field generally change. Runtime preconditions,
    grammar, histories, callbacks and event selection require separate state.
    """

    before: SineWeightedPhaseOffsetPartition
    capacity_after: tuple[Q, ...]
    after: SineWeightedPhaseOffsetPartition
    capacity_ratios: tuple[Q, ...]
    block_ratios: tuple[Q, ...] | None
    within_block_ratio_residuals: tuple[Q, ...]
    blockwise_scaling_certified: bool
    preservation_certified: bool
    status: str
    common_clock_scale: Q | None
    same_collective_law: bool
    mean_weight_changes: tuple[Q, ...]
    storage_jump: Q = Q(0)
    scope: tuple[str, ...] = (
        "actual_positive_binary64_represented_outputs_not_ideal_factor_products",
        "fixed_primary_form_phase_support_conductance_and_partition",
        "closure_preserved_iff_actual_capacity_ratios_are_constant_within_each_block",
        "nonuniform_block_scaling_preserves_closure_but_changes_collective_law",
        "zero_storage_jump_does_not_preserve_mean_weights_or_temporal_trigger_evidence",
        "no_generic_SHA_contract_grammar_admission_or_runtime_event_execution",
    )

    def to_dict(self):
        fresh = assess_sine_weighted_capacity_reset(
            self.before, capacity_after=self.capacity_after
        )
        return _export(
            "tnfr.sine-weighted-capacity-reset.v1", fresh, fresh.before.source
        )


def _rebuild_partition(report):
    if type(report) is not SineWeightedPhaseOffsetPartition:
        raise TypeError("a SineWeightedPhaseOffsetPartition is required")
    return assess_sine_weighted_phase_offset_partition(
        report.source,
        blocks=report.blocks,
        phase_offset_turns=report.phase_offset_turns,
    )


def assess_sine_weighted_phase_offset_partition(source, *, blocks, phase_offset_turns):
    """Assess ``x_i=X_a, theta_i=Theta_a+psi_i`` for every collective state.

    Source admission uses the shared weighted comparison owner, then narrows
    to positive conductances/capacities and exact coefficients (0,1,1). Bare
    degree normalizes both original-clock rows. A proper ordered partition has
    2..n-1 nonempty blocks. Offsets require exact rational turns, not rounded
    radians. Cached rows and source verdicts are not premises.
    """
    admitted = _rebuild_sine_support_comparison(source)
    if _sine_model_coefficients(admitted.reference_model) != (0, 1, 1):
        raise ValueError(
            "weighted partition requires zero loss and unit exchange and beta"
        )
    if any(value <= 0 for value in admitted.conductances):
        raise ValueError("weighted partition requires strictly positive conductances")
    if any(value <= 0 for value in admitted.capacity):
        raise ValueError("weighted partition requires strictly positive held capacity")
    blocks = _exact_partition(admitted.nodes, blocks)
    offsets = _rational_turns(
        phase_offset_turns, "phase_offset_turns", len(admitted.nodes)
    )
    positions = {node: i for i, node in enumerate(admitted.nodes)}
    indices = tuple(tuple(positions[node] for node in block) for block in blocks)
    node_blocks = [0] * len(admitted.nodes)
    for a, block in enumerate(indices):
        for i in block:
            node_blocks[i] = a
    node_blocks = tuple(node_blocks)
    edges, neighbors = _support_topology(admitted.nodes, admitted.edges)
    conductance = {}
    for (i, j), weight in zip(edges, admitted.conductances):
        conductance[i, j] = conductance[j, i] = weight
    mobility = tuple(
        value / degree for value, degree in zip(admitted.capacity, admitted.degrees)
    )
    inverse_weights = tuple(1 / value for value in mobility)
    block_weights = tuple(
        sum((inverse_weights[i] for i in block), Q(0)) for block in indices
    )
    count = len(blocks)
    weighted, cosines, sines = [], [], []
    for i, adjacent in enumerate(neighbors):
        groups = tuple(
            tuple(j for j in adjacent if node_blocks[j] == b) for b in range(count)
        )
        weighted.append(
            tuple(
                mobility[i] * sum((conductance[i, j] for j in group), Q(0))
                for group in groups
            )
        )
        cosines.append(
            tuple(
                _sine_expression(
                    (offsets[j] - offsets[i] + Q(1, 4), mobility[i]) for j in group
                )
                for group in groups
            )
        )
        sines.append(
            tuple(
                _sine_expression((offsets[j] - offsets[i], mobility[i]) for j in group)
                for group in groups
            )
        )
    weighted, cosines, sines = tuple(weighted), tuple(cosines), tuple(sines)
    cache = {}
    cosine_bounds = tuple(
        tuple(_expression_bound(expr, cache) for expr in row) for row in cosines
    )
    sine_bounds = tuple(
        tuple(_expression_bound(expr, cache) for expr in row) for row in sines
    )
    residuals, moment_statuses = [], []
    for i, a in enumerate(node_blocks):
        representative = indices[a][0]
        residuals.append(
            tuple(
                Q(0) if a == b else weighted[i][b] - weighted[representative][b]
                for b in range(count)
            )
        )
        moment_statuses.append(
            tuple(
                (
                    "not_required"
                    if a == b
                    else _moment_status(
                        _difference(cosines[i][b], cosines[representative][b]),
                        _difference(sines[i][b], sines[representative][b]),
                        cache,
                    )
                )
                for b in range(count)
            )
        )
    residuals, moment_statuses = tuple(residuals), tuple(moment_statuses)
    internal = tuple(sines[i][a] for i, a in enumerate(node_blocks))
    internal_bounds = tuple(sine_bounds[i][a] for i, a in enumerate(node_blocks))
    internal_statuses = tuple(_expression_status(expr, cache) for expr in internal)
    weighted_ok = not any(value for row in residuals for value in row)
    moment_ok = all(
        value in ("certified", "not_required")
        for row in moment_statuses
        for value in row
    )
    internal_ok = all(value == "certified" for value in internal_statuses)
    certified = weighted_ok and moment_ok and internal_ok
    excluded = (
        not weighted_ok
        or "excluded" in internal_statuses
        or any("excluded" in row for row in moment_statuses)
    )
    reasons = tuple(
        label
        for passed, label in (
            (weighted_ok, "cross_weighted_rows_differ_within_a_block"),
            (moment_ok, "cross_bare_complex_moment_equality_excluded_or_unresolved"),
            (internal_ok, "zero_internal_sine_current_excluded_or_unresolved"),
        )
        if not passed
    )

    # These aggregate edge coefficients retain inherited storage separately
    # from normalized quotient rows. Bare counts are not weighted strength.
    bare_counts = [[0] * count for _ in blocks]
    weight_sums = [[Q(0)] * count for _ in blocks]
    cross_cosine = [[[] for _ in blocks] for _ in blocks]
    cross_sine = [[[] for _ in blocks] for _ in blocks]
    for (i, j), weight in zip(edges, admitted.conductances):
        a, b = node_blocks[i], node_blocks[j]
        if a == b:
            continue
        for left, right, start, end in ((a, b, i, j), (b, a, j, i)):
            bare_counts[left][right] += 1
            weight_sums[left][right] += weight
            cross_cosine[left][right].append(
                (offsets[end] - offsets[start] + Q(1, 4), Q(1))
            )
            cross_sine[left][right].append((offsets[end] - offsets[start], Q(1)))

    def quotient(rows, zero):
        return (
            tuple(
                tuple(zero if a == b else rows[block[0]][b] for b in range(count))
                for a, block in enumerate(indices)
            )
            if certified
            else None
        )

    return SineWeightedPhaseOffsetPartition(
        source=admitted,
        blocks=blocks,
        block_indices=indices,
        node_blocks=node_blocks,
        phase_offset_turns=offsets,
        mobility=mobility,
        inverse_capacity_weights=inverse_weights,
        block_mean_weights=block_weights,
        weighted_rows=weighted,
        row_cosine_expressions=cosines,
        row_sine_expressions=sines,
        row_cosine_bounds=cosine_bounds,
        row_sine_bounds=sine_bounds,
        cross_weighted_residuals=residuals,
        cross_moment_statuses=moment_statuses,
        internal_current_expressions=internal,
        internal_current_bounds=internal_bounds,
        internal_current_statuses=internal_statuses,
        weighted_compatibility_certified=weighted_ok,
        moment_compatibility_certified=moment_ok,
        internal_balance_certified=internal_ok,
        invariance_certified=certified,
        status="certified" if certified else "excluded" if excluded else "unavailable",
        reasons=reasons,
        quotient_weighted_rows=quotient(weighted, Q(0)),
        quotient_cosine_expressions=quotient(cosines, ()),
        quotient_sine_expressions=quotient(sines, ()),
        cross_bare_counts=tuple(map(tuple, bare_counts)),
        cross_conductance_sums=tuple(map(tuple, weight_sums)),
        cross_cosine_expressions=tuple(
            tuple(_sine_expression(terms) for terms in row) for row in cross_cosine
        ),
        cross_sine_expressions=tuple(
            tuple(_sine_expression(terms) for terms in row) for row in cross_sine
        ),
    )


def _evaluate(report, collective_form, collective_phase_turns):
    reference = _rebuild_partition(report)
    if not reference.invariance_certified:
        raise ValueError(
            "evaluation requires a certified weighted phase-offset partition"
        )
    count = len(reference.blocks)
    form = _support_values(collective_form, "collective_form", count)
    turns = _rational_turns(collective_phase_turns, "collective_phase_turns", count)
    fine_form = tuple(form[a] for a in reference.node_blocks)
    fine_turns = tuple(
        turns[a] + offset
        for a, offset in zip(reference.node_blocks, reference.phase_offset_turns)
    )
    source = reference.source
    edges, _ = _support_topology(source.nodes, source.edges)
    gradient = [Q(0)] * len(source.nodes)
    fine_terms = [[] for _ in source.nodes]
    internal_form, internal_phase = [Q(0)] * count, [I(0) for _ in reference.blocks]
    cross_form, cross_phase = Q(0), I(0)
    cache = {}

    def trig(turn):
        return (
            _expression_bound(_sine_expression(((turn + Q(1, 4), Q(1)),)), cache),
            _expression_bound(_sine_expression(((turn, Q(1)),)), cache),
        )

    # Complete rows use only actual fine incidence and reconstructed coordinates.
    for (i, j), weight in zip(edges, source.conductances):
        difference = fine_form[i] - fine_form[j]
        gradient[i] += weight * difference
        gradient[j] -= weight * difference
        delta = fine_turns[j] - fine_turns[i]
        fine_terms[i].append((delta, reference.mobility[i]))
        fine_terms[j].append((-delta, reference.mobility[j]))
        form_cost = weight * difference**2 / 2
        phase_cost = 1 - trig(delta)[0]
        a, b = reference.node_blocks[i], reference.node_blocks[j]
        if a == b:
            internal_form[a] += form_cost
            internal_phase[a] += phase_cost
        else:
            cross_form += form_cost
            cross_phase += phase_cost
    fine_expressions = tuple(_sine_expression(terms) for terms in fine_terms)
    pi = pi_interval()
    full_form_rates = tuple(
        _expression_bound(expr, cache) / pi for expr in fine_expressions
    )
    full_phase_numerators = tuple(
        r * value for r, value in zip(reference.mobility, gradient)
    )
    full_phase_rates = tuple(I(value) / pi for value in full_phase_numerators)

    # Collective rows instead use the admitted cross-block coefficient arrays.
    block_form_rates, block_phase_numerators = [], []
    for a in range(count):
        current, phase_numerator = I(0), Q(0)
        for b in range(count):
            if a == b:
                continue
            cosine, sine = trig(turns[b] - turns[a])
            current += (
                _expression_bound(reference.quotient_cosine_expressions[a][b], cache)
                * sine
                + _expression_bound(reference.quotient_sine_expressions[a][b], cache)
                * cosine
            )
            phase_numerator += reference.quotient_weighted_rows[a][b] * (
                form[a] - form[b]
            )
        block_form_rates.append(current / pi)
        block_phase_numerators.append(phase_numerator)
    block_form_rates, block_phase_numerators = tuple(block_form_rates), tuple(
        block_phase_numerators
    )
    block_phase_rates = tuple(I(value) / pi for value in block_phase_numerators)
    phase_residuals = tuple(
        value - block_phase_numerators[a]
        for value, a in zip(full_phase_numerators, reference.node_blocks)
    )
    form_residuals = tuple(
        value - block_form_rates[a]
        for value, a in zip(full_form_rates, reference.node_blocks)
    )

    quotient_form, quotient_phase = Q(0), I(0)
    for a in range(count):
        for b in range(a + 1, count):
            quotient_form += (
                reference.cross_conductance_sums[a][b] * (form[a] - form[b]) ** 2 / 2
            )
            cosine, sine = trig(turns[b] - turns[a])
            real = _expression_bound(reference.cross_cosine_expressions[a][b], cache)
            imaginary = _expression_bound(reference.cross_sine_expressions[a][b], cache)
            quotient_phase += reference.cross_bare_counts[a][b] - (
                real * cosine - imaginary * sine
            )
    # Internal offset storage is a retained constant, not a quotient self-edge.
    for i, j in edges:
        if reference.node_blocks[i] == reference.node_blocks[j]:
            quotient_phase += (
                1
                - trig(
                    reference.phase_offset_turns[j] - reference.phase_offset_turns[i]
                )[0]
            )
    full_form_storage = sum(internal_form, cross_form)
    full_phase_storage = sum(internal_phase, cross_phase)
    weights = reference.inverse_capacity_weights
    total = sum(weights, Q(0))
    form_mean = sum((w * x for w, x in zip(weights, fine_form)), Q(0)) / total
    phase_mean = sum((w * theta for w, theta in zip(weights, fine_turns)), Q(0)) / total
    collective_mean = (
        sum(
            (
                weight * theta
                for weight, theta in zip(reference.block_mean_weights, turns)
            ),
            Q(0),
        )
        / total
    )
    offset_mean = (
        sum(
            (w * offset for w, offset in zip(weights, reference.phase_offset_turns)),
            Q(0),
        )
        / total
    )
    mean_expression = _sine_expression(
        (turn, weight * coefficient / total)
        for weight, expression in zip(weights, fine_expressions)
        for turn, coefficient in expression
    )
    mean_phase_numerator = (
        sum((w * value for w, value in zip(weights, full_phase_numerators)), Q(0))
        / total
    )
    storage_residual = (full_form_storage - quotient_form) + (
        full_phase_storage - quotient_phase
    )
    # These are independent implementation consistency checks, not the
    # all-state proof: interval overlap alone never certifies closure.
    if any(phase_residuals) or any(not value.contains(0) for value in form_residuals):
        raise ArithmeticError("certified partition has inconsistent full/quotient rows")
    if full_form_storage != quotient_form or not storage_residual.contains(0):
        raise ArithmeticError("certified partition has inconsistent inherited storage")
    if mean_expression or mean_phase_numerator:
        raise ArithmeticError("reciprocal complete rows failed the exact mean balance")
    return SineWeightedPhaseOffsetState(
        partition=reference,
        collective_form=form,
        collective_phase_turns=turns,
        fine_form=fine_form,
        fine_phase_turns=fine_turns,
        block_form_rates=block_form_rates,
        block_phase_rates=block_phase_rates,
        full_form_rates=full_form_rates,
        full_phase_rates=full_phase_rates,
        full_form_rate_numerator_expressions=fine_expressions,
        block_phase_rate_numerators=block_phase_numerators,
        full_phase_rate_numerators=full_phase_numerators,
        form_rate_residual_bounds=form_residuals,
        phase_rate_residuals=phase_residuals,
        full_row_equality_certified=reference.invariance_certified,
        internal_form_storage=tuple(internal_form),
        internal_phase_storage_bounds=tuple(internal_phase),
        cross_form_storage=cross_form,
        cross_phase_storage_bounds=cross_phase,
        full_form_storage=full_form_storage,
        full_phase_storage_bounds=full_phase_storage,
        full_storage_bounds=full_form_storage + full_phase_storage,
        quotient_form_storage=quotient_form,
        quotient_phase_storage_bounds=quotient_phase,
        storage_residual_bounds=storage_residual,
        mean_weights=weights,
        form_mean=form_mean,
        phase_mean_turns=phase_mean,
        collective_phase_mean_turns=collective_mean,
        phase_offset_mean_turns=offset_mean,
        form_mean_rate_bounds=_expression_bound(mean_expression, cache) / pi,
        phase_mean_rate_numerator=mean_phase_numerator,
    )


def _zero_offset_partition(report):
    reference = _rebuild_partition(report)
    if not reference.invariance_certified:
        raise ValueError("radian evaluation requires a certified weighted partition")
    if any(reference.phase_offset_turns):
        raise ValueError("radian evaluation requires exact zero phase offsets")
    return reference


def _evaluate_radians(report, collective_form, collective_phase_radians):
    reference = _zero_offset_partition(report)
    count = len(reference.blocks)
    form = _support_values(collective_form, "collective_form", count)
    phases = _support_values(
        collective_phase_radians, "collective_phase_radians", count
    )
    source = reference.source
    fine_form = tuple(form[a] for a in reference.node_blocks)
    fine_phase = tuple(phases[a] for a in reference.node_blocks)
    full = _support_comparison_from_rows(
        reference_model=source.reference_model,
        nodes=source.nodes,
        edges=source.edges,
        conductances=source.conductances,
        epi=fine_form,
        phase=fine_phase,
        capacity=source.capacity,
    )
    block_weights = reference.block_mean_weights
    bare_rows = tuple(
        tuple(Q(value) / block_weights[a] for value in row)
        for a, row in enumerate(reference.cross_bare_counts)
    )
    weighted_rows = tuple(
        tuple(value / block_weights[a] for value in row)
        for a, row in enumerate(reference.cross_conductance_sums)
    )
    if weighted_rows != reference.quotient_weighted_rows or any(
        reference.quotient_cosine_expressions[a][b]
        != _sine_expression(((Q(1, 4), bare_rows[a][b]),))
        or reference.quotient_sine_expressions[a][b]
        for a in range(count)
        for b in range(count)
    ):
        raise ArithmeticError("zero-offset quotient disagrees with aggregate support")

    pi = pi_interval()
    trig = {}
    for a in range(count):
        for b in range(a + 1, count):
            sine, cosine = _mixed_phase_trig(phases[b] - phases[a], Q(0))
            trig[a, b] = sine, cosine
            trig[b, a] = -sine, cosine
    block_form_rates = tuple(
        sum(
            (bare_rows[a][b] * trig[a, b][0] for b in range(count) if a != b),
            I(0),
        )
        / pi
        for a in range(count)
    )
    block_phase_numerators = tuple(
        sum(
            (weighted_rows[a][b] * (form[a] - form[b]) for b in range(count)),
            Q(0),
        )
        for a in range(count)
    )
    block_phase_rates = tuple(I(value) / pi for value in block_phase_numerators)
    full_phase_numerators = _sine_phase_rate_numerators(
        full.reference_model, full.degrees, full.form_gradient, full.capacity
    )
    form_residuals = tuple(
        value - block_form_rates[a]
        for value, a in zip(full.form_rates, reference.node_blocks)
    )
    phase_residuals = tuple(
        value - block_phase_numerators[a]
        for value, a in zip(full_phase_numerators, reference.node_blocks)
    )
    quotient_form = sum(
        (
            reference.cross_conductance_sums[a][b] * (form[a] - form[b]) ** 2 / 2
            for a in range(count)
            for b in range(a + 1, count)
        ),
        Q(0),
    )
    quotient_phase = sum(
        (
            reference.cross_bare_counts[a][b] * (1 - trig[a, b][1])
            for a in range(count)
            for b in range(a + 1, count)
        ),
        I(0),
    )
    storage_residual = full.storage - (quotient_form + quotient_phase)
    weights = reference.inverse_capacity_weights
    total = sum(weights, Q(0))
    form_mean = sum((w * x for w, x in zip(weights, fine_form)), Q(0)) / total
    phase_mean = sum((w * theta for w, theta in zip(weights, fine_phase)), Q(0)) / total
    block_form_mean = sum((w * x for w, x in zip(block_weights, form)), Q(0)) / total
    block_phase_mean = (
        sum((w * theta for w, theta in zip(block_weights, phases)), Q(0)) / total
    )
    mean_form_rate = (
        sum((w * rate for w, rate in zip(weights, full.form_rates)), I(0)) / total
    )
    mean_phase_numerator = (
        sum((w * value for w, value in zip(weights, full_phase_numerators)), Q(0))
        / total
    )
    # The exact all-state criterion proves closure; independent arithmetic
    # residuals guard the implementation without promoting overlap to proof.
    if any(phase_residuals) or any(not value.contains(0) for value in form_residuals):
        raise ArithmeticError("certified partition has inconsistent radian-state rows")
    if full.form_storage != quotient_form or not storage_residual.contains(0):
        raise ArithmeticError("radian family has inconsistent inherited storage")
    if (
        form_mean != block_form_mean
        or phase_mean != block_phase_mean
        or mean_phase_numerator
        or not mean_form_rate.contains(0)
    ):
        raise ArithmeticError("radian family failed its inherited mean balance")
    return SineWeightedRadianState(
        partition=reference,
        collective_form=form,
        collective_phase_radians=phases,
        full_source=full,
        source_membership_certified=(
            source.epi == fine_form and source.phase == fine_phase
        ),
        quotient_bare_rows=bare_rows,
        block_form_rates=block_form_rates,
        block_phase_rates=block_phase_rates,
        block_phase_rate_numerators=block_phase_numerators,
        full_phase_rate_numerators=full_phase_numerators,
        form_rate_residual_bounds=form_residuals,
        phase_rate_residuals=phase_residuals,
        full_row_equality_certified=reference.invariance_certified,
        quotient_form_storage=quotient_form,
        quotient_phase_storage_bounds=quotient_phase,
        storage_residual_bounds=storage_residual,
        mean_weights=weights,
        form_mean=form_mean,
        phase_mean_radians=phase_mean,
        form_mean_rate_bounds=mean_form_rate,
        phase_mean_rate_numerator=mean_phase_numerator,
    )


def assess_sine_weighted_capacity_reset(partition, *, capacity_after):
    """Classify an already represented capacity-only reset of a closed family.

    Positive conductances and connected support ensure every block has a
    nonzero cross weighted coefficient. Therefore closure is preserved iff
    actual ``nu_after/nu_before`` is constant within each block. A common ideal
    multiplier is insufficient when its separately rounded products disagree.
    Input outputs must already equal finite binary64 values; this function
    does not silently materialize an ideal rational product or apply SHA.
    """
    before = _rebuild_partition(partition)
    if not before.invariance_certified:
        raise ValueError(
            "capacity reset requires a certified initial weighted partition"
        )
    raw = _ordered(capacity_after, "capacity_after", limit=len(before.source.nodes) + 1)
    if len(raw) != len(before.source.nodes):
        raise ValueError("capacity_after must contain one represented output per node")
    admitted = []
    for i, value in enumerate(raw):
        label = f"capacity_after[{i}]"
        exact = exact_or_represented_real(value, label)
        represented = finite_represented_real(value, label)[1]
        if exact != represented:
            raise ValueError(f"{label} must already equal its binary64 representation")
        if represented <= 0:
            raise ValueError("capacity_after must be strictly positive")
        admitted.append(represented)
    admitted = tuple(admitted)
    after = assess_sine_weighted_phase_offset_partition(
        replace(before.source, capacity=admitted),
        blocks=before.blocks,
        phase_offset_turns=before.phase_offset_turns,
    )
    ratios = tuple(new / old for new, old in zip(admitted, before.source.capacity))
    residuals = tuple(
        ratios[i] - ratios[before.block_indices[a][0]]
        for i, a in enumerate(before.node_blocks)
    )
    blockwise = not any(residuals)
    preserved = blockwise and after.invariance_certified
    return SineWeightedCapacityReset(
        before=before,
        capacity_after=admitted,
        after=after,
        capacity_ratios=ratios,
        block_ratios=(
            tuple(ratios[block[0]] for block in before.block_indices)
            if blockwise
            else None
        ),
        within_block_ratio_residuals=residuals,
        blockwise_scaling_certified=blockwise,
        preservation_certified=preserved,
        status=(
            "certified" if preserved else "excluded" if not blockwise else "unavailable"
        ),
        common_clock_scale=(
            ratios[0] if all(value == ratios[0] for value in ratios) else None
        ),
        same_collective_law=all(value == 1 for value in ratios),
        mean_weight_changes=tuple(
            new - old
            for new, old in zip(
                after.inverse_capacity_weights, before.inverse_capacity_weights
            )
        ),
    )
