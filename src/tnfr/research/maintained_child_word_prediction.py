"""Conditional THOL/refresh/SHA/refresh on the acquired child-owned history.

This predicts two supplied, instantaneous stages at one structural time. It
executes neither stage, materializes no newborn form from its interval and
supplies no future trajectory. The complete successor explicitly freezes the
new isolate; SHA changes both continuous rows of its actor through capacity.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from fractions import Fraction as Q

from ..mathematics._rational_interval import I
from ..operators import _validated_execution_window
from ..operators.definitions import Dissonance, SelfOrganization, Silence
from ..operators.factor_contracts import resolve_runtime_operator_factors
from ..operators.grammar_debt import advance_debt, advance_prior_coherence
from ..operators.grammar_dynamics import validate_candidate
from ..operators.grammar_execution import ValidatedSequence
from ..operators.preconditions import validate_silence
from ..sdk._state import copy_graph_state
from ..types import Glyph
from .local_capacity_history import CONTEXT
from .maintained_child_history_source import load_child_history_source
from .maintained_child_thol_prediction import (
    ACTOR,
    CHILD,
    CHILD_INDEX,
    ChildTholPrediction,
    _predict_child_thol,
)
from .maintained_parent_sha import _capacity_reset, _mean
from .maintained_parent_thol import _represented, _signature
from .maintained_parent_um_work import StateBounds, _bound_state


@dataclass(frozen=True)
class ChildWordPrediction:
    """Full instantaneous comparison, conditional on two successful stages.

    ``before``, ``ideal`` and ``after`` retain only the existing connected 21
    nodes. Their old-marked fields still mean the original twenty-node subset
    defined by StateBounds. ``successor`` covers all 22 nodes and the entire
    newborn form interval, with a zero isolate mean weight, not a positive
    metric on the enlarged state. Edges are canonical node-index pairs.

    The hypothetical SHA grammar check changes only copied grammar metadata;
    it is neither an executed prefix nor a constructed event successor.
    Zero storage jumps follow from unchanged edge arguments. Rate ratios and
    other-node equalities are instantaneous, not future-trajectory identities.
    """

    thol: ChildTholPrediction
    nodes: tuple
    actor_index: int
    isolate_index: int
    edges: tuple[tuple[int, int], ...]
    conductances: tuple[Q, ...]
    forms: tuple[I, ...]
    phases: tuple[Q, ...]
    capacity_before: tuple[Q, ...]
    ideal_capacity_after: tuple[Q, ...]
    represented_capacity_after: tuple[Q, ...]
    successor_capacities: tuple[Q, ...]
    sha_factor: Q
    ideal_actor_capacity_after: Q
    actor_capacity_after: Q
    capacity_rounding_defect: Q
    capacity_absolute_error: Q
    represented_capacity_ratio: Q
    before: StateBounds
    ideal: StateBounds
    after: StateBounds
    successor: StateBounds
    constitutive_pressure_unchanged: bool
    expected_stored_pressures: tuple[Q, ...]
    other_node_instantaneous_rows_unchanged: bool
    actor_form_rate_multiplier: Q
    actor_phase_rate_multiplier: Q
    mean_weights_before: tuple[Q, ...]
    mean_weights_after: tuple[Q, ...]
    form_mean_before: Q
    form_mean_after: Q
    phase_mean_before: Q
    phase_mean_after: Q
    form_mean_jump: Q
    phase_mean_jump: Q
    thol_storage_jump: Q
    sha_storage_jump: Q
    total_storage_jump: Q
    structural_time: Q
    sha_preconditions_admitted: bool
    hypothetical_sha_grammar_allowed: bool
    hypothetical_actor_glyph_history: tuple[str, ...]
    hypothetical_actor_operator_step: int
    hypothetical_actor_u2_debt: int
    expected_actor_glyph_history: tuple[str, ...]
    expected_actor_operator_step: int
    expected_actor_u2_debt: int
    expected_actor_latent: bool
    preserved_epi: Q
    was_initial_on_silence: bool
    expected_silence_duration: Q
    actor_timed_history_preserved: bool
    expected_new_child_glyph_history: tuple[str, ...]
    new_child_timed_history_available: bool
    isolate_pressure: Q
    isolate_form_rate: Q
    isolate_phase_rate: Q
    global_positive_mean_metric_available: bool
    source_receipts: tuple[dict, ...]
    selected_word: tuple[str, ...] = ("OZ", "THOL", "SHA")
    prospective_stages: tuple[str, ...] = ("THOL", "refresh", "SHA", "refresh")
    successor_law: str = "supplied_frozen_isolate_completion"
    whole_word_atomic: bool = False
    complete_word_executed: bool = False
    events_executed: int = 0
    latency_wall_timestamp_available: bool = False
    future_from_acquired_point_certified: bool = False
    old_retention_transferred_after_capacity_reset: bool = False
    numerical_u5_observation_available: bool = False
    cross_birth_u6_observation_available: bool = False
    autonomous_occurrence_derived: bool = False


def predict_child_word(evidence_dir, *, repository_root):
    """Freshly admit evidence and predict the remaining two supplied stages."""
    admitted = load_child_history_source(evidence_dir, repository_root=repository_root)
    return _predict_child_word(admitted)


def _sha_context(graph, thol):
    """Read-only conditional SHA admission on a detached grammar-only view.

    THOL leaves every coordinate consumed by SHA's strict gate unchanged.
    No new node, form sample, pressure jump or runtime event is installed in
    this view. It only supplies the bookkeeping that a successful THOL would
    have, using the shared grammar recurrences and actual retained prefix.
    """
    validate_silence(graph, ACTOR)
    shadow = copy_graph_state(graph)
    actor = shadow.nodes[ACTOR]
    history = graph.nodes[ACTOR]["glyph_history"]
    window = _validated_execution_window(graph, None)
    if window < 3:
        raise ValueError(
            "selected word requires the retained three-token context window"
        )
    actor["glyph_history"] = deque((*history, "THOL"), maxlen=window)
    actor["_operator_step"] += 1
    actor["_grammar_u2_debt"] = advance_debt(actor["_grammar_u2_debt"], "THOL")
    actor["_grammar_prior_coherence"] = advance_prior_coherence(
        actor["_grammar_prior_coherence"], "THOL"
    )
    if (
        tuple(actor["glyph_history"]) != thol.expected_actor_glyph_history
        or actor["_grammar_u2_debt"] != thol.expected_actor_u2_debt
    ):
        raise ValueError(
            "hypothetical THOL bookkeeping differs from its admitted reset"
        )
    sequence = ValidatedSequence(
        (Dissonance(), SelfOrganization(), Silence()), context=CONTEXT
    )
    signature = _signature(shadow)
    decision = validate_candidate(
        shadow, ACTOR, "SHA", sequence_context=sequence.step(2)
    )
    if _signature(shadow) != signature or not decision.allowed:
        raise ValueError("read-only hypothetical SHA grammar admission failed")
    return (
        tuple(actor["glyph_history"]),
        actor["_operator_step"],
        actor["_grammar_u2_debt"],
    )


def _predict_child_word(admitted):
    """Private same-call composition; mutable cached reports are not public inputs."""
    graph = admitted.graph
    signature = _signature(graph)
    thol = _predict_child_thol(admitted)
    if any(str(key).startswith("SHA_") for key in graph.graph):
        raise ValueError("selected child closure requires absent SHA overrides")
    factor = _represented(
        resolve_runtime_operator_factors(
            graph.graph.get("GLYPH_FACTORS"), Glyph.SHA, graph.graph
        )["SHA_vf_factor"],
        "SHA factor",
    )
    if not 0 < factor < 1:
        raise ValueError("selected closure requires positive SHA attenuation")
    context_history, context_step, context_debt = _sha_context(graph, thol)
    ideal_actor, represented_actor, error = _capacity_reset(thol.actor_capacity, factor)
    if not 0 < represented_actor < thol.actor_capacity or ideal_actor <= 0:
        raise ValueError(
            "selected closure must retain positive attenuated actor capacity"
        )
    ratio = represented_actor / thol.actor_capacity
    ideal_capacities, capacities = list(thol.source_capacities), list(
        thol.source_capacities
    )
    ideal_capacities[CHILD_INDEX], capacities[CHILD_INDEX] = (
        ideal_actor,
        represented_actor,
    )
    ideal_capacities, capacities = tuple(ideal_capacities), tuple(capacities)
    edges = thol.source_principal_edges
    positions = {node: index for index, node in enumerate(thol.source_nodes)}
    weights = {
        tuple(sorted((positions[a], positions[b]))): weight
        for (a, b), weight in zip(
            thol.source_edges, thol.source_conductances, strict=True
        )
    }
    conductances = tuple(weights[edge] for edge in edges)

    def state(capacity_values):
        return _bound_state(
            thol.source_forms,
            tuple(map(I, thol.source_phases)),
            tuple(map(I, capacity_values)),
            edges,
            tuple(map(I, conductances)),
        )

    before, ideal, after = (
        state(row) for row in (thol.source_capacities, ideal_capacities, capacities)
    )
    forms = (*map(I, thol.source_forms), thol.child_form_bounds)
    phases = (*thol.source_phases, thol.child_phase)
    successor_capacities = (*capacities, thol.child_capacity)
    successor = _bound_state(
        forms,
        tuple(map(I, phases)),
        tuple(map(I, successor_capacities)),
        edges,
        tuple(map(I, conductances)),
    )
    weights_before, weights_after = (
        tuple(
            Q(degree) / capacity
            for degree, capacity in zip(thol.source_degrees, row, strict=True)
        )
        for row in (thol.source_capacities, capacities)
    )
    form_before, form_after = (
        _mean(thol.source_forms, row) for row in (weights_before, weights_after)
    )
    phase_before, phase_after = (
        _mean(thol.source_phases, row) for row in (weights_before, weights_after)
    )
    pressure_same = before.pressure == ideal.pressure == after.pressure
    other_rows_same = all(
        before.form_rate[index] == after.form_rate[index]
        and before.phase_rate[index] == after.phase_rate[index]
        for index in range(len(thol.source_nodes))
        if index != CHILD_INDEX
    )
    if not pressure_same or not other_rows_same:
        raise ValueError(
            "instantaneous selected-law SHA identities were not reproduced"
        )
    if _signature(graph) != signature:
        raise ValueError("read-only child word prediction changed persistent state")
    return ChildWordPrediction(
        thol=thol,
        nodes=(*thol.source_nodes, CHILD),
        actor_index=CHILD_INDEX,
        isolate_index=len(thol.source_nodes),
        edges=edges,
        conductances=conductances,
        forms=forms,
        phases=phases,
        capacity_before=thol.source_capacities,
        ideal_capacity_after=ideal_capacities,
        represented_capacity_after=capacities,
        successor_capacities=successor_capacities,
        sha_factor=factor,
        ideal_actor_capacity_after=ideal_actor,
        actor_capacity_after=represented_actor,
        capacity_rounding_defect=represented_actor - ideal_actor,
        capacity_absolute_error=error,
        represented_capacity_ratio=ratio,
        before=before,
        ideal=ideal,
        after=after,
        successor=successor,
        constitutive_pressure_unchanged=pressure_same,
        expected_stored_pressures=(*thol.source_stored_pressures, Q(0)),
        other_node_instantaneous_rows_unchanged=other_rows_same,
        actor_form_rate_multiplier=ratio,
        actor_phase_rate_multiplier=ratio,
        mean_weights_before=weights_before,
        mean_weights_after=weights_after,
        form_mean_before=form_before,
        form_mean_after=form_after,
        phase_mean_before=phase_before,
        phase_mean_after=phase_after,
        form_mean_jump=form_after - form_before,
        phase_mean_jump=phase_after - phase_before,
        thol_storage_jump=Q(0),
        sha_storage_jump=Q(0),
        total_storage_jump=Q(0),
        structural_time=thol.source_time,
        sha_preconditions_admitted=True,
        hypothetical_sha_grammar_allowed=True,
        hypothetical_actor_glyph_history=context_history,
        hypothetical_actor_operator_step=context_step,
        hypothetical_actor_u2_debt=context_debt,
        expected_actor_glyph_history=(*context_history, "SHA"),
        expected_actor_operator_step=context_step + 1,
        expected_actor_u2_debt=advance_debt(context_debt, "SHA"),
        expected_actor_latent=True,
        preserved_epi=thol.actor_form,
        was_initial_on_silence=abs(float(thol.actor_form)) < 1e-6,
        expected_silence_duration=Q(0),
        actor_timed_history_preserved=True,
        expected_new_child_glyph_history=(),
        new_child_timed_history_available=False,
        isolate_pressure=Q(0),
        isolate_form_rate=Q(0),
        isolate_phase_rate=Q(0),
        global_positive_mean_metric_available=False,
        source_receipts=admitted.receipts,
    )
