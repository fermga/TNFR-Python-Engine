"""One frozen parent-UM event and detached admission of its actual endpoint.

Occurrence is supplied. The executor does not advance either continuous row;
the checker rebuilds the hypothetical reset only from the before graph. Source
artifact association and transfer of the separately retained motion certificate
belong to the caller. Neither a cached prediction nor a passing flag is a premise.
"""

from __future__ import annotations

from datetime import datetime, timezone
from fractions import Fraction as Q
from unittest.mock import patch

from ..alias import get_attr
from ..constants.aliases import (
    ALIAS_DNFR,
    ALIAS_EPI,
    ALIAS_SOURCE_GLYPH,
    ALIAS_THETA,
    ALIAS_VF,
)
from ..mathematics._rational_interval import I
from ..operators import network_stage
from ..operators.definitions import Coupling, Expansion, SelfOrganization, Silence
from ..operators.grammar_debt import node_debt
from ..operators.grammar_dynamics import validate_candidate
from ..operators.grammar_execution import ValidatedSequence
from ..operators.nodal_equation import observe_structural_acceleration
from ..operators.preconditions import validate_coupling
from ..operators.stage_contracts import stage_schedule_metadata
from ..operators.word_execution import execute_network_operator_stage
from ..physics.relational_sine_support import bound_relational_sine_support_exchange
from ..utils._structural_signature import structural_proof_signature
from ..utils.cache import GRAPH_RUNTIME_CACHE_KEYS
from ..utils.io import json_dumps
from .artifact_io import encode_exact_tree
from .local_capacity_history import (
    CONTEXT,
    MODEL,
    PRESSURE_ERROR_BUDGET,
    TIMES,
    encode_result,
    refresh_pressure,
)
from .maintained_parent_thol import _signature
from .maintained_parent_um import CHILD, NUMERICAL_ALLOWANCE, PARENT, _represented
from .maintained_parent_um_source import TholSuccessorAdmission
from .maintained_parent_um_work import _bound_state, _old_radius, _storage_work
from .maintained_successor_chamber import _certify_chamber
from .maintained_successor_motion import predict_successor_motion
from .retained_graph_state import (
    encode_recorded_container,
    record_graph_state,
    restore_recorded_graph_state,
)

_PRESSURE_CACHE_KEYS = frozenset({"_dnfrmax", "_dnfrmax_node"})
_STAGE_KEYS = frozenset(
    {network_stage.STAGE_SCHEDULE_KEY, network_stage.STAGE_CONTRACT_KEY}
)
_PARENT_LIFECYCLE_KEYS = frozenset(
    {
        "glyph_history",
        "_operator_step",
        "_grammar_u2_debt",
        "_grammar_prior_coherence",
    }
)


class _EventClock:
    @classmethod
    def now(cls, tz=None):
        return datetime(2030, 1, 2, 3, 4, 7, tzinfo=timezone.utc)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _same(left, right):
    """Retain scalar and container types rather than Python numeric equality."""
    return json_dumps(encode_exact_tree(encode_recorded_container(left))) == json_dumps(
        encode_exact_tree(encode_recorded_container(right))
    )


def _read(graph, node, aliases):
    return _represented(
        get_attr(graph.nodes[node], aliases, strict=True, conv=lambda value: value),
        f"node {node!r} {aliases[0]}",
    )


def _without(mapping, excluded):
    return {key: value for key, value in mapping.items() if key not in excluded}


def _write_alias(data, aliases):
    """Match the shared setter's first-present key without mutating storage."""
    return next((key for key in aliases if key in data), aliases[0])


def _pressure_preservation_signature(graph, *, final_schedule=False, runtime=False):
    """Sign everything except pressure aliases and explicitly permitted maxima.

    The live callback also protects runtime graph caches. Retained primitive
    records omit the shared declared rebuildable caches; the post-stage reader
    separately validates the two schedule metadata fields written after refresh.
    """
    excluded = _PRESSURE_CACHE_KEYS
    if final_schedule:
        excluded |= _STAGE_KEYS
    if not runtime:
        excluded |= GRAPH_RUNTIME_CACHE_KEYS
    state = (
        tuple(graph._node),
        tuple((node, tuple(neighbors)) for node, neighbors in graph._adj.items()),
        tuple(
            (
                node,
                _write_alias(data, ALIAS_DNFR),
                _write_alias(data, ALIAS_DNFR) in data,
                _without(data, {_write_alias(data, ALIAS_DNFR)}),
            )
            for node, data in graph._node.items()
        ),
        tuple(graph.edges(data=True)),
        _without(graph.graph, excluded),
        getattr(graph, "_last_operator_applied", None),
    )
    # Cache managers retain graph.graph as their owner. Its direct protected
    # fields are already signed above; following that back-reference would
    # reintroduce the deliberately excluded pressure maxima. Cache-owned
    # payload remains traversed, while this separately checked owner is opaque.
    return structural_proof_signature(
        state, opaque_references=(graph.graph,) if runtime else ()
    )


def _source_prediction(before, supplied=None):
    """Admit the unchanged preparation, not an already reset successor."""
    prediction = predict_successor_motion(before, horizon=Q(1, 2))
    if supplied is not None:
        _require(
            _same(encode_result(supplied), encode_result(prediction)),
            "supplied UM prediction differs from the independently rebuilt source",
        )
    parent, child = before.nodes[PARENT], before.nodes[CHILD]
    _require(
        tuple(parent.get("glyph_history", ())) == ("VAL", "THOL")
        and type(parent.get("_grammar_u2_debt")) is int
        and parent["_grammar_u2_debt"] == node_debt(parent) == 0
        and type(parent.get("_operator_step")) is int
        and parent["_operator_step"] == 2,
        "source requires the actual VAL,THOL prefix and zero debt",
    )
    _require(
        _represented(before.graph.get("_t"), "source structural clock") == TIMES[-1]
        and parent.get("epi_time_history")
        and _represented(parent["epi_time_history"][-1][0], "history endpoint")
        == TIMES[-1]
        and _represented(parent["epi_time_history"][-1][1], "history final form")
        == _read(before, PARENT, ALIAS_EPI),
        "source history must end at the retained live state",
    )
    _require(
        before.graph.get("hierarchy") == {PARENT: [CHILD]}
        and parent.get("sub_nodes") == [CHILD]
        and type(child.get("parent_node")) is int
        and child["parent_node"] == PARENT
        and child.get("glyph_history") == []
        and "epi_time_history" not in child
        and not observe_structural_acceleration(before, CHILD).available,
        "source child identity or unavailable baseline changed",
    )
    sequence = ValidatedSequence(
        (Expansion(), SelfOrganization(), Coupling(), Silence()), context=CONTEXT
    )
    signature = _signature(before)
    validate_coupling(before, PARENT)
    decision = validate_candidate(
        before, PARENT, "UM", sequence_context=sequence.step(2)
    )
    _require(decision.allowed, "source live UM grammar rejected")
    _require(_signature(before) == signature, "source admission changed state")
    _require(
        prediction.chamber.numerical_allowance == NUMERICAL_ALLOWANCE
        and len(prediction.chamber.edges) == 53
        and len(prediction.chamber.reset.new_edges) == 8,
        "source prediction differs from the fixed allowance or support",
    )
    return prediction


def _farthest_defect(value, ideal):
    """Sufficient absolute error against every enclosed ideal value."""
    ideal = I.coerce(ideal)
    return max(abs(value - ideal.lo), abs(value - ideal.hi))


def _contains(interval, value):
    return interval.lo <= value <= interval.hi


def _interval_contained(inner, outer):
    return outer.lo <= inner.lo <= inner.hi <= outer.hi


def _stage_observation(stage):
    """Read plain executor evidence without treating it as authentication."""
    data = encode_result(stage)
    if data is None:
        return False
    _require(type(data) is dict, "stage observation must be a mapping")
    return (
        data.get("operator") == "coupling"
        and data.get("glyph") == "UM"
        and data.get("schedule") == "two_phase_jacobi"
        and type(data.get("nodes_processed")) is int
        and data["nodes_processed"] == 1
    )


def _persistent_graph(graph, excluded):
    return _without(graph.graph, GRAPH_RUNTIME_CACHE_KEYS | excluded)


def _maximum_matches(graph, aliases, key):
    witness = graph.graph.get(f"{key}_node")
    if witness not in graph:
        return False
    maximum = max(abs(_read(graph, node, aliases)) for node in graph)
    return (
        _represented(graph.graph.get(key), key) == maximum
        and abs(_read(graph, witness, aliases)) == maximum
    )


def _context_checks(before, raw, endpoint, reset):
    """Check the declared semantic footprint, including unchanged histories."""
    excluded_graph = (
        _PRESSURE_CACHE_KEYS
        | _STAGE_KEYS
        | {"_vfmax", "_vfmax_node", "_edge_version", "_dnfr_prep_dirty"}
    )
    graph_context = all(
        # Every added edge invalidates preparation through the shared topology
        # owner. This exact flag is required, not an ignored cache payload.
        state.graph.get("_dnfr_prep_dirty") is True
        and _same(
            _persistent_graph(before, excluded_graph),
            _persistent_graph(state, excluded_graph),
        )
        for state in (raw, endpoint)
    )
    nodal_context = True
    phase_aliases = True
    for index, node in enumerate(reset.nodes):
        allowed = set()
        if index in reset.phase_changed_indices:
            allowed.update(ALIAS_THETA)
            # set_theta deliberately writes both English compatibility keys;
            # ignoring them as a group must not admit a contradictory shadow.
            phase_aliases &= all(
                set(ALIAS_THETA).issubset(state.nodes[node])
                and all(
                    _represented(state.nodes[node][key], "changed phase alias")
                    == _read(state, node, ALIAS_THETA)
                    for key in ALIAS_THETA
                )
                for state in (raw, endpoint)
            )
        if index in reset.capacity_changed_indices:
            allowed.add(_write_alias(before.nodes[node], ALIAS_VF))
        if node == PARENT:
            allowed.add(_write_alias(before.nodes[node], ALIAS_DNFR))
            allowed.add(_write_alias(before.nodes[node], ALIAS_SOURCE_GLYPH))
            allowed.update(_PARENT_LIFECYCLE_KEYS)
        nodal_context &= _same(
            _without(before.nodes[node], allowed), _without(raw.nodes[node], allowed)
        )
    parent, old_parent = endpoint.nodes[PARENT], before.nodes[PARENT]
    expected_history = tuple(old_parent["glyph_history"]) + ("UM",)
    lifecycle = all(
        tuple(state.nodes[PARENT].get("glyph_history", ())) == expected_history
        and type(state.nodes[PARENT].get("glyph_history"))
        is type(old_parent["glyph_history"])
        and getattr(state.nodes[PARENT]["glyph_history"], "maxlen", None)
        == getattr(old_parent["glyph_history"], "maxlen", None)
        and type(state.nodes[PARENT].get("_operator_step")) is int
        and state.nodes[PARENT]["_operator_step"] == 3
        and type(state.nodes[PARENT].get("_grammar_u2_debt")) is int
        and state.nodes[PARENT]["_grammar_u2_debt"] == 0
        and node_debt(state.nodes[PARENT]) == 0
        and state.nodes[PARENT].get("_grammar_prior_coherence") is False
        and get_attr(
            state.nodes[PARENT], ALIAS_SOURCE_GLYPH, strict=True, conv=lambda x: x
        )
        == "UM"
        and getattr(state, "_last_operator_applied", None) == "coupling"
        for state in (raw, endpoint)
    )
    expected_schedule = {
        "operator": "coupling",
        "glyph": "UM",
        "schedule": "two_phase_jacobi",
        "nodes_processed": 1,
    }
    expected_contract = stage_schedule_metadata(
        "coupling", observed_schedule="two_phase_jacobi"
    )
    return {
        "declared_graph_context_preserved": graph_context,
        "declared_nodal_context_preserved": bool(nodal_context),
        "canonical_changed_phase_aliases": bool(phase_aliases),
        "actual_parent_grammar_prefix": lifecycle,
        "parent_temporal_history_preserved": _same(
            parent.get("epi_time_history"), old_parent.get("epi_time_history")
        ),
        "child_temporal_baseline_unavailable": (
            "epi_time_history" not in endpoint.nodes[CHILD]
            and not observe_structural_acceleration(endpoint, CHILD).available
        ),
        "structural_clock_unchanged": all(
            _represented(state.graph.get("_t"), "event structural clock") == TIMES[-1]
            for state in (raw, endpoint)
        ),
        "raw_schedule_precedes_final_metadata": all(
            _same(raw.graph.get(key), before.graph.get(key)) for key in _STAGE_KEYS
        ),
        "final_schedule_metadata": _same(
            endpoint.graph.get(network_stage.STAGE_SCHEDULE_KEY), expected_schedule
        )
        and _same(
            endpoint.graph.get(network_stage.STAGE_CONTRACT_KEY), expected_contract
        ),
        "edge_version_records_eight_additions": all(
            type(state.graph.get("_edge_version")) is int
            and state.graph["_edge_version"]
            == before.graph.get("_edge_version", 0) + len(reset.new_edges)
            for state in (raw, endpoint)
        ),
        "declared_alias_maxima_valid": all(
            _maximum_matches(state, aliases, key)
            for state in (raw, endpoint)
            for aliases, key in ((ALIAS_DNFR, "_dnfrmax"), (ALIAS_VF, "_vfmax"))
        ),
    }


def check_um_endpoint(before, raw_after, endpoint, *, prediction=None, stage=None):
    """Rebuild one endpoint assessment from three primitive graphs, without events.

    The caller associates ``before`` with a freshly admitted retained source.
    A supplied prediction is checked against a rebuild on ``before`` only.
    Finite out-of-budget outputs yield failed checks; an inadmissible source or
    successor law raises instead of coercing coordinates or inventing fields.
    Raw pressure and refreshed pressure have independent error obligations.
    """
    prediction = _source_prediction(before, prediction)
    chamber, reset = prediction.chamber, prediction.chamber.reset
    for graph in (raw_after, endpoint):
        _require(tuple(graph) == reset.nodes, "successor node order changed")
    raw_fields = bound_relational_sine_support_exchange(
        raw_after, reference_model=MODEL
    )
    fields = bound_relational_sine_support_exchange(endpoint, reference_model=MODEL)
    positions = {node: index for index, node in enumerate(reset.nodes)}

    def edge_map(graph):
        return {
            tuple(sorted((positions[left], positions[right]))): graph.edges[left, right]
            for left, right in graph.edges
        }

    raw_edges, edges = edge_map(raw_after), edge_map(endpoint)
    _require(
        tuple(sorted(raw_edges)) == tuple(sorted(edges)) == chamber.edges,
        "successor must have exactly the predicted fifty-three-edge support",
    )
    phases, capacities = tuple(map(I, fields.phase)), tuple(map(I, fields.capacity))
    conductances = tuple(
        I(_represented(edges[edge].get("weight", 1.0), "successor conductance"))
        for edge in chamber.edges
    )
    actual = _bound_state(fields.epi, phases, capacities, chamber.edges, conductances)
    phase_chamber = _certify_chamber(phases, chamber.edges)
    old_map = edge_map(before)
    new_weights = tuple(
        conductances[chamber.edges.index(edge)] for edge in reset.new_edges
    )
    work = _storage_work(
        fields.epi,
        tuple(map(I, reset.phase_before)),
        phases,
        reset.old_edges,
        reset.new_edges,
        new_weights,
    )
    # These operands are admitted exact representations of primitive values.
    # Subtract before enclosing: separately rounded nonzero energies would lose
    # their exact cancellation and cannot establish a zero-work identity.
    old_form_work = tuple(
        I(
            (
                _represented(
                    edges[left, right].get("weight", 1.0), "old-edge conductance"
                )
                * (fields.epi[left] - fields.epi[right]) ** 2
                - old_weight * (reset.epi[left] - reset.epi[right]) ** 2
            )
            / 2
        )
        for (left, right), old_weight in zip(
            reset.old_edges, reset.old_conductances, strict=True
        )
    )
    actual_total_work = work.total + sum(old_form_work, I(0))
    defects = {
        "phase": tuple(
            (index, _farthest_defect(fields.phase[index], reset.phase_after[index]))
            for index in reset.phase_changed_indices
        ),
        "capacity": tuple(
            (
                index,
                _farthest_defect(fields.capacity[index], reset.capacity_after[index]),
            )
            for index in reset.capacity_changed_indices
        ),
        "new_conductance": tuple(
            (edge, _farthest_defect(weight.lo, ideal))
            for edge, weight, ideal in zip(
                reset.new_edges, new_weights, reset.new_conductances, strict=True
            )
        ),
        "raw_parent_pressure": _farthest_defect(
            _read(raw_after, PARENT, ALIAS_DNFR), reset.raw_pressure_after[PARENT]
        ),
    }
    pressure_defects = tuple(
        _farthest_defect(_read(endpoint, node, ALIAS_DNFR), ideal)
        for node, ideal in zip(reset.nodes, actual.pressure, strict=True)
    )
    checks = {
        "one_canonical_atomic_event": _stage_observation(stage),
        "exact_selected_support": True,
        "forms_unchanged": fields.epi == raw_fields.epi == reset.epi,
        "old_edge_attributes_unchanged": all(
            _same(old_map[edge], raw_edges[edge]) and _same(old_map[edge], edges[edge])
            for edge in reset.old_edges
        ),
        "new_edges_have_only_declared_conductance": all(
            set(raw_edges[edge]) == set(edges[edge]) == {"weight"}
            for edge in reset.new_edges
        ),
        "unchanged_phase_lifts_preserved": all(
            fields.phase[index] == raw_fields.phase[index] == reset.phase_before[index]
            for index in range(len(reset.nodes))
            if index not in reset.phase_changed_indices
        ),
        "unchanged_capacities_preserved": all(
            fields.capacity[index]
            == raw_fields.capacity[index]
            == reset.capacity_before[index]
            for index in range(len(reset.nodes))
            if index not in reset.capacity_changed_indices
        ),
        "changed_outputs_within_original_allowance": all(
            error <= NUMERICAL_ALLOWANCE
            for key in ("phase", "capacity", "new_conductance")
            for _, error in defects[key]
        ),
        "raw_parent_pressure_within_original_allowance": defects["raw_parent_pressure"]
        <= NUMERICAL_ALLOWANCE,
        "raw_other_pressures_preserved": all(
            _read(raw_after, node, ALIAS_DNFR) == reset.raw_pressure_before[index]
            for index, node in enumerate(reset.nodes)
            if node != PARENT
        ),
        "pressure_only_refresh_boundary": _pressure_preservation_signature(
            raw_after, final_schedule=True
        )
        == _pressure_preservation_signature(endpoint, final_schedule=True),
        "refreshed_pressure_independently_bounded": all(
            error <= PRESSURE_ERROR_BUDGET for error in pressure_defects
        ),
        "conditional_full_state_and_held_parameters_inclusion": (
            fields.epi == reset.epi
            and all(
                _contains(bound, value)
                for bounds, values in (
                    (chamber.conditional.phases, fields.phase),
                    (chamber.conditional_capacities, fields.capacity),
                    (
                        chamber.conditional_conductances,
                        tuple(v.lo for v in conductances),
                    ),
                )
                for bound, value in zip(bounds, values, strict=True)
            )
        ),
        "all_original_branch_labels_preserved": phase_chamber.shifts
        == chamber.ideal.shifts,
        "complete_rows_inside_conditional_bounds": all(
            _interval_contained(value, bound)
            for values, bounds in (
                (actual.form_rate, prediction.conditional.form_rate),
                (actual.phase_rate, prediction.conditional.phase_rate),
            )
            for value, bound in zip(values, bounds, strict=True)
        ),
        "old_edge_form_work_zero": all(
            value.lo == value.hi == 0 for value in old_form_work
        ),
    }
    checks.update(_context_checks(before, raw_after, endpoint, reset))
    return {
        "schema": "tnfr.maintained-parent-um-assessment.v1",
        "prediction": encode_result(prediction),
        "actual_fields": encode_result(fields),
        "actual_state": encode_result(actual),
        "actual_chamber": encode_result(phase_chamber),
        "actual_work": encode_result(work),
        "old_edge_form_work": encode_result(old_form_work),
        "actual_total_work": encode_result(actual_total_work),
        "actual_old_radius_squared": encode_result(_old_radius(fields.epi, phases)),
        "changed_output_error_bounds": defects,
        "refreshed_pressure_error_bounds": pressure_defects,
        "numerical_allowance": NUMERICAL_ALLOWANCE,
        "pressure_evaluation_allowance": PRESSURE_ERROR_BUDGET,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "complete_word_executed": False,
        "autonomous_occurrence_derived": False,
        "numerical_u5_observation_available": False,
        "cross_birth_u6_observation_available": False,
    }


def evaluate_um_event(admitted, *, frozen_before, prediction, on_raw=None):
    """Execute exactly one supplied stage after the caller freezes all inputs.

    Return primitive records before any response assessment. The caller must
    persist them and invoke ``check_um_endpoint`` separately. ``on_raw`` receives
    the raw tagged graph before refresh; its durable output survives even if
    the surrounding stage later rolls back. Exceptions are not retry signals.
    """
    _require(type(admitted) is TholSuccessorAdmission, "THOL source admission required")
    if on_raw is not None and not callable(on_raw):
        raise TypeError("raw recording callback must be callable")
    graph = admitted.graph
    before = restore_recorded_graph_state(frozen_before).graph
    _require(
        _signature(graph) == _signature(before),
        "live event source differs from the prospectively frozen before graph",
    )
    rebuilt = _source_prediction(before, prediction)
    captured = []

    def handoff(successor):
        if captured:
            raise ValueError("one UM event permits exactly one pressure handoff")
        captured.append(record_graph_state(successor))
        protected = _pressure_preservation_signature(successor, runtime=True)
        if on_raw is not None:
            on_raw(captured[0])
        refresh_pressure(successor)
        _require(
            _pressure_preservation_signature(successor, runtime=True) == protected,
            "UM handoff changed non-pressure state",
        )

    with patch.object(network_stage, "datetime", _EventClock):
        stage = execute_network_operator_stage(
            graph,
            Coupling(),
            (PARENT,),
            sequence_context=admitted.sequence.step(2),
            compute_delta_nfr=handoff,
        )
    _require(len(captured) == 1, "UM stage omitted its raw pressure handoff")
    return {
        "schema": "tnfr.maintained-parent-um-response.v1",
        "before": frozen_before,
        "raw_after": captured[0],
        "endpoint": record_graph_state(graph),
        "prediction": encode_result(rebuilt),
        "stage": encode_result(stage),
        "retained_source_receipts": admitted.receipts,
    }
