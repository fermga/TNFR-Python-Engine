"""One supplied SHA closure and detached admission of its complete successor.

The caller freezes source, prediction and budget before the sole atomic event.
The checker rebuilds from primitives; latency does not install a stopped flow.
No post-SHA trajectory or inherited UM maintenance certificate is evaluated.
"""

from __future__ import annotations

from datetime import datetime, timezone
from fractions import Fraction as Q
from unittest.mock import patch

from ..constants.aliases import ALIAS_DNFR, ALIAS_SOURCE_GLYPH, ALIAS_VF
from ..mathematics._rational_interval import I
from ..operators import network_stage
from ..operators.definitions import Silence
from ..operators.grammar_debt import node_debt
from ..operators.nodal_equation import observe_structural_acceleration
from ..operators.stage_contracts import stage_schedule_metadata
from ..operators.word_execution import execute_network_operator_stage
from ..physics.relational_sine_support import bound_relational_sine_support_exchange
from ..utils.cache import GRAPH_RUNTIME_CACHE_KEYS
from .local_capacity_history import (
    MODEL,
    PARENT,
    PRESSURE_ERROR_BUDGET,
    encode_result,
    refresh_pressure,
)
from .maintained_parent_sha import predict_parent_sha
from .maintained_parent_sha_source import UmSuccessorAdmission
from .maintained_parent_thol import _signature
from .maintained_parent_um import CHILD, NUMERICAL_ALLOWANCE
from .maintained_parent_um_execution import (
    _farthest_defect,
    _maximum_matches,
    _pressure_preservation_signature,
    _read,
    _require,
    _same,
    _without,
    _write_alias,
)
from .maintained_parent_um_work import _bound_state
from .retained_graph_state import record_graph_state, restore_recorded_graph_state

EVENT_TIMESTAMP = "2030-01-02T03:04:08+00:00"
_STAGE_KEYS = frozenset(
    {network_stage.STAGE_SCHEDULE_KEY, network_stage.STAGE_CONTRACT_KEY}
)
_LATENCY_KEYS = frozenset(
    {
        "latent",
        "latency_start_time",
        "preserved_epi",
        "silence_duration",
        "was_initial_on_silence",
    }
)
_LIFECYCLE_KEYS = frozenset(
    {"glyph_history", "_operator_step", "_grammar_u2_debt", "_grammar_prior_coherence"}
)


class _EventClock:
    @classmethod
    def now(cls, tz=None):
        return datetime(2030, 1, 2, 3, 4, 8, tzinfo=timezone.utc)


def _prediction(admitted, supplied):
    rebuilt = predict_parent_sha(admitted)
    if supplied is not None:
        _require(
            _same(encode_result(rebuilt), encode_result(supplied)),
            "supplied SHA prediction differs from source",
        )
    return rebuilt


def check_sha_endpoint(admitted, raw_after, endpoint, *, prediction=None, stage=None):
    """Assess retained primitive records without executing or refreshing them."""
    prediction = _prediction(admitted, prediction)
    before = admitted.graph
    graphs = (raw_after, endpoint)
    fields = tuple(
        bound_relational_sine_support_exchange(graph, reference_model=MODEL)
        for graph in graphs
    )
    raw, actual = fields
    _require(
        raw.nodes == actual.nodes == prediction.nodes,
        "SHA endpoint node order differs from admitted source",
    )
    positions = {node: index for index, node in enumerate(actual.nodes)}
    actual_edge_map = {
        tuple(sorted((positions[left], positions[right]))): weight
        for (left, right), weight in zip(actual.edges, actual.conductances, strict=True)
    }
    actual_edges = tuple(sorted(actual_edge_map))
    stage = encode_result(stage)
    expected_edges = tuple(before.edges(data=True))
    unchanged_storage_arguments = all(
        field.nodes == prediction.nodes
        and field.epi == prediction.epi
        and field.phase == prediction.phases
        and _same(tuple(graph.edges(data=True)), expected_edges)
        for graph, field in zip(graphs, fields, strict=True)
    )
    capacities = tuple(map(I, actual.capacity))
    state = _bound_state(
        actual.epi,
        tuple(map(I, actual.phase)),
        capacities,
        actual_edges,
        tuple(I(actual_edge_map[edge]) for edge in actual_edges),
    )
    pressure_errors = tuple(
        _farthest_defect(_read(endpoint, node, ALIAS_DNFR), bound)
        for node, bound in zip(prediction.nodes, state.pressure, strict=True)
    )
    excluded_graph = (
        GRAPH_RUNTIME_CACHE_KEYS
        | _STAGE_KEYS
        | {"_vfmax", "_vfmax_node", "_dnfrmax", "_dnfrmax_node"}
    )
    graph_context = all(
        _same(
            _without(before.graph, excluded_graph),
            _without(graph.graph, excluded_graph),
        )
        for graph in graphs
    )
    nodal_context = True
    for node in before:
        allowed = set()
        if node == PARENT:
            allowed = (
                _LATENCY_KEYS
                | _LIFECYCLE_KEYS
                | {
                    _write_alias(before.nodes[node], ALIAS_VF),
                    _write_alias(before.nodes[node], ALIAS_SOURCE_GLYPH),
                }
            )
        nodal_context &= _same(
            _without(before.nodes[node], allowed),
            _without(raw_after.nodes[node], allowed),
        )
    parent_before = before.nodes[PARENT]
    expected_latency = {
        "latent": True,
        "latency_start_time": EVENT_TIMESTAMP,
        "preserved_epi": float(prediction.epi[prediction.parent_index]),
        "silence_duration": 0.0,
        "was_initial_on_silence": abs(float(prediction.epi[prediction.parent_index]))
        < 1e-6,
    }
    expected_schedule = {
        "operator": "silence",
        "glyph": "SHA",
        "schedule": "two_phase_jacobi",
        "nodes_processed": 1,
    }
    lifecycle = all(
        tuple(graph.nodes[PARENT].get("glyph_history", ()))
        == prediction.expected_parent_history
        and type(graph.nodes[PARENT].get("glyph_history"))
        is type(parent_before["glyph_history"])
        and getattr(graph.nodes[PARENT]["glyph_history"], "maxlen", None)
        == getattr(parent_before["glyph_history"], "maxlen", None)
        and type(graph.nodes[PARENT].get("_operator_step")) is int
        and graph.nodes[PARENT]["_operator_step"]
        == prediction.expected_parent_operator_step
        and type(graph.nodes[PARENT].get("_grammar_u2_debt")) is int
        and graph.nodes[PARENT]["_grammar_u2_debt"]
        == node_debt(graph.nodes[PARENT])
        == 0
        and graph.nodes[PARENT].get("_grammar_prior_coherence") is False
        and graph.nodes[PARENT].get(_write_alias(parent_before, ALIAS_SOURCE_GLYPH))
        == "SHA"
        and getattr(graph, "_last_operator_applied", None) == "silence"
        for graph in graphs
    )
    capacity_error = abs(
        actual.capacity[prediction.parent_index] - prediction.ideal_parent_capacity
    )
    checks = {
        "one_canonical_atomic_event": type(stage) is dict
        and all(
            _same(stage.get(key), value) for key, value in expected_schedule.items()
        ),
        "storage_arguments_unchanged": unchanged_storage_arguments,
        "node_and_neighbor_order_preserved": all(
            tuple(graph) == tuple(before)
            and all(tuple(graph[node]) == tuple(before[node]) for node in before)
            for graph in graphs
        ),
        "exact_represented_capacity_reset": raw.capacity
        == actual.capacity
        == prediction.represented_capacity_after,
        "capacity_error_within_frozen_allowance": capacity_error <= NUMERICAL_ALLOWANCE,
        "raw_pressure_preserved": all(
            _read(raw_after, node, ALIAS_DNFR) == prediction.raw_pressure_before[index]
            for index, node in enumerate(prediction.nodes)
        ),
        "pressure_only_refresh_boundary": _pressure_preservation_signature(
            raw_after, final_schedule=True
        )
        == _pressure_preservation_signature(endpoint, final_schedule=True),
        "refreshed_pressure_independently_bounded": all(
            error <= PRESSURE_ERROR_BUDGET for error in pressure_errors
        ),
        "complete_rows_and_means_match_prediction": _same(
            encode_result(state), encode_result(prediction.represented)
        ),
        "declared_graph_context_preserved": graph_context,
        "declared_nodal_context_preserved": bool(nodal_context),
        "exact_latency_metadata": all(
            _same(
                {key: graph.nodes[PARENT].get(key) for key in expected_latency},
                expected_latency,
            )
            for graph in graphs
        ),
        "actual_complete_parent_word": lifecycle,
        "parent_temporal_history_preserved": all(
            _same(
                graph.nodes[PARENT].get("epi_time_history"),
                parent_before.get("epi_time_history"),
            )
            for graph in graphs
        ),
        "child_temporal_baseline_unavailable": all(
            "epi_time_history" not in graph.nodes[CHILD]
            and not observe_structural_acceleration(graph, CHILD).available
            for graph in graphs
        ),
        "structural_clock_unchanged": all(
            _same(graph.graph.get("_t"), before.graph.get("_t")) for graph in graphs
        ),
        "raw_schedule_precedes_final_metadata": all(
            _same(raw_after.graph.get(key), before.graph.get(key))
            for key in _STAGE_KEYS
        ),
        "final_schedule_metadata": _same(
            endpoint.graph.get(network_stage.STAGE_SCHEDULE_KEY), expected_schedule
        )
        and _same(
            endpoint.graph.get(network_stage.STAGE_CONTRACT_KEY),
            stage_schedule_metadata("silence", observed_schedule="two_phase_jacobi"),
        ),
        "declared_alias_maxima_valid": all(
            _maximum_matches(graph, aliases, key)
            for graph in graphs
            for aliases, key in ((ALIAS_VF, "_vfmax"), (ALIAS_DNFR, "_dnfrmax"))
        ),
    }
    return {
        "schema": "tnfr.maintained-parent-sha-assessment.v1",
        "prediction": encode_result(prediction),
        "actual_fields": encode_result(actual),
        "actual_state": encode_result(state),
        "capacity_absolute_error": capacity_error,
        "numerical_allowance": NUMERICAL_ALLOWANCE,
        "pressure_evaluation_allowance": PRESSURE_ERROR_BUDGET,
        "refreshed_pressure_error_bounds": pressure_errors,
        "event_storage_jump": Q(0) if unchanged_storage_arguments else None,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "complete_word_executed": all(checks.values()),
        "um_motion_certificate_transferred": False,
        "autonomous_occurrence_derived": False,
        "numerical_u5_observation_available": False,
        "cross_birth_u6_observation_available": False,
    }


def evaluate_sha_event(admitted, *, frozen_before, prediction, on_raw=None):
    """Execute the sole reserved SHA stage; persist the return before assessment."""
    _require(type(admitted) is UmSuccessorAdmission, "UM successor admission required")
    if on_raw is not None and not callable(on_raw):
        raise TypeError("raw recording callback must be callable")
    graph = admitted.graph
    before = restore_recorded_graph_state(frozen_before).graph
    _require(
        _signature(graph) == _signature(before),
        "live source differs from frozen SHA before state",
    )
    rebuilt = _prediction(admitted, prediction)
    captured = []

    def handoff(successor):
        if captured:
            raise ValueError("one SHA event permits exactly one pressure handoff")
        captured.append(record_graph_state(successor))
        protected = _pressure_preservation_signature(successor, runtime=True)
        if on_raw is not None:
            on_raw(captured[0])
        refresh_pressure(successor)
        _require(
            _pressure_preservation_signature(successor, runtime=True) == protected,
            "SHA handoff changed non-pressure state",
        )

    with patch.object(network_stage, "datetime", _EventClock):
        stage = execute_network_operator_stage(
            graph,
            Silence(),
            (PARENT,),
            sequence_context=admitted.sequence.step(3),
            compute_delta_nfr=handoff,
        )
    _require(len(captured) == 1, "SHA stage omitted its raw pressure handoff")
    return {
        "schema": "tnfr.maintained-parent-sha-response.v1",
        "before": frozen_before,
        "raw_after": captured[0],
        "endpoint": record_graph_state(graph),
        "prediction": encode_result(rebuilt),
        "stage": encode_result(stage),
        "retained_source_receipts": admitted.receipts,
    }
