"""One frozen THOL/refresh/SHA/refresh continuation; no flow or word rollback.

The caller freezes source, prediction and producer before evaluating the
retained preparation. Public admission reloads primitives and rebuilds the
prediction. Synthetic controls may exercise the private same-call adapter.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from unittest.mock import patch

from ..operators import network_stage
from ..operators.definitions import SelfOrganization, Silence
from ..operators.word_execution import execute_network_operator_stage
from .local_capacity_history import encode_result
from .maintained_child_history_source import load_child_history_source
from .maintained_child_word_check import (
    EVENT_TIMESTAMP,
    check_child_sha_endpoint,
    check_child_thol_endpoint,
)
from .maintained_parent_sha_source import _same
from .maintained_parent_thol import _signature
from .maintained_parent_thol_execution import refresh_frozen_successor
from .maintained_parent_um_execution import _pressure_preservation_signature
from .retained_graph_state import record_graph_state, restore_recorded_graph_state

ACTOR = "2_sub_0"
CHILD = "2_sub_0_sub_0"
MAXIMUM_EVENTS = 2


class _EventClock:
    @classmethod
    def now(cls, tz=None):
        return datetime.fromisoformat(EVENT_TIMESTAMP)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _prediction(admitted, supplied):
    from .maintained_child_word_prediction import _predict_child_word

    rebuilt = _predict_child_word(admitted)
    _require(
        _same(encode_result(rebuilt), encode_result(supplied)),
        "frozen child word prediction differs from fresh source",
    )
    return rebuilt


def _guarded_refresh(graph):
    protected = _pressure_preservation_signature(graph, runtime=True)
    refresh_frozen_successor(graph)
    _require(
        _pressure_preservation_signature(graph, runtime=True) == protected,
        "child word refresh changed non-pressure state",
    )


def _execute_child_word(admitted, prediction, *, on_record=None):
    """Use one fresh admission; preserve each committed prefix on later failure.

    Raw records are emitted before refresh, including when that stage later
    rolls back. After records identify successful atomic commits. Sink errors
    stop this attempt; their external writes are not rolled back or retried.
    """
    if on_record is not None and not callable(on_record):
        raise TypeError("record sink must be callable")
    original = admitted.graph
    source_signature = _signature(original)
    prediction = _prediction(admitted, prediction)
    _require(_signature(original) == source_signature, "prediction changed source")
    graph = restore_recorded_graph_state(record_graph_state(original)).graph
    _require(
        _signature(graph) == source_signature, "execution clone differs from source"
    )
    captured, stages, assessments, committed = {}, {}, {}, []

    def emit(kind, payload):
        protected = (
            _signature(graph),
            _pressure_preservation_signature(graph, runtime=True),
        )
        if on_record is not None:
            on_record(kind, deepcopy(payload))
        _require(
            protected
            == (
                _signature(graph),
                _pressure_preservation_signature(graph, runtime=True),
            ),
            "record sink changed live child word state",
        )
        _require(
            _signature(original) == source_signature,
            "record sink changed admitted source",
        )

    current = "source"
    try:
        emit(
            "source",
            {
                "endpoint": record_graph_state(graph),
                "source_receipts": admitted.receipts,
                "prediction": encode_result(prediction),
            },
        )
        for glyph, operator, index in (
            ("THOL", SelfOrganization(), 1),
            ("SHA", Silence(), 2),
        ):
            current = glyph
            name = glyph.lower()

            def handoff(live):
                _require(live is graph, "stage callback substituted the live graph")
                _require(name not in captured, f"{glyph} repeated its raw handoff")
                captured[name] = record_graph_state(live)
                emit(
                    name + "_raw",
                    {
                        "endpoint": captured[name],
                        "completed_stage_prefix": tuple(committed),
                    },
                )
                _guarded_refresh(live)

            with patch.object(network_stage, "datetime", _EventClock):
                stage = execute_network_operator_stage(
                    graph,
                    operator,
                    (ACTOR,),
                    sequence_context=admitted.sequence.step(index),
                    compute_delta_nfr=handoff,
                )
            committed.append(glyph)
            stages[name] = encode_result(stage)
            captured[name + "_after"] = record_graph_state(graph)
            emit(
                name + "_after",
                {
                    "endpoint": captured[name + "_after"],
                    "stage": stages[name],
                    "completed_stage_prefix": tuple(committed),
                },
            )
            _require(name in captured, f"{glyph} omitted its raw handoff")
            raw = restore_recorded_graph_state(captured[name]).graph
            before_check = _signature(graph)
            if glyph == "THOL":
                assessment = check_child_thol_endpoint(
                    admitted, raw, graph, prediction=prediction, stage=stage
                )
            else:
                thol = restore_recorded_graph_state(captured["thol_after"]).graph
                assessment = check_child_sha_endpoint(
                    admitted, thol, raw, graph, prediction=prediction, stage=stage
                )
            _require(
                assessment.get("all_checks_passed") is True,
                f"{glyph} primitive endpoint assessment failed",
            )
            _require(
                _signature(graph) == before_check
                and _signature(original) == source_signature,
                f"{glyph} endpoint checker changed persistent state",
            )
            assessments[name] = assessment
        _require(
            _signature(original) == source_signature,
            "child word changed admitted source",
        )
        return {
            "schema": "tnfr.maintained-child-word-response.v1",
            "status": "completed",
            "source_receipts": admitted.receipts,
            "prediction": encode_result(prediction),
            "before": record_graph_state(original),
            "thol_raw": captured["thol"],
            "thol_after": captured["thol_after"],
            "sha_raw": captured["sha"],
            "endpoint": record_graph_state(graph),
            "stages": stages,
            "assessments": assessments,
            "completed_stage_prefix": tuple(committed),
            "complete_word_executed": True,
            "new_births": 1,
            "continuous_steps": 0,
            "new_temporal_samples": 0,
            "autonomous_occurrence_derived": False,
            "future_retention_certified": False,
            "physical_identification": False,
        }
    except BaseException as failure:
        record = {
            "schema": "tnfr.maintained-child-word-failure.v1",
            "failed_at": current,
            "error_type": type(failure).__name__,
            "message": str(failure),
            "endpoint": record_graph_state(graph),
            "completed_stage_prefix": tuple(committed),
            "raw_captures_available": tuple(
                key for key in ("thol", "sha") if key in captured
            ),
            "source_unchanged": _signature(original) == source_signature,
        }
        # A second sink failure is not suppressed: exception chaining retains
        # both failures, and no previously committed stage is undone here.
        emit("failure", record)
        raise


def evaluate_child_word(
    evidence_dir, *, repository_root, frozen_prediction, on_record=None
):
    """Reload once, rebuild/compare the frozen prediction, then execute once.

    The source sink is called before the first event so an instrument can
    additionally require exact agreement with its separately frozen full
    preparation record and source receipts.
    """
    admitted = load_child_history_source(evidence_dir, repository_root=repository_root)
    return _execute_child_word(admitted, frozen_prediction, on_record=on_record)
