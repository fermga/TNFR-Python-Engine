"""Manufactured-state controls for one two-stage continuation, never its source.

Two synthetic nodes, unrelated physical coordinates and fabricated historical
samples exercise the real atomic stages. Source admission is a test seam; no
retained graph, source loader, numerical trajectory or acquisition is used.
"""

from collections import deque
from copy import deepcopy
from dataclasses import dataclass, replace
from fractions import Fraction as Q
from math import pi
from types import SimpleNamespace

import networkx as nx
import pytest

from tnfr.alias import get_attr, set_attr, set_vf
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.constants.canonical import SHA_VF_FACTOR
from tnfr.mathematics._rational_interval import I
from tnfr.operators.definitions import Dissonance, SelfOrganization, Silence
from tnfr.operators.grammar_execution import ValidatedSequence
from tnfr.research import maintained_child_word_check as checker
from tnfr.research import maintained_child_word_execution as owner
from tnfr.research.local_capacity_history import CONTEXT, capture, refresh_pressure
from tnfr.research.maintained_parent_thol import _signature
from tnfr.research.maintained_parent_thol_prediction import (
    NUMERICAL_ALLOWANCE,
    _metabolic_prediction,
    _with_allowance,
)
from tnfr.research.maintained_parent_um_work import _bound_state
from tnfr.research.retained_graph_state import restore_recorded_graph_state

ROOT = "manufactured_root"


@dataclass(frozen=True)
class SyntheticThol:
    actor: str
    child: str
    source_nodes: tuple
    successor_nodes: tuple
    observed_acceleration: Q
    raw_pressure_represented: Q
    child_form_bounds: I
    child_capacity: Q
    child_phase: Q
    child_hierarchy_level: int
    child_hierarchy_path: tuple
    child_record_operator_step: int
    threshold: Q
    metabolic: object
    numerical_allowance: Q = NUMERICAL_ALLOWANCE


@dataclass(frozen=True)
class SyntheticWord:
    thol: SyntheticThol
    structural_time: Q
    actor_capacity_after: Q
    expected_actor_operator_step: int
    before: object
    after: object


def _synthetic_prediction(graph):
    source = capture(graph)
    x, theta, nu = source.epi[1], source.phase[1], source.capacity[1]
    metabolic = _metabolic_prediction(x, theta, (source.epi[0],), (source.phase[0],))
    pressure = get_attr(graph.nodes[owner.ACTOR], ALIAS_DNFR)
    edges = ((0, 1),)
    weights = (I(Q(3, 4)),)
    before = _bound_state(
        source.epi,
        tuple(map(I, source.phase)),
        tuple(map(I, source.capacity)),
        edges,
        weights,
    )
    capacity_after = Q(float(nu * Q(SHA_VF_FACTOR)))
    after = _bound_state(
        source.epi,
        tuple(map(I, source.phase)),
        (I(source.capacity[0]), I(capacity_after)),
        edges,
        weights,
    )
    return SyntheticWord(
        SyntheticThol(
            owner.ACTOR,
            owner.CHILD,
            source.nodes,
            (*source.nodes, owner.CHILD),
            Q(1, 8),
            Q(pressure + (1 / (4 * pi)) * (1 / 8)),
            _with_allowance(metabolic.child_form),
            Q(float(nu * Q(0.95))),
            theta,
            2,
            (ROOT, owner.ACTOR),
            2,
            Q(0.1),
            metabolic,
        ),
        Q(12),
        capacity_after,
        3,
        before,
        after,
    )


def _admitted():
    graph = nx.Graph(
        _t=12.0,
        GAMMA={"type": "none"},
        GRAMMAR_REJECTION_MODE="raise",
        VALIDATE_OPERATOR_PRECONDITIONS=True,
        EDGE_AWARE_ENABLED=False,
        EPI_MIN=-1.0,
        EPI_MAX=1.0,
        hierarchy={ROOT: [owner.ACTOR]},
    )
    for node, x, theta, capacity in (
        (ROOT, 5 / 8, 1 / 4, 2.5),
        (owner.ACTOR, 3 / 8, 0.0, 7.3),
    ):
        graph.add_node(
            node,
            **{
                ALIAS_EPI[0]: x,
                ALIAS_THETA[0]: theta,
                ALIAS_VF[0]: capacity,
                ALIAS_DNFR[0]: 0.0,
                "epi_history": [x],
                "glyph_history": [],
            },
        )
    graph.nodes[ROOT]["sub_nodes"] = [owner.ACTOR]
    graph.nodes[owner.ACTOR].update(
        parent_node=ROOT,
        hierarchy_level=1,
        _bifurcation_level=1,
        _hierarchy_path=[ROOT],
        glyph_history=deque(["OZ"], maxlen=7),
        _operator_step=1,
        _grammar_u2_debt=1,
        _grammar_prior_coherence=False,
        source_glyph="OZ",
        epi_time_history=deque([(10.0, 1 / 4), (11.0, 1 / 4), (12.0, 3 / 8)], maxlen=4),
    )
    graph.add_edge(ROOT, owner.ACTOR, weight=3 / 4)
    set_vf(graph, owner.ACTOR, 7.3)
    graph._last_operator_applied = "dissonance"
    refresh_pressure(graph)
    return SimpleNamespace(
        graph=graph,
        receipts=({"synthetic": True},),
        sequence=ValidatedSequence(
            (Dissonance(), SelfOrganization(), Silence()), context=CONTEXT
        ),
    )


@pytest.fixture(autouse=True)
def synthetic_only(monkeypatch):
    from tnfr.dynamics import integrators
    from tnfr.research import maintained_child_history_acquisition as acquisition
    from tnfr.research import maintained_child_word_prediction as predictor

    def forbidden(*args, **kwargs):
        pytest.fail("synthetic word controls may not load retained state or run flow")

    monkeypatch.setattr(owner, "load_child_history_source", forbidden)
    monkeypatch.setattr(predictor, "load_child_history_source", forbidden)
    monkeypatch.setattr(
        predictor,
        "_predict_child_word",
        lambda admitted: _synthetic_prediction(admitted.graph),
    )
    monkeypatch.setattr(integrators, "rk4_coupled_step", forbidden)
    monkeypatch.setattr(acquisition, "evaluate_child_history", forbidden)


def _run(admitted=None, *, sink=None):
    admitted = admitted or _admitted()
    records = []

    def on_record(kind, record):
        records.append((kind, deepcopy(record)))
        if sink is not None:
            sink(kind, record)

    response = owner._execute_child_word(
        admitted, _synthetic_prediction(admitted.graph), on_record=on_record
    )
    return response, records, admitted


def test_complete_word_uses_real_separate_stages_and_preserves_original_source():
    admitted = _admitted()
    original = _signature(admitted.graph)
    result, records, _ = _run(admitted)
    assert _signature(admitted.graph) == original
    assert [kind for kind, _ in records] == [
        "source",
        "thol_raw",
        "thol_after",
        "sha_raw",
        "sha_after",
    ]
    assert result["completed_stage_prefix"] == ("THOL", "SHA")
    assert result["complete_word_executed"] and result["new_births"] == 1
    assert result["continuous_steps"] == result["new_temporal_samples"] == 0
    assert all(
        assessment["all_checks_passed"] for assessment in result["assessments"].values()
    )
    assert records[0][1]["source_receipts"] == admitted.receipts
    source = restore_recorded_graph_state(records[0][1]["endpoint"]).graph
    assert _signature(source) == original
    endpoint = restore_recorded_graph_state(result["endpoint"]).graph
    actor = endpoint.nodes[owner.ACTOR]
    assert tuple(actor["glyph_history"]) == ("OZ", "THOL", "SHA")
    assert actor["_operator_step"] == 3 and actor["_grammar_u2_debt"] == 0
    assert (
        actor["epi_time_history"]
        == admitted.graph.nodes[owner.ACTOR]["epi_time_history"]
    )
    assert actor["latency_start_time"] == owner.EVENT_TIMESTAMP
    assert actor["preserved_epi"] == 3 / 8 and actor["latent"] is True
    assert endpoint.graph["_t"] == 12.0
    assert endpoint.degree(owner.CHILD) == 0
    assert tuple(endpoint[owner.ACTOR]) == tuple(admitted.graph[owner.ACTOR])
    assert get_attr(endpoint.nodes[owner.CHILD], ALIAS_VF) > get_attr(actor, ALIAS_VF)


def test_public_entry_loads_once_and_passes_frozen_prediction(monkeypatch):
    admitted = _admitted()
    calls = []

    def load(*args, **kwargs):
        calls.append((args, kwargs))
        return admitted

    monkeypatch.setattr(owner, "load_child_history_source", load)
    result = owner.evaluate_child_word(
        "synthetic",
        repository_root="root",
        frozen_prediction=_synthetic_prediction(admitted.graph),
    )
    assert len(calls) == 1 and result["status"] == "completed"


def test_changed_frozen_prediction_rejects_before_any_event(monkeypatch):
    admitted = _admitted()
    prediction = replace(
        _synthetic_prediction(admitted.graph), actor_capacity_after=Q(99)
    )
    monkeypatch.setattr(
        owner,
        "execute_network_operator_stage",
        lambda *a, **k: pytest.fail("must reject before dispatch"),
    )
    with pytest.raises(ValueError, match="frozen child word prediction"):
        owner._execute_child_word(admitted, prediction)


def test_detached_sink_payload_cannot_mutate_future_captures():
    def sink(kind, record):
        record.clear()

    result, _, _ = _run(sink=sink)
    assert result["complete_word_executed"]
    assert restore_recorded_graph_state(result["thol_raw"]).graph.nodes[owner.ACTOR][
        "sub_nodes"
    ] == [owner.CHILD]


@pytest.mark.parametrize("phase", ("THOL", "SHA"))
def test_refresh_violation_rolls_back_only_current_stage_and_retains_raw(
    monkeypatch, phase
):
    admitted = _admitted()
    records = []
    refresh = owner.refresh_frozen_successor

    def bad_refresh(graph):
        refresh(graph)
        glyph = tuple(graph.nodes[owner.ACTOR]["glyph_history"])[-1]
        if glyph == phase:
            graph.graph["_t"] += 1

    monkeypatch.setattr(owner, "refresh_frozen_successor", bad_refresh)
    with pytest.raises(ValueError, match="refresh changed non-pressure"):
        owner._execute_child_word(
            admitted,
            _synthetic_prediction(admitted.graph),
            on_record=lambda kind, record: records.append((kind, record)),
        )
    failure = records[-1][1]
    assert records[-1][0] == "failure"
    expected_prefix = () if phase == "THOL" else ("THOL",)
    assert failure["completed_stage_prefix"] == expected_prefix
    endpoint = restore_recorded_graph_state(failure["endpoint"]).graph
    assert endpoint.graph["_t"] == 12
    assert (owner.CHILD in endpoint) is (phase == "SHA")
    assert (
        tuple(endpoint.nodes[owner.ACTOR]["glyph_history"]) == ("OZ",) + expected_prefix
    )
    assert phase.lower() in failure["raw_captures_available"]


def test_failed_thol_assessment_does_not_proceed_to_sha(monkeypatch):
    records = []
    monkeypatch.setattr(
        owner, "check_child_thol_endpoint", lambda *a, **k: {"all_checks_passed": False}
    )
    admitted = _admitted()
    with pytest.raises(ValueError, match="THOL primitive endpoint assessment failed"):
        owner._execute_child_word(
            admitted,
            _synthetic_prediction(admitted.graph),
            on_record=lambda kind, record: records.append((kind, record)),
        )
    assert [kind for kind, _ in records] == [
        "source",
        "thol_raw",
        "thol_after",
        "failure",
    ]
    assert records[-1][1]["completed_stage_prefix"] == ("THOL",)
    assert owner.CHILD in restore_recorded_graph_state(records[-1][1]["endpoint"]).graph


def test_source_sink_rejection_precedes_dispatch_and_is_not_suppressed(monkeypatch):
    records = []
    monkeypatch.setattr(
        owner,
        "execute_network_operator_stage",
        lambda *a, **k: pytest.fail("no event may follow source rejection"),
    )

    def sink(kind, record):
        records.append(kind)
        if kind == "source":
            raise RuntimeError("frozen preparation mismatch")

    with pytest.raises(RuntimeError, match="frozen preparation mismatch"):
        _run(sink=sink)
    assert records == ["source", "failure"]


def test_source_mutating_sink_is_rejected_before_dispatch(monkeypatch):
    admitted = _admitted()
    monkeypatch.setattr(
        owner,
        "execute_network_operator_stage",
        lambda *a, **k: pytest.fail("no event may follow source mutation"),
    )

    def sink(kind, record):
        if kind == "source":
            admitted.graph.graph["_t"] += 1

    with pytest.raises(ValueError, match="record sink changed admitted source"):
        _run(admitted, sink=sink)


@pytest.mark.parametrize("stage_name", ("thol", "sha"))
@pytest.mark.parametrize(
    "defect", ("pressure", "capacity", "history", "hierarchy", "clock", "instance")
)
def test_checker_rejects_corrupted_primitive_endpoints(stage_name, defect):
    response, _, admitted = _run()
    prediction = _synthetic_prediction(admitted.graph)
    raw = restore_recorded_graph_state(response[stage_name + "_raw"]).graph
    endpoint = restore_recorded_graph_state(
        response["thol_after" if stage_name == "thol" else "endpoint"]
    ).graph
    if defect == "pressure":
        set_attr(raw.nodes[owner.ACTOR], ALIAS_DNFR, 99.0)
    elif defect == "capacity":
        set_attr(raw.nodes[owner.ACTOR], ALIAS_VF, 99.0)
    elif defect == "history":
        raw.nodes[owner.CHILD]["epi_time_history"] = [(12.0, 0.0)]
    elif defect == "hierarchy":
        raw.nodes[owner.CHILD]["parent_node"] = ROOT
    elif defect == "clock":
        raw.graph["_t"] += 1
    else:
        raw._last_operator_applied = "forged"
    with pytest.raises(ValueError):
        if stage_name == "thol":
            checker.check_child_thol_endpoint(
                admitted,
                raw,
                endpoint,
                prediction=prediction,
                stage=response["stages"]["thol"],
            )
        else:
            thol = restore_recorded_graph_state(response["thol_after"]).graph
            checker.check_child_sha_endpoint(
                admitted,
                thol,
                raw,
                endpoint,
                prediction=prediction,
                stage=response["stages"]["sha"],
            )


def test_checker_rejects_unbudgeted_metabolic_output():
    response, _, admitted = _run()
    raw = restore_recorded_graph_state(response["thol_raw"]).graph
    endpoint = restore_recorded_graph_state(response["thol_after"]).graph
    raw.nodes[owner.ACTOR]["sub_epis"][0]["network_signals"]["phase_variance"] = True
    with pytest.raises((TypeError, ValueError)):
        checker.check_child_thol_endpoint(
            admitted,
            raw,
            endpoint,
            prediction=_synthetic_prediction(admitted.graph),
            stage=response["stages"]["thol"],
        )


def test_noncallable_sink_rejects_before_prediction(monkeypatch):
    monkeypatch.setattr(
        owner, "_prediction", lambda *a: pytest.fail("sink must reject first")
    )
    with pytest.raises(TypeError, match="sink must be callable"):
        owner._execute_child_word(_admitted(), None, on_record=1)


@pytest.mark.parametrize("defect", ("missing", "repeated"))
def test_callback_cardinality_stops_before_any_later_stage(monkeypatch, defect):
    dispatch = owner.execute_network_operator_stage
    records = []

    def bad_dispatch(graph, operator, targets, **kwargs):
        assert operator.glyph.value == "THOL"
        callback = kwargs["compute_delta_nfr"]
        if defect == "missing":
            kwargs["compute_delta_nfr"] = None
        else:

            def twice(live):
                callback(live)
                callback(live)

            kwargs["compute_delta_nfr"] = twice
        return dispatch(graph, operator, targets, **kwargs)

    monkeypatch.setattr(owner, "execute_network_operator_stage", bad_dispatch)
    admitted = _admitted()
    with pytest.raises(ValueError, match="omitted|repeated"):
        owner._execute_child_word(
            admitted,
            _synthetic_prediction(admitted.graph),
            on_record=lambda kind, record: records.append((kind, record)),
        )
    assert not any(kind.startswith("sha") for kind, _ in records)
    failure = records[-1][1]
    # Missing callback is detected after the returned stage has committed;
    # repeated callback raises inside that stage and its own rollback applies.
    assert failure["completed_stage_prefix"] == (
        ("THOL",) if defect == "missing" else ()
    )


def test_raw_sink_failure_preserves_raw_and_rolls_back_only_sha():
    records = []

    def sink(kind, record):
        records.append((kind, deepcopy(record)))
        if kind == "sha_raw":
            raise RuntimeError("durable SHA sink rejected")

    with pytest.raises(RuntimeError, match="durable SHA sink rejected"):
        _run(sink=sink)
    assert [kind for kind, _ in records] == [
        "source",
        "thol_raw",
        "thol_after",
        "sha_raw",
        "failure",
    ]
    assert records[-1][1]["completed_stage_prefix"] == ("THOL",)
    endpoint = restore_recorded_graph_state(records[-1][1]["endpoint"]).graph
    assert owner.CHILD in endpoint and "latent" not in endpoint.nodes[owner.ACTOR]
    assert records[-1][1]["raw_captures_available"] == ("thol", "sha")


def test_sink_failure_while_reporting_failure_remains_chained():
    def sink(kind, record):
        if kind == "source":
            raise RuntimeError("source refusal")
        if kind == "failure":
            raise OSError("failure record unavailable")

    with pytest.raises(OSError, match="failure record unavailable") as caught:
        _run(sink=sink)
    assert isinstance(caught.value.__context__, RuntimeError)
    assert str(caught.value.__context__) == "source refusal"


def test_checker_rebuilds_prediction_instead_of_trusting_cached_state():
    response, _, admitted = _run()
    prediction = _synthetic_prediction(admitted.graph)
    forged = replace(prediction, actor_capacity_after=Q(99))
    raw = restore_recorded_graph_state(response["thol_raw"]).graph
    endpoint = restore_recorded_graph_state(response["thol_after"]).graph
    with pytest.raises(ValueError, match="checker prediction differs"):
        checker.check_child_thol_endpoint(
            admitted, raw, endpoint, prediction=forged, stage=response["stages"]["thol"]
        )
