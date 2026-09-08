"""Contract tests for the separately invoked delayed REMESH EPI map."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import FrozenInstanceError

import networkx as nx
import pytest

from tnfr.errors import TNFRValueError
from tnfr.operators import (
    DelayedRemeshResult,
    apply_network_remesh,
    plan_network_remesh,
)
from tnfr.operators import remesh as remesh_module
from tnfr.utils import CallbackEvent, callback_manager


def _graph(
    *,
    current: tuple[float, float] = (0.5, 0.2),
    past: tuple[float, float] = (0.0, 0.1),
    epi_min: float = -10.0,
    epi_max: float = 10.0,
    log_events: bool = False,
) -> nx.Graph:
    graph = nx.path_graph(2)
    graph.graph.update(
        REMESH_TAU_GLOBAL=1,
        REMESH_TAU_LOCAL=1,
        REMESH_ALPHA=0.5,
        REMESH_ALPHA_HARD=True,
        REMESH_LOG_EVENTS=log_events,
        EPI_MIN=epi_min,
        EPI_MAX=epi_max,
        CLIP_MODE="hard",
    )
    for node, value in enumerate(current):
        graph.nodes[node]["EPI"] = value
    graph.graph["_epi_hist"] = deque(
        [
            {node: value for node, value in enumerate(past)},
            {node: value for node, value in enumerate(current)},
        ],
        maxlen=8,
    )
    return graph


def _state(graph: nx.Graph) -> tuple[object, ...]:
    return (
        tuple(graph.nodes),
        deepcopy(dict(graph.nodes(data=True))),
        tuple(
            (left, right, dict(data))
            for left, right, data in graph.edges(data=True)
        ),
        deepcopy(graph.graph),
    )


def test_plan_and_result_are_immutable_and_noop_is_explicit() -> None:
    graph = _graph()
    graph.graph["_epi_hist"] = deque(
        [{0: 0.5, 1: 0.2}],
        maxlen=8,
    )
    before = _state(graph)

    plan = plan_network_remesh(graph)
    result = apply_network_remesh(graph)

    assert plan.status == "insufficient_history"
    assert plan.history_length == 1
    assert plan.required_history_length == 2
    assert plan.proposals == ()
    assert not plan.applied
    assert isinstance(result, DelayedRemeshResult)
    assert result.status == "insufficient_history"
    assert not result.applied
    assert result.metadata == {}
    assert _state(graph) == before
    with pytest.raises(FrozenInstanceError):
        result.status = "applied"  # type: ignore[misc]


@pytest.mark.parametrize(
    "history",
    [
        {0: {0: 0.0, 1: 0.0}},
        iter(({0: 0.0, 1: 0.0}, {0: 0.5, 1: 0.2})),
        "not-a-history",
    ],
)
def test_outer_history_must_be_replayable_and_indexed(history: object) -> None:
    graph = _graph()
    graph.graph["_epi_hist"] = history
    nodes_before = deepcopy(dict(graph.nodes(data=True)))
    metadata_before = deepcopy(
        {
            key: value
            for key, value in graph.graph.items()
            if key != "_epi_hist"
        }
    )

    with pytest.raises(TNFRValueError, match="replayable indexed history"):
        apply_network_remesh(graph)

    assert dict(graph.nodes(data=True)) == nodes_before
    assert graph.graph["_epi_hist"] is history
    assert {
        key: value
        for key, value in graph.graph.items()
        if key != "_epi_hist"
    } == metadata_before


@pytest.mark.parametrize(
    ("snapshot", "message"),
    [
        ((0.0, 0.1), "node-to-EPI mapping"),
        ({0: 0.0}, "current graph node support"),
        ({0: 0.0, 1: 0.1, 2: 0.2}, "current graph node support"),
    ],
)
def test_selected_delayed_snapshot_requires_exact_node_support(
    snapshot: object,
    message: str,
) -> None:
    graph = _graph()
    graph.graph["_epi_hist"][0] = snapshot
    before = _state(graph)

    with pytest.raises(TNFRValueError, match=message):
        apply_network_remesh(graph)

    assert _state(graph) == before


@pytest.mark.parametrize("clip_mode", ["unknown", 1, None])
def test_clip_mode_is_not_silently_rewritten(clip_mode: object) -> None:
    graph = _graph()
    graph.graph["CLIP_MODE"] = clip_mode
    before = _state(graph)

    with pytest.raises(TNFRValueError, match="CLIP_MODE"):
        apply_network_remesh(graph)

    assert _state(graph) == before


def test_commit_failure_restores_every_earlier_node_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _graph()
    before = _state(graph)
    original = remesh_module.set_attr
    calls = 0

    def fail_second(mapping, aliases, value):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("commit failed")
        return original(mapping, aliases, value)

    monkeypatch.setattr(remesh_module, "set_attr", fail_second)

    with pytest.raises(RuntimeError, match="commit failed"):
        apply_network_remesh(graph)

    assert _state(graph) == before


def test_telemetry_failure_restores_epi_metadata_and_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _graph(log_events=True)
    graph.graph["history"] = {"C_steps": [0.8]}
    before = _state(graph)

    def fail_log(target, meta):
        target.graph["_REMESH_META"] = {"corrupt": True}
        target.graph["history"]["remesh_events"] = [dict(meta)]
        raise RuntimeError("telemetry failed")

    monkeypatch.setattr(remesh_module, "_log_remesh_event", fail_log)

    with pytest.raises(RuntimeError, match="telemetry failed"):
        apply_network_remesh(graph)

    assert _state(graph) == before


def test_strict_callback_failure_restores_graph_owned_mutations() -> None:
    graph = _graph(log_events=True)
    graph.graph["CALLBACKS_STRICT"] = True
    graph.graph["history"] = {"C_steps": [0.8]}

    def fail_callback(target, _context):
        target.nodes[0]["EPI"] = 99.0
        target.add_node("leaked")
        target.graph["history"]["callback_leak"] = [True]
        raise RuntimeError("callback failed")

    callback_manager.register_callback(
        graph,
        CallbackEvent.ON_REMESH,
        fail_callback,
        name="delayed-remesh-failure",
    )
    before = _state(graph)

    with pytest.raises(RuntimeError, match="callback failed"):
        apply_network_remesh(graph)

    assert _state(graph) == before


def test_successful_callback_cannot_replace_contracted_epi() -> None:
    graph = _graph()
    graph.graph["CALLBACKS_STRICT"] = True

    def mutate_epi(target, _context):
        target.nodes[0]["EPI"] = 99.0

    callback_manager.register_callback(
        graph,
        CallbackEvent.ON_REMESH,
        mutate_epi,
        name="invalid-remesh-mutation",
    )
    before = _state(graph)

    with pytest.raises(TNFRValueError, match="committed EPI"):
        apply_network_remesh(graph)

    assert _state(graph) == before


def test_opt_in_evidence_separates_affine_mean_and_disagreement_claims() -> None:
    graph = _graph(current=(1.0, -1.0), past=(0.5, 0.5))

    result = apply_network_remesh(
        graph,
        include_stability_evidence=True,
        metric_weights={0: 1.0, 1: 3.0},
    )
    evidence = result.evidence

    assert result.applied
    assert evidence is not None
    assert evidence.metric_weights == (1.0, 3.0)
    assert evidence.coefficient_partition_exact
    assert evidence.beta == pytest.approx(0.25)
    assert evidence.gamma == pytest.approx(0.25)
    assert evidence.delta == pytest.approx(0.5)
    assert evidence.fixed_history_raw_map_preserves_consensus_subspace
    assert evidence.raw_global_disagreement_gain_certified
    assert evidence.fixed_history_raw_global_disagreement_gain_upper_bound == (
        pytest.approx(0.25**2)
    )
    assert evidence.raw_convex_disagreement_bound_certified
    assert not evidence.observed_raw_weighted_mean_preserved
    assert evidence.observed_raw_disagreement_nonincreasing
    assert "No field proves repeated-map stability" in evidence.scope


def test_nonuniform_delayed_offset_abstains_from_global_current_state_gain() -> None:
    graph = _graph(current=(0.0, 0.0), past=(1.0, -1.0))

    result = apply_network_remesh(
        graph,
        include_stability_evidence=True,
    )
    evidence = result.evidence

    assert evidence is not None
    assert not evidence.fixed_history_raw_map_preserves_consensus_subspace
    assert not evidence.raw_global_disagreement_gain_certified
    assert evidence.fixed_history_raw_global_disagreement_gain_upper_bound is None
    assert evidence.observed_current_disagreement_energy == 0.0
    assert evidence.observed_raw_disagreement_energy > 0.0
    assert not evidence.observed_raw_disagreement_nonincreasing


def test_clipping_is_reported_separately_from_the_raw_recurrence() -> None:
    graph = _graph(
        current=(1.0, -1.0),
        past=(1.0, -1.0),
        epi_min=-0.25,
        epi_max=0.25,
    )

    result = apply_network_remesh(
        graph,
        include_stability_evidence=True,
    )
    evidence = result.evidence

    assert evidence is not None
    assert tuple(proposal.raw_epi for proposal in result.proposals) == (1.0, -1.0)
    assert tuple(proposal.bounded_epi for proposal in result.proposals) == (
        0.25,
        -0.25,
    )
    assert evidence.clipped_nodes == (0, 1)
    assert evidence.any_clipping_intervention
    assert not evidence.observed_bounded_output_matches_raw
    assert result.metadata["clipped_node_count"] == 2
    assert result.metadata["epi_raw_mean_after"] == pytest.approx(0.0)


def test_metric_weights_require_opt_in_and_strict_positive_domain() -> None:
    graph = _graph()

    with pytest.raises(TNFRValueError, match="requires"):
        apply_network_remesh(graph, metric_weights=(1.0, 1.0))
    with pytest.raises(TNFRValueError, match="strictly positive"):
        apply_network_remesh(
            graph,
            include_stability_evidence=True,
            metric_weights=(1.0, 0.0),
        )

    graph.graph["_epi_hist"] = deque([{0: 0.5, 1: 0.2}], maxlen=8)
    with pytest.raises(TNFRValueError, match="strictly positive"):
        apply_network_remesh(
            graph,
            include_stability_evidence=True,
            metric_weights=(1.0, 0.0),
        )


def test_stability_gate_does_not_consume_cooldown_on_history_noop() -> None:
    graph = _graph()
    graph.graph["_epi_hist"] = deque(
        [{0: 0.5, 1: 0.2}],
        maxlen=8,
    )
    graph.graph.update(
        REMESH_STABILITY_WINDOW=1,
        REMESH_REQUIRE_STABILITY=False,
        REMESH_COOLDOWN_WINDOW=0,
        FRACTION_STABLE_REMESH=0.5,
        _t=3.0,
    )
    graph.graph["history"] = {"stable_frac": [1.0]}

    remesh_module.apply_remesh_if_globally_stable(graph)

    assert "_last_remesh_step" not in graph.graph
    assert "_last_remesh_ts" not in graph.graph


@pytest.mark.parametrize("hard_override", [1, 0, "false", None])
def test_alpha_precedence_flag_has_a_strict_boolean_domain(
    hard_override: object,
) -> None:
    graph = _graph()
    graph.graph["REMESH_ALPHA_HARD"] = hard_override
    before = _state(graph)

    with pytest.raises(TNFRValueError, match="REMESH_ALPHA_HARD"):
        apply_network_remesh(graph)

    assert _state(graph) == before


def test_invalid_optional_telemetry_rolls_back_the_delayed_commit() -> None:
    graph = _graph()
    graph.graph["history"] = {"stable_frac": [{"not": "a scalar"}]}
    before = _state(graph)

    with pytest.raises(TNFRValueError, match="finite scalars"):
        apply_network_remesh(graph)

    assert _state(graph) == before


def test_result_metadata_property_is_detached_from_canonical_metadata() -> None:
    graph = _graph()

    result = apply_network_remesh(graph)
    detached = result.metadata
    detached["alpha"] = 999.0

    assert result.metadata["alpha"] == pytest.approx(0.5)
    assert graph.graph["_REMESH_META"]["alpha"] == pytest.approx(0.5)


def test_public_remesh_stubs_expose_the_result_contract() -> None:
    import ast
    from pathlib import Path

    package_dir = Path(remesh_module.__file__).resolve().parent
    remesh_stub = ast.parse(
        (package_dir / "remesh.pyi").read_text(encoding="utf-8")
    )
    package_stub = ast.parse(
        (package_dir / "__init__.pyi").read_text(encoding="utf-8")
    )
    functions = {
        node.name: node
        for node in remesh_stub.body
        if isinstance(node, ast.FunctionDef)
    }

    planner = functions["plan_network_remesh"]
    executor = functions["apply_network_remesh"]
    assert ast.unparse(planner.returns) == "DelayedRemeshPlan"
    assert ast.unparse(executor.returns) == "DelayedRemeshResult"
    assert [arg.arg for arg in executor.args.kwonlyargs] == [
        "include_stability_evidence",
        "metric_weights",
    ]

    imported_records = {
        alias.name
        for node in remesh_stub.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "_delayed_remesh_kernel"
        for alias in node.names
    }
    assert imported_records == {
        "DelayedRemeshNodeProposal",
        "DelayedRemeshPlan",
        "DelayedRemeshResult",
        "DelayedRemeshStabilityEvidence",
    }
    assert package_stub.body

def test_exact_evidence_overflow_rejects_before_graph_writes() -> None:
    magnitude = float.fromhex("0x1p+1023")
    float_max = float.fromhex("0x1.fffffffffffffp+1023")

    for operation in (plan_network_remesh, apply_network_remesh):
        graph = _graph(
            current=(magnitude, -magnitude),
            past=(magnitude, -magnitude),
            epi_min=-float_max,
            epi_max=float_max,
        )
        before = _state(graph)

        with pytest.raises(
            TNFRValueError,
            match="finite binary64 diagnostic range",
        ):
            operation(graph, include_stability_evidence=True)

        assert _state(graph) == before


def test_extreme_finite_proposal_mean_has_no_intermediate_overflow() -> None:
    float_max = float.fromhex("0x1.fffffffffffffp+1023")
    graph = _graph(
        current=(float_max, float_max),
        past=(float_max, float_max),
        epi_min=-float_max,
        epi_max=float_max,
    )

    result = apply_network_remesh(graph)

    assert result.applied
    assert result.metadata["epi_mean_before"] == float_max
    assert result.metadata["epi_mean_after"] == float_max
    assert result.metadata["epi_raw_mean_after"] == float_max


def test_empty_support_is_a_side_effect_free_noop() -> None:
    graph = nx.Graph()
    graph.graph.update(
        REMESH_TAU_GLOBAL=1,
        REMESH_TAU_LOCAL=1,
        REMESH_ALPHA=0.5,
        REMESH_ALPHA_HARD=True,
        REMESH_LOG_EVENTS=True,
        REMESH_STABILITY_WINDOW=1,
        REMESH_REQUIRE_STABILITY=False,
        REMESH_COOLDOWN_WINDOW=0,
        FRACTION_STABLE_REMESH=0.5,
        EPI_MIN=-1.0,
        EPI_MAX=1.0,
        CLIP_MODE="hard",
        _epi_hist=deque([{}, {}], maxlen=8),
        history={"stable_frac": [1.0]},
        _t=3.0,
    )
    callback_observations: list[dict[str, object]] = []

    def observe_callback(_target, context):
        callback_observations.append(dict(context))

    callback_manager.register_callback(
        graph,
        CallbackEvent.ON_REMESH,
        observe_callback,
        name="empty-support-remesh-observer",
    )
    history_before = deepcopy(graph.graph["history"])

    plan = plan_network_remesh(
        graph,
        include_stability_evidence=True,
    )
    result = apply_network_remesh(
        graph,
        include_stability_evidence=True,
    )
    remesh_module.apply_remesh_if_globally_stable(graph)

    assert plan.status == "empty_support"
    assert result.status == "empty_support"
    assert not plan.applied
    assert not result.applied
    assert plan.proposals == result.proposals == ()
    assert plan.evidence is None
    assert result.evidence is None
    assert result.metadata == {}
    assert graph.graph["history"] == history_before
    assert "_REMESH_META" not in graph.graph
    assert "_last_remesh_step" not in graph.graph
    assert "_last_remesh_ts" not in graph.graph
    assert callback_observations == []

def test_direct_apply_records_post_remesh_epi_time_right_endpoint() -> None:
    graph = _graph(current=(2.0, 0.0), past=(0.0, 2.0))
    graph.graph["_t"] = 3.0
    for node in graph:
        graph.nodes[node]["epi_time_history"] = deque(
            [(3.0, graph.nodes[node]["EPI"])], maxlen=3
        )

    result = apply_network_remesh(graph)

    assert result.applied
    assert result.epi_time_boundary_recorded
    assert list(graph.nodes[0]["epi_time_history"]) == [(3.0, 0.5)]
    assert list(graph.nodes[1]["epi_time_history"]) == [(3.0, 1.5)]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("capacity", "capacity, pressure or phase"),
        ("pressure", "capacity, pressure or phase"),
        ("phase", "capacity, pressure or phase"),
        ("alpha_source", "alpha source"),
        ("clock", "runtime clock"),
        ("event_log", "hybrid_event_log"),
        ("delayed_history", "_epi_hist"),
        ("time_history", "epi_time_history"),
        ("edge", "edge support or attributes"),
        ("configuration", "delayed-map configuration"),
    ],
)
def test_direct_remesh_observer_is_read_only_on_contracted_surfaces(
    mutation: str,
    message: str,
) -> None:
    graph = _graph()
    graph.graph["_t"] = 0.0
    graph.graph["hybrid_event_log"] = [{"seed": 1}]
    for node in graph:
        graph.nodes[node].update(nu_f=1.0, delta_nfr=0.0, theta=0.0)

    def mutate(target: nx.Graph, _context: object) -> None:
        if mutation == "capacity":
            target.nodes[0]["nu_f"] = 7.0
        elif mutation == "pressure":
            target.nodes[0]["delta_nfr"] = 9.0
        elif mutation == "phase":
            target.nodes[0]["theta"] = 1.25
        elif mutation == "alpha_source":
            target.graph["_REMESH_ALPHA_SRC"] = "forged"
        elif mutation == "clock":
            target.graph["_t"] = 9.0
        elif mutation == "event_log":
            target.graph["hybrid_event_log"].clear()
        elif mutation == "delayed_history":
            target.graph["_epi_hist"].append({0: 9.0, 1: 9.0})
        elif mutation == "time_history":
            target.nodes[0]["epi_time_history"].clear()
        elif mutation == "edge":
            target.edges[0, 1]["weight"] = 5.0
        else:
            target.graph["EPI_MAX"] = 9.0

    callback_manager.register_callback(
        graph,
        CallbackEvent.ON_REMESH,
        mutate,
        name=f"read-only-{mutation}",
    )
    before = _state(graph)
    delayed_history = graph.graph["_epi_hist"]

    with pytest.raises(TNFRValueError, match=message):
        apply_network_remesh(graph)

    assert _state(graph) == before
    assert graph.graph["_epi_hist"] is delayed_history


def test_direct_remesh_preserves_just_appended_canonical_telemetry() -> None:
    graph = _graph(log_events=True)

    def clear_event(target: nx.Graph, _context: object) -> None:
        target.graph["history"]["remesh_events"].clear()

    callback_manager.register_callback(
        graph,
        CallbackEvent.ON_REMESH,
        clear_event,
        name="clear-canonical-remesh-event",
    )
    before = _state(graph)

    with pytest.raises(TNFRValueError, match="canonical remesh_events"):
        apply_network_remesh(graph)

    assert _state(graph) == before


def test_ordinary_runtime_remesh_updates_authoritative_epi_time_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tnfr.dynamics import runtime as runtime_module

    graph = _graph(current=(2.0, 0.0), past=(0.0, 2.0))
    graph.graph["_epi_hist"] = deque([{0: 0.0, 1: 2.0}], maxlen=64)
    graph.graph.update(
        _t=0.0,
        REMESH_STABILITY_WINDOW=1,
        REMESH_REQUIRE_STABILITY=False,
        REMESH_COOLDOWN_WINDOW=0,
        REMESH_COOLDOWN_TS=0.0,
        FRACTION_STABLE_REMESH=0.5,
        history={"stable_frac": [1.0], "C_steps": []},
    )
    for node in graph:
        graph.nodes[node]["epi_time_history"] = deque(
            [(0.0, graph.nodes[node]["EPI"])], maxlen=3
        )

    monkeypatch.setattr(runtime_module, "_run_before_callbacks", lambda *a, **k: None)
    monkeypatch.setattr(runtime_module, "_update_nodes", lambda *a, **k: None)
    monkeypatch.setattr(runtime_module, "_advance_math_engine", lambda *a, **k: None)
    monkeypatch.setattr(runtime_module, "_run_validators", lambda *a, **k: None)
    monkeypatch.setattr(runtime_module, "_run_after_callbacks", lambda *a, **k: None)
    monkeypatch.setattr(
        runtime_module, "publish_graph_cache_metrics", lambda *a, **k: None
    )

    runtime_module.step(graph, dt=0.1, use_Si=False, apply_glyphs=False)

    assert graph.nodes[0]["EPI"] == 0.5
    assert graph.nodes[1]["EPI"] == 1.5
    assert list(graph.nodes[0]["epi_time_history"]) == [(0.0, 0.5)]
    assert list(graph.nodes[1]["epi_time_history"]) == [(0.0, 1.5)]

def test_structural_memory_epi_write_records_a_second_right_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _graph(current=(2.0, 0.0), past=(0.0, 2.0))
    graph.graph["_t"] = 0.0
    for node in graph:
        graph.nodes[node]["epi_time_history"] = deque(
            [(0.0, graph.nodes[node]["EPI"])], maxlen=3
        )

    monkeypatch.setattr(
        remesh_module,
        "detect_recursive_patterns",
        lambda *args, **kwargs: [[0, 1]],
    )
    monkeypatch.setattr(
        remesh_module,
        "identify_pattern_origin",
        lambda *args, **kwargs: 0,
    )

    def propagate(target, _origin, _targets, *, propagation_strength):
        assert propagation_strength == 0.5
        target.nodes[1]["EPI"] = 7.0

    monkeypatch.setattr(remesh_module, "propagate_structural_identity", propagate)

    remesh_module.apply_network_remesh_with_memory(graph)

    assert list(graph.nodes[0]["epi_time_history"]) == [(0.0, 0.5)]
    assert list(graph.nodes[1]["epi_time_history"]) == [(0.0, 7.0)]