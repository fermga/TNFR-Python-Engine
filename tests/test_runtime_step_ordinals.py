"""Execution order survives telemetry retention and is distinct from time."""

import asyncio
from contextlib import nullcontext
from copy import deepcopy

import networkx as nx
import pytest

from tnfr._runtime_steps import (
    RUNTIME_STEP_NEXT_KEY,
    prepare_runtime_step,
    runtime_step_scope,
)
from tnfr.constants import inject_defaults
from tnfr.dynamics import runtime
from tnfr.errors import NetworkConfigError
from tnfr.glyph_history import current_step_idx
from tnfr.metrics.core import register_metrics_callbacks
from tnfr.rng import _rng_for_step, base_seed
from tnfr.utils import CallbackEvent, callback_manager


def _graph(*, bound=0, metrics=True, size=2, attach_metrics=True):
    graph = nx.path_graph(size)
    inject_defaults(graph)
    graph.graph.update(
        HISTORY_MAXLEN=bound,
        PHASE_ADAPT={"enabled": False},
        CALLBACKS_STRICT=True,
        RANDOM_SEED=1729,
        UM_CANDIDATE_COUNT=5,
    )
    graph.graph["METRICS"]["enabled"] = metrics
    for node in graph:
        graph.nodes[node].update(EPI=0.0, nu_f=1.0, theta=0.0, delta_nfr=0.0, Si=0.5)
    if attach_metrics:
        register_metrics_callbacks(graph)
    return graph


def _advance(graph, dt=0.125):
    runtime.step(graph, dt=dt, use_Si=False, apply_glyphs=False)


@pytest.mark.parametrize("bound, metrics", [(0, True), (2, True), (2, False)])
def test_callbacks_share_ordinals_across_retention_and_resumed_calls(bound, metrics):
    graph = _graph(bound=bound, metrics=metrics)
    records = []
    for event in (CallbackEvent.BEFORE_STEP, CallbackEvent.AFTER_STEP):

        def record(graph, ctx, event=event):
            records.append((event, ctx["step"], current_step_idx(graph)))

        callback_manager.register_callback(graph, event, record)
    for dt in (0.125, 0.0, 0.25):
        _advance(graph, dt)
    # Clearing telemetry does not reset the execution epoch.
    graph.graph["history"] = {}
    runtime.run(graph, 2, dt=0.125, use_Si=False, apply_glyphs=False)
    assert records == [
        (event, index, index)
        for index in range(5)
        for event in (CallbackEvent.BEFORE_STEP, CallbackEvent.AFTER_STEP)
    ]
    assert current_step_idx(graph) == 5
    assert graph.graph["_t"] == 0.625
    assert prepare_runtime_step(graph.graph) == 5


def test_preexisting_metric_history_does_not_invent_a_runtime_epoch():
    graph = _graph()
    graph.graph["history"] = {"C_steps": [1.0] * 9}
    assert current_step_idx(graph) == 9  # Standalone compatibility only.
    seen = []
    callback_manager.register_callback(
        graph, CallbackEvent.BEFORE_STEP, lambda graph, ctx: seen.append(ctx["step"])
    )
    _advance(graph)
    assert seen == [0]
    assert current_step_idx(graph) == 1


@pytest.mark.parametrize("value", [True, -1, 1.5, "3", None])
def test_invalid_ordinal_rejects_without_progress(value):
    graph = _graph(attach_metrics=False)
    graph.graph[RUNTIME_STEP_NEXT_KEY] = value
    before = deepcopy(graph)
    with pytest.raises((TypeError, ValueError)):
        _advance(graph)
    assert graph.graph == before.graph
    assert dict(graph.nodes(data=True)) == dict(before.nodes(data=True))


def test_invalid_policy_does_not_reserve_an_ordinal():
    graph = _graph()
    with pytest.raises(NetworkConfigError):
        _advance(graph, -1.0)
    assert RUNTIME_STEP_NEXT_KEY not in graph.graph
    assert prepare_runtime_step(graph.graph) == 0


@pytest.mark.parametrize("event", [CallbackEvent.BEFORE_STEP, CallbackEvent.AFTER_STEP])
def test_admitted_failure_consumes_ordinal_and_cleans_active_marker(event):
    graph = _graph()
    seen = []

    def fail_first(graph, ctx):
        seen.append(ctx["step"])
        assert current_step_idx(graph) == ctx["step"]
        if ctx["step"] == 0:
            raise RuntimeError("declared callback failure")

    callback_manager.register_callback(graph, event, fail_first)
    with pytest.raises(RuntimeError, match="declared callback failure"):
        _advance(graph)
    assert current_step_idx(graph) == 1
    assert prepare_runtime_step(graph.graph) == 1
    _advance(graph)
    assert seen == [0, 1]
    assert current_step_idx(graph) == 2


def test_recursive_same_graph_step_cannot_reserve_another_ordinal():
    graph = _graph()

    def recurse(graph, ctx):
        with pytest.raises(RuntimeError, match="Recursive"):
            _advance(graph)
        assert current_step_idx(graph) == ctx["step"] == 0

    callback_manager.register_callback(graph, CallbackEvent.BEFORE_STEP, recurse)
    _advance(graph)
    assert current_step_idx(graph) == 1


def test_copy_inside_callback_does_not_inherit_a_live_execution_guard():
    graph = _graph()
    copies = []

    def capture(graph, ctx):
        copies.append(graph.copy())

    callback_manager.register_callback(graph, CallbackEvent.BEFORE_STEP, capture)
    _advance(graph)
    copied = copies[0]
    assert current_step_idx(graph) == 1
    assert current_step_idx(copied) == 0  # Position captured before source work.
    assert prepare_runtime_step(copied.graph) == 0
    _advance(copied)
    assert current_step_idx(copied) == 1


@pytest.mark.parametrize("late_failure", [False, True])
def test_inherited_async_guard_tracks_source_scope_liveness(late_failure):
    async def scenario():
        graph = _graph(attach_metrics=False)
        attempted = asyncio.Event()
        resume = asyncio.Event()

        async def deferred_step():
            try:
                with pytest.raises(RuntimeError, match="Recursive"):
                    _advance(graph)
            finally:
                attempted.set()
            await resume.wait()
            assert current_step_idx(graph) == 1
            _advance(graph)

        expectation = (
            pytest.raises(RuntimeError, match="source callback failed")
            if late_failure
            else nullcontext()
        )
        with expectation:
            with runtime_step_scope(graph.graph, 0):
                task = asyncio.create_task(deferred_step())
                await attempted.wait()
                assert current_step_idx(graph) == 0
                if late_failure:
                    raise RuntimeError("source callback failed")

        assert prepare_runtime_step(graph.graph) == 1
        resume.set()
        await task
        assert current_step_idx(graph) == 2
        assert graph.graph["_t"] == 0.125

    asyncio.run(scenario())


def test_candidate_sampling_uses_execution_ordinals_with_metrics_disabled():
    graph = _graph(bound=2, metrics=False, size=60)
    nodes = tuple(graph)
    samples = []
    callback_manager.register_callback(
        graph,
        CallbackEvent.AFTER_STEP,
        lambda graph, ctx: samples.append(
            (ctx["step"], tuple(graph.graph["_node_sample"]))
        ),
    )
    runtime.run(graph, 4, dt=0.125, use_Si=False, apply_glyphs=False)
    expected = [
        (index, tuple(_rng_for_step(base_seed(graph), index).sample(nodes, 5)))
        for index in range(4)
    ]
    assert samples == expected
    assert len({sample for _, sample in samples}) == 4
