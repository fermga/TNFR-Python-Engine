"""Tests for `_ensure_callbacks` behavior."""

from tnfr.callback_utils import (
    _ensure_callbacks,
    register_callback,
    CallbackEvent,
)


def test_ensure_callbacks_drops_unknown_events(graph_canon):
    G = graph_canon()

    def cb(G, ctx):
        pass

    G.graph["callbacks"] = {
        "nope": [("cb", cb)],
        CallbackEvent.BEFORE_STEP.value: [("cb", cb)],
    }

    _ensure_callbacks(G)

    assert list(G.graph["callbacks"]) == [CallbackEvent.BEFORE_STEP.value]


def test_register_callback_cleans_unknown_events(graph_canon):
    G = graph_canon()

    def cb(G, ctx):
        pass

    G.graph["callbacks"] = {"nope": [("cb", cb)]}

    register_callback(G, CallbackEvent.AFTER_STEP, cb, name="cb")

    assert list(G.graph["callbacks"]) == [CallbackEvent.AFTER_STEP.value]


def test_ensure_callbacks_only_processes_dirty_events(graph_canon):
    G = graph_canon()
    from collections import defaultdict

    dummy = object()
    G.graph["callbacks"] = defaultdict(
        list,
        {
            CallbackEvent.BEFORE_STEP.value: [dummy],
            CallbackEvent.AFTER_STEP.value: [dummy],
        },
    )
    G.graph["_callbacks_dirty"] = {CallbackEvent.BEFORE_STEP.value}

    _ensure_callbacks(G)

    assert G.graph["callbacks"][CallbackEvent.BEFORE_STEP.value] == {}
    assert G.graph["callbacks"][CallbackEvent.AFTER_STEP.value] == {}
