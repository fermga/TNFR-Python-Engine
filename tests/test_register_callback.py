"""Pruebas de register callback."""

import pytest

from tnfr.callback_utils import register_callback, invoke_callbacks, CallbackEvent


def test_register_callback_replaces_existing(graph_canon):
    G = graph_canon()

    def cb1(G, ctx):
        pass

    def cb2(G, ctx):
        pass

    # initial registration
    register_callback(G, event=CallbackEvent.BEFORE_STEP, func=cb1, name="cb")
    assert G.graph["callbacks"][CallbackEvent.BEFORE_STEP.value] == [("cb", cb1)]

    # same name should replace existing
    register_callback(G, event=CallbackEvent.BEFORE_STEP, func=cb2, name="cb")
    assert G.graph["callbacks"][CallbackEvent.BEFORE_STEP.value] == [("cb", cb2)]

    # same function with different name should also replace existing
    register_callback(G, event=CallbackEvent.BEFORE_STEP, func=cb2, name="other")
    assert G.graph["callbacks"][CallbackEvent.BEFORE_STEP.value] == [("other", cb2)]


def test_register_callback_rejects_tuple(graph_canon):
    G = graph_canon()

    def cb(G, ctx):
        pass

    with pytest.raises(TypeError, match="must be callable"):
        register_callback(G, event=CallbackEvent.BEFORE_STEP, func=("cb", cb))


def test_enum_registration_and_invocation(graph_canon):
    G = graph_canon()

    def cb(G, ctx):
        ctx["called"] += 1

    register_callback(G, CallbackEvent.AFTER_STEP, cb)
    ctx = {"called": 0}
    invoke_callbacks(G, CallbackEvent.AFTER_STEP, ctx)
    assert ctx["called"] == 1


def test_register_callback_unknown_event(graph_canon):
    G = graph_canon()

    def cb(G, ctx):
        pass

    with pytest.raises(ValueError, match="Unknown event"):
        register_callback(G, event="nope", func=cb)
