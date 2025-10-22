"""Unit tests for callback registration and replacement behavior."""

import logging

import pytest

from tnfr.callback_utils import (
    CallbackEvent,
    CallbackSpec,
    callback_manager,
)


def test_register_callback_replaces_existing(graph_canon):
    G = graph_canon()

    def cb1(G, ctx):
        pass

    def cb2(G, ctx):
        pass

    # initial registration
    callback_manager.register_callback(
        G, event=CallbackEvent.BEFORE_STEP, func=cb1, name="cb"
    )
    assert G.graph["callbacks"][CallbackEvent.BEFORE_STEP.value] == {
        "cb": CallbackSpec("cb", cb1)
    }

    # same name should replace existing
    callback_manager.register_callback(
        G, event=CallbackEvent.BEFORE_STEP, func=cb2, name="cb"
    )
    assert G.graph["callbacks"][CallbackEvent.BEFORE_STEP.value] == {
        "cb": CallbackSpec("cb", cb2)
    }

    # same function with different name should also replace existing
    callback_manager.register_callback(
        G, event=CallbackEvent.BEFORE_STEP, func=cb2, name="other"
    )
    assert G.graph["callbacks"][CallbackEvent.BEFORE_STEP.value] == {
        "other": CallbackSpec("other", cb2)
    }


def test_register_callback_same_callable_dedupes(graph_canon):
    G = graph_canon()

    def cb(G, ctx):
        pass

    callback_manager.register_callback(G, CallbackEvent.BEFORE_STEP, cb)
    callback_manager.register_callback(G, CallbackEvent.BEFORE_STEP, cb)
    assert G.graph["callbacks"][CallbackEvent.BEFORE_STEP.value] == {
        "cb": CallbackSpec("cb", cb)
    }


def test_register_callback_rejects_tuple(graph_canon):
    G = graph_canon()

    def cb(G, ctx):
        pass

    with pytest.raises(TypeError, match="must be callable"):
        callback_manager.register_callback(
            G, event=CallbackEvent.BEFORE_STEP, func=("cb", cb)
        )


def test_enum_registration_and_invocation(graph_canon):
    G = graph_canon()

    def cb(G, ctx):
        ctx["called"] += 1

    callback_manager.register_callback(G, CallbackEvent.AFTER_STEP, cb)
    ctx = {"called": 0}
    callback_manager.invoke_callbacks(G, CallbackEvent.AFTER_STEP, ctx)
    assert ctx["called"] == 1


def test_register_callback_unknown_event(graph_canon):
    G = graph_canon()

    def cb(G, ctx):
        pass

    with pytest.raises(ValueError, match="Unknown event"):
        callback_manager.register_callback(G, event="nope", func=cb)


def test_register_callback_logs_on_overwrite(graph_canon, caplog):
    G = graph_canon()

    def cb1(G, ctx):
        pass

    def cb2(G, ctx):
        pass

    callback_manager.register_callback(G, CallbackEvent.BEFORE_STEP, cb1, name="cb")
    with caplog.at_level(logging.WARNING):
        callback_manager.register_callback(G, CallbackEvent.BEFORE_STEP, cb2, name="cb")
        assert any("already registered" in r.message for r in caplog.records)


def test_register_callback_strict_overwrite_raises(graph_canon):
    G = graph_canon()
    G.graph["CALLBACKS_STRICT"] = True

    def cb1(G, ctx):
        pass

    def cb2(G, ctx):
        pass

    callback_manager.register_callback(G, CallbackEvent.BEFORE_STEP, cb1, name="cb")
    with pytest.raises(ValueError, match="already registered"):
        callback_manager.register_callback(G, CallbackEvent.BEFORE_STEP, cb2, name="cb")
