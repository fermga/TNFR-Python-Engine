"""Tests for invoke_callbacks context handling."""

from tnfr.callback_utils import CallbackEvent, invoke_callbacks, register_callback


def test_invoke_callbacks_preserves_context(graph_canon):
    G = graph_canon()

    def cb(G, ctx):
        ctx["called"] = ctx.get("called", 0) + 1

    register_callback(G, CallbackEvent.BEFORE_STEP, cb)

    ctx = {}
    invoke_callbacks(G, CallbackEvent.BEFORE_STEP, ctx)

    assert ctx["called"] == 1
