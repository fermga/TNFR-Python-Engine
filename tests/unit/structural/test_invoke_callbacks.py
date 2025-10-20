"""Tests for invoke_callbacks context handling."""

import networkx as nx
import pytest

from tnfr.callback_utils import CallbackEvent, callback_manager


def test_invoke_callbacks_preserves_context(graph_canon):
    G = graph_canon()

    def cb(G, ctx):
        ctx["called"] = ctx.get("called", 0) + 1

    callback_manager.register_callback(G, CallbackEvent.BEFORE_STEP, cb)

    ctx = {}
    callback_manager.invoke_callbacks(G, CallbackEvent.BEFORE_STEP, ctx)

    assert ctx["called"] == 1


def test_invoke_callbacks_records_networkx_error(graph_canon):
    G = graph_canon()

    def failing_cb(G, ctx):
        G.remove_node("missing")

    callback_manager.register_callback(G, CallbackEvent.BEFORE_STEP, failing_cb)

    ctx = {"step": 5}
    with pytest.raises(nx.NetworkXError):
        callback_manager.invoke_callbacks(G, CallbackEvent.BEFORE_STEP, ctx)

    err_list = G.graph.get("_callback_errors")
    assert err_list and len(err_list) == 1
    err = err_list[0]
    assert err["event"] == CallbackEvent.BEFORE_STEP.value
    assert err["step"] == 5
    assert "NetworkXError" in err["error"]
