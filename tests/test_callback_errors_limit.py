from collections import deque

from tnfr.callback_utils import (
    CallbackEvent,
    invoke_callbacks,
    register_callback,
    _CALLBACK_ERROR_LIMIT,
)


def test_callback_error_list_resets_limit(graph_canon):
    G = graph_canon()

    def failing_cb(G, ctx):
        raise RuntimeError("boom")

    register_callback(G, CallbackEvent.BEFORE_STEP, failing_cb, name="fail")
    original = deque(maxlen=None)
    G.graph["_callback_errors"] = original

    invoke_callbacks(G, CallbackEvent.BEFORE_STEP, {})

    err_list = G.graph.get("_callback_errors")
    assert err_list is not original
    assert err_list.maxlen == _CALLBACK_ERROR_LIMIT
    assert len(err_list) == 1

