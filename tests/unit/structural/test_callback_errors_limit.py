from collections import deque

from tnfr.callback_utils import (
    CallbackEvent,
    callback_manager,
)


def test_callback_error_list_resets_limit(graph_canon):
    G = graph_canon()

    def failing_cb(G, ctx):
        raise RuntimeError("boom")

    callback_manager.register_callback(
        G, CallbackEvent.BEFORE_STEP, failing_cb, name="fail"
    )
    original = deque(maxlen=None)
    G.graph["_callback_errors"] = original

    prev = callback_manager.get_callback_error_limit()
    callback_manager.set_callback_error_limit(7)
    try:
        callback_manager.invoke_callbacks(G, CallbackEvent.BEFORE_STEP, {})
        err_list = G.graph.get("_callback_errors")
        assert err_list is not original
        assert err_list.maxlen == callback_manager.get_callback_error_limit() == 7
        assert len(err_list) == 1
    finally:
        callback_manager.set_callback_error_limit(prev)
