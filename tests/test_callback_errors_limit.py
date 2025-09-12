from collections import deque

import tnfr.callback_utils as cb_utils

from tnfr.callback_utils import (
    CallbackEvent,
    invoke_callbacks,
    register_callback,
    set_callback_error_limit,
)


def test_callback_error_list_resets_limit(graph_canon):
    G = graph_canon()

    def failing_cb(G, ctx):
        raise RuntimeError("boom")

    register_callback(G, CallbackEvent.BEFORE_STEP, failing_cb, name="fail")
    original = deque(maxlen=None)
    G.graph["_callback_errors"] = original

    prev = set_callback_error_limit(7)
    try:
        invoke_callbacks(G, CallbackEvent.BEFORE_STEP, {})
        err_list = G.graph.get("_callback_errors")
        assert err_list is not original
        assert err_list.maxlen == cb_utils.get_callback_error_limit() == 7
        assert len(err_list) == 1
    finally:
        set_callback_error_limit(prev)
