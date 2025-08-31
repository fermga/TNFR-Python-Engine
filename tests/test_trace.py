from tnfr.trace import register_trace
from tnfr.helpers import register_callback, invoke_callbacks


def test_register_trace_idempotent(graph_canon):
    G = graph_canon()
    register_trace(G)
    # callbacks should be registered once and flag set
    assert G.graph["_trace_registered"] is True
    before = list(G.graph["callbacks"]["before_step"])
    after = list(G.graph["callbacks"]["after_step"])

    register_trace(G)

    assert list(G.graph["callbacks"]["before_step"]) == before
    assert list(G.graph["callbacks"]["after_step"]) == after


def test_trace_metadata_contains_callback_names(graph_canon):
    G = graph_canon()
    register_trace(G)

    def foo(G, ctx):
        pass

    register_callback(G, when="before_step", func=foo, name="custom_cb")
    invoke_callbacks(G, "before_step")

    hist = G.graph["history"]["trace_meta"]
    meta = hist[0]
    assert "callbacks" in meta
    assert "custom_cb" in meta["callbacks"].get("before_step", [])
