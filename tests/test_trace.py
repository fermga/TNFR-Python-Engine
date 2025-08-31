from tnfr.trace import register_trace


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
