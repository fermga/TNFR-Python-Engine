from tnfr.helpers import register_callback


def test_register_callback_replaces_existing(graph_canon):
    G = graph_canon()

    def cb1(G, ctx):
        pass

    def cb2(G, ctx):
        pass

    # initial registration
    register_callback(G, event="before_step", func=cb1, name="cb")
    assert G.graph["callbacks"]["before_step"] == [("cb", cb1)]

    # same name should replace existing
    register_callback(G, event="before_step", func=cb2, name="cb")
    assert G.graph["callbacks"]["before_step"] == [("cb", cb2)]

    # same function with different name should also replace existing
    register_callback(G, event="before_step", func=cb2, name="other")
    assert G.graph["callbacks"]["before_step"] == [("other", cb2)]
