"""Pruebas de trace."""
import pytest

from tnfr.trace import register_trace, _callback_names, gamma_field, grammar_field
from tnfr.callback_utils import register_callback, invoke_callbacks


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

    register_callback(G, event="before_step", func=foo, name="custom_cb")
    invoke_callbacks(G, "before_step")

    hist = G.graph["history"]["trace_meta"]
    meta = hist[0]
    assert "callbacks" in meta
    assert "custom_cb" in meta["callbacks"].get("before_step", [])


def test_trace_sigma_no_glyphs(graph_canon):
    G = graph_canon()
    # add nodes without glyph history
    G.add_nodes_from([1, 2, 3])
    register_trace(G)
    invoke_callbacks(G, "after_step")
    meta = G.graph["history"]["trace_meta"][0]
    assert meta["phase"] == "after"
    assert meta["sigma"] == {
        "x": 0.0,
        "y": 0.0,
        "mag": 0.0,
        "angle": 0.0,
    }


def test_callback_names_empty_tuple():
    """Los tuples vacíos son ignorados y no causan errores."""

    def foo():
        pass

    names = _callback_names([(), (foo,)])
    assert names == ["foo"]


def test_gamma_field_non_mapping_warns(graph_canon):
    G = graph_canon()
    G.graph["GAMMA"] = "not a dict"
    with pytest.warns(UserWarning):
        out = gamma_field(G)
    assert out == {}


def test_grammar_field_non_mapping_warns(graph_canon):
    G = graph_canon()
    G.graph["GRAMMAR_CANON"] = 123
    with pytest.warns(UserWarning):
        out = grammar_field(G)
    assert out == {}
