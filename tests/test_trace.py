"""Pruebas de trace."""

import pytest

from tnfr.trace import (
    register_trace,
    register_trace_field,
    _callback_names,
    gamma_field,
    grammar_field,
    CallbackSpec,
)
from tnfr import trace
from tnfr.helpers import get_graph_mapping
from tnfr.callback_utils import register_callback, invoke_callbacks
from types import MappingProxyType


def test_register_trace_idempotent(graph_canon):
    G = graph_canon()
    register_trace(G)
    # callbacks should be registered once and flag set
    assert G.graph["_trace_registered"] is True
    before = dict(G.graph["callbacks"]["before_step"])
    after = dict(G.graph["callbacks"]["after_step"])

    register_trace(G)

    assert dict(G.graph["callbacks"]["before_step"]) == before
    assert dict(G.graph["callbacks"]["after_step"]) == after


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


def test_callback_names_spec():
    """CallbackSpec entries are handled correctly."""

    def foo():
        pass

    names = _callback_names(
        [CallbackSpec("bar", foo), CallbackSpec(None, foo)]
    )
    assert names == ["bar", "foo"]


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


def test_get_graph_mapping_returns_proxy(graph_canon):
    G = graph_canon()
    data = {"a": 1}
    G.graph["foo"] = data
    out = get_graph_mapping(G, "foo", "msg")
    assert isinstance(out, MappingProxyType)
    assert out["a"] == 1
    with pytest.raises(TypeError):
        out["b"] = 2


def test_register_trace_field_runtime(graph_canon):
    G = graph_canon()
    G.graph["TRACE"] = {"enabled": True, "capture": ["custom"], "history_key": "trace_meta"}
    register_trace(G)

    def custom_field(G):
        return {"custom": 42}

    register_trace_field("before", "custom", custom_field)
    invoke_callbacks(G, "before_step")

    meta = G.graph["history"]["trace_meta"][0]
    assert meta["custom"] == 42
    del trace.TRACE_FIELDS["before"]["custom"]
