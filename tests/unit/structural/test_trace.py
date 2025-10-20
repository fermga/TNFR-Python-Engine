"""Pruebas de trace."""

import pytest

from tnfr.trace import (
    register_trace,
    register_trace_field,
    _callback_names,
    gamma_field,
    grammar_field,
    mapping_field,
    CallbackSpec,
)
from tnfr import trace
from tnfr.utils import get_graph_mapping
from tnfr.callback_utils import CallbackEvent, callback_manager
from types import MappingProxyType


def test_register_trace_idempotent(graph_canon):
    G = graph_canon()
    register_trace(G)
    # callbacks should be registered once and flag set
    assert G.graph["_trace_registered"] is True
    before = dict(G.graph["callbacks"][CallbackEvent.BEFORE_STEP.value])
    after = dict(G.graph["callbacks"][CallbackEvent.AFTER_STEP.value])

    register_trace(G)

    assert dict(G.graph["callbacks"][CallbackEvent.BEFORE_STEP.value]) == before
    assert dict(G.graph["callbacks"][CallbackEvent.AFTER_STEP.value]) == after


def test_trace_metadata_contains_callback_names(graph_canon):
    G = graph_canon()
    register_trace(G)

    def foo(G, ctx):
        pass

    callback_manager.register_callback(
        G,
        event=CallbackEvent.BEFORE_STEP.value,
        func=foo,
        name="custom_cb",
    )
    callback_manager.invoke_callbacks(G, CallbackEvent.BEFORE_STEP.value)

    hist = G.graph["history"]["trace_meta"]
    meta = hist[0]
    assert "callbacks" in meta
    assert "custom_cb" in meta["callbacks"].get(CallbackEvent.BEFORE_STEP.value, [])


def test_trace_sigma_no_glyphs(graph_canon):
    G = graph_canon()
    # add nodes without glyph history
    G.add_nodes_from([1, 2, 3])
    register_trace(G)
    callback_manager.invoke_callbacks(G, CallbackEvent.AFTER_STEP.value)
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


def test_mapping_field_returns_proxy(graph_canon):
    G = graph_canon()
    G.graph["FOO"] = {"a": 1}
    out = mapping_field(G, "FOO", "bar")
    mapping = out["bar"]
    assert isinstance(mapping, MappingProxyType)
    assert mapping["a"] == 1
    with pytest.raises(TypeError):
        mapping["b"] = 2


def test_trace_metadata_fields_have_generators(graph_canon):
    """Each ``TraceMetadata`` key has a registered producer."""

    G = graph_canon()
    register_trace(G)

    produced_keys = set()
    for phase_fields in trace.TRACE_FIELDS.values():
        for getter in phase_fields.values():
            produced_keys.update(getter(G).keys())

    missing = set(trace.TraceMetadata.__annotations__) - produced_keys
    assert not missing, f"Trace fields without producers: {sorted(missing)}"


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
    callback_manager.invoke_callbacks(G, CallbackEvent.BEFORE_STEP.value)

    meta = G.graph["history"]["trace_meta"][0]
    assert meta["custom"] == 42
    del trace.TRACE_FIELDS["before"]["custom"]
