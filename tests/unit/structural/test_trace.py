"""Unit tests for trace registration and associated callback fields."""

from types import MappingProxyType

import pytest

from tnfr import trace
from tnfr.callback_utils import CallbackEvent, callback_manager
from tnfr.telemetry.verbosity import (
    TELEMETRY_VERBOSITY_DEFAULT,
    TELEMETRY_VERBOSITY_LEVELS,
    TelemetryVerbosity,
)
from tnfr.trace import (
    CallbackSpec,
    TraceFieldSpec,
    _callback_names,
    gamma_field,
    grammar_field,
    mapping_field,
    register_trace,
    register_trace_field,
)
from tnfr.utils import get_graph_mapping


def test_trace_verbosity_presets_align_with_shared_levels():
    assert trace.TRACE_VERBOSITY_DEFAULT == TELEMETRY_VERBOSITY_DEFAULT
    assert set(trace.TRACE_VERBOSITY_PRESETS) == set(TELEMETRY_VERBOSITY_LEVELS)
    expected = {
        level.value: tuple(
            spec.name for spec in trace.TRACE_FIELD_SPECS if level in spec.tiers
        )
        for level in TelemetryVerbosity
    }
    assert trace.TRACE_VERBOSITY_PRESETS == expected
    for spec in trace.TRACE_FIELD_SPECS:
        assert isinstance(spec, TraceFieldSpec)


def test_trace_field_specs_register_all_fields():
    registered = {
        (phase, name)
        for phase, phase_fields in trace.TRACE_FIELDS.items()
        for name in phase_fields
    }
    expected = {(spec.phase, spec.name) for spec in trace.TRACE_FIELD_SPECS}
    assert registered == expected


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


def test_trace_basic_verbosity_skips_heavy_fields(graph_canon):
    G = graph_canon()
    G.graph["TRACE"]["verbosity"] = "basic"
    register_trace(G)
    callback_manager.invoke_callbacks(G, CallbackEvent.BEFORE_STEP.value)
    callback_manager.invoke_callbacks(G, CallbackEvent.AFTER_STEP.value)

    hist = G.graph["history"]["trace_meta"]
    before, after = hist

    assert before["phase"] == "before"
    assert "gamma" in before and "dnfr_weights" in before
    assert "kuramoto" not in before and "sigma" not in before

    assert after["phase"] == "after"
    assert "kuramoto" not in after
    assert "sigma" not in after
    assert "glyphs" not in after


def test_trace_detailed_verbosity_skips_glyph_counts(graph_canon):
    G = graph_canon()
    G.graph["TRACE"]["verbosity"] = "detailed"
    register_trace(G)
    callback_manager.invoke_callbacks(G, CallbackEvent.BEFORE_STEP.value)
    callback_manager.invoke_callbacks(G, CallbackEvent.AFTER_STEP.value)

    hist = G.graph["history"]["trace_meta"]
    after = hist[1]

    assert after["phase"] == "after"
    assert "kuramoto" in after
    assert "sigma" in after
    assert "glyphs" not in after


def test_trace_debug_verbosity_includes_all_fields(graph_canon):
    G = graph_canon()
    # Debug is the default verbosity; ensure the full capture set executes.
    register_trace(G)
    callback_manager.invoke_callbacks(G, CallbackEvent.BEFORE_STEP.value)
    callback_manager.invoke_callbacks(G, CallbackEvent.AFTER_STEP.value)

    hist = G.graph["history"]["trace_meta"]
    after = hist[1]

    assert after["phase"] == "after"
    assert "kuramoto" in after
    assert "sigma" in after
    assert "glyphs" in after


def test_trace_unknown_verbosity_warns_and_defaults(graph_canon):
    G = graph_canon()
    G.graph["TRACE"]["verbosity"] = "mystery"

    with pytest.warns(UserWarning):
        register_trace(G)
        callback_manager.invoke_callbacks(G, CallbackEvent.BEFORE_STEP.value)
        callback_manager.invoke_callbacks(G, CallbackEvent.AFTER_STEP.value)

    hist = G.graph["history"]["trace_meta"]
    after = hist[1]

    assert after["phase"] == "after"
    assert "glyphs" in after


def test_callback_names_spec():
    """CallbackSpec entries are handled correctly."""

    def foo():
        pass

    names = _callback_names([CallbackSpec("bar", foo), CallbackSpec(None, foo)])
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
    G.graph["TRACE"] = {
        "enabled": True,
        "capture": ["custom"],
        "history_key": "trace_meta",
    }
    register_trace(G)

    def custom_field(G):
        return {"custom": 42}

    register_trace_field("before", "custom", custom_field)
    callback_manager.invoke_callbacks(G, CallbackEvent.BEFORE_STEP.value)

    meta = G.graph["history"]["trace_meta"][0]
    assert meta["custom"] == 42
    del trace.TRACE_FIELDS["before"]["custom"]


def test_trace_capture_override_takes_priority(graph_canon):
    G = graph_canon()
    G.graph["TRACE"] = {
        "enabled": True,
        "verbosity": "basic",
        "capture": ["kuramoto"],
        "history_key": "trace_meta",
    }
    register_trace(G)
    callback_manager.invoke_callbacks(G, CallbackEvent.AFTER_STEP.value)

    meta = G.graph["history"]["trace_meta"][0]
    assert "kuramoto" in meta
    assert "sigma" not in meta
    assert "glyphs" not in meta


def test_trace_capture_accepts_glyph_alias(graph_canon):
    G = graph_canon()
    G.graph["TRACE"] = {"capture": ["glyphs"]}

    register_trace(G)
    callback_manager.invoke_callbacks(G, CallbackEvent.AFTER_STEP.value)

    meta = G.graph["history"]["trace_meta"][0]
    assert "glyphs" in meta
    assert isinstance(meta["glyphs"], dict)
