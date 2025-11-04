"""Validation tests for the grammar configuration schema."""

from __future__ import annotations

import json
from importlib import resources

import networkx as nx
import pytest

from tnfr.constants import DEFAULTS
from tnfr.validation import (
    GrammarConfigurationError,
    GrammarContext,
    enforce_canonical_grammar,
)
from tnfr.types import Glyph

jsonschema = pytest.importorskip("jsonschema")

@pytest.fixture(scope="module")
def grammar_schema():
    data = resources.files("tnfr.schemas").joinpath("grammar.json").read_text("utf-8")
    return json.loads(data)

@pytest.fixture(scope="module")
def soft_validator(grammar_schema):
    schema = dict(grammar_schema["definitions"]["cfg_soft"])
    schema.setdefault("definitions", grammar_schema["definitions"])
    return jsonschema.Draft7Validator(schema)

@pytest.fixture(scope="module")
def canon_validator(grammar_schema):
    schema = dict(grammar_schema["definitions"]["cfg_canon"])
    schema.setdefault("definitions", grammar_schema["definitions"])
    return jsonschema.Draft7Validator(schema)

def _graph_with_configs(cfg_soft, cfg_canon):
    G = nx.DiGraph()
    G.add_node(0, glyph_history=[])
    G.graph["GRAMMAR"] = cfg_soft
    G.graph["GRAMMAR_CANON"] = cfg_canon
    return G

def test_schema_accepts_minimal_configs(soft_validator, canon_validator):
    soft_validator.validate({})
    canon_validator.validate({})

def test_schema_accepts_default_configs(soft_validator, canon_validator):
    soft_validator.validate(DEFAULTS.get("GRAMMAR", {}))
    canon_validator.validate(DEFAULTS.get("GRAMMAR_CANON", {}))

def test_context_valid_configuration_passes_and_integrates(monkeypatch):
    monkeypatch.setenv("TNFR_GRAMMAR_VALIDATE", "1")
    cfg_soft = {
        "window": 4,
        "avoid_repeats": ["ZHIR", "OZ"],
        "fallbacks": {"ZHIR": "NAV"},
        "force_dnfr": 0.75,
        "force_accel": 0.80,
    }
    cfg_canon = {
        "zhir_requires_oz_window": 5,
        "zhir_dnfr_min": 0.07,
        "thol_min_len": 2,
        "thol_max_len": 6,
        "thol_close_dnfr": 0.20,
        "si_high": 0.7,
    }
    ctx = GrammarContext.from_graph(_graph_with_configs(cfg_soft, cfg_canon))
    assert ctx.cfg_soft["window"] == 4
    # ensure the automaton pipeline still accepts the validated context
    result = enforce_canonical_grammar(ctx.G, 0, Glyph.AL, ctx=ctx)
    assert result == Glyph.AL

def test_context_invalid_soft_window_raises(monkeypatch):
    monkeypatch.setenv("TNFR_GRAMMAR_VALIDATE", "1")
    cfg_soft = {"window": -1}
    cfg_canon = DEFAULTS.get("GRAMMAR_CANON", {})
    with pytest.raises(GrammarConfigurationError) as excinfo:
        GrammarContext.from_graph(_graph_with_configs(cfg_soft, cfg_canon))
    message = str(excinfo.value)
    assert "cfg_soft" in message
    assert "window" in message

def test_context_invalid_canon_type_raises(monkeypatch):
    monkeypatch.setenv("TNFR_GRAMMAR_VALIDATE", "1")
    cfg_soft = DEFAULTS.get("GRAMMAR", {})
    cfg_canon = "not-a-mapping"
    with pytest.raises(GrammarConfigurationError) as excinfo:
        GrammarContext.from_graph(_graph_with_configs(cfg_soft, cfg_canon))
    message = str(excinfo.value)
    assert "cfg_canon" in message
    assert "mapping" in message

def test_validation_can_be_disabled(monkeypatch):
    monkeypatch.setenv("TNFR_GRAMMAR_VALIDATE", "0")
    cfg_soft = {"window": -3}
    cfg_canon = {"thol_min_len": 4, "thol_max_len": 3}
    ctx = GrammarContext.from_graph(_graph_with_configs(cfg_soft, cfg_canon))
    assert ctx.cfg_soft["window"] == -3
    # disable flag should permit enforcing grammar without raising
    result = enforce_canonical_grammar(ctx.G, 0, Glyph.AL, ctx=ctx)
    assert result == Glyph.AL

def test_enforce_raises_when_validation_fails(monkeypatch):
    monkeypatch.setenv("TNFR_GRAMMAR_VALIDATE", "1")
    cfg_soft = DEFAULTS.get("GRAMMAR", {})
    cfg_canon = {"thol_min_len": 5, "thol_max_len": 1}
    with pytest.raises(GrammarConfigurationError):
        enforce_canonical_grammar(
            _graph_with_configs(cfg_soft, cfg_canon),
            0,
            Glyph.AL,
        )
