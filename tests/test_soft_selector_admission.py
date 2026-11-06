"""Original soft-policy inputs cannot silently change selector decisions."""

from copy import deepcopy
from fractions import Fraction
from types import MappingProxyType, SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from tnfr.constants.aliases import ALIAS_D2EPI, ALIAS_DNFR
from tnfr.operators.grammar_context import GrammarContext
from tnfr.types import Glyph
from tnfr.validation.rules import get_norm, glyph_fallback, normalized_dnfr
from tnfr.validation.soft_filters import (
    acceleration_norm,
    check_repeats,
    maybe_force,
    soft_grammar_filters,
)


def _context():
    graph = nx.empty_graph(1)
    graph.nodes[0].update(
        {
            "glyph_history": ["AL", "OZ"],
            ALIAS_DNFR[0]: 0.25,
            ALIAS_D2EPI[0]: 0.0,
        }
    )
    return GrammarContext(
        graph,
        cfg_soft={
            "window": 2,
            "avoid_repeats": ["OZ"],
            "fallbacks": {"OZ": "RA"},
            "force_dnfr": 0.6,
            "force_accel": 0.6,
        },
    )


@pytest.mark.parametrize("key", ["force_dnfr", "force_accel"])
@pytest.mark.parametrize("candidate", ["OZ", "IL"])
@pytest.mark.parametrize(
    "invalid",
    [
        True,
        np.bool_(False),
        "0.6",
        None,
        float("nan"),
        float("inf"),
        Fraction(1, 2**2000),
        -0.01,
        Fraction(1) + Fraction(1, 2**80),
    ],
)
def test_invalid_force_policy_rejects_even_when_no_replacement_is_needed(
    key, candidate, invalid
):
    ctx = _context()
    ctx.cfg_soft[key] = invalid
    before = deepcopy(dict(ctx.G.nodes(data=True)))
    with pytest.raises((TypeError, ValueError), match=key):
        soft_grammar_filters(ctx, 0, candidate)
    assert dict(ctx.G.nodes(data=True)) == before


@pytest.mark.parametrize(
    "score", [True, "0.8", float("nan"), -0.1, Fraction(1) + Fraction(1, 2**80)]
)
def test_public_force_helper_admits_the_consumed_normalized_score(score):
    ctx = _context()
    with pytest.raises((TypeError, ValueError), match="normalized score"):
        maybe_force(ctx, 0, "RA", "OZ", lambda *_: score, "force_dnfr")


@pytest.mark.parametrize(
    "channel,key", [(ALIAS_DNFR, "force_dnfr"), (ALIAS_D2EPI, "force_accel")]
)
@pytest.mark.parametrize(
    "threshold,score,expected",
    [(0.6, 0.59, "RA"), (0.6, 0.6, "OZ"), (0.0, 0.0, "OZ"), (1.0, 1.0, "OZ")],
)
def test_force_comparison_preserves_inclusive_threshold_and_channel_order(
    channel, key, threshold, score, expected
):
    ctx = _context()
    ctx.cfg_soft.update(force_dnfr=1.0, force_accel=1.0)
    ctx.cfg_soft[key] = threshold
    ctx.G.nodes[0][ALIAS_DNFR[0]] = 0.0
    ctx.G.nodes[0][channel[0]] = score
    assert soft_grammar_filters(ctx, 0, "OZ") == expected


def test_absent_force_threshold_keeps_point_six_default():
    ctx = _context()
    ctx.cfg_soft.pop("force_dnfr")
    ctx.G.nodes[0][ALIAS_DNFR[0]] = -0.6
    assert soft_grammar_filters(ctx, 0, "OZ") == "OZ"


@pytest.mark.parametrize("candidate", ["OZ", Glyph.OZ])
def test_filter_preserves_code_or_enum_representation(candidate):
    result = soft_grammar_filters(_context(), 0, candidate)
    assert result == "RA"
    assert type(result) is type(candidate)


def test_explicit_template_controls_representation_without_changing_operator():
    ctx = _context()
    assert soft_grammar_filters(ctx, 0, "OZ", template=Glyph.AL) is Glyph.RA
    result = soft_grammar_filters(ctx, 0, Glyph.OZ, template="AL")
    assert result == "RA" and type(result) is str


@pytest.mark.parametrize("invalid", [None, False, "UNKNOWN", object()])
def test_present_invalid_fallback_cannot_silently_retain_or_supply_an_operator(invalid):
    ctx = _context()
    ctx.cfg_soft["fallbacks"]["OZ"] = invalid
    with pytest.raises(ValueError, match="fallback glyph"):
        soft_grammar_filters(ctx, 0, "OZ")


@pytest.mark.parametrize("history", [["AL", "OZ"], ["Emission", "Dissonance"]])
def test_absent_fallback_retains_candidate_without_granting_repeat_admission(history):
    from tnfr.operators.grammar_types import RepeatWindowError

    assert glyph_fallback("OZ", {}) is Glyph.OZ
    ctx = _context()
    ctx.cfg_soft["fallbacks"] = {}
    ctx.G.nodes[0]["glyph_history"] = history
    before = deepcopy(dict(ctx.G.nodes[0]))
    with pytest.raises(RepeatWindowError) as caught:
        check_repeats(ctx, 0, "OZ")
    assert caught.value.order == ("emission", "dissonance", "dissonance")
    assert caught.value.candidate == "dissonance"
    assert dict(ctx.G.nodes[0]) == before


def test_identifier_aliases_reuse_shared_resolution_for_avoidance_and_fallback():
    ctx = _context()
    ctx.cfg_soft["avoid_repeats"] = ["dissonance"]
    ctx.cfg_soft["fallbacks"]["OZ"] = "resonance"
    assert soft_grammar_filters(ctx, 0, "Glyph.OZ") == "RA"
    assert glyph_fallback("dissonance", {"dissonance": "Glyph.IL"}) is Glyph.IL


@pytest.mark.parametrize("candidate", ["unknown", None, False])
def test_unknown_candidate_is_not_a_soft_permission(candidate):
    with pytest.raises(ValueError, match="candidate glyph"):
        soft_grammar_filters(_context(), 0, candidate)


@pytest.mark.parametrize(
    "field,value",
    [
        ("avoid_repeats", "OZ"),
        ("avoid_repeats", {"OZ": True}),
        ("avoid_repeats", ["UNKNOWN"]),
        ("fallbacks", []),
        ("fallbacks", None),
    ],
)
def test_malformed_soft_collection_cannot_disable_or_rewrite_policy(field, value):
    ctx = _context()
    ctx.cfg_soft[field] = value
    with pytest.raises((TypeError, ValueError)):
        soft_grammar_filters(ctx, 0, "OZ")


def test_one_shot_avoidance_is_rejected_without_consumption():
    ctx = _context()
    source = iter(["OZ"])
    ctx.cfg_soft["avoid_repeats"] = source
    with pytest.raises(TypeError, match="replayable"):
        soft_grammar_filters(ctx, 0, "OZ")
    assert next(source) == "OZ"


def test_normalization_keeps_unit_zero_and_consumer_specific_sign_policies():
    ctx = _context()
    ctx.G.nodes[0][ALIAS_DNFR[0]] = -0.5
    ctx.G.nodes[0][ALIAS_D2EPI[0]] = -0.5
    ctx.G.graph["_sel_norms"] = {"dnfr_max": 100.0, "accel_max": 100.0}
    assert normalized_dnfr(ctx, ctx.G.nodes[0]) == 0.5
    assert acceleration_norm(ctx, ctx.G.nodes[0]) == 0.5
    ctx.norms.update(dnfr_max=0.0, accel_max=-1.0)
    assert get_norm(ctx, "dnfr_max") == 1.0
    assert normalized_dnfr(ctx, ctx.G.nodes[0]) == 0.5
    assert acceleration_norm(ctx, ctx.G.nodes[0]) == 0.0
    ctx.norms["dnfr_max"] = -1.0
    with pytest.raises(ValueError, match="nonnegative"):
        normalized_dnfr(ctx, ctx.G.nodes[0])


@pytest.mark.parametrize("invalid", [False, [], None, 0])
def test_normalization_reader_rejects_malformed_mapping(invalid):
    with pytest.raises(TypeError, match="norms"):
        get_norm(SimpleNamespace(norms=invalid), "dnfr_max")


@pytest.mark.parametrize(
    "channel,key", [(ALIAS_DNFR, "force_dnfr"), (ALIAS_D2EPI, "force_accel")]
)
def test_invalid_authoritative_readout_does_not_use_valid_secondary_alias(channel, key):
    ctx = _context()
    ctx.G.nodes[0][channel[0]] = True
    ctx.G.nodes[0][channel[1]] = 0.8
    accessor = normalized_dnfr if key == "force_dnfr" else acceleration_norm
    with pytest.raises((TypeError, ValueError)):
        maybe_force(ctx, 0, "RA", "OZ", accessor, key)


@pytest.mark.parametrize("field", ["cfg_soft", "cfg_canon", "norms"])
@pytest.mark.parametrize("invalid", [False, 0, [], ""])
def test_context_constructor_rejects_falsy_nonmapping_payloads(field, invalid):
    with pytest.raises(TypeError, match=field):
        GrammarContext(nx.Graph(), **{field: invalid})


@pytest.mark.parametrize("field", ["GRAMMAR", "GRAMMAR_CANON"])
@pytest.mark.parametrize("invalid", [False, 0, [], None, ""])
def test_graph_context_rejects_present_falsy_configuration(field, invalid):
    graph = nx.Graph()
    graph.graph[field] = invalid
    with pytest.raises(TypeError, match=field):
        GrammarContext.from_graph(graph)


def test_context_preserves_constructor_optional_absence_and_live_mapping_ownership(
    monkeypatch,
):
    graph = nx.Graph()
    empty = GrammarContext(graph, cfg_soft=None, cfg_canon=None, norms=None)
    assert empty.cfg_soft == empty.cfg_canon == empty.norms == {}
    policy = {"window": 2}
    graph.graph["GRAMMAR"] = policy
    graph.graph["GRAMMAR_CANON"] = MappingProxyType({"enabled": True})
    monkeypatch.setenv("TNFR_GRAMMAR_VALIDATE", "1")
    ctx = GrammarContext.from_graph(graph)
    assert ctx.cfg_soft is policy
    policy["window"] = 3
    assert ctx.cfg_soft["window"] == 3
    assert ctx.norms == {}


def test_missing_and_empty_graph_configuration_get_independent_default_copies():
    first = GrammarContext.from_graph(nx.Graph())
    graph = nx.Graph()
    graph.graph.update(GRAMMAR={}, GRAMMAR_CANON={})
    second = GrammarContext.from_graph(graph)
    assert first.cfg_soft == second.cfg_soft and first.cfg_soft is not second.cfg_soft
    first.cfg_soft["window"] += 1
    assert first.cfg_soft["window"] != second.cfg_soft["window"]
