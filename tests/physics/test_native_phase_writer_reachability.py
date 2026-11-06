"""Reachable native glyph writes under fresh uniform-capacity default Si.

Each input is an independent declared prism state. Counters and legacy debt
are test inputs, not a reconstructed native history. A single glyph batch or
operator call exercises the actual selector, grammar and primitive dispatcher;
no runtime step, coordinator, integration, or Mutation campaign is executed.
Scalar/NumPy Si equivalence is already covered by uniform-capacity tests.
"""

from copy import deepcopy

from tests.physics._internal_mode_fixture import _nonrepeated_phase_geometry
from tnfr.alias import get_attr
from tnfr.config.operator_names import U2_DEBT_CAPACITY
from tnfr.constants import inject_defaults
from tnfr.constants.aliases import (
    ALIAS_DNFR,
    ALIAS_EPI,
    ALIAS_SI,
    ALIAS_THETA,
    ALIAS_VF,
)
from tnfr.dynamics.dnfr import default_compute_delta_nfr
from tnfr.dynamics.selectors import DefaultGlyphSelector, _apply_glyphs
from tnfr.metrics.sense_index import compute_Si, get_Si_weights
from tnfr.operators import apply_glyph
from tnfr.operators.coherence import Coherence
from tnfr.operators.grammar_debt import U2_DEBT_KEY
from tnfr.operators.grammar_dynamics import enforce_grammar_on_glyph, validate_candidate
from tnfr.types import real_scalar_epi


def _prepared():
    _, graph, _, phases = _nonrepeated_phase_geometry()
    inject_defaults(graph)
    for node, phase in zip(graph, phases, strict=True):
        graph.nodes[node]["theta"] = float(phase)
        graph.nodes[node]["glyph_history"] = []
    # The shared fixture's pure-EPI coefficient is retained. The selector,
    # Si weights, lag limits, grammar and glyph factors use their defaults.
    default_compute_delta_nfr(graph)
    compute_Si(graph, inplace=True)
    alpha, beta, gamma = get_Si_weights(graph)
    assert alpha > 0.5 and beta >= 0 and gamma >= 0
    assert all(get_attr(graph.nodes[node], ALIAS_VF) == 1.0 for node in graph)
    assert all(get_attr(graph.nodes[node], ALIAS_SI) >= alpha for node in graph)
    assert graph.graph["GRAMMAR_CANON"]["enabled"] is True
    return graph


def _state(graph):
    def values(aliases):
        return tuple(
            get_attr(graph.nodes[node], aliases, strict=True) for node in graph
        )

    return {
        "phase": values(ALIAS_THETA),
        "capacity": values(ALIAS_VF),
        "pressure": values(ALIAS_DNFR),
        "epi": tuple(
            real_scalar_epi(
                get_attr(
                    graph.nodes[node], ALIAS_EPI, conv=lambda value: value, strict=True
                )
            )
            for node in graph
        ),
        "nodes": tuple(graph),
        "edges": tuple((i, j, deepcopy(data)) for i, j, data in graph.edges(data=True)),
    }


def _assert_no_phase_capacity_or_support_write(before, after):
    for channel in ("phase", "capacity", "nodes", "edges"):
        assert after[channel] == before[channel]


def _last_glyphs(graph):
    return tuple(
        getattr(
            graph.nodes[node]["glyph_history"][-1],
            "value",
            graph.nodes[node]["glyph_history"][-1],
        )
        for node in graph
    )


def test_actual_native_batch_uses_il_or_lag_al_en_without_phase_renewal():
    graph = _prepared()
    nodes = tuple(graph)
    window = graph.graph["GLYPH_HYSTERESIS_WINDOW"]
    for node in nodes[1:]:
        graph.nodes[node]["glyph_history"] = ["IL"] * window
    selector = DefaultGlyphSelector()
    selector.prepare(graph, nodes)
    assert tuple(selector(graph, node) for node in nodes) == ("IL",) * 6
    assert all(validate_candidate(graph, node, "IL").allowed for node in nodes)

    al_max, en_max = graph.graph["AL_MAX_LAG"], graph.graph["EN_MAX_LAG"]
    history = {
        "since_AL": dict(
            zip(nodes, (0, 0, al_max, al_max, al_max - 1, 0), strict=True)
        ),
        "since_EN": dict(
            zip(nodes, (0, en_max, 0, en_max, en_max - 1, 0), strict=True)
        ),
    }
    expected = ("IL", "EN", "AL", "AL", "IL", "IL")
    assert all(
        validate_candidate(graph, node, code).allowed
        for node, code in zip(nodes, expected, strict=True)
    )
    before = _state(graph)
    _apply_glyphs(graph, selector, history)
    after = _state(graph)
    assert _last_glyphs(graph) == expected
    _assert_no_phase_capacity_or_support_write(before, after)

    # Check actual native effects as well as preservation: IL contracts
    # stored pressure; AL/EN change EPI and leave that pressure untouched.
    factor = graph.graph["GLYPH_FACTORS"]["IL_dnfr_factor"]
    for index, code in enumerate(expected):
        if code == "IL":
            assert after["pressure"][index] == before["pressure"][index] * factor
            assert after["epi"][index] == before["epi"][index]
        else:
            assert after["pressure"][index] == before["pressure"][index]
            assert after["epi"][index] > before["epi"][index]
    # The native batch increments before testing strict limits. AL wins
    # when both limits are crossed, while equality does not force a glyph.
    assert history["since_EN"][nodes[1]] == 0
    assert history["since_AL"][nodes[2]] == history["since_AL"][nodes[3]] == 0
    assert history["since_EN"][nodes[3]] == en_max
    assert history["since_AL"][nodes[4]] == al_max
    assert history["since_EN"][nodes[4]] == en_max


def test_initialized_legacy_debt_falls_back_to_il_before_other_priority_entries():
    graph = _prepared()
    nodes = tuple(graph)
    debt = U2_DEBT_CAPACITY + 2
    for node in nodes:
        graph.nodes[node]["glyph_history"] = ["IL"] * graph.graph[
            "GLYPH_HYSTERESIS_WINDOW"
        ]
        # Persisted debt can exceed what a bounded retained history shows.
        # This declared legacy input tests permitted reduction, not a claim
        # that an admitted default execution generated excess debt.
        graph.nodes[node][U2_DEBT_KEY] = debt
        assert validate_candidate(graph, node, "IL").allowed
        for candidate in ("AL", "EN"):
            result = validate_candidate(graph, node, candidate)
            assert not result.allowed
            assert any(violation.rule == "U2" for violation in result.violations)
            assert enforce_grammar_on_glyph(graph, node, candidate) == "IL"

    history = {
        "since_AL": {node: graph.graph["AL_MAX_LAG"] for node in nodes[:3]},
        "since_EN": {node: graph.graph["EN_MAX_LAG"] for node in nodes[3:]},
    }
    before = _state(graph)
    _apply_glyphs(graph, DefaultGlyphSelector(), history)
    after = _state(graph)
    assert _last_glyphs(graph) == ("IL",) * 6
    _assert_no_phase_capacity_or_support_write(before, after)
    assert after["epi"] == before["epi"]
    assert all(graph.nodes[node][U2_DEBT_KEY] == debt - 1 for node in graph)


def test_public_coherence_phase_stage_is_not_the_native_il_primitive():
    native = _prepared()
    public = _prepared()
    node = next(iter(native))
    before = _state(native)
    assert _state(public) == before
    assert validate_candidate(native, node, "IL").allowed

    apply_glyph(native, node, "IL")
    Coherence()(public, node)
    native_after, public_after = _state(native), _state(public)
    _assert_no_phase_capacity_or_support_write(before, native_after)
    assert native_after["pressure"] == public_after["pressure"]
    assert native_after["epi"] == public_after["epi"] == before["epi"]
    assert public_after["phase"][0] != before["phase"][0]
    assert public_after["phase"][1:] == before["phase"][1:]
    assert public_after["capacity"] == before["capacity"]
    assert public_after["edges"] == before["edges"]
    # The public lifecycle commits its extra bound phase proposal. Default
    # native apply_glyph uses the primitive registry and has no such stage.
    assert "IL_phase_locking" not in native.graph
    assert public.graph["IL_phase_locking"]
