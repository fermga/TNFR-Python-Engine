"""A prepared proposal does not authorize stale live operator admission."""

from copy import deepcopy

import networkx as nx
import pytest

from tnfr.constants import inject_defaults
from tnfr.dynamics import selectors
from tnfr.types import require_finite_real_scalar_epi
from tnfr.validation import StructuralGrammarError


def _source(count=1):
    graph = nx.empty_graph(count)
    inject_defaults(graph)
    graph.graph.update(
        GRAMMAR_CANON={"enabled": False},
        GRAMMAR_REJECTION_MODE="raise",
        AL_MAX_LAG=100,
        EN_MAX_LAG=100,
    )
    for data in graph.nodes.values():
        data.update(EPI=0.5, nu_f=1.0, theta=0.0, delta_nfr=0.2, glyph_history=["AL"])
    return graph


class _HookSelector(selectors.AbstractSelector):
    def __init__(self, boundary, change, proposal="IL"):
        self.boundary = boundary
        self.change = change
        self.proposal = proposal
        self.calls = 0

    def prepare(self, graph, nodes):
        if self.boundary == "prepare":
            self.change(graph)

    def select(self, graph, node):
        self.calls += 1
        if self.boundary == "select" and self.calls == 1:
            self.change(graph)
        return self.proposal


@pytest.mark.parametrize("boundary", ["prepare", "select"])
def test_live_grammar_enabled_by_hook_rejects_unhandled_dissonance(boundary):
    graph, history = _source(), {}
    before = deepcopy(dict(graph.nodes))

    def enable(subject):
        subject.graph["GRAMMAR_CANON"]["enabled"] = True

    with pytest.raises(StructuralGrammarError):
        selectors._apply_glyphs(graph, _HookSelector(boundary, enable, "OZ"), history)
    assert graph.graph["GRAMMAR_CANON"]["enabled"] is True  # Hook is not rolled back.
    assert graph.nodes[0]["delta_nfr"] == before[0]["delta_nfr"]
    assert graph.nodes[0]["glyph_history"] == before[0]["glyph_history"]
    assert history["since_AL"][0] == history["since_EN"][0] == 1


@pytest.mark.parametrize("boundary", ["prepare", "select"])
@pytest.mark.parametrize(
    "key,value",
    [
        ("GLYPH_HYSTERESIS_WINDOW", True),
        ("GRAMMAR_CANON", {"enabled": "unrecognized"}),
        ("AL_MAX_LAG", -1),
        ("_t", float("nan")),
    ],
)
def test_changed_invalid_admission_rejects_before_any_glyph(
    monkeypatch, boundary, key, value
):
    graph, history = _source(), {}

    def unexpected(*args, **kwargs):
        pytest.fail("changed invalid admission reached an operator")

    monkeypatch.setattr(selectors, "apply_glyph", unexpected)
    selector = _HookSelector(boundary, lambda g: g.graph.__setitem__(key, value))
    with pytest.raises((TypeError, ValueError)):
        selectors._apply_glyphs(graph, selector, history)
    assert graph.nodes[0]["glyph_history"] == ["AL"]
    if boundary == "prepare":
        assert history == {}
    else:
        assert history["since_AL"][0] == history["since_EN"][0] == 1


def test_changed_late_target_context_rejects_the_batch_before_execution(monkeypatch):
    graph, history = _source(2), {}

    def change(subject):
        subject.nodes[1]["_grammar_u2_debt"] = True

    def unexpected(*args, **kwargs):
        pytest.fail("an invalid late target authorized an earlier glyph")

    monkeypatch.setattr(selectors, "apply_glyph", unexpected)
    with pytest.raises((TypeError, ValueError), match="nonnegative integer"):
        selectors._apply_glyphs(graph, _HookSelector("select", change), history)
    assert all(data["glyph_history"] == ["AL"] for data in graph.nodes.values())


@pytest.mark.parametrize("boundary", ["prepare", "select"])
@pytest.mark.parametrize("change_kind", ["add", "remove", "reorder"])
def test_hooks_cannot_silently_replace_the_ordered_target_population(
    boundary, change_kind
):
    graph, history = _source(2), {}

    def change(subject):
        if change_kind == "add":
            subject.add_node(2)
        elif change_kind == "remove":
            subject.remove_node(1)
        else:
            data = dict(subject.nodes[0])
            subject.remove_node(0)
            subject.add_node(0, **data)

    with pytest.raises(ValueError, match="ordered batch targets"):
        selectors._apply_glyphs(graph, _HookSelector(boundary, change), history)
    # Custom structural side effects remain visible; no implicit transaction.
    assert all(
        data.get("glyph_history", []) in ([], ["AL"]) for data in graph.nodes.values()
    )


def test_preparation_changes_forced_lag_policy_before_the_single_increment():
    graph, history = _source(), {}

    def change(subject):
        subject.graph["AL_MAX_LAG"] = 0

    selector = _HookSelector("prepare", change)
    selectors._apply_glyphs(graph, selector, history)
    assert selector.calls == 0  # Fresh policy forces AL without a second selection.
    assert graph.nodes[0]["glyph_history"][-1] == "AL"
    assert require_finite_real_scalar_epi(graph.nodes[0]["EPI"], "emitted form") > 0.5
    assert history["since_AL"][0] == 0
    assert history["since_EN"][0] == 1


def test_post_selection_uses_current_counter_owners_without_double_increment():
    graph = _source()
    old_al, old_en = {0: 0}, {0: 0}
    new_al, new_en = {0: 7}, {0: 8}
    history = {"since_AL": old_al, "since_EN": old_en}

    def change(subject):
        history.update(since_AL=new_al, since_EN=new_en)

    selectors._apply_glyphs(graph, _HookSelector("select", change, "AL"), history)
    assert history["since_AL"] is new_al and new_al == {0: 0}
    assert history["since_EN"] is new_en and new_en == {0: 8}
    assert old_al == old_en == {0: 1}


def test_live_window_is_read_after_a_legacy_callable_proposal(monkeypatch):
    graph, history, windows = _source(), {}, []
    original = selectors.apply_glyph

    def select(subject, node):
        subject.graph["GLYPH_HYSTERESIS_WINDOW"] = 2
        return "IL"

    def apply(subject, node, glyph, **kwargs):
        windows.append(kwargs["window"])
        return original(subject, node, glyph, **kwargs)

    monkeypatch.setattr(selectors, "apply_glyph", apply)
    selectors._apply_glyphs(graph, select, history)
    assert windows == [2]
    assert history["since_AL"] == history["since_EN"] == {0: 1}
    assert graph.nodes[0]["delta_nfr"] < 0.2
