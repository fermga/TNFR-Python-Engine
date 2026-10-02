"""Configured selector decisions use admitted observations and explicit batches."""

from copy import deepcopy
from fractions import Fraction

import networkx as nx
import pytest

from tnfr.constants import inject_defaults
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_SI
from tnfr.dynamics import selectors
from tnfr.selector import _selector_thresholds


def _graph():
    graph = nx.path_graph(2)
    inject_defaults(graph)
    graph.graph.update(
        SELECTOR_THRESHOLDS={"si_hi": 0.8, "si_lo": 0.2, "dnfr_hi": 0.5},
        GLYPH_SELECTOR_MARGIN=0.0,
        GRAMMAR_CANON={"enabled": False},
        AL_MAX_LAG=100,
        EN_MAX_LAG=100,
    )
    for node, pressure in enumerate((0.1, 1.0)):
        graph.nodes[node].update({ALIAS_SI[0]: 0.5, ALIAS_DNFR[0]: pressure})
    return graph


@pytest.mark.parametrize(
    "selector_type,prepared,value",
    [
        (selectors.DefaultGlyphSelector, False, True),
        (selectors.DefaultGlyphSelector, True, "0.9"),
        (selectors.ParametricGlyphSelector, False, float("nan")),
        (selectors.ParametricGlyphSelector, True, Fraction(1, 2**2000)),
    ],
)
def test_invalid_authoritative_sense_cannot_become_an_operator_decision(
    selector_type, prepared, value
):
    graph = _graph()
    graph.nodes[0][ALIAS_SI[0]] = value
    graph.nodes[0][ALIAS_SI[1]] = 0.9
    selector = selector_type()
    with pytest.raises((TypeError, ValueError)):
        if prepared:
            selector.prepare(graph, list(graph))
        selector(graph, 0)


@pytest.mark.parametrize("prepared", [False, True])
def test_each_new_decision_or_prepare_refreshes_pressure_normalization(prepared):
    graph = _graph()
    selector = selectors.DefaultGlyphSelector()
    if prepared:
        selector.prepare(graph, list(graph))
    assert selector(graph, 0) == "RA"  # |p0| / max|p| = 0.1
    graph.nodes[1][ALIAS_DNFR[0]] = 0.1
    if prepared:
        selector.prepare(graph, list(graph))
    assert selector(graph, 0) == "NAV"  # The refreshed ratio is 1.


def test_scalar_and_array_preselection_share_raw_admission(monkeypatch):
    graph = _graph()
    graph.nodes[1][ALIAS_SI[0]] = "0.9"
    monkeypatch.setattr(selectors, "np", None)
    with pytest.raises(TypeError):
        selectors.DefaultGlyphSelector().prepare(graph, list(graph))


def test_invalid_late_metric_stops_batch_before_any_operator_or_lag_write(monkeypatch):
    graph = _graph()
    graph.nodes[1][ALIAS_SI[0]] = True
    before = deepcopy(dict(graph.nodes(data=True)))
    history = {}

    def unexpected(*args, **kwargs):
        pytest.fail("invalid selection input reached operator execution")

    monkeypatch.setattr(selectors, "apply_glyph", unexpected)
    selector = selectors.DefaultGlyphSelector()
    with pytest.raises(TypeError):
        selectors._apply_glyphs(graph, selector, history)
    assert dict(graph.nodes(data=True)) == before
    assert history == {}
    assert selector._preselection is None


@pytest.mark.parametrize("fail", [False, True])
def test_batch_snapshot_is_released_after_application_or_failure(monkeypatch, fail):
    graph = _graph()
    selector = selectors.DefaultGlyphSelector()
    selected = []

    def apply(graph, node, glyph, **kwargs):
        selected.append((node, glyph))
        for data in graph.nodes.values():
            data[ALIAS_SI[0]] = 0.9
        if fail:
            raise RuntimeError("application failed")

    monkeypatch.setattr(selectors, "apply_glyph", apply)
    if fail:
        with pytest.raises(RuntimeError, match="application failed"):
            selectors._apply_glyphs(graph, selector, {})
    else:
        selectors._apply_glyphs(graph, selector, {})
        assert selected == [(0, "RA"), (1, "NAV")]
    assert selector._preselection is None
    assert selector(graph, 0) == "IL"  # New call reads current Si, not the batch.


def test_failed_manual_prepare_does_not_retain_an_older_successful_snapshot():
    graph = _graph()
    selector = selectors.DefaultGlyphSelector()
    selector.prepare(graph, list(graph))
    graph.nodes[0][ALIAS_SI[0]] = True
    with pytest.raises(TypeError):
        selector.prepare(graph, list(graph))
    assert selector._preselection is None


def test_score_weights_follow_current_configuration_between_unprepared_calls():
    graph = _graph()
    weights = {"w_si": 1.0, "w_dnfr": 0.0, "w_accel": 0.0}
    graph.graph["SELECTOR_WEIGHTS"] = weights
    score = selectors._compute_selector_score
    assert score(graph, graph.nodes[0], 0.8, 0.2, 0.9, "RA") == 0.8
    weights.update(w_si=0.0, w_accel=1.0)
    assert score(graph, graph.nodes[0], 0.8, 0.2, 0.9, "RA") == pytest.approx(0.1)


def test_prepared_score_weights_are_one_detached_batch_policy():
    graph = _graph()
    graph.graph["SELECTOR_WEIGHTS"] = {"w_si": 1.0, "w_dnfr": 0.0, "w_accel": 0.0}
    selector = selectors.ParametricGlyphSelector()
    selector.prepare(graph, list(graph))
    graph.graph["SELECTOR_WEIGHTS"]["w_si"] = 0.0
    assert selector._preselection.weights == {
        "w_si": 1.0,
        "w_dnfr": 0.0,
        "w_accel": 0.0,
    }


def test_explicitly_disabled_hysteresis_remains_disabled_in_prepared_batch(monkeypatch):
    graph = _graph()
    graph.graph["GLYPH_SELECTOR_MARGIN"] = None
    selector = selectors.ParametricGlyphSelector()
    selector.prepare(graph, list(graph))
    graph.graph["GLYPH_SELECTOR_MARGIN"] = 1.0
    consumed = []
    real_hysteresis = selectors._apply_selector_hysteresis

    def observe(*args):
        consumed.append(args[-1])
        return real_hysteresis(*args)

    monkeypatch.setattr(selectors, "_apply_selector_hysteresis", observe)
    selector(graph, 0)
    assert consumed == [None]


@pytest.mark.parametrize("sign", [-1, 1])
def test_nonzero_threshold_cannot_be_coerced_to_a_zero_policy(sign):
    graph = _graph()
    graph.graph["SELECTOR_THRESHOLDS"]["si_hi"] = sign * Fraction(1, 2**2000)
    with pytest.raises(ValueError, match="si_hi.*underflows"):
        _selector_thresholds(graph)


@pytest.mark.parametrize("prepared", [False, True])
def test_invalid_score_policy_cannot_be_hidden_by_hysteresis(prepared):
    graph = _graph()
    graph.graph["GLYPH_SELECTOR_MARGIN"] = 1.0
    graph.graph["SELECTOR_WEIGHTS"] = {"w_si": "bad", "w_dnfr": -5.0, "w_accel": 1.0}
    graph.nodes[0]["glyph_history"] = ["IL"]
    selector = selectors.ParametricGlyphSelector()
    with pytest.raises((TypeError, ValueError), match="SELECTOR_WEIGHTS"):
        if prepared:
            selector.prepare(graph, list(graph))
        selector(graph, 0)


def test_worker_preselection_matches_array_decisions_for_one_admitted_snapshot(
    monkeypatch,
):
    graph = _graph()
    reference = selectors.DefaultGlyphSelector()
    reference.prepare(graph, list(graph))
    expected = [reference(graph, node) for node in graph]
    graph.graph["GLYPH_SELECTOR_N_JOBS"] = 2
    monkeypatch.setattr(selectors, "np", None)
    parallel = selectors.DefaultGlyphSelector()
    parallel.prepare(graph, list(graph))
    assert [parallel(graph, node) for node in graph] == expected


def test_invalid_hysteresis_margin_cannot_silently_disable_the_policy():
    graph = _graph()
    graph.graph["GLYPH_SELECTOR_MARGIN"] = float("nan")
    with pytest.raises(ValueError, match="GLYPH_SELECTOR_MARGIN"):
        selectors.ParametricGlyphSelector()(graph, 0)


def test_repetition_feedback_validates_consumed_samples_and_compares_without_overflow():
    graph = _graph()
    graph.nodes[0]["glyph_history"] = ["IL"]
    graph.graph["SELECTOR_WEIGHTS"] = {"w_si": 1.0, "w_dnfr": 0.0, "w_accel": 0.0}
    graph.graph["history"] = {"sense_sigma_mag": [1e308, -1e308]}
    score = selectors._compute_selector_score
    assert score(graph, graph.nodes[0], 0.9, 0.2, 0.1, "IL") == pytest.approx(0.85)
    graph.graph["history"]["sense_sigma_mag"][-1] = True
    with pytest.raises(TypeError):
        score(graph, graph.nodes[0], 0.9, 0.2, 0.1, "IL")


def test_malformed_selector_configuration_does_not_select_the_default_controller():
    graph = _graph()
    graph.graph["glyph_selector"] = "parametric"
    with pytest.raises(TypeError, match="glyph_selector"):
        selectors._apply_selector(graph)


def test_missing_observations_keep_the_declared_selector_baseline():
    graph = _graph()
    for data in graph.nodes.values():
        data.clear()
    assert selectors.DefaultGlyphSelector()(graph, 0) == "RA"
