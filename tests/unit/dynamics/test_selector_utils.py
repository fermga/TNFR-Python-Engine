"""Unit tests for glyph selector utility helpers and resource handling."""

import gc
import weakref

import pytest

import tnfr.selector as selector
from tnfr.constants import DEFAULTS, get_aliases
from tnfr.dynamics import _configure_selector_weights
from tnfr.dynamics.selectors import ParametricGlyphSelector
from tnfr.selector import (
    _apply_selector_hysteresis,
    _calc_selector_score,
    _selector_norms,
    _selector_thresholds,
)
from tnfr.utils import normalize_weights

ALIAS_DNFR = get_aliases("DNFR")
ALIAS_D2EPI = get_aliases("D2EPI")
ALIAS_SI = get_aliases("SI")


def test_selector_thresholds_defaults(graph_canon):
    G = graph_canon()
    thr = _selector_thresholds(G)
    sel_def = DEFAULTS["SELECTOR_THRESHOLDS"]
    assert thr["si_hi"] == sel_def["si_hi"]
    assert thr["si_lo"] == sel_def["si_lo"]
    assert thr["dnfr_hi"] == sel_def["dnfr_hi"]
    assert thr["dnfr_lo"] == sel_def["dnfr_lo"]
    assert thr["accel_hi"] == sel_def["accel_hi"]
    assert thr["accel_lo"] == sel_def["accel_lo"]


def test_selector_thresholds_recreate_defaults_when_missing(graph_canon):
    G = graph_canon()
    del G.graph["SELECTOR_THRESHOLDS"]
    thr = _selector_thresholds(G)
    assert thr == DEFAULTS["SELECTOR_THRESHOLDS"]


def test_selector_thresholds_ignore_glyph_thresholds(graph_canon):
    G = graph_canon()
    G.graph["GLYPH_THRESHOLDS"] = {"hi": 0.9, "lo": 0.2}
    thr = _selector_thresholds(G)
    assert thr == DEFAULTS["SELECTOR_THRESHOLDS"]


def test_selector_thresholds_applies_overrides(graph_canon):
    G = graph_canon()
    overrides = {
        "si_hi": 0.9,
        "si_lo": 0.2,
        "dnfr_hi": 0.7,
        "dnfr_lo": 0.2,
        "accel_hi": 0.8,
        "accel_lo": 0.1,
    }
    G.graph["SELECTOR_THRESHOLDS"] = overrides
    assert _selector_thresholds(G) == overrides


def test_selector_thresholds_cached_per_graph(graph_canon):
    """Repeated calls should reuse cached thresholds for the same graph."""
    G = graph_canon()
    thr1 = _selector_thresholds(G)
    thr2 = _selector_thresholds(G)
    assert thr1 is thr2


def test_selector_thresholds_cache_ignores_dict_order(graph_canon):
    """Changing insertion order should not break cached thresholds."""
    G = graph_canon()
    G.graph["SELECTOR_THRESHOLDS"] = {"si_hi": 0.9, "si_lo": 0.2}
    thr1 = _selector_thresholds(G)
    # Reassign with reversed insertion order
    G.graph["SELECTOR_THRESHOLDS"] = {"si_lo": 0.2, "si_hi": 0.9}
    thr2 = _selector_thresholds(G)
    assert thr1 is thr2


def test_selector_thresholds_cache_releases_graph(graph_canon):
    """The selector cache must not keep graphs alive once discarded."""

    selector._SELECTOR_THRESHOLD_CACHE.clear()
    G = graph_canon()
    selector._selector_thresholds(G)
    assert len(selector._SELECTOR_THRESHOLD_CACHE) == 1

    ref = weakref.ref(G)
    del G
    gc.collect()

    assert ref() is None
    assert len(selector._SELECTOR_THRESHOLD_CACHE) == 0


def test_selector_norms_computes_max(graph_canon):
    G = graph_canon()
    G.add_node(0, **{ALIAS_DNFR[-1]: 2.0, ALIAS_D2EPI[-2]: 1.0})
    G.add_node(1, **{ALIAS_DNFR[-1]: -3.0, ALIAS_D2EPI[-2]: 0.5})
    norms = _selector_norms(G)
    assert norms == G.graph["_sel_norms"]
    assert norms["dnfr_max"] == 3.0
    assert norms["accel_max"] == 1.0


def test_calc_selector_score_assumes_normalized_weights():
    W_raw = {"w_si": 0.5, "w_dnfr": 0.3, "w_accel": 0.2}
    W = normalize_weights(W_raw, W_raw.keys())
    assert _calc_selector_score(1.0, 0.0, 0.0, W) == 1.0
    assert _calc_selector_score(0.0, 1.0, 1.0, W) == 0.0


def test_configure_selector_weights_normalizes(graph_canon):
    G = graph_canon()
    G.graph["SELECTOR_WEIGHTS"] = {"w_si": 2.0, "w_dnfr": 1.0, "w_accel": 1.0}
    weights = _configure_selector_weights(G)
    assert weights == pytest.approx({"w_si": 0.5, "w_dnfr": 0.25, "w_accel": 0.25})
    assert G.graph["_selector_weights"] == weights


def test_apply_selector_hysteresis_returns_prev():
    thr = DEFAULTS["SELECTOR_THRESHOLDS"]
    nd = {"glyph_history": ["RA"]}
    # near si_hi threshold
    prev = _apply_selector_hysteresis(nd, thr["si_hi"] - 0.01, 0.2, 0.2, thr, 0.05)
    assert prev == "RA"
    # far from thresholds
    none = _apply_selector_hysteresis(nd, 0.5, 0.2, 0.2, thr, 0.05)
    assert none is None


def test_parametric_selector_skips_hysteresis_with_none_margin(graph_canon):
    G = graph_canon()
    thr = DEFAULTS["SELECTOR_THRESHOLDS"]
    node = 0
    G.add_node(
        node,
        glyph_history=["IL"],
        **{
            ALIAS_SI[0]: thr["si_lo"],
            ALIAS_DNFR[-1]: 0.9,
            ALIAS_D2EPI[-1]: 0.0,
        },
    )
    G.graph["GLYPH_SELECTOR_MARGIN"] = None
    G.graph["_sel_norms"] = {"dnfr_max": 1.0, "accel_max": 1.0}

    selector = ParametricGlyphSelector()
    glyph = selector.select(G, node)

    assert glyph == "OZ"


def test_soft_grammar_prefilter_respects_avoid_repeats(graph_canon):
    G = graph_canon()
    thresholds = {
        "si_hi": 0.9,
        "si_lo": 0.1,
        "dnfr_hi": 0.95,
        "dnfr_lo": 0.05,
        "accel_hi": 0.95,
        "accel_lo": 0.05,
    }
    G.graph["SELECTOR_THRESHOLDS"] = thresholds
    G.graph["_sel_norms"] = {"dnfr_max": 1.0, "accel_max": 1.0}
    G.graph["GLYPH_SELECTOR_MARGIN"] = 0.0
    G.graph["GRAMMAR"] = {
        "window": 3,
        "avoid_repeats": ["RA"],
        "fallbacks": {"RA": "OZ"},
        "force_dnfr": 0.8,
        "force_accel": 0.7,
    }

    node = 0
    G.add_node(
        node,
        glyph_history=["RA"],
        **{
            ALIAS_SI[0]: 0.5,
            ALIAS_DNFR[-1]: 0.2,
            ALIAS_D2EPI[-1]: 0.2,
        },
    )

    selector = ParametricGlyphSelector()
    fallback = selector.select(G, node)
    assert fallback == "OZ"

    G.nodes[node].update(
        {
            ALIAS_DNFR[-1]: 0.85,
            ALIAS_D2EPI[-1]: 0.2,
            "glyph_history": ["RA"],
        }
    )

    forced = selector.select(G, node)
    assert forced == "RA"
