"""Pruebas de selector utils."""

import networkx as nx
import pytest

from tnfr.selector import (
    _selector_thresholds,
    _norms_para_selector,
    _calc_selector_score,
    _apply_selector_hysteresis,
)
from tnfr.constants import DEFAULTS, ALIAS_DNFR, ALIAS_D2EPI
from tnfr.constants.core import SELECTOR_THRESHOLD_DEFAULTS
from tnfr.helpers.numeric import clamp01
from tnfr.collections_utils import normalize_weights
from tnfr.dynamics import _configure_selector_weights


def _selector_thresholds_original(G: nx.Graph) -> dict:
    sel_defaults = DEFAULTS.get("SELECTOR_THRESHOLDS", {})
    thr_sel = {**sel_defaults, **G.graph.get("SELECTOR_THRESHOLDS", {})}
    glyph_defaults = DEFAULTS.get("GLYPH_THRESHOLDS", {})
    thr_def = {**glyph_defaults, **G.graph.get("GLYPH_THRESHOLDS", {})}

    si_hi = clamp01(
        float(
            thr_sel.get(
                "si_hi",
                thr_def.get(
                    "hi",
                    glyph_defaults.get(
                        "hi", SELECTOR_THRESHOLD_DEFAULTS["si_hi"]
                    ),
                ),
            )
        )
    )
    si_lo = clamp01(
        float(
            thr_sel.get(
                "si_lo",
                thr_def.get(
                    "lo",
                    glyph_defaults.get(
                        "lo", SELECTOR_THRESHOLD_DEFAULTS["si_lo"]
                    ),
                ),
            )
        )
    )
    dnfr_hi = clamp01(
        float(
            thr_sel.get(
                "dnfr_hi",
                sel_defaults.get(
                    "dnfr_hi", SELECTOR_THRESHOLD_DEFAULTS["dnfr_hi"]
                ),
            )
        )
    )
    dnfr_lo = clamp01(
        float(
            thr_sel.get(
                "dnfr_lo",
                sel_defaults.get(
                    "dnfr_lo", SELECTOR_THRESHOLD_DEFAULTS["dnfr_lo"]
                ),
            )
        )
    )
    acc_hi = clamp01(
        float(
            thr_sel.get(
                "accel_hi",
                sel_defaults.get(
                    "accel_hi", SELECTOR_THRESHOLD_DEFAULTS["accel_hi"]
                ),
            )
        )
    )
    acc_lo = clamp01(
        float(
            thr_sel.get(
                "accel_lo",
                sel_defaults.get(
                    "accel_lo", SELECTOR_THRESHOLD_DEFAULTS["accel_lo"]
                ),
            )
        )
    )

    return {
        "si_hi": si_hi,
        "si_lo": si_lo,
        "dnfr_hi": dnfr_hi,
        "dnfr_lo": dnfr_lo,
        "accel_hi": acc_hi,
        "accel_lo": acc_lo,
    }


def test_selector_thresholds_defaults():
    G = nx.Graph()
    thr = _selector_thresholds(G)
    sel_def = DEFAULTS["SELECTOR_THRESHOLDS"]
    assert thr["si_hi"] == sel_def["si_hi"]
    assert thr["si_lo"] == sel_def["si_lo"]
    assert thr["dnfr_hi"] == sel_def["dnfr_hi"]
    assert thr["dnfr_lo"] == sel_def["dnfr_lo"]
    assert thr["accel_hi"] == sel_def["accel_hi"]
    assert thr["accel_lo"] == sel_def["accel_lo"]


def test_selector_thresholds_refactor_equivalent_defaults():
    G = nx.Graph()
    assert _selector_thresholds(G) == _selector_thresholds_original(G)


def test_selector_thresholds_refactor_equivalent_legacy():
    G = nx.Graph()
    G.graph["GLYPH_THRESHOLDS"] = {"hi": 0.9, "lo": 0.2}
    assert _selector_thresholds(G) == _selector_thresholds_original(G)


def test_selector_thresholds_refactor_equivalent_overrides():
    G = nx.Graph()
    G.graph["SELECTOR_THRESHOLDS"] = {
        "si_hi": 0.9,
        "si_lo": 0.2,
        "dnfr_hi": 0.7,
        "dnfr_lo": 0.2,
        "accel_hi": 0.8,
        "accel_lo": 0.1,
    }
    assert _selector_thresholds(G) == _selector_thresholds_original(G)


def test_norms_para_selector_computes_max():
    G = nx.Graph()
    G.add_node(0, **{ALIAS_DNFR[-1]: 2.0, ALIAS_D2EPI[-2]: 1.0})
    G.add_node(1, **{ALIAS_DNFR[-1]: -3.0, ALIAS_D2EPI[-2]: 0.5})
    norms = _norms_para_selector(G)
    assert norms == G.graph["_sel_norms"]
    assert norms["dnfr_max"] == 3.0
    assert norms["accel_max"] == 1.0


def test_calc_selector_score_assumes_normalized_weights():
    W_raw = {"w_si": 0.5, "w_dnfr": 0.3, "w_accel": 0.2}
    W = normalize_weights(W_raw, W_raw.keys())
    assert _calc_selector_score(1.0, 0.0, 0.0, W) == 1.0
    assert _calc_selector_score(0.0, 1.0, 1.0, W) == 0.0


def test_configure_selector_weights_normalizes():
    G = nx.Graph()
    G.graph["SELECTOR_WEIGHTS"] = {"w_si": 2.0, "w_dnfr": 1.0, "w_accel": 1.0}
    weights = _configure_selector_weights(G)
    assert weights == pytest.approx(
        {"w_si": 0.5, "w_dnfr": 0.25, "w_accel": 0.25}
    )
    assert G.graph["_selector_weights"] == weights


def test_apply_selector_hysteresis_returns_prev():
    thr = DEFAULTS["SELECTOR_THRESHOLDS"]
    nd = {"glyph_history": ["RA"]}
    # near si_hi threshold
    prev = _apply_selector_hysteresis(
        nd, thr["si_hi"] - 0.01, 0.2, 0.2, thr, 0.05
    )
    assert prev == "RA"
    # far from thresholds
    none = _apply_selector_hysteresis(nd, 0.5, 0.2, 0.2, thr, 0.05)
    assert none is None
