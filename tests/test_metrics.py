"""Pruebas de metrics."""

import pytest
from typing import Any

from tnfr.constants import (
    inject_defaults,
    ALIAS_EPI,
    ALIAS_DNFR,
    ALIAS_dEPI,
    ALIAS_SI,
    ALIAS_VF,
)
from tnfr.alias import get_attr, set_attr
from tnfr.metrics import (
    _metrics_step,
    _update_latency_index,
    _update_epi_support,
    _track_stability,
    _aggregate_si,
    _compute_advanced_metrics,
)
from tnfr.metrics.core import LATENT_GLYPH
from tnfr.metrics.core import _update_sigma
from tnfr.constants import METRIC_DEFAULTS
from tnfr.types import Glyph


def test_track_stability_updates_hist(graph_canon):
    """_track_stability aggregates stability and derivatives."""

    G = graph_canon()
    hist = {"stable_frac": [], "delta_Si": [], "B": []}

    G.add_node(0)
    G.add_node(1)

    # Node 0: stable
    set_attr(G.nodes[0], ALIAS_DNFR, 0.0)
    set_attr(G.nodes[0], ALIAS_dEPI, 0.0)
    set_attr(G.nodes[0], ALIAS_SI, 2.0)
    G.nodes[0]["_prev_Si"] = 1.0
    set_attr(G.nodes[0], ALIAS_VF, 1.0)
    G.nodes[0]["_prev_vf"] = 0.5
    G.nodes[0]["_prev_dvf"] = 0.2

    # Node 1: unstable
    set_attr(G.nodes[1], ALIAS_DNFR, 10.0)
    set_attr(G.nodes[1], ALIAS_dEPI, 10.0)
    set_attr(G.nodes[1], ALIAS_SI, 3.0)
    G.nodes[1]["_prev_Si"] = 1.0
    set_attr(G.nodes[1], ALIAS_VF, 1.0)
    G.nodes[1]["_prev_vf"] = 1.0
    G.nodes[1]["_prev_dvf"] = 0.0

    _track_stability(G, hist, dt=1.0, eps_dnfr=1.0, eps_depi=1.0)

    assert hist["stable_frac"] == [0.5]
    assert hist["delta_Si"] == [pytest.approx(1.5)]
    assert hist["B"] == [pytest.approx(0.15)]


def test_update_sigma_uses_default_window(graph_canon):
    G = graph_canon()
    for n in range(2):
        G.add_node(n, glyph_history=[Glyph.AL.value, Glyph.EN.value])
    hist = {}
    G.graph.pop("GLYPH_LOAD_WINDOW", None)
    _update_sigma(G, hist)
    expected = {}
    G.graph["GLYPH_LOAD_WINDOW"] = METRIC_DEFAULTS["GLYPH_LOAD_WINDOW"]
    _update_sigma(G, expected)
    assert hist == expected


def test_aggregate_si_computes_stats(graph_canon):
    """_aggregate_si computes mean and fractions."""

    G = graph_canon()
    inject_defaults(G)
    hist = {"Si_mean": [], "Si_hi_frac": [], "Si_lo_frac": []}
    G.add_node(0)
    G.add_node(1)
    G.add_node(2)
    set_attr(G.nodes[0], ALIAS_SI, 0.2)
    set_attr(G.nodes[1], ALIAS_SI, 0.5)
    set_attr(G.nodes[2], ALIAS_SI, 0.8)

    _aggregate_si(G, hist)

    assert hist["Si_mean"][0] == pytest.approx(0.5)
    assert hist["Si_hi_frac"][0] == pytest.approx(1 / 3)
    assert hist["Si_lo_frac"][0] == pytest.approx(1 / 3)


def test_compute_advanced_metrics_populates_history(graph_canon):
    """_compute_advanced_metrics records glyph-based metrics."""

    G = graph_canon()
    inject_defaults(G)
    hist: dict[str, Any] = {}
    cfg = G.graph["METRICS"]
    thr = float(G.graph.get("EPI_SUPPORT_THR"))

    G.add_node(0)
    set_attr(G.nodes[0], ALIAS_EPI, 0.1)
    G.nodes[0]["glyph_history"] = ["OZ"]

    G.add_node(1)
    set_attr(G.nodes[1], ALIAS_EPI, 0.2)
    G.nodes[1]["glyph_history"] = [LATENT_GLYPH]

    _compute_advanced_metrics(G, hist, t=0, dt=1.0, cfg=cfg, thr=thr)

    assert hist["glyphogram"][0]["OZ"] == 1
    assert hist["latency_index"][0]["value"] == pytest.approx(0.5)
    rec = hist["EPI_support"][0]
    assert rec["size"] == 2
    assert rec["epi_norm"] == pytest.approx(0.15)
    morph = hist["morph"][0]
    assert morph["ID"] == pytest.approx(0.5)


def test_pp_val_zero_when_no_remesh(graph_canon):
    """PP metric should be 0.0 when no REMESH events occur."""
    G = graph_canon()
    # Nodo en estado SHA, pero sin eventos REMESH
    G.add_node(0, EPI_kind=LATENT_GLYPH)
    inject_defaults(G)

    _metrics_step(G)

    morph = G.graph["history"]["morph"][0]
    assert morph["PP"] == 0.0


def test_pp_val_handles_missing_sha(graph_canon):
    """PP metric handles absence of SHA counts gracefully."""
    G = graph_canon()
    # Nodo en estado REMESH pero sin nodos SHA
    G.add_node(0, EPI_kind="REMESH")
    inject_defaults(G)

    _metrics_step(G)

    morph = G.graph["history"]["morph"][0]
    assert morph["PP"] == 0.0


def test_save_by_node_flag_keeps_metrics_equal(graph_canon):
    """Disabling per-node storage should not alter global metrics."""
    G_true = graph_canon()
    G_true.graph["METRICS"] = dict(G_true.graph["METRICS"])
    G_true.graph["METRICS"]["save_by_node"] = True

    G_false = graph_canon()
    G_false.graph["METRICS"] = dict(G_false.graph["METRICS"])
    G_false.graph["METRICS"]["save_by_node"] = False

    for G in (G_true, G_false):
        G.add_node(0, EPI_kind="OZ")
        G.add_node(1, EPI_kind=LATENT_GLYPH)
        inject_defaults(G)
        for n in G.nodes():
            nd = G.nodes[n]
            nd["glyph_history"] = [nd.get("EPI_kind")]
        G.graph["_t"] = 0
        _metrics_step(G)
        G.nodes[0]["EPI_kind"] = "NAV"
        G.nodes[0].setdefault("glyph_history", []).append("NAV")
        G.graph["_t"] = 1
        _metrics_step(G)

    hist_true = G_true.graph["history"]
    hist_false = G_false.graph["history"]

    assert hist_true["Tg_total"] == hist_false["Tg_total"]
    assert hist_true["glyphogram"] == hist_false["glyphogram"]
    assert hist_true["latency_index"] == hist_false["latency_index"]
    assert hist_true["morph"] == hist_false["morph"]
    assert hist_true["Tg_by_node"] != {}
    assert hist_false.get("Tg_by_node", {}) == {}


def test_latency_index_uses_max_denominator(graph_canon):
    """Latency index uses max(1, n_total) to avoid zero division."""
    G = graph_canon()
    hist = {}
    _update_latency_index(G, hist, n_total=0, n_latent=2, t=0)
    assert hist["latency_index"][0]["value"] == 2.0


def test_update_epi_support_matches_manual(graph_canon):
    """_update_epi_support computes size and norm as expected."""
    G = graph_canon()
    # valores diversos de EPI
    G.add_node(0, EPI=0.06)
    G.add_node(1, EPI=-0.1)
    G.add_node(2, EPI=0.01)
    G.add_node(3, EPI=0.05)
    inject_defaults(G)
    hist = {}
    thr = float(G.graph.get("EPI_SUPPORT_THR"))
    _update_epi_support(G, hist, t=0, thr=thr)

    expected_vals = [
        abs(get_attr(G.nodes[n], ALIAS_EPI, 0.0))
        for n in G.nodes()
        if abs(get_attr(G.nodes[n], ALIAS_EPI, 0.0)) >= thr
    ]
    expected_size = len(expected_vals)
    expected_norm = (
        sum(expected_vals) / expected_size if expected_size else 0.0
    )

    rec = hist["EPI_support"][0]
    assert rec["size"] == expected_size
    assert rec["epi_norm"] == pytest.approx(expected_norm)
