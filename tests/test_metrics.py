"""Pruebas de metrics."""
import pytest
import networkx as nx

from tnfr.constants import attach_defaults, ALIAS_EPI
from tnfr.helpers import get_attr
from tnfr.metrics import _metrics_step, _update_latency_index, _update_epi_support
from tnfr.metrics.core import LATENT_GLYPH


def test_pp_val_zero_when_no_remesh(graph_canon):
    """PP metric should be 0.0 when no REMESH events occur."""
    G = graph_canon()
    # Nodo en estado SHA, pero sin eventos REMESH
    G.add_node(0, EPI_kind=LATENT_GLYPH)
    attach_defaults(G)

    _metrics_step(G)

    morph = G.graph["history"]["morph"][0]
    assert morph["PP"] == 0.0


def test_pp_val_handles_missing_sha(graph_canon):
    """PP metric handles absence of SHA counts gracefully."""
    G = graph_canon()
    # Nodo en estado REMESH pero sin nodos SHA
    G.add_node(0, EPI_kind="REMESH")
    attach_defaults(G)

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
        attach_defaults(G)
        G.graph["_t"] = 0
        _metrics_step(G)
        G.nodes[0]["EPI_kind"] = "NAV"
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


def test_latency_index_uses_max_denominator():
    """Latency index uses max(1, n_total) to avoid zero division."""
    G = nx.Graph()
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
    attach_defaults(G)
    hist = {}
    thr = float(G.graph.get("EPI_SUPPORT_THR"))
    _update_epi_support(G, hist, t=0, thr=thr)

    expected_vals = [
        abs(get_attr(G.nodes[n], ALIAS_EPI, 0.0))
        for n in G.nodes()
        if abs(get_attr(G.nodes[n], ALIAS_EPI, 0.0)) >= thr
    ]
    expected_size = len(expected_vals)
    expected_norm = sum(expected_vals) / expected_size if expected_size else 0.0

    rec = hist["EPI_support"][0]
    assert rec["size"] == expected_size
    assert rec["epi_norm"] == pytest.approx(expected_norm)
