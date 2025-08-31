import networkx as nx

from tnfr.constants import attach_defaults
from tnfr.metrics import _metrics_step, _update_latency_index


def test_pp_val_zero_when_no_remesh(graph_canon):
    """PP metric should be 0.0 when no REMESH events occur."""
    G = graph_canon()
    # Nodo en estado SHA, pero sin eventos REMESH
    G.add_node(0, EPI_kind="SHA")
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
        G.add_node(1, EPI_kind="SHA")
        attach_defaults(G)
        G.graph["_t"] = 0
        _metrics_step(G)
        G.nodes[0]["EPI_kind"] = "NAV"
        G.graph["_t"] = 1
        _metrics_step(G)

    hist_true = G_true.graph["history"]
    hist_false = G_false.graph["history"]

    assert hist_true["Tg_total"] == hist_false["Tg_total"]
    assert hist_true["glifogram"] == hist_false["glifogram"]
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
