"""Unit tests ensuring metric history series are registered and populated during dynamics steps."""



from tnfr.constants import inject_defaults
from tnfr.dynamics import step
from tnfr.metrics import register_metrics_callbacks
from tnfr.gamma import GAMMA_REGISTRY
from tnfr.glyph_history import HistoryDict, ensure_history


def test_history_delta_si_and_B(graph_canon):
    G = graph_canon()
    G.add_node(0, EPI=0.0, νf=0.5, theta=0.0)
    inject_defaults(G)
    register_metrics_callbacks(G)
    step(G, apply_glyphs=False)
    step(G, apply_glyphs=False)
    hist = ensure_history(G)
    assert "delta_Si" in hist and len(hist["delta_Si"]) >= 2
    assert "B" in hist and len(hist["B"]) >= 2


def test_gamma_kuramoto_tanh_registry(graph_canon):
    G = graph_canon()
    G.add_nodes_from([0, 1])
    inject_defaults(G)
    G.nodes[0]["theta"] = 0.0
    G.nodes[1]["theta"] = 0.0
    cfg = {"type": "kuramoto_tanh", "beta": 0.5, "k": 2.0, "R0": 0.0}
    gamma_fn = GAMMA_REGISTRY["kuramoto_tanh"].fn
    val = gamma_fn(G, 0, 0.0, cfg)
    assert abs(val) <= cfg["beta"]


def test_pop_least_used_batch_stops_after_k_even_with_stale():
    hist = HistoryDict({"a": 1, "b": 2})
    hist.get_increment("a")
    hist.get_increment("b")
    hist.get_increment("b")
    hist._counts["stale"] = 0
    hist.pop_least_used_batch(2)
    assert not hist
    assert not hist._counts


def test_pop_least_used_batch_removes_k_elements():
    hist = HistoryDict({f"k{i}": i for i in range(3)})
    for key in list(hist):
        hist._counts[key] = 0
    for i in range(3):
        for _ in range(i):
            hist.get_increment(f"k{i}")
    hist.pop_least_used_batch(3)
    assert not hist
    assert not hist._counts
