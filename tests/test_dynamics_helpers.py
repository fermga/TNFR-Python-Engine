"""Tests for dynamics helpers."""

import pytest
import networkx as nx

from tnfr.dynamics import (
    _init_dnfr_cache,
    _refresh_dnfr_vectors,
    _compute_neighbor_means,
    _choose_glyph,
    _compute_cos_sin,
)
from tnfr.grammar import AL, EN


def test_init_and_refresh_dnfr_cache(graph_canon):
    G = graph_canon()
    for i in range(2):
        G.add_node(i, theta=0.1 * i, EPI=float(i), VF=float(i))
    nodes = list(G.nodes())
    cache, idx, th, epi, vf, cx, sx, refreshed = _init_dnfr_cache(G, nodes, None, 1, False)
    assert refreshed
    _refresh_dnfr_vectors(G, nodes, th, epi, vf, cx, sx)
    assert th[1] == pytest.approx(0.1)
    cache2, *_rest, refreshed2 = _init_dnfr_cache(G, nodes, cache, 1, False)
    assert not refreshed2
    assert cache2 is cache


def test_compute_cos_sin_uses_numpy(monkeypatch):
    calls = []

    class FakeNP:
        def array(self, arr, dtype=float):
            return arr

        def cos(self, arr):
            calls.append("cos")
            return [1.0] * len(arr)

        def sin(self, arr):
            calls.append("sin")
            return [0.0] * len(arr)

    monkeypatch.setattr("tnfr.dynamics.get_numpy", lambda: FakeNP())
    cos, sin = _compute_cos_sin([0.0, 1.0])
    assert calls == ["cos", "sin"]
    assert cos == [1.0, 1.0]
    assert sin == [0.0, 0.0]


def test_compute_neighbor_means_list():
    G = nx.Graph()
    G.add_edge(0, 1)
    data = {
        "w_topo": 0.0,
        "theta": [0.0, 0.0],
        "epi": [0.0, 0.0],
        "vf": [0.0, 0.0],
        "cos_theta": [1.0, 1.0],
        "sin_theta": [0.0, 0.0],
        "idx": {0: 0, 1: 1},
        "nodes": [0, 1],
    }
    x = [1.0, 0.0]
    y = [0.0, 0.0]
    epi_sum = [2.0, 0.0]
    vf_sum = [0.0, 0.0]
    count = [1, 0]
    th_bar, epi_bar, vf_bar, deg_bar = _compute_neighbor_means(
        G, data, x=x, y=y, epi_sum=epi_sum, vf_sum=vf_sum, count=count
    )
    assert th_bar[0] == pytest.approx(0.0)
    assert epi_bar[0] == pytest.approx(2.0)
    assert vf_bar[0] == pytest.approx(0.0)
    assert deg_bar is None


def test_choose_glyph_respects_lags(graph_canon):
    G = graph_canon()
    G.add_node(0)
    selector = lambda G, n: "RA"
    h_al = {0: 2}
    h_en = {0: 0}
    g = _choose_glyph(G, 0, selector, False, h_al, h_en, 1, 5)
    assert g == AL
    h_al[0] = 0
    h_en[0] = 6
    g = _choose_glyph(G, 0, selector, False, h_al, h_en, 1, 5)
    assert g == EN
