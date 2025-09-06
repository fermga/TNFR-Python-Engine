"""Pruebas de sense."""

import time
import networkx as nx
import pytest

from tnfr.sense import (
    sigma_vector_node,
    sigma_vector_from_graph,
    _node_weight,
    _sigma_from_pairs,
    _sigma_from_vectors,
    glyph_unit,
    glyph_angle,
)
from tnfr.types import Glyph


def _make_graph():
    G = nx.Graph()
    G.add_node(0, glyph_history=[Glyph.AL.value], Si=1.0, EPI=2.0)
    G.add_node(1, Si=0.3, EPI=1.5)
    return G


def test_sigma_vector_node_paths():
    G = _make_graph()
    sv_si = sigma_vector_node(G, 0)
    assert sv_si and sv_si["glyph"] == Glyph.AL.value
    assert sv_si["w"] == 1.0
    assert sigma_vector_node(G, 1) is None
    sv_epi = sigma_vector_node(G, 0, weight_mode="EPI")
    assert sv_epi["w"] == 2.0
    assert sv_epi["mag"] == pytest.approx(2 * sv_si["mag"])


def test_sigma_vector_from_graph_paths():
    G = _make_graph()
    sv_si = sigma_vector_from_graph(G)
    sv_epi = sigma_vector_from_graph(G, weight_mode="EPI")
    assert sv_si["n"] == 1
    assert sv_epi["n"] == 1
    assert sv_epi["mag"] == pytest.approx(2 * sv_si["mag"])


def _sigma_vector_from_graph_naive(G, weight_mode: str = "Si"):
    """Referencia que recalcula ``glyph_unit(g) * w`` en cada paso."""
    pairs = []
    for _, nd in G.nodes(data=True):
        nw = _node_weight(nd, weight_mode)
        if not nw:
            continue
        g, w, _ = nw
        pairs.append((g, w))
    vec, n = _sigma_from_pairs(pairs)
    vec["n"] = n
    return vec


def test_sigma_vector_from_graph_matches_naive():
    """La versión optimizada coincide con el cálculo ingenuo y no es
    más lenta."""
    G_opt = nx.Graph()
    glyphs = list(Glyph)
    for i in range(1000):
        g = glyphs[i % len(glyphs)].value
        G_opt.add_node(i, glyph_history=[g], Si=float(i % 10) / 10)
    G_ref = G_opt.copy()

    start = time.perf_counter()
    vec_opt = sigma_vector_from_graph(G_opt)
    t_opt = time.perf_counter() - start

    start = time.perf_counter()
    vec_ref = _sigma_vector_from_graph_naive(G_ref)
    t_ref = time.perf_counter() - start

    for key in ("x", "y", "mag", "angle", "n"):
        assert vec_opt[key] == pytest.approx(vec_ref[key])
    assert t_opt <= t_ref * 2


def test_sigma_from_vectors_accepts_single_complex():
    vec, n = _sigma_from_vectors(1 + 1j)
    assert n == 1
    assert vec["x"] == pytest.approx(1.0)
    assert vec["y"] == pytest.approx(1.0)


def test_sigma_from_vectors_rejects_invalid_iterable():
    with pytest.raises(TypeError, match="iterable of complex"):
        _sigma_from_vectors("abc")


def test_unknown_glyph_raises():
    with pytest.raises(KeyError):
        glyph_angle("ZZ")
    with pytest.raises(KeyError):
        glyph_unit("ZZ")
