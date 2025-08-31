import gc
import math
import statistics as st
import networkx as nx
import pytest

from tnfr.node import NodoNX
from tnfr.operators import random_jitter
from tnfr.constants import ALIAS_THETA
from tnfr.observers import sincronía_fase, orden_kuramoto, carga_glifica, sigma_vector
from tnfr.helpers import angle_diff, _set_attr


def test_random_jitter_cache_cleared_on_node_removal():
    G = nx.Graph()
    G.add_node(0)
    n0 = NodoNX(G, 0)

    random_jitter(n0, 0.1)
    cache = G.graph.get("_rnd_cache")
    assert cache is not None
    assert len(cache) == 1

    del n0
    G.remove_node(0)
    gc.collect()

    assert len(cache) == 0


def test_phase_observers_match_manual_calculation():
    G = nx.Graph()
    angles = [0.0, math.pi / 2, math.pi]
    for idx, th in enumerate(angles):
        G.add_node(idx)
        _set_attr(G.nodes[idx], ALIAS_THETA, th)

    X = [math.cos(th) for th in angles]
    Y = [math.sin(th) for th in angles]
    th_mean = math.atan2(sum(Y) / len(Y), sum(X) / len(X))
    var = st.pvariance([angle_diff(th, th_mean) for th in angles])
    expected_sync = 1.0 / (1.0 + var)
    assert math.isclose(sincronía_fase(G), expected_sync)

    R = ((sum(X) ** 2 + sum(Y) ** 2) ** 0.5) / len(angles)
    assert math.isclose(orden_kuramoto(G), float(R))


def test_carga_glifica_uses_module_constants(monkeypatch):
    G = nx.Graph()
    G.add_node(0, hist_glifos=["A"])
    G.add_node(1, hist_glifos=["B"])

    # Patch constants to custom categories
    monkeypatch.setattr("tnfr.observers.ESTABILIZADORES", ["A"])  # type: ignore[attr-defined]
    monkeypatch.setattr("tnfr.observers.DISRUPTIVOS", ["B"])  # type: ignore[attr-defined]

    dist = carga_glifica(G)

    assert dist["_estabilizadores"] == pytest.approx(0.5)
    assert dist["_disruptivos"] == pytest.approx(0.5)


def test_sigma_vector_consistency(monkeypatch):
    import tnfr.observers as obs

    G = nx.Graph()
    # Distribución ficticia de glifos
    dist = {"IL": 0.4, "RA": 0.3, "ZHIR": 0.1, "AL": 0.2, "_count": 10}
    monkeypatch.setattr(obs, "carga_glifica", lambda G, window=None: dist)

    res = sigma_vector(G)

    # Cálculo esperado con ángulos previos
    angles = {
        "IL": 0.0,
        "RA": math.pi / 4,
        "UM": math.pi / 2,
        "SHA": 3 * math.pi / 4,
        "OZ": math.pi,
        "ZHIR": 5 * math.pi / 4,
        "NAV": 3 * math.pi / 2,
        "THOL": 7 * math.pi / 4,
    }
    total = sum(dist.get(k, 0.0) for k in angles.keys())
    x = sum(dist.get(k, 0.0) / total * math.cos(a) for k, a in angles.items())
    y = sum(dist.get(k, 0.0) / total * math.sin(a) for k, a in angles.items())
    mag = math.hypot(x, y)
    ang = math.atan2(y, x)

    assert math.isclose(res["x"], x)
    assert math.isclose(res["y"], y)
    assert math.isclose(res["mag"], mag)
    assert math.isclose(res["angle"], ang)
