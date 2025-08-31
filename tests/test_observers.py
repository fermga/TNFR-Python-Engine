import gc
import math
import statistics as st
import networkx as nx

from tnfr.node import NodoNX
from tnfr.operators import random_jitter
from tnfr.constants import ALIAS_THETA
from tnfr.observers import sincronía_fase, orden_kuramoto
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
