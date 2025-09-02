"""Pruebas de observers."""
import math
import statistics as st
from collections import deque
import pytest

from tnfr.constants import ALIAS_THETA
from tnfr.observers import phase_sync, kuramoto_order, glyph_load, wbar
from tnfr.gamma import kuramoto_R_psi
from tnfr.sense import sigma_vector
from tnfr.constants_glyphs import ANGLE_MAP, ESTABILIZADORES, DISRUPTIVOS
from tnfr.helpers import angle_diff, set_attr
from tnfr.callback_utils import CallbackEvent
from tnfr.observers import attach_standard_observer

def test_phase_observers_match_manual_calculation(graph_canon):
    G = graph_canon()
    angles = [0.0, math.pi / 2, math.pi]
    for idx, th in enumerate(angles):
        G.add_node(idx)
        set_attr(G.nodes[idx], ALIAS_THETA, th)

    X = [math.cos(th) for th in angles]
    Y = [math.sin(th) for th in angles]
    th_mean = math.atan2(sum(Y) / len(Y), sum(X) / len(X))
    var = st.pvariance([angle_diff(th, th_mean) for th in angles])
    expected_sync = 1.0 / (1.0 + var)
    assert math.isclose(phase_sync(G), expected_sync)

    R = ((sum(X) ** 2 + sum(Y) ** 2) ** 0.5) / len(angles)
    assert math.isclose(kuramoto_order(G), float(R))


def test_kuramoto_order_matches_kuramoto_R_psi(graph_canon):
    G = graph_canon()
    angles = [0.1, 1.5, 2.9]
    for idx, th in enumerate(angles):
        G.add_node(idx)
        set_attr(G.nodes[idx], ALIAS_THETA, th)

    R_ok = kuramoto_order(G)
    R, _ = kuramoto_R_psi(G)
    assert math.isclose(R_ok, R)


def test_glyph_load_uses_module_constants(monkeypatch, graph_canon):
    G = graph_canon()
    G.add_node(0, glyph_history=["A"])
    G.add_node(1, glyph_history=["B"])

    # Patch constants to custom categories
    monkeypatch.setattr("tnfr.observers.GLYPH_GROUPS", {"estabilizadores": ["A"], "disruptivos": ["B"]})

    dist = glyph_load(G)

    assert dist["_estabilizadores"] == pytest.approx(0.5)
    assert dist["_disruptivos"] == pytest.approx(0.5)


def test_sigma_vector_consistency():
    # Distribución ficticia de glyphs
    dist = {"IL": 0.4, "RA": 0.3, "ZHIR": 0.1, "AL": 0.2, "_count": 10}

    res = sigma_vector(dist)

    # Cálculo esperado con el mapa de ángulos canónico
    keys = ESTABILIZADORES + DISRUPTIVOS
    angles = {k: ANGLE_MAP[k] for k in keys}
    total = sum(dist.get(k, 0.0) for k in keys)
    x = sum(dist.get(k, 0.0) / total * math.cos(a) for k, a in angles.items())
    y = sum(dist.get(k, 0.0) / total * math.sin(a) for k, a in angles.items())
    mag = math.hypot(x, y)
    ang = math.atan2(y, x)

    assert math.isclose(res["x"], x)
    assert math.isclose(res["y"], y)
    assert math.isclose(res["mag"], mag)
    assert math.isclose(res["angle"], ang)


def test_wbar_accepts_deque(graph_canon):
    G = graph_canon()
    cs = deque([0.1, 0.5, 0.9], maxlen=10)
    G.graph["history"] = {"C_steps": cs}
    assert wbar(G, window=2) == pytest.approx((0.5 + 0.9) / 2)


def test_attach_standard_observer_registers_callbacks(graph_canon):
    G = graph_canon()
    attach_standard_observer(G)
    for ev in CallbackEvent:
        assert ev in G.graph["callbacks"]
