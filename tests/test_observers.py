"""Pruebas de observers."""
import math
import statistics as st
import pytest

from tnfr.constants import ALIAS_THETA
from tnfr.observers import sincronía_fase, orden_kuramoto, carga_glifica
from tnfr.gamma import kuramoto_R_psi
from tnfr.sense import sigma_vector
from tnfr.constants_glifos import ANGLE_MAP, ESTABILIZADORES, DISRUPTIVOS
from tnfr.helpers import angle_diff, set_attr

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
    assert math.isclose(sincronía_fase(G), expected_sync)

    R = ((sum(X) ** 2 + sum(Y) ** 2) ** 0.5) / len(angles)
    assert math.isclose(orden_kuramoto(G), float(R))


def test_orden_kuramoto_matches_kuramoto_R_psi(graph_canon):
    G = graph_canon()
    angles = [0.1, 1.5, 2.9]
    for idx, th in enumerate(angles):
        G.add_node(idx)
        set_attr(G.nodes[idx], ALIAS_THETA, th)

    R_ok = orden_kuramoto(G)
    R, _ = kuramoto_R_psi(G)
    assert math.isclose(R_ok, R)


def test_carga_glifica_uses_module_constants(monkeypatch, graph_canon):
    G = graph_canon()
    G.add_node(0, hist_glifos=["A"])
    G.add_node(1, hist_glifos=["B"])

    # Patch constants to custom categories
    monkeypatch.setattr("tnfr.observers.GLYPH_GROUPS", {"estabilizadores": ["A"], "disruptivos": ["B"]})

    dist = carga_glifica(G)

    assert dist["_estabilizadores"] == pytest.approx(0.5)
    assert dist["_disruptivos"] == pytest.approx(0.5)


def test_sigma_vector_consistency():
    # Distribución ficticia de glifos
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
