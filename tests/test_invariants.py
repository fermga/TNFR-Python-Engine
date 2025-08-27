from __future__ import annotations
import math
import pytest

from tnfr.constants import inject_defaults, DEFAULTS
from tnfr.scenarios import build_graph
from tnfr.dynamics import step, _update_history
from tnfr.operators import aplicar_glifo, aplicar_remesh_si_estabilizacion_global


@pytest.fixture
def G_small():
    G = build_graph(n=8, topology="ring", seed=7)
    inject_defaults(G, DEFAULTS)
    _update_history(G)
    return G


def _repeat_steps(G, k: int):
    for _ in range(k):
        step(G)


def test_clamps_numeric_stability(G_small):
    _repeat_steps(G_small, 50)
    epi_min = G_small.graph.get("EPI_MIN", -1e9)
    epi_max = G_small.graph.get("EPI_MAX", 1e9)
    for n in G_small.nodes():
        x = float(G_small.nodes[n].get("EPI", 0.0))
        assert math.isfinite(x)
        assert x >= epi_min - 1e-6
        assert x <= epi_max + 1e-6


def test_conservation_under_IL_SHA(G_small):
    for n in G_small.nodes():
        nd = G_small.nodes[n]
        nd["ΔNFR"] = 0.0
        nd["νf"] = 1.0
    G_small.graph["GAMMA"] = {"type": "none"}

    epi0 = {n: float(G_small.nodes[n].get("EPI", 0.0)) for n in G_small.nodes()}

    for _ in range(5):
        for n in G_small.nodes():
            aplicar_glifo(G_small, n, "I’L", window=1)
    epi1 = {n: float(G_small.nodes[n].get("EPI", 0.0)) for n in G_small.nodes()}

    for _ in range(5):
        for n in G_small.nodes():
            aplicar_glifo(G_small, n, "SH’A", window=1)
    epi2 = {n: float(G_small.nodes[n].get("EPI", 0.0)) for n in G_small.nodes()}

    for n in G_small.nodes():
        assert abs(epi1[n] - epi0[n]) < 5e-3
        assert abs(epi2[n] - epi1[n]) < 5e-2


def test_remesh_cooldown_if_present(G_small):
    cooldown = G_small.graph.get("REMESH_COOLDOWN", G_small.graph.get("REMESH_COOLDOWN_VENTANA", None))
    if cooldown is None:
        pytest.skip("No hay REMESH_COOLDOWN definido en el motor")

    w_estab = int(G_small.graph.get("REMESH_STABILITY_WINDOW", 0))
    sf = G_small.graph.setdefault("history", {}).setdefault("stable_frac", [])
    sf.extend([1.0] * w_estab)
    tau = int(G_small.graph.get("REMESH_TAU", 0))
    snap = {n: G_small.nodes[n].get("EPI", 0.0) for n in G_small.nodes()}
    from collections import deque
    G_small.graph["_epi_hist"] = deque([snap.copy() for _ in range(tau + 1)], maxlen=tau + 1)

    aplicar_remesh_si_estabilizacion_global(G_small)
    events = list(G_small.graph.get("history", {}).get("remesh_events", []))
    assert len(events) == 1

    sf.append(1.0)
    aplicar_remesh_si_estabilizacion_global(G_small)
    events2 = list(G_small.graph.get("history", {}).get("remesh_events", []))
    assert len(events2) == 1

    sf.extend([1.0] * int(cooldown))
    aplicar_remesh_si_estabilizacion_global(G_small)
    events3 = list(G_small.graph.get("history", {}).get("remesh_events", []))
    assert len(events3) == 2
