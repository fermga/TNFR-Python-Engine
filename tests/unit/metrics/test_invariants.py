"""Unit tests verifying invariants reported by the metrics subsystem."""

from __future__ import annotations

import math

import networkx as nx
import pytest

from tnfr.constants import inject_defaults
from tnfr.dynamics import step
from tnfr.glyph_history import ensure_history
from tnfr.initialization import init_node_attrs
from tnfr.metrics import register_metrics_callbacks
from tnfr.metrics.core import _metrics_step
from tnfr.operators import apply_glyph, apply_remesh_if_globally_stable
from tnfr.types import Glyph


@pytest.fixture
def G_small():
    G = nx.cycle_graph(8)
    inject_defaults(G)
    init_node_attrs(G, override=True)
    G.graph["RANDOM_SEED"] = 7
    register_metrics_callbacks(G)
    _metrics_step(G, ctx=None)
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
            apply_glyph(G_small, n, Glyph.IL, window=1)
    epi1 = {n: float(G_small.nodes[n].get("EPI", 0.0)) for n in G_small.nodes()}

    for _ in range(5):
        for n in G_small.nodes():
            apply_glyph(G_small, n, Glyph.SHA, window=1)
    epi2 = {n: float(G_small.nodes[n].get("EPI", 0.0)) for n in G_small.nodes()}

    for n in G_small.nodes():
        assert abs(epi1[n] - epi0[n]) < 5e-3
        assert abs(epi2[n] - epi1[n]) < 5e-2


def test_remesh_cooldown_if_present(G_small):
    cooldown = G_small.graph.get(
        "REMESH_COOLDOWN",
        G_small.graph.get("REMESH_COOLDOWN_WINDOW", None),
    )
    if cooldown is None:
        pytest.skip("REMESH_COOLDOWN is not configured in the engine")

    w_estab = int(G_small.graph.get("REMESH_STABILITY_WINDOW", 0))
    sf = G_small.graph.setdefault("history", {}).setdefault("stable_frac", [])
    sf.extend([1.0] * w_estab)
    tau_g = int(G_small.graph.get("REMESH_TAU_GLOBAL", 0))
    snap = {n: G_small.nodes[n].get("EPI", 0.0) for n in G_small.nodes()}
    from collections import deque

    G_small.graph["_epi_hist"] = deque(
        [snap.copy() for _ in range(tau_g + 1)], maxlen=tau_g + 1
    )

    apply_remesh_if_globally_stable(G_small)
    events = list(ensure_history(G_small).get("remesh_events", []))
    assert len(events) == 1

    sf.append(1.0)
    apply_remesh_if_globally_stable(G_small)
    events2 = list(ensure_history(G_small).get("remesh_events", []))
    assert len(events2) == 1

    sf.extend([1.0] * int(cooldown))
    apply_remesh_if_globally_stable(G_small)
    events3 = list(ensure_history(G_small).get("remesh_events", []))
    assert len(events3) == 2
