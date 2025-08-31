from __future__ import annotations
import pytest

from tnfr.constants import inject_defaults, DEFAULTS
from tnfr.scenarios import build_graph
from tnfr.dynamics import step


@pytest.mark.parametrize("method", ["euler", "rk4"])
def test_epi_limits_preserved(method):
    G = build_graph(n=6, topology="ring", seed=1)
    inject_defaults(G, DEFAULTS)
    G.graph["INTEGRATOR_METHOD"] = method
    G.graph["DT_MIN"] = 0.1
    G.graph["GAMMA"] = {"type": "none"}

    def const_dnfr(G):
        for i, n in enumerate(G.nodes()):
            nd = G.nodes[n]
            nd["ΔNFR"] = 5.0 if i % 2 == 0 else -5.0
            nd["νf"] = 1.0
            nd["EPI"] = 0.0

    G.graph["compute_delta_nfr"] = const_dnfr

    step(G, dt=1.0)

    e_min = G.graph["EPI_MIN"]
    e_max = G.graph["EPI_MAX"]
    for i, n in enumerate(G.nodes()):
        epi = G.nodes[n]["EPI"]
        if i % 2 == 0:
            assert epi == pytest.approx(e_max)
        else:
            assert epi == pytest.approx(e_min)
        assert e_min - 1e-6 <= epi <= e_max + 1e-6
