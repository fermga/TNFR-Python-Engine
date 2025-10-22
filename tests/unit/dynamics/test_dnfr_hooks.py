import math

from tnfr.alias import get_attr
from tnfr.constants import get_aliases
from tnfr.dynamics import (
    dnfr_epi_vf_mixed,
    dnfr_laplacian,
    dnfr_phase_only,
    set_delta_nfr_hook,
)

ALIAS_THETA = get_aliases("THETA")
ALIAS_EPI = get_aliases("EPI")
ALIAS_VF = get_aliases("VF")
ALIAS_DNFR = get_aliases("DNFR")


def test_dnfr_phase_only_computes_gradient(graph_canon):
    G = graph_canon()
    G.add_edge(0, 1)
    G.nodes[0][ALIAS_THETA[0]] = 0.0
    G.nodes[1][ALIAS_THETA[0]] = 1.5707963267948966  # pi/2
    dnfr_phase_only(G)
    assert get_attr(G.nodes[0], ALIAS_DNFR, 0.0) == 0.5
    assert get_attr(G.nodes[1], ALIAS_DNFR, 0.0) == -0.5


def test_dnfr_epi_vf_mixed_sets_average(graph_canon):
    G = graph_canon()
    G.add_edge(0, 1)
    G.nodes[0][ALIAS_EPI[0]] = 1.0
    G.nodes[0][ALIAS_VF[0]] = 0.0
    G.nodes[1][ALIAS_EPI[0]] = 0.0
    G.nodes[1][ALIAS_VF[0]] = 1.0
    dnfr_epi_vf_mixed(G)
    assert get_attr(G.nodes[0], ALIAS_DNFR, 1.0) == 0.0
    assert get_attr(G.nodes[1], ALIAS_DNFR, 1.0) == 0.0


def test_dnfr_laplacian_respects_weights(graph_canon):
    G = graph_canon()
    G.add_edge(0, 1)
    G.graph["DNFR_WEIGHTS"] = {"epi": 1.0, "vf": 0.0}
    G.nodes[0][ALIAS_EPI[0]] = 1.0
    G.nodes[1][ALIAS_EPI[0]] = 0.0
    dnfr_laplacian(G)
    assert get_attr(G.nodes[0], ALIAS_DNFR, 0.0) == -1.0
    assert get_attr(G.nodes[1], ALIAS_DNFR, 0.0) == 1.0


def test_dnfr_phase_only_parallel_matches_serial(graph_canon, monkeypatch):
    G_serial = graph_canon()
    G_serial.add_edge(0, 1)
    G_serial.nodes[0][ALIAS_THETA[0]] = 0.0
    G_serial.nodes[1][ALIAS_THETA[0]] = math.pi / 2
    dnfr_phase_only(G_serial)
    serial = {
        node: get_attr(data, ALIAS_DNFR, 0.0)
        for node, data in G_serial.nodes(data=True)
    }

    G_parallel = graph_canon()
    G_parallel.add_edge(0, 1)
    G_parallel.nodes[0][ALIAS_THETA[0]] = 0.0
    G_parallel.nodes[1][ALIAS_THETA[0]] = math.pi / 2
    monkeypatch.setattr("tnfr.dynamics.dnfr.get_numpy", lambda: None)
    dnfr_phase_only(G_parallel, n_jobs=2)
    parallel = {
        node: get_attr(data, ALIAS_DNFR, 0.0)
        for node, data in G_parallel.nodes(data=True)
    }

    assert parallel == serial


def test_set_delta_nfr_hook_forwards_jobs(graph_canon):
    G = graph_canon()
    recorded: list[int | None] = []

    def hook(graph, *, n_jobs=None):
        recorded.append(n_jobs)

    set_delta_nfr_hook(G, hook)
    compute = G.graph["compute_delta_nfr"]
    compute(G, n_jobs=3)
    assert recorded == [3]


def test_set_delta_nfr_hook_ignores_jobs_when_missing(graph_canon):
    G = graph_canon()
    calls: list[bool] = []

    def hook(graph):
        calls.append(True)

    set_delta_nfr_hook(G, hook)
    compute = G.graph["compute_delta_nfr"]
    compute(G, n_jobs=5)
    assert calls == [True]
