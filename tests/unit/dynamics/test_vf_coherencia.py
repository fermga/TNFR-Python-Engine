"""Pruebas de vf coherencia."""

from __future__ import annotations

import copy

import pytest

from tnfr.constants import inject_defaults
from tnfr.dynamics import adapt_vf_by_coherence, step
from tnfr.utils import get_numpy


def _coherence_test_graph(graph_canon):
    G = graph_canon()
    G.add_edge(0, 1)
    inject_defaults(G)
    G.graph["VF_ADAPT_TAU"] = 1
    G.graph["VF_ADAPT_MU"] = 0.5
    for n in G.nodes():
        nd = G.nodes[n]
        nd["Si"] = 0.9
        nd["ΔNFR"] = 0.0
        nd["stable_count"] = 0
    G.nodes[0]["νf"] = 0.2
    G.nodes[1]["νf"] = 1.0
    return G


def test_vf_converge_to_neighbor_average_when_stable(graph_canon):
    G = graph_canon()
    G.add_edge(0, 1)
    inject_defaults(G)
    # configuraciones para estabilidad
    G.graph["DNFR_WEIGHTS"] = {
        "phase": 1.0,
        "epi": 0.0,
        "vf": 0.0,
        "topo": 0.0,
    }
    G.graph["VF_ADAPT_TAU"] = 2
    G.graph["VF_ADAPT_MU"] = 0.5
    for n in G.nodes():
        nd = G.nodes[n]
        nd["theta"] = 0.0
        nd["EPI"] = 0.0
    G.nodes[0]["νf"] = 0.2
    G.nodes[1]["νf"] = 1.0

    for _ in range(3):
        step(G, use_Si=True, apply_glyphs=False)

    assert G.nodes[0]["νf"] == pytest.approx(0.6)
    assert G.nodes[1]["νf"] == pytest.approx(0.6)


def test_adapt_vf_serial_matches_parallel_same_snapshot(graph_canon):
    base = _coherence_test_graph(graph_canon)
    serial = copy.deepcopy(base)
    parallel = copy.deepcopy(base)

    adapt_vf_by_coherence(serial, n_jobs=None)
    adapt_vf_by_coherence(parallel, n_jobs=4)

    for n in serial.nodes:
        assert parallel.nodes[n]["νf"] == pytest.approx(serial.nodes[n]["νf"])

    if get_numpy() is not None:
        # Vector path should keep stable counters synchronised as well.
        for n in serial.nodes:
            assert parallel.nodes[n]["stable_count"] == serial.nodes[n]["stable_count"]


def test_adapt_vf_python_parallel_matches_serial(graph_canon, monkeypatch):
    monkeypatch.setattr("tnfr.dynamics.get_numpy", lambda: None)
    base = _coherence_test_graph(graph_canon)
    serial = copy.deepcopy(base)
    parallel = copy.deepcopy(base)

    adapt_vf_by_coherence(serial, n_jobs=None)
    adapt_vf_by_coherence(parallel, n_jobs=2)

    for n in serial.nodes:
        assert parallel.nodes[n]["νf"] == pytest.approx(serial.nodes[n]["νf"])
        assert parallel.nodes[n]["stable_count"] == serial.nodes[n]["stable_count"]
