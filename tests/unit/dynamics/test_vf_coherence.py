"""Unit tests for adapting VF through coherence measurements."""

from __future__ import annotations

import copy

import networkx as nx
import pytest

from tnfr.constants import inject_defaults
from tnfr.dynamics import adapt_vf_by_coherence, step
from tnfr.dynamics.adaptation import _vf_adapt_chunk
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


def _vf_clamp_test_graph(graph_canon):
    G = graph_canon()
    G.add_edge("clamp_high", "anchor_high")
    G.add_edge("clamp_low", "anchor_low")
    inject_defaults(G)
    G.graph["VF_ADAPT_TAU"] = 1
    G.graph["VF_ADAPT_MU"] = 1.5
    G.graph["EPS_DNFR_STABLE"] = 1.0
    selector_thresholds = dict(G.graph.get("SELECTOR_THRESHOLDS", {}))
    selector_thresholds["si_hi"] = 0.8
    G.graph["SELECTOR_THRESHOLDS"] = selector_thresholds

    for node in G.nodes:
        nd = G.nodes[node]
        nd["ΔNFR"] = 0.0
        nd["stable_count"] = 0
        if node.startswith("clamp_"):
            nd["Si"] = 0.9
        else:
            nd["Si"] = 0.2

    G.nodes["clamp_high"]["νf"] = 0.2
    G.nodes["anchor_high"]["νf"] = 2.5
    G.nodes["clamp_low"]["νf"] = 0.8
    G.nodes["anchor_low"]["νf"] = -1.5

    return G


def test_adapt_vf_requires_injected_defaults():
    bare = nx.Graph()
    bare.add_edge("seed", "anchor")

    with pytest.raises(KeyError, match="inject_defaults"):
        adapt_vf_by_coherence(bare)

    inject_defaults(bare)


def test_adapt_vf_rejects_non_numeric_tau(graph_canon):
    G = graph_canon()
    G.add_edge(0, 1)
    G.graph["VF_ADAPT_TAU"] = "not-an-int"

    with pytest.raises((TypeError, ValueError)):
        adapt_vf_by_coherence(G)

    inject_defaults(G, override=True)


def test_vf_converge_to_neighbor_average_when_stable(graph_canon):
    G = graph_canon()
    G.add_edge(0, 1)
    inject_defaults(G)
    # Configure the weights so the dynamics stay in the stable regime
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


@pytest.mark.parametrize("numpy_mode", ["vectorised", "python"])
def test_adapt_vf_invalid_jobs_fallback_to_serial(graph_canon, monkeypatch, numpy_mode):
    if numpy_mode == "vectorised":
        if get_numpy() is None:
            pytest.skip("NumPy not available for vectorised branch")
    else:
        monkeypatch.setattr("tnfr.dynamics.adaptation.get_numpy", lambda: None)
        monkeypatch.setattr("tnfr.dynamics.get_numpy", lambda: None)

    base = _coherence_test_graph(graph_canon)
    serial = copy.deepcopy(base)
    invalid_hint = copy.deepcopy(base)
    single_job = copy.deepcopy(base)

    adapt_vf_by_coherence(serial, n_jobs=None)
    adapt_vf_by_coherence(invalid_hint, n_jobs="not-an-int")
    adapt_vf_by_coherence(single_job, n_jobs=1)

    for node in serial.nodes:
        assert invalid_hint.nodes[node]["νf"] == pytest.approx(
            serial.nodes[node]["νf"]
        )
        assert single_job.nodes[node]["νf"] == pytest.approx(serial.nodes[node]["νf"])

        assert (
            invalid_hint.nodes[node]["stable_count"]
            == serial.nodes[node]["stable_count"]
        )
        assert (
            single_job.nodes[node]["stable_count"]
            == serial.nodes[node]["stable_count"]
        )


def test_adapt_vf_noop_on_empty_graph():
    empty = nx.Graph()
    inject_defaults(empty)

    # Should exit early without raising even when the graph is empty.
    adapt_vf_by_coherence(empty)


@pytest.mark.parametrize("mode", ["vectorised", "python"])
def test_adapt_vf_clamps_to_bounds(graph_canon, monkeypatch, mode):
    G = _vf_clamp_test_graph(graph_canon)
    vf_min = float(G.graph["VF_MIN"])
    vf_max = float(G.graph["VF_MAX"])

    if mode == "vectorised":
        if get_numpy() is None:
            pytest.skip("NumPy not available for vectorised branch")
        adapt_vf_by_coherence(G)
    else:
        monkeypatch.setattr("tnfr.dynamics.get_numpy", lambda: None)
        monkeypatch.setattr("tnfr.dynamics.adaptation.get_numpy", lambda: None)

        class _DummyExecutor:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def map(self, func, iterable):
                for item in iterable:
                    yield _vf_adapt_chunk(item)

        monkeypatch.setattr(
            "tnfr.dynamics.adaptation.ProcessPoolExecutor", _DummyExecutor
        )
        adapt_vf_by_coherence(G, n_jobs=2)

    assert G.nodes["clamp_high"]["νf"] == pytest.approx(vf_max)
    assert G.nodes["clamp_low"]["νf"] == pytest.approx(vf_min)
