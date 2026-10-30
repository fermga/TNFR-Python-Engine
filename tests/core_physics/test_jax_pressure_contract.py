"""JAX graph-adapter delegation is CPU execution, not an independent law."""

from __future__ import annotations

import networkx as nx

from tnfr.alias import get_attr
from tnfr.backends.jax_backend import JAXBackend
from tnfr.constants.aliases import ALIAS_DNFR


def test_graph_pressure_delegation_options_arithmetic_and_provenance(monkeypatch):
    from tnfr.dynamics import dnfr

    # This tests adapter methods independently of optional library discovery.
    backend = JAXBackend.__new__(JAXBackend)
    graph = nx.Graph()
    graph.add_weighted_edges_from([(0, 1, 1.0), (0, 2, 3.0)])
    for node, epi in enumerate((0.0, 0.0, 1.0)):
        graph.nodes[node].update(EPI=epi, nu_f=2.0, phase=0.0)
    graph.graph["DNFR_WEIGHTS"] = dict(epi=1.0, phase=0.0, vf=0.0, topo=0.0)
    original = dnfr.default_compute_delta_nfr
    calls = []

    def record(target, **kwargs):
        calls.append((target, kwargs))
        return original(target, **kwargs)

    monkeypatch.setattr(dnfr, "default_compute_delta_nfr", record)
    profile = {}
    backend.compute_delta_nfr(graph, cache_size=2, n_jobs=1, profile=profile)
    assert calls == [(graph, dict(cache_size=2, n_jobs=1, profile=profile))]
    assert tuple(get_attr(graph.nodes[n], ALIAS_DNFR) for n in graph) == (
        0.75,
        0.0,
        -1.0,
    )
    assert profile["dnfr_backend"] == "jax"
    assert profile["dnfr_device"] == "cpu"
    assert profile["dnfr_implementation"] == "canonical"
    assert not backend.supports_gpu
    assert not backend.supports_jit


def test_sense_index_delegates_all_options_and_preserves_result(monkeypatch):
    from tnfr.metrics import sense_index

    backend = JAXBackend.__new__(JAXBackend)
    graph = nx.path_graph(2)
    profile = {}
    result = {0: 0.25, 1: 0.75}
    calls = []

    def record(target, **kwargs):
        calls.append((target, kwargs))
        return result

    monkeypatch.setattr(sense_index, "compute_Si", record)
    actual = backend.compute_si(
        graph, inplace=False, n_jobs=2, chunk_size=7, profile=profile
    )
    assert actual is result
    assert calls == [
        (graph, dict(inplace=False, n_jobs=2, chunk_size=7, profile=profile))
    ]
