"""Pruebas de node sample."""

import pytest

from tnfr.dynamics import step, _update_node_sample
from tnfr.rng import clear_rng_cache
from tnfr.constants import inject_defaults
from tnfr.cache import cached_nodes_and_A, increment_edge_version
import networkx as nx
import json
import os
import subprocess
import sys


def _build_graph(n, graph_canon=None):
    G = graph_canon() if graph_canon is not None else nx.Graph()
    inject_defaults(G)
    for i in range(n):
        G.add_node(i, θ=0.0, EPI=0.0)
    return G


def test_node_sample_large_graph(graph_canon):
    G = _build_graph(80, graph_canon)
    G.graph["UM_CANDIDATE_COUNT"] = 10
    step(G, use_Si=False, apply_glyphs=False)
    sample = G.graph.get("_node_sample")
    assert isinstance(sample, list)
    assert len(sample) == 10
    assert set(sample).issubset(set(G.nodes()))


def test_node_sample_small_graph(graph_canon):
    G = _build_graph(20, graph_canon)
    G.graph["UM_CANDIDATE_COUNT"] = 5
    step(G, use_Si=False, apply_glyphs=False)
    sample = G.graph.get("_node_sample")
    assert not isinstance(sample, list)
    assert len(sample) == len(G.nodes())


def test_node_sample_immutable_after_graph_change(graph_canon):
    G = _build_graph(20, graph_canon)
    _update_node_sample(G, step=0)
    sample = G.graph["_node_sample"]
    G.add_node(99)
    assert len(sample) == 20
    assert 99 not in sample


def test_node_sample_deterministic_with_seed(graph_canon):
    clear_rng_cache()
    G1 = _build_graph(80, graph_canon)
    G1.graph["UM_CANDIDATE_COUNT"] = 10
    G1.graph["RANDOM_SEED"] = 123
    _update_node_sample(G1, step=5)
    sample1 = G1.graph["_node_sample"]

    clear_rng_cache()
    G2 = _build_graph(80, graph_canon)
    G2.graph["UM_CANDIDATE_COUNT"] = 10
    G2.graph["RANDOM_SEED"] = 123
    _update_node_sample(G2, step=5)
    sample2 = G2.graph["_node_sample"]

    assert sample1 == sample2


def test_node_sample_adjacency_cache_tracks_mutations(graph_canon):
    pytest.importorskip("numpy")

    G = _build_graph(6, graph_canon)
    G.add_edges_from((i, i + 1) for i in range(5))
    G.add_edge(0, 5)

    _update_node_sample(G, step=0)
    nodes1, A1 = cached_nodes_and_A(G, require_numpy=True)
    assert isinstance(nodes1, tuple)
    assert A1 is not None and A1.shape == (6, 6)

    nodes2, A2 = cached_nodes_and_A(G, require_numpy=True)
    assert nodes2 is nodes1
    assert A2 is A1

    G.add_node(99)
    G.add_edge(0, 99)
    increment_edge_version(G)
    _update_node_sample(G, step=1)

    nodes3, A3 = cached_nodes_and_A(G, require_numpy=True)
    assert len(nodes3) == 7
    assert nodes3 is not nodes1
    assert A3 is not None and A3.shape == (7, 7)

    nodes4, A4 = cached_nodes_and_A(G, require_numpy=True)
    assert nodes4 is nodes3
    assert A4 is A3


def _run_sample_with_hashseed(hashseed):
    code = r"""
import json, sys
import networkx as nx
from tnfr.constants import inject_defaults
from tnfr.dynamics import _update_node_sample

G = nx.Graph()
inject_defaults(G)
for i in range(80):
    G.add_node(i, θ=0.0, EPI=0.0)
G.graph["UM_CANDIDATE_COUNT"] = 10
G.graph["RANDOM_SEED"] = 123
_update_node_sample(G, step=5)
json.dump(G.graph["_node_sample"], sys.stdout)
"""
    env = dict(
        os.environ,
        PYTHONHASHSEED=str(hashseed),
        PYTHONPATH=os.pathsep.join(
            [os.getcwd(), os.path.join(os.getcwd(), "src")]
        ),
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    return json.loads(result.stdout)


def test_node_sample_deterministic_across_hashseed():
    out1 = _run_sample_with_hashseed(1)
    out2 = _run_sample_with_hashseed(2)
    assert out1 == out2
