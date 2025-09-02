"""Pruebas de node sample."""
from tnfr.dynamics import step
from tnfr.constants import attach_defaults
import networkx as nx
import json
import os
import subprocess
import sys


def _build_graph(n):
    G = nx.Graph()
    attach_defaults(G)
    for i in range(n):
        G.add_node(i, **{"θ": 0.0, "EPI": 0.0})
    return G


def test_node_sample_large_graph():
    G = _build_graph(80)
    G.graph["UM_CANDIDATE_COUNT"] = 10
    step(G, use_Si=False, apply_glyphs=False)
    sample = G.graph.get("_node_sample")
    assert isinstance(sample, list)
    assert len(sample) == 10
    assert set(sample).issubset(set(G.nodes()))


def test_node_sample_small_graph():
    G = _build_graph(20)
    G.graph["UM_CANDIDATE_COUNT"] = 5
    step(G, use_Si=False, apply_glyphs=False)
    sample = G.graph.get("_node_sample")
    assert len(sample) == len(G.nodes())


def _run_sample_with_hashseed(hashseed):
    code = r"""
import json
import networkx as nx
from tnfr.constants import attach_defaults
from tnfr.dynamics import _update_node_sample

def _build_graph(n):
    G = nx.Graph()
    attach_defaults(G)
    for i in range(n):
        G.add_node(i, θ=0.0, EPI=0.0)
    return G

G = _build_graph(80)
G.graph["UM_CANDIDATE_COUNT"] = 10
G.graph["RANDOM_SEED"] = 123
_update_node_sample(G, step=5)
print(json.dumps(G.graph["_node_sample"]))
"""
    env = dict(os.environ, PYTHONHASHSEED=str(hashseed))
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env
    )
    return json.loads(result.stdout)


def test_node_sample_deterministic_across_hashseed():
    out1 = _run_sample_with_hashseed(1)
    out2 = _run_sample_with_hashseed(2)
    assert out1 == out2
