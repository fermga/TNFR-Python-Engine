"""Pruebas de node sample."""

from tnfr.dynamics import step, _update_node_sample
from tnfr.rng import get_rng
from tests.utils import build_graph
import json
import os
import subprocess
import sys


def test_node_sample_large_graph():
    G = build_graph(80)
    G.graph["UM_CANDIDATE_COUNT"] = 10
    step(G, use_Si=False, apply_glyphs=False)
    sample = G.graph.get("_node_sample")
    assert isinstance(sample, list)
    assert len(sample) == 10
    assert set(sample).issubset(set(G.nodes()))


def test_node_sample_small_graph():
    G = build_graph(20)
    G.graph["UM_CANDIDATE_COUNT"] = 5
    step(G, use_Si=False, apply_glyphs=False)
    sample = G.graph.get("_node_sample")
    assert not isinstance(sample, list)
    assert len(sample) == len(G.nodes())


def test_node_sample_immutable_after_graph_change():
    G = build_graph(20)
    _update_node_sample(G, step=0)
    sample = G.graph["_node_sample"]
    G.add_node(99)
    assert len(sample) == 20
    assert 99 not in sample


def test_node_sample_deterministic_with_seed():
    get_rng.cache_clear()
    G1 = build_graph(80)
    G1.graph["UM_CANDIDATE_COUNT"] = 10
    G1.graph["RANDOM_SEED"] = 123
    _update_node_sample(G1, step=5)
    sample1 = G1.graph["_node_sample"]

    get_rng.cache_clear()
    G2 = build_graph(80)
    G2.graph["UM_CANDIDATE_COUNT"] = 10
    G2.graph["RANDOM_SEED"] = 123
    _update_node_sample(G2, step=5)
    sample2 = G2.graph["_node_sample"]

    assert sample1 == sample2


def _run_sample_with_hashseed(hashseed):
    code = r"""
import json, sys
from tnfr.dynamics import _update_node_sample
from tests.utils import build_graph

G = build_graph(80)
G.graph["UM_CANDIDATE_COUNT"] = 10
G.graph["RANDOM_SEED"] = 123
_update_node_sample(G, step=5)
json.dump(G.graph["_node_sample"], sys.stdout)
"""
    env = dict(
        os.environ,
        PYTHONHASHSEED=str(hashseed),
        PYTHONPATH=os.pathsep.join([os.getcwd(), os.path.join(os.getcwd(), "src")]),
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
