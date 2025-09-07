import networkx as nx
from tnfr.rng import base_seed


def test_base_seed_returns_value():
    G = nx.Graph()
    G.graph["RANDOM_SEED"] = 123
    assert base_seed(G) == 123


def test_base_seed_defaults_to_zero():
    G = nx.Graph()
    assert base_seed(G) == 0
