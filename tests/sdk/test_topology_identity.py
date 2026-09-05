"""Topology scaffolds preserve existing NFR identity and declared parameters."""

import math
from copy import deepcopy

import networkx as nx
import pytest

from tnfr.sdk.fluent import TNFRNetwork
from tnfr.sdk.simple import Network, TNFR


@pytest.mark.parametrize("topology,kwargs", [
    ("small_world", {"k": 2, "p": 0.4, "seed": 3}),
    ("scale_free", {"m": 1, "seed": 3}),
    ("grid", {}),
])
def test_generated_topology_uses_existing_mixed_node_labels(topology, kwargs):
    labels = ["a", ("b", 1), 9, "d", frozenset({5}), "f"]
    graph = nx.Graph()
    graph.add_nodes_from((node, {"identity": index}) for index, node in enumerate(labels))
    before = deepcopy(dict(graph.nodes(data=True)))
    getattr(Network(graph), topology)(**kwargs)
    assert list(graph) == labels
    assert dict(graph.nodes(data=True)) == before
    assert nx.is_connected(graph)


@pytest.mark.parametrize("topology,kwargs", [
    ("small_world", {"k": 4, "p": 0.7}),
    ("scale_free", {"m": 2}),
])
def test_network_seed_is_the_default_for_every_stochastic_topology(topology, kwargs):
    actual = TNFR.create(20, seed=13)
    explicit = TNFR.create(20)
    getattr(actual, topology)(**kwargs)
    getattr(explicit, topology)(**kwargs, seed=13)
    assert set(actual.G.edges()) == set(explicit.G.edges())


@pytest.mark.parametrize("size", [2, 3, 5, 7, 11])
def test_default_grid_includes_every_node(size):
    network = TNFR.create(size).grid()
    assert len(network.G) == size
    assert nx.is_connected(network.G)


def test_insufficient_explicit_grid_capacity_fails_before_adding_edges():
    network = TNFR.create(5)
    with pytest.raises(ValueError, match="capacity"):
        network.grid(rows=2, cols=2)
    assert network.G.number_of_edges() == 0


def test_star_rejects_an_unknown_center_without_creating_a_bare_node():
    network = TNFR.create(3)
    before = deepcopy(dict(network.G.nodes(data=True)))
    with pytest.raises(ValueError, match="center"):
        network.star(center="missing")
    assert dict(network.G.nodes(data=True)) == before
    assert network.G.number_of_edges() == 0


def test_empty_star_is_an_empty_topology():
    assert len(TNFR.create(0).star().G) == 0


@pytest.mark.parametrize("api", ["simple", "fluent"])
def test_single_node_ring_has_no_self_coupling(api):
    graph = (TNFR.create(1).ring().G if api == "simple"
             else TNFRNetwork().add_nodes(1).connect_nodes(connection_pattern="ring").graph)
    assert graph.number_of_edges() == 0


@pytest.mark.parametrize("probability", [-0.1, 1.1, math.nan, math.inf])
@pytest.mark.parametrize("api", ["simple", "fluent"])
def test_invalid_random_probability_fails_before_topology_changes(api, probability):
    network = TNFR.create(4) if api == "simple" else TNFRNetwork().add_nodes(4)
    with pytest.raises(ValueError, match="probability"):
        if api == "simple":
            network.random(probability)
        else:
            network.connect_nodes(probability)
    graph = network.G if api == "simple" else network.graph
    assert graph.number_of_edges() == 0
