"""SDK seed, copy and initialization contracts are local to each experiment."""

import random
from copy import deepcopy

import networkx as nx
import pytest

from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.sdk import fluent
from tnfr.sdk.builders import TNFRExperimentBuilder as Builders
from tnfr.sdk.fluent import NetworkConfig, TNFRNetwork


def _template(name):
    def run(**kwargs):
        from tnfr.sdk.templates import TNFRTemplates
        return getattr(TNFRTemplates, name)(**kwargs)
    return run


def _triad(graph):
    return [tuple(get_attr(data, alias) for alias in (ALIAS_EPI, ALIAS_VF, ALIAS_THETA))
            for _, data in graph.nodes(data=True)]


@pytest.mark.parametrize("factory,kwargs", [
    (Builders.small_world_study, {"nodes": 6, "steps": 0}),
    (Builders.synchronization_study, {"nodes": 6, "steps": 0}),
    (Builders.creativity_emergence, {"nodes": 6, "steps": 0}),
    (_template("social_network_simulation"), {"people": 6, "simulation_steps": 0}),
    (_template("neural_network_model"), {"neurons": 6, "activation_cycles": 0}),
    (_template("ecosystem_dynamics"), {"species": 6, "evolution_steps": 0}),
    (_template("creative_process_model"), {"ideas": 6, "development_cycles": 0}),
    (_template("organizational_network"), {"agents": 6, "coordination_steps": 0}),
])
def test_builder_seed_controls_actual_triad_and_topology(factory, kwargs):
    first = factory(**kwargs, random_seed=0)
    second = factory(**kwargs, random_seed=0)
    assert _triad(first.graph) == _triad(second.graph)
    assert set(first.graph.edges()) == set(second.graph.edges())


def test_creativity_builder_stores_mutation_intensity_in_glyph_factors():
    result = Builders.creativity_emergence(
        nodes=6, mutation_intensity=0.25, steps=0, random_seed=7
    )

    assert result.graph.graph["GLYPH_FACTORS"]["ZHIR_theta_shift_factor"] == 0.25
    assert "ZHIR_theta_shift_factor" not in result.graph.graph


def test_creativity_builder_reports_fresh_network_mutation_abstention(monkeypatch):
    original = TNFRNetwork.add_nodes

    def coherent_nodes(self, *args, **kwargs):
        kwargs["phase_range"] = (0.0, 0.0)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(TNFRNetwork, "add_nodes", coherent_nodes)
    result = Builders.creativity_emergence(
        nodes=6, mutation_intensity=0.25, steps=1, random_seed=7
    )

    assert result.mutation_workflows[0]["status"] == "mutation_abstained"
    assert result.mutation_workflows[0]["reason"] == "missing_history"
    assert all(
        "ZHIR" not in data["glyph_history"]
        for _, data in result.graph.nodes(data=True)
    )


def test_topology_comparison_starts_with_identical_nodes():
    results = Builders.compare_topologies(node_count=6, steps=0, random_seed=5)
    assert _triad(results["random"].graph) == _triad(results["ring"].graph)
    assert _triad(results["ring"].graph) == _triad(results["small_world"].graph)


def test_coupling_comparison_starts_with_identical_nodes():
    results = Builders.phase_transition_study(
        nodes=6, steps_per_level=0, coupling_levels=3, random_seed=5,
    )
    triads = [_triad(result.graph) for result in results.values()]
    assert triads[0] == triads[1] == triads[2]


def test_fallback_network_rng_does_not_reseed_or_consume_global_random(monkeypatch):
    monkeypatch.setattr(fluent, "_HAS_NUMPY", False)
    before = random.getstate()
    TNFRNetwork(config=NetworkConfig(random_seed=0)).add_nodes(4).connect_nodes(0.5)
    assert random.getstate() == before


def test_fallback_rng_streams_are_independent_when_interleaved(monkeypatch):
    monkeypatch.setattr(fluent, "_HAS_NUMPY", False)
    reference = TNFRNetwork(config=NetworkConfig(random_seed=7)).add_nodes(4)
    actual = TNFRNetwork(config=NetworkConfig(random_seed=7)).add_nodes(2)
    TNFRNetwork(config=NetworkConfig(random_seed=9)).add_nodes(3)
    actual.add_nodes(2)
    assert _triad(reference.graph) == _triad(actual.graph)


@pytest.mark.parametrize("graph_type", [nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph])
def test_clone_retains_graph_kind_and_parallel_edge_keys(graph_type):
    graph = graph_type()
    graph.add_edge("left", "right", weight=2)
    if graph.is_multigraph():
        graph.add_edge("left", "right", key="second", weight=3)
    network = TNFRNetwork()
    network._graph = graph
    cloned = network.clone()
    assert type(cloned.graph) is graph_type
    assert nx.utils.graphs_equal(cloned.graph, graph)


def test_clone_detaches_nested_node_edge_and_configuration_values():
    network = TNFRNetwork(config=NetworkConfig(random_seed=7)).add_nodes(2)
    network.graph.nodes["node_0"]["nested"] = {"child": [1]}
    network.graph.add_edge("node_0", "node_1", nested={"weight": [2]})
    cloned = network.clone()
    cloned.graph.nodes["node_0"]["nested"]["child"].append(3)
    cloned.graph.edges["node_0", "node_1"]["nested"]["weight"].append(4)
    cloned._config.default_epi_range = (0.4, 0.5)
    assert network.graph.nodes["node_0"]["nested"] == {"child": [1]}
    assert network.graph.edges["node_0", "node_1"]["nested"] == {"weight": [2]}
    assert network._config.default_epi_range == (0.1, 0.9)


def test_clone_continues_the_current_rng_state_independently():
    network = TNFRNetwork(config=NetworkConfig(random_seed=7)).add_nodes(3)
    cloned = network.clone()
    network.add_nodes(2)
    cloned.add_nodes(2)
    assert _triad(network.graph) == _triad(cloned.graph)


def test_measured_and_cloned_evolved_graphs_detach_history_and_runtime_adapters():
    from tnfr.operators.definitions import Coherence

    network = TNFRNetwork(config=NetworkConfig(random_seed=0)).add_nodes(6)
    network.connect_nodes(connection_pattern="ring")
    network.apply_sequence(["emission", "coherence", "silence"])
    result = network.measure()
    cloned = network.clone()
    recorded = list(result.graph.nodes["node_0"]["glyph_history"])
    original = list(network.graph.nodes["node_0"]["glyph_history"])
    Coherence()(cloned.graph, "node_0")
    assert list(network.graph.nodes["node_0"]["glyph_history"]) == original
    Coherence()(network.graph, "node_0")
    assert list(result.graph.nodes["node_0"]["glyph_history"]) == recorded
    assert result.graph is not network.graph


@pytest.mark.parametrize("numpy_available", [False, True])
def test_per_operation_seed_does_not_advance_network_rng(monkeypatch, numpy_available):
    monkeypatch.setattr(fluent, "_HAS_NUMPY", numpy_available)
    actual = TNFRNetwork(config=NetworkConfig(random_seed=2))
    reference = TNFRNetwork(config=NetworkConfig(random_seed=2))
    actual.add_nodes(2, random_seed=9)
    actual.add_nodes(2)
    reference.add_nodes(2)
    assert _triad(actual.graph)[-2:] == _triad(reference.graph)


def test_clone_preserves_opaque_node_identity():
    class Node:
        def __deepcopy__(self, memo):
            raise AssertionError("node identity must not be copied")

    node = Node()
    network = TNFRNetwork()
    network._graph = nx.Graph()
    network.graph.add_node(node, values=[1])
    cloned = network.clone()
    assert next(iter(cloned.graph)) is node
    cloned.graph.nodes[node]["values"].append(2)
    assert network.graph.nodes[node]["values"] == [1]


@pytest.mark.parametrize("kwargs", [
    {"count": -1}, {"count": 1.5}, {"count": True},
    {"count": 2, "epi_range": (float("nan"), 1)},
    {"count": 2, "vf_range": (2, 1)},
])
def test_invalid_node_generation_leaves_existing_state_and_rng_unchanged(kwargs):
    network = TNFRNetwork(config=NetworkConfig(random_seed=7)).add_nodes(2)
    before = deepcopy(dict(network.graph.nodes(data=True)))
    counter = network._node_counter
    rng_state = deepcopy(network._rng.get_state())
    with pytest.raises((ValueError, TypeError)):
        network.add_nodes(**kwargs)
    assert dict(network.graph.nodes(data=True)) == before
    assert network._node_counter == counter
    assert network._rng.get_state()[2:] == rng_state[2:]
