"""Template controls must have deterministic graph and cycle semantics."""

import networkx as nx
import pytest

from tnfr.sdk.fluent import NAMED_SEQUENCES, TNFRNetwork
from tnfr.sdk.templates import TNFRTemplates as Templates


@pytest.mark.parametrize("people,degree", [(1, 0), (2, 1), (12, 0), (12, 2), (12, 5), (12, 8), (6, 5), (9, 4)])
def test_social_contact_parameter_sets_realized_mean_degree(people, degree):
    graph = Templates.social_network_simulation(
        people=people, connections_per_person=degree, simulation_steps=0, random_seed=7,
    ).graph
    assert graph.number_of_nodes() == people
    assert 2 * graph.number_of_edges() == people * degree
    assert nx.number_of_selfloops(graph) == 0


def test_social_contact_scaffold_is_repeatable():
    kwargs = dict(people=12, connections_per_person=5, simulation_steps=0, random_seed=7)
    first = Templates.social_network_simulation(**kwargs).graph
    second = Templates.social_network_simulation(**kwargs).graph
    assert set(first.edges()) == set(second.edges())
    assert dict(first.nodes(data=True)) == dict(second.nodes(data=True))


def test_inspiration_controls_rewiring_without_changing_node_initialization():
    kwargs = dict(ideas=12, development_cycles=0, random_seed=7)
    lattice = Templates.creative_process_model(**kwargs, inspiration_level=0).graph
    rewired = Templates.creative_process_model(**kwargs, inspiration_level=1).graph
    repeated = Templates.creative_process_model(**kwargs, inspiration_level=1).graph
    assert set(lattice.edges()) != set(rewired.edges())
    assert lattice.number_of_edges() == rewired.number_of_edges()
    assert dict(lattice.nodes(data=True)) == dict(rewired.nodes(data=True))
    assert set(rewired.edges()) == set(repeated.edges())
    assert nx.number_of_selfloops(rewired) == 0


@pytest.mark.parametrize("size,depth", [(1, 1), (12, 1), (12, 2), (12, 4), (12, 12)])
def test_hierarchy_depth_constructs_connected_nonempty_levels(size, depth):
    graph = Templates.organizational_network(
        agents=size, hierarchy_depth=depth, coordination_steps=0, random_seed=7,
    ).graph
    levels = nx.get_node_attributes(graph, "hierarchy_level")
    assert set(levels) == set(graph)
    assert set(levels.values()) == set(range(depth))
    assert nx.is_connected(graph)
    assert nx.number_of_selfloops(graph) == 0
    assert all(abs(levels[u] - levels[v]) <= 1 for u, v in graph.edges())
    if depth > 1:
        root = next(node for node, level in levels.items() if level == 0)
        assert dict(nx.single_source_shortest_path_length(graph, root)) == levels
    if depth == size and size > 1:
        assert graph.number_of_edges() == size - 1


def test_hierarchy_parameter_changes_topology_reproducibly():
    kwargs = dict(agents=12, coordination_steps=0, random_seed=7)
    shallow = Templates.organizational_network(**kwargs, hierarchy_depth=2).graph
    deep = Templates.organizational_network(**kwargs, hierarchy_depth=4).graph
    repeated = Templates.organizational_network(**kwargs, hierarchy_depth=4).graph
    assert set(shallow.edges()) != set(deep.edges())
    assert set(deep.edges()) == set(repeated.edges())
    assert dict(deep.nodes(data=True)) == dict(repeated.nodes(data=True))


@pytest.mark.parametrize("budget", [0, 1, 2, 3, 9, 10, 11, 20, 30])
def test_ecosystem_budget_counts_canonical_word_applications(monkeypatch, budget):
    scheduled = []

    def record(self, sequence, repeat=1, **kwargs):
        scheduled.extend([sequence] * repeat)
        return self

    monkeypatch.setattr(TNFRNetwork, "apply_sequence", record)
    Templates.ecosystem_dynamics(species=6, evolution_steps=budget, random_seed=7)
    words = ("exploration", "network_sync", "consolidation")
    assert scheduled == [words[index % 3] for index in range(budget)]


@pytest.mark.parametrize("method,kwargs", [
    ("social_network_simulation", {"people": 0}),
    ("social_network_simulation", {"people": 9, "connections_per_person": 3}),
    ("social_network_simulation", {"people": 6, "connections_per_person": 6}),
    ("social_network_simulation", {"connections_per_person": -1}),
    ("social_network_simulation", {"simulation_steps": -1}),
    ("creative_process_model", {"ideas": 0}),
    ("creative_process_model", {"inspiration_level": -0.1}),
    ("creative_process_model", {"inspiration_level": 1.1}),
    ("creative_process_model", {"inspiration_level": float("nan")}),
    ("creative_process_model", {"development_cycles": 1.5}),
    ("organizational_network", {"agents": 0}),
    ("organizational_network", {"agents": 6, "hierarchy_depth": 7}),
    ("organizational_network", {"hierarchy_depth": 0}),
    ("organizational_network", {"hierarchy_depth": True}),
    ("ecosystem_dynamics", {"species": 0}),
    ("ecosystem_dynamics", {"evolution_steps": -1}),
    ("ecosystem_dynamics", {"evolution_steps": True}),
])
def test_invalid_template_controls_fail_before_node_creation(monkeypatch, method, kwargs):
    created = []
    original = TNFRNetwork.add_nodes

    def record(self, *args, **kw):
        created.append(args)
        return original(self, *args, **kw)

    monkeypatch.setattr(TNFRNetwork, "add_nodes", record)
    with pytest.raises(ValueError):
        getattr(Templates, method)(**kwargs, random_seed=7)
    assert created == []


@pytest.mark.parametrize("method", ["social", "ecosystem", "creative", "organization"])
@pytest.mark.parametrize("budget", [1, 2, 3])
def test_real_template_cycles_follow_exact_words_and_repeat(monkeypatch, method, budget):
    from tnfr.operators import word_execution
    from tnfr.operators.network_stage import STAGE_SCHEDULE_KEY
    from tnfr.operators.registry import get_operator_class
    from tnfr.operators.stage_contracts import stage_contract_for

    # Explicit coherent initial phases isolate cycle accounting from random
    # U3 admissibility. Operators and their hard phase gates remain active.
    original_nodes = TNFRNetwork.add_nodes
    original_run = word_execution.run_network_sequence
    callbacks = []

    def coherent_nodes(self, *args, **kwargs):
        kwargs["phase_range"] = (0.0, 0.0)
        return original_nodes(self, *args, **kwargs)

    def traced_run(graph, operator_names, **kwargs):
        caller_callback = kwargs.pop("on_step", None)

        def record_committed_stage(operator_name):
            callbacks.append(
                {
                    "operator": operator_name,
                    "schedule": dict(graph.graph[STAGE_SCHEDULE_KEY]),
                    "histories": {
                        node: tuple(graph.nodes[node]["glyph_history"])
                        for node in graph
                    },
                }
            )
            if caller_callback is not None:
                caller_callback(operator_name)

        return original_run(
            graph,
            operator_names,
            on_step=record_committed_stage,
            **kwargs,
        )

    monkeypatch.setattr(TNFRNetwork, "add_nodes", coherent_nodes)
    monkeypatch.setattr(word_execution, "run_network_sequence", traced_run)
    if method == "social":
        factory = Templates.social_network_simulation
        kwargs = {"people": 6, "connections_per_person": 5, "simulation_steps": budget}
        third = budget // 3
        words = (["basic_activation"] * third + ["network_sync"] * third
                 + ["consolidation"] * (budget - 2 * third))
    elif method == "ecosystem":
        factory = Templates.ecosystem_dynamics
        kwargs = {"species": 6, "interaction_strength": 1.0, "evolution_steps": budget}
        cycle = ("exploration", "network_sync", "consolidation")
        words = [cycle[index % 3] for index in range(budget)]
    elif method == "creative":
        factory = Templates.creative_process_model
        kwargs = {"ideas": 6, "inspiration_level": 0.4, "development_cycles": budget}
        third = budget // 3
        words = (["exploration"] * third + ["exploration"] * third
                 + ["network_sync"] * (budget - 2 * third))
    else:
        factory = Templates.organizational_network
        kwargs = {"agents": 6, "hierarchy_depth": 3, "coordination_steps": budget}
        half = budget // 2
        words = ["network_sync"] * half + ["consolidation"] * (budget - half)

    expected_operators = [
        name for word in words for name in NAMED_SEQUENCES[word]
    ]
    expected_glyphs = [
        get_operator_class(name)().glyph.value for name in expected_operators
    ]

    def assert_execution_trace(records):
        assert [record["operator"] for record in records] == expected_operators
        for index, (record, operator_name, glyph) in enumerate(
            zip(records, expected_operators, expected_glyphs, strict=True)
        ):
            schedule = record["schedule"]
            assert schedule == {
                "operator": operator_name,
                "glyph": glyph,
                "schedule": stage_contract_for(operator_name).current_schedule.value,
                "nodes_processed": 6,
            }
            # The callback runs after the whole stage commits. Every target
            # must therefore expose the same newly appended glyph. Compare the
            # complete observable suffix so bounded histories remain valid.
            expected_prefix = tuple(expected_glyphs[: index + 1])
            for history in record["histories"].values():
                assert history == expected_prefix[-len(history) :]

    first = factory(**kwargs, random_seed=7)
    first_trace = list(callbacks)
    assert_execution_trace(first_trace)
    callbacks.clear()
    second = factory(**kwargs, random_seed=7)
    second_trace = list(callbacks)
    assert_execution_trace(second_trace)
    assert first_trace == second_trace
    assert first.coherence == second.coherence
    assert first.sense_indices == second.sense_indices
    assert set(first.graph.edges()) == set(second.graph.edges())
    for node in first.graph:
        assert list(first.graph.nodes[node]["glyph_history"]) == list(second.graph.nodes[node]["glyph_history"])


def test_sampled_template_phases_still_require_the_live_u3_gate():
    from copy import deepcopy

    from tnfr.alias import get_attr
    from tnfr.constants.aliases import ALIAS_THETA
    from tnfr.constants.canonical import DELTA_PHI_MAX
    from tnfr.operators.definitions import Coupling
    from tnfr.operators.preconditions import OperatorPreconditionError
    from tnfr.utils import angle_diff

    graph = Templates.creative_process_model(
        ideas=12,
        inspiration_level=0.4,
        development_cycles=0,
        random_seed=7,
    ).graph

    def phase(node):
        return get_attr(graph.nodes[node], ALIAS_THETA, 0.0)

    inadmissible = tuple(
        node
        for node in graph
        if graph.degree(node) > 0
        and all(
            abs(angle_diff(phase(node), phase(neighbor))) > DELTA_PHI_MAX
            for neighbor in graph.neighbors(node)
        )
    )
    assert inadmissible
    target = inadmissible[0]

    def mutation_surface():
        return {
            "nodes": deepcopy(dict(graph.nodes(data=True))),
            "edges": deepcopy(tuple(graph.edges(data=True))),
            "graph": tuple(
                (key, repr(value)) for key, value in graph.graph.items()
            ),
            "last_operator": (
                hasattr(graph, "_last_operator_applied"),
                getattr(graph, "_last_operator_applied", None),
            ),
        }

    before = mutation_surface()

    with pytest.raises(OperatorPreconditionError, match="U3 phase gate"):
        Coupling()(graph, target)

    assert mutation_surface() == before


def test_template_topology_helpers_preserve_mixed_node_identities():
    import random

    from tnfr.sdk._topology import hierarchical_edges, rewired_contact_edges

    opaque = object()
    nodes = [opaque, "a", ("b", 2), frozenset({3}), 4, "c"]
    contacts = rewired_contact_edges(nodes, 3, 1.0, random.Random(7))
    hierarchy, levels = hierarchical_edges(nodes, 3)
    assert len(contacts) == 9
    assert set(levels) == set(nodes)
    assert all(node in nodes for edge in contacts + hierarchy for node in edge)
    assert any(node is opaque for edge in contacts + hierarchy for node in edge)
