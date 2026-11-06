"""Diagnostic recording preserves the shared word invocation boundary."""

from copy import deepcopy

import networkx as nx
import pytest

from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.metrics import core
from tnfr.operators import word_execution
from tnfr.sdk import TNFR, Network
from tnfr.sdk.fluent import NAMED_SEQUENCES


def _physical_state(graph):
    return {
        node: tuple(
            get_attr(data, aliases, None)
            for aliases in (ALIAS_EPI, ALIAS_VF, ALIAS_THETA, ALIAS_DNFR)
        )
        for node, data in graph.nodes(data=True)
    }


def test_recording_matches_ordinary_c3_execution_and_samples_completed_words():
    plain = TNFR.create(3, seed=5).ring()
    recorded = TNFR.create(3, seed=5).ring()

    plain.evolve(2)
    recorded.evolve(2, record=True)

    assert _physical_state(recorded.G) == _physical_state(plain.G)
    assert list(recorded.G.edges) == list(plain.G.edges)
    assert recorded.history()["C_steps"] == pytest.approx([1.0, 1.0])
    assert recorded.history()["kuramoto_R"] == pytest.approx([1.0, 1.0])
    assert plain.history()["C_steps"] == []


@pytest.mark.parametrize("record", [False, True])
def test_recording_retains_initial_targets_when_a_stage_adds_support(
    monkeypatch, record
):
    graph = nx.Graph()
    graph.add_nodes_from(("last", "first"))
    network = Network(graph)
    targets_seen = []
    observations = []

    def add_support(graph, operator, targets, **kwargs):
        targets_seen.append(tuple(targets))
        for node in targets:
            graph.nodes[node]["visits"] = graph.nodes[node].get("visits", 0) + 1
        if len(targets_seen) == 1:
            graph.add_node("child", visits=0)

    monkeypatch.setattr(word_execution, "execute_network_operator_stage", add_support)
    monkeypatch.setattr(
        core, "_metrics_step", lambda graph: observations.append(len(targets_seen))
    )

    network.evolve(2, record=record)

    assert targets_seen == [("last", "first")] * 12
    assert [graph.nodes[node]["visits"] for node in ("last", "first", "child")] == [
        12,
        12,
        0,
    ]
    assert observations == ([6, 12] if record else [])


@pytest.mark.parametrize("record", [False, True])
def test_recording_keeps_whole_request_mutation_preflight(monkeypatch, record):
    graph = nx.Graph()
    graph.add_node(
        "source",
        EPI=0.2,
        nu_f=1.0,
        phase=0.0,
        delta_nfr=0.0,
        epi_time_history=[(0.0, 0.0), (1.0, 0.2)],
    )
    graph.graph["ZHIR_THRESHOLD_XI"] = 0.1
    before = deepcopy(dict(graph.nodes(data=True)))
    monkeypatch.setitem(
        NAMED_SEQUENCES,
        "temporal_control",
        [
            "transition",
            "coherence",
            "dissonance",
            "mutation",
            "coherence",
            "resonance",
            "silence",
        ],
    )
    stages = []
    monkeypatch.setattr(
        word_execution,
        "execute_network_operator_stage",
        lambda *args, **kwargs: stages.append(args[1].name),
    )
    monkeypatch.setattr(core, "_metrics_step", lambda graph: None)

    with pytest.raises(ValueError, match="Mutation sequence preflight") as failure:
        Network(graph).evolve(2, sequence="temporal_control", record=record)

    assert (
        failure.value.context["reason"] == "physical_mutation_evidence_would_be_stale"
    )
    assert stages == []
    assert dict(graph.nodes(data=True)) == before


def test_failed_later_stage_keeps_only_completed_word_observations(monkeypatch):
    network = TNFR.create(1)
    stages = []
    samples = []

    def bounded_stage(graph, operator, targets, **kwargs):
        if len(stages) == 8:
            raise ValueError("stage unavailable")
        stages.append(operator.name)
        graph.nodes[0]["completed_stages"] = len(stages)

    monkeypatch.setattr(word_execution, "execute_network_operator_stage", bounded_stage)
    monkeypatch.setattr(
        core, "_metrics_step", lambda graph: samples.append(len(stages))
    )

    with pytest.raises(ValueError, match="stage unavailable"):
        network.evolve(3, record=True)

    assert len(stages) == network.G.nodes[0]["completed_stages"] == 8
    assert samples == [6]


def test_failed_observation_does_not_undo_a_completed_word_or_start_the_next(
    monkeypatch,
):
    network = TNFR.create(1)
    stages = []

    def stage(graph, operator, targets, **kwargs):
        stages.append(operator.name)
        graph.nodes[0]["completed_stages"] = len(stages)

    def unavailable_metrics(graph):
        raise ValueError("diagnostic unavailable")

    monkeypatch.setattr(word_execution, "execute_network_operator_stage", stage)
    monkeypatch.setattr(core, "_metrics_step", unavailable_metrics)

    with pytest.raises(ValueError, match="diagnostic unavailable"):
        network.evolve(2, record=True)

    assert len(stages) == network.G.nodes[0]["completed_stages"] == 6
