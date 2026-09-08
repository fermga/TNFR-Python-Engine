"""Grammar and reproducibility contracts for adaptive word selection."""

from __future__ import annotations

import math

import networkx as nx
import pytest

from tnfr.constants.aliases import ALIAS_DNFR
from tnfr.dynamics.adaptive_sequences import AdaptiveSequenceSelector
from tnfr.operators.grammar import validate_sequence
from tnfr.sdk.adaptive_system import TNFRAdaptiveSystem


def _graph(seed: int = 17) -> nx.Graph:
    graph = nx.Graph()
    graph.graph["RANDOM_SEED"] = seed
    graph.add_node("node")
    return graph


def test_catalogue_contains_only_standalone_grammar_words():
    selector = AdaptiveSequenceSelector(_graph(), "node")

    verdicts = {
        name: validate_sequence(word)
        for name, word in selector.sequences.items()
    }
    assert all(verdict.passed for verdict in verdicts.values()), {
        name: verdict.error
        for name, verdict in verdicts.items()
        if not verdict.passed
    }


def test_graph_seed_drives_a_reproducible_local_stream():
    first = AdaptiveSequenceSelector(_graph(23), "node")
    second = AdaptiveSequenceSelector(_graph(23), "node")
    contexts = [
        {"goal": goal}
        for goal in ("stability", "growth", "adaptation", "unknown")
        for _ in range(20)
    ]

    assert first.seed == second.seed == 23
    assert [
        first.select_sequence(context) for context in contexts
    ] == [
        second.select_sequence(context) for context in contexts
    ]


def test_selected_word_is_detached_from_the_internal_catalogue():
    selector = AdaptiveSequenceSelector(_graph(), "node", seed=3)
    selected = selector.select_sequence({"goal": "stability"})
    selected.append("dissonance")

    assert all(
        word[-1] != "dissonance"
        for word in selector.sequences.values()
    )


def test_performance_score_name_and_legacy_alias_share_one_history():
    selector = AdaptiveSequenceSelector(_graph(), "node")
    selector.record_performance(
        "basic_activation",
        performance_score=0.5,
    )
    selector.record_performance("basic_activation", 0.25)

    assert selector.performance is selector.performance_scores
    assert selector.performance_scores["basic_activation"] == [0.5, 0.25]


@pytest.mark.parametrize("value", [None, True, math.nan, math.inf, "0.5"])
def test_performance_score_must_be_finite(value):
    selector = AdaptiveSequenceSelector(_graph(), "node")
    with pytest.raises(ValueError, match="finite real"):
        selector.record_performance(
            "basic_activation",
            performance_score=value,
        )


def test_performance_alias_conflicts_and_unknown_names_are_rejected():
    selector = AdaptiveSequenceSelector(_graph(), "node")
    with pytest.raises(ValueError, match="aliases"):
        selector.record_performance(
            "basic_activation",
            0.1,
            performance_score=0.2,
        )
    with pytest.raises(KeyError, match="unknown sequence"):
        selector.record_performance("missing", performance_score=0.1)


def test_performance_history_retains_the_last_twenty_scores():
    selector = AdaptiveSequenceSelector(_graph(), "node")
    for value in range(25):
        selector.record_performance(
            "basic_activation",
            performance_score=float(value),
        )

    assert selector.performance_scores["basic_activation"] == [
        float(value) for value in range(5, 25)
    ]


@pytest.mark.parametrize("seed", [True, 1.5, "3"])
def test_explicit_seed_uses_the_shared_strict_domain(seed):
    with pytest.raises(ValueError, match="RANDOM_SEED"):
        AdaptiveSequenceSelector(_graph(), "node", seed=seed)

@pytest.mark.parametrize(
    "normalization",
    [0.0, -0.1, True, math.nan, math.inf, "0.1"],
)
def test_adaptive_system_rejects_invalid_stress_normalization(normalization):
    with pytest.raises(ValueError, match="positive finite real"):
        TNFRAdaptiveSystem(
            _graph(),
            "node",
            stress_normalization=normalization,
        )


@pytest.mark.parametrize("cycles", [-1, True, 1.5])
def test_adaptive_system_rejects_invalid_cycle_count(cycles):
    system = TNFRAdaptiveSystem(_graph(), "node")
    with pytest.raises(ValueError, match="nonnegative integer"):
        system.autonomous_evolution(cycles)


def test_adaptive_system_propagates_selector_seed():
    system = TNFRAdaptiveSystem(_graph(7), "node", random_seed=29)
    assert system.sequence_selector.seed == 29


@pytest.mark.parametrize("pressure", [True, math.nan, math.inf, "0.1"])
def test_adaptive_system_rejects_invalid_pressure_stress_input(pressure):
    graph = _graph()
    graph.nodes["node"][ALIAS_DNFR[0]] = pressure
    system = TNFRAdaptiveSystem(graph, "node")
    with pytest.raises(ValueError, match="DeltaNFR must be a finite real"):
        system._measure_stress()