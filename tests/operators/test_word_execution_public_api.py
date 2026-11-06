"""Public ownership and compatibility checks for network-word execution."""

from __future__ import annotations

import inspect

import networkx as nx
import pytest

import tnfr.operators as operators
import tnfr.physics.signatures as signatures
import tnfr.sdk.simple as simple_sdk
from tnfr.errors import TNFRValueError
from tnfr.operators.word_execution import (
    preflight_network_mutation_sequence,
    run_network_sequence,
)


def test_neutral_word_executor_is_public_and_sdk_aliases_remain_stable() -> None:
    assert operators.run_network_sequence is run_network_sequence
    assert (
        operators.preflight_network_mutation_sequence
        is preflight_network_mutation_sequence
    )
    assert simple_sdk._run_network_sequence is run_network_sequence
    assert (
        simple_sdk._preflight_sdk_mutation_sequence
        is preflight_network_mutation_sequence
    )


def test_physics_signatures_has_no_sdk_dependency() -> None:
    source = inspect.getsource(signatures)
    assert "from ..sdk" not in source
    assert "from tnfr.sdk" not in source


@pytest.mark.parametrize(
    "runner", [run_network_sequence, preflight_network_mutation_sequence]
)
@pytest.mark.parametrize("cycles", [True, -1, 1.0, "1"])
def test_public_word_executors_reject_invalid_cycle_counts(runner, cycles) -> None:
    with pytest.raises(TNFRValueError, match="nonnegative integer"):
        runner(nx.Graph(), [], cycles=cycles)


@pytest.mark.parametrize(
    "runner", [run_network_sequence, preflight_network_mutation_sequence]
)
def test_public_word_executors_keep_zero_cycles_as_noop(runner) -> None:
    graph = nx.Graph()
    graph.graph["sentinel"] = object()
    before = dict(graph.graph)

    assert runner(graph, [], cycles=0) is None
    assert graph.graph == before


def test_zero_cycles_does_not_apply_a_valid_nonempty_word() -> None:
    graph = nx.Graph()
    graph.add_node(0, EPI=0.25, nu_f=1.0, theta=0.0, delta_nfr=0.1)
    before = dict(graph.nodes[0])

    run_network_sequence(
        graph,
        ["emission", "reception", "coherence", "expansion", "resonance", "silence"],
        cycles=0,
    )

    assert dict(graph.nodes[0]) == before
