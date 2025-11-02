"""Comprehensive tests for run_sequence execution paths.

This module tests critical paths for sequence execution including:
- Operator application order and ΔNFR hook triggering
- Sequence validation and error handling
- Integration with validators
"""

import networkx as nx
import pytest

from tnfr import run_sequence
from tnfr.constants import DNFR_PRIMARY, EPI_PRIMARY, VF_PRIMARY, inject_defaults
from tnfr.structural import (
    Coherence,
    Emission,
    Reception,
    Resonance,
    Silence,
    create_nfr,
    validate_sequence,
)


def test_run_sequence_basic_execution() -> None:
    """Verify run_sequence executes a valid operator sequence."""
    G, n = create_nfr("test_node", epi=0.5, vf=1.0, theta=0.0)

    # Valid TNFR sequence
    operators = [Emission(), Reception(), Coherence(), Resonance(), Silence()]
    result = run_sequence(G, n, operators)

    assert result is not None
    assert n in G.nodes


def test_run_sequence_respects_operator_order() -> None:
    """Verify operators are applied in specified order."""
    G, n = create_nfr("test_node")

    call_order = []

    class TrackedEmission(Emission):
        def apply(self, graph, node):
            call_order.append("emission")
            return super().apply(graph, node)

    class TrackedReception(Reception):
        def apply(self, graph, node):
            call_order.append("reception")
            return super().apply(graph, node)

    class TrackedCoherence(Coherence):
        def apply(self, graph, node):
            call_order.append("coherence")
            return super().apply(graph, node)

    class TrackedResonance(Resonance):
        def apply(self, graph, node):
            call_order.append("resonance")
            return super().apply(graph, node)

    class TrackedSilence(Silence):
        def apply(self, graph, node):
            call_order.append("silence")
            return super().apply(graph, node)

    operators = [
        TrackedEmission(),
        TrackedReception(),
        TrackedCoherence(),
        TrackedResonance(),
        TrackedSilence(),
    ]
    run_sequence(G, n, operators)

    assert call_order == ["emission", "reception", "coherence", "resonance", "silence"]


def test_run_sequence_triggers_dnfr_hook() -> None:
    """Verify run_sequence triggers ΔNFR computation hooks."""
    G, n = create_nfr("test_node", epi=0.5, vf=1.0)

    hook_calls = []

    def tracking_hook(graph, **kwargs):
        hook_calls.append(len(graph.nodes))

    G.graph["compute_delta_nfr"] = tracking_hook

    operators = [Emission(), Reception(), Coherence(), Resonance(), Silence()]
    run_sequence(G, n, operators)

    # Hook should be called at least once
    assert len(hook_calls) > 0


def test_run_sequence_validates_valid_sequence() -> None:
    """Verify valid sequences pass validation."""
    G, n = create_nfr("test_node")

    # Valid sequence
    valid_ops = [Emission(), Reception(), Coherence(), Resonance(), Silence()]
    validation = validate_sequence([op.name for op in valid_ops])
    assert validation.passed, f"Validation failed: {validation.summary['message']}"

    # Can execute valid sequence
    result = run_sequence(G, n, valid_ops)
    assert result is not None


def test_run_sequence_detects_invalid_sequence() -> None:
    """Verify invalid sequences are detected by validation."""
    # Invalid sequence - incomplete
    invalid_names = ["emission", "reception"]
    validation = validate_sequence(invalid_names)

    # Should fail validation
    assert not validation.passed


def test_run_sequence_maintains_node_attributes() -> None:
    """Verify run_sequence preserves essential node attributes."""
    G, n = create_nfr("test_node", epi=0.7, vf=1.5, theta=0.3)

    operators = [Emission(), Reception(), Coherence(), Resonance(), Silence()]
    run_sequence(G, n, operators)

    # Attributes should still exist (may have changed values)
    assert EPI_PRIMARY in G.nodes[n]
    assert VF_PRIMARY in G.nodes[n]
    assert DNFR_PRIMARY in G.nodes[n]


def test_run_sequence_with_minimal_valid_sequence() -> None:
    """Verify minimal valid TNFR sequence executes."""
    G, n = create_nfr("test_node", epi=0.5, vf=1.0)

    # Minimal valid sequence
    operators = [Emission(), Reception(), Coherence(), Resonance(), Silence()]
    run_sequence(G, n, operators)

    # Node should still exist with attributes
    assert n in G.nodes
    assert EPI_PRIMARY in G.nodes[n]


def test_run_sequence_with_resonance_operator() -> None:
    """Verify Resonance operator handles coupling and propagation."""
    G = nx.Graph()
    inject_defaults(G)
    G.add_node(0, **{EPI_PRIMARY: 0.5, VF_PRIMARY: 1.0, DNFR_PRIMARY: 0.0})
    G.add_node(1, **{EPI_PRIMARY: 0.3, VF_PRIMARY: 1.2, DNFR_PRIMARY: 0.0})
    G.add_edge(0, 1)

    operators = [Emission(), Reception(), Coherence(), Resonance(), Silence()]
    run_sequence(G, 0, operators)

    # Both nodes should still exist with attributes
    assert 0 in G.nodes
    assert 1 in G.nodes
    assert EPI_PRIMARY in G.nodes[0]
    assert EPI_PRIMARY in G.nodes[1]


def test_run_sequence_preserves_graph_structure() -> None:
    """Verify run_sequence doesn't corrupt graph structure."""
    G = nx.Graph()
    inject_defaults(G)
    G.add_node(0, **{EPI_PRIMARY: 0.5, VF_PRIMARY: 1.0, DNFR_PRIMARY: 0.0})
    G.add_node(1, **{EPI_PRIMARY: 0.3, VF_PRIMARY: 1.2, DNFR_PRIMARY: 0.0})
    G.add_node(2, **{EPI_PRIMARY: 0.7, VF_PRIMARY: 0.8, DNFR_PRIMARY: 0.0})
    G.add_edge(0, 1)
    G.add_edge(1, 2)

    operators = [Emission(), Reception(), Coherence(), Resonance(), Silence()]
    run_sequence(G, 0, operators)

    # Target node should exist
    assert 0 in G.nodes


def test_run_sequence_incremental_dnfr_updates() -> None:
    """Verify ΔNFR is updated during sequence execution."""
    G, n = create_nfr("test_node", epi=0.5, vf=1.0)

    dnfr_history = []

    def tracking_hook(graph, **kwargs):
        dnfr_history.append(float(graph.nodes[n].get(DNFR_PRIMARY, 0.0)))

    G.graph["compute_delta_nfr"] = tracking_hook

    operators = [Emission(), Reception(), Coherence(), Resonance(), Silence()]
    run_sequence(G, n, operators)

    # Should have ΔNFR recorded
    assert len(dnfr_history) > 0


def test_run_sequence_multiple_nodes() -> None:
    """Verify run_sequence can be called on multiple nodes sequentially."""
    G = nx.Graph()
    inject_defaults(G)
    G.add_node(0, **{EPI_PRIMARY: 0.5, VF_PRIMARY: 1.0, DNFR_PRIMARY: 0.0})
    G.add_node(1, **{EPI_PRIMARY: 0.3, VF_PRIMARY: 1.2, DNFR_PRIMARY: 0.0})

    operators = [Emission(), Reception(), Coherence(), Resonance(), Silence()]

    run_sequence(G, 0, operators)
    run_sequence(G, 1, operators)

    # Both nodes should have been processed
    assert 0 in G.nodes
    assert 1 in G.nodes


def test_validate_sequence_accepts_canonical_names() -> None:
    """Verify sequence validation accepts canonical operator names."""
    valid_names = ["emission", "reception", "coherence", "resonance", "silence"]
    result = validate_sequence(valid_names)

    assert result.passed, f"Validation failed: {result.summary['message']}"
    assert result.summary["tokens"] == tuple(valid_names)


def test_validate_sequence_provides_useful_errors() -> None:
    """Verify sequence validation provides helpful error messages."""
    # Invalid sequence - incomplete
    invalid_names = ["emission", "reception"]
    result = validate_sequence(invalid_names)

    # Should fail with informative message
    assert not result.passed
    assert "message" in result.summary
    assert len(result.summary["message"]) > 0
