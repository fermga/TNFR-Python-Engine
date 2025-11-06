"""Shared utilities for run_sequence testing.

This module provides reusable fixtures and helpers to avoid duplication
across run_sequence test modules following DRY principles.
"""

from __future__ import annotations

from typing import Callable

import networkx as nx
import pytest

from tnfr.constants import DNFR_PRIMARY, EPI_PRIMARY, VF_PRIMARY, inject_defaults


@pytest.fixture
def graph_factory() -> Callable[[], nx.Graph]:
    """Factory for creating canonical test graphs with TNFR defaults.

    Returns
    -------
    Callable[[], nx.Graph]
        Function that creates a new graph with TNFR defaults injected.

    Examples
    --------
    >>> def test_something(graph_factory):
    ...     G = graph_factory()
    ...     G.add_nodes_from([0, 1, 2])
    ...     # Test with G
    """

    def _create() -> nx.Graph:
        """Create a graph with TNFR structural defaults."""
        G = nx.Graph()
        inject_defaults(G)
        return G

    return _create


@pytest.fixture
def step_noop() -> Callable[[nx.Graph], None]:
    """Simple no-op step function that advances time.

    Returns
    -------
    Callable[[nx.Graph], None]
        Step function that increments time without side effects.

    Examples
    --------
    >>> def test_sequence(graph_factory, step_noop):
    ...     G = graph_factory()
    ...     G.add_node(0)
    ...     play(G, seq(wait(1)), step_fn=step_noop)
    """

    def _step(graph: nx.Graph) -> None:
        """Advance time by 1.0 without modifying node state."""
        graph.graph["_t"] = graph.graph.get("_t", 0.0) + 1.0

    return _step


def assert_trace_has_operations(
    graph: nx.Graph,
    expected_ops: list[str],
) -> None:
    """Assert program trace contains expected operation types.

    Parameters
    ----------
    graph : nx.Graph
        Graph with executed sequence and program trace.
    expected_ops : list[str]
        List of expected operation names (e.g., ["WAIT", "TARGET"]).

    Raises
    ------
    AssertionError
        If trace is missing or doesn't contain expected operations.

    Examples
    --------
    >>> play(G, seq(wait(1), target([0])), step_fn=step_noop)
    >>> assert_trace_has_operations(G, ["WAIT", "TARGET"])
    """
    assert "history" in graph.graph, "Graph missing history"
    assert "program_trace" in graph.graph["history"], "History missing program_trace"

    trace = list(graph.graph["history"]["program_trace"])
    trace_ops = {e["op"] for e in trace}

    for op in expected_ops:
        assert op in trace_ops, f"Operation {op} not found in trace"


def assert_trace_length(
    graph: nx.Graph,
    expected_length: int | None = None,
    min_length: int | None = None,
) -> None:
    """Assert program trace has expected length or minimum length.

    Parameters
    ----------
    graph : nx.Graph
        Graph with executed sequence and program trace.
    expected_length : int, optional
        Exact expected trace length.
    min_length : int, optional
        Minimum expected trace length.

    Raises
    ------
    AssertionError
        If trace length doesn't match expectations.

    Examples
    --------
    >>> play(G, seq(wait(3)), step_fn=step_noop)
    >>> assert_trace_length(G, min_length=1)
    """
    assert "history" in graph.graph
    trace = list(graph.graph["history"]["program_trace"])

    if expected_length is not None:
        assert (
            len(trace) == expected_length
        ), f"Expected trace length {expected_length}, got {len(trace)}"

    if min_length is not None:
        assert (
            len(trace) >= min_length
        ), f"Expected minimum trace length {min_length}, got {len(trace)}"


def assert_time_progression(graph: nx.Graph) -> None:
    """Assert program trace shows monotonic time progression.

    Parameters
    ----------
    graph : nx.Graph
        Graph with executed sequence and program trace.

    Raises
    ------
    AssertionError
        If time doesn't progress monotonically in trace.

    Examples
    --------
    >>> play(G, seq(wait(1), wait(2)), step_fn=step_noop)
    >>> assert_time_progression(G)
    """
    assert "history" in graph.graph
    trace = list(graph.graph["history"]["program_trace"])

    times = [e.get("t", 0.0) for e in trace]

    for i in range(len(times) - 1):
        assert (
            times[i] <= times[i + 1]
        ), f"Time not monotonic: t[{i}]={times[i]} > t[{i+1}]={times[i+1]}"


def count_trace_operations(
    graph: nx.Graph,
    operation: str,
) -> int:
    """Count occurrences of specific operation in program trace.

    Parameters
    ----------
    graph : nx.Graph
        Graph with executed sequence and program trace.
    operation : str
        Operation name to count (e.g., "WAIT", "TARGET", "GLYPH").

    Returns
    -------
    int
        Number of times operation appears in trace.

    Examples
    --------
    >>> play(G, seq(wait(1), wait(2), wait(3)), step_fn=step_noop)
    >>> count = count_trace_operations(G, "WAIT")
    >>> assert count == 3
    """
    assert "history" in graph.graph
    trace = list(graph.graph["history"]["program_trace"])

    return sum(1 for e in trace if e["op"] == operation)


def create_test_graph_with_nodes(
    num_nodes: int,
    *,
    epi_value: float = 0.5,
    vf_value: float = 1.0,
) -> nx.Graph:
    """Create a test graph with specified number of initialized nodes.

    Parameters
    ----------
    num_nodes : int
        Number of nodes to add to graph.
    epi_value : float, default=0.5
        EPI value for all nodes.
    vf_value : float, default=1.0
        νf value for all nodes.

    Returns
    -------
    nx.Graph
        Graph with nodes and TNFR attributes initialized.

    Examples
    --------
    >>> G = create_test_graph_with_nodes(5, epi_value=0.0, vf_value=1.0)
    >>> assert len(G.nodes) == 5
    """
    G = nx.Graph()
    inject_defaults(G)

    for i in range(num_nodes):
        G.add_node(
            i,
            **{
                EPI_PRIMARY: epi_value,
                VF_PRIMARY: vf_value,
                DNFR_PRIMARY: 0.0,
            },
        )

    return G
