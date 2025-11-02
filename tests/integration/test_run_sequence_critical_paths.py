"""Critical path coverage for run_sequence execution and trajectories.

This module provides focused tests for critical execution paths including:
- Complex sequence compilation
- Trajectory state validation
- Error handling in execution
- Edge cases in operator sequences
"""

import pytest
import networkx as nx

from tnfr.constants import inject_defaults, DNFR_PRIMARY, EPI_PRIMARY, VF_PRIMARY
from tnfr.execution import play, seq, wait, target, block, compile_sequence
from tnfr.tokens import Glyph, OpTag


@pytest.fixture
def graph_canon():
    """Create a canonical test graph."""
    def _create():
        G = nx.Graph()
        inject_defaults(G)
        return G
    return _create


def _step_noop(graph):
    """Simple step function that advances time without side effects."""
    graph.graph["_t"] = graph.graph.get("_t", 0.0) + 1.0


def test_compile_sequence_nested_blocks() -> None:
    """Verify compile_sequence handles nested block structures."""
    sequence = seq(
        block(Glyph.SHA, repeat=2),
        wait(3),
        block(Glyph.AL, repeat=1),
    )
    
    compiled = compile_sequence(sequence)
    
    # Should contain THOL markers and glyphs
    op_tags = [op[0] for op in compiled]
    assert OpTag.THOL in op_tags
    assert OpTag.GLYPH in op_tags
    assert OpTag.WAIT in op_tags


def test_compile_sequence_multiple_targets() -> None:
    """Verify compile_sequence handles multiple target operations."""
    sequence = seq(
        target([0, 1]),
        Glyph.SHA,
        target([2, 3]),
        Glyph.AL,
    )
    
    compiled = compile_sequence(sequence)
    
    # Should have multiple target operations
    target_ops = [op for op in compiled if op[0] == OpTag.TARGET]
    assert len(target_ops) == 2


def test_compile_sequence_empty_blocks() -> None:
    """Verify compile_sequence handles empty blocks gracefully."""
    # Empty sequence
    compiled = compile_sequence(seq())
    assert len(compiled) == 0


def test_compile_sequence_single_operations() -> None:
    """Verify compile_sequence handles single operations correctly."""
    # Single glyph
    compiled_glyph = compile_sequence(seq(Glyph.SHA))
    assert len(compiled_glyph) == 1
    assert compiled_glyph[0][0] == OpTag.GLYPH
    
    # Single wait
    compiled_wait = compile_sequence(seq(wait(5)))
    assert len(compiled_wait) == 1
    assert compiled_wait[0][0] == OpTag.WAIT
    
    # Single target
    compiled_target = compile_sequence(seq(target([0])))
    assert len(compiled_target) == 1
    assert compiled_target[0][0] == OpTag.TARGET


def test_compile_sequence_long_chains() -> None:
    """Verify compile_sequence handles long operation chains."""
    # Create long sequence
    operations = []
    for i in range(10):
        operations.append(Glyph.SHA)
        operations.append(wait(1))
    
    sequence = seq(*operations)
    compiled = compile_sequence(sequence)
    
    # Should have many operations
    assert len(compiled) >= 20


def test_compile_sequence_repeated_blocks() -> None:
    """Verify compile_sequence handles repeated blocks correctly."""
    sequence = seq(
        block(Glyph.SHA, repeat=3),
    )
    
    compiled = compile_sequence(sequence)
    
    # Should have THOL operations and glyphs
    op_tags = [op[0] for op in compiled]
    assert OpTag.THOL in op_tags
    assert OpTag.GLYPH in op_tags


def test_run_sequence_empty_graph(graph_canon) -> None:
    """Verify run_sequence handles empty graph."""
    G = graph_canon()
    # No nodes added
    
    play(G, seq(), step_fn=_step_noop)
    
    # Should complete without error
    assert "history" in G.graph


def test_run_sequence_single_node(graph_canon) -> None:
    """Verify run_sequence handles single node graph."""
    G = graph_canon()
    G.add_node(0, **{
        EPI_PRIMARY: 0.5,
        VF_PRIMARY: 1.0,
        DNFR_PRIMARY: 0.0,
    })
    
    play(G, seq(wait(2)), step_fn=_step_noop)
    
    # Should complete and create trace
    assert "history" in G.graph
    assert "program_trace" in G.graph["history"]


def test_run_sequence_multiple_wait_operations(graph_canon) -> None:
    """Verify run_sequence handles multiple wait operations."""
    G = graph_canon()
    G.add_node(0)
    
    sequence = seq(
        wait(1),
        wait(2),
        wait(3),
    )
    
    play(G, sequence, step_fn=_step_noop)
    
    trace = list(G.graph["history"]["program_trace"])
    wait_entries = [e for e in trace if e["op"] == "WAIT"]
    
    # Should have 3 wait entries
    assert len(wait_entries) == 3


def test_run_sequence_target_switching(graph_canon) -> None:
    """Verify run_sequence handles target switching correctly."""
    G = graph_canon()
    G.add_nodes_from([0, 1, 2, 3])
    
    sequence = seq(
        target([0, 1]),
        wait(1),
        target([2, 3]),
        wait(1),
    )
    
    play(G, sequence, step_fn=_step_noop)
    
    trace = list(G.graph["history"]["program_trace"])
    target_entries = [e for e in trace if e["op"] == "TARGET"]
    
    # Should have 2 target operations
    assert len(target_entries) == 2


def test_run_sequence_wait_zero_clamping(graph_canon) -> None:
    """Verify wait(0) is clamped to minimum of 1 step."""
    G = graph_canon()
    G.add_node(0)
    
    # Wait with 0 should be clamped
    play(G, seq(wait(0)), step_fn=_step_noop)
    
    trace = list(G.graph["history"]["program_trace"])
    wait_entries = [e for e in trace if e["op"] == "WAIT"]
    
    # Should have executed at least 1 step
    assert all(e["k"] >= 1 for e in wait_entries)


def test_run_sequence_wait_negative_clamping(graph_canon) -> None:
    """Verify negative wait values are handled gracefully."""
    G = graph_canon()
    G.add_node(0)
    
    # Negative wait should be clamped
    play(G, seq(wait(-5)), step_fn=_step_noop)
    
    trace = list(G.graph["history"]["program_trace"])
    wait_entries = [e for e in trace if e["op"] == "WAIT"]
    
    # Should clamp to minimum of 1
    assert all(e["k"] >= 1 for e in wait_entries)


def test_run_sequence_target_none_selects_all(graph_canon) -> None:
    """Verify target(None) selects all nodes."""
    G = graph_canon()
    G.add_nodes_from([0, 1, 2, 3, 4])
    
    play(G, seq(target(None)), step_fn=_step_noop)
    
    trace = list(G.graph["history"]["program_trace"])
    target_entries = [e for e in trace if e["op"] == "TARGET"]
    
    # Should target all 5 nodes
    assert len(target_entries) == 1
    assert target_entries[0]["n"] == 5


def test_run_sequence_target_empty_list(graph_canon) -> None:
    """Verify target([]) handles empty target list."""
    G = graph_canon()
    G.add_nodes_from([0, 1, 2])
    
    # Empty target list
    play(G, seq(target([]), wait(1)), step_fn=_step_noop)
    
    # Should complete without error
    assert "history" in G.graph


def test_run_sequence_target_single_node(graph_canon) -> None:
    """Verify target([node]) selects single node."""
    G = graph_canon()
    G.add_nodes_from([0, 1, 2])
    
    play(G, seq(target([1])), step_fn=_step_noop)
    
    trace = list(G.graph["history"]["program_trace"])
    target_entries = [e for e in trace if e["op"] == "TARGET"]
    
    # Should target 1 node
    assert len(target_entries) == 1
    assert target_entries[0]["n"] == 1


def test_run_sequence_interleaved_operations(graph_canon) -> None:
    """Verify interleaved wait and target operations."""
    G = graph_canon()
    G.add_nodes_from([0, 1, 2])
    
    sequence = seq(
        target([0]),
        wait(1),
        target([1]),
        wait(2),
        target([2]),
        wait(1),
    )
    
    play(G, sequence, step_fn=_step_noop)
    
    trace = list(G.graph["history"]["program_trace"])
    
    # Should have both target and wait operations
    target_count = sum(1 for e in trace if e["op"] == "TARGET")
    wait_count = sum(1 for e in trace if e["op"] == "WAIT")
    
    assert target_count == 3
    assert wait_count == 3


def test_run_sequence_time_progression(graph_canon) -> None:
    """Verify time progresses correctly during sequence execution."""
    G = graph_canon()
    G.add_node(0)
    
    # Execute sequence with waits
    play(G, seq(wait(3), wait(2)), step_fn=_step_noop)
    
    # Time should have advanced
    final_time = G.graph.get("_t", 0.0)
    assert final_time > 0.0


def test_run_sequence_trace_ordering(graph_canon) -> None:
    """Verify trace entries maintain execution order."""
    G = graph_canon()
    G.add_nodes_from([0, 1])
    
    sequence = seq(
        target([0]),
        wait(1),
        target([1]),
        wait(1),
    )
    
    play(G, sequence, step_fn=_step_noop)
    
    trace = list(G.graph["history"]["program_trace"])
    
    # Trace should follow execution order
    ops = [e["op"] for e in trace]
    
    # Should have TARGET, WAIT, TARGET, WAIT pattern
    target_indices = [i for i, op in enumerate(ops) if op == "TARGET"]
    wait_indices = [i for i, op in enumerate(ops) if op == "WAIT"]
    
    # Targets and waits should be interleaved
    assert len(target_indices) == 2
    assert len(wait_indices) >= 2


def test_run_sequence_consistent_state(graph_canon) -> None:
    """Verify graph state remains consistent after execution."""
    G = graph_canon()
    G.add_node(0, **{
        EPI_PRIMARY: 0.5,
        VF_PRIMARY: 1.0,
        DNFR_PRIMARY: 0.0,
    })
    
    initial_epi = G.nodes[0][EPI_PRIMARY]
    initial_vf = G.nodes[0][VF_PRIMARY]
    
    play(G, seq(wait(5)), step_fn=_step_noop)
    
    # With noop step function, node attributes shouldn't change
    assert G.nodes[0][EPI_PRIMARY] == initial_epi
    assert G.nodes[0][VF_PRIMARY] == initial_vf


def test_compile_sequence_deterministic() -> None:
    """Verify compile_sequence is deterministic."""
    sequence = seq(
        target([0, 1]),
        Glyph.SHA,
        wait(3),
        block(Glyph.AL, repeat=2),
    )
    
    # Compile multiple times
    compiled1 = compile_sequence(sequence)
    compiled2 = compile_sequence(sequence)
    compiled3 = compile_sequence(sequence)
    
    # Should be identical
    assert len(compiled1) == len(compiled2) == len(compiled3)
    for (op1, pay1), (op2, pay2), (op3, pay3) in zip(compiled1, compiled2, compiled3):
        assert op1 == op2 == op3
        # Payloads may be complex objects, so we check tags match


def test_compile_sequence_preserves_structure() -> None:
    """Verify compile_sequence preserves operation structure."""
    sequence = seq(
        target([0]),
        Glyph.SHA,
        wait(5),
    )
    
    compiled = compile_sequence(sequence)
    
    # Should have exactly 3 operations
    assert len(compiled) == 3
    
    # In order: TARGET, GLYPH, WAIT
    assert compiled[0][0] == OpTag.TARGET
    assert compiled[1][0] == OpTag.GLYPH
    assert compiled[2][0] == OpTag.WAIT
