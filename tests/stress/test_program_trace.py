"""Stress test verifying program trace rotation and THOL handling under nested workloads."""

from __future__ import annotations

import random
from collections import deque

import pytest

from tnfr.execution import HANDLERS, block, compile_sequence, play, seq, target, wait
from tnfr.tokens import OpTag
from tnfr.types import Glyph, TNFRGraph

pytestmark = [pytest.mark.stress]


def _build_medium_graph(graph_canon, *, seed: int, node_count: int) -> TNFRGraph:
    """Return a graph populated with deterministic edges below the stress threshold."""

    rng = random.Random(seed)
    G = graph_canon()
    G.add_nodes_from(range(node_count))

    for left in range(node_count):
        for right in range(left + 1, node_count):
            if rng.random() < 0.045:
                G.add_edge(left, right)

    G.graph["RANDOM_SEED"] = seed
    return G


def test_program_trace_rotates_without_dropping_thol_history(graph_canon):
    """Program trace must rotate safely while preserving glyph order and targets.

    The scenario exercises a deeply nested mix of ``seq``/``block``/``wait`` tokens
    on a medium graph (<<100 nodes) and forces the trace deque to rotate by using a
    tight ``PROGRAM_TRACE_MAXLEN``. It validates that:

    * ``_t`` advances deterministically via the custom ``step_fn``.
    * Trace entries never exceed ``maxlen`` while blocks emit the expected THOL
      glyph order, even under nested repetition.
    * ``HANDLERS`` preserve the active target nodes across long waits, ensuring
      no glyph history is dropped for previously selected nodes (TNFR invariant).
    """

    node_count = 84
    seed = 93421
    maxlen = 36
    dt = 0.125

    G = _build_medium_graph(graph_canon, seed=seed, node_count=node_count)
    G.graph["PROGRAM_TRACE_MAXLEN"] = maxlen
    G.graph["HISTORY_MAXLEN"] = 128
    G.graph["GLYPH_HYSTERESIS_WINDOW"] = 64

    groups = [
        tuple(range(0, 28)),
        tuple(range(28, 56)),
        tuple(range(56, node_count)),
    ]
    after_wait_glyphs = [Glyph.REMESH, Glyph.VAL, Glyph.NAV]

    nested_block = block(
        wait(2),
        Glyph.AL,
        block(Glyph.RA, wait(1), Glyph.ZHIR, close=Glyph.NUL),
        Glyph.OZ,
        wait(1),
        Glyph.SHA,
        repeat=5,
        close=Glyph.IL,
    )

    secondary_block = block(
        Glyph.RA,
        wait(2),
        block(Glyph.SHA, wait(1), Glyph.ZHIR, close=Glyph.NUL),
        Glyph.OZ,
        repeat=4,
        close=Glyph.UM,
    )

    final_block = block(
        wait(1),
        Glyph.SHA,
        block(Glyph.UM, wait(2), Glyph.RA, close=Glyph.NUL),
        Glyph.OZ,
        repeat=4,
        close=Glyph.EN,
    )

    program_tokens = []
    long_wait_steps = 24
    for idx, nodes in enumerate(groups):
        program_tokens.append(target(nodes))
        program_tokens.append(wait(11 + idx))
        program_tokens.append(nested_block)
        program_tokens.append(secondary_block)
        program_tokens.append(wait(long_wait_steps))
        program_tokens.append(after_wait_glyphs[idx])

    program_tokens.append(target())
    program_tokens.append(final_block)

    program = seq(*program_tokens)

    flattened = compile_sequence(program)
    assert 120 <= len(flattened) <= 200

    final_ops = compile_sequence([final_block])
    assert len(final_ops) < maxlen

    tick_times: list[float] = []
    glyph_watch = {glyph.value for glyph in after_wait_glyphs}
    captured_targets: dict[str, list[tuple[int, ...]]] = {}

    original_glyph_handler = HANDLERS[OpTag.GLYPH]

    def glyph_spy(G, payload, curr_target, trace, step_fn):
        glyph_value = payload.value if isinstance(payload, Glyph) else str(payload)
        if glyph_value in glyph_watch:
            captured_targets.setdefault(glyph_value, []).append(
                tuple(curr_target or ())
            )
        return original_glyph_handler(G, payload, curr_target, trace, step_fn)

    def step_monitor(graph):
        new_t = graph.graph.get("_t", 0.0) + dt
        graph.graph["_t"] = new_t
        tick_times.append(new_t)
        trace = graph.graph.get("history", {}).get("program_trace")
        if isinstance(trace, deque):
            assert len(trace) <= maxlen

    HANDLERS[OpTag.GLYPH] = glyph_spy
    try:
        play(G, program, step_fn=step_monitor)
    finally:
        HANDLERS[OpTag.GLYPH] = original_glyph_handler

    history = G.graph["history"]
    trace = history["program_trace"]
    assert isinstance(trace, deque)
    assert trace.maxlen == maxlen
    assert len(trace) <= maxlen

    expected_times = [dt * (idx + 1) for idx in range(len(tick_times))]
    assert tick_times == pytest.approx(expected_times)

    tail = list(trace)[-len(final_ops) :]
    observed_ops = [entry["op"] for entry in tail]
    expected_ops = [op.name for op, _ in final_ops]
    assert observed_ops == expected_ops

    observed_glyphs = [entry.get("g") for entry in tail if entry["op"] in {"THOL", "GLYPH"}]
    expected_glyphs = [payload for op, payload in final_ops if op.name in {"THOL", "GLYPH"}]
    assert observed_glyphs == expected_glyphs

    for glyph, nodes in zip(after_wait_glyphs, groups):
        glyph_value = glyph.value
        recorded = captured_targets.get(glyph_value)
        assert recorded, f"Glyph {glyph_value} did not record targets"
        expected_set = set(nodes)
        for snapshot in recorded:
            assert set(snapshot) == expected_set
