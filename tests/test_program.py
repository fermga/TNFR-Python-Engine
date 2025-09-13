"""Pruebas de program."""

import json
from collections import deque
from collections.abc import Sequence

import pytest

from tnfr.cli import _load_sequence
from tnfr.program import (
    WAIT,
    TARGET,
    OpTag,
    _handle_target,
    _flatten,
    THOLEvaluator,
    block,
    play,
    seq,
    target,
    wait,
    THOL,
    THOL_SENTINEL,
)
from tnfr.constants import get_param
from tnfr.types import Glyph

yaml = pytest.importorskip("yaml")


def _step_noop(G):
    G.graph["_t"] = G.graph.get("_t", 0.0) + 1.0


def test_play_records_program_trace_with_block_and_wait(graph_canon):
    G = graph_canon()
    G.add_node(1)
    program = seq(Glyph.AL, wait(2), block(Glyph.OZ))
    play(G, program, step_fn=_step_noop)
    trace = G.graph["history"]["program_trace"]
    assert [e["op"] for e in trace] == ["GLYPH", "WAIT", "THOL", "GLYPH"]
    assert trace[2]["g"] == Glyph.THOL.value


def test_wait_logs_sanitized_steps(graph_canon):
    G = graph_canon()
    G.add_node(1)
    play(G, seq(WAIT(0)), step_fn=_step_noop)
    trace = G.graph["history"]["program_trace"]
    assert [e["op"] for e in trace] == ["WAIT"]
    assert trace[0]["k"] == 1


def test_flatten_wait_sanitizes_steps():
    program = seq(WAIT(-2.5), WAIT(2.4))
    ops = _flatten(program)
    assert ops == [(OpTag.WAIT, 1), (OpTag.WAIT, 2)]


def test_flatten_accepts_wait_subclass():
    class CustomWait(WAIT):
        pass

    program = seq(CustomWait(3))
    ops = _flatten(program)
    assert ops == [(OpTag.WAIT, 3)]


def test_play_handles_deeply_nested_blocks(graph_canon):
    G = graph_canon()
    G.add_node(1)

    depth = 1500
    inner = Glyph.AL
    for _ in range(depth):
        inner = block(inner)

    play(G, seq(inner), step_fn=_step_noop)
    trace = G.graph["history"]["program_trace"]

    maxlen = int(get_param(G, "PROGRAM_TRACE_MAXLEN"))
    assert len(trace) == maxlen
    assert trace[0]["g"] == Glyph.THOL.value
    assert trace[-1]["g"] == Glyph.AL.value


def test_target_persists_across_wait(graph_canon):
    G = graph_canon()
    G.add_nodes_from([1, 2])

    def step_add_node(G):
        G.graph["_t"] = G.graph.get("_t", 0.0) + 1.0
        if not G.graph.get("added"):
            G.add_node(3)
            G.graph["added"] = True

    play(G, seq(target(), wait(1), Glyph.AL), step_fn=step_add_node)

    assert list(G.nodes[1]["glyph_history"]) == [Glyph.AL.value]
    assert list(G.nodes[2]["glyph_history"]) == [Glyph.AL.value]
    assert "glyph_history" not in G.nodes[3]


def test_target_accepts_string(graph_canon):
    G = graph_canon()
    # Add nodes that would be mistakenly targeted if the string were iterated
    G.add_nodes_from(["node1", "n", "o", "d", "e", "1"])
    play(G, seq(target("node1"), Glyph.AL), step_fn=_step_noop)
    assert list(G.nodes["node1"]["glyph_history"]) == [Glyph.AL.value]
    for c in "node1":
        assert "glyph_history" not in G.nodes[c]


def test_target_accepts_bytes(graph_canon):
    G = graph_canon()
    bname = b"node1"
    codes = list(bname)
    G.add_nodes_from([bname, *codes])
    play(G, seq(target(bname), Glyph.AL), step_fn=_step_noop)
    assert list(G.nodes[bname]["glyph_history"]) == [Glyph.AL.value]
    for code in codes:
        assert "glyph_history" not in G.nodes[code]


def test_handle_target_reuses_sequence(graph_canon):
    G = graph_canon()
    G.add_nodes_from([1, 2])
    nodes = [1]
    trace = deque()
    curr = _handle_target(G, TARGET(nodes), None, trace, None)
    assert curr is nodes


def test_handle_target_materializes_non_sequence(graph_canon):
    G = graph_canon()
    G.add_nodes_from([1, 2])
    trace = deque()
    nodes_view = G.nodes()
    curr = _handle_target(G, TARGET(nodes_view), None, trace, None)
    assert isinstance(curr, tuple)


def test_load_sequence_json_yaml(tmp_path):
    data = [
        "AL",
        {"THOL": {"body": [["OZ", "EN"], "RA"], "repeat": 1}},
        {"WAIT": 1},
    ]

    jpath = tmp_path / "prog.json"
    jpath.write_text(json.dumps(data))

    ypath = tmp_path / "prog.yaml"
    ypath.write_text(yaml.safe_dump(data))

    expected = seq("AL", block("OZ", "EN", "RA"), wait(1))
    assert _load_sequence(jpath) == expected
    assert _load_sequence(ypath) == expected


def test_load_sequence_repeated_calls(tmp_path):
    data = [
        "AL",
        {"THOL": {"body": [["OZ", "EN"], "RA"], "repeat": 1}},
        {"WAIT": 1},
    ]

    path = tmp_path / "prog.json"
    path.write_text(json.dumps(data))

    expected = seq("AL", block("OZ", "EN", "RA"), wait(1))
    for _ in range(5):
        assert _load_sequence(path) == expected


@pytest.mark.parametrize("bad", ["SHA", 123])
def test_block_force_close_invalid_type_raises(graph_canon, bad):
    G = graph_canon()
    G.add_node(1)
    program = seq(block(Glyph.AL, close=bad))
    with pytest.raises(ValueError):
        play(G, program, step_fn=_step_noop)


def test_flatten_nested_blocks_preserves_order():
    program = seq(
        block(
            block(Glyph.AL, Glyph.RA, repeat=2, close=Glyph.NUL),
            Glyph.ZHIR,
        )
    )
    ops = _flatten(program)
    expected = [
        (OpTag.THOL, Glyph.THOL.value),
        (OpTag.THOL, Glyph.THOL.value),
        (OpTag.GLYPH, Glyph.AL.value),
        (OpTag.GLYPH, Glyph.RA.value),
        (OpTag.GLYPH, Glyph.NUL.value),
        (OpTag.GLYPH, Glyph.AL.value),
        (OpTag.GLYPH, Glyph.RA.value),
        (OpTag.GLYPH, Glyph.NUL.value),
        (OpTag.GLYPH, Glyph.ZHIR.value),
    ]
    assert ops == expected


class NoReverseSeq(Sequence):
    def __init__(self, data):
        self._data = data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]


def test_flatten_accepts_sequence_without_reversed():
    program = NoReverseSeq([Glyph.AL, Glyph.OZ])
    ops = _flatten(program)
    assert ops == [(OpTag.GLYPH, Glyph.AL.value), (OpTag.GLYPH, Glyph.OZ.value)]


def test_thol_repeat_lt_one_raises():
    with pytest.raises(ValueError, match="repeat must be ≥1"):
        list(THOLEvaluator(THOL(body=[], repeat=0)))


def test_flatten_thol_multiple_repeats():
    tokens = list(THOLEvaluator(THOL(body=[Glyph.AL, Glyph.RA], repeat=3)))
    assert tokens == [
        THOL_SENTINEL,
        Glyph.AL,
        Glyph.RA,
        Glyph.AL,
        Glyph.RA,
        Glyph.AL,
        Glyph.RA,
    ]


def test_flatten_thol_body_limit_error_message():
    body = (Glyph.AL for _ in range(5))
    with pytest.raises(
        ValueError, match="THOL body exceeds max_materialize=3"
    ):
        list(THOLEvaluator(THOL(body=body), max_materialize=3))
