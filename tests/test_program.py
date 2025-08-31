import json
import pytest

from tnfr.cli import _load_sequence
from tnfr.program import play, seq, block, wait
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
    assert [e["op"] for e in trace] == ["GLYPH", "WAIT", "GLYPH", "GLYPH"]
    assert trace[2]["g"] == Glyph.THOL.value


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
