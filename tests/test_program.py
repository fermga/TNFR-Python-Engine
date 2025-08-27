import networkx as nx

from tnfr.program import play, seq, block, wait


def _step_noop(G):
    G.graph["_t"] = G.graph.get("_t", 0.0) + 1.0


def test_play_records_program_trace_with_block_and_wait():
    G = nx.Graph()
    G.add_node(1)
    program = seq("A’L", wait(2), block("O’Z"))
    play(G, program, step_fn=_step_noop)
    trace = G.graph["history"]["program_trace"]
    assert [e["op"] for e in trace] == ["GLYPH", "WAIT", "GLYPH", "GLYPH"]
    assert trace[2]["g"] == "T’HOL"
