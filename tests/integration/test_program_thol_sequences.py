"""Integration tests covering THOL sequences with explicit closure semantics."""

from __future__ import annotations

from tnfr.execution import block, play, seq, target
from tnfr.flatten import parse_program_tokens
from tnfr.tokens import Glyph, OpTag


def _step_noop(graph):
    """Advance graph time deterministically for test reproducibility."""

    graph.graph["_t"] = graph.graph.get("_t", 0.0) + 1.0


def _patch_history_appender(monkeypatch):
    """Replace glyph application with a pure history recorder."""

    def _record_glyphs(G, glyph, nodes=None):
        glyph_name = glyph.value if isinstance(glyph, Glyph) else str(glyph)
        if nodes is None:
            targets = list(G.nodes())
        elif isinstance(nodes, (str, bytes)):
            targets = [nodes]
        else:
            targets = list(nodes)
        for node in targets:
            history = G.nodes[node].setdefault("glyph_history", [])
            history.append(glyph_name)

    monkeypatch.setattr("tnfr.execution._apply_glyph_to_targets", _record_glyphs)


def test_nested_blocks_force_close_history_and_sentinels(graph_canon, monkeypatch):
    """Nested ``block`` calls with mixed targets emit canonical closures."""

    _patch_history_appender(monkeypatch)

    G = graph_canon()
    nodes = ["root", "leaf_a", "leaf_b"]
    G.add_nodes_from(nodes)
    for node in nodes:
        G.nodes[node]["glyph_history"] = []

    inner = block(
        target(["leaf_a"]),
        Glyph.RA,
        repeat=2,
        close=Glyph.SHA,
    )
    outer = block(
        target(["root", "leaf_a"]),
        Glyph.AL,
        inner,
        target(["root", "leaf_b"]),
        Glyph.ZHIR,
        repeat=2,
        close=Glyph.NUL,
    )

    program = seq(target(nodes), outer)
    play(G, program, step_fn=_step_noop)

    root_history = G.nodes["root"]["glyph_history"]
    leaf_a_history = G.nodes["leaf_a"]["glyph_history"]
    leaf_b_history = G.nodes["leaf_b"]["glyph_history"]

    assert root_history == [
        Glyph.THOL.value,
        Glyph.AL.value,
        Glyph.THOL.value,
        Glyph.ZHIR.value,
        Glyph.NUL.value,
        Glyph.AL.value,
        Glyph.THOL.value,
        Glyph.ZHIR.value,
        Glyph.NUL.value,
    ]
    assert leaf_a_history == [
        Glyph.THOL.value,
        Glyph.AL.value,
        Glyph.THOL.value,
        Glyph.RA.value,
        Glyph.SHA.value,
        Glyph.RA.value,
        Glyph.SHA.value,
        Glyph.AL.value,
        Glyph.THOL.value,
        Glyph.RA.value,
        Glyph.SHA.value,
        Glyph.RA.value,
        Glyph.SHA.value,
    ]
    assert leaf_b_history == [
        Glyph.THOL.value,
        Glyph.ZHIR.value,
        Glyph.NUL.value,
        Glyph.ZHIR.value,
        Glyph.NUL.value,
    ]

    assert Glyph.SHA.value not in root_history
    assert Glyph.NUL.value not in leaf_a_history

    trace = list(G.graph["history"]["program_trace"])
    thol_entries = [entry for entry in trace if entry["op"] == OpTag.THOL.name]
    assert len(thol_entries) == 3
    assert all(entry["g"] == Glyph.THOL.value for entry in thol_entries)


def test_string_force_close_coercion(graph_canon, monkeypatch):
    """String ``force_close`` inputs are coerced to canonical glyph names."""

    _patch_history_appender(monkeypatch)

    G = graph_canon()
    node_id = "solo"
    G.add_node(node_id, glyph_history=[])

    string_close_program = parse_program_tokens(
        [
            {
                "THOL": {
                    "body": [
                        {"TARGET": [node_id]},
                        Glyph.AL.value,
                    ],
                    "repeat": 2,
                    "close": "SHA",
                }
            }
        ]
    )

    play(G, string_close_program, step_fn=_step_noop)

    expected_history = [
        Glyph.THOL.value,
        Glyph.AL.value,
        Glyph.SHA.value,
        Glyph.AL.value,
        Glyph.SHA.value,
    ]
    assert G.nodes[node_id]["glyph_history"] == expected_history

    trace = list(G.graph["history"]["program_trace"])
    assert any(entry["op"] == OpTag.THOL.name for entry in trace)
