"""Integration tests exercising program sequencing, flattening, and execution traces."""

import json
from collections import deque
from collections.abc import Collection, Sequence
from pathlib import Path

import pytest

import tnfr.flatten as flatten_module
from tnfr.cli.execution import _load_sequence
from tnfr.constants import get_param
from tnfr.execution import HANDLERS, block, compile_sequence, play, seq, target, wait
from tnfr.tokens import TARGET, THOL, WAIT, OpTag
from tnfr.types import Glyph
from tnfr.io import StructuredFileError

yaml = pytest.importorskip("yaml")


def _step_noop(G):
    G.graph["_t"] = G.graph.get("_t", 0.0) + 1.0


def test_play_unknown_operation_raises(monkeypatch, graph_canon):
    monkeypatch.delitem(HANDLERS, OpTag.GLYPH, raising=False)

    G = graph_canon()
    G.add_node(1)

    with pytest.raises(ValueError, match="Unknown operation"):
        play(G, seq(Glyph.AL), step_fn=_step_noop)


def test_play_records_program_trace_with_block_and_wait(graph_canon):
    G = graph_canon()
    G.add_node(1)
    program = seq(Glyph.AL, wait(2), block(Glyph.OZ))
    play(G, program, step_fn=_step_noop)
    trace = G.graph["history"]["program_trace"]
    assert [e["op"] for e in trace] == ["GLYPH", "WAIT", "THOL", "GLYPH"]
    assert trace[2]["g"] == Glyph.THOL.value


def test_play_records_progressive_time_with_custom_step(graph_canon):
    G = graph_canon()
    G.add_node(1)

    delta = 0.5
    invocation_times: list[float] = []

    def step_increment(graph):
        new_t = graph.graph.get("_t", 0.0) + delta
        graph.graph["_t"] = new_t
        invocation_times.append(new_t)

    program = seq(wait(3), Glyph.AL)
    play(G, program, step_fn=step_increment)

    trace = list(G.graph["history"]["program_trace"])
    assert [entry["op"] for entry in trace] == ["WAIT", "GLYPH"]
    assert [entry["t"] for entry in trace] == pytest.approx([delta * 3, delta * 4])

    assert len(invocation_times) == 4
    assert invocation_times == pytest.approx([delta, delta * 2, delta * 3, delta * 4])


@pytest.mark.parametrize(
    "seed",
    [
        lambda: [
            {"op": "LEGACY", "idx": 0},
            {"op": "LEGACY", "idx": 1},
            {"op": "LEGACY", "idx": 2},
        ],
        lambda: deque(
            [
                {"op": "LEGACY", "idx": 99},
            ],
            maxlen=1,
        ),
    ],
)
def test_play_normalizes_program_trace_container(graph_canon, seed):
    G = graph_canon()
    G.add_node(1)
    history = G.graph.setdefault("history", {})
    legacy_entries = seed()
    history["program_trace"] = legacy_entries

    program = seq(Glyph.AL, wait(1))
    play(G, program, step_fn=_step_noop)

    maxlen = int(get_param(G, "PROGRAM_TRACE_MAXLEN"))
    trace = history["program_trace"]
    assert isinstance(trace, deque)
    assert trace.maxlen == maxlen

    trace_items = list(trace)
    expected_ops = ["GLYPH", "WAIT"]
    observed_ops = [item["op"] for item in trace_items]
    suffix_length = min(len(observed_ops), len(expected_ops))
    assert observed_ops[-suffix_length:] == expected_ops[-suffix_length:]

    preserved = list(legacy_entries)
    preserve_count = max(0, min(len(preserved), maxlen - suffix_length))
    expected_legacy_tail = preserved[-preserve_count:]
    assert trace_items[:preserve_count] == expected_legacy_tail


def test_play_reuses_existing_program_trace(graph_canon):
    G = graph_canon()
    G.add_node(1)

    maxlen = int(get_param(G, "PROGRAM_TRACE_MAXLEN"))
    trace = deque(
        [
            {"op": "LEGACY", "idx": -1},
        ],
        maxlen=maxlen,
    )

    history = G.graph.setdefault("history", {})
    history["program_trace"] = trace

    program = seq(Glyph.AL, wait(1))
    play(G, program, step_fn=_step_noop)

    assert history["program_trace"] is trace
    ops = [item["op"] for item in trace]
    expected_ops = ["LEGACY", "GLYPH", "WAIT"]
    suffix_length = min(len(ops), len(expected_ops))
    assert ops[-suffix_length:] == expected_ops[-suffix_length:]


def test_wait_logs_sanitized_steps(graph_canon):
    G = graph_canon()
    G.add_node(1)
    play(G, seq(WAIT(0)), step_fn=_step_noop)
    trace = G.graph["history"]["program_trace"]
    assert [e["op"] for e in trace] == ["WAIT"]
    assert trace[0]["k"] == 1


def test_wait_clamps_non_positive_steps():
    zero_wait = wait(0)
    negative_wait = wait(-3)
    positive_wait = wait(2)

    assert zero_wait.steps == 1
    assert negative_wait.steps == 1
    assert positive_wait.steps == 2


def test_flatten_wait_sanitizes_steps(monkeypatch):
    program = seq(WAIT(-2.5), WAIT(2.4))
    expected = [(OpTag.WAIT, 1), (OpTag.WAIT, 2)]
    assert compile_sequence(program) == expected

    calls: list[object] = []
    original = flatten_module.ensure_collection

    def spy(it, *args, **kwargs):
        calls.append(it)
        return original(it, *args, **kwargs)

    monkeypatch.setattr(flatten_module, "ensure_collection", spy)

    def wait_stream():
        yield WAIT(-2.5)
        yield WAIT(2.4)

    assert compile_sequence(wait_stream()) == expected
    assert len(calls) == 1
    assert not isinstance(calls[0], Collection)


def test_flatten_accepts_wait_subclass():
    class CustomWait(WAIT):
        pass

    program = seq(CustomWait(3))
    ops = compile_sequence(program)
    assert ops == [(OpTag.WAIT, 3)]


def test_compile_sequence_rejects_non_iterable():
    with pytest.raises(TypeError, match="is not iterable"):
        compile_sequence(object())


def test_compile_sequence_emits_thol_force_close_sha():
    program = [THOL(body=[Glyph.AL], force_close=Glyph.SHA)]
    ops = compile_sequence(program)
    assert ops[0] == (OpTag.THOL, Glyph.THOL.value)
    assert ops[-1] == (OpTag.GLYPH, Glyph.SHA.value)


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


def test_target_empty_selection_records_zero_count(graph_canon):
    G = graph_canon()
    G.add_nodes_from([1, 2, 3])

    play(G, seq(target([]), Glyph.AL), step_fn=_step_noop)

    trace = list(G.graph["history"]["program_trace"])
    assert trace[0]["op"] == OpTag.TARGET.name
    assert trace[0]["n"] == 0

    for node in G.nodes:
        history = G.nodes[node].get("glyph_history")
        if history is not None:
            assert list(history) == []


def test_target_generator_materializes_selection(graph_canon):
    G = graph_canon()
    G.add_nodes_from(["n1", "n2", "n3"])

    nodes = (node for node in ["n1", "n2"])
    play(G, seq(target(nodes), Glyph.AL), step_fn=_step_noop)

    trace = list(G.graph["history"]["program_trace"])
    assert trace[0]["op"] == OpTag.TARGET.name
    assert trace[0]["n"] == 2
    assert trace[1]["op"] == OpTag.GLYPH.name
    assert trace[1]["g"] == Glyph.AL.value

    assert list(G.nodes["n1"]["glyph_history"]) == [Glyph.AL.value]
    assert list(G.nodes["n2"]["glyph_history"]) == [Glyph.AL.value]
    assert "glyph_history" not in G.nodes["n3"]


@pytest.mark.parametrize(
    "exc_factory",
    [
        pytest.param(
            lambda path: StructuredFileError(path, ValueError("bad data")),
            id="structured_error",
        ),
        pytest.param(
            lambda path: OSError("permission denied"),
            id="os_error",
        ),
    ],
)
def test_load_sequence_errors_raise_system_exit(monkeypatch, caplog, tmp_path, exc_factory):
    target_path = tmp_path / "program.yaml"
    target_path.write_text("[]", encoding="utf-8")

    exception = exc_factory(target_path)

    def blow_up(path: Path) -> None:
        raise exception

    monkeypatch.setattr("tnfr.cli.execution.read_structured_file", blow_up)
    caplog.set_level("ERROR", logger="tnfr.cli.execution")

    expected_message = (
        str(exception)
        if isinstance(exception, StructuredFileError)
        else str(StructuredFileError(target_path, exception))
    )

    with pytest.raises(SystemExit) as excinfo:
        _load_sequence(target_path)

    assert excinfo.value.code == 1

    logged_messages = [record.getMessage() for record in caplog.records]
    assert expected_message in logged_messages


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
    handler = HANDLERS[OpTag.TARGET]
    curr = handler(G, TARGET(nodes), None, trace, _step_noop)
    assert curr is nodes


def test_handle_target_materializes_non_sequence(graph_canon):
    G = graph_canon()
    G.add_nodes_from([1, 2])
    trace = deque()
    nodes_view = G.nodes()
    handler = HANDLERS[OpTag.TARGET]
    curr = handler(G, TARGET(nodes_view), None, trace, _step_noop)
    assert isinstance(curr, tuple)


def test_handle_thol_none_payload_records_fallback(graph_canon):
    G = graph_canon()
    G.add_node(1)
    trace = deque()
    handler = HANDLERS[OpTag.THOL]

    result = handler(G, None, None, trace, _step_noop)

    assert result is None
    assert len(trace) == 1
    entry = trace[0]
    assert entry["op"] == OpTag.THOL.name
    assert entry["g"] == Glyph.THOL.value


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

    expected = flatten_module.parse_program_tokens(data)
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

    expected = flatten_module.parse_program_tokens(data)
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
    ops = compile_sequence(program)
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
    ops = compile_sequence(program)
    assert ops == [(OpTag.GLYPH, Glyph.AL.value), (OpTag.GLYPH, Glyph.OZ.value)]


def test_flatten_plain_sequence_skips_materialization(monkeypatch):
    called = False
    original = flatten_module.ensure_collection

    def spy(it, *args, **kwargs):
        nonlocal called
        called = True
        return original(it, *args, **kwargs)

    monkeypatch.setattr(flatten_module, "ensure_collection", spy)
    ops = compile_sequence([Glyph.AL, Glyph.RA])
    assert ops == [
        (OpTag.GLYPH, Glyph.AL.value),
        (OpTag.GLYPH, Glyph.RA.value),
    ]
    assert called is False


def test_flatten_enforces_limit_for_iterables():
    def token_stream():
        yield Glyph.AL
        yield Glyph.RA
        yield Glyph.ZHIR
        yield Glyph.OZ

    with pytest.raises(ValueError, match=r"Iterable produced 4 items, exceeds limit 3"):
        compile_sequence(token_stream(), max_materialize=3)


def test_compile_sequence_zero_materialize_stops_iteration():
    events: list[str] = []

    def token_stream():
        events.append("iterated")
        yield Glyph.AL

    ops = compile_sequence(token_stream(), max_materialize=0)

    assert ops == []
    assert events == []


def test_thol_repeat_lt_one_raises():
    with pytest.raises(ValueError, match="repeat must be ≥1"):
        compile_sequence([THOL(body=[], repeat=0)])


def test_flatten_rejects_unsupported_token_type():
    class CustomTarget(TARGET):
        pass

    token = CustomTarget([1])
    ops = compile_sequence([token])
    assert ops == [(OpTag.TARGET, token)]

    with pytest.raises(TypeError, match="Unsupported token"):
        compile_sequence([object()])


def test_thol_evaluator_multiple_repeats():
    ops = compile_sequence([THOL(body=[Glyph.AL, Glyph.RA], repeat=3)])
    assert ops == [
        (OpTag.THOL, Glyph.THOL.value),
        (OpTag.GLYPH, Glyph.AL.value),
        (OpTag.GLYPH, Glyph.RA.value),
        (OpTag.GLYPH, Glyph.AL.value),
        (OpTag.GLYPH, Glyph.RA.value),
        (OpTag.GLYPH, Glyph.AL.value),
        (OpTag.GLYPH, Glyph.RA.value),
    ]


def test_thol_evaluator_body_limit_error_message():
    body = (Glyph.AL for _ in range(5))
    with pytest.raises(ValueError, match="THOL body exceeds max_materialize=3"):
        compile_sequence([THOL(body=body)], max_materialize=3)


def test_thol_recursive_expansion():
    inner = THOL(body=[Glyph.RA], repeat=2)
    outer = THOL(body=[Glyph.AL, inner, Glyph.ZHIR])
    ops = compile_sequence([outer])
    assert ops == [
        (OpTag.THOL, Glyph.THOL.value),
        (OpTag.GLYPH, Glyph.AL.value),
        (OpTag.THOL, Glyph.THOL.value),
        (OpTag.GLYPH, Glyph.RA.value),
        (OpTag.GLYPH, Glyph.RA.value),
        (OpTag.GLYPH, Glyph.ZHIR.value),
    ]


def test_compile_sequence_rejects_noncanonical_glyph():
    with pytest.raises(ValueError, match="Non-canonical glyph"):
        compile_sequence(["NOT_CANON"])


@pytest.mark.parametrize(
    "bad, message",
    [
        (THOL(body=[Glyph.AL], repeat=0), "repeat must be ≥1"),
        (THOL(body=[Glyph.AL], force_close="AL"), "force_close must be a Glyph"),
    ],
)
def test_thol_nested_parameter_errors(bad, message):
    outer = THOL(body=[bad])
    with pytest.raises(ValueError, match=message):
        compile_sequence([outer])
