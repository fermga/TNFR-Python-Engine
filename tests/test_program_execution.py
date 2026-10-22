"""Scoped program syntax, explicit glyphs and execution boundaries."""

from copy import deepcopy
from fractions import Fraction

import networkx as nx
import numpy as np
import pytest

from tnfr import execution
from tnfr.flatten import THOLEvaluator, parse_program_tokens
from tnfr.tokens import TARGET, THOL, THOL_SENTINEL, WAIT, BlockBoundary, OpTag
from tnfr.types import Glyph


def _glyphs(program):
    return [
        payload
        for tag, payload in execution.compile_sequence(program)
        if tag in {OpTag.GLYPH, OpTag.THOL}
    ]


@pytest.mark.parametrize(
    "value",
    [True, False, np.bool_(True), 1.0, 1.5, Fraction(1), Fraction(3, 2), "2", None],
)
@pytest.mark.parametrize(
    "builder", [WAIT, execution.wait, lambda n: THOL([], repeat=n)]
)
def test_counts_reject_noninteger_original_values(value, builder):
    with pytest.raises(TypeError, match="integer"):
        builder(value)


@pytest.mark.parametrize("value", [0, -1])
@pytest.mark.parametrize(
    "builder", [WAIT, execution.wait, lambda n: THOL([], repeat=n)]
)
def test_counts_reject_nonpositive_values_instead_of_clamping(value, builder):
    with pytest.raises(ValueError, match="at least 1"):
        builder(value)


def test_integer_index_inputs_are_normalized_without_float_conversion():
    class Count:
        def __index__(self):
            return 2

    assert WAIT(Count()).steps == 2
    assert THOL([], repeat=np.int64(3)).repeat == 3


@pytest.mark.parametrize("close", [None, *Glyph])
def test_closure_is_exactly_the_requested_glyph(close):
    program = [THOL([Glyph.AL], repeat=2, force_close=close)]
    body = ["AL"] + ([] if close is None else [close.value])
    assert _glyphs(program) == ["THOL", *body, *body]


def test_nested_boundaries_preserve_occurrences_and_repetition_scope():
    nested = THOL(["IL"], force_close=Glyph.SHA)
    outer = THOL([nested], repeat=2)
    operations = execution.compile_sequence([outer])
    stack = []
    entered = []
    for tag, payload in operations:
        if tag is OpTag.BLOCK_ENTER:
            assert isinstance(payload, BlockBoundary)
            stack.append(payload)
            entered.append(payload)
        elif tag is OpTag.BLOCK_EXIT:
            assert stack.pop() == payload
    assert stack == []
    assert [(b.path, b.iteration) for b in entered] == [
        ((0,), 0),
        ((0, 0, 0), 0),
        ((0,), 1),
        ((0, 1, 0), 0),
    ]
    assert _glyphs([outer]) == ["THOL", "THOL", "IL", "SHA", "THOL", "IL", "SHA"]


def test_reused_iterable_group_is_not_silently_deduplicated():
    group = ["AL", "IL"]
    assert _glyphs([group, group]) == ["AL", "IL", "AL", "IL"]


@pytest.mark.parametrize("kind", ["list", "typed", "mapping"])
def test_cyclic_nesting_rejects_instead_of_dropping_or_repeating(kind):
    body = []
    if kind == "list":
        body.append(body)
    elif kind == "typed":
        body.append(THOL(body))
    else:
        body.append({"THOL": {"body": body}})
    with pytest.raises(ValueError, match="Cyclic"):
        execution.compile_sequence(body)


@pytest.mark.parametrize(
    "token",
    [
        {"WAIT": {"steps": True}},
        {"WAIT": {"steps": Fraction(3, 2)}},
        {"WAIT": {"steps": 1, "duration": 2}},
        {"THOL": {"body": [], "repeat": "2"}},
        {"THOL": {"body": [], "depth": 2}},
        {"THOL": {"body": [], "close": "unknown"}},
        {"TARGET": [[1]]},
        {"TARGET": "node"},
        {"TARGET": {0, 1}},
        {"AL", "IL"},
        "AL -> IL",
        "A'L",
    ],
)
def test_mapping_and_glyph_validation_precedes_any_program_mutation(token):
    graph = nx.path_graph(2)
    before = deepcopy(graph)
    with pytest.raises((TypeError, ValueError)):
        execution.play(graph, ["AL", token], step_fn=lambda _: pytest.fail("stepped"))
    assert dict(graph.nodes(data=True)) == dict(before.nodes(data=True))
    assert graph.graph == before.graph


def test_mutated_typed_counts_are_readmitted_before_execution():
    token = WAIT(1)
    token.steps = True
    with pytest.raises(TypeError):
        execution.compile_sequence([token])


def test_parser_preserves_owned_explicit_nesting_without_inserting_closure():
    source = {"THOL": {"body": ["AL", {"WAIT": 2}], "repeat": 2}}
    token = parse_program_tokens([source])[0]
    source["THOL"]["body"].append("OZ")
    assert token == THOL(["AL", WAIT(2)], repeat=2)
    assert list(THOLEvaluator(token)) == [THOL_SENTINEL, "AL", WAIT(2), "AL", WAIT(2)]


def test_input_and_expansion_limits_prevent_unbounded_materialization():
    with pytest.raises(ValueError, match="max_materialize"):
        execution.compile_sequence(THOL([], repeat=100), max_materialize=10)
    with pytest.raises(ValueError, match="max_materialize"):
        parse_program_tokens(("AL" for _ in range(20)), max_materialize=10)


@pytest.mark.parametrize("token", ["AL", Glyph.IL, WAIT(1), THOL([]), {"WAIT": 1}])
def test_top_level_and_wrapped_tokens_share_zero_input_budget(token):
    for program in (token, [token]):
        with pytest.raises(ValueError, match="max_materialize"):
            parse_program_tokens(program, max_materialize=0)
    assert parse_program_tokens([], max_materialize=0) == []


def test_lexical_target_scope_and_steps_are_explicit(monkeypatch):
    graph = nx.path_graph(3)
    calls = []
    step_calls = []

    def apply(_graph, glyph, nodes):
        calls.append((glyph, tuple(_graph if nodes is None else nodes)))

    monkeypatch.setattr(execution, "_apply_glyph_to_targets", apply)
    program = [
        TARGET([0]),
        THOL(
            [
                TARGET([1]),
                "AL",
                THOL([TARGET([2]), "IL"], force_close=Glyph.SHA),
                "EN",
            ],
            repeat=2,
        ),
        "SHA",
        WAIT(2),
    ]
    execution.play(graph, program, step_fn=lambda _: step_calls.append(True))
    assert calls == [
        ("THOL", (0,)),
        ("AL", (1,)),
        ("THOL", (1,)),
        ("IL", (2,)),
        ("SHA", (2,)),
        ("EN", (1,)),
        ("AL", (1,)),
        ("THOL", (1,)),
        ("IL", (2,)),
        ("SHA", (2,)),
        ("EN", (1,)),
        ("SHA", (0,)),
    ]
    assert len(step_calls) == len(calls) + 2
    trace = graph.graph["history"]["program_trace"]
    assert sum(row["op"] == "BLOCK_ENTER" for row in trace) == 4
    assert sum(row["op"] == "BLOCK_EXIT" for row in trace) == 4


@pytest.mark.parametrize("configured_mode", [None, "fallback", "raise"])
def test_requested_glyph_is_never_replaced_and_mode_is_restored(configured_mode):
    from tnfr.operators.grammar_types import StructuralGrammarError

    graph = nx.Graph()
    graph.add_node(0, EPI=0.0, nu_f=1.0, theta=0.0, delta_nfr=0.0)
    if configured_mode is not None:
        graph.graph["GRAMMAR_REJECTION_MODE"] = configured_mode
    before = deepcopy(graph.nodes[0])
    with pytest.raises(StructuralGrammarError):
        execution.play(graph, ["SHA"], step_fn=lambda _: pytest.fail("stepped"))
    assert graph.nodes[0] == before
    assert graph.graph.get("GRAMMAR_REJECTION_MODE") == configured_mode
    assert list(graph.graph["history"]["program_trace"]) == []


def test_minimal_default_executes_actual_glyphs_without_substitution():
    graph = nx.Graph()
    graph.add_node(0, EPI=0.0, nu_f=1.0, theta=0.0, delta_nfr=0.0)
    steps = []
    execution.play(
        graph, execution.basic_canonical_example(), step_fn=lambda _: steps.append(1)
    )
    assert list(graph.nodes[0]["glyph_history"]) == ["AL", "IL", "SHA"]
    assert steps == [1, 1, 1]


def test_default_step_remains_the_runtime_with_its_own_operator_selection():
    from tnfr.constants import inject_defaults
    from tnfr.structural import create_nfr

    graph, node = create_nfr("seed", epi=0.0, vf=1.0, theta=0.0)
    inject_defaults(graph)
    execution.play(graph, execution.basic_canonical_example())
    trace = graph.graph["history"]["program_trace"]
    assert [entry["g"] for entry in trace if entry["op"] == "GLYPH"] == [
        "AL",
        "IL",
        "SHA",
    ]
    assert graph.graph["_t"] > 0
    # Each requested glyph is followed by the existing native runtime step,
    # whose independent selector can append its own history entry.
    assert len(graph.nodes[node]["glyph_history"]) > 3
    assert "GRAMMAR_REJECTION_MODE" not in graph.graph
    assert all(entry["selection_origin"] == "supplied_program" for entry in trace)
    assert all(entry["step_origin"] == "native_runtime" for entry in trace)


def test_wait_calls_the_default_step_without_disabling_its_policy(monkeypatch):
    graph = nx.Graph()
    graph.graph["GRAMMAR_REJECTION_MODE"] = "fallback"
    calls = []

    def selected_step(graph):
        calls.append(graph.graph["GRAMMAR_REJECTION_MODE"])

    monkeypatch.setattr(execution, "step", selected_step)
    execution.play(graph, [WAIT(2)])
    assert calls == ["fallback", "fallback"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("GLYPH_HYSTERESIS_WINDOW", True),
        ("PROGRAM_TRACE_MAXLEN", 2.5),
        ("GRAMMAR_REJECTION_MODE", "invalid"),
    ],
)
def test_program_configuration_is_admitted_before_history_creation(key, value):
    graph = nx.Graph()
    graph.graph[key] = value
    before = deepcopy(graph.graph)
    with pytest.raises((TypeError, ValueError)):
        execution.play(graph, [WAIT(1)], step_fn=lambda _: pytest.fail("stepped"))
    assert graph.graph == before


@pytest.mark.parametrize(
    "clock",
    [
        True,
        np.bool_(False),
        float("nan"),
        float("inf"),
        "0",
        Fraction(1, 2**2000),
        Fraction(2**2000),
    ],
)
def test_invalid_initial_trace_clock_rejects_before_graph_mutation(clock):
    graph = nx.Graph()
    graph.graph["_t"] = clock
    with pytest.raises((TypeError, ValueError), match="program trace clock"):
        execution.play(graph, [WAIT(1)], step_fn=lambda _: pytest.fail("stepped"))
    assert set(graph.graph) == {"_t"}
    assert graph.graph["_t"] is clock


@pytest.mark.parametrize("clock", [True, float("nan"), Fraction(1, 2**2000)])
def test_invalid_post_step_clock_is_not_recorded_or_rolled_back(clock):
    graph = nx.Graph()
    graph.graph["_t"] = 0.0

    def invalid_step(graph):
        graph.graph["_t"] = clock
        graph.graph["step_effect"] = "retained"

    with pytest.raises((TypeError, ValueError), match="program trace clock"):
        execution.play(graph, [WAIT(1)], step_fn=invalid_step)
    assert graph.graph["step_effect"] == "retained"
    assert graph.graph["_t"] is clock
    (entry,) = graph.graph["history"]["program_trace"]
    assert entry["t"] == 0.0
    assert entry["step_status"] == "failed"
    assert entry["record_boundary"] == "step_failed"
    assert entry["clock_sample_available"] is False
    assert entry["step_calls_completed"] == 1


def test_supplied_glyph_receipt_exists_before_step_and_keeps_event_time():
    graph = nx.Graph()
    graph.add_node(0, EPI=0.0, nu_f=1.0, theta=0.0, delta_nfr=0.0)
    graph.graph["_t"] = Fraction(1, 4)

    def observed_step(graph):
        (entry,) = graph.graph["history"]["program_trace"]
        assert entry["selection_origin"] == "supplied_program"
        assert entry["step_origin"] == "supplied_step_function"
        assert entry["targets"] == (0,)
        assert entry["g"] == "AL"
        assert entry["event_time"] == 0.25
        assert entry["event_boundary"] == "after_requested_glyph"
        assert entry["t"] == 0.25
        assert entry["step_status"] == "pending"
        assert entry["record_boundary"] == "before_step"
        assert entry["step_calls_started"] == 1
        assert entry["step_calls_completed"] == 0
        graph.graph["_t"] = Fraction(1, 2)

    execution.play(graph, ["AL"], step_fn=observed_step)
    (entry,) = graph.graph["history"]["program_trace"]
    assert entry["event_time"] == 0.25
    assert entry["t"] == 0.5
    assert entry["step_status"] == "completed"
    assert entry["record_boundary"] == "after_step"
    assert entry["step_calls_requested"] == entry["step_calls_completed"] == 1


@pytest.mark.parametrize("clock", [0.125, float("nan")])
def test_failed_step_retains_successful_glyph_and_original_exception(clock):
    graph = nx.Graph()
    graph.add_node(0, EPI=0.0, nu_f=1.0, theta=0.0, delta_nfr=0.0)
    graph.graph["GRAMMAR_REJECTION_MODE"] = "fallback"

    def failed_step(graph):
        assert graph.graph["GRAMMAR_REJECTION_MODE"] == "fallback"
        graph.graph["_t"] = clock
        raise RuntimeError("test step failed")

    with pytest.raises(RuntimeError, match="test step failed"):
        execution.play(graph, ["AL"], step_fn=failed_step)
    assert list(graph.nodes[0]["glyph_history"]) == ["AL"]
    (entry,) = graph.graph["history"]["program_trace"]
    assert entry["g"] == "AL"
    assert entry["targets"] == (0,)
    assert entry["event_time"] == 0.0
    assert entry["t"] == (0.125 if clock == 0.125 else 0.0)
    assert entry["step_status"] == "failed"
    assert entry["step_calls_requested"] == entry["step_calls_started"] == 1
    assert entry["step_calls_completed"] == 0
    assert entry["clock_sample_available"] is (clock == 0.125)


@pytest.mark.parametrize("maxlen", [0, 1])
def test_pending_trace_updates_respect_the_declared_deque_bound(maxlen):
    graph = nx.Graph()
    graph.add_node(0, EPI=0.0, nu_f=1.0, theta=0.0, delta_nfr=0.0)
    graph.graph["PROGRAM_TRACE_MAXLEN"] = maxlen
    execution.play(graph, execution.basic_canonical_example(), step_fn=lambda _: None)
    trace = graph.graph["history"]["program_trace"]
    assert trace.maxlen == len(trace) == maxlen
    if maxlen:
        assert trace[0]["g"] == "SHA"
        assert trace[0]["step_status"] == "completed"


def test_wait_failure_retains_completed_call_count_without_a_glyph_claim():
    graph = nx.Graph()
    calls = []

    def partial_step(graph):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("second call failed")
        graph.graph["_t"] = 0.125

    with pytest.raises(RuntimeError, match="second call failed"):
        execution.play(graph, [WAIT(3)], step_fn=partial_step)
    (entry,) = graph.graph["history"]["program_trace"]
    assert entry["op"] == "WAIT"
    assert "g" not in entry and "event_time" not in entry
    assert entry["t"] == 0.125
    assert entry["step_status"] == "failed"
    assert entry["step_calls_requested"] == 3
    assert entry["step_calls_started"] == 2
    assert entry["step_calls_completed"] == 1


def test_glyph_targets_are_captured_before_population_changes(monkeypatch):
    graph = nx.Graph()
    graph.add_node("parent")

    def simulated_dispatch(graph, glyph, targets):
        if glyph == "THOL":
            assert targets == ["parent"]
            graph.add_node("child")

    monkeypatch.setattr(execution, "_apply_glyph_to_targets", simulated_dispatch)
    execution.play(graph, [THOL(["IL"])], step_fn=lambda _: None)
    records = [
        entry for entry in graph.graph["history"]["program_trace"] if "g" in entry
    ]
    assert [(entry["g"], entry["targets"]) for entry in records] == [
        ("THOL", ("parent",)),
        ("IL", ("parent", "child")),
    ]


def test_target_trace_retains_ordered_ids_and_explicit_population_snapshot(monkeypatch):
    graph = nx.Graph()
    graph.add_nodes_from((2, 0))
    monkeypatch.setattr(execution, "_apply_glyph_to_targets", lambda *args: None)
    execution.play(graph, [TARGET(None), "AL"], step_fn=lambda _: None)
    target_entry, glyph_entry = graph.graph["history"]["program_trace"]
    assert target_entry["targets"] == glyph_entry["targets"] == (2, 0)
    assert target_entry["selection_origin"] == "supplied_program"


@pytest.mark.parametrize("targets", [[0, 1], [(0, 1)]])
def test_program_targets_disambiguate_collections_from_tuple_node_ids(targets):
    graph = nx.Graph()
    for node in (0, 1, (0, 1)):
        graph.add_node(node, EPI=0.0, nu_f=1.0, theta=0.0, delta_nfr=0.0)
    before = deepcopy(dict(graph.nodes(data=True)))

    execution.play(graph, [TARGET(targets), "AL"], step_fn=lambda _: None)

    for node in graph:
        if node in targets:
            assert list(graph.nodes[node]["glyph_history"]) == ["AL"]
            assert dict(graph.nodes[node]) != before[node]
        else:
            assert dict(graph.nodes[node]) == before[node]
    target_entry, glyph_entry = graph.graph["history"]["program_trace"]
    assert target_entry["targets"] == glyph_entry["targets"] == tuple(targets)


def test_presets_explicitly_retain_their_previously_inserted_closures():
    from tnfr.config.presets import get_preset

    assert _glyphs(get_preset("contained_mutation")) == [
        "AL",
        "EN",
        "THOL",
        "OZ",
        "ZHIR",
        "IL",
        "NUL",
        "OZ",
        "ZHIR",
        "IL",
        "NUL",
        "RA",
        "SHA",
    ]
    assert _glyphs(get_preset("coupling_exploration")) == [
        "AL",
        "EN",
        "IL",
        "VAL",
        "UM",
        "THOL",
        "OZ",
        "NAV",
        "IL",
        "NUL",
        "RA",
        "SHA",
    ]


def test_cli_sequence_reader_uses_the_same_typed_admission(tmp_path):
    from tnfr.cli.execution import _load_sequence

    path = tmp_path / "program.json"
    path.write_text('{"sequence":[{"THOL":{"body":["IL"],"close":"SHA"}}]}')
    assert _load_sequence(path) == [THOL(["IL"], force_close=Glyph.SHA)]
    path.write_text('{"sequence":[{"WAIT":true}]}')
    with pytest.raises(TypeError, match="integer"):
        _load_sequence(path)
