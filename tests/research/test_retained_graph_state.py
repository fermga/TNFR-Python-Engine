"""Strict tagged-state admission, lossless context and detached reconstruction."""

from collections import deque
from copy import deepcopy
from fractions import Fraction as Q
from pathlib import Path

import networkx as nx
import pytest

from tests.research import test_collective_pressure_handoff_evidence as retained
from tnfr.mathematics.epi import BEPIElement
from tnfr.operators.grammar_debt import node_debt
from tnfr.operators.grammar_dynamics import validate_sequence_incremental
from tnfr.physics.relational_sine_support import bound_relational_sine_support_exchange
from tnfr.research.artifact_io import file_receipt
from tnfr.research.retained_graph_state import (
    decode_recorded_container,
    encode_recorded_container,
    record_graph_state,
    restore_recorded_graph_state,
)
from tnfr.types import require_finite_real_scalar_epi


def _state():
    return {
        "graph_type": "networkx.classes.graph.Graph",
        "node_order": (0, 2, 1),
        "neighbor_order": ((0, (2, 1)), (2, (0, 1)), (1, (0, 2))),
        "graph": {
            "GRAMMAR_REJECTION_MODE": "raise",
            "nested": {"values": [1, Q(1, 3)]},
        },
        "nodes": (
            (0, {"EPI": -0.0, "glyph_history": deque(("IL", "OZ"), maxlen=8)}),
            (2, {"EPI": -0.5}),
            (1, {"EPI": 0.75}),
        ),
        "edges": ((0, 2, {"weight": 0.5}), (0, 1, {"weight": 1.0}), (2, 1, {})),
        "object_state": {"_last_operator_applied": "dissonance"},
        "omitted_rebuildable_caches": (),
        "omitted_networkx_views": (),
        "omitted_networkx_cache": (),
    }


def _restore(state):
    return restore_recorded_graph_state(encode_recorded_container(state))


def test_nested_container_roundtrip_preserves_types_bits_and_detaches_mutable_values():
    source = {
        "tuple": (Q(1, 3), -0.0, complex(-0.5, 0)),
        "list": [True, None],
        "deque": deque(("IL", "OZ"), maxlen=8),
    }
    record = encode_recorded_container(source)
    decoded = decode_recorded_container(record)
    assert decoded == source
    assert type(decoded["tuple"]) is tuple and type(decoded["list"]) is list
    assert type(decoded["deque"]) is deque and decoded["deque"].maxlen == 8
    assert decoded["tuple"][1].hex() == "-0x0.0p+0"
    decoded["list"].append("new")
    decoded["deque"].append("IL")
    assert source["list"] == [True, None] and tuple(source["deque"]) == ("IL", "OZ")
    assert decode_recorded_container(record) == source


@pytest.mark.parametrize("text", ["nan", "inf", "-inf", "0x1p0", "invalid"])
def test_nonfinite_noncanonical_or_invalid_binary64_rejects(text):
    with pytest.raises(ValueError):
        decode_recorded_container({"type": "binary64", "hex": text})


@pytest.mark.parametrize(
    "record",
    [
        {"type": "builtins.dict", "items": [["x", 1], ["x", 2]]},
        {"type": "builtins.dict", "items": [[True, 1], [1, 2]]},
        {"type": "builtins.dict", "items": [[[1], 2]]},
        {"type": "builtins.dict", "items": [["x"]]},
        {"type": "builtins.list", "items": [], "extra": True},
        {"type": "builtins.tuple", "items": {1, 2}},
        {"type": "arbitrary.Class", "fields": {}},
        {"type": "builtins.complex", "real": True, "imaginary": 0},
    ],
)
def test_duplicate_malformed_or_unexpected_container_records_reject(record):
    with pytest.raises(ValueError):
        decode_recorded_container(record)


@pytest.mark.parametrize("maxlen", [True, -1, 0, 1, 2.0, "2"])
def test_invalid_or_truncating_deque_bound_rejects(maxlen):
    with pytest.raises(ValueError):
        decode_recorded_container(
            {"type": "collections.deque", "items": ["IL", "OZ"], "maxlen": maxlen}
        )


def test_empty_zero_and_unbounded_deques_preserve_explicit_bounds():
    for value in (deque((), maxlen=0), deque(("IL",), maxlen=None)):
        result = decode_recorded_container(encode_recorded_container(value))
        assert result == value and result.maxlen == value.maxlen


def test_graph_restore_preserves_support_order_shared_edges_and_input_detachment():
    state = _state()
    original = deepcopy(state)
    graph = _restore(state).graph
    assert tuple(graph) == (0, 2, 1)
    assert tuple(graph.edges) == ((0, 2), (0, 1), (2, 1))
    assert (
        tuple((node, tuple(graph[node])) for node in graph) == state["neighbor_order"]
    )
    assert graph._adj[0][2] is graph._adj[2][0]
    assert graph._last_operator_applied == "dissonance"
    graph[0][2]["weight"] = 7
    graph.graph["nested"]["values"].append(8)
    graph.nodes[0]["glyph_history"].append("IL")
    assert state == original


def test_mixed_parent_and_child_labels_roundtrip_without_coercion_or_reordering():
    graph = nx.Graph()
    graph.add_nodes_from((2, "2", "2_sub_0", 0))
    graph.add_edge(2, 0, weight=0.75)
    graph.add_edge(2, "2", weight=0.5)
    graph.graph["hierarchy"] = {2: ["2_sub_0"]}
    graph.nodes[2]["sub_nodes"] = ["2_sub_0"]
    graph.nodes["2_sub_0"].update(parent_node=2, _hierarchy_path=[2])
    record = record_graph_state(graph)
    restored = restore_recorded_graph_state(record).graph
    assert tuple(restored) == (2, "2", "2_sub_0", 0)
    assert tuple(restored.edges) == ((2, 0), (2, "2"))
    assert tuple(restored[2]) == (0, "2")
    assert restored._adj[2]["2"] is restored._adj["2"][2]
    assert restored.degree("2_sub_0") == 0
    assert restored.graph["hierarchy"] == {2: ["2_sub_0"]}
    assert restored.nodes["2_sub_0"]["parent_node"] == 2
    restored.nodes["2_sub_0"]["_hierarchy_path"].append("new")
    assert graph.nodes["2_sub_0"]["_hierarchy_path"] == [2]
    # Omission inventories reflect cached views, not persistent state. Compare
    # the original captured primitives to an unobserved fresh reconstruction.
    pristine = restore_recorded_graph_state(record).graph
    assert decode_recorded_container(record_graph_state(pristine))["nodes"] == (
        decode_recorded_container(record)["nodes"]
    )


@pytest.mark.parametrize("label", [False, True, 0.0, Q(0), "", None, (0,)])
@pytest.mark.parametrize(
    "field", ["node_order", "nodes", "edges", "neighbor_owner", "neighbor"]
)
def test_node_identity_admission_precedes_equality_in_every_association(label, field):
    state = _state()
    if field == "node_order":
        state[field] = (label, 2, 1)
    elif field == "nodes":
        state[field] = ((label, {}), *state[field][1:])
    elif field == "edges":
        state[field] = ((label, 2, {}), *state[field][1:])
    elif field == "neighbor_owner":
        state["neighbor_order"] = ((label, (2, 1)), *state["neighbor_order"][1:])
    else:
        state["neighbor_order"] = ((0, (2, label)), *state["neighbor_order"][1:])
    with pytest.raises((ValueError, TypeError)):
        _restore(state)


def test_duplicate_string_labels_and_string_integer_substitutions_reject():
    state = _state()
    state["node_order"] = ("child", "child", 1)
    with pytest.raises(ValueError, match="duplicate"):
        _restore(state)
    state = _state()
    state["nodes"] = (("0", {}), *state["nodes"][1:])
    with pytest.raises(ValueError, match="association"):
        _restore(state)


@pytest.mark.parametrize(
    "key,value",
    [
        ("graph_type", "networkx.classes.digraph.DiGraph"),
        ("node_order", (False, 2, 1)),
        ("node_order", (0, 0, 1)),
        ("node_order", [0, 2, 1]),
        ("graph", ()),
        ("nodes", ((False, {}), (2, {}), (1, {}))),
        ("edges", ((0, 2, {}), (2, 0, {}))),
        ("edges", ((False, 2, {}),)),
        ("edges", ((0, 7, {}),)),
        ("edges", ((0, 0, {}),)),
        ("edges", ((2, 0, {}), (0, 1, {}), (2, 1, {}))),
        ("edges", ((0, 1, {}), (0, 2, {}), (2, 1, {}))),
        ("neighbor_order", ((0, (2, 2)), (2, (0, 1)), (1, (0, 2)))),
        ("neighbor_order", ((0, (2, True)), (2, (0, 1)), (1, (0, 2)))),
        ("object_state", {"unexpected": "value"}),
        ("object_state", {"_last_operator_applied": 3}),
        ("omitted_networkx_views", ({"key": "nodes"},)),
        ("omitted_networkx_cache", ("not a mapping",)),
    ],
)
def test_malformed_graph_schema_labels_support_order_and_context_reject(key, value):
    state = _state()
    state[key] = value
    with pytest.raises((ValueError, TypeError)):
        _restore(state)


@pytest.mark.parametrize("change", ["extra", "missing"])
def test_graph_schema_rejects_extra_or_missing_fields(change):
    state = _state()
    if change == "extra":
        state["unexpected"] = 1
    else:
        state.pop("graph")
    with pytest.raises(ValueError):
        _restore(state)


def test_duplicate_or_simultaneously_supplied_omission_metadata_rejects():
    row = {
        "key": "_node_cache",
        "type": "builtins.dict",
        "scope": "rebuildable",
        "owner": "tnfr.utils.cache.GRAPH_RUNTIME_CACHE_KEYS",
    }
    state = _state()
    state["omitted_rebuildable_caches"] = (row, row.copy())
    with pytest.raises(ValueError, match="duplicate"):
        _restore(state)
    state["omitted_rebuildable_caches"] = (row,)
    state["graph"]["_node_cache"] = {}
    with pytest.raises(ValueError, match="also supplied"):
        _restore(state)


def test_writer_does_not_mutate_input_and_records_only_declared_cache_omissions():
    graph = nx.Graph()
    graph.add_edge(2, 1, weight=0.75)
    graph.graph["_node_cache"] = object()
    graph.graph["_cache_config"] = {"enabled": True}
    graph._last_operator_applied = "silence"
    before_keys = tuple(vars(graph))
    record = record_graph_state(graph)
    assert tuple(vars(graph)) == before_keys
    state = decode_recorded_container(record)
    assert state["omitted_rebuildable_caches"][0]["key"] == "_node_cache"
    restored = restore_recorded_graph_state(record).graph
    assert "_node_cache" not in restored.graph
    assert restored.graph["_cache_config"] == {"enabled": True}
    assert restored._last_operator_applied == "silence"
    graph.unsupported = object()
    with pytest.raises(ValueError, match="instance"):
        record_graph_state(graph)


def test_uniform_real_bepi_writer_uses_shared_signed_admission_without_projection():
    graph = nx.Graph()
    graph.add_node(0, EPI=BEPIElement((-0.5, -0.5), (-0.5, -0.5), (0.0, 1.0)))
    restored = restore_recorded_graph_state(record_graph_state(graph)).graph
    value = restored.nodes[0]["EPI"]
    assert set(value) == {"continuous", "discrete", "grid"}
    assert require_finite_real_scalar_epi(value, "test") == -0.5
    graph.nodes[0]["EPI"] = BEPIElement((1, 2), (1, 1), (0.0, 1.0))
    with pytest.raises(ValueError):
        record_graph_state(graph)


@pytest.fixture(scope="module")
def source():
    base = (
        Path(__file__).resolve().parents[2]
        / "docs/assets/collective_pressure_handoff/correction_v2"
    )
    paths = tuple(sorted(path for path in base.iterdir() if path.is_file()))
    receipts = tuple(file_receipt(path) for path in paths)
    capture = retained._read("response.json")["captures"]["OZ_completed"]
    yield capture
    assert paths == tuple(sorted(path for path in base.iterdir() if path.is_file()))
    assert receipts == tuple(file_receipt(path) for path in paths)


def test_retained_capture_matches_shared_primitive_reader_and_preserves_live_debt(
    source,
):
    original = deepcopy(source)
    result = restore_recorded_graph_state(source["state"])
    graph = result.graph
    expected = retained._rebuild(source)
    actual = bound_relational_sine_support_exchange(
        graph, reference_model=retained.MODEL
    )
    assert actual == expected
    assert result.omitted_rebuildable_caches
    assert graph._last_operator_applied == "dissonance"
    assert tuple(node_debt(graph.nodes[node]) for node in (0, 1)) == (1, 1)
    assert tuple(graph.nodes[0]["glyph_history"])[-2:] == ("IL", "OZ")
    assert all(graph.nodes[node]["glyph_history"].maxlen == 8 for node in (0, 1))
    assert source == original


def test_read_only_nul_il_sha_admission_preserves_context_and_executes_no_event(source):
    graph = restore_recorded_graph_state(source["state"]).graph
    # The diagnostic reads NodeView; materialize that rebuildable native view
    # before comparing complete inventories as well as persistent state.
    assert tuple(graph.nodes) == tuple(graph)
    before = record_graph_state(graph)
    for node in (0, 1):
        verdicts = validate_sequence_incremental(graph, node, ("NUL", "IL", "SHA"))
        assert all(verdict.allowed for verdict in verdicts)
        assert node_debt(graph.nodes[node]) == 1
    assert record_graph_state(graph) == before
