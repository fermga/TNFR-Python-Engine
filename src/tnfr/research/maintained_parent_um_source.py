"""Read the retained THOL successor before detached default-UM analysis.

No producer, proposal or event is replayed. The preceding history certificate,
actual birth reset and successor law are re-admitted from primitive records.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction as Q
from io import BytesIO
from pathlib import Path

from ..alias import get_attr
from ..constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_SI, ALIAS_THETA, ALIAS_VF
from ..operators.definitions import Coupling, Expansion, SelfOrganization, Silence
from ..operators.grammar_debt import node_debt
from ..operators.grammar_dynamics import validate_candidate
from ..operators.grammar_execution import ValidatedSequence
from ..operators.nodal_equation import observe_structural_acceleration
from ..operators.preconditions import validate_coupling
from ..physics.relational_sine_support import bound_relational_sine_frozen_isolates
from ..utils.io import json_loads
from .artifact_io import (
    decode_exact_tree,
    file_receipt,
    read_bytes_bounded,
    sha256_bytes,
    verify_archive_members,
)
from .local_capacity_history import CONTEXT, MODEL, PARENT, TIMES
from .maintained_parent_thol import (
    _check_receipt,
    _contained,
    _represented,
    _signature,
    load_retained_parent,
    validate_live_parent,
)
from .maintained_parent_thol_prediction import predict_thol_reset
from .retained_graph_state import (
    encode_recorded_container,
    restore_recorded_graph_state,
)

CHILD = "2_sub_0"
_MAX_BYTES = 32 * 1024**2


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _value(graph, node, alias):
    return _represented(
        get_attr(graph.nodes[node], alias, strict=True, conv=lambda value: value),
        f"state at node {node!r}",
    )


def _same(left, right):
    return encode_recorded_container(left) == encode_recorded_container(right)


@dataclass(frozen=True)
class TholSuccessorAdmission:
    """Fresh mutable graph and independently rebuilt pre-event premises."""

    graph: object
    prebirth: object
    prediction: object
    fields: object
    sequence: ValidatedSequence
    receipts: tuple[dict, ...] = ()


def admit_thol_successor(response, *, prebirth):
    """Reconstruct actual reset primitives; cached fields/verdicts are ignored."""
    _require(
        type(response) is dict
        and response.get("schema") == "tnfr.maintained-parent-thol-response.v1",
        "unsupported retained THOL response",
    )
    prebirth = validate_live_parent(prebirth)
    before = restore_recorded_graph_state(response["before"]).graph
    raw = restore_recorded_graph_state(response["raw_after"]).graph
    graph = restore_recorded_graph_state(response["endpoint"]).graph
    _require(
        _signature(before) == _signature(prebirth.graph),
        "birth source differs from certified parent",
    )
    prediction = predict_thol_reset(before)
    old_nodes = tuple(before)
    expected_nodes = old_nodes + (CHILD,)
    fields = bound_relational_sine_frozen_isolates(graph, reference_model=MODEL)
    raw_fields = bound_relational_sine_frozen_isolates(raw, reference_model=MODEL)
    event_graph_keys = {
        "hierarchy",
        "_last_network_stage_schedule",
        "_last_network_stage_contract",
        # Alias-owned maxima track the deliberately distinct raw pressure.
        "_dnfrmax",
        "_dnfrmax_node",
    }
    for state in (raw, graph):
        _require(
            _same(
                {
                    key: value
                    for key, value in before.graph.items()
                    if key not in event_graph_keys
                },
                {
                    key: value
                    for key, value in state.graph.items()
                    if key not in event_graph_keys
                },
            ),
            "birth changed undeclared graph configuration or context",
        )
    _require(
        fields.nodes == raw_fields.nodes == expected_nodes,
        "birth must append its single associated child",
    )
    _require(
        fields.isolate_indices == raw_fields.isolate_indices == (20,),
        "child must remain the sole isolate",
    )
    _require(
        _same(tuple(before.edges(data=True)), tuple(raw.edges(data=True)))
        and _same(tuple(before.edges(data=True)), tuple(graph.edges(data=True))),
        "birth changed old support or conductances",
    )
    for node in old_nodes:
        for alias in (ALIAS_EPI, ALIAS_THETA, ALIAS_VF):
            _require(
                _value(before, node, alias)
                == _value(raw, node, alias)
                == _value(graph, node, alias),
                "birth changed an old continuous coordinate",
            )
        _require(
            _value(graph, node, ALIAS_DNFR) == _value(before, node, ALIAS_DNFR),
            "successor pressure is not the retained fresh map",
        )
        if node != PARENT:
            _require(
                _same(before.nodes[node], raw.nodes[node])
                and _same(before.nodes[node], graph.nodes[node]),
                "birth changed unrelated node context",
            )
    for state in (raw, graph):
        parent, child = state.nodes[PARENT], state.nodes[CHILD]
        form = _value(state, CHILD, ALIAS_EPI)
        _require(
            prediction.child_form_bounds.lo <= form <= prediction.child_form_bounds.hi,
            "child form violates the independent construction bound",
        )
        _require(
            _value(state, CHILD, ALIAS_VF) == prediction.child_capacity
            and _value(state, CHILD, ALIAS_THETA) == prediction.child_phase
            and _value(state, CHILD, ALIAS_DNFR) == 0,
            "child capacity, phase or pressure differs from its reset",
        )
        _require(
            parent.get("sub_nodes") == [CHILD]
            and state.graph.get("hierarchy") == {PARENT: [CHILD]}
            and type(child.get("parent_node")) is int
            and child["parent_node"] == PARENT
            and all(
                type(child.get(key)) is int and child[key] == 1
                for key in ("hierarchy_level", "_bifurcation_level")
            )
            and child.get("_hierarchy_path") == [PARENT],
            "invalid retained hierarchy",
        )
        _require(
            tuple(parent.get("glyph_history", ())) == ("VAL", "THOL")
            and type(parent.get("_grammar_u2_debt")) is int
            and parent["_grammar_u2_debt"] == 0
            and node_debt(parent) == 0,
            "retained parent grammar context differs",
        )
        _require(
            _same(
                parent.get("epi_time_history"), before.nodes[PARENT]["epi_time_history"]
            )
            and _represented(state.graph.get("_t"), "structural time") == TIMES[-1],
            "birth changed the pre-birth history or structural clock",
        )
        _require(
            child.get("glyph_history") == []
            and child.get("epi_history") == [float(form)]
            and "epi_time_history" not in child
            and not observe_structural_acceleration(state, CHILD).available,
            "child cannot borrow a temporal baseline",
        )
        records = parent.get("sub_epis")
        _require(
            type(records) is list and len(records) == 1, "one child record required"
        )
        record = records[0]
        _require(
            record.get("node_id") == CHILD
            and record.get("metabolized") is True
            and _represented(record.get("epi"), "child record EPI") == form
            and _represented(record.get("vf"), "child record capacity")
            == prediction.child_capacity
            and _represented(record.get("d2_epi"), "recorded acceleration")
            == prediction.observed_acceleration
            and _represented(record.get("tau"), "recorded threshold")
            == prebirth.threshold,
            "child record differs from its associated inputs",
        )
        _require(
            type(parent.get("_operator_step")) is int
            and parent["_operator_step"] == prediction.child_record_operator_step
            and type(record.get("timestamp")) is int
            and record["timestamp"] == prediction.child_record_operator_step
            and type(record.get("bifurcation_level")) is int
            and record["bifurcation_level"] == prediction.child_hierarchy_level
            and tuple(record.get("hierarchy_path", ()))
            == prediction.child_hierarchy_path,
            "child lifecycle differs from its admitted event context",
        )
    _require(
        _same(raw.nodes[CHILD], graph.nodes[CHILD]), "refresh changed child context"
    )
    raw_parent_pressure = _value(raw, PARENT, ALIAS_DNFR)
    _require(
        prediction.raw_pressure_bounds.lo
        <= raw_parent_pressure
        <= prediction.raw_pressure_bounds.hi,
        "raw pressure differs from the independent THOL reset",
    )
    # UM consumes this diagnostic through its declared default, not a measured
    # child observation. Invalid authoritative aliases cannot fall through.
    for node in old_nodes:
        _require(_value(graph, node, ALIAS_SI) == Q(1, 2), "old sense indices differ")
    _require(
        not any(key in graph.nodes[CHILD] for key in ALIAS_SI),
        "the retained child has no supplied sense index",
    )
    sequence = ValidatedSequence(
        (Expansion(), SelfOrganization(), Coupling(), Silence()), context=CONTEXT
    )
    signature = _signature(graph)
    validate_coupling(graph, PARENT)
    grammar = validate_candidate(graph, PARENT, "UM", sequence_context=sequence.step(2))
    _require(grammar.allowed, "live parent-target UM grammar rejected")
    _require(
        _signature(graph) == signature, "read-only admission changed persistent state"
    )
    return TholSuccessorAdmission(graph, prebirth, prediction, fields, sequence)


def load_thol_successor(evidence_dir, *, repository_root):
    """Verify existing artifact bytes and admit the actual isolated successor.

    Source archives remain immutable; current compatible consuming code is not
    presumed identical to the archived producer. Association is not provenance
    authentication. The newly declared UM analysis does not execute a response.
    """
    root, folder = Path(repository_root).resolve(), Path(evidence_dir).resolve()
    raws, records, receipts = {}, {}, {}

    def read(name):
        if name not in raws:
            raws[name] = read_bytes_bounded(
                _contained(folder, name), max_bytes=_MAX_BYTES
            )
            receipts[name] = {
                "path": name,
                "bytes": len(raws[name]),
                "sha256": sha256_bytes(raws[name]),
            }
        if name not in records:
            records[name] = decode_exact_tree(json_loads(raws[name]))
        return records[name]

    freeze = read("freeze.json")
    _require(
        freeze.get("schema") == "tnfr.maintained-parent-thol-freeze.v1"
        and freeze.get("evaluation_status_at_freeze") == "not_evaluated",
        "unsupported THOL freeze",
    )
    expected = {
        "protocol.json",
        "prediction.json",
        "evaluate.py",
        "source.zip",
        "source-manifest.json",
    }
    entries = freeze["artifacts"]
    _require(
        type(entries) is tuple
        and len(entries) == len(expected)
        and {row["path"] for row in entries} == expected,
        "frozen artifact inventory differs",
    )
    for row in entries:
        name = row["path"]
        raw = read_bytes_bounded(_contained(folder, name), max_bytes=_MAX_BYTES)
        raws[name] = raw
        receipts[name] = {"path": name, "bytes": len(raw), "sha256": sha256_bytes(raw)}
        _check_receipt(receipts[name], row)
    for row in freeze["prior_artifacts"]:
        _check_receipt(
            file_receipt(
                _contained(root, row["path"]), label=row["path"], max_bytes=_MAX_BYTES
            ),
            row,
        )
    members = read("source-manifest.json")["files"]
    _require(
        len({row["path"] for row in members}) == len(members),
        "duplicate source members",
    )
    verify_archive_members(
        BytesIO(raws["source.zip"]),
        {row["path"]: row["sha256"] for row in members},
        max_bytes=_MAX_BYTES,
    )
    attempt, report = read("attempt.json"), read("report.json")
    _require(
        attempt.get("freeze_sha256") == receipts["freeze.json"]["sha256"],
        "THOL attempt association differs",
    )
    response = read("response.json")
    _check_receipt(receipts["response.json"], report["response"])
    prebirth = load_retained_parent(
        root / "docs/assets/local_capacity_history", repository_root=root
    )
    admitted = admit_thol_successor(response, prebirth=prebirth)
    return replace(admitted, receipts=tuple(receipts.values()))
