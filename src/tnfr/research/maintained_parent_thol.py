"""Re-admit the retained local-VAL endpoint before a separate THOL event.

This consumer executes neither the earlier numerical producer nor an operator
proposal. Content associations, independent history certification and live
event preconditions are separate checks; no stored verdict supplies premises.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from fractions import Fraction as Q
from io import BytesIO
from pathlib import Path

from .._exact_time import exact_or_represented_real, finite_represented_real
from ..alias import get_attr, set_attr, set_dnfr
from ..constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA
from ..mathematics._rational_interval import I
from ..operators._thol_config import resolve_thol_bifurcation_threshold
from ..operators.definitions import Expansion, SelfOrganization, Silence
from ..operators.grammar_debt import U2_DEBT_KEY, node_debt
from ..operators.grammar_dynamics import validate_candidate
from ..operators.grammar_execution import ValidatedSequence
from ..operators.nodal_equation import observe_structural_acceleration
from ..operators.preconditions.self_organization import (
    validate_self_organization_strict,
)
from ..utils.io import json_dumps, json_loads
from .artifact_io import (
    decode_exact_tree,
    encode_exact_tree,
    file_receipt,
    read_bytes_bounded,
    sha256_bytes,
    validate_artifact_path,
    verify_archive_members,
)
from .local_capacity_history import (
    CONTEXT,
    PARENT,
    PRESSURE_ERROR_BUDGET,
    TIMES,
    _admit_execution_settings,
    _fresh_pressure_bounds,
    capture,
)
from .local_capacity_history_certificate import certify_history
from .retained_graph_state import record_graph_state, restore_recorded_graph_state

_MAX_BYTES = 32 * 1024**2
_FROZEN_NAMES = frozenset(
    {
        "protocol.json",
        "source.json",
        "evaluate.py",
        "source.zip",
        "source-manifest.json",
    }
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class RetainedParentAdmission:
    """Freshly rebuilt endpoint and evidence; the graph remains mutable.

    A later event consumer must re-admit this graph immediately before use.
    The certificate applies to the retained pre-event history only.
    """

    graph: object
    source: object
    endpoint: object
    certificate: object
    observation: object
    sequence: ValidatedSequence
    threshold: Q
    pressure_defects: tuple[I, ...]
    signature: str
    receipts: tuple[dict, ...] = ()


def _represented(value, label):
    exact = exact_or_represented_real(value, label)
    _, represented = finite_represented_real(value, label)
    _require(exact == represented, f"{label} is not exactly represented in binary64")
    return represented


def _signature(graph):
    """Compare typed persistent state, excluding declared rebuildable caches."""
    record = record_graph_state(graph)
    data = dict(record)
    data["items"] = tuple(
        row for row in record["items"] if not row[0].startswith("omitted_")
    )
    return json_dumps(encode_exact_tree(data))


def _clock(graph, expected):
    _require(
        _represented(graph.graph.get("_t"), "structural clock") == expected,
        "recorded graph clock differs from its associated history time",
    )


def _actual_prefix(graph):
    data = graph.nodes[PARENT]
    _require(
        type(data.get("glyph_history")) in (list, tuple, deque)
        and tuple(data["glyph_history"]) == ("VAL",),
        "the actual target must retain exactly its VAL prefix",
    )
    _require(
        type(data.get(U2_DEBT_KEY)) is int
        and data[U2_DEBT_KEY] == node_debt(data) == 1,
        "the actual target must retain its explicit VAL debt",
    )


def _pressure_defects(graph, source):
    defects = tuple(
        I(_represented(get_attr(graph.nodes[node], ALIAS_DNFR), "stored pressure"))
        - bound
        for node, bound in zip(
            source.nodes, _fresh_pressure_bounds(source), strict=True
        )
    )
    _require(
        all(value.abs_max <= PRESSURE_ERROR_BUDGET for value in defects),
        "stored pressure differs from the independently declared map",
    )
    return defects


def _history_rows(history, label):
    _require(
        type(history) is tuple and len(history) == 3,
        f"{label} requires three samples",
    )
    admitted = []
    for row in history:
        _require(type(row) is tuple and len(row) == 2, f"invalid {label} sample")
        admitted.append(tuple(_represented(value, label) for value in row))
    return tuple(admitted)


def _live_trigger(graph, certificate):
    """Check live scalar/history and grammar contracts separately, read-only."""
    before = _signature(graph)
    observation = observe_structural_acceleration(graph, PARENT)
    validate_self_organization_strict(graph, PARENT, emit_warnings=False)
    sequence = ValidatedSequence(
        (Expansion(), SelfOrganization(), Silence()), context=CONTEXT
    )
    grammar = validate_candidate(
        graph, PARENT, "THOL", sequence_context=sequence.step(1)
    )
    threshold = _represented(
        resolve_thol_bifurcation_threshold(graph.graph), "THOL threshold"
    )
    _require(
        observation.available
        and observation.current_endpoint_matches_state
        and observation.value > threshold
        and certificate.trigger_lower_bound > threshold,
        "the retained history does not certify the unchanged strict trigger",
    )
    _require(grammar.allowed, "live THOL grammar rejected the retained target")
    _require(
        _signature(graph) == before, "admission unexpectedly changed persistent state"
    )
    return observation, sequence, threshold


def admit_retained_parent(response, *, frozen_preparation):
    """Restore and independently admit one retained pre-THOL temporal record.

    ``frozen_preparation`` is the graph record in the earlier ``source.json``.
    This lower-level reader does not associate file bytes; use
    :func:`load_retained_parent` for retained artifacts. Cached certificates,
    checks and report verdicts are deliberately not consumed as premises.
    """
    _require(
        type(response) is dict
        and response.get("schema") == "tnfr.local-capacity-history-response.v1",
        "unsupported retained history response",
    )
    frozen = restore_recorded_graph_state(frozen_preparation).graph
    prepared = restore_recorded_graph_state(response["preparation"]).graph
    source_graph = restore_recorded_graph_state(response["after_val"]).graph
    graph = restore_recorded_graph_state(response["endpoint"]).graph
    initial, source, endpoint = tuple(
        capture(item) for item in (prepared, source_graph, graph)
    )
    capture(frozen)  # Includes current live Gamma registry admission.
    for item in (frozen, prepared, source_graph, graph):
        _admit_execution_settings(item)
    _require(
        _signature(frozen) == _signature(prepared),
        "response preparation differs from frozen source",
    )
    _clock(prepared, TIMES[0])
    _clock(source_graph, TIMES[0])
    _clock(graph, TIMES[-1])
    _require(
        initial.nodes == tuple(range(20))
        and initial.capacity == (Q(1, 2), Q(1, 2)) + (Q(1),) * 18,
        "pre-VAL preparation differs from the fixed twenty-node source",
    )
    _require(
        (initial.epi, initial.phase, initial.edges, initial.conductances)
        == (source.epi, source.phase, source.edges, source.conductances),
        "local VAL changed excluded coordinates or support",
    )
    _require(
        (source.nodes, source.edges, source.conductances, source.capacity)
        == (endpoint.nodes, endpoint.edges, endpoint.conductances, endpoint.capacity),
        "history changed the held support, node order or capacity",
    )
    _actual_prefix(source_graph)
    _actual_prefix(graph)
    _require(
        not prepared.nodes[PARENT].get("glyph_history")
        and "epi_time_history" not in source_graph.nodes[PARENT],
        "source must precede its own temporal samples",
    )
    # The certificate re-admits every physical sample before comparing it.
    certificate = certify_history(source, response["times"], response["states"])
    _require(
        certificate.endpoint_budget_certified, "history endpoint error budget failed"
    )
    _require(
        endpoint.epi + endpoint.phase == certificate.represented_states[-1],
        "live endpoint differs from the certified final full state",
    )
    expected_history = tuple(
        (time, state[PARENT])
        for time, state in zip(
            certificate.times, certificate.represented_states, strict=True
        )
    )
    for label, history in (
        ("response history", response["parent_history"]),
        ("live history", graph.nodes[PARENT].get("epi_time_history")),
    ):
        _require(
            _history_rows(history, label) == expected_history,
            f"{label} differs from certified samples",
        )
    _pressure_defects(source_graph, source)
    defects = _pressure_defects(graph, endpoint)
    _require(
        all(bound.lo > 0 for bound in certificate.true_parent_pressure_bounds),
        "true parent pressure positivity is unavailable",
    )
    # Only the complete rows, independently refreshed stored pressure, clock
    # and selected temporal history may change during this retained flow.
    expected = restore_recorded_graph_state(response["after_val"]).graph
    for i, node in enumerate(endpoint.nodes):
        set_attr(expected.nodes[node], ALIAS_EPI, float(endpoint.epi[i]))
        set_attr(expected.nodes[node], ALIAS_THETA, float(endpoint.phase[i]))
        set_dnfr(expected, node, get_attr(graph.nodes[node], ALIAS_DNFR))
    expected.graph["_t"] = float(TIMES[-1])
    expected.nodes[PARENT]["epi_time_history"] = tuple(
        (float(time), float(value)) for time, value in expected_history
    )
    _require(
        _signature(expected) == _signature(graph),
        "endpoint changed undeclared persistent context",
    )
    observation, sequence, threshold = _live_trigger(graph, certificate)
    return RetainedParentAdmission(
        graph,
        source,
        endpoint,
        certificate,
        observation,
        sequence,
        threshold,
        defects,
        _signature(graph),
    )


def validate_live_parent(admitted):
    """Re-admit the mutable graph immediately before a separate event consumer.

    The signature retains typed persistent input association, not provenance
    authentication. Primitive source/history are independently recertified;
    cached bounds, endpoint reports and pass flags cannot replace premises.
    """
    _require(
        type(admitted) is RetainedParentAdmission, "retained parent admission required"
    )
    graph = admitted.graph
    _require(
        _signature(graph) == admitted.signature,
        "retained endpoint changed after admission",
    )
    endpoint = capture(graph)
    _admit_execution_settings(graph)
    _clock(graph, TIMES[-1])
    _actual_prefix(graph)
    certificate = certify_history(
        admitted.source,
        admitted.certificate.times,
        admitted.certificate.represented_states,
    )
    source = certificate.admission.source
    _require(
        certificate.endpoint_budget_certified, "history endpoint error budget failed"
    )
    _require(
        (source.nodes, source.edges, source.conductances, source.capacity)
        == (endpoint.nodes, endpoint.edges, endpoint.conductances, endpoint.capacity),
        "history changed the held support, node order or capacity",
    )
    _require(
        endpoint.epi + endpoint.phase == certificate.represented_states[-1],
        "live endpoint differs from the certified final full state",
    )
    _require(
        _history_rows(graph.nodes[PARENT].get("epi_time_history"), "live history")
        == tuple(
            (time, row[PARENT])
            for time, row in zip(
                certificate.times, certificate.represented_states, strict=True
            )
        ),
        "live history differs from certified samples",
    )
    _require(
        all(bound.lo > 0 for bound in certificate.true_parent_pressure_bounds),
        "true parent pressure positivity is unavailable",
    )
    defects = _pressure_defects(graph, endpoint)
    observation, sequence, threshold = _live_trigger(graph, certificate)
    return replace(
        admitted,
        source=source,
        endpoint=endpoint,
        certificate=certificate,
        observation=observation,
        sequence=sequence,
        threshold=threshold,
        pressure_defects=defects,
    )


def _contained(root, name):
    validate_artifact_path(name)
    path = (root / name).resolve()
    _require(path.is_relative_to(root), "artifact resolves outside its declared root")
    return path


def _receipt(raw, name):
    return {"path": name, "bytes": len(raw), "sha256": sha256_bytes(raw)}


def _check_receipt(actual, declared):
    _require(
        type(declared) is dict
        and set(declared) == {"path", "bytes", "sha256"}
        and type(declared["bytes"]) is int
        and declared == actual,
        "artifact bytes differ from their declared association",
    )


def load_retained_parent(evidence_dir, *, repository_root):
    """Verify bounded prior artifacts, then rebuild the retained live endpoint.

    The old archive is checked without extraction or import. Current source
    may include compatible migrations; pin it separately in the new study's
    freeze. Byte consistency does not authenticate acquisition or chronology.
    """
    root = Path(repository_root).resolve()
    directory = Path(evidence_dir).resolve()
    raws, records, receipts = {}, {}, {}

    def read(name):
        if name not in raws:
            raws[name] = read_bytes_bounded(
                _contained(directory, name), max_bytes=_MAX_BYTES
            )
            receipts[name] = _receipt(raws[name], name)
        if name not in records:
            records[name] = decode_exact_tree(json_loads(raws[name]))
        return records[name]

    freeze = read("freeze.json")
    _require(
        freeze.get("schema") == "tnfr.local-capacity-history-freeze.v1"
        and freeze.get("evaluation_status_at_freeze") == "not_evaluated",
        "unsupported retained freeze",
    )
    artifacts = freeze["artifacts"]
    _require(
        type(artifacts) is tuple
        and len(artifacts) == len(_FROZEN_NAMES)
        and {entry["path"] for entry in artifacts} == _FROZEN_NAMES,
        "retained freeze artifact inventory differs",
    )
    for entry in artifacts:
        name = entry["path"]
        raw = read_bytes_bounded(_contained(directory, name), max_bytes=_MAX_BYTES)
        raws[name] = raw
        receipts[name] = _receipt(raw, name)
        _check_receipt(receipts[name], entry)
    prior_receipts = []
    for entry in freeze["prior_artifacts"]:
        receipt = file_receipt(
            _contained(root, entry["path"]), label=entry["path"], max_bytes=_MAX_BYTES
        )
        _check_receipt(receipt, entry)
        prior_receipts.append(receipt)
    manifest = read("source-manifest.json")
    _require(
        manifest.get("schema") == "tnfr.source-manifest.v1",
        "unsupported source manifest",
    )
    entries = manifest["files"]
    _require(
        type(entries) is tuple
        and len({entry["path"] for entry in entries}) == len(entries),
        "duplicate source manifest entries",
    )
    verify_archive_members(
        BytesIO(raws["source.zip"]),
        {entry["path"]: entry["sha256"] for entry in entries},
        max_bytes=_MAX_BYTES,
    )
    protocol, attempt, report = (
        read(name) for name in ("protocol.json", "attempt.json", "report.json")
    )
    _require(
        protocol.get("schema") == "tnfr.local-capacity-history-protocol.v1"
        and attempt.get("schema") == "tnfr.local-capacity-history-attempt.v1"
        and report.get("schema") == "tnfr.local-capacity-history-verdict.v1",
        "unsupported retained protocol, attempt or report",
    )
    for value in (
        protocol["maximum_reserved_evaluations"],
        attempt["maximum_evaluations"],
    ):
        _require(
            type(value) is int and value == 1,
            "retained protocol requires one evaluation",
        )
    _require(
        attempt["freeze_sha256"] == receipts["freeze.json"]["sha256"],
        "attempt freeze association differs",
    )
    response = read("response.json")
    _check_receipt(receipts["response.json"], report["response"])
    source = read("source.json")
    _require(
        source.get("schema") == "tnfr.local-capacity-history-source.v1",
        "unsupported retained source",
    )
    admission = admit_retained_parent(response, frozen_preparation=source["graph"])
    return replace(admission, receipts=tuple(receipts.values()) + tuple(prior_receipts))
