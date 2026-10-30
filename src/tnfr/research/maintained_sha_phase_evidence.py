"""Bounded passive admission of complete retained SHA phase tubes.

This reader checks byte associations, primitive/source consistency, exact
chronology and saved Taylor arithmetic. It never evaluates a field, jet,
Picard search, event or trajectory. Derivative and remainder correctness stay
conditional on the frozen producer. Connected records contain no tube rates:
their positive Picard margin is explicitly inherited, not reverified here.
Cut records permit arithmetic verification of the saved-rate Picard image,
but the saved rates' association with the differential equation is inherited.

The outer source-associated consumer must independently admit the fresh source
and the full closed producer/instrument/final-state inventory. Supplied anchor
hashes prove byte consistency, not historical authentication. A live returned
capability means this bounded passive check completed, never that those outer
premises were independently established by a cached report.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from fractions import Fraction as Q
from io import BytesIO
from itertools import islice
from pathlib import Path
from weakref import WeakValueDictionary
from zipfile import ZipFile

from ..mathematics._rational_interval import I
from ..mathematics._validated_taylor import reconstruct_box_taylor_arithmetic
from ..physics.relational_sine_support import _support_topology
from ..utils.io import json_dumps, json_loads
from .artifact_io import (
    decode_exact_tree,
    encode_exact_tree,
    exact_record,
    file_receipt,
    read_bytes_bounded,
    sha256_bytes,
    validate_artifact_path,
    verify_archive_members,
)
from .local_capacity_history import MODEL, TIMES
from .maintained_parent_sha_source import _same
from .maintained_parent_thol import _signature
from .maintained_principal_retention import HORIZON, ORDER, STEP, STEP_COUNT
from .maintained_sha_cut_separator import CutSeparatorSource, _capture_cut_source
from .retained_graph_state import restore_recorded_graph_state

MAX_ARCHIVE_BYTES = 256 * 1024
MAX_MEMBER_BYTES = 1024**2
MAX_ARCHIVES_BYTES = 64 * 1024**2
MAX_EXPANDED_BYTES = 128 * 1024**2
MAX_METADATA_BYTES = 32 * 1024**2
MAX_ANCHOR_BYTES = 128 * 1024**2
MAX_ANCHORS_BYTES = 256 * 1024**2
MAX_PRODUCER_RECEIPTS = 16
MAX_NUMERIC_BITS = 4096
MAX_JSON_DEPTH = 32
MAX_JSON_NODES = 2_000_000
_METHOD = "direct_source_box_Picard_Taylor_dyadic128_v1"
_CAPABILITIES = WeakValueDictionary()
_STEP_FIELDS = frozenset(
    (
        "time",
        "duration",
        "order",
        "initial_box",
        "tube",
        "series",
        "local_remainder_bounds",
        "increment",
        "endpoint",
        "picard_interior_margin",
        "domain_lower_bounds",
        "method",
    )
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _size(branch):
    _require(
        type(branch) is str and branch in ("connected", "cut_host"),
        "unknown phase branch",
    )
    return 42 if branch == "connected" else 40


def _q(value):
    result = exact_record(value)
    _require(
        max(abs(result.numerator).bit_length(), result.denominator.bit_length())
        <= MAX_NUMERIC_BITS,
        "exact value exceeds phase-evidence numeric bit policy",
    )
    return result


def _tree_limits(value):
    """Bound raw integer payloads and nesting before Fraction/tag decoding."""
    pending, count = [(value, 0)], 0
    while pending:
        item, depth = pending.pop()
        count += 1
        _require(
            count <= MAX_JSON_NODES and depth <= MAX_JSON_DEPTH,
            "record tree work policy exceeded",
        )
        if type(item) is int:
            _require(
                abs(item).bit_length() <= MAX_NUMERIC_BITS,
                "integer payload exceeds bit policy",
            )
        elif type(item) is Q:
            _q(item)
        elif type(item) is dict:
            _require(all(type(key) is str for key in item), "record keys must be text")
            pending.extend((child, depth + 1) for child in item.values())
        elif type(item) in (tuple, list):
            pending.extend((child, depth + 1) for child in item)


def _json(data):
    """Preflight lexical depth/numeric length, then use the shared decoder."""
    _require(
        type(data) is bytes and len(data) <= MAX_METADATA_BYTES,
        "JSON byte policy exceeded",
    )
    text = data.decode("utf-8")
    quoted = escaped = False
    depth = digits = 0
    for char in text:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted, digits = True, 0
        elif char in "[{":
            depth += 1
            digits = 0
            _require(depth <= MAX_JSON_DEPTH, "JSON nesting exceeds work policy")
        elif char in "]}":
            depth -= 1
            digits = 0
        elif char in "0123456789":
            digits += 1
            _require(digits <= 1234, "JSON numeric token exceeds bit-policy length")
        elif char not in "+-.eE":
            digits = 0
    raw = json_loads(text)
    _tree_limits(raw)
    return decode_exact_tree(raw)


def _interval(value):
    _require(
        type(value) is dict and set(value) == {"lo", "hi"},
        "canonical interval record required",
    )
    lo, hi = _q(value["lo"]), _q(value["hi"])
    interval = I(lo, hi)
    _require(
        interval.lo == lo and interval.hi == hi,
        "noncanonical dyadic interval endpoints",
    )
    return interval


def _box(values, size):
    _require(
        type(values) is tuple and len(values) == size, "complete ordered state required"
    )
    return tuple(_interval(value) for value in values)


def _initial(values, size, *, point=False):
    _require(
        type(values) is tuple
        and len(values) == size
        and all(type(v) is I for v in values),
        "initial box requires every canonical interval",
    )
    for value in values:
        _q(value.lo)
        _q(value.hi)
        _require(
            not point or value.lo == value.hi, "initial source must be an exact point"
        )
    return values


@dataclass(frozen=True)
class PhaseEvidenceStep:
    index: int
    time: Q
    duration: Q
    initial: tuple[I, ...]
    tube: tuple[I, ...]
    endpoint: tuple[I, ...]
    phases: tuple[I, ...]
    picard_status: str
    producer_premises: tuple[str, ...]


def _audit_step(record, *, index, initial, branch):
    """Check one decoded producer record without evaluating its claimed law."""
    size = _size(branch)
    initial = _initial(initial, size)
    _require(
        type(index) is int and 0 <= index < STEP_COUNT, "step index outside fixed grid"
    )
    base = {"index", "time", "step"}
    expected = base | (
        {"tube_principal_margins", "combined_principal_margins"}
        if branch == "connected"
        else {"failed_tube", "verification", "observation", "accepted", "reason"}
    )
    _require(
        type(record) is dict and set(record) == expected,
        "producer step-record schema differs",
    )
    _require(
        type(record["index"]) is int and record["index"] == index, "step index differs"
    )
    _require(_q(record["time"]) == index * STEP, "record clock differs")
    step = record["step"]
    _require(
        type(step) is dict and set(step) == _STEP_FIELDS,
        "complete Taylor schema required",
    )
    _require(
        _q(step["time"]) == index * STEP and _q(step["duration"]) == STEP,
        "Taylor clock or duration differs",
    )
    _require(
        type(step["order"]) is int
        and step["order"] == ORDER
        and step["method"] == _METHOD,
        "Taylor order or method differs",
    )
    _require(
        type(step["domain_lower_bounds"]) is tuple
        and tuple(map(_q, step["domain_lower_bounds"])) == (Q(1),),
        "global smooth domain differs",
    )
    source = _box(step["initial_box"], size)
    _require(source == initial, "exact endpoint carry differs")
    tube = _box(step["tube"], size)
    _require(
        all(x.subset_of(bound) for x, bound in zip(source, tube, strict=True)),
        "tube omits initial state",
    )
    margin = _q(step["picard_interior_margin"])
    _require(margin > 0, "positive recorded Picard margin required")
    _require(
        type(step["series"]) is tuple and len(step["series"]) == size,
        "Taylor row count differs",
    )
    series = tuple(_box(row, ORDER + 1) for row in step["series"])
    remainder = _box(step["local_remainder_bounds"], size)
    increment, endpoint = reconstruct_box_taylor_arithmetic(
        source, tube, series, remainder, STEP, order=ORDER
    )
    _require(
        increment == _box(step["increment"], size),
        "recorded increment differs from shared arithmetic",
    )
    _require(
        endpoint == _box(step["endpoint"], size),
        "recorded endpoint differs from shared arithmetic",
    )
    if branch == "connected":
        for name in ("tube_principal_margins", "combined_principal_margins"):
            _require(
                type(record[name]) is tuple and len(record[name]) == 53,
                "complete original branch margins required",
            )
            tuple(map(_q, record[name]))
        _require(
            min(map(_q, record["combined_principal_margins"])) > 0,
            "connected producer did not accept this cell",
        )
        picard_status = "inherited_connected_picard_without_saved_rates"
        premises = (
            "inherited_connected_Picard_and_principal_retention_proof",
            "inherited_held_law_derivative_and_whole_tube_remainder_enclosures",
        )
    else:
        _require(
            record["accepted"] is True
            and record["reason"] is None
            and record["failed_tube"] is None
            and type(record["observation"]) is dict,
            "cut record was not accepted with a complete observation",
        )
        verification = record["verification"]
        _require(
            type(verification) is dict
            and set(verification)
            == {
                "tube_rates",
                "picard_interior_margin",
                "solution_order",
                "remainder_order",
                "series_verified",
                "remainder_verified",
                "arithmetic_verified",
            },
            "cut verification schema differs",
        )
        _require(
            type(verification["solution_order"]) is int
            and verification["solution_order"] == ORDER
            and type(verification["remainder_order"]) is int
            and verification["remainder_order"] == ORDER + 1,
            "cut derivative order differs",
        )
        _require(
            all(
                verification[name] is True
                for name in (
                    "series_verified",
                    "remainder_verified",
                    "arithmetic_verified",
                )
            ),
            "cut producer verification failed",
        )
        rates = _box(verification["tube_rates"], size)
        image = tuple(
            x + rate * I(0, STEP) for x, rate in zip(source, rates, strict=True)
        )
        rebuilt = min(
            min(value.lo - bound.lo, bound.hi - value.hi)
            for value, bound in zip(image, tube, strict=True)
        )
        _require(
            rebuilt > 0
            and rebuilt == margin == _q(verification["picard_interior_margin"]),
            "saved-rate strict Picard arithmetic differs",
        )
        picard_status = "saved_cut_rate_Picard_arithmetic_verified"
        premises = (
            "inherited_saved_rates_association_with_complete_cut_field",
            "inherited_held_law_derivative_and_whole_tube_remainder_enclosures",
        )
    return PhaseEvidenceStep(
        index,
        index * STEP,
        STEP,
        source,
        tube,
        endpoint,
        tube[size // 2 :],
        picard_status,
        premises,
    )


@dataclass(frozen=True)
class PhaseReceipt:
    path: str
    bytes: int
    sha256: str


def _receipt(value, *, limit=MAX_ANCHOR_BYTES):
    _require(
        type(value) is dict and set(value) == {"path", "bytes", "sha256"},
        "receipt schema differs",
    )
    path = validate_artifact_path(value["path"])
    _require(
        type(value["bytes"]) is int and 0 < value["bytes"] <= limit,
        "receipt byte policy exceeded",
    )
    digest = value["sha256"]
    _require(
        type(digest) is str
        and len(digest) == 64
        and all(c in "0123456789abcdef" for c in digest),
        "receipt digest differs",
    )
    return PhaseReceipt(path, value["bytes"], digest)


def _path(folder, name):
    result = (folder / validate_artifact_path(name)).resolve()
    _require(result.is_relative_to(folder), "artifact path leaves evidence root")
    return result


def _read_record(folder, receipt):
    _require(receipt.bytes <= MAX_METADATA_BYTES, "metadata byte policy exceeded")
    raw = read_bytes_bounded(_path(folder, receipt.path), max_bytes=MAX_METADATA_BYTES)
    _require(
        len(raw) == receipt.bytes and sha256_bytes(raw) == receipt.sha256,
        "metadata receipt differs",
    )
    return _json(raw)


def _binding(binding, node_order, initial, branch):
    _require(
        type(binding) is dict
        and set(binding)
        == {"complete_source", "source_receipts", "source_graph_signature"},
        "source binding schema differs",
    )
    _tree_limits(binding)
    binding = decode_exact_tree(binding)
    source = binding["complete_source"]
    _require(
        type(source) is dict
        and set(source) == {field.name for field in fields(CutSeparatorSource)},
        "complete primitive source fields required",
    )
    nodes = source["nodes"]
    _require(
        type(nodes) is tuple
        and len(nodes) == 21
        and len(set(nodes)) == 21
        and all(type(node) is int or (type(node) is str and node) for node in nodes),
        "complete original node order required",
    )
    for name in ("epi", "phase", "capacity"):
        _require(
            type(source[name]) is tuple and len(source[name]) == 21,
            "complete source coordinates required",
        )
        source[name] = tuple(map(_q, source[name]))
    _require(min(source["capacity"]) > 0, "source capacity must be positive")
    for name in (
        "structural_time",
        "reference_conductance",
        "form_mean",
        "phase_lift_mean",
        "form_offset",
        "phase_offset",
    ):
        source[name] = _q(source[name])
    _require(source["reference_conductance"] > 0, "marked conductance must be positive")
    _require(
        source["structural_time"] == TIMES[-1],
        "original post-SHA source clock required",
    )
    for name in ("parent_index", "child_index"):
        _require(
            type(source[name]) is int and 0 <= source[name] < 21,
            "marked node index differs",
        )
    _require(
        source["parent_index"] != source["child_index"], "marked nodes must differ"
    )
    for name, count, size in (
        ("edges", 53, 21),
        ("cut_edges", 52, 21),
        ("host_edges", 52, 20),
    ):
        values = source[name]
        _require(
            type(values) is tuple
            and len(values) == count
            and all(
                type(edge) is tuple
                and len(edge) == 2
                and all(type(i) is int and 0 <= i < size for i in edge)
                for edge in values
            ),
            "complete typed source support required",
        )
    for name, count in (
        ("conductances", 53),
        ("cut_conductances", 52),
        ("host_conductances", 52),
        ("host_mean_weights", 20),
        ("host_alpha", 20),
    ):
        _require(
            type(source[name]) is tuple and len(source[name]) == count,
            "complete source coefficient vector required",
        )
        source[name] = tuple(map(_q, source[name]))
        _require(
            min(source[name]) > 0,
            "source conductances and host weights must be positive",
        )
    for name in ("connected_degrees", "cut_degrees"):
        _require(
            type(source[name]) is tuple
            and len(source[name]) == 21
            and all(type(value) is int and 0 <= value <= 20 for value in source[name]),
            "complete ordinary-integer source degrees required",
        )
    _, neighbors = _support_topology(tuple(range(21)), source["edges"])
    degrees = tuple(map(len, neighbors))
    _require(
        degrees == source["connected_degrees"], "connected degree association differs"
    )
    parent, child = source["parent_index"], source["child_index"]
    _require(
        degrees[child] == 1 and neighbors[child] == (parent,),
        "marked child is not pendant",
    )
    bridge = next(i for i, edge in enumerate(source["edges"]) if child in edge)
    _require(
        source["reference_conductance"] == source["conductances"][bridge],
        "marked scale differs from deleted edge",
    )
    cut_edges = tuple(edge for i, edge in enumerate(source["edges"]) if i != bridge)
    cut_weights = tuple(
        weight for i, weight in enumerate(source["conductances"]) if i != bridge
    )
    cut_degrees = tuple(
        degree - (i in (parent, child)) for i, degree in enumerate(degrees)
    )
    _require(
        source["cut_edges"] == cut_edges
        and source["cut_conductances"] == cut_weights
        and source["cut_degrees"] == cut_degrees,
        "cut support/coefficient association differs",
    )
    host = tuple(i for i in range(21) if i != source["child_index"])
    _require(
        type(source["host_indices"]) is tuple
        and all(type(i) is int for i in source["host_indices"])
        and source["host_indices"] == host,
        "complete host embedding differs",
    )
    local = {global_index: index for index, global_index in enumerate(host)}
    host_edges = tuple((local[i], local[j]) for i, j in cut_edges)
    _support_topology(tuple(range(20)), host_edges)
    _require(
        source["host_edges"] == host_edges
        and source["host_conductances"] == cut_weights,
        "complete host support differs",
    )
    rho = tuple(Q(cut_degrees[i]) / source["capacity"][i] for i in host)
    total = sum(rho, Q(0))
    alpha = tuple(value / total for value in rho)
    _require(
        source["host_mean_weights"] == rho and source["host_alpha"] == alpha,
        "cut mean weights differ from actual degrees and capacities",
    )
    for coordinates, mean_name, offset_name in (
        ("epi", "form_mean", "form_offset"),
        ("phase", "phase_lift_mean", "phase_offset"),
    ):
        mean = sum(
            (a * source[coordinates][i] for a, i in zip(alpha, host, strict=True)), Q(0)
        )
        _require(
            source[mean_name] == mean
            and source[offset_name] == source[coordinates][child] - mean,
            "cut conserved means or child offsets differ",
        )
    indices = tuple(range(21)) if branch == "connected" else host
    _require(
        type(node_order) is tuple
        and _same(node_order, tuple(nodes[i] for i in indices)),
        "branch node order differs",
    )
    expected = tuple(I(source[name][i]) for name in ("epi", "phase") for i in indices)
    _require(initial == expected, "branch initial state differs from primitive source")
    receipts = binding["source_receipts"]
    _require(
        type(receipts) is tuple and 0 < len(receipts) <= 4096,
        "bounded source receipt association required",
    )
    for row in receipts:
        _receipt(row)
    _require(
        len({row["path"] for row in receipts}) == len(receipts),
        "duplicate source receipt",
    )
    _require(
        type(binding["source_graph_signature"]) is str
        and binding["source_graph_signature"],
        "fresh persistent source signature required",
    )
    return binding


def _metadata(
    folder, branch, binding, initial, result_receipt, producer_receipts, source_receipt
):
    _require(
        type(producer_receipts) is tuple
        and 1 <= len(producer_receipts) <= MAX_PRODUCER_RECEIPTS,
        "bounded producer anchor inventory required",
    )
    anchors = tuple(_receipt(row) for row in producer_receipts)
    result_anchor = _receipt(result_receipt, limit=MAX_METADATA_BYTES)
    _require(result_anchor.path == "result.json", "result anchor name differs")
    if branch == "connected":
        source_anchor = _receipt(source_receipt, limit=MAX_METADATA_BYTES)
        _require(
            source_anchor.path == "source.json", "connected source anchor name differs"
        )
        anchors += (source_anchor,)
    else:
        _require(source_receipt is None, "cut source is retained inside its result")
    anchors += (result_anchor,)
    names = {row.path for row in anchors}
    _require(len(names) == len(anchors), "duplicate metadata anchor")
    _require(
        {"protocol.json", "attempt.json"} <= names, "missing protocol or attempt anchor"
    )
    _require(
        sum(row.bytes for row in anchors) <= MAX_ANCHORS_BYTES,
        "producer anchor byte budget exceeded",
    )
    indexed = {row.path: row for row in anchors}
    for row in anchors:
        _require(
            file_receipt(
                _path(folder, row.path), label=row.path, max_bytes=MAX_ANCHOR_BYTES
            )
            == asdict(row),
            "producer anchor receipt differs",
        )
    result = _read_record(folder, result_anchor)
    protocol = _read_record(folder, indexed["protocol.json"])
    attempt = _read_record(folder, indexed["attempt.json"])
    _require(
        type(result) is dict and type(attempt) is dict and type(protocol) is dict,
        "producer metadata mappings required",
    )
    _require(
        _same(result["attempt"], asdict(indexed["attempt.json"])),
        "result/attempt association differs",
    )
    complete = binding["complete_source"]
    if branch == "connected":
        _require(
            result["schema"] == "tnfr.maintained-sha-retention-result.v1"
            and attempt["schema"] == "tnfr.maintained-sha-retention-attempt.v1",
            "connected producer schemas differ",
        )
        _require("freeze.json" in indexed, "missing connected freeze anchor")
        _require(
            _same(attempt["freeze"], asdict(indexed["freeze.json"])),
            "attempt/freeze association differs",
        )
        frozen = _read_record(folder, indexed["freeze.json"])
        _require(
            frozen["schema"] == "tnfr.maintained-sha-retention-freeze.v1"
            and type(frozen["artifacts"]) is tuple,
            "connected freeze schema differs",
        )
        frozen_rows = {_receipt(row).path: row for row in frozen["artifacts"]}
        _require(
            len(frozen_rows) == len(frozen["artifacts"]), "duplicate frozen anchor"
        )
        for name in ("source.json", "protocol.json"):
            _require(
                _same(frozen_rows.get(name), asdict(indexed[name])),
                "freeze/source or protocol association differs",
            )
        source = _read_record(folder, indexed["source.json"])
        _require(
            source["schema"] == "tnfr.maintained-sha-retention-source.v1",
            "connected source schema differs",
        )
        graph = restore_recorded_graph_state(source["endpoint"]).graph
        _require(
            _signature(graph) == binding["source_graph_signature"],
            "connected persistent source differs",
        )
        captured = _capture_cut_source(
            graph,
            parent=complete["nodes"][complete["parent_index"]],
            child=complete["nodes"][complete["child_index"]],
            reference_model=MODEL,
        )
        _require(
            _same(captured, complete), "connected complete source primitives differ"
        )
        _require(
            _same(source["retained_source_receipts"], binding["source_receipts"]),
            "connected source receipts differ",
        )
        _require(
            _box(source["initial_box"], 42) == initial,
            "connected source initial box differs",
        )
        expected_protocol = {
            "schema": "tnfr.maintained-sha-retention-protocol.v1",
            "node_order": complete["nodes"],
            "law": "x_dot=M*S(theta); theta_dot=M*L_W*x; M=diag(nu/(pi*d))",
            "prior_clock": complete["structural_time"],
            "horizon": HORIZON,
            "step": STEP,
            "maximum_steps": STEP_COUNT,
            "order": ORDER,
        }
        _require(
            all(
                name in protocol and _same(protocol[name], expected)
                for name, expected in expected_protocol.items()
            ),
            "connected protocol/law/source clock differs",
        )
        report = result["report"]
        _require(
            report["status"] == "completed"
            and report["full_horizon_retention"] is True
            and _q(report["certified_retention_time"]) == HORIZON,
            "connected full-window result unavailable",
        )
        records = result["records"]
    else:
        _require(
            result["schema"] == "tnfr.cut-direct-response-result.v1"
            and attempt["schema"] == "tnfr.cut-direct-response-attempt.v1",
            "cut producer schemas differ",
        )
        _require(
            _same(result["protocol"], asdict(indexed["protocol.json"]))
            and _same(attempt["protocol"], asdict(indexed["protocol.json"])),
            "cut protocol association differs",
        )
        _require(
            _q(attempt["reference_clock"]) == complete["structural_time"]
            and _q(attempt["window"]) == HORIZON,
            "cut source clock or window differs",
        )
        report = result["report"]
        prep = report["preparation"]
        _require(
            _same(prep["source"], complete), "cut complete source primitives differ"
        )
        _require(
            _same(prep["source_receipts"], binding["source_receipts"])
            and _same(result["source_receipts"], binding["source_receipts"]),
            "cut source receipts differ",
        )
        _require(
            prep["persistent_source_unchanged"] is True
            and _box(prep["initial"], 40) == initial,
            "cut primitive preparation differs",
        )
        _require(
            report["status"] == "completed"
            and report["reason"] is None
            and _q(report["certified_time"]) == HORIZON,
            "cut full-window result unavailable",
        )
        _require(
            type(report["cells"]) is tuple and len(report["cells"]) == STEP_COUNT,
            "complete cut observation inventory required",
        )
        records = result["step_records"]
    for name in ("attempted_steps", "accepted_steps"):
        _require(
            type(report[name]) is int and report[name] == STEP_COUNT,
            "complete producer cell count required",
        )
    _require(
        type(records) is tuple and len(records) == STEP_COUNT,
        "all original step records required",
    )
    return result, records, anchors


def _step_receipts(item, index, branch):
    if branch == "connected":
        _require(
            type(item) is dict and set(item) == {"artifact", "member"},
            "connected archive envelope differs",
        )
        archive = _receipt(item["artifact"], limit=MAX_ARCHIVE_BYTES)
        member = _receipt(item["member"], limit=MAX_MEMBER_BYTES)
    else:
        _require(
            type(item) is dict
            and set(item)
            == {
                "index",
                "time",
                "accepted",
                "archive",
                "member",
                "expanded_bytes",
                "expanded_sha256",
            },
            "cut archive envelope differs",
        )
        _require(
            type(item["index"]) is int
            and item["index"] == index
            and _q(item["time"]) == index * STEP
            and item["accepted"] is True,
            "cut archive chronology differs",
        )
        archive = _receipt(item["archive"], limit=MAX_ARCHIVE_BYTES)
        member = _receipt(
            {
                "path": item["member"],
                "bytes": item["expanded_bytes"],
                "sha256": item["expanded_sha256"],
            },
            limit=MAX_MEMBER_BYTES,
        )
    _require(
        archive.path == f"steps/{index:03d}.zip"
        and member.path == f"step-{index:03d}.json",
        "archive/member sequence differs",
    )
    return archive, member


def _read_step(folder, archive_receipt, member_receipt):
    raw = read_bytes_bounded(
        _path(folder, archive_receipt.path), max_bytes=MAX_ARCHIVE_BYTES
    )
    _require(
        len(raw) == archive_receipt.bytes
        and sha256_bytes(raw) == archive_receipt.sha256,
        "step archive receipt differs",
    )
    with ZipFile(BytesIO(raw)) as archive:
        entries = archive.infolist()
        _require(
            len(entries) == 1
            and entries[0].filename == member_receipt.path
            and entries[0].file_size == member_receipt.bytes,
            "bounded archive inventory or expanded size differs",
        )
    verify_archive_members(
        BytesIO(raw),
        {member_receipt.path: member_receipt.sha256},
        max_bytes=MAX_MEMBER_BYTES,
    )
    with ZipFile(BytesIO(raw)) as archive:
        with archive.open(member_receipt.path) as stream:
            data = stream.read(MAX_MEMBER_BYTES + 1)
    _require(
        len(data) == member_receipt.bytes
        and sha256_bytes(data) == member_receipt.sha256,
        "member byte association differs",
    )
    return _json(data)


@dataclass(frozen=True, eq=False)
class PhaseBranchEvidence:
    """Completed passive checks under explicitly external producer premises."""

    branch: str
    node_order: tuple
    initial_box: tuple[I, ...]
    endpoint: tuple[I, ...]
    steps: tuple[PhaseEvidenceStep, ...]
    phase_cells: tuple
    receipts: tuple[PhaseReceipt, ...]
    source_binding_sha256: str
    compressed_bytes: int
    expanded_bytes: int
    producer_premises: tuple[str, ...]


def require_phase_evidence(value):
    """Reject copied/forged reports; this does not authenticate outer premises."""
    _require(
        type(value) is PhaseBranchEvidence and _CAPABILITIES.get(id(value)) is value,
        "a live completed passive evidence capability is required",
    )
    return value


def load_phase_branch(
    evidence_dir,
    *,
    branch,
    node_order,
    initial_box,
    source_binding,
    result_receipt,
    producer_receipts,
    source_receipt=None,
):
    """Passively admit all 256 original cells, never returning a partial claim.

    ``source_binding`` contains independently rebuilt ``complete_source``
    (the full CutSeparatorSource projection), ``source_receipts`` and the
    fresh persistent ``source_graph_signature``. The outer owner establishes
    freshness and producer history. This reader verifies matching saved
    primitives, supplied anchor bytes and all original archived cells again.
    Metadata/step input failures raise; no missing cell is silently dropped.
    Archives are bounded and processed one at a time without extraction.
    """
    size = _size(branch)
    initial = _initial(initial_box, size, point=True)
    binding = _binding(source_binding, node_order, initial, branch)
    folder = Path(evidence_dir).resolve()
    try:
        result, records, anchors = _metadata(
            folder,
            branch,
            binding,
            initial,
            result_receipt,
            producer_receipts,
            source_receipt,
        )
    except (KeyError, IndexError, AttributeError) as error:
        raise ValueError("producer metadata omits a required association") from error
    pairs = tuple(
        _step_receipts(item, index, branch) for index, item in enumerate(records)
    )
    compressed = sum(archive.bytes for archive, _ in pairs)
    expanded = sum(member.bytes for _, member in pairs)
    _require(
        compressed <= MAX_ARCHIVES_BYTES and expanded <= MAX_EXPANDED_BYTES,
        "complete archive expansion budget exceeded",
    )
    actual = tuple(islice((folder / "steps").iterdir(), STEP_COUNT + 1))
    _require(
        len(actual) == STEP_COUNT
        and {path.name for path in actual}
        == {f"{i:03d}.zip" for i in range(STEP_COUNT)},
        "extra or missing step archive",
    )
    if branch == "connected":
        _require(
            not (folder / "incomplete.json").exists(),
            "connected attempt marked incomplete",
        )
    carried, steps = initial, []
    for index, (archive, member) in enumerate(pairs):
        record = _read_step(folder, archive, member)
        step = _audit_step(record, index=index, initial=carried, branch=branch)
        if branch == "cut_host":
            _require(
                _same(record["observation"], result["report"]["cells"][index]),
                "cut cell observation differs from its recorded result",
            )
        steps.append(step)
        carried = step.endpoint
    receipts = tuple(
        sorted(
            anchors + tuple(archive for archive, _ in pairs), key=lambda row: row.path
        )
    )
    binding_digest = sha256_bytes(
        json_dumps(encode_exact_tree(binding), sort_keys=True).encode("utf-8")
    )
    premises = (
        "outer_fresh_source_and_closed_producer_inventory_admission_required",
        "byte_consistency_is_not_historical_or_derivative_authentication",
    ) + steps[0].producer_premises
    admitted = PhaseBranchEvidence(
        branch,
        node_order,
        initial,
        carried,
        tuple(steps),
        tuple(tuple((value.lo, value.hi) for value in step.phases) for step in steps),
        receipts,
        binding_digest,
        compressed,
        expanded,
        premises,
    )
    _CAPABILITIES[id(admitted)] = admitted
    return admitted
