"""Re-admit retained post-SHA cells without regenerating a trajectory.

Derivative jets and remainders retain their frozen producer provenance. This
reader rebuilds the actual changed-capacity law, checks complete Picard tubes,
reconstructs recorded Taylor arithmetic and derives the admitted prefix. Byte
associations authenticate neither acquisition nor chronology. New observations
of these cells are retrospective and are not the frozen producer's readouts.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from io import BytesIO
from math import isfinite
from pathlib import Path
from zipfile import ZipFile

from ..mathematics._rational_interval import I
from ..mathematics._validated_taylor import reconstruct_box_taylor_arithmetic
from ..utils.io import json_loads
from .artifact_io import (
    decode_exact_tree,
    exact_record,
    file_receipt,
    read_bytes_bounded,
    sha256_bytes,
    validate_artifact_path,
    verify_archive_members,
)
from .local_capacity_history import TIMES
from .maintained_parent_sha_source import _same
from .maintained_parent_thol import _signature
from .maintained_principal_retention import (
    HORIZON,
    ORDER,
    STEP,
    STEP_COUNT,
    WALL_BUDGET_SECONDS,
    principal_margins,
)
from .maintained_sha_motion import prepare_sha_motion
from .maintained_sha_motion_source import load_sha_successor
from .retained_graph_state import restore_recorded_graph_state

_MAX_RECORD_BYTES = 32 * 1024**2
_MAX_ARCHIVE_BYTES = 128 * 1024**2
_FROZEN_NAMES = frozenset(
    {"protocol.json", "source.json", "source-manifest.json", "source.zip"}
)


@dataclass(frozen=True)
class RetainedShaStep:
    """One accepted chronological cell with independently rebuilt margins.

    ``tube`` contains its whole-time state; ``initial`` and ``endpoint`` retain
    all 42 coordinates. Taylor coefficients and remainders remain associated
    with the frozen producer rather than independently rederived here.
    """

    index: int
    time: Q
    duration: Q
    initial: tuple[I, ...]
    tube: tuple[I, ...]
    endpoint: tuple[I, ...]
    principal_margins: tuple[Q, ...]
    combined_principal_margins: tuple[Q, ...]


@dataclass(frozen=True)
class ShaRetentionEvidence:
    """Fresh source, admitted prefix and retained byte associations.

    Source graphs, documents and record dictionaries remain mutable inspection
    objects. Consumers reload this reader before a new report-consuming
    calculation; their cached summaries and predictions never supply premises.
    ``steps`` contains only the accepted prefix, including a complete 256-cell
    sequence for the retained successful attempt. No new samples are acquired.
    """

    admitted: object
    prediction: object
    initial: tuple[I, ...]
    steps: tuple[RetainedShaStep, ...]
    summary: dict
    receipts: tuple[dict, ...]
    records: tuple[dict, ...]
    documents: dict
    flow: object
    evidence_dir: Path
    repository_root: Path


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _interval(value):
    _require(type(value) is dict and set(value) == {"lo", "hi"}, "interval record")
    lo, hi = exact_record(value["lo"]), exact_record(value["hi"])
    interval = I(lo, hi)
    _require(interval.lo == lo and interval.hi == hi, "noncanonical interval endpoints")
    return interval


def _box(values):
    _require(type(values) is tuple and len(values) == 42, "complete state required")
    return tuple(_interval(value) for value in values)


def _receipt_fields(record):
    _require(
        type(record) is dict
        and set(record) == {"path", "bytes", "sha256"}
        and type(record["bytes"]) is int
        and record["bytes"] >= 0,
        "receipt representation",
    )
    name = validate_artifact_path(record["path"])
    digest = record["sha256"]
    _require(
        type(digest) is str
        and len(digest) == 64
        and all(char in "0123456789abcdef" for char in digest),
        "receipt digest representation",
    )
    return name


def _receipt(base, record):
    name = _receipt_fields(record)
    path = (base / name).resolve()
    _require(path.is_relative_to(base.resolve()), "receipt leaves root")
    _require(
        file_receipt(path, label=name, max_bytes=_MAX_ARCHIVE_BYTES) == record,
        "artifact receipt differs",
    )


def _audit_step(record, *, index, initial, prediction, flow):
    """Admit one cell without regenerating its derivative jets or remainder."""
    _require(type(record["index"]) is int and record["index"] == index, "step order")
    time = index * STEP
    _require(exact_record(record["time"]) == time, "step chronology")
    step = record["step"]
    if step is None:
        _require(type(record["reason"]) is str and record["reason"], "failure reason")
        if record["failed_tube"] is not None:
            tube = _box(record["failed_tube"])
            _require(
                all(x.subset_of(y) for x, y in zip(initial, tube, strict=True)),
                "failed tube omits the source",
            )
        return None, "enclosure_unavailable"

    _require(exact_record(step["time"]) == time, "Taylor chronology")
    _require(exact_record(step["duration"]) == STEP, "Taylor duration")
    _require(type(step["order"]) is int and step["order"] == ORDER, "Taylor order")
    _require(
        step["method"] == "direct_source_box_Picard_Taylor_dyadic128_v1",
        "Taylor method",
    )
    source = _box(step["initial_box"])
    _require(source == initial, "endpoint carry differs")
    tube = _box(step["tube"])
    _require(
        tuple(map(exact_record, step["domain_lower_bounds"])) == (Q(1),),
        "complete smooth domain",
    )
    image = tuple(
        x + rate * I(0, STEP) for x, rate in zip(source, flow(tube), strict=True)
    )
    interior = min(
        min(value.lo - bound.lo, bound.hi - value.hi)
        for value, bound in zip(image, tube, strict=True)
    )
    _require(interior > 0, "strict Picard inclusion")
    _require(exact_record(step["picard_interior_margin"]) == interior, "Picard margin")
    series = tuple(tuple(map(_interval, row)) for row in step["series"])
    remainder = _box(step["local_remainder_bounds"])
    increment, endpoint = reconstruct_box_taylor_arithmetic(
        source, tube, series, remainder, STEP, order=ORDER
    )
    _require(increment == _box(step["increment"]), "Taylor increment")
    _require(endpoint == _box(step["endpoint"]), "Taylor endpoint")
    margins = principal_margins(tube, prediction.edges, prediction.shifts)
    _require(
        tuple(map(exact_record, record["tube_principal_margins"])) == margins,
        "cached tube margins",
    )
    combined = tuple(
        max(tube_margin, energy.lo)
        for tube_margin, energy in zip(
            margins, prediction.bounds.horizon_remaining_margins, strict=True
        )
    )
    _require(
        tuple(map(exact_record, record["combined_principal_margins"])) == combined,
        "cached combined margins",
    )
    admitted = RetainedShaStep(
        index, time, STEP, source, tube, endpoint, margins, combined
    )
    return admitted, None if min(combined) > 0 else "principal_bound_unresolved"


def _check_summary(report, expected):
    _require(
        type(report) is dict and set(report) == set(expected) | {"elapsed_seconds"},
        "retention-only summary fields",
    )
    normalized = dict(report)
    for key in ("certified_retention_time", "minimum_retained_margin"):
        normalized[key] = exact_record(report[key])
    for key, value in expected.items():
        if type(value) in (bool, int):
            _require(type(report[key]) is type(value), f"summary type: {key}")
        _require(normalized[key] == value, f"summary differs: {key}")


def _audit_summary(records, report, *, initial, prediction, flow):
    minimum = min(principal_margins(initial, prediction.edges, prediction.shifts))
    _require(minimum > 0, "initial identity")
    _require(len(records) <= STEP_COUNT, "step budget")
    steps, terminal, carried = [], None, initial
    for index, record in enumerate(records):
        _require(terminal is None, "records after terminal result")
        step, terminal = _audit_step(
            record, index=index, initial=carried, prediction=prediction, flow=flow
        )
        if terminal is None:
            minimum = min(minimum, min(step.combined_principal_margins))
            steps.append(step)
            carried = step.endpoint
    elapsed = report["elapsed_seconds"]
    _require(
        type(elapsed) in (int, float) and isfinite(elapsed) and elapsed >= 0,
        "elapsed clock",
    )
    if terminal is None:
        terminal = (
            "completed" if len(steps) == STEP_COUNT else "resource_budget_exhausted"
        )
        if len(steps) != STEP_COUNT:
            _require(elapsed >= WALL_BUDGET_SECONDS, "resource stop chronology")
    expected = dict(
        status=terminal,
        attempted_steps=len(records),
        accepted_steps=len(steps),
        certified_retention_time=len(steps) * STEP,
        minimum_retained_margin=minimum,
        full_horizon_retention=len(steps) == STEP_COUNT,
    )
    _check_summary(report, expected)
    return expected, tuple(steps)


def _validate_protocol(protocol, prediction):
    expected = {
        "schema": "tnfr.maintained-sha-retention-protocol.v1",
        "evaluation_status_at_freeze": "not_evaluated",
        "node_order": prediction.nodes,
        "law": "x_dot=M*S(theta); theta_dot=M*L_W*x; M=diag(nu/(pi*d))",
        "prior_clock": TIMES[-1],
        "horizon": HORIZON,
        "step": STEP,
        "maximum_steps": STEP_COUNT,
        "order": ORDER,
        "maximum_attempts": 1,
        "maximum_events": 0,
        "wall_budget_seconds_between_steps": WALL_BUDGET_SECONDS,
        "seed": None,
    }
    _require(
        type(protocol) is dict
        and all(
            key in protocol and _same(protocol[key], value)
            for key, value in expected.items()
        ),
        "retention protocol differs",
    )
    _require(prediction.horizon == HORIZON == STEP_COUNT * STEP, "retention horizon")


def load_sha_retention(evidence_dir, *, repository_root):
    """Read and re-admit one retained attempt, without producer or jet replay.

    All source associations, complete-law primitives, chronological carry,
    Picard inclusions, recorded Taylor arithmetic and original branches are
    rebuilt before any cell is returned. A partial or failed retained attempt
    remains a partial or failed prefix, never an inferred branch crossing.
    """
    folder, root = Path(evidence_dir).resolve(), Path(repository_root).resolve()
    raw_files, receipts = {}, {}

    def raw(name):
        name = validate_artifact_path(name)
        if name not in raw_files:
            path = (folder / name).resolve()
            _require(path.is_relative_to(folder), "artifact leaves evidence root")
            data = read_bytes_bounded(path, max_bytes=_MAX_ARCHIVE_BYTES)
            raw_files[name] = data
            receipts[name] = {
                "path": name,
                "bytes": len(data),
                "sha256": sha256_bytes(data),
            }
        return raw_files[name]

    def read(name):
        data = raw(name)
        _require(len(data) <= _MAX_RECORD_BYTES, "record exceeds audit byte budget")
        return decode_exact_tree(json_loads(data))

    def verify(record, *, expected=None):
        name = _receipt_fields(record)
        _require(expected is None or name == expected, "artifact association differs")
        raw(name)
        _require(receipts[name] == record, "artifact receipt differs")

    names = ("result", "freeze", "source-manifest", "protocol", "attempt", "source")
    documents = {name: read(f"{name}.json") for name in names}
    frozen, manifest = documents["freeze"], documents["source-manifest"]
    _require(
        frozen["schema"] == "tnfr.maintained-sha-retention-freeze.v1"
        and frozen["evaluation_status_at_freeze"] == "not_evaluated",
        "retention freeze schema",
    )
    artifacts = frozen["artifacts"]
    _require(
        type(artifacts) is tuple
        and len(artifacts) == len(_FROZEN_NAMES)
        and {_receipt_fields(row) for row in artifacts} == _FROZEN_NAMES,
        "frozen artifact inventory",
    )
    for row in artifacts:
        verify(row)
    prior = frozen["prior_artifacts"]
    _require(type(prior) is tuple and prior, "prior artifact inventory")
    _require(
        len(prior) == len({_receipt_fields(row) for row in prior}),
        "duplicate prior artifact",
    )
    for row in prior:
        _receipt(root, row)
    _require(manifest["schema"] == "tnfr.source-manifest.v1", "manifest schema")
    members = manifest["files"]
    _require(type(members) is tuple and members, "source manifest inventory")
    _require(
        len(members) == len({_receipt_fields(row) for row in members}),
        "duplicate source member",
    )
    expected = {row["path"]: row["sha256"] for row in members}
    _require(
        {
            "src/tnfr/research/maintained_sha_motion_source.py",
            "src/tnfr/research/maintained_sha_motion.py",
            "src/tnfr/research/maintained_principal_retention.py",
            "src/tnfr/mathematics/_validated_taylor.py",
            "src/tnfr/physics/_sine_flow.py",
            "instrument.py",
            "pyproject.toml",
        }
        <= expected.keys(),
        "missing frozen implementation owner",
    )
    source_archive = raw("source.zip")
    verify_archive_members(
        BytesIO(source_archive), expected, max_bytes=_MAX_ARCHIVE_BYTES
    )
    with ZipFile(BytesIO(source_archive)) as archive:
        _require(
            _same(
                {item.filename: item.file_size for item in archive.infolist()},
                {row["path"]: row["bytes"] for row in members},
            ),
            "source member byte counts",
        )
    attempt, result = documents["attempt"], documents["result"]
    _require(
        attempt["schema"] == "tnfr.maintained-sha-retention-attempt.v1"
        and attempt["evaluation_status_at_start"] == "not_evaluated"
        and type(attempt["attempt_number"]) is int
        and attempt["attempt_number"] == 1
        and attempt["seed"] is None,
        "retention attempt differs",
    )
    verify(attempt["freeze"], expected="freeze.json")
    _require(
        result["schema"] == "tnfr.maintained-sha-retention-result.v1", "result schema"
    )
    verify(result["attempt"], expected="attempt.json")
    _require(not (folder / "incomplete.json").exists(), "incomplete retained attempt")
    admitted = load_sha_successor(
        root / "docs/assets/maintained_parent_sha", repository_root=root
    )
    signature = _signature(admitted.graph)
    prediction, initial, flow = prepare_sha_motion(admitted)
    source = documents["source"]
    _require(
        source["schema"] == "tnfr.maintained-sha-retention-source.v1", "source schema"
    )
    restored = restore_recorded_graph_state(source["endpoint"]).graph
    _require(_signature(restored) == signature, "retained SHA source differs")
    _require(
        _same(source["retained_source_receipts"], admitted.receipts),
        "SHA receipts differ",
    )
    _require(_same(source["prediction"], prediction), "source prediction differs")
    _require(_same(source["initial_box"], initial), "source initial box differs")
    _require(
        len(initial) == 42 and all(x.lo == x.hi for x in initial),
        "represented source point required",
    )
    _validate_protocol(documents["protocol"], prediction)
    items = result["records"]
    _require(type(items) is tuple and len(items) <= STEP_COUNT, "step budget")
    records = []
    for index, item in enumerate(items):
        artifact, member = item["artifact"], item["member"]
        name, member_name = f"steps/{index:03d}.zip", f"step-{index:03d}.json"
        verify(artifact, expected=name)
        _require(_receipt_fields(member) == member_name, "member sequence")
        archived = raw(name)
        verify_archive_members(
            BytesIO(archived),
            {member_name: member["sha256"]},
            max_bytes=_MAX_RECORD_BYTES,
        )
        with ZipFile(BytesIO(archived)) as archive:
            data = archive.read(member_name)
        _require(len(data) == member["bytes"], "member byte count")
        records.append(decode_exact_tree(json_loads(data)))
    _require(
        {path.name for path in (folder / "steps").iterdir()}
        == {f"{index:03d}.zip" for index in range(len(records))},
        "unreported step artifacts",
    )
    summary, steps = _audit_summary(
        tuple(records),
        result["report"],
        initial=initial,
        prediction=prediction,
        flow=flow,
    )
    _require(_signature(admitted.graph) == signature, "retention reader changed source")
    return ShaRetentionEvidence(
        admitted=admitted,
        prediction=prediction,
        initial=initial,
        steps=steps,
        summary=summary,
        receipts=tuple(receipts[name] for name in sorted(receipts)),
        records=tuple(records),
        documents=documents,
        flow=flow,
        evidence_dir=folder,
        repository_root=root,
    )
