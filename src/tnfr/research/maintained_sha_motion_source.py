"""Re-admit the committed SHA endpoint before a separate continuation proof.

The reader rebuilds the default SHA reset and its nineteen endpoint checks
from independently admitted UM primitives. It executes no event, pressure
refresh or trajectory. Receipt consistency associates bytes; it authenticates
neither acquisition chronology nor an autonomous event selector. A later
continuation must admit the capacity-changed complete law separately.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from ..operators.grammar_execution import ValidatedSequence
from ..utils.io import json_loads
from .artifact_io import (
    decode_exact_tree,
    file_receipt,
    read_bytes_bounded,
    sha256_bytes,
    verify_archive_members,
)
from .local_capacity_history import PRESSURE_ERROR_BUDGET
from .maintained_parent_sha import predict_parent_sha
from .maintained_parent_sha_execution import EVENT_TIMESTAMP, check_sha_endpoint
from .maintained_parent_sha_source import (
    UmSuccessorAdmission,
    _inventory,
    _require,
    _same,
    load_um_successor,
)
from .maintained_parent_thol import _check_receipt, _contained, _signature
from .maintained_parent_um import NUMERICAL_ALLOWANCE
from .retained_graph_state import restore_recorded_graph_state

_MAX_BYTES = 32 * 1024**2
_PRIOR_MAX_BYTES = 128 * 1024**2
_FROZEN_NAMES = frozenset(
    {"protocol.json", "source.json", "source-manifest.json", "source.zip"}
)


@dataclass(frozen=True)
class ShaSuccessorAdmission:
    """Fresh endpoint, rebuilt event premises and file-backed source locations.

    The graph is mutable. Consumers must use ``validate_live_sha_successor``
    before deriving a new result from an admission retained across mutations;
    its cached prediction and assessment never supply new proof premises.
    """

    graph: object
    before: object
    prediction: object
    assessment: object
    sequence: ValidatedSequence
    receipts: tuple[dict, ...] = ()
    evidence_dir: Path | None = None
    repository_root: Path | None = None


def admit_sha_successor(response, *, um_source, retained_assessment):
    """Rebuild one retained SHA response against freshly associated UM state.

    File-backed UM re-admission is performed by the shared SHA prediction and
    endpoint checker. This lower-level reader does not itself associate the
    response or report with files, and does not predict a later trajectory.
    """
    _require(type(um_source) is UmSuccessorAdmission, "UM admission required")
    _require(
        type(response) is dict
        and response.get("schema") == "tnfr.maintained-parent-sha-response.v1",
        "unsupported committed SHA response",
    )
    before, raw, endpoint = (
        restore_recorded_graph_state(response[key]).graph
        for key in ("before", "raw_after", "endpoint")
    )
    _require(
        _signature(before) == _signature(um_source.graph),
        "SHA source differs from the independently admitted UM endpoint",
    )
    signatures = tuple(map(_signature, (before, raw, endpoint)))
    prediction = predict_parent_sha(um_source)
    assessment = check_sha_endpoint(
        um_source, raw, endpoint, prediction=prediction, stage=response["stage"]
    )
    _require(
        len(assessment["checks"]) == 19
        and all(value is True for value in assessment["checks"].values())
        and assessment["all_checks_passed"] is True
        and assessment["complete_word_executed"] is True,
        "committed SHA endpoint failed independently rebuilt admission",
    )
    _require(
        _same(response["prediction"], prediction),
        "SHA response prediction differs from the before-state rebuild",
    )
    _require(
        _same(retained_assessment, assessment),
        "retained SHA assessment differs from its primitive rebuild",
    )
    _require(
        _same(response["retained_source_receipts"], um_source.receipts),
        "SHA response source receipts differ",
    )
    _require(
        tuple(map(_signature, (before, raw, endpoint))) == signatures,
        "read-only SHA admission changed persistent state",
    )
    return ShaSuccessorAdmission(
        endpoint, before, prediction, assessment, um_source.sequence
    )


def _validate_protocol(protocol, prediction):
    _require(
        type(protocol) is dict
        and protocol.get("schema") == "tnfr.maintained-parent-sha-protocol.v1"
        and protocol.get("evaluation_status_at_freeze") == "not_evaluated",
        "unsupported SHA protocol",
    )
    expected = {
        "target_order": (prediction.nodes[prediction.parent_index],),
        "seed": 17,
        "word_context": ("VAL", "THOL", "UM", "SHA"),
        "step_index": 3,
        "factor": prediction.factor,
        "structural_time": prediction.structural_time,
        "event_metadata_timestamp": EVENT_TIMESTAMP,
        "maximum_events": 1,
        "maximum_attempts": 1,
        "maximum_trajectories": 0,
        "capacity_error_allowance": NUMERICAL_ALLOWANCE,
        "pressure_refresh_error_allowance": PRESSURE_ERROR_BUDGET,
    }
    _require(
        all(_same(protocol.get(key), value) for key, value in expected.items()),
        "SHA protocol differs from the admitted event and numerical budget",
    )


def load_sha_successor(evidence_dir, *, repository_root):
    """Verify SHA evidence and rebuild the actual committed complete-word state.

    Archived sources are verified without extraction or execution. The current
    reader may contain compatible changes; every later study freezes its own
    consumer. Earlier motion evidence is not transferred through this event.
    """
    root, folder = Path(repository_root).resolve(), Path(evidence_dir).resolve()
    raws, records, receipts = {}, {}, {}

    def raw(name):
        if name not in raws:
            raws[name] = read_bytes_bounded(
                _contained(folder, name), max_bytes=_MAX_BYTES
            )
            receipts[name] = {
                "path": name,
                "bytes": len(raws[name]),
                "sha256": sha256_bytes(raws[name]),
            }
        return raws[name]

    def read(name):
        if name not in records:
            records[name] = decode_exact_tree(json_loads(raw(name)))
        return records[name]

    def verify(row, *, expected=None):
        name = row["path"]
        _require(
            expected is None or name == expected,
            "artifact role differs from its declared association",
        )
        raw(name)
        _check_receipt(receipts[name], row)

    freeze = read("freeze.json")
    _require(
        freeze.get("schema") == "tnfr.maintained-parent-sha-freeze.v1"
        and freeze.get("evaluation_status_at_freeze") == "not_evaluated",
        "unsupported SHA freeze",
    )
    _inventory(freeze["artifacts"], _FROZEN_NAMES, "frozen SHA artifacts")
    for row in freeze["artifacts"]:
        verify(row)
    for row in freeze["prior_artifacts"]:
        _check_receipt(
            file_receipt(
                _contained(root, row["path"]),
                label=row["path"],
                max_bytes=_PRIOR_MAX_BYTES,
            ),
            row,
        )
    manifest = read("source-manifest.json")
    _require(
        manifest.get("schema") == "tnfr.source-manifest.v1", "unsupported manifest"
    )
    members = manifest["files"]
    _inventory(members, {row["path"] for row in members}, "SHA source archive")
    archive_data = raw("source.zip")
    verify_archive_members(
        BytesIO(archive_data),
        {row["path"]: row["sha256"] for row in members},
        max_bytes=_MAX_BYTES,
    )
    with ZipFile(BytesIO(archive_data)) as archive:
        _require(
            _same(
                {item.filename: item.file_size for item in archive.infolist()},
                {row["path"]: row["bytes"] for row in members},
            ),
            "SHA archive member sizes differ",
        )
    source, protocol = read("source.json"), read("protocol.json")
    _require(
        source.get("schema") == "tnfr.maintained-parent-sha-source.v1",
        "unsupported SHA source",
    )
    attempt, report = read("attempt.json"), read("report.json")
    _require(
        attempt.get("schema") == "tnfr.maintained-parent-sha-attempt.v1"
        and attempt.get("evaluation_status_at_start") == "not_evaluated"
        and type(attempt.get("attempt_number")) is int
        and attempt["attempt_number"] == 1
        and type(attempt.get("seed")) is int
        and attempt["seed"] == 17,
        "unsupported SHA attempt",
    )
    verify(attempt["freeze"], expected="freeze.json")
    _require(
        report.get("schema") == "tnfr.maintained-parent-sha-report.v1",
        "unsupported SHA report",
    )
    for key, name in (
        ("attempt", "attempt.json"),
        ("response", "response.json"),
        ("raw_after", "raw-after.json"),
    ):
        verify(report[key], expected=name)
    response = read("response.json")
    _require(
        _same(response["before"], source["before"])
        and _same(response["raw_after"], read("raw-after.json")),
        "SHA primitive response binding differs",
    )
    um = load_um_successor(
        root / "docs/assets/maintained_parent_um_corrected", repository_root=root
    )
    admitted = admit_sha_successor(
        response, um_source=um, retained_assessment=report["assessment"]
    )
    _require(
        _same(source["prediction"], admitted.prediction)
        and _same(source["retained_source_receipts"], um.receipts),
        "frozen SHA source differs from its independent rebuild",
    )
    _validate_protocol(protocol, admitted.prediction)
    return replace(
        admitted,
        receipts=tuple(receipts.values()),
        evidence_dir=folder,
        repository_root=root,
    )


def validate_live_sha_successor(admitted):
    """Re-admit original primitives and reject a changed mutable live endpoint.

    Cached predictions, reports, grammar contexts and receipt arrays are not
    used to admit the source. The fresh returned graph remains independent of
    the caller's graph, which is read only.
    """
    _require(
        type(admitted) is ShaSuccessorAdmission
        and admitted.evidence_dir is not None
        and admitted.repository_root is not None,
        "file-backed committed SHA admission required",
    )
    fresh = load_sha_successor(
        admitted.evidence_dir, repository_root=admitted.repository_root
    )
    _require(
        _signature(admitted.graph) == _signature(fresh.graph),
        "live SHA endpoint differs from its committed source",
    )
    return fresh
