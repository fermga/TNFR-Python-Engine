"""Read the actual committed UM successor before a separate SHA study.

The original 25/27 assessment and its separate corrected reader remain distinct
historical records. Admission rebuilds the response from primitive graphs; it
executes no producer, event, pressure refresh or trajectory. Byte associations
do not authenticate acquisition, and this reader makes no SHA eligibility or
post-SHA continuation claim.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from ..operators.grammar_execution import ValidatedSequence
from ..utils.io import json_dumps, json_loads
from .artifact_io import (
    decode_exact_tree,
    encode_exact_tree,
    file_receipt,
    read_bytes_bounded,
    sha256_bytes,
    verify_archive_members,
)
from .local_capacity_history import encode_result
from .maintained_parent_thol import _check_receipt, _contained, _signature
from .maintained_parent_um_execution import check_um_endpoint
from .maintained_parent_um_source import TholSuccessorAdmission, load_thol_successor
from .maintained_parent_um_work import predict_parent_um_work
from .maintained_successor_motion import predict_successor_motion
from .retained_graph_state import restore_recorded_graph_state

_MAX_BYTES = 32 * 1024**2
_PRIOR_MAX_BYTES = 128 * 1024**2
_FROZEN_NAMES = frozenset(
    {"protocol.json", "source.json", "source-manifest.json", "source.zip"}
)
_REVIEW_INPUTS = _FROZEN_NAMES | {
    "freeze.json",
    "attempt.json",
    "raw-after.json",
    "response.json",
    "report.json",
}
_REVIEW_MEMBERS = frozenset(
    {
        "src/tnfr/research/maintained_parent_um_execution.py",
        "tests/research/test_maintained_parent_um_execution.py",
        "review.py",
    }
)
_CORRECTED_CHECKS = frozenset(
    {"old_edge_form_work_zero", "declared_graph_context_preserved"}
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _same(left, right):
    """Compare projected records without conflating Boolean/integer scalars."""
    return json_dumps(
        encode_exact_tree(encode_result(left)), sort_keys=True
    ) == json_dumps(encode_exact_tree(encode_result(right)), sort_keys=True)


def _inventory(rows, expected, label):
    _require(
        type(rows) is tuple
        and all(type(row) is dict for row in rows)
        and len(rows) == len(expected)
        and {row.get("path") for row in rows} == set(expected),
        f"{label} inventory differs",
    )


@dataclass(frozen=True)
class UmSuccessorAdmission:
    """Fresh mutable endpoint, rebuilt premises and immutable-artifact locations.

    A later consumer must call ``validate_live_um_successor`` before using a
    potentially changed graph. The locations permit re-admission from primitive
    evidence instead of trusting cached derived fields or a saved signature.
    """

    graph: object
    before: object
    prediction: object
    assessment: object
    sequence: ValidatedSequence
    receipts: tuple[dict, ...] = ()
    evidence_dir: Path | None = None
    repository_root: Path | None = None


def admit_um_successor(response, *, thol_source, review_assessment):
    """Rebuild a committed response associated with an admitted THOL source.

    The caller owns artifact association and fresh THOL history admission.
    Saved predictions and verdicts are compared only after primitive rebuilding.
    This lower-level reader does not provide file-backed live re-admission.
    """
    _require(type(thol_source) is TholSuccessorAdmission, "THOL admission required")
    _require(
        type(response) is dict
        and response.get("schema") == "tnfr.maintained-parent-um-response.v1",
        "unsupported committed UM response",
    )
    before, raw, endpoint = (
        restore_recorded_graph_state(response[key]).graph
        for key in ("before", "raw_after", "endpoint")
    )
    _require(
        _signature(before) == _signature(thol_source.graph),
        "UM source differs from the independently admitted THOL history",
    )
    signatures = tuple(map(_signature, (before, raw, endpoint)))
    prediction = predict_successor_motion(before)
    assessment = check_um_endpoint(
        before, raw, endpoint, prediction=prediction, stage=response["stage"]
    )
    _require(
        len(assessment["checks"]) == 27
        and all(value is True for value in assessment["checks"].values())
        and assessment["all_checks_passed"] is True,
        "committed UM endpoint failed independently rebuilt admission",
    )
    _require(
        _same(response["prediction"], prediction),
        "response prediction differs from the before-state rebuild",
    )
    _require(
        _same(review_assessment, assessment),
        "retained review differs from its primitive rebuild",
    )
    _require(
        _same(response["retained_source_receipts"], thol_source.receipts),
        "UM response source receipts differ",
    )
    _require(
        tuple(map(_signature, (before, raw, endpoint))) == signatures,
        "read-only UM admission changed persistent state",
    )
    return UmSuccessorAdmission(
        endpoint, before, prediction, assessment, thol_source.sequence
    )


def _historical_report(original, rebuilt):
    """Preserve the two original reader failures without treating them as laws."""
    _require(
        type(original) is dict
        and set(original) == set(rebuilt)
        and set(original["checks"]) == set(rebuilt["checks"])
        and all(type(value) is bool for value in original["checks"].values())
        and {key for key, value in original["checks"].items() if not value}
        == _CORRECTED_CHECKS
        and original["all_checks_passed"] is False,
        "original UM assessment must retain its historical 25/27 result",
    )
    revised = {"old_edge_form_work", "actual_total_work", "checks", "all_checks_passed"}
    _require(
        all(
            _same(value, rebuilt[key])
            for key, value in original.items()
            if key not in revised
        ),
        "review changed an unaffected assessment field",
    )


def load_um_successor(evidence_dir, *, repository_root):
    """Verify immutable inputs and rebuild the actual committed UM endpoint.

    Archived producer and review sources are verified without extraction or
    execution. Current consuming code can differ compatibly from those sources;
    a subsequent study must freeze its own consumer. The inherited motion
    certificate is neither regenerated nor extended through SHA here.
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
        if expected is not None:
            _require(
                name == expected, "artifact role differs from its declared association"
            )
        raw(name)
        _check_receipt(receipts[name], row)

    def verify_archive(name, rows):
        names = {row["path"] for row in rows}
        _inventory(rows, names, "source archive")
        data = raw(name)
        verify_archive_members(
            BytesIO(data),
            {row["path"]: row["sha256"] for row in rows},
            max_bytes=_MAX_BYTES,
        )
        with ZipFile(BytesIO(data)) as archive:
            _require(
                _same(
                    {item.filename: item.file_size for item in archive.infolist()},
                    {row["path"]: row["bytes"] for row in rows},
                ),
                "archive member sizes differ",
            )

    freeze = read("freeze.json")
    _require(
        freeze.get("schema") == "tnfr.maintained-parent-um-corrected-freeze.v1"
        and freeze.get("evaluation_status_at_freeze") == "not_evaluated",
        "unsupported corrected UM freeze",
    )
    _inventory(freeze["artifacts"], _FROZEN_NAMES, "frozen artifacts")
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
    verify_archive("source.zip", read("source-manifest.json")["files"])
    protocol, source = read("protocol.json"), read("source.json")
    _require(
        protocol.get("schema") == "tnfr.maintained-parent-um-corrected-protocol.v1",
        "unsupported UM protocol",
    )
    _require(
        source.get("schema") == "tnfr.maintained-parent-um-corrected-source.v1",
        "unsupported UM source",
    )
    attempt, report = read("attempt.json"), read("report.json")
    _require(
        attempt.get("schema") == "tnfr.maintained-parent-um-corrected-attempt.v1"
        and attempt.get("evaluation_status_at_start") == "not_evaluated"
        and type(attempt.get("attempt_number")) is int
        and attempt["attempt_number"] == 1
        and type(attempt.get("seed")) is int
        and attempt["seed"] == 17,
        "unsupported UM attempt",
    )
    verify(attempt["freeze"], expected="freeze.json")
    _require(
        report.get("schema") == "tnfr.maintained-parent-um-corrected-report.v1",
        "unsupported original UM report",
    )
    for key, name in (
        ("attempt", "attempt.json"),
        ("response", "response.json"),
        ("raw_after", "raw-after.json"),
    ):
        verify(report[key], expected=name)
    review_protocol, review = read("review-protocol.json"), read("review.json")
    _require(
        review_protocol.get("schema")
        == "tnfr.maintained-parent-um-assessment-review-protocol.v1"
        and review_protocol.get("evaluation_status_at_freeze") == "not_evaluated"
        and review.get("schema") == "tnfr.maintained-parent-um-assessment-review.v1",
        "unsupported separate UM review",
    )
    _inventory(review_protocol["inputs"], _REVIEW_INPUTS, "review inputs")
    for row in review_protocol["inputs"]:
        verify(row)
    verify(review_protocol["source_archive"], expected="review-source.zip")
    _inventory(review_protocol["members"], _REVIEW_MEMBERS, "review source")
    verify_archive("review-source.zip", review_protocol["members"])
    for row in review_protocol["members"]:
        if row["path"] == "review.py":
            verify(row, expected="review.py")
    for key, name in (
        ("protocol", "review-protocol.json"),
        ("original_report", "report.json"),
        ("response", "response.json"),
    ):
        verify(review[key], expected=name)
    _require(
        all(
            type(review_protocol[key]) is int and review_protocol[key] == 0
            for key in ("maximum_events", "maximum_trajectories")
        )
        and all(
            type(review[key]) is int and review[key] == 0
            for key in ("new_events_executed", "new_trajectories_executed")
        )
        and review_protocol["scientific_protocol_changed"] is False
        and review["original_report_preserved"] is True,
        "review must preserve the original response without new execution",
    )
    response = read("response.json")
    _require(
        _same(response["before"], source["before"])
        and _same(response["raw_after"], read("raw-after.json")),
        "UM primitive response binding differs",
    )
    thol = load_thol_successor(
        root / "docs/assets/maintained_parent_thol", repository_root=root
    )
    admitted = admit_um_successor(
        response, thol_source=thol, review_assessment=review["assessment"]
    )
    _require(
        _same(source["motion_prediction"], admitted.prediction)
        and _same(source["work_prediction"], predict_parent_um_work(admitted.before))
        and _same(source["retained_source_receipts"], thol.receipts),
        "frozen UM source differs from its independent rebuild",
    )
    for key in ("existing_motion_receipts", "correction_receipts"):
        for row in source[key]:
            _check_receipt(
                file_receipt(
                    _contained(root, row["path"]),
                    label=row["path"],
                    max_bytes=_MAX_BYTES,
                ),
                row,
            )
    _historical_report(report["assessment"], admitted.assessment)
    return replace(
        admitted,
        receipts=tuple(receipts.values()),
        evidence_dir=folder,
        repository_root=root,
    )


def validate_live_um_successor(admitted):
    """Re-admit original primitives and reject a changed mutable live endpoint.

    No cached assessment, prediction, sequence or signature supplies premises.
    Returned coordinates and derived fields come from the associated artifacts;
    the caller's graph is read only and remains its own independent object.
    """
    _require(
        type(admitted) is UmSuccessorAdmission
        and admitted.evidence_dir is not None
        and admitted.repository_root is not None,
        "file-backed committed UM admission required",
    )
    fresh = load_um_successor(
        admitted.evidence_dir, repository_root=admitted.repository_root
    )
    _require(
        _signature(admitted.graph) == _signature(fresh.graph),
        "live UM endpoint differs from its committed source",
    )
    return fresh
