"""Re-admit the fixed retained child cell without repeating its selection.

The complete SHA certificate is loaded once. Only the already selected cell
12 is then rebuilt under its actual law and unchanged policies. The earlier
first-cell result remains inherited evidence, not a newly repeated search.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from .._exact_time import exact_or_represented_real
from ..operators._thol_config import (
    resolve_thol_bifurcation_threshold,
    resolve_thol_min_epi,
)
from ..operators.grammar_dynamics import validate_candidate
from ..utils.io import json_loads
from .artifact_io import (
    decode_exact_tree,
    read_bytes_bounded,
    sha256_bytes,
    verify_archive_members,
)
from .maintained_child_history import CHILD, CHILD_INDEX, PARENT
from .maintained_child_joint_window import ChildJointCell, classify_joint_conditions
from .maintained_parent_sha_source import _same
from .maintained_parent_thol import _signature
from .maintained_sha_retention_evidence import (
    RetainedShaStep,
    ShaRetentionEvidence,
    _receipt,
    _receipt_fields,
    _require,
    load_sha_retention,
)
from .maintained_temporal_window import NodeHistoryBox, bound_node_history_box

CELL_INDEX = 12
CELL_START = Q(3, 128)
CELL_END = Q(13, 512)
_MAX_BYTES = 32 * 1024**2
_MAX_ARCHIVE_BYTES = 128 * 1024**2
_FROZEN_NAMES = frozenset({"protocol.json", "source-manifest.json", "source.zip"})
_RETENTION_PATH = "docs/assets/maintained_sha_retention"


@dataclass(frozen=True)
class ChildJointAdmission:
    """Fresh fixed-cell bounds and byte associations, without acquired state.

    The retained graph remains mutable. A new report-consuming calculation
    reloads this reader rather than accepting a caller's cached admission.
    The returned cell is rebuilt from its whole-time tube; it is neither a
    runtime endpoint nor measured history. Earlier cells are not reclassified,
    so this reader does not independently reprove first-cell selection.
    """

    retention: ShaRetentionEvidence
    step: RetainedShaStep
    bounds: NodeHistoryBox
    cell: ChildJointCell
    minimum_form: Q
    threshold: Q
    receipts: tuple[dict, ...]


def _read_joint_artifacts(folder, root):
    """Verify the six retained observation artifacts without executing sources."""
    raw_files, receipts = {}, {}

    def raw(name):
        if name not in raw_files:
            path = (folder / name).resolve()
            _require(path.is_relative_to(folder), "joint artifact leaves root")
            data = read_bytes_bounded(path, max_bytes=_MAX_ARCHIVE_BYTES)
            raw_files[name] = data
            receipts[name] = {
                "path": name,
                "bytes": len(data),
                "sha256": sha256_bytes(data),
            }
        return raw_files[name]

    def read(name):
        data = raw(f"{name}.json")
        _require(len(data) <= _MAX_BYTES, "joint record exceeds byte budget")
        return decode_exact_tree(json_loads(data))

    def verify(record, expected=None):
        name = _receipt_fields(record)
        _require(expected is None or name == expected, "joint artifact role differs")
        raw(name)
        _require(receipts[name] == record, "joint artifact receipt differs")

    documents = {
        name: read(name)
        for name in ("protocol", "freeze", "attempt", "result", "source-manifest")
    }
    freeze = documents["freeze"]
    _require(
        freeze["schema"] == "tnfr.maintained-child-joint-window-freeze.v1"
        and freeze["observation_status"] == "not_observed_by_this_instrument"
        and freeze["evidence_kind"]
        == "retrospective observation; source trajectory already evaluated",
        "joint freeze declaration differs",
    )
    artifacts = freeze["artifacts"]
    _require(
        type(artifacts) is tuple
        and len(artifacts) == 3
        and {_receipt_fields(row) for row in artifacts} == _FROZEN_NAMES,
        "joint frozen artifact inventory differs",
    )
    for row in artifacts:
        verify(row)
    prior = freeze["prior_artifacts"]
    _require(
        type(prior) is tuple
        and prior
        and len(prior) == len({_receipt_fields(row) for row in prior}),
        "joint prior artifact inventory differs",
    )
    for row in prior:
        _receipt(root, row)
    manifest = documents["source-manifest"]
    _require(manifest["schema"] == "tnfr.source-manifest.v1", "joint manifest schema")
    members = manifest["files"]
    _require(
        type(members) is tuple
        and members
        and len(members) == len({_receipt_fields(row) for row in members}),
        "joint source inventory differs",
    )
    hashes = {row["path"]: row["sha256"] for row in members}
    _require(
        {
            "src/tnfr/research/maintained_child_joint_window.py",
            "src/tnfr/research/maintained_temporal_window.py",
            "src/tnfr/research/maintained_sha_retention_evidence.py",
            "src/tnfr/operators/_thol_config.py",
            "instrument.py",
            "pyproject.toml",
        }
        <= hashes.keys(),
        "joint frozen implementation owner missing",
    )
    archive_bytes = raw("source.zip")
    verify_archive_members(BytesIO(archive_bytes), hashes, max_bytes=_MAX_ARCHIVE_BYTES)
    with ZipFile(BytesIO(archive_bytes)) as archive:
        _require(
            _same(
                {item.filename: item.file_size for item in archive.infolist()},
                {row["path"]: row["bytes"] for row in members},
            ),
            "joint source member byte counts differ",
        )
    attempt, result = documents["attempt"], documents["result"]
    _require(
        attempt["schema"] == "tnfr.maintained-child-joint-window-attempt.v1"
        and type(attempt["attempt_number"]) is int
        and attempt["attempt_number"] == 1
        and attempt["evidence_kind"] == "retrospective observation",
        "joint attempt declaration differs",
    )
    verify(attempt["freeze"], "freeze.json")
    _require(
        result["schema"] == "tnfr.maintained-child-joint-window-result.v1",
        "joint result schema",
    )
    verify(result["attempt"], "attempt.json")
    return documents, tuple(receipts[name] for name in sorted(receipts))


def _admit_fixed_cell(retention, protocol, report):
    """Rebuild only cell 12 from a retention freshly loaded by the public reader."""
    prediction, graph = retention.prediction, retention.admitted.graph
    _require(
        prediction.nodes == tuple(range(20)) + (CHILD,)
        and graph.nodes[CHILD]["parent_node"] == PARENT
        and tuple(graph[CHILD]) == (PARENT,)
        and len(prediction.edges) == 53
        and retention.summary["status"] == "completed"
        and retention.summary["full_horizon_retention"] is True
        and len(retention.steps) == 256,
        "complete actual child retention required",
    )
    source_time = exact_or_represented_real(graph.graph["_t"], "source structural time")
    minimum = exact_or_represented_real(
        resolve_thol_min_epi(graph.graph), "minimum form"
    )
    threshold = exact_or_represented_real(
        resolve_thol_bifurcation_threshold(graph.graph), "THOL threshold"
    )
    expected_protocol = {
        "schema": "tnfr.maintained-child-joint-window-protocol.v1",
        "evidence_kind": "new_retrospective_observation_of_retained_evidence",
        "not_a_prospective_reserved_prediction": True,
        "child": CHILD,
        "parent": PARENT,
        "node_order": prediction.nodes,
        "state_dimension": 42,
        "original_edges": 53,
        "law": "x_dot=M*S(theta); theta_dot=M*L_W*x; M=diag(nu/(pi*d))",
        "held": ("support", "conductances", "positive capacities"),
        "forcing": 0,
        "loss": 0,
        "clock": "original structural clock",
        "source_time": source_time,
        "elapsed_horizon": Q(1, 2),
        "cell_width": Q(1, 512),
        "maximum_cells": 256,
        "minimum_form": minimum,
        "threshold": threshold,
        "criterion": {
            "form": "whole-cell lower bound >= unchanged represented THOL_MIN_EPI",
            "pressure": "whole-cell constitutive pressure lower bound > 0",
            "acceleration": "whole-cell interval wholly above tau or below -tau",
            "identity": "all original principal margins strictly positive on same cell",
        },
        "selection": "first qualifying whole original cell in chronological order; stop",
        "unresolved_policy": "no certified cell is not a proof of impossibility",
        "sample_acquisition": False,
        "runtime_history_installation": False,
        "maximum_new_trajectories": 0,
        "maximum_events": 0,
        "mesh_refinement": False,
        "parameter_retuning": False,
        "grammar": "empty executed child history independently blocks U4b",
    }
    _require(
        type(protocol) is dict
        and all(
            key in protocol and _same(protocol[key], value)
            for key, value in expected_protocol.items()
        ),
        "joint protocol differs from actual source and policy",
    )
    result_receipt = next(
        row for row in retention.receipts if row["path"] == "result.json"
    )
    _require(
        _same(
            protocol["source_result"],
            {**result_receipt, "path": f"{_RETENTION_PATH}/result.json"},
        ),
        "joint source result association differs",
    )
    grammar = validate_candidate(graph, CHILD, "THOL")
    errors = tuple(item.rule for item in grammar.violations if item.severity == "error")
    expected_report = {
        "status": "joint_window_admitted",
        "child": CHILD,
        "parent": PARENT,
        "source_time": source_time,
        "minimum_form": minimum,
        "threshold": threshold,
        "retained_cells": 256,
        "source_grammar_error_rules": errors,
        "source_receipts": retention.receipts,
        "retrospective_observation": True,
        "acquired_history": False,
        "acquired_u4b_context": False,
        "executed_event": False,
        "full_thol_admission_established": False,
    }
    _require(
        type(report) is dict
        and all(
            key in report and _same(report[key], value)
            for key, value in expected_report.items()
        ),
        "joint report source, policy or scope differs",
    )
    prefix, selected = report["inspected_cells"], report["selected_cell"]
    _require(
        type(prefix) is tuple
        and len(prefix) == CELL_INDEX + 1
        and all(
            type(row) is dict and type(row["index"]) is int and row["index"] == index
            for index, row in enumerate(prefix)
        )
        and _same(selected, prefix[-1]),
        "fixed selected cell association differs",
    )
    step = retention.steps[CELL_INDEX]
    _require(
        type(step.index) is int
        and step.index == CELL_INDEX
        and step.time == CELL_START
        and step.time + step.duration == CELL_END,
        "fixed retained cell chronology differs",
    )
    bounds = bound_node_history_box(
        epi=step.tube[:21],
        phases=step.tube[21:],
        capacities=prediction.capacities,
        edges=prediction.edges,
        conductances=prediction.conductances,
        shifts=prediction.shifts,
        target_index=CHILD_INDEX,
    )
    margins = tuple(
        max(direct, energy.lo)
        for direct, energy in zip(
            bounds.principal_margins,
            prediction.bounds.horizon_remaining_margins,
            strict=True,
        )
    )
    conditions = classify_joint_conditions(
        form=bounds.form,
        pressure=bounds.pressure,
        acceleration=bounds.form_acceleration,
        minimum_form=minimum,
        threshold=threshold,
        principal_margins=margins,
    )
    cell = ChildJointCell(
        step.index,
        step.time,
        step.time + step.duration,
        bounds.form,
        bounds.pressure,
        bounds.form_acceleration,
        min(margins),
        conditions,
    )
    _require(conditions.joint, "fixed cell does not rebuild joint admission")
    _require(_same(selected, cell), "selected cell differs from its primitive rebuild")
    return step, bounds, cell, minimum, threshold


def load_child_joint_window(evidence_dir, *, repository_root):
    """Verify prior observation associations and rebuild its fixed admitted cell.

    One fresh complete retention load supplies the actual source, held law and
    chronological tubes. Earlier child-condition classifications are neither
    inspected nor rerun. Byte associations do not authenticate acquisition;
    frozen derivative jets retain the existing producer-provenance boundary.
    """
    folder, root = Path(evidence_dir).resolve(), Path(repository_root).resolve()
    documents, receipts = _read_joint_artifacts(folder, root)
    retention = load_sha_retention(root / _RETENTION_PATH, repository_root=root)
    signature = _signature(retention.admitted.graph)
    prior_names = {
        _receipt_fields(row) for row in documents["freeze"]["prior_artifacts"]
    }
    _require(
        {f"{_RETENTION_PATH}/{row['path']}" for row in retention.receipts}
        <= prior_names,
        "joint freeze omits retained source artifacts",
    )
    step, bounds, cell, minimum, threshold = _admit_fixed_cell(
        retention, documents["protocol"], documents["result"]["report"]
    )
    _require(
        _signature(retention.admitted.graph) == signature,
        "joint source reader changed persistent state",
    )
    return ChildJointAdmission(
        retention, step, bounds, cell, minimum, threshold, receipts
    )
