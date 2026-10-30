"""Source-associated use of the fixed graph-metric preparation certificate.

This consumer separates historical source recertification, saved observation
and full-phase evidence admission, and conditional preparation arithmetic. It
does not generate a new response trajectory. A caller must freeze its complete
input inventory, implementation and supervised work budget before evaluation.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from fractions import Fraction as Q
from pathlib import Path

from .._exact_time import finite_represented_real
from ..mathematics._rational_interval import I, pi_interval
from ..utils.io import json_loads
from .artifact_io import (
    decode_exact_tree,
    file_receipt,
    read_bytes_bounded,
    validate_artifact_path,
    verify_archive_members,
)
from .local_capacity_history import MODEL, TIMES
from .maintained_parent_thol import _signature
from .maintained_sha_cut_separator import _capture_cut_source
from .maintained_sha_motion_source import validate_live_sha_successor
from .maintained_sha_phase_evidence import load_phase_branch, require_phase_evidence
from .maintained_sha_preparation_response import (
    FrozenNominalEvidence,
    _evidence_record,
    _same_typed,
    _verify_nominal_evidence,
)
from .sine_preparation_bounds import adjacent_coordinate_hull
from .sine_preparation_metric import (
    bound_metric_cells,
    prepare_metric_context,
    prepare_metric_family,
)

RETENTION = "docs/assets/maintained_sha_retention"
DIRECT = "artifacts/parent-child-direct-response-20261022"
CUT = "artifacts/parent-child-cut-direct-response-20261023"
STEP = Q(1, 512)
EDGE_ALLOWANCE = Q(1, 1024)
MAX_INVENTORY_FILES = 4096
MAX_INVENTORY_BYTES = 512 * 1024**2
MAX_FILE_BYTES = 128 * 1024**2
MAX_METADATA_BYTES = 16 * 1024**2


@dataclass(frozen=True)
class FrozenMetricEvidence:
    """Prospective receipts, not cached scientific success or authentication.

    Producer receipt labels are relative to their original evidence folders.
    ``inventory`` labels are repository-relative and must include complete
    closed nominal inventories and retained producer freeze dependencies.
    """

    nominal: FrozenNominalEvidence
    connected_phase_result: dict
    connected_phase_source: dict
    connected_producer_receipts: tuple
    cut_producer_receipts: tuple
    inventory: tuple


@dataclass(frozen=True)
class MetricPreparationResponse:
    source: object
    nominal: object
    hulls: tuple
    connected_family: object
    cut_family: object | None
    connected_run: object
    cut_run: object | None
    connected_radius_margin: Q | None
    cut_radius_margin: Q | None
    nominal_gap: I
    nominal_gap_margin: Q
    status: str
    reason: str | None
    phase_receipts: tuple
    source_receipts: tuple = ()
    family_members: int = 3**42
    scope: tuple = (
        "same_original_full42_coordinate_adjacent_real_family_for_both_laws",
        "fixed_bare_graph_metric_edge_allowance1_over1024_original256_cells",
        "all_host_coordinates_and_asymmetric_constant_child_errors_retained",
        "whole_cell_phase_and_nominal_observation_producer_premises_inherited",
        "no_new_response_field_jet_trajectory_event_or_numerical_tuning",
        "insufficient_bound_is_not_a_family_counterexample_or_instability",
    )


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _receipt(row):
    _require(
        type(row) is dict and set(row) == {"path", "bytes", "sha256"},
        "complete typed evidence receipt required",
    )
    _require(type(row["path"]) is str, "ordinary receipt path required")
    validate_artifact_path(row["path"])
    _require(
        type(row["bytes"]) is int and 0 <= row["bytes"] <= MAX_FILE_BYTES,
        "nonnegative bounded receipt bytes required",
    )
    digest = row["sha256"]
    _require(
        type(digest) is str
        and len(digest) == 64
        and all(char in "0123456789abcdef" for char in digest),
        "ordinary SHA-256 receipt required",
    )
    return dict(row)


def _inventory(evidence):
    """Read content associations only; do not decode any step member."""
    _require(
        type(evidence) is FrozenMetricEvidence
        and type(evidence.nominal) is FrozenNominalEvidence,
        "fixed metric and nominal evidence required",
    )
    rows = evidence.inventory
    _require(
        type(rows) is tuple and 0 < len(rows) <= MAX_INVENTORY_FILES,
        "bounded complete input inventory required",
    )
    admitted = tuple(_receipt(row) for row in rows)
    _require(
        len({row["path"].casefold() for row in admitted}) == len(admitted),
        "duplicate or aliased evidence path",
    )
    _require(
        sum(row["bytes"] for row in admitted) <= MAX_INVENTORY_BYTES,
        "complete input inventory exceeds byte budget",
    )
    root = Path(evidence.nominal.repository_root).resolve()
    inventory = {row["path"]: row for row in admitted}
    for row in admitted:
        path = (root / row["path"]).resolve()
        _require(path.is_relative_to(root), "evidence path leaves repository")
        _require(
            file_receipt(path, label=row["path"], max_bytes=row["bytes"]) == row,
            "frozen input receipt differs: " + row["path"],
        )

    def bind(folder, row):
        row = _receipt(row)
        relative = (Path(folder) / row["path"]).as_posix() if folder else row["path"]
        expected = dict(row, path=relative)
        _require(
            inventory.get(relative) == expected,
            "complete inventory omits or changes dependency: " + relative,
        )

    def read(relative):
        _require(relative in inventory, "missing producer metadata: " + relative)
        raw = read_bytes_bounded(root / relative, max_bytes=MAX_METADATA_BYTES)
        return decode_exact_tree(json_loads(raw))

    for folder in (DIRECT, CUT):
        state = read(folder + "/final-state.json")
        _require(
            type(state) is dict
            and type(state.get("evidence")) is tuple
            and state["evidence"],
            "closed nominal evidence inventory required",
        )
        for row in state["evidence"]:
            bind(folder, row)
        manifest = read(folder + "/instrument-source.json")
        hashes = manifest["files"]
        _require(type(hashes) is dict and hashes, "frozen producer source map required")
        _require(
            manifest["archive"]["path"] == "instrument-source.zip",
            "frozen source archive role differs",
        )
        bind(folder, manifest["archive"])
        archive = folder + "/instrument-source.zip"
        _require(archive in inventory, "missing frozen nominal source archive")
        verify_archive_members(root / archive, hashes, max_bytes=MAX_FILE_BYTES)

    frozen = read(RETENTION + "/freeze.json")
    _require(
        frozen.get("schema") == "tnfr.maintained-sha-retention-freeze.v1"
        and frozen.get("evaluation_status_at_freeze") == "not_evaluated",
        "retained producer freeze differs",
    )
    _require(
        type(frozen.get("artifacts")) is tuple
        and len(frozen["artifacts"]) == 4
        and {row["path"] for row in frozen["artifacts"]}
        == {"protocol.json", "source.json", "source-manifest.json", "source.zip"},
        "retained frozen artifact inventory differs",
    )
    _require(
        type(frozen.get("prior_artifacts")) is tuple and frozen["prior_artifacts"],
        "retained prior dependencies required",
    )
    for row in frozen["artifacts"]:
        bind(RETENTION, row)
    for row in frozen["prior_artifacts"]:
        bind("", row)
    manifest = read(RETENTION + "/source-manifest.json")
    _require(
        manifest.get("schema") == "tnfr.source-manifest.v1"
        and type(manifest.get("files")) is tuple,
        "retained source manifest differs",
    )
    members = tuple(_receipt(row) for row in manifest["files"])
    _require(
        members and len({row["path"] for row in members}) == len(members),
        "duplicate retained source member",
    )
    verify_archive_members(
        root / RETENTION / "source.zip",
        {row["path"]: row["sha256"] for row in members},
        max_bytes=MAX_FILE_BYTES,
    )

    for name in (
        "connected_result",
        "connected_final_state",
        "cut_result",
        "cut_final_state",
        "observation_verifier",
    ):
        bind("", getattr(evidence.nominal, name))
    bind(RETENTION, evidence.connected_phase_result)
    bind(RETENTION, evidence.connected_phase_source)
    for folder, rows in (
        (RETENTION, evidence.connected_producer_receipts),
        (CUT, evidence.cut_producer_receipts),
    ):
        _require(
            type(rows) is tuple and 1 <= len(rows) <= 16,
            "bounded producer receipt tuple required",
        )
        for row in rows:
            bind(folder, row)
    return inventory


def _bind_phase_inventory(inventory, branch):
    """Include every consumed phase input in the frozen and reread inventory."""
    branch = require_phase_evidence(branch)
    folder = RETENTION if branch.branch == "connected" else CUT
    for receipt in branch.receipts:
        row = _receipt(asdict(receipt))
        path = folder + "/" + row["path"]
        _require(
            inventory.get(path) == dict(row, path=path),
            "complete inventory omits or changes phase input: " + path,
        )


def _match_phase_observations(source, branch, saved):
    """Bind marked radius premises to the same admitted whole-state tubes."""
    branch = require_phase_evidence(branch)
    rows = saved["report"]["cells"]
    _require(
        type(rows) is tuple and len(rows) == len(branch.steps) == 256,
        "complete nominal/phase cell association required",
    )
    n = len(source.nodes)
    for step, row in zip(branch.steps, rows, strict=True):
        if branch.branch == "connected":
            tube = step.tube
        else:
            host = source.host_indices
            form = [I(value) for value in source.epi]
            phase = [I(value) for value in source.phase]
            for local, global_index in enumerate(host):
                form[global_index] = step.tube[local]
                phase[global_index] = step.tube[len(host) + local]
            tube = tuple(form + phase)
        a, c = source.parent_index, source.child_index
        expected = (tube[c] - tube[a], tube[n + c] - tube[n + a])
        for key, value in zip(("r", "delta"), expected, strict=True):
            _require(
                _same_typed(row[key], {"lo": value.lo, "hi": value.hi}),
                "marked observation differs from admitted full tube",
            )
    if branch.branch == "connected":
        prior = saved["report"]["retained_receipts"]
        _require(type(prior) is tuple, "connected retained inventory required")
        _require(
            len({row["path"] for row in prior}) == len(prior),
            "duplicate connected retained receipt",
        )
        expected = {row["path"]: row for row in prior}
        for receipt in branch.receipts:
            row = asdict(receipt)
            _require(
                _same_typed(expected.get(row["path"]), row),
                "connected observation refers to different retained cells",
            )


def _bound_metric_response(source, nominal, connected, cut):
    """One fixed conditional arithmetic route after complete evidence admission."""
    connected, cut = map(require_phase_evidence, (connected, cut))
    _require(
        connected.branch == "connected" and cut.branch == "cut_host",
        "both complete branch roles required",
    )
    centers = source.epi + source.phase
    represented = []
    for index, value in enumerate(centers):
        number, exact = finite_represented_real(value, f"family center {index}")
        _require(
            type(value) is Q and exact == value, "exact represented centers required"
        )
        represented.append(number)
    _require(len(represented) == 42, "all42 family coordinates required")
    hulls = tuple(adjacent_coordinate_hull(value) for value in represented)
    form = tuple((item.lower_error, item.upper_error) for item in hulls[:21])
    phase = tuple((item.lower_error, item.upper_error) for item in hulls[21:])

    def prepare(indices, edges, weights, degrees, *, child=None):
        context = prepare_metric_context(
            capacities=tuple(source.capacity[i] for i in indices),
            edges=edges,
            conductances=weights,
            degrees=degrees,
        )
        extra = (
            {"child": child}
            if child is not None
            else {
                "child_form_error": form[source.child_index],
                "child_phase_error": phase[source.child_index],
            }
        )
        return prepare_metric_family(
            context,
            form_errors=tuple(form[i] for i in indices),
            phase_errors=tuple(phase[i] for i in indices),
            parent=indices.index(source.parent_index),
            reference_weight=source.reference_conductance,
            **extra,
        )

    def run(family, branch):
        return bound_metric_cells(
            family,
            phase_cells=branch.phase_cells,
            edge_allowances=(EDGE_ALLOWANCE,) * len(family.context.edges),
            durations=(STEP,) * 256,
        )

    cf = prepare(
        tuple(range(21)),
        source.edges,
        source.conductances,
        source.connected_degrees,
        child=source.child_index,
    )
    cr = run(cf, connected)
    kf = kr = cm = km = None
    reason = "connected_" + cr.reason if cr.status != "completed" else None
    if reason is None:
        host = source.host_indices
        kf = prepare(
            host,
            source.host_edges,
            source.host_conductances,
            tuple(source.cut_degrees[i] for i in host),
        )
        kr = run(kf, cut)
        reason = "cut_" + kr.reason if kr.status != "completed" else None
    gap = nominal.connected_clockwise - nominal.cut_clockwise
    gap_margin = gap.lo - pi_interval().hi
    if reason is None:
        cm = nominal.connected_radius_squared - cr.final_portrait_error_upper**2
        km = nominal.cut_radius_squared - kr.final_portrait_error_upper**2
        if min(cm, km, gap_margin) <= 0:
            reason = "portrait_or_shared_angle_gate"
    return MetricPreparationResponse(
        source,
        nominal,
        hulls,
        cf,
        kf,
        cr,
        kr,
        cm,
        km,
        gap,
        gap_margin,
        "family_ordering_certified" if reason is None else "insufficient_bound",
        reason,
        (connected.receipts, cut.receipts),
    )


def audit_sha_metric_response(admitted, *, evidence):
    """Re-admit the original source and both proofs before any family width.

    The caller's first file-backed source load and this fresh revalidation are
    historical recertifications, with their own old field/jet budget. All later
    stages are passive saved arithmetic or metric comparisons. Missing source
    evidence raises before widths; failed sufficient metric gates are ordinary
    ``insufficient_bound`` outcomes. No automatic alternate attempt is made.
    """
    _require(type(evidence) is FrozenMetricEvidence, "fixed metric evidence required")
    evidence = deepcopy(evidence)
    frozen_inventory = _inventory(evidence)
    fresh = validate_live_sha_successor(admitted)
    root = Path(fresh.repository_root).resolve()
    _require(
        root == Path(evidence.nominal.repository_root).resolve(),
        "fresh source and evidence repositories differ",
    )
    before = _signature(fresh.graph)
    _require(
        len(fresh.graph) == 21 and fresh.graph.number_of_edges() == 53,
        "original21-node53-edge source required",
    )
    _, clock = finite_represented_real(fresh.graph.graph.get("_t"), "structural clock")
    _require(clock == TIMES[-1], "original source clock required")
    source = _capture_cut_source(
        fresh.graph, parent=2, child="2_sub_0", reference_model=MODEL
    )
    receipts = deepcopy(fresh.receipts)
    nominal = _verify_nominal_evidence(source, receipts, evidence.nominal)
    binding = {
        "complete_source": asdict(source),
        "source_receipts": receipts,
        "source_graph_signature": before,
    }
    initial = tuple(I(value) for value in source.epi + source.phase)
    connected = load_phase_branch(
        root / RETENTION,
        branch="connected",
        node_order=source.nodes,
        initial_box=initial,
        source_binding=binding,
        result_receipt=evidence.connected_phase_result,
        source_receipt=evidence.connected_phase_source,
        producer_receipts=evidence.connected_producer_receipts,
    )
    host = source.host_indices
    cut_initial = tuple(
        I(values[i]) for values in (source.epi, source.phase) for i in host
    )
    cut_receipt = dict(evidence.nominal.cut_result, path="result.json")
    cut = load_phase_branch(
        root / CUT,
        branch="cut_host",
        node_order=tuple(source.nodes[i] for i in host),
        initial_box=cut_initial,
        source_binding=binding,
        result_receipt=cut_receipt,
        producer_receipts=evidence.cut_producer_receipts,
    )
    for branch, key in ((connected, "connected_result"), (cut, "cut_result")):
        _bind_phase_inventory(frozen_inventory, branch)
        _match_phase_observations(
            source, branch, _evidence_record(evidence.nominal, key)
        )
    result = _bound_metric_response(source, nominal, connected, cut)
    _require(
        _signature(fresh.graph) == before and _signature(admitted.graph) == before,
        "metric consumer changed persistent source primitives",
    )
    _require(
        _inventory(evidence) == frozen_inventory,
        "metric input association changed during the audit",
    )
    return replace(result, source_receipts=receipts)
