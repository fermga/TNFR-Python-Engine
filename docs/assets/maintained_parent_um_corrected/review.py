"""Review the retained committed UM response; never replay an event or flow."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(r"C:\TNFR-Python-Engine")
sys.path.insert(0, str(ROOT / "src"))

from tnfr.dynamics import integrators, relational
from tnfr.mathematics._rational_interval import I
from tnfr.operators import _coupling_stage_kernel, network_stage, word_execution
from tnfr.research import local_capacity_history, maintained_parent_thol_execution
from tnfr.research import maintained_parent_um_execution as owner
from tnfr.research import maintained_successor_motion_certificate
from tnfr.research.artifact_io import (
    decode_exact_tree,
    encode_exact_tree,
    file_receipt,
    sha256_bytes,
    validate_artifact_path,
    verify_archive_members,
    write_json_once,
)
from tnfr.research.local_capacity_history import encode_result
from tnfr.research.maintained_parent_thol import _represented, _signature
from tnfr.research.maintained_parent_um_source import load_thol_successor
from tnfr.research.maintained_successor_motion import predict_successor_motion
from tnfr.research.retained_graph_state import restore_recorded_graph_state
from tnfr.utils.cache import GRAPH_RUNTIME_CACHE_KEYS
from tnfr.utils.io import json_dumps, json_loads

OUT = ROOT / "docs/assets/maintained_parent_um_corrected"
INPUTS = (
    "protocol.json",
    "source.json",
    "source-manifest.json",
    "source.zip",
    "freeze.json",
    "attempt.json",
    "raw-after.json",
    "response.json",
    "report.json",
)
OWNER = "src/tnfr/research/maintained_parent_um_execution.py"
TEST = "tests/research/test_maintained_parent_um_execution.py"


def read(path):
    return decode_exact_tree(json_loads(path.read_bytes()))


def write(name, value):
    write_json_once(OUT / name, encode_result(value))


def same(left, right):
    return json_dumps(
        encode_exact_tree(encode_result(left)), sort_keys=True
    ) == json_dumps(encode_exact_tree(encode_result(right)), sort_keys=True)


def receipt(path, name=None):
    return file_receipt(path, label=name or path.relative_to(ROOT).as_posix())


def verify(base, rows):
    for row in rows:
        name = validate_artifact_path(row["path"])
        path = (base / name).resolve()
        if not path.is_relative_to(base.resolve()) or receipt(path, name) != row:
            raise ValueError(f"retained input differs: {name}")


def freeze_review():
    inputs = tuple(receipt(OUT / name, name) for name in INPUTS)
    files = {name: (ROOT / name).read_bytes() for name in (OWNER, TEST)}
    files["review.py"] = Path(__file__).read_bytes()
    with ZipFile(OUT / "review-source.zip", "x", compression=ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            archive.writestr(name, data)
    write(
        "review-protocol.json",
        {
            "schema": "tnfr.maintained-parent-um-assessment-review-protocol.v1",
            "evaluation_status_at_freeze": "not_evaluated",
            "inputs": inputs,
            "source_archive": receipt(OUT / "review-source.zip", "review-source.zip"),
            "members": tuple(
                {"path": name, "bytes": len(data), "sha256": sha256_bytes(data)}
                for name, data in sorted(files.items())
            ),
            "corrections": (
                "Subtract old-edge form energies as exact represented rational scalars before interval enclosure; unchanged arguments cancel exactly without tolerance.",
                "Require the canonical topology invalidation flag _dnfr_prep_dirty to be exactly True in raw and committed states; preserve all other graph context and protect the flag during refresh.",
            ),
            "maximum_events": 0,
            "maximum_trajectories": 0,
            "scientific_protocol_changed": False,
            "scope": "Separate reader correction on the immutable committed response. Original 25/27 assessment remains unchanged; no operator, refreshed response or motion is evaluated.",
        },
    )
    print("Read-only review protocol and source frozen", flush=True)


def review():
    if (OUT / "review.json").exists():
        raise FileExistsError("review already retained")
    protocol = read(OUT / "review-protocol.json")
    verify(OUT, protocol["inputs"])
    verify(OUT, (protocol["source_archive"],))
    verify_archive_members(
        OUT / "review-source.zip", {r["path"]: r["sha256"] for r in protocol["members"]}
    )
    for row in protocol["members"]:
        path = Path(__file__) if row["path"] == "review.py" else ROOT / row["path"]
        if receipt(path, row["path"]) != row:
            raise ValueError("review source changed after freeze")
    freeze = read(OUT / "freeze.json")
    verify(OUT, freeze["artifacts"])
    verify(ROOT, freeze["prior_artifacts"])
    manifest = read(OUT / "source-manifest.json")["files"]
    verify_archive_members(
        OUT / "source.zip", {r["path"]: r["sha256"] for r in manifest}
    )
    for row in manifest:
        if row["path"] not in ("instrument.py", OWNER):
            verify(ROOT, (row,))
    source, response, original = (
        read(OUT / name) for name in ("source.json", "response.json", "report.json")
    )
    graphs = tuple(
        restore_recorded_graph_state(response[name]).graph
        for name in ("before", "raw_after", "endpoint")
    )
    before, raw, endpoint = graphs
    before_signatures = tuple(map(_signature, graphs))
    admitted = load_thol_successor(
        ROOT / "docs/assets/maintained_parent_thol", repository_root=ROOT
    )
    assert _signature(admitted.graph) == _signature(before)
    assert _signature(
        restore_recorded_graph_state(source["before"]).graph
    ) == _signature(before)
    prediction = predict_successor_motion(before)
    assert same(prediction, source["motion_prediction"])
    assert same(prediction, response["prediction"])
    assert same(response["raw_after"], read(OUT / "raw-after.json"))
    for rows in (source["correction_receipts"], source["existing_motion_receipts"]):
        verify(ROOT, rows)
    verify(
        ROOT / "docs/assets/maintained_parent_thol", source["retained_source_receipts"]
    )
    assert same(admitted.receipts, source["retained_source_receipts"])
    assert same(admitted.receipts, response["retained_source_receipts"])
    verify(OUT, tuple(original[key] for key in ("attempt", "response", "raw_after")))
    verify(OUT, (read(OUT / "attempt.json")["freeze"],))
    actual = owner.check_um_endpoint(
        *graphs, prediction=prediction, stage=response["stage"]
    )
    old = original["assessment"]
    changed_checks = {
        key for key in old["checks"] if old["checks"][key] != actual["checks"][key]
    }
    assert changed_checks == {
        "old_edge_form_work_zero",
        "declared_graph_context_preserved",
    }
    assert sum(old["checks"].values()) == 25 and old["all_checks_passed"] is False
    assert len(actual["checks"]) == 27 and actual["all_checks_passed"] is True
    exempt = {"old_edge_form_work", "actual_total_work", "checks", "all_checks_passed"}
    assert set(actual) == set(old)
    assert all(same(actual[key], old[key]) for key in old if key not in exempt)
    reset = prediction.chamber.reset
    fields = owner.bound_relational_sine_support_exchange(
        endpoint, reference_model=owner.MODEL
    )
    exact_work, legacy_work = [], []
    for (i, j), old_weight in zip(reset.old_edges, reset.old_conductances, strict=True):
        weight = _represented(
            endpoint[reset.nodes[i]][reset.nodes[j]]["weight"], "old conductance"
        )
        delta = fields.epi[i] - fields.epi[j]
        old_delta = reset.epi[i] - reset.epi[j]
        exact_work.append((weight * delta**2 - old_weight * old_delta**2) / 2)
        legacy_work.append((I(weight) * delta**2 - old_weight * old_delta**2) / 2)
    assert all(value == 0 for value in exact_work)
    assert same(legacy_work, old["old_edge_form_work"])
    assert all(value.contains(0) for value in legacy_work)
    assert any(value.width > 0 for value in legacy_work)
    excluded = (
        GRAPH_RUNTIME_CACHE_KEYS
        | owner._PRESSURE_CACHE_KEYS
        | owner._STAGE_KEYS
        | {"_vfmax", "_vfmax_node", "_edge_version"}
    )
    original_context = {k: v for k, v in before.graph.items() if k not in excluded}
    for graph in (raw, endpoint):
        context = {k: v for k, v in graph.graph.items() if k not in excluded}
        assert "_dnfr_prep_dirty" not in original_context
        assert context.pop("_dnfr_prep_dirty") is True
        assert owner._same(context, original_context)
    assert tuple(map(_signature, graphs)) == before_signatures
    verify(OUT, protocol["inputs"])
    write(
        "review.json",
        {
            "schema": "tnfr.maintained-parent-um-assessment-review.v1",
            "protocol": receipt(OUT / "review-protocol.json", "review-protocol.json"),
            "original_report": receipt(OUT / "report.json", "report.json"),
            "response": receipt(OUT / "response.json", "response.json"),
            "diagnosis": {
                "exact_old_edge_form_work": tuple(exact_work),
                "original_old_edge_enclosure_widths": tuple(
                    v.width for v in legacy_work
                ),
                "sole_added_graph_context_key": "_dnfr_prep_dirty",
                "raw_and_endpoint_invalidation_flag": True,
                "other_assessment_fields_unchanged": True,
                "changed_checks": tuple(sorted(changed_checks)),
            },
            "assessment": actual,
            "new_events_executed": 0,
            "new_trajectories_executed": 0,
            "original_report_preserved": True,
        },
    )
    print(
        "Read-only review:27/27; original25/27 report and response unchanged",
        flush=True,
    )


def forbidden(*args, **kwargs):
    raise RuntimeError("read-only review cannot evaluate any event or motion")


if __name__ == "__main__":
    with patch.multiple(
        owner, evaluate_um_event=forbidden, refresh_pressure=forbidden
    ), patch.object(
        maintained_successor_motion_certificate, "certify_motion", forbidden
    ), patch.object(
        maintained_parent_thol_execution, "evaluate_thol_event", forbidden
    ), patch.object(
        local_capacity_history, "evaluate_local_history", forbidden
    ), patch.object(
        local_capacity_history, "declared_preparation", forbidden
    ), patch.object(
        local_capacity_history, "rk4_coupled_step", forbidden
    ), patch.object(
        _coupling_stage_kernel, "propose_coupling_target", forbidden
    ), patch.object(
        _coupling_stage_kernel, "propose_coupling_stage", forbidden
    ), patch.object(
        word_execution, "execute_network_operator_stage", forbidden
    ), patch.object(
        network_stage, "execute_coupling_stage", forbidden
    ), patch.object(
        integrators, "rk4_coupled_step", forbidden
    ), patch.object(
        relational, "step_relational_exchange", forbidden
    ):
        {"freeze": freeze_review, "review": review}[sys.argv[1]]()
