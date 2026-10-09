"""Read the real neighbor freeze without executing its reserved evaluator.

Byte/base inspection and rebuilt static arithmetic establish associations,
not acquisition, chronological authenticity or derivative generation.
"""

import ast
import re
import zipfile
from dataclasses import asdict
from fractions import Fraction as Q
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.research.artifact_io import (
    decode_exact_tree,
    encode_exact_tree,
    file_receipt,
    read_bytes_bounded,
    sha256_bytes,
)
from tnfr.research.frozen_source import inspect_frozen_source
from tnfr.research.sine_class_neighbor_forward import (
    neighbor_forward_inputs,
    neighbor_forward_policy,
    neighbor_forward_prediction,
    neighbor_forward_sources,
)
from tnfr.utils.io import json_dumps, json_loads

ROOT = Path(__file__).resolve().parents[2]
STEM = "docs/assets/sine_formed_classes/class-neighbor-forward-v1"
EVALUATOR = "build/class-neighbor-forward-freeze/evaluate_neighbor.py"
FREEZER = "build/class-neighbor-forward-freeze/freeze_neighbor.py"
PROOF = "theory/nodal/SINE_CLASS_NEIGHBOR_FORWARD_PROTOCOL.md"
# First immutable freeze; no scientific outcome is part of these receipts.
BASE = "8af5b1932308b81e5fbb4e519de0162b8e82ccfc"
RECEIPT_SHA256 = "c2dec8230791e8b5377283cd088b1414c105f010d3ada82a4e5e075d8ddc0cdf"
ARCHIVE_SHA256 = "2317c3a54dd83704d99ad1bcf9ff976d6f25e5cd7cc2754fa5e38726769733ec"
PROTOCOL_SHA256 = "bcdc0160bc91e628bb1259139f5451772d2af61298f88e068d55232fef498179"
ARCHIVE_BYTES = 149640
PROTOCOL_BYTES = 152589
ARCHIVED_FILE_COUNT = 30


@pytest.fixture(scope="module", autouse=True)
def no_scientific_execution():
    assert re.fullmatch(r"[0-9a-f]{40}", BASE), "first freeze receipt is not installed"
    receipt = json_loads(
        read_bytes_bounded(ROOT / (STEM + ".freeze.json"), max_bytes=1024**2)
    )
    assert receipt["source_base_commit"] == BASE
    with forbid_sine_regeneration(git_root=ROOT, git_base=BASE):
        yield


@pytest.fixture(scope="module")
def frozen(no_scientific_execution):
    paths = tuple(
        ROOT / (STEM + suffix)
        for suffix in (".freeze.json", ".protocol.json", ".source.zip")
    )
    before = tuple(file_receipt(path) for path in paths)
    inspection = inspect_frozen_source(ROOT, STEM + ".freeze.json")
    receipt_raw, protocol_raw = (
        read_bytes_bounded(path, max_bytes=32 * 1024**2) for path in paths[:2]
    )
    receipt, protocol = (
        decode_exact_tree(json_loads(raw)) for raw in (receipt_raw, protocol_raw)
    )
    with zipfile.ZipFile(paths[2]) as archive:
        names = tuple(archive.namelist())
        retained = {
            name: archive.read(name)
            for name in (
                PROOF,
                EVALUATOR,
                FREEZER,
                "source-manifest.json",
                STEM + ".protocol.json",
            )
        }
    manifest = decode_exact_tree(json_loads(retained["source-manifest.json"]))
    yield SimpleNamespace(
        inspection=inspection,
        receipt=receipt,
        protocol=protocol,
        manifest=manifest,
        names=names,
        retained=retained,
        protocol_raw=protocol_raw,
    )
    assert tuple(file_receipt(path) for path in paths) == before


def _same_exact_tree(actual, expected):
    # Equality of Python scalars alone would admit True as1 or False as0.
    assert json_dumps(encode_exact_tree(actual), sort_keys=True) == json_dumps(
        encode_exact_tree(expected), sort_keys=True
    )


def test_receipts_associate_fixed_git_source_and_complete_immutable_inventory(frozen):
    receipt, protocol, manifest, inspection = (
        frozen.receipt,
        frozen.protocol,
        frozen.manifest,
        frozen.inspection,
    )
    assert receipt["schema"] == "tnfr.sine-class-neighbor-forward-freeze.v1"
    assert protocol["schema"] == "tnfr.sine-class-neighbor-forward-protocol.v1"
    assert manifest["schema"] == "tnfr.sine-class-neighbor-forward-source-snapshot.v1"
    assert (
        inspection.source_base_commit
        == receipt["source_base_commit"]
        == protocol["source_base_commit"]
        == manifest["source_base_commit"]
        == BASE
    )
    assert inspection.receipt_sha256 == RECEIPT_SHA256
    assert (
        receipt["runtime_overlays"]
        == protocol["runtime_overlays"]
        == manifest["runtime_overlays"]
        == ()
    )
    assert inspection.future_evaluator == receipt["future_evaluator"] == EVALUATOR
    assert (
        inspection.archived_file_count == len(manifest["files"]) == ARCHIVED_FILE_COUNT
    )
    assert len(frozen.names) == len(set(frozen.names)) == len(manifest["files"]) + 1
    assert frozen.retained[STEM + ".protocol.json"] == frozen.protocol_raw
    expected = {
        STEM + ".protocol.json": (PROTOCOL_BYTES, PROTOCOL_SHA256),
        STEM + ".source.zip": (ARCHIVE_BYTES, ARCHIVE_SHA256),
    }
    assert {
        item["path"]: (item["bytes"], item["sha256"]) for item in receipt["artifacts"]
    } == expected
    paths = {item["path"] for item in manifest["files"]}
    assert {
        "src/tnfr/physics/relational_sine_class_readout.py",
        "src/tnfr/physics/_sine_class_readout_evidence.py",
        "src/tnfr/physics/_sine_class_neighbor_nonadditivity.py",
        "src/tnfr/physics/_sine_class_collective_interface.py",
        "src/tnfr/physics/_sine_flow.py",
        "src/tnfr/mathematics/_validated_taylor.py",
        "src/tnfr/research/sine_class_neighbor_forward.py",
        "src/tnfr/research/frozen_source.py",
        EVALUATOR,
        FREEZER,
        PROOF,
    } <= paths
    for entry in manifest["files"]:
        if entry["path"].startswith("src/"):
            assert re.fullmatch(r"[0-9a-f]{64}", entry["git_blob_sha256"])
            assert re.fullmatch(r"[0-9a-f]{64}", entry["normalized_lf_sha256"])
    # The inspector checks archived runtime against actual Git blob bytes;
    # no restoration or archived module import is performed.


def test_explicit_empty_prior_inventory_and_historical_unevaluated_scope(frozen):
    source = frozen.protocol["source_specification"]
    assert "prior_artifact_receipts" in source
    assert source["prior_artifact_receipts"] == ()
    assert frozen.inspection.prior_artifact_count == 0
    assert frozen.receipt["evaluation_status_at_freeze"] == "not_evaluated"
    assert frozen.protocol["evaluation_status_at_freeze"] == "not_evaluated"
    outcomes = {
        STEM + suffix for suffix in (".attempt.json", ".json", ".export-error.json")
    }
    assert outcomes.isdisjoint(frozen.names)
    assert set(frozen.inspection.existing_outcome_files) <= outcomes
    for path in frozen.inspection.existing_outcome_files:
        assert (ROOT / path).is_file()
    # Future outcomes do not change the historical no-evaluation-at-freeze fact.


def test_protocol_rebuilds_exact_sources_inputs_policy_and_static_prediction(frozen):
    protocol = frozen.protocol
    _same_exact_tree(protocol["producer_inputs"], neighbor_forward_inputs())
    _same_exact_tree(protocol["source_covers"], neighbor_forward_sources())
    _same_exact_tree(protocol["observation_policy"], neighbor_forward_policy())
    _same_exact_tree(
        protocol["static_prediction"], asdict(neighbor_forward_prediction())
    )
    inputs, source = protocol["producer_inputs"], protocol["source_covers"]
    assert source["classes"] == (1, 2, 1)
    assert (
        inputs["first_probe_node"],
        inputs["second_probe_node"],
        inputs["readout_node"],
    ) == (4, 22, 13)
    assert inputs["delay"] == 0 and inputs["total_duration"] == Q(1, 8)
    assert (
        inputs["first_probe_amplitude"]
        == inputs["second_probe_amplitude"]
        == Q(7, 10000)
    )
    assert inputs["time_step"] == Q(1, 128)
    assert type(inputs["order"]) is int and inputs["order"] == 16
    assert type(inputs["max_steps"]) is int and inputs["max_steps"] == 64
    assert 4 * inputs["total_duration"] / inputs["time_step"] == 64
    eps = Q(1, 10**32)
    for node in range(27):
        factor = Q(2 * source["classes"][node // 9] * (node % 9 - 4), 9)
        values = tuple(factor * bound for bound in source["pi_bounds"])
        target = min(values), max(values)
        assert source["reference_form_bounds"][node] == (0, 0)
        assert source["reference_phase_bounds"][node] == target
        assert source["actual_form_bounds"][node] == (-eps, eps)
        assert source["actual_phase_bounds"][node] == (target[0] - eps, target[1] + eps)


def test_full_law_common_source_and_numerical_premises_are_explicit(frozen):
    protocol = frozen.protocol
    source, attempt = (
        protocol["source_specification"],
        protocol["reserved_attempt_policy"],
    )
    assert source["node_order"] == tuple(range(27))
    assert source["state_order"] == "x[0:27], theta[0:27]"
    assert source["capacities"] == (Q(1),) * 27
    assert (
        source["source_coordinates"]
        == "pre-input form x and absolute continuous phase theta at 0-"
    )
    assert source["clock"] == "tau=(1023/1024)*t; both rows transformed"
    assert source["pressure_refresh"] == "continuous state-dependent full sine field"
    assert (
        source["history_source_association"]
        == "all four counterfactual histories share one complete actual source in class (1,2,1)"
    )
    assert (
        source["support"]
        == "three undirected unit-conductance C9 cycles with contacts (4,13) and (13,22)"
    )
    assert (
        source["forcing"]
        == "two simultaneous time-zero form jumps at nodes 4 and 22, selected per history; no continuous Gamma source"
    )
    assert source["seed"] is None
    assert (
        source["preparation_owner"]
        == "theory/nodal/SINE_CLASS_NONLINEAR_ORGANIZATION.md#sine-nonlinear-organization-source"
    )
    assert "zero-sum correlations" in source["retained_source"]
    assert "not a newly acquired source" in source["reference"]
    assert source["physical_identification"] == "not supplied or tested"
    for key, value in (
        ("maximum_attempts", 1),
        ("history_count", 4),
        ("positive_duration_segments", 4),
        ("steps_per_history", 16),
        ("maximum_unique_kernel_attempts", 64),
    ):
        assert type(attempt[key]) is int and attempt[key] == value
    assert attempt["adaptation_or_retry"] is False
    assert attempt["prediction_used_by_producer"] is False
    assert attempt["preserve_incomplete_response"] is True
    assert set(attempt["outcomes"]) == {
        "all_conditions_met",
        "unavailable",
        "consistency_conflict",
        "numerically_unresolved",
        "discrimination_unresolved",
        "execution_or_assessment_error",
    }
    environment = protocol["runtime_environment"]
    assert set(environment) == {"python", "implementation", "platform", "packages"}
    assert all(
        type(environment[key]) is str and environment[key]
        for key in ("python", "implementation", "platform")
    )
    assert set(environment["packages"]) == {"numpy", "scipy", "networkx", "mpmath"}
    assert all(
        type(value) is str and value for value in environment["packages"].values()
    )


def test_prospective_proof_is_an_unchanged_prefix_with_source_and_decision_scope(
    frozen,
):
    archived = frozen.retained[PROOF]
    entry = next(item for item in frozen.manifest["files"] if item["path"] == PROOF)
    assert len(archived) == entry["bytes"] and sha256_bytes(archived) == entry["sha256"]
    for anchor in (
        "sine-class-neighbor-forward-protocol",
        "sine-neighbor-forward-source",
        "sine-neighbor-forward-numerics",
        "sine-neighbor-forward-decisions",
    ):
        assert f'id="{anchor}"'.encode() in archived
    current = read_bytes_bounded(ROOT / PROOF, max_bytes=2 * 1024**2)
    assert current.replace(b"\r\n", b"\n").startswith(archived.replace(b"\r\n", b"\n"))


def _calls(node, name):
    return tuple(
        item
        for item in ast.walk(node)
        if isinstance(item, ast.Call)
        and isinstance(item.func, ast.Name)
        and item.func.id == name
    )


def test_evaluator_routes_only_primitive_inputs_after_exclusive_attempt(frozen):
    tree = ast.parse(frozen.retained[EVALUATOR])
    main = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == "main"
    )
    (producer,) = _calls(tree, "bound_sine_class_four_history_readout")
    assert producer.args == [] and len(producer.keywords) == 1
    assert producer.keywords[0].arg is None
    assert (
        isinstance(producer.keywords[0].value, ast.Name)
        and producer.keywords[0].value.id == "inputs"
    )
    input_assignment = next(
        item
        for item in main.body
        if isinstance(item, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "inputs"
            for target in item.targets
        )
    )
    assert ast.unparse(input_assignment.value) == "neighbor_forward_inputs()"
    assert len(_calls(main, "neighbor_forward_inputs")) == 1
    attempt = next(
        call
        for call in _calls(main, "write_json_once")
        if isinstance(call.args[0], ast.Name) and call.args[0].id == "attempt_path"
    )
    assert input_assignment.lineno < attempt.lineno < producer.lineno
    for name in (
        "inspect_frozen_source",
        "neighbor_forward_sources",
        "neighbor_forward_policy",
        "neighbor_forward_prediction",
    ):
        (call,) = _calls(main, name)
        assert call.lineno < attempt.lineno
    (assessment,) = _calls(main, "assess_neighbor_forward_report")
    assert producer.lineno < assessment.lineno
    protected = next(
        item
        for item in main.body
        if isinstance(item, ast.Try) and producer in tuple(ast.walk(item))
    )
    assert any(
        isinstance(handler.type, ast.Name) and handler.type.id == "BaseException"
        for handler in protected.handlers
    )
    freezer = ast.parse(frozen.retained[FREEZER])
    assert not _calls(freezer, "bound_sine_class_four_history_readout")
    assert any(
        isinstance(item, ast.If)
        and "__name__" in ast.unparse(item.test)
        and "__main__" in ast.unparse(item.test)
        for item in tree.body
    )


def test_frozen_preflight_and_failed_export_do_not_authorize_retry(frozen):
    tree = ast.parse(frozen.retained[EVALUATOR])
    main = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == "main"
    )
    attempt = next(
        call
        for call in _calls(main, "write_json_once")
        if isinstance(call.args[0], ast.Name) and call.args[0].id == "attempt_path"
    )
    preflight = "\n".join(
        ast.unparse(item) for item in main.body if item.lineno < attempt.lineno
    )
    for token in (
        "existing_outcome_files",
        "evaluator.read_bytes()",
        "git",
        "diff",
        "--quiet",
        "ls-files",
        "--others",
        "runtime_environment",
        "static_prediction",
    ):
        assert token in preflight
    writes = _calls(main, "write_json_once")
    assert len(writes) == 3
    export = next(
        item
        for item in main.body
        if isinstance(item, ast.Try)
        and any(
            ".export-error.json" in ast.unparse(call)
            for call in _calls(item, "write_json_once")
        )
    )
    rendered = ast.unparse(export)
    assert "'first_attempt_retained': True" in rendered
    assert "'retry_authorized': False" in rendered
    assert any(
        isinstance(item, ast.Raise)
        for handler in export.handlers
        for item in handler.body
    )
