"""Read the real prospective freeze without executing its reserved evaluator.

The shared inspector validates archive/base association. The prior prediction
reader rebuilds consumed intervals from retained coefficients; coefficient and
Picard generation remain execution premises, not authenticated by this audit.
"""

import ast
import re
import subprocess
import zipfile
from dataclasses import asdict
from fractions import Fraction as Q
from pathlib import Path
from types import SimpleNamespace

import pytest

from tnfr.research.artifact_io import (
    decode_exact_tree,
    encode_exact_tree,
    file_receipt,
    read_bytes_bounded,
    sha256_bytes,
)
from tnfr.research.frozen_source import inspect_frozen_source
from tnfr.research.sine_class_collective_forward import (
    PREDICTION_PATH,
    PREDICTION_SHA256,
    collective_forward_inputs,
    collective_forward_policy,
    collective_forward_sources,
    read_collective_prediction,
)
from tnfr.utils.io import json_dumps, json_loads

ROOT = Path(__file__).resolve().parents[2]
STEM = "docs/assets/sine_formed_classes/class-collective-forward-v1"
EVALUATOR = "build/class-collective-forward-freeze/evaluate_forward.py"
FREEZER = "build/class-collective-forward-freeze/freeze_forward.py"
PROOF = "theory/nodal/SINE_CLASS_COLLECTIVE_FORWARD_PROTOCOL.md"
BASE = "362124d03f34940e23b5d84ca29c6c839d6471a1"
RECEIPT_SHA256 = "919b2d79ba8beddce19664437d9d0c8738142e771c062878b98af986a38da5d2"
ARCHIVE_SHA256 = "a99b8f05153ebebe4964c7512183d606fb3d1e5317f133e20af57c28833204a3"
PROTOCOL_SHA256 = "e47ac667ecedf44d0863dd21a44da7deacc507a742a026cd0fe7d9bfd066f147"


@pytest.fixture(scope="module", autouse=True)
def no_scientific_regeneration():
    from tnfr.mathematics import _validated_taylor
    from tnfr.physics import (
        _sine_class_port_prediction,
        _sine_flow,
        _sine_formed_contact,
        relational_sine_class_comparison_readout,
        relational_sine_class_cubic_response,
        relational_sine_class_port_readout,
        relational_sine_class_readout,
    )
    from tnfr.research import frozen_source

    receipt = json_loads(
        read_bytes_bounded(ROOT / (STEM + ".freeze.json"), max_bytes=1024**2)
    )
    base = receipt["source_base_commit"]
    assert base == BASE and re.fullmatch(r"[0-9a-f]{40}", base)
    original_popen = subprocess.Popen

    def forbidden(*args, **kwargs):
        pytest.fail("freeze audit attempted a producer, restoration or worker")

    def read_only_git(args, *positional, **kwargs):
        command = list(args) if isinstance(args, (tuple, list)) else []
        allowed = command in (
            ["git", "rev-parse", "--show-toplevel"],
            ["git", "cat-file", "-t", base],
        )
        allowed |= (
            len(command) == 3
            and command[:2] == ["git", "show"]
            and command[2].startswith(base + ":src/")
        )
        assert allowed and not kwargs.get(
            "shell"
        ), "unexpected subprocess in freeze audit"
        assert Path(kwargs["cwd"]).resolve() == ROOT
        return original_popen(args, *positional, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        for module, names in (
            (
                _sine_class_port_prediction,
                (
                    "_predict_collective_port_response",
                    "_causal_coefficients",
                    "_kernel_coefficients",
                    "_causal_linear_series",
                    "_nonlinear_forcing",
                    "_grounded_series",
                ),
            ),
            (
                relational_sine_class_cubic_response,
                (
                    "bound_sine_class_cubic_response",
                    "_class_cubic_coefficients",
                    "_time_coefficients",
                    "_coefficient_segment",
                ),
            ),
            (
                _validated_taylor,
                ("validated_box_taylor_step", "flow_jets", "picard_tube"),
            ),
            (_sine_flow, ("_full_sine_field", "_sine_rate_evaluator")),
            (_sine_formed_contact, ("_unprobed_handoff",)),
            (relational_sine_class_readout, ("bound_sine_class_four_history_readout",)),
            (
                relational_sine_class_comparison_readout,
                ("bound_sine_class_comparison_readout",),
            ),
            (
                relational_sine_class_port_readout,
                (
                    "bound_sine_class_port_readout",
                    "_full_sine_field",
                    "validated_box_taylor_step",
                ),
            ),
            (frozen_source, ("restore_frozen_source",)),
        ):
            for name in names:
                patch.setattr(module, name, forbidden)
        patch.setattr(subprocess, "Popen", read_only_git)
        yield


@pytest.fixture(scope="module")
def frozen(no_scientific_regeneration):
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
        decode_exact_tree(json_loads(data)) for data in (receipt_raw, protocol_raw)
    )
    with zipfile.ZipFile(paths[2]) as archive:
        names = tuple(archive.namelist())
        sources = {
            name: archive.read(name)
            for name in (
                PROOF,
                EVALUATOR,
                FREEZER,
                "source-manifest.json",
                STEM + ".protocol.json",
            )
        }
    manifest = decode_exact_tree(json_loads(sources["source-manifest.json"]))
    yield SimpleNamespace(
        inspection=inspection,
        receipt=receipt,
        protocol=protocol,
        sources=sources,
        names=names,
        manifest=manifest,
        protocol_raw=protocol_raw,
    )
    assert tuple(file_receipt(path) for path in paths) == before


@pytest.fixture(scope="module")
def prior(no_scientific_regeneration):
    return read_collective_prediction(ROOT)


def _same_exact_tree(actual, expected):
    # Encoding before comparing preserves Boolean/integer and Fraction tags.
    assert json_dumps(encode_exact_tree(actual), sort_keys=True) == json_dumps(
        encode_exact_tree(expected), sort_keys=True
    )


def test_real_freeze_associates_git_runtime_protocol_and_immutable_inventory(frozen):
    inspection, receipt, protocol, manifest = (
        frozen.inspection,
        frozen.receipt,
        frozen.protocol,
        frozen.manifest,
    )
    assert receipt["schema"] == "tnfr.sine-class-collective-forward-freeze.v1"
    assert protocol["schema"] == "tnfr.sine-class-collective-forward-protocol.v1"
    assert manifest["schema"] == "tnfr.sine-class-collective-forward-source-snapshot.v1"
    assert (
        inspection.source_base_commit
        == receipt["source_base_commit"]
        == protocol["source_base_commit"]
        == manifest["source_base_commit"]
        == BASE
    )
    assert inspection.receipt_sha256 == RECEIPT_SHA256
    assert (
        tuple(receipt["runtime_overlays"])
        == tuple(protocol["runtime_overlays"])
        == tuple(manifest["runtime_overlays"])
        == ()
    )
    assert inspection.future_evaluator == receipt["future_evaluator"] == EVALUATOR
    assert inspection.archived_file_count == len(manifest["files"]) == 24
    assert frozen.sources[STEM + ".protocol.json"] == frozen.protocol_raw
    assert len(frozen.names) == len(set(frozen.names)) == len(manifest["files"]) + 1
    assert {item["path"] for item in receipt["artifacts"]} == {
        STEM + ".protocol.json",
        STEM + ".source.zip",
    }
    expected = {
        STEM + ".protocol.json": (122816, PROTOCOL_SHA256),
        STEM + ".source.zip": (100793, ARCHIVE_SHA256),
    }
    assert {
        item["path"]: (item["bytes"], item["sha256"]) for item in receipt["artifacts"]
    } == expected
    paths = {item["path"] for item in manifest["files"]}
    assert {
        "src/tnfr/physics/relational_sine_class_port_readout.py",
        "src/tnfr/physics/_sine_class_port_readout_evidence.py",
        "src/tnfr/physics/_sine_flow.py",
        "src/tnfr/mathematics/_validated_taylor.py",
        "src/tnfr/research/sine_class_collective_forward.py",
        EVALUATOR,
        PROOF,
    } <= paths
    for entry in manifest["files"]:
        if entry["path"].startswith("src/"):
            assert re.fullmatch(r"[0-9a-f]{64}", entry["git_blob_sha256"])
            assert re.fullmatch(r"[0-9a-f]{64}", entry["normalized_lf_sha256"])
    # inspect_frozen_source independently checks these against actual Git
    # blobs after only CRLF->LF normalization; it never restores or imports.


def test_historical_no_evaluation_scope_does_not_forbid_a_future_outcome(frozen):
    assert frozen.receipt["evaluation_status_at_freeze"] == "not_evaluated"
    assert frozen.protocol["evaluation_status_at_freeze"] == "not_evaluated"
    outcome_names = {
        STEM + suffix for suffix in (".attempt.json", ".json", ".export-error.json")
    }
    assert outcome_names.isdisjoint(frozen.names)
    assert set(frozen.inspection.existing_outcome_files) <= outcome_names
    for path in frozen.inspection.existing_outcome_files:
        assert (ROOT / path).is_file()
    # Existing outcomes are deliberately not required to be absent forever.


def test_canonical_absolute_phase_sources_and_reference_event_are_preserved(frozen):
    protocol = frozen.protocol
    _same_exact_tree(protocol["producer_inputs"], collective_forward_inputs())
    _same_exact_tree(protocol["source_covers"], collective_forward_sources())
    inputs, source = protocol["producer_inputs"], protocol["source_covers"]
    assert set(inputs) == {
        "initial_form_bounds",
        "initial_phase_bounds",
        "port_impulse",
        "horizon",
        "time_step",
        "order",
        "max_steps",
    }
    assert source["classes"] == ((1, 1, 1), (1, 2, 1))
    epsilon = Q(1, 10**32)
    for class_index, classes in enumerate(source["classes"]):
        assert source["reference_form_bounds"][class_index] == ((Q(0), Q(0)),) * 27
        assert source["actual_form_bounds"][class_index] == ((-epsilon, epsilon),) * 27
        for i in range(27):
            factor = Q(2 * classes[i // 9] * (i % 9 - 4), 9)
            exact = tuple(factor * p for p in source["pi_bounds"])
            expected = min(exact), max(exact)
            assert source["reference_phase_bounds"][class_index][i] == expected
            assert source["actual_phase_bounds"][class_index][i] == (
                expected[0] - epsilon,
                expected[1] + epsilon,
            )
    assert inputs["initial_form_bounds"] == source["reference_form_bounds"]
    assert inputs["initial_phase_bounds"] == source["reference_phase_bounds"]
    assert inputs["port_impulse"] == (0, Q(7, 10000), 0)
    assert inputs["horizon"] == 1 and inputs["time_step"] == Q(1, 16)
    assert type(inputs["order"]) is int and inputs["order"] == 12
    assert type(inputs["max_steps"]) is int and inputs["max_steps"] == 32
    assert 2 * inputs["horizon"] / inputs["time_step"] == inputs["max_steps"]


def test_policy_separates_source_noise_resolution_and_prediction(frozen):
    _same_exact_tree(frozen.protocol["observation_policy"], collective_forward_policy())
    policy = frozen.protocol["observation_policy"]
    assert policy["endpoint_radius"] == Q(1, 10**32)
    assert policy["gamma_upper"] == Q(1, 3000)
    assert policy["linear_source_allowance"] == Q(1, 10**32) / (1 - Q(2, 3000))
    assert policy["reference_radius_ceiling"] == Q(1, 10**12)
    assert policy["actual_prediction_resolution_allowance"] == Q(1, 10**10)
    assert policy["reading_error_per_model"] == Q(1, 10**8)
    assert (
        policy["nominal_prediction_comparison"]
        == "closed_interval_overlap_without_intersection"
    )
    assert (
        policy["actual_prediction_comparison"]
        == "whole_forward_band_in_prediction_plus_resolution_allowance"
    )
    assert (
        policy["separation"]
        == "forward_recorded_lower_strictly_above_grounded_recorded_upper"
    )


def test_source_premises_clock_and_attempt_policy_remain_explicit(frozen):
    protocol = frozen.protocol
    source = protocol["source_specification"]
    assert (
        source["source_coordinates"]
        == "pre-input form x and absolute continuous phase theta at 0-"
    )
    assert source["node_order"] == tuple(range(27))
    assert source["state_order"] == "x[0:27], theta[0:27]"
    assert source["capacities"] == (Q(1),) * 27
    assert source["clock"] == "tau=(1023/1024)*t; both rows transformed"
    assert source["pressure_refresh"] == "continuous state-dependent full sine field"
    assert (
        source["forcing"]
        == "one declared time-zero form jump; no continuous Gamma source"
    )
    assert source["seed"] is None
    assert (
        source["preparation_owner"]
        == "theory/nodal/SINE_CLASS_NONLINEAR_ORGANIZATION.md#sine-nonlinear-organization-source"
    )
    assert source["physical_identification"] == "not supplied or tested"
    assert "zero-sum correlations" in source["retained_source"]
    assert "not a newly acquired source" in source["reference"]
    attempt = protocol["reserved_attempt_policy"]
    assert type(attempt["maximum_attempts"]) is int and attempt["maximum_attempts"] == 1
    assert type(attempt["steps_per_class"]) is int and attempt["steps_per_class"] == 16
    assert attempt["adaptation_or_retry"] is False
    assert attempt["prediction_used_by_producer"] is False
    assert attempt["preserve_incomplete_response"] is True
    assert set(attempt["outcomes"]) == {
        "all_conditions_met",
        "unavailable",
        "complete_with_unmet_conditions",
        "execution_or_assessment_error",
    }
    assert set(attempt["classwise_outcomes"]) == {
        "all_conditions_met",
        "consistency_conflict",
        "numerically_unresolved",
        "resolution_not_met",
        "discrimination_unresolved",
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


def test_prior_receipt_and_consumed_prediction_are_readmitted(frozen, prior):
    _same_exact_tree(
        frozen.protocol["inspected_prediction"], tuple(asdict(item) for item in prior)
    )
    receipts = frozen.protocol["source_specification"]["prior_artifact_receipts"]
    matching = tuple(item for item in receipts if item["path"] == PREDICTION_PATH)
    assert len(matching) == 1
    assert (
        matching[0]["sha256"]
        == PREDICTION_SHA256
        == "01b443211665c027bcd5b86ace8a22a4e7ae051f34d172d5520644e6225ba16a"
    )
    assert matching[0]["bytes"] == 1016399
    assert frozen.inspection.prior_artifact_count == len(receipts) == 1
    assert tuple(item.mediator_class for item in prior) == (1, 2)
    assert all(item.numerical_radius < Q(1, 10**12) for item in prior)


def test_archived_prospective_owner_is_a_preserved_live_text_prefix(frozen):
    archived = frozen.sources[PROOF]
    entry = next(item for item in frozen.manifest["files"] if item["path"] == PROOF)
    assert len(archived) == entry["bytes"]
    assert sha256_bytes(archived) == entry["sha256"]
    assert b'id="sine-class-collective-forward-protocol"' in archived
    assert b'id="sine-collective-forward-source-transfer"' in archived
    assert b'id="sine-collective-forward-stopping-criteria"' in archived
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


def test_archived_evaluator_has_one_primitive_only_guarded_attempt(frozen):
    tree = ast.parse(frozen.sources[EVALUATOR])
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    (producer,) = _calls(tree, "bound_sine_class_port_readout")
    assert producer in tuple(ast.walk(main))
    assert producer.args == [] and len(producer.keywords) == 1
    assert (
        producer.keywords[0].arg is None
        and isinstance(producer.keywords[0].value, ast.Name)
        and producer.keywords[0].value.id == "inputs"
    )
    (inputs_call,) = _calls(main, "collective_forward_inputs")
    attempt = next(
        call
        for call in _calls(main, "write_json_once")
        if isinstance(call.args[0], ast.Name) and call.args[0].id == "attempt_path"
    )
    assert inputs_call.lineno < attempt.lineno < producer.lineno
    checks = (
        "inspect_frozen_source",
        "read_collective_prediction",
        "collective_forward_sources",
        "collective_forward_policy",
    )
    for name in checks:
        (call,) = _calls(main, name)
        assert call.lineno < attempt.lineno
    protected = next(
        node
        for node in main.body
        if isinstance(node, ast.Try) and producer in tuple(ast.walk(node))
    )
    assert any(
        isinstance(handler.type, ast.Name) and handler.type.id == "BaseException"
        for handler in protected.handlers
    )
    (assess,) = _calls(main, "assess_collective_forward_report")
    assert producer.lineno < assess.lineno
    assert any(
        isinstance(node, ast.If)
        and "__name__" in ast.unparse(node.test)
        and "__main__" in ast.unparse(node.test)
        for node in tree.body
    )
    # This is source inspection, never an import/exec of the archived helper.


def test_archived_runtime_and_export_guards_precede_or_preserve_first_attempt(frozen):
    tree = ast.parse(frozen.sources[EVALUATOR])
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    attempt = next(
        call
        for call in _calls(main, "write_json_once")
        if isinstance(call.args[0], ast.Name) and call.args[0].id == "attempt_path"
    )
    before = "\n".join(
        ast.unparse(node) for node in main.body if node.lineno < attempt.lineno
    )
    for required in (
        "existing_outcome_files",
        "evaluator.read_bytes()",
        "git",
        "diff",
        "--quiet",
        "ls-files",
        "--others",
        "runtime_environment",
    ):
        assert required in before
    writes = _calls(main, "write_json_once")
    assert len(writes) == 3
    assert any(".export-error.json" in ast.unparse(call) for call in writes)
    export = next(
        node
        for node in main.body
        if isinstance(node, ast.Try)
        and any(
            ".export-error.json" in ast.unparse(call)
            for call in _calls(node, "write_json_once")
        )
    )
    rendered = ast.unparse(export)
    assert "'first_attempt_retained': True" in rendered
    assert "'retry_authorized': False" in rendered
    assert any(
        isinstance(node, ast.Raise)
        for handler in export.handlers
        for node in handler.body
    )
    assert len(_calls(tree, "bound_sine_class_port_readout")) == 1
