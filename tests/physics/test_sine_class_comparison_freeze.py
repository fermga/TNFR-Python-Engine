"""Read the frozen comparison association without evaluating either class.

Archive/base integrity belongs to the shared frozen-source inspector. Only
three selected helper definitions are compiled for synthetic export controls;
no archived imports, main, source acquisition or scientific response executes.
"""

import ast
import subprocess
import zipfile
from fractions import Fraction as Q
from pathlib import Path

import pytest

from tnfr.physics._sine_class_contrast import _contrast_decision
from tnfr.research.artifact_io import (
    decode_exact_tree,
    encode_exact_tree,
    file_receipt,
    write_json_once,
)
from tnfr.research.frozen_source import inspect_frozen_source
from tnfr.research.sine_class_comparison_protocol import (
    canonical_comparison_sources,
    comparison_observation_policy,
    comparison_producer_inputs,
)
from tnfr.sdk.relational_reports import _project
from tnfr.utils.io import json_dumps, json_loads

ROOT = Path(__file__).resolve().parents[2]
STEM = "docs/assets/sine_formed_classes/class-comparison-readout-v1"
BASE = "8e40dac013c026c0f198a2f95131512277286bbf"
ARCHIVE_SHA256 = "564642f21fea44929f53463e2391a10e569bfe81b72e1c4e964eb3c5610424f7"
HELPERS = "build/class-comparison-freeze"
PROOF = "theory/nodal/SINE_CLASS_COMPARISON_PROTOCOL.md"


@pytest.fixture(scope="module", autouse=True)
def no_scientific_execution():
    from tnfr.mathematics import _validated_taylor
    from tnfr.physics import (
        _sine_formed_contact,
        relational_sine_class_comparison_readout,
        relational_sine_class_cubic_response,
        relational_sine_class_mediation,
        relational_sine_class_readout,
    )
    from tnfr.research import frozen_source

    def forbidden(*args, **kwargs):
        pytest.fail("freeze audit must not execute a response, coefficient or worker")

    original_popen = subprocess.Popen

    def read_only_git(args, *positional, **kwargs):
        # The shared inspector needs these Git reads, but never a worktree,
        # Python worker, dependency installation or archived evaluator.
        command = list(args) if isinstance(args, (list, tuple)) else []
        allowed = command == ["git", "rev-parse", "--show-toplevel"] or command == [
            "git",
            "cat-file",
            "-t",
            BASE,
        ]
        allowed |= (
            len(command) == 3
            and command[:2] == ["git", "show"]
            and command[2].startswith(BASE + ":src/")
        )
        assert allowed and not kwargs.get("shell"), "unexpected subprocess in audit"
        assert Path(kwargs["cwd"]).resolve() == ROOT
        return original_popen(args, *positional, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        for module, names in (
            (
                _validated_taylor,
                ("validated_box_taylor_step", "flow_jets", "picard_tube"),
            ),
            (_sine_formed_contact, ("_unprobed_handoff",)),
            (relational_sine_class_mediation, ("assess_sine_class_mediation",)),
            (relational_sine_class_cubic_response, ("_class_cubic_coefficients",)),
            (
                relational_sine_class_readout,
                (
                    "bound_sine_class_four_history_readout",
                    "_full_sine_field",
                    "validated_box_taylor_step",
                ),
            ),
            (
                relational_sine_class_comparison_readout,
                (
                    "bound_sine_class_comparison_readout",
                    "bound_sine_class_four_history_readout",
                ),
            ),
            (frozen_source, ("restore_frozen_source",)),
        ):
            for name in names:
                patch.setattr(module, name, forbidden)
        patch.setattr(subprocess, "Popen", read_only_git)
        yield


@pytest.fixture(scope="module")
def frozen(no_scientific_execution):
    paths = tuple(
        ROOT / (STEM + suffix)
        for suffix in (".freeze.json", ".protocol.json", ".source.zip")
    )
    before = tuple(file_receipt(path) for path in paths)
    inspection = inspect_frozen_source(ROOT, STEM + ".freeze.json")
    receipt, protocol = (json_loads(path.read_bytes()) for path in paths[:2])
    with zipfile.ZipFile(paths[2]) as archive:
        sources = {
            name: archive.read(name)
            for name in (
                PROOF,
                HELPERS + "/freeze_once.py",
                HELPERS + "/evaluate_comparison.py",
            )
        }
        names = tuple(archive.namelist())
    yield inspection, receipt, protocol, sources, names
    assert tuple(file_receipt(path) for path in paths) == before


@pytest.fixture(scope="module")
def pure(frozen):
    source = frozen[3][HELPERS + "/evaluate_comparison.py"]
    selected = {"_same_exact_tree", "_project_assessment", "_retain_outcome"}
    definitions = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name in selected
    ]
    assert {node.name for node in definitions} == selected
    namespace = {
        "encode_exact_tree": encode_exact_tree,
        "json_dumps": json_dumps,
        "_project": _project,
        "write_json_once": write_json_once,
    }
    exec(
        compile(
            ast.Module(body=definitions, type_ignores=[]), "frozen pure helpers", "exec"
        ),
        namespace,
    )
    return namespace


def test_shared_inspection_admits_complete_pinned_association(frozen):
    inspection, receipt, protocol, _, names = frozen
    assert inspection.source_base_commit == receipt["source_base_commit"] == BASE
    assert protocol["source_base_commit"] == BASE
    assert inspection.archived_file_count == 301
    assert inspection.prior_artifact_count == 1
    assert inspection.future_evaluator == HELPERS + "/evaluate_comparison.py"
    archive_receipt = next(
        item for item in receipt["artifacts"] if item["path"].endswith(".zip")
    )
    assert archive_receipt["bytes"] == 1433580
    assert archive_receipt["sha256"] == ARCHIVE_SHA256
    assert receipt["evaluation_status_at_freeze"] == "not_evaluated"
    assert protocol["evaluation_status_at_freeze"] == "not_evaluated"
    assert receipt["runtime_overlays"] == protocol["runtime_overlays"] == []
    # Historical admission remains true after a separate future evaluation.
    # An outcome can never be installed by restoring the source snapshot.
    for suffix in (".attempt.json", ".json", ".export-error.json"):
        assert STEM + suffix not in names


def test_frozen_primitives_match_fresh_canonical_source_and_policy(frozen, pure):
    protocol = frozen[2]
    pure["_same_exact_tree"](protocol["producer_inputs"], comparison_producer_inputs())
    pure["_same_exact_tree"](
        protocol["observation_policy"], comparison_observation_policy()
    )
    specification = protocol["source_specification"]
    pure["_same_exact_tree"](specification["covers"], canonical_comparison_sources())
    decoded = decode_exact_tree(protocol)
    source = decoded["source_specification"]
    assert source["reached_component_error_radius"] == Q(1, 10**32)
    assert source["formation_source_error_radius"] == Q(1, 10**10)
    assert source["reference_is_not_a_preparation_or_hidden_state_reset"] is True
    assert source["source_transfer_owner"] == PROOF + "#sine-comparison-source-transfer"
    assert decoded["intervention_scale"] == Q(7, 5)
    assert decoded["producer_inputs"]["max_steps"] == 1536
    assert decoded["seed"] is None
    assert set(decoded["runtime_dependencies"]) == {"numpy", "networkx"}
    assert all(
        type(value) is str and value
        for value in decoded["runtime_dependencies"].values()
    )
    assert decoded["runtime_python"] and decoded["runtime_implementation"]


def test_prospective_proof_prefix_is_preserved(frozen):
    prospective = frozen[3][PROOF]
    assert len(prospective) == 17358 and len(prospective.splitlines()) == 362
    # Git's newline conversion does not change the mathematical content.
    normalized = prospective.replace(b"\r\n", b"\n")
    assert (ROOT / PROOF).read_bytes().replace(b"\r\n", b"\n").startswith(normalized)


def _calls(node, name):
    return [
        call
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == name
    ]


def test_archived_main_retains_single_attempt_and_guarded_export(frozen):
    sources = frozen[3]
    evaluator = ast.parse(sources[HELPERS + "/evaluate_comparison.py"])
    freezer = ast.parse(sources[HELPERS + "/freeze_once.py"])
    name = "bound_sine_class_comparison_readout"
    assert not _calls(freezer, name)
    (producer,) = _calls(evaluator, name)
    main = next(
        node
        for node in evaluator.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    (attempt,) = _calls(main, "write_json_once")
    assert ".attempt.json" in ast.unparse(attempt)
    protected = next(node for node in main.body if isinstance(node, ast.Try))
    assert producer in tuple(ast.walk(protected))
    assert attempt.lineno < protected.lineno
    assert any(
        isinstance(handler.type, ast.Name) and handler.type.id == "BaseException"
        for handler in protected.handlers
    )
    assert len(_calls(protected, "_project_assessment")) == 2
    (retained,) = _calls(main, "_retain_outcome")
    assert retained.lineno > protected.end_lineno
    assert len(_calls(main, "_same_exact_tree")) == 3
    for guard in (node for node in main.body if isinstance(node, ast.If)):
        if "runtime_dependencies" in ast.unparse(guard.test):
            assert guard.lineno < attempt.lineno
            assert "runtime_python" in ast.unparse(guard.test)
            assert "runtime_implementation" in ast.unparse(guard.test)
            assert any(isinstance(node, ast.Raise) for node in guard.body)
            break
    else:
        pytest.fail("frozen runtime guard missing before the attempt")
    assert ".export-error.json" in ast.unparse(freezer)


def test_exact_tree_keeps_fraction_types_and_optional_metadata(pure):
    expected = {
        "count": 1,
        "value": Q(1, 7),
        "box": (Q(-1, 2), Q(3, 4)),
        "flag": True,
        "missing": None,
    }
    pure["_same_exact_tree"](encode_exact_tree(expected), expected)


@pytest.mark.parametrize("alias", (True, 1.0, {"numerator": True, "denominator": 1}))
def test_exact_tree_rejects_python_equality_aliases(pure, alias):
    with pytest.raises(ValueError, match="primitive policy"):
        pure["_same_exact_tree"]({"count": alias}, {"count": 1})


def test_assessment_projection_keeps_unavailable_none(pure):
    assert pure["_project_assessment"](None) is None


def test_assessment_dictionary_projects_exact_decision_and_bounds(pure):
    decision = _contrast_decision((Q(-9), Q(-8)), Q(1, 4), Q(1, 100))
    assessment = {
        "decision": decision,
        "bounds": (Q(-9), Q(-8)),
        "width": Q(1),
        "missing": None,
        "flag": True,
    }
    projected = pure["_project_assessment"](assessment)
    decoded = decode_exact_tree(json_loads(json_dumps(projected)))
    assert decoded["bounds"] == assessment["bounds"]
    assert decoded["width"] == 1
    assert decoded["decision"]["true_bounds"] == decision.true_bounds
    assert decoded["missing"] is None and decoded["flag"] is True


@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
def test_assessment_projection_rejects_nonfinite_metadata(pure, value):
    with pytest.raises((ValueError, TypeError)):
        pure["_project_assessment"]({"value": value})


def test_successful_outcome_export_is_exact_and_exclusive(pure, tmp_path):
    output, error = tmp_path / "outcome.json", tmp_path / "export-error.json"
    assert pure["_retain_outcome"](output, {"value": Q(2, 7)}, error)
    assert decode_exact_tree(json_loads(output.read_bytes())) == {"value": Q(2, 7)}
    assert not error.exists()


def test_existing_outcome_is_never_replaced(pure, tmp_path):
    output, error = tmp_path / "outcome.json", tmp_path / "export-error.json"
    original = b'{"retained":"first outcome"}\n'
    output.write_bytes(original)
    assert not pure["_retain_outcome"](output, {"replacement": Q(9)}, error)
    assert output.read_bytes() == original
    failure = json_loads(error.read_bytes())
    assert failure["exception_type"] == "FileExistsError"
    assert failure["outcome_file_exists"] is True
    assert failure["retained_outcome_bytes"] == len(original)


def test_serialization_failure_retains_separate_unavailable_export(pure, tmp_path):
    output, error = tmp_path / "outcome.json", tmp_path / "export-error.json"
    assert not pure["_retain_outcome"](output, {"invalid": object()}, error)
    assert not output.exists()
    failure = json_loads(error.read_bytes())
    assert failure["exception_type"] == "TypeError"
    assert failure["outcome_file_exists"] is False
    assert failure["retained_outcome_bytes"] is None


@pytest.mark.parametrize("failure_type", (OSError, KeyboardInterrupt))
def test_partial_write_or_interrupt_preserves_bytes(
    pure, tmp_path, monkeypatch, failure_type
):
    output, error = tmp_path / "outcome.json", tmp_path / "export-error.json"
    partial = b'{"partial":'

    def interrupted_write(path, payload):
        if path == output:
            with path.open("xb") as stream:
                stream.write(partial)
            raise failure_type("synthetic export interruption")
        return write_json_once(path, payload)

    monkeypatch.setitem(pure, "write_json_once", interrupted_write)
    assert not pure["_retain_outcome"](output, {"value": Q(1)}, error)
    assert output.read_bytes() == partial
    failure = json_loads(error.read_bytes())
    assert failure["exception_type"] == failure_type.__name__
    assert failure["outcome_file_exists"] is True
    assert failure["retained_outcome_bytes"] == len(partial)


def test_existing_export_error_cannot_be_replaced(pure, tmp_path):
    output, error = tmp_path / "outcome.json", tmp_path / "export-error.json"
    output.write_bytes(b"partial first outcome")
    error.write_bytes(b"retained first export error")
    before = output.read_bytes(), error.read_bytes()
    with pytest.raises(FileExistsError):
        pure["_retain_outcome"](output, {"replacement": 1}, error)
    assert (output.read_bytes(), error.read_bytes()) == before
