"""Synthetic evaluator wiring; no selected scientific producer is executed.

Pinned archive and member bytes are admitted before parsing or compilation.
Only function definitions are compiled from the frozen evaluator. Main runs
solely against unrelated temporary records and mocked scientific entry points.
Archived imports, source restoration and real subprocesses are never executed.
"""

import ast
import io
import logging
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
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
    write_json_once,
)
from tnfr.utils.io import json_dumps, json_loads

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = (
    ROOT / "docs/assets/sine_formed_classes/class-collective-forward-v1.source.zip"
)
DRIVER = "build/class-collective-forward-freeze/evaluate_forward.py"
STEM = "docs/assets/synthetic-forward"
ARCHIVE_BYTES = 100793
ARCHIVE_SHA256 = "a99b8f05153ebebe4964c7512183d606fb3d1e5317f133e20af57c28833204a3"
DRIVER_BYTES = 6170
DRIVER_SHA256 = "9da61efce1e3a67e738ee1c3cc8384ce13f1134bdefd3dfec0e0915b3ffbf582"


def _compile_evaluator(path):
    # This owner must be safe when selected without the separate freeze audit.
    # Hash the same bounded bytes that ZipFile consumes, before opening it.
    data = read_bytes_bounded(path, max_bytes=ARCHIVE_BYTES)
    if len(data) != ARCHIVE_BYTES or sha256_bytes(data) != ARCHIVE_SHA256:
        raise ValueError("frozen evaluator archive association differs")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        with archive.open(DRIVER) as stream:
            source = stream.read(DRIVER_BYTES + 1)
    if len(source) != DRIVER_BYTES or sha256_bytes(source) != DRIVER_SHA256:
        raise ValueError("frozen evaluator member association differs")
    tree = ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    assert {node.name for node in functions} == {
        "_canonical",
        "_runtime_environment",
        "main",
    }
    return (
        source,
        tree,
        compile(
            ast.Module(body=functions, type_ignores=[]),
            "synthetic evaluator definitions",
            "exec",
        ),
    )


@pytest.fixture(scope="module")
def evaluator_source():
    return _compile_evaluator(ARCHIVE)


@pytest.mark.parametrize("size_delta", [-1, 0, 1])
def test_unassociated_archive_is_rejected_before_open_or_parse(
    tmp_path, monkeypatch, size_delta
):
    path = tmp_path / "unassociated.zip"
    path.write_bytes(b"x" * (ARCHIVE_BYTES + size_delta))

    def forbidden(*args, **kwargs):
        pytest.fail("unassociated archive reached ZIP opening or AST parsing")

    monkeypatch.setattr(zipfile, "ZipFile", forbidden)
    monkeypatch.setattr(ast, "parse", forbidden)
    with pytest.raises(
        ValueError, match="association differs|exceeds audit byte budget"
    ):
        _compile_evaluator(path)


@dataclass
class _Report:
    value: object = Q(1, 7)


@dataclass
class _Prediction:
    value: Q = Q(2, 7)


@pytest.fixture
def harness(evaluator_source, tmp_path, monkeypatch):
    from tnfr.physics import relational_sine_class_port_readout as producer
    from tnfr.research import sine_class_collective_forward as policy

    source, _, compiled = evaluator_source
    root = tmp_path / "synthetic"
    evaluator = root / DRIVER
    evaluator.parent.mkdir(parents=True)
    evaluator.write_bytes(source)
    (root / "docs/assets").mkdir(parents=True)
    with zipfile.ZipFile(root / (STEM + ".source.zip"), "w") as archive:
        archive.writestr(DRIVER, source)
    inputs = {"primitive": Q(1, 7)}
    covers = {"absolute_phase": (Q(-2), Q(2))}
    observation = {"error": Q(1, 100)}
    prediction = (_Prediction(), _Prediction(Q(3, 7)))
    environment = {
        "python": "synthetic",
        "implementation": "synthetic",
        "platform": "synthetic",
        "packages": {name: "0" for name in ("numpy", "scipy", "networkx", "mpmath")},
    }
    protocol = dict(
        producer_inputs=inputs,
        source_covers=covers,
        observation_policy=observation,
        inspected_prediction=tuple(asdict(item) for item in prediction),
        runtime_environment=environment,
    )
    protocol_path = root / (STEM + ".protocol.json")

    def save_protocol():
        protocol_path.write_text(
            json_dumps(encode_exact_tree(protocol)), encoding="utf8"
        )

    save_protocol()
    calls = []
    result = SimpleNamespace(
        report=_Report(), producer_error=None, assessment_error=None
    )
    paths = {
        key: root / (STEM + suffix)
        for key, suffix in (
            ("attempt", ".attempt.json"),
            ("outcome", ".json"),
            ("export", ".export-error.json"),
        )
    }

    def inspect(where, receipt):
        assert where == root and receipt == STEM + ".freeze.json"
        return SimpleNamespace(
            existing_outcome_files=tuple(
                path.name for path in paths.values() if path.exists()
            ),
            future_evaluator=DRIVER,
            source_base_commit="a" * 40,
            receipt_sha256="b" * 64,
        )

    def run_git(command, *, cwd, check):
        assert command == ["git", "diff", "--quiet", "a" * 40, "--", "src"]
        assert cwd == root and check is True
        calls.append("git_diff")

    def read_git(command, *, cwd):
        assert command == [
            "git",
            "ls-files",
            "--others",
            "--exclude-standard",
            "--",
            "src",
        ]
        assert cwd == root
        calls.append("git_untracked")
        return b""

    def fake_producer(**packet):
        assert paths["attempt"].is_file(), "scientific call preceded exclusive attempt"
        assert packet == inputs and set(packet) == {"primitive"}
        calls.append("synthetic_producer")
        if result.producer_error is not None:
            raise result.producer_error
        return result.report

    def fake_assessment(report, prior):
        assert report is result.report and prior is prediction
        calls.append("synthetic_assessment")
        if result.assessment_error is not None:
            raise result.assessment_error
        return {"status": "synthetic_complete", "evidence": _Report(Q(3, 11))}

    monkeypatch.setattr(producer, "bound_sine_class_port_readout", fake_producer)
    monkeypatch.setattr(policy, "collective_forward_inputs", lambda: inputs)
    monkeypatch.setattr(policy, "collective_forward_sources", lambda: covers)
    monkeypatch.setattr(policy, "collective_forward_policy", lambda: observation)
    monkeypatch.setattr(policy, "read_collective_prediction", lambda where: prediction)
    monkeypatch.setattr(policy, "assess_collective_forward_report", fake_assessment)
    namespace = {
        "ROOT": root,
        "__file__": str(evaluator),
        "STEM": STEM,
        "Path": Path,
        "zipfile": zipfile,
        "asdict": asdict,
        "datetime": datetime,
        "timezone": timezone,
        "logging": logging,
        "time": time,
        "json_dumps": json_dumps,
        "json_loads": json_loads,
        "encode_exact_tree": encode_exact_tree,
        "decode_exact_tree": decode_exact_tree,
        "read_bytes_bounded": read_bytes_bounded,
        "write_json_once": write_json_once,
        "file_receipt": file_receipt,
        "inspect_frozen_source": inspect,
        "subprocess": SimpleNamespace(run=run_git, check_output=read_git),
    }
    exec(compiled, namespace)
    namespace["_runtime_environment"] = lambda: environment
    return SimpleNamespace(
        namespace=namespace,
        main=namespace["main"],
        root=root,
        evaluator=evaluator,
        protocol=protocol,
        save_protocol=save_protocol,
        paths=paths,
        calls=calls,
        result=result,
    )


def _read(path):
    return decode_exact_tree(json_loads(path.read_bytes()))


def test_archived_control_flow_has_one_guarded_call_after_exclusive_attempt(
    evaluator_source,
):
    _, tree, _ = evaluator_source
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    calls = [node for node in ast.walk(main) if isinstance(node, ast.Call)]
    scientific = [
        node
        for node in calls
        if isinstance(node.func, ast.Name)
        and node.func.id == "bound_sine_class_port_readout"
    ]
    assert len(scientific) == 1
    attempt = next(
        node
        for node in calls
        if isinstance(node.func, ast.Name)
        and node.func.id == "write_json_once"
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "attempt_path"
    )
    protected = next(node for node in main.body if isinstance(node, ast.Try))
    assert scientific[0] in tuple(ast.walk(protected))
    assert attempt.lineno < protected.lineno
    assert any(
        isinstance(handler.type, ast.Name) and handler.type.id == "BaseException"
        for handler in protected.handlers
    )
    assert not any(
        isinstance(node, (ast.While, ast.For)) for node in ast.walk(protected)
    )


def test_synthetic_success_retains_exact_attempt_and_outcome(harness):
    harness.main()
    attempt, outcome = (_read(harness.paths[key]) for key in ("attempt", "outcome"))
    assert attempt["attempt_policy"] == "one_fixed_attempt_no_retry_or_adaptation"
    assert attempt["source_base_commit"] == "a" * 40
    assert outcome["status"] == "synthetic_complete"
    assert outcome["report"] == {"value": Q(1, 7)}
    assert outcome["assessment"]["evidence"] == {"value": Q(3, 11)}
    assert harness.calls.count("synthetic_producer") == 1
    assert not harness.paths["export"].exists()
    before = harness.paths["outcome"].read_bytes()
    with pytest.raises(FileExistsError):
        harness.main()
    assert harness.paths["outcome"].read_bytes() == before
    assert harness.calls.count("synthetic_producer") == 1


@pytest.mark.parametrize("key", ["attempt", "outcome", "export"])
def test_each_existing_record_blocks_before_preflight_or_producer(harness, key):
    harness.paths[key].write_bytes(b"retained")
    with pytest.raises(FileExistsError):
        harness.main()
    assert harness.calls == []
    assert harness.paths[key].read_bytes() == b"retained"


@pytest.mark.parametrize(
    "fault",
    [
        "self_path",
        "self_bytes",
        "dirty_source",
        "untracked_source",
        "runtime",
        "inputs",
        "covers",
        "policy",
        "prediction",
    ],
)
def test_preflight_mismatch_creates_no_attempt(harness, fault):
    if fault == "self_path":
        harness.namespace["__file__"] = str(harness.root / "unrelated.py")
    elif fault == "self_bytes":
        harness.evaluator.write_bytes(b"changed evaluator")
    elif fault == "dirty_source":

        def fail(*args, **kwargs):
            raise ValueError("synthetic dirty source")

        harness.namespace["subprocess"].run = fail
    elif fault == "untracked_source":
        harness.namespace["subprocess"].check_output = (
            lambda *a, **k: b"src/unrecorded.py\n"
        )
    else:
        key = {
            "runtime": "runtime_environment",
            "inputs": "producer_inputs",
            "covers": "source_covers",
            "policy": "observation_policy",
            "prediction": "inspected_prediction",
        }[fault]
        harness.protocol[key] = {"unmatched": True}
        harness.save_protocol()
    with pytest.raises(ValueError):
        harness.main()
    assert not any(path.exists() for path in harness.paths.values())
    assert "synthetic_producer" not in harness.calls


@pytest.mark.parametrize("alias", [True, 1.0])
def test_canonical_preflight_rejects_equality_aliases(harness, alias):
    harness.protocol["producer_inputs"] = {"primitive": alias}
    harness.save_protocol()
    assert harness.namespace["_canonical"](alias) != harness.namespace["_canonical"](1)
    with pytest.raises(ValueError, match="association differs"):
        harness.main()
    assert not harness.paths["attempt"].exists()


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("synthetic failure"),
        KeyboardInterrupt("synthetic interrupt"),
        SystemExit("synthetic stop"),
    ],
)
def test_scientific_failure_retains_first_attempt_without_retry(harness, error):
    harness.result.producer_error = error
    harness.main()
    outcome = _read(harness.paths["outcome"])
    assert outcome["status"] == "execution_or_assessment_error"
    assert outcome["error_type"] == type(error).__name__
    assert outcome["report"] is None
    assert outcome["report_returned"] is outcome["report_projected"] is False
    assert harness.calls.count("synthetic_producer") == 1


def test_assessment_failure_preserves_returned_report(harness):
    harness.result.assessment_error = ValueError("synthetic assessment rejection")
    harness.main()
    outcome = _read(harness.paths["outcome"])
    assert outcome["report"] == {"value": Q(1, 7)}
    assert outcome["report_returned"] is outcome["report_projected"] is True
    assert outcome["error_type"] == "ValueError"


def test_report_projection_failure_is_not_repeated_in_error_handler(harness):
    class ProjectionFailure:
        def __deepcopy__(self, memo):
            raise ValueError("synthetic report projection failure")

    harness.result.report = _Report(ProjectionFailure())
    harness.main()
    outcome = _read(harness.paths["outcome"])
    assert outcome["report"] is None
    assert outcome["report_returned"] is True
    assert outcome["report_projected"] is False
    assert outcome["error_type"] == "ValueError"


def test_serialization_failure_preserves_attempt_and_separate_error(harness):
    harness.result.report = _Report(object())
    with pytest.raises(TypeError):
        harness.main()
    assert harness.paths["attempt"].exists()
    assert not harness.paths["outcome"].exists()
    failure = _read(harness.paths["export"])
    assert failure["error_type"] == "TypeError"
    assert failure["first_attempt_retained"] is True
    assert failure["retry_authorized"] is False
    assert "report" not in failure


@pytest.mark.parametrize("error_type", [OSError, KeyboardInterrupt])
def test_partial_export_remains_untouched_and_has_separate_failure(harness, error_type):
    partial = b'{"partial":'

    def interrupted(path, payload):
        if path == harness.paths["outcome"]:
            with path.open("xb") as stream:
                stream.write(partial)
            raise error_type("synthetic export interruption")
        return write_json_once(path, payload)

    harness.namespace["write_json_once"] = interrupted
    with pytest.raises(error_type):
        harness.main()
    assert harness.paths["outcome"].read_bytes() == partial
    failure = _read(harness.paths["export"])
    assert failure["error_type"] == error_type.__name__
    assert failure["retry_authorized"] is False
    assert harness.paths["attempt"].exists()
