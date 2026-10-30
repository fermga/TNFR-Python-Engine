"""Synthetic freeze/evaluation lifecycle controls, never a receiver trajectory.

Only function definitions are compiled from the supplemental driver. Before
the first freeze, use its prepared local source; afterward read the associated
archive bytes. Scientific entry points, runtime inspection and Git are mocked.
"""

import ast
import io
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
REAL_STEM = "docs/assets/sine_formed_classes/organized-receiver-enclosure-v1"
DRIVER = "build/organized-receiver-enclosure-freeze/evaluate_receiver.py"
FREEZER = "build/organized-receiver-enclosure-freeze/freeze_receiver.py"
STEM = "docs/assets/synthetic-receiver"


def _driver_source(name):
    archive_path = ROOT / (REAL_STEM + ".source.zip")
    if not archive_path.exists():
        return read_bytes_bounded(ROOT / name, max_bytes=64 * 1024)
    receipt = json_loads(
        read_bytes_bounded(ROOT / (REAL_STEM + ".freeze.json"), max_bytes=65536)
    )
    association = next(
        item
        for item in receipt["artifacts"]
        if item["path"] == REAL_STEM + ".source.zip"
    )
    data = read_bytes_bounded(archive_path, max_bytes=8 * 1024**2)
    assert len(data) == association["bytes"]
    assert sha256_bytes(data) == association["sha256"]
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        with archive.open(name) as stream:
            source = stream.read(65537)
    assert len(source) <= 65536
    return source


def _compile_functions(source):
    tree = ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    return compile(
        ast.Module(body=functions, type_ignores=[]), "synthetic driver", "exec"
    )


@dataclass
class _Source:
    release_bound: Q = Q(1, 70)


@pytest.fixture(scope="module")
def driver_source():
    return _driver_source(DRIVER)


@pytest.fixture
def harness(driver_source, tmp_path, monkeypatch):
    from tnfr.research import organized_receiver_enclosure as owner

    root = tmp_path / "synthetic"
    evaluator = root / DRIVER
    evaluator.parent.mkdir(parents=True)
    evaluator.write_bytes(driver_source)
    (root / "docs/assets").mkdir(parents=True)
    with zipfile.ZipFile(root / (STEM + ".source.zip"), "w") as archive:
        archive.writestr(DRIVER, driver_source)
    policy = {"step": Q(1, 8), "maximum_steps": 2}
    source = _Source()
    environment = {"python": "synthetic", "packages": {"mpmath": "synthetic"}}
    protocol = {
        "runtime_environment": environment,
        "numerical_policy": policy,
        "source_admission": asdict(source),
        "reserved_attempt_policy": {"maximum_step_records": 2},
    }
    protocol_path = root / (STEM + ".protocol.json")

    def save_protocol():
        protocol_path.write_text(
            json_dumps(encode_exact_tree(protocol)), encoding="utf8"
        )

    save_protocol()
    paths = {
        key: root / (STEM + suffix)
        for key, suffix in (
            ("attempt", ".attempt.json"),
            ("outcome", ".json"),
            ("export", ".export-error.json"),
            ("steps", ".steps"),
        )
    }
    calls = []
    result = SimpleNamespace(
        records=[
            {"step_index": 1, "accepted": True, "radius": Q(1, 10)},
            {"step_index": 2, "accepted": False, "reason": "synthetic_limit"},
        ],
        report={"status": "unavailable", "reason": "synthetic_limit"},
        error=None,
    )

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

    def fake_producer(*, record_step):
        assert paths["attempt"].is_file() and paths["steps"].is_dir()
        calls.append("producer")
        for record in result.records:
            record_step(record)
            path = paths["steps"] / f"step-{record['step_index']:04d}.json"
            assert path.is_file(), "step was not retained before continuation"
        if result.error is not None:
            raise result.error
        return result.report

    monkeypatch.setattr(owner, "receiver_enclosure_policy", lambda: policy)
    monkeypatch.setattr(owner, "receiver_enclosure_source", lambda: source)
    monkeypatch.setattr(owner, "run_receiver_enclosure", fake_producer)
    namespace = {
        "ROOT": root,
        "STEM": STEM,
        "__file__": str(evaluator),
        "Path": Path,
        "zipfile": zipfile,
        "asdict": asdict,
        "datetime": datetime,
        "timezone": timezone,
        "time": time,
        "subprocess": SimpleNamespace(run=run_git, check_output=read_git),
        "inspect_frozen_source": inspect,
        "decode_exact_tree": decode_exact_tree,
        "encode_exact_tree": encode_exact_tree,
        "file_receipt": file_receipt,
        "read_bytes_bounded": read_bytes_bounded,
        "write_json_once": write_json_once,
        "json_dumps": json_dumps,
        "json_loads": json_loads,
    }
    exec(_compile_functions(driver_source), namespace)
    namespace["_runtime_environment"] = lambda: environment
    return SimpleNamespace(
        root=root,
        paths=paths,
        protocol=protocol,
        policy=policy,
        result=result,
        namespace=namespace,
        calls=calls,
        save_protocol=save_protocol,
        main=namespace["main"],
    )


def _read(path):
    return decode_exact_tree(json_loads(path.read_bytes()))


def test_synthetic_first_response_retains_each_accepted_and_failed_step(harness):
    harness.main()
    outcome = _read(harness.paths["outcome"])
    assert outcome["status"] == "unavailable"
    assert outcome["report"] == harness.result.report
    assert outcome["incomplete_step_paths"] == ()
    assert harness.calls == ["git_diff", "git_untracked", "producer"]
    assert len(outcome["step_receipts"]) == 2
    for index, receipt in enumerate(outcome["step_receipts"], 1):
        path = harness.paths["steps"] / f"step-{index:04d}.json"
        assert receipt == file_receipt(path, label=receipt["path"])
        assert _read(path) == harness.result.records[index - 1]
    assert outcome["attempt"] == file_receipt(
        harness.paths["attempt"], label=STEM + ".attempt.json"
    )
    with pytest.raises(FileExistsError, match="first receiver attempt"):
        harness.main()
    assert harness.calls.count("producer") == 1


@pytest.mark.parametrize("key", ["attempt", "outcome", "export", "steps"])
def test_existing_attempt_or_prefix_blocks_before_runtime_or_producer(harness, key):
    path = harness.paths[key]
    if key == "steps":
        path.mkdir()
    else:
        path.write_bytes(b"retain")
    with pytest.raises(FileExistsError):
        harness.main()
    assert harness.calls == []


@pytest.mark.parametrize(
    "key", ["runtime_environment", "numerical_policy", "source_admission"]
)
def test_preflight_mismatch_never_creates_attempt(harness, key):
    harness.protocol[key] = {"different": True}
    harness.save_protocol()
    with pytest.raises(ValueError, match="differs"):
        harness.main()
    assert not harness.paths["attempt"].exists()
    assert "producer" not in harness.calls


@pytest.mark.parametrize(
    "error", [RuntimeError("synthetic"), KeyboardInterrupt("synthetic")]
)
def test_interruption_preserves_complete_written_prefix_without_retry(harness, error):
    harness.result.records = harness.result.records[:1]
    harness.result.error = error
    harness.main()
    outcome = _read(harness.paths["outcome"])
    assert outcome["status"] == "execution_or_assessment_error"
    assert outcome["error_type"] == type(error).__name__
    assert len(outcome["step_receipts"]) == 1
    assert outcome["incomplete_step_paths"] == ()
    assert harness.calls.count("producer") == 1


@pytest.mark.parametrize(
    "record",
    [
        {"step_index": True, "accepted": True},
        {"step_index": 2, "accepted": True},
        {"step_index": 1, "accepted": 1},
    ],
)
def test_malformed_step_metadata_does_not_become_retained_evidence(harness, record):
    harness.result.records = [record]
    harness.main()
    outcome = _read(harness.paths["outcome"])
    assert outcome["status"] == "execution_or_assessment_error"
    assert outcome["step_receipts"] == ()
    assert tuple(harness.paths["steps"].iterdir()) == ()


def test_partial_step_export_retains_prefix_and_unadmitted_path(harness):
    original = harness.namespace["write_json_once"]

    def partial(path, value):
        if Path(path).name == "step-0002.json":
            Path(path).write_bytes(b"partial step")
            raise OSError("synthetic interrupted step export")
        original(path, value)

    harness.namespace["write_json_once"] = partial
    harness.main()
    outcome = _read(harness.paths["outcome"])
    assert outcome["status"] == "execution_or_assessment_error"
    assert len(outcome["step_receipts"]) == 1
    assert outcome["incomplete_step_paths"] == (STEM + ".steps/step-0002.json",)
    assert (harness.paths["steps"] / "step-0002.json").read_bytes() == b"partial step"


def test_partial_final_export_preserves_steps_and_separate_error(harness):
    original = harness.namespace["write_json_once"]

    def partial(path, value):
        if path == harness.paths["outcome"]:
            Path(path).write_bytes(b"partial outcome")
            raise OSError("synthetic interrupted outcome export")
        original(path, value)

    harness.namespace["write_json_once"] = partial
    with pytest.raises(OSError, match="synthetic"):
        harness.main()
    error = _read(harness.paths["export"])
    assert error["first_attempt_retained"] is True
    assert error["retry_authorized"] is False
    assert len(error["step_receipts"]) == 2
    assert harness.paths["outcome"].read_bytes() == b"partial outcome"


def test_freezer_has_no_trajectory_entry_point():
    tree = ast.parse(_driver_source(FREEZER))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert "run_receiver_enclosure" not in names
    assert "receiver_enclosure_policy" in names
    assert "receiver_enclosure_source" in names
