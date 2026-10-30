"""Synthetic content-admission controls; explicitly selected by the protocol.

No target reader, field, Taylor kernel, producer or subprocess is called.
The fake package files are never imported. Synthetic review text tests byte
associations only and is expressly not an independent scientific approval.
"""

import json
import sys
import zipfile
from pathlib import Path

import pytest
import tnfr

from tnfr.research import formed_two_port_run as runner
from tnfr.research.artifact_io import encode_exact_tree, file_receipt


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        (json.dumps(encode_exact_tree(value), sort_keys=True) + "\n").encode()
    )


def _archive(root, path, rows, *, corrupt=None):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for row in rows:
            data = (root / row["path"]).read_bytes()
            if row["path"] == corrupt:
                data += b"unreviewed alteration"
            archive.writestr(row["path"], data)


@pytest.fixture
def synthetic_admission(tmp_path, monkeypatch):
    root = (tmp_path / "repository").resolve()
    root.mkdir()
    worker_name = "artifacts/synthetic-worker.py"
    names = sorted(runner._REQUIRED_SOURCES | {worker_name, "src/tnfr/__init__.py"})
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("Inert synthetic content; never imported: " + name).encode())
    rows = [file_receipt(root / name, label=name) for name in names]
    source_zip = root / "artifacts/source.zip"
    _archive(root, source_zip, rows)
    review_rows = []
    for index in range(2):
        name = f"artifacts/review-{index}.md"
        (root / name).write_text(
            "Synthetic byte-association fixture; NOT scientific approval.\n",
            encoding="utf-8",
        )
        review_rows.append(file_receipt(root / name, label=name))
    control_path = root / "artifacts/controls.json"
    _json(control_path, {"passed": True, "reserved_reference_steps": 0})
    worker = root / worker_name
    output = root / "artifacts/attempt"
    output.mkdir()
    protocol_path = root / "artifacts/protocol.json"
    attempt = {
        "wall_seconds": 3600,
        "process_memory_bytes": 1024**3,
        "log_limit_bytes": 1024**2,
        "one_worker": True,
        "script": str(worker),
        "cwd": str(root),
        "arguments": [
            "--root", str(root), "--protocol", str(protocol_path),
            "--output", str(output),
        ],
        "interpreter": str(Path(sys.executable).resolve()),
        "prefix": str(Path(sys.prefix).resolve()),
    }
    _json(output / "attempt.json", attempt)
    record = {
        "schema": runner.PROTOCOL_SCHEMA,
        "policy": runner.declared_policy(),
        "head": "0" * 40,
        "runtime": {
            "python_executable": str(Path(sys.executable).resolve()),
            "python_prefix": str(Path(sys.prefix).resolve()),
            "python_version": sys.version,
            "package_root": str(root / "src/tnfr"),
        },
        "sources": rows,
        "archive": file_receipt(source_zip, label="artifacts/source.zip"),
        "reviews": review_rows,
        "controls": file_receipt(control_path, label="artifacts/controls.json"),
        "worker_entry": file_receipt(worker, label=worker_name),
    }
    # Exercise the real reader on a tiny fake inventory, not the live repository.
    # No source code in that inventory is loaded or executed.
    monkeypatch.setattr(tnfr, "__file__", str(root / "src/tnfr/__init__.py"))
    monkeypatch.setattr(sys, "argv", [str(worker)])

    def admit():
        _json(protocol_path, record)
        writer = runner.EvidenceWriter(output)
        return runner.admit_execution_inputs(root, protocol_path, writer)

    return {
        "root": root, "record": record, "attempt": attempt,
        "output": output, "protocol": protocol_path,
        "source_zip": source_zip, "controls": control_path, "admit": admit,
    }


def test_valid_synthetic_content_associations_are_copied_exactly(synthetic_admission):
    case = synthetic_admission
    result = case["admit"]()
    source_rows = {row["path"]: row for row in case["record"]["sources"]}
    assert result["worker_entry"] == case["record"]["worker_entry"]
    assert result["target_evidence"] == [
        source_rows[name] for name in sorted(runner._TARGET_EVIDENCE)
    ]
    assert (case["output"] / "protocol.json").read_bytes() == case["protocol"].read_bytes()
    assert (case["output"] / "source.zip").read_bytes() == case["source_zip"].read_bytes()
    assert result["archive"] == file_receipt(
        case["output"] / "source.zip", label="source.zip"
    )
    assert not (case["output"] / "admission.json").exists()
    assert list((case["output"] / "steps").iterdir()) == []


def test_missing_frozen_target_is_not_replaced_by_an_internal_manifest(synthetic_admission):
    case = synthetic_admission
    omitted = sorted(runner._TARGET_EVIDENCE)[0]
    case["record"]["sources"] = [
        row for row in case["record"]["sources"] if row["path"] != omitted
    ]
    with pytest.raises(ValueError, match="incomplete or duplicate frozen inventory"):
        case["admit"]()


def test_wrapper_must_be_in_freeze_and_be_the_executed_entry(synthetic_admission, monkeypatch):
    case = synthetic_admission
    monkeypatch.setattr(sys, "argv", [str(case["root"] / "different-wrapper.py")])
    with pytest.raises(ValueError, match="executed wrapper"):
        case["admit"]()


@pytest.mark.parametrize("change", [
    {"wall_seconds": 3601}, {"step_size": 1}, {"attempts": True},
])
def test_changed_or_boolean_policy_is_rejected(synthetic_admission, change):
    case = synthetic_admission
    case["record"]["policy"].update(change)
    with pytest.raises(ValueError, match="selected fixed policy"):
        case["admit"]()


@pytest.mark.parametrize("change", [
    {"wall_seconds": 3599}, {"process_memory_bytes": 1024**3 + 1},
    {"one_worker": 1}, {"arguments": []},
])
def test_parent_declaration_must_match_the_same_fixed_attempt(synthetic_admission, change):
    case = synthetic_admission
    case["attempt"].update(change)
    _json(case["output"] / "attempt.json", case["attempt"])
    with pytest.raises(ValueError, match="supervisor declaration"):
        case["admit"]()


def test_consumed_source_digest_rejects_a_same_size_alteration(synthetic_admission):
    case = synthetic_admission
    path = case["root"] / sorted(runner._TARGET_EVIDENCE)[0]
    original = path.read_bytes()
    path.write_bytes(b"X" + original[1:])
    with pytest.raises(ValueError, match="frozen source receipt differs"):
        case["admit"]()


def test_archive_member_must_match_frozen_source_even_with_new_archive_receipt(synthetic_admission):
    case = synthetic_admission
    _archive(
        case["root"], case["source_zip"], case["record"]["sources"],
        corrupt=case["record"]["worker_entry"]["path"],
    )
    case["record"]["archive"] = file_receipt(case["source_zip"], label="artifacts/source.zip")
    with pytest.raises(ValueError, match="archived source digest or byte count"):
        case["admit"]()


@pytest.mark.parametrize("controls", [
    {"passed": False, "reserved_reference_steps": 0},
    {"passed": True, "reserved_reference_steps": True},
    {"passed": True, "reserved_reference_steps": 1},
])
def test_control_receipt_cannot_admit_failed_or_transit_controls(synthetic_admission, controls):
    case = synthetic_admission
    _json(case["controls"], controls)
    case["record"]["controls"] = file_receipt(case["controls"], label="artifacts/controls.json")
    with pytest.raises(ValueError, match="non-transit controls not passed"):
        case["admit"]()


def test_unlisted_package_source_is_detected(synthetic_admission):
    case = synthetic_admission
    (case["root"] / "src/tnfr/unfrozen.py").write_text("never imported", encoding="utf-8")
    with pytest.raises(ValueError, match="does not cover the installed TNFR source"):
        case["admit"]()


def test_duplicate_reviews_do_not_count_as_independent_associations(synthetic_admission):
    case = synthetic_admission
    first = case["record"]["reviews"][0]
    case["record"]["reviews"] = [first, first]
    with pytest.raises(ValueError, match="duplicate independent review association"):
        case["admit"]()


def test_runtime_identity_must_match_the_selected_interpreter(synthetic_admission):
    case = synthetic_admission
    case["record"]["runtime"]["python_version"] = "different interpreter"
    with pytest.raises(ValueError, match="worker runtime differs"):
        case["admit"]()
