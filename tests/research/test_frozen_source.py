"""Source restoration controls use unrelated tiny Git trees, never a producer."""

import copy
import io
import subprocess
import zipfile
from pathlib import Path

import pytest

from tnfr.research.artifact_io import sha256_bytes
from tnfr.research.frozen_source import inspect_frozen_source, restore_frozen_source
from tnfr.utils.io import json_dumps, json_loads

STEM = "docs/assets/control"
RECEIPT = STEM + ".freeze.json"
SOURCE = "src/control.py"
HELPER = "build/frozen/evaluate.py"


def _git(root, *arguments):
    return subprocess.check_output(["git", *arguments], cwd=root)


def _json(value):
    return (json_dumps(value, indent=2, allow_nan=False) + "\n").encode()


def _receipt(path, data):
    return {"path": path, "bytes": len(data), "sha256": sha256_bytes(data)}


def _pack(root, files, base, mutate_manifest=None):
    entries = []
    for name, data in files.items():
        entry = _receipt(name, data)
        if name.startswith("src/"):
            committed = _git(root, "show", f"{base}:{name}")
            entry["git_blob_sha256"] = sha256_bytes(committed)
            entry["normalized_lf_sha256"] = sha256_bytes(data.replace(b"\r\n", b"\n"))
        entries.append(entry)
    manifest = {
        "schema": "tnfr.sine-class-nonlinear-readout-source-snapshot.v1",
        "source_base_commit": base,
        "runtime_overlays": [],
        "files": entries,
    }
    if mutate_manifest:
        mutate_manifest(manifest)
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
        archive.writestr("source-manifest.json", _json(manifest))
    data = stream.getvalue()
    (root / (STEM + ".source.zip")).write_bytes(data)
    frozen = {
        "schema": "tnfr.sine-class-nonlinear-readout-freeze.v1",
        "source_base_commit": base,
        "runtime_overlays": [],
        "evaluation_status_at_freeze": "not_evaluated",
        "artifacts": [
            _receipt(STEM + ".protocol.json", files[STEM + ".protocol.json"]),
            _receipt(STEM + ".source.zip", data),
        ],
        "future_evaluator": HELPER,
    }
    (root / RECEIPT).write_bytes(_json(frozen))


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "active"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "core.autocrlf", "false")
    (root / "src").mkdir()
    (root / SOURCE).write_bytes(b"original = 1\n")
    (root / "src/unlisted.py").write_bytes(b"dependency = 'pinned'\n")
    (root / "docs/assets").mkdir(parents=True)
    (root / "docs/assets/prior.json").write_bytes(b'{"earlier":true}\n')
    _git(root, "add", ".")
    _git(
        root,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "Unrelated restoration control",
    )
    base = _git(root, "rev-parse", "HEAD").decode().strip()
    protocol = {
        "schema": "tnfr.sine-class-nonlinear-readout-protocol.v1",
        "source_base_commit": base,
        "runtime_overlays": [],
        "source_specification": {
            "prior_artifact_receipts": [
                _receipt(
                    "docs/assets/prior.json",
                    (root / "docs/assets/prior.json").read_bytes(),
                )
            ]
        },
    }
    files = {
        SOURCE: b"original = 1\r\n",
        HELPER: b"raise RuntimeError('Archived code must never execute')\n",
        STEM + ".protocol.json": _json(protocol),
        "theory/proof.md": b"Prospective control proof.\n",
    }
    (root / (STEM + ".protocol.json")).write_bytes(files[STEM + ".protocol.json"])
    _pack(root, files, base)
    # The active source intentionally differs; restoration must use the base.
    (root / SOURCE).write_bytes(b"local_work = 'preserve'\n")
    return root, files, base


def test_inspection_is_read_only_and_uses_git_base(source):
    root, files, base = source
    before = {
        p: p.read_bytes()
        for p in root.rglob("*")
        if p.is_file() and ".git" not in p.parts
    }
    report = inspect_frozen_source(root, RECEIPT)
    assert report.source_base_commit == base
    assert report.archived_file_count == 4
    assert report.prior_artifact_count == 1
    assert report.existing_outcome_files == ()
    assert before == {p: p.read_bytes() for p in before}
    assert not (root / HELPER).exists()


def test_comparison_adapter_requires_its_matching_protocol_and_manifest(source):
    root, files, base = source
    protocol = json_loads(files[STEM + ".protocol.json"])
    protocol["schema"] = "tnfr.sine-class-comparison-protocol.v1"
    files[STEM + ".protocol.json"] = _json(protocol)
    (root / (STEM + ".protocol.json")).write_bytes(files[STEM + ".protocol.json"])

    def comparison_manifest(manifest):
        manifest["schema"] = "tnfr.sine-class-comparison-source-snapshot.v1"

    _pack(root, files, base, mutate_manifest=comparison_manifest)
    receipt = json_loads((root / RECEIPT).read_bytes())
    # Cross-protocol receipts must not silently select the new semantics.
    with pytest.raises(ValueError, match="protocol source association"):
        inspect_frozen_source(root, RECEIPT)
    receipt["schema"] = "tnfr.sine-class-comparison-freeze.v1"
    (root / RECEIPT).write_bytes(_json(receipt))
    report = inspect_frozen_source(root, RECEIPT)
    assert report.source_base_commit == base
    assert report.archived_file_count == 4 and report.existing_outcome_files == ()
    export_error = STEM + ".export-error.json"
    (root / export_error).write_bytes(b'{"error":"retained export failure"}')
    assert inspect_frozen_source(root, RECEIPT).existing_outcome_files == (export_error,)


def test_restore_complete_base_and_exact_supplements_without_execution(
    source, tmp_path
):
    root, files, base = source
    destination = tmp_path / "restored"
    before = (root / SOURCE).read_bytes()
    inspection = restore_frozen_source(root, RECEIPT, destination)
    assert inspection.source_base_commit == base
    assert _git(destination, "rev-parse", "HEAD").decode().strip() == base
    assert _git(destination, "diff", base, "--name-only", "--", "src") == b""
    assert (destination / SOURCE).read_bytes() == b"original = 1\n"
    assert (destination / "src/unlisted.py").read_bytes() == b"dependency = 'pinned'\n"
    assert (root / SOURCE).read_bytes() == before
    for name, data in files.items():
        if not name.startswith("src/"):
            assert (destination / name).read_bytes() == data
    for suffix in (".freeze.json", ".source.zip", ".protocol.json"):
        assert (destination / (STEM + suffix)).read_bytes() == (
            root / (STEM + suffix)
        ).read_bytes()
    assert not (destination / (STEM + ".attempt.json")).exists()
    assert not (destination / (STEM + ".json")).exists()


@pytest.mark.parametrize("suffix", [".attempt.json", ".json"])
def test_existing_outcome_is_reported_and_prevents_new_preparation(
    source, tmp_path, suffix
):
    root, _, _ = source
    evidence = root / (STEM + suffix)
    evidence.write_bytes(b"retained failure")
    assert inspect_frozen_source(root, RECEIPT).existing_outcome_files == (
        STEM + suffix,
    )
    target = tmp_path / "not-created"
    with pytest.raises(FileExistsError, match="first attempt"):
        restore_frozen_source(root, RECEIPT, target)
    assert not target.exists()
    assert evidence.read_bytes() == b"retained failure"


def test_existing_destination_and_active_tree_are_preserved(source, tmp_path):
    root, _, _ = source
    target = tmp_path / "occupied"
    target.mkdir()
    (target / "work.txt").write_text("retain")
    with pytest.raises(FileExistsError):
        restore_frozen_source(root, RECEIPT, target)
    with pytest.raises(ValueError, match="outside"):
        restore_frozen_source(root, RECEIPT, root / "nested")
    assert (target / "work.txt").read_text() == "retain"


@pytest.mark.parametrize(
    "path", [STEM + ".protocol.json", STEM + ".source.zip", "docs/assets/prior.json"]
)
def test_tampered_external_associations_reject_before_creation(source, tmp_path, path):
    root, _, _ = source
    with (root / path).open("ab") as stream:
        stream.write(b"altered")
    target = tmp_path / "not-created"
    with pytest.raises(ValueError):
        restore_frozen_source(root, RECEIPT, target)
    assert not target.exists()


@pytest.mark.parametrize(
    "fault",
    [
        "changed_source",
        "blob_hash",
        "bool_size",
        "overlays",
        "missing_protocol",
        "missing_evaluator",
    ],
)
def test_rehashed_archive_cannot_replace_its_declared_source(source, fault):
    root, original, base = source
    files = copy.deepcopy(original)
    if fault == "changed_source":
        files[SOURCE] = b"changed = 2\n"
    elif fault == "missing_evaluator":
        del files[HELPER]

    def mutate(manifest):
        if fault == "blob_hash":
            manifest["files"][0]["git_blob_sha256"] = "0" * 64
        elif fault == "bool_size":
            manifest["files"][0]["bytes"] = True
        elif fault == "overlays":
            manifest["runtime_overlays"] = ["undeclared"]
        elif fault == "missing_protocol":
            # Removing only the manifest entry violates the exact inventory.
            manifest["files"] = [
                e for e in manifest["files"] if not e["path"].endswith(".protocol.json")
            ]

    _pack(root, files, base, mutate)
    with pytest.raises(ValueError):
        inspect_frozen_source(root, RECEIPT)


def test_retained_current_bundle_has_no_execution_side_effects():
    root = Path(__file__).resolve().parents[2]
    receipt = "docs/assets/sine_formed_classes/class-nonlinear-readout-v1.freeze.json"
    report = inspect_frozen_source(root, receipt)
    assert report.source_base_commit == "fa8e98a9b1bdd755709da481de7fc092b57bfe65"
    assert report.receipt_sha256 == sha256_bytes((root / receipt).read_bytes())
    # This test admits association only; later outcomes are not forbidden here.
    assert (
        report.future_evaluator
        == "build/class-nonlinear-readout-freeze/evaluate_readout.py"
    )


def test_unsupported_freeze_schema_is_not_guessed(source):
    root, _, _ = source
    path = root / RECEIPT
    receipt = json_loads(path.read_bytes())
    receipt["schema"] = "unrelated.v1"
    path.write_bytes(_json(receipt))
    with pytest.raises(ValueError, match="unsupported"):
        inspect_frozen_source(root, RECEIPT)


def test_unrelated_protocol_schema_is_not_guessed(source):
    root, files, base = source
    name = STEM + ".protocol.json"
    protocol = json_loads(files[name])
    protocol["schema"] = "unrelated.v1"
    files[name] = _json(protocol)
    (root / name).write_bytes(files[name])
    _pack(root, files, base)
    with pytest.raises(ValueError, match="protocol"):
        inspect_frozen_source(root, RECEIPT)


@pytest.mark.parametrize(
    "name", ["SRC/additional.py", STEM + ".attempt.json", STEM + ".json"]
)
def test_supplements_cannot_bypass_runtime_or_outcome_policy(source, name):
    root, files, base = source
    files[name] = b"undeclared supplement"
    _pack(root, files, base)
    with pytest.raises(ValueError):
        inspect_frozen_source(root, RECEIPT)


def test_failed_restoration_is_preserved_and_cannot_be_reused(
    source, tmp_path, monkeypatch
):
    from tnfr.research import frozen_source

    root, _, _ = source
    target = tmp_path / "partial"
    original = frozen_source._local_path

    def fail_supplement(current_root, name):
        if current_root == target and name == HELPER:
            raise OSError("synthetic supplement write failure")
        return original(current_root, name)

    monkeypatch.setattr(frozen_source, "_local_path", fail_supplement)
    with pytest.raises(OSError, match="synthetic"):
        restore_frozen_source(root, RECEIPT, target)
    assert (target / "src/unlisted.py").is_file()
    with pytest.raises(FileExistsError):
        restore_frozen_source(root, RECEIPT, target)
