"""Read-only admission of the prospective freeze; no word or response replay."""

import subprocess
from io import BytesIO
from pathlib import Path

import pytest

from tnfr.research.artifact_io import (
    decode_exact_tree,
    encode_exact_tree,
    file_receipt,
    read_bytes_bounded,
    sha256_bytes,
    validate_artifact_path,
    verify_archive_members,
)
from tnfr.utils.io import json_dumps, json_loads

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "docs/assets/weighted_star_relative_probe"


def read_record(path, limit=4 * 1024**2):
    return decode_exact_tree(json_loads(read_bytes_bounded(path, max_bytes=limit)))


@pytest.fixture(scope="module")
def manifest():
    return read_record(ASSETS / "manifest.json")


def test_frozen_records_and_original_source_have_exact_byte_associations(manifest):
    assert manifest["status"] == "admitted; reserved_response_not_evaluated"
    assert set(manifest["artifacts"]) == {
        "protocol.json",
        "preparation.json",
        "admission.json",
        "source_snapshot.zip",
    }
    for name, receipt in manifest["artifacts"].items():
        validate_artifact_path(name)
        assert (
            file_receipt(ASSETS / name, label=name, max_bytes=16 * 1024**2) == receipt
        )
    receipt = manifest["original_preparation"]
    name = validate_artifact_path(receipt["path"])
    assert file_receipt(ROOT / name, label=name, max_bytes=256 * 1024) == receipt
    assert manifest["response_evaluated"] is False


def test_source_recovery_inventory_and_overlay_are_complete(manifest):
    inventory = manifest["source_inventory"]
    assert {
        "src/tnfr/research/weighted_star_relative_probe.py",
        "src/tnfr/research/weighted_star_relative_probe_protocol.py",
        "src/tnfr/physics/_sine_flow.py",
        "src/tnfr/mathematics/_validated_taylor.py",
        "theory/research/RELATIVE_COUPLING_PROBE.md",
        "tests/research/test_weighted_star_relative_probe_evidence.py",
        "freeze/freeze_relative_probe.py",
    } <= inventory.keys()
    overlay = {}
    base = []
    for name, item in inventory.items():
        validate_artifact_path(name)
        if item["recovery"] == "source_snapshot.zip":
            overlay[name] = item["sha256"]
        else:
            assert item["recovery"] == "git_blob"
            base.append((item["git_blob"], item))
    verify_archive_members(ASSETS / "source_snapshot.zip", overlay)
    # Recover unchanged committed bytes from the local object database in one
    # read. No network, extraction, source import or current-file substitution.
    result = subprocess.run(
        ["git", "cat-file", "--batch"],
        cwd=ROOT,
        input="".join(blob + "\n" for blob, _ in base).encode("ascii"),
        capture_output=True,
        check=True,
    )
    stream = BytesIO(result.stdout)
    for blob, expected in base:
        identity, kind, size = stream.readline().decode("ascii").strip().split()
        assert identity == blob and kind == "blob"
        data = stream.read(int(size))
        assert stream.read(1) == b"\n"
        assert len(data) == expected["bytes"]
        assert sha256_bytes(data) == expected["sha256"]
    assert stream.read() == b""


def test_static_admission_rebuilds_without_event_or_response_replay(monkeypatch):
    from tnfr.mathematics import _validated_taylor
    from tnfr.research import weighted_star_relative_probe as preparation_owner
    from tnfr.research.weighted_star_relative_probe_protocol import (
        assess_relative_probe_protocol,
    )

    def forbidden(*args, **kwargs):
        pytest.fail(
            "frozen static admission must not replay words or evaluate responses"
        )

    monkeypatch.setattr(
        preparation_owner, "prepare_weighted_star_relative_probe", forbidden
    )
    monkeypatch.setattr(preparation_owner, "execute_network_operator_stage", forbidden)
    monkeypatch.setattr(_validated_taylor, "validated_box_taylor_step", forbidden)
    monkeypatch.setattr(_validated_taylor, "validated_taylor_step", forbidden)
    original = read_record(
        ROOT / "docs/assets/weighted_star_registration_episode/preparation.json",
        256 * 1024,
    )
    actual = assess_relative_probe_protocol(
        original,
        read_record(ASSETS / "preparation.json"),
        read_record(ASSETS / "protocol.json"),
    )
    frozen = read_record(ASSETS / "admission.json")
    assert json_dumps(encode_exact_tree(actual), sort_keys=True) == json_dumps(
        encode_exact_tree(frozen), sort_keys=True
    )
