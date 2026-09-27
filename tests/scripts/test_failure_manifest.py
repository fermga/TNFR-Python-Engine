"""Failure-manifest recovery validates sources before replacing compact output."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from scripts import rebuild_failure_manifest as recovery


@pytest.fixture(scope="module")
def producer():
    path = (
        Path(__file__).resolve().parents[2]
        / "applications/factorization-lab"
        / "tnfr_factorization"
        / "failure_telemetry.py"
    )
    spec = importlib.util.spec_from_file_location("manifest_test_producer", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _payload(producer, *, n=97, timestamp=20.0, run_id="failure_97_latest"):
    return producer.FailureTelemetryRecord(
        run_id=run_id,
        artifact_path="previous-machine\\failure.json",
        timestamp=timestamp,
        n=n,
        modulus=13,
        failure_reason="No candidate; retained observación",
        failure_stage="spectral",
        metrics={"candidate_count": 0},
        bottlenecks=[
            producer.BottleneckSignal(
                "no_candidate_clusters",
                "high",
                "candidate_count",
                0.0,
                1.0,
                "No candidates",
            )
        ],
        recommendations=[],
        convergence_profile=None,
        partition_summary=None,
        partition_aggregation=None,
        nodal_decoding_snapshot=None,
        verification_report=None,
        replay_metadata=None,
        seed_state_path=None,
        extra_context={},
    ).to_mapping()


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _args(folder, manifest, *, count=None):
    args = ["--artifacts-dir", str(folder), "--manifest", str(manifest)]
    if count is not None:
        args.extend(["--expected-count", str(count)])
    return args


@pytest.mark.parametrize("expected_count", (None, 2))
def test_latest_per_n_compaction_accepts_arbitrary_numbers_and_portable_paths(
    tmp_path, monkeypatch, producer, expected_count
):
    monkeypatch.chdir(tmp_path)
    folder = tmp_path / "señales β"
    manifest = tmp_path / "rebuilt" / "failure_manifest.json"
    paths = []
    for name, n, timestamp in (
        ("failure_97_z_old", 97, 3.0),
        ("failure_97_a_new", 97, 20.0),
        ("failure_97_b_tied", 97, 20.0),
        ("failure_10403_only", 10403, 4.0),
    ):
        paths.append(
            _write(
                folder / (name + ".json"),
                _payload(producer, n=n, timestamp=timestamp, run_id=name),
            )
        )
    original = {path: path.read_bytes() for path in paths}

    assert recovery.main(_args(folder, manifest, count=expected_count)) == 0

    rebuilt = json.loads(manifest.read_text(encoding="utf-8"))
    assert rebuilt["version"] == "1.0"
    records = rebuilt["records"]
    assert [record["n"] for record in records] == [97, 10403]
    assert records[0]["run_id"] == "failure_97_b_tied"
    assert records[0]["timestamp"] == 20.0
    assert records[0]["bottlenecks"] == ["no_candidate_clusters"]
    assert records[0]["artifact_path"] == "señales β/failure_97_b_tied.json"
    assert records[0]["failure_reason"].endswith("observación")
    assert all("\\" not in record["artifact_path"] for record in records)
    assert all(path.read_bytes() == content for path, content in original.items())
    assert not list(manifest.parent.glob(".failure-manifest-*.tmp"))


def test_historical_bottleneck_codes_remain_compatible(tmp_path, producer):
    folder = tmp_path / "artifacts"
    payload = _payload(producer)
    payload["bottlenecks"] = ["low_global_coherence", "no_candidate_clusters"]
    _write(folder / "failure_97_old.json", payload)
    manifest = tmp_path / "manifest.json"

    assert recovery.main(_args(folder, manifest)) == 0
    rebuilt = json.loads(manifest.read_text(encoding="utf-8"))
    assert rebuilt["records"][0]["bottlenecks"] == payload["bottlenecks"]


@pytest.mark.parametrize(
    "defect",
    (
        "invalid-json",
        "non-object",
        "missing-n",
        "boolean-n",
        "timestamp",
        "nonfinite",
        "bottleneck",
    ),
)
def test_any_malformed_artifact_preserves_the_existing_manifest(
    tmp_path, producer, capsys, defect
):
    folder = tmp_path / "artifacts"
    _write(folder / "failure_97_valid.json", _payload(producer))
    bad = _payload(producer, timestamp=1.0, run_id="failure_97_stale")
    if defect == "non-object":
        bad = []
    elif defect == "missing-n":
        del bad["n"]
    elif defect == "boolean-n":
        bad["n"] = True
    elif defect == "timestamp":
        bad["timestamp"] = "yesterday"
    elif defect == "nonfinite":
        bad["timestamp"] = float("nan")
    elif defect == "bottleneck":
        bad["bottlenecks"] = [{"severity": "high"}]
    bad_path = _write(folder / "failure_97_stale.json", bad)
    if defect == "invalid-json":
        bad_path.write_text("{incomplete", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    original = b'{"version":"1.0","records":["original history"]}\n'
    manifest.write_bytes(original)

    assert recovery.main(_args(folder, manifest, count=1)) == 1
    assert manifest.read_bytes() == original
    assert "invalid artifact" in capsys.readouterr().err
    assert not list(tmp_path.glob(".failure-manifest-*.tmp"))


@pytest.mark.parametrize("condition", ("missing-folder", "empty-folder", "wrong-count"))
def test_no_empty_or_wrong_count_replacement(tmp_path, producer, condition):
    folder = tmp_path / "artifacts"
    if condition != "missing-folder":
        folder.mkdir()
    if condition == "wrong-count":
        _write(folder / "failure_97_valid.json", _payload(producer))
    manifest = tmp_path / "manifest.json"
    original = b"retained manifest\n"
    manifest.write_bytes(original)

    assert recovery.main(_args(folder, manifest, count=2)) == 1
    assert manifest.read_bytes() == original


def test_manifest_destination_cannot_replace_a_source_artifact(tmp_path, producer):
    source = _write(tmp_path / "failure_97_valid.json", _payload(producer))
    original = source.read_bytes()
    assert recovery.main(_args(tmp_path, source)) == 1
    assert source.read_bytes() == original


@pytest.mark.parametrize(
    "argv", ([], ["--artifacts-dir", "unused"], ["--manifest", "unused"])
)
def test_both_paths_are_required_without_historical_defaults(argv):
    with pytest.raises(SystemExit) as exc:
        recovery.main(argv)
    assert exc.value.code == 2


@pytest.mark.parametrize("count", ("0", "-1", "2.5"))
def test_expected_count_must_be_a_positive_integer(tmp_path, count):
    manifest = tmp_path / "manifest.json"
    with pytest.raises(SystemExit) as exc:
        recovery.main(_args(tmp_path, manifest, count=count))
    assert exc.value.code == 2
    assert not manifest.exists()
