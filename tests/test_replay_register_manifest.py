from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from scripts.replay.register_manifest import main, register_manifest


def _write_sample(tmp_path: Path, *, inline: bool) -> tuple[Path, Path, Path | None]:
    run_dir = tmp_path / "sample"
    run_dir.mkdir()
    manifest_path = run_dir / "_manifest.json"
    summary_path = run_dir / "_manifest_summary.json"
    archive_path: Path | None = run_dir / "_partition_files.txt.gz"

    entries = [
        {
            "partition_id": "p000",
            "candidate_count": 12,
            "node_count": 256,
            "boundary_count": 16,
            "telemetry": {
                "phi_s": 0.5,
                "phase_gradient": 0.1,
                "phase_curvature": 0.2,
                "coherence_length": 1.4,
            },
        },
        {
            "partition_id": "p001",
            "candidate_count": 8,
            "node_count": 192,
            "boundary_count": 12,
            "telemetry": {
                "phi_s": 0.47,
                "phase_gradient": 0.09,
                "phase_curvature": 0.19,
                "coherence_length": 1.31,
            },
        },
    ]

    partition_files = ["p000.json", "p001.json"] if inline else []
    manifest_payload = {
        "n": 299,
        "modulus": 1764526545,
        "timestamp": 1732960000.0,
        "partition_directory": str(run_dir),
        "partition_files": partition_files,
        "entries": entries,
        "summary": {},
        "aggregation": {},
        "partition_file_archive": "_partition_files.txt.gz" if not inline else None,
        "partition_file_threshold": 1000,
        "partition_files_inlined": inline,
    }
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")

    summary_payload = {
        "n": 299,
        "modulus": 1764526545,
        "timestamp": 1732960000.0,
        "partition_count": len(entries),
        "candidate_stats": {"min": 8, "max": 12, "avg": 10.0},
        "node_stats": {"min": 192, "max": 256, "avg": 224.0},
        "boundary_stats": {"min": 12, "max": 16, "avg": 14.0},
        "telemetry_keys": [
            "phi_s",
            "phase_gradient",
            "phase_curvature",
            "coherence_length",
        ],
        "file_index": {
            "inline": inline,
            "threshold": 1000,
            "archive": "_partition_files.txt.gz" if not inline else None,
            "manifest": "_manifest.json",
        },
    }
    summary_path.write_text(json.dumps(summary_payload), encoding="utf-8")

    if not inline:
        assert archive_path is not None
        with gzip.open(archive_path, "wb") as handle:
            for entry in ("p000.json", "p001.json"):
                handle.write((entry + "\n").encode("utf-8"))
    else:
        archive_path = None

    return summary_path, manifest_path, archive_path


def test_register_manifest_inline(tmp_path: Path) -> None:
    summary_path, manifest_path, _ = _write_sample(tmp_path, inline=True)

    record = register_manifest(
        experiment_id="exp-inline",
        summary_path=summary_path,
        manifest_path=manifest_path,
        sample_size=3,
    )

    assert record["partition_files"]["count"] == 2
    assert record["partition_files"]["source"] == "inline"
    assert set(record["structural_fields"]) == {
        "phi_s",
        "phase_gradient",
        "phase_curvature",
        "coherence_length",
    }
    assert record["manifest_entries"] == 2


def test_register_manifest_archive(tmp_path: Path) -> None:
    summary_path, manifest_path, archive_path = _write_sample(tmp_path, inline=False)

    record = register_manifest(
        experiment_id="exp-archive",
        summary_path=summary_path,
        manifest_path=manifest_path,
        archive_path=archive_path,
        sample_size=1,
    )

    assert record["partition_files"]["count"] == 2
    assert record["partition_files"]["source"] == "archive"
    assert record["partition_files"]["sample"]
    assert record["file_index"]["archive_absolute"].endswith("_partition_files.txt.gz")
    assert (
        record["structural_fields"]["phi_s"]["max"]
        > record["structural_fields"]["phi_s"]["min"]
    )


def test_explicit_false_uses_archive_and_missing_observations_remain_missing(tmp_path):
    summary, manifest, archive = _write_sample(tmp_path, inline=False)
    data = json.loads(summary.read_text(encoding="utf-8"))
    data["file_index"]["inline"] = "false"
    summary.write_text(json.dumps(data), encoding="utf-8")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["entries"][0]["telemetry"] = None
    data["entries"][1]["telemetry"] = {"phi_s": 0.0}
    manifest.write_text(json.dumps(data), encoding="utf-8")

    result = register_manifest(experiment_id="explicit-archive", summary_path=summary)

    assert result["partition_files"]["source"] == "archive"
    assert result["partition_files"]["sample"] == ["p000.json", "p001.json"]
    assert result["structural_fields"] == {
        "phi_s": {"min": 0.0, "max": 0.0, "avg": 0.0, "count": 1.0}
    }
    assert result["archive_path"] == str(archive)


@pytest.mark.parametrize("value", (True, "0.5", [], {}))
def test_present_invalid_telemetry_is_not_dropped_or_coerced(tmp_path, value):
    summary, manifest, _ = _write_sample(tmp_path, inline=True)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["entries"][0]["telemetry"]["phi_s"] = value
    manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises((TypeError, ValueError)):
        register_manifest(experiment_id="invalid", summary_path=summary)


def test_representable_telemetry_mean_does_not_overflow_its_sum(tmp_path):
    summary, manifest, _ = _write_sample(tmp_path, inline=True)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    for entry in data["entries"]:
        entry["telemetry"]["phi_s"] = 1e308
    manifest.write_text(json.dumps(data), encoding="utf-8")
    result = register_manifest(experiment_id="large-finite", summary_path=summary)
    assert result["structural_fields"]["phi_s"]["avg"] == 1e308


@pytest.mark.parametrize(
    "payload",
    (
        '{"entries":[],"entries":[{}]}',
        '{"entries":[{"telemetry":{"phi_s":NaN}}]}',
        '{"entries":[{"telemetry":{"phi_s":1e-999}}]}',
        "[]",
    ),
)
def test_invalid_json_never_replaces_registration_output(tmp_path, payload):
    summary, manifest, _ = _write_sample(tmp_path, inline=True)
    manifest.write_text(payload, encoding="utf-8")
    output = tmp_path / "registration.json"
    original = b'{"retained":"registration"}\n'
    output.write_bytes(original)
    with pytest.raises(ValueError):
        main(
            [
                "--experiment-id",
                "invalid",
                "--summary",
                str(summary),
                "--output",
                str(output),
            ]
        )
    assert output.read_bytes() == original


@pytest.mark.parametrize("source_index", (0, 1, 2))
def test_registration_cannot_overwrite_its_source_files(tmp_path, source_index):
    sources = _write_sample(tmp_path, inline=False)
    summary = sources[0]
    source = sources[source_index]
    original = source.read_bytes()
    with pytest.raises(ValueError, match="cannot replace"):
        main(
            [
                "--experiment-id",
                "protect-input",
                "--summary",
                str(summary),
                "--output",
                str(source),
            ]
        )
    assert source.read_bytes() == original


@pytest.mark.parametrize("sample_size", (True, 0, -1, 1.5))
def test_invalid_sample_size_rejects_before_reading_files(tmp_path, sample_size):
    with pytest.raises(ValueError, match="sample_size"):
        register_manifest(
            experiment_id="invalid-size",
            summary_path=tmp_path / "absent.json",
            sample_size=sample_size,
        )
