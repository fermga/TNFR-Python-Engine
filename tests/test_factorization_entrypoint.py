"""Integration tests for the canonical tnfr.factorization API."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

import tnfr.factorization as factorization_module
from tnfr.factorization import factorize


def test_factorize_returns_spectral_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    partition_root = tmp_path / "partition_outputs"
    monkeypatch.setenv("TNFR_PARTITION_OUTPUT_DIR", str(partition_root))

    result = factorize(221, trace_certificates=True, certificate_dir=tmp_path)

    assert result.n == 221
    assert result.candidate_factors, "Expected at least one candidate factor"
    assert result.optimizer_metadata is not None
    assert result.fft_backend
    assert result.fft_capabilities

    assert result.certificate_path is not None
    cert_path = Path(result.certificate_path)
    assert cert_path.exists()
    assert cert_path.parent == tmp_path
    assert result.partition_summary
    assert result.partition_aggregation
    assert result.partition_artifact_dir
    partition_dir = Path(result.partition_artifact_dir)
    assert partition_dir.exists()
    assert partition_dir.parent == partition_root
    partition_files = sorted(
        path for path in partition_dir.glob("*.json") if not path.name.startswith("_")
    )
    assert partition_files
    payload = json.loads(partition_files[0].read_text())
    assert payload["n"] == 221
    assert payload["partition_id"].startswith("p")
    assert result.partition_manifest_path
    manifest_path = Path(result.partition_manifest_path)
    assert manifest_path.exists()
    manifest_payload = json.loads(manifest_path.read_text())
    assert manifest_payload["partition_files"]
    assert result.partition_manifest_index_path
    summary_path = Path(result.partition_manifest_index_path)
    assert summary_path.exists()
    summary_payload = json.loads(summary_path.read_text())
    assert summary_payload["partition_count"] == len(manifest_payload["entries"])
    assert result.partition_file_archive_path is None


def test_factorize_emits_manifest_for_multiple_partitions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TNFR_PARTITION_TARGET_SIZE", "5")
    monkeypatch.setenv("TNFR_PARTITION_OVERLAP", "0")
    partition_root = tmp_path / "partition_outputs"
    monkeypatch.setenv("TNFR_PARTITION_OUTPUT_DIR", str(partition_root))
    factorization_module._DEFAULT_FACTORIZER = None

    result = factorization_module.factorize(299, trace_certificates=True, certificate_dir=tmp_path)

    assert result.partition_artifact_dir
    assert result.partition_manifest_path

    partition_dir = Path(result.partition_artifact_dir)
    manifest_path = Path(result.partition_manifest_path)
    assert partition_dir.exists()
    assert manifest_path.exists()
    assert partition_dir.parent == partition_root

    partition_files = sorted(
        path for path in partition_dir.glob("*.json") if not path.name.startswith("_")
    )
    assert len(partition_files) > 1

    manifest_payload = json.loads(manifest_path.read_text())
    assert len(manifest_payload["entries"]) == len(partition_files)
    first_entry_path = Path(manifest_payload["entries"][0]["relative_path"])
    if first_entry_path.is_absolute():
        assert first_entry_path.parent == partition_dir
    else:
        assert str(first_entry_path).startswith("partitioned/")
    assert result.partition_manifest_index_path
    summary_path = Path(result.partition_manifest_index_path)
    summary_payload = json.loads(summary_path.read_text())
    assert summary_payload["partition_count"] == len(manifest_payload["entries"])
    assert summary_payload["file_index"]["inline"] is True
    assert result.partition_file_archive_path is None


def test_factorize_emits_compressed_partition_file_index_when_threshold_small(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TNFR_PARTITION_TARGET_SIZE", "4")
    monkeypatch.setenv("TNFR_PARTITION_OVERLAP", "0")
    monkeypatch.setenv("TNFR_PARTITION_FILELIST_THRESHOLD", "1")
    partition_root = tmp_path / "partition_outputs"
    monkeypatch.setenv("TNFR_PARTITION_OUTPUT_DIR", str(partition_root))
    factorization_module._DEFAULT_FACTORIZER = None

    result = factorization_module.factorize(299, trace_certificates=True, certificate_dir=tmp_path)

    assert result.partition_file_archive_path
    archive_path = Path(result.partition_file_archive_path)
    assert archive_path.exists()
    with gzip.open(archive_path, "rt", encoding="utf-8") as archive_stream:
        archived_files = [line.strip() for line in archive_stream if line.strip()]

    manifest_path = Path(result.partition_manifest_path)
    manifest_payload = json.loads(manifest_path.read_text())
    assert manifest_payload["partition_file_archive"]
    assert not manifest_payload["partition_files"]
    assert len(archived_files) == len(manifest_payload["entries"])

    assert result.partition_manifest_index_path
    summary_path = Path(result.partition_manifest_index_path)
    summary_payload = json.loads(summary_path.read_text())
    assert summary_payload["file_index"]["inline"] is False
    assert summary_payload["file_index"]["archive"]
