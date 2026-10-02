"""Integration tests for the canonical tnfr.factorization API."""

from __future__ import annotations

import gzip
import math
from dataclasses import replace
from pathlib import Path

import pytest

import tnfr.factorization as factorization_module
from tnfr.factorization import factorize
from tnfr.sdk.utils import import_from_json


def test_factorize_returns_spectral_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    payload = import_from_json(partition_files[0])
    assert payload["n"] == 221
    assert payload["partition_id"].startswith("p")
    assert result.partition_manifest_path
    manifest_path = Path(result.partition_manifest_path)
    assert manifest_path.exists()
    manifest_payload = import_from_json(manifest_path)
    assert manifest_payload["partition_files"]
    assert result.partition_manifest_index_path
    summary_path = Path(result.partition_manifest_index_path)
    assert summary_path.exists()
    summary_payload = import_from_json(summary_path)
    assert summary_payload["partition_count"] == len(manifest_payload["entries"])
    assert result.partition_file_archive_path is None


def test_factorize_emits_manifest_for_multiple_partitions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TNFR_PARTITION_TARGET_SIZE", "5")
    monkeypatch.setenv("TNFR_PARTITION_OVERLAP", "0")
    partition_root = tmp_path / "partition_outputs"
    monkeypatch.setenv("TNFR_PARTITION_OUTPUT_DIR", str(partition_root))
    monkeypatch.setattr(factorization_module, "_DEFAULT_FACTORIZER", None)
    backend = factorization_module._get_factorizer()._fft_backend
    native_spectral_state = backend.get_spectral_state
    captured_states = []

    def unavailable_length(*args, **kwargs):
        # Supply the missing-observation case instead of requiring the native
        # spectrum to be unavailable. Leave spectra and export owners unchanged.
        state = native_spectral_state(*args, **kwargs)
        captured_states.append(state)
        return replace(state, coherence_length=math.inf)

    monkeypatch.setattr(backend, "get_spectral_state", unavailable_length)

    result = factorization_module.factorize(
        299, trace_certificates=True, certificate_dir=tmp_path
    )

    assert captured_states, "The availability fixture must exercise the native backend"
    assert result.coherence_length == math.inf
    assert math.isnan(result.partition_aggregation["coherence_ratio"])
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

    manifest_payload = import_from_json(manifest_path)
    assert import_from_json(result.certificate_path)["n"] == 299
    for partition_file in partition_files:
        assert import_from_json(partition_file)["n"] == 299
    unavailable_lengths = [
        index
        for index, entry in enumerate(manifest_payload["entries"])
        if entry["telemetry"]["coherence_length"] is None
    ]
    assert unavailable_lengths
    for index in unavailable_lengths:
        assert manifest_payload["numeric_availability"][
            f"/entries/{index}/telemetry/coherence_length"
        ] == {"available": False, "reason": "positive_infinity"}
    assert manifest_payload["aggregation"]["coherence_ratio"] is None
    assert manifest_payload["numeric_availability"]["/aggregation/coherence_ratio"] == {
        "available": False,
        "reason": "undefined_nan",
    }
    assert len(manifest_payload["entries"]) == len(partition_files)
    first_entry_path = Path(manifest_payload["entries"][0]["relative_path"])
    if first_entry_path.is_absolute():
        assert first_entry_path.parent == partition_dir
    else:
        assert str(first_entry_path).startswith("partitioned/")
    assert result.partition_manifest_index_path
    summary_path = Path(result.partition_manifest_index_path)
    summary_payload = import_from_json(summary_path)
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

    result = factorization_module.factorize(
        299, trace_certificates=True, certificate_dir=tmp_path
    )

    assert result.partition_file_archive_path
    archive_path = Path(result.partition_file_archive_path)
    assert archive_path.exists()
    with gzip.open(archive_path, "rt", encoding="utf-8") as archive_stream:
        archived_files = [line.strip() for line in archive_stream if line.strip()]

    manifest_path = Path(result.partition_manifest_path)
    manifest_payload = import_from_json(manifest_path)
    assert import_from_json(result.certificate_path)["n"] == 299
    assert manifest_payload["partition_file_archive"]
    assert not manifest_payload["partition_files"]
    assert len(archived_files) == len(manifest_payload["entries"])

    assert result.partition_manifest_index_path
    summary_path = Path(result.partition_manifest_index_path)
    summary_payload = import_from_json(summary_path)
    assert summary_payload["file_index"]["inline"] is False
    assert summary_payload["file_index"]["archive"]
