"""Tests for scripts.run_self_optimization."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_self_optimization import parse_args, run

DATA_ROOT = Path("tests/data/self_optimization/test_run")
MANIFEST = DATA_ROOT / "_manifest.json"
MANIFEST_SUMMARY = DATA_ROOT / "_manifest_summary.json"


def test_run_self_optimization_dry_run(tmp_path: Path) -> None:
    output_dir = tmp_path / "payloads"
    summary_path = tmp_path / "summary.json"
    args = parse_args(
        [
            "--manifest",
            str(MANIFEST),
            "--manifest-summary",
            str(MANIFEST_SUMMARY),
            "--output-dir",
            str(output_dir),
            "--summary",
            str(summary_path),
            "--quiet",
        ]
    )
    summary = run(args)
    assert summary["success_count"] == 2
    assert summary["failure_count"] == 0
    assert summary["telemetry_summary"]["phi_s_mean"] > 0
    assert summary_path.is_file()
    stored = json.loads(summary_path.read_text(encoding="utf-8"))
    assert stored["success_count"] == 2
    for result in summary["partition_results"]:
        assert result["success"]
        snapshot_path = result["engine"].get("snapshot_path")
        if snapshot_path:
            assert Path(snapshot_path).is_file()
        telemetry = result.get("telemetry") or {}
        assert "delta_phi_s" in telemetry
        assert "delta_c" in telemetry
        deltas = result.get("telemetry_deltas") or {}
        assert "delta_phi_s" in deltas


def test_run_self_optimization_partition_filter(tmp_path: Path) -> None:
    output_dir = tmp_path / "payloads"
    args = parse_args(
        [
            "--manifest",
            str(MANIFEST),
            "--output-dir",
            str(output_dir),
            "--partitions",
            "p0",
            "--max-partitions",
            "1",
            "--quiet",
        ]
    )
    summary = run(args)
    assert summary["success_count"] == 1
    assert summary["partitions_requested"] == 1
    assert summary["partition_results"][0]["partition_id"] == "p0"
