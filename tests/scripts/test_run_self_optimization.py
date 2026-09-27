"""Tests for scripts.run_self_optimization."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import pytest

from scripts.run_self_optimization import parse_args, run
from tnfr.engines.manifest import write_manifest_bundle

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


@pytest.mark.parametrize(
    "invalid", ("pair_attributes", "unknown_field", "duplicate_key", "NaN", "1e-5000")
)
def test_lossy_partition_is_failed_before_optimizer_execution(
    tmp_path, monkeypatch, invalid
):
    graph = nx.path_graph(2)
    graph.nodes[0]["EPI"] = 0.5
    paths = write_manifest_bundle(
        tmp_path, "manifest.json", "summary.json", {}, {}, [("p0", graph, {})]
    )
    partition_path = tmp_path / "manifest_partition_0.json"
    payload = json.loads(partition_path.read_text(encoding="utf-8"))
    if invalid == "pair_attributes":
        payload["graph"]["nodes"][0]["attributes"] = [["EPI", 0.5], ["EPI", 9.0]]
    elif invalid == "unknown_field":
        payload["graph"]["edges"][0]["key"] = "lost channel"
    text = json.dumps(payload)
    if invalid == "duplicate_key":
        text = text.replace('"EPI": 0.5', '"EPI": 0.5, "EPI": 9.0', 1)
    elif invalid in ("NaN", "1e-5000"):
        text = text.replace('"EPI": 0.5', f'"EPI": {invalid}', 1)
    partition_path.write_text(text, encoding="utf-8")

    from scripts.run_self_optimization import PartitionProcessor

    calls = []

    def unexpected_optimizer(self, **kwargs):
        calls.append(kwargs)
        raise AssertionError("Invalid state must not reach the optimizer")

    monkeypatch.setattr(PartitionProcessor, "_run_optimizer", unexpected_optimizer)
    summary = run(
        parse_args(
            [
                "--manifest",
                str(paths["manifest_absolute"]),
                "--output-dir",
                str(tmp_path / "run"),
                "--quiet",
            ]
        )
    )

    assert calls == []
    assert summary["success_count"] == 0 and summary["failure_count"] == 1
    result = summary["partition_results"][0]
    assert result["success"] is False and result["error"]


@pytest.mark.parametrize("summary_input", (False, True))
def test_duplicate_manifest_or_summary_keys_are_not_overwritten(
    tmp_path, summary_input
):
    graph = nx.path_graph(2)
    paths = write_manifest_bundle(
        tmp_path, "manifest.json", "summary.json", {}, {}, [("p0", graph, {})]
    )
    bad_path = paths["summary_absolute" if summary_input else "manifest_absolute"]
    bad_path.write_text('{"ambiguous": 1, "ambiguous": 2}', encoding="utf-8")
    args = ["--manifest", str(paths["manifest_absolute"]), "--quiet"]
    if summary_input:
        args.extend(["--manifest-summary", str(bad_path)])

    with pytest.raises(ValueError, match="keys collide"):
        run(parse_args(args))


def test_manifest_reader_requires_an_object_before_dispatch(tmp_path):
    manifest = tmp_path / "array.json"
    manifest.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="Manifest JSON must be an object"):
        run(parse_args(["--manifest", str(manifest), "--quiet"]))
