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


@pytest.fixture
def graph_manifests(tmp_path):
    graphs = [nx.path_graph(3), nx.cycle_graph(4)]
    for graph in graphs:
        for node in graph:
            graph.nodes[node].update(
                EPI=0.2 + 0.1 * node, theta=0.1 * node, nu_f=1.0, delta_nfr=0.1
            )
    return write_manifest_bundle(
        tmp_path / "inputs",
        "manifest.json",
        "summary.json",
        {"operation_type": "partition"},
        {},
        [
            ("p0", graphs[0], {"phi_s": 0.12}),
            ("p1", graphs[1], {"phi_s": 0.15}),
        ],
    )


@pytest.mark.parametrize("invalid", (float("nan"), float("inf"), object()))
def test_payload_serialization_failure_preserves_previous_output(tmp_path, invalid):
    from scripts.run_self_optimization import _write_json

    target = tmp_path / "payload.json"
    original = b'{"retained":"payload"}\n'
    target.write_bytes(original)
    with pytest.raises((TypeError, ValueError)):
        _write_json(target, {"nested": {"value": invalid}})
    assert target.read_bytes() == original


@pytest.mark.parametrize("destination", ("manifest.json", "summary.json", "p0.json"))
def test_summary_cannot_replace_any_input_before_optimization(
    tmp_path, monkeypatch, destination
):
    from scripts import run_self_optimization as runner

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"entries": [{"partition_id": "p0", "relative_path": "p0.json"}]}),
        encoding="utf-8",
    )
    summary = tmp_path / "summary.json"
    summary.write_text("{}", encoding="utf-8")
    partition = tmp_path / "p0.json"
    partition.write_text("{}", encoding="utf-8")
    original = {path: path.read_bytes() for path in (manifest, summary, partition)}
    monkeypatch.setattr(
        runner,
        "TNFRSelfOptimizingEngine",
        lambda: pytest.fail("optimizer must not be constructed"),
    )
    output = tmp_path / "output"
    args = parse_args(
        [
            "--manifest",
            str(manifest),
            "--manifest-summary",
            str(summary),
            "--summary",
            str(tmp_path / destination),
            "--output-dir",
            str(output),
        ]
    )
    with pytest.raises(ValueError, match="cannot replace"):
        run(args)
    assert all(path.read_bytes() == data for path, data in original.items())
    assert not output.exists()


def test_run_self_optimization_dry_run(tmp_path: Path, graph_manifests) -> None:
    output_dir = tmp_path / "payloads"
    summary_path = tmp_path / "summary.json"
    args = parse_args(
        [
            "--manifest",
            str(graph_manifests["manifest_absolute"]),
            "--manifest-summary",
            str(graph_manifests["summary_absolute"]),
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


def test_run_self_optimization_partition_filter(
    tmp_path: Path, graph_manifests
) -> None:
    output_dir = tmp_path / "payloads"
    args = parse_args(
        [
            "--manifest",
            str(graph_manifests["manifest_absolute"]),
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


def test_modulus_only_manifest_requires_explicit_graphs(tmp_path, monkeypatch):
    from scripts.run_self_optimization import PartitionProcessor

    monkeypatch.setattr(
        PartitionProcessor,
        "_run_optimizer",
        lambda *a, **k: pytest.fail("implicit arithmetic graphs must not execute"),
    )
    original = {path: path.read_bytes() for path in DATA_ROOT.rglob("*.json")}
    summary = run(
        parse_args(
            [
                "--manifest",
                str(MANIFEST),
                "--manifest-summary",
                str(MANIFEST_SUMMARY),
                "--output-dir",
                str(tmp_path / "output"),
                "--quiet",
            ]
        )
    )
    assert summary["success_count"] == 0
    assert summary["failure_count"] == 2
    for result in summary["partition_results"]:
        assert "explicit 'graph' payload" in result["error"]
        assert "tnfr.engines.manifest.encode_graph" in result["error"]
    assert all(path.read_bytes() == data for path, data in original.items())


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


@pytest.mark.parametrize(
    "partition",
    (
        False,
        {"metadata": False},
        {"metadata": {"nodal_state": []}},
        {"metadata": {"sequence": False}},
        {"metadata": {"nodal_state": {"sequence": {"AL": 1}}}},
        {"metadata": {"sequence": ["AL", 4]}},
        {"metadata": {"sequence": ["AL", " "]}},
        {"telemetry": []},
        {"telemetry": {"coherence": True}},
        {"telemetry": {"phi_s": "unavailable"}},
    ),
)
def test_invalid_consumed_partition_metadata_fails_before_dispatch(
    tmp_path, monkeypatch, partition
):
    from scripts.run_self_optimization import PartitionProcessor

    paths = write_manifest_bundle(
        tmp_path,
        "manifest.json",
        "summary.json",
        {},
        {},
        [("p0", nx.path_graph(2), {})],
    )
    partition_path = tmp_path / "manifest_partition_0.json"
    payload = json.loads(partition_path.read_text(encoding="utf-8"))
    payload["partition"] = partition
    partition_path.write_text(json.dumps(payload), encoding="utf-8")
    sources = {
        path: path.read_bytes() for path in (paths["manifest_absolute"], partition_path)
    }
    calls = []
    monkeypatch.setattr(
        PartitionProcessor,
        "_run_optimizer",
        lambda self, **kwargs: calls.append(kwargs),
    )
    result = run(
        parse_args(
            [
                "--manifest",
                str(paths["manifest_absolute"]),
                "--output-dir",
                str(tmp_path / "output"),
                "--quiet",
            ]
        )
    )
    assert calls == []
    assert result["failure_count"] == 1 and result["success_count"] == 0
    assert result["partition_results"][0]["error"]
    assert all(path.read_bytes() == data for path, data in sources.items())
    assert not list((tmp_path / "output").glob("**/*.json"))


@pytest.mark.parametrize(
    ("metadata", "expected"),
    (
        (None, None),
        ({}, None),
        ({"nodal_state": None, "sequence": "AL"}, ["AL"]),
        ({"sequence": ["AL", "EN", "IL", "SHA"]}, ["AL", "EN", "IL", "SHA"]),
        ({"sequence": ["AL"], "nodal_state": {"sequence": ["EN"]}}, ["EN"]),
        ({"sequence": ["AL"], "nodal_state": {"sequence": []}}, []),
        ({"sequence": ["AL"], "nodal_state": {"sequence": None}}, None),
    ),
)
def test_optional_sequence_and_authoritative_nodal_state_are_preserved(
    metadata, expected
):
    from scripts.run_self_optimization import _extract_operator_sequence

    assert _extract_operator_sequence({"metadata": metadata}) == expected


def test_telemetry_preserves_missing_and_unused_metadata_without_fallback():
    from scripts.run_self_optimization import _extract_telemetry

    telemetry = _extract_telemetry(
        {"telemetry": {"coherence": None, "phi_s": 0, "archive": {"note": False}}},
        {
            "partition": {
                "telemetry": {"coherence": "shadowed", "sense_index": 0.25, "extra": []}
            }
        },
    )
    assert telemetry == {
        "coherence": None,
        "phi_s": 0.0,
        "sense_index": 0.25,
        "archive": {"note": False},
        "extra": [],
    }


@pytest.mark.parametrize(
    ("values", "expected"),
    (([1e308, 1e308], 1e308), ([1e308, -1e308], 0.0), ([0.0, 2.0], 1.0)),
)
def test_report_means_preserve_large_finite_values_and_cancellation(values, expected):
    from scripts.run_self_optimization import _aggregate_telemetry, _mean_numeric

    assert _mean_numeric(values) == expected
    aggregate = _aggregate_telemetry(
        [{"telemetry": {"phi_s": value}} for value in (*values, None)]
    )
    assert aggregate["phi_s_mean"] == expected
    assert aggregate["phi_s_min"] == min(values)
    assert aggregate["phi_s_max"] == max(values)


@pytest.mark.parametrize("invalid", (True, "0.5", float("nan"), float("inf")))
def test_invalid_samples_cannot_disappear_from_report_means(invalid):
    from scripts.run_self_optimization import _aggregate_telemetry, _mean_numeric

    with pytest.raises((TypeError, ValueError)):
        _mean_numeric([None, invalid, 0.5])
    with pytest.raises((TypeError, ValueError)):
        _aggregate_telemetry([{"telemetry": {"phi_s": invalid}}])
    assert _mean_numeric([None, 0.5]) is None
    assert _aggregate_telemetry([{}, {"telemetry": None}]) == {}


def test_snapshot_deltas_use_observed_values_and_preserve_missing_baselines():
    from scripts.run_self_optimization import _compute_telemetry_deltas

    deltas = _compute_telemetry_deltas(
        {"coherence": None, "sense_index": 0.5},
        {
            "before": {"coherence": 0.25, "sense_index": 0.75},
            "after": {"coherence": 0.75, "sense_index": 0.5},
        },
    )
    assert deltas["delta_c"] == 0.5
    assert deltas["delta_si"] == -0.25
    assert deltas["manifest_sense_index_drift"] == 0.25
    assert "manifest_coherence_drift" not in deltas
    assert "delta_phi_s" not in deltas


def test_snapshot_deltas_reject_invalid_baselines_and_unrepresentable_difference():
    from scripts.run_self_optimization import _compute_telemetry_deltas

    snapshots = {"before": {"coherence": -1e308}, "after": {"coherence": 1e308}}
    with pytest.raises(ValueError, match="observation difference"):
        _compute_telemetry_deltas({}, snapshots)
    with pytest.raises(TypeError, match="telemetry.coherence"):
        _compute_telemetry_deltas({"coherence": True}, {"before": {"coherence": 0.5}})

    smallest = float.fromhex("0x0.0000000000001p-1022")
    deltas = _compute_telemetry_deltas(
        {}, {"before": {"coherence": 0.0}, "after": {"coherence": smallest}}
    )
    assert deltas["delta_c"] == smallest


def test_cli_stdout_is_json_and_partition_logs_use_stderr(
    tmp_path, monkeypatch, capsys
):
    from scripts import run_self_optimization as runner

    paths = write_manifest_bundle(
        tmp_path,
        "manifest.json",
        "summary.json",
        {},
        {},
        [("p0", nx.path_graph(2), {})],
    )
    monkeypatch.setattr(
        runner.PartitionProcessor, "_run_optimizer", lambda self, **kwargs: {}
    )
    runner.main(
        [
            "--manifest",
            str(paths["manifest_absolute"]),
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )
    captured = capsys.readouterr()
    assert json.loads(captured.out)["success_count"] == 1
    assert "[self-opt] partition=p0 status=ok" in captured.err


def test_cli_rejects_invalid_summary_before_printing_json(monkeypatch, capsys):
    from scripts import run_self_optimization as runner

    monkeypatch.setattr(runner, "run", lambda args: {"value": float("nan")})
    with pytest.raises(ValueError):
        runner.main(["--manifest", "unused.json"])
    assert capsys.readouterr().out == ""
