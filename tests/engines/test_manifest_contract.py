"""Analytic graph-state and reporting checks for supported engine manifests."""

import json
import math
import os
from pathlib import Path
import subprocess
import sys

import networkx as nx
import pytest

from tnfr.constants.aliases import ALIAS_THETA, ALIAS_VF
from tnfr.engines.manifest import (
    collect_manifest_telemetry, decode_graph, encode_graph, write_manifest_bundle,
)
from tnfr.parallel import FractalPartitioner
from scripts.run_self_optimization import (
    _collect_partition_entries, _compute_telemetry_deltas, parse_args, run,
)


@pytest.mark.parametrize("graph_type", [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph])
def test_json_graph_round_trip_preserves_labels_weights_triad_and_history(graph_type):
    graph = graph_type()
    graph.graph.update(seed=42, operator_history=["AL", "IL", "SHA"])
    graph.add_node(1, EPI=0.25, **{"νf": 0.0, "θ": 0.0, "ΔNFR": -0.5})
    graph.add_node("1", EPI=0.75, **{"νf": 1.0, "θ": 0.3, "ΔNFR": 0.5})
    graph.add_edge(1, "1", weight=0.0, history=[{"operator": "UM"}])
    if graph.is_multigraph():
        graph.add_edge(1, "1", key="parallel", weight=2.0)
    encoded = encode_graph(graph)
    restored = decode_graph(json.loads(json.dumps(encoded)))
    assert type(restored) is graph_type
    assert encode_graph(restored) == encoded
    assert list(restored) == [1, "1"]


@pytest.mark.parametrize("value", [object(), math.inf, ("AL", "SHA"), {1: "node"}])
def test_unsupported_graph_state_is_explicitly_rejected(value):
    graph = nx.path_graph(2)
    graph.graph["unsupported"] = value
    with pytest.raises(ValueError, match="finite JSON state"):
        encode_graph(graph)


def test_manifest_telemetry_matches_analytic_coherence():
    graph = nx.path_graph(2)
    for node in graph:
        graph.nodes[node].update(delta_nfr=0.5, dEPI_dt=0.25, vf=1.0, phase=0.0)
    before = encode_graph(graph)
    telemetry = collect_manifest_telemetry(graph)
    assert telemetry["coherence"] == pytest.approx(1 / 1.75)
    assert telemetry["structural_potential_range"] == [0.5, 0.5]
    assert isinstance(telemetry["sense_index"], float)
    assert encode_graph(graph) == before


@pytest.mark.parametrize("adaptive", [False, True])
def test_partition_limit_covers_coherent_graph_without_loss(adaptive):
    graph = nx.path_graph(["n0", 1, "n2", 3, "n4", 5, "n6"])
    nx.set_node_attributes(graph, 1.0, "vf")
    partitions = FractalPartitioner(
        max_partition_size=2, adaptive=adaptive, use_spatial_index=False,
    ).partition_network(graph)
    assert [len(nodes) for nodes, _ in partitions] == [2, 2, 2, 1]
    assert set().union(*(nodes for nodes, _ in partitions)) == set(graph)
    assert sum(len(nodes) for nodes, _ in partitions) == len(graph)
    for nodes, subgraph in partitions:
        assert nx.utils.graphs_equal(subgraph, graph.subgraph(nodes))


def test_zero_structural_aliases_take_precedence_in_partition_coherence():
    graph = nx.path_graph(2)
    graph.nodes[0].update({ALIAS_VF[0]: 0.0, "vf": 7.0, ALIAS_THETA[0]: 0.0, "phase": 3.0})
    graph.nodes[1].update({ALIAS_VF[0]: 0.0, ALIAS_THETA[0]: 0.0})
    partitioner = FractalPartitioner(use_spatial_index=False)
    assert partitioner._compute_community_coherence(graph, {0}, 1) == pytest.approx(1.0)


def test_single_node_partition_with_spatial_index():
    partitions = FractalPartitioner().partition_network(nx.empty_graph(1))
    assert len(partitions) == 1 and partitions[0][0] == {0}


def test_partition_seed_and_ties_are_independent_of_python_hash_seed():
    source = Path(__file__).resolve().parents[2] / "src"
    script = f"""
import sys, json
sys.path.insert(0, {str(source)!r})
import networkx as nx
from tnfr.parallel import FractalPartitioner
graph = nx.path_graph(['left', 'middle', 'right'])
for node, phase in zip(graph, [0., .8, 1.6]):
    graph.nodes[node].update(vf=1., phase=phase)
parts = FractalPartitioner(max_partition_size=2, coherence_threshold=.85, use_spatial_index=False, adaptive=False).partition_network(graph)
print(json.dumps([[n for n in graph if n in nodes] for nodes, _ in parts]))
"""
    outputs = []
    for hash_seed in (1, 42):
        result = subprocess.run(
            [sys.executable, "-c", script], check=True, capture_output=True, text=True,
            env={**os.environ, "PYTHONHASHSEED": str(hash_seed)},
        )
        outputs.append(json.loads(result.stdout.strip().splitlines()[-1]))
    assert outputs[0] == outputs[1] == [["left", "middle"], ["right"]]


def test_dry_run_deltas_are_not_archived_manifest_drift():
    before = {"coherence": 0.8, "sense_index": 0.4}
    deltas = _compute_telemetry_deltas(
        {"coherence": 0.1, "sense_index": 0.2},
        {"before": before, "after": dict(before)},
    )
    assert deltas["delta_c"] == deltas["delta_si"] == 0.0
    assert deltas["manifest_coherence_drift"] == pytest.approx(0.7)


def test_snapshot_improvement_is_reported_without_archived_baseline():
    deltas = _compute_telemetry_deltas(
        {}, {"before": {"coherence": 0.4}, "after": {"coherence": 0.6}},
    )
    assert deltas["delta_c"] == pytest.approx(0.2)


@pytest.mark.parametrize("graph", [nx.DiGraph([(1, "1")]), nx.Graph()])
def test_runner_consumes_generic_graph_and_reports_partition_identity(graph, tmp_path):
    telemetry = collect_manifest_telemetry(graph)
    paths = write_manifest_bundle(
        tmp_path, "manifest.json", "summary.json",
        {"operation_type": "pattern_discovery"}, {}, [("batch:p0", graph, telemetry)],
    )
    result = run(parse_args([
        "--manifest", str(paths["manifest_absolute"]), "--output-dir", str(tmp_path / "run"),
        "--seed", "42", "--quiet",
    ]))
    assert result["success_count"] == 1, result["partition_results"]
    entry = result["partition_results"][0]
    payload = json.loads(Path(entry["engine"]["snapshot_path"]).read_text(encoding="utf-8"))
    assert payload["metadata"]["partition_id"] == "batch:p0"
    assert entry["seed"] == 42
    assert result["operation_type"] == "pattern_discovery"


def test_duplicate_manifest_partition_ids_are_rejected(tmp_path):
    payload = tmp_path / "partition.json"
    payload.write_text("{}", encoding="utf-8")
    manifest = {"entries": [
        {"partition_id": "p0", "relative_path": payload.name},
        {"partition_id": "p0", "relative_path": payload.name},
    ]}
    with pytest.raises(ValueError, match="Duplicate"):
        _collect_partition_entries(manifest, tmp_path / "manifest.json", None)


def test_seed_offset_stays_with_original_entry_after_filtering(tmp_path, monkeypatch):
    graph = nx.path_graph(2)
    paths = write_manifest_bundle(
        tmp_path, "manifest.json", "summary.json", {}, {},
        [("p0", graph, {}), ("p1", graph, {})],
    )
    from scripts.run_self_optimization import PartitionProcessor

    observed = []
    def capture(self, **kwargs):
        observed.append(kwargs["seed_value"])
        return {"dry_run": True}
    monkeypatch.setattr(PartitionProcessor, "_run_optimizer", capture)
    common = ["--manifest", str(paths["manifest_absolute"]), "--output-dir", str(tmp_path / "run"),
              "--seed", "42", "--quiet"]
    run(parse_args(common))
    run(parse_args([*common, "--partitions", "p1"]))
    assert observed == [42, 43, 43]
