"""Snapshots and RNG restoration must preserve the recorded state."""

from copy import deepcopy
from dataclasses import asdict
import importlib
import hashlib
import json
from pathlib import Path
import random
import sqlite3
import sys
from types import MappingProxyType, SimpleNamespace

import networkx as nx
import numpy as np
import pytest


@pytest.fixture
def lab(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "factorization-lab"))
    return SimpleNamespace(
        seeds=importlib.import_module("seed_management"),
        snapshots=importlib.import_module("snapshot_system"),
    )


@pytest.fixture(autouse=True)
def preserve_rng_state():
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    yield
    random.setstate(python_state)
    np.random.set_state(numpy_state)


@pytest.fixture
def seed_manager(lab, monkeypatch):
    # Environment measurement is independent of the RNG restoration contract.
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(
        virtual_memory=lambda: SimpleNamespace(total=8 * 1024**3),
    ))
    return lab.seeds.TNFRSeedManager(123)


@pytest.mark.parametrize("graph_key,field", [
    ("GAMMA", "gamma"), ("GRAMMAR_CANON", "grammar"),
    ("DNFR_WEIGHTS", "dnfr_weights"), ("_Si_weights", "si_weights"),
])
def test_trace_mapping_is_detached_from_later_graph_edits(graph_key, field):
    from tnfr.trace import mapping_field

    graph = nx.Graph()
    graph.graph[graph_key] = {"value": 0.5, "nested": {"values": [1, 2]}}
    snapshot = mapping_field(graph, graph_key, field)[field]
    graph.graph[graph_key]["value"] = 0.9
    graph.graph[graph_key]["nested"]["values"].append(3)
    assert snapshot["value"] == 0.5
    assert snapshot["nested"]["values"] == [1, 2]
    with pytest.raises(TypeError):
        snapshot["value"] = 7


def test_recorded_trace_detaches_custom_field_and_nested_proxy():
    from tnfr.trace import _trace_capture

    graph = nx.Graph()
    graph.graph["TRACE"] = {"enabled": True, "capture": ["custom"]}
    values = {"items": [1], "nested": MappingProxyType({"values": [2]})}
    _trace_capture(graph, "before", {"custom": lambda graph: {"custom": values}})
    snapshot = graph.graph["history"]["trace_meta"][-1]["custom"]
    values["items"].append(9)
    values["nested"]["values"].append(9)
    assert snapshot["items"] == [1]
    assert snapshot["nested"]["values"] == [2]


def test_default_trace_context_detaches_transition_inputs_and_saved_history():
    from tnfr.core.default_implementations import DefaultTraceContext

    graph = nx.Graph()
    pre = {"coherence": 0.5, "nested": [1]}
    post = {"coherence": 0.8, "nested": [2]}
    with DefaultTraceContext(graph) as context:
        context.record_transition("IL", pre, post)
        pre["nested"].append(9)
        post["coherence"] = 0.1
    saved = graph.graph["_trace_transitions"][0]
    assert saved["pre"]["nested"] == [1]
    assert saved["post"]["coherence"] == 0.8
    context.transitions[0]["post"]["nested"].append(9)
    assert saved["post"]["nested"] == [2]


def test_seed_zero_is_an_explicit_reproducible_seed(lab):
    first = lab.seeds.TNFRSeedManager(0)
    second = lab.seeds.TNFRSeedManager(0)
    assert first.master_seed == second.master_seed == 0
    assert first.node_initialization_seed == second.node_initialization_seed


def test_json_seed_state_restores_python_and_numpy_continuations(seed_manager):
    random.random()
    np.random.random(3)
    saved = json.loads(json.dumps(seed_manager.capture_complete_state()))
    expected_python = [random.random() for _ in range(5)]
    expected_numpy = np.random.normal(size=5)
    random.seed(99)
    np.random.seed(99)
    assert seed_manager.restore_complete_state(saved)
    assert [random.random() for _ in range(5)] == expected_python
    np.testing.assert_array_equal(np.random.normal(size=5), expected_numpy)


@pytest.mark.parametrize("failure", ["missing_seed", "numpy_state", "master_mismatch"])
def test_invalid_seed_restore_does_not_modify_live_state(seed_manager, failure):
    saved = seed_manager.capture_complete_state()
    random.seed(777)
    np.random.seed(777)
    before = deepcopy(vars(seed_manager))
    before_python = random.getstate()
    before_numpy = seed_manager._get_numpy_state()
    if failure == "missing_seed":
        del saved["seed_state"]["threshold_jitter_seed"]
    elif failure == "numpy_state":
        saved["seed_state"]["numpy_random_state"]["generator"] = "invalid"
    else:
        saved["master_seed"] += 1
    assert not seed_manager.restore_complete_state(saved)
    assert vars(seed_manager) == before
    assert random.getstate() == before_python
    assert seed_manager._get_numpy_state() == before_numpy


def test_reproducibility_verifier_cannot_ignore_failed_restore(seed_manager, lab, monkeypatch):
    context = {"parameters": asdict(lab.seeds.create_demo_experiment_params(143)),
               "reproducibility_state": {}}
    monkeypatch.setattr(seed_manager, "_load_experiment_context", lambda experiment_id: context)
    result = seed_manager.validate_reproducibility("invalid")
    assert result["valid"] is False
    assert "restor" in result["error"].lower()


def test_seed_capture_without_optional_memory_probe(lab, monkeypatch):
    monkeypatch.setitem(sys.modules, "psutil", None)
    saved = lab.seeds.TNFRSeedManager(123).capture_complete_state()
    assert saved["environment"]["memory_total_gb"] is None


def test_seed_capture_preserves_cached_gaussian_continuations(seed_manager):
    random.gauss(0, 1)
    np.random.normal()
    saved = json.loads(json.dumps(seed_manager.capture_complete_state()))
    expected_python = random.gauss(0, 1)
    expected_numpy = np.random.normal()
    assert seed_manager.restore_complete_state(saved)
    assert random.gauss(0, 1) == expected_python
    assert np.random.normal() == expected_numpy


@pytest.fixture
def snapshot_record(lab, tmp_path):
    module = lab.snapshots
    manager = module.PartitionSnapshotManager(tmp_path / "snapshots.db")
    nodes = [module.create_mock_nodal_state(i) for i in range(3)]
    kwargs = dict(
        verification_stage="verification", modulus_n=143, candidate_factor=11,
        partition_strategy="test", nodal_states=nodes, partition_states=[],
        structural_fields=module.create_mock_structural_fields(3),
        network_topology=module.create_mock_network_topology(3),
        performance_metrics={"coherence": 0.8, "sense_index": 0.7},
    )
    snapshot_id = manager.create_snapshot(**kwargs)
    return module, manager, snapshot_id, kwargs


def test_snapshot_creation_detaches_inputs(snapshot_record):
    module, manager, snapshot_id, kwargs = snapshot_record
    expected = deepcopy(manager.load_snapshot(snapshot_id))
    kwargs["nodal_states"][0].epi_vector.append(99)
    kwargs["structural_fields"].phi_s_distribution.append(99)
    assert manager.load_snapshot(snapshot_id) == expected
    fresh = module.PartitionSnapshotManager(manager.db_path)
    assert fresh.load_snapshot(snapshot_id) == expected


def test_snapshot_reads_cannot_mutate_cached_record(snapshot_record):
    module, manager, snapshot_id, _ = snapshot_record
    snapshot = manager.load_snapshot(snapshot_id)
    expected = deepcopy(snapshot)
    snapshot.nodal_states[0].phase += 1
    snapshot.overall_coherence = 0.0
    assert manager.load_snapshot(snapshot_id) == expected
    assert module.PartitionSnapshotManager(manager.db_path).load_snapshot(snapshot_id) == expected


@pytest.mark.parametrize("field", [
    "epi", "phase", "topology", "sense_index", "candidate", "identifier", "stored_hash",
])
def test_snapshot_integrity_covers_full_stored_state(snapshot_record, field):
    module, manager, snapshot_id, _ = snapshot_record
    snapshot = manager.load_snapshot(snapshot_id)
    if field == "epi":
        snapshot.nodal_states[0].epi_vector[0] += 1
    elif field == "phase":
        snapshot.nodal_states[0].phase += 1
    elif field == "topology":
        snapshot.network_topology.coupling_matrix[0][1] += 1
    elif field == "sense_index":
        snapshot.sense_index += 1
    elif field == "candidate":
        snapshot.candidate_factor = 13
    elif field == "identifier":
        snapshot.snapshot_id = "different-record"
    else:
        snapshot.state_hash = "altered-hash"
    with sqlite3.connect(manager.db_path) as connection:
        connection.execute("UPDATE snapshots SET compressed_data=? WHERE snapshot_id=?",
                           (module.SnapshotCompressor.compress_snapshot(snapshot), snapshot_id))
    fresh = module.PartitionSnapshotManager(manager.db_path)
    with pytest.raises(ValueError, match="integrity"):
        fresh.load_snapshot(snapshot_id)


def test_deleted_snapshots_do_not_survive_in_cache(snapshot_record, monkeypatch):
    module, manager, snapshot_id, _ = snapshot_record
    timestamp = manager.load_snapshot(snapshot_id).timestamp
    monkeypatch.setattr(module.time, "time", lambda: timestamp + 7200)
    assert manager.cleanup_old_snapshots(max_age_hours=1) == 1
    assert manager.load_snapshot(snapshot_id) is None


def test_legacy_snapshot_remains_readable_with_explicit_integrity_limit(snapshot_record):
    module, manager, snapshot_id, _ = snapshot_record
    snapshot = manager.load_snapshot(snapshot_id)
    legacy_data = {
        "modulus_n": snapshot.modulus_n, "stage": snapshot.verification_stage,
        "coherence": round(snapshot.overall_coherence, 6),
        "node_count": len(snapshot.nodal_states),
        "partition_count": len(snapshot.partition_states),
    }
    legacy_hash = hashlib.sha256(json.dumps(legacy_data, sort_keys=True).encode()).hexdigest()[:16]
    snapshot.state_hash = legacy_hash
    with sqlite3.connect(manager.db_path) as connection:
        connection.execute(
            "UPDATE snapshots SET compressed_data=?, state_hash=? WHERE snapshot_id=?",
            (module.SnapshotCompressor.compress_snapshot(snapshot), legacy_hash, snapshot_id),
        )
    fresh = module.PartitionSnapshotManager(manager.db_path)
    with pytest.warns(RuntimeWarning, match="legacy partial integrity"):
        assert fresh.load_snapshot(snapshot_id) == snapshot


def test_nonfinite_snapshot_is_rejected_before_storage(snapshot_record):
    _, manager, _, kwargs = snapshot_record
    before = manager.list_snapshots()
    kwargs["nodal_states"][0].phase = float("nan")
    with pytest.raises(ValueError):
        manager.create_snapshot(**kwargs)
    assert manager.list_snapshots() == before
