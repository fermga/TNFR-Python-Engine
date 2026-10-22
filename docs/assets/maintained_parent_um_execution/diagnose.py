"""One detached diagnosis of the frozen UM pressure-guard rejection.

Run only against the original archived guard. No event, proposal or trajectory
is replayed. The refreshed copy is not a committed runtime successor.
"""

from pathlib import Path

from tnfr.research import maintained_parent_um_execution as owner
from tnfr.research.artifact_io import (
    decode_exact_tree,
    file_receipt,
    write_json_once,
)
from tnfr.research.local_capacity_history import encode_result, refresh_pressure
from tnfr.research.maintained_parent_thol import _signature
from tnfr.research.retained_graph_state import (
    record_graph_state,
    restore_recorded_graph_state,
)
from tnfr.utils._structural_signature import structural_proof_signature
from tnfr.utils.cache import _graph_cache_manager
from tnfr.utils.io import json_loads

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def read(name):
    return decode_exact_tree(json_loads((OUT / name).read_bytes()))


def receipt(name):
    return file_receipt(OUT / name, label=name)


def main():
    if any(
        (OUT / name).exists() for name in ("diagnosis.json", "detached-refreshed.json")
    ):
        raise FileExistsError("detached diagnosis already retained")
    relative = "src/tnfr/research/maintained_parent_um_execution.py"
    expected = next(
        row for row in read("source-manifest.json")["files"] if row["path"] == relative
    )
    if file_receipt(ROOT / relative, label=relative) != expected:
        raise ValueError("diagnosis requires the unchanged frozen guard")
    graph = restore_recorded_graph_state(read("raw-after.json")).graph
    before = restore_recorded_graph_state(read("source.json")["before"]).graph
    rollback = restore_recorded_graph_state(read("failure-endpoint.json")).graph
    manager = _graph_cache_manager(graph.graph)
    assert manager._graph_owner is graph.graph
    maxima = {key: graph.graph[key] for key in owner._PRESSURE_CACHE_KEYS}
    runtime = owner._pressure_preservation_signature(graph, runtime=True)
    persistent = owner._pressure_preservation_signature(graph)
    cache = structural_proof_signature(manager, opaque_references=(graph.graph,))
    refresh_pressure(graph)
    changed_maxima = {key: graph.graph[key] for key in owner._PRESSURE_CACHE_KEYS}
    runtime_same = (
        owner._pressure_preservation_signature(graph, runtime=True) == runtime
    )
    persistent_same = owner._pressure_preservation_signature(graph) == persistent
    cache_same = (
        structural_proof_signature(manager, opaque_references=(graph.graph,)) == cache
    )
    refreshed = record_graph_state(graph)
    graph.graph.update(maxima)
    restoring_only_maxima_restores_guard = (
        owner._pressure_preservation_signature(graph, runtime=True) == runtime
    )
    rollback_matches_before = _signature(rollback) == _signature(before)
    assert not runtime_same
    assert persistent_same and cache_same and restoring_only_maxima_restores_guard
    assert rollback_matches_before
    write_json_once(OUT / "detached-refreshed.json", encode_result(refreshed))
    write_json_once(
        OUT / "diagnosis.json",
        encode_result(
            {
                "schema": "tnfr.maintained-parent-um-guard-diagnosis.v1",
                "inputs": tuple(
                    receipt(name)
                    for name in (
                        "freeze.json",
                        "attempt.json",
                        "raw-after.json",
                        "failure.json",
                        "failure-endpoint.json",
                        "diagnose.py",
                    )
                ),
                "guard_source": expected,
                "detached_refreshed": receipt("detached-refreshed.json"),
                "operation": "shared pressure refresh on a restored raw copy with a fresh graph-owned cache manager; no operator or continuous step",
                "allowed_maxima_before": maxima,
                "allowed_maxima_after": changed_maxima,
                "runtime_guard_unchanged": runtime_same,
                "persistent_pressure_only_guard_unchanged": persistent_same,
                "cache_owned_payload_unchanged_with_opaque_owner": cache_same,
                "restoring_only_allowed_maxima_restores_frozen_guard": restoring_only_maxima_restores_guard,
                "rollback_matches_frozen_before": rollback_matches_before,
                "recursive_path": "graph.graph['_tnfr_cache_manager']._graph_owner -> graph.graph['_dnfrmax'/'_dnfrmax_node']",
                "interpretation": "frozen guard falsely re-traverses separately protected owner state through a cache back-reference",
                "original_attempt_status": "failed_and_rolled_back",
                "runtime_successor_committed": False,
                "motion_certificate_transferred_to_runtime": False,
                "scope": "Separate post-failure diagnosis. This disposable refreshed copy is not an executed or committed UM endpoint.",
            }
        ),
    )
    print(
        "Detached diagnosis retained; frozen guard false rejection reproduced; original attempt still rolled back"
    )


if __name__ == "__main__":
    main()
