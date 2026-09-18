"""One localized public Emission followed by a fixed paired native window.

Replay one archived control preparation, fork its live graph with the shared
cache-detached copy owner, and retain six calls per branch at most. This is a
finite regional response experiment, not a test of autonomous maintenance.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

from benchmarks import thol_full_state_response as full  # noqa: E402
from benchmarks import thol_native_runtime_response as native  # noqa: E402
from benchmarks import thol_native_policy_window as window  # noqa: E402
from benchmarks.thol_regional_balance_audit import _bind_source_edges  # noqa: E402
from tnfr.dynamics.dnfr import default_compute_delta_nfr  # noqa: E402
from tnfr.dynamics.integrators import DefaultIntegrator  # noqa: E402
from tnfr.dynamics.selectors import DefaultGlyphSelector, default_glyph_selector  # noqa: E402
from tnfr.research.claims import ClaimStatus  # noqa: E402
from tnfr.research.core_manifests import CoreExperimentManifest, current_git_source_provenance  # noqa: E402
from tnfr.sdk._state import copy_graph_state  # noqa: E402
from tnfr.utils.cache import GRAPH_RUNTIME_CACHE_KEYS  # noqa: E402
from tnfr.utils.io import safe_write  # noqa: E402

INPUT_PATH, INPUT_SHA256 = native.PRIOR_PATH, native.PRIOR_SHA256
START, DT, STEPS, END = F(3, 2), F(1, 4), 6, F(3)
BRANCHES = ("control", "localized_child_emission")
PROTOCOL = {
    "source": "One complete admitted control replay to t=1.5, followed by two cache-detached live copies",
    "marked_target": "First child in the actual retained ordered lineage; explicit supplied mark",
    "region": "All eight actual children; fixed original full-graph H restricted to this cohort",
    "start_time": START, "dt": DT, "maximum_steps_per_branch": STEPS, "completed_endpoint": END,
    "use_Si": True, "apply_glyphs": True, "global_reduction": "legacy",
    "preflight": "Current grammar and additional strict AL check; configured switches unchanged",
    "initial_gates": "Admitted single-target nonzero jump, positive paired child-shape error and control variance",
    "decision": "All seven control variances positive; endpoint error and error/contrast ratio decrease; control contrast does not decline",
    "policy_status": "Conservative sufficient finite-response policy; not a necessary definition of NFR identity",
    "target_substitution": False, "tuning": False, "retries": 0, "horizon_search": False,
    "autonomous_maintenance_claim": False, "preparation_seed": 17,
    "maximum_native_calls": 12, "maximum_post_preparation_forcing_capture_calls": 26,
}


def _scientific_record(graph):
    value = native._record(graph)
    value["graph_attributes"] = {key: item for key, item in value["graph_attributes"].items()
                                 if key not in GRAPH_RUNTIME_CACHE_KEYS}
    value["graph_instance_last_operator"] = getattr(graph, "_last_operator_applied", None)
    return value


def _source_domain(graph):
    """Keep the fixed source on the default traceable policy and external-effect scope."""
    if type(graph) is not nx.Graph:
        raise ValueError("the fixed source requires a plain undirected Graph")
    record = native._record(graph)
    if any(row["count"] for row in record["callback_registry"].values()):
        raise ValueError("registered callbacks are outside the branch-isolation contract")
    hook = graph.graph.get("compute_delta_nfr", default_compute_delta_nfr)
    selector = graph.graph.get("glyph_selector")
    integrator = graph.graph.get("integrator")
    if hook is not default_compute_delta_nfr:
        raise ValueError("the fixed source requires the default pressure owner")
    if selector is not None and selector is not default_glyph_selector and type(selector) is not DefaultGlyphSelector:
        raise ValueError("the fixed source requires the ordinary default selector")
    if integrator is not None and type(integrator) is not DefaultIntegrator:
        raise ValueError("the fixed source requires the traced default integrator")
    marker = getattr(graph, "_last_operator_applied", None)
    if marker is not None and type(marker) is not str:
        raise ValueError("only the known scalar last-operator instance marker is admitted")


def prepare_branches(graph, prior):
    """Fork live graph data, restoring neighbor order without changing its state."""
    _source_domain(graph)
    if prior["branch"] != "control" or prior["status"] != "executed" or full._state(graph) != prior["endpoint"]:
        raise ValueError("branch preparation requires the complete admitted live control endpoint")
    if F(full._state(graph)["time"]) != START:
        raise ValueError("the declared fork is at t=1.5")
    lineage = full._birth_families(graph, prior["prefix"])
    if full._payload(lineage) != full._payload(prior["lineage"]):
        raise ValueError("live ancestry differs from the replayed lineage")
    before = _scientific_record(graph)
    neighbor_order = tuple((node, tuple(graph.neighbors(node))) for node in graph)
    copies = []
    for _ in BRANCHES:
        copied = copy_graph_state(graph)
        # Shared graph.copy() rebuilds edge order. Preserve actual local reads,
        # as in thol_birth_transport / thol_retained_phase_audit, without
        # replacing shared undirected edge-data dictionaries.
        for node, neighbors in neighbor_order:
            adjacency = copied._adj[node]
            if len(adjacency) != len(neighbors) or set(adjacency) != set(neighbors):
                raise RuntimeError("copied local support differs")
            ordered = tuple((neighbor, adjacency[neighbor]) for neighbor in neighbors)
            adjacency.clear()
            adjacency.update(ordered)
        if hasattr(graph, "_last_operator_applied"):
            copied._last_operator_applied = graph._last_operator_applied
        if GRAPH_RUNTIME_CACHE_KEYS.intersection(copied.graph):
            raise RuntimeError("shared copy retained an excluded runtime cache")
        if _scientific_record(copied) != before:
            raise RuntimeError("cache-detached branch scientific state differs from source")
        copies.append(copied)
    if _scientific_record(graph) != before or copies[0] is copies[1]:
        raise RuntimeError("fork mutated the source or reused a graph")
    # The shared copy owner preserves nested alias topology within each copy.
    # Explicitly bind the outer graph/node/edge data independence here. The
    # source and opposite branch are also checked around every future call.
    for left, right in ((graph, copies[0]), (graph, copies[1]), tuple(copies)):
        if left.graph is right.graph or any(left.nodes[n] is right.nodes[n] for n in graph):
            raise RuntimeError("branches share mutable graph/node dictionaries")
        if any(left.edges[u, v] is right.edges[u, v] for u, v in graph.edges):
            raise RuntimeError("branches share mutable edge dictionaries")
    return tuple(copies), {"source_record": native._record(graph), "scientific_source": before,
                           "excluded_rebuildable_cache_keys": tuple(sorted(GRAPH_RUNTIME_CACHE_KEYS.intersection(graph.graph))),
                           "ordered_neighbors_restored": True, "scientific_records_equal": True,
                           "last_operator_instance_marker": getattr(graph, "_last_operator_applied", None),
                           "lineage": lineage,
                           "scope": "Shared graph-data copy contract; opaque resources are not serialized or independently authenticated. "
                                    "External-only aliases/effects and callback side effects are excluded."}


def _region_state(control, perturbed, original, children):
    nodes = original.source.nodes
    if tuple(control["state"]["nodes"]) != nodes or tuple(perturbed["state"]["nodes"]) != nodes:
        raise ValueError("paired regional readout requires original ordered nodes")
    indices = tuple(nodes.index(node) for node in children)
    weights = tuple(original.metric_weights[i] for i in indices)
    x, y = (tuple(F(record["state"]["epi"][i]) for i in indices) for record in (control, perturbed))
    shape = full._paired_delta((F(0),)*len(children), x, weights)
    paired = full._paired_delta(x, y, weights)
    return {"control_epi": x, "perturbed_epi": y, "metric_weights": weights,
            "control": shape, "paired": paired}


def _analyze_response(*args):
    from benchmarks.thol_regional_restoration_accounting import analyze_response
    return analyze_response(*args)


def run_branches(control, perturbed, prior, *, pool):
    """Execute the fixed admitted paired window on two already verified live copies."""
    if _scientific_record(control) != _scientific_record(perturbed):
        raise ValueError("paired branches do not have the same complete projected source")
    if F(full._state(control)["time"]) != START:
        raise ValueError("paired continuation must start at t=1.5")
    original = full._reference(prior["prefix"]["coupling"]["refreshed_forcing"])
    children = tuple(prior["lineage"]["children"])
    marked = children[0]
    source_control, source_perturbed = native._record(control), native._record(perturbed)
    for record in (source_control, source_perturbed):
        if (tuple(record["state"]["nodes"]) != original.source.nodes
                or tuple(F(value) for value in record["state"]["capacity"]) != original.source.capacity):
            raise ValueError("initial branch nodes/capacity differ from the original fixed-metric domain")
        _bind_source_edges(record["state"], original.source)
    admission = full._emission_admission(perturbed, (marked,))
    if native._record(control) != source_control or native._record(perturbed) != source_perturbed:
        raise RuntimeError("read-only intervention admission mutated a branch")
    common = {"marked_child": marked, "children": children, "original_reference": pool.pack(asdict(original)),
              "admission": pool.pack(admission), "pre_event_records": tuple(pool.pack(row) for row in (source_control, source_perturbed)),
              "forcing_capture_api_calls": 0}
    if not admission["allowed"]:
        return {**common, "status": "intervention_refused", "event": None, "branches": (),
                "analysis": None, "attempted_native_calls": 0, "decision": "inconclusive",
                "reason": "The single predeclared marked child failed AL admission; no substitute"}
    event = full._emission_event(perturbed, (marked,))
    if native._record(control) != source_control:
        raise RuntimeError("AL on the perturbed branch changed the control")
    initial_records = (native._record(control), native._record(perturbed))
    initial = _region_state(*initial_records, original, children)
    nodes = original.source.nodes
    jump = tuple(F(y)-F(x) for x, y in zip(source_perturbed["state"]["epi"], initial_records[1]["state"]["epi"], strict=True))
    index = nodes.index(marked)
    if any(value for i, value in enumerate(jump) if i != index):
        raise RuntimeError("localized Emission changed an unmarked EPI coordinate")
    weights = initial["metric_weights"]
    hmark = original.metric_weights[index]
    expected_error = hmark*(1-hmark/sum(weights))*jump[index]**2/2
    actual_error = initial["paired"]["centered_H_energy"]
    if expected_error != actual_error:
        raise RuntimeError("localized-jump child energy identity failed")
    initial_gates = {"nonzero_localized_jump": jump[index] != 0, "positive_initial_error": actual_error > 0,
                     "positive_control_variance": initial["control"]["centered_H_energy"] > 0,
                     "expected_error": expected_error, "actual_error": actual_error, "jump": jump}
    common.update(event=pool.pack(event), initial_records=tuple(pool.pack(row) for row in initial_records),
                  initial_region=initial, initial_gates=initial_gates, forcing_capture_api_calls=2)
    if not all(initial_gates[key] for key in ("nonzero_localized_jump", "positive_initial_error", "positive_control_variance")):
        return {**common, "status": "initial_gate_not_met", "branches": (), "analysis": None,
                "attempted_native_calls": 0, "decision": "inconclusive",
                "reason": "No declared restoration comparison without positive localized damage and control contrast"}
    graphs = (control, perturbed)
    initial_tetrad = tuple(native._tetrad(graph) for graph in graphs)
    records = [[], []]
    refused = False
    for ordinal in range(STEPS):
        for branch_index, graph in enumerate(graphs):
            other = graphs[1-branch_index]
            other_before = native._record(other)
            before = native._record(graph)
            trace = native._trace_step(graph, capture_generation=True)
            endpoint = native._record(graph)
            if native._record(other) != other_before:
                raise RuntimeError("native execution mutated the other branch")
            if trace["status"] == "executed" and F(endpoint["state"]["time"]) != START+(ordinal+1)*DT:
                raise RuntimeError("completed native call has an unexpected physical endpoint")
            if F(before["state"]["time"]) != START+ordinal*DT:
                raise RuntimeError("native call has an unexpected entry time")
            row = {"ordinal": ordinal, "status": trace["status"], "before": before,
                   "endpoint": endpoint, "native_trace": trace}
            records[branch_index].append(row)
            if trace["status"] != "executed":
                refused = True
                break
        if refused:
            break
    common_completed = 0
    for left, right in zip(*records):
        if left["status"] != "executed" or right["status"] != "executed":
            break
        common_completed += 1
    try:
        analysis = _analyze_response(
            records[0][:common_completed], records[1][:common_completed], original, children,
            initial_records[0]["state"]["epi"], initial_records[1]["state"]["epi"])
    except ValueError as exc:
        analysis = {"available": False, "error_type": type(exc).__name__, "reason": str(exc),
                    "decision": {"outcome": "inconclusive", "gates": {}, "failed_gates": ("analysis_domain",)},
                    "scope": "Retained traces lie outside the declared fixed-domain accounting; no retry"}
    branches = []
    for i, graph in enumerate(graphs):
        packed = tuple(pool.pack(row) for row in records[i])
        if any(window.expand_record(ref, pool.nodes) != full._payload(row) for ref, row in zip(packed, records[i], strict=True)):
            raise RuntimeError("native record compaction was not lossless")
        branches.append({"branch": BRANCHES[i], "steps": packed, "attempted_steps": len(records[i]),
                         "completed_steps": sum(row["status"] == "executed" for row in records[i]),
                         "initial_tetrad": pool.pack(initial_tetrad[i]), "terminal_tetrad": pool.pack(native._tetrad(graph)),
                         "terminal_record": pool.pack(native._record(graph))})
    native_captures = tuple(capture for rows in records for row in rows
                            for capture in row["native_trace"]["captures"].values())
    return {**common, "status": "native_refusal" if refused else "completed", "branches": branches,
            "analysis": pool.pack(analysis), "attempted_native_calls": sum(map(len, records)),
            "forcing_capture_api_calls": 2+len(native_captures),
            "native_forcing_capture_availability": tuple(capture["available"] for capture in native_captures),
            "forcing_capture_scope": "API calls after source replay: two in the AL owner plus retained generation/entry calls; not internal kernel counts",
            "aligned_completed_steps": common_completed,
            "scope": "Only the aligned completed prefix is analyzed; first refusal stops the paired study, without rollback or retry"}


def run_study(input_path=INPUT_PATH, *, expected_sha256=INPUT_SHA256):
    historical, binding = native.load_prior_evidence(input_path, expected_sha256=expected_sha256)
    graph, prior = full.replay_response_branch("control")
    admission = native._admit_replay(prior, historical["branches"][0])
    if not admission["full_payload_equal"]:
        raise ValueError("control replay must equal the complete archived report without UTC exemptions")
    branches, fork = prepare_branches(graph, prior)
    source_before = native._record(graph)
    pool = window.RecordPool()
    # The admitted report has the archive's JSON value semantics, but live
    # arithmetic can retain float64 or string-enum subclasses. Normalize through that same
    # finite JSON boundary before the stricter pool, and finish both source
    # packs before any new branch execution can produce results.
    canonical_prior = json.loads(json.dumps(full._payload(prior), allow_nan=False))
    if canonical_prior != historical["branches"][0]:
        raise ValueError("finite JSON normalization differs from the authenticated control report")
    packed_prior = pool.pack(canonical_prior)
    packed_fork = pool.pack(json.loads(json.dumps(full._payload(fork), allow_nan=False)))
    result = run_branches(*branches, prior, pool=pool)
    if native._record(graph) != source_before:
        raise RuntimeError("paired study mutated the retained source graph")
    if hashlib.sha256(Path(input_path).read_bytes()).hexdigest() != expected_sha256:
        raise RuntimeError("historical input changed during the study")
    return {"protocol": PROTOCOL, "historical_input": binding, "source_replay_admission": admission,
            "source_replay_calls": 1, "replayed_control_report": packed_prior, "fork": packed_fork,
            "source_pack_scope": "Prior and fork normalized to finite native JSON and packed before new branch execution; "
                                 "the canonical prior also equals the authenticated complete control report",
            **result, "record_pool": pool.nodes, "autonomous_maintenance_certified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--expected-sha256", default=INPUT_SHA256)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/research/thol_regional_restoration_2026_09_18.json")
    args = parser.parse_args()
    checkpoint = args.output.with_suffix(".completed.json")
    if args.input.resolve() in (args.output.resolve(), checkpoint.resolve()):
        raise ValueError("output must not overwrite the archived source")
    scope = ("src/tnfr", "benchmarks/thol_regional_restoration.py", "benchmarks/thol_regional_restoration_accounting.py",
             "benchmarks/thol_full_state_response.py", "benchmarks/thol_native_runtime_response.py",
             "benchmarks/thol_native_policy_window.py", "benchmarks/thol_distributed_transport.py",
             "benchmarks/thol_birth_transport.py", "benchmarks/thol_pressure_feedback.py",
             "benchmarks/thol_preparation_policy.py", "benchmarks/thol_distributed_target.py",
             "benchmarks/thol_family_closure.py", "benchmarks/thol_regional_balance_audit.py",
             "benchmarks/thol_regional_recovery_audit.py")
    provenance = current_git_source_provenance(ROOT, scope)
    sha, dirty, digest = provenance
    manifest = CoreExperimentManifest(
        claim_id="O3.a-localized-regional-restoration", git_sha=sha, source_dirty=dirty, dirty_source_hash=digest,
        versions={"python": platform.python_version(), "numpy": np.__version__, "networkx": nx.__version__},
        graph_construction="One authenticated live control replay, two independently copied graph-data branches",
        capacity_specification="Actual native capacity; fixed original full-graph metric for comparisons",
        solver="One localized public AL; six unchanged native steps per branch at most", timestep=float(DT), seed=17,
        result_status=ClaimStatus.MEASURED, operator_sequence=("emission",),
        telemetry=("localized child-shape damage", "paired native stages", "control contrast and drift", "regional restoration policy"),
        controls=("complete source replay", "fixed marked actual child", "no tuning or retry", "evolving control"),
        artifacts=(str(args.output),))
    manifest.validate_for_admission()
    envelope = {"manifest": manifest.to_dict(), "source_scope": scope,
                "completed_checkpoint": str(checkpoint),
                "checkpoint_scope": "Completed result is saved before the final source-provenance recheck; "
                                    "only the final output path denotes that this recheck passed"}
    # Preflight every fixed output field and directory before the expensive
    # causal replay or native calls. This does not execute the study.
    json.dumps(full._payload(envelope), allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = run_study(args.input, expected_sha256=args.expected_sha256)
    encoded = json.dumps(full._payload({**envelope, **result}), indent=2, allow_nan=False)+"\n"
    safe_write(checkpoint, lambda stream: stream.write(encoded))
    if current_git_source_provenance(ROOT, scope) != provenance:
        raise RuntimeError(f"source changed during bounded regional response; completed result retained at {checkpoint}")
    safe_write(args.output, lambda stream: stream.write(encoded))
    print(f"Wrote localized regional restoration study to {args.output}")


if __name__ == "__main__":
    main()
