"""One bounded versioned phase-to-source comparison, with no native evolution."""

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

from benchmarks import thol_phase_source_relevance as source_owner  # noqa: E402
from benchmarks.thol_family_closure import _equal  # noqa: E402
from benchmarks.thol_native_runtime_response import _record  # noqa: E402
from benchmarks.thol_pressure_feedback import _payload  # noqa: E402
from benchmarks.thol_regional_identity_audit import _record_ref  # noqa: E402
from benchmarks.thol_retained_phase_audit import _detached_graph, _float_literal, compare_phases  # noqa: E402
from benchmarks.thol_retained_reset_audit import _load  # noqa: E402
from tnfr.alias import get_theta_attr  # noqa: E402
from tnfr.dynamics.coordination import coordinate_global_local_phase, UndefinedGlobalPhaseError  # noqa: E402
from tnfr.metrics.trig import neighbor_phase_mean_list  # noqa: E402
from tnfr.metrics.trig_cache import get_trig_cache  # noqa: E402
from tnfr.research.claims import ClaimStatus  # noqa: E402
from tnfr.research.core_manifests import CoreExperimentManifest, current_git_source_provenance  # noqa: E402
from tnfr.utils.io import safe_write  # noqa: E402

VERSION = "exact_components_v1"


class ComparisonAdmissionError(ValueError):
    """A declared study gate failed; completed detached records are retained."""

    def __init__(self, message, cases):
        super().__init__(message)
        self.report = {"status": "admission_obstruction", "reason": message, "coordinator_cases": cases,
                       "coordination_calls": len(cases), "forcing_capture_calls": 0, "native_calls": 0,
                       "scope": "No forcing comparison is admitted after this gate failure."}


def _run_case(record, nodes, gains, version):
    original_nodes = tuple(record["state"]["nodes"])
    graph = _detached_graph(record, dict(zip(original_nodes, original_nodes, strict=True)), node_order=nodes)
    graph.graph["_t"] = record["state"]["time"]
    construction_before = _record(graph)
    cache = get_trig_cache(graph)
    phases = tuple(float(cache.theta[n]) for n in nodes)
    components = tuple((float(cache.cos[n]), float(cache.sin[n])) for n in nodes)
    neighbors = tuple((n, tuple(graph.neighbors(n))) for n in nodes)
    local = tuple(neighbor_phase_mean_list(neigh, cache.cos, cache.sin, fallback=phase) if neigh else phase
                  for (_, neigh), phase in zip(neighbors, phases, strict=True))
    before = _record(graph)
    result = {"version": version, "nodes": tuple(nodes), "construction_before": construction_before, "before": before,
              "primitive_phases": phases, "components": components, "neighbor_order": neighbors,
              "local_targets": local, "fixed_gains": gains}
    try:
        evidence = coordinate_global_local_phase(
            graph, global_force=gains[0], local_force=gains[1], n_jobs=1, global_reduction=version)
    except UndefinedGlobalPhaseError as failure:
        result.update(status="undefined_global_direction", reason=str(failure), after=_record(graph), evidence=None)
        return result
    result.update(status="completed", after=_record(graph), evidence=None if evidence is None else asdict(evidence),
                  aligned_output=tuple(float(get_theta_attr(graph.nodes[n])) for n in original_nodes))
    for key in ("epi", "capacity", "pressure", "edges", "nodes"):
        _equal(result["after"]["state"][key], before["state"][key], f"detached coordination fixed {key}")
    _equal(result["after"]["ordered_neighbors"], before["ordered_neighbors"], "detached local neighbor order")
    if evidence is not None:
        _equal(evidence.primitive_phases, phases, "consumed primitive phases")
        _equal(evidence.resultant.components, components, "consumed represented components")
        _equal(evidence.local_targets, local, "unchanged local target owner")
        _equal((evidence.effective_global_force, evidence.effective_local_force), gains, "fixed effective gains")
        _equal(evidence.neighbor_order, neighbors, "consumed local order")
        _equal(evidence.realized_phases, tuple(get_theta_attr(graph.nodes[n]) for n in nodes), "realized exact phases")
    return result


def _aligned(case, field, nodes):
    values = dict(zip(case["nodes"], case[field], strict=True))
    return tuple(values[n] for n in nodes)


def compare_versions(admitted, source_boundary, effective_gains):
    """At most three detached calls and two captures on one admitted endpoint."""
    _equal(_record_ref(admitted["source_boundary"]["json_path"], source_boundary),
           admitted["source_boundary"], "phase source boundary content binding")
    gains = tuple(_float_literal(source_boundary["after"]["graph_attributes"][name])
                  for name in ("PHASE_K_GLOBAL", "PHASE_K_LOCAL"))
    _equal(effective_gains, gains, "archived effective phase gains")
    record = source_boundary["before"]
    nodes = tuple(admitted["record"]["state"]["nodes"])
    _equal(record["state"]["nodes"], nodes, "source and fixed endpoint node order")
    cases = []
    for version, order in (("legacy", nodes), (VERSION, nodes), (VERSION, tuple(reversed(nodes)))):
        case = _run_case(record, order, gains, version)
        cases.append(case)
        if case["status"] != "completed":
            raise ComparisonAdmissionError("exact global direction is unavailable", cases)
        if len(cases) == 1 and tuple(map(F, case["aligned_output"])) != admitted["phases"]["baseline"]:
            raise ComparisonAdmissionError("legacy reference differs from archived phase output", cases)
        for field in ("primitive_phases", "components", "local_targets"):
            if _aligned(case, field, nodes) != _aligned(cases[0], field, nodes):
                raise ComparisonAdmissionError(f"aligned {field} changed across the frozen comparison", cases)
        _equal(dict(case["neighbor_order"]), dict(cases[0]["neighbor_order"]), "fixed named local neighbor order")
    first, exact, reversed_case = cases
    sums = tuple(exact["evidence"]["resultant"][key] for key in ("real_sum", "imag_sum"))
    reverse_sums = tuple(reversed_case["evidence"]["resultant"][key] for key in ("real_sum", "imag_sum"))
    if sums != reverse_sums:
        raise ComparisonAdmissionError("exact represented global sums differ under reversal", cases)
    aligned_proposals = tuple(dict(zip(exact["nodes"], exact["evidence"]["raw_proposals"], strict=True))[n] for n in nodes)
    reversed_proposals = tuple(dict(zip(reversed_case["nodes"], reversed_case["evidence"]["raw_proposals"], strict=True))[n]
                               for n in nodes)
    context = {"reference": {"version": "legacy", "origin": "new detached replay exactly matching archived endpoint"},
               "alternative": {"version": VERSION, "origin": "new detached source-order coordination output"},
               "archived_alternative_phase": admitted["phases"]["alternative"],
               "archived_alternative_scope": "Previous canonical-enumeration output is retained as historical metadata; "
                                             "it is not the new exact-version alternative.",
               "alternate_node_order": tuple(reversed(nodes)), "local_neighbor_order_changed": False}
    comparison_input = {**admitted, "phases": {**admitted["phases"], "alternative": tuple(map(F, exact["aligned_output"]))},
                        "phase_alignment_scope": "Reference and NEW exact-version output are aligned to original nodes. "
                                                 "Both forcing captures use original endpoint and local order."}
    rates = source_owner.evaluate_admitted(comparison_input)
    rates["scope"] = "Shared exact regional accounting of the archived reference versus the newly computed exact-version phase. " \
                     "Historical stored rates remain fixed; this is neither an executed future nor maintenance evidence."
    return {"status": "completed", "coordinator_cases": cases, "comparison_context": context,
            "enumeration_control": {"exact_sums_equal": True,
                                    "aligned_raw_proposals_equal": aligned_proposals == reversed_proposals,
                                    "aligned_realized_phases_equal": exact["aligned_output"] == reversed_case["aligned_output"],
                                    "phase_comparison": compare_phases(exact["aligned_output"], reversed_case["aligned_output"])},
            "legacy_vs_exact": compare_phases(first["aligned_output"], exact["aligned_output"]),
            "source_comparison": rates, "coordination_calls": 3, "forcing_capture_calls": 2, "native_calls": 0,
            "scope": "Three fixed-gain detached coordinator calls and two endpoint forcing captures only. "
                     "Remaining normalized-output differences are retained, not fitted away. No trajectory or maintenance claim."}


def run_study(native_path=source_owner.INPUT_PATH, phase_path=source_owner.PHASE_PATH,
              identity_path=source_owner.IDENTITY_PATH, *, expected_native_sha256=source_owner.INPUT_SHA256,
              expected_phase_sha256=source_owner.PHASE_SHA256, expected_identity_sha256=source_owner.IDENTITY_SHA256):
    native, native_binding = _load(native_path, expected_native_sha256, source_owner.INPUT_CLAIM)
    phase, phase_binding = _load(phase_path, expected_phase_sha256, "O3.a-retained-phase-conditioning")
    identity, identity_binding = _load(identity_path, expected_identity_sha256, "O3.a-retained-regional-identity")
    admitted = source_owner.admit_evidence(native, phase, identity, native_sha256=expected_native_sha256)
    try:
        result = compare_versions(admitted, phase["audit"]["source_boundary"], phase["audit"]["effective_gains"])
    except ComparisonAdmissionError as failure:
        result = failure.report
    bindings = {"native": native_binding, "phase": phase_binding, "regional_identity": identity_binding}
    for binding in bindings.values():
        if hashlib.sha256(Path(binding["path"]).read_bytes()).hexdigest() != binding["sha256"]:
            raise RuntimeError("retained bytes changed during versioned phase comparison")
    return {"inputs": bindings, **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=source_owner.INPUT_PATH)
    parser.add_argument("--phase-input", type=Path, default=source_owner.PHASE_PATH)
    parser.add_argument("--identity-input", type=Path, default=source_owner.IDENTITY_PATH)
    parser.add_argument("--expected-sha256", default=source_owner.INPUT_SHA256)
    parser.add_argument("--expected-phase-sha256", default=source_owner.PHASE_SHA256)
    parser.add_argument("--expected-identity-sha256", default=source_owner.IDENTITY_SHA256)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/research/thol_exact_phase_source_comparison_2026_09_18.json")
    args = parser.parse_args()
    if args.output.resolve() in {p.resolve() for p in (args.input, args.phase_input, args.identity_input)}:
        raise ValueError("output must not overwrite retained input")
    scope = ("src/tnfr", "benchmarks/thol_exact_phase_source_comparison.py", "benchmarks/thol_phase_source_relevance.py",
             "benchmarks/thol_regional_identity_audit.py", "benchmarks/thol_regional_balance_audit.py",
             "benchmarks/thol_retained_phase_audit.py", "benchmarks/thol_retained_reset_audit.py",
             "benchmarks/thol_family_closure.py", "benchmarks/thol_native_runtime_response.py", "benchmarks/thol_pressure_feedback.py")
    provenance = current_git_source_provenance(ROOT, scope)
    result = run_study(args.input, args.phase_input, args.identity_input, expected_native_sha256=args.expected_sha256,
                       expected_phase_sha256=args.expected_phase_sha256, expected_identity_sha256=args.expected_identity_sha256)
    if current_git_source_provenance(ROOT, scope) != provenance:
        raise RuntimeError("source changed during versioned phase comparison")
    sha, dirty, digest = provenance
    manifest = CoreExperimentManifest(
        claim_id="O3.a-exact-phase-source-comparison", git_sha=sha, source_dirty=dirty, dirty_source_hash=digest,
        versions={"python": platform.python_version(), "numpy": np.__version__, "networkx": nx.__version__},
        graph_construction="Detached retained phase boundary and frozen endpoint with original local order",
        capacity_specification="Retained positive capacity and complete conductance fixed", seed=None, timestep=None,
        solver="Legacy/exact global phase version comparison and shared exact regional source accounting",
        result_status=ClaimStatus.DERIVED, operator_sequence=(), telemetry=("phase evidence", "nine regional rate differences"),
        controls=("legacy archive admission", "frozen reverse enumeration", "two forcing captures", "no native evolution"),
        artifacts=(str(args.output),))
    manifest.validate_for_admission()
    report = {"manifest": manifest.to_dict(), "source_scope": scope, **result}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    safe_write(args.output, lambda stream: stream.write(json.dumps(_payload(report), indent=2, allow_nan=False)+"\n"))
    print(f"Wrote versioned phase source comparison to {args.output}")


if __name__ == "__main__":
    main()
