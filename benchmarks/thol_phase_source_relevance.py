"""Detached source sensitivity to two already archived phase outputs.

Exactly two forcing captures are allowed, on the same retained endpoint and
original node/neighbor enumeration. Coordination and native evolution do not run.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
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

from benchmarks.thol_family_closure import _capture, _equal, _match_live_state  # noqa: E402
from benchmarks.thol_pressure_feedback import _payload  # noqa: E402
from benchmarks.thol_regional_balance_audit import (  # noqa: E402
    INPUT_PATH, INPUT_SHA256, INPUT_CLAIM, _ancestry, _bind_source_edges, _component_budget,
)
from benchmarks.thol_regional_identity_audit import _record_ref  # noqa: E402
from benchmarks.thol_retained_phase_audit import _detached_graph as _phase_graph  # noqa: E402
from benchmarks.thol_retained_reset_audit import _load, _literal_number  # noqa: E402
from tnfr.dynamics.dnfr import default_compute_delta_nfr  # noqa: E402
from tnfr.physics.forcing_realization import (  # noqa: E402
    capture_non_epi_forcing, decompose_non_epi_forcing,
)
from tnfr.physics.support_transport import observe_regional_support_balance  # noqa: E402
from tnfr.research.claims import ClaimStatus  # noqa: E402
from tnfr.research.core_manifests import CoreExperimentManifest, current_git_source_provenance  # noqa: E402
from tnfr.utils.io import safe_write  # noqa: E402

PHASE_PATH = ROOT / "artifacts/research/thol_retained_phase_audit_2026_09_18.json"
PHASE_SHA256 = "b9e957bbfbc21ac68c3dfa31c818a9d46db3dbae08da3879b6d33c7991f7371a"
IDENTITY_PATH = ROOT / "artifacts/research/thol_regional_identity_audit_2026_09_18.json"
IDENTITY_SHA256 = "57cc9774c0645779d6e54df5fff2e0f1f33fe4bc7390bce5c90c04a869ab3745"
CHANNELS = ("phase", "epi", "vf", "topo")
DEFAULT_CALLBACK = {"callable_module": "tnfr.dynamics.dnfr", "callable_name": "default_compute_delta_nfr"}


def _difference(after, before):
    return tuple(a-b for a, b in zip(after, before, strict=True))


def _phases(raw, size):
    if not isinstance(raw, (list, tuple)) or len(raw) != size:
        raise ValueError("complete node-aligned phase vector required")
    return tuple(F(_literal_number(value)) for value in raw)


def _source_config(record, archived):
    """Admit the configured default branch; never replace an unknown callback."""
    data = record["graph_attributes"]
    if data.get("compute_delta_nfr") not in (None, DEFAULT_CALLBACK):
        raise ValueError("retained pressure callback is not the configured default")
    if data.get("_dnfr_hook_name") not in (None, "default_compute_delta_nfr"):
        raise ValueError("retained pressure hook name is not the default")
    if "vectorized_dnfr" in data and data["vectorized_dnfr"] is not True:
        raise ValueError("retained pressure branch is not default NumPy")
    cached = data["_dnfr_weights"]
    if set(cached) != set(CHANNELS):
        raise ValueError("all four retained cached pressure weights are required")
    decoded = {name: _literal_number(cached[name]) for name in CHANNELS}
    if any(value < 0 for value in decoded.values()):
        raise ValueError("retained pressure weights must be nonnegative")
    _equal(tuple((name, F(decoded[name])) for name in CHANNELS), archived.normalized_weights,
           "retained cached pressure weights")
    result = {"_dnfr_weights": decoded, "compute_delta_nfr": default_compute_delta_nfr}
    if "DNFR_WEIGHTS" in data:
        if set(data["DNFR_WEIGHTS"]) != set(CHANNELS):
            raise ValueError("complete configured channel weights required")
        result["DNFR_WEIGHTS"] = {name: _literal_number(data["DNFR_WEIGHTS"][name]) for name in CHANNELS}
    if "vectorized_dnfr" in data:
        result["vectorized_dnfr"] = data["vectorized_dnfr"]
    return result


def _initial_projection(record, mapping, order):
    """Reconstruct the saved enumeration projection without any phase kernel."""
    nodes = tuple(record["state"]["nodes"])
    neighbors = dict(record["ordered_neighbors"])
    if len(neighbors) != len(nodes) or set(neighbors) != set(nodes):
        raise ValueError("complete ordered neighbors required")
    adjacency = {mapping[n]: tuple(mapping[v] for v in neighbors[n]) for n in nodes}
    attributes = {frozenset((mapping[u], mapping[v])): data for u, v, data in record["state"]["edges"]}
    seen, edges = set(), []
    for node in order:
        for other in adjacency[node]:
            if other not in seen:
                edges.append((node, other, attributes[frozenset((node, other))]))
        seen.add(node)
    phases = {mapping[n]: value for n, value in zip(nodes, record["state"]["phase"], strict=True)}
    return {"nodes": tuple(order), "edges": edges, "ordered_neighbors": adjacency,
            "phase": tuple(phases[n] for n in order)}


def admit_evidence(native_report, phase_report, identity_report, *, native_sha256=INPUT_SHA256):
    """Pure admission of the saved endpoint, phase vectors and regional identity."""
    prior, branch = native_report["replayed_prior_reports"][0], native_report["branches"][0]
    if prior["branch"] != "control" or branch["branch"] != "control" or branch["status"] != "executed":
        raise ValueError("completed retained control branch required")
    record = branch["endpoint"]
    if F(record["state"]["time"]) != F(7, 4):
        raise ValueError("retained endpoint must be t=1.75")
    raw_capture = branch["endpoint_capture"]
    if raw_capture.get("available") is not True:
        raise ValueError("retained endpoint forcing capture is unavailable")
    snapshot, archived, _ = _capture(raw_capture["payload"])
    _match_live_state(record["state"], raw_capture["payload"], snapshot)
    _bind_source_edges(record["state"], snapshot)
    config = _source_config(record, archived)
    nodes = snapshot.nodes
    pairs, parents, children = _ancestry(prior, record["state"], nodes)
    selected = [(i, row) for i, row in enumerate(branch["native_trace"]["boundaries"])
                if row["boundary"] == "coordinate_global_local_phase"]
    if len(selected) != 1 or selected[0][1]["outcome"] != "completed":
        raise ValueError("one completed archived phase boundary required")
    source_index, source = selected[0]
    phase_audit = phase_report["audit"]
    if (phase_report["selected_branch"] != "control"
            or phase_report["input_evidence"]["sha256"] != native_sha256
            or identity_report["historical_input"]["sha256"] != native_sha256):
        raise ValueError("saved phase/identity evidence belongs to another native input")
    _equal(phase_audit["source_boundary"], source, "complete archived phase source boundary")
    for side in ("before", "after"):
        if F(source[side]["state"]["time"]) != F(7, 4):
            raise ValueError("saved coordination boundary must occur at t=1.75")
        _source_config(source[side], archived)
    for key in ("nodes", "epi", "capacity", "pressure", "edges"):
        _equal(source["before"]["state"][key], source["after"]["state"][key], f"fixed coordination {key}")
    _equal(source["before"]["ordered_neighbors"], source["after"]["ordered_neighbors"], "fixed coordination neighbor order")
    for key in ("nodes", "epi", "capacity", "phase", "pressure", "edges"):
        _equal(source["after"]["state"][key], record["state"][key], f"coordination endpoint {key}")
    _equal(source["after"]["ordered_neighbors"], record["ordered_neighbors"], "coordination endpoint neighbor order")
    if (phase_audit["rotation"] != 1 or type(phase_audit["rotation"]) is not int
            or phase_audit["transported_source_equal"] is not True
            or phase_audit["numpy_replay_matches_archive"] is not True):
        raise ValueError("saved phase transport control is not admitted")
    expected_mapping = {node: pairs[(i+1) % len(pairs)][j]
                        for i, pair in enumerate(pairs) for j, node in enumerate(pair)}
    mapping_rows = phase_audit["node_mapping"]
    if len(mapping_rows) != len(nodes) or dict(mapping_rows) != expected_mapping:
        raise ValueError("saved phase transport differs from actual ancestry rotation")
    mapping = dict(mapping_rows)
    identity = dict(zip(nodes, nodes, strict=True))
    for name, chart, order in (
            ("base_numpy", identity, nodes), ("transported_numpy", mapping, tuple(mapping[n] for n in nodes)),
            ("canonical_order_numpy", mapping, nodes)):
        _equal(phase_audit[name]["initial"], _initial_projection(source["before"], chart, order),
               f"saved {name} phase input enumeration")
    phases = {name: _phases(phase_audit[key]["phase"], len(nodes)) for name, key in (
        ("baseline", "base_numpy"), ("alternative", "canonical_order_numpy"), ("transported", "transported_numpy"))}
    if phases["baseline"] != archived.phase or phases["transported"] != phases["baseline"]:
        raise ValueError("saved baseline/transported phase differs from the native endpoint")
    _equal(identity_report["retained_branch"], _record_ref("branches[0]", branch), "regional identity branch binding")
    _equal(identity_report["retained_prior"], _record_ref("replayed_prior_reports[0]", prior), "regional identity prior binding")
    _equal(identity_report["interval"], (F(3, 2), F(7, 4)), "regional interval")
    _equal(identity_report["actual_lineage"], {"pairs": pairs, "parents": parents, "children": children}, "regional lineage")
    _equal(identity_report["phase_by_stage"]["endpoint"], archived.phase, "regional endpoint phase")
    _equal(identity_report["normalized_channel_weights"]["endpoint"], archived.normalized_weights, "regional weights")
    regions = tuple([(f"ancestry_pair_{i}", pair) for i, pair in enumerate(pairs)]+[("actual_children", children)])
    if tuple(row["label"] for row in identity_report["regions"]) != tuple(label for label, _ in regions):
        raise ValueError("saved regional set differs from the nine actual ancestry regions")
    for row, (_, region) in zip(identity_report["regions"], regions, strict=True):
        _equal(row["finite_budget"]["after"], asdict(snapshot), "regional endpoint snapshot")
        ids = tuple(nodes.index(n) for n in region)
        view = row["identity"]["endpoint"]
        for key, values in (("epi", snapshot.epi), ("capacity", snapshot.capacity), ("phase", archived.phase)):
            _equal(view[key], tuple(values[i] for i in ids), f"regional endpoint {key}")
        _equal(view["nodes"], region, "regional ordered membership")
    return {"record": deepcopy(record), "config": config, "phases": phases, "regions": regions,
            "archived_observation": archived,
            "source_boundary": _record_ref(f"branches[0].native_trace.boundaries[{source_index}]", source),
            "endpoint_record": _record_ref("branches[0].endpoint", record),
            "transported_equal_input_control": True,
            "phase_alignment_scope": "Saved phase outputs are already aligned to original source node identities. "
                                     "Both pressure evaluations retain original endpoint node and neighbor order."}


def _detached_graph(record, phases, config):
    copied = deepcopy(record)
    nodes = tuple(record["state"]["nodes"])
    copied["state"]["phase"] = [float(value) for value in phases]
    graph = _phase_graph(copied, dict(zip(nodes, nodes, strict=True)))
    graph.graph.update(deepcopy(config))
    return graph


def _baseline_matches(admitted, baseline):
    _equal(asdict(baseline), asdict(admitted["archived_observation"]), "complete baseline fresh forcing capture")


def audit_captures(admitted, baseline, alternative):
    """Compare exact regional model rates, with represented-kernel residuals separate."""
    _baseline_matches(admitted, baseline)
    base_components = dict(decompose_non_epi_forcing(baseline))
    alt_components = dict(decompose_non_epi_forcing(alternative))
    _capture(_payload({"observation": asdict(alternative), "components": tuple(alt_components.items())}))
    _equal(asdict(alternative.snapshot), asdict(baseline.snapshot), "fixed endpoint support and scalar state")
    _equal(alternative.phase, admitted["phases"]["alternative"], "alternative original-node phase alignment")
    _equal(alternative.normalized_weights, baseline.normalized_weights, "fixed channel weights")
    for name in ("vf", "topo"):
        _equal(alt_components[name], base_components[name], f"unchanged {name} component")
    delta_phase = _difference(alt_components["phase"], base_components["phase"])
    if _difference(alternative.forcing, baseline.forcing) != delta_phase:
        raise ValueError("forcing difference is not solely the phase-source difference")
    fresh_delta = _difference(alternative.full_kernel_pressure, baseline.full_kernel_pressure)
    kernel_delta = _difference(alternative.kernel_pressure_defect, baseline.kernel_pressure_defect)
    stored_residual_delta = _difference(alternative.stored_pressure_residual, baseline.stored_pressure_residual)
    if (fresh_delta != tuple(f+k for f, k in zip(delta_phase, kernel_delta, strict=True))
            or stored_residual_delta != tuple(-value for value in fresh_delta)):
        raise RuntimeError("fresh/model/kernel/stored residual difference identity failed")
    rows = []
    for label, region in admitted["regions"]:
        first = observe_regional_support_balance(
            baseline.snapshot, region, epi_weight=baseline.epi_weight, forcing=baseline.forcing)
        second = observe_regional_support_balance(
            alternative.snapshot, region, epi_weight=alternative.epi_weight, forcing=alternative.forcing)
        for key in ("mass_boundary_rate", "internal_dissipation", "variance_boundary_rate",
                    "stored_mass_rate", "stored_variance_rate", "mean", "variance"):
            if getattr(first, key) != getattr(second, key):
                raise ValueError(f"fixed endpoint regional {key} changed")
        contribution = _component_budget(first, delta_phase)
        mass, variance = contribution["weighted_total_rate"], contribution["variance_rate"]
        if second.model_mass_rate-first.model_mass_rate != mass or second.model_variance_rate-first.model_variance_rate != variance:
            raise RuntimeError("exact regional phase-source difference identity failed")
        rows.append({"label": label, "region": region, "baseline": asdict(first), "alternative": asdict(second),
                     "model_weighted_total_rate_difference": mass, "model_variance_rate_difference": variance,
                     "weighted_total_difference_identity_residual": F(0), "variance_difference_identity_residual": F(0),
                     "internal_boundary_stored_rates_unchanged": True})
    return {"regions": rows, "phase_component_difference": delta_phase,
            "baseline_capture": {"observation": asdict(baseline), "components": tuple(base_components.items())},
            "alternative_capture": {"observation": asdict(alternative), "components": tuple(alt_components.items())},
            "fresh_kernel_pressure_difference": fresh_delta,
            "kernel_pressure_defect_difference": kernel_delta,
            "stored_pressure_residual_difference": stored_residual_delta,
            "model_nodal_rate_difference": tuple(nu*f for nu, f in zip(baseline.snapshot.capacity, delta_phase, strict=True)),
            "fresh_kernel_nodal_rate_difference": tuple(nu*p for nu, p in zip(baseline.snapshot.capacity, fresh_delta, strict=True)),
            "stored_nodal_rate_difference": tuple(F(0) for _ in baseline.snapshot.nodes),
            "transported_equal_input_control": admitted["transported_equal_input_control"],
            "phase_alignment_scope": admitted["phase_alignment_scope"],
            "source_boundary": admitted["source_boundary"], "endpoint_record": admitted["endpoint_record"],
            "regional_observations": 18, "native_calls": 0, "coordination_calls": 0,
            "scope": "Detached sensitivity of one retained endpoint to two saved phase outputs. Model-rate differences "
                     "are not actual future refreshes, trajectory differences, universal error bounds or maintenance evidence."}


def evaluate_admitted(admitted):
    """Exactly two ordered captures; a failed baseline prevents the second call."""
    baseline = capture_non_epi_forcing(_detached_graph(admitted["record"], admitted["phases"]["baseline"], admitted["config"]))
    _baseline_matches(admitted, baseline)
    alternative = capture_non_epi_forcing(_detached_graph(admitted["record"], admitted["phases"]["alternative"], admitted["config"]))
    return {"forcing_capture_calls": 2, **audit_captures(admitted, baseline, alternative)}


def run_study(native_path=INPUT_PATH, phase_path=PHASE_PATH, identity_path=IDENTITY_PATH, *,
              expected_native_sha256=INPUT_SHA256, expected_phase_sha256=PHASE_SHA256,
              expected_identity_sha256=IDENTITY_SHA256):
    native, native_binding = _load(native_path, expected_native_sha256, INPUT_CLAIM)
    phase, phase_binding = _load(phase_path, expected_phase_sha256, "O3.a-retained-phase-conditioning")
    identity, identity_binding = _load(identity_path, expected_identity_sha256, "O3.a-retained-regional-identity")
    admitted = admit_evidence(native, phase, identity, native_sha256=expected_native_sha256)
    result = evaluate_admitted(admitted)
    bindings = {"native": native_binding, "phase": phase_binding, "regional_identity": identity_binding}
    for binding in bindings.values():
        if hashlib.sha256(Path(binding["path"]).read_bytes()).hexdigest() != binding["sha256"]:
            raise RuntimeError("retained source changed during phase relevance audit")
    return {"inputs": bindings, **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--phase-input", type=Path, default=PHASE_PATH)
    parser.add_argument("--identity-input", type=Path, default=IDENTITY_PATH)
    parser.add_argument("--expected-sha256", default=INPUT_SHA256)
    parser.add_argument("--expected-phase-sha256", default=PHASE_SHA256)
    parser.add_argument("--expected-identity-sha256", default=IDENTITY_SHA256)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/research/thol_phase_source_relevance_2026_09_18.json")
    args = parser.parse_args()
    if args.output.resolve() in {p.resolve() for p in (args.input, args.phase_input, args.identity_input)}:
        raise ValueError("output must not overwrite any retained input")
    scope = ("src/tnfr", "benchmarks/thol_phase_source_relevance.py", "benchmarks/thol_regional_identity_audit.py",
             "benchmarks/thol_regional_balance_audit.py", "benchmarks/thol_retained_reset_audit.py",
             "benchmarks/thol_retained_phase_audit.py", "benchmarks/thol_family_closure.py", "benchmarks/thol_pressure_feedback.py")
    provenance = current_git_source_provenance(ROOT, scope)
    result = run_study(args.input, args.phase_input, args.identity_input, expected_native_sha256=args.expected_sha256,
                       expected_phase_sha256=args.expected_phase_sha256, expected_identity_sha256=args.expected_identity_sha256)
    if current_git_source_provenance(ROOT, scope) != provenance:
        raise RuntimeError("source changed during phase relevance audit")
    sha, dirty, digest = provenance
    manifest = CoreExperimentManifest(
        claim_id="O3.a-retained-phase-source-relevance", git_sha=sha, source_dirty=dirty, dirty_source_hash=digest,
        versions={"python": platform.python_version(), "numpy": np.__version__, "networkx": nx.__version__},
        graph_construction="Two detached scalar projections of one authenticated t=1.75 endpoint",
        capacity_specification="Retained positive capacity and complete conductance fixed",
        solver="Two default NumPy forcing captures and exact regional rate differences", timestep=None, seed=None,
        result_status=ClaimStatus.DERIVED, operator_sequence=(), telemetry=("phase source", "nine regional rate differences"),
        controls=("baseline exact capture match", "transported equal input", "no coordination or native evolution"),
        artifacts=(str(args.output),))
    manifest.validate_for_admission()
    report = {"manifest": manifest.to_dict(), "source_scope": scope, **result}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    safe_write(args.output, lambda stream: stream.write(json.dumps(_payload(report), indent=2, allow_nan=False)+"\n"))
    print(f"Wrote retained phase source relevance to {args.output}")


if __name__ == "__main__":
    main()
