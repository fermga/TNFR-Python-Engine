"""Offline regional identity and finite budgets on the retained control step.

Only the existing t=1.5 to t=1.75 native receipt is read. No graph, native
operator, pressure/phase kernel, trajectory or fitted identity score is used.
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

from benchmarks.thol_regional_balance_audit import (  # noqa: E402
    INPUT_PATH, INPUT_SHA256, INPUT_CLAIM, _ancestry, _bind_source_edges, _component_budget,
)
from benchmarks.thol_retained_reset_audit import _load, _literal_number  # noqa: E402
from benchmarks.thol_family_closure import _capture, _equal, _match_live_state, _reference  # noqa: E402
from benchmarks.thol_pressure_feedback import _payload  # noqa: E402
from tnfr.physics.support_transport import observe_regional_support_euler  # noqa: E402
from tnfr.research.claims import ClaimStatus  # noqa: E402
from tnfr.research.core_manifests import CoreExperimentManifest, current_git_source_provenance  # noqa: E402
from tnfr.utils.io import safe_write  # noqa: E402

DT = F(1, 4)
SOURCE_CONFIG = ("DNFR_WEIGHTS", "_dnfr_weights", "compute_delta_nfr", "_dnfr_hook_name", "vectorized_dnfr")


def _vector(values):
    return tuple(F(value) for value in values)


def _difference(after, before):
    return tuple(a-b for a, b in zip(after, before, strict=True))


def _record_ref(path, value):
    raw = json.dumps(_payload(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {"json_path": path, "sha256": hashlib.sha256(raw).hexdigest(),
            "scope": "Complete retained JSON record; opaque resource descriptors do not expose resource contents"}


def _one(trace, name):
    rows = [row for row in trace["boundaries"] if row["boundary"] == name]
    if len(rows) != 1 or rows[0]["outcome"] != "completed":
        raise ValueError(f"exactly one completed {name} boundary is required")
    return rows[0]


def _lineage_record(record, pairs, nodes):
    state = record["state"]
    hierarchy = {str(parent): [child] for parent, child in pairs}
    children = {**hierarchy, **{str(child): [] for _, child in pairs}}
    if tuple(state["nodes"]) != nodes or state["hierarchy"] != hierarchy or state["children"] != children:
        raise ValueError("regional lineage or ordered node space changed")
    attrs = record["node_attributes"]
    if tuple(node for node, _ in attrs) != nodes:
        raise ValueError("retained node metadata is incomplete or reordered")
    attrs = dict(attrs)
    for parent, child in pairs:
        if attrs[parent].get("sub_nodes") != [child] or attrs[child].get("parent_node") != parent:
            raise ValueError("regional retained parent pointers changed")


def _read_capture(capture, record):
    if capture.get("available") is not True:
        raise ValueError("complete forcing capture is required")
    snapshot, observation, components = _capture(capture["payload"])
    _match_live_state(record["state"], capture["payload"], snapshot)
    _bind_source_edges(record["state"], snapshot)
    return snapshot, observation, components


def _region_identity(before, after, balance, phase_before, phase_after):
    indices = balance.region_indices
    weights = balance.metric_weights

    def view(snapshot, phase):
        epi = tuple(snapshot.epi[i] for i in indices)
        mean = sum((weights[i]*snapshot.epi[i] for i in indices), F(0))/balance.regional_weight
        return {"nodes": balance.region, "epi": epi, "weighted_mean": mean,
                "centered_epi": tuple(x-mean for x in epi),
                "phase": tuple(phase[i] for i in indices), "capacity": tuple(snapshot.capacity[i] for i in indices)}

    left, right = view(before, phase_before), view(after, phase_after)
    return {"before": left, "endpoint": right,
            "same_relative_epi": left["centered_epi"] == right["centered_epi"],
            "same_absolute_epi": left["epi"] == right["epi"],
            "represented_phase_equal": left["phase"] == right["phase"],
            "raw_phase_change": _difference(right["phase"], left["phase"]),
            "same_capacity": left["capacity"] == right["capacity"],
            "weighted_mean_change": right["weighted_mean"]-left["weighted_mean"],
            "scope": "Only same_relative_epi permits a uniform regional translation. Node survival or equality "
                     "of these finite readouts does not certify a persistent temporal entity. Inequality of relative EPI "
                     "does not refute temporal NFR identity or persistence; an evolving region may retain either."}


def audit_branch(prior, branch):
    """Admit the retained control receipt and account for nine fixed regions."""
    if prior["branch"] != "control" or branch["branch"] != "control" or branch["status"] != "executed":
        raise ValueError("one completed retained control branch is required")
    before, endpoint, trace = branch["before"], branch["endpoint"], branch["native_trace"]
    if trace["status"] != "executed" or trace["native_calls"] != 1:
        raise ValueError("receipt must contain exactly one completed native call")
    _equal(before["state"], prior["endpoint"], "prior endpoint at the native source")
    if F(before["state"]["time"]) != F(3, 2) or F(endpoint["state"]["time"]) != F(7, 4):
        raise ValueError("the declared retained interval is t=1.5 to t=1.75")
    generation, integration, phase, adaptation = (_one(trace, name) for name in (
        "_prepare_dnfr", "integrate", "coordinate_global_local_phase", "adapt_vf_after_structural_stability"))
    _one(trace, "_refresh_delta_nfr")
    ordinals = tuple(row["ordinal"] for row in trace["boundaries"])
    if (any(type(i) is not int for i in ordinals) or len(set(ordinals)) != len(ordinals)
            or tuple(sorted(ordinals)) != ordinals
            or not generation["ordinal"] < integration["ordinal"] < phase["ordinal"] < adaptation["ordinal"]):
        raise ValueError("native boundary ordering is inconsistent")
    args = integration["effective_arguments"]
    if (args["method"] != "euler" or F(_literal_number(args["dt"])) != DT
            or integration["integrator_type"] != "tnfr.dynamics.integrators.DefaultIntegrator"):
        raise ValueError("retained default Euler invocation or duration differs")
    for row, side, expected_time in (
            (generation, "before", F(3, 2)), (generation, "after", F(3, 2)),
            (integration, "before", F(3, 2)), (integration, "after", F(7, 4)),
            (phase, "before", F(7, 4)), (phase, "after", F(7, 4)),
            (adaptation, "before", F(7, 4)), (adaptation, "after", F(7, 4))):
        if F(row[side]["state"]["time"]) != expected_time:
            raise ValueError("native boundary timestamp differs from the declared interval")
    bracket = trace["clamp_interval"]
    if (bracket["after_integrate_ordinal"] != integration["ordinal"]
            or bracket["before_phase_ordinal"] != phase["ordinal"]):
        raise ValueError("retained clamp bracket does not identify integration and phase boundaries")
    left, before_obs, _ = _read_capture(branch["before_capture"], before)
    entry, entry_obs, components = _read_capture(trace["captures"]["integrator_entry"], integration["before"])
    right, endpoint_obs, endpoint_components = _read_capture(branch["endpoint_capture"], endpoint)
    if not before_obs.normalized_weights == entry_obs.normalized_weights == endpoint_obs.normalized_weights:
        raise ValueError("normalized channel coefficients changed across the retained interval")
    original = _reference(prior["original_reference"])
    nodes = entry.nodes
    if nodes != original.source.nodes:
        raise ValueError("original reference node identity differs")
    pairs, parents, children = _ancestry(prior, before["state"], nodes)
    records = [("before", before), ("endpoint", endpoint)]
    for i, row in enumerate(trace["boundaries"]):
        if row["outcome"] != "completed":
            raise ValueError("retained interval contains an incomplete boundary")
        records.extend((f"native_trace.boundaries[{i}].{side}", row[side]) for side in ("before", "after"))
    for _, record in records:
        _lineage_record(record, pairs, nodes)
        if any(record["state"][key] != before["state"][key] for key in ("nodes", "capacity", "edges")):
            raise ValueError("regional node space, capacity or support changed")
    for field in ("nodes", "conductance", "support_neighbors", "capacity"):
        if any(getattr(snapshot, field) != getattr(entry, field) for snapshot in (left, right, original.source)):
            raise ValueError("held regional metric or full support differs")
    gen = generation["after"]
    for key in ("nodes", "epi", "phase", "capacity", "edges"):
        _equal(gen["state"][key], integration["before"]["state"][key], f"generation/entry {key}")
    _equal(gen["ordered_neighbors"], integration["before"]["ordered_neighbors"], "generation/entry neighbor order")
    for key in SOURCE_CONFIG:
        a, b = gen["graph_attributes"], integration["before"]["graph_attributes"]
        if (key in a, a.get(key)) != (key in b, b.get(key)):
            raise ValueError("pressure source configuration changed before integration")
    weights = gen["graph_attributes"]["_dnfr_weights"]
    _equal(tuple((name, F(_literal_number(weights[name]))) for name, _ in entry_obs.normalized_weights),
           entry_obs.normalized_weights, "entry capture normalized channel configuration")
    calls = [row for row in trace["boundaries"] if row["boundary"] == "apply_glyph"]
    if (len(calls) != len(nodes) or tuple(row["node"] for row in calls) != nodes
            or any(row["glyph"] != "IL" or not generation["ordinal"] < row["ordinal"] < integration["ordinal"] for row in calls)):
        raise ValueError("sixteen ordered actual IL calls are required")
    _equal(tuple((row["node"], row["glyph"], row["ordinal"], row["outcome"]) for row in calls),
           tuple((row["node"], row["glyph"], row["ordinal"], row["outcome"]) for row in trace["actual_glyph_calls"]),
           "actual glyph call summary")
    current_pressure = gen["state"]["pressure"]
    for i, row in enumerate(calls):
        _equal(row["before"]["state"]["pressure"], current_pressure, "IL pressure continuity")
        after_pressure = row["after"]["state"]["pressure"]
        if any(after_pressure[j] != current_pressure[j] for j in range(len(nodes)) if j != i):
            raise ValueError("a local IL call changed another node's pressure")
        current_pressure = after_pressure
    _equal(current_pressure, integration["before"]["state"]["pressure"], "held pressure consumed by integrator")
    x0, xf = before["state"]["epi"], endpoint["state"]["epi"]
    if entry.epi != left.epi:
        raise ValueError("an EPI reset preceded the retained integration")
    for row in trace["boundaries"]:
        for side in ("before", "after"):
            expected = x0 if row["ordinal"] < integration["ordinal"] or (row is integration and side == "before") else xf
            _equal(row[side]["state"]["epi"], expected, "staged EPI constancy outside integration")
            if row["ordinal"] < integration["ordinal"] or (row is integration and side == "before"):
                _equal(row[side]["state"]["phase"], before["state"]["phase"], "pre-integration represented phase constancy")
    _equal(integration["after"]["state"]["epi"], xf, "no post-integration EPI write")
    # Clamps can normalize phase between these boundaries; retain the change.
    for key in integration["after"]["state"]:
        if key != "phase":
            _equal(integration["after"]["state"][key], phase["before"]["state"][key], f"clamp bracket nonphase {key}")
    _equal(phase["after"]["state"]["phase"], endpoint["state"]["phase"], "phase endpoint after coordination")
    pg = _vector(gen["state"]["pressure"])
    model = tuple(entry_obs.epi_weight*g+f for g, f in zip(entry.epi_gradient, entry_obs.forcing, strict=True))
    split = {"generation_minus_model": _difference(pg, model),
             "IL_associated_write": _difference(entry.stored_pressure, pg),
             "fresh_kernel_minus_model": entry_obs.kernel_pressure_defect,
             "generation_minus_fresh_kernel": _difference(pg, entry_obs.full_kernel_pressure)}
    if any(g+k != p-m for g, k, p, m in zip(split["generation_minus_model"], split["IL_associated_write"],
                                            entry.stored_pressure, model, strict=True)):
        raise RuntimeError("held-pressure decomposition failed")
    region_rows = []
    for label, region in [(f"ancestry_pair_{i}", pair) for i, pair in enumerate(pairs)]+[("actual_children", children)]:
        budget = observe_regional_support_euler(
            entry, right, region, dt=DT, epi_weight=entry_obs.epi_weight, forcing=entry_obs.forcing)
        terms = {}
        for name, values in (*components, *split.items()):
            item = _component_budget(budget.balance, values)
            terms[name] = {
                **item, "weighted_total_first_order_term": DT*item["weighted_total_rate"],
                "variance_first_order_term": DT*item["variance_rate"]}
        for key, expected in (("weighted_total_rate", budget.balance.mass_defect_rate),
                              ("variance_rate", budget.balance.variance_defect_rate)):
            if terms["generation_minus_model"][key]+terms["IL_associated_write"][key] != expected:
                raise RuntimeError("regional first-order pressure attribution failed")
        identity = _region_identity(left, right, budget.balance, before_obs.phase, endpoint_obs.phase)
        region_rows.append({"label": label, "identity": identity,
                            "finite_budget": asdict(budget), "first_order_pressure_terms": terms})
    channel_delta = {name: _difference(dict(endpoint_components)[name], values) for name, values in components}
    return {"interval": (F(3, 2), F(7, 4)), "dt": DT, "regions": region_rows,
            "actual_lineage": {"pairs": pairs, "parents": parents, "children": children},
            "pressure_split": split, "source_channel_delta_endpoint_minus_entry": channel_delta,
            "normalized_channel_weights": {"before": before_obs.normalized_weights, "entry": entry_obs.normalized_weights,
                                           "endpoint": endpoint_obs.normalized_weights, "all_equal": True},
            "phase_by_stage": {"before": before_obs.phase, "generation": _vector(gen["state"]["phase"]),
                               "integration_entry": entry_obs.phase, "integration_exit": _vector(integration["after"]["state"]["phase"]),
                               "after_bracketed_normalization": _vector(phase["before"]["state"]["phase"]),
                               "after_coordination": _vector(phase["after"]["state"]["phase"]), "endpoint": endpoint_obs.phase},
            "generation_source_reuse": {
                "source": "native_trace.captures.integrator_entry", "derived_reuse": True,
                "scope": "No generation capture exists in this historical schema. Captured F is reused only "
                         "after equality of EPI, phase, capacity, support, neighbor order and source configuration."},
            "IL_factor_certified": False,
            "retained_records": [_record_ref(f"branches[0].{path}", value) for path, value in records],
            "retained_branch": _record_ref("branches[0]", branch),
            "retained_prior": _record_ref("replayed_prior_reports[0]", prior),
            "native_calls": 0, "new_trajectories": 0, "regional_observation_count": 9,
            "scope": "Finite retained regional identity and endpoint accounting only; no causal seal, fitted identity score, "
                     "source maintenance or temporal NFR persistence is inferred."}


def run_study(path=INPUT_PATH, *, expected_sha256=INPUT_SHA256):
    report, binding = _load(path, expected_sha256, INPUT_CLAIM)
    result = audit_branch(report["replayed_prior_reports"][0], report["branches"][0])
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != binding["sha256"]:
        raise RuntimeError("retained input changed during regional identity audit")
    return {"historical_input": binding, **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--expected-sha256", default=INPUT_SHA256)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/research/thol_regional_identity_audit_2026_09_18.json")
    args = parser.parse_args()
    if args.output.resolve() == args.input.resolve():
        raise ValueError("output must not overwrite retained input")
    scope = ("src/tnfr", "benchmarks/thol_regional_identity_audit.py", "benchmarks/thol_regional_balance_audit.py",
             "benchmarks/thol_retained_reset_audit.py", "benchmarks/thol_family_closure.py", "benchmarks/thol_pressure_feedback.py")
    provenance = current_git_source_provenance(ROOT, scope)
    result = run_study(args.input, expected_sha256=args.expected_sha256)
    if current_git_source_provenance(ROOT, scope) != provenance:
        raise RuntimeError("source changed during regional identity audit")
    sha, dirty, digest = provenance
    manifest = CoreExperimentManifest(
        claim_id="O3.a-retained-regional-identity", git_sha=sha, source_dirty=dirty, dirty_source_hash=digest,
        versions={"python": platform.python_version()}, graph_construction="Authenticated retained full-graph control records",
        capacity_specification="Fixed retained positive capacity and full degree metric", solver="Exact finite endpoint accounting",
        timestep=None, seed=None, result_status=ClaimStatus.DERIVED, operator_sequence=("coherence",),
        telemetry=("regional state identity", "finite regional budgets", "source changes", "pressure attribution"),
        controls=("no native calls", "nine fixed ancestry regions", "no pressure kernel replay"), artifacts=(str(args.output),))
    manifest.validate_for_admission()
    report = {"manifest": manifest.to_dict(), "source_scope": scope, **result}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    safe_write(args.output, lambda stream: stream.write(json.dumps(_payload(report), indent=2, allow_nan=False)+"\n"))
    print(f"Wrote retained regional identity audit to {args.output}")


if __name__ == "__main__":
    main()
