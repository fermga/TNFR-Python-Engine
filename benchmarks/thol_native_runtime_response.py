"""Trace one native composite step after the two causal full-EPI responses.

The fixed-model Euler counterfactual is frozen before native execution. The
trace observes existing boundaries without replacing selectors or evolution.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timedelta
from fractions import Fraction
from functools import wraps
import hashlib
import json
from pathlib import Path
import platform
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

from benchmarks.thol_full_state_response import (  # noqa: E402
    BRANCHES, COMMON_FIELDS, _add, _digest, _forcing_readout, _mv,
    _paired_delta, _payload, _reference, _state, _subtract, replay_response_branch,
)
from benchmarks.thol_preparation_policy import _literal  # noqa: E402
from tnfr import dynamics  # noqa: E402
from tnfr.alias import get_attr  # noqa: E402
from tnfr.constants.aliases import ALIAS_DEPI, ALIAS_SI  # noqa: E402
from tnfr.config import get_param  # noqa: E402
from tnfr.dynamics import adaptation, coordination, integrators, runtime, selectors  # noqa: E402
from tnfr.metrics.common import compute_coherence  # noqa: E402
from tnfr.operators.grammar_types import StructuralGrammarError  # noqa: E402
from tnfr.operators.preconditions import OperatorPreconditionError  # noqa: E402
from tnfr.operators.factor_contracts import resolve_runtime_operator_factors  # noqa: E402
from tnfr.types import Glyph  # noqa: E402
from tnfr.physics.epi_memory import predict_forced_support_realization_euler  # noqa: E402
from tnfr.physics.forced_support import observe_forced_support_pattern  # noqa: E402
from tnfr.physics.telemetry import compute_structural_telemetry  # noqa: E402
from tnfr.research.claims import ClaimStatus  # noqa: E402
from tnfr.research.core_manifests import (  # noqa: E402
    CoreExperimentManifest, current_git_source_provenance,
)
from tnfr.utils.io import safe_write  # noqa: E402
from tnfr.sdk._state import copy_graph_state  # noqa: E402

PRIOR_PATH = ROOT / "artifacts/research/thol_full_state_response_2026_09_18.json"
PRIOR_SHA256 = "5d95176d28ac633351337d90f3bac0c0fc7edd3e24cf68cc04a5df7de125987e"
DT = Fraction(1, 4)
PROTOCOL = {
    "branches": BRANCHES, "start_time": 1.5, "dt": DT,
    "native_calls_per_branch": 1, "use_Si": True, "apply_glyphs": True,
    "fallback": None, "parameter_tuning": False, "horizon_search": False,
    "scope": "One complete native step; fixed-model prediction is a counterfactual",
}


def load_prior_evidence(path=PRIOR_PATH, *, expected_sha256=PRIOR_SHA256):
    if (type(expected_sha256) is not str or len(expected_sha256) != 64
            or any(c not in "0123456789abcdef" for c in expected_sha256)):
        raise ValueError("expected SHA256 must be 64 lowercase hexadecimal characters")
    path = Path(path)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("prior full-state artifact digest mismatch")
    report = json.loads(raw)
    CoreExperimentManifest(**report["manifest"]).validate_for_admission()
    if (report["manifest"]["claim_id"] != "O1.b-generated-full-EPI-state-response"
            or tuple(row["branch"] for row in report["branches"]) != BRANCHES):
        raise ValueError("prior artifact has a different claim or branch identity")
    return report, {"path": str(path), "sha256": expected_sha256,
                    "historical_manifest": deepcopy(report["manifest"])}


def _admit_replay(prior, retained):
    """Compare every field except three explicit AL UTC metadata locations."""
    current, historical = _payload(prior), deepcopy(retained)
    full_equal = current == historical
    differences = []
    if prior["branch"] == "child_emission":
        expected_children = tuple(prior["lineage"]["children"])
        for payload, label in ((current, "current"), (historical, "historical")):
            attrs = dict(payload["event"]["after_node_attributes"])
            if set(attrs) != set(payload["event"]["after"]["nodes"]):
                raise ValueError("AL metadata does not cover its exact event node space")
            for node in expected_children:
                data = attrs[node]
                timestamp = data["emission_timestamp"]
                if (type(timestamp) is not str or datetime.fromisoformat(timestamp).utcoffset() != timedelta(0)
                        or data["_emission_origin"] != timestamp
                        or data["_structural_lineage"]["origin"] != timestamp):
                    raise ValueError("AL UTC origin aliases are inconsistent")
                differences.append({"payload": label, "node": node, "utc_origin": timestamp})
                # These are detached comparison values. Live metadata is never rewritten.
                data["emission_timestamp"] = "<invocation UTC origin>"
                data["_emission_origin"] = "<invocation UTC origin>"
                data["_structural_lineage"]["origin"] = "<invocation UTC origin>"
    if current != historical:
        raise ValueError("complete scientific replay differs outside the declared AL UTC metadata")
    return {"full_payload_equal": full_equal, "scientific_payload_equal": True,
            "utc_origin_records": differences,
            "exempt_paths": ("event.after_node_attributes[actual_child].emission_timestamp",
                             "event.after_node_attributes[actual_child]._emission_origin",
                             "event.after_node_attributes[actual_child]._structural_lineage.origin"),
            "scope": "Only invocation UTC labels differ; physical time, histories and lifecycle counters compare exactly"}


def _metadata(value):
    """Use the shared exact literal owner; identify opaque resources explicitly."""
    try:
        return _literal(value)
    except TypeError:
        if isinstance(value, np.ndarray):
            return {"array_dtype": str(value.dtype), "shape": value.shape,
                    "values": _literal(value.tolist())}
        if callable(value):
            return {"callable_module": getattr(value, "__module__", type(value).__module__),
                    "callable_name": getattr(value, "__qualname__", type(value).__qualname__)}
        return {"unavailable": "Opaque runtime resource; neither serialized nor interpreted",
                "type": f"{type(value).__module__}.{type(value).__qualname__}"}


def _record(graph):
    callbacks = graph.graph.get("callbacks", {})
    return {
        "state": deepcopy(_state(graph)),
        "node_attributes": tuple((node, {key: _metadata(value) for key, value in data.items()})
                                 for node, data in graph.nodes(data=True)),
        "graph_attributes": {key: _metadata(value) for key, value in graph.graph.items()},
        "ordered_neighbors": tuple((node, tuple(graph.neighbors(node))) for node in graph),
        "coherence": compute_coherence(graph),
        "stored_Si": tuple(get_attr(graph.nodes[node], ALIAS_SI, None) for node in graph),
        "stored_dEPI": tuple(get_attr(graph.nodes[node], ALIAS_DEPI, 0.0) for node in graph),
        "resolved_adaptation_configuration": {key: get_param(graph, key) for key in (
            "VF_ADAPT_TAU", "VF_ADAPT_MU", "EPS_DNFR_STABLE")},
        "callback_registry": {str(event): {"count": len(entries), "entries": tuple(
            (name, _metadata(getattr(spec, "func", spec))) for name, spec in entries.items())}
            for event, entries in callbacks.items() if isinstance(entries, dict)},
        "scope": "Read-only boundary record; opaque runtime resources have explicit unavailable markers",
    }


def _tetrad(graph):
    before = _record(graph)
    detached = copy_graph_state(graph)
    result = compute_structural_telemetry(detached)
    if _record(graph) != before:
        raise RuntimeError("detached structural telemetry changed the live graph")
    return {"fields": _literal({key: result[key] for key in ("phi_s", "grad_phi", "curv_phi", "xi_c")}),
            "scope": "Shared structural telemetry on a detached graph; not a complete state or evolution certificate"}


def _capture(graph):
    # ValueError denotes the documented detached capture domain, not execution
    # refusal. No runtime exception is hidden by this optional refinement.
    try:
        return {"available": True, "payload": _forcing_readout(graph)}
    except ValueError as exc:
        return {"available": False, "reason": str(exc), "scope": "Detached forcing readout unavailable"}


def _trace_step(graph, *, capture_generation=False):
    """Call every original exactly once; restore patched bindings on every exit."""
    records, requests, glyphs = [], [], []
    captured = {}
    resolver_depth = 0

    def instrument(owner, name, graph_position=0):
        original = getattr(owner, name)

        @wraps(original)
        def wrapped(*args, **kwargs):
            nonlocal resolver_depth
            current = args[graph_position]
            if current is not graph:
                return original(*args, **kwargs)
            row = {"ordinal": len(records), "boundary": name, "before": _record(graph)}
            records.append(row)
            if name == "integrate":
                captured["integrator_entry"] = _capture(graph)
                row["effective_arguments"] = _metadata(kwargs)
                row["integrator_type"] = f"{type(args[0]).__module__}.{type(args[0]).__qualname__}"
            if name == "apply_glyph":
                row.update(node=args[1], glyph=str(getattr(args[2], "value", args[2])))
                glyphs.append(row)
                if capture_generation and row["glyph"] == "IL":
                    row["resolved_IL_retention"] = resolve_runtime_operator_factors(
                        graph.graph.get("GLYPH_FACTORS"), Glyph.IL, graph.graph)["IL_dnfr_factor"]
            if name == "_resolve_preselected_glyph":
                row["node"] = args[1]
                row["resolver_nesting_depth"] = resolver_depth
                resolver_depth += 1
            try:
                result = original(*args, **kwargs)
            except BaseException as exc:
                row.update(outcome="raised", error_type=type(exc).__name__, error=str(exc),
                           after=_record(graph))
                raise
            finally:
                if name == "_resolve_preselected_glyph":
                    resolver_depth -= 1
            row.update(outcome="completed", after=_record(graph))
            if capture_generation and name == "_prepare_dnfr":
                captured["pressure_generation"] = _capture(graph)
            if name == "_resolve_preselected_glyph":
                row["requested_glyph"] = str(getattr(result, "value", result))
                if row["resolver_nesting_depth"] == 0:
                    requests.append({"node": args[1], "glyph": row["requested_glyph"],
                                     "trace_ordinal": row["ordinal"]})
            if name == "_apply_selector":
                row["resolved_selector"] = _metadata(result)
            return result

        return patch.object(owner, name, wrapped)

    runtime_names = (
        "_run_before_callbacks", "_record_mutation_flow_boundary", "_update_node_sample",
        "_refresh_delta_nfr", "_prepare_dnfr", "_advance_math_engine", "_update_epi_hist",
        "_maybe_remesh", "_run_validators", "_run_after_callbacks", "publish_graph_cache_metrics",
    )
    status, failure = "executed", None
    with ExitStack() as stack:
        for name in runtime_names:
            stack.enter_context(instrument(runtime, name))
        for name in ("_apply_selector", "_apply_glyphs", "_resolve_preselected_glyph", "apply_glyph"):
            stack.enter_context(instrument(selectors, name))
        stack.enter_context(instrument(dynamics, "compute_Si"))
        stack.enter_context(instrument(integrators.DefaultIntegrator, "integrate", 1))
        stack.enter_context(instrument(coordination, "coordinate_global_local_phase"))
        stack.enter_context(instrument(adaptation, "adapt_vf_after_structural_stability"))
        try:
            dynamics.step(graph, dt=float(DT), use_Si=True, apply_glyphs=True)
        except (OperatorPreconditionError, StructuralGrammarError) as exc:
            status, failure = "refused", {"type": type(exc).__name__, "reason": str(exc)}
    integration = [row for row in records if row["boundary"] == "integrate" and row["outcome"] == "completed"]
    phase = [row for row in records if row["boundary"] == "coordinate_global_local_phase"]
    clamp_interval = None
    if len(integration) == len(phase) == 1:
        clamp_interval = {"after_integrate_ordinal": integration[0]["ordinal"],
                          "before_phase_ordinal": phase[0]["ordinal"],
                          "scope": "Canonical clamps are bracketed by these states; individual calls are not traced"}
    return {"status": status, "failure": failure, "boundaries": records,
            "selector_requests": requests,
            "selector_request_scope": "Outermost preselected resolution returns; recursive implementation calls remain in boundaries",
            "clamp_interval": clamp_interval,
            "actual_glyph_calls": tuple({key: row[key] for key in ("node", "glyph", "ordinal", "outcome")}
                                        for row in glyphs),
            "captures": captured, "native_calls": 1,
            "scope": "Native sequential selection/grammar/execution; no simultaneous-stage claim or whole-step rollback"}


def _state_epi(record):
    return tuple(Fraction(value) for value in record["state"]["epi"])


def _pattern(reference, record):
    if tuple(record["state"]["nodes"]) != reference.source.nodes:
        return {"available": False, "reason": "Ordered node space changed"}
    return {"available": True, "payload": asdict(observe_forced_support_pattern(
        reference, nodes=reference.source.nodes, epi=_state_epi(record)))}


def _runtime_ledger(prediction, before, endpoint, trace):
    integration = [row for row in trace["boundaries"] if row["boundary"] == "integrate"]
    if len(integration) != 1 or integration[0]["outcome"] != "completed":
        return {"available": False, "reason": "No unique completed traced default integration"}
    row = integration[0]
    nodes = prediction.realization.closure.nodes
    records = (before, row["before"], row["after"], endpoint)
    if any(tuple(record["state"]["nodes"]) != nodes for record in records):
        return {"available": False, "reason": "Ordered node space changed"}
    x0, xg, xi, xf = tuple(map(_state_epi, records))
    closure = prediction.realization.closure
    old_drive = _subtract(closure.affine_source, _mv(closure.micro_generator, xg))
    state = row["before"]["state"]
    stored_rate = tuple(Fraction(nu)*Fraction(p) for nu, p in zip(state["capacity"], state["pressure"], strict=True))
    jump = _subtract(xg, x0)
    transported_jump = _mv(prediction.fine_euler_matrices[0], jump)
    changed_consumed_drive = tuple(DT*v for v in _subtract(stored_rate, old_drive))
    integration_remainder = _subtract(_subtract(xi, xg), tuple(DT*v for v in stored_rate))
    after_integration = _subtract(xf, xi)
    total = _subtract(xf, prediction.frames[-1].epi)
    summed = _add(_add(transported_jump, changed_consumed_drive), _add(integration_remainder, after_integration))
    if total != summed:
        raise RuntimeError("exact native-step displacement ledger failed")
    refined = {"available": False, "reason": "Current forcing capture unavailable"}
    capture = trace["captures"]["integrator_entry"]
    if capture["available"]:
        obs = capture["payload"]["observation"]
        snap = obs["snapshot"]
        if (tuple(snap["nodes"]) != nodes or tuple(snap["epi"]) != xg
                or tuple(snap["capacity"]) != tuple(map(Fraction, state["capacity"]))
                or tuple(snap["stored_pressure"]) != tuple(map(Fraction, state["pressure"]))
                or tuple(obs["phase"]) != tuple(map(Fraction, state["phase"]))):
            raise ValueError("forcing capture does not bind the actual integrator entry")
        current_drive = tuple(nu*(obs["epi_weight"]*gradient+forcing) for nu, gradient, forcing in zip(
            snap["capacity"], snap["epi_gradient"], obs["forcing"], strict=True))
        model_change = tuple(DT*v for v in _subtract(current_drive, old_drive))
        stored_fresh = tuple(DT*nu*(stored-fresh) for nu, stored, fresh in zip(
            snap["capacity"], snap["stored_pressure"], obs["full_kernel_pressure"], strict=True))
        kernel_reference = tuple(DT*nu*defect for nu, defect in zip(
            snap["capacity"], obs["kernel_pressure_defect"], strict=True))
        if _add(_add(model_change, stored_fresh), kernel_reference) != changed_consumed_drive:
            raise RuntimeError("refined native drive ledger failed")
        refined = {"available": True, "current_reference_drive": current_drive,
                   "model_change": model_change, "stored_minus_fresh_kernel": stored_fresh,
                   "fresh_kernel_minus_exact_reference": kernel_reference,
                   "identity_residual": (Fraction(0),)*len(nodes)}
    return {
        "available": True, "initial_epi": x0, "integration_entry_epi": xg,
        "integration_exit_epi": xi, "final_epi": xf, "preintegration_jump": jump,
        "transported_jump": transported_jump, "old_model_drive_at_integration_entry": old_drive,
        "actual_stored_nodal_rate": stored_rate, "changed_consumed_drive": changed_consumed_drive,
        "integration_remainder": integration_remainder, "postintegration_epi_change": after_integration,
        "actual_minus_held_forecast": total, "refined_drive": refined,
        "identity_residual": (Fraction(0),)*len(nodes),
        "scope": "Integration remainder may include substeps, Gamma or clipping; not identified as rounding",
    }


def run_native_branch(graph, prior_report):
    if prior_report["status"] != "executed" or _state(graph) != prior_report["endpoint"]:
        raise ValueError("native step requires the exact live replayed previous endpoint")
    before = _record(graph)
    before_tetrad = _tetrad(graph)
    if before["state"]["time"] != 1.5:
        raise ValueError("native step must begin at t=1.5")
    original = _reference(prior_report["prefix"]["coupling"]["refreshed_forcing"])
    before_capture = _capture(graph)
    prediction = predict_forced_support_realization_euler(
        original, prior_report["lineage"]["parent_children"], (DT,), epi=_state_epi(before),
    )
    frozen = {"payload": asdict(prediction), "sha256": _digest(prediction),
              "frozen_before_native_step": True, "time": before["state"]["time"]}
    trace = _trace_step(graph)
    endpoint = _record(graph)
    endpoint_tetrad = _tetrad(graph)
    if _digest(prediction) != frozen["sha256"]:
        raise RuntimeError("held counterfactual changed during native execution")
    if trace["status"] == "executed" and endpoint["state"]["time"] != 1.75:
        raise RuntimeError("native step completed at an undeclared physical time")
    endpoint_capture = _capture(graph)
    preserved = {field: before["state"][field] == endpoint["state"][field]
                 for field in ("nodes", "capacity", "phase", "edges")}
    if before_capture["available"] and endpoint_capture["available"]:
        a, b = before_capture["payload"]["observation"], endpoint_capture["payload"]["observation"]
        preserved.update(forcing=a["forcing"] == b["forcing"], epi_weight=a["epi_weight"] == b["epi_weight"])
    return {
        "branch": prior_report["branch"], "status": trace["status"], "before": before,
        "held_prediction": frozen, "native_trace": trace, "endpoint": endpoint,
        "before_tetrad": before_tetrad, "endpoint_tetrad": endpoint_tetrad,
        "before_capture": before_capture, "endpoint_capture": endpoint_capture,
        "endpoint_coefficient_comparison": preserved,
        "old_target_before": _pattern(original, before), "old_target_endpoint": _pattern(original, endpoint),
        "runtime_ledger": _runtime_ledger(prediction, before, endpoint, trace),
        "scope": "Coefficient endpoint agreement alone does not prove constancy through the native step",
    }


def run_study(prior_path=PRIOR_PATH, *, expected_prior_sha256=PRIOR_SHA256):
    historical, binding = load_prior_evidence(prior_path, expected_sha256=expected_prior_sha256)
    branches, replays, graphs, replay_checks = [], [], [], []
    for name, retained in zip(BRANCHES, historical["branches"], strict=True):
        graph, prior = replay_response_branch(name)
        replay_checks.append(_admit_replay(prior, retained))
        graphs.append(graph)
        replays.append(prior)
    if any(replays[0][field] != replays[1][field] for field in COMMON_FIELDS):
        raise RuntimeError("fresh branches differ in their inherited common source")
    for graph, prior in zip(graphs, replays, strict=True):
        branches.append(run_native_branch(graph, prior))
    paired = None
    paired_available = all(branch["status"] == "executed" and branch["endpoint"]["state"]["time"] == 1.75
                           and branch["old_target_endpoint"]["available"] for branch in branches)
    if paired_available:
        metric = replays[0]["original_reference"]["metric_weights"]
        paired = {key: _paired_delta(_state_epi(branches[0][key]), _state_epi(branches[1][key]), metric)
                  for key in ("before", "endpoint")}
    if hashlib.sha256(Path(prior_path).read_bytes()).hexdigest() != expected_prior_sha256:
        raise RuntimeError("prior artifact changed during execution")
    return {"protocol": PROTOCOL, "prior_evidence": binding, "prior_replay_checks": replay_checks,
            "scientific_prior_replays_equal": True,
            "complete_common_source_equal": True, "replayed_prior_reports": replays,
            "branches": branches, "paired_original_metric_response": paired,
            "paired_abstention_reason": None if paired_available else "Both complete same-time original-node endpoints required",
            "autonomous_maintenance_certified": False,
            "scope": "One finite native invocation per independently replayed branch; no repeated-runtime theorem"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-input", type=Path, default=PRIOR_PATH)
    parser.add_argument("--expected-prior-sha256", default=PRIOR_SHA256)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/research/thol_native_runtime_response_2026_09_18.json")
    args = parser.parse_args()
    if args.output.resolve() == args.prior_input.resolve():
        raise ValueError("output must not overwrite prior evidence")
    scope = ("src/tnfr", "benchmarks/thol_native_runtime_response.py", "benchmarks/thol_full_state_response.py",
             "benchmarks/thol_distributed_target.py", "benchmarks/thol_distributed_transport.py",
             "benchmarks/thol_birth_transport.py", "benchmarks/thol_pressure_feedback.py",
             "benchmarks/thol_eligibility_dispatch.py", "benchmarks/capacity_localization.py",
             "benchmarks/structural_target_compatibility.py", "benchmarks/thol_preparation_policy.py",
             "benchmarks/selection_birth_closure.py")
    provenance = current_git_source_provenance(ROOT, scope)
    sha, dirty, digest = provenance
    manifest = CoreExperimentManifest(
        claim_id="O1.b-generated-native-runtime-response", git_sha=sha, source_dirty=dirty, dirty_source_hash=digest,
        versions={"python": platform.python_version(), "networkx": nx.__version__, "numpy": np.__version__},
        graph_construction="Two complete independently replayed causal THOL/UM/optional AL branches",
        capacity_specification="Unchanged native configuration and live retained counters",
        solver="Native dynamics.step; separately frozen exact held-model Euler counterfactual",
        timestep=float(DT), seed=17, result_status=ClaimStatus.MEASURED,
        operator_sequence=("coherence", "dissonance", "self_organization", "coupling", "emission"),
        telemetry=("native boundary trace", "Si before glyphs", "coefficient changes", "exact displacement ledger"),
        controls=("complete historical replay", "fixed original profile and held-model counterfactual"),
        artifacts=(str(args.output),),
    )
    manifest.validate_for_admission()
    result = run_study(args.prior_input, expected_prior_sha256=args.expected_prior_sha256)
    if current_git_source_provenance(ROOT, scope) != provenance:
        raise RuntimeError("source changed during the native step study")
    report = {"manifest": manifest.to_dict(), "source_scope": scope, **result,
              "experimental_status": "No empirical correspondence tested"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    safe_write(args.output, lambda stream: stream.write(json.dumps(_payload(report), indent=2, allow_nan=False)+"\n"))
    print(f"Wrote native runtime response to {args.output}")


if __name__ == "__main__":
    main()
