"""Forecast one real Emission perturbation of a generated sixteen-EPI pattern.

Two independent causal preparations share one predeclared source and model.
Only the child-emission branch executes AL, once on every actual child. Exact
Euler predictions are frozen before the unchanged two-step native continuation.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from fractions import Fraction
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

from benchmarks.thol_distributed_target import (  # noqa: E402
    _common_source,
    _digest,
    _flow_steps,
    _held_capture_checks,
    _reference,
    _snapshot,
    _target,
)
from benchmarks.thol_distributed_transport import (  # noqa: E402
    STEPS,
    _forcing_readout,
    _physical_flow,
    _require_stage,
    prepare_distributed_transport_support,
)
from benchmarks.thol_pressure_feedback import _payload, _state  # noqa: E402
from benchmarks.thol_preparation_policy import _literal  # noqa: E402
from tnfr.config.thresholds import (  # noqa: E402
    EPI_LATENT_MAX,
    MIN_NETWORK_DEGREE_COUPLING,
    VF_BASAL_THRESHOLD,
)
from tnfr.errors import TNFRValueError  # noqa: E402
from tnfr.operators.definitions import Emission  # noqa: E402
from tnfr.operators.factor_contracts import (
    resolve_runtime_operator_factors,
)  # noqa: E402
from tnfr.operators.grammar_dynamics import validate_candidate  # noqa: E402
from tnfr.operators.preconditions.emission import validate_emission_strict  # noqa: E402
from tnfr.operators.word_execution import execute_network_operator_stage  # noqa: E402
from tnfr.physics._cycle_algebra import dot  # noqa: E402
from tnfr.physics.epi_memory import observe_forced_support_realization  # noqa: E402
from tnfr.physics.forced_support import observe_forced_support_event  # noqa: E402
from tnfr.research.claims import ClaimStatus  # noqa: E402
from tnfr.research.core_manifests import (  # noqa: E402
    CoreExperimentManifest,
    current_git_source_provenance,
)
from tnfr.types import Glyph  # noqa: E402
from tnfr.utils.io import safe_write  # noqa: E402

BRANCHES = ("control", "child_emission")
COMMON_FIELDS = (
    "prefix",
    "original_reference",
    "original_reference_sha256",
    "realization",
    "baseline_flow",
    "baseline_steps",
    "baseline_capture_checks",
    "baseline_target",
    "common_source",
    "before_event",
    "lineage",
)
PROTOCOL = {
    "source": "Actual all-node IL/OZ preparation, distributed THOL birth and first parent UM",
    "event_time": 1.0,
    "branches": BRANCHES,
    "intervention": "One public simultaneous default Emission on every actual born child",
    "steps": STEPS,
    "fallback_operator": None,
    "cohort_search": False,
    "horizon_search": False,
    "coefficient_tuning": False,
    "target_rewrite": False,
    "scope": "Full state means all sixteen scalar EPI coordinates of one held model, not engine state",
}


def _mv(matrix, vector):
    return tuple(dot(row, vector) for row in matrix)


def _add(left, right):
    return tuple(a + b for a, b in zip(left, right, strict=True))


def _subtract(left, right):
    return tuple(a - b for a, b in zip(left, right, strict=True))


def _source(graph):
    """Extend the shared source projection with AL controls, without mutation."""
    return {
        **_common_source(graph),
        "emission_configuration": _literal(
            {
                "EPI_LATENT_MAX": graph.graph.get("EPI_LATENT_MAX", EPI_LATENT_MAX),
                "VF_BASAL_THRESHOLD": graph.graph.get(
                    "VF_BASAL_THRESHOLD", VF_BASAL_THRESHOLD
                ),
                "MIN_NETWORK_DEGREE_COUPLING": graph.graph.get(
                    "MIN_NETWORK_DEGREE_COUPLING", MIN_NETWORK_DEGREE_COUPLING
                ),
                "configured_preconditions_enabled": graph.graph.get(
                    "VALIDATE_OPERATOR_PRECONDITIONS", False
                ),
                "bounds": {
                    key: value
                    for key, value in graph.graph.items()
                    if key.startswith("EPI_") or key.startswith("CLIP_")
                },
                "resolved_factors": resolve_runtime_operator_factors(
                    graph.graph.get("GLYPH_FACTORS"), Glyph.AL, graph.graph
                ),
            }
        ),
    }


def _birth_families(graph, prefix):
    pairs = tuple(tuple(row) for row in prefix["birth"]["parent_children"])
    parents, children = tuple(p for p, _ in pairs), tuple(c for _, c in pairs)
    if (
        len(pairs) != 8
        or len(set(parents + children)) != 16
        or tuple(graph) != parents + children
        or parents != tuple(prefix["birth"]["before"]["nodes"])
    ):
        raise ValueError(
            "the declared experiment requires the actual eight birth families"
        )
    for parent, child in pairs:
        if (
            graph.nodes[child].get("parent_node") != parent
            or tuple(graph.nodes[parent].get("sub_nodes", ())) != (child,)
            or tuple(graph.graph["hierarchy"].get(parent, ())) != (child,)
        ):
            raise ValueError(
                "live parentage differs from the native THOL birth receipt"
            )
    return {
        "parent_children": pairs,
        "parents": parents,
        "children": children,
        "nodes": tuple(graph),
        "selection": "All actual children, once; no label inference",
    }


def _emission_admission(graph, children):
    """Add a strict read-only AL check without changing configured gate switches."""
    before = deepcopy(_source(graph))
    rows = []
    for node in children:
        grammar = validate_candidate(graph, node, "AL")
        reason = None
        try:
            validate_emission_strict(graph, node)
        except TNFRValueError as exc:
            reason = str(exc)
        rows.append(
            {
                "node": node,
                "grammar": asdict(grammar),
                "strict_emission_allowed": reason is None,
                "strict_emission_refusal": reason,
                "allowed": grammar.allowed and reason is None,
            }
        )
    if _source(graph) != before:
        raise RuntimeError("read-only Emission admission changed its source")
    return {
        "targets": children,
        "rows": tuple(rows),
        "allowed": all(row["allowed"] for row in rows),
        "configured_gate_switch_unchanged": True,
        "scope": "Additional strict read-only check; native stage still owns actual admission",
    }


def _emission_event(graph, children):
    before = deepcopy(_state(graph))
    before_capture = _forcing_readout(graph)
    operator = Emission()
    stage = execute_network_operator_stage(
        graph,
        operator,
        children,
        include_epi_jump_certificate=True,
    )
    _require_stage(stage, operator, children)
    after = deepcopy(_state(graph))
    after_capture = _forcing_readout(graph)
    certificate = stage.pointwise_epi_jump_certificate
    if certificate is None or not certificate._proof_fields_are_intact():
        raise RuntimeError("actual Emission stage lacks intact native jump evidence")
    native_nodes, native_targets = tuple(certificate.nodes), tuple(
        certificate.target_nodes
    )
    if (
        len(native_nodes) != len(before["nodes"])
        or len(set(native_nodes)) != len(native_nodes)
        or set(native_nodes) != set(before["nodes"])
        or len(native_targets) != len(children)
        or len(set(native_targets)) != len(native_targets)
        or set(native_targets) != set(children)
    ):
        raise RuntimeError(
            "native Emission receipt has different node or target identities"
        )
    native_to_state = tuple(before["nodes"].index(node) for node in native_nodes)
    exact_before = tuple(Fraction(before["epi"][index]) for index in native_to_state)
    exact_after = tuple(Fraction(after["epi"][index]) for index in native_to_state)
    if (
        tuple(Fraction(float(value)) for value in certificate.state_before)
        != exact_before
        or tuple(
            Fraction(float(value)) for value in certificate.runtime_proposed_state_after
        )
        != exact_after
    ):
        raise RuntimeError(
            "native Emission receipt does not match the actual committed EPI"
        )
    conditions = dict(certificate.runtime_epi_realization_conditions)
    required = (
        "two_phase_jacobi_declared",
        "proposal_builder_replayed_from_declared_inputs",
        "scalar_epi_snapshot",
        "supported_hard_or_inactive_clip_policy",
        "edge_adaptation_inactive",
        "operator_gate_bound",
    )
    if (
        not all(conditions.get(key) is True for key in required)
        or not certificate.fixed_support_declared
        or not certificate.proposal_builder_replayed_from_declared_inputs
    ):
        raise RuntimeError(
            "native Emission partial event lacks replayed admission evidence"
        )
    matrix, offset = (
        certificate.exact_represented_linear_map,
        certificate.exact_represented_offset,
    )
    size = len(native_nodes)
    if (
        matrix
        != tuple(tuple(Fraction(i == j) for j in range(size)) for i in range(size))
        or len(offset) != size
        or any(
            offset[index]
            for index, node in enumerate(native_nodes)
            if node not in children
        )
    ):
        raise RuntimeError("native Emission affine map changes undeclared coordinates")
    affine_after = _add(_mv(matrix, exact_before), offset)
    residual = _subtract(exact_after, affine_after)
    if (
        residual != certificate.exact_runtime_minus_represented_affine
        or certificate.runtime_matches_represented_affine_exactly != (not any(residual))
        or conditions.get("runtime_matches_represented_affine_exactly")
        != (not any(residual))
        or conditions.get("all_graph_nodes_targeted_once")
        != (set(native_targets) == set(native_nodes))
        or certificate.runtime_epi_realization_certified != all(conditions.values())
    ):
        raise RuntimeError(
            "native Emission affine residual or global-scope flags disagree"
        )
    for field in ("nodes", "capacity", "phase", "pressure", "edges", "time"):
        if before[field] != after[field]:
            raise RuntimeError(f"Emission changed protected held-model field {field}")
    _held_capture_checks(before_capture, after_capture)
    jump = _subtract(_snapshot(after_capture).epi, _snapshot(before_capture).epi)
    for index, node in enumerate(before["nodes"]):
        if node not in children and jump[index]:
            raise RuntimeError("Emission changed an unselected EPI coordinate")
    native_fields = (
        "operator_name",
        "glyph",
        "nodes",
        "target_nodes",
        "stage_schedule",
        "fixed_support_declared",
        "exact_represented_linear_map",
        "exact_represented_offset",
        "exact_runtime_minus_represented_affine",
        "runtime_matches_represented_affine_exactly",
        "clip_mode",
        "clip_intervention_nodes",
        "edge_adaptation_nodes",
        "proposal_builder_replayed_from_declared_inputs",
        "runtime_epi_realization_conditions",
        "runtime_epi_realization_certified",
        "scope",
    )
    return {
        "before": before,
        "after": after,
        "before_capture": before_capture,
        "after_capture": after_capture,
        "actual_epi_jump": jump,
        "any_epi_jump": any(jump),
        "stage_result": {
            "operator": stage.operator,
            "glyph": stage.glyph,
            "schedule": stage.schedule,
            "nodes_processed": stage.nodes_processed,
            "epi_jump_certificate_abstention_reason": stage.epi_jump_certificate_abstention_reason,
        },
        "native_epi_jump": {key: getattr(certificate, key) for key in native_fields},
        "native_jump_value_seal_verified": True,
        "partial_event_evidence": {
            "actual_endpoints_bound_by_node": True,
            "native_to_state_indices": native_to_state,
            "native_targets_to_declared_indices": tuple(
                children.index(node) for node in native_targets
            ),
            "native_order_before_epi": exact_before,
            "native_order_after_epi": exact_after,
            "represented_affine_after": affine_after,
            "exact_runtime_affine_residual": residual,
            "proposal_and_gate_replayed": True,
            "affine_residual_identity_verified": True,
            "global_certificate_failed_conditions": tuple(
                key for key, passed in conditions.items() if not passed
            ),
            "scope": "Actual partial AL event and signed runtime residual; no full-network affine-gain certificate",
        },
        "after_node_attributes": tuple(
            (node, _literal(dict(data))) for node, data in graph.nodes(data=True)
        ),
        "scope": (
            "Actual public two-phase partial AL and intact native EPI receipt; the stronger "
            "full-network affine certificate is not required or promoted. Lifecycle/semantic "
            "metadata remain changed; EPI return would not erase an Emission history."
        ),
    }


def _activity(capture):
    snapshot = _snapshot(capture)
    fresh = capture["observation"]["full_kernel_pressure"]
    fresh_rate = tuple(
        nu * pressure for nu, pressure in zip(snapshot.capacity, fresh, strict=True)
    )
    return {
        "positive_capacity_at_every_node": all(nu > 0 for nu in snapshot.capacity),
        "minimum_capacity": min(snapshot.capacity),
        "stored_nodal_rate": snapshot.rate,
        "fresh_kernel_nodal_rate": fresh_rate,
        "nonzero_fresh_rate_nodes": tuple(
            node for node, rate in zip(snapshot.nodes, fresh_rate) if rate
        ),
        "scope": "Instantaneous activity; raw post-AL stored pressure precedes the native refresh",
    }


def _forecast_error_accounting(prediction, steps):
    """Propagate already observed defects; this is an audit, not a second solver."""
    realization = prediction.realization
    c, t = realization.observation, realization.right_inverse
    n = realization.full_state_dimension
    pressure_error = step_error = (Fraction(0),) * n
    rows = []
    if len(steps) != len(prediction.steps):
        raise ValueError("actual segments do not cover the frozen forecast horizon")
    if tuple(steps[0]["before"]["snapshot"]["epi"]) != tuple(prediction.frames[0].epi):
        raise ValueError(
            "native continuation did not begin at the frozen forecast state"
        )
    for index, (record, h, matrix, frame) in enumerate(
        zip(
            steps,
            prediction.steps,
            prediction.fine_euler_matrices,
            prediction.frames[1:],
            strict=True,
        )
    ):
        if record["dt"] != h:
            raise ValueError("actual step duration differs from the frozen forecast")
        left, right = record["before"]["snapshot"], record["after"]["snapshot"]
        if index and steps[index - 1]["after"]["snapshot"]["epi"] != left["epi"]:
            raise ValueError("native Euler segments are not adjacent in EPI")
        local_pressure = tuple(
            h * nu * defect
            for nu, defect in zip(
                left["capacity"], record["before"]["pressure_defect"], strict=True
            )
        )
        local_step = tuple(record["support_budget"]["state_defect"])
        pressure_error = _add(_mv(matrix, pressure_error), local_pressure)
        step_error = _add(_mv(matrix, step_error), local_step)
        expected = _add(pressure_error, step_error)
        actual = _subtract(tuple(right["epi"]), tuple(frame.epi))
        actual_reduced = _mv(c, tuple(right["epi"]))
        reduced_error = _subtract(actual_reduced, frame.reduced_state)
        if (
            actual != expected
            or reduced_error != _mv(c, expected)
            or _mv(t, reduced_error) != actual
        ):
            raise RuntimeError("full-state forecast defect propagation identity failed")
        rows.append(
            {
                "ordinal": index + 1,
                "elapsed_time": frame.time,
                "local_pressure_defect_impulse": local_pressure,
                "local_integrator_state_defect": local_step,
                "propagated_pressure_error": pressure_error,
                "propagated_integrator_error": step_error,
                "actual_epi": tuple(right["epi"]),
                "ideal_epi": tuple(frame.epi),
                "actual_minus_ideal_epi": actual,
                "actual_reduced_state": actual_reduced,
                "ideal_reduced_state": frame.reduced_state,
                "actual_minus_ideal_reduced": reduced_error,
                "identity_residual": (Fraction(0),) * n,
                "maximum_absolute_epi_error": max(map(abs, actual)),
            }
        )
    return {
        "frames": tuple(rows),
        "all_exact_identities_pass": True,
        "zero_runtime_error_claimed": False,
        "scope": "C transports measured pressure and integrator defects; prediction remains frozen",
    }


def replay_response_branch(branch="control"):
    """Return the causally replayed live endpoint and the unchanged report."""
    from tnfr.physics.epi_memory import predict_forced_support_realization_euler

    if branch not in BRANCHES:
        raise ValueError(f"branch must be one of {BRANCHES}")
    graph, prefix = prepare_distributed_transport_support("attached")
    original_capture = prefix["coupling"]["refreshed_forcing"]
    original = _reference(original_capture)
    original_hash = _digest(original)
    lineage = _birth_families(graph, prefix)
    baseline_flow = _physical_flow(graph, _snapshot(original_capture))
    baseline_capture = baseline_flow["after_forcing"]
    held = _held_capture_checks(original_capture, baseline_capture)
    if _reference(baseline_capture, original) is not original:
        raise RuntimeError("baseline changed the original held coefficients")
    baseline_target = _target(original, original, baseline_capture)
    before = deepcopy(_state(graph))
    if before["time"] != 1.0:
        raise RuntimeError("the prescribed perturbation boundary is t=1")
    realization = observe_forced_support_realization(
        original,
        lineage["parent_children"],
        epi=_snapshot(baseline_capture).epi,
    )
    if realization.dimension != 16 or realization.full_state_dimension != 16:
        raise RuntimeError(
            "the predeclared full sixteen-EPI realization was not recovered"
        )
    common = {
        "prefix": prefix,
        "original_reference": asdict(original),
        "original_reference_sha256": original_hash,
        "lineage": lineage,
        "realization": asdict(realization),
        "baseline_flow": baseline_flow,
        "baseline_steps": _flow_steps(original, baseline_flow),
        "baseline_capture_checks": held,
        "baseline_target": asdict(baseline_target),
        "common_source": _source(graph),
        "before_event": before,
    }
    admission = event = None
    post_capture = baseline_capture
    if branch == "child_emission":
        admission = _emission_admission(graph, lineage["children"])
        if not admission["allowed"]:
            return graph, {
                **common,
                "branch": branch,
                "status": "refused",
                "admission": admission,
                "event": None,
                "frozen_prediction": None,
                "continuation_flow": None,
                "scope": "Read-only refusal; no substitute operator, target set or continuation",
            }
        event = _emission_event(graph, lineage["children"])
        post_capture = event["after_capture"]
    model_checks = _held_capture_checks(baseline_capture, post_capture)
    if _reference(post_capture, original) is not original:
        raise RuntimeError("intervention changed the predeclared held model")
    post_target = _target(original, original, post_capture)
    event_budget = observe_forced_support_event(
        original,
        original,
        _snapshot(baseline_capture),
        _snapshot(post_capture),
    )
    prediction = predict_forced_support_realization_euler(
        original,
        lineage["parent_children"],
        tuple(Fraction(h) for h in STEPS),
        epi=_snapshot(post_capture).epi,
    )
    for field in (
        "observation",
        "right_inverse",
        "reduced_generator",
        "output_map",
        "reduced_source",
    ):
        if getattr(prediction.realization, field) != getattr(realization, field):
            raise RuntimeError(
                "forecast replaced the fixed full-state coordinates or dynamics"
            )
    frozen_hash = _digest(prediction)
    frozen = {
        "payload": asdict(prediction),
        "sha256": frozen_hash,
        "frozen_before_continuation": True,
        "physical_time_at_freeze": graph.graph["_t"],
    }
    # The fresh support snapshot supplies the entry energy for this invocation.
    continuation = _physical_flow(graph, _snapshot(post_capture))
    if _digest(prediction) != frozen_hash or graph.graph["_t"] != 1.5:
        raise RuntimeError(
            "forecast or declared horizon changed during native continuation"
        )
    final_capture = continuation["after_forcing"]
    final_checks = _held_capture_checks(post_capture, final_capture)
    if (
        _reference(final_capture, original) is not original
        or _digest(original) != original_hash
    ):
        raise RuntimeError("continuation changed the fixed original reference")
    steps = _flow_steps(original, continuation)
    forecast_errors = _forecast_error_accounting(prediction, steps)
    endpoint = _target(original, original, final_capture)
    damage = post_target.pattern.error_variance - baseline_target.pattern.error_variance
    change = endpoint.pattern.error_variance - post_target.pattern.error_variance
    return graph, {
        **common,
        "branch": branch,
        "status": "executed",
        "admission": admission,
        "event": event,
        "post_event_capture": post_capture,
        "post_event_target": asdict(post_target),
        "event_target_budget": asdict(event_budget),
        "post_event_model_checks": model_checks,
        "frozen_prediction": frozen,
        "continuation_flow": continuation,
        "continuation_steps": steps,
        "continuation_capture_checks": final_checks,
        "forecast_errors": forecast_errors,
        "endpoint_target": asdict(endpoint),
        "endpoint": deepcopy(_state(graph)),
        "activity": {
            "before_event": _activity(baseline_capture),
            "after_event": _activity(post_capture),
            "endpoint": _activity(final_capture),
        },
        "original_target_response": {
            "event_variance_change": damage,
            "target_damage_present": damage > 0,
            "continuation_variance_change": change,
            "partial_target_recovery_observed": damage > 0 and change < 0,
            "returned_to_pre_event_target_score": damage > 0
            and (
                endpoint.pattern.error_variance
                <= baseline_target.pattern.error_variance
            ),
            "scope": "Target-score recovery requires an actual positive event damage; not autonomy",
        },
        "scope": (
            "One actual branch under held coefficients. Full state is sixteen scalar EPI values; "
            "it excludes erasure of AL history, phase/capacity policy state and autonomous maintenance."
        ),
    }


def run_response_branch(branch="control"):
    """Retain the historical mapping-only API."""
    return replay_response_branch(branch)[1]


def _paired_delta(control, perturbed, metric):
    delta = _subtract(perturbed, control)
    mean = dot(metric, delta) / sum(metric)
    centered = tuple(value - mean for value in delta)
    return {
        "epi_difference": delta,
        "weighted_mean_offset": mean,
        "centered_epi_difference": centered,
        "centered_H_energy": dot(metric, tuple(value * value for value in centered))
        / 2,
        "full_H_energy": dot(metric, tuple(value * value for value in delta)) / 2,
        "full_epi_equal": not any(delta),
        "shape_equal_modulo_uniform_offset": not any(centered),
    }


def run_study():
    branches = tuple(run_response_branch(name) for name in BRANCHES)
    control, perturbed = branches
    for field in COMMON_FIELDS:
        if control[field] != perturbed[field]:
            raise RuntimeError(f"independent branches differ in common field {field}")
    paired = None
    if perturbed["status"] == "executed":
        h = control["original_reference"]["metric_weights"]
        start = _paired_delta(
            control["post_event_target"]["state"]["snapshot"]["epi"],
            perturbed["post_event_target"]["state"]["snapshot"]["epi"],
            h,
        )
        end = _paired_delta(
            control["endpoint_target"]["state"]["snapshot"]["epi"],
            perturbed["endpoint_target"]["state"]["snapshot"]["epi"],
            h,
        )
        paired = {
            "after_event": start,
            "endpoint": end,
            "centered_response_energy_change": end["centered_H_energy"]
            - start["centered_H_energy"],
            "mean_offset_change": end["weighted_mean_offset"]
            - start["weighted_mean_offset"],
            "nonzero_intervention": not start["full_epi_equal"],
            "full_epi_return_observed": not start["full_epi_equal"]
            and end["full_epi_equal"],
            "centered_response_decreased": end["centered_H_energy"]
            < start["centered_H_energy"],
            "scope": (
                "Paired causal EPI displacement, using the inherited H metric. Uniform offset "
                "is retained; centered response decay does not imply full EPI or engine-state return."
            ),
        }
    return {
        "protocol": deepcopy(PROTOCOL),
        "branches": branches,
        "complete_common_source_equal": True,
        "common_fields_compared": COMMON_FIELDS,
        "paired_response": paired,
        "new_trajectories_executed": True,
        "coefficient_tuning": False,
        "target_rewrite": False,
        "autonomous_maintenance_certified": False,
        "scope": "Predeclared finite nodal intervention/response; no empirical or asymptotic claim",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=(ROOT / "artifacts/research/thol_full_state_response.json"),
    )
    args = parser.parse_args()
    scope = (
        "src/tnfr",
        "benchmarks/thol_full_state_response.py",
        "benchmarks/thol_distributed_target.py",
        "benchmarks/thol_distributed_transport.py",
        "benchmarks/thol_birth_transport.py",
        "benchmarks/thol_pressure_feedback.py",
        "benchmarks/thol_eligibility_dispatch.py",
        "benchmarks/capacity_localization.py",
        "benchmarks/structural_target_compatibility.py",
        "benchmarks/thol_preparation_policy.py",
        "benchmarks/selection_birth_closure.py",
    )
    provenance = current_git_source_provenance(ROOT, scope)
    sha, dirty, digest = provenance
    manifest = CoreExperimentManifest(
        claim_id="O1.b-generated-full-EPI-state-response",
        git_sha=sha,
        source_dirty=dirty,
        dirty_source_hash=digest,
        versions={
            "python": platform.python_version(),
            "networkx": nx.__version__,
            "numpy": np.__version__,
        },
        graph_construction="Two independent actual distributed THOL/UM preparations",
        capacity_specification="Native default factors; positive held capacities; no channel removal",
        solver="Native pressure-refreshed Euler plus separately frozen exact full-EPI forecast",
        timestep=STEPS[0],
        seed=17,
        result_status=ClaimStatus.MEASURED,
        operator_sequence=(
            "coherence",
            "dissonance",
            "self_organization",
            "coupling",
            "emission",
        ),
        telemetry=(
            "complete causal source",
            "native AL jump",
            "forecast-before-flow",
            "pressure/integrator defects",
            "paired shape and mean response",
        ),
        controls=(
            "independent no-event branch",
            "fixed original profile and full EPI realization",
        ),
        artifacts=(str(args.output),),
    )
    manifest.validate_for_admission()
    result = run_study()
    if current_git_source_provenance(ROOT, scope) != provenance:
        raise RuntimeError("source changed during the finite response study")
    report = {
        "manifest": manifest.to_dict(),
        "source_scope": scope,
        **result,
        "experimental_status": "No empirical correspondence tested",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(_payload(report), indent=2, allow_nan=False) + "\n"
    safe_write(args.output, lambda stream: stream.write(encoded))
    print(f"Wrote full-EPI response study to {args.output}")


if __name__ == "__main__":
    main()
