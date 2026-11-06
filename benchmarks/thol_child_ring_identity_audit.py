"""Read the retained child-ring and no-event cohorts through shared observers.

This is retrospective characterization, not a new trajectory or a held-out test.
Historical model targets, actual form, phase topology and regional exchange are
kept separate. The current tetrad is an offline read-out, not archived telemetry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from dataclasses import asdict
from fractions import Fraction as F
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from benchmarks.thol_family_closure import (  # noqa: E402
    CONTROL_PATH,
    CONTROL_SHA256,
    LINEAGE_PATH,
    LINEAGE_SHA256,
    _admit_target,
    _equal,
    _reference,
    load_evidence,
)
from benchmarks.thol_full_state_response import _paired_delta  # noqa: E402
from benchmarks.thol_pressure_feedback import _payload  # noqa: E402
from benchmarks.thol_regional_balance_audit import (  # noqa: E402
    _bind_source_edges,
    _region_report,
)
from benchmarks.thol_regional_identity_audit import _region_identity  # noqa: E402
from tnfr.physics._cycle_algebra import dot, ordered_vector  # noqa: E402
from tnfr.physics.fields import (  # noqa: E402
    compute_structural_potential,
    estimate_coherence_length_with_provenance,
    observe_phase_curvature,
)
from tnfr.physics.forced_support import observe_forced_support_step  # noqa: E402
from tnfr.physics.support_transport import (  # noqa: E402
    observe_regional_support_balance,
    observe_regional_support_euler,
)
from tnfr.physics.winding_certificates import certify_phase_winding  # noqa: E402
from tnfr.research.core_manifests import (  # noqa: E402
    CoreExperimentManifest,
    current_git_source_provenance,
)
from tnfr.research.recorded_support import (  # noqa: E402
    bind_recorded_nodal_state,
    graph_from_recorded_nodal_state,
    read_recorded_forcing,
    read_recorded_support_snapshot,
)


def observe_shape_retention(initial, current, metric):
    """Exact descriptive projection on a fixed observed shape, without a threshold.

    The projection factor measures amplitude along the initial centered form;
    it is neither a dynamical gain nor an accepted identity/maintenance score.
    """
    first, last, weights = (
        ordered_vector(v, name)
        for v, name in ((initial, "initial"), (current, "current"), (metric, "metric"))
    )
    if (
        not first
        or len(first) != len(last)
        or len(first) != len(weights)
        or any(w <= 0 for w in weights)
    ):
        raise ValueError(
            "matching nonempty vectors and a positive fixed metric required"
        )
    zero = (F(0),) * len(first)
    left = _paired_delta(zero, first, weights)["centered_epi_difference"]
    right = _paired_delta(zero, last, weights)["centered_epi_difference"]
    initial_norm = dot(weights, tuple(x * x for x in left))
    current_norm = dot(weights, tuple(x * x for x in right))
    factor = (
        dot(weights, tuple(x * y for x, y in zip(left, right, strict=True)))
        / initial_norm
        if initial_norm
        else None
    )
    remainder = (
        tuple(y - factor * x for x, y in zip(left, right, strict=True))
        if factor is not None
        else None
    )
    defect = (
        dot(weights, tuple(x * x for x in remainder)) if remainder is not None else None
    )
    return {
        "initial_squared_norm": initial_norm,
        "current_squared_norm": current_norm,
        "signed_amplitude_projection": factor,
        "orthogonal_residual_squared_norm": defect,
        "relative_squared_residual": (
            defect / current_norm if defect is not None and current_norm else None
        ),
        "exact_collinearity": defect == 0 if defect is not None else None,
        "scope": "Fixed observed centered form and metric; undefined normalization stays unavailable. No fitted evolution coefficient or acceptance threshold.",
    }


def observe_recorded_cohort(state, capture, region, *, phase_gate):
    """Bind one archived triad and compose existing full-graph observers.

    The ordered region also declares the winding cycle. Its cycle may be
    absent; undefined winding is retained instead of being converted to zero.
    Current field kernels run only on a detached graph carrying saved inputs.
    """
    snapshot, observation, components = read_recorded_forcing(capture)
    bind_recorded_nodal_state(state, capture, snapshot)
    _bind_source_edges(state, snapshot)
    regional = _region_report(
        snapshot, observation, components, region, "actual_children"
    )
    graph = graph_from_recorded_nodal_state(state)
    phase = observe_phase_curvature(graph)
    potential = compute_structural_potential(graph, alpha=2.0)
    coherence = asdict(estimate_coherence_length_with_provenance(graph))
    if not math.isfinite(coherence["value"]):
        coherence["value"] = None
    winding = certify_phase_winding(graph, tuple(region), phase_gate=phase_gate)
    indices = tuple(snapshot.nodes.index(node) for node in region)
    return {
        "time": state["time"],
        "region": tuple(region),
        "triad": {
            "epi": tuple(snapshot.epi[i] for i in indices),
            "capacity": tuple(snapshot.capacity[i] for i in indices),
            "phase": tuple(observation.phase[i] for i in indices),
        },
        "regional": regional,
        "winding": asdict(winding),
        "offline_tetrad": {
            "nodes": snapshot.nodes,
            "phi_s": tuple(potential[node] for node in snapshot.nodes),
            "grad_phi": tuple(row.gradient for row in phase.rows),
            "curv_phi": tuple(row.curvature for row in phase.rows),
            "phase_readout": asdict(phase),
            "xi_c": coherence,
            "scope": (
                "Current canonical numerical read-outs on complete saved state, not historical "
                "telemetry or dynamical feedback. Potential uses alpha=2 and explicit length, "
                "else saved weight, else unit distance. Coherence fit uses its documented "
                "static pressure read-out; fit/fallback provenance is retained. Neighbor "
                "order is reconstructed from saved edges, not certified original runtime insertion order."
            ),
        },
    }


def _finite_budgets(branch, reference, region):
    """Reuse both held-model and regional identities on each retained segment."""
    flow = branch["continuation_flow"]
    segments, retained_steps = flow["segments"], branch["continuation_steps"]
    if len(segments) != len(retained_steps) or not segments:
        raise ValueError("complete nonempty segment/step correspondence required")
    rows = []
    previous = None
    previous_time = F(flow["before"]["time"])
    for index, (segment, retained) in enumerate(
        zip(segments, retained_steps, strict=True)
    ):
        if segment["index"] != index or segment["method"] != "euler":
            raise ValueError("ordered retained Euler segments required")
        before = read_recorded_support_snapshot(segment["before_support"])
        after = read_recorded_support_snapshot(segment["after_support"])
        duration = F(segment["interval"]["exact_duration"])
        start, end = (
            F(segment["interval"][key])
            for key in ("exact_start_time", "exact_end_time")
        )
        if duration <= 0 or start != previous_time or end - start != duration:
            raise ValueError("contiguous positive exact intervals required")
        if duration != F(segment["interval"]["duration"]):
            raise ValueError("represented duration differs from exact interval")
        if previous is None:
            for field, key in (
                ("epi", "epi"),
                ("capacity", "capacity"),
                ("stored_pressure", "pressure"),
            ):
                if getattr(before, field) != tuple(
                    F(value) for value in flow["before"][key]
                ):
                    raise ValueError(
                        "first segment differs from recorded continuation source"
                    )
            _equal(
                before.nodes, flow["before"]["nodes"], "continuation source node order"
            )
            _bind_source_edges(flow["before"], before)
        if previous is not None:
            # Pressure is refreshed between segments; form and the held domain persist.
            for field in (
                "nodes",
                "conductance",
                "support_neighbors",
                "capacity",
                "epi",
            ):
                if getattr(previous, field) != getattr(before, field):
                    raise ValueError("retained segment continuity differs")
        held = observe_forced_support_step(reference, before, after, duration)
        _equal(asdict(held), retained, "complete held-model step")
        budget = observe_regional_support_euler(
            before,
            after,
            region,
            dt=duration,
            epi_weight=reference.epi_weight,
            forcing=reference.forcing,
        )
        rows.append(
            {"interval": segment["interval"], "regional_budget": asdict(budget)}
        )
        previous = after
        previous_time = end
    if previous_time != F(flow["after"]["time"]):
        raise ValueError("last segment differs from recorded endpoint time")
    for field, key in (("epi", "epi"), ("capacity", "capacity")):
        if getattr(previous, field) != tuple(F(value) for value in flow["after"][key]):
            raise ValueError("last segment differs from recorded endpoint state")
    _equal(previous.nodes, flow["after"]["nodes"], "continuation endpoint node order")
    _bind_source_edges(flow["after"], previous)
    first, last = rows[0]["regional_budget"], rows[-1]["regional_budget"]
    change = last["after_variance"] - first["balance"]["variance"]
    residual = change - sum(
        (row["regional_budget"]["variance_change"] for row in rows), F(0)
    )
    if residual:
        raise RuntimeError("regional finite variance telescope failed")
    return {
        "segments": rows,
        "variance_change": change,
        "variance_telescope_residual": residual,
        "whole_graph_dirichlet_change": (
            last["after"]["dirichlet_energy"] - first["before"]["dirichlet_energy"]
        ),
        "scope": (
            "Exact represented-input finite Euler identities; boundary terms are held-step "
            "accounting, not exact continuous-time integrated fluxes. Producer receipts remain "
            "historical evidence; this reader proves neither solver accuracy nor future stability."
        ),
    }


def audit_branch(branch, region, *, phase_gate):
    """Compare actual form with its preparation separately from the model target."""
    original = _reference(branch["original_reference"])
    current = _reference(branch["post_event_target"]["reference"])
    initial_capture = branch["prefix"]["coupling"]["refreshed_forcing"]
    before_capture = branch["baseline_flow"]["after_forcing"]
    after_capture = (
        before_capture
        if branch["event"] is None
        else branch["event"]["coupling"]["refreshed_forcing"]
    )
    rows = (
        (
            "initial_attached",
            branch["baseline_flow"]["before"],
            initial_capture,
            branch["initial_target"],
            original,
        ),
        (
            "before_event",
            branch["before_optional_event"],
            before_capture,
            branch["baseline_target"],
            original,
        ),
        (
            "after_event",
            branch["after_optional_event"],
            after_capture,
            branch["post_event_target"],
            current,
        ),
        (
            "endpoint",
            branch["endpoint"],
            branch["continuation_flow"]["after_forcing"],
            branch["endpoint_target"],
            current,
        ),
    )
    _equal(branch["baseline_flow"]["after"], rows[1][1], "baseline endpoint")
    _equal(branch["continuation_flow"]["before"], rows[2][1], "continuation start")
    _equal(branch["continuation_flow"]["after"], rows[3][1], "continuation endpoint")
    indices = tuple(original.source.nodes.index(node) for node in region)
    fixed_metric = tuple(original.metric_weights[i] for i in indices)
    initial_epi = tuple(F(rows[0][1]["epi"][i]) for i in indices)
    post_event_epi = tuple(F(rows[2][1]["epi"][i]) for i in indices)
    observations = []
    for label, state, capture, target_record, reference in rows:
        snapshot, target = _admit_target(target_record, original, reference, capture)
        observed = observe_recorded_cohort(
            state, capture, region, phase_gate=phase_gate
        )
        form = tuple(snapshot.epi[i] for i in indices)
        observations.append(
            {
                "label": label,
                **observed,
                "form_change_from_initial_attachment": _paired_delta(
                    initial_epi, form, fixed_metric
                ),
                "form_contrast_in_fixed_metric": _paired_delta(
                    (F(0),) * len(form), form, fixed_metric
                ),
                "shape_relative_to_post_event": observe_shape_retention(
                    post_event_epi, form, fixed_metric
                ),
                "full_graph_model_target": {
                    "error_variance": target.pattern.error_variance,
                    "compatible": target.target_compatible,
                    "compatibility_energy": target.compatibility_energy,
                    "scope": "Frozen held-model profile, not the actual born form or an identity score.",
                },
            }
        )
    before, obs_before, _ = read_recorded_forcing(after_capture)
    after, obs_after, _ = read_recorded_forcing(rows[-1][2])
    balance = observe_regional_support_balance(
        before,
        region,
        epi_weight=current.epi_weight,
        forcing=current.forcing,
    )
    return {
        "branch": branch["branch"],
        "fixed_region_metric": fixed_metric,
        "observations": observations,
        "flow_identity": _region_identity(
            before, after, balance, obs_before.phase, obs_after.phase
        ),
        "continuation": _finite_budgets(branch, current, region),
        "captured_model_endpoint_checks": branch["continuation_capture_checks"],
        "physical_perturbation_recovery_test_present": False,
    }


def run_audit(
    control_path=CONTROL_PATH,
    lineage_path=LINEAGE_PATH,
    *,
    expected_control_sha256=CONTROL_SHA256,
    expected_lineage_sha256=LINEAGE_SHA256,
):
    controls, lineage, partition, bindings = load_evidence(
        control_path,
        lineage_path,
        expected_control_sha256=expected_control_sha256,
        expected_lineage_sha256=expected_lineage_sha256,
    )
    ring = next(row for row in lineage if row["branch"] == "born_children")
    children = tuple(child for _, child in partition["parent_children"])
    gates = {
        row["effective_phase_limit"]
        for row in ring["event"]["coupling"]["kernel_proposal"]["target_proposals"]
    }
    if len(gates) != 1:
        raise ValueError("one captured phase gate required")
    gate = gates.pop()
    control = next(row for row in controls if row["branch"] == "no_event")
    outputs = tuple(
        audit_branch(row, children, phase_gate=gate) for row in (control, ring)
    )
    _equal(
        outputs[0]["fixed_region_metric"],
        outputs[1]["fixed_region_metric"],
        "common metric",
    )
    _equal(
        outputs[0]["observations"][1],
        outputs[1]["observations"][1],
        "full pre-event readout",
    )
    metric = outputs[0]["fixed_region_metric"]
    final = [row["observations"][-1] for row in outputs]
    return {
        "historical_inputs": bindings,
        "region": children,
        "phase_gate": gate,
        "branches": outputs,
        "endpoint_comparison": {
            "ring_minus_no_event_form": _paired_delta(
                final[0]["triad"]["epi"], final[1]["triad"]["epi"], metric
            ),
            "fixed_model_target_error_difference": (
                final[1]["full_graph_model_target"]["error_variance"]
                - final[0]["full_graph_model_target"]["error_variance"]
            ),
            "scope": (
                "Common pre-event history and fixed observation metric; the UM intervention "
                "changes support and capacity together. This is not a single-channel ablation, "
                "a deliberately damaged/restored pair, or a comparison of raw changing-metric variances."
            ),
        },
        "new_trajectories_executed": 0,
        "native_operators_executed": 0,
        "pressure_generation_or_phase_evolution_kernels_executed": 0,
        "current_tetrad_readouts_computed_offline": True,
        "scope": (
            "Prepared parent winding, inherited child phase and supplied event/cohort selection. "
            "Finite identity components and exact regional balances do not establish autonomous "
            "NFR formation, recovery, future maintenance or physical particles."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/research/thol_child_ring_identity_2026_09_19.json",
    )
    args = parser.parse_args()
    if args.output.resolve() in (CONTROL_PATH.resolve(), LINEAGE_PATH.resolve()):
        raise ValueError("audit output must not overwrite historical inputs")
    scope = [
        "src/tnfr",
        "benchmarks/thol_child_ring_identity_audit.py",
        "benchmarks/thol_family_closure.py",
        "benchmarks/thol_lineage_coordination.py",
        "benchmarks/thol_full_state_response.py",
        "benchmarks/thol_pressure_feedback.py",
        "benchmarks/thol_regional_balance_audit.py",
        "benchmarks/thol_regional_identity_audit.py",
    ]
    provenance = current_git_source_provenance(ROOT, scope)
    result = run_audit()
    if current_git_source_provenance(ROOT, scope) != provenance:
        raise RuntimeError("audit sources changed during the read-only analysis")
    sha, dirty, digest = provenance
    manifest = CoreExperimentManifest(
        claim_id="O3.a-retained-child-ring-full-triad-identity",
        git_sha=sha,
        source_dirty=dirty,
        dirty_source_hash=digest,
        versions={"python": platform.python_version()},
        result_status="measured",
        graph_construction="Detached complete saved snapshots from authenticated lineage and no-event records",
        capacity_specification="Captured capacities and forcing; original metric frozen for cross-branch form comparison",
        solver="No trajectory; shared exact endpoint budgets and current offline tetrad/winding observers",
        seed=None,
        timestep=None,
        operator_sequence=(),
        telemetry=(
            "full triad",
            "original form versus model target",
            "regional boundary balance",
            "offline tetrad and winding",
        ),
        controls=(
            "authenticated no-event common source",
            "fixed common form metric",
            "complete saved target/step comparisons",
        ),
        artifacts=(str(args.output.relative_to(ROOT)),),
    )
    manifest.validate_for_admission()
    report = {"manifest": manifest.to_dict(), "source_scope": scope, **result}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_payload(report), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "branches": [row["branch"] for row in result["branches"]],
                "new_trajectories": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
