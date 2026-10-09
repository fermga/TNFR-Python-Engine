"""Compare child-target UM with a held control from the retained t=6.5 state.

The original theoretical profile remains the same comparison target across
the actual event. A newly derived reference describes the changed regime;
changing that reference is not evidence that EPI recovered or deteriorated.
"""

from __future__ import annotations

import argparse
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

from benchmarks.forced_support_balance import (  # noqa: E402
    _same_frozen_inputs,
    prepare_forced_support_endpoint,
)
from benchmarks.thol_pressure_feedback import _payload, _state  # noqa: E402
from tnfr.dynamics.dnfr import default_compute_delta_nfr  # noqa: E402
from tnfr.dynamics.sampling import update_node_sample  # noqa: E402
from tnfr.operators import (  # noqa: E402
    build_operator_event_schedule,
    execute_operator_event_schedule,
)
from tnfr.operators._coupling_stage_kernel import (  # noqa: E402
    propose_coupling_stage,
)
from tnfr.operators.definitions import Coupling, Silence  # noqa: E402
from tnfr.operators.factor_contracts import (  # noqa: E402
    resolve_runtime_operator_factors,
)
from tnfr.operators.grammar_dynamics import validate_candidate  # noqa: E402
from tnfr.operators.grammar_execution import ValidatedSequence  # noqa: E402
from tnfr.physics.forced_support import (  # noqa: E402
    derive_forced_support_balance,
    observe_forced_support_pattern,
    observe_forced_support_reset,
    observe_forced_support_state,
    observe_forced_support_step,
)
from tnfr.physics.forcing_realization import capture_non_epi_forcing  # noqa: E402
from tnfr.research.claims import ClaimStatus  # noqa: E402
from tnfr.research.core_manifests import (  # noqa: E402
    CoreExperimentManifest,
    current_git_source_provenance,
)
from tnfr.rng import base_seed  # noqa: E402
from tnfr.types import Glyph  # noqa: E402
from tnfr.utils import angle_diff, ensure_node_offset_map  # noqa: E402
from tnfr.validation import validate_sequence  # noqa: E402

CASES = ("child_coupling", "no_extra_event")
STEP = 0.25
SEGMENT_COUNT = 12
CHILD_WORD = ("coupling", "silence")


def _apply_live(graph, node, operator):
    admission = validate_candidate(graph, node, operator.glyph.value)
    if not admission.allowed:
        raise RuntimeError(f"unexpected live refusal: {admission}")
    operator(graph, node, collect_metrics=True)
    return {"target": node, "candidate": admission.candidate, "allowed": True}


def _pattern(reference, capture):
    return observe_forced_support_pattern(
        reference,
        nodes=capture.snapshot.nodes,
        epi=capture.snapshot.epi,
    )


def _compatibility_channels(reference, capture):
    weights = dict(capture.normalized_weights)
    channels = {
        "phase": capture.phase_gradient,
        "vf": capture.snapshot.capacity_gradient,
        "topo": capture.snapshot.topology_gradient,
    }
    result = {
        key: weights[key]
        * sum(
            (d * value for d, value in zip(reference.strengths, values, strict=True)),
            Fraction(0),
        )
        for key, values in channels.items()
    }
    return {
        "weighted_forcing_by_channel": result,
        "exact_sum_residual": (
            sum(result.values(), Fraction(0)) - reference.compatibility_residual
        ),
    }


def _executor_record(evidence, before, raw):
    certificate = evidence.certificate
    if certificate is None:
        raise RuntimeError(
            "the declared default Euler interval lacks endpoint evidence"
        )

    def endpoint(value):
        return {
            "nodes": value.nodes,
            "epi": value.epi,
            "capacity": value.nu_f,
            "pressure": value.delta_nfr,
        }

    left, right = endpoint(certificate.left), endpoint(certificate.right)
    return {
        "integrator_name": evidence.integrator_name,
        "integrator_provenance_certified": evidence.integrator_provenance_certified,
        "resolved_method": evidence.resolved_method,
        "resolved_substeps": evidence.resolved_substeps,
        "gamma_is_none": evidence.gamma_is_none,
        "extended_dynamics_requested": evidence.extended_dynamics_requested,
        "clipping_applied": evidence.clipping_applied,
        "duration": certificate.duration,
        "captured_left": left,
        "captured_right": right,
        "left_binding": {key: value == before[key] for key, value in left.items()},
        "right_binding": {key: value == raw[key] for key, value in right.items()},
        "scope": (
            "Retained executor provenance and actual held-input endpoints; "
            "no solver accuracy, future execution or global gain is inferred"
        ),
    }


def _checkpoint_record(record):
    return {
        "helper": "benchmarks.forced_support_balance.prepare_forced_support_endpoint",
        "source_case": record["case"],
        "initial": record["initial"],
        "retained_endpoint": record["final"],
        "executed_prefix_segment_count": len(record["segments"]),
        "prefix_elapsed_time": record["physical_elapsed_time"],
        "prefix_reference": record["reference"],
        "prefix_final_relative_state": record["final_relative_state"],
        "prefix_mean_identity_residuals": tuple(
            item["exact_step_observation"]["mean_identity_residual"]
            for item in record["segments"]
        ),
        "scope": (
            "The existing live preparation and all 24 prior held-flow steps "
            "executed in this invocation; parent SHA has not executed"
        ),
    }


def _advance_forced_support_interval(
    graph,
    original_reference,
    reference,
    current,
    frozen_capture,
    *,
    duration,
):
    """Run one shared held Euler interval with exact budgets and endpoint bindings."""
    left = _state(graph)
    original_before = _pattern(original_reference, current)
    schedule = build_operator_event_schedule(
        (),
        start_time=left["time"],
        flow_durations=(duration,),
    )
    execution = execute_operator_event_schedule(
        graph,
        schedule,
        method="euler",
        include_flow_certificates=True,
    )
    evidence = execution.flow_interval_evidence[0]
    raw_endpoint = _state(graph)
    default_compute_delta_nfr(graph)
    following = capture_non_epi_forcing(graph)
    frozen = _same_frozen_inputs(frozen_capture, following)
    if not all(frozen.values()):
        raise RuntimeError("a postevent frozen input changed during flow")
    step = observe_forced_support_step(
        reference,
        current.snapshot,
        following.snapshot,
        dt=duration,
    )
    step_payload = asdict(step)
    step_payload.pop("reference")
    return following, {
        "before": left,
        "raw_after_integrator": raw_endpoint,
        "after_refresh": _state(graph),
        "duration": duration,
        "executor_evidence": _executor_record(evidence, left, raw_endpoint),
        "forcing_capture": asdict(following),
        "frozen_input_checks": frozen,
        "original_pattern_before": asdict(original_before),
        "original_pattern_after": asdict(_pattern(original_reference, following)),
        "regime_step_budget": step_payload,
    }


def prepare_child_feedback_endpoint(case="child_coupling"):
    """Return the actual t=9.5 endpoint before child or parent SHA.

    Callers retain the executed histories and own the pending word closures.
    The returned record is the original campaign without its terminal writes.
    """
    if case not in CASES:
        raise ValueError(f"case must be one of {CASES}")
    graph, checkpoint = prepare_forced_support_endpoint()
    child = checkpoint["source_preparation"]["preparation"]["children"][0]
    before = _state(graph)
    old_capture = capture_non_epi_forcing(graph)
    old_reference = derive_forced_support_balance(
        old_capture.snapshot,
        epi_weight=old_capture.epi_weight,
        forcing=old_capture.forcing,
    )
    child_index = before["nodes"].index(child)
    child_context = {"initial_epi_nonzero": before["epi"][child_index] > 0}
    ValidatedSequence((Coupling(), Silence()), context=child_context)
    child_word = validate_sequence(list(CHILD_WORD), context=child_context)
    if not child_word.passed:
        raise RuntimeError("the initialized child UM/SHA word was not admitted")
    sample_before = tuple(graph.graph.get("_node_sample", ()))
    update_node_sample(graph, step=2)
    proposal = None
    admission = None
    if case == "child_coupling":
        factors = resolve_runtime_operator_factors(
            graph.graph.get("GLYPH_FACTORS"),
            Glyph.UM,
            graph.graph,
        )
        proposal = propose_coupling_stage(
            graph,
            (child,),
            factors,
            resolved_seed=base_seed(graph),
            node_offsets=dict(ensure_node_offset_map(graph)),
        )
        admission = _apply_live(graph, child, Coupling())
    raw = _state(graph)
    default_compute_delta_nfr(graph)
    after_event = _state(graph)
    current = capture_non_epi_forcing(graph)
    reference = derive_forced_support_balance(
        current.snapshot,
        epi_weight=current.epi_weight,
        forcing=current.forcing,
    )
    reset = observe_forced_support_reset(
        old_reference,
        reference,
        old_capture.snapshot,
        current.snapshot,
    )
    reset_payload = asdict(reset)
    reset_payload.pop("before_reference")
    reset_payload.pop("after_reference")
    old_edges = {frozenset((u, v)) for u, v, _ in before["edges"]}
    actual_edges = tuple(
        (u, v, dict(data))
        for u, v, data in graph.edges(data=True)
        if frozenset((u, v)) not in old_edges
    )
    phases = dict(zip(raw["nodes"], raw["phase"], strict=True))
    event = {
        "before": before,
        "raw_after_event": raw,
        "after_refresh": after_event,
        "target": child,
        "actual_admission": admission,
        "child_word": CHILD_WORD,
        "initialized_child_context": child_context,
        "child_word_both_validators_passed": True,
        "child_word_executed_prefix": ("UM",) if admission else (),
        "sample_before": sample_before,
        "actual_candidate_sample": tuple(graph.graph["_node_sample"]),
        "readonly_kernel_proposal": asdict(proposal) if proposal else None,
        "actual_new_edges": actual_edges,
        "new_edge_phase_separations": tuple(
            abs(angle_diff(phases[u], phases[v])) for u, v, _ in actual_edges
        ),
        "before_forcing_capture": asdict(old_capture),
        "after_forcing_capture": asdict(current),
        "before_compatibility_channels": _compatibility_channels(
            old_reference,
            old_capture,
        ),
        "after_compatibility_channels": _compatibility_channels(reference, current),
        "original_pattern_before": asdict(_pattern(old_reference, old_capture)),
        "original_pattern_after": asdict(_pattern(old_reference, current)),
        "same_epi_reference_reset": reset_payload,
        "scope": (
            "The fixed original theoretical profile/metric is selected before "
            "UM. The changed regime gets a separate reference; the event's "
            "reference and metric jump is not an EPI recovery measurement"
        ),
    }
    frozen_capture = current
    segments = []
    for _ in range(SEGMENT_COUNT):
        current, segment = _advance_forced_support_interval(
            graph,
            old_reference,
            reference,
            current,
            frozen_capture,
            duration=STEP,
        )
        segments.append(segment)
    final = _state(graph)
    final_pattern = _pattern(old_reference, current)
    final_regime = observe_forced_support_state(reference, current.snapshot)
    return graph, {
        "case": case,
        "checkpoint": _checkpoint_record(checkpoint),
        "original_reference": asdict(old_reference),
        "postevent_reference": asdict(reference),
        "event": event,
        "segments": segments,
        "final_before_closure": final,
        "final_original_pattern": asdict(final_pattern),
        "final_regime_state": asdict(final_regime),
        "postevent_elapsed_time": final["time"] - after_event["time"],
        "scope": (
            "Finite actual child UM versus independent no-extra-event control "
            "from the retained t=6.5 checkpoint. Each node's declared word "
            "and live admission are separate; parent/child UM are not one "
            "single-target repeated-UM word. Runtime provenance identifies "
            "the captured Euler intervals, not complete-runtime stability "
            "or physical recovery. No later perturbation/recovery is claimed"
        ),
    }


def run_child_feedback_case(case):
    """Execute the original campaign and its unchanged terminal closures."""
    graph, record = prepare_child_feedback_endpoint(case)
    closures = []
    if case == "child_coupling":
        closures.append(
            {
                "admission": _apply_live(graph, record["event"]["target"], Silence()),
                "after": _state(graph),
            }
        )
    closures.append(
        {
            "admission": _apply_live(graph, 0, Silence()),
            "after": _state(graph),
        }
    )
    record["closures_after_measurement"] = closures
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/research/child_coupling_feedback.json",
    )
    args = parser.parse_args()
    scope = (
        "src/tnfr",
        "benchmarks/capacity_localization.py",
        "benchmarks/thol_pressure_feedback.py",
        "benchmarks/thol_birth_transport.py",
        "benchmarks/forced_support_balance.py",
        "benchmarks/child_coupling_feedback.py",
    )
    provenance = current_git_source_provenance(ROOT, scope)
    sha, dirty, digest = provenance
    manifest = CoreExperimentManifest(
        claim_id="O3.a-retained-child-coupling-reference-reset",
        git_sha=sha,
        source_dirty=dirty,
        dirty_source_hash=digest,
        versions={
            "python": platform.python_version(),
            "networkx": nx.__version__,
            "numpy": np.__version__,
        },
        graph_construction="Actual retained C8+child after 24 held Euler intervals",
        capacity_specification=(
            "Actual child UM capacity change; held during measured flow"
        ),
        solver="Existing default nodal Euler intervals; explicit canonical refresh",
        timestep=STEP,
        seed=17,
        result_status=ClaimStatus.MEASURED,
        operator_sequence=("parent IL OZ THOL UM", "child UM", "separate final SHA"),
        telemetry=(
            "actual functional links and auxiliary writes",
            "fixed original pattern",
            "reference/metric reset budgets",
            "pressure/Euler defects and mean drift",
        ),
        controls=("independent no-extra-event continuation",),
        artifacts=(str(args.output),),
    )
    manifest.validate_for_admission()
    report = {
        "manifest": manifest.to_dict(),
        "source_scope": scope,
        "cases": [run_child_feedback_case(case) for case in CASES],
        "experimental_status": "No empirical correspondence tested",
    }
    if current_git_source_provenance(ROOT, scope) != provenance:
        raise RuntimeError("source changed while executing the research artifact")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_payload(report), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote finite retained-child feedback observations to {args.output}")


if __name__ == "__main__":
    main()
