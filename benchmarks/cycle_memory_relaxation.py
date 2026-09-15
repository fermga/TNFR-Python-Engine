"""Finite fixed-capacity cycle diffusion with explicit delayed REMESH.

UM/IL/Recursivity is a grammar-admitted word; Recursivity is only advisory.
The existing causal executor separately samples pre-map history and applies
delayed REMESH. A second case executes the same advisory word and physical
flow without that separate map. Neither case uses a zero-alpha runtime map.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, fields, is_dataclass, replace
from fractions import Fraction
import json
import math
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

from benchmarks.capacity_localization import build_cycle  # noqa: E402
from tnfr._remesh_contract import remesh_history_maxlen  # noqa: E402
from tnfr.alias import get_attr  # noqa: E402
from tnfr.constants import DEFAULTS  # noqa: E402
from tnfr.constants.aliases import (  # noqa: E402
    ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF,
)
from tnfr.operators import (  # noqa: E402
    build_operator_event_schedule, build_physical_flow_partition,
    execute_operator_event_schedule,
)
from tnfr.operators.event_remesh_causal_runtime import (  # noqa: E402
    EventRemeshCycleExecutionSpec, execute_event_remesh_cycle_sequence,
)
from tnfr.physics.cycle_memory_relaxation import (  # noqa: E402
    certify_cycle_memory_relaxation,
)
from tnfr.physics.cycle_support_dynamics import (  # noqa: E402
    observe_cycle_support_balance, observe_cycle_support_euler,
)
from tnfr.physics.remesh_history_stability import (  # noqa: E402
    observe_uniform_remesh_history_transition,
)
from tnfr.physics.winding_certificates import certify_phase_winding  # noqa: E402
from tnfr.research.claims import ClaimStatus  # noqa: E402
from tnfr.research.core_manifests import (  # noqa: E402
    CoreExperimentManifest, current_git_source_provenance,
)
from tnfr.utils.numeric import angle_diff  # noqa: E402

WORD = ("coupling", "coherence", "recursivity")
FLOW_DURATIONS = (0.0, 0.0, 0.0, 0.5)
STEPS = (0.25, 0.25)
CYCLE_COUNT = 4
INITIAL_EPI = (1.0,) + (0.5,) * 7


def _values(graph, aliases):
    return tuple(float(get_attr(graph.nodes[node], aliases, None)) for node in graph)


def _exact(values):
    return tuple(Fraction.from_float(float(value)) for value in values)


def _difference(left, right):
    return tuple(a - b for a, b in zip(left, right, strict=True))


def _phase_observation(phase):
    """Use the shared observer on a detached cycle of recorded phase values."""
    readout = nx.cycle_graph(len(phase))
    for node, value in enumerate(phase):
        readout.nodes[node]["theta"] = value
    return {
        "phase": tuple(phase),
        "winding": asdict(certify_phase_winding(readout, tuple(readout))),
        "maximum_wrapped_drift_from_regular_twist": max(
            abs(angle_diff(value, math.tau * node / len(phase)))
            for node, value in enumerate(phase)
        ),
    }


def _state(epi, capacity, phase, pressure, *, time):
    exact = _exact(epi)
    mean = sum(exact, Fraction(0)) / len(exact)
    return {
        "time": float(time), "epi": tuple(epi), "capacity": tuple(capacity),
        "pressure": tuple(pressure), "phase_readout": _phase_observation(phase),
        "exact_mean": mean,
        "exact_disagreement_energy": sum(
            ((value - mean) ** 2 for value in exact), Fraction(0),
        ) / 2,
        "epi_range": max(epi) - min(epi),
        "core_minus_background_mean": epi[0] - sum(epi[1:]) / (len(epi) - 1),
    }


def _live_state(graph):
    return _state(
        *(_values(graph, alias) for alias in (
            ALIAS_EPI, ALIAS_VF, ALIAS_THETA, ALIAS_DNFR,
        )),
        time=graph.graph["_t"],
    )


def _regular_reference(epi, capacity, weights):
    return observe_cycle_support_balance(
        epi, capacity, (0,) * len(epi), epi_weight=weights["epi"],
        vf_weight=weights["vf"], phase_weight=weights["phase"],
    )


def _prepare_graph():
    graph = build_cycle(8, epi=INITIAL_EPI)
    graph.graph.update(
        REMESH_TAU_LOCAL=1, REMESH_TAU_GLOBAL=1,
        REMESH_ALPHA=DEFAULTS["REMESH_ALPHA"], REMESH_ALPHA_HARD=True,
        REMESH_LOG_EVENTS=False,
    )
    graph.graph["_epi_hist"] = deque(
        [dict(enumerate(INITIAL_EPI))], maxlen=remesh_history_maxlen(1, 1),
    )
    return graph


def _spec(index):
    schedule = build_operator_event_schedule(
        WORD, start_time=index / 2, flow_durations=FLOW_DURATIONS,
    )
    partition = build_physical_flow_partition(schedule.intervals[-1], STEPS)
    return EventRemeshCycleExecutionSpec(schedule, (partition,))


def _flow_records(event, weights):
    evidence = event.physical_flow_partition_evidence[0]
    boundaries = []
    for boundary in evidence.boundary_observations:
        actual = boundary.after
        model = _regular_reference(actual.exact_epi, actual.exact_nu_f, weights)
        boundaries.append({
            "time": boundary.time, "epi": actual.epi,
            "capacity": actual.nu_f, "pressure": actual.delta_nfr,
            "exact_runtime_minus_regular_twist_pressure": _difference(
                actual.exact_delta_nfr, model.pressure,
            ),
            "pressure_only_refresh": boundary.nonpressure_state_preserved,
            "phase_preserved_during_refresh": boundary.phase_preserved,
            "capacity_preserved_during_refresh": boundary.capacity_preserved,
            "callback_name": boundary.callback_name,
        })
    segments = []
    for segment, left, right in zip(
        evidence.segment_flow_evidence, evidence.boundary_observations[:-1],
        evidence.boundary_observations[1:], strict=True,
    ):
        source, endpoint = left.after, right.before
        dt = segment.interval.exact_duration
        model = _regular_reference(source.exact_epi, source.exact_nu_f, weights)
        prediction = observe_cycle_support_euler(model, dt=dt)
        pressure_defect = _difference(source.exact_delta_nfr, model.pressure)
        held = tuple(
            x + dt * nu * pressure for x, nu, pressure in zip(
                source.exact_epi, source.exact_nu_f,
                source.exact_delta_nfr, strict=True,
            )
        )
        held_defect = _difference(endpoint.exact_epi, held)
        endpoint_defect = _difference(endpoint.exact_epi, prediction.after.epi)
        explained = tuple(
            dt * nu * pressure + error for nu, pressure, error in zip(
                source.exact_nu_f, pressure_defect, held_defect, strict=True,
            )
        )
        segments.append({
            "duration": segment.interval.duration,
            "exact_duration": dt, "reference": prediction,
            "exact_runtime_minus_regular_twist_pressure": pressure_defect,
            "exact_held_input_euler_residual": held_defect,
            "exact_model_endpoint_residual": endpoint_defect,
            "exact_endpoint_decomposition_residual": _difference(
                endpoint_defect, explained,
            ),
            "node_clock_increments": tuple(dt * nu for nu in source.exact_nu_f),
            "capacity_held": source.nu_f == endpoint.nu_f,
            "method": segment.resolved_method,
            "integrator": segment.integrator_name,
            "gamma_is_none": segment.gamma_is_none,
            "clipping_applied": segment.clipping_applied,
            "extended_dynamics_requested": segment.extended_dynamics_requested,
        })
    return {
        "boundaries": boundaries, "segments": segments,
        "events": [{
            "operator": stage.event.operator_name,
            "epi_before": stage.left.epi, "epi_after": stage.right.epi,
            "capacity_before": stage.left.nu_f,
            "capacity_after": stage.right.nu_f,
            "stored_pressure_before": stage.left.delta_nfr,
            "stored_pressure_after": stage.right.delta_nfr,
            "certificate_abstention_reason": stage.certificate_abstention_reason,
        } for stage in event.glyph_stage_evidence],
        "physical_pressure_refresh_calls": (
            event.physical_pressure_refresh_callback_invocations
        ),
        "phase_scope": (
            "UM and default IL use local circular means; the ideal regular "
            "twist is fixed. Captured endpoint drift and the full canonical "
            "pressure discrepancy are retained. Refresh flags concern only "
            "the callback, not an unobserved continuous phase path."
        ),
    }


def _memory_cycle_record(cycle, reference, weights):
    history = cycle.history_transition
    active = tuple(reversed(history.outgoing_exact_history[-2:]))
    ideal = observe_uniform_remesh_history_transition(
        reference.remesh_certificate, active, (1,) * 8, nodes=range(8),
    )
    raw = _exact(item.raw_epi for item in cycle.remesh.proposals)
    output = _exact(item.bounded_epi for item in cycle.remesh.proposals)
    rounding = _difference(raw, ideal.exact_next_field)
    clipping = _difference(output, raw)
    total = _difference(output, ideal.exact_next_field)
    start = cycle.event_execution.schedule.start_time
    end = cycle.event_execution.final_time
    return {
        "before": _state(
            cycle.pre_schedule_epi.epi_values, cycle.capacity_before_schedule,
            cycle.phase_before_schedule, cycle.pressure_before_schedule,
            time=start,
        ),
        "pre_remesh": _state(
            cycle.pre_remesh_epi.epi_values, cycle.capacity_before_remesh,
            cycle.phase_before_remesh, cycle.pressure_before_remesh, time=end,
        ),
        "after": _state(
            cycle.post_remesh_epi.epi_values, cycle.capacity_after_remesh,
            cycle.phase_after_optional_refresh,
            cycle.pressure_after_optional_refresh, time=end,
        ),
        "flow": _flow_records(cycle.event_execution, weights),
        "history": {
            "incoming": history.incoming_exact_history,
            "outgoing": history.outgoing_exact_history,
            "appended": history.appended_exact_pre_remesh_epi,
            "selected_local": history.selected_local_delayed_epi,
            "selected_global": history.selected_global_delayed_epi,
            "convention": cycle.history_convention,
            "canonical_transition_certified": (
                history.canonical_history_transition_certified
            ),
            "schedule_left_history_unchanged": cycle.schedule_left_history_unchanged,
            "active_newest_first": active,
        },
        "remesh": {
            "applied": cycle.remesh.applied,
            "alpha": cycle.remesh.plan.alpha,
            "alpha_source": cycle.remesh.plan.alpha_source,
            "tau_local": cycle.remesh.plan.tau_local,
            "tau_global": cycle.remesh.plan.tau_global,
            "exact_reference_transition": ideal,
            "exact_rounding_residual": rounding,
            "exact_clipping_residual": clipping,
            "exact_total_residual": total,
            "exact_residual_decomposition": _difference(
                total, tuple(a + b for a, b in zip(rounding, clipping)),
            ),
            "clipping_intervened": cycle.remesh.plan.any_clipping_intervention,
            "phase_preserved": (
                cycle.phase_before_remesh == cycle.phase_after_remesh_before_refresh
                == cycle.phase_after_optional_refresh
            ),
            "capacity_preserved": not cycle.remesh_capacity_changed,
            "same_time_epi_boundary_recorded": (
                cycle.post_remesh_epi_time_boundary_recorded
            ),
            "post_map_pressure_refresh_calls": (
                cycle.post_remesh_pressure_refresh_callback_invocations
            ),
        },
    }


def _finite_envelope_observations(cycles, reference):
    """Compare captured pre-map histories with the detached policy envelope.

    These signed finite slacks are observations on this tuple, not a sealed
    runtime contraction certificate or a bound for any unexecuted cycle.
    """
    transitions = tuple(
        cycle["remesh"]["exact_reference_transition"] for cycle in cycles
    )
    policy = reference.euler_policy
    q = reference.euler_energy_gain_upper_bound
    matrix = policy.schedule_energy_domination_matrix
    energies = tuple(item.exact_history_energies for item in transitions)
    augmented = tuple(item.exact_augmented_energy_before for item in transitions)
    steps = []
    for index, (left, right) in enumerate(zip(energies[:-1], energies[1:])):
        bound = tuple(
            sum((a * b for a, b in zip(row, left, strict=True)), Fraction(0))
            for row in matrix
        )
        steps.append({
            "from_pre_map_sample": index, "to_pre_map_sample": index + 1,
            "actual_energy_vector_before": left,
            "actual_energy_vector_after": right,
            "exact_policy_envelope": bound,
            "signed_observed_envelope_slack": _difference(bound, right),
        })
    horizon = policy.universal_block_horizon
    blocks = [{
        "from_pre_map_sample": index,
        "to_pre_map_sample": index + horizon,
        "actual_augmented_energy_before": augmented[index],
        "actual_augmented_energy_after": augmented[index + horizon],
        "signed_observed_block_slack": (
            q * augmented[index] - augmented[index + horizon]
        ),
    } for index in range(len(augmented) - horizon)]
    return {
        "exact_reference_q": q, "block_horizon": horizon,
        "actual_augmented_energies": augmented,
        "adjacent_transitions": steps, "complete_blocks": blocks,
        "first_augmented_sample": (
            "V[0] uses first executed pre-map y[0] and explicitly prepared y[-1]"
        ),
        "runtime_contraction_certified": False,
        "scope": (
            "Exact arithmetic on captured finite energy vectors; signed "
            "observed slacks are not a future runtime guarantee"
        ),
    }


def run_cycle_memory_case(*, memory=True):
    """Execute four admitted cycles with or without the separate delayed map."""
    if type(memory) is not bool:
        raise TypeError("memory must be a bool")
    graph = _prepare_graph()
    initial = _live_state(graph)
    edge_attributes = tuple((u, v, dict(data)) for u, v, data in graph.edges(data=True))
    weights = dict(graph.graph["_dnfr_weights"])
    reference = certify_cycle_memory_relaxation(
        8, capacity=1, epi_weight=weights["epi"],
        step_sizes=tuple(Fraction.from_float(dt) for dt in STEPS),
        alpha=graph.graph["REMESH_ALPHA"], tau_local=1, tau_global=1,
    )
    specs = tuple(_spec(index) for index in range(CYCLE_COUNT))
    if memory:
        execution = execute_event_remesh_cycle_sequence(
            graph, specs, metric_weights=(1.0,) * 8,
            context={"initial_epi_nonzero": all(x != 0 for x in INITIAL_EPI)},
            method="euler", require_runtime_telescope=False,
        )
        cycles = [_memory_cycle_record(cycle, reference, weights)
                  for cycle in execution.cycles]
        # These three public provenance properties share the same seal
        # predicate. Retain that read-only result once; constructing the
        # detached report does not mutate the sealed execution or its graph.
        execution_intact = execution.causal_cycle_order_certified
        boundary_continuity = (
            execution_intact
            and execution.observed_sequence.exact_recorded_boundary_continuity_certified
        )
        provenance = {
            "causal_order_certified": execution_intact,
            "same_graph_provenance_certified": execution_intact,
            "whole_sequence_graph_state_atomic": execution_intact,
            "exact_recorded_boundary_continuity_certified": boundary_continuity,
            "runtime_telescope": execution.runtime_telescope,
            "runtime_global_gain_certified": execution.runtime_global_gain_certified,
            "future_stability_certified": execution.future_stability_certified,
        }
    else:
        cycles = []
        for spec in specs:
            before = _live_state(graph)
            event = execute_operator_event_schedule(
                graph, spec.schedule, method="euler",
                context={"initial_epi_nonzero": all(x != 0 for x in before["epi"])},
                include_stage_certificates=True,
                physical_flow_partitions=spec.physical_flow_partitions,
            )
            after = _live_state(graph)
            cycles.append({
                "before": before, "pre_remesh": after, "after": after,
                "flow": _flow_records(event, weights), "remesh": None,
                "whole_call_graph_state_atomic": (
                    event.whole_schedule_graph_state_atomic
                ),
            })
        provenance = {
            "scope": "Four consecutive same-graph calls; no outer transaction",
            "runtime_telescope": None, "runtime_global_gain_certified": False,
            "future_stability_certified": False,
        }
    final = _live_state(graph)
    clocks = tuple(
        sum((segment["node_clock_increments"][node]
             for cycle in cycles for segment in cycle["flow"]["segments"]),
            Fraction(0))
        for node in graph
    )
    return {
        "case": "delayed_memory" if memory else "advisory_only_control",
        "word": WORD, "flow_durations": FLOW_DURATIONS,
        "initial": initial, "final": final, "cycles": cycles,
        "normalized_channel_weights": weights,
        "initial_history": (_exact(INITIAL_EPI),),
        "final_history": tuple(
            _exact(row[node] for node in graph) for row in graph.graph["_epi_hist"]
        ),
        "history_preparation_scope": (
            "One explicitly supplied y[-1]=initial EPI row; no fabricated "
            "claim of an earlier executed trajectory"
        ),
        "pre_remesh_companion_transitions_observed": CYCLE_COUNT - 1 if memory else 0,
        "physical_elapsed_time": final["time"] - initial["time"],
        "node_reorganization_clocks": clocks,
        "edge_attributes_before": edge_attributes,
        "edge_attributes_after": tuple(
            (u, v, dict(data)) for u, v, data in graph.edges(data=True)
        ),
        "actual_glyph_history": tuple(
            tuple(graph.nodes[node]["glyph_history"]) for node in graph
        ),
        "exact_reference": reference,
        "finite_envelope_observations": (
            _finite_envelope_observations(cycles, reference) if memory else None
        ),
        "execution_provenance": provenance,
        "operator_and_map_scope": (
            "The admitted UM IL Recursivity word ends in an advisory. Only "
            "the memory case then uses the separate executor-owned history "
            "append and delayed map. No zero-alpha REMESH is executed."
        ),
        "reference_runtime_boundary": (
            "Exact regular-twist diffusion and REMESH envelope are detached "
            "model statements. Actual phase drift, remaining-channel pressure, "
            "rounding and clipping are measured separately; no asymptotic "
            "runtime convergence or self-localization is certified."
        ),
    }


def _payload(value):
    if isinstance(value, Fraction):
        return str(value)
    if is_dataclass(value):
        return {
            field.name: _payload(getattr(value, field.name))
            for field in fields(value) if not field.name.startswith("_")
        }
    if isinstance(value, dict):
        return {str(key): _payload(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_payload(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "artifacts/research/cycle_memory_relaxation.json",
    )
    args = parser.parse_args()
    scope = (
        "src/tnfr", "benchmarks/capacity_localization.py",
        "benchmarks/cycle_memory_relaxation.py",
    )
    provenance = current_git_source_provenance(ROOT, scope)
    sha, dirty, digest = provenance
    manifest = CoreExperimentManifest(
        claim_id="O3.a-finite-cycle-diffusion-with-delayed-memory",
        git_sha=sha, source_dirty=dirty, dirty_source_hash=digest,
        versions={
            "python": platform.python_version(),
            "networkx": nx.__version__, "numpy": np.__version__,
        },
        graph_construction="Unit-edge C8, regular winding=1, positive EPI bump",
        capacity_specification="Uniform positive nu=1; default UM and IL factors",
        solver="Existing physical event/history executors; four two-segment cycles",
        timestep=0.25, seed=17, result_status=ClaimStatus.MEASURED,
        operator_sequence=(
            "UM IL advisory REMESH", "separate delayed map in memory case",
        ),
        telemetry=(
            "canonical history append and delayed indices", "nodal triad and clocks",
            "pressure and Euler defects", "REMESH rounding and clipping defects",
            "finite same-invocation causal and atomic memory execution",
        ),
        controls=("same word and physical flow without the separate delayed map",),
        artifacts=(str(args.output),),
    )
    manifest.validate_for_admission()
    exact_manifest = replace(
        manifest, claim_id="O3.a-exact-fixed-cycle-memory-relaxation",
        solver="Detached exact diffusion gain and shared REMESH envelope theorem",
        timestep=None, result_status=ClaimStatus.DERIVED, operator_sequence=(),
    )
    exact_manifest.validate_for_admission()
    report = {
        "manifest": manifest.to_dict(),
        "exact_reference_manifest": exact_manifest.to_dict(), "source_scope": scope,
        "cases": [run_cycle_memory_case(memory=memory) for memory in (True, False)],
        "experimental_status": "No empirical correspondence tested",
    }
    if current_git_source_provenance(ROOT, scope) != provenance:
        raise RuntimeError("source changed while executing the research artifact")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_payload(report), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote finite cycle-memory observations to {args.output}")


if __name__ == "__main__":
    main()
