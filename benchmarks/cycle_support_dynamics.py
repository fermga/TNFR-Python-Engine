"""Finite UM/SHA support changes coupled to the canonical nodal EPI flow.

Three C8 preparations share four physical half-unit intervals. Existing event
execution owns every advance; exact cycle identities observe recorded inputs.
No continuous phase law, whole-experiment transaction or empirical mapping is
inferred from these finite endpoints.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
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
from tnfr.alias import get_attr, set_attr  # noqa: E402
from tnfr.constants.aliases import (  # noqa: E402
    ALIAS_DEPI, ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF,
)
from tnfr.constants.canonical import (  # noqa: E402
    COUPLING_GENTLE, SHA_VF_FACTOR, UM_THETA_PUSH,
)
from tnfr.dynamics.dnfr import default_compute_delta_nfr  # noqa: E402
from tnfr.operators import (  # noqa: E402
    build_operator_event_schedule, build_physical_flow_partition,
    execute_operator_event_schedule,
)
from tnfr.operators.definitions import Silence  # noqa: E402
from tnfr.physics.cycle_support_dynamics import (  # noqa: E402
    observe_cycle_support_balance, observe_cycle_support_euler,
    observe_cycle_support_reset,
)
from tnfr.physics.winding_certificates import (  # noqa: E402
    certify_phase_winding, observe_winding_word,
)
from tnfr.research.claims import ClaimStatus  # noqa: E402
from tnfr.research.core_manifests import (  # noqa: E402
    CoreExperimentManifest, current_git_source_provenance,
)
from tnfr.utils.numeric import angle_diff  # noqa: E402

CASES = ("evolving_support", "held_support", "uniform_bump")


def _values(graph, aliases):
    return tuple(float(get_attr(graph.nodes[node], aliases, 0.0)) for node in graph)


def _exact(values):
    return tuple(Fraction.from_float(float(value)) for value in values)


def _difference(left, right):
    return tuple(a - b for a, b in zip(left, right, strict=True))


def _phase_chart(phases):
    """Read a small periodic offset from the declared represented regular twist.

    These are rationalized measured coordinates, not an exact representation of
    mathematical pi. Their pressure/reset discrepancies remain in the report.
    """
    count = len(phases)
    return tuple(
        Fraction.from_float(angle_diff(phase, math.tau * node / count))
        / Fraction.from_float(math.pi)
        for node, phase in enumerate(phases)
    )


def _reference(epi, capacity, phase_chart, weights):
    return observe_cycle_support_balance(
        epi, capacity, phase_chart,
        epi_weight=weights["epi"], vf_weight=weights["vf"],
        phase_weight=weights["phase"],
    )


def _state(graph):
    epi, capacity, phase, pressure = (
        _values(graph, aliases)
        for aliases in (ALIAS_EPI, ALIAS_VF, ALIAS_THETA, ALIAS_DNFR)
    )
    offsets = _phase_chart(phase)
    weights = dict(graph.graph["_dnfr_weights"])
    model = _reference(epi, capacity, offsets, weights)
    gaps = tuple(
        angle_diff(phase[(node + 1) % len(phase)], value)
        for node, value in enumerate(phase)
    )
    return {
        "time": float(graph.graph["_t"]),
        "epi": epi, "capacity": capacity, "phase": phase,
        "capacity_range": max(capacity) - min(capacity),
        "pressure": pressure, "depi": _values(graph, ALIAS_DEPI),
        "pressure_origin": "stored value after the last canonical refresh",
        "normalized_channel_weights": weights,
        "phase_offset_over_represented_pi": offsets,
        "phase_gaps": gaps,
        "gap_spread": sum((gap - sum(gaps) / len(gaps)) ** 2 for gap in gaps) / 2,
        "strict_semicircle_margin": min(math.pi / 2 - abs(gap) for gap in gaps),
        "winding": asdict(certify_phase_winding(graph, tuple(graph))),
        "edge_support": tuple(graph.edges),
        "edge_attributes": tuple(
            (u, v, dict(data)) for u, v, data in graph.edges(data=True)
        ),
        "glyph_history": tuple(
            tuple(graph.nodes[node].get("glyph_history", ())) for node in graph
        ),
        "epi_range": max(epi) - min(epi),
        "core_minus_background_mean": epi[0] - sum(epi[1:]) / (len(epi) - 1),
        "ordinary_epi_dirichlet_energy": sum(
            (epi[(node + 1) % len(epi)] - value) ** 2
            for node, value in enumerate(epi)
        ) / 2,
        "reference": asdict(model),
        "exact_pressure_realization_residual": _difference(
            _exact(pressure), model.pressure,
        ),
    }


def _snapshot(snapshot):
    return {
        "epi": snapshot.epi, "capacity": snapshot.nu_f,
        "pressure": snapshot.delta_nfr,
        "phase_vector_retained": False,
    }


def _execute_cycle(graph, *, events):
    before = _state(graph)
    weights = before["normalized_channel_weights"]
    initial_model = _reference(
        before["epi"], before["capacity"],
        before["phase_offset_over_represented_pi"], weights,
    )
    word = ("coupling", "silence") if events else ()
    schedule = build_operator_event_schedule(
        word, start_time=graph.graph["_t"],
        flow_durations=(0.0, 0.0, 0.5) if events else (0.5,),
    )
    partition = build_physical_flow_partition(schedule.intervals[-1], (0.25, 0.25))
    result = execute_operator_event_schedule(
        graph, schedule, method="euler", include_stage_certificates=True,
        context={"initial_epi_nonzero": all(value != 0 for value in before["epi"])},
        physical_flow_partitions=(partition,),
    )
    after = _state(graph)
    chart = after["phase_offset_over_represented_pi"]
    evidence = result.physical_flow_partition_evidence[0]
    start = evidence.boundary_observations[0].after
    actual_reset = _reference(start.exact_epi, start.exact_nu_f, chart, weights)
    reset = None
    if events:
        predicted_reset = observe_cycle_support_reset(initial_model)
        reset = {
            "reference": asdict(predicted_reset),
            "actual_after_events_reference": asdict(actual_reset),
            "exact_epi_residual": _difference(
                actual_reset.epi, predicted_reset.after.epi,
            ),
            "exact_capacity_residual": _difference(
                actual_reset.capacity, predicted_reset.after.capacity,
            ),
            "exact_phase_coordinate_residual": _difference(
                chart, predicted_reset.after.phase_offset_over_pi,
            ),
            "actual_energy_change": (
                actual_reset.dirichlet_energy - initial_model.dirichlet_energy
            ),
            "exact_energy_realization_residual": (
                actual_reset.dirichlet_energy - initial_model.dirichlet_energy
                - predicted_reset.energy_change
            ),
        }
    boundaries = []
    for boundary in evidence.boundary_observations:
        model = _reference(
            boundary.after.exact_epi, boundary.after.exact_nu_f, chart, weights,
        )
        boundaries.append({
            "time": boundary.time, "exact_time": boundary.exact_time,
            "before": _snapshot(boundary.before), "after": _snapshot(boundary.after),
            "callback_name": boundary.callback_name,
            "pressure_only_refresh": boundary.nonpressure_state_preserved,
            "phase_preserved_during_refresh": boundary.phase_preserved,
            "capacity_preserved_during_refresh": boundary.capacity_preserved,
            "node_support_preserved": boundary.node_support_preserved,
            "edge_state_preserved": boundary.edge_state_preserved,
            "reference": asdict(model),
            "exact_pressure_realization_residual": _difference(
                boundary.after.exact_delta_nfr, model.pressure,
            ),
        })
    segments = []
    for segment, left, right in zip(
        evidence.segment_flow_evidence, evidence.boundary_observations[:-1],
        evidence.boundary_observations[1:], strict=True,
    ):
        model = _reference(left.after.exact_epi, left.after.exact_nu_f, chart, weights)
        dt = segment.interval.exact_duration
        prediction = observe_cycle_support_euler(model, dt=dt)
        actual_end = _reference(
            right.before.exact_epi, right.before.exact_nu_f, chart, weights,
        )
        pressure_defect = _difference(left.after.exact_delta_nfr, model.pressure)
        held_prediction = tuple(
            x + dt * nu * pressure
            for x, nu, pressure in zip(
                left.after.exact_epi, left.after.exact_nu_f,
                left.after.exact_delta_nfr, strict=True,
            )
        )
        held_defect = _difference(right.before.exact_epi, held_prediction)
        endpoint_defect = _difference(actual_end.epi, prediction.after.epi)
        decomposed = tuple(
            dt * nu * pressure + arithmetic
            for nu, pressure, arithmetic in zip(
                model.capacity, pressure_defect, held_defect, strict=True,
            )
        )
        segments.append({
            "duration": segment.interval.duration, "exact_duration": dt,
            "reference": asdict(prediction),
            "exact_pressure_realization_residual": pressure_defect,
            "exact_held_input_euler_residual": held_defect,
            "exact_model_endpoint_residual": endpoint_defect,
            "exact_endpoint_decomposition_residual": _difference(
                endpoint_defect, decomposed,
            ),
            "actual_energy_change": (
                actual_end.dirichlet_energy - model.dirichlet_energy
            ),
            "exact_energy_realization_residual": (
                actual_end.dirichlet_energy - model.dirichlet_energy
                - prediction.energy_change
            ),
            "node_clock_increments": tuple(dt * nu for nu in model.capacity),
            "capacity_held_during_flow": (
                left.after.exact_nu_f == right.before.exact_nu_f
            ),
            "integrator": segment.integrator_name,
            "method": segment.resolved_method,
            "clipping_applied": segment.clipping_applied,
            "gamma_is_none": segment.gamma_is_none,
            "extended_dynamics_requested": segment.extended_dynamics_requested,
        })
    exact_change = (
        after["reference"]["dirichlet_energy"] - initial_model.dirichlet_energy
    )
    actual_reset_change = actual_reset.dirichlet_energy - initial_model.dirichlet_energy
    telescope = actual_reset_change + sum(
        (item["actual_energy_change"] for item in segments), Fraction(0),
    )
    return {
        "before": before, "after": after,
        "schedule_word": word, "flow_durations": (0.0, 0.0, 0.5) if events else (0.5,),
        "grammar_admission": (
            "Existing executor validates each complete finite word "
            "against the live graph"
        ),
        "events": [{
            "operator": stage.event.operator_name,
            "left": _snapshot(stage.left), "right": _snapshot(stage.right),
            "pressure_scope": (
                "Stored executor stage endpoints; not an isolated observation "
                "of the internal auxiliary pressure write"
            ),
            "certificate_kind": stage.certificate_kind,
            "certificate_abstention_reason": stage.certificate_abstention_reason,
        } for stage in result.glyph_stage_evidence],
        "intermediate_phase_observation": False,
        "flow_phase_chart_scope": (
            "Inferred from the recorded cycle endpoint and the declared "
            "phase-preserving SHA/classical EPI-flow kernels; boundary flags "
            "certify pressure refresh only. "
            "No intermediate phase trajectory was retained."
        ),
        "reset": reset, "boundaries": boundaries, "segments": segments,
        "actual_shifted_energy_change": exact_change,
        "exact_energy_telescope_residual": exact_change - telescope,
        "whole_call_graph_state_atomic": result.whole_schedule_graph_state_atomic,
        "physical_pressure_refresh_calls": (
            result.physical_pressure_refresh_callback_invocations
        ),
        "stage_pressure_refresh_calls": (
            result.stage_pressure_refresh_callback_invocations
        ),
        "future_execution_certified": False,
    }


def run_cycle_support_case(case):
    """Execute four bounded existing-runtime invocations on one prepared C8."""
    if case not in CASES:
        raise ValueError(f"case must be one of {CASES}")
    graph = build_cycle(8, epi=(1.0,) + (0.5,) * 7 if case == "uniform_bump" else None)
    construction = _state(graph)
    if case != "uniform_bump":
        for node in graph:
            phase = math.tau * node / 8 + (math.pi / 16) * math.sin(math.tau * node / 8)
            set_attr(graph.nodes[node], ALIAS_THETA, phase % math.tau)
        default_compute_delta_nfr(graph)
    before_preparation = _state(graph)
    prep_history = ()
    if case != "uniform_bump":
        preparation = observe_winding_word(graph, tuple(graph), 0, [Silence()])
        prep_history = preparation.actual_history
        default_compute_delta_nfr(graph)
    initial = _state(graph)
    cycles = [_execute_cycle(graph, events=case != "held_support") for _ in range(4)]
    node_clocks = tuple(
        sum(
            (segment["node_clock_increments"][node]
             for cycle in cycles for segment in cycle["segments"]),
            Fraction(0),
        )
        for node in graph
    )
    energy_change = (
        cycles[-1]["after"]["reference"]["dirichlet_energy"]
        - initial["reference"]["dirichlet_energy"]
    )
    return {
        "case": case, "construction": construction,
        "phase_preparation": (
            "regular twist" if case == "uniform_bump" else
            "theta_i=2*pi*i/8+(pi/16)*sin(2*pi*i/8), initial preparation only"
        ),
        "capacity_preparation": {
            "before": before_preparation, "after": initial,
            "actual_operator_history": prep_history,
        },
        "cycles": cycles, "initial": initial, "final": cycles[-1]["after"],
        "physical_elapsed_time": 2.0, "node_reorganization_clocks": node_clocks,
        "reset_defaults": {
            "eta": UM_THETA_PUSH, "vf_sync": COUPLING_GENTLE,
            "silence_factor": SHA_VF_FACTOR,
        },
        "actual_shifted_energy_change": energy_change,
        "exact_multicall_telescope_residual": energy_change - sum(
            (cycle["actual_shifted_energy_change"] for cycle in cycles), Fraction(0),
        ),
        "atomicity_scope": (
            "Each invocation is atomic; the four-call experiment "
            "has no outer transaction"
        ),
        "claim_scope": (
            "Finite endpoints only; no autonomous confinement, future admission, "
            "mesh convergence or empirical correspondence"
        ),
    }


def _payload(value):
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _payload(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_payload(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "artifacts/research/cycle_support_dynamics.json",
    )
    args = parser.parse_args()
    scope = (
        "src/tnfr", "benchmarks/capacity_localization.py",
        "benchmarks/cycle_support_dynamics.py",
    )
    provenance = current_git_source_provenance(ROOT, scope)
    sha, dirty, digest = provenance
    manifest = CoreExperimentManifest(
        claim_id="O3.a-finite-changing-phase-capacity-support",
        git_sha=sha, source_dirty=dirty, dirty_source_hash=digest,
        versions={
            "python": platform.python_version(),
            "networkx": nx.__version__, "numpy": np.__version__,
        },
        graph_construction=(
            "Unit-edge C8 with winding=1; initial small periodic phase offsets "
            "or regular twist"
        ),
        capacity_specification=(
            "Actual one-SHA dip or uniform capacity; default UM synchronization "
            "and SHA attenuation"
        ),
        solver=(
            "Existing physical event executor, four half-unit calls, "
            "two refreshed shared Euler segments per call"
        ),
        timestep=0.25, result_status=ClaimStatus.MEASURED, seed=17,
        operator_sequence=(
            "single-node SHA preparation where declared",
            "four whole UM SHA words or empty-word held-support controls",
        ),
        telemetry=(
            "nodal triad and endpoint winding",
            "exact support reset and Euler energy budgets",
            "pressure and arithmetic residuals",
            "physical pressure boundaries and node clocks",
        ),
        controls=(
            "held support at equal elapsed time",
            "uniform capacity and regular twist with EPI bump",
        ),
        artifacts=(str(args.output),),
    )
    manifest.validate_for_admission()
    reference_manifest = replace(
        manifest, claim_id="O3.a-exact-cycle-support-energy-budget",
        solver=(
            "Detached exact rational cycle support/reset/Euler identities; "
            "no runtime advance"
        ),
        timestep=None, result_status=ClaimStatus.DERIVED, operator_sequence=(),
    )
    reference_manifest.validate_for_admission()
    report = {
        "manifest": manifest.to_dict(),
        "exact_reference_manifest": reference_manifest.to_dict(),
        "source_scope": scope,
        "cases": [run_cycle_support_case(case) for case in CASES],
        "exact_reference_scope": (
            "Fixed unit cycle with a consistent strict local phase chart; "
            "supplied normalized phase coordinates and recorded full-channel "
            "coefficients"
        ),
        "experimental_status": "No empirical correspondence tested",
    }
    if current_git_source_provenance(ROOT, scope) != provenance:
        raise RuntimeError("source changed while executing the research artifact")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_payload(report), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote finite changing-support observations to {args.output}")


if __name__ == "__main__":
    main()
