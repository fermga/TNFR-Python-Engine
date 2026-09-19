"""Distinguish winding, a decaying EPI bump, and held-capacity balance.

The benchmark executes the existing physical event scheduler and refreshed
nodal Euler partitions. Exact fixed-cycle identities, binary64 endpoints and
explicit preparations have separate scope. No laboratory correspondence is
claimed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from fractions import Fraction
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

from tnfr.alias import get_attr, set_attr  # noqa: E402
from tnfr.constants.aliases import (  # noqa: E402
    ALIAS_DNFR,
    ALIAS_EPI,
    ALIAS_THETA,
    ALIAS_VF,
)
from tnfr.dynamics.dnfr import default_compute_delta_nfr  # noqa: E402
from tnfr.operators import (  # noqa: E402
    build_operator_event_schedule,
    build_physical_flow_partition,
    execute_operator_event_schedule,
)
from tnfr.operators.definitions import Silence  # noqa: E402
from tnfr.operators.word_execution import run_network_sequence  # noqa: E402
from tnfr.physics.capacity_localization import (  # noqa: E402
    observe_cycle_capacity_balance,
)
from tnfr.physics.emergent_particles import winding_ring  # noqa: E402
from tnfr.physics.winding_certificates import (  # noqa: E402
    certify_phase_winding,
    observe_winding_word,
)
from tnfr.research.claims import ClaimStatus  # noqa: E402
from tnfr.research.core_manifests import (  # noqa: E402
    CoreExperimentManifest,
    current_git_source_provenance,
)


def build_cycle(count, *, winding=1, epi=None, capacity=None):
    """Prepare a declared unit cycle before executing any dynamics."""
    graph = winding_ring(count, winding)
    epi = (0.5,) * count if epi is None else tuple(epi)
    capacity = (1.0,) * count if capacity is None else tuple(capacity)
    if len(epi) != count or len(capacity) != count:
        raise ValueError("preparation vectors must match the cycle")
    graph.graph.update(
        _t=0.0,
        RANDOM_SEED=17,
        GLYPH_HYSTERESIS_WINDOW=64,
        _gamma_spec={"type": "none"},
        GAMMA={"type": "none"},
        use_extended_dynamics=False,
        DT_MIN=0.0,
        EPI_MIN=-4.0,
        EPI_MAX=4.0,
        CLIP_MODE="hard",
        UM_BIDIRECTIONAL=False,
        UM_FUNCTIONAL_LINKS=False,
        compute_delta_nfr=default_compute_delta_nfr,
    )
    for node in graph:
        set_attr(graph.nodes[node], ALIAS_EPI, float(epi[node]))
        set_attr(graph.nodes[node], ALIAS_VF, float(capacity[node]))
        graph.nodes[node]["glyph_history"] = []
    for edge in graph.edges:
        graph.edges[edge].update(weight=1.0, length=1.0)
    default_compute_delta_nfr(graph)
    return graph


def _values(graph, aliases):
    return tuple(float(get_attr(graph.nodes[node], aliases, None)) for node in graph)


def _reference(graph, *, epi=None, capacity=None):
    weights = graph.graph["_dnfr_weights"]
    return observe_cycle_capacity_balance(
        _values(graph, ALIAS_EPI) if epi is None else epi,
        _values(graph, ALIAS_VF) if capacity is None else capacity,
        epi_weight=weights["epi"],
        vf_weight=weights["vf"],
    )


def _channel_profile(reference, pressure):
    epi_pressure = tuple(
        -reference.epi_weight
        * sum((entry * value for entry, value in zip(row, reference.epi)), Fraction(0))
        for row in reference.laplacian
    )
    capacity_pressure = tuple(
        total - epi for total, epi in zip(reference.pressure, epi_pressure)
    )
    return {
        "exact_model_epi_pressure": epi_pressure,
        "exact_model_capacity_pressure": capacity_pressure,
        "exact_runtime_minus_model_pressure": tuple(
            Fraction.from_float(float(actual)) - ideal
            for actual, ideal in zip(pressure, reference.pressure, strict=True)
        ),
        "scope": "Remaining-channel and arithmetic residual; not silently zeroed",
    }


def _state(graph, *, pressure_origin="stored graph value; readout does not refresh"):
    reference = _reference(graph)
    epi = _values(graph, ALIAS_EPI)
    pressure = _values(graph, ALIAS_DNFR)
    return {
        "time": float(graph.graph["_t"]),
        "epi": epi,
        "epi_range": max(epi) - min(epi),
        "core_minus_background_mean": epi[0] - sum(epi[1:]) / (len(epi) - 1),
        "capacity": _values(graph, ALIAS_VF),
        "phase": _values(graph, ALIAS_THETA),
        "pressure": pressure,
        "pressure_origin": pressure_origin,
        "normalized_channel_weights": dict(graph.graph["_dnfr_weights"]),
        "edge_support": tuple(graph.edges),
        "winding": asdict(certify_phase_winding(graph, tuple(graph))),
        "reference": asdict(reference),
        "channel_profile": _channel_profile(reference, pressure),
    }


def execute_refreshed_flow(graph, *, duration=4.0, step=0.25):
    """Advance this graph through one existing executor-owned Euler partition."""
    horizon = Fraction.from_float(float(duration))
    mesh = Fraction.from_float(float(step))
    if horizon <= 0 or mesh <= 0 or horizon % mesh:
        raise ValueError("duration must be a positive integer multiple of step")
    count = int(horizon / mesh)
    if not 2 <= count <= 256:
        raise ValueError("this finite benchmark requires 2..256 segments")
    initial = _state(
        graph,
        pressure_origin=(
            "pre-executor stored state; first captured boundary refreshes pressure"
        ),
    )
    reference = _reference(graph)
    schedule = build_operator_event_schedule(
        (), start_time=graph.graph["_t"], flow_durations=(duration,)
    )
    partition = build_physical_flow_partition(schedule.intervals[0], (step,) * count)
    result = execute_operator_event_schedule(
        graph, schedule, method="euler", physical_flow_partitions=(partition,)
    )
    evidence = result.physical_flow_partition_evidence[0]
    boundaries = []
    for boundary in evidence.boundary_observations:
        snapshot = boundary.after
        model = _reference(graph, epi=snapshot.epi, capacity=snapshot.nu_f)
        boundaries.append(
            {
                "time": boundary.time,
                "exact_time": boundary.exact_time,
                "epi": snapshot.epi,
                "capacity": snapshot.nu_f,
                "pressure": snapshot.delta_nfr,
                "channel_profile": _channel_profile(model, snapshot.delta_nfr),
                "callback_name": boundary.callback_name,
                "pressure_only_refresh": boundary.nonpressure_state_preserved,
            }
        )
    segments = []
    for segment, left, right in zip(
        evidence.segment_flow_evidence,
        evidence.boundary_observations[:-1],
        evidence.boundary_observations[1:],
        strict=True,
    ):
        dt = segment.interval.exact_duration
        predicted = tuple(
            x + dt * nu * pressure
            for x, nu, pressure in zip(
                left.after.exact_epi,
                left.after.exact_nu_f,
                left.after.exact_delta_nfr,
                strict=True,
            )
        )
        segments.append(
            {
                "duration": segment.interval.duration,
                "exact_held_input_euler_residual": tuple(
                    actual - ideal
                    for actual, ideal in zip(
                        right.before.exact_epi, predicted, strict=True
                    )
                ),
                "integrator": segment.integrator_name,
                "method": segment.resolved_method,
                "clipping_applied": segment.clipping_applied,
                "gamma_is_none": segment.gamma_is_none,
            }
        )
    final = _state(
        graph, pressure_origin="executor-owned terminal canonical pressure refresh"
    )
    exact_final = tuple(Fraction.from_float(value) for value in final["epi"])
    deviation = tuple(
        value - equilibrium
        for value, equilibrium in zip(
            exact_final, reference.equilibrium_epi, strict=True
        )
    )
    final_error = (
        sum(
            (
                weight * error**2
                for weight, error in zip(
                    reference.metric_weights, deviation, strict=True
                )
            ),
            Fraction(0),
        )
        / 2
    )
    drift = sum(
        (
            weight * (end - start)
            for weight, end, start in zip(
                reference.metric_weights, exact_final, reference.epi, strict=True
            )
        ),
        Fraction(0),
    )
    return {
        "initial": initial,
        "final": final,
        "duration": duration,
        "step": step,
        "segments": segments,
        "boundaries": boundaries,
        "initial_error_energy_to_fixed_profile": reference.lyapunov_value,
        "final_error_energy_to_fixed_profile": final_error,
        "exact_fixed_metric_total_drift": drift,
        "physical_pressure_refresh_calls": (
            result.physical_pressure_refresh_callback_invocations
        ),
        "scope": (
            "Finite executor-owned refreshed Euler trajectory; no exact solver "
            "accuracy, mesh convergence, future schedule or experimental claim"
        ),
    }


def run_fixed_capacity_case(count, preparation, *, winding=1):
    """Separate imposed EPI contrast from one operator-created capacity dip."""
    if preparation == "epi_bump":
        epi = (1.0,) + (0.5,) * (count - 1)
        graph = build_cycle(count, winding=winding, epi=epi)
        preparation_record = {
            "kind": "explicit initial EPI bump",
            "actual_operator_history": (),
        }
    elif preparation == "single_silence":
        graph = build_cycle(count, winding=winding)
        before = _state(graph)
        result = observe_winding_word(graph, tuple(graph), 0, [Silence()])
        preparation_record = {
            "kind": "one actual canonical SHA at node0; then held capacity",
            "before": before,
            "after": _state(graph),
            "actual_operator_history": result.actual_history,
        }
    else:
        raise ValueError("unknown declared preparation")
    return {
        "nodes": count,
        "prepared_winding": winding,
        "preparation": preparation_record,
        "flow": execute_refreshed_flow(graph),
    }


def run_capacity_release_case(count=8):
    """Apply canonical UM/SHA to an explicitly balanced initial preparation."""
    preparation_graph = build_cycle(count)
    preparation = observe_winding_word(
        preparation_graph, tuple(preparation_graph), 0, [Silence()]
    )
    reference = _reference(preparation_graph)
    # This is a new declared initial state from the derived balance, not an
    # assignment into an executing graph or a claim it evolved into balance.
    graph = build_cycle(
        count, epi=reference.equilibrium_epi, capacity=reference.capacity
    )
    balanced = _state(graph)
    events = []
    run_network_sequence(
        graph,
        ["coupling", "silence"],
        context={"initial_epi_nonzero": all(x != 0.0 for x in balanced["epi"])},
        on_step=lambda operator: events.append(
            {
                "operator": operator,
                **_state(
                    graph,
                    pressure_origin=(
                        "after dispatcher-owned default_compute_delta_nfr refresh; "
                        "on_step only reads the graph"
                    ),
                ),
            }
        ),
    )
    return {
        "capacity_preparation_history": preparation.actual_history,
        "balance_preparation": (
            "explicit derived initial profile, not an emerged state"
        ),
        "balanced_initial": balanced,
        "canonical_release_events": events,
        "released_flow": execute_refreshed_flow(graph),
        "scope": (
            "UM synchronizes capacity; SHA attenuates it; balance must be rederived"
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
        "--output",
        type=Path,
        default=ROOT / "artifacts/research/capacity_localization.json",
    )
    args = parser.parse_args()
    scope = ("src/tnfr", "benchmarks/capacity_localization.py")
    sha, dirty, digest = current_git_source_provenance(ROOT, scope)
    manifest = CoreExperimentManifest(
        claim_id="O3.a-held-capacity-localization-and-release",
        git_sha=sha,
        source_dirty=dirty,
        dirty_source_hash=digest,
        versions={
            "python": platform.python_version(),
            "networkx": nx.__version__,
            "numpy": np.__version__,
        },
        graph_construction=("Unit-edge C8/C16 with winding=1; C8 winding=0 controls"),
        capacity_specification=(
            "Uniform or actual one-SHA preparation; held in each flow"
        ),
        solver="Existing event executor, refreshed shared nodal Euler, T=4",
        timestep=0.25,
        result_status=ClaimStatus.MEASURED,
        seed=17,
        operator_sequence=(
            "single SHA preparation",
            "UM SHA release",
            "event-free flow",
        ),
        telemetry=(
            "exact held-capacity balance",
            "per-channel pressure residual",
            "executor physical boundaries",
            "clipping and Euler residual",
            "nodal triad and winding",
            "fixed-metric total and profile error",
        ),
        controls=(
            "uniform capacity",
            "winding=0",
            "C8/C16",
            "actual capacity-writing release",
        ),
        artifacts=(str(args.output),),
    )
    manifest.validate_for_admission()
    exact_manifest = replace(
        manifest,
        claim_id="O3.a-exact-held-capacity-cycle-balance",
        solver="Detached exact rational fixed-cycle reference; no runtime advance",
        timestep=None,
        result_status=ClaimStatus.DERIVED,
        operator_sequence=(),
        telemetry=(
            "shifted pressure identity",
            "fixed-metric invariant",
            "stationary profile",
            "shifted Lyapunov/Dirichlet identities",
        ),
    )
    exact_manifest.validate_for_admission()
    report = {
        "manifest": manifest.to_dict(),
        "exact_reference_manifest": exact_manifest.to_dict(),
        "source_scope": scope,
        "exact_reference_scope": (
            "Held positive capacity on a unit cycle; zero remaining channel "
            "pressures and no clipping. Runtime residuals remain separate."
        ),
        "fixed_capacity_cases": [
            run_fixed_capacity_case(count, preparation)
            for count in (8, 16)
            for preparation in ("epi_bump", "single_silence")
        ],
        "zero_winding_controls": [
            run_fixed_capacity_case(8, preparation, winding=0)
            for preparation in ("epi_bump", "single_silence")
        ],
        "capacity_release": run_capacity_release_case(),
        "experimental_status": "No empirical correspondence tested",
    }
    final_sha, final_dirty, final_digest = current_git_source_provenance(ROOT, scope)
    if (sha, dirty, digest) != (final_sha, final_dirty, final_digest):
        raise RuntimeError("source changed while executing the research artifact")
    report["manifest"] = replace(
        manifest,
        git_sha=final_sha,
        source_dirty=final_dirty,
        dirty_source_hash=final_digest,
    ).to_dict()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_payload(report), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote finite capacity-localization observations to {args.output}")


if __name__ == "__main__":
    main()
