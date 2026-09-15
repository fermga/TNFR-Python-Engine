"""Measure relative-profile relaxation and mean drift on frozen born support.

The attached case starts from actual THOL/UM execution. Compatible and clipped
controls are separately prepared graphs. All subsequent EPI changes use the
existing event executor and shared nodal Euler integrator.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

from benchmarks.thol_birth_transport import (  # noqa: E402
    _admitted_apply, prepare_birth_transport_support,
)
from benchmarks.thol_pressure_feedback import _payload, _state  # noqa: E402
from tnfr.config import inject_defaults  # noqa: E402
from tnfr.dynamics.dnfr import default_compute_delta_nfr  # noqa: E402
from tnfr.operators import (  # noqa: E402
    build_operator_event_schedule, execute_operator_event_schedule,
)
from tnfr.operators.definitions import Silence  # noqa: E402
from tnfr.physics.forced_support import (  # noqa: E402
    derive_forced_support_balance, observe_forced_support_state,
    observe_forced_support_step,
)
from tnfr.physics.forcing_realization import capture_non_epi_forcing  # noqa: E402
from tnfr.research.claims import ClaimStatus  # noqa: E402
from tnfr.research.core_manifests import (  # noqa: E402
    CoreExperimentManifest, current_git_source_provenance,
)

CASES = ("causal_attached", "prepared_compatible", "prepared_clipping")
SEGMENT_COUNTS = {
    "causal_attached": 24, "prepared_compatible": 12, "prepared_clipping": 2,
}
STEP = 0.25


def _prepared_control(source, *, clipping):
    """Prepare a separate graph; never overwrite the actual birth trajectory."""
    captured = _state(source)
    graph = nx.Graph()
    for index, node in enumerate(captured["nodes"]):
        graph.add_node(
            node,
            EPI=3.99 if clipping else captured["epi"][index],
            nu_f=captured["capacity"][index] if clipping else 1.0,
            theta=captured["phase"][index] if clipping else 0.0,
            delta_nfr=0.0, glyph_history=[],
        )
    graph.add_edges_from(captured["edges"])
    graph.graph.update(
        _t=0.0, RANDOM_SEED=17, GLYPH_HYSTERESIS_WINDOW=64,
        _gamma_spec={"type": "none"}, GAMMA={"type": "none"},
        use_extended_dynamics=False, DT_MIN=0.0,
        EPI_MIN=-4.0, EPI_MAX=4.0, CLIP_MODE="hard",
        compute_delta_nfr=default_compute_delta_nfr,
    )
    inject_defaults(graph)
    default_compute_delta_nfr(graph)
    return graph


def _same_frozen_inputs(initial, observed):
    before, after = initial.snapshot, observed.snapshot
    return {
        "nodes": before.nodes == after.nodes,
        "conductance": before.conductance == after.conductance,
        "support": before.support_neighbors == after.support_neighbors,
        "capacity": before.capacity == after.capacity,
        "phase": initial.phase == observed.phase,
        "normalized_weights": initial.normalized_weights == observed.normalized_weights,
        "forcing": initial.forcing == observed.forcing,
    }


def prepare_forced_support_endpoint(case="causal_attached"):
    """Return the actual campaign endpoint before any terminal SHA.

    The retained graph has executed the same fixed-input flow as the original
    benchmark. Callers own later events and the outstanding word closure.
    """
    if case not in CASES:
        raise ValueError(f"case must be one of {CASES}")
    source, preparation = prepare_birth_transport_support()
    if case == "causal_attached":
        graph = source
        preparation_scope = (
            "Actual post-THOL/UM graph at t=.5; no EPI, phase, capacity or "
            "support reassignment before this frozen-input campaign"
        )
    else:
        graph = _prepared_control(source, clipping=case == "prepared_clipping")
        preparation_scope = (
            "Independent prepared graph using the captured support. Node labels "
            "do not assert a birth history in this control. EPI/phase/capacity "
            "are declared initial data, not changes to the causal graph"
        )
    default_compute_delta_nfr(graph)
    initial_graph = _state(graph)
    captured = capture_non_epi_forcing(graph)
    reference = derive_forced_support_balance(
        captured.snapshot, epi_weight=captured.epi_weight, forcing=captured.forcing,
    )
    initial_state = observe_forced_support_state(reference, captured.snapshot)
    current = captured
    segments = []
    for _ in range(SEGMENT_COUNTS[case]):
        before = _state(graph)
        schedule = build_operator_event_schedule(
            (), start_time=before["time"], flow_durations=(STEP,),
        )
        execution = execute_operator_event_schedule(
            graph, schedule, method="euler", include_flow_certificates=True,
        )
        evidence = execution.flow_interval_evidence[0]
        raw = _state(graph)
        default_compute_delta_nfr(graph)
        following = capture_non_epi_forcing(graph)
        frozen = _same_frozen_inputs(captured, following)
        if not all(frozen.values()):
            raise RuntimeError("a declared frozen input changed during the campaign")
        observation = observe_forced_support_step(
            reference, current.snapshot, following.snapshot, dt=STEP,
        )
        step_payload = asdict(observation)
        # One common model is serialized at case level, not once per segment.
        step_payload.pop("reference")
        segments.append({
            "before": before, "raw_after_integrator": raw,
            "after_refresh": _state(graph), "duration": STEP,
            "method": evidence.resolved_method,
            "clipping_applied": evidence.clipping_applied,
            "frozen_input_checks": frozen,
            "forcing_capture": asdict(following),
            "exact_step_observation": step_payload,
        })
        current = following
    final_graph = _state(graph)
    final_state = observe_forced_support_state(reference, current.snapshot)
    return graph, {
        "case": case, "source_preparation": preparation,
        "preparation_scope": preparation_scope,
        "initial": initial_graph, "initial_forcing_capture": asdict(captured),
        "reference": asdict(reference), "initial_relative_state": asdict(initial_state),
        "segments": segments, "final": final_graph,
        "final_relative_state": asdict(final_state),
        "physical_elapsed_time": final_graph["time"] - initial_graph["time"],
        "scope": (
            "Exact held-input reference and finite actual Euler observations. "
            "Pressure is refreshed explicitly between existing event-executor "
            "intervals; each executor invocation owns its transaction. The "
            "relative profile is an offline reference, never assigned to live "
            "EPI. Clipping and binary64 defects are retained; no complete "
            "runtime convergence or empirical correspondence is asserted"
        ),
    }


def run_forced_support_case(case):
    """Execute the original campaign and its separately recorded word closure."""
    graph, record = prepare_forced_support_endpoint(case)
    closure = None
    if case == "causal_attached":
        closure = {
            "admission": _admitted_apply(graph, Silence()),
            "after": _state(graph),
            "scope": "Parent SHA closes the actual word after the measured campaign",
        }
    return {**record, "closure_after_measurement": closure}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "artifacts/research/forced_support_balance.json",
    )
    args = parser.parse_args()
    scope = (
        "src/tnfr", "benchmarks/capacity_localization.py",
        "benchmarks/thol_pressure_feedback.py", "benchmarks/thol_birth_transport.py",
        "benchmarks/forced_support_balance.py",
    )
    provenance = current_git_source_provenance(ROOT, scope)
    sha, dirty, digest = provenance
    manifest = CoreExperimentManifest(
        claim_id="O3.a-frozen-born-support-profile-and-mean-drift",
        git_sha=sha, source_dirty=dirty, dirty_source_hash=digest,
        versions={
            "python": platform.python_version(),
            "networkx": nx.__version__, "numpy": np.__version__,
        },
        graph_construction=(
            "Actual attached C8+child; separate prepared support controls"
        ),
        capacity_specification="Held actual positive capacity; unit-capacity control",
        solver="Shared nodal Euler through existing event intervals; explicit refresh",
        timestep=STEP, seed=17, result_status=ClaimStatus.MEASURED,
        operator_sequence=("actual IL OZ THOL UM", "held-input flow", "terminal SHA"),
        telemetry=(
            "non-EPI forcing realization", "relative profile and metric mean",
            "pressure and Euler defects", "hard clipping and energy balances",
        ),
        controls=("prepared zero forcing", "prepared upper-bound clipping"),
        artifacts=(str(args.output),),
    )
    manifest.validate_for_admission()
    report = {
        "manifest": manifest.to_dict(), "source_scope": scope,
        "cases": [run_forced_support_case(case) for case in CASES],
        "experimental_status": "No empirical correspondence tested",
    }
    if current_git_source_provenance(ROOT, scope) != provenance:
        raise RuntimeError("source changed while executing the research artifact")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_payload(report), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote finite frozen-support observations to {args.output}")


if __name__ == "__main__":
    main()
