"""Test a centered-energy tube with an independent mean interval at the B43 tail.

Replaying retained arithmetic does not recreate a graph-owned execution seal.
The escape witnesses are hypothetical members of a candidate domain, never
continuations or carry resets of the actual retained B43 endpoint.
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

from benchmarks.c6_winding_carry_itinerary import SOURCE_SCOPE  # noqa: E402
from benchmarks.c6_winding_phase_response import BASE_PHASE  # noqa: E402
from benchmarks.c6_winding_phase_tail_runtime import (  # noqa: E402
    EPI_LOWER, EPI_UPPER, INTERVAL_DURATION, SEGMENT_DURATION, TAIL_ENTRY_CYCLE,
    _declared_schedule,
)
from benchmarks.c6_winding_pressure_sign import _canonical  # noqa: E402
from benchmarks.thol_pressure_feedback import _payload  # noqa: E402
from tnfr.config import DEFAULTS  # noqa: E402
from tnfr.dynamics._euler_kernel import NodalRemainderState, advance_nodal_remainder  # noqa: E402
from tnfr.mathematics.unified_numerical import np  # noqa: E402
from tnfr.physics.c6_carried_closure import derive_c6_carried_closure  # noqa: E402
from tnfr.physics.c6_carried_affine_mean import (  # noqa: E402
    derive_c6_carried_affine_mean_obstruction, observe_c6_carried_affine_mean_escape,
)
from tnfr.physics.c6_carried_mean_cylinder import (  # noqa: E402
    derive_c6_carried_mean_cylinder_obstruction, observe_c6_carried_mean_cylinder_escape,
)
from tnfr.physics.c6_carried_profile import derive_c6_carried_profile  # noqa: E402
from tnfr.physics.c6_phase_orbit import observe_c6_coupling_coherence_phase_step  # noqa: E402
from tnfr.physics.c6_pressure_lattice import (  # noqa: E402
    derive_c6_pressure_lattice, _observe_rebuilt_c6_pressure_lattice,
)
from tnfr.research.claims import ClaimStatus  # noqa: E402
from tnfr.research.core_manifests import CoreExperimentManifest, current_git_source_provenance  # noqa: E402
from tnfr.utils import normalize_weights  # noqa: E402

INPUT_NAME = "c6_winding_temporal_compatibility.json"


def _same(observed, expected, label):
    if _canonical(observed) != _canonical(expected):
        raise ValueError(f"retained B43 {label} differs from its declared arithmetic replay")


def _event(record, name, time, before, after, state, capacity_after=(1.,) * 6):
    _same(record, {
        "operator_name": name, "time": time, "phase_before": before, "phase_after": after,
        "capacity_before": (1.,) * 6, "capacity_after": capacity_after,
        "state_before": asdict(state), "state_after": asdict(state), "binding_preserved": True,
        "stage_schedule": "two_phase_jacobi", "target_count": 6,
    }, f"{name} record")


def replay_c6_phase_tail_evidence(parent):
    """Validate the retained phase/area chain through shared numerical owners.

    This authenticates the serialized numerical consistency of the declared
    source, not the historical graph, auxiliary writes or an active seal.
    No new operator word or future nodal step is executed.
    """
    manifest = CoreExperimentManifest(**parent["manifest"])
    manifest.validate_for_admission()
    if (manifest.claim_id != "O3.a-C6-temporal-cell-compatibility"
            or tuple(parent["source_scope"]) != SOURCE_SCOPE):
        raise ValueError("the mean-cylinder study requires the retained B43 report")
    runtime = parent["B43_original_phase_tail_runtime"]
    if not isinstance(runtime, dict):
        raise ValueError("the original full B43 capture is required")
    for key, value in {
        "cycle_count": 89, "event_count": 179, "flow_count": 356,
        "runtime_provenance_certified_at_capture": True,
        "phase_observations_retained_in_executor_seal": True,
        "declared_phase_tail_reached": True, "phase_projection_fixed_at_endpoint": True,
        "all_measured_flows_have_unit_capacity": True, "unit_capacity_after_terminal_SHA": False,
        "historical_detached_endpoint_reachability_certified": False,
        "future_runtime_certified": False, "indefinite_trapping_certified": False,
    }.items():
        _same(runtime[key], value, key)
    phase = tuple(BASE_PHASE)
    state = NodalRemainderState((.5,) * 6, (F(0),) * 6, EPI_LOWER, EPI_UPPER)
    initial = state
    weights = normalize_weights(DEFAULTS["DNFR_WEIGHTS"], ("phase", "epi", "vf", "topo"))
    _same(runtime["source"], {
        "preparation": "canonical null C6 winding", "initial_phase": phase,
        "initial_state": asdict(state), "initial_capacity": (1.,) * 6,
        "ordered_nodes": tuple(range(6)), "support": ((0, 1), (0, 5), (1, 2), (2, 3), (3, 4), (4, 5)),
        "normalized_weights": tuple((key, F(value)) for key, value in weights.items()),
        "random_seed": 17, "declared_carried_band": (EPI_LOWER, EPI_UPPER), "carry_imported_or_reset": False,
    }, "source")
    _same(runtime["schedule"], asdict(_declared_schedule(TAIL_ENTRY_CYCLE)[0]), "schedule")
    if len(runtime["cycles"]) != TAIL_ENTRY_CYCLE:
        raise ValueError("the retained B43 cycle chain must be complete")
    area = (F(0),) * 6
    for index, cycle in enumerate(runtime["cycles"]):
        _same(cycle["ordinal"], index + 1, "cycle ordinal")
        expected = observe_c6_coupling_coherence_phase_step(phase=phase)
        _same(cycle["phase_projection"], asdict(expected), "phase projection")
        time = index * INTERVAL_DURATION
        _event(cycle["coupling"], "coupling", time, phase, expected.phase_after_coupling, state)
        phase = expected.phase_after_coherence
        _event(cycle["coherence"], "coherence", time, expected.phase_after_coupling, phase, state)
        lattice = derive_c6_pressure_lattice(
            phase=phase, epi_weight=weights["epi"], phase_weight=weights["phase"],
            epi_lower=EPI_LOWER, epi_upper=EPI_UPPER,
        )
        if len(cycle["flows"]) != 4:
            raise ValueError("each retained B43 cycle must contain four refreshed segments")
        tail = state
        for segment, flow in enumerate(cycle["flows"]):
            pressure = _observe_rebuilt_c6_pressure_lattice(lattice, state.epi).pressure
            step = advance_nodal_remainder(state, timestep=SEGMENT_DURATION, capacity=(1.,) * 6, pressure=pressure)
            _same(flow, {
                "step": asdict(step), "phase_before": phase, "phase_after": phase,
                "start_time": time + segment * SEGMENT_DURATION,
                "end_time": time + (segment + 1) * SEGMENT_DURATION, "fresh_pressure_matches": True,
            }, "refreshed flow")
            state = step.after
            area = tuple(a + b for a, b in zip(area, step.exact_increment, strict=True))
    capacity_after = tuple(runtime["terminal_silence"]["capacity_after"])
    if (len(capacity_after) != 6
            or any(type(value) is not float or not 0 <= value < 1 for value in capacity_after)):
        raise ValueError("the retained SHA capacity boundary must be explicit")
    _event(runtime["terminal_silence"], "silence", 22.25, phase, phase, state, capacity_after)
    _same(runtime["tail_entry_after_last_IL"], asdict(tail), "tail-entry state")
    _same(runtime["endpoint_before_SHA"], asdict(state), "pre-SHA endpoint")
    _same(runtime["tail_entry_time"], 22., "tail-entry time")
    _same(runtime["endpoint_time"], 22.25, "endpoint time")
    _same(runtime["endpoint_phase"], phase, "endpoint phase")
    _same(observe_c6_coupling_coherence_phase_step(phase=phase).phase_after_coherence, phase, "fixed phase")
    change = tuple(b - a for a, b in zip(initial.exact_epi, state.exact_epi, strict=True))
    _same(runtime["exact_nodal_area"], area, "nodal area")
    _same(runtime["exact_reconstructed_change"], change, "reconstructed change")
    _same(runtime["exact_nodal_balance_residual"], tuple(a - b for a, b in zip(change, area)), "nodal residual")
    _same(runtime["mean_area"], sum(area, F(0)) / 6, "mean area")
    profile = derive_c6_carried_profile(lattice)
    tail_closure = asdict(derive_c6_carried_closure(profile, state=tail, timestep=SEGMENT_DURATION))
    tail_closure.pop("base_tube")
    _same(runtime["tail_entry_closure"], tail_closure, "tail closure")
    _same(runtime["tail_entry_profile"], asdict(profile.forced_balance), "tail profile")
    return profile, state


def analyze_c6_winding_mean_cylinder(parent):
    """Refute the separable candidate without advancing the retained endpoint."""
    profile, state = replay_c6_phase_tail_evidence(parent)
    closure = derive_c6_carried_closure(profile, state=state, timestep=SEGMENT_DURATION)
    rows = parent["source"]["B38_visible_rows"]
    if len(rows) != 7:
        raise ValueError("the retained comparison requires the seven B38 visible rows")

    def template(index):
        row = tuple(rows[index])
        carry = closure.base_tube.initial_mean - sum(map(F, row), F(0)) / 6
        return NodalRemainderState(row, (carry,) * 6, EPI_LOWER, EPI_UPPER)

    obstruction = derive_c6_carried_mean_cylinder_obstruction(
        closure, positive_state=template(3), negative_state=template(0),
    )
    bound = asdict(obstruction)
    bound.pop("closure")
    mean = closure.base_tube.initial_mean
    cases = {}
    for name, (lower, upper) in {
        "complete_local_mean_window": (obstruction.mean_lower, obstruction.mean_upper),
        "actual_initial_mean_slice": (mean, mean),
    }.items():
        record = asdict(observe_c6_carried_mean_cylinder_escape(obstruction, mean_lower=lower, mean_upper=upper))
        record.pop("obstruction")
        cases[name] = record
    affine = derive_c6_carried_affine_mean_obstruction(
        closure, positive_state=obstruction.positive_point.state, negative_state=obstruction.negative_point.state,
    )
    affine_bound = asdict(affine)
    affine_bound.pop("base_obstruction")
    affine_cases = {}
    for name, (lower, upper) in {
        "complete_affine_mean_window": (affine.mean_lower, affine.mean_upper),
        "actual_initial_mean_slice": (mean, mean),
    }.items():
        record = asdict(observe_c6_carried_affine_mean_escape(affine, mean_lower=lower, mean_upper=upper))
        record.pop("obstruction")
        affine_cases[name] = record
    return {
        "source": {"retained_B43_endpoint": asdict(state), "phase": profile.lattice.source.phase,
                   "historical_nodal_steps_replayed": 356, "serialized_numerical_chain_verified": True,
                   "live_execution_seal_recreated": False, "origin_mean": mean,
                   "energy_bound": closure.energy_bound, "origin_energy": closure.base_tube.initial_energy},
        "B44_mean_cylinder_obstruction": bound, "B44_hypothetical_boundary_witnesses": cases,
        "B45_affine_mean_obstruction": affine_bound, "B45_hypothetical_boundary_witnesses": affine_cases,
        "local_energy_mean_cylinder_invariance_excluded": True,
        "B44_coordinate_affine_coset_restriction_applied": False,
        "B45_coordinate_affine_coset_restriction_applied": True,
        "local_affine_mean_cylinder_invariance_excluded": True,
        "general_correlated_region_excluded": False, "saved_trajectory_escape_certified": False,
        "new_saved_trajectory_steps": 0, "future_runtime_certified": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "artifacts/research" / INPUT_NAME)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/research/c6_winding_mean_cylinder.json")
    args = parser.parse_args()
    if args.output.resolve() == args.input.resolve():
        raise ValueError("the derived report must not overwrite its historical input")
    raw = args.input.read_bytes()
    parent = json.loads(raw)
    provenance = current_git_source_provenance(ROOT, SOURCE_SCOPE)
    report = analyze_c6_winding_mean_cylinder(parent)
    sha, dirty, digest = provenance
    manifest = CoreExperimentManifest(
        claim_id="O3.a-C6-carried-mean-cylinder-obstruction", git_sha=sha,
        source_dirty=dirty, dirty_source_hash=digest,
        versions={"python": platform.python_version(), "numpy": np.__version__},
        graph_construction="Fixed unit C6 source from retained B43 pre-SHA state; no new graph invocation",
        capacity_specification="Conditional unit capacity before historical terminal SHA",
        solver="Exact translated carry cells, canonical increment gcds and affine mean lifts; historical arithmetic replay",
        result_status=ClaimStatus.DERIVED,
        telemetry=("centered energy", "exact reconstructed mean", "coordinate increment gcds", "outward boundary witnesses"),
        controls=("actual B43 carry preserved", "no historical live seal recreated", "bounded local mean window",
                  "affine membership does not establish temporal reachability or saved-trajectory escape"),
        artifacts=(str(args.output),),
    )
    manifest.validate_for_admission()
    if current_git_source_provenance(ROOT, SOURCE_SCOPE) != provenance or args.input.read_bytes() != raw:
        raise RuntimeError("source or historical input changed during the mean-cylinder audit")
    report.update(manifest=manifest.to_dict(), source_scope=SOURCE_SCOPE, input_evidence={
        "path": str(args.input), "sha256": hashlib.sha256(raw).hexdigest(),
        "producer_manifest": parent["manifest"], "producer_source_scope": parent["source_scope"],
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_payload(report), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote C6 mean-cylinder report to {args.output}")


if __name__ == "__main__":
    main()
