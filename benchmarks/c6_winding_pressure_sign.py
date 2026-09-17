"""Bounded, carried continuation to the first generated node-1 pressure sign hit.

The retained B27 endpoint is verified against its B26 source. At most eight
new cell boundaries and 256 numerical steps are available. Each complete
boundary preserves the actual remainder and refreshes the canonical pressure;
the first positive node-1 pressure stops the audit before that pressure acts.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from benchmarks.c6_winding_carry_itinerary import (  # noqa: E402
    SOURCE_SCOPE, _belongs, _itinerary_record, _prefix_summary, _state,
    analyze_c6_winding_carry_itinerary,
)
from benchmarks.c6_winding_pressure_lattice import (  # noqa: E402
    LOCAL_LOWER, LOCAL_UPPER, STEP, _record,
)
from benchmarks.c6_winding_rounding_cells import _represented_scalar  # noqa: E402
from benchmarks.thol_pressure_feedback import _payload  # noqa: E402
from tnfr.mathematics.unified_numerical import np  # noqa: E402
from tnfr.physics.nodal_remainder_pressure import observe_nodal_remainder_cycle_gradient  # noqa: E402
from tnfr.physics.c6_pressure_lattice import (  # noqa: E402
    derive_c6_pressure_lattice, observe_c6_pressure_lattice, observe_c6_pressure_sector_exit,
)
from tnfr.physics.nodal_remainder import (  # noqa: E402
    derive_nodal_remainder_cell_horizon, derive_nodal_remainder_itinerary,
    observe_nodal_remainder_cell_exit, observe_nodal_remainder_sequence,
)
from tnfr.research.claims import ClaimStatus  # noqa: E402
from tnfr.research.core_manifests import (  # noqa: E402
    CoreExperimentManifest, current_git_source_provenance,
)

MAX_BOUNDARIES, MAX_STEPS = 8, 256
TARGET_NODE = 1
CAPACITY = (1.0,) * 6


def _canonical(value):
    """Compare the complete retained schema without bool/int equivalence."""
    return json.dumps(_payload(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _budget(value, maximum, label):
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{label} must be an integer in [1, {maximum}]")
    return value


def _verified_source(parent, source_parent):
    manifest = CoreExperimentManifest(**parent["manifest"])
    manifest.validate_for_admission()
    if manifest.claim_id != "O3.a-C6-carried-cell-itinerary" or tuple(parent["source_scope"]) != SOURCE_SCOPE:
        raise ValueError("input must be the retained B27 carried-itinerary report")
    evidence = parent["input_evidence"]
    if (_canonical(evidence["producer_manifest"]) != _canonical(source_parent["manifest"])
            or tuple(evidence["producer_source_scope"]) != tuple(source_parent["source_scope"])
            or _canonical(evidence["producer_input_evidence"]) != _canonical(source_parent["input_evidence"])):
        raise ValueError("the B27 producer lineage differs from the supplied B26 source")
    rebuilt = analyze_c6_winding_carry_itinerary(source_parent)
    if any(key not in parent or _canonical(parent[key]) != _canonical(value) for key, value in rebuilt.items()):
        raise ValueError("the retained B27 report differs from its complete shared-owner replay")
    source = source_parent["lattice_reference"]["source"]
    reference = derive_c6_pressure_lattice(
        phase=tuple(rebuilt["source"]["phase"]),
        epi_weight=_represented_scalar(source["epi_weight"]),
        phase_weight=_represented_scalar(source["phase_weight"]),
        epi_lower=LOCAL_LOWER, epi_upper=LOCAL_UPPER,
    )
    initial = _state(parent["continuation"]["endpoint"])
    sector = observe_c6_pressure_sector_exit(
        reference, state=initial, node=TARGET_NODE, sign=-1, timestep=STEP,
    )
    if (_canonical(_record(sector.initial_observation))
            != _canonical(rebuilt["boundaries"][-1]["refreshed_pressure"])):
        raise ValueError("the inherited endpoint pressure differs from actual canonical refresh")
    return reference, initial, sector


def _gradient_balance(initial, endpoint, area, quantum):
    """Preserve the report schema through the shared physical area owner."""
    return asdict(observe_nodal_remainder_cycle_gradient(
        initial=initial, endpoint=endpoint, nodal_area=area, epi_quantum=quantum,
    ))


def _sector_record(sector):
    data = asdict(sector)
    data["sector"].pop("reference")
    data["initial_observation"] = _record(sector.initial_observation)
    data["opposite_sign_hit_certified"] = sector.opposite_sign_hit_certified
    data["positive_band_exit_certified"] = sector.positive_band_exit_certified
    return data


def analyze_c6_winding_pressure_sign(parent, source_parent, *, boundary_budget=MAX_BOUNDARIES, step_budget=MAX_STEPS):
    """Replay one bounded continuation; limits are analysis resources, not dynamics."""
    boundary_budget = _budget(boundary_budget, MAX_BOUNDARIES, "boundary_budget")
    step_budget = _budget(step_budget, MAX_STEPS, "step_budget")
    reference, initial, sector = _verified_source(parent, source_parent)
    state, pressure = initial, sector.initial_observation
    boundaries, steps, pressure_rows = [], [], []
    censor = None
    stop = "first_positive_node1_pressure" if pressure.pressure[TARGET_NODE] > 0 else None
    while stop is None:
        if len(boundaries) == boundary_budget:
            stop = "boundary_budget_exhausted"
            break
        remaining = step_budget - len(steps)
        if remaining == 0:
            stop = "step_budget_exhausted"
            break
        horizon = derive_nodal_remainder_cell_horizon(
            state=state, timestep=STEP, capacity=CAPACITY, pressure=pressure.pressure,
        )
        count = horizon.first_exit_step
        if count is None or any(horizon.first_exit_leaves_band) or count > remaining:
            stop = ("no_next_cell_exit" if count is None else "next_boundary_leaves_band"
                    if any(horizon.first_exit_leaves_band) else "next_boundary_exceeds_step_budget")
            censor = {"horizon": asdict(horizon), "remaining_step_budget": remaining}
            break
        exit_record = observe_nodal_remainder_cell_exit(
            state=state, timestep=STEP, capacity=CAPACITY, pressure=pressure.pressure,
            step_budget=remaining,
        )
        if exit_record.horizon != horizon:
            raise RuntimeError("preflight and shared cell-exit horizons differ")
        sequence = exit_record.sequence
        refreshed = observe_c6_pressure_lattice(reference, epi=sequence.endpoint.epi)
        local = _gradient_balance(state, sequence.endpoint, sequence.prefixes[-1].cumulative_nodal_area,
                                  reference.epi_quantum)
        if local["gradient_indices_before"] != pressure.gradient_indices or local["gradient_indices_after"] != refreshed.gradient_indices:
            raise RuntimeError("the nodal/carry gradient identity disagrees with canonical pressure observations")
        boundaries.append({
            "ordinal": len(boundaries) + 1, "initial": asdict(state), "horizon": asdict(horizon),
            "step_count": count, "cumulative_steps": len(steps) + count,
            "source_pressure": _record(pressure), "endpoint": asdict(sequence.endpoint),
            "refreshed_pressure": _record(refreshed), "prefix_balances": _prefix_summary(sequence),
            "gradient_balance": local,
        })
        steps.extend(sequence.steps)
        pressure_rows.extend((pressure.pressure,) * count)
        state, pressure = sequence.endpoint, refreshed
        if pressure.pressure[TARGET_NODE] > 0:
            stop = "first_positive_node1_pressure"

    actual = itinerary = None
    membership = None
    prefix_gradients = ()
    total_area = (Fraction(0),) * 6
    if steps:
        times, capacities = (STEP,) * len(steps), (CAPACITY,) * len(steps)
        actual = observe_nodal_remainder_sequence(
            initial=initial, timesteps=times, capacities=capacities, pressures=tuple(pressure_rows),
        )
        if actual.steps != tuple(steps) or actual.endpoint != state:
            raise RuntimeError("the whole continued prefix differs from its boundary replays")
        itinerary = derive_nodal_remainder_itinerary(
            epi_states=(initial.epi,) + tuple(step.after.epi for step in steps),
            timesteps=times, capacities=capacities, pressures=tuple(pressure_rows),
            epi_lower=LOCAL_LOWER, epi_upper=LOCAL_UPPER,
        )
        membership = _belongs(initial, itinerary)
        if not itinerary.feasible or not all(membership):
            raise RuntimeError("the continued itinerary lost the actual inherited carry")
        total_area = actual.prefixes[-1].cumulative_nodal_area
        prefix_gradients = tuple({
            "ordinal": prefix.ordinal,
            **_gradient_balance(initial, step.after, prefix.cumulative_nodal_area, reference.epi_quantum),
            "node1_source_pressure": step.pressure[TARGET_NODE],
            "node1_nodal_area": prefix.cumulative_nodal_area[TARGET_NODE],
        } for step, prefix in zip(actual.steps, actual.prefixes, strict=True))
    mean_area = sum(total_area, Fraction(0)) / 6
    final_balance = _gradient_balance(initial, state, total_area, reference.epi_quantum)
    return {
        "source": {
            "location": "B27.continuation.endpoint", "inherited_state": asdict(initial),
            "retained_B27_report_replayed": True, "retained_B26_selected_prefix_replayed": True,
            "phase": reference.source.phase, "epi_quantum": reference.epi_quantum,
            "pressure_generated_by_shared_cpu_kernel": True,
        },
        "analysis_limits": {"boundary_budget": boundary_budget, "step_budget": step_budget,
                            "maximum_boundaries": MAX_BOUNDARIES, "maximum_steps": MAX_STEPS},
        "initial_negative_sector": _sector_record(sector), "stop_reason": stop, "censor": censor,
        "boundaries": tuple(boundaries),
        "continuation": {
            "area_reference": "B27.continuation.endpoint",
            "step_count": len(steps), "boundary_count": len(boundaries), "steps": tuple(map(asdict, steps)),
            "prefix_balances": _prefix_summary(actual) if actual is not None else (),
            "gradient_prefixes": prefix_gradients, "gradient_balance": final_balance,
            "endpoint": asdict(state), "refreshed_pressure": _record(pressure),
            "total_nodal_area": total_area, "mean_nodal_area": mean_area,
            "mean_reconstructed_change": sum((b - a for a, b in zip(initial.exact_epi, state.exact_epi, strict=True)), Fraction(0)) / 6,
            "mean_visible_change": sum((Fraction(b) - Fraction(a) for a, b in zip(initial.epi, state.epi, strict=True)), Fraction(0)) / 6,
            "itinerary": _itinerary_record(itinerary) if itinerary is not None else None,
            "supplied_initial_carry_coordinate_membership": membership,
            "supplied_initial_carry_feasible": all(membership) if membership is not None else None,
        },
        "first_positive_pressure_observed": pressure.pressure[TARGET_NODE] > 0,
        "positive_node1_pressure_steps_integrated": sum(step.pressure[TARGET_NODE] > 0 for step in steps),
        "node1_nodal_area_compensated": bool(steps) and total_area[TARGET_NODE] == 0,
        "total_nodal_area_compensated": bool(steps) and not any(total_area),
        "mean_nodal_area_compensated": bool(steps) and mean_area == 0,
        "runtime_executed": False, "new_graph_trajectories": 0, "live_provenance_certified": False,
        "original_tail_reachability_certified": False, "future_compensation_cycle_certified": False,
        "positive_band_exit_certified": False,
        "scope": (
            "Fixed-source C6 carried continuation with canonical visible-state pressure refresh and explicit "
            "analysis budgets. The first positive node-1 source is observed before it is integrated. "
            "The sector residence bound is conditional and does not certify an opposite sign hit or band exit. "
            "Pressure sign, signed accumulated nodal area and a closed augmented-state cycle are distinct. "
            "No graph word, original-tail reachability, future compensation or full-runtime stability is asserted."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "artifacts/research/c6_winding_carry_itinerary.json")
    parser.add_argument("--source-input", type=Path, default=ROOT / "artifacts/research/c6_winding_pressure_lattice.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/research/c6_winding_pressure_sign.json")
    args = parser.parse_args()
    if args.output.resolve() in (args.input.resolve(), args.source_input.resolve()):
        raise ValueError("derived output must not overwrite either historical input")
    input_bytes, source_bytes = args.input.read_bytes(), args.source_input.read_bytes()
    parent, source_parent = json.loads(input_bytes), json.loads(source_bytes)
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    if parent["input_evidence"]["sha256"] != source_hash:
        raise ValueError("the B27 lineage hash does not identify the supplied B26 bytes")
    provenance = current_git_source_provenance(ROOT, SOURCE_SCOPE)
    report = analyze_c6_winding_pressure_sign(parent, source_parent)
    sha, dirty, digest = provenance
    manifest = CoreExperimentManifest(
        claim_id="O3.a-C6-carried-pressure-sign-hit", git_sha=sha,
        source_dirty=dirty, dirty_source_hash=digest,
        versions={"python": platform.python_version(), "numpy": np.__version__},
        graph_construction="Verified retained B27 endpoint with actual carry on the closed B25 C6 phase slice",
        capacity_specification="Unit capacity/support, default coefficients and local EPI slab [3/8,5/8]",
        solver="Shared exact cell-exit prefixes with fresh canonical pressure; stop at first positive node-1 source",
        result_status=ClaimStatus.DERIVED,
        telemetry=("conditional sign-sector residence bound", "generated source sign hit", "inverse carried itinerary",
                   "integer gradient nodal/carry identity", "signed cumulative nodal area"),
        controls=("at most eight new boundaries and 256 new steps", "no carry reset", "no positive-source step after hit"),
        artifacts=(str(args.output),),
    )
    manifest.validate_for_admission()
    if (current_git_source_provenance(ROOT, SOURCE_SCOPE) != provenance or args.input.read_bytes() != input_bytes
            or args.source_input.read_bytes() != source_bytes):
        raise RuntimeError("analysis source or either retained input changed during the bounded sign audit")
    report.update(
        manifest=manifest.to_dict(), source_scope=SOURCE_SCOPE,
        input_evidence={"path": str(args.input), "sha256": hashlib.sha256(input_bytes).hexdigest(),
                        "producer_manifest": parent["manifest"], "producer_source_scope": parent["source_scope"],
                        "producer_input_evidence": parent["input_evidence"]},
        source_input_evidence={"path": str(args.source_input), "sha256": source_hash,
                               "producer_manifest": source_parent["manifest"],
                               "producer_source_scope": source_parent["source_scope"]},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_payload(report), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote bounded C6 pressure-sign audit to {args.output}")


if __name__ == "__main__":
    main()
