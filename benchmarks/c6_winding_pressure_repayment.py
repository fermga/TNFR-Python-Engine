"""One carried cell exit tests whether a positive nodal source repays prior area.

The initial debt is the complete B28 area relative to the B27 endpoint.
Only the next analytic cell exit is executed. A constant-source crossing
outside that certified prefix is an algebraic comparison, not a trajectory.
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
)
from benchmarks.c6_winding_pressure_lattice import LOCAL_LOWER, LOCAL_UPPER, _record  # noqa: E402
from benchmarks.c6_winding_pressure_sign import (  # noqa: E402
    CAPACITY, MAX_STEPS, STEP, TARGET_NODE, _budget, _canonical, _gradient_balance,
    analyze_c6_winding_pressure_sign,
)
from benchmarks.c6_winding_rounding_cells import _represented_scalar  # noqa: E402
from benchmarks.thol_pressure_feedback import _payload  # noqa: E402
from tnfr.mathematics.unified_numerical import np  # noqa: E402
from tnfr.physics.c6_pressure_lattice import derive_c6_pressure_lattice, observe_c6_pressure_lattice  # noqa: E402
from tnfr.physics.nodal_remainder import (  # noqa: E402
    derive_nodal_remainder_cell_horizon, derive_nodal_remainder_itinerary, observe_nodal_remainder_cell_exit,
)
from tnfr.physics.nodal_remainder_pressure import (  # noqa: E402
    derive_nodal_area_crossings, derive_two_level_nodal_return,
)
from tnfr.research.claims import ClaimStatus  # noqa: E402
from tnfr.research.core_manifests import CoreExperimentManifest, current_git_source_provenance  # noqa: E402


def _verified_source(parent, previous, source_parent):
    manifest = CoreExperimentManifest(**parent["manifest"])
    manifest.validate_for_admission()
    if manifest.claim_id != "O3.a-C6-carried-pressure-sign-hit" or tuple(parent["source_scope"]) != SOURCE_SCOPE:
        raise ValueError("input must be the retained B28 carried-pressure-sign report")
    for evidence, producer in ((parent["input_evidence"], previous), (parent["source_input_evidence"], source_parent)):
        if (_canonical(evidence["producer_manifest"]) != _canonical(producer["manifest"])
                or tuple(evidence["producer_source_scope"]) != tuple(producer["source_scope"])):
            raise ValueError("the supplied B27/B26 sources differ from the B28 producer lineage")
    if _canonical(parent["input_evidence"]["producer_input_evidence"]) != _canonical(previous["input_evidence"]):
        raise ValueError("the B28 input lineage differs from the supplied B27 source chain")
    rebuilt = analyze_c6_winding_pressure_sign(previous, source_parent)
    if any(key not in parent or _canonical(parent[key]) != _canonical(value) for key, value in rebuilt.items()):
        raise ValueError("the retained B28 report differs from its complete bounded shared-owner replay")
    if rebuilt["continuation"]["area_reference"] != "B27.continuation.endpoint":
        raise ValueError("the inherited accumulated area uses an unexpected reference")
    initial = _state(parent["continuation"]["endpoint"])
    baseline = _state(parent["source"]["inherited_state"])
    incoming = tuple(rebuilt["continuation"]["total_nodal_area"])
    if incoming != tuple(b - a for a, b in zip(baseline.exact_epi, initial.exact_epi, strict=True)):
        raise ValueError("incoming nodal area does not equal the actual reconstructed change from B27")
    source = source_parent["lattice_reference"]["source"]
    reference = derive_c6_pressure_lattice(
        phase=tuple(rebuilt["source"]["phase"]), epi_weight=_represented_scalar(source["epi_weight"]),
        phase_weight=_represented_scalar(source["phase_weight"]), epi_lower=LOCAL_LOWER, epi_upper=LOCAL_UPPER,
    )
    pressure = tuple(rebuilt["continuation"]["refreshed_pressure"]["pressure"])
    if pressure[TARGET_NODE] <= 0 or incoming[TARGET_NODE] >= 0:
        raise ValueError("this discriminator requires the first positive source with an outstanding negative node-1 area")
    return reference, initial, baseline, incoming, pressure, rebuilt["continuation"]["refreshed_pressure"]


def _net_prefixes(baseline, incoming, sequence):
    records = []
    for step, prefix in zip(sequence.steps, sequence.prefixes, strict=True):
        net = tuple(a + b for a, b in zip(incoming, prefix.cumulative_nodal_area, strict=True))
        reconstructed = tuple(b - a for a, b in zip(baseline.exact_epi, step.after.exact_epi, strict=True))
        visible = tuple(Fraction(b) - Fraction(a) for a, b in zip(baseline.epi, step.after.epi, strict=True))
        remainder = tuple(b - a for a, b in zip(baseline.remainder, step.after.remainder, strict=True))
        residual = tuple(a - b for a, b in zip(net, reconstructed, strict=True))
        display_residual = tuple(a - b - c for a, b, c in zip(net, visible, remainder, strict=True))
        if any(residual) or any(display_residual):
            raise RuntimeError("the B27-reference nodal area lost its reconstructed/display/carry identity")
        records.append({
            "ordinal": prefix.ordinal, "local_nodal_area": prefix.cumulative_nodal_area,
            "B27_nodal_area": net, "B27_reconstructed_change": reconstructed,
            "B27_visible_change": visible, "B27_remainder_change": remainder,
            "identity_residual": residual, "display_carry_identity_residual": display_residual,
            "local_mean_nodal_area": prefix.mean_nodal_area, "B27_mean_nodal_area": sum(net, Fraction(0)) / 6,
            "exact_zero_nodes": tuple(i for i, value in enumerate(net) if value == 0),
            "displayed_matches_B27_reference": tuple(value == 0 for value in visible),
            "reconstructed_matches_B27_reference": tuple(value == 0 for value in reconstructed),
            "node1_first_side_restored": net[TARGET_NODE] >= 0,
            "joint_zero": not any(net), "mean_zero": sum(net, Fraction(0)) == 0,
        })
    return tuple(records)


def analyze_c6_winding_pressure_repayment(parent, previous, source_parent, *, step_budget=MAX_STEPS):
    """Execute at most one analytic boundary, retaining the incoming nodal debt."""
    step_budget = _budget(step_budget, MAX_STEPS, "step_budget")
    reference, initial, baseline, incoming, pressure, initial_observation = _verified_source(parent, previous, source_parent)
    two_level = derive_two_level_nodal_return(
        negative_pressure=_represented_scalar(parent["initial_negative_sector"]["initial_observation"]["pressure"][TARGET_NODE]),
        positive_pressure=pressure[TARGET_NODE],
    )
    horizon = derive_nodal_remainder_cell_horizon(state=initial, timestep=STEP, capacity=CAPACITY, pressure=pressure)
    count = horizon.first_exit_step
    crossings = None
    if count is not None and not any(horizon.first_exit_leaves_band):
        crossings = derive_nodal_area_crossings(
            initial_area=incoming, timestep=STEP, capacity=CAPACITY, pressure=pressure, max_steps=count,
        )
    sequence = itinerary = None
    membership = None
    state, refreshed = initial, initial_observation
    local_area = (Fraction(0),) * 6
    net_prefixes = ()
    if count is None or any(horizon.first_exit_leaves_band) or count > step_budget:
        stop = ("no_next_cell_exit" if count is None else "next_boundary_leaves_band"
                if any(horizon.first_exit_leaves_band) else "next_boundary_exceeds_step_budget")
    else:
        observed = observe_nodal_remainder_cell_exit(
            state=initial, timestep=STEP, capacity=CAPACITY, pressure=pressure, step_budget=step_budget,
        )
        if observed.horizon != horizon:
            raise RuntimeError("preflight and shared cell-exit horizons differ")
        sequence = observed.sequence
        state = sequence.endpoint
        refreshed = _record(observe_c6_pressure_lattice(reference, epi=state.epi))
        local_area = sequence.prefixes[-1].cumulative_nodal_area
        net_prefixes = _net_prefixes(baseline, incoming, sequence)
        itinerary = derive_nodal_remainder_itinerary(
            epi_states=(initial.epi,) + tuple(step.after.epi for step in sequence.steps),
            timesteps=(STEP,) * count, capacities=(CAPACITY,) * count, pressures=(pressure,) * count,
            epi_lower=LOCAL_LOWER, epi_upper=LOCAL_UPPER,
        )
        membership = _belongs(initial, itinerary)
        if not itinerary.feasible or not all(membership):
            raise RuntimeError("the one-boundary itinerary lost the inherited actual carry")
        stop = "first_nonpositive_node1_pressure" if refreshed["pressure"][TARGET_NODE] <= 0 else "one_boundary_complete_positive_sector_open"
    net = tuple(a + b for a, b in zip(incoming, local_area, strict=True))
    completed = sequence is not None
    first_cross = next((row["ordinal"] for row in net_prefixes if row["node1_first_side_restored"]), None)
    first_zero = next((row["ordinal"] for row in net_prefixes if TARGET_NODE in row["exact_zero_nodes"]), None)
    gradient = _gradient_balance(initial, state, local_area, reference.epi_quantum)
    crossing_record = asdict(crossings) if crossings is not None else None
    if crossing_record is not None:
        crossing_record.update(pressure_provenance_certified=crossings.pressure_provenance_certified,
                               band_provenance_certified=crossings.band_provenance_certified)
    two_level_record = asdict(two_level)
    two_level_record.update(pressure_provenance_certified=two_level.pressure_provenance_certified,
                            periodic_execution_certified=two_level.periodic_execution_certified)
    return {
        "source": {
            "location": "B28.continuation.endpoint", "inherited_state": asdict(initial),
            "retained_B28_report_replayed": True, "B27_area_reference_state": asdict(baseline),
            "incoming_B27_nodal_area": incoming, "incoming_B27_mean_nodal_area": sum(incoming, Fraction(0)) / 6,
            "phase": reference.source.phase, "pressure_generated_by_shared_cpu_kernel": True,
        },
        "analysis_limits": {"boundary_budget": 1, "step_budget": step_budget, "maximum_steps": MAX_STEPS},
        "horizon": asdict(horizon), "source_pressure": initial_observation,
        "held_source_crossings": crossing_record,
        "held_source_crossings_scope": (
            "Constant supplied source arithmetic; max_steps is the certified next-cell prefix. "
            "Crossing indices beyond that prefix are not executed or certified canonical future inputs."
        ),
        "two_level_return": two_level_record,
        "two_level_return_scope": (
            "Necessary primitive counts for exact node-1 area return using only these two captured pressure "
            "levels with one fixed positive timestep and capacity. Other source levels and actual reachability "
            "are not covered; the primitive count is not an executed trajectory or a global cycle bound."
        ),
        "stop_reason": stop,
        "positive_sector": {
            "completed_steps": len(sequence.steps) if completed else 0,
            "duration_exact_steps": count if completed and refreshed["pressure"][TARGET_NODE] <= 0 else None,
            "duration_lower_bound_steps": count if completed else 0,
            "termination_observed": completed and refreshed["pressure"][TARGET_NODE] <= 0,
            "future_duration_open": not completed or refreshed["pressure"][TARGET_NODE] > 0,
        },
        "continuation": {
            "local_area_reference": "B28.continuation.endpoint", "net_area_reference": "B27.continuation.endpoint",
            "step_count": len(sequence.steps) if completed else 0,
            "steps": tuple(map(asdict, sequence.steps)) if completed else (),
            "prefix_balances": _prefix_summary(sequence) if completed else (), "net_prefixes": net_prefixes,
            "endpoint": asdict(state), "refreshed_pressure": refreshed,
            "local_nodal_area": local_area, "B27_nodal_area": net,
            "displayed_matches_B27_reference": tuple(a == b for a, b in zip(baseline.epi, state.epi, strict=True)),
            "reconstructed_matches_B27_reference": tuple(value == 0 for value in net),
            "displayed_vector_returns_to_B27_reference": baseline.epi == state.epi,
            "reconstructed_vector_returns_to_B27_reference": not any(net),
            "local_mean_nodal_area": sum(local_area, Fraction(0)) / 6,
            "B27_mean_nodal_area": sum(net, Fraction(0)) / 6,
            "gradient_balance": gradient,
            "itinerary": _itinerary_record(itinerary) if completed else None,
            "supplied_initial_carry_coordinate_membership": membership,
            "supplied_initial_carry_feasible": all(membership) if membership is not None else None,
        },
        "first_observed_node1_nonnegative_area_step": first_cross,
        "first_observed_node1_exact_zero_step": first_zero,
        "node1_nodal_area_compensated": net[TARGET_NODE] == 0,
        "total_nodal_area_compensated": not any(net), "mean_nodal_area_compensated": sum(net, Fraction(0)) == 0,
        "post_exit_steps_integrated": 0, "hypothetical_crossing_steps_integrated": 0,
        "runtime_executed": False, "new_graph_trajectories": 0, "live_provenance_certified": False,
        "original_tail_reachability_certified": False, "future_compensation_cycle_certified": False,
        "positive_band_exit_certified": False,
        "scope": (
            "One conditional C6 cell boundary from the verified B28 carried endpoint, with actual canonical "
            "pressure refresh. Net nodal areas retain the B27 endpoint baseline; local areas use the B28 endpoint. "
            "The positive-sector duration is exact only if the first refreshed pressure is nonpositive, otherwise "
            "it is a lower bound. No hypothetical repayment steps, live graph word or future compensation are inferred."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "artifacts/research/c6_winding_pressure_sign.json")
    parser.add_argument("--parent-input", type=Path, default=ROOT / "artifacts/research/c6_winding_carry_itinerary.json")
    parser.add_argument("--source-input", type=Path, default=ROOT / "artifacts/research/c6_winding_pressure_lattice.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/research/c6_winding_pressure_repayment.json")
    args = parser.parse_args()
    paths = (args.input, args.parent_input, args.source_input)
    if args.output.resolve() in tuple(path.resolve() for path in paths):
        raise ValueError("derived output must not overwrite any historical input")
    raw = tuple(path.read_bytes() for path in paths)
    parent, previous, source_parent = tuple(json.loads(value) for value in raw)
    hashes = tuple(hashlib.sha256(value).hexdigest() for value in raw)
    if (parent["input_evidence"]["sha256"] != hashes[1]
            or parent["source_input_evidence"]["sha256"] != hashes[2]
            or previous["input_evidence"]["sha256"] != hashes[2]):
        raise ValueError("the B28/B27 lineage hashes do not identify the supplied historical bytes")
    provenance = current_git_source_provenance(ROOT, SOURCE_SCOPE)
    report = analyze_c6_winding_pressure_repayment(parent, previous, source_parent)
    sha, dirty, digest = provenance
    manifest = CoreExperimentManifest(
        claim_id="O3.a-C6-carried-pressure-repayment", git_sha=sha, source_dirty=dirty, dirty_source_hash=digest,
        versions={"python": platform.python_version(), "numpy": np.__version__},
        graph_construction="Verified B28 first-positive-pressure endpoint with retained carry and B27 area reference",
        capacity_specification="Unit C6 capacity/support, default channel weights and EPI slab [3/8,5/8]",
        solver="One analytically derived shared carried cell exit and fresh canonical pressure; no graph word",
        result_status=ClaimStatus.DERIVED,
        telemetry=("finite positive-source duration", "exact held-source area crossing indices",
                   "local and inherited nodal area budgets", "fresh negative pressure and partial repayment"),
        controls=("one cell boundary", "256-step computational ceiling", "actual carry retained",
                  "no hypothetical crossing steps executed"), artifacts=(str(args.output),),
    )
    manifest.validate_for_admission()
    if current_git_source_provenance(ROOT, SOURCE_SCOPE) != provenance or tuple(path.read_bytes() for path in paths) != raw:
        raise RuntimeError("analysis source or historical input changed during the one-boundary repayment audit")
    report.update(manifest=manifest.to_dict(), source_scope=SOURCE_SCOPE)
    for key, path, digest, producer in zip(
        ("input_evidence", "parent_input_evidence", "source_input_evidence"), paths, hashes,
        (parent, previous, source_parent), strict=True,
    ):
        report[key] = {"path": str(path), "sha256": digest, "producer_manifest": producer["manifest"],
                       "producer_source_scope": producer["source_scope"]}
    report["input_evidence"]["producer_input_evidence"] = parent["input_evidence"]
    report["input_evidence"]["producer_source_input_evidence"] = parent["source_input_evidence"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_payload(report), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote bounded C6 pressure-repayment audit to {args.output}")


if __name__ == "__main__":
    main()
