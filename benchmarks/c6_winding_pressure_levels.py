"""Stop a carried C6 continuation at the first source outside two retained levels.

Eight new cell boundaries and 256 shared steps are strict analysis ceilings.
The captured new pressure is not integrated. Exact finite-level arithmetic and
a signed-coordinate bound audit the observed class without asserting closure.
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
    SOURCE_SCOPE,
    _belongs,
    _itinerary_record,
    _prefix_summary,
    _state,
)
from benchmarks.c6_winding_pressure_lattice import (
    LOCAL_LOWER,
    LOCAL_UPPER,
    _record,
)  # noqa: E402
from benchmarks.c6_winding_pressure_repayment import (
    _net_prefixes,
    analyze_c6_winding_pressure_repayment,
)  # noqa: E402
from benchmarks.c6_winding_pressure_sign import (  # noqa: E402
    CAPACITY,
    MAX_BOUNDARIES,
    MAX_STEPS,
    STEP,
    TARGET_NODE,
    _budget,
    _canonical,
    _gradient_balance,
)
from benchmarks.c6_winding_rounding_cells import _represented_scalar  # noqa: E402
from benchmarks.thol_pressure_feedback import _payload  # noqa: E402
from tnfr.mathematics.unified_numerical import np  # noqa: E402
from tnfr.physics.c6_pressure_lattice import (
    derive_c6_pressure_lattice,
    observe_c6_pressure_lattice,
)  # noqa: E402
from tnfr.physics.nodal_remainder import (  # noqa: E402
    derive_nodal_remainder_cell_horizon,
    derive_nodal_remainder_itinerary,
    observe_nodal_remainder_cell_exit,
    observe_nodal_remainder_sequence,
)
from tnfr.physics.nodal_remainder_pressure import (  # noqa: E402
    derive_finite_level_nodal_return,
    observe_finite_nodal_pressure_drift,
)
from tnfr.research.claims import ClaimStatus  # noqa: E402
from tnfr.research.core_manifests import (
    CoreExperimentManifest,
    current_git_source_provenance,
)  # noqa: E402


def _verified_source(parent, previous, ancestor, source_parent):
    manifest = CoreExperimentManifest(**parent["manifest"])
    manifest.validate_for_admission()
    if (
        manifest.claim_id != "O3.a-C6-carried-pressure-repayment"
        or tuple(parent["source_scope"]) != SOURCE_SCOPE
    ):
        raise ValueError(
            "input must be the retained B29 carried-pressure-repayment report"
        )
    for key, producer in (
        ("input_evidence", previous),
        ("parent_input_evidence", ancestor),
        ("source_input_evidence", source_parent),
    ):
        evidence = parent[key]
        if _canonical(evidence["producer_manifest"]) != _canonical(
            producer["manifest"]
        ) or tuple(evidence["producer_source_scope"]) != tuple(
            producer["source_scope"]
        ):
            raise ValueError(
                "the supplied B28/B27/B26 sources differ from the B29 producer lineage"
            )
    for retained, key in (
        ("producer_input_evidence", "input_evidence"),
        ("producer_source_input_evidence", "source_input_evidence"),
    ):
        if _canonical(parent["input_evidence"][retained]) != _canonical(previous[key]):
            raise ValueError(
                "the B29 lineage differs from the supplied B28 input chain"
            )
    rebuilt = analyze_c6_winding_pressure_repayment(previous, ancestor, source_parent)
    if any(
        key not in parent or _canonical(parent[key]) != _canonical(value)
        for key, value in rebuilt.items()
    ):
        raise ValueError(
            "the retained B29 report differs from its complete shared-owner replay"
        )
    initial = _state(parent["continuation"]["endpoint"])
    baseline = _state(parent["source"]["B27_area_reference_state"])
    incoming = tuple(rebuilt["continuation"]["B27_nodal_area"])
    if rebuilt["continuation"][
        "net_area_reference"
    ] != "B27.continuation.endpoint" or incoming != tuple(
        b - a for a, b in zip(baseline.exact_epi, initial.exact_epi, strict=True)
    ):
        raise ValueError(
            "incoming area differs from the complete reconstructed change since the B27 endpoint"
        )
    source = source_parent["lattice_reference"]["source"]
    reference = derive_c6_pressure_lattice(
        phase=tuple(rebuilt["source"]["phase"]),
        epi_weight=_represented_scalar(source["epi_weight"]),
        phase_weight=_represented_scalar(source["phase_weight"]),
        epi_lower=LOCAL_LOWER,
        epi_upper=LOCAL_UPPER,
    )
    levels = tuple(
        rebuilt["two_level_return"][key]
        for key in ("negative_pressure", "positive_pressure")
    )
    observation = rebuilt["continuation"]["refreshed_pressure"]
    if observation["pressure"][TARGET_NODE] not in levels:
        raise ValueError(
            "the inherited B29 endpoint is already outside the declared two-level source"
        )
    return reference, initial, baseline, incoming, levels, observation


def _return_record(levels):
    result = derive_finite_level_nodal_return(pressure_levels=levels)
    return {
        **asdict(result),
        "pressure_provenance_certified": result.pressure_provenance_certified,
        "periodic_execution_certified": result.periodic_execution_certified,
    }


def _finite_class(observations):
    unique = {}
    for observation in observations:
        key = tuple(observation["epi"])
        values = tuple(observation["pressure"])
        if key in unique and unique[key] != values:
            raise RuntimeError(
                "one visible state generated two different pressures under the fixed source"
            )
        unique[key] = values
    states, pressures = tuple(unique), tuple(unique.values())
    observed = observe_finite_nodal_pressure_drift(
        epi_states=states,
        pressure_vectors=pressures,
        functional=(Fraction(0),) * 4 + (Fraction(1), Fraction(0)),
        timestep=STEP,
        epi_lower=LOCAL_LOWER,
        epi_upper=LOCAL_UPPER,
    )
    return {
        "unique_state_count": len(states),
        "observation": asdict(observed),
        "conditional_class_escape_certified": observed.conditional_class_escape_certified,
        "pressure_provenance_certified": observed.pressure_provenance_certified,
        "positive_band_exit_certified": observed.positive_band_exit_certified,
        "scope": (
            "The selected +e4 functional uses the already generated canonical pressures on deduplicated visible "
            "states. Its bound excludes indefinite residence in this finite class with any admissible carry; "
            "it does not assert that the class is invariant, that the EPI band is exited or that a graph is stable."
        ),
    }


def analyze_c6_winding_pressure_levels(
    parent,
    previous,
    ancestor,
    source_parent,
    *,
    boundary_budget=MAX_BOUNDARIES,
    step_budget=MAX_STEPS,
):
    """Continue under resource ceilings until the first captured new node-1 level."""
    boundary_budget = _budget(boundary_budget, MAX_BOUNDARIES, "boundary_budget")
    step_budget = _budget(step_budget, MAX_STEPS, "step_budget")
    reference, initial, baseline, incoming, levels, observation = _verified_source(
        parent, previous, ancestor, source_parent
    )
    state, current = initial, observation
    boundaries, steps, pressures = [], [], []
    observations = [current]
    stop = censor = None
    while stop is None:
        if len(boundaries) == boundary_budget:
            stop = "boundary_budget_exhausted"
            break
        remaining = step_budget - len(steps)
        if remaining == 0:
            stop = "step_budget_exhausted"
            break
        pressure = tuple(current["pressure"])
        horizon = derive_nodal_remainder_cell_horizon(
            state=state, timestep=STEP, capacity=CAPACITY, pressure=pressure
        )
        count = horizon.first_exit_step
        if count is None or any(horizon.first_exit_leaves_band) or count > remaining:
            stop = (
                "no_next_cell_exit"
                if count is None
                else (
                    "next_boundary_leaves_band"
                    if any(horizon.first_exit_leaves_band)
                    else "next_boundary_exceeds_step_budget"
                )
            )
            censor = {"horizon": asdict(horizon), "remaining_step_budget": remaining}
            break
        exit_record = observe_nodal_remainder_cell_exit(
            state=state,
            timestep=STEP,
            capacity=CAPACITY,
            pressure=pressure,
            step_budget=remaining,
        )
        if exit_record.horizon != horizon:
            raise RuntimeError(
                "the preflight horizon differs from the shared cell-exit replay"
            )
        sequence = exit_record.sequence
        refreshed = _record(
            observe_c6_pressure_lattice(reference, epi=sequence.endpoint.epi)
        )
        boundaries.append(
            {
                "ordinal": len(boundaries) + 1,
                "step_count": count,
                "cumulative_steps": len(steps) + count,
                "initial": asdict(state),
                "horizon": asdict(horizon),
                "source_pressure": current,
                "endpoint": asdict(sequence.endpoint),
                "refreshed_pressure": refreshed,
                "prefix_balances": _prefix_summary(sequence),
                "gradient_balance": _gradient_balance(
                    state,
                    sequence.endpoint,
                    sequence.prefixes[-1].cumulative_nodal_area,
                    reference.epi_quantum,
                ),
            }
        )
        steps.extend(sequence.steps)
        pressures.extend((pressure,) * count)
        observations.append(refreshed)
        state, current = sequence.endpoint, refreshed
        if current["pressure"][TARGET_NODE] not in levels:
            stop = "first_node1_pressure_outside_two_levels"

    sequence = itinerary = None
    membership = None
    net_prefixes = ()
    local = (Fraction(0),) * 6
    if steps:
        count = len(steps)
        sequence = observe_nodal_remainder_sequence(
            initial=initial,
            timesteps=(STEP,) * count,
            capacities=(CAPACITY,) * count,
            pressures=tuple(pressures),
        )
        if sequence.steps != tuple(steps) or sequence.endpoint != state:
            raise RuntimeError(
                "the complete prefix differs from its cell-boundary replays"
            )
        local = sequence.prefixes[-1].cumulative_nodal_area
        net_prefixes = _net_prefixes(baseline, incoming, sequence)
        itinerary = derive_nodal_remainder_itinerary(
            epi_states=(initial.epi,) + tuple(step.after.epi for step in steps),
            timesteps=(STEP,) * count,
            capacities=(CAPACITY,) * count,
            pressures=tuple(pressures),
            epi_lower=LOCAL_LOWER,
            epi_upper=LOCAL_UPPER,
        )
        membership = _belongs(initial, itinerary)
        if not itinerary.feasible or not all(membership):
            raise RuntimeError(
                "inverse itinerary constraints lost the actual inherited carry"
            )
    net = tuple(a + b for a, b in zip(incoming, local, strict=True))
    novel = (
        current["pressure"][TARGET_NODE]
        if current["pressure"][TARGET_NODE] not in levels
        else None
    )
    level_counts = tuple(
        sum(step.pressure[TARGET_NODE] == value for step in steps) for value in levels
    )
    if sum(level_counts) != len(steps):
        raise RuntimeError(
            "a novel pressure level was integrated after the stopping condition"
        )
    first_cross = next(
        (row["ordinal"] for row in net_prefixes if row["node1_first_side_restored"]),
        None,
    )
    first_zero = next(
        (
            row["ordinal"]
            for row in net_prefixes
            if TARGET_NODE in row["exact_zero_nodes"]
        ),
        None,
    )
    return {
        "source": {
            "location": "B29.continuation.endpoint",
            "inherited_state": asdict(initial),
            "retained_B29_report_replayed": True,
            "B27_area_reference_state": asdict(baseline),
            "incoming_B27_nodal_area": incoming,
            "phase": reference.source.phase,
            "pressure_generated_by_shared_cpu_kernel": True,
        },
        "analysis_limits": {
            "boundary_budget": boundary_budget,
            "step_budget": step_budget,
            "maximum_boundaries": MAX_BOUNDARIES,
            "maximum_steps": MAX_STEPS,
        },
        "stop_reason": stop,
        "censor": censor,
        "boundaries": tuple(boundaries),
        "pressure_levels": {
            "inherited_levels": levels,
            "executed_step_counts": level_counts,
            "captured_novel_level": novel,
            "novel_level_steps_integrated": 0,
            "inherited_return_arithmetic": _return_record(levels),
            "captured_return_arithmetic": (
                _return_record(levels + (novel,)) if novel is not None else None
            ),
            "scope": (
                "The modular condition is necessary for an exact scalar nodal-area return using only the supplied "
                "finite levels at one fixed positive timestep/capacity. A captured level is not an executed step; "
                "the supplied set is not asserted to contain all future generated pressures."
            ),
        },
        "continuation": {
            "local_area_reference": "B29.continuation.endpoint",
            "net_area_reference": "B27.continuation.endpoint",
            "step_count": len(steps),
            "boundary_count": len(boundaries),
            "steps": tuple(map(asdict, steps)),
            "prefix_balances": (
                _prefix_summary(sequence) if sequence is not None else ()
            ),
            "net_prefixes": net_prefixes,
            "endpoint": asdict(state),
            "refreshed_pressure": current,
            "local_nodal_area": local,
            "B27_nodal_area": net,
            "local_mean_nodal_area": sum(local, Fraction(0)) / 6,
            "B27_mean_nodal_area": sum(net, Fraction(0)) / 6,
            "gradient_balance": _gradient_balance(
                initial, state, local, reference.epi_quantum
            ),
            "displayed_matches_B27_reference": tuple(
                a == b for a, b in zip(baseline.epi, state.epi, strict=True)
            ),
            "reconstructed_matches_B27_reference": tuple(value == 0 for value in net),
            "itinerary": (
                _itinerary_record(itinerary) if itinerary is not None else None
            ),
            "supplied_initial_carry_coordinate_membership": membership,
            "supplied_initial_carry_feasible": (
                all(membership) if membership is not None else None
            ),
        },
        "first_observed_node1_nonnegative_area_step": first_cross,
        "first_observed_node1_exact_zero_step": first_zero,
        "node1_nodal_area_compensated": net[TARGET_NODE] == 0,
        "total_nodal_area_compensated": not any(net),
        "mean_nodal_area_compensated": sum(net, Fraction(0)) == 0,
        "finite_class_drift": _finite_class(observations),
        "runtime_executed": False,
        "new_graph_trajectories": 0,
        "live_provenance_certified": False,
        "original_tail_reachability_certified": False,
        "future_compensation_cycle_certified": False,
        "positive_band_exit_certified": False,
        "scope": (
            "Resource-bounded conditional C6 continuation preserving actual carry and default pressure generation. "
            "The first novel node-1 pressure is captured but not integrated. Signed area can cross zero without "
            "ever equaling zero; visible revisits do not reset carry. The observed finite class has its own "
            "conditional residence bound, not an invariant-class, complete-runtime or whole-band claim."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "artifacts/research/c6_winding_pressure_repayment.json",
    )
    parser.add_argument(
        "--parent-input",
        type=Path,
        default=ROOT / "artifacts/research/c6_winding_pressure_sign.json",
    )
    parser.add_argument(
        "--previous-input",
        type=Path,
        default=ROOT / "artifacts/research/c6_winding_carry_itinerary.json",
    )
    parser.add_argument(
        "--source-input",
        type=Path,
        default=ROOT / "artifacts/research/c6_winding_pressure_lattice.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/research/c6_winding_pressure_levels.json",
    )
    args = parser.parse_args()
    paths = (args.input, args.parent_input, args.previous_input, args.source_input)
    if args.output.resolve() in tuple(path.resolve() for path in paths):
        raise ValueError("derived output must not overwrite any historical input")
    raw = tuple(path.read_bytes() for path in paths)
    parent, previous, ancestor, source_parent = producers = tuple(
        json.loads(value) for value in raw
    )
    hashes = tuple(hashlib.sha256(value).hexdigest() for value in raw)
    edges = (
        (parent, "input_evidence", 1),
        (parent, "parent_input_evidence", 2),
        (parent, "source_input_evidence", 3),
        (previous, "input_evidence", 2),
        (previous, "source_input_evidence", 3),
        (ancestor, "input_evidence", 3),
    )
    if any(producer[key]["sha256"] != hashes[index] for producer, key, index in edges):
        raise ValueError(
            "the retained lineage hashes do not identify the supplied historical bytes"
        )
    provenance = current_git_source_provenance(ROOT, SOURCE_SCOPE)
    report = analyze_c6_winding_pressure_levels(*producers)
    sha, dirty, digest = provenance
    manifest = CoreExperimentManifest(
        claim_id="O3.a-C6-carried-finite-pressure-levels",
        git_sha=sha,
        source_dirty=dirty,
        dirty_source_hash=digest,
        versions={"python": platform.python_version(), "numpy": np.__version__},
        graph_construction="Verified B29 carried endpoint on the inherited closed C6 phase slice",
        capacity_specification="Unit C6 support/capacity, default pressure weights and EPI slab [3/8,5/8]",
        solver="Shared carried cell-exit prefixes; first novel node-1 pressure stops execution before acting",
        result_status=ClaimStatus.DERIVED,
        telemetry=(
            "captured versus executed pressure levels",
            "finite-level return modulus",
            "signed nodal area overshoot",
            "deduplicated finite-class pressure drift",
            "exact inverse itinerary",
        ),
        controls=(
            "at most eight new boundaries and 256 steps",
            "actual carry retained",
            "no novel-level step",
        ),
        artifacts=(str(args.output),),
    )
    manifest.validate_for_admission()
    if (
        current_git_source_provenance(ROOT, SOURCE_SCOPE) != provenance
        or tuple(path.read_bytes() for path in paths) != raw
    ):
        raise RuntimeError(
            "analysis source or historical input changed during the finite-level audit"
        )
    report.update(manifest=manifest.to_dict(), source_scope=SOURCE_SCOPE)
    for key, path, digest, producer in zip(
        (
            "input_evidence",
            "parent_input_evidence",
            "previous_input_evidence",
            "source_input_evidence",
        ),
        paths,
        hashes,
        producers,
        strict=True,
    ):
        report[key] = {
            "path": str(path),
            "sha256": digest,
            "producer_manifest": producer["manifest"],
            "producer_source_scope": producer["source_scope"],
        }
    report["input_evidence"]["producer_input_evidence"] = parent["input_evidence"]
    report["input_evidence"]["producer_parent_input_evidence"] = parent[
        "parent_input_evidence"
    ]
    report["input_evidence"]["producer_source_input_evidence"] = parent[
        "source_input_evidence"
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_payload(report), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote bounded C6 pressure-level audit to {args.output}")


if __name__ == "__main__":
    main()
