"""A bounded carried audit distinguishes local-stencil exit from pressure sign exit.

The B30 endpoint supplies actual carry and a positive node-4 pressure. At
most eight new cell boundaries and 256 steps are available. The first
displayed stencil change is retained independently of the sign stopping rule.
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
from benchmarks.c6_winding_pressure_levels import (
    analyze_c6_winding_pressure_levels,
)  # noqa: E402
from benchmarks.c6_winding_pressure_repayment import _net_prefixes  # noqa: E402
from benchmarks.c6_winding_pressure_sign import (  # noqa: E402
    CAPACITY,
    MAX_BOUNDARIES,
    MAX_STEPS,
    STEP,
    _budget,
    _canonical,
    _gradient_balance,
)
from benchmarks.c6_winding_rounding_cells import _represented_scalar  # noqa: E402
from benchmarks.thol_pressure_feedback import _payload  # noqa: E402
from tnfr.mathematics.unified_numerical import np  # noqa: E402
from tnfr.physics.c6_pressure_lattice import (  # noqa: E402
    derive_c6_pressure_lattice,
    observe_c6_frozen_pressure_stencil,
    observe_c6_pressure_lattice,
)
from tnfr.physics.nodal_remainder import (  # noqa: E402
    derive_nodal_remainder_cell_horizon,
    derive_nodal_remainder_itinerary,
    observe_nodal_remainder_cell_exit,
    observe_nodal_remainder_sequence,
)
from tnfr.research.claims import ClaimStatus  # noqa: E402
from tnfr.research.core_manifests import (
    CoreExperimentManifest,
    current_git_source_provenance,
)  # noqa: E402

TARGET_NODE = 4
STENCIL = (3, 4, 5)
EVIDENCE_KEYS = (
    "input_evidence",
    "parent_input_evidence",
    "previous_input_evidence",
    "source_input_evidence",
)


def _verified_source(parent, previous, ancestor, earlier, source_parent):
    manifest = CoreExperimentManifest(**parent["manifest"])
    manifest.validate_for_admission()
    if (
        manifest.claim_id != "O3.a-C6-carried-finite-pressure-levels"
        or tuple(parent["source_scope"]) != SOURCE_SCOPE
    ):
        raise ValueError("input must be the retained B30 finite-pressure-level report")
    producers = (previous, ancestor, earlier, source_parent)
    for key, producer in zip(EVIDENCE_KEYS, producers, strict=True):
        evidence = parent[key]
        if _canonical(evidence["producer_manifest"]) != _canonical(
            producer["manifest"]
        ) or tuple(evidence["producer_source_scope"]) != tuple(
            producer["source_scope"]
        ):
            raise ValueError(
                "supplied B29/B28/B27/B26 sources differ from the B30 producer lineage"
            )
    for retained, key in (
        ("producer_input_evidence", "input_evidence"),
        ("producer_parent_input_evidence", "parent_input_evidence"),
        ("producer_source_input_evidence", "source_input_evidence"),
    ):
        if _canonical(parent["input_evidence"][retained]) != _canonical(previous[key]):
            raise ValueError(
                "the B30 lineage differs from the supplied B29 historical input chain"
            )
    rebuilt = analyze_c6_winding_pressure_levels(*producers)
    if any(
        key not in parent or _canonical(parent[key]) != _canonical(value)
        for key, value in rebuilt.items()
    ):
        raise ValueError(
            "the retained B30 report differs from its complete bounded shared-owner replay"
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
            "incoming area differs from the actual reconstructed change since the B27 endpoint"
        )
    source = source_parent["lattice_reference"]["source"]
    reference = derive_c6_pressure_lattice(
        phase=tuple(rebuilt["source"]["phase"]),
        epi_weight=_represented_scalar(source["epi_weight"]),
        phase_weight=_represented_scalar(source["phase_weight"]),
        epi_lower=LOCAL_LOWER,
        epi_upper=LOCAL_UPPER,
    )
    observation = rebuilt["continuation"]["refreshed_pressure"]
    if observation["pressure"][TARGET_NODE] <= 0:
        raise ValueError(
            "the inherited B30 endpoint must begin with positive node-4 pressure"
        )
    return reference, initial, baseline, incoming, observation


def _stencil_record(reference, state, observation):
    result = observe_c6_frozen_pressure_stencil(
        reference, state=state, node=TARGET_NODE, timestep=STEP
    )
    if _canonical(_record(result.initial_observation)) != _canonical(observation):
        raise RuntimeError(
            "the frozen-stencil owner disagrees with the freshly generated endpoint pressure"
        )
    data = asdict(result)
    data.pop("reference")
    data["initial_observation"] = _record(result.initial_observation)
    data["graph_provenance_certified"] = result.graph_provenance_certified
    data["sign_hit_certified"] = result.sign_hit_certified
    data["positive_band_exit_certified"] = result.positive_band_exit_certified
    return data


def _nonpositive_cut(reference, observation):
    current = observation["gradient_indices"][TARGET_NODE]
    cut = reference.rows[TARGET_NODE].nonpositive_max_index
    return {
        "node": TARGET_NODE,
        "current_gradient_index": current,
        "nonpositive_max_index": cut,
        "necessary_gradient_change_upper_bound": cut - current,
        "future_feasible_transition_certified": False,
        "scope": "Necessary integer-gradient condition for nonpositive pressure; no future path to the cut is supplied",
    }


def analyze_c6_winding_pressure_stencil(
    parent,
    previous,
    ancestor,
    earlier,
    source_parent,
    *,
    boundary_budget=MAX_BOUNDARIES,
    step_budget=MAX_STEPS,
):
    """Preserve actual carry and censor at fixed resources if no pressure sign exit occurs."""
    boundary_budget = _budget(boundary_budget, MAX_BOUNDARIES, "boundary_budget")
    step_budget = _budget(step_budget, MAX_STEPS, "step_budget")
    reference, initial, baseline, incoming, observation = _verified_source(
        parent, previous, ancestor, earlier, source_parent
    )
    initial_stencil = _stencil_record(reference, initial, observation)
    state, current = initial, observation
    boundaries, steps, pressures = [], [], []
    first_stencil_change = None
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
        observed = observe_nodal_remainder_cell_exit(
            state=state,
            timestep=STEP,
            capacity=CAPACITY,
            pressure=pressure,
            step_budget=remaining,
        )
        if observed.horizon != horizon:
            raise RuntimeError(
                "the preflight and shared carried cell-exit horizons differ"
            )
        sequence = observed.sequence
        refreshed = _record(
            observe_c6_pressure_lattice(reference, epi=sequence.endpoint.epi)
        )
        changed = tuple(i for i in STENCIL if state.epi[i] != sequence.endpoint.epi[i])
        if first_stencil_change is None and changed:
            first_stencil_change = {
                "step": len(steps) + count,
                "boundary": len(boundaries) + 1,
                "changed_nodes": changed,
                "before": asdict(state),
                "after": asdict(sequence.endpoint),
                "pressure_before": current["pressure"][TARGET_NODE],
                "pressure_after": refreshed["pressure"][TARGET_NODE],
            }
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
                "stencil_before": tuple(state.epi[i] for i in STENCIL),
                "stencil_after": tuple(sequence.endpoint.epi[i] for i in STENCIL),
                "changed_stencil_nodes": changed,
                "node4_gradient_before": current["gradient_indices"][TARGET_NODE],
                "node4_gradient_after": refreshed["gradient_indices"][TARGET_NODE],
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
        state, current = sequence.endpoint, refreshed
        if current["pressure"][TARGET_NODE] <= 0:
            stop = "first_nonpositive_node4_pressure"

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
                "the full carried prefix differs from its cell-boundary replays"
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
            raise RuntimeError("the inverse itinerary lost the inherited actual carry")
    endpoint_stencil = _stencil_record(reference, state, current)
    net = tuple(a + b for a, b in zip(incoming, local, strict=True))
    return {
        "source": {
            "location": "B30.continuation.endpoint",
            "inherited_state": asdict(initial),
            "retained_B30_report_replayed": True,
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
        "target_node": TARGET_NODE,
        "stencil_nodes": STENCIL,
        "first_stencil_change": first_stencil_change,
        "initial_frozen_stencil": initial_stencil,
        "endpoint_frozen_stencil": endpoint_stencil,
        "initial_nonpositive_cut": _nonpositive_cut(reference, observation),
        "endpoint_nonpositive_cut": _nonpositive_cut(reference, current),
        "node4_nonpositive_pressure_observed": current["pressure"][TARGET_NODE] <= 0,
        "nonpositive_node4_pressure_steps_integrated": sum(
            step.pressure[TARGET_NODE] <= 0 for step in steps
        ),
        "continuation": {
            "local_area_reference": "B30.continuation.endpoint",
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
        "total_nodal_area_compensated": not any(net),
        "mean_nodal_area_compensated": sum(net, Fraction(0)) == 0,
        "runtime_executed": False,
        "new_graph_trajectories": 0,
        "live_provenance_certified": False,
        "original_tail_reachability_certified": False,
        "future_pressure_sign_exit_certified": False,
        "future_compensation_cycle_certified": False,
        "positive_band_exit_certified": False,
        "scope": (
            "Bounded conditional C6 carried continuation. The first displayed local-stencil change is recorded "
            "separately from a nonpositive node-4 pressure. Frozen-stencil horizons cease to predict the actual "
            "source once a stencil value changes. Resource censoring does not refute the conditional theorem. "
            "Integer cut distances are necessary conditions, not feasible future paths or pressure extrapolations."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "artifacts/research/c6_winding_pressure_levels.json",
    )
    parser.add_argument(
        "--parent-input",
        type=Path,
        default=ROOT / "artifacts/research/c6_winding_pressure_repayment.json",
    )
    parser.add_argument(
        "--previous-input",
        type=Path,
        default=ROOT / "artifacts/research/c6_winding_pressure_sign.json",
    )
    parser.add_argument(
        "--earlier-input",
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
        default=ROOT / "artifacts/research/c6_winding_pressure_stencil.json",
    )
    args = parser.parse_args()
    paths = (
        args.input,
        args.parent_input,
        args.previous_input,
        args.earlier_input,
        args.source_input,
    )
    if args.output.resolve() in tuple(path.resolve() for path in paths):
        raise ValueError("derived output must not overwrite any historical input")
    raw = tuple(path.read_bytes() for path in paths)
    producers = tuple(json.loads(value) for value in raw)
    hashes = tuple(hashlib.sha256(value).hexdigest() for value in raw)
    links = (
        (
            ("input_evidence", 1),
            ("parent_input_evidence", 2),
            ("previous_input_evidence", 3),
            ("source_input_evidence", 4),
        ),
        (
            ("input_evidence", 2),
            ("parent_input_evidence", 3),
            ("source_input_evidence", 4),
        ),
        (("input_evidence", 3), ("source_input_evidence", 4)),
        (("input_evidence", 4),),
    )
    if any(
        producer[key]["sha256"] != hashes[index]
        for producer, edges in zip(producers[:-1], links, strict=True)
        for key, index in edges
    ):
        raise ValueError(
            "the retained lineage hashes do not identify the supplied historical bytes"
        )
    provenance = current_git_source_provenance(ROOT, SOURCE_SCOPE)
    report = analyze_c6_winding_pressure_stencil(*producers)
    sha, dirty, digest = provenance
    manifest = CoreExperimentManifest(
        claim_id="O3.a-C6-carried-pressure-stencil",
        git_sha=sha,
        source_dirty=dirty,
        dirty_source_hash=digest,
        versions={"python": platform.python_version(), "numpy": np.__version__},
        graph_construction="Verified B30 carried endpoint with node-4 local pressure stencil and inherited B27 area baseline",
        capacity_specification="Unit C6 capacity/support, default pressure weights and EPI slab [3/8,5/8]",
        solver="Shared exact carried cell-exit prefixes; first nonpositive node-4 pressure or fixed resource censor",
        result_status=ClaimStatus.DERIVED,
        telemetry=(
            "first actual stencil change",
            "frozen-stencil coordinate and uniform horizons",
            "necessary integer pressure-sign cut",
            "local and inherited six-node budgets",
        ),
        controls=(
            "at most eight new boundaries and 256 steps",
            "no carry reset",
            "no post-sign-hit steps",
        ),
        artifacts=(str(args.output),),
    )
    manifest.validate_for_admission()
    if (
        current_git_source_provenance(ROOT, SOURCE_SCOPE) != provenance
        or tuple(path.read_bytes() for path in paths) != raw
    ):
        raise RuntimeError(
            "analysis source or historical input changed during the bounded stencil audit"
        )
    report.update(manifest=manifest.to_dict(), source_scope=SOURCE_SCOPE)
    for key, path, digest, producer in zip(
        (
            "input_evidence",
            "parent_input_evidence",
            "previous_input_evidence",
            "earlier_input_evidence",
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
    for key in EVIDENCE_KEYS:
        report["input_evidence"]["producer_" + key] = producers[0][key]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_payload(report), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote bounded C6 pressure-stencil audit to {args.output}")


if __name__ == "__main__":
    main()
