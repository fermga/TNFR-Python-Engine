"""Resolve the retained node-4 cut using a derived finite passage budget.

B37 closes spatial/rounding feedback. B38 supplies an exact static convex
pressure balance, refuting a universal strict linear separator on that class.
B39 proves a finite sign passage. B40 follows the actual incoming carry only
until its first nonpositive node-4 pressure, within the theorem's budget.
The fixed TNFR source and nodal update are inherited; no coefficients, carry
or pressure are adjusted. A graph word and indefinite trapping remain separate.
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

from benchmarks import c6_winding_coupled_budget as previous  # noqa: E402
from benchmarks.c6_winding_carry_itinerary import SOURCE_SCOPE, _state  # noqa: E402
from benchmarks.c6_winding_pressure_lattice import _record  # noqa: E402
from benchmarks.c6_winding_pressure_sign import CAPACITY, STEP, _canonical  # noqa: E402
from benchmarks.thol_pressure_feedback import _payload  # noqa: E402
from tnfr.dynamics._euler_kernel import NodalRemainderState  # noqa: E402
from tnfr.mathematics.unified_numerical import np  # noqa: E402
from tnfr.physics.c6_carried_balance import (
    derive_c6_carried_pressure_balance,
)  # noqa: E402
from tnfr.physics.c6_carried_closure import derive_c6_carried_closure  # noqa: E402
from tnfr.physics.c6_carried_passage import (
    derive_c6_carried_positive_pressure_passage,
)  # noqa: E402
from tnfr.physics.c6_carried_profile import derive_c6_carried_profile  # noqa: E402
from tnfr.physics.c6_pressure_lattice import (
    derive_c6_pressure_lattice,
    observe_c6_pressure_lattice,
)  # noqa: E402
from tnfr.physics.forced_support import observe_forced_support_pattern  # noqa: E402
from tnfr.physics.nodal_remainder import (  # noqa: E402
    observe_nodal_remainder_cell_exit,
    observe_nodal_remainder_sequence,
)
from tnfr.physics.nodal_remainder_pressure import (
    observe_nodal_remainder_cycle_gradient,
)  # noqa: E402
from tnfr.research.claims import ClaimStatus  # noqa: E402
from tnfr.research.core_manifests import (
    CoreExperimentManifest,
    current_git_source_provenance,
)  # noqa: E402

INPUT_NAMES = ("c6_winding_coupled_budget.json",) + previous.INPUT_NAMES
# These static represented points are a retained algebraic countercertificate,
# not preparations, probabilities, a schedule or an optimization at execution.
BALANCE_OFFSETS = (
    (-6, -1, 2, 2, 8, -7),
    (-5, 2, 4, -2, 6, -7),
    (-4, -2, 0, 2, 6, -4),
    (-4, -1, 2, -2, 8, -5),
    (-3, 0, 2, -2, 8, -7),
    (-2, -1, 4, -2, 6, -7),
    (-2, 2, 0, -2, 6, -6),
)


def _check_lineage(producers, hashes=None):
    if len(producers) != 7 or hashes is not None and len(hashes) != 7:
        raise ValueError(
            "the passage audit requires B36 and its six retained input reports"
        )
    parent, *ancestors = producers
    previous._check_lineage(tuple(ancestors), None if hashes is None else hashes[1:])
    for index, (key, producer) in enumerate(
        zip(previous.INPUT_KEYS, ancestors, strict=True), 1
    ):
        evidence = parent[key]
        if _canonical(evidence["producer_manifest"]) != _canonical(
            producer["manifest"]
        ) or tuple(evidence["producer_source_scope"]) != tuple(
            producer["source_scope"]
        ):
            raise ValueError(
                "the B36 lineage metadata differs from the supplied historical source"
            )
        if hashes is not None and evidence["sha256"] != hashes[index]:
            raise ValueError(
                "the B36 lineage hashes do not identify the retained historical bytes"
            )


def _verified_source(producers):
    _check_lineage(producers)
    parent, *ancestors = producers
    manifest = CoreExperimentManifest(**parent["manifest"])
    manifest.validate_for_admission()
    if (
        manifest.claim_id != "O3.a-C6-coupled-carried-budget"
        or tuple(parent["source_scope"]) != SOURCE_SCOPE
    ):
        raise ValueError("input must be the retained B32-B36 coupled-budget report")
    rebuilt = previous.analyze_c6_winding_coupled_budget(*ancestors)
    if any(
        key not in parent or _canonical(parent[key]) != _canonical(value)
        for key, value in rebuilt.items()
    ):
        raise ValueError(
            "the retained B36 report differs from its complete shared-owner replay"
        )
    # This is exactly the same already-revalidated pressure lattice; no new
    # initial preparation or phase search enters the continuation.
    source = rebuilt["B32_profile"]["reference"]["lattice"]["source"]
    lattice = derive_c6_pressure_lattice(
        phase=source["phase"],
        epi_weight=float(source["epi_weight"]),
        phase_weight=float(source["phase_weight"]),
        epi_lower=source["epi_lower"],
        epi_upper=source["epi_upper"],
    )
    endpoint = _state(producers[0]["source"]["endpoint_state"])
    return (
        derive_c6_carried_profile(lattice),
        endpoint,
        _state(ancestors[0]["source"]["B27_area_reference_state"]),
    )


def _compact_passage(passage):
    data = asdict(passage)
    data.pop("tube")
    data["band_horizon"].pop("tube")
    data["positive_sector"].pop("reference")
    data.update(
        graph_provenance_certified=False,
        future_runtime_certified=False,
        infinite_trapping_certified=False,
    )
    return data


def _advance_to_cut(profile, initial, closure, passage, baseline):
    if not passage.sign_hit_certified or passage.latest_pressure_index is None:
        raise ValueError(
            "a proved finite pressure passage is required before continuation"
        )
    state = initial
    current = observe_c6_pressure_lattice(profile.lattice, epi=state.epi)
    boundaries, steps, pressures = [], [], []
    source_area = rounding_area = F(0)
    while current.pressure[4] > 0:
        remaining = passage.latest_pressure_index - len(steps)
        if remaining <= 0:
            raise RuntimeError(
                "the actual pressure contradicts the derived first-passage bound"
            )
        observed = observe_nodal_remainder_cell_exit(
            state=state,
            timestep=STEP,
            capacity=CAPACITY,
            pressure=current.pressure,
            step_budget=remaining,
        )
        sequence = observed.sequence
        count = len(sequence.steps)
        refreshed = observe_c6_pressure_lattice(
            profile.lattice, epi=sequence.endpoint.epi
        )
        source_area += count * F(STEP) * current.mean_phase_contribution
        rounding_area += (
            count
            * F(STEP)
            * (current.mean_epi_reduction_error + current.mean_assembly_error)
        )
        for step in sequence.steps:
            pattern = observe_forced_support_pattern(
                profile.forced_balance,
                nodes=tuple(range(6)),
                epi=step.after.exact_epi,
            )
            if pattern.error_variance > closure.energy_bound:
                raise RuntimeError(
                    "an actual carried endpoint escaped its self-consistent spatial envelope"
                )
        boundaries.append(
            {
                "ordinal": len(boundaries) + 1,
                "cumulative_steps": len(steps) + count,
                "step_count": count,
                "initial": asdict(state),
                "endpoint": asdict(sequence.endpoint),
                "source_pressure": _record(current),
                "refreshed_pressure": _record(refreshed),
                "cell_horizon": asdict(observed.horizon),
            }
        )
        steps.extend(sequence.steps)
        pressures.extend((current.pressure,) * count)
        state, current = sequence.endpoint, refreshed
    count = len(steps)
    sequence = observe_nodal_remainder_sequence(
        initial=initial,
        timesteps=(STEP,) * count,
        capacities=(CAPACITY,) * count,
        pressures=tuple(pressures),
    )
    if sequence.steps != tuple(steps) or sequence.endpoint != state:
        raise RuntimeError(
            "the complete shared carried replay differs from the boundary continuation"
        )
    area = tuple(b - a for a, b in zip(initial.exact_epi, state.exact_epi, strict=True))
    mean = sum(area, F(0)) / 6
    if mean != source_area + rounding_area:
        raise RuntimeError(
            "the first-passage signed mean budget lost its exact decomposition"
        )
    gradient = observe_nodal_remainder_cycle_gradient(
        initial=initial,
        endpoint=state,
        nodal_area=area,
        epi_quantum=profile.lattice.epi_quantum,
    )
    incoming = tuple(
        b - a for a, b in zip(baseline.exact_epi, initial.exact_epi, strict=True)
    )
    incoming_mean = sum(incoming, F(0)) / 6
    mean_prefixes = tuple(
        incoming_mean + prefix.mean_nodal_area for prefix in sequence.prefixes
    )
    repayment = next(
        (i for i, value in enumerate(mean_prefixes, 1) if incoming_mean < 0 <= value),
        None,
    )
    net_area = tuple(a + b for a, b in zip(incoming, area, strict=True))
    return {
        "stop_reason": "first_nonpositive_node4_pressure",
        "step_count": count,
        "boundary_count": len(boundaries),
        "initial": asdict(initial),
        "endpoint": asdict(state),
        "boundaries": tuple(boundaries),
        "steps": tuple(asdict(step) for step in steps),
        "refreshed_pressure": _record(current),
        "local_nodal_area": area,
        "local_mean_area": mean,
        "mean_source_area": source_area,
        "mean_rounding_area": rounding_area,
        "mean_carry_feedback_area": F(0),
        "mean_identity_residual": mean - source_area - rounding_area,
        "gradient_balance": asdict(gradient),
        "all_prior_node4_pressures_positive": all(
            pressure[4] > 0 for pressure in pressures
        ),
        "all_centered_energies_within_closure": True,
        "within_proven_pressure_index": count <= passage.latest_pressure_index,
        "B27_area_reference_state": asdict(baseline),
        "B27_incoming_mean_area": incoming_mean,
        "B27_nodal_area": net_area,
        "B27_mean_area": sum(net_area, F(0)) / 6,
        "B27_mean_area_prefixes": mean_prefixes,
        "first_B27_mean_repayment_step": repayment,
        "B27_mean_repayment_exact_zero": repayment is not None
        and mean_prefixes[repayment - 1] == 0,
        "complete_return_observed": any(
            not any(
                a + b
                for a, b in zip(incoming, prefix.cumulative_nodal_area, strict=True)
            )
            for prefix in sequence.prefixes
        ),
    }, state


def analyze_c6_winding_coupled_passage(*producers):
    """Derive the bounds first, then stop at the first certified cut observation."""
    profile, initial, baseline = _verified_source(producers)
    closure = derive_c6_carried_closure(profile, state=initial, timestep=STEP)
    points = tuple(
        tuple(float(F(1, 2) + index * profile.lattice.epi_quantum) for index in row)
        for row in BALANCE_OFFSETS
    )
    common_carry = sum(initial.remainder, F(0)) / 6
    static_states = tuple(
        NodalRemainderState(
            epi, (common_carry,) * 6, initial.epi_lower, initial.epi_upper
        )
        for epi in points
    )
    balance = derive_c6_carried_pressure_balance(closure, states=static_states)
    if not balance.on_origin_mean_slice:
        raise RuntimeError(
            "the static pressure countercertificate must retain the exact inherited reconstructed mean"
        )
    passage = derive_c6_carried_positive_pressure_passage(closure.base_tube, node=4)
    continuation, endpoint = _advance_to_cut(
        profile, initial, closure, passage, baseline
    )
    closure_record = asdict(closure)
    closure_record.pop("base_tube")
    closure_record.update(
        infinite_mean_control_certified=False,
        gradient_reachability_certified=False,
        future_runtime_certified=False,
    )
    balance_record = asdict(balance)
    balance_record.pop("closure", None)
    balance_record.update(
        class_wide_strict_linear_drift_excluded=True,
        temporal_compensation_certified=False,
        bounded_trajectory_certified=False,
        future_runtime_certified=False,
    )
    for point in balance_record.get("points", ()):
        point["observation"].pop("reference", None)
        point["reachable_from_initial_state_certified"] = False
    if continuation["B27_nodal_area"] != tuple(
        b - a for a, b in zip(baseline.exact_epi, endpoint.exact_epi, strict=True)
    ):
        raise RuntimeError(
            "the complete inherited B27 vector area differs from the reconstructed endpoint change"
        )
    return {
        "source": {
            "initial_state": asdict(initial),
            "phase": profile.lattice.source.phase,
            "source_report_replayed": True,
            "origin": "B31.continuation.endpoint (unchanged by B32-B36)",
        },
        "B37_self_consistent_closure": closure_record,
        "B38_static_pressure_balance": balance_record,
        "B39_finite_pressure_passage": _compact_passage(passage),
        "B40_continuation": continuation,
        "runtime_executed": False,
        "new_graph_trajectories": 0,
        "new_conditional_numerical_steps": continuation["step_count"],
        "live_provenance_certified": False,
        "original_tail_reachability_certified": False,
        "infinite_band_invariance_certified": False,
        "full_runtime_stability_certified": False,
        "static_balance_is_temporal_itinerary": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-dir", type=Path, default=ROOT / "artifacts/research")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/research/c6_winding_coupled_passage.json",
    )
    args = parser.parse_args()
    paths = tuple(args.history_dir / name for name in INPUT_NAMES)
    if args.output.resolve() in tuple(path.resolve() for path in paths):
        raise ValueError("the derived output must not overwrite a historical input")
    raw = tuple(path.read_bytes() for path in paths)
    producers = tuple(json.loads(data) for data in raw)
    hashes = tuple(hashlib.sha256(data).hexdigest() for data in raw)
    _check_lineage(producers, hashes)
    provenance = current_git_source_provenance(ROOT, SOURCE_SCOPE)
    report = analyze_c6_winding_coupled_passage(*producers)
    sha, dirty, digest = provenance
    manifest = CoreExperimentManifest(
        claim_id="O3.a-C6-coupled-pressure-passage",
        git_sha=sha,
        source_dirty=dirty,
        dirty_source_hash=digest,
        versions={"python": platform.python_version(), "numpy": np.__version__},
        graph_construction="Retained carried C6 endpoint with fixed closed phase source and complete B36-B26 lineage",
        capacity_specification="Inherited unit C6 support/capacity, existing channel weights, positive slab [3/8,5/8]",
        solver="Shared carried nodal kernel with h=1/16 and pressure refresh; stop at first cut within proved finite budget",
        result_status=ClaimStatus.DERIVED,
        telemetry=(
            "self-consistent spatial closure",
            "exact static pressure convex balance",
            "finite sign passage",
            "signed prefix area",
        ),
        controls=(
            "no parameter changes",
            "incoming carry retained",
            "static convex weights are not a dynamical law",
            "numerical theorem and observed sign hit distinct from full runtime and indefinite trapping",
        ),
        artifacts=(str(args.output),),
    )
    manifest.validate_for_admission()
    if (
        current_git_source_provenance(ROOT, SOURCE_SCOPE) != provenance
        or tuple(path.read_bytes() for path in paths) != raw
    ):
        raise RuntimeError(
            "analysis source or historical input changed during the coupled passage"
        )
    report.update(manifest=manifest.to_dict(), source_scope=SOURCE_SCOPE)
    report["input_evidence"] = tuple(
        {
            "path": str(path),
            "sha256": digest,
            "producer_manifest": producer["manifest"],
            "producer_source_scope": producer["source_scope"],
        }
        for path, digest, producer in zip(paths, hashes, producers, strict=True)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_payload(report), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote coupled C6 pressure-passage report to {args.output}")


if __name__ == "__main__":
    main()
