"""Test temporal compatibility of the static C6 pressure-balance family.

B41 excludes trapping by complete carry cells in the bounded-gradient class.
B42 gives an exact uniform escape deadline for B38's seven complete cells.
B41 supplies an outward hypothetical carry; B42 bounds every legal carry in
its seven-cell family. Neither continues saved B40.
The optional B43 branch starts a separate graph-owned original preparation;
it never imports the detached B40 remainder into that live graph.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from benchmarks import c6_winding_coupled_passage as previous  # noqa: E402
from benchmarks.c6_winding_carry_itinerary import SOURCE_SCOPE, _state  # noqa: E402
from benchmarks.c6_winding_pressure_sign import STEP, _canonical  # noqa: E402
from benchmarks.thol_pressure_feedback import _payload  # noqa: E402
from tnfr.mathematics.unified_numerical import np  # noqa: E402
from tnfr.physics.c6_carried_cell_escape import (  # noqa: E402
    derive_c6_carried_complete_cell_obstruction,
    observe_c6_carried_complete_cell_escape,
)
from tnfr.physics.c6_carried_cell_graph import (
    derive_c6_carried_cell_graph,
)  # noqa: E402
from tnfr.physics.c6_carried_closure import derive_c6_carried_closure  # noqa: E402
from tnfr.research.claims import ClaimStatus  # noqa: E402
from tnfr.research.core_manifests import (
    CoreExperimentManifest,
    current_git_source_provenance,
)  # noqa: E402

INPUT_NAMES = ("c6_winding_coupled_passage.json",) + previous.INPUT_NAMES


def _check_lineage(producers, hashes=None):
    if len(producers) != 8 or hashes is not None and len(hashes) != 8:
        raise ValueError(
            "the temporal audit requires B40 and its seven historical inputs"
        )
    parent, *ancestors = producers
    previous._check_lineage(tuple(ancestors), None if hashes is None else hashes[1:])
    evidence = parent["input_evidence"]
    if len(evidence) != len(ancestors):
        raise ValueError(
            "the retained B40 report must identify seven original producers"
        )
    for index, (entry, producer) in enumerate(zip(evidence, ancestors, strict=True), 1):
        if _canonical(entry["producer_manifest"]) != _canonical(
            producer["manifest"]
        ) or tuple(entry["producer_source_scope"]) != tuple(producer["source_scope"]):
            raise ValueError(
                "the temporal lineage metadata differs from its historical producer"
            )
        if hashes is not None and entry["sha256"] != hashes[index]:
            raise ValueError(
                "the temporal lineage hashes differ from the historical input bytes"
            )


def _verified_source(producers):
    _check_lineage(producers)
    parent, *ancestors = producers
    manifest = CoreExperimentManifest(**parent["manifest"])
    manifest.validate_for_admission()
    if (
        manifest.claim_id != "O3.a-C6-coupled-pressure-passage"
        or tuple(parent["source_scope"]) != SOURCE_SCOPE
    ):
        raise ValueError("input must be the retained B37-B40 pressure-passage report")
    rebuilt = previous.analyze_c6_winding_coupled_passage(*ancestors)
    if any(
        key not in parent or _canonical(parent[key]) != _canonical(value)
        for key, value in rebuilt.items()
    ):
        raise ValueError(
            "the retained B40 report differs from its complete shared-owner replay"
        )
    # Reuse the already checked primitive source from the B36 report. The
    # B40 endpoint has its own carry; the static comparison points keep theirs.
    source = ancestors[0]["B32_profile"]["reference"]["lattice"]["source"]
    from benchmarks.c6_winding_rounding_cells import _represented_scalar
    from tnfr.physics.c6_carried_profile import derive_c6_carried_profile
    from tnfr.physics.c6_pressure_lattice import derive_c6_pressure_lattice

    lattice = derive_c6_pressure_lattice(
        phase=tuple(source["phase"]),
        epi_weight=_represented_scalar(source["epi_weight"]),
        phase_weight=_represented_scalar(source["phase_weight"]),
        epi_lower=_represented_scalar(source["epi_lower"]),
        epi_upper=_represented_scalar(source["epi_upper"]),
    )
    state = _state(parent["B40_continuation"]["endpoint"])
    points = tuple(
        tuple(point["state"]["epi"])
        for point in rebuilt["B38_static_pressure_balance"]["points"]
    )
    return derive_c6_carried_profile(lattice), state, points


def _runtime_bridge():
    from benchmarks.c6_winding_phase_tail_runtime import (
        run_c6_phase_tail_runtime_bridge,
    )

    return run_c6_phase_tail_runtime_bridge()


def analyze_c6_winding_temporal_compatibility(*producers, include_runtime=False):
    """Derive complete-cell obstructions without advancing the saved state."""
    if type(include_runtime) is not bool:
        raise TypeError("include_runtime must be boolean")
    profile, current, points = _verified_source(producers)
    closure = derive_c6_carried_closure(profile, state=current, timestep=STEP)
    obstruction = derive_c6_carried_complete_cell_obstruction(closure)
    family = points + (() if current.epi in points else (current.epi,))
    witness = observe_c6_carried_complete_cell_escape(obstruction, epi_states=family)
    graph = derive_c6_carried_cell_graph(
        profile.lattice, epi_states=points, timestep=STEP
    )
    obstruction_record = asdict(obstruction)
    obstruction_record.pop("closure")
    obstruction_record.update(
        finite_complete_cell_union_invariance_excluded=obstruction.finite_complete_cell_union_invariance_excluded,
        correlated_carry_region_excluded=False,
        saved_trajectory_escape_certified=False,
        future_runtime_certified=False,
    )
    witness_record = asdict(witness)
    witness_record.pop("obstruction")
    witness_record.update(
        family_invariance_excluded=True, saved_trajectory_escape_certified=False
    )
    graph_record = asdict(graph)
    graph_record.pop("reference")
    graph_record.update(
        finite_family_escape_certified=graph.finite_family_escape_certified,
        carried_cycle_in_family_excluded=graph.carried_cycle_in_family_excluded,
        whole_band_exit_certified=False,
        future_runtime_certified=False,
        reachable_from_saved_state_certified=False,
    )
    runtime = _runtime_bridge() if include_runtime else None
    return {
        "source": {
            "source_report_replayed": True,
            "saved_B40_state": asdict(current),
            "phase": profile.lattice.source.phase,
            "B38_visible_rows": points,
        },
        "B41_complete_cell_obstruction": obstruction_record,
        "B41_hypothetical_cell_witness": witness_record,
        "B42_finite_cell_graph": graph_record,
        "B43_original_phase_tail_runtime": runtime,
        "new_saved_trajectory_steps": 0,
        "static_cell_witness_is_saved_continuation": False,
        "infinite_band_invariance_certified": False,
        "full_runtime_stability_certified": False,
        "B40_origin_reachability_certified": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-dir", type=Path, default=ROOT / "artifacts/research")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/research/c6_winding_temporal_compatibility.json",
    )
    parser.add_argument(
        "--skip-runtime",
        action="store_true",
        help="Produce only the static B41-B42 certificates",
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
    report = analyze_c6_winding_temporal_compatibility(
        *producers, include_runtime=not args.skip_runtime
    )
    sha, dirty, digest = provenance
    manifest = CoreExperimentManifest(
        claim_id="O3.a-C6-temporal-cell-compatibility",
        git_sha=sha,
        source_dirty=dirty,
        dirty_source_hash=digest,
        versions={"python": platform.python_version(), "numpy": np.__version__},
        graph_construction="Revalidated B40 endpoint and B38 static pressure family; separately declared original live preparation",
        capacity_specification="Fixed numerical C6 unit support/capacity; live branch records terminal Silence capacity change",
        solver="Exact carried-cell inverse itineraries, held-cell horizons and complete-cell obstruction; optional sealed live execution",
        result_status=ClaimStatus.DERIVED,
        telemetry=(
            "complete-carry-cell obstruction",
            "all pairwise feasible edges",
            "finite family escape deadline",
            "optional live phase trace",
        ),
        controls=(
            "no continuation or reset of saved B40 carry",
            "existential edges not assumed composable",
            "family escape distinct from full-band escape",
            "hypothetical and graph-owned states remain separate",
        ),
        artifacts=(str(args.output),),
    )
    manifest.validate_for_admission()
    if (
        current_git_source_provenance(ROOT, SOURCE_SCOPE) != provenance
        or tuple(path.read_bytes() for path in paths) != raw
    ):
        raise RuntimeError(
            "analysis source or historical input changed during the temporal audit"
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
    print(f"Wrote C6 temporal-compatibility report to {args.output}")


if __name__ == "__main__":
    main()
