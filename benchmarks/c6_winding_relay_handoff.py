"""Extend B46's surviving local relay budget across its node-4 boundary.

The exact pressure dependencies permit arbitrary free-node motion inside the
declared band. The finite observation stops at the first held-neighborhood
change, within a previously derived bound; it is not a new live graph word.
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
from benchmarks.c6_winding_carry_itinerary import _state  # noqa: E402
from benchmarks.c6_winding_mean_cylinder import (
    INPUT_NAME as HISTORICAL_NAME,
)  # noqa: E402
from benchmarks.c6_winding_pressure_sign import _canonical  # noqa: E402
from benchmarks.c6_winding_relay_budget import (
    replay_c6_relay_budget_evidence,
)  # noqa: E402
from benchmarks.thol_pressure_feedback import _payload  # noqa: E402
from tnfr.mathematics.unified_numerical import np  # noqa: E402
from tnfr.physics.c6_carried_relay import (  # noqa: E402
    derive_c6_carried_local_relay_budget,
    observe_c6_carried_local_relay_exit,
)
from tnfr.physics.c6_pressure_lattice import (
    _observe_rebuilt_c6_pressure_lattice,
)  # noqa: E402
from tnfr.physics.c6_carried_profile import derive_c6_carried_profile  # noqa: E402
from tnfr.physics.c6_pressure_lattice import derive_c6_pressure_lattice  # noqa: E402
from tnfr.research.claims import ClaimStatus  # noqa: E402
from tnfr.research.core_manifests import (
    CoreExperimentManifest,
    current_git_source_provenance,
)  # noqa: E402

INPUT_NAME = "c6_winding_relay_budget.json"


def _boundary_interaction(anchor, state):
    """Bind the eight boundary rows without assuming independent switches."""
    rows = tuple(
        tuple(
            state.epi[4] if node == 4 and mask & 4 else value
            for node, value in enumerate(anchor.visible_rows[mask & 3])
        )
        for mask in range(8)
    )
    reference = anchor.closure.base_tube.contraction.profile.lattice
    pressures = tuple(
        _observe_rebuilt_c6_pressure_lattice(reference, row).pressure for row in rows
    )
    h = F(anchor.closure.base_tube.contraction.timestep)
    increments = tuple(tuple(h * F(p) for p in row) for row in pressures)
    mixed = tuple(
        increments[6][i] - increments[2][i] - increments[4][i] + increments[0][i]
        for i in range(6)
    )
    node0_mixed = tuple(
        tuple(
            increments[1 | bit][i]
            - increments[1][i]
            - increments[bit][i]
            + increments[0][i]
            for i in range(6)
        )
        for bit in (2, 4)
    )
    triple = tuple(
        sum(
            (
                (-1 if (3 - mask.bit_count()) % 2 else 1) * increments[mask][i]
                for mask in range(8)
            ),
            F(0),
        )
        for i in range(6)
    )
    return {
        "visible_rows": rows,
        "pressure_rows": pressures,
        "exact_increments": increments,
        "node3_node4_mixed_increment": mixed,
        "node0_pair_mixed_increments": node0_mixed,
        "triple_mixed_increment": triple,
        "three_independent_relays_refuted": any(mixed),
        "universal_locality_is_not_inferred_from_eight_samples": True,
    }


def analyze_c6_winding_relay_handoff(parent, *, historical_bytes, step_budget=1024):
    """Replay the source lineage, derive a deadline and observe only its first gate."""
    anchor, state = replay_c6_relay_budget_evidence(
        parent, historical_bytes=historical_bytes
    )
    budget = derive_c6_carried_local_relay_budget(
        anchor, state=state, relay_node=0, budget_node=1
    )
    observed = observe_c6_carried_local_relay_exit(budget, step_budget=step_budget)
    certificate = asdict(budget)
    for field in ("anchor", "closure", "band_horizon"):
        certificate.pop(field)
    band = asdict(budget.band_horizon)
    band.pop("tube")
    passage = asdict(observed)
    passage.pop("budget")
    old_area = tuple(
        F(x) for x in parent["B46_conditional_first_exit"]["total_nodal_area"]
    )
    area = tuple(
        a + b for a, b in zip(old_area, observed.total_nodal_area, strict=True)
    )
    historical = json.loads(historical_bytes)
    original_area = tuple(
        F(x) for x in historical["B43_original_phase_tail_runtime"]["exact_nodal_area"]
    )
    global_area = tuple(a + b for a, b in zip(original_area, area, strict=True))
    return {
        "contract": {
            "question": "Does the node-1 corrected budget survive B46's node-4 boundary?",
            "derivation": "Exact C6 pressure locality and the inherited autonomous node-0 relay strip",
            "stop_condition": "First held-node visible change or band failure within the derived deadline",
            "computational_step_budget": step_budget,
            "budget_is_physical_parameter": False,
            "pressure_projected_or_carry_reset": False,
        },
        "source": {
            "retained_B46_endpoint": asdict(state),
            "historical_nodal_steps_replayed": 356,
            "previous_conditional_steps_replayed": 7,
            "serialized_numerical_chain_verified": True,
            "live_execution_seal_recreated": False,
            "phase": anchor.closure.base_tube.contraction.profile.lattice.source.phase,
            "capacity": (1.0,) * 6,
            "timestep": anchor.closure.base_tube.contraction.timestep,
            "origin_mean": budget.closure.base_tube.initial_mean,
            "origin_energy": budget.closure.base_tube.initial_energy,
            "energy_bound": budget.closure.energy_bound,
        },
        "B47_boundary_pressure_interaction": _boundary_interaction(anchor, state),
        "B47_local_relay_budget": certificate,
        "B47_independent_band_horizon": band,
        "B47_conditional_first_exit": passage,
        "B43_to_B47_conditional_nodal_area": area,
        "original_preparation_plus_conditional_nodal_area": global_area,
        "original_preparation_plus_conditional_mean_area": sum(global_area, F(0)) / 6,
        "conditional_numerical_steps": observed.exit_step,
        "conditional_elapsed_time": F(budget.closure.base_tube.contraction.timestep)
        * observed.exit_step,
        "total_conditional_steps_after_B43": 7 + observed.exit_step,
        "conditional_lineage_from_B46_numerically_verified": True,
        "new_live_graph_steps": 0,
        "whole_band_exit_certified": False,
        "indefinite_trapping_certified": False,
        "future_runtime_certified": False,
    }


def replay_c6_relay_handoff_evidence(parent, *, relay_bytes, historical_bytes):
    """Rebuild the full B43/B46/B47 numeric lineage without recreating a seal.

    The complete B47 payload, input hashes and producer metadata are checked.
    The returned profile and carried endpoint are rebuilt from that replay;
    neither a caller-supplied endpoint nor a cached proof flag is trusted.
    """
    manifest = CoreExperimentManifest(**parent["manifest"])
    manifest.validate_for_admission()
    if (
        manifest.claim_id != "O3.a-C6-carried-local-relay-handoff"
        or tuple(parent["source_scope"]) != SOURCE_SCOPE
    ):
        raise ValueError(
            "the invariant-region study requires the retained B47 handoff report"
        )
    if type(relay_bytes) is not bytes or type(historical_bytes) is not bytes:
        raise TypeError(
            "relay_bytes and historical_bytes must be exact retained input bytes"
        )
    evidence = parent["input_evidence"]
    if (
        hashlib.sha256(relay_bytes).hexdigest() != evidence["sha256"]
        or hashlib.sha256(historical_bytes).hexdigest() != evidence["historical_sha256"]
    ):
        raise ValueError(
            "the retained B46/B43 input hashes differ from B47's declared sources"
        )
    relay = json.loads(relay_bytes)
    if _canonical(evidence["producer_manifest"]) != _canonical(
        relay["manifest"]
    ) or _canonical(evidence["producer_source_scope"]) != _canonical(
        relay["source_scope"]
    ):
        raise ValueError("the retained B46 producer metadata differs from B47's source")
    expected = analyze_c6_winding_relay_handoff(
        relay,
        historical_bytes=historical_bytes,
        step_budget=parent["contract"]["computational_step_budget"],
    )
    payload = {
        key: value
        for key, value in parent.items()
        if key not in ("manifest", "source_scope", "input_evidence")
    }
    if _canonical(payload) != _canonical(expected):
        raise ValueError(
            "the retained B47 result differs from its complete numerical replay"
        )
    endpoint = expected["B47_conditional_first_exit"]["endpoint"]
    if endpoint is None:
        raise ValueError("a band-failed B47 branch has no admitted endpoint")
    state = _state(_payload(endpoint))
    historical = json.loads(historical_bytes)
    weights = dict(
        historical["B43_original_phase_tail_runtime"]["source"]["normalized_weights"]
    )
    reference = derive_c6_pressure_lattice(
        phase=tuple(expected["source"]["phase"]),
        epi_weight=float(F(weights["epi"])),
        phase_weight=float(F(weights["phase"])),
        epi_lower=state.epi_lower,
        epi_upper=state.epi_upper,
    )
    return derive_c6_carried_profile(reference), state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=ROOT / "artifacts/research" / INPUT_NAME
    )
    parser.add_argument(
        "--historical-input",
        type=Path,
        default=ROOT / "artifacts/research" / HISTORICAL_NAME,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/research/c6_winding_relay_handoff.json",
    )
    parser.add_argument("--step-budget", type=int, default=1024)
    args = parser.parse_args()
    if args.output.resolve() in (args.input.resolve(), args.historical_input.resolve()):
        raise ValueError(
            "the derived report must not overwrite either historical input"
        )
    raw, historical_raw = args.input.read_bytes(), args.historical_input.read_bytes()
    parent = json.loads(raw)
    provenance = current_git_source_provenance(ROOT, SOURCE_SCOPE)
    report = analyze_c6_winding_relay_handoff(
        parent, historical_bytes=historical_raw, step_budget=args.step_budget
    )
    sha, dirty, digest = provenance
    manifest = CoreExperimentManifest(
        claim_id="O3.a-C6-carried-local-relay-handoff",
        git_sha=sha,
        source_dirty=dirty,
        dirty_source_hash=digest,
        versions={"python": platform.python_version(), "numpy": np.__version__},
        graph_construction="Fixed unit C6 source; original B43 and conditional B46 evidence replayed",
        capacity_specification="Conditional unit capacity; historical terminal SHA is not undone",
        solver="Canonical refreshed pressure, shared carried nodal kernel and exact local corrected budget",
        result_status=ClaimStatus.DERIVED,
        telemetry=(
            "pressure dependency support",
            "nonzero coupled-switch interaction",
            "signed corrected budget",
            "independent band horizon",
            "first held-node exit",
            "complete nodal area and mean",
        ),
        controls=(
            "all incoming carries preserved",
            "arbitrary free-node values allowed by locality",
            "source reports fully replayed",
            "conditional state lineage differs from a live execution seal",
        ),
        artifacts=(str(args.output),),
    )
    manifest.validate_for_admission()
    if (
        current_git_source_provenance(ROOT, SOURCE_SCOPE) != provenance
        or args.input.read_bytes() != raw
        or args.historical_input.read_bytes() != historical_raw
    ):
        raise RuntimeError(
            "source or historical input changed during the local-relay audit"
        )
    report.update(
        manifest=manifest.to_dict(),
        source_scope=SOURCE_SCOPE,
        input_evidence={
            "path": str(args.input),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "producer_manifest": parent["manifest"],
            "producer_source_scope": parent["source_scope"],
            "historical_path": str(args.historical_input),
            "historical_sha256": hashlib.sha256(historical_raw).hexdigest(),
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_payload(report), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote C6 relay-handoff report to {args.output}")


if __name__ == "__main__":
    main()
