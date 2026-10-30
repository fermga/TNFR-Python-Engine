"""An exact two-relay budget from the retained B43 carried C6 endpoint.

The observation advances a conditional fixed-source numerical branch through
its first proved family exit. It does not execute a later graph-owned word or
override the capacity change caused by the historical terminal Silence.
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
from benchmarks.c6_winding_mean_cylinder import (
    INPUT_NAME,
    replay_c6_phase_tail_evidence,
)  # noqa: E402
from benchmarks.c6_winding_phase_tail_runtime import SEGMENT_DURATION  # noqa: E402
from benchmarks.c6_winding_pressure_sign import _canonical  # noqa: E402
from benchmarks.thol_pressure_feedback import _payload  # noqa: E402
from tnfr.mathematics.unified_numerical import np  # noqa: E402
from tnfr.physics.c6_carried_closure import derive_c6_carried_closure  # noqa: E402
from tnfr.physics.c6_carried_relay import (  # noqa: E402
    derive_c6_carried_relay,
    observe_c6_carried_relay_exit,
)
from tnfr.research.claims import ClaimStatus  # noqa: E402
from tnfr.research.core_manifests import (
    CoreExperimentManifest,
    current_git_source_provenance,
)  # noqa: E402


def _derive_relay_budget(parent, step_budget):
    profile, state = replay_c6_phase_tail_evidence(parent)
    closure = derive_c6_carried_closure(profile, state=state, timestep=SEGMENT_DURATION)
    relay = derive_c6_carried_relay(closure, relay_nodes=(0, 3))
    result = observe_c6_carried_relay_exit(relay, step_budget=step_budget)
    return relay, result


def _relay_budget_record(parent, relay, result, step_budget):
    closure = relay.closure
    state = closure.base_tube.state
    profile = closure.base_tube.contraction.profile
    bound = asdict(relay)
    bound.pop("closure")
    observation = asdict(result)
    observation.pop("relay")
    return {
        "contract": {
            "question": "Can the two opposite-node relay strips contain the conditional B43 continuation?",
            "derivation": "Exact canonical pressure differences cancel relay switches in corrected coordinates",
            "stop_condition": "First non-relay visible-cell exit or band failure, within the analytic deadline",
            "computational_step_budget": step_budget,
            "budget_is_physical_parameter": False,
            "pressure_projected_or_carry_reset": False,
        },
        "source": {
            "retained_B43_endpoint": asdict(state),
            "phase": profile.lattice.source.phase,
            "historical_nodal_steps_replayed": 356,
            "serialized_numerical_chain_verified": True,
            "live_execution_seal_recreated": False,
            "capacity": (1.0,) * 6,
            "timestep": SEGMENT_DURATION,
            "origin_mean": closure.base_tube.initial_mean,
            "origin_energy": closure.base_tube.initial_energy,
            "energy_bound": closure.energy_bound,
            "historical_terminal_SHA_capacity": tuple(
                parent["B43_original_phase_tail_runtime"]["terminal_silence"][
                    "capacity_after"
                ],
            ),
        },
        "B46_two_relay_certificate": bound,
        "B46_conditional_first_exit": observation,
        "conditional_product_relay_inclusion_certified": relay.conditional_product_relay_inclusion,
        "conditional_family_escape_certified": True,
        "full_region_invariant": False,
        "conditional_numerical_steps": result.exit_step,
        "conditional_elapsed_time": F(SEGMENT_DURATION) * result.exit_step,
        "new_live_graph_steps": 0,
        "whole_band_exit_certified": result.band_failure,
        "indefinite_trapping_certified": False,
        "general_correlated_region_excluded": False,
        "future_runtime_certified": False,
    }


def analyze_c6_winding_relay_budget(parent, *, step_budget=1024):
    """Derive the stopping bound before observing any conditional new step."""
    relay, result = _derive_relay_budget(parent, step_budget)
    return _relay_budget_record(parent, relay, result, step_budget)


def replay_c6_relay_budget_evidence(parent, *, historical_bytes):
    """Rebuild B46 and bind its endpoint to the original retained B43 input.

    This verifies source bytes and the complete serialized numerical result,
    including both historical and conditional steps. A live graph seal is
    not recreated, and supplied cached certificate fields are not trusted.
    """
    manifest = CoreExperimentManifest(**parent["manifest"])
    manifest.validate_for_admission()
    if (
        manifest.claim_id != "O3.a-C6-carried-two-relay-budget"
        or tuple(parent["source_scope"]) != SOURCE_SCOPE
    ):
        raise ValueError("the handoff requires the retained B46 relay-budget report")
    if type(historical_bytes) is not bytes:
        raise TypeError("historical_bytes must be the exact retained B43 file bytes")
    evidence = parent["input_evidence"]
    if hashlib.sha256(historical_bytes).hexdigest() != evidence["sha256"]:
        raise ValueError(
            "the retained B43 input hash differs from B46's declared source"
        )
    historical = json.loads(historical_bytes)
    if _canonical(evidence["producer_manifest"]) != _canonical(
        historical["manifest"]
    ) or _canonical(evidence["producer_source_scope"]) != _canonical(
        historical["source_scope"]
    ):
        raise ValueError("the retained B43 producer metadata differs from B46's source")
    step_budget = parent["contract"]["computational_step_budget"]
    relay, result = _derive_relay_budget(historical, step_budget)
    expected = _relay_budget_record(historical, relay, result, step_budget)
    payload = {
        key: value
        for key, value in parent.items()
        if key not in ("manifest", "source_scope", "input_evidence")
    }
    if _canonical(payload) != _canonical(expected):
        raise ValueError(
            "the retained B46 result differs from its complete numerical replay"
        )
    if result.endpoint is None:
        raise ValueError(
            "a band-failed B46 branch has no admitted endpoint for handoff"
        )
    return relay, result.endpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=ROOT / "artifacts/research" / INPUT_NAME
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/research/c6_winding_relay_budget.json",
    )
    parser.add_argument("--step-budget", type=int, default=1024)
    args = parser.parse_args()
    if args.output.resolve() == args.input.resolve():
        raise ValueError("the derived report must not overwrite its historical input")
    raw = args.input.read_bytes()
    parent = json.loads(raw)
    provenance = current_git_source_provenance(ROOT, SOURCE_SCOPE)
    report = analyze_c6_winding_relay_budget(parent, step_budget=args.step_budget)
    sha, dirty, digest = provenance
    manifest = CoreExperimentManifest(
        claim_id="O3.a-C6-carried-two-relay-budget",
        git_sha=sha,
        source_dirty=dirty,
        dirty_source_hash=digest,
        versions={"python": platform.python_version(), "numpy": np.__version__},
        graph_construction="Fixed unit C6 source replayed from original B43 pre-SHA evidence",
        capacity_specification="Conditional unit capacity; historical terminal SHA is not undone",
        solver="Shared carried nodal integrator and exact two-relay corrected-coordinate budget",
        result_status=ClaimStatus.DERIVED,
        telemetry=(
            "canonical pressure rows",
            "relay strip inclusion",
            "exact corrected-coordinate drift",
            "analytic family-exit bound",
            "exact nodal area and mean",
        ),
        controls=(
            "unaltered B43 endpoint carry",
            "both nearest-even tie orientations",
            "no new graph-owned word or live seal",
            "family exit differs from whole-band escape",
        ),
        artifacts=(str(args.output),),
    )
    manifest.validate_for_admission()
    if (
        current_git_source_provenance(ROOT, SOURCE_SCOPE) != provenance
        or args.input.read_bytes() != raw
    ):
        raise RuntimeError("source or historical input changed during the relay audit")
    report.update(
        manifest=manifest.to_dict(),
        source_scope=SOURCE_SCOPE,
        input_evidence={
            "path": str(args.input),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "producer_manifest": parent["manifest"],
            "producer_source_scope": parent["source_scope"],
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_payload(report), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote C6 relay-budget report to {args.output}")


if __name__ == "__main__":
    main()
