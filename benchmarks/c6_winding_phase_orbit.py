"""Bounded default C6 phase-orbit discovery and exact closure replay.

Only the inherited null phase is used. The fixed ceiling of 256 UM/IL
proposal transitions is a computational stop, not a model parameter.
Detached phase fixtures execute no admitted operator word or EPI flow.
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

from benchmarks.c6_winding_defect_budget import analyze_c6_defect_case  # noqa: E402
from benchmarks.c6_winding_phase_response import BASE_PHASE  # noqa: E402
from benchmarks.c6_winding_pressure_cells import _validate_parent  # noqa: E402
from benchmarks.c6_winding_rounding_cells import (  # noqa: E402
    _represented_scalar,
    _represented_vector,
)
from benchmarks.thol_pressure_feedback import _payload  # noqa: E402
from tnfr.config import DEFAULTS  # noqa: E402
from tnfr.mathematics.unified_numerical import np  # noqa: E402
from tnfr.physics.binary64_pressure_equilibrium import (  # noqa: E402
    derive_binary64_c6_pressure_equilibrium_obstruction,
)
from tnfr.physics.c6_phase_orbit import (  # noqa: E402
    derive_c6_coupling_coherence_phase_orbit,
    observe_c6_coupling_coherence_phase_step,
)
from tnfr.physics.nodal_remainder_pressure import (
    derive_periodic_phase_source_budget,
)  # noqa: E402
from tnfr.research.claims import ClaimStatus  # noqa: E402
from tnfr.research.core_manifests import (  # noqa: E402
    CoreExperimentManifest,
    current_git_source_provenance,
)
from tnfr.utils import normalize_weights  # noqa: E402

SOURCE_SCOPE = ("src/tnfr", "benchmarks")
DISCOVERY_CEILING = 256
BLOCK_DURATION = 0.25


def _discover(initial):
    states, steps = [initial], []
    seen = {tuple(value.hex() for value in initial): 0}
    for _ in range(DISCOVERY_CEILING):
        step = observe_c6_coupling_coherence_phase_step(phase=states[-1])
        after = step.phase_after_coherence
        steps.append(step)
        states.append(after)
        signature = tuple(value.hex() for value in after)
        if signature in seen:
            return tuple(states), tuple(steps), seen[signature]
        seen[signature] = len(steps)
    return tuple(states), tuple(steps), None


def _retained_prefix_bindings(case, steps):
    bindings = []
    for retained, step in zip(case["cycles"], steps[:2], strict=True):

        def matches(actual, recorded):
            return tuple(value.hex() for value in actual) == tuple(
                value.hex() for value in _represented_vector(recorded)
            )

        checks = {
            "before_matches": matches(
                step.phase_before, retained["before_capture"]["phase"]
            ),
            "UM_matches": matches(
                step.phase_after_coupling, retained["um"]["raw_capture"]["phase"]
            ),
            "IL_matches": matches(
                step.phase_after_coherence, retained["il"]["raw_capture"]["phase"]
            ),
        }
        if not all(checks.values()):
            raise ValueError(
                "phase proposal prefix differs from the two retained B20 UM/IL transitions"
            )
        bindings.append({"ordinal": retained["ordinal"], **checks})
    return tuple(bindings)


def analyze_c6_winding_phase_orbit(parent):
    """Discover one finite phase closure, then verify it through the owner."""
    case = _validate_parent(parent)[0]
    analyze_c6_defect_case(case)
    initial = _represented_vector(case["initial_capture"]["phase"])
    if initial != BASE_PHASE:
        raise ValueError(
            "the discovery source must be the inherited null winding preparation"
        )
    weights = {
        name: _represented_scalar(value)
        for name, value in case["initial_capture"]["normalized_weights"]
    }
    defaults = normalize_weights(
        DEFAULTS["DNFR_WEIGHTS"], ("phase", "epi", "vf", "topo")
    )
    if weights != defaults:
        raise ValueError(
            "the inherited phase-source coefficients must retain the canonical defaults"
        )
    states, steps, cycle_start = _discover(initial)
    prefix = _retained_prefix_bindings(case, steps)
    report = {
        "source": {
            "mode": "null",
            "epsilon": Fraction(0),
            "location": "current_cases[0].initial_capture.phase",
            "phase": initial,
            "first_two_retained_bindings": prefix,
            "normalized_weights": tuple(
                (name, Fraction(value)) for name, value in weights.items()
            ),
        },
        "discovery_ceiling": DISCOVERY_CEILING,
        "discovery_transitions": len(steps),
        "status": (
            "exact_phase_cycle"
            if cycle_start is not None
            else "no_repeat_within_ceiling"
        ),
        "runtime_executed": False,
        "phase_proposal_kernels_executed": True,
        "new_graph_trajectories": 0,
        "operator_word_admission_certified": False,
        "live_provenance_certified": False,
        "future_runtime_certified": False,
        "EPI_band_invariance_certified": False,
        "actual_mean_compensation_verified": False,
        "scope": (
            "Exact finite closure of the default phase projection in the same numerical environment, "
            "conditional on its fixed support/configuration and admitted stages. This is not an "
            "executed UM/IL word, an EPI-flow trajectory, or proof of future EPI-band membership. "
            "A periodic phase contribution does not make the full generated pressure periodic."
        ),
    }
    if cycle_start is None:
        report["finite_phase_path"] = {
            "phase_states": states,
            "steps": tuple(map(asdict, steps)),
        }
        return report
    orbit = derive_c6_coupling_coherence_phase_orbit(
        phase_states=states, cycle_start=cycle_start
    )
    if orbit.steps != steps:
        raise RuntimeError(
            "independent supplied-orbit replay differs from bounded discovery"
        )
    # Each inherited flow block follows IL. For a nontrivial cycle this is
    # the output of each transition, not its input phase (a cyclic rotation).
    cycle_phases = tuple(
        step.phase_after_coherence for step in orbit.steps[cycle_start:]
    )
    obstructions = tuple(
        derive_binary64_c6_pressure_equilibrium_obstruction(
            phase=phase,
            epi_weight=weights["epi"],
            phase_weight=weights["phase"],
        )
        for phase in cycle_phases
    )
    contributions = tuple(
        tuple(row.phase_contribution for row in result.rows) for result in obstructions
    )
    source_budget = derive_periodic_phase_source_budget(
        phase_contributions=contributions,
        block_duration=BLOCK_DURATION,
    )
    report.update(
        phase_orbit=asdict(orbit),
        conditional_phase_periodic=orbit.conditional_phase_periodic,
        cycle_pressure_obstructions=tuple(map(asdict, obstructions)),
        periodic_phase_source=asdict(source_budget),
        every_cycle_phase_excludes_zero_pressure=all(
            item.no_zero_pressure for item in obstructions
        ),
        source_budget_alignment="Each quarter-duration block follows the cycle step's Coherence output",
        remaining_mean_condition=(
            "The nonphase nodal area, including EPI-pressure reduction and channel-assembly defects, "
            "has not been evolved or bounded. Bounded reconstructed mean requires compensation of "
            "the periodic phase source's linear mean area; no actual compensation is supplied here."
        ),
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "artifacts/research/c6_winding_phase_kernel.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/research/c6_winding_phase_orbit.json",
    )
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        raise ValueError("derived output must not overwrite its historical input")
    source = args.input.read_bytes()
    parent = json.loads(source)
    provenance = current_git_source_provenance(ROOT, SOURCE_SCOPE)
    report = analyze_c6_winding_phase_orbit(parent)
    sha, dirty, digest = provenance
    manifest = CoreExperimentManifest(
        claim_id="O3.a-C6-default-phase-orbit-closure",
        git_sha=sha,
        source_dirty=dirty,
        dirty_source_hash=digest,
        versions={"python": platform.python_version(), "numpy": np.__version__},
        graph_construction="Detached phase-only fixtures from the retained B20 null winding source",
        capacity_specification="Unit-capacity phase projection; default coefficients and fixed unit C6 support",
        solver="At most 256 pure UM/IL phase transitions; exact supplied-orbit replay; no EPI integration",
        result_status=ClaimStatus.DERIVED,
        telemetry=(
            "complete transient and periodic phase states",
            "UM midpoint and all U3 margins",
            "terminal pressure-equilibrium obstruction",
            "periodic phase-source mean budget",
        ),
        controls=(
            "one inherited null preparation",
            "fixed discovery ceiling 256",
            "first two retained phase transitions",
            "no full-word admission or nonphase-area claim",
        ),
        artifacts=(str(args.output),),
    )
    manifest.validate_for_admission()
    if (
        current_git_source_provenance(ROOT, SOURCE_SCOPE) != provenance
        or args.input.read_bytes() != source
    ):
        raise RuntimeError(
            "analysis source or historical input changed during the phase-only audit"
        )
    report.update(
        manifest=manifest.to_dict(),
        source_scope=SOURCE_SCOPE,
        input_evidence={
            "path": str(args.input),
            "sha256": hashlib.sha256(source).hexdigest(),
            "producer_manifest": parent["manifest"],
            "producer_source_scope": parent["source_scope"],
            "producer_input_evidence": parent["input_evidence"],
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_payload(report), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote bounded C6 phase-orbit audit to {args.output}")


if __name__ == "__main__":
    main()
