"""Static local pressure compensation on the closed default C6 phase slice.

The declared stencil contains uniform EPI .5 and one immediate binary64
neighbor at each node in either direction. A single analytic cell exit is
evaluated for its unique balanced source. No state is claimed reachable at
the terminal phase of the original graph preparation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from benchmarks.c6_winding_phase_response import BASE_PHASE  # noqa: E402
from benchmarks.c6_winding_rounding_cells import (  # noqa: E402
    _represented_scalar,
    _represented_vector,
)
from benchmarks.thol_pressure_feedback import _payload  # noqa: E402
from tnfr.config import DEFAULTS  # noqa: E402
from tnfr.dynamics._euler_kernel import initialize_nodal_remainder  # noqa: E402
from tnfr.mathematics.unified_numerical import np  # noqa: E402
from tnfr.physics.c6_phase_orbit import (
    derive_c6_coupling_coherence_phase_orbit,
)  # noqa: E402
from tnfr.physics.c6_pressure_lattice import (  # noqa: E402
    derive_c6_pressure_lattice,
    observe_c6_pressure_lattice,
)
from tnfr.physics.nodal_remainder import (  # noqa: E402
    derive_nodal_remainder_cell_horizon,
    observe_nodal_remainder_cell_exit,
)
from tnfr.physics.nodal_remainder_pressure import (  # noqa: E402
    derive_periodic_phase_source_budget,
    observe_finite_nodal_pressure_drift,
)
from tnfr.research.claims import ClaimStatus  # noqa: E402
from tnfr.research.core_manifests import (  # noqa: E402
    CoreExperimentManifest,
    current_git_source_provenance,
)
from tnfr.utils import normalize_weights  # noqa: E402

SOURCE_SCOPE = ("src/tnfr", "benchmarks")
LOCAL_LOWER, LOCAL_UPPER = 0.375, 0.625
STEP = 1 / 16


def _validated_phase_source(parent):
    manifest = CoreExperimentManifest(**parent["manifest"])
    manifest.validate_for_admission()
    if (
        manifest.claim_id != "O3.a-C6-default-phase-orbit-closure"
        or tuple(parent["source_scope"]) != SOURCE_SCOPE
    ):
        raise ValueError("the input must be the retained B25 phase-orbit report")
    if (
        parent["status"] != "exact_phase_cycle"
        or parent["conditional_phase_periodic"] is not True
        or type(parent["discovery_ceiling"]) is not int
        or parent["discovery_ceiling"] != 256
        or type(parent["discovery_transitions"]) is not int
        or parent["discovery_transitions"] != 90
        or parent["phase_proposal_kernels_executed"] is not True
        or type(parent["new_graph_trajectories"]) is not int
        or parent["new_graph_trajectories"] != 0
        or any(
            parent[name] is not False
            for name in (
                "runtime_executed",
                "operator_word_admission_certified",
                "live_provenance_certified",
                "future_runtime_certified",
                "EPI_band_invariance_certified",
                "actual_mean_compensation_verified",
            )
        )
    ):
        raise ValueError(
            "the parent must retain its declared phase-only discovery and unproved runtime conditions"
        )
    data = parent["phase_orbit"]
    states = tuple(_represented_vector(row) for row in data["phase_states"])
    if (
        len(states) != 91
        or type(data["preperiod"]) is not int
        or data["preperiod"] != 89
        or type(data["period"]) is not int
        or data["period"] != 1
    ):
        raise ValueError("the inherited phase orbit requires its complete 89+1 closure")
    signatures = tuple(tuple(value.hex() for value in phase) for phase in states)
    if (
        signatures[0] != tuple(value.hex() for value in BASE_PHASE)
        or len(set(signatures[:-1])) != 90
    ):
        raise ValueError(
            "the retained path must start at the null winding and preserve its first-repeat boundary"
        )
    source = parent["source"]
    if (
        source["mode"] != "null"
        or _represented_scalar(source["epsilon"]) != 0.0
        or tuple(value.hex() for value in _represented_vector(source["phase"]))
        != signatures[0]
    ):
        raise ValueError(
            "the retained source differs from the declared null initial phase"
        )
    orbit = derive_c6_coupling_coherence_phase_orbit(
        phase_states=states, cycle_start=89
    )
    if _payload(asdict(orbit)) != data:
        raise ValueError(
            "the complete retained phase orbit differs from its shared-owner replay"
        )
    weights = {
        name: _represented_scalar(value) for name, value in source["normalized_weights"]
    }
    if weights != normalize_weights(
        DEFAULTS["DNFR_WEIGHTS"], ("phase", "epi", "vf", "topo")
    ):
        raise ValueError(
            "the phase source must retain the canonical default coefficients"
        )
    reference = derive_c6_pressure_lattice(
        phase=orbit.steps[-1].phase_after_coherence,
        epi_weight=weights["epi"],
        phase_weight=weights["phase"],
        epi_lower=LOCAL_LOWER,
        epi_upper=LOCAL_UPPER,
    )
    periodic = derive_periodic_phase_source_budget(
        phase_contributions=(reference.sources,), block_duration=0.25
    )
    if _payload(asdict(periodic)) != parent["periodic_phase_source"]:
        raise ValueError(
            "the retained periodic source differs from the regenerated canonical pressure contribution"
        )
    return orbit, reference


def _record(observation):
    result = asdict(observation)
    del result["reference"]
    return result


def _stencil():
    yield "half", (0.5,) * 6
    for node in range(6):
        for name, direction in (("down", -math.inf), ("up", math.inf)):
            yield f"node_{node}_{name}", tuple(
                math.nextafter(0.5, direction) if index == node else 0.5
                for index in range(6)
            )


def _balanced_boundary(reference, observation):
    initial = initialize_nodal_remainder(
        observation.epi, epi_lower=LOCAL_LOWER, epi_upper=LOCAL_UPPER
    )
    horizon = derive_nodal_remainder_cell_horizon(
        state=initial,
        timestep=STEP,
        capacity=(1.0,) * 6,
        pressure=observation.pressure,
    )
    if horizon.first_exit_step is None or any(horizon.first_exit_leaves_band):
        raise ValueError(
            "the balanced witness must have a finite first cell exit inside the declared slab"
        )
    count = horizon.first_exit_step
    replay = observe_nodal_remainder_cell_exit(
        state=initial,
        timestep=STEP,
        capacity=(1.0,) * 6,
        pressure=observation.pressure,
        step_budget=count,
    ).sequence
    step = replay.steps[-1]
    before_unchanged = tuple(item.before.epi == initial.epi for item in replay.steps)
    intermediate_unchanged = tuple(
        item.after.epi == initial.epi for item in replay.steps[:-1]
    )
    refreshed = observe_c6_pressure_lattice(reference, epi=step.after.epi)
    mean_exact_change = (
        sum(step.after.exact_epi, Fraction(0)) / 6
        - sum(initial.exact_epi, Fraction(0)) / 6
    )
    if (
        observation.mean_pressure != 0
        or horizon.mean_increment != 0
        or mean_exact_change != 0
    ):
        raise RuntimeError(
            "the balanced finite source lost its exact reconstructed-mean identity"
        )
    return {
        "initial": asdict(initial),
        "horizon": asdict(horizon),
        "last_unchanged_state": asdict(step.before),
        "first_exit_step": asdict(step),
        "refreshed_pressure": _record(refreshed),
        "shared_replay_before_states_unchanged": before_unchanged,
        "shared_replay_intermediate_states_unchanged": intermediate_unchanged,
        "shared_prefix_balances": tuple(
            {
                "ordinal": prefix.ordinal,
                "mean_nodal_area": prefix.mean_nodal_area,
                "mean_reconstructed_change": prefix.mean_reconstructed_change,
                "mean_visible_change": prefix.mean_visible_change,
                "mean_identity_residual": prefix.mean_identity_residual,
                "nodal_identity_residual": prefix.identity_residual,
            }
            for prefix in replay.prefixes
        ),
        "reconstructed_mean_change_through_exit": mean_exact_change,
        "visible_mean_change_through_exit": sum(
            map(Fraction, step.after.epi), Fraction(0)
        )
        / 6
        - sum(map(Fraction, initial.epi), Fraction(0)) / 6,
        "pressure_balanced_after_refresh": refreshed.mean_pressure == 0,
        "scope": (
            "One conditional shared-kernel replay from zero carry to the analytically determined first exit; "
            "pressure is regenerated at that visible endpoint. No later step is executed "
            "and this preparation is not claimed reachable from the inherited graph trajectory."
        ),
    }


def _finite_class_drift(observations, balanced):
    b = tuple(map(Fraction, balanced.pressure))
    ratios = []
    for item in observations:
        if item is balanced:
            continue
        p = tuple(map(Fraction, item.pressure))
        deficit = -sum(p, Fraction(0))
        if deficit <= 0:
            return None
        ratios.append(
            abs(sum((a * value for a, value in zip(b, p, strict=True)), Fraction(0)))
            / deficit
        )
    bound = max(ratios)
    epsilon = 1 / (1 + 2 * bound)
    functional = tuple(-1 + epsilon * value for value in b)
    drift = observe_finite_nodal_pressure_drift(
        epi_states=tuple(item.epi for item in observations),
        pressure_vectors=tuple(item.pressure for item in observations),
        functional=functional,
        timestep=STEP,
        epi_lower=LOCAL_LOWER,
        epi_upper=LOCAL_UPPER,
    )
    return {
        "ratio_bound": bound,
        "separator_epsilon": epsilon,
        "observation": asdict(drift),
        "conditional_class_escape_certified": drift.conditional_class_escape_certified,
        "pressure_provenance_certified": drift.pressure_provenance_certified,
        "positive_band_exit_certified": drift.positive_band_exit_certified,
        "scope": (
            "The rational separating functional observes the original generated pressures and "
            "does not change them. Escape concerns only the thirteen displayed stencil states, "
            "with admissible carry, not exit from the slab or the full positive EPI band."
        ),
    }


def analyze_c6_winding_pressure_lattice(parent):
    """Replay retained phase closure, then inspect exactly thirteen static states."""
    orbit, reference = _validated_phase_source(parent)
    points = tuple(_stencil())
    observations = tuple(
        observe_c6_pressure_lattice(reference, epi=epi) for _, epi in points
    )
    balanced = tuple(
        index for index, item in enumerate(observations) if item.mean_pressure == 0
    )
    boundary = (
        _balanced_boundary(reference, observations[balanced[0]])
        if len(balanced) == 1
        else None
    )
    drift = (
        _finite_class_drift(observations, observations[balanced[0]])
        if len(balanced) == 1
        else None
    )
    return {
        "phase_source": {
            "initial": orbit.phase_states[0],
            "terminal": orbit.phase_states[-1],
            "preperiod": orbit.preperiod,
            "period": orbit.period,
            "complete_retained_orbit_replayed": True,
            "new_phase_search_executed": False,
        },
        "lattice_reference": asdict(reference),
        "stencil": tuple(
            {"name": name, "observation": _record(item)}
            for (name, _), item in zip(points, observations, strict=True)
        ),
        "balanced_stencil_indices": balanced,
        "balanced_boundary": boundary,
        "finite_class_drift": drift,
        "runtime_executed": False,
        "new_graph_trajectories": 0,
        "static_pressure_points": len(points) + (boundary is not None),
        "live_provenance_certified": False,
        "tail_epi_state_reachability_certified": False,
        "future_mean_compensation_verified": False,
        "correlated_invariant_set_excluded": False,
        "positive_band_exit_certified": reference.positive_band_exit_certified,
        "scope": (
            "Static canonical pressure generation on a fixed declared stencil, a local Cartesian "
            "trapping obstruction, and one finite balanced-cell boundary. No long EPI trajectory, "
            "new operator word, arbitrary pressure assignment, or inherited-tail reachability is claimed."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "artifacts/research/c6_winding_phase_orbit.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/research/c6_winding_pressure_lattice.json",
    )
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        raise ValueError("derived output must not overwrite its historical input")
    source = args.input.read_bytes()
    parent = json.loads(source)
    provenance = current_git_source_provenance(ROOT, SOURCE_SCOPE)
    report = analyze_c6_winding_pressure_lattice(parent)
    sha, dirty, digest = provenance
    manifest = CoreExperimentManifest(
        claim_id="O3.a-C6-local-pressure-lattice-compensation",
        git_sha=sha,
        source_dirty=dirty,
        dirty_source_hash=digest,
        versions={"python": platform.python_version(), "numpy": np.__version__},
        graph_construction="Thirteen static EPI stencil points on the replayed B25 closed C6 phase slice",
        capacity_specification="Unit capacity and support, canonical default weights, local slab [3/8,5/8]",
        solver="Exact local lattice, analytic cell horizon and conditional shared-kernel replay to first exit",
        result_status=ClaimStatus.DERIVED,
        telemetry=(
            "generated pressure compensation",
            "Cartesian corner sign obstruction",
            "finite stencil separating functional",
            "first-exit fresh pressure",
        ),
        controls=(
            "uniform half-EPI and twelve immediate float neighbors",
            "no reachable-tail assumption",
            "no long trajectory or changed dynamics",
        ),
        artifacts=(str(args.output),),
    )
    manifest.validate_for_admission()
    if (
        current_git_source_provenance(ROOT, SOURCE_SCOPE) != provenance
        or args.input.read_bytes() != source
    ):
        raise RuntimeError(
            "analysis source or historical input changed during the local lattice audit"
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
    print(f"Wrote bounded C6 pressure-lattice audit to {args.output}")


if __name__ == "__main__":
    main()
