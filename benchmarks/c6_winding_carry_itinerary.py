"""Two exact cell boundaries continuing the retained B26 carried witness.

The inherited remainder is preserved. Pressure is regenerated on each new
visible state, while inverse-cell constraints audit the complete finite
itinerary. Two additional visible self-loop controls distinguish carry
feasibility from a genuinely closed augmented-state cycle.
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

from benchmarks.c6_winding_defect_budget import _number  # noqa: E402
from benchmarks.c6_winding_phase_response import BASE_PHASE  # noqa: E402
from benchmarks.c6_winding_pressure_lattice import (  # noqa: E402
    LOCAL_LOWER,
    LOCAL_UPPER,
    STEP,
    _balanced_boundary,
    _record,
    _stencil,
)
from benchmarks.c6_winding_rounding_cells import (  # noqa: E402
    _represented_scalar,
    _represented_vector,
)
from benchmarks.thol_pressure_feedback import _payload  # noqa: E402
from tnfr.config import DEFAULTS  # noqa: E402
from tnfr.dynamics._euler_kernel import NodalRemainderState  # noqa: E402
from tnfr.mathematics.unified_numerical import np  # noqa: E402
from tnfr.physics.c6_phase_orbit import (
    observe_c6_coupling_coherence_phase_step,
)  # noqa: E402
from tnfr.physics.c6_pressure_lattice import (  # noqa: E402
    derive_c6_pressure_lattice,
    observe_c6_pressure_lattice,
)
from tnfr.physics.nodal_remainder import (  # noqa: E402
    derive_nodal_remainder_cell_horizon,
    derive_nodal_remainder_itinerary,
    observe_nodal_remainder_cell_exit,
    observe_nodal_remainder_sequence,
)
from tnfr.physics.nodal_remainder_pressure import (
    observe_finite_nodal_pressure_drift,
)  # noqa: E402
from tnfr.research.claims import ClaimStatus  # noqa: E402
from tnfr.research.core_manifests import (  # noqa: E402
    CoreExperimentManifest,
    current_git_source_provenance,
)
from tnfr.utils import normalize_weights  # noqa: E402

SOURCE_SCOPE = ("src/tnfr", "benchmarks")
BOUNDARY_COUNT = 2
SELF_LOOP_LENGTHS = (12, 13)


def _state(data):
    state = NodalRemainderState(
        _represented_vector(data["epi"]),
        tuple(_number(value) for value in data["remainder"]),
        _represented_scalar(data["epi_lower"]),
        _represented_scalar(data["epi_upper"]),
    )
    state.exact_epi
    return state


def _validated_source(parent):
    manifest = CoreExperimentManifest(**parent["manifest"])
    manifest.validate_for_admission()
    if (
        manifest.claim_id != "O3.a-C6-local-pressure-lattice-compensation"
        or tuple(parent["source_scope"]) != SOURCE_SCOPE
    ):
        raise ValueError("input must be the retained B26 local pressure-lattice report")
    if (
        type(parent["new_graph_trajectories"]) is not int
        or parent["new_graph_trajectories"] != 0
        or type(parent["static_pressure_points"]) is not int
        or parent["static_pressure_points"] != 14
        or any(
            parent[name] is not False
            for name in (
                "runtime_executed",
                "live_provenance_certified",
                "tail_epi_state_reachability_certified",
                "future_mean_compensation_verified",
                "correlated_invariant_set_excluded",
                "positive_band_exit_certified",
            )
        )
    ):
        raise ValueError(
            "the parent must retain its static and conditional carried-prefix scope"
        )
    phase = parent["phase_source"]
    terminal = _represented_vector(phase["terminal"])
    if (
        type(phase["preperiod"]) is not int
        or phase["preperiod"] != 89
        or type(phase["period"]) is not int
        or phase["period"] != 1
        or phase["complete_retained_orbit_replayed"] is not True
        or phase["new_phase_search_executed"] is not False
        or tuple(value.hex() for value in _represented_vector(phase["initial"]))
        != tuple(value.hex() for value in BASE_PHASE)
    ):
        raise ValueError(
            "the parent phase declaration differs from the inherited null orbit"
        )
    closed = observe_c6_coupling_coherence_phase_step(phase=terminal)
    if tuple(value.hex() for value in closed.phase_after_coherence) != tuple(
        value.hex() for value in terminal
    ):
        raise ValueError(
            "the inherited terminal phase is not closed under the actual proposal owners"
        )
    source = parent["lattice_reference"]["source"]
    defaults = normalize_weights(
        DEFAULTS["DNFR_WEIGHTS"], ("phase", "epi", "vf", "topo")
    )
    if (
        _represented_scalar(source["epi_weight"]) != defaults["epi"]
        or _represented_scalar(source["phase_weight"]) != defaults["phase"]
        or tuple(value.hex() for value in _represented_vector(source["phase"]))
        != tuple(value.hex() for value in terminal)
    ):
        raise ValueError(
            "the inherited pressure source differs from its phase or default coefficients"
        )
    reference = derive_c6_pressure_lattice(
        phase=terminal,
        epi_weight=defaults["epi"],
        phase_weight=defaults["phase"],
        epi_lower=LOCAL_LOWER,
        epi_upper=LOCAL_UPPER,
    )
    if _payload(asdict(reference)) != parent["lattice_reference"]:
        raise ValueError(
            "the inherited local lattice differs from its shared-owner reconstruction"
        )
    points = tuple(_stencil())
    indices = parent["balanced_stencil_indices"]
    if (
        tuple(indices) != (1,)
        or any(type(index) is not int for index in indices)
        or len(parent["stencil"]) != len(points)
        or any(
            record["name"] != name
            or _represented_vector(record["observation"]["epi"]) != epi
            for record, (name, epi) in zip(parent["stencil"], points, strict=True)
        )
    ):
        raise ValueError(
            "the retained stencil or unique balanced-state declaration changed"
        )
    balanced = observe_c6_pressure_lattice(reference, epi=points[1][1])
    if (
        balanced.mean_pressure != 0
        or _payload(_record(balanced)) != parent["stencil"][1]["observation"]
    ):
        raise ValueError(
            "the retained balanced pressure differs from actual canonical generation"
        )
    # Reuse B26's shared six-step prefix replay and fresh endpoint pressure.
    # No additional historical input file or graph trajectory is required.
    verified_boundary = _balanced_boundary(reference, balanced)
    if _payload(verified_boundary) != parent["balanced_boundary"]:
        raise ValueError(
            "the inherited carried boundary differs from its complete shared-prefix replay"
        )
    state = _state(parent["balanced_boundary"]["first_exit_step"]["after"])
    return reference, balanced, state, verified_boundary["refreshed_pressure"]


def _prefix_summary(sequence):
    return tuple(
        {
            "ordinal": prefix.ordinal,
            "mean_nodal_area": prefix.mean_nodal_area,
            "mean_reconstructed_change": prefix.mean_reconstructed_change,
            "mean_visible_change": prefix.mean_visible_change,
            "mean_carry_transfer": prefix.mean_carry_transfer,
            "mean_identity_residual": prefix.mean_identity_residual,
            "nodal_identity_residual": prefix.identity_residual,
        }
        for prefix in sequence.prefixes
    )


def _itinerary_record(itinerary):
    data = asdict(itinerary)
    for name in (
        "witness_sequence",
        "epi_states",
        "timesteps",
        "capacities",
        "pressures",
    ):
        data.pop(name, None)
    data["pressure_provenance_certified"] = itinerary.pressure_provenance_certified
    data["runtime_provenance_certified"] = itinerary.runtime_provenance_certified
    data["witness_prefix_balances"] = (
        _prefix_summary(itinerary.witness_sequence)
        if itinerary.witness_sequence is not None
        else None
    )
    return data


def _belongs(state, itinerary):
    values = state.exact_epi
    return tuple(
        (value > cell.lower or cell.lower_closed and value == cell.lower)
        and (value < cell.upper or cell.upper_closed and value == cell.upper)
        for value, cell in zip(values, itinerary.coordinates, strict=True)
    )


def _four_state_drift(balanced, initial, first_pressure, boundaries):
    states = (balanced.epi, initial.epi) + tuple(
        tuple(item["endpoint"]["epi"]) for item in boundaries
    )
    pressures = (balanced.pressure, tuple(first_pressure["pressure"])) + tuple(
        tuple(item["refreshed_pressure"]["pressure"]) for item in boundaries
    )
    for node in range(6):
        column = tuple(row[node] for row in pressures)
        sign = 1 if min(column) > 0 else -1 if max(column) < 0 else 0
        if not sign:
            continue
        functional = tuple(Fraction(sign if i == node else 0) for i in range(6))
        observed = observe_finite_nodal_pressure_drift(
            epi_states=states,
            pressure_vectors=pressures,
            functional=functional,
            timestep=STEP,
            epi_lower=LOCAL_LOWER,
            epi_upper=LOCAL_UPPER,
        )
        return {
            "status": "axis_separator",
            "node": node,
            "sign": sign,
            "observation": asdict(observed),
            "conditional_class_escape_certified": observed.conditional_class_escape_certified,
            "pressure_provenance_certified": observed.pressure_provenance_certified,
            "positive_band_exit_certified": observed.positive_band_exit_certified,
            "scope": (
                "The first coordinate with one strict pressure sign across these four supplied states "
                "excludes a closed carried cycle confined to this finite displayed-state class; "
                "it does not exclude other visited states or certify whole-band exit."
            ),
        }
    return {"status": "no_axis_separator", "broader_separator_decided": False}


def analyze_c6_winding_carry_itinerary(parent):
    """Continue exactly two analytic boundaries without resetting carried state."""
    reference, balanced, initial, current_pressure = _validated_source(parent)
    state, first_pressure = initial, current_pressure
    boundaries, steps, pressures = [], [], []
    stencil = {epi for _, epi in _stencil()}
    for ordinal in range(1, BOUNDARY_COUNT + 1):
        pressure = tuple(current_pressure["pressure"])
        horizon = derive_nodal_remainder_cell_horizon(
            state=state,
            timestep=STEP,
            capacity=(1.0,) * 6,
            pressure=pressure,
        )
        count = horizon.first_exit_step
        if count is None or any(horizon.first_exit_leaves_band):
            raise ValueError(
                "the declared next boundary must exist inside the local EPI slab"
            )
        replay = observe_nodal_remainder_cell_exit(
            state=state,
            timestep=STEP,
            capacity=(1.0,) * 6,
            pressure=pressure,
            step_budget=count,
        ).sequence
        refreshed = observe_c6_pressure_lattice(reference, epi=replay.endpoint.epi)
        boundaries.append(
            {
                "ordinal": ordinal,
                "initial": asdict(state),
                "horizon": asdict(horizon),
                "source_pressure": current_pressure,
                "endpoint": asdict(replay.endpoint),
                "refreshed_pressure": _record(refreshed),
                "prefix_balances": _prefix_summary(replay),
                "endpoint_in_original_stencil": replay.endpoint.epi in stencil,
            }
        )
        steps.extend(replay.steps)
        pressures.extend((pressure,) * count)
        state, current_pressure = replay.endpoint, _record(refreshed)
    visible = (initial.epi,) + tuple(step.after.epi for step in steps)
    times, capacities = (STEP,) * len(steps), ((1.0,) * 6,) * len(steps)
    actual = observe_nodal_remainder_sequence(
        initial=initial,
        timesteps=times,
        capacities=capacities,
        pressures=tuple(pressures),
    )
    if actual.steps != tuple(steps) or actual.endpoint != state:
        raise RuntimeError(
            "whole continuation replay differs from the two boundary prefixes"
        )
    itinerary = derive_nodal_remainder_itinerary(
        epi_states=visible,
        timesteps=times,
        capacities=capacities,
        pressures=tuple(pressures),
        epi_lower=LOCAL_LOWER,
        epi_upper=LOCAL_UPPER,
    )
    membership = _belongs(initial, itinerary)
    if not itinerary.feasible or not all(membership):
        raise RuntimeError("inverse itinerary cells lost the actual inherited carry")
    loops = []
    for length in SELF_LOOP_LENGTHS:
        control = derive_nodal_remainder_itinerary(
            epi_states=(balanced.epi,) * (length + 1),
            timesteps=(STEP,) * length,
            capacities=((1.0,) * 6,) * length,
            pressures=(balanced.pressure,) * length,
            epi_lower=LOCAL_LOWER,
            epi_upper=LOCAL_UPPER,
        )
        loops.append(
            {
                "length": length,
                "epi": balanced.epi,
                "pressure": balanced.pressure,
                "pressure_mean": balanced.mean_pressure,
                "itinerary": _itinerary_record(control),
                "scope": "A proposed visible self-loop with existential carry, not a reached augmented-state cycle",
            }
        )
    return {
        "source": {
            "location": "balanced_boundary.first_exit_step.after",
            "inherited_state": asdict(initial),
            "retained_B26_prefix_replayed": True,
            "phase": reference.source.phase,
            "pressure_generated_by_shared_cpu_kernel": True,
        },
        "boundary_budget": BOUNDARY_COUNT,
        "boundaries": tuple(boundaries),
        "continuation": {
            "step_count": len(steps),
            "steps": tuple(map(asdict, steps)),
            "prefix_balances": _prefix_summary(actual),
            "endpoint": asdict(actual.endpoint),
            "mean_nodal_area": actual.prefixes[-1].mean_nodal_area,
            "mean_reconstructed_change": actual.prefixes[-1].mean_reconstructed_change,
            "itinerary": _itinerary_record(itinerary),
            "supplied_initial_carry_coordinate_membership": membership,
            "supplied_initial_carry_feasible": all(membership),
        },
        "balanced_visible_self_loops": tuple(loops),
        "four_state_drift": _four_state_drift(
            balanced, initial, first_pressure, boundaries
        ),
        "runtime_executed": False,
        "new_graph_trajectories": 0,
        "live_provenance_certified": False,
        "original_tail_reachability_certified": False,
        "future_compensation_cycle_certified": False,
        "positive_band_exit_certified": False,
        "scope": (
            "Two declared cell boundaries with inherited carry and freshly generated endpoint pressures; "
            "exact inverse-cell feasibility distinguishes supplied, zero and existential initial carry. "
            "Visible closure alone does not close the carried nodal state; a signed-coordinate "
            "separator excludes indefinite residence in the four supplied visible states. No long trajectory, live "
            "operator word, original-tail reachability or future compensation cycle is asserted."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "artifacts/research/c6_winding_pressure_lattice.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/research/c6_winding_carry_itinerary.json",
    )
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        raise ValueError("derived output must not overwrite its historical input")
    source = args.input.read_bytes()
    parent = json.loads(source)
    provenance = current_git_source_provenance(ROOT, SOURCE_SCOPE)
    report = analyze_c6_winding_carry_itinerary(parent)
    sha, dirty, digest = provenance
    manifest = CoreExperimentManifest(
        claim_id="O3.a-C6-carried-cell-itinerary",
        git_sha=sha,
        source_dirty=dirty,
        dirty_source_hash=digest,
        versions={"python": platform.python_version(), "numpy": np.__version__},
        graph_construction="Retained B26 conditional carried endpoint on the closed B25 C6 phase slice",
        capacity_specification="Inherited unit capacity/support/default weights and local slab [3/8,5/8]",
        solver="Two analytically determined cell boundaries through shared carried prefix kernels; no graph execution",
        result_status=ClaimStatus.DERIVED,
        telemetry=(
            "actual-carry continuation",
            "generated pressure mean sign changes",
            "exact inverse itinerary cells",
            "twelve/thirteen-step visible self-loop controls",
            "four-state signed-coordinate drift",
        ),
        controls=(
            "exactly two boundary transitions",
            "no carry reset",
            "no reachable-tail assumption",
        ),
        artifacts=(str(args.output),),
    )
    manifest.validate_for_admission()
    if (
        current_git_source_provenance(ROOT, SOURCE_SCOPE) != provenance
        or args.input.read_bytes() != source
    ):
        raise RuntimeError(
            "analysis source or retained input changed during the finite itinerary audit"
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
    print(f"Wrote bounded C6 carried-itinerary audit to {args.output}")


if __name__ == "__main__":
    main()
