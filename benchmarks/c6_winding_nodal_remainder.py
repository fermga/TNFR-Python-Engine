"""Prescribed-input nodal remainder comparison on retained B20 C6 pressures.

The three preparations each supply two held-pressure tuples, four substeps
per tuple. Remainders continue across the tuple boundary. Pressures are never
refreshed on the changed states: this is a detached arithmetic counterfactual,
not a graph trajectory or a new production-integrator certificate.
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

from benchmarks.c6_winding_pressure_cells import analyze_c6_pressure_cells  # noqa: E402
from benchmarks.thol_pressure_feedback import _payload  # noqa: E402
from tnfr.dynamics._euler_kernel import euler_update, initialize_nodal_remainder  # noqa: E402
from tnfr.mathematics.unified_numerical import np  # noqa: E402
from tnfr.physics.nodal_remainder import observe_nodal_remainder_sequence  # noqa: E402
from tnfr.research.claims import ClaimStatus  # noqa: E402
from tnfr.research.core_manifests import (  # noqa: E402
    CoreExperimentManifest, current_git_source_provenance,
)

SOURCE_SCOPE = ("src/tnfr", "benchmarks")
SEQUENCES = ("original", "zero_sum_witness", "opposite_pair_witness")
SUBSTEPS = 4
TIMESTEP = 1 / 16
CAPACITY = (1.0,) * 6


def _mean(values):
    values = tuple(values)
    return sum(values, Fraction(0)) / len(values)


def _prescribed_sequence(case, name):
    cycles = case["cycles"]
    if name == "original":
        pressure_schedule = tuple(cycle["pressure_box"]["flow"]["pressure"] for cycle in cycles)
    else:
        missing = tuple(cycle["ordinal"] for cycle in cycles if cycle[name]["status"] != "feasible")
        if missing:
            return {
                "status": "unavailable", "missing_witness_cycles": missing,
                "scope": "The declared finite pressure construction has no witness for every cycle",
            }
        pressure_schedule = tuple(cycle[name]["pressure"] for cycle in cycles)
    source = cycles[0]["pressure_box"]["flow"]
    initial = initialize_nodal_remainder(
        source["epi"], epi_lower=source["epi_lower"], epi_upper=source["epi_upper"],
    )
    pressures = tuple(pressure for pressure in pressure_schedule for _ in range(SUBSTEPS))
    count = len(pressures)
    observation = observe_nodal_remainder_sequence(
        initial=initial, timesteps=(TIMESTEP,) * count,
        capacities=(CAPACITY,) * count, pressures=pressures,
    )
    ordinary, ordinary_trace = initial.epi, []
    for pressure in pressures:
        rate = tuple(1.0 * value + 0.0 for value in pressure)
        ordinary = tuple(float(euler_update(value, TIMESTEP, slope))
                         for value, slope in zip(ordinary, rate, strict=True))
        ordinary_trace.append(ordinary)
    retained_trace = tuple(tuple(step["after"]) for cycle in cycles
                           for step in cycle["pressure_box"]["flow"]["substeps"])
    if tuple(ordinary_trace) != retained_trace:
        raise RuntimeError("prescribed ordinary Euler no longer reproduces its validated retained trace")
    states = (initial,) + tuple(step.after for step in observation.steps)
    exact_lower, exact_upper = Fraction(initial.epi_lower), Fraction(initial.epi_upper)
    all_states_in_band = all(
        initial.epi_lower <= visible <= initial.epi_upper
        and exact_lower <= exact <= exact_upper
        for state in states for visible, exact in zip(state.epi, state.exact_epi, strict=True)
    )
    identities_hold = all(not any(prefix.identity_residual) and prefix.mean_identity_residual == 0
                          for prefix in observation.prefixes)
    if not all_states_in_band or not identities_hold:
        raise RuntimeError("carried sequence lost its inherited band or nodal prefix identity")
    cycle_records = []
    for index, cycle in enumerate(cycles):
        start, end = index * SUBSTEPS, (index + 1) * SUBSTEPS
        actual_source = states[start]
        captured_source = cycle["pressure_box"]["flow"]["epi"]
        cycle_records.append({
            "ordinal": cycle["ordinal"], "pressure": pressure_schedule[index],
            "pressure_sum": sum(map(Fraction, pressure_schedule[index]), Fraction(0)),
            "recorded_pressure_source_epi": captured_source,
            "carried_source": asdict(actual_source),
            "carried_source_matches_recorded_visible_epi": actual_source.epi == captured_source,
            "carried_source_matches_recorded_exact_epi": actual_source.exact_epi == tuple(map(Fraction, captured_source)),
            "ordinary_endpoint": ordinary_trace[end - 1],
            "carried_endpoint": asdict(states[end]),
            "terminal_prefix_ordinal": end,
            "pressure_refresh_executed": False,
        })
    terminal = observation.prefixes[-1]
    ordinary_mean_change = _mean(map(Fraction, ordinary)) - _mean(initial.exact_epi)
    endpoint = observation.endpoint
    return {
        "status": "observed", "pressure_schedule": pressure_schedule,
        "observation": asdict(observation), "ordinary_trace": tuple(ordinary_trace),
        "cycles": cycle_records,
        "summary": {
            "accumulated_nodal_mean_area": terminal.mean_nodal_area,
            "visible_mean_change": terminal.mean_visible_change,
            "reconstructed_mean_change": terminal.mean_reconstructed_change,
            "ordinary_mean_change": ordinary_mean_change,
            "endpoint_remainder": endpoint.remainder,
            "mean_endpoint_remainder": _mean(endpoint.remainder),
            "max_endpoint_gap": max(abs(Fraction(visible) - Fraction(reference))
                                    for visible, reference in zip(endpoint.epi, ordinary, strict=True)),
            "ordinary_matches_retained_trace": True,
            "prefix_identities_hold": identities_hold,
            "all_states_in_band": all_states_in_band,
            "exact_nodal_mean_source_is_zero": terminal.mean_nodal_area == 0,
        },
        "carry_reset_between_cycles": False,
        "pressure_recomputed_from_carried_state": False,
        "canonical_pressure_generated_for_carried_state": False,
        "scope": (
            "Two prescribed held-pressure tuples continued through eight arithmetic steps; "
            "once the encoded state changes, later inputs remain counterfactual rather "
            "than freshly realized canonical pressure"
        ),
    }


def analyze_c6_nodal_remainder(parent):
    """Reuse B21 validation once, then compare three fixed pressure schedules."""
    pressure_boxes = analyze_c6_pressure_cells(parent)
    cases = [{
        "mode": case["mode"], "epsilon": case["epsilon"],
        "sequences": {name: _prescribed_sequence(case, name) for name in SEQUENCES},
    } for case in pressure_boxes["cases"]]
    return {
        "cases": cases, "runtime_executed": False, "graph_events_executed": False,
        "detached_record_validation": True, "live_provenance_certified": False,
        "future_bounds_verified": False, "production_integrator_modified": False,
        "empirical_correspondence_tested": False,
        "scope": (
            "Prescribed-input arithmetic counterfactual from three retained B20 preparations "
            "and their B21 trace-equivalent pressure witnesses; no pressure refresh on "
            "carried states, graph execution, event/reset adoption or future runtime claim. "
            "The sequence theorem bounds visible rounding error relative to accumulated "
            "nodal area; it does not remove a nonzero pressure source."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "artifacts/research/c6_winding_phase_kernel.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/research/c6_winding_nodal_remainder.json")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        raise ValueError("derived output must not overwrite its historical input")
    source = args.input.read_bytes()
    parent = json.loads(source)
    provenance = current_git_source_provenance(ROOT, SOURCE_SCOPE)
    report = analyze_c6_nodal_remainder(parent)
    sha, dirty, digest = provenance
    manifest = CoreExperimentManifest(
        claim_id="O3.a-C6-prescribed-nodal-remainder", git_sha=sha,
        source_dirty=dirty, dirty_source_hash=digest,
        versions={"python": platform.python_version(), "numpy": np.__version__},
        graph_construction="Detached B20 null/k1/k3 pressure records and B21 arithmetic witnesses",
        capacity_specification="Prescribed unit capacity, source EPI=.5, retained positive band",
        solver="Eight explicit carried nodal steps and ordinary Euler comparisons at h=1/16",
        result_status=ClaimStatus.DERIVED,
        telemetry=("exact nodal area and carried remainder", "visible versus reconstructed mean",
                   "continuous finite prefix balances", "ordinary Euler endpoint difference"),
        controls=("original and two B21 balanced pressure schedules", "no pressure refresh on changed states",
                  "no default-integrator or graph event changes"),
        artifacts=(str(args.output),),
    )
    manifest.validate_for_admission()
    if current_git_source_provenance(ROOT, SOURCE_SCOPE) != provenance or args.input.read_bytes() != source:
        raise RuntimeError("analysis source or historical input changed during the detached comparison")
    report.update(
        manifest=manifest.to_dict(), source_scope=SOURCE_SCOPE,
        input_evidence={
            "path": str(args.input), "sha256": hashlib.sha256(source).hexdigest(),
            "producer_manifest": parent["manifest"], "producer_source_scope": parent["source_scope"],
            "producer_input_evidence": parent["input_evidence"],
            "pressure_witness_source": "benchmarks.c6_winding_pressure_cells.analyze_c6_pressure_cells",
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_payload(report), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote detached C6 nodal remainder comparison to {args.output}")


if __name__ == "__main__":
    main()
