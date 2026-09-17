"""Detached pressure boxes for the six retained B20 C6 Euler traces.

Each box is maximal for its complete four-substep numeric trace, not for its
endpoint alone. Balanced pressure witnesses are arithmetic comparisons; no
graph, operator word or canonical pressure-production path is executed.
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

from benchmarks.c6_winding_defect_budget import (  # noqa: E402
    _number, _vector, analyze_c6_defect_case,
)
from benchmarks.c6_winding_joint_domain import CASES  # noqa: E402
from benchmarks.c6_winding_rounding_cells import (  # noqa: E402
    _analyze_case, _represented_scalar, _represented_vector,
)
from benchmarks.thol_pressure_feedback import _payload  # noqa: E402
from tnfr.mathematics.unified_numerical import np  # noqa: E402
from tnfr.physics._cycle_algebra import c6_pair_sums  # noqa: E402
from tnfr.physics.binary64_nodal_flow import (  # noqa: E402
    derive_binary64_quarter_pressure_box, observe_binary64_unit_quarter_flow,
)
from tnfr.research.claims import ClaimStatus  # noqa: E402
from tnfr.research.core_manifests import (  # noqa: E402
    CoreExperimentManifest, current_git_source_provenance,
)

SOURCE_SCOPE = ("src/tnfr", "benchmarks")
PAIR_INDICES = ((0, 3), (1, 4), (2, 5))
PARENT_CLAIM = "O3.a-C6-certified-phase-midpoint-comparison"


def _mean_budget(flow):
    actual = flow.mean_after - flow.mean_before
    pressure = flow.mean_pressure / 4
    scaling, addition = flow.mean_scaling_defect, flow.mean_addition_defect
    residual = actual - pressure - scaling - addition
    if residual:
        raise RuntimeError("held flow lost its signed pressure/scaling/addition mean identity")
    return {
        "held_pressure_mean_effect": pressure,
        "scaling_mean_effect": scaling,
        "addition_mean_effect": addition,
        "actual_mean_change": actual,
        "identity_residual": residual,
        "exact_mean_preserved": actual == 0,
    }


def _pairs(values):
    return c6_pair_sums(tuple(map(Fraction, values)))


def _substep_pairs(flow):
    return tuple({
        "ordinal": step.ordinal,
        "before": _pairs(step.before),
        "after": _pairs(step.after),
        "change": tuple(after - before for after, before in zip(
            _pairs(step.after), _pairs(step.before), strict=True,
        )),
        "addition_error": c6_pair_sums(step.addition_error),
    } for step in flow.substeps)


def _witness(box, pressure):
    if not box.contains(pressure):
        raise RuntimeError("constructed diagnostic pressure is outside the certified trace box")
    original = box.flow
    flow = observe_binary64_unit_quarter_flow(
        epi=original.epi, pressure=pressure,
        epi_lower=original.epi_lower, epi_upper=original.epi_upper,
    )
    trace = tuple(step.after for step in flow.substeps)
    same_trace = trace == tuple(step.after for step in original.substeps)
    same_mean = flow.mean_after == original.mean_after
    if not same_trace or not same_mean:
        raise RuntimeError("box member failed to reproduce the complete retained Euler trace")
    exact_pressure = tuple(map(Fraction, pressure))
    return {
        "status": "feasible",
        "pressure": pressure,
        "pressure_hex": tuple(value.hex() for value in pressure),
        "pressure_sum": sum(exact_pressure, Fraction(0)),
        "exact_zero_sum": sum(exact_pressure, Fraction(0)) == 0,
        "pressure_pair_sums": c6_pair_sums(exact_pressure),
        "max_pressure_change": max(abs(actual - source) for actual, source in zip(
            exact_pressure, original.exact_pressure, strict=True,
        )),
        "minimum_pressure_margin": min(min(value - coordinate.pressure_lower,
                                           coordinate.pressure_upper - value)
                                       for value, coordinate in zip(
                                           exact_pressure, box.coordinates, strict=True,
                                       )),
        "substeps": trace,
        "substep_pair_sums": _substep_pairs(flow),
        "endpoint": flow.endpoint,
        "same_trace": same_trace,
        "same_mean": same_mean,
        "mean_budget": _mean_budget(flow),
        "canonical_pressure_generated": False,
    }


def _zero_sum_witness(box):
    """Try the fixed six single-coordinate exact compensations in node order."""
    pressure = box.flow.pressure
    total = sum(map(Fraction, pressure), Fraction(0))
    for node, coordinate in enumerate(box.coordinates):
        exact = Fraction(pressure[node]) - total
        try:
            candidate = float(exact)
        except OverflowError:
            continue
        if (not math.isfinite(candidate) or Fraction(candidate) != exact
                or not coordinate.contains(candidate)):
            continue
        values = tuple(candidate if index == node else value
                       for index, value in enumerate(pressure))
        witness = _witness(box, values)
        if not witness["exact_zero_sum"]:
            raise RuntimeError("single-coordinate exact compensation failed to balance pressure")
        return {
            **witness, "adjusted_node": node, "recorded_pressure_sum": total,
            "construction": "Lowest node whose exact p_i-sum(p) is a represented box member",
        }
    return {
        "status": "no_single_coordinate_witness",
        "construction": "All six exact single-coordinate compensations were checked",
        "general_zero_sum_feasibility_decided": False,
        "canonical_pressure_generated": False,
    }


def _opposite_pair_witness(box):
    """Intersect represented intervals, then clamp the first recorded pressure."""
    intervals, first_values = [], []
    feasible = True
    for first, second in PAIR_INDICES:
        left, right = box.coordinates[first], box.coordinates[second]
        lower = max(Fraction(left.first_pressure), -Fraction(right.last_pressure))
        upper = min(Fraction(left.last_pressure), -Fraction(right.first_pressure))
        nonempty = lower <= upper
        intervals.append({
            "nodes": (first, second), "lower": lower, "upper": upper,
            "represented_endpoints_inclusive": True, "feasible": nonempty,
        })
        feasible = feasible and nonempty
        if nonempty:
            # All three choices are represented binary64 values: the source
            # or a signed inclusive endpoint. This is not a parameter search.
            selected = min(max(Fraction(box.flow.pressure[first]), lower), upper)
            represented = float(selected)
            if Fraction(represented) != selected:
                raise RuntimeError("represented pressure intersection lost an exact endpoint")
            first_values.append(represented)
    if not feasible:
        return {
            "status": "infeasible", "pair_intervals": intervals,
            "canonical_pressure_generated": False,
            "scope": "At least one represented opposite-pair interval intersection is empty",
        }
    pressure = tuple(first_values + [-value for value in first_values])
    witness = _witness(box, pressure)
    if not witness["exact_zero_sum"] or any(witness["pressure_pair_sums"]):
        raise RuntimeError("opposite-pair completion failed its exact algebraic constraints")
    return {
        **witness, "pair_intervals": intervals,
        "construction": "Clamp p_i to I_i intersect -I_(i+3), then set p_(i+3)=-p_i",
    }


def _validate_parent(parent):
    manifest = CoreExperimentManifest(**parent["manifest"])
    manifest.validate_for_admission()
    if manifest.claim_id != PARENT_CLAIM:
        raise ValueError("input must be the retained B20 phase-kernel comparison")
    if (parent["runtime_executed"] is not True
            or type(parent["cycle_count_per_case"]) is not int
            or parent["cycle_count_per_case"] != 2
            or any(parent[key] is not False for key in (
                "future_mean_bound_verified", "production_invariant_class_certified",
                "empirical_correspondence_tested",
            ))):
        raise ValueError("the parent must retain its finite two-cycle runtime scope")
    if (type(manifest.seed) is not int or manifest.seed != 17
            or _number(manifest.timestep) != Fraction(1, 4)
            or tuple(manifest.operator_sequence) != (
                "all-target UM IL", "Euler .25", "all-target UM IL", "Euler .25", "terminal SHA",
            ) or tuple(manifest.controls) != (
                "historical B16 tuples", "same null/k1/k3 current word", "true pi versus represented wrap",
            ) or tuple(parent["source_scope"]) != SOURCE_SCOPE):
        raise ValueError("the parent manifest differs from the declared B20 controls")
    cases = parent["current_cases"]
    if tuple((case["mode"], _number(case["epsilon"])) for case in cases) != CASES:
        raise ValueError("input requires exactly the ordered null/k1/k3 preparations")
    initial_controls = cases[0]["initial"]["configured_controls"]
    for case in cases:
        controls = case["initial"]["configured_controls"]
        if (controls != initial_controls or controls["RANDOM_SEED"] != 17
                or controls["INTEGRATOR_METHOD"] != "euler"
                or _number(controls["DT_MIN"]) != Fraction(1, 16)
                or _number(case["initial"]["state"]["time"]) != 0
                or _vector(case["initial_capture"]["snapshot"]["epi"]) != (Fraction(1, 2),) * 6):
            raise ValueError("the current cases do not retain their common initial numeric controls")
    return cases


def analyze_c6_pressure_cells(parent):
    """Validate retained B20 records once and audit their finite trace boxes."""
    cases = []
    for case in _validate_parent(parent):
        audited = analyze_c6_defect_case(case)
        rounding = _analyze_case(case, audited)
        lower = _represented_scalar(case["admission_band"]["positive_epi_lower"])
        upper = _represented_scalar(case["admission_band"]["epi_upper"])
        cycles = []
        for retained, measured in zip(case["cycles"], rounding["cycles"], strict=True):
            source = retained["flow"]["before"]
            box = derive_binary64_quarter_pressure_box(
                epi=_represented_vector(source["epi"]),
                pressure=_represented_vector(source["pressure"]),
                epi_lower=lower, epi_upper=upper,
            )
            if asdict(box.flow) != measured["observation"]:
                raise RuntimeError("pressure box does not bind the independently audited recorded flow")
            cycles.append({
                "ordinal": retained["ordinal"], "pressure_box": asdict(box),
                "endpoint_bindings": measured["endpoint_bindings"],
                "mean_budget": _mean_budget(box.flow),
                "recorded_pressure_pairs": c6_pair_sums(box.flow.exact_pressure),
                "recorded_substep_pair_sums": _substep_pairs(box.flow),
                "zero_sum_witness": _zero_sum_witness(box),
                "opposite_pair_witness": _opposite_pair_witness(box),
            })
        cases.append({"mode": case["mode"], "epsilon": _number(case["epsilon"]), "cycles": cycles})
    return {
        "cases": cases, "runtime_executed": False,
        "detached_record_validation": True, "live_provenance_certified": False,
        "future_bounds_verified": False, "canonical_pressure_witnesses_generated": False,
        "scope": (
            "Maximal binary64 pressure boxes for six complete retained four-substep traces; "
            "not maximal endpoint classes, newly executed graph trajectories, canonical "
            "forcing generation, or future mean/runtime certificates"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "artifacts/research/c6_winding_phase_kernel.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/research/c6_winding_pressure_cells.json")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        raise ValueError("derived output must not overwrite its historical input")
    source = args.input.read_bytes()
    parent = json.loads(source)
    provenance = current_git_source_provenance(ROOT, SOURCE_SCOPE)
    report = analyze_c6_pressure_cells(parent)
    sha, dirty, digest = provenance
    manifest = CoreExperimentManifest(
        claim_id="O3.a-C6-retained-pressure-trace-boxes", git_sha=sha,
        source_dirty=dirty, dirty_source_hash=digest,
        versions={"python": platform.python_version(), "numpy": np.__version__},
        graph_construction="Detached B20 null/k1/k3 unit-C6 two-cycle records; no graph execution",
        capacity_specification="Retained unit capacity and positive EPI band; no coefficient changes",
        solver="Inverse cells and shared scalar Euler replay of four held-pressure substeps of 1/16",
        result_status=ClaimStatus.DERIVED,
        telemetry=("maximal full-trace pressure boxes", "signed pressure/scaling/addition mean budgets",
                   "exact zero-sum and opposite-pair diagnostic witnesses", "all four replayed substeps"),
        controls=("three retained preparations and six intervals", "no canonical witness generation",
                  "full-trace equivalence is not an endpoint-only maximal class"),
        artifacts=(str(args.output),),
    )
    manifest.validate_for_admission()
    if current_git_source_provenance(ROOT, SOURCE_SCOPE) != provenance or args.input.read_bytes() != source:
        raise RuntimeError("analysis source or historical input changed during the detached audit")
    report.update(
        manifest=manifest.to_dict(), source_scope=SOURCE_SCOPE,
        input_evidence={
            "path": str(args.input), "sha256": hashlib.sha256(source).hexdigest(),
            "producer_manifest": parent["manifest"], "producer_source_scope": parent["source_scope"],
            "producer_input_evidence": parent["input_evidence"],
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_payload(report), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote detached C6 pressure boxes to {args.output}")


if __name__ == "__main__":
    main()
