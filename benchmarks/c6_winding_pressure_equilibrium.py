"""Static pressure-equilibrium obstruction for the retained null C6 phase.

The source is the first actual UM/IL endpoint retained by B20. This audit
recomputes its pressure with the shared CPU kernel but executes no graph,
event word or trajectory. B23 did not retain its post-IL phase coordinates;
this report therefore makes no new claim about their causal provenance.
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
from benchmarks.c6_winding_pressure_cells import _validate_parent  # noqa: E402
from benchmarks.c6_winding_rounding_cells import (  # noqa: E402
    _represented_scalar, _represented_vector,
)
from benchmarks.thol_pressure_feedback import _payload  # noqa: E402
from tnfr.config import DEFAULTS  # noqa: E402
from tnfr.dynamics._euler_kernel import initialize_nodal_remainder  # noqa: E402
from tnfr.dynamics.fused_dnfr import compute_fused_gradients_symmetric  # noqa: E402
from tnfr.mathematics.unified_numerical import np  # noqa: E402
from tnfr.physics.binary64_pressure_equilibrium import (  # noqa: E402
    derive_binary64_c6_pressure_equilibrium_obstruction,
)
from tnfr.physics.nodal_remainder import derive_nodal_remainder_cell_horizon  # noqa: E402
from tnfr.research.claims import ClaimStatus  # noqa: E402
from tnfr.research.core_manifests import (  # noqa: E402
    CoreExperimentManifest, current_git_source_provenance,
)
from tnfr.utils import normalize_weights  # noqa: E402

SOURCE_SCOPE = ("src/tnfr", "benchmarks")
SOURCE_LOCATION = "current_cases[0].cycles[0].post_il_capture"


def analyze_c6_pressure_equilibrium(parent):
    """Bind one retained null phase to static lattice and cell theorems."""
    case = _validate_parent(parent)[0]
    analyze_c6_defect_case(case)
    source = case["cycles"][0]["post_il_capture"]
    snapshot = source["snapshot"]
    phase = _represented_vector(source["phase"])
    epi = _represented_vector(snapshot["epi"])
    capacity = _represented_vector(snapshot["capacity"])
    pressure = _represented_vector(snapshot["stored_pressure"])
    weights = {name: _represented_scalar(value) for name, value in source["normalized_weights"]}
    defaults = normalize_weights(DEFAULTS["DNFR_WEIGHTS"], ("phase", "epi", "vf", "topo"))
    if weights != defaults or epi != (.5,) * 6 or capacity != (1.0,) * 6:
        raise ValueError("the selected source must retain default coefficients and uniform half-EPI/unit capacity")
    lower = _represented_scalar(case["admission_band"]["positive_epi_lower"])
    upper = _represented_scalar(case["admission_band"]["epi_upper"])
    if lower != .05 or upper != 1.0:
        raise ValueError("the inherited source must retain the declared [.05,1] EPI band")
    edge_source = np.asarray((0, 0, 1, 2, 3, 4), dtype=np.intp)
    edge_target = np.asarray((1, 5, 2, 3, 4, 5), dtype=np.intp)
    realized = tuple(map(float, compute_fused_gradients_symmetric(
        edge_src=edge_source, edge_dst=edge_target, phase=np.asarray(phase),
        epi=np.asarray(epi), vf=np.asarray(capacity),
        weights={f"w_{name}": value for name, value in weights.items()},
        edge_weight=np.ones(6), use_jit=False,
    )))
    if realized != pressure:
        raise ValueError("the retained source pressure differs from the shared canonical CPU realization")
    obstruction = derive_binary64_c6_pressure_equilibrium_obstruction(
        phase=phase, epi_weight=weights["epi"], phase_weight=weights["phase"],
        epi_lower=lower, epi_upper=upper,
    )
    if tuple(row.phase_contribution for row in obstruction.rows) != pressure:
        raise RuntimeError("uniform EPI source does not equal the independently bound phase contribution")
    rational_sum = sum((row.phase_response.delta_rational for row in obstruction.rows), Fraction(0))
    pi_sum = sum((row.phase_response.delta_pi_coefficient for row in obstruction.rows), Fraction(0))
    if rational_sum or pi_sum:
        raise ValueError("the retained null phase lacks the exact cycle midpoint cancellation")
    initial = initialize_nodal_remainder(epi, epi_lower=lower, epi_upper=upper)
    horizon = derive_nodal_remainder_cell_horizon(
        state=initial, timestep=1 / 16, capacity=capacity, pressure=pressure,
    )
    exact_pressure = tuple(map(Fraction, pressure))
    return {
        "source": {
            "mode": "null", "epsilon": Fraction(0), "location": SOURCE_LOCATION,
            "event_prefix": ("coupling", "coherence"), "physical_time": Fraction(0),
            "phase": phase, "epi": epi, "capacity": capacity,
            "normalized_weights": tuple((name, Fraction(value)) for name, value in weights.items()),
            "pressure": pressure, "pressure_sum": sum(exact_pressure, Fraction(0)),
            "pressure_mean": sum(exact_pressure, Fraction(0)) / 6,
            "shared_cpu_pressure_matches_capture": True,
            "exact_midpoint_sum_rational": rational_sum, "exact_midpoint_sum_pi_coefficient": pi_sum,
            "support": "Ordered unit C6; both actual neighbors participate in phase and EPI pressure",
        },
        "equilibrium_obstruction": asdict(obstruction),
        "fixed_positive_step_convergence_excluded": obstruction.fixed_positive_step_convergence_excluded,
        "conditional_cell_horizon": asdict(horizon),
        "cell_horizon_conditions": (
            "Fixed source phases, unit capacities, unit support and default coefficients; "
            "pressure reads visible EPI, so it remains the recomputed source until the first cell exit. "
            "This conditional held-state horizon is not the B23 schedule with intervening UM/IL events."
        ),
        "runtime_executed": False, "new_graph_trajectories": 0,
        "retained_source_validation": True, "live_provenance_certified": False,
        "B23_phase_identity_certified": False,
        "positive_band_exit_certified": obstruction.positive_band_exit_certified,
        "signed_mean_convergence_decided": False, "full_runtime_stability_certified": False,
        "scope": (
            "Static sufficient absence of a zero-pressure EPI tuple on the fixed retained phase slice, "
            "and an exact conditional carried-state rounding-cell horizon. Empty inverse lattice cells "
            "exclude fixed-positive-step reconstructed convergence on that slice; they do not prove "
            "band exit, signed mean drift, or behavior under future phase-changing events."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "artifacts/research/c6_winding_phase_kernel.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/research/c6_winding_pressure_equilibrium.json")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        raise ValueError("derived output must not overwrite its historical input")
    source = args.input.read_bytes()
    parent = json.loads(source)
    provenance = current_git_source_provenance(ROOT, SOURCE_SCOPE)
    report = analyze_c6_pressure_equilibrium(parent)
    sha, dirty, digest = provenance
    manifest = CoreExperimentManifest(
        claim_id="O3.a-C6-fixed-phase-pressure-equilibrium-obstruction", git_sha=sha,
        source_dirty=dirty, dirty_source_hash=digest,
        versions={"python": platform.python_version(), "numpy": np.__version__},
        graph_construction="Retained B20 null first UM/IL endpoint; static shared CPU pressure only",
        capacity_specification="Unit capacities, unit C6, inherited default weights, EPI band [.05,1]",
        solver="Exact inverse rounding cells and conditional carried-cell horizon; no graph integration",
        result_status=ClaimStatus.DERIVED,
        telemetry=("inverse EPI-gradient lattice cells", "actual canonical phase source",
                   "exact phase midpoint cancellation", "conditional first rounding-cell exit"),
        controls=("one inherited null source", "fixed phase is conditional", "no trajectory extension"),
        artifacts=(str(args.output),),
    )
    manifest.validate_for_admission()
    if current_git_source_provenance(ROOT, SOURCE_SCOPE) != provenance or args.input.read_bytes() != source:
        raise RuntimeError("analysis source or historical input changed during the static audit")
    report.update(
        manifest=manifest.to_dict(), source_scope=SOURCE_SCOPE,
        input_evidence={
            "path": str(args.input), "sha256": hashlib.sha256(source).hexdigest(),
            "producer_manifest": parent["manifest"], "producer_source_scope": parent["source_scope"],
            "producer_input_evidence": parent["input_evidence"], "selected_source": SOURCE_LOCATION,
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_payload(report), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote static C6 pressure-equilibrium audit to {args.output}")


if __name__ == "__main__":
    main()
