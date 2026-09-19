"""Exact rounding-cell audit of retained C6 held-flow records.

The six existing B16 endpoints are checked through the shared Euler primitive.
No graph, operator word or new physical trajectory is executed. Historical
producer declarations remain distinct from this detached arithmetic result.
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
    _number,
    _vector,
    analyze_c6_defect_report,
)
from benchmarks.thol_pressure_feedback import _payload  # noqa: E402
from tnfr.dynamics._euler_kernel import euler_update  # noqa: E402
from tnfr.mathematics.unified_numerical import np  # noqa: E402
from tnfr.physics.binary64_nodal_flow import (
    observe_binary64_unit_quarter_flow,
)  # noqa: E402
from tnfr.research.claims import ClaimStatus  # noqa: E402
from tnfr.research.core_manifests import (  # noqa: E402
    CoreExperimentManifest,
    current_git_source_provenance,
)

SOURCE_SCOPE = ("src/tnfr", "benchmarks")


def _represented_scalar(value):
    """Decode an exact represented scalar without rounding supplied rationals."""
    exact = _number(value)
    try:
        represented = float(value) if type(value) is float else float(exact)
    except OverflowError as error:
        raise ValueError("recorded scalar exceeds finite binary64 range") from error
    if not math.isfinite(represented) or Fraction(represented) != exact:
        raise ValueError("recorded scalar is not exactly represented in binary64")
    return represented


def _represented_vector(values):
    _vector(values)
    return tuple(_represented_scalar(value) for value in values)


def _numpy_replay(epi, pressure):
    if np is None:
        raise ValueError(
            "the retained NumPy backend must be available for primitive comparison"
        )
    state = np.asarray(epi, dtype=float)
    base = np.multiply(np.ones_like(state), np.asarray(pressure, dtype=float))
    rate = np.add(base, np.zeros_like(base))
    states = []
    for _ in range(4):
        state = euler_update(state, 1 / 16, rate)
        if not np.all(np.isfinite(state)):
            raise ValueError("the detached Euler replay is not finite")
        states.append(tuple(float(value) for value in state))
    return tuple(states)


def _analyze_case(retained, audited):
    lower = _represented_scalar(retained["admission_band"]["positive_epi_lower"])
    upper = _represented_scalar(retained["admission_band"]["epi_upper"])
    cycles = []
    for cycle, defect in zip(retained["cycles"], audited["cycles"], strict=True):
        flow = cycle["flow"]
        before, after = flow["before"], flow["raw_after_integrator"]
        epi = _represented_vector(before["epi"])
        pressure = _represented_vector(before["pressure"])
        endpoint = _represented_vector(after["epi"])
        observation = observe_binary64_unit_quarter_flow(
            epi=epi,
            pressure=pressure,
            epi_lower=lower,
            epi_upper=upper,
        )
        numpy_states = _numpy_replay(epi, pressure)
        bindings = {
            "scalar_endpoint_matches_capture": observation.endpoint == endpoint,
            "exact_endpoint_matches_capture": observation.exact_endpoint
            == tuple(map(Fraction, endpoint)),
            "numpy_endpoint_matches_capture": numpy_states[-1] == endpoint,
            "all_numpy_substeps_match_scalar": numpy_states
            == tuple(step.after for step in observation.substeps),
            "integrator_defect_matches_B17": observation.endpoint_defect
            == defect["pressure_budget"]["integrator_epi_effect"],
        }
        if not all(bindings.values()):
            raise ValueError(
                "retained endpoint or B17 defect differs from shared Euler replay"
            )
        mean_pressure_effect = sum(map(Fraction, pressure), Fraction(0)) / 24
        mean_actual_change = (
            sum(
                (Fraction(a) - Fraction(b) for a, b in zip(endpoint, epi, strict=True)),
                Fraction(0),
            )
            / 6
        )
        mean_rounding_effect = sum(observation.endpoint_defect, Fraction(0)) / 6
        residual = mean_actual_change - mean_pressure_effect - mean_rounding_effect
        if (
            residual
            or mean_rounding_effect != observation.mean_endpoint_defect
            or mean_pressure_effect != observation.mean_pressure / 4
            or mean_actual_change != observation.mean_after - observation.mean_before
        ):
            raise RuntimeError(
                "captured mean change lost its pressure and rounding identity"
            )
        is_half = all(value == 0.5 for value in epi)
        null_stasis = {
            "source_epi_is_uniform_half": is_half,
            "all_pressures_in_half_epi_band": all(
                observation.half_epi_pressure_band_membership
            ),
            "all_epi_coordinates_stay_fixed": all(observation.stasis),
            "captured_epi_stays_fixed": epi == endpoint,
            "phase_stays_fixed_across_operator_cycle": (
                _vector(cycle["before_capture"]["phase"])
                == _vector(cycle["post_il_capture"]["phase"])
            ),
            "future_pressure_band_verified": False,
            "full_state_fixed_point_certified": False,
        }
        cycles.append(
            {
                "ordinal": cycle["ordinal"],
                "observation": asdict(observation),
                "numpy_replay": numpy_states,
                "endpoint_bindings": bindings,
                "mean_budget": {
                    "held_pressure_mean_effect": mean_pressure_effect,
                    "integrator_rounding_mean_effect": mean_rounding_effect,
                    "actual_mean_change": mean_actual_change,
                    "identity_residual": residual,
                    "exact_mean_preserved": mean_actual_change == 0,
                },
                "null_stasis": null_stasis,
                "bound_scope": (
                    "The observer derives its local rounding bound from the declared binary64 "
                    "band and primitive, independently of finite observed error maxima"
                ),
            }
        )
    return {
        "mode": retained["mode"],
        "epsilon": _number(retained["epsilon"]),
        "cycles": cycles,
    }


def analyze_c6_rounding_report(parent):
    """Validate historical continuity before identifying detached numeric steps."""
    validated = analyze_c6_defect_report(parent)
    cases = []
    for retained, audited in zip(parent["cases"], validated["cases"], strict=True):
        cases.append(_analyze_case(retained, audited))
    return {
        "cases": cases,
        "runtime_executed": False,
        "detached_record_validation": True,
        "live_provenance_certified": False,
        "future_bounds_verified": False,
        "full_state_fixed_point_certified": False,
        "scope": (
            "Detached validation of retained records and scalar/NumPy Euler arithmetic; "
            "no new live seal, future phase or mean bound, or full-state fixed point"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "artifacts/research/c6_winding_joint_domain.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/research/c6_winding_rounding_cells.json",
    )
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        raise ValueError("derived output must not overwrite input evidence")
    source = args.input.read_bytes()
    parent = json.loads(source)
    provenance = current_git_source_provenance(ROOT, SOURCE_SCOPE)
    report = analyze_c6_rounding_report(parent)
    sha, dirty, digest = provenance
    manifest = CoreExperimentManifest(
        claim_id="O3.a-C6-held-flow-rounding-cells",
        git_sha=sha,
        source_dirty=dirty,
        dirty_source_hash=digest,
        versions={
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        graph_construction="Detached B16 ordered unit C6 captures; no new graph or word execution",
        capacity_specification="Retained uniform unit capacity; no coefficient changes",
        solver="Detached shared Euler arithmetic: four held-pressure substeps of 1/16",
        result_status=ClaimStatus.DERIVED,
        telemetry=(
            "exact addition cells",
            "scalar/NumPy endpoint agreement",
            "signed pressure and rounding mean contributions",
            "derived local error bounds",
        ),
        controls=(
            "retained null/k1/k3",
            "conditional EPI stasis is not a full-state fixed point",
        ),
        artifacts=(str(args.output),),
    )
    manifest.validate_for_admission()
    if (
        current_git_source_provenance(ROOT, SOURCE_SCOPE) != provenance
        or args.input.read_bytes() != source
    ):
        raise RuntimeError("analysis source or input evidence changed during the audit")
    report.update(
        manifest=manifest.to_dict(),
        source_scope=SOURCE_SCOPE,
        input_evidence={
            "path": str(args.input),
            "sha256": hashlib.sha256(source).hexdigest(),
            "producer_manifest": parent["manifest"],
            "producer_source_scope": parent["source_scope"],
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_payload(report), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote detached C6 rounding cells to {args.output}")


if __name__ == "__main__":
    main()
