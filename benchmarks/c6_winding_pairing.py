"""Detached centered-pair audit of the retained two-cycle C6 campaign.

Phase projections and pressure identities are evaluated on existing captures.
They do not authenticate a reflected physical phase state or establish an
invariant production class. No graph or operator word is executed here.
"""

from __future__ import annotations

import argparse
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

from benchmarks.c6_winding_defect_budget import _number  # noqa: E402
from benchmarks.c6_winding_phase_response import (
    BASE_PHASE,
    _phase_readout,
)  # noqa: E402
from benchmarks.c6_winding_rounding_cells import (
    analyze_c6_rounding_report,
)  # noqa: E402
from benchmarks.structural_target_compatibility import _capture  # noqa: E402
from benchmarks.thol_pressure_feedback import _payload  # noqa: E402
from tnfr.mathematics.unified_numerical import np  # noqa: E402
from tnfr.physics._cycle_algebra import c6_pair_sums, laplacian_action  # noqa: E402
from tnfr.physics.coupling_winding import c6_centered_opposite_pairs  # noqa: E402
from tnfr.research.claims import ClaimStatus  # noqa: E402
from tnfr.research.core_manifests import (  # noqa: E402
    CoreExperimentManifest,
    current_git_source_provenance,
)

SOURCE_SCOPE = ("src/tnfr", "benchmarks")
REPRESENTED_PI = Fraction(math.pi)
PAIR_INDICES = ((0, 3), (1, 4), (2, 5))


def _stage(data):
    capture = _capture(data)
    lifted = _phase_readout(capture)["represented_lift"]
    phase = tuple(value / REPRESENTED_PI for value in lifted)
    epi = capture.snapshot.epi
    h_phase = c6_centered_opposite_pairs(phase)
    h_epi = c6_centered_opposite_pairs(epi)
    sigma = tuple(
        actual + laplacian
        for actual, laplacian in zip(
            capture.phase_gradient,
            laplacian_action(phase),
            strict=True,
        )
    )
    weights = dict(capture.normalized_weights)
    if any(capture.snapshot.capacity_gradient) or any(
        capture.snapshot.topology_gradient
    ):
        raise ValueError(
            "retained C6 pairing requires zero capacity and topology gradients"
        )
    terms = {
        "epi_pressure": tuple(
            -Fraction(3, 2) * capture.epi_weight * value for value in h_epi
        ),
        "ideal_phase_pressure": tuple(
            -Fraction(3, 2) * weights["phase"] * value for value in h_phase
        ),
        "phase_realization": tuple(
            weights["phase"] * value for value in c6_pair_sums(sigma)
        ),
        "pressure_assembly": c6_pair_sums(capture.kernel_pressure_defect),
        "stored_pressure_residual": c6_pair_sums(capture.stored_pressure_residual),
    }
    pressure = c6_pair_sums(capture.snapshot.stored_pressure)
    residual = tuple(
        actual - sum((term[i] for term in terms.values()), Fraction(0))
        for i, actual in enumerate(pressure)
    )
    epi_laplacian_residual = tuple(
        actual + Fraction(3, 2) * centered
        for actual, centered in zip(
            c6_pair_sums(capture.snapshot.epi_gradient),
            h_epi,
            strict=True,
        )
    )
    if any(residual) or any(epi_laplacian_residual):
        raise RuntimeError("exact C6 pressure-pair decomposition lost its identity")
    return {
        "phase_lift_radians": lifted,
        "phase_pi_represented": phase,
        "phase_mean_pi": sum(phase, Fraction(0)) / 6,
        "phase_pair_sums_pi": c6_pair_sums(phase),
        "phase_centered_pairs_pi": h_phase,
        "epi": epi,
        "epi_mean": sum(epi, Fraction(0)) / 6,
        "epi_pair_sums": c6_pair_sums(epi),
        "epi_centered_pairs": h_epi,
        "phase_gradient_pairs": c6_pair_sums(capture.phase_gradient),
        "phase_realization_error": sigma,
        "pressure_pairs": pressure,
        "pressure_pair_terms": terms,
        "pressure_pair_identity_residual": residual,
        "epi_pair_identity_residual": epi_laplacian_residual,
        "pressure_is_opposite_paired": not any(pressure),
    }


def _difference(after, before, factor):
    return tuple(
        right - factor * left for left, right in zip(before, after, strict=True)
    )


def _case(retained, rounding):
    parameters = retained["joint_domain_reference"]
    um_factor = 1 - _number(parameters["coupling_phase_factor"])
    il_factor = 1 - Fraction(3, 2) * _number(parameters["coherence_phase_factor"])
    combined = um_factor * il_factor
    initial = _stage(retained["initial_capture"])
    before = initial
    cycles = []
    for cycle, checked_flow in zip(retained["cycles"], rounding["cycles"], strict=True):
        stages = {
            "um_raw": _stage(cycle["um"]["raw_capture"]),
            "um_refresh": _stage(cycle["um"]["after_capture"]),
            "il_raw": _stage(cycle["il"]["raw_capture"]),
            "il_refresh": _stage(cycle["il"]["after_capture"]),
            "flow_refresh": _stage(cycle["after_capture"]),
        }
        h0 = before["phase_centered_pairs_pi"]
        hu = stages["um_refresh"]["phase_centered_pairs_pi"]
        hi = stages["il_refresh"]["phase_centered_pairs_pi"]
        ru, ri = _difference(hu, h0, um_factor), _difference(hi, hu, il_factor)
        full = _difference(hi, h0, combined)
        quotient_residual = tuple(
            total - il_factor * first - second
            for total, first, second in zip(full, ru, ri, strict=True)
        )
        if any(quotient_residual):
            raise RuntimeError(
                "observed phase-pair residuals lost their exact composition"
            )
        epi_start, epi_end = stages["il_refresh"]["epi"], stages["flow_refresh"]["epi"]
        cycles.append(
            {
                "ordinal": cycle["ordinal"],
                "stages": stages,
                "phase_pair_quotient": {
                    "before": h0,
                    "after_um": hu,
                    "after_il": hi,
                    "um_residual": ru,
                    "il_residual": ri,
                    "combined_residual": full,
                    "composition_identity_residual": quotient_residual,
                    "scope": "Residuals against conditional exact centered-pair factors, not a production invariance proof",
                },
                "flow_binding": {
                    "endpoint_bindings": checked_flow["endpoint_bindings"],
                    "mean_budget": checked_flow["mean_budget"],
                    "starts_straddling_half_binade": min(epi_start)
                    < Fraction(1, 2)
                    < max(epi_start),
                    "ends_straddling_half_binade": min(epi_end)
                    < Fraction(1, 2)
                    < max(epi_end),
                },
                "opposite_pressure_lost_by_il_refresh": (
                    stages["um_refresh"]["pressure_is_opposite_paired"]
                    and stages["il_raw"]["pressure_is_opposite_paired"]
                    and not stages["il_refresh"]["pressure_is_opposite_paired"]
                ),
            }
        )
        before = stages["flow_refresh"]
    return {
        "mode": retained["mode"],
        "epsilon": _number(retained["epsilon"]),
        "phase_pair_factors": {"um": um_factor, "il": il_factor, "combined": combined},
        "initial": initial,
        "cycles": cycles,
    }


def analyze_c6_pairing_report(parent):
    """Reuse B18/B17 record and flow checks once before reading pair defects."""
    validated = analyze_c6_rounding_report(parent)
    base = tuple(map(Fraction, BASE_PHASE))
    reflected_pairs = ((0, 3), (1, 2), (4, 5))
    targets = (REPRESENTED_PI, REPRESENTED_PI, 3 * REPRESENTED_PI)
    return {
        "cases": [
            _case(retained, rounding)
            for retained, rounding in zip(
                parent["cases"],
                validated["cases"],
                strict=True,
            )
        ],
        "pair_indices": PAIR_INDICES,
        "phase_convention": {
            "represented_pi": REPRESENTED_PI,
            "stored_base": base,
            "scope": (
                "Wrapped represented radian lifts about stored i*math.pi/3 are divided "
                "by Fraction(math.pi); these are not exact mathematical-pi coordinates"
            ),
        },
        "stored_base_reflection": {
            "pairs": reflected_pairs,
            "affine_targets": targets,
            "residuals": tuple(
                base[i] + base[j] - target
                for (i, j), target in zip(
                    reflected_pairs,
                    targets,
                    strict=True,
                )
            ),
            "scope": "Exact arithmetic on the stored base and represented pi, not a physical conjugation certificate",
        },
        "runtime_executed": False,
        "live_provenance_certified": False,
        "production_pairing_invariance_certified": False,
        "future_mean_bound_verified": False,
        "scope": "Detached retained-stage discrimination; no new word, invariant runtime class or future mean bound",
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
        default=ROOT / "artifacts/research/c6_winding_pairing.json",
    )
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        raise ValueError("derived output must not overwrite input evidence")
    source = args.input.read_bytes()
    parent = json.loads(source)
    provenance = current_git_source_provenance(ROOT, SOURCE_SCOPE)
    report = analyze_c6_pairing_report(parent)
    sha, dirty, digest = provenance
    manifest = CoreExperimentManifest(
        claim_id="O3.a-C6-retained-centered-pairing",
        git_sha=sha,
        source_dirty=dirty,
        dirty_source_hash=digest,
        versions={"python": platform.python_version(), "numpy": np.__version__},
        graph_construction="Detached B16 unit C6 stage captures; no graph or operator execution",
        capacity_specification="Inherited uniform unit capacity and default channel weights",
        solver="Offline exact pair identities with inherited B18 shared-Euler endpoint checks",
        result_status=ClaimStatus.DERIVED,
        telemetry=(
            "centered phase/EPI pairs",
            "pressure-pair decomposition",
            "operator/refresh distinctions",
            "conditional quotient residuals",
            "inherited mean and binade evidence",
        ),
        controls=(
            "retained null/k1/k3",
            "raw and centered pairing distinguished",
            "represented pi explicitly scoped",
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
    print(f"Wrote detached C6 pairing audit to {args.output}")


if __name__ == "__main__":
    main()
