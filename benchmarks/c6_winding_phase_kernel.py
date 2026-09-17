"""Compare certified phase midpoints on retained inputs and the same finite C6 word.

The B16 evidence is read-only. A separate three-case, two-cycle execution uses
the current engine; neither trajectory is relabeled as a future invariant class.
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

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

from benchmarks.c6_winding_defect_budget import (  # noqa: E402
    analyze_c6_defect_case, analyze_c6_defect_report,
)
from benchmarks.c6_winding_joint_domain import CASES, run_c6_joint_case  # noqa: E402
from benchmarks.capacity_feedback import _artifact_payload  # noqa: E402
from benchmarks.c6_winding_rounding_cells import (  # noqa: E402
    _analyze_case as analyze_rounding_case, _represented_scalar,
)
from benchmarks.thol_pressure_feedback import _payload  # noqa: E402
from tnfr.mathematics._phase_midpoint import certified_two_neighbor_phase  # noqa: E402
from tnfr.research.claims import ClaimStatus  # noqa: E402
from tnfr.research.core_manifests import (  # noqa: E402
    CoreExperimentManifest, current_git_source_provenance,
)

SOURCE_SCOPE = ("src/tnfr", "benchmarks")


def _mean(values):
    values = tuple(map(Fraction, values))
    return sum(values, Fraction(0)) / len(values)


def _midpoint_rows(case):
    """Compare the recorded independent IL proposals on their own input tuples."""
    rows = []
    for cycle in case["cycles"]:
        stage = cycle["il"]
        source = stage["before_capture"]
        phases = tuple(_represented_scalar(v) for v in source["phase"])
        endpoint = tuple(map(Fraction, stage["raw_capture"]["phase"]))
        neighbors = source["snapshot"]["support_neighbors"]
        predictions = stage["independent_prediction"]
        if len(predictions) != 6:
            raise ValueError("IL comparison requires six ordered retained proposals")
        for node, proposal in enumerate(predictions):
            phase = proposal["phase"]
            if (proposal["node"] != node
                    or Fraction(phase["theta_before"]) != Fraction(phases[node])
                    or Fraction(phase["theta_after"]) != endpoint[node]):
                raise ValueError("IL proposal does not bind its recorded source and endpoint")
            if len(neighbors[node]) != 2:
                raise ValueError("C6 midpoint comparison requires two support neighbors")
            first, second = (phases[i] for i in neighbors[node])
            center = phases[node] % math.tau  # The actual IL input convention.
            certified = certified_two_neighbor_phase(center, first, second)
            gradient = _represented_scalar(source["phase_gradient"][node])
            delta = _represented_scalar(phase["delta_theta"])
            mean = _represented_scalar(phase["theta_network"])
            # Pressure receives the stored center, whereas IL first normalizes it.
            pressure_certificate = certified_two_neighbor_phase(phases[node], first, second)
            rows.append({
                "ordinal": cycle["ordinal"], "node": node,
                "stored_center": Fraction(phases[node]), "il_center": Fraction(center),
                "neighbors": (Fraction(first), Fraction(second)),
                "recorded_delta": Fraction(delta), "recorded_mean": Fraction(mean),
                "recorded_phase_gradient": Fraction(gradient),
                "midpoint": asdict(certified) if certified is not None else None,
                "delta_matches_certified_midpoint": (
                    delta == certified.delta if certified is not None else None
                ),
                "mean_matches_certified_midpoint": (
                    mean == certified.mean if certified is not None else None
                ),
                "pressure_matches_certified_midpoint": (
                    gradient == pressure_certificate.delta / math.pi
                    if pressure_certificate is not None else None
                ),
                "recorded_method": phase.get("method", "historical_phasor_pipeline"),
            })
    return rows


def _summary(case, rounding):
    rows = _midpoint_rows(case)
    return {
        "mode": case["mode"], "epsilon": Fraction(case["epsilon"]),
        "il_midpoint_rows": rows,
        "eligible_il_rows": sum(row["midpoint"] is not None for row in rows),
        "different_il_deltas": sum(row["delta_matches_certified_midpoint"] is False for row in rows),
        "different_il_means": sum(row["mean_matches_certified_midpoint"] is False for row in rows),
        "different_phase_gradients": sum(row["pressure_matches_certified_midpoint"] is False for row in rows),
        "initial_epi_mean": _mean(case["initial_capture"]["snapshot"]["epi"]),
        "final_epi_mean": _mean(case["final_capture"]["snapshot"]["epi"]),
        "total_epi_mean_change": (
            _mean(case["final_capture"]["snapshot"]["epi"])
            - _mean(case["initial_capture"]["snapshot"]["epi"])
        ),
        "flow_mean_budgets": [cycle["mean_budget"] for cycle in rounding["cycles"]],
        "post_il_pressure_means": [
            _mean(cycle["post_il_capture"]["snapshot"]["stored_pressure"])
            for cycle in case["cycles"]
        ],
        "phase_write_at_represented_tau": [
            tuple(i for i, value in enumerate(cycle["post_il_capture"]["phase"])
                  if Fraction(value) == Fraction(math.tau))
            for cycle in case["cycles"]
        ],
        "scope": "Finite tuple and signed mean observations; no summability or future bound",
    }


def compare_c6_phase_kernel(parent):
    """Validate old records once, then rerun exactly their registered preparations."""
    old_audits = analyze_c6_defect_report(parent)
    old_summaries = []
    for case, audit in zip(parent["cases"], old_audits["cases"], strict=True):
        old_summaries.append(_summary(case, analyze_rounding_case(case, audit)))
    current_cases, current_summaries = [], []
    for control, old in zip(CASES, parent["cases"], strict=True):
        case = _artifact_payload({"cases": [run_c6_joint_case(*control)]})["cases"][0]
        # Pressure changes are expected; preparation, coefficients and horizon are fixed.
        if case["initial_capture"]["phase"] != old["initial_capture"]["phase"]:
            raise RuntimeError("the kernel comparison changed its initial phase preparation")
        for name in ("epi", "capacity", "conductance", "support_neighbors"):
            if case["initial_capture"]["snapshot"][name] != old["initial_capture"]["snapshot"][name]:
                raise RuntimeError("the kernel comparison changed its initial structural preparation")
        if (case["joint_domain_reference"] != old["joint_domain_reference"]
                or case["cycle_count"] != old["cycle_count"]):
            raise RuntimeError("the kernel comparison changed its coefficients or duration")
        if (any(case["initial"][name] != old["initial"][name]
                for name in ("configured_controls", "random_provenance"))
                or case["initial"]["state"]["time"] != old["initial"]["state"]["time"]
                or case["word"]["names"] != old["word"]["names"]):
            raise RuntimeError("the kernel comparison changed its controls, seed, initial clock or word")
        audit = analyze_c6_defect_case(case)
        rounding = analyze_rounding_case(case, audit)
        summary = _summary(case, rounding)
        if summary["different_il_deltas"] or summary["different_il_means"] or summary["different_phase_gradients"]:
            raise RuntimeError("eligible current IL or pressure did not use the shared midpoint")
        current_cases.append(case)
        current_summaries.append(summary)
    return {
        "retained": old_summaries, "current": current_summaries,
        "current_cases": current_cases,
        "runtime_executed": True, "cycle_count_per_case": 2,
        "future_mean_bound_verified": False, "production_invariant_class_certified": False,
        "empirical_correspondence_tested": False,
        "scope": (
            "Old source tuples remain historical observations. The current default engine "
            "executes the same null/k1/k3 two-cycle controls with separate source provenance. "
            "Correct midpoint arithmetic does not certify UM, phase writes, full pressure "
            "assembly, refreshed mean conservation or future admission."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "artifacts/research/c6_winding_joint_domain.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/research/c6_winding_phase_kernel.json")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        raise ValueError("comparison output must not overwrite historical input")
    source = args.input.read_bytes()
    parent = json.loads(source)
    provenance = current_git_source_provenance(ROOT, SOURCE_SCOPE)
    report = compare_c6_phase_kernel(parent)
    sha, dirty, digest = provenance
    manifest = CoreExperimentManifest(
        claim_id="O3.a-C6-certified-phase-midpoint-comparison", git_sha=sha,
        source_dirty=dirty, dirty_source_hash=digest,
        versions={"python": platform.python_version(), "numpy": np.__version__, "networkx": nx.__version__},
        graph_construction="Same unit C6, stored winding i*pi/3, null/+k1/+k3 at2^-12",
        capacity_specification="Unchanged unit capacity, initial EPI=.5 and default coefficients",
        solver="Same default Euler, two h=.25 intervals with four held-pressure substeps each",
        timestep=.25, seed=17, result_status=ClaimStatus.MEASURED,
        operator_sequence=("all-target UM IL", "Euler .25", "all-target UM IL", "Euler .25", "terminal SHA"),
        telemetry=("certified midpoint versus recorded IL argument", "shared pressure realization",
                   "signed pressure and rounding mean budgets", "same preparation and finite admission"),
        controls=("historical B16 tuples", "same null/k1/k3 current word", "true pi versus represented wrap"),
        artifacts=(str(args.output),),
    )
    manifest.validate_for_admission()
    if current_git_source_provenance(ROOT, SOURCE_SCOPE) != provenance or args.input.read_bytes() != source:
        raise RuntimeError("comparison source or historical input changed during execution")
    report.update(
        manifest=manifest.to_dict(), source_scope=SOURCE_SCOPE,
        input_evidence={"path": str(args.input), "sha256": hashlib.sha256(source).hexdigest(),
                        "producer_manifest": parent["manifest"], "producer_source_scope": parent["source_scope"]},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_payload(report), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote bounded phase-kernel comparison to {args.output}")


if __name__ == "__main__":
    main()
