"""One regional response criterion on two authenticated retained EN witnesses.

No trajectory or scalar kernel is run. Common map coefficients are admitted
before comparing directions; the realized residual is accounted for separately.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from benchmarks import thol_regional_response_admission as admission  # noqa: E402
from benchmarks.thol_full_state_response import _payload  # noqa: E402
from tnfr.physics._cycle_algebra import dot  # noqa: E402
from tnfr.physics.regional_response import observe_regional_response  # noqa: E402
from tnfr.research.claims import ClaimStatus  # noqa: E402
from tnfr.research.core_manifests import CoreExperimentManifest, current_git_source_provenance  # noqa: E402
from tnfr.utils.io import safe_write  # noqa: E402


def analyze_admitted(admitted):
    """Use saved, separately admitted maps; authenticate files in the caller."""
    reference, common = admitted["original_reference"], admitted["common_coefficients"]
    nodes, metric = reference.source.nodes, reference.metric_weights
    indices = tuple(nodes.index(node) for node in admitted["children"])
    witnesses = {}
    for name, witness in admitted["witnesses"].items():
        if any(witness["paired_source_difference"]):
            raise ValueError("paired source cancellation is required")
        d, parts = witness["delta"], witness["residuals"]
        total = tuple(sum((values[i] for values in parts.values()), F(0)) for i in range(len(nodes)))
        stages = {}
        for stage, matrix, residual, target in (
            ("reception", common["S"], parts["reset"], d["xg"]),
            ("held_pressure_interval", common["T"], total, d["xf"]),
        ):
            expected = tuple(dot(row, d["x0"])+r for row, r in zip(matrix, residual, strict=True))
            if expected != target:
                raise ValueError("admitted map/residual does not reconstruct full endpoint")
            result = observe_regional_response(matrix, metric, indices, d["x0"], residual=residual)
            stages[stage] = asdict(result)
        witnesses[name] = {"stages": stages, "delta": d, "residual_components": parts,
                           "total_residual": total, "source_difference": witness["paired_source_difference"]}
    return {
        "nodes": nodes, "children": admitted["children"], "region_indices": indices,
        "common_coefficients": common, "witnesses": witnesses,
        "native_calls": 0, "kernel_calls": 0, "forcing_capture_calls": 0, "new_trajectories": 0,
        "prospective_binary64_bound_certified": False,
        "autonomous_maintenance_certified": False,
        "scope": "Conditional exact-map criterion, verified on retained binary64 endpoints with measured residuals. "
                 "No forecast evaluation, new intervention, uniform runtime gain, policy derivation or physical identification.",
    }


def _dataclass_payload(value):
    """Keep exact dataclass evidence in the ordinary report serializer."""
    if not is_dataclass(value) or isinstance(value, type):
        raise TypeError(f"Unsupported report value: {type(value).__name__}")
    return _payload(asdict(value))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reset-input", type=Path, default=admission.RESET_PATH)
    parser.add_argument("--restoration-input", type=Path, default=admission.RESTORATION_PATH)
    parser.add_argument("--expected-reset-sha256", default=admission.RESET_SHA256)
    parser.add_argument("--expected-restoration-sha256", default=admission.RESTORATION_SHA256)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/research/thol_regional_response_criterion_2026_09_18.json")
    args = parser.parse_args()
    if args.output.resolve() in (args.reset_input.resolve(), args.restoration_input.resolve()):
        raise ValueError("output must not overwrite retained inputs")
    # Include transitive shared benchmark readers/serializers as well as the
    # engine; their already-retained arithmetic is part of this derivation.
    scope = ("src/tnfr", "benchmarks")
    provenance = current_git_source_provenance(ROOT, scope)
    sha, dirty, digest = provenance
    manifest = CoreExperimentManifest(
        claim_id="O3.a-conditional-regional-response", git_sha=sha, source_dirty=dirty, dirty_source_hash=digest,
        versions={"python": platform.python_version()},
        graph_construction="No graph: two authenticated retained sixteen-node EN intervals",
        capacity_specification="Admitted common positive capacities; original full H=d/nu",
        solver="Exact quadratic and signed input criterion; no solver/kernel execution", timestep=None, seed=None,
        result_status=ClaimStatus.DERIVED, operator_sequence=("reception",),
        telemetry=("regional quadratic response", "shape/input split", "signed realization residual", "nullspace leakage"),
        controls=("common coefficients checked", "existing perturbations only", "no trajectory or kernel calls"),
        artifacts=(str(args.output),))
    manifest.validate_for_admission()
    admitted = admission.load_comparable_witnesses(
        args.reset_input, args.restoration_input, expected_reset_sha256=args.expected_reset_sha256,
        expected_restoration_sha256=args.expected_restoration_sha256)
    result = analyze_admitted(admitted)
    for binding in admitted["historical_inputs"].values():
        if hashlib.sha256(Path(binding["path"]).read_bytes()).hexdigest() != binding["sha256"]:
            raise RuntimeError("retained input changed during analysis")
    if current_git_source_provenance(ROOT, scope) != provenance:
        raise RuntimeError("source changed during retained response analysis")
    report = {"manifest": manifest.to_dict(), "source_scope": scope,
              "historical_inputs": admitted["historical_inputs"], "admission": admitted, **result}
    rendered = json.dumps(_payload(report), default=_dataclass_payload, indent=2, allow_nan=False)+"\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    safe_write(args.output, lambda stream: stream.write(rendered))
    print(f"Wrote conditional regional response to {args.output}")


if __name__ == "__main__":
    main()
