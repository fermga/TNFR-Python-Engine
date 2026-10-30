"""Exact environmental-input geometry of two already admitted regional maps.

Read one authenticated conditional-response report. No historical producer,
admission replay, trajectory, scalar kernel or forcing capture is executed.
The declared independent input space is not a reachability certificate.
"""

from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from benchmarks import thol_retained_reset_audit as reset  # noqa: E402
from benchmarks.thol_full_state_response import _payload  # noqa: E402
from benchmarks.thol_regional_response_criterion import _dataclass_payload  # noqa: E402
from tnfr.mathematics.krylov import exact_rank  # noqa: E402
from tnfr.physics._cycle_algebra import dot  # noqa: E402
from tnfr.physics.regional_response import observe_regional_input_geometry  # noqa: E402
from tnfr.research.claims import ClaimStatus  # noqa: E402
from tnfr.research.core_manifests import (
    CoreExperimentManifest,
    current_git_source_provenance,
)  # noqa: E402
from tnfr.utils.io import safe_write  # noqa: E402

INPUT_PATH = (
    ROOT / "artifacts/research/thol_regional_response_criterion_2026_09_18.json"
)
INPUT_SHA256 = "257dea7579a432affe4f068642c05fcf50c0ac3c0d98d0f3d19671e02d05fb5d"
INPUT_CLAIM = "O3.a-conditional-regional-response"
STAGES = (("reception", "S"), ("held_pressure_interval", "T"))
WITNESSES = ("cohort", "localized")


def _vector(value, label):
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be an ordered vector")
    return reset._v(value)


def _matrix(value, rows, columns, label):
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be an ordered matrix")
    result = tuple(_vector(row, label) for row in value)
    if len(result) != rows or any(len(row) != columns for row in result):
        raise ValueError(f"{label} has unexpected dimensions")
    return result


def _equal(actual, expected, label):
    if actual != expected:
        raise ValueError(f"retained {label} mismatch")


def analyze_retained(retained):
    """Classify two maps once and verify every retained witness's geometry.

    Authentication belongs to ``reset._load`` in the CLI. The detached function
    rederives geometry from the declared coefficients; it does not recertify the
    native history that supplied those coefficients in the authenticated input.
    """
    nodes, children = tuple(retained["nodes"]), tuple(retained["children"])
    size = len(nodes)
    if (
        not 2 <= size <= 16
        or len(set(nodes)) != size
        or not children
        or len(set(children)) != len(children)
        or any(child not in nodes for child in children)
    ):
        raise ValueError(
            "a bounded distinct node space and original cohort are required"
        )
    region = tuple(nodes.index(child) for child in children)
    stored_region = tuple(retained["region_indices"])
    if any(type(i) is not int for i in stored_region):
        raise ValueError("retained regional indices must be integers")
    _equal(stored_region, region, "cohort indices")
    admission = retained["admission"]
    reference = admission["original_reference"]
    _equal(tuple(reference["source"]["nodes"]), nodes, "original node order")
    _equal(tuple(admission["children"]), children, "original cohort")
    common = retained["common_coefficients"]
    _equal(common, admission["common_coefficients"], "common coefficient binding")
    metric = _vector(common["metric_weights"], "full metric")
    _equal(
        metric,
        _vector(reference["metric_weights"], "original metric"),
        "original metric",
    )
    if len(metric) != size:
        raise ValueError("full metric must match the node space")
    dt = _vector((common["dt"],), "timestep")[0]
    if dt != F(1, 4):
        raise ValueError("the admitted held-pressure interval is one quarter")
    matrices = {key: _matrix(common[key], size, size, key) for key in ("S", "A", "T")}
    expected_t = tuple(
        tuple(s - dt * a for s, a in zip(sr, ar, strict=True))
        for sr, ar in zip(matrices["S"], matrices["A"], strict=True)
    )
    _equal(matrices["T"], expected_t, "held-pressure identity T=S-hA")
    if set(retained["witnesses"]) != set(WITNESSES):
        raise ValueError("both original retained witnesses are required")
    geometries, summaries = {}, {}
    for stage, key in STAGES:
        geometry = observe_regional_input_geometry(matrices[key], metric, region)
        columns = tuple(
            tuple(row[j] for row in geometry.input_map)
            for j in range(len(geometry.input_labels))
        )
        images = tuple(zip(geometry.input_labels, columns, strict=True))
        for name in WITNESSES:
            stages = retained["witnesses"][name]["stages"]
            if set(stages) != {label for label, _ in STAGES}:
                raise ValueError("both retained stage geometries are required")
            stored = stages[stage]
            _equal(
                _matrix(stored["transition"], size, size, "transition"),
                geometry.transition,
                f"{name}/{stage} transition",
            )
            _equal(
                _vector(stored["metric_weights"], "stage metric"),
                geometry.metric_weights,
                f"{name}/{stage} metric",
            )
            indices = tuple(stored["region_indices"])
            if any(type(i) is not int for i in indices):
                raise ValueError("retained stage indices must be integers")
            _equal(indices, region, f"{name}/{stage} indices")
            _equal(
                _matrix(stored["centering"], len(region), size, "centering"),
                geometry.centering,
                f"{name}/{stage} centering",
            )
            stored_images = tuple(
                (label, _vector(values, "nullspace image"))
                for label, values in stored["nullspace_images"]
            )
            _equal(stored_images, images, f"{name}/{stage} nullspace images")
        geometries[stage] = geometry
        summaries[stage] = {
            "shape_dimension": geometry.shape_dimension,
            "rank": geometry.rank,
            "parent_only_rank": exact_rank(
                tuple(row[1:] for row in geometry.input_map)
            ),
            "protected_dimension": geometry.protected_dimension,
        }
    reception, held = (geometries[label] for label, _ in STAGES)
    difference = tuple(
        tuple(t - s for t, s in zip(tr, sr, strict=True))
        for tr, sr in zip(held.input_map, reception.input_map, strict=True)
    )
    pressure_columns = tuple(
        tuple(
            -dt * dot(row, tuple(dot(ar, vector) for ar in matrices["A"]))
            for row in reception.centering
        )
        for vector in reception.input_basis
    )
    pressure = tuple(
        tuple(column[i] for column in pressure_columns) for i in range(len(region))
    )
    _equal(difference, pressure, "input-map difference -h C A N")
    return {
        "nodes": nodes,
        "children": children,
        "region_indices": region,
        "metric_weights": metric,
        "common_coefficients": common,
        "verified_witnesses": WITNESSES,
        "geometries": geometries,
        "geometry_summaries": summaries,
        "comparison": {
            "input_map_difference": difference,
            "expected_pressure_contribution": pressure,
            "rank_change": held.rank - reception.rank,
            "parent_only_rank_change": summaries["held_pressure_interval"][
                "parent_only_rank"
            ]
            - summaries["reception"]["parent_only_rank"],
            "protected_dimension_change": held.protected_dimension
            - reception.protected_dimension,
        },
        "native_calls": 0,
        "kernel_calls": 0,
        "forcing_capture_calls": 0,
        "new_trajectories": 0,
        "geometry_evaluations": 2,
        "input_reachability_certified": False,
        "repeated_invariance_certified": False,
        "autonomous_maintenance_certified": False,
        "scope": "One-step geometry of two declared common exact maps in the original full metric. "
        "Regional mean and outside coordinates are independent algebraic inputs, not certified "
        "reachable interventions. No new trajectory, scalar kernel, numerical coefficient search, "
        "repeated invariance, future policy or physical-particle identification.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--expected-sha256", default=INPUT_SHA256)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "artifacts/research/thol_regional_input_geometry_2026_09_18.json",
    )
    args = parser.parse_args()
    if args.output.resolve() == args.input.resolve():
        raise ValueError("output must not overwrite retained input")
    scope = ("src/tnfr", "benchmarks")
    provenance = current_git_source_provenance(ROOT, scope)
    sha, dirty, digest = provenance
    manifest = CoreExperimentManifest(
        claim_id="O3.a-regional-environmental-input-geometry",
        git_sha=sha,
        source_dirty=dirty,
        dirty_source_hash=digest,
        versions={"python": platform.python_version()},
        graph_construction="No graph: authenticated original regional response coefficients and cohort",
        capacity_specification="Retained common positive capacities and original full H=d/nu",
        solver="Exact image and H-orthogonal annihilator; no runtime or kernel execution",
        timestep=None,
        seed=None,
        result_status=ClaimStatus.DERIVED,
        operator_sequence=("reception",),
        telemetry=(
            "environmental input rank",
            "protected read-outs",
            "weighted projections",
            "held-pressure attribution",
        ),
        controls=(
            "authenticated prior claim",
            "all retained witness geometries checked",
            "original cohort and coefficients",
        ),
        artifacts=(str(args.output),),
    )
    manifest.validate_for_admission()
    retained, binding = reset._load(args.input, args.expected_sha256, INPUT_CLAIM)
    result = analyze_retained(retained)
    if hashlib.sha256(args.input.read_bytes()).hexdigest() != args.expected_sha256:
        raise RuntimeError("retained input changed during geometry analysis")
    if current_git_source_provenance(ROOT, scope) != provenance:
        raise RuntimeError("source changed during retained geometry analysis")
    report = {
        "manifest": manifest.to_dict(),
        "source_scope": scope,
        "historical_inputs": {"criterion": binding},
        **result,
    }
    rendered = (
        json.dumps(
            _payload(report), default=_dataclass_payload, indent=2, allow_nan=False
        )
        + "\n"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    safe_write(args.output, lambda stream: stream.write(rendered))
    print(f"Wrote regional input geometry to {args.output}")


if __name__ == "__main__":
    main()
