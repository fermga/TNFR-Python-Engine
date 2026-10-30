"""Offline regional paired response versus control flattening on six fixed records.

This audit neither evolves a graph nor certifies NFR maintenance. The declared
reference is the already recorded evolving control, with its mean and centered
form retained separately. No regional score or tolerance is fitted to outcomes.
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
from benchmarks.thol_family_closure import _equal  # noqa: E402
from benchmarks.thol_full_state_response import _paired_delta, _payload  # noqa: E402
from benchmarks.thol_regional_balance_audit import (
    _ancestry,
    _bind_source_edges,
)  # noqa: E402
from benchmarks.thol_regional_identity_audit import _record_ref  # noqa: E402
from tnfr.physics.support_transport import _rebuild  # noqa: E402
from tnfr.research.claims import ClaimStatus  # noqa: E402
from tnfr.research.core_manifests import (
    CoreExperimentManifest,
    current_git_source_provenance,
)  # noqa: E402
from tnfr.utils.io import safe_write  # noqa: E402

TIMES = tuple(F(7 + i, 4) for i in range(6))
PROTOCOL = {
    "times": TIMES,
    "regions": "All eight actual ancestry pairs, then their complete child cohort",
    "reference": "Authenticated evolving control at each same-time endpoint; no fitted target",
    "metric": "Fixed original full-graph H=d/nu, restricted without induced renormalization",
    "ratio": "Centered paired error / control variance, only for positive control variance",
    "historical_global_reduction": "legacy",
}


def _ratio(error, variance):
    if error < 0 or variance < 0:
        raise ValueError("energies must be nonnegative")
    return {
        "available": variance > 0,
        "value": error / variance if variance > 0 else None,
        "zero_case": (
            None
            if variance > 0
            else (
                "zero_control_and_zero_error"
                if error == 0
                else "zero_control_with_positive_error"
            )
        ),
    }


def _endpoints(retained, ordinal):
    pool = retained["record_pool"]
    branch = retained["branches"][ordinal]
    first = reset.window.expand_record(retained["first_step_replays"][ordinal], pool)
    if (
        branch["branch"] != reset.window.BRANCHES[ordinal]
        or first["branch"] != branch["branch"]
        or first["status"] != "executed"
        or branch["status"] != "completed"
        or branch["completed_steps"] != 5
        or len(branch["steps"]) != 5
    ):
        raise ValueError("both original completed five-step branches are required")
    records = [first["endpoint"]]
    paths = [f"first_step_replays[{ordinal}].endpoint"]
    for index, reference in enumerate(branch["steps"]):
        step = reset.window.expand_record(reference, pool)
        if step["ordinal"] != index or step["status"] != "executed":
            raise ValueError("complete ordered window steps required")
        _equal(step["before"], records[-1], "complete adjacent endpoint")
        records.append(step["endpoint"])
        paths.append(f"branches[{ordinal}].steps[{index}].endpoint")
    _equal(
        records[-1],
        reset.window.expand_record(branch["final_record"], pool),
        "final window endpoint",
    )
    if tuple(F(row["state"]["time"]) for row in records) != TIMES:
        raise ValueError("the six fixed physical endpoints are required")
    return tuple(records), tuple(
        _record_ref(path, record) for path, record in zip(paths, records, strict=True)
    )


def audit_window(retained, original, native_report):
    """Pure detached admission and regional arithmetic; public records are unsealed."""
    if (
        len(retained["branches"]) != 2
        or len(native_report["replayed_prior_reports"]) != 2
    ):
        raise ValueError("exactly the original paired branches are required")
    source = _rebuild(original.source)
    nodes = source.nodes
    strengths = tuple(
        sum((w for left, _right, w in source.conductance if left == i), F(0))
        for i in range(len(nodes))
    )
    if any(d <= 0 or nu <= 0 for d, nu in zip(strengths, source.capacity, strict=True)):
        raise ValueError("positive full strengths and capacities are required")
    metric = tuple(d / nu for d, nu in zip(strengths, source.capacity, strict=True))
    _equal(metric, original.metric_weights, "original full-graph metric")
    branches, references, pairs = [], [], None
    for ordinal in range(2):
        records, refs = _endpoints(retained, ordinal)
        prior = native_report["replayed_prior_reports"][ordinal]
        if prior["branch"] != reset.window.BRANCHES[ordinal]:
            raise ValueError("native prior branch order differs")
        for record in records:
            state = record["state"]
            if (
                tuple(state["nodes"]) != nodes
                or reset._v(state["capacity"]) != source.capacity
            ):
                raise ValueError(
                    "regional endpoints require unchanged ordered nodes and capacity"
                )
            _bind_source_edges(state, source)
            values = reset._v(state["epi"])
            if len(values) != len(nodes):
                raise ValueError("complete finite EPI coordinates required")
            current_pairs, _parents, children = _ancestry(prior, state, nodes)
            if pairs is not None and current_pairs != pairs:
                raise ValueError(
                    "regional ancestry changed between endpoints or branches"
                )
            pairs = current_pairs
        branches.append(records)
        references.append(refs)
    regions = tuple(
        [(f"ancestry_pair_{i}", pair) for i, pair in enumerate(pairs)]
        + [("actual_children", children)]
    )
    result = []
    for label, region in regions:
        indices = tuple(nodes.index(node) for node in region)
        weights = tuple(metric[i] for i in indices)
        zeros = (F(0),) * len(region)
        initial_control = tuple(F(branches[0][0]["state"]["epi"][i]) for i in indices)
        rows = []
        for time, control, perturbed in zip(TIMES, *branches, strict=True):
            x = tuple(F(control["state"]["epi"][i]) for i in indices)
            y = tuple(F(perturbed["state"]["epi"][i]) for i in indices)
            paired = _paired_delta(x, y, weights)
            shape = _paired_delta(zeros, x, weights)
            drift = _paired_delta(initial_control, x, weights)
            variance = shape["centered_H_energy"]
            rows.append(
                {
                    "time": time,
                    "control_epi": x,
                    "perturbed_epi": y,
                    "paired": paired,
                    "control_mean": shape["weighted_mean_offset"],
                    "control_variance": variance,
                    "control_centered_epi": shape["centered_epi_difference"],
                    "control_centered_drift_from_start": drift[
                        "centered_epi_difference"
                    ],
                    "control_centered_drift_energy": drift["centered_H_energy"],
                    "control_mean_drift_from_start": drift["weighted_mean_offset"],
                    "relative_error": _ratio(paired["centered_H_energy"], variance),
                }
            )
        first, last = rows[0], rows[-1]
        e0, e1 = (row["paired"]["centered_H_energy"] for row in (first, last))
        v0, v1 = first["control_variance"], last["control_variance"]
        ratios_available = (
            first["relative_error"]["available"] and last["relative_error"]["available"]
        )
        result.append(
            {
                "label": label,
                "nodes": region,
                "full_node_indices": indices,
                "metric_weights": weights,
                "endpoints": rows,
                "endpoint_summary": {
                    "centered_error_change": e1 - e0,
                    "centered_error_decreased": e1 < e0,
                    "initial_centered_error_nonzero": e0 > 0,
                    "control_variance_change": v1 - v0,
                    "control_variance_decreased": v1 < v0,
                    "control_variance_positive_at_every_endpoint": all(
                        row["control_variance"] > 0 for row in rows
                    ),
                    "control_centered_form_unchanged_at_every_endpoint": all(
                        row["control_centered_drift_energy"] == 0 for row in rows
                    ),
                    "relative_error_change": (
                        last["relative_error"]["value"]
                        - first["relative_error"]["value"]
                        if ratios_available
                        else None
                    ),
                    "relative_error_decreased": (
                        last["relative_error"]["value"]
                        < first["relative_error"]["value"]
                        if ratios_available
                        else None
                    ),
                },
            }
        )
    return {
        "protocol": PROTOCOL,
        "nodes": nodes,
        "full_metric_weights": metric,
        "actual_lineage": {"pairs": pairs, "children": children},
        "retained_endpoints": references,
        "regions": result,
        "regional_endpoint_count": 54,
        "native_calls": 0,
        "coordination_calls": 0,
        "forcing_capture_calls": 0,
        "autonomous_maintenance_certified": False,
        "scope": "Finite legacy-runtime regional response toward a predeclared evolving control. "
        "Raw error, control contrast, control-form drift and mean remain separate. "
        "Attenuation can follow shared transport/reset policies; neither a falling ratio nor nonzero "
        "control variance proves restored identity, source feedback, future stability or engine-state return.",
    }


def run_study(
    window_path=reset.WINDOW_PATH,
    native_path=reset.window.NATIVE_PATH,
    *,
    expected_window_sha256=reset.WINDOW_SHA256,
    expected_native_sha256=reset.window.NATIVE_SHA256,
):
    retained, original, bindings = reset.load_evidence(
        window_path,
        native_path,
        expected_window_sha256=expected_window_sha256,
        expected_native_sha256=expected_native_sha256,
    )
    native, _ = reset._load(
        native_path, expected_native_sha256, "O1.b-generated-native-runtime-response"
    )
    result = audit_window(retained, original, native)
    for binding in bindings.values():
        if (
            hashlib.sha256(Path(binding["path"]).read_bytes()).hexdigest()
            != binding["sha256"]
        ):
            raise RuntimeError("retained regional response input changed during audit")
    return {"historical_inputs": bindings, **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-input", type=Path, default=reset.WINDOW_PATH)
    parser.add_argument("--native-input", type=Path, default=reset.window.NATIVE_PATH)
    parser.add_argument("--expected-window-sha256", default=reset.WINDOW_SHA256)
    parser.add_argument("--expected-native-sha256", default=reset.window.NATIVE_SHA256)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "artifacts/research/thol_regional_recovery_audit_2026_09_18.json",
    )
    args = parser.parse_args()
    if args.output.resolve() in {
        args.window_input.resolve(),
        args.native_input.resolve(),
    }:
        raise ValueError("output must not overwrite retained inputs")
    scope = (
        "src/tnfr",
        "benchmarks/thol_regional_recovery_audit.py",
        "benchmarks/thol_retained_reset_audit.py",
        "benchmarks/thol_native_policy_window.py",
        "benchmarks/thol_native_runtime_response.py",
        "benchmarks/thol_full_state_response.py",
        "benchmarks/thol_regional_balance_audit.py",
        "benchmarks/thol_regional_identity_audit.py",
        "benchmarks/thol_family_closure.py",
        "benchmarks/thol_pressure_feedback.py",
    )
    provenance = current_git_source_provenance(ROOT, scope)
    result = run_study(
        args.window_input,
        args.native_input,
        expected_window_sha256=args.expected_window_sha256,
        expected_native_sha256=args.expected_native_sha256,
    )
    if current_git_source_provenance(ROOT, scope) != provenance:
        raise RuntimeError("working source changed during regional recovery audit")
    sha, dirty, digest = provenance
    manifest = CoreExperimentManifest(
        claim_id="O3.a-retained-regional-paired-response",
        git_sha=sha,
        source_dirty=dirty,
        dirty_source_hash=digest,
        versions={"python": platform.python_version()},
        graph_construction="Authenticated existing paired sixteen-node THOL native-window endpoints",
        capacity_specification="Fixed complete support and positive capacity; original full-graph metric",
        solver="Exact detached regional comparisons; no solver or kernel executed",
        timestep=None,
        seed=None,
        result_status=ClaimStatus.DERIVED,
        operator_sequence=(),
        telemetry=(
            "regional paired error",
            "control contrast and centered drift",
            "separate mean offset",
        ),
        controls=(
            "six predeclared times",
            "all nine lineage regions",
            "zero denominator explicit",
            "no new trajectories",
        ),
        artifacts=(str(args.output),),
    )
    manifest.validate_for_admission()
    report = {"manifest": manifest.to_dict(), "source_scope": scope, **result}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    safe_write(
        args.output,
        lambda stream: stream.write(
            json.dumps(_payload(report), indent=2, allow_nan=False) + "\n"
        ),
    )
    print(f"Wrote retained regional paired response to {args.output}")


if __name__ == "__main__":
    main()
