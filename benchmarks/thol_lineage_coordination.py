"""Compare public UM dispatch on actual THOL parents and actual born children.

Only two fresh finite continuations execute. Historical no-event/all-node
controls are admitted by pinned bytes and complete common-source/observer
comparison, preserving their producer metadata rather than rebranding them.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

from benchmarks.thol_distributed_target import (  # noqa: E402
    LINEAGE_BRANCHES,
    run_distributed_target_branch,
)
from benchmarks.thol_pressure_feedback import _payload  # noqa: E402
from tnfr.research.claims import ClaimStatus  # noqa: E402
from tnfr.research.core_manifests import (  # noqa: E402
    CoreExperimentManifest,
    current_git_source_provenance,
)
from tnfr.utils import angle_diff  # noqa: E402
from tnfr.utils.io import safe_write  # noqa: E402

CONTROL_PATH = ROOT / "artifacts/research/thol_distributed_target_2026_09_18.json"
CONTROL_SHA256 = "09757606385536245ae35b2224d0d1932e8f33b465f37ab22209ef7dd6689574"
COMMON_FIELDS = (
    "prefix",
    "original_reference",
    "original_reference_sha256",
    "original_reference_frozen_time",
    "original_reference_frozen_before_baseline_flow",
    "initial_target",
    "baseline_flow",
    "baseline_steps",
    "baseline_capture_checks",
    "baseline_target",
    "common_source",
    "before_optional_event",
)


def _json_payload(value):
    return json.loads(json.dumps(_payload(value), allow_nan=False))


def load_control_evidence(path=CONTROL_PATH, *, expected_sha256=CONTROL_SHA256):
    """Admit declared immutable bytes, keeping their original producer manifest.

    The explicit expected digest also permits portable complete fixture files;
    the CLI defaults to the pinned historical digest. A regenerated control
    report needs an explicitly supplied digest, never automatic hash admission.
    Hash identity alone does not replace full baseline/observer comparison.
    """
    path = Path(path)
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise ValueError("retained control bytes differ from their expected digest")
    report = json.loads(raw)
    manifest = CoreExperimentManifest(**report["manifest"])
    manifest.validate_for_admission()
    if manifest.claim_id != "O1.b-distributed-fixed-relative-target-response":
        raise ValueError(
            "controls must retain the original distributed-target producer"
        )
    branches = tuple(report["branches"])
    if tuple(row["branch"] for row in branches) != ("no_event", "all_node_um"):
        raise ValueError(
            "controls must preserve the complete ordered no-event/all-node branches"
        )
    for field in COMMON_FIELDS:
        if branches[0][field] != branches[1][field]:
            raise ValueError(f"retained controls disagree on common field {field}")
    if not report.get("common_causal_baseline_reproduced"):
        raise ValueError(
            "retained controls do not establish their common causal baseline"
        )
    return branches, {
        "path": str(path),
        "sha256": actual,
        "byte_count": len(raw),
        "historical_manifest": report["manifest"],
        "historical_source_scope": report["source_scope"],
        "historical_producer_preserved": True,
    }


def compare_common_source(branch, controls):
    """Require complete JSON payload identity, not merely matching metrics."""
    compared = _json_payload({field: branch[field] for field in COMMON_FIELDS})
    for control in controls:
        for field in COMMON_FIELDS:
            if compared[field] != control[field]:
                raise ValueError(
                    f"fresh branch differs from retained control field {field}"
                )
    return {
        "all_fields_equal": True,
        "fields": COMMON_FIELDS,
        "control_branches": tuple(control["branch"] for control in controls),
    }


def _proposal_and_effects(branch, all_node_control):
    if branch["status"] != "executed":
        return None
    event = branch["event"]["coupling"]
    before, after = event["before"], event["after_raw"]
    parents, children = map(
        set, (branch["lineage"]["parents"], branch["lineage"]["children"])
    )
    proposal = event["kernel_proposal"]
    old_rows = {
        row["node"]: row
        for row in all_node_control["event"]["coupling"]["kernel_proposal"][
            "target_proposals"
        ]
    }
    rows = proposal["target_proposals"]
    if any(_json_payload(row) != old_rows[row["node"]] for row in rows):
        raise ValueError(
            "selected target-local proposals differ from the all-node control"
        )
    recipients = {node: [] for node in before["nodes"]}
    for row in rows:
        for phase in row["phase_proposals"]:
            recipients[phase["node"]].append(phase["source"])
    phase_changes = tuple(
        {
            "node": node,
            "before": left,
            "after": right,
            "exact_represented_difference": Fraction.from_float(right)
            - Fraction.from_float(left),
            "circular_difference": angle_diff(right, left),
        }
        for node, left, right in zip(
            before["nodes"], before["phase"], after["phase"], strict=True
        )
    )
    capacity_changes = tuple(
        {
            "node": node,
            "before": left,
            "after": right,
            "exact_represented_difference": Fraction.from_float(right)
            - Fraction.from_float(left),
        }
        for node, left, right in zip(
            before["nodes"], before["capacity"], after["capacity"], strict=True
        )
    )

    def edge_kind(left, right):
        if left in parents and right in parents:
            return "parent_parent"
        if left in children and right in children:
            return "child_child"
        return "parent_child"

    return {
        "selected_target_local_proposals_match_all_node_control": True,
        "phase_proposal_sources_by_node": tuple(
            {"node": node, "sources": tuple(sources)}
            for node, sources in recipients.items()
        ),
        "multiply_proposed_phase_nodes": tuple(
            node for node, sources in recipients.items() if len(sources) > 1
        ),
        "untargeted_phase_write_nodes": tuple(
            update["node"]
            for update in proposal["node_updates"]
            if update["theta_after"] is not None
            and update["node"] not in proposal["targets"]
        ),
        "phase_changes": phase_changes,
        "capacity_changes": capacity_changes,
        "new_edge_classes": tuple(
            {
                "left": left,
                "right": right,
                "weight": data["weight"],
                "kind": edge_kind(left, right),
            }
            for left, right, data in event["new_edges"]
        ),
        "epi_unchanged": before["epi"] == after["epi"],
        "scope": (
            "Target-local proposals share the authenticated source. Bidirectional phase writes "
            "can cross cohorts and overlap before merging. Actual accepted edges and weights "
            "are observed separately; this is not an additive decomposition of all-node UM."
        ),
    }


def _outcome(branch):
    if branch.get("status") == "refused":
        return {"branch": branch["branch"], "status": "refused", "endpoint_time": 1.0}
    endpoint = branch["endpoint_target"]
    return {
        "branch": branch["branch"],
        "status": "executed",
        "endpoint_time": branch["endpoint"]["time"],
        "original_reference_sha256": branch["original_reference_sha256"],
        "fixed_target_variance": Fraction(endpoint["pattern"]["error_variance"]),
        "target_compatible": endpoint["target_compatible"],
        "compatibility_energy": Fraction(endpoint["compatibility_energy"]),
        "current_profile_mismatch_variance": Fraction(
            endpoint["limiting_pattern"]["error_variance"]
        ),
        "current_model_mean_drift": Fraction(endpoint["reference"]["mean_drift"]),
        "original_metric_mean_change": Fraction(
            branch["comparison"]["original_metric_mean_change"]
        ),
    }


def run_study(control_path=CONTROL_PATH, *, expected_control_sha256=CONTROL_SHA256):
    controls, binding = load_control_evidence(
        control_path, expected_sha256=expected_control_sha256
    )
    branches = []
    for name in LINEAGE_BRANCHES:
        branch = run_distributed_target_branch(name)
        comparison = compare_common_source(branch, controls)
        branches.append(
            {
                **branch,
                "retained_common_source_comparison": comparison,
                "proposal_and_effects": _proposal_and_effects(branch, controls[1]),
            }
        )
    if hashlib.sha256(Path(control_path).read_bytes()).hexdigest() != binding["sha256"]:
        raise RuntimeError("retained control bytes changed during the study")
    return {
        "branches": tuple(branches),
        "retained_controls": binding,
        "control_outcomes": tuple(_outcome(control) for control in controls),
        "lineage_outcomes": tuple(_outcome(branch) for branch in branches),
        "fresh_control_trajectories_executed": False,
        "refusal_scope": (
            "Incremental grammar, hard U3 and configured optional per-node refusals are "
            "retained before committing UM. A joint merged-stage refusal or an unexpected "
            "execution/accounting error aborts without retry; neither occurred in admitted branches."
        ),
        "scope": (
            "Two predeclared ancestry-based dispatch interventions, each from an independent "
            "causal replay. Complete common source, frozen observer and baseline agree with "
            "the retained controls. No response-based cohort selection, parameter tuning, "
            "single-channel isolation or autonomous recovery claim."
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "If historical controls are unavailable, run thol_distributed_target.py with a new "
            "--output, retain its SHA256, and supply both --controls and "
            "--expected-control-sha256. Full common-source comparison remains mandatory."
        ),
    )
    parser.add_argument("--controls", type=Path, default=CONTROL_PATH)
    parser.add_argument("--expected-control-sha256", default=CONTROL_SHA256)
    parser.add_argument(
        "--output",
        type=Path,
        default=(ROOT / "artifacts/research/thol_lineage_coordination.json"),
    )
    args = parser.parse_args()
    if args.controls.resolve() == args.output.resolve():
        raise ValueError("new evidence must not overwrite its historical controls")
    scope = (
        "src/tnfr",
        "benchmarks/thol_lineage_coordination.py",
        "benchmarks/thol_distributed_target.py",
        "benchmarks/thol_distributed_transport.py",
        "benchmarks/thol_birth_transport.py",
        "benchmarks/thol_pressure_feedback.py",
        "benchmarks/thol_eligibility_dispatch.py",
        "benchmarks/capacity_localization.py",
        "benchmarks/structural_target_compatibility.py",
        "benchmarks/thol_preparation_policy.py",
        "benchmarks/selection_birth_closure.py",
    )
    provenance = current_git_source_provenance(ROOT, scope)
    sha, dirty, digest = provenance
    manifest = CoreExperimentManifest(
        claim_id="O1.b-lineage-scoped-UM-fixed-target-response",
        git_sha=sha,
        source_dirty=dirty,
        dirty_source_hash=digest,
        versions={
            "python": platform.python_version(),
            "networkx": nx.__version__,
            "numpy": np.__version__,
        },
        graph_construction="Unchanged distributed THOL ancestry with verified parent/child partition",
        capacity_specification="Default factors; original z0/H0 frozen before baseline continuation",
        solver="Shared refreshed Euler partitions; one .25,.25 continuation per admitted cohort",
        timestep=0.25,
        seed=17,
        result_status=ClaimStatus.MEASURED,
        operator_sequence=(
            "coherence",
            "dissonance",
            "self_organization",
            "coupling",
            "coupling",
        ),
        telemetry=(
            "complete actual lineage",
            "live admission/refusal",
            "overlapping UM proposals",
            "fixed-target compatibility and exact event/flow budgets",
        ),
        controls=("authenticated retained no-event and all-node controls",),
        artifacts=(str(args.output),),
    )
    manifest.validate_for_admission()
    report = {
        "manifest": manifest.to_dict(),
        "source_scope": scope,
        **run_study(
            args.controls, expected_control_sha256=args.expected_control_sha256
        ),
        "experimental_status": "No empirical correspondence tested",
    }
    if current_git_source_provenance(ROOT, scope) != provenance:
        raise RuntimeError("source changed during the finite lineage study")
    encoded = json.dumps(_payload(report), indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    safe_write(args.output, lambda stream: stream.write(encoded))
    print(f"Wrote lineage-scoped UM observations to {args.output}")


if __name__ == "__main__":
    main()
