"""Observe set-valued THOL eligibility and explicitly dispatch one causal C8.

The marked IL/OZ preparation and default threshold/factors are unchanged.
Selecting every eligible parent is a declared policy, not a nodal derivation
of that policy. No coupling, pressure refresh or subsequent flow is performed.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

from benchmarks.thol_birth_transport import (  # noqa: E402
    prepare_birth_selection_source,
)
from benchmarks.thol_pressure_feedback import _payload, _state  # noqa: E402
from tnfr.operators.self_organization_selection import (  # noqa: E402
    execute_eligible_self_organization_stage,
    observe_self_organization_eligibility,
)
from tnfr.research.claims import ClaimStatus  # noqa: E402
from tnfr.research.core_manifests import (  # noqa: E402
    CoreExperimentManifest, current_git_source_provenance,
)
from tnfr.sdk._state import copy_graph_state  # noqa: E402
from tnfr.utils.io import safe_write  # noqa: E402


def _eligibility_record(observation):
    """Serialize stored evidence and its explicitly computed eligibility sets."""
    return {
        **asdict(observation),
        "eligible_nodes": observation.eligible_nodes,
        "candidates": tuple({**asdict(row), "eligible": row.eligible}
                            for row in observation.candidates),
    }


def _run_branch(graph, *, policy_change):
    """Observe first, then let the public dispatcher rederive current inputs."""
    before = deepcopy(_state(graph))
    node_data = deepcopy(dict(graph.nodes(data=True)))
    observation = observe_self_organization_eligibility(graph)
    projection_unchanged = before == _state(graph)
    node_attributes_unchanged = node_data == dict(graph.nodes(data=True))
    if not projection_unchanged or not node_attributes_unchanged:
        raise RuntimeError("eligibility observation changed its live input")
    dispatch = execute_eligible_self_organization_stage(graph)
    after = deepcopy(_state(graph))
    births = tuple(dispatch.parent_children)
    return {
        "policy_change": policy_change,
        "optional_precondition_gate_enabled": bool(
            graph.graph.get("VALIDATE_OPERATOR_PRECONDITIONS", False)
        ),
        "before": before,
        "eligibility": _eligibility_record(observation),
        "observer_state_projection_unchanged": projection_unchanged,
        "observer_node_attributes_unchanged": node_attributes_unchanged,
        "dispatch_eligibility": _eligibility_record(dispatch.eligibility),
        "stage_result": (
            None if dispatch.stage_result is None else asdict(dispatch.stage_result)
        ),
        "parent_children": births,
        "children": tuple({
            "parent": parent, "child": child,
            "degree": graph.degree(child),
            "node_data": deepcopy(dict(graph.nodes[child])),
        } for parent, child in births),
        "after": after,
        "scope": (
            "Complete original eight-node candidate inventory, followed by an "
            "explicit dispatch-all-eligible policy. Readouts are projections, "
            "not full runtime proof seals; no future or physical claim"
        ),
    }


def run_study():
    """Run one marked source and its explicitly changed optional-gate control."""
    graph, preparation = prepare_birth_selection_source()
    counterfactual = copy_graph_state(graph)
    counterfactual.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True
    return {
        "preparation": preparation,
        "configured_policy": _run_branch(graph, policy_change=None),
        "enabled_gate_counterfactual": _run_branch(
            counterfactual,
            policy_change="Enable existing VALIDATE_OPERATOR_PRECONDITIONS only",
        ),
        "limitations": (
            "Parent zero already carries the causal IL/OZ preparation mark",
            "The dispatch-all-eligible choice is an explicit execution policy",
            "The optional-gate branch changes policy, not the birth threshold",
            "This singleton eligible set does not test multi-parent conflicts",
            "The child has no transport edge; no UM, refresh or flow is run",
            "No autonomous persistence, physical particle or empirical validation",
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=(
        ROOT / "artifacts/research/thol_eligibility_dispatch.json"
    ))
    args = parser.parse_args()
    scope = (
        "src/tnfr", "benchmarks/thol_eligibility_dispatch.py",
        "benchmarks/thol_birth_transport.py", "benchmarks/thol_pressure_feedback.py",
        "benchmarks/capacity_localization.py",
    )
    provenance = current_git_source_provenance(ROOT, scope)
    sha, dirty, digest = provenance
    manifest = CoreExperimentManifest(
        claim_id="O1.b-THOL-eligibility-explicit-dispatch", git_sha=sha,
        source_dirty=dirty, dirty_source_hash=digest,
        versions={"python": platform.python_version(), "networkx": nx.__version__,
                  "numpy": np.__version__},
        graph_construction="Existing causally prepared, IL/OZ-marked C8",
        capacity_specification="Unit preparation; unchanged default THOL factors",
        solver="Shared refreshed Euler preparation only; no post-dispatch flow",
        timestep=0.25, seed=17, result_status=ClaimStatus.MEASURED,
        operator_sequence=("IL", "OZ", "two physical preparation steps",
                           "explicit eligible-set THOL stage or empty no-op"),
        telemetry=("all candidate gates", "joint viability", "actual parent/child map"),
        controls=("pure observation", "existing optional gate enabled", "empty dispatch"),
        artifacts=(str(args.output),),
    )
    manifest.validate_for_admission()
    report = {"manifest": manifest.to_dict(), "source_scope": scope, **run_study()}
    if current_git_source_provenance(ROOT, scope) != provenance:
        raise RuntimeError("source changed while executing the study")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(_payload(report), indent=2, allow_nan=False) + "\n"
    safe_write(args.output, lambda stream: stream.write(encoded))
    print(f"Wrote bounded THOL eligibility/dispatch study to {args.output}")


if __name__ == "__main__":
    main()
