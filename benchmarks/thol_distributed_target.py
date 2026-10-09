"""Compare a fixed derived relative profile with a later public UM response.

Each branch independently replays the same causal distributed preparation.
The original target is derived at t=.5, before any measured continuation. It
is a held-model relative equilibrium, not the actual shape created by THOL.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
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

from benchmarks.structural_target_compatibility import (
    compare_target_channels,
)  # noqa: E402
from benchmarks.thol_distributed_transport import (  # noqa: E402
    STEPS,
    _couple_parents,
    _physical_flow,
    prepare_distributed_transport_support,
)
from benchmarks.thol_pressure_feedback import _payload, _state  # noqa: E402
from benchmarks.thol_preparation_policy import _literal  # noqa: E402
from tnfr.dynamics.sampling import update_node_sample  # noqa: E402
from tnfr.operators.grammar_dynamics import validate_candidate  # noqa: E402
from tnfr.operators.preconditions import (  # noqa: E402
    OperatorPreconditionError,
    validate_coupling,
    validate_phase_gate_u3,
)
from tnfr.physics.forced_support import (  # noqa: E402
    derive_forced_support_balance,
    observe_forced_support_event,
    observe_forced_support_step,
    observe_forced_support_target,
)
from tnfr.physics.support_transport import SupportTransportSnapshot  # noqa: E402
from tnfr.research.claims import ClaimStatus  # noqa: E402
from tnfr.research.core_manifests import (  # noqa: E402
    CoreExperimentManifest,
    current_git_source_provenance,
)
from tnfr.utils.io import safe_write  # noqa: E402

BRANCHES = ("no_event", "all_node_um")
LINEAGE_BRANCHES = ("original_parents", "born_children")


def _snapshot(readout):
    return SupportTransportSnapshot(**readout["observation"]["snapshot"])


def _reference(readout, previous=None):
    observation = readout["observation"]
    snapshot = _snapshot(readout)
    if (
        previous is not None
        and previous.epi_weight == observation["epi_weight"]
        and previous.forcing == observation["forcing"]
        and all(
            getattr(previous.source, field) == getattr(snapshot, field)
            for field in ("nodes", "conductance", "support_neighbors", "capacity")
        )
    ):
        return previous
    return derive_forced_support_balance(
        snapshot,
        epi_weight=observation["epi_weight"],
        forcing=observation["forcing"],
    )


def _digest(reference):
    encoded = json.dumps(
        _payload(asdict(reference)),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _target(original, current, readout):
    return observe_forced_support_target(
        original,
        current,
        _snapshot(readout),
        forcing_components=readout["components"],
    )


def _flow_steps(reference, flow):
    """Use the existing exact held-model observer once per actual segment."""
    records = []
    for segment in flow["segments"]:
        before = SupportTransportSnapshot(**segment["before_support"])
        after = SupportTransportSnapshot(**segment["after_support"])
        observation = observe_forced_support_step(
            reference,
            before,
            after,
            segment["interval"]["duration"],
        )
        records.append(asdict(observation))
    return tuple(records)


def _held_capture_checks(before, after):
    first, last = before["observation"], after["observation"]
    checks = {
        "phase_preserved": first["phase"] == last["phase"],
        "normalized_weights_preserved": first["normalized_weights"]
        == last["normalized_weights"],
        "forcing_preserved": first["forcing"] == last["forcing"],
        "epi_weight_preserved": first["epi_weight"] == last["epi_weight"],
        "support_and_capacity_preserved": all(
            first["snapshot"][key] == last["snapshot"][key]
            for key in ("nodes", "conductance", "support_neighbors", "capacity")
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(
            "event-free flow changed captured phase, coefficients or support"
        )
    return {
        **checks,
        "scope": (
            "Exact endpoint capture agreement plus the retained executor's protected-state "
            "checks inside this finite invocation; no future constancy inferred"
        ),
    }


def _common_source(graph):
    """Retain ordered inputs relevant to the subsequent public UM decision."""
    return {
        "state": deepcopy(_state(graph)),
        "node_attributes": tuple(
            (node, _literal(dict(data))) for node, data in graph.nodes(data=True)
        ),
        "ordered_neighbors": tuple(
            (node, tuple(graph.neighbors(node))) for node in graph
        ),
        "node_sample": tuple(graph.graph["_node_sample"]),
        "configuration": deepcopy(
            {
                key: value
                for key, value in graph.graph.items()
                if key.startswith("UM_")
                or key
                in (
                    "GLYPH_FACTORS",
                    "DNFR_WEIGHTS",
                    "_dnfr_weights",
                    "VALIDATE_OPERATOR_PRECONDITIONS",
                    "DELTA_PHI_MAX",
                    "RANDOM_SEED",
                    "SORT_NODES",
                )
            }
        ),
    }


def _same_state_rate_change(before, after):
    """Compare signed old-target rates at one unchanged EPI state."""
    if (
        before.target_reference != after.target_reference
        or before.pattern != after.pattern
    ):
        raise ValueError(
            "rate comparison requires one unchanged EPI pattern and original target"
        )
    homogeneous = after.homogeneous_energy_rate - before.homogeneous_energy_rate
    target_source = after.target_source_energy_rate - before.target_source_energy_rate
    realization = (
        after.stored_pressure_energy_rate_defect
        - before.stored_pressure_energy_rate_defect
    )
    total = after.stored_nodal_energy_rate - before.stored_nodal_energy_rate
    residual = total - homogeneous - target_source - realization
    if residual:
        raise RuntimeError("same-state signed nodal-rate decomposition failed")
    return {
        "before_stored_nodal_energy_rate": before.stored_nodal_energy_rate,
        "after_stored_nodal_energy_rate": after.stored_nodal_energy_rate,
        "homogeneous_energy_rate_change": homogeneous,
        "target_source_energy_rate_change": target_source,
        "stored_pressure_energy_rate_defect_change": realization,
        "stored_nodal_energy_rate_change": total,
        "identity_residual": residual,
        "scope": (
            "Exact signed derivative comparison in the original H0 metric at unchanged EPI. "
            "It is not a finite trajectory improvement or separately executed channel ablation."
        ),
    }


def _lineage_targets(graph, prefix, branch):
    """Derive disjoint cohorts from actual birth receipts and live parentage."""
    if branch not in LINEAGE_BRANCHES:
        raise ValueError(
            "a lineage branch must select original parents or actual children"
        )
    pairs = tuple(prefix["birth"]["parent_children"])
    parents = tuple(parent for parent, _ in pairs)
    children = tuple(child for _, child in pairs)
    if (
        not parents
        or parents != tuple(prefix["birth"]["before"]["nodes"])
        or len(set(parents)) != len(parents)
        or len(set(children)) != len(children)
        or set(parents) & set(children)
        or tuple(graph) != parents + children
    ):
        raise ValueError(
            "birth cohorts must be unique, disjoint and exhaust actual ordered support"
        )
    for parent, child in pairs:
        if (
            graph.nodes[child].get("parent_node") != parent
            or tuple(graph.nodes[parent].get("sub_nodes", ())) != (child,)
            or tuple(graph.graph.get("hierarchy", {}).get(parent, ())) != (child,)
        ):
            raise ValueError(
                "retained birth receipt disagrees with actual graph lineage"
            )
    selected = parents if branch == "original_parents" else children
    return selected, {
        "parent_children": pairs,
        "parents": parents,
        "children": children,
        "current_nodes": tuple(graph),
        "selected_targets": selected,
        "disjoint_and_exhaustive": True,
        "live_parentage_verified": True,
        "selection_rule": branch,
        "scope": "Selection uses actual ancestry, never a measured response or label convention",
    }


def _cohort_admission(graph, targets):
    """Read every selected live gate; retain only deliberate admission refusals."""
    before = _common_source(graph)
    optional = bool(graph.graph.get("VALIDATE_OPERATOR_PRECONDITIONS", False))
    rows = []
    for node in targets:
        grammar = validate_candidate(graph, node, "UM")
        hard_reason = None
        optional_reason = None
        try:
            validate_phase_gate_u3(graph, node, "Coupling")
        except OperatorPreconditionError as exc:
            hard_reason = str(exc)
        if optional:
            try:
                validate_coupling(graph, node)
            except OperatorPreconditionError as exc:
                optional_reason = str(exc)
        rows.append(
            {
                "node": node,
                "grammar": asdict(grammar),
                "hard_u3_allowed": hard_reason is None,
                "hard_u3_refusal": hard_reason,
                "optional_gate_enabled": optional,
                "optional_gate_allowed": (
                    None if not optional else optional_reason is None
                ),
                "optional_gate_refusal": optional_reason,
                "allowed": grammar.allowed
                and hard_reason is None
                and optional_reason is None,
            }
        )
    if before != _common_source(graph):
        raise RuntimeError("read-only cohort admission changed its projected source")
    return {
        "targets": targets,
        "candidates": tuple(rows),
        "allowed": all(row["allowed"] for row in rows),
        "source_projection_unchanged": True,
        "scope": (
            "Per-node grammar, hard U3 and configured optional admission only. "
            "The subsequent public stage separately checks merged proposals; a joint "
            "stage failure propagates without retry or an executed-branch claim."
        ),
    }


def run_distributed_target_branch(branch="no_event"):
    """Execute one independently prepared, finite fixed-target comparison."""
    if branch not in BRANCHES + LINEAGE_BRANCHES:
        raise ValueError(f"branch must be one of {BRANCHES + LINEAGE_BRANCHES}")
    graph, prefix = prepare_distributed_transport_support("attached")
    original_readout = prefix["coupling"]["refreshed_forcing"]
    original = _reference(original_readout)
    original_digest = _digest(original)
    original_frozen_time = graph.graph["_t"]
    if original_frozen_time != 0.5:
        raise RuntimeError("the target must be frozen at the first post-UM boundary")
    initial_target = _target(original, original, original_readout)
    baseline_flow = _physical_flow(graph, original.source)
    baseline_readout = baseline_flow["after_forcing"]
    baseline_capture_checks = _held_capture_checks(original_readout, baseline_readout)
    baseline_reference = _reference(baseline_readout, original)
    if baseline_reference is not original:
        raise RuntimeError("baseline event-free flow changed its captured held model")
    baseline_steps = _flow_steps(original, baseline_flow)
    baseline_target = _target(original, original, baseline_readout)
    preevent = deepcopy(_state(graph))
    common_source = _common_source(graph)
    event = None
    current = original
    post_event_target = baseline_target
    lineage = admission = None
    if branch in LINEAGE_BRANCHES:
        targets, lineage = _lineage_targets(graph, prefix, branch)
        update_node_sample(graph, step=1)
        admission = _cohort_admission(graph, targets)
        if not admission["allowed"]:
            return {
                "branch": branch,
                "status": "refused",
                "lineage": lineage,
                "admission": admission,
                "prefix": prefix,
                "original_reference": asdict(original),
                "original_reference_sha256": original_digest,
                "original_reference_frozen_time": original_frozen_time,
                "original_reference_frozen_before_baseline_flow": True,
                "initial_target": asdict(initial_target),
                "baseline_flow": baseline_flow,
                "baseline_steps": baseline_steps,
                "baseline_capture_checks": baseline_capture_checks,
                "baseline_target": asdict(baseline_target),
                "common_source": common_source,
                "before_optional_event": preevent,
                "event": None,
                "after_optional_event": deepcopy(_state(graph)),
                "post_event_target": asdict(baseline_target),
                "continuation_flow": None,
                "continuation_steps": (),
                "continuation_capture_checks": None,
                "endpoint_target": None,
                "endpoint": deepcopy(_state(graph)),
                "scope": (
                    "Explicit live refusal before UM; no retry, tuning, fabricated history "
                    "or continuation. Accounting and unexpected implementation errors propagate."
                ),
            }
    if branch != "no_event":
        if branch == "all_node_um":
            targets = tuple(graph)
        _, coupling = _couple_parents(
            graph,
            targets,
            case="attached",
            target_policy=(
                "Every current node once, after birth and baseline transport"
                if branch == "all_node_um"
                else f"Actual {branch} cohort once, after the common baseline"
            ),
        )
        current_readout = coupling["refreshed_forcing"]
        current = _reference(current_readout, original)
        event_budget = observe_forced_support_event(
            baseline_reference,
            current,
            _snapshot(baseline_readout),
            _snapshot(current_readout),
        )
        post_event_target = _target(original, current, current_readout)
        if post_event_target.pattern != baseline_target.pattern:
            raise RuntimeError(
                "UM changed the fixed-target EPI pattern without an EPI jump"
            )
        event = {
            "coupling": coupling,
            "exact_forced_event": asdict(event_budget),
            "signed_target_channel_change": compare_target_channels(
                baseline_target, post_event_target
            ),
            "same_state_rate_change": _same_state_rate_change(
                baseline_target, post_event_target
            ),
            "fixed_target_pattern_unchanged": True,
            "scope": (
                "One actual simultaneous public UM, with live admission and a fresh sample. "
                "Its metric/profile reset is accounting, not measured shape recovery."
            ),
        }
    postevent = deepcopy(_state(graph))
    continuation_flow = _physical_flow(
        graph,
        _snapshot(
            baseline_readout
            if event is None
            else event["coupling"]["refreshed_forcing"]
        ),
    )
    endpoint_readout = continuation_flow["after_forcing"]
    continuation_capture_checks = _held_capture_checks(
        baseline_readout if event is None else event["coupling"]["refreshed_forcing"],
        endpoint_readout,
    )
    if _reference(endpoint_readout, current) is not current:
        raise RuntimeError(
            "continuation event-free flow changed its captured held model"
        )
    continuation_steps = _flow_steps(current, continuation_flow)
    endpoint_target = _target(original, current, endpoint_readout)
    if _digest(original) != original_digest:
        raise RuntimeError("the original target changed after its pre-flow freeze")
    return {
        "branch": branch,
        "prefix": prefix,
        **(
            {"status": "executed", "lineage": lineage, "admission": admission}
            if lineage is not None
            else {}
        ),
        "original_reference": asdict(original),
        "original_reference_sha256": original_digest,
        "original_reference_frozen_time": original_frozen_time,
        "original_reference_frozen_before_baseline_flow": True,
        "initial_target": asdict(initial_target),
        "baseline_flow": baseline_flow,
        "baseline_steps": baseline_steps,
        "baseline_capture_checks": baseline_capture_checks,
        "baseline_target": asdict(baseline_target),
        "common_source": common_source,
        "before_optional_event": preevent,
        "event": event,
        "after_optional_event": postevent,
        "post_event_target": asdict(post_event_target),
        "continuation_flow": continuation_flow,
        "continuation_steps": continuation_steps,
        "continuation_capture_checks": continuation_capture_checks,
        "endpoint_target": asdict(endpoint_target),
        "endpoint": deepcopy(_state(graph)),
        "comparison": {
            "baseline_fixed_target_variance_change": (
                baseline_target.pattern.error_variance
                - initial_target.pattern.error_variance
            ),
            "continuation_fixed_target_variance_change": (
                endpoint_target.pattern.error_variance
                - post_event_target.pattern.error_variance
            ),
            "original_metric_mean_change": endpoint_target.pattern.mean
            - initial_target.pattern.mean,
            "mean_scope": (
                "Total mean change in the original H0 observer, not a current-model uniform "
                "drift coefficient. Current-reference step budgets separate modeled drift, "
                "pressure and Euler defects; the event budget separates metric reweighting."
            ),
            "initial_target_error_is_zero": not any(
                initial_target.pattern.relative_error
            ),
            "current_model_target_compatible": endpoint_target.target_compatible,
        },
        "scope": (
            "Fixed z0, H0 and B0 are derived from the first post-UM held model; z0 is not the "
            "actual born EPI pattern. Current references diagnose model compatibility and event "
            "accounting without replacing that target. Finite Euler readouts and signed budgets "
            "do not prove asymptotic recovery, autonomous persistence or empirical particles. "
            "A live admission or read-out refusal aborts this bounded study without retry or tuning."
        ),
    }


def run_study():
    branches = tuple(run_distributed_target_branch(branch) for branch in BRANCHES)
    left, right = branches
    if (
        left["original_reference"] != right["original_reference"]
        or left["before_optional_event"] != right["before_optional_event"]
        or left["prefix"] != right["prefix"]
        or left["common_source"] != right["common_source"]
        or left["baseline_flow"] != right["baseline_flow"]
    ):
        raise RuntimeError(
            "independent causal branches did not reproduce the common baseline"
        )
    return {
        "branches": branches,
        "common_causal_baseline_reproduced": True,
        "scope": "Two independent causal replays; no cloned or assigned execution histories",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=(ROOT / "artifacts/research/thol_distributed_target.json"),
    )
    args = parser.parse_args()
    scope = (
        "src/tnfr",
        "benchmarks/thol_distributed_target.py",
        "benchmarks/thol_distributed_transport.py",
        "benchmarks/thol_birth_transport.py",
        "benchmarks/thol_pressure_feedback.py",
        "benchmarks/thol_eligibility_dispatch.py",
        "benchmarks/capacity_localization.py",
        "benchmarks/structural_target_compatibility.py",
        "benchmarks/thol_preparation_policy.py",
    )
    provenance = current_git_source_provenance(ROOT, scope)
    sha, dirty, digest = provenance
    manifest = CoreExperimentManifest(
        claim_id="O1.b-distributed-fixed-relative-target-response",
        git_sha=sha,
        source_dirty=dirty,
        dirty_source_hash=digest,
        versions={
            "python": platform.python_version(),
            "networkx": nx.__version__,
            "numpy": np.__version__,
        },
        graph_construction="Unchanged causal all-node C8 preparation and actual eight-node birth",
        capacity_specification="Default THOL/UM; original target metric retained through later UM",
        solver="Existing refreshed Euler partitions, .5 to 1 and 1 to 1.5",
        timestep=STEPS[0],
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
            "original relative profile and initial discrepancy",
            "current-model compatibility",
            "event reset and old-target error",
            "exact mean drift and finite step budgets",
        ),
        controls=("independent continuation without the second UM",),
        artifacts=(str(args.output),),
    )
    manifest.validate_for_admission()
    report = {
        "manifest": manifest.to_dict(),
        "source_scope": scope,
        **run_study(),
        "experimental_status": "No empirical correspondence tested",
    }
    if current_git_source_provenance(ROOT, scope) != provenance:
        raise RuntimeError("source changed during the finite target study")
    encoded = json.dumps(_payload(report), indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    safe_write(args.output, lambda stream: stream.write(encoded))
    print(f"Wrote distributed fixed-target observations to {args.output}")


if __name__ == "__main__":
    main()
