"""Nine regional balances on one authenticated post-attachment THOL snapshot.

The regions are the eight retained ancestry pairs and the complete child
cohort. All degrees, boundary edges and forcing remain from the full graph.
No graph, native execution, trajectory, fitted coefficient or new partition
is constructed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from benchmarks.thol_retained_reset_audit import _load  # noqa: E402
from benchmarks.thol_family_closure import (
    _capture,
    _equal,
    _match_live_state,
    _reference,
)  # noqa: E402
from benchmarks.thol_distributed_target import _digest  # noqa: E402
from benchmarks.thol_pressure_feedback import _payload  # noqa: E402
from tnfr._exact_time import finite_represented_real  # noqa: E402
from tnfr.physics.support_transport import (
    observe_regional_support_balance,
)  # noqa: E402
from tnfr.research.claims import ClaimStatus  # noqa: E402
from tnfr.research.core_manifests import (
    CoreExperimentManifest,
    current_git_source_provenance,
)  # noqa: E402
from tnfr.utils.io import safe_write  # noqa: E402

INPUT_PATH = ROOT / "artifacts/research/thol_native_runtime_response_2026_09_18.json"
INPUT_SHA256 = "71252d116d8933d15a797707ed9f44ed865406422da6db492b48f6989d739c95"
INPUT_CLAIM = "O1.b-generated-native-runtime-response"


def _bind_source_edges(state, snapshot):
    """Bind this retained simple undirected source, without building a graph."""
    positions = {node: i for i, node in enumerate(snapshot.nodes)}
    edges, support = {}, [set() for _ in snapshot.nodes]
    for left, right, attributes in state["edges"]:
        if left not in positions or right not in positions or left == right:
            raise ValueError(
                "retained source must have simple nonloop edges on its nodes"
            )
        i, j = positions[left], positions[right]
        _, weight = finite_represented_real(
            attributes.get("weight", 1.0), "retained source conductance"
        )
        if weight <= 0 or (i, j) in edges or (j, i) in edges:
            raise ValueError("retained source edges must be distinct and positive")
        edges[i, j] = edges[j, i] = weight
        support[i].add(j)
        support[j].add(i)
    _equal(
        tuple((i, j, w) for (i, j), w in sorted(edges.items())),
        snapshot.conductance,
        "live full conductance",
    )
    _equal(
        tuple(tuple(sorted(row)) for row in support),
        snapshot.support_neighbors,
        "live full support",
    )


def _ancestry(prior, state, nodes):
    birth = prior["prefix"]["birth"]
    pairs = tuple(tuple(row) for row in birth["parent_children"])
    if len(pairs) != 8 or any(len(row) != 2 for row in pairs):
        raise ValueError("the retained source requires eight actual ancestry pairs")
    parents, children = tuple(p for p, _ in pairs), tuple(c for _, c in pairs)
    if (
        len(set(parents + children)) != 16
        or parents + children != nodes
        or len(set(map(str, nodes))) != 16
        or tuple(birth["before"]["nodes"]) != parents
        or tuple(birth["after"]["nodes"]) != nodes
    ):
        raise ValueError("birth ancestry must be disjoint, exhaustive and ordered")
    for key, expected in (
        ("parent_children", pairs),
        ("parents", parents),
        ("children", children),
        ("nodes", nodes),
    ):
        _equal(prior["lineage"][key], expected, f"actual lineage {key}")
    hierarchy = {str(parent): [child] for parent, child in pairs}
    child_map = {**hierarchy, **{str(child): [] for child in children}}
    for row in (birth["after"], state, prior["common_source"]["state"]):
        if (
            tuple(row["nodes"]) != nodes
            or row["hierarchy"] != hierarchy
            or row["children"] != child_map
        ):
            raise ValueError(
                "retained state hierarchy differs from the actual birth pairs"
            )
    attrs = prior["common_source"]["node_attributes"]
    if tuple(node for node, _ in attrs) != nodes:
        raise ValueError("retained parentage must cover the complete ordered support")
    attributes = dict(attrs)
    receipts = birth["children"]
    if tuple((row["parent"], row["child"]) for row in receipts) != pairs:
        raise ValueError("birth child receipts differ from the selected ancestry")
    for (parent, child), receipt in zip(pairs, receipts, strict=True):
        if (
            attributes[parent].get("sub_nodes") != [child]
            or attributes[child].get("parent_node") != parent
            or receipt["node_data"].get("parent_node") != parent
        ):
            raise ValueError("retained actual parent pointers differ from ancestry")
    return pairs, parents, children


def _component_budget(balance, values):
    values = tuple(values)
    indices = balance.region_indices
    if len(values) != len(balance.source.nodes):
        raise ValueError("component must preserve the full node space")
    total = sum((balance.strengths[i] * values[i] for i in indices), F(0))
    variance = sum(
        (
            balance.strengths[i] * z * values[i]
            for i, z in zip(indices, balance.centered_epi, strict=True)
        ),
        F(0),
    )
    return {"pressure": values, "weighted_total_rate": total, "variance_rate": variance}


def _region_report(snapshot, observation, components, region, label):
    balance = observe_regional_support_balance(
        snapshot, region, epi_weight=observation.epi_weight, forcing=observation.forcing
    )
    channels = {name: _component_budget(balance, values) for name, values in components}
    kernel = _component_budget(balance, observation.kernel_pressure_defect)
    stored = _component_budget(balance, observation.stored_pressure_residual)
    if (
        sum((row["weighted_total_rate"] for row in channels.values()), F(0))
        != balance.mass_forcing_rate
        or sum((row["variance_rate"] for row in channels.values()), F(0))
        != balance.variance_forcing_rate
        or kernel["weighted_total_rate"] + stored["weighted_total_rate"]
        != balance.mass_defect_rate
        or kernel["variance_rate"] + stored["variance_rate"]
        != balance.variance_defect_rate
    ):
        raise RuntimeError("regional channel or pressure-defect decomposition failed")
    return {
        "label": label,
        "balance": asdict(balance),
        "forcing_channels": channels,
        "fresh_kernel_defect": kernel,
        "stored_minus_fresh_residual": stored,
        "fresh_kernel_weighted_total_rate": balance.model_mass_rate
        + kernel["weighted_total_rate"],
        "fresh_kernel_variance_rate": balance.model_variance_rate
        + kernel["variance_rate"],
        "all_exact_decompositions_pass": True,
        "mass_field_meaning": "H-weighted EPI total; no physical mass correspondence",
    }


def _held_parent_response(snapshot, observation, parents, children):
    lookup = {node: i for i, node in enumerate(snapshot.nodes)}
    child_indices = {lookup[node] for node in children}
    parent_indices = {lookup[node] for node in parents}
    if any(
        j in child_indices for i in child_indices for j in snapshot.support_neighbors[i]
    ):
        raise ValueError(
            "conditional independent-child response requires no child-child support edges"
        )
    rows = []
    for child in children:
        i = lookup[child]
        edges = tuple(
            (j, weight) for source, j, weight in snapshot.conductance if source == i
        )
        degree = sum((weight for _, weight in edges), F(0))
        if degree <= 0 or any(j not in parent_indices for j, _ in edges):
            raise ValueError(
                "every child must receive positive conductance only from actual parents"
            )
        weights = tuple((snapshot.nodes[j], weight / degree) for j, weight in edges)
        neighbor_mean = (
            sum((weight * snapshot.epi[j] for j, weight in edges), F(0)) / degree
        )
        force_offset = observation.forcing[i] / observation.epi_weight
        equilibrium = neighbor_mean + force_offset
        rate = observation.epi_weight * snapshot.capacity[i]
        pressure_residual = (
            observation.epi_weight * (neighbor_mean - equilibrium)
            + observation.forcing[i]
        )
        represented_model_rate = snapshot.capacity[i] * (
            observation.epi_weight * snapshot.epi_gradient[i] + observation.forcing[i]
        )
        affine_identity = represented_model_rate + rate * (
            snapshot.epi[i] - equilibrium
        )
        if (
            pressure_residual
            or affine_identity
            or rate <= 0
            or sum((w for _, w in weights), F(0)) != 1
        ):
            raise RuntimeError("conditional held-parent child response identity failed")
        rows.append(
            {
                "child": child,
                "parent_weights": weights,
                "full_strength": degree,
                "parent_neighbor_mean": neighbor_mean,
                "forcing_offset": force_offset,
                "conditional_equilibrium_epi": equilibrium,
                "relaxation_rate": rate,
                "current_epi": snapshot.epi[i],
                "model_nodal_rate": represented_model_rate,
                "equilibrium_pressure_residual": pressure_residual,
                "affine_rate_identity_residual": affine_identity,
            }
        )
    return {
        "no_child_child_support_edges": True,
        "rows": rows,
        "formula": "x_child_star = weighted_parent_mean + F_child/e; xdot_child = -e*nu_child*(x_child-x_child_star)",
        "scope": "Conditional exact held-parent, held-capacity/phase/support/forcing model. Parents actually evolve; "
        "these are not observed equilibria, runtime forecasts, admissible clipped states or autonomous child regions.",
    }


def audit_prior(prior):
    """Compute the nine fixed observations from one supplied retained branch."""
    if prior["branch"] != "control" or prior["status"] != "executed":
        raise ValueError("audit source must be the retained completed control branch")
    coupling = prior["prefix"]["coupling"]
    state, raw_capture = coupling["after_refreshed"], coupling["refreshed_forcing"]
    if F(state["time"]) != F(1, 2):
        raise ValueError(
            "regional source must be the first post-attachment t=0.5 snapshot"
        )
    snapshot, observation, components = _capture(raw_capture)
    _match_live_state(state, raw_capture, snapshot)
    _bind_source_edges(state, snapshot)
    reference = _reference(prior["original_reference"])
    _equal(
        asdict(reference.source), asdict(snapshot), "original reference source snapshot"
    )
    if (
        reference.epi_weight != observation.epi_weight
        or reference.forcing != observation.forcing
        or prior["original_reference_sha256"] != _digest(reference)
    ):
        raise ValueError(
            "original reference digest or forcing differs from the actual source"
        )
    if tuple(name for name, _ in components) != ("phase", "vf", "topo"):
        raise ValueError("full canonical non-EPI channel order is required")
    pairs, parents, children = _ancestry(prior, state, snapshot.nodes)
    regions = [
        _region_report(snapshot, observation, components, pair, f"ancestry_pair_{i}")
        for i, pair in enumerate(pairs)
    ]
    regions.append(
        _region_report(snapshot, observation, components, children, "actual_children")
    )
    response = _held_parent_response(snapshot, observation, parents, children)
    return {
        "source_time": F(1, 2),
        "source_state": state,
        "source_capture": raw_capture,
        "original_reference": asdict(reference),
        "actual_lineage": {"pairs": pairs, "parents": parents, "children": children},
        "regions": regions,
        "child_cohort_response": response,
        "snapshot_count": 1,
        "regional_observation_count": 9,
        "native_calls": 0,
        "new_trajectories": 0,
        "partition_search": False,
        "regional_formation_or_persistence_certified": False,
        "scope": "Instantaneous full-graph regional budgets at one retained state; boundary dependence neither proves "
        "nor refutes relational NFR formation or autonomous maintenance.",
    }


def load_evidence(path=INPUT_PATH, *, expected_sha256=INPUT_SHA256):
    report, binding = _load(path, expected_sha256, INPUT_CLAIM)
    if tuple(row["branch"] for row in report["replayed_prior_reports"]) != (
        "control",
        "child_emission",
    ) or tuple(row["branch"] for row in report["branches"]) != (
        "control",
        "child_emission",
    ):
        raise ValueError("retained native artifact branch order differs")
    return report["replayed_prior_reports"][0], binding


def run_study(path=INPUT_PATH, *, expected_sha256=INPUT_SHA256):
    prior, binding = load_evidence(path, expected_sha256=expected_sha256)
    result = audit_prior(prior)
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != binding["sha256"]:
        raise RuntimeError("retained input changed during regional audit")
    return {"historical_input": binding, **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--expected-sha256", default=INPUT_SHA256)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/research/thol_regional_balance_audit_2026_09_18.json",
    )
    args = parser.parse_args()
    if args.output.resolve() == args.input.resolve():
        raise ValueError("output must not overwrite the retained input")
    scope = (
        "src/tnfr",
        "benchmarks/thol_regional_balance_audit.py",
        "benchmarks/thol_retained_reset_audit.py",
        "benchmarks/thol_family_closure.py",
        "benchmarks/thol_distributed_target.py",
        "benchmarks/thol_pressure_feedback.py",
    )
    provenance = current_git_source_provenance(ROOT, scope)
    result = run_study(args.input, expected_sha256=args.expected_sha256)
    if current_git_source_provenance(ROOT, scope) != provenance:
        raise RuntimeError("source changed during regional audit")
    sha, dirty, digest = provenance
    manifest = CoreExperimentManifest(
        claim_id="O1.b-retained-THOL-regional-balance",
        git_sha=sha,
        source_dirty=dirty,
        dirty_source_hash=digest,
        versions={"python": platform.python_version()},
        graph_construction="One retained full post-attachment graph snapshot",
        capacity_specification="Actual positive capacities; full graph degrees and boundary couplings",
        solver="Nine exact instantaneous regional observations; no integration",
        timestep=None,
        seed=None,
        result_status=ClaimStatus.DERIVED,
        operator_sequence=("self_organization", "coupling"),
        telemetry=(
            "regional weighted-total and variance budgets",
            "source channels",
            "conditional child response",
        ),
        controls=(
            "pinned input",
            "actual ancestry",
            "one snapshot",
            "no induced normalization",
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
    print(f"Wrote retained regional balance audit to {args.output}")


if __name__ == "__main__":
    main()
