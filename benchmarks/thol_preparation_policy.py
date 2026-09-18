"""Separate supplied preparation, eligible births and built-in runtime policy.

Six declared C8 preparations reuse real public prefixes and the same physical
partition. Each continuation replays its preparation afresh. The rotation is
a transported labeling/order control, not an unmarked symmetry theorem.
"""

from __future__ import annotations

import argparse
from collections import deque
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

from benchmarks.selection_birth_closure import (  # noqa: E402
    SELECTORS, _policy, _run_prepared_selector,
)
from benchmarks.thol_birth_transport import (  # noqa: E402
    PREPARATIONS, prepare_birth_selection_source,
)
from benchmarks.thol_eligibility_dispatch import (  # noqa: E402
    _eligibility_record, _run_branch,
)
from benchmarks.thol_pressure_feedback import _payload, _state  # noqa: E402
from tnfr.operators.self_organization_selection import (  # noqa: E402
    observe_self_organization_eligibility,
)
from tnfr.research.claims import ClaimStatus  # noqa: E402
from tnfr.research.core_manifests import (  # noqa: E402
    CoreExperimentManifest, current_git_source_provenance,
)
from tnfr.utils.io import safe_write  # noqa: E402

ROTATIONS = (0, 3)


def _literal(value):
    """Exact finite projection, preserving float bits and deque retention."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return ("binary64", value.hex())
    if isinstance(value, complex):
        return ("complex", _literal(value.real), _literal(value.imag))
    if isinstance(value, Fraction):
        return ("rational", value.numerator, value.denominator)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, deque):
        return ("deque", value.maxlen, tuple(map(_literal, value)))
    if isinstance(value, dict):
        return {key: _literal(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(map(_literal, value))
    raise TypeError(f"unreviewed projected value type: {type(value).__name__}")


def _source_projection(graph):
    return {
        **deepcopy(_state(graph)),
        "node_attributes": {
            node: _literal(dict(data)) for node, data in graph.nodes(data=True)
        },
        "ordered_neighbors": {node: tuple(graph.neighbors(node)) for node in graph},
        "node_sample": tuple(graph.graph.get("_node_sample", ())),
    }


def _transport_state(state, inverse):
    """Normalize named support/history fields, without reconstructing a graph."""
    result = deepcopy(state)
    result["nodes"] = tuple(inverse[node] for node in state["nodes"])
    result["edges"] = tuple((inverse[u], inverse[v], data)
                            for u, v, data in state["edges"])
    for name in ("glyph_history", "physical_epi_history", "node_attributes"):
        if name in state:
            result[name] = {inverse[node]: value for node, value in state[name].items()}
    for name in ("children", "hierarchy", "ordered_neighbors"):
        if name in state:
            result[name] = {
                inverse[node]: tuple(inverse[child] for child in children)
                for node, children in state[name].items()
            }
    if "node_sample" in state:
        result["node_sample"] = tuple(inverse[node] for node in state["node_sample"])
    return _literal(result)


def _transport_eligibility(record, inverse):
    result = deepcopy(record)
    result["eligible_nodes"] = tuple(inverse[node] for node in record["eligible_nodes"])
    result["candidates"] = tuple(
        {**row, "node": inverse[row["node"]]} for row in record["candidates"]
    )
    return _literal(result)


def run_preparation_case(preparation, *, rotation=0):
    """Compare policy outcomes at their actual dispatch boundaries."""
    graph, prep = prepare_birth_selection_source(preparation=preparation, rotation=rotation)
    source = _source_projection(graph)
    policy = _policy(graph, preparation=prep)
    explicit = _run_branch(graph, policy_change=None)
    explicit.update(preparation_record=prep, source_projection=source, policy=policy)
    runtime = []
    for name in SELECTORS:
        graph, repeated = prepare_birth_selection_source(
            preparation=preparation, rotation=rotation,
        )
        projected = _source_projection(graph)
        observation = observe_self_organization_eligibility(graph)
        if _source_projection(graph) != projected:
            raise RuntimeError("eligibility observation changed prepared state")
        if _literal(projected) != _literal(source):
            raise RuntimeError("continuation did not replay the same prepared projection")
        row = _run_prepared_selector(graph, repeated, name)
        if len(row["integration"]) != 1 or len(row["selection_contexts"]) != 1:
            raise RuntimeError("one-step runtime control must expose exactly one boundary")
        boundary = row["integration"][0]["before"]
        old_nodes = frozenset(projected["nodes"])
        row.update(
            source_projection=projected,
            eligibility=_eligibility_record(observation),
            dispatch_boundary=boundary,
            new_nodes_at_dispatch=tuple(node for node in boundary["nodes"]
                                        if node not in old_nodes),
            boundary_scope=(
                "Source eligibility and pre-integrator support at t=0.5; the "
                "whole runtime endpoint at t=0.75 also includes Euler and "
                "phase/capacity updates. No equal-horizon endpoint comparison"
            ),
        )
        runtime.append(row)
    return {
        "preparation": preparation, "rotation": rotation,
        "explicit": explicit, "runtime": tuple(runtime),
    }


def _rotation_control(base, moved):
    shift = moved["rotation"]
    identity = {node: node for node in range(8)}
    inverse = {node: (node - shift) % 8 for node in range(8)}
    left, right = base["explicit"], moved["explicit"]
    source_equal = (
        _transport_state(left["source_projection"], identity)
        == _transport_state(right["source_projection"], inverse)
    )
    eligibility_equal = (
        _transport_eligibility(left["eligibility"], identity)
        == _transport_eligibility(right["eligibility"], inverse)
    )
    # New node names are allocated by the existing owner; compare by lineage.
    base_birth_map, moved_birth_map = dict(identity), dict(inverse)
    for branch, mapping in ((left, base_birth_map), (right, moved_birth_map)):
        for parent, child in branch["parent_children"]:
            mapping[child] = ("birth", mapping[parent])
    dispatch_equal = (
        _transport_state(left["after"], base_birth_map)
        == _transport_state(right["after"], moved_birth_map)
    )
    runtime = []
    for first, second in zip(base["runtime"], moved["runtime"], strict=True):
        runtime.append({
            "selector": first["selector"],
            "proposals_equal": first["actual_selector_proposals"] == {
                inverse[node]: glyph
                for node, glyph in second["actual_selector_proposals"].items()
            },
            "dispatch_state_equal": (
                _transport_state(first["dispatch_boundary"], identity)
                == _transport_state(second["dispatch_boundary"], inverse)
            ),
            "endpoint_state_equal": (
                _transport_state(first["endpoint"], identity)
                == _transport_state(second["endpoint"], inverse)
            ),
        })
    return {
        "preparation": base["preparation"], "rotation": shift,
        "source_projection_equal": source_equal,
        "eligibility_equal": eligibility_equal,
        "explicit_dispatch_state_equal": dispatch_equal,
        "runtime": tuple(runtime),
        "scope": (
            "Exact binary64 equality of declared projections after inverse "
            "transport of labels and order, on this finite pair only. Child "
            "labels use parent lineage. No phase-gauge, arbitrary insertion "
            "order, complete-state or universal equivariance theorem"
        ),
    }


def run_study():
    cases = tuple(run_preparation_case(mode, rotation=rotation)
                  for mode in PREPARATIONS for rotation in ROTATIONS)
    return {
        "cases": cases,
        "rotation_controls": tuple(_rotation_control(cases[index], cases[index + 1])
                                   for index in range(0, len(cases), 2)),
        "limitations": (
            "All prefixes, including the common all-node prefix, are supplied inputs",
            "IL/OZ can change phase/pressure as well as grammar history",
            "Initial checkerboard EPI, winding phase, support and capacity are supplied",
            "All-eligible public dispatch is an explicit policy, not spontaneous generation",
            "Built-in runtime dispatch has different semantics and a later whole-step endpoint",
            "No UM, post-birth physical flow, restoration or sustained pattern is certified",
            "Finite relabeling evidence does not derive physical particles or correspondence",
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=(
        ROOT / "artifacts/research/thol_preparation_policy_2026_09_18.json"
    ))
    args = parser.parse_args()
    scope = (
        "src/tnfr", "benchmarks/thol_preparation_policy.py",
        "benchmarks/selection_birth_closure.py", "benchmarks/thol_eligibility_dispatch.py",
        "benchmarks/thol_birth_transport.py", "benchmarks/thol_pressure_feedback.py",
        "benchmarks/capacity_localization.py",
    )
    provenance = current_git_source_provenance(ROOT, scope)
    sha, dirty, digest = provenance
    manifest = CoreExperimentManifest(
        claim_id="O1.b-THOL-preparation-and-policy-dependence", git_sha=sha,
        source_dirty=dirty, dirty_source_hash=digest,
        versions={"python": platform.python_version(), "networkx": nx.__version__,
                  "numpy": np.__version__},
        graph_construction="Same initialized C8; actual none/single/all-node prefixes; rotation 3",
        capacity_specification="Unit preparation, unchanged default factors and thresholds",
        solver="Two refreshed Euler preparation steps; runtime controls add one ordinary step",
        timestep=0.25, seed=17, result_status=ClaimStatus.MEASURED,
        operator_sequence=("declared IL/OZ prefix or none", "two physical steps",
                           "explicit all-eligible THOL or actual built-in runtime selection"),
        telemetry=("full eligibility", "supplied marks", "birth support", "transported state"),
        controls=("no prefix", "all-node prefix", "rotation", "unchanged built-in selectors"),
        artifacts=(str(args.output),),
    )
    manifest.validate_for_admission()
    report = {"manifest": manifest.to_dict(), "source_scope": scope, **run_study()}
    if current_git_source_provenance(ROOT, scope) != provenance:
        raise RuntimeError("source changed while executing the study")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    safe_write(args.output, lambda stream: stream.write(
        json.dumps(_payload(report), indent=2, allow_nan=False) + "\n"
    ))
    print(f"Wrote bounded preparation/policy study to {args.output}")


if __name__ == "__main__":
    main()
