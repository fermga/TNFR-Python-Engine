"""Observe causal THOL birth, admitted UM attachment and subsequent transport.

The physical preparation crosses the existing THOL threshold without changing
its factors. A public parent-target Coupling decides whether the isolated
child obtains an edge. Matched disabled-link and stale-sample controls retain
the same nodal preparation. No empirical correspondence is asserted.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
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

from benchmarks.capacity_localization import build_cycle  # noqa: E402
from benchmarks.thol_pressure_feedback import (  # noqa: E402
    _acceleration,
    _payload,
    _state,
)
from tnfr.config import inject_defaults  # noqa: E402
from tnfr.dynamics.dnfr import default_compute_delta_nfr  # noqa: E402
from tnfr.dynamics.sampling import update_node_sample  # noqa: E402
from tnfr.operators import (  # noqa: E402
    build_operator_event_schedule,
    build_physical_flow_partition,
    execute_operator_event_schedule,
)
from tnfr.operators._coupling_stage_kernel import (  # noqa: E402
    propose_coupling_stage,
)
from tnfr.operators.definitions import (  # noqa: E402
    Coherence,
    Coupling,
    Dissonance,
    SelfOrganization,
    Silence,
)
from tnfr.operators.factor_contracts import (  # noqa: E402
    resolve_runtime_operator_factors,
)
from tnfr.operators.grammar_dynamics import validate_candidate  # noqa: E402
from tnfr.operators.grammar_execution import ValidatedSequence  # noqa: E402
from tnfr.operators.network_stage import TWO_PHASE_JACOBI  # noqa: E402
from tnfr.operators.self_organization import _configured_tau  # noqa: E402
from tnfr.operators.word_execution import execute_network_operator_stage  # noqa: E402
from tnfr.physics.support_transport import (  # noqa: E402
    observe_support_transport,
    observe_support_transport_euler,
    observe_support_transport_reset,
)
from tnfr.research.claims import ClaimStatus  # noqa: E402
from tnfr.research.core_manifests import (  # noqa: E402
    CoreExperimentManifest,
    current_git_source_provenance,
)
from tnfr.rng import base_seed  # noqa: E402
from tnfr.sdk._state import copy_graph_state  # noqa: E402
from tnfr.types import Glyph  # noqa: E402
from tnfr.utils import angle_diff, ensure_node_offset_map  # noqa: E402
from tnfr.validation import validate_sequence  # noqa: E402

INITIAL_EPI = (2.0, 0.5) * 4
STEPS = (0.25, 0.25)
WORD = ("coherence", "dissonance", "self_organization", "coupling", "silence")
CASES = ("attached", "links_disabled", "stale_sample")
PREPARATIONS = ("none", "single_parent", "all_nodes")


def _support_readout(graph):
    snapshot = observe_support_transport(graph)
    weights = {
        key: Fraction.from_float(float(value))
        for key, value in graph.graph["_dnfr_weights"].items()
    }
    partial = tuple(
        weights["epi"] * epi + weights["vf"] * vf + weights["topo"] * topo
        for epi, vf, topo in zip(
            snapshot.epi_gradient,
            snapshot.capacity_gradient,
            snapshot.topology_gradient,
            strict=True,
        )
    )
    return snapshot, {
        "snapshot": asdict(snapshot),
        "normalized_channel_weights": weights,
        "exact_nonphase_pressure": partial,
        "exact_stored_minus_nonphase_pressure": tuple(
            stored - modeled
            for stored, modeled in zip(
                snapshot.stored_pressure,
                partial,
                strict=True,
            )
        ),
        "scope": (
            "Captured stored pressure minus exact weighted EPI, unweighted "
            "capacity and topology channels; canonical phase and arithmetic "
            "remain an explicit residual"
        ),
    }


def _admitted_apply(graph, operator, *, node=0):
    admission = validate_candidate(graph, node, operator.glyph.value)
    if not admission.allowed:
        raise RuntimeError(f"unexpected live refusal: {admission}")
    operator(graph, node, collect_metrics=True)
    return {
        "candidate": admission.candidate,
        "allowed": admission.allowed,
        "scope": "Incremental admission followed by actual public application",
    }


def _prepare_birth_source(graph, *, preparation="single_parent", rotation=0):
    """Execute the shared causal preparation, stopping before any THOL call."""
    projection_order = tuple(graph)
    targets = (
        ()
        if preparation == "none"
        else projection_order if preparation == "all_nodes" else projection_order[:1]
    )
    initial = _state(graph)
    update_node_sample(graph, step=0)
    sample = tuple(graph.graph["_node_sample"])
    admissions = []
    prefix = []
    for operator in (Coherence(), Dissonance()) if targets else ():
        if preparation == "single_parent":
            admitted = _admitted_apply(graph, operator, node=targets[0])
            admissions.append(admitted)
            prefix.append(
                {
                    "glyph": operator.glyph.value,
                    "targets": targets,
                    "route": "direct_public",
                    "admissions": (admitted,),
                    "stage_result": None,
                }
            )
        else:
            checked = tuple(
                validate_candidate(graph, node, operator.glyph.value)
                for node in targets
            )
            if any(not admission.allowed for admission in checked):
                raise RuntimeError(f"unexpected simultaneous prefix refusal: {checked}")
            stage = execute_network_operator_stage(graph, operator, targets)
            if (
                stage.schedule != TWO_PHASE_JACOBI
                or stage.nodes_processed != len(targets)
                or stage.glyph != operator.glyph.value
                or stage.operator != operator.name
            ):
                raise RuntimeError(
                    "preparation requires the complete simultaneous public stage"
                )
            admitted = tuple(
                {
                    "node": node,
                    "candidate": admission.candidate,
                    "allowed": admission.allowed,
                    "scope": "Incremental admission followed by simultaneous public application",
                }
                for node, admission in zip(targets, checked, strict=True)
            )
            admissions.extend(admitted)
            prefix.append(
                {
                    "glyph": operator.glyph.value,
                    "targets": targets,
                    "route": "simultaneous_public",
                    "admissions": admitted,
                    "stage_result": asdict(stage),
                }
            )
    after_prefix = _state(graph)
    schedule = build_operator_event_schedule(
        (),
        start_time=0.0,
        flow_durations=(sum(STEPS),),
    )
    partition = build_physical_flow_partition(schedule.intervals[0], STEPS)
    execution = execute_operator_event_schedule(
        graph,
        schedule,
        method="euler",
        physical_flow_partitions=(partition,),
    )
    evidence = execution.physical_flow_partition_evidence[0]
    observed = _acceleration(graph, projection_order[0])
    before = _state(graph)
    threshold = _configured_tau(graph.graph, {})
    epi_weight = Fraction.from_float(graph.graph["_dnfr_weights"]["epi"])
    exact_acceleration = 3 * epi_weight**2
    remaining_amplitude = Fraction(3, 4) * (1 - epi_weight / 2) ** 2
    exact_endpoint = tuple(
        Fraction(5, 4) + (1 if index % 2 == 0 else -1) * remaining_amplitude
        for index in range(8)
    )
    return {
        "preparation": preparation,
        "rotation": rotation,
        "prep_targets": targets,
        "projection_node_order": projection_order,
        "acceleration_reference_node": projection_order[0],
        "actual_prefix": tuple(prefix),
        "initial": initial,
        "initial_candidate_sample": sample,
        "after_prefix": after_prefix,
        "actual_admissions": admissions,
        "physical_steps": STEPS,
        "physical_acceleration": observed,
        "default_birth_threshold": threshold,
        "checkerboard_reference": {
            "represented_epi_weight": epi_weight,
            "exact_model_acceleration": exact_acceleration,
            "exact_model_threshold_margin": (
                exact_acceleration - Fraction.from_float(threshold)
            ),
            "exact_observed_acceleration_residual": (
                Fraction.from_float(observed["observed_acceleration"])
                - exact_acceleration
            ),
            "exact_model_endpoint": exact_endpoint,
            "exact_endpoint_residual": tuple(
                Fraction.from_float(actual) - ideal
                for actual, ideal in zip(
                    before["epi"],
                    exact_endpoint,
                    strict=True,
                )
            ),
            "scope": (
                "Detached exact regular-cycle mode reference: acceleration "
                "3e^2 and endpoint 5/4 +/- 3/4*(1-e/2)^2. Phase realization "
                "and binary64 arithmetic remain captured residuals"
            ),
        },
        "before_birth": before,
        "physical_boundaries": [
            {
                "time": boundary.time,
                "epi": boundary.after.epi,
                "pressure": boundary.after.delta_nfr,
                "pressure_only_refresh": boundary.nonpressure_state_preserved,
            }
            for boundary in evidence.boundary_observations
        ],
        "segment_methods": tuple(
            item.resolved_method for item in evidence.segment_flow_evidence
        ),
        "clipping_applied": tuple(
            item.clipping_applied for item in evidence.segment_flow_evidence
        ),
        "scope": (
            (
                "Actual IL/OZ, then three executor-owned physical samples. "
                if preparation == "single_parent"
                else f"Declared {preparation} prefix, then three executor-owned physical samples. "
            )
            + "No THOL has executed. EPI chart and hard bounds are explicit preparation; "
            "unit capacity, THOL factor and birth threshold are unchanged"
        ),
    }


def _prepare_birth(graph):
    record = _prepare_birth_source(graph)
    record["actual_admissions"].append(_admitted_apply(graph, SelfOrganization()))
    raw = _state(graph)
    children = tuple(graph.nodes[0].get("sub_nodes", ()))
    default_compute_delta_nfr(graph)
    _, support = _support_readout(graph)
    record.update(
        raw_after_birth=raw,
        after_birth_refresh=_state(graph),
        after_birth_support=support,
        children=children,
        child_degrees_after_birth=tuple(graph.degree(node) for node in children),
        scope=(
            "Actual IL/OZ, then three executor-owned physical samples and "
            "public THOL. EPI chart and hard bounds are explicit preparation; "
            "unit capacity, THOL factor and birth threshold are unchanged"
        ),
    )
    return record


def _birth_graph(case):
    if case not in CASES:
        raise ValueError(f"case must be one of {CASES}")
    graph = build_cycle(8, epi=INITIAL_EPI)
    graph.graph.pop("UM_BIDIRECTIONAL")  # Restore the actual default True branch.
    graph.graph["UM_FUNCTIONAL_LINKS"] = case != "links_disabled"
    inject_defaults(graph)
    return graph


def prepare_birth_selection_source(*, preparation="single_parent", rotation=0):
    """Return a declared C8 preparation before selection or public birth.

    The default preserves the original parent-zero IL/OZ preparation. ``none``
    skips that prefix; ``all_nodes`` executes each prefix operator as one
    simultaneous stage. All modes retain the same physical flow partition.
    Rotation transports initialized node data and iteration order before any
    execution history is recorded. It is a relabeling, not a phase shift or
    a change to the initialized checkerboard. The mark is an explicit input.
    """
    if type(preparation) is not str or preparation not in PREPARATIONS:
        raise ValueError(f"preparation must be one of {PREPARATIONS}")
    if type(rotation) is not int or not 0 <= rotation < 8:
        raise ValueError("rotation must be a non-boolean integer in [0, 7]")
    graph = _birth_graph("attached")
    if rotation:
        mapping = {node: (node + rotation) % 8 for node in graph}
        neighbor_order = {
            mapping[node]: tuple(
                mapping[neighbor] for neighbor in graph.neighbors(node)
            )
            for node in graph
        }
        # Initial pressure construction has populated rebuildable caches.
        # Detach them before relabeling; no executed history exists yet.
        initialized = copy_graph_state(graph)
        graph = nx.relabel_nodes(initialized, mapping, copy=True)
        # Graph copying rebuilds undirected adjacency in edge-list order.
        # Restore each transported order without replacing edge-data objects:
        # reductions and selection must see the same ordered neighborhoods.
        for node, neighbors in neighbor_order.items():
            adjacency = graph._adj[node]
            ordered = tuple((neighbor, adjacency[neighbor]) for neighbor in neighbors)
            adjacency.clear()
            adjacency.update(ordered)
        if "_dnfrmax_node" in graph.graph:
            graph.graph["_dnfrmax_node"] = mapping[graph.graph["_dnfrmax_node"]]
        default_compute_delta_nfr(graph)
    return graph, _prepare_birth_source(
        graph,
        preparation=preparation,
        rotation=rotation,
    )


def _couple_parent(graph, *, refresh_sample):
    before_sample = tuple(graph.graph["_node_sample"])
    if refresh_sample:
        update_node_sample(graph, step=1)
    actual_sample = tuple(graph.graph["_node_sample"])
    before = _state(graph)
    before_snapshot, before_support = _support_readout(graph)
    factors = resolve_runtime_operator_factors(
        graph.graph.get("GLYPH_FACTORS"),
        Glyph.UM,
        graph.graph,
    )
    functional = bool(graph.graph.get("UM_FUNCTIONAL_LINKS", True))
    seed = base_seed(graph) if functional else None
    offsets = dict(ensure_node_offset_map(graph)) if functional else {}
    proposal = propose_coupling_stage(
        graph,
        (0,),
        factors,
        resolved_seed=seed,
        node_offsets=offsets,
    )
    admission = _admitted_apply(graph, Coupling())
    raw = _state(graph)
    default_compute_delta_nfr(graph)
    refreshed = _state(graph)
    after_snapshot, after_support = _support_readout(graph)
    reset = observe_support_transport_reset(before_snapshot, after_snapshot)
    target = proposal.target_proposals[0]
    old_edges = {frozenset((u, v)) for u, v, _ in before["edges"]}
    actual_edges = tuple(
        (u, v, dict(data))
        for u, v, data in graph.edges(data=True)
        if frozenset((u, v)) not in old_edges
    )
    phases = dict(zip(raw["nodes"], raw["phase"], strict=True))
    return {
        "target": 0,
        "actual_admission": admission,
        "functional_links": functional,
        "bidirectional": bool(graph.graph.get("UM_BIDIRECTIONAL", True)),
        "candidate_limit": graph.graph["UM_CANDIDATE_COUNT"],
        "candidate_mode": graph.graph["UM_CANDIDATE_MODE"],
        "sample_before_optional_refresh": before_sample,
        "sampling_refresh_executed": refresh_sample,
        "actual_candidate_sample": actual_sample,
        "before": before,
        "before_support": before_support,
        "readonly_kernel_proposal": asdict(proposal),
        "proposal_scope": (
            "Pure production-kernel replay on the preoperator graph with "
            "the actual seed and node offsets; actual public commits are "
            "recorded independently and compared, not assumed"
        ),
        "compatible_existing_neighbors": target.compatible_neighbors,
        "effective_phase_limit": target.effective_phase_limit,
        "compatibility_threshold": target.compatibility_threshold,
        "actual_new_edges": actual_edges,
        "actual_new_edge_phase_separations": tuple(
            abs(angle_diff(phases[u], phases[v])) for u, v, _ in actual_edges
        ),
        "raw_after_coupling": raw,
        "after_refresh": refreshed,
        "after_support": after_support,
        "support_reset": asdict(reset),
    }


def _refreshed_postbirth_flow(graph):
    """Observe two existing held-input event intervals on fixed born support."""
    segments = []
    for dt in STEPS:
        default_compute_delta_nfr(graph)
        before = _state(graph)
        before_snapshot, before_support = _support_readout(graph)
        schedule = build_operator_event_schedule(
            (),
            start_time=before["time"],
            flow_durations=(dt,),
        )
        execution = execute_operator_event_schedule(
            graph,
            schedule,
            method="euler",
            include_flow_certificates=True,
        )
        evidence = execution.flow_interval_evidence[0]
        raw = _state(graph)
        default_compute_delta_nfr(graph)
        after = _state(graph)
        after_snapshot, after_support = _support_readout(graph)
        budget = observe_support_transport_euler(
            before_snapshot,
            after_snapshot,
            dt=dt,
        )
        segments.append(
            {
                "before": before,
                "before_support": before_support,
                "raw_after_integrator": raw,
                "after_refresh": after,
                "after_support": after_support,
                "duration": dt,
                "method": evidence.resolved_method,
                "clipping_applied": evidence.clipping_applied,
                "exact_euler_budget": asdict(budget),
                "scope": (
                    "One existing event-executor interval; canonical refresh is "
                    "explicitly called at the live endpoints by this benchmark. "
                    "Atomicity covers each executor invocation, not the full "
                    "birth/attachment/flow orchestration"
                ),
            }
        )
    return segments


def prepare_birth_transport_support(case="attached"):
    """Return the actual live post-UM graph before flow and final SHA.

    The retained complete word is statically admitted. Only IL/OZ/THOL/UM
    has executed at this boundary; callers own subsequent flow and closure.
    """
    graph = _birth_graph(case)
    context = {"initial_epi_nonzero": graph.nodes[0]["EPI"] > 0}
    operators = (Coherence(), Dissonance(), SelfOrganization(), Coupling(), Silence())
    ValidatedSequence(operators, context=context)
    compatibility = validate_sequence(list(WORD), context=context)
    if not compatibility.passed:
        raise RuntimeError("the declared complete word was not admitted")
    preparation = _prepare_birth(graph)
    coupling = _couple_parent(graph, refresh_sample=case != "stale_sample")
    return graph, {
        "case": case,
        "word": WORD,
        "whole_word_validation": {
            "initialized_context": context,
            "both_validators_passed": True,
            "scope": "Static complete-word validation; live gates checked per call",
        },
        "preparation": preparation,
        "coupling": coupling,
    }


def run_birth_transport_case(case):
    """Execute one declared case with a live admitted parent-only word."""
    graph, preparation_record = prepare_birth_transport_support(case)
    flow = _refreshed_postbirth_flow(graph)
    measured_endpoint = _state(graph)
    closure_admission = _admitted_apply(graph, Silence())
    return {
        **preparation_record,
        "postbirth_flow": flow,
        "measured_endpoint_before_closure": measured_endpoint,
        "closure": {
            "admission": closure_admission,
            "after": _state(graph),
            "scope": "Actual parent SHA after the measured flow endpoint",
        },
        "scope": (
            "Finite causal birth, optional functional attachment and flow on "
            "its subsequently fixed support. Initial fields are prepared data; "
            "all later EPI changes use canonical operators or shared nodal "
            "integration. No spontaneous vacuum origin, changing-node event "
            "executor, sustained localization or future persistence is inferred"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/research/thol_birth_transport.json",
    )
    args = parser.parse_args()
    scope = (
        "src/tnfr",
        "benchmarks/capacity_localization.py",
        "benchmarks/thol_pressure_feedback.py",
        "benchmarks/thol_birth_transport.py",
    )
    provenance = current_git_source_provenance(ROOT, scope)
    sha, dirty, digest = provenance
    manifest = CoreExperimentManifest(
        claim_id="O3.a-causal-thol-birth-coupling-transport",
        git_sha=sha,
        source_dirty=dirty,
        dirty_source_hash=digest,
        versions={
            "python": platform.python_version(),
            "networkx": nx.__version__,
            "numpy": np.__version__,
        },
        graph_construction="Unit C8 with checkerboard EPI (2,.5); unit capacity",
        capacity_specification="Unit preparation; default THOL, UM and final SHA",
        solver="Existing refreshed partition and explicitly refreshed event intervals",
        timestep=STEPS[0],
        seed=17,
        result_status=ClaimStatus.MEASURED,
        operator_sequence=WORD,
        telemetry=(
            "physical acceleration",
            "candidate sample and actual UM edge",
            "transport pressure channels",
            "support and Euler energy budgets",
        ),
        controls=("functional links disabled", "sample retained from before birth"),
        artifacts=(str(args.output),),
    )
    manifest.validate_for_admission()
    report = {
        "manifest": manifest.to_dict(),
        "source_scope": scope,
        "cases": [run_birth_transport_case(case) for case in CASES],
        "experimental_status": "No empirical correspondence tested",
    }
    if current_git_source_provenance(ROOT, scope) != provenance:
        raise RuntimeError("source changed while executing the research artifact")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_payload(report), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote finite THOL birth/transport observations to {args.output}")


if __name__ == "__main__":
    main()
