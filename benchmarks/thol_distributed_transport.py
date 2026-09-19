"""Observe distributed public THOL births, simultaneous UM and refreshed flow.

The all-node IL/OZ preparation and default operator factors are reused. The
dispatch-all-eligible and parent-only UM choices are explicit execution policies.
No prescribed birth edges, fabricated histories or empirical claims are used.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
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

from benchmarks.thol_birth_transport import (  # noqa: E402
    CASES,
    STEPS,
    _support_readout,
    prepare_birth_selection_source,
)
from benchmarks.thol_eligibility_dispatch import _eligibility_record  # noqa: E402
from benchmarks.thol_pressure_feedback import _payload, _state  # noqa: E402
from tnfr.dynamics.dnfr import default_compute_delta_nfr  # noqa: E402
from tnfr.dynamics.sampling import update_node_sample  # noqa: E402
from tnfr.operators import (  # noqa: E402
    build_operator_event_schedule,
    build_physical_flow_partition,
    execute_operator_event_schedule,
)
from tnfr.operators._coupling_stage_kernel import propose_coupling_stage  # noqa: E402
from tnfr.operators.definitions import Coupling  # noqa: E402
from tnfr.operators.factor_contracts import (
    resolve_runtime_operator_factors,
)  # noqa: E402
from tnfr.operators.grammar_dynamics import validate_candidate  # noqa: E402
from tnfr.operators.network_stage import TWO_PHASE_JACOBI  # noqa: E402
from tnfr.operators.self_organization_selection import (  # noqa: E402
    execute_eligible_self_organization_stage,
)
from tnfr.operators.word_execution import execute_network_operator_stage  # noqa: E402
from tnfr.physics.forcing_realization import (  # noqa: E402
    capture_non_epi_forcing,
    decompose_non_epi_forcing,
)
from tnfr.physics.support_transport import (  # noqa: E402
    _from_data,
    observe_support_transport_euler,
    observe_support_transport_reset,
)
from tnfr.research.claims import ClaimStatus  # noqa: E402
from tnfr.research.core_manifests import (  # noqa: E402
    CoreExperimentManifest,
    current_git_source_provenance,
)
from tnfr.rng import base_seed  # noqa: E402
from tnfr.types import Glyph  # noqa: E402
from tnfr.utils import angle_diff, ensure_node_offset_map  # noqa: E402
from tnfr.utils.io import safe_write  # noqa: E402


def _require_stage(stage, operator, targets):
    if (
        stage.schedule != TWO_PHASE_JACOBI
        or stage.glyph != operator.glyph.value
        or stage.operator != operator.name
        or stage.nodes_processed != len(targets)
    ):
        raise RuntimeError("the complete simultaneous public stage was not executed")


def _forcing_readout(graph):
    observation = capture_non_epi_forcing(graph)
    return {
        "observation": asdict(observation),
        "components": decompose_non_epi_forcing(observation),
    }


def _support_inventory(snapshot):
    positive = nx.Graph()
    positive.add_nodes_from(range(len(snapshot.nodes)))
    positive.add_edges_from((i, j) for i, j, _ in snapshot.conductance)
    return {
        "positive_conductance_components": tuple(
            tuple(snapshot.nodes[i] for i in sorted(component))
            for component in nx.connected_components(positive)
        ),
        "directed_positive_conductance_entries": len(snapshot.conductance),
        "directed_unique_support_entries": sum(map(len, snapshot.support_neighbors)),
        "minimum_capacity": min(snapshot.capacity),
    }


def _weighted_source_budget(coupling, parents, children):
    """Account for the connected witness's captured channels, without evolution.

    The full disconnected controls are explicitly outside the positive-H
    reference. No child is discarded to make that reference admissible.
    """
    if len(coupling["support_inventory"]["positive_conductance_components"]) != 1:
        return {
            "available": False,
            "reason": "Whole generated support is disconnected; no positive-H reference",
        }
    captured = coupling["refreshed_forcing"]
    observation = captured["observation"]
    snapshot = observation["snapshot"]
    nodes = snapshot["nodes"]
    if nodes != parents + children:
        raise ValueError(
            "capacity-source witness requires the complete parent/child order"
        )
    index = {node: i for i, node in enumerate(nodes)}
    weights = {(nodes[i], nodes[j]): value for i, j, value in snapshot["conductance"]}
    capacities = dict(zip(nodes, snapshot["capacity"], strict=True))
    child_capacities = {capacities[node] for node in children}
    if (
        not children
        or len(child_capacities) != 1
        or any(capacities[node] != 1 for node in parents)
    ):
        raise ValueError(
            "capacity-source witness requires unit parents and uniform children"
        )
    child_capacity = next(iter(child_capacities))
    if not 0 < child_capacity < 1:
        raise ValueError(
            "capacity-source witness requires positive subunit child capacity"
        )
    parent_set, child_set = set(parents), set(children)
    incidence = []
    for node in nodes:
        neighbors = tuple(nodes[j] for j in snapshot["support_neighbors"][index[node]])
        if any((node, neighbor) not in weights for neighbor in neighbors):
            raise ValueError("capacity-source witness excludes zero-weight support")
        if node in child_set:
            if not neighbors or not set(neighbors) <= parent_set:
                raise ValueError("every child must have parent-only positive neighbors")
            continue
        old = tuple(neighbor for neighbor in neighbors if neighbor in parent_set)
        new = tuple(neighbor for neighbor in neighbors if neighbor in child_set)
        if len(old) != 2 or any(weights[node, neighbor] != 1 for neighbor in old):
            raise ValueError(
                "every parent must retain two unit-conductance parent neighbors"
            )
        incidence.append(
            {
                "parent": node,
                "child_count": len(new),
                "child_strength": sum((weights[node, n] for n in new), Fraction(0)),
            }
        )
    strengths = tuple(
        sum((w for (source, _), w in weights.items() if source == node), Fraction(0))
        for node in nodes
    )
    metric = tuple(
        d / nu for d, nu in zip(strengths, snapshot["capacity"], strict=True)
    )
    mass = sum(metric, Fraction(0))

    def weighted(values):
        return sum(
            (d * value for d, value in zip(strengths, values, strict=True)), Fraction(0)
        )

    components = dict(captured["components"])
    epi = tuple(observation["epi_weight"] * value for value in snapshot["epi_gradient"])
    sources = {
        "epi": weighted(epi),
        **{name: weighted(values) for name, values in components.items()},
    }
    w_vf = dict(observation["normalized_weights"])["vf"]
    closed = (
        2
        * w_vf
        * (1 - child_capacity)
        * sum(
            (
                (row["child_strength"] - row["child_count"]) / (2 + row["child_count"])
                for row in incidence
            ),
            Fraction(0),
        )
    )
    kernel_defect = weighted(observation["kernel_pressure_defect"]) / mass
    stored_defect = weighted(observation["stored_pressure_residual"]) / mass
    represented_rate = weighted(snapshot["stored_pressure"]) / mass
    residual = (
        represented_rate
        - sum(sources.values(), Fraction(0)) / mass
        - (kernel_defect + stored_defect)
    )
    if sources["epi"] or sources["vf"] != closed or residual:
        raise RuntimeError("exact weighted source accounting failed")
    return {
        "available": True,
        "strengths": strengths,
        "metric_weights": metric,
        "parent_child_incidence": tuple(incidence),
        "child_capacity": child_capacity,
        "capacity_weighted_source": sources["vf"],
        "capacity_closed_form_source": closed,
        "capacity_identity_residual": sources["vf"] - closed,
        "weighted_source_by_channel": sources,
        "mean_rate_by_channel": {name: value / mass for name, value in sources.items()},
        "kernel_mean_rate_defect": kernel_defect,
        "stored_pressure_mean_rate_defect": stored_defect,
        "represented_weighted_mean_rate": represented_rate,
        "mean_rate_identity_residual": residual,
        "scope": (
            "Exact accounting of independently captured channels at the refreshed post-UM "
            "state. Instantaneous fixed-metric nodal rate, not a derivative fitted from "
            "endpoints, future drift theorem, equilibrium or restoration certificate."
        ),
    }


def _support_from_flow_state(reference, state):
    """Bridge a sealed nodal state to the existing exact support read-out.

    ``support_transport._from_data`` rebuilds every derived cache. The native
    trace authenticates flow, while this detached arithmetic supplies its
    support-energy accounting; it does not create another executor certificate.
    """
    size = len(reference.nodes)
    dense = [[Fraction(0) for _ in range(size)] for _ in range(size)]
    for i, j, weight in reference.conductance:
        dense[i][j] = weight
    if (
        state.nodes != reference.nodes
        or state.conductance != tuple(tuple(row) for row in dense)
        or state.exact_nu_f != reference.capacity
    ):
        raise ValueError(
            "flow state must retain the captured node order, support and capacity"
        )
    for raw, exact in (
        (state.epi, state.exact_epi),
        (state.nu_f, state.exact_nu_f),
        (state.delta_nfr, state.exact_delta_nfr),
    ):
        if tuple(Fraction.from_float(float(value)) for value in raw) != exact:
            raise ValueError("flow state represented and exact channels differ")
    return _from_data(
        reference.nodes,
        reference.conductance,
        reference.support_neighbors,
        state.exact_epi,
        state.exact_nu_f,
        state.exact_delta_nfr,
    )


def _couple_parents(graph, parents, *, case, target_policy=None):
    graph.graph["UM_FUNCTIONAL_LINKS"] = case != "links_disabled"
    sample_before = tuple(graph.graph["_node_sample"])
    refresh_sample = case != "stale_sample"
    if refresh_sample:
        update_node_sample(graph, step=1)
    sample_used = tuple(graph.graph["_node_sample"])
    before = deepcopy(_state(graph))
    before_snapshot, before_support = _support_readout(graph)
    factors = resolve_runtime_operator_factors(
        graph.graph.get("GLYPH_FACTORS"),
        Glyph.UM,
        graph.graph,
    )
    functional = bool(graph.graph["UM_FUNCTIONAL_LINKS"])
    proposal = propose_coupling_stage(
        graph,
        parents,
        factors,
        resolved_seed=base_seed(graph) if functional else None,
        node_offsets=dict(ensure_node_offset_map(graph)) if functional else {},
    )
    checked = tuple(validate_candidate(graph, parent, "UM") for parent in parents)
    if any(not admission.allowed for admission in checked):
        raise RuntimeError(f"public parent UM admission failed: {checked}")
    operator = Coupling()
    stage = execute_network_operator_stage(graph, operator, parents)
    _require_stage(stage, operator, parents)
    raw = deepcopy(_state(graph))
    raw_snapshot, raw_support = _support_readout(graph)
    raw_forcing = _forcing_readout(graph)
    for update in proposal.node_updates:
        index = raw["nodes"].index(update.node)
        for key, expected in (
            ("phase", update.theta_after),
            ("capacity", update.vf_after),
            ("pressure", update.dnfr_after),
        ):
            if expected is not None and raw[key][index] != expected:
                raise RuntimeError(
                    "actual shared UM update differs from its kernel proposal"
                )
    old_edges = {frozenset((u, v)) for u, v, _ in before["edges"]}
    new_edges = tuple(
        (u, v, deepcopy(data))
        for u, v, data in graph.edges(data=True)
        if frozenset((u, v)) not in old_edges
    )
    proposed_edges = {
        frozenset((edge.left, edge.right)): edge.weight for edge in proposal.edges
    }
    actual_edges = {frozenset((u, v)): data["weight"] for u, v, data in new_edges}
    if actual_edges != proposed_edges:
        raise RuntimeError(
            "actual shared UM new edges differ from the complete proposal"
        )
    default_compute_delta_nfr(graph)
    refreshed = deepcopy(_state(graph))
    after_snapshot, after_support = _support_readout(graph)
    refreshed_forcing = _forcing_readout(graph)
    reset = observe_support_transport_reset(before_snapshot, raw_snapshot)
    refresh_reset = observe_support_transport_reset(raw_snapshot, after_snapshot)
    phases = dict(zip(raw["nodes"], raw["phase"], strict=True))
    return after_snapshot, {
        "targets": parents,
        "target_policy": (
            target_policy
            if target_policy is not None
            else "All actual birth parents once; newborns are not UM targets"
        ),
        "admissions": tuple(
            {
                "node": parent,
                "candidate": admission.candidate,
                "allowed": admission.allowed,
            }
            for parent, admission in zip(parents, checked, strict=True)
        ),
        "stage_result": asdict(stage),
        "kernel_proposal": asdict(proposal),
        "all_proposed_node_updates_committed": True,
        "all_proposed_edges_committed": True,
        "functional_links": functional,
        "bidirectional": bool(graph.graph.get("UM_BIDIRECTIONAL", True)),
        "candidate_limit": graph.graph["UM_CANDIDATE_COUNT"],
        "candidate_mode": graph.graph["UM_CANDIDATE_MODE"],
        "node_sample_before": sample_before,
        "node_sample_used": sample_used,
        "sampling_refresh_executed": refresh_sample,
        "before": before,
        "after_raw": raw,
        "after_refreshed": refreshed,
        "before_support": before_support,
        "raw_support": raw_support,
        "after_support": after_support,
        "new_edges": new_edges,
        "raw_forcing": raw_forcing,
        "refreshed_forcing": refreshed_forcing,
        "support_inventory": _support_inventory(after_snapshot),
        "new_edge_phase_separations": tuple(
            abs(angle_diff(phases[u], phases[v])) for u, v, _ in new_edges
        ),
        "support_reset_budget": asdict(reset),
        "pressure_refresh_reset_budget": asdict(refresh_reset),
        "scope": "Read-only proposal versus actual simultaneous commits; U3 and grammar retained",
    }


def _physical_flow(graph, reference):
    before = deepcopy(_state(graph))
    schedule = build_operator_event_schedule(
        (),
        start_time=before["time"],
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
    if not evidence.physical_pressure_reevaluated_partition_established:
        raise RuntimeError("fresh physical partition lost its executor provenance")
    boundaries = []
    for boundary in evidence.boundary_observations:
        if not boundary.nonpressure_state_preserved:
            raise RuntimeError("pressure refresh changed protected non-pressure state")
        left = _support_from_flow_state(reference, boundary.before)
        right = _support_from_flow_state(reference, boundary.after)
        boundaries.append(
            {
                "index": boundary.boundary_index,
                "time": boundary.time,
                "exact_time": boundary.exact_time,
                "offset": boundary.offset,
                "callback_name": boundary.callback_name,
                "callback_completed": boundary.callback_completed,
                "nonpressure_state_preserved": boundary.nonpressure_state_preserved,
                "pressure_changed": boundary.pressure_changed,
                "before": asdict(boundary.before),
                "after": asdict(boundary.after),
                "before_support": asdict(left),
                "after_support": asdict(right),
                "refresh_reset_budget": asdict(
                    observe_support_transport_reset(left, right)
                ),
            }
        )
    segments = []
    for index, flow in enumerate(evidence.segment_flow_evidence):
        certificate = flow.certificate
        if (
            certificate is None
            or not flow.integrator_provenance_certified
            or certificate.left != evidence.boundary_observations[index].after
            or certificate.right != evidence.boundary_observations[index + 1].before
            or flow.interval.start_time != boundaries[index]["time"]
            or flow.interval.end_time != boundaries[index + 1]["time"]
        ):
            raise RuntimeError("segment and refreshed boundaries do not align")
        left = _support_from_flow_state(reference, certificate.left)
        right = _support_from_flow_state(reference, certificate.right)
        budget = observe_support_transport_euler(left, right, flow.interval.duration)
        if not certificate.pressure_unchanged:
            raise RuntimeError("pressure did not remain held inside the Euler segment")
        segments.append(
            {
                "index": index,
                "interval": asdict(flow.interval),
                "before": asdict(certificate.left),
                "held_after": asdict(certificate.right),
                "before_support": asdict(left),
                "after_support": asdict(right),
                "exact_euler_budget": asdict(budget),
                "method": flow.resolved_method,
                "clipping_applied": flow.clipping_applied,
                "integrator_provenance_certified": flow.integrator_provenance_certified,
                "pressure_unchanged": certificate.pressure_unchanged,
                "binary64_held_pressure_interval_identified": (
                    flow.runtime_bound_binary64_held_pressure_interval_identified
                ),
                "binary64_interval_identified": flow.runtime_bound_binary64_interval_identified,
                "exact_affine_map_identified": flow.runtime_bound_exact_affine_map_identified,
                "pure_epi_diffusion_eligible": certificate.pure_epi_diffusion_eligible,
                "global_disagreement_contraction_certified": (
                    flow.runtime_bound_global_disagreement_contraction_certified
                ),
            }
        )
    after = deepcopy(_state(graph))
    actual_after, actual_support = _support_readout(graph)
    if actual_after != _support_from_flow_state(
        reference, evidence.boundary_observations[-1].after
    ):
        raise RuntimeError(
            "live final read-out differs from the sealed terminal boundary"
        )
    energy_change = actual_after.dirichlet_energy - reference.dirichlet_energy
    accounted = sum(
        (row["exact_euler_budget"]["energy_change"] for row in segments), Fraction(0)
    )
    if energy_change != accounted:
        raise RuntimeError("finite support-energy telescope failed")
    return {
        "before": before,
        "after": after,
        "after_support": actual_support,
        "after_forcing": _forcing_readout(graph),
        "partition": asdict(partition),
        "boundaries": tuple(boundaries),
        "segments": tuple(segments),
        "pressure_refresh_callback_invocations": evidence.pressure_refresh_callback_invocations,
        "physical_pressure_reevaluated_partition_established": True,
        "all_segment_binary64_replays_identified": evidence.all_segment_binary64_replays_identified,
        "all_segment_binary64_intervals_identified": evidence.all_segment_binary64_intervals_identified,
        "all_segment_exact_affine_maps_identified": evidence.all_segment_exact_affine_maps_identified,
        "exact_energy_change": energy_change,
        "exact_segment_energy_change_sum": accounted,
        "energy_telescope_residual": energy_change - accounted,
        "solver_accuracy_certified": False,
        "mesh_convergence_certified": False,
        "future_or_repeated_behavior_certified": False,
        "scope": (
            "One graph-owned event-free invocation after growth, on fixed captured support. "
            "Native sealed traces authenticate refresh and integration; detached exact support "
            "read-outs supply accounting. The enclosing birth/UM/flow study is not one transaction."
        ),
    }


def prepare_distributed_transport_support(case="attached"):
    """Return actual distributed births and parent UM at t=.5, before flow.

    The graph is live; the returned prefix contains detached observations.
    This shared preparation adds no post-UM evolution or fixed-target model.
    """
    if case not in CASES:
        raise ValueError(f"case must be one of {CASES}")
    graph, preparation = prepare_birth_selection_source(preparation="all_nodes")
    before_birth = deepcopy(_state(graph))
    dispatch = execute_eligible_self_organization_stage(graph)
    births = tuple(dispatch.parent_children)
    parents = tuple(parent for parent, _ in births)
    children = tuple(child for _, child in births)
    if dispatch.stage_result is None or parents != tuple(before_birth["nodes"]):
        raise RuntimeError(
            "the unchanged all-node preparation did not birth at every original node"
        )
    after_birth = deepcopy(_state(graph))
    if (
        any(graph.degree(child) != 0 for child in children)
        or after_birth["epi"][: len(parents)] != before_birth["epi"]
        or tuple(after_birth["nodes"]) != parents + children
    ):
        raise RuntimeError(
            "birth did not preserve old EPI and append isolated children"
        )
    birth = {
        "before": before_birth,
        "after": after_birth,
        "eligibility": _eligibility_record(dispatch.eligibility),
        "stage_result": asdict(dispatch.stage_result),
        "parent_children": births,
        "policy": dispatch.policy,
        "old_epi_preserved": True,
        "children_initially_isolated": True,
        "children": tuple(
            {
                "parent": parent,
                "child": child,
                "degree": graph.degree(child),
                "node_data": deepcopy(dict(graph.nodes[child])),
            }
            for parent, child in births
        ),
        "scope": "Actual 8-to-16 public birth; no same-dimensional support-reset certificate",
    }
    reference, coupling = _couple_parents(graph, parents, case=case)
    source_budget = _weighted_source_budget(coupling, parents, children)
    return graph, {
        "case": case,
        "preparation": preparation,
        "birth": birth,
        "coupling": coupling,
        "weighted_source_budget": source_budget,
    }


def run_distributed_transport_case(case):
    """Execute one finite case from the unchanged causal all-node preparation."""
    graph, prefix = prepare_distributed_transport_support(case)
    births = prefix["birth"]["parent_children"]
    after_birth = prefix["birth"]["after"]
    coupling = prefix["coupling"]
    reference, _ = _support_readout(graph)
    flow = _physical_flow(graph, reference)
    endpoint = deepcopy(_state(graph))
    responses = []
    for parent, child in births:
        rows = []
        for node in (parent, child):
            index = endpoint["nodes"].index(node)
            initial = Fraction.from_float(after_birth["epi"][index])
            final = Fraction.from_float(endpoint["epi"][index])
            rows.append(
                {
                    "node": node,
                    "epi_after_birth": initial,
                    "epi_endpoint": final,
                    "exact_epi_change": final - initial,
                    "neighbors_after_um": tuple(graph.neighbors(node)),
                    "degree_after_um": graph.degree(node),
                    "raw_pressure_after_um": coupling["after_raw"]["pressure"][index],
                    "refreshed_pressure_after_um": coupling["after_refreshed"][
                        "pressure"
                    ][index],
                    "endpoint_pressure": endpoint["pressure"][index],
                }
            )
        responses.append(
            {
                "parent": parent,
                "child": child,
                "parent_response": rows[0],
                "child_response": rows[1],
            }
        )
    return {
        **prefix,
        "physical_flow": flow,
        "endpoint": endpoint,
        "responses": tuple(responses),
        "scope": (
            "Explicit all-eligible THOL dispatch and original-parent UM policy, followed by two "
            "refreshed steps. Open executed prefix; no final SHA, autonomous policy, sustained "
            "localization, physical particle identification or empirical validation is claimed."
        ),
    }


def run_study():
    """Run the three predeclared controls with no threshold or factor changes."""
    return {"cases": tuple(run_distributed_transport_case(case) for case in CASES)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=(ROOT / "artifacts/research/thol_distributed_transport.json"),
    )
    args = parser.parse_args()
    scope = (
        "src/tnfr",
        "benchmarks/thol_distributed_transport.py",
        "benchmarks/thol_birth_transport.py",
        "benchmarks/thol_pressure_feedback.py",
        "benchmarks/thol_eligibility_dispatch.py",
        "benchmarks/capacity_localization.py",
    )
    provenance = current_git_source_provenance(ROOT, scope)
    sha, dirty, digest = provenance
    manifest = CoreExperimentManifest(
        claim_id="O1.b-distributed-THOL-coupling-refreshed-transport",
        git_sha=sha,
        source_dirty=dirty,
        dirty_source_hash=digest,
        versions={
            "python": platform.python_version(),
            "networkx": nx.__version__,
            "numpy": np.__version__,
        },
        graph_construction="Existing C8 checkerboard with actual all-node IL/OZ preparation",
        capacity_specification="Unit preparation; unchanged default THOL and UM factors",
        solver="Shared event-free pressure-refreshed Euler partition after distributed birth",
        timestep=STEPS[0],
        seed=17,
        result_status=ClaimStatus.MEASURED,
        operator_sequence=("coherence", "dissonance", "self_organization", "coupling"),
        telemetry=(
            "complete eligibility and birth inventory",
            "UM proposal and actual support",
            "raw and refreshed pressure",
            "native flow and exact support-energy budgets",
        ),
        controls=("functional links disabled", "sample retained from before birth"),
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
        raise RuntimeError("source changed during the finite study")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(_payload(report), indent=2, allow_nan=False) + "\n"
    safe_write(args.output, lambda stream: stream.write(encoded))
    print(f"Wrote distributed THOL transport observations to {args.output}")


if __name__ == "__main__":
    main()
