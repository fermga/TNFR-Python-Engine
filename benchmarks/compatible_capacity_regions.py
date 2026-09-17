"""One all-target UM and nodal interval on two linked prepared regions.

The split-phase case tests preservation by the actual U3 gate. The aligned
control retains default functional links, so its support changes too. Neither
preparation is an autonomous birth or a repeated complete-runtime policy.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

from benchmarks.capacity_feedback import SOURCE_SCOPE, STEP, _artifact_payload  # noqa: E402
from benchmarks.child_coupling_feedback import _advance_forced_support_interval  # noqa: E402
from benchmarks.structural_perturbation_response import (  # noqa: E402
    _diagnostics, _materialized, _plain, _reference, _word,
)
from tnfr.config import inject_defaults  # noqa: E402
from tnfr.constants.aliases import (  # noqa: E402
    ALIAS_DEPI, ALIAS_DNFR, ALIAS_EPI, ALIAS_SI, ALIAS_THETA, ALIAS_VF,
)
from tnfr.dynamics.dnfr import default_compute_delta_nfr  # noqa: E402
from tnfr.operators._coupling_stage_kernel import propose_coupling_stage  # noqa: E402
from tnfr.operators.definitions import Coupling, Silence  # noqa: E402
from tnfr.operators.factor_contracts import resolve_runtime_operator_factors  # noqa: E402
from tnfr.operators.grammar_dynamics import validate_candidate  # noqa: E402
from tnfr.operators.network_stage import (  # noqa: E402
    TWO_PHASE_JACOBI, execute_coupling_stage, execute_pointwise_stage,
)
from tnfr.physics.coupling_support import observe_coupling_support  # noqa: E402
from tnfr.physics.forced_support import (  # noqa: E402
    observe_forced_support_event, observe_forced_support_pattern,
)
from tnfr.physics.forcing_realization import capture_non_epi_forcing  # noqa: E402
from tnfr.physics.support_transport import _energy  # noqa: E402
from tnfr.research.claims import ClaimStatus  # noqa: E402
from tnfr.research.core_manifests import (  # noqa: E402
    CoreExperimentManifest, current_git_source_provenance,
)
from tnfr.types import Glyph  # noqa: E402

CASES = ("split_phase", "aligned_phase")
EDGES = ((0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5), (2, 3))


def prepare_capacity_regions(case):
    """Declare the contrast as initial data; all later writes use canonical paths."""
    if case not in CASES:
        raise ValueError(f"case must be one of {CASES}")
    graph = nx.Graph()
    graph.add_nodes_from(range(6))
    graph.add_edges_from(EDGES, weight=1.0, length=1.0)
    inject_defaults(graph)
    graph.graph.update(RANDOM_SEED=17, _t=0.0, compute_delta_nfr=default_compute_delta_nfr)
    for node in graph:
        graph.nodes[node].update({
            ALIAS_EPI[0]: 0.5, ALIAS_VF[0]: 1.0 if node < 3 else 2.0,
            ALIAS_THETA[0]: math.pi if case == "split_phase" and node >= 3 else 0.0,
            ALIAS_DNFR[0]: 0.0, ALIAS_DEPI[0]: 0.0, ALIAS_SI[0]: 0.5,
            "glyph_history": [],
        })
    default_compute_delta_nfr(graph)
    return graph


def _captured_state(graph):
    result = _materialized(graph)
    result["random_provenance"]["stream_scope"] = (
        "Prepared six-node graph, seed 17; all candidates under the default "
        "zero candidate-count setting; deterministic per-node UM streams"
    )
    return result


def _admissions(graph, operator, step):
    result = tuple(asdict(validate_candidate(
        graph, node, operator.glyph, sequence_context=step,
    )) for node in graph)
    if not all(record["allowed"] for record in result):
        raise RuntimeError(f"declared all-target {operator.name} refused: {result}")
    return _plain(result)


def _fixed_profile(capture, reference):
    """Identify an existing held capacity-forcing theorem, not a fitted source."""
    n = len(capture.snapshot.nodes)
    weights = dict(capture.normalized_weights)
    checks = {
        "zero_phase_pressure": capture.phase_gradient == (0,) * n,
        "zero_topology_weight": weights["topo"] == 0,
        "unit_conductance": all(w == 1 for _, _, w in capture.snapshot.conductance),
        "full_support_matches_pressure": all(
            set(row) == {j for a, j, _ in capture.snapshot.conductance if a == i}
            for i, row in enumerate(capture.snapshot.support_neighbors)
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"held capacity-profile hypotheses failed: {checks}")
    k = weights["vf"] / capture.epi_weight
    nu, x, metric = capture.snapshot.capacity, capture.snapshot.epi, reference.metric_weights
    mass = sum(metric)
    mean_capacity = sum(h * v for h, v in zip(metric, nu)) / mass
    mean_epi = sum(h * value for h, value in zip(metric, x)) / mass
    derived_profile = tuple(-k * (v - mean_capacity) for v in nu)
    if derived_profile != reference.relative_profile or reference.mean_drift:
        raise RuntimeError("capacity-profile identity failed")
    lifted = tuple(value + k * v for value, v in zip(x, nu))
    return {
        "checks": checks, "forcing_ratio": k, "capacity_mean": mean_capacity,
        "relative_profile": derived_profile,
        "conditional_exact_limit": tuple(mean_epi + z for z in derived_profile),
        "lifted_initial": lifted,
        "conditional_convex_epi_lower": tuple(min(lifted) - k * v for v in nu),
        "conditional_convex_epi_upper": tuple(max(lifted) - k * v for v in nu),
        "max_refreshed_convex_step": reference.max_convex_step,
        "scope": (
            "Exact held-coefficient diffusion limit and refreshed-Euler convex box. "
            "The actual finite interval has held pressure and signed endpoint defects; "
            "this is not observed convergence or repeated full-word admission."
        ),
    }


def run_compatible_capacity_case(case):
    graph = prepare_capacity_regions(case)
    initial_state = _captured_state(graph)
    initial = capture_non_epi_forcing(graph)
    initial_diagnostics = _diagnostics(graph)
    original = _reference(initial)
    before_support = observe_coupling_support(graph)
    ops = (Coupling(), Silence())
    word, admission = _word(ops, initialized=True)
    if word is None:
        raise RuntimeError(f"declared UM/SHA word refused: {admission}")
    targets = tuple(graph)
    admissions = _admissions(graph, ops[0], word.step(0))
    factors = resolve_runtime_operator_factors(graph.graph["GLYPH_FACTORS"], Glyph.UM, graph.graph)
    prediction = propose_coupling_stage(
        graph, targets, factors, resolved_seed=17,
        node_offsets=initial_state["random_provenance"]["node_offsets"],
    )
    metrics_start = len(graph.graph.get("operator_metrics", ()))
    stage = execute_coupling_stage(
        graph, ops[0], targets, sequence_context=word.step(0), collect_metrics=True,
    )
    if stage.schedule != TWO_PHASE_JACOBI or stage.nodes_processed != len(targets):
        raise RuntimeError("the declared simultaneous UM was not executed")
    raw_state = _captured_state(graph)
    raw = capture_non_epi_forcing(graph)
    predicted_fields = all(
        (update.theta_after is None or graph.nodes[update.node][ALIAS_THETA[0]] == update.theta_after)
        and (update.vf_after is None or graph.nodes[update.node][ALIAS_VF[0]] == update.vf_after)
        and (update.dnfr_after is None or graph.nodes[update.node][ALIAS_DNFR[0]] == update.dnfr_after)
        for update in prediction.node_updates
    )
    new_edges = tuple((u, v, dict(data)) for u, v, data in graph.edges(data=True)
                      if (u, v) not in EDGES and (v, u) not in EDGES)
    predicted_edges = {(min(e.left, e.right), max(e.left, e.right), e.weight)
                       for e in prediction.edges}
    actual_edges = {(min(u, v), max(u, v), data["weight"]) for u, v, data in new_edges}
    prediction_matches = {
        "all_proposed_fields": predicted_fields, "new_edges": actual_edges == predicted_edges,
        "epi_preserved": initial.snapshot.epi == raw.snapshot.epi,
        "all_requested_histories": all(graph.nodes[n]["glyph_history"][-1] == "UM" for n in targets),
    }
    if not all(prediction_matches.values()):
        raise RuntimeError("independent proposal prediction differs from actual UM writes")
    operator_metrics = _plain(graph.graph.get("operator_metrics", ())[metrics_start:])
    default_compute_delta_nfr(graph)
    current = capture_non_epi_forcing(graph)
    post_um_state = _captured_state(graph)
    held = _reference(current)
    after_support = observe_coupling_support(graph)
    event_budget = asdict(observe_forced_support_event(
        original, held, initial.snapshot, current.snapshot,
    ))
    event_budget.pop("before_reference")
    event_budget.pop("after_reference")
    prediction_record = _plain(asdict(prediction))
    event = {
        "targets": targets, "admissions": admissions, "stage_result": asdict(stage),
        "resolved_factors": factors, "prediction": prediction_record,
        "prediction_scope": "Independent pure-kernel prediction, not executor-retained proposals",
        "prediction_matches_actual": prediction_matches, "new_edges": new_edges,
        "actual_operator_metrics": operator_metrics, "raw_state": raw_state,
        "raw_capture": asdict(raw), "after_refresh": post_um_state, "budget": event_budget,
        "capacity_exact_model_defect": tuple(
            a - b for a, b in zip(current.snapshot.capacity, before_support.balance.capacity_after)
        ),
    }
    profile = _fixed_profile(current, held)
    following, flow = _advance_forced_support_interval(
        graph, original, held, current, current, duration=STEP,
    )
    record = {
        "case": case, "word": admission, "initial": initial_state,
        "initial_diagnostics": initial_diagnostics,
        "initial_capture": asdict(initial), "post_um_capture": asdict(current),
        "final_capture": asdict(following), "before_support": asdict(before_support),
        "after_support": asdict(after_support), "original_reference": asdict(original),
        "post_um_reference": asdict(held), "event": event, "flow": flow,
        "fixed_profile_prediction": profile,
        "full_support_capacity_energy_before": _energy(
            initial.snapshot.conductance, initial.snapshot.capacity,
        ),
        "final_before_closure": _captured_state(graph), "final_diagnostics": _diagnostics(graph),
        "initial_fixed_target_error": asdict(observe_forced_support_pattern(
            original, nodes=initial.snapshot.nodes, epi=initial.snapshot.epi,
        )),
        "final_fixed_target_error": asdict(observe_forced_support_pattern(
            original, nodes=following.snapshot.nodes, epi=following.snapshot.epi,
        )),
    }
    closure_admissions = _admissions(graph, ops[1], word.step(1))
    metrics_start = len(graph.graph.get("operator_metrics", ()))
    closure = execute_pointwise_stage(
        graph, ops[1], targets, sequence_context=word.step(1), collect_metrics=True,
    )
    if (closure.schedule != TWO_PHASE_JACOBI or closure.nodes_processed != len(targets)
            or any(graph.nodes[n]["glyph_history"][-1] != "SHA" for n in targets)):
        raise RuntimeError("declared terminal SHA did not execute on every node")
    record["closure_after_measurement"] = {
        "stage_result": asdict(closure), "admissions": closure_admissions,
        "actual_operator_metrics": _plain(graph.graph.get("operator_metrics", ())[metrics_start:]),
        "after": _captured_state(graph),
    }
    record["scope"] = (
        "Prepared contrast; one default simultaneous UM and one nodal interval. "
        "Split phases separate the capacity-mixing support while pressure support "
        "stays connected. Aligned phases also admit new links, so the control is "
        "not a fixed-support ablation. Phase preparation, grammar repetition and "
        "autonomous region formation remain open. SHA is outside maintenance measurements."
    )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/research/compatible_capacity_regions.json",
    )
    args = parser.parse_args()
    provenance = current_git_source_provenance(ROOT, SOURCE_SCOPE)
    sha, dirty, digest = provenance
    manifest = CoreExperimentManifest(
        claim_id="O3.a-U3-compatible-capacity-regions", git_sha=sha,
        source_dirty=dirty, dirty_source_hash=digest,
        versions={"python": platform.python_version(), "networkx": nx.__version__,
                  "numpy": np.__version__},
        graph_construction="Two unit triangles joined by edge (2,3); six prepared nodes",
        capacity_specification="(1,1,1,2,2,2), EPI=0.5, phases split 0/pi or all zero",
        solver="Default shared Euler: four held-input substeps; explicit pressure refresh",
        timestep=STEP, seed=17, result_status=ClaimStatus.MEASURED,
        operator_sequence=("all-target UM", "one held Euler interval", "all-target SHA"),
        telemetry=("U3 and pressure supports", "exact capacity balance", "actual phase/capacity/links",
                   "fixed profile and event/flow defects", "tetrad/coherence"),
        controls=("prepared split phases", "aligned phases with default functional links"),
        artifacts=(str(args.output),),
    )
    manifest.validate_for_admission()
    report = {"manifest": manifest.to_dict(), "source_scope": SOURCE_SCOPE,
              "cases": [run_compatible_capacity_case(case) for case in CASES],
              "experimental_status": "No empirical correspondence tested"}
    if current_git_source_provenance(ROOT, SOURCE_SCOPE) != provenance:
        raise RuntimeError("source changed while executing the region artifact")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_artifact_payload(report), indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    print(f"Wrote finite compatible-support comparison to {args.output}")


if __name__ == "__main__":
    main()
