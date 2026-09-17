"""One declared P2 capacity/EPI comparison, with finite runtime defects.

Default VAL/IL prepares both branches. Sixteen default target UM/IL / refreshed
Euler cycles are compared with sixteen held-capacity Euler intervals. The
supplied word is an input, not an autonomous selector or a physical law.
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

from benchmarks.child_coupling_feedback import (  # noqa: E402
    _advance_forced_support_interval, _pattern,
)
from benchmarks.structural_perturbation_response import (  # noqa: E402
    _diagnostics, _event, _materialized, _reference, _word,
)
from benchmarks.thol_pressure_feedback import _payload  # noqa: E402
from tnfr.config import inject_defaults  # noqa: E402
from tnfr.constants.aliases import (  # noqa: E402
    ALIAS_DEPI, ALIAS_DNFR, ALIAS_EPI, ALIAS_SI, ALIAS_THETA, ALIAS_VF,
)
from tnfr.dynamics.dnfr import default_compute_delta_nfr  # noqa: E402
from tnfr.operators.definitions import (  # noqa: E402
    Coherence, Coupling, Expansion, Silence,
)
from tnfr.operators.factor_contracts import (  # noqa: E402
    resolve_runtime_operator_factors,
)
from tnfr.physics.capacity_feedback import (  # noqa: E402
    bound_p2_capacity_feedback, derive_p2_capacity_feedback,
    observe_p2_capacity_feedback_cycle,
)
from tnfr.physics.forced_support import (  # noqa: E402
    observe_forced_support_pattern, observe_forced_support_step,
)
from tnfr.physics.forcing_realization import capture_non_epi_forcing  # noqa: E402
from tnfr.research.claims import ClaimStatus  # noqa: E402
from tnfr.research.core_manifests import (  # noqa: E402
    CoreExperimentManifest, current_git_source_provenance,
)
from tnfr.types import Glyph  # noqa: E402

STEP = 0.25
CYCLE_COUNT = 16
CASES = ("coupled", "held_capacity")
SOURCE_SCOPE = ("src/tnfr", "benchmarks")


def prepare_p2(*, capacity=(1.0, 1.0), epi=(0.5, 0.5)):
    """Declare initial data only; subsequent state writes use canonical paths."""
    graph = nx.path_graph(2)
    inject_defaults(graph)
    graph.graph.update(RANDOM_SEED=17, _t=0.0, compute_delta_nfr=default_compute_delta_nfr)
    graph.edges[0, 1].update(weight=1.0, length=1.0)
    for node, nu, value in zip(graph, capacity, epi, strict=True):
        graph.nodes[node].update({
            ALIAS_EPI[0]: value, ALIAS_VF[0]: nu, ALIAS_THETA[0]: 0.0,
            ALIAS_DNFR[0]: 0.0, ALIAS_DEPI[0]: 0.0, ALIAS_SI[0]: 0.5,
            "glyph_history": [],
        })
    default_compute_delta_nfr(graph)
    return graph


def _family(capture, initial, *, base_capacity):
    """Check the actual finite coefficients, never infer them from a label."""
    snapshot = capture.snapshot
    checks = {
        "ordered_p2": snapshot.nodes == (0, 1),
        "support": snapshot.support_neighbors == ((1,), (0,)),
        "conductance": snapshot.conductance == initial.snapshot.conductance,
        "zero_phase": capture.phase == capture.phase_gradient == (0, 0),
        "channel_weights": capture.normalized_weights == initial.normalized_weights,
        "zero_topology_weight": dict(capture.normalized_weights)["topo"] == 0,
        "fixed_neighbor_capacity": snapshot.capacity[1] == base_capacity,
        "nonnegative_capacity_gap": snapshot.capacity[0] >= base_capacity > 0,
        "fresh_pressure": capture.stored_pressure_residual == (0, 0),
    }
    if not all(checks.values()):
        raise ValueError(f"P2 capacity-family hypotheses failed: {checks}")
    return checks


def _domain(reference, capture):
    a = capture.snapshot.capacity[0] - reference.base_capacity
    k, lower, upper = reference.forcing_ratio, reference.epi_lower, reference.epi_upper
    x0, x1 = capture.snapshot.epi
    return {
        "capacity_gap": a, "epi": (x0, x1),
        "lower_margins": (x0 - lower, x1 - lower - k * a),
        "upper_margins": (upper - k * a - x0, upper - x1),
        "inside": (0 <= a <= reference.max_capacity_gap
                   and lower <= x0 <= upper - k * a
                   and lower + k * a <= x1 <= upper),
    }


def _cycle_defects(reference, before, post_event, following):
    """Resolve one actual UM/Euler cycle into three signed error sources."""
    if before.snapshot.epi != post_event.snapshot.epi:
        raise ValueError("capacity-only UM comparison requires unchanged EPI")
    c, h = reference.base_capacity, reference.timestep
    a = before.snapshot.capacity[0] - c
    actual_gap = post_event.snapshot.capacity[0] - c
    ideal = observe_p2_capacity_feedback_cycle(reference, capacity_gap=a, epi=before.snapshot.epi)
    ideal_gap = reference.capacity_retention * a
    budget = observe_forced_support_step(
        _reference(post_event), post_event.snapshot, following.snapshot, h,
    )
    ideal_epi = ideal.epi_after
    represented_gap_epi = tuple(x + h * nu * pressure for x, nu, pressure in zip(
        post_event.snapshot.epi, post_event.snapshot.capacity,
        budget.before.modeled_pressure, strict=True,
    ))
    capacity_effect = tuple(b - a for a, b in zip(ideal_epi, represented_gap_epi))
    pressure_effect = tuple(h * nu * defect for nu, defect in zip(
        post_event.snapshot.capacity, budget.before.pressure_defect, strict=True,
    ))
    execution_effect = budget.support_budget.state_defect
    total = tuple(actual - model for actual, model in zip(following.snapshot.epi, ideal_epi))
    residual = tuple(value - dc - dp - dx for value, dc, dp, dx in zip(
        total, capacity_effect, pressure_effect, execution_effect, strict=True,
    ))
    if any(residual):
        raise RuntimeError("capacity/pressure/execution defect identity was lost")
    return {
        "conditional_model_cycle": asdict(ideal),
        "capacity_gap_defect": actual_gap - ideal_gap,
        "ideal_epi": ideal_epi, "epi_defect": total,
        "capacity_transition_epi_effect": capacity_effect,
        "pressure_realization_epi_effect": pressure_effect,
        "execution_epi_effect": execution_effect,
        "identity_residual": residual,
        "mean_budget": {
            key: getattr(budget, key) for key in (
                "mean_change", "mean_model_change", "mean_pressure_defect",
                "mean_step_defect", "mean_identity_residual",
            )
        },
        "scope": "Exact accounting at observed inputs; no uniform runtime error bound",
    }


def run_capacity_feedback_case(case):
    if case not in CASES:
        raise ValueError(f"case must be one of {CASES}")
    graph = prepare_p2()
    initial = capture_non_epi_forcing(graph)
    target = _reference(initial)
    ops = [Expansion(), Coherence()]
    if case == "coupled":
        for _ in range(CYCLE_COUNT):
            ops.extend((Coupling(), Coherence()))
    ops.append(Silence())
    word, admission = _word(ops, initialized=graph.nodes[0][ALIAS_EPI[0]] > 0)
    if word is None:
        raise RuntimeError(f"declared capacity-feedback word refused: {admission}")
    current = initial
    preparation = []
    for index in range(2):
        current, event = _event(
            graph, 0, ops[index], word.step(index), target, current, refresh=True,
        )
        preparation.append(event)
        if event["status"] != "executed":
            raise RuntimeError(f"declared preparation refused: {event['reason']}")
    prepared = current
    factors = resolve_runtime_operator_factors(graph.graph["GLYPH_FACTORS"], Glyph.UM, graph.graph)
    k = dict(current.normalized_weights)["vf"] / current.epi_weight
    gap = current.snapshot.capacity[0] - current.snapshot.capacity[1]
    # Tight sufficient box read from actual initial data; no search or fit.
    x0, x1 = current.snapshot.epi
    lower, upper = min(x0, x1 - k * gap), max(x0 + k * gap, x1)
    reference = derive_p2_capacity_feedback(
        base_capacity=current.snapshot.capacity[1], epi_weight=current.epi_weight,
        vf_weight=dict(current.normalized_weights)["vf"],
        coupling_factor=factors["UM_vf_sync"], timestep=STEP,
        max_capacity_gap=gap, epi_lower=lower, epi_upper=upper,
    )
    if (lower < graph.graph.get("UM_MIN_EPI", 0.05)
            or reference.base_capacity < graph.graph.get("UM_MIN_VF", 0.01)
            or lower < graph.graph["EPI_MIN"] or upper > graph.graph["EPI_MAX"]
            or graph.graph["CLIP_MODE"] != "hard"):
        raise RuntimeError("derived box does not fit the configured positive admission interval")
    record = {
        "case": case, "word": admission, "preparation": preparation,
        "prepared": _materialized(graph), "prepared_diagnostics": _diagnostics(graph),
        "initial_capture": asdict(initial), "original_target": asdict(target),
        "coupled_model": asdict(reference), "initial_domain": _domain(reference, current),
        "model_bound": asdict(bound_p2_capacity_feedback(
            reference, capacity_gap=gap, epi=current.snapshot.epi, cycles=CYCLE_COUNT,
        )) if case == "coupled" else None,
        "cycles": [],
    }
    _, rejected_word = _word(
        [Expansion(), Coherence()] + [Coupling() for _ in range(CYCLE_COUNT)] + [Silence()],
        initialized=True,
    )
    record["direct_repeated_um_control"] = rejected_word
    for index in range(CYCLE_COUNT):
        before = current
        family_before = _family(before, initial, base_capacity=reference.base_capacity)
        event = None
        separator = None
        if case == "coupled":
            op_index = 2 * index + 2
            current, event = _event(
                graph, 0, ops[op_index], word.step(op_index), target, before, refresh=True,
            )
            if event["status"] != "executed":
                raise RuntimeError(f"declared UM refused: {event['reason']}")
            post_um = current
            current, separator = _event(
                graph, 0, ops[op_index + 1], word.step(op_index + 1), target, current, refresh=True,
            )
            if separator["status"] != "executed":
                raise RuntimeError(f"declared IL separator refused: {separator['reason']}")
            for name in ("epi", "capacity", "conductance", "support_neighbors", "stored_pressure"):
                if getattr(post_um.snapshot, name) != getattr(current.snapshot, name):
                    raise RuntimeError(f"refreshed IL separator changed the modeled {name}")
            if post_um.phase != current.phase or post_um.forcing != current.forcing:
                raise RuntimeError("IL separator changed the modeled phase/forcing")
        after_event = current
        family_event = _family(after_event, initial, base_capacity=reference.base_capacity)
        held = _reference(current)
        following, flow = _advance_forced_support_interval(
            graph, target, held, current, current, duration=STEP,
        )
        family_after = _family(following, initial, base_capacity=reference.base_capacity)
        record["cycles"].append({
            "ordinal": index, "event": event, "separator": separator, "flow": flow,
            "family_checks": (family_before, family_event, family_after),
            "domain_before": _domain(reference, before),
            "domain_after_event": _domain(reference, after_event),
            "domain_after_flow": _domain(reference, following),
            "defects": _cycle_defects(reference, before, after_event, following)
            if case == "coupled" else None,
        })
        current = following
    record.update(
        status="measured", final_before_closure=_materialized(graph),
        final_diagnostics=_diagnostics(graph), final_capture=asdict(current),
        initial_original_pattern=asdict(_pattern(target, prepared)),
        final_original_pattern=asdict(_pattern(target, current)),
        conditional_frozen_reference=asdict(_reference(current)),
        conditional_frozen_limit=asdict(observe_forced_support_pattern(
            target, nodes=current.snapshot.nodes, epi=_reference(current).relative_profile,
        )),
    )
    _, closure = _event(
        graph, 0, ops[-1], word.step(len(ops) - 1), target, current, refresh=True,
    )
    if closure["status"] != "executed":
        raise RuntimeError(f"closure refused: {closure['reason']}")
    record["closure_after_measurement"] = closure
    record["scope"] = (
        "Finite supplied words and executor-identified flow intervals. The exact "
        "model theorem is conditional; observed defects and finite box membership "
        "do not prove binary64 infinite-time restoration, future grammar admission "
        "or an autonomous mechanism. SHA is outside the measured cycle model."
    )
    return record


def _artifact_payload(report):
    """Preserve the documented infinite SHA time without nonstandard JSON.

    Only this named operator metric admits the explicit infinity tag. Other
    nonfinite numbers still fail the strict JSON encoder; no state or defect
    is silently converted to null or discarded.
    """
    payload = _payload(report)
    for case in payload["cases"]:
        for metric in case["closure_after_measurement"]["actual_operator_metrics"]:
            if metric.get("time_to_collapse") == math.inf:
                metric["time_to_collapse"] = {"numeric_kind": "positive_infinity"}
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/research/capacity_feedback.json")
    args = parser.parse_args()
    provenance = current_git_source_provenance(ROOT, SOURCE_SCOPE)
    sha, dirty, digest = provenance
    manifest = CoreExperimentManifest(
        claim_id="O3.a-P2-coupled-capacity-EPI-finite-response",
        git_sha=sha, source_dirty=dirty, dirty_source_hash=digest,
        versions={"python": platform.python_version(), "networkx": nx.__version__,
                  "numpy": np.__version__},
        graph_construction="Prepared unit undirected P2, zero phase, uniform EPI=0.5 and capacity=1",
        capacity_specification="Actual default VAL then target-only default UM; neighbor fixed",
        solver="Shared default Euler, explicit canonical pressure refresh at every interval",
        timestep=STEP, seed=17, result_status=ClaimStatus.MEASURED,
        operator_sequence=("VAL IL", "16 target UM IL / Euler cycles or 16 held Euler intervals", "SHA"),
        telemetry=("EPI/capacity/phase and tetrad", "fixed original target", "admission and histories",
                   "capacity/pressure/execution defects", "finite invariant-box membership"),
        controls=("matched held-capacity branch",), artifacts=(str(args.output),),
    )
    manifest.validate_for_admission()
    report = {"manifest": manifest.to_dict(), "source_scope": SOURCE_SCOPE,
              "cases": [run_capacity_feedback_case(case) for case in CASES],
              "experimental_status": "No empirical correspondence tested"}
    if report["cases"][0]["prepared"] != report["cases"][1]["prepared"]:
        raise RuntimeError("independent branches did not reproduce the same preparation")
    if current_git_source_provenance(ROOT, SOURCE_SCOPE) != provenance:
        raise RuntimeError("source changed while executing the research artifact")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_artifact_payload(report), indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    print(f"Wrote finite capacity/EPI comparison to {args.output}")


if __name__ == "__main__":
    main()
