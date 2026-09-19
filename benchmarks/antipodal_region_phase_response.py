"""Five fixed phase perturbations of the prepared two-region witness.

Each executes simultaneous UM/IL, one shared nodal interval, then SHA.
No coefficient, direction or horizon is selected from the observed response.
Exact local phase instability and finite binary64 measurements remain distinct.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from fractions import Fraction
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

from benchmarks.capacity_feedback import (
    SOURCE_SCOPE,
    STEP,
    _artifact_payload,
)  # noqa: E402
from benchmarks.child_coupling_feedback import (
    _advance_forced_support_interval,
)  # noqa: E402
from benchmarks.compatible_capacity_regions import (  # noqa: E402
    _admissions,
    _captured_state,
    prepare_capacity_regions,
)
from benchmarks.structural_perturbation_response import (  # noqa: E402
    _diagnostics,
    _plain,
    _reference,
    _word,
)
from tnfr.constants.aliases import ALIAS_THETA  # noqa: E402
from tnfr.dynamics.dnfr import default_compute_delta_nfr  # noqa: E402
from tnfr.metrics.common import compute_coherence  # noqa: E402
from tnfr.operators._coherence_stage_kernel import (  # noqa: E402
    DEFAULT_PHASE_LOCKING_COEFFICIENT,
    propose_coherence_stage,
)
from tnfr.operators._coupling_stage_kernel import propose_coupling_stage  # noqa: E402
from tnfr.operators.definitions import Coherence, Coupling, Silence  # noqa: E402
from tnfr.operators.factor_contracts import (
    resolve_runtime_operator_factors,
)  # noqa: E402
from tnfr.operators.network_stage import (  # noqa: E402
    TWO_PHASE_JACOBI,
    execute_coupling_stage,
    execute_pointwise_stage,
)
from tnfr.operators.preconditions.coherence import (  # noqa: E402
    coherence_precondition_warnings,
    validate_coherence_strict,
)
from tnfr.physics.coupling_support import (  # noqa: E402
    derive_antipodal_region_phase_balance,
    observe_antipodal_region_phase_response,
    observe_coupling_support,
)
from tnfr.physics.forced_support import (  # noqa: E402
    observe_forced_support_event,
    observe_forced_support_pattern,
)
from tnfr.physics.forcing_realization import (  # noqa: E402
    capture_non_epi_forcing,
    decompose_non_epi_forcing,
)
from tnfr.research.claims import ClaimStatus  # noqa: E402
from tnfr.research.core_manifests import (  # noqa: E402
    CoreExperimentManifest,
    current_git_source_provenance,
)
from tnfr.types import Glyph  # noqa: E402
from tnfr.utils import angle_diff  # noqa: E402

EPSILONS = (0.0, 2.0**-12, -(2.0**-12), 2.0**-16, -(2.0**-16))
BASE_PHASE = (0.0, 0.0, 0.0, math.pi, math.pi, math.pi)


def prepare_phase_response(epsilon):
    if type(epsilon) is not float or epsilon not in EPSILONS:
        raise ValueError(
            f"epsilon must be one of the predeclared float controls {EPSILONS}"
        )
    graph = prepare_capacity_regions("split_phase")
    # Additional initial data, before any operator or physical flow is executed.
    for node, base in zip(graph, BASE_PHASE, strict=True):
        graph.nodes[node][ALIAS_THETA[0]] = (
            base + (epsilon if node < 3 else -epsilon)
        ) % math.tau
    default_compute_delta_nfr(graph)
    return graph


def _phase_readout(capture):
    lift = tuple(
        Fraction(angle_diff(float(value), base))
        for value, base in zip(capture.phase, BASE_PHASE, strict=True)
    )
    energy = sum(value * value for value in lift) / 2
    a = (lift[0] + lift[1] - lift[4] - lift[5]) / 4
    b = (lift[2] - lift[3]) / 2
    projected = (a, a, b, -b, -a, -a)
    return {
        "represented_lift": lift,
        "energy": energy,
        "interior": a,
        "bridge": b,
        "antisymmetric_projection_residual": tuple(
            v - p for v, p in zip(lift, projected)
        ),
        "bridge_separation": Fraction(
            abs(angle_diff(float(capture.phase[2]), float(capture.phase[3])))
        ),
        "scope": "Production wrapped binary64 differences from a fixed represented 0/pi chart",
    }


def _observed_stage(graph, operator, step, before, *, phase_readout=_phase_readout):
    targets = tuple(graph)
    state_before = _captured_state(graph)
    admissions = _admissions(graph, operator, step)
    strict_readiness = None
    if operator.glyph is Glyph.IL:
        strict_readiness = []
        for node in targets:
            validate_coherence_strict(graph, node, emit_warnings=False)
            strict_readiness.append(
                {
                    "node": node,
                    "passed": True,
                    "warnings": coherence_precondition_warnings(graph, node),
                }
            )
    factors = resolve_runtime_operator_factors(
        graph.graph["GLYPH_FACTORS"], operator.glyph, graph.graph
    )
    coherence_before = float(compute_coherence(graph))
    if operator.glyph is Glyph.UM:
        predicted = propose_coupling_stage(
            graph,
            targets,
            factors,
            resolved_seed=17,
            node_offsets=state_before["random_provenance"]["node_offsets"],
        )
        predicted_phase = tuple(update.theta_after for update in predicted.node_updates)
        predicted_pressure = tuple(
            update.dnfr_after for update in predicted.node_updates
        )
        predicted_capacity = tuple(update.vf_after for update in predicted.node_updates)
        prediction = _plain(asdict(predicted))
        executor = execute_coupling_stage
    else:
        predicted = tuple(
            propose_coherence_stage(graph, n, factors["IL_dnfr_factor"])
            for n in targets
        )
        predicted_phase = tuple(p.phase.theta_after for p in predicted)
        predicted_pressure = tuple(p.dnfr_after for p in predicted)
        predicted_capacity = tuple(float(v) for v in before.snapshot.capacity)
        prediction = _plain(predicted)
        executor = execute_pointwise_stage
    metrics_start = len(graph.graph.get("operator_metrics", ()))
    stage = executor(
        graph, operator, targets, sequence_context=step, collect_metrics=True
    )
    if (
        stage.schedule != TWO_PHASE_JACOBI
        or stage.nodes_processed != len(targets)
        or any(
            graph.nodes[n]["glyph_history"][-1] != operator.glyph.value for n in targets
        )
    ):
        raise RuntimeError("the requested simultaneous phase stage was not executed")
    raw = capture_non_epi_forcing(graph)
    raw_state = _captured_state(graph)
    coherence_raw = float(compute_coherence(graph))
    matches = {
        "phase": raw.phase == tuple(Fraction(v) for v in predicted_phase),
        "pressure": raw.snapshot.stored_pressure
        == tuple(Fraction(v) for v in predicted_pressure),
        "capacity": raw.snapshot.capacity
        == tuple(Fraction(v) for v in predicted_capacity),
        "epi_preserved": raw.snapshot.epi == before.snapshot.epi,
        "support_preserved": raw.snapshot.conductance == before.snapshot.conductance,
    }
    if not all(matches.values()):
        raise RuntimeError(
            f"prepared stage predictions or support boundary failed: {matches}"
        )
    default_compute_delta_nfr(graph)
    following = capture_non_epi_forcing(graph)
    budget = asdict(
        observe_forced_support_event(
            _reference(before),
            _reference(following),
            before.snapshot,
            following.snapshot,
        )
    )
    budget.pop("before_reference")
    budget.pop("after_reference")
    return following, {
        "targets": targets,
        "admissions": admissions,
        "stage_result": asdict(stage),
        "independent_strict_il_readiness": strict_readiness,
        "before": state_before,
        "before_capture": asdict(before),
        "raw_state": raw_state,
        "raw_capture": asdict(raw),
        "after_refresh": _captured_state(graph),
        "after_capture": asdict(following),
        "resolved_factors": factors,
        "independent_prediction": prediction,
        "prediction_matches_actual": matches,
        "prediction_scope": "Independent pure-kernel predictions, not executor-retained proposals",
        "actual_operator_metrics": _plain(
            graph.graph.get("operator_metrics", ())[metrics_start:]
        ),
        "coherence_before": coherence_before,
        "coherence_raw": coherence_raw,
        "coherence_refreshed": float(compute_coherence(graph)),
        "event_budget": budget,
        "phase_before": phase_readout(before),
        "phase_after": phase_readout(following),
        "compatible_support_after": asdict(observe_coupling_support(graph)),
    }


def run_antipodal_phase_case(epsilon):
    graph = prepare_phase_response(epsilon)
    initial_state = _captured_state(graph)
    initial = capture_non_epi_forcing(graph)
    # A shared independently prepared reference, not an invented earlier event.
    nominal = capture_non_epi_forcing(prepare_capacity_regions("split_phase"))
    target = _reference(nominal)
    factors = resolve_runtime_operator_factors(
        graph.graph["GLYPH_FACTORS"], Glyph.UM, graph.graph
    )
    model = derive_antipodal_region_phase_balance(
        coupling_phase_factor=factors["UM_theta_push"],
        coherence_phase_factor=DEFAULT_PHASE_LOCKING_COEFFICIENT,
    )
    linear = observe_antipodal_region_phase_response(
        model, interior=epsilon, bridge=epsilon
    )
    ops = (Coupling(), Coherence(), Silence())
    word, admission = _word(ops, initialized=True)
    if word is None:
        raise RuntimeError(f"declared UM/IL/SHA word refused: {admission}")
    record = {
        "epsilon": Fraction(epsilon),
        "initial": initial_state,
        "initial_capture": asdict(initial),
        "initial_diagnostics": _diagnostics(graph),
        "word": admission,
        "phase_model": asdict(model),
        "linear_response": asdict(linear),
        "nominal_reference": asdict(target),
        "comparison_scope": "Independent nominal split-profile reference, not an observed ancestor",
        "initial_compatible_support": asdict(observe_coupling_support(graph)),
    }
    after_um, um = _observed_stage(graph, ops[0], word.step(0), initial)
    after_il, il = _observed_stage(graph, ops[1], word.step(1), after_um)
    before_phase, after_phase = _phase_readout(initial), _phase_readout(after_il)
    initial_ideal = linear.embedded_before
    final_ideal = linear.embedded_after
    initial_lift, final_lift = (
        before_phase["represented_lift"],
        after_phase["represented_lift"],
    )
    held = _reference(after_il)
    following, flow = _advance_forced_support_interval(
        graph,
        target,
        held,
        after_il,
        after_il,
        duration=STEP,
    )
    record.update(
        um=um,
        il=il,
        post_il_capture=asdict(after_il),
        final_capture=asdict(following),
        phase_before=before_phase,
        phase_after=after_phase,
        preparation_lift_residual=tuple(
            a - b for a, b in zip(initial_lift, initial_ideal)
        ),
        observed_minus_linear=tuple(a - b for a, b in zip(final_lift, final_ideal)),
        observed_phase_energy_gain=(
            after_phase["energy"] / before_phase["energy"]
            if before_phase["energy"]
            else None
        ),
        gain_scope="Finite represented phase deviation from the nominal chart; no Lyapunov claim",
        forcing_components_before=decompose_non_epi_forcing(initial),
        forcing_components_after=decompose_non_epi_forcing(after_il),
        held_reference=asdict(held),
        flow=flow,
        final_before_closure=_captured_state(graph),
        final_diagnostics=_diagnostics(graph),
        final_nominal_pattern=asdict(
            observe_forced_support_pattern(
                target,
                nodes=following.snapshot.nodes,
                epi=following.snapshot.epi,
            )
        ),
    )
    targets = tuple(graph)
    closure_admissions = _admissions(graph, ops[2], word.step(2))
    metrics_start = len(graph.graph.get("operator_metrics", ()))
    closure = execute_pointwise_stage(
        graph,
        ops[2],
        targets,
        sequence_context=word.step(2),
        collect_metrics=True,
    )
    if (
        closure.schedule != TWO_PHASE_JACOBI
        or closure.nodes_processed != len(targets)
        or any(graph.nodes[n]["glyph_history"][-1] != "SHA" for n in targets)
    ):
        raise RuntimeError("declared SHA closure failed")
    record["closure_after_measurement"] = {
        "stage_result": asdict(closure),
        "admissions": closure_admissions,
        "actual_operator_metrics": _plain(
            graph.graph.get("operator_metrics", ())[metrics_start:]
        ),
        "after": _captured_state(graph),
    }
    record["scope"] = (
        "One UM/IL phase response and one nodal interval on prepared inputs. "
        "Observed-minus-linear includes nonlinear, binary64 and chart effects. "
        "Finite probes do not prove an asymptotic derivative, eventual U3 crossing, "
        "complete-runtime instability or autonomous preparation. No gate or factor was tuned."
    )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/research/antipodal_region_phase_response.json",
    )
    args = parser.parse_args()
    provenance = current_git_source_provenance(ROOT, SOURCE_SCOPE)
    sha, dirty, digest = provenance
    manifest = CoreExperimentManifest(
        claim_id="O3.a-antipodal-UM-IL-phase-response",
        git_sha=sha,
        source_dirty=dirty,
        dirty_source_hash=digest,
        versions={
            "python": platform.python_version(),
            "networkx": nx.__version__,
            "numpy": np.__version__,
        },
        graph_construction="Prepared unit triangles (0,1,2)/(3,4,5), bridge(2,3)",
        capacity_specification="(1,1,1,2,2,2), EPI=.5; phases epsilon/pi-epsilon",
        solver="Default shared Euler; four held-pressure substeps; explicit stage refresh",
        timestep=STEP,
        seed=17,
        result_status=ClaimStatus.MEASURED,
        operator_sequence=(
            "all-target UM",
            "all-target IL",
            "one Euler interval",
            "all-target SHA",
        ),
        telemetry=(
            "exact local Jacobians",
            "represented phase lifts and linear defects",
            "raw/refreshed coherence and pressure",
            "nodal endpoint and forcing budgets",
        ),
        controls=("epsilon=0", "epsilon=+/-2^-12", "epsilon=+/-2^-16"),
        artifacts=(str(args.output),),
    )
    manifest.validate_for_admission()
    report = {
        "manifest": manifest.to_dict(),
        "source_scope": SOURCE_SCOPE,
        "cases": [run_antipodal_phase_case(e) for e in EPSILONS],
        "experimental_status": "No empirical correspondence tested",
    }
    if current_git_source_provenance(ROOT, SOURCE_SCOPE) != provenance:
        raise RuntimeError("source changed while executing phase controls")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_artifact_payload(report), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote finite phase-response controls to {args.output}")


if __name__ == "__main__":
    main()
