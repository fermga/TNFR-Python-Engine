"""Five declared C6 winding controls under default simultaneous UM/IL.

One null preparation and signed k1/k3 perturbations each execute one word
and one nodal interval. Exact local phase maps, finite winding readouts and
binary64 pressure/flow residuals remain separate evidence.
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

from benchmarks.antipodal_region_phase_response import _observed_stage  # noqa: E402
from benchmarks.capacity_feedback import SOURCE_SCOPE, STEP, _artifact_payload  # noqa: E402
from benchmarks.child_coupling_feedback import _advance_forced_support_interval  # noqa: E402
from benchmarks.compatible_capacity_regions import _admissions, _captured_state  # noqa: E402
from benchmarks.structural_perturbation_response import (  # noqa: E402
    _diagnostics, _plain, _reference, _word,
)
from tnfr.config import inject_defaults  # noqa: E402
from tnfr.constants.aliases import (  # noqa: E402
    ALIAS_DEPI, ALIAS_DNFR, ALIAS_EPI, ALIAS_SI, ALIAS_THETA, ALIAS_VF,
)
from tnfr.dynamics.dnfr import default_compute_delta_nfr  # noqa: E402
from tnfr.operators._coherence_stage_kernel import DEFAULT_PHASE_LOCKING_COEFFICIENT  # noqa: E402
from tnfr.operators.definitions import Coherence, Coupling, Silence  # noqa: E402
from tnfr.operators.factor_contracts import resolve_runtime_operator_factors  # noqa: E402
from tnfr.operators.network_stage import TWO_PHASE_JACOBI, execute_pointwise_stage  # noqa: E402
from tnfr.physics.coupling_support import observe_coupling_support  # noqa: E402
from tnfr.physics.coupling_winding import (  # noqa: E402
    derive_c6_winding_phase_response, observe_c6_winding_phase_response,
)
from tnfr.physics.forcing_realization import (  # noqa: E402
    capture_non_epi_forcing, decompose_non_epi_forcing,
)
from tnfr.physics.winding_certificates import certify_phase_winding  # noqa: E402
from tnfr.research.claims import ClaimStatus  # noqa: E402
from tnfr.research.core_manifests import (  # noqa: E402
    CoreExperimentManifest, current_git_source_provenance,
)
from tnfr.types import Glyph  # noqa: E402
from tnfr.utils import angle_diff  # noqa: E402

BASE_PHASE = tuple(i * math.pi / 3 for i in range(6))
AMPLITUDE = 2.0**-12
DIRECTIONS = {
    "null": (0, 0, 0, 0, 0, 0),
    "k1": (1, Fraction(1, 2), Fraction(-1, 2), -1, Fraction(-1, 2), Fraction(1, 2)),
    "k3": (1, -1, 1, -1, 1, -1),
}
CASES = (("null", 0.0), ("k1", AMPLITUDE), ("k1", -AMPLITUDE),
         ("k3", AMPLITUDE), ("k3", -AMPLITUDE))


def prepare_c6_phase_response(mode, epsilon):
    if type(epsilon) is not float or (mode, epsilon) not in CASES:
        raise ValueError("mode and epsilon must identify one of the five declared C6 controls")
    graph = nx.cycle_graph(6)
    inject_defaults(graph)
    graph.graph.update(RANDOM_SEED=17, _t=0.0, compute_delta_nfr=default_compute_delta_nfr)
    for node in graph:
        graph.nodes[node].update({
            ALIAS_EPI[0]: 0.5, ALIAS_VF[0]: 1.0,
            ALIAS_THETA[0]: (BASE_PHASE[node] + epsilon * float(DIRECTIONS[mode][node])) % math.tau,
            ALIAS_DNFR[0]: 0.0, ALIAS_DEPI[0]: 0.0, ALIAS_SI[0]: 0.5,
            "glyph_history": [],
        })
    for edge in graph.edges:
        graph.edges[edge].update(weight=1.0, length=1.0)
    default_compute_delta_nfr(graph)
    return graph


def _phase_readout(capture):
    lift = tuple(Fraction(angle_diff(float(value), base))
                 for value, base in zip(capture.phase, BASE_PHASE, strict=True))
    mean = sum(lift) / 6
    centered = tuple(value - mean for value in lift)
    return {
        "represented_lift": lift, "mean_shift": mean, "centered_lift": centered,
        "centered_energy": sum(value * value for value in centered) / 2,
        "scope": "Wrapped represented perturbation around one stored winding base; rotation separated",
    }


def _winding_state(graph):
    support = observe_coupling_support(graph)
    gate = float(support.effective_phase_limit)
    certificate = certify_phase_winding(graph, tuple(range(6)), phase_gate=gate)
    margins = tuple(abs(angle_diff(graph.nodes[i][ALIAS_THETA[0]], graph.nodes[j][ALIAS_THETA[0]]))
                    - gate for i in range(6) for j in range(i + 1, 6) if not graph.has_edge(i, j))
    if (not certificate.is_defined or certificate.winding != 1 or not certificate.u3_admissible
            or len(margins) != 9 or min(margins) <= 0
            or support.blocked_targets or support.excluded_edges):
        raise RuntimeError("declared C6 winding or compatible support boundary failed")
    return {"certificate": asdict(certificate), "nonedge_exclusion_margins": margins,
            "minimum_nonedge_exclusion_margin": min(margins), "support": asdict(support)}


def run_c6_phase_case(mode, epsilon):
    graph = prepare_c6_phase_response(mode, epsilon)
    initial = capture_non_epi_forcing(graph)
    factors = resolve_runtime_operator_factors(graph.graph["GLYPH_FACTORS"], Glyph.UM, graph.graph)
    model = derive_c6_winding_phase_response(
        coupling_phase_factor=factors["UM_theta_push"],
        coherence_phase_factor=DEFAULT_PHASE_LOCKING_COEFFICIENT,
    )
    direction = tuple(Fraction(epsilon) * value for value in DIRECTIONS[mode])
    tangent = observe_c6_winding_phase_response(model, direction=direction)
    ops = (Coupling(), Coherence(), Silence())
    word, admission = _word(ops, initialized=True)
    if word is None:
        raise RuntimeError(f"declared default UM/IL/SHA word refused: {admission}")
    record = {
        "mode": mode, "epsilon": Fraction(epsilon), "declared_tangent": direction,
        "initial": _captured_state(graph), "initial_capture": asdict(initial),
        "initial_diagnostics": _diagnostics(graph), "initial_winding": _winding_state(graph),
        "word": admission, "phase_model": asdict(model), "tangent_observation": asdict(tangent),
    }
    after_um, um = _observed_stage(graph, ops[0], word.step(0), initial, phase_readout=_phase_readout)
    um_winding = _winding_state(graph)
    after_il, il = _observed_stage(graph, ops[1], word.step(1), after_um, phase_readout=_phase_readout)
    il_winding = _winding_state(graph)
    before_phase, after_phase = _phase_readout(initial), _phase_readout(after_il)
    post_lift = after_phase["represented_lift"]
    realized_scaled_pressure = tuple(Fraction(math.pi) * value for value in after_il.phase_gradient)
    # The exact C6 midpoint identity is pi*g=(Adj/2-I)e; test the actual
    # endpoint too, separately from the linearized prediction at preparation.
    midpoint_scaled_pressure = tuple(
        (post_lift[(i - 1) % 6] + post_lift[(i + 1) % 6]) / 2 - post_lift[i]
        for i in range(6)
    )
    reference = _reference(after_il)
    following, flow = _advance_forced_support_interval(
        graph, _reference(initial), reference, after_il, after_il, duration=STEP,
    )
    record.update(
        um=um, il=il, um_winding=um_winding, il_winding=il_winding,
        post_il_capture=asdict(after_il), final_capture=asdict(following),
        phase_before=before_phase, phase_after=after_phase,
        preparation_lift_residual=tuple(value - ideal for value, ideal in zip(
            before_phase["represented_lift"], direction,
        )),
        observed_minus_linear=tuple(value - ideal for value, ideal in zip(post_lift, tangent.after_coherence)),
        observed_centered_energy_gain=(after_phase["centered_energy"] / before_phase["centered_energy"]
                                       if before_phase["centered_energy"] else None),
        realized_phase_pressure_pi_scaled=realized_scaled_pressure,
        endpoint_midpoint_pressure_pi_scaled=midpoint_scaled_pressure,
        midpoint_pressure_residual=tuple(
            a - b for a, b in zip(realized_scaled_pressure, midpoint_scaled_pressure)
        ),
        linear_pressure_residual=tuple(
            a - b for a, b in zip(realized_scaled_pressure, tangent.phase_pressure_pi_scaled)
        ),
        forcing_components=decompose_non_epi_forcing(after_il),
        held_reference=asdict(reference), flow=flow,
        final_before_closure=_captured_state(graph), final_diagnostics=_diagnostics(graph),
        final_winding=_winding_state(graph),
    )
    targets = tuple(graph)
    closure_admissions = _admissions(graph, ops[2], word.step(2))
    metrics_start = len(graph.graph.get("operator_metrics", ()))
    closure = execute_pointwise_stage(
        graph, ops[2], targets, sequence_context=word.step(2), collect_metrics=True,
    )
    if (closure.schedule != TWO_PHASE_JACOBI or closure.nodes_processed != 6
            or any(graph.nodes[n]["glyph_history"][-1] != "SHA" for n in targets)):
        raise RuntimeError("declared C6 SHA closure failed")
    record["closure_after_measurement"] = {
        "stage_result": asdict(closure), "admissions": closure_admissions,
        "actual_operator_metrics": _plain(graph.graph.get("operator_metrics", ())[metrics_start:]),
        "after": _captured_state(graph),
    }
    record["scope"] = (
        "Prepared winding; one default all-target UM/IL word and one nodal interval. "
        "Centered phase energy excludes neutral rotation. Wrapped-lift and pressure "
        "residuals retain finite arithmetic and nonlinear effects. Winding is read "
        "from the actual oriented cycle, not inferred from rational tangent sums. "
        "No autonomous formation, binary64 asymptotic convergence or future complete-word admission."
    )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/research/c6_winding_phase_response.json",
    )
    args = parser.parse_args()
    provenance = current_git_source_provenance(ROOT, SOURCE_SCOPE)
    sha, dirty, digest = provenance
    manifest = CoreExperimentManifest(
        claim_id="O3.a-C6-default-winding-phase-response", git_sha=sha,
        source_dirty=dirty, dirty_source_hash=digest,
        versions={"python": platform.python_version(), "networkx": nx.__version__,
                  "numpy": np.__version__},
        graph_construction="Unit C6, ordered0..5, prepared winding1 with stored base i*pi/3",
        capacity_specification="Uniform nu=1,EPI=.5; null or signed k1/k3 phase perturbation2^-12",
        solver="Default shared Euler with four held-pressure substeps; explicit stage refresh",
        timestep=STEP, seed=17, result_status=ClaimStatus.MEASURED,
        operator_sequence=("all-target UM", "all-target IL", "one Euler interval", "all-target SHA"),
        telemetry=("centered phase response", "actual winding and gate margins", "phase pressure",
                   "stage provenance", "signed nodal-flow budgets", "tetrad/coherence"),
        controls=("null", "+/- k1 at2^-12", "+/- k3 at2^-12"),
        artifacts=(str(args.output),),
    )
    manifest.validate_for_admission()
    report = {"manifest": manifest.to_dict(), "source_scope": SOURCE_SCOPE,
              "cases": [run_c6_phase_case(mode, epsilon) for mode, epsilon in CASES],
              "experimental_status": "No empirical correspondence tested"}
    if current_git_source_provenance(ROOT, SOURCE_SCOPE) != provenance:
        raise RuntimeError("source changed while executing C6 controls")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_artifact_payload(report), indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    print(f"Wrote finite C6 winding controls to {args.output}")


if __name__ == "__main__":
    main()
