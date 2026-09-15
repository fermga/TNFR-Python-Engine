"""Finite canonical Coupling winding observations and a Transition control.

Run ``python benchmarks/canonical_winding_persistence.py`` from the checkout.
All continuous-time, physical-particle, and future binary64 stability claims
remain outside this artifact. Exact gap identities and recorded endpoints are
reported separately.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from fractions import Fraction
import json
import math
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

from tnfr.alias import get_attr  # noqa: E402
from tnfr.constants.aliases import (  # noqa: E402
    ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF,
)
from tnfr.constants.canonical import UM_THETA_PUSH  # noqa: E402
from tnfr.dynamics.dnfr import default_compute_delta_nfr  # noqa: E402
from tnfr.operators.definitions import Transition  # noqa: E402
from tnfr.operators.word_execution import run_network_sequence  # noqa: E402
from tnfr.physics.coupling_winding import observe_coupling_gap_step  # noqa: E402
from tnfr.physics.emergent_particles import winding_ring  # noqa: E402
from tnfr.physics.fields import (  # noqa: E402
    compute_phase_curvature, compute_phase_gradient,
    compute_structural_potential, estimate_coherence_length_with_provenance,
)
from tnfr.physics.winding_certificates import (  # noqa: E402
    certify_phase_winding, observe_winding_word,
)
from tnfr.research.claims import ClaimStatus  # noqa: E402
from tnfr.research.core_manifests import (  # noqa: E402
    CoreExperimentManifest, current_git_source_provenance,
)
from tnfr.utils.numeric import angle_diff  # noqa: E402


def build_ring(count, winding, *, perturb=True, history_window=64):
    """Prepare one declared phase state and the restricted existing policy."""
    graph = winding_ring(count, winding)
    graph.graph.update(
        RANDOM_SEED=17,
        GLYPH_HYSTERESIS_WINDOW=history_window,
        UM_BIDIRECTIONAL=False,
        UM_FUNCTIONAL_LINKS=False,
    )
    for node, data in graph.nodes(data=True):
        perturbation = math.pi / 16 * math.sin(math.tau * node / count)
        phase = (data["theta"] + (perturbation if perturb else 0.0)) % math.tau
        data.update(theta=phase, phase=phase, glyph_history=[])
    for edge in graph.edges:
        graph.edges[edge].update(weight=1.0, length=1.0)
    return graph


def _values(graph, aliases):
    return tuple(float(get_attr(graph.nodes[node], aliases, None)) for node in graph)


def phase_gaps(phases):
    return tuple(
        float(angle_diff(phases[(i + 1) % len(phases)], phase))
        for i, phase in enumerate(phases)
    )


def _spread(gaps):
    values = tuple(Fraction.from_float(value) for value in gaps)
    mean = sum(values, Fraction(0)) / len(values)
    return sum(((value - mean) ** 2 for value in values), Fraction(0)) / 2


def _concentration(gaps, winding):
    # Positive oriented circulation fractions are an extendedness diagnostic,
    # not physical probabilities. Signed zero-winding gaps have no such use.
    if winding != 1 or any(value <= 0.0 for value in gaps):
        return None
    values = tuple(Fraction.from_float(value) for value in gaps)
    total = sum(values, Fraction(0))
    return sum(((value / total) ** 2 for value in values), Fraction(0))


def _tetrad(graph):
    length = estimate_coherence_length_with_provenance(graph)
    return {
        "structural_potential": tuple(compute_structural_potential(graph).values()),
        "phase_gradient": tuple(compute_phase_gradient(graph).values()),
        "phase_curvature": tuple(compute_phase_curvature(graph).values()),
        "coherence_length": asdict(length),
    }


def run_coupling_case(count, winding, *, coupling_steps=8):
    """Execute a finite full-validator-admitted word of UM/SHA cycles."""
    if isinstance(coupling_steps, bool) or not isinstance(coupling_steps, int):
        raise TypeError("coupling_steps must be a positive integer")
    if coupling_steps <= 0:
        raise ValueError("coupling_steps must be a positive integer")
    graph = build_ring(count, winding, history_window=2 * coupling_steps + 8)
    # The standard nodal pressure reader is refreshed after each actual stage.
    graph.graph["compute_delta_nfr"] = default_compute_delta_nfr
    default_compute_delta_nfr(graph)
    initial = certify_phase_winding(graph, range(count))
    initial_phases = _values(graph, ALIAS_THETA)
    initial_gaps = phase_gaps(initial_phases)
    initial_tetrad = _tetrad(graph)
    initial_epi = _values(graph, ALIAS_EPI)
    initial_support = tuple(graph.edges)
    previous_phases = initial_phases
    observations = []

    def on_step(operator):
        nonlocal previous_phases
        phases = _values(graph, ALIAS_THETA)
        gaps = phase_gaps(phases)
        certificate = certify_phase_winding(graph, range(count))
        before_gaps = phase_gaps(previous_phases)
        row = {
            "operator": operator,
            "certificate": asdict(certificate),
            "phases": phases,
            "gaps": gaps,
            "wrapped_endpoint_increments": tuple(
                angle_diff(after, before)
                for after, before in zip(phases, previous_phases, strict=True)
            ),
            "phase_endpoint_unchanged": phases == previous_phases,
            "epi": _values(graph, ALIAS_EPI),
            "capacity": _values(graph, ALIAS_VF),
            "pressure": _values(graph, ALIAS_DNFR),
            "exact_observed_gap_spread": _spread(gaps),
            "positive_circulation_concentration": _concentration(
                gaps, certificate.winding
            ),
            "edge_support": tuple(graph.edges),
            "actual_last_glyphs": tuple(
                tuple(graph.nodes[node]["glyph_history"])[-1] for node in graph
            ),
        }
        if operator == "coupling":
            exact = observe_coupling_gap_step(before_gaps)
            residual = tuple(
                Fraction.from_float(actual) - ideal
                for actual, ideal in zip(gaps, exact.output_gaps, strict=True)
            )
            row.update(
                exact_gap_model=asdict(exact),
                exact_runtime_gap_residual=residual,
                max_runtime_gap_residual=max(map(abs, residual)),
            )
        observations.append(row)
        previous_phases = phases

    names = ["coupling", "silence"] * coupling_steps
    # Admission derives from the actual nonzero EPI preparation.
    initialized = all(value != 0.0 for value in initial_epi)
    run_network_sequence(
        graph, names, context={"initial_epi_nonzero": initialized}, on_step=on_step
    )
    return {
        "nodes": count,
        "prepared_winding": winding,
        "coupling_steps": coupling_steps,
        "initial": asdict(initial),
        "initial_phases": initial_phases,
        "initial_epi": initial_epi,
        "initial_edge_support": initial_support,
        "initial_gap_spread": _spread(initial_gaps),
        "initial_positive_circulation_concentration": _concentration(
            initial_gaps, initial.winding
        ),
        "initial_tetrad": initial_tetrad,
        "final_tetrad": _tetrad(graph),
        "requested_word": tuple(names),
        "actual_histories": tuple(
            tuple(graph.nodes[node]["glyph_history"]) for node in graph
        ),
        "observations": observations,
        "endpoint_winding_preserved": all(
            row["certificate"]["winding"] == initial.winding
            for row in observations
        ),
        "observed_endpoint_lifetime_stages": len(observations),
        "policy": {
            "UM_BIDIRECTIONAL": False,
            "UM_FUNCTIONAL_LINKS": False,
            "UM_theta_push": UM_THETA_PUSH,
            "Silence_capacity": "existing default factor; recorded attenuation",
            "pressure_refresh": "default_compute_delta_nfr after every stage",
            "eta_status": "existing configured default; not a fitted constant",
        },
        "scope": (
            "Finite binary64 endpoints of admitted all-target UM/SHA stages; "
            "SHA leaves phase fixed and attenuates capacity by its default. "
            "Gap-model residuals are observations, not uniform runtime bounds. "
            "No future invocation admission, observed continuous interpolation "
            "or physical-particle claim."
        ),
    }


def run_transition_counterexample(*, steps=14):
    """Retain default canonical NAV writes and observe their finite endpoints."""
    graph = build_ring(8, 1, perturb=False, history_window=steps + 8)
    # This direct operator word retains stored pressure. It is not a solver
    # trajectory or the refreshed-pressure UM policy used above.
    initial = certify_phase_winding(graph, range(8))
    observations = []
    for index in range(steps):
        phases_before = _values(graph, ALIAS_THETA)
        result = observe_winding_word(graph, range(8), 0, [Transition()])
        step = result.steps[0]
        observations.append({
            "invocation": index + 1,
            "certificate": asdict(step.certificate),
            "theta_before": phases_before[0],
            "theta_after": _values(graph, ALIAS_THETA)[0],
            "phase_changes": step.phase_changes,
            "actual_history": result.actual_history,
            "epi": _values(graph, ALIAS_EPI),
            "capacity": _values(graph, ALIAS_VF),
            "pressure": _values(graph, ALIAS_DNFR),
        })
    first_loss = next(
        (row["invocation"] for row in observations
         if row["certificate"]["winding"] != initial.winding), None
    )
    return {
        "initial": asdict(initial),
        "observations": observations,
        "first_observed_winding_loss": first_loss,
        "scope": (
            "Repeated default NAV events at node zero; all auxiliary writes "
            "retained. An endpoint winding change excludes branch-free "
            "continuous completion but supplies no actual between-event path."
        ),
    }


def _payload(value):
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _payload(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_payload(item) for item in value]
    return value


def winding_domain_controls():
    """Read deliberately invalid preparations without claiming operator paths."""
    branch = build_ring(8, 1, perturb=False)
    branch.nodes[1]["theta"] = math.pi
    absent = build_ring(8, 1, perturb=False)
    absent.remove_edge(3, 4)
    return {
        "branch_boundary": asdict(certify_phase_winding(branch, range(8))),
        "missing_cycle_edge": asdict(certify_phase_winding(absent, range(8))),
        "scope": "Prepared diagnostic controls; no canonical event removed an edge",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "artifacts/research/canonical_winding_persistence.json",
    )
    args = parser.parse_args()
    source_scope = ("src/tnfr", "benchmarks/canonical_winding_persistence.py")
    sha, dirty, digest = current_git_source_provenance(ROOT, source_scope)
    manifest = CoreExperimentManifest(
        claim_id="O3.a-configured-coupling-winding-endpoints",
        git_sha=sha, source_dirty=dirty, dirty_source_hash=digest,
        versions={
            "python": platform.python_version(),
            "networkx": nx.__version__, "numpy": np.__version__,
        },
        graph_construction=(
            "Unit C8/C16, phase=2*pi*W*i/n+(pi/16)*sin(2*pi*i/n), W=0,1"
        ),
        capacity_specification=(
            "Initially one; recorded default SHA attenuation after each UM"
        ),
        solver="Canonical discrete operator stages; no physical timestep",
        result_status=ClaimStatus.MEASURED, seed=17,
        operator_sequence=("(UM SHA)^8", "NAV at node0,14 calls"),
        telemetry=(
            "winding/branch/U3 margins", "exact companion gap residual",
            "nodal triad and pressure", "initial/final tetrad",
            "positive circulation concentration", "actual per-node history",
        ),
        controls=(
            "W=0", "C8/C16", "default NAV changes W=1 to W=0",
            "branch-boundary and absent-cycle tests",
        ),
        artifacts=(str(args.output),),
    )
    manifest.validate_for_admission()
    exact_manifest = replace(
        manifest,
        claim_id="O3.a-exact-target-only-coupling-gap-companion",
        solver="Detached exact rational cycle-gap map; no runtime binding",
        result_status=ClaimStatus.DERIVED,
        operator_sequence=(),
        capacity_specification=(
            "Gap-map identities use the UM factor; no physical clock inferred"
        ),
        telemetry=("conserved gap sum", "invariant interval", "exact spread drop"),
        controls=(
            "signed zero-winding observations supplied separately",
            "runtime residuals retained separately from exact gap identities",
        ),
    )
    exact_manifest.validate_for_admission()
    report = {
        "manifest": manifest.to_dict(),
        "exact_gap_manifest": exact_manifest.to_dict(),
        "source_scope": source_scope,
        "empirical_status": "Untested: no physical data or laboratory experiment",
        "positive_and_zero_cases": [
            run_coupling_case(count, winding)
            for count in (8, 16) for winding in (0, 1)
        ],
        "canonical_transition_counterexample": run_transition_counterexample(),
        "domain_controls": winding_domain_controls(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_payload(report), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote finite canonical winding observations to {args.output}")


if __name__ == "__main__":
    main()
