"""Minimal structural health + telemetry demo (Phase 3).

Shows integration of TelemetryEmitter with the structural validation
aggregator and health summary utilities.

Run:
    python examples/structural_health_demo.py

Outputs:
 - Human-readable health summary
 - Telemetry JSONL lines (in-memory example)

Physics Alignment:
Sequence chosen: [AL, UM, IL, SHA]
 - AL (Emission)   : Generator (U1a)
 - UM (Coupling)   : Requires phase compatibility (U3)
 - IL (Coherence)  : Stabilizer (U2)
 - SHA (Silence)   : Closure (U1b)
This satisfies U1a initiation and U1b closure; includes stabilizer
after coupling; safe canonical bootstrap variant.
"""

from __future__ import annotations

import random
from typing import List

try:
    import networkx as nx  # type: ignore
except ImportError:  # pragma: no cover
    raise SystemExit("networkx required for demo")

from tnfr.metrics.telemetry import TelemetryEmitter
from tnfr.validation.health import compute_structural_health
from tnfr.validation.aggregator import run_structural_validation


def _make_graph(n: int = 16, p: float = 0.15, seed: int = 42):
    random.seed(seed)
    G = nx.erdos_renyi_graph(n, p)  # type: ignore
    # Populate minimal phase & ΔNFR attributes for field computations
    for node in G.nodes:
        G.nodes[node]["phase"] = random.random() * 2.0 * 3.141592653589793
        G.nodes[node]["delta_nfr"] = random.random() * 0.05  # low pressure
    return G


def main() -> None:
    sequence: List[str] = ["AL", "UM", "IL", "SHA"]
    G = _make_graph()

    # Baseline structural potential snapshot
    from tnfr.physics.fields import compute_structural_potential

    baseline_phi_s = compute_structural_potential(G)

    # Telemetry emitter demonstration
    telemetry_path = "results/telemetry/structural_health_demo.jsonl"
    with TelemetryEmitter(telemetry_path) as emitter:
        emitter.record(
            G,
            operator="start",
            extra={"nodes": G.number_of_nodes()},
        )
        report = run_structural_validation(
            G,
            sequence=sequence,
            baseline_structural_potential=baseline_phi_s,
        )
        emitter.record(
            G,
            operator="validation",
            extra={
                "risk_level": report.risk_level,
                "status": report.status,
                "max_phase_gradient": report.field_metrics[
                    "max_phase_gradient"
                ],
            },
        )
        health = compute_structural_health(
            G, sequence=sequence, baseline_phi_s=baseline_phi_s
        )
        emitter.record(
            G,
            operator="health",
            extra={
                "risk_level": health["risk_level"],
                "recommended": health["recommended_actions"],
            },
        )
        emitter.flush()
    print("Telemetry Events (last run):")
    try:
        for ln in open(
            telemetry_path, "r", encoding="utf-8"
        ).read().splitlines()[-3:]:
            print("  ", ln)
    except FileNotFoundError:
        print("  (no telemetry file found)")

    # Human health summary
    print("\nStructural Health Summary:")
    print(f"Status       : {health['status']}")
    print(f"Risk Level   : {health['risk_level']}")
    print("Thresholds   :")
    for k, v in health["thresholds_exceeded"].items():
        print(f"  - {k}: {'EXCEEDED' if v else 'ok'}")
    if health["recommended_actions"]:
        print("Recommended  :", ", ".join(health["recommended_actions"]))
    if health["notes"]:
        print("Notes:")
        for n in health["notes"]:
            print("  -", n)


if __name__ == "__main__":  # pragma: no cover
    main()
