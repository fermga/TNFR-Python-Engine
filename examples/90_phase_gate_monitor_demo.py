#!/usr/bin/env python3
"""Example 90 — Phase-Gate Graph Signal Monitor.

This example shows a concrete application of TNFR phase-gated telemetry:
detecting when a graph signal has acceptable global statistics but broken
local coupling along graph edges.

Run:
    python examples/90_phase_gate_monitor_demo.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

loaded_tnfr = sys.modules.get("tnfr")
loaded_path = str(getattr(loaded_tnfr, "__file__", "")) if loaded_tnfr else ""
if loaded_tnfr is not None and not loaded_path.startswith(str(SRC)):
    for name in list(sys.modules):
        if name == "tnfr" or name.startswith("tnfr."):
            del sys.modules[name]

import networkx as nx  # noqa: E402

from tnfr.validation.phase_gate import (  # noqa: E402
    DEFAULT_PHASE_GATE,
    analyze_phase_gate,
    export_phase_gate_report,
)


def build_sensor_ring(n: int = 24, scrambled: bool = False) -> nx.Graph:
    """Build a ring-like sensor network with smooth or scrambled phases."""
    G = nx.cycle_graph(n)
    phases = [2.0 * math.pi * i / n for i in range(n)]
    if scrambled:
        # Deterministic permutation: same phase histogram, broken locality.
        phases = [phases[(i * 7) % n] for i in range(n)]
    for node, phase in enumerate(phases):
        G.nodes[node]["phase"] = phase
        G.nodes[node]["theta"] = phase
        G.nodes[node]["delta_nfr"] = 0.05
        G.nodes[node]["dnfr"] = 0.05
    return G


def print_report(name: str, G: nx.Graph) -> None:
    report = analyze_phase_gate(G, gate=DEFAULT_PHASE_GATE, top_n=3)
    baselines = report.baseline_summary
    print(f"\n{name}")
    print("-" * len(name))
    print(f"Recommendation: {report.recommendation}")
    print(f"Edge compliance: {report.compliance.compliance_ratio:.3f}")
    print(f"Global order R:  {baselines['global_order_r']:.6f}")
    print(f"TNFR grad mean:  {baselines['tnfr_mean_phase_gradient']:.6f}")
    print("Top hotspots:")
    for hotspot in report.hotspots:
        print(
            f"  node={hotspot.node:>2} "
            f"grad={hotspot.phase_gradient:.3f} "
            f"violations={hotspot.incident_violation_count} "
            f"score={hotspot.stress_score:.3f}"
        )
    print("TNFR operator prescription:")
    for prescription in report.operator_prescriptions[:3]:
        sequence = " -> ".join(prescription.sequence)
        print(
            f"  {prescription.scope}:{prescription.target} "
            f"priority={prescription.priority:.3f} sequence={sequence}"
        )


def main() -> None:
    smooth = build_sensor_ring(scrambled=False)
    scrambled = build_sensor_ring(scrambled=True)

    print("TNFR Phase-Gate Graph Signal Monitor")
    print("Same topology. Same phase histogram. Different local compatibility.")
    print_report("Smooth local signal", smooth)
    print_report("Scrambled local signal", scrambled)

    output = ROOT / "results" / "reports" / "phase_gate_monitor_demo.html"
    export_phase_gate_report(
        scrambled,
        output,
        gate=DEFAULT_PHASE_GATE,
        title="Phase-Gate Monitor Demo: Scrambled Signal",
    )
    print(f"\nExported hotspot report: {output}")


if __name__ == "__main__":
    main()
