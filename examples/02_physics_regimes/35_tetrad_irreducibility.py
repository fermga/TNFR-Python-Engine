"""Example 35: Structural tetrad diagnostic complementarity.

The legacy filename is retained for discoverability. The experiment constructs
four reproducible probes that emphasize different tetrad read-outs:

- ``Phi_s`` aggregates graph-distance-weighted structural pressure.
- ``|grad_phi|`` reports local unsigned phase mismatch.
- ``K_phi`` reports signed circular phase curvature.
- ``xi_C`` reports a non-local correlation estimate or spectral fallback.

The comparisons use selected monitoring policies. Only the wrapped phase
bounds ``|grad_phi| <= pi`` and ``|K_phi| <= pi`` are exact. A crossed policy
is a telemetry flag, and a finite ``xi_C`` relative to graph diameter does not
prove correlation-length divergence or a phase transition. These finite probes
show complementary information; they do not prove that the tetrad is minimal,
complete, irreducible, or sufficient to reconstruct graph state or dynamics.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Any

import networkx as nx
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tnfr.constants import inject_defaults
from tnfr.constants.canonical import (
    GRAD_PHI_CANONICAL_THRESHOLD,
    K_PHI_CANONICAL_THRESHOLD,
    PHI_S_VON_KOCH_THRESHOLD,
    PI,
)
from tnfr.physics.fields import (
    compute_phase_curvature,
    compute_phase_gradient,
    compute_structural_potential,
    estimate_coherence_length_with_provenance,
)


def _build_and_inject(graph: nx.Graph, seed: int = 42) -> None:
    """Inject TNFR defaults and deterministic non-trivial node state."""
    rng = np.random.default_rng(seed)
    inject_defaults(graph)
    for node in graph.nodes():
        phase = float(rng.uniform(0.0, 2.0 * PI))
        graph.nodes[node]["phase"] = phase
        graph.nodes[node]["theta"] = phase
        graph.nodes[node]["delta_nfr"] = float(rng.uniform(-0.3, 0.3))
        graph.nodes[node]["nu_f"] = float(rng.uniform(0.8, 1.2))


def _set_phase(graph: nx.Graph, node: Any, phase: float) -> None:
    """Set both supported phase keys to keep the probe representation aligned."""
    graph.nodes[node]["phase"] = float(phase)
    graph.nodes[node]["theta"] = float(phase)


def _safe_max(values: dict[Any, float]) -> float:
    return max((abs(float(value)) for value in values.values()), default=0.0)


def _report_fields(graph: nx.Graph) -> dict[str, dict[str, Any]]:
    """Capture tetrad summaries and label each comparison by its scope."""
    phi_s = compute_structural_potential(graph)
    grad_phi = compute_phase_gradient(graph)
    k_phi = compute_phase_curvature(graph)
    xi_c = estimate_coherence_length_with_provenance(graph)
    diameter = float(nx.diameter(graph)) if nx.is_connected(graph) else float("nan")
    xi_comparison = diameter

    return {
        "Phi_s": {
            "value": _safe_max(phi_s),
            "comparison": PHI_S_VON_KOCH_THRESHOLD,
            "kind": "selected pi/4 policy",
            "crossed": _safe_max(phi_s) >= PHI_S_VON_KOCH_THRESHOLD,
        },
        "|grad_phi|": {
            "value": _safe_max(grad_phi),
            "comparison": GRAD_PHI_CANONICAL_THRESHOLD,
            "kind": "selected pi/16 policy",
            "crossed": _safe_max(grad_phi) >= GRAD_PHI_CANONICAL_THRESHOLD,
            "exact_bound_holds": _safe_max(grad_phi) <= PI + 1e-12,
        },
        "K_phi": {
            "value": _safe_max(k_phi),
            "comparison": K_PHI_CANONICAL_THRESHOLD,
            "kind": "selected 0.9pi margin",
            "crossed": _safe_max(k_phi) >= K_PHI_CANONICAL_THRESHOLD,
            "exact_bound_holds": _safe_max(k_phi) <= PI + 1e-12,
        },
        "xi_C": {
            "value": float(xi_c.value),
            "comparison": xi_comparison,
            "kind": f"descriptive diameter; {xi_c.method}",
            "crossed": bool(
                math.isfinite(xi_c.value)
                and math.isfinite(xi_comparison)
                and xi_c.value >= xi_comparison
            ),
        },
    }


def _print_field_status(
    fields: dict[str, dict[str, Any]], emphasized_field: str
) -> None:
    """Print field values without turning monitoring flags into diagnoses."""
    print(
        f"  {'Field':<14}  {'Value':>10}  {'Comparison':>10}  "
        f"{'At/above?':>9}  {'Scope':<34}  {'Focus':>5}"
    )
    print("  " + "-" * 92)
    for name, info in fields.items():
        marker = "<<<" if name == emphasized_field else ""
        print(
            f"  {name:<14}  {info['value']:10.4g}  "
            f"{info['comparison']:10.4g}  {str(info['crossed']):>9}  "
            f"{info['kind']:<34}  {marker:>5}"
        )


def demo_pressure_aggregation_probe() -> None:
    """Show that Phi_s carries graph-distance-weighted pressure information."""
    print("=" * 72)
    print("  PROBE 1: Structural-pressure aggregation")
    print("=" * 72)
    print("\n  Protocol: uniform phase with elevated DELTA_NFR on a star graph.\n")

    graph = nx.star_graph(30)
    _build_and_inject(graph, seed=10)
    for node in graph.nodes():
        _set_phase(graph, node, 0.25)
        graph.nodes[node]["delta_nfr"] = 2.0
    graph.nodes[0]["delta_nfr"] = 3.0

    fields = _report_fields(graph)
    _print_field_status(fields, "Phi_s")
    print("\n  Phi_s exposes non-local source aggregation while both exact")
    print("  phase bounds remain satisfied. The pi/4 crossing is a selected")
    print("  magnitude warning, not a graph-independent potential bound.")


def demo_local_phase_mismatch_probe() -> None:
    """Show the unsigned local mismatch carried by |grad_phi|."""
    print("\n" + "=" * 72)
    print("  PROBE 2: Local phase mismatch")
    print("=" * 72)
    print("\n  Protocol: two phase domains on a seeded small-world graph.\n")

    graph = nx.watts_strogatz_graph(40, 4, 0.2, seed=42)
    _build_and_inject(graph, seed=42)
    nodes = sorted(graph.nodes())
    half = len(nodes) // 2
    for node in nodes[:half]:
        _set_phase(graph, node, 0.05)
        graph.nodes[node]["delta_nfr"] = 0.1
    for node in nodes[half:]:
        _set_phase(graph, node, PI - 0.05)
        graph.nodes[node]["delta_nfr"] = 0.1

    fields = _report_fields(graph)
    _print_field_status(fields, "|grad_phi|")
    print("\n  |grad_phi| records unsigned neighbor mismatch. K_phi may also")
    print("  respond because the same phase boundary has curvature; this probe")
    print("  demonstrates complementary readings rather than exclusive detection.")


def demo_phase_curvature_probe() -> None:
    """Show the signed circular-curvature channel on a localized phase defect."""
    print("\n" + "=" * 72)
    print("  PROBE 3: Circular phase curvature")
    print("=" * 72)
    print("\n  Protocol: one localized phase defect on an otherwise uniform ring.\n")

    graph = nx.cycle_graph(12)
    _build_and_inject(graph, seed=7)
    for node in graph.nodes():
        _set_phase(graph, node, 0.0)
        graph.nodes[node]["delta_nfr"] = 0.1
    _set_phase(graph, 0, PI)

    fields = _report_fields(graph)
    _print_field_status(fields, "K_phi")
    print("\n  K_phi records signed departure from the circular neighbor mean;")
    print("  |grad_phi| records mismatch magnitude. Both may cross their selected")
    print("  policies, while their exact wrapped-angle bounds remain pi.")


def demo_nonlocal_scale_probe() -> None:
    """Show xi_C provenance without claiming divergence or criticality."""
    print("\n" + "=" * 72)
    print("  PROBE 4: Non-local coherence-length read-out")
    print("=" * 72)
    print("\n  Protocol: uniform phase and pressure on a seeded small-world graph.\n")

    graph = nx.watts_strogatz_graph(50, 4, 0.3, seed=42)
    _build_and_inject(graph, seed=42)
    for node in graph.nodes():
        _set_phase(graph, node, 1.0)
        graph.nodes[node]["delta_nfr"] = 0.1

    fields = _report_fields(graph)
    _print_field_status(fields, "xi_C")
    print("\n  Pointwise phase derivatives vanish, while xi_C still reports a")
    print("  non-local scale with explicit estimator provenance. Its finite value")
    print("  and diameter comparison do not establish divergence or a transition.")


def demo_complementarity_summary() -> None:
    """Summarize the information each diagnostic retains and its limitation."""
    print("\n" + "=" * 72)
    print("  DIAGNOSTIC COMPLEMENTARITY SUMMARY")
    print("=" * 72)

    table = [
        ("Phi_s", "source aggregation", "depends on pressure and graph metric"),
        ("|grad_phi|", "unsigned local mismatch", "does not retain curvature sign"),
        ("K_phi", "signed circular curvature", "is a local phase read-out"),
        ("xi_C", "non-local length estimate", "fit/fallback and graph scope matter"),
    ]
    print(f"\n  {'Field':<14}  {'Information retained':<28}  {'Scope limit':<38}")
    print("  " + "-" * 84)
    for field, information, limitation in table:
        print(f"  {field:<14}  {information:<28}  {limitation:<38}")

    print("\n  These probes support using all four diagnostics together. They do")
    print("  not prove a minimal or complete state basis, universal thresholds,")
    print("  transition criticality, or reconstruction of the nodal trajectory.")


def main() -> None:
    print()
    print("*" * 72)
    print("  TNFR Example 35: Structural Tetrad Diagnostic Complementarity")
    print("  Finite seeded probes; exact bounds and selected policies separated")
    print("*" * 72)

    demo_pressure_aggregation_probe()
    demo_local_phase_mismatch_probe()
    demo_phase_curvature_probe()
    demo_nonlocal_scale_probe()
    demo_complementarity_summary()


if __name__ == "__main__":
    main()
