#!/usr/bin/env python3
"""
Optimized ξ_C validation experiment with minimal verbosity
Focus on data collection efficiency
"""

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import networkx as nx

from benchmarks.benchmark_utils import create_tnfr_topology
from tnfr.config import DNFR_PRIMARY
from tnfr.dynamics.dnfr import default_compute_delta_nfr
from tnfr.metrics.common import compute_coherence
from tnfr.operators.definitions import (
    Coherence,
    Coupling,
    Dissonance,
    Emission,
    Reception,
    Resonance,
    Silence,
    Transition,
)
from tnfr.operators.grammar import validate_sequence
from tnfr.physics.fields import (
    compute_phase_curvature,
    compute_phase_gradient,
    compute_structural_potential,
    estimate_coherence_length,
    measure_phase_symmetry,
)
from tnfr.types import NodeId

# Configure logging to reduce output
logging.basicConfig(level=logging.ERROR)

# Historical candidate intensity used only as a descriptive split.
REFERENCE_INTENSITY = 2.015

# Local grid around the historical candidate.
INTENSITIES = np.array([1.900, 1.950, 2.000, 2.010, 2.015, 2.020, 2.030, 2.050, 2.100])

TOPOLOGIES = ["ws", "scale_free", "grid"]
RUNS_PER_POINT = 20  # Reduced for speed
EVOLUTION_STEPS = 100  # Reduced steps


def create_test_network(topology: str, size: int = 30) -> nx.Graph:
    """Fast network creation"""
    seed = np.random.randint(1000, 9999)
    return create_tnfr_topology(topology, size, seed)


def xi_c_experiment(
    G: nx.Graph, intensity: float, verbose: bool = False
) -> Dict[str, Any]:
    """Optimized experiment - minimal computation"""

    # Basic probe sequence - streamlined
    sequence = [
        (Emission(), 0),
        (Coupling(), 1),
        (Dissonance(), 2),
        (Coherence(), 2),
        (Silence(), 0),
    ]

    try:
        # Apply sequence manually
        for op, node in sequence:
            op(G, node)

        # Fast evolution - manual steps
        for step in range(5):  # Reduced steps
            default_compute_delta_nfr(G)
            # Stabilize some nodes
            if step % 2 == 0:
                coherence_op = Coherence()
                for node in list(G.nodes())[:2]:
                    coherence_op(G, node)

        # Final DNFR calculation
        default_compute_delta_nfr(G)

        # Add per-node coherence
        for node in G.nodes():
            dnfr = abs(G.nodes[node].get(DNFR_PRIMARY, 0.0))
            G.nodes[node]["coherence"] = 1.0 / (1.0 + dnfr)

        # Quick metrics
        results = {}

        # Phase symmetry (core measurement)
        phase_sym_result = measure_phase_symmetry(G)
        xi_c = estimate_coherence_length(G, coherence_key="coherence")

        results["xi_c"] = xi_c
        results["phase_symmetry"] = phase_sym_result.get("symmetry_index", 0.0)
        results["reference_crossing"] = (
            1 if intensity > REFERENCE_INTENSITY else 0
        )

        # Only compute canonical fields if needed for correlation
        if intensity in [2.000, 2.015, 2.030]:  # Sample points for correlation
            phi_s = compute_structural_potential(G)
            grad_phi = compute_phase_gradient(G)
            k_phi = compute_phase_curvature(G)

            results["phi_s_mean"] = np.mean(list(phi_s["values"]))
            results["grad_phi_mean"] = np.mean(list(grad_phi["values"]))
            results["k_phi_mean"] = np.mean(list(k_phi["values"]))

        if verbose:
            print(f"    xi_C = {xi_c:.3f}, symmetry = {results['phase_symmetry']:.3f}")

        return results

    except Exception as e:
        if verbose:
            print(f"    Error: {e}")
        return {"xi_c": np.nan, "error": str(e)}


def run_fast_experiment():
    """Optimized multi-topology experiment"""

    print("Starting optimized xi_C validation experiment...")
    print(f"Intensities: {len(INTENSITIES)} points")
    print(f"Topologies: {TOPOLOGIES}")
    print(f"Runs per point: {RUNS_PER_POINT}")
    print(f"Total measurements: {len(INTENSITIES) * len(TOPOLOGIES) * RUNS_PER_POINT}")

    all_results = []

    for topology in TOPOLOGIES:
        print(f"\n=== {topology.upper()} TOPOLOGY ===")

        for i, intensity in enumerate(INTENSITIES):
            print(f"Intensity {intensity:.3f} ({i+1}/{len(INTENSITIES)})")

            intensity_results = []

            for run in range(RUNS_PER_POINT):
                if run % 5 == 0 and run > 0:
                    print(f"  Run {run}/{RUNS_PER_POINT}")

                # Create fresh network
                G = create_test_network(topology)

                # Run experiment
                result = xi_c_experiment(G, intensity, verbose=False)
                result.update(
                    {"topology": topology, "intensity": intensity, "run": run}
                )

                intensity_results.append(result)
                all_results.append(result)

            # Quick stats for this intensity
            valid_xi_c = [
                r["xi_c"] for r in intensity_results if not np.isnan(r["xi_c"])
            ]
            if valid_xi_c:
                mean_xi_c = np.mean(valid_xi_c)
                print(f"  Mean xi_C: {mean_xi_c:.3f} (n={len(valid_xi_c)})")

    return all_results


def analyze_reference_intensity(results: List[Dict]) -> Dict[str, Any]:
    """Compare samples below and above the legacy reference intensity."""

    print("\n=== REFERENCE-INTENSITY COMPARISON ===")

    analysis = {}

    # Group by topology
    by_topology = {}
    for r in results:
        if r["topology"] not in by_topology:
            by_topology[r["topology"]] = []
        if not np.isnan(r["xi_c"]):
            by_topology[r["topology"]].append(r)

    # Compare both sampled sides for each topology.
    for topology in TOPOLOGIES:
        topo_results = by_topology.get(topology, [])
        if not topo_results:
            continue

        # Group by intensity
        by_intensity = {}
        for r in topo_results:
            intensity = r["intensity"]
            if intensity not in by_intensity:
                by_intensity[intensity] = []
            by_intensity[intensity].append(r["xi_c"])

        # Aggregate the finite measurements.
        intensities = sorted(by_intensity.keys())
        mean_xi_c = []

        for intensity in intensities:
            xi_c_values = by_intensity[intensity]
            mean_val = np.mean(xi_c_values)
            mean_xi_c.append(mean_val)

        # Compare both sides of the historical candidate intensity.
        below_reference = [
            xi for i, xi in zip(intensities, mean_xi_c) if i < REFERENCE_INTENSITY
        ]
        above_reference = [
            xi for i, xi in zip(intensities, mean_xi_c) if i > REFERENCE_INTENSITY
        ]

        if below_reference and above_reference:
            below_mean = np.mean(below_reference)
            above_mean = np.mean(above_reference)
            relative_change = (
                abs(above_mean - below_mean) / below_mean if below_mean > 0 else 0
            )

            analysis[topology] = {
                "below_reference_xi_c": below_mean,
                "above_reference_xi_c": above_mean,
                "relative_change": relative_change,
                "n_points": len(topo_results),
            }

            print(f"{topology.upper()}:")
            print(f"  Below-reference xi_C: {below_mean:.3f}")
            print(f"  Above-reference xi_C: {above_mean:.3f}")
            print(f"  Relative change: {relative_change:.3f}")
            print(f"  Data points: {len(topo_results)}")

    return analysis


def analyze_critical_threshold(results: List[Dict]) -> Dict[str, Any]:
    """Compatibility alias for :func:`analyze_reference_intensity`.

    The historical name does not imply that the reference is a critical point.
    """
    return analyze_reference_intensity(results)


def main():
    """Fast experiment execution"""

    print("xi_C Fast Validation Experiment")
    print("=" * 40)

    # Run experiment
    results = run_fast_experiment()

    # Compare samples around the historical reference intensity.
    analysis = analyze_reference_intensity(results)

    # Summary
    print("\n=== SUMMARY ===")
    print(
        f"Total valid measurements: {len([r for r in results if not np.isnan(r['xi_c'])])}"
    )

    all_xi_c = [r["xi_c"] for r in results if not np.isnan(r["xi_c"])]
    if all_xi_c:
        print(f"Overall xi_C range: {np.min(all_xi_c):.3f} - {np.max(all_xi_c):.3f}")
        print(f"Mean xi_C: {np.mean(all_xi_c):.3f} +/- {np.std(all_xi_c):.3f}")

    # Count large descriptive changes; this is not a transition test.
    large_changes = sum(
        1
        for topo_data in analysis.values()
        if topo_data.get("relative_change", 0) > 0.1
    )

    print(
        f"\nTopologies with >10% cross-reference change: "
        f"{large_changes}/{len(analysis)}"
    )

    if large_changes >= 2:
        print("Descriptive change detected in multiple sampled topologies")
    else:
        print("No broad descriptive change detected on this finite grid")

    print(
        "\nSCOPE: retain as a candidate measurement; a declared dynamic "
        "finite-size protocol is required before any transition claim"
    )

    return results, analysis


if __name__ == "__main__":
    results, analysis = main()
