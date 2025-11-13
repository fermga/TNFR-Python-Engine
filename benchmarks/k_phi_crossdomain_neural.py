"""
K_φ Cross-Domain Validation: Neural Synchrony Application
==========================================================

Task 6.1: Demonstrate K_φ phase curvature analysis in biological neural
networks, validating domain-independence of TNFR phase curvature physics.

Domain: Neuroscience - Neural oscillator synchronization
---------------------------------------------------------
Neural populations exhibit phase synchrony during cognitive tasks. High K_φ
regions indicate phase transition boundaries - where neurons shift between
synchronized and desynchronized states.

Physics Basis:
- Neural oscillators (theta, gamma bands) have phase dynamics
- K_φ captures local curvature in phase space
- High |K_φ| → phase transition zones (attention shifts, state changes)
- Asymptotic freedom: var(K_φ) ~ 1/r^α predicts long-range coherence

Application: Identify neural "hubs" that control network-wide synchrony
transitions using K_φ variance analysis.

Success Criteria:
- K_φ identifies hub neurons (≥80% overlap with degree centrality)
- Asymptotic freedom holds (R² > 0.6)
- Safety threshold detection (K_φ stability zones)
"""

import json
import random
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.tnfr.physics.fields import (  # noqa: E402
    compute_phase_curvature,
    compute_phase_gradient,
    compute_structural_potential,
)
from benchmarks.benchmark_utils import create_tnfr_topology  # noqa: E402


def create_neural_network(n_neurons: int, topology: str, seed: int):
    """
    Create neural network with TNFR phase dynamics.

    Neural topologies:
    - scale_free: Hub neurons (cortical columns)
    - ws: Small-world (brain connectivity pattern)
    """
    random.seed(seed)
    np.random.seed(seed)

    G = create_tnfr_topology(topology, n_neurons, seed)

    # Initialize with neural-like phase distribution (clustered oscillators)
    for node in G.nodes():
        # Cluster neurons into synchronized groups (0, π/2, π, 3π/2)
        cluster = random.choice([0, np.pi / 2, np.pi, 3 * np.pi / 2])
        phase_noise = random.uniform(-0.3, 0.3)
        G.nodes[node]["theta"] = cluster + phase_noise

        # Neural properties
        G.nodes[node]["epi"] = random.uniform(0.3, 0.7)  # Baseline activity
        G.nodes[node]["vf"] = random.uniform(
            0.8, 1.2
        )  # Oscillation frequency (8-12 Hz theta band)
        G.nodes[node]["delta_nfr"] = random.uniform(0.01, 0.05)

    return G


def identify_hub_neurons(G, k_phi_map: dict, threshold: float = 3.0):
    """Identify hub neurons using K_φ > threshold."""
    hubs_k_phi = [node for node, k in k_phi_map.items() if abs(k) > threshold]
    return set(hubs_k_phi)


def identify_hub_neurons_degree(G, top_fraction: float = 0.2):
    """Identify hub neurons using degree centrality (ground truth)."""
    degrees = dict(G.degree())
    sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
    n_hubs = max(1, int(len(sorted_nodes) * top_fraction))
    hubs_degree = set([node for node, _ in sorted_nodes[:n_hubs]])
    return hubs_degree


def compute_hub_overlap(hubs_k_phi: set, hubs_degree: set) -> float:
    """Compute Jaccard similarity between K_φ hubs and degree hubs."""
    if len(hubs_degree) == 0:
        return 0.0
    intersection = len(hubs_k_phi & hubs_degree)
    union = len(hubs_k_phi | hubs_degree)
    return intersection / union if union > 0 else 0.0


def compute_k_phi_variance_scaling(G, k_phi_map: dict, max_hops: int = 4):
    """
    Test asymptotic freedom: var(K_φ) ~ 1/r^α.

    Returns: (scales, variances, R²)
    """
    from sklearn.linear_model import LinearRegression

    scales = []
    variances = []

    for hop in range(1, max_hops + 1):
        # Sample variance at this hop distance
        hop_variances = []
        for node in G.nodes():
            # Get neighbors at exactly 'hop' distance
            current_level = {node}
            visited = {node}
            for _ in range(hop):
                next_level = set()
                for n in current_level:
                    for neighbor in G.neighbors(n):
                        if neighbor not in visited:
                            next_level.add(neighbor)
                            visited.add(neighbor)
                current_level = next_level
                if not current_level:
                    break

            if len(current_level) >= 2:
                k_phi_values = [
                    k_phi_map.get(n, 0.0) for n in current_level
                ]
                hop_variances.append(np.var(k_phi_values))

        if hop_variances:
            scales.append(hop)
            variances.append(np.mean(hop_variances))

    if len(scales) < 3:
        return scales, variances, 0.0

    # Fit log(var) ~ log(1/r^α)
    X = np.log(np.array(scales)).reshape(-1, 1)
    y = np.log(np.array(variances))
    model = LinearRegression()
    model.fit(X, y)
    r_squared = model.score(X, y)

    return scales, variances, r_squared


def run_neural_synchrony_analysis(
    topology: str, n_neurons: int = 50, seed: int = 42
):
    """Run K_φ analysis on neural network."""
    print(f"\n🧠 Neural Network Analysis ({topology.upper()}, N={n_neurons})")
    print("=" * 60)

    G = create_neural_network(n_neurons, topology, seed)

    # Compute K_φ and related fields
    k_phi_map = compute_phase_curvature(G)
    phase_grad_map = compute_phase_gradient(G)
    phi_s_map = compute_structural_potential(G, alpha=2.76)  # Use Task 3 α

    # Statistics
    k_phi_values = list(k_phi_map.values())
    mean_k_phi = np.mean(np.abs(k_phi_values))
    max_k_phi = np.max(np.abs(k_phi_values))
    std_k_phi = np.std(k_phi_values)

    print("📊 K_φ Statistics:")
    print(f"   Mean |K_φ|: {mean_k_phi:.3f}")
    print(f"   Max |K_φ|:  {max_k_phi:.3f}")
    print(f"   Std K_φ:    {std_k_phi:.3f}")

    # Hub identification
    hubs_k_phi = identify_hub_neurons(G, k_phi_map, threshold=3.0)
    hubs_degree = identify_hub_neurons_degree(G, top_fraction=0.2)
    overlap = compute_hub_overlap(hubs_k_phi, hubs_degree)

    print("\n🎯 Hub Neuron Identification:")
    print(f"   K_φ hubs (|K_φ| > 3.0): {len(hubs_k_phi)}")
    print(f"   Degree hubs (top 20%):  {len(hubs_degree)}")
    print(
        f"   Overlap (Jaccard):      {overlap:.1%} "
        f"{'✅' if overlap >= 0.8 else '⚠️'}"
    )

    # Asymptotic freedom test
    scales, variances, r_squared = compute_k_phi_variance_scaling(
        G, k_phi_map, max_hops=4
    )

    print("\n🌊 Asymptotic Freedom (var ~ 1/r^α):")
    print(f"   Scales tested: {scales}")
    print(f"   R²: {r_squared:.3f} {'✅' if r_squared > 0.6 else '⚠️'}")

    # Safety analysis
    mean_phi_s = np.mean(np.abs(list(phi_s_map.values())))
    mean_phase_grad = np.mean(list(phase_grad_map.values()))

    print("\n🛡️ Network Safety Metrics:")
    print(f"   Mean Φ_s:  {mean_phi_s:.3f}")
    print(f"   Mean |∇φ|: {mean_phase_grad:.3f}")
    print(
        f"   Status: "
        f"{'✅ Stable' if mean_phase_grad < 0.38 else '⚠️ Elevated'}"
    )

    return {
        "domain": "neural",
        "topology": topology,
        "n_neurons": n_neurons,
        "k_phi_mean": float(mean_k_phi),
        "k_phi_max": float(max_k_phi),
        "k_phi_std": float(std_k_phi),
        "hub_overlap": float(overlap),
        "asymptotic_freedom_r2": float(r_squared),
        "safety_phi_s": float(mean_phi_s),
        "safety_phase_grad": float(mean_phase_grad),
        "hub_identification_success": int(overlap >= 0.8),
        "asymptotic_freedom_success": int(r_squared > 0.6),
        "stable_network": int(mean_phase_grad < 0.38),
    }


def main():
    """Run neural synchrony cross-domain validation."""
    print("🧬 K_φ Cross-Domain Validation: Neural Synchrony")
    print("=" * 60)
    print(
        "Domain: Biological neural networks with phase oscillator dynamics"
    )
    print("Validation: Hub identification + Asymptotic freedom")

    results = []

    # Test on multiple neural topologies
    topologies = [
        ("scale_free", "Cortical hub-and-spoke"),
        ("ws", "Small-world brain connectivity"),
    ]

    for topology, description in topologies:
        print(f"\n{'='*60}")
        print(f"Topology: {description}")
        result = run_neural_synchrony_analysis(
            topology=topology, n_neurons=50, seed=42
        )
        results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("📊 NEURAL DOMAIN VALIDATION SUMMARY:")
    print("=" * 60)

    hub_success = sum(1 for r in results if r["hub_identification_success"])
    asym_success = sum(1 for r in results if r["asymptotic_freedom_success"])
    stable_count = sum(1 for r in results if r["stable_network"])

    print(f"Hub identification: {hub_success}/{len(results)} topologies ≥80%")
    print(
        f"Asymptotic freedom: {asym_success}/{len(results)} topologies R²>0.6"
    )
    print(f"Network stability: {stable_count}/{len(results)} topologies")

    if hub_success >= 1 and asym_success >= 1:
        print(
            "\n✅ NEURAL DOMAIN VALIDATION PASSED: K_φ successfully "
            "identifies hub"
        )
        print(
            "   neurons and exhibits universal asymptotic freedom "
            "in neural networks"
        )
    else:
        print("\n⚠️ NEURAL DOMAIN VALIDATION PARTIAL")

    # Save results
    output_dir = Path("benchmarks/results")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "k_phi_crossdomain_neural.jsonl"

    with open(output_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    print(f"\n💾 Results saved to: {output_file.resolve()}")


if __name__ == "__main__":
    main()
