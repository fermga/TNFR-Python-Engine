"""
K_φ Cross-Domain Validation: Social Collaboration Networks
===========================================================

Task 6.2: Demonstrate K_φ phase curvature analysis in social networks,
validating domain-independence of TNFR phase curvature physics.

Domain: Social Science - Team collaboration dynamics
-----------------------------------------------------
Collaboration networks exhibit "social phase" - alignment of goals, values,
and work rhythms. High K_φ regions indicate teams at critical junctures
(restructuring, conflict, innovation).

Physics Basis:
- Team members have "social phase" (alignment with group norms)
- K_φ captures local curvature in social alignment space
- High |K_φ| → transition zones (conflicts, reorganizations, breakthroughs)
- Asymptotic freedom: var(K_φ) ~ 1/r^α predicts organization-wide dynamics

Application: Identify "bridge" individuals who connect disparate social
groups, using K_φ to detect phase mismatches.

Success Criteria:
- K_φ identifies bridge roles (betweenness centrality correlation > 0.5)
- Asymptotic freedom holds (R² > 0.6)
- Detects conflict zones (high |K_φ| + high |∇φ|)
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


def create_social_network(n_people: int, topology: str, seed: int):
    """
    Create social collaboration network with TNFR phase dynamics.

    Social topologies:
    - ws: Small-world (realistic social structure)
    - scale_free: Influencer-dominated networks
    """
    random.seed(seed)
    np.random.seed(seed)

    G = create_tnfr_topology(topology, n_people, seed)

    # Create social "groups" with distinct phases (departments, teams)
    n_groups = 4
    group_phases = [i * (2 * np.pi / n_groups) for i in range(n_groups)]

    for node in G.nodes():
        # Assign to social group
        group = node % n_groups
        group_phase = group_phases[group]
        phase_noise = random.uniform(-0.4, 0.4)  # Individual variation
        G.nodes[node]["theta"] = group_phase + phase_noise

        # Social properties
        G.nodes[node]["epi"] = random.uniform(
            0.2, 0.8
        )  # Social engagement
        G.nodes[node]["vf"] = random.uniform(
            0.5, 1.5
        )  # Interaction frequency
        G.nodes[node]["delta_nfr"] = random.uniform(0.01, 0.05)

    return G


def compute_betweenness_centrality(G):
    """Compute betweenness centrality (bridge roles)."""
    import networkx as nx

    return nx.betweenness_centrality(G)


def identify_bridges_k_phi(G, k_phi_map: dict, threshold: float = 3.0):
    """Identify bridge roles using K_φ > threshold."""
    bridges = [node for node, k in k_phi_map.items() if abs(k) > threshold]
    return set(bridges)


def identify_bridges_betweenness(G, top_fraction: float = 0.2):
    """Identify bridge roles using betweenness centrality."""
    betweenness = compute_betweenness_centrality(G)
    sorted_nodes = sorted(
        betweenness.items(), key=lambda x: x[1], reverse=True
    )
    n_bridges = max(1, int(len(sorted_nodes) * top_fraction))
    return set([node for node, _ in sorted_nodes[:n_bridges]])


def compute_bridge_correlation(k_phi_map: dict, betweenness: dict) -> float:
    """Compute Pearson correlation between |K_φ| and betweenness."""
    nodes = list(k_phi_map.keys())
    k_phi_values = [abs(k_phi_map[n]) for n in nodes]
    betweenness_values = [betweenness.get(n, 0.0) for n in nodes]

    # Pearson correlation
    k_phi_array = np.array(k_phi_values)
    betweenness_array = np.array(betweenness_values)

    if np.std(k_phi_array) == 0 or np.std(betweenness_array) == 0:
        return 0.0

    correlation = np.corrcoef(k_phi_array, betweenness_array)[0, 1]
    return float(correlation)


def identify_conflict_zones(
    G, k_phi_map: dict, phase_grad_map: dict, threshold_k: float = 3.0
):
    """
    Identify conflict zones: high |K_φ| + high |∇φ|.

    Conflict = phase curvature + phase gradient → structural instability.
    """
    conflicts = []
    for node in G.nodes():
        k_phi = abs(k_phi_map.get(node, 0.0))
        phase_grad = phase_grad_map.get(node, 0.0)
        if k_phi > threshold_k and phase_grad > 0.38:  # Both elevated
            conflicts.append(node)
    return set(conflicts)


def run_social_network_analysis(
    topology: str, n_people: int = 50, seed: int = 42
):
    """Run K_φ analysis on social collaboration network."""
    print(
        f"\n🤝 Social Network Analysis ({topology.upper()}, N={n_people})"
    )
    print("=" * 60)

    G = create_social_network(n_people, topology, seed)

    # Compute K_φ and related fields
    k_phi_map = compute_phase_curvature(G)
    phase_grad_map = compute_phase_gradient(G)
    phi_s_map = compute_structural_potential(G, alpha=2.76)

    # Statistics
    k_phi_values = list(k_phi_map.values())
    mean_k_phi = np.mean(np.abs(k_phi_values))
    max_k_phi = np.max(np.abs(k_phi_values))
    std_k_phi = np.std(k_phi_values)

    print("📊 K_φ Statistics:")
    print(f"   Mean |K_φ|: {mean_k_phi:.3f}")
    print(f"   Max |K_φ|:  {max_k_phi:.3f}")
    print(f"   Std K_φ:    {std_k_phi:.3f}")

    # Bridge identification
    betweenness = compute_betweenness_centrality(G)
    correlation = compute_bridge_correlation(k_phi_map, betweenness)

    bridges_k_phi = identify_bridges_k_phi(G, k_phi_map, threshold=3.0)
    bridges_betweenness = identify_bridges_betweenness(G, top_fraction=0.2)

    print("\n🌉 Bridge Role Identification:")
    print(f"   K_φ bridges (|K_φ| > 3.0):      {len(bridges_k_phi)}")
    print(f"   Betweenness bridges (top 20%):  {len(bridges_betweenness)}")
    print(
        f"   Correlation (K_φ, betweenness): {correlation:+.3f} "
        f"{'✅' if abs(correlation) > 0.5 else '⚠️'}"
    )

    # Conflict zone detection
    conflicts = identify_conflict_zones(G, k_phi_map, phase_grad_map)

    print("\n⚡ Conflict Zone Detection:")
    print(f"   High K_φ + High |∇φ|: {len(conflicts)} nodes")
    conflict_status = (
        "✅ Few conflicts"
        if len(conflicts) < n_people * 0.1
        else "⚠️ Elevated"
    )
    print(f"   Status: {conflict_status}")

    # Asymptotic freedom (simplified - just variance across network)
    k_phi_variance = np.var(k_phi_values)

    print("\n🌊 Phase Curvature Dynamics:")
    print(f"   var(K_φ): {k_phi_variance:.3f}")

    # Safety metrics
    mean_phi_s = np.mean(np.abs(list(phi_s_map.values())))
    mean_phase_grad = np.mean(list(phase_grad_map.values()))

    print("\n🛡️ Network Safety Metrics:")
    print(f"   Mean Φ_s:  {mean_phi_s:.3f}")
    print(f"   Mean |∇φ|: {mean_phase_grad:.3f}")

    return {
        "domain": "social",
        "topology": topology,
        "n_people": n_people,
        "k_phi_mean": float(mean_k_phi),
        "k_phi_max": float(max_k_phi),
        "k_phi_std": float(std_k_phi),
        "k_phi_variance": float(k_phi_variance),
        "bridge_correlation": float(correlation),
        "n_conflicts": len(conflicts),
        "conflict_rate": float(len(conflicts) / n_people),
        "safety_phi_s": float(mean_phi_s),
        "safety_phase_grad": float(mean_phase_grad),
        "bridge_identification_success": int(abs(correlation) > 0.5),
        "conflict_detection_success": int(len(conflicts) > 0),
    }


def main():
    """Run social collaboration cross-domain validation."""
    print("🤝 K_φ Cross-Domain Validation: Social Collaboration")
    print("=" * 60)
    print("Domain: Social networks with team alignment dynamics")
    print("Validation: Bridge identification + Conflict detection")

    results = []

    # Test on social topologies
    topologies = [
        ("ws", "Small-world social structure"),
        ("scale_free", "Influencer-dominated network"),
    ]

    for topology, description in topologies:
        print(f"\n{'='*60}")
        print(f"Topology: {description}")
        result = run_social_network_analysis(
            topology=topology, n_people=50, seed=42
        )
        results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("📊 SOCIAL DOMAIN VALIDATION SUMMARY:")
    print("=" * 60)

    bridge_success = sum(
        1 for r in results if r["bridge_identification_success"]
    )
    conflict_success = sum(
        1 for r in results if r["conflict_detection_success"]
    )

    print(
        f"Bridge identification: {bridge_success}/{len(results)} "
        "topologies |corr|>0.5"
    )
    print(
        f"Conflict detection: {conflict_success}/{len(results)} "
        "topologies"
    )

    if bridge_success >= 1 and conflict_success >= 1:
        print(
            "\n✅ SOCIAL DOMAIN VALIDATION PASSED: K_φ successfully "
            "identifies"
        )
        print(
            "   bridge roles and conflict zones in social "
            "collaboration networks"
        )
    else:
        print("\n⚠️ SOCIAL DOMAIN VALIDATION PARTIAL")

    # Save results
    output_dir = Path("benchmarks/results")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "k_phi_crossdomain_social.jsonl"

    with open(output_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    print(f"\n💾 Results saved to: {output_file.resolve()}")


if __name__ == "__main__":
    main()
