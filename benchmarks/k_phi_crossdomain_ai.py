"""
K_φ Cross-Domain Validation: AI Attention Mechanisms
=====================================================

Task 6.3: Demonstrate K_φ phase curvature analysis in artificial attention
networks, validating domain-independence of TNFR phase curvature physics.

Domain: AI/ML - Transformer attention patterns
-----------------------------------------------
Attention mechanisms create dynamic "focus" patterns across token sequences.
High K_φ regions indicate attention shifts - where the model transitions
between different semantic contexts.

Physics Basis:
- Attention heads have "attention phase" (alignment with query vectors)
- K_φ captures local curvature in attention space
- High |K_φ| → transition zones (context switches, ambiguity resolution)
- Asymptotic freedom: var(K_φ) ~ 1/r^α predicts global attention dynamics

Application: Identify "pivot tokens" where attention dramatically shifts
using K_φ variance analysis.

Success Criteria:
- K_φ identifies pivot tokens (attention variance correlation > 0.4)
- Asymptotic freedom holds (R² > 0.6)
- Detects attention bottlenecks (high |K_φ| nodes)
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
)
from benchmarks.benchmark_utils import create_tnfr_topology  # noqa: E402


def create_attention_network(n_tokens: int, topology: str, seed: int):
    """
    Create attention network with TNFR phase dynamics.

    AI topologies:
    - scale_free: Hub-based attention (key tokens dominate)
    - ring: Sequential attention (causal masking)
    """
    random.seed(seed)
    np.random.seed(seed)

    G = create_tnfr_topology(topology, n_tokens, seed)

    # Create "semantic clusters" with distinct attention phases
    n_clusters = 3  # Subject, predicate, object
    cluster_phases = [i * (2 * np.pi / n_clusters) for i in range(n_clusters)]

    for node in G.nodes():
        # Assign to semantic cluster
        cluster = node % n_clusters
        cluster_phase = cluster_phases[cluster]
        phase_noise = random.uniform(-0.5, 0.5)  # Token-specific variation
        G.nodes[node]["theta"] = cluster_phase + phase_noise

        # Attention properties
        G.nodes[node]["epi"] = random.uniform(
            0.1, 0.9
        )  # Attention weight
        G.nodes[node]["vf"] = random.uniform(
            0.5, 1.5
        )  # Update frequency
        G.nodes[node]["delta_nfr"] = random.uniform(0.01, 0.05)

    return G


def compute_attention_variance(G):
    """
    Compute attention variance for each token.

    Simulates how much attention weight varies across time steps.
    """
    attention_variance = {}
    for node in G.nodes():
        # Approximate: variance of EPI (attention weight) across neighbors
        neighbors = list(G.neighbors(node))
        if len(neighbors) >= 2:
            epi_values = [G.nodes[n]["epi"] for n in neighbors]
            attention_variance[node] = float(np.var(epi_values))
        else:
            attention_variance[node] = 0.0
    return attention_variance


def identify_pivot_tokens_k_phi(G, k_phi_map: dict, threshold: float = 3.0):
    """Identify pivot tokens using K_φ > threshold."""
    pivots = [node for node, k in k_phi_map.items() if abs(k) > threshold]
    return set(pivots)


def identify_pivot_tokens_attention(G, top_fraction: float = 0.2):
    """Identify pivot tokens using attention variance."""
    attention_var = compute_attention_variance(G)
    sorted_nodes = sorted(
        attention_var.items(), key=lambda x: x[1], reverse=True
    )
    n_pivots = max(1, int(len(sorted_nodes) * top_fraction))
    return set([node for node, _ in sorted_nodes[:n_pivots]])


def compute_pivot_correlation(k_phi_map: dict, attention_var: dict) -> float:
    """Compute Pearson correlation between |K_φ| and attention variance."""
    nodes = list(k_phi_map.keys())
    k_phi_values = [abs(k_phi_map[n]) for n in nodes]
    attention_values = [attention_var.get(n, 0.0) for n in nodes]

    k_phi_array = np.array(k_phi_values)
    attention_array = np.array(attention_values)

    if np.std(k_phi_array) == 0 or np.std(attention_array) == 0:
        return 0.0

    correlation = np.corrcoef(k_phi_array, attention_array)[0, 1]
    return float(correlation)


def compute_asymptotic_freedom_simple(G, k_phi_map: dict) -> float:
    """
    Simplified asymptotic freedom test: variance decay with distance.

    Returns R² of power law fit.
    """
    from sklearn.linear_model import LinearRegression

    max_hops = min(4, len(G.nodes()) // 10)
    if max_hops < 2:
        return 0.0

    scales = []
    variances = []

    for hop in range(1, max_hops + 1):
        hop_vars = []
        for node in G.nodes():
            # Sample neighbors at hop distance
            neighbors_at_hop = set()
            current = {node}
            visited = {node}
            for _ in range(hop):
                next_level = set()
                for n in current:
                    for neighbor in G.neighbors(n):
                        if neighbor not in visited:
                            next_level.add(neighbor)
                            visited.add(neighbor)
                current = next_level
                if not current:
                    break
                neighbors_at_hop = current

            if len(neighbors_at_hop) >= 2:
                k_vals = [k_phi_map.get(n, 0.0) for n in neighbors_at_hop]
                hop_vars.append(np.var(k_vals))

        if hop_vars:
            scales.append(hop)
            variances.append(np.mean(hop_vars))

    if len(scales) < 2:
        return 0.0

    X = np.log(np.array(scales)).reshape(-1, 1)
    y = np.log(np.array(variances))
    model = LinearRegression()
    model.fit(X, y)
    return float(model.score(X, y))


def run_attention_network_analysis(
    topology: str, n_tokens: int = 50, seed: int = 42
):
    """Run K_φ analysis on AI attention network."""
    print(f"\n🤖 Attention Network Analysis ({topology.upper()}, N={n_tokens})")
    print("=" * 60)

    G = create_attention_network(n_tokens, topology, seed)

    # Compute K_φ and related fields
    k_phi_map = compute_phase_curvature(G)
    phase_grad_map = compute_phase_gradient(G)

    # Statistics
    k_phi_values = list(k_phi_map.values())
    mean_k_phi = np.mean(np.abs(k_phi_values))
    max_k_phi = np.max(np.abs(k_phi_values))
    std_k_phi = np.std(k_phi_values)

    print("📊 K_φ Statistics:")
    print(f"   Mean |K_φ|: {mean_k_phi:.3f}")
    print(f"   Max |K_φ|:  {max_k_phi:.3f}")
    print(f"   Std K_φ:    {std_k_phi:.3f}")

    # Pivot token identification
    attention_var = compute_attention_variance(G)
    correlation = compute_pivot_correlation(k_phi_map, attention_var)

    pivots_k_phi = identify_pivot_tokens_k_phi(G, k_phi_map, threshold=3.0)
    pivots_attention = identify_pivot_tokens_attention(G, top_fraction=0.2)

    print("\n🎯 Pivot Token Identification:")
    print(f"   K_φ pivots (|K_φ| > 3.0):        {len(pivots_k_phi)}")
    print(f"   Attention pivots (top 20%):      {len(pivots_attention)}")
    print(
        f"   Correlation (K_φ, attn_var):     {correlation:+.3f} "
        f"{'✅' if abs(correlation) > 0.4 else '⚠️'}"
    )

    # Asymptotic freedom
    r_squared = compute_asymptotic_freedom_simple(G, k_phi_map)

    print("\n🌊 Asymptotic Freedom (var ~ 1/r^α):")
    print(f"   R²: {r_squared:.3f} {'✅' if r_squared > 0.6 else '⚠️'}")

    # Attention bottlenecks
    bottlenecks = [node for node, k in k_phi_map.items() if abs(k) > 4.0]

    print("\n⚡ Attention Bottlenecks:")
    print(f"   High |K_φ| (>4.0): {len(bottlenecks)} tokens")

    # Safety metrics
    mean_phase_grad = np.mean(list(phase_grad_map.values()))

    print("\n🛡️ Network Safety:")
    print(f"   Mean |∇φ|: {mean_phase_grad:.3f}")

    return {
        "domain": "ai",
        "topology": topology,
        "n_tokens": n_tokens,
        "k_phi_mean": float(mean_k_phi),
        "k_phi_max": float(max_k_phi),
        "k_phi_std": float(std_k_phi),
        "pivot_correlation": float(correlation),
        "asymptotic_freedom_r2": float(r_squared),
        "n_bottlenecks": len(bottlenecks),
        "bottleneck_rate": float(len(bottlenecks) / n_tokens),
        "safety_phase_grad": float(mean_phase_grad),
        "pivot_identification_success": int(abs(correlation) > 0.4),
        "asymptotic_freedom_success": int(r_squared > 0.6),
    }


def main():
    """Run AI attention cross-domain validation."""
    print("🤖 K_φ Cross-Domain Validation: AI Attention Mechanisms")
    print("=" * 60)
    print("Domain: Artificial attention networks with dynamic focus")
    print("Validation: Pivot identification + Asymptotic freedom")

    results = []

    # Test on AI topologies
    topologies = [
        ("scale_free", "Hub-based attention (key tokens)"),
        ("ring", "Sequential attention (causal)"),
    ]

    for topology, description in topologies:
        print(f"\n{'='*60}")
        print(f"Topology: {description}")
        result = run_attention_network_analysis(
            topology=topology, n_tokens=50, seed=42
        )
        results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("📊 AI DOMAIN VALIDATION SUMMARY:")
    print("=" * 60)

    pivot_success = sum(
        1 for r in results if r["pivot_identification_success"]
    )
    asym_success = sum(
        1 for r in results if r["asymptotic_freedom_success"]
    )

    print(
        f"Pivot identification: {pivot_success}/{len(results)} "
        "topologies |corr|>0.4"
    )
    print(
        f"Asymptotic freedom: {asym_success}/{len(results)} "
        "topologies R²>0.6"
    )

    if pivot_success >= 1 and asym_success >= 1:
        print(
            "\n✅ AI DOMAIN VALIDATION PASSED: K_φ successfully "
            "identifies pivot"
        )
        print(
            "   tokens and exhibits asymptotic freedom in "
            "attention networks"
        )
    else:
        print("\n⚠️ AI DOMAIN VALIDATION PARTIAL")

    # Save results
    output_dir = Path("benchmarks/results")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "k_phi_crossdomain_ai.jsonl"

    with open(output_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    print(f"\n💾 Results saved to: {output_file.resolve()}")


if __name__ == "__main__":
    main()
