"""
K_φ-Based Mutation Prediction Optimization Test
================================================

Task 4: Validate that high |K_φ| variance nodes are optimal ZHIR (Mutation)
targets with ≥20% improvement over random targeting.

Strategy:
---------
1. Create networks across 4 topologies (ring, scale_free, ws, tree)
2. Compute 1-hop K_φ variance for all nodes
3. Apply ZHIR to:
   a) Top 20% high-variance nodes (K_φ-targeted)
   b) Random 20% nodes (baseline)
4. Measure mutation success via:
   - Phase transition (θ change > 0.2 radians)
   - Coherence preservation (C drop < 30%)
   - ΔNFR elevation (indicates structural change)
5. Compare success rates: targeted vs random

Success Criterion: ≥20% improvement in mutation success rate

Physics Basis:
--------------
High K_φ variance indicates regions where phase curvature fluctuates
strongly - precisely where structural reorganization (ZHIR) can trigger
phase transitions most effectively.
"""

import json
import logging
import random
import sys
from pathlib import Path

import networkx as nx
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.tnfr.dynamics.dnfr import default_compute_delta_nfr  # noqa: E402
from src.tnfr.metrics.common import compute_coherence  # noqa: E402
from src.tnfr.operators.definitions import (  # noqa: E402
    Coherence,
    Dissonance,
    Mutation,
)
from src.tnfr.physics.fields import compute_phase_curvature  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


def compute_k_phi_1hop_variance(G: nx.Graph) -> dict[int, float]:
    """
    Compute 1-hop K_φ variance for each node.

    Higher variance = more fluctuation in local phase curvature
    = better mutation target.

    Args:
        G: NetworkX graph with 'theta' and phase structure

    Returns:
        Dict mapping node_id -> K_φ variance
    """
    k_phi_map = compute_phase_curvature(G)
    variance_map = {}

    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if len(neighbors) < 2:
            variance_map[node] = 0.0
            continue

        # Collect K_φ values in 1-hop neighborhood
        k_phi_values = [k_phi_map.get(node, 0.0)] + [
            k_phi_map.get(n, 0.0) for n in neighbors
        ]
        variance_map[node] = float(np.var(k_phi_values))

    return variance_map


def select_top_variance_nodes(
    variance_map: dict[int, float], fraction: float = 0.2
) -> list[int]:
    """Select top fraction of nodes by K_φ variance."""
    sorted_nodes = sorted(
        variance_map.items(), key=lambda x: x[1], reverse=True
    )
    n_select = max(1, int(len(sorted_nodes) * fraction))
    return [node for node, _ in sorted_nodes[:n_select]]


def select_random_nodes(G: nx.Graph, fraction: float = 0.2) -> list[int]:
    """Select random fraction of nodes."""
    nodes = list(G.nodes())
    n_select = max(1, int(len(nodes) * fraction))
    return list(np.random.choice(nodes, size=n_select, replace=False))


def prepare_for_mutation(
    G: nx.Graph, node: int, strength: float = 0.8
) -> None:
    """
    Prepare node for mutation with destabilizer sequence.

    Grammar U4b: ZHIR needs recent destabilizer + prior IL.
    Sequence: [Coherence, Dissonance(high)]
    """
    # Prior stabilization (U4b requirement)
    Coherence()(G, node)  # Call operator directly without parameters

    # Recent destabilizer (U4b requirement)
    Dissonance()(G, node)  # Call operator directly without parameters


def apply_mutation_and_measure(
    G: nx.Graph, node: int
) -> dict[str, float]:
    """
    Apply ZHIR (Mutation) and measure success indicators.

    Success metrics:
    - Phase transition: |Δθ| > 0.2 radians
    - Coherence preservation: ΔC < 30% drop
    - ΔNFR elevation: indicates structural change

    Returns:
        Dict with success indicators
    """
    # Pre-mutation state
    theta_before = G.nodes[node]["theta"]
    C_before = compute_coherence(G)
    
    # Compute ΔNFR (writes to node attributes)
    default_compute_delta_nfr(G)
    delta_nfr_before = abs(G.nodes[node].get("delta_nfr", 0.0))

    # Apply mutation
    Mutation()(G, node)  # Call operator directly without parameters

    # Post-mutation state
    theta_after = G.nodes[node]["theta"]
    C_after = compute_coherence(G)
    
    # Recompute ΔNFR (writes to node attributes)
    default_compute_delta_nfr(G)
    delta_nfr_after = abs(G.nodes[node].get("delta_nfr", 0.0))

    # Compute metrics
    delta_theta = abs(theta_after - theta_before)
    delta_C = C_before - C_after
    delta_nfr_change = delta_nfr_after - delta_nfr_before

    # Success criteria
    phase_transition = delta_theta > 0.2
    coherence_preserved = delta_C < 0.30
    structural_change = delta_nfr_change > 0.1

    # Overall success: phase transition + coherence preservation
    success = phase_transition and coherence_preserved

    return {
        "delta_theta": float(delta_theta),
        "delta_C": float(delta_C),
        "delta_nfr_change": float(delta_nfr_change),
        "phase_transition": bool(phase_transition),
        "coherence_preserved": bool(coherence_preserved),
        "structural_change": bool(structural_change),
        "success": bool(success),
    }


def run_mutation_experiment(
    topology: str, n_nodes: int = 40, n_trials: int = 15, seed: int = 42
) -> dict:
    """
    Run mutation prediction experiment for one topology.

    Args:
        topology: 'ring', 'scale_free', 'ws', 'tree'
        n_nodes: Network size
        n_trials: Number of trials per strategy
        seed: Random seed

    Returns:
        Dict with experiment results
    """
    np.random.seed(seed)
    random.seed(seed)

    # Create topology using benchmark utils
    from benchmarks.benchmark_utils import (
        create_tnfr_topology,
        initialize_tnfr_nodes,
    )

    G = create_tnfr_topology(topology, n_nodes, seed)
    initialize_tnfr_nodes(G, nu_f=1.0, epi_range=(0.2, 0.8), seed=seed)

    # Compute K_φ variance map
    variance_map = compute_k_phi_1hop_variance(G)

    # Storage
    targeted_results = []
    random_results = []

    for trial in range(n_trials):
        # Reset seed for reproducibility per trial
        trial_seed = seed + trial
        np.random.seed(trial_seed)
        random.seed(trial_seed)

        # K_φ-targeted strategy
        target_nodes = select_top_variance_nodes(variance_map, fraction=0.2)
        node_targeted = random.choice(target_nodes)

        G_targeted = G.copy()
        prepare_for_mutation(G_targeted, node_targeted, strength=0.8)
        result_targeted = apply_mutation_and_measure(G_targeted, node_targeted)
        result_targeted["node"] = int(node_targeted)
        result_targeted["variance"] = float(variance_map[node_targeted])
        targeted_results.append(result_targeted)

        # Random strategy (baseline)
        random_nodes = select_random_nodes(G, fraction=0.2)
        node_random = random.choice(random_nodes)

        G_random = G.copy()
        prepare_for_mutation(G_random, node_random, strength=0.8)
        result_random = apply_mutation_and_measure(G_random, node_random)
        result_random["node"] = int(node_random)
        result_random["variance"] = float(variance_map[node_random])
        random_results.append(result_random)

    # Aggregate statistics
    targeted_success_rate = np.mean(
        [r["success"] for r in targeted_results]
    )
    random_success_rate = np.mean([r["success"] for r in random_results])

    improvement = (
        (targeted_success_rate - random_success_rate) / random_success_rate
        if random_success_rate > 0
        else 0.0
    )

    return {
        "topology": topology,
        "n_nodes": n_nodes,
        "n_trials": n_trials,
        "targeted_success_rate": float(targeted_success_rate),
        "random_success_rate": float(random_success_rate),
        "improvement": float(improvement),
        "improvement_pct": float(improvement * 100),
        "targeted_results": targeted_results,
        "random_results": random_results,
        "variance_stats": {
            "mean": float(np.mean(list(variance_map.values()))),
            "std": float(np.std(list(variance_map.values()))),
            "max": float(np.max(list(variance_map.values()))),
        },
    }


def main():
    """Run mutation prediction optimization tests."""
    print("🧬 K_φ-Based Mutation Prediction Optimization")
    print("=" * 60)

    topologies = ["ring", "scale_free", "ws", "tree"]
    results_all = []

    for topology in topologies:
        print(f"\n🌀 Testing {topology.upper()}...")
        result = run_mutation_experiment(
            topology=topology, n_nodes=40, n_trials=15, seed=42
        )

        targeted_rate = result["targeted_success_rate"]
        random_rate = result["random_success_rate"]
        improvement = result["improvement_pct"]

        print(f"  K_φ-Targeted: {targeted_rate:.1%} success")
        print(f"  Random:       {random_rate:.1%} success")
        print(
            f"  Improvement:  {improvement:+.1f}% "
            f"{'✅' if improvement >= 20 else '❌'}"
        )

        results_all.append(result)

    # Overall statistics
    print("\n" + "=" * 60)
    print("📊 OVERALL MUTATION PREDICTION ANALYSIS:")
    print("=" * 60)

    mean_improvement = np.mean([r["improvement_pct"] for r in results_all])
    n_success = sum(1 for r in results_all if r["improvement_pct"] >= 20)

    print(f"Mean improvement: {mean_improvement:+.1f}%")
    print(
        f"Topologies achieving ≥20%: {n_success}/{len(results_all)} "
        f"({n_success/len(results_all):.0%})"
    )

    if mean_improvement >= 20:
        print("\n✅ TASK 4 SUCCESS: K_φ variance predicts optimal ZHIR targets")
        print(f"   Mean improvement: {mean_improvement:+.1f}% (target: ≥20%)")
    else:
        print("\n❌ TASK 4 INCOMPLETE: Improvement below target")
        print(f"   Mean improvement: {mean_improvement:+.1f}% (target: ≥20%)")

    # Save results
    output_dir = Path("benchmarks/results")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "mutation_prediction_test.jsonl"

    with open(output_file, "w") as f:
        for result in results_all:
            f.write(json.dumps(result) + "\n")

    print(f"\n💾 Results saved to: {output_file.resolve()}")


if __name__ == "__main__":
    main()
