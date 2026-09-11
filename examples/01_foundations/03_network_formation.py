"""03 - Network Formation: Building Connections Step by Step

PHYSICS: Demonstrates how individual nodes create network-level coherence.
LEARNING: Understand network topology effects on structural stability.

This shows the emergence of collective behavior from individual dynamics.
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Configure font for better Unicode support
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "sans-serif"]
# Suppress missing-glyph warnings
import warnings

warnings.filterwarnings("ignore", "Glyph .* missing from font.*")


def simple_coherence_measure(G):
    """Compute a simple coherence measure based on network structure."""
    if G.number_of_nodes() == 0:
        return 0.0

    # Simple coherence: connectivity + phase alignment + frequency harmony
    connectivity = G.number_of_edges() / (
        G.number_of_nodes() * (G.number_of_nodes() - 1) / 2
    )

    # Phase coherence: how aligned are the phases?
    phases = [G.nodes[n].get("theta", 0) for n in G.nodes()]
    if len(phases) > 1:
        # Convert to unit vectors and compute alignment
        phase_vectors = np.array([[np.cos(p), np.sin(p)] for p in phases])
        mean_vector = np.mean(phase_vectors, axis=0)
        phase_coherence = np.linalg.norm(mean_vector)
    else:
        phase_coherence = 1.0

    # Frequency harmony: how similar are the frequencies?
    frequencies = [G.nodes[n].get("nf", 1.0) for n in G.nodes()]
    if len(frequencies) > 1:
        freq_std = np.std(frequencies)
        freq_harmony = 1.0 / (1.0 + freq_std)
    else:
        freq_harmony = 1.0

    # Combine measures
    total_coherence = (connectivity * phase_coherence * freq_harmony) ** (1 / 3)
    return total_coherence


def network_formation_demo():
    """Demonstrate step-by-step network formation and coherence evolution."""

    print("=" * 75)
    print(" " * 20 + "🕸️ NETWORK FORMATION DYNAMICS 🕸️")
    print("=" * 75)
    print()
    print("Building networks step by step and watching coherence emerge...")
    print("PHYSICS: Individual nodes → structural connections → collective coherence")
    print("INSIGHT: Network topology determines information flow and stability")
    print()

    # Create different network formation strategies
    n_nodes = 8

    def test_formation_strategy(strategy_name, formation_function, description):
        """Test a network formation strategy."""

        print(f"🏗️ {strategy_name}")
        print(f"   Strategy: {description}")

        # Start with isolated nodes
        G = nx.Graph()
        G.add_nodes_from(range(n_nodes))

        # Initialize node properties randomly but consistently
        np.random.seed(42)  # Reproducible
        for node in G.nodes():
            G.nodes[node]["theta"] = np.random.uniform(0, 2 * np.pi)  # Random phases
            G.nodes[node]["nu_f"] = (
                1.0 + 0.2 * np.random.normal()
            )  # Frequency variation

        print(f"   Initial coherence: {simple_coherence_measure(G):.3f}")

        # Apply formation strategy
        formation_function(G)

        final_coherence = simple_coherence_measure(G)
        print(f"   Final coherence: {final_coherence:.3f}")
        print(f"   Edges created: {G.number_of_edges()}")
        print(
            f"   Connectivity: {G.number_of_edges()}/{n_nodes*(n_nodes-1)//2} = {2*G.number_of_edges()/(n_nodes*(n_nodes-1)):.2%}"
        )
        print()

        return G, final_coherence

    print("🔬 NETWORK FORMATION EXPERIMENTS")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print()

    results = {}

    # Strategy 1: Random connections
    def random_formation(G):
        """Add random edges until desired density."""
        np.random.seed(42)
        nodes = list(G.nodes())
        target_edges = n_nodes  # Average degree = 2

        while G.number_of_edges() < target_edges:
            i, j = np.random.choice(nodes, 2, replace=False)
            if not G.has_edge(i, j):
                G.add_edge(i, j)

    G1, coh1 = test_formation_strategy(
        "Random Formation",
        random_formation,
        "Connect nodes randomly until target density",
    )
    results["Random"] = coh1

    # Strategy 2: Preferential attachment (rich get richer)
    def preferential_formation(G):
        """Add edges preferentially to high-degree nodes."""
        nodes = list(G.nodes())
        edges_to_add = n_nodes

        for _ in range(edges_to_add):
            # Choose first node randomly
            i = np.random.choice(nodes)

            # Choose second node preferentially by degree
            degrees = dict(G.degree())
            if sum(degrees.values()) == 0:
                # No edges yet, connect randomly
                j = np.random.choice([n for n in nodes if n != i])
            else:
                # Preferential attachment
                weights = [
                    degrees[n] + 1 for n in nodes if n != i and not G.has_edge(i, n)
                ]
                if weights:
                    candidates = [n for n in nodes if n != i and not G.has_edge(i, n)]
                    weights = np.array(weights, dtype=float)
                    weights /= weights.sum()
                    j = np.random.choice(candidates, p=weights)
                else:
                    continue  # Skip if no valid candidates

            G.add_edge(i, j)

    G2, coh2 = test_formation_strategy(
        "Preferential Attachment",
        preferential_formation,
        "Rich nodes get richer (scale-free network)",
    )
    results["Preferential"] = coh2

    # Strategy 3: Phase-guided formation (TNFR-inspired)
    def phase_guided_formation(G):
        """Connect nodes with similar phases preferentially."""
        nodes = list(G.nodes())
        target_edges = n_nodes

        for _ in range(target_edges):
            # Find the pair with smallest phase difference
            best_pair = None
            min_phase_diff = float("inf")

            for i in nodes:
                for j in nodes:
                    if i < j and not G.has_edge(i, j):
                        phase_diff = abs(G.nodes[i]["theta"] - G.nodes[j]["theta"])
                        # Handle circular distance
                        phase_diff = min(phase_diff, 2 * np.pi - phase_diff)

                        if phase_diff < min_phase_diff:
                            min_phase_diff = phase_diff
                            best_pair = (i, j)

            if best_pair:
                G.add_edge(*best_pair)

    G3, coh3 = test_formation_strategy(
        "Phase-Guided Formation",
        phase_guided_formation,
        "Connect nodes with similar phases (TNFR coupling)",
    )
    results["Phase-Guided"] = coh3

    # Strategy 4: Ring formation (structured)
    def ring_formation(G):
        """Form a ring structure for guaranteed connectivity."""
        nodes = list(G.nodes())
        # Create ring
        for i in range(len(nodes)):
            G.add_edge(nodes[i], nodes[(i + 1) % len(nodes)])

        # Add a few random shortcuts
        for _ in range(2):
            i, j = np.random.choice(nodes, 2, replace=False)
            if not G.has_edge(i, j):
                G.add_edge(i, j)

    G4, coh4 = test_formation_strategy(
        "Ring + Shortcuts", ring_formation, "Ring topology with random shortcuts"
    )
    results["Ring"] = coh4

    print("📊 FORMATION STRATEGY COMPARISON")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print()

    # Rank strategies by final coherence
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

    print("🏆 COHERENCE RANKINGS:")
    for i, (strategy, coherence) in enumerate(sorted_results, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"   {medal} {i}. {strategy:<20}: {coherence:.3f}")

    print()
    print("🎯 NETWORK FORMATION INSIGHTS:")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print()
    print("🏗️ FORMATION PRINCIPLES:")
    print("   • Structure emerges from connection strategy")
    print("   • Random → low coherence, high variance")
    print("   • Preferential → hubs, but potential fragility")
    print("   • Phase-guided → natural TNFR coupling")
    print("   • Structured → guaranteed connectivity")
    print()
    print("⚛️ TNFR PHYSICS:")
    print("   • Phase similarity enables stable coupling")
    print("   • Frequency harmony reduces structural stress")
    print("   • Network topology determines information flow")
    print("   • Coherence emerges from compatible connections")
    print()
    print("📊 FORMATION STRATEGIES:")
    print("   Random: Simple but unpredictable")
    print("   Preferential: Creates hubs and shortcuts")
    print("   Phase-guided: Mimics natural TNFR coupling")
    print("   Structured: Guarantees basic connectivity")
    print()
    print("🧠 REAL-WORLD APPLICATIONS:")
    print("   • Social networks form through similarity")
    print("   • Neural networks wire by activity patterns")
    print("   • Economic networks follow preferential attachment")
    print("   • Molecular networks use energy compatibility")
    print()

    # Show topology visualization
    if not os.path.exists("output"):
        os.makedirs("output")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    networks = [G1, G2, G3, G4]
    names = ["Random", "Preferential", "Phase-Guided", "Ring + Shortcuts"]
    coherences = [coh1, coh2, coh3, coh4]

    for i, (G, name, coh) in enumerate(zip(networks, names, coherences)):
        ax = axes[i // 2, i % 2]

        # Color nodes by phase
        phases = [G.nodes[n]["theta"] for n in G.nodes()]

        pos = nx.spring_layout(G, seed=42)
        nx.draw(
            G,
            pos,
            ax=ax,
            node_color=phases,
            node_size=300,
            cmap="hsv",
            with_labels=True,
            edge_color="gray",
            alpha=0.7,
        )

        ax.set_title(f"{name}\nCoherence: {coh:.3f}", fontsize=12)

    plt.suptitle("🕸️ Network Formation Strategies", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig("output/network_formation_comparison.png", dpi=300, bbox_inches="tight")
    print(
        "📊 Network topology visualization saved to: output/network_formation_comparison.png"
    )

    best_strategy, best_coherence = sorted_results[0]
    print(
        f"🌟 Best formation strategy: {best_strategy} (coherence: {best_coherence:.3f})"
    )
    print("🕸️ Network structure determines collective coherence!")


if __name__ == "__main__":
    network_formation_demo()
