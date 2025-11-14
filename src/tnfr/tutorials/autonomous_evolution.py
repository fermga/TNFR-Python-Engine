"""Tutorial: Autonomous Evolution with TNFR Adaptive Systems.

This tutorial demonstrates the complete adaptive dynamics system integrating
feedback loops, adaptive sequence selection, homeostasis, learning, and
metabolism into autonomous structural evolution.

**Learning Objectives:**

1. Understand feedback loops and homeostatic regulation
2. Use adaptive sequence selection for optimal trajectories
3. Integrate all components into autonomous evolution
4. Monitor and measure adaptive dynamics

**Prerequisites:**

- Basic TNFR concepts (NFR, operators, coherence)
- Understanding of structural metrics (C(t), Si, ΔNFR)

All examples are designed as narrated console walkthroughs, so they emit
human-readable print statements instead of structured logs when run.
"""

from __future__ import annotations

import networkx as nx

from tnfr.structural import create_nfr
from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_VF
from tnfr.dynamics.feedback import StructuralFeedbackLoop
from tnfr.dynamics.adaptive_sequences import AdaptiveSequenceSelector
from tnfr.dynamics.homeostasis import StructuralHomeostasis
from tnfr.sdk.adaptive_system import TNFRAdaptiveSystem


def example_1_feedback_loop():
    """Example 1: Basic feedback loop regulation.

    Demonstrates how a feedback loop measures coherence and selects
    appropriate operators to maintain target coherence.
    """
    print("=" * 70)
    print("Example 1: Basic Feedback Loop Regulation")
    print("=" * 70)

    # Create a simple node
    G, node = create_nfr("feedback_node", epi=0.5, vf=1.0)

    # Initialize feedback loop with target coherence
    loop = StructuralFeedbackLoop(
        G,
        node,
        target_coherence=0.7,  # Target C(t)
        tau_adaptive=0.1,  # Initial bifurcation threshold
        learning_rate=0.05,  # Threshold adaptation rate
    )

    print("\nInitial state:")
    print(f"  EPI: {get_attr(G.nodes[node], ALIAS_EPI, 0.0):.3f}")
    print(f"  νf: {get_attr(G.nodes[node], ALIAS_VF, 1.0):.3f}")
    print(f"  ΔNFR: {get_attr(G.nodes[node], ALIAS_DNFR, 0.0):.3f}")

    # Run feedback regulation
    print("\nRunning feedback regulation...")
    for step in range(5):
        coherence_before = loop._compute_local_coherence()
        operator = loop.regulate()
        print(f"\n  Step {step + 1}:")
        print(f"    Coherence: {coherence_before:.3f}")
        print(f"    Selected operator: {operator}")

    print(
        "\nFeedback loop maintains coherence through adaptive operator "
        "selection."
    )


def example_2_adaptive_sequences():
    """Example 2: Adaptive sequence selection.

    Demonstrates how the system learns which operator sequences work best
    for different goals and adapts its selection over time.
    """
    print("\n" + "=" * 70)
    print("Example 2: Adaptive Sequence Selection")
    print("=" * 70)

    G, node = create_nfr("learning_node", epi=0.5, vf=1.0)

    selector = AdaptiveSequenceSelector(G, node)

    print("\nAvailable canonical sequences:")
    for name, sequence in selector.sequences.items():
        print(f"  {name}: {' → '.join(sequence)}")

    # Select sequences for different goals
    print("\n\nSequence selection for different goals:")

    goals = ["stability", "growth", "adaptation"]
    for goal in goals:
        context = {"goal": goal, "urgency": 0.5}
        sequence = selector.select_sequence(context)
        print(f"\n  Goal: {goal}")
        print(f"  Selected: {' → '.join(sequence)}")

    # Simulate learning through performance recording
    print("\n\nSimulating learning through performance feedback:")
    selector.record_performance("basic_activation", 0.85)
    selector.record_performance("deep_learning", 0.92)
    selector.record_performance("basic_activation", 0.88)

    print("  Recorded performances:")
    for name, perfs in selector.performance.items():
        if perfs:
            avg = sum(perfs) / len(perfs)
            print(f"    {name}: avg = {avg:.3f} ({len(perfs)} samples)")

    print(
        "\nAdaptive selection learns from experience to optimize "
        "trajectories."
    )


def example_3_homeostasis():
    """Example 3: Homeostatic regulation.

    Demonstrates how homeostasis maintains parameters within healthy ranges
    through automatic corrective operators.
    """
    print("\n" + "=" * 70)
    print("Example 3: Homeostatic Regulation")
    print("=" * 70)

    G, node = create_nfr("homeostatic_node", epi=0.5, vf=1.0)

    homeostasis = StructuralHomeostasis(G, node)

    print("\nHomeostatic target ranges:")
    print(f"  EPI: {homeostasis.epi_range}")
    print(f"  νf: {homeostasis.vf_range}")
    print(f"  ΔNFR: {homeostasis.dnfr_range}")

    print("\nCurrent state:")
    print(f"  EPI: {get_attr(G.nodes[node], ALIAS_EPI, 0.0):.3f}")
    print(f"  νf: {get_attr(G.nodes[node], ALIAS_VF, 1.0):.3f}")
    print(f"  ΔNFR: {get_attr(G.nodes[node], ALIAS_DNFR, 0.0):.3f}")

    print("\nApplying homeostatic regulation...")
    homeostasis.maintain_equilibrium()

    print("\nHomeostasis automatically corrects out-of-range parameters.")


def example_4_integrated_system():
    """Example 4: Complete integrated adaptive system.

    Demonstrates the full TNFRAdaptiveSystem combining all components into
    autonomous evolution cycles.
    """
    print("\n" + "=" * 70)
    print("Example 4: Complete Integrated Adaptive System")
    print("=" * 70)

    G, node = create_nfr("adaptive_node", epi=0.5, vf=1.0)

    # Initialize complete adaptive system
    system = TNFRAdaptiveSystem(G, node)

    print("\nIntegrated components:")
    print("  ✓ Feedback loop")
    print("  ✓ Adaptive sequence selector")
    print("  ✓ Homeostasis")
    print("  ✓ Learning system")
    print("  ✓ Structural metabolism")

    print("\nInitial state:")
    print(f"  EPI: {get_attr(G.nodes[node], ALIAS_EPI, 0.0):.3f}")
    print(f"  νf: {get_attr(G.nodes[node], ALIAS_VF, 1.0):.3f}")
    print(f"  ΔNFR: {get_attr(G.nodes[node], ALIAS_DNFR, 0.0):.3f}")

    # Run autonomous evolution
    print("\nRunning autonomous evolution (10 cycles)...")
    system.autonomous_evolution(num_cycles=10)

    print("\nFinal state:")
    print(f"  EPI: {get_attr(G.nodes[node], ALIAS_EPI, 0.0):.3f}")
    print(f"  νf: {get_attr(G.nodes[node], ALIAS_VF, 1.0):.3f}")
    print(f"  ΔNFR: {get_attr(G.nodes[node], ALIAS_DNFR, 0.0):.3f}")

    print("\nThe system self-regulates through integrated adaptive dynamics.")


def example_5_multi_node_network():
    """Example 5: Adaptive dynamics in multi-node networks.

    Demonstrates how adaptive systems work in networks with multiple
    interacting nodes.
    """
    print("\n" + "=" * 70)
    print("Example 5: Multi-Node Adaptive Network")
    print("=" * 70)

    # Create a small network
    G = nx.Graph()
    nodes = []
    for i in range(3):
        _, node = create_nfr(f"node_{i}", graph=G, epi=0.5, vf=1.0)
        nodes.append(node)

    # Connect nodes
    G.add_edge(nodes[0], nodes[1])
    G.add_edge(nodes[1], nodes[2])

    print(
        f"\nCreated network with {len(nodes)} nodes "
        f"and {G.number_of_edges()} edges"
    )

    # Create adaptive system for each node
    systems = [TNFRAdaptiveSystem(G, node) for node in nodes]

    print("\nRunning synchronized autonomous evolution...")
    for cycle in range(5):
        print(f"\n  Cycle {cycle + 1}:")
        for i, system in enumerate(systems):
            # Each node evolves autonomously
            system.autonomous_evolution(num_cycles=2)

            stress = system._measure_stress()
            print(f"    Node {i}: stress = {stress:.3f}")

    print("\nMultiple adaptive systems can coevolve in networked structures.")


def example_6_stress_response():
    """Example 6: Metabolic stress response.

    Demonstrates how the system measures and responds to structural stress
    through metabolic adaptation.
    """
    print("\n" + "=" * 70)
    print("Example 6: Metabolic Stress Response")
    print("=" * 70)

    G, node = create_nfr("stress_node", epi=0.5, vf=1.0)
    system = TNFRAdaptiveSystem(G, node)

    print("\nStress measurement from ΔNFR:")
    print("  ΔNFR = 0.0 → stress = 0.0 (relaxed)")
    print("  ΔNFR = 0.1 → stress = 0.5 (moderate)")
    print("  ΔNFR = 0.2 → stress = 1.0 (maximum)")

    print("\n\nSimulating stress response:")
    from tnfr.alias import set_attr

    stress_levels = [0.0, 0.05, 0.1, 0.15, 0.2]
    for dnfr_val in stress_levels:
        set_attr(G.nodes[node], ALIAS_DNFR, dnfr_val)
        stress = system._measure_stress()
        print(f"  ΔNFR = {dnfr_val:.2f} → stress = {stress:.2f}")

    print("\nThe system adapts metabolically to structural stress levels.")


def run_all_examples():
    """Run all tutorial examples in sequence."""
    examples = [
        example_1_feedback_loop,
        example_2_adaptive_sequences,
        example_3_homeostasis,
        example_4_integrated_system,
        example_5_multi_node_network,
        example_6_stress_response,
    ]

    for example in examples:
        example()

    print("\n" + "=" * 70)
    print("Tutorial Complete: Autonomous Evolution with TNFR")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  1. Feedback loops enable autonomous coherence regulation")
    print("  2. Adaptive selection learns optimal operator sequences")
    print("  3. Homeostasis maintains healthy parameter ranges")
    print("  4. Integration creates self-regulating adaptive systems")
    print("  5. Networks support multi-node coevolution")
    print("  6. Metabolic response adapts to structural stress")
    print("\nNext steps:")
    print("  - Explore custom feedback strategies")
    print("  - Define domain-specific sequences")
    print("  - Tune homeostatic ranges for your application")
    print("  - Integrate with existing TNFR workflows")


if __name__ == "__main__":
    run_all_examples()
