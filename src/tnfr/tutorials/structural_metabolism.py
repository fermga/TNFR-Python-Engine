"""Tutorial: T'HOL Structural Metabolism and Bifurcation

This tutorial demonstrates the canonical implementation of T'HOL
(Self-Organization) as structural metabolism, including:

1. Bifurcation dynamics (∂²EPI/∂t² > τ)
2. Metabolic cycles (EN → THOL → IL)
3. Adaptive metabolism
4. Emergence metrics

Theory
------
T'HOL is not just self-organization - it's **structural metabolism**:

> "T'HOL no reacciona: reorganiza. No adapta: reinventa. T'HOL es el
> metabolismo estructural: permite que una forma se reorganice sin romperse."

**Key Characteristics:**
- **Bifurcation nodal**: When acceleration exceeds threshold, spawns sub-EPIs
- **Autonomous reorganization**: No external instruction required
- **Vibrational metabolism**: Digests external experience into internal
    structure
- **Emergence engine**: Creates complexity and novelty

Examples
--------
This tutorial intentionally prints narrative walkthroughs to stdout so
readers can follow each metabolic phase when running the module directly.
These messages provide step-by-step context and are not meant to be routed
through the shared logging infrastructure.
"""

from __future__ import annotations

from tnfr.constants import DNFR_PRIMARY, EPI_PRIMARY, VF_PRIMARY
from tnfr.dynamics.metabolism import (
    StructuralMetabolism,
)
from tnfr.metrics.emergence import (
    compute_bifurcation_rate,
    compute_emergence_index,
    compute_metabolic_efficiency,
    compute_structural_complexity,
)
from tnfr.operators.definitions import SelfOrganization
from tnfr.structural import create_nfr


def example_1_basic_bifurcation():
    """Example 1: Basic bifurcation when acceleration exceeds threshold."""
    print("\n=== Example 1: Basic Bifurcation ===\n")

    # Create a node with sufficient EPI
    G, node = create_nfr("cell", epi=0.5, vf=1.0)

    # Set positive ΔNFR (required for T'HOL)
    G.nodes[node][DNFR_PRIMARY] = 0.2

    # Provide EPI history showing strong acceleration
    # d²EPI = abs(0.7 - 2*0.5 + 0.3) = 0.1
    G.nodes[node]["epi_history"] = [0.3, 0.5, 0.7]

    print("Initial state:")
    print(f"  EPI: {G.nodes[node][EPI_PRIMARY]:.3f}")
    print(f"  EPI history: {G.nodes[node]['epi_history']}")
    print("  Acceleration: 0.1")

    # Apply T'HOL with low threshold to trigger bifurcation
    SelfOrganization()(G, node, tau=0.08)

    # Check for bifurcation
    sub_epis = G.nodes[node].get("sub_epis", [])
    print("\nAfter T'HOL:")
    print(f"  EPI: {G.nodes[node][EPI_PRIMARY]:.3f}")
    print(f"  Bifurcations: {len(sub_epis)}")

    if sub_epis:
        print("  Sub-EPI details:")
        for i, sub in enumerate(sub_epis):
            print(
                f"    [{i}] epi={sub['epi']:.3f}, d2_epi={sub['d2_epi']:.3f}"
            )


def example_2_metabolic_cycle():
    """Example 2: Complete metabolic cycle (EN → THOL → IL)."""
    print("\n=== Example 2: Metabolic Cycle ===\n")

    # Create a node with a neighbor (for Reception)
    G, node = create_nfr("neuron", epi=0.4, vf=1.0)
    G.add_node("neighbor", **{EPI_PRIMARY: 0.6, VF_PRIMARY: 1.0})
    G.add_edge(node, "neighbor")

    # Set positive ΔNFR and history for bifurcation
    G.nodes[node][DNFR_PRIMARY] = 0.2
    G.nodes[node]["epi_history"] = [0.2, 0.35, 0.5]

    print("Initial state:")
    print(f"  EPI: {G.nodes[node][EPI_PRIMARY]:.3f}")
    print(f"  Neighbors: {list(G.neighbors(node))}")

    # Use StructuralMetabolism for complete cycle
    metabolism = StructuralMetabolism(G, node)
    metabolism.digest(tau=0.08)

    print("\nAfter metabolic cycle:")
    # Get final EPI value safely using unified function
    from tnfr.alias import get_attr
    from tnfr.constants.aliases import ALIAS_EPI

    final_epi = float(get_attr(G.nodes[node], ALIAS_EPI, 0.0))
    print(f"  EPI: {final_epi:.3f}")
    # Get glyph history safely
    glyph_hist = G.nodes[node].get("glyph_history", [])
    if glyph_hist and isinstance(glyph_hist, list):
        print(f"  Glyph history: {glyph_hist[-5:]}")  # Last 5 glyphs


def example_3_adaptive_metabolism():
    """Example 3: Adaptive metabolism responds to stress levels."""
    print("\n=== Example 3: Adaptive Metabolism ===\n")

    # Import unified functions
    from tnfr.alias import get_attr
    from tnfr.constants.aliases import ALIAS_EPI

    # High stress scenario
    G_stress, node_stress = create_nfr("organism_stressed", epi=0.6, vf=1.0)
    G_stress.nodes[node_stress][DNFR_PRIMARY] = 0.3
    G_stress.nodes[node_stress]["epi_history"] = [0.4, 0.5, 0.7]

    print("High stress (>= 0.5):")
    initial_epi_stress = float(
        get_attr(G_stress.nodes[node_stress], ALIAS_EPI, 0.0)
    )
    print(f"  Initial EPI: {initial_epi_stress:.3f}")

    metabolism_stress = StructuralMetabolism(G_stress, node_stress)
    metabolism_stress.adaptive_metabolism(stress_level=0.7)

    final_epi_stress = float(
        get_attr(G_stress.nodes[node_stress], ALIAS_EPI, 0.0)
    )
    print(f"  After adaptive metabolism: EPI={final_epi_stress:.3f}")

    # Low stress scenario
    G_calm, node_calm = create_nfr("organism_calm", epi=0.5, vf=1.0)
    G_calm.nodes[node_calm][DNFR_PRIMARY] = 0.1
    G_calm.nodes[node_calm]["epi_history"] = [0.45, 0.48, 0.52]

    print("\nLow stress (< 0.5):")
    initial_epi_calm = float(get_attr(G_calm.nodes[node_calm], ALIAS_EPI, 0.0))
    print(f"  Initial EPI: {initial_epi_calm:.3f}")

    metabolism_calm = StructuralMetabolism(G_calm, node_calm)
    metabolism_calm.adaptive_metabolism(stress_level=0.3)

    final_epi_calm = float(get_attr(G_calm.nodes[node_calm], ALIAS_EPI, 0.0))
    print(f"  After adaptive metabolism: EPI={final_epi_calm:.3f}")


def example_4_cascading_reorganization():
    """Example 4: Cascading T'HOL for multi-scale reorganization."""
    print("\n=== Example 4: Cascading Reorganization ===\n")

    # Import unified functions
    from tnfr.alias import get_attr
    from tnfr.constants.aliases import ALIAS_EPI

    G, node = create_nfr("system", epi=0.7, vf=1.2)
    G.nodes[node][DNFR_PRIMARY] = 0.25
    G.nodes[node]["epi_history"] = [0.5, 0.6, 0.8]

    print("Initial state:")
    initial_epi = float(get_attr(G.nodes[node], ALIAS_EPI, 0.0))
    print(f"  EPI: {initial_epi:.3f}")

    # Cascading reorganization with 3 levels
    metabolism = StructuralMetabolism(G, node)
    metabolism.cascading_reorganization(depth=3)

    sub_epis = G.nodes[node].get("sub_epis", [])
    final_epi = float(get_attr(G.nodes[node], ALIAS_EPI, 0.0))
    print("\nAfter cascading reorganization (depth=3):")
    print(f"  EPI: {final_epi:.3f}")
    print(f"  Total bifurcations: {len(sub_epis)}")


def example_5_emergence_metrics():
    """Example 5: Measuring structural complexity and metabolic efficiency."""
    print("\n=== Example 5: Emergence Metrics ===\n")

    # Import unified functions
    from tnfr.alias import get_attr
    from tnfr.constants.aliases import ALIAS_EPI

    G, node = create_nfr("evolving_system", epi=0.6, vf=1.0)

    # Record initial EPI for efficiency calculation
    G.nodes[node]["epi_initial"] = 0.3

    # Simulate several T'HOL applications with bifurcations
    G.nodes[node][DNFR_PRIMARY] = 0.2
    G.nodes[node]["glyph_history"] = []

    for i in range(3):
        # Update history to show acceleration
        G.nodes[node]["epi_history"] = [
            0.3 + i * 0.1,
            0.4 + i * 0.1,
            0.6 + i * 0.1,
        ]
        SelfOrganization()(G, node, tau=0.08)

    print("After 3 T'HOL applications:")
    final_epi = float(get_attr(G.nodes[node], ALIAS_EPI, 0.0))
    print(f"  Final EPI: {final_epi:.3f}")

    # Compute emergence metrics
    complexity = compute_structural_complexity(G, node)
    rate = compute_bifurcation_rate(G, node, window=10)
    efficiency = compute_metabolic_efficiency(G, node)
    emergence = compute_emergence_index(G, node)

    print("\nEmergence Metrics:")
    print(f"  Structural complexity: {complexity}")
    print(f"  Bifurcation rate: {rate:.3f} per step")
    print(f"  Metabolic efficiency: {efficiency:.3f} EPI/THOL")
    print(f"  Emergence index: {emergence:.3f}")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("T'HOL STRUCTURAL METABOLISM TUTORIAL")
    print("=" * 70)

    example_1_basic_bifurcation()
    example_2_metabolic_cycle()
    example_3_adaptive_metabolism()
    example_4_cascading_reorganization()
    example_5_emergence_metrics()

    print("\n" + "=" * 70)
    print("Tutorial complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
