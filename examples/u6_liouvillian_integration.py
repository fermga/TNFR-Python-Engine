"""Example: U6 Temporal Ordering with Liouvillian Relaxation Time

This example demonstrates the complete workflow for U6 temporal ordering
validation using Liouvillian slow-mode relaxation time computation.

Key Steps:
1. Construct a TNFR graph with Lindblad dynamics
2. Compute Liouvillian spectrum from Hamiltonian and collapse operators
3. Store spectrum in graph metadata
4. Apply destabilizers (OZ) with U6 spacing validation
5. Measure observed relaxation time with Liouvillian-based τ_relax

This showcases the dual-path estimation:
- Preferred: Liouvillian slow mode (1/|Re(λ_slow)|)
- Fallback: Spectral topological proxy (k_top/νf)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import networkx as nx

# TNFR imports
from tnfr.mathematics.generators import build_lindblad_delta_nfr
from tnfr.mathematics.liouville import (
    compute_liouvillian_spectrum,
    store_liouvillian_spectrum,
    get_slow_relaxation_mode,
)
from tnfr.operators.metrics_u6 import measure_tau_relax_observed


def create_tnfr_graph_with_liouvillian():
    """Create a TNFR graph with Liouvillian dynamics and spectrum."""
    print("Step 1: Creating TNFR graph with Lindblad dynamics")
    print("-" * 60)

    # Create graph structure
    G = nx.karate_club_graph()

    # Initialize node attributes (simplified)
    for node in G.nodes():
        G.nodes[node]["EPI"] = 0.5 + 0.1 * np.random.randn()
        G.nodes[node]["nu_f"] = 1.0 + 0.2 * np.random.randn()
        G.nodes[node]["DELTA_NFR"] = 0.1 * np.random.randn()
        G.nodes[node]["phase"] = 2 * np.pi * np.random.rand()

    print(f"  Created graph with {G.number_of_nodes()} nodes")

    # Define Hamiltonian (coherent evolution)
    # For a 3-level system as example
    dim = 3
    H = np.array(
        [[1.0, 0.2, 0.0], [0.2, 0.0, 0.1], [0.0, 0.1, -1.0]], dtype=complex
    )

    # Define collapse operators (dissipation channels)
    L1 = np.array([[0.0, 0.5, 0.0], [0.0, 0.0, 0.3], [0.0, 0.0, 0.0]], dtype=complex)
    L2 = np.array([[0.0, 0.0, 0.2], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=complex)

    print(f"\n  Hamiltonian (dim={dim}):")
    print(f"    {H}")
    print(f"\n  Collapse operators: {len([L1, L2])} channels")

    # Build Liouvillian generator
    print("\nStep 2: Building Liouvillian generator (Lindblad form)")
    print("-" * 60)

    liouvillian = build_lindblad_delta_nfr(
        hamiltonian=H,
        collapse_operators=[L1, L2],
        nu_f=1.2,  # structural frequency scaling
        ensure_contractive=True,
    )

    print(f"  Liouvillian shape: {liouvillian.shape}")
    print(f"  Dimension: {dim}² × {dim}² = {dim**2} × {dim**2}")

    # Compute spectrum
    print("\nStep 3: Computing Liouvillian spectrum")
    print("-" * 60)

    eigenvalues = compute_liouvillian_spectrum(
        liouvillian, sort=True, validate_contractivity=True
    )

    print(f"  Computed {len(eigenvalues)} eigenvalues")
    print(f"  All Re(λ) ≤ 0: {np.all(eigenvalues.real <= 1e-9)}")

    # Find slow relaxation mode
    slow_mode = get_slow_relaxation_mode(eigenvalues)
    if slow_mode is not None:
        tau_liouv = 1.0 / abs(slow_mode.real)
        print(f"\n  Slow relaxation mode:")
        print(f"    λ_slow = {slow_mode}")
        print(f"    τ_relax = 1/|Re(λ)| = {tau_liouv:.3f}")
    else:
        print("\n  Warning: No slow mode found (unusual)")

    # Store in graph metadata
    print("\nStep 4: Storing spectrum in graph metadata")
    print("-" * 60)

    store_liouvillian_spectrum(G, eigenvalues, key="LIOUVILLIAN_EIGS")
    print(f"  ✓ Stored {len(eigenvalues)} eigenvalues under key 'LIOUVILLIAN_EIGS'")

    return G


def apply_destabilizers_with_u6_validation(G):
    """Apply destabilizer operators with U6 temporal ordering."""
    print("\n" + "=" * 60)
    print("Step 5: Applying destabilizers with U6 validation")
    print("=" * 60)

    # Select target nodes for destabilization
    target_nodes = [0, 5, 10]  # Example nodes

    for node in target_nodes:
        print(f"\nNode {node}:")

        # Measure relaxation time with Liouvillian integration
        result = measure_tau_relax_observed(G, node)

        print(f"  νf = {result['vf']:.3f}")
        print(f"  ΔNFR initial = {result['dnfr_initial']:.3f}")
        print(f"  k_top (spectral) = {result['k_top']:.3f}")

        # Show dual-path estimation
        tau_liouv = result["estimated_tau_relax_liouvillian"]
        tau_spectral = result["estimated_tau_relax_spectral"]
        tau_final = result["estimated_tau_relax"]

        print(f"\n  Relaxation time estimates:")
        print(f"    Liouvillian slow-mode: τ = {tau_liouv:.3f}" if tau_liouv else "    Liouvillian: None (fallback)")
        print(f"    Spectral topological:  τ = {tau_spectral:.3f}")
        print(f"    Final estimate:        τ = {tau_final:.3f}")

        if tau_liouv is not None:
            print(f"    ✓ Using Liouvillian-based estimate (preferred)")
        else:
            print(f"    ⚠ Falling back to spectral estimate")

        # U6 spacing recommendation
        print(f"\n  U6 Recommendation: Δt ≥ {tau_final:.3f} between destabilizers")


def demonstrate_u6_workflow():
    """Complete demonstration of U6 temporal ordering workflow."""
    print("\n")
    print("=" * 60)
    print("U6 Temporal Ordering with Liouvillian Integration")
    print("=" * 60)
    print()

    # Create graph with Liouvillian spectrum
    G = create_tnfr_graph_with_liouvillian()

    # Apply destabilizers with U6 validation
    apply_destabilizers_with_u6_validation(G)

    print("\n" + "=" * 60)
    print("Summary: Liouvillian Integration Benefits")
    print("=" * 60)
    print("""
Key Advantages:

1. **Physical Accuracy**: Liouvillian slow mode directly captures the
   system's intrinsic relaxation timescale from first principles.

2. **Dual-Path Robustness**: Spectral topological fallback ensures
   τ_relax is always available, even without full Liouvillian data.

3. **U6 Compliance**: Relaxation time informs minimum spacing between
   destabilizers, preventing nonlinear accumulation (α > 1.1).

4. **Operator-Specific Tuning**: Future enhancement can map glyph types
   to k_op factors (OZ, ZHIR, VAL) for refined estimates.

5. **Telemetry Transparency**: All three values (Liouvillian, spectral,
   final) exposed in metrics for downstream analysis.
    """)


if __name__ == "__main__":
    demonstrate_u6_workflow()
