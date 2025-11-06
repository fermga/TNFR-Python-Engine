"""Example: Dynamic Canonical Limits in TNFR

This example demonstrates the theoretical proposition that canonical limits
(EPI_MAX, VF_MAX) should be dynamic rather than static, adapting based on
network coherence to preserve TNFR's self-organizing principles.

Theoretical Context:
-------------------
Fixed limits may contradict TNFR's core principles:
1. **Operational fractality** - patterns should scale without artificial bounds
2. **Self-organization** - system should find its own natural limits  
3. **Coherence emergence** - stability arises from resonance, not external constraints

This example compares static vs dynamic limits across different network states.
"""

import networkx as nx
from tnfr.dynamics.dynamic_limits import (
    DynamicLimitsConfig,
    compute_dynamic_limits,
)


def create_coherent_network(n_nodes: int = 10) -> nx.Graph:
    """Create a highly coherent TNFR network.
    
    Characteristics:
    - High phase synchronization (all nodes near same phase)
    - High sense index (stable reorganization patterns)
    - Low gradients (low ΔNFR and dEPI)
    - Result: High coherence C(t) and Kuramoto order parameter R
    """
    G = nx.Graph()
    
    for i in range(n_nodes):
        G.add_node(i, **{
            "νf": 1.0 + i * 0.05,  # Slightly varying frequencies
            "theta": 0.0 + i * 0.05,  # Nearly aligned phases
            "EPI": 0.5,
            "Si": 0.85 + i * 0.01,  # High sense index
            "ΔNFR": 0.01,  # Very low reorganization gradient
            "dEPI_dt": 0.01,  # Very low velocity
        })
    
    return G


def create_chaotic_network(n_nodes: int = 10) -> nx.Graph:
    """Create a chaotic (low coherence) TNFR network.
    
    Characteristics:
    - Low phase synchronization (dispersed phases)
    - Low sense index (unstable reorganization)
    - High gradients (high ΔNFR and dEPI)
    - Result: Low coherence C(t) and Kuramoto order parameter R
    """
    G = nx.Graph()
    
    for i in range(n_nodes):
        G.add_node(i, **{
            "νf": 2.0 + i * 0.3,  # Highly varying frequencies
            "theta": i * 0.7,  # Dispersed phases
            "EPI": 0.5,
            "Si": 0.2 + i * 0.05,  # Low sense index
            "ΔNFR": 0.6 + i * 0.05,  # High reorganization gradient
            "dEPI_dt": 0.5 + i * 0.02,  # High velocity
        })
    
    return G


def create_transitional_network(n_nodes: int = 10) -> nx.Graph:
    """Create a transitional TNFR network.
    
    Characteristics:
    - Moderate phase synchronization
    - Moderate sense index
    - Moderate gradients
    - Result: Medium coherence and Kuramoto order
    """
    G = nx.Graph()
    
    for i in range(n_nodes):
        G.add_node(i, **{
            "νf": 1.5 + i * 0.1,
            "theta": i * 0.3,  # Moderately dispersed
            "EPI": 0.5,
            "Si": 0.5 + i * 0.02,
            "ΔNFR": 0.2 + i * 0.02,
            "dEPI_dt": 0.15 + i * 0.01,
        })
    
    return G


def print_limits_comparison(G: nx.Graph, label: str):
    """Print comparison of static vs dynamic limits for a network."""
    
    # Compute dynamic limits with default configuration
    limits = compute_dynamic_limits(G)
    
    # Static limits (from CoreDefaults)
    static_epi_max = 1.0
    static_vf_max = 10.0
    
    print(f"\n{'='*70}")
    print(f"{label.upper()}")
    print(f"{'='*70}")
    print(f"Nodes: {G.number_of_nodes()}")
    print()
    
    # Coherence metrics
    print("Coherence Metrics:")
    print(f"  C(t) (global coherence):     {limits.coherence:.4f}")
    print(f"  Si_avg (sense index):        {limits.si_avg:.4f}")
    print(f"  R (Kuramoto order):          {limits.kuramoto_r:.4f}")
    print(f"  Coherence factor (C×Si):     {limits.coherence_factor:.4f}")
    print()
    
    # EPI limits comparison
    epi_expansion = (limits.epi_max_effective / static_epi_max - 1) * 100
    print("EPI Limits:")
    print(f"  Static limit:                {static_epi_max:.4f}")
    print(f"  Dynamic limit:               {limits.epi_max_effective:.4f}")
    print(f"  Expansion:                   {epi_expansion:+.2f}%")
    print()
    
    # νf limits comparison
    vf_expansion = (limits.vf_max_effective / static_vf_max - 1) * 100
    print("νf (Structural Frequency) Limits:")
    print(f"  Static limit:                {static_vf_max:.4f} Hz_str")
    print(f"  Dynamic limit:               {limits.vf_max_effective:.4f} Hz_str")
    print(f"  Expansion:                   {vf_expansion:+.2f}%")
    print()
    
    # Theoretical interpretation
    print("Theoretical Interpretation:")
    if limits.coherence > 0.7:
        print("  → High coherence: Network can sustain expanded limits")
        print("  → Self-organization is strong and stable")
    elif limits.coherence < 0.4:
        print("  → Low coherence: Limits remain conservative")
        print("  → Self-organization is weak, system needs constraints")
    else:
        print("  → Moderate coherence: Limits moderately expanded")
        print("  → System is in transition state")


def demonstrate_dynamic_limits():
    """Main demonstration of dynamic limits theory."""
    
    print("""
╔════════════════════════════════════════════════════════════════════╗
║  Dynamic Canonical Limits in TNFR                                  ║
║  Theoretical Investigation: Issue fermga/TNFR-Python-Engine#2624   ║
╚════════════════════════════════════════════════════════════════════╝

This demonstration explores whether fixed canonical limits (EPI_MAX, VF_MAX)
contradict TNFR's self-organizing principles, and proposes dynamic limits
that adapt based on network coherence metrics.

Key Question:
Should limits be imposed externally (static) or emerge from the system's
own dynamics (dynamic)?
""")
    
    # Test 1: Highly coherent network
    print("\n" + "="*70)
    print("TEST 1: HIGHLY COHERENT NETWORK")
    print("="*70)
    print("A network with strong phase synchronization and high sense index.")
    print("Expected: Dynamic limits should expand significantly.")
    
    coherent_net = create_coherent_network()
    print_limits_comparison(coherent_net, "Coherent Network")
    
    # Test 2: Chaotic network
    print("\n" + "="*70)
    print("TEST 2: CHAOTIC NETWORK")
    print("="*70)
    print("A network with dispersed phases and low sense index.")
    print("Expected: Dynamic limits should remain conservative.")
    
    chaotic_net = create_chaotic_network()
    print_limits_comparison(chaotic_net, "Chaotic Network")
    
    # Test 3: Transitional network
    print("\n" + "="*70)
    print("TEST 3: TRANSITIONAL NETWORK")
    print("="*70)
    print("A network in transition with moderate coherence.")
    print("Expected: Dynamic limits should moderately expand.")
    
    transitional_net = create_transitional_network()
    print_limits_comparison(transitional_net, "Transitional Network")
    
    # Comparison and conclusions
    print("\n" + "="*70)
    print("THEORETICAL CONCLUSIONS")
    print("="*70)
    print("""
1. OPERATIONAL FRACTALITY PRESERVED:
   - Dynamic limits allow patterns to scale naturally with coherence
   - No artificial bounds interrupt self-organization
   - Fractality maintained through proportional scaling

2. SELF-ORGANIZATION RESPECTED:
   - Limits emerge from system state (C(t), Si, R_kuramoto)
   - Network finds its own natural bounds
   - External constraints only provide safety maximum

3. COHERENCE EMERGENCE VALIDATED:
   - Stability arises from resonance (measured by metrics)
   - High coherence enables higher operational values
   - Low coherence naturally restricts system dynamics

4. TNFR INVARIANTS MAINTAINED:
   - Operator closure preserved (limits remain finite)
   - Structural semantics: expansion ∝ coherence
   - νf in Hz_str units (structural frequency)
   - ΔNFR not reinterpreted as error gradient

RECOMMENDATION:
Replace static limits with dynamic limits as the canonical implementation.
This better aligns with TNFR's theoretical foundation and preserves the
paradigm's self-organizing nature.
""")
    
    print("\n" + "="*70)
    print("END OF DEMONSTRATION")
    print("="*70)


if __name__ == "__main__":
    demonstrate_dynamic_limits()
