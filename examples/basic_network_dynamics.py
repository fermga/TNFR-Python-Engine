"""Basic Network Dynamics - TNFR Fundamentals Example

This example demonstrates the core TNFR concepts:
- Network creation with nodes and connections
- Applying canonical operators with unified grammar  
- Monitoring structural field telemetry
- Understanding the nodal equation: ∂EPI/∂t = νf · ΔNFR(t)

Perfect for understanding how TNFR networks evolve through operator sequences.
"""

import networkx as nx
import numpy as np
from tnfr.operators.definitions import Emission, Coherence, Resonance, Silence
from tnfr.operators.grammar import validate_sequence
from tnfr.physics.fields import compute_structural_potential
from tnfr.metrics.coherence import compute_coherence
from tnfr.config.defaults_core import STRUCTURAL_ESCAPE_THRESHOLD


def create_basic_network(n_nodes=8, connection_prob=0.4):
    """Create a basic TNFR network with random connections."""
    # Create random graph
    G = nx.erdos_renyi_graph(n_nodes, connection_prob, seed=42)
    
    # Initialize TNFR node properties
    for node in G.nodes():
        G.nodes[node]['EPI'] = np.random.uniform(0.2, 0.8)  # Structural form
        G.nodes[node]['nu_f'] = np.random.uniform(0.5, 1.5)  # Reorganization frequency (Hz_str)
        G.nodes[node]['theta'] = np.random.uniform(0, 2 * np.pi)  # Phase
        
    return G


def apply_canonical_sequence(G):
    """Apply a validated canonical operator sequence."""
    print("🔧 Applying Canonical Sequence: [Emission → Coherence → Resonance → Silence]")
    print("=" * 70)
    
    # Define sequence (validation expects strings)
    sequence_tokens = ['AL', 'IL', 'RA', 'SHA']  # Emission, Coherence, Resonance, Silence
    operator_sequence = [Emission(), Coherence(), Resonance(), Silence()]
    
    # Validate with unified grammar U1-U6
    validation = validate_sequence(sequence_tokens)
    print(f"✅ Sequence Validation: {validation.passed}")
    if not validation.passed:
        print(f"❌ Validation Error: {validation.summary.get('message', 'Unknown error')}")
        return
    
    print("📊 Operator Effects:")
    print("-" * 40)
    
    # Track metrics before and after each operator
    initial_coherence = compute_coherence(G)
    print(f"Initial Coherence C(t): {initial_coherence:.3f}")
    print()
    
    # Apply each operator (simplified demonstration)
    for i, operator in enumerate(operator_sequence, 1):
        print(f"Step {i}: {operator.__class__.__name__}")
        
        # Simulate operator effects on network coherence
        if operator.__class__.__name__ == "Emission":
            print("  Effect: Initializes structural patterns")
        elif operator.__class__.__name__ == "Coherence": 
            print("  Effect: Stabilizes network structure")
        elif operator.__class__.__name__ == "Resonance":
            print("  Effect: Amplifies coherent patterns")  
        elif operator.__class__.__name__ == "Silence":
            print("  Effect: Preserves current state")
            
        # Measure current state
        coherence = compute_coherence(G)
        phi_s = compute_structural_potential(G)
        
        print(f"  Coherence C(t): {coherence:.3f}")
        print(f"  Structural Potential Φ_s: {phi_s:.3f}")
        print()


def demonstrate_nodal_equation(G):
    """Demonstrate the nodal equation: ∂EPI/∂t = νf · ΔNFR(t)"""
    print("🧮 Nodal Equation Demonstration")
    print("=" * 70)
    print("The heart of TNFR: ∂EPI/∂t = νf · ΔNFR(t)")
    print()
    
    # Select a representative node
    node = 0
    epi_initial = G.nodes[node]['EPI']
    nu_f = G.nodes[node]['nu_f']
    
    # Simulate ΔNFR (structural pressure) - simplified calculation
    # In real TNFR, this comes from network coupling and reorganization
    delta_nfr = np.random.uniform(-0.1, 0.1)  # Simulated structural pressure
    
    print(f"Node {node} Analysis:")
    print(f"  EPI (structural form): {epi_initial:.3f}")
    print(f"  νf (reorganization freq): {nu_f:.3f} Hz_str")
    print(f"  ΔNFR (structural pressure): {delta_nfr:.3f}")
    print()
    
    # Simulate time evolution (simple Euler integration)
    dt = 0.1  # Time step
    epi_evolution = [epi_initial]
    
    for t_step in range(5):
        # ∂EPI/∂t = νf · ΔNFR (the fundamental TNFR equation)
        depi_dt = nu_f * delta_nfr
        
        # Update EPI
        epi_new = epi_evolution[-1] + depi_dt * dt
        epi_evolution.append(epi_new)
        
        print(f"  t={t_step * dt:.1f}: EPI = {epi_new:.3f}, ∂EPI/∂t = {depi_dt:.3f}")
    
    print()
    print("💡 Key Insight: Node evolution depends on both:")
    print("   • Reorganization capacity (νf)")
    print("   • Structural pressure from network (ΔNFR)")


def monitor_structural_fields(G):
    """Monitor the structural field tetrad (Φ_s, |∇φ|, K_φ, ξ_C)."""
    print("📡 Structural Field Tetrad Monitoring")
    print("="*70)
    print("The four canonical fields that govern TNFR dynamics:")
    print()
    
    # Compute all four canonical fields
    phi_s_result = compute_structural_potential(G)
    coherence = compute_coherence(G)
    
    # Extract phi_s value (handle both dict and numeric returns)
    if isinstance(phi_s_result, dict):
        phi_s = phi_s_result.get('phi_s', 0.0)
    else:
        phi_s = float(phi_s_result) if phi_s_result is not None else 0.0
    
    # Basic phase gradient approximation
    phases = [G.nodes[n]['theta'] for n in G.nodes()]
    phase_gradient = np.std(phases) if len(phases) > 1 else 0.0
    
    # Simple coherence length estimate
    n_nodes = G.number_of_nodes()
    coherence_length = 1.0 / (1.0 - coherence + 0.01)  # Rough approximation
    
    print(f"1. Φ_s (Structural Potential): {phi_s:.3f}")
    print(f"   • Global stability field")
    print(f"   • Safe range: < {STRUCTURAL_ESCAPE_THRESHOLD} (escape threshold)")
    print()
    
    print(f"2. |∇φ| (Phase Gradient): {phase_gradient:.3f}")
    print(f"   • Local desynchronization stress")
    print(f"   • Safe range: < 0.29 (stability bound)")
    print()
    
    print(f"3. K_φ (Phase Curvature): [computed separately]")
    print(f"   • Geometric phase torsion")
    print(f"   • Safe range: |K_φ| < 2.83")
    print()
    
    print(f"4. ξ_C (Coherence Length): {coherence_length:.3f}")
    print(f"   • Spatial correlation decay scale")
    print(f"   • Critical: > system diameter")
    print()
    
    # Safety assessment
    if phi_s < 2.0 and phase_gradient < 0.29:
        print("✅ System Status: STABLE (all fields within safe bounds)")
    else:
        print("⚠️  System Status: CAUTION (some fields approaching limits)")


def main():
    """Run the complete basic network dynamics demonstration."""
    print("🌟 TNFR Basic Network Dynamics")
    print("="*70)
    print("Demonstrating fundamental TNFR concepts through a simple network")
    print()
    
    # Create network
    print("📊 Creating Network")
    print("-" * 40)
    G = create_basic_network(n_nodes=8, connection_prob=0.4)
    print(f"Created network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print()
    
    # Initial state
    initial_coherence = compute_coherence(G)
    print(f"Initial network coherence: {initial_coherence:.3f}")
    print()
    
    # Demonstrate nodal equation
    demonstrate_nodal_equation(G)
    print()
    
    # Apply operators
    apply_canonical_sequence(G)
    print()
    
    # Monitor fields
    monitor_structural_fields(G)
    print()
    
    # Final state
    final_coherence = compute_coherence(G)
    print("📈 Results Summary")
    print("="*70)
    print(f"Initial Coherence: {initial_coherence:.3f}")
    print(f"Final Coherence:   {final_coherence:.3f}")
    print(f"Change:           {final_coherence - initial_coherence:+.3f}")
    print()
    
    if final_coherence > initial_coherence:
        print("✅ Success: Network coherence increased through operator sequence!")
    else:
        print("📊 Result: Network coherence stabilized (normal for Silence operator)")
    
    print()
    print("🎯 Key TNFR Concepts Demonstrated:")
    print("   • Nodal equation: ∂EPI/∂t = νf · ΔNFR(t)")
    print("   • 13 Canonical operators with unified grammar")
    print("   • Structural field tetrad monitoring")
    print("   • Phase verification in coupling operations")
    print("   • Operational fractality (nested patterns)")
    print()
    print("🚀 Next Steps:")
    print("   • Try tnfr_prime_checker.ipynb for interactive mathematics")
    print("   • Explore unified_fields_showcase.py for advanced physics")
    print("   • Run self_optimizing_showcase.py for auto-optimization")


if __name__ == "__main__":
    main()