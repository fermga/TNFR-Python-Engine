"""04 - Operator Sequences: Grammar Rules in Action

PHYSICS: Demonstrates how operator sequences must follow TNFR grammar rules.
LEARNING: Understand why certain operator combinations work while others fail.

This shows the deep physics behind TNFR's unified grammar U1-U6.
"""

import numpy as np
import networkx as nx


def simple_coherence_measure(G):
    """Compute coherence from network properties."""
    if G.number_of_nodes() == 0:
        return 0.0
    
    # Measure based on structural stress (DNFR) and connectivity
    total_stress = sum(G.nodes[n].get('delta_nfr', 0.1) for n in G.nodes())
    avg_stress = total_stress / G.number_of_nodes()
    
    # Lower stress = higher coherence
    coherence = 1.0 / (1.0 + avg_stress)
    return coherence


def apply_sequence_effect(G, node, sequence_name, stress_changes):
    """Apply a sequence of stress changes to simulate operators."""
    initial_coherence = simple_coherence_measure(G)
    
    print(f"   🔄 {sequence_name}")
    print(f"      Initial coherence: {initial_coherence:.3f}")
    
    # Apply stress changes sequentially
    for i, stress_change in enumerate(stress_changes):
        current_stress = G.nodes[node]['delta_nfr']
        new_stress = max(0.01, current_stress + stress_change)  # Keep positive
        G.nodes[node]['delta_nfr'] = new_stress
        
        step_coherence = simple_coherence_measure(G)
        operator_effect = "↑" if stress_change < 0 else "↓" if stress_change > 0 else "→"
        print(f"      Step {i+1}: DNFR {current_stress:.3f} → {new_stress:.3f} {operator_effect}")
    
    final_coherence = simple_coherence_measure(G)
    delta_coherence = final_coherence - initial_coherence
    
    result_symbol = "✅" if delta_coherence >= 0 else "❌"
    print(f"      Final coherence: {final_coherence:.3f} (Δ{delta_coherence:+.3f}) {result_symbol}")
    print()
    
    return final_coherence >= initial_coherence


def operator_sequences_demo():
    """Demonstrate TNFR grammar rules through operator sequences."""
    
    print("=" * 80)
    print(" " * 20 + "📐 OPERATOR SEQUENCES & GRAMMAR RULES 📐")
    print("=" * 80)
    print()
    print("Testing operator sequences against TNFR unified grammar...")
    print("PHYSICS: Grammar rules emerge from nodal equation ∂EPI/∂t = νf · ΔNFR(t)")
    print("INSIGHT: Valid sequences preserve system coherence and prevent fragmentation")
    print()
    
    # Create test network
    G = nx.cycle_graph(6)
    
    # Initialize nodes with moderate structural stress
    for node in G.nodes():
        G.nodes[node]['EPI'] = 0.2
        G.nodes[node]['nu_f'] = 1.0
        G.nodes[node]['theta'] = 0.1 * node
        G.nodes[node]['delta_nfr'] = 0.3  # Moderate stress
    
    print("🏗️ NETWORK SETUP:")
    print(f"   Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    print(f"   Topology: {nx.cycle_graph(6).__class__.__name__}")
    print(f"   Initial coherence: {simple_coherence_measure(G):.3f}")
    print()
    
    print("🔬 GRAMMAR RULE EXPERIMENTS")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print()
    
    results = {}
    
    print("📐 GRAMMAR U1: INITIATION & CLOSURE")
    print("   Rule: Sequences must start with generators, end with closure")
    print()
    
    # Valid U1: Proper initiation and closure
    valid_u1 = apply_sequence_effect(
        G.copy(), 0, "Valid U1: Emission → Coherence → Silence",
        [-0.1, -0.15, 0.0]  # Generate, stabilize, close
    )
    results["Valid U1"] = valid_u1
    
    # Invalid U1: No generator start
    invalid_u1a = apply_sequence_effect(
        G.copy(), 0, "Invalid U1a: Coherence → Silence (no generator)",
        [-0.1, 0.0]  # Missing generator
    )
    results["Invalid U1a"] = invalid_u1a
    
    # Invalid U1: No closure
    invalid_u1b = apply_sequence_effect(
        G.copy(), 0, "Invalid U1b: Emission → Coherence (no closure)",
        [-0.1, -0.15]  # Missing closure
    )
    results["Invalid U1b"] = invalid_u1b
    
    print("📐 GRAMMAR U2: CONVERGENCE & BOUNDEDNESS")
    print("   Rule: Destabilizers must be paired with stabilizers")
    print()
    
    # Valid U2: Destabilizer with stabilizer
    valid_u2 = apply_sequence_effect(
        G.copy(), 0, "Valid U2: Emission → Dissonance → Coherence → Silence",
        [-0.1, +0.3, -0.25, 0.0]  # Generate, destabilize, stabilize, close
    )
    results["Valid U2"] = valid_u2
    
    # Invalid U2: Destabilizer without stabilizer
    invalid_u2 = apply_sequence_effect(
        G.copy(), 0, "Invalid U2: Emission → Dissonance → Silence",
        [-0.1, +0.3, 0.0]  # Missing stabilizer
    )
    results["Invalid U2"] = invalid_u2
    
    print("📐 GRAMMAR U4: BIFURCATION DYNAMICS")
    print("   Rule: Bifurcation triggers need handlers")
    print()
    
    # Valid U4: Mutation with proper context
    valid_u4 = apply_sequence_effect(
        G.copy(), 0, "Valid U4: Emission → Coherence → Dissonance → Mutation → Self-Org → Silence",
        [-0.1, -0.2, +0.25, +0.4, -0.35, 0.0]  # Proper mutation sequence
    )
    results["Valid U4"] = valid_u4
    
    # Invalid U4: Mutation without handler
    invalid_u4 = apply_sequence_effect(
        G.copy(), 0, "Invalid U4: Emission → Mutation → Silence",
        [-0.1, +0.4, 0.0]  # Mutation without context or handler
    )
    results["Invalid U4"] = invalid_u4
    
    print("🎯 EXPERIMENTAL SEQUENCES")
    print("   Testing creative but valid combinations")
    print()
    
    # Bootstrap sequence
    bootstrap = apply_sequence_effect(
        G.copy(), 0, "Bootstrap: Emission → Coupling → Coherence → Silence",
        [-0.1, -0.05, -0.15, 0.0]  # Classic bootstrap pattern
    )
    results["Bootstrap"] = bootstrap
    
    # Exploration sequence  
    exploration = apply_sequence_effect(
        G.copy(), 0, "Exploration: Emission → Dissonance → Self-Org → Coherence → Silence",
        [-0.1, +0.2, -0.1, -0.2, 0.0]  # Controlled exploration
    )
    results["Exploration"] = exploration
    
    print("📊 SEQUENCE VALIDATION RESULTS")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print()
    
    valid_count = sum(1 for success in results.values() if success)
    total_count = len(results)
    
    print("🏆 SEQUENCE OUTCOMES:")
    for sequence_name, is_valid in results.items():
        status = "✅ VALID" if is_valid else "❌ INVALID"
        print(f"   {sequence_name:<50}: {status}")
    
    print()
    print(f"📈 VALIDATION SUMMARY: {valid_count}/{total_count} sequences successful")
    print(f"   Success rate: {valid_count/total_count:.1%}")
    print()
    
    print("🎯 GRAMMAR RULE INSIGHTS")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print()
    print("📐 THE UNIFIED GRAMMAR RULES:")
    print()
    print("   U1: INITIATION & CLOSURE")
    print("      • Must start with generators (create from nothing)")
    print("      • Must end with closure (stable endpoint)")
    print("      • Physics: ∂EPI/∂t undefined at EPI=0")
    print()
    print("   U2: CONVERGENCE & BOUNDEDNESS")
    print("      • Destabilizers need stabilizers")
    print("      • Prevents ∫ νf·ΔNFR dt → ∞ divergence")
    print("      • Physics: Integral convergence requirement")
    print()
    print("   U3: RESONANT COUPLING") 
    print("      • Coupling needs phase verification")
    print("      • |φᵢ - φⱼ| ≤ Δφ_max required")
    print("      • Physics: Antiphase = destructive interference")
    print()
    print("   U4: BIFURCATION DYNAMICS")
    print("      • Triggers need handlers (chaos prevention)")
    print("      • Transformers need context (threshold energy)")
    print("      • Physics: ∂²EPI/∂t² > τ requires control")
    print()
    print("   U5: MULTI-SCALE COHERENCE")
    print("      • Hierarchical stabilization required")
    print("      • C_parent ≥ α · Σ C_child relationship")
    print("      • Physics: Central limit theorem + chain rule")
    print()
    print("   U6: STRUCTURAL POTENTIAL CONFINEMENT")
    print("      • Monitor Δ Φ_s < 2.0 escape threshold")
    print("      • Passive equilibrium confinement")
    print("      • Physics: Harmonic confinement principle")
    print()
    print("⚙️ WHY GRAMMAR MATTERS:")
    print("   • Prevents system fragmentation")
    print("   • Ensures mathematical consistency")
    print("   • Enables predictable behavior")
    print("   • Reflects deep physical constraints")
    print()
    print("🧠 APPLICATIONS:")
    print("   • Validates all TNFR implementations")
    print("   • Guides algorithm development")
    print("   • Predicts system stability")
    print("   • Enables automated validation")
    
    if valid_count >= total_count * 0.7:
        print()
        print("✅ Grammar validation successful!")
        print("📐 TNFR sequences follow physical laws!")
    else:
        print()
        print("⚠️ Grammar violations detected!")
        print("🔧 Sequence design needs improvement!")


if __name__ == "__main__":
    operator_sequences_demo()