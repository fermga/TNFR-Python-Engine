"""01 - Hello World: Your First TNFR Experience

This is the absolute simplest introduction to TNFR - perfect for complete beginners!

PHYSICS: Demonstrates basic TNFR concepts without complex API calls.
LEARNING: Understand what "coherent patterns" and "structural reorganization" mean.
"""

import networkx as nx
import numpy as np


def hello_world():
    """The simplest possible TNFR demonstration."""
    
    print("=" * 60)
    print(" " * 18 + "🌊 Hello, TNFR! 🌊")
    print("=" * 60)
    print()
    print("Welcome to Resonant Fractal Nature Theory!")
    print("Let's explore coherent systems in 3 simple steps...")
    print()
    
    # STEP 1: Create a simple network
    print("📡 STEP 1: Creating a simple network...")
    print("   Think of this as nodes that can synchronize")
    
    # Create a simple network manually
    G = nx.cycle_graph(6)  # 6 nodes in a circle
    
    # Add basic TNFR properties
    np.random.seed(42)  # For reproducible results
    for node in G.nodes():
        G.nodes[node]['nu_f'] = 1.0  # Structural frequency
        G.nodes[node]['theta'] = np.random.uniform(0, 2*np.pi)  # Phase
        G.nodes[node]['EPI'] = np.random.uniform(0.1, 0.3)  # Small EPI values
    
    print("   ✅ Created 6-node cycle network")
    print(f"   📊 Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    print()
    
    # STEP 2: Measure initial state
    print("🎵 STEP 2: Measuring initial coherence...")
    print("   Physics: Coherence measures how well synchronized the system is")
    
    # Phase synchronization: the canonical Kuramoto order parameter
    # R = |<e^{iθ}>|  (R = 1 fully aligned, R -> 0 desynchronized/antiphase).
    thetas = np.array([G.nodes[n]['theta'] for n in G.nodes()])
    initial_coherence = float(abs(np.mean(np.exp(1j * thetas))))

    print(f"   📊 Initial phase synchrony R: {initial_coherence:.3f}")
    print()
    
    # STEP 3: Simulate synchronization
    print("🔄 STEP 3: Simulating synchronization...")
    print("   Physics: Nodes adjust their phases toward neighbors")
    print("   This is like musicians tuning to each other")
    
    # Simple synchronization simulation
    for step in range(5):
        new_phases = {}
        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            if neighbors:
                # Average neighbor phases
                neighbor_phases = [G.nodes[n]['theta'] for n in neighbors]
                avg_neighbor_phase = np.mean(neighbor_phases)
                
                # Adjust toward neighbors (simple model)
                current_phase = G.nodes[node]['theta']
                new_phases[node] = 0.8 * current_phase + 0.2 * avg_neighbor_phase
            else:
                new_phases[node] = G.nodes[node]['theta']
        
        # Update phases
        for node in G.nodes():
            G.nodes[node]['theta'] = new_phases[node]
    
    # Measure final phase synchrony (same canonical Kuramoto R)
    final_thetas = np.array([G.nodes[n]['theta'] for n in G.nodes()])
    final_coherence = float(abs(np.mean(np.exp(1j * final_thetas))))
    
    print("   ✅ Synchronization complete!")
    print(f"   📊 Final phase synchrony R: {final_coherence:.3f}")
    print(f"   📈 Improvement: {final_coherence - initial_coherence:.3f}")
    print()
    
    # PHYSICS EXPLANATION
    print("🧮 WHAT JUST HAPPENED? (The Physics)")
    print("=" * 60)
    print()
    print("1. NODAL EQUATION in simple terms:")
    print("   ∂EPI/∂t = νf · ΔNFR(t)")
    print("   • EPI = the coherent pattern (what we're measuring)")
    print("   • νf = how fast things can change (structural frequency)")
    print("   • ΔNFR = the 'pressure' to reorganize (mismatch with neighbors)")
    print()
    print("2. WHAT WE SIMULATED:")
    print("   • Started with random phases (low coherence)")
    print("   • Each node adjusted toward its neighbors")
    print("   • System self-organized toward higher coherence")
    print("   • This is emergence in action!")
    print()
    print("3. REAL-WORLD EXAMPLES:")
    print("   • Fireflies synchronizing their flashes")
    print("   • Heart cells beating in rhythm")
    print("   • Brain waves coordinating during focus")
    print("   • Orchestra musicians playing in time")
    print()
    print("4. TNFR INSIGHT:")
    print("   • Reality isn't made of 'things' - it's made of PATTERNS")
    print("   • These patterns exist because they RESONATE together")
    print("   • Coherence = how well patterns synchronize")
    print("   • Higher coherence = more stable, organized systems")
    print()
    
    # RESULTS SUMMARY
    print("📊 RESULTS SUMMARY:")
    print(f"   🎯 Initial coherence: {initial_coherence:.3f}")
    print(f"   🎯 Final coherence:   {final_coherence:.3f}")
    print(f"   📈 Improvement:      +{final_coherence - initial_coherence:.3f}")
    coherence_success = "Successful" if final_coherence > initial_coherence else "Failed"
    print(f"   🌊 Synchronization:   {coherence_success}")
    print()
    
    # NEXT STEPS
    print("🚀 NEXT STEPS:")
    print("   • Try: python 02_musical_resonance.py")
    print("   • Read: examples/README.md for learning paths")
    print("   • Explore: AGENTS.md for complete theory")
    print("   • Think: Where else do you see synchronization in nature?")
    print()
    print("=" * 60)
    
    return {
        'initial_coherence': initial_coherence,
        'final_coherence': final_coherence,
        'improvement': final_coherence - initial_coherence,
        'network_size': G.number_of_nodes()
    }


if __name__ == "__main__":
    hello_world()