"""05 - Coherence Evolution: Understanding System Dynamics

Advanced demonstration of coherence evolution through TNFR nodal dynamics.

PHYSICS: Shows how ∂EPI/∂t = νf · ΔNFR(t) governs system evolution.
LEARNING: Understand coherence trajectories and stability landscapes.
"""

import networkx as nx
import numpy as np
import time


def compute_coherence(G):
    """Compute network coherence from phase synchronization."""
    phases = [G.nodes[n].get('phase', 0) for n in G.nodes()]
    if len(phases) < 2:
        return 1.0
    
    # Coherence = 1 - (phase variance / maximum possible variance)
    phase_diffs = []
    for i in range(len(phases)):
        for j in range(i + 1, len(phases)):
            diff = abs(phases[i] - phases[j])
            diff = min(diff, 2 * np.pi - diff)  # Circular distance
            phase_diffs.append(diff)
    
    if not phase_diffs:
        return 1.0
        
    mean_diff = np.mean(phase_diffs)
    return 1.0 - (mean_diff / np.pi)


def compute_delta_nfr(G, node):
    """Compute ΔNFR (structural pressure) for a node."""
    if node not in G.nodes():
        return 0.0
    
    node_phase = G.nodes[node].get('phase', 0)
    neighbors = list(G.neighbors(node))
    
    if not neighbors:
        return 0.0
    
    # ΔNFR = mismatch with neighbors (structural pressure)
    neighbor_phases = [G.nodes[n].get('phase', 0) for n in neighbors]
    mean_neighbor_phase = np.mean(neighbor_phases)
    
    phase_diff = abs(node_phase - mean_neighbor_phase)
    phase_diff = min(phase_diff, 2 * np.pi - phase_diff)
    
    return phase_diff / np.pi  # Normalize to [0,1]


def evolve_network_step(G, dt=0.1):
    """Single evolution step: ∂EPI/∂t = νf · ΔNFR(t)"""
    new_phases = {}
    
    for node in G.nodes():
        # Get current state
        current_phase = G.nodes[node].get('phase', 0)
        vf = G.nodes[node].get('vf', 1.0)  # Structural frequency
        
        # Compute ΔNFR (structural pressure)
        delta_nfr = compute_delta_nfr(G, node)
        
        # Apply nodal equation: ∂EPI/∂t = νf · ΔNFR
        # Here EPI is represented by phase
        phase_change = vf * delta_nfr * dt
        
        # Update phase toward neighbors (synchronization)
        neighbors = list(G.neighbors(node))
        if neighbors:
            neighbor_phases = [G.nodes[n].get('phase', 0) for n in neighbors]
            target_phase = np.mean(neighbor_phases)
            
            # Move toward target with rate determined by nodal equation
            direction = target_phase - current_phase
            if direction > np.pi:
                direction -= 2 * np.pi
            elif direction < -np.pi:
                direction += 2 * np.pi
            
            new_phase = current_phase + np.sign(direction) * phase_change
            new_phases[node] = new_phase % (2 * np.pi)
        else:
            new_phases[node] = current_phase
    
    # Update all phases
    for node, phase in new_phases.items():
        G.nodes[node]['phase'] = phase


def coherence_evolution_demo():
    """Demonstrate coherence evolution in various network topologies."""
    
    print("=" * 70)
    print("               🔬 COHERENCE EVOLUTION DYNAMICS 🔬")
    print("=" * 70)
    print()
    print("Exploring how coherence evolves through TNFR nodal dynamics...")
    print("PHYSICS: ∂EPI/∂t = νf · ΔNFR(t)")
    print()
    
    # Test different network topologies
    topologies = {
        "Ring": nx.cycle_graph(8),
        "Star": nx.star_graph(7), 
        "Complete": nx.complete_graph(6),
        "Random": nx.erdos_renyi_graph(8, 0.3)
    }
    
    results = {}
    
    for name, G in topologies.items():
        print(f"📊 TOPOLOGY: {name}")
        print(f"   Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
        
        # Initialize random phases and structural frequencies
        np.random.seed(42)  # Reproducible
        for node in G.nodes():
            G.nodes[node]['phase'] = np.random.uniform(0, 2*np.pi)
            G.nodes[node]['vf'] = np.random.uniform(0.8, 1.2)  # Structural frequency
        
        # Measure initial state
        initial_coherence = compute_coherence(G)
        initial_delta_nfr = np.mean([compute_delta_nfr(G, n) for n in G.nodes()])
        
        print(f"   🎯 Initial coherence: {initial_coherence:.3f}")
        print(f"   ⚡ Initial ΔNFR: {initial_delta_nfr:.3f}")
        
        # Evolution simulation
        coherence_history = [initial_coherence]
        delta_nfr_history = [initial_delta_nfr]
        
        print("   🔄 Evolving system...", end="")
        
        for step in range(50):
            evolve_network_step(G)
            
            if step % 10 == 0:
                print(".", end="", flush=True)
                
            coherence = compute_coherence(G)
            avg_delta_nfr = np.mean([compute_delta_nfr(G, n) for n in G.nodes()])
            
            coherence_history.append(coherence)
            delta_nfr_history.append(avg_delta_nfr)
        
        print(" ✅")
        
        # Final measurements
        final_coherence = coherence_history[-1]
        final_delta_nfr = delta_nfr_history[-1]
        coherence_change = final_coherence - initial_coherence
        
        print(f"   📈 Final coherence: {final_coherence:.3f}")
        print(f"   📉 Final ΔNFR: {final_delta_nfr:.3f}")
        print(f"   🌊 Coherence change: {coherence_change:+.3f}")
        
        # Analyze convergence
        recent_coherence = coherence_history[-10:]
        coherence_stability = np.std(recent_coherence)
        
        print(f"   🎯 Convergence stability: {coherence_stability:.4f}")
        
        if coherence_stability < 0.01:
            print("   ✅ System reached stable equilibrium")
        else:
            print("   ⚠️  System still evolving")
        
        results[name] = {
            'initial_coherence': initial_coherence,
            'final_coherence': final_coherence,
            'change': coherence_change,
            'stability': coherence_stability,
            'history': coherence_history
        }
        
        print()
    
    # Comparative analysis
    print("🧮 COMPARATIVE ANALYSIS")
    print("=" * 50)
    
    # Best converging topology
    best_topology = max(results.keys(), key=lambda k: results[k]['final_coherence'])
    print(f"🏆 Best coherence: {best_topology} ({results[best_topology]['final_coherence']:.3f})")
    
    # Fastest stabilizing topology
    most_stable = min(results.keys(), key=lambda k: results[k]['stability'])
    print(f"🎯 Most stable: {most_stable} (σ = {results[most_stable]['stability']:.4f})")
    
    # Biggest improvement
    biggest_improvement = max(results.keys(), key=lambda k: results[k]['change'])
    print(f"📈 Biggest improvement: {biggest_improvement} ({results[biggest_improvement]['change']:+.3f})")
    
    print()
    print("🧠 INSIGHTS FROM NODAL DYNAMICS:")
    print("━" * 50)
    print("• Complete graphs converge fastest (all nodes coupled)")
    print("• Ring topologies show slower but steady improvement")  
    print("• Star topologies concentrate coherence at center")
    print("• Random graphs show variable behavior based on connectivity")
    print()
    print("🔬 PHYSICS EXPLANATION:")
    print("━" * 50)
    print("• ΔNFR measures structural pressure (phase mismatch)")
    print("• νf controls reorganization rate (structural frequency)")
    print("• System evolves to minimize ΔNFR → maximize coherence")
    print("• Network topology determines convergence dynamics")
    print()
    print("🚀 NEXT EXPERIMENTS:")
    print("━" * 50)
    print("• Try different νf distributions")
    print("• Add dynamic topology changes")
    print("• Explore multi-scale networks")
    print("• Study bifurcation points")


if __name__ == "__main__":
    coherence_evolution_demo()