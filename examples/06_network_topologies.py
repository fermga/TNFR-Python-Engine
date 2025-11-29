"""07 - Network Topologies: TNFR Across Different Structures

Comprehensive exploration of TNFR dynamics across various network topologies.

PHYSICS: Shows how network structure affects nodal equation evolution.
LEARNING: Understand topology-dependent coherence patterns and stability.
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

# Configurar fuente para mejor soporte de Unicode
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
# Suprimir warnings de glifos faltantes
import warnings
warnings.filterwarnings('ignore', 'Glyph .* missing from font.*')


def compute_coherence(G):
    """Compute network coherence from phase synchronization."""
    phases = [G.nodes[n].get('phase', 0) for n in G.nodes()]
    if len(phases) < 2:
        return 1.0
    
    phase_diffs = []
    for i in range(len(phases)):
        for j in range(i + 1, len(phases)):
            diff = abs(phases[i] - phases[j])
            diff = min(diff, 2 * np.pi - diff)
            phase_diffs.append(diff)
    
    return 1.0 - (np.mean(phase_diffs) / np.pi) if phase_diffs else 1.0


def compute_delta_nfr(G, node):
    """Compute ΔNFR (structural pressure) for a node."""
    if node not in G.nodes():
        return 0.0
    
    node_phase = G.nodes[node].get('phase', 0)
    neighbors = list(G.neighbors(node))
    
    if not neighbors:
        return 0.0
    
    neighbor_phases = [G.nodes[n].get('phase', 0) for n in neighbors]
    mean_neighbor_phase = np.mean(neighbor_phases)
    
    phase_diff = abs(node_phase - mean_neighbor_phase)
    return min(phase_diff, 2 * np.pi - phase_diff) / np.pi


def evolve_network_step(G, dt=0.1):
    """Single evolution step applying nodal equation."""
    new_phases = {}
    
    for node in G.nodes():
        current_phase = G.nodes[node].get('phase', 0)
        vf = G.nodes[node].get('vf', 1.0)
        
        neighbors = list(G.neighbors(node))
        if neighbors:
            neighbor_phases = [G.nodes[n].get('phase', 0) for n in neighbors]
            target_phase = np.mean(neighbor_phases)
            
            direction = target_phase - current_phase
            if direction > np.pi:
                direction -= 2 * np.pi
            elif direction < -np.pi:
                direction += 2 * np.pi
            
            delta_nfr = compute_delta_nfr(G, node)
            
            # Apply nodal equation: ∂EPI/∂t = νf · ΔNFR
            phase_change = vf * delta_nfr * dt * np.sign(direction)
            new_phases[node] = (current_phase + phase_change) % (2 * np.pi)
        else:
            new_phases[node] = current_phase
    
    for node, phase in new_phases.items():
        G.nodes[node]['phase'] = phase


def create_topology_comparison_visualization(topology_results):
    """Create comprehensive topology comparison visualization."""
    
    print("🎨 Creating topology comparison visualization...")
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()
    
    # Plot each topology's evolution and final state
    topology_names = list(topology_results.keys())
    
    for idx, (name, data) in enumerate(topology_results.items()):
        if idx >= 6:  # Limit to 6 topologies for layout
            break
            
        # Evolution plot
        ax = axes[idx]
        ax.plot(data['evolution'], linewidth=3, color=data.get('color', 'blue'))
        ax.set_title(f'{name}\nFinal: {data["final_coherence"]:.3f}', fontweight='bold')
        ax.set_xlabel('Evolution Steps')
        ax.set_ylabel('Coherence')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    # Summary comparison bar chart
    if len(topology_names) > 6:
        ax_summary = axes[6]
        coherences = [topology_results[name]['final_coherence'] for name in topology_names]
        colors = [topology_results[name].get('color', 'blue') for name in topology_names]
        
        bars = ax_summary.bar(range(len(topology_names)), coherences, color=colors, alpha=0.8)
        ax_summary.set_xlabel('Topology')
        ax_summary.set_ylabel('Final Coherence')
        ax_summary.set_title('📊 Final Coherence Comparison')
        ax_summary.set_xticks(range(len(topology_names)))
        ax_summary.set_xticklabels(topology_names, rotation=45, ha='right')
        ax_summary.grid(True, alpha=0.3)
        ax_summary.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, coherence in zip(bars, coherences):
            height = bar.get_height()
            ax_summary.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{coherence:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Network structure visualization
    if len(topology_names) > 7:
        ax_network = axes[7]
        
        # Show one example topology (Complete graph)
        G_example = nx.complete_graph(8)
        pos = nx.spring_layout(G_example, seed=42)
        
        nx.draw_networkx_edges(G_example, pos, ax=ax_network, alpha=0.6, edge_color='gray')
        nx.draw_networkx_nodes(G_example, pos, ax=ax_network, node_color='lightblue',
                              node_size=300, edgecolors='black')
        nx.draw_networkx_labels(G_example, pos, ax=ax_network, font_size=8)
        
        ax_network.set_title('Example: Complete Graph')
        ax_network.axis('off')
    
    # Physics summary
    if len(topology_names) > 8:
        ax_physics = axes[8]
        ax_physics.text(0.1, 0.9, 'TNFR Physics Summary:', fontsize=14, fontweight='bold',
                       transform=ax_physics.transAxes)
        ax_physics.text(0.1, 0.7, '• ∂EPI/∂t = νf · ΔNFR(t)', fontsize=12,
                       transform=ax_physics.transAxes)
        ax_physics.text(0.1, 0.5, '• Topology → Information flow', fontsize=12,
                       transform=ax_physics.transAxes)
        ax_physics.text(0.1, 0.3, '• Structure → Coherence rate', fontsize=12,
                       transform=ax_physics.transAxes)
        ax_physics.text(0.1, 0.1, '• Complete graphs → Fast sync', fontsize=12,
                       transform=ax_physics.transAxes)
        ax_physics.axis('off')
    
    plt.suptitle('🕸️ TNFR Dynamics Across Network Topologies\n' +
                 'How Structure Shapes Evolution', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/topology_comparison_detailed.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Saved: output/topology_comparison_detailed.png")


def network_topologies_demo():
    """Comprehensive demonstration of TNFR across different network topologies."""
    
    print("=" * 80)
    print(" " * 20 + "🕸️ Network Topologies Analysis 🕸️")
    print("=" * 80)
    print()
    print("PHYSICS: How does network structure affect ∂EPI/∂t = νf · ΔNFR evolution?")
    print("DISCOVERY: Different topologies create different coherence landscapes!")
    print()
    
    # Define topologies to test
    topologies = {
        'Complete': {
            'graph': nx.complete_graph(8),
            'description': 'Every node connected to every other',
            'color': '#FF6B6B'
        },
        'Ring': {
            'graph': nx.cycle_graph(8),
            'description': 'Nodes connected in a circle',
            'color': '#4ECDC4'
        },
        'Star': {
            'graph': nx.star_graph(7),
            'description': 'Central hub with spokes',
            'color': '#45B7D1'
        },
        'Path': {
            'graph': nx.path_graph(8),
            'description': 'Linear chain of connections',
            'color': '#96CEB4'
        },
        'Grid 2D': {
            'graph': nx.grid_2d_graph(3, 3),
            'description': 'Regular 2D lattice',
            'color': '#FECA57'
        },
        'Random': {
            'graph': nx.erdos_renyi_graph(8, 0.4),
            'description': 'Random connections (p=0.4)',
            'color': '#FF9FF3'
        },
        'Small World': {
            'graph': nx.watts_strogatz_graph(8, 3, 0.3),
            'description': 'Small-world rewiring',
            'color': '#54A0FF'
        },
        'Scale-Free': {
            'graph': nx.barabasi_albert_graph(8, 2),
            'description': 'Preferential attachment',
            'color': '#5F27CD'
        }
    }
    
    results = {}
    
    for topo_name, topo_data in topologies.items():
        print(f"🔍 TESTING: {topo_name} Topology")
        print(f"   Description: {topo_data['description']}")
        
        G = topo_data['graph']
        
        # Initialize with random phases
        np.random.seed(42)  # Reproducible
        for node in G.nodes():
            G.nodes[node]['phase'] = np.random.uniform(0, 2 * np.pi)
            G.nodes[node]['vf'] = 1.0
        
        initial_coherence = compute_coherence(G)
        
        # Track evolution
        steps = 40
        coherence_history = []
        
        for step in range(steps):
            coherence = compute_coherence(G)
            coherence_history.append(coherence)
            evolve_network_step(G, dt=0.1)
        
        final_coherence = compute_coherence(G)
        
        # Calculate metrics
        improvement = final_coherence - initial_coherence
        convergence_speed = 0
        
        # Find convergence point (when coherence stabilizes)
        for i in range(10, len(coherence_history)):
            if abs(coherence_history[i] - coherence_history[i-5]) < 0.01:
                convergence_speed = i
                break
        
        results[topo_name] = {
            'evolution': coherence_history,
            'initial_coherence': initial_coherence,
            'final_coherence': final_coherence,
            'improvement': improvement,
            'convergence_speed': convergence_speed,
            'color': topo_data['color']
        }
        
        print(f"   Initial coherence: {initial_coherence:.3f}")
        print(f"   Final coherence:   {final_coherence:.3f}")
        print(f"   Improvement:       {improvement:+.3f}")
        print(f"   Convergence step:  {convergence_speed}")
        print()
    
    # Create comprehensive visualization
    create_topology_comparison_visualization(results)
    
    # ANALYSIS SUMMARY
    print("📊 TOPOLOGY ANALYSIS SUMMARY:")
    print("=" * 80)
    print()
    
    # Sort by final coherence
    sorted_results = sorted(results.items(), key=lambda x: x[1]['final_coherence'], reverse=True)
    
    print("🏆 RANKING BY FINAL COHERENCE:")
    for rank, (name, data) in enumerate(sorted_results, 1):
        print(f"   {rank}. {name:12s}: {data['final_coherence']:.3f} "
              f"(+{data['improvement']:+.3f} in {data['convergence_speed']} steps)")
    print()
    
    # Best and worst performers
    best_topo, best_data = sorted_results[0]
    worst_topo, worst_data = sorted_results[-1]
    
    print("📈 PERFORMANCE INSIGHTS:")
    print(f"   🥇 Best performer:  {best_topo} ({best_data['final_coherence']:.3f})")
    print(f"   📉 Worst performer: {worst_topo} ({worst_data['final_coherence']:.3f})")
    print(f"   📊 Performance gap: {best_data['final_coherence'] - worst_data['final_coherence']:.3f}")
    print()
    
    # THEORETICAL INSIGHTS
    print("🧮 THEORETICAL INSIGHTS:")
    print("=" * 80)
    print()
    print("1. CONNECTIVITY vs COHERENCE:")
    print("   • Higher connectivity → Faster convergence")
    print("   • Complete graphs achieve maximum coherence")
    print("   • Bottlenecks (like paths) slow information flow")
    print()
    print("2. STRUCTURE DETERMINES DYNAMICS:")
    print("   • Hub nodes (stars) create convergence centers")
    print("   • Regular structures (rings, grids) show steady evolution")
    print("   • Random structures balance exploration/exploitation")
    print()
    print("3. NODAL EQUATION MANIFESTATIONS:")
    print("   • ΔNFR reflects local vs global phase misalignment")
    print("   • νf uniform → topology is the only variable")
    print("   • Evolution rate ∝ information flow efficiency")
    print()
    print("4. REAL-WORLD IMPLICATIONS:")
    print("   • Social networks: Dense connections → faster consensus")
    print("   • Neural networks: Topology affects learning speed")
    print("   • Internet: Structure determines resilience")
    print()
    
    # NEXT STEPS
    print("🚀 EXPLORATION SUGGESTIONS:")
    print("   • Modify νf values: What if nodes have different frequencies?")
    print("   • Dynamic topology: What if connections change over time?")
    print("   • Directed graphs: How does direction affect coherence?")
    print("   • Weighted edges: Do connection strengths matter?")
    print()
    
    return results


if __name__ == "__main__":
    network_topologies_demo()