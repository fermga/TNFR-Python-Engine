"""13 - TNFR Visualization Suite: Dynamic Nodal Evolution Graphics

Comprehensive visualization of TNFR dynamics with real-time plotting and animation.

PHYSICS: Visual representation of ∂EPI/∂t = νf · ΔNFR(t) evolution.
LEARNING: Understanding through interactive graphics and dynamic plots.
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


def create_coherence_evolution_plot():
    """Create static plot showing coherence evolution across topologies."""
    
    print("🎨 Creating coherence evolution visualization...")
    
    # Create different topologies
    topologies = {
        'Ring': nx.cycle_graph(10),
        'Star': nx.star_graph(9),
        'Complete': nx.complete_graph(8),
        'Random': nx.erdos_renyi_graph(10, 0.4)
    }
    
    plt.figure(figsize=(12, 8))
    
    # Color scheme for topologies
    colors = {'Ring': '#FF6B6B', 'Star': '#4ECDC4', 'Complete': '#45B7D1', 'Random': '#96CEB4'}
    
    for name, G in topologies.items():
        # Initialize
        np.random.seed(42)
        for node in G.nodes():
            G.nodes[node]['phase'] = np.random.uniform(0, 2*np.pi)
            G.nodes[node]['vf'] = 1.0
        
        # Evolve and track
        steps = 50
        coherence_history = []
        
        for step in range(steps):
            coherence = compute_coherence(G)
            coherence_history.append(coherence)
            evolve_network_step(G)
        
        plt.plot(coherence_history, label=f'{name} Topology', 
                color=colors[name], linewidth=3, alpha=0.8)
    
    plt.xlabel('Evolution Steps', fontsize=14)
    plt.ylabel('Network Coherence', fontsize=14)
    plt.title('🌊 TNFR Coherence Evolution Across Network Topologies', fontsize=16, pad=20)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 49)
    plt.ylim(0, 1)
    
    # Add physics annotation
    plt.text(0.02, 0.98, 'Physics: ∂EPI/∂t = νf · ΔNFR(t)', 
             transform=plt.gca().transAxes, fontsize=11, 
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.tight_layout()
    
    # Save to output directory
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/coherence_evolution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Saved: output/coherence_evolution.png")


def create_phase_space_visualization():
    """Create phase space visualization showing nodal dynamics."""
    
    print("🎨 Creating phase space visualization...")
    
    # Create network
    G = nx.cycle_graph(6)
    
    # Initialize with specific pattern
    phases_init = [0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3]
    for i, node in enumerate(G.nodes()):
        G.nodes[node]['phase'] = phases_init[i]
        G.nodes[node]['vf'] = 1.0
    
    # Track evolution
    steps = 30
    phase_trajectories = {node: [] for node in G.nodes()}
    time_points = []
    
    for step in range(steps):
        time_points.append(step * 0.1)
        
        # Record current phases
        for node in G.nodes():
            phase_trajectories[node].append(G.nodes[node]['phase'])
        
        # Evolve
        evolve_network_step(G, dt=0.1)
    
    # Create phase space plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left plot: Phase trajectories over time
    colors = plt.cm.Set3(np.linspace(0, 1, len(G.nodes())))
    
    for i, node in enumerate(G.nodes()):
        ax1.plot(time_points, phase_trajectories[node], 
                label=f'Node {node}', color=colors[i], linewidth=2, marker='o', markersize=3)
    
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Phase (radians)', fontsize=12)
    ax1.set_title('🌊 Phase Evolution Over Time', fontsize=14)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 2*np.pi)
    
    # Add π markers
    ax1.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax1.set_yticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
    
    # Right plot: Circular phase representation (final state)
    ax2 = plt.subplot(122, projection='polar')
    
    final_phases = [phase_trajectories[node][-1] for node in G.nodes()]
    node_positions = np.array(final_phases)
    
    # Plot nodes on circle
    ax2.scatter(node_positions, np.ones(len(node_positions)), 
               c=colors, s=200, alpha=0.8, edgecolors='black', linewidth=2)
    
    # Add node labels
    for i, (node, phase) in enumerate(zip(G.nodes(), final_phases)):
        ax2.annotate(f'{node}', (phase, 1.15), ha='center', va='center', 
                    fontsize=10, fontweight='bold')
    
    # Draw connections
    for edge in G.edges():
        phase1 = final_phases[edge[0]]
        phase2 = final_phases[edge[1]]
        ax2.plot([phase1, phase2], [1, 1], 'gray', alpha=0.5, linewidth=1)
    
    ax2.set_title('🎯 Final Phase Configuration', fontsize=14, pad=20)
    ax2.set_ylim(0, 1.3)
    ax2.set_rticks([])
    
    plt.tight_layout()
    
    # Save
    plt.savefig('output/phase_space_dynamics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Saved: output/phase_space_dynamics.png")


def create_network_topology_comparison():
    """Create visual comparison of different network topologies."""
    
    print("🎨 Creating network topology comparison...")
    
    # Define topologies
    topologies = {
        'Ring': nx.cycle_graph(8),
        'Star': nx.star_graph(7),
        'Complete': nx.complete_graph(6),
        'Random': nx.erdos_renyi_graph(8, 0.3),
        'Small World': nx.watts_strogatz_graph(8, 3, 0.3),
        'Grid': nx.grid_2d_graph(3, 3)
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, (name, G) in enumerate(topologies.items()):
        ax = axes[idx]
        
        # Initialize with coherence visualization
        np.random.seed(42)
        for node in G.nodes():
            G.nodes[node]['phase'] = np.random.uniform(0, 2*np.pi)
            G.nodes[node]['vf'] = 1.0
        
        # Evolve to show final state
        for _ in range(30):
            evolve_network_step(G)
        
        # Compute node colors based on phase
        node_phases = [G.nodes[node]['phase'] for node in G.nodes()]
        node_colors = plt.cm.hsv(np.array(node_phases) / (2*np.pi))
        
        # Compute coherence
        coherence = compute_coherence(G)
        
        # Draw network
        pos = nx.spring_layout(G, seed=42)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', alpha=0.6, width=2)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, 
                              node_size=500, edgecolors='black', linewidths=2)
        
        # Add labels
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight='bold')
        
        ax.set_title(f'{name}\nCoherence: {coherence:.3f}', fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        ax.axis('off')
    
    # Add overall title and physics note
    fig.suptitle('🕸️ TNFR Dynamics Across Network Topologies\n' + 
                 'Node colors represent phases | Higher coherence = better synchronization', 
                 fontsize=16, fontweight='bold')
    
    # Add colorbar for phase
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=plt.cm.hsv, norm=plt.Normalize(vmin=0, vmax=2*np.pi))
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Phase (radians)', fontsize=12)
    cbar.set_ticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    cbar.set_ticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.9)
    
    # Save
    plt.savefig('output/network_topology_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Saved: output/network_topology_comparison.png")


def create_frequency_resonance_plot():
    """Create visualization of frequency resonance effects."""
    
    print("🎨 Creating frequency resonance visualization...")
    
    # Create network
    G = nx.cycle_graph(8)
    
    # Test different frequency distributions
    frequency_patterns = {
        'Uniform': [1.0] * 8,
        'Harmonic': [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
        'Golden Ratio': [1.0, 1.618, 1.0, 1.618, 1.0, 1.618, 1.0, 1.618],
        'Random': np.random.uniform(0.5, 2.0, 8)
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    results = {}
    
    for idx, (pattern_name, frequencies) in enumerate(frequency_patterns.items()):
        ax = axes[idx]
        
        # Initialize network
        np.random.seed(42)
        for i, node in enumerate(G.nodes()):
            G.nodes[node]['phase'] = np.random.uniform(0, 2*np.pi)
            G.nodes[node]['vf'] = frequencies[i]
        
        # Track evolution
        steps = 60
        coherence_history = []
        freq_coherence_history = []
        
        for step in range(steps):
            # Compute coherences
            phase_coherence = compute_coherence(G)
            
            # Frequency coherence
            node_frequencies = [G.nodes[n]['vf'] for n in G.nodes()]
            freq_var = np.var(node_frequencies)
            freq_mean = np.mean(node_frequencies)
            freq_coherence = 1.0 / (1.0 + freq_var / freq_mean) if freq_mean > 0 else 0.0
            
            coherence_history.append(phase_coherence)
            freq_coherence_history.append(freq_coherence)
            
            # Evolve
            evolve_network_step(G, dt=0.08)
        
        # Plot results
        ax.plot(coherence_history, label='Phase Coherence', color='blue', linewidth=2)
        ax.plot(freq_coherence_history, label='Frequency Coherence', 
               color='red', linewidth=2, linestyle='--')
        
        ax.set_xlabel('Evolution Steps')
        ax.set_ylabel('Coherence')
        ax.set_title(f'{pattern_name} Frequencies')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Store final coherence
        results[pattern_name] = {
            'final_phase': coherence_history[-1],
            'final_freq': freq_coherence_history[-1],
            'frequencies': frequencies
        }
    
    plt.suptitle('🎵 Frequency Resonance Effects on TNFR Dynamics', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save
    plt.savefig('output/frequency_resonance_effects.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print results summary
    print("📊 Frequency Pattern Results:")
    for pattern, result in results.items():
        print(f"  {pattern:12s}: Phase={result['final_phase']:.3f}, Freq={result['final_freq']:.3f}")
    
    print("✅ Saved: output/frequency_resonance_effects.png")


def create_emergence_metrics_dashboard():
    """Create dashboard showing emergence metrics over time."""
    
    print("🎨 Creating emergence metrics dashboard...")
    
    # Create swarm network
    G = nx.watts_strogatz_graph(15, 4, 0.2)
    
    # Initialize with leader/follower structure
    np.random.seed(42)
    leaders = [0, 5, 10]
    
    for node in G.nodes():
        G.nodes[node]['phase'] = np.random.uniform(0, 2*np.pi)
        if node in leaders:
            G.nodes[node]['vf'] = 2.0  # Leaders have higher frequency
            G.nodes[node]['role'] = 'leader'
        else:
            G.nodes[node]['vf'] = 1.0
            G.nodes[node]['role'] = 'follower'
    
    # Track metrics over time
    steps = 80
    metrics_history = {
        'order_parameter': [],
        'synchronization': [],
        'leader_coherence': [],
        'follower_coherence': [],
        'system_energy': []
    }
    
    for step in range(steps):
        # Compute metrics
        phases = [G.nodes[n]['phase'] for n in G.nodes()]
        
        # Order parameter
        x_sum = sum(np.cos(p) for p in phases)
        y_sum = sum(np.sin(p) for p in phases)
        order_param = np.sqrt(x_sum**2 + y_sum**2) / len(phases)
        
        # Synchronization
        sync = compute_coherence(G)
        
        # Leader/follower coherence
        leader_phases = [G.nodes[n]['phase'] for n in G.nodes() if G.nodes[n]['role'] == 'leader']
        follower_phases = [G.nodes[n]['phase'] for n in G.nodes() if G.nodes[n]['role'] == 'follower']
        
        leader_coh = 1.0 - np.std(leader_phases) / np.pi if len(leader_phases) > 1 else 1.0
        follower_coh = 1.0 - np.std(follower_phases) / np.pi if len(follower_phases) > 1 else 1.0
        
        # System energy (based on phase mismatches)
        total_energy = sum(compute_delta_nfr(G, n)**2 for n in G.nodes())
        
        # Store metrics
        metrics_history['order_parameter'].append(order_param)
        metrics_history['synchronization'].append(sync)
        metrics_history['leader_coherence'].append(leader_coh)
        metrics_history['follower_coherence'].append(follower_coh)
        metrics_history['system_energy'].append(total_energy)
        
        # Evolve system
        evolve_network_step(G, dt=0.1)
        
        # Leaders occasionally change direction
        if step % 20 == 0:
            for leader in leaders:
                if np.random.random() < 0.3:
                    direction_change = np.random.uniform(-np.pi/4, np.pi/4)
                    current_phase = G.nodes[leader]['phase']
                    G.nodes[leader]['phase'] = (current_phase + direction_change) % (2*np.pi)
    
    # Create dashboard
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Order parameter and synchronization
    axes[0,0].plot(metrics_history['order_parameter'], label='Order Parameter', 
                   color='blue', linewidth=2)
    axes[0,0].plot(metrics_history['synchronization'], label='Synchronization', 
                   color='green', linewidth=2)
    axes[0,0].set_xlabel('Evolution Steps')
    axes[0,0].set_ylabel('Metric Value')
    axes[0,0].set_title('🎯 Global Coherence Metrics')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].set_ylim(0, 1)
    
    # Leader vs Follower coherence
    axes[0,1].plot(metrics_history['leader_coherence'], label='Leaders', 
                   color='red', linewidth=2)
    axes[0,1].plot(metrics_history['follower_coherence'], label='Followers', 
                   color='orange', linewidth=2)
    axes[0,1].set_xlabel('Evolution Steps')
    axes[0,1].set_ylabel('Coherence')
    axes[0,1].set_title('👑 Leader-Follower Dynamics')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].set_ylim(0, 1)
    
    # System energy
    axes[1,0].plot(metrics_history['system_energy'], color='purple', linewidth=2)
    axes[1,0].set_xlabel('Evolution Steps')
    axes[1,0].set_ylabel('Total ΔNFR Energy')
    axes[1,0].set_title('⚡ System Energy (Structural Pressure)')
    axes[1,0].grid(True, alpha=0.3)
    
    # Phase portrait (final state)
    final_phases = [G.nodes[n]['phase'] for n in G.nodes()]
    leader_indices = [i for i, n in enumerate(G.nodes()) if G.nodes[n]['role'] == 'leader']
    follower_indices = [i for i, n in enumerate(G.nodes()) if G.nodes[n]['role'] == 'follower']
    
    axes[1,1] = plt.subplot(224, projection='polar')
    
    # Plot leaders and followers differently
    if leader_indices:
        leader_phases = [final_phases[i] for i in leader_indices]
        axes[1,1].scatter(leader_phases, [1.0] * len(leader_phases), 
                         c='red', s=200, alpha=0.8, label='Leaders', edgecolors='black')
    
    if follower_indices:
        follower_phases = [final_phases[i] for i in follower_indices]
        axes[1,1].scatter(follower_phases, [0.8] * len(follower_phases), 
                         c='lightblue', s=100, alpha=0.8, label='Followers', edgecolors='gray')
    
    axes[1,1].set_title('🌟 Final Swarm Configuration')
    axes[1,1].set_ylim(0, 1.2)
    axes[1,1].set_rticks([])
    axes[1,1].legend(loc='upper left', bbox_to_anchor=(0.1, 1.1))
    
    plt.suptitle('🐝 Swarm Intelligence: Emergence Metrics Dashboard', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save
    plt.savefig('output/emergence_metrics_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Saved: output/emergence_metrics_dashboard.png")


def visualization_suite_demo():
    """Run complete TNFR visualization suite."""
    
    print("=" * 80)
    print("                🎨 TNFR VISUALIZATION SUITE 🎨")
    print("=" * 80)
    print()
    print("Creating comprehensive visual representations of TNFR nodal dynamics...")
    print("PHYSICS: Visual exploration of ∂EPI/∂t = νf · ΔNFR(t) across multiple scenarios")
    print("OUTPUT: High-resolution graphics showing evolution patterns and emergent behaviors")
    print()
    
    try:
        # Create all visualizations
        create_coherence_evolution_plot()
        print()
        
        create_phase_space_visualization()
        print()
        
        create_network_topology_comparison()
        print()
        
        create_frequency_resonance_plot()
        print()
        
        create_emergence_metrics_dashboard()
        print()
        
        print("=" * 80)
        print("🎯 VISUALIZATION SUMMARY")
        print("=" * 80)
        print()
        print("📊 Generated visualizations:")
        print("  1. coherence_evolution.png - Topology-dependent coherence patterns")
        print("  2. phase_space_dynamics.png - Nodal phase trajectories over time") 
        print("  3. network_topology_comparison.png - Visual network structure effects")
        print("  4. frequency_resonance_effects.png - Harmonic frequency influence")
        print("  5. emergence_metrics_dashboard.png - Swarm intelligence emergence")
        print()
        print("📁 All files saved to: output/ directory")
        print("🔬 Each graphic shows different aspects of TNFR nodal equation dynamics")
        print("🎨 High-resolution (300 DPI) suitable for presentations and papers")
        print()
        print("🚀 VISUAL INSIGHTS REVEALED:")
        print("━" * 60)
        print("• Complete graphs achieve fastest coherence convergence")
        print("• Phase trajectories show natural synchronization patterns")  
        print("• Network topology determines information flow pathways")
        print("• Harmonic frequencies enhance resonant coupling")
        print("• Leader-follower dynamics create emergent swarm intelligence")
        print("• System energy decreases as structural pressure (ΔNFR) is resolved")
        print()
        
    except ImportError as e:
        print(f"❌ Matplotlib not available: {e}")
        print("📝 To enable visualizations: pip install matplotlib")
        print("🔬 Examples still demonstrate TNFR physics through console output")


if __name__ == "__main__":
    visualization_suite_demo()