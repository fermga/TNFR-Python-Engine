"""08 - Emergent Phenomena: TNFR Collective Behaviors

Exploration of emergent collective behaviors arising from TNFR nodal dynamics.

PHYSICS: Demonstrates how individual nodal equations create system-level phenomena.
LEARNING: Understanding emergence, collective intelligence, and macro-scale patterns.
"""

import networkx as nx
import numpy as np


def compute_coherence(G):
    """Network phase synchronization: the canonical Kuramoto order
    parameter R = |<e^{iθ}>|.

    R = 1 when phases are fully aligned, R -> 0 when desynchronized
    (random or antiphase). AGENTS.md frames TNFR phase coupling as
    Kuramoto synchronization, so this is the canonical phase-synchrony
    measure. The distinct total coherence
    C(t) = 1/(1 + mean|ΔNFR| + mean|dEPI|) lives in
    tnfr.metrics.coherence and requires the dynamics pipeline.
    """
    thetas = np.array(
        [G.nodes[n].get('theta', G.nodes[n].get('phase', 0.0))
         for n in G.nodes()],
        dtype=float,
    )
    if thetas.size == 0:
        return 1.0
    return float(abs(np.mean(np.exp(1j * thetas))))


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


def detect_clusters(G, threshold=0.5):
    """Detect coherent clusters in the network."""
    clusters = []
    nodes = list(G.nodes())
    visited = set()
    
    for node in nodes:
        if node in visited:
            continue
        
        # Find nodes with similar phases
        cluster = [node]
        visited.add(node)
        node_phase = G.nodes[node]['phase']
        
        for other_node in nodes:
            if other_node in visited:
                continue
            
            other_phase = G.nodes[other_node]['phase']
            phase_diff = abs(node_phase - other_phase)
            phase_diff = min(phase_diff, 2 * np.pi - phase_diff)
            
            if phase_diff < threshold:
                cluster.append(other_node)
                visited.add(other_node)
        
        if len(cluster) > 1:  # Only count multi-node clusters
            clusters.append(cluster)
    
    return clusters


def compute_emergence_metrics(G):
    """Compute metrics that indicate emergent behavior."""
    
    # 1. Order parameter (global phase coherence)
    phases = [G.nodes[n]['phase'] for n in G.nodes()]
    x_sum = sum(np.cos(p) for p in phases)
    y_sum = sum(np.sin(p) for p in phases)
    order_parameter = np.sqrt(x_sum**2 + y_sum**2) / len(phases)
    
    # 2. Clustering coefficient
    try:
        clustering = nx.average_clustering(G)
    except:
        clustering = 0.0
    
    # 3. Synchronization index
    sync_index = compute_coherence(G)
    
    # 4. Information integration (simplified Φ)
    # Measure how much information the whole system has vs parts
    clusters = detect_clusters(G, threshold=np.pi/4)
    if clusters:
        cluster_coherences = []
        for cluster in clusters:
            if len(cluster) >= 2:
                cluster_phases = [G.nodes[n]['phase'] for n in cluster]
                cluster_diffs = []
                for i in range(len(cluster_phases)):
                    for j in range(i + 1, len(cluster_phases)):
                        diff = abs(cluster_phases[i] - cluster_phases[j])
                        diff = min(diff, 2 * np.pi - diff)
                        cluster_diffs.append(diff)
                cluster_coh = 1.0 - (np.mean(cluster_diffs) / np.pi)
                cluster_coherences.append(cluster_coh)
        
        if cluster_coherences:
            avg_cluster_coherence = np.mean(cluster_coherences)
            integration = sync_index - avg_cluster_coherence  # Whole vs parts
        else:
            integration = sync_index
    else:
        integration = sync_index
    
    # 5. Complexity (balance between order and disorder)
    frequencies = [G.nodes[n]['nu_f'] for n in G.nodes()]
    freq_entropy = -sum(f * np.log(f + 1e-10) for f in frequencies) / len(frequencies)
    complexity = sync_index * freq_entropy  # Order × Diversity
    
    return {
        'order_parameter': order_parameter,
        'clustering': clustering,
        'synchronization': sync_index,
        'integration': integration,
        'complexity': complexity,
        'num_clusters': len(clusters)
    }


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


def swarm_intelligence_demo():
    """Demonstrate swarm intelligence emergence from individual agents."""
    
    print("🐝 SWARM INTELLIGENCE EMERGENCE")
    print("━" * 50)
    
    # Create swarm network (small-world for local + global connections)
    G = nx.watts_strogatz_graph(20, 4, 0.3)
    
    # Initialize agents with diverse behaviors
    np.random.seed(42)
    for node in G.nodes():
        G.nodes[node]['phase'] = np.random.uniform(0, 2*np.pi)
        G.nodes[node]['nu_f'] = np.random.uniform(0.5, 2.0)  # Diverse frequencies
        G.nodes[node]['role'] = 'follower'
    
    # Designate some nodes as "leaders" with higher frequencies
    leaders = np.random.choice(list(G.nodes()), size=3, replace=False)
    for leader in leaders:
        G.nodes[leader]['nu_f'] = 2.5  # Higher frequency = more influential
        G.nodes[leader]['role'] = 'leader'
    
    print(f"Swarm configuration:")
    print(f"  Agents: {G.number_of_nodes()}")
    print(f"  Leaders: {len(leaders)}")
    print(f"  Connections: {G.number_of_edges()}")
    
    # Initial measurements
    initial_metrics = compute_emergence_metrics(G)
    print(f"\nInitial emergence metrics:")
    for metric, value in initial_metrics.items():
        print(f"  {metric}: {value:.3f}")
    
    # Evolution simulation
    print(f"\n🌊 Swarm evolution:")
    
    metrics_history = []
    
    for step in range(60):
        evolve_network_step(G)
        
        # Leaders occasionally change direction (exploration)
        if step % 15 == 0:
            for leader in leaders:
                if np.random.random() < 0.4:  # 40% chance
                    direction_change = np.random.uniform(-np.pi/3, np.pi/3)
                    current_phase = G.nodes[leader]['phase']
                    G.nodes[leader]['phase'] = (current_phase + direction_change) % (2*np.pi)
        
        metrics = compute_emergence_metrics(G)
        metrics_history.append(metrics)
    
    # Final measurements
    final_metrics = compute_emergence_metrics(G)
    print(f"\nFinal emergence metrics:")
    for metric, value in final_metrics.items():
        improvement = value - initial_metrics[metric]
        print(f"  {metric}: {value:.3f} ({improvement:+.3f})")
    
    # Analyze swarm behavior
    print(f"\n🧮 Swarm intelligence analysis:")
    
    # Check for collective decision making
    final_order = final_metrics['order_parameter']
    if final_order > 0.8:
        print("  ✨ Strong collective coherence achieved")
    elif final_order > 0.6:
        print("  🎯 Moderate collective coordination")
    else:
        print("  🌀 Distributed individual behaviors")
    
    # Check for leader-follower dynamics
    leader_phases = [G.nodes[leader]['phase'] for leader in leaders]
    follower_phases = [G.nodes[node]['phase'] for node in G.nodes() 
                      if G.nodes[node]['role'] == 'follower']
    
    if leader_phases and follower_phases:
        leader_coherence = 1.0 - np.std(leader_phases) / np.pi
        follower_coherence = 1.0 - np.std(follower_phases) / np.pi
        
        print(f"  Leader coherence: {leader_coherence:.3f}")
        print(f"  Follower coherence: {follower_coherence:.3f}")
        
        if leader_coherence > follower_coherence + 0.1:
            print("  👑 Leaders maintain distinct coordination")
        else:
            print("  🤝 Leaders and followers converged")


def consensus_formation_demo():
    """Demonstrate consensus formation in opinion networks."""
    
    print("\n🗳️ CONSENSUS FORMATION")
    print("━" * 50)
    
    # Create opinion network (random graph)
    G = nx.erdos_renyi_graph(15, 0.4)
    
    # Initialize with polarized opinions (phases represent opinions)
    np.random.seed(123)
    for node in G.nodes():
        # Create two opinion clusters initially
        if np.random.random() < 0.5:
            G.nodes[node]['phase'] = np.random.normal(0.5, 0.3) % (2*np.pi)  # Cluster 1
        else:
            G.nodes[node]['phase'] = np.random.normal(4.0, 0.3) % (2*np.pi)  # Cluster 2
        
        G.nodes[node]['nu_f'] = np.random.uniform(0.8, 1.5)  # Varying influence
        G.nodes[node]['conviction'] = np.random.uniform(0.3, 1.0)  # Opinion strength
    
    # Identify initial opinion clusters
    initial_clusters = detect_clusters(G, threshold=1.0)
    print(f"Initial opinion clusters: {len(initial_clusters)}")
    for i, cluster in enumerate(initial_clusters):
        avg_phase = np.mean([G.nodes[n]['phase'] for n in cluster])
        print(f"  Cluster {i+1}: {len(cluster)} agents, opinion {avg_phase:.2f}")
    
    initial_metrics = compute_emergence_metrics(G)
    print(f"Initial consensus level: {initial_metrics['synchronization']:.3f}")
    
    # Evolution with opinion dynamics
    print(f"\n💭 Opinion evolution:")
    
    consensus_history = []
    
    for step in range(80):
        # Modified evolution with conviction weighting
        new_phases = {}
        
        for node in G.nodes():
            current_phase = G.nodes[node]['phase']
            vf = G.nodes[node]['nu_f']
            conviction = G.nodes[node]['conviction']
            neighbors = list(G.neighbors(node))
            
            if neighbors:
                # Weight neighbor opinions by their conviction
                weighted_influences = []
                for neighbor in neighbors:
                    neighbor_phase = G.nodes[neighbor]['phase']
                    neighbor_conviction = G.nodes[neighbor]['conviction']
                    weighted_influences.append(neighbor_phase * neighbor_conviction)
                
                if weighted_influences:
                    target_phase = np.mean(weighted_influences)
                    
                    direction = target_phase - current_phase
                    if direction > np.pi:
                        direction -= 2 * np.pi
                    elif direction < -np.pi:
                        direction += 2 * np.pi
                    
                    # Higher conviction = slower opinion change
                    resistance = conviction
                    delta_nfr = compute_delta_nfr(G, node)
                    
                    phase_change = (vf / resistance) * delta_nfr * 0.1 * np.sign(direction)
                    new_phases[node] = (current_phase + phase_change) % (2 * np.pi)
                else:
                    new_phases[node] = current_phase
            else:
                new_phases[node] = current_phase
        
        for node, phase in new_phases.items():
            G.nodes[node]['phase'] = phase
        
        # Track consensus formation
        metrics = compute_emergence_metrics(G)
        consensus_history.append(metrics['synchronization'])
    
    # Final consensus analysis
    final_clusters = detect_clusters(G, threshold=1.0)
    final_consensus = consensus_history[-1]
    
    print(f"Final consensus level: {final_consensus:.3f}")
    print(f"Final opinion clusters: {len(final_clusters)}")
    
    if len(final_clusters) == 1:
        print("  ✅ Full consensus achieved")
    elif len(final_clusters) < len(initial_clusters):
        print("  🔄 Partial consensus (cluster reduction)")
    else:
        print("  ❌ Consensus failed (polarization maintained)")
    
    # Analyze consensus trajectory
    if len(consensus_history) >= 20:
        early_consensus = np.mean(consensus_history[:20])
        late_consensus = np.mean(consensus_history[-20:])
        consensus_improvement = late_consensus - early_consensus
        
        print(f"  Consensus improvement: {consensus_improvement:+.3f}")
        
        if consensus_improvement > 0.3:
            print("  🚀 Strong consensus formation")
        elif consensus_improvement > 0.1:
            print("  📈 Gradual consensus building")
        else:
            print("  📊 Weak consensus dynamics")


def self_organization_demo():
    """Demonstrate spontaneous self-organization."""
    
    print("\n🌱 SPONTANEOUS SELF-ORGANIZATION")
    print("━" * 50)
    
    # Start with random network
    G = nx.erdos_renyi_graph(18, 0.3)
    
    # Initialize with maximum disorder
    np.random.seed(456)
    for node in G.nodes():
        G.nodes[node]['phase'] = np.random.uniform(0, 2*np.pi)
        G.nodes[node]['nu_f'] = np.random.uniform(0.1, 3.0)
        G.nodes[node]['organization_level'] = 0.0  # Track self-organization
    
    initial_metrics = compute_emergence_metrics(G)
    print(f"Initial state (maximum disorder):")
    print(f"  Order parameter: {initial_metrics['order_parameter']:.3f}")
    print(f"  Complexity: {initial_metrics['complexity']:.3f}")
    print(f"  Integration: {initial_metrics['integration']:.3f}")
    
    # Self-organization evolution
    print(f"\n🔄 Self-organization process:")
    
    organization_history = []
    
    for step in range(100):
        evolve_network_step(G, dt=0.08)
        
        # Measure organization level
        metrics = compute_emergence_metrics(G)
        organization_level = (metrics['order_parameter'] + 
                            metrics['synchronization'] + 
                            metrics['integration']) / 3.0
        
        organization_history.append(organization_level)
        
        # Update individual organization levels
        for node in G.nodes():
            delta_nfr = compute_delta_nfr(G, node)
            # Lower ΔNFR = higher local organization
            local_organization = 1.0 - min(delta_nfr, 1.0)
            G.nodes[node]['organization_level'] = local_organization
        
        # Adaptive frequency adjustment (self-tuning)
        if step % 20 == 0:
            for node in G.nodes():
                local_org = G.nodes[node]['organization_level']
                # Higher organization → more stable frequency
                if local_org > 0.7:
                    G.nodes[node]['nu_f'] *= 0.95  # Slightly reduce frequency
                elif local_org < 0.3:
                    G.nodes[node]['nu_f'] *= 1.05  # Slightly increase frequency
    
    # Final organization analysis
    final_metrics = compute_emergence_metrics(G)
    final_organization = organization_history[-1]
    initial_organization = organization_history[0]
    
    print(f"Final state:")
    print(f"  Order parameter: {final_metrics['order_parameter']:.3f}")
    print(f"  Complexity: {final_metrics['complexity']:.3f}")
    print(f"  Integration: {final_metrics['integration']:.3f}")
    print(f"  Organization improvement: {final_organization - initial_organization:+.3f}")
    
    # Check for self-organization success
    if final_organization > 0.7:
        print("  🌟 Strong self-organization achieved")
    elif final_organization > 0.5:
        print("  ✨ Moderate self-organization")
    else:
        print("  📊 Weak self-organization")
    
    # Analyze organization trajectory
    organization_trend = np.polyfit(range(len(organization_history)), organization_history, 1)[0]
    
    if organization_trend > 0.002:
        print("  📈 Consistent organization growth")
    elif organization_trend > 0:
        print("  📊 Gradual organization increase")
    else:
        print("  📉 Organization plateaued or declined")


def emergent_phenomena_demo():
    """Comprehensive demonstration of emergent TNFR phenomena."""
    
    print("=" * 80)
    print("                🌟 EMERGENT PHENOMENA & COLLECTIVE INTELLIGENCE 🌟")
    print("=" * 80)
    print()
    print("Exploring how individual nodal dynamics create system-level behaviors...")
    print("PHYSICS: ∂EPI/∂t = νf · ΔNFR at individual level → collective intelligence")
    print("INSIGHT: Emergence transcends individual components through coherent coupling")
    print()
    
    swarm_intelligence_demo()
    
    consensus_formation_demo()
    
    self_organization_demo()
    
    print("\n" + "=" * 80)
    print("🧮 EMERGENT PHENOMENA INSIGHTS")
    print("=" * 80)
    
    print("\n🌟 EMERGENCE PRINCIPLES:")
    print("━" * 60)
    print("• Individual nodal equations → collective system behavior")
    print("• Local coupling → global synchronization patterns")
    print("• Diverse frequencies → rich collective dynamics")
    print("• Phase alignment → coherent group behaviors")
    print("• Network topology → emergence pathway constraints")
    
    print("\n🐝 SWARM INTELLIGENCE MECHANISMS:")
    print("━" * 60)
    print("• Leader nodes with higher νf guide collective motion")
    print("• Follower adaptation through phase coupling")
    print("• Exploration via periodic leader direction changes")
    print("• Exploitation through coherence amplification")
    print("• Collective decision-making from individual interactions")
    
    print("\n🗳️ CONSENSUS FORMATION DYNAMICS:")
    print("━" * 60)
    print("• Opinion clusters emerge from initial diversity")
    print("• Conviction affects opinion change resistance")
    print("• Neighbor influence weighted by conviction strength")
    print("• Gradual cluster merging toward consensus")
    print("• Network structure affects consensus speed")
    
    print("\n🌱 SELF-ORGANIZATION FEATURES:")
    print("━" * 60)
    print("• Spontaneous order from maximum initial disorder")
    print("• Adaptive frequency tuning based on local organization")
    print("• Integration of information across network scales")
    print("• Complexity balance between order and diversity")
    print("• Persistent organization through structural stability")
    
    print("\n🔬 EMERGENCE METRICS:")
    print("━" * 60)
    print("• Order parameter: Global phase coherence measure")
    print("• Integration: Whole system vs parts information")
    print("• Complexity: Balance of order and diversity")
    print("• Synchronization: Phase alignment degree")
    print("• Clustering: Local coherence organization")
    
    print("\n🚀 ADVANCED EMERGENCE RESEARCH:")
    print("━" * 60)
    print("• Multi-level emergence across hierarchical scales")
    print("• Adaptive emergence with environmental feedback")
    print("• Emergence-guided network evolution")
    print("• Quantum-coherent collective states")
    print("• Consciousness emergence from neural TNFR networks")


if __name__ == "__main__":
    emergent_phenomena_demo()