"""02 - Musical Resonance: Understanding Coherence Through Sound

PHYSICS: Demonstrates phase synchronization φᵢ ≈ φⱼ as the foundation of resonance.
LEARNING: Understand how TNFR models harmony, dissonance, and musical structure.

This example shows why musicians intuitively understand TNFR - music IS coherent organization!
"""

import numpy as np
import networkx as nx
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
    phases = [G.nodes[n].get('theta', 0) for n in G.nodes()]
    if len(phases) < 2:
        return 1.0
    
    phase_diffs = []
    for i in range(len(phases)):
        for j in range(i + 1, len(phases)):
            diff = abs(phases[i] - phases[j])
            diff = min(diff, 2 * np.pi - diff)
            phase_diffs.append(diff)
    
    return 1.0 - (np.mean(phase_diffs) / np.pi) if phase_diffs else 1.0


def apply_resonance_evolution(G, steps=10):
    """Apply TNFR evolution to achieve resonance."""
    for _ in range(steps):
        new_phases = {}
        
        for node in G.nodes():
            current_phase = G.nodes[node].get('theta', 0)
            neighbors = list(G.neighbors(node))
            
            if neighbors:
                neighbor_phases = [G.nodes[n].get('theta', 0) for n in neighbors]
                target_phase = np.mean(neighbor_phases)
                
                # Direction to synchronize
                direction = target_phase - current_phase
                if direction > np.pi:
                    direction -= 2 * np.pi
                elif direction < -np.pi:
                    direction += 2 * np.pi
                
                # Apply gradual synchronization
                step_size = 0.2
                new_phases[node] = (current_phase + step_size * direction) % (2 * np.pi)
            else:
                new_phases[node] = current_phase
        
        # Update phases
        for node, phase in new_phases.items():
            G.nodes[node]['theta'] = phase


def create_musical_visualization(results):
    """Create visualization of musical resonance patterns."""
    
    print("🎨 Creating musical resonance visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Coherence comparison
    axes[0, 0].bar(results.keys(), [r['final_coherence'] for r in results.values()], 
                   color=['gold', 'lightblue', 'salmon'])
    axes[0, 0].set_ylabel('Coherence')
    axes[0, 0].set_title('Musical Coherence Comparison')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 1)
    
    # Phase evolution for harmony
    axes[0, 1].plot(results['Perfect Harmony']['evolution'], 
                    label='Perfect Harmony', color='gold', linewidth=3)
    axes[0, 1].plot(results['Musical Intervals']['evolution'], 
                    label='Musical Intervals', color='lightblue', linewidth=3)
    axes[0, 1].plot(results['Random Chaos']['evolution'], 
                    label='Random Chaos', color='salmon', linewidth=3)
    axes[0, 1].set_xlabel('Evolution Steps')
    axes[0, 1].set_ylabel('Coherence')
    axes[0, 1].set_title('Coherence Evolution Over Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Circular phase plots for final states
    for idx, (name, result) in enumerate(results.items()):
        if idx >= 2:  # Only show first two in circular plots
            break
        
        ax = plt.subplot(2, 2, 3 + idx, projection='polar')
        
        phases = result['final_phases']
        colors = plt.cm.hsv(np.array(phases) / (2 * np.pi))
        
        ax.scatter(phases, [1.0] * len(phases), c=colors, s=200, 
                   alpha=0.8, edgecolors='black', linewidth=2)
        
        # Add node labels
        for i, phase in enumerate(phases):
            ax.annotate(f'N{i+1}', (phase, 1.15), ha='center', va='center', 
                       fontsize=10, fontweight='bold')
        
        ax.set_title(f'{name}\nFinal Configuration')
        ax.set_ylim(0, 1.3)
        ax.set_rticks([])
    
    plt.suptitle('Musical Resonance in TNFR Networks\n'
                 'Phase Synchronization Creates Harmony', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/musical_resonance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Saved: output/musical_resonance.png")


def musical_resonance_demo():
    """Demonstrate TNFR through musical harmony and dissonance."""
    
    print("=" * 60)
    print(" " * 15 + "🎵 Musical Resonance Demo 🎵")
    print("=" * 60)
    print()
    print("Music is the perfect metaphor for TNFR!")
    print("Let's explore harmony and dissonance through coherent systems...")
    print()
    
    results = {}
    
    # EXPERIMENT 1: Perfect Harmony (In-Phase Resonance)
    print("🎯 EXPERIMENT 1: Perfect Harmony")
    print("   Theory: When phases align φᵢ ≈ φⱼ, we get constructive resonance")
    print()
    
    G = nx.cycle_graph(7)  # 7 notes like a musical scale
    
    # Set all nodes to similar phases (harmony)
    base_phase = 0.0
    for i, node in enumerate(G.nodes()):
        G.nodes[node]['theta'] = base_phase + i * 0.1  # Slight variation
        G.nodes[node]['nf'] = 1.0  # Standard frequency
    
    # Track evolution
    evolution = []
    for step in range(20):
        coherence = compute_coherence(G)
        evolution.append(coherence)
        apply_resonance_evolution(G, steps=1)
    
    final_coherence = compute_coherence(G)
    final_phases = [G.nodes[n]['theta'] for n in G.nodes()]
    
    results['Perfect Harmony'] = {
        'evolution': evolution,
        'final_coherence': final_coherence,
        'final_phases': final_phases
    }
    
    print(f"   📊 Harmonic coherence: {final_coherence:.3f}")
    print("   🎵 Result: Beautiful harmonic resonance!")
    print()
    
    # EXPERIMENT 2: Musical Intervals
    print("🎯 EXPERIMENT 2: Musical Intervals")
    print("   Theory: Phase differences create intervals, but still musical")
    print()
    
    G2 = nx.cycle_graph(7)
    
    # Create musical intervals
    musical_intervals = [0, 0.3, 0.7, 1.0, 1.4, 1.7, 2.0]
    for i, node in enumerate(G2.nodes()):
        G2.nodes[node]['theta'] = musical_intervals[i]
        G2.nodes[node]['nf'] = 1.0
    
    # Track evolution
    evolution2 = []
    for step in range(20):
        coherence = compute_coherence(G2)
        evolution2.append(coherence)
        apply_resonance_evolution(G2, steps=1)
    
    final_coherence2 = compute_coherence(G2)
    final_phases2 = [G2.nodes[n]['theta'] for n in G2.nodes()]
    
    results['Musical Intervals'] = {
        'evolution': evolution2,
        'final_coherence': final_coherence2,
        'final_phases': final_phases2
    }
    
    print(f"   📊 Interval coherence: {final_coherence2:.3f}")
    print("   🎼 Result: Rich harmonic intervals!")
    print()
    
    # EXPERIMENT 3: Random Chaos
    print("🎯 EXPERIMENT 3: Random Chaos")
    print("   Theory: Random phases φᵢ ≈ random → destructive interference")
    print()
    
    G3 = nx.cycle_graph(7)
    
    # Set random, incompatible phases
    np.random.seed(42)
    for node in G3.nodes():
        G3.nodes[node]['theta'] = np.random.uniform(0, 2 * np.pi)
        G3.nodes[node]['nf'] = 1.0
    
    # Track evolution
    evolution3 = []
    for step in range(20):
        coherence = compute_coherence(G3)
        evolution3.append(coherence)
        apply_resonance_evolution(G3, steps=1)
    
    final_coherence3 = compute_coherence(G3)
    final_phases3 = [G3.nodes[n]['theta'] for n in G3.nodes()]
    
    results['Random Chaos'] = {
        'evolution': evolution3,
        'final_coherence': final_coherence3,
        'final_phases': final_phases3
    }
    
    print(f"   📊 Chaotic coherence: {final_coherence3:.3f}")
    print("   💥 Result: Even chaos finds some order through TNFR!")
    print()
    
    # Create visualization
    create_musical_visualization(results)
    
    # PHYSICS DEEP DIVE
    print("🧮 PHYSICS EXPLANATION:")
    print("=" * 60)
    print()
    print("1. PHASE SYNCHRONIZATION:")
    print("   • Musical harmony ≡ Phase alignment (φᵢ ≈ φⱼ)")
    print("   • Dissonance ≡ Phase differences (|φᵢ - φⱼ| > threshold)")
    print("   • Evolution: ∂θ/∂t = coupling × phase_difference")
    print()
    print("2. CONSTRUCTIVE vs DESTRUCTIVE RESONANCE:")
    print("   • Constructive: Aligned phases amplify each other")
    print("   • Destructive: Misaligned phases cancel each other")
    print("   • Music uses BOTH for emotional expression!")
    print()
    print("3. NODAL EQUATION IN MUSIC:")
    print("   • EPI = musical phrase structure")
    print("   • νf = tempo/rhythm (reorganization rate)")
    print("   • ΔNFR = harmonic tension (drive to resolve)")
    print()
    print("4. WHY MUSICIANS 'GET' TNFR:")
    print("   • Music IS coherent organization in time")
    print("   • Composers intuitively use resonance principles")
    print("   • Harmony/dissonance = constructive/destructive coherence")
    print()
    
    # COMPARATIVE ANALYSIS
    print("📈 COHERENCE COMPARISON:")
    for name, result in results.items():
        print(f"   🎵 {name:17s}: {result['final_coherence']:.3f}")
    print()
    print("💡 INSIGHT: Musical beauty emerges from the same physics")
    print("   that governs atoms, cells, and galaxies!")
    print()
    
    # NEXT STEPS
    print("🚀 NEXT STEPS:")
    print("   • Try: python 03_simple_network.py")
    print("   • Explore: How does this apply to other domains?")
    print("   • Think: What other systems show harmony/dissonance patterns?")
    print()
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    musical_resonance_demo()