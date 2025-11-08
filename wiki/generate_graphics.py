#!/usr/bin/env python3
"""
Generate visual graphics for the TNFR wiki.
Creates diagrams illustrating key TNFR concepts.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path

# Set output directory
OUTPUT_DIR = Path(__file__).parent / "images"
OUTPUT_DIR.mkdir(exist_ok=True)

# Use a clean style
plt.style.use('default')


def create_network_resonance_diagram():
    """Create a diagram showing NFR nodes in a resonant network."""
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
    
    # Define node positions
    nodes = {
        'A': (2, 7),
        'B': (5, 8),
        'C': (8, 7),
        'D': (3, 4),
        'E': (7, 4),
        'F': (5, 1.5)
    }
    
    # Define connections (edges)
    edges = [
        ('A', 'B'), ('B', 'C'), ('A', 'D'),
        ('C', 'E'), ('D', 'E'), ('D', 'F'), ('E', 'F')
    ]
    
    # Draw edges with varying thickness (representing coupling strength)
    for i, (n1, n2) in enumerate(edges):
        x1, y1 = nodes[n1]
        x2, y2 = nodes[n2]
        # Alternate line styles for visual interest
        lw = 2 + (i % 2)
        ax.plot([x1, x2], [y1, y2], 'b-', alpha=0.3, linewidth=lw, zorder=1)
    
    # Draw nodes with different states
    for name, (x, y) in nodes.items():
        # Outer circle (phase indicator)
        circle_outer = patches.Circle((x, y), 0.5, 
                                     facecolor='#4a90e2', 
                                     edgecolor='#2c5aa0',
                                     linewidth=2, 
                                     alpha=0.7,
                                     zorder=2)
        ax.add_patch(circle_outer)
        
        # Inner circle (EPI core)
        circle_inner = patches.Circle((x, y), 0.25, 
                                     facecolor='#ffffff', 
                                     edgecolor='#2c5aa0',
                                     linewidth=1.5,
                                     zorder=3)
        ax.add_patch(circle_inner)
        
        # Add node label
        ax.text(x, y-1, f'NFR {name}', 
               ha='center', va='top',
               fontsize=10, fontweight='bold',
               color='#333333')
        
        # Add frequency annotation for some nodes
        if name in ['A', 'C', 'F']:
            ax.text(x+0.7, y+0.3, f'νf={np.random.randint(10,50)}Hz', 
                   ha='left', va='center',
                   fontsize=7, style='italic',
                   color='#666666')
    
    # Add title and annotations
    ax.text(5, 9.5, 'TNFR Network: Resonant Fractal Nodes', 
           ha='center', fontsize=16, fontweight='bold',
           color='#2c5aa0')
    
    # Add legend
    legend_y = 0.5
    ax.plot([0.5, 1.5], [legend_y, legend_y], 'b-', alpha=0.3, linewidth=2)
    ax.text(1.8, legend_y, 'Coupling (phase-synchronized)', 
           va='center', fontsize=9, color='#666666')
    
    circle_legend = patches.Circle((1, legend_y-0.8), 0.3, 
                                  facecolor='#4a90e2', 
                                  edgecolor='#2c5aa0',
                                  linewidth=2, alpha=0.7)
    ax.add_patch(circle_legend)
    ax.text(1.8, legend_y-0.8, 'NFR Node (EPI + νf + φ)', 
           va='center', fontsize=9, color='#666666')
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'network_resonance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created network_resonance.png")


def create_structural_operators_diagram():
    """Create a visual reference for the 13 structural operators."""
    fig, axes = plt.subplots(3, 5, figsize=(15, 10), facecolor='white')
    axes = axes.flatten()
    
    operators = [
        ('Emission\n(AL)', '♪', '#ff6b6b'),
        ('Reception\n(EN)', '⟲', '#4ecdc4'),
        ('Coherence\n(IL)', '◉', '#45b7d1'),
        ('Dissonance\n(OZ)', '⚡', '#f38181'),
        ('Coupling\n(UM)', '⟷', '#aa96da'),
        ('Resonance\n(RA)', '≋', '#5c7cfa'),
        ('Silence\n(SHA)', '○', '#95afc0'),
        ('Expansion\n(VAL)', '↗', '#38ada9'),
        ('Contraction\n(NUL)', '↘', '#ee5a6f'),
        ('Self-org\n(THOL)', '❋', '#26de81'),
        ('Mutation\n(ZHIR)', '※', '#fd79a8'),
        ('Transition\n(NAV)', '→', '#fdcb6e'),
        ('Recursivity\n(REMESH)', '↻', '#6c5ce7'),
    ]
    
    for idx, (name, symbol, color) in enumerate(operators):
        ax = axes[idx]
        
        # Create operator box
        rect = patches.FancyBboxPatch((0.1, 0.2), 0.8, 0.6,
                                      boxstyle="round,pad=0.05",
                                      facecolor=color,
                                      edgecolor='#333333',
                                      linewidth=2,
                                      alpha=0.7)
        ax.add_patch(rect)
        
        # Add symbol
        ax.text(0.5, 0.65, symbol, ha='center', va='center',
               fontsize=48, fontweight='bold', color='white')
        
        # Add operator name
        ax.text(0.5, 0.35, name, ha='center', va='center',
               fontsize=10, fontweight='bold',
               color='white',
               bbox=dict(boxstyle='round', facecolor='#333333', alpha=0.8, pad=0.3))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    # Hide extra subplots
    for idx in range(len(operators), len(axes)):
        axes[idx].axis('off')
    
    # Add title
    fig.suptitle('13 Canonical Structural Operators', 
                fontsize=20, fontweight='bold', y=0.98,
                color='#2c5aa0')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUTPUT_DIR / 'structural_operators.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created structural_operators.png")


def create_paradigm_comparison():
    """Create a comparison chart between traditional and TNFR paradigms."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8), facecolor='white')
    
    # Traditional paradigm (left)
    ax1.text(0.5, 0.95, 'Traditional Paradigm', 
            ha='center', fontsize=16, fontweight='bold',
            color='#666666')
    
    traditional_concepts = [
        ('Objects exist\nindependently', 0.80),
        ('Causality:\nA causes B', 0.65),
        ('Static\nrepresentations', 0.50),
        ('Observer watches\nfrom outside', 0.35),
        ('Domain-specific\nmodels', 0.20),
    ]
    
    for text, y_pos in traditional_concepts:
        rect = patches.FancyBboxPatch((0.15, y_pos-0.05), 0.7, 0.08,
                                      boxstyle="round,pad=0.01",
                                      facecolor='#e0e0e0',
                                      edgecolor='#999999',
                                      linewidth=1.5)
        ax1.add_patch(rect)
        ax1.text(0.5, y_pos, text, ha='center', va='center',
                fontsize=10, color='#333333')
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    
    # TNFR paradigm (right)
    ax2.text(0.5, 0.95, 'TNFR Paradigm', 
            ha='center', fontsize=16, fontweight='bold',
            color='#2c5aa0')
    
    tnfr_concepts = [
        ('Patterns exist through\nresonance', 0.80),
        ('Coherence:\nA and B co-organize', 0.65),
        ('Dynamic\nreorganization', 0.50),
        ('Observer is a\nresonating node', 0.35),
        ('Trans-scale,\ntrans-domain', 0.20),
    ]
    
    for text, y_pos in tnfr_concepts:
        rect = patches.FancyBboxPatch((0.15, y_pos-0.05), 0.7, 0.08,
                                      boxstyle="round,pad=0.01",
                                      facecolor='#4a90e2',
                                      edgecolor='#2c5aa0',
                                      linewidth=2,
                                      alpha=0.7)
        ax2.add_patch(rect)
        ax2.text(0.5, y_pos, text, ha='center', va='center',
                fontsize=10, color='white', fontweight='bold')
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    # Add arrow between paradigms
    fig.text(0.5, 0.05, '→ Paradigm Shift →', 
            ha='center', fontsize=14, fontweight='bold',
            color='#2c5aa0')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'paradigm_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created paradigm_comparison.png")


def create_coherence_visualization():
    """Create a visualization of coherence metrics over time."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), facecolor='white')
    
    # Generate sample time series data
    t = np.linspace(0, 10, 100)
    
    # C(t) - Total Coherence
    coherence = 0.3 + 0.5 * np.exp(-0.1*t) * np.sin(2*t) + 0.2 * (1 - np.exp(-0.5*t))
    coherence = np.clip(coherence, 0, 1)
    
    ax = axes[0, 0]
    ax.plot(t, coherence, 'b-', linewidth=2.5)
    ax.fill_between(t, 0, coherence, alpha=0.3, color='#4a90e2')
    ax.axhline(y=0.7, color='g', linestyle='--', alpha=0.5, label='Strong coherence')
    ax.axhline(y=0.3, color='r', linestyle='--', alpha=0.5, label='Weak coherence')
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('C(t)', fontsize=11)
    ax.set_title('Total Coherence C(t)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Si - Sense Index
    sense_index = 0.5 + 0.3 * np.sin(1.5*t) * np.exp(-0.05*t)
    sense_index = np.clip(sense_index, 0, 1)
    
    ax = axes[0, 1]
    ax.plot(t, sense_index, 'g-', linewidth=2.5)
    ax.fill_between(t, 0, sense_index, alpha=0.3, color='#26de81')
    ax.axhline(y=0.8, color='g', linestyle='--', alpha=0.5, label='Excellent stability')
    ax.axhline(y=0.4, color='orange', linestyle='--', alpha=0.5, label='Caution zone')
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Si', fontsize=11)
    ax.set_title('Sense Index (Si)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # ΔNFR - Reorganization Gradient
    delta_nfr = 0.5 * np.sin(3*t) * np.exp(-0.1*t)
    
    ax = axes[1, 0]
    ax.plot(t, delta_nfr, 'purple', linewidth=2.5)
    ax.fill_between(t, 0, delta_nfr, where=(delta_nfr>0), alpha=0.3, color='green', label='Expansion')
    ax.fill_between(t, 0, delta_nfr, where=(delta_nfr<0), alpha=0.3, color='red', label='Contraction')
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('ΔNFR', fontsize=11)
    ax.set_title('Reorganization Gradient (ΔNFR)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Structural Frequency νf
    nu_f = 20 + 15 * np.sin(t) * np.exp(-0.15*t)
    
    ax = axes[1, 1]
    ax.plot(t, nu_f, 'orange', linewidth=2.5)
    ax.fill_between(t, 0, nu_f, alpha=0.3, color='#fdcb6e')
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('νf (Hz_str)', fontsize=11)
    ax.set_title('Structural Frequency (νf)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, None)
    
    fig.suptitle('TNFR Coherence Metrics Evolution', 
                fontsize=16, fontweight='bold', y=0.995,
                color='#2c5aa0')
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(OUTPUT_DIR / 'coherence_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created coherence_metrics.png")


def create_nodal_equation_visual():
    """Create a visual representation of the nodal equation."""
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
    
    # Main equation components
    ax.text(0.5, 0.7, r'∂EPI/∂t = νf · ΔNFR(t)', 
           ha='center', fontsize=32, fontweight='bold',
           color='#2c5aa0',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#e8f4f8', 
                    edgecolor='#2c5aa0', linewidth=3))
    
    # Component explanations
    components = [
        (0.15, 0.45, '∂EPI/∂t', 'Rate of structural\nchange', '#4a90e2'),
        (0.5, 0.45, 'νf', 'Structural frequency\n(Hz_str)', '#26de81'),
        (0.85, 0.45, 'ΔNFR(t)', 'Reorganization\ngradient', '#fd79a8'),
    ]
    
    for x, y, symbol, desc, color in components:
        # Symbol box
        rect = patches.FancyBboxPatch((x-0.08, y-0.05), 0.16, 0.08,
                                      boxstyle="round,pad=0.01",
                                      facecolor=color,
                                      edgecolor='#333333',
                                      linewidth=2,
                                      alpha=0.7)
        ax.add_patch(rect)
        ax.text(x, y, symbol, ha='center', va='center',
               fontsize=14, fontweight='bold', color='white')
        
        # Description
        ax.text(x, y-0.15, desc, ha='center', va='top',
               fontsize=10, color='#333333')
    
    # Title
    ax.text(0.5, 0.95, 'The Canonical Nodal Equation', 
           ha='center', fontsize=18, fontweight='bold',
           color='#2c5aa0')
    
    # Footer
    ax.text(0.5, 0.05, 'Structure changes proportionally to frequency and gradient', 
           ha='center', fontsize=12, style='italic',
           color='#666666')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'nodal_equation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created nodal_equation.png")


if __name__ == '__main__':
    print("Generating TNFR wiki graphics...")
    print("-" * 50)
    
    create_network_resonance_diagram()
    create_structural_operators_diagram()
    create_paradigm_comparison()
    create_coherence_visualization()
    create_nodal_equation_visual()
    
    print("-" * 50)
    print(f"✓ All graphics generated successfully in {OUTPUT_DIR}/")
    print("  Ready to be used in wiki pages!")
