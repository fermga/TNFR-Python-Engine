"""
Biological Operator Sequences: The Grammar of Life
==================================================

This script bridges TNFR Structural Physics with Biology by modeling biological processes
as sequences of Canonical Operators.

Simulations:
1.  **Fractal Growth (The Fern)**:
    Models plant growth not just as geometry, but as an Operator Sequence:
    - AL (Emission): Sprout
    - VAL (Expansion): Stem Elongation
    - OZ (Dissonance): Branching (Symmetry Breaking)
    - REMESH (Recursivity): Self-similarity
    - IL (Coherence): Leaf Termination

2.  **The Cell Cycle (Mitosis as Bifurcation)**:
    Visualizes the structural stability metrics (Coherence vs. Stress) during cell division.
    Demonstrates that Mitosis is a controlled Dissonance event (OZ) followed by
    Re-stabilization (THOL/IL).

Output:
- results/geocentric_vortex_study/biological_operator_fern.png
- results/geocentric_vortex_study/mitosis_structural_dynamics.png
"""

import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "results/geocentric_vortex_study"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def simulate_operator_fern():
    print("Simulating Fractal Growth as Operator Sequence...")
    
    # L-System Parameters for a Fern-like structure
    # F: VAL (Forward/Expand)
    # +: OZ (Rotate Right)
    # -: OZ (Rotate Left)
    # [: REMESH (Push State/Start Branch)
    # ]: SHA (Pop State/End Branch)
    
    axiom = "X"
    rules = {
        "X": "F+[[X]-X]-F[-FX]+X", 
        "F": "FF"
    }
    iterations = 5
    angle = 25
    
    # Generate String
    current_string = axiom
    for _ in range(iterations):
        next_string = ""
        for char in current_string:
            next_string += rules.get(char, char)
        current_string = next_string
        
    # Draw
    stack = []
    x, y = 0, 0
    theta = 90 # Start pointing up
    points = [(x, y)]
    
    fig, ax = plt.subplots(figsize=(10, 12), facecolor='#050510')
    ax.set_facecolor('#000005')
    
    # Map characters to colors/operators for visualization
    # We will draw segments
    
    lines = []
    
    for char in current_string:
        if char == "F": # VAL (Expansion)
            x_new = x + np.cos(np.radians(theta))
            y_new = y + np.sin(np.radians(theta))
            ax.plot([x, x_new], [y, y_new], color='#44FF88', linewidth=0.8, alpha=0.6)
            x, y = x_new, y_new
        elif char == "+": # OZ (Right)
            theta -= angle
        elif char == "-": # OZ (Left)
            theta += angle
        elif char == "[": # REMESH (Start Branch)
            stack.append((x, y, theta))
        elif char == "]": # SHA (End Branch)
            x, y, theta = stack.pop()
            
    # Annotate Operators
    ax.text(0, 0, "AL (Emission)", color='white', fontsize=12, ha='center', fontweight='bold')
    ax.text(0, 10, "VAL (Expansion)", color='#44FF88', fontsize=10, ha='center')
    ax.text(15, 30, "OZ (Branching)", color='#FF4444', fontsize=10, ha='center')
    ax.text(-15, 50, "REMESH (Recursion)", color='#4488FF', fontsize=10, ha='center')
    ax.text(0, 80, "IL (Leaf/Coherence)", color='#FFFF00', fontsize=10, ha='center')
    
    ax.set_title("THE GRAMMAR OF GROWTH\nBiological Form as Operator Sequence", color='white', pad=20)
    ax.axis('equal')
    ax.axis('off')
    
    output_path = os.path.join(OUTPUT_DIR, "biological_operator_fern.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#050510')
    print(f"Saved to: {output_path}")
    plt.close()

def simulate_mitosis_dynamics():
    print("Simulating Mitosis Structural Dynamics...")
    
    # Time steps
    t = np.linspace(0, 100, 500)
    
    # Phases of Cell Cycle
    # G1 (Growth): VAL -> High Coherence, Low Stress
    # S (Synthesis): REMESH -> Increasing Stress (Duplication)
    # G2 (Prep): IL -> Stabilization before storm
    # M (Mitosis): OZ -> Extreme Stress (Bifurcation), Coherence Dip
    # Cytokinesis: THOL -> Reorganization into two stable nodes
    
    # Model Coherence C(t)
    # Starts high, dips slightly during S, dips massive during M, recovers
    coherence = np.ones_like(t) * 0.9
    
    # S Phase Dip (30-60)
    coherence[150:300] -= 0.1 * np.sin(np.linspace(0, np.pi, 150))
    
    # M Phase Crash (70-90)
    coherence[350:450] -= 0.5 * np.sin(np.linspace(0, np.pi, 100))
    
    # Model Stress (Delta NFR)
    # Inverse of coherence mostly
    stress = np.zeros_like(t) + 0.1
    stress[150:300] += 0.2 * np.sin(np.linspace(0, np.pi, 150)) # S phase stress
    stress[350:450] += 0.8 * np.sin(np.linspace(0, np.pi, 100)) # Mitosis spike
    
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='#050510')
    ax.set_facecolor('#000005')
    
    # Plot Coherence
    ax.plot(t, coherence, color='#44FF88', linewidth=3, label='Structural Coherence C(t)')
    
    # Plot Stress
    ax.plot(t, stress, color='#FF4444', linewidth=3, label='Structural Pressure ΔNFR')
    
    # Annotate Phases / Operators
    ax.axvspan(0, 30, color='#44FF88', alpha=0.1)
    ax.text(15, 0.95, "G1 Phase\n(VAL)", color='white', ha='center')
    
    ax.axvspan(30, 60, color='#4488FF', alpha=0.1)
    ax.text(45, 0.95, "S Phase\n(REMESH)", color='white', ha='center')
    
    ax.axvspan(60, 70, color='#FFFF00', alpha=0.1)
    ax.text(65, 0.95, "G2\n(IL)", color='white', ha='center')
    
    ax.axvspan(70, 90, color='#FF4444', alpha=0.2)
    ax.text(80, 0.5, "MITOSIS\n(OZ -> THOL)", color='white', ha='center', fontweight='bold')
    
    ax.axvspan(90, 100, color='#44FF88', alpha=0.1)
    ax.text(95, 0.95, "G1 (New)\n(AL)", color='white', ha='center')
    
    ax.set_title("MITOSIS AS A BIFURCATION EVENT\nStructural Dynamics of Cell Division", color='white')
    ax.set_xlabel("Time (Cell Cycle)", color='white')
    ax.set_ylabel("Magnitude", color='white')
    ax.tick_params(colors='white')
    ax.legend(facecolor='black', labelcolor='white')
    ax.grid(color='white', alpha=0.1)
    
    output_path = os.path.join(OUTPUT_DIR, "mitosis_structural_dynamics.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#050510')
    print(f"Saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    simulate_operator_fern()
    simulate_mitosis_dynamics()
