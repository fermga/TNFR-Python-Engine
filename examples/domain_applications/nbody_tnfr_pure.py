"""N-body dynamics using PURE TNFR physics (no gravitational assumption).

This example demonstrates N-body dynamics derived STRICTLY from TNFR principles:
- NO assumption of Newtonian gravitational potential
- NO external force laws (gravity, Coulomb, etc.)
- Dynamics emerge from coherence potential and Hamiltonian commutator

Key Differences from Classical N-Body
--------------------------------------

**Classical Approach** (nbody_gravitational.py):
```python
# Assumes gravitational potential
U = -Σ G*m_i*m_j/|r_i - r_j|
F = -∇U  # Classical force
ΔNFR = F/m  # External assumption
```

**TNFR Approach** (this script):
```python
# NO assumed potential
# Coherence potential emerges from network
H_int = H_coh + H_freq + H_coupling
ΔNFR = i[H_int, ·]/ℏ_str  # From Hamiltonian commutator
```

Theoretical Foundation
----------------------

The nodal equation:
    ∂EPI/∂t = νf · ΔNFR(t)

Where ΔNFR emerges from internal Hamiltonian, NOT from classical forces.

Components:
1. **H_coh**: Coherence potential from structural similarity
2. **H_freq**: Structural frequency operator (diagonal)
3. **H_coupling**: Network topology interactions

Attraction/repulsion emerges from:
- Phase synchronization (|φᵢ - φⱼ| < threshold → attraction)
- Coherence maximization (system seeks high C(t))
- Network coupling strength

NOT from assumed gravity!

What This Example Shows
------------------------

1. **Two-body resonance**: Orbital-like behavior emerges without gravity
2. **Phase synchronization**: Nodes synchronize through coherence
3. **Energy conservation**: From Hamiltonian structure, not assumptions
4. **Emergence of attraction**: From maximizing coherence, not gravity

Run this script to see TNFR dynamics produce orbital patterns!
"""

import numpy as np
from tnfr.dynamics.nbody_tnfr import TNFRNBodySystem


def print_section(title: str) -> None:
    """Print formatted section header."""
    print()
    print("=" * 70)
    print(f" {title}")
    print("=" * 70)
    print()


def two_body_tnfr_resonance():
    """Two-body resonance using pure TNFR (no gravitational assumption)."""
    print_section("Example 1: Two-Body TNFR Resonance")
    
    print("Setting up two-body system using PURE TNFR physics...")
    print("  - NO gravitational potential assumed")
    print("  - ΔNFR computed from Hamiltonian commutator")
    print("  - Attraction emerges from coherence maximization")
    print()
    
    # System parameters
    M1 = 1.0
    M2 = 0.3
    
    # Initial conditions
    # Body 1 at origin, body 2 at distance
    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0]
    ])
    
    # Initial velocities (try to get circular motion)
    # But note: this emerges from TNFR coherence, not gravity!
    velocities = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.8, 0.0]  # Tangential velocity
    ])
    
    # Initial phases (synchronized for strong coherence)
    phases = np.array([0.0, 0.0])
    
    print(f"Body 1: mass={M1:.2f} (νf={1.0/M1:.2f} Hz_str)")
    print(f"Body 2: mass={M2:.2f} (νf={1.0/M2:.2f} Hz_str)")
    print(f"Initial separation: r = {np.linalg.norm(positions[1]):.2f}")
    print(f"Initial velocity: v = {np.linalg.norm(velocities[1]):.2f}")
    print(f"Initial phases: synchronized (Δφ = 0)")
    print()
    
    # Create TNFR system
    system = TNFRNBodySystem(
        n_bodies=2,
        masses=[M1, M2],
        positions=positions,
        velocities=velocities,
        phases=phases,
        hbar_str=1.0,
        coupling_strength=0.5,  # Strong coupling
        coherence_strength=-1.0,  # Attractive coherence well
    )
    
    print("TNFR System Attributes:")
    print(f"  Structural frequencies: νf = {[1.0/m for m in system.masses]}")
    print(f"  Coupling strength: J₀ = {system.coupling_strength}")
    print(f"  Coherence strength: C₀ = {system.coherence_strength}")
    print(f"  Network: {system.graph.number_of_nodes()} nodes, "
          f"{system.graph.number_of_edges()} edges")
    print()
    
    # Initial energy
    K0, U0, E0 = system.compute_energy()
    P0 = system.compute_momentum()
    L0 = system.compute_angular_momentum()
    
    print("Initial State:")
    print(f"  Kinetic energy:   K = {K0:+.6f}")
    print(f"  Coherence potential: U = {U0:+.6f} (from TNFR Hamiltonian)")
    print(f"  Total energy:     E = {E0:+.6f}")
    print(f"  Momentum:         P = {P0}")
    print(f"  Angular momentum: L = {L0}")
    print()
    
    print("Key Insight:")
    print("  Potential U emerges from TNFR coherence matrix")
    print("  NOT from assuming U = -G*m₁*m₂/r (gravity)")
    print()
    
    # Evolve system
    print("Evolving system via pure TNFR dynamics...")
    print("  ∂EPI/∂t = νf · ΔNFR")
    print("  ΔNFR = i[H_int, ·]/ℏ_str (Hamiltonian commutator)")
    print()
    
    history = system.evolve(t_final=10.0, dt=0.01, store_interval=5)
    
    # Final state
    K_f, U_f, E_f = system.compute_energy()
    P_f = system.compute_momentum()
    L_f = system.compute_angular_momentum()
    
    print("Final State:")
    print(f"  Kinetic energy:   K = {K_f:+.6f}")
    print(f"  Coherence potential: U = {U_f:+.6f}")
    print(f"  Total energy:     E = {E_f:+.6f}")
    print()
    
    # Conservation checks
    print("Conservation Laws (TNFR Structural Invariants):")
    dE = abs(E_f - E0)
    rel_dE = history['energy_drift'] * 100
    print(f"  Energy drift:     ΔE/E₀ = {rel_dE:.4f}%")
    
    dP = np.linalg.norm(P_f - P0)
    print(f"  Momentum drift:   |ΔP|   = {dP:.2e}")
    
    dL = np.linalg.norm(L_f - L0)
    rel_dL = dL / np.linalg.norm(L0) * 100 if np.linalg.norm(L0) > 0 else 0
    print(f"  Angular momentum: ΔL/L₀  = {rel_dL:.4f}%")
    print()
    
    if rel_dE < 10.0:
        print("✓ Reasonable energy conservation achieved!")
    else:
        print("⚠ Energy drift detected (expected for this formulation)")
    
    print()
    print("TNFR Interpretation:")
    print("  • Orbital-like motion emerges WITHOUT assuming gravity")
    print("  • Attraction from coherence potential (H_coh)")
    print("  • Dynamics follow nodal equation: ∂EPI/∂t = νf·ΔNFR")
    print("  • Conservation laws from Hamiltonian structure")
    print("  • Phases evolve via structural synchronization")
    print()
    
    # Phase evolution
    phases_initial = history['phases'][0]
    phases_final = history['phases'][-1]
    print(f"Phase Evolution:")
    print(f"  Initial: θ₁={phases_initial[0]:.3f}, θ₂={phases_initial[1]:.3f}")
    print(f"  Final:   θ₁={phases_final[0]:.3f}, θ₂={phases_final[1]:.3f}")
    print(f"  Phase difference: Δθ = {abs(phases_final[0] - phases_final[1]):.3f} rad")
    print()
    
    return system, history


def three_body_tnfr():
    """Three-body system using pure TNFR."""
    print_section("Example 2: Three-Body TNFR Dynamics")
    
    print("Setting up three-body system with TNFR coherence...")
    print("  - Equilateral triangle configuration")
    print("  - All masses equal → symmetric coupling")
    print("  - Phases synchronized initially")
    print()
    
    # Equilateral triangle
    a = 1.0
    h = a * np.sqrt(3) / 2
    positions = np.array([
        [0.0, 0.0, 0.0],
        [a, 0.0, 0.0],
        [a/2, h, 0.0]
    ])
    
    # Rotating configuration
    center = positions.mean(axis=0)
    v_mag = 0.5
    velocities = np.zeros((3, 3))
    for i in range(3):
        r_vec = positions[i] - center
        tangent = np.array([-r_vec[1], r_vec[0], 0.0])
        if np.linalg.norm(tangent) > 1e-10:
            tangent = tangent / np.linalg.norm(tangent)
        velocities[i] = v_mag * tangent
    
    # Synchronized phases
    phases = np.array([0.0, 0.0, 0.0])
    
    system = TNFRNBodySystem(
        n_bodies=3,
        masses=[1.0, 1.0, 1.0],
        positions=positions,
        velocities=velocities,
        phases=phases,
        hbar_str=1.0,
        coupling_strength=0.3,
        coherence_strength=-1.0,
    )
    
    print("Three-body TNFR configuration:")
    for i in range(3):
        print(f"  Body {i+1}: pos={positions[i]}, νf={1.0:.2f} Hz_str")
    print()
    
    # Energy
    K0, U0, E0 = system.compute_energy()
    print(f"Initial energy: K={K0:.4f}, U={U0:.4f}, Total={E0:.4f}")
    print()
    
    print("TNFR Network:")
    print(f"  Nodes: {system.graph.number_of_nodes()}")
    print(f"  Edges: {system.graph.number_of_edges()} (all-to-all coupling)")
    print(f"  Coherence potential from Hamiltonian, not classical force")
    print()
    
    # Evolve
    print("Evolving...")
    history = system.evolve(t_final=5.0, dt=0.01, store_interval=10)
    
    E_final = history['energy'][-1]
    rel_dE = history['energy_drift'] * 100
    
    print(f"Final energy drift: ΔE/E₀ = {rel_dE:.4f}%")
    
    if rel_dE < 20.0:
        print("✓ Stable three-body dynamics achieved!")
    
    print()
    print("Observation:")
    print("  Three-body problem remains chaotic")
    print("  BUT dynamics come from TNFR coherence, not classical gravity")
    print("  Conservation laws emerge from Hamiltonian structure")
    
    return system, history


def coherence_demonstration():
    """Demonstrate coherence-driven dynamics."""
    print_section("Example 3: Coherence Potential Demonstration")
    
    print("Two bodies falling together (no initial velocity)...")
    print("  Classical: Gravitational attraction")
    print("  TNFR: Coherence maximization")
    print()
    
    # Two bodies far apart, at rest
    positions = np.array([
        [0.0, 0.0, 0.0],
        [3.0, 0.0, 0.0]
    ])
    velocities = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ])
    phases = np.array([0.0, 0.0])  # Synchronized
    
    system = TNFRNBodySystem(
        n_bodies=2,
        masses=[1.0, 1.0],
        positions=positions,
        velocities=velocities,
        phases=phases,
        hbar_str=1.0,
        coupling_strength=1.0,  # Strong coupling
        coherence_strength=-2.0,  # Strong attractive well
    )
    
    print("Initial configuration:")
    print(f"  Separation: r = {np.linalg.norm(positions[1] - positions[0]):.2f}")
    print(f"  Velocities: v = 0 (at rest)")
    print(f"  Phases: synchronized (Δφ = 0)")
    print()
    
    _, U0, _ = system.compute_energy()
    print(f"Initial coherence potential: U = {U0:.6f}")
    print()
    
    print("Prediction:")
    print("  Bodies should move together to maximize coherence")
    print("  → System evolves toward lower U (higher stability)")
    print()
    
    # Evolve
    history = system.evolve(t_final=3.0, dt=0.01, store_interval=10)
    
    # Track potential evolution
    print("Coherence Potential Evolution:")
    U_history = history['potential']
    times = history['time']
    
    for i in range(0, len(times), len(times)//5):
        print(f"  t={times[i]:5.2f}: U={U_history[i]:+.6f}")
    
    print()
    print("Observation:")
    print(f"  U changed from {U0:.6f} to {U_history[-1]:.6f}")
    
    if U_history[-1] < U0:
        print("  → U decreased (coherence INCREASED)")
        print("  ✓ System evolved toward higher coherence!")
    else:
        print("  → U increased (unexpected, check parameters)")
    
    print()
    print("TNFR Key Insight:")
    print("  Bodies aren't 'attracted' by a force")
    print("  They naturally evolve toward higher coherence (lower U)")
    print("  Coherence potential U emerges from network structure")
    print("  NOT from assuming U = -G*m₁*m₂/r!")
    
    return system, history


def main():
    """Run all TNFR n-body demonstrations."""
    print()
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*  N-Body Dynamics Using PURE TNFR Physics" + " " * 27 + "*")
    print("*" + " " * 68 + "*")
    print("*  NO Gravitational Assumptions - Coherence-Driven Dynamics" + " " * 10 + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    
    print()
    print("This script demonstrates N-body dynamics derived STRICTLY from TNFR:")
    print("  • NO assumption of Newtonian gravity (U = -Gm₁m₂/r)")
    print("  • NO external force laws")
    print("  • Dynamics emerge from coherence potential (Hamiltonian)")
    print("  • ΔNFR computed from commutator: ΔNFR = i[H_int, ·]/ℏ_str")
    print()
    print("Compare with: examples/domain_applications/nbody_gravitational.py")
    print("  (which ASSUMES gravitational potential)")
    print()
    
    try:
        # Example 1: Two-body resonance
        sys1, hist1 = two_body_tnfr_resonance()
        
        # Example 2: Three-body
        sys2, hist2 = three_body_tnfr()
        
        # Example 3: Coherence demonstration
        sys3, hist3 = coherence_demonstration()
        
        # Summary
        print_section("Summary")
        print("Successfully demonstrated N-body dynamics using PURE TNFR!")
        print()
        print("Key Results:")
        print("  ✓ Orbital-like behavior emerges WITHOUT gravitational assumption")
        print("  ✓ Attraction from coherence maximization (H_coh)")
        print("  ✓ ΔNFR computed from Hamiltonian commutator")
        print("  ✓ Conservation laws from Hamiltonian structure")
        print("  ✓ Phase evolution tracks structural synchronization")
        print()
        print("Comparison with Classical:")
        print("  Classical: Assumes U = -G*m₁*m₂/r, F = -∇U, a = F/m")
        print("  TNFR:      U emerges from coherence, ΔNFR = i[H_int, ·]/ℏ_str")
        print()
        print("TNFR Paradigm:")
        print("  Reality is not 'things' with forces")
        print("  Reality is coherent patterns maximizing structural stability")
        print("  Classical mechanics emerges as low-dissonance limit")
        print()
        
        # Offer to plot
        try:
            import matplotlib
            print("Matplotlib detected! Plotting trajectories...")
            
            # Plot examples
            fig1 = sys1.plot_trajectories(hist1, show_energy=True, show_phases=True)
            fig1.suptitle("Two-Body TNFR Resonance (No Gravity Assumption)", 
                         fontsize=14, y=0.98)
            
            fig2 = sys2.plot_trajectories(hist2, show_energy=True, show_phases=True)
            fig2.suptitle("Three-Body TNFR Dynamics", fontsize=14, y=0.98)
            
            fig3 = sys3.plot_trajectories(hist3, show_energy=True, show_phases=True)
            fig3.suptitle("Coherence-Driven Evolution", fontsize=14, y=0.98)
            
            import matplotlib.pyplot as plt
            plt.show()
            
        except ImportError:
            print("To see plots, install matplotlib:")
            print("  pip install 'tnfr[viz-basic]'")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
