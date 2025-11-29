import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from tnfr.physics.quantum_mechanics import QuantumMechanicsMapper

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def run_quantum_simulation():
    print("Initializing Quantum Regime Simulation (TNFR)...")
    
    # Simulation Parameters
    NUM_NODES = 1000
    BOX_LENGTH = 1.0
    STEPS = 200
    LEARNING_RATE = 0.05
    NOISE_LEVEL = 0.02
    
    # Initialize nodes with random frequencies (Energies)
    # We assume E ~ k^2 relation (Schrodinger-like dispersion)
    # Phase accumulation Phi = 2 * L * k = 2 * L * sqrt(E)
    # We want to see E converge to n^2 * pi^2 / L^2
    
    energies = np.random.uniform(0.5, 50.0, NUM_NODES)
    
    history = []
    
    print(f"Evolving {NUM_NODES} nodes in Structural Potential Well...")
    
    for step in range(STEPS):
        # 1. Calculate Phase Accumulation for each node
        # k = sqrt(E)
        # Round trip phase = 2 * L * k
        k = np.sqrt(energies)
        phase_accumulation = 2 * BOX_LENGTH * k
        
        # 2. Calculate Dissonance (Distance from Resonance 2*pi*n)
        # We want phase to be a multiple of 2*pi
        # remainder = phase % (2*pi)
        # distance = min(remainder, 2*pi - remainder)
        
        remainder = np.mod(phase_accumulation, 2 * np.pi)
        dissonance = np.minimum(remainder, 2 * np.pi - remainder)
        
        # 3. Nodal Update (Minimizing Dissonance)
        # Delta NFR is proportional to Dissonance
        # The system tries to reduce Delta NFR
        
        # Gradient descent on Dissonance with respect to Energy
        # d(Dissonance)/dE = d(Dissonance)/dPhi * dPhi/dk * dk/dE
        # dPhi/dk = 2L
        # dk/dE = 1 / (2*sqrt(E))
        # sign depends on which side of the peak we are
        
        # Determine direction: are we above or below the nearest multiple?
        # If remainder < pi, we are "above" a multiple (phase > 2pi*n), so we should decrease phase
        # If remainder > pi, we are "below" the next multiple, so we should increase phase
        
        direction = np.where(remainder < np.pi, -1.0, 1.0)
        
        # Apply update (Operator IL - Stabilization)
        # dE = - alpha * dissonance * direction
        # We add some noise (Operator OZ) to prevent getting stuck in shallow local minima if any
        
        delta_E = (LEARNING_RATE * dissonance * direction) + np.random.normal(0, NOISE_LEVEL, NUM_NODES)
        
        energies += delta_E
        
        # Ensure positive energy
        energies = np.maximum(energies, 0.1)
        
        if step % 20 == 0:
            mean_diss = np.mean(dissonance)
            print(f"Step {step}: Mean Dissonance = {mean_diss:.4f}")
            history.append(energies.copy())
            
    print("Simulation Complete.")
    
    # --- Analysis & Visualization ---
    results_dir = Path("results/quantum_demo")
    ensure_dir(results_dir)
    
    # 1. Energy Level Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(energies, bins=100, color='purple', alpha=0.7, label='Simulated Nodes')
    
    # Plot theoretical levels
    # Resonance condition: 2*L*k = 2*pi*n => k = pi*n/L => E = k^2 = (pi*n/L)^2
    max_E = np.max(energies)
    n_max = int(np.sqrt(max_E) * BOX_LENGTH / np.pi) + 1
    
    # Use Mapper for theoretical levels
    theoretical_levels = QuantumMechanicsMapper.calculate_theoretical_levels(BOX_LENGTH, n_max + 1)
    
    # Scale levels by pi^2 because Mapper uses simplified units E=n^2/(8L^2) 
    # but our simulation uses E=k^2=(pi*n/L)^2. 
    # Mapper: n^2 / (8 L^2). Simulation: pi^2 * n^2 / L^2.
    # Ratio: 8 * pi^2. 
    # Actually, let's just use the formula consistent with our simulation here for clarity
    # or update the Mapper to match. 
    # Let's stick to the simulation formula for the plot to match the data.
    
    theoretical_levels = []
    for n in range(1, n_max + 2):
        E_n = (np.pi * n / BOX_LENGTH) ** 2
        theoretical_levels.append(E_n)
        plt.axvline(x=E_n, color='orange', linestyle='--', alpha=0.8, label=f'n={n}' if n==1 else "")
        
    plt.title("Emergent Quantization: Energy Level Selection")
    plt.xlabel("Structural Frequency (Energy)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(results_dir / "01_quantization_levels.png")
    print(f"Saved quantization plot to {results_dir / '01_quantization_levels.png'}")
    
    # 2. Convergence Plot
    plt.figure(figsize=(10, 6))
    # Plot a subset of trajectories
    subset = np.array(history)[:, :50] # First 50 nodes
    plt.plot(range(0, STEPS, 20), subset)
    
    # Add theoretical lines
    for E_n in theoretical_levels:
        if E_n < max_E:
            plt.axhline(y=E_n, color='orange', linestyle='--', alpha=0.5)
            
    plt.title("Trajectory Convergence to Eigenstates")
    plt.xlabel("Simulation Step")
    plt.ylabel("Energy")
    plt.grid(True, alpha=0.3)
    plt.savefig(results_dir / "02_convergence.png")
    print(f"Saved convergence plot to {results_dir / '02_convergence.png'}")
    
    # 3. Wavefunction Visualization (Reconstruction)
    # For the first few levels, plot the standing wave pattern
    plt.figure(figsize=(10, 6))
    x = np.linspace(0, BOX_LENGTH, 200)
    
    for i, n in enumerate(range(1, 4)):
        # psi_n(x) = sin(n * pi * x / L)
        psi = np.sin(n * np.pi * x / BOX_LENGTH)
        # Shift vertically for visibility
        plt.plot(x, psi + i*2.5, label=f'n={n} (E={theoretical_levels[i]:.1f})', linewidth=2)
        plt.axhline(y=i*2.5, color='black', linestyle='-', alpha=0.2)
        
    plt.title("Reconstructed Structural Standing Waves (Eigenmodes)")
    plt.xlabel("Position (x)")
    plt.ylabel("Amplitude (Psi)")
    plt.yticks([])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(results_dir / "03_wavefunctions.png")
    print(f"Saved wavefunction plot to {results_dir / '03_wavefunctions.png'}")

if __name__ == "__main__":
    run_quantum_simulation()
