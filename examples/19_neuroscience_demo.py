import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
from pathlib import Path

# --- Configuration ---
NUM_NEURONS = 50
STEPS = 600
DT = 0.1
RESULTS_DIR = Path("results/neuroscience_demo")

# Physics Parameters
BASE_COUPLING = 0.5
LEARNING_RATE = 0.05   # Hebbian plasticity rate
DECAY_RATE = 0.01      # Forgetting rate
NOISE_LEVEL = 0.1

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def run_neuroscience_demo() -> None:
    print("--- TNFR Neuroscience: Recursive Resonance & Memory ---")
    ensure_dir(RESULTS_DIR)
    
    # 1. Initialize Brain Network
    # Small-world network (like real brains)
    G = nx.watts_strogatz_graph(n=NUM_NEURONS, k=6, p=0.3)
    
    # State Variables
    phases = np.random.uniform(-np.pi, np.pi, NUM_NEURONS)
    frequencies = np.random.normal(1.0, 0.1, NUM_NEURONS) # Intrinsic firing rates
    
    # Coupling Matrix (Synaptic Weights)
    # Initialize with adjacency matrix
    adj = nx.to_numpy_array(G)
    weights = adj * BASE_COUPLING
    
    # History
    phase_history = []
    weight_history = []
    
    # Define Input Pattern (The "Thought")
    # We will stimulate neurons 0-9 to synchronize
    input_indices = np.arange(10)
    
    print("Simulating Brain Dynamics...")
    print("   Phase 1: Sensory Input (Steps 0-200)")
    print("   Phase 2: Silence/Reverberation (Steps 200-600)")
    
    for step in range(STEPS):
        # A. Apply Sensory Input (Only in Phase 1)
        if step < 200:
            # Force input neurons to a specific frequency/phase
            # Simulating "Seeing a Red Apple"
            phases[input_indices] = 0.0 # Lock phase
            frequencies[input_indices] = 2.0 # High activity
        else:
            # Remove input - let the network think for itself
            # Reset frequencies to intrinsic
            frequencies[input_indices] = 1.0
            
        # B. Calculate Phase Dynamics (Kuramoto)
        # d_phi_i = w_i + sum(K_ij * sin(phi_j - phi_i))
        
        # Vectorized coupling calculation
        # sin(phi_j - phi_i) matrix
        diff_matrix = phases[None, :] - phases[:, None]
        interaction = np.sin(diff_matrix)
        
        # Weighted interaction
        coupling_force = np.sum(weights * interaction, axis=1)
        
        # Update Phases
        noise = np.random.normal(0, NOISE_LEVEL, NUM_NEURONS)
        phases += (frequencies + coupling_force + noise) * DT
        phases = wrap_angle(phases)
        
        # C. Hebbian Plasticity (Learning)
        # dK_ij = alpha * cos(phi_i - phi_j) - decay
        # If phases aligned (cos ~ 1), strengthen weight
        # Only update existing connections (structural plasticity is slower)
        
        if step % 5 == 0:  # Plasticity is slower than firing
            # Correlation matrix (1 if synced, -1 if anti-synced)
            correlation = np.cos(diff_matrix)
            
            # Delta Weights
            dW = (LEARNING_RATE * correlation) - DECAY_RATE
            
            # Apply only to existing synapses (topology constraint)
            weights += dW * adj * DT
            
            # Clip weights (Synapses can't be negative or infinite)
            weights = np.clip(weights, 0, 2.0)
            
        # Record
        phase_history.append(phases.copy())
        if step % 50 == 0:
            weight_history.append(weights.copy())
            
    # --- Visualization ---
    
    # 1. Raster Plot (Brain Activity)
    plt.figure(figsize=(12, 6))
    ph_hist = np.array(phase_history)
    # Plot sin(phase) to see oscillations clearly
    plt.imshow(np.sin(ph_hist.T), aspect='auto', cmap='RdBu', interpolation='nearest', 
               extent=[0, STEPS, NUM_NEURONS, 0])
    
    # Mark phases
    plt.axvline(x=200, color='yellow', linestyle='--', linewidth=3, label='Input Removed')
    
    plt.title("Neural Raster Plot: Emergence of Working Memory")
    plt.xlabel("Time Step")
    plt.ylabel("Neuron ID")
    plt.colorbar(label="Activity (sin(phase))")
    plt.legend()
    
    plt.savefig(RESULTS_DIR / "01_brain_activity.png")
    print(f"Saved raster plot to {RESULTS_DIR / '01_brain_activity.png'}")
    
    # 2. Connectivity Evolution
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Initial Weights
    im0 = axes[0].imshow(weight_history[0], cmap='viridis', vmin=0, vmax=2)
    axes[0].set_title("Initial Connectivity (Random)")
    
    # Mid Weights (During Input)
    mid_idx = len(weight_history) // 3
    im1 = axes[1].imshow(weight_history[mid_idx], cmap='viridis', vmin=0, vmax=2)
    axes[1].set_title("During Stimulation (Learning)")
    
    # Final Weights (Memory)
    im2 = axes[2].imshow(weight_history[-1], cmap='viridis', vmin=0, vmax=2)
    axes[2].set_title("Final Connectivity (Memory Encoded)")
    
    plt.suptitle("Hebbian Plasticity: Structural Encoding of Thought")
    plt.colorbar(im2, ax=axes.ravel().tolist(), label="Synaptic Weight")
    
    plt.savefig(RESULTS_DIR / "02_synaptic_weights.png")
    print(f"Saved connectivity map to {RESULTS_DIR / '02_synaptic_weights.png'}")


if __name__ == "__main__":
    run_neuroscience_demo()
