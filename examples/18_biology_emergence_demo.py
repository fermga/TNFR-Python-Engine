import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# --- Configuration ---
GRID_SIZE = 60
STEPS = 500
DT = 0.1
RESULTS_DIR = Path("results/biology_demo")

# Physics Parameters
NOISE_STRENGTH = 0.2       # Thermodynamic entropy force
COUPLING_STRENGTH = 0.1    # Passive physics coupling
LIFE_STRENGTH = 0.8        # Active stabilization (IL Operator strength)
CRITICAL_COHERENCE = 0.85  # Threshold to activate "Life"
NUTRIENT_REGEN = 0.01      # Environmental resource recovery
METABOLIC_COST = 0.05      # Cost of being alive

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def get_local_coherence(grid):
    """
    Calculate Kuramoto Order Parameter locally (3x3 window).
    Returns value 0.0 (Chaos) to 1.0 (Perfect Sync).
    """
    # Complex order parameter R = |1/N sum(e^i*theta)|
    # We'll do a simplified version: 1 - average phase difference
    padded = np.pad(grid, 1, mode='wrap') # Toroidal world
    
    diffs = []
    center = padded[1:-1, 1:-1]
    
    # Check 4 neighbors
    for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
        neighbor = padded[1+dy : -1+dy if -1+dy < 0 else None, 
                          1+dx : -1+dx if -1+dx < 0 else None]
        # Fix slicing for edge cases if needed, but 'wrap' pad handles it mostly
        # Actually simpler: just roll the array
        neighbor = np.roll(grid, (dy, dx), axis=(0, 1))
        d = np.abs(wrap_angle(center - neighbor))
        diffs.append(d)
        
    avg_diff = np.mean(diffs, axis=0)
    # Coherence = 1 - (diff / pi)
    coherence = 1.0 - (avg_diff / np.pi)
    return np.clip(coherence, 0, 1)

def run_biology_demo() -> None:
    print("--- TNFR Biology: Emergence of Autopoietic Patterns ---")
    ensure_dir(RESULTS_DIR)
    
    # 1. Initialize Primordial Soup
    # Random phases (Chaos)
    phases = np.random.uniform(-np.pi, np.pi, (GRID_SIZE, GRID_SIZE))
    
    # Nutrients (Energy source for order)
    nutrients = np.ones((GRID_SIZE, GRID_SIZE))
    
    # Seed a few "Protocells" (Small ordered clusters)
    # Without this, random emergence takes too long for a demo
    print("Seeding protocells...")
    seeds = [(GRID_SIZE//4, GRID_SIZE//4), (GRID_SIZE//2, GRID_SIZE//2), (3*GRID_SIZE//4, 3*GRID_SIZE//4)]
    for sy, sx in seeds:
        phases[sy-2:sy+3, sx-2:sx+3] = 0.0 # Force sync
    
    history_alive_count = []
    snapshots = []
    snapshot_times = [0, 50, 150, 300, 499]
    
    print(f"Simulating {STEPS} steps of Evolutionary Dynamics...")
    
    for step in range(STEPS):
        # A. Calculate Local Coherence (The "Health" of the pattern)
        coherence = get_local_coherence(phases)
        
        # B. Determine "Alive" Nodes
        # Alive if Coherence > Threshold AND Nutrients > 0
        is_alive = (coherence > CRITICAL_COHERENCE) & (nutrients > 0.1)
        
        # C. Update Phases
        
        # 1. Entropic Noise (Thermodynamics)
        noise = np.random.normal(0, NOISE_STRENGTH, (GRID_SIZE, GRID_SIZE))
        
        # 2. Passive Coupling (Physics)
        # Simple diffusion towards neighbors
        # Note: Laplacian on phases needs careful wrapping, simplified here for demo
        # Better: sum of sin diffs
        passive_force = np.zeros_like(phases)
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = np.roll(phases, (dy, dx), axis=(0, 1))
            passive_force += np.sin(neighbor - phases)
             
        # 3. Active Stabilization (Life)
        # Living nodes exert EXTRA force to align neighbors (The IL Operator)
        active_force = passive_force * LIFE_STRENGTH * is_alive
        
        # Total Update
        d_phi = (COUPLING_STRENGTH * passive_force) + active_force + noise
        phases += d_phi * DT
        phases = wrap_angle(phases)
        
        # D. Metabolism
        # Living nodes consume nutrients
        nutrients -= METABOLIC_COST * is_alive * DT
        # Environment regenerates nutrients
        nutrients += NUTRIENT_REGEN * DT
        nutrients = np.clip(nutrients, 0, 1)
        
        # Track stats
        alive_count = np.sum(is_alive)
        history_alive_count.append(alive_count)
        
        if step in snapshot_times:
            # Visualize: Brightness = Coherence, Color = Phase
            # We'll just store Coherence for the heatmap
            snapshots.append(coherence.copy())
            print(f"   Step {step}: Living Cells = {alive_count}")

    # --- Visualization ---
    
    # 1. Population Plot
    plt.figure(figsize=(10, 6))
    plt.plot(history_alive_count, 'g-', linewidth=2)
    plt.title("Population Dynamics of Coherent Structures")
    plt.xlabel("Time Step")
    plt.ylabel("Number of 'Living' (Coherent) Nodes")
    plt.grid(True, alpha=0.3)
    plt.savefig(RESULTS_DIR / "01_population_curve.png")
    print(f"Saved population curve to {RESULTS_DIR / '01_population_curve.png'}")
    
    # 2. Emergence Snapshots
    fig, axes = plt.subplots(1, len(snapshots), figsize=(20, 5))
    
    for i, ax in enumerate(axes):
        # Plot Coherence (Order)
        im = ax.imshow(snapshots[i], cmap='magma', vmin=0, vmax=1)
        ax.set_title(f"Step {snapshot_times[i]}")
        ax.axis('off')
        
    plt.suptitle("Emergence of Life: Coherent Islands in Entropic Sea")
    plt.colorbar(im, ax=axes.ravel().tolist(), label="Structural Coherence (Life)")
    plt.savefig(RESULTS_DIR / "02_emergence_map.png")
    print(f"Saved emergence map to {RESULTS_DIR / '02_emergence_map.png'}")


if __name__ == "__main__":
    run_biology_demo()
