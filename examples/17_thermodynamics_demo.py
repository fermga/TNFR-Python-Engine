import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# --- Configuration ---
GRID_SIZE = 50
STEPS = 1000
DT = 0.05
COUPLING_K = 0.5  # Thermal conductivity equivalent
COFFEE_RADIUS = 8
RESULTS_DIR = Path("results/thermodynamics_demo")

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def wrap_angle(angle):
    """Wrap angle to [-pi, pi]"""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def get_neighbors_diff(grid):
    """Calculate sum of phase differences with 4 neighbors (Vectorized)"""
    # Shift grids to get neighbors
    # Pad with edge values (Neumann boundary conditions - insulated room walls)
    padded = np.pad(grid, 1, mode='edge')
    
    center = padded[1:-1, 1:-1]
    up = padded[0:-2, 1:-1]
    down = padded[2:, 1:-1]
    left = padded[1:-1, 0:-2]
    right = padded[1:-1, 2:]
    
    # Sin coupling (Kuramoto)
    diff = np.sin(up - center) + np.sin(down - center) + \
           np.sin(left - center) + np.sin(right - center)
           
    return diff

def calculate_temperature(grid):
    """
    Temperature defined as Local Phase Gradient Magnitude.
    High disorder (random phases) = High Temperature.
    Aligned phases = Low Temperature.
    """
    padded = np.pad(grid, 1, mode='edge')
    center = padded[1:-1, 1:-1]
    
    # Gradient magnitude approximation
    # Average absolute phase difference with neighbors
    diff_up = np.abs(wrap_angle(padded[0:-2, 1:-1] - center))
    diff_down = np.abs(wrap_angle(padded[2:, 1:-1] - center))
    diff_left = np.abs(wrap_angle(padded[1:-1, 0:-2] - center))
    diff_right = np.abs(wrap_angle(padded[1:-1, 2:] - center))
    
    avg_grad = (diff_up + diff_down + diff_left + diff_right) / 4.0
    return avg_grad

def run_thermodynamics_demo():
    print("--- TNFR Thermodynamics: The Cooling Coffee Cup ---")
    ensure_dir(RESULTS_DIR)
    
    # 1. Initialize Grid (The "Room")
    # Low energy state: Phases aligned (0), Low frequency
    phases = np.zeros((GRID_SIZE, GRID_SIZE))
    frequencies = np.ones((GRID_SIZE, GRID_SIZE)) * 1.0
    
    # 2. Initialize "Coffee Cup"
    # High energy state: Random phases (Incoherent), High frequency
    y, x = np.ogrid[:GRID_SIZE, :GRID_SIZE]
    center_y, center_x = GRID_SIZE // 2, GRID_SIZE // 2
    mask = (x - center_x)**2 + (y - center_y)**2 <= COFFEE_RADIUS**2
    
    phases[mask] = np.random.uniform(-np.pi, np.pi, size=phases[mask].shape)
    frequencies[mask] = 5.0  # Higher intrinsic activity
    
    # History tracking
    coffee_temp_history = []
    room_temp_history = []
    snapshots = []
    snapshot_times = [0, STEPS // 10, STEPS // 2, STEPS - 1]
    
    print(f"Simulating {STEPS} steps of Phase Diffusion...")
    
    for step in range(STEPS):
        # A. Calculate Coupling (Heat Flow)
        # d(phi)/dt = omega + K * sum(sin(phi_j - phi_i))
        coupling_force = get_neighbors_diff(phases)
        
        # B. Update Phases
        phases += (frequencies + COUPLING_K * coupling_force) * DT
        phases = wrap_angle(phases)
        
        # C. Measure "Temperature" (Structural Disorder)
        temp_grid = calculate_temperature(phases)
        
        # Track averages
        avg_coffee_temp = np.mean(temp_grid[mask])
        avg_room_temp = np.mean(temp_grid[~mask])
        
        coffee_temp_history.append(avg_coffee_temp)
        room_temp_history.append(avg_room_temp)
        
        if step in snapshot_times:
            snapshots.append(temp_grid.copy())
            print(f"   Step {step}: Coffee T={avg_coffee_temp:.3f}, Room T={avg_room_temp:.3f}")

    # --- Visualization ---
    
    # 1. Temperature Decay Plot
    plt.figure(figsize=(10, 6))
    time_axis = np.arange(STEPS) * DT
    plt.plot(time_axis, coffee_temp_history, 'r-', label='Coffee Cup (High Disorder)', linewidth=2)
    plt.plot(time_axis, room_temp_history, 'b-', label='Room (Low Disorder)', linewidth=2)
    plt.title("Emergent Newton's Law of Cooling\n(From Pure Phase Dynamics)")
    plt.xlabel("Time (t)")
    plt.ylabel("Structural Temperature (Phase Gradient Variance)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(RESULTS_DIR / "01_cooling_curve.png")
    print(f"Saved cooling curve to {RESULTS_DIR / '01_cooling_curve.png'}")
    
    # 2. Heatmap Snapshots
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    for i, ax in enumerate(axes):
        im = ax.imshow(snapshots[i], cmap='inferno', vmin=0, vmax=np.pi)
        ax.set_title(f"Step {snapshot_times[i]}")
        ax.axis('off')
        
    plt.suptitle("Structural Heat Diffusion (Phase Disorder Propagation)")
    plt.colorbar(im, ax=axes.ravel().tolist(), label="Temperature (Phase Gradient)")
    plt.savefig(RESULTS_DIR / "02_heat_diffusion.png")
    print(f"Saved diffusion map to {RESULTS_DIR / '02_heat_diffusion.png'}")

if __name__ == "__main__":
    run_thermodynamics_demo()
