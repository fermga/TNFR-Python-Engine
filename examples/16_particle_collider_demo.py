import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

@dataclass
class ParticleNode:
    id: int
    q: np.ndarray # Position
    p: np.ndarray # Momentum
    mass: float   # Inverse Frequency
    active: bool = True
    history_q: List[np.ndarray] = field(default_factory=list)
    
    def update(self, dt):
        if not self.active:
            return
        # Simple kinematic update (Drift)
        # q_new = q + (p/m) * dt
        v = self.p / self.mass
        self.q += v * dt
        self.history_q.append(self.q.copy())


def run_collider_demo() -> None:
    print("--- TNFR High-Energy Physics: Collider Simulation ---")
    
    # Simulation Parameters
    DT = 0.01
    STEPS = 200
    COLLISION_RADIUS = 0.5
    FRAGMENTATION_COUNT = 20  # Number of particles in a jet
    
    # Initialize Beams
    # Beam 1: Moving Right
    p1 = ParticleNode(
        id=1,
        q=np.array([-10.0, 0.0]),
        p=np.array([50.0, 0.0]),  # High Momentum
        mass=1.0
    )
    
    # Beam 2: Moving Left
    p2 = ParticleNode(
        id=2,
        q=np.array([10.0, 0.0]),
        p=np.array([-50.0, 0.0]),  # High Momentum
        mass=1.0
    )
    
    particles = [p1, p2]
    next_id = 3
    collision_occurred = False
    collision_point = None
    
    print("Accelerating beams...")
    
    for step in range(STEPS):
        # Update all active particles
        for p in particles:
            p.update(DT)
            
        # Check for Collision (only between the two parents if active)
        if not collision_occurred and p1.active and p2.active:
            dist = np.linalg.norm(p1.q - p2.q)
            
            if dist < COLLISION_RADIUS:
                print(f"-> COLLISION DETECTED at Step {step} (r={dist:.3f})")
                print("-> Triggering Structural Bifurcation (Fragmentation)...")
                
                collision_occurred = True
                collision_point = (p1.q + p2.q) / 2
                
                # Deactivate Parents
                p1.active = False
                p2.active = False
                
                # Total Energy/Momentum to conserve
                # E ~ |p|^2 / 2m (Kinetic) + Internal (Mass)
                # Here we just conserve Momentum vector sum
                total_p = p1.p + p2.p  # Should be ~0
                total_energy = (np.linalg.norm(p1.p)**2 + np.linalg.norm(p2.p)**2) / 2.0
                
                print(f"   Total Energy: {total_energy:.1f}")
                print(f"   Total Momentum: {total_p}")
                
                # Generate Jets
                # We create N particles with random directions but net momentum ~ 0
                # Energy distributed randomly
                
                for i in range(FRAGMENTATION_COUNT):
                    # Random direction
                    theta = np.random.uniform(0, 2 * np.pi)
                    speed = np.random.normal(10.0, 3.0)  # High speed ejecta
                    
                    # Momentum vector
                    px = speed * np.cos(theta)
                    py = speed * np.sin(theta)
                    
                    # Add some asymmetry to make it look like jets
                    # Jet 1 preference
                    if i < FRAGMENTATION_COUNT // 2:
                        px += 5.0
                    else:
                        px -= 5.0
                        
                    new_p = np.array([px, py])
                    
                    # Create Child Node
                    child = ParticleNode(
                        id=next_id,
                        q=collision_point.copy(),
                        p=new_p,
                        mass=np.random.uniform(0.1, 0.5)  # Lighter particles
                    )
                    next_id += 1
                    particles.append(child)
                    
                print(f"   Created {FRAGMENTATION_COUNT} decay products (Jets).")
                
    print("Simulation Complete.")
    
    # --- Visualization ---
    results_dir = Path("results/collider_demo")
    ensure_dir(results_dir)
    
    plt.figure(figsize=(10, 10))
    
    # Plot Parents
    h1 = np.array(p1.history_q)
    h2 = np.array(p2.history_q)
    
    # Only plot up to collision for parents
    if len(h1) > 0:
        plt.plot(h1[:, 0], h1[:, 1], 'b-', linewidth=3, label='Beam 1 (Proton)', alpha=0.6)
    if len(h2) > 0:
        plt.plot(h2[:, 0], h2[:, 1], 'r-', linewidth=3, label='Beam 2 (Proton)', alpha=0.6)
        
    # Plot Children
    for p in particles:
        if p.id > 2:  # Child
            h = np.array(p.history_q)
            if len(h) > 0:
                # Color by momentum magnitude (Energy)
                plt.plot(h[:, 0], h[:, 1], '-', linewidth=1, alpha=0.8)
                
    # Mark Collision
    if collision_point is not None:
        plt.plot(collision_point[0], collision_point[1], 'k*', markersize=20, label='Interaction Vertex')
        
    plt.title("TNFR Collider Event Display: Structural Bifurcation")
    plt.xlabel("x (Structural Dimension 1)")
    plt.ylabel("y (Structural Dimension 2)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axis('equal')
    
    plt.savefig(results_dir / "01_collision_event.png")
    print(f"Saved event display to {results_dir / '01_collision_event.png'}")
    
    # Energy Histogram
    energies = []
    for p in particles:
        if p.id > 2:
            e = np.linalg.norm(p.p)**2 / (2 * p.mass)
            energies.append(e)
            
    plt.figure(figsize=(10, 6))
    plt.hist(energies, bins=20, color='orange', edgecolor='black', alpha=0.7)
    plt.title("Decay Product Energy Spectrum")
    plt.xlabel("Energy (Structural Frequency)")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    
    plt.savefig(results_dir / "02_energy_spectrum.png")
    print(f"Saved spectrum to {results_dir / '02_energy_spectrum.png'}")


if __name__ == "__main__":
    run_collider_demo()
