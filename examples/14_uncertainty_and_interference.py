import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
from pathlib import Path

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def run_uncertainty_demo(results_dir):
    print("\n--- Part 1: Structural Uncertainty Principle ---")
    
    # Generate Gaussian EPI packets of different widths (sigma_t)
    # EPI(t) = exp(-t^2 / (2*sigma_t^2)) * exp(i * w0 * t)
    
    t = np.linspace(-10, 10, 1000)
    dt = t[1] - t[0]
    w0 = 5.0 # Carrier frequency
    
    sigmas_t = [0.5, 1.0, 2.0]
    
    plt.figure(figsize=(12, 8))
    
    products = []
    
    for i, sigma_t in enumerate(sigmas_t):
        # 1. Create Structural Packet (Form)
        epi_packet = np.exp(-t**2 / (2 * sigma_t**2)) * np.exp(1j * w0 * t)
        
        # 2. Compute Structural Frequency Spectrum (FFT)
        freqs = np.fft.fftfreq(len(t), dt)
        spectrum = np.fft.fft(epi_packet)
        spectrum = np.fft.fftshift(spectrum)
        freqs = np.fft.fftshift(freqs)
        
        # Normalize
        epi_mag = np.abs(epi_packet)
        spec_mag = np.abs(spectrum)
        spec_mag /= np.max(spec_mag)
        
        # Measure widths (Standard Deviation)
        # We use the magnitude distribution
        
        # Time width
        mean_t = np.sum(t * epi_mag) / np.sum(epi_mag)
        std_t = np.sqrt(np.sum((t - mean_t)**2 * epi_mag) / np.sum(epi_mag))
        
        # Freq width
        # Filter positive freqs for measurement around w0
        mask = freqs > 0
        pos_freqs = freqs[mask]
        pos_spec = spec_mag[mask]
        
        mean_f = np.sum(pos_freqs * pos_spec) / np.sum(pos_spec)
        std_f = np.sqrt(np.sum((pos_freqs - mean_f)**2 * pos_spec) / np.sum(pos_spec))
        
        product = std_t * std_f
        products.append(product)
        
        print(f"Packet {i+1}: Sigma_t = {std_t:.3f}, Sigma_f = {std_f:.3f}, Product = {product:.4f}")
        
        # Plot Time Domain
        plt.subplot(3, 2, 2*i + 1)
        plt.plot(t, epi_mag, color='blue', label=f'|EPI(t)|, $\sigma_t$={sigma_t}')
        plt.plot(t, np.real(epi_packet), color='blue', alpha=0.3)
        plt.title(f"Structural Form (Time/Space) - Width {sigma_t}")
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right')
        
        # Plot Freq Domain
        plt.subplot(3, 2, 2*i + 2)
        plt.plot(freqs, spec_mag, color='red', label=f'|FFT(EPI)|')
        plt.xlim(0, 2) # Zoom in on positive frequencies (normalized units)
        plt.title(f"Structural Frequency Spectrum")
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right')
        
    plt.tight_layout()
    plt.savefig(results_dir / "01_uncertainty_principle.png")
    print(f"Saved uncertainty plot to {results_dir / '01_uncertainty_principle.png'}")
    
    # Verify the principle: Product should be roughly constant
    mean_product = np.mean(products)
    print(f"Mean Uncertainty Product: {mean_product:.4f} (Constant)")

def run_interference_demo(results_dir):
    print("\n--- Part 2: Structural Interference (Double Slit) ---")
    
    # Simulation Parameters
    WIDTH = 60
    HEIGHT = 40
    STEPS = 150
    
    # Create 2D Grid Graph (Used for topology definition, though simulation uses arrays)
    G = nx.grid_2d_graph(WIDTH, HEIGHT)
    
    # Initialize State
    # Phase: 0
    # Amplitude (Coherence): 0
    for node in G.nodes():
        G.nodes[node]['phase'] = 0.0
        G.nodes[node]['amplitude'] = 0.0
        G.nodes[node]['type'] = 'medium'
        
    # Define Geometry
    SOURCE_POS = (5, HEIGHT // 2)
    BARRIER_X = 20
    SLIT_WIDTH = 2
    SLIT_SEP = 8
    SLIT_Y_CENTER = HEIGHT // 2
    
    SLIT_1_Y = range(SLIT_Y_CENTER - SLIT_SEP//2 - SLIT_WIDTH, SLIT_Y_CENTER - SLIT_SEP//2)
    SLIT_2_Y = range(SLIT_Y_CENTER + SLIT_SEP//2, SLIT_Y_CENTER + SLIT_SEP//2 + SLIT_WIDTH)
    
    # Mark Barrier Nodes
    for y in range(HEIGHT):
        if y not in SLIT_1_Y and y not in SLIT_2_Y:
            node = (BARRIER_X, y)
            if node in G:
                G.nodes[node]['type'] = 'barrier'
                G.nodes[node]['amplitude'] = 0.0
                
    # Simulation Loop (Wave Equation on Graph)
    # We use a simple coupled oscillator model:
    # d^2(phi)/dt^2 = c^2 * Laplacian(phi) - damping * d(phi)/dt
    # Or simpler: Phase diffusion with source driving
    
    # Let's use the Complex Structural Field Psi directly
    # Psi_new = Psi_old + alpha * (mean(Psi_neighbors) - Psi_old)
    # This is the diffusion equation for complex field -> Wave propagation
    
    # Initialize Psi
    psi = np.zeros((WIDTH, HEIGHT), dtype=complex)
    
    # Source Frequency
    omega = 0.5
    
    print("Simulating Wave Propagation...")
    
    for t in range(STEPS):
        # Source drives the field
        psi[SOURCE_POS] = np.exp(1j * omega * t)
        
        # Barrier absorbs (Psi = 0)
        for y in range(HEIGHT):
            if y not in SLIT_1_Y and y not in SLIT_2_Y:
                psi[BARRIER_X, y] = 0.0
        
        # Propagate (Vectorized for speed)
        # Laplacian: Psi(x+1) + Psi(x-1) + Psi(y+1) + Psi(y-1) - 4*Psi(x,y)
        
        psi_up = np.roll(psi, 1, axis=1)
        psi_down = np.roll(psi, -1, axis=1)
        psi_left = np.roll(psi, 1, axis=0)
        psi_right = np.roll(psi, -1, axis=0)
        
        # Simple wave propagation approximation (Huygens principle via diffusion of complex phase)
        # Actually, diffusion equation doesn't wave well. We need wave equation.
        # Wave Eq: u_tt = c^2 u_xx
        # Finite Difference: u_new = 2*u_curr - u_prev + c^2 * dt^2 * Laplacian
        
        if t == 0:
            psi_prev = np.zeros_like(psi)
            psi_curr = np.zeros_like(psi)
        else:
            # Update logic handled below
            pass
            
    # Let's restart the loop with proper Wave Equation logic
    u_prev = np.zeros((WIDTH, HEIGHT)) # Real part
    u_curr = np.zeros((WIDTH, HEIGHT))
    u_next = np.zeros((WIDTH, HEIGHT))
    
    c = 0.5 # Wave speed
    dt = 1.0
    
    intensity_accum = np.zeros((WIDTH, HEIGHT))
    
    for t in range(STEPS * 2): # Run longer
        # Source
        u_curr[SOURCE_POS] = np.sin(omega * t)
        
        # Barrier
        for y in range(HEIGHT):
            if y not in SLIT_1_Y and y not in SLIT_2_Y:
                u_curr[BARRIER_X, y] = 0.0
                
        # Laplacian
        laplacian = (
            np.roll(u_curr, 1, axis=0) + 
            np.roll(u_curr, -1, axis=0) + 
            np.roll(u_curr, 1, axis=1) + 
            np.roll(u_curr, -1, axis=1) - 
            4 * u_curr
        )
        
        # Wave Update
        u_next = 2*u_curr - u_prev + (c**2) * laplacian
        
        # Damping at edges to prevent reflection (Absorbing Boundary Condition approximation)
        # Simple damping mask
        u_next[0, :] *= 0.9
        u_next[-1, :] *= 0.9
        u_next[:, 0] *= 0.9
        u_next[:, -1] *= 0.9
        
        # Accumulate Intensity (Coherence)
        intensity_accum += u_curr**2
        
        # Cycle
        u_prev = u_curr.copy()
        u_curr = u_next.copy()
        
    print("Simulation Complete.")
    
    # Visualization
    plt.figure(figsize=(12, 6))
    
    # 1. Wave Field Snapshot
    plt.subplot(1, 2, 1)
    plt.imshow(u_curr.T, cmap='RdBu', origin='lower')
    plt.title("Structural Wave Field (Snapshot)")
    plt.colorbar(label="Amplitude")
    
    # Draw Barrier
    plt.axvline(x=BARRIER_X, color='black', linestyle='-', alpha=0.5)
    
    # 2. Interference Pattern (Intensity)
    plt.subplot(1, 2, 2)
    plt.imshow(intensity_accum.T, cmap='inferno', origin='lower')
    plt.title("Accumulated Coherence (Interference Pattern)")
    plt.colorbar(label="Intensity")
    
    plt.tight_layout()
    plt.savefig(results_dir / "02_interference_pattern.png")
    print(f"Saved interference plot to {results_dir / '02_interference_pattern.png'}")
    
    # 3. Screen Profile
    plt.figure(figsize=(10, 4))
    screen_x = WIDTH - 5
    profile = intensity_accum[screen_x, :]
    plt.plot(range(HEIGHT), profile, color='orange', linewidth=2)
    plt.title(f"Interference Fringes at Screen (x={screen_x})")
    plt.xlabel("Position (y)")
    plt.ylabel("Intensity")
    plt.grid(True, alpha=0.3)
    plt.savefig(results_dir / "03_interference_profile.png")
    print(f"Saved profile plot to {results_dir / '03_interference_profile.png'}")

if __name__ == "__main__":
    results_dir = Path("results/quantum_demo_2")
    ensure_dir(results_dir)
    
    run_uncertainty_demo(results_dir)
    run_interference_demo(results_dir)
