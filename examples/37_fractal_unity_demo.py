import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# Ensure results directory exists
OUTPUT_DIR = "results/visual_evidence"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_fractal_unity_gif():
    """
    Demonstrates that Galaxies, Shells, and Flowers are the same 
    Logarithmic Spiral equation with different parameters.
    
    Equation: r = a * e^(b * theta)
    """
    print("Generating Fractal Unity GIF...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='black')
    fig.suptitle("One Equation, Infinite Forms: r = a * exp(b * theta)", color='white', fontsize=16)
    
    for ax in axes:
        ax.set_facecolor('black')
        ax.axis('off')
        ax.set_aspect('equal')

    # Parameters for the three forms
    # 1. Galaxy (Low growth rate, many turns)
    # 2. Nautilus (Medium growth rate, Golden Ratio)
    # 3. Flower (Phyllotaxis, discrete points)
    
    theta_max = 6 * np.pi
    theta = np.linspace(0, theta_max, 1000)
    
    # Galaxy
    b_galaxy = 0.15
    r_galaxy = np.exp(b_galaxy * theta)
    
    # Nautilus
    b_nautilus = 0.306 # Golden Spiral approx
    r_nautilus = np.exp(b_nautilus * theta)
    
    # Flower (Phyllotaxis)
    golden_angle = np.pi * (3 - np.sqrt(5))
    n_seeds = 300
    indices = np.arange(n_seeds)
    theta_flower = indices * golden_angle
    r_flower = np.sqrt(indices) # Fermat's spiral for packing

    # Initialize plots
    galaxy_line, = axes[0].plot([], [], color='cyan', lw=1, alpha=0.8)
    axes[0].set_title("Cosmos (Galaxy)\nb = 0.15", color='cyan')
    
    nautilus_line, = axes[1].plot([], [], color='orange', lw=2)
    axes[1].set_title("Matter (Nautilus)\nb = 0.306 (Golden)", color='orange')
    
    flower_scatter = axes[2].scatter([], [], c='magenta', s=20)
    axes[2].set_title("Life (Sunflower)\nFermat Spiral", color='magenta')

    # Set limits
    axes[0].set_xlim(-20, 20)
    axes[0].set_ylim(-20, 20)
    axes[1].set_xlim(-50, 50)
    axes[1].set_ylim(-50, 50)
    axes[2].set_xlim(-20, 20)
    axes[2].set_ylim(-20, 20)

    def update(frame):
        # Animate Galaxy (Rotation)
        rot = frame * 0.05
        x_gal = r_galaxy * np.cos(theta + rot)
        y_gal = r_galaxy * np.sin(theta + rot)
        # Add symmetric arms
        x_gal2 = r_galaxy * np.cos(theta + rot + np.pi)
        y_gal2 = r_galaxy * np.sin(theta + rot + np.pi)
        
        galaxy_line.set_data(np.concatenate([x_gal, x_gal2]), np.concatenate([y_gal, y_gal2]))
        
        # Animate Nautilus (Growth)
        idx = int((frame % 100) / 100 * len(theta))
        x_nau = r_nautilus[:idx] * np.cos(theta[:idx])
        y_nau = r_nautilus[:idx] * np.sin(theta[:idx])
        nautilus_line.set_data(x_nau, y_nau)
        
        # Animate Flower (Blooming)
        seeds = int((frame % 100) / 100 * n_seeds) + 10
        x_flo = r_flower[:seeds] * np.cos(theta_flower[:seeds])
        y_flo = r_flower[:seeds] * np.sin(theta_flower[:seeds])
        flower_scatter.set_offsets(np.c_[x_flo, y_flo])
        
        return galaxy_line, nautilus_line, flower_scatter

    ani = animation.FuncAnimation(fig, update, frames=100, interval=50, blit=False)
    ani.save(f"{OUTPUT_DIR}/fractal_unity.gif", writer='pillow', fps=20)
    plt.close()

if __name__ == "__main__":
    create_fractal_unity_gif()
    print(f"Fractal Unity demonstration generated in {OUTPUT_DIR}")
