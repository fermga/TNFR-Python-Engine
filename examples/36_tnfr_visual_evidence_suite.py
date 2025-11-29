import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# Ensure results directory exists
OUTPUT_DIR = "results/visual_evidence"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_etheric_vortex_gif():
    """Generates an animation of the Rankine Vortex Velocity Field."""
    print("Generating Etheric Vortex GIF...")
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="black")
    ax.set_facecolor("black")

    # Grid
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)

    # Rankine Vortex Model
    Rc = 2.0  # Core radius
    Gamma = 10.0

    # Streamplot for background structure
    # Alpha not supported directly in streamplot, using RGBA color
    ax.streamplot(X, Y, -Y, X, color=(0, 1, 1, 0.3), linewidth=0.5, density=1.5, arrowsize=0.5)

    # Particles for animation
    num_particles = 200
    particles = np.random.rand(num_particles, 2) * 20 - 10
    scat = ax.scatter(particles[:, 0], particles[:, 1], c="white", s=2, alpha=0.8)

    circle = plt.Circle((0, 0), Rc, color="blue", fill=False, linestyle="--", label="Vortex Core")
    ax.add_artist(circle)

    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_title("The Etheric Vortex (Rankine Model)", color="white")
    ax.axis("off")

    def update(frame):
        # Update particle positions based on vortex velocity
        r = np.sqrt(particles[:, 0]**2 + particles[:, 1]**2)

        # Tangential velocity
        v_theta = np.where(r < Rc, Gamma * r / (2 * np.pi * Rc**2), Gamma / (2 * np.pi * r))

        # Convert to cartesian components
        theta = np.arctan2(particles[:, 1], particles[:, 0])
        theta_new = theta + v_theta / r * 0.1  # dt = 0.1

        particles[:, 0] = r * np.cos(theta_new)
        particles[:, 1] = r * np.sin(theta_new)

        # Reset particles that go too far or get stuck in center
        mask = (r > 10) | (r < 0.1)
        particles[mask] = np.random.rand(np.sum(mask), 2) * 20 - 10

        scat.set_offsets(particles)
        return scat,

    ani = animation.FuncAnimation(fig, update, frames=100, interval=50, blit=True)
    ani.save(f"{OUTPUT_DIR}/etheric_vortex.gif", writer="pillow", fps=20)
    plt.close()


def create_cymatic_planet_gif():
    """Generates an animation of a planetary resonant orbit (Lissajous)."""
    print("Generating Cymatic Planet GIF...")
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="black")
    ax.set_facecolor("black")

    # Venus-Earth Resonance (Rose of Venus approx 13:8)
    # Simplified Lissajous/Epitrochoid
    R = 5
    r = 2
    d = 1.5

    line, = ax.plot([], [], color="magenta", linewidth=1.5, alpha=0.8)
    planet, = ax.plot([], [], "o", color="white", markersize=8)
    sun, = ax.plot([0], [0], "o", color="yellow", markersize=15)

    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    ax.set_title("Planetary Resonance (Cymatic Node)", color="white")
    ax.axis("off")

    # Precompute path
    full_t = np.linspace(0, 10 * np.pi, 500)
    # Epitrochoid equations
    x_path = (R - r) * np.cos(full_t) + d * np.cos((R - r) / r * full_t)
    y_path = (R - r) * np.sin(full_t) - d * np.sin((R - r) / r * full_t)

    def update(frame):
        current_idx = frame % len(full_t)
        # Draw trail
        line.set_data(x_path[:current_idx], y_path[:current_idx])
        # Draw planet
        planet.set_data([x_path[current_idx]], [y_path[current_idx]])
        return line, planet

    ani = animation.FuncAnimation(fig, update, frames=len(full_t), interval=20, blit=True)
    ani.save(f"{OUTPUT_DIR}/cymatic_planet.gif", writer="pillow", fps=30)
    plt.close()


def create_dna_antenna_gif():
    """Generates an animation of DNA interacting with a helical field."""
    print("Generating DNA Antenna GIF...")
    fig = plt.figure(figsize=(8, 8), facecolor="black")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("black")
    ax.grid(False)
    ax.axis("off")

    # DNA Helix
    z = np.linspace(0, 10, 100)
    r = 1
    x1 = r * np.cos(z * 2)
    y1 = r * np.sin(z * 2)
    x2 = r * np.cos(z * 2 + np.pi)
    y2 = r * np.sin(z * 2 + np.pi)

    ax.plot(x1, y1, z, color="cyan", linewidth=2, alpha=0.8)
    ax.plot(x2, y2, z, color="magenta", linewidth=2, alpha=0.8)

    # Base pairs
    for i in range(len(z)):
        if i % 5 == 0:
            ax.plot([x1[i], x2[i]], [y1[i], y2[i]], [z[i], z[i]], color="white", alpha=0.3)

    # Field Waves (Animated)
    waves = []
    for i in range(5):
        wave, = ax.plot([], [], [], color="yellow", alpha=0.0)  # Start invisible
        waves.append(wave)

    ax.set_title("DNA: Fractal Antenna for Etheric Waves", color="white")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(0, 10)

    def update(frame):
        phase = frame * 0.2

        # Animate a wave traveling down the helix
        # Wave is a spiral matching the DNA pitch

        for j, wave in enumerate(waves):
            offset = j * 2
            z_wave = np.linspace(0, 10, 50)
            # Wave travels down
            z_eff = (z_wave + phase + offset) % 10

            # Sort for plotting
            idx = np.argsort(z_eff)
            z_eff = z_eff[idx]

            # Spiral field
            r_field = 1.5
            x_field = r_field * np.cos(z_eff * 2 + phase)
            y_field = r_field * np.sin(z_eff * 2 + phase)

            wave.set_data(x_field, y_field)
            wave.set_3d_properties(z_eff)
            wave.set_alpha(0.5)

        return waves

    ani = animation.FuncAnimation(fig, update, frames=50, interval=50, blit=False)
    ani.save(f"{OUTPUT_DIR}/dna_antenna.gif", writer="pillow", fps=20)
    plt.close()


def create_solar_analemma_gif():
    """Generates an animation of the Solar Analemma formation."""
    print("Generating Solar Analemma GIF...")
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="black")
    ax.set_facecolor("black")

    # Analemma Parameters
    days = np.linspace(0, 365, 365)
    # Declination (North-South)
    declination = 23.44 * np.sin(2 * np.pi * (days + 284) / 365)
    # Equation of Time (East-West) - Simplified
    eot = 9.87 * np.sin(2 * 2 * np.pi * (days - 81) / 365) - 7.53 * np.cos(2 * np.pi * (days - 81) / 365) - 1.5 * np.sin(2 * np.pi * (days - 81) / 365)

    line, = ax.plot([], [], color="gold", linewidth=2, alpha=0.8)
    sun, = ax.plot([], [], "o", color="yellow", markersize=12, markeredgecolor="orange", markeredgewidth=2)

    ax.set_xlim(-20, 20)
    ax.set_ylim(-30, 30)
    ax.set_title("The Solar Analemma (Vortex Breathing)", color="white")
    ax.set_xlabel("Equation of Time (Minutes)", color="white")
    ax.set_ylabel("Declination (Degrees)", color="white")
    ax.grid(True, color="gray", alpha=0.3)
    ax.tick_params(colors="white")

    def update(frame):
        # Draw trail up to current frame
        line.set_data(eot[:frame], declination[:frame])
        # Draw sun
        if frame > 0:
            sun.set_data([eot[frame - 1]], [declination[frame - 1]])
        return line, sun

    ani = animation.FuncAnimation(fig, update, frames=len(days), interval=20, blit=True)
    ani.save(f"{OUTPUT_DIR}/solar_analemma.gif", writer="pillow", fps=30)
    plt.close()


if __name__ == "__main__":
    print("Starting Visual Evidence Generation...")
    create_etheric_vortex_gif()
    create_cymatic_planet_gif()
    create_dna_antenna_gif()
    create_solar_analemma_gif()
    print(f"Visual evidence generated in {OUTPUT_DIR}")
