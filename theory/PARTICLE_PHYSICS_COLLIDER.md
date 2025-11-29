# High-Energy Particle Physics from Nodal Dynamics

TNFR models subatomic particles not as fundamental point-masses, but as **Coherent Structural Modes** (EPIs) of the underlying field. High-energy collisions (like those at CERN) are interpreted as **Structural Bifurcation Events**.

## The Collider Experiment

In a collider, two stable high-frequency nodes (e.g., Protons) are accelerated to relativistic velocities and smashed together.

### TNFR Interpretation

1.  **Acceleration**: Increasing the Phase Current ($J_\phi$) of the nodes.
2.  **Collision**: When nodes overlap, their structural fields interfere.
    -   If phases are aligned: Resonance (Merger).
    -   If phases are misaligned (typical): Extreme Dissonance ($\Delta NFR \to \infty$).
3.  **Bifurcation (The "Bang")**:
    -   The structural pressure exceeds the stability threshold of the parent nodes.
    -   **Grammar Rule U4 (Bifurcation)** activates: The system must reorganize to minimize $\Delta NFR$.
    -   **Fragmentation**: The high energy ($\nu_f$) is distributed into multiple new stable modes (Child Nodes / Particles).
4.  **Jets**: The child nodes propagate outward, carrying the momentum of the parents.

## Conservation Laws

In TNFR, conservation emerges from the continuity of the structural field:
-   **Energy Conservation**: $\sum \nu_{f, \text{parents}} \approx \sum \nu_{f, \text{children}}$ (Structural Frequency is conserved).
-   **Momentum Conservation**: $\sum \mathbf{J}_{\phi, \text{parents}} = \sum \mathbf{J}_{\phi, \text{children}}$ (Phase Current is conserved).

## Simulation: The "Higgs" Event

We simulate a head-on collision of two "Proton" nodes.
-   **Input**: 2 Nodes with high $p$ and high $\nu_f$.
-   **Interaction**: Short-range repulsive potential (Strong Force analog).
-   **Event**: At $r < r_{crit}$, the nodes "shatter" into a shower of lighter nodes.
-   **Output**: Visualization of the particle tracks (Jets) and energy deposition.

## Visual Proofs

The script `examples/16_particle_collider_demo.py` generates:
-   `results/collider_demo/01_collision_event.png`: A classic "event display" showing the incoming beams and the explosion of tracks.
-   `results/collider_demo/02_energy_spectrum.png`: Histogram of the decay product energies.
