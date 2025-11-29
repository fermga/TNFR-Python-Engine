# Quantum Mechanics from Nodal Dynamics

TNFR posits that Quantum Mechanics is the **High Dissonance / High Phase Gradient** regime of Nodal Dynamics. In this regime, the continuous trajectories of Classical Mechanics break down, and structural stability is only possible at discrete resonant modes (eigenstates).

## Theoretical Correspondence

| Quantum Concept | Symbol | TNFR Structural Equivalent | Symbol | Relation |
|-----------------|--------|----------------------------|--------|----------|
| **Wavefunction** | $\psi$ | Complex Structural Field | $\Psi = K_\phi + i J_\phi$ | $\psi \sim \Psi$ |
| **Energy** | $E$ | Structural Frequency | $\nu_f$ | $E \propto \nu_f$ |
| **Potential** | $V(x)$ | Structural Potential | $\Phi_s(x)$ | $V \sim \Phi_s$ |
| **Quantization** | $n$ | Topological Winding Number | $w$ | $\oint \nabla \phi = 2\pi w$ |
| **Collapse** | - | Decoherence / Stabilization | U2 | Operator `IL` |

## Emergent Quantization Mechanism

In TNFR, quantization is not a postulate but a geometric necessity of **Resonant Stability**.

1.  **Continuous Evolution**: A node evolves according to $\partial EPI / \partial t = \nu_f \cdot \Delta NFR$.
2.  **Self-Interaction**: In a bounded domain (box/well), the node's structural ripples (phase waves) reflect and interact with the node.
3.  **Interference**:
    *   **Constructive**: If the phase accumulates to $2\pi n$ per cycle, the self-interaction reinforces the pattern (Resonance). $\Delta NFR \to 0$.
    *   **Destructive**: If the phase is mismatched, the self-interaction creates dissonance. $\Delta NFR$ remains high.
4.  **Selection**: The Nodal Equation drives the system to minimize $\Delta NFR$. Thus, the system naturally drifts away from non-resonant frequencies and settles into resonant modes (Eigenstates).

## The "Particle in a Box" Experiment

We simulate a TNFR node confined in a 1D structural cavity of length $L$.
- The node emits phase waves that reflect off boundaries.
- The node adjusts its internal frequency $\nu_f$ (Energy) to maximize Coherence (minimize $\Delta NFR$).
- **Prediction**: Starting from random frequencies, nodes will converge to discrete levels $\nu_n \propto n^2$ (or $n$ depending on dispersion), reproducing the quantum energy spectrum.

## Visual Proofs

The accompanying script `examples/13_quantum_mechanics_demo.py` generates:

- `results/quantum_demo/01_quantization_levels.png`: Histogram showing convergence to discrete frequencies (Energy Levels).
- `results/quantum_demo/02_convergence.png`: Trajectories of nodes settling into eigenstates over time.
- `results/quantum_demo/03_wavefunctions.png`: Reconstruction of the structural standing waves corresponding to the first few modes.

This experiment demonstrates that **Quantization emerges naturally** from the Nodal Equation when applied to bounded structural manifolds, without requiring explicit quantum postulates.

