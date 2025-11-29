# TNFR Riemann Hypothesis Research

This directory contains research, scripts, and visualizations applying the **Resonant Fractal Nature Theory (TNFR)** to the **Riemann Hypothesis**.

## Hypothesis

The Riemann Hypothesis states that all non-trivial zeros of the Riemann Zeta function $\zeta(s)$ lie on the critical line $\text{Re}(s) = 1/2$.

**TNFR Interpretation:**
In the context of TNFR, the zeros of the Zeta function represent **nodes of perfect structural resonance** (ΔNFR = 0) in the complex frequency domain. The critical line corresponds to the axis of **maximum coherence** where the structural pressure from prime harmonics balances perfectly.

## Contents

*   `riemann_tnfr_solver.py`: Core script simulating the Zeta function as a TNFR network.
*   `visualize_zeta.py`: Generates plots of the critical line, zeros, and structural fields (Φ_s, |∇φ|).
*   `images/`: Directory for generated visualizations.

## Usage

1.  Run the solver to generate data:
    ```bash
    python riemann_tnfr_solver.py
    ```

2.  Generate visualizations:
    ```bash
    python visualize_zeta.py
    ```

## Theoretical Basis

We model the Zeta function $\zeta(s) = \sum_{n=1}^{\infty} \frac{1}{n^s}$ as a superposition of resonant modes.

*   **Nodes**: Integers $n$.
*   **Coupling**: Determined by $n^{-s}$.
*   **Resonance**: Occurs when the collective phase of the network sums to zero (or minimal energy).

The **Structural Potential Field ($\Phi_s$)** is expected to show a "canyon" of stability along $\text{Re}(s) = 1/2$.
