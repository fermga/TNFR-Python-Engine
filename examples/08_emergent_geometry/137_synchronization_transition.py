#!/usr/bin/env python3
"""
Example 137 — Auxiliary circular-mean synchronization sweep
==============================================================================

This example changes register from the diffusion/heat-kernel arc (ex 99, 134,
135, 136) to a separate phase-oscillator comparison. The phase component read
by canonical DeltaNFR contains the circular-neighbour mismatch

    g_phase(i) = -angle_diff(theta_i, theta_bar_neighbours) / pi

(dnfr.py). The auxiliary phase dynamics used here is

    dtheta_i/dt = nu_f_i + K * angle_diff(theta_bar_neighbours, theta_i)

which is a Kuramoto-type coupled-oscillator model. It is not the nodal EPI
equation, does not apply a canonical operator word, and does not establish the
order or universality class of a TNFR phase transition.

Doctrine compliance
-------------------
The coupling kernel is the same pull toward the circular mean used by TNFR
phase telemetry. The example verifies numerically that the
vectorized coupling angle(sum_j A_ij e^{i theta_j}) - theta_i equals the
canonical neighbour-circular-mean pull (neighbor_phase_mean_list + angle_diff).
The order parameter R is the Kuramoto order. The time evolution, coupling sweep
and threshold read-outs are explicitly imposed experimental choices.

NOTE (honest): the canonical coupling uses the ANGLE to the neighbour circular
mean, not the textbook sin-sum form K/N * sum sin(theta_j - theta_i). The two
are the same Kuramoto-type attraction toward the mean field, but the exact
mean-field threshold formula K_c = 2 sigma sqrt(2/pi) is for the sin-sum form;
here the threshold is measured, and only its STRUCTURE (proportional to the nu_f
dispersion) is claimed, not the textbook constant.

Three measured results
----------------------
M1 SYNCHRONIZATION CROSSOVER. Sweeping the coupling K on an all-to-all
   network with heterogeneous nu_f, the Kuramoto order parameter R rises from
   ~0 (incoherent drift) toward collective lock on the sampled grid. The finite
   sweep does not identify a transition order.

M2 FREQUENCY-DISPERSION TREND. The selected crossing statistic K_c (the first
   sampled K where R exceeds 1/2) grows with the frequency dispersion sigma in
   this finite six-seed protocol. The 1/2 crossing is a reporting policy, not a
   canonical critical threshold.

M3 LONG-RANGE PHASE ORDER. On a 2D lattice the phase correlation
   C(r) = <cos(theta_i - theta_{i+r})> decays fast below threshold (short-range
   order, the walker's phases are uncorrelated beyond a few steps) and develops
   long-range order above threshold (C(r) stays high across the lattice). The
   correlation range grows across the sampled crossover. This raw phase
   correlation is not the canonical xi_C estimator.

Honest scope
------------
The example establishes finite measurements only for the displayed auxiliary
model, integration scheme, graph sizes, seeds and parameter grid. Its coupling
kernel matches one TNFR phase mismatch construction, but this does not make its
trajectory an execution of the nodal EPI equation. A phase-transition theorem,
finite-size limit and universality comparison remain open.

References
----------
- src/tnfr/dynamics/dnfr.py (the phase channel g_phase of dNFR)
- src/tnfr/dynamics/coordination.py (canonical global/local phase coordination)
- src/tnfr/observers.py (kuramoto_order, the order parameter R)
- src/tnfr/metrics/trig.py (neighbor_phase_mean_list, the circular mean)
- AGENTS.md "Transport Content of the Nodal Equation" (phase channel -> Kuramoto)
- examples/08_emergent_geometry/135_arrow_of_time_h_theorem.py (EPI channel arc)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import networkx as nx
import numpy as np

from tnfr.metrics.trig import neighbor_phase_mean_list
from tnfr.utils import angle_diff


def wrap(x):
    """Wrap angle(s) to (-pi, pi]."""
    return (x + np.pi) % (2 * np.pi) - np.pi


def evolve(A, theta, omega, K, steps=400, dt=0.05):
    """Integrate dtheta_i = omega_i + K * (circular-mean-neighbour pull).

    The coupling wrap(angle(sum_j A_ij e^{i theta_j}) - theta_i) is the
    canonical TNFR phase channel (pull toward the neighbour circular mean).
    """
    for _ in range(steps):
        z = A @ np.exp(1j * theta)
        theta = theta + dt * (omega + K * wrap(np.angle(z) - theta))
    return theta


def order_param(theta):
    """Kuramoto order parameter R = |<e^{i theta}>| in [0, 1]."""
    return float(abs(np.mean(np.exp(1j * theta))))


def experiment_1_transition():
    """M1: the synchronization transition R(K), with the canonical anchor."""
    print("=" * 74)
    print("M1: SAMPLED SYNCHRONIZATION CROSSOVER R(K)")
    print("=" * 74)
    # anchor: vectorized coupling == canonical phase channel
    G = nx.cycle_graph(8)
    th = np.random.default_rng(0).uniform(0, 2 * np.pi, 8)
    nodes = list(G.nodes())
    A = nx.to_numpy_array(G)
    vec0 = wrap(np.angle((A @ np.exp(1j * th)))[0] - th[0])
    cm = {n: np.cos(th[i]) for i, n in enumerate(nodes)}
    sm = {n: np.sin(th[i]) for i, n in enumerate(nodes)}
    canon = angle_diff(
        neighbor_phase_mean_list(list(G.neighbors(0)), cm, sm, fallback=th[0]), th[0]
    )
    print(
        f"  anchor: vectorized coupling = {vec0:+.6f}, canonical phase channel"
        f" = {canon:+.6f} (|diff|={abs(vec0 - canon):.0e})"
    )
    print("  -> the coupling IS the canonical neighbour-circular-mean pull.")
    print()
    N = 200
    A = nx.to_numpy_array(nx.complete_graph(N))
    print(f"  all-to-all N={N}, nu_f dispersion sigma=0.5")
    print(f"  {'K':>6} {'R':>8} {'phase':>14}")
    for K in [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0]:
        rng = np.random.default_rng(1)
        om = rng.normal(0, 0.5, N)
        om -= om.mean()
        th = evolve(A, rng.uniform(0, 2 * np.pi, N), om, K)
        R = order_param(th)
        tag = "incoherent" if R < 0.3 else ("critical" if R < 0.7 else "synchronized")
        print(f"  {K:>6.2f} {R:>8.4f} {tag:>14}")
    print()
    print("  -> R rises across this finite coupling grid.")
    print("     The sweep does not determine a transition order.")


def experiment_2_threshold_vs_dispersion():
    """M2: the threshold K_c grows linearly with the nu_f dispersion sigma."""
    print()
    print("=" * 74)
    print("M2: SAMPLED CROSSING VS FREQUENCY DISPERSION")
    print("=" * 74)
    print("K_c = first sampled coupling where R exceeds 1/2 (6-seed average).")
    print("The 1/2 cut and coupling grid are reporting choices.")
    print("More heterogeneous nu_f needs stronger coupling to lock.")
    print()
    N = 200
    A = nx.to_numpy_array(nx.complete_graph(N))
    Ks = np.linspace(0.02, 1.5, 24)
    print(f"  {'sigma':>6} {'K_c':>8} {'K_c/sigma':>10}")
    for sigma in [0.2, 0.4, 0.6, 0.8, 1.0]:
        Kcs = []
        for s in range(6):
            rng = np.random.default_rng(50 + s)
            om = rng.normal(0, sigma, N)
            om -= om.mean()
            Rs = np.array(
                [
                    order_param(
                        evolve(A, rng.uniform(0, 2 * np.pi, N), om, K, steps=250)
                    )
                    for K in Ks
                ]
            )
            Kcs.append(Ks[np.argmax(Rs > 0.5)] if Rs.max() > 0.5 else np.nan)
        Kc = float(np.nanmean(Kcs))
        print(f"  {sigma:>6.1f} {Kc:>8.3f} {Kc / sigma:>10.3f}")
    print()
    print("  -> the sampled crossing grows with sigma in this protocol.")


def experiment_3_long_range_order():
    """M3: phase correlation C(r) develops long-range order above threshold."""
    print()
    print("=" * 74)
    print("M3: FINITE-TORUS PHASE-CORRELATION RANGE")
    print("=" * 74)
    print("On a 2D lattice, C(r) = <cos(theta_i - theta_{i+r})> decays fast")
    print("below threshold and stays high across the lattice above it.")
    print()
    L = 24
    A = nx.to_numpy_array(nx.grid_2d_graph(L, L, periodic=True))
    coords = [(x, y) for x in range(L) for y in range(L)]
    idx = {c: i for i, c in enumerate(coords)}
    print(f"  2D torus L={L}")
    print(f"  {'K':>6} {'R':>7}   C(r) for r=1..6")
    for K in [0.2, 0.6, 1.0, 2.0, 4.0]:
        rng = np.random.default_rng(3)
        om = rng.normal(0, 0.3, L * L)
        om -= om.mean()
        th = evolve(A, rng.uniform(0, 2 * np.pi, L * L), om, K)
        Cr = []
        for r in range(1, 7):
            vals = [
                np.cos(th[idx[(x, y)]] - th[idx[((x + r) % L, y)]])
                for x in range(L)
                for y in range(L)
            ]
            Cr.append(float(np.mean(vals)))
        print(f"  {K:>6.2f} {order_param(th):>7.3f}   {np.round(Cr, 3)}")
    print()
    print("  -> below threshold C(r) decays to ~0 within a few steps (short-")
    print("     range order); above threshold it stays high across the lattice")
    print("     (long-range order on this finite torus). This does not establish")
    print("     divergence and is not the canonical xi_C estimator.")


def main():
    print()
    print("  ===============================================================")
    print("  Auxiliary Circular-Mean Synchronization Sweep")
    print("  ===============================================================")
    print()
    experiment_1_transition()
    experiment_2_threshold_vs_dispersion()
    experiment_3_long_range_order()
    print()
    print("=" * 74)
    print("FINITE-PROTOCOL RESULT")
    print("=" * 74)
    print("The auxiliary circular-mean model shows a reproducible synchronization")
    print("crossover, a sampled crossing that grows with frequency dispersion,")
    print("and increased phase-correlation range on a finite torus. Its coupling")
    print("kernel matches the TNFR circular-neighbour mismatch, but its trajectory")
    print("is not the nodal EPI equation or a canonical operator word. No")
    print("transition order, divergent correlation length, or universality class")
    print("is inferred from these finite measurements.")


if __name__ == "__main__":
    main()
