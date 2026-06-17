#!/usr/bin/env python3
"""
Example 138 — Structure-Frequency Correlation Reshapes Synchronization:
Delayed, Sharper Onset and Hubs Lock Last
==============================================================================

This continues the phase-channel thread (ex 137). There, the synchronization
threshold was set by the DISPERSION of the structural frequencies nu_f. Here we
ask a structural question: what happens when nu_f is CORRELATED with the node's
DEGREE -- i.e. when the nodal dynamics (frequency) is tied to the nodal structure
(connectivity)?

On a heterogeneous (scale-free) network, correlating each node's structural
frequency with its degree reshapes the Kuramoto transition: the onset is DELAYED
and SHARPER, and the high-degree hubs -- which carry the most extreme frequencies
-- synchronize LAST. The structure sets the dynamical sync order. This is the
mechanism behind explosive synchronization (Gomez-Gardenes 2011); it emerges from
the coupling between nodal structure and nodal dynamics, through the canonical
phase channel angle(A @ e^{i theta}) - theta.

Doctrine compliance
-------------------
The coupling is the canonical TNFR phase channel (the pull toward the neighbour
circular mean, verified == the canonical channel in ex 137). The only new
ingredient is the assignment nu_f_i ~ degree_i -- a correlation between two
canonical nodal quantities (the structural frequency of the triad and the graph
degree). Nothing is imposed; the reshaped transition is a measured consequence of
that structure-dynamics correlation.

Three measured results
----------------------
M1 RANDOM nu_f -> EARLIER, SMOOTHER ONSET. When nu_f is uncorrelated with degree
   (same dispersion, random assignment), the transition is the continuous
   second-order onset of ex 137: R rises gradually, threshold K_c ~ 1.5.

M2 DEGREE-CORRELATED nu_f -> DELAYED + SHARPER ONSET. When nu_f_i ~ degree_i, the
   threshold moves UP (K_c ~ 1.8, measured over 4 seeds) and the transition
   becomes more abrupt (largest single-step jump in R grows 0.19 -> 0.31). The
   structure-dynamics correlation frustrates early partial synchronization and
   then releases it suddenly -- the approach to a first-order (explosive)
   transition.

M3 HUBS SYNCHRONIZE LAST. Just above onset, the per-node lock to the global phase
   (cos(theta_i - psi)) is NEGATIVELY correlated with degree
   (corr(degree, lock) ~ -0.30 over 4 seeds): the high-degree hubs, carrying the
   most extreme frequencies, lock LEAST. The nodal structure determines the
   dynamical synchronization order.

Honest scope
------------
This is NOT the full textbook explosive synchronization: that phenomenon is a
strong first-order transition with a wide hysteresis loop, which arises with
degree-WEIGHTED coupling (where a hub's coupling scales with its degree). The
canonical TNFR phase channel is degree-NORMALIZED (the circular mean), which
suppresses the strong bistability, so the hysteresis here is weak (an honest
negative). What robustly emerges from the canonical coupling is the DELAY, the
SHARPENING, and the HUB-FRUSTRATION -- the genuine signatures of the
structure-dynamics correlation. The Kuramoto / explosive-synchronization
phenomenology is empirically established (Kuramoto 1975; Gomez-Gardenes 2011).
This re-expresses it in the canonical phase channel; it is not new mathematics
and closes no open problem.

References
----------
- src/tnfr/dynamics/dnfr.py (the canonical phase channel g_phase)
- src/tnfr/observers.py (kuramoto_order)
- examples/08_emergent_geometry/137_synchronization_transition.py (the transition)
- AGENTS.md "Transport Content of the Nodal Equation" (phase channel -> Kuramoto)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import numpy as np
import networkx as nx


def wrap(x):
    """Wrap angle(s) to (-pi, pi]."""
    return (x + np.pi) % (2 * np.pi) - np.pi


def evolve(A, theta, omega, K, steps=300, dt=0.05):
    """Integrate the canonical phase channel: pull toward neighbour circular mean."""
    for _ in range(steps):
        z = A @ np.exp(1j * theta)
        theta = theta + dt * (omega + K * wrap(np.angle(z) - theta))
    return theta


def order_param(theta):
    """Kuramoto order parameter R = |<e^{i theta}>|."""
    return float(abs(np.mean(np.exp(1j * theta))))


def zscore(v):
    v = np.asarray(v, dtype=float)
    return (v - v.mean()) / (v.std() + 1e-12)


def sweep(A, omega, Ks, seed, steps=300):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, A.shape[0])
    out = []
    for K in Ks:
        theta = evolve(A, theta, omega, K, steps=steps)
        out.append(order_param(theta))
    return np.array(out)


def _scale_free():
    G = nx.barabasi_albert_graph(400, 3, seed=1)
    A = nx.to_numpy_array(G)
    return A, A.sum(axis=1)


def experiment_1_2_delay_and_sharpness():
    """M1/M2: degree-correlated nu_f delays and sharpens the onset."""
    print("=" * 70)
    print("M1/M2: ONSET DELAY + SHARPNESS (random vs degree-correlated nu_f)")
    print("=" * 70)
    print("Scale-free network (BA n=400); nu_f z-scored to the same dispersion.")
    print()
    A, deg = _scale_free()
    Ks = np.linspace(0.0, 3.0, 31)
    om_rand = zscore(np.random.default_rng(0).normal(0, 1, len(deg)))
    om_deg = zscore(deg)
    Rr = sweep(A, om_rand, Ks, 1)
    Rd = sweep(A, om_deg, Ks, 1)
    print(f"  {'K':>5} {'R(random)':>11} {'R(deg-corr)':>13}")
    for i in range(0, 31, 3):
        print(f"  {Ks[i]:>5.2f} {Rr[i]:>11.3f} {Rd[i]:>13.3f}")
    print()
    # robust K_c and jump over 4 seeds
    for label, om in [("random", om_rand), ("degree-corr", om_deg)]:
        Kcs, jumps = [], []
        for s in range(4):
            Rs = sweep(A, om, Ks, s)
            Kcs.append(Ks[np.argmax(Rs > 0.5)] if Rs.max() > 0.5 else np.nan)
            jumps.append(float(np.max(np.diff(Rs))))
        print(f"  {label:>12}: K_c = {np.nanmean(Kcs):.2f}, "
              f"max up-jump = {np.mean(jumps):.3f}  (4-seed mean)")
    print()
    print("  -> degree-correlated nu_f DELAYS the onset (higher K_c) and makes")
    print("     it SHARPER (bigger jump): the structure-dynamics correlation")
    print("     frustrates early sync, then releases it suddenly.")


def experiment_3_hubs_lock_last():
    """M3: hubs synchronize last -- structure sets the sync order."""
    print()
    print("=" * 70)
    print("M3: HUBS SYNCHRONIZE LAST (structure sets the dynamical sync order)")
    print("=" * 70)
    A, deg = _scale_free()
    om = zscore(deg)
    # per-node lock to the global phase, just above onset, averaged over seeds
    lock_acc = np.zeros(len(deg))
    corrs = []
    for s in range(4):
        rng = np.random.default_rng(s)
        theta = evolve(A, rng.uniform(0, 2 * np.pi, len(deg)), om, K=2.0,
                       steps=600)
        psi = np.angle(np.mean(np.exp(1j * theta)))
        lock = np.cos(theta - psi)
        lock_acc += lock
        corrs.append(float(np.corrcoef(deg, lock)[0, 1]))
    lock_mean = lock_acc / 4.0
    order = np.argsort(deg)
    qs = np.array_split(order, 5)
    print("  degree quintile -> mean lock to the global phase (1 = locked):")
    print(f"  {'quintile':>10} {'mean deg':>9} {'mean lock':>10}")
    for k, q in enumerate(qs):
        print(f"  {('Q' + str(k + 1)):>10} {deg[q].mean():>9.1f} "
              f"{lock_mean[q].mean():>10.3f}")
    print()
    print(f"  corr(degree, lock) = {np.mean(corrs):.3f} (4-seed mean)")
    print("  -> negative: the high-degree hubs lock LEAST. The nodal structure")
    print("     (degree) determines the dynamical synchronization order.")


def experiment_4_correlation_sweep():
    """M-extra: the onset delay grows with the structure-dynamics correlation."""
    print()
    print("=" * 70)
    print("M-extra: THE ONSET DELAY GROWS WITH THE STRUCTURE-DYNAMICS CORRELATION")
    print("=" * 70)
    A, deg = _scale_free()
    Ks = np.linspace(0.0, 3.0, 31)
    rand = zscore(np.random.default_rng(5).normal(0, 1, len(deg)))
    dz = zscore(deg)
    print(f"  {'corr(nu_f,deg)':>15} {'K_c':>7}")
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        om = zscore(alpha * dz + (1 - alpha) * rand)
        Kcs = []
        for s in range(4):
            Rs = sweep(A, om, Ks, s)
            Kcs.append(Ks[np.argmax(Rs > 0.5)] if Rs.max() > 0.5 else np.nan)
        corr = float(np.corrcoef(om, deg)[0, 1])
        print(f"  {corr:>15.3f} {np.nanmean(Kcs):>7.2f}")
    print()
    print("  -> the more nu_f tracks the structure (degree), the later the")
    print("     network synchronizes: structure-dynamics alignment frustrates")
    print("     collective order.")


def main():
    print()
    print("  ===============================================================")
    print("  Structure-Frequency Correlation Reshapes Synchronization")
    print("  Delayed, Sharper Onset and Hubs Lock Last")
    print("  ===============================================================")
    print()
    experiment_1_2_delay_and_sharpness()
    experiment_3_hubs_lock_last()
    experiment_4_correlation_sweep()
    print()
    print("=" * 70)
    print("WHAT THIS ESTABLISHES")
    print("=" * 70)
    print("Correlating each node's structural frequency nu_f with its degree --")
    print("tying the nodal dynamics to the nodal structure -- reshapes the")
    print("Kuramoto transition of the canonical phase channel: the onset is")
    print("DELAYED and SHARPER (M1/M2), and the high-degree hubs synchronize LAST")
    print("(M3, corr(degree, lock) < 0); the delay grows with the correlation")
    print("(M-extra). HONEST SCOPE: this is NOT the full textbook explosive")
    print("synchronization (strong first-order + wide hysteresis), which needs")
    print("degree-WEIGHTED coupling; the canonical phase channel is degree-")
    print("NORMALIZED (circular mean), so the hysteresis is weak (an honest")
    print("negative) -- but the delay, sharpening, and hub-frustration robustly")
    print("emerge. The Kuramoto / explosive-sync phenomenology is empirically")
    print("established (Kuramoto 1975; Gomez-Gardenes 2011); this re-expresses it")
    print("in the canonical phase channel, not new mathematics, closes no open")
    print("problem.")


if __name__ == "__main__":
    main()
