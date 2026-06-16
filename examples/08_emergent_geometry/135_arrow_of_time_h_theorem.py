#!/usr/bin/env python3
"""
Example 135 — The Emergent Arrow of Time: the Structural H-Theorem of the EPI
Diffusion Channel
==============================================================================

The EPI channel of the nodal equation is the discrete diffusion equation
(AGENTS.md "Transport Content of the Nodal Equation"; examples 99, 134):

    dEPI/dt = nu_f * dNFR = -nu_f * L_rw * EPI,   L_rw = I - D^{-1} W.

Diffusion is IRREVERSIBLE. It carries a monotone Lyapunov functional -- the
H-theorem (Boltzmann 1872) -- and a structural ARROW OF TIME: the forward flow
smooths and forgets, while the time-reversed flow is ill-posed. This is the
second law of thermodynamics emerging directly from the nodal dynamics, anchored
to one of the most empirically-established phenomena in all of physics (Clausius
1865, Boltzmann 1872).

Doctrine compliance
-------------------
Everything emerges from the canonical structural-diffusion operator: the flow is
the EPI channel of the nodal equation, the H-functional is the total squared
canonical structural Fick current (structural_current), and the stationary
measure is the canonical stationary_distribution. Nothing is imported -- this is
the CLASSICAL Shannon/Dirichlet H of the EPI random-walk channel. It is DISTINCT
from (and complementary to) two existing canonical pieces: the tetrad Lyapunov
energy of conservation.py (which is the tetrad-field energy, not the EPI Dirichlet
energy) and the Von Neumann entropy of dissipative_conservation.py (which is the
Lindblad density-matrix entropy, a separate quantum-dissipative abstraction).

Three measured results
----------------------
M1 THE STRUCTURAL H-THEOREM. The Dirichlet energy F = (1/2) sum_ij A_ij
   (EPI_i - EPI_j)^2 -- which equals the total squared canonical structural Fick
   current (structural_current; verified |diff|=0) -- decreases MONOTONICALLY to
   0 under the EPI diffusion flow (dF/dt <= 0 exact, verified on a fine grid).
   F is a Lyapunov functional; equilibrium is reached when the currents vanish.

M2 ENTROPY INCREASES (THE SECOND LAW). The random-walk distribution
   p_t = e^{-t L_rw} delta (a row of the heat kernel, a probability distribution
   -- ex 134) has relative entropy D(p_t || pi) to the stationary measure pi that
   DECREASES monotonically to 0 (the rigorous H-functional for any graph); on a
   regular graph the Shannon entropy S(p_t) INCREASES monotonically to log n. The
   structural field forgets its initial condition.

M3 THE ARROW OF TIME IS STRUCTURAL. Forward diffusion is a smoothing contraction
   (F bounded, -> 0); the time-reversed anti-diffusion dEPI/dt = +nu_f L_rw EPI is
   ill-posed (F DIVERGES as ~ e^{2 nu_f lambda_max t}). Only the forward direction
   is well-posed -- the arrow of time emerges from the NON-NEGATIVE spectrum of the
   canonical operator (every lambda_k >= 0).

Honest scope
------------
The H-theorem / entropy increase for diffusion is an EXACT, provable fact (the
Dirichlet energy and the relative entropy to the stationary measure are Lyapunov
functionals of the heat semigroup / reversible Markov chain), and the arrow of
time / second law is one of the most empirically-established phenomena in physics
(Clausius, Boltzmann). This re-expresses the irreversibility of the diffusion
equation -- which we established IS the nodal EPI channel (ex 99, 134) -- in
thermodynamic language. It is distinct from the tetrad Lyapunov energy
(conservation.py) and the Lindblad / Von Neumann entropy
(dissipative_conservation.py). It is not new mathematics and closes no open
problem.

References
----------
- src/tnfr/physics/structural_diffusion.py (structural_diffusion_operator,
  structural_current, stationary_distribution)
- src/tnfr/physics/conservation.py (the tetrad Lyapunov energy -- distinct)
- src/tnfr/physics/dissipative_conservation.py (Lindblad/Von Neumann -- distinct)
- AGENTS.md "Transport Content of the Nodal Equation (Structural Diffusion)"
- examples/08_emergent_geometry/99_structural_diffusion.py (the diffusion layer)
- examples/08_emergent_geometry/134_spectral_dimension_heat_kernel.py (heat kernel)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import numpy as np
import networkx as nx

try:
    import scipy.linalg as sla
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False

from tnfr.alias import set_attr
from tnfr.constants.aliases import ALIAS_EPI
from tnfr.physics.structural_diffusion import (
    structural_diffusion_operator,
    structural_current,
    stationary_distribution,
)


def _expm(M):
    if _HAVE_SCIPY:
        return sla.expm(M)
    w, V = np.linalg.eig(M)
    return (V @ np.diag(np.exp(w)) @ np.linalg.inv(V)).real


def adjacency_mask(G, nodes):
    idx = {n: i for i, n in enumerate(nodes)}
    A = np.zeros((len(nodes), len(nodes)))
    for u, v in G.edges():
        A[idx[u], idx[v]] = 1.0
        A[idx[v], idx[u]] = 1.0
    return A


def dirichlet_energy(A, epi):
    """F = (1/2) sum_ij A_ij (epi_i - epi_j)^2 = sum of squared Fick currents."""
    d = epi[:, None] - epi[None, :]
    return 0.5 * float(np.sum(A * d * d))


def experiment_1_h_theorem():
    """M1: the structural H-theorem -- Dirichlet energy decreases monotonically."""
    print("=" * 74)
    print("M1: THE STRUCTURAL H-THEOREM (Dirichlet energy -> 0 monotonically)")
    print("=" * 74)
    G = nx.watts_strogatz_graph(40, 6, 0.2, seed=1)
    nodes, lap = structural_diffusion_operator(G)
    A = adjacency_mask(G, nodes)
    L = np.asarray(lap)
    epi0 = np.random.default_rng(0).standard_normal(len(nodes))
    # anchor: F == total squared canonical structural Fick current
    for i, n in enumerate(nodes):
        set_attr(G.nodes[n], ALIAS_EPI, float(epi0[i]))
    _, Jc = structural_current(G)
    F_canon = 0.5 * float(np.sum(Jc * Jc))
    F_direct = dirichlet_energy(A, epi0)
    print(f"  anchor: 1/2 sum J^2 (canonical) = {F_canon:.6f}")
    print(f"          Dirichlet F (direct)    = {F_direct:.6f}  "
          f"|diff|={abs(F_canon - F_direct):.1e}")
    print("  -> the H-functional IS the total squared structural Fick current.")
    print()
    print(f"  {'t':>6} {'F(t)':>12} {'monotone?':>10}")
    prevF = None
    for t in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]:
        epit = _expm(-t * L) @ epi0
        F = dirichlet_energy(A, epit)
        tag = "" if prevF is None else ("yes" if F <= prevF + 1e-9 else "NO")
        print(f"  {t:>6.1f} {F:>12.6f} {tag:>10}")
        prevF = F
    tg = np.linspace(0, 20, 400)
    Fg = [dirichlet_energy(A, _expm(-tt * L) @ epi0) for tt in tg]
    dF = np.diff(Fg)
    print()
    print(f"  fine grid (400 steps): max dF = {dF.max():.2e} "
          f"(<=0 => monotone), all-monotone = {bool(np.all(dF <= 1e-9))}")
    print("  -> F decays monotonically to 0 (equilibrium = no Fick currents):")
    print("     the structural H-theorem. F is a Lyapunov functional.")


def experiment_2_entropy():
    """M2: entropy increases (relative entropy to pi decreases monotonically)."""
    print()
    print("=" * 74)
    print("M2: ENTROPY INCREASES (the second law)")
    print("=" * 74)
    print("The random-walk distribution p_t = e^{-t L_rw} delta has relative")
    print("entropy D(p_t || pi) to the stationary measure that decreases monotone.")
    print()
    G = nx.watts_strogatz_graph(40, 6, 0.2, seed=1)
    nodes, lap = structural_diffusion_operator(G)
    L = np.asarray(lap)
    _, pi = stationary_distribution(G)
    pi = np.asarray(pi)
    print(f"  {'t':>6} {'D(p_t||pi)':>12} {'monotone?':>10}")
    prevD = None
    for t in [0.05, 0.2, 0.5, 1.0, 2.0, 5.0, 15.0]:
        p = np.clip(_expm(-t * L)[0], 1e-300, None)
        D = float(np.sum(p * np.log(p / pi)))
        tag = "" if prevD is None else ("yes" if D <= prevD + 1e-9 else "NO")
        print(f"  {t:>6.2f} {D:>12.6f} {tag:>10}")
        prevD = D
    print()
    Greg = nx.cycle_graph(60)
    nr, lr = structural_diffusion_operator(Greg)
    Lr = np.asarray(lr)
    print(f"  regular ring n=60: Shannon entropy S(p_t) increases to "
          f"log n = {np.log(60):.3f}")
    print(f"    {'t':>7} {'S(p_t)':>9} {'monotone?':>10}")
    prevS = None
    smono = True
    for t in [0.1, 1.0, 5.0, 20.0, 100.0, 500.0]:
        p = np.clip(_expm(-t * Lr)[0], 1e-300, None)
        S = float(-np.sum(p * np.log(p)))
        tag = "" if prevS is None else ("up" if S >= prevS - 1e-9 else "DOWN")
        if prevS is not None and S < prevS - 1e-9:
            smono = False
        print(f"    {t:>7.1f} {S:>9.4f} {tag:>10}")
        prevS = S
    print()
    print("  -> relative entropy decreases (rigorous H-functional, any graph);")
    print(f"     ring Shannon entropy increases monotonically: {smono}. The")
    print("     structural field forgets its initial condition (mixing).")

def experiment_3_arrow_of_time():
    """M3: the arrow of time -- forward smooths, backward blows up."""
    print()
    print("=" * 74)
    print("M3: THE ARROW OF TIME IS STRUCTURAL")
    print("=" * 74)
    print("Forward diffusion smooths (F -> 0); time-reversed anti-diffusion")
    print("dEPI/dt = +nu_f L_rw EPI is ill-posed (F diverges exponentially).")
    print()
    Greg = nx.cycle_graph(60)
    nr, lr = structural_diffusion_operator(Greg)
    Lr = np.asarray(lr)
    Ar = adjacency_mask(Greg, nr)
    lam_max = float(np.linalg.eigvalsh(0.5 * (Lr + Lr.T))[-1])
    epi0 = np.random.default_rng(3).standard_normal(len(nr))
    print(f"  ring lambda_max = {lam_max:.4f}; anti-diffusion F ~ "
          f"e^(2 lambda_max t)")
    print()
    print(f"  {'t':>6} {'F forward':>14} {'F backward':>18}")
    for t in [0.0, 0.5, 1.0, 2.0, 4.0]:
        Ff = dirichlet_energy(Ar, _expm(-t * Lr) @ epi0)
        Fb = dirichlet_energy(Ar, _expm(+t * Lr) @ epi0)
        print(f"  {t:>6.1f} {Ff:>14.6f} {Fb:>18.2f}")
    print()
    print("  -> forward F is bounded and -> 0 (smoothing contraction); backward F")
    print("     diverges exponentially (anti-diffusion is ill-posed). Only the")
    print("     forward direction is well-posed: the arrow of time emerges from")
    print("     the non-negative spectrum of the canonical operator (lambda_k >= 0).")


def main():
    print()
    print("  ===============================================================")
    print("  The Emergent Arrow of Time")
    print("  The Structural H-Theorem of the EPI Diffusion Channel")
    print("  ===============================================================")
    print()
    experiment_1_h_theorem()
    experiment_2_entropy()
    experiment_3_arrow_of_time()
    print()
    print("=" * 74)
    print("WHAT THIS ESTABLISHES")
    print("=" * 74)
    print("The EPI channel of the nodal equation is the diffusion equation, which")
    print("is IRREVERSIBLE. Its Dirichlet energy (the total squared structural Fick")
    print("current) is a Lyapunov functional decreasing monotonically to 0 (M1,")
    print("the structural H-theorem); the random-walk relative entropy decreases")
    print("monotonically and the Shannon entropy increases (M2, the second law);")
    print("and the time-reversed anti-diffusion is ill-posed while the forward flow")
    print("smooths (M3, the arrow of time from the non-negative spectrum). HONEST")
    print("SCOPE: the H-theorem / entropy increase for diffusion is exact and")
    print("provable (Lyapunov functionals of the heat semigroup), and the second")
    print("law / arrow of time is empirically ironclad (Clausius, Boltzmann). It")
    print("re-expresses the irreversibility of the EPI diffusion channel (ex 99,")
    print("134) in thermodynamic language; distinct from the tetrad Lyapunov energy")
    print("(conservation.py) and the Lindblad / Von Neumann entropy")
    print("(dissipative_conservation.py). Not new mathematics, closes no open problem.")


if __name__ == "__main__":
    main()
