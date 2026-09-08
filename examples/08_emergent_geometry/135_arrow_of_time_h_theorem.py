#!/usr/bin/env python3
"""
Example 135 — Dirichlet Dissipation in the EPI Diffusion Channel
================================================================

The EPI channel of the nodal equation is the discrete diffusion equation
(AGENTS.md "Transport content"; examples 99 and 134):

    dEPI/dt = nu_f * DeltaNFR = -nu_f * L_rw * EPI.

On a fixed connected undirected graph with nonnegative conductance and positive
fixed capacity, this restricted flow has a monotone Dirichlet functional. The
associated reversible Markov semigroup contracts relative entropy toward its
stationary distribution. These results establish dissipation and mixing within
that model; they do not identify EPI with heat or probability, derive physical
entropy, or prove a universal time direction for the full TNFR engine.

Three measured results
----------------------
M1 DIRICHLET DISSIPATION. The graph Dirichlet energy decreases monotonically to
   zero. It equals the squared structural-current readout under the unit-weight
   convention used by this experiment.

M2 MARKOV MIXING. Relative entropy of a heat-kernel row to the stationary
   distribution decreases. On a regular graph the Shannon entropy of that
   auxiliary probability distribution increases toward log(n).

M3 INVERSE SENSITIVITY. On every finite graph both forward diffusion and its
   exact inverse exist for finite time. The inverse amplifies high-frequency
   perturbations exponentially. The inverse heat problem becomes ill-conditioned
   as time or the resolved spectral range grows, and is ill-posed in the usual
   continuum or unbounded-spectrum limit.

Scope
-----
The experiment concerns fixed pure-EPI diffusion. Its Dirichlet functional is
distinct from the five-field structural-energy candidate in conservation.py and
from the von Neumann entropy diagnostic in dissipative_conservation.py. The
nu_f*lambda_2 relaxation scale applies under the corresponding fixed-diffusion
hypotheses. No thermodynamic identification or open-problem result follows.

References
----------
- src/tnfr/physics/structural_diffusion.py
- src/tnfr/physics/conservation.py
- src/tnfr/physics/dissipative_conservation.py
- examples/08_emergent_geometry/99_structural_diffusion.py
- examples/08_emergent_geometry/134_spectral_dimension_heat_kernel.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import networkx as nx
import numpy as np

try:
    import scipy.linalg as sla

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False

from tnfr.alias import set_attr
from tnfr.constants.aliases import ALIAS_EPI
from tnfr.physics.structural_diffusion import (
    stationary_distribution,
    structural_current,
    structural_diffusion_operator,
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
    """M1: verify monotone Dirichlet dissipation."""
    print("=" * 74)
    print("M1: DIRICHLET DISSIPATION (energy -> 0 monotonically)")
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
    print(
        f"          Dirichlet F (direct)    = {F_direct:.6f}  "
        f"|diff|={abs(F_canon - F_direct):.1e}"
    )
    print("  -> the functionals agree under this unit-weight convention.")
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
    print(
        f"  fine grid (400 steps): max dF = {dF.max():.2e} "
        f"(<=0 => monotone), all-monotone = {bool(np.all(dF <= 1e-9))}"
    )
    print("  -> F decays monotonically to 0 (equilibrium = no Fick currents):")
    print("     F is a Lyapunov functional for this fixed diffusion flow.")


def experiment_2_entropy():
    """M2: verify relative-entropy contraction for the Markov semigroup."""
    print()
    print("=" * 74)
    print("M2: AUXILIARY MARKOV-DISTRIBUTION MIXING")
    print("=" * 74)
    print("The random-walk distribution p_t = e^{-t L_rw} delta has relative")
    print("entropy D(p_t || pi) to stationarity that decreases monotonically.")
    print()
    G = nx.watts_strogatz_graph(40, 6, 0.2, seed=1)
    nodes, lap = structural_diffusion_operator(G)
    L = np.asarray(lap)
    _, pi = stationary_distribution(G)
    pi = np.asarray(pi)
    print(f"  {'t':>6} {'D(p_t||pi)':>12} {'monotone?':>10}")
    prevD = None
    for t in [0.05, 0.2, 0.5, 1.0, 2.0, 5.0, 15.0]:
        p = np.maximum(_expm(-t * L)[0], 0.0)
        p /= float(np.sum(p))
        support = p > 0.0
        D = float(np.sum(p[support] * np.log(p[support] / pi[support])))
        tag = "" if prevD is None else ("yes" if D <= prevD + 1e-9 else "NO")
        print(f"  {t:>6.2f} {D:>12.6f} {tag:>10}")
        prevD = D
    print()
    Greg = nx.cycle_graph(60)
    nr, lr = structural_diffusion_operator(Greg)
    Lr = np.asarray(lr)
    print(
        f"  regular ring n=60: Shannon entropy S(p_t) increases to "
        f"log n = {np.log(60):.3f}"
    )
    print(f"    {'t':>7} {'S(p_t)':>9} {'monotone?':>10}")
    prevS = None
    smono = True
    for t in [0.1, 1.0, 5.0, 20.0, 100.0, 500.0]:
        p = np.maximum(_expm(-t * Lr)[0], 0.0)
        p /= float(np.sum(p))
        support = p > 0.0
        S = float(-np.sum(p[support] * np.log(p[support])))
        tag = "" if prevS is None else ("up" if S >= prevS - 1e-9 else "DOWN")
        if prevS is not None and S < prevS - 1e-9:
            smono = False
        print(f"    {t:>7.1f} {S:>9.4f} {tag:>10}")
        prevS = S
    print()
    print("  -> relative entropy decreases (rigorous H-functional, any graph);")
    print(f"     ring Shannon entropy increases monotonically: {smono}.")
    print("     The heat-kernel row loses its starting-node information (mixing).")


def experiment_3_inverse_sensitivity():
    """M3: contrast forward smoothing with finite-graph inverse sensitivity."""
    print()
    print("=" * 74)
    print("M3: FINITE-GRAPH INVERSE SENSITIVITY")
    print("=" * 74)
    print("Forward diffusion smooths (F -> 0); reversing the finite flow means")
    print("the exact inverse, which amplifies high-frequency error exponentially.")
    print()
    Greg = nx.cycle_graph(60)
    nr, lr = structural_diffusion_operator(Greg)
    Lr = np.asarray(lr)
    Ar = adjacency_mask(Greg, nr)
    lam_max = float(np.linalg.eigvalsh(0.5 * (Lr + Lr.T))[-1])
    epi0 = np.random.default_rng(3).standard_normal(len(nr))
    print(
        f"  ring lambda_max = {lam_max:.4f}; anti-diffusion F ~ " f"e^(2 lambda_max t)"
    )
    print()
    print(f"  {'t':>6} {'F forward':>14} {'F backward':>18}")
    for t in [0.0, 0.5, 1.0, 2.0, 4.0]:
        Ff = dirichlet_energy(Ar, _expm(-t * Lr) @ epi0)
        Fb = dirichlet_energy(Ar, _expm(+t * Lr) @ epi0)
        print(f"  {t:>6.1f} {Ff:>14.6f} {Fb:>18.2f}")
    print()
    print("  -> forward F contracts while inverse-flow F grows exponentially.")
    print("     Both maps exist on this finite graph for finite t; inversion becomes")
    print("     exponentially ill-conditioned and is ill-posed only in the usual")
    print("     continuum or unbounded-spectrum inverse-heat limit.")


def main():
    print()
    print("  ===============================================================")
    print("  Dirichlet Dissipation and Inverse Sensitivity")
    print("  Fixed-Graph EPI Diffusion")
    print("  ===============================================================")
    print()
    experiment_1_h_theorem()
    experiment_2_entropy()
    experiment_3_inverse_sensitivity()
    print()
    print("=" * 74)
    print("WHAT THIS ESTABLISHES")
    print("=" * 74)
    print("For fixed pure-EPI diffusion, the Dirichlet energy is a Lyapunov")
    print("functional (M1). A heat-kernel probability row contracts relative")
    print("entropy toward stationarity; regular-graph Shannon entropy rises (M2).")
    print("The finite-graph inverse exists but amplifies spectral error")
    print("exponentially, exposing the inverse heat problem's conditioning (M3).")
    print("These are scoped diffusion and Markov-semigroup results. They do not")
    print("identify EPI with thermodynamic state, prove a physical second law, or")
    print("establish a universal arrow of time for multichannel TNFR dynamics.")

if __name__ == "__main__":
    main()
