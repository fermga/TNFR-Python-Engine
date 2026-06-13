#!/usr/bin/env python3
"""
Example 112 — Where Does the Coherence Flow Go? Structure Predicts the Flow
==========================================================================

The nodal equation gives the INSTANTANEOUS rate of coherence change,

    dEPI/dt = nu_f * dNFR,

but does the STRUCTURE already tell us WHERE the flow is heading, before we
integrate anything? For the EPI channel the answer is yes, exactly: the
canonical dNFR is the random-walk graph Laplacian acting on EPI,

    dNFR = neighbour-mean - self = -L_rw * EPI   (structural_diffusion.py),

so the flow is a linear gradient descent whose entire future is fixed by the
eigendecomposition of the PURELY STRUCTURAL operator L_rw:

    EPI(t) = sum_k c_k * exp(-nu_f * lambda_k * t) * u_k,   L_rw u_k = lambda_k u_k.

The structure therefore contains a complete ROADMAP of the coherence flow,
readable without running the dynamics:

  - nu_f (structural frequency)  = the CLOCK    -> sets the timescale.
  - dNFR = -L_rw*EPI (pressure)  = the DIRECTION -> the steepest-descent heading now.
  - the spectrum {lambda_k}       = the ROUTE    -> modes dissolve fast->slow by lambda.
  - lambda_1 = 0 (uniform mode)   = the DESTINATION -> degree-weighted mean (conserved).
  - lambda_2 (spectral gap)       = the CLOCK HAND  -> relaxation time 1/(nu_f*lambda_2).
  - u_2 (Fiedler eigenvector)     = the FINAL HEADING -> the last surviving direction.

This example PREDICTS the destination, the relaxation time, and the final
heading from structure alone, then VALIDATES each against the actual
integrated nodal flow.

Honest scope
------------
- Exact for the EPI channel (dNFR = -L_rw*EPI). The full canonical dNFR is
  multi-channel (phase / nu_f / topology add their own relaxation); the EPI
  channel is the clean exact statement (the channels are approximately
  independent, |Pearson r| ~ 0.12).
- The single-vector Fiedler heading is sharp only when the spectral gap is
  ISOLATED (lambda_2 well below lambda_3). Under near-degeneracy lambda_2 ~ lambda_3
  the slowest "direction" is a 2D subspace and the single-vector alignment
  softens (honest: see the lambda_2/lambda_3 ratio per graph below). The
  decay-RATE prediction nu_f*lambda_2 stays robust regardless.
- This is a CHARACTERIZATION of the nodal flow (linear gradient descent on a
  structural Laplacian), not new physics. It reuses the canonical
  structural_diffusion operators.

References
----------
- src/tnfr/physics/structural_diffusion.py (L_rw, relaxation spectrum, Fiedler)
- examples/08_emergent_geometry/99_structural_diffusion.py (the transport layer this builds on)
- src/tnfr/physics/variational.py (nodal equation as gradient flow)
- AGENTS.md section "Transport Content of the Nodal Equation"
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import numpy as np
import networkx as nx

from tnfr.physics.structural_diffusion import (
    structural_diffusion_operator,
    degree_weighted_total,
)


def build_graph(seed, n):
    G = nx.watts_strogatz_graph(n, 4, 0.3, seed=seed)
    rng = np.random.default_rng(seed)
    for v in G.nodes():
        G.nodes[v]["EPI"] = float(rng.uniform(-0.5, 0.5))
        G.nodes[v]["nu_f"] = 1.0
    return G


def structure_only_predictions(G):
    """Compute destination, relaxation time, Fiedler heading from L_rw alone."""
    nodes, L = structural_diffusion_operator(G)
    degs = np.array([sum(1.0 for _ in G.neighbors(v)) for v in nodes])
    nu_f = float(np.mean([G.nodes[v]["nu_f"] for v in nodes]))
    w, U = np.linalg.eig(L)                  # L_rw is non-symmetric
    order = np.argsort(w.real)
    lam = w.real[order]
    U = U.real[:, order]
    lam2 = lam[1]
    gap_ratio = lam[1] / lam[2] if lam[2] > 1e-12 else 0.0   # isolation of gap
    u2 = U[:, 1] / np.linalg.norm(U[:, 1])
    dest = degree_weighted_total(G) / degs.sum()             # degree-weighted mean
    return nodes, L, nu_f, lam2, u2, dest, gap_ratio


def integrate_flow(G, nodes, L, nu_f, dest, u2, lam2):
    """Integrate dEPI/dt = -nu_f L EPI; measure heading + decay rate."""
    epi = np.array([G.nodes[v]["EPI"] for v in nodes])
    dt = 0.01
    t = 0.0
    t_end = 8.0 / (nu_f * lam2)
    aligns, log_norm, ts = [], [], []
    while t < t_end:
        epi = epi - dt * nu_f * (L @ epi)
        t += dt
        if t > 2.0 / (nu_f * lam2):          # asymptotic regime
            resid = epi - dest
            nr = np.linalg.norm(resid)
            if nr > 1e-12:
                aligns.append(abs(float(resid @ u2) / nr))
                log_norm.append(np.log(nr))
                ts.append(t)
    dest_err = abs(float(np.mean(epi)) - dest)
    # measured late-time decay rate from residual-norm slope
    rate_meas = -float(np.polyfit(ts, log_norm, 1)[0])
    return float(np.mean(aligns[-50:])), dest_err, rate_meas


def experiment_1_predict_and_validate():
    print("=" * 72)
    print("EXP 1: predict destination + heading from STRUCTURE, validate by flow")
    print("=" * 72)
    print()
    print("From L_rw alone: destination = degree-weighted mean; final heading")
    print("= Fiedler eigenvector u_2. Then integrate the nodal flow and check.")
    print()
    print(f"  {'graph':>14} {'lam2/lam3':>10} {'align u2':>9} {'dest err':>9}")
    for seed in range(6):
        n = 30 + seed
        G = build_graph(seed, n)
        nodes, L, nu_f, lam2, u2, dest, gap = structure_only_predictions(G)
        align, derr, _ = integrate_flow(G, nodes, L, nu_f, dest, u2, lam2)
        print(f"  ws(n={n},s={seed})  {gap:10.3f} {align:9.4f} {derr:9.1e}")
    print()
    print("align u2 -> 1: the actual flow's asymptotic heading IS the")
    print("structure-only Fiedler vector. dest err ~0: destination is the")
    print("degree-weighted mean, predicted exactly. (lam2/lam3 near 1 = a")
    print("near-degenerate spectral gap, where the slowest 'direction' is a 2D")
    print("subspace and the single-vector alignment softens -- honest caveat.)")
    print()


def experiment_2_predict_decay_rate():
    print("=" * 72)
    print("EXP 2: predict the relaxation rate nu_f*lambda_2 from STRUCTURE")
    print("=" * 72)
    print()
    print("The spectral gap lambda_2 (Fiedler value) sets the slowest decay.")
    print("Predicted late-time rate = nu_f*lambda_2; measured from the flow.")
    print()
    print(f"  {'graph':>14} {'nu_f*lam2 (pred)':>16} {'measured':>10} "
          f"{'rel err':>9}")
    for seed in range(6):
        n = 30 + seed
        G = build_graph(seed, n)
        nodes, L, nu_f, lam2, u2, dest, gap = structure_only_predictions(G)
        _, _, rate = integrate_flow(G, nodes, L, nu_f, dest, u2, lam2)
        pred = nu_f * lam2
        rel = abs(rate - pred) / pred
        print(f"  ws(n={n},s={seed})  {pred:16.4f} {rate:10.4f} {rel:9.1e}")
    print()
    print("measured -> nu_f*lambda_2: the relaxation TIME of the coherence")
    print("flow (1/(nu_f*lambda_2)) is read off the structure's spectral gap,")
    print("robustly (this prediction holds even under near-degeneracy).")
    print()


def main():
    print()
    print("  TNFR Example 112: Where Does the Coherence Flow Go?")
    print("  Structure predicts destination, timing, and heading of the flow")
    print("  ===============================================================")
    print()
    experiment_1_predict_and_validate()
    experiment_2_predict_decay_rate()
    print("=" * 72)
    print("WHAT THIS ESTABLISHES")
    print("=" * 72)
    print()
    print("The coherence flow of the nodal equation does NOT need to be run to")
    print("know where it goes: for the EPI channel it is linear gradient")
    print("descent on the structural Laplacian L_rw, so the structure contains")
    print("the flow's whole roadmap. nu_f is the clock; dNFR=-L_rw*EPI is the")
    print("instantaneous direction; the spectrum {lambda_k} is the route (modes")
    print("dissolve fast->slow); the destination is the degree-weighted mean")
    print("(the lambda_1=0 conserved mode); the relaxation time is")
    print("1/(nu_f*lambda_2); and the final heading is the Fiedler eigenvector")
    print("u_2 -- all read from structure alone and validated against the")
    print("actual integrated flow. Characterization of the nodal dynamics,")
    print("not new physics.")
    print()


if __name__ == "__main__":
    main()
