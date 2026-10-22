#!/usr/bin/env python3
"""Spectral prediction for held pure-EPI diffusion on reciprocal graphs.

For fixed connected reciprocal nonnegative conductance and common positive
capacity nu, the selected law x_dot=-nu*L_rw*x conserves the degree-weighted
mean. The full initial field and generator determine its future; topology alone
does not determine amplitudes, destination or the supplied capacity clock.
The slowest excited nonuniform eigenspace determines late relative decay.
A Fiedler direction requires a nonzero initial projection; a repeated gap allows
multiple slow directions. Proximity of eigenvalues affects finite-time separation.
This is a weighted gradient flow of the Dirichlet functional, not generally
Euclidean steepest descent. Phase, capacity and support are held; small sample
correlations between diagnostics do not prove channel independence or closure.
See theory/TNFR_DIFFUSION_STABILITY_THEOREM.md for exact scope.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import networkx as nx
import numpy as np

from tnfr.physics.structural_diffusion import (
    degree_weighted_total,
    structural_diffusion_operator,
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
    w, U = np.linalg.eig(L)  # L_rw is non-symmetric
    order = np.argsort(w.real)
    lam = w.real[order]
    U = U.real[:, order]
    lam2 = lam[1]
    gap_ratio = lam[1] / lam[2] if lam[2] > 1e-12 else 0.0  # isolation of gap
    u2 = U[:, 1] / np.linalg.norm(U[:, 1])
    dest = degree_weighted_total(G) / degs.sum()  # degree-weighted mean
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
        if t > 2.0 / (nu_f * lam2):  # asymptotic regime
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
    print(
        f"  {'graph':>14} {'nu_f*lam2 (pred)':>16} {'measured':>10} " f"{'rel err':>9}"
    )
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
    print(
        "the held generator. Initial state and nu_f are also required; pressure is the"
    )
    print("instantaneous direction; the spectrum {lambda_k} is the route (modes")
    print("dissolve fast->slow); the destination is the degree-weighted mean")
    print("(the lambda_1=0 conserved mode); the relaxation time is")
    print("1/(nu_f*lambda_2) when that mode is excited; late heading is in the")
    print("slowest excited eigenspace -- conditional predictions checked against the")
    print("actual integrated flow. Characterization of the nodal dynamics,")
    print("not new physics.")
    print()


if __name__ == "__main__":
    main()
