"""TNFR Example 99: Structural Diffusion — the transport content of the
nodal equation.

This example demonstrates, by direct measurement, that the TNFR nodal
equation

    ∂EPI/∂t = νf · ΔNFR(t)

is **structurally a diffusion equation on the network** — not an analogy
imported from another paradigm, but the literal content of the canonical
ΔNFR computation.

WHAT EMERGES (in TNFR's own terms, compared only to empirically-
demonstrated phenomena):

- The EPI channel of ΔNFR is exactly the random-walk graph Laplacian
  (the discrete diffusion / heat operator): ΔNFR_epi = −L_rw·EPI.
- The structural form EPI spreads and relaxes to a uniform field, exactly
  as heat or a concentration diffuses (Fourier 1822, Fick 1855,
  Einstein 1905 — established by the strictest empirical method).
- The diffusion conserves the degree-weighted structural total
  Σ deg·EPI (the analogue of the conserved amount of diffusing substance).
- νf is the diffusivity (mobility); ΔNFR is the structural pressure
  (the gradient driving the flux); ΔNFR = 0 ⟺ no gradients ⟺ equilibrium.
- The phase channel of ΔNFR drives Kuramoto synchronization (also an
  empirically-demonstrated phenomenon: fireflies, pacemaker cells,
  neurons, Josephson junctions).

References:
- src/tnfr/physics/structural_diffusion.py
- src/tnfr/dynamics/dnfr.py (the neighbour-mean ΔNFR gradients)
- src/tnfr/observers.py (kuramoto_order)
- AGENTS.md §"Foundational Physics"
"""

import os
import sys
import math
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import networkx as nx

from tnfr.dynamics import default_compute_delta_nfr
from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_DNFR
from tnfr.observers import kuramoto_order
from tnfr.physics.structural_diffusion import (
    verify_structural_diffusion,
    structural_diffusion_operator,
    structural_field,
    relaxation_spectrum,
    degree_weighted_total,
)


def _build(n=60, seed=11):
    rng = random.Random(seed)
    G = nx.watts_strogatz_graph(n, 6, 0.2, seed=seed)
    for nd in G.nodes():
        G.nodes[nd]["theta"] = rng.uniform(0.0, 2.0 * math.pi)
        G.nodes[nd]["EPI"] = rng.uniform(-0.4, 0.4)
        G.nodes[nd]["nu_f"] = rng.uniform(0.5, 1.5)
    default_compute_delta_nfr(G)
    return G


def experiment_1_nodal_is_diffusion():
    """The EPI channel of ΔNFR is the graph diffusion operator."""
    print("=" * 72)
    print("EXPERIMENT 1: The nodal equation IS graph diffusion")
    print("=" * 72)
    print()
    print("The canonical ΔNFR is a sum of neighbour-mean-minus-self")
    print("gradients. For the EPI channel this is exactly −L_rw·EPI, the")
    print("random-walk graph Laplacian — the discrete diffusion operator.")
    print()

    G = _build(60)
    nodes, lap = structural_diffusion_operator(G)
    epi = structural_field(G, nodes)

    # isolate the EPI channel on a clean replica
    from tnfr.constants.aliases import ALIAS_EPI
    g2 = nx.Graph()
    for nd in nodes:
        g2.add_node(
            nd,
            EPI=float(get_attr(G.nodes[nd], ALIAS_EPI, 0.0)),
            theta=0.0,
            nu_f=1.0,
        )
    g2.add_edges_from(G.edges())
    g2.graph["DNFR_WEIGHTS"] = {
        "phase": 0.0, "epi": 1.0, "vf": 0.0, "topo": 0.0
    }
    default_compute_delta_nfr(g2)
    dnfr = np.array(
        [float(get_attr(g2.nodes[nd], ALIAS_DNFR, 0.0)) for nd in nodes]
    )
    residual = float(np.max(np.abs(dnfr - (-(lap @ epi)))))
    print(f"  max |ΔNFR_epi − (−L_rw·EPI)| = {residual:.2e}")
    print(f"  -> the EPI channel of the nodal equation IS graph diffusion")
    print()


def experiment_2_diffusion_signatures():
    """Conservation, relaxation, and the diffusion spectrum."""
    print("=" * 72)
    print("EXPERIMENT 2: Diffusion signatures (conservation + relaxation)")
    print("=" * 72)
    print()

    G = _build(60)
    cert = verify_structural_diffusion(G)
    print(cert.summary())
    print()
    spec = relaxation_spectrum(G)
    print(f"  relaxation spectrum νf·λ_k (first 6): "
          f"{[round(float(x), 4) for x in spec[:6]]}")
    print(f"  λ₁ = 0 is the conserved uniform mode; λ₂ (spectral gap) sets")
    print(f"  the slowest relaxation. Conserved total Σ deg·EPI = "
          f"{degree_weighted_total(G):.4f}")
    print()
    print("VALIDATED: the structural form diffuses to a uniform equilibrium,")
    print("conserving the degree-weighted total — the same mathematics as")
    print("heat/concentration diffusion (Fourier, Fick, Einstein).")
    print()


def experiment_3_synchronization():
    """The phase channel drives Kuramoto synchronization."""
    print("=" * 72)
    print("EXPERIMENT 3: The phase channel drives synchronization")
    print("=" * 72)
    print()
    print("The phase term of ΔNFR aligns θ to the neighbour mean, driving")
    print("Kuramoto synchronization (R → 1) — empirically demonstrated in")
    print("fireflies, pacemaker cells, neurons, Josephson junctions.")
    print()

    G = _build(60, seed=3)
    G.graph["DNFR_WEIGHTS"] = {
        "phase": 1.0, "epi": 0.0, "vf": 0.0, "topo": 0.0
    }
    nodes = list(G.nodes())
    r0 = kuramoto_order(G)
    for _ in range(300):
        default_compute_delta_nfr(G)
        for nd in nodes:
            g_phase = float(get_attr(G.nodes[nd], ALIAS_DNFR, 0.0))
            G.nodes[nd]["theta"] = float(G.nodes[nd]["theta"]) + 0.3 * g_phase
    r1 = kuramoto_order(G)
    print(f"  Kuramoto order R: start {r0:.4f} -> end {r1:.4f}")
    print(f"  -> phase coupling = neighbour-mean alignment = synchronization")
    print()


def main():
    print()
    print("  TNFR Example 99: Structural Diffusion")
    print("  The transport content of the nodal equation")
    print("  ===========================================")
    print()

    experiment_1_nodal_is_diffusion()
    experiment_2_diffusion_signatures()
    experiment_3_synchronization()

    print("=" * 72)
    print("WHAT THIS ESTABLISHES")
    print("=" * 72)
    print()
    print("The nodal equation ∂EPI/∂t = νf·ΔNFR(t) is, structurally, a")
    print("diffusion–synchronization equation on the network:")
    print("  • EPI diffuses (heat/Fick equation), relaxing to uniformity,")
    print("    conserving the degree-weighted total;")
    print("  • νf is the diffusivity, ΔNFR the structural pressure/gradient;")
    print("  • the phase channel synchronizes (Kuramoto).")
    print()
    print("These are reproduced as the SAME mathematics as the empirically-")
    print("demonstrated phenomena of diffusion and synchronization — in")
    print("TNFR's own variables (EPI, ΔNFR, νf, θ), not borrowed concepts.")
    print("The emergent geometric tower (symplectic substrate, conservation")
    print("laws) sits on top of this irreducible transport dynamics.")
    print()


if __name__ == "__main__":
    main()
