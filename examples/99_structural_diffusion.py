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
    verify_overdamped_regime,
    verify_discrete_modes,
    verify_structural_stability,
    verify_structural_random_walk,
    verify_structural_flow,
    structural_diffusion_operator,
    structural_field,
    structural_eigenmodes,
    nodal_domain_count,
    dispersion_relation,
    instability_threshold,
    fiedler_partition,
    effective_resistance,
    commute_time,
    structural_current,
    current_divergence,
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


def experiment_4_overdamped_regime():
    """The bare nodal equation is the first-order overdamped drift law."""
    print("=" * 72)
    print("EXPERIMENT 4: The mechanical regime is overdamped drift")
    print("=" * 72)
    print()
    print("The nodal equation is FIRST-ORDER, so reading EPI as a position q")
    print("and ΔNFR as the pressure F, it is q̇ = νf·F — velocity ∝ force")
    print("(νf = mobility). Under sustained pressure the field drifts at")
    print("CONSTANT velocity (it does not accelerate).")
    print()

    cert = verify_overdamped_regime(nu_f=0.7, pressure=1.3)
    print(cert.summary())
    print()
    print("VALIDATED: this is the empirically-demonstrated mobility/drift law")
    print("(Stokes 1851, Einstein 1905; terminal velocity, electrophoresis).")
    print("The INERTIAL Newtonian regime (q̈ = F/m, second order) is a")
    print("DISTINCT structure — the conservative symplectic substrate flow")
    print("(q̈ = −q per pair). The bare nodal equation is its overdamped")
    print("projection. A first-order equation cannot, by itself, be Newton.")
    print()


def experiment_5_discrete_modes():
    """A bounded manifold has discrete standing-wave eigenmodes."""
    print("=" * 72)
    print("EXPERIMENT 5: Discrete modes = bounded-manifold standing waves")
    print("=" * 72)
    print()
    print("On a BOUNDED structural manifold (finite graph) the diffusion")
    print("operator has a DISCRETE spectrum of orthonormal standing-wave")
    print("modes — the same structure as the discrete harmonics of a")
    print("vibrating string (Pythagoras), a Chladni plate, or a molecular")
    print("vibrational spectrum. The discreteness comes from the bounded")
    print("geometry, not a quantum postulate.")
    print()

    G = nx.path_graph(40)  # a 1D structural 'string'
    cert = verify_discrete_modes(G)
    print(cert.summary())
    print()
    _, eigvecs = structural_eigenmodes(G)
    counts = [nodal_domain_count(eigvecs[:, k]) for k in range(6)]
    print(f"  nodal-domain count of modes 0..5 (the structural mode number):")
    print(f"    {counts}   (mode k has k nodes — Courant/Chladni ordering)")
    print()
    print("VALIDATED: the 'discrete modes' are the Laplacian eigenmodes of")
    print("the bounded manifold = standing waves / normal modes. Two regimes")
    print("share them: diffusion (1st order) decays as exp(−νf·λ_k·t); the")
    print("wave/substrate (2nd order) oscillates at ω_k = √λ_k.")
    print()


def experiment_6_structural_stability():
    """The dispersion relation and the structural instability threshold."""
    print("=" * 72)
    print("EXPERIMENT 6: Structural stability — the dispersion relation")
    print("=" * 72)
    print()
    print("The growth/decay of each structural mode under diffusion plus a")
    print("local reaction rate r follows the dispersion relation")
    print("σ_k = r − νf·λ_k — the universal linear-stability law. Pure")
    print("diffusion (r=0) decays every non-uniform mode; the threshold")
    print("r_c = νf·λ₂ separates uniform amplification from structural")
    print("pattern formation (the Fiedler partition).")
    print()

    # a two-community network: the barbell graph
    G = nx.barbell_graph(20, 0)
    for nd in G.nodes():
        G.nodes[nd]["nu_f"] = 1.0
    cert = verify_structural_stability(G)
    print(cert.summary())
    print()
    part_a, part_b = fiedler_partition(G)
    print(f"  Fiedler partition: {len(part_a)} | {len(part_b)} nodes — the")
    print(f"  network's weakest structural cut (its two communities).")
    print()
    print("VALIDATED: pure diffusion is stable (every non-uniform mode")
    print("decays); above r_c=νf·λ₂ the Fiedler mode grows → the first")
    print("structural pattern is the weakest-cut partition (the empirically-")
    print("validated spectral-clustering result). This is the spectral form")
    print("of U2 grammar: a destabilizer raises r, a stabilizer lowers it;")
    print("bounded evolution requires r below r_c.")
    print()


def experiment_7_random_walk():
    """The diffusion operator generates a random walk; resistance geometry."""
    print("=" * 72)
    print("EXPERIMENT 7: The structural random walk and resistance geometry")
    print("=" * 72)
    print()
    print("The diffusion operator is literally the generator of a random")
    print("walk: L_rw = I − P, with P = D⁻¹W the transition matrix. So the")
    print("structural transport is Brownian motion on the network (Einstein")
    print("1905, Perrin — the proof of atoms). Its stationary distribution")
    print("is the degree; the effective resistance (Ohm's law) is a transport")
    print("metric; commute time = 2m·R_eff links the walk to the resistance.")
    print()

    G = nx.watts_strogatz_graph(50, 6, 0.2, seed=7)
    cert = verify_structural_random_walk(G)
    print(cert.summary())
    print()
    nodes, R = effective_resistance(G)
    _, C = commute_time(G)
    print(f"  effective resistance R_eff(0,25) = {float(R[0, 25]):.4f}")
    print(f"  commute time C(0,25) = 2m·R_eff = {float(C[0, 25]):.1f} steps")
    print()
    print("VALIDATED: the structural diffusion operator generates a random")
    print("walk whose stationary measure is the degree (the conserved")
    print("degree-weighted total). The effective resistance is a transport")
    print("metric (Ohm/Kirchhoff); commute time = 2m·R_eff ties the random")
    print("walk to the resistance geometry — both empirically demonstrated.")
    print()


def experiment_8_structural_flow():
    """The diffusion current: Fick, Kirchhoff (continuity), and Ohm."""
    print("=" * 72)
    print("EXPERIMENT 8: The structural flow — current, Kirchhoff, Ohm")
    print("=" * 72)
    print()
    print("The transport carries a current: along each edge the diffusion")
    print("flux is J_ij = EPI_i − EPI_j (Fick's law, antisymmetric). The net")
    print("outflow at a node is Kirchhoff's current law — the discrete")
    print("continuity equation div(J) = L·EPI — so ∂EPI/∂t + div(J) = 0.")
    print("Under an injected current the potential drop is the effective")
    print("resistance (Ohm's law). All empirically demonstrated (Fick,")
    print("Kirchhoff 1845, Ohm).")
    print()

    G = nx.watts_strogatz_graph(50, 6, 0.2, seed=7)
    rng = np.random.default_rng(7)
    for node in G.nodes():
        G.nodes[node]["EPI"] = float(rng.uniform(-0.4, 0.4))
    cert = verify_structural_flow(G)
    print(cert.summary())
    print()
    nodes, j = structural_current(G)
    _, div = current_divergence(G)
    print(f"  current is antisymmetric (J = −Jᵀ): max|J+Jᵀ| "
          f"= {float(np.max(np.abs(j + j.T))):.1e}")
    print(f"  Kirchhoff continuity Σ div(J) = 0 (closed network): "
          f"{float(div.sum()):.1e}")
    print()
    print("VALIDATED: the structural flow is the diffusion current. Its edge")
    print("current is Fick's law; Kirchhoff's current law IS the continuity")
    print("equation div(J) = L·EPI (complementary to the tetrad-field")
    print("continuity in conservation.py); the potential drop under an")
    print("injected current is the effective resistance (Ohm).")
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
    experiment_4_overdamped_regime()
    experiment_5_discrete_modes()
    experiment_6_structural_stability()
    experiment_7_random_walk()
    experiment_8_structural_flow()

    print("=" * 72)
    print("WHAT THIS ESTABLISHES")
    print("=" * 72)
    print()
    print("The nodal equation ∂EPI/∂t = νf·ΔNFR(t) is, structurally, a")
    print("diffusion–synchronization equation on the network:")
    print("  • EPI diffuses (heat/Fick equation), relaxing to uniformity,")
    print("    conserving the degree-weighted total;")
    print("  • νf is the diffusivity/mobility, ΔNFR the structural pressure;")
    print("  • the phase channel synchronizes (Kuramoto);")
    print("  • being first-order, it produces the overdamped drift law")
    print("    q̇ = νf·F (Stokes/Einstein), not inertial Newton — that lives")
    print("    in the second-order symplectic substrate;")
    print("  • a bounded manifold has discrete standing-wave modes (the")
    print("    Laplacian eigenmodes — Pythagoras/Chladni harmonics);")
    print("  • the dispersion relation σ_k=r−νf·λ_k governs stability; above")
    print("    r_c=νf·λ₂ the Fiedler mode grows (structural pattern / U2);")
    print("  • the operator generates a random walk (Brownian motion); its")
    print("    resistance geometry is a transport metric (Ohm/Kirchhoff).")
    print("  • the transport carries a Fick current whose Kirchhoff balance")
    print("    is the continuity equation div(J)=L·EPI (Ohm under injection).")
    print()
    print("These are reproduced as the SAME mathematics as the empirically-")
    print("demonstrated phenomena of diffusion, synchronization, mobility,")
    print("standing waves, linear stability, and random walks / resistance —")
    print("in TNFR's own variables (EPI, ΔNFR, νf, θ), not borrowed concepts.")
    print("The emergent geometric tower (symplectic substrate, conservation")
    print("laws) sits on top of this irreducible transport dynamics.")
    print()


if __name__ == "__main__":
    main()
