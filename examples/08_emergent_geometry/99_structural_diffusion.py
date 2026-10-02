"""Finite demonstrations of the isolated EPI diffusion channel.

The EPI contribution is p_epi=-L_rw*x. With fixed connected reciprocal weights
and common positive capacity, x_dot=-nu*L_rw*x relaxes to the degree-weighted
mean. Heterogeneous capacities instead use invariant weights d_i/nu_i, and the
full multichannel pressure need not be an unforced diffusion.
The reaction comparison explicitly adds r*x, giving modal growth r-nu*lambda.
Phase-pressure observations do not themselves define a Kuramoto phase law;
capacity is not automatically angular frequency or a material diffusivity.
These graph identities provide conditional models, not physical identification.
See theory/TNFR_DIFFUSION_STABILITY_THEOREM.md and
 theory/NODAL_PARAMETER_FOUNDATIONS.md.
"""

import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import networkx as nx
import numpy as np

from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_DNFR
from tnfr.dynamics import default_compute_delta_nfr
from tnfr.observers import kuramoto_order
from tnfr.physics.structural_diffusion import (
    commute_time,
    current_divergence,
    degree_weighted_total,
    effective_resistance,
    fiedler_partition,
    nodal_domain_count,
    relaxation_spectrum,
    structural_current,
    structural_diffusion_operator,
    structural_eigenmodes,
    structural_field,
    verify_discrete_modes,
    verify_overdamped_regime,
    verify_structural_diffusion,
    verify_structural_flow,
    verify_structural_random_walk,
    verify_structural_stability,
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
    print("EXPERIMENT 1: The isolated EPI channel is graph diffusion")
    print("=" * 72)
    print()
    print("The full configured ΔNFR has several distinct channels.")
    print("Its isolated EPI channel is −L_rw·EPI in exact arithmetic, the")
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
    g2.graph["DNFR_WEIGHTS"] = {"phase": 0.0, "epi": 1.0, "vf": 0.0, "topo": 0.0}
    default_compute_delta_nfr(g2)
    dnfr = np.array([float(get_attr(g2.nodes[nd], ALIAS_DNFR, 0.0)) for nd in nodes])
    residual = float(np.max(np.abs(dnfr - (-(lap @ epi)))))
    print(f"  max |ΔNFR_epi − (−L_rw·EPI)| = {residual:.2e}")
    print("  -> the EPI channel of the nodal equation IS graph diffusion")
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
    print(
        f"  decay rates of diag(νf)·L_rw (first 6): "
        f"{[round(float(x), 4) for x in spec[:6]]}"
    )
    print("  The zero rate corresponds to uniform form; the first positive")
    print(
        f"  rate sets slow relaxation. Initial Σ deg·EPI = "
        f"{degree_weighted_total(G):.4f}"
    )
    print()
    print("This diagnostic freezes the prepared heterogeneous capacities")
    print("and samples pure-EPI diffusion, not the full configured engine.")
    print("With these positive capacities and connected reciprocal support,")
    print("the invariant weights are d_i/ν_i; degree weights alone require")
    print("common capacity. The certificate retains the distinction.")
    print()


def experiment_3_synchronization():
    """Observe a separately supplied phase-response update on one preparation."""
    print("=" * 72)
    print("EXPERIMENT 3: Auxiliary phase-response demonstration")
    print("=" * 72)
    print()
    print("This comparison supplies θ_next = θ + 0.3·ΔNFR_phase.")
    print("Using the pressure read-out as phase motion is an additional law,")
    print("not a consequence of dEPI/dt = νf·ΔNFR or a Kuramoto derivation.")
    print()

    G = _build(60, seed=3)
    G.graph["DNFR_WEIGHTS"] = {"phase": 1.0, "epi": 0.0, "vf": 0.0, "topo": 0.0}
    nodes = list(G.nodes())
    r0 = kuramoto_order(G)
    for _ in range(300):
        default_compute_delta_nfr(G)
        for nd in nodes:
            g_phase = float(get_attr(G.nodes[nd], ALIAS_DNFR, 0.0))
            G.nodes[nd]["theta"] = float(G.nodes[nd]["theta"]) + 0.3 * g_phase
    r1 = kuramoto_order(G)
    print(f"  Kuramoto order R: start {r0:.4f} -> end {r1:.4f}")
    print("  This finite response does not guarantee synchronization in general.")
    print()


def experiment_4_overdamped_regime():
    """Compare the nodal product with a selected first-order mobility law."""
    print("=" * 72)
    print("EXPERIMENT 4: Conditional first-order mobility comparison")
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
    print("The arithmetic supports a mobility interpretation once a force")
    print("mapping and clock are supplied; it does not establish that mapping.")
    print("A separately damped graph wave q''+γq'+L_rw q=0 can approach")
    print("diffusion with mobility 1/γ in its overdamped slow-rate limit.")
    print("The isotropic substrate instead has identity stiffness, q''=-q.")
    print("No projection from it to the full nodal dynamics is established.")
    print()


def experiment_5_discrete_modes():
    """Compare finite normalized graph modes under two supplied equations."""
    print("=" * 72)
    print("EXPERIMENT 5: Finite graph modes and an auxiliary wave comparison")
    print("=" * 72)
    print()
    print("The symmetric normalized Laplacian of this reciprocal finite")
    print("graph has an orthonormal eigenbasis and a finite spectrum.")
    print("These modes describe selected graph diffusion or a separately")
    print("supplied graph wave. No physical mode identification is inferred.")
    print()

    G = nx.path_graph(40)  # a 1D structural 'string'
    cert = verify_discrete_modes(G)
    print(cert.summary())
    print()
    _, eigvecs = structural_eigenmodes(G)
    counts = [nodal_domain_count(eigvecs[:, k]) for k in range(6)]
    print("  nodal-domain count of modes 0..5:")
    print(f"    {counts}   (connected sign domains, not zero locations)")
    print()
    print("With normalized modal coordinates, fixed diffusion decays as")
    print("exp(−νf·λ_k·t); the separate wave q''=-L_sym q oscillates at")
    print("ω_k=√λ_k. The isotropic substrate has different stiffness and")
    print("does not inherit these graph-wave frequencies automatically.")
    print()


def experiment_6_structural_stability():
    """The dispersion relation and the structural instability threshold."""
    print("=" * 72)
    print("EXPERIMENT 6: Structural stability — the dispersion relation")
    print("=" * 72)
    print()
    print("The growth/decay of each structural mode under diffusion plus a")
    print("local reaction rate r follows the dispersion relation")
    print("σ_k = r − νf·λ_k — the declared reaction-diffusion modal law. Pure")
    print("diffusion (r=0) decays every non-uniform mode; the threshold")
    print("r_c = νf·λ₂ marks growth of the first nonuniform mode.")
    print("The uniform mode already grows whenever r>0.")
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
    print("  low-energy spectral partition on this prepared two-community graph.")
    print()
    print("Pure diffusion decays every nonuniform mode. Above r_c=νf·λ₂")
    print("the Fiedler mode also grows under this supplied linear reaction.")
    print("This does not prove nonlinear pattern maintenance or a minimum")
    print("graph cut, and U2 grammar does not derive the chosen reaction r.")
    print("For r>0, arbitrary initial states need not remain bounded even")
    print("below r_c, because their uniform component can grow.")
    print()


def experiment_7_random_walk():
    """The diffusion operator generates a random walk; resistance geometry."""
    print("=" * 72)
    print("EXPERIMENT 7: The structural random walk and resistance geometry")
    print("=" * 72)
    print()
    print("The negative diffusion operator -L_rw=P-I generates a graph")
    print("Markov process; P=D⁻¹W is the supplied transition matrix.")
    print("On connected reciprocal support, the stationary distribution")
    print("is proportional to degree. For this unit-edge graph, effective")
    print("resistance and commute time obey C=2m·R_eff.")
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
    print("The selected graph generator defines a random walk whose")
    print("stationary measure is degree-weighted (the conserved")
    print("degree-weighted total). The effective resistance is a transport")
    print("metric (Ohm/Kirchhoff); commute time = 2m·R_eff ties the random")
    print("walk to resistance geometry. Physical identification is separate.")
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
    print("balance div(J)=(D-W)·EPI. The selected row then satisfies")
    print("d_i·dEPI_i/dt + ν_i·div(J)_i = 0, with zero rows at isolates.")
    print("Under an independently injected unit current, the solved")
    print("potential drop is the effective resistance of the graph.")
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
    print(
        f"  current is antisymmetric (J = −Jᵀ): max|J+Jᵀ| "
        f"= {float(np.max(np.abs(j + j.T))):.1e}"
    )
    print(
        f"  Kirchhoff continuity Σ div(J) = 0 (closed network): "
        f"{float(div.sum()):.1e}"
    )
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
    print("The held pure-EPI law ∂EPI/∂t = -νf L_rw EPI is a")
    print("diffusion equation under the stated reciprocal-graph hypotheses:")
    print("  • EPI diffuses (heat/Fick equation), relaxing to uniformity,")
    print("    conserving the degree-weighted total;")
    print("  • νf is the diffusivity/mobility, ΔNFR the structural pressure;")
    print("  • the phase experiment uses a separately supplied synchronization law;")
    print("  • a selected mobility/force interpretation gives a first-order drift;")
    print("    the units and physical force mapping require separate justification.")
    print("    The auxiliary symplectic substrate is a different declared model;")
    print("  • a finite reciprocal graph has normalized Laplacian modes;")
    print("    a separately supplied graph wave assigns their frequencies;")
    print("  • the dispersion relation σ_k=r−νf·λ_k governs stability; above")
    print("    r_c=νf·λ₂ its nonuniform mode grows; the uniform mode grows for r>0;")
    print("  • the generator defines a graph Markov process; its")
    print("    resistance geometry is a transport metric (Ohm/Kirchhoff).")
    print("  • the selected current has balance div(J)=(D-W)·EPI;")
    print("    the normalized nodal row also retains degree and capacity.")
    print()
    print("These conditional graph identities support comparisons with")
    print("models of diffusion, separately supplied synchronization, mobility,")
    print("standing waves, linear stability, and random walks / resistance —")
    print("with declared mappings and units, rather than physical identification.")
    print("The auxiliary substrate and full multichannel dynamics require their own")
    print("bridge, realizability and conservation assumptions.")
    print()


if __name__ == "__main__":
    main()
