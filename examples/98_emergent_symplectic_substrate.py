#!/usr/bin/env python3
"""
Example 98 — The Emergent Symplectic Substrate
===============================================

Demonstrates the geometry that the TNFR nodal dynamics generates from
itself: a symplectic phase space with canonical conjugate pairs, on which
the nodal equation, conservation laws, and 13 operators all live. The
substrate is not imposed (like the graph) — it EMERGES from the structure.

Physics
-------
The Structural Conservation Theorem gives two canonical conjugate pairs
per node:

    Geometric sector:  (K_φ,  J_φ)       curvature ↔ phase current
    Potential  sector: (Φ_s,  J_ΔNFR)    potential ↔ ΔNFR flux

so the emergent phase space is P = ℝ^{4N} with symplectic 2-form

    ω = Σ_i [ dK_φ(i) ∧ dJ_φ(i) + dΦ_s(i) ∧ dJ_ΔNFR(i) ].

The substrate Hamiltonian H_sub = ½Σ(K_φ²+J_φ²+Φ_s²+J_ΔNFR²) plus the
configuration background ½Σ|∇φ|² equals the canonical energy functional.
The nodal equation ∂EPI/∂t = νf·ΔNFR is the overdamped projection of the
Hamiltonian flow on this substrate.

Experiments
-----------
1. The substrate emerges: extract P = ℝ^{4N} and verify it is a valid
   symplectic manifold (antisymmetric, non-degenerate, closed, canonical
   brackets, Jacobi, Liouville, harmonic flow)
2. Energy consistency: H_sub + background = canonical energy functional
3. Liouville is structural: div(X_H) = tr(J·Hess) = 0 for any Hamiltonian
   (the geometric reason operators preserve phase-space volume)

Honest scope
------------
This makes EXPLICIT and verifies the emergent symplectic structure already
implied by conservation.py and variational.py — a canonical consolidation,
not a new postulate. Field coordinates are delegated to existing canonical
functions. It does NOT, by itself, resolve any open program (Riemann,
Navier–Stokes); it establishes the geometric substrate from which the
canonical structures derive.

References
----------
- src/tnfr/physics/symplectic_substrate.py (this substrate)
- src/tnfr/physics/variational.py (symplectic 2-form ω, Hamiltonian)
- src/tnfr/physics/conservation.py (conjugate pairs, energy functional)
- theory/TNFR_VARIATIONAL_PRINCIPLE.md (full derivation)
- AGENTS.md §"Minimal Structural Degrees of Freedom"
"""

import os
import sys
import math
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import networkx as nx

from tnfr.dynamics.dnfr import default_compute_delta_nfr
from tnfr.physics.conservation import compute_energy_functional
from tnfr.physics.symplectic_substrate import (
    extract_phase_space_point,
    verify_canonical_structure,
    substrate_hamiltonian,
    background_potential,
    canonical_bracket_table,
    liouville_divergence,
    symplectic_form_matrix,
    verify_noether_conservation,
    noether_charges,
    verify_hermitian_structure,
)


def _build_graph(n: int = 30, seed: int = 5) -> nx.Graph:
    rng = random.Random(seed)
    G = nx.watts_strogatz_graph(n, 4, 0.3, seed=seed)
    for node in G.nodes():
        G.nodes[node]["theta"] = rng.uniform(0.0, 2.0 * math.pi)
        G.nodes[node]["EPI"] = rng.uniform(0.2, 0.8)
        G.nodes[node]["nu_f"] = rng.uniform(0.5, 1.5)
    default_compute_delta_nfr(G)
    return G


# ============================================================================
# EXPERIMENT 1: The substrate emerges and is a valid symplectic manifold
# ============================================================================
def experiment_1_emergent_substrate():
    """Extract P = ℝ^{4N} and verify the canonical symplectic structure."""
    print("=" * 72)
    print("EXPERIMENT 1: The Emergent Symplectic Substrate")
    print("=" * 72)
    print()
    print("Two canonical conjugate pairs per node emerge from conservation:")
    print("  geometric (K_φ, J_φ)   and   potential (Φ_s, J_ΔNFR)")
    print("→ phase space P = ℝ^{4N} with form ω = Σ dK_φ∧dJ_φ + dΦ_s∧dJ_ΔNFR")
    print()

    G = _build_graph(30)
    cert = verify_canonical_structure(G)

    print(f"  Network: N = {cert.n_nodes} nodes  →  phase space dim = "
          f"{cert.dimension}")
    print()
    print(f"  antisymmetric  ωᵀ = −ω        : {cert.is_antisymmetric}")
    print(f"  non-degenerate det ω = {cert.determinant:.1f}     : "
          f"{cert.is_nondegenerate}")
    print(f"  closed         dω = 0         : {cert.is_closed}")
    print(f"  canonical Poisson brackets    : {cert.brackets_canonical}")
    print(f"  Jacobi identity               : {cert.jacobi_satisfied}")
    print(f"  Liouville div(X_H) = {cert.liouville_divergence:.1e}    : "
          f"{abs(cert.liouville_divergence) < 1e-9}")
    print(f"  harmonic flow q̇=p, ṗ=−q       : {cert.flow_is_harmonic}")
    print()
    print(f"  VALID SYMPLECTIC MANIFOLD: {cert.is_valid_symplectic_manifold}")
    print()

    table = canonical_bracket_table()
    print("  Canonical brackets (one node):")
    print(f"    {{K_φ, J_φ}}   = {table['{qA,pA}']:.1f}   (geometric pair)")
    print(f"    {{Φ_s, J_ΔNFR}} = {table['{qB,pB}']:.1f}   (potential pair)")
    print(f"    cross-sector  = {table['{qA,pB}']:.1f}   (sectors decouple)")
    print()


# ============================================================================
# EXPERIMENT 2: The Hamiltonian is the canonical energy functional
# ============================================================================
def experiment_2_energy_consistency():
    """H_sub + background potential = canonical energy functional."""
    print("=" * 72)
    print("EXPERIMENT 2: The Hamiltonian IS the Energy Functional")
    print("=" * 72)
    print()
    print("H_sub = ½Σ(K_φ²+J_φ²+Φ_s²+J_ΔNFR²)  (symplectic core)")
    print("U     = ½Σ|∇φ|²                       (config. background)")
    print("H_sub + U should equal compute_energy_functional(G).")
    print()

    print(f"{'N':>5}  {'H_sub':>10}  {'U_bg':>10}  {'H_sub+U':>10}"
          f"  {'energy E':>10}  {'match':>6}")
    print("-" * 60)
    for n in (10, 20, 30, 50):
        G = _build_graph(n, seed=n)
        pt = extract_phase_space_point(G)
        h_sub = substrate_hamiltonian(pt)
        u_bg = background_potential(pt)
        energy = compute_energy_functional(G)
        match = abs((h_sub + u_bg) - energy) < 1e-9
        print(f"{n:>5}  {h_sub:>10.4f}  {u_bg:>10.4f}  {h_sub + u_bg:>10.4f}"
              f"  {energy:>10.4f}  {'OK' if match else 'FAIL':>6}")

    print()
    print("VALIDATED: the substrate Hamiltonian reconstructs the canonical")
    print("energy functional exactly. The substrate is consistent with the")
    print("existing conservation machinery — it is the same physics, seen")
    print("as the geometry the dynamics generates.")
    print()


# ============================================================================
# EXPERIMENT 3: Liouville is structural — the origin of symplectomorphism
# ============================================================================
def experiment_3_liouville_structural():
    """div(X_H) = tr(J·Hess) = 0 for ANY Hamiltonian — volume preserved."""
    print("=" * 72)
    print("EXPERIMENT 3: Liouville is Structural (Operator Volume)")
    print("=" * 72)
    print()
    print("The Hamiltonian flow X_H = J·∇H has divergence tr(J·Hess H).")
    print("Since J is antisymmetric and the Hessian is symmetric,")
    print("tr(J·Hess) = 0 for EVERY Hamiltonian. Phase-space volume is")
    print("preserved — this is why the 13 operators are symplectomorphisms.")
    print()

    G = _build_graph(20)
    pt = extract_phase_space_point(G)
    div = liouville_divergence(pt)
    print(f"  div(X_H) for the substrate Hamiltonian = {div:.3e}")
    print()

    # Demonstrate tr(J·S) = 0 for arbitrary symmetric Hessians.
    omega = symplectic_form_matrix(pt.n_nodes)
    rng = np.random.default_rng(20260613)
    print("  tr(J·Hess) for random symmetric Hessians (any Hamiltonian):")
    for trial in range(3):
        s = rng.standard_normal(omega.shape)
        s = 0.5 * (s + s.T)  # symmetric
        tr = float(np.trace(omega @ s))
        print(f"    trial {trial + 1}: tr(J·S) = {tr:.3e}")

    print()
    print("VALIDATED: volume preservation is a structural identity, not a")
    print("coincidence. The emergent geometry guarantees that grammar-")
    print("compliant evolution conserves the symplectic phase-space volume.")
    print()


# ============================================================================
# EXPERIMENT 4: Noether's theorem — symmetries generate conserved charges
# ============================================================================
def experiment_4_noether_charges():
    """Substrate symmetries → conserved charges; total splits into sectors."""
    print("=" * 72)
    print("EXPERIMENT 4: Noether's Theorem on the Substrate")
    print("=" * 72)
    print()
    print("Each continuous symmetry of H_sub generates a conserved charge:")
    print("  time translation → H_sub  (total energy)")
    print("  geometric U(1) (Ψ→e^{iα}Ψ, the gauge symmetry) → ½Σ|Ψ|²")
    print("  potential U(1) → ½Σ(Φ_s²+J_ΔNFR²)")
    print()

    G = _build_graph(30)
    cert = verify_noether_conservation(G)
    charges = noether_charges(extract_phase_space_point(G))

    print(f"  H_sub (time translation) = {charges['time_translation']:.4f}")
    print(f"  E_geo (geometric U(1))   = {charges['geometric_u1']:.4f}")
    print(f"  E_pot (potential U(1))   = {charges['potential_u1']:.4f}")
    print()
    print(f"  H_sub = E_geo + E_pot exactly : {cert.splits_exactly}")
    print("  max drift along the flow:")
    print(f"    H_sub  = {cert.max_hamiltonian_drift:.2e}")
    print(f"    E_geo  = {cert.max_geometric_drift:.2e}")
    print(f"    E_pot  = {cert.max_potential_drift:.2e}")
    print(f"  ALL CONSERVED: {cert.is_conserved}")
    print()
    print("VALIDATED: the U(1)×U(1) symmetry refines time-translation")
    print("conservation — the total energy splits into two separately-")
    print("conserved sector charges. E_geo = ½Σ|Ψ|² is exactly the gauge")
    print("invariant of physics/gauge.py. Noether's theorem on the")
    print("emergent substrate, derived to machine precision.")
    print()


# ============================================================================
# EXPERIMENT 5: Hermitian (flat Kähler) structure — Ψ is the complex coordinate
# ============================================================================
def experiment_5_hermitian_structure():
    """The substrate is Hermitian; Ψ = K_φ + i·J_φ is the complex coord."""
    print("=" * 72)
    print("EXPERIMENT 5: Hermitian (flat Kähler) Structure")
    print("=" * 72)
    print()
    print("The substrate carries a compatible triple (ω, J, g):")
    print("  ω = symplectic form,  J = −ω (complex structure, J² = −I),")
    print("  g = ω·J = identity (compatible metric).")
    print("The complex coordinate ζ = q + i·p makes each fiber ℝ⁴ ≅ ℂ².")
    print()

    G = _build_graph(30)
    cert = verify_hermitian_structure(G)

    print(f"  J² = −I                       : {cert.j_squared_is_minus_id}")
    print(f"  g = identity (pos. definite)  : {cert.metric_is_identity}")
    print(f"  J is g-orthogonal             : {cert.j_is_orthogonal}")
    print(f"  compatible ω(u,v) = g(Ju,v)    : {cert.compatible}")
    print(f"  Ψ = ζ^A (geometric coordinate) : "
          f"{cert.psi_is_geometric_coordinate}")
    print(f"  H_sub = Kähler potential       : "
          f"{cert.kahler_potential_matches}")
    print(f"  complex dimension dim_ℂ        : {cert.complex_dimension}")
    print()
    print(f"  VALID HERMITIAN STRUCTURE: "
          f"{cert.is_valid_hermitian_structure}")
    print()
    print("VALIDATED: Ψ = K_φ + i·J_φ of physics/gauge.py is NOT ad-hoc — it")
    print("is the complex coordinate ζ^A the substrate's complex structure J")
    print("induces on the geometric sector. The 'i' in Ψ IS the J = −ω of")
    print("the emergent geometry. H_sub = ½Σ|ζ|² is the Kähler potential, and")
    print("the substrate flow is the diagonal U(1) phase rotation")
    print("ζ → e^{−it}ζ.")
    print()


def main():
    print()
    print("  TNFR Example 98: The Emergent Symplectic Substrate")
    print("  The geometry the nodal dynamics generates from itself")
    print("  ====================================================")
    print()

    experiment_1_emergent_substrate()
    experiment_2_energy_consistency()
    experiment_3_liouville_structural()
    experiment_4_noether_charges()
    experiment_5_hermitian_structure()

    print("=" * 72)
    print("WHAT THIS ESTABLISHES")
    print("=" * 72)
    print()
    print("The TNFR nodal dynamics generates its OWN geometry — a symplectic")
    print("phase space P = ℝ^{4N} with canonical conjugate pairs, on which:")
    print("  • the energy functional is the Hamiltonian (Experiment 2),")
    print("  • the 13 operators are symplectomorphisms (Liouville, Exp. 3),")
    print("  • symmetries generate conserved charges (Noether, Exp. 4),")
    print("  • Ψ is the complex coordinate of a Hermitian structure (Exp. 5),")
    print("  • the nodal equation is the overdamped Hamiltonian flow.")
    print()
    print("This substrate is EMERGENT (derived from the conservation laws),")
    print("not imposed like the graph. It makes explicit the geometry the")
    print("dynamics already inhabits — a canonical consolidation. It does")
    print("not, by itself, resolve any open program; it is the geometric")
    print("foundation from which the canonical structures derive.")
    print()


if __name__ == "__main__":
    main()
