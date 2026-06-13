#!/usr/bin/env python3
"""
Example 104 — Navier–Stokes Is Not Riemann: Native Transport vs Foreign Arithmetic
=================================================================================

Answers the comparative question raised after Example 103: does the
Navier–Stokes program meet the same obstacle as Riemann under the new
physics (the emergent symplectic substrate, Example 98; structural
transport, Example 99)? The measured answer is NO — and the asymmetry has
a precise, structural origin. Neither problem is solved; but the new
physics relates to them in opposite ways.

The structural asymmetry (measured)
-----------------------------------
The emergent substrate reads the tetrad (K_φ, J_φ, Φ_s, J_ΔNFR), which is
built from the PHASE θ and the pressure ΔNFR — never from ν_f.

  • RIEMANN (P14): the entire prime content lives in ν_f = k·log p, with
    θ = 0 and ΔNFR = 0 by construction. So the static substrate is EXACTLY
    zero — BLIND to the primes (Example 103). Under the dynamics it
    re-expresses {k·log p} but adds nothing; the residual S(T) =
    (1/π)·arg ζ(½+iT) is ARITHMETIC — orthogonal to the transport/diffusion
    geometry the new physics provides.

  • NAVIER–STOKES: the canonical field dictionary is u_a ↔ φ^(a)
    (velocity IS the phase field), ω ↔ K_φ (vorticity IS phase curvature),
    p ↔ Φ_s (pressure IS the structural potential). So the physical content
    lives in the PHASE — exactly what the tetrad reads. The substrate is
    POPULATED from the initial condition (NOT blind), and the NS observables
    (vorticity, enstrophy) ARE tetrad quantities. The residual — the K_φ
    cascade at scale → 0 (the blow-up question) — IS transport, native to
    the new physics (structural diffusion, REMESH-∞).

Measured here (3D Taylor–Green n=8, P14 n_primes=10×K=4):
  • Riemann P14 substrate: |Ψ| = |Φ_s| = |∇φ| = 0 (exact, machine zero).
  • NS substrate (velocity → phase): |Ψ| ≈ 0.41, |∇φ| ≈ 0.31, |K_φ| ≈ 0.29
    — the geometric sector is populated by the velocity field.
  • NS enstrophy as a tetrad sum Z ~ Σ K_φ² is finite and non-zero: the
    physics IS a tetrad observable.

Already-established cross-program evidence (cited, not re-measured)
------------------------------------------------------------------
The canonical REMESH-∞ asymptotic limit, applied to each program, gives
OPPOSITE verdicts — consistent with this asymmetry:
  • NS (scale → 0 on the K_φ cascade, N12/N13): STRUCTURAL_EFFECT
    (non-degenerate traction on 3D Taylor–Green).
  • Riemann (τ → ∞ on the prime ladder, R∞-1b/1c): INDETERMINATE_
    DEGENERATE (the S_n prime-relabelling symmetry obstruction, CCET).
"Not in tension — different substrates" (NS status memo §N13).

Honest scope
------------
- This does NOT solve, close, or advance either Clay/Millennium problem.
  3D NS global regularity (NS-G_blowup / Clay) remains OPEN; Riemann
  G4 = RH remains OPEN. The program statuses are unchanged.
- "Native arena" is NOT "solved". For NS the new physics is the right
  language (the obstacle is transport — a K_φ cascade), but the analytic
  scale → 0 limit (NS-G_blowup) is still open, exactly as G4 is open for
  Riemann. The asymmetry is about WHICH new-physics tools are native, not
  about closure.
- The field dictionary (u↔φ, ω↔K_φ, p↔Φ_s) is the canonical NS-program
  translation (NS status memo); this example measures its consequence for
  the substrate, it does not re-derive it.
- The KS/spectral comparisons of Example 103 are not repeated here; this
  example isolates the populated-vs-blind asymmetry and its transport
  reading.

References
----------
- examples/08_emergent_geometry/103_emergent_substrate_meets_riemann.py (Riemann side; blindness)
- examples/08_emergent_geometry/99_structural_diffusion.py (the nodal equation IS graph diffusion)
- src/tnfr/navier_stokes/operator.py (u_a = φ^(a); 3D Taylor–Green)
- src/tnfr/physics/symplectic_substrate.py (extract_phase_space_point)
- AGENTS.md §"TNFR-Riemann Program" / §"Transport Content of the Nodal Equation"
- theory/TNFR_NAVIER_STOKES_RESEARCH_NOTES.md §11 (REMESH-∞ cross-link)
"""

import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import numpy as np

from tnfr.riemann.prime_ladder_hamiltonian import build_prime_ladder_graph
from tnfr.navier_stokes.operator import (
    build_torus_graph_3d,
    taylor_green_initial_condition_3d,
)
from tnfr.physics.symplectic_substrate import extract_phase_space_point


def _substrate_geometric(G):
    """Extract substrate geometric-sector magnitudes (Ψ, |∇φ|, K_φ).

    The potential sector (Φ_s) is sourced by ΔNFR (pressure); with a pure
    velocity IC it is trivial, so we read the geometric sector the velocity
    populates. The np.errstate guard silences the 0/0 in the unused Φ_s.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(invalid="ignore", divide="ignore"):
            pt = extract_phase_space_point(G)
    psi = np.abs(pt.k_phi + 1j * pt.j_phi)
    return pt, psi


# ============================================================================
# EXPERIMENT 1: Blind (Riemann) vs populated (Navier–Stokes)
# ============================================================================
def experiment_1_blind_vs_populated():
    """P14 substrate is exactly zero; the NS substrate is populated."""
    print("=" * 72)
    print("EXPERIMENT 1: Blind (Riemann P14) vs Populated (Navier–Stokes)")
    print("=" * 72)
    print()
    print("The substrate reads the tetrad from the PHASE θ. Where does each")
    print("program put its content?")
    print()

    # Riemann P14: content in ν_f, phase = 0
    Gp = build_prime_ladder_graph(10, max_power=4)
    _ptp, psi_p = _substrate_geometric(Gp)
    print("  RIEMANN P14 (content in ν_f = k·log p; θ = 0, ΔNFR = 0):")
    print(f"    |Ψ| max = {psi_p.max():.2e}  ->  substrate BLIND: {psi_p.max() < 1e-9}")
    print("    (the tetrad does not read ν_f — Example 103)")
    print()

    # Navier–Stokes: velocity IS the phase field
    n = 8
    G = build_torus_graph_3d(n)
    u, _v, _w = taylor_green_initial_condition_3d(G, amplitude=1.0)
    for idx, node in enumerate(G.nodes):
        G.nodes[node]["phase"] = float(u[idx])
        G.nodes[node]["theta"] = float(u[idx])
        G.nodes[node]["EPI"] = float(u[idx])
        G.nodes[node]["nu_f"] = 1.0
    pt_n, psi_n = _substrate_geometric(G)
    print("  NAVIER–STOKES (velocity u_a IS the phase field φ^(a)):")
    print(f"    |Ψ| max = {psi_n.max():.3f}, mean = {psi_n.mean():.3f}")
    print(f"    |∇φ| max = {np.abs(pt_n.grad_phi).max():.3f},  "
          f"|K_φ| max = {np.abs(pt_n.k_phi).max():.3f}")
    print(f"    -> substrate BLIND: {psi_n.max() < 1e-9}  "
          f"(the velocity populates the geometric sector)")
    print()
    print("VERDICT: the same substrate is BLIND for Riemann (content in ν_f)")
    print("but POPULATED for Navier–Stokes (content in the phase = velocity).")
    print("The new geometry is native to NS, foreign to the Riemann content.")
    print()
    return pt_n


# ============================================================================
# EXPERIMENT 2: In Navier–Stokes the physics IS the tetrad
# ============================================================================
def experiment_2_physics_is_tetrad(pt_n):
    """Vorticity ↔ K_φ, enstrophy ↔ Σ K_φ²: NS observables are tetrad fields."""
    print("=" * 72)
    print("EXPERIMENT 2: In Navier–Stokes the Physics IS the Tetrad")
    print("=" * 72)
    print()
    print("Canonical NS field dictionary (NS status memo):")
    print("    u_a ↔ φ^(a)   (velocity      = phase field)")
    print("    ω   ↔ K_φ     (vorticity     = phase curvature)")
    print("    p   ↔ Φ_s     (pressure      = structural potential)")
    print()
    z_tetrad = float(np.sum(pt_n.k_phi ** 2))
    print(f"  enstrophy as a tetrad sum   Z ~ Σ K_φ² = {z_tetrad:.2f}")
    print("  (finite, non-zero: the NS observable IS a tetrad quantity)")
    print()
    print("VERDICT: the Navier–Stokes observables (vorticity, enstrophy) are")
    print("not foreign to the substrate — they ARE its fields. Contrast")
    print("Riemann, where the observable (the prime distribution / zeros)")
    print("is NOT a tetrad field; the substrate only re-expresses {k·log p}.")
    print()


# ============================================================================
# EXPERIMENT 3: The residuals — transport (NS) vs arithmetic (Riemann)
# ============================================================================
def experiment_3_residuals():
    """NS residual is the K_φ cascade (transport); Riemann's is S(T)."""
    print("=" * 72)
    print("EXPERIMENT 3: The Residuals — Transport (NS) vs Arithmetic (Riemann)")
    print("=" * 72)
    print()
    print("Each program's open residual sits in a different place:")
    print()
    print("  NAVIER–STOKES residual = the K_φ cascade at scale → 0 (whether")
    print("    enstrophy stays bounded or blows up). K_φ is phase curvature —")
    print("    a TRANSPORT object. The nodal equation IS graph diffusion")
    print("    (Example 99), so this residual is NATIVE to the new physics.")
    print()
    print("  RIEMANN residual = S(T) = (1/π)·arg ζ(½+iT), the oscillatory")
    print("    half of the admissible rescaling (P28/P30, N15 = ker R_∞).")
    print("    It is ARITHMETIC — orthogonal to the transport geometry; the")
    print("    substrate re-expresses {k·log p} but cannot supply it (Ex 103).")
    print()
    print("Cross-program evidence already on record (cited, not re-run):")
    print("  the canonical REMESH-∞ limit gives OPPOSITE verdicts —")
    print("    NS (scale → 0, N12/N13):    STRUCTURAL_EFFECT (non-degenerate)")
    print("    Riemann (τ → ∞, R∞-1b/1c):  INDETERMINATE_DEGENERATE (S_n)")
    print("  'Not in tension — different substrates' (NS status memo).")
    print()


# ============================================================================
# EXPERIMENT 4: Synthesis — not the same, but neither is solved
# ============================================================================
def experiment_4_synthesis():
    """Honest placement: native arena ≠ solved."""
    print("=" * 72)
    print("EXPERIMENT 4: Synthesis — Not the Same, but Neither Is Solved")
    print("=" * 72)
    print()
    print("  Riemann:  the new geometry is the right ARENA but the obstacle")
    print("    (S(T)) is ARITHMETIC — foreign to transport. The substrate")
    print("    re-expresses {k·log p}; G4 = RH stays OPEN.")
    print()
    print("  Nav–Stk:  the new geometry is the right arena AND the obstacle")
    print("    (the K_φ cascade) is TRANSPORT — native to the new physics.")
    print("    But native ≠ solved: the analytic scale → 0 limit")
    print("    (NS-G_blowup / Clay) is still OPEN.")
    print()
    print("So the answer to 'is Navier–Stokes the same as Riemann?' is NO:")
    print("the obstacles live in different places (transport vs arithmetic),")
    print("and the new physics is native to one and foreign to the other.")
    print("Neither is thereby closed.")
    print()


def main():
    print()
    print("  TNFR Example 104: Navier–Stokes Is Not Riemann")
    print("  Native transport vs foreign arithmetic — neither solved")
    print("  =======================================================")
    print()
    pt_n = experiment_1_blind_vs_populated()
    experiment_2_physics_is_tetrad(pt_n)
    experiment_3_residuals()
    experiment_4_synthesis()
    print("=" * 72)
    print("WHAT THIS ESTABLISHES")
    print("=" * 72)
    print()
    print("Under the emergent substrate, Navier–Stokes and Riemann are NOT")
    print("the same. The substrate is BLIND to Riemann (prime content lives")
    print("in ν_f, which the tetrad does not read) but POPULATED for")
    print("Navier–Stokes (velocity IS the phase field; vorticity = K_φ,")
    print("pressure = Φ_s). The NS residual — the K_φ cascade at scale → 0 —")
    print("is a TRANSPORT object, native to the new physics (structural")
    print("diffusion, REMESH-∞ STRUCTURAL_EFFECT). The Riemann residual S(T)")
    print("is ARITHMETIC, orthogonal to it (REMESH-∞ DEGENERATE). The new")
    print("physics is native to one and foreign to the other. This is a")
    print("characterization, NOT a closure: 3D NS global regularity and")
    print("G4 = RH both remain OPEN.")
    print()


if __name__ == "__main__":
    main()
