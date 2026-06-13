#!/usr/bin/env python3
"""
Example 95 — Primes from Spectral Waves
========================================

Reconstructs the prime-counting staircase ψ(x) as a superposition of
spectral waves — one per Riemann zero — making tangible the duality

    prime positions  ⟷  spectral geometry {γ_n}.

Physics
-------
The Riemann–von Mangoldt explicit formula expresses the prime staircase
ψ(x) = Σ_{p^k ≤ x} log p as

    ψ(x) = x  −  Σ_n x^{ρ_n}/ρ_n  −  log(2π)  −  ½ log(1 − x^{-2})

with ρ_n = ½ + i·γ_n the non-trivial zeros. Each oscillatory term

    x^{ρ_n}/ρ_n = (√x / |ρ_n|) · exp(i·γ_n·log x − i·arg ρ_n)

is a WAVE of angular frequency γ_n in the coordinate log x and amplitude
√x/|ρ_n|. Primes appear where they do because their distribution is the
Fourier synthesis of these spectral waves: the "coherence flow ordering
itself" is literally the superposition of e^{i·γ_n·log x}.

Spectral coherence = Riemann Hypothesis
---------------------------------------
RH states Re(ρ_n) = ½ for EVERY zero. Then every wave has amplitude
√x/|ρ_n| — uniformly bounded — and the staircase stays coherent
(|ψ(x) − x| = O(√x log²x)). A single off-critical-line zero at
Re = ½ + δ would inject a wave of amplitude x^{½+δ} that grows faster
and DESTROYS the bound. So RH ⟺ "all spectral waves equally bounded"
⟺ the spectral geometry is maximally coherent.

Experiments
-----------
1. Staircase synthesis: ψ(x) reconstructed from N = 5, 20, 80 zeros
2. Wave decomposition: each zero as a bounded wave (√x/|ρ_n|, freq γ_n)
3. Coherence ⟺ RH: on-line (Re=½) stays √x-bounded; off-line (Re=0.7)
   error grows like x^{0.7} — incoherent

Honest scope
------------
This REPRODUCES the classical explicit formula (Riemann 1859, von
Mangoldt 1895); it is NOT a TNFR discovery. TNFR's contribution is to
realise the prime-side spectrum as a self-adjoint prime-ladder
Hamiltonian (P14, spectrum {k·log p}) and verify the identity to machine
precision (P15). The "spectral geometry" is the geometry of an operator
on the critical line — NOT physical space; no cosmological claim is made.

PROVING the geometry is coherent (all γ_n real ⟺ spectrum real ⟺ RH) is
the OPEN problem. TNFR formalises it as Conjecture T-HP and supplies
RH-equivalent coherence/positivity diagnostics (P16 Li–Keiper, P26
Lyapunov-spectral), but the residual oscillation S(T) = (1/π)·arg ζ(½+iT)
— visible here as the slow N^{-1/2} convergence — is the genuine
obstruction. The TNFR-Riemann program is PAUSED at this boundary; G4 = RH
remains open. Coincidence of names ("coherence", "sense") with the TNFR
metrics C(t), Si is suggestive but is NOT a proof: see the closing notes.

References
----------
- theory/TNFR_NUMBER_THEORY.md §10 (prime path graphs, Riemann link)
- theory/TNFR_RIEMANN_RESEARCH_NOTES.md §13septies (Conjecture T-HP)
- src/tnfr/riemann/analytic_continuation.py (P13 explicit formula)
- src/tnfr/riemann/prime_ladder_hamiltonian.py (P14 spectrum {k log p})
- AGENTS.md §"TNFR-Riemann Program Overview" (status, open G4)
"""

import os
import sys
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tnfr.riemann.analytic_continuation import (
    reconstruct_psi_via_explicit_formula,
    fetch_riemann_zeros,
)


# ============================================================================
# EXPERIMENT 1: Staircase synthesis from spectral waves
# ============================================================================
def experiment_1_staircase_synthesis():
    """Reconstruct ψ(x) from increasing numbers of spectral modes."""
    print("=" * 72)
    print("EXPERIMENT 1: Prime Staircase from Spectral Waves")
    print("=" * 72)
    print()
    print("ψ(x) = x − Σ_n x^{ρ_n}/ρ_n − log(2π) − ½log(1−x^{-2})")
    print("Each Riemann zero adds one wave. More zeros → sharper staircase.")
    print()

    xs = [10.5, 12.5, 16.5, 18.5, 22.5, 28.5]
    zeros80 = fetch_riemann_zeros(80)
    res = {n: reconstruct_psi_via_explicit_formula(xs, zeros=list(zeros80[:n]))
           for n in (5, 20, 80)}

    print(f"{'x':>6}  {'ψ_true':>9}  {'N=5':>8}  {'N=20':>8}  {'N=80':>8}")
    print("-" * 50)
    for i, x in enumerate(xs):
        print(f"{x:>6.1f}  {res[80].psi_classical[i]:>9.3f}  "
              f"{res[5].psi_explicit[i]:>8.3f}  "
              f"{res[20].psi_explicit[i]:>8.3f}  "
              f"{res[80].psi_explicit[i]:>8.3f}")

    print()
    for n in (5, 20, 80):
        max_err = float(max(res[n].abs_error))
        print(f"  N={n:>3} zeros → max reconstruction error = {max_err:.3f}")
    print()
    print("Error shrinks with more spectral modes (slowly, ~N^{-1/2} — the")
    print("explicit formula converges only conditionally). Primes ARE the")
    print("synthesis of these waves.")
    print()


# ============================================================================
# EXPERIMENT 2: Each zero is a bounded spectral wave
# ============================================================================
def experiment_2_wave_decomposition():
    """Show each zero as a wave: amplitude √x/|ρ|, frequency γ_n."""
    print("=" * 72)
    print("EXPERIMENT 2: Each Zero is a Spectral Wave")
    print("=" * 72)
    print()
    print("x^{ρ}/ρ = (√x/|ρ|)·exp(i·γ·log x − i·arg ρ)  with ρ = ½ + i·γ")
    print("→ amplitude √x/|ρ|, angular frequency γ in the log-x coordinate.")
    print()

    x = 100.5
    zeros = fetch_riemann_zeros(8)
    sqrt_x = math.sqrt(x)
    log_x = math.log(x)

    print(f"At x = {x}:  √x = {sqrt_x:.3f},  log x = {log_x:.4f}")
    print()
    print(f"{'n':>3}  {'γ_n':>9}  {'|ρ_n|':>8}  {'amplitude':>10}"
          f"  {'phase γ·logx':>12}")
    print("-" * 54)
    for n, rho in enumerate(zeros, start=1):
        gamma = float(rho.imag)
        mod = abs(complex(0.5, gamma))
        amplitude = 2.0 * sqrt_x / mod   # factor 2 from ρ + conjugate
        phase = (gamma * log_x) % (2 * math.pi)
        print(f"{n:>3}  {gamma:>9.4f}  {mod:>8.4f}  {amplitude:>10.4f}"
              f"  {phase:>12.4f}")

    print()
    print("Every amplitude scales as √x (NOT faster) because Re(ρ) = ½.")
    print("Lower zeros (small γ) carry the loudest waves; higher zeros add")
    print("fine detail. This is the spectral geometry of the primes.")
    print()


# ============================================================================
# EXPERIMENT 3: Spectral coherence ⟺ Riemann Hypothesis
# ============================================================================
def experiment_3_coherence_is_rh():
    """On-line zeros stay √x-bounded; off-line zeros grow like x^{Re}."""
    print("=" * 72)
    print("EXPERIMENT 3: Spectral Coherence ⟺ Riemann Hypothesis")
    print("=" * 72)
    print()
    print("RH: Re(ρ) = ½ for ALL zeros ⟹ every wave bounded by √x/|ρ| ⟹")
    print("coherent staircase. A hypothetical off-line zero (Re = 0.7)")
    print("injects a wave of amplitude x^{0.7} that breaks the bound.")
    print()

    xs = [20.5, 50.5, 100.5, 200.5]
    zeros = fetch_riemann_zeros(40)
    on = reconstruct_psi_via_explicit_formula(xs, zeros=list(zeros))
    off_zeros = [complex(0.7, r.imag) for r in zeros]
    off = reconstruct_psi_via_explicit_formula(xs, zeros=off_zeros)

    print(f"{'x':>7}  {'ψ_true':>9}  {'err ON (Re=½)':>13}"
          f"  {'err OFF (Re=.7)':>15}  {'√x':>7}  {'x^0.7':>8}")
    print("-" * 72)
    for i, x in enumerate(xs):
        print(f"{x:>7.1f}  {on.psi_classical[i]:>9.2f}  "
              f"{on.abs_error[i]:>13.3f}  {off.abs_error[i]:>15.3f}  "
              f"{x ** 0.5:>7.2f}  {x ** 0.7:>8.2f}")

    on_ratio = on.abs_error[-1] / on.abs_error[0]
    off_ratio = off.abs_error[-1] / off.abs_error[0]
    print()
    print(f"  Error growth (x: 20→200):  ON-line ×{on_ratio:.1f}"
          f"  vs  OFF-line ×{off_ratio:.1f}")
    print()
    print("On-line error tracks √x (coherent). Off-line error tracks x^{0.7}")
    print("(incoherent — diverges faster). RH ⟺ the spectral geometry is")
    print("maximally coherent: all waves share the same √x envelope.")
    print()


def main():
    print()
    print("  TNFR Example 95: Primes from Spectral Waves")
    print("  prime positions ⟷ spectral geometry {γ_n}")
    print("  ==========================================")
    print()

    experiment_1_staircase_synthesis()
    experiment_2_wave_decomposition()
    experiment_3_coherence_is_rh()

    print("=" * 72)
    print("HONEST SCOPE & THE COHERENCE QUESTION")
    print("=" * 72)
    print()
    print("This demo REPRODUCES the classical Riemann–von Mangoldt explicit")
    print("formula. TNFR realises the prime side as a self-adjoint")
    print("prime-ladder Hamiltonian (P14, spectrum {k·log p}) and verifies")
    print("the identity to machine precision (P15).")
    print()
    print("RH is literally a COHERENCE statement: the spectral geometry is")
    print("coherent ⟺ every γ_n is real ⟺ the operator is self-adjoint ⟺")
    print("all waves share the √x envelope (Experiment 3).")
    print()
    print("Is this what TNFR's C(t) / Sense Index (Si) can prove? The name")
    print("coincidence is suggestive but is NOT itself a proof:")
    print("  - C(t) = 1/(1 + mean|ΔNFR| + mean|dEPI|) and Si are metrics of")
    print("    a network's dynamic coherence, with specific formulas.")
    print("  - RH-coherence is the reality of an operator's spectrum.")
    print("  - Equating them is exactly Conjecture T-HP (§13septies): does a")
    print("    canonical operator built from the tetrad + (φ,γ,π,e) + U1-U6")
    print("    have spectrum {γ_n}? That would make RH a TNFR coherence")
    print("    theorem.")
    print("  - TNFR already pursued this: P16 (Li–Keiper positivity) and P26")
    print("    (Lyapunov-spectral positivity) are RH-equivalent coherence")
    print("    diagnostics. The residual S(T) = (1/π)·arg ζ(½+iT) — the slow")
    print("    N^{-1/2} convergence seen in Experiment 1 — is the genuine")
    print("    obstruction (lives in Fix(S_n)^⊥, unreachable by the current")
    print("    canonical constructions).")
    print()
    print("Verdict: the intuition 'coherence proves the geometry' names the")
    print("RIGHT program (T-HP), but the current C(t)/Si metrics do NOT close")
    print("it. G4 = RH remains open; the program is paused at this boundary.")
    print()


if __name__ == "__main__":
    main()
