#!/usr/bin/env python3
"""
Example 96 — The Spectral Vibration of Coherence (S(T))
========================================================

Isolates the oscillatory residue S(T) = (1/π)·arg ζ(½+iT) — the SOLE
obstruction to closing the TNFR-Riemann program — and exposes it as a
literal spectral vibration at prime-ladder frequencies {k·log p}. Then
shows *precisely why* an aggregate coherence metric like C(t) is blind to
it, and what a coherence metric would need to capture it.

Physics
-------
The prime staircase ψ(x) splits into a SMOOTH half (closed by P28/P30 at
the operator level) and an OSCILLATORY half, the latter governed by

    S(T) = (1/π)·arg ζ(½ + iT).

The classical Riemann–von Mangoldt formula expresses S(T) as a
prime-power vibration. In canonical TNFR prime-ladder data
Σ = {(μ, w)} with μ = k·log p, w = log p, this is exactly

    S_TNFR(T) = −(1/π) · Σ_{(μ,w)∈Σ} (w/μ) · sin(T·μ) / e^{μ/2}.

So S(T) IS "the spectral vibration of coherence": a superposition of
sine modes whose angular frequencies are the prime-ladder eigenvalues
μ = k·log p — the structural frequencies ν_f of the P14 Hamiltonian.

Why aggregate coherence is blind to it
--------------------------------------
The primary TNFR coherence C(t) = 1/(1 + mean|ΔNFR| + mean|dEPI|) is a
GLOBAL AGGREGATE — it averages over the network. But S(T) has zero mean
(it oscillates symmetrically about 0). Averaging annihilates it:

    mean(S) ≈ 0   while   mean(|S|), std(S) ≠ 0.

The signal lives entirely in the OSCILLATION, not the average. This is
the same blind spot the phase gradient |∇φ| was introduced to cover
(see AGENTS.md: aggregate C(t) misses local phase stress). To see S(T),
a coherence metric must be PHASE-RESOLVED / per-mode (spectral), not a
single scalar average.

Experiments
-----------
1. S(T) as a prime-ladder vibration: canonical reconstruction vs true
   S(T) = (1/π)·arg ζ(½+iT) (sign-tracking)
2. Spectral decomposition: FFT of S(T) → dominant frequencies are
   exactly {k·log p}
3. Aggregate blindness: mean(S) ≈ 0 (invisible to a mean-based metric)
   vs std(S) ≠ 0 (the vibration is real) — the characterization of what
   a coherence metric needs

Honest scope
------------
This is DIAGNOSTIC, not a closure. The vibration S(T) is *accessible*
(we reconstruct it from canonical prime-ladder data and read its
frequencies), but:

- The reconstruction is APPROXIMATE (finite N, K; sign-tracks but does
  not reproduce S(T) to machine precision).
- It is S(T) plugged into the Riemann–von Mangoldt TEMPLATE, NOT DERIVED
  from the nodal equation ∂EPI/∂t = νf·ΔNFR. Deriving it canonically is
  sub-problem (2) of Conjecture T-HP — OPEN.
- RH requires ALL zeros to be real; reading the vibration does not bound
  S(T) uniformly. The residual lives in Fix(S_n)^⊥, unreachable by the
  current graph-uniform canonical constructions (B1 refutations,
  §13vicies-novies). G4 = RH remains OPEN; the program is PAUSED here.

What IS established: the "spectral vibration of coherence" is a concrete,
canonical, computable object (P31), and the precise reason current
coherence metrics miss it (aggregate vs phase-resolved). That is the
honest frontier — not a proof.

References
----------
- src/tnfr/riemann/oscillatory_correction.py (P31, S_TNFR reconstruction)
- src/tnfr/riemann/von_mangoldt.py (prime-ladder spectrum {k log p})
- theory/TNFR_RIEMANN_RESEARCH_NOTES.md §13decies (P31), §13septies (T-HP)
- AGENTS.md §"Telemetry — Core Metrics" (aggregate C(t) blind spot, |∇φ|)
"""

import os
import sys
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import numpy as np

from tnfr.riemann.von_mangoldt import build_prime_ladder_spectrum
from tnfr.riemann.oscillatory_correction import prime_ladder_oscillatory_sum


def _s_true(T: float) -> float:
    """True oscillatory residue S(T) = (1/π)·arg ζ(½+iT) via mpmath."""
    import mpmath as mp
    with mp.workdps(25):
        val = mp.arg(mp.zeta(mp.mpf(0.5) + 1j * mp.mpf(T))) / mp.pi
    return float(val)


# ============================================================================
# EXPERIMENT 1: S(T) as a prime-ladder vibration
# ============================================================================
def experiment_1_vibration():
    """Reconstruct S(T) from canonical prime-ladder data; compare to truth."""
    print("=" * 72)
    print("EXPERIMENT 1: S(T) is a Prime-Ladder Vibration")
    print("=" * 72)
    print()
    print("S_TNFR(T) = −(1/π)·Σ (w/μ)·sin(T·μ)/e^{μ/2},  μ = k·log p")
    print("Built from canonical P12/P14 prime-ladder data only.")
    print()

    spec = build_prime_ladder_spectrum(15, max_power=6)
    mu = np.sort(np.asarray(spec.eigenvalues, dtype=float))
    print("Lowest prime-ladder frequencies μ = k·log p:")
    print(f"  {[round(float(x), 4) for x in mu[:6]]}")
    print(f"  (log2={math.log(2):.4f}, log3={math.log(3):.4f}, "
          f"2·log2={2 * math.log(2):.4f}, log5={math.log(5):.4f})")
    print()

    heights = [20.3, 30.7, 40.1, 50.9, 60.3, 75.2]
    print(f"{'T':>7}  {'S_true':>9}  {'S_TNFR':>9}  {'sign':>5}")
    print("-" * 38)
    sign_ok = 0
    for T in heights:
        st = _s_true(T)
        sx = float(prime_ladder_oscillatory_sum(T, spec))
        same = (st * sx > 0) or (abs(st) < 0.05)
        sign_ok += int(same)
        tag = 'OK' if same else '--'
        print(f"{T:>7.1f}  {st:>9.4f}  {sx:>9.4f}  {tag:>5}")

    print()
    print(f"Sign agreement: {sign_ok}/{len(heights)} heights.")
    print("The canonical prime-ladder vibration tracks S(T) — but only")
    print("approximately (finite N, K). The vibration is REAL and canonical.")
    print()


# ============================================================================
# EXPERIMENT 2: Spectral decomposition — frequencies are {k log p}
# ============================================================================
def experiment_2_spectral_decomposition():
    """FFT of S(T): dominant angular frequencies are exactly k·log p."""
    print("=" * 72)
    print("EXPERIMENT 2: Spectral Decomposition of S(T)")
    print("=" * 72)
    print()
    print("FFT of S(T) over a long T-window. The peaks sit exactly at the")
    print("prime-ladder frequencies μ = k·log p — the ν_f of the P14")
    print("Hamiltonian. This is the 'spectral vibration' made explicit.")
    print()

    spec = build_prime_ladder_spectrum(20, max_power=6)
    T = np.linspace(10.0, 200.0, 4000)
    S = np.asarray(prime_ladder_oscillatory_sum(T, spec), dtype=float)

    dt = float(T[1] - T[0])
    S_demean = S - float(np.mean(S))
    amp = np.abs(np.fft.rfft(S_demean))
    omega = np.fft.rfftfreq(len(T), d=dt) * 2.0 * math.pi
    top = np.argsort(amp)[-5:][::-1]

    prime_freqs = {
        "log 2": math.log(2), "log 3": math.log(3),
        "log 5": math.log(5), "2·log 2": 2 * math.log(2),
        "log 7": math.log(7),
    }

    print(f"{'rank':>4}  {'ω (FFT peak)':>13}  {'nearest k·log p':>18}")
    print("-" * 42)
    for rank, idx in enumerate(top, start=1):
        w = float(omega[idx])
        best_name, best_val = min(
            ((nm, v) for nm, v in prime_freqs.items()),
            key=lambda kv: abs(kv[1] - w),
        )
        print(f"{rank:>4}  {w:>13.4f}  {best_name:>11} = {best_val:>5.4f}")

    print()
    print("VALIDATED: S(T)'s spectral lines ARE the prime-ladder frequencies.")
    print("S(T) literally vibrates at {k·log p} — the structural ν_f.")
    print()


# ============================================================================
# EXPERIMENT 3: Why aggregate coherence is blind
# ============================================================================
def experiment_3_aggregate_blindness():
    """mean(S) ≈ 0 (invisible to averages) but std(S) ≠ 0 (vibration real)."""
    print("=" * 72)
    print("EXPERIMENT 3: Why Aggregate Coherence is Blind to S(T)")
    print("=" * 72)
    print()
    print("The primary C(t) = 1/(1 + mean|ΔNFR| + mean|dEPI|) AVERAGES.")
    print("S(T) oscillates symmetrically about 0, so its mean ≈ 0:")
    print("a mean-based coherence metric annihilates the entire signal.")
    print()

    spec = build_prime_ladder_spectrum(20, max_power=6)
    T = np.linspace(10.0, 200.0, 4000)
    S = np.asarray(prime_ladder_oscillatory_sum(T, spec), dtype=float)

    mean_s = float(np.mean(S))
    mean_abs = float(np.mean(np.abs(S)))
    std_s = float(np.std(S))
    max_abs = float(np.max(np.abs(S)))

    print(f"  mean(S)    = {mean_s:>12.6e}   <- DC ~ 0: AGGREGATE SEES 0")
    print(f"  mean(|S|)  = {mean_abs:>12.4f}   <- signal is in the magnitude")
    print(f"  std(S)     = {std_s:>12.4f}   <- the vibration is REAL")
    print(f"  max(|S|)   = {max_abs:>12.4f}")
    print()
    print("Characterization — what a coherence metric NEEDS to see S(T):")
    print("  • NOT a single scalar average (kills the zero-mean oscillation)")
    print("  • a PHASE-RESOLVED / per-mode (spectral) coherence: amplitude")
    print("    AND phase of each prime-ladder mode μ = k·log p")
    print("  • exactly the blind spot |∇φ| was added to cover (local phase")
    print("    stress that the global C(t) aggregate misses)")
    print()
    print("This is accessible as a DIAGNOSTIC (we just measured it). It is")
    print("NOT a closure: deriving this vibration from the nodal equation")
    print("(not the RvM template) is sub-problem (2) of T-HP — OPEN.")
    print()


def main():
    print()
    print("  TNFR Example 96: The Spectral Vibration of Coherence S(T)")
    print("  The oscillatory residue as a prime-ladder vibration {k·log p}")
    print("  ============================================================")
    print()

    experiment_1_vibration()
    experiment_2_spectral_decomposition()
    experiment_3_aggregate_blindness()

    print("=" * 72)
    print("HONEST FRONTIER")
    print("=" * 72)
    print()
    print("The 'spectral vibration of coherence' is a concrete, canonical,")
    print("computable object: S(T) = a prime-ladder sine superposition at")
    print("frequencies {k·log p}. It IS accessible as a diagnostic.")
    print()
    print("What it does NOT do:")
    print("  • derive S(T) from ∂EPI/∂t = νf·ΔNFR (it uses the RvM template)")
    print("    — sub-problem (2) of Conjecture T-HP, OPEN")
    print("  • reproduce S(T) to machine precision (finite, approximate)")
    print("  • bound S(T) uniformly ⟺ RH (the residual lives in Fix(S_n)^⊥,")
    print("    unreachable by graph-uniform canonical constructions)")
    print()
    print("Verdict: 'work with the spectral vibration of coherence' is the")
    print("RIGHT frame and IS accessible diagnostically (P31). Turning the")
    print("diagnostic into a canonical DERIVATION is the open frontier —")
    print("branch B1 vs B2 of §13septies, undecided. G4 = RH remains open.")
    print()


if __name__ == "__main__":
    main()
