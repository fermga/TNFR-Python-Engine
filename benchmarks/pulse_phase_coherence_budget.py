"""Pulse-phase coherence budget: how tightly the nodal pulse auto-bounds S(T).

THE PARADIGM (re-founded Riemann program, 2026-07): the RH-content oscillation
S(T) = (1/pi) arg zeta(1/2+iT) is the collective PHASE of the integer-NFR nodal
pulse P(T) = sum_n n^{-1/2} e^{-i(log n)T}.  The paused self-adjoint program was
provably blind to S(T) (Fix(S_n)^perp); the pulse ACCESSES it (see
src/tnfr/riemann/pulse_coherence.py, examples/03_riemann_zeta/157_*).  This
benchmark asks a measure-first question with NO pre-judged answer: does the
pulse PHASE show auto-bounding structure -- a "coherence budget" (the numerical
face of the U2 boundedness requirement int nu_f dNFR dt < infinity)?

THE STRUCTURAL FACTS (all canonical / definitional, nothing invented):
  - S(T) is computed EXACTLY by continuous variation of arg zeta(sigma+iT) as
    sigma descends from +inf (arg=0) to 1/2 -- the definition of S(T).  No RH
    assumption, no zeros used as input.
  - Self-consistency: N(T) = theta(T)/pi + 1 + S(T) must equal the integer
    number of zeros below T (Riemann-von Mangoldt); theta is the SAME
    Riemann-Siegel gamma kernel shipped in structural_zero_density (P28).
  - The U2 reading: RH <=> the prime-pulse NFR stays coherent (U2-bounded); an
    off-line zero would be a U2 runaway.  So "how big can the pulse phase get?"
    is the TNFR-native form of the RH obstruction.

WHAT EMERGES (measured; T up to 3000, exact zeta):
  - M1 THE PHASE IS EXACT: theta/pi + 1 + S(T) reproduces the integer zero-count
    (1, 6, 19, ... = known zeros below T) with no RH input -- S(T) is the true
    fluctuation, and the shipped truncated pulse phase agrees off zeros.
  - M2 THE COHERENCE BUDGET IS REAL AND TIGHT: over TWO decades in T the RMS of
    S(T) grows only ~0.32 -> ~0.39, the glacial sqrt(log log T) of Selberg;
    mean(S) ~ 0 (centred, no drift).  The pulse phase does NOT run away.
  - M3 HUGE SLACK TO THE WALL: the measured peaks max|S| stay ~1, i.e. ~1/4 of
    the unconditional |S(T)| = O(log T) envelope (~4 at these heights).
  - The Selberg variance fit RMS^2 ~ a*log log T gives a ~ 0.066, the same order
    as 1/(2 pi^2) = 0.051 (finite-height excess; the log log T range is tiny).

HONEST SCOPE: the sqrt(log log T) tightness IS Selberg's theorem -- a classical,
UNCONDITIONAL result, consistent with RH but NOT implying it.  RH lives in the
EXTREMES / the exact count identity, not the typical size: unconditionally the
peaks are Omega(sqrt(log T / log log T)) (grow without bound, very slowly), so no
finite measurement excludes a large excursion at astronomical height.  The
measurement CONFIRMS the U2 coherence budget at the RMS/typical level and
RELOCATES the obstruction sharply to "lift the budget from the RMS level to the
supremum level" (control the PEAKS of S(T)).  It closes NOTHING; G4 = RH stays
OPEN.  The Selberg constant and the Trudgian envelope are EXTERNAL analytic
number-theory references used only as comparison anchors (not TNFR primitives).

Run:
    PYTHONPATH=src python benchmarks/pulse_phase_coherence_budget.py

Theoretical anchor: src/tnfr/riemann/pulse_coherence.py (S(T) as pulse phase),
src/tnfr/riemann/structural_zero_density.py (riemann_siegel_theta, P28),
theory/TNFR_RIEMANN_RESEARCH_NOTES.md (re-founding + coherence-budget note).
Status: RESEARCH.
"""

import math
import os
import sys
import time

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tnfr.riemann.nodal_pulse import KNOWN_RIEMANN_ZEROS  # noqa: E402
from tnfr.riemann.pulse_coherence import argument_fluctuation  # noqa: E402
from tnfr.riemann.structural_zero_density import (  # noqa: E402
    riemann_siegel_theta,
)

# --- external comparison anchors (classical analytic number theory) ---
SELBERG_SLOPE = 1.0 / (2.0 * math.pi**2)  # Var S(T) ~ SELBERG_SLOPE * log log T
# Trudgian (2014) unconditional bound: |S(T)| <= 0.112 log T + 0.278 llt + 2.510
ENV_A, ENV_B, ENV_C = 0.112, 0.278, 2.510

# --- measurement grid ---
CENTRES = (30.0, 100.0, 300.0, 1000.0, 3000.0)
WIDTH, STEP = 20.0, 0.2
SIGMA_HI, N_SIGMA, DPS = 4.0, 40, 15


def s_exact(t, sigma_hi=SIGMA_HI, n=N_SIGMA, dps=DPS):
    """S(T) = (1/pi) * continuous-arg zeta(1/2+iT), by descent in sigma.

    Continuous variation of arg from sigma=+inf (arg=0) to 1/2 -- the definition
    of S(T).  Returns None if a sample lands on a zero (arg undefined).
    """
    mp.mp.dps = dps
    sig = np.linspace(sigma_hi, 0.5, n)
    prev = float(mp.arg(mp.zeta(complex(sig[0], t))))
    total = prev
    for s in sig[1:]:
        z = mp.zeta(complex(s, t))
        if abs(z) < 1e-9:
            return None
        ang = float(mp.arg(z))
        d = ang - prev
        while d > math.pi:
            d -= 2.0 * math.pi
        while d <= -math.pi:
            d += 2.0 * math.pi
        total += d
        prev = ang
    return total / math.pi


def selberg_sigma(t):
    """Selberg RMS prediction (1/pi) sqrt(log log T / 2)."""
    return (1.0 / math.pi) * math.sqrt(0.5 * math.log(math.log(t)))


def envelope(t):
    """Unconditional (Trudgian 2014) envelope on |S(T)|."""
    return ENV_A * math.log(t) + ENV_B * math.log(math.log(t)) + ENV_C


def m1_phase_is_exact():
    print("\nM1 -- the pulse phase is exact (no RH input): "
          "theta/pi + 1 + S(T) == integer zero-count")
    for t in (20.0, 40.0, 76.0):
        s = s_exact(t)
        n_of_t = riemann_siegel_theta(t) / math.pi + 1.0 + s
        known = sum(1 for g in KNOWN_RIEMANN_ZEROS if g < t)
        trunc = argument_fluctuation(t)  # shipped truncated pulse phase
        print(f"   T={t:5.1f}: S_exact={s:+.4f}  N=theta/pi+1+S={n_of_t:6.3f}"
              f"  known<T={known:2d}  |  shipped pulse S={trunc:+.4f}"
              f"  (dS={abs(trunc - s):.3f})")


def m2_m3_coherence_budget():
    print("\nM2/M3 -- the coherence budget (RMS vs Selberg) and slack to the "
          "wall:")
    print("   T_c |  mean(S) | RMS(S) | Selberg | max|S| | envelope | peak/env")
    llt, rms2 = [], []
    for tc in CENTRES:
        ts = np.arange(tc - WIDTH / 2, tc + WIDTH / 2, STEP)
        vals = [s for s in (s_exact(float(t)) for t in ts) if s is not None]
        a = np.array(vals)
        rms = float(np.sqrt(np.mean(a**2)))
        mx = float(np.max(np.abs(a)))
        env = envelope(tc)
        print(f"   {tc:5.0f} | {np.mean(a):+7.3f} | {rms:6.3f} |"
              f" {selberg_sigma(tc):7.3f} | {mx:6.3f} | {env:8.3f} |"
              f" {mx / env:7.2f}")
        llt.append(math.log(math.log(tc)))
        rms2.append(rms**2)
    mat = np.vstack([np.array(llt), np.ones(len(llt))]).T
    slope, intercept = np.linalg.lstsq(mat, np.array(rms2), rcond=None)[0]
    print(f"\n   Selberg fit: RMS^2 = {slope:.4f}*log log T + {intercept:+.4f}"
          f"   (1/(2 pi^2) = {SELBERG_SLOPE:.4f})")
    print("   (max|S| slightly UNDER-estimates the true peak: step 0.2 "
          "under-samples the zero-spaced peaks.)")


def main() -> None:
    t0 = time.time()
    print("=" * 72)
    print("PULSE-PHASE COHERENCE BUDGET -- how tightly S(T) auto-bounds")
    print("=" * 72)
    m1_phase_is_exact()
    m2_m3_coherence_budget()
    print("\n" + "=" * 72)
    print(
        "VERDICT: the pulse phase is auto-bounded at the RMS/typical level\n"
        "(sqrt(log log T), Selberg) -- the numerical face of the U2 coherence\n"
        "budget, measured.  But that is a known UNCONDITIONAL result; RH lives\n"
        "in the EXTREMES.  The obstruction is RELOCATED, not removed: lift the\n"
        "budget from the RMS level to the supremum (control the peaks of S(T)).\n"
        "Closes nothing; G4 = RH stays OPEN."
    )
    print("=" * 72)
    print(f"[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
