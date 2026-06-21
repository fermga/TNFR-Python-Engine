"""
benchmarks/primes_as_consequence.py

Camino 11 -- are the primes a PRIMITIVE INPUT to TNFR, or a CONSEQUENCE of its
structure and dynamics? The change of optic the user asked for, made falsifiable.

Every earlier Camino fed the primes IN: nu_f = log p is the adelic carrier
(src/tnfr/dynamics/adelic.py), read by the nodal equation dEPI/dt = nu_f . dNFR
as IMPOSED input (the C5-C7 audit grounded the S_n-breaking diagonal in exactly
this carrier). This harness asks the opposite question: can "primality" be READ
OUT of TNFR structural equilibrium WITHOUT being fed in? AGENTS.md and
theory/TNFR_NUMBER_THEORY.md S4 already answer "primality = structural
equilibrium dNFR = 0" -- but that statement splits into two very different
regimes, and only one of them is a genuine derivation. This harness separates
them with running code.

THE TNFR MEANING (canonical, theory/TNFR_NUMBER_THEORY.md S3.3-S4):
  A natural number n carries structural pressure
      dNFR(n) = zeta.(Omega(n) - 1) + eta.(tau(n) - 2)
                + theta.(sigma(n)/n - (1 + 1/n))
  with coefficients zeta = phi.gamma, eta = (gamma/phi).pi, theta = 1/phi
  (notational combos approximating empirical values; audit 2026: not derived). For a prime p:
  Omega(p) = 1, tau(p) = 2, sigma(p)/p = 1 + 1/p, so ALL THREE terms vanish:
      n is prime  <=>  dNFR(n) = 0.
  So a prime is a ZERO-PRESSURE structural equilibrium -- the optic-shift is real
  and it does sharpen understanding: primes are not "atoms", they are the nodes
  where structural reorganization pressure vanishes (maximal coherence).

TWO READINGS OF "dNFR = 0" (the distinction the meaning alone blurs):

  READING A -- RE-DESCRIPTION (exact, but circular as a derivation).
    Compute dNFR(n) from Omega, tau, sigma. It is exactly 0 iff n is prime -- a
    theorem. BUT Omega, tau, sigma are obtained by trial division (n % i): you
    must ALREADY know the factorization to evaluate the pressure. As a *meaning*
    this is faithful; as a *derivation of primality from structure* it is
    circular (it consumes the divisibility it claims to explain). This is exactly
    what composition_arithmetic.py records: the primality module "CONSUMES
    divisibility (trial division n % i) to re-read primality as dNFR = 0."

  READING B -- EMERGENCE (genuinely non-circular, but partial + scale-sector).
    Build the residue circulant of n from quadratic residues mod n (squares
    x*x % n) -- this NEVER computes n % k for candidate factors k, so it never
    trial-divides n. Read primality off a SPECTRAL equilibrium: the Paley gap
    g(n) = |lambda_2(residue circulant) - (n - sqrt n)/2| = 0 occurs exactly at
    primes n == 1 (mod 4) (Gauss-sum identity, Zenodo 10.5281/zenodo.17665853;
    canonical P25 src/tnfr/riemann/paley_gap_coercivity.py, reused via
    paley_bridge.py / Camino 9). Here primality genuinely COMES OUT of the
    self-adjoint spectrum -- primes-OUT, not primes-IN. But it is PARTIAL (only
    the == 1 (mod 4) class; misses 2 and the == 3 (mod 4) primes) and REAL/self-
    adjoint (the Camino-8 scale sector: it reaches the support, never the phase).

  FRONTIER -- irreducibility is NOT primality (composition_arithmetic.py).
    A tempting third reading -- "primes = irreducible representation modes" -- is
    REFUTED: the dim-4 mode of K5 (Aut = S5) is irreducible yet 4 = 2 x 2. So the
    representation-theoretic optic does not reproduce arithmetic primality either;
    it bounds how far pure emergence can go.

THE CLAIM (what the optic-shift buys, and where it stops):
  understood:  primality's TNFR meaning is dNFR = 0 (zero structural pressure);
               and there IS a non-circular spectral emergence (Reading B) -- so
               the primes are, in part, a CONSEQUENCE of self-adjoint structure,
               not a primitive. The user's reframing is correct and productive.
  unreached:   making ALL primes emerge non-circularly (not just == 1 (mod 4))
               AND reaching the continuous phase S(T) = (1/pi) arg zeta(1/2 + iT)
               is the SAME e-pi / Fix(G)^perp wall as Caminos 5-10. Reading A is
               exact-but-circular; Reading B is non-circular-but-partial; neither
               derives every prime through the phase. The optic-shift LOCATES the
               residual precisely; it does not dissolve it.

ENGINE (independent ground truth, all pre-TNFR):
  - n prime <=> dNFR(n) = 0 is algebra once Omega, tau, sigma are known (theory
    S4); trial division supplies them (the circularity we measure, not hide).
  - Quadratic Gauss sum |sum_x exp(2 pi i x^2 / n)| = sqrt(n) for prime n; the
    residue circulant on prime n == 1 (mod 4) is the Paley graph with Laplacian
    spectrum {0, (n - sqrt n)/2, (n + sqrt n)/2}, so g(n) = 0 <=> prime == 1 mod 4.
  - Schur's lemma: <chi, chi> = 1 marks an irreducible mode; a dim-4 irreducible
    mode (K5) shows representational irreducibility != arithmetic primality.

HONEST SCOPE -- structural CHECKS pass; the THESIS verdict is OPEN (by design):
  We show at machine precision that (A) dNFR(n) = 0 reproduces the primes exactly
  but is computed by consuming the factorization (circular as a derivation);
  (B) g(n) = 0 reproduces the primes == 1 (mod 4) using only squares mod n -- a
  genuine non-circular spectral emergence -- but partial and self-adjoint;
  (C) irreducibility != primality. So the optic-shift converts the IMPOSED adelic
  carrier into a PARTIALLY EMERGENT one and pins the residual at the phase /
  the == 3 (mod 4) class. It SHARPENS the picture; it does not close G4 / RH.
  R (continuum) and phi, gamma, pi, e remain assumed substrate.

Run:
    python benchmarks/primes_as_consequence.py

Status: RESEARCH (primes-out-vs-in falsifier; Camino 11 of the unification map).
"""
from __future__ import annotations

import math
import os
import sys

import networkx as nx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Robust fallback so the harness also runs without PYTHONPATH=src preset.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from composition_arithmetic import (  # noqa: E402
    automorphism_matrices,
    character_norm,
    eigenspaces,
)
# Reading B reuses the canonical Paley machinery (Camino 9, Zenodo 17665853).
from paley_bridge import is_prime, paley_gap, _GAP_EPS  # noqa: E402

# Optional: the canonical TNFR primality pressure dNFR(n) (Reading A).
try:  # pragma: no cover - exercised only when the package is importable
    from tnfr_primality.core import tnfr_delta_nfr as _canon_delta_nfr  # noqa: E402
    _HAVE_PRIMALITY = True
except Exception:  # pragma: no cover
    _HAVE_PRIMALITY = False

# Optional: the canonical adelic carrier nu_f = log p -- the "primes-IN" input
# that this Camino questions (and that the C5-C7 audit grounded as IMPOSED).
try:  # pragma: no cover - exercised only when the package is importable
    from tnfr.dynamics.adelic import AdelicDynamics  # noqa: E402
    _HAVE_ADELIC = True
except Exception:  # pragma: no cover
    _HAVE_ADELIC = False

# Optional: the canonical P25 Paley-gap module (its honest scope: "does not close
# G4 / not a primality proof") -- the canonical home of Reading B.
try:  # pragma: no cover
    from tnfr.riemann import paley_gap_coercivity as _canon_paley  # noqa: E402
    _HAVE_CANON_PALEY = True
except Exception:  # pragma: no cover
    _HAVE_CANON_PALEY = False

TOL = 1e-9
_DNFR_EPS = 1e-9                  # dNFR below this counts as a zero-pressure prime

# Pressure coefficients (notational (phi,gamma,pi,e) combos; audit 2026: not derived).
PHI = (1.0 + math.sqrt(5.0)) / 2.0
GAMMA = 0.5772156649015329
PI = math.pi
ZETA = PHI * GAMMA               # factorization pressure  ~ 0.9340
ETA = (GAMMA / PHI) * PI         # divisor pressure         ~ 1.1207
THETA = 1.0 / PHI                # abundance pressure       ~ 0.6180


# --------------------------------------------------------------------------- #
# Reading A: dNFR(n) -- exact, but consumes the factorization (circular).
# --------------------------------------------------------------------------- #
def factorization_with_opcount(n: int) -> tuple[int, int, int, int]:
    """Return (Omega, tau, sigma, n_trial_divisions) for n, by trial division.

    The trial-division count makes Reading A's circularity a NUMBER, not a vibe:
    every n % d below is a divisibility query the pressure equation consumes.
    """
    if n <= 1:
        return 0, 0, 0, 0
    omega = 0          # prime-factor count with multiplicity (big Omega)
    tau = 0            # number of divisors
    sigma = 0          # sum of divisors
    ops = 0
    # Omega via factor extraction.
    d = 2
    temp = n
    while d * d <= temp:
        while True:
            ops += 1
            if temp % d != 0:
                break
            omega += 1
            temp //= d
        d += 1
    if temp > 1:
        omega += 1
    # tau, sigma via divisor scan.
    i = 1
    while i * i <= n:
        ops += 1
        if n % i == 0:
            tau += 1
            sigma += i
            j = n // i
            if j != i:
                tau += 1
                sigma += j
        i += 1
    return omega, tau, sigma, ops


def delta_nfr_pressure(n: int) -> tuple[float, int]:
    """TNFR arithmetic pressure dNFR(n) and the trial-division cost it consumed.

    Prefers the canonical tnfr_primality.core implementation for the pressure
    value; the op-count is computed locally so the circularity is visible even
    when the canonical module is present.
    """
    omega, tau, sigma, ops = factorization_with_opcount(n)
    if _HAVE_PRIMALITY:
        return float(_canon_delta_nfr(n)), ops
    pressure = (
        ZETA * (omega - 1)
        + ETA * (tau - 2)
        + THETA * (sigma / n - (1.0 + 1.0 / n))
    )
    return float(pressure), ops


# --------------------------------------------------------------------------- #
# TEST A -- dNFR = 0 reproduces the primes EXACTLY, but is circular (primes-IN).
# --------------------------------------------------------------------------- #
def test_reading_a_redescription(limit: int = 200) -> bool:
    print("=" * 78)
    print("READING A -- primality as dNFR = 0 (exact structural meaning,")
    print("             but a RE-DESCRIPTION: it CONSUMES the factorization)")
    print("=" * 78)

    primes = [n for n in range(2, limit + 1) if is_prime(n)]
    zero_pressure = []
    total_ops = 0
    for n in range(2, limit + 1):
        pressure, ops = delta_nfr_pressure(n)
        total_ops += ops
        if abs(pressure) <= _DNFR_EPS:
            zero_pressure.append(n)

    extra = sorted(set(zero_pressure) - set(primes))    # composites called prime
    miss = sorted(set(primes) - set(zero_pressure))     # primes missed
    exact = (not extra) and (not miss)

    src = "canonical tnfr_primality.core" if _HAVE_PRIMALITY else "inline fallback"
    print(f"  pressure source : {src} (coeffs zeta=phi.gamma, eta=(gamma/phi).pi,")
    print("                    theta=1/phi -- notational, not derived)")
    print(f"  range           : n = 2..{limit}")
    print(f"  dNFR(n) = 0 set == primes ?  exact = {exact} "
          f"(extra = {extra}, missed = {miss})")
    print(f"  trial divisions consumed to evaluate dNFR over the range: {total_ops}")
    print("  => EXACT structural meaning (prime = zero-pressure equilibrium), but")
    print("     dNFR is computed FROM Omega, tau, sigma, each obtained by n % d.")
    print("     As a derivation of primality this is CIRCULAR: primes go IN")
    print("     (consumed as divisibility) and come back out re-labelled as dNFR=0.")
    print(f"  VERDICT: {'PASS' if exact else 'FAIL'} "
          "-- faithful re-description, NOT a from-structure derivation")
    return exact


# --------------------------------------------------------------------------- #
# TEST B -- g(n) = 0 reproduces the primes == 1 (mod 4) with NO trial division.
# --------------------------------------------------------------------------- #
def test_reading_b_emergence(limit: int = 200) -> bool:
    print()
    print("=" * 78)
    print("READING B -- primality as a SPECTRAL equilibrium g(n) = 0")
    print("             (genuine emergence: primes-OUT, no n % k consumed)")
    print("=" * 78)

    candidates = [m for m in range(5, limit + 1) if m % 4 == 1]
    primes14 = [m for m in candidates if is_prime(m)]
    zeros = [m for m in candidates if paley_gap(m) <= _GAP_EPS]

    extra = sorted(set(zeros) - set(primes14))
    miss = sorted(set(primes14) - set(zeros))
    exact = (not extra) and (not miss)

    # The detector consumes only squares mod n (x*x % n), never n % k: it asks
    # for the SHAPE of n's residue spectrum, never whether a candidate divides n.
    missed_classes = sorted({2} | {p for p in range(3, 40)
                                   if is_prime(p) and p % 4 == 3})
    print("  detector        : g(n) = |lambda_2(residue circulant) - (n-sqrt n)/2|")
    print("                    built from quadratic residues x*x % n (mod the")
    print("                    candidate itself) -- it NEVER computes n % k.")
    if _HAVE_CANON_PALEY:
        p25 = f"present ({_canon_paley.__name__}, 'does not close G4')"
    else:
        p25 = "absent (reusing paley_bridge / Camino 9 machinery)"
    print(f"  canonical P25   : {p25}")
    print(f"  range           : n == 1 (mod 4), n = 5..{limit}")
    print(f"  g(n) = 0 set == primes == 1 (mod 4) ?  exact = {exact} "
          f"(extra = {extra}, missed = {miss})")
    print(f"  genuinely emergent (no n % k): True ; primes-OUT count = {len(zeros)}")
    print("  HONEST PARTIALITY: the detector is blind to 2 and to the == 3 (mod 4)")
    print(f"                    primes (e.g. {missed_classes[:8]}...) -- they live")
    print("                    outside the Paley == 1 (mod 4) class. And the residue")
    print("                    circulant is symmetric => REAL spectrum => scale")
    print("                    sector (Camino 8): reaches the support, not S(T).")
    print(f"  VERDICT: {'PASS' if exact else 'FAIL'} "
          "-- non-circular emergence, but PARTIAL and self-adjoint")
    return exact


# --------------------------------------------------------------------------- #
# FRONTIER -- representational irreducibility is NOT arithmetic primality.
# --------------------------------------------------------------------------- #
def test_frontier_irreducibility(verbose: bool = True) -> bool:
    print()
    print("=" * 78)
    print("FRONTIER -- 'primes = irreducible modes' is REFUTED")
    print("            (irreducibility is a property of the SYSTEM, not the integer)")
    print("=" * 78)

    K5 = nx.complete_graph(5)
    nodes5 = list(K5.nodes())
    mats5 = automorphism_matrices(K5, nodes5)
    order5 = len(mats5)
    four_irreducible = False
    for val, mult, P in eigenspaces(K5, nodes5):
        if mult == 4:
            chi = character_norm(P, mats5, order5)
            four_irreducible = abs(chi - 1.0) < 0.4
            if verbose:
                print(f"  K5 (Aut = S5, |Aut| = {order5}): dim-4 mode <chi,chi> = "
                      f"{chi:.2f} -> {'IRREDUCIBLE' if four_irreducible else '?'}")
    print("  yet 4 = 2 x 2 arithmetically: the same cardinal is atomic in K5 and")
    print("  compositional in K3 [] K3. So irreducibility (physics) != primality")
    print("  (arithmetic). Pure emergence via representation theory cannot, by")
    print("  itself, reproduce unique factorisation.")
    print(f"  VERDICT: {'PASS' if four_irreducible else 'FAIL'} "
          "-- 'prime <=> irreducible' correctly refuted")
    return four_irreducible


# --------------------------------------------------------------------------- #
# BRIDGE -- primes-IN (adelic carrier) vs primes-OUT (spectral emergence).
# --------------------------------------------------------------------------- #
def test_bridge_in_vs_out(limit: int = 60) -> bool:
    print()
    print("=" * 78)
    print("BRIDGE -- the adelic carrier nu_f = log p reads primes IN;")
    print("          Reading B reads (some) primes OUT. Where is the residual?")
    print("=" * 78)

    # primes-IN: the carrier the whole programme (and the C5-C7 audit) imposes.
    if _HAVE_ADELIC:
        eng = AdelicDynamics(max_prime=max(30, limit))
        carrier_primes = [int(p) for p in np.asarray(eng.primes) if p <= limit]
        src = "canonical AdelicDynamics.nu_f"
    else:
        carrier_primes = [n for n in range(2, limit + 1) if is_prime(n)]
        src = "sieve fallback"
    # primes-OUT: the non-circular spectral emergence (== 1 (mod 4) only).
    emergent = [m for m in range(5, limit + 1)
                if m % 4 == 1 and paley_gap(m) <= _GAP_EPS]

    residual = sorted(set(carrier_primes) - set(emergent))
    covered = sorted(set(carrier_primes) & set(emergent))
    print(f"  primes IN  (carrier, {src}): {carrier_primes}")
    print(f"  primes OUT (spectral emergence, == 1 mod 4): {emergent}")
    print(f"  covered by emergence: {covered}")
    print(f"  RESIDUAL (imposed, not yet emergent): {residual}")
    print("  => the optic-shift converts the IMPOSED carrier into a PARTIALLY")
    print("     EMERGENT one. The residual (2, the == 3 (mod 4) primes, and the")
    print("     continuous phase S(T)) is the SAME wall as Caminos 5-10:")
    print("     real/self-adjoint structure reaches the support, not the phase.")
    # Structural check: emergence is a strict, correct SUBSET of the carrier.
    is_subset = set(emergent).issubset(set(carrier_primes))
    print(f"  VERDICT: {'PASS' if is_subset else 'FAIL'} -- emergence is a correct "
          "(partial) subset of the carrier; full emergence stays OPEN")
    return is_subset


def main() -> int:
    print(__doc__)
    ra = test_reading_a_redescription()
    rb = test_reading_b_emergence()
    rf = test_frontier_irreducibility()
    rbr = test_bridge_in_vs_out()

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  Reading A: dNFR = 0 exact (re-description, circular)  : "
          f"{'PASS' if ra else 'FAIL'}")
    print(f"  Reading B: g(n) = 0 emergent (non-circular, partial)  : "
          f"{'PASS' if rb else 'FAIL'}")
    print(f"  Frontier : irreducibility != primality                : "
          f"{'PASS' if rf else 'FAIL'}")
    print(f"  Bridge   : emergence subset of imposed carrier        : "
          f"{'PASS' if rbr else 'FAIL'}")
    structural = all([ra, rb, rf, rbr])
    print(f"\n  STRUCTURAL CHECKS: {'ALL PASS' if structural else 'SOME FAILED'}")
    print("  THESIS VERDICT: PARTIAL / OPEN (by design)")
    print()
    print("  Reading: the optic-shift is REAL and clarifying -- in TNFR a prime is")
    print("  a zero-pressure structural equilibrium (dNFR = 0), not a primitive")
    print("  atom, and there IS a genuine non-circular spectral emergence (g(n)=0)")
    print("  for the == 1 (mod 4) class. So primes ARE, in part, a CONSEQUENCE of")
    print("  self-adjoint structure. What the shift does NOT do is dissolve the")
    print("  wall: the exact reading (A) consumes the factorization, the emergent")
    print("  reading (B) is partial and scale-sector, and reaching every prime")
    print("  through the continuous phase S(T) remains the G4/RH-equivalent")
    print("  obstruction of Caminos 5-10. It LOCATES the residual; it does not")
    print("  close it. R and phi, gamma, pi, e remain assumed substrate.")
    return 0 if structural else 1


if __name__ == "__main__":
    raise SystemExit(main())
