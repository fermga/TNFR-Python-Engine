"""
benchmarks/paley_bridge.py

Camino 9 -- do the zeros come from the Paley gap? Yes for the PRIME SUPPORT (a
real, self-adjoint spectral identity); no for the RIEMANN ORDINATES / the phase
residue S(T) (which stays RH-equivalent).

This harness answers a direct objection to Camino 8 (phase_wall.py). Camino 8
called the adelic carrier's content "imposed (a prime sieve)". That was too glib:
the primes feeding nu_f = log p are NOT arbitrary -- they emerge from a genuine
TNFR-native spectral mechanism, the Paley gap of Martinez Gamo, *Spectral note:
Paley gap via lambda_2 (residue circulants)*, Zenodo 10.5281/zenodo.17665853 v2
(November 2025), wired canonically into the repo as P25
(src/tnfr/riemann/paley_gap_coercivity.py). So the objection is correct: the prime
support DOES come from a structural place. This harness concedes that point with
running code -- and then shows exactly why it does NOT breach the Camino-8 wall.

TWO DIFFERENT "ZEROS" (the distinction Camino 8 blurred):
  (1) Paley-gap zeros: g(n) = |lambda_2(residue circulant) - (n - sqrt n)/2| = 0
      occurs exactly at primes n == 1 (mod 4). These are REAL integer locations;
      the mechanism detects PRIMALITY by spectral IDENTITY (not by a bound).
  (2) adelic known_zeros = {14.1347, 21.0220, ...}: the imaginary ordinates
      gamma_n of the Riemann zeta zeros zeta(1/2 + i gamma_n) = 0. The RH object.
  The Paley gap produces (1), never (2). They are different mathematical objects.

THE CLAIM (Paley grounds the REAL support, not the PHASE):
  reachable:  the prime support {p == 1 (mod 4)} of nu_f = log p emerges from
              g(n) = 0 -- a SELF-ADJOINT spectral identity (the residue circulant
              is symmetric => real spectrum => lambda_2 real => g(n) real => its
              zeros are REAL integers). So the carrier's real magnitudes are
              spectrally grounded, not sieved.
  residue:    S(T) = (1/pi) arg zeta(1/2 + iT) is a CONTINUOUS phase on the e-pi
              circle. No real g(n) produces it; the Paley zeros (real integers)
              are disjoint from the ordinates gamma_n.
  the point:  grounding the prime SUPPORT in a real/self-adjoint identity CONFIRMS
              Camino 8 -- the real/scale sector reaches the support (the "where"),
              the phase/oscillation sector (the "argument") remains unreachable.

ENGINE (known theorems -- independent ground truth, all pre-TNFR):
  - Quadratic Gauss sum: for n prime, |sum_x exp(2 pi i x^2 / n)| = sqrt(n). The
    residue circulant on a prime n == 1 (mod 4) is exactly the Paley graph, whose
    Laplacian spectrum is {0, (n - sqrt n)/2, (n + sqrt n)/2}; hence its first
    positive Laplacian eigenvalue lambda_2 = (n - sqrt n)/2 by identity. For
    composite n == 1 (mod 4) the residue circulant is not a Paley graph and
    lambda_2 deviates -- so g(n) = 0 <=> n prime == 1 (mod 4) (tested to 2601 in
    the source note; this harness re-verifies to a smaller limit).
  - Circulant diagonalisation: a circulant's eigenvalues are the DFT of its first
    row, so lambda_2 is computed by FFT in O(n log n); a symmetric first row
    (a[k] = a[n-k]) forces a real spectrum.
  - arg zeta(1/2 + iT) is a continuous real-valued function of T (Riemann-Siegel
    theta / S(T)); it is not an integer location and not confined to {0, pi}.

TNFR reading (AGENTS.md "Number Theory: primality as structural equilibrium
DNFR = 0" + src/tnfr/riemann/paley_gap_coercivity.py): the Paley gap realises
primality as a spectral equilibrium of a self-adjoint operator -- a genuine
structural source for the prime support of nu_f = log p. But it is REAL/self-
adjoint by construction, so it lives in the scale sector of the Camino-8 tetrad;
the carrier U(t) = diag(exp(i t nu_f)) still maps these real magnitudes to the
circle, and the collective phase reaching arg zeta on the critical line is the
explicit-formula / RH-equivalent content the Paley gap does not touch.

HONEST SCOPE -- structural CHECKS pass; the THESIS verdict is OPEN with a genuine
PARTIAL CONCESSION:
  We show at machine precision that (1) g(n) = 0 reproduces the primes == 1 (mod 4)
  exactly -- the prime support is spectrally grounded, NOT sieved (objection
  conceded); (2) the whole Paley mechanism is real/self-adjoint (symmetric
  circulant, real lambda_2, eigen-phases in {0, pi}) -- it lives in the Camino-8
  real sector; (3) the Paley-derived primes match the adelic carrier's == 1 (mod 4)
  support, so nu_f's real magnitudes are grounded (covers the == 1 (mod 4) class
  only); (4) the Paley zeros (real integers) are DISJOINT from the Riemann ordinates
  and a real g(n) cannot produce the continuous phase S(T). The source note itself
  says "reproducible; not a primality proof"; the canonical P25 module says it "does
  not close G4". So the Paley gap grounds the SUPPORT (real "where"), not the PHASE
  residue (continuous, RH-equivalent). It SHARPENS the Camino-8 wall; it does not
  breach it. R (continuum) and phi, gamma, pi, e remain assumed substrate.

Run:
    python benchmarks/paley_bridge.py

Status: RESEARCH (Paley-bridge falsifier; Camino 9 of the unification map).
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
from composition_arithmetic import adj_spectrum  # noqa: E402

# Optional: real Riemann zeta for the residue phase S(T).
try:  # pragma: no cover - exercised only when mpmath is installed
    import mpmath  # noqa: E402
    _HAVE_MPMATH = True
except Exception:  # pragma: no cover
    _HAVE_MPMATH = False

# Optional: the canonical adelic engine (nu_f = log p carrier whose prime support
# the Paley gap is meant to ground).
try:  # pragma: no cover - exercised only when the package is importable
    from tnfr.dynamics.adelic import AdelicDynamics  # noqa: E402
    _HAVE_ADELIC = True
except Exception:  # pragma: no cover
    _HAVE_ADELIC = False

# Optional: the canonical P25 Paley-gap module (its own honest scope: "does not
# close G4 / not a primality proof").
try:  # pragma: no cover
    from tnfr.riemann import paley_gap_coercivity as _canon_paley  # noqa: E402
    _HAVE_CANON_PALEY = True
except Exception:  # pragma: no cover
    _HAVE_CANON_PALEY = False

TOL = 1e-9
_GAP_EPS = 1e-9                     # g(n) below this counts as a Paley-gap zero
_ZERO_EIG = 1e-6                    # eigenvalues below this have undefined phase
_REAL_AXIS = np.array([0.0, np.pi, -np.pi])   # arg of a real number

# The four tetrad-associated constants (audit 2026: only pi is a genuine scale).
PHI = (1.0 + np.sqrt(5.0)) / 2.0
GAMMA = 0.5772156649015329
PI = np.pi
E = np.e

# First few Riemann non-trivial zero heights (the OTHER kind of zero).
_KNOWN_ORDINATES = (14.1347, 21.0220, 25.0109, 30.4249, 32.9351, 37.5862)


# --------------------------------------------------------------------------- #
# The Paley gap (residue-circulant lambda_2), faithful to Zenodo 17665853 v2.
# --------------------------------------------------------------------------- #
def is_prime(n: int) -> bool:
    """Trial-division primality (independent ground truth)."""
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    r = int(n ** 0.5)
    f = 3
    while f <= r:
        if n % f == 0:
            return False
        f += 2
    return True


def quadratic_residues(n: int) -> set[int]:
    """Nonzero quadratic residues mod n."""
    return {(x * x) % n for x in range(1, n) if (x * x) % n != 0}


def residue_first_row(n: int) -> np.ndarray:
    """Symmetric circulant first row: a[k] = 1 if k or n-k is a quadratic residue.

    The symmetrisation a[k] = a[n-k] makes the circulant undirected, hence its
    spectrum is real (self-adjoint sector). For prime n == 1 (mod 4) this is the
    Paley graph (since -1 is a residue, the 'or' is redundant and deg = (n-1)/2).
    """
    R = quadratic_residues(n)
    a = np.zeros(n, dtype=float)
    for k in range(1, n):
        if (k in R) or ((n - k) in R):
            a[k] = 1.0
    return a


def lambda2_residue_fft(n: int) -> float:
    """First positive Laplacian eigenvalue of the residue circulant via FFT.

    Circulant adjacency eigenvalues = DFT of the first row; Laplacian = D - A.
    """
    a = residue_first_row(n)
    d = float(a.sum())
    eig_adj = np.fft.fft(a).real            # real because a is symmetric
    mu = np.sort(d - eig_adj)               # Laplacian eigenvalues
    for v in mu:
        if v > 1e-12:
            return float(v)
    return float(mu[1])


def paley_formula(n: int) -> float:
    """Closed-form reference (n - sqrt n)/2 = lambda_2 of a genuine Paley graph."""
    return 0.5 * (n - math.sqrt(n))


def paley_gap(n: int) -> float:
    """g(n) = |lambda_2 - (n - sqrt n)/2|, meaningful only for n == 1 (mod 4)."""
    if n % 4 != 1:
        return float("inf")
    return abs(lambda2_residue_fft(n) - paley_formula(n))


def residue_circulant_matrix(n: int) -> np.ndarray:
    """Full symmetric circulant matrix M[i, j] = a[(j - i) mod n]."""
    a = residue_first_row(n)
    idx = (np.arange(n)[None, :] - np.arange(n)[:, None]) % n
    return a[idx]


def riemann_s_phase(T: float, nu_f: np.ndarray, primes: np.ndarray) -> float:
    """S(T) = (1/pi) arg zeta(1/2 + iT) via mpmath; fallback = the prime-oscillator
    phase (1/pi) arg sum_p p^(-1/2) exp(i T log p). Both are CONTINUOUS in T."""
    if _HAVE_MPMATH:
        z = mpmath.zeta(mpmath.mpc(0.5, T))
        return float(mpmath.arg(z)) / np.pi
    z = np.sum(np.exp(1j * T * nu_f) / np.sqrt(primes))
    return float(np.angle(z)) / np.pi


def _distance_to_real_axis(phases: np.ndarray) -> float:
    """Max distance from each phase to the nearest of {0, pi, -pi}."""
    if phases.size == 0:
        return 0.0
    d = np.min(np.abs(phases[:, None] - _REAL_AXIS[None, :]), axis=1)
    return float(np.max(d))


# --------------------------------------------------------------------------- #
# TEST 1 -- the Paley gap PRODUCES the primes (the support is not sieved)
# --------------------------------------------------------------------------- #
def test_paley_gap_produces_primes(limit: int = 200) -> bool:
    print("=" * 78)
    print("TEST 1 -- the Paley gap g(n) = 0 reproduces the primes == 1 (mod 4)")
    print("          (the prime support of nu_f = log p comes from a SPECTRAL place)")
    print("=" * 78)

    candidates = [m for m in range(5, limit + 1) if m % 4 == 1]
    primes14 = [m for m in candidates if is_prime(m)]
    zeros = [m for m in candidates if paley_gap(m) <= _GAP_EPS]

    extra = sorted(set(zeros) - set(primes14))     # composites flagged prime
    miss = sorted(set(primes14) - set(zeros))       # primes missed
    exact = (not extra) and (not miss)

    print(f"  tested n == 1 (mod 4) up to {limit}")
    print(f"  Paley-gap zeros           : {len(zeros)}")
    print(f"  primes == 1 (mod 4)       : {len(primes14)}")
    print(f"  composites flagged as zero: {extra if extra else 'none'}")
    print(f"  primes missed             : {miss if miss else 'none'}")
    print(f"  first zeros               : {zeros[:8]}")
    ok = exact
    print(f"  VERDICT: {'PASS' if ok else 'FAIL'} -- "
          f"{'g(n)=0 IS primality, by identity (support is structural, not sieved)' if ok else 'mismatch'}")
    print()
    return ok


# --------------------------------------------------------------------------- #
# TEST 2 -- the Paley mechanism is REAL / SELF-ADJOINT (Camino-8 sector)
# --------------------------------------------------------------------------- #
def test_paley_mechanism_is_real_self_adjoint() -> bool:
    print("=" * 78)
    print("TEST 2 -- the residue circulant is symmetric => real lambda_2 => the")
    print("          whole Paley mechanism lives in the REAL / self-adjoint sector")
    print("=" * 78)

    worst_sym = 0.0
    worst_imag = 0.0
    worst_phase = 0.0
    worst_adj = 0.0
    for n in (5, 13, 17, 29, 37):
        M = residue_circulant_matrix(n)
        sym = float(np.linalg.norm(M - M.T))
        eig = np.linalg.eigvals(M)
        imag = float(np.max(np.abs(eig.imag)))
        # eigen-phases of the symmetric circulant (exclude ~0 eigenvalues)
        keep = np.abs(eig) > _ZERO_EIG
        phase_dist = _distance_to_real_axis(np.angle(eig[keep]))
        # cross-check the adjacency spectrum (shared helper) against the closed
        # form: a Paley graph on prime n == 1 (mod 4) has eigenvalues
        # {(n-1)/2, (-1 +/- sqrt n)/2}.
        G = nx.from_numpy_array(M)
        spec = adj_spectrum(G)
        closed = np.sort(np.concatenate([
            [(n - 1) / 2.0],
            np.full((n - 1) // 2, (-1 + math.sqrt(n)) / 2.0),
            np.full((n - 1) // 2, (-1 - math.sqrt(n)) / 2.0),
        ]))
        worst_adj = max(worst_adj, float(np.max(np.abs(spec - closed))))
        worst_sym = max(worst_sym, sym)
        worst_imag = max(worst_imag, imag)
        worst_phase = max(worst_phase, phase_dist)

    # g(n) itself is a real-valued function (a difference of two reals).
    g_is_real = all(np.isreal(paley_gap(n)) for n in (5, 13, 17, 25, 29))

    print(f"  max ||M - M^T||             : {worst_sym:.2e}  (symmetric => self-adjoint)")
    print(f"  max |Im(spectrum)|          : {worst_imag:.2e}  (real spectrum)")
    print(f"  max adj-spec vs closed form : {worst_adj:.2e}  (Paley eigenvalues (-1+/-sqrt n)/2)")
    print(f"  max eigen-phase dist {{0,pi}} : {worst_phase:.2e}  (arg in {{0, pi}})")
    print(f"  g(n) is real-valued         : {g_is_real}")
    ok = (worst_sym < TOL and worst_imag < TOL and worst_adj < 1e-8
          and worst_phase < 1e-6 and g_is_real)
    print(f"  VERDICT: {'PASS' if ok else 'FAIL'} -- "
          f"{'Paley gap is real/self-adjoint: it grounds REAL support, in the Camino-8 scale sector' if ok else 'not self-adjoint'}")
    print()
    return ok


# --------------------------------------------------------------------------- #
# TEST 3 -- the Paley primes GROUND the carrier's nu_f support (not sieved)
# --------------------------------------------------------------------------- #
def test_paley_primes_ground_nu_f(limit: int = 200) -> bool:
    print("=" * 78)
    print("TEST 3 -- the Paley-derived primes match the adelic carrier's nu_f support")
    print("          on the == 1 (mod 4) class (real magnitudes grounded spectrally)")
    print("=" * 78)

    paley_primes = [m for m in range(5, limit + 1) if m % 4 == 1 and paley_gap(m) <= _GAP_EPS]

    if _HAVE_ADELIC:
        eng = AdelicDynamics(max_prime=limit)
        carrier_primes = [int(p) for p in eng.primes]
        src = "tnfr.dynamics.adelic (CANONICAL)"
    else:
        # sieve fallback only to provide a comparison set
        carrier_primes = [m for m in range(2, limit + 1) if is_prime(m)]
        src = "sieve fallback"

    carrier_14 = sorted(p for p in carrier_primes if p % 4 == 1 and p >= 5)
    match = sorted(set(paley_primes)) == carrier_14
    # nu_f magnitudes for the Paley-grounded primes
    nu_f_paley = np.log(np.array(paley_primes, dtype=float))

    print(f"  carrier nu_f source         : {src}")
    print(f"  Paley primes (== 1 mod 4)   : {len(paley_primes)}  e.g. {paley_primes[:6]}")
    print(f"  carrier primes (== 1 mod 4) : {len(carrier_14)}  e.g. {carrier_14[:6]}")
    print(f"  support match (== 1 mod 4)  : {match}")
    print(f"  nu_f = log p (first three)  : {np.round(nu_f_paley[:3], 4).tolist()}")
    print("  HONEST LIMIT: the Paley gap covers the == 1 (mod 4) class only; the")
    print("  == 3 (mod 4) primes (and 2) need a complementary construction.")
    ok = match and nu_f_paley.size > 0
    print(f"  VERDICT: {'PASS' if ok else 'FAIL'} -- "
          f"{'nu_f real support is spectrally grounded (objection conceded), not sieved' if ok else 'support mismatch'}")
    print()
    return ok


# --------------------------------------------------------------------------- #
# TEST 4 -- the Paley gap does NOT reach the ordinates / the phase S(T)
# --------------------------------------------------------------------------- #
def test_paley_does_not_reach_the_phase(limit: int = 200) -> bool:
    print("=" * 78)
    print("TEST 4 -- Paley zeros (real integers) are DISJOINT from the Riemann")
    print("          ordinates; a real g(n) cannot produce the continuous phase S(T)")
    print("=" * 78)

    paley_primes = [m for m in range(5, limit + 1) if m % 4 == 1 and paley_gap(m) <= _GAP_EPS]
    paley_set = np.array(paley_primes, dtype=float)

    # (a) the two kinds of zeros are disjoint: integer primes vs real ordinates
    min_dist = min(
        float(np.min(np.abs(paley_set - g))) for g in _KNOWN_ORDINATES
    )
    disjoint = min_dist > 0.5

    # (b) S(T) is a continuous phase off the {0, pi} axis (sampled near ordinates)
    nu_f = np.log(paley_set) if paley_set.size else np.array([math.log(5.0)])
    primes_arr = paley_set if paley_set.size else np.array([5.0])
    samples = []
    for g in _KNOWN_ORDINATES:
        for off in (-0.7, 0.0, 0.9):
            samples.append(riemann_s_phase(g + off, nu_f, primes_arr))
    phases = np.array(samples) * np.pi      # back to radians for axis distance
    s_dist = _distance_to_real_axis(np.array([(p % (2 * np.pi)) for p in phases]))
    off_axis = int(np.sum(np.min(np.abs(phases[:, None] % (2 * np.pi) - _REAL_AXIS[None, :]), axis=1) > 0.3))
    s_src = "mpmath zeta(1/2+iT)" if _HAVE_MPMATH else "prime-oscillator fallback"

    # (c) g(n) is real-valued => its 'phase content' is in {0, pi}; S(T) is not.
    g_phase = _distance_to_real_axis(np.angle(np.array([paley_gap(n) + 0j for n in (5, 13, 17)])))

    canon = "n/a"
    if _HAVE_CANON_PALEY:
        canon = (getattr(_canon_paley, "__name__", "paley_gap_coercivity")
                 + " present (P25: 'does not close G4; not a primality proof')")

    print(f"  Paley zeros (integers)      : {paley_primes[:6]} ...")
    print(f"  Riemann ordinates (reals)   : {list(_KNOWN_ORDINATES)}")
    print(f"  min |Paley - ordinate|      : {min_dist:.3f}  (>> 0 => disjoint)")
    print(f"  S(T) source                 : {s_src}")
    print(f"  S(T) max dist from {{0,pi}}   : {s_dist:.3f} rad  ({off_axis} samples off-axis)")
    print(f"  g(n) phase dist from {{0,pi}} : {g_phase:.2e}  (g is real => arg in {{0, pi}})")
    print(f"  canonical P25               : {canon}")
    ok = disjoint and s_dist > 0.3 and off_axis >= len(_KNOWN_ORDINATES) and g_phase < 1e-6
    print(f"  VERDICT: {'PASS' if ok else 'FAIL'} -- "
          f"{'Paley grounds the real support, NOT the continuous phase S(T) (RH-equivalent)' if ok else 'phase reached?!'}")
    print()
    return ok


def main() -> int:
    print(__doc__)
    r1 = test_paley_gap_produces_primes()
    r2 = test_paley_mechanism_is_real_self_adjoint()
    r3 = test_paley_primes_ground_nu_f()
    r4 = test_paley_does_not_reach_the_phase()

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  TEST 1 Paley gap produces primes == 1 (mod 4) : {'PASS' if r1 else 'FAIL'}")
    print(f"  TEST 2 Paley mechanism is real/self-adjoint   : {'PASS' if r2 else 'FAIL'}")
    print(f"  TEST 3 Paley primes ground nu_f real support  : {'PASS' if r3 else 'FAIL'}")
    print(f"  TEST 4 Paley does NOT reach the phase S(T)    : {'PASS' if r4 else 'FAIL'}")
    structural = r1 and r2 and r3 and r4
    print()
    print(f"  STRUCTURAL CHECKS: {'ALL PASS' if structural else 'SOME FAILED'}")
    print()
    print("  THESIS VERDICT: OPEN, with a genuine PARTIAL CONCESSION.")
    print("  The objection is correct: the prime support of nu_f = log p is NOT")
    print("  sieved -- it emerges from the Paley gap g(n) = 0, a SELF-ADJOINT")
    print("  spectral IDENTITY that realises primality (== 1 mod 4) as DNFR = 0")
    print("  structural equilibrium. But that mechanism is REAL/self-adjoint, so it")
    print("  lives in the Camino-8 scale sector: it grounds the support (the real")
    print("  'where'), and the Paley zeros (integers) are disjoint from the Riemann")
    print("  ordinates. The residue S(T) = (1/pi) arg zeta(1/2 + iT) is a CONTINUOUS")
    print("  phase on the e-pi circle; no real g(n) produces it. The source note")
    print("  says 'not a primality proof'; the canonical P25 module says it 'does")
    print("  not close G4'. So the Paley gap SHARPENS the real-vs-phase wall: it")
    print("  shows the real sector reaches even the prime SUPPORT, while the phase")
    print("  residue stays unreachable. Reaching S(T) remains RH-equivalent. R and")
    print("  phi, gamma, pi, e remain assumed substrate.")
    return 0 if structural else 1


if __name__ == "__main__":
    raise SystemExit(main())
