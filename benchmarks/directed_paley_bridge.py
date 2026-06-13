"""Camino 14 -- the == 3 (mod 4) complement of the Paley bridge.

CONTEXT (where this sits in the emergence-of-numbers line)
----------------------------------------------------------
Camino 9 (``paley_bridge.py``) conceded a sharp point: the prime support of the
adelic carrier nu_f = log p is NOT sieved -- it EMERGES from a self-adjoint
spectral identity, the Paley gap g(n) = 0, which realises primality as a
DNFR = 0 structural equilibrium. But that mechanism is REAL / self-adjoint, so
it only reaches the primes p == 1 (mod 4). Camino 9 left an explicit, honest
gap (echoed in ``primes_as_consequence.py``):

    "the == 3 (mod 4) primes (and 2) need a complementary construction."

This harness IS that complementary construction. Its thesis:

    The prime residue classes mod 4 split EXACTLY along the real/phase boundary
    of the Equivariance Wall, because the split is the arithmetic of whether -1
    is a quadratic residue.

        p == 1 (mod 4)  <=>  -1 is a QR  <=>  symmetric Paley GRAPH
                              (A = A^T, real spectrum, SCALE sector) [C9]

        p == 3 (mod 4)  <=>  -1 is NOT a QR  <=>  Paley TOURNAMENT
                              (A + A^T = J - I, spectrum on Re = -1/2,
                               PURELY IMAGINARY secondary part, PHASE sector)

For a prime q == 3 (mod 4) the Gauss sum is g = i*sqrt(q), so the directed
quadratic-residue circulant has eigenvalues

        { (q - 1)/2 ;  (-1 +/- i*sqrt q)/2  (each (q-1)/2 times) } .

The sqrt(q) Gauss-sum signature now appears as the IMAGINARY part +/- sqrt(q)/2
(the real counterpart of Camino 9's real signature -1/2 +/- sqrt(q)/2). The new
detector

        h(n) = deviation of the directed residue circulant from the doubly
               regular tournament spectrum {(n-1)/2, (-1 +/- i sqrt n)/2}

satisfies h(n) = 0  <=>  n is a prime == 3 (mod 4). It is built ONLY from the
squares x*x % n (never from trial division n % k), so it is a genuine
primes-OUT emergence, extending Camino 9's Reading B to the second odd class.

Together:  == 1 (mod 4) [real, Camino 9]  (+)  == 3 (mod 4) [imaginary, here]
           =  ALL odd primes, by two NORMAL spectral identities.

HONEST SCOPE (this closes NOTHING)
----------------------------------
1. The == 3 (mod 4) operator is a circulant, hence NORMAL, with a DISCRETE
   point spectrum on the vertical line Re = -1/2. "Self-adjoint up to a factor
   i" is still a normal, discrete object -- NOT the continuous phase
   S(T) = (1/pi) arg zeta(1/2 + iT), which remains RH-equivalent and
   unreachable.
2. The prime 2 (== 2 mod 4) belongs to NEITHER class: it is the even,
   characteristic-2 exception, outside both the real and the imaginary sector.
3. R and phi, gamma, pi, e remain assumed substrate. G4 = RH stays OPEN.

So Camino 14 LOCATES and EXTENDS the emergence-of-numbers line (all odd primes
now emerge structurally, split exactly by the real/phase wall) and CONNECTS the
mod-4 prime split to the wall -- but it does not move the wall.

Reuses canonical helpers from ``paley_bridge.py`` (is_prime,
quadratic_residues, paley_gap, riemann_s_phase) and cross-checks the
== 3 (mod 4) prime support against the canonical ``tnfr.dynamics.adelic``
carrier when available.

Run:
    python benchmarks/directed_paley_bridge.py
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_SRC = os.path.abspath(os.path.join(_HERE, "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from paley_bridge import (  # noqa: E402  (path set above)
    is_prime,
    paley_gap,
    quadratic_residues,
    riemann_s_phase,
)

try:  # canonical adelic carrier (same guarded import as Camino 9)
    from tnfr.dynamics.adelic import AdelicDynamics  # noqa: E402

    _HAVE_ADELIC = True
except Exception:  # pragma: no cover
    _HAVE_ADELIC = False

try:  # mpmath only used for the S(T) wall sample (Camino 9 parity)
    import mpmath  # noqa: F401, E402

    _HAVE_MPMATH = True
except Exception:  # pragma: no cover
    _HAVE_MPMATH = False

TOL = 1e-9
_GAP_EPS = 1e-9        # h(n) below this counts as a tournament zero
_REAL_AXIS = np.array([0.0, np.pi, -np.pi])    # arg of a real number

# First few Riemann non-trivial zero heights (the continuous-phase witness).
_KNOWN_ORDINATES = (14.1347, 21.0220, 25.0109, 30.4249, 32.9351, 37.5862)


# --------------------------------------------------------------------------- #
# The directed (NON-symmetrised) quadratic-residue circulant.
# --------------------------------------------------------------------------- #
def directed_residue_first_row(n: int) -> np.ndarray:
    """Raw QR-indicator first row: a[k] = 1 iff k is a nonzero QR mod n.

    Unlike paley_bridge.residue_first_row (which symmetrises a[k] = a[n-k] to
    force a REAL spectrum), this keeps the raw indicator. For a prime
    n == 3 (mod 4), -1 is a NON-residue, so exactly one of k, n-k is a QR:
    a[k] + a[n-k] = 1, i.e. A + A^T = J - I (a tournament), and the spectrum
    is non-real.
    """
    R = quadratic_residues(n)
    a = np.zeros(n, dtype=float)
    for k in range(1, n):
        if k in R:
            a[k] = 1.0
    return a


def directed_residue_eigenvalues(n: int) -> np.ndarray:
    """Complex eigenvalues of the directed residue circulant.

    Circulant eigenvalues are the DFT of the first row; here the row is NOT
    symmetric, so the eigenvalues are genuinely complex.
    """
    return np.fft.fft(directed_residue_first_row(n))


def directed_residue_matrix(n: int) -> np.ndarray:
    """Full directed circulant M[i, j] = a[(j - i) mod n] (asymmetric)."""
    a = directed_residue_first_row(n)
    idx = (np.arange(n)[None, :] - np.arange(n)[:, None]) % n
    return a[idx]


def tournament_imag_signature(n: int) -> float:
    """Largest |Im(eigenvalue)| of the directed residue circulant.

    For a prime n == 3 (mod 4) this equals sqrt(n)/2 (Gauss sum i*sqrt n),
    the imaginary counterpart of Camino 9's real signature.
    """
    eig = directed_residue_eigenvalues(n)
    return float(np.max(np.abs(eig.imag)))


def tournament_gap(n: int) -> float:
    """h(n): deviation from the doubly-regular tournament spectrum.

    A prime q == 3 (mod 4) yields secondary eigenvalues exactly at
    (-1 +/- i sqrt q)/2: real part -1/2 and |imag| = sqrt(q)/2. h(n) measures
    the worst deviation of the secondary eigenvalues from that target, so
    h(n) = 0  <=>  n is a prime == 3 (mod 4). Defined (finite) only on that
    class; +inf otherwise.
    """
    if n % 4 != 3:
        return float("inf")
    eig = directed_residue_eigenvalues(n)
    secondary = eig[1:]                       # drop DC (row-sum) eigenvalue
    target_im = 0.5 * math.sqrt(n)
    dev_re = float(np.max(np.abs(secondary.real + 0.5)))
    dev_im = float(np.max(np.abs(np.abs(secondary.imag) - target_im)))
    return max(dev_re, dev_im)


# --------------------------------------------------------------------------- #
# TEST 1 -- the mod-4 prime split IS the real/phase split of the wall
# --------------------------------------------------------------------------- #
def test_mod4_split_is_real_phase_split() -> bool:
    print("=" * 78)
    print("TEST 1 -- the SAME residue-QR circulant is real for")
    print("          p == 1 (mod 4) and imaginary for p == 3 (mod 4):")
    print("          the split tracks whether -1 is a quadratic residue")
    print("          (= the real/phase boundary of the wall)")
    print("=" * 78)

    real_class = (13, 17, 29, 37)             # p == 1 (mod 4): -1 is a QR
    phase_class = (3, 7, 11, 19, 23, 31)      # p == 3 (mod 4): -1 is NOT a QR

    worst_real_imag = 0.0                     # == 1 class should be real
    worst_real_asym = 0.0
    for p in real_class:
        R = quadratic_residues(p)
        minus_one_is_qr = (p - 1) in R
        M = directed_residue_matrix(p)
        asym = float(np.linalg.norm(M - M.T))
        imag = tournament_imag_signature(p)
        worst_real_asym = max(worst_real_asym, asym)
        worst_real_imag = max(worst_real_imag, imag)
        assert minus_one_is_qr, f"-1 should be a QR for p == 1 mod 4 ({p})"

    worst_phase_tourn = 0.0  # == 3 class is a tournament
    worst_phase_gap = 0.0
    for p in phase_class:
        R = quadratic_residues(p)
        minus_one_is_qr = (p - 1) in R
        M = directed_residue_matrix(p)
        n = M.shape[0]
        # A + A^T = J - I exactly for a Paley tournament
        tourn = float(np.linalg.norm(M + M.T - (np.ones((n, n)) - np.eye(n))))
        sig = tournament_imag_signature(p)
        expected = 0.5 * math.sqrt(p)
        worst_phase_tourn = max(worst_phase_tourn, tourn)
        worst_phase_gap = max(worst_phase_gap, abs(sig - expected))
        assert not minus_one_is_qr, f"-1 should NOT be a QR for {p}"

    print("  p == 1 (mod 4)  (-1 IS a QR => symmetric Paley GRAPH):")
    print(f"    max ||M - M^T||      : {worst_real_asym:.2e}  (symmetric)")
    print(f"    max |Im(spectrum)|   : {worst_real_imag:.2e}  (REAL)")
    print("  p == 3 (mod 4)  (-1 NOT a QR => Paley TOURNAMENT):")
    print(f"    max ||A+A^T-(J-I)||  : {worst_phase_tourn:.2e}  (tournament)")
    print(f"    max ||Im|-sqrt(p)/2| : {worst_phase_gap:.2e}  (IMAG)")
    ok = (worst_real_imag < TOL and worst_real_asym < TOL
          and worst_phase_tourn < 1e-8 and worst_phase_gap < 1e-8)
    msg = ("mod-4 prime split == real/phase wall split "
           "(the arithmetic of -1 being a QR)") if ok else "split not aligned"
    print(f"  VERDICT: {'PASS' if ok else 'FAIL'} -- {msg}")
    print()
    return ok


# --------------------------------------------------------------------------- #
# TEST 2 -- the == 3 (mod 4) primes EMERGE (primes-OUT, squares only)
# --------------------------------------------------------------------------- #
def test_imag_gap_produces_primes(limit: int = 200) -> bool:
    print("=" * 78)
    print("TEST 2 -- the imaginary gap h(n) = 0 reproduces the primes")
    print("          == 3 (mod 4), built ONLY from squares x*x % n")
    print("          (genuine primes-OUT, never trial division)")
    print("=" * 78)

    candidates = [m for m in range(3, limit + 1) if m % 4 == 3]
    primes34 = [m for m in candidates if is_prime(m)]
    zeros = [m for m in candidates if tournament_gap(m) <= _GAP_EPS]

    extra = sorted(set(zeros) - set(primes34))     # composites flagged prime
    miss = sorted(set(primes34) - set(zeros))      # primes missed
    exact = (not extra) and (not miss)

    print(f"  tested n == 3 (mod 4) up to {limit}")
    print(f"  tournament-gap zeros         : {len(zeros)}")
    print(f"  primes == 3 (mod 4)          : {len(primes34)}")
    print(f"  composites flagged as zero   : {extra if extra else 'none'}")
    print(f"  primes missed                : {miss if miss else 'none'}")
    print(f"  first zeros                  : {zeros[:8]}")
    ok = exact
    msg = ("h(n)=0 IS primality (== 3 mod 4) via the imaginary "
           "Gauss-sum identity (squares only)") if ok else "mismatch"
    print(f"  VERDICT: {'PASS' if ok else 'FAIL'} -- {msg}")
    print()
    return ok


# --------------------------------------------------------------------------- #
# TEST 3 -- completion: real (==1) (+) imaginary (==3) = ALL odd primes
# --------------------------------------------------------------------------- #
def test_real_plus_imag_covers_odd_primes(limit: int = 200) -> bool:
    print("=" * 78)
    print("TEST 3 -- == 1 (mod 4) real gap g(n) [Camino 9] UNION")
    print("          == 3 (mod 4) imaginary gap h(n) [here] = ALL odd")
    print("          primes (the only residual is the even prime 2)")
    print("=" * 78)

    real_primes = [m for m in range(5, limit + 1)
                   if m % 4 == 1 and paley_gap(m) <= _GAP_EPS]
    imag_primes = [m for m in range(3, limit + 1)
                   if m % 4 == 3 and tournament_gap(m) <= _GAP_EPS]
    emerged = sorted(set(real_primes) | set(imag_primes))

    odd_primes = [m for m in range(3, limit + 1) if is_prime(m)]
    cover_ok = emerged == odd_primes

    # the ONLY prime not covered by either class is 2 (== 2 mod 4)
    all_primes = [m for m in range(2, limit + 1) if is_prime(m)]
    residual = sorted(set(all_primes) - set(emerged))

    src = "sieve fallback"
    carrier_ok = True
    if _HAVE_ADELIC:
        eng = AdelicDynamics(max_prime=limit)
        carrier_odd = sorted(int(p) for p in eng.primes if int(p) >= 3)
        carrier_ok = carrier_odd == emerged
        src = "tnfr.dynamics.adelic (CANONICAL)"

    print(f"  real primes  (== 1 mod 4)    : {len(real_primes)}  "
          f"e.g. {real_primes[:5]}")
    print(f"  imag primes  (== 3 mod 4)    : {len(imag_primes)}  "
          f"e.g. {imag_primes[:5]}")
    print(f"  union = odd primes <= {limit}    : {cover_ok} "
          f"({len(emerged)} vs {len(odd_primes)})")
    print(f"  residual prime(s)            : {residual}  (the even prime 2)")
    print(f"  carrier cross-check ({src}) : {carrier_ok}")
    ok = cover_ok and residual == [2] and carrier_ok
    msg = ("real (+) imaginary = every odd prime; only 2 sits "
           "outside both sectors") if ok else "coverage gap"
    print(f"  VERDICT: {'PASS' if ok else 'FAIL'} -- {msg}")
    print()
    return ok


# --------------------------------------------------------------------------- #
# TEST 4 -- honest scope: discrete/normal sectors do NOT reach S(T); 2 remains
# --------------------------------------------------------------------------- #
def test_does_not_reach_the_phase(limit: int = 200) -> bool:
    print("=" * 78)
    print("TEST 4 -- the == 3 (mod 4) operator is NORMAL with a")
    print("          DISCRETE spectrum on Re = -1/2; 'i times")
    print("          self-adjoint' is still not the continuous phase")
    print("          S(T); and the prime 2 stays outside both classes")
    print("=" * 78)

    # (a) the directed circulant is normal (A A^T = A^T A) with spectrum on the
    #     vertical line Re = -1/2 (a DISCRETE point set), never a continuum.
    worst_normal = 0.0
    worst_re = 0.0
    for p in (3, 7, 11, 19, 23, 31):
        M = directed_residue_matrix(p)
        comm = float(np.linalg.norm(M @ M.T - M.T @ M))
        eig = directed_residue_eigenvalues(p)
        sec = eig[1:]
        worst_normal = max(worst_normal, comm)
        worst_re = max(worst_re, float(np.max(np.abs(sec.real + 0.5))))

    # (b) S(T) is a CONTINUOUS phase off the {0, pi} axis near the ordinates.
    imag_primes = [m for m in range(3, limit + 1)
                   if m % 4 == 3 and tournament_gap(m) <= _GAP_EPS]
    primes_arr = np.array(imag_primes, dtype=float)
    nu_f = np.log(primes_arr)
    samples = []
    for g in _KNOWN_ORDINATES:
        for off in (-0.7, 0.0, 0.9):
            samples.append(riemann_s_phase(g + off, nu_f, primes_arr))
    phases = np.array(samples) * np.pi
    off_axis = int(np.sum(
        np.min(np.abs(phases[:, None] % (2 * np.pi) - _REAL_AXIS[None, :]),
               axis=1) > 0.3))
    s_src = "mpmath zeta(1/2+iT)" if _HAVE_MPMATH else "prime-oscillator"

    # (c) the prime 2 is == 2 (mod 4): in neither the real nor the imaginary
    #     class -- the characteristic-2 exception.
    two_in_real = (2 % 4 == 1)
    two_in_imag = (2 % 4 == 3)
    two_residual = (not two_in_real) and (not two_in_imag)

    print(f"  max ||A A^T - A^T A||        : {worst_normal:.2e}  (NORMAL)")
    print(f"  max |Re(secondary) + 1/2|    : {worst_re:.2e}  (vertical line)")
    print("  => spectrum is a DISCRETE point set (i * real), not a continuum")
    print(f"  S(T) source                  : {s_src}")
    print(f"  S(T) samples off {{0,pi}}      : {off_axis} / {len(samples)} "
          f"(continuous phase)")
    print(f"  prime 2 in real|imag class   : {two_in_real}|{two_in_imag}  "
          f"(residual = {two_residual})")
    ok = (worst_normal < TOL and worst_re < 1e-8
          and off_axis >= len(_KNOWN_ORDINATES) and two_residual)
    msg = ("real (+) imaginary cover the odd primes via "
           "NORMAL/discrete identities; S(T) and 2 stay out -- "
           "wall unchanged") if ok else "phase reached?!"
    print(f"  VERDICT: {'PASS' if ok else 'FAIL'} -- {msg}")
    print()
    return ok


def main() -> int:
    print(__doc__)
    r1 = test_mod4_split_is_real_phase_split()
    r2 = test_imag_gap_produces_primes()
    r3 = test_real_plus_imag_covers_odd_primes()
    r4 = test_does_not_reach_the_phase()

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  TEST 1 mod-4 split == real/phase wall split   : "
          f"{'PASS' if r1 else 'FAIL'}")
    print(f"  TEST 2 imaginary gap produces == 3 mod 4 primes: "
          f"{'PASS' if r2 else 'FAIL'}")
    print(f"  TEST 3 real (+) imaginary = all odd primes     : "
          f"{'PASS' if r3 else 'FAIL'}")
    print(f"  TEST 4 discrete/normal sectors do NOT reach S(T): "
          f"{'PASS' if r4 else 'FAIL'}")
    structural = r1 and r2 and r3 and r4
    print()
    label = "ALL PASS" if structural else "SOME FAILED"
    print(f"  STRUCTURAL CHECKS: {label}")
    print()
    print("  THESIS VERDICT: OPEN, by design (it EXTENDS, it does")
    print("  not close). Camino 9 grounded the == 1 (mod 4) primes in a")
    print("  REAL/self-adjoint Paley gap (the scale sector). This harness")
    print("  grounds the == 3 (mod 4) primes in the IMAGINARY Gauss-sum")
    print("  signature of the Paley TOURNAMENT (eigenvalues")
    print("  (-1 +/- i sqrt q)/2, the phase-sector counterpart). The two")
    print("  together make EVERY odd prime emerge from squares alone,")
    print("  split EXACTLY by whether -1 is a quadratic residue -- which")
    print("  is precisely the real-vs-phase boundary of the Equivariance")
    print("  Wall. But both sectors are NORMAL operators with DISCRETE")
    print("  spectra: 'i times self-adjoint' is not the continuous phase")
    print("  S(T) = (1/pi) arg zeta(1/2 + iT), which remains RH-equivalent")
    print("  and unreachable; and the prime 2 sits outside both classes.")
    print("  So the emergence-of-numbers line is EXTENDED to all odd")
    print("  primes and CONNECTED to the wall, but the wall is not moved.")
    print("  G4 = RH stays OPEN; R and phi, gamma, pi, e remain assumed")
    print("  substrate.")
    return 0 if structural else 1


if __name__ == "__main__":
    raise SystemExit(main())
