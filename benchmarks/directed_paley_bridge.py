"""Finite Paley graph/tournament comparisons on supplied modular arithmetic.

The instrument compares symmetric and directed quadratic-residue circulants,
checks the classical prime-case Gauss-sum spectrum, and scans candidates up to
200 with a stated numerical gap tolerance. The real and imaginary gap union
is compared with trial-division labels and the separately supplied adelic
prime list. These are finite classification checks, not a universal primality
proof or a mechanism generating arithmetic from nodal dynamics.

A separate block reports normality on six graphs and sampled principal zeta
arguments (or an explicitly labeled finite prime-oscillator fallback). It does
not map analytic S(T) into a graph state or prove it unreachable. Normal finite
spectra and continuous observations have no universal obstruction merely from
being different objects. The prime 2 lies outside the two chosen odd classes.

See theory/TNFR_NUMBER_THEORY.md and
theory/TNFR_STRUCTURAL_OBSERVABILITY.md#6-limits-beyond-linear-symmetry for the current conditional scope.
The retained comparisons execute no canonical operator word or physical model.
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

from tnfr.utils.numeric import angle_diff_array  # noqa: E402

try:  # Optional supplied adelic prime list for the finite comparison
    from tnfr.dynamics.adelic import AdelicDynamics  # noqa: E402

    _HAVE_ADELIC = True
except Exception:  # pragma: no cover
    _HAVE_ADELIC = False

try:  # Optional principal zeta arguments for the finite sample
    import mpmath  # noqa: F401, E402

    _HAVE_MPMATH = True
except Exception:  # pragma: no cover
    _HAVE_MPMATH = False

TOL = 1e-9
_GAP_EPS = 1e-9  # h(n) below this counts as a tournament zero
_REAL_AXIS = np.array([0.0, np.pi, -np.pi])  # arg of a real number

# Supplied approximate zeta-zero heights used only to locate finite samples.
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
    only the prime-case formula is used as the reference here. The bounded
    composite scan below is separate evidence, not a converse proof. The
    diagnostic is finite only for n == 3 (mod 4); +inf otherwise.
    """
    if n % 4 != 3:
        return float("inf")
    eig = directed_residue_eigenvalues(n)
    secondary = eig[1:]  # drop DC (row-sum) eigenvalue
    target_im = 0.5 * math.sqrt(n)
    dev_re = float(np.max(np.abs(secondary.real + 0.5)))
    dev_im = float(np.max(np.abs(np.abs(secondary.imag) - target_im)))
    return max(dev_re, dev_im)


# --------------------------------------------------------------------------- #
# TEST 1 -- symmetric graphs and tournaments on the selected prime inputs
# --------------------------------------------------------------------------- #
def test_mod4_graph_and_tournament_controls() -> bool:
    print("=" * 78)
    print("TEST 1 -- the SAME residue-QR circulant is real for")
    print("          p == 1 (mod 4) and imaginary for p == 3 (mod 4):")
    print("          the split tracks whether -1 is a quadratic residue")
    print("          (classical finite quadratic-residue constructions)")
    print("=" * 78)

    real_class = (13, 17, 29, 37)  # p == 1 (mod 4): -1 is a QR
    phase_class = (3, 7, 11, 19, 23, 31)  # p == 3 (mod 4): -1 is NOT a QR

    worst_real_imag = 0.0  # == 1 class should be real
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
    ok = (
        worst_real_imag < TOL
        and worst_real_asym < TOL
        and worst_phase_tourn < 1e-8
        and worst_phase_gap < 1e-8
    )
    msg = (
        (
            "selected mod-4 graph/tournament formulas agree "
            "(the arithmetic of -1 being a QR)"
        )
        if ok
        else "split not aligned"
    )
    print(f"  VERDICT: {'PASS' if ok else 'FAIL'} -- {msg}")
    print()
    return ok


# --------------------------------------------------------------------------- #
# TEST 2 -- finite gap classification versus trial-division labels
# --------------------------------------------------------------------------- #
def test_imag_gap_produces_primes(limit: int = 200) -> bool:
    print("=" * 78)
    print("TEST 2 -- the imaginary gap h(n) = 0 reproduces the primes")
    print("          == 3 (mod 4), built ONLY from squares x*x % n")
    print("          (trial division supplies independent comparison labels)")
    print("=" * 78)

    candidates = [m for m in range(3, limit + 1) if m % 4 == 3]
    primes34 = [m for m in candidates if is_prime(m)]
    zeros = [m for m in candidates if tournament_gap(m) <= _GAP_EPS]

    extra = sorted(set(zeros) - set(primes34))  # composites flagged prime
    miss = sorted(set(primes34) - set(zeros))  # primes missed
    exact = (not extra) and (not miss)

    print(f"  tested n == 3 (mod 4) up to {limit}")
    print(f"  tournament-gap zeros         : {len(zeros)}")
    print(f"  primes == 3 (mod 4)          : {len(primes34)}")
    print(f"  composites flagged as zero   : {extra if extra else 'none'}")
    print(f"  primes missed                : {miss if miss else 'none'}")
    print(f"  first zeros                  : {zeros[:8]}")
    ok = exact
    msg = (
        (
            "the chosen gap tolerance matches prime labels in this finite range; "
            "the Gauss-sum formula supplies the prime-case reference"
        )
        if ok
        else "mismatch"
    )
    print(f"  VERDICT: {'PASS' if ok else 'FAIL'} -- {msg}")
    print()
    return ok


# --------------------------------------------------------------------------- #
# TEST 3 -- finite union of the two selected odd residue classes
# --------------------------------------------------------------------------- #
def test_real_plus_imag_covers_odd_primes(limit: int = 200) -> bool:
    print("=" * 78)
    print("TEST 3 -- == 1 (mod 4) real gap g(n) [Camino 9] UNION")
    print("          == 3 (mod 4) imaginary gap h(n), compared with odd")
    print("          prime labels through the supplied finite limit")
    print("=" * 78)

    real_primes = [
        m for m in range(5, limit + 1) if m % 4 == 1 and paley_gap(m) <= _GAP_EPS
    ]
    imag_primes = [
        m for m in range(3, limit + 1) if m % 4 == 3 and tournament_gap(m) <= _GAP_EPS
    ]
    emerged = sorted(set(real_primes) | set(imag_primes))

    odd_primes = [m for m in range(3, limit + 1) if is_prime(m)]
    cover_ok = emerged == odd_primes

    # Compare the finite label set; 2 is outside the two odd classes.
    all_primes = [m for m in range(2, limit + 1) if is_prime(m)]
    residual = sorted(set(all_primes) - set(emerged))

    src = "trial-division labels; no adelic comparison"
    carrier_ok = True
    if _HAVE_ADELIC:
        eng = AdelicDynamics(max_prime=limit)
        carrier_odd = sorted(int(p) for p in eng.primes if int(p) >= 3)
        carrier_ok = carrier_odd == emerged
        src = "tnfr.dynamics.adelic (supplied prime list)"

    print(
        f"  real primes  (== 1 mod 4)    : {len(real_primes)}  "
        f"e.g. {real_primes[:5]}"
    )
    print(
        f"  imag primes  (== 3 mod 4)    : {len(imag_primes)}  "
        f"e.g. {imag_primes[:5]}"
    )
    print(
        f"  union = odd primes <= {limit}    : {cover_ok} "
        f"({len(emerged)} vs {len(odd_primes)})"
    )
    print(f"  residual prime(s)            : {residual}  (the even prime 2)")
    print(f"  carrier cross-check ({src}) : {carrier_ok}")
    ok = cover_ok and residual == [2] and carrier_ok
    msg = (
        ("finite union matches the odd-prime labels; 2 is outside both classes")
        if ok
        else "coverage gap"
    )
    print(f"  VERDICT: {'PASS' if ok else 'FAIL'} -- {msg}")
    print()
    return ok


# --------------------------------------------------------------------------- #
# TEST 4 -- independent finite normality and phase-sample comparisons
# --------------------------------------------------------------------------- #
def real_axis_distances(phases: np.ndarray) -> np.ndarray:
    """Shortest circular distance of each supplied angle from the real axis."""
    return np.min(
        np.abs(angle_diff_array(phases[:, None], _REAL_AXIS[None, :], np=np)),
        axis=1,
    )


def test_finite_spectra_and_phase_samples(limit: int = 200) -> bool:
    print("=" * 78)
    print("TEST 4 -- selected normal circulants and independent phase samples")
    print("          No analytic phase reachability or obstruction is tested.")
    print("=" * 78)

    # (a) the directed circulant is normal (A A^T = A^T A) with spectrum on the
    #     secondary vertical line Re = -1/2 on these selected prime inputs.
    worst_normal = 0.0
    worst_re = 0.0
    for p in (3, 7, 11, 19, 23, 31):
        M = directed_residue_matrix(p)
        comm = float(np.linalg.norm(M @ M.T - M.T @ M))
        eig = directed_residue_eigenvalues(p)
        sec = eig[1:]
        worst_normal = max(worst_normal, comm)
        worst_re = max(worst_re, float(np.max(np.abs(sec.real + 0.5))))

    # (b) Sample principal zeta arguments or the separately defined fallback.
    # The 0.3-radian cut is a descriptive choice, not an analytic obstruction.
    imag_primes = [
        m for m in range(3, limit + 1) if m % 4 == 3 and tournament_gap(m) <= _GAP_EPS
    ]
    primes_arr = np.array(imag_primes, dtype=float)
    nu_f = np.log(primes_arr)
    samples = []
    for g in _KNOWN_ORDINATES:
        for off in (-0.7, 0.0, 0.9):
            samples.append(riemann_s_phase(g + off, nu_f, primes_arr))
    phases = np.array(samples) * np.pi
    off_axis = int(np.sum(real_axis_distances(phases) > 0.3))
    s_src = (
        "mpmath principal arg(zeta)/pi"
        if _HAVE_MPMATH
        else "finite prime-oscillator proxy"
    )

    # (c) the prime 2 is == 2 (mod 4): in neither the real nor the imaginary
    #     class -- the characteristic-2 exception.
    two_in_real = 2 % 4 == 1
    two_in_imag = 2 % 4 == 3
    two_residual = (not two_in_real) and (not two_in_imag)

    print(f"  max ||A A^T - A^T A||        : {worst_normal:.2e}  (NORMAL)")
    print(f"  max |Re(secondary) + 1/2|    : {worst_re:.2e}  (vertical line)")
    print("  Secondary eigenvalues match the specified vertical line in these cases.")
    print(f"  phase-sample source          : {s_src}")
    print(
        f"  samples beyond chosen angular cut: {off_axis} / {len(samples)} "
        f"(finite principal-phase observations)"
    )
    print(
        f"  prime 2 in real|imag class   : {two_in_real}|{two_in_imag}  "
        f"(residual = {two_residual})"
    )
    ok = (
        worst_normal < TOL
        and worst_re < 1e-8
        and off_axis >= len(_KNOWN_ORDINATES)
        and two_residual
    )
    msg = (
        (
            "finite normality and descriptive phase-sample checks agree; "
            "no relation to an analytic symmetry complement follows"
        )
        if ok
        else "finite comparison differs from the declared expectation"
    )
    print(f"  VERDICT: {'PASS' if ok else 'FAIL'} -- {msg}")
    print()
    return ok


def main() -> int:
    print(__doc__)
    r1 = test_mod4_graph_and_tournament_controls()
    r2 = test_imag_gap_produces_primes()
    r3 = test_real_plus_imag_covers_odd_primes()
    r4 = test_finite_spectra_and_phase_samples()

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(
        f"  TEST 1 selected graph/tournament formulas    : "
        f"{'PASS' if r1 else 'FAIL'}"
    )
    print(
        f"  TEST 2 finite imaginary-gap classification   : "
        f"{'PASS' if r2 else 'FAIL'}"
    )
    print(
        f"  TEST 3 finite union versus odd-prime labels  : "
        f"{'PASS' if r3 else 'FAIL'}"
    )
    print(
        f"  TEST 4 finite normality/phase observations   : "
        f"{'PASS' if r4 else 'FAIL'}"
    )
    structural = r1 and r2 and r3 and r4
    print()
    label = "ALL PASS" if structural else "SOME FAILED"
    print(f"  STRUCTURAL CHECKS: {label}")
    print()
    print("  Scope: supplied modular arithmetic, a finite gap scan, and fixed")
    print("  graph/phase comparisons. The results neither generate primes from")
    print("  nodal dynamics nor place analytic S(T) in a symmetry complement.")
    return 0 if structural else 1


if __name__ == "__main__":
    raise SystemExit(main())
