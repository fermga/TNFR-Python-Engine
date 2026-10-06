"""Finite comparisons for supplied symmetric quadratic-residue circulants.

The controls compare a numerical Paley-gap criterion with trial-division
labels through a finite limit, check classical prime-case adjacency formulas,
compare the selected labels with a supplied adelic prime list, and verify
combinatorial/random-walk Laplacian rescaling on regular graphs.

For a prime n == 1 (mod 4), the Paley Laplacian has nonzero eigenvalues
(n +/- sqrt(n))/2. A finite composite scan does not prove the converse of
this prime-case identity. Gap values within the stated tolerance are numerical
classifications, not exact zeros or nodal equilibria. For positive regular
degree d, L_rw = L/d; this preserves exact gap zeros for these two definitions,
without making a claim about arbitrary Laplacians.

These controls supply modular arithmetic, graph construction and comparison
labels. They neither derive primes from nodal dynamics nor establish an
analytic Riemann-phase obstruction. The separate principal-argument helper
is retained for the directed finite instrument; it is not used as a bridge.

See theory/TNFR_NUMBER_THEORY.md and
theory/TNFR_STRUCTURAL_OBSERVABILITY.md#6-limits-beyond-linear-symmetry.
Run: python benchmarks/paley_bridge.py
"""

from __future__ import annotations

import math
import os
import sys

import networkx as nx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Robust fallback so the harness also runs without PYTHONPATH=src preset.
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
)
from composition_arithmetic import adj_spectrum  # noqa: E402

# Optional principal zeta arguments for the separately labeled helper.
try:  # pragma: no cover - exercised only when mpmath is installed
    import mpmath  # noqa: E402

    _HAVE_MPMATH = True
except Exception:  # pragma: no cover
    _HAVE_MPMATH = False

# Optional supplied adelic prime list for a finite label comparison.
try:  # pragma: no cover - exercised only when the package is importable
    from tnfr.dynamics.adelic import AdelicDynamics  # noqa: E402

    _HAVE_ADELIC = True
except Exception:  # pragma: no cover
    _HAVE_ADELIC = False

TOL = 1e-9
_GAP_EPS = 1e-9  # numerical zero tolerance for the finite gap scan
_ZERO_EIG = 1e-6  # eigenvalues below this have undefined phase
_REAL_AXIS = np.array([0.0, np.pi, -np.pi])  # arg of a real number


# --------------------------------------------------------------------------- #
# The Paley gap (residue-circulant lambda_2), faithful to Zenodo 17665853 v2.
# --------------------------------------------------------------------------- #
def is_prime(n: int) -> bool:
    """Trial-division primality (independent ground truth)."""
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    r = int(n**0.5)
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
    eig_adj = np.fft.fft(a).real  # real because a is symmetric
    mu = np.sort(d - eig_adj)  # Laplacian eigenvalues
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


# --------------------------------------------------------------------------- #
# Canonical TNFR form: same gap through the RANDOM-WALK Laplacian L_rw = I-D^-1 W
# (the emergent DNFR EPI channel, AGENTS.md sec. 2). Regular circulant => exact
# rescaling L_rw = (1/d) L, so lambda_2(L_rw) = lambda_2(L)/d and the faithful gap
# g_rw = g_comb/d has the same exact zeros; its target reduces to the clean closed
# form sqrt n/(sqrt n+1) on the prime locus. TEST 4 checks the d-equivalence to
# machine precision.
# --------------------------------------------------------------------------- #
def lambda2_residue_rw(n: int) -> float:
    """First positive eigenvalue of the CANONICAL random-walk Laplacian L_rw.

    L_rw = I - D^-1 A is the emergent TNFR DNFR EPI channel (AGENTS.md sec. 2),
    computed independently of ``lambda2_residue_fft`` so the d-rescaling identity
    lambda_2(L) == d * lambda_2(L_rw) is a genuine check, not a tautology. The
    residue circulant is regular (D = d*I), so L_rw eigenvalues = 1 - (DFT of a)/d.
    """
    a = residue_first_row(n)
    d = float(a.sum())
    eig_adj = np.fft.fft(a).real  # real because a is symmetric
    mu_rw = np.sort(1.0 - eig_adj / d)  # random-walk Laplacian eigenvalues
    for v in mu_rw:
        if v > 1e-12:
            return float(v)
    return float(mu_rw[1])


def paley_formula_rw(n: int) -> float:
    """Canonical closed form sqrt n/(sqrt n + 1) = 1 - 1/(sqrt n + 1).

    Equals the combinatorial (n - sqrt n)/2 divided by the Paley degree
    d = (n-1)/2; bounded in (0, 1) for these positive prime inputs.
    """
    s = math.sqrt(n)
    return s / (s + 1.0)


def paley_gap_rw(n: int) -> float:
    """Canonical g_rw(n) = |lambda_2(L_rw) - (n - sqrt n)/(2 d)|.

    The target uses the ACTUAL residue-circulant degree d = a.sum(); on the prime
    locus d = (n-1)/2 and it reduces to the clean closed form sqrt n/(sqrt n + 1).
    Because g_rw = g_comb/d exactly (regular-graph rescaling), its exact zeros
    coincide with those of ``paley_gap``. A finite tolerance scan is a separate
    numerical comparison. (The naive prime-target gap
    |lambda_2(L_rw) - sqrt n/(sqrt n + 1)| is faithful only on the prime locus:
    off it the closed-form target assumes the Paley degree, so a composite whose
    degree differs can be mis-flagged.) Meaningful only for n == 1 (mod 4).
    """
    if n % 4 != 1:
        return float("inf")
    a = residue_first_row(n)
    d = float(a.sum())
    target = (n - math.sqrt(n)) / (2.0 * d)
    return abs(lambda2_residue_rw(n) - target)


def residue_circulant_matrix(n: int) -> np.ndarray:
    """Full symmetric circulant matrix M[i, j] = a[(j - i) mod n]."""
    a = residue_first_row(n)
    idx = (np.arange(n)[None, :] - np.arange(n)[:, None]) % n
    return a[idx]


def riemann_s_phase(T: float, nu_f: np.ndarray, primes: np.ndarray) -> float:
    """Return the principal zeta argument divided by pi when mpmath is present.

    Otherwise return the distinct finite prime-oscillator principal argument.
    Principal arguments have branch jumps; this is not a continuous lift or
    the analytic number-theory definition of S(T).
    """
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
# TEST 1 -- finite Paley-gap scan versus supplied trial-division labels
# --------------------------------------------------------------------------- #
def test_paley_gap_produces_primes(limit: int = 200) -> bool:
    print("=" * 78)
    print("TEST 1 -- finite Paley-gap classification for n == 1 (mod 4)")
    print("          (numerical gap tolerance compared with trial-division labels)")
    print("=" * 78)

    candidates = [m for m in range(5, limit + 1) if m % 4 == 1]
    primes14 = [m for m in candidates if is_prime(m)]
    zeros = [m for m in candidates if paley_gap(m) <= _GAP_EPS]

    extra = sorted(set(zeros) - set(primes14))  # composites flagged prime
    miss = sorted(set(primes14) - set(zeros))  # primes missed
    exact = (not extra) and (not miss)

    print(f"  tested n == 1 (mod 4) up to {limit}")
    print(f"  Paley-gap zeros           : {len(zeros)}")
    print(f"  primes == 1 (mod 4)       : {len(primes14)}")
    print(f"  composites flagged as zero: {extra if extra else 'none'}")
    print(f"  primes missed             : {miss if miss else 'none'}")
    print(f"  first zeros               : {zeros[:8]}")
    ok = exact
    print(
        f"  VERDICT: {'PASS' if ok else 'FAIL'} -- "
        f"{'finite gap classification matches the supplied prime labels' if ok else 'mismatch'}"
    )
    print()
    return ok


# --------------------------------------------------------------------------- #
# TEST 2 -- selected symmetric matrices and prime-case spectrum formulas
# --------------------------------------------------------------------------- #
def test_paley_mechanism_is_real_self_adjoint() -> bool:
    print("=" * 78)
    print("TEST 2 -- selected symmetric residue circulants have real spectra;")
    print("          compare their eigenvalues with the prime-case Paley formulas")
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
        closed = np.sort(
            np.concatenate(
                [
                    [(n - 1) / 2.0],
                    np.full((n - 1) // 2, (-1 + math.sqrt(n)) / 2.0),
                    np.full((n - 1) // 2, (-1 - math.sqrt(n)) / 2.0),
                ]
            )
        )
        worst_adj = max(worst_adj, float(np.max(np.abs(spec - closed))))
        worst_sym = max(worst_sym, sym)
        worst_imag = max(worst_imag, imag)
        worst_phase = max(worst_phase, phase_dist)

    # g(n) itself is a real-valued function (a difference of two reals).
    g_is_real = all(np.isreal(paley_gap(n)) for n in (5, 13, 17, 25, 29))

    print(
        f"  max ||M - M^T||             : {worst_sym:.2e}  (symmetric => self-adjoint)"
    )
    print(f"  max |Im(spectrum)|          : {worst_imag:.2e}  (real spectrum)")
    print(
        f"  max adj-spec vs closed form : {worst_adj:.2e}  (Paley eigenvalues (-1+/-sqrt n)/2)"
    )
    print(f"  max eigen-phase dist {{0,pi}} : {worst_phase:.2e}  (arg in {{0, pi}})")
    print(f"  g(n) is real-valued         : {g_is_real}")
    ok = (
        worst_sym < TOL
        and worst_imag < TOL
        and worst_adj < 1e-8
        and worst_phase < 1e-6
        and g_is_real
    )
    print(
        f"  VERDICT: {'PASS' if ok else 'FAIL'} -- "
        f"{'selected symmetric circulants match the prime-case spectral formulas' if ok else 'not self-adjoint'}"
    )
    print()
    return ok


# --------------------------------------------------------------------------- #
# TEST 3 -- finite selected-label comparison with supplied carrier inputs
# --------------------------------------------------------------------------- #
def test_paley_primes_ground_nu_f(limit: int = 200) -> bool:
    print("=" * 78)
    print("TEST 3 -- compare finite gap-selected labels with supplied carrier labels")
    print("          on the == 1 (mod 4) class; nu_f = log p is a supplied encoding")
    print("=" * 78)

    paley_primes = [
        m for m in range(5, limit + 1) if m % 4 == 1 and paley_gap(m) <= _GAP_EPS
    ]

    if _HAVE_ADELIC:
        eng = AdelicDynamics(max_prime=limit)
        carrier_primes = [int(p) for p in eng.primes]
        src = "tnfr.dynamics.adelic (CANONICAL)"
    else:
        # trial-division fallback only to provide a comparison set
        carrier_primes = [m for m in range(2, limit + 1) if is_prime(m)]
        src = "trial-division fallback"

    carrier_14 = sorted(p for p in carrier_primes if p % 4 == 1 and p >= 5)
    match = sorted(set(paley_primes)) == carrier_14
    # Supplied logarithmic encoding of the selected labels
    nu_f_paley = np.log(np.array(paley_primes, dtype=float))

    print(f"  carrier nu_f source         : {src}")
    print(
        f"  Paley primes (== 1 mod 4)   : {len(paley_primes)}  e.g. {paley_primes[:6]}"
    )
    print(f"  carrier primes (== 1 mod 4) : {len(carrier_14)}  e.g. {carrier_14[:6]}")
    print(f"  support match (== 1 mod 4)  : {match}")
    print(f"  nu_f = log p (first three)  : {np.round(nu_f_paley[:3], 4).tolist()}")
    print("  HONEST LIMIT: the Paley gap covers the == 1 (mod 4) class only; the")
    print("  == 3 (mod 4) primes (and 2) need a complementary construction.")
    ok = match and nu_f_paley.size > 0
    print(
        f"  VERDICT: {'PASS' if ok else 'FAIL'} -- "
        f"{'finite selected labels match the supplied comparison set' if ok else 'support mismatch'}"
    )
    print()
    return ok


# --------------------------------------------------------------------------- #
# TEST 4 -- regular-graph random-walk/combinatorial rescaling
# --------------------------------------------------------------------------- #
def test_canonical_rw_equivalence(limit: int = 200) -> bool:
    print("=" * 78)
    print("TEST 4 -- random-walk Laplacian L_rw = I - D^-1 W versus")
    print("          the combinatorial L = D - A: equivalence by the degree factor d")
    print("=" * 78)

    print("  Paley circulants are REGULAR (deg d = (n-1)/2 for prime n == 1 mod 4), so")
    print("  L_rw = (1/d) L EXACTLY: same eigenvectors and ordering, eigenvalues / d.")
    print(
        "  Canonical closed form: lambda_2(L_rw) = sqrt n/(sqrt n+1) = 1 - 1/(sqrt n+1)."
    )
    print()
    print(
        "      n | deg d |  lam2(L) comb | lam2(L_rw) can | sqrt/(sqrt+1) |"
        "   g_comb |     g_rw"
    )
    print("  " + "-" * 82)

    worst_rescale = 0.0
    worst_canon_form = 0.0
    worst_closed = 0.0
    for n in (5, 13, 17, 29, 37):
        d = (n - 1) / 2.0
        lam_comb = lambda2_residue_fft(n)
        lam_rw = lambda2_residue_rw(n)
        ref_rw = paley_formula_rw(n)
        # (i) regular-graph rescaling identity: lambda_2(L) == d * lambda_2(L_rw)
        worst_rescale = max(worst_rescale, abs(lam_comb - d * lam_rw))
        # (ii) canonical closed form matches the measured L_rw Fiedler value
        worst_canon_form = max(worst_canon_form, abs(lam_rw - ref_rw))
        # (iii) closed forms are consistent: (n-sqrt n)/2 == d * sqrt n/(sqrt n+1)
        worst_closed = max(worst_closed, abs(paley_formula(n) - d * ref_rw))
        print(
            f"    {n:3d} | {d:5.1f} | {lam_comb:13.6f} | {lam_rw:14.9f} |"
            f" {ref_rw:13.9f} | {paley_gap(n):8.1e} | {paley_gap_rw(n):8.1e}"
        )

    # The closed-form value is in (0, 1) on these selected prime inputs.
    bounded = all(0.0 < lambda2_residue_rw(n) < 1.0 for n in (5, 13, 17, 29, 37))

    # Compare both numerical gap selections over the finite candidate range.
    comb_primes = [
        m for m in range(5, limit + 1) if m % 4 == 1 and paley_gap(m) <= _GAP_EPS
    ]
    rw_primes = [
        m for m in range(5, limit + 1) if m % 4 == 1 and paley_gap_rw(m) <= _GAP_EPS
    ]
    same_selection = comb_primes == rw_primes

    print()
    print(
        f"  max |lam2(L) - d*lam2(L_rw)|          : {worst_rescale:.2e}"
        "  (regular-graph identity L_rw = L/d)"
    )
    print(
        f"  max |lam2(L_rw) - sqrt n/(sqrt n+1)|  : {worst_canon_form:.2e}"
        "  (canonical closed form)"
    )
    print(
        f"  max |(n-sqrt n)/2 - d*sqrt n/(sqrt+1)|: {worst_closed:.2e}"
        "  (closed-form equivalence)"
    )
    print(
        f"  lam2(L_rw) in (0, 1)                  : {bounded}"
        "   (selected prime-case values)"
    )
    print(
        f"  same primes as combinatorial gap     : {same_selection}"
        f"   ({len(rw_primes)} primes == 1 mod 4)"
    )
    ok = (
        worst_rescale < 1e-9
        and worst_canon_form < 1e-9
        and worst_closed < 1e-9
        and bounded
        and same_selection
    )
    print(
        f"  VERDICT: {'PASS' if ok else 'FAIL'} -- "
        f"{'the selected regular-graph L_rw and combinatorial L agree up to degree d' if ok else 'mismatch'}"
    )
    print()
    return ok


def main() -> int:
    print(__doc__)
    checks = (
        ("finite gap classification", test_paley_gap_produces_primes()),
        ("selected symmetric spectra", test_paley_mechanism_is_real_self_adjoint()),
        ("supplied carrier-label comparison", test_paley_primes_ground_nu_f()),
        ("regular-graph Laplacian rescaling", test_canonical_rw_equivalence()),
    )
    print("=" * 78)
    print("FINITE COMPARISON SUMMARY")
    for label, passed in checks:
        print(f"  {label}: {'PASS' if passed else 'FAIL'}")
    print("  Supplied arithmetic and finite numerical checks do not establish a")
    print(
        "  universal primality criterion, nodal equilibrium or analytic phase bridge."
    )
    return 0 if all(passed for _, passed in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
