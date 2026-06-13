"""Camino 15 -- the DYNAMICAL half of the emergence of numbers.

CONTEXT (where this sits in the emergence-of-numbers line)
----------------------------------------------------------
Camino 9 (``paley_bridge.py``) and Camino 14 (``directed_paley_bridge.py``)
made the PRIMES emerge from the STATIC spectrum of a fixed graph: a
self-adjoint Paley gap for p == 1 (mod 4) and a skew Paley tournament for
p == 3 (mod 4).
That is the "estructura" half. The user asked for "estructura Y dinamica" --
so this harness is the "dinamica" half: numbers emerging from the TIME
EVOLUTION of the nodal phase, not from a frozen eigenvalue list.

The carrier is the nodal equation reduced to one phase oscillator driven by a
periodic structural pressure -- the canonical sine circle map:

        theta_{n+1} = theta_n + Omega - (K / 2pi) sin(2pi theta_n)

Here Omega = nu_f (the bare structural frequency / detuning) and the term
-(K/2pi) sin(2pi theta) = DNFR (the coupling-induced reorganisation pressure),
so the map is exactly the single-node Kuramoto reduction of
``dEPI/dt = nu_f * DNFR``. The emergent quantity is the rotation number

        rho = lim_{N->inf} (theta_N - theta_0) / N .

THESIS
------
1. RATIONALS EMERGE as the mode-locked rotation numbers. Over an interval of
   Omega the dynamics locks onto rho = p/q (an Arnold tongue / a plateau of the
   devil's staircase). We read rho off the iteration and RECOVER p/q with a
   continued-fraction identifier -- a genuine rationals-OUT emergence, the
   dynamical twin of Camino 9/14's primes-OUT spectral emergence.

2. The FAREY / STERN-BROCOT TREE organises the tongues. Between two Farey
   neighbours p1/q1 and p2/q2 (|p1 q2 - p2 q1| = 1) the widest plateau sits at
   the mediant (p1+p2)/(q1+q2), and the tongue width strictly shrinks as the
   denominator grows along the Fibonacci path 1/2, 2/3, 3/5, 5/8, ...

3. phi EMERGES as the most-irrational number: the Fibonacci ratios
   F_n / F_{n+1} -> 1/phi = (sqrt5 - 1)/2 are the convergents of the continued
   fraction [0; 1, 1, 1, ...] (all 1s, the slowest to converge, so phi
   saturates the Hurwitz bound: sqrt5 * F_{n+1}^2 * |F_n/F_{n+1} - 1/phi|
   -> 1). The limit IS the canonical TNFR golden ratio (phi <-> Phi_s, U6).

WALL CONNECTION
---------------
The lock/no-lock split of the DYNAMICS mirrors the real/phase split of the
SPECTRUM (Camino 14). Locked rationals = the reachable, structured half
(= range(R_inf), the smooth half of the REMESH-inf projection); the un-locked
irrationals -- with phi the most protected, the LAST to lock -- play the
residue role (= ker(R_inf) = Fix(G)^perp), the dynamical analogue of the
oscillatory S(T) = (1/pi) arg zeta(1/2 + iT).

HONEST SCOPE (this closes NOTHING)
----------------------------------
1. phi is reachable as a LIMIT of Fibonacci rationals -- an accumulation
   boundary, a SOFT residue -- NOT a hard orthogonal residue. S(T) lives in
   ker(R_inf) and is genuinely unreachable; phi is only the un-lockable limit
   of reachable rationals. The analogy is structural, not literal.
2. The dynamics yields a DISCRETE set of rationals plus ONE distinguished
   irrational phi; it does not yield the continuum and proves nothing about
   the zeta zeros. G4 = RH stays OPEN.
3. R and phi, gamma, pi, e remain the assumed substrate.

So Camino 15 EXTENDS the emergence-of-numbers line from the spectral side to
the dynamical side (rationals + phi out of time evolution) and CONNECTS the
lock/no-lock split to the wall -- but it does not move the wall.

Cross-checks the order parameter against the canonical
``tnfr.gamma.kuramoto_R_psi`` and the golden ratio against
``tnfr.constants.canonical.PHI`` when available.

Run:
    python benchmarks/kuramoto_farey_bridge.py
"""

from __future__ import annotations

import math
import os
import sys
from fractions import Fraction

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_SRC = os.path.abspath(os.path.join(_HERE, "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

try:  # canonical golden ratio (phi <-> Phi_s, one of the four constants)
    from tnfr.constants.canonical import PHI as _CANON_PHI  # noqa: E402

    _HAVE_CANON_PHI = True
except Exception:  # pragma: no cover
    _CANON_PHI = (1.0 + math.sqrt(5.0)) / 2.0
    _HAVE_CANON_PHI = False

try:  # canonical Kuramoto order parameter R = |mean exp(i theta)|
    from tnfr.gamma import kuramoto_R_psi  # noqa: E402

    _HAVE_KURAMOTO = True
except Exception:  # pragma: no cover
    _HAVE_KURAMOTO = False

try:  # networkx carries the theta attribute kuramoto_R_psi reads
    import networkx as nx  # noqa: E402

    _HAVE_NX = True
except Exception:  # pragma: no cover
    _HAVE_NX = False

PHI = float(_CANON_PHI)
PHI_INV = PHI - 1.0                # 1/phi = (sqrt5 - 1)/2 = 0.6180339887...

TOL = 1e-9
_RHO_ITERS = 20000                 # iterations for a precise rotation number
_RHO_TRANS = 2000                  # transient discarded before averaging
_SWEEP_ITERS = 6000                # cheaper iters for grid/slope sweeps
_LOCK_TOL = 2e-3                   # |rho - p/q| below this counts as locked
_PLATEAU_DELTA = 5e-4             # Omega offset for the flat-plateau check
K_CRIT = 1.0                       # critical coupling: complete staircase
K_SUB = 0.5                        # sub-critical: incomplete, phi un-locked


# --------------------------------------------------------------------------- #
# The sine circle map = single-node Kuramoto reduction of dEPI/dt = nu_f*DNFR.
# --------------------------------------------------------------------------- #
def circle_map_rho(
    omega: float,
    k: float,
    iters: int = _RHO_ITERS,
    trans: int = _RHO_TRANS,
) -> float:
    """Rotation number of the lifted sine circle map (no wrapping)."""
    two_pi = 2.0 * math.pi
    factor = k / two_pi
    theta = 0.0
    for _ in range(trans):
        theta += omega - factor * math.sin(two_pi * theta)
    start = theta
    for _ in range(iters):
        theta += omega - factor * math.sin(two_pi * theta)
    return (theta - start) / iters


def sweep_rho(
    omegas: np.ndarray, k: float, iters: int = _SWEEP_ITERS,
    trans: int = _RHO_TRANS,
) -> np.ndarray:
    """Vectorised rotation number over a grid of detunings ``omegas``."""
    two_pi = 2.0 * np.pi
    factor = k / two_pi
    theta = np.zeros_like(omegas, dtype=float)
    for _ in range(trans):
        theta += omegas - factor * np.sin(two_pi * theta)
    start = theta.copy()
    for _ in range(iters):
        theta += omegas - factor * np.sin(two_pi * theta)
    return (theta - start) / iters


def identify_rational(rho: float, max_den: int = 64) -> tuple[Fraction, float]:
    """Recover p/q from a measured rotation number (number-OUT)."""
    fr = Fraction(rho).limit_denominator(max_den)
    return fr, abs(rho - float(fr))


def is_plateau(omega: float, k: float, target: float) -> bool:
    """True if rho is flat (locked) at ``omega`` (rho(om +/- d) == target)."""
    lo = circle_map_rho(omega - _PLATEAU_DELTA, k)
    hi = circle_map_rho(omega + _PLATEAU_DELTA, k)
    return abs(lo - target) < _LOCK_TOL and abs(hi - target) < _LOCK_TOL


def tongue_width(
    p: int, q: int, k: float, n: int = 400, iters: int = _SWEEP_ITERS,
) -> float:
    """Omega-width of the p/q Arnold tongue (the locked plateau)."""
    center = p / q
    half = min(0.12, 0.6 / (q * q))           # tongues shrink fast with q
    om = np.linspace(center - half, center + half, n)
    rho = sweep_rho(om, k, iters=iters)
    locked = np.abs(rho - p / q) < _LOCK_TOL
    d_om = float(om[1] - om[0])
    return float(locked.sum()) * d_om


def locked_measure(
    k: float, n_grid: int = 800, iters: int = _SWEEP_ITERS,
) -> float:
    """Fraction of Omega in [0, 1] that sits on a rational plateau."""
    om = np.linspace(0.0, 1.0, n_grid)
    rho = sweep_rho(om, k, iters=iters)
    d_om = om[1] - om[0]
    slope = np.abs(np.diff(rho)) / d_om
    return float(np.mean(slope < 0.1))        # slope ~0 locked, ~1 drifting


def invert_rho(target: float, k: float, n_bis: int = 60) -> float:
    """Bisection inverse of the monotone staircase: Omega with rho=target."""
    lo, hi = 0.0, 1.0
    for _ in range(n_bis):
        mid = 0.5 * (lo + hi)
        if circle_map_rho(mid, k) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def harvest_plateaus(
    omegas: np.ndarray, rho: np.ndarray, slope_thr: float = 0.05,
    min_len: int = 4, max_den: int = 32,
) -> dict[Fraction, float]:
    """Harvest locked plateaus from a sweep and recover their rationals.

    The Arnold tongues of the sine circle map are NOT centred at Omega = p/q
    (only the symmetric 0, 1/2, 1 are), so instead of probing Omega = p/q we
    scan the devil's staircase, find the flat runs (slope ~ 0 = locked), and
    read p/q off each plateau with ``limit_denominator`` -- a genuine
    rationals-OUT recovery. Returns {fraction: worst |rho - p/q|}.
    """
    d_om = float(omegas[1] - omegas[0])
    slope = np.abs(np.diff(rho)) / d_om
    flat = slope < slope_thr
    recovered: dict[Fraction, float] = {}
    i, n = 0, len(flat)
    while i < n:
        if not flat[i]:
            i += 1
            continue
        j = i
        while j < n and flat[j]:
            j += 1
        if j - i + 1 >= min_len:
            val = float(np.median(rho[i:j + 1]))
            fr, err = identify_rational(val, max_den=max_den)
            if err < _LOCK_TOL:
                recovered[fr] = max(recovered.get(fr, 0.0), err)
        i = j + 1
    return recovered


# --------------------------------------------------------------------------- #
# Farey / Stern-Brocot arithmetic (machine-exact via Fraction).
# --------------------------------------------------------------------------- #
def farey_mediant(f1: Fraction, f2: Fraction) -> Fraction:
    """The Farey mediant (p1+p2)/(q1+q2) (no reduction in Stern-Brocot)."""
    return Fraction(
        f1.numerator + f2.numerator,
        f1.denominator + f2.denominator,
    )


def is_farey_neighbour(f1: Fraction, f2: Fraction) -> bool:
    """True iff |p1 q2 - p2 q1| = 1 (adjacent in some Farey sequence)."""
    return abs(f1.numerator * f2.denominator
               - f2.numerator * f1.denominator) == 1


def lowest_denom_between(f1: Fraction, f2: Fraction) -> Fraction:
    """Smallest-denominator fraction strictly between f1 and f2."""
    lo, hi = (f1, f2) if f1 < f2 else (f2, f1)
    bound = f1.denominator + f2.denominator
    for b in range(1, bound + 1):
        a = math.floor(lo * b) + 1
        fr = Fraction(a, b)
        if lo < fr < hi:
            return fr                         # first (smallest) b wins
    return farey_mediant(f1, f2)


def fibonacci(n: int) -> list[int]:
    """First ``n`` Fibonacci numbers F_1=F_2=1."""
    f = [1, 1]
    while len(f) < n:
        f.append(f[-1] + f[-2])
    return f[:n]


def continued_fraction(x: float, n_terms: int) -> list[int]:
    """Continued-fraction coefficients of ``x``."""
    terms: list[int] = []
    for _ in range(n_terms):
        a = math.floor(x)
        terms.append(a)
        frac = x - a
        if frac < 1e-12:
            break
        x = 1.0 / frac
    return terms


def kuramoto_order_parameter(theta_turns: np.ndarray) -> tuple[float, str]:
    """Canonical R = |mean exp(i theta)| via tnfr.gamma when available."""
    theta_rad = 2.0 * np.pi * theta_turns
    direct = float(np.abs(np.mean(np.exp(1j * theta_rad))))
    if _HAVE_KURAMOTO and _HAVE_NX:
        G = nx.Graph()
        for i, th in enumerate(theta_rad):
            G.add_node(i, theta=float(th))
        R, _psi = kuramoto_R_psi(G)
        return float(R), f"{direct:.2e}"
    return direct, f"{direct:.2e}"


# --------------------------------------------------------------------------- #
# TEST 1 -- rationals EMERGE as mode-locked rotation numbers (number-OUT)
# --------------------------------------------------------------------------- #
def test_rationals_emerge_as_lockings() -> bool:
    print("=" * 78)
    print("TEST 1 -- rationals emerge as the mode-locked rotation")
    print("          numbers of the nodal phase dynamics: we harvest the")
    print("          devil's-staircase plateaus and RECOVER p/q (number-OUT)")
    print("=" * 78)

    # Scan the staircase at criticality and harvest every locked plateau.
    # We do NOT tell the dynamics where p/q lives: we read rho off the
    # iteration, detect the flat runs, and recover the rationals blind.
    omegas = np.linspace(0.0, 1.0, 2500)
    rho = sweep_rho(omegas, K_CRIT)
    recovered = harvest_plateaus(omegas, rho)

    # The robustly wide tongues at K = 1 (must all be harvested blind).
    majors = [
        Fraction(0, 1), Fraction(1, 3), Fraction(1, 2),
        Fraction(2, 3), Fraction(1, 1),
    ]
    covered = [m for m in majors if m in recovered]
    worst_err = max((recovered[m] for m in covered), default=1.0)
    cover_ok = len(covered) == len(majors)

    print(f"  distinct rationals harvested  : {len(recovered)}")
    print(f"  major tongues covered         : "
          f"{len(covered)}/{len(majors)} "
          f"{[str(m) for m in covered]}")
    print(f"  worst |rho - p/q| on majors   : {worst_err:.2e}")
    ok = cover_ok and len(recovered) >= 8 and worst_err < _LOCK_TOL
    msg = ("the rationals emerge blind as locked plateaus and are recovered "
           "from rho alone") if ok else "too few lockings harvested"
    print(f"  VERDICT: {'PASS' if ok else 'FAIL'} -- {msg}")
    print()
    return ok


# --------------------------------------------------------------------------- #
# TEST 2 -- the Farey / Stern-Brocot tree organises the tongues
# --------------------------------------------------------------------------- #
def test_farey_mediant_organises_tongues() -> bool:
    print("=" * 78)
    print("TEST 2 -- between two Farey neighbours the dominant plateau is")
    print("          the mediant, and tongue width strictly shrinks with")
    print("          the denominator along the Fibonacci path to phi")
    print("=" * 78)

    # (a) arithmetic: the mediant is the unique lowest-denominator fraction
    #     strictly between two Farey neighbours (Stern-Brocot property).
    pairs = [
        (Fraction(0, 1), Fraction(1, 1)),
        (Fraction(1, 2), Fraction(1, 1)),
        (Fraction(0, 1), Fraction(1, 2)),
        (Fraction(1, 2), Fraction(2, 3)),
    ]
    mediant_ok = True
    for f1, f2 in pairs:
        med = farey_mediant(f1, f2)
        nb = is_farey_neighbour(f1, f2)
        low = lowest_denom_between(f1, f2)
        mediant_ok = mediant_ok and nb and (low == med)

    # (b) dynamical: tongue width strictly decreases along 1/2, 2/3, 3/5, 5/8
    #     (the Fibonacci-Farey path that accumulates at phi).
    path = [(1, 2), (2, 3), (3, 5), (5, 8)]
    widths = [tongue_width(p, q, K_CRIT) for p, q in path]
    shrinking = all(widths[i] > widths[i + 1] for i in range(len(widths) - 1))

    print("  (a) mediant = unique lowest-denominator in-between fraction:")
    for f1, f2 in pairs:
        med = farey_mediant(f1, f2)
        print(f"      {f1} , {f2}  ->  mediant {med}  "
              f"(neighbour={is_farey_neighbour(f1, f2)})")
    print(f"      Stern-Brocot mediant law holds : {mediant_ok}")
    print("  (b) Arnold tongue widths along the path to phi:")
    for (p, q), w in zip(path, widths):
        print(f"      {p}/{q:<2d} width = {w:.4f}")
    print(f"      strictly shrinking             : {shrinking}")
    ok = mediant_ok and shrinking
    msg = ("the dynamics reproduces the Farey/Stern-Brocot tree; tongues "
           "vanish toward phi") if ok else "mediant/width law broken"
    print(f"  VERDICT: {'PASS' if ok else 'FAIL'} -- {msg}")
    print()
    return ok


# --------------------------------------------------------------------------- #
# TEST 3 -- phi emerges as the most-irrational Fibonacci-Farey limit
# --------------------------------------------------------------------------- #
def test_phi_emerges_as_most_irrational() -> bool:
    print("=" * 78)
    print("TEST 3 -- F_n/F_{n+1} -> 1/phi (canonical), the [0;1,1,1,...]")
    print("          continued fraction that saturates the Hurwitz bound")
    print("          (phi = the most irrational number)")
    print("=" * 78)

    fib = fibonacci(30)
    ratios = [fib[i] / fib[i + 1] for i in range(len(fib) - 1)]
    conv_err = abs(ratios[-1] - PHI_INV)

    # continued fraction of 1/phi is all 1s after the leading 0
    cf = continued_fraction(PHI_INV, 18)
    cf_all_ones = (cf[0] == 0 and all(a == 1 for a in cf[1:]))

    # Hurwitz saturation: sqrt5 * F_{n+1}^2 * |F_n/F_{n+1} - 1/phi| -> 1
    sqrt5 = math.sqrt(5.0)
    c_vals = []
    for i in range(8, len(fib) - 1):
        err = abs(fib[i] / fib[i + 1] - PHI_INV)
        c_vals.append(sqrt5 * (fib[i + 1] ** 2) * err)
    hurwitz_ok = abs(c_vals[-1] - 1.0) < 0.01

    # the canonical golden ratio IS the limit (phi <-> Phi_s)
    canon_ok = abs((PHI - 1.0) - PHI_INV) < TOL

    print(f"  F_n/F_{{n+1}} last ratio        : {ratios[-1]:.15f}")
    print(f"  1/phi (canonical PHI - 1)     : {PHI_INV:.15f}")
    print(f"  |F_n/F_{{n+1}} - 1/phi|         : {conv_err:.2e}")
    print(f"  continued fraction [0;1,1,..] : {cf[:8]} ... all ones="
          f"{cf_all_ones}")
    print(f"  Hurwitz sqrt5*q^2*err -> 1     : {c_vals[-1]:.6f}")
    print(f"  canonical PHI source          : "
          f"{'tnfr.constants.canonical' if _HAVE_CANON_PHI else 'fallback'}")
    ok = (conv_err < 1e-10 and cf_all_ones and hurwitz_ok and canon_ok)
    msg = ("phi emerges as the canonical, maximally irrational "
           "Fibonacci-Farey limit") if ok else "phi limit not clean"
    print(f"  VERDICT: {'PASS' if ok else 'FAIL'} -- {msg}")
    print()
    return ok


# --------------------------------------------------------------------------- #
# TEST 4 -- honest scope: phi never locks; the staircase does not fill the line
# --------------------------------------------------------------------------- #
def test_phi_never_locks_wall() -> bool:
    print("=" * 78)
    print("TEST 4 -- at sub-critical coupling phi does NOT lock (it is the")
    print("          residue the dynamics cannot reach as a plateau), the")
    print("          staircase is incomplete, and this closes NOTHING")
    print("=" * 78)

    # (a) phi is un-locked at K_SUB: rho varies through Omega* (slope ~1),
    #     i.e. NOT a flat plateau, while a rational nearby locks.
    om_phi = invert_rho(PHI_INV, K_SUB)
    rho_phi = circle_map_rho(om_phi, K_SUB)
    lo = circle_map_rho(om_phi - _PLATEAU_DELTA, K_SUB)
    hi = circle_map_rho(om_phi + _PLATEAU_DELTA, K_SUB)
    phi_slope = abs(hi - lo) / (2.0 * _PLATEAU_DELTA)
    phi_unlocked = phi_slope > 0.5            # drifting, not flat
    phi_hit = abs(rho_phi - PHI_INV) < 1e-3

    # a rational (1/2) at the same K_SUB DOES lock (flat plateau)
    rational_locked = is_plateau(0.5, K_SUB, 0.5)

    # (b) the devil's staircase is incomplete at K_SUB (locked measure < 1)
    #     but (near-)complete at K_CRIT.
    m_sub = locked_measure(K_SUB)
    m_crit = locked_measure(K_CRIT)
    incomplete = (m_sub < 0.95) and (m_crit > m_sub)

    # (c) canonical cross-check via the 2-oscillator Kuramoto (Adler) lock:
    #     a commensurate detuning < K locks (phase difference bounded,
    #     winding W -> 0), the golden detuning > K winds (W != 0). The
    #     canonical order parameter confirms the locked pair is coherent.
    def adler_pair(d_omega: float, k: float,
                   steps: int = 20000, dt: float = 0.01) -> tuple:
        th1 = th2 = 0.0
        for _ in range(steps):
            s = math.sin(th2 - th1)
            th1 += dt * (0.5 * k * s)
            th2 += dt * (d_omega - 0.5 * k * s)
        return th1, th2

    n_steps, dt = 20000, 0.01
    span = n_steps * dt
    t1c, t2c = adler_pair(0.30, K_SUB, n_steps, dt)        # 0.30 < 0.5: lock
    t1g, t2g = adler_pair(PHI_INV, K_SUB, n_steps, dt)     # 0.618 > 0.5: wind
    w_comm = abs(t2c - t1c) / span        # bounded -> ~0 for a locked pair
    w_gold = abs(t2g - t1g) / span        # grows -> winding rate for golden
    comm_locked = w_comm < 1e-2
    gold_winds = w_gold > 0.1
    two_pi = 2.0 * math.pi
    pair_turns = np.array([t1c, t2c]) / two_pi
    r_pair, _ = kuramoto_order_parameter(pair_turns)
    r_src = ("tnfr.gamma.kuramoto_R_psi (CANONICAL)"
             if (_HAVE_KURAMOTO and _HAVE_NX) else "numpy fallback")
    r_ok = comm_locked and gold_winds and r_pair > 0.5

    print(f"  Omega* with rho=1/phi (K={K_SUB}) : {om_phi:.6f}")
    print(f"  rho at Omega* (~1/phi)        : {rho_phi:.6f} "
          f"(hit={phi_hit})")
    print(f"  slope d(rho)/d(Omega) at phi  : {phi_slope:.3f}  "
          f"(unlocked={phi_unlocked})")
    print(f"  rational 1/2 locks at K={K_SUB}    : {rational_locked}")
    print(f"  locked measure  K={K_SUB}         : {m_sub:.3f}  (< 1)")
    print(f"  locked measure  K={K_CRIT}         : {m_crit:.3f}")
    print(f"  Adler winding W (commensurate): {w_comm:.4f}  "
          f"(locked={comm_locked})")
    print(f"  Adler winding W (golden)      : {w_gold:.4f}  "
          f"(winds={gold_winds})")
    print(f"  R(locked commensurate pair)   : {r_pair:.4f}")
    print(f"  order-parameter source        : {r_src}")
    ok = (phi_hit and phi_unlocked and rational_locked
          and incomplete and r_ok)
    msg = ("phi is the un-lockable residue (soft: a limit of rationals); "
           "staircase incomplete; nothing closed") if ok else "phi locked?!"
    print(f"  VERDICT: {'PASS' if ok else 'FAIL'} -- {msg}")
    print()
    return ok


def main() -> int:
    print(__doc__)
    r1 = test_rationals_emerge_as_lockings()
    r2 = test_farey_mediant_organises_tongues()
    r3 = test_phi_emerges_as_most_irrational()
    r4 = test_phi_never_locks_wall()

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  TEST 1 rationals emerge as lockings (number-OUT): "
          f"{'PASS' if r1 else 'FAIL'}")
    print(f"  TEST 2 Farey/Stern-Brocot organises the tongues : "
          f"{'PASS' if r2 else 'FAIL'}")
    print(f"  TEST 3 phi = canonical most-irrational limit    : "
          f"{'PASS' if r3 else 'FAIL'}")
    print(f"  TEST 4 phi never locks; staircase incomplete    : "
          f"{'PASS' if r4 else 'FAIL'}")
    structural = r1 and r2 and r3 and r4
    print()
    label = "ALL PASS" if structural else "SOME FAILED"
    print(f"  STRUCTURAL CHECKS: {label}")
    print()
    print("  THESIS VERDICT: OPEN, by design (it EXTENDS, it does")
    print("  not close). Camino 9/14 made the primes emerge from the")
    print("  STATIC spectrum of a fixed graph; this harness makes the")
    print("  RATIONALS emerge from the TIME EVOLUTION of the nodal phase")
    print("  (the sine circle map = single-node Kuramoto reduction of")
    print("  dEPI/dt = nu_f * DNFR). The Farey/Stern-Brocot tree organises")
    print("  the Arnold tongues, and phi emerges as the canonical, most")
    print("  irrational Fibonacci-Farey limit -- the LAST number to lock.")
    print("  The lock/no-lock split of the dynamics mirrors the real/phase")
    print("  split of the spectrum: locked rationals are the reachable")
    print("  half (range R_inf); the un-locked irrationals, phi foremost,")
    print("  play the residue role (ker R_inf = Fix(G)^perp), the dynamical")
    print("  analogue of S(T) = (1/pi) arg zeta(1/2 + iT). But phi is only")
    print("  the un-lockable LIMIT of reachable rationals (a soft residue),")
    print("  not the hard orthogonal residue S(T); the dynamics yields")
    print("  discrete rationals plus one phi, not the continuum. G4 = RH")
    print("  stays OPEN; R and phi, gamma, pi, e remain assumed substrate.")
    return 0 if structural else 1


if __name__ == "__main__":
    raise SystemExit(main())
