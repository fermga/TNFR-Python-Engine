"""Camino 16 -- the golden orbit meets the N15 R-infinity projector.

WHAT THIS IS (and is NOT)
-------------------------
This is a CONSISTENCY demonstration, not a mathematical advance. It does
NOT prove anything new and does NOT move G4 = RH. It feeds the dynamical
objects of Camino 15 (``kuramoto_farey_bridge.py``) through the ONE
canonical TNFR object that is an actual proven theorem -- the N15
REMESH-infinity orthogonal projector ``R_infinity``
(``theory/REMESH_INFINITY_DERIVATION.md``, Branch A) -- and checks where
they land. The point is to turn Camino 15's deliberately SOFT analogy
("phi never locks ~ the wall residue") into a PRECISE statement, and to
expose the analogy's honest LIMIT.

THE CANONICAL OBJECT
--------------------
N15 proves ``R_infinity`` is a bounded self-adjoint orthogonal projection
whose range is the resonant Fourier lattice ``{2 pi m / L}`` with
``L = lcm(tau_l, tau_g) = lcm(4, 8) = 8`` (the documented canonical pair).
The engine implements it as a DFT-bin mask in
``tnfr.riemann.split_residue_by_remesh_infinity``; P50
(``examples/77_remesh_infinity_residue_split_demo.py``) already used it to
show the prime-ladder reconstruction ``S_TNFR(T)`` lives in
``ker(R_infinity)`` (because ``{k log p}`` is incommensurate with the
pi-lattice, Baker's theorem).

THE DYNAMICAL OBJECTS
---------------------
The sine circle map ``theta_{n+1} = theta_n + Omega - (K/2pi)
sin(2pi theta_n)`` -- the single-node Kuramoto reduction of
``dEPI/dt = nu_f * DNFR`` (Camino 15) -- produces orbits with a rotation
number ``rho``. We build the real signal ``cos(2pi theta_n)`` (demeaned,
so the trivially-resonant DC bin is excluded, exactly as P50's S_TNFR is
zero-mean) and project it with the canonical ``R_infinity``.

THESIS (all four checks pass; closes NOTHING)
---------------------------------------------
1. The GOLDEN (quasi-periodic) orbit ``rho = 1/phi`` lies in
   ``ker(R_infinity)``: its incommensurate frequency ``2pi/phi`` misses the
   resonant lattice ``2pi m / 8``. This is the DYNAMICAL twin of P50's
   ``S_TNFR in ker`` -- the same kernel, reached by a different
   incommensurate carrier (phi here, log p there).

2. A REMESH-COMMENSURATE locking (period ``q | L = 8``: ``rho = 1/2``,
   ``rho = 1/4``) lies in ``range(R_infinity)``: all its harmonics sit on
   the resonant lattice.

3. HONEST LIMIT. A locking whose period does NOT divide ``L = 8``
   (``rho = 1/3``, period 3) is genuinely LOCKED yet still lands in
   ``ker(R_infinity)``. So Camino 15's lock/no-lock dichotomy does NOT map
   one-to-one onto N15's range/ker split: ``R_infinity``'s lattice is
   COARSER than the full Farey set of lockings. ``range(R_infinity)`` is
   the period-divides-L sub-lattice only; everything else -- un-locked
   irrationals (phi) AND locked rationals with period coprime-ish to L --
   is in the kernel. This SHARPENS the analogy and LIMITS it at once.

4. Canonical cross-check + scope. The engine's own controls reproduce
   (``sin(2pi T / L) -> range``, ``sin(gamma T) -> ker``), and the P50
   certificate confirms the prime-ladder ``S_TNFR in ker``. The kernel is
   large (its complement, the resonant lattice, is measure-zero among all
   frequencies); it holds BOTH the arithmetic residue ``S(T)`` carrier and
   the golden orbit. Membership in ``ker(R_infinity)`` LOCATES the residue;
   it is NOT a route to RH.

HONEST SCOPE
------------
This closes NOTHING. It makes the Camino-15 / N15 connection precise and
exposes its limit. ``G4 = RH`` stays OPEN; ``R`` and ``pi``
remain the assumed substrate. The whole construction lives inside the
13-operator catalog (R_infinity is derivable from REMESH, N15 Branch A);
no new operator is introduced.

Run:
    python benchmarks/golden_residue_remesh_bridge.py
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "src"))
for _p in (_HERE, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Sibling Camino-15 harness: the circle-map dynamics (always present).
from kuramoto_farey_bridge import (  # noqa: E402
    K_SUB,
    PHI_INV,
    circle_map_rho,
    invert_rho,
    sweep_rho,
)

# Canonical N15 R_infinity projector (the subject of this harness).
try:
    from tnfr.riemann import (
        compute_residue_split_certificate,
        split_residue_by_remesh_infinity,
    )

    _HAVE_RINF = True
except Exception:  # pragma: no cover - canonical engine optional
    _HAVE_RINF = False


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
TAU_L, TAU_G = 4, 8  # documented canonical REMESH pair
LCM_L = math.lcm(TAU_L, TAU_G)  # L = 8: resonant lattice 2*pi*m / L
# 960 = 24 * 40: multiple of L = 8 (mask) AND of lcm(2,3,4) = 12 (clean
# periods for the period-2/3/4 locked orbits, no spectral leakage).
N_SAMPLES = 960
_TRANSIENT = 6000  # discard before sampling the orbit
K_LOCK = 0.9  # near-critical: wide, clean tongues
THRESH = 0.05  # P50 certificate decision threshold
_GAMMA_EM = 0.5772156649015329  # Euler-Mascheroni (non-resonant ctrl)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def circle_map_orbit(
    omega: float,
    k: float,
    n: int = N_SAMPLES,
    trans: int = _TRANSIENT,
) -> np.ndarray:
    """Demeaned ``cos(2*pi*theta_n)`` signal of the lifted circle map.

    The lift ``theta_n`` grows without wrapping; ``cos(2*pi*theta_n)``
    folds it onto the circle. We subtract the mean so the trivially
    resonant DC bin (k = 0) is excluded -- matching P50, whose S_TNFR is
    a sum of sines (zero mean).
    """
    two_pi = 2.0 * math.pi
    factor = k / two_pi
    theta = 0.0
    for _ in range(trans):
        theta += omega - factor * math.sin(two_pi * theta)
    sig = np.empty(n, dtype=float)
    for j in range(n):
        theta += omega - factor * math.sin(two_pi * theta)
        sig[j] = math.cos(two_pi * theta)
    sig -= sig.mean()
    return sig


def orbit_period(
    sig: np.ndarray,
    max_q: int = 16,
    tol: float = 1e-6,
) -> int:
    """Smallest period ``q <= max_q`` of ``sig`` (0 if quasi-periodic)."""
    m = min(256, sig.size // 2)
    head = sig[:m]
    for q in range(1, max_q + 1):
        if np.max(np.abs(sig[q : q + m] - head)) < tol:
            return q
    return 0


def tongue_centre(
    target: float,
    k: float,
    span: float = 0.05,
    n: int = 4001,
    lock_tol: float = 2e-3,
) -> float:
    """Centre of the Arnold tongue holding ``rho = target`` at coupling k.

    ``invert_rho`` lands on a tongue EDGE (marginal locking, long
    transients); we scan a window around it, keep the locked points
    (``|rho - target| < lock_tol``) and return their median Omega -- a
    robustly INTERIOR, cleanly periodic detuning.
    """
    om0 = invert_rho(target, k)
    lo = max(0.0, om0 - span)
    hi = min(1.0, om0 + span)
    oms = np.linspace(lo, hi, n)
    rhos = sweep_rho(oms, k)
    locked = np.abs(rhos - target) < lock_tol
    if not locked.any():
        return om0
    return float(np.median(oms[locked]))


def range_kernel_fractions(sig: np.ndarray) -> tuple[float, float, str]:
    """Energy fractions of ``sig`` in range / ker of canonical R_infinity.

    Uses the engine's ``split_residue_by_remesh_infinity`` when present;
    otherwise a NumPy DFT-bin-mask fallback identical to the canonical
    construction. Fractions are squared-norm (Parseval) and sum to 1.
    """
    if _HAVE_RINF:
        rng, ker = split_residue_by_remesh_infinity(sig, tau_l=TAU_L, tau_g=TAU_G)
        src = "tnfr.riemann.split_residue_by_remesh_infinity (CANONICAL)"
    else:
        period = math.lcm(TAU_L, TAU_G)
        step = sig.size // period
        mask = np.zeros(sig.size, dtype=bool)
        mask[::step] = True
        spec = np.fft.fft(sig)
        rng = np.real(np.fft.ifft(np.where(mask, spec, 0.0 + 0.0j)))
        ker = sig - rng
        src = "numpy DFT-bin-mask fallback"
    total = float(np.linalg.norm(sig))
    if total <= 0.0:
        return 0.0, 0.0, src
    r = (float(np.linalg.norm(rng)) / total) ** 2
    kf = (float(np.linalg.norm(ker)) / total) ** 2
    return r, kf, src


# --------------------------------------------------------------------------- #
# TEST 1 -- the golden orbit lies in ker(R_infinity)
# --------------------------------------------------------------------------- #
def test_golden_orbit_in_kernel() -> bool:
    print("=" * 78)
    print("TEST 1 -- the golden (quasi-periodic) orbit rho = 1/phi lies")
    print("          in ker(R_infinity): its incommensurate frequency")
    print("          misses the resonant lattice 2*pi*m / 8 (= P50's twin)")
    print("=" * 78)

    om = invert_rho(PHI_INV, K_SUB)
    rho = circle_map_rho(om, K_SUB)
    sig = circle_map_orbit(om, K_SUB)
    q = orbit_period(sig)
    r_frac, k_frac, src = range_kernel_fractions(sig)

    hit = abs(rho - PHI_INV) < 1e-3
    quasi = q == 0  # no small period = quasi-periodic
    in_kernel = r_frac < THRESH

    print(f"  Omega* (rho = 1/phi, K={K_SUB})  : {om:.6f}")
    print(f"  measured rho                  : {rho:.6f} (hit={hit})")
    print(f"  detected period (0=quasi)     : {q}  (quasi={quasi})")
    print(f"  range(R_inf) fraction         : {100.0 * r_frac:7.4f} %")
    print(f"  ker(R_inf)   fraction         : {100.0 * k_frac:7.4f} %")
    print(f"  projector source              : {src}")
    ok = hit and quasi and in_kernel
    msg = (
        ("the golden orbit is in ker(R_inf), the dynamical twin of " "P50's S_TNFR")
        if ok
        else "golden orbit not cleanly in kernel"
    )
    print(f"  VERDICT: {'PASS' if ok else 'FAIL'} -- {msg}")
    print()
    return ok


# --------------------------------------------------------------------------- #
# TEST 2 -- REMESH-commensurate lockings lie in range(R_infinity)
# --------------------------------------------------------------------------- #
def test_commensurate_lockings_in_range() -> bool:
    print("=" * 78)
    print("TEST 2 -- a locking whose period divides L = 8 (rho = 1/2,")
    print("          rho = 1/4) lies in range(R_infinity): every harmonic")
    print("          sits on the resonant lattice")
    print("=" * 78)

    cases = [(1, 2), (1, 4)]  # periods 2 and 4, both | 8
    all_ok = True
    for p, qd in cases:
        target = p / qd
        om = tongue_centre(target, K_LOCK)
        sig = circle_map_orbit(om, K_LOCK)
        q = orbit_period(sig)
        r_frac, k_frac, _ = range_kernel_fractions(sig)
        locked = q == qd
        in_range = r_frac > (1.0 - THRESH)
        ok = locked and in_range
        all_ok = all_ok and ok
        print(
            f"  rho = {p}/{qd}: Omega*={om:.5f} period={q} "
            f"range={100.0 * r_frac:6.2f}% ker={100.0 * k_frac:6.2f}% "
            f"-> {'range' if in_range else 'NOT range'}"
        )
    msg = (
        "commensurate lockings (period | 8) land in range(R_inf)"
        if all_ok
        else "a commensurate locking missed range"
    )
    print(f"  VERDICT: {'PASS' if all_ok else 'FAIL'} -- {msg}")
    print()
    return all_ok


# --------------------------------------------------------------------------- #
# TEST 3 -- HONEST LIMIT: a locked rho=1/3 (period 3 does NOT divide 8)
#           is still in ker(R_infinity); the lattice is coarser
# --------------------------------------------------------------------------- #
def test_lattice_coarser_than_lockings() -> bool:
    print("=" * 78)
    print("TEST 3 -- HONEST LIMIT: rho = 1/3 is genuinely LOCKED (period 3)")
    print("          yet 3 does NOT divide L = 8, so it lands in")
    print("          ker(R_infinity): R_inf's lattice is coarser than the")
    print("          Farey set of lockings (analogy is partial)")
    print("=" * 78)

    target = 1.0 / 3.0
    om = tongue_centre(target, K_LOCK)
    sig = circle_map_orbit(om, K_LOCK)
    q = orbit_period(sig)
    r_frac, k_frac, _ = range_kernel_fractions(sig)

    genuinely_locked = q == 3
    in_kernel = r_frac < THRESH

    print(f"  rho = 1/3: Omega*             : {om:.6f}")
    print(f"  detected period               : {q}  " f"(locked={genuinely_locked})")
    print(f"  range(R_inf) fraction         : {100.0 * r_frac:7.4f} %")
    print(f"  ker(R_inf)   fraction         : {100.0 * k_frac:7.4f} %")
    print("  => LOCKED in the dynamics, yet in ker(R_inf): the lock/")
    print("     no-lock split does NOT equal the range/ker split.")
    ok = genuinely_locked and in_kernel
    msg = (
        (
            "locked-but-incommensurate orbit is in ker: R_inf lattice "
            "coarser than lockings"
        )
        if ok
        else "1/3 did not behave"
    )
    print(f"  VERDICT: {'PASS' if ok else 'FAIL'} -- {msg}")
    print()
    return ok


# --------------------------------------------------------------------------- #
# TEST 4 -- canonical R_infinity controls + P50 reconciliation + scope
# --------------------------------------------------------------------------- #
def test_canonical_crosscheck_and_scope() -> bool:
    print("=" * 78)
    print("TEST 4 -- canonical R_infinity controls + P50 reconciliation:")
    print("          the same kernel holds the prime residue AND the")
    print("          golden orbit; ker membership LOCATES, never closes")
    print("=" * 78)

    t = np.arange(N_SAMPLES, dtype=float)
    # Positive control: a pure resonant tone sin(2*pi*T / L) -> range.
    omega_res = 2.0 * math.pi / LCM_L
    ctrl_res = np.sin(omega_res * t)
    r_res, _, src = range_kernel_fractions(ctrl_res)
    # Negative control: sin(gamma * T), gamma transcendental -> ker.
    ctrl_non = np.sin(_GAMMA_EM * t)
    r_non, k_non, _ = range_kernel_fractions(ctrl_non)

    res_ok = r_res > (1.0 - THRESH)
    non_ok = k_non > (1.0 - THRESH)

    print(
        f"  control sin(2*pi*T/{LCM_L}) range : "
        f"{100.0 * r_res:7.4f} %  (expect ~100, ok={res_ok})"
    )
    print(
        f"  control sin(gamma*T)   ker    : "
        f"{100.0 * k_non:7.4f} %  (expect ~100, ok={non_ok})"
    )
    print(f"  projector source              : {src}")

    # P50 reconciliation: the canonical prime-ladder S_TNFR in ker.
    p50_ok = True
    if _HAVE_RINF:
        cert = compute_residue_split_certificate(
            n_primes=200, max_power=8, n_periods=64
        )
        p50_ok = cert.verdict == "RESIDUE_IN_KER_ONLY"
        print(
            f"  P50 prime-ladder S_TNFR       : {cert.verdict} "
            f"(ker={100.0 * cert.ratio_in_kernel:.2f}%)"
        )
        print("  => SAME kernel as the golden orbit (Test 1): log p")
        print("     (Baker) and 1/phi are two incommensurate carriers of")
        print("     the ONE residue subspace ker(R_inf).")
    else:
        print("  P50 certificate              : skipped (engine absent)")

    ok = res_ok and non_ok and p50_ok
    msg = (
        (
            "canonical R_inf confirmed; residue LOCATED in ker, NOT "
            "closed (G4 = RH OPEN)"
        )
        if ok
        else "canonical control failed"
    )
    print(f"  VERDICT: {'PASS' if ok else 'FAIL'} -- {msg}")
    print()
    return ok


def main() -> int:
    print(__doc__)
    r1 = test_golden_orbit_in_kernel()
    r2 = test_commensurate_lockings_in_range()
    r3 = test_lattice_coarser_than_lockings()
    r4 = test_canonical_crosscheck_and_scope()

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(
        f"  TEST 1 golden orbit in ker(R_inf)        : " f"{'PASS' if r1 else 'FAIL'}"
    )
    print(
        f"  TEST 2 commensurate lockings in range    : " f"{'PASS' if r2 else 'FAIL'}"
    )
    print(
        f"  TEST 3 HONEST LIMIT: lattice coarser     : " f"{'PASS' if r3 else 'FAIL'}"
    )
    print(
        f"  TEST 4 canonical R_inf + P50 + scope     : " f"{'PASS' if r4 else 'FAIL'}"
    )
    print()
    all_ok = r1 and r2 and r3 and r4
    print("  STRUCTURAL CHECKS: " f"{'ALL PASS' if all_ok else 'SOME FAILED'}")
    print()
    print("  THESIS VERDICT: OPEN, by design (it CONNECTS, it")
    print("  does not close). Camino 15 made the rationals + phi")
    print("  emerge from the TIME EVOLUTION of the nodal phase and")
    print("  noted, as a SOFT analogy, that phi (the last to lock)")
    print("  plays the wall-residue role. This harness makes that")
    print("  precise through the ONE canonical proven object, the")
    print("  N15 R_infinity orthogonal projector: the golden orbit")
    print("  is in ker(R_inf) (Test 1, the dynamical twin of P50's")
    print("  prime-ladder S_TNFR in ker), and REMESH-commensurate")
    print("  lockings are in range (Test 2). But the map is PARTIAL")
    print("  (Test 3): a locked rho = 1/3 with period 3 does NOT")
    print("  divide L = 8, so it sits in ker too -- R_inf's lattice")
    print("  is coarser than the Farey set of lockings. The kernel")
    print("  is large and holds BOTH the arithmetic residue carrier")
    print("  (log p, Baker) and the golden orbit; ker membership")
    print("  LOCATES the residue, it is NOT a route to RH. G4 = RH")
    print("  stays OPEN; R and pi remain substrate.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
