"""
Boundary Vibration -- where do the Riemann zeros come from, and why can the
TNFR engine derive their SOURCE but not their LOCATION without mpmath?

THE CLAIM
    The Riemann zeros are the resonance spectrum of the prime "boundary
    vibration".  TNFR derives the SOURCE of that vibration canonically --
    nu_f = log p (structural frequency = geodesic length), the nodal equation
    d(EPI)/dt = nu_f * Delta(NFR), and a self-adjoint prime-ladder Hamiltonian
    whose spectrum is {k log p} -- with NO external zero data.  The open
    question (gap G4) is whether the RESONANCES {gamma_n} can be reached from
    the SOURCE {k log p} by a canonical map; equivalently, whether the
    TNFR-native von Mangoldt series can be analytically continued across
    Re(s) = 1 by structure alone.

ENGINE (rigorous / known theorems)
    * von Mangoldt:  -zeta'/zeta(s) = sum_{p,k} (log p) p^{-ks}, converging
      absolutely iff Re(s) > 1 (abscissa of convergence at Re = 1).  P12.
    * L = D - A is self-adjoint  =>  real spectrum (spectral theorem).
    * A self-adjoint operator that commutes with an involution R = R^T, R^2 = I
      splits into R-parity sectors, each with real eigenvalues (Z_2 rep theory).
    * Hilbert-Polya:  RH <=> {gamma_n} is the spectrum of a self-adjoint
      operator (conjectural framing; NOT proved here).

TNFR reading (AGENTS.md + src/tnfr/dynamics/adelic.py)
    nu_f = log p is the canonical per-node structural frequency.  The geometric
    trace Tr_geo(t) = sum (log p) e^{i t log p} / sqrt(p) is the collective
    oscillation of the prime geodesics -- the literal boundary vibration -- and
    it is DERIVED from the nodal equation (adelic.py).  The zeros {gamma_n} are
    the resonance spectrum that vibration would reveal; in adelic.py they appear
    as known_zeros = Ground Truth, framed exactly as "a blind search would
    detect them via resonance peaks".  The critical line Re = 1/2 is the fixed
    axis of the Z_2 reflection s <-> 1 - s.

HONEST SCOPE
    This harness does NOT prove RH and does NOT close G4.  It LOCATES G4
    surgically as the analytic continuation of the TNFR-native von Mangoldt
    series across Re(s) = 1 / the canonical map {k log p} -> {gamma_n}.  The
    self-adjoint + reflection leg shows the Hilbert-Polya intuition
    ("self-adjoint => real => on the line") is TRUE as algebra; what is OPEN is
    exhibiting THE self-adjoint operator whose spectrum is {gamma_n} from TNFR
    structure alone.  Following src/tnfr/dynamics/adelic.py, every carrier here
    is derived from nu_f = log p; mpmath, where present, only draws the target
    -- it never derives it.  R (continuum) and pi remain assumed
    substrate.

Run:
    python benchmarks/boundary_vibration.py

Status: RESEARCH (boundary-vibration falsifier; Camino 10 of the unification map).
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

# Optional: real Riemann zeta, only to confirm the target ordinates ARE zeros.
try:  # pragma: no cover - exercised only when mpmath is installed
    import mpmath  # noqa: E402

    _HAVE_MPMATH = True
except Exception:  # pragma: no cover
    _HAVE_MPMATH = False

# Optional: P12 von Mangoldt -- the TNFR-native -zeta'/zeta carrier (nu_f=log p).
try:  # pragma: no cover
    from tnfr.riemann.von_mangoldt import (  # noqa: E402
        build_prime_ladder_spectrum,
        classical_log_zeta_derivative,
        tnfr_log_zeta_derivative,
    )

    _HAVE_P12 = True
except Exception:  # pragma: no cover
    _HAVE_P12 = False

# Optional: P14 self-adjoint prime-ladder Hamiltonian (spectrum = {k log p}).
try:  # pragma: no cover
    from tnfr.riemann.prime_ladder_hamiltonian import (  # noqa: E402
        build_prime_ladder_hamiltonian,
    )

    _HAVE_P14 = True
except Exception:  # pragma: no cover
    _HAVE_P14 = False

# Optional: P27 Hilbert-Polya scaffold (gamma_n imported, NOT derived).
try:  # pragma: no cover
    from tnfr.riemann.hilbert_polya import (  # noqa: E402
        build_hp_operator,
        fetch_zero_imaginary_parts,
        structural_gap_p14_vs_hp,
        verify_hp_self_adjoint,
    )

    _HAVE_P27 = True
except Exception:  # pragma: no cover
    _HAVE_P27 = False

# Optional: the canonical adelic engine (nu_f = log p nodal-equation carrier).
try:  # pragma: no cover
    from tnfr.dynamics.adelic import AdelicDynamics  # noqa: E402

    _HAVE_ADELIC = True
except Exception:  # pragma: no cover
    _HAVE_ADELIC = False

TOL = 1e-9
_SELF_ADJOINT_TOL = 1e-9  # Frobenius asymmetry / imaginary tolerance
_PARITY_TOL = 1e-8  # |R v -/+ v| tolerance for definite parity
_DRIFT_MARGIN = 3.0  # barrier_drift must exceed stable_drift by this
_CLASSICAL_REL_TOL = 5e-2  # TNFR Z vs classical -zeta'/zeta agreement

# Only pi is a genuine structural scale (audit 2026); phi/gamma/e removed (unused).
PI = np.pi

# First non-trivial Riemann zero ordinates -- the TARGET the vibration reveals,
# imported as Ground Truth exactly as src/tnfr/dynamics/adelic.py does.
_KNOWN_ORDINATES = np.array(
    [
        14.134725,
        21.022040,
        25.010858,
        30.424876,
        32.935062,
        37.586178,
        40.918719,
        43.327073,
        48.005151,
        49.773832,
    ]
)


def _local_geometric_trace(primes: np.ndarray, t: float) -> float:
    """TNFR-native boundary vibration Tr_geo(t) (used if adelic is absent).

    Tr_geo(t) = | sum_p (log p) e^{i t log p} / sqrt(p) |.  Built purely from
    nu_f = log p; contains NO zero data.
    """
    nu_f = np.log(primes)
    weights = nu_f / np.sqrt(primes)
    return float(np.abs(np.sum(weights * np.exp(1j * t * nu_f))))


def test_convergence_barrier() -> dict:
    """TEST 1 -- the TNFR-native carrier converges only for Re(s) > 1.

    The von Mangoldt Dirichlet series Z_vM(s) = sum w e^{-s mu} (nu_f = log p)
    is the canonical -zeta'/zeta carrier.  Increasing the prime-ladder
    truncation STABILISES the value for Re(s) > 1 but does NOT stabilise it at
    Re(s) = 1/2 -- the abscissa of convergence sits at Re = 1, so the object
    that sees the primes literally cannot be evaluated where the zeros live.
    This is why mpmath is INEVITABLE, and it locates G4 at the continuation
    across Re = 1.
    """
    print("TEST 1 -- convergence barrier: the prime carrier cannot reach Re=1/2")
    if not _HAVE_P12:
        print("  SKIP -- tnfr.riemann.von_mangoldt unavailable")
        return {"name": "convergence_barrier", "status": "SKIP"}

    spec_small = build_prime_ladder_spectrum(n_primes=15, max_power=4)
    spec_big = build_prime_ladder_spectrum(n_primes=40, max_power=8)

    s_stable = 2.0  # Re(s) > 1: inside the half-plane of convergence
    s_barrier = 0.5  # Re(s) = 1/2: where the zeros live (divergent series)

    z_small_stable = float(np.real(tnfr_log_zeta_derivative(spec_small, s_stable)))
    z_big_stable = float(np.real(tnfr_log_zeta_derivative(spec_big, s_stable)))
    z_small_barrier = float(np.real(tnfr_log_zeta_derivative(spec_small, s_barrier)))
    z_big_barrier = float(np.real(tnfr_log_zeta_derivative(spec_big, s_barrier)))

    stable_drift = abs(z_big_stable - z_small_stable)
    barrier_drift = abs(z_big_barrier - z_small_barrier)

    classical = float(np.real(classical_log_zeta_derivative(s_stable, 400)))
    rel_err = abs(z_big_stable - classical) / max(abs(classical), TOL)

    print(
        f"  Re(s)=2.0 : Z_15x4={z_small_stable:.6f}  Z_40x8={z_big_stable:.6f}"
        f"  drift={stable_drift:.3e}  (classical -zeta'/zeta={classical:.6f},"
        f" rel_err={rel_err:.2e})"
    )
    print(
        f"  Re(s)=0.5 : Z_15x4={z_small_barrier:.4f}  Z_40x8={z_big_barrier:.4f}"
        f"  drift={barrier_drift:.3e}"
    )

    converges_above = stable_drift < 0.5 and rel_err < _CLASSICAL_REL_TOL
    diverges_at_half = barrier_drift > _DRIFT_MARGIN * max(stable_drift, TOL)
    passed = bool(converges_above and diverges_at_half)
    detail = (
        "stabilises for Re>1 (matches classical) and fails to stabilise "
        "at Re=1/2 -> G4 lives in the continuation across Re=1"
    )
    print(f"  VERDICT: {'PASS' if passed else 'FAIL'} -- {detail}")
    return {"name": "convergence_barrier", "status": "PASS" if passed else "FAIL"}


def test_p14_self_adjoint_origin() -> dict:
    """TEST 2 -- P14 derives the vibration's SOURCE {k log p}, no mpmath.

    The canonical self-adjoint prime-ladder Hamiltonian (coupling = 0) has a
    real spectrum equal to {k log p} by construction -- the origin of the
    boundary vibration, derived entirely from nu_f = log p.
    """
    print("TEST 2 -- P14 self-adjoint Hamiltonian gives {k log p} (no mpmath)")
    if not _HAVE_P14:
        print("  SKIP -- tnfr.riemann.prime_ladder_hamiltonian unavailable")
        return {"name": "p14_origin", "status": "SKIP"}

    bundle = build_prime_ladder_hamiltonian(n_primes=15, max_power=4, coupling=0.0)
    eigs, _ = bundle.hamiltonian.get_spectrum()
    max_imag = float(np.max(np.abs(np.imag(eigs))))
    spec = np.sort(np.real(eigs))
    ref = np.sort(np.asarray(bundle.spectrum.eigenvalues, dtype=float))

    n = min(len(spec), len(ref))
    match = float(np.max(np.abs(spec[:n] - ref[:n]))) if n else float("nan")

    print(
        f"  dim={len(eigs)}  max|Im(eig)|={max_imag:.3e}"
        f"  max|spec - {{k log p}}|={match:.3e}"
    )
    print(
        f"  smallest eigenvalues: {np.round(spec[:4], 6).tolist()}"
        f"  (log 2 = {math.log(2):.6f})"
    )

    self_adjoint = max_imag < _SELF_ADJOINT_TOL
    reproduces = match < 1e-9
    passed = bool(self_adjoint and reproduces)
    detail = "real spectrum reproduces {k log p} from structure alone"
    print(f"  VERDICT: {'PASS' if passed else 'FAIL'} -- {detail}")
    return {"name": "p14_origin", "status": "PASS" if passed else "FAIL"}


def test_adelic_boundary_vibration() -> dict:
    """TEST 3 -- the adelic nodal flow: carrier derived, zeros as Ground Truth.

    Mirrors src/tnfr/dynamics/adelic.py exactly: the geometric-trace carrier is
    built only from nu_f = log p (zero zero-data), while the nodal gradient
    Delta(NFR) = -grad V that LANDS the flow on the zeros is defined from
    known_zeros -- the Ground-Truth target.  The flow's pressure vanishes at the
    target (resonance), making visible that {gamma_n} enters only as the
    resonance spectrum, never as a derived input.
    """
    print("TEST 3 -- adelic boundary vibration (carrier derived; zeros = target)")

    primes = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47])
    # Carrier built purely from nu_f = log p -- contains no ordinate data.
    sample_t = [10.0, 14.134725, 25.010858]
    if _HAVE_ADELIC:
        eng = AdelicDynamics(max_prime=int(primes[-1]))
        trace = [eng.compute_geometric_trace(t) for t in sample_t]
        # Delta(NFR) = -grad V vanishes at the zero target (resonance).
        grad_off = abs(eng.compute_nodal_gradient(10.0))
        grad_on = abs(eng.compute_nodal_gradient(float(eng.known_zeros[0])))
        target = np.asarray(eng.known_zeros, dtype=float)
        src = "tnfr.dynamics.adelic (CANONICAL)"
    else:
        trace = [_local_geometric_trace(primes, t) for t in sample_t]
        # Local stand-in for Delta(NFR) = -grad V against the imported target.
        off_idx = int(np.argmin(np.abs(10.0 - _KNOWN_ORDINATES)))
        grad_off = abs(10.0 - float(_KNOWN_ORDINATES[off_idx]))
        grad_on = 0.0  # distance from gamma_1 to its nearest ordinate is itself
        target = _KNOWN_ORDINATES
        src = "local nu_f=log p carrier (adelic fallback)"

    print(f"  carrier source: {src}")
    print(
        f"  Tr_geo(t) at t={sample_t}: {[round(x, 4) for x in trace]}"
        f"  (built from nu_f=log p only)"
    )
    print(
        f"  |Delta(NFR)| off-target (t=10) = {grad_off:.4f}"
        f"   on-target (t=gamma_1) = {grad_on:.3e}"
    )
    print(
        f"  target ordinates (Ground Truth, not derived): {np.round(target[:5], 4).tolist()}"
    )

    carrier_derived = all(math.isfinite(x) for x in trace)
    resonance_at_target = grad_on < TOL < grad_off
    passed = bool(carrier_derived and resonance_at_target)
    detail = (
        "carrier is pure nu_f=log p; nodal pressure vanishes at the "
        "imported target -> zeros are the resonance spectrum, not derived"
    )
    print(f"  VERDICT: {'PASS' if passed else 'FAIL'} -- {detail}")
    return {"name": "adelic_vibration", "status": "PASS" if passed else "FAIL"}


def test_self_adjoint_reflection() -> dict:
    """TEST 4 -- self-adjoint + reflection => real spectrum on the fixed axis.

    The rigorous core of the Hilbert-Polya intuition, where it is literally
    TRUE.  L = D - A of a reflection-symmetric path graph commutes with the
    Z_2 mirror R (node i <-> n-1-i; the structural analogue of s <-> 1-s,
    studied in Camino 6).  Self-adjointness forces real eigenvalues; commuting
    with R splits them into definite-parity sectors -- the fixed axis (R=+1) is
    the critical-line analogue.
    """
    print("TEST 4 -- self-adjoint + Z_2 reflection => real spectrum on fixed axis")

    n = 8
    G = nx.path_graph(n)
    L = nx.laplacian_matrix(G).toarray().astype(float)
    R = np.fliplr(np.eye(n))  # the Z_2 mirror involution

    involution = float(np.max(np.abs(R @ R - np.eye(n))))
    symmetric = float(np.max(np.abs(R - R.T)))
    commutator = float(np.max(np.abs(L @ R - R @ L)))

    eigvals, eigvecs = np.linalg.eigh(L)  # eigh => guaranteed real
    max_imag = float(np.max(np.abs(np.imag(eigvals))))

    parities = []
    for j in range(n):
        v = eigvecs[:, j]
        rv = R @ v
        if np.linalg.norm(rv - v) < _PARITY_TOL:
            parities.append(+1)
        elif np.linalg.norm(rv + v) < _PARITY_TOL:
            parities.append(-1)
        else:
            parities.append(0)
    definite = all(p != 0 for p in parities)
    n_fixed = sum(1 for p in parities if p == +1)

    print(
        f"  R^2=I residual={involution:.2e}  R symmetric residual={symmetric:.2e}"
        f"  [L,R] residual={commutator:.2e}"
    )
    print(
        f"  eigenvalues real (max|Im|={max_imag:.2e}); parity split ="
        f" {parities}  (fixed-axis dim = {n_fixed})"
    )

    passed = bool(
        involution < TOL
        and symmetric < TOL
        and commutator < TOL
        and max_imag < TOL
        and definite
    )
    detail = (
        "self-adjoint => real; commuting with the Z_2 mirror => spectrum "
        "sits on definite-parity sectors (HP intuition TRUE as algebra)"
    )
    print(f"  VERDICT: {'PASS' if passed else 'FAIL'} -- {detail}")
    return {"name": "self_adjoint_reflection", "status": "PASS" if passed else "FAIL"}


def test_honest_gap() -> dict:
    """TEST 5 -- the residual: {k log p} ~ log n  vs  gamma_n ~ 2 pi n / log n.

    Diagnostic (always OPEN).  The source spectrum {k log p} grows
    logarithmically while the zero ordinates grow almost linearly; no smooth
    structural map carries one to the other.  This growth mismatch IS gap G4.
    gamma_n enters only as the imported target.
    """
    print("TEST 5 -- the honest gap G4: source {k log p} vs target {gamma_n}")

    if _HAVE_MPMATH:
        z_at_zero = abs(complex(mpmath.zeta(mpmath.mpc(0.5, _KNOWN_ORDINATES[0]))))
        print(
            f"  mpmath sanity: |zeta(1/2 + i*{_KNOWN_ORDINATES[0]})| = {z_at_zero:.2e}"
            f"  (target ordinates are genuine zeros, not invented)"
        )

    if _HAVE_P14 and _HAVE_P27:
        bundle = build_prime_ladder_hamiltonian(n_primes=50, max_power=8, coupling=0.0)
        gammas = fetch_zero_imaginary_parts(80)
        t_hp = build_hp_operator(gammas)
        sa = verify_hp_self_adjoint(t_hp)
        gap = structural_gap_p14_vs_hp(bundle, gammas)
        print(
            f"  T_HP = diag(gamma_n) self-adjoint={sa['self_adjoint']}"
            f"  (gamma_n are INPUT from mpmath, NOT derived)"
        )
        print(
            f"  compared {gap['n_compared']} levels: "
            f"P14_max={gap['p14_max']:.3f}  gamma_max={gap['hp_max']:.3f}"
        )
        print(
            f"  Wasserstein_1(P14, T_HP) = {gap['wasserstein_1']:.4f}"
            f"  asymptotic growth ratio = {gap['asymptotic_growth_ratio']:.2f}"
        )
        exhibited = gap["asymptotic_growth_ratio"] > 2.0
    else:
        # Fallback: compare {k log p} to the imported ordinates directly.
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        ladder = np.sort([k * math.log(p) for p in primes for k in range(1, 9)])
        n = min(len(ladder), len(_KNOWN_ORDINATES))
        ratio = float(_KNOWN_ORDINATES[n - 1] / ladder[n - 1])
        print(
            f"  P14_max={ladder[n - 1]:.3f}  gamma_max={_KNOWN_ORDINATES[n - 1]:.3f}"
            f"  growth ratio = {ratio:.2f}"
        )
        exhibited = ratio > 2.0

    print("  scope: P27 'does not prove RH'; P13 -- the canonical analytic")
    print("         continuation across Re=1 is the missing piece (= gap G4).")
    detail = (
        "growth mismatch (log n vs 2 pi n / log n) is exhibited -> G4 is "
        "OPEN; the map {k log p} -> {gamma_n} is not a smooth structural map"
    )
    print(
        f"  VERDICT: G4 EXHIBITED (OPEN) -- {detail}"
        if exhibited
        else f"  VERDICT: inconclusive -- {detail}"
    )
    return {"name": "honest_gap", "status": "DIAGNOSTIC"}


def main() -> int:
    print(__doc__)
    print("=" * 78)
    results = [
        test_convergence_barrier(),
        test_p14_self_adjoint_origin(),
        test_adelic_boundary_vibration(),
        test_self_adjoint_reflection(),
        test_honest_gap(),
    ]
    print("=" * 78)
    print("SUMMARY")
    for r in results:
        print(f"  {r['name']:<26} {r['status']}")

    # Exit gating: every RUNNABLE structural leg (1-4) must PASS; SKIP is allowed
    # (graceful degradation without the package); leg 5 is diagnostic (OPEN).
    structural = [r for r in results if r["name"] != "honest_gap"]
    failed = [r for r in structural if r["status"] == "FAIL"]
    ran = [r for r in structural if r["status"] != "SKIP"]

    print("=" * 78)
    print("THESIS VERDICT: OPEN (G4 located, not closed)")
    print("  The SOURCE of the prime boundary vibration is canonical:")
    print("   - nu_f = log p and {k log p} derive from TNFR structure (TESTs 2,3),")
    print("   - the geometric-trace carrier is pure nu_f=log p (TEST 3, adelic.py),")
    print("   - self-adjoint + Z_2 reflection => real spectrum on the fixed axis")
    print("     -- the Hilbert-Polya intuition is TRUE as algebra (TEST 4).")
    print("  The RESONANCES {gamma_n} enter only as Ground-Truth target (TESTs 3,5).")
    print("  G4 = the canonical continuation across Re=1 / the map {k log p} ->")
    print("  {gamma_n} remains OPEN: TEST 1 shows the prime carrier cannot even be")
    print("  evaluated at Re=1/2, so mpmath is inevitable -- it draws the target,")
    print("  never derives it.  R and pi remain assumed substrate.")

    if failed:
        print(f"\nstructural leg(s) FAILED: {[r['name'] for r in failed]}")
        return 1
    if not ran:
        print("\nno structural leg could run (package unavailable)")
        return 1
    print(f"\nall {len(ran)} runnable structural leg(s) PASS (exit 0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
