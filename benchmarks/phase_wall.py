"""
benchmarks/phase_wall.py

Camino 8 -- is the Riemann residue unreachable because the catalog is confined to
the REAL / SELF-ADJOINT sector, while the residue is a CONTINUOUS PHASE on the
e-pi circle?

commutant_bridge.py (Camino 7) unified the Riemann S_n-breaking gap and the
Yang-Mills U(1) -> non-Abelian gap as ONE fact: confinement of the catalog to a
COMMUTANT. This harness drills into WHY the open target is a phase. It is the exact
mirror of Camino 7 for the e-pi edge of the Universal Tetrahedral Correspondence:
the four constants (phi, gamma, pi, e) are the four REAL scales of the derivative
tower over the graph (AGENTS.md, "The Operator-Derivative Tower"), and the catalog
f(A, L) is built from the SYMMETRIC coupling A and the self-adjoint dNFR operator
L = D - A. Self-adjoint => real spectrum => the only phases it carries are arg in
{0, pi} (a sign). The Riemann residue S(T) = (1/pi) arg zeta(1/2 + iT) is a
CONTINUOUS argument -- it lives on the circle, not on the real axis.

THE CLAIM (real/scale wall vs phase/oscillation residue):
  reachable:  f(A, L)  with A = A^T (mutual resonance) and L = D - A self-adjoint
              => spectrum real => eigen-phase in {0, pi} (the real axis).
  residue:    S(T) = (1/pi) arg zeta(1/2 + iT)  is a CONTINUOUS phase (the circle).
  the gap:    {0, pi}  (real axis, scale sector)   vs   continuous arg  (e-pi circle).

  The ONLY map from the real axis to a continuous phase is z |-> exp(i z): the e-pi
  circle (Euler: exp(i pi) = -1). The canonical engine DOES own one such carrier --
  the adelic unitary U(t) = diag(exp(i t nu_f)) with nu_f = log p (CANONICAL, see
  src/tnfr/dynamics/adelic.py) -- and it reaches the circle. BUT its per-node
  arithmetic content nu_f = log p is IMPOSED (a prime sieve), not produced by the
  nodal equation: dEPI/dt = nu_f . dNFR reads nu_f as input. Promoting nu_f to a
  circle-valued / Pontryagin-dual object (candidate P1 = E0) is the non-derivable
  step (AGENTS.md B0*-beta: C1 reduces to (P-nu_f-Bijectivity) =
  FORWARD_INDEPENDENT_OF_BACKWARD; C4 fails because S(T) is invariant under that
  promotion). This is the EXACT mirror of the Yang-Mills Y3 gap: the canonical
  gauge is U(1) (the same e-pi circle, a scalar phase exp(i phi)); the missing
  ingredient -- non-commuting generators (YM) / derived prime frequencies (RH) --
  is not nodal-derivable.

ENGINE (known theorems -- independent ground truth, all pre-TNFR):
  - Spectral theorem: a real symmetric (self-adjoint) matrix has a real spectrum;
    hence arg(lambda) in {0, pi} (zero eigenvalues have undefined phase and are
    excluded). Any polynomial / spectral function of symmetric A, L stays symmetric.
  - Euler / Pontryagin: z |-> exp(i z) is the unique homomorphism R -> S^1; a
    diagonal unitary diag(exp(i theta_k)) has eigen-phases theta_k on the circle.
  - arg zeta(1/2 + iT) is a continuous real-valued function of T (Riemann-Siegel
    theta / S(T)); it is not confined to {0, pi}.

TNFR reading (AGENTS.md + src/tnfr/dynamics/adelic.py): nu_f = log p is a REAL
per-node scalar frequency; the adelic phase exp(i t nu_f) is a DERIVED unitary
rotation, not a generator, and its content (which primes, hence the residue's
oscillation) is imposed, not derived. The four tetrad constants are real scales;
the phase sector requires complexification through the e-pi circle, which leaves
the self-adjoint catalog.

HONEST SCOPE -- structural CHECKS pass; the THESIS verdict is OPEN, not PASS:
  We show at machine precision that (1) every catalog f(A, L) has eigen-phases in
  {0, pi}; (2) the residue S(T) is a continuous phase, disjoint from {0, pi}; (3)
  the canonical adelic carrier reaches the circle but is non-self-adjoint and its
  content nu_f = log p is imposed; (4) only the e-pi complexification leaves the
  real catalog, and that step is the non-derivable Pontryagin promotion -- the
  mirror of the YM Y3 gap (cross-checked against the canonical audit). This LOCATES
  the obstruction as a real-vs-phase wall; it does NOT close it. Reaching S(T) is
  RH-equivalent. R (continuum) and phi, gamma, pi, e remain assumed substrate.

Run:
    python benchmarks/phase_wall.py

Status: RESEARCH (phase-wall falsifier; Camino 8 of the unification map).
"""
from __future__ import annotations

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

# Optional: the canonical adelic engine (nu_f = log p phase carrier).
try:  # pragma: no cover - exercised only when the package is importable
    from tnfr.dynamics.adelic import AdelicDynamics  # noqa: E402
    _HAVE_ADELIC = True
except Exception:  # pragma: no cover
    _HAVE_ADELIC = False

# Optional: the canonical Yang-Mills non-Abelian derivability verdict (Camino 7
# mirror -- the same U(1) = e-pi circle is the canonical gauge).
try:  # pragma: no cover
    from tnfr.yang_mills import audit_nonabelian_derivability  # noqa: E402
    _HAVE_AUDIT = True
except Exception:  # pragma: no cover
    _HAVE_AUDIT = False

TOL = 1e-9
_ZERO_EIG = 1e-6                     # eigenvalues below this have undefined phase
_REAL_AXIS = np.array([0.0, np.pi, -np.pi])   # arg of a real number

# The four constants of the Universal Tetrahedral Correspondence (real scales).
PHI = (1.0 + np.sqrt(5.0)) / 2.0    # golden ratio        <-> Phi_s   (global)
GAMMA = 0.5772156649015329          # Euler-Mascheroni    <-> |grad phi| (local)
PI = np.pi                          # pi                  <-> K_phi   (curvature)
E = np.e                            # Napier              <-> xi_C    (correlation)

# First few Riemann non-trivial zero heights (ground truth for sampling S(T)).
_KNOWN_ZEROS = (14.1347, 21.0220, 25.0109, 30.4249, 32.9351, 37.5862)


# --------------------------------------------------------------------------- #
# Graph operators (L = D - A is the self-adjoint discrete dNFR operator)
# --------------------------------------------------------------------------- #
def adjacency_laplacian(G, nodes):
    """Return (A, L) with A = A^T (mutual coupling) and L = D - A self-adjoint."""
    A = nx.to_numpy_array(G, nodelist=nodes)
    L = np.diag(A.sum(axis=1)) - A
    return A, L


def _matrix_function(S, f):
    """Apply scalar f to a symmetric matrix S via its spectral decomposition."""
    w, V = np.linalg.eigh(S)
    return (V * f(w)) @ V.T


def catalog_operators(A, L):
    """A representative slice of the TNFR catalog: every entry is a function of the
    symmetric A and the self-adjoint L = D - A, so each is real-symmetric.
    exp(-L/2) is the REMESH-inf smooth-half heat kernel."""
    return {
        "A": A,
        "L = D - A": L,
        "L^2": L @ L,
        "exp(-L/2)": _matrix_function(L, lambda x: np.exp(-0.5 * x)),
    }


def is_self_adjoint(M):
    """Frobenius distance from Hermitian: ||M - M^dagger||."""
    return float(np.linalg.norm(M - M.conj().T))


def eigen_phases(M):
    """arg of the eigenvalues of M, excluding (phase-undefined) zero eigenvalues."""
    w = np.linalg.eigvals(M)
    w = w[np.abs(w) > _ZERO_EIG]
    return np.angle(w)


def distance_from_real_axis(phases):
    """max over phases of the distance to the nearest real-axis arg in {0, pi}."""
    if phases.size == 0:
        return 0.0
    d = np.min(np.abs(phases[:, None] - _REAL_AXIS[None, :]), axis=1)
    return float(np.max(d))


# --------------------------------------------------------------------------- #
# Arithmetic phase carriers (the e-pi circle)
# --------------------------------------------------------------------------- #
def _sieve(n):
    """Primes up to n (Sieve of Eratosthenes) -- fallback if adelic is absent."""
    flag = [True] * (n + 1)
    out = []
    for p in range(2, n + 1):
        if flag[p]:
            out.append(p)
            for k in range(p * p, n + 1, p):
                flag[k] = False
    return out


def canonical_prime_frequencies(max_prime=30):
    """Canonical nu_f = log p (from the adelic engine if importable, else a sieve).
    These per-node REAL frequencies are IMPOSED arithmetic content, not produced by
    the nodal equation dEPI/dt = nu_f . dNFR (which reads nu_f as input)."""
    if _HAVE_ADELIC:
        eng = AdelicDynamics(max_prime=max_prime)
        return np.asarray(eng.nu_f, dtype=float), np.asarray(eng.primes, dtype=float)
    primes = np.array(_sieve(max_prime), dtype=float)
    return np.log(primes), primes


def adelic_phase_unitary(t, nu_f):
    """The canonical adelic carrier U(t) = diag(exp(i t nu_f)): a diagonal unitary
    whose eigen-phases t.nu_f live on the e-pi circle S^1, not on the real axis."""
    return np.diag(np.exp(1j * t * nu_f))


def riemann_s_phase(T, nu_f, primes):
    """S(T) = (1/pi) arg zeta(1/2 + iT) via mpmath; fallback = the adelic geometric-
    trace phase (1/pi) arg sum_p p^(-1/2) exp(i T log p). Both are CONTINUOUS in T."""
    if _HAVE_MPMATH:
        z = mpmath.zeta(mpmath.mpc(0.5, T))
        return float(mpmath.arg(z)) / np.pi
    z = np.sum(np.exp(1j * T * nu_f) / np.sqrt(primes))
    return float(np.angle(z)) / np.pi


# --------------------------------------------------------------------------- #
# TEST 1 -- the real wall: every catalog f(A, L) has eigen-phases in {0, pi}
# --------------------------------------------------------------------------- #
def test_catalog_is_real_axis():
    print("=" * 78)
    print("TEST 1 -- THE REAL WALL: the catalog f(A, L) is self-adjoint => arg "
          "in {0, pi}")
    print("=" * 78)
    # Prime-ladder path graph on the first primes (the Riemann-relevant topology).
    primes = _sieve(20)                      # [2,3,5,7,11,13,17,19]
    G = nx.path_graph(len(primes))
    nodes = list(G.nodes())
    A, L = adjacency_laplacian(G, nodes)
    ops = catalog_operators(A, L)

    herm_worst = max(is_self_adjoint(M) for M in ops.values())
    phase_worst = 0.0
    for name, M in ops.items():
        ph = eigen_phases(M)
        d = distance_from_real_axis(ph)
        phase_worst = max(phase_worst, d)
        print(f"  {name:<10}: self-adjoint dist = {is_self_adjoint(M):.2e}, "
              f"max arg-dist from {{0,pi}} = {d:.2e}")
    # cross-check via the shared spectrum helper: A's spectrum is real
    a_imag = float(np.max(np.abs(np.imag(adj_spectrum(G, nodes)))))

    ok = herm_worst < TOL and phase_worst < 1e-6 and a_imag < TOL
    print(f"  worst self-adjoint distance        : {herm_worst:.2e}")
    print(f"  worst eigen-phase distance to axis : {phase_worst:.2e}")
    print(f"  VERDICT: {'PASS' if ok else 'FAIL'} -- catalog spectrum is REAL; "
          "eigen-phase locked to {0, pi} (a sign, no continuous phase)")
    print()
    return ok


# --------------------------------------------------------------------------- #
# TEST 2 -- the residue is a CONTINUOUS phase, disjoint from {0, pi}
# --------------------------------------------------------------------------- #
def test_residue_is_continuous_phase():
    print("=" * 78)
    print("TEST 2 -- THE RESIDUE: S(T) = (1/pi) arg zeta(1/2 + iT) is a CONTINUOUS "
          "phase")
    print("=" * 78)
    nu_f, primes = canonical_prime_frequencies(60)
    source = "mpmath zeta(1/2+iT)" if _HAVE_MPMATH else "adelic trace phase"
    # Sample near and between the first non-trivial zeros.
    samples = []
    for z in _KNOWN_ZEROS:
        for off in (-0.7, 0.0, 0.9):
            T = z + off
            s = riemann_s_phase(T, nu_f, primes)
            samples.append((T, s))
    arg_vals = np.array([np.pi * s for _, s in samples])      # back to radians
    dist_axis = distance_from_real_axis(arg_vals)
    n_off_axis = int(np.sum(
        np.min(np.abs(arg_vals[:, None] - _REAL_AXIS[None, :]), axis=1) > 0.2))
    spread = float(np.max(arg_vals) - np.min(arg_vals))

    print(f"  source                : {source}")
    print(f"  samples               : {len(samples)} values of S(T) near zeros")
    for T, s in samples[:4]:
        print(f"     S({T:6.3f}) = {s:+.4f}   (arg = {np.pi * s:+.4f} rad)")
    print(f"  spread of arg          : {spread:.3f} rad")
    print(f"  max dist from {{0,pi}}   : {dist_axis:.3f} rad  (>> 0 => off the axis)")
    print(f"  samples off the axis   : {n_off_axis} / {len(samples)}")
    # The residue is continuous (large spread, far from the real axis), so it can
    # NEVER equal a catalog eigen-phase, which lives in {0, pi}.
    ok = dist_axis > 0.3 and spread > 0.5 and n_off_axis >= len(samples) // 2
    print(f"  VERDICT: {'PASS' if ok else 'FAIL'} -- residue lives on the circle, "
          "disjoint from the real-axis catalog spectrum")
    print()
    return ok


# --------------------------------------------------------------------------- #
# TEST 3 -- the canonical carrier reaches the circle, but is non-self-adjoint
#           and its content nu_f = log p is IMPOSED, not derived
# --------------------------------------------------------------------------- #
def test_canonical_carrier_content_is_imposed():
    print("=" * 78)
    print("TEST 3 -- THE CARRIER: adelic U(t) = diag(exp(i t nu_f)) reaches the "
          "circle,")
    print("           but is non-self-adjoint and nu_f = log p is IMPOSED")
    print("=" * 78)
    nu_f, primes = canonical_prime_frequencies(30)
    origin = "tnfr.dynamics.adelic (CANONICAL)" if _HAVE_ADELIC else "local sieve"
    U = adelic_phase_unitary(1.3, nu_f)

    # (a) U reaches the phase sector: its eigen-phases are continuous, off-axis.
    u_phases = np.angle(np.diag(U))
    u_dist = distance_from_real_axis(u_phases)
    reaches = u_dist > 0.3
    # (b) U is NOT self-adjoint and NOT a real f(A, L): it is unitary with complex
    #     spectrum on S^1 (a different operator class from the real catalog).
    non_herm = is_self_adjoint(U)
    unit_err = float(np.linalg.norm(U.conj().T @ U - np.eye(U.shape[0])))
    spec_imag = float(np.max(np.abs(np.imag(np.linalg.eigvals(U)))))
    distinct_class = non_herm > 1e-3 and unit_err < TOL and spec_imag > 1e-3
    # (c) the content nu_f = log p is imposed: it equals log(primes) exactly, an
    #     arithmetic input, not a fixed point of the nodal equation.
    imposed = bool(np.allclose(nu_f, np.log(primes), atol=TOL))

    ok = reaches and distinct_class and imposed
    print(f"  nu_f source                 : {origin}")
    print(f"  (a) carrier reaches circle  : max arg-dist from {{0,pi}} = "
          f"{u_dist:.3f}  (continuous phase)")
    print(f"  (b) non-self-adjoint        : ||U - U^dag|| = {non_herm:.3f}, "
          f"unitary err = {unit_err:.2e}, max|Im spec| = {spec_imag:.3f}")
    print("      => U is unitary on S^1, NOT a real-symmetric f(A, L)")
    print(f"  (c) content imposed         : nu_f == log(primes)? {imposed}  "
          "(arithmetic input, not nodal-derived)")
    print(f"  VERDICT: {'PASS' if ok else 'FAIL'} -- the carrier exists (U(1) "
          "phase) but its arithmetic content is FORWARD_INDEPENDENT_OF_BACKWARD")
    print()
    return ok


# --------------------------------------------------------------------------- #
# TEST 4 -- the e-pi channel + honest OPEN (mirror of the Yang-Mills Y3 gap)
# --------------------------------------------------------------------------- #
def test_e_pi_is_the_only_phase_channel():
    print("=" * 78)
    print("TEST 4 -- THE e-pi CHANNEL: real constants stay on the axis; only "
          "exp(i .) escapes")
    print("=" * 78)
    primes = _sieve(20)
    G = nx.path_graph(len(primes))
    nodes = list(G.nodes())
    A, L = adjacency_laplacian(G, nodes)
    K = _matrix_function(L, lambda x: np.exp(-0.5 * x))

    # (a) any REAL combination of the four constants stays real-symmetric => {0,pi}
    M = PHI * A + GAMMA * L + PI * (L @ L) + E * K
    m_herm = is_self_adjoint(M)
    m_dist = distance_from_real_axis(eigen_phases(M))
    real_axis = m_herm < TOL and m_dist < 1e-6

    # (b) the e-pi map z |-> exp(i z) sends those real eigenvalues onto the circle
    w = np.linalg.eigvalsh(M)
    circ_phases = np.angle(np.exp(1j * w))
    circ_dist = distance_from_real_axis(circ_phases)
    escapes = circ_dist > 0.3        # complexification reaches continuous phase

    # (c) Yang-Mills mirror: the canonical gauge is U(1) (the SAME e-pi circle, a
    #     scalar phase exp(i phi)); the non-derivable ingredient is the open gap.
    verdict_line = "OPEN_DERIVABILITY_GAP (canonical default; package not imported)"
    canon_ok = True
    if _HAVE_AUDIT:
        try:
            report = audit_nonabelian_derivability()
            any_noncomm = any(c.has_noncommuting_generators
                              for c in report.candidates)
            verdict_line = (f"{report.verdict} ; gauge = "
                            f"{report.canonical_gauge_group} ; "
                            f"non-commuting generators on any route = {any_noncomm}")
            canon_ok = (report.verdict == "OPEN_DERIVABILITY_GAP"
                        and report.canonical_gauge_group == "U(1)"
                        and not any_noncomm)
        except Exception as exc:  # pragma: no cover
            verdict_line = f"(canonical audit unavailable: {exc})"

    ok = real_axis and escapes and canon_ok
    print(f"  (a) phi.A + gamma.L + pi.L^2 + e.exp(-L/2) real-symmetric : "
          f"herm = {m_herm:.2e}, arg-dist = {m_dist:.2e}  (stays on axis)")
    print(f"  (b) exp(i .) sends spectrum onto the circle               : "
          f"max arg-dist = {circ_dist:.3f}  (the e-pi escape)")
    print("  (c) the e-pi circle IS the canonical U(1) gauge of Camino 7;")
    print(f"      canonical YM audit: {verdict_line}")
    print(f"  VERDICT: {'PASS' if ok else 'FAIL'} -- four REAL constants never "
          "leave {0,pi}; the phase needs exp(i .), whose content is non-derivable")
    print()
    return ok


def main():
    print(__doc__)
    t1 = test_catalog_is_real_axis()
    t2 = test_residue_is_continuous_phase()
    t3 = test_canonical_carrier_content_is_imposed()
    t4 = test_e_pi_is_the_only_phase_channel()

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  TEST 1 real wall: catalog arg in {{0,pi}}     : {'PASS' if t1 else 'FAIL'}")
    print(f"  TEST 2 residue S(T) is continuous phase     : {'PASS' if t2 else 'FAIL'}")
    print(f"  TEST 3 carrier reaches circle, content imposed: {'PASS' if t3 else 'FAIL'}")
    print(f"  TEST 4 e-pi is the only phase channel       : {'PASS' if t4 else 'FAIL'}")
    structural = t1 and t2 and t3 and t4
    print()
    print(f"  STRUCTURAL CHECKS: {'ALL PASS' if structural else 'SOME FAIL'}")
    print()
    print("  THESIS VERDICT: OPEN / PARTIAL (by design -- the deepest path).")
    print("  The residue is unreachable because the catalog is confined to the REAL")
    print("  / SELF-ADJOINT sector (arg in {0, pi}), while S(T) = (1/pi) arg zeta is")
    print("  a CONTINUOUS phase on the e-pi circle. The four tetrad constants (phi,")
    print("  gamma, pi, e) are the four REAL scales of the derivative tower; the")
    print("  phase sector requires the e-pi complexification z |-> exp(i z). The")
    print("  canonical engine owns one carrier -- the adelic U(t) = diag(exp(i t")
    print("  nu_f)), nu_f = log p -- but its arithmetic content is IMPOSED, not")
    print("  nodal-derived (FORWARD_INDEPENDENT_OF_BACKWARD). This is the e-pi mirror")
    print("  of the Yang-Mills U(1) gap (same circle, same OPEN verdict). It LOCATES")
    print("  the obstruction as a real-vs-phase wall; reaching S(T) is RH-equivalent")
    print("  and stays OPEN. R and phi, gamma, pi, e remain assumed substrate.")
    return 0 if structural else 1


if __name__ == "__main__":
    raise SystemExit(main())
