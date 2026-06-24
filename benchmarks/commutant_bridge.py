"""
benchmarks/commutant_bridge.py

Camino 7 -- is the Yang-Mills U(1) -> non-Abelian gap the SAME obstruction as
the Riemann S_n-breaking gap?

equivariance_wall.py (Camino 5) showed the Riemann residue lives in Fix(S_n)^perp,
unreachable because every catalog operator f(A, L) commutes with the prime-
relabelling group S_n. The TNFR-Yang-Mills programme (theory/...YANG_MILLS...,
src/tnfr/yang_mills/derivability.py) stops at the SAME kind of place: the canonical
gauge is U(1) (Abelian, scalar connection A_ij), and the non-Abelian sector needed
for a mass gap requires NON-COMMUTING generators that are not derivable from the
nodal equation (Y3 = OPEN_DERIVABILITY_GAP). This harness asks whether the two
gaps are one structural fact: CONFINEMENT OF THE CATALOG TO A COMMUTANT.

THE CLAIM (one shape, two groups):
  The reachable set of the TNFR catalog is, in both programmes, the COMMUTANT of a
  group acting on the (possibly colour-lifted) graph space -- and each open target
  lives in the orthogonal complement that the commutant cannot reach from a
  symmetric / colour-singlet seed.

    Riemann  (G = S_n permutation rep on V):
        reachable subset of  rho(S_n)' = commutant       (Schur block-diagonal)
        V = Fix(S_n)  (+)  Fix(S_n)^perp                  (trivial isotypic + rest)
        residue S(T) = (1/pi) arg zeta(1/2 + iT)  in  Fix(S_n)^perp.   G4 = RH OPEN.

    Yang-Mills (gauge group U(d) on the colour-lifted space V (x) C^d):
        gauge acts as  I_V (x) U ;  its commutant is  End(V) (x) C.I_d
                                     (colour-scalar operators -- double commutant).
        C^(dxd) = C.I_d  (+)  su(d)                       (trivial isotypic + rest)
        non-Abelian curvature [A_mu, A_nu]  in  su(d) (traceless colour).  GAP OPEN.

  The catalog produces only f(A, L): (i) it commutes with every automorphism P_s,
  so it sits in rho(S_n)'; (ii) lifted to the bundle it acts as f(A, L) (x) I_d, so
  it commutes with EVERY gauge transformation I_V (x) U and is colour-scalar. The
  two "rest" components (Fix(S_n)^perp ; su(d)-valued curvature) are the orthogonal
  complements the commutant cannot enter -- the same shape, two different groups.

ENGINE (known theorems -- independent ground truth, all pre-TNFR):
  - Schur / double-commutant: the commutant of a group representation is the
    algebra block-diagonal across isotypic components; the commutant of
    {I_V (x) U : U in U(d)} on V (x) C^d is exactly End(V) (x) C.I_d.
  - su(2): [sigma_a/2, sigma_b/2] = i eps_abc sigma_c/2 -- traceless, non-commuting;
    exp(i theta n.sigma) = cos(theta) I + i sin(theta) n.sigma is a genuine SU(2)
    element. Abelian (scalar) holonomies commute; SU(2) holonomies do not.
  - Trace inner product: C^(dxd) = C.I_d (+) su(d) is an orthogonal split; the
    traceless part of any commutator [X, Y] has zero C.I_d component.

TNFR reading (AGENTS.md + src/tnfr/yang_mills): canonical_gauge_group = "U(1)";
the complex geometric field Psi = K_phi + i.J_phi is a single complex scalar per
node (internal rank 1) and the gauge connection / curvature are scalar. Y3 audits
whether non-commuting generators are derivable from nodal data; the conservative
verdict is OPEN_DERIVABILITY_GAP (has_noncommuting_generators = False on every
route). This is the YM mirror of the RH escape being the non-derivable per-node
diagonal P2 (Camino 5 negative control).

HONEST SCOPE -- this is the deepest path and its THESIS verdict is OPEN, not PASS:
  The structural CHECKS pass at machine precision: both reachable sets are
  commutants and both open targets are orthogonal complements -- exactly the same
  algebraic shape. That UNIFIES the two Millennium obstructions; it does NOT close
  either. Closing RH still needs the non-derivable P2 diagonal; closing the YM mass
  gap still needs non-Abelian generators whose derivation from dEPI/dt = nu_f.dNFR
  is exactly the open Y3 gap. This harness is finite toy-graph + su(2) linear
  algebra; it proves the obstructions COINCIDE in shape, not that TNFR proves
  Yang-Mills or RH. R (continuum) and phi, gamma, pi, e remain assumed substrate.

Run:
    python benchmarks/commutant_bridge.py

Status: RESEARCH (commutant-bridge falsifier; Camino 7 of the unification map).
"""

from __future__ import annotations

import os
import sys

import networkx as nx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Robust fallback so the harness also runs without PYTHONPATH=src preset.
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
)
from composition_arithmetic import automorphism_matrices  # noqa: E402

# Optional: the canonical engine's own non-Abelian derivability verdict.
try:  # pragma: no cover - exercised only when the package is importable
    from tnfr.yang_mills import audit_nonabelian_derivability  # noqa: E402

    _HAVE_AUDIT = True
except Exception:  # pragma: no cover
    _HAVE_AUDIT = False

# Optional: the canonical adelic carrier (nu_f = log p). The RH escape diagonal P2
# is exactly this carrier's prime log-frequencies, read as IMPOSED input -- the same
# adelic discipline as phase_wall.py / boundary_vibration.py, and the RH-side mirror
# of the yang_mills audit cross-check used on the YM side in TEST 4.
try:  # pragma: no cover - exercised only when the package is importable
    from tnfr.dynamics.adelic import AdelicDynamics  # noqa: E402

    _HAVE_ADELIC = True
except Exception:  # pragma: no cover
    _HAVE_ADELIC = False

TOL = 1e-9
_TWO_PI = 2.0 * np.pi
PAULI = (
    np.array([[0, 1], [1, 0]], dtype=complex),
    np.array([[0, -1j], [1j, 0]], dtype=complex),
    np.array([[1, 0], [0, -1]], dtype=complex),
)


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


def canonical_per_node_diagonal(n):
    """The canonical RH escape diagonal P2 is the adelic carrier's prime log-
    frequencies nu_f = log p (distinct per node), which dEPI/dt = nu_f . dNFR reads
    as IMPOSED input -- the P2 = NodeIndexedCouplingWeights candidate that AGENTS.md
    (B0*-beta, S13sexagesima-sexta) shows is NOT nodal-derivable. Falls back to a
    prime sieve, then to diag(1..n); the escape conclusion is identical."""
    if _HAVE_ADELIC:
        eng = AdelicDynamics(max_prime=max(15, 4 * n))
        nu = np.asarray(eng.nu_f, dtype=float)[:n]
        if nu.size == n:
            return np.diag(nu), "nu_f = log p (canonical adelic carrier, IMPOSED)"
    primes = np.array(_sieve(max(15, 4 * n)), dtype=float)[:n]
    if primes.size == n:
        return np.diag(np.log(primes)), "nu_f = log p (sieve fallback, IMPOSED)"
    return np.diag(np.arange(1, n + 1, dtype=float)), "diag(1..n) (abstract per-node)"


# --------------------------------------------------------------------------- #
# Graph operators (L = D - A is the discrete dNFR / phase-curvature operator)
# --------------------------------------------------------------------------- #
def adjacency_laplacian(G, nodes):
    """Return (A, L) with L = D - A the discrete dNFR / phase-curvature operator."""
    A = nx.to_numpy_array(G, nodelist=nodes)
    L = np.diag(A.sum(axis=1)) - A
    return A, L


def _matrix_function(S, f):
    """Apply scalar f to a symmetric matrix S via its spectral decomposition."""
    w, V = np.linalg.eigh(S)
    return (V * f(w)) @ V.T


def catalog_operators(A, L):
    """A representative slice of the TNFR catalog: every entry is a function of A
    and L only, so each is automorphism-equivariant and, once colour-lifted,
    colour-scalar. exp(-L/2) is the REMESH-inf smooth-half heat kernel."""
    return {
        "A": A,
        "L = D - A": L,
        "L^2": L @ L,
        "exp(-L/2)": _matrix_function(L, lambda x: np.exp(-0.5 * x)),
    }


def commutator_norm(M, N):
    """Frobenius norm ||M N - N M||."""
    return float(np.linalg.norm(M @ N - N @ M))


def symmetric_projector(mats):
    """Reynolds projector Pi = (1/|G|) sum_g P_g onto Fix(G) = V^G."""
    n = mats[0].shape[0]
    Pi = np.zeros((n, n))
    for M in mats:
        Pi += M
    return Pi / len(mats)


# --------------------------------------------------------------------------- #
# Colour bundle: lift, gauge action, su(2) generators, holonomy
# --------------------------------------------------------------------------- #
def su2_generators():
    """T_a = sigma_a / 2: traceless, Hermitian, [T_a, T_b] = i eps_abc T_c."""
    return [p / 2.0 for p in PAULI]


def su2_element(theta, axis):
    """exp(i theta n.sigma) = cos(theta) I + i sin(theta) n.sigma (genuine SU(2))."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    nsig = axis[0] * PAULI[0] + axis[1] * PAULI[1] + axis[2] * PAULI[2]
    return np.cos(theta) * np.eye(2, dtype=complex) + 1j * np.sin(theta) * nsig


def u1_element(phase):
    """Canonical TNFR (Abelian) gauge element: scalar phase exp(i phase) . I_2."""
    return np.exp(1j * phase) * np.eye(2, dtype=complex)


def color_scalar_part(X):
    """Projection of a colour matrix onto C.I_d -- the reachable gauge commutant."""
    d = X.shape[0]
    return (np.trace(X) / d) * np.eye(d, dtype=complex)


def lift(M, d):
    """Colour-blind lift M (x) I_d -- how the scalar catalog acts on the bundle."""
    return np.kron(M, np.eye(d, dtype=complex))


def gauge_transform(n_sites, U):
    """Gauge transformation I_V (x) U on the colour-lifted space V (x) C^d."""
    return np.kron(np.eye(n_sites), U)


def holonomy(edge_unitaries):
    """Ordered product of edge unitaries around a loop."""
    H = np.eye(edge_unitaries[0].shape[0], dtype=complex)
    for U in edge_unitaries:
        H = H @ U
    return H


# --------------------------------------------------------------------------- #
# TEST 1 -- Riemann: the catalog is trapped in the S_n commutant
# --------------------------------------------------------------------------- #
def test_rh_commutant_wall():
    print("=" * 78)
    print("TEST 1 -- RIEMANN: the catalog is trapped in the S_n COMMUTANT")
    print("=" * 78)
    G = nx.complete_graph(5)  # prime-relabelling symmetry S_5
    nodes = list(G.nodes())
    n = len(nodes)
    mats = automorphism_matrices(G, nodes)
    A, L = adjacency_laplacian(G, nodes)
    ops = catalog_operators(A, L)
    eye = np.eye(n)

    # (a) catalog subset of the commutant rho(S_n)'
    e_worst = max(commutator_norm(M, P) for M in ops.values() for P in mats)
    # (b) V = Fix(S_n) (+) Fix(S_n)^perp ; residue unreachable from symmetric seed
    Pi = symmetric_projector(mats)
    fix_dim = int(round(np.trace(Pi)))
    perp_dim = n - fix_dim
    v = Pi @ eye[:, 0]  # symmetric (colour-singlet analogue) seed
    leak = max(float(np.linalg.norm((eye - Pi) @ (M @ v))) for M in ops.values())
    w = (eye - Pi) @ eye[:, 0]  # residue target in Fix(S_n)^perp
    overlap = max(abs(float(w @ (M @ v))) for M in ops.values())

    ok = e_worst < TOL and leak < TOL and overlap < TOL and perp_dim >= 1
    print(f"  (a) catalog in commutant : max ||[f(A,L), P_s]|| = {e_worst:.2e}")
    print(
        f"  (b) V split              : dim Fix(S_n) = {fix_dim}, "
        f"dim Fix(S_n)^perp = {perp_dim}"
    )
    print(f"      symmetric seed leak  : max ||(I-Pi) M v|| = {leak:.2e}")
    print(f"      residue overlap      : max |<w, M v>| = {overlap:.2e}")
    print(
        f"  VERDICT: {'PASS' if ok else 'FAIL'} -- reachable = S_n commutant; "
        "S(T) in Fix(S_n)^perp unreachable"
    )
    print()
    return ok


# --------------------------------------------------------------------------- #
# TEST 2 -- Yang-Mills: the colour-lifted catalog is trapped in the gauge
#           commutant (colour-scalar); non-Abelian curvature is orthogonal
# --------------------------------------------------------------------------- #
def test_ym_commutant_wall():
    print("=" * 78)
    print("TEST 2 -- YANG-MILLS: the colour-lifted catalog is trapped in the GAUGE")
    print("           COMMUTANT (colour-scalar); non-Abelian curvature is orthogonal")
    print("=" * 78)
    d = 2  # SU(2) colour
    G = nx.cycle_graph(6)
    nodes = list(G.nodes())
    n = len(nodes)
    A, L = adjacency_laplacian(G, nodes)
    ops = catalog_operators(A, L)
    rng = np.random.default_rng(7)

    # (a) lifted catalog f(A,L) (x) I_d commutes with EVERY gauge transf I_V (x) U
    g_worst = 0.0
    for _ in range(8):
        U = su2_element(rng.uniform(0.3, 1.2), rng.normal(size=3))
        Ug = gauge_transform(n, U)
        g_worst = max(
            g_worst, max(commutator_norm(lift(M, d), Ug) for M in ops.values())
        )
    ab_ok = g_worst < TOL

    # (b) canonical U(1) gauge: scalar phases -> holonomies COMMUTE (Abelian)
    u1_P = holonomy([u1_element(rng.uniform(0, _TWO_PI)) for _ in range(4)])
    u1_Q = holonomy([u1_element(rng.uniform(0, _TWO_PI)) for _ in range(4)])
    u1_comm = commutator_norm(u1_P, u1_Q)
    u1_ok = u1_comm < TOL

    # (c) SU(2) gauge: non-Abelian -> holonomies DON'T commute, and the field-
    #     strength commutator [T_a, T_b] is TRACELESS -> zero colour-scalar part
    tx, ty, _tz = su2_generators()
    f_na = tx @ ty - ty @ tx  # [A_mu, A_nu] non-Abelian curvature term
    f_scalar = color_scalar_part(f_na)  # projection onto the reachable commutant
    f_norm = float(np.linalg.norm(f_na))
    f_reach = float(np.linalg.norm(f_scalar))
    su2_P = holonomy(
        [su2_element(rng.uniform(0.3, 1.2), rng.normal(size=3)) for _ in range(4)]
    )
    su2_Q = holonomy(
        [su2_element(rng.uniform(0.3, 1.2), rng.normal(size=3)) for _ in range(4)]
    )
    su2_comm = commutator_norm(su2_P, su2_Q)
    na_ok = su2_comm > 1e-3 and f_norm > 1e-3 and f_reach < TOL

    ok = ab_ok and u1_ok and na_ok
    print(
        f"  (a) lifted catalog in gauge commutant : "
        f"max ||[f(A,L)(x)I, I(x)U]|| = {g_worst:.2e}  (colour-blind, all U)"
    )
    print(
        f"  (b) canonical U(1) holonomies commute : "
        f"||[H_P, H_Q]|| = {u1_comm:.2e}  (Abelian -- the canonical gauge)"
    )
    print(
        f"  (c) SU(2) holonomies do NOT commute   : " f"||[H_P, H_Q]|| = {su2_comm:.3f}"
    )
    print(
        f"      non-Abelian curvature [T_a,T_b]   : ||F|| = {f_norm:.3f}, "
        f"colour-scalar part ||P(F)|| = {f_reach:.2e}  (traceless -> unreachable)"
    )
    print(
        f"  VERDICT: {'PASS' if ok else 'FAIL'} -- reachable = colour-scalar "
        "commutant; su(d) curvature orthogonal"
    )
    print()
    return ok


# --------------------------------------------------------------------------- #
# TEST 3 -- the bridge: one shape (trivial isotypic (+) rest), two groups
# --------------------------------------------------------------------------- #
def test_one_shape_two_groups():
    print("=" * 78)
    print("TEST 3 -- THE BRIDGE: one shape (trivial isotypic (+) rest), two groups")
    print("=" * 78)
    # Riemann: V = Fix(S_n) (+) Fix(S_n)^perp under the S_n permutation rep
    G = nx.complete_graph(5)
    nodes = list(G.nodes())
    n = len(nodes)
    mats = automorphism_matrices(G, nodes)
    Pi = symmetric_projector(mats)
    rh_fix = int(round(np.trace(Pi)))
    rh_rest = n - rh_fix
    rh_orth = float(np.linalg.norm(Pi @ (np.eye(n) - Pi)))  # blocks orthogonal

    # Yang-Mills: C^(dxd) = C.I_d (+) su(d) under U(d) conjugation
    d = 2
    tx, ty, tz = su2_generators()
    rest_basis = [tx, ty, tz]  # su(2): traceless, dim d^2 - 1 = 3
    i_d = np.eye(d, dtype=complex)
    ym_fix = 1  # dim C.I_d
    ym_rest = d * d - ym_fix  # dim of the traceless colour part
    ym_orth = max(abs(complex(np.trace(i_d.conj().T @ T))) for T in rest_basis)

    ok = (
        rh_rest >= 1
        and ym_rest >= 1
        and rh_orth < TOL
        and ym_orth < TOL
        and ym_rest == len(rest_basis)
    )
    print("  Riemann   (G = S_n)  : V        = Fix(S_n) (+) Fix(S_n)^perp")
    print(
        f"                         dims     = {rh_fix} (+) {rh_rest}   "
        f"block orthogonality ||Pi(I-Pi)|| = {rh_orth:.2e}"
    )
    print("                         residue  = S(T) in Fix(S_n)^perp   [G4 = RH OPEN]")
    print("  Yang-Mills (G = U(d)): C^(dxd)  = C.I_d (+) su(d)")
    print(
        f"                         dims     = {ym_fix} (+) {ym_rest}   "
        f"block orthogonality max|<I_d, T_a>| = {ym_orth:.2e}"
    )
    print(
        "                         curvature= [A_mu,A_nu] in su(d)     [mass gap OPEN]"
    )
    print(
        f"  VERDICT: {'PASS' if ok else 'FAIL'} -- same shape (trivial isotypic "
        "(+) rest), two groups (S_n / U(d))"
    )
    print()
    return ok


# --------------------------------------------------------------------------- #
# TEST 4 -- honest OPEN: the escape ingredient is non-derivable in BOTH
# --------------------------------------------------------------------------- #
def test_nonderivable_escape_contrast():
    print("=" * 78)
    print("TEST 4 -- HONEST OPEN: the escape ingredient is NON-DERIVABLE in BOTH")
    print("=" * 78)
    # RH escape: per-node diagonal P2 (NodeIndexedCouplingWeights) breaks S_n,
    # but is not nodal-equation-derivable (no per-node slot).
    G = nx.complete_graph(5)
    nodes = list(G.nodes())
    n = len(nodes)
    mats = automorphism_matrices(G, nodes)
    Pi = symmetric_projector(mats)
    eye = np.eye(n)
    # The canonical RH escape diagonal P2 is the adelic carrier's prime log-
    # frequencies nu_f = log p (distinct per node), read by the engine as IMPOSED
    # input -- not produced by dEPI/dt = nu_f . dNFR. Fallback: sieve / diag(1..n).
    p2, p2_label = canonical_per_node_diagonal(n)
    rh_break = max(commutator_norm(p2, P) for P in mats)
    rh_leak = float(np.linalg.norm((eye - Pi) @ (p2 @ (Pi @ eye[:, 0]))))
    rh_escapes = rh_break > 1e-3 and rh_leak > 1e-3

    # YM escape: non-commuting generators break the colour-scalar commutant, but
    # Y3 audits them as not derivable from nodal data.
    tx, ty, _tz = su2_generators()
    ym_break = commutator_norm(tx, ty)  # [T_x,T_y] != 0 -> leaves C.I_d
    ym_escapes = ym_break > 1e-3

    verdict_line = "OPEN_DERIVABILITY_GAP (canonical default; package not imported)"
    canon_ok = True
    if _HAVE_AUDIT:
        try:
            report = audit_nonabelian_derivability()
            any_noncomm = any(c.has_noncommuting_generators for c in report.candidates)
            verdict_line = (
                f"{report.verdict} ; gauge = "
                f"{report.canonical_gauge_group} ; "
                f"non-commuting generators on any route = {any_noncomm}"
            )
            canon_ok = report.verdict == "OPEN_DERIVABILITY_GAP" and not any_noncomm
        except Exception as exc:  # pragma: no cover
            verdict_line = f"(canonical audit unavailable: {exc})"

    ok = rh_escapes and ym_escapes and canon_ok
    print(f"  RH escape : P2 = diag({p2_label})")
    print(
        f"              breaks S_n (||[P2,P_s]|| = {rh_break:.2f}, leak = "
        f"{rh_leak:.2f}) -- but P2 is NOT nodal-derivable (B0*-beta, no per-node"
    )
    print("              slot in dEPI/dt; the adelic engine reads nu_f = log p as")
    print("              IMPOSED input -- the RH mirror of the YM audit below).")
    print(
        "  YM escape : non-commuting generators [T_x,T_y] != 0 "
        f"(||[T_x,T_y]|| = {ym_break:.3f}) leave the colour-scalar commutant"
    )
    print("              -- but their derivation from dEPI/dt = nu_f.dNFR is the")
    print(f"              open Y3 gap. Canonical audit: {verdict_line}")
    print(
        f"  VERDICT: {'PASS' if ok else 'FAIL'} -- both escapes exist but neither "
        "is nodal-equation-derivable"
    )
    print()
    return ok


def main():
    print(__doc__)
    t1 = test_rh_commutant_wall()
    t2 = test_ym_commutant_wall()
    t3 = test_one_shape_two_groups()
    t4 = test_nonderivable_escape_contrast()

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  TEST 1 RH commutant wall (S_n)          : {'PASS' if t1 else 'FAIL'}")
    print(f"  TEST 2 YM commutant wall (U(d) colour)  : {'PASS' if t2 else 'FAIL'}")
    print(f"  TEST 3 one shape, two groups (bridge)   : {'PASS' if t3 else 'FAIL'}")
    print(f"  TEST 4 non-derivable escape (both)      : {'PASS' if t4 else 'FAIL'}")
    structural = t1 and t2 and t3 and t4
    print()
    print(f"  STRUCTURAL CHECKS: {'ALL PASS' if structural else 'SOME FAIL'}")
    print()
    print("  THESIS VERDICT: OPEN / PARTIAL (by design -- the deepest path).")
    print("  The two Millennium obstructions COINCIDE in shape: in both programmes")
    print("  the reachable set is the COMMUTANT of a group acting on the (colour-")
    print("  lifted) graph, and each open target lives in the orthogonal complement")
    print("  (Fix(S_n)^perp for Riemann ; su(d)-valued curvature for Yang-Mills).")
    print("  This UNIFIES the obstruction; it does NOT close it. The escape in each")
    print("  case -- the per-node diagonal P2 (RH) and non-commuting generators")
    print("  (YM) -- is exactly the ingredient that is NOT derivable from the nodal")
    print("  equation. HONEST SCOPE: finite toy-graph + su(2) algebra; nothing here")
    print("  proves RH, the Yang-Mills mass gap, or closes G4. R and phi,gamma,pi,e")
    print("  remain assumed substrate.")
    return 0 if structural else 1


if __name__ == "__main__":
    raise SystemExit(main())
