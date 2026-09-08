r"""Projective p-adic network tower (R4).

The p-adic tower is the inverse system

    ℤ/pℤ  ←  ℤ/p²ℤ  ←  ℤ/p³ℤ  ←  ⋯

with reduction maps ``π_e : ℤ/p^{e+1}ℤ → ℤ/p^eℤ`` (``x ↦ x mod p^e``).  Each
coarse node has a **fiber** of ``p`` fine nodes.  Two structural maps encode the
scale change:

* the **fiber-averaging** (aggregation) map ``R_e`` (:func:`projective_scale_map`),
  a stochastic ``p^e × p^{e+1}`` operator averaging each fiber;
* the **lift** (prolongation) map ``Lift_e`` (:func:`padic_lift_map`), the
  ``p^{e+1} × p^e`` pullback that is constant on fibers.

For the **reduction-compatible** connection family
``S_e = {x : x mod p ∈ base}`` (any base pattern mod ``p`` lifted by ignoring
higher p-adic digits) the random-walk transport is **projective**: exactly over ℚ,

    R_e P_{e+1} = P_e R_e            (fiber-averaging commutes with transport),
    P_{e+1} Lift_e = Lift_e P_e      (the lift intertwines the two levels),
    R_e Lift_e = I                    (the lift is a right inverse of averaging).

The intertwining makes every coarse eigenvector lift to a fine one with the same
eigenvalue, so ``spec(L_e) ⊆ spec(L_{e+1})``: **the coarse modes survive the
lift**, and level ``e+1`` only *adds* finer, fiber-varying modes.

**REMESH is NOT claimed.**  The scale map is given the neutral name
``projective_scale_map`` on purpose.  To be called REMESH it must pass the
recursivity contract (EPI recursion, NETWORK scale, preserved identity, U5);
:func:`remesh_contract_audit` records those conditions as **unverified**, so the
tower→REMESH identification (claim ``NT-P04``) stays **CONJECTURAL**.

Scope: exact rational linear algebra on small ``p`` and low exponents.  No
extrapolation to the projective limit ``ℤ_p`` is claimed from finitely many
levels; no complexity, cryptographic, or Millennium claim is made.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from ..physics.spectral_projectors import derived_tolerance

__all__ = [
    "projective_scale_map",
    "padic_lift_map",
    "compatible_connection_set",
    "padic_transition",
    "padic_laplacian",
    "projective_commutation_residual",
    "laplacian_commutation_residual",
    "lift_intertwining_residual",
    "lift_reduction_residual",
    "surviving_spectrum_containment",
    "padic_spectral_gaps",
    "RemeshContractAudit",
    "remesh_contract_audit",
]

Matrix = list[list[Fraction]]


# --------------------------------------------------------------------------- #
# Scale maps
# --------------------------------------------------------------------------- #
def projective_scale_map(p: int, e: int) -> Matrix:
    r"""Fiber-averaging map ``R_e : ℝ^{p^{e+1}} → ℝ^{p^e}`` (exact).

    ``(R_e f)(y) = (1/p) Σ_{x ≡ y (mod p^e)} f(x)`` — the aggregation/averaging
    operator over each ``p``-element fiber.  Deliberately **not** named REMESH.
    """
    lo, hi = p ** e, p ** (e + 1)
    inv = Fraction(1, p)
    R: Matrix = [[Fraction(0)] * hi for _ in range(lo)]
    for x in range(hi):
        R[x % lo][x] = inv
    return R


def padic_lift_map(p: int, e: int) -> Matrix:
    r"""Prolongation map ``Lift_e : ℝ^{p^e} → ℝ^{p^{e+1}}`` (exact).

    ``(Lift_e g)(x) = g(x mod p^e)`` — the pullback that is constant on fibers;
    the right inverse of :func:`projective_scale_map`.
    """
    lo, hi = p ** e, p ** (e + 1)
    L: Matrix = [[Fraction(0)] * lo for _ in range(hi)]
    for x in range(hi):
        L[x][x % lo] = Fraction(1)
    return L


def compatible_connection_set(p: int, e: int, base: frozenset[int]) -> set[int]:
    r"""Reduction-compatible connection set ``S_e = {x : x mod p ∈ base}``.

    ``base`` is a set of non-zero residues mod ``p``.  Lifting a base pattern by
    ignoring higher p-adic digits makes every fiber uniformly represented, which
    is what makes transport projective (:func:`projective_commutation_residual`).
    """
    if not base:
        raise ValueError("base connection set must be non-empty")
    if any((r % p) == 0 for r in base):
        raise ValueError("base residues must be non-zero mod p (no self-loops)")
    return {x for x in range(p ** e) if (x % p) in {r % p for r in base}}


# --------------------------------------------------------------------------- #
# Transport
# --------------------------------------------------------------------------- #
def padic_transition(p: int, e: int, connection: set[int]) -> Matrix:
    r"""Exact transition ``P = (1/d) W`` for ``Cay(ℤ/p^eℤ, connection)``."""
    n = p ** e
    d = len(connection)
    if d == 0:
        raise ValueError("empty connection set")
    inv = Fraction(1, d)
    P: Matrix = [[Fraction(0)] * n for _ in range(n)]
    for i in range(n):
        for s in connection:
            P[i][(i + s) % n] += inv
    return P


def padic_laplacian(p: int, e: int, connection: set[int]) -> Matrix:
    r"""Exact random-walk Laplacian ``L = I − P`` at level ``e``."""
    P = padic_transition(p, e, connection)
    n = p ** e
    return [
        [(Fraction(1) if i == j else Fraction(0)) - P[i][j] for j in range(n)]
        for i in range(n)
    ]


# --------------------------------------------------------------------------- #
# Exact matrix helpers
# --------------------------------------------------------------------------- #
def _matmul(A: Matrix, B: Matrix) -> Matrix:
    n, k, m = len(A), len(B), len(B[0])
    return [
        [sum((A[i][t] * B[t][j] for t in range(k)), Fraction(0))
         for j in range(m)]
        for i in range(n)
    ]


def _sub(A: Matrix, B: Matrix) -> Matrix:
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def _maxabs(A: Matrix) -> Fraction:
    return max((abs(x) for row in A for x in row), default=Fraction(0))


def _identity(n: int) -> Matrix:
    return [[Fraction(1) if i == j else Fraction(0) for j in range(n)]
            for i in range(n)]


def _to_float(L: Matrix) -> np.ndarray:
    return np.array([[float(x) for x in row] for row in L], dtype=float)


# --------------------------------------------------------------------------- #
# Projective consistency (all exact over ℚ for the compatible family)
# --------------------------------------------------------------------------- #
def projective_commutation_residual(
    p: int, e: int, base: frozenset[int]
) -> Fraction:
    r"""Exact ``‖R_e P_{e+1} − P_e R_e‖_max`` for the compatible family.

    Zero iff fiber-averaging commutes with transport (projective transport).
    """
    R = projective_scale_map(p, e)
    Pe = padic_transition(p, e, compatible_connection_set(p, e, base))
    Pe1 = padic_transition(p, e + 1, compatible_connection_set(p, e + 1, base))
    return _maxabs(_sub(_matmul(R, Pe1), _matmul(Pe, R)))


def laplacian_commutation_residual(
    p: int, e: int, base: frozenset[int]
) -> Fraction:
    r"""Exact ``‖R_e L_{e+1} − L_e R_e‖_max`` (Laplacian projective transport)."""
    R = projective_scale_map(p, e)
    Le = padic_laplacian(p, e, compatible_connection_set(p, e, base))
    Le1 = padic_laplacian(p, e + 1, compatible_connection_set(p, e + 1, base))
    return _maxabs(_sub(_matmul(R, Le1), _matmul(Le, R)))


def lift_intertwining_residual(
    p: int, e: int, base: frozenset[int]
) -> Fraction:
    r"""Exact ``‖P_{e+1} Lift_e − Lift_e P_e‖_max``.

    Zero means the lift intertwines the two levels, so every coarse eigenvector
    lifts to a fine one with the same eigenvalue and ``spec(L_e) ⊆ spec(L_{e+1})``
    — the coarse modes survive the lift.
    """
    Lift = padic_lift_map(p, e)
    Pe = padic_transition(p, e, compatible_connection_set(p, e, base))
    Pe1 = padic_transition(p, e + 1, compatible_connection_set(p, e + 1, base))
    return _maxabs(_sub(_matmul(Pe1, Lift), _matmul(Lift, Pe)))


def lift_reduction_residual(p: int, e: int) -> Fraction:
    r"""Exact ``‖R_e Lift_e − I‖_max`` (the lift is a right inverse)."""
    R = projective_scale_map(p, e)
    Lift = padic_lift_map(p, e)
    return _maxabs(_sub(_matmul(R, Lift), _identity(p ** e)))


# --------------------------------------------------------------------------- #
# Spectral scaling telemetry
# --------------------------------------------------------------------------- #
def _eig_mags(L: Matrix) -> list[float]:
    return sorted(abs(z) for z in np.linalg.eigvals(_to_float(L)))


def surviving_spectrum_containment(
    p: int, e: int, base: frozenset[int]
) -> float:
    r"""Max distance from each ``|λ| ∈ spec(L_e)`` to the nearest fine eigenvalue.

    Numerical corroboration of ``spec(L_e) ⊆ spec(L_{e+1})`` (should sit below the
    derived tolerance); the exact proof is :func:`lift_intertwining_residual`.
    """
    Le = padic_laplacian(p, e, compatible_connection_set(p, e, base))
    Le1 = padic_laplacian(p, e + 1, compatible_connection_set(p, e + 1, base))
    coarse = np.linalg.eigvals(_to_float(Le))
    fine = np.linalg.eigvals(_to_float(Le1))
    worst = 0.0
    for lam in coarse:
        worst = max(worst, float(min(abs(lam - mu) for mu in fine)))
    return worst


def spectral_gap(L: Matrix) -> float:
    r"""Slowest non-zero relaxation ``λ₂`` = smallest ``|λ| > tol`` of ``L``."""
    arr = _to_float(L)
    tol = derived_tolerance(arr)
    for m in sorted(abs(z) for z in np.linalg.eigvals(arr)):
        if m > tol:
            return float(m)
    return 0.0


def padic_spectral_gaps(
    p: int, e_max: int, base: frozenset[int]
) -> list[tuple[int, float]]:
    r"""Spectral gap ``λ₂(L_e)`` for ``e = 1 … e_max`` (how the gap scales)."""
    out: list[tuple[int, float]] = []
    for e in range(1, e_max + 1):
        L = padic_laplacian(p, e, compatible_connection_set(p, e, base))
        out.append((e, spectral_gap(L)))
    return out


# --------------------------------------------------------------------------- #
# REMESH contract audit — the scale map is NOT REMESH until this passes
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RemeshContractAudit:
    """The four conditions the scale map must pass to be named REMESH.

    All default to ``False`` (unverified): the projective scale map is transport-
    consistent, but that alone does not establish the TNFR REMESH operator
    contract (recursive EPI echo across scales, NETWORK-scale generator/closure,
    preserved nodal identity, U5 multiscale coherence). U5 may be marked verified
    only from a declared parent/child hierarchy evaluated by the canonical U5
    assessment; scalar EPI dispersion is not such evidence. Until every field is
    independently verified, the map keeps its neutral name and the tower→REMESH
    claim (``NT-P04``) remains CONJECTURAL.
    """

    epi_recursion_verified: bool = False
    network_scale_verified: bool = False
    identity_preserved_verified: bool = False
    u5_multiscale_verified: bool = False

    @property
    def realizes_remesh(self) -> bool:
        """REMESH may be claimed only when every contract condition holds."""
        return (
            self.epi_recursion_verified
            and self.network_scale_verified
            and self.identity_preserved_verified
            and self.u5_multiscale_verified
        )

    def to_dict(self) -> dict:
        return {
            "epi_recursion_verified": self.epi_recursion_verified,
            "network_scale_verified": self.network_scale_verified,
            "identity_preserved_verified": self.identity_preserved_verified,
            "u5_multiscale_verified": self.u5_multiscale_verified,
            "realizes_remesh": self.realizes_remesh,
        }


def remesh_contract_audit() -> RemeshContractAudit:
    r"""Current REMESH-contract status for the projective scale map.

    Returns an audit with all conditions **unverified**: projective transport
    consistency is established (exact), but the REMESH operator contract is not,
    so no REMESH claim is permitted (``realizes_remesh == False``).
    """
    return RemeshContractAudit()
