r"""TNFR Emergent Symplectic Substrate — the geometry the dynamics generates.

This module establishes, as a first-class canonical object, the **symplectic
phase space that emerges from the TNFR nodal dynamics itself** — rather than
being imposed externally like the underlying graph. It is the substrate on
which the nodal equation, the conservation laws, and the 13 operators all
live as intrinsic geometric structures.

MOTIVATION (substrate from emergence, not imposition)
=====================================================
TNFR computes its fields on a *graph* G — an imposed combinatorial
substrate. But the Structural Conservation Theorem
(:mod:`tnfr.physics.conservation`) and the Variational Principle
(:mod:`tnfr.physics.variational`) reveal that the nodal dynamics carries
its **own** geometry: a symplectic phase space with canonical conjugate
pairs. This module makes that emergent geometry explicit and primary.

THE EMERGENT PHASE SPACE
========================
From the conservation-law structure, every node carries two canonical
conjugate pairs (position ↔ momentum):

    Geometric sector:  (q^A, p^A) = (K_φ,  J_φ)       curvature ↔ phase current
    Potential  sector: (q^B, p^B) = (Φ_s,  J_ΔNFR)    potential ↔ ΔNFR flux

So the phase space is P = ℝ^{4N} for an N-node network, with coordinates
z = (K_φ, J_φ, Φ_s, J_ΔNFR) per node. **These coordinates are derived from
canonical field functions** — this module never recomputes them, it
delegates to :mod:`tnfr.physics.canonical` and :mod:`tnfr.physics.extended`.

THE SYMPLECTIC FORM
===================
The emergent symplectic 2-form (named in :mod:`tnfr.physics.variational`):

    ω = Σ_i [ dK_φ(i) ∧ dJ_φ(i) + dΦ_s(i) ∧ dJ_ΔNFR(i) ].

In the per-node basis (q^A, p^A, q^B, p^B) its matrix is block-diagonal
with N copies of the canonical block

    J₄ = [[ 0, 1, 0, 0],
          [-1, 0, 0, 0],
          [ 0, 0, 0, 1],
          [ 0, 0,-1, 0]].

Verified properties (all exact): **antisymmetric** (J₄ᵀ = −J₄),
**non-degenerate** (det J₄ = 1), **closed** (dω = 0, constant
coefficients).

CANONICAL POISSON BRACKETS
==========================
The symplectic form induces the canonical bracket
{f, g} = (∇f)ᵀ J (∇g), giving

    {q^A_i, p^A_j} = δ_ij,   {q^B_i, p^B_j} = δ_ij,
    {q, q} = {p, p} = 0,     {A-sector, B-sector} = 0.

The Jacobi identity holds (constant J), so (P, {·,·}) is a Poisson
manifold.

THE HAMILTONIAN IS THE ENERGY FUNCTIONAL
========================================
The generator of the flow is the canonical TNFR energy (the same
:func:`tnfr.physics.conservation.compute_energy_functional`). Its
symplectic core — the part living on the conjugate pairs — is

    H_sub = ½ Σ_i [ K_φ² + J_φ² + Φ_s² + J_ΔNFR² ].

(The remaining ½Σ|∇φ|² term of the full energy is a configuration-space
background potential — it has no conjugate momentum and is not part of the
symplectic core.) The Hamiltonian flow X_H = J ∇H reproduces the harmonic
canonical dynamics q̇ = p, ṗ = −q per sector.

LIOUVILLE'S THEOREM (structural)
================================
Every Hamiltonian flow on this substrate is volume-preserving, for a
structural reason: div(X_H) = tr(J · Hess H) = 0 because J is
antisymmetric and the Hessian is symmetric. This is the geometric origin
of why the 13 operators preserve phase-space volume (they are
**symplectomorphisms** — verified by
:func:`tnfr.physics.variational.check_symplectic_preservation`).

THE NODAL EQUATION LIVES HERE
=============================
The nodal equation ∂EPI/∂t = νf·ΔNFR(t) is the **overdamped projection**
of the Hamiltonian flow on this substrate (Variational Principle §3.4),
with ΔNFR = −∂V/∂EPI the negative functional gradient of the structural
potential. The substrate is therefore not an analogy bolted on after the
fact: it is the geometric arena the nodal equation already inhabits.

HONEST SCOPE
============
- This module makes EXPLICIT and verifies the emergent symplectic
  structure already implied by conservation.py and variational.py. It is
  a *canonical consolidation*, not a new physical postulate.
- The phase-space coordinates are derived from existing canonical fields;
  no field formula is duplicated or redefined.
- The symplectic form, brackets, Liouville theorem, and operator
  symplectomorphism are EXACT structural results.
- The nodal-equation correspondence is the established overdamped limit
  of the variational principle (cited, not re-derived numerically).
- This does NOT, by itself, resolve any open program (Riemann G4, NS).
  It is foundational geometry: the substrate from which the canonical
  structures derive.

References
----------
- :mod:`tnfr.physics.variational` — Lagrangian/Hamiltonian, symplectic 2-form
- :mod:`tnfr.physics.conservation` — conjugate pairs, energy, Noether charge
- :mod:`tnfr.physics.gauge` — U(1) gauge structure of Psi = K_phi + i*J_phi
- theory/TNFR_VARIATIONAL_PRINCIPLE.md — full derivation
- AGENTS.md §"Minimal Structural Degrees of Freedom" — why the tetrad
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..mathematics.unified_numerical import np
from .canonical import (
    compute_structural_potential,
    compute_phase_gradient,
    compute_phase_curvature,
)
from .extended import (
    compute_phase_current,
    compute_dnfr_flux,
)

__all__ = [
    "BLOCK_SYMPLECTIC_FORM",
    "CONJUGATE_PAIR_LABELS",
    "PhaseSpacePoint",
    "CanonicalStructureCertificate",
    "extract_phase_space_point",
    "symplectic_form_matrix",
    "substrate_hamiltonian",
    "background_potential",
    "hamiltonian_vector_field",
    "poisson_bracket",
    "canonical_bracket_table",
    "liouville_divergence",
    "verify_canonical_structure",
]


# Canonical per-node block of the emergent symplectic form ω.
# Basis order: (q^A, p^A, q^B, p^B) = (K_φ, J_φ, Φ_s, J_ΔNFR).
BLOCK_SYMPLECTIC_FORM = np.array(
    [
        [0.0, 1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0, 0.0],
    ],
    dtype=float,
)

# Conjugate pair labels (position, momentum) per sector.
CONJUGATE_PAIR_LABELS = (
    ("K_phi", "J_phi"),       # geometric sector
    ("Phi_s", "J_dnfr"),      # potential sector
)


@dataclass(frozen=True)
class PhaseSpacePoint:
    r"""A point in the emergent symplectic phase space P = ℝ^{4N}.

    Coordinates per node ``i`` are the two canonical conjugate pairs

        (q^A, p^A) = (K_φ(i), J_φ(i))      geometric sector
        (q^B, p^B) = (Φ_s(i), J_ΔNFR(i))   potential sector

    Attributes
    ----------
    nodes : tuple
        Ordered node identifiers (defines the coordinate ordering).
    k_phi, j_phi, phi_s, j_dnfr : np.ndarray
        Per-node field arrays (shape ``(N,)``), aligned with ``nodes``.
    grad_phi : np.ndarray
        Per-node |∇φ| — the configuration-space background potential
        coordinate (NOT part of a conjugate pair).
    """

    nodes: tuple
    k_phi: Any
    j_phi: Any
    phi_s: Any
    j_dnfr: Any
    grad_phi: Any

    @property
    def n_nodes(self) -> int:
        """Number of network nodes N."""
        return len(self.nodes)

    @property
    def dimension(self) -> int:
        """Phase-space dimension 4N (two conjugate pairs per node)."""
        return 4 * len(self.nodes)

    def to_vector(self) -> Any:
        r"""Pack into the canonical phase-space vector z ∈ ℝ^{4N}.

        Ordering per node: (q^A, p^A, q^B, p^B) = (K_φ, J_φ, Φ_s, J_ΔNFR).
        """
        return np.stack(
            [self.k_phi, self.j_phi, self.phi_s, self.j_dnfr], axis=1
        ).reshape(-1)

    def gradient_vector(self) -> Any:
        r"""Gradient of the substrate Hamiltonian ∇H_sub at this point.

        Since H_sub = ½Σ(K_φ² + J_φ² + Φ_s² + J_ΔNFR²), the gradient is
        simply the coordinate vector z itself.
        """
        return self.to_vector()


@dataclass(frozen=True)
class CanonicalStructureCertificate:
    r"""Verification that the emergent geometry is a valid symplectic manifold.

    Attributes
    ----------
    n_nodes : int
    dimension : int
        Phase-space dimension 4N.
    is_antisymmetric : bool
        ω matrix satisfies Ωᵀ = −Ω.
    is_nondegenerate : bool
        det Ω ≠ 0 (here det Ω = 1).
    is_closed : bool
        dω = 0 (constant coefficients → exactly closed).
    brackets_canonical : bool
        {q^A,p^A} = {q^B,p^B} = 1, all cross/self brackets 0.
    jacobi_satisfied : bool
        Jacobi identity holds on sampled observables.
    liouville_divergence : float
        div(X_H) = tr(J·Hess H); should be 0 (volume-preserving flow).
    flow_is_harmonic : bool
        Hamiltonian flow reproduces q̇ = p, ṗ = −q per sector.
    determinant : float
        det Ω (= 1 for the canonical form).
    """

    n_nodes: int
    dimension: int
    is_antisymmetric: bool
    is_nondegenerate: bool
    is_closed: bool
    brackets_canonical: bool
    jacobi_satisfied: bool
    liouville_divergence: float
    flow_is_harmonic: bool
    determinant: float

    @property
    def is_valid_symplectic_manifold(self) -> bool:
        """True when all structural conditions hold."""
        return (
            self.is_antisymmetric
            and self.is_nondegenerate
            and self.is_closed
            and self.brackets_canonical
            and self.jacobi_satisfied
            and abs(self.liouville_divergence) < 1e-9
            and self.flow_is_harmonic
        )

    def summary(self) -> str:
        """Human-readable one-line verdict."""
        ok = "VALID" if self.is_valid_symplectic_manifold else "INVALID"
        return (
            f"Emergent symplectic substrate [{ok}]: dim={self.dimension}, "
            f"antisym={self.is_antisymmetric}, "
            f"nondeg={self.is_nondegenerate}, "
            f"closed={self.is_closed}, canonical_brackets="
            f"{self.brackets_canonical}, jacobi={self.jacobi_satisfied}, "
            f"div(X_H)={self.liouville_divergence:.2e}, "
            f"harmonic_flow={self.flow_is_harmonic}"
        )


def extract_phase_space_point(G: Any) -> PhaseSpacePoint:
    r"""Extract the emergent phase-space point z(G) from a TNFR network.

    Delegates to the canonical field functions (no recomputation of
    field formulas):

    - K_φ   ← :func:`tnfr.physics.canonical.compute_phase_curvature`
    - J_φ   ← :func:`tnfr.physics.extended.compute_phase_current`
    - Φ_s   ← :func:`tnfr.physics.canonical.compute_structural_potential`
    - J_ΔNFR ← :func:`tnfr.physics.extended.compute_dnfr_flux`
    - |∇φ|  ← :func:`tnfr.physics.canonical.compute_phase_gradient`

    Parameters
    ----------
    G : TNFRGraph

    Returns
    -------
    PhaseSpacePoint
    """
    nodes = tuple(sorted(G.nodes(), key=lambda x: (str(type(x)), repr(x))))
    k_phi = compute_phase_curvature(G)
    j_phi = compute_phase_current(G)
    phi_s = compute_structural_potential(G)
    j_dnfr = compute_dnfr_flux(G)
    grad_phi = compute_phase_gradient(G)

    def _arr(d: dict[Any, float]) -> Any:
        return np.array([float(d.get(n, 0.0)) for n in nodes], dtype=float)

    return PhaseSpacePoint(
        nodes=nodes,
        k_phi=_arr(k_phi),
        j_phi=_arr(j_phi),
        phi_s=_arr(phi_s),
        j_dnfr=_arr(j_dnfr),
        grad_phi=_arr(grad_phi),
    )


def symplectic_form_matrix(n_nodes: int) -> Any:
    r"""Return the emergent symplectic form ω as a 4N×4N block matrix.

    Block-diagonal with ``n_nodes`` copies of :data:`BLOCK_SYMPLECTIC_FORM`.

    Parameters
    ----------
    n_nodes : int

    Returns
    -------
    np.ndarray
        Antisymmetric, non-degenerate (det = 1), constant matrix.
    """
    if n_nodes < 1:
        raise ValueError("n_nodes must be >= 1")
    dim = 4 * n_nodes
    omega = np.zeros((dim, dim), dtype=float)
    for i in range(n_nodes):
        s = 4 * i
        omega[s:s + 4, s:s + 4] = BLOCK_SYMPLECTIC_FORM
    return omega


def substrate_hamiltonian(point: PhaseSpacePoint) -> float:
    r"""Symplectic-core Hamiltonian H_sub = ½Σ(K_φ²+J_φ²+Φ_s²+J_ΔNFR²).

    This is the part of the canonical TNFR energy functional that lives on
    the conjugate pairs (the symplectic core).  The full energy adds the
    configuration background ½Σ|∇φ|² (see :func:`background_potential`).
    """
    z = point.to_vector()
    return 0.5 * float(np.dot(z, z))


def background_potential(point: PhaseSpacePoint) -> float:
    r"""Configuration-space background potential U = ½Σ|∇φ|².

    The |∇φ| term of the full TNFR energy has no conjugate momentum, so it
    is a background potential on the base manifold, not part of the
    symplectic core.  ``substrate_hamiltonian + background_potential``
    equals the full energy functional.
    """
    g = np.asarray(point.grad_phi, dtype=float)
    return 0.5 * float(np.dot(g, g))


def hamiltonian_vector_field(point: PhaseSpacePoint) -> Any:
    r"""Hamiltonian vector field X_H = J ∇H_sub on the substrate.

    Returns the phase-space velocity ż = J ∇H_sub ∈ ℝ^{4N}.  Per node and
    sector this is the harmonic canonical flow

        q̇^A = J_φ,   ṗ^A = −K_φ,   q̇^B = J_ΔNFR,   ṗ^B = −Φ_s,

    i.e. q̇ = p and ṗ = −q (Hamilton's equations for H = ½(q²+p²)).

    Parameters
    ----------
    point : PhaseSpacePoint

    Returns
    -------
    np.ndarray
        ż of shape ``(4N,)``.
    """
    omega = symplectic_form_matrix(point.n_nodes)
    grad_h = point.gradient_vector()
    return omega @ grad_h


def poisson_bracket(
    grad_f: Any,
    grad_g: Any,
    n_nodes: int,
) -> float:
    r"""Canonical Poisson bracket {f, g} = (∇f)ᵀ J (∇g).

    Parameters
    ----------
    grad_f, grad_g : np.ndarray
        Gradients of the observables f, g with respect to the phase-space
        coordinates z ∈ ℝ^{4N}.
    n_nodes : int

    Returns
    -------
    float
    """
    omega = symplectic_form_matrix(n_nodes)
    gf = np.asarray(grad_f, dtype=float)
    gg = np.asarray(grad_g, dtype=float)
    return float(gf @ (omega @ gg))


def canonical_bracket_table(n_nodes: int = 1) -> dict[str, float]:
    r"""Return the canonical coordinate brackets for one node.

    Computes {q^A,p^A}, {q^B,p^B}, {q^A,q^B}, {p^A,p^B}, {q^A,p^B} from the
    symplectic form, confirming the canonical structure
    {q,p}=δ, {q,q}={p,p}=0, cross-sector = 0.

    Returns
    -------
    dict[str, float]
    """
    dim = 4 * n_nodes
    # Coordinate gradients for node 0: indices qA=0, pA=1, qB=2, pB=3.
    e = [np.zeros(dim) for _ in range(4)]
    for idx in range(4):
        e[idx][idx] = 1.0
    qa, pa, qb, pb = e
    return {
        "{qA,pA}": poisson_bracket(qa, pa, n_nodes),
        "{qB,pB}": poisson_bracket(qb, pb, n_nodes),
        "{qA,qB}": poisson_bracket(qa, qb, n_nodes),
        "{pA,pB}": poisson_bracket(pa, pb, n_nodes),
        "{qA,pB}": poisson_bracket(qa, pb, n_nodes),
        "{pA,qB}": poisson_bracket(pa, qb, n_nodes),
    }


def liouville_divergence(point: PhaseSpacePoint) -> float:
    r"""Phase-space divergence of the Hamiltonian flow, div(X_H).

    Equals tr(J · Hess H_sub).  For H_sub = ½|z|² the Hessian is the
    identity, so div(X_H) = tr(J) = 0 exactly — Liouville's theorem.  This
    is the structural reason the flow (and the operators that generate it)
    preserve phase-space volume.

    Returns
    -------
    float
        Should be 0 to machine precision.
    """
    omega = symplectic_form_matrix(point.n_nodes)
    # Hess H_sub = I (substrate Hamiltonian is ½|z|²).
    hess = np.eye(4 * point.n_nodes, dtype=float)
    return float(np.trace(omega @ hess))


def _check_jacobi(n_nodes: int) -> bool:
    r"""Verify the Jacobi identity on sampled quadratic observables.

    For the constant symplectic form, {{f,g},h} + cyclic = 0 holds
    identically.  We confirm numerically on three quadratic observables
    whose brackets are linear, then bracket again.
    """
    dim = 4 * n_nodes
    rng = np.random.default_rng(20260613)
    # Quadratic observables f = ½ zᵀ A z with symmetric A.
    mats = []
    for _ in range(3):
        a = rng.standard_normal((dim, dim))
        mats.append(0.5 * (a + a.T))
    omega = symplectic_form_matrix(n_nodes)
    z = rng.standard_normal(dim)

    def grad(a: Any) -> Any:
        return a @ z

    # Value-level Jacobi via the bracket of brackets on quadratics.
    # Constant J ⇒ Jacobi holds; verify the cyclic sum vanishes.
    a, b, c = mats
    # {f,g} = (Az)ᵀ J (Bz) = zᵀ Aᵀ J B z. Its gradient = (AᵀJB + BᵀJ ᵀA) z.
    g_fg = (a.T @ omega @ b + b.T @ omega.T @ a) @ z
    g_h = grad(c)
    lhs1 = float(g_fg @ (omega @ g_h))
    g_gh = (b.T @ omega @ c + c.T @ omega.T @ b) @ z
    g_f = grad(a)
    lhs2 = float(g_gh @ (omega @ g_f))
    g_hf = (c.T @ omega @ a + a.T @ omega.T @ c) @ z
    g_g = grad(b)
    lhs3 = float(g_hf @ (omega @ g_g))
    return abs(lhs1 + lhs2 + lhs3) < 1e-6


def verify_canonical_structure(G: Any) -> CanonicalStructureCertificate:
    r"""Verify the emergent geometry of ``G`` is a valid symplectic manifold.

    Checks all structural conditions: antisymmetry, non-degeneracy,
    closedness, canonical Poisson brackets, the Jacobi identity, Liouville
    volume preservation, and the harmonic Hamiltonian flow.

    Parameters
    ----------
    G : TNFRGraph

    Returns
    -------
    CanonicalStructureCertificate
    """
    point = extract_phase_space_point(G)
    n = point.n_nodes
    omega = symplectic_form_matrix(max(n, 1))

    is_antisym = bool(np.allclose(omega.T, -omega))
    det = float(np.linalg.det(BLOCK_SYMPLECTIC_FORM))
    is_nondeg = abs(det) > 1e-12
    is_closed = True  # constant coefficients ⇒ dω = 0 exactly

    table = canonical_bracket_table(1)
    brackets_ok = (
        abs(table["{qA,pA}"] - 1.0) < 1e-12
        and abs(table["{qB,pB}"] - 1.0) < 1e-12
        and abs(table["{qA,qB}"]) < 1e-12
        and abs(table["{pA,pB}"]) < 1e-12
        and abs(table["{qA,pB}"]) < 1e-12
        and abs(table["{pA,qB}"]) < 1e-12
    )

    jacobi_ok = _check_jacobi(1)

    if n >= 1:
        div = liouville_divergence(point)
        # Harmonic flow check: ż = J∇H should give q̇=p, ṗ=−q.
        z = point.to_vector()
        zdot = hamiltonian_vector_field(point)
        zr = z.reshape(n, 4)
        zdr = zdot.reshape(n, 4)
        # Expect (q̇A,ṗA,q̇B,ṗB) = (pA,−qA,pB,−qB).
        expected = np.stack(
            [zr[:, 1], -zr[:, 0], zr[:, 3], -zr[:, 2]], axis=1
        )
        flow_ok = bool(np.allclose(zdr, expected))
    else:
        div = 0.0
        flow_ok = True

    return CanonicalStructureCertificate(
        n_nodes=n,
        dimension=4 * n,
        is_antisymmetric=is_antisym,
        is_nondegenerate=is_nondeg,
        is_closed=is_closed,
        brackets_canonical=brackets_ok,
        jacobi_satisfied=jacobi_ok,
        liouville_divergence=div,
        flow_is_harmonic=flow_ok,
        determinant=det,
    )
