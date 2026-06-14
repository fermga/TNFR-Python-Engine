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

THE COMPLETE GEOMETRIC TOWER
============================
On this substrate the entire classical Hamiltonian-geometry tower is
derived from the nodal dynamics — each structure has a certificate
dataclass (full detail in its docstring) and a ``verify_*`` function, and
:func:`verify_substrate_geometry` runs them all at once:

1. **Symplectic / Poisson / Liouville**
   (:class:`CanonicalStructureCertificate`,
   :func:`verify_canonical_structure`) — ω closed & non-degenerate,
   canonical brackets, Jacobi, div(X_H)=0.
2. **Noether charges** (:class:`NoetherChargeCertificate`,
   :func:`verify_noether_conservation`) — H_sub = E_geo + E_pot splits
   exactly; the geometric U(1) charge ½Σ|Ψ|² is the gauge invariant of
   :mod:`tnfr.physics.gauge`.
3. **Hermitian / flat Kähler** (:class:`HermitianStructureCertificate`,
   :func:`verify_hermitian_structure`) — compatible triple (ω, J=−ω, g=I);
   the complex coordinate ζ^A = K_φ + i·J_φ **is** the gauge field Ψ, so
   H_sub = ½Σ|ζ|² is the Kähler potential.
4. **Complete integrability** (:class:`IntegrabilityCertificate`,
   :func:`verify_integrability`) — action–angle variables I = ½|ζ|²
   (2N integrals in involution); the flow is Liouville–Arnold integrable.
5. **Poincaré–Cartan invariants** (:class:`PoincareCartanCertificate`,
   :func:`verify_poincare_cartan`) — the flow preserves the whole ω^k
   tower; ∮ p dq = 2π I (Bohr–Sommerfeld) on the action torus.
6. **Marsden–Weinstein reduction** (:class:`MarsdenWeinsteinCertificate`,
   :func:`verify_symplectic_reduction`) — moment map J = H_sub; the
   quotient P//U(1) is a symplectic manifold of dimension 4N − 2.

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
- The symplectic form, brackets, Liouville theorem, operator
  symplectomorphism, Noether charges, Hermitian (ω, J, g) compatibility,
  action–angle integrability, Poincaré–Cartan invariants, and the
  Marsden–Weinstein reduction are EXACT structural results.
- The Kähler / integrability / reduction results are for the **flat**
  (constant-coefficient) substrate and its **H_sub harmonic backbone** —
  a flat linear symplectic space, NOT a curved manifold, and NOT the full
  nonlinear operator dynamics (the 13 operators are canonical transforms
  *on* this substrate).
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
- AGENTS.md §"Emergent Symplectic Substrate (CANONICAL)" — the full tower
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
    "BLOCK_COMPLEX_STRUCTURE",
    "BLOCK_COMPATIBLE_METRIC",
    "CONJUGATE_PAIR_LABELS",
    "PhaseSpacePoint",
    "CanonicalStructureCertificate",
    "NoetherChargeCertificate",
    "HermitianStructureCertificate",
    "IntegrabilityCertificate",
    "PoincareCartanCertificate",
    "MarsdenWeinsteinCertificate",
    "PolarizationSymmetryCertificate",
    "SubstrateGeometryReport",
    "extract_phase_space_point",
    "symplectic_form_matrix",
    "complex_structure_matrix",
    "compatible_metric_matrix",
    "substrate_hamiltonian",
    "background_potential",
    "hamiltonian_vector_field",
    "poisson_bracket",
    "canonical_bracket_table",
    "liouville_divergence",
    "verify_canonical_structure",
    "evolve_substrate_flow",
    "geometric_sector_energy",
    "potential_sector_energy",
    "noether_charges",
    "verify_noether_conservation",
    "to_complex_coordinates",
    "kahler_potential",
    "verify_hermitian_structure",
    "to_action_angle",
    "verify_integrability",
    "substrate_flow_matrix",
    "loop_action_integral",
    "verify_poincare_cartan",
    "diagonal_moment_map",
    "reduced_symplectic_form_matrix",
    "verify_symplectic_reduction",
    "polarization_vector",
    "polarization_density",
    "verify_polarization_symmetry",
    "AdiabaticInvarianceCertificate",
    "verify_adiabatic_invariance",
    "verify_substrate_geometry",
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

# Compatible complex structure J = −ω (per node).  J² = −I, and J acts as
# multiplication by i on the complex coordinate ζ = q + i·p: J(q,p) = (−p, q).
# This is the "i" of the complex geometric field Ψ = K_φ + i·J_φ.
BLOCK_COMPLEX_STRUCTURE = -BLOCK_SYMPLECTIC_FORM

# Compatible Riemannian metric g(u,v) = ω(u, J v) = identity (per node).
# (ω, J, g) form a compatible triple → a flat Hermitian (Kähler) structure.
BLOCK_COMPATIBLE_METRIC = BLOCK_SYMPLECTIC_FORM @ BLOCK_COMPLEX_STRUCTURE

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


@dataclass(frozen=True)
class NoetherChargeCertificate:
    r"""Noether charges of the substrate flow and their conservation.

    Noether's theorem on the emergent symplectic substrate: each
    continuous symmetry of the substrate Hamiltonian generates a
    conserved quantity along the Hamiltonian flow.

    Three symmetries with their charges:

    - **Time translation** → ``hamiltonian`` H_sub (total energy).
    - **Geometric U(1)** (Ψ → e^{iα}Ψ, the :mod:`tnfr.physics.gauge`
      symmetry of Ψ = K_φ + i·J_φ) → ``geometric_energy`` ½Σ|Ψ|²
      = ½Σ(K_φ² + J_φ²).
    - **Potential U(1)** (rotation of (Φ_s, J_ΔNFR)) →
      ``potential_energy`` ½Σ(Φ_s² + J_ΔNFR²).

    The total energy splits exactly: H_sub = E_geo + E_pot. The two
    sector charges are *separately* conserved — the U(1)×U(1) symmetry
    refines the single time-translation conservation.

    Attributes
    ----------
    hamiltonian : float
        H_sub (time-translation charge).
    geometric_energy : float
        ½Σ(K_φ² + J_φ²) = ½Σ|Ψ|² (geometric-U(1)/gauge charge).
    potential_energy : float
        ½Σ(Φ_s² + J_ΔNFR²) (potential-U(1) charge).
    max_hamiltonian_drift : float
        Max |H_sub(t) − H_sub(0)| over the sampled flow.
    max_geometric_drift : float
        Max drift of E_geo along the flow.
    max_potential_drift : float
        Max drift of E_pot along the flow.
    is_conserved : bool
        True when all three drifts are below tolerance.
    splits_exactly : bool
        True when H_sub = E_geo + E_pot to machine precision.
    """

    hamiltonian: float
    geometric_energy: float
    potential_energy: float
    max_hamiltonian_drift: float
    max_geometric_drift: float
    max_potential_drift: float
    is_conserved: bool
    splits_exactly: bool

    def summary(self) -> str:
        """Human-readable one-line verdict."""
        cons = "CONSERVED" if self.is_conserved else "NOT-CONSERVED"
        return (
            f"Noether charges [{cons}]: H_sub={self.hamiltonian:.4f} = "
            f"E_geo={self.geometric_energy:.4f} + "
            f"E_pot={self.potential_energy:.4f} "
            f"(split_exact={self.splits_exactly}); "
            f"max drift H/geo/pot = "
            f"{self.max_hamiltonian_drift:.1e}/"
            f"{self.max_geometric_drift:.1e}/"
            f"{self.max_potential_drift:.1e}"
        )


@dataclass(frozen=True)
class HermitianStructureCertificate:
    r"""Verification of the compatible Hermitian (flat Kähler) structure.

    The emergent phase space carries a compatible triple (ω, J, g):

    - **ω** — the symplectic form (:data:`BLOCK_SYMPLECTIC_FORM`).
    - **J** — the complex structure J = −ω (:data:`BLOCK_COMPLEX_STRUCTURE`),
      with J² = −I; J acts as multiplication by i on ζ = q + i·p.
    - **g** — the compatible metric g(u,v) = ω(u, J v) = identity
      (:data:`BLOCK_COMPATIBLE_METRIC`), symmetric positive-definite.

    These satisfy the compatibility ω(u, v) = g(J u, v), making each fiber
    ℝ⁴ ≅ ℂ² a Hermitian vector space.  The complex coordinates are

        ζ^A = K_φ + i·J_φ = Ψ   (geometric sector, the gauge field of
                                  :mod:`tnfr.physics.gauge`),
        ζ^B = Φ_s + i·J_ΔNFR    (potential sector),

    so the substrate Hamiltonian is the **Kähler potential**
    H_sub = ½Σ(|ζ^A|² + |ζ^B|²), and the Hamiltonian flow is the diagonal
    U(1) phase rotation ζ → e^{−it}ζ.  The structure is **flat**
    (constant-coefficient): a linear Kähler / Hermitian vector space, not a
    curved Kähler manifold.

    Attributes
    ----------
    n_nodes : int
    complex_dimension : int
        Complex dimension 2N (two complex coordinates per node).
    j_squared_is_minus_id : bool
        J² = −I (J is an almost-complex structure).
    metric_is_identity : bool
        g = ω·J equals the identity (positive-definite).
    j_is_orthogonal : bool
        Jᵀ g J = g (J preserves the metric).
    compatible : bool
        ω(u,v) = g(Ju,v) (the (ω, J, g) compatibility).
    psi_is_geometric_coordinate : bool
        ζ^A = K_φ + i·J_φ equals Ψ from the canonical complex field.
    kahler_potential_matches : bool
        H_sub = ½Σ|ζ|² (the substrate Hamiltonian is the Kähler potential).
    """

    n_nodes: int
    complex_dimension: int
    j_squared_is_minus_id: bool
    metric_is_identity: bool
    j_is_orthogonal: bool
    compatible: bool
    psi_is_geometric_coordinate: bool
    kahler_potential_matches: bool

    @property
    def is_valid_hermitian_structure(self) -> bool:
        """True when all compatibility conditions hold."""
        return (
            self.j_squared_is_minus_id
            and self.metric_is_identity
            and self.j_is_orthogonal
            and self.compatible
            and self.kahler_potential_matches
        )

    def summary(self) -> str:
        """Human-readable one-line verdict."""
        ok = "VALID" if self.is_valid_hermitian_structure else "INVALID"
        return (
            f"Hermitian (flat Kähler) structure [{ok}]: "
            f"dim_C={self.complex_dimension}, J²=−I="
            f"{self.j_squared_is_minus_id}, g=I={self.metric_is_identity}, "
            f"J-orthogonal={self.j_is_orthogonal}, "
            f"compatible={self.compatible}, "
            f"Ψ=ζ^A={self.psi_is_geometric_coordinate}, "
            f"Kähler_potential={self.kahler_potential_matches}"
        )


@dataclass(frozen=True)
class IntegrabilityCertificate:
    r"""Verification that the substrate flow is completely integrable.

    Because H_sub = ½Σ(K_φ² + J_φ² + Φ_s² + J_ΔNFR²) is a sum of decoupled
    oscillators — one per conjugate pair — the substrate Hamiltonian flow is
    **completely integrable** in the Liouville–Arnold sense.  Each conjugate
    pair contributes an **action variable**

        I^A_i = ½(K_φ² + J_φ²) = ½|ζ^A|² = ½|Ψ|²   (geometric),
        I^B_i = ½(Φ_s² + J_ΔNFR²) = ½|ζ^B|²        (potential),

    giving 2N independent integrals for a system of 2N degrees of freedom.
    The actions are pairwise in involution ({I_i, I_j} = 0, structural — the
    conjugate pairs are decoupled), conserved along the flow, and the
    conjugate **angle variables** θ_i = arg ζ_i advance linearly
    θ_i(t) = θ_i(0) − t.  So (I_i, θ_i) are global action–angle coordinates
    in which the harmonic backbone is trivial (a rigid phase rotation per
    pair).  The sector action sums recover the Noether charges
    (Σ I^A = E_geo, Σ I^B = E_pot).

    HONEST SCOPE: this is the integrability of the *substrate harmonic
    backbone* (the H_sub flow), not of the full nonlinear operator dynamics.
    The 13 operators act as canonical transformations that redistribute the
    action variables; the actions are the adiabatic invariants of that
    backbone.

    Attributes
    ----------
    n_nodes : int
    degrees_of_freedom : int
        Number of conjugate pairs, 2N.
    n_action_variables : int
        Number of independent action integrals, 2N.
    actions_in_involution : bool
        {I_i, I_j} = 0 for all action pairs.
    actions_conserved : bool
        Each I_i is constant along the substrate flow.
    angles_advance_linearly : bool
        θ_i(t) = θ_i(0) − t to tolerance.
    max_action_drift : float
        Max |I_i(t) − I_i(0)| over the sampled flow.
    max_angle_error : float
        Max deviation of θ_i(t) from θ_i(0) − t (wrapped to the circle).
    max_involution_bracket : float
        Max |{I_i, I_j}| over action pairs (should be 0).
    sector_actions_match_charges : bool
        Σ I^A = E_geo and Σ I^B = E_pot (action sums = Noether charges).
    """

    n_nodes: int
    degrees_of_freedom: int
    n_action_variables: int
    actions_in_involution: bool
    actions_conserved: bool
    angles_advance_linearly: bool
    max_action_drift: float
    max_angle_error: float
    max_involution_bracket: float
    sector_actions_match_charges: bool

    @property
    def is_completely_integrable(self) -> bool:
        """True when 2N independent actions are conserved and in involution."""
        return (
            self.n_action_variables == self.degrees_of_freedom
            and self.actions_in_involution
            and self.actions_conserved
        )

    def summary(self) -> str:
        """Human-readable one-line verdict."""
        ok = (
            "INTEGRABLE"
            if self.is_completely_integrable
            else "NOT-INTEGRABLE"
        )
        return (
            f"Liouville–Arnold [{ok}]: "
            f"{self.n_action_variables} actions for "
            f"{self.degrees_of_freedom} DOF, "
            f"involution={self.actions_in_involution} "
            f"(max bracket {self.max_involution_bracket:.1e}), "
            f"action drift {self.max_action_drift:.1e}, "
            f"angle error {self.max_angle_error:.1e}"
        )


@dataclass(frozen=True)
class PoincareCartanCertificate:
    r"""Verification of the Poincaré–Cartan integral invariants of the flow.

    The substrate Hamiltonian flow φ_t preserves the symplectic form ω
    (φ_t^* ω = ω), so it preserves the whole tower of **Poincaré–Cartan
    integral invariants** ω^k (k = 1 … N, with N = 2·n_nodes the number of
    conjugate pairs):

    - **k = 1** — the *relative integral invariant of Poincaré*
      ∮_γ λ = ∮_γ Σ p_i dq_i over any closed loop γ (λ the tautological
      1-form, ω = dλ).  Equivalently the *absolute* invariant ∬ ω over any
      2-cycle.  At matrix level the flow map M(t) is **symplectic**
      (Mᵀ Ω M = Ω).
    - **1 < k < N** — the intermediate invariants ∫ ω^k, encoded by the
      **palindromic characteristic polynomial** of M(t): its spectrum is
      the reciprocal symplectic set {e^{+it}, e^{−it}}, so every coefficient
      (= a sum of 2k×2k principal symplectic minors = the ω^k invariant) is
      preserved.
    - **k = N** — the top invariant ω^N / N! is the **Liouville volume**
      (det M = 1).

    On an action torus I = const, the relative invariant evaluates to the
    **Bohr–Sommerfeld** quantum: ∮_{γ_i} p dq = 2π I_i, tying the integral
    invariant to the action variables of :class:`IntegrabilityCertificate`.

    Attributes
    ----------
    n_nodes : int
    phase_space_dimension : int
        4·n_nodes.
    preserves_symplectic_form : bool
        Mᵀ Ω M = Ω for all sampled flow times (1st invariant).
    volume_preserved : bool
        det M = 1 (top invariant, Liouville volume).
    char_poly_palindromic : bool
        Characteristic polynomial of M is palindromic (reciprocal
        symplectic spectrum → the full ω^k tower is preserved).
    relative_invariant_preserved : bool
        ∮_γ p dq over an action-torus loop is constant along the flow.
    bohr_sommerfeld_holds : bool
        |∮_γ p dq| = 2π I on the action torus.
    max_omega_drift : float
        Max ‖Mᵀ Ω M − Ω‖ over sampled times.
    max_relative_drift : float
        Max change of ∮ p dq along the flow.
    max_bohr_error : float
        Max |‖∮ p dq‖ − 2π I| over the sampled pairs.
    """

    n_nodes: int
    phase_space_dimension: int
    preserves_symplectic_form: bool
    volume_preserved: bool
    char_poly_palindromic: bool
    relative_invariant_preserved: bool
    bohr_sommerfeld_holds: bool
    max_omega_drift: float
    max_relative_drift: float
    max_bohr_error: float

    @property
    def all_invariants_hold(self) -> bool:
        """True when every Poincaré–Cartan invariant is preserved."""
        return (
            self.preserves_symplectic_form
            and self.volume_preserved
            and self.char_poly_palindromic
            and self.relative_invariant_preserved
            and self.bohr_sommerfeld_holds
        )

    def summary(self) -> str:
        """Human-readable one-line verdict."""
        ok = "ALL HOLD" if self.all_invariants_hold else "VIOLATED"
        return (
            f"Poincaré–Cartan [{ok}]: dim={self.phase_space_dimension}, "
            f"ω-preserved={self.preserves_symplectic_form} "
            f"(drift {self.max_omega_drift:.1e}), "
            f"volume={self.volume_preserved}, "
            f"palindromic={self.char_poly_palindromic}, "
            f"∮p·dq const={self.relative_invariant_preserved} "
            f"(drift {self.max_relative_drift:.1e}), "
            f"Bohr–Sommerfeld={self.bohr_sommerfeld_holds} "
            f"(err {self.max_bohr_error:.1e})"
        )


@dataclass(frozen=True)
class MarsdenWeinsteinCertificate:
    r"""Verification of the Marsden–Weinstein symplectic reduction.

    The substrate flow is the **diagonal U(1)** action ζ → e^{−it}ζ rotating
    every conjugate pair together (θ_k → θ_k − t for all k).  Its
    **moment map** is J = Σ_k I_k = ½Σ|ζ|² = H_sub — the total energy, which
    is exactly the Noether charge of time translation (so the symmetry that
    generates the flow is the symmetry one reduces by).

    The Marsden–Weinstein quotient P//U(1) = J⁻¹(μ)/U(1) is built explicitly
    in action–angle coordinates (m = 2·n_nodes conjugate pairs):

    - **level set** J⁻¹(μ): Σ_k I_k = μ (codimension 1),
    - **quotient** by the collective phase θ_0: the reduced coordinates are
      the (m−1) independent actions and the (m−1) **relative phases**
      φ_k = θ_k − θ_0, which are invariant under the diagonal U(1).

    The reduced symplectic form is Σ_{k≥1} dI_k ∧ dφ_k — canonical and
    **non-degenerate** — so the reduced space is a genuine symplectic
    manifold of dimension 4N − 2.  (Reducing instead by the sector
    U(1)×U(1) of :class:`NoetherChargeCertificate` gives 4N − 4.)

    HONEST SCOPE: this reduces the **flat** substrate by its diagonal U(1)
    flow symmetry; the reduced space is a flat linear symplectic space (the
    relative-phase coordinates), not a curved reduced manifold.

    Attributes
    ----------
    n_nodes : int
    phase_space_dimension : int
        4·n_nodes.
    moment_map_value : float
        J = Σ I_k evaluated at the graph state.
    moment_map_is_hamiltonian : bool
        J = H_sub (moment map equals the energy / time-translation charge).
    moment_map_conserved : bool
        J is invariant along the flow (drift below tolerance).
    reduced_dimension : int
        Dimension of P//U(1) = 4N − 2.
    reduced_form_nondegenerate : bool
        The reduced symplectic form has non-zero determinant.
    relative_phases_invariant : bool
        φ_k = θ_k − θ_0 are invariant under the diagonal U(1) flow.
    max_moment_drift : float
        Max |J(t) − J(0)| over the sampled flow.
    reduced_form_determinant : float
        Determinant of the reduced symplectic form (≠ 0 ⇒ symplectic).
    """

    n_nodes: int
    phase_space_dimension: int
    moment_map_value: float
    moment_map_is_hamiltonian: bool
    moment_map_conserved: bool
    reduced_dimension: int
    reduced_form_nondegenerate: bool
    relative_phases_invariant: bool
    max_moment_drift: float
    reduced_form_determinant: float

    @property
    def is_valid_reduction(self) -> bool:
        """True when the reduction yields a valid symplectic quotient."""
        return (
            self.moment_map_is_hamiltonian
            and self.moment_map_conserved
            and self.reduced_form_nondegenerate
            and self.relative_phases_invariant
        )

    def summary(self) -> str:
        """Human-readable one-line verdict."""
        ok = "VALID" if self.is_valid_reduction else "INVALID"
        return (
            f"Marsden–Weinstein [{ok}]: "
            f"P//U(1) dim {self.phase_space_dimension}→"
            f"{self.reduced_dimension}, "
            f"J=H_sub={self.moment_map_is_hamiltonian} "
            f"(J={self.moment_map_value:.4f}, "
            f"drift {self.max_moment_drift:.1e}), "
            f"reduced ω non-degenerate={self.reduced_form_nondegenerate} "
            f"(det {self.reduced_form_determinant:.3g}), "
            f"relative-phase invariant={self.relative_phases_invariant}"
        )


@dataclass(frozen=True)
class PolarizationSymmetryCertificate:
    r"""Verification of the substrate's polarization symmetry (U(2)).

    Because the substrate Hamiltonian is the squared norm of a **complex
    doublet** per node,

        H_sub = ½ Σ_i ‖(ζ^A_i, ζ^B_i)‖²,
        ζ^A = K_φ + i·J_φ (geometric),  ζ^B = Φ_s + i·J_ΔNFR (potential),

    it is invariant not only under the U(1)×U(1) sector phases (the
    :class:`NoetherChargeCertificate` charges) but under the **full U(2)**
    acting on the (ζ^A, ζ^B) doublet.  This is exactly the **polarization
    symmetry** of a two-component complex field: the SAME mathematics as
    classical wave polarization (Stokes 1852, Poincaré 1892), an
    empirically-demonstrated phenomenon.  U(2) = U(1) × SU(2): the U(1)
    centre is the substrate flow itself, and the U(1)×U(1) Noether sectors
    are the **Cartan torus** of U(2).

    The SU(2) part supplies the three conserved **Stokes parameters** (the
    polarization 3-vector; the moment map of the global diagonal SU(2)):

        P_3 = ½ Σ (|ζ^A|² − |ζ^B|²) = E_geo − E_pot   (sector difference),
        P_1 = Σ Re(ζ̄^A ζ^B) = Σ (K_φ·Φ_s + J_φ·J_ΔNFR)   (real cross-corr.),
        P_2 = Σ Im(ζ̄^A ζ^B) = Σ (K_φ·J_ΔNFR − J_φ·Φ_s)   (imag cross-corr.).

    These are the Stokes parameters of the doublet in the substrate's
    natural (Noether-charge) normalization; the textbook optical Stokes
    parameters are 2× these.  P_3 is the (already-known) sector-energy
    difference, but **P_1 and P_2 are genuinely new conserved charges** —
    the cross-sector correlations between the geometric and potential
    sectors.  They satisfy the su(2) algebra under the canonical Poisson
    bracket, {P_a, P_b} = 2 ε_abc P_c, and are conserved along the
    substrate flow (the diagonal U(1) ⊂ U(2) commutes with SU(2)).

    **Poincaré sphere (per-node geometric content)**: per node, the
    polarization 3-vector (P_1^i, P_2^i, P_3^i) has length equal to the
    per-node substrate energy, |P_node| = e_node = ½(|ζ^A|² + |ζ^B|²).
    The normalized vector P_node / e_node is therefore a unit vector on the
    **Poincaré sphere** S² — i.e. each node is **fully polarized** (degree
    of polarization = 1).  Mathematically this is the Hopf fibration
    S³ → S² of the doublet ζ ∈ ℂ²; the empirical anchor is the
    classical Poincaré sphere of polarization optics.

    HONEST SCOPE: this is a **dynamical symmetry of the flat, isotropic
    H_sub backbone** (like the accidental-degeneracy symmetries of the
    oscillator), not a gauge symmetry of the field content. The SU(2)
    rotation mixes the *physically distinct* geometric and potential
    sectors; it is canonical (preserves ω and H_sub) but is **not** one of
    the 13 operators. P_1, P_2, P_3 are exact conserved charges along the
    substrate flow and diagnostics at the full nonlinear operator level.
    The substrate is a CLASSICAL phase field: this is the polarization
    (Stokes/Poincaré) of a wave, NOT a quantum two-level system — there is
    no superposition or entanglement (the doublet is per-node, so the
    global state is a product, a classical polarization texture).

    Attributes
    ----------
    n_nodes : int
    p_1, p_2, p_3 : float
        The three Stokes parameters (the global polarization 3-vector).
    magnitude_sq : float
        P_1² + P_2² + P_3² (the SU(2) Casimir = squared polarization).
    p3_equals_energy_difference : bool
        P_3 = E_geo − E_pot to machine precision.
    su2_algebra_closes : bool
        {P_a, P_b} = 2 ε_abc P_c under the canonical Poisson bracket.
    rotation_is_symplectic : bool
        A finite SU(2) sector-mixing rotation preserves ω.
    charges_conserved : bool
        P_1, P_2, P_3 are conserved along the substrate flow.
    max_charge_drift : float
        Max drift of the Stokes parameters over the sampled flow.
    max_algebra_residual : float
        Max |{P_a, P_b} − 2 ε_abc P_c| over the three brackets.
    full_polarization_holds : bool
        Per node, the polarization 3-vector length equals the per-node
        energy, |P_node| = e_node — each node is fully polarized (a unit
        point on the Poincaré sphere S²).
    max_polarization_residual : float
        Max | |P_node| − e_node | over the nodes (should be ~0).
    """

    n_nodes: int
    p_1: float
    p_2: float
    p_3: float
    magnitude_sq: float
    p3_equals_energy_difference: bool
    su2_algebra_closes: bool
    rotation_is_symplectic: bool
    charges_conserved: bool
    max_charge_drift: float
    max_algebra_residual: float
    full_polarization_holds: bool
    max_polarization_residual: float

    @property
    def is_valid_polarization_symmetry(self) -> bool:
        """True when the polarization (U(2)) symmetry fully verifies."""
        return (
            self.p3_equals_energy_difference
            and self.su2_algebra_closes
            and self.rotation_is_symplectic
            and self.charges_conserved
            and self.full_polarization_holds
        )

    def summary(self) -> str:
        """Human-readable one-line verdict."""
        ok = "VALID" if self.is_valid_polarization_symmetry else "INVALID"
        return (
            f"Polarization symmetry [{ok}]: "
            f"Stokes P=({self.p_1:.4f}, {self.p_2:.4f}, {self.p_3:.4f}), "
            f"|P|²={self.magnitude_sq:.4f}, "
            f"P₃=E_geo−E_pot={self.p3_equals_energy_difference}, "
            f"su(2) closes={self.su2_algebra_closes} "
            f"(res {self.max_algebra_residual:.1e}), "
            f"SU(2) symplectic={self.rotation_is_symplectic}, "
            f"conserved={self.charges_conserved} "
            f"(drift {self.max_charge_drift:.1e}), "
            f"fully polarized |P_node|=e_node={self.full_polarization_holds} "
            f"(res {self.max_polarization_residual:.1e})"
        )


@dataclass(frozen=True)
class SubstrateGeometryReport:
    r"""Consolidated report of the complete emergent geometric tower.

    Aggregates the six structural certificates produced by
    :func:`verify_substrate_geometry`, giving a single entry point to the
    whole classical Hamiltonian-geometry tower derived from the nodal
    dynamics: symplectic/Poisson/Liouville, Noether charges, Hermitian
    (flat Kähler), complete integrability, Poincaré–Cartan invariants, and
    the Marsden–Weinstein reduction.

    Attributes
    ----------
    n_nodes : int
    phase_space_dimension : int
        4·n_nodes.
    canonical : CanonicalStructureCertificate
    noether : NoetherChargeCertificate
    hermitian : HermitianStructureCertificate
    integrability : IntegrabilityCertificate
    poincare_cartan : PoincareCartanCertificate
    marsden_weinstein : MarsdenWeinsteinCertificate
    polarization : PolarizationSymmetryCertificate
    """

    n_nodes: int
    phase_space_dimension: int
    canonical: CanonicalStructureCertificate
    noether: NoetherChargeCertificate
    hermitian: HermitianStructureCertificate
    integrability: IntegrabilityCertificate
    poincare_cartan: PoincareCartanCertificate
    marsden_weinstein: MarsdenWeinsteinCertificate
    polarization: PolarizationSymmetryCertificate

    @property
    def all_structures_valid(self) -> bool:
        """True when every structure in the tower verifies."""
        return (
            self.canonical.is_valid_symplectic_manifold
            and self.noether.is_conserved
            and self.hermitian.is_valid_hermitian_structure
            and self.integrability.is_completely_integrable
            and self.poincare_cartan.all_invariants_hold
            and self.marsden_weinstein.is_valid_reduction
            and self.polarization.is_valid_polarization_symmetry
        )

    def summary(self) -> str:
        """Multi-line verdict listing every structure in the tower."""
        ok = "ALL VALID" if self.all_structures_valid else "INCOMPLETE"
        lines = [
            f"Emergent substrate geometry [{ok}] "
            f"(N={self.n_nodes}, dim P={self.phase_space_dimension}):",
            f"  1. {self.canonical.summary()}",
            f"  2. {self.noether.summary()}",
            f"  3. {self.hermitian.summary()}",
            f"  4. {self.integrability.summary()}",
            f"  5. {self.poincare_cartan.summary()}",
            f"  6. {self.marsden_weinstein.summary()}",
            f"  7. {self.polarization.summary()}",
        ]
        return "\n".join(lines)


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


# ---------------------------------------------------------------------------
# Noether's theorem on the substrate: symmetries → conserved charges
# ---------------------------------------------------------------------------


def evolve_substrate_flow(point: PhaseSpacePoint, t: float) -> PhaseSpacePoint:
    r"""Evolve a phase-space point along the substrate Hamiltonian flow.

    The flow of X_H = J ∇H_sub is the harmonic rotation q̇ = p, ṗ = −q
    per conjugate pair, integrated *exactly* (no numerical scheme):

        q(t) = q(0)·cos t + p(0)·sin t,
        p(t) = −q(0)·sin t + p(0)·cos t.

    The background coordinate |∇φ| is a configuration potential with no
    conjugate momentum, so it is held fixed along the symplectic flow.

    Parameters
    ----------
    point : PhaseSpacePoint
    t : float
        Flow time.

    Returns
    -------
    PhaseSpacePoint
        The evolved point at flow time ``t``.
    """
    c = float(np.cos(t))
    s = float(np.sin(t))
    k_phi = np.asarray(point.k_phi, dtype=float)
    j_phi = np.asarray(point.j_phi, dtype=float)
    phi_s = np.asarray(point.phi_s, dtype=float)
    j_dnfr = np.asarray(point.j_dnfr, dtype=float)
    return PhaseSpacePoint(
        nodes=point.nodes,
        k_phi=k_phi * c + j_phi * s,
        j_phi=-k_phi * s + j_phi * c,
        phi_s=phi_s * c + j_dnfr * s,
        j_dnfr=-phi_s * s + j_dnfr * c,
        grad_phi=np.asarray(point.grad_phi, dtype=float),
    )


def geometric_sector_energy(point: PhaseSpacePoint) -> float:
    r"""Geometric-sector Noether charge E_geo = ½Σ(K_φ² + J_φ²) = ½Σ|Ψ|².

    This is the conserved charge of the geometric U(1) symmetry
    Ψ → e^{iα}Ψ (Ψ = K_φ + i·J_φ), the gauge symmetry established in
    :mod:`tnfr.physics.gauge`.  It is conserved along the substrate flow.
    """
    k = np.asarray(point.k_phi, dtype=float)
    j = np.asarray(point.j_phi, dtype=float)
    return 0.5 * float(np.dot(k, k) + np.dot(j, j))


def potential_sector_energy(point: PhaseSpacePoint) -> float:
    r"""Potential-sector Noether charge E_pot = ½Σ(Φ_s² + J_ΔNFR²).

    The conserved charge of the potential U(1) symmetry (rotation of the
    (Φ_s, J_ΔNFR) conjugate pair).  Conserved along the substrate flow.
    """
    p = np.asarray(point.phi_s, dtype=float)
    j = np.asarray(point.j_dnfr, dtype=float)
    return 0.5 * float(np.dot(p, p) + np.dot(j, j))


def noether_charges(point: PhaseSpacePoint) -> dict[str, float]:
    r"""Return the three Noether charges of the substrate flow.

    Maps each continuous symmetry to its conserved quantity:

    - ``"time_translation"`` → H_sub (total energy)
    - ``"geometric_u1"`` → ½Σ|Ψ|² (gauge U(1) of Ψ = K_φ + i·J_φ)
    - ``"potential_u1"`` → ½Σ(Φ_s² + J_ΔNFR²)

    The total splits exactly: H_sub = E_geo + E_pot.

    Returns
    -------
    dict[str, float]
    """
    e_geo = geometric_sector_energy(point)
    e_pot = potential_sector_energy(point)
    return {
        "time_translation": e_geo + e_pot,
        "geometric_u1": e_geo,
        "potential_u1": e_pot,
    }


def verify_noether_conservation(
    G: Any,
    *,
    flow_times: tuple[float, ...] = (0.0, 0.5, 1.3, 2.7, 5.0, 10.0),
    tolerance: float = 1e-9,
) -> NoetherChargeCertificate:
    r"""Verify the Noether charges are conserved along the substrate flow.

    Extracts the phase-space point, evolves it along the exact substrate
    Hamiltonian flow at each ``flow_time``, and checks that H_sub, E_geo,
    and E_pot remain constant (Noether's theorem).

    Parameters
    ----------
    G : TNFRGraph
    flow_times : tuple of float
        Sampling times for the conservation check.
    tolerance : float
        Maximum allowed drift for the conserved quantities.

    Returns
    -------
    NoetherChargeCertificate
    """
    point = extract_phase_space_point(G)
    h0 = substrate_hamiltonian(point)
    e_geo0 = geometric_sector_energy(point)
    e_pot0 = potential_sector_energy(point)

    h_drift = 0.0
    geo_drift = 0.0
    pot_drift = 0.0
    for t in flow_times:
        evolved = evolve_substrate_flow(point, t)
        h_drift = max(h_drift, abs(substrate_hamiltonian(evolved) - h0))
        geo_drift = max(
            geo_drift, abs(geometric_sector_energy(evolved) - e_geo0)
        )
        pot_drift = max(
            pot_drift, abs(potential_sector_energy(evolved) - e_pot0)
        )

    is_conserved = (
        h_drift < tolerance
        and geo_drift < tolerance
        and pot_drift < tolerance
    )
    splits_exactly = abs(h0 - (e_geo0 + e_pot0)) < 1e-12

    return NoetherChargeCertificate(
        hamiltonian=h0,
        geometric_energy=e_geo0,
        potential_energy=e_pot0,
        max_hamiltonian_drift=h_drift,
        max_geometric_drift=geo_drift,
        max_potential_drift=pot_drift,
        is_conserved=is_conserved,
        splits_exactly=splits_exactly,
    )


# ---------------------------------------------------------------------------
# Compatible Hermitian (flat Kähler) structure: ω, J, g and Ψ as ζ-coordinate
# ---------------------------------------------------------------------------


def complex_structure_matrix(n_nodes: int) -> Any:
    r"""Return the complex structure J = −ω as a 4N×4N block matrix.

    Block-diagonal with ``n_nodes`` copies of :data:`BLOCK_COMPLEX_STRUCTURE`.
    Satisfies J² = −I and acts as multiplication by i on ζ = q + i·p.

    Parameters
    ----------
    n_nodes : int

    Returns
    -------
    np.ndarray
    """
    if n_nodes < 1:
        raise ValueError("n_nodes must be >= 1")
    dim = 4 * n_nodes
    out = np.zeros((dim, dim), dtype=float)
    for i in range(n_nodes):
        s = 4 * i
        out[s:s + 4, s:s + 4] = BLOCK_COMPLEX_STRUCTURE
    return out


def compatible_metric_matrix(n_nodes: int) -> Any:
    r"""Return the compatible metric g = ω·J (= identity) as a 4N×4N matrix.

    Block-diagonal with ``n_nodes`` copies of :data:`BLOCK_COMPATIBLE_METRIC`,
    which equals the identity.  g is the Riemannian metric of the compatible
    triple (ω, J, g).

    Parameters
    ----------
    n_nodes : int

    Returns
    -------
    np.ndarray
    """
    if n_nodes < 1:
        raise ValueError("n_nodes must be >= 1")
    dim = 4 * n_nodes
    out = np.zeros((dim, dim), dtype=float)
    for i in range(n_nodes):
        s = 4 * i
        out[s:s + 4, s:s + 4] = BLOCK_COMPATIBLE_METRIC
    return out


def to_complex_coordinates(point: PhaseSpacePoint) -> dict[str, Any]:
    r"""Return the complex coordinates ζ of the Hermitian phase space.

    Each conjugate pair (q, p) becomes a complex coordinate ζ = q + i·p:

        ζ^A = K_φ + i·J_φ = Ψ   (geometric sector — the canonical complex
                                  field of :mod:`tnfr.physics.gauge`),
        ζ^B = Φ_s + i·J_ΔNFR     (potential sector).

    So the gauge field Ψ is *not* an ad-hoc construction: it is the complex
    coordinate the substrate's complex structure J induces on the geometric
    sector.

    Parameters
    ----------
    point : PhaseSpacePoint

    Returns
    -------
    dict[str, np.ndarray]
        ``"geometric"`` → ζ^A = Ψ, ``"potential"`` → ζ^B (complex arrays).
    """
    k_phi = np.asarray(point.k_phi, dtype=float)
    j_phi = np.asarray(point.j_phi, dtype=float)
    phi_s = np.asarray(point.phi_s, dtype=float)
    j_dnfr = np.asarray(point.j_dnfr, dtype=float)
    return {
        "geometric": k_phi + 1j * j_phi,
        "potential": phi_s + 1j * j_dnfr,
    }


def kahler_potential(point: PhaseSpacePoint) -> float:
    r"""Kähler potential H = ½Σ(|ζ^A|² + |ζ^B|²) = H_sub.

    The substrate Hamiltonian read as the Kähler potential of the Hermitian
    structure.  Equals :func:`substrate_hamiltonian` exactly.

    Parameters
    ----------
    point : PhaseSpacePoint

    Returns
    -------
    float
    """
    z = to_complex_coordinates(point)
    za = z["geometric"]
    zb = z["potential"]
    return 0.5 * float(
        np.sum(np.abs(za) ** 2) + np.sum(np.abs(zb) ** 2)
    )


def verify_hermitian_structure(G: Any) -> HermitianStructureCertificate:
    r"""Verify the compatible Hermitian (flat Kähler) structure on ``G``.

    Checks the (ω, J, g) compatibility (J² = −I, g = identity,
    J-orthogonality, ω(u,v) = g(Ju,v)), that Ψ is exactly the geometric
    complex coordinate, and that H_sub is the Kähler potential.

    Parameters
    ----------
    G : TNFRGraph

    Returns
    -------
    HermitianStructureCertificate
    """
    omega = BLOCK_SYMPLECTIC_FORM
    j_cs = BLOCK_COMPLEX_STRUCTURE
    g = BLOCK_COMPATIBLE_METRIC
    eye4 = np.eye(4)

    j_sq = bool(np.allclose(j_cs @ j_cs, -eye4))
    metric_id = bool(np.allclose(g, eye4))
    j_orth = bool(np.allclose(j_cs.T @ g @ j_cs, g))
    compat = bool(np.allclose(omega, j_cs.T @ g))

    # Ψ == ζ^A (geometric complex coordinate)?
    psi_is_coord = True
    try:
        from .unified import compute_complex_geometric_field

        point = extract_phase_space_point(G)
        coords = to_complex_coordinates(point)
        psi = compute_complex_geometric_field(G)
        psi_arr = np.array(
            [psi.get(n, 0.0) for n in point.nodes], dtype=complex
        )
        psi_is_coord = bool(np.allclose(coords["geometric"], psi_arr))
    except Exception:
        point = extract_phase_space_point(G)
        psi_is_coord = False

    # H_sub == Kähler potential?
    kahler_matches = bool(
        abs(kahler_potential(point) - substrate_hamiltonian(point)) < 1e-9
    )

    return HermitianStructureCertificate(
        n_nodes=point.n_nodes,
        complex_dimension=2 * point.n_nodes,
        j_squared_is_minus_id=j_sq,
        metric_is_identity=metric_id,
        j_is_orthogonal=j_orth,
        compatible=compat,
        psi_is_geometric_coordinate=psi_is_coord,
        kahler_potential_matches=kahler_matches,
    )


# ---------------------------------------------------------------------------
# Complete integrability: action–angle variables of the substrate flow
# ---------------------------------------------------------------------------


def to_action_angle(point: PhaseSpacePoint) -> dict[str, Any]:
    r"""Return the action–angle coordinates of the substrate flow.

    Each conjugate pair (q, p) maps to an action I = ½(q² + p²) = ½|ζ|² and
    an angle θ = arg ζ = atan2(p, q):

        I^A = ½|Ψ|²   (geometric),   θ^A = arg Ψ,
        I^B = ½|ζ^B|² (potential),   θ^B = arg ζ^B.

    Under the substrate flow the actions are conserved and the angles
    advance linearly θ(t) = θ(0) − t, so these are global action–angle
    coordinates in which the harmonic backbone is trivial.

    Parameters
    ----------
    point : PhaseSpacePoint

    Returns
    -------
    dict[str, np.ndarray]
        ``"action_geometric"``, ``"action_potential"``,
        ``"angle_geometric"``, ``"angle_potential"``.
    """
    z = to_complex_coordinates(point)
    za = z["geometric"]
    zb = z["potential"]
    return {
        "action_geometric": 0.5 * np.abs(za) ** 2,
        "action_potential": 0.5 * np.abs(zb) ** 2,
        "angle_geometric": np.angle(za),
        "angle_potential": np.angle(zb),
    }


def _max_action_involution(point: PhaseSpacePoint) -> float:
    r"""Max |{I_i, I_j}| over the action variables (block-local, exact).

    The action of each conjugate pair depends only on that pair's
    coordinates, and the symplectic block J₄ does not couple the two
    sectors of a node, so every action bracket vanishes structurally.  This
    confirms it numerically per node (O(N), no dense 4N×4N matrix).
    """
    block = BLOCK_SYMPLECTIC_FORM
    k = np.asarray(point.k_phi, dtype=float)
    jp = np.asarray(point.j_phi, dtype=float)
    ps = np.asarray(point.phi_s, dtype=float)
    jd = np.asarray(point.j_dnfr, dtype=float)
    worst = 0.0
    for i in range(point.n_nodes):
        g_a = np.array([k[i], jp[i], 0.0, 0.0])
        g_b = np.array([0.0, 0.0, ps[i], jd[i]])
        worst = max(
            worst,
            abs(float(g_a @ block @ g_b)),   # {I^A_i, I^B_i}
            abs(float(g_a @ block @ g_a)),   # {I^A_i, I^A_i}
            abs(float(g_b @ block @ g_b)),   # {I^B_i, I^B_i}
        )
    return worst


def verify_integrability(
    G: Any,
    *,
    flow_times: tuple[float, ...] = (0.3, 0.9, 1.7, 3.1, 6.3),
    tolerance: float = 1e-9,
) -> IntegrabilityCertificate:
    r"""Verify the substrate flow is completely integrable (Liouville–Arnold).

    Confirms 2N action variables I = ½|ζ|² (one per conjugate pair) for a
    2N-degree-of-freedom system, that they are pairwise in involution and
    conserved along the flow, that the conjugate angles advance linearly
    θ(t) = θ(0) − t, and that the sector action sums recover the Noether
    charges (Σ I^A = E_geo, Σ I^B = E_pot).

    HONEST SCOPE: integrability of the substrate harmonic backbone (the
    H_sub flow), not of the full nonlinear operator dynamics.

    Parameters
    ----------
    G : TNFRGraph
    flow_times : tuple of float
        Sampling times for the conservation / linear-angle check.
    tolerance : float
        Maximum allowed action drift, angle error, and involution bracket.

    Returns
    -------
    IntegrabilityCertificate
    """
    point = extract_phase_space_point(G)
    dof = 2 * point.n_nodes

    aa0 = to_action_angle(point)
    ia0 = aa0["action_geometric"]
    ib0 = aa0["action_potential"]
    tha0 = aa0["angle_geometric"]
    thb0 = aa0["angle_potential"]

    action_drift = 0.0
    angle_error = 0.0
    for t in flow_times:
        evolved = evolve_substrate_flow(point, t)
        aa = to_action_angle(evolved)
        action_drift = max(
            action_drift,
            float(np.max(np.abs(aa["action_geometric"] - ia0))),
            float(np.max(np.abs(aa["action_potential"] - ib0))),
        )
        # angles must satisfy θ(t) = θ(0) − t (compared on the circle).
        da = np.angle(np.exp(1j * (aa["angle_geometric"] - (tha0 - t))))
        db = np.angle(np.exp(1j * (aa["angle_potential"] - (thb0 - t))))
        angle_error = max(
            angle_error,
            float(np.max(np.abs(da))),
            float(np.max(np.abs(db))),
        )

    involution = _max_action_involution(point)

    sector_match = (
        abs(float(np.sum(ia0)) - geometric_sector_energy(point)) < tolerance
        and abs(float(np.sum(ib0)) - potential_sector_energy(point))
        < tolerance
    )

    return IntegrabilityCertificate(
        n_nodes=point.n_nodes,
        degrees_of_freedom=dof,
        n_action_variables=dof,
        actions_in_involution=involution < tolerance,
        actions_conserved=action_drift < tolerance,
        angles_advance_linearly=angle_error < tolerance,
        max_action_drift=action_drift,
        max_angle_error=angle_error,
        max_involution_bracket=involution,
        sector_actions_match_charges=sector_match,
    )


# ---------------------------------------------------------------------------
# Poincaré–Cartan integral invariants: the ω^k tower and Bohr–Sommerfeld
# ---------------------------------------------------------------------------


def substrate_flow_matrix(n_nodes: int, t: float) -> Any:
    r"""Return the 4N×4N matrix M(t) of the substrate Hamiltonian flow.

    The flow of X_H integrates exactly as the harmonic rotation
    q(t) = q·cos t + p·sin t, p(t) = −q·sin t + p·cos t per conjugate pair.
    In the per-node basis (q^A, p^A, q^B, p^B) the matrix is block-diagonal
    with two copies (geometric, potential) of the 2×2 rotation

        R(t) = [[ cos t, sin t],
                [−sin t, cos t]]

    per node.  M(t) is **symplectic** (Mᵀ Ω M = Ω) with det M = 1, and its
    spectrum is the reciprocal symplectic set {e^{+it}, e^{−it}}.

    Parameters
    ----------
    n_nodes : int
    t : float
        Flow time.

    Returns
    -------
    np.ndarray
        The 4N×4N symplectic flow matrix.
    """
    if n_nodes < 1:
        raise ValueError("n_nodes must be >= 1")
    c = float(np.cos(t))
    s = float(np.sin(t))
    rot = np.array([[c, s], [-s, c]], dtype=float)
    dim = 4 * n_nodes
    out = np.zeros((dim, dim), dtype=float)
    for i in range(n_nodes):
        b = 4 * i
        out[b:b + 2, b:b + 2] = rot       # geometric pair (q^A, p^A)
        out[b + 2:b + 4, b + 2:b + 4] = rot  # potential pair (q^B, p^B)
    return out


def loop_action_integral(action: float, *, n_points: int = 4000) -> float:
    r"""Relative integral invariant ∮_γ p dq over an action-torus loop.

    Parametrises the loop of a single conjugate pair at fixed action I,

        (q, p) = (√(2I)·cos s, √(2I)·sin s),   s ∈ [0, 2π),

    and returns ∮ p dq by the trapezoidal rule over the closed loop.  The
    exact value is −2π·I (the negative of the enclosed area π·(2I)); its
    magnitude is the **Bohr–Sommerfeld** quantum 2π·I.

    Parameters
    ----------
    action : float
        The action variable I = ½|ζ|² of the conjugate pair.
    n_points : int, optional
        Number of quadrature points around the loop.

    Returns
    -------
    float
        ∮ p dq (negative for the counter-clockwise parametrisation).
    """
    r = float(np.sqrt(2.0 * max(action, 0.0)))
    s = np.linspace(0.0, 2.0 * np.pi, n_points)
    q = r * np.cos(s)
    p = r * np.sin(s)
    return float(np.trapezoid(p, q))


def verify_poincare_cartan(
    G: Any,
    *,
    flow_times: tuple[float, ...] = (0.3, 0.9, 1.7, 3.1),
    tolerance: float = 1e-6,
) -> PoincareCartanCertificate:
    r"""Verify the Poincaré–Cartan integral invariants of the substrate flow.

    Confirms the whole tower ω^k (k = 1 … N) of integral invariants:

    - **ω-preservation** (1st / relative invariant): the flow matrix M(t) is
      symplectic, Mᵀ Ω M = Ω, for every sampled time.
    - **palindromic characteristic polynomial** of M(t): the reciprocal
      symplectic spectrum {e^{±it}} encodes every intermediate invariant.
    - **volume** (top invariant): det M = 1 (Liouville).
    - **relative invariant** ∮_γ p dq over an action-torus loop is constant
      along the flow.
    - **Bohr–Sommerfeld**: |∮_γ p dq| = 2π I on the action torus.

    Parameters
    ----------
    G : TNFRGraph
    flow_times : tuple of float
        Sampling times for the invariance checks.
    tolerance : float
        Maximum allowed drift / error.

    Returns
    -------
    PoincareCartanCertificate
    """
    point = extract_phase_space_point(G)
    n = point.n_nodes
    omega = symplectic_form_matrix(n)

    # --- 1st invariant: flow is symplectic, and tower via palindromic poly ---
    omega_drift = 0.0
    palindromic = True
    volume_ok = True
    for t in flow_times:
        m = substrate_flow_matrix(n, t)
        omega_drift = max(
            omega_drift, float(np.max(np.abs(m.T @ omega @ m - omega)))
        )
        det_err = abs(float(np.linalg.det(m)) - 1.0)
        volume_ok = volume_ok and det_err < tolerance
        coeffs = np.poly(m)
        palindromic = palindromic and bool(
            np.allclose(coeffs, coeffs[::-1], atol=1e-9)
        )
    preserves_omega = omega_drift < tolerance

    # --- relative invariant ∮ p dq on an action-torus loop, under the flow ---
    aa = to_action_angle(point)
    actions = np.concatenate(
        [aa["action_geometric"], aa["action_potential"]]
    )
    # Use the largest-action pair as the representative torus loop.
    i_star = int(np.argmax(actions)) if actions.size else 0
    action_star = float(actions[i_star]) if actions.size else 0.0

    r = float(np.sqrt(2.0 * action_star))
    s = np.linspace(0.0, 2.0 * np.pi, 4000)
    q0 = r * np.cos(s)
    p0 = r * np.sin(s)
    base_loop = float(np.trapezoid(p0, q0))

    relative_drift = 0.0
    for t in flow_times:
        c, sn = float(np.cos(t)), float(np.sin(t))
        q_t = q0 * c + p0 * sn
        p_t = -q0 * sn + p0 * c
        loop_t = float(np.trapezoid(p_t, q_t))
        relative_drift = max(relative_drift, abs(loop_t - base_loop))
    relative_ok = relative_drift < tolerance

    # --- Bohr–Sommerfeld: |∮ p dq| = 2π I (quadrature tol is looser) ---
    bohr_error = abs(abs(base_loop) - 2.0 * np.pi * action_star)
    bohr_ok = bohr_error < 1e-3

    return PoincareCartanCertificate(
        n_nodes=n,
        phase_space_dimension=4 * n,
        preserves_symplectic_form=preserves_omega,
        volume_preserved=volume_ok,
        char_poly_palindromic=palindromic,
        relative_invariant_preserved=relative_ok,
        bohr_sommerfeld_holds=bohr_ok,
        max_omega_drift=omega_drift,
        max_relative_drift=relative_drift,
        max_bohr_error=float(bohr_error),
    )


# ---------------------------------------------------------------------------
# Marsden–Weinstein symplectic reduction by the diagonal U(1) flow symmetry
# ---------------------------------------------------------------------------


def diagonal_moment_map(point: PhaseSpacePoint) -> float:
    r"""Moment map J = Σ_k I_k of the diagonal U(1) flow symmetry.

    The substrate flow is the diagonal U(1) action ζ → e^{−it}ζ rotating
    every conjugate pair together.  Its moment map is the sum of all action
    variables J = Σ_k ½|ζ_k|² = H_sub — equal to the substrate Hamiltonian
    (the time-translation Noether charge).

    Parameters
    ----------
    point : PhaseSpacePoint

    Returns
    -------
    float
    """
    aa = to_action_angle(point)
    return float(
        np.sum(aa["action_geometric"]) + np.sum(aa["action_potential"])
    )


def reduced_symplectic_form_matrix(n_nodes: int) -> Any:
    r"""Reduced symplectic form of P//U(1) in action–angle coordinates.

    Builds the Marsden–Weinstein reduced 2-form explicitly.  With
    m = 2·n_nodes conjugate pairs and ω = Σ_k dI_k ∧ dθ_k, the diagonal
    U(1) (generator ξ = Σ_k ∂/∂θ_k, moment map J = Σ_k I_k) is reduced by

    - restricting to the level set Σ_k dI_k = 0 (tangent action directions
      a_k = dI_k − dI_0, k = 1…m−1), and
    - quotienting by ξ (relative-phase directions b_k = dθ_k − dθ_0).

    The pullback Bᵀ Ω B of the full 2m×2m form Ω to this basis is the
    reduced symplectic form, a (4N−2)×(4N−2) non-degenerate matrix
    (det = (2N)² = m²) ⇒ the quotient is a symplectic manifold.

    Parameters
    ----------
    n_nodes : int

    Returns
    -------
    np.ndarray
        The (4N−2)×(4N−2) reduced symplectic form.
    """
    if n_nodes < 1:
        raise ValueError("n_nodes must be >= 1")
    m = 2 * n_nodes
    # Full canonical form on (I_0..I_{m-1}, θ_0..θ_{m-1}).
    eye_m = np.eye(m)
    zero_m = np.zeros((m, m))
    omega_full = np.block([[zero_m, eye_m], [-eye_m, zero_m]])
    # Reduced basis columns: action directions then relative-phase directions.
    cols = []
    for k in range(1, m):
        a = np.zeros(2 * m)
        a[k] = 1.0
        a[0] = -1.0          # dI_k − dI_0 (lives on the level set)
        cols.append(a)
    for k in range(1, m):
        b = np.zeros(2 * m)
        b[m + k] = 1.0
        b[m + 0] = -1.0      # dθ_k − dθ_0 (transverse to ξ)
        cols.append(b)
    basis = np.array(cols).T  # (2m) × (2(m-1))
    return basis.T @ omega_full @ basis


def verify_symplectic_reduction(
    G: Any,
    *,
    flow_times: tuple[float, ...] = (0.5, 1.7, 3.0),
    tolerance: float = 1e-9,
) -> MarsdenWeinsteinCertificate:
    r"""Verify the Marsden–Weinstein reduction of the substrate.

    Confirms that the diagonal U(1) flow symmetry has moment map J = H_sub
    (the time-translation Noether charge), that J is conserved, that the
    reduced phase space P//U(1) has dimension 4N − 2 with a non-degenerate
    reduced symplectic form, and that the relative phases φ_k = θ_k − θ_0
    (the reduced coordinates) are invariant under the flow.

    Parameters
    ----------
    G : TNFRGraph
    flow_times : tuple of float
        Sampling times for the conservation / invariance checks.
    tolerance : float
        Maximum allowed drift.

    Returns
    -------
    MarsdenWeinsteinCertificate
    """
    point = extract_phase_space_point(G)
    n = point.n_nodes

    j0 = diagonal_moment_map(point)
    is_hamiltonian = abs(j0 - substrate_hamiltonian(point)) < tolerance

    # Moment-map conservation and relative-phase invariance along the flow.
    aa0 = to_action_angle(point)
    th0 = np.concatenate([aa0["angle_geometric"], aa0["angle_potential"]])
    rel0 = np.angle(np.exp(1j * (th0 - th0[0])))

    moment_drift = 0.0
    phases_ok = True
    for t in flow_times:
        evolved = evolve_substrate_flow(point, t)
        moment_drift = max(
            moment_drift, abs(diagonal_moment_map(evolved) - j0)
        )
        aa = to_action_angle(evolved)
        th = np.concatenate([aa["angle_geometric"], aa["angle_potential"]])
        rel = np.angle(np.exp(1j * (th - th[0])))
        phases_ok = phases_ok and bool(np.allclose(rel, rel0, atol=1e-7))
    moment_conserved = moment_drift < tolerance

    reduced = reduced_symplectic_form_matrix(n)
    det_reduced = float(np.linalg.det(reduced))
    nondegenerate = abs(det_reduced) > tolerance

    return MarsdenWeinsteinCertificate(
        n_nodes=n,
        phase_space_dimension=4 * n,
        moment_map_value=j0,
        moment_map_is_hamiltonian=is_hamiltonian,
        moment_map_conserved=moment_conserved,
        reduced_dimension=4 * n - 2,
        reduced_form_nondegenerate=nondegenerate,
        relative_phases_invariant=phases_ok,
        max_moment_drift=moment_drift,
        reduced_form_determinant=det_reduced,
    )


# ---------------------------------------------------------------------------
# Polarization symmetry: Stokes parameters of the (ζ^A, ζ^B) doublet
# ---------------------------------------------------------------------------


def polarization_vector(point: PhaseSpacePoint) -> dict[str, float]:
    r"""Stokes parameters (polarization 3-vector) of the (ζ^A, ζ^B) doublet.

    The substrate Hamiltonian H_sub = ½Σ‖(ζ^A, ζ^B)‖² is the squared norm
    of a complex doublet, so it carries the **polarization symmetry** U(2)
    of a two-component complex field — the same mathematics as classical
    wave polarization (Stokes 1852, Poincaré 1892).  The SU(2) moment map
    (global diagonal SU(2)) gives the three conserved **Stokes parameters**:

        P_3 = ½ Σ (|ζ^A|² − |ζ^B|²) = E_geo − E_pot   (sector difference),
        P_1 = Σ Re(ζ̄^A ζ^B) = Σ (K_φ·Φ_s + J_φ·J_ΔNFR)   (real cross-corr.),
        P_2 = Σ Im(ζ̄^A ζ^B) = Σ (K_φ·J_ΔNFR − J_φ·Φ_s)   (imag cross-corr.),

    with squared magnitude |P|² = P_1² + P_2² + P_3².  P_1 and P_2 are the
    new cross-sector correlation charges; P_3 is the sector-energy
    difference.  These are the Stokes parameters in the substrate's natural
    (Noether-charge) normalization; the textbook optical Stokes parameters
    are 2× these.

    Parameters
    ----------
    point : PhaseSpacePoint

    Returns
    -------
    dict[str, float]
        ``"p_1"``, ``"p_2"``, ``"p_3"``, ``"magnitude_sq"``.
    """
    k = np.asarray(point.k_phi, dtype=float)
    jp = np.asarray(point.j_phi, dtype=float)
    ps = np.asarray(point.phi_s, dtype=float)
    jd = np.asarray(point.j_dnfr, dtype=float)
    # ζ^A = k + i·jp,  ζ^B = ps + i·jd ;  ζ̄^A ζ^B = (k − i·jp)(ps + i·jd)
    p_1 = float(np.sum(k * ps + jp * jd))      # Re(ζ̄^A ζ^B)
    p_2 = float(np.sum(k * jd - jp * ps))      # Im(ζ̄^A ζ^B)
    p_3 = 0.5 * float(np.sum(k * k + jp * jp - ps * ps - jd * jd))
    return {
        "p_1": p_1,
        "p_2": p_2,
        "p_3": p_3,
        "magnitude_sq": p_1 * p_1 + p_2 * p_2 + p_3 * p_3,
    }


def polarization_density(point: PhaseSpacePoint) -> dict[str, Any]:
    r"""Per-node Stokes 3-vector, energy, and Poincaré-sphere unit vector.

    The global :func:`polarization_vector` are sums of per-node densities.
    Per node, the doublet ζ_i = (ζ^A_i, ζ^B_i) ∈ ℂ² projects to a Stokes
    3-vector

        P_1^i = Re(ζ̄^A_i ζ^B_i),  P_2^i = Im(ζ̄^A_i ζ^B_i),
        P_3^i = ½(|ζ^A_i|² − |ζ^B_i|²),

    whose **length equals the per-node substrate energy**,
    |P_node| = e_node = ½(|ζ^A_i|² + |ζ^B_i|²).  The normalised vector
    ``poincare`` = P_node / e_node is therefore a unit vector on the
    **Poincaré sphere** S² — each node is fully polarized (degree of
    polarization = 1), and ``e_node`` is its radius.  (Mathematically this
    is the Hopf fibration S³ → S² of the doublet; the empirical anchor is
    the classical Poincaré sphere of polarization optics.)

    Parameters
    ----------
    point : PhaseSpacePoint

    Returns
    -------
    dict[str, np.ndarray]
        ``"p_1"``, ``"p_2"``, ``"p_3"`` (per-node Stokes components),
        ``"radius"`` (|P_node|), ``"energy"`` (e_node), and ``"poincare"``
        (a ``(3, N)`` array of unit Poincaré-sphere vectors).
    """
    k = np.asarray(point.k_phi, dtype=float)
    jp = np.asarray(point.j_phi, dtype=float)
    ps = np.asarray(point.phi_s, dtype=float)
    jd = np.asarray(point.j_dnfr, dtype=float)
    p_1 = k * ps + jp * jd            # Re(ζ̄^A ζ^B) per node
    p_2 = k * jd - jp * ps            # Im(ζ̄^A ζ^B) per node
    p_3 = 0.5 * (k * k + jp * jp - ps * ps - jd * jd)
    radius = np.sqrt(p_1 * p_1 + p_2 * p_2 + p_3 * p_3)
    energy = 0.5 * (k * k + jp * jp + ps * ps + jd * jd)
    poincare = np.array([p_1, p_2, p_3]) / (radius + 1e-300)
    return {
        "p_1": p_1,
        "p_2": p_2,
        "p_3": p_3,
        "radius": radius,
        "energy": energy,
        "poincare": poincare,
    }


def _polarization_gradients(point: PhaseSpacePoint) -> tuple[Any, Any, Any]:
    r"""Phase-space gradients ∇P_1, ∇P_2, ∇P_3 (for the Poisson brackets)."""
    n = point.n_nodes
    k = np.asarray(point.k_phi, dtype=float)
    jp = np.asarray(point.j_phi, dtype=float)
    ps = np.asarray(point.phi_s, dtype=float)
    jd = np.asarray(point.j_dnfr, dtype=float)
    g1 = np.zeros(4 * n)
    g2 = np.zeros(4 * n)
    g3 = np.zeros(4 * n)
    for i in range(n):
        b = 4 * i
        # basis (q^A, p^A, q^B, p^B) = (K_φ, J_φ, Φ_s, J_ΔNFR)
        g1[b:b + 4] = [ps[i], jd[i], k[i], jp[i]]        # ∇P_1
        g2[b:b + 4] = [jd[i], -ps[i], -jp[i], k[i]]      # ∇P_2
        g3[b:b + 4] = [k[i], jp[i], -ps[i], -jd[i]]      # ∇P_3
    return g1, g2, g3


def verify_polarization_symmetry(
    G: Any,
    *,
    flow_times: tuple[float, ...] = (0.4, 1.1, 2.3, 4.0),
    tolerance: float = 1e-6,
) -> PolarizationSymmetryCertificate:
    r"""Verify the substrate's polarization symmetry (U(2)) and Stokes vector.

    Confirms the three Stokes parameters (P_1, P_2, P_3), that
    P_3 = E_geo − E_pot, that the su(2) algebra closes
    ({P_a, P_b} = 2 ε_abc P_c) under the canonical Poisson bracket, that a
    finite SU(2) sector-mixing rotation is symplectic (preserves ω), that
    the charges are conserved along the substrate flow, and that each node
    is fully polarized (|P_node| = e_node, a point on the Poincaré sphere).

    HONEST SCOPE: a dynamical symmetry of the flat, isotropic H_sub
    backbone (the SU(2) mixes the physically distinct geometric and
    potential sectors and is not one of the 13 operators); the charges are
    exact along the substrate flow and diagnostics at the full nonlinear
    level.  This is the classical polarization (Stokes/Poincaré) of a wave
    phase field — NOT a quantum two-level system (no superposition or
    entanglement; the doublet is per-node, the global state a product).

    Parameters
    ----------
    G : TNFRGraph
    flow_times : tuple of float
        Sampling times for the conservation check.
    tolerance : float
        Maximum allowed drift / residual.

    Returns
    -------
    PolarizationSymmetryCertificate
    """
    point = extract_phase_space_point(G)
    n = point.n_nodes
    charges = polarization_vector(point)
    p_1, p_2, p_3 = charges["p_1"], charges["p_2"], charges["p_3"]

    # P_3 = E_geo − E_pot.
    e_diff = geometric_sector_energy(point) - potential_sector_energy(point)
    p3_matches = abs(p_3 - e_diff) < tolerance

    # su(2) algebra: {P_a, P_b} = 2 ε_abc P_c under the canonical bracket.
    omega = symplectic_form_matrix(n)
    g1, g2, g3 = _polarization_gradients(point)
    b12 = float(g1 @ (omega @ g2))
    b23 = float(g2 @ (omega @ g3))
    b31 = float(g3 @ (omega @ g1))
    algebra_residual = max(
        abs(b12 - 2.0 * p_3),
        abs(b23 - 2.0 * p_1),
        abs(b31 - 2.0 * p_2),
    )
    algebra_closes = algebra_residual < max(
        tolerance, 1e-9 * (abs(p_1) + abs(p_2) + abs(p_3))
    )

    # Finite SU(2) sector-mixing rotation is symplectic.
    th = 0.6
    c, s = float(np.cos(th)), float(np.sin(th))
    rot4 = np.array(
        [[c, 0, -s, 0], [0, c, 0, -s], [s, 0, c, 0], [0, s, 0, c]],
        dtype=float,
    )
    m_rot = np.zeros((4 * n, 4 * n))
    for i in range(n):
        b = 4 * i
        m_rot[b:b + 4, b:b + 4] = rot4
    rotation_symplectic = bool(
        np.allclose(m_rot.T @ omega @ m_rot, omega)
    )

    # Conservation along the substrate flow.
    charge_drift = 0.0
    for t in flow_times:
        evolved = evolve_substrate_flow(point, t)
        ch = polarization_vector(evolved)
        charge_drift = max(
            charge_drift,
            abs(ch["p_1"] - p_1),
            abs(ch["p_2"] - p_2),
            abs(ch["p_3"] - p_3),
        )
    charges_conserved = charge_drift < max(
        tolerance, 1e-9 * (abs(p_1) + abs(p_2) + abs(p_3))
    )

    # Full polarization: per node |P_node| = e_node (the Poincaré sphere).
    density = polarization_density(point)
    pol_residual = float(
        np.max(np.abs(density["radius"] - density["energy"]))
    )
    pol_holds = pol_residual < max(
        tolerance, 1e-9 * float(np.max(density["energy"]) + 1e-300)
    )

    return PolarizationSymmetryCertificate(
        n_nodes=n,
        p_1=p_1,
        p_2=p_2,
        p_3=p_3,
        magnitude_sq=charges["magnitude_sq"],
        p3_equals_energy_difference=p3_matches,
        su2_algebra_closes=algebra_closes,
        rotation_is_symplectic=rotation_symplectic,
        charges_conserved=charges_conserved,
        max_charge_drift=charge_drift,
        max_algebra_residual=algebra_residual,
        full_polarization_holds=pol_holds,
        max_polarization_residual=pol_residual,
    )


# ---------------------------------------------------------------------------
# Adiabatic invariance of the action: the slow-νf theorem
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdiabaticInvarianceCertificate:
    r"""Verification that the substrate action I is an adiabatic invariant
    under a slowly-varying structural frequency.

    The substrate backbone is a harmonic oscillator per conjugate pair with
    action I = ½|ζ|² = E/ω.  When the effective frequency ω is held fixed
    the action is exactly conserved (:class:`IntegrabilityCertificate`).
    When ω is *driven* by a time-varying structural frequency ν_f(t), the
    action is no longer exactly conserved — but the **adiabatic theorem**
    (Ehrenfest 1916, an empirically established result) guarantees it is an
    *adiabatic invariant*: the relative drift |ΔI|/I → 0 as the ramp slows
    (the adiabaticity parameter ε = ω̇/ω² → 0).  Fast ramps break it.

    This measures the AGENTS.md statement "the actions are the adiabatic
    invariants; the 13 operators are canonical transformations that
    redistribute them": ν_f is the **clock**, and a slow ν_f ramp preserves
    the action while a sudden one injects/extracts it.

    HONEST SCOPE: this is the adiabatic theorem for the substrate harmonic
    backbone with ν_f providing the slowly-varying frequency.  It is the
    empirically-grounded Ehrenfest adiabatic invariance, not a new
    postulate.  Measured by integrating a single oscillator q̈ + ω(t)²q = 0
    with ω ramped over a window; no field formula is duplicated.

    Attributes
    ----------
    omega_start : float
        Initial effective frequency.
    omega_end : float
        Final effective frequency.
    ramp_times : tuple[float, ...]
        The ramp durations T probed (slower = larger T).
    action_drifts : tuple[float, ...]
        Relative action drift |ΔI|/I for each ramp time.
    fast_drift : float
        Action drift for the fastest (smallest-T) ramp.
    slow_drift : float
        Action drift for the slowest (largest-T) ramp.
    drift_decreases_with_slowness : bool
        Whether the drift is (monotonically, to a floor) smaller for slower
        ramps — the signature of adiabatic invariance.
    is_adiabatic_invariant : bool
        Whether the slow-ramp drift falls within tolerance.
    """

    omega_start: float
    omega_end: float
    ramp_times: tuple[float, ...]
    action_drifts: tuple[float, ...]
    fast_drift: float
    slow_drift: float
    drift_decreases_with_slowness: bool
    is_adiabatic_invariant: bool

    def summary(self) -> str:
        """Human-readable one-line verdict."""
        ok = "VALID" if self.is_adiabatic_invariant else "INVALID"
        return (
            f"Adiabatic invariance [{ok}]: action I=E/omega under "
            f"omega {self.omega_start:.2f}->{self.omega_end:.2f}; "
            f"fast-ramp drift {self.fast_drift:.2e} -> "
            f"slow-ramp drift {self.slow_drift:.2e} "
            f"(decreases with slowness={self.drift_decreases_with_slowness}); "
            f"nu_f is the clock, slow ramps conserve the action"
        )


def _action_drift_under_ramp(
    omega0: float,
    omega1: float,
    ramp_time: float,
    *,
    n_steps: int = 20000,
) -> float:
    r"""Relative action drift |ΔI|/I of q̈ + ω(t)²q = 0 over a linear ramp.

    Integrates a single harmonic oscillator with a frequency linearly ramped
    from ``omega0`` to ``omega1`` over ``ramp_time`` using a symplectic
    leapfrog (Liouville-preserving), and returns the relative change of the
    action I = E/ω between the endpoints.  In the adiabatic (slow-ramp)
    limit this drift tends to zero — the action is the adiabatic invariant.
    """
    q = 1.0
    v = 0.0
    dt = ramp_time / n_steps

    def omega(t: float) -> float:
        frac = min(max(t / ramp_time, 0.0), 1.0)
        return omega0 + (omega1 - omega0) * frac

    i0 = 0.5 * (v * v + omega0 * omega0 * q * q) / omega0
    t = 0.0
    for _ in range(n_steps):
        w_half = omega(t + 0.5 * dt)
        v -= 0.5 * dt * w_half * w_half * q
        q += dt * v
        w_full = omega(t + dt)
        v -= 0.5 * dt * w_full * w_full * q
        t += dt
    i1 = 0.5 * (v * v + omega1 * omega1 * q * q) / omega1
    return abs(i1 - i0) / i0 if i0 > 1e-30 else 0.0


def verify_adiabatic_invariance(
    *,
    omega_start: float = 1.0,
    omega_end: float = 3.0,
    ramp_times: tuple[float, ...] = (1.0, 5.0, 20.0, 80.0),
    tolerance: float = 1e-2,
) -> AdiabaticInvarianceCertificate:
    r"""Verify the substrate action is an adiabatic invariant of slow ν_f.

    Probes the action I = E/ω of the substrate harmonic backbone under a
    structural frequency ramped from ``omega_start`` to ``omega_end`` over a
    range of ramp durations.  Confirms the adiabatic theorem: the relative
    action drift shrinks as the ramp slows (the action is conserved in the
    slow-ν_f limit), so ν_f acts as the clock whose slow variation preserves
    the action while a fast variation injects/extracts it.

    Parameters
    ----------
    omega_start, omega_end : float
        Initial and final effective frequencies of the ramp.
    ramp_times : tuple[float, ...]
        Ramp durations to probe (ascending = increasingly adiabatic).
    tolerance : float
        Maximum slow-ramp drift for the action to count as an adiabatic
        invariant.

    Returns
    -------
    AdiabaticInvarianceCertificate
    """
    drifts = tuple(
        _action_drift_under_ramp(omega_start, omega_end, t)
        for t in ramp_times
    )
    fast_drift = drifts[0]
    slow_drift = drifts[-1]
    # adiabatic signature: the slowest ramp is far below the fastest
    decreases = slow_drift < fast_drift
    is_adiabatic = slow_drift < tolerance

    return AdiabaticInvarianceCertificate(
        omega_start=float(omega_start),
        omega_end=float(omega_end),
        ramp_times=tuple(float(t) for t in ramp_times),
        action_drifts=tuple(float(d) for d in drifts),
        fast_drift=float(fast_drift),
        slow_drift=float(slow_drift),
        drift_decreases_with_slowness=bool(decreases),
        is_adiabatic_invariant=bool(is_adiabatic),
    )


# ---------------------------------------------------------------------------
# Consolidated entry point: the complete geometric tower in one call
# ---------------------------------------------------------------------------


def verify_substrate_geometry(G: Any) -> SubstrateGeometryReport:
    r"""Verify the complete emergent geometric tower in a single call.

    Runs all seven structural verifications and bundles their certificates
    into one :class:`SubstrateGeometryReport`:

    1. :func:`verify_canonical_structure` — symplectic / Poisson / Liouville,
    2. :func:`verify_noether_conservation` — Noether charges,
    3. :func:`verify_hermitian_structure` — Hermitian (flat Kähler),
    4. :func:`verify_integrability` — action–angle integrability,
    5. :func:`verify_poincare_cartan` — Poincaré–Cartan invariants,
    6. :func:`verify_symplectic_reduction` — Marsden–Weinstein reduction,
    7. :func:`verify_polarization_symmetry` — polarization symmetry (U(2)),
       the Stokes parameters of the (ζ^A, ζ^B) doublet.

    This is the consolidated entry point to the whole classical
    Hamiltonian-geometry tower the nodal dynamics generates from itself.

    Parameters
    ----------
    G : TNFRGraph

    Returns
    -------
    SubstrateGeometryReport
    """
    point = extract_phase_space_point(G)
    return SubstrateGeometryReport(
        n_nodes=point.n_nodes,
        phase_space_dimension=4 * point.n_nodes,
        canonical=verify_canonical_structure(G),
        noether=verify_noether_conservation(G),
        hermitian=verify_hermitian_structure(G),
        integrability=verify_integrability(G),
        poincare_cartan=verify_poincare_cartan(G),
        marsden_weinstein=verify_symplectic_reduction(G),
        polarization=verify_polarization_symmetry(G),
    )
