r"""Equivariance certificates for the canonical emergent operator (R1).

Certifies the **diffusion-sector observability** result: the structural-diffusion
operator ``L_rw`` commutes with every automorphism of the graph (equivariance
residual ``max_σ ‖P_σ L − L P_σ‖ ≈ 0``) and preserves the Reynolds sectors
(``‖L Q_Γ − Q_Γ L‖ ≈ 0``).  Consequently ``Fix(Γ)`` and ``Fix(Γ)^⊥`` are
invariant under the overdamped nodal-equation flow ``ẋ = −D_νf · L_rw · x``
whenever ``νf`` is orbit-constant, so a symmetric seed evolved by the canonical
dynamics can never manufacture per-node structure that distinguishes nodes within
one orbit.  This is the proved, implementation-certified base case that the
operator-by-operator equivariance audit (a later R1 stage) builds on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .spectral_projectors import derived_tolerance
from .structural_diffusion import structural_diffusion_operator
from .symmetry_sectors import (
    automorphism_orbits,
    automorphism_permutations,
    permutation_matrix,
    reynolds_projector,
)

__all__ = [
    "EquivarianceCertificate",
    "equivariance_residual",
    "sector_preservation_residual",
    "verify_diffusion_equivariance",
]


def equivariance_residual(operator_matrix, permutations, nodes) -> float:
    r"""``max_σ ‖P_σ L − L P_σ‖₂`` over the given automorphism permutations."""
    L = np.asarray(operator_matrix)
    best = 0.0
    for mapping in permutations:
        P = permutation_matrix(mapping, nodes)
        best = max(best, float(np.linalg.norm(P @ L - L @ P, 2)))
    return best


def sector_preservation_residual(operator_matrix, projector) -> float:
    r"""``‖L Q_Γ − Q_Γ L‖₂``; zero iff ``L`` preserves ``Fix(Γ) ⊕ Fix(Γ)^⊥``."""
    L = np.asarray(operator_matrix)
    Q = np.asarray(projector)
    return float(np.linalg.norm(L @ Q - Q @ L, 2))


@dataclass(frozen=True)
class EquivarianceCertificate:
    r"""Certificate that ``L_rw`` is equivariant and sector-preserving on ``G``."""

    equivariance_residual: float
    sector_preservation_residual: float
    tolerance: float
    n_automorphisms: int
    orbit_count: int
    is_equivariant: bool


def verify_diffusion_equivariance(
    G,
    *,
    tol: float | None = None,
    weight: str | None = None,
    cap: int = 2000,
) -> EquivarianceCertificate:
    r"""Certify that ``L_rw`` commutes with ``Aut(G)`` and preserves its sectors.

    The tolerance defaults to the derived ``√ε·‖L‖₂``; the certificate reports
    both residuals, the number of automorphisms found and the orbit count.
    """
    nodes, L = structural_diffusion_operator(G)
    perms = automorphism_permutations(G, weight=weight, cap=cap)
    if tol is None:
        tol = derived_tolerance(L)
    eq = equivariance_residual(L, perms, nodes)
    Q = reynolds_projector(G, nodes=nodes, permutations=perms)
    pres = sector_preservation_residual(L, Q)
    orbits = automorphism_orbits(G, nodes=nodes, permutations=perms)
    return EquivarianceCertificate(
        equivariance_residual=eq,
        sector_preservation_residual=pres,
        tolerance=tol,
        n_automorphisms=len(perms),
        orbit_count=len(orbits),
        is_equivariant=(eq < tol and pres < tol),
    )
