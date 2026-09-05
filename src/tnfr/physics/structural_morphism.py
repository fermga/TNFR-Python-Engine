r"""Structural morphisms: nodal-flow transports between networks (R4/R8).

The operator-certification audit (R8) rejects four arithmetic maps *as operators*
— CRT projection (a relabeling), the affine map (an automorphism), the power map
(an endomorphism) and the p-adic lift (a transport map) — while the p-adic tower
(R4) supplies exact projection/lift **intertwiners**.  Those rejections and that
transport are the same thing seen twice: they are **morphisms of the structural
category**, not nodal reorganizations.

A canonical operator changes nodal state through the nodal equation
``∂EPI/∂t = ν_f · ΔNFR`` (a NODE- or NETWORK-scale reorganization).  A structural
morphism is a linear map ``M`` **between** two networks that transports the
diffusion generator, ``M L_src = L_tgt M`` (an intertwiner); it relabels,
aggregates or prolongs structure but performs no reorganization.  This module
classifies such maps and issues a certificate — sharpening the 13-operator
boundary without inventing a fourteenth operator.

**Emergence from the nodal equation.**  These morphisms are **not** an imposed
category: they emerge directly from ``∂EPI/∂t = ν_f · ΔNFR``.  For the EPI channel
with a common ``ν_f`` the nodal equation is ``dEPI/dt = −ν_f L · EPI`` with flow
``EPI(t) = e^{−ν_f t L} EPI(0)``.  A map ``M`` carries **every** source solution to
a target solution,

    ``M e^{−s L_src} EPI₀ = e^{−s L_tgt} M EPI₀``  for all ``s, EPI₀``,

**iff** ``M L_src = L_tgt M`` (differentiate at ``s = 0`` for ⇒; for ⇐ both sides
solve the same ODE ``d/ds(·) = −L_tgt(·)`` with equal initial data).  So the
intertwining defect *is* the nodal-flow-preservation defect, and the
structure-preserving morphisms are exactly the maps that commute with the
nodal-equation semigroup.  The taxonomy below is the classification of these
nodal-flow transports by their dimension change and rank; a folding
``ENDOMORPHISM`` generally fails the intertwining test, so it does **not** emerge
as a nodal transport — precisely the R8 boundary.

**Taxonomy.**

- ``RELABELING`` — bijection to a **re-labelled** graph (``L_src ≠ L_tgt``);
  ``AUTOMORPHISM`` is the sub-case onto the **same** graph (``L_src = L_tgt``,
  the symmetry of one NFR — R1).
- ``INTERTWINER`` — full-rank same-dimension transport that is not a permutation
  (a change of coordinates conjugating isospectral generators).
- ``PROJECTION`` — an idempotent onto an ``L``-invariant sector: either
  dimension-dropping (a non-partition surjection) or same-dimension (the ambient
  Reynolds sector projector ``Q_Γ`` of R1); ``COARSE_GRAINING`` is the sub-case
  that is a fiber **partition average** (the U5 quotient — R4).
- ``LIFT`` — dimension-raising injection (prolongation, the U5 lift — R4).
- ``ENDOMORPHISM`` — a rank-deficient **folding** self-map that does **not**
  intertwine, so it does **not** emerge from the nodal flow (the R8 boundary).

The genus is the intertwiner; the species are the cells of the (dimension change)
× (rank type) grid, with ``AUTOMORPHISM ⊆ RELABELING`` and
``COARSE_GRAINING ⊆ PROJECTION`` as canonical refinements.  The
``INTERTWINER`` property (``M L_src = L_tgt M``) is orthogonal to the kind: a
lift, a coarse-graining and an automorphism are all intertwiners; the kind is
fixed by the dimension change and the injectivity/surjectivity of ``M``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .spectral_projectors import derived_tolerance, matrix_exponential

__all__ = [
    "StructuralMorphismKind",
    "is_permutation_matrix",
    "is_partition_average",
    "is_idempotent",
    "intertwining_residual",
    "nodal_flow_preservation_residual",
    "classify_morphism",
    "StructuralMorphismCertificate",
    "certify_morphism",
    "audit_structural_morphisms",
]


class StructuralMorphismKind(Enum):
    """The kinds of structure-preserving map between TNFR networks."""

    RELABELING = "relabeling"
    AUTOMORPHISM = "automorphism"
    ENDOMORPHISM = "endomorphism"
    PROJECTION = "projection"
    LIFT = "lift"
    INTERTWINER = "intertwiner"
    COARSE_GRAINING = "coarse_graining"


def _as_float(matrix) -> np.ndarray:
    return np.asarray(matrix, dtype=float)


def is_permutation_matrix(matrix, *, tol: float = 1e-9) -> bool:
    r"""Whether ``M`` is a permutation matrix (a bijective relabeling)."""
    m = _as_float(matrix)
    if m.shape[0] != m.shape[1]:
        return False
    binary = np.all((np.abs(m) < tol) | (np.abs(m - 1.0) < tol))
    rows = np.all(np.abs(m.sum(axis=1) - 1.0) < tol)
    cols = np.all(np.abs(m.sum(axis=0) - 1.0) < tol)
    return bool(binary and rows and cols)


def is_partition_average(matrix, *, tol: float = 1e-9) -> bool:
    r"""Whether ``M`` is a fiber-average quotient (row-stochastic partition).

    Row-stochastic (each coarse node is a weighted average) **and** every fine
    node contributes to exactly one coarse node (each column has one nonzero).
    """
    m = _as_float(matrix)
    if m.shape[0] >= m.shape[1]:
        return False
    nonneg = bool(np.all(m >= -tol))
    row_stoch = bool(np.all(np.abs(m.sum(axis=1) - 1.0) < tol))
    one_per_col = bool(np.all((np.abs(m) > tol).sum(axis=0) == 1))
    return nonneg and row_stoch and one_per_col


def is_idempotent(matrix, *, tol: float = 1e-9) -> bool:
    r"""Whether ``M² = M`` — an idempotent (a projector onto its image)."""
    m = _as_float(matrix)
    if m.shape[0] != m.shape[1]:
        return False
    return bool(np.linalg.norm(m @ m - m, 2) < max(tol, 1e-9))


def intertwining_residual(morphism, laplacian_src, laplacian_tgt) -> float:
    r"""``‖M L_src − L_tgt M‖₂`` — the transport defect of the morphism.

    Zero means ``M`` conjugates the source diffusion generator into the target
    one (an intertwiner), so coarse modes survive: ``spec(L_src)`` relates to
    ``spec(L_tgt)`` through ``M``.  By the emergence theorem this defect equals
    the nodal-flow-preservation defect (the ``s = 0`` derivative of
    :func:`nodal_flow_preservation_residual`).
    """
    m = _as_float(morphism)
    ls = _as_float(laplacian_src)
    lt = _as_float(laplacian_tgt)
    return float(np.linalg.norm(m @ ls - lt @ m, 2))


def nodal_flow_preservation_residual(
    morphism, laplacian_src, laplacian_tgt, x0=None, *,
    s_max: float = 6.0, samples: int = 60
) -> float:
    r"""``max_s ‖M e^{−s L_src} x₀ − e^{−s L_tgt} M x₀‖`` — the direct nodal test.

    Measures whether ``M`` carries a solution of the source nodal equation
    ``dEPI/dt = −L_src EPI`` to a solution of the target one over the whole
    trajectory (not just infinitesimally).  By the emergence theorem this is
    ``≈ 0`` **iff** ``M`` intertwines; a morphism that fails it does not emerge
    from ``∂EPI/∂t = ν_f · ΔNFR`` (e.g. a folding endomorphism).
    """
    m = _as_float(morphism)
    ls = _as_float(laplacian_src)
    lt = _as_float(laplacian_tgt)
    n_src = m.shape[1]
    if x0 is None:  # deterministic non-consensus probe
        x0 = np.array([(-1.0) ** i * (1.0 + i) for i in range(n_src)])
    x0 = _as_float(x0)
    resid = 0.0
    for s in np.linspace(0.0, s_max, samples):
        lhs = m @ (matrix_exponential(-ls * s) @ x0)
        rhs = matrix_exponential(-lt * s) @ (m @ x0)
        resid = max(resid, float(np.linalg.norm(lhs - rhs)))
    return resid


def classify_morphism(
    morphism, laplacian_src, laplacian_tgt, *, tol: float | None = None
) -> StructuralMorphismKind:
    r"""Classify ``M : (V_src, L_src) → (V_tgt, L_tgt)`` into its structural kind."""
    m = _as_float(morphism)
    n_tgt, n_src = m.shape
    if tol is None:
        tol = derived_tolerance(m)
    rank = int(np.linalg.matrix_rank(m, tol=max(tol, 1e-12)))

    if n_tgt < n_src:  # dimension reduction
        if is_partition_average(m, tol=tol):
            return StructuralMorphismKind.COARSE_GRAINING
        return StructuralMorphismKind.PROJECTION
    if n_tgt > n_src:  # dimension increase
        return StructuralMorphismKind.LIFT
    # square: n_tgt == n_src
    if is_permutation_matrix(m, tol=tol):
        same = np.linalg.norm(
            _as_float(laplacian_src) - _as_float(laplacian_tgt), 2
        ) < max(tol, 1e-9)
        return (StructuralMorphismKind.AUTOMORPHISM if same
                else StructuralMorphismKind.RELABELING)
    if rank < n_src:  # rank-deficient self-map
        # an intertwining idempotent projects onto an L-invariant sector (it
        # emerges from the nodal flow, e.g. the Reynolds projector Q_Γ); a
        # folding map that fails intertwining does not emerge (R8 boundary).
        intertwines = intertwining_residual(
            m, laplacian_src, laplacian_tgt) < max(tol, 1e-6)
        if intertwines and is_idempotent(m, tol=tol):
            return StructuralMorphismKind.PROJECTION
        return StructuralMorphismKind.ENDOMORPHISM
    return StructuralMorphismKind.INTERTWINER


@dataclass(frozen=True)
class StructuralMorphismCertificate:
    """Classification + structural diagnostics of a network morphism."""

    kind: StructuralMorphismKind
    domain_dim: int
    codomain_dim: int
    rank: int
    is_injective: bool
    is_surjective: bool
    is_bijection: bool
    intertwining_residual: float
    is_intertwiner: bool
    nodal_flow_residual: float
    emerges_from_nodal_equation: bool
    is_operator: bool  # always False: a morphism is not a nodal reorganization
    tolerance: float
    claim_status: str


def certify_morphism(
    morphism, laplacian_src, laplacian_tgt, *, tol: float | None = None,
    flow_probe=None,
) -> StructuralMorphismCertificate:
    r"""Bundle the classification and structure diagnostics for a morphism.

    ``emerges_from_nodal_equation`` is the direct test that ``M`` carries the
    nodal-equation flow (``M e^{−sL_src} = e^{−sL_tgt} M``); by the emergence
    theorem it agrees with ``is_intertwiner``.  ``is_operator`` is always
    ``False``: a structural morphism transports the flow but performs no
    ``∂EPI/∂t = ν_f · ΔNFR`` reorganization, so it is **not** one of the 13
    canonical operators (the R8 boundary).
    """
    m = _as_float(morphism)
    n_tgt, n_src = m.shape
    if tol is None:
        tol = derived_tolerance(m)
    rank = int(np.linalg.matrix_rank(m, tol=max(tol, 1e-12)))
    resid = intertwining_residual(m, laplacian_src, laplacian_tgt)
    flow = nodal_flow_preservation_residual(
        m, laplacian_src, laplacian_tgt, flow_probe
    )
    kind = classify_morphism(m, laplacian_src, laplacian_tgt, tol=tol)
    inj = rank == n_src
    surj = rank == n_tgt
    threshold = max(tol, 1e-6)
    return StructuralMorphismCertificate(
        kind=kind,
        domain_dim=n_src,
        codomain_dim=n_tgt,
        rank=rank,
        is_injective=inj,
        is_surjective=surj,
        is_bijection=inj and surj,
        intertwining_residual=resid,
        is_intertwiner=resid < threshold,
        nodal_flow_residual=flow,
        emerges_from_nodal_equation=flow < threshold,
        is_operator=False,
        tolerance=tol,
        claim_status=(
            "taxonomy DERIVED from the nodal equation (intertwiner = nodal-flow "
            "transport); example kinds MEASURED; not a canonical operator (R8 "
            "boundary); no 14th operator"
        ),
    )


def audit_structural_morphisms() -> list[tuple[str, StructuralMorphismCertificate]]:
    r"""Certify one canonical example of each morphism kind.

    Six kinds emerge from the nodal equation (intertwiners: automorphism,
    relabeling, coarse-graining, lift, conjugation-intertwiner, and the Reynolds
    sector projector); the folding endomorphism does **not** (the R8 boundary).
    """
    import networkx as nx

    from ..mathematics.padic_tower import (
        compatible_connection_set,
        padic_laplacian,
        padic_lift_map,
        projective_scale_map,
    )
    from .directed_diffusion import directed_rw_laplacian
    from .symmetry_sectors import permutation_matrix, reynolds_projector

    def lap(g):
        return directed_rw_laplacian(nx.to_numpy_array(g))

    def frac(matrix):
        return np.array([[float(x) for x in row] for row in matrix], dtype=float)

    out: list[tuple[str, StructuralMorphismCertificate]] = []

    # AUTOMORPHISM: cycle rotation on a fixed graph
    lc = lap(nx.cycle_graph(6))
    rot = permutation_matrix({i: (i + 1) % 6 for i in range(6)}, list(range(6)))
    out.append(("automorphism", certify_morphism(rot, lc, lc)))

    # RELABELING: path relabelled to an isomorphic copy
    lp = lap(nx.path_graph(4))
    q = permutation_matrix({0: 2, 1: 0, 2: 3, 3: 1}, list(range(4)))
    out.append(("relabeling", certify_morphism(q, lp, q @ lp @ q.T)))

    # COARSE_GRAINING and LIFT: the p-adic scale adjunction (R4, U5)
    base = frozenset({1, 2})
    l_hi = frac(padic_laplacian(3, 2, compatible_connection_set(3, 2, base)))
    l_lo = frac(padic_laplacian(3, 1, compatible_connection_set(3, 1, base)))
    out.append(("coarse_graining",
                certify_morphism(frac(projective_scale_map(3, 1)), l_hi, l_lo)))
    out.append(("lift",
                certify_morphism(frac(padic_lift_map(3, 1)), l_lo, l_hi)))

    # INTERTWINER: a non-permutation change of coordinates conjugating L
    ls = lap(nx.cycle_graph(4))
    shear = np.array([[1, 0.3, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.2], [0, 0, 0, 1]],
                     dtype=float)
    out.append(("intertwiner",
                certify_morphism(shear, ls, shear @ ls @ np.linalg.inv(shear))))

    # PROJECTION: the Reynolds sector projector Q_Γ (R1) — the emergent case that
    # a rank test alone would miss
    star = nx.star_graph(4)
    l_star = lap(star)
    out.append(("projection",
                certify_morphism(reynolds_projector(star, nodes=list(range(5))),
                                 l_star, l_star)))

    # ENDOMORPHISM: the folding power map x -> x^2 (mod 7) — does NOT emerge
    p = 7
    fold = np.zeros((p, p))
    for x in range(p):
        fold[(x * x) % p, x] = 1.0
    l7 = lap(nx.cycle_graph(7))
    out.append(("endomorphism", certify_morphism(fold, l7, l7)))
    return out
