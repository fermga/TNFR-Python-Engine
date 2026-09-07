r"""Structural morphisms: nodal-flow transports between networks.

The operator-contract boundary rejects four arithmetic maps *as operators*
— CRT projection (a relabeling), the affine map (an automorphism), the power map
(an endomorphism) and the p-adic lift (a transport map) — while the p-adic tower
supplies exact projection/lift **intertwiners**.  Those rejections and that
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
as a nodal transport.

**Taxonomy.**

- ``RELABELING`` — bijection to a **re-labelled** graph (``L_src ≠ L_tgt``);
  ``AUTOMORPHISM`` is the sub-case onto the **same** graph (``L_src = L_tgt``,
    the symmetry of one NFR).
- ``INTERTWINER`` — full-rank same-dimension transport that is not a permutation
  (a change of coordinates conjugating isospectral generators).
- ``PROJECTION`` — an idempotent onto an ``L``-invariant sector: either
  dimension-dropping (a non-partition surjection) or same-dimension (the ambient
    Reynolds sector projector ``Q_Γ``); ``COARSE_GRAINING`` is the sub-case that
    is a fiber **partition average** (a U5-compatible quotient).
- ``LIFT`` — dimension-raising injection (prolongation, a U5-compatible lift).
- ``ENDOMORPHISM`` — a rank-deficient **folding** self-map that does **not**
    intertwine, so it does **not** emerge from the nodal flow.

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
import math

import numpy as np

from .spectral_projectors import matrix_exponential

__all__ = [
    "StructuralMorphismKind",
    "is_permutation_matrix",
    "is_partition_average",
    "is_idempotent",
    "intertwining_residual",
    "finite_time_intertwining_bound",
    "nodal_flow_preservation_residual",
    "classify_morphism",
    "StructuralMorphismCertificate",
    "EpiCoarseGrainingCertificate",
    "certify_morphism",
    "certify_epi_coarse_graining",
    "audit_structural_morphisms",
]


def _reject_boolean_numeric(value, name: str) -> None:
    """Reject booleans before NumPy can coerce them to zero or one."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be numeric, not boolean")


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
    if np.iscomplexobj(matrix):
        raise ValueError("structural morphism matrices must be real")
    try:
        value = np.asarray(matrix, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("structural morphism matrices must be finite") from exc
    if not np.all(np.isfinite(value)):
        raise ValueError("structural morphism matrices must be finite")
    return value


def _finite_product(left: np.ndarray, right: np.ndarray, name: str) -> np.ndarray:
    """Multiply finite arrays or reject an unrepresentable result explicitly."""
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            result = left @ right
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError(f"{name} exceeds finite floating-point range") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} exceeds finite floating-point range")
    return result


def _finite_difference(left: np.ndarray, right: np.ndarray, name: str) -> np.ndarray:
    """Subtract finite arrays without allowing an infinite residual."""
    try:
        with np.errstate(over="raise", invalid="raise"):
            result = left - right
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError(f"{name} exceeds finite floating-point range") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} exceeds finite floating-point range")
    return result


def _finite_norm(value: np.ndarray, *, matrix: bool, name: str) -> float:
    """Return a scale-safe 2-norm or reject an unrepresentable norm."""
    array = np.asarray(value, dtype=float)
    scale = float(np.max(np.abs(array), initial=0.0))
    if scale == 0.0:
        return 0.0
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            normalized = array / scale
            norm = float(np.linalg.norm(normalized, 2 if matrix else None))
            result = scale * norm
    except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
        raise ValueError(f"{name} exceeds finite floating-point range") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} exceeds finite floating-point range")
    return result


def _intertwining_diagnostics(
    morphism: np.ndarray,
    laplacian_src: np.ndarray,
    laplacian_tgt: np.ndarray,
) -> tuple[float, float]:
    """Return absolute defect and its declared relative comparison scale."""
    source_transport = _finite_product(
        morphism, laplacian_src, "source intertwining product"
    )
    target_transport = _finite_product(
        laplacian_tgt, morphism, "target intertwining product"
    )
    defect = _finite_difference(
        source_transport, target_transport, "intertwining residual"
    )
    residual = _finite_norm(
        defect, matrix=True, name="intertwining residual"
    )
    scale = max(
        1.0,
        _finite_norm(
            source_transport, matrix=True, name="source intertwining scale"
        ),
        _finite_norm(
            target_transport, matrix=True, name="target intertwining scale"
        ),
    )
    return residual, scale


def _validated_morphism_system(morphism, laplacian_src, laplacian_tgt):
    m = _as_float(morphism)
    ls = _as_float(laplacian_src)
    lt = _as_float(laplacian_tgt)
    if m.ndim != 2 or ls.ndim != 2 or lt.ndim != 2:
        raise ValueError("morphism and generators must be matrices")
    if ls.shape[0] != ls.shape[1] or lt.shape[0] != lt.shape[1]:
        raise ValueError("source and target generators must be square")
    if m.shape != (lt.shape[0], ls.shape[0]):
        raise ValueError("morphism shape must map source to target coordinates")
    return m, ls, lt


def _relative_tolerance(matrix: np.ndarray) -> float:
    """Return the dimensionless default tolerance used by morphism decisions."""
    return math.sqrt(np.finfo(float).eps) * max(matrix.shape, default=1)


def _scale_invariant_rank(matrix: np.ndarray, tolerance: float) -> int:
    """Return numerical rank without making it depend on a global map scale."""
    scale = float(np.max(np.abs(matrix), initial=0.0))
    if scale == 0.0:
        return 0
    return int(np.linalg.matrix_rank(matrix / scale, tol=tolerance))


def is_permutation_matrix(matrix, *, tol: float = 1e-9) -> bool:
    r"""Whether ``M`` is a permutation matrix (a bijective relabeling)."""
    _reject_boolean_numeric(tol, "tol")
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
    _reject_boolean_numeric(tol, "tol")
    m = _as_float(matrix)
    if m.shape[0] >= m.shape[1]:
        return False
    nonneg = bool(np.all(m >= -tol))
    row_stoch = bool(np.all(np.abs(m.sum(axis=1) - 1.0) < tol))
    one_per_col = bool(np.all((np.abs(m) > tol).sum(axis=0) == 1))
    return nonneg and row_stoch and one_per_col


def is_idempotent(matrix, *, tol: float = 1e-9) -> bool:
    r"""Whether ``M² = M`` — an idempotent (a projector onto its image)."""
    _reject_boolean_numeric(tol, "tol")
    m = _as_float(matrix)
    if m.shape[0] != m.shape[1]:
        return False
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tol must be finite and positive")
    return bool(np.linalg.norm(m @ m - m, 2) <= tol)


def intertwining_residual(morphism, laplacian_src, laplacian_tgt) -> float:
    r"""``‖M L_src − L_tgt M‖₂`` — the transport defect of the morphism.

    Zero means ``M`` conjugates the source diffusion generator into the target
    one (an intertwiner), so coarse modes survive: ``spec(L_src)`` relates to
    ``spec(L_tgt)`` through ``M``.  By the emergence theorem this defect equals
    the nodal-flow-preservation defect (the ``s = 0`` derivative of
    :func:`nodal_flow_preservation_residual`).
    """
    m, ls, lt = _validated_morphism_system(
        morphism, laplacian_src, laplacian_tgt
    )
    residual, _ = _intertwining_diagnostics(m, ls, lt)
    return residual


def finite_time_intertwining_bound(
    morphism, laplacian_src, laplacian_tgt, x0, *, structural_time: float
) -> tuple[float, float]:
    r"""Return direct defect and a finite-time Duhamel Euclidean bound.

    For ``R = M L_src - L_tgt M``, the flow defect is bounded by
    ``s exp(s max(||L_src||₂, ||L_tgt||₂)) ||R||₂ ||x0||₂``. This generic
    finite-time bound is deliberately conservative. Contractive symmetric
    diffusion semigroups admit the tighter factor ``s`` without the generic
    exponential amplification; this implementation retains the all-matrix
    bound and does not claim tightness. It does not certify a nonlinear
    observer, a tail, U5, or the temporal REMESH contract.
    """
    _reject_boolean_numeric(structural_time, "structural_time")
    if not np.isfinite(structural_time) or structural_time < 0.0:
        raise ValueError("structural_time must be finite and nonnegative")
    m, ls, lt = _validated_morphism_system(
        morphism, laplacian_src, laplacian_tgt
    )
    state = _as_float(x0)
    if m.shape != (lt.shape[0], ls.shape[0]) or state.shape != (ls.shape[0],):
        raise ValueError(
            "morphism, generators and source state have incompatible shapes"
        )
    defect = m @ (matrix_exponential(-structural_time * ls) @ state)
    defect -= matrix_exponential(-structural_time * lt) @ (m @ state)
    residual = intertwining_residual(m, ls, lt)
    growth = max(float(np.linalg.norm(ls, 2)), float(np.linalg.norm(lt, 2)))
    bound = (
        structural_time
        * math.exp(structural_time * growth)
        * residual
        * float(np.linalg.norm(state, 2))
    )
    return float(np.linalg.norm(defect, 2)), bound


def nodal_flow_preservation_residual(
    morphism, laplacian_src, laplacian_tgt, x0=None, *,
    s_max: float = 6.0, samples: int = 60
) -> float:
    r"""``max_s ‖M e^{−s L_src} x₀ − e^{−s L_tgt} M x₀‖`` — the direct nodal test.

    Measures whether ``M`` carries one declared solution of the source nodal equation
    ``dEPI/dt = −L_src EPI`` to a solution of the target one over the whole
    sampled trajectory.  A nonzero value disproves transport for that probe,
    while a zero value does not prove the all-state intertwining identity: the
    probe can lie in a shared invariant subspace.  The exact emergence theorem
    is instead ``M L_src = L_tgt M`` iff flow transport holds for every initial
    state and structural time.
    """
    _reject_boolean_numeric(s_max, "s_max")
    _reject_boolean_numeric(samples, "samples")
    m, ls, lt = _validated_morphism_system(
        morphism, laplacian_src, laplacian_tgt
    )
    ls = _as_float(laplacian_src)
    lt = _as_float(laplacian_tgt)
    n_src = m.shape[1]
    if not np.isfinite(s_max) or s_max <= 0.0:
        raise ValueError("s_max must be finite and positive")
    if not isinstance(samples, (int, np.integer)) or samples < 2:
        raise ValueError("samples must be an integer of at least two")
    if x0 is None:  # deterministic non-consensus probe
        x0 = np.array([(-1.0) ** i * (1.0 + i) for i in range(n_src)])
    x0 = _as_float(x0)
    if x0.shape != (n_src,):
        raise ValueError("flow probe must match the source dimension")
    resid = 0.0
    for s in np.linspace(0.0, s_max, samples):
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                source_flow = matrix_exponential(-ls * s)
                target_flow = matrix_exponential(-lt * s)
        except (FloatingPointError, OverflowError) as exc:
            raise ValueError(
                "sampled nodal flow exceeds finite floating-point range"
            ) from exc
        if (not np.all(np.isfinite(source_flow))
                or not np.all(np.isfinite(target_flow))):
            raise ValueError("sampled nodal flow exceeds finite floating-point range")
        source_state = _finite_product(source_flow, x0, "sampled source flow")
        lhs = _finite_product(m, source_state, "transported source flow")
        mapped_initial = _finite_product(m, x0, "mapped initial flow state")
        rhs = _finite_product(target_flow, mapped_initial, "sampled target flow")
        defect = _finite_difference(lhs, rhs, "sampled nodal-flow residual")
        resid = max(
            resid,
            _finite_norm(
                defect, matrix=False, name="sampled nodal-flow residual"
            ),
        )
    return resid


def classify_morphism(
    morphism, laplacian_src, laplacian_tgt, *, tol: float | None = None
) -> StructuralMorphismKind:
    r"""Classify ``M : (V_src, L_src) → (V_tgt, L_tgt)`` into its structural kind."""
    if tol is not None:
        _reject_boolean_numeric(tol, "tol")
    m, ls, lt = _validated_morphism_system(
        morphism, laplacian_src, laplacian_tgt
    )
    n_tgt, n_src = m.shape
    if tol is None:
        tol = _relative_tolerance(m)
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tol must be finite and positive")
    rank = _scale_invariant_rank(m, float(tol))

    if n_tgt < n_src:  # dimension reduction
        if is_partition_average(m, tol=tol):
            return StructuralMorphismKind.COARSE_GRAINING
        return StructuralMorphismKind.PROJECTION
    if n_tgt > n_src:  # dimension increase
        return StructuralMorphismKind.LIFT
    # square: n_tgt == n_src
    if is_permutation_matrix(m, tol=tol):
        residual, residual_scale = _intertwining_diagnostics(m, ls, lt)
        intertwines = residual <= float(tol) * residual_scale
        same_defect = _finite_difference(ls, lt, "generator equality residual")
        same_scale = max(
            1.0,
            _finite_norm(ls, matrix=True, name="source generator scale"),
            _finite_norm(lt, matrix=True, name="target generator scale"),
        )
        same = _finite_norm(
            same_defect, matrix=True, name="generator equality residual"
        ) <= float(tol) * same_scale
        # A permutation of one graph is an automorphism only when it preserves
        # that graph's nodal generator.  A non-commuting vertex permutation is
        # merely a relabeling and the certificate records that it is not an
        # intertwiner for the supplied target.
        return (
            StructuralMorphismKind.AUTOMORPHISM
            if same and intertwines
            else StructuralMorphismKind.RELABELING
        )
    if rank < n_src:  # rank-deficient self-map
        # an intertwining idempotent projects onto an L-invariant sector (it
        # emerges from the nodal flow, e.g. the Reynolds projector Q_Γ); a
        # A folding map that fails intertwining does not emerge.
        residual, residual_scale = _intertwining_diagnostics(m, ls, lt)
        intertwines = residual <= float(tol) * residual_scale
        if intertwines and is_idempotent(m, tol=tol):
            return StructuralMorphismKind.PROJECTION
        return StructuralMorphismKind.ENDOMORPHISM
    return StructuralMorphismKind.INTERTWINER


@dataclass(frozen=True)
class StructuralMorphismCertificate:
    """Classification and numerical diagnostics of a network morphism.

    The exact theorem is ``M L_src = L_tgt M`` iff ``M`` transports every
    corresponding nodal flow.  The Boolean fields below only certify that the
    floating-point residual divided by ``residual_scale`` satisfies the
    declared dimensionless relative tolerance.
    """

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
    residual_scale: float = 1.0
    relative_intertwining_residual: float = 0.0

    @property
    def intertwines_within_tolerance(self) -> bool:
        """Accurate name for the legacy stored numerical-intertwiner flag."""
        return self.is_intertwiner

    @property
    def nodal_flow_transport_within_tolerance(self) -> bool:
        """Accurate name for the legacy stored numerical-transport flag."""
        return self.emerges_from_nodal_equation


@dataclass(frozen=True)
class EpiCoarseGrainingCertificate:
    """Numerical closure test for a partition of the pure EPI nodal flow.

    The projection averages each block with the reversible metric
    ``h_i=d_i/nu_f_i``.  The quotient conductance is the total conductance
    between blocks and its effective capacity is ``d_bar/h_bar``.
    """

    nodes: tuple
    blocks: tuple[tuple, ...]
    projection: np.ndarray
    lift: np.ndarray
    micro_generator: np.ndarray
    macro_generator: np.ndarray
    macro_conductance: np.ndarray
    macro_frequency: np.ndarray
    macro_metric_weights: np.ndarray
    macro_epi: np.ndarray
    projection_residual: float
    lift_residual: float
    information_loss_dimension: int
    nodal_closure_within_tolerance: bool
    morphism: StructuralMorphismCertificate
    scope: str
    projection_residual_scale: float = 1.0
    lift_residual_scale: float = 1.0
    relative_projection_residual: float = 0.0
    relative_lift_residual: float = 0.0


def certify_morphism(
    morphism, laplacian_src, laplacian_tgt, *, tol: float | None = None,
    flow_probe=None,
) -> StructuralMorphismCertificate:
    r"""Bundle the classification and structure diagnostics for a morphism.

    ``nodal_flow_transport_within_tolerance`` is the numerical generator-level
    test associated with the exact transport theorem.  The optional flow probe
    is retained as a sampled diagnostic and cannot establish this flag by itself.
    ``is_operator`` is always
    ``False``: a structural morphism transports the flow but performs no
    ``∂EPI/∂t = ν_f · ΔNFR`` reorganization, so it is **not** one of the 13
    canonical operators.
    """
    if tol is not None:
        _reject_boolean_numeric(tol, "tol")
    m, ls, lt = _validated_morphism_system(
        morphism, laplacian_src, laplacian_tgt
    )
    n_tgt, n_src = m.shape
    if tol is None:
        tol = _relative_tolerance(m)
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tol must be finite and positive")
    rank = _scale_invariant_rank(m, float(tol))
    resid, residual_scale = _intertwining_diagnostics(m, ls, lt)
    relative_residual = resid / residual_scale
    flow = nodal_flow_preservation_residual(
        m, ls, lt, flow_probe
    )
    kind = classify_morphism(m, ls, lt, tol=tol)
    inj = rank == n_src
    surj = rank == n_tgt
    return StructuralMorphismCertificate(
        kind=kind,
        domain_dim=n_src,
        codomain_dim=n_tgt,
        rank=rank,
        is_injective=inj,
        is_surjective=surj,
        is_bijection=inj and surj,
        intertwining_residual=resid,
        is_intertwiner=relative_residual <= float(tol),
        nodal_flow_residual=flow,
        # A single probe can lie in a shared invariant subspace (for example,
        # the constant consensus vector), so it cannot certify every state.
        emerges_from_nodal_equation=relative_residual <= float(tol),
        is_operator=False,
        tolerance=tol,
        claim_status=(
            "exact intertwiner/flow equivalence DERIVED from the nodal equation; "
            "this floating-point instance is MEASURED within the declared "
            "relative tolerance; not a canonical operator; no 14th operator"
        ),
        residual_scale=residual_scale,
        relative_intertwining_residual=relative_residual,
    )


def certify_epi_coarse_graining(
    graph, partition, *, tolerance: float = 1e-10,
) -> EpiCoarseGrainingCertificate:
    r"""Construct and test the canonical reversible quotient of EPI diffusion.

    For ``A=diag(nu_f)L_rw=H^-1 B`` with ``H=diag(d_i/nu_f_i)``, let ``P`` lift
    one macro value to every node in its block and let

    ``R=(P^T H P)^-1 P^T H``.

    Thus ``R P=I`` and macro EPI is the ``H``-weighted block mean.  Aggregating
    the symmetric conductance gives ``B_bar`` and
    ``A_bar=diag(P^T h)^-1 B_bar``.  Algebraic closure for every micro state is
    equivalent to ``R A=A_bar R``; invariance of block-constant states is
    ``A P=P A_bar``.  For this reversible construction the two conditions
    coincide.  A nonzero defect measures unresolved within-block dynamics and
    prevents promotion to a U5/coarse-graining law.  The returned Boolean uses
    the declared dimensionless relative tolerance separately for each identity:
    every absolute residual is divided by the maximum of one and the norms of
    its two sides.  The residuals and scales carry the quantitative evidence.

    ``partition`` must contain at least two nonempty disjoint blocks, cover each
    graph node exactly once and reduce dimension.  This certificate concerns
    the fixed symmetric pure-EPI channel; it does not coarse-grain phase,
    changing topology, nonlinear operators or REMESH's temporal echo.
    """
    from ..alias import get_attr
    from ..constants.aliases import ALIAS_EPI, ALIAS_VF
    from ._conductance import read_conductance

    _reject_boolean_numeric(tolerance, "tolerance")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    nodes = tuple(graph)
    blocks = tuple(tuple(block) for block in partition)
    if not 2 <= len(blocks) < len(nodes):
        raise ValueError("partition must strictly reduce to at least two blocks")
    if any(not block for block in blocks):
        raise ValueError("partition blocks must be nonempty")
    flattened = tuple(node for block in blocks for node in block)
    if len(flattened) != len(nodes) or set(flattened) != set(nodes):
        raise ValueError("partition must contain every graph node exactly once")

    conductance = read_conductance(graph, list(nodes), symmetric=True)
    adjacency = conductance.dense()
    strength = conductance.strength
    if np.any(strength <= 0.0):
        raise ValueError("EPI coarse-graining requires positive row strength")

    def read_scalar(node, aliases, default: float, name: str) -> float:
        raw = get_attr(
            graph.nodes[node], aliases, default, conv=lambda value: value,
            strict=True,
        )
        _reject_boolean_numeric(raw, f"{name} at node {node!r}")
        try:
            return float(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"EPI coarse-graining requires scalar {name}"
            ) from exc

    frequency = np.array(
        [read_scalar(node, ALIAS_VF, 0.0, "capacity") for node in nodes],
        dtype=float,
    )
    field = np.array(
        [read_scalar(node, ALIAS_EPI, 0.0, "EPI") for node in nodes],
        dtype=float,
    )
    if not np.all(np.isfinite(frequency)) or np.any(frequency <= 0.0):
        raise ValueError("EPI coarse-graining requires positive finite capacity")
    if not np.all(np.isfinite(field)):
        raise ValueError("EPI coarse-graining requires finite scalar EPI")

    node_index = {node: index for index, node in enumerate(nodes)}
    lift = np.zeros((len(nodes), len(blocks)), dtype=float)
    for block_index, block in enumerate(blocks):
        for node in block:
            lift[node_index[node], block_index] = 1.0

    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            metric = strength / frequency
            if not np.all(np.isfinite(metric)) or np.any(metric <= 0.0):
                raise ValueError(
                    "EPI coarse-graining metric exceeds floating-point dynamic range"
                )
            macro_metric = lift.T @ metric
            if (not np.all(np.isfinite(macro_metric))
                    or np.any(macro_metric <= 0.0)):
                raise ValueError(
                    "EPI coarse-graining macro metric exceeds finite "
                    "floating-point range"
                )
            projection = (lift.T * metric[None, :]) / macro_metric[:, None]
            micro_laplacian = np.diag(strength) - adjacency
            micro_generator = (frequency / strength)[:, None] * micro_laplacian
            macro_conductance = lift.T @ adjacency @ lift
            np.fill_diagonal(macro_conductance, 0.0)
            macro_strength = np.sum(macro_conductance, axis=1)
            if np.any(macro_strength <= 0.0):
                raise ValueError(
                    "EPI coarse-graining requires positive macro capacity"
                )
            support = macro_conductance > 0.0
            reached = {0}
            frontier = [0]
            while frontier:
                source = frontier.pop()
                for target in np.flatnonzero(support[source]):
                    index = int(target)
                    if index not in reached:
                        reached.add(index)
                        frontier.append(index)
            if len(reached) != len(blocks):
                raise ValueError(
                    "EPI coarse-graining requires a connected macro quotient"
                )
            macro_laplacian = np.diag(macro_strength) - macro_conductance
            macro_generator = macro_laplacian / macro_metric[:, None]
            macro_frequency = macro_strength / macro_metric
            if np.any(macro_frequency <= 0.0):
                raise ValueError(
                    "EPI coarse-graining requires positive macro capacity"
                )
            macro_epi = projection @ field
    except FloatingPointError as exc:
        raise ValueError(
            "EPI coarse-graining exceeds finite floating-point range"
        ) from exc

    for name, value in (
        ("projection", projection),
        ("micro generator", micro_generator),
        ("macro generator", macro_generator),
        ("macro conductance", macro_conductance),
        ("macro frequency", macro_frequency),
        ("macro EPI", macro_epi),
    ):
        if not np.all(np.isfinite(value)):
            raise ValueError(
                f"EPI coarse-graining {name} exceeds finite floating-point range"
            )

    projected_micro = _finite_product(
        projection, micro_generator, "coarse projection transport"
    )
    macro_projection = _finite_product(
        macro_generator, projection, "coarse macro transport"
    )
    micro_lift = _finite_product(
        micro_generator, lift, "coarse lifted micro transport"
    )
    lifted_macro = _finite_product(
        lift, macro_generator, "coarse lifted macro transport"
    )
    projection_defect = _finite_difference(
        projected_micro, macro_projection, "coarse projection residual"
    )
    lift_defect = _finite_difference(
        micro_lift, lifted_macro, "coarse lift residual"
    )
    projection_residual = _finite_norm(
        projection_defect, matrix=True, name="coarse projection residual"
    )
    lift_residual = _finite_norm(
        lift_defect, matrix=True, name="coarse lift residual"
    )
    projection_scale = max(
        1.0,
        _finite_norm(
            projected_micro, matrix=True, name="coarse projection scale"
        ),
        _finite_norm(
            macro_projection, matrix=True, name="coarse projection scale"
        ),
    )
    lift_scale = max(
        1.0,
        _finite_norm(micro_lift, matrix=True, name="coarse lift scale"),
        _finite_norm(lifted_macro, matrix=True, name="coarse lift scale"),
    )
    relative_projection_residual = projection_residual / projection_scale
    relative_lift_residual = lift_residual / lift_scale
    closure_within_tolerance = (
        relative_projection_residual <= tolerance
        and relative_lift_residual <= tolerance
    )
    morphism = certify_morphism(
        projection,
        micro_generator,
        macro_generator,
        tol=tolerance,
        flow_probe=field,
    )
    return EpiCoarseGrainingCertificate(
        nodes=nodes,
        blocks=blocks,
        projection=projection,
        lift=lift,
        micro_generator=micro_generator,
        macro_generator=macro_generator,
        macro_conductance=macro_conductance,
        macro_frequency=macro_frequency,
        macro_metric_weights=macro_metric,
        macro_epi=macro_epi,
        projection_residual=projection_residual,
        lift_residual=lift_residual,
        information_loss_dimension=len(nodes) - len(blocks),
        nodal_closure_within_tolerance=closure_within_tolerance,
        morphism=morphism,
        scope="fixed symmetric positive-capacity pure-EPI partition quotient",
        projection_residual_scale=projection_scale,
        lift_residual_scale=lift_scale,
        relative_projection_residual=relative_projection_residual,
        relative_lift_residual=relative_lift_residual,
    )


def audit_structural_morphisms() -> list[tuple[str, StructuralMorphismCertificate]]:
    r"""Certify one canonical example of each morphism kind.

    Six kinds emerge from the nodal equation (intertwiners: automorphism,
    relabeling, coarse-graining, lift, conjugation-intertwiner, and the Reynolds
    sector projector); the folding endomorphism does **not**.
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

    # COARSE_GRAINING and LIFT: the p-adic scale adjunction (U5-compatible)
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

    # PROJECTION: the Reynolds sector projector Q_Γ — the emergent case that
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
