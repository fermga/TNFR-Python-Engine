r"""Y1 finite structural gauge gap diagnostic.

The routines in this module implement the first TNFR–Yang–Mills milestone:
construct a finite, self-adjoint structural gauge operator from canonical TNFR
telemetry and measure its first spectral gap.

TNFR framing
------------
The diagnostic is built exclusively from the nodal structural stack:

    ∂EPI/∂t = νf · ΔNFR(t)

and the already-canonical gauge sector Ψ = K_φ + i·J_φ.  No separate quantum
substrate is introduced.  The external term "mass gap" is represented here as
spectral isolation of the first non-trivial admissible nodal reorganisation
mode above the coherent attractor mode.

Honest scope
------------
This is a finite-graph diagnostic.  It does not prove the Clay Yang–Mills and
Mass Gap theorem, does not introduce a non-Abelian gauge group, and does not
address the continuum / thermodynamic limit.  Those remain YMG-4 and YMG-5 in
``theory/TNFR_YANG_MILLS_RESEARCH_NOTES.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

try:  # pragma: no cover - imported in tests when optional dependency exists
    import networkx as nx
except ImportError:  # pragma: no cover
    nx = None

from ..constants import inject_defaults
from ..constants.canonical import DELTA_PHI_MAX, PHI, PI
from ..mathematics.unified_numerical import np
from ..physics._helpers import wrap_angle
from ..physics.canonical import compute_structural_potential
from ..physics.conservation_gauge_unification import (
    compute_grammar_symmetry_mapping,
)
from ..physics.gauge import (
    compute_gauge_connection,
    compute_gauge_coupling_constant,
    compute_gauge_curvature,
    compute_yang_mills_action,
)


@dataclass(frozen=True)
class StructuralGaugeGapOperator:
    """Finite TNFR structural gauge operator.

    Attributes
    ----------
    matrix : numpy.ndarray
        Hermitian matrix ``H_YM^TNFR = L_A + V_F + V_U6``.
    node_order : tuple[Any, ...]
        Node ordering used for rows and columns.
    connection : dict[tuple, float]
        Gauge connection values used to assemble the covariant Laplacian.
    curvature_potential : dict[Any, float]
        Per-node curvature contribution, normalised by π².
    confinement_potential : dict[Any, float]
        Per-node U6 structural-potential contribution, normalised by φ².
    metadata : dict[str, Any]
        Reproducibility and structural telemetry metadata.
    """

    matrix: Any
    node_order: tuple[Any, ...]
    connection: dict[tuple, float]
    curvature_potential: dict[Any, float]
    confinement_potential: dict[Any, float]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class StructuralGaugeGapResult:
    """Y1 finite structural gauge gap report.

    Attributes
    ----------
    operator : StructuralGaugeGapOperator
        Operator used for the spectral analysis.
    eigenvalues : numpy.ndarray
        Sorted real eigenvalues of the Hermitian operator.
    lambda0 : float
        Lowest eigenvalue (coherent attractor baseline for this finite graph).
    lambda1 : float
        First eigenvalue above ``lambda0`` by ``eigen_tolerance`` if present;
        otherwise the second eigenvalue for graphs with at least two nodes.
    gap : float
        ``lambda1 - lambda0``.  Non-negative up to numerical tolerance.
    is_self_adjoint : bool
        Whether ``H = H†`` within tolerance.
    self_adjoint_deviation : float
        Maximum absolute Hermitian defect.
    gauge_invariant : bool
        Whether the spectrum is invariant under the seeded local U(1) gauge
        rotation within tolerance.
    gauge_spectral_deviation : float
        Maximum absolute eigenvalue deviation after the seeded rotation.
    transformed_eigenvalues : numpy.ndarray
        Eigenvalues after the seeded local gauge rotation.
    verdict : str
        Conservative finite-graph classification string.
    metadata : dict[str, Any]
        Combined operator and diagnostic metadata.
    """

    operator: StructuralGaugeGapOperator
    eigenvalues: Any
    lambda0: float
    lambda1: float
    gap: float
    is_self_adjoint: bool
    self_adjoint_deviation: float
    gauge_invariant: bool
    gauge_spectral_deviation: float
    transformed_eigenvalues: Any
    verdict: str
    metadata: dict[str, Any]


def build_structural_gauge_graph(
    n: int = 16,
    *,
    topology: str = "cycle",
    seed: int = 42,
    phase_spread: float = 0.05,
    delta_nfr_scale: float = 0.08,
) -> Any:
    """Build a reproducible TNFR-ready graph for Y1 diagnostics.

    The generated graph is intentionally modest and grammar-friendly: phases
    are clustered within ``phase_spread`` around one base phase, ``ΔNFR`` is
    small, ``frequency`` is positive, and ``EPI`` is initialised.  The graph is
    suitable for finite spectral diagnostics, not for a continuum claim.

    Parameters
    ----------
    n : int
        Number of nodes for non-grid topologies.  For ``topology='grid'`` the
        largest square ``side*side <= n`` is used.
    topology : str
        ``'cycle'``, ``'complete'``, ``'watts_strogatz'`` or ``'grid'``.
    seed : int
        Reproducibility seed.
    phase_spread : float
        Maximum phase deviation around the base phase.  Must be non-negative.
    delta_nfr_scale : float
        Range scale for small structural pressure values.

    Returns
    -------
    networkx.Graph
        TNFR-ready graph with canonical node attributes.
    """
    if nx is None:  # pragma: no cover
        raise RuntimeError("networkx required for Y1 structural gauge graphs")
    if n < 2:
        raise ValueError("Y1 structural gauge graph requires at least 2 nodes")
    if phase_spread < 0:
        raise ValueError("phase_spread must be non-negative")
    if delta_nfr_scale < 0:
        raise ValueError("delta_nfr_scale must be non-negative")

    if topology == "cycle":
        G = nx.cycle_graph(n)
    elif topology == "complete":
        G = nx.complete_graph(n)
    elif topology == "watts_strogatz":
        k = min(4, n - 1)
        if k % 2 == 1:
            k -= 1
        k = max(2, k)
        G = nx.watts_strogatz_graph(n, k, 0.25, seed=seed)
    elif topology == "grid":
        side = max(2, int(math.sqrt(n)))
        G = nx.grid_2d_graph(side, side, periodic=True)
    else:
        raise ValueError(
            "topology must be one of: cycle, complete, watts_strogatz, grid"
        )

    inject_defaults(G)
    rng = np.random.default_rng(seed)
    base_phase = float(rng.uniform(0.0, 2.0 * math.pi))
    for idx, node in enumerate(G.nodes()):
        G.nodes[node]["phase"] = float(
            wrap_angle(base_phase + rng.uniform(-phase_spread, phase_spread))
        )
        G.nodes[node]["frequency"] = float(rng.uniform(0.2, 1.0))
        G.nodes[node]["delta_nfr"] = float(
            rng.uniform(-delta_nfr_scale, delta_nfr_scale)
        )
        G.nodes[node]["EPI"] = f"ymg_epi_{idx}"

    G.graph["delta_phi_max"] = max(float(DELTA_PHI_MAX), phase_spread * 2.0)
    G.graph["tnfr_program"] = "TNFR-Yang-Mills-Y1"
    G.graph["seed"] = seed
    G.graph["topology"] = topology
    return G


def build_structural_gauge_gap_operator(
    G: Any,
    *,
    connection: Mapping[tuple, float] | None = None,
    curvature_weight: float = 1.0,
    confinement_weight: float = 1.0,
) -> StructuralGaugeGapOperator:
    r"""Assemble ``H_YM^TNFR = L_A + V_F + V_U6`` on a finite graph.

    Terms
    -----
    ``L_A``
        Gauge-covariant graph Laplacian built from ``A_ij``.
    ``V_F``
        Gauge-curvature potential from cycle holonomies ``F_C² / π²``.
    ``V_U6``
        Structural-potential confinement contribution ``Φ_s² / φ²``.

    The construction is read-only: it does not mutate EPI or any graph
    attribute.
    """
    if G.number_of_nodes() < 2:
        raise ValueError("structural gauge gap requires at least two nodes")
    if curvature_weight < 0.0 or confinement_weight < 0.0:
        raise ValueError("operator weights must be non-negative")

    nodes = tuple(G.nodes())
    index = {node: idx for idx, node in enumerate(nodes)}
    n = len(nodes)

    conn = (
        dict(connection)
        if connection is not None
        else compute_gauge_connection(G)
    )
    matrix = np.zeros((n, n), dtype=complex)

    for u, v in G.edges():
        i = index[u]
        j = index[v]
        weight = float(G.edges[u, v].get("weight", 1.0))
        if weight < 0.0:
            raise ValueError("edge weights must be non-negative")
        a_uv = float(conn.get((u, v), -conn.get((v, u), 0.0)))
        phase = complex(math.cos(a_uv), math.sin(a_uv))
        matrix[i, i] += weight
        matrix[j, j] += weight
        matrix[i, j] -= weight * phase
        matrix[j, i] -= weight * phase.conjugate()

    curvature = compute_gauge_curvature(G)
    curvature_potential = {node: 0.0 for node in nodes}
    curvature_counts = {node: 0 for node in nodes}
    for cycle, f_c in curvature.items():
        f_norm = (float(f_c) / PI) ** 2 if PI else float(f_c) ** 2
        for node in cycle:
            if node in curvature_potential:
                curvature_potential[node] += f_norm
                curvature_counts[node] += 1
    for node in nodes:
        count = curvature_counts[node]
        if count:
            curvature_potential[node] /= count

    phi_s = compute_structural_potential(G)
    confinement_potential = {
        node: (abs(float(phi_s.get(node, 0.0))) / PHI) ** 2 if PHI else 0.0
        for node in nodes
    }

    for node in nodes:
        diag = (
            curvature_weight * curvature_potential[node]
            + confinement_weight * confinement_potential[node]
        )
        matrix[index[node], index[node]] += float(diag)

    max_abs_phi_s = max(
        (abs(float(phi_s.get(node, 0.0))) for node in nodes),
        default=0.0,
    )
    try:
        grammar = compute_grammar_symmetry_mapping(G)
        grammar_rules_satisfied = sum(
            1 for item in grammar if item.is_satisfied
        )
        grammar_rules_total = len(grammar)
    except Exception as exc:  # pragma: no cover - defensive metadata only
        grammar_rules_satisfied = None
        grammar_rules_total = None
        grammar_error = repr(exc)
    else:
        grammar_error = None

    metadata: dict[str, Any] = {
        "operator": "H_YM_TNFR = L_A + V_F + V_U6",
        "n_nodes": n,
        "n_edges": G.number_of_edges(),
        "n_cycles": len(curvature),
        "curvature_weight": float(curvature_weight),
        "confinement_weight": float(confinement_weight),
        "yang_mills_action": float(compute_yang_mills_action(G)),
        "gauge_coupling_constant": float(compute_gauge_coupling_constant(G)),
        "max_abs_phi_s": float(max_abs_phi_s),
        "u6_threshold_phi": float(PHI),
        "u6_confined": bool(max_abs_phi_s < PHI),
        "grammar_rules_satisfied": grammar_rules_satisfied,
        "grammar_rules_total": grammar_rules_total,
        "grammar_error": grammar_error,
        "scope": "finite_graph_y1_diagnostic_not_clay_proof",
    }

    return StructuralGaugeGapOperator(
        matrix=matrix,
        node_order=nodes,
        connection=conn,
        curvature_potential=curvature_potential,
        confinement_potential=confinement_potential,
        metadata=metadata,
    )


def compute_structural_gauge_gap(
    G: Any,
    *,
    gauge_seed: int = 42,
    tolerance: float = 1e-10,
    eigen_tolerance: float = 1e-9,
    curvature_weight: float = 1.0,
    confinement_weight: float = 1.0,
) -> StructuralGaugeGapResult:
    """Compute the Y1 finite TNFR structural gauge gap.

    The routine assembles the operator, diagonalises it with ``eigvalsh``, and
    verifies spectral invariance under a seeded local U(1) transformation of
    the connection.  The graph is not mutated.
    """
    operator = build_structural_gauge_gap_operator(
        G,
        curvature_weight=curvature_weight,
        confinement_weight=confinement_weight,
    )
    matrix = operator.matrix
    hermitian_defect = matrix - matrix.conjugate().T
    self_adjoint_deviation = (
        float(np.max(np.abs(hermitian_defect))) if matrix.size else 0.0
    )
    is_self_adjoint = self_adjoint_deviation < tolerance

    eigenvalues = np.linalg.eigvalsh(matrix)
    eigenvalues = np.sort(np.real(eigenvalues))
    lambda0 = float(eigenvalues[0])
    lambda1 = _first_excited_eigenvalue(eigenvalues, eigen_tolerance)
    gap = max(0.0, float(lambda1 - lambda0))

    transformed_connection = _seeded_gauge_transformed_connection(
        operator.connection,
        operator.node_order,
        gauge_seed,
    )
    transformed_operator = build_structural_gauge_gap_operator(
        G,
        connection=transformed_connection,
        curvature_weight=curvature_weight,
        confinement_weight=confinement_weight,
    )
    transformed_eigenvalues = np.linalg.eigvalsh(transformed_operator.matrix)
    transformed_eigenvalues = np.sort(np.real(transformed_eigenvalues))
    gauge_spectral_deviation = float(
        np.max(np.abs(eigenvalues - transformed_eigenvalues))
    )
    gauge_invariant = gauge_spectral_deviation < max(tolerance, 1e-9)

    if not is_self_adjoint:
        verdict = "DIAGNOSTIC_FAILED_NON_SELF_ADJOINT"
    elif not gauge_invariant:
        verdict = "DIAGNOSTIC_FAILED_GAUGE_VARIANCE"
    elif gap > eigen_tolerance:
        verdict = "FINITE_POSITIVE_STRUCTURAL_GAP"
    else:
        verdict = "FINITE_GAP_NOT_RESOLVED"

    metadata = dict(operator.metadata)
    metadata.update(
        {
            "gauge_seed": int(gauge_seed),
            "tolerance": float(tolerance),
            "eigen_tolerance": float(eigen_tolerance),
            "lambda0": lambda0,
            "lambda1": float(lambda1),
            "gap": float(gap),
            "gauge_spectral_deviation": gauge_spectral_deviation,
            "verdict": verdict,
        }
    )

    return StructuralGaugeGapResult(
        operator=operator,
        eigenvalues=eigenvalues,
        lambda0=lambda0,
        lambda1=float(lambda1),
        gap=float(gap),
        is_self_adjoint=is_self_adjoint,
        self_adjoint_deviation=self_adjoint_deviation,
        gauge_invariant=gauge_invariant,
        gauge_spectral_deviation=gauge_spectral_deviation,
        transformed_eigenvalues=transformed_eigenvalues,
        verdict=verdict,
        metadata=metadata,
    )


def _first_excited_eigenvalue(eigenvalues: Any, tolerance: float) -> float:
    """Return the first eigenvalue separated from the ground mode."""
    if len(eigenvalues) == 1:
        return float(eigenvalues[0])
    ground = float(eigenvalues[0])
    for val in eigenvalues[1:]:
        val_f = float(val)
        if val_f - ground > tolerance:
            return val_f
    return float(eigenvalues[min(1, len(eigenvalues) - 1)])


def _seeded_gauge_transformed_connection(
    connection: Mapping[tuple, float],
    nodes: tuple[Any, ...],
    seed: int,
) -> dict[tuple, float]:
    """Apply ``A_ij → A_ij + α_j − α_i`` with deterministic α."""
    rng = np.random.default_rng(seed)
    alpha = {node: float(rng.uniform(0.0, 2.0 * math.pi)) for node in nodes}
    transformed: dict[tuple, float] = {}
    for (u, v), a_uv in connection.items():
        transformed[(u, v)] = float(wrap_angle(a_uv + alpha[v] - alpha[u]))
    return transformed
