r"""Y1 finite structural gauge gap diagnostic.

The routines in this module implement the first TNFR–Yang–Mills milestone:
construct a finite, self-adjoint structural gauge operator from canonical TNFR
telemetry and measure its first spectral gap.

TNFR framing
------------
The diagnostic starts from the nodal structural stack:

    ∂EPI/∂t = νf · ΔNFR(t)

and augments its derived field Ψ = K_φ + i·J_φ with an auxiliary U(1)
coordinate construction. This construction is not a canonical dynamical gauge
sector. No separate quantum substrate is introduced. The external term "mass
gap" is represented here as the separation from the selected matrix baseline
to its next distinct mode. No engine-trajectory theorem identifies those modes
with admissible dynamics or the baseline with a coherent attractor.

Honest scope
------------
This is a finite-graph diagnostic.  It does not prove the Clay Yang–Mills and
Mass Gap theorem, does not introduce a non-Abelian gauge group, and does not
address the continuum / thermodynamic limit.  Those remain YMG-4 and YMG-5 in
``theory/TNFR_YANG_MILLS_RESEARCH_NOTES.md``.

The bundled connection is the difference of vertex phases, hence an exact
one-form with zero cycle holonomy in exact arithmetic. Its computed curvature
sector is a floating-point residual. A supplied connection is accepted only
when it lies on the same gauge orbit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Mapping

try:  # pragma: no cover - imported in tests when optional dependency exists
    import networkx as nx
except ImportError:  # pragma: no cover
    nx = None

from ..constants import inject_defaults
from ..constants.canonical import (
    DELTA_PHI_MAX,
    PHI_S_VON_KOCH_THRESHOLD,
    PI,
    U6_STRUCTURAL_POTENTIAL_LIMIT,
)
from ..mathematics.unified_numerical import np
from ..physics._helpers import wrap_angle
from ..physics.canonical import compute_structural_potential
from ..physics.conservation_gauge_unification import compute_grammar_symmetry_mapping
from ..physics.gauge import (
    GAUGE_CLOSURE_TOLERANCE,
    compute_gauge_connection,
    compute_gauge_coupling_constant,
    compute_gauge_curvature,
    compute_yang_mills_action,
)
from ..rng import validate_seed


@dataclass(frozen=True)
class StructuralGaugeGapOperator:
    """Finite TNFR structural gauge operator.

    Attributes
    ----------
    matrix : numpy.ndarray
        Hermitian diagnostic matrix ``H = L_A + V_F + V_Phi``.
    node_order : tuple[Any, ...]
        Node ordering used for rows and columns.
    connection : dict[tuple, float]
        Gauge connection values used to assemble the covariant Laplacian.
    curvature_potential : dict[Any, float]
        Per-node cycle-closure residual penalty, normalised by π² and zeroed
        within the numerical closure tolerance.
    confinement_potential : dict[Any, float]
        Historical field name for the per-node structural-potential magnitude
        penalty ``Phi_s²/(π/2)²``. It does not evaluate the two-snapshot U6
        drift policy.
    metadata : dict[str, Any]
        Reproducibility and structural telemetry metadata.
    """

    matrix: Any
    node_order: tuple[Any, ...]
    connection: dict[tuple, float]
    curvature_potential: dict[Any, float]
    confinement_potential: dict[Any, float]
    metadata: dict[str, Any]

    @property
    def potential_magnitude_penalty(self) -> dict[Any, float]:
        """Return the accurately named alias for ``confinement_potential``."""

        return self.confinement_potential


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
        Lowest eigenvalue of the selected finite diagnostic matrix.
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

    The generated graph is intentionally modest and phase-compatible: phases
    are clustered within ``phase_spread`` around one base phase, ``ΔNFR`` is
    small, ``frequency`` is positive, and ``EPI`` is initialised. It does not
    contain an executed standalone operator history. The graph is suitable for
    finite spectral diagnostics, not for a continuum claim.

    Parameters
    ----------
    n : int
        Number of nodes for non-grid topologies.  For ``topology='grid'`` the
        size is ``max(2, floor(sqrt(n)))²``; this preserves the historical
        four-node minimum when ``n < 4``.
    topology : str
        ``'cycle'``, ``'complete'``, ``'watts_strogatz'`` or ``'grid'``.
    seed : int
        Reproducibility seed.
    phase_spread : float
        Maximum phase deviation around the base phase. It must lie in
        ``[0, DELTA_PHI_MAX/2]`` so every generated pair is compatible with
        the canonical phase gate.
    delta_nfr_scale : float
        Range scale for small structural pressure values.

    Returns
    -------
    networkx.Graph
        TNFR-ready graph with canonical node attributes.
    """
    if nx is None:  # pragma: no cover
        raise RuntimeError("networkx required for Y1 structural gauge graphs")
    if isinstance(n, bool) or not isinstance(n, Integral):
        raise ValueError("n must be an integer")
    n = int(n)
    if n < 2:
        raise ValueError("Y1 structural gauge graph requires at least 2 nodes")
    if not isinstance(topology, str):
        raise TypeError("topology must be a string")
    seed = validate_seed(seed, allow_none=False)
    phase_spread = _finite_nonnegative(phase_spread, "phase_spread")
    delta_nfr_scale = _finite_nonnegative(delta_nfr_scale, "delta_nfr_scale")
    if phase_spread > float(DELTA_PHI_MAX) / 2.0:
        raise ValueError(
            "phase_spread must not exceed DELTA_PHI_MAX/2 for a "
            "phase-compatible diagnostic graph"
        )

    if topology == "cycle":
        G = nx.cycle_graph(n)
    elif topology == "complete":
        G = nx.complete_graph(n)
    elif topology == "watts_strogatz":
        if n < 3:
            raise ValueError("watts_strogatz topology requires at least 3 nodes")
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
    rng = np.random.default_rng(seed & ((1 << 64) - 1))
    base_phase = float(rng.uniform(0.0, 2.0 * math.pi))
    for node in G.nodes():
        G.nodes[node]["phase"] = float(
            wrap_angle(base_phase + rng.uniform(-phase_spread, phase_spread))
        )
        G.nodes[node]["frequency"] = float(rng.uniform(0.2, 1.0))
        G.nodes[node]["delta_nfr"] = float(
            rng.uniform(-delta_nfr_scale, delta_nfr_scale)
        )
        # Zero is the canonical scalar EPI initialization.  Keeping this
        # assignment deterministic and draw-free preserves the seeded phase,
        # frequency, and pressure streams above.
        G.nodes[node]["EPI"] = 0.0

    G.graph["delta_phi_max"] = float(DELTA_PHI_MAX)
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
    r"""Assemble ``H = L_A + V_F + V_Phi`` on a finite graph.

    Terms
    -----
    ``L_A``
        Gauge-covariant graph Laplacian built from ``A_ij``.
    ``V_F``
        Cycle-closure residual penalty ``F_C² / π²``. It is zero in exact
        arithmetic for the bundled pure-gauge connection and is explicitly
        zeroed within ``GAUGE_CLOSURE_TOLERANCE``.
    ``V_Phi``
        Structural-potential magnitude penalty ``Φ_s² / (π/2)²``. Borrowing
        the U6 drift threshold as a normalization choice neither assesses U6
        nor replaces the separate π/4 per-node magnitude warning.

    The construction is read-only: it does not mutate EPI or any graph
    attribute.
    """
    if G.number_of_nodes() < 2:
        raise ValueError("structural gauge gap requires at least two nodes")
    if G.is_directed() or G.is_multigraph():
        raise ValueError(
            "structural gauge gap requires an undirected simple graph"
        )
    curvature_weight = _finite_nonnegative(curvature_weight, "curvature_weight")
    confinement_weight = _finite_nonnegative(
        confinement_weight,
        "confinement_weight",
    )

    nodes = tuple(G.nodes())
    index = {node: idx for idx, node in enumerate(nodes)}
    n = len(nodes)

    canonical_connection = _normalise_connection(G, compute_gauge_connection(G))
    if connection is None:
        conn = canonical_connection
        connection_source = "canonical_vertex_phase_difference"
        holonomy_deviation = 0.0
    else:
        conn = _normalise_connection(G, connection)
        holonomy_deviation = _maximum_holonomy_deviation(
            G,
            canonical_connection,
            conn,
        )
        if holonomy_deviation > 1e-9:
            raise ValueError(
                "connection must be gauge-equivalent to the canonical "
                "vertex-phase connection"
            )
        connection_source = "supplied_canonical_gauge_orbit"
    matrix = np.zeros((n, n), dtype=complex)

    for u, v in G.edges():
        i = index[u]
        j = index[v]
        weight = _finite_nonnegative(G.edges[u, v].get("weight", 1.0), "edge weight")
        a_uv = conn[(u, v)]
        phase = complex(math.cos(a_uv), math.sin(a_uv))
        matrix[i, i] += weight
        matrix[j, j] += weight
        matrix[i, j] -= weight * phase
        matrix[j, i] -= weight * phase.conjugate()

    curvature = compute_gauge_curvature(G)
    curvature_values = tuple(float(value) for value in curvature.values())
    if not all(math.isfinite(value) for value in curvature_values):
        raise ValueError("gauge curvature must contain only finite values")
    curvature_potential = {node: 0.0 for node in nodes}
    curvature_counts = {node: 0 for node in nodes}
    for cycle, f_c in curvature.items():
        effective_f_c = (
            0.0 if abs(float(f_c)) <= GAUGE_CLOSURE_TOLERANCE else float(f_c)
        )
        f_norm = (effective_f_c / PI) ** 2 if PI else effective_f_c**2
        for node in cycle:
            if node in curvature_potential:
                curvature_potential[node] += f_norm
                curvature_counts[node] += 1
    for node in nodes:
        count = curvature_counts[node]
        if count:
            curvature_potential[node] /= count

    phi_s = compute_structural_potential(G)
    phi_s_values = {node: float(phi_s.get(node, 0.0)) for node in nodes}
    if not all(math.isfinite(value) for value in phi_s_values.values()):
        raise ValueError("structural potential must contain only finite values")
    confinement_potential = {
        node: (abs(phi_s_values[node]) / U6_STRUCTURAL_POTENTIAL_LIMIT) ** 2
        if U6_STRUCTURAL_POTENTIAL_LIMIT
        else 0.0
        for node in nodes
    }
    if not all(math.isfinite(value) for value in confinement_potential.values()):
        raise ValueError("structural-potential penalty exceeds floating-point range")

    for node in nodes:
        diag = (
            curvature_weight * curvature_potential[node]
            + confinement_weight * confinement_potential[node]
        )
        if not math.isfinite(diag):
            raise ValueError("operator diagonal exceeds floating-point range")
        matrix[index[node], index[node]] += float(diag)
    if not bool(np.all(np.isfinite(matrix))):
        raise ValueError("operator entries exceed floating-point range")

    max_abs_phi_s = max(
        (abs(phi_s_values[node]) for node in nodes),
        default=0.0,
    )
    magnitude_warning_ratio = (
        max_abs_phi_s / PHI_S_VON_KOCH_THRESHOLD
        if PHI_S_VON_KOCH_THRESHOLD
        else 0.0
    )
    max_abs_curvature = max((abs(value) for value in curvature_values), default=0.0)
    try:
        grammar = compute_grammar_symmetry_mapping(G)
        applicable_grammar = tuple(item for item in grammar if item.is_applicable)
        grammar_rules_satisfied = sum(
            1 for item in applicable_grammar if item.is_satisfied
        )
        grammar_rules_total = len(applicable_grammar)
        grammar_rules_unassessed = tuple(
            item.rule for item in grammar if not item.is_applicable
        )
    except Exception as exc:  # pragma: no cover - defensive metadata only
        grammar_rules_satisfied = None
        grammar_rules_total = None
        grammar_rules_unassessed = None
        grammar_error = repr(exc)
    else:
        grammar_error = None

    metadata: dict[str, Any] = {
        "operator": "H_structural = L_A + V_F + V_Phi",
        "n_nodes": n,
        "n_edges": G.number_of_edges(),
        "n_cycles": len(curvature),
        "curvature_weight": float(curvature_weight),
        "confinement_weight": float(confinement_weight),
        "yang_mills_action": float(compute_yang_mills_action(G)),
        "gauge_coupling_constant": float(compute_gauge_coupling_constant(G)),
        "connection_source": connection_source,
        "connection_holonomy_deviation": float(holonomy_deviation),
        "canonical_connection_is_pure_gauge": True,
        "canonical_connection_flat_by_construction": True,
        "zero_potential_spectral_baseline": (
            "ordinary_weighted_combinatorial_graph_laplacian"
        ),
        "max_abs_curvature_residual": float(max_abs_curvature),
        "curvature_is_numerical_residual": True,
        "curvature_closure_tolerance": float(GAUGE_CLOSURE_TOLERANCE),
        "curvature_potential_zero_within_tolerance": True,
        "max_abs_phi_s": float(max_abs_phi_s),
        "potential_magnitude_warning_threshold": float(
            PHI_S_VON_KOCH_THRESHOLD
        ),
        "potential_magnitude_ratio": float(magnitude_warning_ratio),
        "potential_magnitude_within_warning": bool(magnitude_warning_ratio < 1.0),
        "legacy_u6_normalization_scale": float(
            U6_STRUCTURAL_POTENTIAL_LIMIT
        ),
        # Kept for consumers of the first scope-correction pass. This is a
        # normalization scale, not a second structural-potential snapshot.
        "potential_magnitude_reference_scale": float(
            U6_STRUCTURAL_POTENTIAL_LIMIT
        ),
        "potential_magnitude_below_pi_scale": bool(
            magnitude_warning_ratio < 1.0
        ),
        "u6_drift_assessed": False,
        "u6_reference_required": True,
        "u6_drift_threshold": float(U6_STRUCTURAL_POTENTIAL_LIMIT),
        "u6_aggregation": "mean_absolute_nodewise_drift",
        "u6_comparison": "strict_less_than",
        "u6_definition": (
            "mean_i |Phi_s_after(i)-Phi_s_before(i)| < threshold"
        ),
        # Compatibility aliases from Y1. They denote the single-snapshot
        # magnitude proxy above and must not be read as a U6 verdict.
        "u6_threshold_phi": float(U6_STRUCTURAL_POTENTIAL_LIMIT),
        "u6_confined": bool(max_abs_phi_s < U6_STRUCTURAL_POTENTIAL_LIMIT),
        "legacy_u6_fields_are_magnitude_proxies": True,
        "grammar_rules_satisfied": grammar_rules_satisfied,
        "grammar_rules_total": grammar_rules_total,
        "grammar_rules_unassessed": grammar_rules_unassessed,
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
    gauge_seed = validate_seed(gauge_seed, allow_none=False)
    tolerance = _finite_positive(tolerance, "tolerance")
    eigen_tolerance = _finite_nonnegative(eigen_tolerance, "eigen_tolerance")
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
    gauge_invariant = gauge_spectral_deviation < tolerance

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
    seed = validate_seed(seed, allow_none=False)
    rng = np.random.default_rng(seed & ((1 << 64) - 1))
    alpha = {node: float(rng.uniform(0.0, 2.0 * math.pi)) for node in nodes}
    transformed: dict[tuple, float] = {}
    for (u, v), a_uv in connection.items():
        transformed[(u, v)] = float(wrap_angle(a_uv + alpha[v] - alpha[u]))
    return transformed


def _finite_nonnegative(value: Any, name: str) -> float:
    """Validate a finite, non-negative real scalar without bool coercion."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _finite_positive(value: Any, name: str) -> float:
    """Validate a finite, strictly positive real scalar."""

    result = _finite_nonnegative(value, name)
    if result == 0.0:
        raise ValueError(f"{name} must be strictly positive")
    return result


def _normalise_connection(
    G: Any,
    connection: Mapping[tuple, float],
) -> dict[tuple, float]:
    """Return a finite antisymmetric connection on exactly the graph edges."""

    if not isinstance(connection, Mapping):
        raise TypeError("connection must be a mapping")
    try:
        raw = dict(connection)
    except (TypeError, ValueError) as exc:
        raise TypeError("connection must be a mapping") from exc
    allowed = {
        oriented
        for u, v in G.edges()
        for oriented in ((u, v), (v, u))
    }
    extras = tuple(key for key in raw if key not in allowed)
    if extras:
        raise ValueError("connection contains keys outside the graph edge set")

    normalised: dict[tuple, float] = {}
    for u, v in G.edges():
        forward_present = (u, v) in raw
        reverse_present = (v, u) in raw
        if forward_present:
            forward = _finite_real(raw[(u, v)], "connection value")
        elif reverse_present:
            forward = -_finite_real(raw[(v, u)], "connection value")
        else:
            forward = 0.0
        if reverse_present:
            reverse = _finite_real(raw[(v, u)], "connection value")
            if abs(float(wrap_angle(forward + reverse))) > 1e-10:
                raise ValueError("connection orientations must be antisymmetric")
        forward = float(wrap_angle(forward))
        normalised[(u, v)] = forward
        normalised[(v, u)] = float(wrap_angle(-forward))
    return normalised


def _finite_real(value: Any, name: str) -> float:
    """Validate a finite real scalar without imposing a sign."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _maximum_holonomy_deviation(
    G: Any,
    reference: Mapping[tuple, float],
    candidate: Mapping[tuple, float],
) -> float:
    """Compare gauge-invariant cycle holonomies for two connections."""

    if nx is None:  # pragma: no cover
        raise RuntimeError("networkx required for connection validation")
    deviations: list[float] = []
    for cycle in nx.cycle_basis(G):
        reference_sum = 0.0
        candidate_sum = 0.0
        for index, source in enumerate(cycle):
            target = cycle[(index + 1) % len(cycle)]
            reference_sum += float(reference[(source, target)])
            candidate_sum += float(candidate[(source, target)])
        deviations.append(
            abs(float(wrap_angle(candidate_sum - reference_sum)))
        )
    return max(deviations, default=0.0)
