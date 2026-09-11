r"""Auxiliary U(1) field-coordinate diagnostics for TNFR snapshots.

The complex read-out Ψ = K_φ + i·J_φ admits a nodewise rotation

    Ψ(i) → exp(i α(i)) Ψ(i).

Local quadratic identities then preserve |Ψ|, the corresponding contribution
to the snapshot energy, and the two bilinear norms exposed below.  This is an
algebraic symmetry of the auxiliary field coordinates.  It does not establish
a gauge symmetry of the nodal equation or of the 13 engine operators.

The public connection has a particularly restricted definition:

    A_ij = wrap(arg Ψ(j) - arg Ψ(i)).

It is the wrapped discrete differential of a vertex phase and is therefore a
pure-gauge connection.  Its cycle holonomy F_C vanishes analytically modulo
2π.  ``compute_gauge_curvature`` retains the historical name and reports only
floating-point closure residuals for this connection; those residuals are not
independent curvature, vortices, flux, or confinement.  The legacy
``strong_like`` channel likewise records a closure-residual score and is not a
derived physical interaction sector.

The covariant-difference identities remain useful: if A is reconstructed from
the rotated field, D_ij Ψ transforms by the phase at j and |D_ij Ψ| is
invariant.  For the derived connection its magnitude reduces to the difference
of endpoint magnitudes.  The Yang-Mills-named functions are compatibility
diagnostics evaluated on this constrained pure-gauge surface; they do not vary
an independent edge field or derive engine dynamics.

Status: auxiliary algebraic model with legacy public names.

References
----------
- Nodal equation: ∂EPI/∂t = νf · ΔNFR(t)  [TNFR.pdf §2.1]
- Complex geometric field: src/tnfr/physics/unified.py
- Conservation laws: src/tnfr/physics/conservation.py
- Variational principle: src/tnfr/physics/variational.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Mapping

from ..mathematics.unified_numerical import np
from ..rng import validate_seed

try:
    import networkx as nx
except ImportError:  # pragma: no cover
    nx = None

from ..constants.canonical import PI as PI_CONST
from .canonical import (
    compute_phase_curvature,
    compute_phase_gradient,
    compute_structural_potential,
)
from .extended import compute_dnfr_flux, compute_phase_current
from .unified import (
    compute_complex_geometric_field,
    compute_energy_density,
    compute_symmetry_breaking_field,
)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GaugeSnapshot:
    """Auxiliary U(1) field-coordinate snapshot of a TNFR network.

    All quantities are read-only telemetry (no EPI mutation).

    Attributes
    ----------
    psi : dict[Any, complex]
        Complex geometric field Ψ(i) = K_φ(i) + i·J_φ(i).
    psi_magnitude : dict[Any, float]
        Gauge-invariant magnitude |Ψ(i)|.
    psi_phase : dict[Any, float]
        Gauge-dependent internal phase arg(Ψ(i)).
    connection : dict[tuple, float]
        Pure-gauge connection A_ij = d(arg Ψ)_ij on oriented edges.
    curvature : dict[tuple, float]
        Floating-point cycle-closure residual F_C. Analytically zero.
    energy_density : dict[Any, float]
        Gauge-invariant energy density ℰ(i).
    topological_norm : dict[Any, float]
        Gauge-invariant |𝒯(i)|² = 𝒬² + 𝒬̃².
    chirality_norm : dict[Any, float]
        Gauge-invariant |𝒳(i)|² = χ² + χ̃².
    """

    psi: dict[Any, complex]
    psi_magnitude: dict[Any, float]
    psi_phase: dict[Any, float]
    connection: dict[tuple, float]
    curvature: dict[tuple, float]
    energy_density: dict[Any, float]
    topological_norm: dict[Any, float]
    chirality_norm: dict[Any, float]

    @property
    def canonical_connection_is_pure_gauge(self) -> bool:
        """Whether the bundled connection is a derived exact one-form."""
        return True

    @property
    def curvature_is_numerical_residual(self) -> bool:
        """Whether ``curvature`` contains only cycle-closure residuals."""
        return True


@dataclass(frozen=True)
class GaugeInvarianceResult:
    """Result of a gauge invariance verification test.

    Attributes
    ----------
    is_invariant : bool
        True if all gauge-invariant quantities remain unchanged (within tol).
    energy_max_deviation : float
        Maximum per-node energy density change under gauge transformation.
    magnitude_max_deviation : float
        Maximum per-node |Ψ| change.
    topological_norm_max_deviation : float
        Maximum per-node |𝒯|² change.
    chirality_norm_max_deviation : float
        Maximum per-node |𝒳|² change.
    symmetry_breaking_max_deviation : float
        Maximum per-node 𝒮 change (expected: NOT invariant).
    noether_charge_deviation : float
        Change in the legacy quantity named Noether charge (not invariant).
    coherence_deviation : float
        Change in C(t) (expected: invariant).
    details : dict[str, Any]
        Additional diagnostic information.
    """

    is_invariant: bool
    energy_max_deviation: float
    magnitude_max_deviation: float
    topological_norm_max_deviation: float
    chirality_norm_max_deviation: float
    symmetry_breaking_max_deviation: float
    noether_charge_deviation: float
    coherence_deviation: float
    details: dict[str, Any]


# ---------------------------------------------------------------------------
# Gauge transformation
# ---------------------------------------------------------------------------


def _validated_gauge_angles(
    G: Any,
    alpha: Mapping[Any, Real],
) -> dict[Any, float]:
    """Materialize finite node angles, retaining zero for omitted nodes."""

    if not isinstance(alpha, Mapping):
        raise TypeError("alpha must be a mapping from nodes to finite angles")
    result: dict[Any, float] = {}
    for node in G.nodes():
        value = alpha.get(node, 0.0)
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("alpha values must be finite real numbers")
        angle = float(value)
        if not math.isfinite(angle):
            raise ValueError("alpha values must be finite real numbers")
        result[node] = angle
    return result


def apply_gauge_transformation(
    G: Any,
    alpha: Mapping[Any, Real],
) -> Any:
    """Apply local U(1) gauge transformation Ψ(i) → e^{iα(i)}·Ψ(i).

    This transforms the node-level fields (K_φ, J_φ) by rotation:
        K_φ'(i) = K_φ(i)·cos α(i) − J_φ(i)·sin α(i)
        J_φ'(i) = K_φ(i)·sin α(i) + J_φ(i)·cos α(i)

    **Important**: This is a read-only field transformation for analysis.
    It does NOT modify the graph's EPI or structural state. Instead, it
    returns transformed field values for invariance verification.

    The external phase φ and all non-Ψ fields (Φ_s, |∇φ|, J_ΔNFR, ξ_C)
    are gauge singlets and remain unchanged.

    Parameters
    ----------
    G : TNFRGraph
        Network with node phase/ΔNFR attributes.
    alpha : dict[node, float]
        Local gauge parameter α(i) at each node (radians).

    Returns
    -------
    dict[str, dict[Any, float]]
        Transformed fields: 'k_phi', 'j_phi', 'psi' (complex),
        'psi_magnitude', 'psi_phase'.

    Notes
    -----
    Read-only telemetry operation. Does not mutate EPI or graph state.
    """
    angles = _validated_gauge_angles(G, alpha)
    k_phi = compute_phase_curvature(G)
    j_phi = compute_phase_current(G)

    k_phi_prime: dict[Any, float] = {}
    j_phi_prime: dict[Any, float] = {}
    psi_prime: dict[Any, complex] = {}

    for node in G.nodes():
        a = angles[node]
        cos_a = math.cos(a)
        sin_a = math.sin(a)

        kp = k_phi.get(node, 0.0)
        jp = j_phi.get(node, 0.0)

        k_phi_prime[node] = kp * cos_a - jp * sin_a
        j_phi_prime[node] = kp * sin_a + jp * cos_a
        psi_prime[node] = complex(k_phi_prime[node], j_phi_prime[node])

    return {
        "k_phi": k_phi_prime,
        "j_phi": j_phi_prime,
        "psi": psi_prime,
        "psi_magnitude": {n: abs(v) for n, v in psi_prime.items()},
        "psi_phase": {n: float(np.angle(v)) for n, v in psi_prime.items()},
    }


# ---------------------------------------------------------------------------
# Gauge connection (1-form on edges)
# ---------------------------------------------------------------------------

# Import canonical wrap_angle from shared helpers (single source of truth)
from ._helpers import wrap_angle as _wrap_angle


def compute_gauge_connection(G: Any) -> dict[tuple, float]:
    r"""Compute the pure-gauge connection A_ij on oriented edges.

    The connection is the wrapped exact one-form

        A_ij = arg(Ψ_j) − arg(Ψ_i)  ∈ [−π, π)

    At endpoints where Ψ is nonzero, reconstructing it after
    Ψ → e^{iα}Ψ gives, modulo 2π,

        A_ij → A_ij + α(j) − α(i).

    Because A is derived from vertex phases, it has no independent edge
    degree of freedom and every cycle holonomy is analytically zero.  Nodes
    where |Ψ| is numerically zero use phase zero as a deterministic convention;
    the phase and complex covariance law are undefined there, although the
    covariant-difference magnitude remains the endpoint-amplitude contrast.

    Parameters
    ----------
    G : TNFRGraph

    Returns
    -------
    dict[(i, j), float]
        Connection A_ij for each oriented edge.
        For undirected graphs, both (i,j) and (j,i) are included
        with A_ji = −A_ij (antisymmetry).

    Notes
    -----
    Read-only telemetry. Never mutates EPI.
    """
    psi = compute_complex_geometric_field(G)

    connection: dict[tuple, float] = {}

    for u, v in G.edges():
        psi_u = psi.get(u, complex(0, 0))
        psi_v = psi.get(v, complex(0, 0))

        phase_u = float(np.angle(psi_u)) if abs(psi_u) > 1e-15 else 0.0
        phase_v = float(np.angle(psi_v)) if abs(psi_v) > 1e-15 else 0.0

        a_uv = _wrap_angle(phase_v - phase_u)
        connection[(u, v)] = a_uv

        if not G.is_directed():
            connection[(v, u)] = -a_uv

    return connection


# ---------------------------------------------------------------------------
# Cycle-closure residual of the pure-gauge connection
# ---------------------------------------------------------------------------


# Large enough to absorb a handful of wrapped binary64 angle operations while
# remaining many orders below any TNFR phase scale.  This is a numerical
# closure tolerance, not a physical threshold.
GAUGE_CLOSURE_TOLERANCE = 64.0 * math.ulp(PI_CONST)


def _canonical_cycle(cycle: list[Any], order: dict[Any, int]) -> tuple[Any, ...]:
    """Return a deterministic traversal key without comparing node labels."""
    start = min(range(len(cycle)), key=lambda idx: order[cycle[idx]])
    forward = cycle[start:] + cycle[:start]
    reverse = [forward[0], *reversed(forward[1:])]
    forward_order = tuple(order[node] for node in forward)
    reverse_order = tuple(order[node] for node in reverse)
    return tuple(reverse if reverse_order < forward_order else forward)


def _short_undirected_cycles(G: Any, max_cycle_length: int) -> list[tuple[Any, ...]]:
    """Enumerate all triangles and four-cycles with deterministic keys."""
    nodes = list(G.nodes())
    order = {node: idx for idx, node in enumerate(nodes)}
    adjacency = {node: set(G.neighbors(node)) for node in nodes}
    cycles: set[tuple[Any, ...]] = set()

    for u in nodes:
        for v in adjacency[u]:
            if order[v] <= order[u]:
                continue
            for w in adjacency[u] & adjacency[v]:
                if order[w] <= order[v]:
                    continue
                cycles.add(_canonical_cycle([u, v, w], order))

    if max_cycle_length >= 4 and len(nodes) <= 200:
        for u_index, u in enumerate(nodes):
            for w in nodes[u_index + 1 :]:
                common = sorted(
                    adjacency[u] & adjacency[w], key=order.__getitem__
                )
                for first in range(len(common)):
                    for second in range(first + 1, len(common)):
                        v = common[first]
                        x = common[second]
                        cycles.add(_canonical_cycle([u, v, w, x], order))

    return sorted(cycles, key=lambda cycle: tuple(order[node] for node in cycle))


def _short_directed_cycles(G: Any, max_cycle_length: int) -> list[tuple[Any, ...]]:
    """Enumerate directed triangles and four-cycles without label ordering."""
    nodes = list(G.nodes())
    order = {node: idx for idx, node in enumerate(nodes)}
    length_limit = max(3, min(max_cycle_length, 4))
    cycles: set[tuple[Any, ...]] = set()

    def visit(start: Any, current: Any, path: list[Any], seen: set[Any]) -> None:
        for successor in G.successors(current):
            if successor == start:
                if len(path) >= 3:
                    cycles.add(tuple(path))
                continue
            if (
                len(path) < length_limit
                and successor not in seen
                and order[successor] >= order[start]
            ):
                visit(start, successor, [*path, successor], seen | {successor})

    for start in nodes:
        visit(start, start, [start], {start})

    return sorted(cycles, key=lambda cycle: tuple(order[node] for node in cycle))


def compute_gauge_curvature(
    G: Any,
    *,
    max_cycle_length: int = 6,
) -> dict[tuple, float]:
    r"""Compute cycle-closure residuals of the derived connection.

    The discrete curvature on a cycle C = (v_0, v_1, ..., v_k, v_0) is:

        F_C = Σ_{(i,j) ∈ C} A_ij  (mod 2π, wrapped to [−π, π])

    Since ``A_ij = d(arg Ψ)_ij`` is exact, the sum telescopes and F_C is
    identically zero modulo 2π.  Returned non-zero values measure only
    floating-point wrapping and accumulation error.  They cannot diagnose an
    independent field strength, vortex, flux, or confinement.

    The bounded implementation checks triangles and four-cycles by default.

    Parameters
    ----------
    G : TNFRGraph
    max_cycle_length : int, default=6
        Three checks triangles only. Four and larger values also check
        four-cycles. Values above four are retained for API compatibility;
        this bounded implementation still checks cycles of length at most four.

    Returns
    -------
    dict[tuple_of_nodes, float]
        Wrapped closure residual F_C for each detected cycle.  Tuple order is
        the actual deterministic cycle traversal, including for heterogeneous
        node-label types.

    Notes
    -----
    Read-only.  Four-cycle enumeration is skipped above 200 nodes to retain the
    historical cost bound; triangles are always checked.
    """
    if nx is None:
        raise RuntimeError("networkx required for cycle detection")
    if isinstance(max_cycle_length, bool) or not isinstance(
        max_cycle_length, Integral
    ):
        raise TypeError("max_cycle_length must be an integer of at least 3")
    max_cycle_length = int(max_cycle_length)
    if max_cycle_length < 3:
        raise ValueError("max_cycle_length must be an integer of at least 3")

    connection = compute_gauge_connection(G)
    curvature: dict[tuple, float] = {}
    cycles = (
        _short_directed_cycles(G, max_cycle_length)
        if G.is_directed()
        else _short_undirected_cycles(G, max_cycle_length)
    )
    for cycle in cycles:
        links = [
            connection[(cycle[index], cycle[(index + 1) % len(cycle)])]
            for index in range(len(cycle))
        ]
        curvature[cycle] = _wrap_angle(math.fsum(links))
    return curvature


# ---------------------------------------------------------------------------
# Discrete covariant derivative
# ---------------------------------------------------------------------------


def compute_covariant_derivative(G: Any) -> dict[tuple, complex]:
    r"""Compute the legacy-named covariant difference of Ψ on each edge.

    The covariant derivative along edge (i, j) is:

        D_ij Ψ = Ψ(j) − e^{iA_ij} · Ψ(i)

    where A_ij = arg(Ψ_j) − arg(Ψ_i) is the gauge connection.

    At nonzero field endpoints, under Ψ → e^{iα}Ψ and reconstruction of A:
        D_ij Ψ → e^{iα(j)} · D_ij Ψ    (covariant!)

    Hence **|D_ij Ψ|** is gauge-invariant. At a zero of Ψ, phase is undefined
    and the implementation selects zero; the complex covariance formula then
    depends on that convention, while the magnitude identity below remains
    valid.

    For the bundled exact connection, phase transport cancels identically and
    ``|D_ij Ψ| = ||Ψ(j)| - |Ψ(i)||`` up to floating error.  It therefore reads
    endpoint amplitude contrast.  No engine-operator monotonicity follows from
    this snapshot identity.

    Parameters
    ----------
    G : TNFRGraph

    Returns
    -------
    dict[(i, j), complex]
        Covariant derivative D_ij Ψ for each oriented edge.
    """
    psi = compute_complex_geometric_field(G)
    connection = compute_gauge_connection(G)

    cov_deriv: dict[tuple, complex] = {}

    for u, v in G.edges():
        psi_u = psi.get(u, complex(0, 0))
        psi_v = psi.get(v, complex(0, 0))
        a_uv = connection.get((u, v), 0.0)

        # Parallel transport of Ψ(u) to site v
        psi_u_transported = psi_u * complex(math.cos(a_uv), math.sin(a_uv))

        d_uv = psi_v - psi_u_transported
        cov_deriv[(u, v)] = d_uv

        if not G.is_directed():
            a_vu = connection.get((v, u), 0.0)
            psi_v_transported = psi_v * complex(math.cos(a_vu), math.sin(a_vu))
            cov_deriv[(v, u)] = psi_u - psi_v_transported

    return cov_deriv


def compute_covariant_derivative_magnitude(G: Any) -> dict[tuple, float]:
    """Compute |D_ij Ψ| — gauge-invariant field variation per edge.

    Parameters
    ----------
    G : TNFRGraph

    Returns
    -------
    dict[(i, j), float]
        Gauge-invariant magnitude of covariant derivative per edge.
    """
    cov_deriv = compute_covariant_derivative(G)
    return {edge: abs(val) for edge, val in cov_deriv.items()}


# ---------------------------------------------------------------------------
# Gauge-invariant quantities
# ---------------------------------------------------------------------------


def compute_topological_norm(G: Any) -> dict[Any, float]:
    r"""Compute the gauge-invariant topological norm |𝒯(i)|² per node.

    Defined as:

        |𝒯(i)|² = 𝒬(i)² + 𝒬̃(i)²

    where:
        𝒬  = |∇φ|·J_φ − K_φ·J_ΔNFR   (topological charge)
        𝒬̃ = K_φ·|∇φ| + J_φ·J_ΔNFR    (dual topological charge)

    PROOF OF INVARIANCE:
    Under Ψ → e^{iα}Ψ, the pair (𝒬, 𝒬̃) transforms as a 2D rotation:
        𝒬' =  𝒬·cos α + 𝒬̃·sin α
        𝒬̃' = 𝒬̃·cos α − 𝒬·sin α

    Therefore |𝒯|² = 𝒬² + 𝒬̃² is invariant. ∎

    Geometrically, |𝒯|² = |Ψ|² · |Ω|² where Ω = |∇φ| + i·J_ΔNFR
    is the gradient-flux sector (gauge singlet). This factorisation
    confirms invariance since both factors are individually invariant.

    Parameters
    ----------
    G : TNFRGraph

    Returns
    -------
    dict[node, float]
        |𝒯(i)|² per node.
    """
    grad_phi = compute_phase_gradient(G)
    k_phi = compute_phase_curvature(G)
    j_phi = compute_phase_current(G)
    j_dnfr = compute_dnfr_flux(G)

    result: dict[Any, float] = {}
    for n in G.nodes():
        gp = grad_phi.get(n, 0.0)
        kp = k_phi.get(n, 0.0)
        jp = j_phi.get(n, 0.0)
        jd = j_dnfr.get(n, 0.0)

        # Topological charge
        q = gp * jp - kp * jd
        # Dual topological charge
        q_dual = kp * gp + jp * jd

        result[n] = q * q + q_dual * q_dual

    return result


def compute_chirality_norm(G: Any) -> dict[Any, float]:
    r"""Compute the gauge-invariant chirality norm |𝒳(i)|² per node.

    Defined as:

        |𝒳(i)|² = χ(i)² + χ̃(i)²

    where:
        χ  = |∇φ|·K_φ − J_φ·J_ΔNFR    (chirality)
        χ̃ = |∇φ|·J_φ + K_φ·J_ΔNFR     (dual chirality)

    PROOF OF INVARIANCE:
    Under Ψ → e^{iα}Ψ, the pair (χ, χ̃) rotates by angle α.
    Therefore |𝒳|² = χ² + χ̃² is invariant.

    Factorisation: |𝒳|² = |Ψ|² · |Ω̃|² where Ω̃ = |∇φ| − i·J_ΔNFR
    (parity-reflected gradient sector, also a gauge singlet).

    Parameters
    ----------
    G : TNFRGraph

    Returns
    -------
    dict[node, float]
        |𝒳(i)|² per node.
    """
    grad_phi = compute_phase_gradient(G)
    k_phi = compute_phase_curvature(G)
    j_phi = compute_phase_current(G)
    j_dnfr = compute_dnfr_flux(G)

    result: dict[Any, float] = {}
    for n in G.nodes():
        gp = grad_phi.get(n, 0.0)
        kp = k_phi.get(n, 0.0)
        jp = j_phi.get(n, 0.0)
        jd = j_dnfr.get(n, 0.0)

        # Chirality
        chi = gp * kp - jp * jd
        # Dual chirality
        chi_dual = gp * jp + kp * jd

        result[n] = chi * chi + chi_dual * chi_dual

    return result


def compute_dual_topological_charge(G: Any) -> dict[Any, float]:
    """Compute dual topological charge 𝒬̃ = K_φ·|∇φ| + J_φ·J_ΔNFR.

    The dual charge pairs with 𝒬 to form a gauge doublet.
    Together, 𝒬² + 𝒬̃² = |Ψ|²·|Ω|² is gauge-invariant.

    Parameters
    ----------
    G : TNFRGraph

    Returns
    -------
    dict[node, float]
        𝒬̃(i) per node.
    """
    grad_phi = compute_phase_gradient(G)
    k_phi = compute_phase_curvature(G)
    j_phi = compute_phase_current(G)
    j_dnfr = compute_dnfr_flux(G)

    return {
        n: k_phi.get(n, 0.0) * grad_phi.get(n, 0.0)
        + j_phi.get(n, 0.0) * j_dnfr.get(n, 0.0)
        for n in G.nodes()
    }


def compute_dual_chirality(G: Any) -> dict[Any, float]:
    """Compute dual chirality χ̃ = |∇φ|·J_φ + K_φ·J_ΔNFR.

    The dual chirality pairs with χ to form a gauge doublet.
    Together, χ² + χ̃² = |Ψ|²·|Ω̃|² is gauge-invariant.

    Parameters
    ----------
    G : TNFRGraph

    Returns
    -------
    dict[node, float]
        χ̃(i) per node.
    """
    grad_phi = compute_phase_gradient(G)
    k_phi = compute_phase_curvature(G)
    j_phi = compute_phase_current(G)
    j_dnfr = compute_dnfr_flux(G)

    return {
        n: grad_phi.get(n, 0.0) * j_phi.get(n, 0.0)
        + k_phi.get(n, 0.0) * j_dnfr.get(n, 0.0)
        for n in G.nodes()
    }


# ---------------------------------------------------------------------------
# Gauge invariance verification
# ---------------------------------------------------------------------------


def verify_gauge_invariance(
    G: Any,
    alpha: Mapping[Any, Real] | None = None,
    *,
    tolerance: float = 1e-10,
    seed: int | None = None,
) -> GaugeInvarianceResult:
    r"""Verify auxiliary rotation identities under Ψ → e^{iα}Ψ.

    Applies a local gauge transformation and checks that all
    gauge-invariant quantities remain unchanged within tolerance.

    Tests:
    1. ℰ(i) invariance (energy density)
    2. |Ψ(i)| invariance (field magnitude)
    3. |𝒯(i)|² invariance (topological norm)
    4. |𝒳(i)|² invariance (chirality norm)
    5. C(t) invariance (global coherence — external to Ψ)

    Also measures (expected non-invariant):
    6. ΔQ = change in Noether charge Q (expected ≠ 0 for non-trivial α)
    7. Δ𝒮 = change in symmetry breaking (NOT invariant: K_φ², J_φ²
       individually change under rotation)

    Parameters
    ----------
    G : TNFRGraph
    alpha : dict[node, float], optional
        Gauge parameters. If None, random α ∈ [0, 2π) is generated
        for each node using the given seed.
    tolerance : float, default=1e-10
        Maximum allowed deviation for "invariant" classification.
    seed : int, optional
        Random seed for reproducible α generation.

    Returns
    -------
    GaugeInvarianceResult
        Comprehensive invariance diagnostic.
    """
    if isinstance(tolerance, bool) or not isinstance(tolerance, Real):
        raise TypeError("tolerance must be a finite positive real number")
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be a finite positive real number")

    nodes = list(G.nodes())

    if alpha is None:
        validated_seed = validate_seed(
            42 if seed is None else seed,
            allow_none=False,
        )
        rng = np.random.default_rng(validated_seed & ((1 << 64) - 1))
        angles = {n: float(rng.uniform(0, 2 * math.pi)) for n in nodes}
    else:
        angles = _validated_gauge_angles(G, alpha)

    # --- Before transformation ---
    psi_before = compute_complex_geometric_field(G)
    energy_before = compute_energy_density(G)
    topo_norm_before = compute_topological_norm(G)
    chiral_norm_before = compute_chirality_norm(G)
    symbreak_before = compute_symmetry_breaking_field(G)

    # Noether charge (NOT expected to be invariant)
    k_phi_before = compute_phase_curvature(G)
    phi_s = compute_structural_potential(G)
    q_before = sum(phi_s.get(n, 0.0) + k_phi_before.get(n, 0.0) for n in nodes)

    # Canonical C(t) depends on pressure and EPI rate, not the internal Ψ angle.
    from ..metrics.common import compute_coherence

    c_before = compute_coherence(G)

    # --- Apply gauge transformation ---
    transformed = apply_gauge_transformation(G, angles)
    k_phi_after = transformed["k_phi"]
    j_phi_after = transformed["j_phi"]

    # --- Recompute invariants with transformed fields ---
    grad_phi = compute_phase_gradient(G)
    j_dnfr = compute_dnfr_flux(G)

    energy_after: dict[Any, float] = {}
    topo_norm_after: dict[Any, float] = {}
    chiral_norm_after: dict[Any, float] = {}
    symbreak_after: dict[Any, float] = {}

    for n in nodes:
        ps = phi_s.get(n, 0.0)
        gp = grad_phi.get(n, 0.0)
        kp = k_phi_after.get(n, 0.0)
        jp = j_phi_after.get(n, 0.0)
        jd = j_dnfr.get(n, 0.0)

        # Energy density (should be invariant)
        energy_after[n] = ps**2 + gp**2 + kp**2 + jp**2 + jd**2

        # Topological norm (should be invariant)
        q = gp * jp - kp * jd
        q_dual = kp * gp + jp * jd
        topo_norm_after[n] = q * q + q_dual * q_dual

        # Chirality norm (should be invariant)
        chi = gp * kp - jp * jd
        chi_dual = gp * jp + kp * jd
        chiral_norm_after[n] = chi * chi + chi_dual * chi_dual

        # Symmetry breaking is variant because its two Ψ terms have
        # different signs even though K_φ² + J_φ² is unchanged.
        symbreak_after[n] = (gp**2 - kp**2) + (jp**2 - jd**2)

    # Noether charge after (NOT expected invariant)
    q_after = sum(phi_s.get(n, 0.0) + k_phi_after.get(n, 0.0) for n in nodes)

    # C(t) unchanged because the field rotation changes neither pressure nor EPI rate
    c_after = c_before  # By construction, external fields unchanged

    # --- Compute deviations ---
    energy_devs = [abs(energy_after[n] - energy_before[n]) for n in nodes]
    mag_devs = [abs(abs(transformed["psi"][n]) - abs(psi_before[n])) for n in nodes]
    topo_devs = [abs(topo_norm_after[n] - topo_norm_before[n]) for n in nodes]
    chiral_devs = [abs(chiral_norm_after[n] - chiral_norm_before[n]) for n in nodes]
    symbreak_devs = [abs(symbreak_after[n] - symbreak_before[n]) for n in nodes]

    energy_max = max(energy_devs) if energy_devs else 0.0
    mag_max = max(mag_devs) if mag_devs else 0.0
    topo_max = max(topo_devs) if topo_devs else 0.0
    chiral_max = max(chiral_devs) if chiral_devs else 0.0
    symbreak_max = max(symbreak_devs) if symbreak_devs else 0.0

    delta_q = abs(q_after - q_before)
    delta_c = abs(c_after - c_before)

    # All gauge-invariant quantities within tolerance
    # Note: 𝒮 (symmetry breaking) is NOT gauge-invariant because
    # K_φ² and J_φ² individually change under rotation, even though
    # K_φ² + J_φ² = |Ψ|² is preserved.
    all_invariant = (
        energy_max < tolerance
        and mag_max < tolerance
        and topo_max < tolerance
        and chiral_max < tolerance
        and delta_c < tolerance
    )

    # Determine if alpha is non-trivial (at least some nodes have α ≠ 0)
    has_nontrivial_alpha = any(abs(a) > 1e-10 for a in angles.values())

    details: dict[str, Any] = {
        "num_nodes": len(nodes),
        "alpha_range": (
            (min(angles.values()), max(angles.values()))
            if angles
            else (0.0, 0.0)
        ),
        "has_nontrivial_alpha": has_nontrivial_alpha,
        "noether_charge_before": q_before,
        "noether_charge_after": q_after,
        "noether_charge_expected_variant": has_nontrivial_alpha,
        "energy_rms_deviation": (
            float(np.sqrt(np.mean(np.array(energy_devs) ** 2))) if energy_devs else 0.0
        ),
    }

    return GaugeInvarianceResult(
        is_invariant=all_invariant,
        energy_max_deviation=float(energy_max),
        magnitude_max_deviation=float(mag_max),
        topological_norm_max_deviation=float(topo_max),
        chirality_norm_max_deviation=float(chiral_max),
        symmetry_breaking_max_deviation=float(symbreak_max),
        noether_charge_deviation=float(delta_q),
        coherence_deviation=float(delta_c),
        details=details,
    )


# ---------------------------------------------------------------------------
# Comprehensive gauge snapshot
# ---------------------------------------------------------------------------


def capture_gauge_snapshot(G: Any) -> GaugeSnapshot:
    """Capture the auxiliary U(1) field-coordinate diagnostics.

    Computes the field coordinates, derived connection, cycle residuals, and
    algebraic invariants in a single pass. Read-only telemetry.

    Parameters
    ----------
    G : TNFRGraph

    Returns
    -------
    GaugeSnapshot
    """
    psi = compute_complex_geometric_field(G)
    connection = compute_gauge_connection(G)
    curvature = compute_gauge_curvature(G)
    energy = compute_energy_density(G)
    topo_norm = compute_topological_norm(G)
    chiral_norm = compute_chirality_norm(G)

    return GaugeSnapshot(
        psi=psi,
        psi_magnitude={n: abs(v) for n, v in psi.items()},
        psi_phase={n: float(np.angle(v)) for n, v in psi.items()},
        connection=connection,
        curvature=curvature,
        energy_density=energy,
        topological_norm=topo_norm,
        chirality_norm=chiral_norm,
    )


# ---------------------------------------------------------------------------
# Interaction regime classification
# ---------------------------------------------------------------------------


def classify_interaction_regime(
    G: Any,
    node: Any,
) -> dict[str, Any]:
    r"""Compute the historical four-label field-coordinate heuristic.

    The labels are retained for API compatibility.  They do not identify
    fundamental interactions and are not invariant under arbitrary local
    rotations of Ψ:

    1. **em_like**: arg(Ψ) ≈ 0 or π — geometric-dominant (K_φ ≫ |J_φ|).
       Nearly real Ψ in the selected auxiliary coordinate frame.

    2. **weak_like**: arg(Ψ) ≈ ±π/2 — transport-dominant (|J_φ| ≫ K_φ).
       Nearly imaginary Ψ in the selected auxiliary coordinate frame.

    3. **strong_like**: Large numerical cycle-closure residual.  The canonical
       connection is pure gauge, so this channel should be zero within
       ``GAUGE_CLOSURE_TOLERANCE`` and cannot represent confinement.

    4. **gravity_like**: Φ_s ≫ |Ψ| — the structural potential dominates.
       Structural-potential magnitude dominates the Ψ magnitude.

    Parameters
    ----------
    G : TNFRGraph
    node : Any
        Node to classify.

    Returns
    -------
    dict[str, Any]
        - regime: str — dominant interaction type
        - psi_phase: float — arg(Ψ) at node
        - psi_magnitude: float — |Ψ| at node
        - phi_s: float — structural potential at node
        - mean_curvature: float — mean |F_C| of adjacent plaquettes
        - regime_scores: dict[str, float] — score for each regime
    """
    psi = compute_complex_geometric_field(G)
    phi_s = compute_structural_potential(G)
    curvature = compute_gauge_curvature(G)

    psi_val = psi.get(node, complex(0, 0))
    psi_mag = abs(psi_val)
    psi_phase = float(np.angle(psi_val))
    ps = abs(phi_s.get(node, 0.0))

    # Mean gauge curvature of plaquettes containing this node
    adjacent_curvatures = [abs(f) for cycle, f in curvature.items() if node in cycle]
    mean_curv = float(np.mean(adjacent_curvatures)) if adjacent_curvatures else 0.0

    # Historical heuristic decomposition.  Suppress binary64 closure noise so
    # the legacy strong_like slot cannot become active on an exact connection.
    if mean_curv <= GAUGE_CLOSURE_TOLERANCE:
        mean_curv = 0.0
    total = ps + psi_mag + mean_curv + 1e-15

    # em_like: K_φ-dominant → |cos(arg Ψ)| near 1
    em_score = abs(math.cos(psi_phase)) * psi_mag / total if psi_mag > 1e-15 else 0.0

    # weak_like: J_φ-dominant → |sin(arg Ψ)| near 1
    weak_score = abs(math.sin(psi_phase)) * psi_mag / total if psi_mag > 1e-15 else 0.0

    # strong_like: anomalous closure residual, not a physical interaction
    strong_score = mean_curv / total

    # gravity_like: Φ_s-dominant
    gravity_score = ps / total

    scores = {
        "em_like": float(em_score),
        "weak_like": float(weak_score),
        "strong_like": float(strong_score),
        "gravity_like": float(gravity_score),
    }

    dominant = max(scores, key=scores.get)  # type: ignore

    return {
        "regime": dominant,
        "psi_phase": float(psi_phase),
        "psi_magnitude": float(psi_mag),
        "phi_s": float(ps),
        "mean_curvature": float(mean_curv),
        "regime_scores": scores,
    }


def classify_network_regimes(G: Any) -> dict[str, Any]:
    """Aggregate the historical four-label snapshot heuristic.

    Returns
    -------
    dict[str, Any]
        - per_node: dict[node, dict] — regime classification per node
        - regime_distribution: dict[str, int] — count of each regime
        - dominant_regime: str — most common regime
        - mean_gauge_curvature: float — mean cycle-closure residual
        - gauge_flatness: float — fraction of plaquettes with |F_C| < π/10
    """
    nodes = list(G.nodes())
    per_node = {n: classify_interaction_regime(G, n) for n in nodes}

    # Distribution
    regime_counts: dict[str, int] = {
        "em_like": 0,
        "weak_like": 0,
        "strong_like": 0,
        "gravity_like": 0,
    }
    for info in per_node.values():
        regime_counts[info["regime"]] = regime_counts.get(info["regime"], 0) + 1

    dominant = max(regime_counts, key=regime_counts.get)  # type: ignore

    # Global gauge curvature statistics
    curvature = compute_gauge_curvature(G)
    curv_values = [abs(f) for f in curvature.values()] if curvature else [0.0]
    mean_curv = float(np.mean(curv_values))
    flat_fraction = float(
        np.mean([1.0 if c < math.pi / 10 else 0.0 for c in curv_values])
    )

    return {
        "per_node": per_node,
        "regime_distribution": regime_counts,
        "dominant_regime": dominant,
        "mean_gauge_curvature": mean_curv,
        "gauge_flatness": flat_fraction,
    }


# ---------------------------------------------------------------------------
# Yang-Mills-like action on graph
# ---------------------------------------------------------------------------


def compute_yang_mills_action(G: Any) -> float:
    r"""Compute the legacy quadratic cycle-closure diagnostic.

    The Yang-Mills action on a graph with plaquettes {C} is:

        S_YM = (1/2) Σ_C F_C²

    where F_C is the floating-point closure residual on cycle C.

    The canonical A=d(arg Ψ) is pure gauge, so this quantity is analytically
    zero.  The historical function name is retained; a nonzero return measures
    numerical closure error rather than Yang-Mills field energy.

    Parameters
    ----------
    G : TNFRGraph

    Returns
    -------
    float
        Yang-Mills action (non-negative).
    """
    curvature = compute_gauge_curvature(G)
    if not curvature:
        return 0.0
    return 0.5 * sum(f * f for f in curvature.values())


def compute_gauge_energy_decomposition(G: Any) -> dict[str, float]:
    r"""Decompose snapshot energy into legacy field-coordinate sectors.

    The energy density ℰ = Φ_s² + |∇φ|² + |Ψ|² + J_ΔNFR² can be
    decomposed into sectors:

    1. **Potential sector**: Φ_s² — long-range structural potential
    2. **Gradient sector**: |∇φ|² — local phase stress
    3. **Gauge sector**: |Ψ|² = K_φ² + J_φ² — geometric-transport energy
    4. **Flux sector**: J_ΔNFR² — reorganisation transport
    5. **Legacy Yang-Mills diagnostic**: squared cycle-closure residual

    The returned ``yang_mills_action`` is not included in ``total_energy`` and
    is analytically zero for the derived connection.

    Parameters
    ----------
    G : TNFRGraph

    Returns
    -------
    dict[str, float]
        Energy contribution from each sector (summed over nodes).
    """
    phi_s = compute_structural_potential(G)
    grad_phi = compute_phase_gradient(G)
    k_phi = compute_phase_curvature(G)
    j_phi = compute_phase_current(G)
    j_dnfr = compute_dnfr_flux(G)

    e_potential = sum(v**2 for v in phi_s.values())
    e_gradient = sum(v**2 for v in grad_phi.values())
    e_gauge = sum(k_phi.get(n, 0.0) ** 2 + j_phi.get(n, 0.0) ** 2 for n in G.nodes())
    e_flux = sum(v**2 for v in j_dnfr.values())
    e_ym = compute_yang_mills_action(G)

    total = e_potential + e_gradient + e_gauge + e_flux

    return {
        "potential_sector": float(e_potential),
        "gradient_sector": float(e_gradient),
        "gauge_sector": float(e_gauge),
        "flux_sector": float(e_flux),
        "yang_mills_action": float(e_ym),
        "total_energy": float(0.5 * total),
        "potential_fraction": float(e_potential / (total + 1e-15)),
        "gradient_fraction": float(e_gradient / (total + 1e-15)),
        "gauge_fraction": float(e_gauge / (total + 1e-15)),
        "flux_fraction": float(e_flux / (total + 1e-15)),
    }


# =========================================================================
# LEGACY YANG-MILLS-NAMED DIAGNOSTICS ON THE PURE-GAUGE SURFACE
# =========================================================================
#
# These APIs evaluate expressions borrowed from a lattice U(1) action:
#
#   S[A, Ψ] = S_YM + S_matter
#            = (1/2g²) Σ_P F_P²  +  Σ_{(i,j)} |D_ij Ψ|²
#
# where g is the gauge coupling constant, F_P is the plaquette curvature,
# D_ij is the covariant derivative, and the sum runs over all plaquettes P
# and edges (i,j).
#
# A genuine Euler-Lagrange equation δS/δA_ij = 0 requires A to be an
# independently varied edge field.  This module instead fixes
# A=d(arg Ψ), so F=0 analytically and no such dynamical derivation follows.
# The residual function keeps the historical expression
#
#   (1/g²) Σ_{P ∋ (i,j)} ε_P(i,j) · sin(F_P) = J_matter(i,j)
#
# where ε_P(i,j) = ±1 is the orientation of edge (i,j) within plaquette P,
# and J_matter is the matter current:
#
#   J_matter(i,j) = Im[ Ψ*(j) · e^{+iA_ij} · Ψ(i) ]
#
# as a finite-snapshot consistency diagnostic.  Its legacy field names do not
# assert a Yang-Mills sector, a Ward identity, or an operator derivation.
# =========================================================================


@dataclass(frozen=True)
class YangMillsFieldEquations:
    r"""Legacy-named pure-gauge consistency diagnostics.

    The fields evaluate the historical lattice-action formulas after imposing
    A=d(arg Ψ).  They are not Euler-Lagrange equations of the TNFR engine,
    because the implementation does not vary A independently.

    Attributes
    ----------
    matter_current : dict[tuple, float]
        J_matter(i,j) = Im[Ψ*(j) · e^{+iA_ij} · Ψ(i)] per oriented edge.
    gauge_divergence : dict[tuple, float]
        (1/g²) Σ_{P ∋ (i,j)} ε_P · sin(F_P) per edge (LHS of field eqn).
    equation_residual : dict[tuple, float]
        |gauge_divergence − J_matter| per edge.  A numerical consistency
        residual, not an engine equation-of-motion residual.
    yang_mills_action : float
        S_YM = (1/2g²) Σ_P F_P².
    matter_action : float
        S_matter = Σ_{(i,j)} |D_ij Ψ|².
    total_action : float
        S_YM + S_matter.
    coupling_constant : float
        User-supplied scale or mean squared closure residual.  The derived
        connection gives zero analytically.
    mean_residual : float
        Mean equation residual across all edges.
    max_residual : float
        Maximum equation residual (worst-case violation).
    """

    matter_current: dict[tuple, float]
    gauge_divergence: dict[tuple, float]
    equation_residual: dict[tuple, float]
    yang_mills_action: float
    matter_action: float
    total_action: float
    coupling_constant: float
    mean_residual: float
    max_residual: float

    @property
    def canonical_connection_is_pure_gauge(self) -> bool:
        """Whether the evaluated connection lacks independent edge freedom."""
        return True

    @property
    def is_dynamical_derivation(self) -> bool:
        """Whether this snapshot evaluation derives TNFR dynamics."""
        return False


@dataclass(frozen=True)
class BianchiIdentityResult:
    r"""Verification of cycle closure for the exact derived connection.

    The public class name is retained for compatibility.  On a graph, this
    implementation checks F_C=0 for every enumerated cycle of A=d(arg Ψ); it
    does not construct higher-dimensional cells on which a general dF could be
    evaluated.

    Attributes
    ----------
    is_satisfied : bool
        True if max_residual <= tolerance.
    max_residual : float
        Maximum absolute cycle-closure residual.
    mean_residual : float
        Mean absolute cycle-closure residual.
    num_coboundaries_tested : int
        Number of cycles checked (legacy field name).
    """

    is_satisfied: bool
    max_residual: float
    mean_residual: float
    num_coboundaries_tested: int

    @property
    def num_cycles_tested(self) -> int:
        """Number of cycle-closure relations checked."""
        return self.num_coboundaries_tested


# ---------------------------------------------------------------------------
# Legacy regime-score activity convention
# ---------------------------------------------------------------------------
# The four historical labels are normalised to a unit score budget.  A score
# above the equal-share reference 1/4 is marked active.  This is a reporting
# convention, not a derived TNFR threshold or a map to fundamental forces.
N_REGIMES = 4
REGIME_ACTIVITY_SHARE = 1.0 / N_REGIMES  # equipartition reference = 0.25


@dataclass(frozen=True)
class InteractionRegimeMetrics:
    r"""Per-node values for the historical four-label heuristic.

    The names and fields are retained for compatibility.  ``em_like`` and
    ``weak_like`` depend on the chosen Ψ coordinate angle, ``strong_like`` is
    a numerical closure-residual slot, and ``gravity_like`` measures potential
    dominance.  They are not four derived interactions.

    O_em  = |cos(arg Ψ)| — geometric (K_φ) dominance fraction
    O_wk  = |sin(arg Ψ)| — transport (J_φ) dominance fraction
    O_st  = ⟨|F_C|⟩ / π  — normalised numerical closure residual
    O_gr  = Φ_s² / (Φ_s² + |Ψ|²) — potential dominance fraction

    Activity criterion (uniform across sectors):
    - a sector is active ⟺ regime_scores[sector] > 1/N_REGIMES = 0.25.

    Attributes
    ----------
    node : Any
        Node identifier.
    em_order_parameter : float
        O_em = |cos(arg Ψ)| in the selected auxiliary frame.
    weak_order_parameter : float
        O_wk = |sin(arg Ψ)| in the selected auxiliary frame.
    strong_order_parameter : float
        O_st = ⟨|F_C|⟩ / π.  Values above numerical tolerance indicate
        failure of pure-gauge cycle closure, not confinement.
    gravity_order_parameter : float
        O_gr = Φ_s² / (Φ_s² + |Ψ|²). High → potential-dominant.
    dominant_regime : str
        One of 'em_like', 'weak_like', 'strong_like', 'gravity_like'.
    regime_scores : dict[str, float]
        Normalised scores for each regime (sum ≈ 1).
    above_threshold : dict[str, bool]
        Whether each regime's normalised score exceeds the equipartition
        share 1/N_REGIMES = 0.25 (legacy reporting convention).
    mixing_angle : float
        arg(Ψ) in radians — the gauge-dependent mixing angle between
        geometric (K_φ) and transport (J_φ) sectors.
    """

    node: Any
    em_order_parameter: float
    weak_order_parameter: float
    strong_order_parameter: float
    gravity_order_parameter: float
    dominant_regime: str
    regime_scores: dict[str, float]
    above_threshold: dict[str, bool]
    mixing_angle: float


@dataclass(frozen=True)
class NetworkInteractionProfile:
    r"""Network aggregation of the historical four-label heuristic.

    Attributes
    ----------
    per_node : dict[Any, InteractionRegimeMetrics]
        Formal regime metrics per node.
    regime_distribution : dict[str, int]
        Count of nodes in each regime.
    regime_fractions : dict[str, float]
        Fraction of nodes in each regime.
    mean_order_parameters : dict[str, float]
        Network-averaged order parameters.
    dominant_regime : str
        Most common regime across the network.
    mixing_entropy : float
        Shannon entropy H = −Σ p·ln(p) of the regime distribution.
        H = 0 → pure single regime; H = ln(4) ≈ 1.386 → uniform mixing.
    gauge_coupling_constant : float
        Mean squared numerical cycle-closure residual (legacy name).
    yang_mills_action : float
        Quadratic cycle-closure residual (legacy name).
    mean_curvature : float
        Network mean absolute cycle-closure residual.
    gauge_flatness : float
        Fraction of plaquettes with |F_C| < π/10.
    """

    per_node: dict[Any, InteractionRegimeMetrics]
    regime_distribution: dict[str, int]
    regime_fractions: dict[str, float]
    mean_order_parameters: dict[str, float]
    dominant_regime: str
    mixing_entropy: float
    gauge_coupling_constant: float
    yang_mills_action: float
    mean_curvature: float
    gauge_flatness: float


# ---------------------------------------------------------------------------
# Legacy gauge-invariant link-current expression
# ---------------------------------------------------------------------------


def compute_matter_current(G: Any) -> dict[tuple, float]:
    r"""Compute the legacy-named invariant link current on each edge.

    With the transport convention used by ``compute_covariant_derivative``,
    the invariant bilinear is

        J_matter(i,j) = Im[ Ψ*(j) · e^{+iA_ij} · Ψ(i) ].

    Under gauge transformation Ψ → e^{iα}Ψ, the current transforms as:
        J_matter → J_matter    (gauge-invariant!)

    This is because:
        Im[e^{-iα(j)} Ψ*(j) · e^{i(A_ij + α(j) - α(i))} · e^{iα(i)} Ψ(i)]
        = Im[Ψ*(j) · e^{iA_ij} · Ψ(i)]

    For A=d(arg Ψ), the transported endpoint phases align and this imaginary
    part is analytically zero.  Returned nonzero values are floating-point
    residuals; this function does not establish a Noether or Ward current for
    the nodal dynamics.

    Parameters
    ----------
    G : TNFRGraph

    Returns
    -------
    dict[(i, j), float]
        Legacy link-current residual. Antisymmetric on undirected graphs.
    """
    psi = compute_complex_geometric_field(G)
    connection = compute_gauge_connection(G)

    current: dict[tuple, float] = {}

    for u, v in G.edges():
        psi_u = psi.get(u, complex(0, 0))
        psi_v = psi.get(v, complex(0, 0))
        a_uv = connection.get((u, v), 0.0)

        # The +A sign is required by A' = A + alpha(v) - alpha(u).
        transport = (
            psi_v.conjugate()
            * complex(math.cos(a_uv), math.sin(a_uv))
            * psi_u
        )
        current[(u, v)] = transport.imag

        if not G.is_directed():
            current[(v, u)] = -transport.imag  # antisymmetric

    return current


# ---------------------------------------------------------------------------
# Legacy Yang-Mills-named snapshot residual
# ---------------------------------------------------------------------------


def compute_yang_mills_equations(
    G: Any,
    *,
    coupling: float | None = None,
) -> YangMillsFieldEquations:
    r"""Evaluate the historical lattice-action residual on a TNFR snapshot.

    The returned fields preserve the public API for the expression

        (1/g²) Σ_{P ∋ (i,j)} ε_P(i,j) · sin(F_P) = J_matter(i,j)

    but the canonical connection is constrained to A=d(arg Ψ), not varied
    independently.  Consequently F_P and the invariant link current vanish
    analytically.  This routine is a numerical consistency check and does not
    derive a Maxwell/Yang-Mills equation or TNFR operator dynamics.

    Parameters
    ----------
    G : TNFRGraph
    coupling : float, optional
        Non-negative diagnostic scale.  If None, use the mean squared closure
        residual after suppressing binary64 noise.

    Returns
    -------
    YangMillsFieldEquations
    """
    curvature = compute_gauge_curvature(G)
    j_matter = compute_matter_current(G)
    cov_deriv = compute_covariant_derivative(G)

    effective_curvature = {
        cycle: 0.0 if abs(value) <= GAUGE_CLOSURE_TOLERANCE else value
        for cycle, value in curvature.items()
    }
    curv_values = list(effective_curvature.values())
    if coupling is None:
        g_sq = float(np.mean(np.array(curv_values) ** 2)) if curv_values else 0.0
    else:
        if isinstance(coupling, bool) or not isinstance(coupling, Real):
            raise TypeError("coupling must be a finite non-negative real number")
        g_sq = float(coupling)
        if not math.isfinite(g_sq) or g_sq < 0.0:
            raise ValueError("coupling must be a finite non-negative real number")
    if g_sq == 0.0 and any(value != 0.0 for value in curv_values):
        raise ValueError("zero coupling is undefined for a nonzero closure residual")

    squared_closure = sum(value * value for value in curv_values)
    s_ym = 0.5 * squared_closure / g_sq if g_sq > 0.0 else 0.0

    # --- Matter action ---
    s_matter = sum(abs(d) ** 2 for d in cov_deriv.values())

    # --- Build plaquette-to-edge incidence for gauge divergence ---
    # For each oriented edge (u,v), collect plaquettes containing it
    edge_plaquettes: dict[tuple, list[tuple[tuple, float]]] = {}
    for cycle_key, f_c in curvature.items():
        cycle_nodes = list(cycle_key)
        n_cycle = len(cycle_nodes)
        for idx in range(n_cycle):
            u_c = cycle_nodes[idx]
            v_c = cycle_nodes[(idx + 1) % n_cycle]
            # Forward orientation: edge (u_c, v_c) appears with ε = +1
            edge = (u_c, v_c)
            if edge not in edge_plaquettes:
                edge_plaquettes[edge] = []
            edge_plaquettes[edge].append((cycle_key, +1.0))
            # Reverse: edge (v_c, u_c) has ε = −1
            rev_edge = (v_c, u_c)
            if rev_edge not in edge_plaquettes:
                edge_plaquettes[rev_edge] = []
            edge_plaquettes[rev_edge].append((cycle_key, -1.0))

    # --- Gauge divergence: (1/g²) Σ_{P ∋ e} ε · sin(F_P) ---
    gauge_div: dict[tuple, float] = {}
    for edge in j_matter:
        divg = 0.0
        for cycle_key, epsilon in edge_plaquettes.get(edge, []):
            f_c = effective_curvature.get(cycle_key, 0.0)
            divg += epsilon * math.sin(f_c)
        gauge_div[edge] = divg / g_sq if g_sq > 0.0 else 0.0

    # --- Equation residual ---
    residuals: dict[tuple, float] = {}
    for edge in j_matter:
        r = abs(gauge_div.get(edge, 0.0) - j_matter[edge])
        residuals[edge] = r

    res_values = list(residuals.values())
    mean_res = float(np.mean(res_values)) if res_values else 0.0
    max_res = float(np.max(res_values)) if res_values else 0.0

    return YangMillsFieldEquations(
        matter_current=j_matter,
        gauge_divergence=gauge_div,
        equation_residual=residuals,
        yang_mills_action=float(s_ym),
        matter_action=float(s_matter),
        total_action=float(s_ym + s_matter),
        coupling_constant=float(g_sq),
        mean_residual=mean_res,
        max_residual=max_res,
    )


# ---------------------------------------------------------------------------
# Legacy-named exact-connection closure verification
# ---------------------------------------------------------------------------


def verify_bianchi_identity(
    G: Any,
    *,
    tolerance: float = 1e-10,
) -> BianchiIdentityResult:
    r"""Verify cycle closure of A=d(arg Ψ) within ``tolerance``.

    The historical function and result names are preserved.  Since a graph
    supplies no higher-dimensional cells here, the implemented check is the
    exact-one-form identity ``Σ_C A = 0 (mod 2π)`` on every enumerated cycle,
    rather than a co-boundary sum around vertices.

    Parameters
    ----------
    G : TNFRGraph
    tolerance : float, default=1e-10

    Returns
    -------
    BianchiIdentityResult
    """
    if isinstance(tolerance, bool) or not isinstance(tolerance, Real):
        raise TypeError("tolerance must be a finite non-negative real number")
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be a finite non-negative real number")

    curvature = compute_gauge_curvature(G)
    if not curvature:
        return BianchiIdentityResult(
            is_satisfied=True,
            max_residual=0.0,
            mean_residual=0.0,
            num_coboundaries_tested=0,
        )

    residuals = [abs(value) for value in curvature.values()]
    max_res = float(np.max(residuals)) if residuals else 0.0
    mean_res = float(np.mean(residuals)) if residuals else 0.0

    return BianchiIdentityResult(
        is_satisfied=max_res <= tolerance,
        max_residual=max_res,
        mean_residual=mean_res,
        num_coboundaries_tested=len(residuals),
    )


# ---------------------------------------------------------------------------
# Legacy-named link-current divergence
# ---------------------------------------------------------------------------


def compute_gauss_law_residual(G: Any) -> dict[Any, float]:
    r"""Compute divergence magnitude of the legacy link current.

    The discrete Gauss law states that the divergence of the matter current
    at each node vanishes (current conservation):

        Σ_{j ∈ N(i)} J_matter(i, j) = 0

    For the derived pure-gauge link this current is analytically zero.  The
    result therefore measures numerical cancellation only; it is not a Ward
    identity or a certificate of an engine equilibrium.

    Parameters
    ----------
    G : TNFRGraph

    Returns
    -------
    dict[node, float]
        |Σ_j J(i,j)| per node (legacy field interpretation).
    """
    j_matter = compute_matter_current(G)

    residuals: dict[Any, float] = {}
    for node in G.nodes():
        divergence = 0.0
        for neighbor in G.neighbors(node):
            divergence += j_matter.get((node, neighbor), 0.0)
        residuals[node] = abs(divergence)

    return residuals


# ---------------------------------------------------------------------------
# Legacy mean-squared closure statistic
# ---------------------------------------------------------------------------


def compute_gauge_coupling_constant(G: Any) -> float:
    r"""Compute the mean squared cycle-closure residual (legacy name).

    The historical statistic is

        g² = ⟨F²⟩ = (1/N_P) Σ_P  F_P²

    where N_P is the number of checked cycles.  It is zero analytically for
    A=d(arg Ψ) and is not an independently determined coupling constant.

    Parameters
    ----------
    G : TNFRGraph

    Returns
    -------
    float
        Mean squared floating-point closure residual, non-negative.
    """
    curvature = compute_gauge_curvature(G)
    if not curvature:
        return 0.0
    curv_arr = np.array(list(curvature.values()))
    return float(np.mean(curv_arr**2))


# ---------------------------------------------------------------------------
# Formalized version of the legacy four-label heuristic
# ---------------------------------------------------------------------------


def classify_interaction_regime_formal(
    G: Any,
    node: Any,
) -> InteractionRegimeMetrics:
    r"""Compute the formalized historical four-label snapshot heuristic.

    This function retains legacy labels and adds normalized-score reporting.
    The equal-share activity cut is a reporting convention.  The first two
    coordinates are gauge-frame dependent, the third is numerical closure
    error, and only the potential-to-|Ψ| comparison is locally invariant.

    **Order parameters**:

    1. O_em = |cos(arg Ψ)| — geometric (K_φ) contribution to |Ψ|.

    2. O_wk = |sin(arg Ψ)| — transport (J_φ) contribution to |Ψ|.

    3. O_st = ⟨|F_C|⟩ / π — normalised cycle-closure residual.

    4. O_gr = Φ_s² / (Φ_s² + |Ψ|²) — potential dominance.

    The dominant regime is the one with the highest normalised score; a sector
    is "active" (``above_threshold``) when its score exceeds the equipartition
    share 1/N_REGIMES = 0.25, uniformly across all four sectors.

    Parameters
    ----------
    G : TNFRGraph
    node : Any

    Returns
    -------
    InteractionRegimeMetrics
    """
    psi = compute_complex_geometric_field(G)
    phi_s = compute_structural_potential(G)
    curvature = compute_gauge_curvature(G)

    psi_val = psi.get(node, complex(0, 0))
    psi_mag = abs(psi_val)
    psi_arg = float(np.angle(psi_val))
    ps = abs(phi_s.get(node, 0.0))

    # Adjacent cycle-closure residuals.  Suppress expected binary64 noise.
    adj_curv = [abs(f) for cycle, f in curvature.items() if node in cycle]
    mean_curv = float(np.mean(adj_curv)) if adj_curv else 0.0
    if mean_curv <= GAUGE_CLOSURE_TOLERANCE:
        mean_curv = 0.0

    # --- Order parameters ---
    # O_em: geometric dominance (K_φ axis)
    o_em = abs(math.cos(psi_arg)) if psi_mag > 1e-15 else 0.0

    # O_wk: transport dominance (J_φ axis)
    o_wk = abs(math.sin(psi_arg)) if psi_mag > 1e-15 else 0.0

    # O_st: anomalous closure residual; analytically zero for this connection
    o_st = mean_curv / PI_CONST if PI_CONST > 0 else 0.0

    # O_gr: potential dominance
    denom_gr = ps * ps + psi_mag * psi_mag
    o_gr = (ps * ps) / denom_gr if denom_gr > 1e-30 else 0.0

    # --- Normalised regime scores ---
    # Weight em and weak by the gauge sector fraction,
    # so that when |Ψ| is negligible they don't score.
    gauge_weight = psi_mag / (ps + psi_mag + mean_curv + 1e-15)
    total_score = o_em * gauge_weight + o_wk * gauge_weight + o_st + o_gr
    total_score = max(total_score, 1e-15)

    scores = {
        "em_like": float(o_em * gauge_weight / total_score),
        "weak_like": float(o_wk * gauge_weight / total_score),
        "strong_like": float(o_st / total_score),
        "gravity_like": float(o_gr / total_score),
    }

    dominant = max(scores, key=scores.get)  # type: ignore

    # Historical equal-share activity convention.
    above = {k: scores[k] > REGIME_ACTIVITY_SHARE for k in scores}

    return InteractionRegimeMetrics(
        node=node,
        em_order_parameter=float(o_em),
        weak_order_parameter=float(o_wk),
        strong_order_parameter=float(o_st),
        gravity_order_parameter=float(o_gr),
        dominant_regime=dominant,
        regime_scores=scores,
        above_threshold=above,
        mixing_angle=float(psi_arg),
    )


def compute_network_interaction_profile(G: Any) -> NetworkInteractionProfile:
    r"""Aggregate the historical four-label snapshot heuristic.

    The legacy names are stable API.  The output is descriptive telemetry,
    not a classification of fundamental interactions.

    Parameters
    ----------
    G : TNFRGraph

    Returns
    -------
    NetworkInteractionProfile
    """
    nodes = list(G.nodes())
    per_node = {n: classify_interaction_regime_formal(G, n) for n in nodes}

    # Distribution
    regime_counts: dict[str, int] = {
        "em_like": 0,
        "weak_like": 0,
        "strong_like": 0,
        "gravity_like": 0,
    }
    for m in per_node.values():
        regime_counts[m.dominant_regime] = regime_counts.get(m.dominant_regime, 0) + 1

    n_total = max(len(nodes), 1)
    regime_frac = {k: v / n_total for k, v in regime_counts.items()}

    dominant = max(regime_counts, key=regime_counts.get)  # type: ignore

    # Shannon entropy of regime distribution
    entropy = 0.0
    for p in regime_frac.values():
        if p > 1e-15:
            entropy -= p * math.log(p)

    # Mean order parameters. Empty networks carry finite zero telemetry.
    def mean_parameter(attribute: str) -> float:
        values = [getattr(metrics, attribute) for metrics in per_node.values()]
        return float(np.mean(values)) if values else 0.0

    mean_ops: dict[str, float] = {
        "em_like": mean_parameter("em_order_parameter"),
        "weak_like": mean_parameter("weak_order_parameter"),
        "strong_like": mean_parameter("strong_order_parameter"),
        "gravity_like": mean_parameter("gravity_order_parameter"),
    }

    # Gauge coupling and curvature
    g_sq = compute_gauge_coupling_constant(G)
    s_ym = compute_yang_mills_action(G)

    curvature = compute_gauge_curvature(G)
    curv_vals = [abs(f) for f in curvature.values()] if curvature else [0.0]
    mean_curv = float(np.mean(curv_vals))
    flatness = float(np.mean([1.0 if c < PI_CONST / 10 else 0.0 for c in curv_vals]))

    return NetworkInteractionProfile(
        per_node=per_node,
        regime_distribution=regime_counts,
        regime_fractions=regime_frac,
        mean_order_parameters=mean_ops,
        dominant_regime=dominant,
        mixing_entropy=float(entropy),
        gauge_coupling_constant=float(g_sq),
        yang_mills_action=float(s_ym),
        mean_curvature=float(mean_curv),
        gauge_flatness=float(flatness),
    )
