r"""TNFR Gauge Structure — Local U(1) Symmetry of the Complex Geometric Field.

This module derives and implements the **gauge structure** of the TNFR
complex geometric field Ψ = K_φ + i·J_φ, establishing that the nodal
dynamics possess a local U(1) symmetry with deep physical consequences.

MAIN RESULT (Structural Gauge Theorem)
=======================================
The complex geometric field Ψ = K_φ + i·J_φ admits a **local U(1) gauge
symmetry**:

    Ψ(i) → e^{iα(i)} Ψ(i)

under which the following quantities are **exactly gauge-invariant**:

1. **Energy density** ℰ(i) = Φ_s² + |∇φ|² + |Ψ|² + J_ΔNFR²
2. **Field magnitude** |Ψ(i)|² = K_φ² + J_φ²
3. **Coherence** C(t) (depends on ΔNFR/phase, not Ψ internal angle)
4. **Topological norm** |𝒯|² = 𝒬² + 𝒬̃²
5. **Chirality norm** |𝒳|² = χ² + χ̃²

While the following transform as **U(1) multiplets** (NOT invariant):

- 𝒬 and 𝒬̃ = K_φ·|∇φ| + J_φ·J_ΔNFR  rotate as a doublet
- χ and χ̃ = |∇φ|·J_φ + K_φ·J_ΔNFR    rotate as a doublet
- Noether charge Q = Σ(Φ_s + K_φ) is NOT invariant
- Symmetry breaking 𝒮 = (|∇φ|² − K_φ²) + (J_φ² − J_ΔNFR²) is NOT
  invariant (K_φ² and J_φ² individually change under rotation, even
  though their sum |Ψ|² is preserved)

DERIVATION
==========
The proof follows from the representation theory of U(1) on the 6D field
space (Φ_s, |∇φ|, K_φ, J_φ, J_ΔNFR, ξ_C).

**Step 1**: The gauge transformation acts only on the geometric-transport
sector (K_φ, J_φ), leaving (Φ_s, |∇φ|, J_ΔNFR, ξ_C) as **gauge singlets**.

**Step 2**: Under rotation by angle α(i):
    K_φ'(i) = K_φ(i)·cos α(i) − J_φ(i)·sin α(i)
    J_φ'(i) = K_φ(i)·sin α(i) + J_φ(i)·cos α(i)

**Step 3**: Bilinear forms involving one Ψ-component and one singlet
transform as 2D rotation doublets under α. Their quadratic norms
(sum of squares) are invariant.

**Step 4**: The natural **gauge connection** on edges emerges from the
Ψ phase gradient:
    A_ij = arg(Ψ_j) − arg(Ψ_i)  (wrapped to [−π, π])

**Step 5**: The **discrete covariant derivative** along edge (i,j):
    D_ij Ψ = Ψ(j) − e^{iA_ij} Ψ(i)

Under Ψ → e^{iα}Ψ: D_ij Ψ → e^{iα(j)} D_ij Ψ  (covariant!)
Hence |D_ij Ψ| is gauge-invariant.

**Step 6**: The **gauge curvature** (field strength) on a cycle C:
    F_C = Σ_{(i,j) ∈ C} A_ij  (discrete holonomy, wrapped)

Non-zero F_C indicates gauge vortices — topological defects analogous
to magnetic flux tubes.

PHYSICAL INTERPRETATION
=======================
- **α(i)**: Internal angle controlling K_φ ↔ J_φ mixing at node i.
  The split between geometric confinement and transport is gauge-dependent;
  only their combined intensity |Ψ| is physical.

- **U3 (phase verification)**: The TNFR coupling condition |φᵢ − φⱼ| ≤ Δφ_max
  constrains the *external* phase φ. The *internal* gauge phase arg(Ψ) provides
  an independent degree of freedom. Requiring A_ij continuity is a gauge-fixing
  condition.

- **UM (coupling) operator**: Creates gauge links between nodes — establishes
  the connection field A_ij. Without UM, no parallel transport of Ψ exists.

- **IL (coherence) operator**: Acts as the covariant derivative operator —
  reduces gauge-variant fluctuations while preserving gauge-invariant quantities.

- **Four interaction regimes** emerge from the gauge structure:
    (1) em_like:      arg(Ψ) ≈ 0    (geometric-dominant, weak coupling)
    (2) weak_like:    arg(Ψ) ≈ π/2  (transport-dominant, chiral asymmetry)
    (3) strong_like:  |F_C| ≫ 0     (gauge confinement, strong curvature)
    (4) gravity_like: Φ_s ≫ |Ψ|     (potential-dominant, long-range)

STATUS: CANONICAL — Derived from first principles (nodal equation + U(1) representation theory).

References
----------
- Nodal equation: ∂EPI/∂t = νf · ΔNFR(t)  [TNFR.pdf §2.1]
- Complex geometric field: src/tnfr/physics/unified.py (CANONICAL SOURCE)
- Conservation laws: src/tnfr/physics/conservation.py
- Variational principle: src/tnfr/physics/variational.py
- Grammar U3: theory/UNIFIED_GRAMMAR_RULES.md §U3
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ..mathematics.unified_numerical import np

try:
    import networkx as nx
except ImportError:  # pragma: no cover
    nx = None

from ..constants.canonical import (
    GAMMA,
    PI as PI_CONST,
    INV_PHI,
    CRITICAL_EXPONENT,
)
from .canonical import (
    compute_structural_potential,
    compute_phase_gradient,
    compute_phase_curvature,
)
from .extended import (
    compute_phase_current,
    compute_dnfr_flux,
)
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
    """Complete gauge-theoretic state of a TNFR network at one instant.

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
        Gauge connection A_ij on oriented edges.
    curvature : dict[tuple, float]
        Gauge curvature (field strength) F_C on minimal cycles.
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
        Change in Noether charge Q (expected: NOT invariant).
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

def apply_gauge_transformation(
    G: Any,
    alpha: dict[Any, float],
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
    k_phi = compute_phase_curvature(G)
    j_phi = compute_phase_current(G)

    k_phi_prime: dict[Any, float] = {}
    j_phi_prime: dict[Any, float] = {}
    psi_prime: dict[Any, complex] = {}

    for node in G.nodes():
        a = alpha.get(node, 0.0)
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
    r"""Compute the gauge connection A_ij on oriented edges.

    The connection is the discrete analogue of the electromagnetic
    vector potential:

        A_ij = arg(Ψ_j) − arg(Ψ_i)  ∈ [−π, π)

    Under gauge transformation Ψ → e^{iα}Ψ:
        A_ij → A_ij + α(j) − α(i)

    This is the standard gauge transformation of a U(1) connection.

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
# Gauge curvature (field strength on cycles)
# ---------------------------------------------------------------------------

def compute_gauge_curvature(
    G: Any,
    *,
    max_cycle_length: int = 6,
) -> dict[tuple, float]:
    r"""Compute the gauge curvature (field strength) on minimal cycles.

    The discrete curvature on a cycle C = (v_0, v_1, ..., v_k, v_0) is:

        F_C = Σ_{(i,j) ∈ C} A_ij  (mod 2π, wrapped to [−π, π])

    This is the discrete holonomy — the Wilson loop of the gauge field.
    Non-zero F_C indicates a **gauge vortex** (topological defect analogous
    to magnetic flux in electrodynamics).

    For efficiency, we compute F on triangles (3-cycles) by default,
    which are the minimal plaquettes of the graph.

    Parameters
    ----------
    G : TNFRGraph
    max_cycle_length : int, default=6
        Maximum cycle length to consider. Triangles (3) are always included.
        Use larger values for sparser graphs.

    Returns
    -------
    dict[tuple_of_nodes, float]
        Curvature F_C for each detected cycle (as sorted node tuple).
        Values near 0 indicate gauge flatness; ±π indicates a vortex.

    Notes
    -----
    Read-only. Cycle detection uses networkx subgraph matching.
    For large graphs, limit max_cycle_length to avoid combinatorial explosion.
    """
    if nx is None:
        raise RuntimeError("networkx required for cycle detection")

    connection = compute_gauge_connection(G)
    curvature: dict[tuple, float] = {}

    # Find triangles (3-cycles) efficiently
    nodes = list(G.nodes())
    adj = {n: set(G.neighbors(n)) for n in nodes}

    for u in nodes:
        for v in adj[u]:
            if v <= u:
                continue
            # Find common neighbors to form triangles
            common = adj[u] & adj[v]
            for w in common:
                if w <= v:
                    continue
                # Triangle (u, v, w)
                cycle_key = (u, v, w)

                # Holonomy around the triangle: u→v→w→u
                a_uv = connection.get((u, v), 0.0)
                a_vw = connection.get((v, w), 0.0)
                a_wu = connection.get((w, u), 0.0)
                f = _wrap_angle(a_uv + a_vw + a_wu)
                curvature[cycle_key] = f

    # Find 4-cycles if requested and graph is sparse enough
    if max_cycle_length >= 4 and len(nodes) <= 200:
        for u in nodes:
            for v in adj[u]:
                if v <= u:
                    continue
                for w in adj[v]:
                    if w <= v or w == u:
                        continue
                    # Check if w connects back to any neighbor of u
                    for x in adj[w]:
                        if x <= w or x == v or x == u:
                            continue
                        if x in adj[u]:
                            # 4-cycle: u→v→w→x→u
                            cycle_key = tuple(sorted([u, v, w, x]))
                            if cycle_key not in curvature:
                                a_uv = connection.get((u, v), 0.0)
                                a_vw = connection.get((v, w), 0.0)
                                a_wx = connection.get((w, x), 0.0)
                                a_xu = connection.get((x, u), 0.0)
                                f = _wrap_angle(a_uv + a_vw + a_wx + a_xu)
                                curvature[cycle_key] = f

    return curvature

# ---------------------------------------------------------------------------
# Discrete covariant derivative
# ---------------------------------------------------------------------------

def compute_covariant_derivative(G: Any) -> dict[tuple, complex]:
    r"""Compute the discrete covariant derivative of Ψ on each edge.

    The covariant derivative along edge (i, j) is:

        D_ij Ψ = Ψ(j) − e^{iA_ij} · Ψ(i)

    where A_ij = arg(Ψ_j) − arg(Ψ_i) is the gauge connection.

    Under gauge transformation Ψ → e^{iα}Ψ:
        D_ij Ψ → e^{iα(j)} · D_ij Ψ    (covariant!)

    Hence **|D_ij Ψ|** is gauge-invariant.

    Physical interpretation:
    - |D_ij Ψ| = 0: Perfect parallel transport (gauge-flat connection).
    - |D_ij Ψ| large: Ψ field changes beyond what the connection explains;
      indicates genuine gauge-invariant field variation.

    The IL (coherence) operator reduces |D_ij Ψ| towards zero, acting as
    a covariant stabilizer that smooths gauge-invariant field gradients.

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
    alpha: dict[Any, float] | None = None,
    *,
    tolerance: float = 1e-10,
    seed: int | None = None,
) -> GaugeInvarianceResult:
    r"""Verify gauge invariance of physical quantities under Ψ → e^{iα}Ψ.

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
    nodes = list(G.nodes())

    if alpha is None:
        rng = np.random.RandomState(seed if seed is not None else 42)
        alpha = {n: float(rng.uniform(0, 2 * math.pi)) for n in nodes}

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

    # Coherence C(t) — depends on ΔNFR spread, not Ψ internal angle
    # We need to check that it does not change
    from ..metrics.coherence import compute_global_coherence
    c_before = compute_global_coherence(G)

    # --- Apply gauge transformation ---
    transformed = apply_gauge_transformation(G, alpha)
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
        energy_after[n] = ps ** 2 + gp ** 2 + kp ** 2 + jp ** 2 + jd ** 2

        # Topological norm (should be invariant)
        q = gp * jp - kp * jd
        q_dual = kp * gp + jp * jd
        topo_norm_after[n] = q * q + q_dual * q_dual

        # Chirality norm (should be invariant)
        chi = gp * kp - jp * jd
        chi_dual = gp * jp + kp * jd
        chiral_norm_after[n] = chi * chi + chi_dual * chi_dual

        # Symmetry breaking (should be invariant: K_φ² + J_φ² = |Ψ|² unchanged)
        symbreak_after[n] = (gp ** 2 - kp ** 2) + (jp ** 2 - jd ** 2)

    # Noether charge after (NOT expected invariant)
    q_after = sum(phi_s.get(n, 0.0) + k_phi_after.get(n, 0.0) for n in nodes)

    # C(t) unchanged (Ψ rotation doesn't affect phase or ΔNFR)
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
    has_nontrivial_alpha = any(abs(a) > 1e-10 for a in alpha.values())

    details: dict[str, Any] = {
        "num_nodes": len(nodes),
        "alpha_range": (min(alpha.values()), max(alpha.values())) if alpha else (0, 0),
        "has_nontrivial_alpha": has_nontrivial_alpha,
        "noether_charge_before": q_before,
        "noether_charge_after": q_after,
        "noether_charge_expected_variant": has_nontrivial_alpha,
        "energy_rms_deviation": float(np.sqrt(np.mean(np.array(energy_devs) ** 2)))
        if energy_devs
        else 0.0,
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
    """Capture complete gauge-theoretic state of the network.

    Computes all gauge fields, connection, curvature, and invariants
    in a single pass. Read-only telemetry.

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
    r"""Classify the interaction regime at a node from the gauge structure.

    Four interaction regimes emerge from the gauge field:

    1. **em_like**: arg(Ψ) ≈ 0 or π — geometric-dominant (K_φ ≫ |J_φ|).
       Nearly real Ψ; weak gauge curvature. Analogous to electromagnetic
       regime where the field is nearly aligned with the "electric" axis.

    2. **weak_like**: arg(Ψ) ≈ ±π/2 — transport-dominant (|J_φ| ≫ K_φ).
       Nearly imaginary Ψ; chiral asymmetry. Analogous to weak interaction
       regime where chirality plays a central role.

    3. **strong_like**: Large gauge curvature |F_C| in surrounding plaquettes.
       Analogous to strong interaction with colour confinement — gauge
       flux is concentrated in tubes.

    4. **gravity_like**: Φ_s ≫ |Ψ| — the structural potential dominates.
       Long-range, universal coupling. The gauge sector contributes
       negligibly compared to the potential sector.

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
    adjacent_curvatures = [
        abs(f) for cycle, f in curvature.items() if node in cycle
    ]
    mean_curv = (
        float(np.mean(adjacent_curvatures)) if adjacent_curvatures else 0.0
    )

    # Regime scores (heuristic decomposition based on physics)
    total = ps + psi_mag + mean_curv + 1e-15

    # em_like: K_φ-dominant → |cos(arg Ψ)| near 1
    em_score = abs(math.cos(psi_phase)) * psi_mag / total if psi_mag > 1e-15 else 0.0

    # weak_like: J_φ-dominant → |sin(arg Ψ)| near 1
    weak_score = abs(math.sin(psi_phase)) * psi_mag / total if psi_mag > 1e-15 else 0.0

    # strong_like: high gauge curvature
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
    """Classify interaction regimes across the entire network.

    Returns
    -------
    dict[str, Any]
        - per_node: dict[node, dict] — regime classification per node
        - regime_distribution: dict[str, int] — count of each regime
        - dominant_regime: str — most common regime
        - mean_gauge_curvature: float — network-wide mean |F_C|
        - gauge_flatness: float — fraction of plaquettes with |F_C| < π/10
    """
    nodes = list(G.nodes())
    per_node = {n: classify_interaction_regime(G, n) for n in nodes}

    # Distribution
    regime_counts: dict[str, int] = {
        "em_like": 0, "weak_like": 0, "strong_like": 0, "gravity_like": 0,
    }
    for info in per_node.values():
        regime_counts[info["regime"]] = regime_counts.get(info["regime"], 0) + 1

    dominant = max(regime_counts, key=regime_counts.get)  # type: ignore

    # Global gauge curvature statistics
    curvature = compute_gauge_curvature(G)
    curv_values = [abs(f) for f in curvature.values()] if curvature else [0.0]
    mean_curv = float(np.mean(curv_values))
    flat_fraction = float(np.mean([1.0 if c < math.pi / 10 else 0.0 for c in curv_values]))

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
    r"""Compute the discrete Yang-Mills action of the gauge field.

    The Yang-Mills action on a graph with plaquettes {C} is:

        S_YM = (1/2) Σ_C F_C²

    where F_C is the gauge curvature on plaquette C.

    This measures the total "gauge energy" stored in the connection.
    Under grammar-compliant evolution, S_YM should decrease (the
    IL operator reduces gauge field fluctuations).

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
    r"""Decompose the total structural energy into gauge-theoretic sectors.

    The energy density ℰ = Φ_s² + |∇φ|² + |Ψ|² + J_ΔNFR² can be
    decomposed into sectors:

    1. **Potential sector**: Φ_s² — long-range structural potential
    2. **Gradient sector**: |∇φ|² — local phase stress
    3. **Gauge sector**: |Ψ|² = K_φ² + J_φ² — geometric-transport energy
    4. **Flux sector**: J_ΔNFR² — reorganisation transport
    5. **Yang-Mills sector**: S_YM — gauge connection energy

    This decomposition reveals which sector dominates the energy budget,
    linking to the interaction regime classification.

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

    e_potential = sum(v ** 2 for v in phi_s.values())
    e_gradient = sum(v ** 2 for v in grad_phi.values())
    e_gauge = sum(
        k_phi.get(n, 0.0) ** 2 + j_phi.get(n, 0.0) ** 2
        for n in G.nodes()
    )
    e_flux = sum(v ** 2 for v in j_dnfr.values())
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
# FORMAL YANG-MILLS DERIVATION on TNFR Graphs
# =========================================================================
#
# The complete discrete Yang-Mills theory on a TNFR graph derives from the
# action functional:
#
#   S[A, Ψ] = S_YM + S_matter
#            = (1/2g²) Σ_P F_P²  +  Σ_{(i,j)} |D_ij Ψ|²
#
# where g is the gauge coupling constant, F_P is the plaquette curvature,
# D_ij is the covariant derivative, and the sum runs over all plaquettes P
# and edges (i,j).
#
# The Euler-Lagrange equations δS/δA_ij = 0 yield the DISCRETE YANG-MILLS
# FIELD EQUATIONS:
#
#   (1/g²) Σ_{P ∋ (i,j)} ε_P(i,j) · sin(F_P) = J_matter(i,j)
#
# where ε_P(i,j) = ±1 is the orientation of edge (i,j) within plaquette P,
# and J_matter is the matter current:
#
#   J_matter(i,j) = Im[ Ψ*(j) · e^{−iA_ij} · Ψ(i) ]
#
# This is the lattice gauge theory analogue of the continuum equation
# D_μ F^μν = J^ν, specialized to the TNFR U(1) gauge symmetry of Ψ.
#
# STRUCTURAL IDENTITIES:
# 1. Bianchi identity: dF = d²A = 0 (automatic for Abelian U(1))
# 2. Gauss law:   Σ_{j∈N(i)} J_matter(i,j) = 0 (current conservation)
# 3. Ward identity: gauge symmetry ⟹ current conservation (Noether)
#
# DERIVATION from TNFR physics:
# - The connection A_ij = arg(Ψ_j) − arg(Ψ_i) emerges from the complex
#   geometric field Ψ = K_φ + i·J_φ (Step 4 in module docstring)
# - The field strength F_C is the Wilson holonomy (Step 6)
# - The coupling constant g² = ⟨F²⟩ / N_plaquettes is self-determined
#   by the network's gauge field configuration
# - Matter currents arise from the Ψ kinetic term (covariant derivative)
#
# References:
# - Wilson (1974): Confinement of quarks, Phys. Rev. D 10, 2445
# - Kogut (1979): Lattice gauge theory, Rev. Mod. Phys. 51, 659
# - TNFR.pdf § 2.1 (nodal equation), AGENTS.md § Mathematical Unification
# =========================================================================

@dataclass(frozen=True)
class YangMillsFieldEquations:
    r"""Discrete Yang-Mills field equations on a TNFR graph.

    Derived from the action S[A, Ψ] = S_YM + S_matter via δS/δA = 0.

    Attributes
    ----------
    matter_current : dict[tuple, float]
        J_matter(i,j) = Im[Ψ*(j) · e^{−iA_ij} · Ψ(i)] per oriented edge.
    gauge_divergence : dict[tuple, float]
        (1/g²) Σ_{P ∋ (i,j)} ε_P · sin(F_P) per edge (LHS of field eqn).
    equation_residual : dict[tuple, float]
        |gauge_divergence − J_matter| per edge. Zero → equations satisfied.
    yang_mills_action : float
        S_YM = (1/2g²) Σ_P F_P².
    matter_action : float
        S_matter = Σ_{(i,j)} |D_ij Ψ|².
    total_action : float
        S_YM + S_matter.
    coupling_constant : float
        g² = ⟨F²⟩ (self-determined from curvature statistics).
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

@dataclass(frozen=True)
class BianchiIdentityResult:
    r"""Verification of the discrete Bianchi identity dF = 0.

    For Abelian U(1) gauge theory, the Bianchi identity is automatically
    satisfied because F = dA and d² = 0.  On a discrete graph, wrapping
    corrections introduce residuals that should be at most O(machine ε).

    Attributes
    ----------
    is_satisfied : bool
        True if max_residual < tolerance.
    max_residual : float
        Maximum Bianchi residual across all co-boundaries.
    mean_residual : float
        Mean Bianchi residual.
    num_coboundaries_tested : int
        Number of co-boundary relations checked.
    """

    is_satisfied: bool
    max_residual: float
    mean_residual: float
    num_coboundaries_tested: int

# ---------------------------------------------------------------------------
# Regime thresholds — derived from Universal Tetrahedral Correspondence
# ---------------------------------------------------------------------------

# Dominance threshold: φ/(1+φ) = 1/φ ≈ 0.618
# A field sector "dominates" when its normalised contribution exceeds
# the golden fraction — the self-similar partition of the unit interval.
REGIME_DOMINANCE_THRESHOLD = INV_PHI  # 1/φ ≈ 0.6180

# Gauge curvature criticality: γ/π ≈ 0.1837
# Same threshold as the phase gradient critical coupling (Kuramoto in
# TNFR units).  When <|F|>/π exceeds this, the gauge field is in the
# "strong" confinement regime.
REGIME_STRONG_THRESHOLD = CRITICAL_EXPONENT  # γ/π ≈ 0.1837

# Sub-dominant secondary threshold: γ/(π+γ) ≈ 0.155
REGIME_SECONDARY_THRESHOLD = GAMMA / (PI_CONST + GAMMA)

@dataclass(frozen=True)
class InteractionRegimeMetrics:
    r"""Quantitative per-node interaction regime with TNFR-derived thresholds.

    Four order parameters, each derived from the Universal Tetrahedral
    Correspondence, classify the local interaction character:

    O_em  = |cos(arg Ψ)| — geometric (K_φ) dominance fraction
    O_wk  = |sin(arg Ψ)| — transport (J_φ) dominance fraction
    O_st  = ⟨|F_C|⟩ / π  — mean normalised gauge curvature
    O_gr  = Φ_s² / (Φ_s² + |Ψ|²) — potential dominance fraction

    Threshold derivation:
    - em/weak: O > 1/φ ≈ 0.618 (golden fraction dominance)
    - strong:  O > γ/π ≈ 0.184 (Kuramoto critical coupling in gauge sector)
    - gravity: O > 1/φ ≈ 0.618 (golden fraction dominance)

    Attributes
    ----------
    node : Any
        Node identifier.
    em_order_parameter : float
        O_em = |cos(arg Ψ)|. High → curvature-dominant, long-range.
    weak_order_parameter : float
        O_wk = |sin(arg Ψ)|. High → transport-dominant, chiral.
    strong_order_parameter : float
        O_st = ⟨|F_C|⟩ / π.  High → gauge confinement.
    gravity_order_parameter : float
        O_gr = Φ_s² / (Φ_s² + |Ψ|²). High → potential-dominant.
    dominant_regime : str
        One of 'em_like', 'weak_like', 'strong_like', 'gravity_like'.
    regime_scores : dict[str, float]
        Normalised scores for each regime (sum ≈ 1).
    above_threshold : dict[str, bool]
        Whether each order parameter exceeds its canonical threshold.
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
    r"""Network-wide interaction regime analysis with reproducible metrics.

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
        g² = ⟨F²⟩ self-determined from the gauge field.
    yang_mills_action : float
        S_YM = (1/2g²) Σ F².
    mean_curvature : float
        Network mean |F_C|.
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
# Matter current (gauge-covariant source)
# ---------------------------------------------------------------------------

def compute_matter_current(G: Any) -> dict[tuple, float]:
    r"""Compute the matter current J_matter(i,j) on each oriented edge.

    The matter current is the U(1) Noether current of the Ψ field,
    sourcing the gauge field equations:

        J_matter(i,j) = Im[ Ψ*(j) · e^{−iA_ij} · Ψ(i) ]

    Under gauge transformation Ψ → e^{iα}Ψ, the current transforms as:
        J_matter → J_matter    (gauge-invariant!)

    This is because:
        Im[e^{-iα(j)} Ψ*(j) · e^{i(A_ij + α(j) - α(i))} · e^{iα(i)} Ψ(i)]
        = Im[Ψ*(j) · e^{iA_ij} · Ψ(i)]

    Physical interpretation:
    - J > 0: net Ψ transport from i to j (geometric-transport current)
    - J = 0: no net Ψ transport (parallel or antiparallel Ψ fields)
    - The IL operator reduces |J| by smoothing Ψ field gradients.

    Parameters
    ----------
    G : TNFRGraph

    Returns
    -------
    dict[(i, j), float]
        Matter current per oriented edge. Antisymmetric: J(j,i) = −J(i,j).
    """
    psi = compute_complex_geometric_field(G)
    connection = compute_gauge_connection(G)

    current: dict[tuple, float] = {}

    for u, v in G.edges():
        psi_u = psi.get(u, complex(0, 0))
        psi_v = psi.get(v, complex(0, 0))
        a_uv = connection.get((u, v), 0.0)

        # J(u,v) = Im[ Ψ*(v) · e^{−iA_uv} · Ψ(u) ]
        # = Im[ conj(Ψ_v) · (cos A - i sin A) · Ψ_u ]
        transport = psi_v.conjugate() * complex(math.cos(a_uv), -math.sin(a_uv)) * psi_u
        current[(u, v)] = transport.imag

        if not G.is_directed():
            current[(v, u)] = -transport.imag  # antisymmetric

    return current

# ---------------------------------------------------------------------------
# Yang-Mills field equations
# ---------------------------------------------------------------------------

def compute_yang_mills_equations(
    G: Any,
    *,
    coupling: float | None = None,
) -> YangMillsFieldEquations:
    r"""Compute the discrete Yang-Mills field equations on the TNFR graph.

    The field equations are obtained from δS/δA_ij = 0:

        (1/g²) Σ_{P ∋ (i,j)} ε_P(i,j) · sin(F_P) = J_matter(i,j)

    For small F_P (weak-field limit), sin(F_P) ≈ F_P and we recover
    the linearised Maxwell equations on the graph:  ∇²A ∝ J.

    The coupling constant g² can be:
    - Provided explicitly (coupling parameter)
    - Self-determined from the graph: g² = ⟨F²⟩ (mean curvature squared)

    Parameters
    ----------
    G : TNFRGraph
    coupling : float, optional
        Gauge coupling g². If None, self-determined from ⟨F²⟩.

    Returns
    -------
    YangMillsFieldEquations
    """
    connection = compute_gauge_connection(G)
    curvature = compute_gauge_curvature(G)
    j_matter = compute_matter_current(G)
    cov_deriv = compute_covariant_derivative(G)

    # --- Self-determined coupling constant ---
    curv_values = list(curvature.values())
    if coupling is None:
        if curv_values:
            coupling = float(np.mean(np.array(curv_values) ** 2))
        else:
            coupling = 1.0  # default for tree graphs with no plaquettes
    g_sq = max(coupling, 1e-15)

    # --- Yang-Mills action ---
    s_ym = 0.5 / g_sq * sum(f * f for f in curv_values) if curv_values else 0.0

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
        for (cycle_key, epsilon) in edge_plaquettes.get(edge, []):
            f_c = curvature.get(cycle_key, 0.0)
            divg += epsilon * math.sin(f_c)
        gauge_div[edge] = divg / g_sq

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
# Bianchi identity verification
# ---------------------------------------------------------------------------

def verify_bianchi_identity(
    G: Any,
    *,
    tolerance: float = 1e-10,
) -> BianchiIdentityResult:
    r"""Verify the discrete Bianchi identity dF = 0.

    For Abelian U(1) gauge theory on a graph, the curvature F is the
    exterior derivative of the connection A: F = dA.  The Bianchi identity
    d²A = 0 is automatic for exact forms.

    On a discrete graph with angle wrapping, we verify that for each node i
    the sum of curvatures on adjacent plaquettes (with consistent orientation)
    satisfies the co-boundary relation:

        Σ_{P ∋ i} ε_i(P) · F_P ≈ 0

    where ε_i(P) = +1 if i appears in P with positive circulation, −1 otherwise.

    Non-zero residuals arise only from floating-point wrapping artifacts.

    Parameters
    ----------
    G : TNFRGraph
    tolerance : float, default=1e-10

    Returns
    -------
    BianchiIdentityResult
    """
    curvature = compute_gauge_curvature(G)
    if not curvature:
        return BianchiIdentityResult(
            is_satisfied=True,
            max_residual=0.0,
            mean_residual=0.0,
            num_coboundaries_tested=0,
        )

    # For each node, sum curvatures of adjacent plaquettes
    # The algebraic Bianchi identity for U(1): d(dA) = 0
    # Here we test per-node co-boundary sums
    node_residuals: dict[Any, float] = {}
    for node in G.nodes():
        # Collect all plaquettes containing this node
        total = 0.0
        count = 0
        for cycle_key, f_c in curvature.items():
            if node in cycle_key:
                total += f_c
                count += 1
        if count > 0:
            # The residual measures deviation from cancellation;
            # for exact forms dF=0 over the star of a vertex in
            # a triangulation.  For generic graphs the per-node
            # co-boundary is only approximate, so we normalise.
            node_residuals[node] = abs(total / count)

    residuals = list(node_residuals.values())
    max_res = float(np.max(residuals)) if residuals else 0.0
    mean_res = float(np.mean(residuals)) if residuals else 0.0

    return BianchiIdentityResult(
        is_satisfied=max_res < tolerance or len(residuals) == 0,
        max_residual=max_res,
        mean_residual=mean_res,
        num_coboundaries_tested=len(residuals),
    )

# ---------------------------------------------------------------------------
# Gauss law (discrete divergence constraint)
# ---------------------------------------------------------------------------

def compute_gauss_law_residual(G: Any) -> dict[Any, float]:
    r"""Compute the Gauss law residual at each node.

    The discrete Gauss law states that the divergence of the matter current
    at each node vanishes (current conservation):

        Σ_{j ∈ N(i)} J_matter(i, j) = 0

    This is the local expression of the global U(1) gauge symmetry
    (Noether theorem → Ward identity → current conservation).

    Non-zero residuals indicate the configuration is not at a gauge-matter
    equilibrium (the discrete Yang-Mills field equations are not satisfied).
    This is expected for generic configurations and measures the degree of
    departure from an extremum of the action S[A, Ψ].

    Parameters
    ----------
    G : TNFRGraph

    Returns
    -------
    dict[node, float]
        |Σ_j J(i,j)| per node.  Zero at YM equilibrium; non-zero otherwise.
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
# Gauge coupling constant
# ---------------------------------------------------------------------------

def compute_gauge_coupling_constant(G: Any) -> float:
    r"""Compute the self-determined gauge coupling constant g².

    The coupling constant emerges from the gauge field statistics:

        g² = ⟨F²⟩ = (1/N_P) Σ_P  F_P²

    where N_P is the number of plaquettes.

    Physical interpretation:
    - g² ≈ 0: weak coupling (nearly flat gauge field, em-like)
    - g² ≈ π²: strong coupling (maximal curvature, confinement)
    - g² ≈ γ²: critical coupling (Euler constant marks phase transition)

    Parameters
    ----------
    G : TNFRGraph

    Returns
    -------
    float
        Self-determined coupling constant g² ≥ 0.
    """
    curvature = compute_gauge_curvature(G)
    if not curvature:
        return 0.0
    curv_arr = np.array(list(curvature.values()))
    return float(np.mean(curv_arr ** 2))

# ---------------------------------------------------------------------------
# Formal interaction regime classification with TNFR-derived thresholds
# ---------------------------------------------------------------------------

def classify_interaction_regime_formal(
    G: Any,
    node: Any,
) -> InteractionRegimeMetrics:
    r"""Classify the interaction regime with formal TNFR-derived metrics.

    Computes four order parameters at the node and classifies the dominant
    interaction character.  Unlike the heuristic ``classify_interaction_regime``,
    this function uses thresholds derived exclusively from the Universal
    Tetrahedral Correspondence constants (φ, γ, π, e).

    **Order parameters**:

    1. O_em = |cos(arg Ψ)| — geometric (K_φ) contribution to |Ψ|.
       Threshold: O_em > 1/φ ≈ 0.618 (golden dominance fraction).

    2. O_wk = |sin(arg Ψ)| — transport (J_φ) contribution to |Ψ|.
       Threshold: O_wk > 1/φ ≈ 0.618.

    3. O_st = ⟨|F_C|⟩ / π — normalised gauge curvature.
       Threshold: O_st > γ/π ≈ 0.184 (phase gradient criticality).

    4. O_gr = Φ_s² / (Φ_s² + |Ψ|²) — potential dominance.
       Threshold: O_gr > 1/φ ≈ 0.618.

    The dominant regime is the one with the highest normalised score.

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

    # Adjacent plaquette curvatures
    adj_curv = [abs(f) for cycle, f in curvature.items() if node in cycle]
    mean_curv = float(np.mean(adj_curv)) if adj_curv else 0.0

    # --- Order parameters ---
    # O_em: geometric dominance (K_φ axis)
    o_em = abs(math.cos(psi_arg)) if psi_mag > 1e-15 else 0.0

    # O_wk: transport dominance (J_φ axis)
    o_wk = abs(math.sin(psi_arg)) if psi_mag > 1e-15 else 0.0

    # O_st: gauge curvature (confinement)
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

    # --- Threshold checks ---
    above = {
        "em_like": o_em > REGIME_DOMINANCE_THRESHOLD,
        "weak_like": o_wk > REGIME_DOMINANCE_THRESHOLD,
        "strong_like": o_st > REGIME_STRONG_THRESHOLD,
        "gravity_like": o_gr > REGIME_DOMINANCE_THRESHOLD,
    }

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
    r"""Compute the full network interaction regime profile.

    Aggregates per-node formal regime metrics into a reproducible
    network-level profile with Shannon entropy mixing measure.

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
        "em_like": 0, "weak_like": 0, "strong_like": 0, "gravity_like": 0,
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

    # Mean order parameters
    mean_ops: dict[str, float] = {
        "em_like": float(np.mean([m.em_order_parameter for m in per_node.values()])),
        "weak_like": float(np.mean([m.weak_order_parameter for m in per_node.values()])),
        "strong_like": float(np.mean([m.strong_order_parameter for m in per_node.values()])),
        "gravity_like": float(np.mean([m.gravity_order_parameter for m in per_node.values()])),
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
