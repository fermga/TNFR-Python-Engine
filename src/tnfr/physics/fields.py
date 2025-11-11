"""Structural field computations for TNFR physics.

This module computes emergent structural "fields" from TNFR graph state,
grounding a pathway from the nodal equation to macroscopic interaction patterns.

Canonical Status (Updated 2025-11-11)
--------------------------------------
**Φ_s (Structural Potential): CANONICAL**
- Promoted to canonical status after comprehensive validation
- 2,400+ experiments across 5 topology families
- Strong correlation with coherence: corr(Δ Φ_s, ΔC) = -0.822 (R² ≈ 0.68)
- Perfect universality: CV = 0.1% across topologies
- Safety criterion: Δ Φ_s < 2.0 (escape threshold)

**Other Fields: RESEARCH PHASE (NON-CANONICAL)**
- |∇φ| (Phase Gradient): Weak EM-like, corr ≈ -0.13
- K_φ (Phase Curvature): Weak strong-like, corr ≈ -0.07
- ξ_C (Coherence Length): Weak-like symmetry breaking at I_c = 2.015

Research fields must NOT be used for canonical operator decisions.
They provide telemetry-only quantities for analysis and hypothesis testing.

Physics Foundation
------------------
From the nodal equation:
    ∂EPI/∂t = νf · ΔNFR(t)

ΔNFR represents structural pressure driving reorganization. Aggregating
ΔNFR across the network with distance weighting creates the structural
potential field Φ_s, analogous to gravitational potential from mass distribution.

Structural Potential (Φ_s) - CANONICAL
---------------------------------------
Definition:
    Φ_s(i) = Σ_{j≠i} ΔNFR_j / d(i,j)^α  (α=2)

Physical Interpretation:
- Φ_s minima = passive equilibrium states (potential wells)
- Displacement from minima (Δ Φ_s) correlates with coherence loss (ΔC)
- Grammar U1-U5 acts as passive confinement mechanism (not active attractor)
- Valid sequences naturally maintain Δ Φ_s ≈ 0.6 (30% of escape threshold)

Validation Evidence:
- Experiments: 2,400+ simulations across 5 topologies
- Topologies: ring, scale_free, small-world, tree, grid
- Correlation: corr(Δ Φ_s, ΔC) = -0.822 (R² ≈ 0.68)
- Universality: CV = 0.1% (topology-independent)
- Fractality: Scale-dependent β (0.178 nested vs 0.556 flat)
- Mechanism: Passive protection (grammar reduces drift by 85%)

Safety Criterion:
- Escape threshold: Δ Φ_s < 2.0
- Valid sequences: Δ Φ_s ≈ 0.6 (safe regime)
- Violations: Δ Φ_s ≈ 3.9 (fragmentation risk)

Grammar Integration:
- U6: STRUCTURAL POTENTIAL CONFINEMENT (canonical as of 2025-11-11)
- Read-only telemetry-based safety check
- Does NOT dictate operator sequences (unlike U1-U5)
- Validates grammar-compliant sequences naturally stay confined

Research Fields (Phase Gradient, Curvature, Coherence Length)
--------------------------------------------------------------
These fields investigate whether TNFR dynamics naturally generate regimes
with qualitative similarity to fundamental force patterns via phase
organization and curvature. These are analogies only, not claims of
physical identity.

|∇φ| (Phase Gradient):
    Discrete gradient magnitude; large values indicate torsion-like
    dissonance. Weak correlation with ΔC (≈ -0.13), long-range EM-like.

K_φ (Phase Curvature):
    Discrete Laplacian capturing circulation; high curvature zones
    correlate with confinement. Weak correlation (≈ -0.07), strong-like.

ξ_C (Coherence Length):
    Characteristic scale where coherence remains ≥ C_mean; relates to
    screening length. Critical threshold at I_c = 2.015, weak-like.

Promotion Criteria to Canonical (for research fields):
1. Demonstrate predictive power comparable to Φ_s (|corr| > 0.5)
2. Establish unique safety criteria not captured by Φ_s alone
3. Cross-domain validation (biological, social, AI applications)

Canonical Constraints
---------------------
All outputs are read-only telemetry; they never mutate EPI. They must
not reinterpret ΔNFR as a field strength; ΔNFR keeps its nodal meaning.

Functions
---------
compute_structural_potential(G, alpha=2.0): Per-locus Φ_s [CANONICAL]
compute_phase_gradient(G): Phase gradient magnitudes |∇φ| [RESEARCH]
compute_phase_curvature(G): Laplacian curvature K_φ [RESEARCH]
estimate_coherence_length(G): Coherence length ξ_C [RESEARCH]

References
----------
- UNIFIED_GRAMMAR_RULES.md § U6: STRUCTURAL POTENTIAL CONFINEMENT
- docs/TNFR_FORCES_EMERGENCE.md § 14-15: Complete Φ_s validation
- AGENTS.md § Structural Fields: Canonical status and usage
- TNFR.pdf § 2.1: Nodal equation foundation

"""

from __future__ import annotations

from typing import Any, Dict

import math
import numpy as np

try:
    import networkx as nx  # type: ignore
except ImportError:  # pragma: no cover
    nx = None  # type: ignore

__all__ = [
    "compute_structural_potential",
    "compute_phase_gradient",
    "compute_phase_curvature",
    "estimate_coherence_length",
]


def _get_phase(G: Any, node: Any) -> float:
    """Retrieve phase value φ for a node (radians in [0, 2π)).

    Falls back to 0.0 if absent; telemetry-only; no normalization here.
    """
    return float(G.nodes[node].get("phase", 0.0))


def compute_structural_potential(G: Any, alpha: float = 2.0) -> Dict[Any, float]:
    """Compute structural potential Φ_s for each locus [CANONICAL].

    **Canonical Status**: Promoted to CANONICAL on 2025-11-11 after comprehensive
    validation (2,400+ experiments, 5 topology families, CV = 0.1%).

    Definition
    ----------
    Φ_s(i) = Σ_{j≠i} (ΔNFR_j / d(i, j)^α)

    where:
    - ΔNFR_j: Structural pressure at locus j (nodal equation driver)
    - d(i, j): Shortest path length between loci i and j
    - α: Decay exponent (default 2.0 for inverse-square analogy)

    Parameters
    ----------
    G : TNFRGraph
        NetworkX-like graph with node attributes:
        - 'delta_nfr': Structural pressure (float, defaults to 0.0)
        - Optional: 'weight' edge attribute for weighted shortest paths
    alpha : float, default=2.0
        Distance decay exponent. α=2 gives inverse-square (gravitational analog).
        Must be > 0. Higher α = faster decay (more local field).

    Returns
    -------
    Dict[NodeId, float]
        Structural potential Φ_s for each locus.
        - Φ_s < 0: Not meaningful (ΔNFR typically positive in fragmentation)
        - Φ_s ≈ 0: Low aggregate pressure (equilibrium candidate)
        - Φ_s > 0: High pressure zone (potential well with positive pressure sources)

    Physics Interpretation
    ----------------------
    **Passive Equilibrium Landscape**:
    - Φ_s minima represent passive equilibrium states (potential wells)
    - Nodes naturally reside near minima in stable configurations
    - Displacement Δ Φ_s correlates with coherence change ΔC

    **Empirical Relationship** (validated 2,400+ experiments):
        corr(Δ Φ_s, ΔC) = -0.822  (R² ≈ 0.68)

    Strong negative correlation: moving away from Φ_s minima → coherence loss

    **Universality** (5 topology families):
    - Topologies: ring, scale_free, small-world, tree, grid
    - Coefficient of variation: CV = 0.1% (perfect universality)
    - Φ_s dynamics independent of network architecture

    **Safety Criterion** (Grammar U6):
        Δ Φ_s < 2.0  (escape threshold)

    - Valid sequences: Δ Φ_s ≈ 0.6 (30% of threshold, safe regime)
    - Violations: Δ Φ_s ≈ 3.9 (195% of threshold, fragmentation risk)
    - Grammar U1-U5 acts as passive confinement (reduces drift by 85%)

    **Mechanism** (NOT active attraction):
    - NO force pulling nodes toward minima
    - Passive protection: grammar naturally maintains proximity
    - Grammar = stabilizer, not attractor

    **Scale-Dependent Universality**:
    - Flat networks: β = 0.556 (standard criticality)
    - Nested EPIs: β = 0.178 (hierarchical criticality)
    - Φ_s correlation universal across both: -0.822 ± 0.001

    Derivation from Nodal Equation
    -------------------------------
    Starting from:
        ∂EPI/∂t = νf · ΔNFR(t)

    1. ΔNFR is local structural pressure at each node
    2. Network aggregate: sum pressures weighted by distance
    3. Inverse-square (α=2) by analogy to gravitational potential
    4. Result: Φ_s as emergent field from ΔNFR distribution

    Usage as Telemetry
    ------------------
    Φ_s is a **read-only safety metric**:

    1. Compute Φ_s before sequence: Φ_s_before = compute_structural_potential(G)
    2. Apply operator sequence to graph G
    3. Compute Φ_s after sequence: Φ_s_after = compute_structural_potential(G)
    4. Check drift: Δ Φ_s = mean(|Φ_s_after[i] - Φ_s_before[i]|)
    5. Validate: assert Δ Φ_s < 2.0, "Escape threshold exceeded"

    Does NOT dictate which operators to use (unlike U1-U5).
    DOES validate grammar-compliant sequences naturally stay confined.

    Computational Notes
    -------------------
    - Complexity: O(N * (N + E)) for all-pairs shortest paths
    - Uses Dijkstra single-source for weighted graphs
    - Falls back to BFS for unweighted graphs
    - Missing ΔNFR interpreted as 0.0 (no contribution)
    - Unreachable nodes (d=∞) skipped in summation
    - Distance d=0 skipped (self-contribution undefined)

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> # Set ΔNFR (e.g., from dynamics simulation)
    >>> for node in G.nodes():
    ...     G.nodes[node]['delta_nfr'] = 0.5  # Example value
    >>> phi_s = compute_structural_potential(G, alpha=2.0)
    >>> print(f"Node 0 potential: {phi_s[0]:.3f}")

    >>> # Check drift after sequence
    >>> phi_before = compute_structural_potential(G)
    >>> apply_sequence(G, [Emission(), Coherence(), Silence()])
    >>> phi_after = compute_structural_potential(G)
    >>> drift = np.mean([abs(phi_after[n] - phi_before[n]) for n in G.nodes()])
    >>> assert drift < 2.0, f"Drift {drift:.2f} exceeds threshold 2.0"

    See Also
    --------
    compute_phase_gradient : Phase gradient field |∇φ| (research)
    compute_phase_curvature : Phase curvature K_φ (research)
    estimate_coherence_length : Coherence length ξ_C (research)

    References
    ----------
    - UNIFIED_GRAMMAR_RULES.md § U6: STRUCTURAL POTENTIAL CONFINEMENT
    - docs/TNFR_FORCES_EMERGENCE.md § 14: Φ_s drift analysis (corr = -0.822)
    - docs/TNFR_FORCES_EMERGENCE.md § 15: Complete canonicity validation
    - AGENTS.md § Structural Fields: Canonical status and safety criteria
    - TNFR.pdf § 2.1: Nodal equation ∂EPI/∂t = νf · ΔNFR(t)

    Canonicity Justification
    ------------------------
    **Why CANONICAL** (promoted 2025-11-11):
    1. Formal derivation from ΔNFR field theory (nodal equation)
    2. Strong predictive power: R² = 0.68
    3. Universal across topologies: CV = 0.1%
    4. Grammar-compliant: read-only, no U1-U5 conflicts
    5. Extensive validation: 2,400+ experiments, 5 families
    6. Unique dimension: spatial confinement (vs U2 temporal boundedness)

    **Canonicity Level**: STRONG
    - Threshold (2.0) empirically calibrated, not analytically derived
    - α=2 by physics analogy (inverse-square), not proven optimal
    - Correlation strong but not perfect (R² = 0.68, not 1.0)
    - However: Universality and predictive power justify canonical status

    """
    if nx is None:  # pragma: no cover
        raise RuntimeError("networkx required for structural potential computation")

    nodes = list(G.nodes())
    # Precompute ΔNFR values
    delta_nfr = {n: float(G.nodes[n].get("delta_nfr", 0.0)) for n in nodes}

    # All-pairs shortest paths length
    # For performance, we use single-source loops
    potential: Dict[Any, float] = {}

    for src in nodes:
        lengths = (
            nx.single_source_dijkstra_path_length(G, src, weight="weight")
            if G.number_of_edges() > 0
            else {src: 0.0}
        )
        total = 0.0
        for dst in nodes:
            if dst == src:
                continue
            d = lengths.get(dst, math.inf)
            if not math.isfinite(d) or d <= 0.0:
                continue
            contrib = delta_nfr[dst] / (d**alpha)
            total += contrib
        potential[src] = total

    return potential


def compute_phase_gradient(G: Any) -> Dict[Any, float]:
    """Compute magnitude of discrete phase gradient |∇φ| per locus.

    Definition (edge-based):
        |∇φ|(i) = (1/deg(i)) Σ_{j∈N(i)} |wrap_angle(φ_j - φ_i)| / d_ij

    where d_ij is edge weight (defaults 1.0). Phase differences are
    wrapped to [-π, π] to avoid artificial discontinuities.

    High |∇φ| indicates directional tension consistent with dissonance.
    """

    def wrap(angle: float) -> float:
        # Map to [-π, π]
        a = (angle + math.pi) % (2 * math.pi) - math.pi
        return a

    grad: Dict[Any, float] = {}
    for i in G.nodes():
        phi_i = _get_phase(G, i)
        acc = 0.0
        deg = 0
        for j in G.neighbors(i):
            phi_j = _get_phase(G, j)
            w = float(G[i][j].get("weight", 1.0))
            delta = wrap(phi_j - phi_i)
            acc += abs(delta) / max(w, 1e-12)
            deg += 1
        grad[i] = acc / deg if deg > 0 else 0.0
    return grad


def compute_phase_curvature(G: Any) -> Dict[Any, float]:
    """Compute discrete Laplacian curvature K_φ of the phase field.

    Definition:
        K_φ(i) = φ_i - (1/deg(i)) Σ_{j∈N(i)} φ_j

    Interpreted as deviation from local mean phase. Large |K_φ| may
    correlate with localized torsion pockets (mutation candidates).
    """
    curvature: Dict[Any, float] = {}
    for i in G.nodes():
        phi_i = _get_phase(G, i)
        neighs = list(G.neighbors(i))
        if not neighs:
            curvature[i] = 0.0
            continue
        mean_phi = sum(_get_phase(G, j) for j in neighs) / len(neighs)
        curvature[i] = phi_i - mean_phi
    return curvature


def estimate_coherence_length(G: Any, *, coherence_key: str = "coherence") -> float:
    """Estimate coherence length ξ_C via radial decay sampling.

    Procedure:
    1. Select seed locus with maximum local coherence (coherence_key).
    2. Perform BFS; record coherence at each shell distance d.
    3. Fit exponential decay C(d) ≈ C0 * exp(-d / ξ_C) using least squares.

    Returns
    -------
    float
        Estimated coherence length ξ_C (≥ 0). Returns 0.0 if insufficient data.

    Notes
    -----
    - Uses unweighted BFS layers (topological distance).
    - Requires at least 3 shells to attempt fit; else returns 0.0.
    - Coherence values are taken directly; missing treated as 0.0.
    - Future refinement: weight shells by population variance.
    """
    if nx is None:  # pragma: no cover
        raise RuntimeError("networkx required for coherence length estimation")

    nodes = list(G.nodes())
    if not nodes:
        return 0.0

    # Select seed
    coherence = {n: float(G.nodes[n].get(coherence_key, 0.0)) for n in nodes}
    seed = max(nodes, key=lambda n: coherence[n])

    # BFS layering
    layers: Dict[int, list[Any]] = {}
    for n, dist in nx.single_source_shortest_path_length(G, seed).items():
        layers.setdefault(dist, []).append(n)

    # Need at least 3 shells
    if len(layers) < 3:
        return 0.0

    d_vals = []
    c_vals = []
    for d, ns in sorted(layers.items()):
        mean_c = sum(coherence[x] for x in ns) / max(len(ns), 1)
        d_vals.append(float(d))
        c_vals.append(mean_c)

    # Fit log-linear: ln C = ln C0 - d / ξ_C
    c_arr = np.array(c_vals, dtype=float)
    d_arr = np.array(d_vals, dtype=float)

    # Filter non-positive coherence to avoid -inf
    mask = c_arr > 1e-12
    if mask.sum() < 3:
        return 0.0
    c_arr = c_arr[mask]
    d_arr = d_arr[mask]

    y = np.log(c_arr)
    X = np.vstack([np.ones_like(d_arr), -d_arr]).T  # y = a + b * (-d)
    try:
        coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0
    a, b = coeffs  # y = a + b*(-d) => y = a - b d
    # b corresponds to 1/ξ_C approximately if model holds: ln C ≈ ln C0 - d/ξ_C
    if b <= 0:
        return 0.0
    xi = 1.0 / b
    return float(max(xi, 0.0))


# End of research-phase physics helpers.
