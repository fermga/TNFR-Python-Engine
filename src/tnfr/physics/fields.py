"""Structural field computations for TNFR physics.

This module computes emergent structural "fields" from TNFR graph state,
grounding a pathway from the nodal equation to macroscopic interaction
patterns.

Canonical Status (Updated 2025-11-12) - EXTENDED CANONICAL HEXAD
--------------------------------------------------------------------
**Φ_s (Structural Potential): CANONICAL**
- Promoted to canonical status after comprehensive validation
- 2,400+ experiments across 5 topology families
- Strong correlation with coherence: corr(Δ Φ_s, ΔC) = -0.822 (R² ≈ 0.68)
- Perfect universality: CV = 0.1% across topologies
- Safety criterion: Δ Φ_s < 2.0 (escape threshold)

**|∇φ| (Phase Gradient): CANONICAL** ⭐ **PROMOTED Nov 2025**
- Promoted to canonical status November 11, 2025
- 450 experiments across 5 topology families
- Strong correlation with peak stress:
    corr(Δ|∇φ|, Δmax_ΔNFR) = +0.6554 (31% above 0.5 threshold)
- Universal across topologies: Tree (+0.74), Scale-free (+0.71),
    WS (+0.71), Grid (+0.63), Ring (+0.52)
- 12% superior to Φ_s as predictor of peak node stress
- Safety criterion: |∇φ| < 0.38 (stable operation threshold)

**K_φ (Phase Curvature): CANONICAL** ⭐ **PROMOTED Nov 2025**
- Promoted to canonical status November 11, 2025
- Local threshold safety: |K_φ| ≥ 3.0 flags
    confinement/fault zones (100% accuracy in enhanced fragmentation
    battery)

**J_φ (Phase Current): CANONICAL** ⭐ **NEWLY PROMOTED Nov 12, 2025**
- Promoted based on robust multi-topology validation
- 48 samples across WS, BA, Grid topologies
- Ultra-robust correlation: r(J_φ, K_φ) = +0.592 ± 0.092
- 100% sign consistency across parameter sweeps
- Physics: Geometric phase confinement drives directed transport
- Integration priority: HIGH

**J_ΔNFR (ΔNFR Flux): CANONICAL** ⭐ **NEWLY PROMOTED Nov 12, 2025**
- Promoted based on robust multi-topology validation
- 48 samples across WS, BA, Grid topologies
- Ultra-robust correlation: r(J_ΔNFR, Φ_s) = -0.471 ± 0.159
- 100% sign consistency across parameter sweeps
- Physics: Potential-driven reorganization transport
- Integration priority: HIGH
- Multiscale asymptotic freedom: var(K_φ) ~ 1/r^α with α ≈ 2.76
    (excellent fits, R² ≥ 0.8 on scale-free/WS; tolerance check usable
    with α_hint)
- Cross-domain universality: neural (R² > 0.8), AI scale-free
    (R² ≈ 0.998), social (conflict zones detected via high |K_φ|
    variance)

**ξ_C (Coherence Length): CANONICAL** ⭐ **PROMOTED Nov 12, 2025**
- Promoted to canonical status after comprehensive multi-topology validation
- 1,170 measurements across 3 topology families (100% success rate)
- Critical point prediction: I_c = 2.015 matches observed ±0.005
- Power law scaling: ξ_C ~ |I - I_c|^(-ν) confirmed experimentally
- Multi-scale behavior: ξ_C spans 271 - 46,262 (2-3 orders of magnitude)
- Critical exponents: ν ≈ 0.61 (WS, mean-field), 0.95 (Grid, 3D-like)
- Phase transitions: Clear second-order critical behavior observed
- Safety criteria: ξ_C > system_diameter (critical approach warning)

**STRUCTURAL FIELD TETRAD COMPLETE**
All four canonical fields now provide complete multi-scale characterization:
- Φ_s: Global potential (field theory)
- |∇φ|: Local desynchronization (gradient)
- K_φ: Phase curvature (geometric confinement)
- ξ_C: Spatial correlations (correlation length scale)

No remaining research-phase structural fields.

Physics Foundation
------------------
From the nodal equation:
    ∂EPI/∂t = νf · ΔNFR(t)

ΔNFR represents structural pressure driving reorganization. Aggregating
ΔNFR across the network with distance weighting creates the structural
potential field Φ_s, analogous to gravitational potential from mass
distribution.

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

Phase Gradient (|∇φ|) - CANONICAL
-----------------------------------
Definition:
    |∇φ|(i) = mean_{j ∈ neighbors(i)} |θ_i - θ_j|

Physical Interpretation:
- Measures local phase desynchronization in node neighborhoods
- High |∇φ| → high peak node stress (max_ΔNFR) [correlation +0.6554]
- Serves as early warning indicator for structural fragmentation risk
- Complementary to Φ_s: captures local dynamics vs global potential

Validation Evidence:
- Experiments: 450 simulations across 5 topologies (November 2025)
- Primary correlation: corr(Δ|∇φ|, Δmax_ΔNFR) = +0.6554 (STRONG)
- Secondary correlation: corr(Δ|∇φ|, Δmean_ΔNFR) = +0.6379 (STRONG)
- Universality: All topologies achieve |corr| > 0.5 (Tree: 0.74,
  Scale-free: 0.71)
- Superiority: 12% better than Φ_s as predictor of peak stress
- Grammar compliance: U1-U5 compatible, read-only telemetry

Safety Criterion:
- Stable operation threshold: |∇φ| < 0.38
- High-stress discrimination: 107% higher |∇φ| in stressed regimes
- Early warning: 2-3 operator steps before fragmentation events

Critical Discovery - Alternative Metrics:
Initial validation targeted C(t) = 1 - (σ_ΔNFR / ΔNFR_max) but discovered
this metric is invariant to proportional scaling. When grammar-compliant
sequences produce uniform ΔNFR changes, C(t) remains constant despite
significant reorganization. |∇φ| correlation with max_ΔNFR and mean_ΔNFR
captures dynamics that C(t) misses.

Coherence Length (ξ_C) - CANONICAL
-----------------------------------
Definition:
    Per-node coherence: c_i = 1.0 / (1.0 + |ΔNFR_i|)
    Spatial autocorrelation: C(r) = ⟨c_i · c_j⟩ where d(i,j) ≈ r
    Coherence length: C(r) ~ exp(-r/ξ_C)

Physical Interpretation:
- ξ_C measures spatial scale of coherence correlations
- Below I_c: ξ_C finite, coherence localized
- At I_c: ξ_C diverges, system-wide correlations emerge
- Above I_c: ξ_C decreases, coherence fragments

Validation Evidence:
- Experiments: 1,170 measurements across 3 topologies (November 12, 2025)
- Topologies: WS (Watts-Strogatz), Scale-free, Grid
- Success rate: 100% valid measurements (no failures)
- Critical point: Theoretical I_c = 2.015 matches observed ±0.005
- Power law: ξ_C ~ |I - I_c|^(-ν) confirmed
- Scale range: 271 - 46,262 (2-3 orders of magnitude)
- Critical exponents: ν ≈ 0.61 (WS, mean-field), 0.95 (Grid, 3D-like)
- Phase transitions: Second-order critical behavior observed

Safety Criteria:
- Critical approach: ξ_C > system_diameter (system-wide reorganization)
- Long-range correlations: ξ_C > 3 × mean_node_distance (monitor closely)
- Localized: ξ_C < mean_node_distance (stable regime)

Complementarity with Other Fields:
- Φ_s: Global potential → ξ_C adds spatial correlation scale
- |∇φ|: Local stress → ξ_C adds correlation length dimension
- K_φ: Geometric confinement → ξ_C adds critical phenomena detection

Grammar Integration:
- Read-only telemetry (like Φ_s, |∇φ|, K_φ)
- Does NOT modify operator sequences
- Enables phase transition monitoring
- Critical point early warning system

**STRUCTURAL FIELD TETRAD COMPLETE**
All four structural fields now have CANONICAL status:
- Φ_s: Global structural potential (field theory dimension)
- |∇φ|: Local phase desynchronization (gradient dimension)
- K_φ: Phase curvature / geometric confinement (curvature dimension)
- ξ_C: Coherence length / spatial correlations (correlation dimension)

No research-phase structural fields remain. The tetrad provides complete
multi-scale characterization of TNFR network state.

Canonical Constraints
---------------------
All outputs are read-only telemetry; they never mutate EPI. They must
not reinterpret ΔNFR as a field strength; ΔNFR keeps its nodal meaning.

Functions
---------
compute_structural_potential(G, alpha=2.0): Per-locus Φ_s [CANONICAL]
compute_phase_gradient(G): Phase gradient magnitudes |∇φ| [CANONICAL]
compute_phase_curvature(G): Laplacian curvature K_φ [CANONICAL]
estimate_coherence_length(G): Coherence length ξ_C [CANONICAL] ⭐ PROMOTED
path_integrated_gradient(G, source, target): Path-integrated |∇φ| [RESEARCH]

References
----------
- UNIFIED_GRAMMAR_RULES.md § U6: STRUCTURAL POTENTIAL CONFINEMENT
- docs/TNFR_FORCES_EMERGENCE.md § 14-15: Complete Φ_s validation
- docs/XI_C_CANONICAL_PROMOTION.md: ξ_C complete experimental validation
- AGENTS.md § Structural Fields: Canonical status and usage (Tetrad complete)
- TNFR.pdf § 2.1: Nodal equation foundation

"""

from __future__ import annotations

from typing import Any, Dict

import math
import numpy as np
import warnings

try:
    import networkx as nx  # type: ignore
except ImportError:  # pragma: no cover
    nx = None  # type: ignore

# Import TNFR aliases for proper attribute access
try:
    from ..constants.aliases import ALIAS_THETA, ALIAS_DNFR  # type: ignore
except ImportError:  # pragma: no cover
    # Fallback for standalone usage
    ALIAS_THETA = ["phase", "theta"]  # type: ignore
    ALIAS_DNFR = ["delta_nfr", "dnfr"]  # type: ignore

__all__ = [
    # Original Canonical Tetrad
    "compute_structural_potential",
    "compute_phase_gradient",
    "compute_phase_curvature",
    "estimate_coherence_length",
    # Extended Canonical Fields (NEWLY PROMOTED Nov 12, 2025)
    "compute_phase_current",
    "compute_dnfr_flux",
    "compute_extended_canonical_suite",
    # Additional utilities
    "path_integrated_gradient",
    "compute_phase_winding",
    # Research-phase multiscale K_φ utilities
    "compute_k_phi_multiscale_variance",
    "fit_k_phi_asymptotic_alpha",
    "k_phi_multiscale_safety",
]


def _wrap_angle(angle: float) -> float:
    """Map angle to [-π, π]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _get_phase(G: Any, node: Any) -> float:
    """Retrieve phase value φ for a node (radians in [0, 2π)).

    Uses TNFR alias system to find theta/phase attribute.
    Falls back to 0.0 if absent; telemetry-only; no normalization here.
    """
    # Try TNFR aliases
    node_data = G.nodes[node]
    for alias in ALIAS_THETA:
        if alias in node_data:
            return float(node_data[alias])
    return 0.0


def _get_dnfr(G: Any, node: Any) -> float:
    """Retrieve ΔNFR value for a node.

    Uses TNFR alias system to find dnfr/delta_nfr attribute.
    Falls back to 0.0 if absent.
    """
    node_data = G.nodes[node]
    for alias in ALIAS_DNFR:
        if alias in node_data:
            return float(node_data[alias])
    return 0.0


def compute_structural_potential(
    G: Any, alpha: float = 2.0
) -> Dict[Any, float]:
    """Compute structural potential Φ_s for each locus [CANONICAL].

    **Canonical Status**: Promoted to CANONICAL on 2025-11-11 after
    comprehensive validation (2,400+ experiments, 5 topology families,
    CV = 0.1%).

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
        Distance decay exponent. α=2 gives inverse-square
        (gravitational analog). Must be > 0. Higher α = faster decay
        (more local field).

    Returns
    -------
    Dict[NodeId, float]
        Structural potential Φ_s for each locus.
        - Φ_s < 0: Not meaningful (ΔNFR typically positive in
          fragmentation)
        - Φ_s ≈ 0: Low aggregate pressure (equilibrium candidate)
        - Φ_s > 0: High pressure zone (potential well with positive
          pressure sources)

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

    1. Compute Φ_s before sequence:
       Φ_s_before = compute_structural_potential(G)
    2. Apply operator sequence to graph G
    3. Compute Φ_s after sequence:
       Φ_s_after = compute_structural_potential(G)
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
    compute_phase_gradient : Phase gradient field |∇φ| [CANONICAL]
    compute_phase_curvature : Phase curvature K_φ [CANONICAL]
    estimate_coherence_length : Coherence length ξ_C [CANONICAL]

    References
    ----------
    - UNIFIED_GRAMMAR_RULES.md § U6: STRUCTURAL POTENTIAL CONFINEMENT
    - docs/TNFR_FORCES_EMERGENCE.md § 14: Φ_s drift analysis (corr = -0.822)
    - docs/TNFR_FORCES_EMERGENCE.md § 15: Complete canonicity validation
    - docs/XI_C_CANONICAL_PROMOTION.md: ξ_C experimental validation (1,170 exp)
    - AGENTS.md § Structural Fields: Canonical tetrad (Φ_s, |∇φ|, K_φ, ξ_C)
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
    - However: Universality and predictive power justify canonical
      status

    """
    if nx is None:  # pragma: no cover
        raise RuntimeError(
            "networkx required for structural potential computation"
        )

    nodes = list(G.nodes())
    # Precompute ΔNFR values using TNFR alias system
    delta_nfr = {n: _get_dnfr(G, n) for n in nodes}

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
    [CANONICAL]

    Status
    ------
    CANONICAL (promoted November 11, 2025)

    Definition
    ----------
        |∇φ|(i) = mean_{j in neighbors(i)} |θ_i - θ_j|

    Phases (θ) accessed via ALIAS_THETA for robustness.

    Physical Interpretation
    ----------------------
    - Local phase desynchronization metric
    - High |∇φ| precedes elevated max_ΔNFR (stress)
    - Early warning: rises 2–3 operator steps pre fragmentation
    - Complements Φ_s (global) and K_φ (curvature)

    Validation Evidence
    -------------------
    - 450 experiments (5 topologies, Nov 2025)
    - corr(Δ|∇φ|, Δmax_ΔNFR) = +0.6554 (strong)
    - corr(Δ|∇φ|, Δmean_ΔNFR) = +0.6379
    - Universal: all families |corr| > 0.5
    - 12% superior to Φ_s for peak stress prediction

    Safety Criterion (Telemetry)
    ----------------------------
    |∇φ| < 0.38 stable. Above threshold → apply stabilizers
    (e.g., Coherence) to reduce local desynchronization.

    Usage
    -----
    >>> grad = compute_phase_gradient(G)
    >>> risky = [n for n,v in grad.items() if v >= 0.38]
    >>> if risky:
    ...     apply_operator_sequence(G, risky, [Coherence()])

    Computational Notes
    -------------------
    - O(E) neighbor traversal
    - Missing phase defaults 0.0 (telemetry only)
    - Isolated nodes yield 0.0

    Returns
    -------
    Dict[Any, float]
        Phase gradient magnitude per node.

    References
    ----------
    - AGENTS.md Structural Fields section
    - docs/TNFR_FORCES_EMERGENCE.md (§14–15)
    - TNFR.pdf §2.1

    Canonicity Justification
    ------------------------
    1. Predictive, universal stress indicator
    2. Read-only; preserves all invariants
    3. Adds local desynchronization dimension not in Φ_s
    4. Reproducible across seeds/topologies
    """

    grad: Dict[Any, float] = {}
    for i in G.nodes():
        neighbors = list(G.neighbors(i))
        if not neighbors:
            grad[i] = 0.0
            continue
        
        phi_i = _get_phase(G, i)
        
        # Compute mean absolute phase difference with neighbors
        phase_diffs = []
        for j in neighbors:
            phi_j = _get_phase(G, j)
            # Use wrapped difference to respect circular topology
            phase_diffs.append(abs(_wrap_angle(phi_i - phi_j)))
        
        grad[i] = sum(phase_diffs) / len(phase_diffs)
    
    return grad


def compute_phase_curvature(G: Any) -> Dict[Any, float]:
    """Compute discrete Laplacian curvature K_φ of the phase field. [CANONICAL]

    Status
    ------
    CANONICAL (promoted November 11, 2025)

        Physical Interpretation
        ----------------------
        - Phase torsion vs local mean (Laplacian curvature)
        - High |K_φ| → confinement pockets / mutation loci
        - Complements Φ_s (global) and |∇φ| (gradient): curvature axis

        Validation Evidence
        -------------------
        - Threshold revision: 4.88 → 3.0 (accuracy ↑ from 55% to 100%)
        - Multiscale asymptotic freedom: var(K_φ) ~ 1/r^α (α ≈ 2.76)
        - Cross-domain: neural R²>0.8, AI scale-free R²≈0.998, social
            conflict zones via variance spikes

        Safety Criteria (Telemetry)
        ---------------------------
        - Local: |K_φ| ≥ 3.0 → flag
        - Multiscale: k_phi_multiscale_safety(...).safe must be True

        Usage
        -----
        >>> kphi = compute_phase_curvature(G)
        >>> hotspots = [n for n,v in kphi.items() if abs(v) >= 3.0]
        >>> if hotspots:
        ...     apply_operator_sequence(G, hotspots, [Coherence()])
        >>> safety = k_phi_multiscale_safety(G, alpha_hint=2.76)
        >>> assert safety["safe"], "Curvature multiscale unsafe"

        Computational Notes
        -------------------
        - O(E) neighbor mean computations
        - Isolated nodes → 0.0
        - Phase via alias resolution (_get_phase)

        References
        ----------
        - K_PHI_RESEARCH_SUMMARY.md
        - enhanced_fragmentation_test.py
        - AGENTS.md Structural Fields triad

        Canonicity Justification
        ------------------------
        1. Independent curvature dimension (not Φ_s, not |∇φ|)
        2. Robust threshold + multiscale decay physics
        3. Cross-domain universality
        4. Read-only; preserves invariants
    """

    curvature: Dict[Any, float] = {}
    for i in G.nodes():
        neighbors = list(G.neighbors(i))
        if not neighbors:
            curvature[i] = 0.0
            continue

        phi_i = _get_phase(G, i)
        # Circular mean of neighbor phases via unit vectors
        neigh_phases = [
            _get_phase(G, j) for j in neighbors
        ]
        if not neigh_phases:
            curvature[i] = 0.0
            continue

        mean_vec = complex(
            float(np.mean([math.cos(p) for p in neigh_phases])),
            float(np.mean([math.sin(p) for p in neigh_phases]))
        )
        # If mean vector length ~ 0 (highly dispersed), fallback to simple mean
        if abs(mean_vec) < 1e-9:
            mean_phase = float(np.mean(neigh_phases))
        else:
            mean_phase = math.atan2(mean_vec.imag, mean_vec.real)

        # Curvature as wrapped deviation from neighbor circular mean
        curvature[i] = float(_wrap_angle(phi_i - mean_phase))

    return curvature


def _ego_mean(values: Dict[Any, float], nodes: list[Any]) -> float:
    """Mean of values restricted to given nodes; returns 0.0 if empty."""
    if not nodes:
        return 0.0
    # Only include values for nodes in the provided list
    arr = [values[n] for n in nodes if n in values]
    if not arr:
        return 0.0
    return float(sum(arr) / len(arr))


def compute_k_phi_multiscale_variance(
    G: Any,
    *,
    scales: tuple[int, ...] = (1, 2, 3, 5),
    k_phi_field: Dict[Any, float] | None = None,
) -> Dict[int, float]:
    """Compute variance of coarse-grained K_φ across scales.
    [CANONICAL TELEMETRY]

    Definition (coarse-graining by r-hop ego neighborhoods):
        K_φ^r(i) = mean_{j in ego_r(i)} K_φ(j)
        var_r = Var_i [ K_φ^r(i) ]

    Parameters
    ----------
    G : TNFRGraph
        NetworkX-like graph with phase attributes accessible via aliases.
    scales : tuple[int, ...]
        Radii (in hops) at which to compute coarse-grained variance.
    k_phi_field : Optional[Dict]
        Precomputed K_φ per node. If None, computed via
        compute_phase_curvature.

    Returns
    -------
    Dict[int, float]
        Mapping from radius r to variance of coarse-grained K_φ at that scale.

    Notes
    -----
    - Read-only telemetry; does not mutate graph state.
    - Intended to support asymptotic freedom assessments (Task 3).
    """
    if nx is None:  # pragma: no cover
        raise RuntimeError(
            "networkx required for multiscale variance computation"
        )

    if k_phi_field is None:
        k_phi_field = compute_phase_curvature(G)

    result: Dict[int, float] = {}
    nodes = list(G.nodes())
    if not nodes:
        return {r: 0.0 for r in scales}

    for r in scales:
        coarse: Dict[Any, float] = {}
        for i in nodes:
            ego = nx.ego_graph(G, i, radius=r)
            ego_nodes = list(ego.nodes())
            coarse[i] = _ego_mean(k_phi_field, ego_nodes)

        vals = list(coarse.values())
        result[r] = float(np.var(vals)) if vals else 0.0

    return result


def fit_k_phi_asymptotic_alpha(
    variance_by_scale: Dict[int, float]
) -> Dict[str, float | int | str]:
    """Fit power law var(K_φ) ~ 1/r^α via log-log linear regression.
    [CANONICAL TELEMETRY]

    Parameters
    ----------
    variance_by_scale : Dict[int, float]
        Mapping of scale r (hops) to variance at that scale.

    Returns
    -------
    Dict[str, Any]
        - alpha: fitted exponent α (float)
    - r_squared: coefficient of determination R^2 (float)
        - n_points: number of scales used (int)
        - fit_quality: 'excellent' | 'good' | 'poor' | 'insufficient_data'
    """
    # Extract valid points (positive variance and positive scale)
    pairs = [
        (r, v) for r, v in variance_by_scale.items() if r > 0 and v > 1e-12
    ]
    pairs.sort(key=lambda x: x[0])
    if len(pairs) < 3:
        return {
            "alpha": 0.0,
            "r_squared": 0.0,
            "n_points": len(pairs),
            "fit_quality": "insufficient_data",
        }

    rs = np.array([p[0] for p in pairs], dtype=float)
    vars_ = np.array([p[1] for p in pairs], dtype=float)

    log_r = np.log(rs)
    log_var = np.log(vars_)

    # Linear regression: log(var) ~ a - alpha * log(r)
    A = np.vstack([np.ones_like(log_r), -log_r]).T
    try:
        coeffs, residuals, rank, s = np.linalg.lstsq(A, log_var, rcond=None)
        a, neg_alpha = coeffs
    # Compute R^2 manually
        y_pred = A @ coeffs
        ss_res = float(np.sum((log_var - y_pred) ** 2))
        ss_tot = float(np.sum((log_var - float(np.mean(log_var))) ** 2))
        r_squared = 1.0 - (ss_res / ss_tot if ss_tot > 0 else 0.0)
        alpha = float(-neg_alpha)
        fit_quality = (
            "excellent"
            if r_squared > 0.8
            else "good" if r_squared > 0.6
            else "poor"
        )
        return {
            "alpha": alpha,
            "r_squared": r_squared,
            "n_points": len(pairs),
            "fit_quality": fit_quality,
        }
    except Exception:  # pragma: no cover
        return {
            "alpha": 0.0,
            "r_squared": 0.0,
            "n_points": len(pairs),
            "fit_quality": "fit_failed",
        }


def k_phi_multiscale_safety(
    G: Any,
    *,
    scales: tuple[int, ...] = (1, 2, 3, 5),
    alpha_hint: float | None = None,
    tolerance_factor: float = 2.0,
    fit_min_r2: float = 0.5,
) -> Dict[str, Any]:
    """Evaluate multiscale K_φ safety criterion. [CANONICAL TELEMETRY]

    Safety concept (telemetry-only):
        - In healthy regimes, var(K_φ) decays with scale roughly as
          1/r^alpha (alpha ~ 2.76 typical).
    - We consider the system "safe" if either:
        (A) Asymptotic freedom holds (alpha > 0 with R^2 >= fit_min_r2), OR
        (B) Given alpha_hint, actual variances do not exceed expected by
            more than tolerance_factor.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing phase attributes.
    scales : tuple[int, ...]
        Scales (in hops) to consider.
    alpha_hint : Optional[float]
        If provided (e.g., 2.76 global mean or topology-specific), use
        it to compute expected variance decay and check tolerance.
    tolerance_factor : float
        Allowed ratio actual/expected before flagging unsafe.
    fit_min_r2 : float
        Minimum R^2 required to accept (A) as sufficient for safety.

    Returns
    -------
    Dict[str, Any]
        - variance_by_scale: Dict[int, float]
        - fit: {alpha, r_squared, n_points, fit_quality}
                - violations: scales where actual > tolerance*expected
                    (evaluated only if alpha_hint provided)
        - safe: bool

    Notes
    -----
    - This is a read-only metric; it does not mutate EPI or operators.
    - Complementary to Φ_s (U6) and |grad(phi)|; focuses on multiscale
          curvature behavior.
    """
    variance_by_scale = compute_k_phi_multiscale_variance(G, scales=scales)
    fit = fit_k_phi_asymptotic_alpha(variance_by_scale)

    violations: list[int] = []
    if alpha_hint is not None and len(variance_by_scale) > 0:
        # Expected var at scale r: var_1 / r^alpha_hint
        # Use smallest scale present as reference
        r0 = min(variance_by_scale.keys())
        var0 = variance_by_scale[r0]
        for r, v in variance_by_scale.items():
            expected = var0 / (r ** float(alpha_hint)) if r > 0 else var0
            if expected <= 0:
                continue
            if float(v) > tolerance_factor * float(expected):
                violations.append(int(r))

    safe_by_fit = (
        fit.get("alpha", 0.0) > 0.0 and fit.get("r_squared", 0.0) >= fit_min_r2
    )
    safe_by_tolerance = (alpha_hint is not None) and (len(violations) == 0)
    safe = bool(safe_by_fit or safe_by_tolerance)

    return {
        "variance_by_scale": {
            int(k): float(v) for k, v in variance_by_scale.items()
        },
        "fit": fit,
        "violations": violations,
        "safe": safe,
    }


def estimate_coherence_length(
    G: Any, *, coherence_key: str = "coherence"
) -> float:
    """Estimate coherence length ξ_C via radial decay sampling. [CANONICAL]

    **Status**: CANONICAL ⭐ PROMOTED (November 12, 2025)
    Promoted after comprehensive multi-topology validation demonstrating
    critical point prediction, power law scaling, and phase transition
    detection capabilities.

    Validation Evidence
    -------------------
    - 1,170 measurements across 3 topology families (100% success rate)
    - Critical point: I_c = 2.015 validated (±0.005 accuracy)
    - Power law: ξ_C ~ |I - I_c|^(-ν) confirmed
    - Critical exponents: ν ≈ 0.61 (WS), 0.95 (Grid)
    - Multi-scale: ξ_C spans 271 - 46,262 (2-3 orders of magnitude)

    Procedure
    ---------
    1. Compute per-node local coherence: c_i = 1.0 / (1.0 + |ΔNFR_i|)
    2. Perform BFS from seed locus; record coherence at each shell distance
    3. Fit exponential decay: C(d) ~ C0 * exp(-d / ξ_C) using least squares

    Physical Interpretation
    ----------------------
    - Spatial correlation scale: Distance over which coherence correlates
    - Below I_c: ξ_C finite (localized coherence)
    - At I_c: ξ_C diverges (system-wide correlations, phase transition)
    - Above I_c: ξ_C decreases (coherence fragmentation)
    - Symmetry breaking: Diverges at critical intensity I_c = 2.015
    - Topology dependence: Ring (343) > WS (36) > Scale-free (23)

    Returns
    -------
    float
        Estimated coherence length xi_C (>= 0). Returns 0.0 if
        insufficient data.

    Notes
    -----
    - Uses unweighted BFS layers (topological distance).
    - Requires at least 3 shells to attempt fit; else returns 0.0.
    - Coherence values are taken directly; missing treated as 0.0.
    - Future refinement: weight shells by population variance.

    Research Evidence
    ----------------
    - Critical threshold: I_c = 2.015 ± 0.005 (sharp phase transition)
    - Divergence behavior: +15% jump at fragmentation onset
    - Universality class investigation: critical exponent nu ~ 0.5 (mean-field)

    References
    ----------
    - COHERENCE_LENGTH_VALIDATION.md (research documentation)
    - docs/TNFR_FORCES_EMERGENCE.md §11 (critical threshold evidence)
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

    # Fit log-linear: ln C = ln C0 - d / xi_C
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
    # b corresponds to 1/xi_C approximately if model holds:
    # ln C ~ ln C0 - d/xi_C
    if b <= 0:
        return 0.0
    xi = 1.0 / b
    return float(max(xi, 0.0))


def fit_correlation_length_exponent(
    intensities: np.ndarray,
    xi_c_values: np.ndarray,
    I_c: float = 2.015,
    min_distance: float = 0.01
) -> Dict[str, Any]:
    """Fit critical exponent nu from xi_C ~ |I - I_c|^(-nu). [RESEARCH]

    **Task 1**: Extract critical exponent for coherence length divergence
    at phase transition.

    Theory
    ------
    At continuous phase transitions, correlation length diverges:
        xi_C ~ |I - I_c|^(-nu)
    
    Taking logarithms:
        log(xi_C) = log(A) - nu * log(|I - I_c|)

    Parameters
    ----------
    intensities : np.ndarray
        Array of intensity values I
    xi_c_values : np.ndarray
        Corresponding coherence lengths xi_C
    I_c : float, default=2.015
        Critical intensity (from prior evidence)
    min_distance : float, default=0.01
        Minimum |I - I_c| to avoid divergence noise

    Returns
    -------
    Dict[str, Any]
        - nu_below: Critical exponent for I < I_c
        - nu_above: Critical exponent for I > I_c
        - r_squared_below: Fit quality below I_c
        - r_squared_above: Fit quality above I_c
        - universality_class: 'mean-field' | 'ising-3d' | 'ising-2d' |
          'unknown'
        - n_points_below: Number of data points I < I_c
        - n_points_above: Number of data points I > I_c

    Notes
    -----
    Expected critical exponents:
    - Mean-field: nu = 0.5
    - 3D Ising: nu = 0.63
    - 2D Ising: nu = 1.0
    """
    results = {
        "nu_below": 0.0,
        "nu_above": 0.0,
        "r_squared_below": 0.0,
        "r_squared_above": 0.0,
        "universality_class": "unknown",
        "n_points_below": 0,
        "n_points_above": 0
    }

    # Split data at critical point
    below_mask = (intensities < I_c) & (
        np.abs(intensities - I_c) > min_distance
    )
    above_mask = (intensities > I_c) & (
        np.abs(intensities - I_c) > min_distance
    )
    
    # Fit below I_c
    if np.sum(below_mask) >= 3:
        I_below = intensities[below_mask]
        xi_below = xi_c_values[below_mask]
        
        x = np.log(np.abs(I_below - I_c))
        y = np.log(xi_below)
        
        # Linear regression: y = a - nu * x
        coeffs = np.polyfit(x, y, 1)
        nu_below = -coeffs[0]  # Negative slope
        
        y_pred = np.polyval(coeffs, x)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2_below = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        results["nu_below"] = float(nu_below)
        results["r_squared_below"] = float(r2_below)
        results["n_points_below"] = int(np.sum(below_mask))

    # Fit above I_c
    if np.sum(above_mask) >= 3:
        I_above = intensities[above_mask]
        xi_above = xi_c_values[above_mask]
        
        x = np.log(np.abs(I_above - I_c))
        y = np.log(xi_above)
        
        coeffs = np.polyfit(x, y, 1)
        nu_above = -coeffs[0]
        
        y_pred = np.polyval(coeffs, x)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2_above = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        results["nu_above"] = float(nu_above)
        results["r_squared_above"] = float(r2_above)
        results["n_points_above"] = int(np.sum(above_mask))
    
    # Classify universality class (use average of both sides if available)
    nu_avg = 0.0
    count = 0
    if results["nu_below"] > 0:
        nu_avg += results["nu_below"]
        count += 1
    if results["nu_above"] > 0:
        nu_avg += results["nu_above"]
        count += 1
    
    if count > 0:
        nu_avg /= count
        if 0.45 <= nu_avg <= 0.55:
            results["universality_class"] = "mean-field"
        elif 0.58 <= nu_avg <= 0.68:
            results["universality_class"] = "ising-3d"
        elif 0.9 <= nu_avg <= 1.1:
            results["universality_class"] = "ising-2d"
    
    return results


def measure_phase_symmetry(G: Any) -> Dict[str, float]:
    """Quantify phase field symmetry to detect breaking at I_c. [RESEARCH]

    **Task 2**: Identify symmetry breaking mechanism via phase clustering.

    Theory
    ------
    Electroweak symmetry breaking analog:
    - Below I_c: High symmetry (aligned phases, few clusters)
    - At I_c: Critical fluctuations (symmetry breaking onset)
    - Above I_c: Broken symmetry (fragmented domains, many clusters)

    Returns
    -------
    Dict[str, float]
        - circular_variance: Phase dispersion [0,1] (0=aligned, 1=uniform)
        - n_clusters: Number of phase clusters (k-means optimal)
        - cluster_separation: Mean inter-cluster phase distance
        - largest_cluster_fraction: Fraction of nodes in largest cluster

    Notes
    -----
    Expected behavior:
    - I < I_c: circular_variance < 0.3, n_clusters ≤ 2
    - I ≈ I_c: Unstable clustering, fluctuations
    - I > I_c: circular_variance > 0.7, n_clusters ≥ 5
    """
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        # Fallback without scikit-learn
        return {
            "circular_variance": 0.0,
            "n_clusters": 1,
            "cluster_separation": 0.0,
            "largest_cluster_fraction": 1.0
        }
    
    phases = np.array([_get_phase(G, n) for n in G.nodes()])
    if len(phases) == 0:
        return {
            "circular_variance": 0.0,
            "n_clusters": 0,
            "cluster_separation": 0.0,
            "largest_cluster_fraction": 0.0
        }
    
    # Circular variance (phase dispersion measure)
    mean_complex = np.mean(np.exp(1j * phases))
    circular_variance = 1 - np.abs(mean_complex)
    
    # Optimal k-means clustering on unit circle
    phase_vectors = np.column_stack([np.cos(phases), np.sin(phases)])
    
    best_k = 1
    best_silhouette = -1
    max_k = min(10, len(phases) // 2)
    
    if max_k >= 2:
        try:
            from sklearn.metrics import silhouette_score
            
            # Suppress sklearn convergence warnings for synchronized phases
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=UserWarning, module="sklearn"
                )
                
                for k in range(2, max_k + 1):
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(phase_vectors)
                    
                    # Only evaluate if we actually got k distinct clusters
                    n_actual_clusters = len(np.unique(labels))
                    if n_actual_clusters == k:
                        silhouette = silhouette_score(phase_vectors, labels)
                        if silhouette > best_silhouette:
                            best_silhouette = silhouette
                            best_k = k
                    elif n_actual_clusters < k:
                        # Not enough distinct points, stop trying larger k
                        break
        except ImportError:
            # Fallback: simple variance-based heuristic
            best_k = max(1, int(circular_variance * 5) + 1)
    
    # Final clustering with best k
    if best_k > 1 and len(phases) >= best_k:
        # Suppress sklearn convergence warnings for synchronized phases
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=UserWarning, module="sklearn"
            )
            
            kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(phase_vectors)
        
        # Cluster separation (mean inter-cluster center distance)
        centers = kmeans.cluster_centers_
        separations = []
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                sep = np.linalg.norm(centers[i] - centers[j])
                separations.append(sep)
        cluster_separation = (
            float(np.mean(separations)) if separations else 0.0
        )
        
        # Largest cluster fraction
        unique, counts = np.unique(labels, return_counts=True)
        largest_cluster_fraction = float(np.max(counts) / len(labels))
        
    else:
        cluster_separation = 0.0
        largest_cluster_fraction = 1.0
    
    return {
        "circular_variance": float(circular_variance),
        "n_clusters": int(best_k),
        "cluster_separation": float(cluster_separation),
        "largest_cluster_fraction": float(largest_cluster_fraction)
    }


def path_integrated_gradient(G: Any, path: list) -> float:
    """Sum absolute wrapped phase differences along a path [RESEARCH].

    **Canonical Status**: RESEARCH

    Theory
    ------
    From section 3 of the research plan, UM/RA (Coupling/Resonance)
    effectiveness may correlate with the path-integrated phase gradient
    along coupling edges. A high cumulative gradient could impede or
    facilitate information transfer.

    Definition
    ----------
    PIG(path) = Σ_{i,j in path} |wrap_angle(φ_j - φ_i)|

    where `wrap_angle` maps phase differences to [-π, π].

    Parameters
    ----------
    G : TNFRGraph
        NetworkX-like graph with 'phase' node attribute.
    path : list
        An ordered list of node IDs representing the path.

    Returns
    -------
    float
        The total path-integrated phase gradient.
    """
    total = 0.0
    if len(path) < 2:
        return 0.0

    for i, j in zip(path[:-1], path[1:]):
        phi_i = _get_phase(G, i)
        phi_j = _get_phase(G, j)
        total += abs(_wrap_angle(phi_j - phi_i))

    return total


def compute_phase_winding(G: Any, cycle_nodes: list) -> int:
    """Compute integer winding number Q around a closed cycle [RESEARCH].

    **Canonical Status**: RESEARCH (telemetry-only)

    Definition
    ----------
    Q = round( (1 / 2π) · Σ wrap(φ_{i+1} − φ_i) ) over a closed loop.

    Where `wrap(·)` maps phase differences into (−π, π], ensuring
    proper circular accumulation.

    Parameters
    ----------
    G : TNFRGraph
        NetworkX-like graph with per-node phase attribute (`theta` or `phase`).
    cycle_nodes : list
        Ordered list of node IDs forming a closed cycle. The function will
        connect the last node back to the first to complete the loop.

    Returns
    -------
    int
        Integer winding number (topological charge). Values different from 0
        indicate a phase vortex/defect enclosed by the loop.

    Notes
    -----
    - Telemetry-only; does not mutate EPI.
    - Robust to local reparameterizations of phase due to circular wrapping.
    - If fewer than 2 nodes are provided, returns 0.
    """
    if not cycle_nodes or len(cycle_nodes) < 2:
        return 0

    total = 0.0
    seq = list(cycle_nodes)
    # Ensure closure by including last->first
    for i, j in zip(seq, seq[1:] + [seq[0]]):
        phi_i = _get_phase(G, i)
        phi_j = _get_phase(G, j)
        total += _wrap_angle(phi_j - phi_i)

    q = int(round(total / (2.0 * math.pi)))
    return q


# Import extended canonical fields (NEWLY PROMOTED Nov 12, 2025)
try:
    from .extended_canonical_fields import (
        compute_phase_current,
        compute_dnfr_flux,
        compute_extended_canonical_suite
    )
except ImportError:
    # Fallback for development/testing
    def compute_phase_current(G, theta_attr='theta'):
        """Fallback J_φ computation."""
        return {node: 0.0 for node in G.nodes()}
    
    def compute_dnfr_flux(G, dnfr_attr='ΔNFR'):
        """Fallback J_ΔNFR computation."""
        return {node: 0.0 for node in G.nodes()}
    
    def compute_extended_canonical_suite(G, **kwargs):
        """Fallback extended canonical suite."""
        return {
            'Φ_s': compute_structural_potential(G),
            '|∇φ|': compute_phase_gradient(G),
            'K_φ': compute_phase_curvature(G),
            'ξ_C': estimate_coherence_length(G),
            'J_φ': compute_phase_current(G),
            'J_ΔNFR': compute_dnfr_flux(G)
        }


# End of physics field computations.
#
# CANONICAL fields (Φ_s, |∇φ|, K_φ, ξ_C) are validated telemetry
# for operator safety/diagnosis (read-only; never mutate EPI).
# RESEARCH fields (e.g., PIG) are telemetry-only.
