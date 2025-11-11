"""TNFR Physics Module - Structural Field Computations.

This module provides physics-based field computations derived from the TNFR
nodal equation and validated through extensive empirical studies.

Canonical Fields
----------------
**Structural Potential (Φ_s)**: CANONICAL (promoted 2025-11-11)
    - Definition: Φ_s(i) = Σ_{j≠i} ΔNFR_j / d(i,j)^α (α=2)
    - Validation: 2,400+ experiments, corr(Δ Φ_s, ΔC) = -0.822, CV = 0.1%
    - Safety criterion: Δ Φ_s < 2.0 (escape threshold)
    - Grammar: U6 STRUCTURAL POTENTIAL CONFINEMENT
    - Status: Read-only telemetry, passive equilibrium mechanism

Research Fields (NON-CANONICAL)
--------------------------------
**Phase Gradient (|∇φ|)**: Research phase
    - Weak EM-like behavior, corr ≈ -0.13
    - Long-range phase organization

**Phase Curvature (K_φ)**: Research phase
    - Weak strong-like behavior, corr ≈ -0.07
    - Confinement at |K_φ| > 4.88

**Coherence Length (ξ_C)**: Research phase
    - Weak-like symmetry breaking at I_c = 2.015
    - Characteristic scale of stabilizing influence

Physics Foundation
------------------
From the nodal equation:
    ∂EPI/∂t = νf · ΔNFR(t)

Where:
- EPI: Estructura Primaria de Información (coherent structural form)
- νf: Structural frequency (reorganization rate, Hz_str)
- ΔNFR: Reorganization operator (structural pressure)

Modules
-------
fields : Structural field computations
    - compute_structural_potential (CANONICAL)
    - compute_phase_gradient (research)
    - compute_phase_curvature (research)
    - estimate_coherence_length (research)

See Also
--------
tnfr.operators.grammar : Grammar validation including U6
tnfr.dynamics : Nodal equation integration
tnfr.metrics : Coherence and stability metrics

References
----------
- UNIFIED_GRAMMAR_RULES.md § U6: Structural potential confinement
- docs/TNFR_FORCES_EMERGENCE.md § 14-15: Complete Φ_s validation
- AGENTS.md § Structural Fields: Canonical status and criteria
- TNFR.pdf § 2.1: Nodal equation foundation

Examples
--------
>>> from tnfr.physics.fields import compute_structural_potential
>>> import networkx as nx
>>> G = nx.karate_club_graph()
>>> for node in G.nodes():
...     G.nodes[node]['delta_nfr'] = 0.5
>>> phi_s = compute_structural_potential(G, alpha=2.0)
>>> print(f"Potential at node 0: {phi_s[0]:.3f}")

>>> # U6 validation
>>> from tnfr.operators.grammar import validate_structural_potential_confinement
>>> phi_before = compute_structural_potential(G)
>>> # ... apply sequence ...
>>> phi_after = compute_structural_potential(G)
>>> valid, drift, msg = validate_structural_potential_confinement(
...     G, phi_before, phi_after, threshold=2.0
... )
>>> print(f"U6 status: {msg}")

"""

from .fields import (
    compute_structural_potential,
    compute_phase_gradient,
    compute_phase_curvature,
    estimate_coherence_length,
)

__all__ = [
    "compute_structural_potential",
    "compute_phase_gradient",
    "compute_phase_curvature",
    "estimate_coherence_length",
]
