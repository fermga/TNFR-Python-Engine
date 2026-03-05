"""TNFR Unified Derived Fields — Single Source of Truth

This module is the **authoritative implementation** for all derived
structural fields that combine two or more canonical / extended fields.

Canonical base fields (computed elsewhere):
    Φ_s, |∇φ|, K_φ, ξ_C  →  canonical.py
    J_φ, J_ΔNFR            →  extended.py

Conservation laws (computed elsewhere):
    ρ, ∂ρ/∂t, Noether Q, energy E, Ward, Lyapunov  →  conservation.py

Derived fields defined HERE (single location, no duplicates):
    Complex Geometric Field  Ψ = K_φ + i·J_φ  (geometry-transport unification)
    Chirality Field          χ = |∇φ|·K_φ − J_φ·J_ΔNFR  (handedness)
    Symmetry Breaking Field  𝒮 = (|∇φ|² − K_φ²) + (J_φ² − J_ΔNFR²)
    Coherence Coupling Field 𝒞 = Φ_s · |Ψ|  (multi-scale connector)
    Energy Density           ℰ = Φ_s² + |∇φ|² + K_φ² + J_φ² + J_ΔNFR²
    Action Density           𝒜 = Φ_s·|∇φ| + K_φ·J_φ + |∇φ|·J_ΔNFR  (bilinear)
    Topological Charge       𝒬 = |∇φ|·J_φ − K_φ·J_ΔNFR  (topological invariant)

Physics Foundation:
    K_φ and J_φ exhibit strong anticorrelation r ≈ −0.85 to −0.997, indicating
    they are dual aspects of a single complex geometric object Ψ.  This module
    formalises that duality and all derived quantities that emerge from it.
"""

from __future__ import annotations

from ..mathematics.unified_numerical import np
from typing import Dict, Any

from ..constants.canonical import PHI  # noqa: F401 — used by fallback stubs

try:
    import networkx as nx
except ImportError:
    nx = None

# Import canonical fields (Layer 1)
from .canonical import (
    compute_structural_potential,
    compute_phase_gradient,
    compute_phase_curvature,
)
from .extended import (
    compute_phase_current,
    compute_dnfr_flux,
)


# ============================================================================
# COMPLEX GEOMETRIC FIELD  Ψ = K_φ + i·J_φ
# ============================================================================

def compute_complex_geometric_field(G: Any) -> Dict[Any, complex]:
    """Compute unified complex geometric field Ψ = K_φ + i·J_φ.

    K_φ (curvature) and J_φ (current) show strong anticorrelation
    (r ≈ −0.85 to −0.997), indicating they are dual aspects of a single
    complex geometric object.

    - Real part (K_φ):  Static geometric confinement
    - Imaginary part (J_φ):  Dynamic transport flow
    - Magnitude |Ψ|:  Total geometric-transport intensity
    - Phase arg(Ψ):  Balance between geometry and transport

    Parameters
    ----------
    G : NetworkX graph
        Network with 'theta'/'phase' attributes on nodes.

    Returns
    -------
    Dict[node_id, complex]
        Complex field values Ψ(i) = K_φ(i) + i·J_φ(i).
    """
    k_phi = compute_phase_curvature(G)
    j_phi = compute_phase_current(G)
    return {node: complex(k_phi[node], j_phi[node]) for node in G.nodes()}


def compute_field_magnitude(complex_field: Dict[Any, complex]) -> Dict[Any, float]:
    """Compute magnitude |Ψ| of complex field."""
    return {node: abs(value) for node, value in complex_field.items()}


def compute_field_phase(complex_field: Dict[Any, complex]) -> Dict[Any, float]:
    """Compute phase angle arg(Ψ) of complex field."""
    return {node: float(np.angle(value)) for node, value in complex_field.items()}


# ============================================================================
# EMERGENT FIELDS
# ============================================================================

def compute_chirality_field(G: Any) -> Dict[Any, float]:
    """Compute chirality field χ = |∇φ|·K_φ − J_φ·J_ΔNFR.

    High |χ| indicates chiral patterns and broken mirror symmetry.
    """
    grad_phi = compute_phase_gradient(G)
    k_phi = compute_phase_curvature(G)
    j_phi = compute_phase_current(G)
    j_dnfr = compute_dnfr_flux(G)
    return {
        n: grad_phi[n] * k_phi[n] - j_phi[n] * j_dnfr[n]
        for n in G.nodes()
    }


def compute_symmetry_breaking_field(G: Any) -> Dict[Any, float]:
    """Compute symmetry breaking field 𝒮 = (|∇φ|² − K_φ²) + (J_φ² − J_ΔNFR²).

    Quantifies imbalance between conjugate field pairs.  Signals
    phase transitions and reorganisation events.
    """
    grad_phi = compute_phase_gradient(G)
    k_phi = compute_phase_curvature(G)
    j_phi = compute_phase_current(G)
    j_dnfr = compute_dnfr_flux(G)
    return {
        n: (grad_phi[n] ** 2 - k_phi[n] ** 2) + (j_phi[n] ** 2 - j_dnfr[n] ** 2)
        for n in G.nodes()
    }


def compute_coherence_coupling_field(G: Any) -> Dict[Any, float]:
    """Compute coherence coupling field 𝒞 = Φ_s · |Ψ|.

    Connects global structural potential with local geometry-transport
    intensity.  Predicts multi-scale coupling strength.
    """
    phi_s = compute_structural_potential(G)
    psi = compute_complex_geometric_field(G)
    return {n: phi_s[n] * abs(psi[n]) for n in G.nodes()}


# ============================================================================
# TENSOR INVARIANTS
# ============================================================================

def compute_energy_density(G: Any) -> Dict[Any, float]:
    """Compute energy density ℰ = Φ_s² + |∇φ|² + K_φ² + J_φ² + J_ΔNFR².

    Quadratic invariant — analogous to electromagnetic energy density.
    Consistent with the energy functional E = ½Σℰ in conservation.py.
    """
    phi_s = compute_structural_potential(G)
    grad_phi = compute_phase_gradient(G)
    k_phi = compute_phase_curvature(G)
    j_phi = compute_phase_current(G)
    j_dnfr = compute_dnfr_flux(G)
    return {
        n: (phi_s[n] ** 2 + grad_phi[n] ** 2 + k_phi[n] ** 2
            + j_phi[n] ** 2 + j_dnfr[n] ** 2)
        for n in G.nodes()
    }


def compute_action_density(G: Any) -> Dict[Any, float]:
    """Compute action density 𝒜 = Φ_s·|∇φ| + K_φ·J_φ + |∇φ|·J_ΔNFR.

    Bilinear field interactions — related to Lagrangian action.
    """
    phi_s = compute_structural_potential(G)
    grad_phi = compute_phase_gradient(G)
    k_phi = compute_phase_curvature(G)
    j_phi = compute_phase_current(G)
    j_dnfr = compute_dnfr_flux(G)
    return {
        n: (phi_s[n] * grad_phi[n]
            + k_phi[n] * j_phi[n]
            + grad_phi[n] * j_dnfr[n])
        for n in G.nodes()
    }


def compute_topological_charge(G: Any) -> Dict[Any, float]:
    """Compute topological charge 𝒬 = |∇φ|·J_φ − K_φ·J_ΔNFR.

    Topological invariant — conserved under continuous deformations.
    Characterises vortex structures and topological defects.
    """
    grad_phi = compute_phase_gradient(G)
    k_phi = compute_phase_curvature(G)
    j_phi = compute_phase_current(G)
    j_dnfr = compute_dnfr_flux(G)
    return {
        n: grad_phi[n] * j_phi[n] - k_phi[n] * j_dnfr[n]
        for n in G.nodes()
    }


# ============================================================================
# COMPREHENSIVE UNIFIED ANALYSIS
# ============================================================================

def compute_unified_field_suite(G: Any) -> Dict[str, Any]:
    """Compute complete unified field analysis.

    Returns all derived fields, tensor invariants, and conservation
    measures in a single comprehensive call.  Conservation quantities
    delegate to :mod:`tnfr.physics.conservation`.

    Returns
    -------
    Dict[str, Any]
        - ``psi_magnitude``, ``psi_phase``: Complex field Ψ
        - ``chirality``, ``symmetry_breaking``, ``coherence_coupling``
        - ``energy_density``, ``action_density``, ``topological_charge``
        - ``charge_density``, ``current_j_phi``, ``current_j_dnfr``
        - ``conservation_metrics``: {noether_charge, structural_energy}
    """
    from .conservation import (
        compute_charge_density as _rho,
        compute_noether_charge,
        compute_energy_functional,
    )

    results: Dict[str, Any] = {}

    # Complex geometric field
    psi = compute_complex_geometric_field(G)
    results['psi_magnitude'] = compute_field_magnitude(psi)
    results['psi_phase'] = compute_field_phase(psi)

    # Emergent fields
    results['chirality'] = compute_chirality_field(G)
    results['symmetry_breaking'] = compute_symmetry_breaking_field(G)
    results['coherence_coupling'] = compute_coherence_coupling_field(G)

    # Tensor invariants
    results['energy_density'] = compute_energy_density(G)
    results['action_density'] = compute_action_density(G)
    results['topological_charge'] = compute_topological_charge(G)

    # Conservation quantities (canonical source)
    results['charge_density'] = _rho(G)
    j_phi = compute_phase_current(G)
    j_dnfr = compute_dnfr_flux(G)
    results['current_j_phi'] = j_phi
    results['current_j_dnfr'] = j_dnfr

    # Scalar conservation diagnostics
    results['conservation_metrics'] = {
        'noether_charge': compute_noether_charge(G),
        'structural_energy': compute_energy_functional(G),
    }

    return results


# ============================================================================
# ANALYSIS UTILITIES
# ============================================================================

def analyze_field_correlations(
    results: Dict[str, Dict[Any, float]],
) -> Dict[str, float]:
    """Pairwise Pearson correlations between node-level fields in *results*.

    Useful for verifying theoretical predictions (e.g. K_φ ↔ J_φ
    anticorrelation).
    """
    fields: Dict[str, Any] = {}
    first_field = next(
        (v for v in results.values() if isinstance(v, dict)), None
    )
    if first_field is None:
        return {}
    sample_nodes = list(first_field.keys())

    for name, data in results.items():
        if isinstance(data, dict) and sample_nodes[0] in data:
            fields[name] = np.array([data[n] for n in sample_nodes])

    correlations: Dict[str, float] = {}
    names = list(fields.keys())
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            if i < j:
                r = np.corrcoef(fields[n1], fields[n2])[0, 1]
                correlations[f'{n1}_vs_{n2}'] = float(r) if not np.isnan(r) else 0.0
    return correlations


def summary_statistics(
    results: Dict[str, Dict[Any, float]],
) -> Dict[str, Dict[str, float]]:
    """Summary statistics (mean, std, min, max, range) per field."""
    stats: Dict[str, Any] = {}
    for name, data in results.items():
        if isinstance(data, dict):
            vals = [v for v in data.values() if isinstance(v, (int, float))]
            if vals:
                arr = np.array(vals)
                stats[name] = {
                    'mean': float(np.mean(arr)),
                    'std': float(np.std(arr)),
                    'min': float(np.min(arr)),
                    'max': float(np.max(arr)),
                    'range': float(np.max(arr) - np.min(arr)),
                }
    return stats
