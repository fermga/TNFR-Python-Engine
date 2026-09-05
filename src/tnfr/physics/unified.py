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

Algebraic scope:
    Ψ combines K_φ and J_φ as real and imaginary coordinates by definition.
    Their measured correlation depends on graph topology, phase state, and
    sampling; no universal correlation range is assumed by these formulas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..mathematics.unified_numerical import np

try:
    import networkx as nx
except ImportError:
    nx = None

# Import canonical fields (Layer 1)
from .canonical import (
    compute_phase_curvature,
    compute_phase_gradient,
    compute_structural_potential,
)
from .extended import compute_dnfr_flux, compute_phase_current


@dataclass(frozen=True)
class _StructuralFieldReadout:
    """Owned base-field maps for one synchronous composite readout.

    This object is local to the caller, never persisted on the graph. It does
    not lock the graph against concurrent evolution; callers must hold state
    fixed while reading it, as for the individual field functions.
    """

    phi_s: dict[Any, float]
    grad_phi: dict[Any, float]
    k_phi: dict[Any, float]
    j_phi: dict[Any, float]
    j_dnfr: dict[Any, float]


def _capture_structural_fields(G: Any) -> _StructuralFieldReadout:
    """Read each required kernel once and detach its returned mapping."""
    return _StructuralFieldReadout(
        phi_s=dict(compute_structural_potential(G)),
        grad_phi=dict(compute_phase_gradient(G)),
        k_phi=dict(compute_phase_curvature(G)),
        j_phi=dict(compute_phase_current(G)),
        j_dnfr=dict(compute_dnfr_flux(G)),
    )


def _complex_geometric_field(
    k_phi: dict[Any, float], j_phi: dict[Any, float]
) -> dict[Any, complex]:
    return {n: complex(k_phi[n], j_phi[n]) for n in k_phi}


def _chirality_field(
    grad_phi: dict[Any, float], k_phi: dict[Any, float],
    j_phi: dict[Any, float], j_dnfr: dict[Any, float],
) -> dict[Any, float]:
    return {n: grad_phi[n] * k_phi[n] - j_phi[n] * j_dnfr[n] for n in grad_phi}


def _symmetry_breaking_field(
    grad_phi: dict[Any, float], k_phi: dict[Any, float],
    j_phi: dict[Any, float], j_dnfr: dict[Any, float],
) -> dict[Any, float]:
    return {
        n: (grad_phi[n] ** 2 - k_phi[n] ** 2) + (j_phi[n] ** 2 - j_dnfr[n] ** 2)
        for n in grad_phi
    }


def _coherence_coupling_field(
    phi_s: dict[Any, float], psi: dict[Any, complex]
) -> dict[Any, float]:
    return {n: phi_s[n] * abs(psi[n]) for n in phi_s}


def _energy_density_from_fields(
    phi_s: dict[Any, float], grad_phi: dict[Any, float], k_phi: dict[Any, float],
    j_phi: dict[Any, float], j_dnfr: dict[Any, float],
) -> dict[Any, float]:
    """The shared raw quadratic form for live and captured field readouts."""
    return {
        n: (
            phi_s[n] ** 2 + grad_phi[n] ** 2 + k_phi[n] ** 2
            + j_phi[n] ** 2 + j_dnfr[n] ** 2
        )
        for n in phi_s
    }


def _action_density_from_fields(
    phi_s: dict[Any, float], grad_phi: dict[Any, float], k_phi: dict[Any, float],
    j_phi: dict[Any, float], j_dnfr: dict[Any, float],
) -> dict[Any, float]:
    """The shared bilinear interaction, without another graph read."""
    return {
        n: phi_s[n] * grad_phi[n] + k_phi[n] * j_phi[n] + grad_phi[n] * j_dnfr[n]
        for n in phi_s
    }


def _topological_charge(
    grad_phi: dict[Any, float], k_phi: dict[Any, float],
    j_phi: dict[Any, float], j_dnfr: dict[Any, float],
) -> dict[Any, float]:
    return {n: grad_phi[n] * j_phi[n] - k_phi[n] * j_dnfr[n] for n in grad_phi}

# ============================================================================
# COMPLEX GEOMETRIC FIELD  Ψ = K_φ + i·J_φ
# ============================================================================


def compute_complex_geometric_field(G: Any) -> dict[Any, complex]:
    """Compute unified complex geometric field Ψ = K_φ + i·J_φ.

    This is an algebraic pairing of curvature and current. Their correlation
    is a fixture-dependent measurement, not a prerequisite or a consequence
    of representing the two maps as one complex field.

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
    dict[node_id, complex]
        Complex field values Ψ(i) = K_φ(i) + i·J_φ(i).
    """
    k_phi = compute_phase_curvature(G)
    j_phi = compute_phase_current(G)
    return _complex_geometric_field(k_phi, j_phi)


def compute_field_magnitude(complex_field: dict[Any, complex]) -> dict[Any, float]:
    """Compute magnitude |Ψ| of complex field."""
    return {node: abs(value) for node, value in complex_field.items()}


def compute_field_phase(complex_field: dict[Any, complex]) -> dict[Any, float]:
    """Compute phase angle arg(Ψ) of complex field."""
    return {node: float(np.angle(value)) for node, value in complex_field.items()}


# ============================================================================
# EMERGENT FIELDS
# ============================================================================


def compute_chirality_field(G: Any) -> dict[Any, float]:
    """Compute chirality field χ = |∇φ|·K_φ − J_φ·J_ΔNFR.

    High |χ| indicates chiral patterns and broken mirror symmetry.
    """
    grad_phi = compute_phase_gradient(G)
    k_phi = compute_phase_curvature(G)
    j_phi = compute_phase_current(G)
    j_dnfr = compute_dnfr_flux(G)
    return _chirality_field(grad_phi, k_phi, j_phi, j_dnfr)


def compute_symmetry_breaking_field(G: Any) -> dict[Any, float]:
    """Compute symmetry breaking field 𝒮 = (|∇φ|² − K_φ²) + (J_φ² − J_ΔNFR²).

    Quantifies imbalance between conjugate field pairs.  Signals
    phase transitions and reorganisation events.
    """
    grad_phi = compute_phase_gradient(G)
    k_phi = compute_phase_curvature(G)
    j_phi = compute_phase_current(G)
    j_dnfr = compute_dnfr_flux(G)
    return _symmetry_breaking_field(grad_phi, k_phi, j_phi, j_dnfr)


def compute_coherence_coupling_field(G: Any) -> dict[Any, float]:
    """Compute coherence coupling field 𝒞 = Φ_s · |Ψ|.

    Connects global structural potential with local geometry-transport
    intensity.  Predicts multi-scale coupling strength.
    """
    phi_s = compute_structural_potential(G)
    psi = compute_complex_geometric_field(G)
    return _coherence_coupling_field(phi_s, psi)


# ============================================================================
# TENSOR INVARIANTS
# ============================================================================


def compute_energy_density(G: Any) -> dict[Any, float]:
    r"""Compute the raw quadratic energy density per node (CANONICAL SOURCE).

    .. math::

        \mathcal{E}(i) = \Phi_s^2 + |\nabla\phi|^2 + K_\phi^2
                         + J_\phi^2 + J_{\Delta NFR}^2

    This is the **unnormalised** quadratic form — the single source of
    truth from which all other energy quantities derive:

    +------------------------------------+----------------------------------+
    | Derived quantity                   | Relation to ℰ                   |
    +====================================+==================================+
    | Hamiltonian density (variational)  | H(i) = ½·ℰ(i)                  |
    | Energy functional (conservation)   | E = ½·Σ_i ℰ(i)  = Σ_i H(i)    |
    | Kinetic density (variational)      | T(i) = ½[J_φ² + J_ΔNFR²]      |
    | Potential density (variational)    | V(i) = ½[Φ_s² + |∇φ|² + K_φ²] |
    | Lagrangian density (variational)   | ℒ(i)  = T(i) − V(i)            |
    +------------------------------------+----------------------------------+

    The ½ factor is the conventional Hamiltonian normalisation (like
    ½(E² + B²) in electrodynamics).  This function returns the **raw**
    quadratic form without the ½ so that callers can apply it as needed.

    See Also
    --------
    variational.compute_hamiltonian_density : H(i) = ½·ℰ(i)
    conservation.compute_energy_functional  : E = ½·Σ_i ℰ(i)
    """
    phi_s = compute_structural_potential(G)
    grad_phi = compute_phase_gradient(G)
    k_phi = compute_phase_curvature(G)
    j_phi = compute_phase_current(G)
    j_dnfr = compute_dnfr_flux(G)
    return _energy_density_from_fields(phi_s, grad_phi, k_phi, j_phi, j_dnfr)


def compute_action_density(G: Any) -> dict[Any, float]:
    r"""Compute action density (bilinear coupling) per node (CANONICAL SOURCE).

    .. math::

        \mathcal{A}(i) = \Phi_s \cdot |\nabla\phi|
                         + K_\phi \cdot J_\phi
                         + |\nabla\phi| \cdot J_{\Delta NFR}

    This is the **single source of truth** for the bilinear field
    interaction.  In the variational formulation, 𝒜 represents the
    **interaction Lagrangian** (cross-sector coupling), distinct from
    the free Lagrangian ℒ_free = T − V.

    ``variational.compute_interaction_density()`` delegates here.

    See Also
    --------
    variational.compute_interaction_density : Alias for this function.
    """
    phi_s = compute_structural_potential(G)
    grad_phi = compute_phase_gradient(G)
    k_phi = compute_phase_curvature(G)
    j_phi = compute_phase_current(G)
    j_dnfr = compute_dnfr_flux(G)
    return _action_density_from_fields(phi_s, grad_phi, k_phi, j_phi, j_dnfr)


def compute_topological_charge(G: Any) -> dict[Any, float]:
    """Compute topological charge 𝒬 = |∇φ|·J_φ − K_φ·J_ΔNFR.

    Topological invariant — conserved under continuous deformations.
    Characterises vortex structures and topological defects.
    """
    grad_phi = compute_phase_gradient(G)
    k_phi = compute_phase_curvature(G)
    j_phi = compute_phase_current(G)
    j_dnfr = compute_dnfr_flux(G)
    return _topological_charge(grad_phi, k_phi, j_phi, j_dnfr)


# ============================================================================
# COMPREHENSIVE UNIFIED ANALYSIS
# ============================================================================


def compute_unified_field_suite(G: Any) -> dict[str, Any]:
    """Compute complete unified field analysis.

    Returns all derived fields, tensor invariants, and conservation
    measures from one local collection of the five required base fields.
    Conservation density uses :mod:`tnfr.physics.conservation`; scalar totals
    are reduced from the same returned density maps.

    The caller must hold topology, attributes and configuration fixed during
    the call. Returned maps are detached; capture is not atomic with concurrent
    graph evolution.

    Returns
    -------
    dict[str, Any]
        - ``psi_magnitude``, ``psi_phase``: Complex field Ψ
        - ``chirality``, ``symmetry_breaking``, ``coherence_coupling``
        - ``energy_density``, ``action_density``, ``topological_charge``
        - ``charge_density``, ``current_j_phi``, ``current_j_dnfr``
        - ``conservation_metrics``: {noether_charge, structural_energy}
    """
    from .conservation import _charge_density_from_fields

    fields = _capture_structural_fields(G)
    phi_s, grad_phi, k_phi = fields.phi_s, fields.grad_phi, fields.k_phi
    j_phi, j_dnfr = fields.j_phi, fields.j_dnfr
    results: dict[str, Any] = {}

    # Complex geometric field
    psi = _complex_geometric_field(k_phi, j_phi)
    results["psi_magnitude"] = compute_field_magnitude(psi)
    results["psi_phase"] = compute_field_phase(psi)

    # Emergent fields
    results["chirality"] = _chirality_field(grad_phi, k_phi, j_phi, j_dnfr)
    results["symmetry_breaking"] = _symmetry_breaking_field(
        grad_phi, k_phi, j_phi, j_dnfr
    )
    results["coherence_coupling"] = _coherence_coupling_field(phi_s, psi)

    # Tensor invariants
    results["energy_density"] = _energy_density_from_fields(
        phi_s, grad_phi, k_phi, j_phi, j_dnfr
    )
    results["action_density"] = _action_density_from_fields(
        phi_s, grad_phi, k_phi, j_phi, j_dnfr
    )
    results["topological_charge"] = _topological_charge(grad_phi, k_phi, j_phi, j_dnfr)

    # Conservation quantities (canonical source)
    results["charge_density"] = _charge_density_from_fields(phi_s, k_phi)
    results["current_j_phi"] = j_phi
    results["current_j_dnfr"] = j_dnfr

    # Scalar conservation diagnostics
    results["conservation_metrics"] = {
        "noether_charge": sum(results["charge_density"].values()),
        "structural_energy": 0.5 * sum(results["energy_density"].values()),
    }

    return results


# ============================================================================
# ANALYSIS UTILITIES
# ============================================================================


def analyze_field_correlations(
    results: dict[str, dict[Any, float]],
) -> dict[str, float]:
    """Pairwise Pearson correlations between node-level fields in *results*.

    Useful for verifying theoretical predictions (e.g. K_φ ↔ J_φ
    anticorrelation).
    """
    fields: dict[str, Any] = {}
    first_field = next((v for v in results.values() if isinstance(v, dict)), None)
    if first_field is None:
        return {}
    sample_nodes = list(first_field.keys())

    for name, data in results.items():
        if isinstance(data, dict) and sample_nodes[0] in data:
            fields[name] = np.array([data[n] for n in sample_nodes])

    correlations: dict[str, float] = {}
    names = list(fields.keys())
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            if i < j:
                r = np.corrcoef(fields[n1], fields[n2])[0, 1]
                correlations[f"{n1}_vs_{n2}"] = float(r) if not np.isnan(r) else 0.0
    return correlations


def summary_statistics(
    results: dict[str, dict[Any, float]],
) -> dict[str, dict[str, float]]:
    """Summary statistics (mean, std, min, max, range) per field."""
    stats: dict[str, Any] = {}
    for name, data in results.items():
        if isinstance(data, dict):
            vals = [v for v in data.values() if isinstance(v, (int, float))]
            if vals:
                arr = np.array(vals)
                stats[name] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "range": float(np.max(arr) - np.min(arr)),
                }
    return stats
