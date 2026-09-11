"""TNFR Unified Derived Fields — Single Source of Truth

This module is the **authoritative implementation** for all derived
structural fields that combine two or more canonical / extended fields.

Canonical base fields (computed elsewhere):
    Φ_s, |∇φ|, K_φ, ξ_C  →  canonical.py
    J_φ, J_ΔNFR            →  extended.py

Finite balance diagnostics (computed elsewhere):
    ρ, ∂ρ/∂t, historical Q, energy candidate, Ward residual  →  conservation.py

Derived snapshot fields defined HERE (single location, no duplicates):
    Complex coordinate       Ψ = K_φ + i·J_φ
    Historical chirality     χ = |∇φ|·K_φ − J_φ·J_ΔNFR  (bilinear)
    Historical symmetry      𝒮 = (|∇φ|² − K_φ²) + (J_φ² − J_ΔNFR²)
    Coherence coupling       𝒞 = Φ_s · |Ψ|  (product coordinate)
    Energy Density           ℰ = Φ_s² + |∇φ|² + K_φ² + J_φ² + J_ΔNFR²
    Action Density           𝒜 = Φ_s·|∇φ| + K_φ·J_φ + |∇φ|·J_ΔNFR  (bilinear)
    Historical Q Density    𝒬 = |∇φ|·J_φ − K_φ·J_ΔNFR  (bilinear snapshot)

Algebraic scope:
    Ψ combines K_φ and J_φ as real and imaginary coordinates by definition.
    Their measured correlation depends on graph topology, phase state, and
    sampling; no universal correlation range is assumed by these formulas.
    Prefer ``compute_historical_q_density``. The public
    ``compute_topological_charge`` name is retained for compatibility; its
    bilinear output is not an integer winding, a homotopy invariant, or a
    conserved quantity without separate trajectory evidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ..mathematics.unified_numerical import np

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

    - Real part: ``K_φ``
    - Imaginary part: ``J_φ``
    - Magnitude: the Euclidean norm ``sqrt(K_φ² + J_φ²)``
    - Phase: the coordinate angle ``atan2(J_φ, K_φ)``, undefined at ``Ψ=0``

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
    """Compute NumPy's principal coordinate angle ``arg(Ψ)``.

    At ``Ψ=0`` the mathematical phase is undefined; NumPy returns ``0.0`` by
    convention, so callers requiring a phase map must check nonzero support.
    """
    return {node: float(np.angle(value)) for node, value in complex_field.items()}


# ============================================================================
# DERIVED BILINEAR FIELDS
# ============================================================================


def compute_chirality_field(G: Any) -> dict[Any, float]:
    """Compute chirality field χ = |∇φ|·K_φ − J_φ·J_ΔNFR.

    ``χ`` is a signed bilinear snapshot coordinate. A physical chirality or
    broken-symmetry interpretation requires a declared transformation test.
    """
    grad_phi = compute_phase_gradient(G)
    k_phi = compute_phase_curvature(G)
    j_phi = compute_phase_current(G)
    j_dnfr = compute_dnfr_flux(G)
    return _chirality_field(grad_phi, k_phi, j_phi, j_dnfr)


def compute_symmetry_breaking_field(G: Any) -> dict[Any, float]:
    """Compute symmetry breaking field 𝒮 = (|∇φ|² − K_φ²) + (J_φ² − J_ΔNFR²).

    Quantifies a same-snapshot imbalance between the declared coordinate
    pairs. It does not by itself establish a phase transition.
    """
    grad_phi = compute_phase_gradient(G)
    k_phi = compute_phase_curvature(G)
    j_phi = compute_phase_current(G)
    j_dnfr = compute_dnfr_flux(G)
    return _symmetry_breaking_field(grad_phi, k_phi, j_phi, j_dnfr)


def compute_coherence_coupling_field(G: Any) -> dict[Any, float]:
    """Compute coherence coupling field 𝒞 = Φ_s · |Ψ|.

    Multiplies structural potential by local geometry-transport intensity.
    Predictive meaning must be established on a declared trajectory or dataset.
    """
    phi_s = compute_structural_potential(G)
    psi = compute_complex_geometric_field(G)
    return _coherence_coupling_field(phi_s, psi)


# ============================================================================
# SNAPSHOT QUADRATIC AND BILINEAR READ-OUTS
# ============================================================================


def compute_energy_density(G: Any) -> dict[Any, float]:
    r"""Compute the raw quadratic structural-energy read-out per node.

    .. math::

        \mathcal{E}(i) = \Phi_s^2 + |\nabla\phi|^2 + K_\phi^2
                         + J_\phi^2 + J_{\Delta NFR}^2

    This is the **unnormalised** quadratic form shared by the listed
    finite-snapshot diagnostics:

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
    r"""Compute the historical action-density bilinear per node.

    .. math::

        \mathcal{A}(i) = \Phi_s \cdot |\nabla\phi|
                         + K_\phi \cdot J_\phi
                         + |\nabla\phi| \cdot J_{\Delta NFR}

    This is the shared implementation of the bilinear snapshot coordinate.
    Its historical variational interpretation is an interaction term in a
    declared auxiliary model; the formula alone supplies no engine action.

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


def compute_historical_q_density(G: Any) -> dict[Any, float]:
    """Compute historical Q density 𝒬 = |∇φ|·J_φ − K_φ·J_ΔNFR.

    This is a continuous bilinear graph snapshot, not the integer winding of a
    phase map. It has no general deformation or trajectory-conservation
    guarantee.
    """
    grad_phi = compute_phase_gradient(G)
    k_phi = compute_phase_curvature(G)
    j_phi = compute_phase_current(G)
    j_dnfr = compute_dnfr_flux(G)
    return _topological_charge(grad_phi, k_phi, j_phi, j_dnfr)


def compute_topological_charge(G: Any) -> dict[Any, float]:
    """Compatibility alias for :func:`compute_historical_q_density`."""
    return compute_historical_q_density(G)


# ============================================================================
# AGGREGATED SINGLE-SNAPSHOT READ-OUT
# ============================================================================


def compute_unified_field_suite(G: Any) -> dict[str, Any]:
    """Compute the aggregated structural read-out for one graph snapshot.

    Returns derived fields, quadratic/bilinear coordinates, and finite balance
    diagnostics from one local collection of the five required base fields.
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
        - ``energy_density``, ``action_density``, ``historical_q_density``
        - ``topological_charge``: legacy alias of ``historical_q_density``
        - ``charge_density``, ``current_j_phi``, ``current_j_dnfr``
        - ``conservation_metrics``: historical charge and structural-energy
          snapshot totals; neither key asserts temporal conservation
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

    # Snapshot quadratic and bilinear read-outs
    results["energy_density"] = _energy_density_from_fields(
        phi_s, grad_phi, k_phi, j_phi, j_dnfr
    )
    results["action_density"] = _action_density_from_fields(
        phi_s, grad_phi, k_phi, j_phi, j_dnfr
    )
    q_density = _topological_charge(grad_phi, k_phi, j_phi, j_dnfr)
    results["historical_q_density"] = q_density
    results["topological_charge"] = dict(q_density)

    # Finite balance coordinates (historical public keys retained)
    results["charge_density"] = _charge_density_from_fields(phi_s, k_phi)
    results["current_j_phi"] = j_phi
    results["current_j_dnfr"] = j_dnfr

    # Scalar snapshot totals
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
    """Return descriptive Pearson correlations between node-level read-outs.

    Correlations are sample statistics for the supplied snapshot. Their sign
    and magnitude are not universal identities of the field definitions.
    Pairs with fewer than two samples or a constant member are omitted because
    Pearson correlation is undefined there.
    """
    fields: dict[str, Any] = {}
    first_field = next(
        (v for v in results.values() if isinstance(v, dict) and v), None
    )
    if first_field is None:
        return {}
    sample_nodes = list(first_field.keys())

    for name, data in results.items():
        if not isinstance(data, dict) or set(data) != set(sample_nodes):
            continue
        values = np.asarray([data[node] for node in sample_nodes])
        if np.iscomplexobj(values):
            continue
        try:
            real_values = np.asarray(values, dtype=float)
        except (TypeError, ValueError):
            continue
        if real_values.ndim == 1 and bool(np.all(np.isfinite(real_values))):
            fields[name] = real_values

    correlations: dict[str, float] = {}
    names = list(fields.keys())
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            if i < j:
                left, right = fields[n1], fields[n2]
                if (
                    left.size < 2
                    or float(np.std(left)) == 0.0
                    or float(np.std(right)) == 0.0
                ):
                    continue
                raw = float(np.corrcoef(left, right)[0, 1])
                if math.isfinite(raw):
                    correlations[f"{n1}_vs_{n2}"] = raw
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
