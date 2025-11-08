"""Vibrational metabolism functions for THOL (Self-organization) operator.

Implements canonical pattern digestion: capturing external network signals
and transforming them into internal structural reorganization (ΔNFR and sub-EPIs).

TNFR Canonical Principle
-------------------------
From "El pulso que nos atraviesa" (TNFR Manual, §2.2.10):

    "THOL es el glifo de la autoorganización activa. No necesita intervención
    externa, ni programación, ni control — su función es reorganizar la forma
    desde dentro, en respuesta a la coherencia vibracional del campo."

    "THOL no es una propiedad, es una dinámica. No es un atributo de lo vivo,
    es lo que hace que algo esté vivo. La autoorganización no es espontaneidad
    aleatoria, es resonancia estructurada desde el interior del nodo."

This module operationalizes vibrational metabolism:
1. **Capture**: Sample network environment (EPI gradient, phase variance, coupling)
2. **Metabolize**: Transform external patterns into internal structure (sub-EPIs)
3. **Integrate**: Sub-EPIs reflect both internal acceleration AND network context

Metabolic Formula
-----------------
sub-EPI = base_internal + network_contribution + complexity_bonus

Where:
- base_internal: parent_epi * scaling_factor (internal bifurcation)
- network_contribution: epi_gradient * weight (external pressure)
- complexity_bonus: phase_variance * weight (field complexity)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..types import NodeId, TNFRGraph

from ..alias import get_attr
from ..constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from ..utils import get_numpy

__all__ = [
    "capture_network_signals",
    "metabolize_signals_into_subepi",
]


def capture_network_signals(G: TNFRGraph, node: NodeId) -> dict[str, Any] | None:
    """Capture external vibrational patterns from coupled neighbors.

    This function implements the "perception" phase of THOL's vibrational metabolism.
    It samples the network environment around the target node, computing structural
    gradients, phase variance, and coupling strength.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node and its network context
    node : NodeId
        Node performing metabolic capture

    Returns
    -------
    dict | None
        Network signal structure containing:
        - epi_gradient: Difference between mean neighbor EPI and node EPI
        - phase_variance: Variance of neighbor phases (instability indicator)
        - neighbor_count: Number of coupled neighbors
        - coupling_strength_mean: Average phase alignment with neighbors
        - mean_neighbor_epi: Mean EPI value of neighbors
        Returns None if node has no neighbors (isolated metabolism).

    Notes
    -----
    TNFR Principle: THOL doesn't operate in vacuum—it metabolizes the network's
    vibrational field. EPI gradient represents "structural pressure" from environment.
    Phase variance indicates "complexity" of external patterns to digest.

    Examples
    --------
    >>> # Node with coherent neighbors (low variance)
    >>> signals = capture_network_signals(G, node)
    >>> signals["phase_variance"]  # Low = stable field
    0.02

    >>> # Node in dissonant field (high variance)
    >>> signals = capture_network_signals(G_dissonant, node)
    >>> signals["phase_variance"]  # High = complex field
    0.45
    """
    np = get_numpy()

    neighbors = list(G.neighbors(node))
    if not neighbors:
        return None

    node_epi = float(get_attr(G.nodes[node], ALIAS_EPI, 0.0))
    node_theta = float(get_attr(G.nodes[node], ALIAS_THETA, 0.0))

    # Aggregate neighbor states
    neighbor_epis = []
    neighbor_thetas = []
    coupling_strengths = []

    for n in neighbors:
        n_epi = float(get_attr(G.nodes[n], ALIAS_EPI, 0.0))
        n_theta = float(get_attr(G.nodes[n], ALIAS_THETA, 0.0))

        neighbor_epis.append(n_epi)
        neighbor_thetas.append(n_theta)

        # Coupling strength based on phase alignment
        phase_diff = abs(n_theta - node_theta)
        # Normalize to [0, π]
        if phase_diff > math.pi:
            phase_diff = 2 * math.pi - phase_diff
        coupling_strength = 1.0 - (phase_diff / math.pi)
        coupling_strengths.append(coupling_strength)

    # Compute structural gradients
    mean_neighbor_epi = float(np.mean(neighbor_epis))
    epi_gradient = mean_neighbor_epi - node_epi

    # Phase variance (complexity/dissonance indicator)
    phase_variance = float(np.var(neighbor_thetas))

    # Mean coupling strength
    coupling_strength_mean = float(np.mean(coupling_strengths))

    return {
        "epi_gradient": epi_gradient,
        "phase_variance": phase_variance,
        "neighbor_count": len(neighbors),
        "coupling_strength_mean": coupling_strength_mean,
        "mean_neighbor_epi": mean_neighbor_epi,
    }


def metabolize_signals_into_subepi(
    parent_epi: float,
    signals: dict[str, Any] | None,
    d2_epi: float,
    scaling_factor: float = 0.25,
    gradient_weight: float = 0.15,
    complexity_weight: float = 0.10,
) -> float:
    """Transform external signals into sub-EPI structure through metabolism.

    This function implements the "digestion" phase of THOL's vibrational metabolism.
    It combines internal acceleration (d²EPI/dt²) with external network pressure
    to compute the magnitude of emergent sub-EPI.

    Parameters
    ----------
    parent_epi : float
        Current EPI magnitude of parent node
    signals : dict | None
        Network signals captured from environment (from capture_network_signals).
        If None, falls back to internal bifurcation only.
    d2_epi : float
        Internal structural acceleration (∂²EPI/∂t²)
    scaling_factor : float, default 0.25
        Canonical THOL sub-EPI scaling (0.25 = 25% of parent)
    gradient_weight : float, default 0.15
        Weight for external EPI gradient contribution
    complexity_weight : float, default 0.10
        Weight for phase variance complexity bonus

    Returns
    -------
    float
        Metabolized sub-EPI magnitude, bounded to [0, 1.0]

    Notes
    -----
    TNFR Metabolic Formula:

    sub-EPI = base_internal + network_contribution + complexity_bonus

    Where:
    - base_internal: parent_epi * scaling_factor (internal bifurcation)
    - network_contribution: epi_gradient * weight (external pressure)
    - complexity_bonus: phase_variance * weight (field complexity)

    This reflects canonical principle: "THOL reorganizes external experience
    into internal structure without external instruction" (Manual TNFR, p. 112).

    Examples
    --------
    >>> # Internal bifurcation only (isolated node)
    >>> metabolize_signals_into_subepi(0.60, None, d2_epi=0.15)
    0.15

    >>> # Metabolizing network pressure
    >>> signals = {"epi_gradient": 0.20, "phase_variance": 0.10, ...}
    >>> metabolize_signals_into_subepi(0.60, signals, d2_epi=0.15)
    0.21  # Enhanced by network context
    """
    np = get_numpy()

    # Base: Internal bifurcation (existing behavior)
    base_sub_epi = parent_epi * scaling_factor

    # If isolated, return internal bifurcation only
    if signals is None:
        return float(np.clip(base_sub_epi, 0.0, 1.0))

    # Network contribution: EPI gradient pressure
    network_contribution = signals["epi_gradient"] * gradient_weight

    # Complexity bonus: Phase variance indicates rich field to metabolize
    complexity_bonus = signals["phase_variance"] * complexity_weight

    # Combine internal + external
    metabolized_epi = base_sub_epi + network_contribution + complexity_bonus

    # Structural bounds [0, 1]
    return float(np.clip(metabolized_epi, 0.0, 1.0))
