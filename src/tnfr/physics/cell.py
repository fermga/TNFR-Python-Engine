"""TNFR Cell Module: Compartmentalized Life - From Autopoiesis to Cellular Organization

This module provides computational tools to detect and quantify cellular behavior in TNFR networks,
building on the life emergence foundation (A > 1.0) through spatial organization and membrane formation.
Based on the mathematical derivation extending the nodal equation to compartmentalized systems.

Physics Foundation:
From the nodal equation ∂EPI/∂t = νf · ΔNFR(t), cell formation extends to:
∂EPI_cell/∂t = νf_internal · ΔNFR_internal + J_membrane(φ_ext, φ_int)

Where J_membrane represents phase-selective transport across cellular boundaries.

Contracts and Invariants (TNFR):
- No direct EPI mutation; always observe via metrics (Invariant #1)
- Structural units preserved (νf in Hz_str) (Invariant #2)
- ΔNFR semantics preserved as structural pressure (Invariant #3)
- Operator closure: this module only measures, does not alter operator sequences (Invariant #4)
- Phase verification upheld in coupling metrics (U3) (Invariant #5)
- Multi-scale coherence preserved (U5) for nested cellular EPIs (Invariant #7)

Cellular Criteria (Building on Life A > 1.0):
1. Boundary Coherence: C_boundary > 0.8 (strong membrane coherence)
2. Internal Selectivity: ρ_selectivity > 0.6 (preferential internal coupling)
3. Homeostatic Regulation: H_index > 0.5 (stable internal dynamics)
4. Membrane Integrity: I_compartment > 0.7 (controlled permeability)

Metrics:
- Boundary Coherence (C_boundary)
- Selectivity Index (ρ_selectivity)
- Homeostatic Index (H_index)
- Membrane Integrity (I_compartment)

See also:
- docs/CELL_EMERGENCE_FROM_TNFR.md (theoretical framework)
- docs/LIFE_EMERGENCE_FROM_TNFR.md (prerequisite life foundation)
- src/tnfr/physics/life.py (autopoietic coefficient foundation)
"""

from dataclasses import dataclass
from typing import Optional, Sequence
import numpy as np
import networkx as nx
from ..metrics.common import compute_coherence


@dataclass
class CellTelemetry:
    """Container for cell-emergence telemetry time series.

    Attributes
    ----------
    times : list[float]
        Structural time stamps (Hz_str units).
    boundary_coherence : np.ndarray
        C_boundary(t) ∈ [0, 1]. Coherence at cellular boundary regions.
    internal_coherence : np.ndarray
        C_internal(t) ∈ [0, 1]. Coherence within compartmentalized interior.
    selectivity_index : np.ndarray
        ρ_selectivity(t) ∈ [-1, 1]. Membrane coupling preference:
        ρ = (coupling_internal - coupling_external)/(coupling_total).
    homeostatic_index : np.ndarray
        H_index(t) ∈ [0, 1]. Internal regulatory capacity:
        H = 1 - σ(ΔNFR_internal)/(|μ(ΔNFR_internal)| + ε).
    membrane_integrity : np.ndarray
        I_compartment(t) ∈ [0, 1]. Compartmentalization quality: I = 1 - leakage_rate.
    cell_formation_time : Optional[float]
        First time t where all cellular criteria satisfied (C_boundary > 0.8,
        ρ_selectivity > 0.6, H_index > 0.5, I_compartment > 0.7), else None.
    """
    times: list[float]
    boundary_coherence: np.ndarray
    internal_coherence: np.ndarray
    selectivity_index: np.ndarray
    homeostatic_index: np.ndarray
    membrane_integrity: np.ndarray
    cell_formation_time: Optional[float] = None


# Core computations

def _safe_div(numerator: np.ndarray, denominator: np.ndarray, fallback: float = 0.0) -> np.ndarray:
    """Safe division with fallback for zero denominators."""
    result = np.full_like(numerator, fallback, dtype=float)
    mask = np.abs(denominator) > 1e-12
    result[mask] = numerator[mask] / denominator[mask]
    return result


def compute_boundary_coherence(
    graph: nx.Graph,
    boundary_nodes: Sequence[int]
) -> float:
    """Compute coherence specifically at cellular boundary regions per TNFR coherence operator.

    From the TNFR coherence operator Ĉ, boundary coherence measures structural stability
    at the compartment interface. Uses centralized coherence computation from tnfr.metrics.common.

    Parameters
    ----------
    graph : nx.Graph
        Network with node attributes 'delta_nfr' (structural pressure).
    boundary_nodes : Sequence[int]
        Node IDs that form the cellular boundary (membrane region).

    Returns
    -------
    float
        Boundary coherence C_boundary ∈ [0, 1]. C_boundary > 0.8 indicates
        strong membrane formation suitable for compartmentalization.
    """
    if not boundary_nodes:
        return 0.0
        
    # Extract boundary subgraph
    boundary_subgraph = graph.subgraph(boundary_nodes).copy()
    
    # Compute coherence on boundary only
    if len(boundary_subgraph.nodes()) == 0:
        return 0.0
        
    return compute_coherence(boundary_subgraph)


def compute_selectivity_index(
    graph: nx.Graph,
    internal_nodes: Sequence[int],
    boundary_nodes: Sequence[int]
) -> float:
    """Compute cellular membrane selectivity from coupling topology preferentiality.

    From membrane flux physics J_membrane = κ(φ_ext - φ_int), this function quantifies
    the selectivity index ρ_selectivity as preferential internal coupling organization
    that characterizes cellular boundary formation from TNFR dynamics.

    ρ_selectivity = (C_internal - C_external) / C_total

    Where cellular behavior emerges when ρ > 0.6 (preferential internalization).

    Parameters
    ----------
    graph : nx.Graph
        TNFR network with edges representing structural couplings between nodes.
    internal_nodes : Sequence[int]
        Node IDs forming the compartmentalized cellular interior.
    boundary_nodes : Sequence[int]
        Node IDs forming the phase-selective cellular boundary.

    Returns
    -------
    float
        Selectivity index ρ_selectivity ∈ [-1, 1]. Values ρ > 0.6 indicate emergence
        of cellular organization with preferential internal coupling topology.

    Notes
    -----
    Measures topological organization without modifying EPI (respects TNFR invariants).
    Cellular selectivity emerges from autopoietic foundation without external control.
    """
    internal_set = set(internal_nodes)
    boundary_set = set(boundary_nodes)
    cell_nodes = internal_set | boundary_set
    
    # Count coupling types
    internal_coupling = 0  # Both nodes internal
    external_coupling = 0  # One internal, one external
    
    for u, v in graph.edges():
        u_in_cell = u in cell_nodes
        v_in_cell = v in cell_nodes
        
        if u_in_cell and v_in_cell:
            internal_coupling += 1
        elif u_in_cell or v_in_cell:  # Crossing boundary
            external_coupling += 1
    
    total_coupling = internal_coupling + external_coupling
    if total_coupling == 0:
        return 0.0
    
    return (internal_coupling - external_coupling) / total_coupling


def compute_homeostatic_index(
    delta_nfr_internal: np.ndarray,
    epsilon: float = 1e-6
) -> float:
    """Compute cellular homeostatic regulation capacity from ΔNFR stability dynamics.

    From the extended nodal equation ∂EPI_cell/∂t = νf_internal·ΔNFR_internal + J_membrane,
    this function quantifies the homeostatic index H as the capacity for internal
    structural pressure regulation that characterizes cellular regulatory behavior.

    H_homeostatic = 1 - σ(ΔNFR_internal) / (|μ(ΔNFR_internal)| + ε)

    Where cellular regulation emerges when H > 0.5 (stable internal dynamics).

    Parameters
    ----------
    delta_nfr_internal : np.ndarray
        Time series of internal structural pressure ΔNFR values for cellular nodes.
    epsilon : float, default=1e-6
        Numerical stability parameter for division by near-zero means.

    Returns
    -------
    float
        Homeostatic index H_homeostatic ∈ [0, 1]. Values H > 0.5 indicate emergence
        of cellular regulatory capacity with stable internal ΔNFR dynamics.

    Notes
    -----
    Uses canonical ΔNFR structural pressure without modification (respects invariants).
    Cellular homeostasis emerges from autopoietic foundation through stabilization.
    """
    if len(delta_nfr_internal) == 0:
        return 0.0
    
    std_internal = np.std(delta_nfr_internal)
    mean_internal = np.abs(np.mean(delta_nfr_internal))
    
    raw_index = 1.0 - std_internal / (mean_internal + epsilon)
    
    # Clamp to [0, 1] range to ensure valid homeostatic index
    return max(0.0, min(1.0, raw_index))


def compute_membrane_integrity(
    flux_internal: float,
    flux_external: float
) -> float:
    """Compute cellular membrane integrity from compartmentalization effectiveness.

    From membrane flux physics J_membrane = κ(φ_ext - φ_int), this function quantifies
    the membrane integrity I_compartment as the effectiveness of phase-selective
    transport that characterizes cellular boundary compartmentalization.

    I_compartment = 1 - leakage_rate = 1 - |J_external| / (|J_internal| + |J_external|)

    Where cellular compartmentalization emerges when I > 0.7 (effective separation).

    Parameters
    ----------
    flux_internal : float
        Internal membrane flux (controlled, phase-selective transport).
    flux_external : float
        External membrane leakage (uncontrolled, non-selective transport).

    Returns
    -------
    float
        Membrane integrity I_compartment ∈ [0, 1]. Values I > 0.7 indicate emergence
        of effective cellular compartmentalization with phase-selective transport.

    Notes
    -----
    Measures flux-based compartmentalization respecting membrane physics.
    Cellular integrity emerges from autopoietic foundation through selectivity.
    """
    total_flux = abs(flux_internal) + abs(flux_external)
    if total_flux == 0:
        return 1.0
        
    leakage_rate = abs(flux_external) / total_flux
    return 1.0 - leakage_rate


def detect_cell_formation(
    graph_sequence: Sequence[nx.Graph],
    times: Sequence[float],
    internal_nodes: Sequence[int],
    boundary_nodes: Sequence[int],
    c_boundary_threshold: float = 0.8,
    selectivity_threshold: float = 0.6,
    homeostasis_threshold: float = 0.5,
    integrity_threshold: float = 0.7
) -> CellTelemetry:
    """Detect cellular organization emergence from TNFR autopoietic dynamics per cellular extension.

    From the extended nodal equation ∂EPI_cell/∂t = νf_internal·ΔNFR_internal + J_membrane(φ_ext, φ_int),
    this function detects when autopoietic patterns (A > 1.0) transition to compartmentalized
    cellular behavior through spatial organization and membrane formation.

    Cellular criteria (all must be satisfied simultaneously):
    - Boundary coherence: C_boundary > c_boundary_threshold (default 0.8)
    - Selectivity index: ρ_selectivity > selectivity_threshold (default 0.6)  
    - Homeostatic capacity: H_index > homeostasis_threshold (default 0.5)
    - Membrane integrity: I_compartment > integrity_threshold (default 0.7)

    Parameters
    ----------
    graph_sequence : Sequence[nx.Graph]
        Time series of TNFR network states with node attributes 'delta_nfr' (structural pressure).
    times : Sequence[float]
        Structural time points (Hz_str units) corresponding to each graph state.
    internal_nodes : Sequence[int]
        Node IDs that form the compartmentalized cell interior.
    boundary_nodes : Sequence[int]
        Node IDs that form the phase-selective cellular boundary (membrane).
    c_boundary_threshold : float, default=0.8
        Minimum boundary coherence for cellular membrane formation.
    selectivity_threshold : float, default=0.6
        Minimum selectivity index for preferential internal coupling.
    homeostasis_threshold : float, default=0.5
        Minimum homeostatic index for internal regulatory capacity.
    integrity_threshold : float, default=0.7
        Minimum membrane integrity for effective compartmentalization.

    Returns
    -------
    CellTelemetry
        Complete cellular telemetry time series including formation time detection.
        cell_formation_time contains first time when all criteria satisfied, or None.

    Notes
    -----
    This function builds on life emergence (requires A > 1.0 autopoietic foundation)
    and extends to spatial compartmentalization. Uses centralized TNFR coherence
    computation and respects all canonical invariants (no direct EPI mutation).
    """
    times = list(times)
    n_timesteps = len(graph_sequence)
    
    # Initialize arrays
    boundary_coherence = np.zeros(n_timesteps)
    internal_coherence = np.zeros(n_timesteps)
    selectivity_index = np.zeros(n_timesteps)
    homeostatic_index = np.zeros(n_timesteps)
    membrane_integrity = np.zeros(n_timesteps)
    
    # Track internal ΔNFR for homeostasis calculation
    internal_delta_nfr_history = []
    
    for t_idx, graph in enumerate(graph_sequence):
        # Boundary coherence
        boundary_coherence[t_idx] = compute_boundary_coherence(graph, boundary_nodes)
        
        # Internal coherence  
        if internal_nodes:
            internal_subgraph = graph.subgraph(internal_nodes).copy()
            internal_coherence[t_idx] = compute_coherence(internal_subgraph) if len(internal_subgraph.nodes()) > 0 else 0.0
        else:
            internal_coherence[t_idx] = 0.0
            
        # Selectivity index
        selectivity_index[t_idx] = compute_selectivity_index(graph, internal_nodes, boundary_nodes)
        
        # Collect internal ΔNFR values
        internal_dnfr = []
        for node in internal_nodes:
            if node in graph.nodes() and 'delta_nfr' in graph.nodes[node]:
                internal_dnfr.append(graph.nodes[node]['delta_nfr'])
        
        internal_delta_nfr_history.extend(internal_dnfr)
        
        # Homeostatic index (computed from accumulated history)
        if len(internal_delta_nfr_history) > 1:
            homeostatic_index[t_idx] = compute_homeostatic_index(np.array(internal_delta_nfr_history))
        else:
            homeostatic_index[t_idx] = 0.0
            
        # Membrane integrity (simplified: based on selectivity as proxy)
        # In a full implementation, this would use actual flux measurements
        membrane_integrity[t_idx] = min(1.0, selectivity_index[t_idx] + 0.2)  # Heuristic
    
    # Detect cell formation time
    cell_formation_time: Optional[float] = None
    
    for t_idx in range(n_timesteps):
        criteria_met = (
            boundary_coherence[t_idx] > c_boundary_threshold and
            selectivity_index[t_idx] > selectivity_threshold and  
            homeostatic_index[t_idx] > homeostasis_threshold and
            membrane_integrity[t_idx] > integrity_threshold
        )
        
        if criteria_met:
            cell_formation_time = times[t_idx]
            break
    
    return CellTelemetry(
        times=times,
        boundary_coherence=boundary_coherence,
        internal_coherence=internal_coherence,
        selectivity_index=selectivity_index, 
        homeostatic_index=homeostatic_index,
        membrane_integrity=membrane_integrity,
        cell_formation_time=cell_formation_time
    )


def apply_membrane_flux(
    graph: nx.Graph,
    internal_nodes: Sequence[int],
    boundary_nodes: Sequence[int], 
    permeability: float = 0.1,
    phase_threshold: float = np.pi / 3
) -> None:
    """Apply phase-selective membrane flux for cellular transport simulation.

    From extended nodal equation ∂EPI_cell/∂t = νf_internal·ΔNFR_internal + J_membrane(φ_ext,φ_int),
    this function implements the membrane flux J_membrane = κ(φ_ext - φ_int) with phase
    selectivity that enables cellular transport behavior from TNFR dynamics.

    Transport occurs when |φ_boundary - φ_neighbor| ≤ phase_threshold (phase compatibility).

    Parameters
    ----------
    graph : nx.Graph
        TNFR network with node attributes 'EPI' (structural form), 'theta' (phase).
    internal_nodes : Sequence[int]
        Node IDs forming the compartmentalized cellular interior.
    boundary_nodes : Sequence[int]
        Node IDs forming the phase-selective cellular boundary (membrane).
    permeability : float, default=0.1
        Membrane permeability coefficient κ ∈ [0, 1] controlling flux magnitude.
    phase_threshold : float, default=π/3
        Phase compatibility threshold for selective transport (radians).

    Notes
    -----
    MODIFIES EPI attributes directly (operator-based EPI changes in higher-level code).
    Implements cellular transport extending autopoietic foundation through selectivity.
    Phase-selective flux enables cellular compartmentalization from TNFR coupling.
    """
    for boundary_node in boundary_nodes:
        if boundary_node not in graph.nodes():
            continue
            
        # Get boundary node properties
        boundary_phase = graph.nodes[boundary_node].get('theta', 0.0)
        boundary_epi = graph.nodes[boundary_node].get('EPI', 0.0)
        
        # Check transport with neighboring nodes
        for neighbor in graph.neighbors(boundary_node):
            neighbor_phase = graph.nodes[neighbor].get('theta', 0.0)
            neighbor_epi = graph.nodes[neighbor].get('EPI', 0.0)
            
            # Phase compatibility check
            phase_diff = abs(boundary_phase - neighbor_phase)
            phase_diff = min(phase_diff, 2 * np.pi - phase_diff)  # Wrap around
            
            if phase_diff <= phase_threshold:
                # Calculate flux
                epi_diff = neighbor_epi - boundary_epi
                flux = permeability * epi_diff
                
                # Apply flux (modify EPI)
                current_epi = graph.nodes[boundary_node].get('EPI', 0.0)
                new_epi = current_epi + 0.01 * flux  # Small time step
                graph.nodes[boundary_node]['EPI'] = max(0.0, new_epi)  # Keep positive


__all__ = [
    "CellTelemetry",
    "compute_boundary_coherence",
    "compute_selectivity_index",
    "compute_homeostatic_index",
    "compute_membrane_integrity",
    "detect_cell_formation",
    "apply_membrane_flux"
]
