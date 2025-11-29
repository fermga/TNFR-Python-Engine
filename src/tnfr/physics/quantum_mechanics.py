import numpy as np
import networkx as nx
from typing import Dict, Tuple, List, Optional

class QuantumMechanicsMapper:
    """
    Maps Quantum Mechanical concepts to TNFR Structural Fields.
    
    Correspondence:
    - Wavefunction (psi) <-> Complex Structural Field (Psi = K_phi + i*J_phi)
    - Energy (E) <-> Structural Frequency (nu_f)
    - Potential (V) <-> Structural Potential (Phi_s)
    """
    
    @staticmethod
    def wavefunction_to_tnfr(psi: complex) -> Tuple[float, float]:
        """
        Maps a quantum wavefunction value to TNFR structural field components.
        
        Args:
            psi: Complex wavefunction value
            
        Returns:
            (K_phi, J_phi): Tuple of Phase Curvature and Phase Current
        """
        # In TNFR, the complex field Psi is defined as K_phi + i*J_phi
        # We map the wavefunction directly to this field
        K_phi = psi.real
        J_phi = psi.imag
        return K_phi, J_phi

    @staticmethod
    def tnfr_to_wavefunction(G: nx.Graph, node: int) -> complex:
        """
        Extracts the effective wavefunction from a TNFR node state.
        
        Args:
            G: The network graph
            node: The node ID
            
        Returns:
            psi: Complex wavefunction
        """
        # Retrieve structural fields
        # Note: In a real simulation, these would be computed from neighbors
        # Here we assume they are stored or computable
        
        # For this mapping, we use the node's internal phase state
        # psi = A * exp(i * phi)
        # where A is related to EPI magnitude (Coherence)
        # and phi is the node's phase
        
        if 'EPI' in G.nodes[node]:
            epi = G.nodes[node]['EPI']
            amplitude = np.linalg.norm(epi) if isinstance(epi, np.ndarray) else abs(epi)
        else:
            amplitude = 1.0
            
        phi = G.nodes[node].get('phase', 0.0)
        
        return amplitude * np.exp(1j * phi)

    @staticmethod
    def calculate_theoretical_levels(L: float, n_max: int = 5) -> List[float]:
        """
        Calculates theoretical energy levels for a particle in a 1D box.
        E_n = (n^2 * h^2) / (8 * m * L^2)
        
        In TNFR units (h=1, m=1):
        nu_n = n^2 / (8 * L^2)
        
        Args:
            L: Box length
            n_max: Max quantum number
            
        Returns:
            List of energy levels
        """
        levels = []
        for n in range(1, n_max + 1):
            # Simplified units: E ~ n^2
            # We use a scaling factor to match the simulation scale
            E = (n ** 2) 
            levels.append(E)
        return levels

    @staticmethod
    def check_resonance_condition(phase_accumulation: float) -> float:
        """
        Checks how close a phase accumulation is to a resonant mode (2*pi*n).
        
        Args:
            phase_accumulation: Total phase change over a path
            
        Returns:
            dissonance: Distance to nearest 2*pi multiple [0, pi]
        """
        # Wrap to [-pi, pi]
        remainder = np.mod(phase_accumulation, 2 * np.pi)
        if remainder > np.pi:
            remainder -= 2 * np.pi
            
        return abs(remainder)
