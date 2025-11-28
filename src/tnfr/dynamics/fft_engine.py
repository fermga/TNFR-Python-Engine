"""
TNFR FFT-Accelerated Dynamics Engine

Implements FFT-based optimizations for TNFR dynamics that emerge naturally
from the mathematical structure of the nodal equation:

∂EPI/∂t = νf · ΔNFR(t)

Key insights:
1. Graph Laplacian operations → Spectral domain multiplication
2. Phase coupling dynamics → Convolution operations  
3. Temporal evolution → Frequency domain integration
4. Multi-scale coherence → Wavelet-like decomposition

Status: EXPERIMENTAL → CANONICAL TRANSITION
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union
from dataclasses import dataclass
import time

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx = None

# Import existing modules
try:
    from ..mathematics.spectral import get_laplacian_spectrum, gft, igft
    from .structural_cache import get_structural_cache, StructuralCacheEntry
    HAS_SPECTRAL = True
except ImportError:
    HAS_SPECTRAL = False

# Import cache system
try:
    from ..utils.cache import cache_tnfr_computation, CacheLevel
    _CACHE_AVAILABLE = True
except ImportError:
    _CACHE_AVAILABLE = False

# Import FFT cache coordinator
try:
    from .fft_cache_coordinator import FFTCacheCoordinator, get_fft_cache_coordinator
    HAS_FFT_CACHE = True
except ImportError:
    HAS_FFT_CACHE = False
    FFTCacheCoordinator = None


@dataclass
class FFTDynamicsState:
    """State container for FFT-accelerated dynamics."""
    spectral_epi: np.ndarray
    spectral_phase: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    time: float = 0.0
    dt: float = 0.01
    

class FFTDynamicsEngine:
    """
    FFT-accelerated TNFR dynamics engine.
    
    Leverages spectral methods to achieve O(N log N) complexity 
    for many TNFR operations that would otherwise be O(N²) or O(N³).
    """
    
    def __init__(
        self, 
        enable_caching: bool = True,
        cache_coordinator: Optional[FFTCacheCoordinator] = None
    ):
        self.enable_caching = enable_caching
        self.cache_coordinator = (
            cache_coordinator 
            if cache_coordinator is not None 
            else (get_fft_cache_coordinator() if HAS_FFT_CACHE else None)
        )
        
        # Fallback local cache when coordinator unavailable
        self._spectral_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {} if self.cache_coordinator is None else {}
        
        # Performance tracking
        self.total_operations = 0
        self.fft_operations = 0
        self.cache_hits = 0
        
        # Integration with structural cache
        self.structural_cache = get_structural_cache()
    
    def preprocess_graph_for_fft(self, G: Any) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess graph to extract spectral basis for FFT operations.
        
        Returns eigenvalues and eigenvectors of the graph Laplacian.
        """
        if not HAS_NETWORKX or not HAS_SPECTRAL or G is None:
            return np.array([]), np.array([])
        
        # Use cache coordinator if available
        if self.cache_coordinator is not None:
            spectral_basis = self.cache_coordinator.get_spectral_basis(G)
            self.cache_hits += 1
            return spectral_basis.eigenvalues, spectral_basis.eigenvectors
            
        # Fallback to local cache
        topology_hash = self.structural_cache.get_topology_hash(G)
        if self.enable_caching and topology_hash in self._spectral_cache:
            self.cache_hits += 1
            return self._spectral_cache[topology_hash]
        
        # Compute spectral decomposition
        eigenvals, eigenvecs = get_laplacian_spectrum(G)
        
        # Cache result locally
        if self.enable_caching:
            self._spectral_cache[topology_hash] = (eigenvals, eigenvecs)
            
        return eigenvals, eigenvecs
    
    def create_fft_state(self, G: Any) -> FFTDynamicsState:
        """
        Create FFT-accelerated state representation of the graph.
        
        Transforms spatial domain node properties into spectral domain.
        """
        if not HAS_NETWORKX or G is None:
            return FFTDynamicsState(
                spectral_epi=np.array([]),
                spectral_phase=np.array([]),
                eigenvalues=np.array([]),
                eigenvectors=np.array([])
            )
            
        # Get spectral basis
        eigenvals, eigenvecs = self.preprocess_graph_for_fft(G)
        
        if len(eigenvals) == 0:
            return FFTDynamicsState(
                spectral_epi=np.array([]),
                spectral_phase=np.array([]),
                eigenvalues=eigenvals,
                eigenvectors=eigenvecs
            )
        
        # Extract spatial domain data
        nodes = list(G.nodes())
        epi_spatial = np.array([G.nodes[node].get('EPI', 0.0) for node in nodes])
        phase_spatial = np.array([G.nodes[node].get('phase', 0.0) for node in nodes])
        
        # Transform to spectral domain using Graph Fourier Transform
        spectral_epi = gft(epi_spatial, eigenvecs)
        spectral_phase = gft(phase_spatial, eigenvecs)
        
        return FFTDynamicsState(
            spectral_epi=spectral_epi,
            spectral_phase=spectral_phase,
            eigenvalues=eigenvals,
            eigenvectors=eigenvecs
        )
    
    def fft_accelerated_step(
        self, 
        G: Any, 
        fft_state: FFTDynamicsState, 
        dt: float
    ) -> FFTDynamicsState:
        """
        Perform one dynamics step using FFT acceleration.
        
        Implements: ∂EPI/∂t = νf · ΔNFR in the spectral domain.
        """
        if not HAS_NETWORKX or len(fft_state.eigenvalues) == 0:
            return fft_state
            
        self.total_operations += 1
        
        # Extract νf values in node order
        nodes = list(G.nodes())
        vf_spatial = np.array([G.nodes[node].get('nu_f', 1.0) for node in nodes])
        
        # Transform νf to spectral domain  
        vf_spectral = gft(vf_spatial, fft_state.eigenvectors)
        
        # Compute ΔNFR in spectral domain (Laplacian operation)
        # In spectral domain: L[f] = eigenvals * f_spectral
        dnfr_spectral = -fft_state.eigenvalues * fft_state.spectral_epi
        
        # Nodal equation in spectral domain: ∂EPI/∂t = νf · ΔNFR
        depi_dt_spectral = vf_spectral * dnfr_spectral
        
        # Forward Euler integration in spectral domain
        new_spectral_epi = fft_state.spectral_epi + dt * depi_dt_spectral
        
        # Phase evolution using coupling dynamics
        new_spectral_phase = self._evolve_spectral_phase(
            G, fft_state, dt, vf_spectral
        )
        
        self.fft_operations += 1
        
        return FFTDynamicsState(
            spectral_epi=new_spectral_epi,
            spectral_phase=new_spectral_phase,
            eigenvalues=fft_state.eigenvalues,
            eigenvectors=fft_state.eigenvectors,
            time=fft_state.time + dt,
            dt=dt
        )
    
    def _evolve_spectral_phase(
        self, 
        G: Any, 
        fft_state: FFTDynamicsState,
        dt: float,
        vf_spectral: np.ndarray
    ) -> np.ndarray:
        """
        Evolve phase in spectral domain using coupling dynamics.
        
        Implements: ∂θ/∂t = νf + K·sin(θ_j - θ_i) 
        """
        # Transform current phase back to spatial for nonlinear coupling
        phase_spatial = igft(fft_state.spectral_phase, fft_state.eigenvectors)
        
        # Simple phase evolution: θ(t+dt) = θ(t) + νf*dt
        vf_spatial = igft(vf_spectral, fft_state.eigenvectors)
        new_phase_spatial = phase_spatial + dt * vf_spatial.real
        
        # Add coupling effects (simplified Kuramoto dynamics)
        if HAS_NETWORKX and G is not None:
            coupling_strength = 0.1
            nodes = list(G.nodes())
            
            for i, node in enumerate(nodes):
                coupling_sum = 0.0
                degree = 0
                
                for neighbor in G.neighbors(node):
                    if neighbor in nodes:
                        j = nodes.index(neighbor)
                        phase_diff = phase_spatial[j] - phase_spatial[i]
                        coupling_sum += np.sin(phase_diff)
                        degree += 1
                
                if degree > 0:
                    coupling_effect = coupling_strength * coupling_sum / degree
                    new_phase_spatial[i] += dt * coupling_effect
        
        # Wrap phases and transform back to spectral domain
        new_phase_spatial = np.arctan2(
            np.sin(new_phase_spatial), 
            np.cos(new_phase_spatial)
        )
        
        return gft(new_phase_spatial, fft_state.eigenvectors)
    
    def reconstruct_graph_from_fft(
        self, 
        G: Any, 
        fft_state: FFTDynamicsState
    ) -> None:
        """
        Reconstruct spatial domain graph from FFT state.
        
        Updates graph node properties from spectral representation.
        """
        if not HAS_NETWORKX or G is None or len(fft_state.eigenvalues) == 0:
            return
            
        # Transform back to spatial domain
        epi_spatial = igft(fft_state.spectral_epi, fft_state.eigenvectors)
        phase_spatial = igft(fft_state.spectral_phase, fft_state.eigenvectors)
        
        # Update graph nodes
        nodes = list(G.nodes())
        for i, node in enumerate(nodes):
            G.nodes[node]['EPI'] = float(epi_spatial[i].real)
            G.nodes[node]['phase'] = float(phase_spatial[i].real)
    
    @cache_tnfr_computation(
        level=CacheLevel.DERIVED_METRICS, 
        dependencies={"fft_dynamics"}
    ) if _CACHE_AVAILABLE else lambda **kwargs: lambda f: f
    def run_fft_simulation(
        self, 
        G: Any, 
        num_steps: int, 
        dt: float = 0.01,
        return_trajectory: bool = False
    ) -> Dict[str, Any]:
        """
        Run complete FFT-accelerated simulation.
        
        Returns performance metrics and final state.
        """
        if not HAS_NETWORKX or G is None:
            return {"status": "error", "message": "Invalid graph"}
            
        start_time = time.perf_counter()
        
        # Initialize FFT state
        fft_state = self.create_fft_state(G)
        
        # Store trajectory if requested
        trajectory = []
        if return_trajectory:
            # Record initial state
            initial_coherence = self._compute_spectral_coherence(fft_state)
            trajectory.append({
                "time": 0.0,
                "coherence": initial_coherence,
                "energy": self._compute_spectral_energy(fft_state)
            })
        
        # Run simulation steps
        for step in range(num_steps):
            fft_state = self.fft_accelerated_step(G, fft_state, dt)
            
            if return_trajectory and step % 10 == 0:  # Sample every 10 steps
                coherence = self._compute_spectral_coherence(fft_state)
                energy = self._compute_spectral_energy(fft_state)
                trajectory.append({
                    "time": fft_state.time,
                    "coherence": coherence,
                    "energy": energy
                })
        
        # Reconstruct final graph state
        self.reconstruct_graph_from_fft(G, fft_state)
        
        simulation_time = time.perf_counter() - start_time
        
        # Performance metrics
        results = {
            "status": "success",
            "simulation_time": simulation_time,
            "total_steps": num_steps,
            "fft_operations": self.fft_operations,
            "cache_hits": self.cache_hits,
            "steps_per_second": num_steps / simulation_time if simulation_time > 0 else 0,
            "final_time": fft_state.time,
            "final_coherence": self._compute_spectral_coherence(fft_state)
        }
        
        if return_trajectory:
            results["trajectory"] = trajectory
            
        return results
    
    def _compute_spectral_coherence(self, fft_state: FFTDynamicsState) -> float:
        """Compute coherence measure in spectral domain."""
        if len(fft_state.spectral_phase) == 0:
            return 0.0
            
        # Transform phase to spatial domain for coherence computation
        phase_spatial = igft(fft_state.spectral_phase, fft_state.eigenvectors)
        
        # Kuramoto order parameter
        z = np.mean(np.exp(1j * phase_spatial))
        return float(np.abs(z))
    
    def _compute_spectral_energy(self, fft_state: FFTDynamicsState) -> float:
        """Compute total energy in spectral domain."""
        if len(fft_state.spectral_epi) == 0:
            return 0.0
            
        # Energy = ||EPI||²
        energy = np.sum(np.abs(fft_state.spectral_epi) ** 2)
        return float(energy)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            "total_operations": self.total_operations,
            "fft_operations": self.fft_operations,  
            "cache_hits": self.cache_hits,
            "cache_hit_rate": self.cache_hits / max(1, self.total_operations),
            "cached_spectra": len(self._spectral_cache),
            "fft_usage_ratio": self.fft_operations / max(1, self.total_operations)
        }


# Factory functions for easy access
def create_fft_engine(**kwargs) -> FFTDynamicsEngine:
    """Create FFT dynamics engine."""
    return FFTDynamicsEngine(**kwargs)


def run_fft_optimized_simulation(
    G: Any, 
    num_steps: int, 
    dt: float = 0.01, 
    **kwargs
) -> Dict[str, Any]:
    """Convenience function for FFT-optimized simulation."""
    engine = create_fft_engine()
    return engine.run_fft_simulation(G, num_steps, dt, **kwargs)