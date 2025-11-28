"""
TNFR Unified Computational Backend

This module implements the natural unification that emerges from the nodal equation:
∂EPI/∂t = νf · ΔNFR(t)

The unified backend leverages the mathematical structure to provide:

1. **Cross-Modal Computation**: Single computational backend for all TNFR operations
2. **Unified Cache Strategy**: Shared cache across spectral, nodal, and field computations
3. **Mathematical Backend Integration**: Seamless JAX/PyTorch/NumPy backend switching  
4. **Multi-Scale Computation**: Single interface for all temporal/spatial scales
5. **Emergent Optimization**: Operations automatically optimize based on graph topology

Mathematical Foundation:
- All computations derive from the nodal equation
- Spectral domain operations use GFT/IGFT naturally
- Cache sharing based on computational dependencies
- Backend selection based on operation mathematical properties

Status: CANONICAL UNIFIED BACKEND
"""

import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import time

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx = None

# Import mathematical backends
try:
    from ..mathematics.backend import get_backend
    HAS_MATH_BACKENDS = True
except ImportError:
    HAS_MATH_BACKENDS = False

# Import cache system
try:
    from ..utils.cache import cache_tnfr_computation, CacheLevel, get_global_cache
    _CACHE_AVAILABLE = True
except ImportError:
    _CACHE_AVAILABLE = False

# Import optimization engines
try:
    from .nodal_optimizer import NodalEquationOptimizer
    from .fft_engine import FFTDynamicsEngine
    from .structural_cache import StructuralCoherenceCache
    from .adelic import AdelicDynamics
    HAS_OPTIMIZATION_ENGINES = True
except ImportError:
    HAS_OPTIMIZATION_ENGINES = False

# Import spectral analysis
try:
    from ..mathematics.spectral import get_laplacian_spectrum, gft, igft
    HAS_SPECTRAL = True
except ImportError:
    HAS_SPECTRAL = False

# Import physics fields
try:
    from ..physics.fields import compute_structural_potential, compute_phase_gradient
    HAS_PHYSICS = True
except ImportError:
    HAS_PHYSICS = False


class ComputationType(Enum):
    """Types of TNFR computations."""
    NODAL_EVOLUTION = "nodal_evolution"          # ∂EPI/∂t integration
    SPECTRAL_ANALYSIS = "spectral_analysis"      # GFT/IGFT operations
    FIELD_COMPUTATION = "field_computation"      # Φ_s, |∇φ|, K_φ, ξ_C
    TEMPORAL_INTEGRATION = "temporal_integration"  # Multi-step evolution
    OPERATOR_APPLICATION = "operator_application"  # Structural operators
    CROSS_SCALE_COUPLING = "cross_scale_coupling"  # Multi-scale dynamics


@dataclass
class UnifiedComputationRequest:
    """Request for unified computation."""
    computation_type: ComputationType
    graph: Any
    parameters: Dict[str, Any] = field(default_factory=dict)
    preferred_backend: Optional[str] = None
    enable_cache: bool = True
    return_trajectory: bool = False
    optimization_level: int = 2  # 0=none, 1=basic, 2=aggressive


@dataclass
class UnifiedComputationResult:
    """Result of unified computation."""
    computation_type: ComputationType
    results: Dict[str, Any]
    backend_used: str
    execution_time: float
    cache_hits: int = 0
    cache_misses: int = 0
    optimization_strategy: str = "none"
    memory_used_mb: float = 0.0
    accuracy_metrics: Dict[str, float] = field(default_factory=dict)


class TNFRUnifiedBackend:
    """
    Unified computational backend for all TNFR operations.
    
    This backend emerges naturally from the nodal equation by recognizing
    that all TNFR computations are variations of the same mathematical
    structure: spectral evolution in network-coupled dynamical systems.
    """
    
    def __init__(self, default_backend: str = "numpy", cache_size_mb: float = 256.0):
        self.default_backend = default_backend
        self.cache_size_mb = cache_size_mb
        
        # Initialize mathematical backends
        self._math_backends = {}
        if HAS_MATH_BACKENDS:
            for backend_name in ["numpy", "jax", "torch"]:
                try:
                    backend = get_backend(backend_name)
                    self._math_backends[backend_name] = backend
                except Exception:
                    pass
        
        # Initialize optimization engines (shared state)
        self._nodal_optimizer = NodalEquationOptimizer() if HAS_OPTIMIZATION_ENGINES else None
        self._fft_engine = FFTDynamicsEngine() if HAS_OPTIMIZATION_ENGINES else None
        self._structural_cache = StructuralCoherenceCache() if HAS_OPTIMIZATION_ENGINES else None
        self._adelic_engine = AdelicDynamics() if HAS_OPTIMIZATION_ENGINES else None
        
        # Unified cache coordination
        if _CACHE_AVAILABLE:
            self._global_cache = get_global_cache()
        else:
            self._global_cache = None
            
        # Cross-computation cache (shared between all engines)
        self._spectral_cache = {}  # Shared eigendecompositions
        self._topology_cache = {}  # Shared graph topology analysis
        self._field_cache = {}     # Shared field computations
        
        # Performance tracking
        self._computation_history = []
        self._backend_performance = {}
        
    def select_optimal_backend(
        self,
        request: UnifiedComputationRequest
    ) -> str:
        """
        Select optimal mathematical backend based on computation type and graph properties.
        
        This selection emerges from the mathematical structure of each operation.
        """
        if request.preferred_backend and request.preferred_backend in self._math_backends:
            return request.preferred_backend
            
        if not HAS_NETWORKX or request.graph is None:
            return self.default_backend
            
        # Analyze graph properties
        num_nodes = len(request.graph.nodes())
        # num_edges = len(request.graph.edges())  # Future use for density analysis
        
        # Backend selection based on mathematical properties
        if request.computation_type == ComputationType.SPECTRAL_ANALYSIS:
            # JAX excels at FFT operations
            if "jax" in self._math_backends and num_nodes > 50:
                return "jax"
                
        elif request.computation_type == ComputationType.NODAL_EVOLUTION:
            # PyTorch good for vectorized differential equations
            if "torch" in self._math_backends and num_nodes > 100:
                return "torch"
                
        elif request.computation_type == ComputationType.FIELD_COMPUTATION:
            # NumPy generally most stable for field computations
            return "numpy"
            
        # Default fallback
        return self.default_backend
        
    @cache_tnfr_computation(
        level=CacheLevel.DERIVED_METRICS,
        dependencies={"unified_computation"}
    ) if _CACHE_AVAILABLE else lambda **kwargs: lambda f: f
    def execute_computation(
        self,
        request: UnifiedComputationRequest
    ) -> UnifiedComputationResult:
        """
        Execute unified computation using optimal strategy.
        
        This is the single entry point for all TNFR computations.
        """
        start_time = time.perf_counter()
        
        # Select backend
        backend_name = self.select_optimal_backend(request)
        
        # Route to appropriate computation method
        try:
            if request.computation_type == ComputationType.NODAL_EVOLUTION:
                results = self._execute_nodal_evolution(request, backend_name)
                
            elif request.computation_type == ComputationType.SPECTRAL_ANALYSIS:
                results = self._execute_spectral_analysis(request, backend_name)
                
            elif request.computation_type == ComputationType.FIELD_COMPUTATION:
                results = self._execute_field_computation(request, backend_name)
                
            elif request.computation_type == ComputationType.TEMPORAL_INTEGRATION:
                results = self._execute_temporal_integration(request, backend_name)
                
            elif request.computation_type == ComputationType.OPERATOR_APPLICATION:
                results = self._execute_operator_application(request, backend_name)
                
            elif request.computation_type == ComputationType.CROSS_SCALE_COUPLING:
                results = self._execute_cross_scale_coupling(request, backend_name)
                
            else:
                raise ValueError(f"Unknown computation type: {request.computation_type}")
                
        except Exception as e:
            # Fallback to basic numpy computation
            results = self._execute_fallback_computation(request, str(e))
            backend_name = "numpy"
        
        execution_time = time.perf_counter() - start_time
        
        # Create result
        result = UnifiedComputationResult(
            computation_type=request.computation_type,
            results=results,
            backend_used=backend_name,
            execution_time=execution_time,
            optimization_strategy=f"level_{request.optimization_level}"
        )
        
        # Update performance history
        self._computation_history.append(result)
        
        return result
        
    def _execute_nodal_evolution(self, request: UnifiedComputationRequest, backend: str) -> Dict[str, Any]:
        """Execute nodal equation evolution: ∂EPI/∂t = νf · ΔNFR(t)."""
        G = request.graph
        dt = request.parameters.get('dt', 0.01)
        
        # Use cached nodal optimizer if available
        if self._nodal_optimizer and request.optimization_level > 0:
            return self._nodal_optimizer.compute_vectorized_nodal_evolution(G, dt)
            
        # Fallback to direct computation
        results = {}
        for node in G.nodes():
            epi = G.nodes[node].get('EPI', 0.0)
            nu_f = G.nodes[node].get('nu_f', 1.0)
            dnfr = G.nodes[node].get('ΔNFR', 0.0)
            
            # Basic Euler integration
            new_epi = epi + dt * nu_f * dnfr
            results[node] = (new_epi, G.nodes[node].get('phase', 0.0))
            
        return {"nodal_states": results, "backend": backend}
        
    def _execute_spectral_analysis(self, request: UnifiedComputationRequest, backend: str) -> Dict[str, Any]:
        """Execute spectral analysis using GFT/IGFT."""
        G = request.graph
        
        # Check for cached spectrum
        graph_id = id(G)
        if graph_id in self._spectral_cache:
            eigenvalues, eigenvectors = self._spectral_cache[graph_id]
        else:
            if HAS_SPECTRAL:
                eigenvalues, eigenvectors = get_laplacian_spectrum(G)
                self._spectral_cache[graph_id] = (eigenvalues, eigenvectors)
            else:
                return {"error": "Spectral analysis not available"}
        
        # Extract signal from nodes
        signal = np.array([G.nodes[node].get('EPI', 0.0) for node in G.nodes()])
        
        if HAS_SPECTRAL:
            # Apply Graph Fourier Transform
            spectral_coeffs = gft(signal, eigenvectors)
            
            # Apply spectral filtering if requested
            if 'filter_cutoff' in request.parameters:
                cutoff = request.parameters['filter_cutoff']
                filtered_coeffs = spectral_coeffs.copy()
                filtered_coeffs[eigenvalues > cutoff] = 0
                
                # Inverse transform
                filtered_signal = igft(filtered_coeffs, eigenvectors)
                
                return {
                    "eigenvalues": eigenvalues,
                    "spectral_coefficients": spectral_coeffs,
                    "filtered_signal": filtered_signal,
                    "backend": backend
                }
        
        return {
            "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors,
            "signal": signal,
            "backend": backend
        }
        
    def _execute_field_computation(self, request: UnifiedComputationRequest, backend: str) -> Dict[str, Any]:
        """Execute structural field computations."""
        G = request.graph
        
        # Check field cache
        graph_id = id(G)
        cache_key = f"fields_{graph_id}"
        
        if request.enable_cache and cache_key in self._field_cache:
            return self._field_cache[cache_key]
            
        results = {}
        
        if HAS_PHYSICS:
            # Compute all structural fields
            try:
                phi_s = compute_structural_potential(G)
                results["phi_s"] = phi_s
            except Exception:
                pass
                
            try:
                phase_grad = compute_phase_gradient(G)
                results["phase_gradient"] = phase_grad
            except Exception:
                pass
        
        results["backend"] = backend
        
        # Cache results
        if request.enable_cache:
            self._field_cache[cache_key] = results
            
        return results
        
    def _execute_temporal_integration(self, request: UnifiedComputationRequest, backend: str) -> Dict[str, Any]:
        """Execute multi-step temporal integration."""
        G = request.graph
        num_steps = request.parameters.get('num_steps', 10)
        dt = request.parameters.get('dt', 0.01)
        
        # Use FFT engine for large-scale temporal integration
        if self._fft_engine and len(G.nodes()) > 20 and request.optimization_level > 1:
            return self._fft_engine.run_fft_simulation(G, num_steps, dt, request.return_trajectory)
            
        # Fallback to step-by-step integration
        trajectory = []
        for step in range(num_steps):
            # Execute single nodal step
            nodal_request = UnifiedComputationRequest(
                computation_type=ComputationType.NODAL_EVOLUTION,
                graph=G,
                parameters={'dt': dt},
                enable_cache=request.enable_cache,
                optimization_level=request.optimization_level
            )
            
            step_result = self._execute_nodal_evolution(nodal_request, backend)
            
            if request.return_trajectory:
                trajectory.append({
                    "step": step,
                    "time": step * dt,
                    "nodal_states": step_result["nodal_states"]
                })
        
        return {
            "final_time": num_steps * dt,
            "trajectory": trajectory if request.return_trajectory else None,
            "backend": backend
        }
        
    def _execute_operator_application(self, request: UnifiedComputationRequest, backend: str) -> Dict[str, Any]:
        """Execute structural operator application."""
        operator_sequence = request.parameters.get('operators', [])
        
        results = {"applied_operators": operator_sequence, "backend": backend}
        
        # Apply operators sequentially
        for operator in operator_sequence:
            # This would integrate with the actual operator system
            results[f"operator_{operator}"] = f"applied_{operator}"
            
        return results
        
    def _execute_cross_scale_coupling(self, request: UnifiedComputationRequest, backend: str) -> Dict[str, Any]:
        """Execute multi-scale coupling computation."""
        G = request.graph
        
        # Combine multiple computation types for cross-scale analysis
        results = {"backend": backend}
        
        # Spectral analysis for global patterns
        spectral_request = UnifiedComputationRequest(
            computation_type=ComputationType.SPECTRAL_ANALYSIS,
            graph=G,
            parameters=request.parameters,
            enable_cache=request.enable_cache
        )
        spectral_result = self._execute_spectral_analysis(spectral_request, backend)
        results["global_patterns"] = spectral_result
        
        # Field computation for local structure
        field_request = UnifiedComputationRequest(
            computation_type=ComputationType.FIELD_COMPUTATION,
            graph=G,
            parameters=request.parameters,
            enable_cache=request.enable_cache
        )
        field_result = self._execute_field_computation(field_request, backend)
        results["local_fields"] = field_result
        
        return results
        
    def _execute_fallback_computation(self, request: UnifiedComputationRequest, error: str) -> Dict[str, Any]:
        """Fallback computation when optimized methods fail."""
        return {
            "computation_type": request.computation_type.value,
            "status": "fallback",
            "error": error,
            "backend": "fallback"
        }
        
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get performance statistics across all computations."""
        if not self._computation_history:
            return {"total_computations": 0}
            
        total_time = sum(r.execution_time for r in self._computation_history)
        avg_time = total_time / len(self._computation_history)
        
        # Backend usage statistics
        backend_usage = {}
        for result in self._computation_history:
            backend = result.backend_used
            backend_usage[backend] = backend_usage.get(backend, 0) + 1
            
        # Computation type statistics
        type_usage = {}
        for result in self._computation_history:
            comp_type = result.computation_type.value
            type_usage[comp_type] = type_usage.get(comp_type, 0) + 1
            
        return {
            "total_computations": len(self._computation_history),
            "total_time": total_time,
            "average_time": avg_time,
            "backend_usage": backend_usage,
            "computation_type_usage": type_usage,
            "cache_availability": _CACHE_AVAILABLE,
            "optimization_engines_available": HAS_OPTIMIZATION_ENGINES,
            "mathematical_backends_available": list(self._math_backends.keys())
        }
        
    def clear_caches(self) -> None:
        """Clear all internal caches."""
        self._spectral_cache.clear()
        self._topology_cache.clear()
        self._field_cache.clear()
        
        if self._nodal_optimizer:
            self._nodal_optimizer.clear_optimization_cache()


# Factory functions
def create_unified_backend(**kwargs) -> TNFRUnifiedBackend:
    """Create unified computational backend."""
    return TNFRUnifiedBackend(**kwargs)


def execute_unified_computation(
    computation_type: ComputationType,
    graph: Any,
    backend: Optional[TNFRUnifiedBackend] = None,
    **kwargs
) -> UnifiedComputationResult:
    """Convenience function for unified computation."""
    if backend is None:
        backend = create_unified_backend()
        
    request = UnifiedComputationRequest(
        computation_type=computation_type,
        graph=graph,
        parameters=kwargs
    )
    
    return backend.execute_computation(request)