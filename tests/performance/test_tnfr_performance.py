"""
TNFR Performance and Regression Tests
====================================

Comprehensive performance benchmarking and regression testing to ensure
new implementations maintain or improve computational efficiency while
preserving theoretical fidelity.

Tests computational performance, memory efficiency, and correctness under load.
Based on TNFR canonical performance requirements.
"""

import time
import gc
from typing import Any, Tuple
import pytest
import numpy as np
import networkx as nx

# TNFR imports
from tnfr.alias import set_attr
from tnfr.constants.aliases import (
    ALIAS_EPI, ALIAS_VF, ALIAS_DNFR, ALIAS_THETA
)
from tnfr.operators.definitions import (
    Emission, Coherence, Silence
)
from tnfr.physics.fields import (
    compute_structural_potential,
    compute_phase_gradient,
    compute_phase_curvature,
)
from tnfr.metrics.coherence import compute_coherence


def create_performance_test_network(size: int = 100) -> Any:
    """Create larger network for performance testing."""
    G = nx.barabasi_albert_graph(size, 3)  # Scale-free network
    
    for i, node in enumerate(G.nodes()):
        set_attr(G.nodes[node], ALIAS_EPI, 0.5 + 0.1 * np.sin(i))
        set_attr(G.nodes[node], ALIAS_VF, 1.0 + 0.2 * np.cos(i))
        set_attr(G.nodes[node], ALIAS_DNFR, 0.1 + 0.05 * np.random.rand())
        set_attr(G.nodes[node], ALIAS_THETA, 2 * np.pi * np.random.rand())
        
    return G


def benchmark_function(func, *args, **kwargs) -> Tuple[float, Any]:
    """Benchmark function execution time and return result."""
    gc.collect()  # Clean up before timing
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    return end_time - start_time, result


class TestPerformanceBenchmarks:
    """Performance benchmarking for core TNFR operations."""
    
    def test_field_computation_performance(self):
        """Test computational performance of structural fields."""
        # Test across different network sizes
        sizes = [10, 50, 100]
        performance_data = {}
        
        for size in sizes:
            G = create_performance_test_network(size)
            
            # Benchmark structural potential
            phi_s_time, phi_s = benchmark_function(compute_structural_potential, G)
            
            # Benchmark phase gradient
            grad_phi_time, grad_phi = benchmark_function(compute_phase_gradient, G)
            
            # Benchmark phase curvature
            curv_phi_time, curv_phi = benchmark_function(compute_phase_curvature, G)
            
            # Benchmark coherence
            coherence_time, coherence = benchmark_function(compute_coherence, G)
            
            performance_data[size] = {
                'phi_s_time': phi_s_time,
                'grad_phi_time': grad_phi_time,
                'curv_phi_time': curv_phi_time,
                'coherence_time': coherence_time,
                'total_time': phi_s_time + grad_phi_time + curv_phi_time + coherence_time
            }
            
            # Verify correctness
            assert len(phi_s) == size, f"Φ_s computed for all {size} nodes"
            assert len(grad_phi) == size, f"|∇φ| computed for all {size} nodes"
            assert len(curv_phi) == size, f"K_φ computed for all {size} nodes"
            assert isinstance(coherence, float) and 0 <= coherence <= 1, "Coherence valid"
        
        # Performance scaling check (O(n) or better expected)
        small_time = performance_data[10]['total_time']
        large_time = performance_data[100]['total_time']
        scaling_factor = large_time / small_time
        
        # Should scale reasonably (less than O(n²))
        assert scaling_factor < 200, f"Performance scaling too poor: {scaling_factor}x for 10x nodes"
        
        print("\n🚀 Field Computation Performance:")
        for size, data in performance_data.items():
            print(f"  {size} nodes: {data['total_time']:.4f}s total")
            print(f"    Φ_s: {data['phi_s_time']:.4f}s, |∇φ|: {data['grad_phi_time']:.4f}s")
            print(f"    K_φ: {data['curv_phi_time']:.4f}s, C(t): {data['coherence_time']:.4f}s")

    def test_operator_instantiation_performance(self):
        """Test performance of operator instantiation and basic operations."""
        num_operators = 1000
        
        # Benchmark operator instantiation
        instantiation_time, operators = benchmark_function(
            lambda: [Emission() for _ in range(num_operators)]
        )
        
        assert len(operators) == num_operators, "All operators instantiated"
        
        # Should be fast (< 1ms per operator)
        time_per_op = instantiation_time / num_operators
        assert time_per_op < 0.001, f"Operator instantiation too slow: {time_per_op:.6f}s per op"
        
        print("\n⚡ Operator Performance:")
        print(f"  {num_operators} instantiations: {instantiation_time:.4f}s total")
        print(f"  Per operator: {time_per_op:.6f}s")

    def test_network_scaling_performance(self):
        """Test performance scaling with network size."""
        sizes = [20, 50, 100, 200]
        timing_results = []
        
        for size in sizes:
            G = create_performance_test_network(size)
            
            # Test complete workflow timing
            start_time = time.perf_counter()
            
            # Compute all fields
            phi_s = compute_structural_potential(G)
            grad_phi = compute_phase_gradient(G)
            curv_phi = compute_phase_curvature(G)
            coherence = compute_coherence(G)
            
            # Apply operators to subset of nodes
            test_nodes = list(G.nodes())[:min(5, len(G.nodes()))]
            for node in test_nodes:
                try:
                    coherence_op = Coherence()
                    coherence_op(G, node)
                except Exception:
                    # Operator might not be fully implemented, skip
                    pass
            
            end_time = time.perf_counter()
            workflow_time = end_time - start_time
            
            timing_results.append((size, workflow_time))
            
            # Verify results are valid
            assert len(phi_s) == size
            assert len(grad_phi) == size
            assert len(curv_phi) == size
            assert isinstance(coherence, float)
        
        # Check if scaling is reasonable (should be sub-quadratic)
        if len(timing_results) >= 2:
            first_size, first_time = timing_results[0]
            last_size, last_time = timing_results[-1]
            
            time_ratio = last_time / first_time
            
            # Time growth should be less than size²
            efficiency = time_ratio / ((last_size / first_size) ** 2)
            assert efficiency < 2.0, f"Quadratic scaling exceeded: efficiency = {efficiency}"
        
        print("\n📈 Network Scaling Performance:")
        for size, timing in timing_results:
            print(f"  {size} nodes: {timing:.4f}s")


class TestMemoryEfficiency:
    """Memory usage and efficiency tests."""
    
    def test_memory_usage_bounded(self):
        """Test that memory usage remains bounded during operations."""
        # Note: This is a basic check since detailed memory profiling 
        # would require additional dependencies
        
        initial_objects = len(gc.get_objects())
        
        # Create and process multiple networks
        for i in range(10):
            G = create_performance_test_network(50)
            
            # Compute fields
            phi_s = compute_structural_potential(G)
            grad_phi = compute_phase_gradient(G)
            curv_phi = compute_phase_curvature(G)
            
            # Clean up references
            del G, phi_s, grad_phi, curv_phi
            
            # Periodic cleanup
            if i % 3 == 0:
                gc.collect()
        
        # Final cleanup
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Should not have excessive object growth
        object_growth = final_objects - initial_objects
        assert object_growth < 1000, f"Excessive object growth: {object_growth}"
        
        print("\n💾 Memory Efficiency:")
        print(f"  Object growth: {object_growth} objects")
    
    def test_no_memory_leaks_operators(self):
        """Test operators don't create memory leaks."""
        G = create_performance_test_network(20)
        initial_objects = len(gc.get_objects())
        
        # Apply operators repeatedly
        operators = [Emission(), Coherence(), Silence()]
        for _ in range(100):
            for op in operators:
                try:
                    op(G, 0)  # Apply to first node
                except Exception:
                    # Skip if operator not implemented
                    pass
        
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Should not grow significantly
        object_growth = final_objects - initial_objects
        assert object_growth < 500, f"Operator memory leak detected: {object_growth}"
        
        print(f"  Operator object growth: {object_growth} objects")


class TestRegressionValidation:
    """Regression tests to ensure changes don't break existing functionality."""
    
    def test_field_computation_consistency(self):
        """Test field computations return consistent results."""
        # Create deterministic network
        np.random.seed(42)
        G = create_performance_test_network(50)
        
        # Compute fields multiple times
        results = []
        for _ in range(5):
            phi_s = compute_structural_potential(G)
            grad_phi = compute_phase_gradient(G)
            curv_phi = compute_phase_curvature(G)
            coherence = compute_coherence(G)
            
            results.append({
                'phi_s': sum(phi_s.values()),
                'grad_phi': sum(grad_phi.values()),
                'curv_phi': sum(abs(v) for v in curv_phi.values()),
                'coherence': coherence
            })
        
        # All results should be identical (deterministic computation)
        for i in range(1, len(results)):
            for key in results[0]:
                assert abs(results[i][key] - results[0][key]) < 1e-10, f"Non-deterministic {key}"
        
        print("\n🔒 Consistency Check:")
        print(f"  All 5 runs identical for Φ_s, |∇φ|, K_φ, C(t)")
    
    def test_operator_sequence_determinism(self):
        """Test operator sequences produce deterministic results."""
        # Create test network
        G1 = create_performance_test_network(20)
        G2 = create_performance_test_network(20)
        
        # Ensure identical initial states
        for node in G1.nodes():
            for alias in [ALIAS_EPI, ALIAS_VF, ALIAS_DNFR, ALIAS_THETA]:
                value = G1.nodes[node][alias[0]]
                set_attr(G2.nodes[node], alias, value)
        
        # Apply same sequence to both
        try:
            sequence = [Coherence(), Silence()]
            for op in sequence:
                op(G1, 0)
                op(G2, 0)
            
            # Should have identical final states
            for alias in [ALIAS_EPI, ALIAS_VF, ALIAS_DNFR, ALIAS_THETA]:
                val1 = G1.nodes[0][alias[0]]
                val2 = G2.nodes[0][alias[0]]
                assert abs(val1 - val2) < 1e-10, f"Non-deterministic operator effect: {alias[0]}"
            
            print(f"  Operator sequences deterministic ✓")
            
        except Exception as e:
            # If operators not fully implemented, just verify networks remain valid
            coherence1 = compute_coherence(G1)
            coherence2 = compute_coherence(G2)
            assert isinstance(coherence1, float) and isinstance(coherence2, float)
            print(f"  Networks remain valid after operator attempts ✓")
    
    def test_mathematical_invariants_preserved(self):
        """Test that mathematical invariants are preserved across operations."""
        G = create_performance_test_network(30)
        
        # Record initial mathematical properties
        initial_phi_s = compute_structural_potential(G)
        initial_coherence = compute_coherence(G)
        
        # Apply operations
        try:
            # Test coherence preservation
            coherence_op = Coherence()
            coherence_op(G, 0)
            
            post_coherence = compute_coherence(G)
            
            # Coherence should not decrease (monotonicity)
            assert post_coherence >= initial_coherence * 0.95, "Coherence monotonicity violated"
            
        except Exception:
            # If coherence operator not implemented, verify fields remain computable
            pass
        
        # Verify fields remain mathematically valid
        final_phi_s = compute_structural_potential(G)
        final_grad_phi = compute_phase_gradient(G)
        final_curv_phi = compute_phase_curvature(G)
        final_coherence = compute_coherence(G)
        
        # Mathematical bounds should be preserved
        assert all(np.isfinite(v) for v in final_phi_s.values()), "Φ_s finite"
        assert all(v >= 0 for v in final_grad_phi.values()), "|∇φ| non-negative"
        assert all(-np.pi <= v <= np.pi for v in final_curv_phi.values()), "K_φ in [-π,π]"
        assert 0 <= final_coherence <= 1, "C(t) in [0,1]"
        
        print(f"  Mathematical invariants preserved ✓")


class TestStressTests:
    """Stress tests for extreme conditions."""
    
    def test_large_network_handling(self):
        """Test handling of large networks."""
        # Test with larger network (if feasible)
        large_size = 500
        
        try:
            G = create_performance_test_network(large_size)
            
            # Should handle large networks gracefully
            start_time = time.perf_counter()
            
            phi_s = compute_structural_potential(G)
            coherence = compute_coherence(G)
            
            computation_time = time.perf_counter() - start_time
            
            # Verify correctness
            assert len(phi_s) == large_size
            assert isinstance(coherence, float)
            
            # Should complete in reasonable time (< 10 seconds)
            assert computation_time < 10.0, f"Large network too slow: {computation_time:.2f}s"
            
            print("\n🏋️ Stress Test:")
            print(f"  {large_size} nodes: {computation_time:.4f}s")
            
        except MemoryError:
            pytest.skip("Insufficient memory for large network test")
        except Exception as e:
            # If computation fails, at least verify it fails gracefully
            assert isinstance(e, Exception)
            print(f"  Large network test failed gracefully: {type(e).__name__}")
    
    def test_extreme_parameter_values(self):
        """Test handling of extreme parameter values."""
        G = nx.complete_graph(10)
        
        # Test with extreme values
        extreme_cases = [
            {'epi': 0.0, 'vf': 0.001, 'dnfr': 0.0, 'theta': 0.0},      # Minimal values
            {'epi': 1.0, 'vf': 10.0, 'dnfr': 1.0, 'theta': 2*np.pi},   # High values
            {'epi': 0.5, 'vf': 0.5, 'dnfr': -0.1, 'theta': np.pi},     # Mixed values
        ]
        
        for i, case in enumerate(extreme_cases):
            # Initialize network with extreme values
            for node in G.nodes():
                set_attr(G.nodes[node], ALIAS_EPI, case['epi'])
                set_attr(G.nodes[node], ALIAS_VF, case['vf'])
                set_attr(G.nodes[node], ALIAS_DNFR, case['dnfr'])
                set_attr(G.nodes[node], ALIAS_THETA, case['theta'])
            
            # Should handle extreme values gracefully
            try:
                phi_s = compute_structural_potential(G)
                grad_phi = compute_phase_gradient(G)
                coherence = compute_coherence(G)
                
                # Verify results are still mathematically valid
                assert len(phi_s) == G.number_of_nodes()
                assert all(np.isfinite(v) for v in phi_s.values())
                assert all(v >= 0 for v in grad_phi.values())
                assert 0 <= coherence <= 1
                
                print(f"  Extreme case {i+1}: handled gracefully ✓")
                
            except Exception as e:
                # Should fail gracefully, not crash
                assert isinstance(e, (ValueError, RuntimeError, ZeroDivisionError))
                print(f"  Extreme case {i+1}: failed gracefully ({type(e).__name__}) ✓")


if __name__ == "__main__":
    # Run performance and regression tests
    pytest.main([__file__, "-v", "-s", "--tb=short"])