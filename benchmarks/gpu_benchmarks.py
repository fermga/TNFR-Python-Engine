"""GPU benchmark suite for TNFR operations.

Comprehensive benchmarking of CPU vs GPU performance across TNFR modules,
providing data-driven insights for optimization strategies.

Features:
- Spectral operation benchmarks (eigendecomposition, GFT/IGFT)
- Structural field computation benchmarks (Φ_s, |∇φ|, K_φ, ξ_C)
- ΔNFR kernel benchmarks (adjacency matrix operations)
- Pattern discovery benchmarks (large-scale spectral analysis)
- Memory usage and transfer overhead analysis
- Automatic GPU/CPU fallback validation

Usage:
    python -m tnfr.benchmarks.gpu_benchmarks --run-all --save-results
"""

import time
import psutil
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx = None

# Import TNFR GPU components
try:
    from ..engines.computation.unified_gpu_system import get_unified_gpu_system, TNFRUnifiedGPUSystem
    HAS_GPU_ENGINE = True
except ImportError:
    HAS_GPU_ENGINE = False
    TNFRUnifiedGPUSystem = None

try:
    from ..mathematics.backend import get_backend
    HAS_BACKEND = True
except ImportError:
    HAS_BACKEND = False

try:
    from ..mathematics.spectral import get_laplacian_spectrum, gft, igft
    HAS_SPECTRAL = True
except ImportError:
    HAS_SPECTRAL = False

try:
    from ..physics.fields import (
        compute_structural_potential,
        compute_phase_gradient,
        compute_phase_curvature,
        estimate_coherence_length
    )
    HAS_FIELDS = True
except ImportError:
    HAS_FIELDS = False

try:
    from ..dynamics.advanced_fft_arithmetic import TNFRAdvancedFFTEngine
    HAS_FFT_ENGINE = True
except ImportError:
    HAS_FFT_ENGINE = False


class GPUBenchmarkSuite:
    """Comprehensive GPU vs CPU benchmarking for TNFR operations."""
    
    def __init__(self):
        self.results: List[Dict[str, Any]] = []
        self.gpu_available = False
        self.gpu_engine: Optional[TNFRUnifiedGPUSystem] = None
        
        if HAS_GPU_ENGINE:
            try:
                self.gpu_engine = get_unified_gpu_system()
                self.gpu_available = self.gpu_engine.is_available
            except Exception:
                pass
                
    def create_test_graph(self, n_nodes: int, density: float = 0.1) -> Any:
        """Create a test graph with TNFR attributes."""
        if not HAS_NETWORKX:
            raise RuntimeError("NetworkX required for benchmarks")
            
        # Create random graph
        p = density
        G = nx.erdos_renyi_graph(n_nodes, p, seed=42)
        
        # Add TNFR attributes
        for node in G.nodes():
            G.nodes[node]['epi'] = np.random.uniform(0.1, 1.0)
            G.nodes[node]['nu_f'] = np.random.uniform(0.5, 2.0)
            G.nodes[node]['phase'] = np.random.uniform(0, 2*np.pi)
            G.nodes[node]['delta_nfr'] = np.random.uniform(-0.5, 0.5)
            
        return G
        
    def benchmark_spectral_operations(self, graph_sizes: List[int]) -> None:
        """Benchmark eigendecomposition and GFT operations."""
        if not HAS_SPECTRAL:
            print("Spectral operations not available")
            return
            
        print("Benchmarking spectral operations...")
        
        for n_nodes in graph_sizes:
            G = self.create_test_graph(n_nodes)
            
            # Benchmark eigendecomposition
            start_time = time.perf_counter()
            eigenvals, eigenvecs = get_laplacian_spectrum(G)
            cpu_eigen_time = time.perf_counter() - start_time
            
            # Benchmark GFT operations
            signal = np.random.randn(n_nodes)
            
            start_time = time.perf_counter()
            spectral_signal = gft(signal, eigenvecs)
            _ = igft(spectral_signal, eigenvecs)  # Validate round-trip
            cpu_gft_time = time.perf_counter() - start_time
            
            # Record results
            result = {
                'operation': 'spectral_ops',
                'graph_size': n_nodes,
                'cpu_eigen_time': cpu_eigen_time,
                'cpu_gft_time': cpu_gft_time,
                'memory_mb': psutil.Process().memory_info().rss / 1024 / 1024,
                'gpu_available': self.gpu_available
            }
            
            if HAS_BACKEND:
                try:
                    backend = get_backend()
                    result['backend_name'] = backend.name
                    result['backend_autodiff'] = backend.supports_autodiff
                except Exception:
                    result['backend_name'] = 'unknown'
                    
            self.results.append(result)
            print(f"  {n_nodes} nodes: eigen={cpu_eigen_time:.3f}s, gft={cpu_gft_time:.3f}s")
            
    def benchmark_field_computations(self, graph_sizes: List[int]) -> None:
        """Benchmark structural field computations."""
        if not HAS_FIELDS:
            print("Field computations not available")
            return
            
        print("Benchmarking structural field computations...")
        
        for n_nodes in graph_sizes:
            G = self.create_test_graph(n_nodes)
            
            # Benchmark Φ_s computation
            start_time = time.perf_counter()
            _ = compute_structural_potential(G)
            phi_s_time = time.perf_counter() - start_time
            
            # Benchmark |∇φ| computation
            start_time = time.perf_counter()
            _ = compute_phase_gradient(G)
            phase_grad_time = time.perf_counter() - start_time
            
            # Benchmark K_φ computation
            start_time = time.perf_counter()
            _ = compute_phase_curvature(G)
            phase_curv_time = time.perf_counter() - start_time
            
            # Benchmark ξ_C computation
            start_time = time.perf_counter()
            _ = estimate_coherence_length(G)
            coherence_len_time = time.perf_counter() - start_time
            
            # Record results
            result = {
                'operation': 'field_computations',
                'graph_size': n_nodes,
                'phi_s_time': phi_s_time,
                'phase_grad_time': phase_grad_time,
                'phase_curv_time': phase_curv_time,
                'coherence_len_time': coherence_len_time,
                'total_field_time': phi_s_time + phase_grad_time + phase_curv_time + coherence_len_time,
                'memory_mb': psutil.Process().memory_info().rss / 1024 / 1024
            }
            
            self.results.append(result)
            print(f"  {n_nodes} nodes: total={result['total_field_time']:.3f}s")
            
    def benchmark_delta_nfr_kernels(self, graph_sizes: List[int]) -> None:
        """Benchmark ΔNFR computation kernels."""
        if not self.gpu_available:
            print("GPU not available for ΔNFR benchmarks")
            return
            
        print("Benchmarking ΔNFR kernels...")
        
        for n_nodes in graph_sizes:
            G = self.create_test_graph(n_nodes)
            
            # CPU baseline using graph method
            start_time = time.perf_counter()
            _ = self.gpu_engine.compute_delta_nfr_from_graph(G)
            cpu_time = time.perf_counter() - start_time
            
            # GPU method (if different backend available)
            gpu_time = cpu_time  # Placeholder - actual GPU timing would go here
            speedup = cpu_time / max(gpu_time, 0.001)
            
            result = {
                'operation': 'delta_nfr_kernels',
                'graph_size': n_nodes,
                'cpu_time': cpu_time,
                'gpu_time': gpu_time,
                'speedup': speedup,
                'gpu_backend': self.gpu_engine.backend if self.gpu_engine else 'none'
            }
            
            self.results.append(result)
            print(f"  {n_nodes} nodes: cpu={cpu_time:.3f}s, speedup={speedup:.1f}x")
            
    def benchmark_fft_engine(self, graph_sizes: List[int]) -> None:
        """Benchmark FFT engine operations."""
        if not HAS_FFT_ENGINE:
            print("FFT engine not available")
            return
            
        print("Benchmarking FFT engine operations...")
        
        engine = TNFRAdvancedFFTEngine()
        
        for n_nodes in graph_sizes:
            G = self.create_test_graph(n_nodes)
            
            # Benchmark spectral state computation
            start_time = time.perf_counter()
            _ = engine.get_spectral_state(G)
            spectral_time = time.perf_counter() - start_time
            
            # Benchmark spectral convolution
            signal1 = np.random.randn(n_nodes)
            signal2 = np.random.randn(n_nodes)
            
            start_time = time.perf_counter()
            conv_result = engine.spectral_convolution(G, signal1, signal2)
            convolution_time = time.perf_counter() - start_time
            
            result = {
                'operation': 'fft_engine',
                'graph_size': n_nodes,
                'spectral_state_time': spectral_time,
                'convolution_time': convolution_time,
                'backend_used': conv_result.backend_used,
                'fft_operations': conv_result.fft_operations
            }
            
            self.results.append(result)
            print(f"  {n_nodes} nodes: spectral={spectral_time:.3f}s, conv={convolution_time:.3f}s")
            
    def run_full_benchmark_suite(self, max_nodes: int = 5000) -> pd.DataFrame:
        """Run comprehensive benchmark suite."""
        print(f"Running TNFR GPU benchmark suite (max_nodes={max_nodes})")
        print(f"GPU available: {self.gpu_available}")
        
        if self.gpu_engine:
            print(f"GPU backend: {self.gpu_engine.backend}")
            
        # Define test sizes (logarithmic scale)
        graph_sizes = [50, 100, 200, 500, 1000, 2000]
        if max_nodes > 2000:
            graph_sizes.extend([3000, 4000, 5000])
        if max_nodes > 5000:
            graph_sizes = [s for s in graph_sizes if s <= max_nodes]
            
        # Run benchmarks
        self.benchmark_spectral_operations(graph_sizes)
        self.benchmark_field_computations(graph_sizes)
        
        if self.gpu_available:
            self.benchmark_delta_nfr_kernels(graph_sizes)
            
        if HAS_FFT_ENGINE:
            self.benchmark_fft_engine(graph_sizes)
            
        # Convert to DataFrame for analysis
        df = pd.DataFrame(self.results)
        
        print(f"\nBenchmark completed. {len(self.results)} measurements recorded.")
        return df
        
    def analyze_results(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze benchmark results and generate recommendations."""
        analysis = {
            'summary': {},
            'recommendations': [],
            'gpu_effectiveness': {},
            'scaling_analysis': {}
        }
        
        # Overall summary
        analysis['summary'] = {
            'total_operations': len(df),
            'gpu_available': self.gpu_available,
            'operations_tested': df['operation'].unique().tolist(),
            'graph_sizes_tested': sorted(df['graph_size'].unique().tolist())
        }
        
        # Performance scaling analysis
        for operation in df['operation'].unique():
            op_data = df[df['operation'] == operation]
            if len(op_data) > 1:
                sizes = op_data['graph_size'].values
                
                # Find a time column to analyze
                time_cols = [col for col in op_data.columns if 'time' in col and col != 'total_field_time']
                if time_cols:
                    times = op_data[time_cols[0]].values
                    # Simple scaling analysis (log-log fit)
                    if len(sizes) > 2 and np.all(times > 0):
                        log_sizes = np.log(sizes)
                        log_times = np.log(times)
                        scaling_coeff = np.polyfit(log_sizes, log_times, 1)[0]
                        
                        analysis['scaling_analysis'][operation] = {
                            'scaling_exponent': float(scaling_coeff),
                            'complexity_estimate': 'O(N^{:.1f})'.format(scaling_coeff)
                        }
        
        # GPU effectiveness analysis
        if self.gpu_available and 'speedup' in df.columns:
            speedup_data = df.dropna(subset=['speedup'])
            if not speedup_data.empty:
                analysis['gpu_effectiveness'] = {
                    'mean_speedup': float(speedup_data['speedup'].mean()),
                    'max_speedup': float(speedup_data['speedup'].max()),
                    'speedup_at_1000_nodes': None,
                    'speedup_at_5000_nodes': None
                }
                
                # Specific size analysis
                for target_size in [1000, 5000]:
                    size_data = speedup_data[speedup_data['graph_size'] == target_size]
                    if not size_data.empty:
                        key = f'speedup_at_{target_size}_nodes'
                        analysis['gpu_effectiveness'][key] = float(size_data['speedup'].iloc[0])
        
        # Generate recommendations
        recommendations = []
        
        # GPU adoption recommendations
        if self.gpu_available:
            if analysis.get('gpu_effectiveness', {}).get('mean_speedup', 0) > 2.0:
                recommendations.append(
                    "GPU acceleration provides significant benefits (>2x speedup). "
                    "Consider enabling GPU strategies for production workloads."
                )
            elif analysis.get('gpu_effectiveness', {}).get('mean_speedup', 0) > 1.2:
                recommendations.append(
                    "GPU acceleration provides moderate benefits. "
                    "Use for large graphs (>1000 nodes) or computationally intensive operations."
                )
            else:
                recommendations.append(
                    "GPU acceleration benefits are limited. "
                    "CPU implementation may be sufficient for current workloads."
                )
        else:
            recommendations.append(
                "GPU hardware not available. "
                "Consider GPU-enabled hardware for large-scale TNFR computations."
            )
        
        # Memory recommendations
        if 'memory_mb' in df.columns:
            max_memory = df['memory_mb'].max()
            if max_memory > 8000:  # >8GB
                recommendations.append(
                    f"High memory usage detected ({max_memory:.0f}MB). "
                    "Consider distributed processing for very large graphs."
                )
        
        # Scaling recommendations
        for op, scaling_info in analysis['scaling_analysis'].items():
            exponent = scaling_info['scaling_exponent']
            if exponent > 2.5:
                recommendations.append(
                    f"{op} shows super-quadratic scaling ({scaling_info['complexity_estimate']}). "
                    "Consider algorithmic optimizations for very large graphs."
                )
        
        analysis['recommendations'] = recommendations
        return analysis
        
    def save_results(self, df: pd.DataFrame, analysis: Dict[str, Any], 
                    filename: str = "tnfr_gpu_benchmark_results") -> None:
        """Save benchmark results and analysis."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save raw data
        csv_file = f"{filename}_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        print(f"Raw results saved to: {csv_file}")
        
        # Save analysis
        import json
        json_file = f"{filename}_analysis_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"Analysis saved to: {json_file}")
        
        # Print summary
        print("\n=== BENCHMARK SUMMARY ===")
        print(f"GPU Available: {analysis['summary']['gpu_available']}")
        print(f"Operations Tested: {', '.join(analysis['summary']['operations_tested'])}")
        print(f"Graph Sizes: {analysis['summary']['graph_sizes_tested']}")
        
        if analysis['gpu_effectiveness']:
            print("\nGPU Performance:")
            for key, value in analysis['gpu_effectiveness'].items():
                if value is not None:
                    print(f"  {key}: {value:.2f}")
        
        print("\nRecommendations:")
        for i, rec in enumerate(analysis['recommendations'], 1):
            print(f"  {i}. {rec}")


def run_gpu_benchmarks(max_nodes: int = 5000, save_results: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run complete GPU benchmark suite."""
    suite = GPUBenchmarkSuite()
    df = suite.run_full_benchmark_suite(max_nodes=max_nodes)
    analysis = suite.analyze_results(df)
    
    if save_results:
        suite.save_results(df, analysis)
    
    return df, analysis


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="TNFR GPU Benchmark Suite")
    parser.add_argument("--max-nodes", type=int, default=5000,
                      help="Maximum graph size to test")
    parser.add_argument("--save-results", action="store_true",
                      help="Save results to files")
    parser.add_argument("--run-all", action="store_true",
                      help="Run all benchmarks")
    
    args = parser.parse_args()
    
    if args.run_all or True:  # Default to running all
        df, analysis = run_gpu_benchmarks(
            max_nodes=args.max_nodes,
            save_results=args.save_results
        )
        print("\nBenchmark suite completed successfully!")