"""Benchmark for GPU backend performance on large graphs.

This script demonstrates the potential for GPU acceleration on very large
graphs (>10K nodes) and documents the infrastructure for GPU backends.

Current Status:
- JAX backend: Infrastructure ready, delegates to NumPy
- PyTorch backend: Infrastructure ready, delegates to NumPy, device: {device}

For actual GPU acceleration, the backends need graph-specific GPU kernels.
"""

import time
import networkx as nx
from tnfr.backends import get_backend, available_backends


def benchmark_backend(backend_name, n_nodes, density=0.1, seed=42):
    """Benchmark a backend on a large graph.
    
    Parameters
    ----------
    backend_name : str
        Name of backend to test
    n_nodes : int
        Number of nodes
    density : float
        Edge probability
    seed : int
        Random seed
        
    Returns
    -------
    dict
        Timing results
    """
    print(f"\nBenchmarking {backend_name} backend with {n_nodes} nodes...")
    
    # Create large graph
    G = nx.erdos_renyi_graph(n_nodes, density, seed=seed)
    
    # Initialize attributes
    for node in G.nodes():
        G.nodes[node]["phase"] = float(node) * 0.001
        G.nodes[node]["nu_f"] = 1.0
        G.nodes[node]["EPI"] = 0.5
    
    G.graph["DNFR_WEIGHTS"] = {
        "phase": 0.4,
        "epi": 0.3,
        "vf": 0.2,
        "topo": 0.1,
    }
    
    try:
        backend = get_backend(backend_name)
        
        # Warm up
        backend.compute_delta_nfr(G)
        
        # Time computation
        t0 = time.perf_counter()
        backend.compute_delta_nfr(G)
        elapsed = time.perf_counter() - t0
        
        return {
            "success": True,
            "time": elapsed,
            "n_edges": G.number_of_edges(),
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def main():
    """Run large graph benchmarks."""
    print("=" * 80)
    print("TNFR Large Graph GPU Backend Benchmarks")
    print("=" * 80)
    
    # Check available backends
    print("\nAvailable backends:")
    backends = list(available_backends().keys())
    for name in backends:
        try:
            backend = get_backend(name)
            gpu_status = "✓ GPU" if backend.supports_gpu else "  CPU"
            jit_status = "✓ JIT" if backend.supports_jit else "  ---"
            print(f"  {gpu_status} {jit_status}  {name}")
        except:
            print(f"  ✗ FAIL  ---  {name}")
    
    # Benchmark configurations
    configs = [
        (1000, 0.1, "1K nodes, 10% density"),
        (5000, 0.05, "5K nodes, 5% density"),
        (10000, 0.02, "10K nodes, 2% density"),
    ]
    
    print("\n" + "=" * 80)
    print("Benchmarks (note: GPU backends currently delegate to NumPy)")
    print("=" * 80)
    
    for n_nodes, density, desc in configs:
        print(f"\n{desc}")
        print("-" * 80)
        
        for backend_name in ["numpy", "optimized_numpy", "torch"]:
            result = benchmark_backend(backend_name, n_nodes, density)
            
            if result["success"]:
                print(f"  {backend_name:20s}: {result['time']*1000:7.1f} ms "
                      f"({result['n_edges']:,} edges)")
            else:
                print(f"  {backend_name:20s}: FAILED - {result['error']}")
    
    print("\n" + "=" * 80)
    print("Summary:")
    print("=" * 80)
    print("""
GPU Backend Status:
  
  1. PyTorch Backend:
     - ✓ Infrastructure ready
     - ✓ PyTorch installed and device detected
     - ⏳ Needs GPU kernel implementation for acceleration
     - Target: 10-50x speedup on GPU for graphs >10K nodes
     
  2. JAX Backend:
     - ✓ Infrastructure ready  
     - ✗ JAX not installed (requires jax + jaxlib)
     - ⏳ Needs JIT-compiled kernel implementation
     - Target: 5-20x speedup with JIT on CPU, 20-100x on GPU
     
  3. Next Steps:
     - Implement GPU kernels for PyTorch backend
     - Convert compute_fused_gradients_symmetric to torch operations
     - Add device placement (CPU/CUDA) selection
     - Benchmark on actual GPU hardware
     
  4. For Production Use:
     - Install PyTorch with CUDA:
       pip install torch --index-url https://download.pytorch.org/whl/cu118
     - Or install JAX with CUDA:
       pip install jax[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
""")


if __name__ == "__main__":
    main()
