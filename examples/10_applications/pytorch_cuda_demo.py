"""PyTorch CUDA integration example for TNFR.

Demonstrates how to use TNFR with PyTorch CUDA acceleration for
large-scale structural field computations and network analysis.

Usage:
    # With CUDA (if available):
    TNFR_CUDA_ENABLED=true python examples/10_applications/pytorch_cuda_demo.py
    
    # CPU only:
    TNFR_CUDA_ENABLED=false python examples/10_applications/pytorch_cuda_demo.py
"""

import os
import time
import numpy as np

# Set PyTorch backend for TNFR before imports
os.environ['TNFR_MATH_BACKEND'] = 'torch'

from tnfr.mathematics.backend import get_backend
from tnfr.engines.computation.unified_gpu_system import get_unified_gpu_system

try:
    from tnfr.networks import create_random_network
except ImportError:
    # Fallback for networks module if not available
    import networkx as nx
    
    def create_random_network(n_nodes: int, density: float = 0.1, seed: int = 42):
        """Fallback network creator."""
        np.random.seed(seed)
        G = nx.erdos_renyi_graph(n_nodes, density, seed=seed)
        
        # Add TNFR attributes
        for node in G.nodes():
            G.nodes[node]['EPI'] = np.random.uniform(0.1, 1.0)
            G.nodes[node]['nu_f'] = np.random.uniform(0.5, 2.0)
            G.nodes[node]['phase'] = np.random.uniform(0, 2*np.pi)
            
        return G


def demonstrate_pytorch_cuda():
    """Demonstrate PyTorch CUDA integration with TNFR."""
    
    print("🔥 TNFR PyTorch CUDA Integration Demo")
    print("=" * 50)
    
    # Check PyTorch installation
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f}GB)")
        else:
            print("ℹ️ CUDA not available - using CPU backend")
            
    except ImportError:
        print("❌ PyTorch not installed")
        return
    
    print()
    
    # Initialize TNFR backends
    print("🧮 Initializing TNFR Mathematics Backend")
    backend = get_backend('torch')
    print(f"✓ Backend: {backend.name}")
    
    if hasattr(backend, 'get_device_info'):
        device_info = backend.get_device_info()
        print(f"✓ Device: {device_info['device']}")
        print(f"✓ CUDA enabled: {device_info['use_cuda']}")
    
    print()
    
    # Test GPU engine
    print("⚡ Testing GPU Acceleration Engine")
    try:
        # Use unified system
        gpu_engine = get_unified_gpu_system()
        print(f"✓ GPU Engine backend: {gpu_engine.math_backend.name}")
        print(f"✓ GPU acceleration: {gpu_engine.is_available}")
            
    except ImportError as e:
        print(f"⚠️ GPU engine error: {e}")
        print("ℹ️ Falling back to CPU-only computation")
        gpu_engine = None
    
    print()
    
    # Create test network
    print("🌐 Creating Test Network")
    n_nodes = 1000
    G = create_random_network(n_nodes, density=0.1, seed=42)
    print(f"✓ Created network: {n_nodes} nodes, {G.number_of_edges()} edges")
    
    print()
    
    # Benchmark computations
    print("⏱️ Benchmarking Structural Field Computations")
    
    if gpu_engine:
        # Test ΔNFR computation
        start_time = time.perf_counter()
        delta_nfr_results = gpu_engine.compute_delta_nfr_from_graph(G)
        computation_time = time.perf_counter() - start_time
        
        print(f"✓ ΔNFR computation: {computation_time:.3f}s")
        print(f"✓ Results: {len(delta_nfr_results)} nodes processed")
        
        # Show sample results
        sample_nodes = list(delta_nfr_results.keys())[:5]
        print(f"✓ Sample ΔNFR values:")
        for node in sample_nodes:
            print(f"  Node {node}: {delta_nfr_results[node]:.4f}")
    
    print()
    
    # Test tensor operations
    print("🔧 Testing Tensor Operations")
    
    # Create test matrices
    matrix_size = 500
    test_matrix = backend.as_array(np.random.randn(matrix_size, matrix_size))
    print(f"✓ Created test matrix: {matrix_size}x{matrix_size}")
    
    if hasattr(test_matrix, 'device'):
        print(f"✓ Matrix device: {test_matrix.device}")
    
    # Eigendecomposition benchmark
    start_time = time.perf_counter()
    eigenvals, eigenvecs = backend.eigh(test_matrix)
    eigen_time = time.perf_counter() - start_time
    
    print(f"✓ Eigendecomposition: {eigen_time:.3f}s")
    print(f"✓ Eigenvalues shape: {eigenvals.shape}")
    
    # Matrix multiplication benchmark
    start_time = time.perf_counter()
    result = backend.matmul(test_matrix, test_matrix.T)
    matmul_time = time.perf_counter() - start_time
    
    print(f"✓ Matrix multiplication: {matmul_time:.3f}s")
    print(f"✓ Result shape: {result.shape}")
    
    print()
    
    # Performance summary
    print("📊 Performance Summary")
    print(f"• Backend: {backend.name}")
    print(f"• Device: {'CUDA' if torch.cuda.is_available() and hasattr(backend, '_use_cuda') and backend._use_cuda else 'CPU'}")
    print(f"• Network size: {n_nodes} nodes")
    print(f"• Matrix operations: {matrix_size}x{matrix_size}")
    
    if gpu_engine:
        print(f"• ΔNFR computation: {computation_time:.3f}s")
    print(f"• Eigendecomposition: {eigen_time:.3f}s")
    print(f"• Matrix multiplication: {matmul_time:.3f}s")
    
    print()
    print("🎉 PyTorch CUDA integration demo completed!")
    
    # Usage recommendations
    print("\n💡 Usage Recommendations:")
    if torch.cuda.is_available():
        print("• Set TNFR_CUDA_ENABLED=true for CUDA acceleration")
        print("• Use TNFRGPUEngine(backend='torch') for CUDA operations")
        print("• Large networks (>1000 nodes) benefit most from GPU acceleration")
    else:
        print("• Install CUDA-enabled PyTorch for GPU acceleration:")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print("• Current CPU backend is optimized for smaller networks")
    
    print("• Set TNFR_MATH_BACKEND=torch in environment for consistent PyTorch usage")
    print("• Use gpu_memory_managed decorator for automatic memory management")


if __name__ == "__main__":
    demonstrate_pytorch_cuda()