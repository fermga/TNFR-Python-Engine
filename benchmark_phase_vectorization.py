"""Benchmark vectorized phase operations."""
import time
import networkx as nx
import numpy as np

from tnfr.physics.fields import (
    compute_phase_gradient,
    compute_phase_curvature,
)

print("=" * 80)
print("Phase Operations Vectorization Benchmark")
print("=" * 80)

# Test graphs of varying sizes
sizes = [100, 500, 1000, 2000]

for N in sizes:
    print(f"\n{'='*80}")
    print(f"Graph: {N} nodes (scale-free, k=3)")
    print(f"{'='*80}")
    
    G = nx.barabasi_albert_graph(N, 3, seed=42)
    
    # Initialize phases
    for i in G.nodes():
        G.nodes[i]['phase'] = np.random.uniform(0, 2*np.pi)
        G.nodes[i]['delta_nfr'] = 0.1
        G.nodes[i]['vf'] = 1.0
        G.nodes[i]['coherence'] = 0.8
    
    # Phase gradient benchmark
    times_grad = []
    for _ in range(5):
        t0 = time.perf_counter()
        grad = compute_phase_gradient(G)
        t1 = time.perf_counter()
        times_grad.append((t1 - t0) * 1000)
    
    mean_grad = np.mean(times_grad)
    std_grad = np.std(times_grad)
    
    # Phase curvature benchmark
    times_curv = []
    for _ in range(5):
        t0 = time.perf_counter()
        curv = compute_phase_curvature(G)
        t1 = time.perf_counter()
        times_curv.append((t1 - t0) * 1000)
    
    mean_curv = np.mean(times_curv)
    std_curv = np.std(times_curv)
    
    print(f"\n|∇φ| (phase gradient):")
    print(f"  Mean: {mean_grad:.3f} ms")
    print(f"  Std:  {std_grad:.3f} ms")
    print(f"  Range: {min(times_grad):.3f} - {max(times_grad):.3f} ms")
    
    print(f"\nK_φ (phase curvature):")
    print(f"  Mean: {mean_curv:.3f} ms")
    print(f"  Std:  {std_curv:.3f} ms")
    print(f"  Range: {min(times_curv):.3f} - {max(times_curv):.3f} ms")
    
    print(f"\nTotal (|∇φ| + K_φ): {mean_grad + mean_curv:.3f} ms")

print("\n" + "=" * 80)
print("✅ Benchmark complete")
print("=" * 80)
