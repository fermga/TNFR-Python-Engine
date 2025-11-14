"""Phase 2 Performance Baseline - Post-Split Measurements.

Measures performance of critical paths after definitions.py split:
1. Import times (cold start)
2. Operator instantiation
3. Grammar validation
4. Metrics computation
5. Memory footprint

Results establish baseline for future optimizations.
"""

import gc
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def measure_import_time(module_name: str) -> float:
    """Measure cold import time for a module."""
    # Clear module from cache
    if module_name in sys.modules:
        del sys.modules[module_name]
    
    # Clear all tnfr modules
    to_clear = [k for k in sys.modules.keys() if k.startswith("tnfr")]
    for k in to_clear:
        del sys.modules[k]
    
    gc.collect()
    
    start = time.perf_counter()
    __import__(module_name)
    elapsed = time.perf_counter() - start
    
    return elapsed * 1000  # Convert to ms


def measure_operator_instantiation() -> dict[str, float]:
    """Measure time to instantiate all 13 operators."""
    from tnfr.operators.definitions import (
        Coherence,
        Contraction,
        Coupling,
        Dissonance,
        Emission,
        Expansion,
        Mutation,
        Reception,
        Recursivity,
        Resonance,
        SelfOrganization,
        Silence,
        Transition,
    )
    
    operators = [
        ("Emission", Emission),
        ("Reception", Reception),
        ("Coherence", Coherence),
        ("Dissonance", Dissonance),
        ("Coupling", Coupling),
        ("Resonance", Resonance),
        ("Silence", Silence),
        ("Expansion", Expansion),
        ("Contraction", Contraction),
        ("SelfOrganization", SelfOrganization),
        ("Mutation", Mutation),
        ("Transition", Transition),
        ("Recursivity", Recursivity),
    ]
    
    results = {}
    
    for name, OpClass in operators:
        # Warm up
        for _ in range(10):
            _ = OpClass()
        
        # Measure
        iterations = 1000
        start = time.perf_counter()
        for _ in range(iterations):
            _ = OpClass()
        elapsed = time.perf_counter() - start
        
        results[name] = (elapsed / iterations) * 1_000_000  # μs per op
    
    return results


def measure_grammar_validation() -> dict[str, float]:
    """Measure grammar validation performance."""
    from tnfr import create_nfr
    from tnfr.operators.definitions import Coherence, Dissonance, Emission
    from tnfr.operators.grammar import validate_sequence
    
    # Create test graph
    G, node = create_nfr("perf_test", epi=0.5, vf=1.0)
    G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True
    
    # Test sequences
    sequences = {
        "simple_valid": [Emission(), Coherence()],
        "complex_valid": [Emission(), Coherence(), Dissonance(), Coherence()],
        "invalid": [Coherence()],  # Missing generator
    }
    
    results = {}
    iterations = 100
    
    for seq_name, seq in sequences.items():
        start = time.perf_counter()
        for _ in range(iterations):
            try:
                validate_sequence(G, node, seq)
            except Exception:
                pass  # Expected for invalid
        elapsed = time.perf_counter() - start
        
        results[seq_name] = (elapsed / iterations) * 1000  # ms per validation
    
    return results


def measure_metrics_computation() -> dict[str, float]:
    """Measure metrics computation performance."""
    from tnfr import create_nfr
    from tnfr.operators.definitions import Coherence, Emission
    
    # Create test graph
    G, node = create_nfr("metrics_test", epi=0.5, vf=1.0)
    
    results = {}
    iterations = 100
    
    # Emission metrics
    start = time.perf_counter()
    for _ in range(iterations):
        Emission()(G, node)
    elapsed = time.perf_counter() - start
    results["emission_with_metrics"] = (elapsed / iterations) * 1000
    
    # Coherence metrics
    start = time.perf_counter()
    for _ in range(iterations):
        Coherence()(G, node)
    elapsed = time.perf_counter() - start
    results["coherence_with_metrics"] = (elapsed / iterations) * 1000
    
    return results


def get_memory_usage() -> float:
    """Get current process memory usage in MB."""
    import psutil
    
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def main():
    """Run all performance benchmarks."""
    print("=" * 70)
    print("TNFR Phase 2 Performance Baseline")
    print("Post-definitions.py split measurements")
    print("=" * 70)
    print()
    
    # Check if psutil is available
    try:
        import psutil
        memory_available = True
    except ImportError:
        memory_available = False
        print("WARNING: psutil not installed - memory measurements disabled")
        print()
    
    # 1. Import Times
    print("[Import Performance]")
    print("-" * 70)
    
    modules_to_test = [
        "tnfr.operators.definitions",
        "tnfr.operators.grammar",
        "tnfr.operators.metrics",
        "tnfr",
    ]
    
    import_times = {}
    for module in modules_to_test:
        time_ms = measure_import_time(module)
        import_times[module] = time_ms
        print(f"  {module:40s} {time_ms:6.2f} ms")
    
    print(f"\n  Total import time: {sum(import_times.values()):.2f} ms")
    print()
    
    # 2. Operator Instantiation
    print("[Operator Instantiation]")
    print("-" * 70)
    
    op_times = measure_operator_instantiation()
    for name, time_us in sorted(op_times.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name:30s} {time_us:6.2f} μs")
    
    avg_time = sum(op_times.values()) / len(op_times)
    print(f"\n  Average: {avg_time:.2f} μs per instantiation")
    print()
    
    # 3. Grammar Validation
    print("[Grammar Validation]")
    print("-" * 70)
    
    grammar_times = measure_grammar_validation()
    for name, time_ms in grammar_times.items():
        print(f"  {name:30s} {time_ms:6.3f} ms")
    print()
    
    # 4. Metrics Computation
    print("[Metrics Computation]")
    print("-" * 70)
    
    metrics_times = measure_metrics_computation()
    for name, time_ms in metrics_times.items():
        print(f"  {name:30s} {time_ms:6.3f} ms")
    print()
    
    # 5. Memory Usage
    if memory_available:
        print("[Memory Usage]")
        print("-" * 70)
        
        mem_before = get_memory_usage()
        
        # Load everything
        from tnfr import create_nfr
        from tnfr.operators.definitions import (
            Coherence,
            Emission,
            Resonance,
        )
        
        # Create some graphs
        graphs = []
        for i in range(10):
            G, node = create_nfr(f"test_{i}", epi=0.5, vf=1.0)
            Emission()(G, node)
            Coherence()(G, node)
            Resonance()(G, node)
            graphs.append(G)
        
        mem_after = get_memory_usage()
        
        print(f"  Baseline (imports only): {mem_before:.2f} MB")
        print(f"  With 10 graphs:          {mem_after:.2f} MB")
        print(f"  Delta:                   {mem_after - mem_before:.2f} MB")
        print(f"  Per graph:               {(mem_after - mem_before) / 10:.3f} MB")
        print()
    
    # Summary
    print("=" * 70)
    print("[Summary]")
    print("-" * 70)
    print(f"  Total import time:       {sum(import_times.values()):.2f} ms")
    print(f"  Avg operator creation:   {avg_time:.2f} μs")
    print(f"  Grammar validation:      {grammar_times['simple_valid']:.3f} ms (simple)")
    print(f"  Operator execution:      {metrics_times['emission_with_metrics']:.3f} ms (Emission)")
    if memory_available:
        print(f"  Memory per graph:        {(mem_after - mem_before) / 10:.3f} MB")
    print("=" * 70)
    print()
    print("[SUCCESS] Baseline established successfully")
    print("          Results can be used for future performance regression testing")


if __name__ == "__main__":
    main()
