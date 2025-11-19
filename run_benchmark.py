#!/usr/bin/env python3
"""Simple benchmark runner - validates the ~70% speedup claim."""

import sys
from pathlib import Path

# Ensure we can import tnfr
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    import tnfr
    print(f"[OK] TNFR imported from {tnfr.__file__}")
except ImportError as e:
    print(f"[ERROR] Cannot import tnfr: {e}")
    print(f"[INFO] Python path: {sys.path[:3]}")
    sys.exit(1)

# Now run the benchmark
try:
    from benchmarks.benchmark_optimization_tracks import (
        BenchmarkConfig,
        OptimizationBenchmark,
        BenchmarkReporter,
    )
    
    print("\n[BENCHMARK] Starting optimization tracks validation...\n")
    
    config = BenchmarkConfig()
    benchmark = OptimizationBenchmark(config)
    reporter = BenchmarkReporter()
    
    print("[TRACK 1] Phase Fusion Benchmark...")
    track1_results = benchmark.benchmark_phase_fusion()
    
    print("[TRACK 2] Grammar Memoization Benchmark...")
    track2_results = benchmark.benchmark_grammar_memoization()
    
    print("[TRACK 3] Phi_s Optimization Benchmark...")
    track3_results = benchmark.benchmark_phi_s_optimization()
    
    print("[TRACK 4] Telemetry Pipeline Benchmark...")
    track4_results = benchmark.benchmark_telemetry_pipeline()
    
    print("[TRACK 5] Precision Modes Benchmark (standard vs high)...")
    track5_results = benchmark.benchmark_precision_modes()
    
    # Compile results
    all_results = {
        "phase_fusion": track1_results,
        "grammar_memoization": track2_results,
        "phi_s_optimization": track3_results,
        "telemetry_pipeline": track4_results,
        "precision_modes": track5_results,
    }
    
    # Report
    reporter.print_summary(all_results)
    reporter.export_json(all_results, "benchmark_results.json")
    
    print("\n[OK] Benchmark completed successfully!")

except Exception as e:
    print(f"\n[ERROR] Benchmark failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
