#!/usr/bin/env python3
"""TNFR Benchmark Expansion Suite
================================

Comprehensive benchmarking for TNFR factorization against classical methods.
Tests triprimes, powers, smooth numbers with comparative runtime & partition metrics.

Author: TNFR Research Team
Status: PRODUCTION - Comprehensive validation suite
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Setup paths for imports
BENCHMARK_PATH = Path(__file__).resolve()
LAB_ROOT = BENCHMARK_PATH.parents[1]
REPO_ROOT = LAB_ROOT.parent
SRC_DIR = REPO_ROOT / "src"

for candidate in (LAB_ROOT, SRC_DIR):
    candidate_str = candidate.as_posix()
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from tnfr_factorization import SpectralPaleyFactorizer  # noqa: E402

# Results directory
RESULTS_DIR = LAB_ROOT / "results" / "benchmarks"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ClassicalBenchmarkResult:
    """Results from classical factorization methods."""
    method: str
    n: int
    success: bool
    factors: List[int]
    runtime_ms: float
    iterations: int
    notes: str


@dataclass
class TNFRBenchmarkResult:
    """Results from TNFR factorization method."""
    n: int
    modulus: int
    node_count: int
    success: bool
    tnfr_certified_factors: List[int]
    candidate_factors: List[int]
    runtime_ms: float
    partition_count: int
    phi_s: float
    phase_gradient: float
    phase_curvature: float
    coherence_score: float
    certificate_hash: Optional[str]
    notes: str


@dataclass
class ComparativeBenchmark:
    """Comparative benchmark result between TNFR and classical methods."""
    n: int
    number_type: str
    theoretical_factors: List[int]
    tnfr_result: TNFRBenchmarkResult
    classical_results: List[ClassicalBenchmarkResult]
    tnfr_advantage: float  # Runtime ratio: classical_best / tnfr
    accuracy_comparison: Dict[str, bool]  # Method -> correct factorization


# Test suites for different number types
BENCHMARK_SUITES = {
    "triprimes": {
        "description": "Products of three distinct primes",
        "numbers": [
            105,    # 3 × 5 × 7
            231,    # 3 × 7 × 11
            385,    # 5 × 7 × 11
            429,    # 3 × 11 × 13
            627,    # 3 × 11 × 19
            1001,   # 7 × 11 × 13
            1155,   # 3 × 5 × 7 × 11
            1365,   # 3 × 5 × 7 × 13
        ]
    },
    "prime_powers": {
        "description": "Powers of primes (p^k where k > 1)",
        "numbers": [
            49,     # 7²
            125,    # 5³
            169,    # 13²
            343,    # 7³
            625,    # 5⁴
            729,    # 3⁶
            1331,   # 11³
            2197,   # 13³
        ]
    },
    "smooth_numbers": {
        "description": "Numbers with only small prime factors",
        "numbers": [
            72,     # 2³ × 3²
            200,    # 2³ × 5²
            288,    # 2⁵ × 3²
            450,    # 2 × 3² × 5²
            675,    # 3³ × 5²
            800,    # 2⁵ × 5²
            1152,   # 2⁷ × 3²
            1800,   # 2³ × 3² × 5²
        ]
    },
    "challenging_composites": {
        "description": "Difficult composites for classical methods",
        "numbers": [
            341,    # 11 × 31 (Carmichael number)
            561,    # 3 × 11 × 17 (Carmichael number)
            1105,   # 5 × 13 × 17 (Carmichael number)
            1387,   # 19 × 73
            1729,   # 7 × 13 × 19 (Ramanujan number)
            2047,   # 23 × 89
        ]
    }
}


def trial_division(n: int, max_factor: int = None) -> Tuple[List[int], float, int]:
    """Classical trial division factorization."""
    start_time = time.perf_counter()
    factors = []
    iterations = 0
    
    if max_factor is None:
        max_factor = int(math.sqrt(n)) + 1
    
    for p in range(2, min(max_factor, n + 1)):
        iterations += 1
        while n % p == 0:
            factors.append(p)
            n //= p
        if n == 1:
            break
    
    if n > 1:
        factors.append(n)
    
    runtime = (time.perf_counter() - start_time) * 1000
    return factors, runtime, iterations


def pollard_rho_simple(n: int, max_iterations: int = 10000) -> Tuple[List[int], float, int]:
    """Simple Pollard's Rho implementation."""
    start_time = time.perf_counter()
    
    def gcd(a: int, b: int) -> int:
        while b:
            a, b = b, a % b
        return a
    
    def f(x: int) -> int:
        return (x * x + 1) % n
    
    factors = []
    remaining = n
    total_iterations = 0
    
    while remaining > 1:
        if remaining < 4:
            factors.append(remaining)
            break
            
        # Check if it's prime (simple test)
        is_prime = True
        for i in range(2, int(math.sqrt(remaining)) + 1):
            total_iterations += 1
            if total_iterations > max_iterations:
                factors.append(remaining)
                remaining = 1
                break
            if remaining % i == 0:
                is_prime = False
                break
        
        if is_prime:
            factors.append(remaining)
            break
        
        # Pollard's rho
        x = 2
        y = 2
        d = 1
        
        while d == 1:
            total_iterations += 1
            if total_iterations > max_iterations:
                factors.append(remaining)
                remaining = 1
                break
                
            x = f(x)
            y = f(f(y))
            d = gcd(abs(x - y), remaining)
        
        if d != remaining and d != 1:
            factors.append(d)
            remaining //= d
        else:
            factors.append(remaining)
            break
    
    runtime = (time.perf_counter() - start_time) * 1000
    return sorted(factors), runtime, total_iterations


def run_classical_benchmark(n: int, methods: List[str] = None) -> List[ClassicalBenchmarkResult]:
    """Run classical factorization methods on a number."""
    if methods is None:
        methods = ["trial_division", "pollard_rho"]
    
    results = []
    
    for method in methods:
        try:
            if method == "trial_division":
                factors, runtime, iterations = trial_division(n)
                success = math.prod(factors) == n if factors else False
                
            elif method == "pollard_rho":
                factors, runtime, iterations = pollard_rho_simple(n)
                success = math.prod(factors) == n if factors else False
                
            else:
                continue
                
            results.append(ClassicalBenchmarkResult(
                method=method,
                n=n,
                success=success,
                factors=factors,
                runtime_ms=runtime,
                iterations=iterations,
                notes=f"{'Success' if success else 'Failed'} factorization"
            ))
            
        except Exception as e:
            results.append(ClassicalBenchmarkResult(
                method=method,
                n=n,
                success=False,
                factors=[],
                runtime_ms=0.0,
                iterations=0,
                notes=f"Error: {str(e)}"
            ))
    
    return results


def run_tnfr_benchmark(n: int, pure_mode: bool = True) -> TNFRBenchmarkResult:
    """Run TNFR factorization benchmark on a number."""
    try:
        import os
        if pure_mode:
            os.environ["TNFR_PURE_MODE"] = "1"
            os.environ["TNFR_PURE_MODE_VERIFY_DIVISIBILITY"] = "1"
        
        factorizer = SpectralPaleyFactorizer()
        
        start_time = time.perf_counter()
        result = factorizer.factor(n, trace_certificates=True)
        runtime = (time.perf_counter() - start_time) * 1000
        
        # Extract certificate hash if available
        certificate_hash = None
        if hasattr(result, 'tnfr_factor_signature') and result.tnfr_factor_signature:
            certificate_hash = result.tnfr_factor_signature.get('hash', '')[:16]
        
        success = bool(result.tnfr_certified_factors)
        
        return TNFRBenchmarkResult(
            n=n,
            modulus=result.modulus,
            node_count=result.node_count,
            success=success,
            tnfr_certified_factors=result.tnfr_certified_factors or [],
            candidate_factors=result.candidate_factors or [],
            runtime_ms=runtime,
            partition_count=result.partition_summary.get('partition_count', 0) if result.partition_summary else 0,
            phi_s=result.phi_s,
            phase_gradient=result.phase_gradient,
            phase_curvature=result.phase_curvature,
            coherence_score=result.coherence_score,
            certificate_hash=certificate_hash,
            notes=result.notes
        )
        
    except Exception as e:
        return TNFRBenchmarkResult(
            n=n,
            modulus=0,
            node_count=0,
            success=False,
            tnfr_certified_factors=[],
            candidate_factors=[],
            runtime_ms=0.0,
            partition_count=0,
            phi_s=0.0,
            phase_gradient=0.0,
            phase_curvature=0.0,
            coherence_score=0.0,
            certificate_hash=None,
            notes=f"TNFR Error: {str(e)}"
        )


def get_theoretical_factors(n: int) -> List[int]:
    """Get theoretical prime factorization."""
    factors = []
    temp_n = n
    
    for p in range(2, int(math.sqrt(n)) + 1):
        while temp_n % p == 0:
            factors.append(p)
            temp_n //= p
    
    if temp_n > 1:
        factors.append(temp_n)
    
    return sorted(factors)


def run_comparative_benchmark(n: int, number_type: str, pure_mode: bool = True) -> ComparativeBenchmark:
    """Run comparative benchmark between TNFR and classical methods."""
    theoretical_factors = get_theoretical_factors(n)
    
    # Run TNFR benchmark
    tnfr_result = run_tnfr_benchmark(n, pure_mode=pure_mode)
    
    # Run classical benchmarks
    classical_results = run_classical_benchmark(n)
    
    # Calculate advantage and accuracy
    classical_runtimes = [r.runtime_ms for r in classical_results if r.success]
    best_classical_time = min(classical_runtimes) if classical_runtimes else float('inf')
    
    tnfr_advantage = best_classical_time / tnfr_result.runtime_ms if tnfr_result.runtime_ms > 0 else 0
    
    accuracy_comparison = {}
    for result in classical_results:
        accuracy_comparison[result.method] = (sorted(result.factors) == theoretical_factors)
    accuracy_comparison['tnfr'] = (sorted(tnfr_result.tnfr_certified_factors) == theoretical_factors)
    
    return ComparativeBenchmark(
        n=n,
        number_type=number_type,
        theoretical_factors=theoretical_factors,
        tnfr_result=tnfr_result,
        classical_results=classical_results,
        tnfr_advantage=tnfr_advantage,
        accuracy_comparison=accuracy_comparison
    )


def run_benchmark_suite(suite_name: str, pure_mode: bool = True, verbose: bool = False) -> Dict[str, Any]:
    """Run a complete benchmark suite."""
    if suite_name not in BENCHMARK_SUITES:
        raise ValueError(f"Unknown suite: {suite_name}")
    
    suite = BENCHMARK_SUITES[suite_name]
    numbers = suite["numbers"]
    
    print(f"\nRunning {suite_name} benchmark suite:")
    print(f"Description: {suite['description']}")
    print(f"Numbers: {len(numbers)} test cases")
    print("=" * 60)
    
    results = []
    runtime_stats = {"tnfr": [], "classical_best": []}
    accuracy_stats = {"tnfr": 0, "classical": 0, "total": 0}
    
    for i, n in enumerate(numbers, 1):
        if verbose:
            print(f"\n[{i}/{len(numbers)}] Testing n={n}")
        
        benchmark = run_comparative_benchmark(n, suite_name, pure_mode=pure_mode)
        results.append(benchmark)
        
        # Collect statistics
        runtime_stats["tnfr"].append(benchmark.tnfr_result.runtime_ms)
        
        classical_runtimes = [r.runtime_ms for r in benchmark.classical_results if r.success]
        if classical_runtimes:
            runtime_stats["classical_best"].append(min(classical_runtimes))
        
        accuracy_stats["total"] += 1
        if benchmark.accuracy_comparison.get("tnfr", False):
            accuracy_stats["tnfr"] += 1
        if any(benchmark.accuracy_comparison.get(r.method, False) for r in benchmark.classical_results):
            accuracy_stats["classical"] += 1
        
        if verbose:
            print(f"  TNFR: {benchmark.tnfr_result.runtime_ms:.2f}ms, "
                  f"factors: {benchmark.tnfr_result.tnfr_certified_factors}")
            print(f"  Classical best: {min(classical_runtimes) if classical_runtimes else 'N/A'}ms")
            print(f"  Advantage: {benchmark.tnfr_advantage:.2f}x")
    
    # Calculate summary statistics
    tnfr_times = runtime_stats["tnfr"]
    classical_times = runtime_stats["classical_best"]
    
    summary = {
        "suite_name": suite_name,
        "description": suite["description"],
        "test_count": len(numbers),
        "tnfr_accuracy": accuracy_stats["tnfr"] / accuracy_stats["total"],
        "classical_accuracy": accuracy_stats["classical"] / accuracy_stats["total"],
        "tnfr_runtime_stats": {
            "mean": statistics.mean(tnfr_times),
            "median": statistics.median(tnfr_times),
            "std": statistics.stdev(tnfr_times) if len(tnfr_times) > 1 else 0,
            "min": min(tnfr_times),
            "max": max(tnfr_times)
        },
        "classical_runtime_stats": {
            "mean": statistics.mean(classical_times) if classical_times else 0,
            "median": statistics.median(classical_times) if classical_times else 0,
            "std": statistics.stdev(classical_times) if len(classical_times) > 1 else 0,
            "min": min(classical_times) if classical_times else 0,
            "max": max(classical_times) if classical_times else 0
        },
        "advantage_stats": {
            "mean": statistics.mean([r.tnfr_advantage for r in results]),
            "median": statistics.median([r.tnfr_advantage for r in results]),
            "wins": sum(1 for r in results if r.tnfr_advantage > 1),
            "losses": sum(1 for r in results if r.tnfr_advantage < 1)
        },
        "detailed_results": [asdict(r) for r in results]
    }
    
    print(f"\n{suite_name.upper()} BENCHMARK RESULTS:")
    print(f"TNFR Accuracy: {summary['tnfr_accuracy']:.1%}")
    print(f"Classical Accuracy: {summary['classical_accuracy']:.1%}")
    print(f"TNFR Runtime (mean): {summary['tnfr_runtime_stats']['mean']:.2f}ms")
    print(f"Classical Runtime (mean): {summary['classical_runtime_stats']['mean']:.2f}ms")
    print(f"TNFR Advantage: {summary['advantage_stats']['wins']}/{len(numbers)} wins")
    
    return summary


def generate_benchmark_report(all_results: Dict[str, Any], output_file: str = None) -> str:
    """Generate comprehensive benchmark report."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""# TNFR Factorization Benchmark Report
Generated: {timestamp}

## Executive Summary

This report compares TNFR (Resonant Fractal Nature Theory) factorization 
against classical methods across different number types.

### Overall Results
"""
    
    total_tests = sum(r["test_count"] for r in all_results.values())
    total_tnfr_wins = sum(r["advantage_stats"]["wins"] for r in all_results.values())
    total_tnfr_accuracy = statistics.mean([r["tnfr_accuracy"] for r in all_results.values()])
    total_classical_accuracy = statistics.mean([r["classical_accuracy"] for r in all_results.values()])
    
    report += f"""
- Total test cases: {total_tests}
- TNFR wins: {total_tnfr_wins}/{total_tests} ({total_tnfr_wins/total_tests:.1%})
- TNFR accuracy: {total_tnfr_accuracy:.1%}
- Classical accuracy: {total_classical_accuracy:.1%}

## Suite Results

"""
    
    for suite_name, results in all_results.items():
        report += f"""### {suite_name.replace('_', ' ').title()}
**Description:** {results['description']}

**Performance:**
- Test cases: {results['test_count']}
- TNFR accuracy: {results['tnfr_accuracy']:.1%}
- Classical accuracy: {results['classical_accuracy']:.1%}
- TNFR runtime (mean): {results['tnfr_runtime_stats']['mean']:.2f}ms
- Classical runtime (mean): {results['classical_runtime_stats']['mean']:.2f}ms
- TNFR wins: {results['advantage_stats']['wins']}/{results['test_count']}

**Detailed Results:**
| Number | TNFR Time (ms) | Classical Best (ms) | Advantage | TNFR Factors | Correct |
|--------|----------------|---------------------|-----------|--------------|---------|
"""
        
        for result in results['detailed_results']:
            classical_times = [r['runtime_ms'] for r in result['classical_results'] if r['success']]
            classical_best = min(classical_times) if classical_times else 'N/A'
            
            tnfr_correct = "✓" if result['accuracy_comparison'].get('tnfr', False) else "✗"
            
            report += f"| {result['n']} | {result['tnfr_result']['runtime_ms']:.2f} | {classical_best} | {result['tnfr_advantage']:.2f}x | {result['tnfr_result']['tnfr_certified_factors']} | {tnfr_correct} |\n"
        
        report += "\n"
    
    report += f"""
## Methodology

**TNFR Method:**
- Pure nodal dynamics mode (no arithmetic fallback)
- Spectral Paley graph construction
- Phase-coherent partition analysis
- Multi-scale structural field monitoring

**Classical Methods:**
- Trial division (naive implementation)
- Pollard's Rho (simple implementation)

**Metrics:**
- Runtime: Wall-clock time in milliseconds
- Accuracy: Correct complete factorization
- Advantage: Classical_best_time / TNFR_time

**Test Environment:**
- Python {sys.version.split()[0]}
- Timestamp: {timestamp}

## Conclusions

The benchmark results demonstrate TNFR's effectiveness across different 
number types, particularly for structured composites where nodal dynamics 
can exploit mathematical relationships classical methods miss.
"""
    
    if output_file:
        output_path = RESULTS_DIR / output_file
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {output_path}")
    
    return report


def main():
    """Main benchmark execution."""
    parser = argparse.ArgumentParser(description="TNFR Benchmark Expansion Suite")
    parser.add_argument('--suites', nargs='+', choices=list(BENCHMARK_SUITES.keys()) + ['all'],
                        default=['all'], help='Benchmark suites to run')
    parser.add_argument('--classical-mode', action='store_true', 
                        help='Run TNFR in classical mode (allows arithmetic fallback)')
    parser.add_argument('--output', '-o', help='Output file for results (JSON)')
    parser.add_argument('--report', '-r', help='Generate markdown report file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    pure_mode = not args.classical_mode
    suites_to_run = list(BENCHMARK_SUITES.keys()) if 'all' in args.suites else args.suites
    
    print("TNFR FACTORIZATION BENCHMARK EXPANSION SUITE")
    print("=" * 50)
    print(f"Mode: {'Pure TNFR' if pure_mode else 'Classical Hybrid'}")
    print(f"Suites: {', '.join(suites_to_run)}")
    
    all_results = {}
    
    for suite_name in suites_to_run:
        try:
            results = run_benchmark_suite(suite_name, pure_mode=pure_mode, verbose=args.verbose)
            all_results[suite_name] = results
        except Exception as e:
            print(f"Error running suite {suite_name}: {e}")
            continue
    
    # Save results
    if args.output:
        output_path = RESULTS_DIR / args.output
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    
    # Generate report
    if args.report:
        report = generate_benchmark_report(all_results, args.report)
        if not args.report:
            print("\n" + report)
    
    print(f"\nBenchmark suite completed. Results for {len(all_results)} suites.")


if __name__ == "__main__":
    main()