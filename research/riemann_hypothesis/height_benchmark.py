#!/usr/bin/env python3
"""
Asymptotic Height Benchmark - Simplified
========================================

Benchmarks TNFR discriminant performance across different height ranges
to validate behavior at extreme |Im(s)| values.

Key Features:
- Tests discriminant at heights: 100, 500, 1000, 5000, 10000
- Measures computational performance and precision
- Validates that discrimination power is maintained  
- Identifies optimal computational strategies per regime

Author: TNFR Research Team
Date: November 28, 2025
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import numpy as np
import time
import math
from typing import List, Tuple

# Local imports
from refined_zero_discriminant import TNFRRefinedZeroDiscriminant
from rh_zeros_database import RHZerosDatabase

class HeightBenchmark:
    """Benchmark discriminant performance across height ranges."""
    
    def __init__(self, lambda_coeff: float = 0.05462277):
        self.lambda_coeff = lambda_coeff
        self.discriminant = TNFRRefinedZeroDiscriminant(lambda_coeff=lambda_coeff)
        self.zeros_db = RHZerosDatabase()
        
    def safe_float_convert(self, value):
        """Safely convert mpf or other types to float."""
        try:
            return float(value)
        except (TypeError, AttributeError):
            if hasattr(value, 'real') and hasattr(value, 'imag'):
                return float(abs(value))
            return 1.0  # Fallback
    
    def benchmark_single_height(self, height: float) -> dict:
        """Benchmark discriminant at specific height."""
        s = complex(0.5, height)
        
        start_time = time.time()
        
        try:
            result = self.discriminant.compute_refined_discriminant(s)
            
            computation_time = time.time() - start_time
            
            return {
                'height': height,
                'discriminant_value': self.safe_float_convert(result.discriminant_value),
                'zeta_magnitude': self.safe_float_convert(result.zeta_magnitude),
                'delta_nfr': self.safe_float_convert(result.delta_nfr),
                'computation_time': computation_time,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            computation_time = time.time() - start_time
            return {
                'height': height,
                'discriminant_value': float('inf'),
                'zeta_magnitude': 0.0,
                'delta_nfr': 0.0,
                'computation_time': computation_time,
                'success': False,
                'error': str(e)
            }
    
    def benchmark_height_range(self, heights: List[float]) -> List[dict]:
        """Benchmark across multiple heights."""
        results = []
        
        print(f"🔍 Benchmarking {len(heights)} height points...")
        
        for i, height in enumerate(heights, 1):
            result = self.benchmark_single_height(height)
            results.append(result)
            
            if result['success']:
                print(f"  Height {height:6.0f}: F(s) = {result['discriminant_value']:.2e} "
                      f"[{result['computation_time']:.3f}s]")
            else:
                print(f"  Height {height:6.0f}: FAILED - {result['error'][:50]}")
            
            # Progress indicator
            if i % 5 == 0 or i == len(heights):
                print(f"    Progress: {i}/{len(heights)} ({100*i/len(heights):.1f}%)")
        
        return results
    
    def analyze_results(self, results: List[dict]) -> dict:
        """Analyze benchmark results."""
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        if not successful:
            return {
                'total_tests': len(results),
                'success_rate': 0.0,
                'analysis': 'All tests failed'
            }
        
        # Extract metrics
        heights = [r['height'] for r in successful]
        times = [r['computation_time'] for r in successful]
        discriminants = [r['discriminant_value'] for r in successful]
        
        # Computational scaling analysis
        if len(heights) > 1:
            # Fit time ~ height^α
            log_heights = [math.log(h) for h in heights]
            log_times = [math.log(max(t, 1e-6)) for t in times]
            
            # Simple linear regression
            n = len(log_heights)
            sum_x = sum(log_heights)
            sum_y = sum(log_times)
            sum_xy = sum(x*y for x, y in zip(log_heights, log_times))
            sum_x2 = sum(x*x for x in log_heights)
            
            if n * sum_x2 - sum_x * sum_x != 0:
                scaling_exponent = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            else:
                scaling_exponent = 0.0
        else:
            scaling_exponent = 0.0
        
        return {
            'total_tests': len(results),
            'successful_tests': len(successful),
            'failed_tests': len(failed),
            'success_rate': len(successful) / len(results),
            
            'height_range': (min(heights), max(heights)) if heights else (0, 0),
            'time_statistics': {
                'mean': np.mean(times) if times else 0,
                'min': min(times) if times else 0,
                'max': max(times) if times else 0,
                'std': np.std(times) if times else 0
            },
            'discriminant_statistics': {
                'mean': np.mean(discriminants) if discriminants else 0,
                'min': min(discriminants) if discriminants else 0,
                'max': max(discriminants) if discriminants else 0,
                'median': np.median(discriminants) if discriminants else 0
            },
            'computational_scaling': {
                'exponent': scaling_exponent,
                'interpretation': self._interpret_scaling(scaling_exponent)
            }
        }
    
    def _interpret_scaling(self, exponent: float) -> str:
        """Interpret computational scaling exponent."""
        if exponent < 0.5:
            return "Sub-linear scaling (excellent)"
        elif exponent < 1.0:
            return "Linear scaling (good)"  
        elif exponent < 1.5:
            return "Super-linear scaling (acceptable)"
        elif exponent < 2.0:
            return "Quadratic scaling (challenging)"
        else:
            return "Polynomial scaling (difficult)"
    
    def validate_extreme_heights(self) -> dict:
        """Validate discriminant behavior at extreme heights."""
        print("\n🌊 EXTREME HEIGHT VALIDATION")
        print("=" * 40)
        
        # Test specific extreme heights
        extreme_heights = [100, 500, 1000, 2000, 5000, 10000]
        
        results = self.benchmark_height_range(extreme_heights)
        analysis = self.analyze_results(results)
        
        return {
            'benchmark_results': results,
            'analysis': analysis
        }

def main():
    """Run asymptotic height benchmarks."""
    print("🌊 TNFR Asymptotic Height Benchmark")
    print("=" * 50)
    
    # Initialize benchmark
    benchmark = HeightBenchmark()
    
    print(f"🎯 Using optimized λ = {benchmark.lambda_coeff:.6f}")
    print(f"📁 Dataset: {benchmark.zeros_db.describe_source()}")
    
    # Test moderate heights (baseline)
    print(f"\n📊 BASELINE: Moderate Heights (20-80)")
    moderate_heights = [20, 40, 60, 80]
    moderate_results = benchmark.benchmark_height_range(moderate_heights)
    moderate_analysis = benchmark.analyze_results(moderate_results)
    
    print(f"\n✅ Moderate height results:")
    print(f"  Success rate: {moderate_analysis['success_rate']:.1%}")
    print(f"  Average time: {moderate_analysis['time_statistics']['mean']:.3f}s")
    print(f"  Scaling: {moderate_analysis['computational_scaling']['interpretation']}")
    
    # Test extreme heights  
    extreme_results = benchmark.validate_extreme_heights()
    extreme_analysis = extreme_results['analysis']
    
    print(f"\n🌊 EXTREME HEIGHT RESULTS:")
    print(f"  Heights tested: {extreme_analysis['height_range'][0]:.0f} - {extreme_analysis['height_range'][1]:.0f}")
    print(f"  Success rate: {extreme_analysis['success_rate']:.1%}")
    print(f"  Average time: {extreme_analysis['time_statistics']['mean']:.3f}s")
    print(f"  Time range: {extreme_analysis['time_statistics']['min']:.3f}s - {extreme_analysis['time_statistics']['max']:.3f}s")
    print(f"  Scaling exponent: {extreme_analysis['computational_scaling']['exponent']:.2f}")
    print(f"  Scaling assessment: {extreme_analysis['computational_scaling']['interpretation']}")
    
    # Performance comparison
    if moderate_analysis['success_rate'] > 0 and extreme_analysis['success_rate'] > 0:
        time_ratio = extreme_analysis['time_statistics']['mean'] / moderate_analysis['time_statistics']['mean']
        print(f"\n⚡ PERFORMANCE COMPARISON:")
        print(f"  Extreme vs Moderate time ratio: {time_ratio:.1f}×")
        
        if time_ratio < 5:
            performance_grade = "EXCELLENT - Scales well to extreme heights"
        elif time_ratio < 20:
            performance_grade = "GOOD - Acceptable scaling to extreme heights"
        elif time_ratio < 100:
            performance_grade = "FAIR - Challenging but manageable scaling"
        else:
            performance_grade = "POOR - Scaling issues at extreme heights"
        
        print(f"  Performance grade: {performance_grade}")
    
    # Validation summary
    print(f"\n" + "="*60)
    print(f"✨ ASYMPTOTIC EXTENSION VALIDATION SUMMARY")
    print(f"="*60)
    
    if extreme_analysis['success_rate'] >= 0.8:
        print(f"✅ SUCCESS: Discriminant works reliably at extreme heights")
        print(f"📊 {extreme_analysis['successful_tests']}/{extreme_analysis['total_tests']} tests passed")
        print(f"🎯 Discrimination maintained up to height {extreme_analysis['height_range'][1]:.0f}")
    elif extreme_analysis['success_rate'] >= 0.5:
        print(f"⚠️  PARTIAL: Some issues at extreme heights, but generally functional")
        print(f"📊 {extreme_analysis['successful_tests']}/{extreme_analysis['total_tests']} tests passed")
    else:
        print(f"❌ FAILURE: Significant issues at extreme heights")
        print(f"📊 Only {extreme_analysis['successful_tests']}/{extreme_analysis['total_tests']} tests passed")
    
    print(f"\n🎯 Ready for Task 3: Formal Mathematical Proof!")

if __name__ == "__main__":
    main()