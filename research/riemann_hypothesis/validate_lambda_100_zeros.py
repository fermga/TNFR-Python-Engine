#!/usr/bin/env python3
"""
Lambda Validation with 100 Known RH Zeros
==========================================

Comprehensive validation of the optimized λ = 0.05462277 coefficient using 
the first 100 known non-trivial zeros of the Riemann zeta function.

This validator implements:
1. Systematic testing of F(s) = ΔNFR(s) + λ|ζ(s)|² on all 100 zeros
2. Statistical analysis of discrimination performance
3. Comparison with counterexamples (non-zeros)  
4. Performance benchmarking and precision analysis
5. Comprehensive reporting with visualizations

Key Metrics:
- Zero Detection Rate: F(s) ≈ 0 for known zeros
- False Positive Rate: F(s) >> 0 for non-zeros
- Separation Ratio: Mean(non-zeros) / Mean(zeros) 
- Classification Accuracy: Binary classification performance

Author: TNFR Research Team  
Date: November 28, 2025
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional
import time
from dataclasses import dataclass
from pathlib import Path
import json
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Local imports
from rh_zeros_database import RHZerosDatabase
from refined_zero_discriminant import TNFRRefinedZeroDiscriminant

# TNFR optimization imports (optional)
TNFR_OPTIMIZATIONS_AVAILABLE = False
try:
    from tnfr.utils.cache import cache_tnfr_computation  # noqa: F401
    from tnfr.mathematics.zeta import zeta_function, structural_pressure  # noqa: F401
    TNFR_OPTIMIZATIONS_AVAILABLE = True
except ImportError:
    pass


@dataclass
class ValidationResult:
    """Results from λ validation on known zeros."""
    
    # Test configuration
    lambda_value: float
    zeros_tested: int
    counterexamples_tested: int
    
    # Discrimination results
    zero_f_values: List[float]
    nonzero_f_values: List[float]
    
    # Statistical metrics
    zero_detection_rate: float          # Fraction with |F(s)| < threshold
    false_positive_rate: float          # Fraction of non-zeros with |F(s)| < threshold
    separation_ratio: float             # Mean(non-zeros) / Mean(zeros)
    classification_accuracy: float      # Overall binary classification accuracy
    
    # Performance metrics
    computation_time: float
    average_time_per_zero: float
    
    # Additional statistics
    zero_f_mean: float
    zero_f_std: float
    nonzero_f_mean: float
    nonzero_f_std: float
    optimal_threshold: float


class LambdaValidator:
    """Validator for optimal λ coefficient using RH zeros with maximum optimization."""
    
    def __init__(self, lambda_value: float = 0.05462277, use_multiprocessing: bool = True,
                 batch_size: int = 500, num_workers: Optional[int] = None):
        """
        Initialize validator with optimized λ coefficient.
        
        Args:
            lambda_value: The λ coefficient from optimization (default: optimal value)
            use_multiprocessing: Enable parallel processing for large datasets
            batch_size: Number of zeros to process per batch
            num_workers: Number of worker processes (default: CPU count - 1)
        """
        self.lambda_value = lambda_value
        
        # Initialize components
        self.zeros_db = RHZerosDatabase()
        self.discriminant = TNFRRefinedZeroDiscriminant(lambda_coeff=lambda_value)
        
        # Validation parameters
        self.zero_threshold = 1e-6  # Threshold for considering F(s) ≈ 0
        
        # Optimization parameters
        self.use_multiprocessing = use_multiprocessing and mp.cpu_count() > 1
        self.batch_size = batch_size
        self.num_workers = num_workers or max(1, mp.cpu_count() - 1)
        
        # Cache for discriminant computations
        self._discriminant_cache = {}
        
        print(f"Lambda Validator initialized with lambda = {lambda_value:.8f}")
        if self.use_multiprocessing:
            print(f"Multiprocessing enabled: {self.num_workers} workers, batch size {self.batch_size}")
        if TNFR_OPTIMIZATIONS_AVAILABLE:
            print("TNFR optimizations available (cache + vectorization)")
        
    @lru_cache(maxsize=50000)
    def _compute_discriminant_cached(self, s_real: float, s_imag: float) -> float:
        """
        Cached discriminant computation for single point.
        
        Args:
            s_real: Real part of s
            s_imag: Imaginary part of s
            
        Returns:
            Discriminant F(s) value
        """
        s = complex(s_real, s_imag)
        result = self.discriminant.compute_refined_discriminant(s)
        return float(np.abs(result.discriminant_value))
    
    def validate_single_zero(self, s: complex, is_zero: bool = True) -> Tuple[float, float]:
        """
        Validate discriminant F(s) for a single point with caching.
        
        Args:
            s: Complex number to test
            is_zero: Whether s is expected to be a zero
            
        Returns:
            Tuple of (F_value, computation_time)
        """
        start_time = time.time()
        
        try:
            # Use cached computation
            f_value = self._compute_discriminant_cached(s.real, s.imag)
            computation_time = time.time() - start_time
            
            return f_value, computation_time
            
        except Exception as e:
            print(f"⚠️  Error computing F({s}): {e}")
            return float('inf'), time.time() - start_time
    
    def _validate_batch(self, zeros_batch: List[complex]) -> List[Tuple[float, float]]:
        """
        Validate a batch of zeros (for multiprocessing).
        
        Args:
            zeros_batch: List of complex zeros to validate
            
        Returns:
            List of (f_value, comp_time) tuples
        """
        results = []
        for s in zeros_batch:
            f_val, comp_time = self.validate_single_zero(s, is_zero=True)
            results.append((f_val, comp_time))
        return results
    
    def validate_all_zeros(self, max_zeros: int = 100) -> ValidationResult:
        """
        Comprehensive validation using all available known zeros.
        
        Args:
            max_zeros: Maximum number of zeros to test
            
        Returns:
            Complete validation results
        """
        print(f"\n🔍 Starting validation with first {max_zeros} RH zeros...")
        print(f"🎛️  Using λ = {self.lambda_value:.8f}")
        
        # Get known zeros and counterexamples
        known_zeros = self.zeros_db.get_zeros_complex(max_zeros)
        counterexamples = self.zeros_db.generate_counterexamples(max_zeros)
        
        print(f"📊 Testing {len(known_zeros)} known zeros")
        print(f"📊 Testing {len(counterexamples)} counterexamples")
        
        # Test known zeros
        zero_f_values = []
        zero_times = []
        
        print("\n✅ Testing known zeros...")
        # Adaptive progress reporting based on dataset size
        if max_zeros <= 100:
            progress_interval = 10
        elif max_zeros <= 1000:
            progress_interval = 100
        elif max_zeros <= 10000:
            progress_interval = 1000
        else:
            progress_interval = 5000
        
        # Use batch processing for large datasets
        if self.use_multiprocessing and len(known_zeros) > 1000:
            print(f"⚡ Using parallel batch processing with {self.num_workers} workers")
            
            # Split into batches
            batches = [known_zeros[i:i+self.batch_size] 
                      for i in range(0, len(known_zeros), self.batch_size)]
            
            # Process batches in parallel
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                future_to_batch = {executor.submit(self._validate_batch, batch): i 
                                  for i, batch in enumerate(batches)}
                
                completed = 0
                for future in as_completed(future_to_batch):
                    batch_results = future.result()
                    for f_val, comp_time in batch_results:
                        zero_f_values.append(f_val)
                        zero_times.append(comp_time)
                        completed += 1
                        
                        if completed <= 10 or completed % progress_interval == 0:
                            print(f"  ρ_{completed:5d}: Progress {completed}/{len(known_zeros)} "
                                 f"[{np.mean(zero_times[-100:]):.3f}s/zero avg]")
        else:
            # Sequential processing for smaller datasets
            for i, s in enumerate(known_zeros, 1):
                f_val, comp_time = self.validate_single_zero(s, is_zero=True)
                zero_f_values.append(f_val)
                zero_times.append(comp_time)
                
                if i <= 10 or i % progress_interval == 0:  # Show progress
                    print(f"  ρ_{i:5d}: F({s:.3f}) = {f_val:.2e} [{comp_time:.3f}s]")
        
        # Test counterexamples
        nonzero_f_values = []
        nonzero_times = []
        
        print("\n🚫 Testing counterexamples...")
        
        # Use batch processing for counterexamples too
        if self.use_multiprocessing and len(counterexamples) > 1000:
            batches = [counterexamples[i:i+self.batch_size] 
                      for i in range(0, len(counterexamples), self.batch_size)]
            
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                future_to_batch = {executor.submit(self._validate_batch, batch): i 
                                  for i, batch in enumerate(batches)}
                
                completed = 0
                for future in as_completed(future_to_batch):
                    batch_results = future.result()
                    for f_val, comp_time in batch_results:
                        nonzero_f_values.append(f_val)
                        nonzero_times.append(comp_time)
                        completed += 1
                        
                        if completed <= 10 or completed % progress_interval == 0:
                            print(f"  Non_{completed:5d}: Progress {completed}/{len(counterexamples)} "
                                 f"[{np.mean(nonzero_times[-100:]):.3f}s/zero avg]")
        else:
            for i, s in enumerate(counterexamples, 1):
                f_val, comp_time = self.validate_single_zero(s, is_zero=False)
                nonzero_f_values.append(f_val)
                nonzero_times.append(comp_time)
                
                if i <= 10 or i % progress_interval == 0:  # Show progress
                    print(f"  Non_{i:5d}: F({s:.3f}) = {f_val:.2e} [{comp_time:.3f}s]")
        
        # Compute statistics
        total_time = sum(zero_times) + sum(nonzero_times)
        avg_time = total_time / (len(known_zeros) + len(counterexamples))
        
        # Detection rates
        zeros_detected = sum(1 for f in zero_f_values if f < self.zero_threshold)
        false_positives = sum(1 for f in nonzero_f_values if f < self.zero_threshold)
        
        zero_detection_rate = zeros_detected / len(zero_f_values)
        false_positive_rate = false_positives / len(nonzero_f_values)
        
        # Separation metrics
        zero_f_mean = np.mean(zero_f_values)
        nonzero_f_mean = np.mean(nonzero_f_values)
        separation_ratio = nonzero_f_mean / zero_f_mean if zero_f_mean > 0 else float('inf')
        
        # Classification accuracy (binary classification)
        correct_predictions = zeros_detected + (len(nonzero_f_values) - false_positives)
        total_predictions = len(zero_f_values) + len(nonzero_f_values)
        classification_accuracy = correct_predictions / total_predictions
        
        # Find optimal threshold
        optimal_threshold = self._find_optimal_threshold(zero_f_values, nonzero_f_values)
        
        # Create result object
        result = ValidationResult(
            lambda_value=self.lambda_value,
            zeros_tested=len(zero_f_values),
            counterexamples_tested=len(nonzero_f_values),
            zero_f_values=zero_f_values,
            nonzero_f_values=nonzero_f_values,
            zero_detection_rate=zero_detection_rate,
            false_positive_rate=false_positive_rate,
            separation_ratio=separation_ratio,
            classification_accuracy=classification_accuracy,
            computation_time=total_time,
            average_time_per_zero=avg_time,
            zero_f_mean=zero_f_mean,
            zero_f_std=np.std(zero_f_values),
            nonzero_f_mean=nonzero_f_mean,
            nonzero_f_std=np.std(nonzero_f_values),
            optimal_threshold=optimal_threshold
        )
        
        return result
    
    def _find_optimal_threshold(self, zero_f_values: List[float], 
                              nonzero_f_values: List[float]) -> float:
        """
        Find optimal threshold that maximizes classification accuracy.
        
        Args:
            zero_f_values: F(s) values for known zeros
            nonzero_f_values: F(s) values for non-zeros
            
        Returns:
            Optimal threshold value
        """
        # Try different thresholds
        all_values = sorted(zero_f_values + nonzero_f_values)
        best_threshold = self.zero_threshold
        best_accuracy = 0
        
        for threshold in all_values:
            # Count correct classifications
            zeros_correct = sum(1 for f in zero_f_values if f <= threshold)
            nonzeros_correct = sum(1 for f in nonzero_f_values if f > threshold)
            
            accuracy = (zeros_correct + nonzeros_correct) / (len(zero_f_values) + len(nonzero_f_values))
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
        
        return best_threshold
    
    def generate_report(self, result: ValidationResult) -> str:
        """
        Generate comprehensive validation report.
        
        Args:
            result: Validation results
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("🎯 TNFR Lambda Validation Report")
        report.append("=" * 50)
        report.append("")
        
        # Configuration
        report.append("📋 Configuration:")
        report.append(f"  λ coefficient: {result.lambda_value:.8f}")
        report.append(f"  Known zeros tested: {result.zeros_tested}")
        report.append(f"  Counterexamples tested: {result.counterexamples_tested}")
        report.append(f"  Detection threshold: {self.zero_threshold:.0e}")
        report.append("")
        
        # Performance Summary
        report.append("🎯 Performance Summary:")
        report.append(f"  Zero Detection Rate: {result.zero_detection_rate:.2%} ({result.zeros_tested * result.zero_detection_rate:.0f}/{result.zeros_tested})")
        report.append(f"  False Positive Rate: {result.false_positive_rate:.2%} ({result.counterexamples_tested * result.false_positive_rate:.0f}/{result.counterexamples_tested})")
        report.append(f"  Classification Accuracy: {result.classification_accuracy:.2%}")
        report.append(f"  Separation Ratio: {result.separation_ratio:.2f}×")
        report.append("")
        
        # Statistical Analysis
        report.append("📊 Statistical Analysis:")
        report.append(f"  Known Zeros F(s):")
        report.append(f"    Mean: {result.zero_f_mean:.2e}")
        report.append(f"    Std:  {result.zero_f_std:.2e}")
        report.append(f"    Min:  {min(result.zero_f_values):.2e}")
        report.append(f"    Max:  {max(result.zero_f_values):.2e}")
        report.append("")
        report.append(f"  Non-Zeros F(s):")
        report.append(f"    Mean: {result.nonzero_f_mean:.2e}")
        report.append(f"    Std:  {result.nonzero_f_std:.2e}")
        report.append(f"    Min:  {min(result.nonzero_f_values):.2e}")
        report.append(f"    Max:  {max(result.nonzero_f_values):.2e}")
        report.append("")
        
        # Optimization Analysis
        report.append("🔧 Optimization Analysis:")
        report.append(f"  Current threshold: {self.zero_threshold:.0e}")
        report.append(f"  Optimal threshold: {result.optimal_threshold:.2e}")
        report.append(f"  Threshold efficiency: {(result.classification_accuracy):.2%}")
        report.append("")
        
        # Performance Metrics
        report.append("⚡ Performance Metrics:")
        report.append(f"  Total computation time: {result.computation_time:.2f} seconds")
        report.append(f"  Average time per test: {result.average_time_per_zero:.3f} seconds")
        report.append(f"  Throughput: {(result.zeros_tested + result.counterexamples_tested) / result.computation_time:.1f} tests/second")
        report.append("")
        
        # Conclusions
        if result.classification_accuracy >= 0.90:
            grade = "EXCELLENT"
        elif result.classification_accuracy >= 0.80:
            grade = "GOOD"
        elif result.classification_accuracy >= 0.70:
            grade = "ACCEPTABLE"
        else:
            grade = "NEEDS IMPROVEMENT"
            
        report.append("✨ Conclusions:")
        report.append(f"  Overall Grade: {grade}")
        report.append(f"  λ = {result.lambda_value:.6f} demonstrates {grade.lower()} discrimination")
        
        if result.separation_ratio > 10:
            report.append(f"  Strong separation ({result.separation_ratio:.1f}×) between zeros and non-zeros")
        elif result.separation_ratio > 2:
            report.append(f"  Moderate separation ({result.separation_ratio:.1f}×) between zeros and non-zeros")
        else:
            report.append(f"  Weak separation ({result.separation_ratio:.1f}×) - may need refinement")
        
        return "\n".join(report)
    
    def create_visualizations(self, result: ValidationResult, output_dir: str = "validation_plots"):
        """
        Create comprehensive visualization plots.
        
        Args:
            result: Validation results
            output_dir: Directory to save plots
        """
        Path(output_dir).mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Distribution comparison
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Histogram comparison
        ax1.hist(result.zero_f_values, bins=30, alpha=0.7, label='Known Zeros', color='green')
        ax1.hist(result.nonzero_f_values, bins=30, alpha=0.7, label='Non-Zeros', color='red')
        ax1.set_xlabel('|F(s)| Value')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of F(s) Values')
        ax1.legend()
        ax1.set_yscale('log')
        
        # Box plot comparison
        data_for_box = [result.zero_f_values, result.nonzero_f_values]
        ax2.boxplot(data_for_box, labels=['Known Zeros', 'Non-Zeros'])
        ax2.set_ylabel('|F(s)| Value')
        ax2.set_title('F(s) Distribution Comparison')
        ax2.set_yscale('log')
        
        # Scatter plot of values by index
        ax3.scatter(range(len(result.zero_f_values)), result.zero_f_values, 
                   alpha=0.6, label='Known Zeros', color='green', s=20)
        ax3.scatter(range(len(result.nonzero_f_values)), result.nonzero_f_values,
                   alpha=0.6, label='Non-Zeros', color='red', s=20)
        ax3.axhline(y=self.zero_threshold, color='black', linestyle='--', 
                   label=f'Threshold ({self.zero_threshold:.0e})')
        ax3.set_xlabel('Test Index')
        ax3.set_ylabel('|F(s)| Value')
        ax3.set_title('F(s) Values by Test Order')
        ax3.set_yscale('log')
        ax3.legend()
        
        # Performance metrics
        categories = ['Zero\nDetection', 'False Positive\nPrevention', 'Overall\nAccuracy']
        values = [result.zero_detection_rate, 1-result.false_positive_rate, result.classification_accuracy]
        colors = ['green' if v >= 0.9 else 'orange' if v >= 0.8 else 'red' for v in values]
        
        bars = ax4.bar(categories, values, color=colors, alpha=0.7)
        ax4.set_ylabel('Performance Rate')
        ax4.set_title('Classification Performance')
        ax4.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.2%}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/lambda_validation_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. ROC-style analysis
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Threshold sensitivity
        thresholds = np.logspace(-10, -1, 50)
        accuracies = []
        zero_rates = []
        false_pos_rates = []
        
        for thresh in thresholds:
            zeros_detected = sum(1 for f in result.zero_f_values if f <= thresh)
            false_positives = sum(1 for f in result.nonzero_f_values if f <= thresh)
            
            zero_rate = zeros_detected / len(result.zero_f_values)
            false_pos_rate = false_positives / len(result.nonzero_f_values)
            accuracy = (zeros_detected + len(result.nonzero_f_values) - false_positives) / (len(result.zero_f_values) + len(result.nonzero_f_values))
            
            zero_rates.append(zero_rate)
            false_pos_rates.append(false_pos_rate)
            accuracies.append(accuracy)
        
        ax1.plot(thresholds, zero_rates, 'g-', label='Zero Detection Rate', linewidth=2)
        ax1.plot(thresholds, false_pos_rates, 'r-', label='False Positive Rate', linewidth=2)
        ax1.plot(thresholds, accuracies, 'b-', label='Overall Accuracy', linewidth=2)
        ax1.axvline(x=self.zero_threshold, color='black', linestyle='--', 
                   label=f'Current Threshold')
        ax1.axvline(x=result.optimal_threshold, color='purple', linestyle='--',
                   label=f'Optimal Threshold')
        ax1.set_xlabel('Threshold Value')
        ax1.set_ylabel('Rate')
        ax1.set_title('Threshold Sensitivity Analysis')
        ax1.set_xscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # ROC curve
        ax2.plot(false_pos_rates, zero_rates, 'b-', linewidth=2)
        ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate (Zero Detection)')
        ax2.set_title('ROC Curve for Zero Classification')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Mark current operating point
        current_tpr = result.zero_detection_rate
        current_fpr = result.false_positive_rate
        ax2.plot(current_fpr, current_tpr, 'ro', markersize=8, 
                label=f'Current (λ={result.lambda_value:.4f})')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/lambda_validation_roc.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Visualizations saved to {output_dir}/")
    
    def export_results(self, result: ValidationResult, output_file: str = "lambda_validation_results.json"):
        """
        Export results to JSON file.
        
        Args:
            result: Validation results
            output_file: Output JSON file path
        """
        data = {
            'metadata': {
                'validator': 'TNFR Lambda Validator',
                'lambda_value': result.lambda_value,
                'validation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'zeros_database_source': self.zeros_db.metadata['source']
            },
            'configuration': {
                'zeros_tested': result.zeros_tested,
                'counterexamples_tested': result.counterexamples_tested,
                'zero_threshold': self.zero_threshold
            },
            'performance_metrics': {
                'zero_detection_rate': result.zero_detection_rate,
                'false_positive_rate': result.false_positive_rate,
                'classification_accuracy': result.classification_accuracy,
                'separation_ratio': result.separation_ratio,
                'optimal_threshold': result.optimal_threshold
            },
            'statistics': {
                'zero_f_values': {
                    'mean': result.zero_f_mean,
                    'std': result.zero_f_std,
                    'min': min(result.zero_f_values),
                    'max': max(result.zero_f_values)
                },
                'nonzero_f_values': {
                    'mean': result.nonzero_f_mean,
                    'std': result.nonzero_f_std,
                    'min': min(result.nonzero_f_values),
                    'max': max(result.nonzero_f_values)
                }
            },
            'performance': {
                'total_computation_time': result.computation_time,
                'average_time_per_test': result.average_time_per_zero,
                'throughput_tests_per_second': (result.zeros_tested + result.counterexamples_tested) / result.computation_time
            },
            'raw_data': {
                'zero_f_values': result.zero_f_values,
                'nonzero_f_values': result.nonzero_f_values
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"📄 Results exported to {output_file}")

def main():
    """Run comprehensive λ validation with known RH zeros."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate lambda coefficient with RH zeros")
    parser.add_argument("--max-zeros", type=int, default=100,
                       help="Maximum number of zeros to test (default: 100)")
    parser.add_argument("--lambda-value", type=float, default=0.05462277217684343,
                       help="Lambda coefficient to test")
    args = parser.parse_args()
    
    print(f"🎯 TNFR Lambda Validation with RH Zeros")
    print("=" * 55)
    
    # Initialize validator with optimized λ
    validator = LambdaValidator(lambda_value=args.lambda_value)
    
    # Report dataset source
    print(f"📁 Dataset: {validator.zeros_db.describe_source()}")
    
    # Run comprehensive validation
    print(f"\n🚀 Starting validation with λ = {args.lambda_value:.8f}")
    print(f"🎯 Testing up to {args.max_zeros:,} zeros")
    result = validator.validate_all_zeros(max_zeros=args.max_zeros)
    
    # Generate and display report
    report = validator.generate_report(result)
    print(f"\n{report}")
    
    # Create visualizations
    validator.create_visualizations(result)
    
    # Export results
    validator.export_results(result)
    
    # Final summary
    print("\n" + "=" * 60)
    print("✨ VALIDATION COMPLETE")
    print("=" * 60)
    print(f"🎯 λ = {args.lambda_value:.6f}")
    print(f"📊 Classification Accuracy: {result.classification_accuracy:.2%}")
    print(f"🔍 Zero Detection Rate: {result.zero_detection_rate:.2%}")
    print(f"📈 Separation Ratio: {result.separation_ratio:.2f}×")
    
    if result.classification_accuracy >= 0.90:
        print("✅ EXCELLENT performance - λ coefficient validated!")
    elif result.classification_accuracy >= 0.80:
        print("✅ GOOD performance - λ coefficient acceptable")
    else:
        print("⚠️  Performance below expectations - consider refinement")


if __name__ == "__main__":
    main()
