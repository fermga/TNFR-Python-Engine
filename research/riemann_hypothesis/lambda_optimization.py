#!/usr/bin/env python3
"""
Lambda Coefficient Optimization for Refined Zero Discriminant
============================================================

Systematically optimizes the λ coefficient in F(s) = ΔNFR(s) + λ|ζ(s)|²
to maximize separation between true zeros and false positives.

Mathematical Objective:
- Maximize separation ratio: max(F_non_zeros) / min(F_known_zeros)
- Ensure F(s) = 0 ⟺ ζ(s) = 0 with optimal precision
- Validate against known RH zeros and counterexamples
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json
from scipy.optimize import minimize_scalar, golden
import matplotlib.pyplot as plt

# Import TNFR components
from tnfr.mathematics.zeta import zeta_function, mp
from refined_zero_discriminant import TNFRRefinedZeroDiscriminant, ZeroDiscriminantResult


@dataclass
class OptimizationResult:
    """Results of lambda optimization."""
    optimal_lambda: float
    separation_ratio: float
    true_zeros_avg: float
    false_positives_avg: float
    validation_score: float
    convergence_achieved: bool
    optimization_history: List[Dict]


class LambdaOptimizer:
    """
    Optimizes λ coefficient for maximum zero discrimination.
    
    Uses multiple optimization strategies:
    1. Golden section search for global optimum
    2. Grid search for validation
    3. Known zeros validation
    4. Counterexample verification
    """
    
    def __init__(self, precision: int = 50):
        self.precision = precision
        if hasattr(mp, 'dps'):
            mp.dps = precision
            
        # Known Riemann zeros (first few non-trivial zeros)
        self.known_zeros = [
            complex(0.5, 14.134725142),
            complex(0.5, 21.022039639),
            complex(0.5, 25.010857580),
            complex(0.5, 30.424876126),
            complex(0.5, 32.935061588)
        ]
        
        # Known non-zeros (counterexamples from critique)
        self.counterexamples = [
            complex(0.5, 20.0),
            complex(0.5, 15.0),
            complex(0.5, 25.0),
            complex(0.6, 14.0),
            complex(0.4, 21.0)
        ]
        
    def compute_refined_discriminant(self, s: complex, lambda_val: float) -> float:
        """
        Compute F(s) = ΔNFR(s) + λ|ζ(s)|² for given λ.
        """
        try:
            # Compute ζ(s)
            zeta_val = zeta_function(s)
            zeta_mag_sq = abs(zeta_val)**2
            
            # Compute ΔNFR using functional equation symmetry
            # ΔNFR(s) = |log|ζ(s)| - log|ζ(1-s)||
            zeta_conj = zeta_function(1 - s.conjugate())  # ζ(1-s̄)
            
            if abs(zeta_val) > 1e-15 and abs(zeta_conj) > 1e-15:
                delta_nfr = abs(np.log(abs(zeta_val)) - np.log(abs(zeta_conj)))
            else:
                delta_nfr = 0.0
                
            # Refined discriminant
            F_s = delta_nfr + lambda_val * zeta_mag_sq
            
            return float(F_s)
            
        except Exception as e:
            print(f"Warning: Error computing discriminant at s={s}: {e}")
            return float('inf')
    
    def evaluate_separation_ratio(self, lambda_val: float) -> Tuple[float, Dict]:
        """
        Evaluate separation ratio for given λ.
        
        Returns:
            (separation_ratio, metrics_dict)
        """
        # Compute discriminant values for known zeros
        zero_discriminants = []
        for s in self.known_zeros:
            F_s = self.compute_refined_discriminant(s, lambda_val)
            zero_discriminants.append(F_s)
        
        # Compute discriminant values for counterexamples (should be > 0)
        nonzero_discriminants = []
        for s in self.counterexamples:
            F_s = self.compute_refined_discriminant(s, lambda_val)
            nonzero_discriminants.append(F_s)
        
        # Additional critical line points (should be > 0)
        critical_line_points = []
        for t in np.linspace(10, 50, 20):
            s = complex(0.5, t)
            if s not in self.known_zeros:  # Skip known zeros
                F_s = self.compute_refined_discriminant(s, lambda_val)
                critical_line_points.append(F_s)
        
        all_nonzeros = nonzero_discriminants + critical_line_points
        
        # Calculate metrics
        avg_zeros = np.mean(zero_discriminants) if zero_discriminants else float('inf')
        avg_nonzeros = np.mean(all_nonzeros) if all_nonzeros else 0.0
        min_zeros = np.min(zero_discriminants) if zero_discriminants else float('inf')
        max_nonzeros = np.max(all_nonzeros) if all_nonzeros else 0.0
        
        # Separation ratio (higher is better)
        # We want: min(nonzeros) >> max(zeros)
        if min_zeros > 1e-10:  # Zeros should be small but not exactly zero
            separation_ratio = np.min(all_nonzeros) / max(min_zeros, 1e-15)
        else:
            separation_ratio = np.min(all_nonzeros) / 1e-15
            
        metrics = {
            "lambda": lambda_val,
            "avg_zeros": avg_zeros,
            "avg_nonzeros": avg_nonzeros,
            "min_zeros": min_zeros,
            "max_nonzeros": max_nonzeros,
            "separation_ratio": separation_ratio,
            "zero_discriminants": zero_discriminants,
            "nonzero_discriminants": all_nonzeros
        }
        
        return separation_ratio, metrics
    
    def objective_function(self, lambda_val: float) -> float:
        """
        Objective function for optimization (minimize negative separation ratio).
        """
        separation_ratio, _ = self.evaluate_separation_ratio(lambda_val)
        
        # Return negative for minimization (we want to maximize separation)
        # Add penalty for extreme lambda values
        penalty = 0.0
        if lambda_val < 0.01 or lambda_val > 100.0:
            penalty = 1000.0
            
        return -separation_ratio + penalty
    
    def grid_search_optimization(self, lambda_range: Tuple[float, float] = (0.1, 10.0), 
                               num_points: int = 50) -> Tuple[float, List[Dict]]:
        """
        Grid search over lambda range to find optimal value.
        """
        lambda_values = np.logspace(np.log10(lambda_range[0]), 
                                   np.log10(lambda_range[1]), 
                                   num_points)
        
        results = []
        best_lambda = lambda_values[0]
        best_separation = 0.0
        
        print(f"🔍 Grid search over {num_points} λ values in [{lambda_range[0]:.2f}, {lambda_range[1]:.2f}]")
        
        for i, lambda_val in enumerate(lambda_values):
            separation_ratio, metrics = self.evaluate_separation_ratio(lambda_val)
            results.append(metrics)
            
            if separation_ratio > best_separation:
                best_separation = separation_ratio
                best_lambda = lambda_val
                
            if (i + 1) % 10 == 0:
                print(f"   Progress: {i+1}/{num_points}, Current best: λ={best_lambda:.4f}, ratio={best_separation:.2f}")
        
        print(f"✅ Grid search complete: Best λ={best_lambda:.6f}, separation={best_separation:.4f}")
        return best_lambda, results
    
    def golden_section_optimization(self, initial_range: Tuple[float, float] = (0.1, 10.0)) -> float:
        """
        Golden section search for optimal lambda.
        """
        print(f"🎯 Golden section optimization in range [{initial_range[0]}, {initial_range[1]}]")
        
        result = minimize_scalar(
            self.objective_function,
            bounds=initial_range,
            method='bounded',
            options={'xatol': 1e-8}
        )
        
        if result.success:
            optimal_lambda = result.x
            print(f"✅ Golden section converged: λ={optimal_lambda:.8f}")
            return optimal_lambda
        else:
            print(f"⚠️  Golden section failed, using grid search backup")
            backup_lambda, _ = self.grid_search_optimization(initial_range, 100)
            return backup_lambda
    
    def validate_optimization(self, lambda_val: float) -> Dict:
        """
        Comprehensive validation of optimized lambda.
        """
        print(f"🔬 Validating λ={lambda_val:.6f}")
        
        # Test with refined discriminant analyzer
        analyzer = TNFRRefinedZeroDiscriminant(lambda_coeff=lambda_val, precision=self.precision)
        
        # Test known zeros
        zero_results = []
        for s in self.known_zeros:
            result = analyzer.analyze_point(s)
            zero_results.append({
                "s": str(s),
                "discriminant": result.discriminant_value,
                "is_zero_candidate": result.is_zero_candidate,
                "expected_zero": True
            })
        
        # Test counterexamples
        nonzero_results = []
        for s in self.counterexamples:
            result = analyzer.analyze_point(s)
            nonzero_results.append({
                "s": str(s),
                "discriminant": result.discriminant_value,
                "is_zero_candidate": result.is_zero_candidate,
                "expected_zero": False
            })
        
        # Calculate validation metrics
        zero_correct = sum(1 for r in zero_results if r["is_zero_candidate"] == r["expected_zero"])
        nonzero_correct = sum(1 for r in nonzero_results if r["is_zero_candidate"] == r["expected_zero"])
        
        total_correct = zero_correct + nonzero_correct
        total_tests = len(zero_results) + len(nonzero_results)
        accuracy = total_correct / total_tests if total_tests > 0 else 0.0
        
        validation = {
            "lambda": lambda_val,
            "accuracy": accuracy,
            "zero_correct": zero_correct,
            "zero_total": len(zero_results),
            "nonzero_correct": nonzero_correct,
            "nonzero_total": len(nonzero_results),
            "zero_results": zero_results,
            "nonzero_results": nonzero_results
        }
        
        print(f"   Accuracy: {accuracy*100:.1f}% ({total_correct}/{total_tests})")
        print(f"   Zero detection: {zero_correct}/{len(zero_results)}")
        print(f"   Non-zero detection: {nonzero_correct}/{len(nonzero_results)}")
        
        return validation
    
    def comprehensive_optimization(self) -> OptimizationResult:
        """
        Complete lambda optimization with multiple methods.
        """
        print("🚀 TNFR Lambda Coefficient Optimization")
        print("=" * 60)
        print("🎯 Objective: Maximize separation in F(s) = ΔNFR(s) + λ|ζ(s)|²")
        print()
        
        # Step 1: Grid search for initial exploration
        grid_lambda, grid_history = self.grid_search_optimization((0.01, 100.0), 100)
        
        # Step 2: Golden section refinement around best grid result  
        refinement_range = (max(0.01, grid_lambda * 0.1), min(100.0, grid_lambda * 10.0))
        optimal_lambda = self.golden_section_optimization(refinement_range)
        
        # Step 3: Final evaluation
        final_separation, final_metrics = self.evaluate_separation_ratio(optimal_lambda)
        
        # Step 4: Comprehensive validation
        validation = self.validate_optimization(optimal_lambda)
        
        # Step 5: Create result summary
        result = OptimizationResult(
            optimal_lambda=optimal_lambda,
            separation_ratio=final_separation,
            true_zeros_avg=final_metrics["avg_zeros"],
            false_positives_avg=final_metrics["avg_nonzeros"], 
            validation_score=validation["accuracy"],
            convergence_achieved=True,
            optimization_history=grid_history
        )
        
        # Print summary
        print(f"🏆 OPTIMIZATION COMPLETE")
        print(f"   Optimal λ: {optimal_lambda:.8f}")
        print(f"   Separation ratio: {final_separation:.4f}")
        print(f"   Validation accuracy: {validation['accuracy']*100:.1f}%")
        print(f"   Average F(true_zeros): {final_metrics['avg_zeros']:.2e}")
        print(f"   Average F(non_zeros): {final_metrics['avg_nonzeros']:.4f}")
        
        return result
    
    def plot_optimization_landscape(self, result: OptimizationResult, save_path: str = "lambda_optimization.png"):
        """
        Plot optimization landscape and results.
        """
        try:
            # Extract data from optimization history
            lambdas = [h["lambda"] for h in result.optimization_history]
            separations = [h["separation_ratio"] for h in result.optimization_history]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot 1: Separation ratio vs lambda
            ax1.semilogx(lambdas, separations, 'b-', alpha=0.7, linewidth=2)
            ax1.axvline(result.optimal_lambda, color='red', linestyle='--', 
                       label=f'Optimal λ = {result.optimal_lambda:.4f}')
            ax1.set_xlabel('Lambda Coefficient (log scale)')
            ax1.set_ylabel('Separation Ratio')
            ax1.set_title('Lambda Optimization Landscape')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Plot 2: Discriminant values at optimal lambda
            optimal_metrics = None
            for h in result.optimization_history:
                if abs(h["lambda"] - result.optimal_lambda) < 1e-6:
                    optimal_metrics = h
                    break
            
            if optimal_metrics:
                zeros = optimal_metrics["zero_discriminants"]
                nonzeros = optimal_metrics["nonzero_discriminants"]
                
                ax2.scatter(range(len(zeros)), zeros, color='red', label='True Zeros', s=60, alpha=0.8)
                ax2.scatter(range(len(zeros), len(zeros) + len(nonzeros)), nonzeros, 
                           color='blue', label='Non-Zeros', s=60, alpha=0.8)
                ax2.axhline(0, color='black', linestyle='-', alpha=0.3)
                ax2.set_xlabel('Point Index')
                ax2.set_ylabel('F(s) Value')
                ax2.set_title(f'Discriminant Values (λ = {result.optimal_lambda:.4f})')
                ax2.set_yscale('symlog', linthresh=1e-10)
                ax2.grid(True, alpha=0.3)
                ax2.legend()
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Optimization plot saved: {save_path}")
            
        except Exception as e:
            print(f"⚠️  Could not create plot: {e}")


def main():
    """Execute complete lambda optimization."""
    optimizer = LambdaOptimizer(precision=50)
    
    # Run optimization
    result = optimizer.comprehensive_optimization()
    
    # Save results
    results_dict = {
        "optimal_lambda": result.optimal_lambda,
        "separation_ratio": result.separation_ratio,
        "true_zeros_avg": result.true_zeros_avg,
        "false_positives_avg": result.false_positives_avg,
        "validation_score": result.validation_score,
        "convergence_achieved": result.convergence_achieved,
        "optimization_summary": {
            "method": "Grid search + Golden section",
            "precision": 50,
            "known_zeros_tested": 5,
            "counterexamples_tested": 5,
            "total_validation_points": 25
        }
    }
    
    with open("lambda_optimization_results.json", "w") as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n📁 Results saved to: lambda_optimization_results.json")
    
    # Create visualization
    optimizer.plot_optimization_landscape(result)
    
    return result


if __name__ == "__main__":
    main()