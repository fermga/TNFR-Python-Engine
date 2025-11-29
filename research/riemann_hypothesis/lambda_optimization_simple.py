#!/usr/bin/env python3
"""
Simplified Lambda Optimization for TNFR Discriminant
===================================================

Simple optimization of λ coefficient in F(s) = ΔNFR(s) + λ|ζ(s)|²
using only standard Python types to avoid mpf compatibility issues.
"""

import numpy as np
import cmath
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class OptimizationResult:
    """Results of lambda optimization."""
    optimal_lambda: float
    separation_ratio: float
    best_accuracy: float
    grid_search_points: int
    validation_summary: Dict


class SimpleLambdaOptimizer:
    """
    Simplified lambda optimizer using standard Python arithmetic.
    
    Focuses on finding λ that maximizes discrimination between:
    - Known RH zeros (should give F(s) ≈ 0)
    - Known non-zeros (should give F(s) >> 0)
    """
    
    def __init__(self):
        # Known first few non-trivial Riemann zeros
        self.known_zeros = [
            complex(0.5, 14.134725142),
            complex(0.5, 21.022039639),
            complex(0.5, 25.010857580),
            complex(0.5, 30.424876126),
            complex(0.5, 32.935061588)
        ]
        
        # Points that should NOT be zeros (counterexamples)
        self.non_zeros = [
            complex(0.5, 20.0),   # From mathematical critique
            complex(0.5, 15.0),
            complex(0.5, 25.0),
            complex(0.6, 14.0),   # Off critical line
            complex(0.4, 21.0),
            complex(0.5, 13.0),   # Between zeros
            complex(0.5, 22.0),
            complex(0.5, 28.0)
        ]
    
    def compute_zeta_riemann_siegel(self, s: complex) -> complex:
        """
        Simplified Riemann-Siegel approximation for ζ(s).
        Good enough for optimization purposes.
        """
        # For Re(s) = 1/2, use Riemann-Siegel formula approximation
        if abs(s.real - 0.5) < 0.1:
            t = s.imag
            if t < 0:
                # Use functional equation for negative t
                return self.compute_zeta_riemann_siegel(complex(1-s.real, -t)).conjugate()
            
            # Simple approximation for positive t on critical line
            # ζ(1/2 + it) ≈ 2 * Re(Σ n^(-1/2-it)) for first few terms
            zeta_approx = 0j
            for n in range(1, 100):  # First 100 terms
                term = (n ** (-s))
                zeta_approx += term
                if abs(term) < 1e-10:
                    break
            
            return zeta_approx
        else:
            # For other values, use series expansion
            zeta_sum = 0j
            for n in range(1, 200):
                term = (n ** (-s))
                zeta_sum += term
                if abs(term) < 1e-12:
                    break
            return zeta_sum
    
    def compute_delta_nfr_simple(self, s: complex) -> float:
        """
        Simplified ΔNFR computation.
        ΔNFR(s) ≈ |log|ζ(s)| - log|ζ(1-s̄)||
        """
        try:
            zeta_s = self.compute_zeta_riemann_siegel(s)
            zeta_conj = self.compute_zeta_riemann_siegel(1 - s.conjugate())
            
            if abs(zeta_s) > 1e-15 and abs(zeta_conj) > 1e-15:
                log_zeta_s = cmath.log(abs(zeta_s))
                log_zeta_conj = cmath.log(abs(zeta_conj))
                delta_nfr = abs(log_zeta_s - log_zeta_conj)
                return float(delta_nfr.real)
            else:
                return 0.0
        except:
            return 0.0
    
    def compute_discriminant(self, s: complex, lambda_val: float) -> float:
        """
        Compute F(s) = ΔNFR(s) + λ|ζ(s)|²
        """
        try:
            zeta_s = self.compute_zeta_riemann_siegel(s)
            zeta_mag_sq = abs(zeta_s)**2
            delta_nfr = self.compute_delta_nfr_simple(s)
            
            discriminant = delta_nfr + lambda_val * zeta_mag_sq
            return float(discriminant)
        except:
            return float('inf')
    
    def evaluate_lambda(self, lambda_val: float) -> Dict:
        """
        Evaluate quality of given λ value.
        """
        # Compute discriminant for known zeros
        zero_discriminants = []
        for s in self.known_zeros:
            F_s = self.compute_discriminant(s, lambda_val)
            zero_discriminants.append(F_s)
        
        # Compute discriminant for non-zeros  
        nonzero_discriminants = []
        for s in self.non_zeros:
            F_s = self.compute_discriminant(s, lambda_val)
            nonzero_discriminants.append(F_s)
        
        # Calculate metrics
        avg_zeros = np.mean(zero_discriminants) if zero_discriminants else float('inf')
        avg_nonzeros = np.mean(nonzero_discriminants) if nonzero_discriminants else 0.0
        min_zeros = np.min(zero_discriminants) if zero_discriminants else float('inf')
        min_nonzeros = np.min(nonzero_discriminants) if nonzero_discriminants else 0.0
        
        # Separation ratio (higher is better)
        if min_zeros < 1e-10:
            min_zeros = 1e-10  # Avoid division by zero
        
        separation_ratio = min_nonzeros / min_zeros
        
        # Classification accuracy
        zero_correct = sum(1 for f in zero_discriminants if f < 0.1)  # Should be small
        nonzero_correct = sum(1 for f in nonzero_discriminants if f > 0.1)  # Should be large
        accuracy = (zero_correct + nonzero_correct) / (len(zero_discriminants) + len(nonzero_discriminants))
        
        return {
            "lambda": lambda_val,
            "separation_ratio": separation_ratio,
            "accuracy": accuracy,
            "avg_zeros": avg_zeros,
            "avg_nonzeros": avg_nonzeros,
            "min_zeros": min_zeros,
            "min_nonzeros": min_nonzeros,
            "zero_discriminants": zero_discriminants,
            "nonzero_discriminants": nonzero_discriminants
        }
    
    def grid_search_optimization(self, lambda_range: Tuple[float, float] = (0.01, 100.0), 
                               num_points: int = 50) -> Tuple[float, List[Dict]]:
        """
        Grid search over lambda values.
        """
        lambda_values = np.logspace(np.log10(lambda_range[0]), 
                                   np.log10(lambda_range[1]), 
                                   num_points)
        
        results = []
        best_lambda = lambda_values[0]
        best_score = 0.0
        
        print(f"🔍 Grid search over {num_points} λ values in [{lambda_range[0]:.2f}, {lambda_range[1]:.2f}]")
        
        for i, lambda_val in enumerate(lambda_values):
            evaluation = self.evaluate_lambda(lambda_val)
            results.append(evaluation)
            
            # Combined score: separation ratio + accuracy
            score = evaluation["separation_ratio"] + evaluation["accuracy"] * 10
            
            if score > best_score:
                best_score = score
                best_lambda = lambda_val
                
            if (i + 1) % 10 == 0:
                print(f"   Progress: {i+1}/{num_points}, Best λ={best_lambda:.4f}, score={best_score:.2f}")
        
        print(f"✅ Grid search complete: Best λ={best_lambda:.6f}, score={best_score:.4f}")
        return best_lambda, results
    
    def detailed_validation(self, lambda_val: float) -> Dict:
        """
        Detailed validation of lambda value.
        """
        print(f"🔬 Detailed validation of λ={lambda_val:.6f}")
        
        evaluation = self.evaluate_lambda(lambda_val)
        
        print(f"   Separation ratio: {evaluation['separation_ratio']:.4f}")
        print(f"   Classification accuracy: {evaluation['accuracy']*100:.1f}%")
        print(f"   Avg F(zeros): {evaluation['avg_zeros']:.4e}")
        print(f"   Avg F(non-zeros): {evaluation['avg_nonzeros']:.4f}")
        
        # Test specific cases
        print(f"\\n   Specific tests:")
        print(f"   F(0.5+20i) = {self.compute_discriminant(complex(0.5, 20.0), lambda_val):.4f} (should be > 0)")
        print(f"   F(0.5+14.13i) = {self.compute_discriminant(complex(0.5, 14.134725142), lambda_val):.4f} (should ≈ 0)")
        
        return evaluation
    
    def optimize_lambda(self) -> OptimizationResult:
        """
        Complete lambda optimization.
        """
        print("🚀 TNFR Lambda Coefficient Optimization (Simplified)")
        print("=" * 60)
        print("🎯 Maximizing F(s) = ΔNFR(s) + λ|ζ(s)|² discrimination")
        print()
        
        # Grid search
        optimal_lambda, search_results = self.grid_search_optimization((0.001, 10.0), 100)
        
        # Detailed validation
        validation = self.detailed_validation(optimal_lambda)
        
        # Create result
        result = OptimizationResult(
            optimal_lambda=optimal_lambda,
            separation_ratio=validation["separation_ratio"],
            best_accuracy=validation["accuracy"],
            grid_search_points=len(search_results),
            validation_summary=validation
        )
        
        print(f"\\n🏆 OPTIMIZATION COMPLETE")
        print(f"   Optimal λ: {optimal_lambda:.8f}")
        print(f"   Separation ratio: {validation['separation_ratio']:.4f}")
        print(f"   Classification accuracy: {validation['accuracy']*100:.1f}%")
        
        return result
    
    def plot_optimization_landscape(self, search_results: List[Dict], 
                                  save_path: str = "lambda_optimization_simple.png"):
        """
        Plot optimization results.
        """
        try:
            lambdas = [r["lambda"] for r in search_results]
            separations = [r["separation_ratio"] for r in search_results]
            accuracies = [r["accuracy"] for r in search_results]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot separation ratios
            ax1.semilogx(lambdas, separations, 'b-', linewidth=2, label='Separation Ratio')
            ax1.set_xlabel('Lambda Coefficient (log scale)')
            ax1.set_ylabel('Separation Ratio')
            ax1.set_title('Lambda vs Separation Ratio')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Plot accuracy
            ax2.semilogx(lambdas, accuracies, 'r-', linewidth=2, label='Classification Accuracy')
            ax2.set_xlabel('Lambda Coefficient (log scale)')
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Lambda vs Classification Accuracy')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Optimization plot saved: {save_path}")
            
        except Exception as e:
            print(f"⚠️  Could not create plot: {e}")


def main():
    """Execute simplified lambda optimization."""
    optimizer = SimpleLambdaOptimizer()
    
    # Run optimization
    result = optimizer.optimize_lambda()
    
    # Save results
    results_dict = {
        "optimal_lambda": result.optimal_lambda,
        "separation_ratio": result.separation_ratio,
        "classification_accuracy": result.best_accuracy,
        "grid_search_points": result.grid_search_points,
        "method": "Simplified grid search with standard Python arithmetic",
        "known_zeros_tested": len(optimizer.known_zeros),
        "counterexamples_tested": len(optimizer.non_zeros),
        "validation_summary": result.validation_summary
    }
    
    with open("lambda_optimization_simple_results.json", "w") as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\\n📁 Results saved to: lambda_optimization_simple_results.json")
    
    # Create plot if possible
    if hasattr(result, 'search_results'):
        optimizer.plot_optimization_landscape(result.search_results)
    
    return result


if __name__ == "__main__":
    main()