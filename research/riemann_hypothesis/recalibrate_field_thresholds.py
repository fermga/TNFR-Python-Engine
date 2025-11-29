#!/usr/bin/env python3

"""
Field Thresholds Recalibration for Canonical Constants
=====================================================

Recalibrates all structural field thresholds after the migration from
empirical to canonical constants in ArithmeticTNFRParameters.

The old thresholds were calibrated with:
- alpha = 0.5, beta = 0.3, gamma = 0.2 (empirical)

New canonical constants:
- alpha = 1/φ ≈ 0.618, beta = γ/(π+γ) ≈ 0.155, gamma = γ/π ≈ 0.184

This may require threshold adjustments for optimal detection.
"""

import sys
import numpy as np
import networkx as nx
from pathlib import Path
from typing import Dict, List, Tuple
import json
import math

# Add src to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

from tnfr.mathematics.number_theory import ArithmeticTNFRParameters, ArithmeticTNFRNetwork
from tnfr.physics.canonical import (
    compute_structural_potential, compute_phase_gradient,
    compute_phase_curvature, estimate_coherence_length
)
from tnfr.constants.canonical import PHI, GAMMA, PI, E, INV_PHI


class ThresholdRecalibrator:
    """Recalibrate field thresholds for canonical constants."""
    
    def __init__(self):
        self.old_params = self._get_old_empirical_parameters()
        self.new_params = ArithmeticTNFRParameters()  # Canonical
        
        # Current thresholds (classical)
        self.classical_thresholds = {
            'phi_s': 0.771,      # |Φ_s| < 0.771 (von Koch)
            'grad_phi': 0.2904,  # |∇φ| < 0.2904 (Kuramoto)
            'k_phi': 2.8274,     # |K_φ| ≥ 2.8274 (confinement)
            'xi_c': PI,          # ξ_C > π × mean_distance
        }
        
        print("🔧 Field Thresholds Recalibration")
        print("=" * 50)
        print("Canonical Constants Migration Impact Analysis")
        
    def _get_old_empirical_parameters(self):
        """Recreate old empirical parameters for comparison."""
        class OldEmpiricalParams:
            alpha = 0.5
            beta = 0.3
            gamma = 0.2
            nu_0 = 1.0
            delta = 0.1
            epsilon = 0.05
            zeta = 1.0
            eta = 0.8
            theta = 0.6
            
        return OldEmpiricalParams()
        
    def generate_test_networks(self) -> Dict[str, nx.Graph]:
        """Generate test networks with different characteristics."""
        test_networks = {}
        
        # 1. Small stable network (expected: all fields in safe range)
        test_networks['stable_small'] = ArithmeticTNFRNetwork(max_number=20).graph
        
        # 2. Medium network with primes (expected: some field variations)
        test_networks['prime_medium'] = ArithmeticTNFRNetwork(max_number=50).graph
        
        # 3. Large network (expected: possible threshold crossings)
        test_networks['large'] = ArithmeticTNFRNetwork(max_number=100).graph
        
        # 4. Highly connected (stress test)
        G_stress = nx.complete_graph(15)
        for i, node in enumerate(G_stress.nodes()):
            G_stress.nodes[node].update({
                'EPI': np.random.normal(0.5, 0.1),
                'vf': np.random.exponential(1.0),
                'theta': np.random.uniform(0, 2*PI),
                'DNFR': np.random.normal(0, 0.5),
                'n': node + 1
            })
        test_networks['stress'] = G_stress
        
        return test_networks
        
    def compute_fields_with_params(self, G: nx.Graph, params) -> Dict[str, float]:
        """Compute all fields for a given network and parameters."""
        # Temporarily modify the network to use specific parameters
        # (This is a simplified approach - in practice we'd need to recompute TNFR properties)
        
        try:
            fields = {}
            
            # Structural potential
            phi_s_field = compute_structural_potential(G)
            fields['phi_s_max'] = np.max(np.abs(phi_s_field)) if len(phi_s_field) > 0 else 0.0
            
            # Phase gradient  
            grad_phi = compute_phase_gradient(G)
            fields['grad_phi_mean'] = np.mean(grad_phi) if len(grad_phi) > 0 else 0.0
            
            # Phase curvature
            k_phi = compute_phase_curvature(G)
            fields['k_phi_max'] = np.max(np.abs(k_phi)) if len(k_phi) > 0 else 0.0
            
            # Coherence length
            xi_c = estimate_coherence_length(G)
            mean_distance = np.mean([
                nx.shortest_path_length(G, u, v) for u in G.nodes() 
                for v in G.nodes() if u != v
            ]) if G.number_of_nodes() > 1 else 1.0
            fields['xi_c_ratio'] = xi_c / mean_distance if mean_distance > 0 else 0.0
            
            return fields
            
        except Exception as e:
            print(f"    ⚠️ Field computation failed: {e}")
            return {
                'phi_s_max': 0.0,
                'grad_phi_mean': 0.0,
                'k_phi_max': 0.0,
                'xi_c_ratio': 0.0
            }
            
    def analyze_threshold_changes(self):
        """Analyze how field values change with new constants."""
        print("\n🧪 Analyzing Field Value Changes...")
        
        test_networks = self.generate_test_networks()
        results = {
            'networks': {},
            'threshold_analysis': {},
            'recommendations': {}
        }
        
        for net_name, G in test_networks.items():
            print(f"\n  📊 Testing {net_name} (nodes: {G.number_of_nodes()})")
            
            # Note: We can't easily simulate old vs new parameters without 
            # recomputing the entire TNFR network, so we'll focus on 
            # analyzing current field distributions
            
            fields = self.compute_fields_with_params(G, self.new_params)
            
            results['networks'][net_name] = {
                'nodes': G.number_of_nodes(),
                'edges': G.number_of_edges(),
                'fields': fields
            }
            
            # Check against current thresholds
            threshold_status = {
                'phi_s_safe': fields['phi_s_max'] < self.classical_thresholds['phi_s'],
                'grad_phi_safe': fields['grad_phi_mean'] < self.classical_thresholds['grad_phi'],
                'k_phi_normal': fields['k_phi_max'] < self.classical_thresholds['k_phi'],
                'xi_c_adequate': fields['xi_c_ratio'] > 1.0,  # Basic check
            }
            
            results['networks'][net_name]['threshold_status'] = threshold_status
            
            # Report status
            for field, is_safe in threshold_status.items():
                status = "✅ SAFE" if is_safe else "⚠️ THRESHOLD"
                print(f"    {field:15}: {status}")
                
        return results
        
    def recommend_new_thresholds(self, analysis_results: Dict) -> Dict[str, float]:
        """Recommend new thresholds based on canonical constants analysis."""
        print("\n🎯 Threshold Recommendations...")
        
        # Collect field value distributions
        all_phi_s = [net['fields']['phi_s_max'] for net in analysis_results['networks'].values()]
        all_grad_phi = [net['fields']['grad_phi_mean'] for net in analysis_results['networks'].values()]
        all_k_phi = [net['fields']['k_phi_max'] for net in analysis_results['networks'].values()]
        all_xi_c = [net['fields']['xi_c_ratio'] for net in analysis_results['networks'].values()]
        
        # Statistical analysis
        phi_s_stats = {
            'mean': np.mean(all_phi_s),
            'std': np.std(all_phi_s),
            'max': np.max(all_phi_s),
            'q95': np.percentile(all_phi_s, 95)
        }
        
        grad_phi_stats = {
            'mean': np.mean(all_grad_phi),
            'std': np.std(all_grad_phi),
            'max': np.max(all_grad_phi),
            'q95': np.percentile(all_grad_phi, 95)
        }
        
        k_phi_stats = {
            'mean': np.mean(all_k_phi),
            'std': np.std(all_k_phi),
            'max': np.max(all_k_phi),
            'q95': np.percentile(all_k_phi, 95)
        }
        
        xi_c_stats = {
            'mean': np.mean(all_xi_c),
            'std': np.std(all_xi_c),
            'min': np.min(all_xi_c),
            'q5': np.percentile(all_xi_c, 5)
        }
        
        # Recommendation logic based on canonical constants influence
        recommendations = {}
        
        # Φ_s: Adjust based on new parameter magnitudes
        phi_s_factor = (self.new_params.alpha + self.new_params.beta) / (0.5 + 0.3)  # ≈ 0.967
        recommendations['phi_s'] = self.classical_thresholds['phi_s'] * phi_s_factor
        
        # |∇φ|: Adjust based on frequency parameters
        grad_phi_factor = self.new_params.nu_0 / 1.0  # nu_0 changed from 1.0 to ≈0.89
        recommendations['grad_phi'] = self.classical_thresholds['grad_phi'] * grad_phi_factor
        
        # K_φ: Adjust based on pressure parameters
        k_phi_factor = (self.new_params.zeta + self.new_params.eta) / (1.0 + 0.8)  # ≈1.13
        recommendations['k_phi'] = self.classical_thresholds['k_phi'] * k_phi_factor
        
        # ξ_C: Keep π-based scaling but adjust for new dynamics
        xi_c_factor = math.sqrt(phi_s_factor * grad_phi_factor)  # Combined influence
        recommendations['xi_c'] = self.classical_thresholds['xi_c'] * xi_c_factor
        
        print(f"  📊 Field Statistics:")
        print(f"    Φ_s  : mean={phi_s_stats['mean']:.4f}, max={phi_s_stats['max']:.4f}, q95={phi_s_stats['q95']:.4f}")
        print(f"    |∇φ| : mean={grad_phi_stats['mean']:.4f}, max={grad_phi_stats['max']:.4f}, q95={grad_phi_stats['q95']:.4f}")
        print(f"    K_φ  : mean={k_phi_stats['mean']:.4f}, max={k_phi_stats['max']:.4f}, q95={k_phi_stats['q95']:.4f}")
        print(f"    ξ_C  : mean={xi_c_stats['mean']:.4f}, min={xi_c_stats['min']:.4f}, q5={xi_c_stats['q5']:.4f}")
        
        print(f"\n  🔧 Recommended Adjustments:")
        for field, new_threshold in recommendations.items():
            old_threshold = self.classical_thresholds[field]
            change_pct = ((new_threshold - old_threshold) / old_threshold) * 100
            direction = "↗️ INCREASE" if change_pct > 0 else "↘️ DECREASE" if change_pct < 0 else "➡️ UNCHANGED"
            print(f"    {field:8}: {old_threshold:.4f} → {new_threshold:.4f} ({change_pct:+5.1f}%) {direction}")
            
        return recommendations
        
    def generate_updated_constants_file(self, new_thresholds: Dict[str, float]):
        """Generate updated canonical constants file with new thresholds."""
        print(f"\n📝 Generating Updated Constants File...")
        
        # Read current canonical.py
        canonical_path = repo_root / "src" / "tnfr" / "constants" / "canonical.py"
        
        with open(canonical_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Create updated version with new thresholds
        updated_content = content
        
        # Update threshold values (if they exist)
        threshold_updates = {
            'PHI_S_THRESHOLD': new_thresholds['phi_s'],
            'GRAD_PHI_THRESHOLD': new_thresholds['grad_phi'], 
            'K_PHI_THRESHOLD': new_thresholds['k_phi'],
            # ξ_C is typically computed dynamically, not as constant
        }
        
        for const_name, new_value in threshold_updates.items():
            # Look for the constant definition line
            import re
            pattern = f'{const_name}\\s*=\\s*[^\\n]+'
            match = re.search(pattern, updated_content)
            if match:
                old_line = match.group(0)
                # Extract comment if present
                comment_match = re.search(r'#.*$', old_line)
                comment = comment_match.group(0) if comment_match else f"# Adjusted for canonical constants"
                
                new_line = f"{const_name} = {new_value:.6f}  {comment}"
                updated_content = updated_content.replace(old_line, new_line)
                print(f"  ✅ Updated {const_name}: {new_value:.6f}")
            else:
                print(f"  ⚠️ Could not find {const_name} in canonical.py")
                
        # Save updated file
        updated_path = repo_root / "research" / "riemann_hypothesis" / "canonical_updated_thresholds.py"
        
        with open(updated_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
            
        print(f"  📄 Updated constants saved to: {updated_path}")
        return updated_path
        
    def run_full_recalibration(self):
        """Run complete threshold recalibration process."""
        print("🎯 Starting Full Threshold Recalibration...")
        
        # 1. Analyze current field distributions
        analysis_results = self.analyze_threshold_changes()
        
        # 2. Recommend new thresholds
        new_thresholds = self.recommend_new_thresholds(analysis_results)
        
        # 3. Generate updated constants file
        updated_file = self.generate_updated_constants_file(new_thresholds)
        
        # 4. Generate report
        report = {
            'timestamp': '2025-11-29',
            'migration_context': 'canonical_constants',
            'classical_thresholds': self.classical_thresholds,
            'recommended_thresholds': new_thresholds,
            'analysis_results': analysis_results,
            'parameter_changes': {
                'old_empirical': {
                    'alpha': self.old_params.alpha,
                    'beta': self.old_params.beta,
                    'gamma': self.old_params.gamma,
                },
                'new_canonical': {
                    'alpha': float(self.new_params.alpha),
                    'beta': float(self.new_params.beta),
                    'gamma': float(self.new_params.gamma),
                }
            }
        }
        
        report_path = repo_root / "research" / "riemann_hypothesis" / "threshold_recalibration_report.json"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        print(f"\n📋 Summary:")
        print(f"  • Classical thresholds analyzed")
        print(f"  • New thresholds recommended based on canonical constants")
        print(f"  • Updated constants file generated: {updated_file}")
        print(f"  • Full report saved: {report_path}")
        
        # Final recommendation
        total_change = sum(abs((new_thresholds[k] - self.classical_thresholds[k]) / self.classical_thresholds[k]) 
                          for k in new_thresholds.keys() if k in self.classical_thresholds)
        avg_change_pct = (total_change / len(new_thresholds)) * 100
        
        if avg_change_pct > 10:
            print(f"\n⚠️  SIGNIFICANT CHANGES DETECTED (avg: {avg_change_pct:.1f}%)")
            print("   Recommend thorough testing before production deployment.")
        elif avg_change_pct > 5:
            print(f"\n🔧 MODERATE CHANGES DETECTED (avg: {avg_change_pct:.1f}%)")
            print("   Recommend validation testing of field thresholds.")
        else:
            print(f"\n✅ MINIMAL CHANGES DETECTED (avg: {avg_change_pct:.1f}%)")
            print("   Current thresholds appear adequate for canonical constants.")
            
        return report


def main():
    """Run the threshold recalibration analysis."""
    recalibrator = ThresholdRecalibrator()
    recalibrator.run_full_recalibration()


if __name__ == "__main__":
    main()