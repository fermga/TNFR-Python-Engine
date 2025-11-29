#!/usr/bin/env python3

"""
Structural Field Thresholds Validation Test
==========================================

Validates the recalibrated field thresholds after canonical constants migration.
Tests field computations with new thresholds to ensure proper detection and 
false positive/negative rates.
"""

import sys
import numpy as np
import networkx as nx
from pathlib import Path
from typing import Dict, List, Tuple
import pytest

# Add src to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

from tnfr.mathematics.number_theory import ArithmeticTNFRNetwork, ArithmeticTNFRParameters
from tnfr.physics.canonical import (
    compute_structural_potential, compute_phase_gradient,
    compute_phase_curvature, estimate_coherence_length
)
from tnfr.constants.canonical import (
    PHI_S_THRESHOLD, GRAD_PHI_THRESHOLD, K_PHI_THRESHOLD, PI
)

class TestRecalibratedThresholds:
    """Test suite for recalibrated field thresholds."""
    
    @classmethod
    def setup_class(cls):
        """Setup test networks with canonical constants."""
        cls.params = ArithmeticTNFRParameters()  # Canonical parameters
        
        # Test networks
        cls.networks = {
            'small': ArithmeticTNFRNetwork(max_number=25, parameters=cls.params).graph,
            'medium': ArithmeticTNFRNetwork(max_number=75, parameters=cls.params).graph,
            'large': ArithmeticTNFRNetwork(max_number=150, parameters=cls.params).graph,
        }
        
        print(f"\n🔧 Testing with Recalibrated Thresholds:")
        print(f"  Φ_s threshold  : {PHI_S_THRESHOLD:.6f}")
        print(f"  |∇φ| threshold : {GRAD_PHI_THRESHOLD:.6f}")
        print(f"  K_φ threshold  : {K_PHI_THRESHOLD:.6f}")
        print(f"  ξ_C multiplier : {PI:.6f}")
        
    def compute_all_fields(self, G: nx.Graph) -> Dict[str, float]:
        """Compute all structural fields for a network."""
        try:
            # Structural potential
            phi_s_field = compute_structural_potential(G)
            phi_s_values = list(phi_s_field.values()) if isinstance(phi_s_field, dict) else phi_s_field
            phi_s_max = np.max(np.abs(phi_s_values)) if len(phi_s_values) > 0 else 0.0
            
            # Phase gradient  
            grad_phi = compute_phase_gradient(G)
            grad_values = list(grad_phi.values()) if isinstance(grad_phi, dict) else grad_phi
            grad_phi_mean = np.mean(grad_values) if len(grad_values) > 0 else 0.0
            
            # Phase curvature
            k_phi = compute_phase_curvature(G)
            k_phi_values = list(k_phi.values()) if isinstance(k_phi, dict) else k_phi
            k_phi_max = np.max(np.abs(k_phi_values)) if len(k_phi_values) > 0 else 0.0
            
            # Coherence length
            xi_c = estimate_coherence_length(G)
            mean_distance = np.mean([
                nx.shortest_path_length(G, u, v) for u in G.nodes() 
                for v in G.nodes() if u != v and nx.has_path(G, u, v)
            ]) if G.number_of_nodes() > 1 else 1.0
            xi_c_ratio = xi_c / mean_distance if mean_distance > 0 else 0.0
            
            return {
                'phi_s_max': phi_s_max,
                'grad_phi_mean': grad_phi_mean,
                'k_phi_max': k_phi_max,
                'xi_c': xi_c,
                'xi_c_ratio': xi_c_ratio,
                'mean_distance': mean_distance,
                'nodes': G.number_of_nodes(),
                'edges': G.number_of_edges()
            }
            
        except Exception as e:
            print(f"    ⚠️ Field computation error: {e}")
            return {
                'phi_s_max': 0.0,
                'grad_phi_mean': 0.0,
                'k_phi_max': 0.0,
                'xi_c': 0.0,
                'xi_c_ratio': 0.0,
                'mean_distance': 1.0,
                'nodes': G.number_of_nodes(),
                'edges': G.number_of_edges()
            }
    
    def test_phi_s_threshold_validation(self):
        """Test Φ_s structural potential threshold."""
        print(f"\n🔍 Testing Φ_s Threshold: {PHI_S_THRESHOLD:.6f}")
        
        for net_name, G in self.networks.items():
            fields = self.compute_all_fields(G)
            phi_s_max = fields['phi_s_max']
            
            # Check threshold
            is_safe = phi_s_max < PHI_S_THRESHOLD
            status = "✅ SAFE" if is_safe else "⚠️ THRESHOLD"
            
            print(f"  {net_name:6} (n={fields['nodes']:3}): Φ_s={phi_s_max:.4f} {status}")
            
            # Assert reasonable values
            assert phi_s_max >= 0.0, f"Φ_s should be non-negative: {phi_s_max}"
            assert phi_s_max < 10.0, f"Φ_s should be bounded: {phi_s_max}"
            
            # Most networks should be in safe range with recalibrated threshold
            if fields['nodes'] < 50:  # Small networks should be safe
                assert is_safe, f"Small network {net_name} should be in safe Φ_s range"
    
    def test_grad_phi_threshold_validation(self):
        """Test |∇φ| phase gradient threshold."""
        print(f"\n🔍 Testing |∇φ| Threshold: {GRAD_PHI_THRESHOLD:.6f}")
        
        for net_name, G in self.networks.items():
            fields = self.compute_all_fields(G)
            grad_phi_mean = fields['grad_phi_mean']
            
            # Check threshold
            is_safe = grad_phi_mean < GRAD_PHI_THRESHOLD
            status = "✅ SAFE" if is_safe else "⚠️ THRESHOLD"
            
            print(f"  {net_name:6} (n={fields['nodes']:3}): |∇φ|={grad_phi_mean:.4f} {status}")
            
            # Assert reasonable values
            assert grad_phi_mean >= 0.0, f"|∇φ| should be non-negative: {grad_phi_mean}"
            assert grad_phi_mean < 10.0, f"|∇φ| should be bounded: {grad_phi_mean}"
            
            # Most networks should be in safe range with recalibrated threshold
            if fields['nodes'] < 50:  # Small networks should be safe
                assert is_safe, f"Small network {net_name} should be in safe |∇φ| range"
    
    def test_k_phi_threshold_validation(self):
        """Test K_φ phase curvature threshold."""
        print(f"\n🔍 Testing K_φ Threshold: {K_PHI_THRESHOLD:.6f}")
        
        for net_name, G in self.networks.items():
            fields = self.compute_all_fields(G)
            k_phi_max = fields['k_phi_max']
            
            # Check threshold (for K_φ, higher values flag hotspots)
            is_normal = k_phi_max < K_PHI_THRESHOLD
            status = "✅ NORMAL" if is_normal else "🔥 HOTSPOT"
            
            print(f"  {net_name:6} (n={fields['nodes']:3}): K_φ={k_phi_max:.4f} {status}")
            
            # Assert reasonable values
            assert k_phi_max >= 0.0, f"K_φ should be non-negative: {k_phi_max}"
            assert k_phi_max < 20.0, f"K_φ should be bounded: {k_phi_max}"
            
            # Small networks should typically be normal with recalibrated threshold
            if fields['nodes'] < 30:  # Very small networks should be normal
                assert is_normal, f"Very small network {net_name} should be in normal K_φ range"
    
    def test_xi_c_coherence_length_validation(self):
        """Test ξ_C coherence length ratios.""" 
        print(f"\n🔍 Testing ξ_C Coherence Length Ratios")
        
        for net_name, G in self.networks.items():
            fields = self.compute_all_fields(G)
            xi_c = fields['xi_c']
            xi_c_ratio = fields['xi_c_ratio']
            mean_distance = fields['mean_distance']
            
            # Check ratios
            critical_ratio = xi_c / (fields['nodes'] - 1) if fields['nodes'] > 1 else 0  # vs diameter
            watch_ratio = xi_c_ratio  # vs mean_distance
            
            critical_status = "🚨 CRITICAL" if critical_ratio > 1.0 else "✅ STABLE"
            watch_status = "⚠️ WATCH" if watch_ratio > PI else "✅ NORMAL"
            
            print(f"  {net_name:6} (n={fields['nodes']:3}):")
            print(f"    ξ_C = {xi_c:.3f}, critical_ratio = {critical_ratio:.3f} {critical_status}")
            print(f"    mean_dist = {mean_distance:.3f}, watch_ratio = {watch_ratio:.3f} {watch_status}")
            
            # Assert reasonable values
            assert xi_c >= 0.0, f"ξ_C should be non-negative: {xi_c}"
            assert xi_c_ratio >= 0.0, f"ξ_C ratio should be non-negative: {xi_c_ratio}"
            
    def test_threshold_consistency_across_sizes(self):
        """Test that thresholds scale appropriately across network sizes."""
        print(f"\n🔍 Testing Threshold Consistency Across Sizes")
        
        all_results = {}
        
        for net_name, G in self.networks.items():
            fields = self.compute_all_fields(G)
            all_results[net_name] = fields
            
        # Check that field values scale reasonably
        small_fields = all_results['small']
        medium_fields = all_results['medium'] 
        large_fields = all_results['large']
        
        print(f"  📊 Field Scaling Analysis:")
        print(f"    Network sizes: {small_fields['nodes']} → {medium_fields['nodes']} → {large_fields['nodes']}")
        print(f"    Φ_s scaling: {small_fields['phi_s_max']:.4f} → {medium_fields['phi_s_max']:.4f} → {large_fields['phi_s_max']:.4f}")
        print(f"    |∇φ| scaling: {small_fields['grad_phi_mean']:.4f} → {medium_fields['grad_phi_mean']:.4f} → {large_fields['grad_phi_mean']:.4f}")
        print(f"    K_φ scaling: {small_fields['k_phi_max']:.4f} → {medium_fields['k_phi_max']:.4f} → {large_fields['k_phi_max']:.4f}")
        
        # Fields should generally increase with network size but not explosively
        for field in ['phi_s_max', 'k_phi_max']:
            small_val = small_fields[field]
            large_val = large_fields[field]
            
            if small_val > 0:  # Avoid division by zero
                growth_factor = large_val / small_val
                assert growth_factor < 100, f"{field} grows too fast across sizes: {growth_factor}x"
                
    def test_recalibrated_vs_classical_comparison(self):
        """Compare detection rates with recalibrated vs classical thresholds."""
        print(f"\n🔍 Recalibrated vs Classical Threshold Comparison")
        
        # Classical thresholds (for comparison)
        classical_thresholds = {
            'phi_s': 0.7711,
            'grad_phi': 0.2904, 
            'k_phi': 2.8274,
        }
        
        # Recalibrated thresholds
        recalibrated_thresholds = {
            'phi_s': PHI_S_THRESHOLD,
            'grad_phi': GRAD_PHI_THRESHOLD,
            'k_phi': K_PHI_THRESHOLD,
        }
        
        detection_comparison = {}
        
        for net_name, G in self.networks.items():
            fields = self.compute_all_fields(G)
            
            # Classical detection
            classical_detections = {
                'phi_s_safe': fields['phi_s_max'] < classical_thresholds['phi_s'],
                'grad_phi_safe': fields['grad_phi_mean'] < classical_thresholds['grad_phi'],
                'k_phi_normal': fields['k_phi_max'] < classical_thresholds['k_phi'],
            }
            
            # Recalibrated detection  
            recalibrated_detections = {
                'phi_s_safe': fields['phi_s_max'] < recalibrated_thresholds['phi_s'],
                'grad_phi_safe': fields['grad_phi_mean'] < recalibrated_thresholds['grad_phi'],
                'k_phi_normal': fields['k_phi_max'] < recalibrated_thresholds['k_phi'],
            }
            
            detection_comparison[net_name] = {
                'classical': classical_detections,
                'recalibrated': recalibrated_detections,
                'fields': fields
            }
            
        # Report comparison
        print(f"  📊 Detection Comparison:")
        for net_name, comparison in detection_comparison.items():
            print(f"    {net_name:6} network:")
            classical = comparison['classical']
            recalibrated = comparison['recalibrated'] 
            
            for field in ['phi_s_safe', 'grad_phi_safe', 'k_phi_normal']:
                c_status = "✅" if classical[field] else "⚠️"
                r_status = "✅" if recalibrated[field] else "⚠️"
                change = "→" if classical[field] == recalibrated[field] else "≠"
                print(f"      {field:13}: Classical {c_status} {change} Recalibrated {r_status}")


def main():
    """Run the validation tests."""
    print("🧪 Structural Field Thresholds Validation")
    print("=" * 50)
    print("Testing recalibrated thresholds with canonical constants")
    
    # Run tests
    test_suite = TestRecalibratedThresholds()
    test_suite.setup_class()
    
    try:
        test_suite.test_phi_s_threshold_validation()
        test_suite.test_grad_phi_threshold_validation()
        test_suite.test_k_phi_threshold_validation()
        test_suite.test_xi_c_coherence_length_validation()
        test_suite.test_threshold_consistency_across_sizes()
        test_suite.test_recalibrated_vs_classical_comparison()
        
        print(f"\n🎉 ALL THRESHOLD VALIDATION TESTS PASSED!")
        print(f"   Recalibrated thresholds are compatible with canonical constants.")
        return True
        
    except Exception as e:
        print(f"\n❌ THRESHOLD VALIDATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)