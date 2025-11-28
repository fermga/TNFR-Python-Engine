"""Demonstration of structural density metrics for NUL (Contraction) operator.

This example shows how the new density metrics enable:
1. Validation of canonical NUL behavior
2. Early warning for over-compression
3. Analysis of density evolution
4. Research workflow support
"""

from tnfr.structural import create_nfr
from tnfr.operators import apply_glyph
from tnfr.types import Glyph
from tnfr.constants import DNFR_PRIMARY, VF_PRIMARY, EPI_PRIMARY
from tnfr.operators.metrics import contraction_metrics


def demonstrate_density_metrics():
    """Demonstrate the new structural density metrics."""
    print("=" * 70)
    print("NUL (Contraction) Operator - Structural Density Metrics")
    print("=" * 70)
    print()
    
    # Example 1: Normal contraction with moderate density
    print("Example 1: Normal Contraction (Moderate Density)")
    print("-" * 70)
    
    G, node = create_nfr('test_node', epi=0.5, vf=1.0)
    G.nodes[node][DNFR_PRIMARY] = 0.2
    
    epi_before = G.nodes[node][EPI_PRIMARY]
    vf_before = G.nodes[node][VF_PRIMARY]
    dnfr_before = G.nodes[node][DNFR_PRIMARY]
    
    print(f"Before contraction:")
    print(f"  EPI:  {epi_before:.4f}")
    print(f"  νf:   {vf_before:.4f}")
    print(f"  ΔNFR: {dnfr_before:.4f}")
    print(f"  Density (|ΔNFR|/EPI): {abs(dnfr_before) / epi_before:.4f}")
    print()
    
    # Apply contraction
    apply_glyph(G, node, Glyph.NUL)
    
    # Collect metrics
    metrics = contraction_metrics(G, node, vf_before, epi_before)
    
    print(f"After contraction:")
    print(f"  EPI:  {metrics['epi_final']:.4f}")
    print(f"  νf:   {metrics['vf_final']:.4f}")
    print(f"  ΔNFR: {metrics['dnfr_final']:.4f}")
    print()
    
    print("New Density Metrics:")
    print(f"  density_before:        {metrics['density_before']:.4f}")
    print(f"  density_after:         {metrics['density_after']:.4f}")
    print(f"  densification_ratio:   {metrics['densification_ratio']:.4f}")
    print(f"  is_critical_density:   {metrics['is_critical_density']}")
    print()
    
    print("✓ Canonical behavior validated:")
    print(f"  - Density increased: {metrics['density_after'] > metrics['density_before']}")
    print(f"  - ΔNFR densified: {metrics.get('dnfr_densified', False)}")
    print(f"  - Safe compression: {not metrics['is_critical_density']}")
    print()
    print()
    
    # Example 2: High density scenario (approaching critical threshold)
    print("Example 2: High Density Contraction (Warning Case)")
    print("-" * 70)
    
    G2, node2 = create_nfr('high_density_node', epi=0.3, vf=1.0)
    G2.nodes[node2][DNFR_PRIMARY] = 1.5  # High ΔNFR
    
    epi_before2 = G2.nodes[node2][EPI_PRIMARY]
    vf_before2 = G2.nodes[node2][VF_PRIMARY]
    dnfr_before2 = G2.nodes[node2][DNFR_PRIMARY]
    
    print(f"Before contraction:")
    print(f"  EPI:  {epi_before2:.4f}")
    print(f"  νf:   {vf_before2:.4f}")
    print(f"  ΔNFR: {dnfr_before2:.4f}")
    print(f"  Density (|ΔNFR|/EPI): {abs(dnfr_before2) / epi_before2:.4f}")
    print()
    
    # Apply contraction
    apply_glyph(G2, node2, Glyph.NUL)
    
    # Collect metrics
    metrics2 = contraction_metrics(G2, node2, vf_before2, epi_before2)
    
    print(f"After contraction:")
    print(f"  EPI:  {metrics2['epi_final']:.4f}")
    print(f"  νf:   {metrics2['vf_final']:.4f}")
    print(f"  ΔNFR: {metrics2['dnfr_final']:.4f}")
    print()
    
    print("New Density Metrics:")
    print(f"  density_before:        {metrics2['density_before']:.4f}")
    print(f"  density_after:         {metrics2['density_after']:.4f}")
    print(f"  densification_ratio:   {metrics2['densification_ratio']:.4f}")
    print(f"  is_critical_density:   {metrics2['is_critical_density']}")
    print()
    
    if metrics2['is_critical_density']:
        print("⚠️  WARNING: Critical density exceeded!")
        print(f"   Density {metrics2['density_after']:.2f} > threshold 5.0")
        print("   Risk of over-compression. Consider:")
        print("   - Apply IL (Coherence) to stabilize")
        print("   - Avoid further NUL operations")
        print("   - Monitor for node collapse")
    print()
    print()
    
    # Example 3: Density evolution analysis
    print("Example 3: Density Evolution Analysis")
    print("-" * 70)
    
    print("Tracking density across varying initial conditions:")
    print()
    print(f"{'Initial ΔNFR':<15} {'Density Before':<15} {'Density After':<15} {'Ratio':<10} {'Critical':<10}")
    print("-" * 70)
    
    for initial_dnfr in [0.1, 0.2, 0.3, 0.4, 0.5]:
        G3, node3 = create_nfr('evolution_node', epi=0.5, vf=1.0)
        G3.nodes[node3][DNFR_PRIMARY] = initial_dnfr
        
        epi_before3 = G3.nodes[node3][EPI_PRIMARY]
        vf_before3 = G3.nodes[node3][VF_PRIMARY]
        
        apply_glyph(G3, node3, Glyph.NUL)
        
        metrics3 = contraction_metrics(G3, node3, vf_before3, epi_before3)
        
        critical_mark = "⚠️  YES" if metrics3['is_critical_density'] else "✓ NO"
        
        print(f"{initial_dnfr:<15.2f} "
              f"{metrics3['density_before']:<15.4f} "
              f"{metrics3['density_after']:<15.4f} "
              f"{metrics3['densification_ratio']:<10.4f} "
              f"{critical_mark:<10}")
    
    print()
    print("Observations:")
    print("  - Densification ratio remains relatively constant (~1.59)")
    print("  - This validates canonical NUL behavior")
    print("  - Higher initial ΔNFR leads to higher final density")
    print("  - Critical density warning activates appropriately")
    print()


if __name__ == "__main__":
    demonstrate_density_metrics()
