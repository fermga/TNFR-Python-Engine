"""Example demonstrating enhanced OZ (Dissonance) metrics.

This example shows how the new comprehensive dissonance metrics provide
detailed topological analysis, bifurcation scoring, and recovery guidance.
"""

from tnfr.structural import create_nfr
from tnfr.operators.definitions import Emission, Coherence, Dissonance


def demonstrate_enhanced_oz_metrics():
    """Demonstrate comprehensive OZ metrics collection."""
    
    print("="*70)
    print("Enhanced OZ (Dissonance) Metrics Demo")
    print("="*70)
    
    # Create a node with network structure
    G, node = create_nfr("central", epi=0.5, vf=1.2)
    
    # Add neighbors for network analysis
    print("\n1. Creating network structure...")
    for i in range(4):
        neighbor = f"neighbor_{i}"
        G.add_node(neighbor)
        G.add_edge(node, neighbor)
    print(f"   Created central node with {len(list(G.neighbors(node)))} neighbors")
    
    # Enable metrics collection
    G.graph['COLLECT_OPERATOR_METRICS'] = True
    
    # Build EPI history through operator sequence
    print("\n2. Building EPI history...")
    from tnfr.alias import get_attr
    from tnfr.constants.aliases import ALIAS_EPI
    
    Emission()(G, node)
    epi_val = get_attr(G.nodes[node], ALIAS_EPI, 0.0)
    print(f"   After Emission: EPI = {epi_val:.3f}")
    
    Emission()(G, node)
    epi_val = get_attr(G.nodes[node], ALIAS_EPI, 0.0)
    print(f"   After 2nd Emission: EPI = {epi_val:.3f}")
    
    Coherence()(G, node)
    epi_val = get_attr(G.nodes[node], ALIAS_EPI, 0.0)
    print(f"   After Coherence: EPI = {epi_val:.3f}")
    
    # Apply Dissonance to collect enhanced metrics
    print("\n3. Applying OZ (Dissonance) operator...")
    Dissonance()(G, node)
    print(f"   Dissonance applied successfully")
    
    # Retrieve enhanced metrics
    metrics = G.graph['operator_metrics'][-1]
    
    # Display comprehensive metrics
    print("\n" + "="*70)
    print("COMPREHENSIVE DISSONANCE METRICS")
    print("="*70)
    
    print("\n📊 QUANTITATIVE DYNAMICS:")
    print(f"   ΔNFR increase:    {metrics['dnfr_increase']:+.4f}")
    print(f"   ΔNFR final:       {metrics['dnfr_final']:+.4f}")
    print(f"   Phase shift:      {metrics['theta_shift']:.4f} rad")
    print(f"   Phase final:      {metrics['theta_final']:.4f} rad")
    print(f"   ∂²EPI/∂t²:        {metrics['d2epi']:+.4f}")
    
    print("\n🌀 BIFURCATION ANALYSIS:")
    print(f"   Bifurcation score:    {metrics['bifurcation_score']:.3f} / 1.00")
    print(f"   Bifurcation active:   {'YES ✅' if metrics['bifurcation_active'] else 'NO ❌'}")
    print(f"   Viable path count:    {metrics['viable_path_count']}")
    print(f"   Viable paths:")
    for path in metrics['viable_paths']:
        print(f"      • {path}")
    print(f"   Mutation readiness:   {'YES ✅' if metrics['mutation_readiness'] else 'NO ❌'}")
    
    print("\n🕸️  TOPOLOGICAL DISRUPTION:")
    print(f"   Asymmetry delta:      {metrics['topological_asymmetry_delta']:.4f}")
    print(f"   Symmetry disrupted:   {'YES ✅' if metrics['symmetry_disrupted'] else 'NO ❌'}")
    
    print("\n🌐 NETWORK IMPACT:")
    print(f"   Total neighbors:      {metrics['neighbor_count']}")
    print(f"   Impacted neighbors:   {metrics['impacted_neighbors']}")
    print(f"   Impact radius:        {metrics['network_impact_radius']:.1%}")
    
    print("\n💉 RESOLUTION GUIDANCE:")
    print(f"   IL applications:      {metrics['recovery_estimate_IL']}x needed")
    print(f"   Dissonance level:     {metrics['dissonance_level']:.4f}")
    print(f"   Critical dissonance:  {'YES ⚠️' if metrics['critical_dissonance'] else 'NO ✅'}")
    
    print("\n" + "="*70)
    print("✅ Enhanced OZ metrics provide comprehensive structural analysis")
    print("="*70 + "\n")
    
    return G, metrics


def compare_bifurcation_states():
    """Compare metrics for different bifurcation states."""
    
    print("\n" + "="*70)
    print("Bifurcation Score Comparison")
    print("="*70)
    
    # Low bifurcation potential
    print("\n1. LOW BIFURCATION POTENTIAL:")
    G1, node1 = create_nfr("stable", epi=0.3, vf=0.8)
    G1.graph['COLLECT_OPERATOR_METRICS'] = True
    G1.nodes[node1]["_epi_history"] = [0.28, 0.29, 0.30]  # Low acceleration
    
    Dissonance()(G1, node1)
    metrics1 = G1.graph['operator_metrics'][-1]
    
    print(f"   ∂²EPI/∂t²:            {metrics1['d2epi']:+.4f}")
    print(f"   Bifurcation score:    {metrics1['bifurcation_score']:.3f}")
    print(f"   Status:               {'ACTIVE 🌀' if metrics1['bifurcation_active'] else 'STABLE ✅'}")
    
    # High bifurcation potential
    print("\n2. HIGH BIFURCATION POTENTIAL:")
    G2, node2 = create_nfr("critical", epi=0.5, vf=1.5)
    G2.graph['COLLECT_OPERATOR_METRICS'] = True
    G2.nodes[node2]["_epi_history"] = [0.1, 0.3, 0.7]  # High acceleration
    
    Dissonance()(G2, node2)
    metrics2 = G2.graph['operator_metrics'][-1]
    
    print(f"   ∂²EPI/∂t²:            {metrics2['d2epi']:+.4f}")
    print(f"   Bifurcation score:    {metrics2['bifurcation_score']:.3f}")
    print(f"   Status:               {'ACTIVE 🌀' if metrics2['bifurcation_active'] else 'STABLE ✅'}")
    
    print("\n" + "="*70)
    print(f"Score increase: {metrics2['bifurcation_score'] - metrics1['bifurcation_score']:+.3f}")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_enhanced_oz_metrics()
    compare_bifurcation_states()
    
    print("\n✨ Enhanced OZ metrics enable:")
    print("   • Quantitative bifurcation assessment (not just boolean)")
    print("   • Topological disruption measurement")
    print("   • Viable path identification for resolution")
    print("   • Network impact analysis")
    print("   • Recovery planning with IL estimations\n")
