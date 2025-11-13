"""Test of the use_extended_dynamics feature flag in integrators.py."""

import sys
import os
import networkx as nx

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from tnfr.dynamics.integrators import update_epi_via_nodal_equation
from tnfr.structural import create_nfr
from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_EPI, ALIAS_VF, ALIAS_DNFR


def test_classical_mode_default():
    """Test that classical mode is default and works."""
    print("🧪 Test 1: Classical mode (default)")
    
    # Create basic TNFR graph
    G, node = create_nfr("test_node", epi=0.5, vf=1.2, theta=0.3)
    
    # Set ΔNFR to have evolution
    G.nodes[node]['ΔNFR'] = 0.1
    
    # Initial state
    epi_initial = get_attr(G.nodes[node], ALIAS_EPI, 0.0)
    
    # Don't set flag (should use classical mode by default)
    dt = 0.05
    update_epi_via_nodal_equation(G, dt=dt)
    
    # Final state
    epi_final = get_attr(G.nodes[node], ALIAS_EPI, 0.0) 
    
    # Should have EPI evolution
    delta_epi = epi_final - epi_initial
    expected_delta = 1.2 * 0.1 * 0.05  # νf * ΔNFR * dt
    
    print(f"  EPI inicial: {epi_initial:.6f}")
    print(f"  EPI final: {epi_final:.6f}")
    print(f"  ΔEPI actual: {delta_epi:.6f}")
    print(f"  ΔEPI esperado: {expected_delta:.6f}")
    print(f"  Error: {abs(delta_epi - expected_delta):.6f}")
    
    # Verify there are no extended dynamics attributes
    has_extended_attrs = (
        'dtheta_dt' in G.nodes[node] or 
        'ddnfr_dt' in G.nodes[node]
    )
    
    success = (
        abs(delta_epi - expected_delta) < 0.001 and  # Correct evolution
        not has_extended_attrs  # No extended attributes
    )
    
    print(f"  Atributos extendidos: {'❌ Presentes' if has_extended_attrs else '✅ Ausentes'}")
    print(f"  ✅ PASS" if success else "  ❌ FAIL")
    
    return success


def test_extended_mode_enabled():
    """Test that extended mode works when enabled."""
    print("\n🧪 Test 2: Extended mode (enabled)")
    
    # Create network with multiple nodes to have neighbors
    G = nx.connected_watts_strogatz_graph(5, 3, 0.2, seed=42)
    
    # Initialize nodes with TNFR attributes
    for i, node in enumerate(G.nodes()):
        G.nodes[node][ALIAS_EPI] = 0.4 + i * 0.1
        G.nodes[node][ALIAS_VF] = 1.0 + i * 0.2
        G.nodes[node][ALIAS_DNFR] = 0.05 * (-1)**i  # Alternating
        G.nodes[node]['theta'] = i * 0.3
    
    # Enable extended dynamics
    G.graph['use_extended_dynamics'] = True
    
    # Initial state of first node
    node_0 = list(G.nodes())[0]
    epi_initial = G.nodes[node_0][ALIAS_EPI]
    theta_initial = G.nodes[node_0]['theta']
    dnfr_initial = G.nodes[node_0][ALIAS_DNFR]
    
    # Integrate
    dt = 0.02
    update_epi_via_nodal_equation(G, dt=dt)
    
    # Final state
    epi_final = G.nodes[node_0][ALIAS_EPI]
    theta_final = G.nodes[node_0]['theta'] 
    dnfr_final = G.nodes[node_0][ALIAS_DNFR]
    
    # Verify changes
    delta_epi = epi_final - epi_initial
    delta_theta = theta_final - theta_initial
    delta_dnfr = dnfr_final - dnfr_initial
    
    print(f"  ΔEPI: {delta_epi:.6f}")
    print(f"  Δθ: {delta_theta:.6f}")
    print(f"  ΔΔNFR: {delta_dnfr:.6f}")
    
    # Verify that there are extended dynamics attributes
    has_extended_attrs = (
        'dtheta_dt' in G.nodes[node_0] and
        'ddnfr_dt' in G.nodes[node_0]
    )
    
    # Verify evolution
    has_evolution = (
        abs(delta_epi) > 1e-6 or
        abs(delta_theta) > 1e-6 or  
        abs(delta_dnfr) > 1e-6
    )
    
    success = has_extended_attrs and has_evolution
    
    print(f"  Extended attributes: {'✅ Present' if has_extended_attrs else '❌ Absent'}")
    print(f"  Evolution detected: {'✅ Yes' if has_evolution else '❌ No'}")
    print(f"  ✅ PASS" if success else "  ❌ FAIL")
    
    return success


def test_mode_switching():
    """Test that modes can be switched dynamically."""
    print("\n🧪 Test 3: Dynamic mode switching")
    
    # Create graph
    G, node = create_nfr("switch_test", epi=0.6, vf=1.5, theta=0.5)
    G.nodes[node][ALIAS_DNFR] = 0.08
    
    # Step 1: Classical mode
    G.graph['use_extended_dynamics'] = False
    epi_1 = get_attr(G.nodes[node], ALIAS_EPI, 0.0)
    
    update_epi_via_nodal_equation(G, dt=0.01)
    epi_2 = get_attr(G.nodes[node], ALIAS_EPI, 0.0)
    
    classical_change = epi_2 - epi_1
    has_extended_1 = 'dtheta_dt' in G.nodes[node]
    
    # Step 2: Switch to extended mode
    G.graph['use_extended_dynamics'] = True
    epi_3 = get_attr(G.nodes[node], ALIAS_EPI, 0.0)
    
    update_epi_via_nodal_equation(G, dt=0.01)
    epi_4 = get_attr(G.nodes[node], ALIAS_EPI, 0.0)
    
    extended_change = epi_4 - epi_3
    has_extended_2 = 'dtheta_dt' in G.nodes[node]
    
    print(f"  Classical mode change: {classical_change:.6f}")
    print(f"  Extended mode change: {extended_change:.6f}")
    print(f"  Attributes in classical: {'❌ Present' if has_extended_1 else '✅ Absent'}")
    print(f"  Attributes in extended: {'✅ Present' if has_extended_2 else '❌ Absent'}")
    
    # Both modes must work
    success = (
        abs(classical_change) > 1e-6 and  # Classical evolves
        abs(extended_change) > 1e-6 and   # Extended evolves
        not has_extended_1 and            # Classical without extra attributes
        has_extended_2                    # Extended with extra attributes
    )
    
    print(f"  ✅ PASS" if success else "  ❌ FAIL")
    
    return success


def main():
    """Run all feature flag tests."""
    print("=" * 60)
    print("🎯 VALIDATION: Feature flag use_extended_dynamics")
    print("=" * 60)
    
    results = []
    
    # Test 1: Classical mode default
    results.append(test_classical_mode_default())
    
    # Test 2: Extended mode enabled
    results.append(test_extended_mode_enabled())
    
    # Test 3: Dynamic mode switching
    results.append(test_mode_switching())
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)
    
    test_names = [
        "Classical mode (default)",
        "Extended mode enabled",
        "Dynamic mode switching"
    ]
    
    for i, (name, passed) in enumerate(zip(test_names, results)):
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  Test {i+1} ({name}): {status}")
    
    total_pass = sum(results)
    print(f"\nResult: {total_pass}/{len(results)} tests passed")
    
    if total_pass == len(results):
        print("🎉 SUCCESS: Feature flag working correctly")
        return True
    else:
        print("⚠️ FAILURE: Some tests failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)