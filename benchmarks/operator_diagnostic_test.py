#!/usr/bin/env python3
"""
TNFR Operator Diagnostic Test

Investigate why operators are not causing observable changes to K_φ or network structure.
"""

import sys
from pathlib import Path

import numpy as np
import networkx as nx

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.tnfr.physics.fields import compute_phase_curvature
from benchmarks.benchmark_utils import create_tnfr_topology, initialize_tnfr_nodes
from src.tnfr.operators.definitions import (
    Dissonance, Emission, Coherence, Mutation, Silence
)
from src.tnfr.config import EPI_PRIMARY, VF_PRIMARY, THETA_PRIMARY, DNFR_PRIMARY


def diagnostic_test():
    """Diagnose operator effects on network state."""
    print("🔍 TNFR Operator Diagnostic Test")
    print("=" * 40)
    
    # Create test network
    G = create_tnfr_topology('ring', 10, seed=42)
    initialize_tnfr_nodes(G, seed=42)
    
    print("\n📊 Initial Network State:")
    print_network_state(G)
    
    # Test each operator individually
    operators_to_test = [
        ('Emission', Emission()),
        ('Dissonance', Dissonance()),
        ('Mutation', Mutation()),
        ('Coherence', Coherence()),
        ('Silence', Silence())
    ]
    
    for op_name, operator in operators_to_test:
        print(f"\n⚡ Testing {op_name} Operator:")
        
        # Save state before
        state_before = capture_network_state(G)
        k_phi_before = compute_phase_curvature(G)
        
        try:
            # Apply operator to first node
            target_node = list(G.nodes())[0]
            print(f"   Applying {op_name} to node {target_node}...")
            
            operator(G, target_node)
            
            # Check state after
            state_after = capture_network_state(G)
            k_phi_after = compute_phase_curvature(G)
            
            # Compare states
            changes = compare_states(state_before, state_after)
            k_phi_changes = {
                node: k_phi_after[node] - k_phi_before[node]
                for node in G.nodes()
            }
            
            print(f"   Node attribute changes: {sum(changes.values())}")
            print(f"   K_φ changes: {sum(abs(v) for v in k_phi_changes.values()):.6f}")
            
            if sum(changes.values()) > 0:
                print(f"   ✅ {op_name} caused changes!")
                for node, change_count in changes.items():
                    if change_count > 0:
                        print(f"      Node {node}: {change_count} attributes changed")
            else:
                print(f"   ❌ {op_name} caused NO changes")
                
        except Exception as e:
            print(f"   💥 {op_name} failed: {e}")
    
    print("\n" + "=" * 40)
    print("📊 Final Network State:")
    print_network_state(G)
    
    # Test manual modification
    print("\n🛠️ Manual Modification Test:")
    node_0 = list(G.nodes())[0]
    
    print(f"Before manual EPI change:")
    print(f"   Node {node_0} EPI: {G.nodes[node_0][EPI_PRIMARY]}")
    
    # Direct EPI modification
    G.nodes[node_0][EPI_PRIMARY] = 0.99
    
    print(f"After manual EPI change:")
    print(f"   Node {node_0} EPI: {G.nodes[node_0][EPI_PRIMARY]}")
    
    k_phi_manual = compute_phase_curvature(G)
    print(f"   K_φ after manual change: {k_phi_manual[node_0]:.6f}")


def print_network_state(G):
    """Print current network state summary."""
    nodes = list(G.nodes())[:3]  # First 3 nodes
    
    print(f"   Nodes: {len(G.nodes())}, Edges: {len(G.edges())}")
    for node in nodes:
        attrs = G.nodes[node]
        epi = attrs.get(EPI_PRIMARY, 'N/A')
        vf = attrs.get(VF_PRIMARY, 'N/A')
        theta = attrs.get(THETA_PRIMARY, 'N/A')
        dnfr = attrs.get(DNFR_PRIMARY, 'N/A')
        
        print(f"   Node {node}: EPI={epi:.3f} νf={vf:.3f} θ={theta:.3f} ΔNFR={dnfr:.3f}")
    
    k_phi = compute_phase_curvature(G)
    k_phi_sample = {node: k_phi[node] for node in nodes}
    print(f"   K_φ sample: {k_phi_sample}")


def capture_network_state(G):
    """Capture complete network state for comparison."""
    state = {}
    for node in G.nodes():
        state[node] = dict(G.nodes[node])  # Copy attributes
    return state


def compare_states(state_before, state_after):
    """Compare two network states and count changes."""
    changes = {}
    
    for node in state_before:
        changes[node] = 0
        attrs_before = state_before[node]
        attrs_after = state_after.get(node, {})
        
        for attr, value_before in attrs_before.items():
            value_after = attrs_after.get(attr, None)
            
            # Check for numerical changes (with tolerance)
            if isinstance(value_before, (int, float)) and isinstance(value_after, (int, float)):
                if abs(value_before - value_after) > 1e-10:
                    changes[node] += 1
            else:
                if value_before != value_after:
                    changes[node] += 1
    
    return changes


if __name__ == "__main__":
    diagnostic_test()