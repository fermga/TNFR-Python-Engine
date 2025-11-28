"""Test canonical promotion of Phase Gradient |∇φ|.

Verifies that:
1. compute_phase_gradient returns expected values
2. Implementation matches validated definition
3. Function is marked as CANONICAL in docstring
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import networkx as nx
from tnfr.physics.fields import compute_phase_gradient
from tnfr.constants import THETA_PRIMARY
sys.path.insert(0, str(Path(__file__).parent.parent / "benchmarks"))
from benchmark_utils import create_tnfr_topology, initialize_tnfr_nodes


def test_canonical_phase_gradient():
    """Test canonical phase gradient implementation."""
    # Create simple graph
    G = create_tnfr_topology('ring', 4, seed=42)
    initialize_tnfr_nodes(G, seed=42)
    
    # Set known phases for verification
    G.nodes[0][THETA_PRIMARY] = 0.0
    G.nodes[1][THETA_PRIMARY] = 1.0  
    G.nodes[2][THETA_PRIMARY] = 2.0
    G.nodes[3][THETA_PRIMARY] = 1.5
    
    # Compute phase gradients
    grad = compute_phase_gradient(G)
    
    print("Phase Gradient Test Results:")
    print("="*40)
    
    # Node 0: neighbors are [1, 3] with phases [1.0, 1.5]
    # |∇φ|(0) = mean(|0-1|, |0-1.5|) = mean(1.0, 1.5) = 1.25
    expected_0 = (abs(0.0 - 1.0) + abs(0.0 - 1.5)) / 2
    print(f"Node 0: expected={expected_0:.3f}, actual={grad[0]:.3f}")
    assert abs(grad[0] - expected_0) < 1e-6
    
    # Node 1: neighbors are [0, 2] with phases [0.0, 2.0] 
    # |∇φ|(1) = mean(|1-0|, |1-2|) = mean(1.0, 1.0) = 1.0
    expected_1 = (abs(1.0 - 0.0) + abs(1.0 - 2.0)) / 2
    print(f"Node 1: expected={expected_1:.3f}, actual={grad[1]:.3f}")
    assert abs(grad[1] - expected_1) < 1e-6
    
    # Node 2: neighbors are [1, 3] with phases [1.0, 1.5]
    # |∇φ|(2) = mean(|2-1|, |2-1.5|) = mean(1.0, 0.5) = 0.75
    expected_2 = (abs(2.0 - 1.0) + abs(2.0 - 1.5)) / 2
    print(f"Node 2: expected={expected_2:.3f}, actual={grad[2]:.3f}")
    assert abs(grad[2] - expected_2) < 1e-6
    
    # Node 3: neighbors are [0, 2] with phases [0.0, 2.0]
    # |∇φ|(3) = mean(|1.5-0|, |1.5-2|) = mean(1.5, 0.5) = 1.0
    expected_3 = (abs(1.5 - 0.0) + abs(1.5 - 2.0)) / 2
    print(f"Node 3: expected={expected_3:.3f}, actual={grad[3]:.3f}")
    assert abs(grad[3] - expected_3) < 1e-6
    
    print("\n✅ All phase gradient calculations correct!")
    
    # Verify docstring mentions CANONICAL status
    docstring = compute_phase_gradient.__doc__
    assert "CANONICAL" in docstring, "Function should be marked as CANONICAL"
    assert "corr(Δ|∇φ|, Δmax_ΔNFR) = +0.6554" in docstring, "Should include validation results"
    
    print("✅ CANONICAL status properly documented!")
    
    return grad


if __name__ == "__main__":
    result = test_canonical_phase_gradient()
    print(f"\nPhase Gradient |∇φ| CANONICAL implementation validated! 🎉")
    print(f"Results: {result}")