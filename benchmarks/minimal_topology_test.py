#!/usr/bin/env python3
"""
Minimal Multi-Topology Test for ξ_C
Test basic functionality without complex telemetry
"""

import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from tnfr.physics.fields import estimate_coherence_length
    from benchmark_utils import create_tnfr_topology
    from tnfr.operators.definitions import Dissonance, Coherence
    from tnfr.dynamics.dnfr import default_compute_delta_nfr
    from tnfr.config import DNFR_PRIMARY
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root")
    sys.exit(1)

def minimal_test():
    """Minimal test to verify basic functionality"""
    
    print("MINIMAL MULTI-TOPOLOGY TEST")
    print("="*40)
    
    # Very simple test
    topology = "ws"
    n_nodes = 15
    seed = 12345
    
    try:
        # Create topology
        G = create_tnfr_topology(topology, n_nodes, seed)
        print(f"✅ Created {topology} topology with {len(G.nodes())} nodes")
        
        # Set some DNFR values
        np.random.seed(seed)
        for node in G.nodes():
            G.nodes[node][DNFR_PRIMARY] = np.random.normal(0, 0.3)
        
        # Add coherence values
        for node in G.nodes():
            dnfr = abs(G.nodes[node].get(DNFR_PRIMARY, 0.0))
            G.nodes[node]['coherence'] = 1.0 / (1.0 + dnfr)
        
        # Estimate coherence length
        xi_c = estimate_coherence_length(G, coherence_key="coherence")
        print(f"✅ Coherence length: {xi_c:.2f}")
        
        # Test other topologies
        for topo in ["scale_free", "grid"]:
            G2 = create_tnfr_topology(topo, n_nodes, seed+1)
            print(f"✅ Created {topo} topology with {len(G2.nodes())} nodes")
        
        print("\n🎉 All basic tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = minimal_test()
    sys.exit(0 if success else 1)