#!/usr/bin/env python3
"""
Debug comparison error
"""

import sys
import os
import numpy as np
import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from tnfr.dynamics.dnfr import default_compute_delta_nfr

print("Creating basic ws network...")
G = nx.watts_strogatz_graph(n=10, k=2, p=0.1, seed=42)
print(f"Network nodes: {G.number_of_nodes()}")

# Initialize the network
for node in G.nodes():
    G.nodes[node]['EPI'] = np.random.uniform(0, 1, (2,))
    G.nodes[node]['nu_f'] = np.random.uniform(0.1, 1.0)
    G.nodes[node]['theta'] = np.random.uniform(0, 2*np.pi)

print("Computing ΔNFR...")
try:
    default_compute_delta_nfr(G)
    print("✅ Success!")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()