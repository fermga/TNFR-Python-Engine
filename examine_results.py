#!/usr/bin/env python3
"""
Examine the multi-topology ξ_C experimental results
"""
import json

# Load the JSONL file 
jsonl_path = "benchmarks/results/multi_topology_critical_exponent_20251112_000541.jsonl"

with open(jsonl_path, 'r') as f:
    lines = f.readlines()
    
print(f"Total lines in JSONL: {len(lines)}")

# Examine each topology
for i, line in enumerate(lines):
    data = json.loads(line)
    
    if "topology" in data:
        topo = data["topology"]
        print(f"\n--- Topology {i+1}: {topo} ---")
        
        # Check xi_c data 
        xi_c_data = data["raw_data"]["xi_c_data"]
        
        # Sample a few intensities
        for j in [0, 5, 10]:  # Check intensities at positions 0, 5, 10
            if j < len(xi_c_data):
                intensity = xi_c_data[j]["intensity"]
                values = xi_c_data[j]["values"][:5]  # First 5 measurements
                mean_val = xi_c_data[j]["mean"]
                print(f"  Intensity {intensity}: mean={mean_val:.3f}, sample={values}")
                
        # Check if analysis succeeded
        analysis = data.get("analysis", {})
        print(f"  Analysis success: {analysis.get('success', 'Unknown')}")
        if not analysis.get("success"):
            print(f"  Error: {analysis.get('error', 'Unknown error')}")
    else:
        print(f"Line {i+1}: Summary data")