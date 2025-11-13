#!/usr/bin/env python3
"""
Debug script to reproduce the exact analysis error.
Reproduce the '<' not supported between instances of 'list' and 'float' error.
"""

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def debug_analysis_error():
    """Reproduce the analysis error using actual experiment data."""
    print("🔍 DEBUGGING ANALYSIS ERROR")
    print("=" * 50)
    
    # Load the actual experimental data
    results_file = "benchmarks/results/multi_topology_critical_exponent_20251111_233723.jsonl"
    
    results = []
    with open(results_file, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    
    print(f"Loaded {len(results)} results")
    
    # Examine the first result that failed
    for i, result in enumerate(results):
        if 'topology' in result:
            topology = result['topology']
            print(f"\n--- ANALYZING {topology.upper()} DATA ---")
            
            # Get the raw data
            raw_data = result.get('raw_data', {})
            xi_c_data = raw_data.get('xi_c_data', [])
            
            print(f"xi_c_data length: {len(xi_c_data)}")
            
            if xi_c_data:
                print("First few xi_c entries:")
                for j, entry in enumerate(xi_c_data[:3]):
                    print(f"  {j}: {entry}")
                    
                    # Check the 'values' field - this might be where the problem is
                    values = entry.get('values', [])
                    print(f"     values type: {type(values)}")
                    print(f"     values content: {values}")
                    if values:
                        print(f"     values[0] type: {type(values[0])}")
                        print(f"     values[0] repr: {repr(values[0])}")
                        
                        # Check if any value is a list
                        for k, val in enumerate(values):
                            if hasattr(val, '__len__') and not isinstance(val, (str, bytes)):
                                print(f"     ❌ Found list/array at values[{k}]: {val} (type: {type(val)})")
                            else:
                                print(f"     ✅ values[{k}] is scalar: {val} (type: {type(val)})")
                
                # Try to reproduce the exact analysis that was failing
                print(f"\n--- TESTING ANALYSIS OPERATIONS ---")
                try:
                    # This is what the experiment does:
                    xi_c_means = [data["mean"] for data in xi_c_data]
                    print(f"xi_c_means: {xi_c_means}")
                    print(f"xi_c_means types: {[type(x) for x in xi_c_means]}")
                    
                    # Test the operations that might be failing
                    print("Testing min/max operations...")
                    min_val = min(xi_c_means)
                    max_val = max(xi_c_means)
                    print(f"min: {min_val}, max: {max_val}")
                    print("✅ min/max operations succeeded")
                    
                except Exception as e:
                    print(f"❌ Error in analysis: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Only analyze the first one for now
                break

if __name__ == "__main__":
    debug_analysis_error()