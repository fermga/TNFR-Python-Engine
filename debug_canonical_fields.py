#!/usr/bin/env python3
"""
Debug canonical fields data structure
"""

import json

def debug_canonical_fields():
    """Check the structure of canonical fields data."""
    results_file = "benchmarks/results/multi_topology_critical_exponent_20251111_233723.jsonl"
    
    results = []
    with open(results_file, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    
    # Look at the first topology's canonical data
    for result in results:
        if 'topology' in result:
            topology = result['topology']
            print(f"=== {topology.upper()} CANONICAL FIELDS ===")
            
            raw_data = result.get('raw_data', {})
            canonical_data = raw_data.get('canonical_fields_data', [])
            
            print(f"canonical_fields_data length: {len(canonical_data)}")
            
            if canonical_data:
                print("First canonical entry:")
                first_entry = canonical_data[0]
                print(f"Keys: {list(first_entry.keys())}")
                
                for field in ['phi_s', 'grad_phi', 'k_phi']:
                    if field in first_entry:
                        field_data = first_entry[field]
                        print(f"\n{field}:")
                        print(f"  Type: {type(field_data)}")
                        print(f"  Content: {field_data}")
                        
                        if isinstance(field_data, dict):
                            print(f"  Keys: {list(field_data.keys())}")
                            if 'values' in field_data:
                                values = field_data['values']
                                print(f"  values type: {type(values)}")
                                print(f"  values content: {values}")
                                if hasattr(values, '__len__') and not isinstance(values, (str, bytes)):
                                    print(f"  ❌ values is list/array with {len(values)} elements")
                                    if len(values) > 0:
                                        print(f"     First element type: {type(values[0])}")
                                else:
                                    print(f"  ✅ values is scalar")
            break

if __name__ == "__main__":
    debug_canonical_fields()